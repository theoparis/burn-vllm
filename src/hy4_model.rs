//! Hy4-flash CPU forward (P2), transcribed from llama.cpp `hy-v4-flash.cpp` @ 15fcfc735.
//!
//! Deliberate approximations for the first gate:
//! - Engram/PLE deltas are skipped: their lookup table lives on shard 2 as F8_E4M3.
//! - The indexer's top-2048 selection is replaced by all causal MLA cells (valid while the
//!   prompt is shorter than ~2044 cells).
//! - M-RoPE text inputs use one position plane, i.e. plain RoPE on the tail 64 dims.

use crate::gguf::{Gguf, TensorInfo};
use crate::hy4::Config;
use crate::quant::{GgmlType, dequant};
use anyhow::{Result, ensure};
use rayon::prelude::*;

pub struct State {
    /// MLA K rows [576] per layer, oldest first.
    pub kv: Vec<Vec<Vec<f32>>>,
    pub pos: usize,
    pub prev_token: u32,
    /// Token history (oldest first) for the Engram n-gram hash.
    pub tokens: Vec<u32>,
    /// Indexer LID rows [iv(128), ig(128)] per sparse layer, cell order.
    pub lid: Vec<Vec<Vec<f32>>>,
}

impl State {
    pub fn new(cfg: &Config) -> Self {
        Self {
            kv: vec![Vec::new(); cfg.n_layers - cfg.nextn_layers],
            pos: 0,
            prev_token: 0,
            tokens: Vec::new(),
            lid: vec![Vec::new(); cfg.n_layers - cfg.nextn_layers],
        }
    }
}

pub struct Model<'a> {
    g: &'a Gguf,
    pub cfg: Config,
    shard2: Option<std::path::PathBuf>,
}

fn dump(name: &str, data: &[f32]) {
    let Ok(dir) = std::env::var("HY4_DUMP") else {
        return;
    };
    let path = format!("{dir}/rust_s0_{name}.bin");
    use std::io::Write;
    if let Ok(mut f) = std::fs::File::create(path) {
        for &v in data {
            writeln!(f, "{v:.7}").ok();
        }
    }
}

fn f32_vec(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let (t, raw) = tensor(g, name)?;
    ensure!(
        t.ty == GgmlType::F32,
        "{name}: expected F32, got {:?}",
        t.ty
    );
    Ok(bytemuck::cast_slice::<u8, f32>(raw).to_vec())
}

fn tensor<'a>(g: &'a Gguf, name: &str) -> Result<(&'a TensorInfo, &'a [u8])> {
    let t = g.tensor(name)?;
    Ok((t, g.bytes(t)))
}

/// Exact dequantized matvec: `out[r] = sum_j dequant(w[r,j]) * x[j]`.
fn matvec(t: &TensorInfo, raw: &[u8], x: &[f32]) -> Vec<f32> {
    ensure_dims(t, x.len());
    let cols = t.dims[0] as usize;
    let rows = t.dims[1] as usize;
    let rb = t.ty.row_bytes(cols);
    (0..rows)
        .into_par_iter()
        .map_init(
            || vec![0.0f32; cols],
            |scratch, r| {
                dequant(t.ty, &raw[r * rb..(r + 1) * rb], scratch);
                scratch.iter().zip(x).map(|(a, b)| a * b).sum()
            },
        )
        .collect()
}

/// Expert slice of a 3-D [cols, rows, experts] weight.
fn expert_matvec(t: &TensorInfo, raw: &[u8], e: usize, x: &[f32]) -> Vec<f32> {
    ensure_dims(t, x.len());
    let cols = t.dims[0] as usize;
    let rows = t.dims[1] as usize;
    let rb = t.ty.row_bytes(cols);
    let per = rows * rb;
    let base = &raw[e * per..(e + 1) * per];
    (0..rows)
        .into_par_iter()
        .map_init(
            || vec![0.0f32; cols],
            |scratch, r| {
                dequant(t.ty, &base[r * rb..(r + 1) * rb], scratch);
                scratch.iter().zip(x).map(|(a, b)| a * b).sum()
            },
        )
        .collect()
}

fn ensure_dims(t: &TensorInfo, cols: usize) {
    assert_eq!(t.dims[0] as usize, cols, "{}: cols mismatch", t.name);
}

fn mv(g: &Gguf, name: &str, x: &[f32]) -> Result<Vec<f32>> {
    let (t, raw) = tensor(g, name)?;
    Ok(matvec(t, raw, x))
}

fn emv(g: &Gguf, name: &str, e: usize, x: &[f32]) -> Result<Vec<f32>> {
    let (t, raw) = tensor(g, name)?;
    Ok(expert_matvec(t, raw, e, x))
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum: f32 = x.iter().map(|v| v * v).sum();
    let scale = 1.0 / (sum / x.len() as f32 + eps).sqrt();
    x.iter().zip(w).map(|(&v, &s)| v * scale * s).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn rope_tail(v: &mut [f32], dim: usize, pos: usize, base: f64) {
    let n = v.len();
    let start = n - dim;
    let mut neox = vec![0.0f32; dim];
    for i in 0..dim / 2 {
        neox[i] = v[start + 2 * i];
        neox[i + dim / 2] = v[start + 2 * i + 1];
    }
    for i in 0..dim / 2 {
        let theta = pos as f64 / base.powf(2.0 * i as f64 / dim as f64);
        let (c, s) = (theta.cos() as f32, theta.sin() as f32);
        let a = neox[i];
        let b = neox[i + dim / 2];
        neox[i] = a * c - b * s;
        neox[i + dim / 2] = a * s + b * c;
    }
    for i in 0..dim / 2 {
        v[start + 2 * i] = neox[i];
        v[start + 2 * i + 1] = neox[i + dim / 2];
    }
}

/// iHC gate affine from llama.cpp `build_hc_pre` (hy-v4-flash.cpp:558-575).
/// The pre half gets `hc_eps` after the sigmoid; the post half is scaled by
/// `hc_magnitude` first and then gets the same additive epsilon.
fn hc_gates(
    mixes: &[f32],
    base: &[f32],
    scale: &[f32],
    hc: usize,
    eps: f32,
    magnitude: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut pre = vec![0.0; hc];
    let mut post = vec![0.0; hc];
    for i in 0..hc {
        pre[i] = sigmoid(mixes[i] * scale[0] + base[i]) + eps;
        post[i] = magnitude * sigmoid(mixes[hc + i] * scale[1] + base[hc + i]) + eps;
    }
    (pre, post)
}

fn hc_pre(g: &Gguf, b: &str, x: &[f32], cfg: &Config) -> Result<(Vec<f32>, Vec<f32>)> {
    let hc = cfg.hc_count;
    let mut flat = Vec::with_capacity(x.len() * hc);
    for s in 0..hc {
        flat.extend_from_slice(&x[s * cfg.hidden..(s + 1) * cfg.hidden]);
    }
    let normed = rms_norm(&flat, &vec![1.0; flat.len()], cfg.eps);
    let mixes = mv(g, &format!("{b}hc_attn_fn.weight"), &normed)?;
    let base = f32_vec(g, &format!("{b}hc_attn_base.weight"))?;
    let scale = f32_vec(g, &format!("{b}hc_attn_scale.weight"))?;
    Ok(hc_gates(
        &mixes,
        &base,
        &scale,
        hc,
        cfg.hc_eps,
        cfg.hc_magnitude,
    ))
}

impl<'a> Model<'a> {
    pub fn new(g: &'a Gguf, cfg: Config, shard2: Option<std::path::PathBuf>) -> Result<Self> {
        Ok(Self { g, cfg, shard2 })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn forward(&self, st: &mut State, token: u32) -> Result<Vec<f32>> {
        let g = self.g;
        let cfg = &self.cfg;
        let pos = st.pos;
        let (te, traw) = tensor(g, "token_embd.weight")?;
        if std::env::var_os("HY4_DEBUG_OFFSETS").is_some() {
            eprintln!(
                "token_embd offset={} ty={:?} rb={}",
                te.offset,
                te.ty,
                te.ty.row_bytes((te.dims[0] as usize))
            );
        }
        // token embedding is a row lookup, not a projection.
        let cols = te.dims[0] as usize;
        let rb = te.ty.row_bytes(cols);
        let mut emb = vec![0.0f32; cols];
        dequant(
            te.ty,
            &traw[token as usize * rb..(token as usize + 1) * rb],
            &mut emb,
        );
        if std::env::var_os("HY4_DEBUG_OFFSETS").is_some() {
            eprintln!("EMB token={token} first={:?}", &emb[..8]);
        }
        let mut streams = Vec::with_capacity(cfg.hidden * cfg.hc_count);
        for _ in 0..cfg.hc_count {
            streams.extend_from_slice(&emb);
        }
        dump("hc_init", &streams);

        let engram = self
            .shard2
            .as_ref()
            .map(|p| crate::hy4_engram::Engram::new(g, p, cfg))
            .transpose()?;

        for il in 0..cfg.n_layers - cfg.nextn_layers {
            let b = format!("blk.{il}.");
            if let Some(ref en) = engram {
                if en.has_layer(il) {
                    let n_prev = cfg.ple_ngram.saturating_sub(1);
                    let start = st.tokens.len().saturating_sub(n_prev);
                    let delta = en.delta(il, token, &st.tokens[start..], &streams)?;
                    for (s, &d) in streams.iter_mut().zip(&delta) {
                        *s += d;
                    }
                }
            }
            let (pre, post) = hc_pre(g, &b, &streams, cfg)?;
            let mut cur = vec![0.0f32; cfg.hidden];
            for s in 0..cfg.hc_count {
                for (c, &v) in cur
                    .iter_mut()
                    .zip(&streams[s * cfg.hidden..(s + 1) * cfg.hidden])
                {
                    *c += v * pre[s];
                }
            }
            if il == 0 {
                dump("hc_attn_pre", &pre);
                dump("hc_attn_post_gate", &post);
            }
            let an = f32_vec(g, &format!("{b}attn_norm.weight"))?;
            cur = rms_norm(&cur, &an, cfg.eps);
            if il == 0 {
                dump("attn_norm", &cur);
            }

            // MLA q path.
            let qa = mv(g, &format!("{b}attn_q_a.weight"), &cur)?;
            let qan = f32_vec(g, &format!("{b}attn_q_a_norm.weight"))?;
            let qr = rms_norm(&qa, &qan, cfg.eps);
            if il == 0 {
                dump("q_resid", &qr);
            }
            let heads = cfg.head_counts[il];
            let mut qb = mv(g, &format!("{b}attn_q_b.weight"), &qr)?;
            let qdim = cfg.key_len_mla; // 256 per head
            let rope_base = if cfg.swa_pattern[il] {
                cfg.freq_base_swa
            } else {
                cfg.freq_base
            };
            for h in 0..heads {
                rope_tail(
                    &mut qb[h * qdim..(h + 1) * qdim],
                    cfg.qk_rope_dim,
                    pos,
                    rope_base,
                );
            }

            // MLA kv path: compressed [512] + rope tail [64].
            let kva = mv(g, &format!("{b}attn_kv_a_mqa.weight"), &cur)?;
            let kvn = f32_vec(g, &format!("{b}attn_kv_a_norm.weight"))?;
            let mut kv = vec![0.0f32; cfg.kv_lora_rank];
            kv.copy_from_slice(&kva[..cfg.kv_lora_rank]);
            kv = rms_norm(&kv, &kvn, cfg.eps);
            let mut kpe = vec![0.0f32; cfg.qk_rope_dim];
            kpe.copy_from_slice(&kva[cfg.kv_lora_rank..]);
            rope_tail(&mut kpe, cfg.qk_rope_dim, pos, rope_base);
            if il == 0 {
                dump("k_pe", &kpe);
                dump("kv_cmpr", &kv);
            }
            let mut krow = Vec::with_capacity(cfg.kv_lora_rank + cfg.qk_rope_dim);
            krow.extend_from_slice(&kv);
            krow.extend_from_slice(&kpe);
            st.kv[il].push(krow);

            // Absorb q_nope with wk_b: [192,512,heads] -> [512,heads].
            let (kt, kraw) = tensor(g, &format!("{b}attn_k_b.weight"))?;
            let qnope_dim = cfg.key_len_mla - cfg.qk_rope_dim;
            let mut qnope = vec![0.0f32; qnope_dim * heads];
            for h in 0..heads {
                qnope[h * qnope_dim..(h + 1) * qnope_dim]
                    .copy_from_slice(&qb[h * qdim..h * qdim + qnope_dim]);
            }
            if std::env::var_os("HY4_DEBUG_OFFSETS").is_some() && il == 0 {
                eprintln!(
                    "q_b offset={} wk_b offset={} wv_b offset={}",
                    tensor(g, &format!("{b}attn_q_b.weight"))
                        .map(|x| x.0.offset)
                        .unwrap_or(0),
                    kt.offset,
                    tensor(g, &format!("{b}attn_v_b.weight"))
                        .map(|x| x.0.offset)
                        .unwrap_or(0)
                );
            }
            let kc = kt.dims[0] as usize; // 192
            let kr = kt.dims[1] as usize; // 512
            let kper = kraw.len() / heads;
            let mut qabsorbed = vec![0.0f32; kr * heads];
            for h in 0..heads {
                let base = &kraw[h * kper..];
                for r in 0..kr {
                    let mut s = 0.0f32;
                    for c in 0..kc {
                        let w = dequant_one(kt.ty, base, r * kc + c);
                        s += w * qnope[h * qnope_dim + c];
                    }
                    qabsorbed[h * kr + r] = s;
                }
            }

            if il == 0 {
                dump("qabsorbed", &qabsorbed);
            }
            // Attention over all cached cells (indexer stub: full causal set).
            let sinks = f32_vec(g, &format!("{b}attn_sinks.weight"))?;
            if il == 0 {
                dump("sinks", &sinks);
                dump("q_pe", &qb);
            }
            // Optional real pooled indexer; short prompts can use the all-causal stub.
            let selected_keys: Option<Vec<Vec<f32>>> = if std::env::var_os("HY4_USE_INDEXER")
                .is_some()
                && cfg.indexer_types.get(il).copied().unwrap_or(false)
            {
                let idx = crate::hy4_indexer::Indexer::new(g, &b, cfg)?;
                let row = idx.lid_row(&cur)?;
                st.lid[il].push(row);
                let sel = idx.select(&cur, &qr, pos, &st.lid[il], rope_base)?;
                let n_cells = st.kv[il].len();
                let mut seen = std::collections::HashSet::new();
                Some(
                    sel.into_iter()
                        .filter(|&i| i < n_cells && seen.insert(i))
                        .map(|i| st.kv[il][i].clone())
                        .collect(),
                )
            } else {
                None
            };
            let keys: &[Vec<f32>] = selected_keys.as_deref().unwrap_or(&st.kv[il]);
            let scale = 1.0 / (cfg.key_len_mla as f32).sqrt();
            let mut attn = vec![0.0f32; heads * cfg.value_len];
            for h in 0..heads {
                let mut scores = Vec::with_capacity(keys.len() + 1);
                for key in keys {
                    let mut dot = 0.0f32;
                    for i in 0..kr {
                        dot += qabsorbed[h * kr + i] * key[i];
                    }
                    for i in 0..cfg.qk_rope_dim {
                        dot += qb[h * qdim + qnope_dim + i] * key[cfg.kv_lora_rank + i];
                    }
                    scores.push(dot * scale);
                }
                if il == 0 && h == 0 {
                    dump("scores_h0", &scores);
                }
                scores.push(sinks[h]); // sink logit
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                let vdim = cfg.value_len;
                for (j, key) in keys.iter().enumerate() {
                    let w = exps[j] / sum;
                    for i in 0..vdim {
                        attn[h * vdim + i] += w * key[i];
                    }
                }
            }
            // MLA value decompression via wv_b [512,256,heads].
            let (vt, vraw) = tensor(g, &format!("{b}attn_v_b.weight"))?;
            let vc = vt.dims[0] as usize;
            let vr = vt.dims[1] as usize;
            let vper = vraw.len() / heads;
            let mut decompressed = vec![0.0f32; heads * vr];
            for h in 0..heads {
                let base = &vraw[h * vper..];
                for r in 0..vr {
                    let mut s = 0.0f32;
                    for c in 0..vc {
                        s += dequant_one(vt.ty, base, r * vc + c) * attn[h * vc + c];
                    }
                    decompressed[h * vr + r] = s;
                }
            }
            if il == 0 {
                dump("attn_values", &attn);
                dump("attn_kqv", &decompressed);
            }
            let gate = mv(g, &format!("{b}attn_gate.weight"), &cur)?;
            let mut gated = decompressed.clone();
            for (a, &gval) in gated.iter_mut().zip(&gate) {
                *a *= sigmoid(gval);
            }
            if il == 0 {
                dump("attn_gated", &gated);
            }
            let mut ao = mv(g, &format!("{b}attn_output.weight"), &gated)?;
            if il == 0 {
                dump("attn_gated", &ao);
            }
            let pan = f32_vec(g, &format!("{b}post_attention_norm.weight"))?;
            ao = rms_norm(&ao, &pan, cfg.eps);
            if il == 0 {
                dump("attn_out", &ao);
            }

            // Scatter back into four streams with attention post gates.
            for s in 0..cfg.hc_count {
                for i in 0..cfg.hidden {
                    streams[s * cfg.hidden + i] += post[s] * ao[i];
                }
            }

            // FFN pre: recompute both gate sets from the scattered streams, reduce with `pre`.
            let (pre_ffn, post_ffn) = hc_ffn(g, &b, &streams, cfg)?;
            if il == 0 {
                // The oracle's cb_eval dump_cb writes every tensor whose name
                // matches, so duplicate `hc_pre-0` / `hc_post_gate-0` names
                // (attention and FFN in the same layer) leave the FFN values
                // in the file. Use the same names here for an apples-to-apples
                // comparison; the attention gates stay under their own names.
                dump("hc_pre", &pre_ffn);
                dump("hc_post_gate", &post_ffn);
            }
            let mut ffn_in = vec![0.0f32; cfg.hidden];
            for s in 0..cfg.hc_count {
                for i in 0..cfg.hidden {
                    ffn_in[i] += streams[s * cfg.hidden + i] * pre_ffn[s];
                }
            }
            let fnorm = f32_vec(g, &format!("{b}ffn_norm.weight"))?;
            let ffn_in = rms_norm(&ffn_in, &fnorm, cfg.eps);
            if il == 0 {
                dump("ffn_norm", &ffn_in);
            }

            let ffn_out = if il < cfg.leading_dense {
                // Dense FFN at layer 0; other PLE layers use MoE too in llama.cpp.
                dense_ffn(g, &b, &ffn_in)?
            } else {
                moe_ffn(g, &b, &ffn_in, cfg)?
            };
            let pfn = f32_vec(g, &format!("{b}post_ffw_norm.weight"))?;
            let ffn_out = rms_norm(&ffn_out, &pfn, cfg.eps);
            if il == 0 {
                dump("ffn_out", &ffn_out);
            }
            for s in 0..cfg.hc_count {
                for i in 0..cfg.hidden {
                    streams[s * cfg.hidden + i] += post_ffn[s] * ffn_out[i];
                }
            }
        }

        dump("l_out", &streams);
        // Head: hc reduce -> output norm -> lm_head.
        let mut flat = Vec::with_capacity(cfg.hidden * cfg.hc_count);
        for s in 0..cfg.hc_count {
            flat.extend_from_slice(&streams[s * cfg.hidden..(s + 1) * cfg.hidden]);
        }
        let mixes = mv(
            g,
            "output_hc_fn.weight",
            &rms_norm(&flat, &vec![1.0; flat.len()], cfg.eps),
        )?;
        let hbase = f32_vec(g, "output_hc_base.weight")?;
        let hscale = f32_vec(g, "output_hc_scale.weight")?[0];
        let mut hidden = vec![0.0f32; cfg.hidden];
        for s in 0..cfg.hc_count {
            let w = sigmoid(mixes[s] * hscale + hbase[s]) + 1e-6;
            for i in 0..cfg.hidden {
                hidden[i] += w * streams[s * cfg.hidden + i];
            }
        }
        dump("hc_head", &hidden);
        let on = f32_vec(g, "output_norm.weight")?;
        hidden = rms_norm(&hidden, &on, cfg.eps);
        dump("result_norm", &hidden);
        let logits = matvec(
            tensor(g, "output.weight")?.0,
            g.bytes(g.tensor("output.weight")?),
            &hidden,
        );
        st.pos += 1;
        st.prev_token = token;
        st.tokens.push(token);
        Ok(logits)
    }
}

fn dequant_one(ty: GgmlType, row: &[u8], i: usize) -> f32 {
    let (be, bb) = ty.block();
    let block = i / be;
    let off = block * bb;
    let mut scratch = vec![0.0f32; be];
    dequant(ty, &row[off..off + bb], &mut scratch);
    scratch[i % be]
}

fn hc_ffn(g: &Gguf, b: &str, streams: &[f32], cfg: &Config) -> Result<(Vec<f32>, Vec<f32>)> {
    let hc = cfg.hc_count;
    let mut flat = Vec::with_capacity(cfg.hidden * hc);
    for s in 0..hc {
        flat.extend_from_slice(&streams[s * cfg.hidden..(s + 1) * cfg.hidden]);
    }
    let normed = rms_norm(&flat, &vec![1.0; flat.len()], cfg.eps);
    let mixes = mv(g, &format!("{b}hc_ffn_fn.weight"), &normed)?;
    let base = f32_vec(g, &format!("{b}hc_ffn_base.weight"))?;
    let scale = f32_vec(g, &format!("{b}hc_ffn_scale.weight"))?;
    Ok(hc_gates(
        &mixes,
        &base,
        &scale,
        hc,
        cfg.hc_eps,
        cfg.hc_magnitude,
    ))
}

fn dense_ffn(g: &Gguf, b: &str, x: &[f32]) -> Result<Vec<f32>> {
    let up = mv(g, &format!("{b}ffn_up.weight"), x)?;
    let gate = mv(g, &format!("{b}ffn_gate.weight"), x)?;
    let act: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
        .collect();
    mv(g, &format!("{b}ffn_down.weight"), &act)
}

fn moe_ffn(g: &Gguf, b: &str, x: &[f32], cfg: &Config) -> Result<Vec<f32>> {
    let logits = mv(g, &format!("{b}ffn_gate_inp.weight"), x)?;
    let bias = f32_vec(g, &format!("{b}exp_probs_b.bias"))?;
    let mut sel: Vec<(f32, usize)> = logits
        .iter()
        .zip(&bias)
        .enumerate()
        .map(|(e, (&l, &b))| (sigmoid(l) + b, e))
        .collect();
    sel.sort_by(|a, b| b.0.total_cmp(&a.0));
    sel.truncate(cfg.n_expert_used);
    let mut weights: Vec<f32> = sel.iter().map(|&(_, e)| sigmoid(logits[e])).collect();
    let wsum: f32 = weights.iter().sum::<f32>().max(6.103515625e-5);
    for w in &mut weights {
        *w = *w / wsum * cfg.expert_weights_scale;
    }
    let routed = mv(g, &format!("{b}ffn_routed_down.weight"), x)?;
    let mut out = vec![0.0f32; cfg.hidden];
    for (idx, &(_, e)) in sel.iter().enumerate() {
        let up = emv(g, &format!("{b}ffn_up_exps.weight"), e, &routed)?;
        let gate = emv(g, &format!("{b}ffn_gate_exps.weight"), e, &routed)?;
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| {
                let g = g.min(10.0);
                let u = u.clamp(-10.0, 10.0);
                g / (1.0 + (-g).exp()) * u
            })
            .collect();
        let down = emv(g, &format!("{b}ffn_down_exps.weight"), e, &act)?;
        let latent = mv(g, &format!("{b}ffn_routed_up.weight"), &down)?;
        for (o, &v) in out.iter_mut().zip(&latent) {
            *o += weights[idx] * v;
        }
    }
    let sg = mv(g, &format!("{b}ffn_gate_shexp.weight"), x)?;
    let su = mv(g, &format!("{b}ffn_up_shexp.weight"), x)?;
    let sa: Vec<f32> = sg
        .iter()
        .zip(&su)
        .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
        .collect();
    let shared = mv(g, &format!("{b}ffn_down_shexp.weight"), &sa)?;
    for (o, &v) in out.iter_mut().zip(&shared) {
        *o += v;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hc_gates_match_llama_cpp_oracle() {
        // Mixes reconstructed from layer 0's GGUF hc_ffn weights and the
        // oracle's pre-FFN streams (derived from its l_out/ffn_out dumps);
        // expected gates are the oracle's hc_pre-0 / hc_post_gate-0 dumps.
        let mixes = [
            46.363404919962626,
            13.768153751825583,
            61.25920194286107,
            -0.21866374971841163,
            -71.4978042631353,
            -73.01675480395812,
            -68.49864367902363,
            -75.44309509037527,
        ];
        let base = [
            -0.45957013964653015,
            -1.0124915838241577,
            -0.5869624018669128,
            -1.2386435270309448,
            -1.0106960535049438,
            -0.9643658995628357,
            -0.9665372967720032,
            -0.9494650959968567,
        ];
        let scale = [0.036176469177007675, 0.03920025750994682];
        let (pre, post) = hc_gates(&mixes, &base, &scale, 4, 1e-6, 2.0);
        let want_pre = [0.77165854, 0.37416163, 0.8360582, 0.2232982];
        let want_post = [0.04319239, 0.04263756, 0.05058242, 0.0394159];
        for (got, want) in pre.iter().zip(want_pre) {
            assert!((got - want).abs() < 1e-6, "pre: {got} != {want}");
        }
        for (got, want) in post.iter().zip(want_post) {
            assert!((got - want).abs() < 1e-6, "post: {got} != {want}");
        }
    }
}
