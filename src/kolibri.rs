//! Kolibri-1 (`kolibri1`) CPU forward pass.
//!
//! Reference: llama.cpp `src/models/kolibri1.cpp` and vllm.cpp
//! `src/vllm/model_executor/models/kolibri1_forward.cpp`. Standard GQA attention with
//! per-head QK-norm, RoPE only on sliding-window layers (global layers are NoPE), followed by
//! a sigmoid-routed top-6 MoE with an always-on ungated shared expert.

use crate::cpu_moe::{QuantX, dot_q8};
use crate::gguf::{Gguf, TensorInfo};
use crate::quant::GgmlType;
use anyhow::{Result, ensure};
use rayon::prelude::*;

fn dims<'a>(g: &'a Gguf, name: &str, want: &[u64]) -> Result<&'a TensorInfo> {
    let t = g.tensor(name)?;
    ensure!(
        t.dims.as_slice() == want,
        "{name}: dims {:?}, expected {want:?}",
        t.dims
    );
    Ok(t)
}

#[derive(Debug, Clone)]
pub struct Config {
    pub layers: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub experts: usize,
    pub topk: usize,
    pub expert_inter: usize,
    pub shared_inter: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub rope_dim: usize,
    pub swa: usize,
    pub is_swa: Vec<bool>,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let p = "kolibri1.";
        let u = |s: &str| g.meta_u64(&format!("{p}{s}")).map(|v| v as usize);
        let f = |s: &str| g.meta_f64(&format!("{p}{s}")).map(|v| v as f32);
        let cfg_head_dim = u("attention.key_length")?;
        let is_swa = g
            .meta_array(&format!("{p}attention.sliding_window_pattern"))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) != 0)
            .collect();
        Ok(Self {
            layers: u("block_count")?,
            hidden: u("embedding_length")?,
            vocab: g.tensor("token_embd.weight")?.dims[1] as usize,
            heads: u("attention.head_count")?,
            kv_heads: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            experts: u("expert_count")?,
            topk: u("expert_used_count")?,
            expert_inter: u("expert_feed_forward_length")?,
            shared_inter: u("expert_shared_feed_forward_length")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            rope_theta: f("rope.freq_base")?,
            // 1.0 GGUFs omit rope.dimension_count; head_dim is the rotary width.
            rope_dim: u("rope.dimension_count").unwrap_or(cfg_head_dim),
            swa: u("attention.sliding_window")?,
            is_swa,
        })
    }
}

/// Check every tensor family needed by the forward pass before attempting to load.
pub fn validate(g: &Gguf) -> Result<()> {
    ensure!(
        g.meta_str("general.architecture")? == "kolibri1",
        "not a kolibri1 GGUF"
    );
    // 1.1 GGUFs reuse arch="kolibri1"; refuse them by their marker KVs.
    // Six markers, identical to the vllm.cpp U1 guard: the first three discriminate the
    // official 1.1 file, the rest are belt-and-braces.
    let markers = [
        "kolibri1.attention.indexer.top_k",
        "kolibri1.attention.indexer.types",
        "kolibri1.attention.kv_source_map",
        "kolibri1.attention.indexer.candidate_block_size",
        "kolibri1.attention.indexer.candidate_top_k_blocks",
        "kolibri1.leading_dense_block_count",
    ];
    ensure!(
        !markers.iter().any(|k| g.meta.contains_key(*k)),
        "kolibri1_1 is not supported (GGUF reports arch kolibri1)"
    );
    let cfg = Config::from_gguf(g)?;
    let h = cfg.hidden as u64;
    let e = cfg.experts as u64;
    let fi = cfg.expert_inter as u64;
    let si = cfg.shared_inter as u64;
    let q = (cfg.head_dim * cfg.heads) as u64;
    let kv = (cfg.head_dim * cfg.kv_heads) as u64;
    dims(g, "token_embd.weight", &[h, cfg.vocab as u64])?;
    dims(g, "output.weight", &[h, cfg.vocab as u64])?;
    dims(g, "output_norm.weight", &[h])?;
    for i in 0..cfg.layers {
        let b = format!("blk.{i}.");
        dims(g, &format!("{b}attn_norm.weight"), &[h])?;
        dims(g, &format!("{b}post_attention_norm.weight"), &[h])?;
        dims(g, &format!("{b}attn_q.weight"), &[h, q])?;
        dims(g, &format!("{b}attn_q_norm.weight"), &[cfg.head_dim as u64])?;
        dims(g, &format!("{b}attn_k_norm.weight"), &[cfg.head_dim as u64])?;
        dims(g, &format!("{b}attn_output.weight"), &[q, h])?;
        dims(g, &format!("{b}ffn_norm.weight"), &[h])?;
        dims(g, &format!("{b}post_ffw_norm.weight"), &[h])?;
        dims(g, &format!("{b}ffn_gate_inp.weight"), &[h, e])?;
        dims(g, &format!("{b}exp_probs_b.bias"), &[e])?;
        dims(g, &format!("{b}ffn_gate_exps.weight"), &[h, fi, e])?;
        dims(g, &format!("{b}ffn_up_exps.weight"), &[h, fi, e])?;
        dims(g, &format!("{b}ffn_down_exps.weight"), &[fi, h, e])?;
        dims(g, &format!("{b}ffn_gate_shexp.weight"), &[h, si])?;
        dims(g, &format!("{b}ffn_up_shexp.weight"), &[h, si])?;
        dims(g, &format!("{b}ffn_down_shexp.weight"), &[si, h])?;
        // K/V exist on layers that own their cache; 1.1's shared-KV layers omit them.
        if g.tensors.contains_key(&format!("{b}attn_k.weight")) {
            dims(g, &format!("{b}attn_k.weight"), &[h, kv])?;
            dims(g, &format!("{b}attn_v.weight"), &[h, kv])?;
        }
    }
    Ok(())
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum: f32 = x.iter().map(|v| v * v).sum();
    let s = 1.0 / (sum / x.len() as f32 + eps).sqrt();
    x.iter().zip(w).map(|(&v, &k)| v * s * k).collect()
}

fn matvec_f32(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    (0..rows)
        .into_par_iter()
        .with_min_len(16)
        .map(|r| {
            w[r * cols..(r + 1) * cols]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

/// GGML 2-D weights are [input, output]; `rows` is the output dimension.
/// `qx` is the activation quantization, computed once per input vector and reused across
/// all matvecs that read it (q/k/v, router, every selected expert and the shared expert).
fn matvec(t: &TensorInfo, raw: &[u8], x: &[f32], qx: Option<&QuantX>) -> Vec<f32> {
    let cols = t.dims[0] as usize;
    let rows = t.dims[1] as usize;
    if t.ty == GgmlType::F32 {
        let w: &[f32] = bytemuck::cast_slice(&raw[..rows * cols * 4]);
        return matvec_f32(w, rows, cols, x);
    }
    let owned;
    let qx = match qx {
        Some(q) => q,
        None => {
            owned = QuantX::from_f32(x);
            &owned
        }
    };
    let rb = t.ty.row_bytes(cols);
    (0..rows)
        .into_par_iter()
        .with_min_len(16)
        .map(|r| dot_q8(t.ty, &raw[r * rb..(r + 1) * rb], qx, x))
        .collect()
}

#[derive(Clone)]
struct Kv {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

pub struct State {
    kv: Vec<Kv>,
    rope: Vec<Vec<Vec<(f32, f32)>>>,
    pos: usize,
}

impl State {
    pub fn new(cfg: &Config) -> Self {
        Self {
            kv: vec![
                Kv {
                    keys: Vec::new(),
                    values: Vec::new(),
                };
                cfg.layers
            ],
            rope: Vec::new(),
            pos: 0,
        }
    }
}

pub struct Model<'a> {
    g: &'a Gguf,
    cfg: Config,
}

impl<'a> Model<'a> {
    pub fn new(g: &'a Gguf) -> Result<Self> {
        validate(g)?;
        let cfg = Config::from_gguf(g)?;
        let mut rope: Vec<Vec<Vec<(f32, f32)>>> = Vec::with_capacity(cfg.layers);
        for _ in 0..cfg.layers {
            rope.push(Vec::new());
        }
        // Precompute cos/sin per position up to the context, lazily extended in forward.
        Ok(Self { g, cfg })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    fn tensor(&self, name: &str) -> Result<(&TensorInfo, &[u8])> {
        let t = self.g.tensor(name)?;
        Ok((t, self.g.bytes(t)))
    }

    fn mat(&self, name: &str, x: &[f32], qx: Option<&QuantX>) -> Result<Vec<f32>> {
        let (t, raw) = self.tensor(name)?;
        Ok(matvec(t, raw, x, qx))
    }

    fn vec(&self, name: &str) -> Result<Vec<f32>> {
        let (_, raw) = self.tensor(name)?;
        Ok(bytemuck::cast_slice::<u8, f32>(raw).to_vec())
    }

    fn expert(t: &TensorInfo, raw: &[u8], e: usize, x: &[f32], qx: Option<&QuantX>) -> Vec<f32> {
        let cols = t.dims[0] as usize;
        let rows = t.dims[1] as usize;
        if t.ty == GgmlType::F32 {
            let per = rows * cols * 4;
            let w: &[f32] = bytemuck::cast_slice(&raw[e * per..(e + 1) * per]);
            return matvec_f32(w, rows, cols, x);
        }
        let rb = t.ty.row_bytes(cols);
        let per = rows * rb;
        let owned;
        let qx = match qx {
            Some(q) => q,
            None => {
                owned = QuantX::from_f32(x);
                &owned
            }
        };
        (0..rows)
            .into_par_iter()
            .map(|r| dot_q8(t.ty, &raw[e * per + r * rb..e * per + (r + 1) * rb], &qx, x))
            .collect()
    }

    /// Three projections in one parallel loop: q/k/v row work is dispatched once instead of
    /// three times, which cuts the per-invocation rayon epoch overhead.
    fn matvec3(ws: [(&TensorInfo, &[u8]); 3], x: &[f32], qx: &QuantX) -> [Vec<f32>; 3] {
        let rows: Vec<usize> = ws.iter().map(|(t, _)| t.dims[1] as usize).collect();
        let mut flat = vec![0.0f32; rows.iter().sum()];
        flat.par_iter_mut()
            .enumerate()
            .with_min_len(16)
            .for_each(|(idx, dst)| {
                let (wi, row) = if idx < rows[0] {
                    (0, idx)
                } else if idx < rows[0] + rows[1] {
                    (1, idx - rows[0])
                } else {
                    (2, idx - rows[0] - rows[1])
                };
                let (t, raw) = ws[wi];
                let cols = t.dims[0] as usize;
                let rb = t.ty.row_bytes(cols);
                *dst = dot_q8(t.ty, &raw[row * rb..(row + 1) * rb], qx, x);
            });
        let mut out = Vec::with_capacity(3);
        let mut off = 0;
        for r in rows {
            out.push(flat[off..off + r].to_vec());
            off += r;
        }
        [out.remove(0), out.remove(0), out.remove(0)]
    }

    /// Sequential row loop: used when experts themselves are scheduled in parallel.
    fn expert_seq(t: &TensorInfo, raw: &[u8], e: usize, x: &[f32], qx: &QuantX) -> Vec<f32> {
        let cols = t.dims[0] as usize;
        let rows = t.dims[1] as usize;
        let rb = t.ty.row_bytes(cols);
        let per = rows * rb;
        let base = &raw[e * per..(e + 1) * per];
        (0..rows)
            .map(|r| dot_q8(t.ty, &base[r * rb..(r + 1) * rb], qx, x))
            .collect()
    }

    fn ensure_rope(st: &mut State, cfg: &Config, pos: usize) {
        if st.rope.len() < cfg.layers {
            st.rope.resize(cfg.layers, Vec::new());
        }
        for layer in 0..cfg.layers {
            if !cfg.is_swa[layer] {
                continue;
            }
            let row = &mut st.rope[layer];
            while row.len() <= pos {
                let p = row.len() as f32;
                let mut pair = Vec::with_capacity(cfg.head_dim / 2);
                for j in 0..cfg.head_dim / 2 {
                    let angle = p / cfg.rope_theta.powf(2.0 * j as f32 / cfg.head_dim as f32);
                    pair.push((angle.cos(), angle.sin()));
                }
                row.push(pair);
            }
        }
    }

    fn apply_rope(x: &[f32], head_dim: usize, table: &[(f32, f32)]) -> Vec<f32> {
        let mut out = x.to_vec();
        for (i, chunk) in out.chunks_exact_mut(head_dim).enumerate() {
            let angle = &table[i % (head_dim / 2)];
            for j in 0..head_dim / 2 {
                let a = chunk[j];
                let b = chunk[j + head_dim / 2];
                chunk[j] = a * angle.0 - b * angle.1;
                chunk[j + head_dim / 2] = a * angle.1 + b * angle.0;
            }
        }
        out
    }

    /// Run one token, returning unpadded logits.
    pub fn forward(&self, st: &mut State, token: u32) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let pos = st.pos;
        Self::ensure_rope(st, cfg, pos);
        let (et, eraw) = self.tensor("token_embd.weight")?;
        let rb = et.ty.row_bytes(cfg.hidden);
        let mut emb = vec![0.0f32; cfg.hidden];
        crate::quant::dequant(
            et.ty,
            &eraw[token as usize * rb..(token as usize + 1) * rb],
            &mut emb,
        );
        let mut x = emb;

        for il in 0..cfg.layers {
            let b = format!("blk.{il}.");
            let residual = x.clone();
            let h = rms_norm(&x, &self.vec(&format!("{b}attn_norm.weight"))?, cfg.eps);
            let qx_h = QuantX::from_f32(&h);
            let (qt, qr) = self.tensor(&format!("{b}attn_q.weight"))?;
            let (kt, kr) = self.tensor(&format!("{b}attn_k.weight"))?;
            let (vt, vr) = self.tensor(&format!("{b}attn_v.weight"))?;
            let [mut q, mut k, v] = Self::matvec3([(qt, qr), (kt, kr), (vt, vr)], &h, &qx_h);
            // Per-head QK norm before RoPE.
            for (buf, name) in [
                (&mut q, "attn_q_norm.weight"),
                (&mut k, "attn_k_norm.weight"),
            ] {
                let w = self.vec(&format!("{b}{name}"))?;
                for head in buf.chunks_exact_mut(cfg.head_dim) {
                    let sum: f32 = head.iter().map(|v| v * v).sum();
                    let s = 1.0 / (sum / cfg.head_dim as f32 + cfg.eps).sqrt();
                    for (v, &k) in head.iter_mut().zip(&w) {
                        *v *= s * k;
                    }
                }
            }
            if cfg.is_swa[il] {
                let table = &st.rope[il][pos][..];
                q = Self::apply_rope(&q, cfg.head_dim, table);
                k = Self::apply_rope(&k, cfg.head_dim, table);
            }
            let kv = &mut st.kv[il];
            kv.keys.push(k);
            kv.values.push(v);
            let start =
                kv.keys
                    .len()
                    .saturating_sub(if cfg.is_swa[il] { cfg.swa } else { pos + 1 });
            let scale = 1.0 / (cfg.head_dim as f32).sqrt();
            let kv_heads = cfg.kv_heads;
            let group = cfg.heads / kv_heads;
            let mut attn = vec![0.0f32; cfg.heads * cfg.head_dim];
            attn.par_chunks_mut(cfg.head_dim)
                .enumerate()
                .for_each(|(head, out)| {
                    let kh = head / group;
                    let qh = &q[head * cfg.head_dim..(head + 1) * cfg.head_dim];
                    let mut scores = Vec::with_capacity(kv.keys.len() - start);
                    for key in &kv.keys[start..] {
                        let kk = &key[kh * cfg.head_dim..(kh + 1) * cfg.head_dim];
                        scores.push(qh.iter().zip(kk).map(|(a, b)| a * b).sum::<f32>() * scale);
                    }
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                    let sum: f32 = exps.iter().sum();
                    for (w, val) in exps.iter().zip(&kv.values[start..]) {
                        let vv = &val[kh * cfg.head_dim..(kh + 1) * cfg.head_dim];
                        for (o, &x) in out.iter_mut().zip(vv) {
                            *o += w / sum * x;
                        }
                    }
                });
            let qx_attn = QuantX::from_f32(&attn);
            let a = self.mat(&format!("{b}attn_output.weight"), &attn, Some(&qx_attn))?;
            let a = rms_norm(
                &a,
                &self.vec(&format!("{b}post_attention_norm.weight"))?,
                cfg.eps,
            );
            let ffn_inp: Vec<f32> = residual.iter().zip(&a).map(|(r, a)| r + a).collect();
            let z = rms_norm(
                &ffn_inp,
                &self.vec(&format!("{b}ffn_norm.weight"))?,
                cfg.eps,
            );

            let qx_z = QuantX::from_f32(&z);
            let logits = self.mat(&format!("{b}ffn_gate_inp.weight"), &z, Some(&qx_z))?;
            let bias = self.vec(&format!("{b}exp_probs_b.bias"))?;
            let mut selected: Vec<(f32, usize)> = logits
                .iter()
                .zip(&bias)
                .enumerate()
                .map(|(e, (&l, &b))| (l + b, e))
                .collect();
            selected.sort_by(|a, b| b.0.total_cmp(&a.0));
            selected.truncate(cfg.topk);
            let (gt, gr) = self.tensor(&format!("{b}ffn_gate_exps.weight"))?;
            let (ut, ur) = self.tensor(&format!("{b}ffn_up_exps.weight"))?;
            let (dt, dr) = self.tensor(&format!("{b}ffn_down_exps.weight"))?;
            let (weighted, shared) = rayon::join(
                || {
                    selected
                        .par_iter()
                        .map(|&(_, e)| {
                            let gate = Self::expert_seq(gt, gr, e, &z, &qx_z);
                            let up = Self::expert_seq(ut, ur, e, &z, &qx_z);
                            let act: Vec<f32> = gate
                                .iter()
                                .zip(&up)
                                .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
                                .collect();
                            let qx_act = QuantX::from_f32(&act);
                            let down = Self::expert_seq(dt, dr, e, &act, &qx_act);
                            // Sigmoid of the UNBIASED logit; no top-k renormalisation.
                            let w = 1.0 / (1.0 + (-logits[e]).exp());
                            down.into_iter().map(|d| w * d).collect::<Vec<f32>>()
                        })
                        .collect::<Vec<Vec<f32>>>()
                },
                || -> Result<Vec<f32>> {
                    let sg = self.mat(&format!("{b}ffn_gate_shexp.weight"), &z, Some(&qx_z))?;
                    let su = self.mat(&format!("{b}ffn_up_shexp.weight"), &z, Some(&qx_z))?;
                    let sa: Vec<f32> = sg
                        .iter()
                        .zip(&su)
                        .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
                        .collect();
                    let qx_sa = QuantX::from_f32(&sa);
                    self.mat(&format!("{b}ffn_down_shexp.weight"), &sa, Some(&qx_sa))
                },
            );
            let shared = shared?;
            let mut moe = vec![0.0f32; cfg.hidden];
            for contribution in weighted {
                for (o, d) in moe.iter_mut().zip(contribution) {
                    *o += d;
                }
            }
            for (o, s) in moe.iter_mut().zip(shared) {
                *o += s;
            }
            let out = rms_norm(
                &moe,
                &self.vec(&format!("{b}post_ffw_norm.weight"))?,
                cfg.eps,
            );
            x = ffn_inp.iter().zip(&out).map(|(f, o)| f + o).collect();
        }
        let norm = rms_norm(&x, &self.vec("output_norm.weight")?, cfg.eps);
        let qx_norm = QuantX::from_f32(&norm);
        let logits = self.mat("output.weight", &norm, Some(&qx_norm))?;
        st.pos += 1;
        Ok(logits.into_iter().take(cfg.vocab).collect())
    }
}

pub fn generate(
    model: &Model,
    st: &mut State,
    prompt: &[u32],
    max_new: usize,
    temp: f32,
    mut on_token: impl FnMut(u32),
) -> Result<()> {
    let mut sampler = crate::sampler::Sampler::new(crate::sampler::Params {
        temperature: temp,
        top_k: if temp <= 0.0 { 1 } else { 128 },
        top_p: 0.99,
        seed: 0x5eed,
    });
    let mut logits = Vec::new();
    for &t in prompt {
        logits = model.forward(st, t)?;
    }
    if std::env::var_os("INKLING2_DEBUG").is_some() {
        let mut top: Vec<(f32, usize)> = logits
            .iter()
            .copied()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect();
        top.sort_by(|a, b| b.0.total_cmp(&a.0));
        eprintln!(
            "kolibri1 top10: {}",
            top[..8]
                .iter()
                .map(|(v, i)| format!("{i}:{v:.2}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    for _ in 0..max_new {
        let next = sampler.sample(&logits);
        if std::env::var_os("INKLING2_DEBUG").is_some() {
            eprintln!("TOKEN {next}");
        }
        on_token(next);
        logits = model.forward(st, next)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn kolibri_1_q4km_inventory() {
        let path = Path::new("/mnt/models/Hob-forge-Kolibri-1-GGUF/Kolibri-1-Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("skipping: Kolibri-1 GGUF not present");
            return;
        }
        let g = Gguf::open(path).unwrap();
        validate(&g).unwrap();
    }
}
