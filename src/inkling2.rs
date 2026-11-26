//! Inkling-2 (`inkling2_loop`) tensor inventory and forward pass.
//!
//! The reference is llama.cpp `src/models/inkling2.cpp` + `inkling2-loop.cpp`. This is a
//! CPU-first port: it reads mmap'd GGUF bytes directly and uses `cpu_moe::dot_q8` (native
//! NEON on aarch64), so decode measurements are meaningful without the CubeCL CPU backend.

use crate::cpu_moe::{QuantX, dot_q8};
use crate::gguf::{Gguf, TensorInfo};
use crate::quant::GgmlType;
use anyhow::{Result, ensure};

/// Tensor dimensions in gguf order (`dims[0]` is the contiguous dimension).
fn dims<'a>(g: &'a Gguf, name: &str, want: &[u64]) -> Result<&'a TensorInfo> {
    let t = g.tensor(name)?;
    ensure!(
        t.dims.as_slice() == want,
        "{name}: dims {:?}, expected {want:?}",
        t.dims
    );
    Ok(t)
}

/// Validate the global and per-layer tensor inventory of an Inkling-2 checkpoint.
pub fn validate(g: &Gguf) -> Result<()> {
    ensure!(
        g.meta_str("general.architecture")? == "inkling2_loop",
        "not an inkling2_loop GGUF"
    );
    let n_layers = g.meta_u64("inkling2_loop.block_count")? as usize;
    let hidden = g.meta_u64("inkling2_loop.embedding_length")? as usize as u64;
    let vocab = g.meta_u64("inkling2_loop.vocab_size")? as usize as u64;
    let n_experts = g.meta_u64("inkling2_loop.expert_count")? as usize as u64;

    dims(g, "token_embd.weight", &[hidden, vocab])?;
    dims(g, "output.weight", &[hidden, vocab])?;
    dims(g, "output_norm.weight", &[hidden])?;

    for i in 0..n_layers {
        let b = format!("blk.{i}.");
        let has = |s: &str| g.tensors.contains_key(&format!("{b}{s}"));
        dims(g, &format!("{b}attn_q.weight"), &[hidden, 3072])?;
        dims(g, &format!("{b}attn_gate.weight"), &[hidden, 3072])?;
        dims(g, &format!("{b}attn_k.weight"), &[hidden, 1024])?;
        dims(g, &format!("{b}attn_v.weight"), &[hidden, 1024])?;
        dims(g, &format!("{b}attn_output.weight"), &[3072, hidden])?;
        dims(g, &format!("{b}attn_rel.weight"), &[hidden, 384])?;
        dims(g, &format!("{b}attn_rel_bias.weight"), &[16, 1024])?;
        dims(g, &format!("{b}attn_rel_clip.weight"), &[16, 24])?;
        dims(g, &format!("{b}attn_sink_proj.weight"), &[hidden, 24])?;
        dims(g, &format!("{b}attn_q_clip.weight"), &[128, 24])?;
        dims(g, &format!("{b}attn_k_conv.weight"), &[4, 1024])?;
        dims(g, &format!("{b}attn_v_conv.weight"), &[4, 1024])?;
        dims(g, &format!("{b}attn_post_conv.weight"), &[4, hidden])?;
        dims(g, &format!("{b}ffn_post_conv.weight"), &[4, hidden])?;
        dims(g, &format!("{b}ffn_gate_inp.weight"), &[hidden, n_experts])?;
        dims(g, &format!("{b}exp_probs_b.bias"), &[n_experts])?;
        let gu = g.tensor(&format!("{b}ffn_gate_up_exps.weight"))?;
        ensure!(
            gu.dims.as_slice() == [hidden, 2 * 1152, n_experts],
            "{b}ffn_gate_up_exps.weight: unexpected dims {:?}",
            gu.dims
        );
        let down = g.tensor(&format!("{b}ffn_down_exps.weight"))?;
        ensure!(
            down.dims.as_slice() == [1152, hidden, n_experts],
            "{b}ffn_down_exps.weight: unexpected dims {:?}",
            down.dims
        );
        if i == 0 {
            ensure!(
                !has("attn_res_query.weight"),
                "blk.0: unexpected attention residual query"
            );
        } else if i % 6 == 5 {
            dims(g, &format!("{b}attn_res_query.weight"), &[hidden, 3])?;
        } else {
            dims(g, &format!("{b}attn_res_query.weight"), &[hidden])?;
        }
        dims(g, &format!("{b}ffn_res_query.weight"), &[hidden])?;
        if (6..18).contains(&i) {
            for stream in ["q", "k", "v", "ffn"] {
                dims(
                    g,
                    &format!("{b}loop_{stream}_ctrl_down.weight"),
                    &[4608, 64],
                )?;
                dims(
                    g,
                    &format!("{b}loop_{stream}_ctrl_forget.weight"),
                    &[64, hidden],
                )?;
                dims(
                    g,
                    &format!("{b}loop_{stream}_ctrl_gain.weight"),
                    &[64, hidden],
                )?;
            }
            dims(g, &format!("{b}loop_msg_query.weight"), &[hidden, 4])?;
        } else {
            ensure!(
                !has("loop_msg_query.weight"),
                "{b}: unexpected loop controllers"
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Config and small numeric helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Config {
    pub n_layers: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub vocab_unpadded: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub ff_exp: usize,
    pub n_experts: usize,
    pub n_expert_used: usize,
    pub eps: f32,
    pub qk_clip: f32,
    pub rel_len: usize,
    pub rel_extent: usize,
    pub rel_clip: f32,
    pub log_floor: usize,
    pub log_alpha: f32,
    pub situ_beta: f32,
    pub situ_lb: f32,
    pub expert_scale: f32,
    pub softcap: f32,
    pub swa: usize,
    pub conv: usize,
    pub block: usize,
    pub loop_start: usize,
    pub loop_end: usize,
    pub loop_passes: usize,
    pub gate_rank: usize,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let p = "inkling2_loop.";
        let u = |s: &str| g.meta_u64(&format!("{p}{s}")).map(|v| v as usize);
        let f = |s: &str| g.meta_f64(&format!("{p}{s}")).map(|v| v as f32);
        Ok(Self {
            n_layers: u("block_count")?,
            hidden: u("embedding_length")?,
            vocab: u("vocab_size")?,
            vocab_unpadded: u("vocab_size_unpadded")?,
            n_head: g
                .meta_array(&format!("{p}attention.head_count"))?
                .first()
                .and_then(|v| v.as_u64())
                .unwrap_or(24) as usize,
            n_head_kv: g
                .meta_array(&format!("{p}attention.head_count_kv"))?
                .first()
                .and_then(|v| v.as_u64())
                .unwrap_or(8) as usize,
            head_dim: u("attention.key_length")?,
            ff_exp: u("expert_feed_forward_length")?,
            n_experts: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            qk_clip: f("attention.qk_clip")?,
            rel_len: u("attention.relative_length")?,
            rel_extent: u("attention.relative_extent")?,
            rel_clip: f("attention.relative_clip")?,
            log_floor: u("attention.log_scale_floor")?,
            log_alpha: f("attention.log_scale_alpha")?,
            situ_beta: f("activation.situ_beta")?,
            situ_lb: f("activation.situ_linear_beta")?,
            expert_scale: f("expert_weights_scale").unwrap_or(1.0),
            softcap: f("final_logit_softcapping").unwrap_or(0.0),
            swa: u("attention.sliding_window")?,
            conv: u("shortconv.l_cache")?,
            block: u("attn_res.block_size")?,
            loop_start: u("loop.start")?,
            loop_end: u("loop.end")?,
            loop_passes: u("loop.count")?,
            gate_rank: u("loop.gate_rank")?,
        })
    }

    /// `sliding_window_pattern` is [T,T,T,T,T,F] x4: T = sliding-window attention,
    /// F (layers 5, 11, 17, 23) = full attention with separate Q/K/V input norms.
    fn is_swa(&self, layer: usize) -> bool {
        layer % self.block != self.block - 1
    }
}

// ---------------------------------------------------------------------------
// LoopCD decoding (arXiv:2610.02185)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoopCdMode {
    Off,
    Hidden,
    Logits,
}

#[derive(Debug, Clone, Copy)]
struct LoopCd {
    mode: LoopCdMode,
    /// Guidance strength ω; the upper bound when `adaptive` is set.
    omega: f32,
    adaptive: bool,
    /// Zero-based loop pass used as the weak reference (0 = first pass).
    ref_pass: usize,
}

impl LoopCd {
    fn from_env() -> Self {
        let mode = match std::env::var("LOOPCD").ok().as_deref() {
            Some("hidden") => LoopCdMode::Hidden,
            Some("logits") => LoopCdMode::Logits,
            _ => LoopCdMode::Off,
        };
        // The paper's generation window is ω ∈ [0.2, 0.3]; larger values compound errors.
        let omega = std::env::var("LOOPCD_OMEGA")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.25);
        let adaptive = std::env::var_os("LOOPCD_ADAPTIVE").is_some();
        let ref_pass = std::env::var("LOOPCD_REF_PASS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        Self {
            mode,
            omega,
            adaptive,
            ref_pass,
        }
    }

    /// Effective strength for this token. Adaptive scaling needs the unguided logits, so it
    /// is only used by `Logits` mode (Hidden has no extra output pass to read them from).
    fn strength(&self, logits: &[f32]) -> f32 {
        if !self.adaptive {
            return self.omega;
        }
        let (mut first, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &v in logits {
            if v > first {
                second = first;
                first = v;
            } else if v > second {
                second = v;
            }
        }
        if !second.is_finite() {
            return 0.0;
        }
        // ω = ω_max [1 - (p_(1) - p_(2))], Eq. (4). A softmax shift keeps exp in range.
        let mut sum = 0.0f32;
        for &v in logits {
            sum += (v - first).exp();
        }
        let p1 = 1.0 / sum;
        let p2 = (second - first).exp() / sum;
        (self.omega * (1.0 - (p1 - p2))).clamp(0.0, self.omega)
    }
}

/// `strong + w * (strong - weak)` — continue the direction recurrence took from the weak
/// state. A zero strength returns the strong vector unchanged (no rounding drift).
fn extrapolate(strong: &[f32], weak: &[f32], w: f32) -> Vec<f32> {
    assert_eq!(
        strong.len(),
        weak.len(),
        "LoopCD reference and final states have different lengths"
    );
    if w == 0.0 {
        return strong.to_vec();
    }
    strong
        .iter()
        .zip(weak)
        .map(|(&s, &r)| s + w * (s - r))
        .collect()
}

/// Softcap and drop padding tokens, exactly as the unguided path always did.
fn finish_logits(logits: &[f32], cfg: &Config) -> Vec<f32> {
    logits
        .iter()
        .take(cfg.vocab_unpadded)
        .map(|&v| {
            if cfg.softcap != 0.0 {
                cfg.softcap * (v / cfg.softcap).tanh()
            } else {
                v
            }
        })
        .collect()
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum: f32 = x.iter().map(|v| v * v).sum();
    let scale = 1.0 / (sum / x.len() as f32 + eps).sqrt();
    x.iter().zip(w).map(|(&v, &s)| v * scale * s).collect()
}

/// Clip each row (head) of `x` to L2 norm <= threshold, after optional per-channel scale.
fn clip(x: &[f32], width: usize, scale: Option<&[f32]>, threshold: f32) -> Vec<f32> {
    let mut out = x.to_vec();
    if let Some(s) = scale {
        for (v, &c) in out.iter_mut().zip(s) {
            *v *= c;
        }
    }
    for row in out.chunks_exact_mut(width) {
        // llama.cpp: sqrt(sum_rows(sqr(x))) then scale_bias(+1e-6), i.e. sqrt(sum) + eps.
        let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-6;
        let factor = (threshold / norm).min(1.0).max(0.0);
        for v in row {
            *v *= factor;
        }
    }
    out
}

/// Causal short convolution with residual, from llama.cpp `graph::convolve`.
/// `history` has `conv - 1` rows of `channels` values (oldest first); updated in place.
fn convolve(
    cur: &[f32],
    weight: &[f32],
    dynamic: Option<&[f32]>,
    history: &mut [f32],
    channels: usize,
    heads: usize,
    conv: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; channels];
    for lag in 0..conv {
        // ggml stores conv weights and dynamic taps as [lag, channel/head] with lag
        // contiguous: element (lag, c) lives at `c * conv + lag`.
        let window = if lag + 1 < conv {
            &history[lag * channels..(lag + 1) * channels]
        } else {
            cur
        };
        for (i, &x) in window.iter().enumerate() {
            out[i] += x * weight[i * conv + lag];
        }
        if let Some(d) = dynamic {
            let per_head = channels / heads;
            for h in 0..heads {
                let t = d[h * conv + lag];
                for i in 0..per_head {
                    out[h * per_head + i] += window[h * per_head + i] * t;
                }
            }
        }
    }
    // Shift history left by one row and append the current input.
    for lag in 0..conv - 2 {
        let (dst, src) = history.split_at_mut((lag + 1) * channels);
        dst[lag * channels..(lag + 1) * channels].copy_from_slice(&src[..channels]);
    }
    history[(conv - 2) * channels..(conv - 1) * channels].copy_from_slice(cur);
    let scale = 1.0 / (((if dynamic.is_some() { 2 } else { 1 }) * conv + 1) as f32).sqrt();
    out.iter()
        .zip(cur)
        .map(|(&o, &c)| (o + c) * scale)
        .collect()
}

/// Attention over the residual stream, from llama.cpp `graph::mix`.
fn mix(blocks: &[Vec<f32>], cur: &[f32], queries: &[f32], hidden: usize, eps: f32) -> Vec<f32> {
    let nq = queries.len() / hidden;
    let n = blocks.len() + 1;
    // llama.cpp prepends each block, so the newest block comes first and `cur` is last.
    let mut all: Vec<Vec<f32>> = blocks.iter().rev().cloned().collect();
    all.push(cur.to_vec());
    let mut states = Vec::with_capacity(n * hidden);
    for b in &all {
        states.extend_from_slice(b);
    }
    // ggml_rms_norm normalizes dim 0 (n_embd) independently for each residual block.
    let ones = vec![1.0f32; hidden];
    for block in states.chunks_exact_mut(hidden) {
        let sum: f32 = block.iter().map(|v| v * v).sum();
        let s = 1.0 / (sum / hidden as f32 + eps).sqrt();
        for (v, &one) in block.iter_mut().zip(&ones) {
            *v *= s * one;
        }
    }
    let scale = 1.0 / (hidden as f32).sqrt();
    let mut out = vec![0.0f32; hidden * nq];
    for q in 0..nq {
        let query = &queries[q * hidden..(q + 1) * hidden];
        let mut scores = vec![0.0f32; n];
        for (j, state) in states.chunks_exact(hidden).enumerate() {
            scores[j] = state.iter().zip(query).map(|(a, b)| a * b).sum::<f32>() * scale;
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (j, e) in exps.iter().enumerate() {
            let w = e / sum;
            for (o, &v) in out[q * hidden..(q + 1) * hidden]
                .iter_mut()
                .zip(&states[j * hidden..(j + 1) * hidden])
            {
                *o += w * v;
            }
        }
    }
    out
}

/// Gated cross-pass message merge, from llama.cpp `graph::control`.
fn control(
    cur: &[f32],
    msg: &[f32],
    down: &[f32],
    gain: &[f32],
    forget: &[f32],
    cfg: &Config,
) -> Vec<f32> {
    let mut gate = rms_norm(cur, &vec![1.0; cur.len()], 1e-6);
    gate.extend_from_slice(&rms_norm(msg, &vec![1.0; msg.len()], 1e-6));
    let h = matvec_f32(down, cfg.gate_rank, gate.len(), &gate);
    let g = matvec_f32(gain, cfg.hidden, cfg.gate_rank, &h);
    let f = matvec_f32(forget, cfg.hidden, cfg.gate_rank, &h);
    cur.iter()
        .zip(msg)
        .zip(g)
        .zip(f)
        .map(|(((&c, &m), gv), fv)| c - fv.tanh() * c + gv.tanh() * m)
        .collect()
}

/// Matrix-vector product over a flat F32 weight matrix (row-major, `rows` rows).
fn matvec_f32(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    assert_eq!(w.len(), rows * cols);
    use rayon::prelude::*;
    (0..rows)
        .into_par_iter()
        .map(|r| {
            w[r * cols..(r + 1) * cols]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

/// Matrix-vector product over a GGUF weight row slice, using native quantized dots.
fn matvec_gguf(ty: GgmlType, raw: &[u8], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    if ty == GgmlType::F32 {
        let w: &[f32] = bytemuck::cast_slice(&raw[..rows * cols * 4]);
        return matvec_f32(w, rows, cols, x);
    }
    let rb = ty.row_bytes(cols);
    if std::env::var_os("INKLING2_Q8_DOT").is_some() {
        // Faster but less accurate: per-32 Q8 activations. llama.cpp uses per-256 Q8_K
        // activations, so this shifts top logits enough to change greedy tokens.
        let qx = QuantX::from_f32(x);
        return (0..rows)
            .into_par_iter()
            .map(|r| dot_q8(ty, &raw[r * rb..(r + 1) * rb], &qx, x))
            .collect();
    }
    // Default: exact dequantized dot. Matches llama.cpp's top-1 token on the APEX oracle.
    use rayon::prelude::*;
    (0..rows)
        .into_par_iter()
        .map_init(
            || vec![0.0f32; cols], // one scratch buffer per worker, not per row
            |w, r| {
                crate::quant::dequant(ty, &raw[r * rb..(r + 1) * rb], w);
                w.iter().zip(x).map(|(a, b)| a * b).sum()
            },
        )
        .collect()
}

// ---------------------------------------------------------------------------
// Model, state and forward pass
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct KvSlot {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

#[derive(Clone)]
pub struct State {
    kv: Vec<KvSlot>,
    conv: Vec<Vec<f32>>,
    pos: usize,
}

impl State {
    pub fn new(cfg: &Config) -> Self {
        let slots = cfg.n_layers + (cfg.loop_passes - 1) * (cfg.loop_end - cfg.loop_start);
        let channels = 2 * cfg.n_head_kv * cfg.head_dim + 2 * cfg.hidden;
        Self {
            kv: (0..slots)
                .map(|_| KvSlot {
                    keys: Vec::new(),
                    values: Vec::new(),
                })
                .collect(),
            conv: vec![vec![0.0f32; (cfg.conv - 1) * channels]; slots],
            pos: 0,
        }
    }
}

/// Private state branch for the weak coda pass. Only the slots the coda touches are copied;
/// the real cache keeps the strong path's KV and conv history.
fn branch_state(st: &State, layers: std::ops::Range<usize>) -> State {
    let slots = st.kv.len();
    let mut kv = Vec::with_capacity(slots);
    let mut conv = Vec::with_capacity(slots);
    for i in 0..slots {
        let used = layers.contains(&i);
        kv.push(if used {
            st.kv[i].clone()
        } else {
            KvSlot {
                keys: Vec::new(),
                values: Vec::new(),
            }
        });
        conv.push(if used { st.conv[i].clone() } else { Vec::new() });
    }
    State {
        kv,
        conv,
        pos: st.pos,
    }
}

pub struct Model<'a> {
    g: &'a Gguf,
    cfg: Config,
    loopcd: LoopCd,
}

impl<'a> Model<'a> {
    pub fn new(g: &'a Gguf) -> Result<Self> {
        validate(g)?;
        let cfg = Config::from_gguf(g)?;
        let loopcd = LoopCd::from_env();
        if loopcd.mode != LoopCdMode::Off {
            ensure!(
                loopcd.ref_pass < cfg.loop_passes,
                "LOOPCD_REF_PASS {} >= loop.count {}",
                loopcd.ref_pass,
                cfg.loop_passes
            );
            if loopcd.mode == LoopCdMode::Hidden && loopcd.adaptive {
                eprintln!(
                    "loopcd: ignoring LOOPCD_ADAPTIVE in hidden mode (needs an extra output pass)"
                );
            }
            if std::env::var_os("INKLING2_DEBUG").is_some() {
                eprintln!(
                    "loopcd: {:?} omega={} adaptive={} ref_pass={}",
                    loopcd.mode, loopcd.omega, loopcd.adaptive, loopcd.ref_pass
                );
            }
        }
        Ok(Self { g, cfg, loopcd })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    fn tensor(&self, name: &str) -> Result<(&TensorInfo, &[u8])> {
        let t = self.g.tensor(name)?;
        Ok((t, self.g.bytes(t)))
    }

    /// Matvec over a 2-D weight tensor; `rows` is `dims[0]`, `cols` is `dims[1]`.
    fn mat(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        let (t, raw) = self.tensor(name)?;
        if t.dims.len() == 1 {
            // A 1-D query vector (residual query): no projection, just its weights.
            return Ok(bytemuck::cast_slice::<u8, f32>(raw).to_vec());
        }
        // ggml weights are stored [input, output]; mul_mat reads rows of `dims[0]` inputs.
        let cols = t.dims[0] as usize;
        let rows = t.dims[1] as usize;
        let rb = t.ty.row_bytes(cols);
        ensure!(
            raw.len() >= rows * rb,
            "{name}: {} bytes for {rows} rows of {rb} bytes ({:?})",
            raw.len(),
            t.ty
        );
        Ok(matvec_gguf(t.ty, raw, rows, cols, x))
    }

    /// Read an F32 residual-query weight matrix as raw values (not a projection).
    fn query(&self, name: &str) -> Result<Vec<f32>> {
        let (_, raw) = self.tensor(name)?;
        Ok(bytemuck::cast_slice::<u8, f32>(raw).to_vec())
    }

    /// The fixed post-loop readout: residual attention, final norm, lm_head and softcap.
    /// Factored out so LoopCD-Logits can decode the reference state through the same path.
    fn coda(&self, blocks: &[Vec<f32>], cur: &[f32]) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let oq: &[f32] = bytemuck::cast_slice(self.tensor("output_res_query.weight")?.1);
        let mixed = mix(blocks, cur, oq, cfg.hidden, cfg.eps);
        let on: &[f32] = bytemuck::cast_slice(self.tensor("output_norm.weight")?.1);
        let normed = rms_norm(&mixed, on, cfg.eps);
        let logits = self.mat("output.weight", &normed)?;
        Ok(finish_logits(&logits, cfg))
    }

    /// Read a 1-D F32 tensor (RMSNorm weights, biases) as a vector.
    fn f32_vec(&self, name: &str) -> Result<Vec<f32>> {
        let (_, raw) = self.tensor(name)?;
        Ok(bytemuck::cast_slice::<u8, f32>(raw).to_vec())
    }

    /// Per-expert output scale (`<weight>.scale`, one f32 per expert); empty when absent.
    fn expert_scale(&self, weight: &str) -> Result<Vec<f32>> {
        let name = format!("{}.scale", weight.trim_end_matches(".weight"));
        if self.g.tensors.contains_key(&name) {
            self.f32_vec(&name)
        } else {
            Ok(Vec::new())
        }
    }

    /// One expert's slice of a 3-D [rows, cols, experts] tensor.
    fn expert(&self, name: &str, e: usize) -> Result<(GgmlType, &[u8], usize, usize)> {
        let (t, raw) = self.tensor(name)?;
        let cols = t.dims[0] as usize;
        let rows = t.dims[1] as usize;
        let rb = t.ty.row_bytes(cols);
        ensure!(
            raw.len() >= (e + 1) * rows * rb,
            "{name}: expert {e} out of bounds ({} bytes, need {})",
            raw.len(),
            (e + 1) * rows * rb
        );
        Ok((t.ty, &raw[e * rows * rb..(e + 1) * rows * rb], rows, cols))
    }

    fn convolve(
        &self,
        st: &mut State,
        slot: usize,
        cur: &[f32],
        weight_name: &str,
        dynamic: Option<&[f32]>,
        offset: usize,
        heads: usize,
    ) -> Result<Vec<f32>> {
        let (_, raw) = self.tensor(weight_name)?;
        let channels = cur.len();
        let w: &[f32] = bytemuck::cast_slice(&raw[..self.cfg.conv * channels * 4]);
        let start = offset * (self.cfg.conv - 1);
        let hist = &mut st.conv[slot][start..start + (self.cfg.conv - 1) * channels];
        Ok(convolve(
            cur,
            w,
            dynamic,
            hist,
            channels,
            heads,
            self.cfg.conv,
        ))
    }

    /// Relative-position bias: R projected onto the bias table, minus the sink logit.
    fn rel_bias(&self, layer_name: &str, r: &[f32], sinks: &[f32]) -> Result<Vec<f32>> {
        let (_, raw) = self.tensor(&format!("{layer_name}attn_rel_bias.weight"))?;
        let table: &[f32] = bytemuck::cast_slice(raw);
        let (rel_len, extent) = (self.cfg.rel_len, self.cfg.rel_extent);
        let heads = self.cfg.n_head;
        let mut out = vec![0.0f32; (extent + 1) * heads];
        for d in 0..=extent {
            for h in 0..heads {
                // ggml layout: table is (rel_len, extent) with rel_len contiguous, padded
                // with one zero row at distance `extent`; r is (rel_len, heads).
                let mut s = 0.0f32;
                if d < extent {
                    for i in 0..rel_len {
                        s += table[d * rel_len + i] * r[h * rel_len + i];
                    }
                }
                out[d * heads + h] = s - sinks[h];
            }
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        st: &mut State,
        li: usize,
        ip: usize,
        slot: usize,
        x: &[f32],
        blocks: &mut Vec<Vec<f32>>,
        message: Option<(&[f32], usize)>,
        tau: f32,
    ) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let b = format!("blk.{li}.");
        let hidden = cfg.hidden;
        let is_swa = cfg.is_swa(ip);
        let n_kv_heads = cfg.n_head_kv;
        let kv_dim = n_kv_heads * cfg.head_dim;
        let inp_sa = x.to_vec();

        // Attention inputs: residual stream, optionally mixed with earlier blocks and
        // merged with the previous pass's gated message.
        let mut inputs = vec![x.to_vec(); 3];
        let res_name = format!("{b}attn_res_query.weight");
        if self.g.tensors.contains_key(&res_name) {
            let q = self.query(&res_name)?;
            let nq = q.len() / hidden;
            let mixed = mix(blocks, x, &q, hidden, cfg.eps);
            for (j, input) in inputs.iter_mut().enumerate() {
                let off = if is_swa {
                    0
                } else {
                    j.min(nq.saturating_sub(1))
                } * hidden;
                *input = mixed[off..off + hidden].to_vec();
            }
        }
        if let Some((msg, phys)) = message {
            for j in 0..3 {
                let base = format!("blk.{phys}.loop_{}_ctrl", ["q", "k", "v"][j]);
                let down: &[f32] =
                    bytemuck::cast_slice(self.tensor(&format!("{base}_down.weight"))?.1);
                let gain: &[f32] =
                    bytemuck::cast_slice(self.tensor(&format!("{base}_gain.weight"))?.1);
                let forget: &[f32] =
                    bytemuck::cast_slice(self.tensor(&format!("{base}_forget.weight"))?.1);
                let one = &msg[j * hidden..(j + 1) * hidden];
                inputs[j] = control(&inputs[j], one, down, gain, forget, cfg);
            }
        }
        let norm_name = if is_swa {
            format!("{b}attn_norm.weight")
        } else {
            format!("{b}attn_q_inp_norm.weight")
        };
        if is_swa && message.is_some() {
            let w = self.f32_vec(&norm_name)?;
            for input in &mut inputs {
                *input = rms_norm(input, &w, cfg.eps);
            }
        } else if is_swa {
            let w = self.f32_vec(&norm_name)?;
            inputs[0] = rms_norm(&inputs[0], &w, cfg.eps);
            inputs[1] = inputs[0].clone();
            inputs[2] = inputs[0].clone();
        } else {
            for (j, stream) in ["q", "k", "v"].iter().enumerate() {
                let w = self.f32_vec(&format!("{b}attn_{stream}_inp_norm.weight"))?;
                inputs[j] = rms_norm(&inputs[j], &w, cfg.eps);
            }
        }

        let mut q = self.mat(&format!("{b}attn_q.weight"), &inputs[0])?;
        let mut k = self.mat(&format!("{b}attn_k.weight"), &inputs[1])?;
        let mut v = self.mat(&format!("{b}attn_v.weight"), &inputs[2])?;
        let mut r = self.mat(&format!("{b}attn_rel.weight"), &inputs[0])?;

        let dynamic_k = if is_swa {
            None
        } else {
            Some(self.mat(&format!("{b}attn_k_conv_proj.weight"), &inputs[1])?)
        };
        let dynamic_v = if is_swa {
            None
        } else {
            Some(self.mat(&format!("{b}attn_v_conv_proj.weight"), &inputs[2])?)
        };
        k = self.convolve(
            st,
            slot,
            &k,
            &format!("{b}attn_k_conv.weight"),
            dynamic_k.as_deref(),
            0,
            n_kv_heads,
        )?;
        v = self.convolve(
            st,
            slot,
            &v,
            &format!("{b}attn_v_conv.weight"),
            dynamic_v.as_deref(),
            kv_dim,
            n_kv_heads,
        )?;

        let q_clip: &[f32] =
            bytemuck::cast_slice(self.tensor(&format!("{b}attn_q_clip.weight"))?.1);
        let r_clip: &[f32] =
            bytemuck::cast_slice(self.tensor(&format!("{b}attn_rel_clip.weight"))?.1);
        q = clip(&q, cfg.head_dim, Some(q_clip), cfg.qk_clip);
        k = clip(&k, cfg.head_dim, None, cfg.qk_clip);
        // Per-head RMS norm of V.
        for head in v.chunks_exact_mut(cfg.head_dim) {
            let s: f32 = head.iter().map(|x| x * x).sum::<f32>() / cfg.head_dim as f32;
            let scale = 1.0 / (s + cfg.eps).sqrt();
            for x in head {
                *x *= scale;
            }
        }
        r = clip(&r, cfg.rel_len, Some(r_clip), cfg.rel_clip);
        if !is_swa {
            for x in q.iter_mut().chain(r.iter_mut()) {
                *x *= tau;
            }
        }

        let sinks = self.mat(&format!("{b}attn_sink_proj.weight"), &inputs[0])?;
        let bias = self.rel_bias(&b, &r, &sinks)?;
        let kv = &mut st.kv[slot];
        kv.keys.push(k.clone());
        kv.values.push(v.clone());
        let start = kv.keys.len().saturating_sub(cfg.swa);
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        let mut attn = vec![0.0f32; cfg.n_head * cfg.head_dim];
        for h in 0..cfg.n_head {
            let kh = h / (cfg.n_head / n_kv_heads);
            let qh = &q[h * cfg.head_dim..(h + 1) * cfg.head_dim];
            let mut scores = Vec::with_capacity(kv.keys.len() - start + 1);
            let mut vids = Vec::with_capacity(kv.keys.len() - start);
            for (kpos, (key, val)) in kv.keys[start..].iter().zip(&kv.values[start..]).enumerate() {
                let _ = kpos;
                let kk = &key[kh * cfg.head_dim..(kh + 1) * cfg.head_dim];
                let dot: f32 = qh.iter().zip(kk).map(|(a, b)| a * b).sum();
                let d = st.pos.saturating_sub(start + kpos);
                scores.push(dot * scale + bias[d.min(cfg.rel_extent) * cfg.n_head + h]);
                vids.push(val);
            }
            scores.push(0.0); // zero sink
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let out = &mut attn[h * cfg.head_dim..(h + 1) * cfg.head_dim];
            for (e, val) in exps.iter().zip(vids) {
                let w = e / sum;
                let vv = &val[kh * cfg.head_dim..(kh + 1) * cfg.head_dim];
                for (o, &x) in out.iter_mut().zip(vv) {
                    *o += w * x;
                }
            }
        }

        let gate = self.mat(&format!("{b}attn_gate.weight"), &inputs[0])?;
        for (a, g) in attn.iter_mut().zip(gate) {
            *a *= 1.0 / (1.0 + (-g).exp());
        }
        let wo = self.mat(&format!("{b}attn_output.weight"), &attn)?;
        let mut cur = self.convolve(
            st,
            slot,
            &wo,
            &format!("{b}attn_post_conv.weight"),
            None,
            2 * kv_dim,
            cfg.hidden,
        )?;

        if ip % cfg.block == 0 {
            blocks.push(inp_sa);
        } else {
            for (c, &s) in cur.iter_mut().zip(&inp_sa) {
                *c += s;
            }
        }

        // MoE.
        let mut ffn_inp = cur.clone();
        let fq = self.mat(&format!("{b}ffn_res_query.weight"), &cur)?;
        cur = mix(blocks, &cur, &fq, hidden, cfg.eps);
        if let Some((msg, phys)) = message {
            let base = format!("blk.{phys}.loop_ffn_ctrl");
            let down: &[f32] = bytemuck::cast_slice(self.tensor(&format!("{base}_down.weight"))?.1);
            let gain: &[f32] = bytemuck::cast_slice(self.tensor(&format!("{base}_gain.weight"))?.1);
            let forget: &[f32] =
                bytemuck::cast_slice(self.tensor(&format!("{base}_forget.weight"))?.1);
            cur = control(&cur, &msg[3 * hidden..4 * hidden], down, gain, forget, cfg);
        }
        let norm: &[f32] = bytemuck::cast_slice(self.tensor(&format!("{b}ffn_norm.weight"))?.1);
        cur = rms_norm(&cur, norm, cfg.eps);

        let logits = self.mat(&format!("{b}ffn_gate_inp.weight"), &cur)?;
        let probs: Vec<f32> = logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
        let bias: &[f32] = bytemuck::cast_slice(self.tensor(&format!("{b}exp_probs_b.bias"))?.1);
        let selection: Vec<(f32, usize)> = probs
            .iter()
            .zip(bias)
            .enumerate()
            .map(|(e, (&p, &b))| (p + b, e))
            .collect();
        let mut selected: Vec<(f32, usize)> = selection;
        selected.sort_by(|a, b| b.0.total_cmp(&a.0));
        selected.truncate(cfg.n_expert_used);
        let expert_in = cur.clone();
        let wsum: f32 = selected.iter().map(|&(_, e)| probs[e]).sum::<f32>() + 1e-20;
        cur = vec![0.0f32; hidden];
        let gu_scale = self.expert_scale(&format!("{b}ffn_gate_up_exps.weight"))?;
        let dn_scale = self.expert_scale(&format!("{b}ffn_down_exps.weight"))?;
        for &(_, e) in &selected {
            let (gu_ty, gu, gu_rows, gu_cols) =
                self.expert(&format!("{b}ffn_gate_up_exps.weight"), e)?;
            let mut guv = matvec_gguf(gu_ty, gu, gu_rows, gu_cols, &expert_in);
            if !gu_scale.is_empty() {
                guv.iter_mut().for_each(|v| *v *= gu_scale[e]);
            }
            let ff = gu_rows / 2;
            let act: Vec<f32> = guv[..ff]
                .iter()
                .zip(&guv[ff..])
                .map(|(&gate, &up)| {
                    cfg.situ_beta * (gate / cfg.situ_beta).tanh() / (1.0 + (-gate).exp())
                        * cfg.situ_lb
                        * (up / cfg.situ_lb).tanh()
                })
                .collect();
            let (dn_ty, dn, dn_rows, dn_cols) =
                self.expert(&format!("{b}ffn_down_exps.weight"), e)?;
            let mut down = matvec_gguf(dn_ty, dn, dn_rows, dn_cols, &act);
            if !dn_scale.is_empty() {
                down.iter_mut().for_each(|v| *v *= dn_scale[e]);
            }
            let w = probs[e] / wsum * cfg.expert_scale;
            for (c, d) in cur.iter_mut().zip(down) {
                *c += w * d;
            }
        }
        let ffn_out = self.convolve(
            st,
            slot,
            &cur,
            &format!("{b}ffn_post_conv.weight"),
            None,
            2 * kv_dim + hidden,
            hidden,
        )?;
        for (c, f) in ffn_inp.iter_mut().zip(&ffn_out) {
            *c += f;
        }
        Ok(ffn_inp)
    }

    fn embedding(&self, token: u32) -> Result<Vec<f32>> {
        let (t, raw) = self.tensor("token_embd.weight")?;
        let cols = t.dims[0] as usize;
        let rb = t.ty.row_bytes(cols);
        let mut out = vec![0.0f32; cols];
        crate::quant::dequant(
            t.ty,
            &raw[token as usize * rb..(token as usize + 1) * rb],
            &mut out,
        );
        Ok(out)
    }

    /// Run one token through the whole network, returning unpadded logits.
    pub fn forward(&self, st: &mut State, token: u32) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let pos = st.pos;
        let emb = self.embedding(token)?;
        let ln: &[f32] = bytemuck::cast_slice(self.tensor("token_embd_norm.weight")?.1);
        let mut cur = rms_norm(&emb, ln, cfg.eps);

        let tau = 1.0 + cfg.log_alpha * (((pos + 1) / cfg.log_floor).max(1) as f32).ln();
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        for il in 0..cfg.loop_start {
            cur = self.layer(st, il, il, il, &cur, &mut blocks, None, tau)?;
        }
        let prelude = cur.clone();
        let prelude_blocks = blocks.clone();
        // The selected recurrent state is the weak reference for LoopCD. Capture it before
        // later passes overwrite `cur` and its residual bank.
        let mut reference: Option<(Vec<f32>, Vec<Vec<f32>>)> = None;
        for pass in 0..cfg.loop_passes {
            let (bank, bank_out) = if pass > 0 {
                (
                    blocks[prelude_blocks.len() + 1..].to_vec(),
                    Some(cur.clone()),
                )
            } else {
                (Vec::new(), None)
            };
            if pass > 0 {
                cur = prelude.clone();
                blocks = prelude_blocks.clone();
            }
            for ip in cfg.loop_start..cfg.loop_end {
                let phys = if pass == 0 {
                    ip
                } else {
                    cfg.n_layers
                        + (pass - 1) * (cfg.loop_end - cfg.loop_start)
                        + (ip - cfg.loop_start)
                };
                let msg = if pass > 0 {
                    let mq = self.query(&format!("blk.{ip}.loop_msg_query.weight"))?;
                    let merged = mix(&bank, bank_out.as_ref().unwrap(), &mq, cfg.hidden, cfg.eps);
                    Some((merged, ip))
                } else {
                    None
                };
                cur = self.layer(
                    st,
                    ip,
                    ip,
                    phys,
                    &cur,
                    &mut blocks,
                    msg.as_ref().map(|(m, p)| (m.as_slice(), *p)),
                    tau,
                )?;
            }
            if pass == self.loopcd.ref_pass {
                reference = Some((cur.clone(), blocks.clone()));
            }
        }
        // LoopCD-Hidden contrasts before the coda layers so they still run exactly once.
        if self.loopcd.mode == LoopCdMode::Hidden {
            let (ref_cur, ref_blocks) = reference
                .as_ref()
                .expect("LoopCD reference pass was not captured");
            let w = self.loopcd.omega;
            cur = extrapolate(&cur, ref_cur, w);
            let mut guided: Vec<Vec<f32>> = blocks
                .iter()
                .zip(ref_blocks)
                .map(|(s, r)| extrapolate(s, r, w))
                .collect();
            // Blocks added after the reference pass belong to later coda layers; keep them.
            let n = guided.len();
            guided.extend(blocks.iter().skip(n).cloned());
            blocks = guided;
        }

        let logits = if self.loopcd.mode == LoopCdMode::Logits {
            let (ref_cur, ref_blocks) = reference
                .as_ref()
                .expect("LoopCD reference pass was not captured");
            // The weak pass must not append KV or shift conv history in the real cache.
            let mut weak_st = branch_state(st, cfg.loop_end..cfg.n_layers);
            let mut weak_cur = ref_cur.clone();
            let mut weak_blocks = ref_blocks.clone();
            for il in cfg.loop_end..cfg.n_layers {
                weak_cur = self.layer(
                    &mut weak_st,
                    il,
                    il,
                    il,
                    &weak_cur,
                    &mut weak_blocks,
                    None,
                    tau,
                )?;
            }
            let weak = self.coda(&weak_blocks, &weak_cur)?;
            for il in cfg.loop_end..cfg.n_layers {
                cur = self.layer(st, il, il, il, &cur, &mut blocks, None, tau)?;
            }
            let strong = self.coda(&blocks, &cur)?;
            let w = self.loopcd.strength(&strong);
            if std::env::var_os("INKLING2_DEBUG").is_some() {
                eprintln!("loopcd: omega={w:.3}");
            }
            extrapolate(&strong, &weak, w)
        } else {
            for il in cfg.loop_end..cfg.n_layers {
                cur = self.layer(st, il, il, il, &cur, &mut blocks, None, tau)?;
            }
            self.coda(&blocks, &cur)?
        };
        st.pos += 1;
        Ok(logits)
    }
}

/// Token-by-token generation with the shared sampler. Prefill and decode both run through
/// `forward`, so the reported decode rate includes all layer work.
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
        top_k: if temp <= 0.0 { 1 } else { 20 },
        top_p: 0.95,
        seed: 0x5eed,
    });
    let mut logits = Vec::new();
    for &t in prompt {
        logits = model.forward(st, t)?;
    }
    let debug = std::env::var_os("INKLING2_DEBUG").is_some();
    for step in 0..max_new {
        if debug {
            let mut top: Vec<(f32, usize)> = logits
                .iter()
                .copied()
                .enumerate()
                .map(|(i, v)| (v, i))
                .collect();
            top.sort_by(|a, b| b.0.total_cmp(&a.0));
            eprintln!(
                "STEP {step} TOP: {}",
                top[..8]
                    .iter()
                    .map(|(v, i)| format!("{i}:{v:.2}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        let next = sampler.sample(&logits);
        if debug {
            eprint!("{next} ");
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
    fn loopcd_zero_strength_is_identity() {
        let strong = [1.0f32, -2.5, 3.25, 0.0];
        let weak = [0.5f32, 4.0, -1.0, 9.0];
        assert_eq!(extrapolate(&strong, &weak, 0.0), strong);
        // Both variants reduce to the unguided logits at ω = 0.
        assert_eq!(extrapolate(&strong, &weak, 0.0).len(), strong.len());
        let w = 0.25;
        let guided = extrapolate(&strong, &weak, w);
        for (i, (&s, &r)) in strong.iter().zip(&weak).enumerate() {
            assert!((guided[i] - (s + w * (s - r))).abs() < 1e-6);
        }
    }

    #[test]
    fn loopcd_adaptive_strength_from_top_two_margin() {
        let fixed = LoopCd {
            mode: LoopCdMode::Logits,
            omega: 0.25,
            adaptive: false,
            ref_pass: 0,
        };
        assert_eq!(fixed.strength(&[5.0, 4.0, 0.0]), 0.25);

        let adaptive = LoopCd {
            adaptive: true,
            omega: 1.0,
            ..fixed
        };
        // exp(5), exp(4), exp(0) shifted by the max -> p1 ≈ 0.7275, p2 ≈ 0.2677.
        let w = adaptive.strength(&[5.0, 4.0, 0.0]);
        assert!((w - 0.54018).abs() < 1e-3, "adaptive ω = {w}");
        assert!((0.0..=adaptive.omega).contains(&w));
        // A settled prediction (dominant top-1) gets almost no guidance.
        let settled = adaptive.strength(&[20.0, 0.0, -1.0]);
        assert!(settled < 1e-6, "settled ω = {settled}");
        assert_eq!(adaptive.strength(&[]), 0.0);
    }

    #[test]
    fn apex_balanced_inventory() {
        let path =
            Path::new("/mnt/models/inkling2-mini-apex-gguf/Inkling-2-Mini-APEX-Balanced.gguf");
        if !path.exists() {
            eprintln!("skipping: APEX GGUF not present");
            return;
        }
        let g = Gguf::open(path).unwrap();
        validate(&g).unwrap();
    }
}
