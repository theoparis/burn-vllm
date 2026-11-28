//! Hy4-flash DSA pooled indexer / top-2048 cell selection.
//!
//! Transcribed from llama.cpp `hy-v4-flash.cpp` (`build_pool_select`,
//! `build_indexer_rope`) and `llama-kv-cache.cpp` (`set_input_kpool`):
//!
//! - `iq`  = RoPE(q_lora_residual @ indexer.attn_q_b), `[heads, key_len]`
//! - `ik`  = rms_norm(hidden @ indexer.attn_k)
//! - `iv`, `ig` = `ik @ indexer_compressor_{value,gate}`, cached as one
//!   `[2 * key_len]` row per token (first half value, second half gate).
//! - each pool of `kpool` consecutive rows is gated-softmax pooled per channel,
//!   rms-normalized and RoPE'd at its first member's position;
//! - `score[p] = sum_h w[h] * relu(dot(iq[h], pooled_k[p]))`, with `w` from
//!   `indexer.proj` scaled by `1/sqrt(key_len * heads)`;
//! - the best `top_k/kpool - 1` pools are expanded to their cells and the
//!   current token's (possibly partial) pool is appended as the tail.
//!
//! The cache is assumed to hold one cell per token in sequence order; M-RoPE
//! image positions are not represented by this CPU path, so cell `i` has
//! position `pos - (last_cell - i)`.

use crate::gguf::{Gguf, TensorInfo};
use crate::hy4::Config;
use crate::quant::dequant;
use anyhow::{Result, ensure};

/// Dequantized indexer weights for a single sparse block. Rows are stored
/// contiguously in ggml order (`dims[0]` is the row length).
#[derive(Debug, Clone)]
pub struct Weights {
    /// `[heads * key_len, q_lora_rank]` (Q8_0 in the GGUF).
    pub q_b: Vec<f32>,
    /// `[key_len, hidden]` (Q8_0 in the GGUF).
    pub k: Vec<f32>,
    /// `[key_len]` (F32).
    pub k_norm: Vec<f32>,
    /// `[key_len, key_len]` (Q8_0 in the GGUF).
    pub value: Vec<f32>,
    /// `[key_len, key_len]` (Q8_0 in the GGUF).
    pub gate: Vec<f32>,
    /// `[key_len, kpool]` (F32), gate bias per pool member.
    pub ape: Vec<f32>,
    /// `[key_len]` (F32).
    pub comp_norm: Vec<f32>,
    /// `[heads, hidden]` (F32).
    pub proj: Vec<f32>,
}

fn load_vec(g: &Gguf, name: &str, len: usize) -> Result<Vec<f32>> {
    let t = g.tensor(name)?;
    ensure!(
        t.numel() as usize == len,
        "{name}: expected {len} elements, got {}",
        t.numel()
    );
    let mut out = vec![0.0f32; len];
    dequant(t.ty, g.bytes(t), &mut out);
    Ok(out)
}

fn load_mat(g: &Gguf, name: &str, rows: usize, cols: usize) -> Result<Vec<f32>> {
    let t: &TensorInfo = g.tensor(name)?;
    ensure!(
        t.dims.len() >= 2 && t.dims[0] as usize == cols && t.dims[1] as usize == rows,
        "{name}: expected [{cols}, {rows}], got {:?}",
        t.dims
    );
    let rb = t.ty.row_bytes(cols);
    let raw = g.bytes(t);
    ensure!(raw.len() >= rows * rb, "{name}: truncated tensor data");
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        dequant(
            t.ty,
            &raw[r * rb..(r + 1) * rb],
            &mut out[r * cols..(r + 1) * cols],
        );
    }
    Ok(out)
}

impl Weights {
    /// Load and dequantize all indexer tensors of block `b` (e.g. `"blk.1."`).
    pub fn load(g: &Gguf, b: &str, p: &Params) -> Result<Self> {
        let n = p.key_len;
        let h = p.n_heads;
        Ok(Self {
            q_b: load_mat(
                g,
                &format!("{b}indexer.attn_q_b.weight"),
                h * n,
                p.q_lora_rank,
            )?,
            k: load_mat(g, &format!("{b}indexer.attn_k.weight"), n, p.hidden)?,
            k_norm: load_vec(g, &format!("{b}indexer.k_norm.weight"), n)?,
            value: load_mat(g, &format!("{b}indexer_compressor_value.weight"), n, n)?,
            gate: load_mat(g, &format!("{b}indexer_compressor_gate.weight"), n, n)?,
            ape: load_vec(g, &format!("{b}indexer_compressor_ape.weight"), n * p.kpool)?,
            comp_norm: load_vec(g, &format!("{b}indexer_compressor_norm.weight"), n)?,
            proj: load_mat(g, &format!("{b}indexer.proj.weight"), h, p.hidden)?,
        })
    }
}

/// Geometry of one sparse block's indexer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    pub hidden: usize,
    pub q_lora_rank: usize,
    pub n_heads: usize,
    pub key_len: usize,
    pub top_k: usize,
    pub kpool: usize,
    pub select_tail: bool,
    pub eps: f32,
    /// Number of rotated trailing channels of each indexer head (the model's
    /// `rope.dimension_count`, 64 for Hy4-flash).
    pub n_rot: usize,
}

impl Params {
    pub fn from_config(cfg: &Config) -> Result<Self> {
        let p = Self {
            hidden: cfg.hidden,
            q_lora_rank: cfg.q_lora_rank,
            n_heads: cfg.indexer_heads,
            key_len: cfg.indexer_key_len,
            top_k: cfg.indexer_top_k,
            kpool: cfg.indexer_kpool,
            select_tail: cfg.indexer_kpool_select_tail,
            eps: cfg.eps,
            n_rot: cfg.qk_rope_dim,
        };
        ensure!(p.n_heads > 0 && p.key_len > 0, "invalid indexer geometry");
        ensure!(p.kpool > 0 && p.top_k > 0, "invalid indexer top-k geometry");
        ensure!(p.n_rot <= p.key_len, "indexer rope wider than a head");
        Ok(p)
    }
}

/// Complete consecutive-cell pools and the current token's tail cells.
/// A tail member the token has not produced yet repeats the current cell,
/// matching `tail_idxs` in `set_input_kpool`.
pub fn pool_layout(n_cells: usize, kpool: usize) -> (Vec<Vec<usize>>, Vec<usize>) {
    assert!(kpool > 0);
    let n_pool = n_cells / kpool;
    let pools = (0..n_pool)
        .map(|g| (g * kpool..(g + 1) * kpool).collect())
        .collect();
    let c = n_cells.saturating_sub(1);
    let (g, r) = (c / kpool, c % kpool);
    let tail = (0..kpool)
        .map(|m| {
            let cell = g * kpool + m;
            if m <= r && cell < n_cells { cell } else { c }
        })
        .collect();
    (pools, tail)
}

/// Gated softmax pooling of one pool's `kpool` LID rows. Each row is
/// `[iv(key_len), ig(key_len)]`; `ape` is `[key_len, kpool]`.
pub fn pool_key(rows: &[f32], ape: &[f32], key_len: usize, kpool: usize) -> Vec<f32> {
    assert_eq!(rows.len(), kpool * 2 * key_len);
    assert!(ape.len() >= key_len * kpool);
    let mut pooled = vec![0.0f32; key_len];
    for c in 0..key_len {
        // soft_max runs over the pool axis, independently per channel
        let mut max_logit = f32::NEG_INFINITY;
        for m in 0..kpool {
            max_logit = max_logit.max(rows[m * 2 * key_len + key_len + c] + ape[c + m * key_len]);
        }
        let mut denom = 0.0f32;
        let mut probs = vec![0.0f32; kpool];
        for (m, p) in probs.iter_mut().enumerate() {
            *p = (rows[m * 2 * key_len + key_len + c] + ape[c + m * key_len] - max_logit).exp();
            denom += *p;
        }
        for (m, &p) in probs.iter().enumerate() {
            pooled[c] += (p / denom) * rows[m * 2 * key_len + c];
        }
    }
    pooled
}

fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Result<Vec<f32>> {
    ensure!(
        x.len() == cols,
        "matvec: expected {cols} inputs, got {}",
        x.len()
    );
    ensure!(w.len() == rows * cols, "matvec: weight shape mismatch");
    Ok((0..rows)
        .map(|r| {
            w[r * cols..(r + 1) * cols]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect())
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Result<Vec<f32>> {
    ensure!(x.len() == w.len(), "rms_norm: shape mismatch");
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let scale = 1.0 / (ss / x.len() as f32 + eps).sqrt();
    Ok(x.iter().zip(w).map(|(&v, &s)| v * scale * s).collect())
}

/// RoPE on the trailing `dim` channels, interleaved-pair (llama) convention.
/// This matches `build_indexer_rope`'s reorders: converting to NeoX halves,
/// rotating, then converting back is exactly a rotation of pairs `(2i, 2i+1)`.
fn rope_tail(v: &mut [f32], dim: usize, pos: usize, base: f64) {
    let start = v.len() - dim;
    for i in 0..dim / 2 {
        let theta = pos as f64 / base.powf(2.0 * i as f64 / dim as f64);
        let (c, s) = (theta.cos() as f32, theta.sin() as f32);
        let a = v[start + 2 * i];
        let b = v[start + 2 * i + 1];
        v[start + 2 * i] = a * c - b * s;
        v[start + 2 * i + 1] = a * s + b * c;
    }
}

/// Reusable per-block indexer: weights are dequantized once, then `select`
/// can be called for every token.
pub struct Indexer {
    pub p: Params,
    pub w: Weights,
}

impl Indexer {
    pub fn new(g: &Gguf, b: &str, cfg: &Config) -> Result<Self> {
        let p = Params::from_config(cfg)?;
        let w = Weights::load(g, b, &p)?;
        Ok(Self { p, w })
    }

    /// Build the `[iv, ig]` LID row of the current token from its attention-
    /// normed hidden state. Callers that keep a LID cache should append this
    /// row (or let `select` do it).
    pub fn lid_row(&self, cur: &[f32]) -> Result<Vec<f32>> {
        let n = self.p.key_len;
        let ik = rms_norm(
            &matvec(&self.w.k, n, self.p.hidden, cur)?,
            &self.w.k_norm,
            self.p.eps,
        )?;
        let mut row = matvec(&self.w.value, n, n, &ik)?;
        row.extend(matvec(&self.w.gate, n, n, &ik)?);
        Ok(row)
    }

    /// Select the `top_k` MLA cells the current token attends to: the best
    /// history pools expanded to cells, followed by the current tail.
    ///
    /// `lid` holds `[iv, ig]` rows in cell order, including the current token
    /// or not (it is appended automatically when absent). `pos` is the
    /// current token position; `rope_base` is the block's frequency base
    /// (sparse layers use `freq_base_swa` on this model).
    pub fn select(
        &self,
        cur: &[f32],
        qr: &[f32],
        pos: usize,
        lid: &[Vec<f32>],
        rope_base: f64,
    ) -> Result<Vec<usize>> {
        let p = &self.p;
        let n = p.key_len;
        ensure!(
            !lid.is_empty() || cur.len() == p.hidden,
            "missing current hidden state"
        );

        // query: q-lora residual -> heads, rope only the trailing n_rot channels
        let mut iq = matvec(&self.w.q_b, p.n_heads * n, p.q_lora_rank, qr)?;
        for h in 0..p.n_heads {
            rope_tail(&mut iq[h * n..(h + 1) * n], p.n_rot, pos, rope_base);
        }

        // the current token's LID row, appended when the caller passed history only
        let cur_row = self.lid_row(cur)?;
        let mut cells = lid.to_vec();
        let has_current = cells.last().is_some_and(|last| {
            last.len() == cur_row.len()
                && last
                    .iter()
                    .zip(&cur_row)
                    .fold(0.0f32, |mx, (&a, &b)| mx.max((a - b).abs()))
                    < 1e-3
        });
        if !has_current {
            cells.push(cur_row);
        }
        let n_cells = cells.len();
        let last = n_cells - 1;
        let (pools, tail) = pool_layout(n_cells, p.kpool);
        let cur_pool = last / p.kpool;

        // pooled keys: gated softmax over members, rms norm, rope at first cell
        let mut pooled = Vec::with_capacity(pools.len() * n);
        for members in &pools {
            let mut rows = Vec::with_capacity(p.kpool * 2 * n);
            for &cell in members {
                ensure!(
                    cells[cell].len() == 2 * n,
                    "LID row {cell}: expected {} values",
                    2 * n
                );
                rows.extend_from_slice(&cells[cell]);
            }
            let k = pool_key(&rows, &self.w.ape, n, p.kpool);
            let mut k = rms_norm(&k, &self.w.comp_norm, p.eps)?;
            // cell positions are contiguous in the single-stream CPU cache
            let p0 = pos as isize - (last as isize - members[0] as isize);
            rope_tail(&mut k, p.n_rot, p0.max(0) as usize, rope_base);
            pooled.extend(k);
        }

        // lightning indexer weights from the (unnormalized) hidden state
        let mut iw = matvec(&self.w.proj, p.n_heads, p.hidden, cur)?;
        let scale = 1.0 / ((n * p.n_heads) as f32).sqrt();
        for w in &mut iw {
            *w *= scale;
        }

        let mut scored: Vec<(f32, usize)> = Vec::with_capacity(pools.len());
        for (pi, key) in pooled.chunks_exact(n).enumerate() {
            let mut score = 0.0f32;
            for h in 0..p.n_heads {
                let q = &iq[h * n..(h + 1) * n];
                let dot: f32 = q.iter().zip(key).map(|(a, b)| a * b).sum();
                score += dot.max(0.0) * iw[h];
            }
            // the token only sees full pools before its own
            if pi == cur_pool {
                score = f32::NEG_INFINITY;
            }
            scored.push((score, pi));
        }

        let n_top = p
            .top_k
            .checked_div(p.kpool)
            .unwrap_or(0)
            .saturating_sub(usize::from(p.select_tail));
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        let mut out = Vec::with_capacity(p.top_k);
        for &(_, pi) in scored.iter().take(n_top.min(scored.len())) {
            out.extend_from_slice(&pools[pi]);
        }
        if p.select_tail {
            out.extend_from_slice(&tail);
        }
        // Unused picks are -inf-scored in the graph; repeating the current
        // cell keeps the gathered cells causal and the output shape fixed.
        out.resize(p.top_k, last);
        out.truncate(p.top_k);
        Ok(out)
    }
}

/// Convenience wrapper around [`Indexer::new`] + [`Indexer::select`] with the
/// API shape used by the forward pass. It re-dequantizes the block's weights;
/// keep an [`Indexer`] for per-token decode.
pub fn select(
    g: &Gguf,
    b: &str,
    cfg: &Config,
    cur: &[f32],
    qr: &[f32],
    pos: usize,
    lid: &[Vec<f32>],
    rope_base: f64,
) -> Result<Vec<usize>> {
    Indexer::new(g, b, cfg)?.select(cur, qr, pos, lid, rope_base)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Params {
        Params {
            hidden: 8,
            q_lora_rank: 4,
            n_heads: 1,
            key_len: 4,
            top_k: 8,
            kpool: 2,
            select_tail: true,
            eps: 1e-6,
            n_rot: 2,
        }
    }

    /// One head, two rotary channels; only channels 0 and 1 carry signal so
    /// the rope tail (2..4) has no effect on the scores.
    fn synthetic() -> Indexer {
        let p = params();
        let n = p.key_len;
        let mut q_b = vec![0.0f32; (p.n_heads * n) * p.q_lora_rank];
        q_b[1 * p.q_lora_rank + 0] = 1.0; // head 0 channel 1 <- qr[0]
        let mut k = vec![0.0f32; n * p.hidden];
        k[0] = 1.0; // ik[0] <- cur[0]
        k[1 * p.hidden + 1] = 1.0; // ik[1] <- cur[1]
        let value = (0..n)
            .flat_map(|r| (0..n).map(move |c| if r == c { 1.0 } else { 0.0 }))
            .collect();
        let gate = vec![0.0f32; n * n];
        let mut proj = vec![0.0f32; p.n_heads * p.hidden];
        proj[0] = 1.0;
        Indexer {
            p,
            w: Weights {
                q_b,
                k,
                k_norm: vec![1.0; n],
                value,
                gate,
                ape: vec![0.0; n * p.kpool],
                comp_norm: vec![1.0; n],
                proj,
            },
        }
    }

    fn lid_row(iv0: f32, iv1: f32) -> Vec<f32> {
        let mut row = vec![0.0f32; 8];
        row[0] = iv0;
        row[1] = iv1;
        row
    }

    #[test]
    fn pool_layout_complete_pools_and_partial_tail() {
        let (pools, tail) = pool_layout(10, 2);
        assert_eq!(
            pools,
            vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7], vec![8, 9]]
        );
        // current cell 9 is the second member of its pool
        assert_eq!(tail, vec![8, 9]);
        let (pools, tail) = pool_layout(9, 2);
        assert_eq!(pools.last(), Some(&vec![6, 7]));
        // current cell 8 starts a new pool: the only other member repeats it
        assert_eq!(tail, vec![8, 8]);
        let (_, tail) = pool_layout(0, 4);
        assert_eq!(tail, vec![0; 4]);
    }

    #[test]
    fn gated_softmax_pools_members_per_channel() {
        // key_len=2, kpool=2; channel 0 prefers member 1, channel 1 member 0
        let rows = vec![
            1.0, 2.0, 0.0, 0.0, // cell 0: iv=[1,2], ig=[0,0]
            10.0, 20.0, 50.0, 0.0, // cell 1: iv=[10,20], ig=[50,0]
        ];
        let ape = vec![0.0; 4];
        let pooled = pool_key(&rows, &ape, 2, 2);
        assert!((pooled[0] - 10.0).abs() < 1e-3);
        // channel 1 has no gate signal, so its members get equal weight
        assert!((pooled[1] - 11.0).abs() < 1e-3);

        // a positive APE on member 0 flips both channels back to member 0
        let ape = vec![60.0, 60.0, 0.0, 0.0];
        let pooled = pool_key(&rows, &ape, 2, 2);
        assert!((pooled[0] - 1.0).abs() < 1e-3);
        assert!((pooled[1] - 2.0).abs() < 1e-3);
    }

    #[test]
    fn select_returns_top_k_cells_with_history_only() {
        let idx = synthetic();
        let mut cur = vec![0.0f32; idx.p.hidden];
        cur[0] = 1.0;
        cur[1] = 1.0; // ik = [1,1,0,0]
        let qr = vec![1.0, 0.0, 0.0, 0.0];

        // 4 history cells (2 complete pools) + current cell = 5 cells;
        // current pool 2 is masked, so the two visible pools are picked.
        let mut lid = vec![
            lid_row(1.0, 1.0),
            lid_row(1.0, 2.0), // pool 0, mean iv[1] = 1.5
            lid_row(1.0, 5.0),
            lid_row(1.0, 6.0), // pool 1, mean iv[1] = 5.5 (higher score)
        ];
        let sel = idx.select(&cur, &qr, 4, &lid, 1e6).unwrap();
        assert_eq!(sel.len(), idx.p.top_k);
        assert_eq!(&sel[..6], &[2, 3, 0, 1, 4, 4]);
        assert!(sel.iter().all(|&c| c < 5));

        // passing the current row explicitly must give the same selection
        let row = idx.lid_row(&cur).unwrap();
        lid.push(row);
        assert_eq!(idx.select(&cur, &qr, 4, &lid, 1e6).unwrap(), sel);
    }

    #[test]
    fn select_full_2048_shape_with_synthetic_cache() {
        // Scale the geometry to the real model's top-k while keeping the
        // weights tiny (one head, one rotary pair).
        let mut p = params();
        p.top_k = 2048;
        p.kpool = 4;
        p.n_rot = 0;
        let n = p.key_len;
        let idx = Indexer {
            p,
            w: Weights {
                q_b: vec![0.0; n * p.q_lora_rank],
                k: vec![0.0; n * p.hidden],
                k_norm: vec![1.0; n],
                value: vec![0.0; n * n],
                gate: vec![0.0; n * n],
                ape: vec![0.0; n * p.kpool],
                comp_norm: vec![1.0; n],
                proj: vec![0.0; p.n_heads * p.hidden],
            },
        };
        let cur = vec![0.0f32; p.hidden];
        let qr = vec![0.0f32; p.q_lora_rank];
        let lid = vec![lid_row(0.0, 0.0); 2048];

        let sel = idx.select(&cur, &qr, 2047, &lid, 1e6).unwrap();
        assert_eq!(sel.len(), 2048);
        // all scores are equal -> 511 lowest-index pools, then the tail
        assert_eq!(&sel[..8], &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(&sel[2044..2048], &[2044, 2045, 2046, 2047]);
        assert!(sel.iter().all(|&c| c < 2048));
    }
}
