//! Hy4-flash Engram/PLE n-gram delta, transcribed from llama.cpp `hy-v4-flash.cpp`.
//!
//! This module is deliberately standalone: it does not touch the common forward pass.
//! `Engram::delta(layer, token, prev, streams)` returns the `[hidden * hc]` additive
//! delta (stream-major, `stream * hidden + j`) that the caller adds to the hc residual
//! streams before the rest of that block. Shard 1 supplies the projection weights and
//! the PLE metadata; shard 2 supplies the F8_E4M3 lookup table, which the shared GGUF
//! reader does not understand, so its header is parsed locally.
//!
//! `prev` holds up to `ngram_size - 1` preceding tokens in chronological order
//! (oldest first, last element immediately before `token`). Missing entries behave
//! like a negative position in the oracle: those heads are padded with the pad id.
//! An out-of-range `token` marks an embedding position and produces a zero delta.

#![allow(dead_code)] // staged API: the main forward will call it in a later change

use crate::gguf::Gguf;
use crate::hy4::Config;
use crate::quant::{GgmlType, dequant};
use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use rayon::prelude::*;
use std::{fs::File, path::Path};

const GGUF_F8_E4M3: u32 = 43;
const ENGRAM_EPS: f32 = 1.1920929e-7f32;
const DEAD: i64 = -1;

/// Minimal E4M3FN (OCP FP8, ggml type 43) to f32 conversion.
///
/// Matches llama.cpp's `ggml_f8_e4m3_to_fp32` (`ggml-impl.h`): subnormals are
/// `M * 2^-9`, normals are `(1 + M/8) * 2^(E-7)`, and only `0x7f`/`0xff` are NaN.
pub fn f8_e4m3_to_f32(b: u8) -> f32 {
    let ax = b & 0x7f;
    if ax == 0x7f {
        // OCP FP8 defines both 0x7F and 0xFF as NaN; llama.cpp returns its plain
        // NAN here, before applying the sign bit.
        return f32::NAN;
    }
    let exp = ((ax >> 3) & 0xf) as u32;
    let man = (ax & 7) as u32;
    let value = if exp == 0 {
        (man as f32) * (1.0f32 / 512.0f32)
    } else {
        f32::from_bits(((exp + 120) << 23) | (man << 20))
    };
    f32::from_bits(value.to_bits() | (((b & 0x80) as u32) << 24))
}

/// Header-only reader for shard 2's `per_layer_token_embd.weight`.
struct LookupTable {
    map: Mmap,
    offset: usize,
    rows: usize,
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("gguf: overflow")?;
        if end > self.buf.len() {
            bail!("gguf: truncated header");
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    /// Skip one metadata value; return it only when it is a numeric scalar.
    fn skip_value(&mut self, ty: u32) -> Result<Option<u64>> {
        Ok(match ty {
            0 => Some(self.take(1)?[0] as u64),
            1 => Some(self.take(1)?[0] as u64),
            2 => Some(u16::from_le_bytes(self.take(2)?.try_into()?) as u64),
            3 => Some(u16::from_le_bytes(self.take(2)?.try_into()?) as u64),
            4 => Some(self.u32()? as u64),
            5 => Some(self.u32()? as u64),
            6 => {
                self.take(4)?;
                None
            }
            7 => Some(self.take(1)?[0] as u64),
            8 => {
                self.string()?;
                None
            }
            9 => {
                let et = self.u32()?;
                let n = self.u64()? as usize;
                for _ in 0..n {
                    self.skip_value(et)?;
                }
                None
            }
            10 => Some(self.u64()?),
            11 => Some(self.u64()?),
            12 => {
                self.take(8)?;
                None
            }
            t => bail!("gguf: unknown value type {t}"),
        })
    }
}

impl LookupTable {
    fn open(path: &Path, head_dim: usize) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: the model file is treated as immutable for the process lifetime.
        let map = unsafe { Mmap::map(&file)? };
        let mut c = Cursor { buf: &map, pos: 0 };
        ensure!(c.take(4)? == b"GGUF", "shard 2: not a GGUF file");
        let version = c.u32()?;
        ensure!((2..=3).contains(&version), "shard 2: GGUF v{version}");
        let n_tensors = c.u64()? as usize;
        let n_kv = c.u64()? as usize;
        let mut align = 32usize;
        for _ in 0..n_kv {
            let k = c.string()?;
            let ty = c.u32()?;
            if let Some(v) = c.skip_value(ty)?
                && k == "general.alignment"
            {
                align = v as usize;
            }
        }
        let mut found = None;
        for _ in 0..n_tensors {
            let name = c.string()?;
            let nd = c.u32()? as usize;
            let mut dims = Vec::with_capacity(nd);
            for _ in 0..nd {
                dims.push(c.u64()?);
            }
            let ty = c.u32()?;
            let off = c.u64()? as usize;
            if name == "per_layer_token_embd.weight" {
                ensure!(
                    ty == GGUF_F8_E4M3,
                    "shard 2: expected F8_E4M3 ({}), got {ty}",
                    GGUF_F8_E4M3
                );
                ensure!(
                    dims.len() == 2 && dims[0] as usize == head_dim,
                    "shard 2: unexpected lookup dims {:?}",
                    dims
                );
                found = Some((off, dims[1] as usize));
            }
        }
        let (off, rows) = found.context("shard 2: missing per_layer_token_embd.weight")?;
        let offset = c.pos.div_ceil(align) * align + off;
        let nbytes = rows
            .checked_mul(head_dim)
            .context("lookup table too large")?;
        ensure!(
            offset.checked_add(nbytes).is_some_and(|e| e <= map.len()),
            "shard 2: lookup table out of bounds"
        );
        Ok(Self { map, offset, rows })
    }

    fn row(&self, id: i32, head_dim: usize) -> Result<&[u8]> {
        ensure!(
            id >= 0 && (id as usize) < self.rows,
            "engram row {id} out of range ({} rows)",
            self.rows
        );
        let start = self.offset + id as usize * head_dim;
        Ok(&self.map[start..start + head_dim])
    }
}

/// The four Engram modules of Hy4-flash, usable without changing the common forward.
pub struct Engram<'a> {
    g1: &'a Gguf,
    cfg: &'a Config,
    token_map: Vec<u32>,
    ple_layers: Vec<usize>,
    multipliers: Vec<u64>,
    offsets: Vec<u64>,
    primes: Vec<u64>,
    n_cols: usize,
    pad: u32,
    table: LookupTable,
}

/// Whether `layer` is a PLE layer, checkable before constructing an [`Engram`].
pub fn is_engram_layer(cfg: &Config, layer: usize) -> bool {
    cfg.ple_layers.contains(&layer)
}

impl<'a> Engram<'a> {
    /// `shard2` is the second GGUF, containing the F8_E4M3 lookup tensor.
    pub fn new(g1: &'a Gguf, shard2: &Path, cfg: &'a Config) -> Result<Self> {
        let p = "hy_v4_flash.";
        let u64s = |key: &str| -> Result<Vec<u64>> {
            g1.meta_array(&format!("{p}{key}"))?
                .iter()
                .map(|v| v.as_u64().context("engram: invalid metadata entry"))
                .collect()
        };
        let token_map = g1
            .meta_array(&format!("{p}engram.token_map"))?
            .iter()
            .map(|v| {
                let x = v.as_u64().context("engram: invalid token_map entry")?;
                ensure!(x <= u32::MAX as u64, "engram: token_map entry too large");
                Ok(x as u32)
            })
            .collect::<Result<Vec<_>>>()?;
        let ple_layers = cfg.ple_layers.clone();
        let n_e = ple_layers.len();
        let n_gram = cfg.ple_ngram;
        let n_head = cfg.ple_heads_per_ngram;
        ensure!(
            n_e > 0 && n_gram >= 2 && n_head > 0,
            "engram: missing or invalid PLE configuration"
        );
        let n_cols = (n_gram - 1) * n_head;
        let multipliers = u64s("ple.layer_multipliers")?;
        let offsets = u64s("ple.head_offsets")?;
        let primes = u64s("ple.head_vocab_sizes")?;
        ensure!(
            multipliers.len() == n_e * n_gram
                && offsets.len() == n_e * n_cols
                && primes.len() == n_e * n_cols,
            "engram: inconsistent PLE metadata lengths"
        );
        for (i, &l) in ple_layers.iter().enumerate() {
            ensure!(
                l < cfg.n_layers && (i == 0 || l > ple_layers[i - 1]),
                "engram: layer {l} is out of range or out of order"
            );
        }
        for c in 0..n_e * n_cols {
            ensure!(
                primes[c] >= 2
                    && offsets[c] <= i32::MAX as u64
                    && primes[c] <= i32::MAX as u64 - offsets[c],
                "engram: head {c} bucket range out of range"
            );
        }
        let pad = token_map
            .get(cfg.engram_pad)
            .copied()
            .context("engram: pad token id out of range")?;
        let table = LookupTable::open(shard2, cfg.ple_head_dim)?;
        ensure!(
            offsets
                .iter()
                .zip(&primes)
                .all(|(&o, &p)| o + p <= table.rows as u64),
            "engram: lookup table smaller than the head vocabularies"
        );
        Ok(Self {
            g1,
            cfg,
            token_map,
            ple_layers,
            multipliers,
            offsets,
            primes,
            n_cols,
            pad,
            table,
        })
    }

    /// Whether `layer` is one of the PLE/Engram layers.
    pub fn has_layer(&self, layer: usize) -> bool {
        self.ple_layers.contains(&layer)
    }

    /// Index of the Engram module attached to `layer`, if any.
    fn engram_index(&self, layer: usize) -> Option<usize> {
        self.ple_layers.iter().position(|&l| l == layer)
    }

    fn hash(&self, e: usize, token: u32, prev: &[u32]) -> (Vec<i32>, Vec<bool>) {
        hash_ids(
            self.cfg.ple_ngram,
            self.cfg.ple_heads_per_ngram,
            &self.multipliers[e * self.cfg.ple_ngram..(e + 1) * self.cfg.ple_ngram],
            &self.primes[e * self.n_cols..(e + 1) * self.n_cols],
            &self.offsets[e * self.n_cols..(e + 1) * self.n_cols],
            self.pad,
            &self.token_map,
            token,
            prev,
        )
    }

    fn f32_vec(&self, name: &str, nelem: usize) -> Result<&[f32]> {
        let t = self.g1.tensor(name)?;
        ensure!(
            t.ty == GgmlType::F32 && t.numel() as usize == nelem,
            "{name}: expected {nelem} F32 elements, got {:?}/{}",
            t.ty,
            t.numel()
        );
        Ok(bytemuck::cast_slice(self.g1.bytes(t)))
    }

    /// Additive Engram delta for one token, shaped `[hidden * hc]` (stream-major).
    pub fn delta(
        &self,
        layer: usize,
        token: u32,
        prev: &[u32],
        streams: &[f32],
    ) -> Result<Vec<f32>> {
        let e = self
            .engram_index(layer)
            .context("layer has no Engram/PLE module")?;
        let n = self.cfg.hidden;
        let hc = self.cfg.hc_count;
        ensure!(
            streams.len() == n * hc,
            "expected {} stream elements, got {}",
            n * hc,
            streams.len()
        );
        let (ids, valid) = self.hash(e, token, prev);

        // Gather and RMS-norm each looked-up head row, then apply its norm weight
        // and the n-gram validity mask.
        let d = self.cfg.ple_head_dim;
        let ln = self.f32_vec(
            &format!("blk.{layer}.engram_lookup_norm.weight"),
            d * ids.len(),
        )?;
        let mut rows = vec![0.0f32; d * ids.len()];
        for (h, id) in ids.iter().enumerate() {
            if !valid[h] {
                continue;
            }
            let base = h * d;
            let src = self.table.row(*id, d)?;
            for (j, b) in src.iter().enumerate() {
                rows[base + j] = f8_e4m3_to_f32(*b);
            }
            let scale = rms_scale(&rows[base..base + d], ENGRAM_EPS);
            for j in 0..d {
                rows[base + j] = rows[base + j] * scale * ln[base + j];
            }
        }

        // Fused key/value projection with its per-stream bias: [n*(hc+1)] rows.
        let kt = self.g1.tensor(&format!("blk.{layer}.engram_kv.weight"))?;
        let cols = kt.dims[0] as usize;
        let out_rows = kt.dims[1] as usize;
        ensure!(
            kt.ty == GgmlType::Q8_0 && cols == rows.len() && out_rows == n * (hc + 1),
            "blk.{layer}.engram_kv.weight: unexpected shape/type {:?}/{:?}",
            kt.ty,
            kt.dims
        );
        let raw = self.g1.bytes(kt);
        let rb = kt.ty.row_bytes(cols);
        let bias = self.f32_vec(&format!("blk.{layer}.engram_kv_b.weight"), out_rows)?;
        let mut kv = vec![0.0f32; out_rows];
        kv.par_iter_mut().enumerate().for_each(|(i, y)| {
            let mut scratch = vec![0.0f32; cols];
            dequant(GgmlType::Q8_0, &raw[i * rb..(i + 1) * rb], &mut scratch);
            *y = scratch.iter().zip(&rows).map(|(a, b)| a * b).sum::<f32>() + bias[i];
        });

        // Gate: dot(rms(x), rms(key), gate weight) per stream, signed sqrt, sigmoid.
        let mut xn = vec![0.0f32; n * hc];
        let mut kn = vec![0.0f32; n * hc];
        for s in 0..hc {
            let xs = rms_scale(&streams[s * n..(s + 1) * n], ENGRAM_EPS);
            let ks = rms_scale(&kv[s * n..(s + 1) * n], ENGRAM_EPS);
            for j in 0..n {
                xn[s * n + j] = streams[s * n + j] * xs;
                kn[s * n + j] = kv[s * n + j] * ks;
            }
        }
        let egate = self.f32_vec(&format!("blk.{layer}.engram_gate.weight"), n * hc)?;
        let inv_sqrt = 1.0 / (n as f32).sqrt();
        let mut gate = vec![0.0f32; hc];
        for s in 0..hc {
            let mut dot = 0.0f32;
            for j in 0..n {
                dot += xn[s * n + j] * kn[s * n + j] * egate[s * n + j];
            }
            dot *= inv_sqrt;
            let x = if dot < 0.0 {
                -dot.abs().max(1e-6).sqrt()
            } else {
                dot.abs().max(1e-6).sqrt()
            };
            gate[s] = sigmoid(x);
        }

        // Value is the last of the hc+1 projected streams.
        let value = &kv[n * hc..n * (hc + 1)];
        let vs = rms_scale(value, ENGRAM_EPS);
        let gamma = self.f32_vec(&format!("blk.{layer}.engram_pre_gamma.weight"), n)?;
        let mut signal = vec![0.0f32; n];
        for j in 0..n {
            signal[j] = value[j] * vs * gamma[j];
        }
        let hs = self.f32_vec(&format!("blk.{layer}.engram_hc_scale.weight"), n * hc)?;
        let any = if valid.iter().any(|v| *v) { 1.0 } else { 0.0 };
        let mut delta = vec![0.0f32; n * hc];
        for s in 0..hc {
            for j in 0..n {
                delta[s * n + j] = signal[j] * (gate[s] + hs[s * n + j]) * any;
            }
        }
        Ok(delta)
    }
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn rms_scale(x: &[f32], eps: f32) -> f32 {
    let sum: f32 = x.iter().map(|v| v * v).sum();
    1.0 / (sum / x.len() as f32 + eps).sqrt()
}

/// Host-side transcription of `llama_model_hy_v4_flash::engram_hash` for one token.
fn hash_ids(
    n_gram: usize,
    n_head: usize,
    mult: &[u64],
    primes: &[u64],
    offsets: &[u64],
    pad: u32,
    token_map: &[u32],
    token: u32,
    prev: &[u32],
) -> (Vec<i32>, Vec<bool>) {
    let n_cols = (n_gram - 1) * n_head;
    let cid = |t: u32| -> i64 {
        if (t as usize) < token_map.len() {
            token_map[t as usize] as i64
        } else {
            DEAD
        }
    };
    let c = cid(token);
    let dead = c == DEAD;
    let mut toks = vec![0u64; n_gram];
    toks[0] = (if dead { pad as i64 } else { c }) as u64;
    for s in 1..n_gram {
        // `prev` is chronological; the oracle indexes prev[n_prev - s].
        let idx = prev.len() as isize - s as isize;
        let c = if idx < 0 {
            DEAD
        } else {
            cid(prev[idx as usize])
        };
        toks[s] = (if c == DEAD { pad as i64 } else { c }) as u64;
    }
    let mut rolling = toks[0].wrapping_mul(mult[0]);
    let mut ids = vec![0i32; n_cols];
    for o in 1..n_gram {
        rolling ^= toks[o].wrapping_mul(mult[o]);
        for h in 0..n_head {
            let col = (o - 1) * n_head + h;
            ids[col] = (rolling % primes[col]) as i32 + offsets[col] as i32;
        }
    }
    (ids, vec![!dead; n_cols])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_matches_llamacpp_traits() {
        // Values taken from llama.cpp's ggml_f8_e4m3_to_fp32 (OCP E4M3FN).
        let cases: &[(u8, f32)] = &[
            (0x00, 0.0),
            (0x01, 1.0 / 512.0),
            (0x07, 7.0 / 512.0),
            (0x08, 1.0 / 64.0),
            (0x0f, 1.875 / 64.0),
            (0x38, 1.0),
            (0x3f, 1.875),
            (0x40, 2.0),
            (0x7e, 448.0), // max finite E4M3FN
            (0x80, -0.0),
            (0xc0, -2.0),
            (0xfe, -448.0),
        ];
        for &(b, want) in cases {
            let got = f8_e4m3_to_f32(b);
            assert_eq!(got.to_bits(), want.to_bits(), "byte {b:#04x}");
        }
        for b in [0x7f, 0xff] {
            // llama.cpp returns a plain NAN (no sign) for both encodings.
            assert_eq!(f8_e4m3_to_f32(b).to_bits(), f32::NAN.to_bits());
        }
    }

    #[test]
    fn hash_rolling_and_padding() {
        let map = [10, 20, 30, 40, 50, 60, 70];
        let mult = [3, 5, 7];
        let primes = [13, 17, 19, 23];
        let offsets = [100, 200, 300, 400];
        let (ids, valid) = hash_ids(3, 2, &mult, &primes, &offsets, 70, &map, 2, &[0, 1]);
        // toks = [30, 20, 10]; 30*3=90, 90^(20*5)=62, 62^(10*7)=120.
        assert_eq!(ids, [110, 211, 306, 405]);
        assert_eq!(valid, [true; 4]);

        // A token outside token_map is an embedding position: the entire delta is masked.
        let (_, valid) = hash_ids(3, 2, &mult, &primes, &offsets, 70, &map, 999, &[0, 1]);
        assert_eq!(valid, [false; 4]);

        // Short history pads the *older* position with the mapped pad id. `prev`
        // is chronological, so a full-length equivalent is [pad_token, recent].
        let (short, valid) = hash_ids(3, 2, &mult, &primes, &offsets, 70, &map, 1, &[0]);
        assert_eq!(valid, [true; 4]);
        assert_eq!(short, [101, 214, 309, 401]);
        let (padded, _) = hash_ids(3, 2, &mult, &primes, &offsets, 70, &map, 1, &[6, 0]);
        assert_eq!(short, padded);
    }
}
