//! GGML quant types: host-side dequantization (reference + embedding lookup) and
//! repacking into the 4-byte-aligned layouts the GPU kernels read.
//!
//! GPU layouts (all indexed by the global 32-element block index `b`):
//! - `F32`:   w0 = raw f32 bits.
//! - `Q8_0`:  w0 = int8 quants, 8 words per block; w1 = f16 scales, 2 per word.
//! - `MXFP4`: w0 = packed nibbles, 4 words per block; w1 = E8M0 exponents, 4 per word.
//! - `Q5_K`:  w0 = raw 176-byte superblocks (44 words), w1 unused.
//! - `Q6_K`:  w0 = raw 210-byte superblocks padded to 212 bytes (53 words), w1 unused.
//! - `BF16`:  w0 = raw bf16 pairs (2 per word), w1 unused.
//! - `Q4_K`:  w0 = raw 144-byte superblocks (36 words), w1 unused.
//! - `Q3_K`:  w0 = raw 110-byte superblocks padded to 112 bytes (28 words), w1 unused.
//! - `IQ3_S`: w0 = raw 110-byte superblocks padded to 112 bytes (28 words), w1 = codebook.
//! - `IQ2_S`: w0 = raw 82-byte superblocks padded to 84 bytes (21 words), w1 = codebook.
//! - `NVFP4`, `IQ2_XXS`, `IQ2_XS`, `IQ4_NL`: host-dequantized into F32 words like `Q5_1`.

use anyhow::{Result, bail};
use half::f16;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q8_0,
    Q5_1,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Iq3S,
    Iq3Xxs,
    Iq2S,
    Bf16,
    Mxfp4,
    Nvfp4,
    Iq2Xxs,
    Iq2Xs,
    Iq4Nl,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::F32,
            1 => Self::F16,
            8 => Self::Q8_0,
            11 => Self::Q3K,
            7 => Self::Q5_1,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            18 => Self::Iq3Xxs,
            21 => Self::Iq3S,
            22 => Self::Iq2S,
            30 => Self::Bf16,
            39 => Self::Mxfp4,
            40 => Self::Nvfp4,
            16 => Self::Iq2Xxs,
            17 => Self::Iq2Xs,
            20 => Self::Iq4Nl,
            _ => bail!("unsupported ggml type {v}"),
        })
    }

    /// (elements per block, bytes per block)
    pub fn block(self) -> (usize, usize) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::Bf16 => (1, 2),
            Self::Q8_0 => (32, 34),
            Self::Q5_1 => (32, 24),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Iq3S => (256, 110),
            Self::Iq3Xxs => (256, 98),
            Self::Iq2S => (256, 82),
            Self::Mxfp4 => (32, 17),
            Self::Nvfp4 => (64, 36),
            Self::Iq2Xxs => (256, 66),
            Self::Iq2Xs => (256, 74),
            Self::Iq4Nl => (32, 18),
        }
    }

    pub fn row_bytes(self, n: usize) -> usize {
        let (be, bb) = self.block();
        assert!(n % be == 0, "{self:?}: {n} not a multiple of {be}");
        n / be * bb
    }

    /// The comptime format id the kernels dispatch on.
    pub fn gpu_fmt(self) -> u32 {
        match self {
            Self::F32 | Self::F16 => fmt::F32,
            // Host-dequantized into F32 words by `repack`; no dedicated GPU kernel.
            Self::Q5_1 | Self::Nvfp4 | Self::Iq2Xxs | Self::Iq2Xs | Self::Iq4Nl => fmt::F32,
            Self::Q8_0 => fmt::Q8_0,
            Self::Mxfp4 => fmt::MXFP4,
            Self::Q5K => fmt::Q5K,
            Self::Q6K => fmt::Q6K,
            Self::Bf16 => fmt::BF16,
            Self::Q4K => fmt::Q4K,
            Self::Q3K => fmt::Q3K,
            Self::Iq3S => fmt::IQ3S,
            Self::Iq3Xxs => fmt::IQ3XXS,
            Self::Iq2S => fmt::IQ2S,
        }
    }

    /// Whether the GPU layout keeps a per-tensor codebook in w1 (not poolable per expert).
    pub fn has_codebook(self) -> bool {
        matches!(self, Self::Iq3S | Self::Iq3Xxs | Self::Iq2S)
    }

    /// The per-tensor codebook (GPU w1 contents) of a codebook format.
    pub fn codebook(self) -> Option<Vec<u32>> {
        use crate::iq_grids::*;
        match self {
            Self::Iq3S => Some(IQ3S_GRID.to_vec()),
            Self::Iq3Xxs => Some(IQ3XXS_GRID.to_vec()),
            Self::Iq2S => Some(
                IQ2S_GRID
                    .iter()
                    .flat_map(|g| [*g as u32, (*g >> 32) as u32])
                    .collect(),
            ),
            _ => None,
        }
    }
}

/// Kernel format ids.
pub mod fmt {
    pub const F32: u32 = 0;
    pub const Q8_0: u32 = 1;
    pub const MXFP4: u32 = 2;
    pub const Q5K: u32 = 3;
    pub const Q6K: u32 = 4;
    pub const BF16: u32 = 5;
    pub const Q4K: u32 = 6;
    pub const Q3K: u32 = 7;
    pub const IQ3S: u32 = 8;
    pub const IQ2S: u32 = 9;
    pub const IQ3XXS: u32 = 10;

    /// u32 words of w0 per 32-element block (K-quants: per 256-element superblock).
    pub fn w0_words(f: u32, numel: usize) -> usize {
        match f {
            F32 => numel,
            Q8_0 => numel / 32 * 8,
            MXFP4 => numel / 32 * 4,
            Q5K => numel / 256 * 44,
            Q6K => numel / 256 * 53,
            BF16 => numel / 2,
            Q4K => numel / 256 * 36,
            Q3K | IQ3S => numel / 256 * 28,
            IQ2S => numel / 256 * 21,
            IQ3XXS => numel / 256 * 25,
            _ => unreachable!(),
        }
    }

    pub fn w1_words(f: u32, numel: usize) -> usize {
        match f {
            Q8_0 => (numel / 32).div_ceil(2),
            MXFP4 => (numel / 32).div_ceil(4),
            IQ3S => 512,
            IQ2S => 2048,
            IQ3XXS => 256,
            _ => 1,
        }
    }
}

const MXFP4_VALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

fn e8m0_half(e: u8) -> f32 {
    // 2^(e - 128), matching ggml's GGML_E8M0_TO_FP32_HALF.
    if e < 2 {
        (e as f32 - 128.0).exp2()
    } else {
        f32::from_bits(((e as u32) - 1) << 23)
    }
}

/// UE4M3 scale (4 exponent bits, bias 7, 3 mantissa bits), halved to match the doubled
/// E2M1 table, as in ggml's `ggml_ue4m3_to_fp32`.
fn ue4m3_half(e: u8) -> f32 {
    if e == 0 || e == 0x7F {
        return 0.0;
    }
    let (exp, man) = ((e >> 3) as i32 & 0xF, (e & 7) as f32);
    let raw = if exp == 0 {
        man * (-9f32).exp2()
    } else {
        (1.0 + man / 8.0) * ((exp - 7) as f32).exp2()
    };
    raw * 0.5
}

const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Dequantize `out.len()` elements from `src` (which starts on a block boundary).
pub fn dequant(ty: GgmlType, src: &[u8], out: &mut [f32]) {
    let (be, bb) = ty.block();
    assert_eq!(out.len() % be, 0);
    match ty {
        GgmlType::F32 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(4)) {
                *o = f32::from_le_bytes(c.try_into().unwrap());
            }
        }
        GgmlType::F16 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(2)) {
                *o = f16::from_le_bytes(c.try_into().unwrap()).to_f32();
            }
        }
        GgmlType::Q8_0 => {
            for (y, b) in out.chunks_exact_mut(be).zip(src.chunks_exact(bb)) {
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                for j in 0..32 {
                    y[j] = d * (b[2 + j] as i8) as f32;
                }
            }
        }
        GgmlType::Q5_1 => {
            for (y, b) in out.chunks_exact_mut(be).zip(src.chunks_exact(bb)) {
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                let m = f16::from_le_bytes([b[2], b[3]]).to_f32();
                let (qh, qs) = (&b[4..8], &b[8..24]);
                for j in 0..32 {
                    let lo = ((qs[j / 2] >> (4 * (j % 2))) & 0xF) as u32;
                    let hi = (((qh[j / 8] >> (j % 8)) & 1) << 4) as u32;
                    y[j] = d * (lo | hi) as f32 - m;
                }
            }
        }
        GgmlType::Mxfp4 => {
            for (y, b) in out.chunks_exact_mut(be).zip(src.chunks_exact(bb)) {
                let d = e8m0_half(b[0]);
                for j in 0..16 {
                    y[j] = d * MXFP4_VALUES[(b[1 + j] & 0xF) as usize] as f32;
                    y[j + 16] = d * MXFP4_VALUES[(b[1 + j] >> 4) as usize] as f32;
                }
            }
        }
        GgmlType::Nvfp4 => {
            for (y, b) in out.chunks_exact_mut(be).zip(src.chunks_exact(bb)) {
                for s in 0..4 {
                    let d = ue4m3_half(b[s]);
                    let qs = &b[4 + 8 * s..4 + 8 * s + 8];
                    let ys = &mut y[16 * s..16 * s + 16];
                    for j in 0..8 {
                        ys[j] = d * MXFP4_VALUES[(qs[j] & 0xF) as usize] as f32;
                        ys[j + 8] = d * MXFP4_VALUES[(qs[j] >> 4) as usize] as f32;
                    }
                }
            }
        }
        GgmlType::Iq4Nl => {
            for (y, b) in out.chunks_exact_mut(be).zip(src.chunks_exact(bb)) {
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                for j in 0..16 {
                    y[j] = d * KVALUES_IQ4NL[(b[2 + j] & 0xF) as usize] as f32;
                    y[j + 16] = d * KVALUES_IQ4NL[(b[2 + j] >> 4) as usize] as f32;
                }
            }
        }
        GgmlType::Iq2Xxs => {
            for (y, b) in out.chunks_exact_mut(be).zip(src.chunks_exact(bb)) {
                iq2xxs_block(b, y);
            }
        }
        GgmlType::Iq2Xs => {
            for (y, b) in out.chunks_exact_mut(be).zip(src.chunks_exact(bb)) {
                iq2xs_block(b, y);
            }
        }
        GgmlType::Q5K => {
            for (y, b) in out.chunks_exact_mut(be).zip(src.chunks_exact(bb)) {
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
                let scales = &b[4..16];
                let qh = &b[16..48];
                let qs = &b[48..176];
                for s in 0..8 {
                    let (sc, m) = scale_min_k4(s, scales);
                    let (d1, m1) = (d * sc as f32, dmin * m as f32);
                    let ql = &qs[32 * (s / 2)..];
                    for l in 0..32 {
                        let lo = (ql[l] >> (4 * (s & 1))) & 0xF;
                        let hi = ((qh[l] >> s) & 1) << 4;
                        y[s * 32 + l] = d1 * (lo | hi) as f32 - m1;
                    }
                }
            }
        }
        GgmlType::Q6K => {
            for (y, b) in out.chunks_exact_mut(be).zip(src.chunks_exact(bb)) {
                let ql = &b[0..128];
                let qh = &b[128..192];
                let sc = &b[192..208];
                let d = f16::from_le_bytes([b[208], b[209]]).to_f32();
                for n in 0..2 {
                    for l in 0..32 {
                        let is = l / 16;
                        let (ql, qh, sc) = (&ql[64 * n..], qh[32 * n + l], &sc[8 * n..]);
                        let q1 = ((ql[l] & 0xF) | ((qh & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xF) | (((qh >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh >> 6) & 3) << 4)) as i32 - 32;
                        let y = &mut y[128 * n..];
                        y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                        y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                        y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                        y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                    }
                }
            }
        }
        GgmlType::Bf16 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(2)) {
                *o = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
            }
        }
        GgmlType::Q4K => {
            for (y, b) in out.chunks_exact_mut(be).zip(src.chunks_exact(bb)) {
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
                let scales = &b[4..16];
                let qs = &b[16..144];
                for s in 0..8 {
                    let (sc, m) = scale_min_k4(s, scales);
                    let (d1, m1) = (d * sc as f32, dmin * m as f32);
                    let ql = &qs[32 * (s / 2)..];
                    for l in 0..32 {
                        let lo = (ql[l] >> (4 * (s & 1))) & 0xF;
                        y[s * 32 + l] = d1 * lo as f32 - m1;
                    }
                }
            }
        }
        GgmlType::Q3K => {
            for (y, b) in out.chunks_exact_mut(be).zip(src.chunks_exact(bb)) {
                let hm = &b[0..32];
                let qs = &b[32..96];
                let sc = &b[96..108];
                let d = f16::from_le_bytes([b[108], b[109]]).to_f32();
                for s in 0..8 {
                    let (n, j) = (s / 4, s % 4);
                    for half in 0..2 {
                        let dl = d * q3k_scale(sc, s * 2 + half) as f32;
                        for l in 0..16 {
                            let li = half * 16 + l;
                            let q = (qs[32 * n + li] >> (2 * j)) & 3;
                            let hi = if hm[li] & (1 << (n * 4 + j)) != 0 {
                                0
                            } else {
                                4
                            };
                            y[s * 32 + li] = dl * (q as i32 - hi) as f32;
                        }
                    }
                }
            }
        }
        GgmlType::Iq3S => {
            for (y, b) in out.chunks_exact_mut(be).zip(src.chunks_exact(bb)) {
                iq3s_block(b, y);
            }
        }
        GgmlType::Iq3Xxs => {
            for (y, b) in out.chunks_exact_mut(be).zip(src.chunks_exact(bb)) {
                iq3xxs_block(b, y);
            }
        }
        GgmlType::Iq2S => {
            for (y, b) in out.chunks_exact_mut(be).zip(src.chunks_exact(bb)) {
                iq2s_block(b, y);
            }
        }
    }
}

/// ksigns_iq2xs: 7 sign bits plus an odd-parity bit in bit 7.
pub fn ksign(i: usize) -> u8 {
    (i | (((i.count_ones() as usize) & 1) << 7)) as u8
}

fn iq3xxs_block(b: &[u8], y: &mut [f32]) {
    use crate::iq_grids::IQ3XXS_GRID;
    let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
    let (qs, gas) = (&b[2..66], &b[66..98]);
    for s in 0..8 {
        let aux = u32::from_le_bytes([gas[4 * s], gas[4 * s + 1], gas[4 * s + 2], gas[4 * s + 3]]);
        let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
        for l in 0..4 {
            let sg = ksign(((aux >> (7 * l)) & 127) as usize);
            let g1 = IQ3XXS_GRID[qs[8 * s + 2 * l] as usize];
            let g2 = IQ3XXS_GRID[qs[8 * s + 2 * l + 1] as usize];
            for j in 0..4 {
                let s1 = if sg & (1 << j) != 0 { -1.0 } else { 1.0 };
                let s2 = if sg & (1 << (j + 4)) != 0 { -1.0 } else { 1.0 };
                y[s * 32 + l * 8 + j] = db * ((g1 >> (8 * j)) & 0xFF) as f32 * s1;
                y[s * 32 + l * 8 + 4 + j] = db * ((g2 >> (8 * j)) & 0xFF) as f32 * s2;
            }
        }
    }
}
/// Q3_K 6-bit scale `i` (0..16) with the -32 bias applied.
pub fn q3k_scale(sc: &[u8], i: usize) -> i32 {
    let (g, k) = (i / 4, i % 4);
    let low = (sc[(g % 2) * 4 + k] >> ((g / 2) * 4)) & 0xF;
    let hi = (sc[8 + k] >> (2 * g)) & 3;
    (low | (hi << 4)) as i32 - 32
}

fn iq3s_block(b: &[u8], y: &mut [f32]) {
    use crate::iq_grids::IQ3S_GRID;
    let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
    let (qs, qh, signs, scales) = (&b[2..66], &b[66..74], &b[74..106], &b[106..110]);
    for s in 0..8 {
        let db = d * (1 + 2 * ((scales[s / 2] >> (4 * (s % 2))) & 0xF) as u32) as f32;
        for l in 0..4 {
            let g1 =
                IQ3S_GRID[qs[8 * s + 2 * l] as usize | (((qh[s] as usize) << (8 - 2 * l)) & 256)];
            let g2 = IQ3S_GRID
                [qs[8 * s + 2 * l + 1] as usize | (((qh[s] as usize) << (7 - 2 * l)) & 256)];
            let sg = signs[4 * s + l];
            for j in 0..4 {
                let s1 = if sg & (1 << j) != 0 { -1.0 } else { 1.0 };
                let s2 = if sg & (1 << (j + 4)) != 0 { -1.0 } else { 1.0 };
                y[s * 32 + l * 8 + j] = db * ((g1 >> (8 * j)) & 0xFF) as f32 * s1;
                y[s * 32 + l * 8 + 4 + j] = db * ((g2 >> (8 * j)) & 0xFF) as f32 * s2;
            }
        }
    }
}

fn iq2s_block(b: &[u8], y: &mut [f32]) {
    use crate::iq_grids::IQ2S_GRID;
    let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
    let (qs, signs, qh, scales) = (&b[2..34], &b[34..66], &b[66..74], &b[74..82]);
    for s in 0..8 {
        let db = [
            d * (0.5 + (scales[s] & 0xF) as f32) * 0.25,
            d * (0.5 + (scales[s] >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let g = IQ2S_GRID[qs[4 * s + l] as usize | (((qh[s] as usize) << (8 - 2 * l)) & 0x300)];
            let sg = signs[4 * s + l];
            for j in 0..8 {
                let sgn = if sg & (1 << j) != 0 { -1.0 } else { 1.0 };
                y[s * 32 + l * 8 + j] = db[l / 2] * ((g >> (8 * j)) & 0xFF) as f32 * sgn;
            }
        }
    }
}

fn iq2xxs_block(b: &[u8], y: &mut [f32]) {
    use crate::iq_grids::IQ2XXS_GRID;
    let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
    let qs = &b[2..66];
    for ib in 0..8 {
        let q = &qs[8 * ib..8 * ib + 8];
        let aux1 = u32::from_le_bytes([q[4], q[5], q[6], q[7]]);
        let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
        for l in 0..4 {
            let grid = IQ2XXS_GRID[q[l] as usize].to_le_bytes();
            let sg = ksign(((aux1 >> (7 * l)) & 127) as usize);
            for j in 0..8 {
                let s = if sg & (1 << j) != 0 { -1.0 } else { 1.0 };
                y[ib * 32 + l * 8 + j] = db * grid[j] as f32 * s;
            }
        }
    }
}

fn iq2xs_block(b: &[u8], y: &mut [f32]) {
    use crate::iq_grids::IQ2XS_GRID;
    let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
    let (qs, scales) = (&b[2..66], &b[66..74]);
    for ib in 0..8 {
        let db = [
            d * (0.5 + (scales[ib] & 0xF) as f32) * 0.25,
            d * (0.5 + (scales[ib] >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let k = 4 * ib + l;
            let q = u16::from_le_bytes([qs[2 * k], qs[2 * k + 1]]);
            let grid = IQ2XS_GRID[(q & 511) as usize].to_le_bytes();
            let sg = ksign((q >> 9) as usize);
            for j in 0..8 {
                let s = if sg & (1 << j) != 0 { -1.0 } else { 1.0 };
                y[ib * 32 + l * 8 + j] = db[l / 2] * grid[j] as f32 * s;
            }
        }
    }
}

/// A tensor repacked into the GPU layout described in the module docs.
pub struct Packed {
    pub fmt: u32,
    pub w0: Vec<u32>,
    pub w1: Vec<u32>,
}

const PAR_CHUNK_BLOCKS: usize = 1 << 14;

/// Repack raw ggml bytes (a whole number of blocks) into the GPU layout.
pub fn repack(ty: GgmlType, src: &[u8]) -> Packed {
    let (be, bb) = ty.block();
    let nblocks = src.len() / bb;
    let numel = nblocks * be;
    let f = ty.gpu_fmt();
    let mut w0 = vec![0u32; fmt::w0_words(f, numel)];
    let mut w1 = vec![0u32; fmt::w1_words(f, numel)];
    match ty {
        GgmlType::F32 | GgmlType::Q5K => copy_words(&mut w0, src),
        GgmlType::Q5_1 | GgmlType::Nvfp4 | GgmlType::Iq2Xxs | GgmlType::Iq2Xs | GgmlType::Iq4Nl => {
            // Rare in practice (14 tensors in Inkling-2 APEX Balanced): dequantize on the
            // host and reuse the F32 GPU path instead of adding a dedicated kernel.
            let mut f = vec![0f32; numel];
            dequant(ty, src, &mut f);
            for (o, v) in w0.iter_mut().zip(f) {
                *o = v.to_bits();
            }
        }
        GgmlType::F16 => {
            for (o, c) in w0.iter_mut().zip(src.chunks_exact(2)) {
                *o = f16::from_le_bytes([c[0], c[1]]).to_f32().to_bits();
            }
        }
        GgmlType::Q8_0 => {
            let w0b: &mut [u8] = bytemuck::cast_slice_mut(&mut w0);
            let w1h: &mut [u16] = bytemuck::cast_slice_mut(&mut w1);
            if nblocks >= PAR_CHUNK_BLOCKS * 2 {
                w0b.par_chunks_mut(32 * PAR_CHUNK_BLOCKS)
                    .zip(w1h.par_chunks_mut(PAR_CHUNK_BLOCKS))
                    .zip(src.par_chunks(34 * PAR_CHUNK_BLOCKS))
                    .for_each(|((dst0, dst1), s)| {
                        for ((d0, d1), b) in dst0
                            .chunks_exact_mut(32)
                            .zip(dst1.iter_mut())
                            .zip(s.chunks_exact(34))
                        {
                            *d1 = u16::from_le_bytes([b[0], b[1]]);
                            d0.copy_from_slice(&b[2..34]);
                        }
                    });
            } else {
                for ((d0, d1), b) in w0b
                    .chunks_exact_mut(32)
                    .zip(w1h.iter_mut())
                    .zip(src.chunks_exact(34))
                {
                    *d1 = u16::from_le_bytes([b[0], b[1]]);
                    d0.copy_from_slice(&b[2..34]);
                }
            }
        }
        GgmlType::Mxfp4 => {
            let w0b: &mut [u8] = bytemuck::cast_slice_mut(&mut w0);
            let w1b: &mut [u8] = bytemuck::cast_slice_mut(&mut w1);
            if nblocks >= PAR_CHUNK_BLOCKS * 2 {
                w0b.par_chunks_mut(16 * PAR_CHUNK_BLOCKS)
                    .zip(w1b.par_chunks_mut(PAR_CHUNK_BLOCKS))
                    .zip(src.par_chunks(17 * PAR_CHUNK_BLOCKS))
                    .for_each(|((dst0, dst1), s)| {
                        for ((d0, d1), b) in dst0
                            .chunks_exact_mut(16)
                            .zip(dst1.iter_mut())
                            .zip(s.chunks_exact(17))
                        {
                            *d1 = b[0];
                            d0.copy_from_slice(&b[1..17]);
                        }
                    });
            } else {
                for ((d0, d1), b) in w0b
                    .chunks_exact_mut(16)
                    .zip(w1b.iter_mut())
                    .zip(src.chunks_exact(17))
                {
                    *d1 = b[0];
                    d0.copy_from_slice(&b[1..17]);
                }
            }
        }
        GgmlType::Q6K => {
            let w0b: &mut [u8] = bytemuck::cast_slice_mut(&mut w0);
            for (d, b) in w0b.chunks_exact_mut(212).zip(src.chunks_exact(210)) {
                d[..210].copy_from_slice(b);
            }
        }
        GgmlType::Bf16 => copy_words(&mut w0, src),
        GgmlType::Q4K => copy_words(&mut w0, src),
        GgmlType::Q3K | GgmlType::Iq3S => {
            let w0b: &mut [u8] = bytemuck::cast_slice_mut(&mut w0);
            for (d, b) in w0b.chunks_exact_mut(112).zip(src.chunks_exact(110)) {
                d[..110].copy_from_slice(b);
            }
            if ty == GgmlType::Iq3S {
                w1.copy_from_slice(&crate::iq_grids::IQ3S_GRID);
            }
        }
        GgmlType::Iq2S => {
            let w0b: &mut [u8] = bytemuck::cast_slice_mut(&mut w0);
            for (d, b) in w0b.chunks_exact_mut(84).zip(src.chunks_exact(82)) {
                d[..82].copy_from_slice(b);
            }
            for (i, g) in crate::iq_grids::IQ2S_GRID.iter().enumerate() {
                w1[2 * i] = *g as u32;
                w1[2 * i + 1] = (*g >> 32) as u32;
            }
        }
        GgmlType::Iq3Xxs => {
            let w0b: &mut [u8] = bytemuck::cast_slice_mut(&mut w0);
            for (d, b) in w0b.chunks_exact_mut(100).zip(src.chunks_exact(98)) {
                d[..98].copy_from_slice(b);
            }
            w1.copy_from_slice(&crate::iq_grids::IQ3XXS_GRID);
        }
    }
    Packed { fmt: f, w0, w1 }
}

fn copy_words(dst: &mut [u32], src: &[u8]) {
    #[cfg(target_endian = "little")]
    {
        let dst_b: &mut [u8] = bytemuck::cast_slice_mut(dst);
        dst_b.copy_from_slice(&src[..dst_b.len()]);
    }
    #[cfg(not(target_endian = "little"))]
    {
        dst.par_chunks_mut(1 << 16)
            .zip(src.par_chunks(4 << 16))
            .for_each(|(d, s)| {
                for (o, c) in d.iter_mut().zip(s.chunks_exact(4)) {
                    *o = u32::from_le_bytes(c.try_into().unwrap());
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Q5_1 block = f16 d, f16 m, 4 high-bit bytes, 16 low-nibble bytes (24 bytes / 32 values).
    #[test]
    fn q5_1_dequant() {
        let mut block = [0u8; 24];
        block[0..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        block[4] = 0b1010_0100; // high bits for values 0..8 (bit 0 clear, bit 2 set)
        block[8] = 0x10; // value 0 low nibble 0, value 1 low nibble 1
        let mut out = [0f32; 32];
        dequant(GgmlType::Q5_1, &block, &mut out);
        assert_eq!(out[0], 0.5 * 0.0 - 1.0);
        assert_eq!(out[1], 0.5 * 1.0 - 1.0);
        // Value 2's high bit is bit 2 of qh[0] (0xA5 → 1), its low nibble is 0.
        assert_eq!(out[2], 0.5 * 16.0 - 1.0);
        assert_eq!(GgmlType::from_u32(7).unwrap(), GgmlType::Q5_1);
        assert_eq!(GgmlType::Q5_1.block(), (32, 24));
    }

    #[test]
    fn e8m0() {
        assert_eq!(e8m0_half(128), 1.0);
        assert_eq!(e8m0_half(127), 0.5);
        assert_eq!(e8m0_half(130), 4.0);
    }
}
