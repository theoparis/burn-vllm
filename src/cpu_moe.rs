//! Fast SIMD (AVX2+FMA+F16C) CPU quantized dot products and MoE expert execution.
//!
//! Computes missed experts directly from the mmap'd GGUF bytes (zero repack, zero PCIe
//! weight transfer) in parallel across CPU cores while the GPU executes the shared expert
//! and VRAM-resident hit experts.

use crate::quant::GgmlType;
use half::f16;
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::LazyLock;

const MXFP4_VALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Precomputed E8M0 half-scale table: `E8M0_LUT[e] = 2^(e - 128)`.
const E8M0_LUT: [f32; 256] = {
    let mut t = [0.0f32; 256];
    // e < 2 maps to 0.0 (matching GPU kernel and avoiding x86 subnormal penalties).
    let mut e = 2usize;
    while e < 256 {
        t[e] = f32::from_bits(((e as u32) - 1) << 23);
        e += 1;
    }
    t
};

/// `SIGN_LUT[b]`: per-byte multipliers (0xFF = -1, 0x01 = +1) for the 8 sign bits of `b`,
/// applied with `_mm_sign_epi8`.
const SIGN_LUT: [u64; 256] = {
    let mut t = [0u64; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut v = 0u64;
        let mut j = 0;
        while j < 8 {
            let byte = if (i >> j) & 1 == 1 { 0xFFu64 } else { 0x01u64 };
            v |= byte << (8 * j);
            j += 1;
        }
        t[i] = v;
        i += 1;
    }
    t
};
#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline(always)]
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

/// Whether the fused AVX2+FMA+F16C kernels can be used on this host. Everything else falls
/// back to the portable scalar paths, so the same source builds on x86_64, aarch64, and any
/// other target without `cfg` at the call sites.
#[inline]
pub(crate) fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Activation vector quantized per 32-element block to symmetric `i8` (`[-127, 127]`),
/// plus per-block scale `d` and exact f32 block sum `bsum` (for Q5_K's `-m1 * sum(x)`).
pub struct QuantX {
    pub qs: Vec<i8>,
    pub d: Vec<f32>,
    pub bsum: Vec<f32>,
}

impl QuantX {
    pub fn new() -> Self {
        Self {
            qs: Vec::new(),
            d: Vec::new(),
            bsum: Vec::new(),
        }
    }

    pub fn quantize_into(&mut self, x: &[f32]) {
        debug_assert_eq!(x.len() % 32, 0);
        let nb = x.len() / 32;
        self.qs.resize(x.len(), 0);
        self.d.resize(nb, 0.0);
        self.bsum.resize(nb, 0.0);
        crate::vec::backend().quantize_into(x, &mut self.qs, &mut self.d, &mut self.bsum);
    }

    pub fn from_f32(x: &[f32]) -> Self {
        let mut q = Self::new();
        q.quantize_into(x);
        q
    }
}

/// Fast quantized dot product using Q8-quantized activation `qx` (with f32 fallback).
#[inline]
pub fn dot_q8(ty: GgmlType, src: &[u8], qx: &QuantX, x_f32: &[f32]) -> f32 {
    crate::vec::backend().dot_q8(ty, src, qx, x_f32)
}

/// Fused gate+up MXFP4 dots when `fused` is set and this host has the AVX2 kernels; otherwise
/// two portable `dot_q8` calls. Keeping the dispatch here means call sites need no `#[cfg]`,
/// and unsupported architectures get real scalar code instead of a dead `unreachable!()`.
#[inline]
fn gate_up(
    fused: bool,
    gate_ty: GgmlType,
    up_ty: GgmlType,
    g_row: &[u8],
    u_row: &[u8],
    qx: &QuantX,
    x: &[f32],
) -> (f32, f32) {
    #[cfg(target_arch = "x86_64")]
    if fused {
        // SAFETY: `fused` implies an AVX2/FMA/F16C host and MXFP4 gate/up weights.
        return unsafe { avx2::dot_mxfp4_q8_gu(g_row, u_row, qx) };
    }
    let _ = fused; // unused on the portable path
    (dot_q8(gate_ty, g_row, qx, x), dot_q8(up_ty, u_row, qx, x))
}

/// 2..=4 token version of `gate_up`, writing the results into `gs`/`us`.
#[inline]
fn gate_up_batch(
    fused: bool,
    gate_ty: GgmlType,
    up_ty: GgmlType,
    g_row: &[u8],
    u_row: &[u8],
    qxs: &[&QuantX],
    xs: &[&[f32]],
    gs: &mut [f32],
    us: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if fused {
        // SAFETY: see `gate_up`.
        unsafe { avx2::dot_mxfp4_q8_gu_batch(g_row, u_row, qxs, gs, us) };
        return;
    }
    let _ = fused;
    for i in 0..gs.len() {
        (gs[i], us[i]) = gate_up(false, gate_ty, up_ty, g_row, u_row, qxs[i], xs[i]);
    }
}

/// Batched Q5_K down-projection dots, with a portable scalar fallback.
#[inline]
fn down_batch_q5k(batch: bool, d_row: &[u8], qxs: &[&QuantX], xs: &[&[f32]], ys: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if batch {
        // SAFETY: `batch` implies an AVX2/FMA/F16C host and Q5_K down weights.
        unsafe { avx2::dot_q5k_q8_batch(d_row, qxs, ys) };
        return;
    }
    let _ = batch;
    for i in 0..ys.len() {
        ys[i] = dot_q8(GgmlType::Q5K, d_row, qxs[i], xs[i]);
    }
}

/// Exact f32 dot product (used as reference / fallback).
#[inline]
pub fn dot(ty: GgmlType, src: &[u8], x: &[f32]) -> f32 {
    let qx = QuantX::from_f32(x);
    dot_q8(ty, src, &qx, x)
}

/// Entry points used by `crate::vec::Neon`.
#[cfg(target_arch = "aarch64")]
pub(crate) fn neon_available() -> bool {
    neon::dotprod_available()
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn neon_dot_q8(ty: GgmlType, src: &[u8], qx: &QuantX) -> Option<f32> {
    match ty {
        GgmlType::Mxfp4 => neon::dot_mxfp4_q8(src, qx),
        GgmlType::Q5K => neon::dot_q5k_q8(src, qx),
        GgmlType::Q6K => neon::dot_q6k_q8(src, qx),
        GgmlType::Q4K => neon::dot_q4k_q8(src, qx),
        GgmlType::Q3K => neon::dot_q3k_q8(src, qx),
        GgmlType::Q8_0 => neon::dot_q8_0_q8(src, qx),
        _ => None,
    }
}

pub(crate) fn dot_scalar(ty: GgmlType, src: &[u8], x: &[f32]) -> f32 {
    let (be, _bb) = ty.block();
    debug_assert_eq!(x.len() % be, 0);
    let mut sum = 0.0f32;
    match ty {
        GgmlType::Mxfp4 => {
            for (xc, b) in x.chunks_exact(32).zip(src.chunks_exact(17)) {
                let d = E8M0_LUT[b[0] as usize];
                let mut s = 0.0f32;
                for j in 0..16 {
                    s += MXFP4_VALUES[(b[1 + j] & 0xF) as usize] as f32 * xc[j]
                        + MXFP4_VALUES[(b[1 + j] >> 4) as usize] as f32 * xc[j + 16];
                }
                sum += d * s;
            }
        }
        GgmlType::Q5K => {
            for (xc, b) in x.chunks_exact(256).zip(src.chunks_exact(176)) {
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
                let scales = &b[4..16];
                let qh = &b[16..48];
                let qs = &b[48..176];
                for s in 0..8 {
                    let (sc, m) = scale_min_k4(s, scales);
                    let (d1, m1) = (d * sc as f32, dmin * m as f32);
                    let ql = &qs[32 * (s / 2)..];
                    let xs = &xc[s * 32..(s + 1) * 32];
                    let mut sq = 0.0f32;
                    let mut sx = 0.0f32;
                    for l in 0..32 {
                        let lo = (ql[l] >> (4 * (s & 1))) & 0xF;
                        let hi = ((qh[l] >> s) & 1) << 4;
                        sq += (lo | hi) as f32 * xs[l];
                        sx += xs[l];
                    }
                    sum += d1 * sq - m1 * sx;
                }
            }
        }
        GgmlType::Q6K => {
            for (xc, b) in x.chunks_exact(256).zip(src.chunks_exact(210)) {
                let ql = &b[0..128];
                let qh = &b[128..192];
                let sc = &b[192..208];
                let d = f16::from_le_bytes([b[208], b[209]]).to_f32();
                let mut sb = 0.0f32;
                for n in 0..2 {
                    let (ql, qh, sc, xn) =
                        (&ql[64 * n..], &qh[32 * n..], &sc[8 * n..], &xc[128 * n..]);
                    for l in 0..32 {
                        let is = l / 16;
                        let qhv = qh[l];
                        let q1 = ((ql[l] & 0xF) | ((qhv & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xF) | (((qhv >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qhv >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qhv >> 6) & 3) << 4)) as i32 - 32;
                        sb += (sc[is] as i8) as f32 * (q1 as f32) * xn[l]
                            + (sc[is + 2] as i8) as f32 * (q2 as f32) * xn[l + 32]
                            + (sc[is + 4] as i8) as f32 * (q3 as f32) * xn[l + 64]
                            + (sc[is + 6] as i8) as f32 * (q4 as f32) * xn[l + 96];
                    }
                }
                sum += d * sb;
            }
        }
        GgmlType::Q8_0 => {
            for (xc, b) in x.chunks_exact(32).zip(src.chunks_exact(34)) {
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                let mut s = 0.0f32;
                for j in 0..32 {
                    s += (b[2 + j] as i8) as f32 * xc[j];
                }
                sum += d * s;
            }
        }
        GgmlType::F32 => {
            for (&xv, c) in x.iter().zip(src.chunks_exact(4)) {
                sum += f32::from_le_bytes(c.try_into().unwrap()) * xv;
            }
        }
        GgmlType::F16 => {
            for (&xv, c) in x.iter().zip(src.chunks_exact(2)) {
                sum += f16::from_le_bytes([c[0], c[1]]).to_f32() * xv;
            }
        }
        // Remaining formats (BF16, IQ*): dequantize the row, then plain f32 dot.
        GgmlType::Bf16 | GgmlType::Iq3S | GgmlType::Iq3Xxs | GgmlType::Iq2S => {
            let mut tmp = vec![0.0f32; x.len()];
            crate::quant::dequant(ty, src, &mut tmp);
            sum = tmp.iter().zip(x).map(|(a, b)| a * b).sum();
        }
        GgmlType::Q4K
        | GgmlType::Q3K
        | GgmlType::Q5_1
        | GgmlType::Nvfp4
        | GgmlType::Iq2Xxs
        | GgmlType::Iq2Xs
        | GgmlType::Iq4Nl => {
            let mut tmp = vec![0.0f32; x.len()];
            crate::quant::dequant(ty, src, &mut tmp);
            sum = tmp.iter().zip(x).map(|(a, b)| a * b).sum();
        }
    }
    sum
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn f16_bits_to_f32(bits: u16) -> f32 {
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits as i32)))
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn hsum256_ps(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let s = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(s);
        let s = _mm_add_ps(s, shuf);
        let shuf = _mm_movehl_ps(shuf, s);
        let s = _mm_add_ss(s, shuf);
        _mm_cvtss_f32(s)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn hmax256_ps(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let m = _mm_max_ps(lo, hi);
        let shuf = _mm_movehdup_ps(m);
        let m = _mm_max_ps(m, shuf);
        let shuf = _mm_movehl_ps(shuf, m);
        let m = _mm_max_ss(m, shuf);
        _mm_cvtss_f32(m)
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn quantize_into_avx2(x: &[f32], qs: &mut [i8], d: &mut [f32], bsum: &mut [f32]) {
        unsafe {
            let nb = x.len() / 32;
            let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
            let clamp_lo = _mm256_set1_epi8(-127);
            let inv127 = 1.0f32 / 127.0;
            let xp = x.as_ptr();
            let qp = qs.as_mut_ptr();
            let dp = d.as_mut_ptr();
            let bp = bsum.as_mut_ptr();
            for bi in 0..nb {
                let p = xp.add(bi * 32);
                let v0 = _mm256_loadu_ps(p);
                let v1 = _mm256_loadu_ps(p.add(8));
                let v2 = _mm256_loadu_ps(p.add(16));
                let v3 = _mm256_loadu_ps(p.add(24));

                let s01 = _mm256_add_ps(v0, v1);
                let s23 = _mm256_add_ps(v2, v3);
                *bp.add(bi) = hsum256_ps(_mm256_add_ps(s01, s23));

                let m01 = _mm256_max_ps(_mm256_and_ps(v0, abs_mask), _mm256_and_ps(v1, abs_mask));
                let m23 = _mm256_max_ps(_mm256_and_ps(v2, abs_mask), _mm256_and_ps(v3, abs_mask));
                let amax = hmax256_ps(_mm256_max_ps(m01, m23));
                let scale = amax * inv127;
                *dp.add(bi) = scale;
                let vinv = _mm256_set1_ps(if scale > 0.0 { 1.0 / scale } else { 0.0 });

                let i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, vinv));
                let i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, vinv));
                let i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, vinv));
                let i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, vinv));
                let p01 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i0, i1), 0xD8);
                let p23 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i2, i3), 0xD8);
                let packed = _mm256_permute4x64_epi64(_mm256_packs_epi16(p01, p23), 0xD8);
                let clamped = _mm256_max_epi8(packed, clamp_lo);
                _mm256_storeu_si256(qp.add(bi * 32) as *mut __m256i, clamped);
            }
        }
    }

    /// Signed 32 x `i8` dot product (where `a` is in `[-127, 127]` or `b` is in `[-127, 127]`):
    /// returns 8 `i32` partial sums.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn dot32_i8_i8(a: __m256i, b: __m256i, ones: __m256i) -> __m256i {
        let ax = _mm256_sign_epi8(a, a);
        let sy = _mm256_sign_epi8(b, a);
        let p16 = _mm256_maddubs_epi16(ax, sy);
        _mm256_madd_epi16(p16, ones)
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_mxfp4_q8(src: &[u8], qx: &QuantX) -> f32 {
        unsafe {
            let nb = qx.d.len();
            debug_assert!(src.len() >= nb * 17);
            let lut128 = _mm_loadu_si128(MXFP4_VALUES.as_ptr() as *const __m128i);
            let lut = _mm256_broadcastsi128_si256(lut128);
            let mask0f = _mm256_set1_epi8(0x0F);
            let ones = _mm256_set1_epi16(1);
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut bp = src.as_ptr();
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            let mut b = 0usize;
            while b + 2 <= nb {
                let d0 = _mm256_set1_ps(*E8M0_LUT.get_unchecked(*bp as usize) * *dp.add(b));
                let raw0 = _mm_loadu_si128(bp.add(1) as *const __m128i);
                let idx0 =
                    _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw0, 4), raw0), mask0f);
                let q0 = _mm256_shuffle_epi8(lut, idx0);
                let x0 = _mm256_loadu_si256(qp as *const __m256i);
                let p0 = _mm256_cvtepi32_ps(dot32_i8_i8(q0, x0, ones));
                acc0 = _mm256_fmadd_ps(d0, p0, acc0);

                let d1 =
                    _mm256_set1_ps(*E8M0_LUT.get_unchecked(*bp.add(17) as usize) * *dp.add(b + 1));
                let raw1 = _mm_loadu_si128(bp.add(18) as *const __m128i);
                let idx1 =
                    _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw1, 4), raw1), mask0f);
                let q1 = _mm256_shuffle_epi8(lut, idx1);
                let x1 = _mm256_loadu_si256(qp.add(32) as *const __m256i);
                let p1 = _mm256_cvtepi32_ps(dot32_i8_i8(q1, x1, ones));
                acc1 = _mm256_fmadd_ps(d1, p1, acc1);

                bp = bp.add(34);
                qp = qp.add(64);
                b += 2;
            }
            if b < nb {
                let d0 = _mm256_set1_ps(*E8M0_LUT.get_unchecked(*bp as usize) * *dp.add(b));
                let raw0 = _mm_loadu_si128(bp.add(1) as *const __m128i);
                let idx0 =
                    _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw0, 4), raw0), mask0f);
                let q0 = _mm256_shuffle_epi8(lut, idx0);
                let x0 = _mm256_loadu_si256(qp as *const __m256i);
                let p0 = _mm256_cvtepi32_ps(dot32_i8_i8(q0, x0, ones));
                acc0 = _mm256_fmadd_ps(d0, p0, acc0);
            }
            hsum256_ps(_mm256_add_ps(acc0, acc1))
        }
    }

    /// Fused gate+up MXFP4 dot product sharing one activation load per block.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_mxfp4_q8_gu(g_src: &[u8], u_src: &[u8], qx: &QuantX) -> (f32, f32) {
        unsafe {
            let nb = qx.d.len();
            debug_assert!(g_src.len() >= nb * 17 && u_src.len() >= nb * 17);
            let lut128 = _mm_loadu_si128(MXFP4_VALUES.as_ptr() as *const __m128i);
            let lut = _mm256_broadcastsi128_si256(lut128);
            let mask0f = _mm256_set1_epi8(0x0F);
            let ones = _mm256_set1_epi16(1);
            let mut g_acc = _mm256_setzero_ps();
            let mut u_acc = _mm256_setzero_ps();
            let mut gp = g_src.as_ptr();
            let mut up = u_src.as_ptr();
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            for b in 0..nb {
                let dx = *dp.add(b);
                let x = _mm256_loadu_si256(qp as *const __m256i);

                let dg = _mm256_set1_ps(*E8M0_LUT.get_unchecked(*gp as usize) * dx);
                let rawg = _mm_loadu_si128(gp.add(1) as *const __m128i);
                let idxg =
                    _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(rawg, 4), rawg), mask0f);
                let qg = _mm256_shuffle_epi8(lut, idxg);
                let pg = _mm256_cvtepi32_ps(dot32_i8_i8(qg, x, ones));
                g_acc = _mm256_fmadd_ps(dg, pg, g_acc);

                let du = _mm256_set1_ps(*E8M0_LUT.get_unchecked(*up as usize) * dx);
                let rawu = _mm_loadu_si128(up.add(1) as *const __m128i);
                let idxu =
                    _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(rawu, 4), rawu), mask0f);
                let qu = _mm256_shuffle_epi8(lut, idxu);
                let pu = _mm256_cvtepi32_ps(dot32_i8_i8(qu, x, ones));
                u_acc = _mm256_fmadd_ps(du, pu, u_acc);

                gp = gp.add(17);
                up = up.add(17);
                qp = qp.add(32);
            }
            (hsum256_ps(g_acc), hsum256_ps(u_acc))
        }
    }

    /// Multi-token fused gate+up MXFP4 dot product for 2..=4 activation vectors, decoding
    /// each weight row once.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_mxfp4_q8_gu_batch(
        g_src: &[u8],
        u_src: &[u8],
        qxs: &[&QuantX],
        g_out: &mut [f32],
        u_out: &mut [f32],
    ) {
        unsafe {
            let k = qxs.len();
            let nb = qxs[0].d.len();
            let lut128 = _mm_loadu_si128(MXFP4_VALUES.as_ptr() as *const __m128i);
            let lut = _mm256_broadcastsi128_si256(lut128);
            let mask0f = _mm256_set1_epi8(0x0F);
            let ones = _mm256_set1_epi16(1);
            let mut g_acc = [_mm256_setzero_ps(); 4];
            let mut u_acc = [_mm256_setzero_ps(); 4];
            let mut gp = g_src.as_ptr();
            let mut up = u_src.as_ptr();
            for b in 0..nb {
                let eg = *E8M0_LUT.get_unchecked(*gp as usize);
                let rawg = _mm_loadu_si128(gp.add(1) as *const __m128i);
                let idxg =
                    _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(rawg, 4), rawg), mask0f);
                let qg = _mm256_shuffle_epi8(lut, idxg);
                let axg = _mm256_sign_epi8(qg, qg);

                let eu = *E8M0_LUT.get_unchecked(*up as usize);
                let rawu = _mm_loadu_si128(up.add(1) as *const __m128i);
                let idxu =
                    _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(rawu, 4), rawu), mask0f);
                let qu = _mm256_shuffle_epi8(lut, idxu);
                let axu = _mm256_sign_epi8(qu, qu);

                for ti in 0..k {
                    let qx = qxs.get_unchecked(ti);
                    let dx = *qx.d.get_unchecked(b);
                    let x = _mm256_loadu_si256(qx.qs.as_ptr().add(b * 32) as *const __m256i);
                    let syg = _mm256_sign_epi8(x, qg);
                    let pg =
                        _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(axg, syg), ones));
                    g_acc[ti] = _mm256_fmadd_ps(_mm256_set1_ps(eg * dx), pg, g_acc[ti]);

                    let syu = _mm256_sign_epi8(x, qu);
                    let pu =
                        _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(axu, syu), ones));
                    u_acc[ti] = _mm256_fmadd_ps(_mm256_set1_ps(eu * dx), pu, u_acc[ti]);
                }
                gp = gp.add(17);
                up = up.add(17);
            }
            for ti in 0..k {
                g_out[ti] = hsum256_ps(g_acc[ti]);
                u_out[ti] = hsum256_ps(u_acc[ti]);
            }
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q5k_q8(src: &[u8], qx: &QuantX) -> f32 {
        unsafe {
            let nsb = qx.d.len() / 8;
            debug_assert!(src.len() >= nsb * 176);
            let mask0f = _mm256_set1_epi8(0x0F);
            let bit4 = _mm256_set1_epi8(0x10);
            let ones = _mm256_set1_epi16(1);
            let mut acc = _mm256_setzero_ps();
            let mut min_acc = 0.0f32;
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            let bp_sum = qx.bsum.as_ptr();
            for sb in 0..nsb {
                let b = src.as_ptr().add(sb * 176);
                let d = f16_bits_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
                let dmin = f16_bits_to_f32(u16::from_le_bytes([*b.add(2), *b.add(3)]));
                let scales = std::slice::from_raw_parts(b.add(4), 12);
                let qh = _mm256_loadu_si256(b.add(16) as *const __m256i);
                let qs = b.add(48);
                let mut hm = _mm256_set1_epi8(1);
                let base_bi = sb * 8;
                for pair in 0..4usize {
                    let ql = _mm256_loadu_si256(qs.add(pair * 32) as *const __m256i);

                    // Even subblock
                    {
                        let s = pair * 2;
                        let bi = base_bi + s;
                        let (sc, m) = scale_min_k4(s, scales);
                        let h =
                            _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, hm), hm), bit4);
                        let q = _mm256_or_si256(_mm256_and_si256(ql, mask0f), h);
                        let x = _mm256_loadu_si256(qp as *const __m256i);
                        let p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), ones);
                        let scale = _mm256_set1_ps(d * (sc as f32) * *dp.add(bi));
                        acc = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(p32), acc);
                        min_acc += dmin * (m as f32) * *bp_sum.add(bi);
                        qp = qp.add(32);
                        hm = _mm256_add_epi8(hm, hm);
                    }
                    // Odd subblock
                    {
                        let s = pair * 2 + 1;
                        let bi = base_bi + s;
                        let (sc, m) = scale_min_k4(s, scales);
                        let h =
                            _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, hm), hm), bit4);
                        let q =
                            _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql, 4), mask0f), h);
                        let x = _mm256_loadu_si256(qp as *const __m256i);
                        let p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), ones);
                        let scale = _mm256_set1_ps(d * (sc as f32) * *dp.add(bi));
                        acc = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(p32), acc);
                        min_acc += dmin * (m as f32) * *bp_sum.add(bi);
                        qp = qp.add(32);
                        hm = _mm256_add_epi8(hm, hm);
                    }
                }
            }
            hsum256_ps(acc) - min_acc
        }
    }

    /// Multi-pair Q5_K dot product for 2..=4 activation vectors, decoding each weight
    /// superblock once.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q5k_q8_batch(src: &[u8], qxs: &[&QuantX], out: &mut [f32]) {
        unsafe {
            let k = qxs.len();
            let nsb = qxs[0].d.len() / 8;
            let mask0f = _mm256_set1_epi8(0x0F);
            let bit4 = _mm256_set1_epi8(0x10);
            let ones = _mm256_set1_epi16(1);
            let mut acc = [_mm256_setzero_ps(); 4];
            let mut min_acc = [0.0f32; 4];
            for sb in 0..nsb {
                let b = src.as_ptr().add(sb * 176);
                let d = f16_bits_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
                let dmin = f16_bits_to_f32(u16::from_le_bytes([*b.add(2), *b.add(3)]));
                let scales = std::slice::from_raw_parts(b.add(4), 12);
                let qh = _mm256_loadu_si256(b.add(16) as *const __m256i);
                let qs = b.add(48);
                let mut hm = _mm256_set1_epi8(1);
                let base_bi = sb * 8;
                for pair in 0..4usize {
                    let ql = _mm256_loadu_si256(qs.add(pair * 32) as *const __m256i);
                    for odd in 0..2usize {
                        let s = pair * 2 + odd;
                        let bi = base_bi + s;
                        let (sc, m) = scale_min_k4(s, scales);
                        let h =
                            _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, hm), hm), bit4);
                        let q = if odd == 0 {
                            _mm256_or_si256(_mm256_and_si256(ql, mask0f), h)
                        } else {
                            _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql, 4), mask0f), h)
                        };
                        let dsc = d * (sc as f32);
                        let dmn = dmin * (m as f32);
                        for ti in 0..k {
                            let qx = qxs.get_unchecked(ti);
                            let x =
                                _mm256_loadu_si256(qx.qs.as_ptr().add(bi * 32) as *const __m256i);
                            let p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), ones);
                            let scale = _mm256_set1_ps(dsc * *qx.d.get_unchecked(bi));
                            acc[ti] = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(p32), acc[ti]);
                            min_acc[ti] += dmn * *qx.bsum.get_unchecked(bi);
                        }
                        hm = _mm256_add_epi8(hm, hm);
                    }
                }
            }
            for ti in 0..k {
                out[ti] = hsum256_ps(acc[ti]) - min_acc[ti];
            }
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q6k_q8(src: &[u8], qx: &QuantX) -> f32 {
        unsafe {
            let nsb = qx.d.len() / 8;
            debug_assert!(src.len() >= nsb * 210);
            let mask0f = _mm256_set1_epi8(0x0F);
            let mask03 = _mm256_set1_epi8(0x03);
            let bias32 = _mm256_set1_epi8(32);
            let ones = _mm256_set1_epi16(1);
            let mut acc = _mm256_setzero_ps();
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            for sb in 0..nsb {
                let b = src.as_ptr().add(sb * 210);
                let d = f16_bits_to_f32(u16::from_le_bytes([*b.add(208), *b.add(209)]));
                for n in 0..2usize {
                    let ql = b.add(64 * n);
                    let qh = b.add(128 + 32 * n);
                    let sc = b.add(192 + 8 * n) as *const i8;
                    let ql0 = _mm256_loadu_si256(ql as *const __m256i);
                    let ql1 = _mm256_loadu_si256(ql.add(32) as *const __m256i);
                    let qhv = _mm256_loadu_si256(qh as *const __m256i);

                    let q1 = _mm256_sub_epi8(
                        _mm256_or_si256(
                            _mm256_and_si256(ql0, mask0f),
                            _mm256_slli_epi16(_mm256_and_si256(qhv, mask03), 4),
                        ),
                        bias32,
                    );
                    let q2 = _mm256_sub_epi8(
                        _mm256_or_si256(
                            _mm256_and_si256(ql1, mask0f),
                            _mm256_slli_epi16(
                                _mm256_and_si256(_mm256_srli_epi16(qhv, 2), mask03),
                                4,
                            ),
                        ),
                        bias32,
                    );
                    let q3 = _mm256_sub_epi8(
                        _mm256_or_si256(
                            _mm256_and_si256(_mm256_srli_epi16(ql0, 4), mask0f),
                            _mm256_slli_epi16(
                                _mm256_and_si256(_mm256_srli_epi16(qhv, 4), mask03),
                                4,
                            ),
                        ),
                        bias32,
                    );
                    let q4 = _mm256_sub_epi8(
                        _mm256_or_si256(
                            _mm256_and_si256(_mm256_srli_epi16(ql1, 4), mask0f),
                            _mm256_slli_epi16(
                                _mm256_and_si256(_mm256_srli_epi16(qhv, 6), mask03),
                                4,
                            ),
                        ),
                        bias32,
                    );

                    // Each 32-element block has two 16-element sub-scales: lo 128-bit uses sc[2k],
                    // hi 128-bit uses sc[2k+1].
                    let base_bi = sb * 8 + n * 4;
                    let qs_arr = [q1, q2, q3, q4];
                    for (k, &qv) in qs_arr.iter().enumerate() {
                        let xv = _mm256_loadu_si256(qp.add(k * 32) as *const __m256i);
                        let p32 = dot32_i8_i8(qv, xv, ones);
                        let pf = _mm256_cvtepi32_ps(p32);
                        let dx = *dp.add(base_bi + k);
                        let slo = d * dx * (*sc.add(k * 2) as f32);
                        let shi = d * dx * (*sc.add(k * 2 + 1) as f32);
                        let sv = _mm256_set_m128(_mm_set1_ps(shi), _mm_set1_ps(slo));
                        acc = _mm256_fmadd_ps(sv, pf, acc);
                    }
                    qp = qp.add(128);
                }
            }
            hsum256_ps(acc)
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q8_0_q8(src: &[u8], qx: &QuantX) -> f32 {
        unsafe {
            let nb = qx.d.len();
            debug_assert!(src.len() >= nb * 34);
            let ones = _mm256_set1_epi16(1);
            let mut acc = _mm256_setzero_ps();
            let mut bp = src.as_ptr();
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            for b in 0..nb {
                let dw = f16_bits_to_f32(u16::from_le_bytes([*bp, *bp.add(1)]));
                let q = _mm256_loadu_si256(bp.add(2) as *const __m256i);
                let x = _mm256_loadu_si256(qp as *const __m256i);
                // x is in [-127, 127], so abs(x) never overflows i8.
                let p32 = dot32_i8_i8(x, q, ones);
                acc = _mm256_fmadd_ps(
                    _mm256_set1_ps(dw * *dp.add(b)),
                    _mm256_cvtepi32_ps(p32),
                    acc,
                );
                bp = bp.add(34);
                qp = qp.add(32);
            }
            hsum256_ps(acc)
        }
    }
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q4k_q8(src: &[u8], qx: &QuantX) -> f32 {
        unsafe {
            let nsb = qx.d.len() / 8;
            debug_assert!(src.len() >= nsb * 144);
            let mask0f = _mm256_set1_epi8(0x0F);
            let ones = _mm256_set1_epi16(1);
            let mut acc = _mm256_setzero_ps();
            let mut min_acc = 0.0f32;
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            let bp_sum = qx.bsum.as_ptr();
            for sb in 0..nsb {
                let b = src.as_ptr().add(sb * 144);
                let d = f16_bits_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
                let dmin = f16_bits_to_f32(u16::from_le_bytes([*b.add(2), *b.add(3)]));
                let scales = std::slice::from_raw_parts(b.add(4), 12);
                let qs = b.add(16);
                let base_bi = sb * 8;
                for pair in 0..4usize {
                    let ql = _mm256_loadu_si256(qs.add(pair * 32) as *const __m256i);
                    for odd in 0..2usize {
                        let s = pair * 2 + odd;
                        let bi = base_bi + s;
                        let (sc, m) = scale_min_k4(s, scales);
                        let q = if odd == 0 {
                            _mm256_and_si256(ql, mask0f)
                        } else {
                            _mm256_and_si256(_mm256_srli_epi16(ql, 4), mask0f)
                        };
                        let x = _mm256_loadu_si256(qp as *const __m256i);
                        let p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), ones);
                        let scale = _mm256_set1_ps(d * (sc as f32) * *dp.add(bi));
                        acc = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(p32), acc);
                        min_acc += dmin * (m as f32) * *bp_sum.add(bi);
                        qp = qp.add(32);
                    }
                }
            }
            hsum256_ps(acc) - min_acc
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q3k_q8(src: &[u8], qx: &QuantX) -> f32 {
        unsafe {
            let nsb = qx.d.len() / 8;
            debug_assert!(src.len() >= nsb * 110);
            let mask03 = _mm256_set1_epi8(0x03);
            let four = _mm256_set1_epi8(4);
            let zero = _mm256_setzero_si256();
            let ones = _mm256_set1_epi16(1);
            let mut acc = _mm256_setzero_ps();
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            for sb in 0..nsb {
                let b = src.as_ptr().add(sb * 110);
                let d = f16_bits_to_f32(u16::from_le_bytes([*b.add(108), *b.add(109)]));
                let sc = std::slice::from_raw_parts(b.add(96), 12);
                let hm = _mm256_loadu_si256(b as *const __m256i);
                let q0 = _mm256_loadu_si256(b.add(32) as *const __m256i);
                let q1 = _mm256_loadu_si256(b.add(64) as *const __m256i);
                for s in 0..8usize {
                    let (n, j) = (s / 4, s % 4);
                    let qn = if n == 0 { q0 } else { q1 };
                    let q2 = match j {
                        0 => _mm256_and_si256(qn, mask03),
                        1 => _mm256_and_si256(_mm256_srli_epi16(qn, 2), mask03),
                        2 => _mm256_and_si256(_mm256_srli_epi16(qn, 4), mask03),
                        _ => _mm256_and_si256(_mm256_srli_epi16(qn, 6), mask03),
                    };
                    let bit = _mm256_set1_epi8((1u32 << (n * 4 + j)) as i8);
                    let clear = _mm256_cmpeq_epi8(_mm256_and_si256(hm, bit), zero);
                    let q = _mm256_sub_epi8(q2, _mm256_and_si256(clear, four));
                    let x = _mm256_loadu_si256(qp as *const __m256i);
                    let p32 = dot32_i8_i8(q, x, ones);
                    let dx = d * *dp.add(sb * 8 + s);
                    let slo = dx * crate::quant::q3k_scale(sc, s * 2) as f32;
                    let shi = dx * crate::quant::q3k_scale(sc, s * 2 + 1) as f32;
                    let sv = _mm256_set_m128(_mm_set1_ps(shi), _mm_set1_ps(slo));
                    acc = _mm256_fmadd_ps(sv, _mm256_cvtepi32_ps(p32), acc);
                    qp = qp.add(32);
                }
            }
            hsum256_ps(acc)
        }
    }
    /// IQ3_S: 8 grid pairs per 32-block expanded to signed i8, then an int8 dot.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_iq3s_q8(src: &[u8], qx: &QuantX) -> f32 {
        use crate::iq_grids::IQ3S_GRID;
        unsafe {
            let nsb = qx.d.len() / 8;
            debug_assert!(src.len() >= nsb * 110);
            let ones = _mm256_set1_epi16(1);
            let mut acc = _mm256_setzero_ps();
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            for sb in 0..nsb {
                let b = src.as_ptr().add(sb * 110);
                let d = f16_bits_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
                for s in 0..8usize {
                    let qs = b.add(2 + 8 * s);
                    let qh = *b.add(66 + s) as usize;
                    let signs = b.add(74 + 4 * s);
                    let mut g = [0u64; 4];
                    let mut m = [0u64; 4];
                    for l in 0..4usize {
                        let i1 = *qs.add(2 * l) as usize | ((qh << (8 - 2 * l)) & 256);
                        let i2 = *qs.add(2 * l + 1) as usize | ((qh << (7 - 2 * l)) & 256);
                        g[l] = IQ3S_GRID[i1] as u64 | ((IQ3S_GRID[i2] as u64) << 32);
                        m[l] = SIGN_LUT[*signs.add(l) as usize];
                    }
                    let gv = _mm256_set_epi64x(g[3] as i64, g[2] as i64, g[1] as i64, g[0] as i64);
                    let mv = _mm256_set_epi64x(m[3] as i64, m[2] as i64, m[1] as i64, m[0] as i64);
                    let q = _mm256_sign_epi8(gv, mv);
                    let x = _mm256_loadu_si256(qp as *const __m256i);
                    let p32 = dot32_i8_i8(q, x, ones);
                    let sc = 1 + 2 * ((*b.add(106 + s / 2) >> (4 * (s % 2))) & 0xF) as u32;
                    let scale = _mm256_set1_ps(d * sc as f32 * *dp.add(sb * 8 + s));
                    acc = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(p32), acc);
                    qp = qp.add(32);
                }
            }
            hsum256_ps(acc)
        }
    }

    /// IQ3_XXS: like IQ3_S but signs come from 7-bit indices into ksigns.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_iq3xxs_q8(src: &[u8], qx: &QuantX) -> f32 {
        use crate::iq_grids::IQ3XXS_GRID;
        unsafe {
            let nsb = qx.d.len() / 8;
            debug_assert!(src.len() >= nsb * 98);
            let ones = _mm256_set1_epi16(1);
            let mut acc = _mm256_setzero_ps();
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            for sb in 0..nsb {
                let b = src.as_ptr().add(sb * 98);
                let d = f16_bits_to_f32(u16::from_le_bytes([*b, *b.add(1)]));
                for s in 0..8usize {
                    let qs = b.add(2 + 8 * s);
                    let aux = u32::from_le_bytes([
                        *b.add(66 + 4 * s),
                        *b.add(67 + 4 * s),
                        *b.add(68 + 4 * s),
                        *b.add(69 + 4 * s),
                    ]);
                    let mut g = [0u64; 4];
                    let mut m = [0u64; 4];
                    for l in 0..4usize {
                        g[l] = IQ3XXS_GRID[*qs.add(2 * l) as usize] as u64
                            | ((IQ3XXS_GRID[*qs.add(2 * l + 1) as usize] as u64) << 32);
                        let sg = crate::quant::ksign(((aux >> (7 * l)) & 127) as usize);
                        m[l] = SIGN_LUT[sg as usize];
                    }
                    let gv = _mm256_set_epi64x(g[3] as i64, g[2] as i64, g[1] as i64, g[0] as i64);
                    let mv = _mm256_set_epi64x(m[3] as i64, m[2] as i64, m[1] as i64, m[0] as i64);
                    let q = _mm256_sign_epi8(gv, mv);
                    let x = _mm256_loadu_si256(qp as *const __m256i);
                    let p32 = dot32_i8_i8(q, x, ones);
                    let scale =
                        _mm256_set1_ps(d * (0.5 + (aux >> 28) as f32) * 0.5 * *dp.add(sb * 8 + s));
                    acc = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(p32), acc);
                    qp = qp.add(32);
                }
            }
            hsum256_ps(acc)
        }
    }
}

/// Sized by `threads::CpuBudget::init` before first use; the fallbacks below only matter
/// when this module is exercised directly (unit tests) rather than through `main`.
static MOE_POOL: LazyLock<rayon::ThreadPool> = LazyLock::new(|| {
    let n = std::env::var("BURN_CPU_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .or_else(|| {
            std::env::var("RAYON_NUM_THREADS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
        })
        .unwrap_or_else(|| {
            let avail = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8);
            // On hybrid/SMT CPUs, using ~5/8 of logical threads (capped at 10)
            // maximizes DRAM bandwidth without hyperthread contention.
            (avail * 5 / 8).clamp(4, 10)
        })
        .max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("moe-cpu-{i}"))
        .build()
        .expect("build moe rayon pool")
});

/// Run `cpu` on the MoE pool while `main` runs on the calling thread; returns once both
/// finish. Lets the host keep enqueueing GPU work while missed experts are computed.
pub fn overlap<R>(cpu: impl FnOnce() + Send, main: impl FnOnce() -> R) -> R {
    MOE_POOL.in_place_scope(|s| {
        s.spawn(|_| cpu());
        main()
    })
}
/// One missed expert for single-token decode (`t == 1`), zero heap allocations.
#[derive(Clone, Copy)]
pub struct CpuDecodeJob<'a> {
    pub gate_raw: &'a [u8],
    pub up_raw: &'a [u8],
    pub down_raw: &'a [u8],
    pub weight: f32,
}

/// One missed expert to compute on CPU (general multi-token prefill).
pub struct CpuExpertJob<'a> {
    pub gate_raw: &'a [u8],
    pub up_raw: &'a [u8],
    pub down_raw: &'a [u8],
    /// `(token_idx, routing_weight)` pairs routed to this expert.
    pub pairs: Vec<(usize, f32)>,
}

/// Zero-overhead single-token decode CPU MoE using the spin-wait worker pool.
#[allow(clippy::too_many_arguments)]
pub fn run_experts_cpu_decode(
    gate_ty: GgmlType,
    up_ty: GgmlType,
    down_ty: GgmlType,
    jobs: &[CpuDecodeJob<'_>],
    x: &[f32],
    hidden: usize,
    inter: usize,
    out: &mut [f32],
) {
    let m = jobs.len();
    if m == 0 {
        return;
    }
    let gate_row_bytes = gate_ty.row_bytes(hidden);
    let up_row_bytes = up_ty.row_bytes(hidden);
    let down_row_bytes = down_ty.row_bytes(inter);
    struct DecodeScratch {
        qx0: QuantX,
        acts: Vec<f32>,
        qacts: Vec<QuantX>,
    }
    thread_local! {
        static SCRATCH: RefCell<DecodeScratch> = RefCell::new(DecodeScratch {
            qx0: QuantX::new(),
            acts: Vec::new(),
            qacts: Vec::new(),
        });
    }

    let x0 = &x[..hidden];
    SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        sc.qx0.quantize_into(x0);
        sc.acts.resize(m * inter, 0.0);
        while sc.qacts.len() < m {
            sc.qacts.push(QuantX::new());
        }
        let DecodeScratch { qx0, acts, qacts } = &mut *sc;
        let qx0: &QuantX = qx0;
        let acts_slice = &mut acts[..m * inter];

        let has_avx2 = avx2_available();
        let fused_mxfp4 = has_avx2 && gate_ty == GgmlType::Mxfp4 && up_ty == GgmlType::Mxfp4;

        MOE_POOL.install(|| {
            // Pass 1: gate + up + silu across all missed experts in one parallel dispatch.
            acts_slice
                .par_chunks_mut(32)
                .enumerate()
                .for_each(|(ci, dst)| {
                    let start = ci * 32;
                    for (k, o) in dst.iter_mut().enumerate() {
                        let idx = start + k;
                        let mi = idx / inter;
                        let r = idx % inter;
                        let job = &jobs[mi];
                        let g_row = &job.gate_raw[r * gate_row_bytes..(r + 1) * gate_row_bytes];
                        let u_row = &job.up_raw[r * up_row_bytes..(r + 1) * up_row_bytes];
                        let (g, u) = gate_up(fused_mxfp4, gate_ty, up_ty, g_row, u_row, qx0, x0);
                        *o = silu(g) * u;
                    }
                });

            if m <= 16 {
                for mi in 0..m {
                    qacts[mi].quantize_into(&acts_slice[mi * inter..(mi + 1) * inter]);
                }
            } else {
                qacts[..m].par_iter_mut().enumerate().for_each(|(mi, q)| {
                    q.quantize_into(&acts_slice[mi * inter..(mi + 1) * inter]);
                });
            }
            let qacts_ref = &qacts[..m];

            // Pass 2: down projection + weighted sum into `out[0..hidden]`.
            // Stream each expert sequentially across a 64-row tile to preserve hardware prefetching.
            out[..hidden]
                .par_chunks_mut(64)
                .enumerate()
                .for_each(|(ci, dst)| {
                    let d0 = ci * 64;
                    let mut acc = [0.0f32; 64];
                    for (mi, job) in jobs.iter().enumerate() {
                        let act = &acts_slice[mi * inter..(mi + 1) * inter];
                        let qa = &qacts_ref[mi];
                        let wt = job.weight;
                        for k in 0..dst.len() {
                            let off = (d0 + k) * down_row_bytes;
                            acc[k] += wt
                                * dot_q8(
                                    down_ty,
                                    &job.down_raw[off..off + down_row_bytes],
                                    qa,
                                    act,
                                );
                        }
                    }
                    dst.copy_from_slice(&acc[..dst.len()]);
                });
        });
    });
}

/// Which CPU expert kernel to run. `Decode` is the zero-allocation single-token fast path
/// (jobs converted into `scratch`); `General` is the multi-token path.
enum ExpertBatch<'a> {
    Decode(&'a [CpuDecodeJob<'a>]),
    General,
}

/// Convert to the decode layout when `t == 1` and the jobs fit in `scratch`. The fast path
/// assumes one routing pair per expert — assert it instead of silently dropping pairs.
fn classify_expert_batch<'a>(
    t: usize,
    jobs: &[CpuExpertJob<'a>],
    scratch: &'a mut [CpuDecodeJob<'a>; 16],
) -> ExpertBatch<'a> {
    if t == 1 && jobs.len() <= scratch.len() {
        for (i, j) in jobs.iter().enumerate() {
            debug_assert_eq!(j.pairs.len(), 1, "t == 1 needs one pair per expert");
            scratch[i] = CpuDecodeJob {
                gate_raw: j.gate_raw,
                up_raw: j.up_raw,
                down_raw: j.down_raw,
                weight: j.pairs[0].1,
            };
        }
        ExpertBatch::Decode(&scratch[..jobs.len()])
    } else {
        ExpertBatch::General
    }
}

/// Run a set of missed experts on CPU and accumulate their weighted outputs into
/// `out` (`[t, hidden]`, zero-initialized by the caller).
#[allow(clippy::too_many_arguments)]
pub fn run_experts_cpu(
    gate_ty: GgmlType,
    up_ty: GgmlType,
    down_ty: GgmlType,
    jobs: &[CpuExpertJob<'_>],
    x: &[f32],
    t: usize,
    hidden: usize,
    inter: usize,
    out: &mut [f32],
) {
    if jobs.is_empty() {
        return;
    }
    let mut decode_scratch = [CpuDecodeJob {
        gate_raw: &[],
        up_raw: &[],
        down_raw: &[],
        weight: 0.0,
    }; 16];
    if let ExpertBatch::Decode(djobs) = classify_expert_batch(t, jobs, &mut decode_scratch) {
        run_experts_cpu_decode(gate_ty, up_ty, down_ty, djobs, x, hidden, inter, out);
        return;
    }
    let gate_row_bytes = gate_ty.row_bytes(hidden);
    let up_row_bytes = up_ty.row_bytes(hidden);
    let down_row_bytes = down_ty.row_bytes(inter);

    struct MultiScratch {
        qxs: Vec<QuantX>,
        acts: Vec<f32>,
        qacts: Vec<QuantX>,
        off: Vec<usize>,
    }
    thread_local! {
        static MULTI_SCRATCH: RefCell<MultiScratch> = RefCell::new(MultiScratch {
            qxs: Vec::new(),
            acts: Vec::new(),
            qacts: Vec::new(),
            off: Vec::new(),
        });
    }

    let has_avx2 = avx2_available();
    let fused_mxfp4 = has_avx2 && gate_ty == GgmlType::Mxfp4 && up_ty == GgmlType::Mxfp4;
    let batch_q5k = has_avx2 && down_ty == GgmlType::Q5K;

    MULTI_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        while sc.qxs.len() < t {
            sc.qxs.push(QuantX::new());
        }
        sc.off.clear();
        sc.off.push(0usize);
        let mut total_pairs = 0usize;
        for j in jobs {
            total_pairs += j.pairs.len();
            sc.off.push(total_pairs * inter);
        }
        let total_acts = total_pairs * inter;
        if sc.acts.len() < total_acts {
            sc.acts.resize(total_acts, 0.0);
        }
        while sc.qacts.len() < total_pairs {
            sc.qacts.push(QuantX::new());
        }

        let MultiScratch {
            qxs,
            acts,
            qacts,
            off,
        } = &mut *sc;
        let qxs_slice = &mut qxs[..t];
        let acts_slice = &mut acts[..total_acts];
        let qacts_slice = &mut qacts[..total_pairs];
        let off_slice = &off[..jobs.len() + 1];

        let acts_ptr = acts_slice.as_mut_ptr() as usize;

        MOE_POOL.install(|| {
            if t <= 4 {
                for (tok, q) in qxs_slice.iter_mut().enumerate() {
                    q.quantize_into(&x[tok * hidden..(tok + 1) * hidden]);
                }
            } else {
                qxs_slice.par_iter_mut().enumerate().for_each(|(tok, q)| {
                    q.quantize_into(&x[tok * hidden..(tok + 1) * hidden]);
                });
            }
            let qxs_ref: &[QuantX] = qxs_slice;

            // Pass 1: gate + up + silu for every (expert, row, pair).
            (0..jobs.len() * inter)
                .into_par_iter()
                .with_min_len(16)
                .for_each(|idx| {
                    let (ji, r) = (idx / inter, idx % inter);
                    let job = &jobs[ji];
                    let g_row = &job.gate_raw[r * gate_row_bytes..(r + 1) * gate_row_bytes];
                    let u_row = &job.up_raw[r * up_row_bytes..(r + 1) * up_row_bytes];
                    let dst = acts_ptr as *mut f32;
                    let np = job.pairs.len();
                    if fused_mxfp4 && np == 1 {
                        let tok = job.pairs[0].0;
                        let (g, u) = gate_up(
                            fused_mxfp4,
                            gate_ty,
                            up_ty,
                            g_row,
                            u_row,
                            &qxs_ref[tok],
                            &x[tok * hidden..(tok + 1) * hidden],
                        );
                        unsafe { *dst.add(off_slice[ji] + r) = silu(g) * u };
                    } else if fused_mxfp4 && (2..=4).contains(&np) {
                        let mut qptrs = [&qxs_ref[0]; 4];
                        let mut xs: [&[f32]; 4] = [&[]; 4];
                        for (pi, &(tok, _)) in job.pairs.iter().enumerate() {
                            qptrs[pi] = &qxs_ref[tok];
                            xs[pi] = &x[tok * hidden..(tok + 1) * hidden];
                        }
                        let mut gs = [0.0f32; 4];
                        let mut us = [0.0f32; 4];
                        gate_up_batch(
                            fused_mxfp4,
                            gate_ty,
                            up_ty,
                            g_row,
                            u_row,
                            &qptrs[..np],
                            &xs[..np],
                            &mut gs[..np],
                            &mut us[..np],
                        );
                        for pi in 0..np {
                            unsafe {
                                *dst.add(off_slice[ji] + pi * inter + r) = silu(gs[pi]) * us[pi];
                            }
                        }
                    } else {
                        for (pi, &(tok, _)) in job.pairs.iter().enumerate() {
                            let xt = &x[tok * hidden..(tok + 1) * hidden];
                            let (g, u) = gate_up(
                                fused_mxfp4,
                                gate_ty,
                                up_ty,
                                g_row,
                                u_row,
                                &qxs_ref[tok],
                                xt,
                            );
                            unsafe { *dst.add(off_slice[ji] + pi * inter + r) = silu(g) * u };
                        }
                    }
                });

            if total_pairs <= 16 {
                for (pidx, q) in qacts_slice.iter_mut().enumerate() {
                    let s = pidx * inter;
                    q.quantize_into(&acts_slice[s..s + inter]);
                }
            } else {
                qacts_slice
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(pidx, q)| {
                        let s = pidx * inter;
                        q.quantize_into(&acts_slice[s..s + inter]);
                    });
            }
            let qacts_ref: &[QuantX] = qacts_slice;
            let pair_off: Vec<usize> = {
                let mut v = Vec::with_capacity(jobs.len());
                let mut acc_p = 0usize;
                for j in jobs {
                    v.push(acc_p);
                    acc_p += j.pairs.len();
                }
                v
            };

            // Pass 2: down projection, tiled in 64-column chunks so each expert streams
            // sequentially across 64 rows without ping-ponging across weight buffers.
            let out_ptr = out.as_mut_ptr() as usize;
            let ntiles = hidden.div_ceil(64);
            (0..ntiles).into_par_iter().for_each(|ci| {
                let d0 = ci * 64;
                let len = (hidden - d0).min(64);
                let dst = out_ptr as *mut f32;
                if t <= 8 {
                    let mut acc = [[0.0f32; 64]; 8];
                    for (ji, job) in jobs.iter().enumerate() {
                        let p0 = pair_off[ji];
                        let np = job.pairs.len();
                        if batch_q5k && (2..=4).contains(&np) {
                            let mut qptrs = [&qacts_ref[0]; 4];
                            let mut acts: [&[f32]; 4] = [&[]; 4];
                            for pi in 0..np {
                                qptrs[pi] = &qacts_ref[p0 + pi];
                                acts[pi] = &acts_slice[(p0 + pi) * inter..(p0 + pi + 1) * inter];
                            }
                            let mut ys = [0.0f32; 4];
                            for k in 0..len {
                                let off = (d0 + k) * down_row_bytes;
                                let d_row = &job.down_raw[off..off + down_row_bytes];
                                down_batch_q5k(
                                    batch_q5k,
                                    d_row,
                                    &qptrs[..np],
                                    &acts[..np],
                                    &mut ys[..np],
                                );
                                for (pi, &(tok, wt)) in job.pairs.iter().enumerate() {
                                    acc[tok][k] += wt * ys[pi];
                                }
                            }
                        } else {
                            for k in 0..len {
                                let off = (d0 + k) * down_row_bytes;
                                let d_row = &job.down_raw[off..off + down_row_bytes];
                                for (pi, &(tok, wt)) in job.pairs.iter().enumerate() {
                                    let s = (p0 + pi) * inter;
                                    let y = wt
                                        * dot_q8(
                                            down_ty,
                                            d_row,
                                            &qacts_ref[p0 + pi],
                                            &acts_slice[s..s + inter],
                                        );
                                    acc[tok][k] += y;
                                }
                            }
                        }
                    }
                    for tok in 0..t {
                        for k in 0..len {
                            unsafe { *dst.add(tok * hidden + d0 + k) += acc[tok][k] };
                        }
                    }
                } else {
                    for (ji, job) in jobs.iter().enumerate() {
                        let p0 = pair_off[ji];
                        for k in 0..len {
                            let d = d0 + k;
                            let off = d * down_row_bytes;
                            let d_row = &job.down_raw[off..off + down_row_bytes];
                            for (pi, &(tok, wt)) in job.pairs.iter().enumerate() {
                                let s = (p0 + pi) * inter;
                                let y = wt
                                    * dot_q8(
                                        down_ty,
                                        d_row,
                                        &qacts_ref[p0 + pi],
                                        &acts_slice[s..s + inter],
                                    );
                                unsafe { *dst.add(tok * hidden + d) += y };
                            }
                        }
                    }
                }
            });
        });
    });
}

/// Entry points used by `crate::vec::Avx2`. Kept here so the private kernels stay private.
#[cfg(target_arch = "x86_64")]
pub(crate) unsafe fn avx2_dot_q8(ty: GgmlType, src: &[u8], qx: &QuantX, x: &[f32]) -> f32 {
    match ty {
        GgmlType::Mxfp4 => avx2::dot_mxfp4_q8(src, qx),
        GgmlType::Q5K => avx2::dot_q5k_q8(src, qx),
        GgmlType::Q6K => avx2::dot_q6k_q8(src, qx),
        GgmlType::Q4K => avx2::dot_q4k_q8(src, qx),
        GgmlType::Q3K => avx2::dot_q3k_q8(src, qx),
        GgmlType::Q8_0 => avx2::dot_q8_0_q8(src, qx),
        GgmlType::Iq3S => avx2::dot_iq3s_q8(src, qx),
        GgmlType::Iq3Xxs => avx2::dot_iq3xxs_q8(src, qx),
        _ => dot_scalar(ty, src, x),
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) unsafe fn avx2_quantize_into(x: &[f32], qs: &mut [i8], d: &mut [f32], bsum: &mut [f32]) {
    avx2::quantize_into_avx2(x, qs, d, bsum);
}

/// ARM NEON quantized dots. NEON is baseline on aarch64; Q6_K additionally needs DotProd
/// (USDOT), which is checked at runtime so other aarch64 cores keep using the scalar path.
/// The accumulation order mirrors `dot_scalar` so results are bit-identical.
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;
    use std::arch::is_aarch64_feature_detected;
    use std::sync::OnceLock;

    pub(super) fn dotprod_available() -> bool {
        static AVAILABLE: OnceLock<bool> = OnceLock::new();
        *AVAILABLE.get_or_init(|| is_aarch64_feature_detected!("dotprod"))
    }

    /// Q6_K superblocks against our per-32 Q8 activations. `None` means DotProd is absent
    /// and the caller should fall back to `dot_scalar`.
    pub fn dot_q6k_q8(src: &[u8], qx: &QuantX) -> Option<f32> {
        if !dotprod_available() {
            return None;
        }
        // SAFETY: the runtime check above confirmed DotProd support.
        Some(unsafe { dot_q6k_q8_dotprod(src, qx) })
    }

    /// Q4_K superblocks against our per-32 Q8 activations, mirroring `avx2::dot_q4k_q8`:
    /// one signed dot per 32-value sub-block, a per-sub-block scale, and a separate
    /// `dmin * min * bsum` correction. `None` selects the portable fallback.
    pub fn dot_q4k_q8(src: &[u8], qx: &QuantX) -> Option<f32> {
        if !dotprod_available() {
            return None;
        }
        // SAFETY: the runtime check above confirmed DotProd support.
        Some(unsafe { dot_q4k_q8_dotprod(src, qx) })
    }

    /// MXFP4 superblocks (17-byte blocks: E8M0 scale + 16 packed nibbles).
    pub fn dot_mxfp4_q8(src: &[u8], qx: &QuantX) -> Option<f32> {
        if !dotprod_available() {
            return None;
        }
        Some(unsafe { dot_mxfp4_q8_dotprod(src, qx) })
    }

    /// Q5_K superblocks (176-byte blocks).
    pub fn dot_q5k_q8(src: &[u8], qx: &QuantX) -> Option<f32> {
        if !dotprod_available() {
            return None;
        }
        Some(unsafe { dot_q5k_q8_dotprod(src, qx) })
    }

    /// Q8_0 blocks (34 bytes: f16 scale + 32 i8 weights).
    pub fn dot_q8_0_q8(src: &[u8], qx: &QuantX) -> Option<f32> {
        if !dotprod_available() {
            return None;
        }
        Some(unsafe { dot_q8_0_q8_dotprod(src, qx) })
    }

    /// Q3_K superblocks (110 bytes).
    pub fn dot_q3k_q8(src: &[u8], qx: &QuantX) -> Option<f32> {
        if !dotprod_available() {
            return None;
        }
        Some(unsafe { dot_q3k_q8_dotprod(src, qx) })
    }

    #[target_feature(enable = "dotprod")]
    unsafe fn dot_mxfp4_q8_dotprod(src: &[u8], qx: &QuantX) -> f32 {
        unsafe {
            let nb = qx.qs.len() / 32;
            debug_assert!(src.len() >= nb * 17);
            let lo4 = vdupq_n_u8(0x0F);
            let zero = vdupq_n_s32(0);
            // The 16-entry table covers both signs: indices 0..7 are positive, 8..15 negative.
            let table = vld1q_s8(MXFP4_VALUES.as_ptr());
            let mut sum = 0.0f32;
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            for b in 0..nb {
                let p = src.as_ptr().add(b * 17);
                let d = E8M0_LUT[*p as usize] * *dp.add(b);
                let packed = vld1q_u8(p.add(1));
                let w0 = vqtbl1q_s8(table, vandq_u8(packed, lo4));
                let w1 = vqtbl1q_s8(table, vshrq_n_u8(packed, 4));
                let dot0 = vaddvq_s32(vdotq_s32(zero, w0, vld1q_s8(qp)));
                let dot1 = vaddvq_s32(vdotq_s32(zero, w1, vld1q_s8(qp.add(16))));
                sum += d * ((dot0 + dot1) as f32);
                qp = qp.add(32);
            }
            sum
        }
    }

    #[target_feature(enable = "dotprod")]
    unsafe fn dot_q8_0_q8_dotprod(src: &[u8], qx: &QuantX) -> f32 {
        unsafe {
            let nb = qx.qs.len() / 32;
            debug_assert!(src.len() >= nb * 34);
            let zero = vdupq_n_s32(0);
            let mut sum = 0.0f32;
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            for b in 0..nb {
                let p = src.as_ptr().add(b * 34);
                let d = f16::from_le_bytes([*p, *p.add(1)]).to_f32() * *dp.add(b);
                let dot0 = vaddvq_s32(vdotq_s32(
                    zero,
                    vld1q_s8(p.add(2) as *const i8),
                    vld1q_s8(qp),
                ));
                let dot1 = vaddvq_s32(vdotq_s32(
                    zero,
                    vld1q_s8(p.add(18) as *const i8),
                    vld1q_s8(qp.add(16)),
                ));
                sum += d * ((dot0 + dot1) as f32);
                qp = qp.add(32);
            }
            sum
        }
    }

    #[target_feature(enable = "dotprod")]
    unsafe fn dot_q5k_q8_dotprod(src: &[u8], qx: &QuantX) -> f32 {
        unsafe {
            let nsb = qx.qs.len() / 256;
            debug_assert!(src.len() >= nsb * 176);
            let lo4 = vdupq_n_u8(0x0F);
            let one = vdupq_n_u8(1);
            let zero = vdupq_n_s32(0);
            let mut sum = 0.0f32;
            let mut min_acc = 0.0f32;
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            let bp = qx.bsum.as_ptr();
            for sb in 0..nsb {
                let b = src.as_ptr().add(sb * 176);
                let d = f16::from_le_bytes([*b, *b.add(1)]).to_f32();
                let dmin = f16::from_le_bytes([*b.add(2), *b.add(3)]).to_f32();
                let scales = std::slice::from_raw_parts(b.add(4), 12);
                let qh = b.add(16);
                let qs = b.add(48);
                for s in 0..8usize {
                    let (sc, m) = scale_min_k4(s, scales);
                    let nibbles = qs.add(32 * (s / 2));
                    let scale = d * (sc as f32) * *dp.add(sb * 8 + s);
                    for half in 0..2usize {
                        let raw = vld1q_u8(nibbles.add(half * 16));
                        let nib = if s & 1 == 0 {
                            vandq_u8(raw, lo4)
                        } else {
                            vshrq_n_u8(raw, 4)
                        };
                        let hi = vandq_u8(
                            vshlq_u8(vld1q_u8(qh.add(half * 16)), vdupq_n_s8(-(s as i8))),
                            one,
                        );
                        let w = vorrq_u8(nib, vshlq_n_u8(hi, 4));
                        let dot = vaddvq_s32(vdotq_s32(
                            zero,
                            vreinterpretq_s8_u8(w),
                            vld1q_s8(qp.add(half * 16)),
                        ));
                        sum += (dot as f32) * scale;
                    }
                    min_acc += dmin * (m as f32) * *bp.add(sb * 8 + s);
                    qp = qp.add(32);
                }
            }
            sum - min_acc
        }
    }

    #[target_feature(enable = "dotprod")]
    unsafe fn dot_q3k_q8_dotprod(src: &[u8], qx: &QuantX) -> f32 {
        unsafe {
            let nsb = qx.qs.len() / 256;
            debug_assert!(src.len() >= nsb * 110);
            let three = vdupq_n_u8(3);
            let four = vdupq_n_u8(4);
            let zero8 = vdupq_n_u8(0);
            let zero32 = vdupq_n_s32(0);
            let mut sum = 0.0f32;
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            for sb in 0..nsb {
                let b = src.as_ptr().add(sb * 110);
                let d = f16::from_le_bytes([*b.add(108), *b.add(109)]).to_f32();
                let sc = b.add(96);
                let hm = b;
                let qs = b.add(32);
                for s in 0..8usize {
                    let (n, j) = (s / 4, s % 4);
                    let qn = qs.add(32 * n);
                    // `n*4+j` is the bit position within each sign byte, not a byte index.
                    let mask = vdupq_n_u8(1u8 << (n * 4 + j));
                    for half in 0..2usize {
                        let sign_bits = vld1q_u8(hm.add(16 * half));
                        let q = vandq_u8(
                            vshlq_u8(vld1q_u8(qn.add(16 * half)), vdupq_n_s8(-((2 * j) as i8))),
                            three,
                        );
                        // A clear sign bit contributes +4 before subtraction, i.e. -4..3.
                        let hi = vandq_u8(vceqq_u8(vandq_u8(sign_bits, mask), zero8), four);
                        let w = vreinterpretq_s8_u8(vsubq_u8(q, hi));
                        let dot = vaddvq_s32(vdotq_s32(zero32, w, vld1q_s8(qp)));
                        let scale = d
                            * *dp.add(sb * 8 + s)
                            * crate::quant::q3k_scale(
                                std::slice::from_raw_parts(sc, 12),
                                s * 2 + half,
                            ) as f32;
                        sum += (dot as f32) * scale;
                        qp = qp.add(16);
                    }
                }
            }
            sum
        }
    }

    #[target_feature(enable = "dotprod")]
    unsafe fn dot_q4k_q8_dotprod(src: &[u8], qx: &QuantX) -> f32 {
        unsafe {
            let nsb = qx.d.len() / 8;
            debug_assert!(src.len() >= nsb * 144);
            let lo4 = vdupq_n_u8(0x0F);
            let zero = vdupq_n_s32(0);
            let mut sum = 0.0f32;
            let mut min_acc = 0.0f32;
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            let bp = qx.bsum.as_ptr();
            for sb in 0..nsb {
                let b = src.as_ptr().add(sb * 144);
                let d = f16::from_le_bytes([*b, *b.add(1)]).to_f32();
                let dmin = f16::from_le_bytes([*b.add(2), *b.add(3)]).to_f32();
                let scales = std::slice::from_raw_parts(b.add(4), 12);
                let qs = b.add(16);
                for pair in 0..4usize {
                    let ql = qs.add(pair * 32);
                    for odd in 0..2usize {
                        let s = pair * 2 + odd;
                        let bi = sb * 8 + s;
                        let (sc, m) = scale_min_k4(s, scales);
                        let v0 = vld1q_u8(ql);
                        let v1 = vld1q_u8(ql.add(16));
                        let (n0, n1) = if odd == 0 {
                            (vandq_u8(v0, lo4), vandq_u8(v1, lo4))
                        } else {
                            (vshrq_n_u8(v0, 4), vshrq_n_u8(v1, 4))
                        };
                        let x0 = vld1q_s8(qp);
                        let x1 = vld1q_s8(qp.add(16));
                        let dot0 = vaddvq_s32(vdotq_s32(zero, vreinterpretq_s8_u8(n0), x0));
                        let dot1 = vaddvq_s32(vdotq_s32(zero, vreinterpretq_s8_u8(n1), x1));
                        sum += ((dot0 + dot1) as f32) * (d * (sc as f32) * *dp.add(bi));
                        min_acc += dmin * (m as f32) * *bp.add(bi);
                        qp = qp.add(32);
                    }
                }
            }
            sum - min_acc
        }
    }

    #[target_feature(enable = "dotprod")]
    unsafe fn dot_q6k_q8_dotprod(src: &[u8], qx: &QuantX) -> f32 {
        unsafe {
            let nsb = qx.d.len() / 8;
            debug_assert!(src.len() >= nsb * 210);
            let lo4 = vdupq_n_u8(0x0F);
            let hi2 = vdupq_n_u8(0x03);
            let bias32 = vdupq_n_s8(32);
            let zero = vdupq_n_s32(0);
            let mut sum = 0.0f32;
            let mut qp = qx.qs.as_ptr();
            let dp = qx.d.as_ptr();
            for sb in 0..nsb {
                let b = src.as_ptr().add(sb * 210);
                let d = f16::from_le_bytes([*b.add(208), *b.add(209)]).to_f32();
                ql_block(b, qp, dp, d, sb, lo4, hi2, bias32, zero, &mut sum);
                qp = qp.add(256);
            }
            sum
        }
    }

    /// One super-block: two halves of 128 values each, four 32-value planes per half.
    #[target_feature(enable = "dotprod")]
    unsafe fn ql_block(
        b: *const u8,
        qp: *const i8,
        dp: *const f32,
        d: f32,
        sb: usize,
        lo4: uint8x16_t,
        hi2: uint8x16_t,
        bias32: int8x16_t,
        zero: int32x4_t,
        sum: &mut f32,
    ) {
        unsafe {
            for n in 0..2usize {
                let ql = b.add(64 * n);
                let qh = b.add(128 + 32 * n);
                let sc = b.add(192 + 8 * n) as *const i8;
                for k in 0..4usize {
                    let shift = k * 2;
                    let qlo = ql.add((k & 1) * 32);
                    let dx = *dp.add(sb * 8 + n * 4 + k);
                    for half in 0..2usize {
                        let lo = vld1q_u8(qlo.add(half * 16));
                        let hv = vld1q_u8(qh.add(half * 16));
                        let nib = if k < 2 {
                            vandq_u8(lo, lo4)
                        } else {
                            vshrq_n_u8(lo, 4)
                        };
                        let bits = match shift {
                            0 => hv,
                            2 => vshrq_n_u8(hv, 2),
                            4 => vshrq_n_u8(hv, 4),
                            _ => vshrq_n_u8(hv, 6),
                        };
                        let w = vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(nib, vshlq_n_u8(vandq_u8(bits, hi2), 4))),
                            bias32,
                        );
                        let x = vld1q_s8(qp.add(n * 128 + k * 32 + half * 16));
                        let dot = vaddvq_s32(vdotq_s32(zero, w, x));
                        let scale = d * dx * *sc.add(k * 2 + half) as i8 as f32;
                        *sum += (dot as f32) * scale;
                    }
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Order-matched scalar reference: same (sb, n, k, half) traversal as the NEON kernel.
        fn ordered_scalar(src: &[u8], qx: &QuantX) -> f32 {
            let nsb = qx.d.len() / 8;
            let mut sum = 0.0f32;
            for sb in 0..nsb {
                let b = &src[sb * 210..(sb + 1) * 210];
                let d = f16::from_le_bytes([b[208], b[209]]).to_f32();
                for n in 0..2 {
                    let ql = &b[64 * n..64 * n + 64];
                    let qh = &b[128 + 32 * n..128 + 32 * n + 32];
                    let sc = &b[192 + 8 * n..192 + 8 * n + 8];
                    for k in 0..4 {
                        let dx = qx.d[sb * 8 + n * 4 + k];
                        for half in 0..2 {
                            let mut dot = 0i32;
                            for i in 0..16 {
                                let qlo = ql[(k & 1) * 32 + half * 16 + i];
                                let nib = if k < 2 { qlo & 0x0F } else { qlo >> 4 };
                                let bits = (qh[half * 16 + i] >> (k * 2)) & 3;
                                let w = ((nib | (bits << 4)) as i8) as i32 - 32;
                                let xv = qx.qs[sb * 256 + n * 128 + k * 32 + half * 16 + i] as i32;
                                dot += w * xv;
                            }
                            let scale = d * dx * sc[k * 2 + half] as i8 as f32;
                            sum += (dot as f32) * scale;
                        }
                    }
                }
            }
            sum
        }

        fn sample() -> (Vec<u8>, Vec<f32>, QuantX) {
            let mut state = 0x9e37_79b9_7f4a_7c15u64;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            let blocks = 4;
            let mut src = vec![0u8; blocks * 210];
            let mut x = vec![0.0f32; blocks * 256];
            for sb in 0..blocks {
                let b = &mut src[sb * 210..(sb + 1) * 210];
                b[208..210].copy_from_slice(
                    &f16::from_f32(0.5 + (next() & 0xFF) as f32 / 255.0).to_le_bytes(),
                );
                for byte in b[..208].iter_mut() {
                    *byte = next() as u8;
                }
                for v in x[sb * 256..(sb + 1) * 256].iter_mut() {
                    *v = ((next() >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
                }
            }
            let qx = QuantX::from_f32(&x);
            (src, x, qx)
        }

        fn sample_dot(ty: GgmlType, blocks: usize) -> (Vec<u8>, Vec<f32>, QuantX) {
            let (be, bb) = ty.block();
            let mut state = 0x5bf0_36a8_5eb3_7f11u64;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            let src: Vec<u8> = (0..blocks * bb).map(|_| next() as u8).collect();
            let x: Vec<f32> = (0..blocks * be)
                .map(|_| ((next() >> 40) as f32 / (1u64 << 24) as f32) - 0.5)
                .collect();
            let qx = QuantX::from_f32(&x);
            (src, x, qx)
        }

        fn assert_close(ty: GgmlType, neon: Option<f32>, reference: f32) {
            let v = neon.expect("DotProd is available on this host");
            assert!(
                (v - reference).abs() <= 1e-3 * reference.abs().max(1.0),
                "{ty:?}: neon={v} reference={reference}"
            );
        }

        /// Quantized-activation references: the SIMD kernels use `qx.qs * qx.d`, not the
        /// full-precision `x` that `dot_scalar` uses.
        fn mxfp4_ordered(src: &[u8], qx: &QuantX) -> f32 {
            let mut sum = 0.0f32;
            for b in 0..qx.d.len() {
                let p = &src[b * 17..];
                let d = E8M0_LUT[p[0] as usize] * qx.d[b];
                let mut dot = 0i32;
                for j in 0..16 {
                    dot +=
                        MXFP4_VALUES[(p[1 + j] & 0xF) as usize] as i32 * qx.qs[b * 32 + j] as i32;
                    dot += MXFP4_VALUES[(p[1 + j] >> 4) as usize] as i32
                        * qx.qs[b * 32 + 16 + j] as i32;
                }
                sum += d * dot as f32;
            }
            sum
        }

        fn q8_0_ordered(src: &[u8], qx: &QuantX) -> f32 {
            let mut sum = 0.0f32;
            for b in 0..qx.d.len() {
                let p = &src[b * 34..];
                let d = f16::from_le_bytes([p[0], p[1]]).to_f32() * qx.d[b];
                let mut dot = 0i32;
                for j in 0..32 {
                    dot += (p[2 + j] as i8) as i32 * qx.qs[b * 32 + j] as i32;
                }
                sum += d * dot as f32;
            }
            sum
        }

        fn q5k_ordered(src: &[u8], qx: &QuantX) -> f32 {
            let nsb = qx.d.len() / 8;
            let mut sum = 0.0f32;
            let mut min_acc = 0.0f32;
            for sb in 0..nsb {
                let b = &src[sb * 176..];
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
                for s in 0..8 {
                    let (sc, m) = scale_min_k4(s, &b[4..16]);
                    let mut dot = 0i32;
                    for l in 0..32 {
                        let lo = (b[48 + 32 * (s / 2) + l] >> (4 * (s & 1))) & 0xF;
                        let hi = ((b[16 + l] >> s) & 1) << 4;
                        dot += (lo | hi) as i32 * qx.qs[sb * 256 + s * 32 + l] as i32;
                    }
                    let bi = sb * 8 + s;
                    sum += (dot as f32) * (d * (sc as f32) * qx.d[bi]);
                    min_acc += dmin * (m as f32) * qx.bsum[bi];
                }
            }
            sum - min_acc
        }

        fn q3k_ordered(src: &[u8], qx: &QuantX) -> f32 {
            let nsb = qx.d.len() / 8;
            let mut sum = 0.0f32;
            for sb in 0..nsb {
                let b = &src[sb * 110..];
                let d = f16::from_le_bytes([b[108], b[109]]).to_f32();
                for s in 0..8 {
                    let (n, j) = (s / 4, s % 4);
                    for half in 0..2 {
                        let dl = d
                            * qx.d[sb * 8 + s]
                            * crate::quant::q3k_scale(&b[96..108], s * 2 + half) as f32;
                        let mut dot = 0i32;
                        for l in 0..16 {
                            let li = half * 16 + l;
                            let q = (b[32 + 32 * n + li] >> (2 * j)) & 3;
                            let hi = if b[li] & (1 << (n * 4 + j)) != 0 {
                                0
                            } else {
                                4
                            };
                            dot += (q as i32 - hi) * qx.qs[sb * 256 + s * 32 + li] as i32;
                        }
                        sum += dl * dot as f32;
                    }
                }
            }
            sum
        }

        #[test]
        fn mxfp4_neon_matches_ordered_scalar() {
            if !dotprod_available() {
                return;
            }
            let (src, _, qx) = sample_dot(GgmlType::Mxfp4, 4);
            assert_close(
                GgmlType::Mxfp4,
                dot_mxfp4_q8(&src, &qx),
                mxfp4_ordered(&src, &qx),
            );
        }

        #[test]
        fn q8_0_neon_matches_ordered_scalar() {
            if !dotprod_available() {
                return;
            }
            let (src, _, qx) = sample_dot(GgmlType::Q8_0, 4);
            assert_close(
                GgmlType::Q8_0,
                dot_q8_0_q8(&src, &qx),
                q8_0_ordered(&src, &qx),
            );
        }

        #[test]
        fn q5k_neon_matches_ordered_scalar() {
            if !dotprod_available() {
                return;
            }
            let (src, _, qx) = sample_dot(GgmlType::Q5K, 4);
            assert_close(GgmlType::Q5K, dot_q5k_q8(&src, &qx), q5k_ordered(&src, &qx));
        }

        #[test]
        fn q3k_neon_matches_ordered_scalar() {
            if !dotprod_available() {
                return;
            }
            let (src, _, qx) = sample_dot(GgmlType::Q3K, 4);
            assert_close(GgmlType::Q3K, dot_q3k_q8(&src, &qx), q3k_ordered(&src, &qx));
        }

        /// Order-matched scalar reference for the Q4_K kernel above.
        fn q4k_ordered(src: &[u8], qx: &QuantX) -> f32 {
            let nsb = qx.d.len() / 8;
            let mut sum = 0.0f32;
            let mut min_acc = 0.0f32;
            for sb in 0..nsb {
                let b = &src[sb * 144..(sb + 1) * 144];
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
                let scales = &b[4..16];
                let qs = &b[16..];
                for pair in 0..4 {
                    for odd in 0..2 {
                        let s = pair * 2 + odd;
                        let bi = sb * 8 + s;
                        let (sc, m) = scale_min_k4(s, scales);
                        let mut dot = 0i32;
                        for i in 0..32 {
                            let byte = qs[pair * 32 + i];
                            let nib = if odd == 0 { byte & 0x0F } else { byte >> 4 };
                            dot += (nib as i32) * (qx.qs[bi * 32 + i] as i32);
                        }
                        sum += (dot as f32) * (d * (sc as f32) * qx.d[bi]);
                        min_acc += dmin * (m as f32) * qx.bsum[bi];
                    }
                }
            }
            sum - min_acc
        }

        fn sample_q4k() -> (Vec<u8>, Vec<f32>, QuantX) {
            let mut state = 0x243f_6a88_85a3_08d3u64;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            let blocks = 4;
            let mut src = vec![0u8; blocks * 144];
            let mut x = vec![0.0f32; blocks * 256];
            for sb in 0..blocks {
                let b = &mut src[sb * 144..(sb + 1) * 144];
                b[0..2].copy_from_slice(
                    &f16::from_f32(0.25 + (next() & 0xFF) as f32 / 512.0).to_le_bytes(),
                );
                b[2..4].copy_from_slice(
                    &f16::from_f32(0.125 + (next() & 0xFF) as f32 / 1024.0).to_le_bytes(),
                );
                for byte in b[4..].iter_mut() {
                    *byte = next() as u8;
                }
                for v in x[sb * 256..(sb + 1) * 256].iter_mut() {
                    *v = ((next() >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
                }
            }
            let qx = QuantX::from_f32(&x);
            (src, x, qx)
        }

        #[test]
        fn q4k_neon_matches_ordered_scalar_bit_exactly() {
            if !dotprod_available() {
                eprintln!("skipping: host has no DotProd support");
                return;
            }
            let (src, _, qx) = sample_q4k();
            let expected = q4k_ordered(&src, &qx);
            let actual = unsafe { dot_q4k_q8_dotprod(&src, &qx) };
            assert_eq!(actual.to_bits(), expected.to_bits());
        }

        #[test]
        fn q4k_neon_is_close_to_the_portable_dot() {
            if !dotprod_available() {
                eprintln!("skipping: host has no DotProd support");
                return;
            }
            let (src, x, qx) = sample_q4k();
            let neon = unsafe { dot_q4k_q8_dotprod(&src, &qx) };
            let portable = dot_scalar(GgmlType::Q4K, &src, &x);
            assert!((neon - portable).abs() <= 1e-2 * portable.abs().max(1.0));
        }

        #[test]
        fn q6k_neon_matches_ordered_scalar_bit_exactly() {
            if !dotprod_available() {
                eprintln!("skipping: host has no DotProd support");
                return;
            }
            let (src, _, qx) = sample();
            let expected = ordered_scalar(&src, &qx);
            let actual = unsafe { dot_q6k_q8_dotprod(&src, &qx) };
            assert_eq!(actual.to_bits(), expected.to_bits());
        }

        #[test]
        fn dot_q8_delegates_to_selected_backend() {
            // The selected backend must produce the same result as calling the dispatch
            // path for a format where the AVX2 and scalar reductions are order-matched.
            let (src, x, qx) = sample();
            let via_backend = crate::vec::backend().dot_q8(GgmlType::Q6K, &src, &qx, &x);
            let via_dispatch = dot_q8(GgmlType::Q6K, &src, &qx, &x);
            assert_eq!(via_backend, via_dispatch);
        }

        #[test]
        fn q6k_neon_is_close_to_the_portable_dot() {
            if !dotprod_available() {
                eprintln!("skipping: host has no DotProd support");
                return;
            }
            let (src, x, qx) = sample();
            let neon = unsafe { dot_q6k_q8_dotprod(&src, &qx) };
            let portable = dot(GgmlType::Q6K, &src, &x);
            assert!((neon - portable).abs() <= 1e-3 * portable.abs().max(1.0));
        }
    }
}
