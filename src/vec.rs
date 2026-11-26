//! Backend-neutral vector operations.
//!
//! One implementation is selected at startup (AVX2 on x86_64, NEON DotProd on aarch64,
//! portable scalar otherwise) and published as a `dyn VecBackend`. Call sites no longer
//! need `#[cfg]`, `match GgmlType`, or ad-hoc feature booleans; adding a format or an
//! architecture means adding one impl here.
//!
//! Each backend intentionally reproduces the reduction order of the code it replaces on
//! that architecture. Logits are byte-identical to the previous implementation on the
//! same host; cross-architecture tiny differences remain expected.

use crate::cpu_moe::QuantX;
use crate::quant::GgmlType;
use std::sync::LazyLock;

pub trait VecBackend: Send + Sync {
    fn name(&self) -> &'static str;

    /// Quantized dot of a GGUF weight row against a per-32-block Q8 activation.
    fn dot_q8(&self, ty: GgmlType, src: &[u8], qx: &QuantX, x: &[f32]) -> f32;

    /// Per-32-block symmetric i8 quantization: scales, exact block sums and codes.
    fn quantize_into(&self, x: &[f32], qs: &mut [i8], d: &mut [f32], bsum: &mut [f32]);

    /// RMS norm. The reduction stays in scalar order (so results are bit-identical);
    /// only the elementwise scale is vectorized.
    fn rms_norm(&self, x: &[f32], w: &[f32], eps: f32) -> Vec<f32>;
}

fn rms_scale(x: &[f32], eps: f32) -> f32 {
    let sum: f32 = x.iter().map(|v| v * v).sum();
    1.0 / (sum / x.len() as f32 + eps).sqrt()
}

struct Scalar;

impl VecBackend for Scalar {
    fn name(&self) -> &'static str {
        "scalar"
    }

    fn dot_q8(&self, ty: GgmlType, src: &[u8], _qx: &QuantX, x: &[f32]) -> f32 {
        crate::cpu_moe::dot_scalar(ty, src, x)
    }

    fn rms_norm(&self, x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let scale = rms_scale(x, eps);
        x.iter().zip(w).map(|(&v, &s)| v * scale * s).collect()
    }

    fn quantize_into(&self, x: &[f32], qs: &mut [i8], d: &mut [f32], bsum: &mut [f32]) {
        for (bi, xc) in x.chunks_exact(32).enumerate() {
            let mut amax = 0.0f32;
            let mut sum = 0.0f32;
            for &v in xc {
                let a = v.abs();
                if a > amax {
                    amax = a;
                }
                sum += v;
            }
            let scale = amax / 127.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            d[bi] = scale;
            bsum[bi] = sum;
            for (o, &v) in qs[bi * 32..(bi + 1) * 32].iter_mut().zip(xc.iter()) {
                *o = (v * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn rms_elements(x: &[f32], w: &[f32], out: &mut [f32], scale: f32) {
    use std::arch::aarch64::*;
    unsafe {
        let vs = vdupq_n_f32(scale);
        let chunks = x.len() / 4 * 4;
        for i in (0..chunks).step_by(4) {
            let v = vmulq_f32(vld1q_f32(x[i..].as_ptr()), vs);
            let o = vmulq_f32(v, vld1q_f32(w[i..].as_ptr()));
            vst1q_f32(out[i..].as_mut_ptr(), o);
        }
    }
    for i in x.len() / 4 * 4..x.len() {
        out[i] = x[i] * scale * w[i];
    }
}

#[cfg(target_arch = "x86_64")]
fn rms_elements(x: &[f32], w: &[f32], out: &mut [f32], scale: f32) {
    use std::arch::x86_64::*;
    unsafe {
        let vs = _mm_set1_ps(scale);
        let chunks = x.len() / 4 * 4;
        for i in (0..chunks).step_by(4) {
            let v = _mm_mul_ps(_mm_loadu_ps(x[i..].as_ptr()), vs);
            let o = _mm_mul_ps(v, _mm_loadu_ps(w[i..].as_ptr()));
            _mm_storeu_ps(out[i..].as_mut_ptr(), o);
        }
    }
    for i in x.len() / 4 * 4..x.len() {
        out[i] = x[i] * scale * w[i];
    }
}

fn rms_norm_vectorized(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let scale = rms_scale(x, eps);
    let mut out = vec![0.0f32; x.len()];
    rms_elements(x, w, &mut out, scale);
    out
}

#[cfg(target_arch = "x86_64")]
struct Avx2;

#[cfg(target_arch = "x86_64")]
impl VecBackend for Avx2 {
    fn name(&self) -> &'static str {
        "avx2"
    }

    fn dot_q8(&self, ty: GgmlType, src: &[u8], qx: &QuantX, x: &[f32]) -> f32 {
        // SAFETY: this impl is only selected after avx2_available() returned true.
        unsafe { crate::cpu_moe::avx2_dot_q8(ty, src, qx, x) }
    }

    fn quantize_into(&self, x: &[f32], qs: &mut [i8], d: &mut [f32], bsum: &mut [f32]) {
        // SAFETY: selected only on an AVX2+FMA+F16C host.
        unsafe { crate::cpu_moe::avx2_quantize_into(x, qs, d, bsum) }
    }

    fn rms_norm(&self, x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        rms_norm_vectorized(x, w, eps)
    }
}

#[cfg(target_arch = "aarch64")]
struct Neon;

#[cfg(target_arch = "aarch64")]
impl VecBackend for Neon {
    fn name(&self) -> &'static str {
        "neon"
    }

    fn dot_q8(&self, ty: GgmlType, src: &[u8], qx: &QuantX, x: &[f32]) -> f32 {
        crate::cpu_moe::neon_dot_q8(ty, src, qx).unwrap_or_else(|| Self::fallback(ty, src, qx, x))
    }

    fn quantize_into(&self, x: &[f32], qs: &mut [i8], d: &mut [f32], bsum: &mut [f32]) {
        // SAFETY: NEON is baseline on aarch64 and this impl is aarch64-only.
        unsafe { crate::cpu_moe::neon_quantize_into(x, qs, d, bsum) }
    }

    fn rms_norm(&self, x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        rms_norm_vectorized(x, w, eps)
    }
}

#[cfg(target_arch = "aarch64")]
impl Neon {
    fn fallback(ty: GgmlType, src: &[u8], _qx: &QuantX, x: &[f32]) -> f32 {
        crate::cpu_moe::dot_scalar(ty, src, x)
    }
}

static BACKEND: LazyLock<&'static dyn VecBackend> = LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu_moe::avx2_available() {
        return with_name(&Avx2);
    }
    #[cfg(target_arch = "aarch64")]
    if crate::cpu_moe::neon_available() {
        return with_name(&Neon);
    }
    with_name(&Scalar)
});

fn with_name(backend: &'static dyn VecBackend) -> &'static dyn VecBackend {
    eprintln!("vec backend: {}", backend.name());
    backend
}

#[inline]
pub fn backend() -> &'static dyn VecBackend {
    *BACKEND
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_is_selected() {
        assert!(!backend().name().is_empty());
    }

    #[test]
    fn backend_rms_norm_matches_scalar_bit_exactly() {
        let x: Vec<f32> = (0..1025)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.125)
            .collect();
        let w: Vec<f32> = (0..1025).map(|i| 0.5 + ((i % 11) as f32) * 0.05).collect();
        let expected = Scalar.rms_norm(&x, &w, 1e-5);
        let got = backend().rms_norm(&x, &w, 1e-5);
        assert_eq!(got.len(), expected.len());
        for (a, b) in got.iter().zip(&expected) {
            assert_eq!(a.to_bits(), b.to_bits(), "rms_norm output differs");
        }
    }

    #[test]
    fn backend_quantization_matches_scalar() {
        let x: Vec<f32> = (0..512)
            .map(|i| {
                let s = (i as f32) * 0.017;
                s.sin() * 3.0 + ((i % 13) as f32) * 0.25
            })
            .collect();
        let mut expected = QuantX::new();
        expected.qs.resize(512, 0);
        expected.d.resize(16, 0.0);
        expected.bsum.resize(16, 0.0);
        Scalar.quantize_into(&x, &mut expected.qs, &mut expected.d, &mut expected.bsum);

        let mut got = QuantX::new();
        got.qs.resize(512, 0);
        got.d.resize(16, 0.0);
        got.bsum.resize(16, 0.0);
        backend().quantize_into(&x, &mut got.qs, &mut got.d, &mut got.bsum);

        assert_eq!(got.qs, expected.qs, "quantized codes differ");
        assert_eq!(got.d, expected.d, "scales differ");
        // AVX2 reduces each 32-value block pairwise; scalar walks it in order. The
        // reduction order is part of x86's existing numerics, so require bit-exact
        // block sums only where the backend deliberately preserves scalar order.
        #[cfg(target_arch = "aarch64")]
        assert_eq!(got.bsum, expected.bsum, "block sums differ");
        #[cfg(target_arch = "x86_64")]
        for (a, b) in got.bsum.iter().zip(&expected.bsum) {
            assert!(
                (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                "block sums differ: {a} vs {b}"
            );
        }
    }

    #[test]
    fn scalar_quantization_is_deterministic() {
        let x: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.125).collect();
        let mut a = QuantX::new();
        let mut b = QuantX::new();
        a.qs.resize(128, 0);
        a.d.resize(4, 0.0);
        a.bsum.resize(4, 0.0);
        b.qs.resize(128, 0);
        b.d.resize(4, 0.0);
        b.bsum.resize(4, 0.0);
        Scalar.quantize_into(&x, &mut a.qs, &mut a.d, &mut a.bsum);
        Scalar.quantize_into(&x, &mut b.qs, &mut b.d, &mut b.bsum);
        assert_eq!(a.qs, b.qs);
        assert_eq!(a.d, b.d);
        assert_eq!(a.bsum, b.bsum);
    }
}
