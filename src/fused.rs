//! Fused dequantize-and-dot for the formats that dominate the Inkling-2 APEX checkpoint.
//!
//! The unguided path dequantizes a weight row into f32 scratch and then dot-products it.
//! Here each dequantized value is multiplied by the activation and accumulated immediately,
//! in the same element order, avoiding the scratch round-trip while keeping the exact
//! reduction order. Opt in with `INKLING2_FUSED_DOT=1`; the default path is unchanged.

use crate::quant::GgmlType;
use half::f16;

pub fn supports(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::Q8_0 | GgmlType::Q5_1 | GgmlType::Q5K | GgmlType::Q6K
    )
}

/// Dot one GGUF weight row against `x`, accumulating in element order.
/// Returns `None` for formats without a fused implementation.
pub fn dot_row(ty: GgmlType, row: &[u8], x: &[f32]) -> Option<f32> {
    let (be, bb) = ty.block();
    if x.len() % be != 0 || row.len() < x.len() / be * bb {
        return None;
    }
    let mut sum = 0.0f32;
    match ty {
        GgmlType::Q8_0 => {
            for (xc, b) in x.chunks_exact(32).zip(row.chunks_exact(34)) {
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                for j in 0..32 {
                    sum += d * (b[2 + j] as i8) as f32 * xc[j];
                }
            }
        }
        GgmlType::Q5_1 => {
            for (xc, b) in x.chunks_exact(32).zip(row.chunks_exact(24)) {
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                let m = f16::from_le_bytes([b[2], b[3]]).to_f32();
                for j in 0..32 {
                    let lo = ((b[8 + j / 2] >> (4 * (j % 2))) & 0xF) as u32;
                    let hi = (((b[4 + j / 8] >> (j % 8)) & 1) << 4) as u32;
                    sum += (d * (lo | hi) as f32 - m) * xc[j];
                }
            }
        }
        GgmlType::Q5K => {
            for (xc, b) in x.chunks_exact(256).zip(row.chunks_exact(176)) {
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                let dmin = f16::from_le_bytes([b[2], b[3]]).to_f32();
                let scales = &b[4..16];
                let qh = &b[16..48];
                let qs = &b[48..176];
                for s in 0..8 {
                    let (sc, m) = crate::cpu_moe::scale_min_k4(s, scales);
                    let d1 = d * sc as f32;
                    let m1 = dmin * m as f32;
                    let ql = &qs[32 * (s / 2)..];
                    for l in 0..32 {
                        let lo = (ql[l] >> (4 * (s & 1))) & 0xF;
                        let hi = ((qh[l] >> s) & 1) << 4;
                        sum += (d1 * (lo | hi) as f32 - m1) * xc[s * 32 + l];
                    }
                }
            }
        }
        GgmlType::Q6K => {
            for (xc, b) in x.chunks_exact(256).zip(row.chunks_exact(210)) {
                let ql = &b[0..128];
                let qh = &b[128..192];
                let sc = &b[192..208];
                let d = f16::from_le_bytes([b[208], b[209]]).to_f32();
                // Accumulate in element order, matching dequant-then-dot exactly.
                for j in 0..256 {
                    let n = j / 128;
                    let idx = j % 128;
                    let plane = idx / 32;
                    let l = idx % 32;
                    let (ql, qh, sc) = (&ql[64 * n..], &qh[32 * n..], &sc[8 * n..]);
                    let is = l / 16;
                    let qhv = qh[l];
                    let (q, sidx) = match plane {
                        0 => (((ql[l] & 0xF) | ((qhv & 3) << 4)) as i32 - 32, is),
                        1 => (
                            ((ql[l + 32] & 0xF) | (((qhv >> 2) & 3) << 4)) as i32 - 32,
                            is + 2,
                        ),
                        2 => (((ql[l] >> 4) | (((qhv >> 4) & 3) << 4)) as i32 - 32, is + 4),
                        _ => (
                            ((ql[l + 32] >> 4) | (((qhv >> 6) & 3) << 4)) as i32 - 32,
                            is + 6,
                        ),
                    };
                    sum += d * (sc[sidx] as i8) as f32 * q as f32 * xc[j];
                }
            }
        }
        _ => return None,
    }
    Some(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::dequant;

    #[test]
    fn fused_matches_dequant_then_dot_bit_exactly() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for ty in [GgmlType::Q8_0, GgmlType::Q5_1, GgmlType::Q5K, GgmlType::Q6K] {
            let (be, bb) = ty.block();
            let blocks = 4;
            let row: Vec<u8> = (0..blocks * bb).map(|_| next() as u8).collect();
            let x: Vec<f32> = (0..blocks * be)
                .map(|_| ((next() >> 40) as f32 / (1u64 << 24) as f32) - 0.5)
                .collect();
            let mut tmp = vec![0.0f32; x.len()];
            dequant(ty, &row, &mut tmp);
            let expected: f32 = tmp.iter().zip(&x).map(|(a, b)| a * b).sum();
            let got = dot_row(ty, &row, &x).unwrap();
            assert_eq!(got.to_bits(), expected.to_bits(), "{ty:?} differs");
        }
    }
}
