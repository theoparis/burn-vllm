//! ggml-compatible Q8_K activation quantization and K-quant × Q8_K dot products.
//!
//! Transcribed from ggml's own reference implementations:
//! - `ggml/src/ggml-quants.c`: `quantize_row_q8_K_ref`, `dequantize_row_q8_K`,
//!   `nearest_int`.
//! - `ggml/src/ggml-cpu/quants.c`: `ggml_vec_dot_q4_K_q8_K_generic`,
//!   `ggml_vec_dot_q5_K_q8_K_generic`, `ggml_vec_dot_q6_K_q8_K_generic`.
//!
//! Only one 256-element K-block is handled at a time: callers quantize/slice
//! activations block by block. The dot routines take **raw ggml block bytes**
//! (little-endian f16 scales first for Q4_K/Q5_K, last for Q6_K), so they can be
//! applied directly to mmap-ed GGUF weight data. Byte layouts:
//! - Q4_K: 144 bytes = d(f16) dmin(f16) scales(12) qs(128)
//! - Q5_K: 176 bytes = d(f16) dmin(f16) scales(12) qh(32) qs(128)
//! - Q6_K: 210 bytes = ql(128) qh(64) scales(16) d(f16)
//!
//! All arithmetic is deliberately kept in the same order/types as the C code;
//! the tests compare serialized `Q8K` blocks byte-for-byte with ggml's ref and
//! dot results with ggml's generic vec-dot kernels.

#![allow(dead_code)] // module is being adopted incrementally; tests exercise all items

use anyhow::{Result, bail};
use half::f16;

pub const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12;
const Q4_K_SIZE: usize = 144;
const Q5_K_SIZE: usize = 176;
const Q6_K_SIZE: usize = 210;

/// One ggml `block_q8_K`: f32 scale `d`, 256 signed quants and 16 per-16 sums.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Q8K {
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; QK_K / 16],
}

impl Q8K {
    /// Serialize in ggml's in-memory layout: f32 `d` (LE), `qs`, `bsums` (LE).
    pub fn to_bytes(&self) -> [u8; 4 + QK_K + 2 * (QK_K / 16)] {
        let mut out = [0u8; 4 + QK_K + 2 * (QK_K / 16)];
        out[..4].copy_from_slice(&self.d.to_le_bytes());
        for (i, &q) in self.qs.iter().enumerate() {
            out[4 + i] = q as u8;
        }
        for (i, &b) in self.bsums.iter().enumerate() {
            out[4 + QK_K + 2 * i..4 + QK_K + 2 * i + 2].copy_from_slice(&b.to_le_bytes());
        }
        out
    }

    /// Dequantize to 256 f32 values (`dequantize_row_q8_K` reference).
    pub fn dequantize(&self) -> [f32; QK_K] {
        let mut out = [0.0f32; QK_K];
        for (o, &q) in out.iter_mut().zip(self.qs.iter()) {
            *o = self.d * (q as f32);
        }
        out
    }
}

/// ggml's `nearest_int`: add 2^23 + 2^22 and steal the mantissa bits.
/// This is round-to-nearest-even, unlike Rust's `f32::round` (ties away).
#[inline]
fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12582912.0f32;
    let i = val.to_bits() as i32;
    (i & 0x007fffff) - 0x00400000
}

/// Quantize exactly one 256-element activation block (`quantize_row_q8_K_ref`).
///
/// `iscale = -127/max` is computed over the whole block; `d = 1/iscale` is
/// signed. An all-zero block gets `d = 0` and zero quants/sums.
pub fn quantize_q8_k(x: &[f32]) -> Result<Q8K> {
    if x.len() != QK_K {
        bail!("quantize_q8_k expects {} f32 values, got {}", QK_K, x.len());
    }

    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &v in x {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    if amax == 0.0 {
        return Ok(Q8K {
            d: 0.0,
            qs: [0; QK_K],
            bsums: [0; QK_K / 16],
        });
    }

    // ggml changed this from -128/max so the AVX IQ2_XXS path stays simple.
    let iscale = -127.0f32 / max;
    let mut qs = [0i8; QK_K];
    for (q, &v) in qs.iter_mut().zip(x.iter()) {
        let v = nearest_int(iscale * v);
        *q = if v < 127 { v as i8 } else { 127i8 };
    }
    let mut bsums = [0i16; QK_K / 16];
    for (j, b) in bsums.iter_mut().enumerate() {
        let mut sum = 0i32;
        for &q in &qs[j * 16..j * 16 + 16] {
            sum += q as i32;
        }
        *b = sum as i16;
    }
    Ok(Q8K {
        d: 1.0f32 / iscale,
        qs,
        bsums,
    })
}

/// Quantize any number of whole K-blocks into a row of `Q8K` blocks.
pub fn quantize_q8_k_blocks(x: &[f32]) -> Result<Vec<Q8K>> {
    if x.len() % QK_K != 0 {
        bail!(
            "quantize_q8_k_blocks expects a multiple of {}, got {}",
            QK_K,
            x.len()
        );
    }
    x.chunks_exact(QK_K).map(quantize_q8_k).collect()
}

#[inline]
fn f16_at(b: &[u8], off: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([b[off], b[off + 1]])).to_f32()
}

#[inline]
fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// ggml's `get_scale_min_k4`: unpack the 6-bit scale/min pair `j`.
#[cfg(test)]
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | (q[j] >> 6) << 4,
        )
    }
}

/// Unpack Q4_K/Q5_K's 12 bytes of interleaved 6-bit scales and mins the same
/// way as ggml's generic vec-dot kernels (12 bytes copied into `u32[4]`, then
/// shuffled into 8 scale bytes + 8 min bytes).
#[inline]
fn unpack_scales(scales: &[u8]) -> ([u8; 16], [u8; 8]) {
    // ggml memcpy-s only 12 bytes into u32[4] and overwrites u[3] below.
    let mut u = [0u32; 4];
    for i in 0..3 {
        u[i] = u32_at(scales, 4 * i);
    }
    u[3] = ((u[2] >> 4) & 0x0f0f0f0f) | (((u[1] >> 6) & 0x03030303) << 4);
    let uaux = u[1] & 0x3f3f3f3f;
    u[1] = (u[2] & 0x0f0f0f0f) | (((u[0] >> 6) & 0x03030303) << 4);
    u[2] = uaux;
    u[0] &= 0x3f3f3f3f;

    let mut sc = [0u8; 16];
    let mut mn = [0u8; 8]; // ggml's `mins` starts at byte 8 of utmp
    for (i, v) in sc.iter_mut().enumerate() {
        *v = (u[i / 4] >> (8 * (i % 4))) as u8;
    }
    for (i, v) in mn.iter_mut().enumerate() {
        *v = (u[2 + i / 4] >> (8 * (i % 4))) as u8;
    }
    (sc, mn)
}

/// Accumulate `q8_quant * unpacked q` into eight i32 lanes, 8 elements per lane
/// at a time, exactly as ggml's generic kernels do.
#[inline]
fn accumulate(q8: &[i8], a: &[i8], scales: &[u8; 16], aux32: &mut [i32; 8], groups: usize) {
    for group in 0..groups {
        let base = 32 * group;
        let scale = scales[group] as i32;
        for l in 0..8 {
            aux32[l] += scale * (q8[base + l] as i32 * a[base + l] as i32);
        }
        for l in 0..8 {
            aux32[l] += scale * (q8[base + 8 + l] as i32 * a[base + 8 + l] as i32);
        }
        for l in 0..8 {
            aux32[l] += scale * (q8[base + 16 + l] as i32 * a[base + 16 + l] as i32);
        }
        for l in 0..8 {
            aux32[l] += scale * (q8[base + 24 + l] as i32 * a[base + 24 + l] as i32);
        }
    }
}

/// Q4_K (256 weights) × Q8_K (256 activations), one K-block.
///
/// `w` must hold at least 144 raw ggml block bytes (extra padding is ignored).
pub fn dot_q4_k_q8_k(w: &[u8], q: &Q8K) -> Result<f32> {
    if w.len() < Q4_K_SIZE {
        bail!("Q4_K block needs {} bytes, got {}", Q4_K_SIZE, w.len());
    }
    let w = &w[..Q4_K_SIZE];

    let mut a = [0i8; QK_K];
    for j in 0..QK_K / 64 {
        for l in 0..32 {
            a[64 * j + l] = (w[16 + 32 * j + l] & 0x0f) as i8;
        }
        for l in 0..32 {
            a[64 * j + 32 + l] = (w[16 + 32 * j + l] >> 4) as i8;
        }
    }

    let (scales, mins) = unpack_scales(&w[4..4 + K_SCALE_SIZE]);
    let mut sumi = 0i32;
    for j in 0..QK_K / 16 {
        sumi += q.bsums[j] as i32 * mins[j / 2] as i32;
    }

    let mut aux32 = [0i32; 8];
    accumulate(&q.qs, &a, &scales, &mut aux32, QK_K / 32);
    let d = f16_at(w, 0) * q.d;
    let dmin = f16_at(w, 2) * q.d;

    let mut sums = [0f32; 8];
    for l in 0..8 {
        sums[l] += d * aux32[l] as f32;
    }
    let mut sumf = -dmin * sumi as f32;
    for &s in sums.iter() {
        sumf += s;
    }
    Ok(sumf)
}

/// Q5_K (256 weights) × Q8_K (256 activations), one K-block.
///
/// `w` must hold at least 176 raw ggml block bytes (extra padding is ignored).
pub fn dot_q5_k_q8_k(w: &[u8], q: &Q8K) -> Result<f32> {
    if w.len() < Q5_K_SIZE {
        bail!("Q5_K block needs {} bytes, got {}", Q5_K_SIZE, w.len());
    }
    let w = &w[..Q5_K_SIZE];

    let mut a = [0i8; QK_K];
    let mut m = 1u8;
    for j in 0..QK_K / 64 {
        for l in 0..32 {
            a[64 * j + l] = (w[48 + 32 * j + l] & 0x0f) as i8;
        }
        for l in 0..32 {
            let v = &mut a[64 * j + l];
            if w[16 + l] & m != 0 {
                *v += 16;
            }
        }
        m <<= 1;
        for l in 0..32 {
            a[64 * j + 32 + l] = (w[48 + 32 * j + l] >> 4) as i8;
        }
        for l in 0..32 {
            let v = &mut a[64 * j + 32 + l];
            if w[16 + l] & m != 0 {
                *v += 16;
            }
        }
        m <<= 1;
    }

    let (scales, mins) = unpack_scales(&w[4..4 + K_SCALE_SIZE]);
    let mut sumi = 0i32;
    for j in 0..QK_K / 16 {
        sumi += q.bsums[j] as i32 * mins[j / 2] as i32;
    }

    let mut aux32 = [0i32; 8];
    accumulate(&q.qs, &a, &scales, &mut aux32, QK_K / 32);
    let d = f16_at(w, 0) * q.d;
    let dmin = f16_at(w, 2) * q.d;

    let mut sums = [0f32; 8];
    for l in 0..8 {
        sums[l] += d * aux32[l] as f32;
    }
    let mut sumf = -dmin * sumi as f32;
    for &s in sums.iter() {
        sumf += s;
    }
    Ok(sumf)
}

/// Q6_K (256 weights) × Q8_K (256 activations), one K-block.
///
/// `w` must hold at least 210 raw ggml block bytes (212-byte padded rows are
/// accepted; the two trailing bytes are ignored).
pub fn dot_q6_k_q8_k(w: &[u8], q: &Q8K) -> Result<f32> {
    if w.len() < Q6_K_SIZE {
        bail!("Q6_K block needs {} bytes, got {}", Q6_K_SIZE, w.len());
    }
    let w = &w[..Q6_K_SIZE];

    let mut a = [0i8; QK_K];
    for j in (0..QK_K).step_by(128) {
        for l in 0..32 {
            let ql = w[j / 2 + l];
            let ql2 = w[j / 2 + 32 + l];
            let qh = w[128 + j / 4 + l];
            a[j + l] = ((ql & 0x0f) | ((qh & 0x03) << 4)) as i8 - 32;
            a[j + 32 + l] = ((ql2 & 0x0f) | (((qh >> 2) & 0x03) << 4)) as i8 - 32;
            a[j + 64 + l] = ((ql >> 4) | (((qh >> 4) & 0x03) << 4)) as i8 - 32;
            a[j + 96 + l] = ((ql2 >> 4) | (((qh >> 6) & 0x03) << 4)) as i8 - 32;
        }
    }

    // Q6_K scales are signed i8 (unlike the 6-bit unsigned Q4_K/Q5_K scales).
    let mut scales = [0i8; 16];
    for (s, &b) in scales.iter_mut().zip(&w[192..208]) {
        *s = b as i8;
    }
    let mut aux32 = [0i32; 8];
    for (j, &scale) in scales.iter().enumerate() {
        for l in 0..8 {
            aux32[l] += scale as i32 * (q.qs[16 * j + l] as i32 * a[16 * j + l] as i32);
        }
        for l in 0..8 {
            aux32[l] += scale as i32 * (q.qs[16 * j + 8 + l] as i32 * a[16 * j + 8 + l] as i32);
        }
    }
    let d = f16_at(w, 208) * q.d;

    let mut sums = [0f32; 8];
    for l in 0..8 {
        sums[l] += d * aux32[l] as f32;
    }
    let mut sumf = 0.0f32;
    for &s in sums.iter() {
        sumf += s;
    }
    Ok(sumf)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Q8KCase {
        name: &'static str,
        seed: u64,
        scale: f32,
        mode: i32,
        q8k: &'static str,
        w4: &'static str,
        w5: &'static str,
        w6: &'static str,
        dot4: u32,
        dot5: u32,
        dot6: u32,
    }

    // Generated from ggml's quantize_row_q8_K_ref / ggml_vec_dot_*_K_q8_K_generic.
    // Golden data produced by compiling a C harness against the *actual*
    // ggml/src/ggml-quants.c and ggml/src/ggml-cpu/quants.c (-O2,
    // -ffp-contract=off): quantize_row_q8_K_ref emits the raw 292-byte
    // block_q8_K memory; the three dots come from ggml_vec_dot_*_K_q8_K_generic.
    // Inputs are deterministic (splitmix64) and reproduced by `make_x` below.
    const Q8K_GOLDEN: &[Q8KCase] = &[
        Q8KCase {
            name: "random",
            seed: 0x123456789abcdef0,
            scale: 1.0,
            mode: 0,
            q8k: "f4efa0bdea9109ae4f5b53336a77aeac33af0f83b2250a787af8e66b353cd1984c92bd86627cc2c2477fb76d859f4c4ddec12e1c61444c71aa6e88a0b8b42e537ab26f4abae92a7b8ebe3800fbff5103d0d350f18697f45f48cc596d7635ae413ff897c4b603e74e076b5261b2778a2ee0c4c335e4bcc05ca2ddc00dc77ad2ee0bbcd9757b98e90d397677f51b1f5ca3ebd5c4f8a246df0aaa76d5d517438d18e1c4dc5578ad060c31e80481a51d7a447d124a1c1f9a3c2cd9bb8af0c75cce69ee5ac48a135e96e6d33f615ed13c1e379b4b0982cafefa45bf04f4f2e9b521fde8de455ab030ab7a0f0e26a67645a552cdbd49edd099ce011a53a69a4666886a291c8c4511001700f2007401feff760090001effd90070ff4a011fff9100d1ff600099ff",
            w4: "003000282e013f9d8690ab9ce8d17cfb1c5bf6ee76093f7cd83e0819efb74574cbbbc7e15e87065e91a1871a3397b8c72bb62db14b4c5688eadc3019a433c75918624a82e478333cecf393c8a1d88673eb4d7d75877873ee706b08d5b3c070b9a6cb065950c4821097708d3e3451a49745f1fee47187bfe7c2a8056a047ae48ebf536dca7f486c9ddf6fd44f86412077",
            w5: "00300028b748f3158b2160f233d09bedf193ab44af2ec479893c2a107e11f7148c3d32bf407ee3f4f265d86cc2c61ea554716a19688bce9ec402984b48d5efa3c4f8875455cef89e9e9cd3c46e8dd86cdce846c00818130ea74ffdff523437c52fe4bd3e3660a6b2f6bf59b875035752e4d43e11d7d08f9865fa14fd6f82f352bad0fb540c3205df72b6097635c99141e4874b5f1015ac240c0ba7c330d3c57c32af423a9dbdf3b84784845407764fdf",
            w6: "1a5845ba26808c1cd434215fc34a9179a6a3a37a2c67365f7bdb9e34310124f5b19faad253466f28de2bb09a42eaa82b100de66ef19f751e89ee466d9de9f93847b037e88896537632660a90cb6ba2e41a9b58e7c8f65dfc5796e911b73941b652b838dc8a5846ac6a02213503678a8526f51257bc68273694dd10a7966c47cf6fdac1c4bb1e290dd4eac839ef056d4daf8fc68c8abad12b9a785cfb27c53a784db83848bef49d44100008bb8cc1c33c128a12f3fa634d5c571ac7aee489f78be36d5c3544b540bae3bdeefbef0e7a120030",
            dot4: 0xc456bc0b,
            dot5: 0xc35656ba,
            dot6: 0xc6ca28eb,
        },
        Q8KCase {
            name: "zeros",
            seed: 0xdeadbeefcafebabe,
            scale: 1.0,
            mode: 1,
            q8k: "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
            w4: "003000289fd8fdbe02640a825264f722abf7b165dedfafd16abbd7ce0b5b645dfdd69f975a764a4aff76e40247cb100edcc0874056b3496f1d221ce680924a62c17b60c408d661b4d4f2a3e61f27be38be91550bb32f04be09dfcb4f5ae8cfdfc8cad8f7b8b7419a77028ef875934245938db48d9943f46f23acba0d64a0e2c85c88f677ef39505fd8d8c6600c418dc8",
            w5: "00300028b48c0e09b73b1c5445fbdcd5901bd5d32a8d3cbc966b353df05e3dba74ce65f692cf397de6993b9ad63f8b974fd73d672861928b8efd86ed34c2c26ac65eb243739af57e8d79b3043b05bbd99a70b9e1490cd3ff70962bfc49c344e19b7a1c51ad7ed91521e6a67d496d63c85ee8e62378c90860841f94abe53278335da45e748a133189327da71539995f6549677eebf88eccd241d0644cc810383616de0f9112488454f2c90f81b6c28847",
            w6: "9414bc99213d8d5353cb7ddaa1100b3370f74a56f95260b6803656a50a144022645f984445e211f8732985b1868593733d69254aa788d749377307946272480007d5ef1236f5c17eeeaae88eee0ef8c756c8e57ff0adfa9d7e96609c071137c6c91725f9925ac0d75e9b34b29f9633b6c85a926616894135551106c42d2c83d3f67e00bc7ad30315ce77b84fc829ae46c4ad7a588b70872a596de87a985734ad000f2f336e83935f35a773bea6a5c1c6e9f00f5bff30f820c3cdb0f5d3aaa4d34b94b102546731980e7d13381ef3e8900030",
            dot4: 0x00000000,
            dot5: 0x00000000,
            dot6: 0x00000000,
        },
        Q8KCase {
            name: "negative",
            seed: 0x0123456789abcdef,
            scale: 1.0,
            mode: 2,
            q8k: "abc6a03d91b389ecf5a8b5a684899490e0dced96eebafa9e94c6fae0dafbaf9bf9ae8abfb8fde4d29b9ab3d2aefbd9ccc889afe8e1f7dbe1f9f3d0aafcb594a3e4ff91d28fe599fd94d2ead3b2eaaee9c6f3d286aac8f191abfc82c681b4f7dbe98dad8acb9cc4aafda8dacee9daf49ddcb2a9c2e600fdd68df9d1cd98cfca918b85d3f79cded38cb4d4d1cebcb5d7cbe7fc9bd5aed5b8d7bcc8cea1f2f1dba4b8fbafc8c8c3cdca85ace4abdd93e1a082b7b1b5929398abc8bcd7fade94dadaa79fda8da4dfa8cfd39586e191d5edff8d8793d1b5c9c4e8d7e1acb2d6f3a79397e9cda8a59fe690a4a0a7c7c2cd85ca8bdda7c68fd4c6ed93f5fce0e894c7beecbc8ca021fb83fc5bfc28fd71fc97fb6ffc79fc66fc91fc72fb90fb93fb38fc1ffb4ffc",
            w4: "00300028ca3ea7242455ab6593b4b0d40c408879d6e5ba95ac613c9e0d11bfc05db81c54244ba745157613f8bcbf3efd516a41d2b0e9b922e3399a5e437eb5a4f4633e2ce230724caf4e1e632ad3734e362e306623eabddb8d36354b52fed7fab33c8265e8e9e77c3b7092078c14360cbfcdd0d6c1e3d57e7ff39f5fdc4fb015045f74e4822fb370a530a1556c900c7c",
            w5: "003000287ba3d77ddb372d869d91d422b20b3cdf7a012f9a1931e8500c6f17e58c964c1d75854196b8ecb0e5756c56d00545b08d57d45a3facf91b17b89a961d4a6f3b40072ab77b7fffa9cadb73f453661f155b8dbe67e4337a76664069e398b1079e27b3ab80793d5912b4423ab2eafd97a190b7b7a3e263ac7516c6ef7edd906005a15c3382c95620114320b7897d1bf33d7e13241d43d13d5e61ec2bd7bf138c6f14835f9c93db2a3e2c1c224b9b",
            w6: "6be42abe3f3cad85f3c62d5e00faeb0d820a2d8005d067fe3116dcf81290bd33c1fce87c38a683fde6f8f20eef6a481761290c110c376e1cddfc5bc860a42f22eaef1042fc0eee99e9fc9eaa67dd749c8c8b8c88fc775aa35000ca26a874224018a20af3728727c2f69255cd899c986d16394400fc82450aef6702cc200643a0604d52c4e00b23ee57209ac5fc044a95cb1a8098afad8b6ead61dab6dfe3e352febf681f1a221de0fdf792f325f538105b966fab48cfbda41e78ff0f2f7b07fcab529ba90fb3874d0d147cbeff105d260030",
            dot4: 0xc6edb23c,
            dot5: 0xc7b1e05a,
            dot6: 0xc4cf88f0,
        },
        Q8KCase {
            name: "tiny",
            seed: 0xfedcba9876543210,
            scale: 9.99999968e-21,
            mode: 0,
            q8k: "b874699cd4e2a0601c3c0fb8a501bf0f30e470ac8eee6b8e48677c4edf1dd84ab9e31dec5563812a0e41a48255a10f837bc51122606a3ae6db127e8bdb606cf74d2b36ed4fab738e2e97bdaef3b1b09b746d7902fb1b143049045869b245bbe2ce291d48271928c58d3e58835ceabadf48238416bd601b2fcd0da43547974d728fdaacd81a5432d3b0a4cbe1e47e643b32f1cc44120c31c55d0bc3c346b84e43d342a4efcf0c00222a5ec73168ff13ed3eb3cdf7376a9a03de2b9944aa542db476cc55e68654786e039ed1fffcf460cb41a29c91e5400230d5cf4a342235afd8926aa2e81b40e76b794c4023088bc397bea64eba70dbd74e3925e0ec7d656358d2dd8a2979ffb100d3ff190276ff5801b7ffa4ffa7003900990080005cffddff2e009901",
            w4: "00300028b73d6d3d0e2a1fbdcb7bba4f3ad94c5ef676201f107fc1baccd0c65465c3d714fcbcb67f46f30218d7b11ddee9ae922fc8f8abffaecc43b15dd20f63824ac2c85106635e334d2936fbd0ea5962b83b1bdafda3b4994c9771a940d64b2d6e60243badd51f04483917fedbdb65bf9c14f7c3574700a749b1663ffd0838aa01e893349c0eca3613b5a196c3d897",
            w5: "003000281bb18f86c3f645c7f670755a32fcc51528dbb1d3b33a81ebfb60281b196d21b5f0fb9320ef49167393beb54d6968a3d39db9242660ae3580ee518a6559f896fa4c5800f088d3447801f3dd056048a0ecfdd6144e1db06ec427c7782b7ab8cb4b8bde8682f3ca9f8afb4ac34a7fd7a6cc37d594b63cb2927a9169d794f73ef1819ecb482f52f9bdb3499f95323c21f35ba0bb27ce42bcb7b6dfccbecf1aceebf55ea8ec2b783de3d5076a1405",
            w6: "d09b612aae95f7245686dbdd99a17e12e72c49a4b5014c7f0d5d0fc0e800908391c6bebc589ed1ddabfcbe294c1955d3c212971c85474bca7e8b686a900bd6d4f49607d7c365231f8b5f2569d5e72d04c2eadf4617c70036a910cf433863586f327fe5661b169ca762db016f82e4c9628a767b624590bd3d40cad989ad2c384f3ba50499a7527e7a5aa29db9251e7a8a83b1093e52d7074979693595bc8cf3c86b7239fa0b9d9bd2fda5fce7dadc441921f397fb975d67fb133958ababd4437f7fff0bde2900f322d4c9340df07dfccb0030",
            dot4: 0xa45cbe9c,
            dot5: 0xa4625388,
            dot6: 0xa5415c70,
        },
        Q8KCase {
            name: "huge",
            seed: 0xaaaabbbbccccdddd,
            scale: 1.00000002e+20,
            mode: 0,
            q8k: "d124d8de6e6dfe970c832f05fe8456d8ba89ebc89814b60c340e345e42336e82332704a16133c20ccc7e0b45b3709103b602f0f3b04f0c04314dc33b2508bd307e8ec1e64ddc927d3c9c6bf1a84ddf4c41ef8578c74eca03d3b3e728c5857b239862fb1a5c753e97ecb18e65ebcc290e3247aa40e893a826d711905cc1b9a4b7d176ec0e6bd05c58efe8e0032ce722b895207ef44996ba5893a0fb877279a29f4867c916c2461c1b9191a70aa2b93572d87ed3d5f684b7acc0e01df6afeb523501f465331006370a888601b73e7e3b7653ccc7dcac7289d196645e54fb5f608955812400492b020b4de1daf147cb250c32291f436d000629c64bb80ecff73bf48b90ef68d9fea6004e005800b9006eff870033febd0060ff12ff3eff4c0061007a01daff",
            w4: "003000282fd978010f8ad293f827561d4313cc7235831cbba0807cd74e1188b0456afeaf18b39052a685cb8fc9ffcc2f3ee2296b1dc8a895ee2df38d1d0b1db53479ca564dae722f7091781c0074098ca032c03400247a6ed0a323f380d36e14287e2b303d6deb38f3b17c67e18a1c4a08d58bd68337a14cf799bee49f9a11baf02c82faaf09e4ee8b3cdd28e4608b68",
            w5: "003000286a9e3fdb31ab7afdf6587e6c056d38882786217bbc8931f902258e474bf7faac62a3382019613fada7e988a8cfc9d7f40d2bed7c22e9a242f5691611a77a6e6d27d4f5fd5e0f405fc15f7c053574e89920db92e8b6cd7e7d381adfa2deef845a83a4c9c89d1eb633f0eb35a2d28cb1d404e5998f5d0c93ba14422f59ebed17d3ada115ff0efd2e7c5b1f420de2acb8b74b35ef6f28663a13fb3655fce602af7271c5d065873dd99c856b3b0a",
            w6: "92755a2ba466a24d89172cc791c59882482822995688835a836af6fc3746e34f0264769804fbcaa179321a2bfde8d03c0c7ed62ad1020d167ef2b11b4bbaf5239c5474ee76619de73f252fcda6c3d40bd1d60e0869e0227b76f99d09e0f491aeeff382492df2b4c6e2123563974bb3dd5c4d797972e95b6be8f13143089893f3ec032af03b5800f83a18675493f1650d490859a40405ff6f864cb38d39cfb289099a0b9db1d146193422bb4a4d231a78f919ebb2556ec8674b9640c990342d1c95b810e5165945fc325f35b7917eb7750030",
            dot4: 0xe620d060,
            dot5: 0x65c630b8,
            dot6: 0x66c4ee9a,
        },
        Q8KCase {
            name: "ties",
            seed: 0x5555666677778888,
            scale: 1.0,
            mode: 3,
            q8k: "000080bf0a08080606040402020000fefefcfcfafaf8f8f6f60a08080606040402020000fefefcfcfafaf8f8f6f60a08080606040402020000fefefcfcfafaf8f8f6f60a08080606040402020000fefefcfcfafaf8f8f6f60a08080606040402020000fefefcfcfafaf8f8f6f60a08080606040402020000fefefcfcfafaf8f8f6f60a08080606040402020000fefefcfcfafaf8f8f6f60a08080606040402020000fefefcfcfafaf8f8f6f60a08080606040402020000fefefcfcfafaf8f8f6f60a08080606040402020000fefefcfcfafaf8f8f6f60a08080606040402020000fefefcfcfafaf8f8f6f60a08080606040402020000fefefcfcfafaf8f8f6f60a08088120000800eeffd6ff10000c00f4ffdaff00001200f8ffe0fff0ff1600feff5fff",
            w4: "00300028643d029ab543d3610184fe1736401e93d9e12a1659a5d5fb2cdc37b44bcf30788d54b4c61a7a5418eec81687911afd0f49e51157321d49ce9584a185ca4fd4c514e3b8f57aebd132266076f5736725fb69d6d853518f23028ff3b933805c59fd924926dc723d3d4fc24fc489ac2c22f94790bedc689f357cd518e143a79557e85f08589b2ca9bd9f5ed0757c",
            w5: "00300028f365e2c356bbafb36ec7575f50864d64d23a0daab449051c9cc7942352d6ae0550e01c857c0d0642c7995a04873cbd4639dfc663772b4d1c2b90bae86b25610c9f9b4ed3ce4941b1118e7aca88644d77d2a09323e90ea33b4b85a0c37ae66a31e71e79ad571ae1975d71a6a070a44a35b0691eed8a55f21c837e13c112bd1dd24eed7191a760b6ce23c8525ebb403639cff1c2683be6b5fefbda74f460b1d503a2ff5f1a70c8f0dc227f0d9b",
            w6: "c70a812c8ff4ba15cb458dffac4c8a41ed1e2c4abcb8c5fde902e0ff37b168ab96b390a7890e4befa107c51388fbd0a0d55798ff20432fb0c959d260f767f945be2a12a9663d344de41af98aa31e74f72ec78c034980112dbaee31dad70cce13f2cac4616776ae7df71374c18a8eea1bc54a9cece7014c2a2261da84f353665f766c6bac6d929c79b463435e9f77fcf83145ac220b465fd2b5e1ae3c82d93db60ebf64b1ff79b059bde39e8eb979c3c8388a613ea3a000b93bae2e7b66f5987546cdaf9ba6dca2d0e2976904bbda68ca0030",
            dot4: 0x45d95e40,
            dot5: 0x46312720,
            dot6: 0x46097b00,
        },
        Q8KCase {
            name: "spike",
            seed: 0x9999aaaabbbbcccc,
            scale: -1.0,
            mode: 4,
            q8k: "0402013c81010000ff0001ff01000001010101ff010101ff00000000ff0000ffff00ff000001ff00ff000100ff01010101010000010000010001ff00ff00ff010101ffff0000000001ffffffff000000ff01010100ff01ff00000001010001ff0000ff000001ff0000ffffff01ff00ff01ffff01ffff0100010100000000000100ffffff00000001ff00010000010100ff010001ff0001ff000100ff01010001ffff000000010101010000ff00ffff01ffff0001ffff0000ff00ffffffffff0100000101ff0100010001ff00ff010000ff010001ff00ff00ff01ffff00ffff000001ff00ff01010100ff00010101000100ff0100ffff000101000000ff01ff000100000085fffeff04000100ffff0000fcffffff040000000100fbff0200faff05000000",
            w4: "00300028d3892b223698356510b8e4a165e1cbfac7bbf82f2fca12147cfb3391632b5db91d1ecdc9e09562b2b70797305e6e7c55cec2879bf510e2f87424ef69a106d16d2570e1fdd4574c36daab58fc938b2ef14dd9b2ae024bd61600ea1c5794518ab4d2d8bdaf182f7197f15db2054f9bd634b667973146eccac836c9a5e678842cadeda24bacb1bb56ac0dffd228",
            w5: "00300028de15bdcbd775edc0084e5e09a93aa9676fabc3d546e3fcee69a7e0b3c1995cdf1621340c2aec3cfb6941339fa05486bbbcafc082fce85ad5ad1170061cf58891264c1e6c989cf543ce0d55b3ebec26dafc62533e9a00e92c32e1fdf2de06bc5bb4d73339501023bc9e77c15e454e8e4d924c00764eade8f0e99608550752f0fcf5841dfb99240891dfc9e2e525b61c6d110393420addface9b0fd56a5e45f8e3cb880e5bbdfe9926ac9a7919",
            w6: "b69a94792ae86e588b5f81abb28c9e3c0891b4d74b72b00bb672770c39dc66f4c6ec117e26c61727f5ffaf9e3db20f40e0869d7feef1aff1ef1a1121ea3aa01259bb650f8914684f62b1adf976213ea1518f450080b9ef615909fedb4f5b68b2f9149c93fe5476753441501b41851f3e771886055c1af9a99ec9396c82918dea3f9d5950fe1cbef2c54865b724eec232755e711d78eaa13315e276b69fd16de88962cb96f9deb348acb537faba30b56cfb865d2a72315db0654e2cd9e1ed37d576b0da7fd761af2ba8a1eea333438b160030",
            dot4: 0xc12a66cf,
            dot5: 0xc25548d2,
            dot6: 0xc39e1e1c,
        },
    ];

    fn next_rand(s: &mut u64) -> u64 {
        *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Identical to `make_x` in the C golden-vector generator.
    fn make_x(seed: u64, scale: f32, mode: i32) -> [f32; QK_K] {
        let mut s = seed;
        let mut x = [0.0f32; QK_K];
        for (i, v) in x.iter_mut().enumerate() {
            let u = next_rand(&mut s);
            let mut val = ((u % 2001) as i32 - 1000) as f32 * 0.01f32;
            if mode == 1 {
                val = 0.0;
            } else if mode == 2 {
                val = -val.abs();
            } else if mode == 3 {
                val = (i as i32 % 21 - 10) as f32 + 0.5;
            } else if mode == 4 {
                val = if i == 0 { 1.0 } else { val * 0.001 };
            }
            *v = val * scale;
        }
        if mode == 3 {
            x[255] = 127.0;
        }
        x
    }

    fn unhex(s: &str) -> Vec<u8> {
        assert_eq!(s.len() % 2, 0);
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// Relative tolerance used for the dot comparisons: ggml's generic kernels
    /// accumulate in 8 f32 lanes and then sum; a scalar dequant+dot oracle sums
    /// in a different order, so exact equality is not expected there.
    fn close(got: f32, want: f32, rel: f32) -> bool {
        (got - want).abs() <= rel * want.abs().max(1.0)
    }

    #[test]
    fn quantize_matches_ggml_ref_byte_for_byte() {
        assert_eq!(std::mem::size_of::<Q8K>(), 292);
        for c in Q8K_GOLDEN {
            let x = make_x(c.seed, c.scale, c.mode);
            let q = quantize_q8_k(&x).unwrap();
            let got = q.to_bytes();
            let want = unhex(c.q8k);
            assert_eq!(
                got.as_slice(),
                want.as_slice(),
                "block bytes, case {}",
                c.name
            );
            assert_eq!(
                q.d.to_bits(),
                u32::from_le_bytes(want[..4].try_into().unwrap()),
                "d bits, case {}",
                c.name
            );
            for i in 0..QK_K {
                assert_eq!(q.qs[i] as u8, want[4 + i], "qs[{i}], case {}", c.name);
            }
            for i in 0..QK_K / 16 {
                assert_eq!(
                    q.bsums[i].to_le_bytes(),
                    [want[4 + QK_K + 2 * i], want[5 + QK_K + 2 * i]],
                    "bsums[{i}], case {}",
                    c.name
                );
            }
        }
    }

    #[test]
    fn dots_match_ggml_generic_kernels() {
        for c in Q8K_GOLDEN {
            let x = make_x(c.seed, c.scale, c.mode);
            let q = quantize_q8_k(&x).unwrap();
            let cases = [
                (
                    "q4_K",
                    unhex(c.w4),
                    c.dot4,
                    dot_q4_k_q8_k(&unhex(c.w4), &q).unwrap(),
                ),
                (
                    "q5_K",
                    unhex(c.w5),
                    c.dot5,
                    dot_q5_k_q8_k(&unhex(c.w5), &q).unwrap(),
                ),
                (
                    "q6_K",
                    unhex(c.w6),
                    c.dot6,
                    dot_q6_k_q8_k(&unhex(c.w6), &q).unwrap(),
                ),
            ];
            for (name, _w, bits, got) in cases {
                let want = f32::from_bits(bits);
                assert!(
                    close(got, want, 1e-6),
                    "{name} case {}: got {got:?} ({:08x}), ggml {want:?} ({:08x})",
                    c.name,
                    got.to_bits(),
                    want.to_bits()
                );
            }
        }
    }

    // Scalar oracles transcribed from ggml-quants.c's dequantize_row_q{4,5,6}_K.
    fn dequant4(w: &[u8]) -> [f32; QK_K] {
        let d = f16_at(w, 0);
        let min = f16_at(w, 2);
        let mut out = [0.0f32; QK_K];
        let mut is = 0;
        for g in 0..QK_K / 64 {
            let (sc, m) = get_scale_min_k4(is, &w[4..16]);
            let (d1, m1) = (d * sc as f32, min * m as f32);
            let (sc, m) = get_scale_min_k4(is + 1, &w[4..16]);
            let (d2, m2) = (d * sc as f32, min * m as f32);
            for l in 0..32 {
                out[64 * g + l] = d1 * (w[16 + 32 * g + l] & 0x0f) as f32 - m1;
            }
            for l in 0..32 {
                out[64 * g + 32 + l] = d2 * (w[16 + 32 * g + l] >> 4) as f32 - m2;
            }
            is += 2;
        }
        out
    }

    fn dequant5(w: &[u8]) -> [f32; QK_K] {
        let d = f16_at(w, 0);
        let min = f16_at(w, 2);
        let mut out = [0.0f32; QK_K];
        let mut is = 0;
        let (mut u1, mut u2) = (1u8, 2u8);
        for g in 0..QK_K / 64 {
            let (sc, m) = get_scale_min_k4(is, &w[4..16]);
            let (d1, m1) = (d * sc as f32, min * m as f32);
            let (sc, m) = get_scale_min_k4(is + 1, &w[4..16]);
            let (d2, m2) = (d * sc as f32, min * m as f32);
            for l in 0..32 {
                let q = (w[48 + 32 * g + l] & 0x0f) as f32
                    + if w[16 + l] & u1 != 0 { 16.0 } else { 0.0 };
                out[64 * g + l] = d1 * q - m1;
            }
            for l in 0..32 {
                let q =
                    (w[48 + 32 * g + l] >> 4) as f32 + if w[16 + l] & u2 != 0 { 16.0 } else { 0.0 };
                out[64 * g + 32 + l] = d2 * q - m2;
            }
            u1 <<= 2;
            u2 <<= 2;
            is += 2;
        }
        out
    }

    fn dequant6(w: &[u8]) -> [f32; QK_K] {
        let d = f16_at(w, 208);
        let mut out = [0.0f32; QK_K];
        for n in (0..QK_K).step_by(128) {
            for l in 0..32 {
                let qh = w[128 + n / 4 + l];
                let ql = w[n / 2 + l];
                let ql2 = w[n / 2 + 32 + l];
                let sc = |i: usize| w[192 + n / 16 + i] as i8 as f32;
                out[n + l] =
                    d * sc(l / 16) * (((ql & 0x0f) | ((qh & 0x03) << 4)) as i8 - 32) as f32;
                out[n + 32 + l] = d
                    * sc(l / 16 + 2)
                    * (((ql2 & 0x0f) | (((qh >> 2) & 0x03) << 4)) as i8 - 32) as f32;
                out[n + 64 + l] = d
                    * sc(l / 16 + 4)
                    * (((ql >> 4) | (((qh >> 4) & 0x03) << 4)) as i8 - 32) as f32;
                out[n + 96 + l] = d
                    * sc(l / 16 + 6)
                    * (((ql2 >> 4) | (((qh >> 6) & 0x03) << 4)) as i8 - 32) as f32;
            }
        }
        out
    }

    #[test]
    fn dots_match_scalar_dequant_dot_oracle() {
        for c in Q8K_GOLDEN {
            let x = make_x(c.seed, c.scale, c.mode);
            let q = quantize_q8_k(&x).unwrap();
            let deqx = q.dequantize();
            let pairs = [
                (
                    "q4_K",
                    unhex(c.w4),
                    dequant4(&unhex(c.w4)),
                    dot_q4_k_q8_k(&unhex(c.w4), &q).unwrap(),
                ),
                (
                    "q5_K",
                    unhex(c.w5),
                    dequant5(&unhex(c.w5)),
                    dot_q5_k_q8_k(&unhex(c.w5), &q).unwrap(),
                ),
                (
                    "q6_K",
                    unhex(c.w6),
                    dequant6(&unhex(c.w6)),
                    dot_q6_k_q8_k(&unhex(c.w6), &q).unwrap(),
                ),
            ];
            for (name, _w, wf, got) in pairs {
                let want = wf.iter().zip(deqx.iter()).map(|(a, b)| a * b).sum::<f32>();
                assert!(
                    close(got, want, 1e-5),
                    "{name} case {}: generic {got:?}, scalar dequant+dot {want:?}",
                    c.name
                );
            }
        }
    }

    #[test]
    fn nearest_int_is_round_to_nearest_even() {
        assert_eq!(nearest_int(0.5), 0);
        assert_eq!(nearest_int(1.5), 2);
        assert_eq!(nearest_int(-0.5), 0);
        assert_eq!(nearest_int(-1.5), -2);
        assert_eq!(nearest_int(1.4), 1);
        assert_eq!(nearest_int(-1.6), -2);
    }

    #[test]
    fn multi_block_and_validation() {
        let x = make_x(0x1234_5678_9abc_def0, 1.0, 0);
        let mut row = x.to_vec();
        row.extend_from_slice(&make_x(0xfedc_ba98_7654_3210, -2.5, 0));
        let blocks = quantize_q8_k_blocks(&row).unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0], quantize_q8_k(&x).unwrap());
        assert_eq!(blocks[1], quantize_q8_k(&row[256..]).unwrap());

        assert!(quantize_q8_k(&x[..255]).is_err());
        assert!(quantize_q8_k(&x[1..]).is_err());
        assert!(quantize_q8_k_blocks(&x[..255]).is_err());
        let q = quantize_q8_k(&x).unwrap();
        assert!(dot_q4_k_q8_k(&[0u8; 143], &q).is_err());
        assert!(dot_q5_k_q8_k(&[0u8; 175], &q).is_err());
        assert!(dot_q6_k_q8_k(&[0u8; 209], &q).is_err());
        // Padded Q6_K rows (212 bytes) are accepted by ignoring the padding.
        let padded = [0u8; 212];
        assert!(dot_q6_k_q8_k(&padded, &q).is_ok());
    }

    #[test]
    fn randomized_dots_vs_scalar_oracle() {
        let mut s = 0x1357_9bdf_2468_ace0u64;
        for _ in 0..64 {
            let mode = (next_rand(&mut s) % 5) as i32;
            let sign = if next_rand(&mut s) & 1 == 0 {
                1.0
            } else {
                -1.0
            };
            let q = quantize_q8_k(&make_x(next_rand(&mut s), sign, mode)).unwrap();
            let deqx = q.dequantize();

            let mut w4 = [0u8; 144];
            let mut w5 = [0u8; 176];
            let mut w6 = [0u8; 212]; // padded row, like GGUF Q6_K tensors
            for w in [&mut w4[..], &mut w5[..], &mut w6[..]] {
                for b in w {
                    *b = (next_rand(&mut s) >> 24) as u8;
                }
            }
            for w in [&mut w4[..], &mut w5[..]] {
                let d = f16::from_f32(0.02 + (next_rand(&mut s) % 100) as f32 * 0.01)
                    .to_bits()
                    .to_le_bytes();
                let m = f16::from_f32(0.01 + (next_rand(&mut s) % 100) as f32 * 0.005)
                    .to_bits()
                    .to_le_bytes();
                w[..2].copy_from_slice(&d);
                w[2..4].copy_from_slice(&m);
            }
            let d6 = f16::from_f32(0.02 + (next_rand(&mut s) % 100) as f32 * 0.01)
                .to_bits()
                .to_le_bytes();
            w6[208..210].copy_from_slice(&d6);

            let pairs = [
                ("q4_K", dequant4(&w4), dot_q4_k_q8_k(&w4, &q).unwrap()),
                ("q5_K", dequant5(&w5), dot_q5_k_q8_k(&w5, &q).unwrap()),
                ("q6_K", dequant6(&w6), dot_q6_k_q8_k(&w6, &q).unwrap()),
            ];
            for (name, wf, got) in pairs {
                let want = wf.iter().zip(deqx.iter()).map(|(a, b)| a * b).sum::<f32>();
                assert!(
                    close(got, want, 1e-4),
                    "{name}: generic {got:?}, scalar dequant+dot {want:?}"
                );
            }
        }
    }
}
