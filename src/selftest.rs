//! GPU kernels vs straightforward CPU references, using real tensors from the GGUF for
//! every quant format the model uses.

use crate::cpu_moe::{self, CpuExpertJob};
use crate::gguf::Gguf;
use crate::gpu::{Buf, Gpu, QWeight};
use crate::kernels::{MOE_DESC, MOE_MAXM};
use crate::quant::{self, GgmlType};
use crate::radix::{GdnCheckpoint, KV_PAGE_SIZE, KvPool, RadixTree};
use anyhow::{Result, ensure};
use std::sync::Arc;
use std::time::Instant;

/// Targeted A/B: `SELFTEST_FILTER` keeps only labels containing the substring, and
/// `SELFTEST_M` keeps only those m values (comma separated). Unset means run everything.
fn selected(label: &str) -> bool {
    match std::env::var("SELFTEST_FILTER") {
        Ok(f) => !f.is_empty() && label.contains(&f),
        Err(_) => true,
    }
}

fn selected_m(m: usize) -> bool {
    match std::env::var("SELFTEST_M") {
        Ok(v) => v
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .any(|x| x == m),
        Err(_) => true,
    }
}

fn filtered() -> bool {
    std::env::var_os("SELFTEST_FILTER").is_some() || std::env::var_os("SELFTEST_M").is_some()
}

/// Per-kernel decode-shape timings (device-synchronized averages).
pub fn bench(gpu: &Gpu, g: &Gguf) -> Result<()> {
    let mut rng = Rng(42);
    let time = |label: &str, iters: usize, f: &mut dyn FnMut()| {
        f();
        gpu.sync();
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        gpu.sync();
        let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!("{label:<40} {us:>9.1} us");
        us
    };
    let small = gpu.upload_f32(&rng.vec(2048));
    let small2 = gpu.upload_f32(&rng.vec(2048));
    if selected("launch overhead (add 2048)") {
        time("launch overhead (add 2048)", 2000, &mut || {
            gpu.add_inplace(&small, &small2, 2048)
        });
    }
    for (label, name) in [
        ("qmatmul Q8_0 8192x2048 m=1", "blk.0.attn_qkv.weight"),
        ("qmatmul Q8_0 2048x4096 m=1", "blk.0.ssm_out.weight"),
        (
            "qmatmul Q8_0 32x2048 m=1 (ssm_alpha)",
            "blk.0.ssm_alpha.weight",
        ),
        (
            "qmatmul F32 256x2048 m=1 (router)",
            "blk.0.ffn_gate_inp.weight",
        ),
        (
            "qmatmul Q8_0 512x2048 m=1 (shexp)",
            "blk.0.ffn_up_shexp.weight",
        ),
        ("qmatmul Q8_0 248320x2048 m=1", "output.weight"),
    ] {
        if !selected(label) {
            continue;
        }
        let t = g.tensor(name)?;
        let w = qweight(
            gpu,
            t.ty,
            g.bytes(t),
            t.dims[1] as usize,
            t.dims[0] as usize,
        );
        let x = gpu.upload_f32(&rng.vec(w.k));
        let out = gpu.alloc(w.n);
        let us = time(label, 50, &mut || gpu.qmatmul(&w, &x, 1, &out, false));
        println!("{:<40} {:>9.1} GB/s", "", w.bytes() as f64 / us / 1e3);
        // Native dot on the same mmap'd rows, for A/B against the device kernel.
        let raw = g.bytes(t);
        let rb = t.ty.row_bytes(w.k);
        let xn = rng.vec(w.k);
        time(&format!("  cpu_moe::dot {label} m=1"), 200, &mut || {
            let sum: f32 = (0..w.n)
                .map(|r| cpu_moe::dot(t.ty, &raw[r * rb..(r + 1) * rb], &xn))
                .sum();
            std::hint::black_box(sum);
        });
        // Prefill crossover between the streamed kernel and dequant + tiled GEMM.
        if w.n < 100_000 {
            for m in [32usize, 128] {
                if !selected_m(m) {
                    continue;
                }
                let x = gpu.upload_f32(&rng.vec(m * w.k));
                let o = gpu.alloc(m * w.n);
                time(&format!("  qmatmul m={m}"), 10, &mut || {
                    gpu.qmatmul(&w, &x, m, &o, false)
                });
                time(&format!("  dequant + gemm m={m}"), 10, &mut || {
                    drop(gpu.linear_gemm(&w, &x, m))
                });
            }
        }
    }
    if filtered() {
        // A targeted bench run only measures the selected qmatmul cases above.
        return Ok(());
    }
    let h = gpu.upload_f32(&rng.vec(2048));
    let nw = gpu.upload_f32(&rng.vec(2048));
    time("rmsnorm 2048", 200, &mut || {
        drop(gpu.rmsnorm(&h, &nw, 1, 2048, 1e-6))
    });
    {
        let (nk, nv, dk) = (16usize, 32usize, 128usize);
        let c = 2 * nk * dk + nv * dk;
        let conv = gpu.upload_f32(&rng.vec(c));
        let st = gpu.zeros(3 * c);
        let w = gpu.upload_f32(&rng.vec(4 * c));
        time("gdn conv t=1", 200, &mut || {
            drop(gpu.gdn_conv(&conv, &w, &st, None, c, 1))
        });
        let a = gpu.upload_f32(&rng.vec(nv));
        let z = gpu.upload_f32(&rng.vec(nv * dk));
        let s = gpu.zeros(nv * dk * dk);
        let nw = gpu.upload_f32(&rng.vec(dk));
        time("gdn recurrent t=1", 200, &mut || {
            drop(gpu.gdn_recurrent(
                &conv, &a, &a, &z, &a, &a, &nw, &s, None, 1, nk, nv, dk, 1e-6,
            ))
        });
    }
    {
        let (nh, nkv, hd, ctx) = (16usize, 2usize, 256usize, 1024usize);
        let kc = gpu.upload_f32(&rng.vec(ctx * nkv * hd));
        let vc = gpu.upload_f32(&rng.vec(ctx * nkv * hd));
        let q = gpu.upload_f32(&rng.vec(nh * hd));
        let qg = gpu.upload_f32(&rng.vec(nh * hd * 2));
        for pos in [128usize, 1000] {
            time(&format!("attention t=1 pos={pos}"), 100, &mut || {
                drop(gpu.attention(&q, &qg, &kc, &vc, 1, nh, nkv, hd, pos))
            });
        }
    }
    {
        // 8 single-pair tiles over resident experts 0..8 of layer 0.
        let mut pools = Vec::new();
        for name in [
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        ] {
            let t = g.tensor(&format!("blk.0.{name}"))?;
            let per = t.nbytes / 256;
            let p = quant::repack(t.ty, &g.bytes(t)[..8 * per]);
            pools.push((p.fmt, gpu.upload_u32(&p.w0), gpu.upload_u32(&p.w1)));
        }
        let ids = gpu.upload_u32(&(0..8).collect::<Vec<u32>>());
        let desc = gpu.moe_direct_desc(&ids, 8);
        let x = gpu.upload_f32(&rng.vec(2048));
        let act = gpu.alloc(8 * 512);
        let y = gpu.alloc(8 * 2048);
        let wq = QWeight {
            fmt: pools[0].0,
            n: 8 * 512,
            k: 2048,
            w0: pools[0].1.clone(),
            w1: pools[0].2.clone(),
        };
        let o = gpu.alloc(8 * 512);
        let us = time("qmatmul MXFP4 4096x2048 m=1", 200, &mut || {
            gpu.qmatmul(&wq, &x, 1, &o, false)
        });
        println!("{:<40} {:>9.1} GB/s", "", wq.bytes() as f64 / us / 1e3);
        let wd = QWeight {
            fmt: pools[2].0,
            n: 8 * 2048,
            k: 512,
            w0: pools[2].1.clone(),
            w1: pools[2].2.clone(),
        };
        let xd = gpu.upload_f32(&rng.vec(512));
        let od = gpu.alloc(8 * 2048);
        let us = time("qmatmul Q5_K 16384x512 m=1", 200, &mut || {
            gpu.qmatmul(&wd, &xd, 1, &od, false)
        });
        println!("{:<40} {:>9.1} GB/s", "", wd.bytes() as f64 / us / 1e3);
        time("moe gate_up 8 pairs (MXFP4)", 200, &mut || {
            gpu.moe_gate_up(
                (&pools[0].1, &pools[0].2),
                (&pools[1].1, &pools[1].2),
                pools[0].0,
                &x,
                &desc,
                8,
                &act,
                512,
                2048,
                8,
                1,
            )
        });
        time("moe down 8 pairs (Q5_K)", 200, &mut || {
            gpu.moe_down(
                (&pools[2].1, &pools[2].2),
                pools[2].0,
                &act,
                &desc,
                8,
                &y,
                2048,
                512,
                1,
            )
        });
    }
    // Expert refill path: repack (host) and repack + enqueue + DMA (synced).
    let names = [
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
    ];
    let ts: Vec<_> = names.iter().map(|n| g.tensor(n)).collect::<Result<_>>()?;
    let mut e = 0usize;
    let repack_all = |e: usize| -> Vec<quant::Packed> {
        ts.iter()
            .map(|t| {
                let per = t.nbytes / 256;
                quant::repack(t.ty, &g.bytes(t)[e * per..(e + 1) * per])
            })
            .collect()
    };
    time("expert repack (host only)", 50, &mut || {
        e = (e + 1) % 256;
        std::hint::black_box(repack_all(e));
    });
    for n_exp in [1usize, 2, 3, 4] {
        let x = rng.vec(2048);
        let mut out = vec![0.0f32; 2048];
        let jobs: Vec<CpuExpertJob<'_>> = (0..n_exp)
            .map(|i| {
                let raw = |ti: usize| {
                    let per = ts[ti].nbytes / 256;
                    &g.bytes(ts[ti])[i * per..(i + 1) * per]
                };
                CpuExpertJob {
                    gate_raw: raw(0),
                    up_raw: raw(1),
                    down_raw: raw(2),
                    pairs: vec![(0, 0.25)],
                }
            })
            .collect();
        let us = time(&format!("cpu_moe x{n_exp} (MXFP4+Q5_K)"), 50, &mut || {
            cpu_moe::run_experts_cpu(
                ts[0].ty, ts[1].ty, ts[2].ty, &jobs, &x, 1, 2048, 512, &mut out,
            );
            std::hint::black_box(&out);
        });
        let bytes: usize = ts.iter().map(|t| t.nbytes / 256).sum::<usize>() * n_exp;
        println!("{:<40} {:>9.1} GB/s", "", bytes as f64 / us / 1e3);
    }
    let pools: Vec<Buf> = ts
        .iter()
        .map(|t| gpu.alloc(t.nbytes / 4 / 256 * 64))
        .collect();
    for batch in [16usize, 64] {
        let us = time(
            &format!("expert repack + upload x{batch} (synced)"),
            5,
            &mut || {
                for i in 0..batch {
                    e = (e + 1) % 256;
                    for (p, pool) in repack_all(e).into_iter().zip(&pools) {
                        gpu.write_at(pool, i * p.w0.len(), p.w0);
                    }
                }
                gpu.sync();
            },
        );
        let bytes: usize = ts.iter().map(|t| t.nbytes / 256).sum::<usize>() * batch;
        println!("{:<40} {:>9.1} GB/s", "", bytes as f64 / us / 1e3);
    }
    time("sync only", 100, &mut || gpu.sync());
    let big = gpu.alloc(8 << 20);
    for mb in [1usize, 4, 16, 32] {
        let words = mb << 18;
        let data = vec![7u32; words];
        let us = time(&format!("write {mb} MiB (synced)"), 10, &mut || {
            gpu.write_at(&big, 0, data.clone());
            gpu.sync();
        });
        println!("{:<40} {:>9.1} GB/s", "", (words * 4) as f64 / us / 1e3);
    }
    Ok(())
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f()).collect()
    }
}

fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
}

fn check(name: &str, got: &[f32], want: &[f32], tol: f32) -> Result<()> {
    ensure!(
        got.len() == want.len(),
        "{name}: length {} vs {}",
        got.len(),
        want.len()
    );
    let e = max_rel_err(got, want);
    let ok = e <= tol && got.iter().all(|v| v.is_finite());
    println!(
        "{} {name}: max rel err {e:.2e}",
        if ok { "ok  " } else { "FAIL" }
    );
    ensure!(ok, "{name} mismatch");
    Ok(())
}

/// The first `rows` rows of expert `e` (or of a 2-D tensor when `e` is None).
fn slice<'a>(
    g: &'a Gguf,
    name: &str,
    e: Option<usize>,
    rows: usize,
) -> Result<(GgmlType, usize, &'a [u8])> {
    let t = g.tensor(name)?;
    let k = t.dims[0] as usize;
    let row_bytes = t.ty.row_bytes(k);
    let base = match e {
        Some(e) => e * (t.dims[1] as usize) * row_bytes,
        None => 0,
    };
    Ok((t.ty, k, &g.bytes(t)[base..base + rows * row_bytes]))
}

fn cpu_matmul(w: &[f32], x: &[f32], n: usize, k: usize, m: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for mi in 0..m {
        for r in 0..n {
            out[mi * n + r] = (0..k).map(|i| w[r * k + i] * x[mi * k + i]).sum();
        }
    }
    out
}

fn qweight(gpu: &Gpu, ty: GgmlType, raw: &[u8], n: usize, k: usize) -> QWeight {
    let p = quant::repack(ty, raw);
    QWeight {
        fmt: p.fmt,
        n,
        k,
        w0: gpu.upload_u32(&p.w0),
        w1: gpu.upload_u32(&p.w1),
    }
}

pub fn run(gpu: &Gpu, g: &Gguf) -> Result<()> {
    let mut rng = Rng(0x1234_5678);
    let cases = [
        ("F32 router", "blk.0.ffn_gate_inp.weight", None, 256),
        // Odd row count: GEMM outputs whose rows the runtime may pitch.
        ("Q8_0 attn_qkv", "blk.0.attn_qkv.weight", None, 257),
        (
            "MXFP4 gate_exps",
            "blk.0.ffn_gate_exps.weight",
            Some(7),
            512,
        ),
        ("Q5_K down_exps", "blk.0.ffn_down_exps.weight", Some(3), 384),
        (
            "Q5_K gate_exps",
            "blk.10.ffn_gate_exps.weight",
            Some(1),
            128,
        ),
        (
            "Q6_K down_exps",
            "blk.10.ffn_down_exps.weight",
            Some(5),
            384,
        ),
        ("BF16 output", "output.weight", None, 257),
        ("Q4_K attn_qkv", "blk.0.attn_qkv.weight", None, 257),
        ("IQ2_S ffn_gate", "blk.0.ffn_gate.weight", None, 257),
        ("IQ3_S ffn_down", "blk.0.ffn_down.weight", None, 129),
        ("exps gate L1", "blk.1.ffn_gate_exps.weight", Some(2), 257),
        ("exps down L1", "blk.1.ffn_down_exps.weight", Some(2), 257),
    ];
    for (label, name, e, rows) in cases {
        if !selected(label) || g.tensor(name).is_err() {
            continue;
        }
        let (ty, k, raw) = slice(g, name, e, rows)?;
        let mut wf = vec![0f32; rows * k];
        quant::dequant(ty, raw, &mut wf);
        let w = qweight(gpu, ty, raw, rows, k);
        check(
            &format!("{label} dequant"),
            &gpu.read_f32(&gpu.dequant(&w)),
            &wf,
            1e-6,
        )?;
        {
            let x = rng.vec(k);
            let want = cpu_matmul(&wf, &x, rows, k, 1);
            let rb = ty.row_bytes(k);
            let got: Vec<f32> = (0..rows)
                .map(|r| cpu_moe::dot(ty, &raw[r * rb..(r + 1) * rb], &x))
                .collect();
            check(&format!("{label} cpu_moe::dot"), &got, &want, 1e-2)?;
        }
        for m in [1usize, 3, 8, 13, 40] {
            if !selected_m(m) {
                continue;
            }
            let x = rng.vec(m * k);
            let want = cpu_matmul(&wf, &x, rows, k, m);
            let xb = gpu.upload_f32(&x);
            let out = gpu.alloc(m * rows);
            gpu.qmatmul(&w, &xb, m, &out, false);
            check(
                &format!("{label} qmatmul m={m}"),
                &gpu.read_f32(&out),
                &want,
                1e-4,
            )?;
            if m >= 8 {
                let o = gpu.linear_gemm(&w, &xb, m);
                // tf32 tensor cores may be used by the tiled GEMM.
                check(
                    &format!("{label} gemm m={m}"),
                    &gpu.read_f32(&o),
                    &want,
                    5e-3,
                )?;
            }
        }
    }
    if !filtered() {
        slot_write_test(gpu)?;
        if g.tensor("blk.0.ffn_gate_exps.weight").is_ok() {
            moe_test(gpu, g, &mut rng)?;
        }
        gdn_test(gpu, &mut rng)?;
        attn_test(gpu, &mut rng)?;
        argmax_test(gpu, &mut rng)?;
        paged_attn_test(gpu, &mut rng)?;
        radix_test(gpu)?;
    }
    println!("selftest passed");
    Ok(())
}

/// Partial writes into a slot pool must leave every other slot intact (wgpu copies the
/// whole bound range of the destination handle).
fn slot_write_test(gpu: &Gpu) -> Result<()> {
    let (per, slots) = (1usize << 12, 6usize);
    let pool = gpu.alloc(per * slots);
    for s in 0..slots {
        gpu.write_at(&pool, s * per, vec![(1.0 + s as f32).to_bits(); per]);
    }
    gpu.write_at(&pool, 2 * per, vec![100f32.to_bits(); per]);
    let w = QWeight {
        fmt: quant::fmt::F32,
        n: slots,
        k: per,
        w0: pool,
        w1: gpu.zeros(1),
    };
    let want: Vec<f32> = (0..slots * per)
        .map(|i| {
            if i / per == 2 {
                100.0
            } else {
                1.0 + (i / per) as f32
            }
        })
        .collect();
    check(
        "slot pool partial writes",
        &gpu.read_f32(&gpu.dequant(&w)),
        &want,
        0.0,
    )
}

/// Grouped expert kernels against CPU, with experts written into non-zero pool slots.
fn moe_test(gpu: &Gpu, g: &Gguf, rng: &mut Rng) -> Result<()> {
    // Fused shared-expert gate+up+SwiGLU against CPU reference.
    {
        let gt = g.tensor("blk.0.ffn_gate_shexp.weight")?;
        let ut = g.tensor("blk.0.ffn_up_shexp.weight")?;
        let (k, n) = (gt.dims[0] as usize, gt.dims[1] as usize);
        let gw = qweight(gpu, gt.ty, g.bytes(gt), n, k);
        let uw = qweight(gpu, ut.ty, g.bytes(ut), n, k);
        let mut g_cpu = vec![0f32; n * k];
        let mut u_cpu = vec![0f32; n * k];
        quant::dequant(gt.ty, g.bytes(gt), &mut g_cpu);
        quant::dequant(ut.ty, g.bytes(ut), &mut u_cpu);
        for m in [1usize, 2, 3] {
            let x = rng.vec(m * k);
            let xb = gpu.upload_f32(&x);
            let got = gpu.linear_swiglu(&gw, &uw, &xb, m, 80);
            let g_out = cpu_matmul(&g_cpu, &x, n, k, m);
            let u_out = cpu_matmul(&u_cpu, &x, n, k, m);
            let want: Vec<f32> = g_out
                .iter()
                .zip(&u_out)
                .map(|(&gv, &uv)| gv / (1.0 + (-gv).exp()) * uv)
                .collect();
            check(
                &format!("linear_swiglu Q8_0 m={m}"),
                &gpu.read_f32(&got),
                &want,
                1e-4,
            )?;
        }
    }
    for layer in [0usize, 10] {
        let (inter, hidden, topk) = (512usize, 2048usize, 8usize);
        let experts = [5usize, 77, 200];
        let slots = 4;
        let load = |name: &str, e: usize| -> Result<(GgmlType, Vec<u8>)> {
            let t = g.tensor(&format!("blk.{layer}.{name}"))?;
            let per = t.nbytes / 256;
            Ok((t.ty, g.bytes(t)[e * per..(e + 1) * per].to_vec()))
        };
        let mut pools = Vec::new();
        let mut cpu_w = Vec::new();
        for name in [
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        ] {
            let (ty, _) = load(name, 0)?;
            let f = ty.gpu_fmt();
            let numel = inter * hidden;
            let w0p = quant::fmt::w0_words(f, numel);
            let w1p = if matches!(f, quant::fmt::Q8_0 | quant::fmt::MXFP4) {
                quant::fmt::w1_words(f, numel)
            } else {
                0
            };
            let w0 = gpu.zeros(w0p * slots);
            let w1 = match ty.codebook() {
                Some(cb) => gpu.upload_u32(&cb),
                None => gpu.zeros((w1p * slots).max(1)),
            };
            let mut per_expert = Vec::new();
            for (i, &e) in experts.iter().enumerate() {
                let (ty, raw) = load(name, e)?;
                let p = quant::repack(ty, &raw);
                gpu.write_at(&w0, (i + 1) * w0p, p.w0);
                if w1p > 0 {
                    gpu.write_at(&w1, (i + 1) * w1p, p.w1);
                }
                let mut wf = vec![0f32; numel];
                quant::dequant(ty, &raw, &mut wf);
                per_expert.push(wf);
            }
            pools.push((f, w0, w1));
            cpu_w.push(per_expert);
        }
        // 3 tokens x topk pairs; pair p uses expert (p % 3).
        let t = 3;
        let npairs = t * topk;
        let x = rng.vec(t * hidden);
        let mut groups: Vec<Vec<u32>> = vec![Vec::new(); experts.len()];
        for p in 0..npairs {
            groups[p % 3].push(p as u32);
        }
        let xb = gpu.upload_f32(&x);
        let (gp, up, dp) = (
            (&pools[0].1, &pools[0].2),
            (&pools[1].1, &pools[1].2),
            (&pools[2].1, &pools[2].2),
        );
        let mut want_act = vec![0f32; npairs * inter];
        let mut want_y = vec![0f32; npairs * hidden];
        for p in 0..npairs {
            let ei = p % 3;
            let xt = &x[(p / topk) * hidden..(p / topk + 1) * hidden];
            let gte = cpu_matmul(&cpu_w[0][ei], xt, inter, hidden, 1);
            let upe = cpu_matmul(&cpu_w[1][ei], xt, inter, hidden, 1);
            for j in 0..inter {
                let gv = gte[j];
                want_act[p * inter + j] = gv / (1.0 + (-gv).exp()) * upe[j];
            }
            let yo = cpu_matmul(
                &cpu_w[2][ei],
                &want_act[p * inter..(p + 1) * inter],
                hidden,
                inter,
                1,
            );
            want_y[p * hidden..(p + 1) * hidden].copy_from_slice(&yo);
        }
        // Host-grouped tiles at every kernel tile width.
        for width in [2usize, 3, 4, MOE_MAXM] {
            let mut desc = Vec::new();
            for (i, grp) in groups.iter().enumerate() {
                for chunk in grp.chunks(width) {
                    desc.push((i + 1) as u32);
                    desc.push(chunk.len() as u32);
                    desc.extend((0..MOE_MAXM).map(|j| chunk.get(j).copied().unwrap_or(0)));
                }
            }
            let ntiles = desc.len() / MOE_DESC;
            let desc = gpu.upload_u32(&desc);
            let act = gpu.alloc(npairs * inter);
            let y = gpu.alloc(npairs * hidden);
            gpu.moe_gate_up(
                gp, up, pools[0].0, &xb, &desc, ntiles, &act, inter, hidden, topk, width,
            );
            gpu.moe_down(
                dp, pools[2].0, &act, &desc, ntiles, &y, hidden, inter, width,
            );
            check(
                &format!("moe gate_up layer {layer} tiles<={width}"),
                &gpu.read_f32(&act),
                &want_act,
                1e-4,
            )?;
            check(
                &format!("moe down layer {layer} tiles<={width}"),
                &gpu.read_f32(&y),
                &want_y,
                1e-4,
            )?;
        }
        // Decode-style path: device-built single-pair tiles (slot = routed id).
        let ids = gpu.upload_u32(&(0..npairs).map(|p| (p % 3 + 1) as u32).collect::<Vec<_>>());
        let ddesc = gpu.moe_direct_desc(&ids, npairs);
        let act1 = gpu.alloc(npairs * inter);
        let y1 = gpu.alloc(npairs * hidden);
        gpu.moe_gate_up(
            gp, up, pools[0].0, &xb, &ddesc, npairs, &act1, inter, hidden, topk, 1,
        );
        gpu.moe_down(dp, pools[2].0, &act1, &ddesc, npairs, &y1, hidden, inter, 1);
        check(
            &format!("moe direct gate_up layer {layer}"),
            &gpu.read_f32(&act1),
            &want_act,
            1e-4,
        )?;
        check(
            &format!("moe direct down layer {layer}"),
            &gpu.read_f32(&y1),
            &want_y,
            1e-4,
        )?;
    }
    Ok(())
}

fn gdn_test(gpu: &Gpu, rng: &mut Rng) -> Result<()> {
    let (nk, nv, dk, t) = (4usize, 8usize, 128usize, 5usize);
    let c = 2 * nk * dk + nv * dk;
    let eps = 1e-6f32;
    let x = rng.vec(t * c);
    let conv_w = rng.vec(c * 4);
    let state0 = rng.vec(3 * c);
    let alpha = rng.vec(t * nv);
    let beta = rng.vec(t * nv);
    let z = rng.vec(t * nv * dk);
    let ssm_a: Vec<f32> = rng.vec(nv).iter().map(|v| -v.abs() - 0.1).collect();
    let dt_bias = rng.vec(nv);
    let norm_w = rng.vec(dk);
    let s0: Vec<f32> = rng.vec(nv * dk * dk).iter().map(|v| v * 0.1).collect();

    // CPU conv
    let silu = |v: f32| v / (1.0 + (-v).exp());
    let mut conv = vec![0f32; t * c];
    let mut cst = state0.clone();
    for ch in 0..c {
        let mut h = [cst[ch], cst[c + ch], cst[2 * c + ch]];
        for ti in 0..t {
            let v = x[ti * c + ch];
            let w = &conv_w[ch * 4..ch * 4 + 4];
            conv[ti * c + ch] = silu(w[0] * h[0] + w[1] * h[1] + w[2] * h[2] + w[3] * v);
            h = [h[1], h[2], v];
        }
        cst[ch] = h[0];
        cst[c + ch] = h[1];
        cst[2 * c + ch] = h[2];
    }
    // CPU recurrence
    let mut s = s0.clone();
    let mut out = vec![0f32; t * nv * dk];
    for ti in 0..t {
        let row = &conv[ti * c..(ti + 1) * c];
        for h in 0..nv {
            let kh = h % nk;
            let l2 = |v: &[f32]| {
                let n = v.iter().map(|a| a * a).sum::<f32>();
                v.iter().map(|a| a / (n + 1e-6).sqrt()).collect::<Vec<_>>()
            };
            let q: Vec<f32> = l2(&row[kh * dk..(kh + 1) * dk])
                .iter()
                .map(|v| v / (dk as f32).sqrt())
                .collect();
            let k = l2(&row[nk * dk + kh * dk..nk * dk + (kh + 1) * dk]);
            let v = &row[2 * nk * dk + h * dk..2 * nk * dk + (h + 1) * dk];
            let av = alpha[ti * nv + h] + dt_bias[h];
            let g = (ssm_a[h] * (1.0 + av.exp()).ln()).exp();
            let b = 1.0 / (1.0 + (-beta[ti * nv + h]).exp());
            let sh = &mut s[h * dk * dk..(h + 1) * dk * dk];
            sh.iter_mut().for_each(|v| *v *= g);
            let mut o = vec![0f32; dk];
            for j in 0..dk {
                let kv: f32 = (0..dk).map(|i| sh[i * dk + j] * k[i]).sum();
                let delta = (v[j] - kv) * b;
                for i in 0..dk {
                    sh[i * dk + j] += k[i] * delta;
                }
                o[j] = (0..dk).map(|i| sh[i * dk + j] * q[i]).sum();
            }
            let ms = o.iter().map(|a| a * a).sum::<f32>() / dk as f32;
            for j in 0..dk {
                out[ti * nv * dk + h * dk + j] =
                    o[j] / (ms + eps).sqrt() * norm_w[j] * silu(z[ti * nv * dk + h * dk + j]);
            }
        }
    }
    let state_b = gpu.upload_f32(&state0);
    let conv_b = gpu.gdn_conv(
        &gpu.upload_f32(&x),
        &gpu.upload_f32(&conv_w),
        &state_b,
        None,
        c,
        t,
    );
    check("gdn conv", &gpu.read_f32(&conv_b), &conv, 1e-5)?;
    check("gdn conv state", &gpu.read_f32(&state_b), &cst, 1e-6)?;
    let ssm_b = gpu.upload_f32(&s0);
    let o = gpu.gdn_recurrent(
        &conv_b,
        &gpu.upload_f32(&alpha),
        &gpu.upload_f32(&beta),
        &gpu.upload_f32(&z),
        &gpu.upload_f32(&ssm_a),
        &gpu.upload_f32(&dt_bias),
        &gpu.upload_f32(&norm_w),
        &ssm_b,
        None,
        t,
        nk,
        nv,
        dk,
        eps,
    );
    check("gdn recurrent out", &gpu.read_f32(&o), &out, 1e-4)?;
    check("gdn recurrent state", &gpu.read_f32(&ssm_b), &s, 1e-4)?;
    // Rollback snapshots: the state saved after token j must equal running tokens 0..=j alone.
    let csnap = gpu.alloc((t - 1) * 3 * c);
    let ssnap = gpu.alloc((t - 1) * nv * dk * dk);
    let (xb, wb) = (gpu.upload_f32(&x), gpu.upload_f32(&conv_w));
    let conv_s = gpu.gdn_conv(
        &xb,
        &wb,
        &gpu.upload_f32(&state0),
        Some((&csnap, t - 1)),
        c,
        t,
    );
    check("gdn conv (snap)", &gpu.read_f32(&conv_s), &conv, 1e-5)?;
    let _ = gpu.gdn_recurrent(
        &conv_s,
        &gpu.upload_f32(&alpha),
        &gpu.upload_f32(&beta),
        &gpu.upload_f32(&z),
        &gpu.upload_f32(&ssm_a),
        &gpu.upload_f32(&dt_bias),
        &gpu.upload_f32(&norm_w),
        &gpu.upload_f32(&s0),
        Some((&ssnap, t - 1)),
        t,
        nk,
        nv,
        dk,
        eps,
    );
    let (csnap_h, ssnap_h) = (gpu.read_f32(&csnap), gpu.read_f32(&ssnap));
    for n in 1..t {
        let stn = gpu.upload_f32(&state0);
        let cn = gpu.gdn_conv(&gpu.upload_f32(&x[..n * c]), &wb, &stn, None, c, n);
        let ssn = gpu.upload_f32(&s0);
        let _ = gpu.gdn_recurrent(
            &cn,
            &gpu.upload_f32(&alpha[..n * nv]),
            &gpu.upload_f32(&beta[..n * nv]),
            &gpu.upload_f32(&z[..n * nv * dk]),
            &gpu.upload_f32(&ssm_a),
            &gpu.upload_f32(&dt_bias),
            &gpu.upload_f32(&norm_w),
            &ssn,
            None,
            n,
            nk,
            nv,
            dk,
            eps,
        );
        let cs = 3 * c;
        let ss = nv * dk * dk;
        check(
            &format!("gdn conv snapshot {n}"),
            &csnap_h[(n - 1) * cs..n * cs],
            &gpu.read_f32(&stn),
            1e-6,
        )?;
        check(
            &format!("gdn recurrent snapshot {n}"),
            &ssnap_h[(n - 1) * ss..n * ss],
            &gpu.read_f32(&ssn),
            1e-6,
        )?;
    }
    Ok(())
}

fn attn_test(gpu: &Gpu, rng: &mut Rng) -> Result<()> {
    let (nh, nkv, hd, rope_dim, theta, eps) =
        (4usize, 2usize, 256usize, 64usize, 1.0e7f32, 1e-6f32);
    let max_ctx = 512;
    // Two calls: a 37-token prefill then a 1-token decode at pos 37.
    let kc = gpu.zeros(max_ctx * nkv * hd);
    let vc = gpu.zeros(max_ctx * nkv * hd);
    let q_norm = rng.vec(hd);
    let k_norm = rng.vec(hd);
    let qn_b = gpu.upload_f32(&q_norm);
    let kn_b = gpu.upload_f32(&k_norm);
    let mut cpu_k = vec![0f32; max_ctx * nkv * hd];
    let mut cpu_v = vec![0f32; max_ctx * nkv * hd];
    let norm_rope = |x: &[f32], w: &[f32], pos: usize| -> Vec<f32> {
        let ms = x.iter().map(|a| a * a).sum::<f32>() / hd as f32;
        let xn: Vec<f32> = x
            .iter()
            .zip(w)
            .map(|(a, b)| a / (ms + eps).sqrt() * b)
            .collect();
        let mut y = xn.clone();
        let half = rope_dim / 2;
        for i in 0..half {
            let f = theta.powf(-2.0 * i as f32 / rope_dim as f32);
            let (s, c) = (pos as f32 * f).sin_cos();
            y[i] = xn[i] * c - xn[i + half] * s;
            y[i + half] = xn[i + half] * c + xn[i] * s;
        }
        y
    };
    let mut pos0 = 0;
    // Prefill, then decodes long enough to take the split-K path (> 256 keys).
    for t in [300usize, 1, 1] {
        let qg = rng.vec(t * nh * 2 * hd);
        let k = rng.vec(t * nkv * hd);
        let v = rng.vec(t * nkv * hd);
        let qg_b = gpu.upload_f32(&qg);
        let q = gpu.attn_prep(
            &qg_b,
            &gpu.upload_f32(&k),
            &gpu.upload_f32(&v),
            &qn_b,
            &kn_b,
            &kc,
            &vc,
            t,
            nh,
            nkv,
            hd,
            pos0,
            rope_dim,
            theta,
            eps,
        );
        let o = gpu.attention(&q, &qg_b, &kc, &vc, t, nh, nkv, hd, pos0);
        let mut want = vec![0f32; t * nh * hd];
        for ti in 0..t {
            let p = pos0 + ti;
            for kh in 0..nkv {
                let kr = norm_rope(
                    &k[(ti * nkv + kh) * hd..(ti * nkv + kh + 1) * hd],
                    &k_norm,
                    p,
                );
                cpu_k[(p * nkv + kh) * hd..(p * nkv + kh + 1) * hd].copy_from_slice(&kr);
                cpu_v[(p * nkv + kh) * hd..(p * nkv + kh + 1) * hd]
                    .copy_from_slice(&v[(ti * nkv + kh) * hd..(ti * nkv + kh + 1) * hd]);
            }
        }
        for ti in 0..t {
            let p = pos0 + ti;
            for h in 0..nh {
                let kvh = h / (nh / nkv);
                let base = (ti * nh + h) * 2 * hd;
                let q = norm_rope(&qg[base..base + hd], &q_norm, p);
                let scores: Vec<f32> = (0..=p)
                    .map(|s| {
                        (0..hd)
                            .map(|d| q[d] * cpu_k[(s * nkv + kvh) * hd + d])
                            .sum::<f32>()
                            / (hd as f32).sqrt()
                    })
                    .collect();
                let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
                let e: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
                let sum: f32 = e.iter().sum();
                for d in 0..hd {
                    let a: f32 = (0..=p)
                        .map(|s| e[s] * cpu_v[(s * nkv + kvh) * hd + d])
                        .sum::<f32>()
                        / sum;
                    let gate = qg[base + hd + d];
                    want[(ti * nh + h) * hd + d] = a / (1.0 + (-gate).exp());
                }
            }
        }
        check(
            &format!("attention t={t} pos0={pos0}"),
            &gpu.read_f32(&o),
            &want,
            1e-4,
        )?;
        pos0 += t;
    }
    let _: Option<Buf> = None;
    Ok(())
}

fn argmax_test(gpu: &Gpu, rng: &mut Rng) -> Result<()> {
    let (rows, n) = (4usize, 248_320usize);
    let data = rng.vec(rows * n);
    let buf = gpu.upload_f32(&data);
    let got = gpu.argmax_rows(&buf, rows, n);
    for r in 0..rows {
        let row = &data[r * n..(r + 1) * n];
        let mut want = 0usize;
        for (i, &v) in row.iter().enumerate() {
            if v > row[want] {
                want = i;
            }
        }
        ensure!(
            got[r] == want as u32,
            "argmax row {r}: got {}, want {want}",
            got[r]
        );
    }
    println!("ok   argmax_rows {rows}x{n}");
    Ok(())
}

fn paged_attn_test(gpu: &Gpu, rng: &mut Rng) -> Result<()> {
    let (nh, nkv, hd, rope_dim, theta, eps) =
        (4usize, 2usize, 256usize, 64usize, 1.0e7f32, 1e-6f32);
    let num_pages = 2060usize; // > 32768 tokens (2048 pages)
    let pool = KvPool::new(gpu, 1, num_pages, nkv, hd);
    // Permuted page table so logical pages map non-contiguously into physical pages.
    let mut page_table: Vec<u32> = (0..num_pages as u32).collect();
    for i in 0..num_pages {
        let j = (i * 131 + 17) % num_pages;
        page_table.swap(i, j);
    }
    let pt_buf = gpu.upload_u32(&page_table);
    let q_norm = rng.vec(hd);
    let k_norm = rng.vec(hd);
    let qn_b = gpu.upload_f32(&q_norm);
    let kn_b = gpu.upload_f32(&k_norm);
    let norm_rope = |x: &[f32], w: &[f32], pos: usize| -> Vec<f32> {
        let ms = x.iter().map(|a| a * a).sum::<f32>() / hd as f32;
        let xn: Vec<f32> = x
            .iter()
            .zip(w)
            .map(|(a, b)| a / (ms + eps).sqrt() * b)
            .collect();
        let mut y = xn.clone();
        let half = rope_dim / 2;
        for i in 0..half {
            let f = theta.powf(-2.0 * i as f32 / rope_dim as f32);
            let (s, c) = (pos as f32 * f).sin_cos();
            y[i] = xn[i] * c - xn[i + half] * s;
            y[i + half] = xn[i + half] * c + xn[i] * s;
        }
        y
    };

    // Populate 32768 keys in chunks of 1024, then verify decode at pos0=1100 (split-K) and pos0=32768 (>=32k).
    let total_keys = 32769usize;
    let mut cpu_k = vec![0f32; total_keys * nkv * hd];
    let mut cpu_v = vec![0f32; total_keys * nkv * hd];
    let mut pos0 = 0usize;
    let steps = [1100usize, 1, 31667, 1];
    for &t in &steps {
        let qg = rng.vec(t * nh * 2 * hd);
        let k = rng.vec(t * nkv * hd);
        let v = rng.vec(t * nkv * hd);
        let qg_b = gpu.upload_f32(&qg);
        let q = gpu.attn_prep_paged(
            &qg_b,
            &gpu.upload_f32(&k),
            &gpu.upload_f32(&v),
            &qn_b,
            &kn_b,
            &pool.k_pools[0],
            &pool.v_pools[0],
            &pt_buf,
            t,
            nh,
            nkv,
            hd,
            pos0,
            KV_PAGE_SIZE,
            rope_dim,
            theta,
            eps,
        );
        for ti in 0..t {
            let p = pos0 + ti;
            for kh in 0..nkv {
                let kr = norm_rope(
                    &k[(ti * nkv + kh) * hd..(ti * nkv + kh + 1) * hd],
                    &k_norm,
                    p,
                );
                cpu_k[(p * nkv + kh) * hd..(p * nkv + kh + 1) * hd].copy_from_slice(&kr);
                cpu_v[(p * nkv + kh) * hd..(p * nkv + kh + 1) * hd]
                    .copy_from_slice(&v[(ti * nkv + kh) * hd..(ti * nkv + kh + 1) * hd]);
            }
        }
        if t == 1 {
            let o = gpu.attention_paged(
                &q,
                &qg_b,
                &pool.k_pools[0],
                &pool.v_pools[0],
                &pt_buf,
                t,
                nh,
                nkv,
                hd,
                pos0,
                KV_PAGE_SIZE,
            );
            let p = pos0;
            let mut want = vec![0f32; nh * hd];
            for h in 0..nh {
                let kvh = h / (nh / nkv);
                let base = h * 2 * hd;
                let q_cpu = norm_rope(&qg[base..base + hd], &q_norm, p);
                let scores: Vec<f32> = (0..=p)
                    .map(|s| {
                        (0..hd)
                            .map(|d| q_cpu[d] * cpu_k[(s * nkv + kvh) * hd + d])
                            .sum::<f32>()
                            / (hd as f32).sqrt()
                    })
                    .collect();
                let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
                let e: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
                let sum: f32 = e.iter().sum();
                for d in 0..hd {
                    let a: f32 = (0..=p)
                        .map(|s| e[s] * cpu_v[(s * nkv + kvh) * hd + d])
                        .sum::<f32>()
                        / sum;
                    let gate = qg[base + hd + d];
                    want[h * hd + d] = a / (1.0 + (-gate).exp());
                }
            }
            check(
                &format!("paged_attention fp16 t=1 pos0={pos0}"),
                &gpu.read_f32(&o),
                &want,
                3e-3,
            )?;
        }
        pos0 += t;
    }
    Ok(())
}

fn radix_test(gpu: &Gpu) -> Result<()> {
    let mut pool = KvPool::new(gpu, 1, 6, 2, 256);
    let mut radix = RadixTree::new(4);
    let mk_ckpt = |v: f32| {
        Arc::new(GdnCheckpoint {
            conv: vec![Some(vec![v; 4])],
            ssm: vec![Some(vec![v; 4])],
            last_hidden: Some(vec![v; 8]),
        })
    };
    // Insert 2 pages (32 tokens): [0..32]
    let p0 = pool.alloc_page(&mut radix).unwrap();
    let p1 = pool.alloc_page(&mut radix).unwrap();
    let toks_a: Vec<u32> = (0..32).collect();
    radix.insert_prefix(&toks_a, &[p0, p1], Some(mk_ckpt(1.0)), &mut pool);
    pool.unpin(p0);
    pool.unpin(p1);
    ensure!(pool.free_count() == 4, "expected 4 free pages");

    // Match diverging at token 20 (page 0 matches = 16 tokens; splits node [0..32] -> [0..16] + [16..32]).
    // Wait: node [0..32] had GDN checkpoint only at 32, so require_gdn=false matches 16 tokens,
    // and matching exact 32+1 tokens with require_gdn=true matches all 32 tokens + checkpoint!
    let mut toks_a_ext = toks_a.clone();
    toks_a_ext.push(999);
    let m = radix.match_prefix(&toks_a_ext, &mut pool, true, true);
    ensure!(
        m.matched_tokens == 32 && m.pages == vec![p0, p1],
        "exact prefix hit"
    );
    ensure!(m.gdn_ckpt.as_ref().unwrap().conv[0].as_ref().unwrap()[0] == 1.0);
    for p in m.pages {
        pool.unpin(p);
    }

    // Insert overlapping branch [0..16, 100..116] (splits [0..32] at 16)
    let p2 = pool.alloc_page(&mut radix).unwrap();
    let p3 = pool.alloc_page(&mut radix).unwrap();
    let mut toks_b: Vec<u32> = (0..16).collect();
    toks_b.extend(100..116);
    radix.insert_prefix(&toks_b, &[p2, p3], Some(mk_ckpt(2.0)), &mut pool);
    pool.unpin(p2);
    pool.unpin(p3);

    // Allocate remaining 2 free pages + 2 more forcing LRU eviction of unpinned radix leaves.
    let mut held = Vec::new();
    for _ in 0..4 {
        held.push(
            pool.alloc_page(&mut radix)
                .expect("eviction should free pages"),
        );
    }
    for p in held {
        pool.unpin(p);
    }
    println!("ok   radix tree + paged kv eviction");
    Ok(())
}
