//! Qwen3.5 MoE (hybrid Gated-DeltaNet / gated full attention) forward pass.

/// Opt-in (`BURN_PROF=1`) per-phase device-synchronized timing of `forward`.
pub struct Prof {
    on: bool,
    t: Instant,
    acc: [f64; PROF_N],
}

const PROF_N: usize = 10;
const PROF_NAMES: [&str; PROF_N] = [
    "gdn",
    "attn",
    "ffn/route",
    "shared",
    "experts",
    "upload",
    "cpu",
    "head",
    "draft",
    "catchup",
];

impl Prof {
    fn new() -> Self {
        Self {
            on: std::env::var_os("BURN_PROF").is_some(),
            t: Instant::now(),
            acc: [0.0; PROF_N],
        }
    }

    /// Wait for the device, then charge the time since the previous tick to `cat`.
    fn tick(&mut self, gpu: &Gpu, cat: usize) {
        if self.on {
            gpu.sync();
            self.acc[cat] += self.t.elapsed().as_secs_f64();
            self.t = Instant::now();
        }
    }

    pub fn summary(&self) -> Option<String> {
        self.on.then(|| {
            let total: f64 = self.acc.iter().sum();
            let mut s = format!("profile (total {total:.2}s):");
            for (n, a) in PROF_NAMES.iter().zip(self.acc) {
                s += &format!(" {n} {a:.2}s");
            }
            s
        })
    }
}

use crate::cpu_moe::{self, CpuDecodeJob, CpuExpertJob};
use crate::gguf::{Gguf, TensorInfo, Value};
use crate::gpu::{Buf, Gpu, QWeight};
use crate::kernels::{MOE_DESC, MOE_MAXM};
use crate::quant::{self, fmt};
use crate::radix::{GdnCheckpoint, KV_PAGE_SIZE, KvPool, RadixTree};
use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct Config {
    pub n_layers: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub eps: f32,
    pub n_experts: usize,
    pub topk: usize,
    pub expert_inter: usize,
    pub shared_inter: usize,
    pub conv_k: usize,
    /// Key/value head dim of the linear-attention layers.
    pub ssm_dk: usize,
    /// Number of key heads (ssm.group_count).
    pub ssm_nk: usize,
    /// Number of value heads (ssm.time_step_rank).
    pub ssm_nv: usize,
    pub full_attn_interval: usize,
    /// FFN width of dense (non-MoE) models; 0 for MoE.
    pub dense_inter: usize,
    /// Trailing multi-token-prediction (NextN) blocks, excluded from `n_layers`.
    pub n_mtp: usize,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.meta_str("general.architecture")?;
        // Kolibri 1.1 GGUFs carry arch="kolibri1" (llama.cpp's converter does not use a
        // distinct 1.1 arch string). Refuse them by name using the 1.1-only marker KVs so a
        // 1.1 file is never silently treated as 1.0.
        if arch == "kolibri1"
            && [
                "kolibri1.attention.indexer.top_k",
                "kolibri1.attention.indexer.types",
                "kolibri1.attention.kv_source_map",
                "kolibri1.attention.indexer.candidate_block_size",
                "kolibri1.attention.indexer.candidate_top_k_blocks",
                "kolibri1.leading_dense_block_count",
            ]
            .iter()
            .any(|k| g.meta.contains_key(*k))
        {
            bail!(
                "kolibri1_1 is not supported (GGUF reports arch kolibri1; detected 1.1 marker KVs)"
            );
        }
        let moe = arch == "qwen35moe";
        // Inkling-2 can be tabulated and its config parsed, but its forward pass is not
        // implemented yet. Parsing it here lets loaders fail later on a specific missing
        // tensor rather than rejecting the whole file up front.
        let inkling2 = arch == "inkling2_loop";
        ensure!(
            moe || arch == "qwen35" || inkling2,
            "unsupported architecture {arch} (want qwen35, qwen35moe or inkling2_loop)"
        );
        let k = |s: &str| g.meta_u64(&format!("{arch}.{s}")).map(|v| v as usize);
        // Inkling-2 stores head counts as one entry per layer; all layers agree on them,
        // so the first entry is the scalar the rest of the config expects.
        let head_count = |s: &str| -> Result<usize> {
            if !inkling2 {
                return k(s);
            }
            let v = g
                .meta_array(&format!("{arch}.{s}"))?
                .first()
                .and_then(Value::as_u64)
                .with_context(|| format!("missing {arch}.{s}[0]"))?;
            Ok(v as usize)
        };
        let vocab = g.tensor("token_embd.weight")?.dims[1] as usize;
        let c = Self {
            n_layers: k("block_count")? - k("nextn_predict_layers").unwrap_or(0),
            hidden: k("embedding_length")?,
            vocab,
            n_heads: head_count("attention.head_count")?,
            n_kv: head_count("attention.head_count_kv")?,
            head_dim: k("attention.key_length")?,
            rope_dim: k("rope.dimension_count")?,
            // Inkling-2 has no RoPE and therefore no freq_base metadata.
            rope_theta: if inkling2 {
                10_000.0
            } else {
                g.meta_f64(&format!("{arch}.rope.freq_base"))? as f32
            },
            eps: g.meta_f64(&format!("{arch}.attention.layer_norm_rms_epsilon"))? as f32,
            n_experts: if moe || inkling2 {
                k("expert_count")?
            } else {
                0
            },
            topk: if moe || inkling2 {
                k("expert_used_count")?
            } else {
                0
            },
            expert_inter: if moe || inkling2 {
                k("expert_feed_forward_length")?
            } else {
                0
            },
            shared_inter: if moe {
                k("expert_shared_feed_forward_length")?
            } else {
                0
            },
            // Inkling-2 uses shortconv (l_cache=4) instead of GDN/SSM state.
            conv_k: if inkling2 { 4 } else { k("ssm.conv_kernel")? },
            ssm_dk: if inkling2 { 0 } else { k("ssm.state_size")? },
            ssm_nk: if inkling2 { 0 } else { k("ssm.group_count")? },
            ssm_nv: if inkling2 {
                0
            } else {
                k("ssm.time_step_rank")?
            },
            full_attn_interval: if inkling2 {
                k("attn_res.block_size")?
            } else {
                k("full_attention_interval")?
            },
            dense_inter: if moe || inkling2 {
                0
            } else {
                k("feed_forward_length")?
            },
            n_mtp: k("nextn_predict_layers").unwrap_or(0),
        };
        ensure!(c.topk <= MOE_MAXM, "topk {} > {MOE_MAXM}", c.topk);
        ensure!(c.conv_k == 4, "conv kernel {} unsupported", c.conv_k);
        if !inkling2 {
            ensure!(
                k("ssm.inner_size")? == c.ssm_nv * c.ssm_dk,
                "ssm inner size mismatch"
            );
        }
        Ok(c)
    }

    pub fn is_full_attn(&self, layer: usize) -> bool {
        (layer + 1) % self.full_attn_interval == 0
    }

    fn conv_dim(&self) -> usize {
        2 * self.ssm_nk * self.ssm_dk + self.ssm_nv * self.ssm_dk
    }
}

enum GdnProj {
    /// Concatenated `[qkv | z | alpha | beta]` projection in a single weight matrix.
    Fused(QWeight),
    Split {
        qkv: QWeight,
        z: QWeight,
        alpha: QWeight,
        beta: QWeight,
    },
}

struct GdnW {
    proj: GdnProj,
    out: QWeight,
    conv_w: Buf,
    ssm_a: Buf,
    dt_bias: Buf,
    norm: Buf,
}

enum AttnProj {
    /// Concatenated `[q | k | v]` projection in a single weight matrix.
    Fused(QWeight),
    Split {
        q: QWeight,
        k: QWeight,
        v: QWeight,
    },
}

struct AttnW {
    proj: AttnProj,
    o: QWeight,
    q_norm: Buf,
    k_norm: Buf,
}

enum Mixer {
    Gdn(GdnW),
    Attn(AttnW),
}

/// One expert tensor stack ([n_experts, rows, k]) mapped to a slot pool on the device.
struct ExpertTensor {
    info: TensorInfo,
    fmt: u32,
    expert_bytes: usize,
    w0_per: usize,
    w1_per: usize,
    w0: Buf,
    w1: Buf,
}

impl ExpertTensor {
    fn new(gpu: &Gpu, info: &TensorInfo, n_experts: usize, slots: usize) -> Self {
        // Codebook formats keep one shared codebook in w1 (indexed absolutely by the
        // kernels), so w1 is not slot-pooled.
        let codebook = info.ty.codebook();
        let per_numel = (info.numel() as usize) / n_experts;
        let f = info.ty.gpu_fmt();
        let w0_per = fmt::w0_words(f, per_numel);
        let w1_per = if matches!(f, fmt::Q8_0 | fmt::MXFP4) {
            fmt::w1_words(f, per_numel)
        } else {
            0
        };
        let (w0, w1) = gpu.client.memory_persistent_allocation((), |_| {
            (
                gpu.alloc(w0_per * slots),
                match &codebook {
                    Some(cb) => gpu.upload_u32(cb),
                    None => gpu.alloc((w1_per * slots).max(1)),
                },
            )
        });
        Self {
            info: info.clone(),
            fmt: f,
            expert_bytes: info.nbytes / n_experts,
            w0_per,
            w1_per,
            w0,
            w1,
        }
    }

    fn raw<'a>(&self, g: &'a Gguf, e: usize) -> &'a [u8] {
        let all = g.bytes(&self.info);
        &all[e * self.expert_bytes..(e + 1) * self.expert_bytes]
    }

    fn arg(&self) -> (&Buf, &Buf) {
        (&self.w0, &self.w1)
    }
}

/// Expert residency for one layer: either all experts resident (slot = expert id) or
/// an LRU cache of `slots` experts refilled from the mmap'd GGUF on demand.
struct ExpertStore {
    gate: ExpertTensor,
    up: ExpertTensor,
    down: ExpertTensor,
    slots: usize,
    slot_of: Vec<i32>,
    expert_in: Vec<i32>,
    stamp: Vec<u64>,
    freq: Vec<u32>,
}

impl ExpertStore {
    fn resident(&self, n_experts: usize) -> bool {
        self.slots == n_experts
    }

    /// Update both directions of the residency map in one place. The previous occupant of
    /// `slot` (if any) becomes non-resident before the new expert takes it.
    fn occupy(&mut self, e: usize, slot: usize) {
        if self.expert_in[slot] >= 0 {
            self.slot_of[self.expert_in[slot] as usize] = -1;
        }
        self.expert_in[slot] = e as i32;
        self.slot_of[e] = slot as i32;
        debug_assert_eq!(self.slot_of[e], slot as i32);
        debug_assert_eq!(self.expert_in[slot], e as i32);
    }

    /// `slot_of[e] == s` and `expert_in[s] == e` must always agree; both are maintained by
    /// `occupy`, so a mismatch means some other path edited one side by hand.
    fn debug_assert_residency(&self) {
        debug_assert!((0..self.slot_of.len()).all(|e| {
            let s = self.slot_of[e];
            s < 0 || ((s as usize) < self.slots && self.expert_in[s as usize] == e as i32)
        }));
        debug_assert!((0..self.slots).all(|s| {
            let e = self.expert_in[s];
            e < 0 || ((e as usize) < self.slot_of.len() && self.slot_of[e as usize] == s as i32)
        }));
    }

    /// Load experts `(expert, slot)` into their slots: repack in parallel on the host,
    /// then enqueue the device writes (stream-ordered with earlier kernels). Batches are
    /// bounded so the pinned staging they need stays small.
    fn load_many(&mut self, gpu: &Gpu, g: &Gguf, list: &[(usize, usize)]) {
        let tensors = [&self.gate, &self.up, &self.down];
        for batch in list.chunks(16) {
            let packed: Vec<(usize, usize, quant::Packed)> = batch
                .par_iter()
                .flat_map_iter(|&(e, slot)| (0..3).map(move |ti| (e, slot, ti)))
                .map(|(e, slot, ti)| {
                    (
                        slot,
                        ti,
                        quant::repack(tensors[ti].info.ty, tensors[ti].raw(g, e)),
                    )
                })
                .collect();
            let mut writes = Vec::with_capacity(packed.len() * 2);
            for (slot, ti, p) in packed {
                let t = tensors[ti];
                writes.push((&t.w0, slot * t.w0_per, p.w0));
                if t.w1_per > 0 {
                    writes.push((&t.w1, slot * t.w1_per, p.w1));
                }
            }
            gpu.write_many(writes);
        }
        for &(e, slot) in list {
            self.occupy(e, slot);
        }
        self.debug_assert_residency();
    }

    /// Repack a single expert on the host (for overlapped promotion before enqueueing DMA).
    fn repack_one(&self, g: &Gguf, e: usize) -> [quant::Packed; 3] {
        [
            quant::repack(self.gate.info.ty, self.gate.raw(g, e)),
            quant::repack(self.up.info.ty, self.up.raw(g, e)),
            quant::repack(self.down.info.ty, self.down.raw(g, e)),
        ]
    }

    /// Enqueue device writes for a pre-repacked expert and update slot residency tables.
    fn commit_one(&mut self, gpu: &Gpu, e: usize, slot: usize, packed: [quant::Packed; 3]) {
        let tensors = [&self.gate, &self.up, &self.down];
        let mut writes = Vec::with_capacity(6);
        for (ti, p) in packed.into_iter().enumerate() {
            let t = tensors[ti];
            writes.push((&t.w0, slot * t.w0_per, p.w0));
            if t.w1_per > 0 {
                writes.push((&t.w1, slot * t.w1_per, p.w1));
            }
        }
        gpu.write_many(writes);
        self.occupy(e, slot);
        self.debug_assert_residency();
    }
}

/// Weights of the MTP (NextN) draft head beyond its decoder block (stored as the last
/// entry of `layers` / `moes`): the [embedding | hidden] projection and the norms.
pub struct MtpW {
    eh_proj: QWeight,
    enorm: Buf,
    hnorm: Buf,
    head_norm: Buf,
}

struct MoeW {
    /// Router [n_experts + 1, hidden] in f32; the last row is the shared-expert gate.
    router: QWeight,
    gate_sh: QWeight,
    up_sh: QWeight,
    down_sh: QWeight,
    experts: ExpertStore,
}

struct DenseW {
    gate: QWeight,
    up: QWeight,
    down: QWeight,
}

struct Layer {
    attn_norm: Buf,
    post_norm: Buf,
    mixer: Mixer,
}

pub struct Model {
    pub cfg: Config,
    pub gpu: Gpu,
    gguf: Gguf,
    layers: Vec<Layer>,
    /// Per-layer routed-MoE weights (empty for dense models).
    moes: Vec<MoeW>,
    /// Per-layer dense FFN weights (empty for MoE models).
    denses: Vec<DenseW>,
    mtp: Option<MtpW>,
    out_norm: Buf,
    lm_head: QWeight,
    embd: TensorInfo,
    pub kv_pool: KvPool,
    pub radix: RadixTree,
    attn_idx: Vec<Option<usize>>,
    /// Rows at or above which dense layers dequantize + use the tiled GEMM.
    pub gemm_min_m: usize,
    clock: u64,
    pub stats: Stats,
    pub prof: Prof,
}
/// Where a micro-batch writes its GDN rollback snapshots: token offset in the window, and
/// whether its last token's state is needed (another micro-batch follows).
#[derive(Clone, Copy)]
struct Snap {
    off: usize,
    save_last: bool,
}

/// Router output for one batch, before the host readback.
struct Routed {
    logits: Buf,
    wts: Buf,
    pack: Buf,
}

/// A multi-token MoE step between its routing sync and its combine: shared + hit experts
/// are enqueued on the GPU, `misses` still need the CPU.
struct MoeJob {
    li: usize,
    t: usize,
    logits: Buf,
    wts: Buf,
    sh: Buf,
    y: Buf,
    /// Missed experts with their (token row, routing weight) pairs.
    misses: Vec<(usize, Vec<(usize, f32)>)>,
    h2_host: Vec<f32>,
    /// [hit weights (t*topk) | CPU expert sum (t*hidden)], uploaded for the combine.
    pack: Vec<f32>,
}

#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub expert_hits: u64,
    pub expert_misses: u64,
    pub promotions: u64,
    /// Host time spent repacking + enqueueing expert uploads.
    pub upload_secs: f64,
    /// Host time blocked reading routing decisions back from the device.
    pub route_sync_secs: f64,
    /// Host time computing missed experts on CPU (overlapped with GPU).
    pub cpu_secs: f64,
}

impl Stats {
    pub fn since(&self, earlier: &Stats) -> Stats {
        Stats {
            expert_hits: self.expert_hits - earlier.expert_hits,
            expert_misses: self.expert_misses - earlier.expert_misses,
            promotions: self.promotions - earlier.promotions,
            upload_secs: self.upload_secs - earlier.upload_secs,
            route_sync_secs: self.route_sync_secs - earlier.route_sync_secs,
            cpu_secs: self.cpu_secs - earlier.cpu_secs,
        }
    }

    /// One-line summary; `None` when no host-routed expert lookups happened.
    pub fn summary(&self) -> Option<String> {
        let lookups = self.expert_hits + self.expert_misses;
        (lookups > 0).then(|| {
            format!(
                "{} lookups, hit rate {:.1}%, cpu {:.2}s, upload {:.2}s ({} promos), routing sync {:.2}s",
                lookups,
                100.0 * self.expert_hits as f64 / lookups as f64,
                self.cpu_secs,
                self.upload_secs,
                self.promotions,
                self.route_sync_secs
            )
        })
    }
}

pub struct LoadOptions {
    /// Expert slots per layer; `None` = fit to the VRAM budget.
    pub expert_slots: Option<usize>,
    /// VRAM budget in bytes for weights + cache (None = device-reported total).
    pub vram_budget: Option<u64>,
    pub max_ctx: usize,
    /// Draft tokens per speculative step (0 = no speculative decoding); sizes the
    /// rollback snapshots reserved in VRAM.
    pub spec_k: usize,
}

#[derive(Default, Clone, Copy)]
pub struct FwdOpts {
    /// Logits for every input row (default: the last row only).
    pub all_rows: bool,
    /// Return the out-normed hidden state of every row.
    pub keep_hidden: bool,
    /// Save per-token GDN snapshots so a verify window can be rolled back.
    pub snap: bool,
    /// Return GPU-computed argmax token IDs instead of reading full logits back to host.
    pub greedy: bool,
}

pub struct FwdOut {
    /// `rows * vocab` logits (empty when `FwdOpts::greedy` is set).
    pub logits: Vec<f32>,
    /// `rows` argmax token IDs (populated when `FwdOpts::greedy` is set).
    pub tokens: Vec<u32>,
    /// `[t, hidden]` out-normed hidden states, when requested.
    pub hidden: Option<Buf>,
}

mod mtp;

pub struct State {
    pub pos: usize,
    pub max_ctx: usize,
    pub page_table: Vec<u32>,
    page_table_buf: Buf,
    pub tokens: Vec<u32>,
    conv: Vec<Option<Buf>>,
    ssm: Vec<Option<Buf>>,
    /// Per-GDN-layer snapshots of the recurrent / conv state after each token of a
    /// verify window but the last ([spec_k, ...]), for rollback on rejection.
    ssm_snap: Vec<Option<Buf>>,
    conv_snap: Vec<Option<Buf>>,
}

fn upload_weight(gpu: &Gpu, g: &Gguf, name: &str) -> Result<QWeight> {
    let t = g.tensor(name)?;
    ensure!(t.dims.len() == 2, "{name}: expected 2-D, got {:?}", t.dims);
    let p = quant::repack(t.ty, g.bytes(t));
    Ok(QWeight {
        fmt: p.fmt,
        k: t.dims[0] as usize,
        n: t.dims[1] as usize,
        w0: gpu.upload_u32(&p.w0),
        w1: gpu.upload_u32(&p.w1),
    })
}

/// Concatenate multiple `[n_i, k]` weights vertically into a single `[sum(n_i), k]` `QWeight`
/// when they share the same quant type, `k`, and scale-word alignment.
fn try_upload_concat_weights(gpu: &Gpu, g: &Gguf, names: &[&str]) -> Result<Option<QWeight>> {
    let mut infos = Vec::with_capacity(names.len());
    for &name in names {
        let t = g.tensor(name)?;
        ensure!(t.dims.len() == 2, "{name}: expected 2-D, got {:?}", t.dims);
        infos.push(t);
    }
    let ty = infos[0].ty;
    let k = infos[0].dims[0] as usize;
    let f = ty.gpu_fmt();
    if infos.iter().any(|t| t.ty != ty || t.dims[0] as usize != k) {
        return Ok(None);
    }
    let align = match f {
        fmt::Q8_0 => 2,
        fmt::MXFP4 => 4,
        _ => 1,
    };
    if infos
        .iter()
        .any(|t| ((t.dims[1] as usize * k) / 32) % align != 0)
    {
        return Ok(None);
    }
    let mut w0 = Vec::new();
    let mut w1 = Vec::new();
    let mut total_n = 0usize;
    let has_per_block_w1 = matches!(f, fmt::Q8_0 | fmt::MXFP4);
    for t in infos {
        let p = quant::repack(t.ty, g.bytes(t));
        w0.extend_from_slice(&p.w0);
        if has_per_block_w1 {
            w1.extend_from_slice(&p.w1);
        } else if w1.is_empty() {
            w1 = p.w1;
        }
        total_n += t.dims[1] as usize;
    }
    Ok(Some(QWeight {
        fmt: f,
        k,
        n: total_n,
        w0: gpu.upload_u32(&w0),
        w1: gpu.upload_u32(&w1),
    }))
}

fn f32_tensor(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = g.tensor(name)?;
    let mut v = vec![0f32; t.numel() as usize];
    quant::dequant(t.ty, g.bytes(t), &mut v);
    Ok(v)
}

fn upload_f32(gpu: &Gpu, g: &Gguf, name: &str) -> Result<Buf> {
    Ok(gpu.upload_f32(&f32_tensor(g, name)?))
}

impl Model {
    pub fn load(gpu: Gpu, gguf: Gguf, opts: &LoadOptions) -> Result<Self> {
        let cfg = Config::from_gguf(&gguf)?;
        let t0 = Instant::now();
        let g = &gguf;
        let blk = |i: usize, s: &str| format!("blk.{i}.{s}");

        // Size the expert pools before allocating anything.
        let mut expert_bytes_layer = 0usize;
        let mut dense_bytes = 0usize;
        let mtp_prefix = format!("blk.{}.", cfg.n_layers);
        for t in g.tensors.values() {
            let is_mtp = cfg.n_mtp > 0 && t.name.starts_with(&mtp_prefix);
            if t.name.contains("_exps.") {
                // The MTP block's experts stay in RAM (CPU-computed), never on the card.
                if !is_mtp {
                    expert_bytes_layer += t.nbytes;
                }
            } else if t.name != "token_embd.weight" {
                dense_bytes += t.nbytes;
            }
        }
        let expert_bytes_per = if cfg.n_experts > 0 {
            expert_bytes_layer / (cfg.n_layers * cfg.n_experts)
        } else {
            0
        };
        let num_pages = opts.max_ctx.div_ceil(KV_PAGE_SIZE);
        let n_attn_layers = cfg.n_layers / cfg.full_attn_interval + cfg.n_mtp;
        let kv_bytes =
            n_attn_layers * 2 * num_pages * KV_PAGE_SIZE * cfg.n_kv * (cfg.head_dim / 2) * 4;
        let is_wgpu = gpu.client.name().starts_with("wgpu");
        let total_vram = opts
            .vram_budget
            .or(gpu.client.properties().memory.max_memory())
            .unwrap_or(8 << 30) as usize;
        // wgpu reports raw physical VRAM and allocates staging/driver buffers out of it.
        let effective_vram = if opts.vram_budget.is_none() && is_wgpu {
            (total_vram as f64 * 0.80) as usize
        } else {
            total_vram
        };
        // GPU layouts carry a little padding; scratch for a 512-token prefill chunk.
        // Speculative decoding keeps per-token GDN state snapshots for rollback.
        let gdn_layers = cfg.n_layers - cfg.n_layers / cfg.full_attn_interval;
        let snap_bytes = opts.spec_k
            * gdn_layers
            * (cfg.ssm_nv * cfg.ssm_dk * cfg.ssm_dk + (cfg.conv_k - 1) * cfg.conv_dim())
            * 4;
        let reserve = (dense_bytes as f64 * 1.08) as usize + kv_bytes + snap_bytes + (640 << 20);
        let slots = if cfg.n_experts == 0 {
            0
        } else {
            match opts.expert_slots {
                Some(s) => s.clamp(cfg.topk, cfg.n_experts),
                None => {
                    let avail = effective_vram.saturating_sub(reserve);
                    let per_slot = (expert_bytes_per as f64 * 1.08) as usize * cfg.n_layers;
                    (avail / per_slot).clamp(cfg.topk, cfg.n_experts)
                }
            }
        };
        eprintln!(
            "vram budget {:.2} GiB: dense {:.2} GiB, kv {:.2} GiB, experts {}/{} slots per layer ({:.2} GiB)",
            total_vram as f64 / (1u64 << 30) as f64,
            dense_bytes as f64 / (1u64 << 30) as f64,
            kv_bytes as f64 / (1u64 << 30) as f64,
            slots,
            cfg.n_experts,
            (slots * expert_bytes_per * cfg.n_layers) as f64 / (1u64 << 30) as f64,
        );

        let gpu_ref = &gpu;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut moes = Vec::new();
        let mut denses = Vec::new();
        let mut mtp = None;
        gpu.client
            .memory_persistent_allocation((), |_| -> Result<()> {
                for i in 0..cfg.n_layers + cfg.n_mtp {
                    let is_mtp = i >= cfg.n_layers;
                    let mixer = if cfg.is_full_attn(i) || is_mtp {
                        let qn = blk(i, "attn_q.weight");
                        let kn = blk(i, "attn_k.weight");
                        let vn = blk(i, "attn_v.weight");
                        let proj = match try_upload_concat_weights(gpu_ref, g, &[&qn, &kn, &vn])? {
                            Some(fused) => AttnProj::Fused(fused),
                            None => AttnProj::Split {
                                q: upload_weight(gpu_ref, g, &qn)?,
                                k: upload_weight(gpu_ref, g, &kn)?,
                                v: upload_weight(gpu_ref, g, &vn)?,
                            },
                        };
                        Mixer::Attn(AttnW {
                            proj,
                            o: upload_weight(gpu_ref, g, &blk(i, "attn_output.weight"))?,
                            q_norm: upload_f32(gpu_ref, g, &blk(i, "attn_q_norm.weight"))?,
                            k_norm: upload_f32(gpu_ref, g, &blk(i, "attn_k_norm.weight"))?,
                        })
                    } else {
                        let qkvn = blk(i, "attn_qkv.weight");
                        let zn = blk(i, "attn_gate.weight");
                        let an = blk(i, "ssm_alpha.weight");
                        let bn = blk(i, "ssm_beta.weight");
                        let proj =
                            match try_upload_concat_weights(gpu_ref, g, &[&qkvn, &zn, &an, &bn])? {
                                Some(fused) => GdnProj::Fused(fused),
                                None => GdnProj::Split {
                                    qkv: upload_weight(gpu_ref, g, &qkvn)?,
                                    z: upload_weight(gpu_ref, g, &zn)?,
                                    alpha: upload_weight(gpu_ref, g, &an)?,
                                    beta: upload_weight(gpu_ref, g, &bn)?,
                                },
                            };
                        Mixer::Gdn(GdnW {
                            proj,
                            out: upload_weight(gpu_ref, g, &blk(i, "ssm_out.weight"))?,
                            conv_w: upload_f32(gpu_ref, g, &blk(i, "ssm_conv1d.weight"))?,
                            ssm_a: upload_f32(gpu_ref, g, &blk(i, "ssm_a"))?,
                            dt_bias: upload_f32(gpu_ref, g, &blk(i, "ssm_dt.bias"))?,
                            norm: upload_f32(gpu_ref, g, &blk(i, "ssm_norm.weight"))?,
                        })
                    };
                    if cfg.n_experts == 0 {
                        denses.push(DenseW {
                            gate: upload_weight(gpu_ref, g, &blk(i, "ffn_gate.weight"))?,
                            up: upload_weight(gpu_ref, g, &blk(i, "ffn_up.weight"))?,
                            down: upload_weight(gpu_ref, g, &blk(i, "ffn_down.weight"))?,
                        });
                    } else {
                        let mut router = f32_tensor(g, &blk(i, "ffn_gate_inp.weight"))?;
                        router.extend(f32_tensor(g, &blk(i, "ffn_gate_inp_shexp.weight"))?);
                        ensure!(
                            router.len() == (cfg.n_experts + 1) * cfg.hidden,
                            "router shape"
                        );
                        let router = QWeight {
                            fmt: fmt::F32,
                            n: cfg.n_experts + 1,
                            k: cfg.hidden,
                            w0: gpu_ref.upload_u32(bytemuck::cast_slice(&router)),
                            w1: gpu_ref.upload_u32(&[0]),
                        };
                        let gate_t = g.tensor(&blk(i, "ffn_gate_exps.weight"))?;
                        let up_t = g.tensor(&blk(i, "ffn_up_exps.weight"))?;
                        let down_t = g.tensor(&blk(i, "ffn_down_exps.weight"))?;
                        ensure!(
                            gate_t.ty == up_t.ty,
                            "layer {i}: gate/up expert types differ"
                        );
                        let ne = cfg.n_experts;
                        let sl = if is_mtp { 0 } else { slots };
                        let experts = ExpertStore {
                            gate: ExpertTensor::new(gpu_ref, gate_t, ne, sl),
                            up: ExpertTensor::new(gpu_ref, up_t, ne, sl),
                            down: ExpertTensor::new(gpu_ref, down_t, ne, sl),
                            slots: sl,
                            slot_of: vec![-1; ne],
                            expert_in: vec![-1; sl],
                            stamp: vec![0; sl],
                            freq: vec![0; ne],
                        };
                        moes.push(MoeW {
                            router,
                            gate_sh: upload_weight(gpu_ref, g, &blk(i, "ffn_gate_shexp.weight"))?,
                            up_sh: upload_weight(gpu_ref, g, &blk(i, "ffn_up_shexp.weight"))?,
                            down_sh: upload_weight(gpu_ref, g, &blk(i, "ffn_down_shexp.weight"))?,
                            experts,
                        });
                    }
                    layers.push(Layer {
                        attn_norm: upload_f32(gpu_ref, g, &blk(i, "attn_norm.weight"))?,
                        post_norm: upload_f32(gpu_ref, g, &blk(i, "post_attention_norm.weight"))?,
                        mixer,
                    });
                    if is_mtp {
                        mtp = Some(MtpW {
                            eh_proj: upload_weight(gpu_ref, g, &blk(i, "nextn.eh_proj.weight"))?,
                            enorm: upload_f32(gpu_ref, g, &blk(i, "nextn.enorm.weight"))?,
                            hnorm: upload_f32(gpu_ref, g, &blk(i, "nextn.hnorm.weight"))?,
                            head_norm: upload_f32(
                                gpu_ref,
                                g,
                                &blk(i, "nextn.shared_head_norm.weight"),
                            )?,
                        });
                    }
                }
                Ok(())
            })?;
        // Smaller dense Qwen3.5 checkpoints tie the LM head to the embedding table and
        // omit `output.weight`. Fall back to the embedding tensor in that case.
        let head_name = if g.tensors.contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let (out_norm, lm_head) =
            gpu.client
                .memory_persistent_allocation((), |_| -> Result<_> {
                    Ok((
                        upload_f32(gpu_ref, g, "output_norm.weight")?,
                        upload_weight(gpu_ref, g, head_name)?,
                    ))
                })?;
        let embd = g.tensor("token_embd.weight")?.clone();
        let mut attn_idx = Vec::with_capacity(cfg.n_layers + cfg.n_mtp);
        let mut next_ai = 0usize;
        for i in 0..cfg.n_layers + cfg.n_mtp {
            let is_mtp = i >= cfg.n_layers;
            if cfg.is_full_attn(i) || is_mtp {
                attn_idx.push(Some(next_ai));
                next_ai += 1;
            } else {
                attn_idx.push(None);
            }
        }
        let kv_pool = KvPool::new(&gpu, n_attn_layers, num_pages, cfg.n_kv, cfg.head_dim);
        let radix = RadixTree::new(8);
        let mut model = Self {
            cfg,
            gpu,
            gguf,
            layers,
            moes,
            denses,
            mtp,
            out_norm,
            lm_head,
            embd,
            kv_pool,
            radix,
            attn_idx,
            // Measured crossover (`--bench`, RTX 5050): qmatmul is ~18 us per token on
            // 8192x2048 Q8_0, dequant + GEMM ~1.3 ms flat.
            gemm_min_m: 80,
            clock: 0,
            stats: Stats::default(),
            prof: Prof::new(),
        };
        // with the first experts.
        for li in 0..model.moes.len() {
            let store = &mut model.moes[li].experts;
            let all: Vec<(usize, usize)> = (0..store.slots).map(|s| (s, s)).collect();
            store.load_many(&model.gpu, &model.gguf, &all);
            model.gpu.sync();
            eprint!(
                "\rloading experts: layer {}/{}",
                (li + 1).min(model.cfg.n_layers),
                model.cfg.n_layers
            );
        }
        eprintln!();
        eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
        Ok(model)
    }

    pub fn new_state(&self, max_ctx: usize, spec_k: usize) -> State {
        let c = &self.cfg;
        let max_pages = max_ctx.div_ceil(KV_PAGE_SIZE).max(1);
        let mut st = State {
            pos: 0,
            max_ctx,
            page_table: Vec::new(),
            page_table_buf: self.gpu.zeros(max_pages),
            tokens: Vec::new(),
            conv: vec![],
            ssm: vec![],
            ssm_snap: vec![],
            conv_snap: vec![],
        };
        for i in 0..c.n_layers {
            if c.is_full_attn(i) {
                st.conv.push(None);
                st.ssm.push(None);
                st.conv_snap.push(None);
                st.ssm_snap.push(None);
            } else {
                st.conv
                    .push(Some(self.gpu.zeros((c.conv_k - 1) * c.conv_dim())));
                st.ssm
                    .push(Some(self.gpu.zeros(c.ssm_nv * c.ssm_dk * c.ssm_dk)));
                if spec_k > 0 {
                    st.conv_snap
                        .push(Some(self.gpu.alloc(spec_k * (c.conv_k - 1) * c.conv_dim())));
                    st.ssm_snap.push(Some(
                        self.gpu.alloc(spec_k * c.ssm_nv * c.ssm_dk * c.ssm_dk),
                    ));
                } else {
                    st.conv_snap.push(None);
                    st.ssm_snap.push(None);
                }
            }
        }
        st
    }

    /// Ensure physical KV pages are allocated and synced to GPU for `target_tokens`.
    pub fn ensure_pages(&mut self, st: &mut State, target_tokens: usize) -> Result<()> {
        let need = target_tokens.div_ceil(KV_PAGE_SIZE);
        let start = st.page_table.len();
        if need <= start {
            return Ok(());
        }
        let mut added = Vec::with_capacity(need - start);
        for _ in start..need {
            let p = self
                .kv_pool
                .alloc_page(&mut self.radix)
                .context("paged KV pool exhausted")?;
            st.page_table.push(p);
            added.push(p);
        }
        self.gpu.write_at(&st.page_table_buf, start, added);
        Ok(())
    }

    /// Unpin all KV pages held by `st` (pages saved in `self.radix` remain cached until evicted).
    pub fn release_state(&mut self, st: State) {
        for p in st.page_table {
            self.kv_pool.unpin(p);
        }
    }

    fn has_gdn(&self) -> bool {
        self.cfg.n_layers > self.cfg.n_layers / self.cfg.full_attn_interval
    }

    fn save_gdn_checkpoint(&self, st: &State, last_hidden: Option<&Buf>) -> GdnCheckpoint {
        let conv = st
            .conv
            .iter()
            .map(|b| b.as_ref().map(|buf| self.gpu.read_f32(buf)))
            .collect();
        let ssm = st
            .ssm
            .iter()
            .map(|b| b.as_ref().map(|buf| self.gpu.read_f32(buf)))
            .collect();
        let last_hidden = last_hidden.map(|buf| self.gpu.read_f32(buf));
        GdnCheckpoint {
            conv,
            ssm,
            last_hidden,
        }
    }

    fn restore_gdn_checkpoint(&self, st: &State, ckpt: &GdnCheckpoint) {
        let mut writes = Vec::new();
        for (dst, src) in st.conv.iter().zip(&ckpt.conv) {
            if let (Some(d), Some(s)) = (dst, src) {
                let words: Vec<u32> = bytemuck::cast_slice::<f32, u32>(s).to_vec();
                writes.push((d, 0usize, words));
            }
        }
        for (dst, src) in st.ssm.iter().zip(&ckpt.ssm) {
            if let (Some(d), Some(s)) = (dst, src) {
                let words: Vec<u32> = bytemuck::cast_slice::<f32, u32>(s).to_vec();
                writes.push((d, 0usize, words));
            }
        }
        self.gpu.write_many(writes);
    }

    /// Cache the `KV_PAGE_SIZE`-aligned prefix of `st` in the RadixTree (without GDN
    /// checkpoint if `st.pos` is not at the checkpointed position, or with GDN checkpoint
    /// if `st.pos % KV_PAGE_SIZE == 0`).
    pub fn cache_sequence(&mut self, st: &State) {
        let aligned = (st.pos / KV_PAGE_SIZE) * KV_PAGE_SIZE;
        if aligned == 0 || st.tokens.len() < aligned {
            return;
        }
        let nblocks = aligned / KV_PAGE_SIZE;
        let ckpt = (st.pos == aligned).then(|| Arc::new(self.save_gdn_checkpoint(st, None)));
        self.radix.insert_prefix(
            &st.tokens[..aligned],
            &st.page_table[..nblocks],
            ckpt,
            &mut self.kv_pool,
        );
    }

    /// Prefill `prompt` using RadixAttention prefix matching + chunked prefill.
    /// Returns `(last_token_logits, matched_prefix_tokens)`.
    pub fn prefill_radix(
        &mut self,
        prompt: &[u32],
        chunk: usize,
        st: &mut State,
    ) -> Result<(Vec<f32>, usize)> {
        ensure!(!prompt.is_empty(), "empty prompt");
        let require_gdn = self.has_gdn();
        let m = self
            .radix
            .match_prefix(prompt, &mut self.kv_pool, require_gdn, true);
        let matched = m.matched_tokens;
        if matched > 0 {
            st.page_table.extend_from_slice(&m.pages);
            self.gpu.write_at(&st.page_table_buf, 0, m.pages);
            if let Some(ckpt) = &m.gdn_ckpt {
                self.restore_gdn_checkpoint(st, ckpt);
            }
            st.pos = matched;
            st.tokens.extend_from_slice(&prompt[..matched]);
        }
        let aligned_end = ((prompt.len() - 1) / KV_PAGE_SIZE) * KV_PAGE_SIZE;
        let step = (chunk / KV_PAGE_SIZE).max(1) * KV_PAGE_SIZE;
        let mut logits = Vec::new();
        if matched == 0 && aligned_end > KV_PAGE_SIZE {
            logits = self.forward(&prompt[..KV_PAGE_SIZE], st)?;
            let ckpt = Arc::new(self.save_gdn_checkpoint(st, None));
            self.radix.insert_prefix(
                &prompt[..KV_PAGE_SIZE],
                &st.page_table[..1],
                Some(ckpt),
                &mut self.kv_pool,
            );
        }
        let start_pos = st.pos;
        if start_pos < aligned_end {
            for ch in prompt[start_pos..aligned_end].chunks(step) {
                logits = self.forward(ch, st)?;
                let cur_end = st.pos;
                let ckpt = Arc::new(self.save_gdn_checkpoint(st, None));
                let nblocks = cur_end / KV_PAGE_SIZE;
                self.radix.insert_prefix(
                    &prompt[..cur_end],
                    &st.page_table[..nblocks],
                    Some(ckpt),
                    &mut self.kv_pool,
                );
            }
        }
        let tail_start = matched.max(aligned_end);
        for ch in prompt[tail_start..].chunks(chunk.max(1)) {
            logits = self.forward(ch, st)?;
        }
        Ok((logits, matched))
    }

    fn embed(&self, tokens: &[u32]) -> Vec<f32> {
        let h = self.cfg.hidden;
        let row_bytes = self.embd.ty.row_bytes(h);
        let all = self.gguf.bytes(&self.embd);
        let mut x = vec![0f32; tokens.len() * h];
        for (i, &t) in tokens.iter().enumerate() {
            let src = &all[t as usize * row_bytes..(t as usize + 1) * row_bytes];
            quant::dequant(self.embd.ty, src, &mut x[i * h..(i + 1) * h]);
        }
        x
    }

    /// Run `tokens` through the model, appending to `state`; returns last-token logits.
    pub fn forward(&mut self, tokens: &[u32], st: &mut State) -> Result<Vec<f32>> {
        Ok(self.forward_ext(tokens, st, FwdOpts::default())?.logits)
    }

    /// Run `tokens` through the model, appending to `state`; returns last-token argmax.
    pub fn forward_greedy(&mut self, tokens: &[u32], st: &mut State) -> Result<u32> {
        let out = self.forward_ext(
            tokens,
            st,
            FwdOpts {
                greedy: true,
                ..Default::default()
            },
        )?;
        Ok(out.tokens[0])
    }

    /// [`Self::forward`] with optional all-row logits, the normed hidden states of every
    /// row (the MTP head's input) and GDN rollback snapshots.
    pub fn forward_ext(&mut self, tokens: &[u32], st: &mut State, opts: FwdOpts) -> Result<FwdOut> {
        let t = tokens.len();
        ensure!(t > 0, "empty input");
        if st.pos + t > st.max_ctx {
            bail!("context overflow: {} + {} > {}", st.pos, t, st.max_ctx);
        }
        self.ensure_pages(st, st.pos + t)?;
        st.tokens.extend_from_slice(tokens);
        let c = self.cfg.clone();
        let gm = self.gemm_min_m;
        let x = self.gpu.upload_f32(&self.embed(tokens));
        let pipelined =
            c.n_experts > 0 && (2..=64).contains(&t) && std::env::var_os("BURN_NO_PIPE").is_none();
        let mut pending_promo: Option<(usize, usize, usize)> = None;
        if pipelined {
            self.layers_pipelined(&x, t, st, opts.snap, &mut pending_promo);
        } else {
            let snap = opts.snap.then_some(Snap {
                off: 0,
                save_last: false,
            });
            for li in 0..c.n_layers {
                self.mix(li, &x, t, st.pos, st, snap);
                self.prof.tick(
                    &self.gpu,
                    if matches!(self.layers[li].mixer, Mixer::Gdn(_)) {
                        0
                    } else {
                        1
                    },
                );
                let h2 = self
                    .gpu
                    .rmsnorm(&x, &self.layers[li].post_norm, t, c.hidden, c.eps);
                if self.cfg.n_experts == 0 {
                    let d = &self.denses[li];
                    let a = self.gpu.linear_swiglu(&d.gate, &d.up, &h2, t, gm);
                    self.gpu.linear_acc(&d.down, &a, t, &x, gm);
                } else {
                    self.moe(li, &h2, &x, t, &mut pending_promo)?;
                }
                self.prof.tick(&self.gpu, 2);
            }
        }
        st.pos += t;
        let hidden = c.hidden;
        let xn_all = (opts.all_rows || opts.keep_hidden)
            .then(|| self.gpu.rmsnorm(&x, &self.out_norm, t, hidden, c.eps));
        let (xn, rows) = match (&xn_all, opts.all_rows) {
            (Some(a), true) => (a.clone(), t),
            (Some(a), false) => (
                Buf {
                    h: a.h.clone().offset_start(((t - 1) * hidden * 4) as u64),
                    len: hidden,
                },
                1,
            ),
            (None, _) => {
                let last = Buf {
                    h: x.h.clone().offset_start(((t - 1) * hidden * 4) as u64),
                    len: hidden,
                };
                (self.gpu.rmsnorm(&last, &self.out_norm, 1, hidden, c.eps), 1)
            }
        };
        let logits = self.gpu.alloc(rows * c.vocab);
        self.gpu.qmatmul(&self.lm_head, &xn, rows, &logits, false);
        // Overlap any final-layer promotion repack with the ~2 ms lm_head matmul.
        if let Some((prev_li, prev_e, prev_slot)) = pending_promo.take() {
            self.gpu.flush();
            let tl = Instant::now();
            let packed = self.moes[prev_li].experts.repack_one(&self.gguf, prev_e);
            self.moes[prev_li]
                .experts
                .commit_one(&self.gpu, prev_e, prev_slot, packed);
            self.moes[prev_li].experts.stamp[prev_slot] = self.clock;
            self.stats.upload_secs += tl.elapsed().as_secs_f64();
            self.stats.promotions += 1;
        }
        self.prof.tick(&self.gpu, 7);
        let (logits_out, tokens_out) = if opts.greedy {
            (Vec::new(), self.gpu.argmax_rows(&logits, rows, c.vocab))
        } else {
            (self.gpu.read_f32(&logits), Vec::new())
        };
        Ok(FwdOut {
            logits: logits_out,
            tokens: tokens_out,
            hidden: xn_all,
        })
    }

    /// Project Q/K/V and run `attn_prep_paged`; returns `(q, qg_buf, qg_stride)`.
    fn attn_proj_prep(
        &self,
        w: &AttnW,
        h: &Buf,
        t: usize,
        ai: usize,
        pos: usize,
        st: &State,
    ) -> (Buf, Buf, usize) {
        let c = &self.cfg;
        let gm = self.gemm_min_m;
        let gpu = &self.gpu;
        let (kp, vp) = (&self.kv_pool.k_pools[ai], &self.kv_pool.v_pools[ai]);
        match &w.proj {
            AttnProj::Fused(w_qkv) => {
                let qkv = gpu.linear(w_qkv, h, t, gm);
                let qg_w = c.n_heads * 2 * c.head_dim;
                let kv_w = c.n_kv * c.head_dim;
                let stride = qg_w + 2 * kv_w;
                let q = gpu.attn_prep_paged_strided(
                    &qkv,
                    stride,
                    &qkv,
                    stride,
                    qg_w,
                    &qkv,
                    stride,
                    qg_w + kv_w,
                    &w.q_norm,
                    &w.k_norm,
                    kp,
                    vp,
                    &st.page_table_buf,
                    t,
                    c.n_heads,
                    c.n_kv,
                    c.head_dim,
                    pos,
                    KV_PAGE_SIZE,
                    c.rope_dim,
                    c.rope_theta,
                    c.eps,
                );
                (q, qkv, stride)
            }
            AttnProj::Split {
                q: wq,
                k: wk,
                v: wv,
            } => {
                let qg = gpu.linear(wq, h, t, gm);
                let k = gpu.linear(wk, h, t, gm);
                let v = gpu.linear(wv, h, t, gm);
                let q = gpu.attn_prep_paged(
                    &qg,
                    &k,
                    &v,
                    &w.q_norm,
                    &w.k_norm,
                    kp,
                    vp,
                    &st.page_table_buf,
                    t,
                    c.n_heads,
                    c.n_kv,
                    c.head_dim,
                    pos,
                    KV_PAGE_SIZE,
                    c.rope_dim,
                    c.rope_theta,
                    c.eps,
                );
                (q, qg, c.n_heads * 2 * c.head_dim)
            }
        }
    }

    /// Enqueue layer `li`'s token mixer for the `t` rows of `x` (absolute position `pos`),
    /// accumulating into `x`.
    fn mix(&self, li: usize, x: &Buf, t: usize, pos: usize, st: &State, snap: Option<Snap>) {
        let c = &self.cfg;
        let gm = self.gemm_min_m;
        let gpu = &self.gpu;
        let h = gpu.rmsnorm(x, &self.layers[li].attn_norm, t, c.hidden, c.eps);
        match &self.layers[li].mixer {
            Mixer::Gdn(w) => {
                let ssm_sz = c.ssm_nv * c.ssm_dk * c.ssm_dk;
                let conv_sz = (c.conv_k - 1) * c.conv_dim();
                // Snapshot views starting at this micro-batch's first token.
                let view = |b: &Option<Buf>, sz: usize, s: Snap| {
                    b.as_ref().map(|b| Buf {
                        h: b.h.clone().offset_start((s.off * sz * 4) as u64),
                        len: b.len - s.off * sz,
                    })
                };
                let snaps = snap.map(|s| {
                    (
                        view(&st.conv_snap[li], conv_sz, s),
                        view(&st.ssm_snap[li], ssm_sz, s),
                    )
                });
                let (conv_snap, ssm_snap) = snaps.unwrap_or((None, None));
                let snap_cnt = snap.map_or(0, |s| if s.save_last { t } else { t - 1 });
                let state_conv = st.conv[li].as_ref().unwrap();
                let state_ssm = st.ssm[li].as_ref().unwrap();
                let o = match &w.proj {
                    GdnProj::Fused(w_all) => {
                        let all = gpu.linear(w_all, &h, t, gm);
                        let cdim = c.conv_dim();
                        let z_dim = c.ssm_nv * c.ssm_dk;
                        let stride = cdim + z_dim + 2 * c.ssm_nv;
                        let conv = gpu.gdn_conv_strided(
                            &all,
                            &w.conv_w,
                            state_conv,
                            conv_snap.as_ref().map(|b| (b, snap_cnt)),
                            cdim,
                            stride,
                            t,
                        );
                        gpu.gdn_recurrent_strided(
                            &conv,
                            &all,
                            stride,
                            cdim + z_dim,
                            &all,
                            stride,
                            cdim + z_dim + c.ssm_nv,
                            &all,
                            stride,
                            cdim,
                            &w.ssm_a,
                            &w.dt_bias,
                            &w.norm,
                            state_ssm,
                            ssm_snap.as_ref().map(|b| (b, snap_cnt)),
                            t,
                            c.ssm_nk,
                            c.ssm_nv,
                            c.ssm_dk,
                            c.eps,
                        )
                    }
                    GdnProj::Split {
                        qkv,
                        z,
                        alpha,
                        beta,
                    } => {
                        let qkv = gpu.linear(qkv, &h, t, gm);
                        let z = gpu.linear(z, &h, t, gm);
                        let a = gpu.linear(alpha, &h, t, gm);
                        let b = gpu.linear(beta, &h, t, gm);
                        let conv = gpu.gdn_conv(
                            &qkv,
                            &w.conv_w,
                            state_conv,
                            conv_snap.as_ref().map(|b| (b, snap_cnt)),
                            c.conv_dim(),
                            t,
                        );
                        gpu.gdn_recurrent(
                            &conv,
                            &a,
                            &b,
                            &z,
                            &w.ssm_a,
                            &w.dt_bias,
                            &w.norm,
                            state_ssm,
                            ssm_snap.as_ref().map(|b| (b, snap_cnt)),
                            t,
                            c.ssm_nk,
                            c.ssm_nv,
                            c.ssm_dk,
                            c.eps,
                        )
                    }
                };
                gpu.linear_acc(&w.out, &o, t, x, gm);
            }
            Mixer::Attn(w) => {
                let ai = self.attn_idx[li].expect("full-attn layer index");
                let (kp, vp) = (&self.kv_pool.k_pools[ai], &self.kv_pool.v_pools[ai]);
                let (q, qg, qg_stride) = self.attn_proj_prep(w, &h, t, ai, pos, st);
                let o = gpu.attention_paged_strided(
                    &q,
                    &qg,
                    qg_stride,
                    kp,
                    vp,
                    &st.page_table_buf,
                    t,
                    c.n_heads,
                    c.n_kv,
                    c.head_dim,
                    pos,
                    KV_PAGE_SIZE,
                );
                gpu.linear_acc(&w.o, &o, t, x, gm);
            }
        }
    }

    /// Mixer + post-norm + router for one micro-batch; returns (h2, routing).
    fn mix_route(
        &self,
        li: usize,
        x: &Buf,
        t: usize,
        pos: usize,
        st: &State,
        snap: Option<Snap>,
    ) -> (Buf, Routed) {
        let c = &self.cfg;
        self.mix(li, x, t, pos, st, snap);
        let h2 = self
            .gpu
            .rmsnorm(x, &self.layers[li].post_norm, t, c.hidden, c.eps);
        let r = self.moe_route_enqueue(li, &h2, t);
        (h2, r)
    }

    /// Two micro-batches in flight so one batch's CPU experts overlap the other batch's
    /// GPU mixer: per layer, A's CPU experts run while B's mixer is enqueued, and B's
    /// CPU experts run while A's next-layer mixer is enqueued. B only depends on A
    /// through the per-layer GDN state / KV cache, which the stream order preserves.
    fn layers_pipelined(
        &mut self,
        x: &Buf,
        t: usize,
        st: &State,
        snap: bool,
        pending: &mut Option<(usize, usize, usize)>,
    ) {
        let n_layers = self.cfg.n_layers;
        let hd = self.cfg.hidden;
        let ta = t / 2;
        let mb = [(0usize, ta), (ta, t - ta)];
        let xs = mb.map(|(r0, n)| Buf {
            h: x.h.clone().offset_start((r0 * hd * 4) as u64),
            len: n * hd,
        });
        let snaps = [
            snap.then_some(Snap {
                off: 0,
                save_last: true,
            }),
            snap.then_some(Snap {
                off: ta,
                save_last: false,
            }),
        ];
        let pos = st.pos;
        let read_client = self.gpu.client.clone();
        let r0 = self.mix_route(0, &xs[0], ta, pos, st, snaps[0]);
        let f0 = Gpu::read_u32_async(&read_client, &r0.1.pack);
        let mut routed = [Some((r0.0, r0.1, f0)), None];
        let mut ready_promo: Option<(usize, usize, usize, [quant::Packed; 3])> = None;
        let mut pending_finish: Option<(usize, MoeJob)> = None;
        for li in 0..n_layers {
            for m in 0..2 {
                let (h2, r, pack_fut) = routed[m].take().expect("micro-batch routed");
                let ts = Instant::now();
                let pack_host = pack_fut.wait();
                self.stats.route_sync_secs += ts.elapsed().as_secs_f64();
                if let Some((prev_li, prev_e, prev_slot, packed)) = ready_promo.take() {
                    let tl = Instant::now();
                    self.moes[prev_li]
                        .experts
                        .commit_one(&self.gpu, prev_e, prev_slot, packed);
                    self.moes[prev_li].experts.stamp[prev_slot] = self.clock;
                    self.stats.upload_secs += tl.elapsed().as_secs_f64();
                    self.stats.promotions += 1;
                }
                let prev_fin = pending_finish.take().map(|(pm, pjob)| {
                    let cpu = (!pjob.misses.is_empty()).then(|| self.gpu.upload_f32(&pjob.pack));
                    (&xs[pm], pjob, cpu)
                });
                let mut job =
                    self.moe_begin_host(li, &h2, mb[m].1, &pack_host, r, pending, prev_fin);
                let promo_req = pending.take();
                let (nli, nm) = if m == 0 { (li, 1) } else { (li + 1, 0) };
                let this = &*self;
                let rc = &read_client;
                let tc = Instant::now();
                let (next_routed, next_ready) = cpu_moe::overlap(
                    || this.moe_cpu(&mut job),
                    || {
                        let nr = if nli < n_layers {
                            let (nh2, nr) = this.mix_route(
                                nli,
                                &xs[nm],
                                mb[nm].1,
                                pos + mb[nm].0,
                                st,
                                snaps[nm],
                            );
                            // Issue the async readback immediately so wgpu submits the whole
                            // compute pass + staging copy in one queue.submit() while moe_cpu runs.
                            let nf = Gpu::read_u32_async(rc, &nr.pack);
                            Some((nh2, nr, nf))
                        } else {
                            this.gpu.flush();
                            None
                        };
                        let np = promo_req.map(|(p_li, p_e, p_slot)| {
                            let p = this.moes[p_li].experts.repack_one(&this.gguf, p_e);
                            (p_li, p_e, p_slot, p)
                        });
                        (nr, np)
                    },
                );
                ready_promo = next_ready;
                self.stats.cpu_secs += tc.elapsed().as_secs_f64();
                routed[nm] = next_routed;
                pending_finish = Some((m, job));
            }
        }
        if let Some((pm, pjob)) = pending_finish.take() {
            self.moe_finish(&xs[pm], pjob);
        }
        if let Some((prev_li, prev_e, prev_slot, packed)) = ready_promo.take() {
            let tl = Instant::now();
            self.moes[prev_li]
                .experts
                .commit_one(&self.gpu, prev_e, prev_slot, packed);
            self.moes[prev_li].experts.stamp[prev_slot] = self.clock;
            self.stats.upload_secs += tl.elapsed().as_secs_f64();
            self.stats.promotions += 1;
        }
    }

    /// LFU promotion candidate: hottest missed expert replacing the coldest slot, if worthwhile.
    fn pick_promo(&self, li: usize, misses: &[usize]) -> Option<(usize, usize, usize)> {
        let st = &self.moes[li].experts;
        if st.slots == 0 {
            return None;
        }
        let &best_e = misses.iter().max_by_key(|&&e| st.freq[e])?;
        let best_f = st.freq[best_e];
        let victim = (0..st.slots)
            .min_by_key(|&s| {
                let cur = st.expert_in[s];
                let f = if cur >= 0 { st.freq[cur as usize] } else { 0 };
                (f, st.stamp[s])
            })
            .unwrap();
        let cur = st.expert_in[victim];
        let cur_f = if cur >= 0 { st.freq[cur as usize] } else { 0 };
        ((cur_f == 0 && best_f >= 2) || (best_f >= cur_f * 2 + 8)).then_some((li, best_e, victim))
    }

    /// Enqueue the router and the packed (ids, weights, h2) readback buffer.
    fn moe_route_enqueue(&self, li: usize, h2: &Buf, t: usize) -> Routed {
        let c = &self.cfg;
        let (ne, topk) = (c.n_experts, c.topk);
        let logits = self
            .gpu
            .linear(&self.moes[li].router, h2, t, self.gemm_min_m);
        let (_ids, wts, pack) = self
            .gpu
            .moe_route_pack(&logits, h2, t, ne, ne + 1, c.hidden, topk);
        Routed { logits, wts, pack }
    }

    /// Read the routing back (the per-layer sync), overlapping the repack of any pending
    /// promotion with the GPU work still in flight, then commit it.
    fn moe_sync(
        &mut self,
        r: &Routed,
        pending_promo: &mut Option<(usize, usize, usize)>,
    ) -> Vec<u32> {
        let read_client = self.gpu.client.clone();
        let pack_fut = Gpu::read_u32_async(&read_client, &r.pack);
        let ready_promo = pending_promo.take().map(|(prev_li, prev_e, prev_slot)| {
            let tl = Instant::now();
            let packed = self.moes[prev_li].experts.repack_one(&self.gguf, prev_e);
            self.stats.upload_secs += tl.elapsed().as_secs_f64();
            (prev_li, prev_e, prev_slot, packed)
        });
        let ts = Instant::now();
        let pack_host = pack_fut.wait();
        self.stats.route_sync_secs += ts.elapsed().as_secs_f64();
        if let Some((prev_li, prev_e, prev_slot, packed)) = ready_promo {
            let tl = Instant::now();
            self.moes[prev_li]
                .experts
                .commit_one(&self.gpu, prev_e, prev_slot, packed);
            self.moes[prev_li].experts.stamp[prev_slot] = self.clock;
            self.stats.upload_secs += tl.elapsed().as_secs_f64();
            self.stats.promotions += 1;
        }
        pack_host
    }

    /// [`Self::moe_sync`] + [`Self::moe_begin_host`].
    fn moe_begin(
        &mut self,
        li: usize,
        h2: &Buf,
        t: usize,
        r: Routed,
        pending_promo: &mut Option<(usize, usize, usize)>,
    ) -> MoeJob {
        let pack_host = self.moe_sync(&r, pending_promo);
        self.moe_begin_host(li, h2, t, &pack_host, r, pending_promo, None)
    }

    /// Multi-token hybrid MoE, first half: partition routed pairs into VRAM hits and
    /// misses, enqueue the shared expert + hit experts on the GPU, and pick a promotion.
    /// The misses are left for [`Self::moe_cpu`].
    #[allow(clippy::too_many_arguments)]
    fn moe_begin_host(
        &mut self,
        li: usize,
        h2: &Buf,
        t: usize,
        pack_host: &[u32],
        r: Routed,
        pending_promo: &mut Option<(usize, usize, usize)>,
        prev_finish: Option<(&Buf, MoeJob, Option<Buf>)>,
    ) -> MoeJob {
        let c = &self.cfg;
        let (topk, inter, hidden) = (c.topk, c.expert_inter, c.hidden);
        let gm = self.gemm_min_m;
        let npairs = t * topk;
        let ids_host = &pack_host[..npairs];
        let wts_host: &[f32] = bytemuck::cast_slice(&pack_host[npairs..2 * npairs]);
        let h2_host: &[f32] = bytemuck::cast_slice(&pack_host[2 * npairs..2 * npairs + t * hidden]);

        let mut hit_desc = Vec::with_capacity(npairs * MOE_DESC);
        let mut max_cnt = 0usize;
        // Hit weights first; CPU-computed pairs get weight 0 (their sum arrives premixed).
        let mut pack = vec![0.0f32; npairs + t * hidden];
        pack[..npairs].copy_from_slice(wts_host);
        let mut misses = Vec::new();
        {
            let st = &mut self.moes[li].experts;
            let mut sorted: Vec<(u32, u32)> = ids_host
                .iter()
                .enumerate()
                .map(|(p, &e)| (e, p as u32))
                .collect();
            sorted.sort_unstable_by_key(|&(e, p)| (e, p));
            let mut i = 0usize;
            while i < sorted.len() {
                let e = sorted[i].0 as usize;
                let mut j = i + 1;
                while j < sorted.len() && sorted[j].0 as usize == e {
                    j += 1;
                }
                let cnt = j - i;
                self.clock += 1;
                st.freq[e] = st.freq[e].saturating_add(cnt as u32);
                if st.slot_of[e] >= 0 {
                    let slot = st.slot_of[e] as usize;
                    st.stamp[slot] = self.clock;
                    self.stats.expert_hits += cnt as u64;
                    for chunk in sorted[i..j].chunks(MOE_MAXM) {
                        max_cnt = max_cnt.max(chunk.len());
                        hit_desc.push(slot as u32);
                        hit_desc.push(chunk.len() as u32);
                        hit_desc.extend((0..MOE_MAXM).map(|k| chunk.get(k).map_or(0, |&(_, p)| p)));
                    }
                } else {
                    self.stats.expert_misses += cnt as u64;
                    let rows = sorted[i..j]
                        .iter()
                        .map(|&(_, p)| {
                            pack[p as usize] = 0.0;
                            ((p as usize) / topk, wts_host[p as usize])
                        })
                        .collect();
                    misses.push((e, rows));
                }
                i = j;
            }
        }

        // Upload descriptors BEFORE launching any compute kernels so wgpu emits zero
        // mid-pass queue.submit() flushes between host writes and compute dispatches.
        let desc_buf = (!hit_desc.is_empty()).then(|| self.gpu.upload_u32(&hit_desc));
        if let Some((px, pjob, pcpu)) = prev_finish {
            self.moe_combine_job(px, &pjob, pcpu.as_ref());
        }
        let w = &self.moes[li];
        let a = self.gpu.linear_swiglu(&w.gate_sh, &w.up_sh, h2, t, gm);
        let sh = self.gpu.linear(&w.down_sh, &a, t, gm);
        let y = self.gpu.alloc(npairs * hidden);
        if let Some(desc_buf) = desc_buf {
            let act = self.gpu.alloc(npairs * inter);
            let ntiles = hit_desc.len() / MOE_DESC;
            self.run_experts(li, &desc_buf, ntiles, max_cnt, h2, &act, &y);
        }
        if t <= 8 && !misses.is_empty() {
            let ids: Vec<usize> = misses.iter().map(|m| m.0).collect();
            *pending_promo = self.pick_promo(li, &ids);
        }
        MoeJob {
            li,
            t,
            logits: r.logits,
            wts: r.wts,
            sh,
            y,
            misses,
            h2_host: h2_host.to_vec(),
            pack,
        }
    }

    /// Multi-token hybrid MoE, CPU half: compute the missed experts into `job.pack`.
    /// Touches no GPU or mutable model state, so it can run alongside GPU enqueueing.
    fn moe_cpu(&self, job: &mut MoeJob) {
        if job.misses.is_empty() {
            return;
        }
        let c = &self.cfg;
        let st = &self.moes[job.li].experts;
        let jobs: Vec<CpuExpertJob<'_>> = job
            .misses
            .iter()
            .map(|(e, pairs)| CpuExpertJob {
                gate_raw: st.gate.raw(&self.gguf, *e),
                up_raw: st.up.raw(&self.gguf, *e),
                down_raw: st.down.raw(&self.gguf, *e),
                pairs: pairs.clone(),
            })
            .collect();
        let npairs = job.t * c.topk;
        cpu_moe::run_experts_cpu(
            st.gate.info.ty,
            st.up.info.ty,
            st.down.info.ty,
            &jobs,
            &job.h2_host,
            job.t,
            c.hidden,
            c.expert_inter,
            &mut job.pack[npairs..],
        );
    }

    fn moe_combine_job(&self, x: &Buf, job: &MoeJob, cpu: Option<&Buf>) {
        let c = &self.cfg;
        let (ne, topk) = (c.n_experts, c.topk);
        self.gpu.moe_combine(
            x,
            &job.y,
            &job.wts,
            &job.sh,
            &job.logits,
            cpu,
            job.t,
            c.hidden,
            topk,
            ne + 1,
            ne,
        );
    }

    /// Multi-token hybrid MoE, last half: upload the CPU results and combine into `x`.
    fn moe_finish(&self, x: &Buf, job: MoeJob) {
        let cpu = (!job.misses.is_empty()).then(|| self.gpu.upload_f32(&job.pack));
        self.moe_combine_job(x, &job, cpu.as_ref());
    }

    fn moe(
        &mut self,
        li: usize,
        h2: &Buf,
        x: &Buf,
        t: usize,
        pending_promo: &mut Option<(usize, usize, usize)>,
    ) -> Result<()> {
        let c = &self.cfg.clone();
        let (ne, topk, inter, hidden) = (c.n_experts, c.topk, c.expert_inter, c.hidden);
        let gm = self.gemm_min_m;
        let npairs = t * topk;

        if self.moes[li].experts.resident(ne) && t <= 2 {
            // Fully resident: no host round-trip, one tile per (token, k) pair.
            let gpu = &self.gpu;
            let w = &self.moes[li];
            let logits = gpu.linear(&w.router, h2, t, gm);
            let (ids, wts) = gpu.moe_route(&logits, t, ne, ne + 1, topk);
            let a = gpu.linear_swiglu(&w.gate_sh, &w.up_sh, h2, t, gm);
            let sh = gpu.linear(&w.down_sh, &a, t, gm);
            let act = gpu.alloc(npairs * inter);
            let y = gpu.alloc(npairs * hidden);
            let desc = gpu.moe_direct_desc(&ids, npairs);
            self.run_experts(li, &desc, npairs, 1, h2, &act, &y);
            self.gpu
                .moe_combine(x, &y, &wts, &sh, &logits, None, t, hidden, topk, ne + 1, ne);
            return Ok(());
        }

        if t <= 64 {
            // Hybrid GPU + CPU path:
            // 1. Route and pack (ids, wts, h2) in a single device-to-host read BEFORE
            //    launching the shared expert, so the sync returns immediately after routing.
            let r = self.moe_route_enqueue(li, h2, t);
            let pack_host = self.moe_sync(&r, pending_promo);
            if t > 1 {
                let mut job = self.moe_begin_host(li, h2, t, &pack_host, r, pending_promo, None);
                if !job.misses.is_empty() {
                    self.gpu.flush();
                }
                let tc = Instant::now();
                self.moe_cpu(&mut job);
                self.stats.cpu_secs += tc.elapsed().as_secs_f64();
                self.moe_finish(x, job);
                return Ok(());
            }
            let Routed { logits, wts, .. } = r;

            let ids_host = &pack_host[..npairs];
            let wts_host: &[f32] = bytemuck::cast_slice(&pack_host[npairs..2 * npairs]);
            let h2_host: &[f32] =
                bytemuck::cast_slice(&pack_host[2 * npairs..2 * npairs + t * hidden]);

            // 2. Partition routed pairs into VRAM-resident hits (GPU) and misses (CPU).
            if t == 1 {
                let mut hit_desc = [0u32; MOE_MAXM * MOE_DESC];
                let mut hit_tiles = 0usize;
                let mut miss_experts = [0usize; MOE_MAXM];
                let mut miss_weights = [0.0f32; MOE_MAXM];
                let mut n_miss = 0usize;
                let mut hit_wts = [0.0f32; MOE_MAXM];

                {
                    let st = &mut self.moes[li].experts;
                    for (p, &eu) in ids_host.iter().enumerate() {
                        let e = eu as usize;
                        self.clock += 1;
                        st.freq[e] = st.freq[e].saturating_add(1);
                        if st.slot_of[e] >= 0 {
                            let slot = st.slot_of[e] as usize;
                            st.stamp[slot] = self.clock;
                            self.stats.expert_hits += 1;
                            let off = hit_tiles * MOE_DESC;
                            hit_desc[off] = slot as u32;
                            hit_desc[off + 1] = 1;
                            hit_desc[off + 2] = p as u32;
                            hit_tiles += 1;
                            hit_wts[p] = wts_host[p];
                        } else {
                            self.stats.expert_misses += 1;
                            miss_experts[n_miss] = e;
                            miss_weights[n_miss] = wts_host[p];
                            n_miss += 1;
                        }
                    }
                }

                // 3. Immediately enqueue shared expert + hit experts on the GPU so they run
                //    in parallel with CPU execution of the missed experts. Upload descriptors
                //    first while the wgpu command encoder is empty (zero extra submit).
                let desc_buf =
                    (hit_tiles > 0).then(|| self.gpu.upload_u32(&hit_desc[..hit_tiles * MOE_DESC]));
                let w = &self.moes[li];
                let a = self.gpu.linear_swiglu(&w.gate_sh, &w.up_sh, h2, 1, gm);
                let sh = self.gpu.linear(&w.down_sh, &a, 1, gm);

                let y = self.gpu.alloc(npairs * hidden);
                if let Some(desc_buf) = desc_buf {
                    let act = self.gpu.alloc(npairs * inter);
                    self.run_experts(li, &desc_buf, hit_tiles, 1, h2, &act, &y);
                }

                // 4. Compute missed experts on CPU while GPU runs shared + hit experts.
                if n_miss > 0 {
                    self.gpu.flush();
                    let st = &self.moes[li].experts;
                    let mut jobs_arr = [CpuDecodeJob {
                        gate_raw: &[],
                        up_raw: &[],
                        down_raw: &[],
                        weight: 0.0,
                    }; MOE_MAXM];
                    for i in 0..n_miss {
                        let e = miss_experts[i];
                        jobs_arr[i] = CpuDecodeJob {
                            gate_raw: st.gate.raw(&self.gguf, e),
                            up_raw: st.up.raw(&self.gguf, e),
                            down_raw: st.down.raw(&self.gguf, e),
                            weight: miss_weights[i],
                        };
                    }
                    let tc = Instant::now();
                    let mut cpu_pack = vec![0.0f32; topk + hidden];
                    cpu_pack[..topk].copy_from_slice(&hit_wts[..topk]);
                    cpu_moe::run_experts_cpu_decode(
                        st.gate.info.ty,
                        st.up.info.ty,
                        st.down.info.ty,
                        &jobs_arr[..n_miss],
                        h2_host,
                        hidden,
                        inter,
                        &mut cpu_pack[topk..],
                    );
                    self.stats.cpu_secs += tc.elapsed().as_secs_f64();
                    let cpu_pack_buf = self.gpu.upload_f32(&cpu_pack);
                    self.gpu.moe_combine(
                        x,
                        &y,
                        &wts,
                        &sh,
                        &logits,
                        Some(&cpu_pack_buf),
                        1,
                        hidden,
                        topk,
                        ne + 1,
                        ne,
                    );

                    // 5. LFU cache promotion: defer repack to next layer's GPU mixer window.
                    *pending_promo = self.pick_promo(li, &miss_experts[..n_miss]);
                } else {
                    self.gpu.moe_combine(
                        x,
                        &y,
                        &wts,
                        &sh,
                        &logits,
                        None,
                        1,
                        hidden,
                        topk,
                        ne + 1,
                        ne,
                    );
                }
                return Ok(());
            }
        }

        // Large-batch prefill (t > 64): stream missed experts onto GPU in slot-sized rounds.
        self.prof.tick(&self.gpu, 2);
        let gpu = &self.gpu;
        let w = &self.moes[li];
        let logits = gpu.linear(&w.router, h2, t, gm);
        let (ids, wts) = gpu.moe_route(&logits, t, ne, ne + 1, topk);
        self.prof.tick(gpu, 2);
        let a = gpu.linear_swiglu(&w.gate_sh, &w.up_sh, h2, t, gm);
        let sh = gpu.linear(&w.down_sh, &a, t, gm);
        self.prof.tick(gpu, 3);
        let act = gpu.alloc(npairs * inter);
        let y = gpu.alloc(npairs * hidden);

        let ts = Instant::now();
        let ids_host = gpu.read_u32(&ids);
        self.stats.route_sync_secs += ts.elapsed().as_secs_f64();
        let mut by_expert: Vec<Vec<u32>> = vec![Vec::new(); ne];
        for (p, &e) in ids_host.iter().enumerate() {
            by_expert[e as usize].push(p as u32);
        }
        let store = &self.moes[li].experts;
        let mut needed: Vec<usize> = (0..ne).filter(|&e| !by_expert[e].is_empty()).collect();
        needed.sort_by_key(|&e| store.slot_of[e] < 0);
        let slots = store.slots;
        for round in needed.chunks(slots) {
            let st = &self.moes[li].experts;
            let mut pinned = vec![false; slots];
            for &e in round {
                if st.slot_of[e] >= 0 {
                    pinned[st.slot_of[e] as usize] = true;
                }
            }
            let mut desc = Vec::new();
            let mut max_cnt = 0;
            let mut misses = Vec::new();
            let st = &mut self.moes[li].experts;
            for &e in round {
                self.clock += 1;
                st.freq[e] = st.freq[e].saturating_add(by_expert[e].len() as u32);
                let slot = if st.slot_of[e] >= 0 {
                    self.stats.expert_hits += 1;
                    st.slot_of[e] as usize
                } else {
                    self.stats.expert_misses += 1;
                    let victim = (0..slots)
                        .filter(|&s| !pinned[s])
                        .min_by_key(|&s| st.stamp[s])
                        .context("no evictable expert slot")?;
                    misses.push((e, victim));
                    victim
                };
                pinned[slot] = true;
                st.stamp[slot] = self.clock;
                for chunk in by_expert[e].chunks(MOE_MAXM) {
                    max_cnt = max_cnt.max(chunk.len());
                    desc.push(slot as u32);
                    desc.push(chunk.len() as u32);
                    desc.extend((0..MOE_MAXM).map(|i| chunk.get(i).copied().unwrap_or(0)));
                }
            }
            let tl = Instant::now();
            st.load_many(&self.gpu, &self.gguf, &misses);
            self.stats.upload_secs += tl.elapsed().as_secs_f64();
            self.prof.tick(&self.gpu, 5);
            let ntiles = desc.len() / MOE_DESC;
            let desc = self.gpu.upload_u32(&desc);
            self.run_experts(li, &desc, ntiles, max_cnt, h2, &act, &y);
            self.prof.tick(&self.gpu, 4);
        }
        self.gpu
            .moe_combine(x, &y, &wts, &sh, &logits, None, t, hidden, topk, ne + 1, ne);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_experts(
        &self,
        li: usize,
        desc: &Buf,
        ntiles: usize,
        max_cnt: usize,
        h2: &Buf,
        act: &Buf,
        y: &Buf,
    ) {
        let c = &self.cfg;
        let st = &self.moes[li].experts;
        let (inter, hidden) = (c.expert_inter, c.hidden);
        let (g, u, d) = (st.gate.arg(), st.up.arg(), st.down.arg());
        self.gpu.moe_gate_up(
            g,
            u,
            st.gate.fmt,
            h2,
            desc,
            ntiles,
            act,
            inter,
            hidden,
            c.topk,
            max_cnt,
        );
        self.gpu
            .moe_down(d, st.down.fmt, act, desc, ntiles, y, hidden, inter, max_cnt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Kolibri 1.1 GGUFs report `general.architecture=kolibri1`; the arch string alone must
    /// not make us claim 1.0 support.
    #[test]
    fn kolibri_1_1_is_named() {
        let path = Path::new("/mnt/models/Aleph-Alpha/Kolibri-1.1-GGUF/Kolibri-1.1-Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("skipping: Kolibri 1.1 GGUF not present");
            return;
        }
        let g = Gguf::open(path).unwrap();
        let err = Config::from_gguf(&g).unwrap_err();
        assert!(
            err.to_string().contains("kolibri1_1"),
            "unexpected error: {err:#}"
        );
    }
}
