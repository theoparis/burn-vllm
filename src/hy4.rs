//! Hy4-flash (`hy_v4_flash`) config parsing and tensor validation (P1).
//!
//! This is infrastructure only: no forward pass yet. The GGUF is two shards; shard 1
//! contains 1415 tensors (table in `hy4_tensors.rs`) and shard 2 a single F8_E4M3 tensor
//! which this reader does not support yet, so validation covers shard 1.

use crate::gguf::{Gguf, Value};
use anyhow::{Result, ensure};

#[derive(Debug, Clone)]
pub struct Config {
    pub n_layers: usize,
    pub leading_dense: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub ctx: usize,
    pub ff_exp: usize,
    pub expert_ff_exp: usize,
    pub expert_latent: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_expert_shared: usize,
    pub expert_weights_scale: f32,
    pub expert_weights_norm: bool,
    pub expert_gating: u64,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub key_len: usize,
    pub value_len: usize,
    pub key_len_mla: usize,
    pub value_len_mla: usize,
    pub qk_rope_dim: usize,
    pub head_counts: Vec<usize>,
    pub head_count_kv: usize,
    pub indexer_heads: usize,
    pub indexer_key_len: usize,
    pub indexer_top_k: usize,
    pub indexer_types: Vec<bool>,
    pub indexer_kpool: usize,
    pub indexer_kpool_select_tail: bool,
    pub eps: f32,
    pub freq_base: f64,
    pub freq_base_swa: f64,
    pub swa: usize,
    pub swa_pattern: Vec<bool>,
    pub hc_count: usize,
    pub hc_eps: f32,
    pub hc_magnitude: f32,
    pub ple_layers: Vec<usize>,
    pub ple_ngram: usize,
    pub ple_heads_per_ngram: usize,
    pub ple_head_dim: usize,
    pub engram_pad: usize,
    pub nextn_layers: usize,
}

fn u64_array(v: &[Value]) -> Vec<u64> {
    v.iter().filter_map(Value::as_u64).collect()
}

fn bool_array(v: &[Value]) -> Vec<bool> {
    v.iter()
        .filter_map(|x| match x {
            Value::Bool(b) => Some(*b),
            other => other.as_u64().map(|n| n != 0),
        })
        .collect()
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let p = "hy_v4_flash.";
        let u = |s: &str| g.meta_u64(&format!("{p}{s}")).map(|v| v as usize);
        let f = |s: &str| g.meta_f64(&format!("{p}{s}"));
        let arr = |s: &str| g.meta_array(&format!("{p}{s}"));
        let bool_key = |s: &str| -> Result<bool> {
            // GGUF bools are u8 and `meta_u64` accepts them; these keys are scalars.
            Ok(g.meta_u64(&format!("{p}{s}"))? != 0)
        };

        let head_counts: Vec<usize> = u64_array(arr("attention.head_count")?)
            .into_iter()
            .map(|v| v as usize)
            .collect();
        let swa_pattern = bool_array(arr("attention.sliding_window_pattern")?);
        let indexer_types = bool_array(arr("attention.indexer.types")?);
        let ple_layers: Vec<usize> = u64_array(arr("ple.layers")?)
            .into_iter()
            .map(|v| v as usize)
            .collect();

        let cfg = Self {
            n_layers: u("block_count")?,
            leading_dense: u("leading_dense_block_count")?,
            hidden: u("embedding_length")?,
            vocab: u("vocab_size")?,
            ctx: u("context_length")?,
            ff_exp: u("feed_forward_length")?,
            expert_ff_exp: u("expert_feed_forward_length")?,
            expert_latent: u("expert_latent_length")?,
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            n_expert_shared: u("expert_shared_count")?,
            expert_weights_scale: f("expert_weights_scale")? as f32,
            expert_weights_norm: bool_key("expert_weights_norm")?,
            expert_gating: g.meta_u64(&format!("{p}expert_gating_func"))?,
            q_lora_rank: u("attention.q_lora_rank")?,
            kv_lora_rank: u("attention.kv_lora_rank")?,
            key_len: u("attention.key_length")?,
            value_len: u("attention.value_length")?,
            key_len_mla: u("attention.key_length_mla")?,
            value_len_mla: u("attention.value_length_mla")?,
            qk_rope_dim: u("rope.dimension_count")?,
            head_counts,
            head_count_kv: u("attention.head_count_kv")?,
            indexer_heads: u("attention.indexer.head_count")?,
            indexer_key_len: u("attention.indexer.key_length")?,
            indexer_top_k: u("attention.indexer.top_k")?,
            indexer_types,
            indexer_kpool: u("attention.indexer.kpool")?,
            indexer_kpool_select_tail: bool_key("attention.indexer.kpool_select_tail")?,
            eps: f("attention.layer_norm_rms_epsilon")? as f32,
            freq_base: f("rope.freq_base")?,
            freq_base_swa: f("rope.freq_base_swa")?,
            swa: u("attention.sliding_window")?,
            swa_pattern,
            hc_count: u("hyper_connection.count")?,
            hc_eps: f("hyper_connection.epsilon")? as f32,
            hc_magnitude: f("hyper_connection.magnitude")? as f32,
            ple_layers,
            ple_ngram: u("ple.ngram_size")?,
            ple_heads_per_ngram: u("ple.heads_per_ngram")?,
            ple_head_dim: u("embedding_length_per_layer_input")?,
            engram_pad: u("engram.pad_token_id")?,
            nextn_layers: u("nextn_predict_layers")?,
        };
        cfg.check()?;
        Ok(cfg)
    }

    fn check(&self) -> Result<()> {
        ensure!(self.n_layers > self.leading_dense, "no sparse blocks");
        ensure!(
            self.head_counts.len() == self.n_layers,
            "head_count array {} != block_count {}",
            self.head_counts.len(),
            self.n_layers
        );
        ensure!(
            self.swa_pattern.len() == self.n_layers,
            "sliding_window_pattern {} != block_count {}",
            self.swa_pattern.len(),
            self.n_layers
        );
        // The leading dense block has no indexer, so the type array covers sparse layers only.
        ensure!(
            self.indexer_types.len() == self.n_layers - self.leading_dense,
            "indexer.types {} != sparse layers {}",
            self.indexer_types.len(),
            self.n_layers - self.leading_dense
        );
        ensure!(self.head_count_kv == 1, "expected MLA with one KV head");
        ensure!(
            self.indexer_heads > 0 && self.indexer_top_k > 0,
            "invalid indexer config"
        );
        ensure!(
            self.q_lora_rank > 0 && self.kv_lora_rank > 0,
            "invalid MLA ranks"
        );
        Ok(())
    }

    pub fn summary(&self) -> String {
        format!(
            "hy4-flash: {} layers ({} dense), hidden {}, vocab {}, experts {}/{}, \
             heads {:?}/kv{}, MLA q/kv {}//{}, indexer {}:{} top{}, swa {}, iHC {}, PLE {:?}",
            self.n_layers,
            self.leading_dense,
            self.hidden,
            self.vocab,
            self.n_expert_used,
            self.n_expert,
            self.head_counts
                .iter()
                .step_by(6)
                .copied()
                .collect::<Vec<_>>(),
            self.head_count_kv,
            self.q_lora_rank,
            self.kv_lora_rank,
            self.indexer_heads,
            self.indexer_key_len,
            self.indexer_top_k,
            self.swa,
            self.hc_count,
            self.ple_layers,
        )
    }
}

/// Verify all expected shard-1 tensors against the GGUF header (name, dims, dtype).
pub fn validate(g: &Gguf) -> Result<()> {
    let mut missing = Vec::new();
    let mut mismatched = Vec::new();
    for &(name, dims, ty) in crate::hy4_tensors::TENSORS {
        match g.tensor(name) {
            Ok(t) => {
                if t.dims.as_slice() != dims || t.ty != ty {
                    mismatched.push(format!(
                        "{name}: dims {:?}/{:?} type {:?}/{:?}",
                        t.dims, dims, t.ty, ty
                    ));
                }
            }
            Err(_) => missing.push(name.to_string()),
        }
    }
    ensure!(
        missing.is_empty() && mismatched.is_empty(),
        "hy4 tensor inventory mismatch: {} missing, {} mismatched\n{}\n{}",
        missing.len(),
        mismatched.len(),
        missing
            .iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n"),
        mismatched
            .iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn hy4_config_and_inventory() {
        let path = Path::new("/mnt/models/hy4-flash-gguf/Hy4-flash-Q4_K_M-00001-of-00002.gguf");
        if !path.exists() {
            eprintln!("skipping: Hy4-flash GGUF not present");
            return;
        }
        let g = Gguf::open(path).unwrap();
        let cfg = Config::from_gguf(&g).unwrap();
        assert_eq!(cfg.n_layers, 43);
        assert_eq!(cfg.n_expert, 512);
        assert_eq!(cfg.indexer_top_k, 2048);
        assert!(cfg.summary().contains("hy4-flash"));
        validate(&g).unwrap();
    }
}
