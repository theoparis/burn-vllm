//! Generated from the peer's Hy4-flash tensor inventory (shard 1).
//! Expected GGUF tensor name, dims (ggml order), and type.

pub static TENSORS: &[(&str, &[u64], crate::quant::GgmlType)] = &[
    (
        "output.weight",
        &[3072, 120832],
        crate::quant::GgmlType::Bf16,
    ),
    ("output_hc_base.weight", &[4], crate::quant::GgmlType::F32),
    (
        "output_hc_fn.weight",
        &[12288, 4],
        crate::quant::GgmlType::F32,
    ),
    ("output_hc_scale.weight", &[1], crate::quant::GgmlType::F32),
    ("output_norm.weight", &[3072], crate::quant::GgmlType::F32),
    (
        "token_embd.weight",
        &[3072, 120832],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.0.attn_gate.weight",
        &[3072, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.0.attn_k_b.weight",
        &[192, 512, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.0.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.0.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.0.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.0.attn_output.weight",
        &[8192, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.0.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.0.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.0.attn_q_b.weight",
        &[1024, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.0.attn_sinks.weight",
        &[32],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.0.attn_v_b.weight",
        &[512, 256, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.0.ffn_down.weight",
        &[13824, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.0.ffn_gate.weight",
        &[3072, 13824],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.0.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.0.ffn_up.weight",
        &[3072, 13824],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.0.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.0.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.0.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.0.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.0.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.0.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.0.indexer.attn_k.weight",
        &[3072, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.0.indexer.attn_q_b.weight",
        &[1024, 2048],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.0.indexer.k_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.0.indexer.proj.weight",
        &[3072, 16],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.0.indexer_compressor_ape.weight",
        &[128, 4],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.0.indexer_compressor_gate.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.0.indexer_compressor_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.0.indexer_compressor_value.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.0.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.0.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.attn_gate.weight",
        &[3072, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.1.attn_k_b.weight",
        &[192, 512, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.1.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.1.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.attn_output.weight",
        &[8192, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.1.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.1.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.attn_q_b.weight",
        &[1024, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.1.attn_sinks.weight",
        &[32],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.attn_v_b.weight",
        &[512, 256, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.1.engram_gate.weight",
        &[3072, 4],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.engram_hc_scale.weight",
        &[3072, 4],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.engram_kv.weight",
        &[1536, 15360],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.1.engram_kv_b.weight",
        &[3072, 5],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.engram_lookup_norm.weight",
        &[96, 16],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.engram_pre_gamma.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q6K,
    ),
    (
        "blk.1.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.1.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.1.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.1.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.1.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.1.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.1.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.1.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.indexer.attn_k.weight",
        &[3072, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.1.indexer.attn_q_b.weight",
        &[1024, 2048],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.1.indexer.k_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.indexer.proj.weight",
        &[3072, 16],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.indexer_compressor_ape.weight",
        &[128, 4],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.indexer_compressor_gate.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.1.indexer_compressor_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.indexer_compressor_value.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.1.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.1.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.2.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.2.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.2.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.2.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.2.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.2.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.2.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.2.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.2.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.2.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.2.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.2.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.2.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q6K,
    ),
    (
        "blk.2.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.2.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.2.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.2.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.2.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.2.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.2.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.2.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.2.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.2.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.2.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.2.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.2.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.2.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.2.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.2.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.2.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.3.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.3.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.3.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.3.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.3.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.3.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.3.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.3.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.3.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.3.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.3.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.3.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.3.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q6K,
    ),
    (
        "blk.3.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.3.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.3.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.3.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.3.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.3.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.3.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.3.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.3.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.3.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.3.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.3.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.3.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.3.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.3.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.3.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.3.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.4.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.4.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.4.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.4.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.4.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.4.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.4.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.4.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.4.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.4.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.4.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.4.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.4.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q6K,
    ),
    (
        "blk.4.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.4.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.4.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.4.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.4.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.4.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.4.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.4.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.4.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.4.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.4.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.4.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.4.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.4.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.4.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.4.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.4.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.5.attn_gate.weight",
        &[3072, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.5.attn_k_b.weight",
        &[192, 512, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.5.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.5.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.5.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.5.attn_output.weight",
        &[8192, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.5.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.5.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.5.attn_q_b.weight",
        &[1024, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.5.attn_sinks.weight",
        &[32],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.5.attn_v_b.weight",
        &[512, 256, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.5.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.5.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.5.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.5.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.5.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.5.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.5.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.5.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.5.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.5.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.5.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.5.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.5.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.5.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.5.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.5.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.5.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.5.indexer.attn_k.weight",
        &[3072, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.5.indexer.attn_q_b.weight",
        &[1024, 2048],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.5.indexer.k_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.5.indexer.proj.weight",
        &[3072, 16],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.5.indexer_compressor_ape.weight",
        &[128, 4],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.5.indexer_compressor_gate.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.5.indexer_compressor_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.5.indexer_compressor_value.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.5.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.5.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.6.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.6.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.6.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.6.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.6.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.6.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.6.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.6.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.6.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.6.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.6.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.6.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.6.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.6.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.6.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.6.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.6.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.6.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.6.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.6.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.6.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.6.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.6.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.6.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.6.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.6.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.6.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.6.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.6.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.6.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.7.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.7.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.7.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.7.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.7.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.7.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.7.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.7.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.7.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.7.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.7.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.7.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.7.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q6K,
    ),
    (
        "blk.7.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.7.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.7.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.7.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.7.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.7.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.7.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.7.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.7.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.7.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.7.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.7.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.7.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.7.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.7.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.7.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.7.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.8.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.8.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.8.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.8.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.8.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.8.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.8.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.8.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.8.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.8.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.8.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.8.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.8.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.8.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.8.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.8.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.8.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.8.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.8.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.8.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.8.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.8.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.8.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.8.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.8.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.8.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.8.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.8.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.8.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.8.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.9.attn_gate.weight",
        &[3072, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.9.attn_k_b.weight",
        &[192, 512, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.9.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.9.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.9.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.9.attn_output.weight",
        &[8192, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.9.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.9.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.9.attn_q_b.weight",
        &[1024, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.9.attn_sinks.weight",
        &[32],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.9.attn_v_b.weight",
        &[512, 256, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.9.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.9.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.9.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.9.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.9.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.9.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.9.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.9.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.9.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.9.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.9.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.9.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.9.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.9.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.9.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.9.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.9.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.9.indexer.attn_k.weight",
        &[3072, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.9.indexer.attn_q_b.weight",
        &[1024, 2048],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.9.indexer.k_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.9.indexer.proj.weight",
        &[3072, 16],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.9.indexer_compressor_ape.weight",
        &[128, 4],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.9.indexer_compressor_gate.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.9.indexer_compressor_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.9.indexer_compressor_value.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.9.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.9.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.10.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.10.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.10.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.10.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.10.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.10.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.10.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.10.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.10.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.10.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.10.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.10.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.10.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q6K,
    ),
    (
        "blk.10.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.10.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.10.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.10.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.10.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.10.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.10.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.10.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.10.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.10.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.10.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.10.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.10.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.10.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.10.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.10.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.10.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.11.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.11.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.11.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.11.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.11.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.11.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.11.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.11.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.11.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.11.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.11.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.11.engram_gate.weight",
        &[3072, 4],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.11.engram_hc_scale.weight",
        &[3072, 4],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.11.engram_kv.weight",
        &[1536, 15360],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.11.engram_kv_b.weight",
        &[3072, 5],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.11.engram_lookup_norm.weight",
        &[96, 16],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.11.engram_pre_gamma.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.11.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.11.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.11.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.11.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.11.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.11.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.11.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.11.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.11.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.11.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.11.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.11.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.11.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.11.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.11.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.11.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.11.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.11.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.11.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.12.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.12.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.12.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.12.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.12.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.12.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.12.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.12.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.12.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.12.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.12.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.12.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.12.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.12.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.12.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.12.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.12.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.12.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.12.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.12.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.12.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.12.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.12.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.12.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.12.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.12.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.12.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.12.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.12.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.12.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.13.attn_gate.weight",
        &[3072, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.13.attn_k_b.weight",
        &[192, 512, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.13.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.13.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.13.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.13.attn_output.weight",
        &[8192, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.13.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.13.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.13.attn_q_b.weight",
        &[1024, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.13.attn_sinks.weight",
        &[32],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.13.attn_v_b.weight",
        &[512, 256, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.13.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.13.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q6K,
    ),
    (
        "blk.13.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.13.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.13.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.13.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.13.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.13.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.13.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.13.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.13.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.13.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.13.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.13.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.13.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.13.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.13.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.13.indexer.attn_k.weight",
        &[3072, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.13.indexer.attn_q_b.weight",
        &[1024, 2048],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.13.indexer.k_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.13.indexer.proj.weight",
        &[3072, 16],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.13.indexer_compressor_ape.weight",
        &[128, 4],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.13.indexer_compressor_gate.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.13.indexer_compressor_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.13.indexer_compressor_value.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.13.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.13.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.14.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.14.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.14.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.14.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.14.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.14.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.14.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.14.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.14.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.14.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.14.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.14.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.14.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.14.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.14.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.14.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.14.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.14.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.14.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.14.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.14.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.14.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.14.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.14.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.14.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.14.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.14.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.14.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.14.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.14.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.15.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.15.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.15.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.15.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.15.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.15.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.15.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.15.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.15.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.15.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.15.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.15.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.15.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.15.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.15.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.15.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.15.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.15.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.15.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.15.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.15.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.15.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.15.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.15.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.15.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.15.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.15.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.15.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.15.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.15.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.16.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.16.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.16.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.16.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.16.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.16.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.16.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.16.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.16.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.16.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.16.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.16.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.16.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q6K,
    ),
    (
        "blk.16.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.16.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.16.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.16.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.16.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.16.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.16.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.16.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.16.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.16.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.16.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.16.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.16.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.16.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.16.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.16.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.16.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.17.attn_gate.weight",
        &[3072, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.17.attn_k_b.weight",
        &[192, 512, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.17.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.17.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.17.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.17.attn_output.weight",
        &[8192, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.17.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.17.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.17.attn_q_b.weight",
        &[1024, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.17.attn_sinks.weight",
        &[32],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.17.attn_v_b.weight",
        &[512, 256, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.17.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.17.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.17.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.17.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.17.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.17.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.17.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.17.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.17.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.17.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.17.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.17.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.17.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.17.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.17.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.17.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.17.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.17.indexer.attn_k.weight",
        &[3072, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.17.indexer.attn_q_b.weight",
        &[1024, 2048],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.17.indexer.k_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.17.indexer.proj.weight",
        &[3072, 16],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.17.indexer_compressor_ape.weight",
        &[128, 4],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.17.indexer_compressor_gate.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.17.indexer_compressor_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.17.indexer_compressor_value.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.17.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.17.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.18.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.18.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.18.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.18.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.18.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.18.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.18.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.18.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.18.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.18.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.18.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.18.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.18.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.18.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.18.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.18.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.18.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.18.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.18.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.18.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.18.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.18.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.18.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.18.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.18.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.18.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.18.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.18.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.18.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.18.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.19.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.19.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.19.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.19.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.19.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.19.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.19.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.19.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.19.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.19.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.19.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.19.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.19.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q6K,
    ),
    (
        "blk.19.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.19.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.19.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.19.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.19.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.19.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.19.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.19.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.19.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.19.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.19.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.19.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.19.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.19.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.19.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.19.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.19.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.20.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.20.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.20.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.20.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.20.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.20.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.20.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.20.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.20.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.20.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.20.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.20.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.20.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.20.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.20.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.20.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.20.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.20.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.20.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.20.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.20.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.20.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.20.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.20.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.20.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.20.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.20.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.20.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.20.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.20.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.attn_gate.weight",
        &[3072, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.21.attn_k_b.weight",
        &[192, 512, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.21.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.21.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.attn_output.weight",
        &[8192, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.21.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.21.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.attn_q_b.weight",
        &[1024, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.21.attn_sinks.weight",
        &[32],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.attn_v_b.weight",
        &[512, 256, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.21.engram_gate.weight",
        &[3072, 4],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.engram_hc_scale.weight",
        &[3072, 4],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.engram_kv.weight",
        &[1536, 15360],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.21.engram_kv_b.weight",
        &[3072, 5],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.engram_lookup_norm.weight",
        &[96, 16],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.engram_pre_gamma.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.21.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.21.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.21.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.21.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.21.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.21.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.21.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.21.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.indexer.attn_k.weight",
        &[3072, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.21.indexer.attn_q_b.weight",
        &[1024, 2048],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.21.indexer.k_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.indexer.proj.weight",
        &[3072, 16],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.indexer_compressor_ape.weight",
        &[128, 4],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.indexer_compressor_gate.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.21.indexer_compressor_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.indexer_compressor_value.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.21.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.21.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.22.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.22.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.22.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.22.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.22.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.22.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.22.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.22.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.22.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.22.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.22.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.22.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.22.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q6K,
    ),
    (
        "blk.22.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.22.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.22.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.22.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.22.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.22.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.22.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.22.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.22.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.22.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.22.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.22.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.22.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.22.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.22.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.22.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.22.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.23.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.23.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.23.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.23.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.23.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.23.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.23.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.23.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.23.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.23.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.23.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.23.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.23.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.23.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.23.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.23.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.23.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.23.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.23.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.23.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.23.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.23.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.23.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.23.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.23.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.23.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.23.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.23.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.23.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.23.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.24.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.24.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.24.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.24.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.24.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.24.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.24.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.24.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.24.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.24.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.24.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.24.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.24.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.24.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.24.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.24.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.24.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.24.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.24.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.24.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.24.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.24.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.24.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.24.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.24.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.24.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.24.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.24.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.24.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.24.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.25.attn_gate.weight",
        &[3072, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.25.attn_k_b.weight",
        &[192, 512, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.25.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.25.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.25.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.25.attn_output.weight",
        &[8192, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.25.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.25.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.25.attn_q_b.weight",
        &[1024, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.25.attn_sinks.weight",
        &[32],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.25.attn_v_b.weight",
        &[512, 256, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.25.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.25.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q6K,
    ),
    (
        "blk.25.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.25.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.25.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.25.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.25.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.25.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.25.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.25.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.25.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.25.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.25.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.25.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.25.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.25.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.25.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.25.indexer.attn_k.weight",
        &[3072, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.25.indexer.attn_q_b.weight",
        &[1024, 2048],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.25.indexer.k_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.25.indexer.proj.weight",
        &[3072, 16],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.25.indexer_compressor_ape.weight",
        &[128, 4],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.25.indexer_compressor_gate.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.25.indexer_compressor_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.25.indexer_compressor_value.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.25.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.25.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.26.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.26.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.26.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.26.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.26.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.26.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.26.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.26.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.26.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.26.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.26.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.26.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.26.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.26.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.26.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.26.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.26.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.26.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.26.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.26.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.26.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.26.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.26.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.26.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.26.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.26.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.26.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.26.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.26.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.26.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.27.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.27.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.27.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.27.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.27.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.27.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.27.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.27.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.27.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.27.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.27.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.27.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.27.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.27.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.27.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.27.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.27.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.27.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.27.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.27.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.27.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.27.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.27.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.27.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.27.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.27.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.27.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.27.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.27.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.27.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.28.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.28.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.28.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.28.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.28.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.28.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.28.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.28.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.28.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.28.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.28.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.28.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.28.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q6K,
    ),
    (
        "blk.28.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.28.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.28.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.28.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.28.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.28.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.28.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.28.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.28.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.28.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.28.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.28.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.28.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.28.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.28.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.28.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.28.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.29.attn_gate.weight",
        &[3072, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.29.attn_k_b.weight",
        &[192, 512, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.29.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.29.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.29.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.29.attn_output.weight",
        &[8192, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.29.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.29.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.29.attn_q_b.weight",
        &[1024, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.29.attn_sinks.weight",
        &[32],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.29.attn_v_b.weight",
        &[512, 256, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.29.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.29.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.29.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.29.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.29.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.29.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.29.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.29.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.29.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.29.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.29.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.29.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.29.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.29.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.29.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.29.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.29.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.29.indexer.attn_k.weight",
        &[3072, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.29.indexer.attn_q_b.weight",
        &[1024, 2048],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.29.indexer.k_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.29.indexer.proj.weight",
        &[3072, 16],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.29.indexer_compressor_ape.weight",
        &[128, 4],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.29.indexer_compressor_gate.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.29.indexer_compressor_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.29.indexer_compressor_value.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.29.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.29.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.30.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.30.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.30.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.30.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.30.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.30.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.30.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.30.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.30.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.30.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.30.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.30.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.30.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.30.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.30.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.30.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.30.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.30.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.30.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.30.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.30.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.30.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.30.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.30.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.30.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.30.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.30.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.30.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.30.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.30.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.31.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.31.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.31.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.31.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.31.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.31.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.31.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.31.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.31.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.31.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.31.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.31.engram_gate.weight",
        &[3072, 4],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.31.engram_hc_scale.weight",
        &[3072, 4],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.31.engram_kv.weight",
        &[1536, 15360],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.31.engram_kv_b.weight",
        &[3072, 5],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.31.engram_lookup_norm.weight",
        &[96, 16],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.31.engram_pre_gamma.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.31.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.31.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q6K,
    ),
    (
        "blk.31.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.31.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.31.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.31.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.31.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.31.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.31.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.31.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.31.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.31.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.31.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.31.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.31.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.31.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.31.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.31.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.31.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.32.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.32.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.32.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.32.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.32.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.32.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.32.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.32.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.32.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.32.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.32.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.32.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.32.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.32.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.32.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.32.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.32.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.32.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.32.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.32.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.32.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.32.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.32.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.32.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.32.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.32.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.32.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.32.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.32.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.32.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.33.attn_gate.weight",
        &[3072, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.33.attn_k_b.weight",
        &[192, 512, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.33.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.33.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.33.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.33.attn_output.weight",
        &[8192, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.33.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.33.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.33.attn_q_b.weight",
        &[1024, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.33.attn_sinks.weight",
        &[32],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.33.attn_v_b.weight",
        &[512, 256, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.33.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.33.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.33.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.33.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.33.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.33.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.33.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.33.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.33.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.33.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.33.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.33.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.33.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.33.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.33.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.33.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.33.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.33.indexer.attn_k.weight",
        &[3072, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.33.indexer.attn_q_b.weight",
        &[1024, 2048],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.33.indexer.k_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.33.indexer.proj.weight",
        &[3072, 16],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.33.indexer_compressor_ape.weight",
        &[128, 4],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.33.indexer_compressor_gate.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.33.indexer_compressor_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.33.indexer_compressor_value.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.33.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.33.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.34.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.34.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.34.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.34.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.34.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.34.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.34.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.34.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.34.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.34.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.34.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.34.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.34.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q6K,
    ),
    (
        "blk.34.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.34.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.34.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.34.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.34.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.34.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.34.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.34.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.34.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.34.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.34.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.34.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.34.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.34.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.34.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.34.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.34.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.35.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.35.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.35.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.35.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.35.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.35.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.35.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.35.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.35.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.35.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.35.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.35.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.35.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.35.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.35.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.35.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.35.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.35.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.35.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.35.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.35.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.35.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.35.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.35.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.35.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.35.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.35.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.35.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.35.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.35.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.36.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.36.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.36.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.36.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.36.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.36.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.36.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.36.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.36.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.36.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.36.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.36.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.36.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.36.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.36.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.36.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.36.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.36.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.36.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.36.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.36.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.36.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.36.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.36.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.36.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.36.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.36.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.36.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.36.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.36.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.37.attn_gate.weight",
        &[3072, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.37.attn_k_b.weight",
        &[192, 512, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.37.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.37.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.37.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.37.attn_output.weight",
        &[8192, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.37.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.37.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.37.attn_q_b.weight",
        &[1024, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.37.attn_sinks.weight",
        &[32],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.37.attn_v_b.weight",
        &[512, 256, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.37.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.37.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q6K,
    ),
    (
        "blk.37.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.37.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.37.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.37.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.37.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.37.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.37.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.37.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.37.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.37.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.37.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.37.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.37.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.37.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.37.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.37.indexer.attn_k.weight",
        &[3072, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.37.indexer.attn_q_b.weight",
        &[1024, 2048],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.37.indexer.k_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.37.indexer.proj.weight",
        &[3072, 16],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.37.indexer_compressor_ape.weight",
        &[128, 4],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.37.indexer_compressor_gate.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.37.indexer_compressor_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.37.indexer_compressor_value.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.37.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.37.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.38.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.38.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.38.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.38.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.38.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.38.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.38.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.38.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.38.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.38.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.38.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.38.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.38.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q6K,
    ),
    (
        "blk.38.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.38.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.38.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.38.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.38.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.38.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.38.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.38.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.38.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.38.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.38.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.38.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.38.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.38.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.38.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.38.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.38.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.39.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.39.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.39.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.39.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.39.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.39.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.39.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.39.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.39.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.39.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.39.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.39.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.39.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q6K,
    ),
    (
        "blk.39.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.39.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.39.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.39.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.39.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.39.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.39.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.39.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.39.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.39.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.39.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.39.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.39.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.39.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.39.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.39.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.39.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.40.attn_gate.weight",
        &[3072, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.40.attn_k_b.weight",
        &[192, 512, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.40.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.40.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.40.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.40.attn_output.weight",
        &[16384, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.40.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.40.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.40.attn_q_b.weight",
        &[1024, 16384],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.40.attn_sinks.weight",
        &[64],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.40.attn_v_b.weight",
        &[512, 256, 64],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.40.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.40.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q6K,
    ),
    (
        "blk.40.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.40.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.40.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.40.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.40.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.40.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.40.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.40.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.40.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.40.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.40.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.40.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.40.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.40.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.40.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.40.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.40.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.41.attn_gate.weight",
        &[3072, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.41.attn_k_b.weight",
        &[192, 512, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.41.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.41.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.41.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.41.attn_output.weight",
        &[8192, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.41.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.41.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.41.attn_q_b.weight",
        &[1024, 8192],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.41.attn_sinks.weight",
        &[32],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.41.attn_v_b.weight",
        &[512, 256, 32],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.41.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.41.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q6K,
    ),
    (
        "blk.41.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.41.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.41.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.41.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.41.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.41.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.41.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.41.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Q4K,
    ),
    (
        "blk.41.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.41.hc_attn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.41.hc_attn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.41.hc_attn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.41.hc_ffn_base.weight",
        &[8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.41.hc_ffn_fn.weight",
        &[12288, 8],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.41.hc_ffn_scale.weight",
        &[2],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.41.indexer.attn_k.weight",
        &[3072, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.41.indexer.attn_q_b.weight",
        &[1024, 2048],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.41.indexer.k_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.41.indexer.proj.weight",
        &[3072, 16],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.41.indexer_compressor_ape.weight",
        &[128, 4],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.41.indexer_compressor_gate.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.41.indexer_compressor_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.41.indexer_compressor_value.weight",
        &[128, 128],
        crate::quant::GgmlType::Q8_0,
    ),
    (
        "blk.41.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.41.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.42.attn_gate.weight",
        &[3072, 8192],
        crate::quant::GgmlType::Bf16,
    ),
    (
        "blk.42.attn_k_b.weight",
        &[192, 512, 32],
        crate::quant::GgmlType::Bf16,
    ),
    (
        "blk.42.attn_kv_a_mqa.weight",
        &[3072, 576],
        crate::quant::GgmlType::Bf16,
    ),
    (
        "blk.42.attn_kv_a_norm.weight",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.42.attn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.42.attn_output.weight",
        &[8192, 3072],
        crate::quant::GgmlType::Bf16,
    ),
    (
        "blk.42.attn_q_a.weight",
        &[3072, 1024],
        crate::quant::GgmlType::Bf16,
    ),
    (
        "blk.42.attn_q_a_norm.weight",
        &[1024],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.42.attn_q_b.weight",
        &[1024, 8192],
        crate::quant::GgmlType::Bf16,
    ),
    (
        "blk.42.attn_sinks.weight",
        &[32],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.42.attn_v_b.weight",
        &[512, 256, 32],
        crate::quant::GgmlType::Bf16,
    ),
    (
        "blk.42.exp_probs_b.bias",
        &[512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.42.ffn_down_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Bf16,
    ),
    (
        "blk.42.ffn_down_shexp.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Bf16,
    ),
    (
        "blk.42.ffn_gate_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Bf16,
    ),
    (
        "blk.42.ffn_gate_inp.weight",
        &[3072, 512],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.42.ffn_gate_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Bf16,
    ),
    (
        "blk.42.ffn_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.42.ffn_routed_down.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Bf16,
    ),
    (
        "blk.42.ffn_routed_up.weight",
        &[1536, 3072],
        crate::quant::GgmlType::Bf16,
    ),
    (
        "blk.42.ffn_up_exps.weight",
        &[1536, 1536, 512],
        crate::quant::GgmlType::Bf16,
    ),
    (
        "blk.42.ffn_up_shexp.weight",
        &[3072, 1536],
        crate::quant::GgmlType::Bf16,
    ),
    (
        "blk.42.indexer.attn_k.weight",
        &[3072, 128],
        crate::quant::GgmlType::Bf16,
    ),
    (
        "blk.42.indexer.attn_q_b.weight",
        &[1024, 2048],
        crate::quant::GgmlType::Bf16,
    ),
    (
        "blk.42.indexer.k_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.42.indexer.proj.weight",
        &[3072, 16],
        crate::quant::GgmlType::Bf16,
    ),
    (
        "blk.42.indexer_compressor_ape.weight",
        &[128, 4],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.42.indexer_compressor_gate.weight",
        &[128, 128],
        crate::quant::GgmlType::Bf16,
    ),
    (
        "blk.42.indexer_compressor_norm.weight",
        &[128],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.42.indexer_compressor_value.weight",
        &[128, 128],
        crate::quant::GgmlType::Bf16,
    ),
    (
        "blk.42.nextn.eh_proj.weight",
        &[6144, 3072],
        crate::quant::GgmlType::Bf16,
    ),
    (
        "blk.42.nextn.enorm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.42.nextn.hnorm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.42.nextn.shared_head_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.42.post_attention_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
    (
        "blk.42.post_ffw_norm.weight",
        &[3072],
        crate::quant::GgmlType::F32,
    ),
];
