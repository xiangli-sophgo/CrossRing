"""
延迟计算模块

提供各类操作的延迟计算公式，包括：
- PCIe 数据传输
- HBM 内存访问
- Transformer 子操作（Attention、FFN、LayerNorm）
- 集合通信（AllReduce、P2P）
"""

import math
from .types import (
    LLMModelConfig, InferenceConfig, ParallelismStrategy,
    HardwareConfig, MLAConfig, get_bytes_per_element,
)


# ============================================
# 常量定义
# ============================================

# PCIe 协议开销
PCIE_PROTOCOL_OVERHEAD = 0.15  # 15% 协议开销

# HBM 效率因子
HBM_EFFICIENCY = 0.85  # 85% 效率

# AllReduce 算法因子 (Ring AllReduce)
ALLREDUCE_FACTOR = 2.0  # 2(n-1)/n ≈ 2

# Softmax 操作的 FLOPS 因子
SOFTMAX_FLOPS_FACTOR = 5  # exp + sum + div + sub + max


# ============================================
# PCIe 传输延迟
# ============================================

def calc_pcie_h2d_latency(
    data_gb: float,
    hardware: HardwareConfig,
) -> float:
    """
    计算 PCIe Host to Device 传输延迟

    Args:
        data_gb: 数据量 (GB)
        hardware: 硬件配置

    Returns:
        延迟 (ms)
    """
    if data_gb <= 0:
        return 0.0

    effective_bandwidth = hardware.chip.pcie_bandwidth_gbps * (1 - PCIE_PROTOCOL_OVERHEAD)
    transfer_time_ms = (data_gb / effective_bandwidth) * 1000
    startup_latency_ms = hardware.chip.pcie_latency_us / 1000

    return transfer_time_ms + startup_latency_ms


def calc_pcie_d2h_latency(
    data_gb: float,
    hardware: HardwareConfig,
) -> float:
    """
    计算 PCIe Device to Host 传输延迟
    与 H2D 相同，但实际中可能略有差异
    """
    return calc_pcie_h2d_latency(data_gb, hardware)


# ============================================
# HBM 内存访问延迟
# ============================================

def calc_hbm_read_latency(
    data_gb: float,
    hardware: HardwareConfig,
    is_sequential: bool = True,
) -> float:
    """
    计算 HBM 读取延迟

    Args:
        data_gb: 数据量 (GB)
        hardware: 硬件配置
        is_sequential: 是否顺序访问

    Returns:
        延迟 (ms)
    """
    if data_gb <= 0:
        return 0.0

    bandwidth_gbps = hardware.chip.memory_bandwidth_gbps * HBM_EFFICIENCY
    bandwidth_time_ms = (data_gb / bandwidth_gbps) * 1000

    if not is_sequential:
        # 随机访问增加延迟
        latency_overhead_ms = hardware.chip.hbm_random_access_latency_ns / 1e6
        return max(bandwidth_time_ms, latency_overhead_ms) + latency_overhead_ms

    return bandwidth_time_ms


def calc_hbm_write_latency(
    data_gb: float,
    hardware: HardwareConfig,
) -> float:
    """计算 HBM 写入延迟"""
    return calc_hbm_read_latency(data_gb, hardware, is_sequential=True)


def calc_weight_load_latency(
    weight_gb: float,
    hardware: HardwareConfig,
) -> float:
    """
    计算权重加载延迟

    权重加载是顺序读取，效率较高
    """
    return calc_hbm_read_latency(weight_gb, hardware, is_sequential=True)


# ============================================
# KV Cache 访问延迟
# ============================================

def calc_kv_cache_read_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    context_length: int,
) -> float:
    """
    计算 KV Cache 读取延迟

    KV Cache 大小 = 2 * batch * context * num_kv_heads * head_dim * bytes_per_element / TP
    """
    head_dim = model.hidden_size // model.num_attention_heads
    bytes_per_elem = get_bytes_per_element(model.dtype)

    kv_cache_size_bytes = (
        2 * inference.batch_size * context_length *
        model.num_kv_heads * head_dim * bytes_per_elem
    ) / parallelism.tp

    kv_cache_gb = kv_cache_size_bytes / (1024 ** 3)

    # KV Cache 读取模式接近随机访问
    return calc_hbm_read_latency(kv_cache_gb, hardware, is_sequential=False)


def calc_kv_cache_write_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int = 1,
) -> float:
    """
    计算 KV Cache 写入延迟

    每次写入 num_tokens 个 token 的 KV
    """
    head_dim = model.hidden_size // model.num_attention_heads
    bytes_per_elem = get_bytes_per_element(model.dtype)

    kv_size_bytes = (
        2 * inference.batch_size * num_tokens *
        model.num_kv_heads * head_dim * bytes_per_elem
    ) / parallelism.tp

    kv_gb = kv_size_bytes / (1024 ** 3)
    return calc_hbm_write_latency(kv_gb, hardware)


# ============================================
# Transformer 子操作延迟
# ============================================

def calc_embedding_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算 Embedding 层延迟

    主要是从 Embedding 表查找，受内存带宽限制
    """
    bytes_per_elem = get_bytes_per_element(model.dtype)
    data_size_bytes = inference.batch_size * num_tokens * model.hidden_size * bytes_per_elem
    data_gb = data_size_bytes / (1024 ** 3)

    return calc_hbm_read_latency(data_gb, hardware)


def calc_layernorm_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算 LayerNorm/RMSNorm 延迟

    LayerNorm FLOPS ≈ 5 * B * S * H (mean, var, normalize, scale, shift)
    RMSNorm FLOPS ≈ 3 * B * S * H (square, mean, normalize)
    """
    if model.norm_type == "rmsnorm":
        flops_per_token = 3 * model.hidden_size
    else:
        flops_per_token = 5 * model.hidden_size

    total_flops = inference.batch_size * num_tokens * flops_per_token
    tflops = total_flops / 1e12

    # LayerNorm 是逐元素操作，受内存带宽限制
    bytes_per_elem = get_bytes_per_element(model.dtype)
    data_size_bytes = inference.batch_size * num_tokens * model.hidden_size * bytes_per_elem * 2  # 读+写
    data_gb = data_size_bytes / (1024 ** 3)

    memory_time = calc_hbm_read_latency(data_gb, hardware)
    compute_time = tflops / hardware.chip.compute_tflops_fp16 * 1000  # ms

    return max(memory_time, compute_time)


def calc_attention_qkv_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算 Attention QKV 投影延迟

    FLOPS = 2 * B * S * H * (H + 2 * kv_heads * head_dim) / TP
    """
    head_dim = model.hidden_size // model.num_attention_heads
    qkv_size = model.hidden_size + 2 * model.num_kv_heads * head_dim

    flops = 2 * inference.batch_size * num_tokens * model.hidden_size * qkv_size / parallelism.tp
    tflops = flops / 1e12

    compute_time = tflops / hardware.chip.compute_tflops_fp16 * 1000  # ms

    # 读权重 + 读输入 + 写输出
    bytes_per_elem = get_bytes_per_element(model.dtype)
    weight_bytes = model.hidden_size * qkv_size * bytes_per_elem / parallelism.tp
    io_bytes = inference.batch_size * num_tokens * (model.hidden_size + qkv_size) * bytes_per_elem / parallelism.tp
    data_gb = (weight_bytes + io_bytes) / (1024 ** 3)

    memory_time = calc_hbm_read_latency(data_gb, hardware)

    return max(memory_time, compute_time)


def calc_attention_score_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    context_length: int,
) -> float:
    """
    计算 Attention Score (Q @ K^T) 延迟

    FLOPS = 2 * B * heads * S * C * head_dim / TP

    内存访问:
    - 读取 Q: [B, heads, S, head_dim]
    - 读取 K: [B, heads, C, head_dim]
    - 写入 Score: [B, heads, S, C]
    """
    head_dim = model.hidden_size // model.num_attention_heads
    num_heads_per_tp = model.num_attention_heads // parallelism.tp
    bytes_per_elem = get_bytes_per_element(model.dtype)

    # 计算时间
    flops = 2 * inference.batch_size * num_heads_per_tp * num_tokens * context_length * head_dim
    tflops = flops / 1e12
    compute_time = tflops / hardware.chip.compute_tflops_fp16 * 1000

    # 内存访问时间
    q_bytes = inference.batch_size * num_heads_per_tp * num_tokens * head_dim * bytes_per_elem
    k_bytes = inference.batch_size * num_heads_per_tp * context_length * head_dim * bytes_per_elem
    score_bytes = inference.batch_size * num_heads_per_tp * num_tokens * context_length * bytes_per_elem
    data_gb = (q_bytes + k_bytes + score_bytes) / (1024 ** 3)
    memory_time = calc_hbm_read_latency(data_gb, hardware)

    return max(compute_time, memory_time)


def calc_attention_softmax_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    context_length: int,
) -> float:
    """
    计算 Attention Softmax 延迟

    FLOPS = 5 * B * heads * S * C (exp + sum + div + sub + max)
    """
    num_heads_per_tp = model.num_attention_heads // parallelism.tp

    flops = SOFTMAX_FLOPS_FACTOR * inference.batch_size * num_heads_per_tp * num_tokens * context_length
    tflops = flops / 1e12

    compute_time = tflops / hardware.chip.compute_tflops_fp16 * 1000

    # Softmax 受内存带宽限制
    bytes_per_elem = get_bytes_per_element(model.dtype)
    score_bytes = inference.batch_size * num_heads_per_tp * num_tokens * context_length * bytes_per_elem * 2
    data_gb = score_bytes / (1024 ** 3)

    memory_time = calc_hbm_read_latency(data_gb, hardware)

    return max(memory_time, compute_time)


def calc_attention_output_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    context_length: int,
) -> float:
    """
    计算 Attention Output (Softmax @ V + Output Projection) 延迟

    FLOPS = 2 * B * heads * S * C * head_dim + 2 * B * S * H * H / TP

    内存访问:
    - 读取 Softmax scores: [B, heads, S, C]
    - 读取 V: [B, heads, C, head_dim]
    - 读取 Output 权重: [H, H] / TP
    - 写入输出: [B, S, H]
    """
    head_dim = model.hidden_size // model.num_attention_heads
    num_heads_per_tp = model.num_attention_heads // parallelism.tp
    bytes_per_elem = get_bytes_per_element(model.dtype)

    # 计算时间
    # Softmax @ V
    flops_sv = 2 * inference.batch_size * num_heads_per_tp * num_tokens * context_length * head_dim
    # Output projection
    flops_out = 2 * inference.batch_size * num_tokens * model.hidden_size * model.hidden_size / parallelism.tp
    total_flops = flops_sv + flops_out
    tflops = total_flops / 1e12
    compute_time = tflops / hardware.chip.compute_tflops_fp16 * 1000

    # 内存访问时间
    # Softmax @ V 部分
    score_bytes = inference.batch_size * num_heads_per_tp * num_tokens * context_length * bytes_per_elem
    v_bytes = inference.batch_size * num_heads_per_tp * context_length * head_dim * bytes_per_elem
    # Output projection 部分
    output_weight_bytes = model.hidden_size * model.hidden_size * bytes_per_elem / parallelism.tp
    output_bytes = inference.batch_size * num_tokens * model.hidden_size * bytes_per_elem
    data_gb = (score_bytes + v_bytes + output_weight_bytes + output_bytes) / (1024 ** 3)
    memory_time = calc_hbm_read_latency(data_gb, hardware)

    return max(compute_time, memory_time)


# ============================================
# MLA (Multi-head Latent Attention) 延迟计算 - DeepSeek V3/R1 专用
# ============================================

def calc_mla_q_projection_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算 MLA Q 投影延迟 (使用 LoRA 压缩)

    DeepSeek V3 Q 投影分两步:
    1. hidden_size -> q_lora_rank (压缩)
    2. q_lora_rank -> num_heads * (qk_nope_head_dim + qk_rope_head_dim) (解压)

    FLOPS = 2 * B * S * H * q_lora_rank + 2 * B * S * q_lora_rank * head_total_dim
    """
    if model.mla_config is None:
        # 回退到标准 QKV 计算
        return calc_attention_qkv_latency(model, inference, parallelism, hardware, num_tokens)

    mla = model.mla_config
    bytes_per_elem = get_bytes_per_element(model.dtype)

    # Q 的总头维度 = nope + rope
    q_head_dim = mla.qk_nope_head_dim + mla.qk_rope_head_dim
    q_total_dim = model.num_attention_heads * q_head_dim // parallelism.tp

    # FLOPS: down projection + up projection
    flops_down = 2 * inference.batch_size * num_tokens * model.hidden_size * mla.q_lora_rank
    flops_up = 2 * inference.batch_size * num_tokens * mla.q_lora_rank * q_total_dim
    total_flops = (flops_down + flops_up) / parallelism.tp
    tflops = total_flops / 1e12
    compute_time = tflops / hardware.chip.compute_tflops_fp16 * 1000

    # 内存访问: 权重 + IO
    weight_down_bytes = model.hidden_size * mla.q_lora_rank * bytes_per_elem / parallelism.tp
    weight_up_bytes = mla.q_lora_rank * q_total_dim * bytes_per_elem
    io_bytes = inference.batch_size * num_tokens * (model.hidden_size + q_total_dim) * bytes_per_elem
    data_gb = (weight_down_bytes + weight_up_bytes + io_bytes) / (1024 ** 3)
    memory_time = calc_hbm_read_latency(data_gb, hardware)

    return max(compute_time, memory_time)


def calc_mla_kv_compression_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算 MLA KV 压缩投影延迟

    DeepSeek V3 KV 压缩: hidden_size -> kv_lora_rank
    这是 MLA 的关键优化，将 KV 压缩到低维空间

    FLOPS = 2 * B * S * H * kv_lora_rank
    """
    if model.mla_config is None:
        return 0.0

    mla = model.mla_config
    bytes_per_elem = get_bytes_per_element(model.dtype)

    # 压缩投影
    flops = 2 * inference.batch_size * num_tokens * model.hidden_size * mla.kv_lora_rank / parallelism.tp
    tflops = flops / 1e12
    compute_time = tflops / hardware.chip.compute_tflops_fp16 * 1000

    # 内存访问
    weight_bytes = model.hidden_size * mla.kv_lora_rank * bytes_per_elem / parallelism.tp
    io_bytes = inference.batch_size * num_tokens * (model.hidden_size + mla.kv_lora_rank) * bytes_per_elem
    data_gb = (weight_bytes + io_bytes) / (1024 ** 3)
    memory_time = calc_hbm_read_latency(data_gb, hardware)

    return max(compute_time, memory_time)


def calc_mla_attention_score_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    context_length: int,
) -> float:
    """
    计算 MLA Attention Score 延迟

    MLA 在压缩空间计算 Score:
    FLOPS = 2 * B * heads * S * C * (qk_nope_dim + qk_rope_dim)

    关键优化: kv_lora_rank (512) << num_kv_heads * head_dim (传统)
    """
    if model.mla_config is None:
        return calc_attention_score_latency(model, inference, parallelism, hardware, num_tokens, context_length)

    mla = model.mla_config
    bytes_per_elem = get_bytes_per_element(model.dtype)
    num_heads_per_tp = model.num_attention_heads // parallelism.tp

    # MLA Score 计算: Q @ K^T，但 K 是压缩后的
    qk_dim = mla.qk_nope_head_dim + mla.qk_rope_head_dim
    flops = 2 * inference.batch_size * num_heads_per_tp * num_tokens * context_length * qk_dim
    tflops = flops / 1e12
    compute_time = tflops / hardware.chip.compute_tflops_fp16 * 1000

    # 内存访问: Q + 压缩的 K + Score
    q_bytes = inference.batch_size * num_heads_per_tp * num_tokens * qk_dim * bytes_per_elem
    # K 是从压缩的 kv_lora_rank 解压来的
    k_bytes = inference.batch_size * context_length * mla.kv_lora_rank * bytes_per_elem / parallelism.tp
    score_bytes = inference.batch_size * num_heads_per_tp * num_tokens * context_length * bytes_per_elem
    data_gb = (q_bytes + k_bytes + score_bytes) / (1024 ** 3)
    memory_time = calc_hbm_read_latency(data_gb, hardware)

    return max(compute_time, memory_time)


def calc_mla_kv_cache_read_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    context_length: int,
) -> float:
    """
    计算 MLA KV Cache 读取延迟

    MLA 的关键优势: KV Cache 只存储压缩后的 kv_lora_rank 维度
    大小 = 2 * batch * context * kv_lora_rank * bytes / TP

    对比传统 GQA: 2 * batch * context * num_kv_heads * head_dim * bytes / TP
    压缩比 ≈ (num_kv_heads * head_dim) / kv_lora_rank ≈ 32x
    """
    if model.mla_config is None:
        return calc_kv_cache_read_latency(model, inference, parallelism, hardware, context_length)

    mla = model.mla_config
    bytes_per_elem = get_bytes_per_element(model.dtype)

    # MLA KV Cache 大小 (压缩后)
    kv_cache_bytes = (
        2 * inference.batch_size * context_length *
        mla.kv_lora_rank * bytes_per_elem
    ) / parallelism.tp

    kv_cache_gb = kv_cache_bytes / (1024 ** 3)
    return calc_hbm_read_latency(kv_cache_gb, hardware, is_sequential=False)


def calc_mla_kv_cache_write_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算 MLA KV Cache 写入延迟
    """
    if model.mla_config is None:
        return calc_kv_cache_write_latency(model, inference, parallelism, hardware, num_tokens)

    mla = model.mla_config
    bytes_per_elem = get_bytes_per_element(model.dtype)

    # MLA KV Cache 写入 (压缩后)
    kv_bytes = (
        2 * inference.batch_size * num_tokens *
        mla.kv_lora_rank * bytes_per_elem
    ) / parallelism.tp

    kv_gb = kv_bytes / (1024 ** 3)
    return calc_hbm_write_latency(kv_gb, hardware)


def calc_mla_output_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    context_length: int,
) -> float:
    """
    计算 MLA Output 延迟 (Softmax @ V + Output Projection)

    V 需要从压缩空间解压:
    1. kv_lora_rank -> num_kv_heads * v_head_dim (解压)
    2. Softmax @ V
    3. Output projection: num_heads * v_head_dim -> hidden_size
    """
    if model.mla_config is None:
        return calc_attention_output_latency(model, inference, parallelism, hardware, num_tokens, context_length)

    mla = model.mla_config
    bytes_per_elem = get_bytes_per_element(model.dtype)
    num_heads_per_tp = model.num_attention_heads // parallelism.tp

    # V 解压: kv_lora_rank -> v_total_dim
    v_total_dim = model.num_kv_heads * mla.v_head_dim // parallelism.tp
    flops_v_up = 2 * inference.batch_size * context_length * mla.kv_lora_rank * v_total_dim

    # Softmax @ V
    flops_sv = 2 * inference.batch_size * num_heads_per_tp * num_tokens * context_length * mla.v_head_dim

    # Output projection
    flops_out = 2 * inference.batch_size * num_tokens * model.num_attention_heads * mla.v_head_dim * model.hidden_size / parallelism.tp

    total_flops = flops_v_up + flops_sv + flops_out
    tflops = total_flops / 1e12
    compute_time = tflops / hardware.chip.compute_tflops_fp16 * 1000

    # 内存访问
    v_up_weight_bytes = mla.kv_lora_rank * v_total_dim * bytes_per_elem
    score_bytes = inference.batch_size * num_heads_per_tp * num_tokens * context_length * bytes_per_elem
    output_weight_bytes = model.num_attention_heads * mla.v_head_dim * model.hidden_size * bytes_per_elem / parallelism.tp
    output_bytes = inference.batch_size * num_tokens * model.hidden_size * bytes_per_elem
    data_gb = (v_up_weight_bytes + score_bytes + output_weight_bytes + output_bytes) / (1024 ** 3)
    memory_time = calc_hbm_read_latency(data_gb, hardware)

    return max(compute_time, memory_time)


# ============================================
# MLA 细粒度操作延迟计算 - DeepSeek V3 数据流对标
# ============================================

def calc_rmsnorm_q_lora_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算 RMSNorm_q_lora 延迟 (Q LoRA 投影前的 RMSNorm)

    作用于 hidden_size 维度，为 Q 投影做归一化
    FLOPS = 3 * B * S * H (square + mean + normalize)
    """
    flops = 3 * inference.batch_size * num_tokens * model.hidden_size
    tflops = flops / 1e12
    compute_time = tflops / hardware.chip.compute_tflops_fp16 * 1000

    bytes_per_elem = get_bytes_per_element(model.dtype)
    data_bytes = inference.batch_size * num_tokens * model.hidden_size * bytes_per_elem * 2
    data_gb = data_bytes / (1024 ** 3)
    memory_time = calc_hbm_read_latency(data_gb, hardware)

    return max(memory_time, compute_time)


def calc_rmsnorm_kv_lora_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算 RMSNorm_kv_lora 延迟 (KV LoRA 投影前的 RMSNorm)

    作用于 hidden_size 维度，为 KV 压缩做归一化
    """
    return calc_rmsnorm_q_lora_latency(model, inference, parallelism, hardware, num_tokens)


def calc_mm_q_lora_a_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算 mm_q_lora_a 延迟 (Q LoRA 下投影)

    矩阵乘法: hidden_size -> q_lora_rank (压缩)
    FLOPS = 2 * B * S * H * q_lora_rank / TP
    """
    if model.mla_config is None:
        return 0.0

    mla = model.mla_config
    bytes_per_elem = get_bytes_per_element(model.dtype)

    flops = 2 * inference.batch_size * num_tokens * model.hidden_size * mla.q_lora_rank / parallelism.tp
    tflops = flops / 1e12
    compute_time = tflops / hardware.chip.compute_tflops_fp16 * 1000

    weight_bytes = model.hidden_size * mla.q_lora_rank * bytes_per_elem / parallelism.tp
    io_bytes = inference.batch_size * num_tokens * (model.hidden_size + mla.q_lora_rank) * bytes_per_elem
    data_gb = (weight_bytes + io_bytes) / (1024 ** 3)
    memory_time = calc_hbm_read_latency(data_gb, hardware)

    return max(compute_time, memory_time)


def calc_mm_q_lora_b_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算 mm_q_lora_b 延迟 (Q LoRA 上投影)

    矩阵乘法: q_lora_rank -> num_heads * (qk_nope_dim + qk_rope_dim) (解压)
    FLOPS = 2 * B * S * q_lora_rank * Q_total_dim / TP
    """
    if model.mla_config is None:
        return 0.0

    mla = model.mla_config
    bytes_per_elem = get_bytes_per_element(model.dtype)

    q_head_dim = mla.qk_nope_head_dim + mla.qk_rope_head_dim
    q_total_dim = model.num_attention_heads * q_head_dim // parallelism.tp

    flops = 2 * inference.batch_size * num_tokens * mla.q_lora_rank * q_total_dim
    tflops = flops / 1e12
    compute_time = tflops / hardware.chip.compute_tflops_fp16 * 1000

    weight_bytes = mla.q_lora_rank * q_total_dim * bytes_per_elem
    io_bytes = inference.batch_size * num_tokens * (mla.q_lora_rank + q_total_dim) * bytes_per_elem
    data_gb = (weight_bytes + io_bytes) / (1024 ** 3)
    memory_time = calc_hbm_read_latency(data_gb, hardware)

    return max(compute_time, memory_time)


def calc_mm_kv_lora_a_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算 mm_kv_lora_a 延迟 (KV LoRA 压缩)

    矩阵乘法: hidden_size -> kv_lora_rank (压缩)
    这是 MLA 的关键优化点，压缩后的 KV 直接缓存
    FLOPS = 2 * B * S * H * kv_lora_rank / TP
    """
    return calc_mla_kv_compression_latency(model, inference, parallelism, hardware, num_tokens)


def calc_bmm_qk_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    context_length: int,
) -> float:
    """
    计算 bmm Q@K^T 延迟 (批量矩阵乘)

    FLOPS = 2 * B * heads * S * C * head_dim / TP
    """
    if model.mla_config is not None:
        return calc_mla_attention_score_latency(
            model, inference, parallelism, hardware, num_tokens, context_length
        )
    return calc_attention_score_latency(
        model, inference, parallelism, hardware, num_tokens, context_length
    )


def calc_bmm_sv_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    context_length: int,
) -> float:
    """
    计算 bmm Score@V 延迟 (批量矩阵乘)

    FLOPS = 2 * B * heads * S * C * v_head_dim / TP
    """
    bytes_per_elem = get_bytes_per_element(model.dtype)
    num_heads_per_tp = model.num_attention_heads // parallelism.tp

    if model.mla_config is not None:
        v_head_dim = model.mla_config.v_head_dim
    else:
        v_head_dim = model.hidden_size // model.num_attention_heads

    flops = 2 * inference.batch_size * num_heads_per_tp * num_tokens * context_length * v_head_dim
    tflops = flops / 1e12
    compute_time = tflops / hardware.chip.compute_tflops_fp16 * 1000

    score_bytes = inference.batch_size * num_heads_per_tp * num_tokens * context_length * bytes_per_elem
    v_bytes = inference.batch_size * num_heads_per_tp * context_length * v_head_dim * bytes_per_elem
    output_bytes = inference.batch_size * num_heads_per_tp * num_tokens * v_head_dim * bytes_per_elem
    data_gb = (score_bytes + v_bytes + output_bytes) / (1024 ** 3)
    memory_time = calc_hbm_read_latency(data_gb, hardware)

    return max(compute_time, memory_time)


def calc_attn_fc_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算 attn_fc 延迟 (注意力输出投影 wo)

    矩阵乘法: num_heads * v_head_dim -> hidden_size
    FLOPS = 2 * B * S * (heads * v_dim) * H / TP
    """
    bytes_per_elem = get_bytes_per_element(model.dtype)

    if model.mla_config is not None:
        input_dim = model.num_attention_heads * model.mla_config.v_head_dim
    else:
        input_dim = model.hidden_size

    flops = 2 * inference.batch_size * num_tokens * input_dim * model.hidden_size / parallelism.tp
    tflops = flops / 1e12
    compute_time = tflops / hardware.chip.compute_tflops_fp16 * 1000

    weight_bytes = input_dim * model.hidden_size * bytes_per_elem / parallelism.tp
    io_bytes = inference.batch_size * num_tokens * (input_dim + model.hidden_size) * bytes_per_elem
    data_gb = (weight_bytes + io_bytes) / (1024 ** 3)
    memory_time = calc_hbm_read_latency(data_gb, hardware)

    return max(compute_time, memory_time)


# ============================================
# MoE (Mixture of Experts) 延迟计算
# ============================================

def calc_moe_gate_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算 MoE Gate (路由网络) 延迟

    矩阵乘法: hidden_size -> num_experts + Sigmoid
    FLOPS = 2 * B * S * H * num_experts + B * S * num_experts (sigmoid)
    """
    if model.moe_config is None:
        return 0.0

    moe = model.moe_config
    bytes_per_elem = get_bytes_per_element(model.dtype)

    # 路由计算: hidden -> num_experts
    flops_mm = 2 * inference.batch_size * num_tokens * model.hidden_size * moe.num_experts
    # Sigmoid 激活
    flops_sigmoid = inference.batch_size * num_tokens * moe.num_experts
    total_flops = flops_mm + flops_sigmoid
    tflops = total_flops / 1e12
    compute_time = tflops / hardware.chip.compute_tflops_fp16 * 1000

    # 内存访问
    weight_bytes = model.hidden_size * moe.num_experts * bytes_per_elem
    io_bytes = inference.batch_size * num_tokens * (model.hidden_size + moe.num_experts) * bytes_per_elem
    data_gb = (weight_bytes + io_bytes) / (1024 ** 3)
    memory_time = calc_hbm_read_latency(data_gb, hardware)

    return max(compute_time, memory_time)


def calc_moe_expert_ffn_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算 MoE 路由专家 FFN 延迟 (单个专家)

    每个专家处理 num_tokens * num_experts_per_tok / num_experts 个 token
    FLOPS = 2 * tokens_per_expert * H * expert_intermediate * 3 (gate+up+down)
    """
    if model.moe_config is None:
        return calc_ffn_gate_latency(model, inference, parallelism, hardware, num_tokens)

    moe = model.moe_config
    bytes_per_elem = get_bytes_per_element(model.dtype)

    # 每个专家处理的 token 数 (考虑 EP 并行)
    total_tokens = inference.batch_size * num_tokens
    tokens_per_expert = total_tokens * moe.num_experts_per_tok / moe.num_experts
    tokens_per_expert_per_ep = tokens_per_expert / parallelism.ep

    # 专家 FFN 中间维度 (必须配置，不允许 fallback 到 model.intermediate_size)
    if moe.expert_intermediate_size <= 0:
        raise ValueError(f"MoE 配置必须指定 expert_intermediate_size，当前值: {moe.expert_intermediate_size}")
    expert_intermediate = moe.expert_intermediate_size

    # Gate + Up + Down 三个矩阵乘法
    flops_gate = 2 * tokens_per_expert_per_ep * model.hidden_size * expert_intermediate
    flops_up = 2 * tokens_per_expert_per_ep * model.hidden_size * expert_intermediate
    flops_down = 2 * tokens_per_expert_per_ep * expert_intermediate * model.hidden_size
    total_flops = flops_gate + flops_up + flops_down
    tflops = total_flops / 1e12
    compute_time = tflops / hardware.chip.compute_tflops_fp16 * 1000

    # 内存访问 (权重 + IO)
    weight_bytes = 3 * model.hidden_size * expert_intermediate * bytes_per_elem
    io_bytes = tokens_per_expert_per_ep * (model.hidden_size * 2 + expert_intermediate) * bytes_per_elem
    data_gb = (weight_bytes + io_bytes) / (1024 ** 3)
    memory_time = calc_hbm_read_latency(data_gb, hardware)

    return max(compute_time, memory_time)


def calc_moe_shared_expert_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算 MoE 共享专家 FFN 延迟

    共享专家处理所有 token，使用 MoE 专家相同的中间维度
    DeepSeek V3: 1 个共享专家，intermediate_size = 2048
    """
    if model.moe_config is None or model.moe_config.num_shared_experts == 0:
        return 0.0

    moe = model.moe_config
    bytes_per_elem = get_bytes_per_element(model.dtype)

    if moe.expert_intermediate_size <= 0:
        raise ValueError(f"MoE 配置必须指定 expert_intermediate_size")

    expert_intermediate = moe.expert_intermediate_size
    total_tokens = inference.batch_size * num_tokens

    # Gate + Up + Down 三个矩阵乘法 (共享专家数 * 单专家计算量)
    flops_per_expert = 2 * total_tokens * model.hidden_size * expert_intermediate * 3
    total_flops = flops_per_expert * moe.num_shared_experts
    tflops = total_flops / 1e12
    compute_time = tflops / hardware.chip.compute_tflops_fp16 * 1000

    # 内存访问
    weight_bytes = 3 * model.hidden_size * expert_intermediate * bytes_per_elem * moe.num_shared_experts
    io_bytes = total_tokens * (model.hidden_size * 2 + expert_intermediate) * bytes_per_elem
    data_gb = (weight_bytes + io_bytes) / (1024 ** 3)
    memory_time = calc_hbm_read_latency(data_gb, hardware)

    return max(compute_time, memory_time)


def calc_ep_dispatch_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    ep_bandwidth_gbps: float,
    ep_latency_us: float,
) -> float:
    """
    计算 EP Dispatch (Token 分发) 延迟

    All2All 通信: 每个芯片将 token 发送到对应专家所在的芯片
    数据量 = B * S * H * bytes * num_experts_per_tok
    """
    if model.moe_config is None or parallelism.ep <= 1:
        return 0.0

    moe = model.moe_config
    bytes_per_elem = get_bytes_per_element(model.dtype)

    # 每个 token 发送到 num_experts_per_tok 个专家
    data_bytes = inference.batch_size * num_tokens * model.hidden_size * bytes_per_elem * moe.num_experts_per_tok
    data_gb = data_bytes / (1024 ** 3)

    return calc_ep_alltoall_latency(data_gb, ep_bandwidth_gbps, ep_latency_us, parallelism.ep)


def calc_ep_combine_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    ep_bandwidth_gbps: float,
    ep_latency_us: float,
) -> float:
    """
    计算 EP Combine (结果收集) 延迟

    All2All 通信: 每个芯片收集专家计算结果
    数据量与 Dispatch 相同
    """
    return calc_ep_dispatch_latency(
        model, inference, parallelism, hardware, num_tokens,
        ep_bandwidth_gbps, ep_latency_us
    )


def is_moe_layer(layer_index: int, model: LLMModelConfig) -> bool:
    """
    判断是否为 MoE 层

    通过 moe_config.first_k_dense_replace 配置前K层使用Dense FFN
    例如 DeepSeek V3: first_k_dense_replace=3，即 layer 0-2 使用 Dense，layer 3+ 使用 MoE
    """
    if model.moe_config is None:
        return False

    # 使用配置的 first_k_dense_replace 参数
    first_k = model.moe_config.first_k_dense_replace
    return layer_index >= first_k


def calc_ffn_gate_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算 FFN Gate 投影延迟 (SwiGLU 的 gate 部分)

    FLOPS = 2 * B * S * H * I / TP

    内存访问:
    - 读取权重: [H, I] / TP
    - 读取输入: [B, S, H]
    - 写入输出: [B, S, I / TP]
    """
    intermediate = model.intermediate_size
    bytes_per_elem = get_bytes_per_element(model.dtype)

    # 计算时间
    flops = 2 * inference.batch_size * num_tokens * model.hidden_size * intermediate / parallelism.tp
    tflops = flops / 1e12
    compute_time = tflops / hardware.chip.compute_tflops_fp16 * 1000

    # 内存访问时间
    weight_bytes = model.hidden_size * intermediate * bytes_per_elem / parallelism.tp
    input_bytes = inference.batch_size * num_tokens * model.hidden_size * bytes_per_elem
    output_bytes = inference.batch_size * num_tokens * intermediate * bytes_per_elem / parallelism.tp
    data_gb = (weight_bytes + input_bytes + output_bytes) / (1024 ** 3)
    memory_time = calc_hbm_read_latency(data_gb, hardware)

    return max(compute_time, memory_time)


def calc_ffn_up_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算 FFN Up 投影延迟

    FLOPS = 2 * B * S * H * I / TP
    与 Gate 相同的计算和内存模式
    """
    return calc_ffn_gate_latency(model, inference, parallelism, hardware, num_tokens)


def calc_ffn_down_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算 FFN Down 投影延迟

    FLOPS = 2 * B * S * I * H / TP

    内存访问:
    - 读取权重: [I, H] / TP
    - 读取输入: [B, S, I / TP]
    - 写入输出: [B, S, H]
    """
    intermediate = model.intermediate_size
    bytes_per_elem = get_bytes_per_element(model.dtype)

    # 计算时间
    flops = 2 * inference.batch_size * num_tokens * intermediate * model.hidden_size / parallelism.tp
    tflops = flops / 1e12
    compute_time = tflops / hardware.chip.compute_tflops_fp16 * 1000

    # 内存访问时间
    weight_bytes = intermediate * model.hidden_size * bytes_per_elem / parallelism.tp
    input_bytes = inference.batch_size * num_tokens * intermediate * bytes_per_elem / parallelism.tp
    output_bytes = inference.batch_size * num_tokens * model.hidden_size * bytes_per_elem
    data_gb = (weight_bytes + input_bytes + output_bytes) / (1024 ** 3)
    memory_time = calc_hbm_read_latency(data_gb, hardware)

    return max(compute_time, memory_time)


def calc_lm_head_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算 LM Head (最后的词表投影) 延迟

    FLOPS = 2 * B * S * H * V / TP

    内存访问:
    - 读取权重: [H, V] / TP (通常与 embedding 共享)
    - 读取输入: [B, S, H]
    - 写入输出: [B, S, V / TP]
    """
    bytes_per_elem = get_bytes_per_element(model.dtype)

    # 计算时间
    flops = 2 * inference.batch_size * num_tokens * model.hidden_size * model.vocab_size / parallelism.tp
    tflops = flops / 1e12
    compute_time = tflops / hardware.chip.compute_tflops_fp16 * 1000

    # 内存访问时间
    weight_bytes = model.hidden_size * model.vocab_size * bytes_per_elem / parallelism.tp
    input_bytes = inference.batch_size * num_tokens * model.hidden_size * bytes_per_elem
    output_bytes = inference.batch_size * num_tokens * model.vocab_size * bytes_per_elem / parallelism.tp
    data_gb = (weight_bytes + input_bytes + output_bytes) / (1024 ** 3)
    memory_time = calc_hbm_read_latency(data_gb, hardware)

    return max(compute_time, memory_time)


# ============================================
# 集合通信延迟
# ============================================

def calc_tp_allreduce_latency(
    data_gb: float,
    bandwidth_gbps: float,
    latency_us: float,
    tp_size: int,
) -> float:
    """
    计算 TP AllReduce 延迟

    使用 Ring AllReduce 算法:
    时间 = 2 * (n-1)/n * data_size / bandwidth + 2 * (n-1) * latency

    Args:
        data_gb: 数据量 (GB)
        bandwidth_gbps: 链路带宽 (Gbps)
        latency_us: 链路延迟 (us)
        tp_size: TP 组大小

    Returns:
        延迟 (ms)
    """
    if tp_size <= 1 or data_gb <= 0:
        return 0.0

    n = tp_size
    # Ring AllReduce 传输量 = 2 * (n-1)/n * data_size
    transfer_factor = 2 * (n - 1) / n
    transfer_time_ms = (transfer_factor * data_gb / bandwidth_gbps) * 1000

    # 延迟 = 2 * (n-1) * 单跳延迟
    total_latency_ms = 2 * (n - 1) * latency_us / 1000

    return transfer_time_ms + total_latency_ms


def calc_pp_p2p_latency(
    data_gb: float,
    bandwidth_gbps: float,
    latency_us: float,
) -> float:
    """
    计算 PP 点对点通信延迟

    Args:
        data_gb: 数据量 (GB)
        bandwidth_gbps: 链路带宽 (Gbps)
        latency_us: 链路延迟 (us)

    Returns:
        延迟 (ms)
    """
    if data_gb <= 0:
        return 0.0

    transfer_time_ms = (data_gb / bandwidth_gbps) * 1000
    latency_ms = latency_us / 1000

    return transfer_time_ms + latency_ms


def calc_ep_alltoall_latency(
    data_gb: float,
    bandwidth_gbps: float,
    latency_us: float,
    ep_size: int,
) -> float:
    """
    计算 EP All-to-All 通信延迟

    All-to-All 每个节点发送和接收 (n-1)/n 的数据

    Args:
        data_gb: 总数据量 (GB)
        bandwidth_gbps: 链路带宽 (Gbps)
        latency_us: 链路延迟 (us)
        ep_size: EP 组大小

    Returns:
        延迟 (ms)
    """
    if ep_size <= 1 or data_gb <= 0:
        return 0.0

    n = ep_size
    transfer_factor = (n - 1) / n
    transfer_time_ms = (transfer_factor * data_gb / bandwidth_gbps) * 1000

    # 延迟 = (n-1) * 单跳延迟
    total_latency_ms = (n - 1) * latency_us / 1000

    return transfer_time_ms + total_latency_ms


# ============================================
# 综合延迟计算
# ============================================

def calc_single_layer_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    context_length: int,
    tp_bandwidth_gbps: float,
    tp_latency_us: float,
) -> dict[str, float]:
    """
    计算单层 Transformer 的所有子操作延迟

    Returns:
        包含各子操作延迟的字典 (ms)
    """
    bytes_per_elem = get_bytes_per_element(model.dtype)

    # 计算各子操作
    layernorm_1 = calc_layernorm_latency(model, inference, parallelism, hardware, num_tokens)

    qkv = calc_attention_qkv_latency(model, inference, parallelism, hardware, num_tokens)
    score = calc_attention_score_latency(model, inference, parallelism, hardware, num_tokens, context_length)
    softmax = calc_attention_softmax_latency(model, inference, parallelism, hardware, num_tokens, context_length)
    output = calc_attention_output_latency(model, inference, parallelism, hardware, num_tokens, context_length)

    layernorm_2 = calc_layernorm_latency(model, inference, parallelism, hardware, num_tokens)

    ffn_gate = calc_ffn_gate_latency(model, inference, parallelism, hardware, num_tokens)
    ffn_up = calc_ffn_up_latency(model, inference, parallelism, hardware, num_tokens)
    ffn_down = calc_ffn_down_latency(model, inference, parallelism, hardware, num_tokens)

    # TP AllReduce: Attention 输出后 + FFN 输出后
    attn_output_size_gb = (inference.batch_size * num_tokens * model.hidden_size * bytes_per_elem) / (1024 ** 3)
    tp_allreduce_attn = calc_tp_allreduce_latency(attn_output_size_gb, tp_bandwidth_gbps, tp_latency_us, parallelism.tp)
    tp_allreduce_ffn = calc_tp_allreduce_latency(attn_output_size_gb, tp_bandwidth_gbps, tp_latency_us, parallelism.tp)

    # KV Cache 读写
    kv_read = calc_kv_cache_read_latency(model, inference, parallelism, hardware, context_length)
    kv_write = calc_kv_cache_write_latency(model, inference, parallelism, hardware, num_tokens)

    return {
        "layernorm_1": layernorm_1,
        "attention_qkv": qkv,
        "kv_cache_read": kv_read,
        "attention_score": score,
        "attention_softmax": softmax,
        "attention_output": output,
        "tp_allreduce_attn": tp_allreduce_attn,
        "kv_cache_write": kv_write,
        "layernorm_2": layernorm_2,
        "ffn_gate": ffn_gate,
        "ffn_up": ffn_up,
        "ffn_down": ffn_down,
        "tp_allreduce_ffn": tp_allreduce_ffn,
    }


# ============================================
# Kernel Fusion 优化
# ============================================

# Kernel Fusion 效率因子
# 融合后的操作可以减少中间内存访问，提高效率
FUSION_MEMORY_EFFICIENCY = 0.6  # 融合后内存开销降低 40%

# 计算-通信重叠系数
# 实际部署中，TP 通信可以与下一层计算部分重叠
OVERLAP_COEFFICIENTS = {
    'tp_compute_overlap': 0.7,   # 70% 的 TP 通信可与下一层计算重叠
    'pp_compute_overlap': 0.8,   # 80% 的 PP 通信可与计算重叠
}


def calc_fused_layernorm_qkv_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算融合的 LayerNorm + QKV 投影延迟

    Kernel Fusion 原理:
    - 将 LayerNorm 和 QKV 投影融合为一个 kernel
    - 减少中间结果的 HBM 读写
    - 计算取最大（可流水线），内存需要累加但有融合收益
    """
    # 分别计算各部分
    ln_latency = calc_layernorm_latency(model, inference, parallelism, hardware, num_tokens)

    if model.attention_type == 'mla' and model.mla_config is not None:
        # MLA: Q 投影 + KV 压缩
        q_latency = calc_mla_q_projection_latency(model, inference, parallelism, hardware, num_tokens)
        kv_latency = calc_mla_kv_compression_latency(model, inference, parallelism, hardware, num_tokens)
        qkv_latency = q_latency + kv_latency
    else:
        qkv_latency = calc_attention_qkv_latency(model, inference, parallelism, hardware, num_tokens)

    # 融合后: 计算部分可流水线（取较大值的一定比例），内存部分有融合收益
    # 简化模型: 融合后延迟 = max(ln, qkv) + min(ln, qkv) * fusion_factor
    fused_latency = max(ln_latency, qkv_latency) + min(ln_latency, qkv_latency) * FUSION_MEMORY_EFFICIENCY
    return fused_latency


def calc_fused_ffn_gate_up_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """
    计算融合的 FFN Gate + Up 投影延迟

    SwiGLU 的 Gate 和 Up 可以并行计算，然后做 element-wise 乘法
    融合后减少中间内存访问
    """
    gate_latency = calc_ffn_gate_latency(model, inference, parallelism, hardware, num_tokens)
    up_latency = calc_ffn_up_latency(model, inference, parallelism, hardware, num_tokens)

    # Gate 和 Up 可以完全并行，融合后取较大值
    # 但需要额外的 element-wise 乘法 (SiLU)，开销很小
    fused_latency = max(gate_latency, up_latency) * 1.05  # 5% 额外开销
    return fused_latency


def calc_single_layer_latency_with_mla(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    context_length: int,
    tp_bandwidth_gbps: float,
    tp_latency_us: float,
) -> dict[str, float]:
    """
    计算单层 Transformer 的所有子操作延迟 (支持 MLA)

    如果模型配置了 MLA，使用 MLA 专用的计算函数
    """
    bytes_per_elem = get_bytes_per_element(model.dtype)
    use_mla = model.attention_type == 'mla' and model.mla_config is not None

    # LayerNorm
    layernorm_1 = calc_layernorm_latency(model, inference, parallelism, hardware, num_tokens)

    # Attention 部分
    if use_mla:
        qkv = calc_mla_q_projection_latency(model, inference, parallelism, hardware, num_tokens)
        kv_compress = calc_mla_kv_compression_latency(model, inference, parallelism, hardware, num_tokens)
        score = calc_mla_attention_score_latency(model, inference, parallelism, hardware, num_tokens, context_length)
        kv_read = calc_mla_kv_cache_read_latency(model, inference, parallelism, hardware, context_length)
        kv_write = calc_mla_kv_cache_write_latency(model, inference, parallelism, hardware, num_tokens)
        output = calc_mla_output_latency(model, inference, parallelism, hardware, num_tokens, context_length)
    else:
        qkv = calc_attention_qkv_latency(model, inference, parallelism, hardware, num_tokens)
        kv_compress = 0.0
        score = calc_attention_score_latency(model, inference, parallelism, hardware, num_tokens, context_length)
        kv_read = calc_kv_cache_read_latency(model, inference, parallelism, hardware, context_length)
        kv_write = calc_kv_cache_write_latency(model, inference, parallelism, hardware, num_tokens)
        output = calc_attention_output_latency(model, inference, parallelism, hardware, num_tokens, context_length)

    softmax = calc_attention_softmax_latency(model, inference, parallelism, hardware, num_tokens, context_length)

    # FFN
    layernorm_2 = calc_layernorm_latency(model, inference, parallelism, hardware, num_tokens)
    ffn_gate = calc_ffn_gate_latency(model, inference, parallelism, hardware, num_tokens)
    ffn_up = calc_ffn_up_latency(model, inference, parallelism, hardware, num_tokens)
    ffn_down = calc_ffn_down_latency(model, inference, parallelism, hardware, num_tokens)

    # TP AllReduce
    attn_output_size_gb = (inference.batch_size * num_tokens * model.hidden_size * bytes_per_elem) / (1024 ** 3)
    tp_allreduce_attn = calc_tp_allreduce_latency(attn_output_size_gb, tp_bandwidth_gbps, tp_latency_us, parallelism.tp)
    tp_allreduce_ffn = calc_tp_allreduce_latency(attn_output_size_gb, tp_bandwidth_gbps, tp_latency_us, parallelism.tp)

    result = {
        "layernorm_1": layernorm_1,
        "attention_qkv": qkv,
        "kv_cache_read": kv_read,
        "attention_score": score,
        "attention_softmax": softmax,
        "attention_output": output,
        "tp_allreduce_attn": tp_allreduce_attn,
        "kv_cache_write": kv_write,
        "layernorm_2": layernorm_2,
        "ffn_gate": ffn_gate,
        "ffn_up": ffn_up,
        "ffn_down": ffn_down,
        "tp_allreduce_ffn": tp_allreduce_ffn,
    }

    # MLA 特有的 KV 压缩
    if use_mla:
        result["mla_kv_compress"] = kv_compress

    return result


def calc_single_layer_latency_fused(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    context_length: int,
    tp_bandwidth_gbps: float,
    tp_latency_us: float,
    enable_fusion: bool = True,
    enable_overlap: bool = True,
) -> dict[str, float]:
    """
    计算单层 Transformer 的所有子操作延迟 (支持 Kernel Fusion 和计算通信重叠)

    Kernel Fusion 组:
    1. LayerNorm + QKV -> fused_ln_qkv
    2. Gate + Up -> fused_ffn_gate_up
    3. Softmax + Score (FlashAttention 风格) -> fused_attention

    计算-通信重叠:
    - TP 通信的 70% 可以与下一层计算重叠

    Returns:
        包含各子操作延迟的字典 (ms)
    """
    bytes_per_elem = get_bytes_per_element(model.dtype)
    use_mla = model.attention_type == 'mla' and model.mla_config is not None

    if enable_fusion:
        # 融合的 LayerNorm + QKV
        fused_ln_qkv = calc_fused_layernorm_qkv_latency(model, inference, parallelism, hardware, num_tokens)

        # KV Cache (MLA 使用压缩的 KV)
        if use_mla:
            kv_read = calc_mla_kv_cache_read_latency(model, inference, parallelism, hardware, context_length)
            kv_write = calc_mla_kv_cache_write_latency(model, inference, parallelism, hardware, num_tokens)
            score = calc_mla_attention_score_latency(model, inference, parallelism, hardware, num_tokens, context_length)
            output = calc_mla_output_latency(model, inference, parallelism, hardware, num_tokens, context_length)
        else:
            kv_read = calc_kv_cache_read_latency(model, inference, parallelism, hardware, context_length)
            kv_write = calc_kv_cache_write_latency(model, inference, parallelism, hardware, num_tokens)
            score = calc_attention_score_latency(model, inference, parallelism, hardware, num_tokens, context_length)
            output = calc_attention_output_latency(model, inference, parallelism, hardware, num_tokens, context_length)

        # Softmax (FlashAttention 融合到 Score 中，开销降低)
        softmax = calc_attention_softmax_latency(model, inference, parallelism, hardware, num_tokens, context_length)
        fused_attention = score + softmax * 0.3  # FlashAttention 优化，Softmax 开销降低 70%

        # 融合的 FFN Gate + Up
        layernorm_2 = calc_layernorm_latency(model, inference, parallelism, hardware, num_tokens)
        fused_ffn_gate_up = calc_fused_ffn_gate_up_latency(model, inference, parallelism, hardware, num_tokens)
        ffn_down = calc_ffn_down_latency(model, inference, parallelism, hardware, num_tokens)

        # TP AllReduce
        attn_output_size_gb = (inference.batch_size * num_tokens * model.hidden_size * bytes_per_elem) / (1024 ** 3)
        tp_allreduce_attn = calc_tp_allreduce_latency(attn_output_size_gb, tp_bandwidth_gbps, tp_latency_us, parallelism.tp)
        tp_allreduce_ffn = calc_tp_allreduce_latency(attn_output_size_gb, tp_bandwidth_gbps, tp_latency_us, parallelism.tp)

        # 计算-通信重叠
        if enable_overlap:
            # TP 通信可以与计算部分重叠，只计算非重叠部分
            tp_allreduce_attn *= (1 - OVERLAP_COEFFICIENTS['tp_compute_overlap'])
            tp_allreduce_ffn *= (1 - OVERLAP_COEFFICIENTS['tp_compute_overlap'])

        return {
            "fused_ln_qkv": fused_ln_qkv,
            "kv_cache_read": kv_read,
            "fused_attention": fused_attention,
            "attention_output": output,
            "tp_allreduce_attn": tp_allreduce_attn,
            "kv_cache_write": kv_write,
            "layernorm_2": layernorm_2,
            "fused_ffn_gate_up": fused_ffn_gate_up,
            "ffn_down": ffn_down,
            "tp_allreduce_ffn": tp_allreduce_ffn,
        }
    else:
        # 不融合，使用原始计算
        return calc_single_layer_latency_with_mla(
            model, inference, parallelism, hardware,
            num_tokens, context_length, tp_bandwidth_gbps, tp_latency_us
        )
