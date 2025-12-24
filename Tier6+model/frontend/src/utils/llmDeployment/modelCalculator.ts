/**
 * LLM 部署分析系统 - 模型计算器
 *
 * 计算模型参数量、显存需求、FLOPs
 */

import {
  LLMModelConfig,
  InferenceConfig,
  ParallelismStrategy,
  MemoryAnalysis,
  DataType,
  getBytesPerElement,
} from './types';

// ============================================
// 参数量计算
// ============================================

/**
 * 计算模型总参数量
 */
export function calculateModelParams(model: LLMModelConfig): number {
  const H = model.hidden_size;
  const L = model.num_layers;
  const V = model.vocab_size;
  const I = model.intermediate_size;
  const numHeads = model.num_attention_heads;
  const numKVHeads = model.num_kv_heads;

  // 每个头的维度
  const headDim = H / numHeads;

  // Embedding 层: token embedding + position embedding (如果有)
  // 通常只有 token embedding: V * H
  const embeddingParams = V * H;

  // 输出层 (LM Head): H * V (通常与embedding共享权重，但计算时分开)
  const lmHeadParams = H * V;

  // 每层 Transformer 参数:
  // Attention:
  //   - Q: H * H
  //   - K: H * (H / numHeads * numKVHeads) = H * headDim * numKVHeads
  //   - V: H * headDim * numKVHeads
  //   - O: H * H
  const qParams = H * H;
  const kParams = H * headDim * numKVHeads;
  const vParams = H * headDim * numKVHeads;
  const oParams = H * H;
  const attentionParams = qParams + kParams + vParams + oParams;

  // FFN (SwiGLU):
  //   - gate: H * I
  //   - up: H * I
  //   - down: I * H
  let ffnParams = 3 * H * I;

  // MoE: FFN 参数 × 专家数
  if (model.model_type === 'moe' && model.moe_config) {
    const numExperts = model.moe_config.num_experts;
    const numSharedExperts = model.moe_config.num_shared_experts ?? 0;
    ffnParams = 3 * H * I * (numExperts + numSharedExperts);

    // Router: H * numExperts
    ffnParams += H * numExperts;
  }

  // LayerNorm: 2 * H (gamma + beta) × 2 (attention前 + FFN前)
  const layerNormParams = 4 * H;

  // 每层总参数
  const paramsPerLayer = attentionParams + ffnParams + layerNormParams;

  // 总参数
  const totalParams = embeddingParams + L * paramsPerLayer + lmHeadParams;

  return totalParams;
}

/**
 * 计算每层参数量
 */
export function calculateParamsPerLayer(model: LLMModelConfig): {
  attention: number;
  ffn: number;
  layerNorm: number;
  total: number;
} {
  const H = model.hidden_size;
  const I = model.intermediate_size;
  const numHeads = model.num_attention_heads;
  const numKVHeads = model.num_kv_heads;
  const headDim = H / numHeads;

  const attention = H * H + 2 * H * headDim * numKVHeads + H * H;

  let ffn = 3 * H * I;
  if (model.model_type === 'moe' && model.moe_config) {
    const numExperts = model.moe_config.num_experts;
    const numSharedExperts = model.moe_config.num_shared_experts ?? 0;
    ffn = 3 * H * I * (numExperts + numSharedExperts) + H * numExperts;
  }

  const layerNorm = 4 * H;

  return {
    attention,
    ffn,
    layerNorm,
    total: attention + ffn + layerNorm,
  };
}

// ============================================
// 显存计算
// ============================================

/**
 * 计算模型权重显存 (每芯片)
 */
export function calculateModelMemory(
  model: LLMModelConfig,
  parallelism: ParallelismStrategy
): number {
  const totalParams = calculateModelParams(model);
  const bytesPerParam = getBytesPerElement(model.dtype);

  // 模型按 TP 和 PP 切分
  const paramsPerChip = totalParams / parallelism.tp / parallelism.pp;

  // MoE 模型按 EP 切分专家
  let effectiveParams = paramsPerChip;
  if (model.model_type === 'moe' && model.moe_config && parallelism.ep > 1) {
    // 专家参数按EP切分，非专家参数（attention等）不变
    const layerParams = calculateParamsPerLayer(model);
    const expertRatio = layerParams.ffn / layerParams.total;
    const nonExpertParams = paramsPerChip * (1 - expertRatio);
    const expertParams = paramsPerChip * expertRatio / parallelism.ep;
    effectiveParams = nonExpertParams + expertParams;
  }

  return effectiveParams * bytesPerParam / 1e9; // GB
}

/**
 * 计算 KV Cache 显存 (每芯片)
 */
export function calculateKVCacheMemory(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy
): number {
  const numHeads = model.num_attention_heads;
  const numKVHeads = model.num_kv_heads;
  const headDim = model.hidden_size / numHeads;
  const numLayers = model.num_layers;
  const bytesPerElement = getBytesPerElement(model.dtype);

  // KV Cache 大小 = 2 (K+V) × batch × seq × kv_heads × head_dim × layers × bytes
  // 按 TP 切分 KV heads
  const kvHeadsPerChip = Math.ceil(numKVHeads / parallelism.tp);
  // 按 PP 切分 layers
  const layersPerChip = Math.ceil(numLayers / parallelism.pp);

  const kvCacheBytes =
    2 *
    inference.batch_size *
    inference.max_seq_length *
    kvHeadsPerChip *
    headDim *
    layersPerChip *
    bytesPerElement;

  return kvCacheBytes / 1e9; // GB
}

/**
 * 计算激活值显存 (每芯片)
 *
 * 激活值包括中间结果，主要在前向传播时占用
 * 对于推理，激活值相对较小
 */
export function calculateActivationMemory(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy
): number {
  const H = model.hidden_size;
  const I = model.intermediate_size;
  const numLayers = model.num_layers;
  const bytesPerElement = getBytesPerElement(model.dtype);

  // Prefill 阶段激活值 (最大)
  const seqLen = inference.input_seq_length;
  const batch = inference.batch_size;

  // 每层激活值估算:
  // - Attention 输入: batch × seq × H
  // - Q, K, V: 3 × batch × seq × H
  // - Attention 输出: batch × seq × H
  // - FFN 中间: batch × seq × I
  // 总计约: batch × seq × (5H + I)
  const activationPerLayer = batch * seqLen * (5 * H + I) * bytesPerElement;

  // 按 PP 切分 layers，但激活值需要保留用于 PP 通信
  // 简化：只考虑当前 PP stage 的激活值
  const layersPerChip = Math.ceil(numLayers / parallelism.pp);

  // TP 会减少每个芯片的激活值（H 维度切分）
  const activationBytes = activationPerLayer * layersPerChip / parallelism.tp;

  // 推理时只需要前向传播，激活值可以逐层释放
  // 实际只需保留 1-2 层的激活值
  const effectiveActivation = activationBytes * 2 / layersPerChip;

  return effectiveActivation / 1e9; // GB
}

/**
 * 计算其他显存开销
 * 包括：CUDA context、临时缓冲区、碎片等
 */
export function calculateOverheadMemory(
  model: LLMModelConfig,
  inference: InferenceConfig
): number {
  // 固定开销：CUDA context 约 1GB
  const cudaContext = 1.0;

  // 临时缓冲区：约 0.5GB
  const tempBuffers = 0.5;

  // 碎片：约 10% 的显存
  const fragmentation = 0.5;

  return cudaContext + tempBuffers + fragmentation;
}

/**
 * 完整显存分析
 */
export function analyzeMemory(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  chipMemoryGB: number
): MemoryAnalysis {
  const modelMemory = calculateModelMemory(model, parallelism);
  const kvCacheMemory = calculateKVCacheMemory(model, inference, parallelism);
  const activationMemory = calculateActivationMemory(model, inference, parallelism);
  const overhead = calculateOverheadMemory(model, inference);

  const totalPerChip = modelMemory + kvCacheMemory + activationMemory + overhead;
  const utilization = totalPerChip / chipMemoryGB;
  const isSufficient = totalPerChip <= chipMemoryGB * 0.95; // 留 5% 余量

  return {
    model_memory_gb: modelMemory,
    kv_cache_memory_gb: kvCacheMemory,
    activation_memory_gb: activationMemory,
    overhead_gb: overhead,
    total_per_chip_gb: totalPerChip,
    is_memory_sufficient: isSufficient,
    memory_utilization: Math.min(utilization, 1.0),
  };
}

// ============================================
// FLOPs 计算
// ============================================

/**
 * 计算单层 Transformer 的 FLOPs (Prefill 阶段)
 */
export function calculateLayerFlopsPrefill(
  model: LLMModelConfig,
  batchSize: number,
  seqLen: number
): number {
  const H = model.hidden_size;
  const I = model.intermediate_size;
  const numHeads = model.num_attention_heads;
  const numKVHeads = model.num_kv_heads;
  const headDim = H / numHeads;

  // Attention FLOPs:
  // QKV projection: 3 × 2 × batch × seq × H × H (考虑 GQA 时 K,V 更小)
  const qProj = 2 * batchSize * seqLen * H * H;
  const kvProj = 2 * 2 * batchSize * seqLen * H * (headDim * numKVHeads);
  const oProj = 2 * batchSize * seqLen * H * H;

  // Attention score: batch × heads × seq × seq × head_dim × 2 (Q×K + softmax×V)
  const attnScore = 2 * batchSize * numHeads * seqLen * seqLen * headDim;
  const attnOutput = 2 * batchSize * numHeads * seqLen * seqLen * headDim;

  const attentionFlops = qProj + kvProj + oProj + attnScore + attnOutput;

  // FFN FLOPs (SwiGLU):
  // gate: 2 × batch × seq × H × I
  // up: 2 × batch × seq × H × I
  // down: 2 × batch × seq × I × H
  // SiLU: batch × seq × I
  let ffnFlops = 2 * batchSize * seqLen * H * I * 3 + batchSize * seqLen * I;

  // MoE: 只有部分专家被激活
  if (model.model_type === 'moe' && model.moe_config) {
    const expertsPerTok = model.moe_config.num_experts_per_tok;
    const numSharedExperts = model.moe_config.num_shared_experts ?? 0;
    // 每个 token 激活 expertsPerTok 个专家 + 共享专家
    ffnFlops = (2 * batchSize * seqLen * H * I * 3 + batchSize * seqLen * I) *
               (expertsPerTok + numSharedExperts);
    // Router FLOPs
    ffnFlops += 2 * batchSize * seqLen * H * model.moe_config.num_experts;
  }

  // LayerNorm: 约 5 × batch × seq × H × 2
  const layerNormFlops = 10 * batchSize * seqLen * H * 2;

  return attentionFlops + ffnFlops + layerNormFlops;
}

/**
 * 计算单层 Transformer 的 FLOPs (Decode 阶段，单 token)
 */
export function calculateLayerFlopsDecode(
  model: LLMModelConfig,
  batchSize: number,
  contextLen: number
): number {
  const H = model.hidden_size;
  const I = model.intermediate_size;
  const numHeads = model.num_attention_heads;
  const numKVHeads = model.num_kv_heads;
  const headDim = H / numHeads;

  // Decode 时 seq=1，但需要与整个 context 做 attention
  const seqLen = 1;

  // QKV projection (只处理 1 个 token)
  const qProj = 2 * batchSize * seqLen * H * H;
  const kvProj = 2 * 2 * batchSize * seqLen * H * (headDim * numKVHeads);
  const oProj = 2 * batchSize * seqLen * H * H;

  // Attention: 1 个 query 与 contextLen 个 key/value
  const attnScore = 2 * batchSize * numHeads * seqLen * contextLen * headDim;
  const attnOutput = 2 * batchSize * numHeads * seqLen * contextLen * headDim;

  const attentionFlops = qProj + kvProj + oProj + attnScore + attnOutput;

  // FFN FLOPs
  let ffnFlops = 2 * batchSize * seqLen * H * I * 3 + batchSize * seqLen * I;

  if (model.model_type === 'moe' && model.moe_config) {
    const expertsPerTok = model.moe_config.num_experts_per_tok;
    const numSharedExperts = model.moe_config.num_shared_experts ?? 0;
    ffnFlops = (2 * batchSize * seqLen * H * I * 3 + batchSize * seqLen * I) *
               (expertsPerTok + numSharedExperts);
    ffnFlops += 2 * batchSize * seqLen * H * model.moe_config.num_experts;
  }

  const layerNormFlops = 10 * batchSize * seqLen * H * 2;

  return attentionFlops + ffnFlops + layerNormFlops;
}

/**
 * 计算 Prefill 阶段总 FLOPs
 */
export function calculatePrefillFlops(
  model: LLMModelConfig,
  inference: InferenceConfig
): number {
  const layerFlops = calculateLayerFlopsPrefill(
    model,
    inference.batch_size,
    inference.input_seq_length
  );
  return layerFlops * model.num_layers;
}

/**
 * 计算 Decode 阶段每 token 的 FLOPs
 */
export function calculateDecodeFlopsPerToken(
  model: LLMModelConfig,
  inference: InferenceConfig,
  currentContextLen: number
): number {
  const layerFlops = calculateLayerFlopsDecode(
    model,
    inference.batch_size,
    currentContextLen
  );
  return layerFlops * model.num_layers;
}

/**
 * 计算完整推理的总 FLOPs
 */
export function calculateTotalInferenceFlops(
  model: LLMModelConfig,
  inference: InferenceConfig
): number {
  // Prefill FLOPs
  const prefillFlops = calculatePrefillFlops(model, inference);

  // Decode FLOPs: 每个输出 token 的 FLOPs 随 context 增长
  let decodeFlops = 0;
  for (let i = 0; i < inference.output_seq_length; i++) {
    const contextLen = inference.input_seq_length + i;
    decodeFlops += calculateDecodeFlopsPerToken(model, inference, contextLen);
  }

  return prefillFlops + decodeFlops;
}

/**
 * 计算每 token 的平均 FLOPs (用于估算吞吐量)
 */
export function calculateFlopsPerToken(model: LLMModelConfig): number {
  // 简化估算：使用 context=512 的单 token FLOPs
  const inference: InferenceConfig = {
    batch_size: 1,
    input_seq_length: 512,
    output_seq_length: 1,
    max_seq_length: 513,
  };
  return calculateDecodeFlopsPerToken(model, inference, 512);
}
