/**
 * LLM 部署分析系统 - 延迟估算器
 *
 * 估算 Prefill/Decode 延迟，识别瓶颈
 */

import {
  LLMModelConfig,
  InferenceConfig,
  ParallelismStrategy,
  HardwareConfig,
  LatencyAnalysis,
  LatencyPercentiles,
  BottleneckType,
  CostAnalysis,
  BottleneckAnalysis,
  PhaseBottleneckAnalysis,
} from './types';
import {
  calculatePrefillFlops,
  calculateDecodeFlopsPerToken,
  calculateModelMemory,
  calculateKVCacheMemory,
} from './modelCalculator';
import {
  calculateTPCommVolumePrefill,
  calculateTPCommVolumeDecode,
  calculatePPCommVolumePrefill,
  calculatePPCommVolumeDecode,
  calculateEPCommVolumePrefill,
  calculateEPCommVolumeDecode,
  calculateSPCommVolumePrefill,
  calculateSPCommVolumeDecode,
} from './commCalculator';

// ============================================
// 常量定义
// ============================================

/** HBM 效率因子 (实际带宽 / 峰值带宽) */
const HBM_EFFICIENCY = 0.85;

// ============================================
// 动态 MFU 估算
// ============================================

/**
 * 基于 Roofline 模型估算可达 MFU
 *
 * Roofline 模型原理:
 * - 算术强度 (AI) = FLOPs / Bytes
 * - 峰点 (Ridge Point) = 峰值算力 / 峰值带宽
 * - 实际算力利用率 = min(1, AI / Ridge Point)
 *
 * 参考:
 * - NVIDIA: https://developer.nvidia.com/blog/achieving-optimal-performance-with-roofline-analysis/
 */
export function estimateAchievableMFU(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig,
  phase: 'prefill' | 'decode',
  contextLength?: number
): number {
  // 计算每 token 的 FLOPs
  const avgContext = contextLength ?? (inference.input_seq_length + inference.output_seq_length / 2);
  const flopsPerToken = phase === 'prefill'
    ? calculatePrefillFlops(model, inference) / inference.input_seq_length
    : calculateDecodeFlopsPerToken(model, inference, avgContext);

  // 计算每 token 需要读取的数据量 (bytes)
  // 模型权重 (每 token 都需要读取)
  const modelMemoryGB = calculateModelMemory(model, parallelism);
  const modelMemoryBytes = modelMemoryGB * 1e9;

  // KV Cache (仅 decode 阶段)
  let kvCacheBytes = 0;
  if (phase === 'decode') {
    const kvCacheGB = calculateKVCacheMemory(model, inference, parallelism);
    const kvCacheRatio = avgContext / inference.max_seq_length;
    kvCacheBytes = kvCacheGB * kvCacheRatio * 1e9;
  }

  const bytesPerToken = modelMemoryBytes + kvCacheBytes;

  // 算术强度 (FLOPs / Bytes)
  const arithmeticIntensity = flopsPerToken / bytesPerToken;

  // Ridge Point = 峰值算力 (FLOPs/s) / 峰值带宽 (Bytes/s)
  const peakFlops = hardware.chip.compute_tflops_fp16 * 1e12;
  const peakBandwidth = hardware.chip.memory_bandwidth_gbps * 1e9 * HBM_EFFICIENCY;
  const ridgePoint = peakFlops / peakBandwidth;

  // 基于 Roofline 的理论 MFU
  const theoreticalMFU = Math.min(1.0, arithmeticIntensity / ridgePoint);

  // 实际 MFU 通常更低，考虑以下因素:
  // - 并行效率损失 (TP/PP 通信开销)
  // - 启动开销 (kernel launch)
  // - 负载不均衡
  const parallelismOverhead = 1 - (parallelism.tp - 1) * 0.02 - (parallelism.pp - 1) * 0.03;
  const practicalFactor = 0.8; // 实际因素

  // 最终 MFU
  const achievableMFU = theoreticalMFU * parallelismOverhead * practicalFactor;

  // 限制范围
  // Prefill: 通常 30-50%
  // Decode: 通常 15-30%
  if (phase === 'prefill') {
    return Math.max(0.2, Math.min(0.5, achievableMFU));
  } else {
    return Math.max(0.1, Math.min(0.35, achievableMFU));
  }
}

// ============================================
// 计算延迟估算
// ============================================

/**
 * 估算 Prefill 阶段计算延迟
 *
 * 计算延迟 = max(计算时间, 访存时间)
 * - 计算时间 = FLOPs / (峰值算力 × MFU × 并行数)
 * - 访存时间 = 模型权重 / (HBM带宽 × 效率)
 */
export function estimatePrefillComputeLatency(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig,
  mfuEstimate?: number // 可选，不传则使用动态估算
): number {
  // 使用动态 MFU 估算（如果未指定）
  const mfu = mfuEstimate ?? estimateAchievableMFU(model, inference, parallelism, hardware, 'prefill');

  const totalFlops = calculatePrefillFlops(model, inference);

  // 每芯片算力 (FLOPs/s)
  const chipTflops = hardware.chip.compute_tflops_fp16;
  const flopsPerSecond = chipTflops * 1e12;

  // 并行带来的加速
  // TP: 计算完美切分
  // PP: 计算按 stage 切分，但有气泡
  const effectiveParallelism = parallelism.tp * parallelism.pp;

  // 每芯片需处理的 FLOPs
  const flopsPerChip = totalFlops / effectiveParallelism;

  // 计算时间 (秒)
  const computeTimeS = flopsPerChip / (flopsPerSecond * mfu);
  const computeTimeMs = computeTimeS * 1000;

  // 访存时间 (权重加载)
  // Prefill 阶段需要读取模型权重
  const modelMemoryGB = calculateModelMemory(model, parallelism);
  const memoryTimeS = modelMemoryGB / (hardware.chip.memory_bandwidth_gbps * HBM_EFFICIENCY);
  const memoryTimeMs = memoryTimeS * 1000;

  // 取较大值 (Roofline 模型)
  return Math.max(computeTimeMs, memoryTimeMs);
}

/**
 * 估算 Decode 阶段单 token 计算延迟
 *
 * Decode 阶段通常是 memory-bound，但仍需计算 compute time 作为参考
 */
export function estimateDecodeComputeLatency(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig,
  contextLength: number,
  mfuEstimate?: number // 可选，不传则使用动态估算
): number {
  // 使用动态 MFU 估算（如果未指定）
  const mfu = mfuEstimate ?? estimateAchievableMFU(model, inference, parallelism, hardware, 'decode', contextLength);

  const tokenFlops = calculateDecodeFlopsPerToken(model, inference, contextLength);

  const chipTflops = hardware.chip.compute_tflops_fp16;
  const flopsPerSecond = chipTflops * 1e12;
  const effectiveParallelism = parallelism.tp * parallelism.pp;
  const flopsPerChip = tokenFlops / effectiveParallelism;
  const computeTimeS = flopsPerChip / (flopsPerSecond * mfu);

  return computeTimeS * 1000; // ms
}

// ============================================
// 访存延迟估算
// ============================================

/**
 * 估算访存延迟 (Memory Bandwidth Bound)
 *
 * Decode 阶段通常是 memory-bound
 * 延迟 = 需读取的数据量 / (显存带宽 × HBM效率)
 */
export function estimateMemoryLatency(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): number {
  // 模型权重大小 (每芯片)
  const modelMemoryGB = calculateModelMemory(model, parallelism);

  // KV Cache 大小 (每芯片) - 每 token 需要读取
  const kvCacheGB = calculateKVCacheMemory(model, inference, parallelism);

  // Decode 时每 token 需读取的数据量
  // = 模型权重 + 当前 context 的 KV Cache
  const dataToReadGB = modelMemoryGB + kvCacheGB * (inference.input_seq_length / inference.max_seq_length);

  // 显存带宽 (考虑 HBM 效率)
  const effectiveBandwidthGBps = hardware.chip.memory_bandwidth_gbps * HBM_EFFICIENCY;

  // 访存时间
  const memoryTimeS = dataToReadGB / effectiveBandwidthGBps;

  return memoryTimeS * 1000; // ms
}

/**
 * 估算 Decode 阶段每 token 的访存延迟
 */
export function estimateDecodeMemoryLatency(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig,
  contextLength: number
): number {
  // 模型权重 (每 token 都要读一遍)
  const modelMemoryGB = calculateModelMemory(model, parallelism);

  // KV Cache 按当前 context 长度比例
  const kvCacheGB = calculateKVCacheMemory(model, inference, parallelism);
  const kvCacheRatio = contextLength / inference.max_seq_length;
  const currentKVCacheGB = kvCacheGB * kvCacheRatio;

  const dataToReadGB = modelMemoryGB + currentKVCacheGB;
  // 考虑 HBM 效率
  const effectiveBandwidthGBps = hardware.chip.memory_bandwidth_gbps * HBM_EFFICIENCY;
  const memoryTimeS = dataToReadGB / effectiveBandwidthGBps;

  return memoryTimeS * 1000; // ms
}

// ============================================
// 通信延迟估算
// ============================================

/**
 * 估算通信延迟
 *
 * 延迟 = 通信量 / 带宽 + 启动延迟
 */
export function estimateCommLatency(
  commVolumeGB: number,
  bandwidthGBps: number,
  startupLatencyUs: number
): number {
  if (commVolumeGB === 0) return 0;

  const transferTimeMs = (commVolumeGB / bandwidthGBps) * 1000;
  const startupLatencyMs = startupLatencyUs / 1000;

  return transferTimeMs + startupLatencyMs;
}

/**
 * 估算 Prefill 阶段总通信延迟
 */
export function estimatePrefillCommLatency(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): number {
  const numMicroBatches = inference.num_micro_batches ?? Math.max(parallelism.pp, 4);

  // 各策略通信量
  const tpComm = calculateTPCommVolumePrefill(model, inference, parallelism.tp);
  const ppComm = calculatePPCommVolumePrefill(model, inference, parallelism.pp, numMicroBatches);
  const epComm = calculateEPCommVolumePrefill(model, inference, parallelism.ep);
  const spComm = calculateSPCommVolumePrefill(model, inference, parallelism.sp);

  // 判断是否跨节点
  const totalChipsPerNode = hardware.node.chips_per_node;
  const totalChips = parallelism.dp * parallelism.tp * parallelism.pp * parallelism.ep;
  const numNodes = Math.ceil(totalChips / totalChipsPerNode);

  // TP 通常在节点内
  const tpBandwidth = hardware.node.intra_node_bandwidth_gbps;
  const tpLatencyUs = hardware.node.intra_node_latency_us;

  // PP/EP 可能跨节点
  const ppBandwidth = numNodes > 1 ? hardware.cluster.inter_node_bandwidth_gbps : tpBandwidth;
  const ppLatencyUs = numNodes > 1 ? hardware.cluster.inter_node_latency_us : tpLatencyUs;

  // 计算各通信延迟
  const tpLatency = estimateCommLatency(tpComm, tpBandwidth, tpLatencyUs);
  const ppLatency = estimateCommLatency(ppComm, ppBandwidth, ppLatencyUs);
  const epLatency = estimateCommLatency(epComm, ppBandwidth, ppLatencyUs);
  const spLatency = estimateCommLatency(spComm, tpBandwidth, tpLatencyUs);

  // 通信可以部分重叠，但 TP 是关键路径
  // 简化: 取最大值 + 其他的一定比例
  const maxLatency = Math.max(tpLatency, ppLatency, epLatency, spLatency);
  const otherLatency = (tpLatency + ppLatency + epLatency + spLatency - maxLatency) * 0.3;

  return maxLatency + otherLatency;
}

/**
 * 估算 Decode 阶段单 token 通信延迟
 */
export function estimateDecodeCommLatency(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): number {
  // 各策略通信量
  const tpComm = calculateTPCommVolumeDecode(model, inference, parallelism.tp);
  const ppComm = calculatePPCommVolumeDecode(model, inference, parallelism.pp);
  const epComm = calculateEPCommVolumeDecode(model, inference, parallelism.ep);
  const spComm = calculateSPCommVolumeDecode(model, inference, parallelism.sp);

  // 带宽选择
  const totalChipsPerNode = hardware.node.chips_per_node;
  const totalChips = parallelism.dp * parallelism.tp * parallelism.pp * parallelism.ep;
  const numNodes = Math.ceil(totalChips / totalChipsPerNode);

  const tpBandwidth = hardware.node.intra_node_bandwidth_gbps;
  const tpLatencyUs = hardware.node.intra_node_latency_us;
  const ppBandwidth = numNodes > 1 ? hardware.cluster.inter_node_bandwidth_gbps : tpBandwidth;
  const ppLatencyUs = numNodes > 1 ? hardware.cluster.inter_node_latency_us : tpLatencyUs;

  const tpLatency = estimateCommLatency(tpComm, tpBandwidth, tpLatencyUs);
  const ppLatency = estimateCommLatency(ppComm, ppBandwidth, ppLatencyUs);
  const epLatency = estimateCommLatency(epComm, ppBandwidth, ppLatencyUs);
  const spLatency = estimateCommLatency(spComm, tpBandwidth, tpLatencyUs);

  // Decode 通信通常更少，直接求和
  return tpLatency + ppLatency + epLatency + spLatency;
}

// ============================================
// 流水线气泡
// ============================================

/**
 * 计算流水线气泡比
 *
 * 气泡比 = (PP - 1) / (num_micro_batches + PP - 1)
 */
export function calculatePPBubbleRatio(
  ppSize: number,
  numMicroBatches: number
): number {
  if (ppSize <= 1) return 0;

  // 标准流水线气泡公式
  const bubbleRatio = (ppSize - 1) / (numMicroBatches + ppSize - 1);

  return bubbleRatio;
}

/**
 * 计算流水线效率
 */
export function calculatePPEfficiency(
  ppSize: number,
  numMicroBatches: number
): number {
  const bubbleRatio = calculatePPBubbleRatio(ppSize, numMicroBatches);
  return 1 - bubbleRatio;
}

// ============================================
// 瓶颈识别 (Roofline 模型)
// ============================================

/**
 * 计算硬件临界点 (Ridge Point)
 * Ridge Point = Peak Compute / Peak Memory BW
 */
function calculateRidgePoint(hardware: HardwareConfig): number {
  // 优先使用 peak_tflops，否则使用 compute_tflops_fp16
  const peakTflops = hardware.chip.peak_tflops ?? hardware.chip.compute_tflops_fp16;
  const memBwGBps = hardware.chip.memory_bandwidth_gbps;

  // 防止除零和无效值
  if (!peakTflops || !memBwGBps || memBwGBps <= 0) {
    // 返回默认值：H100 的 ridge point 约为 312 ops/byte
    return 312;
  }

  // Ridge Point = TFLOPs / (GB/s) = (TFLOPs * 1000) / (TB/s) = ops/byte
  // 简化: peakTflops (TFLOPs) / (memBwGBps / 1000) (TB/s) = peakTflops * 1000 / memBwGBps
  return (peakTflops * 1000) / memBwGBps;
}

/**
 * 计算 Prefill 阶段算术强度
 * AI = FLOPs / Bytes
 * Prefill: FLOPs ≈ 2 * Params * SeqLen * Batch
 * Bytes ≈ Params * dtype_bytes (模型权重读取一次)
 */
function calculatePrefillArithmeticIntensity(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy
): number {
  const dtypeBytes = model.dtype === 'fp32' ? 4 : 2;
  const params = model.total_params || estimateModelParams(model);

  // FLOPs: 2 * Params * SeqLen * Batch / TP (TP切分后单卡计算量)
  const flops = (2 * params * inference.input_seq_length * inference.batch_size) / parallelism.tp;

  // Bytes: 模型权重 / TP + 激活值
  // 激活值: batch * seq * hidden * dtype_bytes * 2 (输入输出)
  const modelBytes = (params * dtypeBytes) / parallelism.tp;
  const activationBytes = inference.batch_size * inference.input_seq_length * model.hidden_size * dtypeBytes * 2;
  const totalBytes = modelBytes + activationBytes;

  return flops / totalBytes;
}

/**
 * 计算 Decode 阶段算术强度
 * Decode: FLOPs ≈ 2 * Params (每token)
 * Bytes ≈ Params * dtype + KV_cache_per_token
 */
function calculateDecodeArithmeticIntensity(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  contextLen: number
): number {
  const dtypeBytes = model.dtype === 'fp32' ? 4 : 2;
  const params = model.total_params || estimateModelParams(model);

  // FLOPs per token: 2 * Params * Batch / TP
  const flopsPerToken = (2 * params * inference.batch_size) / parallelism.tp;

  // Bytes per token: 模型权重 + KV cache
  const modelBytes = (params * dtypeBytes) / parallelism.tp;
  // KV cache per token: 2 * num_layers * hidden_size * context_len * batch * dtype / TP
  const kvBytesPerToken = (2 * model.num_layers * model.hidden_size * contextLen * inference.batch_size * dtypeBytes) / parallelism.tp;
  const totalBytes = modelBytes + kvBytesPerToken;

  return flopsPerToken / totalBytes;
}

/**
 * 估算模型参数量
 */
function estimateModelParams(model: LLMModelConfig): number {
  // 简化估算: 12 * num_layers * hidden_size^2
  return 12 * model.num_layers * model.hidden_size * model.hidden_size;
}

/**
 * 分析单阶段瓶颈 (Roofline 模型)
 */
function analyzePhaseBottleneck(
  phase: 'prefill' | 'decode',
  arithmeticIntensity: number,
  ridgePoint: number,
  computeLatency: number,
  memoryLatency: number,
  commLatency: number,
  actualLatency: number,
  hardware: HardwareConfig,
  utilization: number
): PhaseBottleneckAnalysis {
  // 判断瓶颈类型
  const aiRatio = arithmeticIntensity / ridgePoint;
  let boundType: 'compute' | 'memory' | 'balanced';

  if (aiRatio < 0.8) {
    boundType = 'memory';
  } else if (aiRatio > 1.2) {
    boundType = 'compute';
  } else {
    boundType = 'balanced';
  }

  // 计算延迟占比
  const totalLatencyComponents = computeLatency + memoryLatency + commLatency;
  const computeRatio = totalLatencyComponents > 0 ? computeLatency / totalLatencyComponents : 0;
  const memoryRatio = totalLatencyComponents > 0 ? memoryLatency / totalLatencyComponents : 0;
  const commRatio = totalLatencyComponents > 0 ? commLatency / totalLatencyComponents : 0;

  // 计算理论最优延迟
  const theoreticalLatency = Math.max(computeLatency, memoryLatency);

  // 效率损失原因
  const efficiencyLoss: string[] = [];

  if (commRatio > 0.2) {
    efficiencyLoss.push(`通信开销 ${(commRatio * 100).toFixed(0)}%`);
  }
  if (utilization < 0.5) {
    efficiencyLoss.push(`硬件利用率低 ${(utilization * 100).toFixed(0)}%`);
  }
  if (boundType === 'memory' && phase === 'prefill') {
    efficiencyLoss.push('Prefill 意外进入 memory-bound (batch 过小?)');
  }
  if (boundType === 'compute' && phase === 'decode') {
    efficiencyLoss.push('Decode 意外进入 compute-bound (batch 过大?)');
  }

  return {
    phase,
    arithmetic_intensity: arithmeticIntensity,
    hardware_ridge_point: ridgePoint,
    bound_type: boundType,
    compute_ratio: computeRatio,
    memory_ratio: memoryRatio,
    comm_ratio: commRatio,
    utilization,
    theoretical_latency_ms: theoreticalLatency,
    actual_latency_ms: actualLatency,
    efficiency_loss: efficiencyLoss,
  };
}

/**
 * 完整瓶颈分析 (Roofline 模型)
 */
export function analyzeBottleneckRoofline(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig,
  prefillCompute: number,
  prefillComm: number,
  prefillTotal: number,
  decodeCompute: number,
  decodeMemory: number,
  decodeComm: number,
  decodePerToken: number,
  mfu: number,
  mbu: number
): BottleneckAnalysis {
  const ridgePoint = calculateRidgePoint(hardware);
  const avgContextLen = inference.input_seq_length + inference.output_seq_length / 2;

  // Prefill 算术强度
  const prefillAI = calculatePrefillArithmeticIntensity(model, inference, parallelism);

  // Decode 算术强度
  const decodeAI = calculateDecodeArithmeticIntensity(model, inference, parallelism, avgContextLen);

  // 分析各阶段瓶颈
  const prefillAnalysis = analyzePhaseBottleneck(
    'prefill', prefillAI, ridgePoint,
    prefillCompute, 0, prefillComm, prefillTotal,
    hardware, mfu
  );

  const decodeAnalysis = analyzePhaseBottleneck(
    'decode', decodeAI, ridgePoint,
    decodeCompute, decodeMemory, decodeComm, decodePerToken,
    hardware, mbu
  );

  // 判断主导阶段
  const prefillTotalTime = prefillTotal;
  const decodeTotalTime = decodePerToken * inference.output_seq_length;
  const dominantPhase = prefillTotalTime > decodeTotalTime ? 'prefill' : 'decode';

  // 综合瓶颈类型
  const dominantAnalysis = dominantPhase === 'prefill' ? prefillAnalysis : decodeAnalysis;
  let overallBottleneck: BottleneckType;

  if (dominantAnalysis.comm_ratio > 0.4) {
    overallBottleneck = 'communication';
  } else if (dominantAnalysis.bound_type === 'memory') {
    overallBottleneck = 'memory';
  } else if (dominantAnalysis.bound_type === 'compute') {
    overallBottleneck = 'compute';
  } else {
    overallBottleneck = 'balanced';
  }

  // 瓶颈严重程度 (1 - 利用率)
  const severity = 1 - dominantAnalysis.utilization;

  // 优化潜力分析
  const dtypeBytes = model.dtype === 'fp32' ? 4 : 2;
  const currentAI = dominantPhase === 'prefill' ? prefillAI : decodeAI;

  // Batch scaling: 增大 batch 提升算术强度
  const batchScalingPotential = decodeAI < ridgePoint ? Math.min(ridgePoint / decodeAI, 4) : 1;

  // 量化: INT8 可减少一半内存访问
  const quantizationPotential = dtypeBytes > 1 ? dtypeBytes / 1 : 1;

  // 减少 TP: 减少通信开销
  const reduceTPPotential = dominantAnalysis.comm_ratio > 0.1 ? 1 / (1 - dominantAnalysis.comm_ratio * 0.5) : 1;

  // 生成摘要
  const phaseLabel = dominantPhase === 'prefill' ? 'Prefill' : 'Decode';
  const boundLabel = dominantAnalysis.bound_type === 'memory' ? '访存瓶颈' :
                     dominantAnalysis.bound_type === 'compute' ? '算力瓶颈' : '均衡状态';
  const summary = `${phaseLabel}阶段主导 (${(dominantPhase === 'prefill' ? prefillTotalTime : decodeTotalTime).toFixed(1)}ms)，` +
                  `${boundLabel}，算术强度 ${currentAI.toFixed(1)} ops/byte (临界点 ${ridgePoint.toFixed(0)})，` +
                  `利用率 ${(dominantAnalysis.utilization * 100).toFixed(0)}%`;

  return {
    prefill: prefillAnalysis,
    decode: decodeAnalysis,
    dominant_phase: dominantPhase,
    overall_bottleneck: overallBottleneck,
    severity,
    optimization_potential: {
      batch_scaling: {
        current_ai: currentAI,
        target_ai: ridgePoint,
        potential_speedup: batchScalingPotential,
      },
      quantization: {
        current_bytes: dtypeBytes,
        target_bytes: 1,
        potential_speedup: quantizationPotential,
      },
      reduce_tp: {
        current_comm_ratio: dominantAnalysis.comm_ratio,
        potential_speedup: reduceTPPotential,
      },
    },
    summary,
  };
}

/**
 * 识别主要瓶颈 (兼容旧接口)
 */
export function identifyBottleneck(
  computeLatency: number,
  memoryLatency: number,
  commLatency: number,
  bubbleRatio: number
): { type: BottleneckType; details: string } {
  // 气泡影响: 如果气泡比 > 20%，认为是瓶颈
  const bubbleImpact = bubbleRatio > 0.2;

  // 计算瓶颈得分
  const scores = [
    { type: 'compute' as BottleneckType, score: computeLatency, details: `计算延迟 ${computeLatency.toFixed(2)}ms` },
    { type: 'memory' as BottleneckType, score: memoryLatency, details: `访存延迟 ${memoryLatency.toFixed(2)}ms` },
    { type: 'communication' as BottleneckType, score: commLatency, details: `通信延迟 ${commLatency.toFixed(2)}ms` },
  ];

  // 排序找最大
  scores.sort((a, b) => b.score - a.score);
  const primary = scores[0];

  // 如果气泡影响显著，也要报告
  if (bubbleImpact && primary.type !== 'pipeline_bubble') {
    return {
      type: primary.type,
      details: `${primary.details}，流水线气泡比 ${(bubbleRatio * 100).toFixed(1)}%`,
    };
  }

  // 如果气泡是主要问题
  if (bubbleRatio > 0.3) {
    return {
      type: 'pipeline_bubble',
      details: `流水线气泡比过高 ${(bubbleRatio * 100).toFixed(1)}%，建议增加 micro-batch 数量`,
    };
  }

  return primary;
}

// ============================================
// 分位数估算
// ============================================

/**
 * 估算延迟分位数
 *
 * 业界基准 (MLPerf Inference v5.0):
 * - TTFT P99 ≤ 450ms (Server scenario)
 * - TPOT P99 ≤ 40ms (Server scenario)
 *
 * 分位数倍率基于实际测量数据的经验值:
 * - P50 ≈ 基准延迟 × 1.0 (中位数接近理论值)
 * - P90 ≈ 基准延迟 × 1.3 (网络/调度抖动)
 * - P99 ≈ 基准延迟 × 1.8 (尾部延迟，含 GC/页面缺失等)
 *
 * 影响因素:
 * - 网络通信: TP/PP 越高，通信抖动越大
 * - 负载变化: 高负载时排队延迟增加
 * - 批次变化: continuous batching 带来请求级差异
 *
 * 参考来源:
 * - Meta LLaMa 3 Benchmark: https://ai.meta.com/blog/meta-llama-3/
 * - Databricks MosaicML: https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices
 */
export function estimateLatencyPercentiles(
  baseLatencyMs: number,
  parallelism: ParallelismStrategy,
  isDecodePhase: boolean = false
): LatencyPercentiles {
  // 基础倍率
  let p90Multiplier = 1.3;
  let p99Multiplier = 1.8;

  // TP/PP 通信引入额外抖动
  if (parallelism.tp > 1) {
    const tpFactor = 1 + (parallelism.tp - 1) * 0.02; // 每增加 1 TP 增加 2%
    p90Multiplier *= tpFactor;
    p99Multiplier *= tpFactor;
  }

  if (parallelism.pp > 1) {
    const ppFactor = 1 + (parallelism.pp - 1) * 0.03; // PP 同步更敏感
    p90Multiplier *= ppFactor;
    p99Multiplier *= ppFactor;
  }

  // Decode 阶段抖动更稳定 (批次更小，通信更少)
  if (isDecodePhase) {
    p90Multiplier *= 0.9;
    p99Multiplier *= 0.85;
  }

  return {
    p50: baseLatencyMs,
    p90: baseLatencyMs * p90Multiplier,
    p99: baseLatencyMs * p99Multiplier,
  };
}

// ============================================
// 综合延迟分析
// ============================================

/**
 * 完整延迟分析
 */
export function analyzeLatency(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): LatencyAnalysis {
  const numMicroBatches = inference.num_micro_batches ?? Math.max(parallelism.pp, 4);

  // ===== Prefill 阶段 =====
  const prefillCompute = estimatePrefillComputeLatency(model, inference, parallelism, hardware);
  const prefillComm = estimatePrefillCommLatency(model, inference, parallelism, hardware);

  // 流水线气泡
  const bubbleRatio = calculatePPBubbleRatio(parallelism.pp, numMicroBatches);

  // Prefill 总延迟 (考虑气泡)
  // 计算和通信部分重叠，取 max；然后考虑气泡损失
  const prefillIdeal = Math.max(prefillCompute, prefillComm);
  const prefillTotal = prefillIdeal / (1 - bubbleRatio);

  // ===== Decode 阶段 =====
  // 使用平均 context 长度
  const avgContextLen = inference.input_seq_length + inference.output_seq_length / 2;

  const decodeCompute = estimateDecodeComputeLatency(
    model, inference, parallelism, hardware, avgContextLen
  );
  const decodeMemory = estimateDecodeMemoryLatency(
    model, inference, parallelism, hardware, avgContextLen
  );
  const decodeComm = estimateDecodeCommLatency(model, inference, parallelism, hardware);

  // Decode 是 memory-bound，取计算和访存的最大值，加上通信
  const decodePerToken = Math.max(decodeCompute, decodeMemory) + decodeComm;

  // ===== 端到端延迟 =====
  const e2eLatency = prefillTotal + decodePerToken * inference.output_seq_length;

  // ===== 瓶颈识别 =====
  // Prefill 瓶颈
  const prefillBottleneck = identifyBottleneck(prefillCompute, 0, prefillComm, bubbleRatio);

  // Decode 瓶颈
  const decodeBottleneck = identifyBottleneck(decodeCompute, decodeMemory, decodeComm, 0);

  // 综合瓶颈 (取更严重的)
  let bottleneckType: BottleneckType;
  let bottleneckDetails: string;

  if (prefillTotal > decodePerToken * inference.output_seq_length) {
    // Prefill 占主导
    bottleneckType = prefillBottleneck.type;
    bottleneckDetails = `Prefill 阶段: ${prefillBottleneck.details}`;
  } else {
    // Decode 占主导
    bottleneckType = decodeBottleneck.type;
    bottleneckDetails = `Decode 阶段: ${decodeBottleneck.details}`;
  }

  // ===== 分位数估算 =====
  const ttftPercentiles = estimateLatencyPercentiles(prefillTotal, parallelism, false);
  const tpotPercentiles = estimateLatencyPercentiles(decodePerToken, parallelism, true);

  return {
    prefill_compute_latency_ms: prefillCompute,
    prefill_comm_latency_ms: prefillComm,
    prefill_total_latency_ms: prefillTotal,
    decode_compute_latency_ms: decodeCompute,
    decode_memory_latency_ms: decodeMemory,
    decode_comm_latency_ms: decodeComm,
    decode_per_token_latency_ms: decodePerToken,
    end_to_end_latency_ms: e2eLatency,
    pipeline_bubble_ratio: bubbleRatio,
    bottleneck_type: bottleneckType,
    bottleneck_details: bottleneckDetails,
    ttft_percentiles: ttftPercentiles,
    tpot_percentiles: tpotPercentiles,
  };
}

// ============================================
// 吞吐量估算
// ============================================

/**
 * 估算 token 吞吐量
 */
export function estimateTokenThroughput(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): number {
  const latency = analyzeLatency(model, inference, parallelism, hardware);

  // 每请求输出 token 数
  const tokensPerRequest = inference.output_seq_length;

  // 每请求延迟
  const latencyPerRequest = latency.end_to_end_latency_ms / 1000; // seconds

  // 单流吞吐量 (考虑 batch)
  const tokensPerSecond = (inference.batch_size * tokensPerRequest) / latencyPerRequest;

  return tokensPerSecond;
}

/**
 * 估算请求吞吐量
 */
export function estimateRequestThroughput(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): number {
  const tokenThroughput = estimateTokenThroughput(model, inference, parallelism, hardware);
  return tokenThroughput / inference.output_seq_length;
}

/**
 * 估算 Decode 阶段 MFU (Model FLOPs Utilization)
 *
 * 注意: Decode 阶段是 memory-bound，MFU 通常很低 (5-15%)
 */
export function estimateMFU(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): number {
  const tokenThroughput = estimateTokenThroughput(model, inference, parallelism, hardware);

  // 每 token 的理论 FLOPs
  const flopsPerToken = calculateDecodeFlopsPerToken(
    model,
    inference,
    inference.input_seq_length + inference.output_seq_length / 2
  );

  // 实际算力使用
  const actualFlopsPerSecond = tokenThroughput * flopsPerToken;

  // 理论峰值算力 (所有芯片)
  const totalChips = parallelism.dp * parallelism.tp * parallelism.pp * parallelism.ep;
  const theoreticalFlopsPerSecond = hardware.chip.compute_tflops_fp16 * 1e12 * totalChips;

  return actualFlopsPerSecond / theoreticalFlopsPerSecond;
}

/**
 * 估算 Prefill 阶段 MFU (Model FLOPs Utilization)
 *
 * Prefill 阶段是 compute-bound，MFU 通常较高 (30-50%)
 * 此函数用于与仿真结果对比 (仿真计算的是 Prefill MFU)
 */
export function estimatePrefillMFU(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): number {
  // Prefill 总 FLOPs
  const prefillFlops = calculatePrefillFlops(model, inference);

  // Prefill 时间
  const prefillTimeMs = estimatePrefillComputeLatency(model, inference, parallelism, hardware);
  const prefillTimeS = prefillTimeMs / 1000;

  // 实际算力 (TFLOPs)
  const achievedTflops = (prefillFlops / 1e12) / prefillTimeS;

  // 单 DP 副本的峰值算力 (tp * pp 个芯片)
  const chipsPerReplica = parallelism.tp * parallelism.pp;
  const peakTflops = hardware.chip.compute_tflops_fp16 * chipsPerReplica;

  return achievedTflops / peakTflops;
}

/**
 * 估算理论最大吞吐量
 */
export function estimateTheoreticalMaxThroughput(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): number {
  // 假设 100% MFU
  const totalChips = parallelism.dp * parallelism.tp * parallelism.pp * parallelism.ep;
  const theoreticalFlopsPerSecond = hardware.chip.compute_tflops_fp16 * 1e12 * totalChips;

  const flopsPerToken = calculateDecodeFlopsPerToken(
    model,
    inference,
    inference.input_seq_length + inference.output_seq_length / 2
  );

  return theoreticalFlopsPerSecond / flopsPerToken;
}

/**
 * 估算 MBU (Memory Bandwidth Utilization)
 *
 * MBU = Achieved_Bandwidth / Peak_Bandwidth
 * 其中 Achieved_Bandwidth = (Model_Size + KV_Cache_Size) / TPOT
 *
 * 业界标准公式来源:
 * - Databricks: https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices
 * - NVIDIA: https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
 */
export function estimateMBU(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): number {
  const latency = analyzeLatency(model, inference, parallelism, hardware);
  const tpotSeconds = latency.decode_per_token_latency_ms / 1000;

  if (tpotSeconds <= 0) return 0;

  // Decode 每 token 需读取的数据量 (GB)
  // = 模型权重 (每 token 都要读一遍) + 当前 context 的 KV Cache
  const modelMemoryGB = calculateModelMemory(model, parallelism);

  // KV Cache 按平均 context 长度计算
  const avgContextLen = inference.input_seq_length + inference.output_seq_length / 2;
  const kvCacheGB = calculateKVCacheMemory(model, inference, parallelism);
  const kvCacheRatio = avgContextLen / inference.max_seq_length;
  const currentKVCacheGB = kvCacheGB * kvCacheRatio;

  // 实际带宽 (GB/s)
  const dataReadPerTokenGB = modelMemoryGB + currentKVCacheGB;
  const achievedBandwidthGBps = dataReadPerTokenGB / tpotSeconds;

  // 峰值带宽 (考虑所有芯片并行读取)
  // 注意: TP 并行时，每个芯片只读取自己的部分，所以总带宽是单芯片带宽
  const peakBandwidthGBps = hardware.chip.memory_bandwidth_gbps;

  // MBU
  const mbu = achievedBandwidthGBps / peakBandwidthGBps;

  // MBU 不应超过 1 (理论上)
  return Math.min(mbu, 1.0);
}

// ============================================
// 成本分析
// ============================================

/**
 * 默认芯片成本表 ($/hour)
 * 数据来源: 云服务商按需实例定价 (2025年)
 * - AWS: https://aws.amazon.com/ec2/instance-types/
 * - Azure: https://azure.microsoft.com/pricing/details/virtual-machines/
 * - GCP: https://cloud.google.com/compute/all-pricing
 */
const DEFAULT_CHIP_COSTS: Record<string, number> = {
  // NVIDIA
  'H100': 4.5,       // ~$32-35/h per 8-GPU node
  'H200': 6.0,       // 预估 (尚未普遍)
  'A100-80GB': 3.0,  // ~$24/h per 8-GPU node
  'A100-40GB': 2.5,
  'L40S': 1.5,
  'A10': 1.0,
  // AMD
  'MI300X': 4.0,     // 预估
  'MI250X': 2.5,
  // 国产
  'Ascend-910B': 2.0, // 预估
};

/**
 * 获取芯片成本
 */
function getChipCost(hardware: HardwareConfig): number {
  // 优先使用配置的成本
  if (hardware.chip.cost_per_hour !== undefined) {
    return hardware.chip.cost_per_hour;
  }

  // 查找默认成本
  const chipType = hardware.chip.chip_type;
  for (const [name, cost] of Object.entries(DEFAULT_CHIP_COSTS)) {
    if (chipType.toLowerCase().includes(name.toLowerCase())) {
      return cost;
    }
  }

  // 默认成本 (基于算力估算: ~$0.01 per TFLOP-hour)
  return hardware.chip.compute_tflops_fp16 * 0.01;
}

/**
 * 估算成本分析
 *
 * 业界成本计算标准:
 * - $/M tokens = (Hardware_Cost_Per_Hour × 1e6) / (Tokens_Per_Second × 3600)
 * - 输入/输出成本比例通常为 1:3 ~ 1:5 (因为 Prefill 计算密度高但快，Decode 慢)
 *
 * 参考:
 * - OpenAI Pricing: https://openai.com/pricing
 * - Anthropic Pricing: https://www.anthropic.com/pricing
 * - Together AI: https://www.together.ai/pricing
 */
export function estimateCost(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): CostAnalysis {
  // 计算总芯片数
  const totalChips = parallelism.dp * parallelism.tp * parallelism.pp * parallelism.ep;

  // 每芯片成本
  const chipCostPerHour = getChipCost(hardware);

  // 总硬件成本
  const totalCostPerHour = chipCostPerHour * totalChips;

  // 获取吞吐量
  const tokenThroughput = estimateTokenThroughput(model, inference, parallelism, hardware);

  if (tokenThroughput <= 0) {
    return {
      hardware_cost_per_hour: chipCostPerHour,
      total_hardware_cost_per_hour: totalCostPerHour,
      cost_per_million_tokens: Infinity,
      input_cost_per_million_tokens: Infinity,
      output_cost_per_million_tokens: Infinity,
      tokens_per_dollar: 0,
    };
  }

  // 每百万 token 成本 (综合)
  // $/M tokens = ($/hour × 1e6) / (tokens/s × 3600s/hour)
  const tokensPerHour = tokenThroughput * 3600;
  const costPerMillionTokens = (totalCostPerHour * 1e6) / tokensPerHour;

  // 输入/输出成本分解
  // 通常输出 token 比输入 token 成本高 3-4 倍
  // 原因: Prefill (输入) 是 compute-bound 且批量处理
  //       Decode (输出) 是 memory-bound 且逐 token 生成
  const latency = analyzeLatency(model, inference, parallelism, hardware);
  const prefillTime = latency.prefill_total_latency_ms;
  const decodeTime = latency.decode_per_token_latency_ms * inference.output_seq_length;
  const totalTime = prefillTime + decodeTime;

  // 按时间比例分配成本
  const inputRatio = prefillTime / totalTime;
  const outputRatio = decodeTime / totalTime;

  // 输入成本 ($/M input tokens)
  const inputCostPerMillion = (totalCostPerHour * 1e6 * inputRatio) /
                               (inference.batch_size * inference.input_seq_length * 3600 / (totalTime / 1000));

  // 输出成本 ($/M output tokens)
  const outputCostPerMillion = (totalCostPerHour * 1e6 * outputRatio) /
                                (inference.batch_size * inference.output_seq_length * 3600 / (totalTime / 1000));

  // Token/美元效率
  const tokensPerDollar = tokensPerHour / totalCostPerHour;

  return {
    hardware_cost_per_hour: chipCostPerHour,
    total_hardware_cost_per_hour: totalCostPerHour,
    cost_per_million_tokens: costPerMillionTokens,
    input_cost_per_million_tokens: inputCostPerMillion,
    output_cost_per_million_tokens: outputCostPerMillion,
    tokens_per_dollar: tokensPerDollar,
  };
}
