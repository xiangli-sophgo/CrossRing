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
  BottleneckType,
  getBytesPerElement,
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
// 计算延迟估算
// ============================================

/**
 * 估算 Prefill 阶段计算延迟
 *
 * 计算延迟 = FLOPs / (峰值算力 × 利用率 × 并行数)
 */
export function estimatePrefillComputeLatency(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig,
  mfuEstimate: number = 0.5 // 模型算力利用率估计
): number {
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
  const computeTimeS = flopsPerChip / (flopsPerSecond * mfuEstimate);

  return computeTimeS * 1000; // ms
}

/**
 * 估算 Decode 阶段单 token 计算延迟
 */
export function estimateDecodeComputeLatency(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig,
  contextLength: number,
  mfuEstimate: number = 0.3 // Decode 阶段 MFU 通常较低
): number {
  const tokenFlops = calculateDecodeFlopsPerToken(model, inference, contextLength);

  const chipTflops = hardware.chip.compute_tflops_fp16;
  const flopsPerSecond = chipTflops * 1e12;
  const effectiveParallelism = parallelism.tp * parallelism.pp;
  const flopsPerChip = tokenFlops / effectiveParallelism;
  const computeTimeS = flopsPerChip / (flopsPerSecond * mfuEstimate);

  return computeTimeS * 1000; // ms
}

// ============================================
// 访存延迟估算
// ============================================

/**
 * 估算访存延迟 (Memory Bandwidth Bound)
 *
 * Decode 阶段通常是 memory-bound
 * 延迟 = 需读取的数据量 / 显存带宽
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

  // 显存带宽
  const bandwidthGBps = hardware.chip.memory_bandwidth_gbps;

  // 访存时间
  const memoryTimeS = dataToReadGB / bandwidthGBps;

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
  const bandwidthGBps = hardware.chip.memory_bandwidth_gbps;
  const memoryTimeS = dataToReadGB / bandwidthGBps;

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
// 瓶颈识别
// ============================================

/**
 * 识别主要瓶颈
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

  return {
    prefill_compute_latency_ms: prefillCompute,
    prefill_comm_latency_ms: prefillComm,
    prefill_total_latency_ms: prefillTotal,
    decode_compute_latency_ms: decodeCompute,
    decode_comm_latency_ms: decodeComm,
    decode_per_token_latency_ms: decodePerToken,
    end_to_end_latency_ms: e2eLatency,
    pipeline_bubble_ratio: bubbleRatio,
    bottleneck_type: bottleneckType,
    bottleneck_details: bottleneckDetails,
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
 * 估算 MFU (Model FLOPs Utilization)
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
