/**
 * LLM 部署分析系统 - 方案分析器
 *
 * 单方案综合分析：资源、延迟、吞吐、评分
 */

import {
  LLMModelConfig,
  InferenceConfig,
  ParallelismStrategy,
  HardwareConfig,
  DeploymentPlan,
  PlanAnalysisResult,
  MemoryAnalysis,
  CommunicationAnalysis,
  LatencyAnalysis,
  ThroughputAnalysis,
  UtilizationAnalysis,
  OverallScore,
  OptimizationSuggestion,
} from './types';
import { analyzeMemory } from './modelCalculator';
import { analyzeCommunication } from './commCalculator';
import {
  analyzeLatency,
  estimateTokenThroughput,
  estimateRequestThroughput,
  estimateMFU,
  estimateTheoreticalMaxThroughput,
} from './latencyEstimator';

// ============================================
// 方案可行性检查
// ============================================

/**
 * 检查方案是否可行
 */
export function checkFeasibility(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): { isFeasible: boolean; reason?: string } {
  // 1. TP 必须整除 attention heads
  if (model.num_attention_heads % parallelism.tp !== 0) {
    return {
      isFeasible: false,
      reason: `TP(${parallelism.tp}) 无法整除 attention_heads(${model.num_attention_heads})`,
    };
  }

  // 2. TP 必须整除 KV heads (对于 GQA)
  if (model.num_kv_heads % parallelism.tp !== 0) {
    return {
      isFeasible: false,
      reason: `TP(${parallelism.tp}) 无法整除 kv_heads(${model.num_kv_heads})`,
    };
  }

  // 3. PP 必须整除 layers
  if (model.num_layers % parallelism.pp !== 0) {
    return {
      isFeasible: false,
      reason: `PP(${parallelism.pp}) 无法整除 layers(${model.num_layers})`,
    };
  }

  // 4. MoE: EP 必须整除专家数
  if (model.model_type === 'moe' && model.moe_config) {
    if (model.moe_config.num_experts % parallelism.ep !== 0) {
      return {
        isFeasible: false,
        reason: `EP(${parallelism.ep}) 无法整除 experts(${model.moe_config.num_experts})`,
      };
    }
  }

  // 5. 显存检查
  const memoryAnalysis = analyzeMemory(
    model,
    inference,
    parallelism,
    hardware.chip.memory_gb
  );

  if (!memoryAnalysis.is_memory_sufficient) {
    return {
      isFeasible: false,
      reason: `显存不足: 需要 ${memoryAnalysis.total_per_chip_gb.toFixed(1)}GB，可用 ${hardware.chip.memory_gb}GB`,
    };
  }

  // 6. 芯片数检查
  const totalChips = parallelism.dp * parallelism.tp * parallelism.pp * parallelism.ep;
  const maxChips = hardware.node.chips_per_node * hardware.cluster.num_nodes;
  if (totalChips > maxChips) {
    return {
      isFeasible: false,
      reason: `需要 ${totalChips} 芯片，但集群最多 ${maxChips} 芯片`,
    };
  }

  // 7. TP 应在节点内 (推荐但不强制)
  if (parallelism.tp > hardware.node.chips_per_node) {
    // 警告但不阻止
  }

  return { isFeasible: true };
}

// ============================================
// 吞吐量分析
// ============================================

/**
 * 分析吞吐量
 */
export function analyzeThroughput(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): ThroughputAnalysis {
  const tokensPerSecond = estimateTokenThroughput(model, inference, parallelism, hardware);
  const requestsPerSecond = estimateRequestThroughput(model, inference, parallelism, hardware);
  const mfu = estimateMFU(model, inference, parallelism, hardware);
  const theoreticalMax = estimateTheoreticalMaxThroughput(model, inference, parallelism, hardware);

  return {
    tokens_per_second: tokensPerSecond,
    requests_per_second: requestsPerSecond,
    model_flops_utilization: mfu,
    theoretical_max_throughput: theoreticalMax,
  };
}

// ============================================
// 资源利用率分析
// ============================================

/**
 * 分析资源利用率
 */
export function analyzeUtilization(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig,
  memory: MemoryAnalysis,
  latency: LatencyAnalysis,
  throughput: ThroughputAnalysis
): UtilizationAnalysis {
  // 计算利用率 = MFU
  const computeUtilization = throughput.model_flops_utilization;

  // 显存利用率
  const memoryUtilization = memory.memory_utilization;

  // 网络利用率 (基于通信延迟占比)
  const commRatio = latency.prefill_comm_latency_ms / latency.prefill_total_latency_ms;
  const networkUtilization = Math.min(commRatio * 2, 1.0); // 简化估算

  // 负载均衡得分
  // TP/PP/EP 均匀切分时得分高
  const loadBalanceScore = calculateLoadBalanceScore(model, parallelism);

  return {
    compute_utilization: computeUtilization,
    memory_utilization: memoryUtilization,
    network_utilization: networkUtilization,
    load_balance_score: loadBalanceScore,
  };
}

/**
 * 计算负载均衡得分
 */
function calculateLoadBalanceScore(
  model: LLMModelConfig,
  parallelism: ParallelismStrategy
): number {
  let score = 1.0;

  // TP 均匀性
  const headsPerTP = model.num_attention_heads / parallelism.tp;
  if (headsPerTP !== Math.floor(headsPerTP)) {
    score *= 0.9;
  }

  // PP 均匀性
  const layersPerPP = model.num_layers / parallelism.pp;
  if (layersPerPP !== Math.floor(layersPerPP)) {
    score *= 0.9;
  }

  // MoE: EP 均匀性
  if (model.model_type === 'moe' && model.moe_config && parallelism.ep > 1) {
    const expertsPerEP = model.moe_config.num_experts / parallelism.ep;
    if (expertsPerEP !== Math.floor(expertsPerEP)) {
      score *= 0.85;
    }
  }

  return score;
}

// ============================================
// 评分计算
// ============================================

/**
 * 计算综合评分
 */
export function calculateOverallScore(
  memory: MemoryAnalysis,
  latency: LatencyAnalysis,
  throughput: ThroughputAnalysis,
  utilization: UtilizationAnalysis,
  parallelism: ParallelismStrategy
): OverallScore {
  // 延迟评分 (TTFT < 100ms 得满分，> 1000ms 得 0 分)
  const ttft = latency.prefill_total_latency_ms;
  const latencyScore = Math.max(0, Math.min(100, 100 - (ttft - 100) / 9));

  // 吞吐评分 (MFU > 50% 得满分)
  const mfu = throughput.model_flops_utilization;
  const throughputScore = Math.min(100, mfu * 200);

  // 效率评分 (综合利用率)
  const avgUtilization =
    (utilization.compute_utilization + utilization.memory_utilization) / 2;
  const efficiencyScore = avgUtilization * 100;

  // 均衡评分
  const balanceScore = utilization.load_balance_score * 100;

  // 综合评分 (加权平均)
  const weights = {
    latency: 0.3,
    throughput: 0.35,
    efficiency: 0.2,
    balance: 0.15,
  };

  const overallScore =
    latencyScore * weights.latency +
    throughputScore * weights.throughput +
    efficiencyScore * weights.efficiency +
    balanceScore * weights.balance;

  return {
    latency_score: latencyScore,
    throughput_score: throughputScore,
    efficiency_score: efficiencyScore,
    balance_score: balanceScore,
    overall_score: overallScore,
  };
}

// ============================================
// 优化建议生成
// ============================================

/**
 * 生成优化建议
 */
export function generateSuggestions(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig,
  memory: MemoryAnalysis,
  latency: LatencyAnalysis,
  throughput: ThroughputAnalysis
): OptimizationSuggestion[] {
  const suggestions: OptimizationSuggestion[] = [];

  // 1. TP 过大导致通信开销
  if (parallelism.tp > 4 && latency.bottleneck_type === 'communication') {
    suggestions.push({
      type: 'reduce_tp',
      description: `当前 TP=${parallelism.tp} 通信开销较大，建议减小到 ${Math.max(2, parallelism.tp / 2)}`,
      expected_improvement: '通信延迟可降低 30-50%',
      priority: 1,
    });
  }

  // 2. PP 气泡过高
  if (latency.pipeline_bubble_ratio > 0.25) {
    const suggestedMicroBatches = Math.ceil(parallelism.pp * 4);
    suggestions.push({
      type: 'increase_pp',
      description: `流水线气泡比 ${(latency.pipeline_bubble_ratio * 100).toFixed(1)}% 过高，建议增加 micro-batch 数量到 ${suggestedMicroBatches}`,
      expected_improvement: `气泡比可降至 ${(100 * (parallelism.pp - 1) / (suggestedMicroBatches + parallelism.pp - 1)).toFixed(1)}%`,
      priority: 2,
    });
  }

  // 3. batch 太小导致吞吐低
  if (inference.batch_size < 8 && throughput.model_flops_utilization < 0.3) {
    suggestions.push({
      type: 'increase_batch',
      description: `当前 batch_size=${inference.batch_size} 导致计算利用率低，建议增加到 ${Math.min(32, inference.batch_size * 4)}`,
      expected_improvement: '吞吐量可提升 2-4 倍',
      priority: 2,
    });
  }

  // 4. 显存接近满载
  if (memory.memory_utilization > 0.9) {
    suggestions.push({
      type: 'reduce_seq',
      description: `显存利用率 ${(memory.memory_utilization * 100).toFixed(1)}% 接近上限，建议减小 max_seq_length 或增加 TP/PP`,
      expected_improvement: '避免 OOM 风险',
      priority: 1,
    });
  }

  // 5. MoE 专家不均衡
  if (model.model_type === 'moe' && parallelism.ep === 1 && model.moe_config) {
    if (model.moe_config.num_experts >= 8) {
      suggestions.push({
        type: 'other',
        description: `MoE 模型有 ${model.moe_config.num_experts} 个专家，建议使用 EP=${Math.min(8, model.moe_config.num_experts)} 分布式部署`,
        expected_improvement: '显存占用可降低，专家负载更均衡',
        priority: 3,
      });
    }
  }

  // 排序
  suggestions.sort((a, b) => a.priority - b.priority);

  return suggestions;
}

// ============================================
// 完整方案分析
// ============================================

/**
 * 分析单个部署方案
 */
export function analyzePlan(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig,
  planId?: string
): PlanAnalysisResult {
  // 生成方案 ID
  const id = planId ?? `dp${parallelism.dp}_tp${parallelism.tp}_pp${parallelism.pp}_ep${parallelism.ep}_sp${parallelism.sp}`;

  // 计算总芯片数
  const totalChips = parallelism.dp * parallelism.tp * parallelism.pp * parallelism.ep;

  // 可行性检查
  const feasibility = checkFeasibility(model, inference, parallelism, hardware);

  // 如果不可行，返回部分结果
  if (!feasibility.isFeasible) {
    return createInfeasibleResult(id, parallelism, totalChips, feasibility.reason!);
  }

  // 显存分析
  const memory = analyzeMemory(model, inference, parallelism, hardware.chip.memory_gb);

  // 通信分析
  const communication = analyzeCommunication(model, inference, parallelism, hardware);

  // 延迟分析
  const latency = analyzeLatency(model, inference, parallelism, hardware);

  // 吞吐分析
  const throughputAnalysis = analyzeThroughput(model, inference, parallelism, hardware);

  // 利用率分析
  const utilization = analyzeUtilization(
    model, inference, parallelism, hardware,
    memory, latency, throughputAnalysis
  );

  // 评分
  const score = calculateOverallScore(memory, latency, throughputAnalysis, utilization, parallelism);

  // 优化建议
  const suggestions = generateSuggestions(
    model, inference, parallelism, hardware,
    memory, latency, throughputAnalysis
  );

  return {
    plan: {
      plan_id: id,
      parallelism,
      total_chips: totalChips,
    },
    memory,
    communication,
    latency,
    throughput: throughputAnalysis,
    utilization,
    score,
    suggestions,
    is_feasible: true,
  };
}

/**
 * 创建不可行方案的结果
 */
function createInfeasibleResult(
  planId: string,
  parallelism: ParallelismStrategy,
  totalChips: number,
  reason: string
): PlanAnalysisResult {
  const zeroMemory: MemoryAnalysis = {
    model_memory_gb: 0,
    kv_cache_memory_gb: 0,
    activation_memory_gb: 0,
    overhead_gb: 0,
    total_per_chip_gb: 0,
    is_memory_sufficient: false,
    memory_utilization: 0,
  };

  const zeroComm: CommunicationAnalysis = {
    tp_comm_volume_gb: 0,
    pp_comm_volume_gb: 0,
    ep_comm_volume_gb: 0,
    sp_comm_volume_gb: 0,
    total_comm_volume_gb: 0,
    tp_comm_latency_ms: 0,
    pp_comm_latency_ms: 0,
  };

  const zeroLatency: LatencyAnalysis = {
    prefill_compute_latency_ms: 0,
    prefill_comm_latency_ms: 0,
    prefill_total_latency_ms: Infinity,
    decode_compute_latency_ms: 0,
    decode_comm_latency_ms: 0,
    decode_per_token_latency_ms: Infinity,
    end_to_end_latency_ms: Infinity,
    pipeline_bubble_ratio: 0,
    bottleneck_type: 'compute',
    bottleneck_details: reason,
  };

  const zeroThroughput: ThroughputAnalysis = {
    tokens_per_second: 0,
    requests_per_second: 0,
    model_flops_utilization: 0,
    theoretical_max_throughput: 0,
  };

  const zeroUtilization: UtilizationAnalysis = {
    compute_utilization: 0,
    memory_utilization: 0,
    network_utilization: 0,
    load_balance_score: 0,
  };

  const zeroScore: OverallScore = {
    latency_score: 0,
    throughput_score: 0,
    efficiency_score: 0,
    balance_score: 0,
    overall_score: 0,
  };

  return {
    plan: {
      plan_id: planId,
      parallelism,
      total_chips: totalChips,
    },
    memory: zeroMemory,
    communication: zeroComm,
    latency: zeroLatency,
    throughput: zeroThroughput,
    utilization: zeroUtilization,
    score: zeroScore,
    suggestions: [],
    is_feasible: false,
    infeasibility_reason: reason,
  };
}

// ============================================
// 快速分析接口
// ============================================

/**
 * 快速分析 - 只返回关键指标
 */
export function quickAnalyze(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): {
  isFeasible: boolean;
  ttft?: number;
  tpot?: number;
  throughput?: number;
  memoryUtilization?: number;
  score?: number;
  reason?: string;
} {
  const feasibility = checkFeasibility(model, inference, parallelism, hardware);

  if (!feasibility.isFeasible) {
    return { isFeasible: false, reason: feasibility.reason };
  }

  const memory = analyzeMemory(model, inference, parallelism, hardware.chip.memory_gb);
  const latency = analyzeLatency(model, inference, parallelism, hardware);
  const throughputAnalysis = analyzeThroughput(model, inference, parallelism, hardware);
  const utilization = analyzeUtilization(
    model, inference, parallelism, hardware,
    memory, latency, throughputAnalysis
  );
  const score = calculateOverallScore(memory, latency, throughputAnalysis, utilization, parallelism);

  return {
    isFeasible: true,
    ttft: latency.prefill_total_latency_ms,
    tpot: latency.decode_per_token_latency_ms,
    throughput: throughputAnalysis.tokens_per_second,
    memoryUtilization: memory.memory_utilization,
    score: score.overall_score,
  };
}
