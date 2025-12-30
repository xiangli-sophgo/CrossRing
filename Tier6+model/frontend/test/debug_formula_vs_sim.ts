/**
 * 调试脚本: 比较公式估算 vs 仿真结果
 *
 * 配置:
 * - Model: DeepSeek-V3-671B
 * - DP=1, TP=16, PP=1, EP=2
 * - Batch Size: 8
 * - Input Length: 512
 * - Output Length: 256
 * - Max Seq Length: 768
 */

// 常量定义 (与 latencyEstimator.ts 一致)
const HBM_EFFICIENCY = 0.85;
const COMPUTE_EFFICIENCY = {
  matmul_large: 0.70,
  matmul_small: 0.50,
  attention: 0.60,
  elementwise: 0.30,
  moe_sparse: 0.20,  // MoE 稀疏计算效率低：小 batch、sparse routing、expert load balancing
};
const RANDOM_ACCESS_PENALTY = 1.5;
const GB_TO_BYTES = 1024 * 1024 * 1024;

// 模型配置: DeepSeek-V3-671B
const model = {
  hidden_size: 7168,
  num_layers: 61,
  num_attention_heads: 128,
  num_kv_heads: 128,
  intermediate_size: 18432,  // Dense FFN (前3层)
  vocab_size: 129280,
  dtype: 'bf16',
  bytesPerElement: 2,
  // MoE config
  moe_expert_intermediate_size: 2048,
  num_experts: 256,
  num_experts_per_tok: 8,
  // MLA config
  mla_kv_lora_rank: 512,
  mla_q_lora_rank: 1536,
  mla_qk_nope_head_dim: 128,
  mla_qk_rope_head_dim: 64,
  mla_v_head_dim: 128,
};

// 推理配置
const inference = {
  batch_size: 8,
  input_seq_length: 512,
  output_seq_length: 256,
  max_seq_length: 768,
};

// 并行策略
const parallelism = {
  dp: 1,
  tp: 16,
  pp: 1,
  ep: 2,
};

// 硬件配置: H100 SXM (4节点 x 8卡 = 32卡)
const hardware = {
  compute_tflops_fp16: 989,
  memory_bandwidth_gbps: 3350,
  intra_node_bandwidth_gbps: 900,  // NVLink 4.0
  intra_node_latency_us: 1,
};

// 计算辅助函数
type OpType = 'matmul_large' | 'matmul_small' | 'attention' | 'elementwise' | 'memory_only' | 'moe_sparse';

function calcOpLatency(
  flops: number,
  memoryBytes: number,
  opType: OpType = 'matmul_large',
  isRandomAccess: boolean = false
): { latencyMs: number; computeMs: number; memoryMs: number } {
  const peakFlops = hardware.compute_tflops_fp16 * 1e12;
  const memoryBandwidthGBps = hardware.memory_bandwidth_gbps;

  const computeEfficiency = opType === 'memory_only' ? 1.0 : (COMPUTE_EFFICIENCY as Record<string, number>)[opType];
  const effectiveFlops = peakFlops * computeEfficiency;
  const computeTimeS = flops > 0 ? flops / effectiveFlops : 0;
  const computeTimeMs = computeTimeS * 1000;

  const memoryGB = memoryBytes / GB_TO_BYTES;
  const effectiveBandwidthGBps = memoryBandwidthGBps * HBM_EFFICIENCY;
  let memoryTimeMs = (memoryGB / effectiveBandwidthGBps) * 1000;

  if (isRandomAccess) {
    memoryTimeMs *= RANDOM_ACCESS_PENALTY;
  }

  return {
    latencyMs: Math.max(computeTimeMs, memoryTimeMs),
    computeMs: computeTimeMs,
    memoryMs: memoryTimeMs,
  };
}

// ============================================
// Prefill 逐操作计算
// ============================================
function calculatePrefillLatency() {
  const H = model.hidden_size;
  const B = inference.batch_size;
  const S = inference.input_seq_length;
  const bytesPerElement = model.bytesPerElement;
  const numLayers = model.num_layers;
  const layersPerChip = Math.ceil(numLayers / parallelism.pp);

  // MLA 参数
  const kvDim = model.mla_kv_lora_rank;  // MLA 使用压缩后的 kv_lora_rank
  const headDim = H / model.num_attention_heads;

  // TP 分片
  const headsPerTP = Math.ceil(model.num_attention_heads / parallelism.tp);
  const kvHeadsPerTP = Math.ceil(model.num_kv_heads / parallelism.tp);

  console.log('\n========== Prefill 延迟计算 ==========');
  console.log(`模型: DeepSeek-V3-671B`);
  console.log(`Batch=${B}, SeqLen=${S}, Hidden=${H}`);
  console.log(`Layers=${numLayers}, LayersPerChip=${layersPerChip}`);
  console.log(`TP=${parallelism.tp}, HeadsPerTP=${headsPerTP}, KVHeadsPerTP=${kvHeadsPerTP}`);
  console.log(`MLA: kv_lora_rank=${kvDim}, headDim=${headDim}`);
  console.log('');

  let totalComputeMs = 0;
  let totalMemoryMs = 0;
  let totalCommMs = 0;
  let totalLatencyMs = 0;

  // 计算单层延迟
  console.log('--- 单层操作延迟 ---');

  // 1. LayerNorm 1
  const ln1Flops = 3 * B * S * H;
  const ln1Memory = B * S * H * bytesPerElement * 2;
  const ln1 = calcOpLatency(ln1Flops, ln1Memory, 'elementwise');
  console.log(`[1] LayerNorm1: compute=${ln1.computeMs.toFixed(4)}ms, memory=${ln1.memoryMs.toFixed(4)}ms, latency=${ln1.latencyMs.toFixed(4)}ms`);

  // 2. QKV Projection (使用 MLA 简化: H -> H + kv_lora_rank)
  // 实际 MLA: Q投影(H->q_lora_rank->Q_dim) + KV压缩(H->kv_lora_rank)
  const qkv_output_dim = H + 2 * kvDim;  // 简化计算
  const qkvFlops = 2 * B * S * H * qkv_output_dim / parallelism.tp;
  const qkvWeightBytes = H * qkv_output_dim * bytesPerElement / parallelism.tp;
  const qkvIOBytes = B * S * (H + qkv_output_dim) * bytesPerElement / parallelism.tp;
  const qkv = calcOpLatency(qkvFlops, qkvWeightBytes + qkvIOBytes, 'matmul_large');
  console.log(`[2] QKV Proj: FLOPs=${(qkvFlops/1e12).toFixed(2)}T, Memory=${((qkvWeightBytes+qkvIOBytes)/1e9).toFixed(2)}GB`);
  console.log(`    compute=${qkv.computeMs.toFixed(4)}ms, memory=${qkv.memoryMs.toFixed(4)}ms, latency=${qkv.latencyMs.toFixed(4)}ms`);

  // 3. Attention Score (Q @ K^T)
  const scoreFlops = 2 * B * headsPerTP * S * S * headDim;
  const scoreQBytes = B * headsPerTP * S * headDim * bytesPerElement;
  const scoreKBytes = B * kvHeadsPerTP * S * headDim * bytesPerElement;
  const scoreOutBytes = B * headsPerTP * S * S * bytesPerElement;
  const score = calcOpLatency(scoreFlops, scoreQBytes + scoreKBytes + scoreOutBytes, 'attention');
  console.log(`[3] Attn Score: FLOPs=${(scoreFlops/1e12).toFixed(2)}T, Memory=${((scoreQBytes+scoreKBytes+scoreOutBytes)/1e9).toFixed(2)}GB`);
  console.log(`    compute=${score.computeMs.toFixed(4)}ms, memory=${score.memoryMs.toFixed(4)}ms, latency=${score.latencyMs.toFixed(4)}ms`);

  // 4. Softmax
  const softmaxFlops = 5 * B * headsPerTP * S * S;
  const softmaxMemory = B * headsPerTP * S * S * bytesPerElement * 2;
  const softmax = calcOpLatency(softmaxFlops, softmaxMemory, 'elementwise');
  console.log(`[4] Softmax: compute=${softmax.computeMs.toFixed(4)}ms, memory=${softmax.memoryMs.toFixed(4)}ms, latency=${softmax.latencyMs.toFixed(4)}ms`);

  // 5. Attention Output (S @ V + Output Proj)
  const svFlops = 2 * B * headsPerTP * S * S * headDim;
  const outProjFlops = 2 * B * S * H * H / parallelism.tp;
  const attnOutFlops = svFlops + outProjFlops;
  const svMemory = B * headsPerTP * S * S * bytesPerElement + B * kvHeadsPerTP * S * headDim * bytesPerElement;
  const outProjMemory = H * H * bytesPerElement / parallelism.tp + B * S * H * bytesPerElement;
  const attnOut = calcOpLatency(attnOutFlops, svMemory + outProjMemory, 'attention');
  console.log(`[5] Attn Out: FLOPs=${(attnOutFlops/1e12).toFixed(2)}T, Memory=${((svMemory+outProjMemory)/1e9).toFixed(2)}GB`);
  console.log(`    compute=${attnOut.computeMs.toFixed(4)}ms, memory=${attnOut.memoryMs.toFixed(4)}ms, latency=${attnOut.latencyMs.toFixed(4)}ms`);

  // 6. TP AllReduce 1
  let tpComm1Ms = 0;
  if (parallelism.tp > 1) {
    const allReduceGB = 2 * B * S * H * bytesPerElement * (parallelism.tp - 1) / parallelism.tp / GB_TO_BYTES;
    tpComm1Ms = (allReduceGB / hardware.intra_node_bandwidth_gbps) * 1000 + hardware.intra_node_latency_us / 1000;
  }
  console.log(`[6] TP AllReduce1: ${tpComm1Ms.toFixed(4)}ms`);

  // 7. LayerNorm 2
  const ln2 = calcOpLatency(ln1Flops, ln1Memory, 'elementwise');
  console.log(`[7] LayerNorm2: latency=${ln2.latencyMs.toFixed(4)}ms`);

  // 8-10. FFN 计算 - 需要区分 Dense 层和 MoE 层
  // DeepSeek-V3: 前3层是 Dense FFN, 后58层是 MoE FFN
  // 这里计算的是 "平均" 单层，后面会分层累加

  // Dense FFN (层 0-2): intermediate_size=18432, 按 TP 切分
  const denseFFNI = model.intermediate_size;  // 18432
  const denseGateFlops = 2 * B * S * H * denseFFNI / parallelism.tp;
  const denseGateWeightBytes = H * denseFFNI * bytesPerElement / parallelism.tp;
  const denseGateIOBytes = B * S * (H + denseFFNI) * bytesPerElement / parallelism.tp;
  const denseGate = calcOpLatency(denseGateFlops, denseGateWeightBytes + denseGateIOBytes, 'matmul_large');
  const denseUp = calcOpLatency(denseGateFlops, denseGateWeightBytes + denseGateIOBytes, 'matmul_large');
  const denseDownFlops = 2 * B * S * denseFFNI * H / parallelism.tp;
  const denseDownWeightBytes = denseFFNI * H * bytesPerElement / parallelism.tp;
  const denseDownIOBytes = B * S * (denseFFNI + H) * bytesPerElement / parallelism.tp;
  const denseDown = calcOpLatency(denseDownFlops, denseDownWeightBytes + denseDownIOBytes, 'matmul_large');
  const denseFFNLatency = denseGate.latencyMs + denseUp.latencyMs + denseDown.latencyMs;
  console.log(`[8-10] Dense FFN (层0-2): intermediate=${denseFFNI}`);
  console.log(`       gate=${denseGate.latencyMs.toFixed(4)}ms, up=${denseUp.latencyMs.toFixed(4)}ms, down=${denseDown.latencyMs.toFixed(4)}ms`);
  console.log(`       Dense FFN 单层总延迟: ${denseFFNLatency.toFixed(4)}ms`);

  // MoE FFN (层 3+): expert_intermediate_size=2048, 8个激活专家, 按 EP 切分 (不按 TP!)
  const moeFFNI = model.moe_expert_intermediate_size;  // 2048
  const numActiveExperts = model.num_experts_per_tok;  // 8
  // Prefill 中每个 token 激活 8 个专家，总 FLOPs = B * S * 8 * 2 * H * expert_I
  const moeGateFlops = 2 * B * S * H * moeFFNI * numActiveExperts;  // 不除以 TP!
  // 权重加载: 需要加载 8 个专家的权重
  const moeExpertWeightBytes = 3 * H * moeFFNI * bytesPerElement;  // 单个专家
  const moeTotalWeightBytes = moeExpertWeightBytes * numActiveExperts;  // 8 个专家
  const moeGateIOBytes = B * S * (H + moeFFNI) * bytesPerElement * numActiveExperts;
  // 使用 moe_sparse 效率 (20%) - MoE 是稀疏计算，效率远低于 Dense matmul
  const moeGate = calcOpLatency(moeGateFlops / 2, moeTotalWeightBytes / 3 + moeGateIOBytes / 2, 'moe_sparse');
  const moeUp = calcOpLatency(moeGateFlops / 2, moeTotalWeightBytes / 3 + moeGateIOBytes / 2, 'moe_sparse');
  const moeDownFlops = 2 * B * S * moeFFNI * H * numActiveExperts;
  const moeDown = calcOpLatency(moeDownFlops, moeTotalWeightBytes / 3 + B * S * (moeFFNI + H) * bytesPerElement * numActiveExperts, 'moe_sparse');

  // EP All-to-All 通信开销 (dispatch + combine)
  let epCommMs = 0;
  if (parallelism.ep > 1) {
    // 每个 token 的 hidden state 需要发送到其他 EP rank
    const allToAllBytes = B * S * H * bytesPerElement * 2;  // dispatch + combine
    const epBandwidthGBps = hardware.intra_node_bandwidth_gbps;  // 假设 EP 在节点内
    epCommMs = (allToAllBytes / GB_TO_BYTES / epBandwidthGBps) * 1000 * 2;  // 双向
    console.log(`[EP All-to-All] EP=${parallelism.ep}, 通信量: ${(allToAllBytes/1e6).toFixed(2)}MB, 延迟: ${epCommMs.toFixed(4)}ms`);
  }

  const moeFFNLatency = moeGate.latencyMs + moeUp.latencyMs + moeDown.latencyMs + epCommMs;
  console.log(`[8-10] MoE FFN (层3+): expert_intermediate=${moeFFNI}, active_experts=${numActiveExperts}`);
  console.log(`       gate=${moeGate.latencyMs.toFixed(4)}ms, up=${moeUp.latencyMs.toFixed(4)}ms, down=${moeDown.latencyMs.toFixed(4)}ms`);
  console.log(`       MoE FFN 单层总延迟: ${moeFFNLatency.toFixed(4)}ms (含 EP 通信)`);
  console.log(`       MoE 专家权重加载: ${(moeTotalWeightBytes/1e6).toFixed(2)}MB`);

  // 占位符 - 后面会按层类型分别计算
  const gate = denseGate;
  const up = denseUp;
  const down = denseDown;

  // 11. TP AllReduce 2
  let tpComm2Ms = 0;
  if (parallelism.tp > 1) {
    const allReduceGB = 2 * B * S * H * bytesPerElement * (parallelism.tp - 1) / parallelism.tp / GB_TO_BYTES;
    tpComm2Ms = (allReduceGB / hardware.intra_node_bandwidth_gbps) * 1000 + hardware.intra_node_latency_us / 1000;
  }
  console.log(`[11] TP AllReduce2: ${tpComm2Ms.toFixed(4)}ms`);

  // 12. KV Cache Write
  const kvWriteBytes = 2 * B * S * kvHeadsPerTP * headDim * bytesPerElement;
  const kvWrite = calcOpLatency(0, kvWriteBytes, 'memory_only');
  console.log(`[12] KV Write: memory=${kvWrite.memoryMs.toFixed(4)}ms`);

  // 公共操作延迟 (不包括 FFN)
  const commonOpsLatencyMs = ln1.latencyMs + qkv.latencyMs + score.latencyMs + softmax.latencyMs +
                             attnOut.latencyMs + tpComm1Ms + ln2.latencyMs + tpComm2Ms + kvWrite.latencyMs;
  const commonOpsComputeMs = ln1.computeMs + qkv.computeMs + score.computeMs + softmax.computeMs +
                             attnOut.computeMs + ln2.computeMs;
  const commonOpsMemoryMs = ln1.memoryMs + qkv.memoryMs + score.memoryMs + softmax.memoryMs +
                            attnOut.memoryMs + ln2.memoryMs + kvWrite.memoryMs;
  const layerCommMs = tpComm1Ms + tpComm2Ms;

  // Dense 层 (前 3 层)
  const numDenseLayers = 3;
  const denseLayerLatencyMs = commonOpsLatencyMs + denseFFNLatency;
  const denseLayerComputeMs = commonOpsComputeMs + denseGate.computeMs + denseUp.computeMs + denseDown.computeMs;
  const denseLayerMemoryMs = commonOpsMemoryMs + denseGate.memoryMs + denseUp.memoryMs + denseDown.memoryMs;

  // MoE 层 (后 58 层)
  const numMoELayers = numLayers - numDenseLayers;  // 58
  const moeLayerLatencyMs = commonOpsLatencyMs + moeFFNLatency;
  const moeLayerComputeMs = commonOpsComputeMs + moeGate.computeMs + moeUp.computeMs + moeDown.computeMs;
  const moeLayerMemoryMs = commonOpsMemoryMs + moeGate.memoryMs + moeUp.memoryMs + moeDown.memoryMs;

  console.log('');
  console.log('--- 层类型汇总 ---');
  console.log(`Dense 层 (0-2): ${numDenseLayers} 层`);
  console.log(`  单层延迟: ${denseLayerLatencyMs.toFixed(4)}ms`);
  console.log(`  Dense 层总延迟: ${(denseLayerLatencyMs * numDenseLayers).toFixed(2)}ms`);
  console.log(`MoE 层 (3-60): ${numMoELayers} 层`);
  console.log(`  单层延迟: ${moeLayerLatencyMs.toFixed(4)}ms`);
  console.log(`  MoE 层总延迟: ${(moeLayerLatencyMs * numMoELayers).toFixed(2)}ms`);

  // 所有层总计
  totalComputeMs = denseLayerComputeMs * numDenseLayers + moeLayerComputeMs * numMoELayers;
  totalMemoryMs = denseLayerMemoryMs * numDenseLayers + moeLayerMemoryMs * numMoELayers;
  totalCommMs = layerCommMs * numLayers;
  totalLatencyMs = denseLayerLatencyMs * numDenseLayers + moeLayerLatencyMs * numMoELayers;

  console.log('');
  console.log('--- Prefill 总计 (所有层) ---');
  console.log(`总层数: ${layersPerChip}`);
  console.log(`总计算时间: ${totalComputeMs.toFixed(2)}ms`);
  console.log(`总访存时间: ${totalMemoryMs.toFixed(2)}ms`);
  console.log(`总通信时间: ${totalCommMs.toFixed(2)}ms`);
  console.log(`总延迟 (TTFT): ${totalLatencyMs.toFixed(2)}ms`);

  return {
    totalMs: totalLatencyMs,
    computeMs: totalComputeMs,
    memoryMs: totalMemoryMs,
    commMs: totalCommMs,
  };
}

// ============================================
// Decode 逐操作计算
// ============================================
function calculateDecodeLatency() {
  const H = model.hidden_size;
  const B = inference.batch_size;
  const S = 1;  // Decode 每次处理 1 个 token
  const C = inference.input_seq_length + inference.output_seq_length / 2;  // 平均 context 长度
  const bytesPerElement = model.bytesPerElement;
  const numLayers = model.num_layers;
  const layersPerChip = Math.ceil(numLayers / parallelism.pp);

  const kvDim = model.mla_kv_lora_rank;
  const headDim = H / model.num_attention_heads;
  const headsPerTP = Math.ceil(model.num_attention_heads / parallelism.tp);
  const kvHeadsPerTP = Math.ceil(model.num_kv_heads / parallelism.tp);

  console.log('\n========== Decode 延迟计算 ==========');
  console.log(`Batch=${B}, S=1 (per token), Context=${C}`);

  // **关键**: Decode 是 memory-bound，需要计算每层的模型权重加载
  //
  // 注意并行策略:
  // - TP (Tensor Parallel): 切分 Attention 和非专家权重
  // - EP (Expert Parallel): 切分 MoE 专家权重
  //
  // DeepSeek-V3 权重分布:
  // - MLA Attention: 由 TP 切分
  // - MoE Experts: 由 EP 切分 (不是 TP!)
  //
  // 每层 Attention 权重 (MLA):
  // - Q down: H * q_lora_rank
  // - Q up: q_lora_rank * num_heads * (qk_nope + qk_rope)
  // - KV compress: H * kv_lora_rank
  // - Output proj: num_heads * v_head_dim * H
  const qLoraDownBytes = H * model.mla_q_lora_rank * bytesPerElement / parallelism.tp;
  const qLoraUpBytes = model.mla_q_lora_rank * model.num_attention_heads * (model.mla_qk_nope_head_dim + model.mla_qk_rope_head_dim) * bytesPerElement / parallelism.tp;
  const kvCompressBytes = H * model.mla_kv_lora_rank * bytesPerElement / parallelism.tp;
  const outProjBytes = model.num_attention_heads * model.mla_v_head_dim * H * bytesPerElement / parallelism.tp;
  const attnWeightBytesPerLayer = qLoraDownBytes + qLoraUpBytes + kvCompressBytes + outProjBytes;

  // MoE FFN 权重: 每个专家有 3 * H * expert_I 参数
  // 专家按 EP 分片! 每个 token 激活 8 个专家
  const ffnI = model.moe_expert_intermediate_size;
  const expertWeightBytes = 3 * H * ffnI * bytesPerElement;  // 单个专家 (不除以 TP!)
  // 每个 EP rank 拥有 num_experts / EP 个专家
  const expertsPerEP = model.num_experts / parallelism.ep;
  // 但每个 token 只激活 num_experts_per_tok 个专家
  // 这些激活的专家可能分布在不同的 EP rank 上，平均每个 EP 处理 num_experts_per_tok / EP
  const activeExpertsPerEP = model.num_experts_per_tok;  // 最坏情况: 所有激活专家在同一 EP
  const ffnWeightBytesPerLayer = expertWeightBytes * activeExpertsPerEP;

  const totalWeightBytesPerLayer = attnWeightBytesPerLayer + ffnWeightBytesPerLayer;

  console.log(`\n--- Decode 模型权重分析 (每层/单芯片) ---`);
  console.log(`MLA Attention 权重:`);
  console.log(`  Q LoRA down: ${(qLoraDownBytes/1e6).toFixed(2)} MB`);
  console.log(`  Q LoRA up: ${(qLoraUpBytes/1e6).toFixed(2)} MB`);
  console.log(`  KV compress: ${(kvCompressBytes/1e6).toFixed(2)} MB`);
  console.log(`  Output proj: ${(outProjBytes/1e6).toFixed(2)} MB`);
  console.log(`  Attention 总计: ${(attnWeightBytesPerLayer/1e6).toFixed(2)} MB`);
  console.log(`MoE FFN 权重:`);
  console.log(`  单个专家: ${(expertWeightBytes/1e6).toFixed(2)} MB`);
  console.log(`  激活 ${activeExpertsPerEP} 个专家: ${(ffnWeightBytesPerLayer/1e6).toFixed(2)} MB`);
  console.log(`单层总权重: ${(totalWeightBytesPerLayer/1e6).toFixed(2)} MB`);
  console.log(`所有层总权重 (${layersPerChip}层): ${(totalWeightBytesPerLayer * layersPerChip/1e9).toFixed(2)} GB`);

  // 计算权重加载时间
  const weightLoadTimePerLayerMs = (totalWeightBytesPerLayer / GB_TO_BYTES) / (hardware.memory_bandwidth_gbps * HBM_EFFICIENCY) * 1000;
  console.log(`\n权重加载时间: ${weightLoadTimePerLayerMs.toFixed(4)} ms/层`);
  console.log(`所有层权重加载: ${(weightLoadTimePerLayerMs * layersPerChip).toFixed(2)} ms`);

  // 1. LayerNorm 1
  const ln1Flops = 3 * B * S * H;
  const ln1Memory = B * S * H * bytesPerElement * 2;
  const ln1 = calcOpLatency(ln1Flops, ln1Memory, 'elementwise');

  // 2. QKV Projection (小矩阵)
  const qkv_output_dim = H + 2 * kvDim;
  const qkvFlops = 2 * B * S * H * qkv_output_dim / parallelism.tp;
  const qkvWeightBytes = H * qkv_output_dim * bytesPerElement / parallelism.tp;
  const qkvIOBytes = B * S * (H + qkv_output_dim) * bytesPerElement / parallelism.tp;
  const qkv = calcOpLatency(qkvFlops, qkvWeightBytes + qkvIOBytes, 'matmul_small');

  // 3. KV Cache Read (随机访问)
  const kvReadBytes = 2 * B * C * kvHeadsPerTP * headDim * bytesPerElement;
  const kvRead = calcOpLatency(0, kvReadBytes, 'memory_only', true);  // 随机访问
  console.log(`[3] KV Read: bytes=${(kvReadBytes/1e6).toFixed(2)}MB, memory=${kvRead.memoryMs.toFixed(4)}ms (random access x1.5)`);

  // 4. Attention Score
  const scoreFlops = 2 * B * headsPerTP * S * C * headDim;
  const scoreMemory = (B * headsPerTP * S * headDim + B * kvHeadsPerTP * C * headDim + B * headsPerTP * S * C) * bytesPerElement;
  const score = calcOpLatency(scoreFlops, scoreMemory, 'attention');

  // 5. Softmax
  const softmaxFlops = 5 * B * headsPerTP * S * C;
  const softmaxMemory = B * headsPerTP * S * C * bytesPerElement * 2;
  const softmax = calcOpLatency(softmaxFlops, softmaxMemory, 'elementwise');

  // 6. Attention Output
  const svFlops = 2 * B * headsPerTP * S * C * headDim;
  const outProjFlops = 2 * B * S * H * H / parallelism.tp;
  const attnOutFlops = svFlops + outProjFlops;
  const svMemory = B * headsPerTP * S * C * bytesPerElement + B * kvHeadsPerTP * C * headDim * bytesPerElement;
  const outProjMemory = H * H * bytesPerElement / parallelism.tp + B * S * H * bytesPerElement;
  const attnOut = calcOpLatency(attnOutFlops, svMemory + outProjMemory, 'matmul_small');

  // 7. TP AllReduce 1
  let tpComm1Ms = 0;
  if (parallelism.tp > 1) {
    const allReduceGB = 2 * B * S * H * bytesPerElement * (parallelism.tp - 1) / parallelism.tp / GB_TO_BYTES;
    tpComm1Ms = (allReduceGB / hardware.intra_node_bandwidth_gbps) * 1000 + hardware.intra_node_latency_us / 1000;
  }

  // 8. LayerNorm 2
  const ln2 = calcOpLatency(ln1Flops, ln1Memory, 'elementwise');

  // 9-11. FFN (MoE: 需要加载 num_experts_per_tok 个专家的权重)
  // 注意: MoE 专家不按 TP 切分，按 EP 切分
  const numActiveExperts = model.num_experts_per_tok;
  // 每个专家: gate (H*I) + up (H*I) + down (I*H) = 3 * H * I 参数
  const singleExpertGateUpBytes = 2 * H * ffnI * bytesPerElement;  // 不除以 TP
  const singleExpertDownBytes = H * ffnI * bytesPerElement;  // 不除以 TP
  const totalExpertGateUpBytes = singleExpertGateUpBytes * numActiveExperts;
  const totalExpertDownBytes = singleExpertDownBytes * numActiveExperts;

  const expertGateFlops = 2 * B * S * H * ffnI * numActiveExperts;  // 所有激活专家的计算量
  // 使用 moe_sparse 效率 (Decode 也是稀疏计算)
  const gate = calcOpLatency(expertGateFlops / 2, totalExpertGateUpBytes / 2 + B * S * (H + ffnI) * bytesPerElement, 'moe_sparse');
  const up = calcOpLatency(expertGateFlops / 2, totalExpertGateUpBytes / 2 + B * S * (H + ffnI) * bytesPerElement, 'moe_sparse');

  const expertDownFlops = 2 * B * S * ffnI * H * numActiveExperts;
  const down = calcOpLatency(expertDownFlops, totalExpertDownBytes + B * S * (ffnI + H) * bytesPerElement, 'moe_sparse');

  // EP All-to-All 通信 (Decode 也需要!)
  let decodeEpCommMs = 0;
  if (parallelism.ep > 1) {
    const allToAllBytes = B * S * H * bytesPerElement * 2;  // dispatch + combine
    decodeEpCommMs = (allToAllBytes / GB_TO_BYTES / hardware.intra_node_bandwidth_gbps) * 1000 * 2;
    console.log(`[EP All-to-All Decode] 通信量: ${(allToAllBytes/1e6).toFixed(4)}MB, 延迟: ${decodeEpCommMs.toFixed(4)}ms`);
  }

  console.log(`[9-11] FFN MoE: 加载 ${numActiveExperts} 个专家 (moe_sparse 效率)`);
  console.log(`       Gate+Up 权重: ${(totalExpertGateUpBytes/1e6).toFixed(2)}MB, Down 权重: ${(totalExpertDownBytes/1e6).toFixed(2)}MB`);
  console.log(`       gate=${gate.latencyMs.toFixed(4)}ms, up=${up.latencyMs.toFixed(4)}ms, down=${down.latencyMs.toFixed(4)}ms`);

  // 12. TP AllReduce 2
  let tpComm2Ms = 0;
  if (parallelism.tp > 1) {
    const allReduceGB = 2 * B * S * H * bytesPerElement * (parallelism.tp - 1) / parallelism.tp / GB_TO_BYTES;
    tpComm2Ms = (allReduceGB / hardware.intra_node_bandwidth_gbps) * 1000 + hardware.intra_node_latency_us / 1000;
  }

  // 13. KV Write
  const kvWriteBytes = 2 * B * S * kvHeadsPerTP * headDim * bytesPerElement;
  const kvWrite = calcOpLatency(0, kvWriteBytes, 'memory_only');

  // 单层总计 (含 EP 通信)
  const layerLatencyMs = ln1.latencyMs + qkv.latencyMs + kvRead.latencyMs + score.latencyMs +
                         softmax.latencyMs + attnOut.latencyMs + tpComm1Ms + ln2.latencyMs +
                         gate.latencyMs + up.latencyMs + down.latencyMs + decodeEpCommMs + tpComm2Ms + kvWrite.latencyMs;

  const layerComputeMs = ln1.computeMs + qkv.computeMs + score.computeMs + softmax.computeMs +
                         attnOut.computeMs + ln2.computeMs + gate.computeMs + up.computeMs + down.computeMs;

  const layerMemoryMs = ln1.memoryMs + qkv.memoryMs + kvRead.memoryMs + score.memoryMs +
                        softmax.memoryMs + attnOut.memoryMs + ln2.memoryMs + gate.memoryMs +
                        up.memoryMs + down.memoryMs + kvWrite.memoryMs;

  const layerCommMs = tpComm1Ms + tpComm2Ms + decodeEpCommMs;

  console.log('');
  console.log('--- Decode 单层汇总 ---');
  console.log(`单层计算时间: ${layerComputeMs.toFixed(4)}ms`);
  console.log(`单层访存时间: ${layerMemoryMs.toFixed(4)}ms`);
  console.log(`单层通信时间: ${layerCommMs.toFixed(4)}ms`);
  console.log(`单层总延迟: ${layerLatencyMs.toFixed(4)}ms`);

  // MoE Router 计算开销 (每个 MoE 层需要计算 routing scores)
  // Router FLOPs = B * S * H * num_experts (计算每个 token 到每个专家的分数)
  const numMoELayers = numLayers - 3;  // 前 3 层是 Dense
  const routerFlopsPerLayer = B * S * H * model.num_experts;  // 7168 * 256 = 1.84M
  const routerWeightBytes = H * model.num_experts * bytesPerElement;  // 7168 * 256 * 2 = 3.67 MB
  const router = calcOpLatency(routerFlopsPerLayer, routerWeightBytes, 'matmul_small');
  const routerOverheadMs = router.latencyMs * numMoELayers;
  console.log(`[MoE Router] 每层: ${router.latencyMs.toFixed(4)}ms, 总计: ${routerOverheadMs.toFixed(2)}ms (${numMoELayers} MoE 层)`);

  // Kernel launch overhead 估算 (通常被忽略但实际影响显著)
  // 每层约 10-15 个 kernel (LN, QKV, Attn, FFN 等)
  // 每个 kernel launch 约 10-20 μs
  const kernelsPerLayer = 12;
  const kernelLaunchUs = 15;  // μs
  const kernelOverheadPerLayerMs = kernelsPerLayer * kernelLaunchUs / 1000;
  const totalKernelOverheadMs = kernelOverheadPerLayerMs * layersPerChip;

  const tpot = layerLatencyMs * layersPerChip;
  const tpotWithOverhead = tpot + totalKernelOverheadMs + routerOverheadMs;

  console.log('');
  console.log('--- Decode 总计 ---');
  console.log(`TPOT (纯计算+访存): ${tpot.toFixed(2)}ms`);
  console.log(`Kernel Launch 开销: ${totalKernelOverheadMs.toFixed(2)}ms`);
  console.log(`MoE Router 开销: ${routerOverheadMs.toFixed(2)}ms`);
  console.log(`TPOT (含所有开销): ${tpotWithOverhead.toFixed(2)}ms`);

  return {
    tpot,
    layerLatencyMs,
    layerComputeMs,
    layerMemoryMs,
  };
}

// ============================================
// 运行计算
// ============================================
console.log('=========================================');
console.log(' 公式 vs 仿真 详细对比');
console.log(' DeepSeek-V3-671B, TP=16, PP=1, EP=2');
console.log('=========================================');

const prefill = calculatePrefillLatency();
const decode = calculateDecodeLatency();

console.log('\n========== 最终汇总 ==========');
console.log(`TTFT (Prefill): ${prefill.totalMs.toFixed(2)}ms`);
console.log(`TPOT (Decode): ${decode.tpot.toFixed(2)}ms`);
console.log(`E2E Latency: ${(prefill.totalMs + decode.tpot * inference.output_seq_length).toFixed(2)}ms`);

// 计算 MFU
const prefillFlops = 2 * 12 * model.num_layers * model.hidden_size * model.hidden_size * inference.batch_size * inference.input_seq_length;
const totalChips = parallelism.dp * parallelism.tp * parallelism.pp * parallelism.ep;
const peakTflops = hardware.compute_tflops_fp16 * totalChips;
const achievedTflops = (prefillFlops / 1e12) / (prefill.totalMs / 1000);
const mfu = achievedTflops / peakTflops;
console.log(`\nPrefill MFU: ${(mfu * 100).toFixed(2)}%`);
console.log(`Peak TFLOPs (${totalChips} chips): ${peakTflops.toFixed(0)} TFLOPs`);
console.log(`Achieved TFLOPs: ${achievedTflops.toFixed(0)} TFLOPs`);

// ============================================
// MBU 计算 (Memory Bandwidth Utilization)
// ============================================
console.log('\n========== MBU 计算 ==========');

// Decode 每 token 需读取的数据量
// 1. 模型权重 (每 token 都需要加载)
const H = model.hidden_size;
const bytesPerElement = model.bytesPerElement;

// MLA Attention 权重 (per layer, per TP rank)
const attnWeightPerLayer = (
  H * model.mla_q_lora_rank +  // Q LoRA down
  model.mla_q_lora_rank * model.num_attention_heads * (model.mla_qk_nope_head_dim + model.mla_qk_rope_head_dim) +  // Q LoRA up
  H * model.mla_kv_lora_rank +  // KV compress
  model.num_attention_heads * model.mla_v_head_dim * H  // Output proj
) * bytesPerElement / parallelism.tp;

// MoE FFN 权重 (8 experts, NOT divided by TP)
const moeExpertWeightPerLayer = 3 * H * model.moe_expert_intermediate_size * bytesPerElement * model.num_experts_per_tok;

// Dense FFN 权重 (前3层, divided by TP)
const denseFFNWeightPerLayer = 3 * H * model.intermediate_size * bytesPerElement / parallelism.tp;

// 每层总权重
const numDenseLayers = 3;
const numMoELayers = model.num_layers - numDenseLayers;

const denseLayerWeight = attnWeightPerLayer + denseFFNWeightPerLayer;
const moeLayerWeight = attnWeightPerLayer + moeExpertWeightPerLayer;

const totalModelWeightBytes = denseLayerWeight * numDenseLayers + moeLayerWeight * numMoELayers;
const totalModelWeightGB = totalModelWeightBytes / GB_TO_BYTES;

console.log(`--- 模型权重分析 (单芯片每次 Decode 需加载) ---`);
console.log(`Dense 层权重 (${numDenseLayers}层): ${(denseLayerWeight * numDenseLayers / 1e9).toFixed(2)} GB`);
console.log(`  Attention: ${(attnWeightPerLayer / 1e6).toFixed(2)} MB/层`);
console.log(`  Dense FFN: ${(denseFFNWeightPerLayer / 1e6).toFixed(2)} MB/层`);
console.log(`MoE 层权重 (${numMoELayers}层): ${(moeLayerWeight * numMoELayers / 1e9).toFixed(2)} GB`);
console.log(`  Attention: ${(attnWeightPerLayer / 1e6).toFixed(2)} MB/层`);
console.log(`  MoE FFN (8 experts): ${(moeExpertWeightPerLayer / 1e6).toFixed(2)} MB/层`);
console.log(`模型权重总计: ${totalModelWeightGB.toFixed(2)} GB`);

// 2. KV Cache (MLA 使用压缩后的 kv_lora_rank)
const avgContext = inference.input_seq_length + inference.output_seq_length / 2;
const kvCachePerLayerBytes = 2 * inference.batch_size * avgContext * model.mla_kv_lora_rank * bytesPerElement / parallelism.tp;
const kvCacheTotalBytes = kvCachePerLayerBytes * model.num_layers;
const kvCacheTotalGB = kvCacheTotalBytes / GB_TO_BYTES;
console.log(`\n--- KV Cache 分析 ---`);
console.log(`MLA KV 维度: ${model.mla_kv_lora_rank} (压缩后)`);
console.log(`Context 长度: ${avgContext}`);
console.log(`KV Cache 总量: ${kvCacheTotalGB.toFixed(2)} GB`);

// 3. 计算 MBU
const dataReadPerTokenGB = totalModelWeightGB + kvCacheTotalGB;
const peakBandwidthGBps = hardware.memory_bandwidth_gbps;

// 理想 MBU (纯计算+访存，不含 kernel 开销)
const tpotIdealSeconds = decode.tpot / 1000;
const achievedBandwidthIdeal = dataReadPerTokenGB / tpotIdealSeconds;
const mbuIdeal = achievedBandwidthIdeal / peakBandwidthGBps;

// 实际 MBU (含 kernel launch 开销)
const kernelsPerLayer = 12;
const kernelLaunchUs = 15;
const kernelOverheadMs = kernelsPerLayer * kernelLaunchUs / 1000 * model.num_layers;
const tpotRealMs = decode.tpot + kernelOverheadMs;
const tpotRealSeconds = tpotRealMs / 1000;
const achievedBandwidthReal = dataReadPerTokenGB / tpotRealSeconds;
const mbuReal = achievedBandwidthReal / peakBandwidthGBps;

console.log(`\n--- MBU 计算 ---`);
console.log(`每 token 数据读取: ${dataReadPerTokenGB.toFixed(2)} GB`);
console.log(`峰值带宽: ${peakBandwidthGBps.toFixed(0)} GB/s`);
console.log(`\n[理想 MBU - 不含 kernel 开销]`);
console.log(`  TPOT: ${decode.tpot.toFixed(2)} ms`);
console.log(`  实际带宽: ${achievedBandwidthIdeal.toFixed(0)} GB/s`);
console.log(`  MBU: ${(mbuIdeal * 100).toFixed(1)}%`);
console.log(`\n[实际 MBU - 含 kernel launch 开销]`);
console.log(`  Kernel 开销: ${kernelOverheadMs.toFixed(2)} ms`);
console.log(`  TPOT: ${tpotRealMs.toFixed(2)} ms`);
console.log(`  实际带宽: ${achievedBandwidthReal.toFixed(0)} GB/s`);
console.log(`  MBU: ${(mbuReal * 100).toFixed(1)}%`);

// 理论 MBU (基于 HBM 效率)
const theoreticalMbu = HBM_EFFICIENCY;
console.log(`\n理论最大 MBU (HBM效率): ${(theoreticalMbu * 100).toFixed(1)}%`);
console.log(`理想 MBU 差距: ${((theoreticalMbu - mbuIdeal) * 100).toFixed(1)}%`);
console.log(`实际 MBU 差距: ${((theoreticalMbu - mbuReal) * 100).toFixed(1)}%`);
