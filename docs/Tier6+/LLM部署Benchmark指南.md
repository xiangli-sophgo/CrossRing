# LLM部署Benchmark指南

## 一、Benchmark的目的与价值

### 1.1 什么是Benchmark？

Benchmark（基准测试）是一套标准化的测试方法和指标体系，用于：

1. **验证分析系统准确性** - 将预测结果与实际测量结果对比，确保建模误差在可接受范围内
2. **公平对比部署方案** - 在相同测试条件下评估不同并行策略的优劣
3. **识别性能瓶颈** - 定位系统的计算、通信、访存等瓶颈
4. **指导优化方向** - 根据瓶颈类型确定优化优先级

### 1.2 为什么需要标准化Benchmark？

| 问题 | 无标准Benchmark | 有标准Benchmark |
|------|-----------------|-----------------|
| 结果可比性 | 不同条件下测试，无法比较 | 相同条件测试，结果可比 |
| 复现性 | 难以复现他人结果 | 按标准流程可复现 |
| 完整性 | 可能遗漏关键场景 | 覆盖主要使用场景 |
| 公正性 | 容易挑选有利数据 | 全面反映真实性能 |

---

## 二、Benchmark维度设计

### 2.1 五大核心维度

```
Benchmark = 模型规模 × 硬件配置 × 并行策略 × 负载场景 × 评测指标
```

#### 维度1：模型规模

| 规模级别 | 参数量 | 代表模型 | 测试意义 |
|----------|--------|----------|----------|
| 小型 | 7B | LLaMA-7B, Mistral-7B | 单卡/少卡验证 |
| 中型 | 13B-34B | LLaMA-13B, CodeLLaMA-34B | 多卡协同 |
| 大型 | 65B-70B | LLaMA-70B | 集群部署 |
| 超大型 | 100B+ | GPT-4, DeepSeek-V2 | 大规模分布式 |
| MoE | 8×7B, 8×22B | Mixtral-8x7B | 专家并行测试 |

#### 维度2：硬件配置

| 配置类型 | 典型配置 | 测试场景 |
|----------|----------|----------|
| 单节点 | 1×H100, 8×H100 | 基准性能 |
| 小集群 | 2-4节点 (16-32卡) | 跨节点通信 |
| 中集群 | 8-16节点 (64-128卡) | 规模化部署 |
| 大集群 | 32+节点 (256+卡) | 生产环境 |

#### 维度3：并行策略

| 策略组合 | 适用场景 | 测试重点 |
|----------|----------|----------|
| TP only | 单节点低延迟 | TP通信开销 |
| PP only | 长序列场景 | 流水线效率 |
| TP+PP | 大模型分布式 | 混合并行效率 |
| DP+TP+PP | 高吞吐服务 | 扩展性 |
| EP+TP | MoE模型 | 专家路由效率 |

#### 维度4：负载场景

| 场景 | batch_size | input_seq | output_seq | 测试目标 |
|------|------------|-----------|------------|----------|
| 交互式 | 1 | 128 | 128 | 最低延迟 |
| 批处理 | 32-64 | 512 | 256 | 最大吞吐 |
| 长上下文 | 1-4 | 32K | 1K | 内存效率 |
| 代码生成 | 4 | 2K | 2K | 平衡负载 |
| 文档摘要 | 8 | 8K | 512 | 长输入处理 |

#### 维度5：评测指标

详见第三节。

---

## 三、核心评测指标

### 3.1 延迟指标

#### 首Token延迟 (Time To First Token, TTFT)

```
TTFT = Prefill完成时间 = 从请求发送到第一个输出token生成的时间
```

| 场景 | 目标TTFT | 说明 |
|------|----------|------|
| 实时对话 | < 100ms | 用户无感知延迟 |
| 交互式 | < 500ms | 可接受的等待时间 |
| 批处理 | < 2s | 吞吐优先 |

**影响因素**：
- 输入序列长度（主要）
- TP并行度（AllReduce开销）
- 模型大小
- batch大小

#### 每Token延迟 (Time Per Output Token, TPOT)

```
TPOT = 平均每个输出token的生成时间 = (总生成时间 - TTFT) / (输出token数 - 1)
```

| 场景 | 目标TPOT | 对应生成速度 |
|------|----------|--------------|
| 流式阅读 | < 50ms | > 20 tokens/s（人类阅读速度） |
| 代码生成 | < 100ms | > 10 tokens/s |
| 批处理 | 不敏感 | 关注总吞吐 |

**影响因素**：
- 显存带宽（主要，Decode是访存密集型）
- KV Cache大小
- batch大小（Continuous Batching）

#### 端到端延迟 (End-to-End Latency, E2E)

```
E2E = TTFT + TPOT × (output_length - 1)
```

### 3.2 吞吐量指标

#### Token吞吐量 (Token Throughput)

```
Token Throughput = 总输出token数 / 总时间 (tokens/s)
```

| 配置 | 典型吞吐量 |
|------|------------|
| 7B @ 1×H100 | 200-500 tokens/s |
| 70B @ 8×H100 | 100-300 tokens/s |
| 70B @ 8×H100 (batch=32) | 1000-3000 tokens/s |

#### 请求吞吐量 (Request Throughput)

```
Request Throughput = 完成的请求数 / 总时间 (req/s)
```

### 3.3 效率指标

#### 模型算力利用率 (Model FLOPs Utilization, MFU)

```
MFU = 实际计算FLOPs / (峰值算力 × 时间) × 100%
```

| 阶段 | 典型MFU |
|------|---------|
| 训练 | 40%-60% |
| Prefill | 30%-50% |
| Decode | 1%-5%（访存瓶颈）|

**注意**：Decode阶段MFU低是正常的，因为受限于访存带宽而非计算。

#### 显存效率

```
显存效率 = 有效使用显存 / 总显存 × 100%
```

其中有效使用显存包括：
- 模型权重
- KV Cache
- 必要的激活值

不包括：
- 碎片化空间
- 冗余缓冲区

#### 性价比 (Cost Efficiency)

```
单token成本 = (硬件租赁成本/小时) / (tokens/小时)
```

| 配置 | 成本估算 |
|------|----------|
| 1×H100 ($3/h) | $0.002-0.006/1K tokens |
| 8×H100 ($24/h) | $0.008-0.024/1K tokens |

---

## 四、标准Benchmark场景

### 4.1 场景1：低延迟交互

**目标**：最小化首token延迟，提供流畅的对话体验

```yaml
benchmark_name: "low_latency_interactive"
model: LLaMA-70B
hardware: 8×H100 (NVLink)
parallelism: TP=8, PP=1
workload:
  batch_size: 1
  input_seq_length: 128
  output_seq_length: 128
optimization_target: TTFT
success_criteria:
  TTFT: < 100ms
  TPOT: < 50ms
```

**测试要点**：
- 单请求延迟（不是吞吐量）
- Prefill阶段效率
- TP AllReduce开销

### 4.2 场景2：高吞吐批处理

**目标**：最大化token吞吐量，服务大量并发请求

```yaml
benchmark_name: "high_throughput_batch"
model: LLaMA-70B
hardware: 8×H100 (NVLink)
parallelism: TP=4, PP=2
workload:
  batch_size: 32
  input_seq_length: 512
  output_seq_length: 256
optimization_target: throughput
success_criteria:
  throughput: > 2000 tokens/s
  TTFT: < 500ms (P99)
```

**测试要点**：
- Continuous Batching效率
- 显存利用率
- 负载均衡

### 4.3 场景3：长上下文处理

**目标**：支持32K+上下文，控制显存和延迟

```yaml
benchmark_name: "long_context"
model: LLaMA-70B
hardware: 8×H100 (NVLink)
parallelism: TP=8, PP=1
workload:
  batch_size: 1
  input_seq_length: 32768
  output_seq_length: 1024
optimization_target: memory_efficiency
success_criteria:
  memory_per_gpu: < 70GB
  TTFT: < 5s
  E2E: < 60s
```

**测试要点**：
- KV Cache显存占用
- 长序列Prefill效率
- 是否需要Context Parallelism

### 4.4 场景4：代码生成

**目标**：平衡延迟和输出长度，支持代码补全

```yaml
benchmark_name: "code_generation"
model: CodeLLaMA-34B
hardware: 4×H100 (NVLink)
parallelism: TP=4, PP=1
workload:
  batch_size: 4
  input_seq_length: 2048
  output_seq_length: 2048
optimization_target: balanced
success_criteria:
  TTFT: < 500ms
  E2E: < 10s
  throughput: > 500 tokens/s
```

**测试要点**：
- 中等长度输入处理
- 长输出生成效率
- 多请求并发

---

## 五、Benchmark矩阵设计

### 5.1 完整测试矩阵

测试矩阵 = 模型 × 硬件 × 并行策略 × 负载场景

```
全量组合数 = |模型| × |硬件| × |并行策略| × |负载场景|
         = 5 × 4 × 5 × 5 = 500 种组合
```

实际测试时需要剪枝：
- 显存不足的组合（自动跳过）
- 不合理的组合（如TP=8用于7B模型）
- 资源限制（选择代表性配置）

### 5.2 核心测试矩阵示例

| 模型 | 硬件 | TP | PP | Batch | Seq | TTFT(ms) | TPOT(ms) | 吞吐(t/s) |
|------|------|----|----|-------|-----|----------|----------|-----------|
| 7B | 1×H100 | 1 | 1 | 1 | 512 | 15 | 8 | 125 |
| 7B | 1×H100 | 1 | 1 | 8 | 512 | 25 | 10 | 800 |
| 70B | 8×H100 | 8 | 1 | 1 | 512 | 50 | 6 | 166 |
| 70B | 8×H100 | 4 | 2 | 1 | 512 | 55 | 7 | 143 |
| 70B | 8×H100 | 8 | 1 | 8 | 512 | 80 | 8 | 1000 |
| 70B | 8×H100 | 8 | 1 | 32 | 512 | 150 | 10 | 3200 |
| 70B | 8×H100 | 8 | 1 | 1 | 4K | 400 | 8 | 125 |
| 70B | 8×H100 | 8 | 1 | 1 | 32K | 3200 | 10 | 100 |
| Mixtral | 8×H100 | 4 | 1 | 8 | 512 | 60 | 9 | 889 |

### 5.3 测试优先级

```
优先级1（必测）：
- 目标模型 + 目标硬件 + 默认并行策略 + 4种标准场景

优先级2（推荐）：
- 目标模型 + 目标硬件 + 所有合理并行策略 + 主要场景

优先级3（完整）：
- 全量测试矩阵
```

---

## 六、Benchmark验证方法

### 6.1 理论值与实测值对比

```
验证流程：
1. 使用分析系统计算预估性能
2. 使用实际框架(vLLM/TensorRT-LLM)运行
3. 对比预估值与实测值
4. 计算预测误差
```

**可接受误差标准**：

| 指标 | 允许误差 | 说明 |
|------|----------|------|
| TTFT | ±20% | 受系统波动影响 |
| TPOT | ±15% | 相对稳定 |
| 吞吐量 | ±20% | 受调度策略影响 |
| 显存占用 | ±10% | 应较准确 |

### 6.2 多次运行统计

```
单次测试可能受以下因素影响：
- GPU温度波动
- 内存分配随机性
- 调度延迟

建议：
- 每个配置运行3-5次
- 取中位数或平均值
- 报告P50/P95/P99延迟
```

### 6.3 预热与稳态测量

```python
# 伪代码示例
def run_benchmark():
    # 预热阶段（结果不计入）
    for _ in range(warmup_rounds):
        run_inference()

    # 正式测量
    results = []
    for _ in range(benchmark_rounds):
        result = run_inference()
        results.append(result)

    return aggregate(results)
```

---

## 七、预设Benchmark配置

### 7.1 标准推理Benchmark

```typescript
const INFERENCE_STANDARD = {
  name: "标准推理Benchmark",
  models: ["LLaMA-7B", "LLaMA-70B"],
  batch_sizes: [1, 8, 32],
  seq_lengths: [512, 2048],
  metrics: ["TTFT", "TPOT", "throughput", "memory"],
  description: "评估不同规模模型在常见负载下的推理性能"
};
```

### 7.2 长上下文Benchmark

```typescript
const LONG_CONTEXT = {
  name: "长上下文Benchmark",
  models: ["LLaMA-70B"],
  batch_sizes: [1, 4],
  seq_lengths: [8192, 16384, 32768],
  metrics: ["memory", "TTFT", "E2E", "throughput"],
  description: "评估长上下文场景的显存效率和延迟表现"
};
```

### 7.3 MoE模型Benchmark

```typescript
const MOE_BENCHMARK = {
  name: "MoE模型Benchmark",
  models: ["Mixtral-8x7B", "DeepSeek-V2-236B"],
  batch_sizes: [8, 32],
  seq_lengths: [2048],
  metrics: ["throughput", "expert_balance", "communication_overhead"],
  description: "评估MoE模型的专家并行效率和负载均衡"
};
```

### 7.4 扩展性Benchmark

```typescript
const SCALABILITY = {
  name: "扩展性Benchmark",
  model: "LLaMA-70B",
  hardware_configs: ["8×H100", "16×H100", "32×H100", "64×H100"],
  parallelism_sweep: true,  // 自动搜索最优并行策略
  metrics: ["throughput_scaling", "efficiency_scaling"],
  description: "评估随硬件规模增加的性能扩展效率"
};
```

---

## 八、Benchmark报告模板

### 8.1 单方案报告

```markdown
# Benchmark Report: LLaMA-70B @ 8×H100

## 配置摘要
- 模型: LLaMA-70B
- 硬件: 8×NVIDIA H100 80GB (NVLink互联)
- 并行策略: TP=8, PP=1, DP=1
- 测试场景: 标准推理

## 性能结果

### 延迟指标
| Batch | Seq | TTFT (ms) | TPOT (ms) | E2E (ms) |
|-------|-----|-----------|-----------|----------|
| 1 | 512 | 48 | 6.2 | 842 |
| 8 | 512 | 75 | 7.8 | 1075 |
| 32 | 512 | 142 | 9.5 | 1360 |

### 吞吐量
| Batch | Token/s | Request/s |
|-------|---------|-----------|
| 1 | 161 | 1.2 |
| 8 | 1026 | 7.7 |
| 32 | 3368 | 25.3 |

### 资源利用
- 显存占用: 65.2 GB / 80 GB (81.5%)
- MFU (Prefill): 42.3%
- 网络带宽利用率: 78.6%

## 瓶颈分析
- Prefill阶段: 计算瓶颈 (MFU=42.3%)
- Decode阶段: 访存瓶颈 (HBM带宽利用率95%)
- TP通信: 占Prefill时间的18%

## 优化建议
1. 增加batch size可提升吞吐量
2. 考虑TP=4+PP=2减少通信开销
```

### 8.2 多方案对比报告

```markdown
# Benchmark Comparison: LLaMA-70B 部署方案对比

## 测试条件
- 模型: LLaMA-70B
- 硬件: 8×H100
- 负载: batch=8, seq=512

## 方案对比

| 方案 | TP | PP | TTFT | TPOT | 吞吐 | 显存 | 综合评分 |
|------|----|----|------|------|------|------|----------|
| A | 8 | 1 | 75ms | 7.8ms | 1026 | 65GB | 85 |
| B | 4 | 2 | 82ms | 8.5ms | 941 | 48GB | 79 |
| C | 2 | 4 | 95ms | 9.2ms | 870 | 38GB | 72 |

## 结论
- 延迟优先: 选择方案A (TP=8)
- 显存受限: 选择方案C (PP=4)
- 平衡选择: 方案A综合评分最高
```

---

## 九、常见Benchmark陷阱

### 9.1 测量陷阱

| 陷阱 | 问题 | 避免方法 |
|------|------|----------|
| 冷启动 | 首次推理包含编译/加载开销 | 预热3-5轮后再测量 |
| 单次测量 | 结果波动大 | 多次运行取统计值 |
| 忽略P99 | 尾延迟影响用户体验 | 报告P50/P95/P99 |
| 过小batch | 无法反映真实服务场景 | 测试多种batch配置 |

### 9.2 对比陷阱

| 陷阱 | 问题 | 避免方法 |
|------|------|----------|
| 不公平对比 | 不同配置下比较 | 固定其他变量 |
| 挑选数据 | 只展示有利结果 | 完整报告所有场景 |
| 忽略约束 | 显存超限的方案 | 标注资源限制 |
| 不同框架 | 框架优化差异大 | 使用相同框架 |

### 9.3 解读陷阱

| 陷阱 | 问题 | 避免方法 |
|------|------|----------|
| MFU误解 | Decode阶段MFU低≠效率差 | 理解瓶颈类型 |
| 线性外推 | 假设性能线性扩展 | 实际测试扩展性 |
| 忽略通信 | 只看计算不看通信 | 分析端到端 |

---

## 十、附录

### 10.1 延迟计算公式

**Prefill延迟**（计算密集型）：
```
T_prefill = FLOPs / (峰值算力 × MFU) + T_comm + T_overhead
```

**Decode延迟**（访存密集型）：
```
T_decode = 模型大小 / 显存带宽 + T_comm_per_token
```

### 10.2 吞吐量计算公式

**最大理论吞吐量**：
```
Throughput_max = min(计算吞吐量, 访存吞吐量, 网络吞吐量)

计算吞吐量 = 峰值算力 × MFU / FLOPs_per_token
访存吞吐量 = 显存带宽 / 模型大小_per_GPU
网络吞吐量 = 带宽 / 通信量_per_token
```

### 10.3 常用模型FLOPs参考

| 模型 | Prefill FLOPs/token | Decode FLOPs/token |
|------|---------------------|---------------------|
| LLaMA-7B | 14.0 GFLOPs | 14.0 GFLOPs |
| LLaMA-70B | 140 GFLOPs | 140 GFLOPs |
| Mixtral-8x7B | 24.6 GFLOPs (2/8 experts) | 24.6 GFLOPs |

### 10.4 参考资料

- [vLLM Benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks)
- [TensorRT-LLM Performance](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance.md)
- [MLPerf Inference](https://mlcommons.org/en/inference-datacenter-30/)
- [LMSys Chatbot Arena](https://chat.lmsys.org/)
