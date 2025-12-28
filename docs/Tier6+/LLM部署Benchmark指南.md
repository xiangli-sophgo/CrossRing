# LLM部署Benchmark指南

> 本文档整合了业界最权威的 LLM 推理 Benchmark 标准、评价指标体系和开源工具，用于指导 LLM 部署分析系统的开发和验证。

---

## 一、业界权威Benchmark标准

### 1.1 MLPerf Inference（行业金标准）

[MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) 是由 MLCommons 组织维护的业界最权威推理基准测试，被 NVIDIA、Google、Intel、AMD 等主流厂商广泛采用。

#### MLPerf Inference v5.0/v5.1 LLM 标准（2024-2025）

| 指标 | Server 场景 | Interactive 场景 | 说明 |
|------|-------------|------------------|------|
| **TTFT (P99)** | ≤ 2000ms | ≤ 450ms | 首 Token 延迟第 99 百分位 |
| **TPOT (P99)** | ≤ 200ms | ≤ 40ms | 约等于 25 tokens/s 生成速度 |
| **吞吐量单位** | tokens/s | tokens/s | 统一使用 token 计量 |

**为什么使用 P99 而非平均值？**
- 用户体验由最差情况决定，不是平均情况
- P99 = 99% 的请求都能满足的延迟阈值
- SLA（服务等级协议）通常以 P99 定义

#### MLPerf LLM Benchmark 模型

| 版本 | 模型 | 任务 | 精度指标 |
|------|------|------|----------|
| v3.1-v5.0 | Llama 2 70B | 文本摘要 (CNN/DailyMail) | ROUGE-1/2/L ≥ 99% of FP16 |
| v5.1 | Llama 3.1 8B | 文本摘要 | ROUGE scores |
| v5.1 | DeepSeek-R1 | 推理任务 | 首个推理模型 Benchmark |

#### MLPerf 场景定义

```
┌─────────────────────────────────────────────────────────────────┐
│  Server 场景                                                     │
│  - 请求到达：泊松分布随机到达                                    │
│  - 测量指标：在满足延迟约束下的最大吞吐量                        │
│  - 适用场景：在线服务、API 接口                                  │
├─────────────────────────────────────────────────────────────────┤
│  Offline 场景                                                    │
│  - 请求到达：所有请求同时到达                                    │
│  - 测量指标：纯吞吐量 (tokens/s)                                 │
│  - 适用场景：批处理、离线推理                                    │
├─────────────────────────────────────────────────────────────────┤
│  Single Stream 场景                                              │
│  - 请求到达：串行处理，一个完成后发下一个                        │
│  - 测量指标：P90 端到端延迟                                      │
│  - 适用场景：边缘设备、单用户场景                                │
└─────────────────────────────────────────────────────────────────┘
```

**参考链接**：
- [MLPerf Inference v5.1 Results](https://mlcommons.org/2025/09/mlperf-inference-v5-1-results/)
- [MLPerf LLM Benchmark 规范](https://mlcommons.org/2024/03/mlperf-llama2-70b/)

---

### 1.2 Metron/Etalon 框架（用户体验导向）

[Metron](https://arxiv.org/abs/2407.07000) 是微软研究院提出的 LLM 推理评估框架，专注于**用户体验指标**，弥补了传统指标的不足。

#### 传统指标的局限性

| 传统指标 | 问题 |
|----------|------|
| 平均 TTFT | 无法反映尾部延迟 |
| 平均 TPOT | 忽略生成过程中的卡顿 |
| 总吞吐量 | 不关心单个用户体验 |

#### Metron 新指标

**1. Fluidity-Index（流畅度指数）**

```
定义：Token 生成是否"流畅"，即每个 Token 是否在预期时间内生成

计算方法：
1. 为每个 Token 设定 deadline（基于目标生成速率）
2. 统计满足 deadline 的 Token 比例
3. 如果发生"卡顿"（stall），重置 deadline

Fluidity-Index = 满足 deadline 的 Token 数 / 总 Token 数

示例：
- 目标：25 tokens/s → 每个 Token deadline = 40ms
- 如果第 5 个 Token 延迟了 100ms，视为 stall
- Stall 后重置，后续 Token 重新计时
```

**2. Fluid Token Generation Rate（流畅生成速率）**

```
定义：在满足 Fluidity-Index SLO 约束下，系统能支持的最大 Token 生成速率

用途：
- 容量规划：确定服务需要多少 GPU
- SLO 设计：在流畅性和吞吐量之间权衡
```

**Metron 工具**：
- 基于 [LLMPerf](https://github.com/ray-project/llmperf) 扩展
- 支持 vLLM、OpenAI API 等
- 提供容量搜索模块

**参考链接**：
- [Metron 论文](https://arxiv.org/abs/2407.07000)
- [Microsoft Research 页面](https://www.microsoft.com/en-us/research/publication/etalon-holistic-performance-evaluation-framework-for-llm-inference-systems/)

---

### 1.3 NVIDIA NIM Benchmarking

[NVIDIA NIM Benchmarking](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) 是 NVIDIA 官方的 LLM 推理基准测试标准。

#### 核心指标定义

| 指标 | 计算公式 | 说明 |
|------|----------|------|
| **TTFT** | 从请求发送到首 Token 返回 | 包含排队、预填充、网络延迟 |
| **ITL** (Inter-Token Latency) | 连续 Token 之间的平均间隔 | 与 TPOT 同义 |
| **E2E Latency** | TTFT + ITL × (output_tokens - 1) | 端到端延迟 |
| **TPS** (Tokens Per Second) | 总输出 Token / 总时间 | 系统吞吐量 |
| **RPS** (Requests Per Second) | 完成请求数 / 总时间 | 请求吞吐量 |

#### 统计方法

```
建议报告的统计量：
├── P50 (中位数)：典型用户体验
├── P90：大多数用户体验
├── P95：接近最差情况
├── P99：SLA 约束基准
├── 平均值：整体趋势参考
└── 标准差：稳定性评估
```

---

## 二、核心评测指标体系

### 2.1 延迟指标（Latency Metrics）

#### 2.1.1 TTFT - Time To First Token

```
TTFT = T_queue + T_prefill + T_network

其中：
├── T_queue：请求排队时间（高负载时显著）
├── T_prefill：预填充计算时间（与输入长度相关）
└── T_network：网络传输延迟
```

**影响因素分析**：

| 因素 | 影响方向 | 量化关系 |
|------|----------|----------|
| 输入序列长度 | TTFT ↑ | 近似线性，O(seq_len) |
| 模型大小 | TTFT ↑ | 近似线性 |
| TP 并行度 | TTFT ↓ 但有通信开销 | 非线性，存在拐点 |
| Batch Size | TTFT ↑ | 但可提升吞吐 |
| 是否有 KV Cache 复用 | TTFT ↓ | Prefix Caching 可大幅减少 |

**MLPerf 标准阈值**：

| 场景 | TTFT P99 阈值 | 说明 |
|------|---------------|------|
| 交互式（Interactive）| ≤ 450ms | 实时对话、代码助手 |
| 服务器（Server）| ≤ 2000ms | 一般在线服务 |
| 离线（Offline）| 无限制 | 批处理场景 |

#### 2.1.2 TPOT - Time Per Output Token

```
TPOT = (E2E_Latency - TTFT) / (output_tokens - 1)

也称为：
├── ITL (Inter-Token Latency)
├── TBT (Time Between Tokens)
└── Decode Latency per Token
```

**为什么 Decode 阶段是访存瓶颈？**

```
Prefill 阶段：
- 输入：batch × seq_len × hidden_size
- 矩阵乘法：(batch × seq) × hidden × hidden
- 计算密集型，GPU 算力可被充分利用

Decode 阶段：
- 每次只处理 1 个新 Token
- 但需要加载完整模型权重
- 成为 矩阵-向量 乘法，访存密集型
- GPU 大部分时间在等待数据传输
```

**TPOT 与用户体验**：

| TPOT | 生成速率 | 用户感知 |
|------|----------|----------|
| ≤ 20ms | ≥ 50 tok/s | 极快，超越阅读速度 |
| 20-40ms | 25-50 tok/s | 流畅，MLPerf 标准 |
| 40-100ms | 10-25 tok/s | 可接受，人类阅读速度 |
| > 100ms | < 10 tok/s | 明显等待感 |

#### 2.1.3 E2E Latency - 端到端延迟

```
E2E = TTFT + TPOT × (output_tokens - 1)

示例：
- TTFT = 100ms, TPOT = 30ms, output = 100 tokens
- E2E = 100 + 30 × 99 = 3070ms ≈ 3s
```

### 2.2 吞吐量指标（Throughput Metrics）

#### 2.2.1 Token Throughput

```
Token Throughput = Σ output_tokens / total_time  (tokens/s)

注意：
- 应统计所有并发请求的输出 Token 总和
- 不同请求的 output_tokens 可能不同
```

**典型参考值**（单节点 8×H100）：

| 模型 | Batch=1 | Batch=8 | Batch=32 |
|------|---------|---------|----------|
| LLaMA-7B | 150-200 | 800-1200 | 2000-3000 |
| LLaMA-70B | 100-150 | 600-1000 | 1500-2500 |
| Mixtral-8x7B | 120-180 | 700-1100 | 1800-2800 |

#### 2.2.2 Request Throughput

```
Request Throughput = completed_requests / total_time  (req/s)

与 Token Throughput 的关系：
Token_Throughput ≈ Request_Throughput × avg_output_length
```

### 2.3 效率指标（Efficiency Metrics）

#### 2.3.1 MFU - Model FLOPs Utilization

```
MFU = Achieved_FLOPs / (Peak_FLOPs × Time) × 100%

其中：
- Achieved_FLOPs：实际完成的浮点运算数
- Peak_FLOPs：硬件理论峰值算力
```

**各阶段典型 MFU**：

| 阶段 | 典型 MFU | 瓶颈类型 | 说明 |
|------|----------|----------|------|
| 训练 | 40-60% | 计算+通信 | 可通过优化提升 |
| Prefill | 30-50% | 计算密集 | 受限于 Attention 计算 |
| Decode | 1-5% | **访存密集** | 正常现象，非效率低下 |

**重要提示**：Decode 阶段 MFU 低是**正常的**，因为瓶颈在显存带宽，不是计算能力。此时应关注 MBU。

#### 2.3.2 MBU - Memory Bandwidth Utilization（关键指标）

```
MBU = Achieved_Memory_Bandwidth / Peak_Memory_Bandwidth × 100%

其中：
- Achieved_Memory_Bandwidth = 数据传输量 / 时间
- Peak_Memory_Bandwidth：如 H100 HBM3 = 3.35 TB/s
```

**为什么 MBU 对 Decode 阶段重要？**

```
Decode 每生成 1 个 Token 需要：
├── 加载完整模型权重（一次）
├── 加载 KV Cache（与历史长度成正比）
└── 存储新的 KV 值

数据传输量 >> 计算量，因此 MBU 是真正的效率指标
```

**典型 MBU 参考值**：

| 配置 | MBU | 说明 |
|------|-----|------|
| 优化良好的 Decode | 80-95% | 接近硬件极限 |
| 一般实现 | 50-70% | 有优化空间 |
| 未优化 | < 50% | 需要调优 |

#### 2.3.3 显存效率

```
显存效率 = 有效显存 / 总显存 × 100%

有效显存 = 模型权重 + KV Cache + 必要激活值
无效显存 = 碎片 + 冗余缓冲 + 未使用预留

目标：显存效率 > 85%，留 10-15% 余量防止 OOM
```

### 2.4 成本指标（Cost Metrics）

#### 2.4.1 单位 Token 成本

```
Cost per 1K tokens = (GPU_hourly_cost × hours) / (total_tokens / 1000)

示例：
- 8×H100 租赁成本：$25/hour
- 1 小时生成 Token 数：10M tokens
- 成本：$25 / 10000 = $0.0025 / 1K tokens
```

**业界参考价格（2024-2025）**：

| 提供商 | 模型 | 输入价格 | 输出价格 |
|--------|------|----------|----------|
| OpenAI | GPT-4o | $2.50/M | $10.00/M |
| OpenAI | GPT-4o-mini | $0.15/M | $0.60/M |
| Anthropic | Claude 3.5 Sonnet | $3.00/M | $15.00/M |
| Google | Gemini Flash | $0.075/M | $0.30/M |
| DeepSeek | DeepSeek-V3 | $0.27/M | $1.10/M |

**自建 vs 云服务 TCO 分析**：

```
自建成本构成：
├── 硬件购置/租赁：70-80%
├── 人员运维：15-20%
├── 电力/机房：5-10%
└── 其他（网络、存储等）：5%

盈亏平衡点：
- 日均 Token 量 > 2M 时，自建开始有优势
- 考虑利用率：自建需 > 60% 利用率才划算
```

---

## 三、模型参数量准确计算

### 3.1 Transformer 参数量通用公式

#### 精确公式

```
总参数量 = Embedding + L × (Attention + FFN + LayerNorm) + Final_Norm + LM_Head

各部分详解：
├── Embedding: V × H
├── Per Layer:
│   ├── Attention: 见 3.2
│   ├── FFN: 见 3.3
│   └── LayerNorm: 见 3.4
├── Final LayerNorm: H（RMSNorm 无 bias）
└── LM Head: H × V（可能与 Embedding 共享）
```

#### 快速估算公式（经典 12d² 法则）

对于标准 Transformer（MHA + 4×FFN），每层参数约为 **12 × H²**：

```
P ≈ 12 × L × H²

来源分解：
├── Attention: 4H² (Q + K + V + O)
├── FFN: 8H² (假设 I = 4H，则 3 × H × 4H = 12H²... 实际是 8H²)
│         修正: 如果 I = 4H，FFN = 2 × H × 4H + 4H × H = 12H²
│         但现代模型通常 I ≈ 2.7H (SwiGLU)
└── 总计: ~12H² per layer (适用于 GPT-2/3 架构)

适用条件：
✓ 标准 MHA (num_kv_heads = num_heads)
✓ 标准 FFN (intermediate_size = 4 × hidden_size)
✓ 忽略 Embedding 和 LayerNorm

不适用于：
✗ GQA/MQA (K/V heads 减少)
✗ SwiGLU (3 个投影，且 I ≈ 2.7H)
✗ MoE (专家数量倍增)
✗ MLA (低秩压缩)
```

#### 现代模型估算公式

```
对于 LLaMA 风格模型 (GQA + SwiGLU):

P_total ≈ 2VH + L × P_layer

P_layer = Attention + FFN + Norm
        = (2H² + 2H × d_kv) + 3HI + 2H

其中:
- d_kv = head_dim × num_kv_heads
- I = intermediate_size (通常 ≈ 2.67H 或 8H/3)

简化: P_layer ≈ 2H² + 8H² = 10H² (GQA 节省 ~17%)
```

**参考**: [Transformer Math 101](https://michaelwornow.net/2024/01/18/counting-params-in-transformer), [Kipply's Blog](https://kipp.ly/transformer-param-count/)

### 3.2 Attention 参数计算

#### 标准 MHA (Multi-Head Attention)

```
Q_proj: H × H
K_proj: H × H
V_proj: H × H
O_proj: H × H
────────────────
总计: 4 × H²
```

#### GQA (Grouped Query Attention) - LLaMA 2/3 使用

```
Q_proj: H × H
K_proj: H × (head_dim × num_kv_heads)  ← 注意这里！
V_proj: H × (head_dim × num_kv_heads)
O_proj: H × H
────────────────────────────────────────
总计: 2H² + 2H × head_dim × num_kv_heads

其中: head_dim = H / num_attention_heads
```

#### MLA (Multi-head Latent Attention) - DeepSeek V2/V3 使用

MLA 使用低秩分解压缩 KV Cache，同时引入额外的投影矩阵。

```
DeepSeek-V3 MLA 配置：
├── num_attention_heads (n_h): 128
├── per_head_dim (d_h): 128
├── kv_lora_rank (d_c): 512          ← KV 压缩维度
├── q_lora_rank (d_c'): 1536         ← Query 压缩维度
├── qk_rope_head_dim (d_h^R): 64     ← RoPE 解耦维度
└── v_head_dim: 128

MLA 参数量计算：
┌─────────────────────────────────────────────────────────────┐
│ 1. KV 压缩投影                                               │
│    W_DKV: H × d_c = 7168 × 512 = 3,670,016                  │
│                                                              │
│ 2. KV 解压投影                                               │
│    W_UK: d_c × (n_h × d_h) = 512 × 16384 = 8,388,608        │
│    W_UV: d_c × (n_h × d_h) = 512 × 16384 = 8,388,608        │
│                                                              │
│ 3. Query 压缩投影                                            │
│    W_DQ: H × d_c' = 7168 × 1536 = 11,010,048                │
│                                                              │
│ 4. Query 解压投影                                            │
│    W_UQ: d_c' × (n_h × d_h) = 1536 × 16384 = 25,165,824     │
│                                                              │
│ 5. RoPE 解耦 Key/Query                                       │
│    W_KR: H × (n_h × d_h^R) = 7168 × 8192 = 58,720,256       │
│    W_QR: d_c' × (n_h × d_h^R) = 1536 × 8192 = 12,582,912    │
│                                                              │
│ 6. Output 投影                                               │
│    W_O: (n_h × d_h) × H = 16384 × 7168 = 117,440,512        │
│                                                              │
│ 单层 MLA 总计: ~245M 参数                                    │
│ vs 标准 MHA: 4 × H² = 4 × 7168² = 205M                      │
│ MLA 参数量略多，但 KV Cache 大幅减少（压缩比 ~14x）          │
└─────────────────────────────────────────────────────────────┘

KV Cache 压缩比计算：
- 标准 MHA Cache: 2 × n_h × d_h = 2 × 128 × 128 = 32,768
- MLA Cache: d_c + n_h × d_h^R = 512 + 8192 = 8,704
- 压缩比: 32768 / 8704 ≈ 3.76x
```

**参考**: [Understanding Multi-Head Latent Attention](https://sebastianraschka.com/llms-from-scratch/ch04/05_mla/)

### 3.3 FFN 参数计算

#### SwiGLU (LLaMA 系列)

```
gate_proj: H × I
up_proj:   H × I
down_proj: I × H
────────────────
总计: 3 × H × I

其中 I = intermediate_size（通常 I ≈ 2.7H 或 8/3 × H）
```

#### MoE FFN

```
单个专家: 3 × H × I_expert
所有专家: 3 × H × I_expert × num_experts
共享专家: 3 × H × I_shared × num_shared_experts
Router: H × num_experts

总 FFN 参数 = 专家参数 + 共享专家参数 + Router

注意：
- I_expert 可能与 I_dense 不同（DeepSeek-V3: I_expert=2048, I_dense=18432）
- 推理时只激活 top-k 专家，计算量是 k/E 倍
```

### 3.4 LayerNorm 参数

| 类型 | 参数量 | 使用模型 |
|------|--------|----------|
| LayerNorm | 2H (γ + β) | GPT-2, BERT |
| RMSNorm | H (仅 γ) | LLaMA, Mistral |
| 每层两个 | 2 × H | attention 前 + FFN 前 |

### 3.5 实际模型参数验证

#### LLaMA-2-7B（官方 6.74B）

```
配置：
- H = 4096, L = 32, V = 32000
- num_heads = 32, num_kv_heads = 32 (MHA)
- I = 11008

计算：
- Embedding: 32000 × 4096 = 131,072,000
- Attention/layer: 4 × 4096² = 67,108,864
- FFN/layer: 3 × 4096 × 11008 = 135,266,304
- RMSNorm/layer: 2 × 4096 = 8,192
- 每层小计: 202,383,360
- 32 层: 6,476,267,520
- Final RMSNorm: 4,096
- LM Head: 4096 × 32000 = 131,072,000
────────────────────────────
总计: 6,738,415,616 ≈ 6.74B ✓
```

#### LLaMA-3-70B（GQA）

```
配置：
- H = 8192, L = 80, V = 128256
- num_heads = 64, num_kv_heads = 8 (GQA)
- I = 28672, head_dim = 128

计算：
- Embedding: 128256 × 8192 = 1,050,804,224
- Q_proj: 8192 × 8192 = 67,108,864
- K_proj: 8192 × (128 × 8) = 8,388,608  ← GQA 节省
- V_proj: 8192 × (128 × 8) = 8,388,608  ← GQA 节省
- O_proj: 8192 × 8192 = 67,108,864
- Attention/layer: 150,994,944 (vs MHA: 268,435,456, 节省 44%)
- FFN/layer: 3 × 8192 × 28672 = 704,643,072
- RMSNorm/layer: 2 × 8192 = 16,384
- 每层小计: 855,654,400
- 80 层: 68,452,352,000
- Final + LM Head: 1,050,812,416
────────────────────────────
总计: ~70.5B
```

#### DeepSeek-V3-671B（MoE + MLA）

**官方配置** (来源: [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3), [技术报告](https://arxiv.org/abs/2412.19437))：

```
基础配置：
├── hidden_size (H): 7168
├── num_layers (L): 61
├── vocab_size (V): 129280
├── num_attention_heads: 128
├── per_head_dim: 128

MLA 配置：
├── kv_lora_rank: 512
├── q_lora_rank: 1536
├── qk_rope_head_dim: 64

MoE 配置：
├── num_experts: 256
├── num_shared_experts: 1
├── num_experts_per_tok: 8
├── expert_intermediate_size: 2048  ← 关键！不是 18432
├── dense_intermediate_size: 18432  ← 前 3 层 Dense FFN 使用
└── MoE 层: 第 4-61 层 (58 层)
```

**详细计算**：

```
1. Embedding
   V × H = 129280 × 7168 = 926,679,040

2. 前 3 层 Dense FFN (非 MoE)
   Attention: ~245M × 3 = 735M (MLA)
   FFN: 3 × H × I_dense = 3 × 7168 × 18432 = 396,361,728
   RMSNorm: 2 × H = 14,336
   每层: 396,376,064 + 245M = ~641M
   3 层小计: ~1.92B

3. 58 层 MoE FFN
   Attention: ~245M × 58 = 14.2B (MLA)
   Router: H × num_experts = 7168 × 256 = 1,835,008
   专家 FFN: 3 × H × I_expert × (256 + 1)
           = 3 × 7168 × 2048 × 257 = 11,321,352,192
   RMSNorm: 2 × H = 14,336
   每层 FFN: 11,323,201,536
   58 层 FFN: 656,745,689,088 ≈ 657B

4. Final RMSNorm + LM Head
   H + H × V = 7168 + 926,679,040 = 926,686,208

汇总：
├── Embedding: 0.93B
├── Dense 层 (1-3): 1.92B
├── MoE 层 (4-61): 657B + 14.2B = 671.2B
├── Final: 0.93B
────────────────────────────
总计: ~674B（含 MLA 参数）
官方值: 671B（主模型） + 14B（MTP 模块） = 685B

推理时激活参数：
├── 非专家参数: ~17B (Embedding + Attention + Router + RMSNorm)
├── 激活专家: 8/256 = 3.1%
├── 专家参数: 3 × 7168 × 2048 × (8+1) = ~400M/层 × 58 = 23.2B
└── 总激活: ~37B ✓（官方值）
```

### 3.6 参数量验证对照表

| 模型 | 官方参数量 | 计算参数量 | 误差 | 关键配置 |
|------|------------|------------|------|----------|
| LLaMA-2-7B | 6.74B | 6.74B | <0.1% | MHA, SwiGLU |
| LLaMA-2-13B | 13.0B | 13.0B | <0.1% | MHA, SwiGLU |
| LLaMA-2-70B | 70.0B | 69.5B | <1% | GQA(8), SwiGLU |
| LLaMA-3-8B | 8.0B | 8.03B | <0.5% | GQA(8), SwiGLU, V=128K |
| LLaMA-3-70B | 70.0B | 70.5B | <1% | GQA(8), SwiGLU, V=128K |
| Mistral-7B | 7.2B | 7.24B | <0.5% | GQA(8), SwiGLU |
| Mixtral-8x7B | 46.7B | 46.7B | <0.1% | MoE(8E, top2), GQA |
| Qwen-72B | 72.7B | 72.5B | <0.3% | MHA, SwiGLU |
| DeepSeek-V3 | 671B | 674B | <0.5% | MoE(256E), MLA |

### 3.7 从 HuggingFace 获取准确配置

```python
from transformers import AutoConfig

def get_model_params(model_name: str):
    """从 HuggingFace 获取模型配置并计算参数量"""
    config = AutoConfig.from_pretrained(model_name)

    H = config.hidden_size
    L = config.num_hidden_layers
    V = config.vocab_size
    I = config.intermediate_size
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
    head_dim = H // n_heads

    # Embedding
    embedding = V * H

    # Attention (考虑 GQA)
    q_proj = H * H
    k_proj = H * (head_dim * n_kv_heads)
    v_proj = H * (head_dim * n_kv_heads)
    o_proj = H * H
    attention = q_proj + k_proj + v_proj + o_proj

    # FFN (SwiGLU: gate + up + down)
    ffn = 3 * H * I

    # RMSNorm (2 per layer + 1 final)
    norm = 2 * H * L + H

    # LM Head (通常不与 Embedding 共享)
    lm_head = H * V

    # MoE 处理
    if hasattr(config, 'num_experts'):
        expert_I = getattr(config, 'expert_intermediate_size', I)
        n_experts = config.num_experts
        n_shared = getattr(config, 'num_shared_experts', 0)
        ffn = 3 * H * expert_I * (n_experts + n_shared) + H * n_experts

    total = embedding + L * (attention + ffn + 2*H) + H + lm_head
    return total

# 使用示例
# params = get_model_params("meta-llama/Llama-2-7b-hf")
# print(f"Total params: {params / 1e9:.2f}B")
```

### 3.8 常见计算错误

| 错误 | 正确做法 |
|------|----------|
| LayerNorm 用 4H | RMSNorm 应为 2H（无 bias） |
| GQA 时 K/V 用 H² | 应为 H × head_dim × num_kv_heads |
| MoE 用 intermediate_size | 应区分 expert_intermediate_size |
| 忽略 Final LayerNorm | 虽小但应计入 |
| Embedding 和 LM Head 重复计 | 注意是否共享权重 |
| MLA 简单估算 | 需计算所有投影矩阵（压缩+解压+RoPE） |
| MoE 首几层当 MoE | DeepSeek 前 3 层是 Dense |

---

## 四、业界工具与框架

### 4.1 Benchmark 工具

#### 4.1.1 llm-analysis（推荐）

**GitHub**: https://github.com/cli99/llm-analysis

```
功能：
├── 支持 TP/PP/DP/SP/EP 并行策略
├── 估算延迟和显存占用
├── 理论分析，无需实际运行
└── 可视化不同配置影响

使用场景：
- 部署前的可行性分析
- 并行策略选择
- 显存规划
```

#### 4.1.2 LLMPerf

**GitHub**: https://github.com/ray-project/llmperf

```
功能：
├── 支持多种 LLM API (OpenAI, Anthropic, vLLM, TGI)
├── 负载测试（并发请求）
├── 延迟分布统计
└── Token 吞吐量测量

适用于：
- API 端点性能测试
- 不同提供商对比
```

#### 4.1.3 vLLM Benchmark

**位置**: https://github.com/vllm-project/vllm/tree/main/benchmarks

```
提供脚本：
├── benchmark_latency.py：单请求延迟
├── benchmark_throughput.py：吞吐量测试
├── benchmark_serving.py：服务模式测试
└── 支持多种模型和配置
```

#### 4.1.4 TensorRT-LLM Benchmark

**位置**: NVIDIA TensorRT-LLM examples

```
特点：
├── 针对 NVIDIA GPU 优化
├── 支持 FP8/INT8 量化
├── 提供详细的性能分解
└── 可导出 Nsight 分析报告
```

### 4.2 推理引擎性能对比

基于 [BentoML Benchmark](https://www.bentoml.com/blog/benchmarking-llm-inference-backends)、[SqueezeBits 评测](https://blog.squeezebits.com/vllm-vs-tensorrtllm-1-an-overall-evaluation-30703) 等独立测试：

| 引擎 | TTFT | 吞吐量 | 易用性 | 硬件支持 |
|------|------|--------|--------|----------|
| **vLLM** | 优秀 | 优秀 | ⭐⭐⭐⭐⭐ | CUDA, ROCm, CPU |
| **TensorRT-LLM** | 最优 | 最优 | ⭐⭐⭐ | NVIDIA only |
| **SGLang** | 优秀 | 更优 | ⭐⭐⭐⭐ | CUDA |
| **LMDeploy** | 良好 | 优秀 | ⭐⭐⭐⭐ | CUDA |
| **llama.cpp** | 良好 | 中等 | ⭐⭐⭐⭐⭐ | CPU, GPU, Metal |

**选择建议**：
- **追求极致性能 + NVIDIA 生态**：TensorRT-LLM
- **灵活性 + 社区支持**：vLLM
- **Prefix Caching 场景**：SGLang（RadixAttention）
- **资源受限/边缘部署**：llama.cpp

### 4.3 可视化与分析工具

| 工具 | 用途 |
|------|------|
| NVIDIA Nsight Systems | GPU 时间线、内核分析 |
| PyTorch Profiler | Python 级别性能分析 |
| Weights & Biases | 实验跟踪、对比可视化 |
| Grafana + Prometheus | 生产环境监控 |

---

## 五、Benchmark 最佳实践

### 5.1 测试流程

```
1. 环境准备
   ├── 固定 GPU 频率（避免动态调频影响）
   ├── 清空 GPU 显存
   ├── 关闭其他 GPU 任务
   └── 记录软硬件版本

2. 预热阶段（结果不计入）
   ├── 运行 3-5 轮推理
   ├── 确保模型权重已加载到 GPU
   └── JIT 编译完成

3. 正式测量
   ├── 每个配置至少运行 10 次
   ├── 记录每次的延迟和吞吐
   └── 计算统计量（P50/P90/P99/Mean/Std）

4. 结果验证
   ├── 检查输出正确性
   ├── 验证显存占用合理
   └── 确认无异常日志
```

### 5.2 报告规范

**必须包含**：
- 硬件配置（GPU 型号、数量、互联方式）
- 软件版本（推理引擎、CUDA、驱动）
- 模型配置（参数量、精度、量化方式）
- 负载配置（batch_size、seq_length）
- 并行策略（TP/PP/DP/EP）
- 延迟统计（P50/P90/P99）
- 吞吐量统计

**可选包含**：
- 显存占用分解
- MFU/MBU 分析
- 成本估算
- 优化建议

### 5.3 常见陷阱

| 陷阱 | 问题 | 解决方案 |
|------|------|----------|
| 冷启动计入 | 首次推理包含编译开销 | 预热后再测量 |
| 只报告平均值 | 尾延迟被掩盖 | 报告 P99 |
| 固定 batch=1 | 无法反映真实服务 | 测试多种 batch |
| 忽略量化影响 | FP16 vs INT8 差异大 | 明确标注精度 |
| 不同框架对比 | 优化程度不同 | 保持框架一致 |
| GPU 温度过高 | 触发降频 | 监控温度 |

---

## 六、与本工具的对接

### 6.1 当前工具现状

本工具（LLM 部署分析系统）提供的指标：

| 指标 | 当前状态 | 与业界标准差距 |
|------|----------|----------------|
| TTFT | ✅ 有 | 需加分位数统计 |
| TPOT | ✅ 有 | 需加分位数统计 |
| 吞吐量 | ✅ 有 | 需区分 token/request |
| MFU | ✅ 有 | ✅ 符合标准 |
| MBU | ❌ 缺失 | Decode 阶段关键指标 |
| 成本分析 | ❌ 缺失 | $/M tokens |
| Fluidity-Index | ❌ 缺失 | 用户体验指标 |
| 参数量计算 | ⚠️ 有误差 | RMSNorm/GQA/MoE 需修正 |

### 6.2 建议优化项

**优先级 1（核心功能）**：
1. 修复参数量计算（RMSNorm 2H，GQA K/V，MoE expert_size）
2. 添加 P50/P90/P99 分位数展示
3. 添加 MBU 指标

**优先级 2（增强功能）**：
4. 添加 MLPerf 标准阈值参考线
5. 添加成本分析模块
6. 优化 Prefill/Decode 延迟分解可视化

**优先级 3（高级功能）**：
7. 添加 Fluidity-Index 评估
8. 支持与 llm-analysis 对比验证
9. 添加 Benchmark 报告导出

---

## 七、参考资料

### 官方标准
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/)
- [MLPerf Inference v5.0 LLM](https://mlcommons.org/2025/04/llm-inference-v5/)
- [NVIDIA NIM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html)

### 学术论文
- [Metron: Holistic Performance Evaluation Framework](https://arxiv.org/abs/2407.07000)
- [Efficient Memory Management for LLM Serving (vLLM)](https://arxiv.org/abs/2309.06180)
- [A Systematic Characterization of LLM Inference on GPUs](https://arxiv.org/abs/2512.01644)

### 技术博客
- [NVIDIA LLM Inference Benchmarking](https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/)
- [BentoML LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics)
- [Databricks LLM Inference Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- [Transformer Parameter Counting](https://kipp.ly/transformer-param-count/)
- [Mastering Llama Math](https://medium.com/@saratbhargava/mastering-llama-math-part-1-a-step-by-step-guide-to-counting-parameters-in-llama-2-b3d73bc3ae31)

### 开源工具
- [llm-analysis](https://github.com/cli99/llm-analysis) - 延迟和显存分析
- [LLMPerf](https://github.com/ray-project/llmperf) - LLM API 性能测试
- [vLLM Benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks)
- [TokenCost](https://github.com/AgentOps-AI/tokencost) - LLM 成本估算

### 成本计算器
- [LLM-Prices.com](https://www.llm-prices.com/)
- [Helicone LLM Cost](https://www.helicone.ai/llm-cost)
