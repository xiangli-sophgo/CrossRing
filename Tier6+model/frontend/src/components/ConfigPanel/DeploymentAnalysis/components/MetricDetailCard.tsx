/**
 * 指标详情卡片组件
 * 参考 Notion 的简洁设计风格
 * 每个公式参数都有详细说明
 */

import React from 'react'
import { Typography } from 'antd'
import {
  FormulaCard,
  VariableList,
  CalculationSteps,
} from './FormulaDisplay'
import { PlanAnalysisResult } from '../../../../utils/llmDeployment/types'

const { Text } = Typography

export type MetricType = 'ttft' | 'tpot' | 'throughput' | 'mfu' | 'mbu' | 'cost' | 'percentiles' | 'bottleneck' | 'e2e' | 'chips'

interface MetricDetailCardProps {
  metric: MetricType
  result: PlanAnalysisResult
}

// 通用卡片样式
const cardStyle: React.CSSProperties = {
  marginTop: 12,
  marginBottom: 16,
  padding: '20px',
  background: '#fff',
  borderRadius: 12,
  border: '1px solid #e5e7eb',
}

// 标题样式
const titleStyle: React.CSSProperties = {
  fontSize: 16,
  fontWeight: 600,
  color: '#1f2937',
  marginBottom: 4,
}

// 副标题样式
const subtitleStyle: React.CSSProperties = {
  fontSize: 12,
  color: '#6b7280',
  marginBottom: 16,
}

// 小节标题样式
const sectionTitleStyle: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 600,
  color: '#374151',
  marginBottom: 10,
}

// 说明文字样式
const descStyle: React.CSSProperties = {
  fontSize: 12,
  color: '#6b7280',
  lineHeight: 1.6,
}

export const MetricDetailCard: React.FC<MetricDetailCardProps> = ({ metric, result }) => {
  const { plan, memory, latency, throughput } = result

  switch (metric) {
    case 'ttft':
      return (
        <div style={cardStyle}>
          <div style={titleStyle}>Time To First Token (TTFT)</div>
          <div style={subtitleStyle}>首Token延迟 · Prefill阶段核心指标</div>

          <div style={{ marginBottom: 20 }}>
            <div style={sectionTitleStyle}>指标定义</div>
            <div style={descStyle}>
              从请求发送到生成第一个输出Token的延迟时间。直接影响用户感知的响应速度，
              是评估LLM服务质量的关键SLO指标。MLPerf标准要求 P99 ≤ 450ms。
            </div>
          </div>

          <FormulaCard
            title="核心公式"
            tex={String.raw`\text{TTFT} = \frac{\max(T_{\text{compute}}, T_{\text{comm}})}{1 - \beta}`}
            description="TTFT由计算延迟和通信延迟中的较大值决定，并受流水线气泡影响"
            result={latency.prefill_total_latency_ms.toFixed(2)}
            unit="ms"
            resultColor="#1890ff"
          />

          <VariableList
            title="参数说明"
            variables={[
              {
                symbol: 'T_{\\text{compute}}',
                name: '计算延迟',
                description: 'Prefill阶段处理所有输入token的计算时间',
              },
              {
                symbol: 'T_{\\text{comm}}',
                name: '通信延迟',
                description: '张量并行/流水线并行的集合通信时间',
              },
              {
                symbol: '\\beta',
                name: '气泡比',
                description: 'PP并行导致的空闲时间占比，$\\beta = \\frac{PP-1}{MB+PP-1}$',
              },
              {
                symbol: '\\text{Params}',
                name: '模型参数量',
                description: '模型总参数数量',
              },
              {
                symbol: '\\text{SeqLen}',
                name: '输入序列长度',
                description: '输入token的数量',
              },
              {
                symbol: '\\text{Batch}',
                name: '批次大小',
                description: '同时处理的请求数量',
              },
              {
                symbol: '\\text{Peak}',
                name: '峰值算力',
                description: '芯片理论峰值计算能力 (TFLOPs)',
              },
              {
                symbol: '\\text{Util}',
                name: '利用率',
                description: '硬件实际可达的计算效率（MFU）',
              },
              {
                symbol: '\\text{TP}',
                name: '张量并行度',
                description: '模型在单层内切分到多少个设备',
              },
              {
                symbol: '\\text{PP}',
                name: '流水线并行度',
                description: '模型层按顺序切分到多少个阶段',
              },
              {
                symbol: '\\text{MB}',
                name: 'Micro-batch数',
                description: '流水线并行中的微批次数量',
              },
              {
                symbol: '\\text{Data}',
                name: '通信数据量',
                description: '每次集合通信传输的数据量',
              },
              {
                symbol: '\\text{BW}',
                name: '通信带宽',
                description: '芯片间互联带宽（NVLink/PCIe）',
              },
              {
                symbol: 't_{\\text{startup}}',
                name: '启动延迟',
                description: '通信操作的固定开销',
              },
            ]}
          />

          <CalculationSteps
            title="计算分解"
            steps={[
              {
                label: 'FLOPs',
                formula: '\\text{FLOPs} = 2 \\times \\text{Params} \\times \\text{SeqLen} \\times \\text{Batch}',
                value: ((2 * 70e9 * 1024 * 1) / 1e12).toFixed(2),
                unit: 'TFLOPs',
              },
              {
                label: '计算延迟',
                formula: 'T_{\\text{compute}} = \\frac{\\text{FLOPs}}{\\text{Peak} \\times \\text{Util} \\times \\text{TP}}',
                value: latency.prefill_compute_latency_ms.toFixed(2),
                unit: 'ms',
              },
              {
                label: '通信延迟',
                formula: 'T_{\\text{comm}} = \\sum\\left(\\frac{\\text{Data}}{\\text{BW}} + t_{\\text{startup}}\\right)',
                value: latency.prefill_comm_latency_ms.toFixed(2),
                unit: 'ms',
              },
              {
                label: '气泡比 β',
                formula: '\\beta = \\frac{\\text{PP} - 1}{\\text{MB} + \\text{PP} - 1}',
                value: (latency.pipeline_bubble_ratio * 100).toFixed(1),
                unit: '%',
              },
            ]}
          />
        </div>
      )

    case 'tpot':
      return (
        <div style={cardStyle}>
          <div style={titleStyle}>Time Per Output Token (TPOT)</div>
          <div style={subtitleStyle}>单Token延迟 · Decode阶段核心指标</div>

          <div style={{ marginBottom: 20 }}>
            <div style={sectionTitleStyle}>指标定义</div>
            <div style={descStyle}>
              Decode阶段生成每个输出Token的平均延迟。决定了文本生成的"打字速度"，
              TPOT越小用户看到的文本流越快。Decode阶段是memory-bound，受限于显存带宽。
            </div>
          </div>

          <FormulaCard
            title="核心公式"
            tex={String.raw`\text{TPOT} = \max(T_{\text{compute}}, T_{\text{memory}}) + T_{\text{comm}}`}
            description="Decode阶段每个token只需处理一次前向传播，瓶颈通常在显存带宽"
            result={latency.decode_per_token_latency_ms.toFixed(3)}
            unit="ms"
            resultColor="#13c2c2"
          />

          <VariableList
            title="参数说明"
            variables={[
              {
                symbol: 'T_{\\text{compute}}',
                name: '计算延迟',
                description: '单token前向传播的计算时间',
              },
              {
                symbol: 'T_{\\text{memory}}',
                name: '访存延迟',
                description: '读取模型权重和KV Cache的时间',
              },
              {
                symbol: 'T_{\\text{comm}}',
                name: '通信延迟',
                description: '张量并行的AllReduce通信时间',
              },
              {
                symbol: '\\text{Params}',
                name: '模型参数量',
                description: '模型总参数数量',
              },
              {
                symbol: '\\text{FLOPs/Token}',
                name: '每Token计算量',
                description: '$\\approx 2 \\times$ 模型参数量',
              },
              {
                symbol: '\\text{Peak}',
                name: '峰值算力',
                description: '芯片理论峰值计算能力 (TFLOPs)',
              },
              {
                symbol: '\\text{Util}',
                name: '利用率',
                description: '硬件实际可达的计算效率',
              },
              {
                symbol: '\\text{Model}',
                name: '模型大小',
                description: '模型权重占用的显存 (GB)',
              },
              {
                symbol: '\\text{KV}',
                name: 'KV缓存',
                description: '存储历史token的Key和Value',
              },
              {
                symbol: '\\text{BW}',
                name: 'HBM带宽',
                description: '显存带宽 (GB/s)',
              },
              {
                symbol: 'H',
                name: '隐藏维度',
                description: '模型的隐藏层大小',
              },
              {
                symbol: '\\text{BW}_{\\text{NVLink}}',
                name: 'NVLink带宽',
                description: '芯片间NVLink互联带宽',
              },
            ]}
          />

          <CalculationSteps
            title="计算分解"
            steps={[
              {
                label: 'FLOPs/Token',
                formula: '\\text{FLOPs/Token} \\approx 2 \\times \\text{Params}',
                value: (2 * 70e9 / 1e9).toFixed(0),
                unit: 'GFLOPs',
              },
              {
                label: '计算延迟',
                formula: 'T_{\\text{compute}} = \\frac{\\text{FLOPs/Token}}{\\text{Peak} \\times \\text{Util}}',
                value: latency.decode_compute_latency_ms.toFixed(3),
                unit: 'ms',
              },
              {
                label: '访存延迟',
                formula: 'T_{\\text{memory}} = \\frac{\\text{Model} + \\text{KV}}{\\text{BW}}',
                value: (memory.model_memory_gb / 3.35).toFixed(2),
                unit: 'ms',
              },
              {
                label: '通信延迟',
                formula: 'T_{\\text{comm}} = \\frac{2 \\times H}{\\text{BW}_{\\text{NVLink}}}',
                value: latency.decode_comm_latency_ms.toFixed(3),
                unit: 'ms',
              },
            ]}
          />
        </div>
      )

    case 'throughput':
      return (
        <div style={cardStyle}>
          <div style={titleStyle}>Throughput (吞吐量)</div>
          <div style={subtitleStyle}>系统处理能力 · 经济性核心指标</div>

          <div style={{ marginBottom: 20 }}>
            <div style={sectionTitleStyle}>指标定义</div>
            <div style={descStyle}>
              每秒生成的Token数量，衡量系统处理能力的核心指标。吞吐量越高，
              单位时间能服务的请求越多，成本效益越好。是容量规划和成本计算的基础。
            </div>
          </div>

          <FormulaCard
            title="核心公式"
            tex={String.raw`\text{Throughput} = \frac{\text{Batch} \times N_{\text{output}}}{T_{\text{e2e}}}`}
            description="吞吐量 = 批次大小 × 输出token数 / 端到端延迟"
            result={throughput.tokens_per_second.toFixed(0)}
            unit="tok/s"
            resultColor="#52c41a"
          />

          <VariableList
            title="参数说明"
            variables={[
              {
                symbol: '\\text{Batch}',
                name: '批次大小',
                description: '同时处理的请求数量',
              },
              {
                symbol: 'N_{\\text{output}}',
                name: '输出Token数',
                description: '每个请求生成的token数量',
              },
              {
                symbol: 'T_{\\text{e2e}}',
                name: '端到端延迟',
                description: 'TTFT + TPOT × 输出长度',
              },
              {
                symbol: '\\text{TTFT}',
                name: '首Token延迟',
                description: 'Prefill阶段延迟',
              },
              {
                symbol: '\\text{TPOT}',
                name: '每Token延迟',
                description: 'Decode阶段单token延迟',
              },
              {
                symbol: '\\text{Peak FLOPs}',
                name: '峰值算力',
                description: '硬件理论峰值计算能力',
              },
              {
                symbol: '\\text{FLOPs/Token}',
                name: '每Token计算量',
                description: '$\\approx 2 \\times$ 模型参数量',
              },
            ]}
          />

          <CalculationSteps
            title="计算分解"
            steps={[
              {
                label: '端到端延迟',
                formula: 'T_{\\text{e2e}} = \\text{TTFT} + \\text{TPOT} \\times N_{\\text{output}}',
                value: latency.end_to_end_latency_ms.toFixed(1),
                unit: 'ms',
              },
              {
                label: '理论峰值吞吐',
                formula: '\\text{Max} = \\frac{\\text{Peak FLOPs}}{\\text{FLOPs/Token}}',
                value: throughput.theoretical_max_throughput.toFixed(0),
                unit: 'tok/s',
              },
            ]}
          />
        </div>
      )

    case 'mfu':
      return (
        <div style={cardStyle}>
          <div style={titleStyle}>Model FLOPs Utilization (MFU)</div>
          <div style={subtitleStyle}>算力利用率 · Prefill效率指标</div>

          <div style={{ marginBottom: 20 }}>
            <div style={sectionTitleStyle}>指标定义</div>
            <div style={descStyle}>
              实际用于模型计算的算力占硬件峰值算力的比例。MFU越高说明硬件利用越充分。
              Prefill阶段是compute-bound，MFU是衡量其效率的关键指标。
              参考值：Prefill 40-60%（优秀），Decode 20-40%（正常，因memory-bound）。
            </div>
          </div>

          <FormulaCard
            title="核心公式"
            tex={String.raw`\text{MFU} = \frac{\text{Achieved FLOPs}}{\text{Peak FLOPs}} \times 100\%`}
            description="实际算力 / 理论峰值算力"
            result={(throughput.model_flops_utilization * 100).toFixed(2)}
            unit="%"
            resultColor="#faad14"
          />

          <VariableList
            title="参数说明"
            variables={[
              {
                symbol: '\\text{Achieved}',
                name: '实际算力',
                description: 'Throughput × FLOPs per Token',
              },
              {
                symbol: '\\text{Peak}',
                name: '峰值算力',
                description: '芯片数 × 单芯片峰值算力',
              },
              {
                symbol: '\\text{Throughput}',
                name: '吞吐量',
                description: '每秒生成的token数',
              },
              {
                symbol: '\\text{FLOPs/Token}',
                name: '每Token计算量',
                description: '$\\approx 2 \\times$ 模型参数量',
              },
              {
                symbol: 'N_{\\text{chips}}',
                name: '芯片数',
                description: '部署使用的芯片总数',
              },
              {
                symbol: '\\text{Chip TFLOPs}',
                name: '单芯片算力',
                description: '单个芯片的理论峰值算力',
              },
            ]}
          />

          <CalculationSteps
            title="计算分解"
            steps={[
              {
                label: '实际算力',
                formula: '\\text{Achieved} = \\text{Throughput} \\times \\text{FLOPs/Token}',
                value: (throughput.tokens_per_second * 2 * 70e9 / 1e12).toFixed(2),
                unit: 'TFLOPs',
              },
              {
                label: '峰值算力',
                formula: '\\text{Peak} = N_{\\text{chips}} \\times \\text{Chip TFLOPs}',
                value: `${plan.total_chips} × Peak`,
              },
            ]}
          />
        </div>
      )

    case 'mbu':
      const achievedBW = (memory.model_memory_gb + memory.kv_cache_memory_gb * 0.5) / (latency.decode_per_token_latency_ms / 1000)
      return (
        <div style={cardStyle}>
          <div style={titleStyle}>Memory Bandwidth Utilization (MBU)</div>
          <div style={subtitleStyle}>带宽利用率 · Decode效率指标</div>

          <div style={{ marginBottom: 20 }}>
            <div style={sectionTitleStyle}>指标定义</div>
            <div style={descStyle}>
              实际显存带宽使用量占峰值带宽的比例。Decode阶段是memory-bound，
              MBU是衡量其效率的核心指标。MBU越高，TPOT越接近理论极限。
            </div>
          </div>

          <FormulaCard
            title="核心公式"
            tex={String.raw`\text{MBU} = \frac{\text{Achieved BW}}{\text{Peak BW}} \times 100\%`}
            description="实际带宽利用 / 硬件峰值带宽"
            result={(throughput.memory_bandwidth_utilization * 100).toFixed(1)}
            unit="%"
            resultColor="#722ed1"
          />

          <VariableList
            title="参数说明"
            variables={[
              {
                symbol: '\\text{Achieved BW}',
                name: '实际带宽',
                description: '(模型大小 + KV Cache) / TPOT',
              },
              {
                symbol: '\\text{Peak BW}',
                name: '峰值带宽',
                description: '芯片HBM带宽，如H100 = 3.35 TB/s',
              },
              {
                symbol: '\\text{Model}',
                name: '模型大小',
                description: '模型权重占用的显存',
              },
              {
                symbol: '\\text{KV Cache}',
                name: 'KV缓存',
                description: '存储历史token的Key/Value',
              },
              {
                symbol: '\\text{TPOT}',
                name: '每Token延迟',
                description: 'Decode阶段单token生成时间',
              },
              {
                symbol: '\\text{Data}',
                name: '数据量',
                description: '每token需要读取的总数据量',
              },
            ]}
          />

          <CalculationSteps
            title="计算分解"
            steps={[
              {
                label: '每Token数据量',
                formula: '\\text{Data} = \\text{Model} + \\text{KV Cache}',
                value: `${memory.model_memory_gb.toFixed(2)} + ${memory.kv_cache_memory_gb.toFixed(2)}`,
                unit: 'GB',
              },
              {
                label: '实际带宽',
                formula: '\\text{Achieved BW} = \\frac{\\text{Data}}{\\text{TPOT}}',
                value: achievedBW.toFixed(0),
                unit: 'GB/s',
              },
            ]}
          />
        </div>
      )

    case 'cost':
      const costData = result.cost
      if (!costData) return null
      return (
        <div style={cardStyle}>
          <div style={titleStyle}>Cost Analysis (成本分析)</div>
          <div style={subtitleStyle}>经济性指标 · 每百万Token成本</div>

          <div style={{ marginBottom: 20 }}>
            <div style={sectionTitleStyle}>指标定义</div>
            <div style={descStyle}>
              每百万Token的推理成本，是衡量部署经济性的核心指标。
              成本按Prefill/Decode时间比例分配到输入/输出Token，输出通常比输入贵3-5倍。
            </div>
          </div>

          <FormulaCard
            title="核心公式"
            tex={String.raw`\text{Cost} = \frac{\text{Hardware Cost/h} \times 10^6}{\text{Tokens/hour}}`}
            description="每小时硬件成本 × 100万 / 每小时处理的Token数"
            result={`$${costData.cost_per_million_tokens.toFixed(4)}`}
            unit="/M tokens"
            resultColor="#fa541c"
          />

          <VariableList
            title="参数说明"
            variables={[
              {
                symbol: '\\text{Hardware Cost}',
                name: '硬件成本',
                description: '云服务商每小时租用价格',
              },
              {
                symbol: '\\text{Tokens/hour}',
                name: '每小时Token数',
                description: 'Throughput × 3600',
              },
              {
                symbol: 'N_{\\text{chips}}',
                name: '芯片数量',
                description: '部署使用的芯片总数',
              },
              {
                symbol: '\\text{Throughput}',
                name: '吞吐量',
                description: '每秒生成的token数',
              },
              {
                symbol: '\\text{Price/chip}',
                name: '单芯片价格',
                description: '每芯片每小时租用成本',
              },
            ]}
          />

          <CalculationSteps
            title="计算分解"
            steps={[
              {
                label: '硬件成本',
                formula: '\\text{Cost/h} = \\text{Price/chip} \\times N_{\\text{chips}}',
                value: `$${costData.hardware_cost_per_hour.toFixed(2)} × ${plan.total_chips}`,
                unit: `= $${costData.total_hardware_cost_per_hour.toFixed(2)}/h`,
              },
              {
                label: 'Tokens/hour',
                formula: '\\text{Throughput} \\times 3600',
                value: (throughput.tokens_per_second * 3600).toExponential(2),
              },
            ]}
          />

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10, marginTop: 16 }}>
            <div style={{
              padding: '14px 12px',
              background: '#fff7e6',
              borderRadius: 10,
              textAlign: 'center',
            }}>
              <div style={{ fontSize: 11, color: '#ad6800', marginBottom: 4 }}>综合成本</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#fa541c' }}>
                ${costData.cost_per_million_tokens.toFixed(4)}
              </div>
              <div style={{ fontSize: 10, color: '#ad6800' }}>/M tokens</div>
            </div>
            <div style={{
              padding: '14px 12px',
              background: '#f6ffed',
              borderRadius: 10,
              textAlign: 'center',
            }}>
              <div style={{ fontSize: 11, color: '#389e0d', marginBottom: 4 }}>输入成本</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#52c41a' }}>
                ${costData.input_cost_per_million_tokens.toFixed(4)}
              </div>
              <div style={{ fontSize: 10, color: '#389e0d' }}>/M tokens</div>
            </div>
            <div style={{
              padding: '14px 12px',
              background: '#fff1f0',
              borderRadius: 10,
              textAlign: 'center',
            }}>
              <div style={{ fontSize: 11, color: '#cf1322', marginBottom: 4 }}>输出成本</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#f5222d' }}>
                ${costData.output_cost_per_million_tokens.toFixed(4)}
              </div>
              <div style={{ fontSize: 10, color: '#cf1322' }}>/M tokens</div>
            </div>
          </div>

          <div style={{
            marginTop: 12,
            padding: '10px 14px',
            background: '#f5f5f5',
            borderRadius: 8,
            fontSize: 13,
            color: '#1f2937',
            textAlign: 'center',
          }}>
            效率: <strong style={{ color: '#fa541c' }}>{costData.tokens_per_dollar.toExponential(2)}</strong> tokens/$
          </div>
        </div>
      )

    case 'percentiles':
      const ttftP = latency.ttft_percentiles
      const tpotP = latency.tpot_percentiles
      return (
        <div style={cardStyle}>
          <div style={titleStyle}>Latency Percentiles (延迟分位数)</div>
          <div style={subtitleStyle}>SLO关键指标 · P50/P90/P99</div>

          <div style={{ marginBottom: 20 }}>
            <div style={sectionTitleStyle}>指标定义</div>
            <div style={descStyle}>
              延迟的统计分布。P99表示99%请求的延迟都低于此值，是SLO的关键指标。
              MLPerf Inference标准要求：TTFT P99 ≤ 450ms，TPOT P99 ≤ 40ms。
            </div>
          </div>

          <VariableList
            title="分位数说明"
            variables={[
              {
                symbol: 'P_{50}',
                name: '中位数',
                description: '50%请求低于此延迟，代表典型用户体验',
              },
              {
                symbol: 'P_{90}',
                name: '90分位',
                description: '90%请求低于此延迟，包含大部分用户',
              },
              {
                symbol: 'P_{99}',
                name: '99分位（尾部延迟）',
                description: '99%请求低于此延迟，SLO的关键指标',
              },
            ]}
          />

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
            {/* TTFT 分位数 */}
            <div style={{ padding: 16, background: '#f0f5ff', borderRadius: 10 }}>
              <Text strong style={{ fontSize: 14, color: '#2f54eb', display: 'block', marginBottom: 12 }}>
                TTFT 分位数
              </Text>
              {ttftP && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', background: '#fff', borderRadius: 6 }}>
                    <span style={{ fontSize: 12, color: '#6b7280' }}>P50</span>
                    <span style={{ fontSize: 14, fontWeight: 600 }}>{ttftP.p50.toFixed(1)} ms</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', background: '#fff', borderRadius: 6 }}>
                    <span style={{ fontSize: 12, color: '#6b7280' }}>P90</span>
                    <span style={{ fontSize: 14, fontWeight: 600 }}>{ttftP.p90.toFixed(1)} ms</span>
                  </div>
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    padding: '8px 12px',
                    background: ttftP.p99 > 450 ? '#fff2f0' : '#f6ffed',
                    borderRadius: 6,
                    border: `1px solid ${ttftP.p99 > 450 ? '#ffa39e' : '#b7eb8f'}`,
                  }}>
                    <span style={{ fontSize: 12, color: '#6b7280' }}>P99</span>
                    <span style={{ fontSize: 14, fontWeight: 600, color: ttftP.p99 > 450 ? '#f5222d' : '#52c41a' }}>
                      {ttftP.p99.toFixed(1)} ms
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* TPOT 分位数 */}
            <div style={{ padding: 16, background: '#e6fffb', borderRadius: 10 }}>
              <Text strong style={{ fontSize: 14, color: '#13c2c2', display: 'block', marginBottom: 12 }}>
                TPOT 分位数
              </Text>
              {tpotP && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', background: '#fff', borderRadius: 6 }}>
                    <span style={{ fontSize: 12, color: '#6b7280' }}>P50</span>
                    <span style={{ fontSize: 14, fontWeight: 600 }}>{tpotP.p50.toFixed(2)} ms</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', background: '#fff', borderRadius: 6 }}>
                    <span style={{ fontSize: 12, color: '#6b7280' }}>P90</span>
                    <span style={{ fontSize: 14, fontWeight: 600 }}>{tpotP.p90.toFixed(2)} ms</span>
                  </div>
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    padding: '8px 12px',
                    background: tpotP.p99 > 40 ? '#fff2f0' : '#f6ffed',
                    borderRadius: 6,
                    border: `1px solid ${tpotP.p99 > 40 ? '#ffa39e' : '#b7eb8f'}`,
                  }}>
                    <span style={{ fontSize: 12, color: '#6b7280' }}>P99</span>
                    <span style={{ fontSize: 14, fontWeight: 600, color: tpotP.p99 > 40 ? '#f5222d' : '#52c41a' }}>
                      {tpotP.p99.toFixed(2)} ms
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div style={{
            marginTop: 16,
            padding: '10px 14px',
            background: '#f0f5ff',
            borderRadius: 8,
            fontSize: 12,
            color: '#2f54eb',
            textAlign: 'center',
          }}>
            📊 MLPerf SLO标准: TTFT P99 ≤ 450ms, TPOT P99 ≤ 40ms
          </div>
        </div>
      )

    case 'bottleneck':
      const bottleneckInfo: Record<string, { name: string; color: string; desc: string; solution: string }> = {
        compute: {
          name: '计算瓶颈',
          color: '#faad14',
          desc: '算力不足，GPU计算单元成为限制因素',
          solution: '增加TP并行度，或使用更强算力的芯片',
        },
        memory: {
          name: '访存瓶颈',
          color: '#1890ff',
          desc: '显存带宽不足，数据读取速度限制了计算',
          solution: '减小batch size，或使用更高带宽的芯片',
        },
        communication: {
          name: '通信瓶颈',
          color: '#722ed1',
          desc: '芯片间通信延迟过高，集合通信成为限制因素',
          solution: '减小TP/PP并行度，或使用更高带宽的互联',
        },
        pipeline_bubble: {
          name: '流水线气泡',
          color: '#13c2c2',
          desc: '流水线并行导致的空闲时间过长',
          solution: '增加micro-batch数量，或减小PP并行度',
        },
      }
      const info = bottleneckInfo[latency.bottleneck_type] || { name: '未知', color: '#666', desc: '', solution: '' }

      return (
        <div style={{ ...cardStyle, background: '#fffbe6', border: '1px solid #ffe58f' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
            <div style={{
              width: 40,
              height: 40,
              borderRadius: 10,
              background: info.color,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#fff',
              fontWeight: 700,
              fontSize: 16,
            }}>
              ⚠️
            </div>
            <div>
              <div style={titleStyle}>性能瓶颈分析</div>
              <div style={{
                display: 'inline-block',
                padding: '2px 8px',
                background: info.color,
                color: '#fff',
                borderRadius: 4,
                fontSize: 12,
              }}>
                {info.name}
              </div>
            </div>
          </div>

          <div style={{ ...descStyle, background: '#fff' }}>
            <div style={{ marginBottom: 8 }}>
              <strong style={{ color: '#ad6800' }}>瓶颈原因：</strong>
              {info.desc}
            </div>
            <div>
              <strong style={{ color: '#ad6800' }}>详细信息：</strong>
              {latency.bottleneck_details}
            </div>
          </div>

          <div style={{
            padding: '12px 16px',
            background: '#fff',
            borderRadius: 8,
            marginBottom: 16,
            borderLeft: `4px solid ${info.color}`,
          }}>
            <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>优化建议</div>
            <div style={{ fontSize: 14, color: info.color, fontWeight: 500 }}>{info.solution}</div>
          </div>

          <CalculationSteps
            title="延迟分解"
            steps={[
              { label: 'Prefill 计算', value: latency.prefill_compute_latency_ms.toFixed(2), unit: 'ms' },
              { label: 'Prefill 通信', value: latency.prefill_comm_latency_ms.toFixed(2), unit: 'ms' },
              { label: 'Decode 计算', value: latency.decode_compute_latency_ms.toFixed(3), unit: 'ms' },
              { label: 'Decode 通信', value: latency.decode_comm_latency_ms.toFixed(3), unit: 'ms' },
              { label: '流水线气泡比', value: (latency.pipeline_bubble_ratio * 100).toFixed(1), unit: '%' },
            ]}
          />
        </div>
      )

    case 'e2e':
      return (
        <div style={cardStyle}>
          <div style={titleStyle}>End-to-End Latency (端到端延迟)</div>
          <div style={subtitleStyle}>完整请求耗时 · 用户感知延迟</div>

          <div style={{ marginBottom: 20 }}>
            <div style={sectionTitleStyle}>指标定义</div>
            <div style={descStyle}>
              从发送请求到接收完整响应的总时间。包括Prefill阶段（TTFT）和Decode阶段的全部时间。
              是用户感知的实际响应时间，直接影响用户体验。
            </div>
          </div>

          <FormulaCard
            title="核心公式"
            tex={String.raw`T_{\text{e2e}} = \text{TTFT} + \text{TPOT} \times N_{\text{output}}`}
            description="端到端延迟 = 首Token延迟 + 每Token延迟 × 输出Token数"
            result={(latency.end_to_end_latency_ms / 1000).toFixed(2)}
            unit="秒"
            resultColor="#eb2f96"
          />

          <VariableList
            title="参数说明"
            variables={[
              {
                symbol: '\\text{TTFT}',
                name: '首Token延迟',
                description: 'Prefill阶段处理输入的时间',
              },
              {
                symbol: '\\text{TPOT}',
                name: '每Token延迟',
                description: 'Decode阶段每个token的生成时间',
              },
              {
                symbol: 'N_{\\text{output}}',
                name: '输出Token数',
                description: '生成的输出token数量',
              },
              {
                symbol: '\\text{Prefill}',
                name: 'Prefill阶段',
                description: '处理输入序列，生成KV Cache',
              },
              {
                symbol: '\\text{Decode}',
                name: 'Decode阶段',
                description: '逐token生成输出',
              },
            ]}
          />

          <CalculationSteps
            title="延迟分解"
            steps={[
              {
                label: 'Prefill占比',
                formula: '\\frac{\\text{TTFT}}{T_{\\text{e2e}}} \\times 100\\%',
                value: (latency.prefill_total_latency_ms / latency.end_to_end_latency_ms * 100).toFixed(1),
                unit: '%',
              },
              {
                label: 'TTFT (Prefill)',
                formula: '\\text{TTFT} = \\frac{\\max(T_{\\text{compute}}, T_{\\text{comm}})}{1 - \\beta}',
                value: latency.prefill_total_latency_ms.toFixed(2),
                unit: 'ms',
              },
              {
                label: 'Decode总时间',
                formula: '\\text{TPOT} \\times N_{\\text{output}}',
                value: (latency.end_to_end_latency_ms - latency.prefill_total_latency_ms).toFixed(1),
                unit: 'ms',
              },
            ]}
          />
        </div>
      )

    case 'chips':
      const { dp, tp, pp, ep } = plan.parallelism
      return (
        <div style={cardStyle}>
          <div style={titleStyle}>Chip Configuration (芯片配置)</div>
          <div style={subtitleStyle}>资源利用 · 并行策略分解</div>

          <div style={{ marginBottom: 20 }}>
            <div style={sectionTitleStyle}>指标定义</div>
            <div style={descStyle}>
              总芯片数由并行策略决定：DP × TP × PP × EP = 总芯片数。
              合理的芯片配置需要平衡延迟、吞吐和成本。
            </div>
          </div>

          <FormulaCard
            title="核心公式"
            tex={String.raw`N_{\text{chips}} = \text{DP} \times \text{TP} \times \text{PP} \times \text{EP}`}
            description="总芯片数 = 数据并行 × 张量并行 × 流水线并行 × 专家并行"
            result={plan.total_chips}
            unit="chips"
            resultColor="#2f54eb"
          />

          <VariableList
            title="并行维度说明"
            variables={[
              {
                symbol: '\\text{DP}',
                name: '数据并行',
                description: '独立处理不同batch的副本数，增加吞吐',
              },
              {
                symbol: '\\text{TP}',
                name: '张量并行',
                description: '单层内切分到多设备，减少单卡显存',
              },
              {
                symbol: '\\text{PP}',
                name: '流水线并行',
                description: '层间切分，适合超大模型',
              },
              {
                symbol: '\\text{EP}',
                name: '专家并行',
                description: 'MoE模型的专家分布',
              },
            ]}
          />

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginTop: 16 }}>
            {[
              { label: 'DP', value: dp, color: '#1890ff' },
              { label: 'TP', value: tp, color: '#52c41a' },
              { label: 'PP', value: pp, color: '#fa8c16' },
              { label: 'EP', value: ep, color: '#722ed1' },
            ].map((item) => (
              <div key={item.label} style={{
                padding: 12,
                background: `${item.color}10`,
                borderRadius: 8,
                textAlign: 'center',
              }}>
                <div style={{ fontSize: 11, color: item.color }}>{item.label}</div>
                <div style={{ fontSize: 24, fontWeight: 700, color: item.color }}>{item.value}</div>
              </div>
            ))}
          </div>
        </div>
      )

    default:
      return null
  }
}

export default MetricDetailCard
