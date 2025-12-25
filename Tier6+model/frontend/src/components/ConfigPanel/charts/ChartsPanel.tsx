/**
 * 图表面板 - 整合所有图表的容器组件
 */

import React, { useState } from 'react'
import { Typography, Select, Row, Col, Empty } from 'antd'
import { ScoreRadarChart } from './ScoreRadarChart'
import { MetricsBarChart } from './MetricsBarChart'
import { MemoryPieChart } from './MemoryPieChart'
import { RooflineChart } from './RooflineChart'
import {
  PlanAnalysisResult,
  HardwareConfig,
  LLMModelConfig,
} from '../../../utils/llmDeployment/types'

const { Text } = Typography

interface ChartsPanelProps {
  result: PlanAnalysisResult | null
  topKPlans: PlanAnalysisResult[]
  hardware: HardwareConfig
  model: LLMModelConfig
}

type MetricType = 'score' | 'ttft' | 'tpot' | 'throughput' | 'mfu'

const chartCardStyle: React.CSSProperties = {
  background: '#fff',
  borderRadius: 10,
  padding: 12,
  border: '1px solid #f0f0f0',
}

const chartTitleStyle: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 600,
  color: '#1a1a1a',
  marginBottom: 8,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
}

export const ChartsPanel: React.FC<ChartsPanelProps> = ({
  result,
  topKPlans,
  hardware,
  model,
}) => {
  const [selectedMetric, setSelectedMetric] = useState<MetricType>('score')

  if (!result) {
    return (
      <Empty
        description="请先运行分析以查看图表"
        style={{ marginTop: 40 }}
      />
    )
  }

  const metricOptions = [
    { value: 'score', label: '综合评分' },
    { value: 'ttft', label: 'TTFT (ms)' },
    { value: 'tpot', label: 'TPOT (ms)' },
    { value: 'throughput', label: '吞吐量 (tok/s)' },
    { value: 'mfu', label: 'MFU (%)' },
  ]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {/* 雷达图 */}
      <div style={chartCardStyle}>
        <div style={chartTitleStyle}>
          <Text strong>四维评分分析</Text>
          {topKPlans.length > 1 && (
            <Text type="secondary" style={{ fontSize: 11 }}>
              对比 Top-{Math.min(5, topKPlans.length)} 方案
            </Text>
          )}
        </div>
        <ScoreRadarChart
          result={result}
          comparisonResults={topKPlans.slice(1, 5)}
          height={260}
        />
      </div>

      {/* 柱状图 */}
      <div style={chartCardStyle}>
        <div style={chartTitleStyle}>
          <Text strong>多方案对比</Text>
          <Select
            size="small"
            value={selectedMetric}
            onChange={setSelectedMetric}
            options={metricOptions}
            style={{ width: 130 }}
          />
        </div>
        <MetricsBarChart
          plans={topKPlans}
          metric={selectedMetric}
          height={220}
        />
      </div>

      {/* 饼图和 Roofline 图 */}
      <Row gutter={12}>
        <Col xs={24} lg={12}>
          <div style={chartCardStyle}>
            <div style={chartTitleStyle}>
              <Text strong>显存占用分解</Text>
              <Text type="secondary" style={{ fontSize: 11 }}>
                {result.memory.is_memory_sufficient ? '✓ 显存充足' : '⚠ 显存不足'}
              </Text>
            </div>
            <MemoryPieChart memory={result.memory} height={220} />
          </div>
        </Col>
        <Col xs={24} lg={12}>
          <div style={{ ...chartCardStyle, marginTop: window.innerWidth < 992 ? 12 : 0 }}>
            <div style={chartTitleStyle}>
              <Text strong>Roofline 性能分析</Text>
              <Text type="secondary" style={{ fontSize: 11 }}>
                {result.latency.bottleneck_type === 'memory'
                  ? '带宽受限'
                  : result.latency.bottleneck_type === 'compute'
                  ? '算力受限'
                  : '通信受限'}
              </Text>
            </div>
            <RooflineChart
              result={result}
              hardware={hardware}
              model={model}
              comparisonResults={topKPlans.slice(1, 4)}
              height={220}
            />
          </div>
        </Col>
      </Row>
    </div>
  )
}
