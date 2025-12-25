/**
 * 柱状图 - 多方案指标对比
 */

import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import { PlanAnalysisResult } from '../../../utils/llmDeployment/types'

type MetricType = 'ttft' | 'tpot' | 'throughput' | 'mfu' | 'score'

interface MetricsBarChartProps {
  plans: PlanAnalysisResult[]
  metric: MetricType
  height?: number
}

const METRIC_CONFIG: Record<MetricType, {
  name: string
  unit: string
  accessor: (p: PlanAnalysisResult) => number
  color: string
  lowerIsBetter?: boolean
}> = {
  ttft: {
    name: 'TTFT',
    unit: 'ms',
    accessor: (p) => p.latency.prefill_total_latency_ms,
    color: '#1890ff',
    lowerIsBetter: true,
  },
  tpot: {
    name: 'TPOT',
    unit: 'ms',
    accessor: (p) => p.latency.decode_per_token_latency_ms,
    color: '#13c2c2',
    lowerIsBetter: true,
  },
  throughput: {
    name: '吞吐量',
    unit: 'tok/s',
    accessor: (p) => p.throughput.tokens_per_second,
    color: '#52c41a',
  },
  mfu: {
    name: 'MFU',
    unit: '%',
    accessor: (p) => p.throughput.model_flops_utilization * 100,
    color: '#faad14',
  },
  score: {
    name: '综合评分',
    unit: '分',
    accessor: (p) => p.score.overall_score,
    color: '#5E6AD2',
  },
}

export const MetricsBarChart: React.FC<MetricsBarChartProps> = ({
  plans,
  metric,
  height = 250,
}) => {
  const config = METRIC_CONFIG[metric]

  const option: EChartsOption = useMemo(() => {
    const feasiblePlans = plans.filter((p) => p.is_feasible)
    const labels = feasiblePlans.map((p) => p.plan.plan_id)
    const values = feasiblePlans.map((p) => config.accessor(p))

    // 找出最优值索引
    const bestIndex = config.lowerIsBetter
      ? values.indexOf(Math.min(...values))
      : values.indexOf(Math.max(...values))

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: unknown) => {
          const items = params as { name: string; value: number }[]
          const item = items[0]
          return `${item.name}<br/>${config.name}: ${item.value.toFixed(2)} ${config.unit}`
        },
      },
      grid: {
        left: 50,
        right: 20,
        top: 20,
        bottom: 40,
      },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: {
          rotate: labels.length > 4 ? 30 : 0,
          fontSize: 10,
          interval: 0,
        },
        axisLine: { lineStyle: { color: '#d9d9d9' } },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value',
        name: `${config.name} (${config.unit})`,
        nameTextStyle: { fontSize: 10, color: '#666' },
        axisLabel: { fontSize: 10 },
        axisLine: { show: false },
        splitLine: { lineStyle: { color: '#f0f0f0' } },
      },
      series: [
        {
          type: 'bar',
          data: values.map((v, i) => ({
            value: v,
            itemStyle: {
              color: i === bestIndex ? config.color : '#d9d9d9',
              borderRadius: [4, 4, 0, 0],
            },
          })),
          barMaxWidth: 40,
          label: {
            show: true,
            position: 'top',
            fontSize: 10,
            color: '#666',
            formatter: (params: unknown) => {
              const p = params as { value: number }
              return p.value.toFixed(1)
            },
          },
        },
      ],
    }
  }, [plans, metric, config])

  if (plans.filter((p) => p.is_feasible).length === 0) {
    return (
      <div
        style={{
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#999',
          fontSize: 12,
        }}
      >
        无可行方案
      </div>
    )
  }

  return (
    <ReactECharts
      option={option}
      style={{ height }}
      opts={{ renderer: 'svg' }}
    />
  )
}
