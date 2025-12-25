/**
 * 雷达图 - 四维评分对比
 */

import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import { PlanAnalysisResult } from '../../../utils/llmDeployment/types'

interface ScoreRadarChartProps {
  result: PlanAnalysisResult
  comparisonResults?: PlanAnalysisResult[]
  height?: number
}

const COLORS = ['#5E6AD2', '#52c41a', '#faad14', '#722ed1', '#eb2f96']

export const ScoreRadarChart: React.FC<ScoreRadarChartProps> = ({
  result,
  comparisonResults = [],
  height = 280,
}) => {
  const option = useMemo((): EChartsOption => {
    const allResults = [result, ...comparisonResults.slice(0, 4)]

    const seriesData = allResults.map((r, index) => ({
      name: r.plan.plan_id,
      value: [
        r.score.latency_score,
        r.score.throughput_score,
        r.score.efficiency_score,
        r.score.balance_score,
      ],
      symbol: 'circle',
      symbolSize: 6,
      lineStyle: {
        width: index === 0 ? 2 : 1,
      },
      areaStyle: {
        opacity: index === 0 ? 0.3 : 0.1,
      },
    }))

    return {
      tooltip: {
        trigger: 'item',
        formatter: (params: unknown) => {
          const p = params as { name: string; value: number[] }
          return `
            <div style="font-weight: bold; margin-bottom: 4px;">${p.name}</div>
            <div>延迟: ${p.value[0]?.toFixed(1) ?? '-'}分</div>
            <div>吞吐: ${p.value[1]?.toFixed(1) ?? '-'}分</div>
            <div>效率: ${p.value[2]?.toFixed(1) ?? '-'}分</div>
            <div>均衡: ${p.value[3]?.toFixed(1) ?? '-'}分</div>
          `
        },
      },
      legend: {
        show: allResults.length > 1,
        bottom: 0,
        itemWidth: 12,
        itemHeight: 8,
        textStyle: { fontSize: 10 },
      },
      radar: {
        indicator: [
          { name: '延迟', max: 100 },
          { name: '吞吐', max: 100 },
          { name: '效率', max: 100 },
          { name: '均衡', max: 100 },
        ],
        shape: 'polygon',
        radius: '65%',
        center: ['50%', allResults.length > 1 ? '45%' : '50%'],
        splitNumber: 4,
        axisName: {
          color: '#666',
          fontSize: 11,
        },
        splitLine: {
          lineStyle: { color: '#e8e8e8' },
        },
        splitArea: {
          areaStyle: {
            color: ['#fff', '#fafafa', '#f5f5f5', '#f0f0f0'],
          },
        },
        axisLine: {
          lineStyle: { color: '#d9d9d9' },
        },
      },
      series: [
        {
          type: 'radar',
          data: seriesData,
        },
      ],
      color: COLORS,
    }
  }, [result, comparisonResults])

  if (!result.is_feasible) {
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
        方案不可行，无法展示评分
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
