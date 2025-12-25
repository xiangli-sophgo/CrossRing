/**
 * 饼图 - 显存占用分解
 */

import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import { MemoryAnalysis } from '../../../utils/llmDeployment/types'

interface MemoryPieChartProps {
  memory: MemoryAnalysis
  height?: number
}

const COLORS = ['#5E6AD2', '#52c41a', '#faad14', '#ff7a45']

export const MemoryPieChart: React.FC<MemoryPieChartProps> = ({
  memory,
  height = 250,
}) => {
  const option: EChartsOption = useMemo(() => {
    const data = [
      { name: '模型参数', value: memory.model_memory_gb },
      { name: 'KV Cache', value: memory.kv_cache_memory_gb },
      { name: '激活值', value: memory.activation_memory_gb },
      { name: '其他开销', value: memory.overhead_gb },
    ].filter((d) => d.value > 0)

    const total = memory.total_per_chip_gb

    return {
      tooltip: {
        trigger: 'item',
        formatter: (params: unknown) => {
          const p = params as { name: string; value: number; percent: number }
          return `${p.name}<br/>${p.value.toFixed(2)} GB (${p.percent.toFixed(1)}%)`
        },
      },
      legend: {
        orient: 'vertical',
        right: 10,
        top: 'center',
        itemWidth: 10,
        itemHeight: 10,
        textStyle: { fontSize: 11 },
        formatter: (name: unknown) => {
          const n = name as string
          const item = data.find((d) => d.name === n)
          if (item) {
            return `${n}: ${item.value.toFixed(1)}GB`
          }
          return n
        },
      },
      series: [
        {
          type: 'pie',
          radius: ['45%', '70%'],
          center: ['35%', '50%'],
          avoidLabelOverlap: false,
          itemStyle: {
            borderRadius: 4,
            borderColor: '#fff',
            borderWidth: 2,
          },
          label: {
            show: false,
          },
          emphasis: {
            label: {
              show: true,
              fontSize: 12,
              fontWeight: 'bold',
            },
          },
          labelLine: {
            show: false,
          },
          data: data.map((d, i) => ({
            ...d,
            itemStyle: { color: COLORS[i % COLORS.length] },
          })),
        },
      ],
      graphic: [
        {
          type: 'text',
          left: '28%',
          top: '45%',
          style: {
            text: `${total.toFixed(1)}`,
            textAlign: 'center',
            fontSize: 18,
            fontWeight: 'bold',
            fill: '#333',
          },
        },
        {
          type: 'text',
          left: '28%',
          top: '55%',
          style: {
            text: 'GB/芯片',
            textAlign: 'center',
            fontSize: 10,
            fill: '#999',
          },
        },
      ],
      color: COLORS,
    }
  }, [memory])

  return (
    <ReactECharts
      option={option}
      style={{ height }}
      opts={{ renderer: 'svg' }}
    />
  )
}
