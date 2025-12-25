/**
 * Roofline 图 - 性能瓶颈分析
 */

import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import {
  PlanAnalysisResult,
  HardwareConfig,
  LLMModelConfig,
} from '../../../utils/llmDeployment/types'

interface RooflineChartProps {
  result: PlanAnalysisResult
  hardware: HardwareConfig
  model: LLMModelConfig
  comparisonResults?: PlanAnalysisResult[]
  height?: number
}

const COLORS = ['#5E6AD2', '#52c41a', '#faad14', '#722ed1']

export const RooflineChart: React.FC<RooflineChartProps> = ({
  result,
  hardware,
  model: _model,
  comparisonResults = [],
  height = 280,
}) => {
  const option: EChartsOption = useMemo(() => {
    const peakTflops = hardware.chip.compute_tflops_fp16
    const memoryBandwidthTBps = hardware.chip.memory_bandwidth_gbps / 1000
    const ridgePoint = peakTflops / memoryBandwidthTBps

    // 生成 Roofline 边界线数据点
    const rooflineData: [number, number][] = []
    const minOI = 0.1
    const maxOI = 1000

    for (let oi = minOI; oi <= maxOI; oi *= 1.5) {
      const memoryBoundPerf = oi * memoryBandwidthTBps
      const actualPerf = Math.min(peakTflops, memoryBoundPerf)
      rooflineData.push([oi, actualPerf])
    }

    // 计算当前方案的工作点
    const calculateWorkPoint = (r: PlanAnalysisResult) => {
      // 估算操作强度 (FLOP/Byte)
      // 简化：使用 MFU * peak / bandwidth 作为近似
      const achievedTflops = r.throughput.model_flops_utilization * peakTflops
      const operationalIntensity = r.latency.bottleneck_type === 'memory'
        ? achievedTflops / memoryBandwidthTBps * 0.8
        : ridgePoint * 1.2

      return {
        oi: operationalIntensity,
        perf: achievedTflops,
        planId: r.plan.plan_id,
        bottleneck: r.latency.bottleneck_type,
      }
    }

    const allResults = [result, ...comparisonResults.slice(0, 3)]
    const workPoints = allResults
      .filter((r) => r.is_feasible)
      .map(calculateWorkPoint)

    return {
      tooltip: {
        trigger: 'item',
        formatter: (params: unknown) => {
          const p = params as { seriesName: string; data: [number, number]; name?: string }
          if (p.seriesName === 'Roofline') {
            return `算术强度: ${p.data[0].toFixed(1)} FLOP/Byte<br/>性能上限: ${p.data[1].toFixed(1)} TFLOPS`
          }
          const point = workPoints.find((pt) => pt.planId === p.name)
          if (point) {
            return `
              <div style="font-weight: bold;">${point.planId}</div>
              <div>算术强度: ${point.oi.toFixed(1)} FLOP/Byte</div>
              <div>实际性能: ${point.perf.toFixed(2)} TFLOPS</div>
              <div>瓶颈: ${point.bottleneck === 'memory' ? '带宽受限' : point.bottleneck === 'compute' ? '算力受限' : '通信受限'}</div>
            `
          }
          return ''
        },
      },
      legend: {
        show: workPoints.length > 1,
        bottom: 0,
        itemWidth: 10,
        itemHeight: 10,
        textStyle: { fontSize: 10 },
      },
      grid: {
        left: 60,
        right: 30,
        top: 30,
        bottom: workPoints.length > 1 ? 40 : 20,
      },
      xAxis: {
        type: 'log',
        name: '算术强度 (FLOP/Byte)',
        nameLocation: 'middle',
        nameGap: 25,
        nameTextStyle: { fontSize: 11, color: '#666' },
        min: 0.1,
        max: 1000,
        axisLabel: { fontSize: 10 },
        axisLine: { lineStyle: { color: '#d9d9d9' } },
        splitLine: { lineStyle: { color: '#f0f0f0', type: 'dashed' } },
      },
      yAxis: {
        type: 'log',
        name: '性能 (TFLOPS)',
        nameLocation: 'middle',
        nameGap: 40,
        nameTextStyle: { fontSize: 11, color: '#666' },
        min: 0.1,
        max: peakTflops * 1.5,
        axisLabel: { fontSize: 10 },
        axisLine: { show: false },
        splitLine: { lineStyle: { color: '#f0f0f0', type: 'dashed' } },
      },
      series: [
        // Roofline 边界线
        {
          name: 'Roofline',
          type: 'line',
          data: rooflineData,
          smooth: false,
          symbol: 'none',
          lineStyle: {
            color: '#ff4d4f',
            width: 2,
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(255, 77, 79, 0.1)' },
                { offset: 1, color: 'rgba(255, 77, 79, 0)' },
              ],
            },
          },
        },
        // 工作点
        ...workPoints.map((point, index) => ({
          name: point.planId,
          type: 'scatter' as const,
          data: [[point.oi, point.perf]],
          symbolSize: index === 0 ? 14 : 10,
          itemStyle: {
            color: COLORS[index % COLORS.length],
            borderColor: '#fff',
            borderWidth: 2,
          },
          label: {
            show: index === 0,
            position: 'top' as const,
            formatter: point.planId,
            fontSize: 10,
            color: '#666',
          },
        })),
        // 峰值性能线
        {
          name: '峰值算力',
          type: 'line',
          data: [[ridgePoint, peakTflops], [1000, peakTflops]],
          symbol: 'none',
          lineStyle: {
            color: '#52c41a',
            width: 1,
            type: 'dashed',
          },
        },
        // 拐点标记
        {
          name: '拐点',
          type: 'scatter',
          data: [[ridgePoint, peakTflops]],
          symbolSize: 8,
          itemStyle: {
            color: '#52c41a',
          },
          label: {
            show: true,
            position: 'right',
            formatter: `拐点: ${ridgePoint.toFixed(1)}`,
            fontSize: 9,
            color: '#52c41a',
          },
        },
      ],
    }
  }, [result, hardware, comparisonResults])

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
        方案不可行，无法展示 Roofline 图
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
