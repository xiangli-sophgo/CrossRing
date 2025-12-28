/**
 * 甘特图组件 - 展示 LLM 推理时序
 *
 * 性能优化版：
 * - 使用事件委托代替单独事件处理
 * - 单一悬浮提示框，无 Tooltip 包装器
 * - 移除过渡动画
 */

import React, { useMemo, useState, useCallback, useRef } from 'react'
import { Empty, Typography } from 'antd'
import type { GanttChartData, GanttTask, GanttTaskType } from '../../../../utils/llmDeployment/simulation/types'

const { Text } = Typography

interface GanttChartProps {
  data: GanttChartData | null
  height?: number
  showLegend?: boolean
}

/** 任务类型颜色映射 */
const TASK_COLORS: Record<GanttTaskType, string> = {
  // 计算任务 - 绿色系
  compute: '#52c41a',
  embedding: '#73d13d',
  layernorm: '#95de64',
  attention_qkv: '#389e0d',
  attention_score: '#52c41a',
  attention_softmax: '#73d13d',
  attention_output: '#95de64',
  ffn_gate: '#237804',
  ffn_up: '#389e0d',
  ffn_down: '#52c41a',
  lm_head: '#135200',
  // 数据搬运 - 橙色系
  pcie_h2d: '#fa8c16',
  pcie_d2h: '#ffa940',
  hbm_write: '#ffc53d',
  hbm_read: '#ffd666',
  weight_load: '#d48806',
  kv_cache_read: '#faad14',
  kv_cache_write: '#ffc53d',
  // 通信 - 蓝/紫色系
  tp_comm: '#1890ff',
  pp_comm: '#722ed1',
  ep_comm: '#eb2f96',
  // 其他
  bubble: '#ff4d4f',
  idle: '#d9d9d9',
}

/** 任务类型标签 */
const TASK_LABELS: Record<GanttTaskType, string> = {
  compute: '计算',
  embedding: 'Embed',
  layernorm: 'LN',
  attention_qkv: 'QKV',
  attention_score: 'Score',
  attention_softmax: 'Softmax',
  attention_output: 'AttnOut',
  ffn_gate: 'Gate',
  ffn_up: 'Up',
  ffn_down: 'Down',
  lm_head: 'LMHead',
  pcie_h2d: 'H2D',
  pcie_d2h: 'D2H',
  hbm_write: 'HBM写',
  hbm_read: 'HBM读',
  weight_load: '权重',
  kv_cache_read: 'KV读',
  kv_cache_write: 'KV写',
  tp_comm: 'TP',
  pp_comm: 'PP',
  ep_comm: 'EP',
  bubble: '气泡',
  idle: '空闲',
}

/** 图例分组配置 */
const LEGEND_GROUPS = [
  {
    name: '计算',
    types: ['compute', 'attention_qkv', 'ffn_gate', 'lm_head'] as GanttTaskType[],
  },
  {
    name: '数据搬运',
    types: ['pcie_h2d', 'weight_load', 'kv_cache_read'] as GanttTaskType[],
  },
  {
    name: '通信',
    types: ['tp_comm', 'pp_comm', 'ep_comm'] as GanttTaskType[],
  },
  {
    name: '其他',
    types: ['bubble', 'idle'] as GanttTaskType[],
  },
]

/** 图表边距 */
const MARGIN = { top: 30, right: 20, bottom: 25, left: 100 }

/** 资源行高度 */
const ROW_HEIGHT = 24

/** 任务条高度 */
const BAR_HEIGHT = 18

/** 悬浮提示框样式 */
const tooltipStyle: React.CSSProperties = {
  position: 'fixed',
  background: 'rgba(0, 0, 0, 0.85)',
  color: '#fff',
  padding: '8px 12px',
  borderRadius: 6,
  fontSize: 12,
  lineHeight: 1.5,
  pointerEvents: 'none',
  zIndex: 1000,
  maxWidth: 250,
  boxShadow: '0 3px 6px -4px rgba(0,0,0,0.12), 0 6px 16px 0 rgba(0,0,0,0.08)',
}

export const GanttChart: React.FC<GanttChartProps> = ({
  data,
  height = 300,
  showLegend = true,
}) => {
  const [tooltip, setTooltip] = useState<{ task: GanttTask; x: number; y: number } | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  // 计算图表尺寸
  const chartWidth = 600
  const chartHeight = data
    ? Math.max(height, MARGIN.top + MARGIN.bottom + data.resources.length * ROW_HEIGHT)
    : height

  // 计算比例尺
  const { xScale, yScale, timeRange } = useMemo(() => {
    if (!data) return { xScale: () => 0, yScale: () => 0, timeRange: { start: 0, end: 1 } }

    const innerWidth = chartWidth - MARGIN.left - MARGIN.right
    const { start, end } = data.timeRange

    const xScale = (time: number) => {
      return MARGIN.left + ((time - start) / (end - start)) * innerWidth
    }

    const yScale = (resourceIndex: number) => {
      return MARGIN.top + resourceIndex * ROW_HEIGHT
    }

    return { xScale, yScale, timeRange: data.timeRange }
  }, [data, chartWidth])

  // 格式化时间
  const formatTime = (ms: number): string => {
    if (ms < 1) return `${(ms * 1000).toFixed(1)}µs`
    if (ms < 1000) return `${ms.toFixed(2)}ms`
    return `${(ms / 1000).toFixed(2)}s`
  }

  // 生成时间刻度
  const timeTicksData = useMemo(() => {
    if (!data) return []

    const { start, end } = data.timeRange
    const duration = end - start
    const numTicks = 6

    const ticks: number[] = []
    for (let i = 0; i <= numTicks; i++) {
      ticks.push(start + (duration / numTicks) * i)
    }
    return ticks
  }, [data])

  // 按资源分组任务，并预计算坐标
  const taskRenderData = useMemo(() => {
    if (!data) return []

    const result: Array<{
      task: GanttTask
      x: number
      y: number
      width: number
      color: string
    }> = []

    // 构建资源索引映射
    const resourceIndexMap = new Map<string, number>()
    data.resources.forEach((r, i) => resourceIndexMap.set(r.id, i))

    for (const task of data.tasks) {
      const isNetworkTask = task.type === 'tp_comm' || task.type === 'pp_comm' || task.type === 'ep_comm'
      const resourceId = `stage${task.ppStage}_${isNetworkTask ? 'network' : 'compute'}`
      let resourceIndex = resourceIndexMap.get(resourceId)

      if (resourceIndex === undefined) {
        const fallbackId = `stage${task.ppStage}_compute`
        resourceIndex = resourceIndexMap.get(fallbackId)
      }

      if (resourceIndex !== undefined) {
        const x = xScale(task.start)
        const width = Math.max(1, xScale(task.end) - xScale(task.start))
        const y = yScale(resourceIndex) + (ROW_HEIGHT - BAR_HEIGHT) / 2

        result.push({
          task,
          x,
          y,
          width,
          color: task.color || TASK_COLORS[task.type],
        })
      }
    }

    return result
  }, [data, xScale, yScale])

  // 任务ID到任务的映射（用于快速查找）
  const taskMap = useMemo(() => {
    if (!data) return new Map<string, GanttTask>()
    const map = new Map<string, GanttTask>()
    for (const task of data.tasks) {
      map.set(task.id, task)
    }
    return map
  }, [data])

  // 事件委托处理鼠标移动
  const handleMouseMove = useCallback((e: React.MouseEvent<SVGGElement>) => {
    const target = e.target as SVGElement
    if (target.tagName === 'rect' && target.dataset.taskId) {
      const task = taskMap.get(target.dataset.taskId)
      if (task) {
        setTooltip({ task, x: e.clientX + 10, y: e.clientY + 10 })
      }
    }
  }, [taskMap])

  const handleMouseLeave = useCallback(() => {
    setTooltip(null)
  }, [])

  if (!data || data.tasks.length === 0) {
    return (
      <Empty
        description="运行模拟以生成甘特图"
        style={{ marginTop: 20 }}
        image={Empty.PRESENTED_IMAGE_SIMPLE}
      />
    )
  }

  return (
    <div style={{ width: '100%', overflowX: 'auto' }}>
      {/* SVG 图表 */}
      <svg ref={svgRef} width={chartWidth} height={chartHeight}>
        {/* 背景网格 */}
        <g className="grid">
          {timeTicksData.map((tick, i) => (
            <line
              key={i}
              x1={xScale(tick)}
              y1={MARGIN.top}
              x2={xScale(tick)}
              y2={chartHeight - MARGIN.bottom}
              stroke="#f0f0f0"
              strokeDasharray="2,2"
            />
          ))}
        </g>

        {/* Prefill/Decode 分界线 */}
        {data.phaseTransition && (
          <g>
            <line
              x1={xScale(data.phaseTransition)}
              y1={MARGIN.top - 5}
              x2={xScale(data.phaseTransition)}
              y2={chartHeight - MARGIN.bottom}
              stroke="#ff4d4f"
              strokeWidth={1.5}
              strokeDasharray="4,4"
            />
            <text
              x={xScale(data.phaseTransition) - 30}
              y={MARGIN.top - 8}
              fontSize={10}
              fill="#ff4d4f"
            >
              Prefill
            </text>
            <text
              x={xScale(data.phaseTransition) + 5}
              y={MARGIN.top - 8}
              fontSize={10}
              fill="#1890ff"
            >
              Decode
            </text>
          </g>
        )}

        {/* 资源行标签和背景 */}
        <g className="y-axis">
          {data.resources.map((resource, i) => (
            <g key={resource.id} transform={`translate(0, ${yScale(i)})`}>
              <text
                x={MARGIN.left - 6}
                y={ROW_HEIGHT / 2 + 4}
                textAnchor="end"
                fontSize={10}
                fill="#666"
              >
                {resource.name}
              </text>
              <rect
                x={MARGIN.left}
                y={(ROW_HEIGHT - BAR_HEIGHT) / 2}
                width={chartWidth - MARGIN.left - MARGIN.right}
                height={BAR_HEIGHT}
                fill={i % 2 === 0 ? '#fafafa' : '#fff'}
                rx={2}
              />
            </g>
          ))}
        </g>

        {/* 任务条 - 使用事件委托 */}
        <g
          className="tasks"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        >
          {taskRenderData.map(({ task, x, y, width, color }) => (
            <rect
              key={task.id}
              data-task-id={task.id}
              x={x}
              y={y}
              width={width}
              height={BAR_HEIGHT}
              fill={color}
              rx={1}
              opacity={0.9}
            />
          ))}
        </g>

        {/* 时间轴 */}
        <g className="x-axis" transform={`translate(0, ${chartHeight - MARGIN.bottom})`}>
          <line
            x1={MARGIN.left}
            y1={0}
            x2={chartWidth - MARGIN.right}
            y2={0}
            stroke="#d9d9d9"
          />
          {timeTicksData.map((tick, i) => (
            <g key={i} transform={`translate(${xScale(tick)}, 0)`}>
              <line y1={0} y2={4} stroke="#999" />
              <text
                y={16}
                textAnchor="middle"
                fontSize={9}
                fill="#666"
              >
                {formatTime(tick)}
              </text>
            </g>
          ))}
        </g>
      </svg>

      {/* 悬浮提示框 */}
      {tooltip && (
        <div
          style={{
            ...tooltipStyle,
            left: tooltip.x,
            top: tooltip.y,
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 4 }}>{tooltip.task.name}</div>
          <div>开始: {formatTime(tooltip.task.start)}</div>
          <div>结束: {formatTime(tooltip.task.end)}</div>
          <div>耗时: {formatTime(tooltip.task.end - tooltip.task.start)}</div>
          <div>阶段: {tooltip.task.phase === 'prefill' ? 'Prefill' : 'Decode'}</div>
          {tooltip.task.layerIndex !== undefined && <div>层: {tooltip.task.layerIndex}</div>}
          {tooltip.task.tokenIndex !== undefined && <div>Token: {tooltip.task.tokenIndex}</div>}
        </div>
      )}

      {/* 图例和统计信息 - 合并在底部 */}
      <div style={{
        marginTop: 8,
        padding: '8px 0',
        borderTop: '1px solid #f0f0f0',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        gap: 16,
      }}>
        {/* 图例 */}
        {showLegend && (
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', flex: 1 }}>
            {LEGEND_GROUPS.map((group) => (
              <div key={group.name} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ fontSize: 10, color: '#999', marginRight: 2 }}>{group.name}:</span>
                {group.types.map((type) => (
                  <span
                    key={type}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 3,
                      fontSize: 10,
                    }}
                  >
                    <span
                      style={{
                        width: 10,
                        height: 10,
                        borderRadius: 2,
                        backgroundColor: TASK_COLORS[type],
                      }}
                    />
                    <span style={{ color: '#666' }}>{TASK_LABELS[type]}</span>
                  </span>
                ))}
              </div>
            ))}
          </div>
        )}

        {/* 统计信息 */}
        <div style={{ display: 'flex', gap: 12, fontSize: 10, color: '#666', whiteSpace: 'nowrap' }}>
          <Text type="secondary" style={{ fontSize: 10 }}>
            总时长: {formatTime(timeRange.end - timeRange.start)}
          </Text>
          {data.phaseTransition && (
            <>
              <Text type="secondary" style={{ fontSize: 10 }}>
                Prefill: {formatTime(data.phaseTransition)}
              </Text>
              <Text type="secondary" style={{ fontSize: 10 }}>
                Decode: {formatTime(timeRange.end - data.phaseTransition)}
              </Text>
            </>
          )}
          <Text type="secondary" style={{ fontSize: 10 }}>
            任务数: {data.tasks.length}
          </Text>
        </div>
      </div>
    </div>
  )
}

export default GanttChart
