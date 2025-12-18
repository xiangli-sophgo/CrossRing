import React, { useMemo, useRef, useState, useEffect, useCallback } from 'react'
import { Modal, Button, Space, Typography, Breadcrumb, Segmented, Tooltip, Checkbox } from 'antd'
import { ZoomInOutlined, ZoomOutOutlined, ReloadOutlined, UndoOutlined, RedoOutlined } from '@ant-design/icons'
import {
  CHIP_TYPE_COLORS,
  SWITCH_LAYER_COLORS,
  HierarchyLevel,
  LayoutType,
  LinkTraffic,
} from '../../types'
import { getHeatmapColor } from '../../utils/trafficAnalysis'
import {
  BOARD_U_COLORS,
  TopologyGraphProps,
  Node,
  Edge,
} from './shared'
import {
  circleLayout,
  torusLayout,
  getTorusGridSize,
  getTorus3DSize,
  getLayoutForTopology,
  hierarchicalLayout,
  hybridLayout,
  isometricStackedLayout,
} from './layouts'
import { LEVEL_PAIR_NAMES } from '../../types'

const { Text } = Typography

// 渲染节点形状的辅助函数
function renderNodeShape(node: Node): React.ReactNode {
  const isSwitch = node.isSwitch
  if (isSwitch) {
    return (
      <g>
        <rect x={-36} y={-14} width={72} height={28} rx={3} fill={node.color} stroke="#fff" strokeWidth={2} />
        <rect x={-32} y={-10} width={64} height={16} rx={2} fill="rgba(0,0,0,0.15)" />
        <rect x={-28} y={-6} width={6} height={8} rx={1} fill="rgba(255,255,255,0.5)" />
        <rect x={-20} y={-6} width={6} height={8} rx={1} fill="rgba(255,255,255,0.5)" />
        <rect x={-12} y={-6} width={6} height={8} rx={1} fill="rgba(255,255,255,0.5)" />
        <rect x={-4} y={-6} width={6} height={8} rx={1} fill="rgba(255,255,255,0.5)" />
        <rect x={6} y={-6} width={6} height={8} rx={1} fill="rgba(255,255,255,0.5)" />
        <rect x={14} y={-6} width={6} height={8} rx={1} fill="rgba(255,255,255,0.5)" />
        <rect x={22} y={-6} width={6} height={8} rx={1} fill="rgba(255,255,255,0.5)" />
        <circle cx={-28} cy={8} r={2} fill="#4ade80" />
        <circle cx={-22} cy={8} r={2} fill="#4ade80" />
        <circle cx={-16} cy={8} r={2} fill="#4ade80" />
        <circle cx={-10} cy={8} r={2} fill="#fbbf24" />
        <rect x={10} y={4} width={18} height={6} rx={1} fill="rgba(255,255,255,0.2)" />
      </g>
    )
  }
  if (node.type === 'pod') {
    return (
      <g>
        <rect x={-28} y={-12} width={56} height={32} rx={3} fill={node.color} stroke="#fff" strokeWidth={2} />
        <polygon points="-32,-12 0,-24 32,-12" fill={node.color} stroke="#fff" strokeWidth={2} />
        <rect x={-20} y={-4} width={8} height={8} rx={1} fill="rgba(255,255,255,0.3)" />
        <rect x={-6} y={-4} width={8} height={8} rx={1} fill="rgba(255,255,255,0.3)" />
        <rect x={8} y={-4} width={8} height={8} rx={1} fill="rgba(255,255,255,0.3)" />
        <rect x={-5} y={8} width={10} height={12} rx={1} fill="rgba(255,255,255,0.4)" />
      </g>
    )
  }
  if (node.type === 'rack') {
    return (
      <g>
        <rect x={-18} y={-28} width={36} height={56} rx={3} fill={node.color} stroke="#fff" strokeWidth={2} />
        <line x1={-14} y1={-16} x2={14} y2={-16} stroke="rgba(255,255,255,0.3)" strokeWidth={1} />
        <line x1={-14} y1={-4} x2={14} y2={-4} stroke="rgba(255,255,255,0.3)" strokeWidth={1} />
        <line x1={-14} y1={8} x2={14} y2={8} stroke="rgba(255,255,255,0.3)" strokeWidth={1} />
        <line x1={-14} y1={20} x2={14} y2={20} stroke="rgba(255,255,255,0.3)" strokeWidth={1} />
        <circle cx={10} cy={-22} r={2} fill="#4ade80" />
        <circle cx={10} cy={-10} r={2} fill="#4ade80" />
        <circle cx={10} cy={2} r={2} fill="#4ade80" />
        <circle cx={10} cy={14} r={2} fill="#fbbf24" />
      </g>
    )
  }
  if (node.type === 'board') {
    return (
      <g>
        <rect x={-32} y={-18} width={64} height={36} rx={2} fill={node.color} stroke="#fff" strokeWidth={2} />
        <path d="M-24,-10 L-24,-2 L-16,-2 L-16,6 L-8,6" stroke="rgba(255,255,255,0.25)" strokeWidth={1.5} fill="none" />
        <path d="M8,-10 L8,0 L16,0 L16,8 L24,8" stroke="rgba(255,255,255,0.25)" strokeWidth={1.5} fill="none" />
        <rect x={-8} y={-8} width={16} height={16} rx={1} fill="rgba(0,0,0,0.2)" stroke="rgba(255,255,255,0.3)" strokeWidth={1} />
        <circle cx={-26} cy={0} r={3} fill="rgba(255,255,255,0.4)" />
        <circle cx={26} cy={0} r={3} fill="rgba(255,255,255,0.4)" />
      </g>
    )
  }
  if (node.type === 'npu' || node.type === 'cpu') {
    return (
      <g>
        <rect x={-20} y={-20} width={40} height={40} rx={2} fill={node.color} stroke="#fff" strokeWidth={2} />
        <rect x={-12} y={-26} width={4} height={6} fill={node.color} />
        <rect x={-2} y={-26} width={4} height={6} fill={node.color} />
        <rect x={8} y={-26} width={4} height={6} fill={node.color} />
        <rect x={-12} y={20} width={4} height={6} fill={node.color} />
        <rect x={-2} y={20} width={4} height={6} fill={node.color} />
        <rect x={8} y={20} width={4} height={6} fill={node.color} />
        <rect x={-26} y={-12} width={6} height={4} fill={node.color} />
        <rect x={-26} y={-2} width={6} height={4} fill={node.color} />
        <rect x={-26} y={8} width={6} height={4} fill={node.color} />
        <rect x={20} y={-12} width={6} height={4} fill={node.color} />
        <rect x={20} y={-2} width={6} height={4} fill={node.color} />
        <rect x={20} y={8} width={6} height={4} fill={node.color} />
        <rect x={-10} y={-10} width={20} height={20} rx={1} fill="rgba(255,255,255,0.15)" />
      </g>
    )
  }
  // 默认: 圆角矩形
  return <rect x={-25} y={-18} width={50} height={36} rx={6} fill={node.color} stroke="#fff" strokeWidth={2} />
}

// 动画化的手动连接组件（必须在组件外部定义，避免每次渲染重新创建）
interface AnimatedManualConnectionProps {
  conn: { id: string; source: string; target: string }
  sourcePos: { x: number; y: number; zLayer: number } | null
  targetPos: { x: number; y: number; zLayer: number } | null
  isSelected: boolean
  isCrossContainer: boolean
  indexDiff: number
  onClick: (e: React.MouseEvent) => void
  layoutType?: string
  containers?: Array<{
    zLayer: number
    bounds: { x: number; y: number; width: number; height: number }
  }>
}

/**
 * 跨容器手动连接组件（带 CSS transition 动画）
 * 跨容器时，中间容器区域显示为虚线
 */
const ManualConnectionLine: React.FC<AnimatedManualConnectionProps> = ({
  sourcePos,
  targetPos,
  isSelected,
  isCrossContainer,
  indexDiff,
  onClick,
  containers,
}) => {
  if (!sourcePos || !targetPos) return null

  const strokeColor = isSelected ? '#52c41a' : (isCrossContainer ? '#722ed1' : '#b0b0b0')
  const strokeWidth = isSelected ? 3 : 2
  const transitionStyle = { transition: 'all 0.3s ease-out' }

  // 计算分段（跨容器时，中间容器显示虚线）
  const getSegments = () => {
    if (!isCrossContainer || !containers || containers.length === 0) {
      return [{ from: sourcePos, to: targetPos, isDashed: false }]
    }

    const minZ = Math.min(sourcePos.zLayer, targetPos.zLayer)
    const maxZ = Math.max(sourcePos.zLayer, targetPos.zLayer)

    // 找出经过的容器，按 zLayer 排序
    const passedContainers = containers
      .filter(c => c.zLayer >= minZ && c.zLayer <= maxZ)
      .sort((a, b) => a.zLayer - b.zLayer)

    if (passedContainers.length <= 2) {
      // 没有中间容器，直接返回一段
      return [{ from: sourcePos, to: targetPos, isDashed: false }]
    }

    const segments: Array<{ from: { x: number; y: number }; to: { x: number; y: number }; isDashed: boolean }> = []

    // 线性插值：根据 y 坐标计算 x 坐标
    const getX = (y: number) => {
      if (Math.abs(targetPos.y - sourcePos.y) < 0.001) return sourcePos.x
      const t = (y - sourcePos.y) / (targetPos.y - sourcePos.y)
      return sourcePos.x + t * (targetPos.x - sourcePos.x)
    }

    // 确保 y 从小到大排序
    const startY = Math.min(sourcePos.y, targetPos.y)
    const endY = Math.max(sourcePos.y, targetPos.y)
    const isSourceAbove = sourcePos.y < targetPos.y

    // 收集所有容器边界的 y 坐标
    const boundaryYs: Array<{ y: number; zLayer: number; isTop: boolean }> = []
    for (const c of passedContainers) {
      boundaryYs.push({ y: c.bounds.y, zLayer: c.zLayer, isTop: true })
      boundaryYs.push({ y: c.bounds.y + c.bounds.height, zLayer: c.zLayer, isTop: false })
    }
    boundaryYs.sort((a, b) => a.y - b.y)

    // 根据 y 坐标分段
    let currentY = startY
    let lastPoint: { x: number; y: number } = isSourceAbove
      ? { x: sourcePos.x, y: sourcePos.y }
      : { x: targetPos.x, y: targetPos.y }

    for (const boundary of boundaryYs) {
      if (boundary.y <= startY || boundary.y >= endY) continue

      const x = getX(boundary.y)
      const nextPoint = { x, y: boundary.y }

      // 判断当前线段所在的容器
      const midY = (currentY + boundary.y) / 2
      let segmentContainer: { zLayer: number } | null = null
      for (const c of passedContainers) {
        if (midY >= c.bounds.y && midY <= c.bounds.y + c.bounds.height) {
          segmentContainer = c
          break
        }
      }

      const isMiddle = segmentContainer &&
        segmentContainer.zLayer !== sourcePos.zLayer &&
        segmentContainer.zLayer !== targetPos.zLayer

      segments.push({
        from: { x: lastPoint.x, y: lastPoint.y },
        to: nextPoint,
        isDashed: !!isMiddle,
      })

      lastPoint = nextPoint
      currentY = boundary.y
    }

    // 最后一段
    const finalPoint = isSourceAbove ? targetPos : sourcePos
    const midY = (currentY + (isSourceAbove ? targetPos.y : sourcePos.y)) / 2
    let segmentContainer: { zLayer: number } | null = null
    for (const c of passedContainers) {
      if (midY >= c.bounds.y && midY <= c.bounds.y + c.bounds.height) {
        segmentContainer = c
        break
      }
    }
    const isMiddle = segmentContainer &&
      segmentContainer.zLayer !== sourcePos.zLayer &&
      segmentContainer.zLayer !== targetPos.zLayer

    segments.push({
      from: { x: lastPoint.x, y: lastPoint.y },
      to: { x: finalPoint.x, y: finalPoint.y },
      isDashed: !!isMiddle,
    })

    return segments
  }

  if (isCrossContainer) {
    // 曲线连接 - 计算分段
    const midX = (sourcePos.x + targetPos.x) / 2
    const midY = (sourcePos.y + targetPos.y) / 2
    const dx = targetPos.x - sourcePos.x
    const dy = targetPos.y - sourcePos.y
    const dist = Math.sqrt(dx * dx + dy * dy) || 1
    const bulge = dist * 0.2 * indexDiff
    const perpX = -dy / dist
    const perpY = dx / dist
    const ctrlX = midX + perpX * bulge
    const ctrlY = midY + perpY * bulge

    // 贝塞尔曲线分段绘制
    const segments = getSegments()
    const hasMiddleSegments = segments.some(s => s.isDashed)

    if (!hasMiddleSegments) {
      // 没有中间段，整条曲线
      const pathD = `M ${sourcePos.x} ${sourcePos.y} Q ${ctrlX} ${ctrlY}, ${targetPos.x} ${targetPos.y}`
      return (
        <g style={transitionStyle}>
          <path d={pathD} fill="none" stroke="transparent" strokeWidth={16}
            style={{ cursor: 'pointer' }} onClick={onClick} />
          <path d={pathD} fill="none" stroke={strokeColor} strokeWidth={strokeWidth}
            strokeOpacity={isSelected ? 1 : 0.8}
            style={{ pointerEvents: 'none', filter: isSelected ? 'drop-shadow(0 0 4px #52c41a)' : 'none' }}
          />
        </g>
      )
    }

    // 有中间段，分段绘制曲线
    // 简化处理：用分段的直线近似（因为贝塞尔曲线分段复杂）
    return (
      <g style={transitionStyle}>
        {/* 透明点击区域 */}
        <path
          d={`M ${sourcePos.x} ${sourcePos.y} Q ${ctrlX} ${ctrlY}, ${targetPos.x} ${targetPos.y}`}
          fill="none" stroke="transparent" strokeWidth={16}
          style={{ cursor: 'pointer' }} onClick={onClick}
        />
        {/* 分段绘制 */}
        {segments.map((seg, i) => (
          <line
            key={i}
            x1={seg.from.x} y1={seg.from.y}
            x2={seg.to.x} y2={seg.to.y}
            stroke={strokeColor}
            strokeWidth={strokeWidth}
            strokeOpacity={isSelected ? 1 : 0.8}
            strokeDasharray={seg.isDashed ? "8 4" : undefined}
            style={{ pointerEvents: 'none', filter: isSelected ? 'drop-shadow(0 0 4px #52c41a)' : 'none' }}
          />
        ))}
      </g>
    )
  } else {
    // 直线连接
    return (
      <g>
        <line
          x1={sourcePos.x} y1={sourcePos.y}
          x2={targetPos.x} y2={targetPos.y}
          stroke="transparent" strokeWidth={16}
          style={{ cursor: 'pointer', ...transitionStyle }} onClick={onClick}
        />
        <line
          x1={sourcePos.x} y1={sourcePos.y}
          x2={targetPos.x} y2={targetPos.y}
          stroke={strokeColor} strokeWidth={strokeWidth}
          strokeOpacity={isSelected ? 1 : 0.6}
          style={{ pointerEvents: 'none', filter: isSelected ? 'drop-shadow(0 0 4px #52c41a)' : 'none', ...transitionStyle }}
        />
      </g>
    )
  }
}

export const TopologyGraph: React.FC<TopologyGraphProps> = ({
  visible,
  onClose,
  topology,
  currentLevel,
  currentPod,
  currentRack,
  currentBoard,
  onNodeDoubleClick,
  onNodeClick,
  onLinkClick,
  selectedNodeId = null,
  selectedLinkId = null,
  onNavigateBack: _onNavigateBack,
  onBreadcrumbClick,
  breadcrumbs = [],
  canGoBack: _canGoBack = false,
  embedded = false,
  // 手动连线相关
  connectionMode = 'view',
  selectedNodes = new Set<string>(),
  onSelectedNodesChange,
  targetNodes = new Set<string>(),
  onTargetNodesChange,
  sourceNode: _sourceNode = null,
  onSourceNodeChange: _onSourceNodeChange,
  onManualConnect: _onManualConnect,
  manualConnections = [],
  onDeleteManualConnection: _onDeleteManualConnection,
  onDeleteConnection: _onDeleteConnection,
  layoutType = 'auto',
  onLayoutTypeChange,
  // 多层级视图相关
  multiLevelOptions,
  onMultiLevelOptionsChange,
  // 流量分析热力图
  trafficAnalysisResult,
}) => {
  void _onNavigateBack
  void _canGoBack
  void _sourceNode
  void _onSourceNodeChange
  void _onManualConnect
  void _onDeleteManualConnection
  void _onDeleteConnection
  void onMultiLevelOptionsChange
  const svgRef = useRef<SVGSVGElement>(null)
  const [zoom, setZoom] = useState(1)
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null)
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null)

  // 多层级模式：悬停的层级索引（用于抬起上方层级）
  const [hoveredLayerIndex, setHoveredLayerIndex] = useState<number | null>(null)

  // 容器展开动画状态
  const [expandingContainer, setExpandingContainer] = useState<{
    id: string
    type: string
  } | null>(null)

  // 容器收缩动画状态（从单层级切换到多层级时使用）
  const [collapsingContainer, setCollapsingContainer] = useState<{
    id: string
    type: string
  } | null>(null)
  // 收缩动画是否已开始（用于两阶段动画：先渲染展开状态，再过渡到正常状态）
  const [collapseAnimationStarted, setCollapseAnimationStarted] = useState(false)

  // 视图切换淡入效果
  const [viewFadeIn, setViewFadeIn] = useState(false)
  const prevMultiLevelEnabled = useRef(multiLevelOptions?.enabled)

  // 检测视图切换，触发动画
  useEffect(() => {
    if (prevMultiLevelEnabled.current && !multiLevelOptions?.enabled) {
      // 从多层级切换到单层级，触发淡入
      setViewFadeIn(true)
      const timer = setTimeout(() => setViewFadeIn(false), 50)
      return () => clearTimeout(timer)
    } else if (!prevMultiLevelEnabled.current && multiLevelOptions?.enabled) {
      // 从单层级切换到多层级，触发收缩动画
      // 根据当前层级确定要收缩到哪个容器
      let containerId = ''
      let containerType = ''
      if (currentRack) {
        containerId = currentRack.id
        containerType = 'rack'
      } else if (currentPod) {
        containerId = currentPod.id
        containerType = 'pod'
      }
      if (containerId) {
        setCollapsingContainer({ id: containerId, type: containerType })
        setCollapseAnimationStarted(false)  // 重置动画开始标志
        // 动画完成后清除状态
        const timer = setTimeout(() => {
          setCollapsingContainer(null)
          setCollapseAnimationStarted(false)
        }, 600)
        return () => clearTimeout(timer)
      }
    }
    prevMultiLevelEnabled.current = multiLevelOptions?.enabled
  }, [multiLevelOptions?.enabled, currentPod, currentRack])

  // 收缩动画：在下一帧开始动画（从展开状态过渡到正常状态）
  useEffect(() => {
    if (collapsingContainer && !collapseAnimationStarted) {
      const frameId = requestAnimationFrame(() => {
        setCollapseAnimationStarted(true)
      })
      return () => cancelAnimationFrame(frameId)
    }
  }, [collapsingContainer, collapseAnimationStarted])

  // 流量分析热力图：创建链路流量查找表
  const linkTrafficMap = useMemo(() => {
    const map = new Map<string, LinkTraffic>()
    if (trafficAnalysisResult?.link_traffic) {
      for (const lt of trafficAnalysisResult.link_traffic) {
        // 使用双向key，因为边可能source/target顺序不同
        map.set(`${lt.source}-${lt.target}`, lt)
        map.set(`${lt.target}-${lt.source}`, lt)
      }
    }
    return map
  }, [trafficAnalysisResult])

  // 获取边的热力图样式
  const getTrafficHeatmapStyle = useCallback((source: string, target: string) => {
    const lt = linkTrafficMap.get(`${source}-${target}`)
    if (!lt) return null
    return {
      stroke: getHeatmapColor(lt.bandwidth_utilization),
      strokeWidth: 2 + lt.bandwidth_utilization * 4,  // 2-6px
      trafficMb: lt.traffic_mb,
      utilization: lt.bandwidth_utilization,
    }
  }, [linkTrafficMap])

  // 手动调整模式开关（内部状态）
  const [isManualMode, setIsManualMode] = useState(false)

  // 手动布局缓存key（按层级、路径和布局类型区分）
  const getManualPositionsCacheKey = (layout: LayoutType) => {
    const pathKey = currentLevel === 'datacenter' ? 'dc' :
      currentLevel === 'pod' ? `pod_${currentPod?.id}` :
      currentLevel === 'rack' ? `rack_${currentRack?.id}` :
      `board_${currentBoard?.id}`
    return `tier6_manual_positions_${pathKey}_${layout}`
  }

  // 手动布局：按布局类型分开存储位置（从localStorage加载）
  const [manualPositionsByLayout, setManualPositionsByLayout] = useState<Record<LayoutType, Record<string, { x: number; y: number }>>>(() => {
    const result: Record<LayoutType, Record<string, { x: number; y: number }>> = {
      auto: {},
      circle: {},
      grid: {},
    }
    try {
      const pathKey = currentLevel === 'datacenter' ? 'dc' : currentLevel
      for (const layout of ['auto', 'circle', 'grid'] as LayoutType[]) {
        const cached = localStorage.getItem(`tier6_manual_positions_${pathKey}_${layout}`)
        if (cached) result[layout] = JSON.parse(cached)
      }
    } catch (e) { /* ignore */ }
    return result
  })

  // 当前布局的手动位置（便捷访问）
  const manualPositions = manualPositionsByLayout[layoutType] || {}

  // 拖动状态
  const [draggingNode, setDraggingNode] = useState<string | null>(null)
  const [dragStart, setDragStart] = useState<{ x: number; y: number; nodeX: number; nodeY: number } | null>(null)

  // 撤销/重做历史
  const [history, setHistory] = useState<Record<string, { x: number; y: number }>[]>([])
  const [historyIndex, setHistoryIndex] = useState(-1)
  const maxHistoryLength = 50

  // 辅助线状态（支持水平、垂直和圆形）
  const [alignmentLines, setAlignmentLines] = useState<{ type: 'h' | 'v' | 'circle'; pos: number; center?: { x: number; y: number } }[]>([])

  // 层级/路径变化时，加载对应的手动位置
  useEffect(() => {
    const result: Record<LayoutType, Record<string, { x: number; y: number }>> = {
      auto: {},
      circle: {},
      grid: {},
    }
    try {
      for (const layout of ['auto', 'circle', 'grid'] as LayoutType[]) {
        const key = getManualPositionsCacheKey(layout)
        const cached = localStorage.getItem(key)
        if (cached) result[layout] = JSON.parse(cached)
      }
    } catch (e) { /* ignore */ }
    setManualPositionsByLayout(result)
    // 重置历史和手动模式
    setHistory([])
    setHistoryIndex(-1)
    setIsManualMode(false)
  }, [currentLevel, currentPod?.id, currentRack?.id, currentBoard?.id])

  // 手动位置变化时自动保存（只保存当前布局）
  useEffect(() => {
    const positions = manualPositionsByLayout[layoutType]
    if (positions && Object.keys(positions).length > 0) {
      try {
        const key = getManualPositionsCacheKey(layoutType)
        localStorage.setItem(key, JSON.stringify(positions))
      } catch (e) { /* ignore */ }
    }
  }, [manualPositionsByLayout, layoutType])


  // 获取当前层级对应的 HierarchyLevel
  const getCurrentHierarchyLevel = (): HierarchyLevel => {
    switch (currentLevel) {
      case 'datacenter': return 'datacenter'
      case 'pod': return 'pod'
      case 'rack': return 'rack'
      case 'board': return 'board'
      default: return 'datacenter'
    }
  }

  // 根据当前层级生成节点和边
  const { nodes, edges, title, directTopology } = useMemo(() => {
    if (!topology) return { nodes: [], edges: [], title: '', directTopology: 'full_mesh' }

    const width = 800
    const height = 600

    // ==========================================
    // 多层级模式：生成堆叠视图数据
    // ==========================================
    if (multiLevelOptions?.enabled && multiLevelOptions.levelPair) {
      const levelPair = multiLevelOptions.levelPair

      let upperNodes: Node[] = []
      let lowerNodesMap = new Map<string, Node[]>()
      let allEdges: Edge[] = []
      let graphTitle = LEVEL_PAIR_NAMES[levelPair] + ' 拓扑'

      // 根据层级组合提取节点
      if (levelPair === 'datacenter_pod') {
        // Datacenter + Pod: 显示所有 Pod，以及每个 Pod 内的 Rack
        upperNodes = topology.pods.map(pod => ({
          id: pod.id,
          label: pod.label,
          type: 'pod',
          x: 0, y: 0,
          color: '#1890ff',
          hierarchyLevel: 'datacenter' as HierarchyLevel,
        }))
        topology.pods.forEach(pod => {
          const racks = pod.racks.map(rack => ({
            id: rack.id,
            label: rack.label,
            type: 'rack',
            x: 0, y: 0,
            color: '#52c41a',
            hierarchyLevel: 'pod' as HierarchyLevel,
          }))
          lowerNodesMap.set(pod.id, racks)
        })
        // 提取连接
        const allNodeIds = new Set([
          ...topology.pods.map(p => p.id),
          ...topology.pods.flatMap(p => p.racks.map(r => r.id))
        ])
        allEdges = topology.connections
          .filter(c => allNodeIds.has(c.source) && allNodeIds.has(c.target))
          .map(c => {
            const sourceIsPod = topology.pods.some(p => p.id === c.source)
            const targetIsPod = topology.pods.some(p => p.id === c.target)
            let connectionType: 'intra_upper' | 'intra_lower' | 'inter_level' = 'inter_level'
            if (sourceIsPod && targetIsPod) {
              connectionType = 'intra_upper'
            } else if (!sourceIsPod && !targetIsPod) {
              // 检查两个rack是否在同一个pod中
              const sourcePod = topology.pods.find(p => p.racks.some(r => r.id === c.source))
              const targetPod = topology.pods.find(p => p.racks.some(r => r.id === c.target))
              if (sourcePod && targetPod && sourcePod.id === targetPod.id) {
                connectionType = 'intra_lower'
              }
              // 否则保持 'inter_level'（跨pod的rack连接）
            }
            return { source: c.source, target: c.target, bandwidth: c.bandwidth, latency: c.latency, connectionType }
          })
      } else if (levelPair === 'pod_rack' && currentPod) {
        // Pod + Rack: 显示当前 Pod 的所有 Rack，以及每个 Rack 内的 Board
        upperNodes = currentPod.racks.map(rack => ({
          id: rack.id,
          label: rack.label,
          type: 'rack',
          x: 0, y: 0,
          color: '#52c41a',
          hierarchyLevel: 'pod' as HierarchyLevel,
        }))
        currentPod.racks.forEach(rack => {
          const boards = rack.boards.map(board => ({
            id: board.id,
            label: board.label,
            type: 'board',
            x: 0, y: 0,
            color: BOARD_U_COLORS[board.u_height] || '#666',
            uHeight: board.u_height,
            hierarchyLevel: 'rack' as HierarchyLevel,
          }))
          lowerNodesMap.set(rack.id, boards)
        })
        graphTitle = `${currentPod.label} - ${LEVEL_PAIR_NAMES[levelPair]}`
        // 提取连接
        const allNodeIds = new Set([
          ...currentPod.racks.map(r => r.id),
          ...currentPod.racks.flatMap(r => r.boards.map(b => b.id))
        ])
        allEdges = topology.connections
          .filter(c => allNodeIds.has(c.source) && allNodeIds.has(c.target))
          .map(c => {
            const sourceIsRack = currentPod.racks.some(r => r.id === c.source)
            const targetIsRack = currentPod.racks.some(r => r.id === c.target)
            let connectionType: 'intra_upper' | 'intra_lower' | 'inter_level' = 'inter_level'
            if (sourceIsRack && targetIsRack) {
              connectionType = 'intra_upper'
            } else if (!sourceIsRack && !targetIsRack) {
              // 检查两个board是否在同一个rack中
              const sourceRack = currentPod.racks.find(r => r.boards.some(b => b.id === c.source))
              const targetRack = currentPod.racks.find(r => r.boards.some(b => b.id === c.target))
              if (sourceRack && targetRack && sourceRack.id === targetRack.id) {
                connectionType = 'intra_lower'
              }
              // 否则保持 'inter_level'（跨rack的board连接）
            }
            return { source: c.source, target: c.target, bandwidth: c.bandwidth, latency: c.latency, connectionType }
          })
      } else if (levelPair === 'rack_board' && currentRack) {
        // Rack + Board: 显示当前 Rack 的所有 Board，以及每个 Board 内的 Chip
        upperNodes = currentRack.boards.map(board => ({
          id: board.id,
          label: board.label,
          type: 'board',
          x: 0, y: 0,
          color: BOARD_U_COLORS[board.u_height] || '#666',
          uHeight: board.u_height,
          hierarchyLevel: 'rack' as HierarchyLevel,
        }))
        currentRack.boards.forEach(board => {
          const chips = board.chips.map(chip => ({
            id: chip.id,
            label: chip.label || chip.type.toUpperCase(),
            type: chip.type,
            x: 0, y: 0,
            color: CHIP_TYPE_COLORS[chip.type] || '#666',
            hierarchyLevel: 'board' as HierarchyLevel,
          }))
          lowerNodesMap.set(board.id, chips)
        })
        graphTitle = `${currentRack.label} - ${LEVEL_PAIR_NAMES[levelPair]}`
        // 提取连接
        const allNodeIds = new Set([
          ...currentRack.boards.map(b => b.id),
          ...currentRack.boards.flatMap(b => b.chips.map(c => c.id))
        ])
        allEdges = topology.connections
          .filter(c => allNodeIds.has(c.source) && allNodeIds.has(c.target))
          .map(c => {
            const sourceIsBoard = currentRack.boards.some(b => b.id === c.source)
            const targetIsBoard = currentRack.boards.some(b => b.id === c.target)
            let connectionType: 'intra_upper' | 'intra_lower' | 'inter_level' = 'inter_level'
            if (sourceIsBoard && targetIsBoard) {
              connectionType = 'intra_upper'
            } else if (!sourceIsBoard && !targetIsBoard) {
              // 检查两个chip是否在同一个board中
              const sourceBoard = currentRack.boards.find(b => b.chips.some(ch => ch.id === c.source))
              const targetBoard = currentRack.boards.find(b => b.chips.some(ch => ch.id === c.target))
              if (sourceBoard && targetBoard && sourceBoard.id === targetBoard.id) {
                connectionType = 'intra_lower'
              }
              // 否则保持 'inter_level'（跨board的chip连接）
            }
            return { source: c.source, target: c.target, bandwidth: c.bandwidth, latency: c.latency, connectionType }
          })
      } else if (levelPair === 'board_chip' && currentBoard) {
        // Board + Chip: 显示当前 Board 的所有 Chip（单层级，无子节点）
        upperNodes = currentBoard.chips.map(chip => ({
          id: chip.id,
          label: chip.label || chip.type.toUpperCase(),
          type: chip.type,
          x: 0, y: 0,
          color: CHIP_TYPE_COLORS[chip.type] || '#666',
          hierarchyLevel: 'board' as HierarchyLevel,
        }))
        graphTitle = `${currentBoard.label} - Chip 拓扑`
        // Chip 层没有子层级，只显示 Chip 间连接
        const chipIds = new Set(currentBoard.chips.map(c => c.id))
        allEdges = topology.connections
          .filter(c => chipIds.has(c.source) && chipIds.has(c.target))
          .map(c => ({ source: c.source, target: c.target, bandwidth: c.bandwidth, latency: c.latency, connectionType: 'intra_upper' as const }))
      }

      // 应用堆叠布局
      if (upperNodes.length > 0) {
        const layoutResult = isometricStackedLayout(upperNodes, lowerNodesMap, width, height)

        // 为每个容器计算单层级布局数据（用于展开动画）
        // 使用与单层级视图完全相同的布局逻辑
        layoutResult.upperNodes.forEach(containerNode => {
          const children = lowerNodesMap.get(containerNode.id) || []
          if (children.length === 0) return

          const bounds = containerNode.containerBounds
          if (!bounds) return

          // 单层级视图的标准尺寸
          const singleLevelWidth = 800
          const singleLevelHeight = 600

          // 根据容器类型确定下层级别和对应的拓扑配置
          let directTopology = 'full_mesh'
          let keepDirectTopology = false
          if (topology.switch_config) {
            if (containerNode.type === 'pod') {
              // Pod 容器内显示 Rack，使用 inter_rack 配置
              const config = topology.switch_config.inter_rack
              directTopology = config?.direct_topology || 'full_mesh'
              keepDirectTopology = config?.enabled && config?.keep_direct_topology || false
            } else if (containerNode.type === 'rack') {
              // Rack 容器内显示 Board，使用 inter_board 配置
              const config = topology.switch_config.inter_board
              directTopology = config?.direct_topology || 'full_mesh'
              keepDirectTopology = config?.enabled && config?.keep_direct_topology || false
            } else if (containerNode.type === 'board') {
              // Board 容器内显示 Chip，使用 inter_chip 配置
              const config = topology.switch_config.inter_chip
              directTopology = config?.direct_topology || 'full_mesh'
              keepDirectTopology = config?.enabled && config?.keep_direct_topology || false
            }
          }

          // 检查子节点是否有 Switch（多层级模式下通常没有 Switch，但保留逻辑以备扩展）
          const hasSwitches = children.some(n => n.isSwitch)

          // 使用与单层级相同的布局逻辑
          let layoutedChildren: Node[]
          if (layoutType === 'circle') {
            if (hasSwitches) {
              layoutedChildren = hybridLayout(children, singleLevelWidth, singleLevelHeight, 'ring')
            } else {
              const radius = Math.min(singleLevelWidth, singleLevelHeight) * 0.35
              layoutedChildren = circleLayout(children, singleLevelWidth / 2, singleLevelHeight / 2, radius)
            }
          } else if (layoutType === 'grid') {
            if (hasSwitches) {
              layoutedChildren = hybridLayout(children, singleLevelWidth, singleLevelHeight, 'full_mesh_2d')
            } else {
              layoutedChildren = torusLayout(children, singleLevelWidth, singleLevelHeight)
            }
          } else {
            // auto 模式：与单层级相同的自动选择逻辑
            if (hasSwitches && keepDirectTopology && directTopology !== 'none') {
              layoutedChildren = hybridLayout(children, singleLevelWidth, singleLevelHeight, directTopology)
            } else if (hasSwitches) {
              layoutedChildren = hierarchicalLayout(children, singleLevelWidth, singleLevelHeight)
            } else {
              layoutedChildren = getLayoutForTopology(directTopology, children, singleLevelWidth, singleLevelHeight)
            }
          }

          // 提取容器内的边（intra_lower 类型）
          const childIds = new Set(children.map(c => c.id))
          const containerEdges = allEdges.filter(e =>
            childIds.has(e.source) && childIds.has(e.target) &&
            e.connectionType === 'intra_lower'
          )

          // 计算缩放比例（使单层级内容适应容器大小）
          const containerPadding = 40
          const availableWidth = bounds.width - containerPadding * 2
          const availableHeight = bounds.height - containerPadding * 2
          const scaleX = availableWidth / singleLevelWidth
          const scaleY = availableHeight / singleLevelHeight
          const scale = Math.min(scaleX, scaleY)

          // 存储单层级数据到容器节点
          containerNode.singleLevelData = {
            nodes: layoutedChildren,
            edges: containerEdges,
            viewBox: { width: singleLevelWidth, height: singleLevelHeight },
            scale,
            directTopology,
          }
        })

        const allNodes = [...layoutResult.upperNodes, ...layoutResult.lowerNodes]
        return {
          nodes: allNodes,
          edges: allEdges,
          title: graphTitle,
          directTopology: 'full_mesh',
        }
      }

      return { nodes: [], edges: [], title: graphTitle, directTopology: 'full_mesh' }
    }

    // ==========================================
    // 单层级模式：原有逻辑
    // ==========================================
    let nodeList: Node[] = []
    let edgeList: Edge[] = []
    let graphTitle = ''

    if (currentLevel === 'datacenter') {
      // 数据中心层：显示所有Pod和数据中心层Switch
      graphTitle = '数据中心拓扑'
      nodeList = topology.pods.map((pod) => ({
        id: pod.id,
        label: pod.label,
        type: 'pod',
        x: 0,
        y: 0,
        color: '#1890ff',
      }))

      // 添加数据中心层Switch
      if (topology.switches) {
        const dcSwitches = topology.switches.filter(s => s.hierarchy_level === 'inter_pod')
        dcSwitches.forEach(sw => {
          nodeList.push({
            id: sw.id,
            label: sw.label,
            type: 'switch',
            subType: sw.layer,
            isSwitch: true,
            x: 0,
            y: 0,
            color: SWITCH_LAYER_COLORS[sw.layer] || '#666',
            portInfo: {
              uplink: sw.uplink_ports_used,
              downlink: sw.downlink_ports_used,
              inter: sw.inter_ports_used,
            },
          })
        })

      }

      // Pod间连接和DC层Switch连接
      const podIds = new Set(topology.pods.map(p => p.id))
      const dcSwitchIds = new Set(
        (topology.switches || []).filter(s => s.hierarchy_level === 'inter_pod').map(s => s.id)
      )
      // 构建Pod层Switch到Pod的映射（用于转换跨层连接）
      const podSwitchToPod: Record<string, string> = {}
      ;(topology.switches || [])
        .filter((s): s is typeof s & { parent_id: string } => s.hierarchy_level === 'inter_rack' && !!s.parent_id)
        .forEach(s => { podSwitchToPod[s.id] = s.parent_id })

      edgeList = topology.connections
        .filter(c => {
          const sourceValid = podIds.has(c.source) || dcSwitchIds.has(c.source)
          const targetValid = podIds.has(c.target) || dcSwitchIds.has(c.target)
          if (sourceValid && targetValid) return true
          // 跨层连接（DC Switch到Pod Switch）
          if (dcSwitchIds.has(c.source) && podSwitchToPod[c.target]) return true
          if (dcSwitchIds.has(c.target) && podSwitchToPod[c.source]) return true
          return false
        })
        .map(c => {
          let source = c.source
          let target = c.target
          if (podSwitchToPod[c.source]) source = podSwitchToPod[c.source]
          if (podSwitchToPod[c.target]) target = podSwitchToPod[c.target]
          return {
            source,
            target,
            bandwidth: c.bandwidth,
            latency: c.latency,
            isSwitch: c.type === 'switch',
          }
        })

    } else if (currentLevel === 'pod' && currentPod) {
      // Pod层：显示所有Rack和Pod层Switch
      graphTitle = `${currentPod.label} - Rack拓扑`
      nodeList = currentPod.racks.map((rack) => ({
        id: rack.id,
        label: rack.label,
        type: 'rack',
        x: 0,
        y: 0,
        color: '#52c41a',
      }))

      // 添加Pod层Switch
      if (topology.switches) {
        const podSwitches = topology.switches.filter(s =>
          s.hierarchy_level === 'inter_rack' && s.parent_id === currentPod.id
        )
        podSwitches.forEach(sw => {
          nodeList.push({
            id: sw.id,
            label: sw.label,
            type: 'switch',
            subType: sw.layer,
            isSwitch: true,
            x: 0,
            y: 0,
            color: SWITCH_LAYER_COLORS[sw.layer] || '#666',
            portInfo: {
              uplink: sw.uplink_ports_used,
              downlink: sw.downlink_ports_used,
              inter: sw.inter_ports_used,
            },
          })
        })

      }

      // Rack间连接和Pod层Switch连接
      const rackIds = new Set(currentPod.racks.map(r => r.id))
      const podSwitchIds = new Set(
        (topology.switches || [])
          .filter(s => s.hierarchy_level === 'inter_rack' && s.parent_id === currentPod.id)
          .map(s => s.id)
      )
      // 构建Rack层Switch到Rack的映射（用于转换跨层连接）
      const rackSwitchToRack: Record<string, string> = {}
      ;(topology.switches || [])
        .filter((s): s is typeof s & { parent_id: string } => s.hierarchy_level === 'inter_board' && !!s.parent_id && rackIds.has(s.parent_id))
        .forEach(s => { rackSwitchToRack[s.id] = s.parent_id })

      edgeList = topology.connections
        .filter(c => {
          // 直接连接（Rack或Pod Switch之间）
          const sourceInPod = rackIds.has(c.source) || podSwitchIds.has(c.source)
          const targetInPod = rackIds.has(c.target) || podSwitchIds.has(c.target)
          if (sourceInPod && targetInPod) return true
          // 跨层连接（Pod Switch到Rack Switch）- 需要转换
          if (podSwitchIds.has(c.source) && rackSwitchToRack[c.target]) return true
          if (podSwitchIds.has(c.target) && rackSwitchToRack[c.source]) return true
          return false
        })
        .map(c => {
          // 转换跨层连接：将Rack Switch替换为对应的Rack
          let source = c.source
          let target = c.target
          if (rackSwitchToRack[c.source]) source = rackSwitchToRack[c.source]
          if (rackSwitchToRack[c.target]) target = rackSwitchToRack[c.target]
          return {
            source,
            target,
            bandwidth: c.bandwidth,
            latency: c.latency,
            isSwitch: c.type === 'switch',
          }
        })

    } else if (currentLevel === 'rack' && currentRack) {
      // Rack层：显示所有Board和Rack层Switch
      graphTitle = `${currentRack.label} - Board拓扑`
      nodeList = currentRack.boards.map((board) => ({
        id: board.id,
        label: board.label,
        type: 'board',
        x: 0,
        y: 0,
        color: BOARD_U_COLORS[board.u_height] || '#722ed1',
        uHeight: board.u_height,
      }))

      // 添加Rack层Switch
      if (topology.switches) {
        const rackSwitches = topology.switches.filter(s =>
          s.hierarchy_level === 'inter_board' && s.parent_id === currentRack.id
        )
        rackSwitches.forEach(sw => {
          nodeList.push({
            id: sw.id,
            label: sw.label,
            type: 'switch',
            subType: sw.layer,
            isSwitch: true,
            x: 0,
            y: 0,
            color: SWITCH_LAYER_COLORS[sw.layer] || '#666',
            portInfo: {
              uplink: sw.uplink_ports_used,
              downlink: sw.downlink_ports_used,
              inter: sw.inter_ports_used,
            },
          })
        })
      }

      // Board间连接和Switch连接
      const boardIds = new Set(currentRack.boards.map(b => b.id))
      const rackSwitchIds = new Set(
        (topology.switches || [])
          .filter(s => s.hierarchy_level === 'inter_board' && s.parent_id === currentRack.id)
          .map(s => s.id)
      )
      // 构建Board层Switch到Board的映射（用于转换跨层连接）
      const boardSwitchToBoard: Record<string, string> = {}
      ;(topology.switches || [])
        .filter(s => s.hierarchy_level === 'inter_chip' && s.parent_id?.startsWith(currentRack.id))
        .forEach(s => { boardSwitchToBoard[s.id] = s.parent_id! })

      edgeList = topology.connections
        .filter(c => {
          const sourceInRack = boardIds.has(c.source) || rackSwitchIds.has(c.source)
          const targetInRack = boardIds.has(c.target) || rackSwitchIds.has(c.target)
          if (sourceInRack && targetInRack) return true
          // 跨层连接（Rack Switch到Board Switch）- 需要转换
          if (rackSwitchIds.has(c.source) && boardSwitchToBoard[c.target]) return true
          if (rackSwitchIds.has(c.target) && boardSwitchToBoard[c.source]) return true
          return false
        })
        .map(c => {
          // 转换跨层连接：将Board Switch替换为对应的Board
          let source = c.source
          let target = c.target
          if (boardSwitchToBoard[c.source]) source = boardSwitchToBoard[c.source]
          if (boardSwitchToBoard[c.target]) target = boardSwitchToBoard[c.target]
          return {
            source,
            target,
            bandwidth: c.bandwidth,
            latency: c.latency,
            isSwitch: c.type === 'switch',
          }
        })

    } else if (currentLevel === 'board' && currentBoard) {
      // Board层：显示所有Chip和Board层Switch
      graphTitle = `${currentBoard.label} - Chip拓扑`
      nodeList = currentBoard.chips.map((chip) => ({
        id: chip.id,
        label: chip.label || chip.type.toUpperCase(),
        type: chip.type,
        x: 0,
        y: 0,
        color: CHIP_TYPE_COLORS[chip.type] || '#666',
      }))

      // 添加Board层Switch
      if (topology.switches) {
        const boardSwitches = topology.switches.filter(s =>
          s.hierarchy_level === 'inter_chip' && s.parent_id === currentBoard.id
        )
        boardSwitches.forEach(sw => {
          nodeList.push({
            id: sw.id,
            label: sw.label,
            type: 'switch',
            subType: sw.layer,
            isSwitch: true,
            x: 0,
            y: 0,
            color: SWITCH_LAYER_COLORS[sw.layer] || '#666',
            portInfo: {
              uplink: sw.uplink_ports_used,
              downlink: sw.downlink_ports_used,
              inter: sw.inter_ports_used,
            },
          })
        })
      }

      // Chip间连接和Switch连接
      const chipIds = new Set(currentBoard.chips.map(c => c.id))
      const boardSwitchIds = new Set(
        (topology.switches || [])
          .filter(s => s.hierarchy_level === 'inter_chip' && s.parent_id === currentBoard.id)
          .map(s => s.id)
      )
      edgeList = topology.connections
        .filter(c => {
          const sourceInBoard = chipIds.has(c.source) || boardSwitchIds.has(c.source)
          const targetInBoard = chipIds.has(c.target) || boardSwitchIds.has(c.target)
          return sourceInBoard && targetInBoard
        })
        .map(c => ({
          source: c.source,
          target: c.target,
          bandwidth: c.bandwidth,
          isSwitch: c.type === 'switch',
          latency: c.latency,
        }))
    }

    // 获取当前层级的直连拓扑类型和是否保留直连
    let directTopology = 'full_mesh'
    let keepDirectTopology = false
    if (topology.switch_config) {
      if (currentLevel === 'datacenter') {
        const dcConfig = topology.switch_config.inter_pod
        directTopology = dcConfig?.direct_topology || 'full_mesh'
        keepDirectTopology = dcConfig?.enabled && dcConfig?.keep_direct_topology || false
      } else if (currentLevel === 'pod') {
        const podConfig = topology.switch_config.inter_rack
        directTopology = podConfig?.direct_topology || 'full_mesh'
        keepDirectTopology = podConfig?.enabled && podConfig?.keep_direct_topology || false
      } else if (currentLevel === 'rack') {
        const rackConfig = topology.switch_config.inter_board
        directTopology = rackConfig?.direct_topology || 'full_mesh'
        keepDirectTopology = rackConfig?.enabled && rackConfig?.keep_direct_topology || false
      } else if (currentLevel === 'board') {
        const boardConfig = topology.switch_config.inter_chip
        directTopology = boardConfig?.direct_topology || 'full_mesh'
        keepDirectTopology = boardConfig?.enabled && boardConfig?.keep_direct_topology || false
      }
    }

    // 应用布局
    const hasSwitches = nodeList.some(n => n.isSwitch)

    // 用户明确选择布局类型时优先使用
    if (layoutType === 'circle') {
      // 强制环形布局
      if (hasSwitches) {
        // 有Switch时使用混合布局，但设备节点强制环形
        nodeList = hybridLayout(nodeList, width, height, 'ring')
      } else {
        const radius = Math.min(width, height) * 0.35
        nodeList = circleLayout(nodeList, width / 2, height / 2, radius)
      }
    } else if (layoutType === 'grid') {
      // 强制网格布局
      if (hasSwitches) {
        // 有Switch时使用混合布局，但设备节点强制网格
        nodeList = hybridLayout(nodeList, width, height, 'full_mesh_2d')
      } else {
        nodeList = torusLayout(nodeList, width, height)
      }
    } else {
      // auto模式：根据是否有Switch和拓扑类型自动选择
      if (hasSwitches && keepDirectTopology && directTopology !== 'none') {
        // 有Switch且保留直连：使用混合布局（设备按拓扑排列，Switch在上方）
        nodeList = hybridLayout(nodeList, width, height, directTopology)
      } else if (hasSwitches) {
        // 只有Switch（无直连）：使用分层布局
        nodeList = hierarchicalLayout(nodeList, width, height)
      } else {
        // 自动布局：根据直连拓扑类型选择布局
        nodeList = getLayoutForTopology(directTopology, nodeList, width, height)
      }
    }

    return { nodes: nodeList, edges: edgeList, title: graphTitle, directTopology }
  }, [topology, currentLevel, currentPod, currentRack, currentBoard, layoutType, multiLevelOptions])

  // 切换到手动模式时，如果没有保存的位置，使用当前布局位置作为初始值
  useEffect(() => {
    if (isManualMode && Object.keys(manualPositions).length === 0 && nodes.length > 0) {
      // 没有保存的位置，使用当前布局的位置
      const currentPositions: Record<string, { x: number; y: number }> = {}
      nodes.forEach(node => {
        currentPositions[node.id] = { x: node.x, y: node.y }
      })
      setManualPositionsByLayout(prev => ({
        ...prev,
        [layoutType]: currentPositions
      }))
    }
  }, [isManualMode, nodes.length, layoutType])

  // 更新当前布局的手动位置
  const setManualPositions = useCallback((updater: Record<string, { x: number; y: number }> | ((prev: Record<string, { x: number; y: number }>) => Record<string, { x: number; y: number }>)) => {
    setManualPositionsByLayout(prev => ({
      ...prev,
      [layoutType]: typeof updater === 'function' ? updater(prev[layoutType] || {}) : updater
    }))
  }, [layoutType])

  // 当节点列表变化时（数量或ID变化），重置手动位置
  const prevNodeIdsRef = useRef<string>('')
  useEffect(() => {
    const currentNodeIds = nodes.map(n => n.id).sort().join(',')
    if (prevNodeIdsRef.current && prevNodeIdsRef.current !== currentNodeIds) {
      // 节点列表发生变化，重置当前布局的手动位置
      setManualPositionsByLayout(prev => ({
        ...prev,
        [layoutType]: {}
      }))
      setHistory([])
      setHistoryIndex(-1)
    }
    prevNodeIdsRef.current = currentNodeIds
  }, [nodes, layoutType])

  // 应用手动位置调整后的节点列表
  const displayNodes = useMemo(() => {
    if (!isManualMode) return nodes
    return nodes.map(node => {
      const manualPos = manualPositions[node.id]
      if (manualPos) {
        return { ...node, x: manualPos.x, y: manualPos.y }
      }
      return node
    })
  }, [nodes, manualPositions, isManualMode])

  // 根据节点数量计算缩放系数
  const nodeScale = useMemo(() => {
    const deviceNodes = displayNodes.filter(n => !n.isSwitch)
    const count = deviceNodes.length
    if (count <= 4) return 1
    if (count <= 8) return 0.85
    if (count <= 16) return 0.7
    if (count <= 32) return 0.55
    if (count <= 64) return 0.45
    return 0.35
  }, [displayNodes])

  // 创建节点位置映射
  const nodePositions = useMemo(() => {
    const map = new Map<string, { x: number; y: number }>()
    displayNodes.forEach(node => {
      map.set(node.id, { x: node.x, y: node.y })
    })
    return map
  }, [displayNodes])

  const handleZoomIn = () => setZoom(z => Math.min(z + 0.2, 2))
  const handleZoomOut = () => setZoom(z => Math.max(z - 0.2, 0.5))

  // 对齐吸附阈值
  const SNAP_THRESHOLD = 10

  // 圆形布局的参数（与 circleLayout 函数一致）
  const CIRCLE_CENTER = { x: 400, y: 300 }
  const CIRCLE_RADIUS = Math.min(800, 600) * 0.35  // 210

  // 检测对齐并返回吸附后的位置
  const checkAlignment = (x: number, y: number, excludeNodeId: string) => {
    const lines: { type: 'h' | 'v' | 'circle'; pos: number; center?: { x: number; y: number } }[] = []
    let snappedX = x
    let snappedY = y

    // 环形布局：优先检测圆形轨迹吸附
    if (layoutType === 'circle') {
      const dx = x - CIRCLE_CENTER.x
      const dy = y - CIRCLE_CENTER.y
      const distance = Math.sqrt(dx * dx + dy * dy)

      // 检测是否接近圆形轨迹
      if (Math.abs(distance - CIRCLE_RADIUS) < SNAP_THRESHOLD * 2) {
        // 吸附到圆上：保持角度，调整距离到半径
        const angle = Math.atan2(dy, dx)
        snappedX = CIRCLE_CENTER.x + CIRCLE_RADIUS * Math.cos(angle)
        snappedY = CIRCLE_CENTER.y + CIRCLE_RADIUS * Math.sin(angle)
        lines.push({ type: 'circle', pos: CIRCLE_RADIUS, center: CIRCLE_CENTER })
      }
    }

    // 获取其他节点的位置
    const otherNodes = displayNodes.filter(n => n.id !== excludeNodeId)

    for (const node of otherNodes) {
      // 水平对齐检测
      if (Math.abs(node.y - y) < SNAP_THRESHOLD) {
        snappedY = node.y
        lines.push({ type: 'h', pos: node.y })
      }
      // 垂直对齐检测
      if (Math.abs(node.x - x) < SNAP_THRESHOLD) {
        snappedX = node.x
        lines.push({ type: 'v', pos: node.x })
      }
    }

    return { snappedX, snappedY, lines }
  }

  // 保存历史记录
  const saveToHistory = useCallback((positions: Record<string, { x: number; y: number }>) => {
    setHistory(prev => {
      // 删除当前位置之后的历史（重做时）
      const newHistory = prev.slice(0, historyIndex + 1)
      newHistory.push({ ...positions })
      // 限制历史长度
      if (newHistory.length > maxHistoryLength) {
        newHistory.shift()
        return newHistory
      }
      return newHistory
    })
    setHistoryIndex(prev => Math.min(prev + 1, maxHistoryLength - 1))
  }, [historyIndex])

  // 撤销
  const handleUndo = useCallback(() => {
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1
      setHistoryIndex(newIndex)
      setManualPositions(history[newIndex] || {})
    } else if (historyIndex === 0) {
      setHistoryIndex(-1)
      setManualPositions({})
    }
  }, [history, historyIndex])

  // 重做
  const handleRedo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1
      setHistoryIndex(newIndex)
      setManualPositions(history[newIndex])
    }
  }, [history, historyIndex])

  // 键盘快捷键：Ctrl+Z 撤销，Ctrl+Y 重做
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isManualMode) return
      if (e.ctrlKey || e.metaKey) {
        if (e.key === 'z' && !e.shiftKey) {
          e.preventDefault()
          handleUndo()
        } else if (e.key === 'y' || (e.key === 'z' && e.shiftKey)) {
          e.preventDefault()
          handleRedo()
        }
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isManualMode, handleUndo, handleRedo])

  // 计算屏幕坐标到SVG坐标的转换比例
  const getScreenToSvgScale = useCallback(() => {
    if (!svgRef.current) return { scaleX: 1, scaleY: 1 }
    const rect = svgRef.current.getBoundingClientRect()
    // viewBox 尺寸（考虑 zoom）
    const viewBoxWidth = 800 / zoom
    const viewBoxHeight = 600 / zoom
    // 屏幕像素到 SVG 坐标的比例
    const scaleX = viewBoxWidth / rect.width
    const scaleY = viewBoxHeight / rect.height
    return { scaleX, scaleY }
  }, [zoom])

  // 手动布局拖动处理
  const handleDragStart = (nodeId: string, e: React.MouseEvent) => {
    if (!isManualMode || !e.shiftKey) return
    e.preventDefault()
    e.stopPropagation()
    const node = displayNodes.find(n => n.id === nodeId)
    if (!node) return
    setDraggingNode(nodeId)
    setDragStart({ x: e.clientX, y: e.clientY, nodeX: node.x, nodeY: node.y })
  }

  const handleDragMove = (e: React.MouseEvent) => {
    if (!draggingNode || !dragStart) return
    e.preventDefault()

    // 使用正确的坐标转换
    const { scaleX, scaleY } = getScreenToSvgScale()
    const dx = (e.clientX - dragStart.x) * scaleX
    const dy = (e.clientY - dragStart.y) * scaleY
    const rawX = dragStart.nodeX + dx
    const rawY = dragStart.nodeY + dy

    // 检测对齐
    const { snappedX, snappedY, lines } = checkAlignment(rawX, rawY, draggingNode)
    setAlignmentLines(lines)

    setManualPositions(prev => ({
      ...prev,
      [draggingNode]: {
        x: snappedX,
        y: snappedY,
      }
    }))
  }

  const handleDragEnd = () => {
    if (draggingNode) {
      // 保存到历史记录
      saveToHistory(manualPositions)
    }
    setDraggingNode(null)
    setDragStart(null)
    setAlignmentLines([])
  }

  // 重置当前布局的手动位置
  const handleResetManualPositions = useCallback(() => {
    setManualPositionsByLayout(prev => ({
      ...prev,
      [layoutType]: {}
    }))
    setHistory([])
    setHistoryIndex(-1)
    try {
      const key = getManualPositionsCacheKey(layoutType)
      localStorage.removeItem(key)
    } catch (e) { /* ignore */ }
  }, [layoutType, currentLevel, currentPod?.id, currentRack?.id, currentBoard?.id])

  // 工具栏组件
  const toolbar = (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: embedded ? '12px 16px' : 0,
      background: embedded ? '#fff' : 'transparent',
      borderBottom: embedded ? '1px solid #f0f0f0' : 'none',
    }}>
      {embedded && breadcrumbs.length > 0 ? (
        <Breadcrumb
          items={breadcrumbs.map((item, index) => ({
            key: item.id,
            title: (
              <a
                onClick={(e) => {
                  e.preventDefault()
                  onBreadcrumbClick?.(index)
                }}
                style={{
                  cursor: index < breadcrumbs.length - 1 ? 'pointer' : 'default',
                  color: index < breadcrumbs.length - 1 ? '#1890ff' : 'rgba(0, 0, 0, 0.88)',
                  fontWeight: index === breadcrumbs.length - 1 ? 500 : 400,
                }}
              >
                {item.label}
              </a>
            ),
          }))}
        />
      ) : (
        <span style={{ fontWeight: 500 }}>{title || '抽象拓扑图'}</span>
      )}
      <Space>
        <Text type="secondary" style={{ fontSize: 12 }}>
          {directTopology !== 'none' ? `拓扑: ${
            directTopology === 'full_mesh' ? '全连接' :
            directTopology === 'ring' ? '环形' :
            directTopology === 'torus_2d' ? '2D Torus' :
            directTopology === 'torus_3d' ? '3D Torus' :
            directTopology === 'hypercube' ? '超立方体' :
            directTopology === 'star' ? '星形' : directTopology
          }` : ''}
        </Text>
        <Button size="small" icon={<ZoomOutOutlined />} onClick={handleZoomOut} />
        <Button size="small" icon={<ZoomInOutlined />} onClick={handleZoomIn} />
      </Space>
    </div>
  )

  // 图形内容
  const graphContent = (
    <div style={{
      width: '100%',
      height: embedded ? '100%' : 650,
      overflow: 'hidden',
      background: '#fafafa',
      position: 'relative',
    }}>
      {/* 悬浮面包屑导航 */}
      {embedded && breadcrumbs.length > 0 && (
        <div style={{
          position: 'absolute',
          top: 16,
          left: 16,
          zIndex: 100,
          background: '#fff',
          padding: '10px 16px',
          borderRadius: 10,
          border: '1px solid rgba(0, 0, 0, 0.08)',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.06)',
        }}>
          <Breadcrumb
            items={breadcrumbs.map((item, index) => ({
              key: item.id,
              title: (
                <a
                  onClick={(e) => {
                    e.preventDefault()
                    onBreadcrumbClick?.(index)
                  }}
                  style={{
                    cursor: index < breadcrumbs.length - 1 ? 'pointer' : 'default',
                    color: index < breadcrumbs.length - 1 ? '#2563eb' : '#171717',
                    fontWeight: index === breadcrumbs.length - 1 ? 500 : 400,
                  }}
                >
                  {item.label}
                </a>
              ),
            }))}
          />
        </div>
      )}

      {/* 右上角控制面板悬浮框 */}
      {embedded && (
        <div style={{
          position: 'absolute',
          top: 16,
          right: 16,
          zIndex: 100,
          background: '#fff',
          padding: '10px 14px',
          borderRadius: 10,
          border: '1px solid rgba(0, 0, 0, 0.08)',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.06)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            {/* 视图模式切换：单层级/多层级 */}
            <Segmented
              size="small"
              className="topology-layout-segmented"
              value={multiLevelOptions?.enabled ? 'multi' : 'single'}
              onChange={(value) => {
                if (onMultiLevelOptionsChange) {
                  if (value === 'multi') {
                    // 切换到多层级时，根据当前层级自动选择合适的 levelPair
                    let levelPair: 'datacenter_pod' | 'pod_rack' | 'rack_board' | 'board_chip' = multiLevelOptions?.levelPair || 'datacenter_pod'
                    if (currentLevel === 'datacenter') {
                      levelPair = 'datacenter_pod'
                    } else if (currentLevel === 'pod') {
                      levelPair = 'pod_rack'
                    } else if (currentLevel === 'rack') {
                      levelPair = 'rack_board'
                    } else if (currentLevel === 'board') {
                      // chip 是最底层，向上一级显示 rack_board
                      levelPair = 'rack_board'
                    }
                    onMultiLevelOptionsChange({
                      ...multiLevelOptions!,
                      enabled: true,
                      levelPair,
                    })
                  } else {
                    onMultiLevelOptionsChange({
                      ...multiLevelOptions!,
                      enabled: false,
                    })
                  }
                }
              }}
              options={[
                { label: '单层级', value: 'single' },
                { label: '多层级', value: 'multi' },
              ]}
            />
            <div style={{ borderLeft: '1px solid rgba(0, 0, 0, 0.08)', height: 20 }} />
            {/* 布局选择 */}
            <Segmented
              size="small"
              className="topology-layout-segmented"
              value={layoutType}
              onChange={(value) => {
                onLayoutTypeChange?.(value as LayoutType)
                // 切换布局时重置历史
                setHistory([])
                setHistoryIndex(-1)
              }}
              options={[
                { label: '自动', value: 'auto' },
                { label: '环形', value: 'circle' },
                { label: '网格', value: 'grid' },
              ]}
            />
            <div style={{ borderLeft: '1px solid rgba(0, 0, 0, 0.08)', height: 20 }} />
            <Checkbox
              checked={isManualMode}
              onChange={(e) => setIsManualMode(e.target.checked)}
              disabled={multiLevelOptions?.enabled}
            >
              <span style={{ fontSize: 12 }}>手动调整</span>
            </Checkbox>
            {isManualMode && (
              <>
                <Tooltip title="撤销 (Ctrl+Z)">
                  <Button
                    type="text"
                    size="small"
                    icon={<UndoOutlined />}
                    onClick={handleUndo}
                    disabled={historyIndex < 0}
                  />
                </Tooltip>
                <Tooltip title="重做 (Ctrl+Y)">
                  <Button
                    type="text"
                    size="small"
                    icon={<RedoOutlined />}
                    onClick={handleRedo}
                    disabled={historyIndex >= history.length - 1}
                  />
                </Tooltip>
                {Object.keys(manualPositions).length > 0 && (
                  <Tooltip title="重置布局">
                    <Button
                      type="text"
                      size="small"
                      icon={<ReloadOutlined />}
                      onClick={handleResetManualPositions}
                    />
                  </Tooltip>
                )}
              </>
            )}
          </div>
          {isManualMode && (
            <div style={{
              marginTop: 10,
              padding: '8px 12px',
              background: 'rgba(37, 99, 235, 0.06)',
              borderRadius: 8,
              border: '1px solid rgba(37, 99, 235, 0.12)',
              fontSize: 12,
              color: '#2563eb',
              fontWeight: 500,
            }}>
              Shift+拖动 · 自动吸附对齐 · 自动保存
            </div>
          )}
        </div>
      )}

        <svg
          ref={svgRef}
          width="100%"
          height="100%"
          viewBox={`${400 - 400/zoom} ${300 - 300/zoom} ${800 / zoom} ${600 / zoom}`}
          style={{
            display: 'block',
            opacity: viewFadeIn ? 0 : 1,
            transition: 'opacity 0.4s ease-out',
          }}
          onMouseMove={handleDragMove}
          onMouseUp={handleDragEnd}
          onMouseLeave={handleDragEnd}
        >
          {/* 背景层 - 用于点击空白区域清除选中状态 */}
          <rect
            x={400 - 400/zoom}
            y={300 - 300/zoom}
            width={800 / zoom}
            height={600 / zoom}
            fill="transparent"
            onClick={() => {
              if (connectionMode === 'view' && !isManualMode) {
                onNodeClick?.(null)
                onLinkClick?.(null)
              }
            }}
          />
          {/* 定义箭头标记 */}
          <defs>
            <marker
              id="arrowhead"
              markerWidth="10"
              markerHeight="7"
              refX="9"
              refY="3.5"
              orient="auto"
            >
              <polygon points="0 0, 10 3.5, 0 7" fill="#999" />
            </marker>
            {/* 手动连接箭头 */}
            <marker
              id="arrowhead-manual"
              markerWidth="10"
              markerHeight="7"
              refX="9"
              refY="3.5"
              orient="auto"
            >
              <polygon points="0 0, 10 3.5, 0 7" fill="#52c41a" />
            </marker>
          </defs>

          {/* 手动布局时的辅助对齐线 */}
          {isManualMode && alignmentLines.map((line, idx) => {
            if (line.type === 'h') {
              return (
                <line
                  key={`align-h-${idx}`}
                  x1={0}
                  y1={line.pos}
                  x2={800}
                  y2={line.pos}
                  stroke="#1890ff"
                  strokeWidth={1}
                  strokeDasharray="4 4"
                  opacity={0.8}
                />
              )
            } else if (line.type === 'v') {
              return (
                <line
                  key={`align-v-${idx}`}
                  x1={line.pos}
                  y1={0}
                  x2={line.pos}
                  y2={600}
                  stroke="#1890ff"
                  strokeWidth={1}
                  strokeDasharray="4 4"
                  opacity={0.8}
                />
              )
            } else if (line.type === 'circle' && line.center) {
              return (
                <circle
                  key={`align-circle-${idx}`}
                  cx={line.center.x}
                  cy={line.center.y}
                  r={line.pos}
                  fill="none"
                  stroke="#52c41a"
                  strokeWidth={2}
                  strokeDasharray="8 4"
                  opacity={0.6}
                />
              )
            }
            return null
          })}

          {/* 多层级模式：按层级分组渲染（容器+边+节点），实现正确遮挡 */}
          {multiLevelOptions?.enabled && (() => {
            const containers = displayNodes
              .filter(n => n.isContainer && n.containerBounds)
              .sort((a, b) => (b.zLayer ?? 0) - (a.zLayer ?? 0))  // 远处先渲染

            // 悬停时上面的层向上移动的距离
            const liftDistance = 150

            // 查找选中容器的 zLayer（支持选中容器或容器内节点，用于保持选中时的移动效果）
            let selectedLayerIndex: number | null = null
            const selectedContainer = containers.find(c => c.id === selectedNodeId)
            if (selectedContainer) {
              selectedLayerIndex = selectedContainer.zLayer ?? null
            } else if (selectedNodeId) {
              // 如果选中的是容器内的节点，查找包含该节点的容器
              for (const container of containers) {
                if (container.singleLevelData?.nodes.some(n => n.id === selectedNodeId)) {
                  selectedLayerIndex = container.zLayer ?? null
                  break
                }
              }
            }

            // 获取所有inter_level边，并计算它们应该在哪个容器中渲染
            const interLevelEdges = edges.filter(e => e.connectionType === 'inter_level')
            const baseSkewAngle = -30
            const skewTan = Math.tan(baseSkewAngle * Math.PI / 180)

            // 查找节点所在的容器及其zLayer和索引
            const getNodeContainerInfo = (nodeId: string) => {
              for (const container of containers) {
                if (container.singleLevelData?.nodes.some(n => n.id === nodeId)) {
                  // 从容器ID中提取索引（例如 "pod_0/rack_2" -> 2）
                  const idParts = container.id.split('/')
                  const lastPart = idParts[idParts.length - 1]
                  const indexMatch = lastPart.match(/_(\d+)$/)
                  const containerIndex = indexMatch ? parseInt(indexMatch[1], 10) : 0
                  return { container, zLayer: container.zLayer ?? 0, containerIndex }
                }
              }
              return null
            }

            // 计算节点位置（考虑yOffset和skew变换）
            const getNodePosition = (nodeId: string, activeLayerIdx: number | null) => {
              for (const container of containers) {
                if (container.singleLevelData) {
                  const slNode = container.singleLevelData.nodes.find(n => n.id === nodeId)
                  if (slNode && container.containerBounds) {
                    const bounds = container.containerBounds
                    const viewBox = container.singleLevelData.viewBox
                    const zLayer = container.zLayer ?? 0

                    let yOffset = 0
                    if (activeLayerIdx !== null && zLayer < activeLayerIdx) {
                      yOffset = -liftDistance
                    }

                    const labelHeight = 10
                    const sidePadding = 10
                    const topPadding = 10
                    const svgWidth = bounds.width - sidePadding * 2
                    const svgHeight = bounds.height - labelHeight - topPadding
                    const svgX = bounds.x + sidePadding
                    const svgY = bounds.y + topPadding

                    const scaleX = svgWidth / viewBox.width
                    const scaleY = svgHeight / viewBox.height
                    const scale = Math.min(scaleX, scaleY)

                    const offsetX = (svgWidth - viewBox.width * scale) / 2
                    const offsetY = (svgHeight - viewBox.height * scale) / 2

                    const baseX = svgX + offsetX + slNode.x * scale
                    const baseY = svgY + offsetY + slNode.y * scale
                    const centerY = bounds.y + bounds.height / 2
                    const skewedX = baseX + (baseY - centerY) * skewTan

                    return { x: skewedX, y: baseY + yOffset, zLayer }
                  }
                }
              }
              return null
            }

            // 计算视口中心和尺寸（用于展开动画）
            // viewBox: ${400 - 400/zoom} ${300 - 300/zoom} ${800/zoom} ${600/zoom}
            const viewportWidth = 800 / zoom
            const viewportHeight = 600 / zoom
            // viewBox 中心点始终是 (400, 300)
            const viewportCenterX = 400
            const viewportCenterY = 300

            // 按层级分组渲染（zLayer大的在下面，先渲染；zLayer小的在上面，后渲染覆盖）
            // 获取当前层级的手动连接（用于最后渲染）
            const currentManualConnections = manualConnections.filter(mc => mc.hierarchy_level === getCurrentHierarchyLevel())

            // 渲染跨容器手动连接（在所有容器之上）
            const renderManualConnectionsOverlay = () => {
              const activeLayerIdx = hoveredLayerIndex ?? selectedLayerIndex
              const getParentIdx = (nodeId: string): number => {
                const parts = nodeId.split('/')
                if (parts.length >= 2) {
                  const parentPart = parts[parts.length - 2]
                  const match = parentPart.match(/_(\d+)$/)
                  return match ? parseInt(match[1], 10) : 0
                }
                return 0
              }

              // 准备容器边界信息（用于分段绘制）
              const containerBoundsInfo = containers.map(c => ({
                zLayer: c.zLayer ?? 0,
                bounds: c.containerBounds!,
              }))

              return currentManualConnections.map((conn) => {
                const sourcePos = getNodePosition(conn.source, activeLayerIdx)
                const targetPos = getNodePosition(conn.target, activeLayerIdx)

                const sourceParentIdx = getParentIdx(conn.source)
                const targetParentIdx = getParentIdx(conn.target)
                const indexDiff = Math.abs(sourceParentIdx - targetParentIdx)
                const isCrossContainer = indexDiff >= 1  // 只要不在同一容器就是跨容器

                const manualEdgeId = `${conn.source}-${conn.target}`
                const isLinkSelected = selectedLinkId === manualEdgeId || selectedLinkId === `${conn.target}-${conn.source}`

                const handleManualClick = (e: React.MouseEvent) => {
                  e.stopPropagation()
                  if (connectionMode !== 'view') return
                  const getDetailedLabel = (nodeId: string) => {
                    const parts = nodeId.split('/')
                    return parts.length >= 2 ? parts.slice(-2).join('/') : nodeId
                  }
                  onLinkClick?.({
                    id: manualEdgeId,
                    sourceId: conn.source,
                    sourceLabel: getDetailedLabel(conn.source),
                    sourceType: conn.source.split('/').pop()?.split('_')[0] || 'unknown',
                    targetId: conn.target,
                    targetLabel: getDetailedLabel(conn.target),
                    targetType: conn.target.split('/').pop()?.split('_')[0] || 'unknown',
                    isManual: true
                  })
                }

                return (
                  <ManualConnectionLine
                    key={`manual-conn-${conn.id}`}
                    conn={conn}
                    sourcePos={sourcePos}
                    targetPos={targetPos}
                    isSelected={isLinkSelected}
                    isCrossContainer={isCrossContainer}
                    indexDiff={indexDiff}
                    onClick={handleManualClick}
                    layoutType={layoutType}
                    containers={containerBoundsInfo}
                  />
                )
              })
            }

            const containersRendered = containers.map(containerNode => {
              const bounds = containerNode.containerBounds!
              const zLayer = containerNode.zLayer ?? 0
              const isExpanding = expandingContainer?.id === containerNode.id
              const isOtherExpanding = expandingContainer !== null && !isExpanding
              const isCollapsing = collapsingContainer?.id === containerNode.id
              const isOtherCollapsing = collapsingContainer !== null && !isCollapsing

              // 计算悬停或选中时的偏移和透明度
              // zLayer=0在最上面，悬停/选中某层时，上面的层（zLayer更小）向上移动
              // 优先使用悬停的层，如果没有悬停则使用选中的层
              const activeLayerIndex = hoveredLayerIndex ?? selectedLayerIndex
              let yOffset = 0
              let layerOpacity = 1
              if (activeLayerIndex !== null && zLayer < activeLayerIndex && !expandingContainer && !collapsingContainer) {
                yOffset = -liftDistance  // 向上移动
                layerOpacity = 0.3  // 上面的层变透明
              }

              // 展开动画时其他容器淡出
              if (isOtherExpanding) {
                layerOpacity = 0
              }

              // 收缩动画时其他容器从透明淡入
              if (isOtherCollapsing) {
                if (!collapseAnimationStarted) {
                  layerOpacity = 0  // 初始状态：透明
                } else {
                  layerOpacity = 1  // 动画开始后：过渡到完全可见
                }
              }

              const x = bounds.x
              const w = bounds.width
              const h = bounds.height
              const y = bounds.y
              const isHoveredLayer = hoveredLayerIndex === zLayer && !expandingContainer && !collapsingContainer

              // 获取该层的子节点
              const layerNodes = displayNodes.filter(n => !n.isContainer && n.zLayer === zLayer)
              // 获取该层的边（两端都在该层）
              const layerEdges = edges.filter(e => {
                const sourceNode = displayNodes.find(n => n.id === e.source)
                const targetNode = displayNodes.find(n => n.id === e.target)
                return sourceNode?.zLayer === zLayer && targetNode?.zLayer === zLayer
              })

              // 容器中心点
              const centerX = x + w / 2
              const centerY = y + h / 2

              // 3D书本效果参数
              const baseSkewAngle = -30  // 基础斜切角度
              // 展开时 skew 角度变为 0
              // 收缩时：开始阶段为0，动画开始后变回baseSkewAngle
              let currentSkewAngle = baseSkewAngle
              if (isExpanding) {
                currentSkewAngle = 0
              } else if (isCollapsing && !collapseAnimationStarted) {
                currentSkewAngle = 0  // 收缩开始前保持展开状态
              }

              // 计算展开/收缩动画的变换参数
              const contentScale = containerNode.singleLevelData?.scale ?? 0.25
              const padding = 40
              const targetContentWidth = viewportWidth - padding * 2
              const targetContentHeight = viewportHeight - padding * 2
              const scaleToFitWidth = targetContentWidth / (800 * contentScale)
              const scaleToFitHeight = targetContentHeight / (600 * contentScale)
              const fullExpandScale = Math.min(scaleToFitWidth, scaleToFitHeight)
              const fullExpandTranslateX = viewportCenterX - centerX
              const fullExpandTranslateY = viewportCenterY - centerY

              // 计算当前帧的变换
              let expandScale = 1
              let expandTranslateX = 0
              let expandTranslateY = 0
              if (isExpanding) {
                expandScale = fullExpandScale
                expandTranslateX = fullExpandTranslateX
                expandTranslateY = fullExpandTranslateY
              } else if (isCollapsing && !collapseAnimationStarted) {
                // 收缩开始前：显示展开状态
                expandScale = fullExpandScale
                expandTranslateX = fullExpandTranslateX
                expandTranslateY = fullExpandTranslateY
              }
              // 收缩动画开始后：expandScale=1, translate=0 (默认值)，容器会过渡到正常状态

              // 是否正在进行展开或收缩动画
              const isAnimating = expandingContainer !== null || collapsingContainer !== null

              // 检测容器是否被选中
              const isContainerSelected = selectedNodeId === containerNode.id
              const isContainerHighlighted = isHoveredLayer || isContainerSelected

              return (
                <g
                  key={`layer-${zLayer}-${containerNode.id}`}
                  style={{
                    transition: isAnimating
                      ? 'transform 0.5s ease-in-out, opacity 0.4s ease-out'
                      : 'transform 0.3s ease-out, opacity 0.3s ease-out',
                    transform: (isExpanding || (isCollapsing && !collapseAnimationStarted))
                      ? `translate(${expandTranslateX}px, ${expandTranslateY}px) scale(${expandScale})`
                      : `translateY(${yOffset}px)`,
                    opacity: layerOpacity,
                    transformOrigin: `${centerX}px ${centerY}px`,
                  }}
                  onMouseEnter={() => !isAnimating && setHoveredLayerIndex(zLayer)}
                  onMouseLeave={() => !isAnimating && setHoveredLayerIndex(null)}
                  onTransitionEnd={(e) => {
                    // 展开动画完成后触发导航
                    if (isExpanding && e.propertyName === 'transform' && onNodeDoubleClick) {
                      onNodeDoubleClick(containerNode.id, containerNode.type)
                      setExpandingContainer(null)
                    }
                  }}
                >
                  {/* 容器：使用 rect + skewX transform 实现平行四边形，这样 skew 角度可以动画化 */}
                  <g
                    style={{
                      transition: isAnimating
                        ? 'transform 0.5s ease-in-out'
                        : 'transform 0.3s ease-out',
                      transform: `skewX(${currentSkewAngle}deg)`,
                      transformOrigin: `${centerX}px ${centerY}px`,
                    }}
                  >
                    <rect
                      x={x}
                      y={y}
                      width={w}
                      height={h}
                      fill="#f8f9fa"
                      stroke={isContainerHighlighted ? containerNode.color : '#d0d0d0'}
                      strokeWidth={isContainerHighlighted ? 2.5 : 1.5}
                      style={{
                        cursor: isAnimating ? 'default' : 'pointer',
                        filter: `drop-shadow(0 ${isContainerHighlighted ? 8 : 3}px ${isContainerHighlighted ? 16 : 6}px rgba(0,0,0,${isContainerHighlighted ? 0.2 : 0.1}))`,
                      }}
                      onClick={(e) => {
                        e.stopPropagation()
                        if (isAnimating || connectionMode !== 'view') return
                        // 单击容器：选中容器并通知父组件
                        if (onNodeClick) {
                          // 计算容器内的连接数
                          const containerConnections = layerEdges.map(edge => {
                            const targetNode = layerNodes.find(n => n.id === (edge.source === containerNode.id ? edge.target : edge.source))
                            return {
                              id: edge.source === containerNode.id ? edge.target : edge.source,
                              label: targetNode?.label || '',
                              bandwidth: edge.bandwidth,
                              latency: edge.latency,
                            }
                          })
                          // 容器的 type（pod/rack/board）就是它的层级
                          onNodeClick({
                            id: containerNode.id,
                            label: containerNode.label,
                            type: containerNode.type,
                            subType: containerNode.type,  // 使用 type 作为层级标识
                            connections: containerConnections,
                          })
                        }
                      }}
                      onDoubleClick={() => {
                        if (!isAnimating) {
                          // 开始展开动画
                          setExpandingContainer({ id: containerNode.id, type: containerNode.type })
                        }
                      }}
                    />
                  </g>

                  {/* 渲染与该容器相关的跨层级边（在容器背景之后、节点之前，确保边在节点下方） */}
                  {(() => {
                    const activeLayerIdx = hoveredLayerIndex ?? selectedLayerIndex

                    // 找出应该在当前容器渲染的inter_level边
                    const edgesToRender = interLevelEdges.filter(edge => {
                      const sourceInfo = getNodeContainerInfo(edge.source)
                      const targetInfo = getNodeContainerInfo(edge.target)
                      if (!sourceInfo || !targetInfo) return false
                      const maxZLayer = Math.max(sourceInfo.zLayer, targetInfo.zLayer)
                      return maxZLayer === zLayer
                    })

                    if (edgesToRender.length === 0) return null

                    return (
                      <g>
                        {edgesToRender.map((edge, i) => {
                          const sourcePos = getNodePosition(edge.source, activeLayerIdx)
                          const targetPos = getNodePosition(edge.target, activeLayerIdx)
                          if (!sourcePos || !targetPos) return null

                          const edgeId = `${edge.source}-${edge.target}`
                          const isLinkSelected = selectedLinkId === edgeId || selectedLinkId === `${edge.target}-${edge.source}`
                          const strokeColor = isLinkSelected ? '#52c41a' : '#722ed1'
                          const strokeWidth = isLinkSelected ? 3 : 2
                          // 直接从节点ID路径中提取父容器索引
                          // 例如 "pod_0/rack_0/board_4" -> 提取 "rack_0" -> 索引 0
                          const getParentIndex = (nodeId: string): { index: number; parentPart: string } => {
                            const parts = nodeId.split('/')
                            if (parts.length >= 2) {
                              // 取倒数第二部分（父容器）
                              const parentPart = parts[parts.length - 2]
                              const match = parentPart.match(/_(\d+)$/)
                              return { index: match ? parseInt(match[1], 10) : 0, parentPart }
                            }
                            return { index: 0, parentPart: '' }
                          }
                          const sourceParentInfo = getParentIndex(edge.source)
                          const targetParentInfo = getParentIndex(edge.target)
                          const indexDiff = Math.abs(sourceParentInfo.index - targetParentInfo.index)
                          const zLayerDiff = Math.abs(sourcePos.zLayer - targetPos.zLayer)
                          // 相邻条件：zLayer差≤1 且 父容器索引差≤1
                          const isAdjacent = zLayerDiff <= 1 && indexDiff <= 1

                          const handleClick = (e: React.MouseEvent) => {
                            e.stopPropagation()
                            if (connectionMode !== 'view') return
                            // 从容器的 singleLevelData 中查找节点详细信息
                            const findNodeInContainers = (nodeId: string) => {
                              for (const c of containers) {
                                const node = c.singleLevelData?.nodes.find(n => n.id === nodeId)
                                if (node) return { node, container: c }
                              }
                              return null
                            }
                            const sourceResult = findNodeInContainers(edge.source)
                            const targetResult = findNodeInContainers(edge.target)
                            // 构建详细的标签：包含容器路径
                            const sourceContainer = sourceResult?.container
                            const targetContainer = targetResult?.container
                            const sourceLabel = sourceResult?.node
                              ? `${sourceContainer?.label || ''}/${sourceResult.node.label}`
                              : edge.source
                            const targetLabel = targetResult?.node
                              ? `${targetContainer?.label || ''}/${targetResult.node.label}`
                              : edge.target
                            onLinkClick?.({
                              id: edgeId,
                              sourceId: edge.source,
                              sourceLabel,
                              sourceType: sourceResult?.node?.type || 'unknown',
                              targetId: edge.target,
                              targetLabel,
                              targetType: targetResult?.node?.type || 'unknown',
                              bandwidth: edge.bandwidth,
                              latency: edge.latency,
                              isManual: false
                            })
                          }

                          if (isAdjacent) {
                            return (
                              <g key={`inter-level-edge-${containerNode.id}-${i}`} style={{ transition: 'all 0.3s ease-out' }}>
                                <line x1={sourcePos.x} y1={sourcePos.y} x2={targetPos.x} y2={targetPos.y}
                                  stroke="transparent" strokeWidth={12} style={{ cursor: 'pointer' }} onClick={handleClick} />
                                <line x1={sourcePos.x} y1={sourcePos.y} x2={targetPos.x} y2={targetPos.y}
                                  stroke={strokeColor} strokeWidth={strokeWidth} strokeDasharray="8,4"
                                  strokeOpacity={isLinkSelected ? 1 : 0.8}
                                  style={{ pointerEvents: 'none', filter: isLinkSelected ? 'drop-shadow(0 0 4px #52c41a)' : 'none' }} />
                              </g>
                            )
                          } else {
                            const midX = (sourcePos.x + targetPos.x) / 2
                            const midY = (sourcePos.y + targetPos.y) / 2
                            const dx = targetPos.x - sourcePos.x
                            const dy = targetPos.y - sourcePos.y
                            const dist = Math.sqrt(dx * dx + dy * dy)
                            // 弯曲程度基于zLayer差和容器索引差
                            const maxDiff = Math.max(zLayerDiff, indexDiff, 1)
                            const bulge = dist * 0.2 * maxDiff
                            const perpX = -dy / dist
                            const perpY = dx / dist
                            const ctrlX = midX + perpX * bulge
                            const ctrlY = midY + perpY * bulge
                            const pathD = `M ${sourcePos.x} ${sourcePos.y} Q ${ctrlX} ${ctrlY}, ${targetPos.x} ${targetPos.y}`

                            return (
                              <g key={`inter-level-edge-${containerNode.id}-${i}`} style={{ transition: 'all 0.3s ease-out' }}>
                                <path d={pathD} fill="none" stroke="transparent" strokeWidth={12}
                                  style={{ cursor: 'pointer' }} onClick={handleClick} />
                                <path d={pathD} fill="none" stroke={strokeColor} strokeWidth={strokeWidth}
                                  strokeDasharray="8,4" strokeOpacity={isLinkSelected ? 1 : 0.8}
                                  style={{ pointerEvents: 'none', filter: isLinkSelected ? 'drop-shadow(0 0 4px #52c41a)' : 'none' }} />
                              </g>
                            )
                          }
                        })}
                      </g>
                    )
                  })()}

                  {/* 内部元素应用 skewX 变换（同样可动画化） */}
                  <g
                    style={{
                      transition: isAnimating
                        ? 'transform 0.5s ease-in-out'
                        : 'transform 0.3s ease-out',
                      transform: `skewX(${currentSkewAngle}deg)`,
                      transformOrigin: `${centerX}px ${centerY}px`,
                    }}
                  >
                    {/* 容器标签 - 左下角 */}
                    <text
                      x={x + 12}
                      y={y + h - 10}
                      fill={containerNode.color}
                      fontSize={12}
                      fontWeight={600}
                    >
                      {containerNode.label}
                    </text>

                    {/* 使用嵌套 SVG 渲染单层级内容（用于展开动画平滑过渡） */}
                    {containerNode.singleLevelData && (() => {
                      const { nodes: slNodes, edges: slEdges, viewBox, directTopology: slDirectTopology } = containerNode.singleLevelData
                      // 让拓扑填充尽可能多的容器空间
                      const labelHeight = 10  // 留给标签的空间
                      const sidePadding = 10
                      const topPadding = 10
                      // SVG 填充容器的大部分空间
                      const svgWidth = w - sidePadding * 2
                      const svgHeight = h - labelHeight - topPadding
                      const svgX = x + sidePadding
                      const svgY = y + topPadding

                      // 计算节点缩放（与单层级视图一致）
                      const deviceCount = slNodes.filter(n => !n.isSwitch).length
                      const slNodeScale = deviceCount > 20 ? 0.8 : deviceCount > 10 ? 0.9 : 1.0

                      return (
                        <svg
                          x={svgX}
                          y={svgY}
                          width={svgWidth}
                          height={svgHeight}
                          viewBox={`0 0 ${viewBox.width} ${viewBox.height}`}
                          preserveAspectRatio="xMidYMid meet"
                          overflow="hidden"
                        >
                          {/* 透明背景 - 点击空白区域：查看模式选中容器，连接模式不做处理 */}
                          <rect
                            x={0}
                            y={0}
                            width={viewBox.width}
                            height={viewBox.height}
                            fill="transparent"
                            style={{ cursor: connectionMode !== 'view' ? 'crosshair' : 'pointer' }}
                            onClick={(e) => {
                              e.stopPropagation()
                              // 连接模式下点击空白区域不做处理
                              if (connectionMode !== 'view') return
                              // 查看模式：点击容器内空白区域选中容器
                              if (onNodeClick) {
                                const containerConnections = layerEdges.map(edge => {
                                  const targetNode = layerNodes.find(n => n.id === (edge.source === containerNode.id ? edge.target : edge.source))
                                  return {
                                    id: edge.source === containerNode.id ? edge.target : edge.source,
                                    label: targetNode?.label || '',
                                    bandwidth: edge.bandwidth,
                                    latency: edge.latency,
                                  }
                                })
                                onNodeClick({
                                  id: containerNode.id,
                                  label: containerNode.label,
                                  type: containerNode.type,
                                  subType: containerNode.type,
                                  connections: containerConnections,
                                })
                              }
                              onLinkClick?.(null)
                            }}
                          />
                          {/* 渲染边 - 使用与单层级相同的颜色和交互逻辑 */}
                          {slEdges.map((edge, i) => {
                            const sourceNode = slNodes.find(n => n.id === edge.source)
                            const targetNode = slNodes.find(n => n.id === edge.target)
                            if (!sourceNode || !targetNode) return null

                            // 检测边是否被选中
                            const edgeId = `${edge.source}-${edge.target}`
                            const reverseEdgeId = `${edge.target}-${edge.source}`
                            const isEdgeSelected = selectedLinkId === edgeId || selectedLinkId === reverseEdgeId

                            // 边颜色：与单层级视图保持一致，优先使用热力图颜色
                            let edgeColor = '#b0b0b0'  // 默认灰色
                            let strokeWidth = 1.5
                            const slTrafficStyle = getTrafficHeatmapStyle(edge.source, edge.target)
                            if (isEdgeSelected) {
                              edgeColor = '#52c41a'  // 选中：绿色
                              strokeWidth = 3
                            } else if (slTrafficStyle) {
                              edgeColor = slTrafficStyle.stroke  // 热力图颜色
                              strokeWidth = slTrafficStyle.strokeWidth
                            } else if (edge.isSwitch) {
                              edgeColor = '#1890ff'  // Switch 连接：蓝色
                              strokeWidth = 2
                            }

                            // tooltip 内容
                            const bandwidthStr = edge.bandwidth ? `${edge.bandwidth}Gbps` : ''
                            const latencyStr = edge.latency ? `${edge.latency}ns` : ''
                            const slTrafficStr = slTrafficStyle ? `流量: ${slTrafficStyle.trafficMb.toFixed(1)}MB, 利用率: ${(slTrafficStyle.utilization * 100).toFixed(0)}%` : ''
                            const propsStr = [bandwidthStr, latencyStr, slTrafficStr].filter(Boolean).join(', ')
                            const edgeTooltip = `${sourceNode.label} ↔ ${targetNode.label}${propsStr ? ` (${propsStr})` : ''}`

                            // 点击处理
                            const handleEdgeClick = (e: React.MouseEvent) => {
                              e.stopPropagation()
                              if (connectionMode !== 'view') return
                              if (onLinkClick) {
                                onLinkClick({
                                  id: edgeId,
                                  sourceId: edge.source,
                                  sourceLabel: sourceNode.label,
                                  sourceType: sourceNode.type,
                                  targetId: edge.target,
                                  targetLabel: targetNode.label,
                                  targetType: targetNode.type,
                                  bandwidth: edge.bandwidth,
                                  latency: edge.latency,
                                  isManual: false,
                                })
                              }
                            }
                            const handleEdgeMouseEnter = (e: React.MouseEvent) => {
                              if (connectionMode !== 'view') return
                              setTooltip({
                                x: e.clientX - (svgRef.current?.getBoundingClientRect().left || 0),
                                y: e.clientY - (svgRef.current?.getBoundingClientRect().top || 0) + 15,
                                content: edgeTooltip,
                              })
                            }
                            const handleEdgeMouseLeave = () => connectionMode === 'view' && setTooltip(null)

                            // 2D FullMesh：非相邻节点使用曲线连接
                            if (slDirectTopology === 'full_mesh_2d') {
                              const sourceGridRow = sourceNode.gridRow
                              const sourceGridCol = sourceNode.gridCol
                              const targetGridRow = targetNode.gridRow
                              const targetGridCol = targetNode.gridCol
                              const sameRow = sourceGridRow === targetGridRow
                              const sameCol = sourceGridCol === targetGridCol
                              const colDiff = Math.abs((sourceGridCol || 0) - (targetGridCol || 0))
                              const rowDiff = Math.abs((sourceGridRow || 0) - (targetGridRow || 0))

                              // 同行非相邻（列差>1）或同列非相邻（行差>1）使用曲线
                              if ((sameRow && colDiff > 1) || (sameCol && rowDiff > 1)) {
                                const midX = (sourceNode.x + targetNode.x) / 2
                                const midY = (sourceNode.y + targetNode.y) / 2
                                const dx = targetNode.x - sourceNode.x
                                const dy = targetNode.y - sourceNode.y
                                const dist = Math.sqrt(dx * dx + dy * dy)

                                const bulge = dist * 0.25 + (sourceGridRow || 0) * 5
                                const perpX = -dy / dist
                                const perpY = dx / dist
                                const ctrlX = midX + perpX * bulge
                                const ctrlY = midY + perpY * bulge

                                const pathD = `M ${sourceNode.x} ${sourceNode.y} Q ${ctrlX} ${ctrlY}, ${targetNode.x} ${targetNode.y}`
                                return (
                                  <g key={`sl-edge-${i}`}>
                                    <path
                                      d={pathD}
                                      fill="none"
                                      stroke="transparent"
                                      strokeWidth={12}
                                      style={{ cursor: 'pointer' }}
                                      onClick={handleEdgeClick}
                                      onMouseEnter={handleEdgeMouseEnter}
                                      onMouseLeave={handleEdgeMouseLeave}
                                    />
                                    <path
                                      d={pathD}
                                      fill="none"
                                      stroke={edgeColor}
                                      strokeWidth={strokeWidth}
                                      strokeOpacity={isEdgeSelected ? 1 : 0.7}
                                      style={{
                                        pointerEvents: 'none',
                                        filter: isEdgeSelected ? 'drop-shadow(0 0 4px #52c41a)' : 'none',
                                      }}
                                    />
                                  </g>
                                )
                              }
                            }

                            // 普通直线连接
                            return (
                              <g key={`sl-edge-${i}`}>
                                {/* 透明触发层 - 增大点击区域 */}
                                <line
                                  x1={sourceNode.x}
                                  y1={sourceNode.y}
                                  x2={targetNode.x}
                                  y2={targetNode.y}
                                  stroke="transparent"
                                  strokeWidth={12}
                                  style={{ cursor: 'pointer' }}
                                  onClick={handleEdgeClick}
                                  onMouseEnter={handleEdgeMouseEnter}
                                  onMouseLeave={handleEdgeMouseLeave}
                                />
                                {/* 可见线条 */}
                                <line
                                  x1={sourceNode.x}
                                  y1={sourceNode.y}
                                  x2={targetNode.x}
                                  y2={targetNode.y}
                                  stroke={edgeColor}
                                  strokeWidth={strokeWidth}
                                  strokeOpacity={isEdgeSelected ? 1 : 0.7}
                                  style={{
                                    pointerEvents: 'none',
                                    filter: isEdgeSelected ? 'drop-shadow(0 0 4px #52c41a)' : 'none',
                                  }}
                                />
                              </g>
                            )
                          })}

                          {/* 渲染节点 - 使用与单层级相同的高亮和交互逻辑 */}
                          {slNodes.map(node => {
                            // 检测节点状态（连接模式）
                            const isSourceSelected = selectedNodes.has(node.id)
                            const isTargetSelected = targetNodes.has(node.id)
                            // 检测节点状态（查看模式）
                            const isNodeSelected = selectedNodeId === node.id
                            const isNodeHovered = hoveredNodeId === node.id
                            const isLinkEndpoint = selectedLinkId && (
                              selectedLinkId.startsWith(node.id + '-') ||
                              selectedLinkId.endsWith('-' + node.id)
                            )
                            const shouldHighlight = isNodeSelected || isNodeHovered || isLinkEndpoint || isSourceSelected || isTargetSelected

                            // 节点连接信息（用于 tooltip 和 onClick）
                            const nodeConnections = slEdges
                              .filter(e => e.source === node.id || e.target === node.id)
                              .map(e => {
                                const otherId = e.source === node.id ? e.target : e.source
                                const otherNode = slNodes.find(n => n.id === otherId)
                                return { id: otherId, label: otherNode?.label || otherId, bandwidth: e.bandwidth, latency: e.latency }
                              })

                            // 根据模式决定高亮颜色
                            let highlightFilter = 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))'
                            if (isSourceSelected) {
                              highlightFilter = 'drop-shadow(0 0 8px rgba(24, 144, 255, 0.8)) drop-shadow(0 0 16px rgba(24, 144, 255, 0.4))'  // 蓝色：源节点
                            } else if (isTargetSelected) {
                              highlightFilter = 'drop-shadow(0 0 8px rgba(82, 196, 26, 0.8)) drop-shadow(0 0 16px rgba(82, 196, 26, 0.4))'  // 绿色：目标节点
                            } else if (shouldHighlight) {
                              highlightFilter = 'drop-shadow(0 0 8px rgba(37, 99, 235, 0.6)) drop-shadow(0 0 16px rgba(37, 99, 235, 0.3))'
                            }

                            // 根据节点类型获取背景尺寸（用于遮挡边）
                            const bgSize = node.type === 'board' ? { x: -32, y: -18, w: 64, h: 36 }
                              : (node.type === 'npu' || node.type === 'cpu') ? { x: -26, y: -26, w: 52, h: 52 }
                              : { x: -25, y: -18, w: 50, h: 36 }

                            return (
                              <g
                                key={`sl-node-${node.id}`}
                                transform={`translate(${node.x}, ${node.y}) scale(${slNodeScale})`}
                                style={{
                                  cursor: connectionMode !== 'view' ? 'crosshair' : 'pointer',
                                }}
                                onClick={(e) => {
                                  e.stopPropagation()
                                  // 连接模式：选择源节点
                                  if (connectionMode === 'select_source' || connectionMode === 'select' || connectionMode === 'connect') {
                                    const currentSet = new Set(selectedNodes)
                                    if (currentSet.has(node.id)) {
                                      currentSet.delete(node.id)
                                    } else {
                                      currentSet.add(node.id)
                                    }
                                    onSelectedNodesChange?.(currentSet)
                                  } else if (connectionMode === 'select_target') {
                                    // 连接模式：选择目标节点
                                    const currentSet = new Set(targetNodes)
                                    if (currentSet.has(node.id)) {
                                      currentSet.delete(node.id)
                                    } else {
                                      currentSet.add(node.id)
                                    }
                                    onTargetNodesChange?.(currentSet)
                                  } else {
                                    // 查看模式
                                    if (onNodeClick) {
                                      onNodeClick({
                                        id: node.id,
                                        label: node.label,
                                        type: node.type,
                                        subType: node.subType,
                                        connections: nodeConnections,
                                      })
                                    }
                                  }
                                }}
                                onDoubleClick={(e) => {
                                  e.stopPropagation()
                                  if (connectionMode === 'view' && onNodeDoubleClick && !node.isSwitch) {
                                    onNodeDoubleClick(node.id, node.type)
                                  }
                                }}
                                onMouseEnter={(e) => {
                                  setHoveredNodeId(node.id)
                                  if (connectionMode !== 'view') return
                                  setTooltip({
                                    x: e.clientX - (svgRef.current?.getBoundingClientRect().left || 0),
                                    y: e.clientY - (svgRef.current?.getBoundingClientRect().top || 0) + 15,
                                    content: `${node.label} (${node.type}) - ${nodeConnections.length} 连接`,
                                  })
                                }}
                                onMouseLeave={() => {
                                  setHoveredNodeId(null)
                                  if (connectionMode !== 'view') return
                                  setTooltip(null)
                                }}
                              >
                                {/* 遮挡层：确保节点在边上面 */}
                                <rect
                                  x={bgSize.x}
                                  y={bgSize.y}
                                  width={bgSize.w}
                                  height={bgSize.h}
                                  fill={node.color || '#6366f1'}
                                  rx={node.type === 'board' || node.type === 'npu' || node.type === 'cpu' ? 2 : 6}
                                />
                                {/* 节点形状（带高亮效果） */}
                                <g style={{ filter: highlightFilter }}>
                                  {renderNodeShape(node)}
                                </g>
                                <text
                                  y={4}
                                  textAnchor="middle"
                                  fontSize={14}
                                  fill="#fff"
                                  fontWeight={600}
                                  style={{ textShadow: '0 1px 2px rgba(0,0,0,0.5)', pointerEvents: 'none' }}
                                >
                                  {node.label.length > 6 ? node.label.match(/\d+/)?.[0] || node.label.slice(-2) : node.label}
                                </text>
                              </g>
                            )
                          })}
                        </svg>
                      )
                    })()}

                    {/* 如果没有 singleLevelData，使用原来的渲染方式（兼容） */}
                    {!containerNode.singleLevelData && (
                      <>
                        {/* 该层的边 */}
                        {layerEdges.map((edge, i) => {
                          const sourceNode = displayNodes.find(n => n.id === edge.source)
                          const targetNode = displayNodes.find(n => n.id === edge.target)
                          if (!sourceNode || !targetNode) return null
                          return (
                            <line
                              key={`layer-edge-${i}`}
                              x1={sourceNode.x}
                              y1={sourceNode.y}
                              x2={targetNode.x}
                              y2={targetNode.y}
                              stroke="#52c41a"
                              strokeWidth={1.5}
                              strokeOpacity={0.6}
                            />
                          )
                        })}

                        {/* 该层的节点 */}
                        {layerNodes.map(node => {
                          const multiLevelScale = 0.5
                          return (
                            <g
                              key={`layer-node-${node.id}`}
                              transform={`translate(${node.x}, ${node.y}) scale(${multiLevelScale})`}
                              style={{ cursor: 'pointer', filter: 'drop-shadow(0 1px 2px rgba(0,0,0,0.2))' }}
                            >
                              {renderNodeShape(node)}
                              <text
                                y={4}
                                textAnchor="middle"
                                fontSize={14}
                                fill="#fff"
                                fontWeight={600}
                              >
                                {node.label.length > 6 ? node.label.match(/\d+/)?.[0] || node.label.slice(-2) : node.label}
                              </text>
                            </g>
                          )
                        })}
                      </>
                    )}
                  </g>
                </g>
              )
            })

            return (
              <>
                {containersRendered}
                {/* 跨容器手动连接在最上层 */}
                {renderManualConnectionsOverlay()}
              </>
            )
          })()}

          {/* 2D Torus：渲染环绕连接的半椭圆弧 */}
          {directTopology === 'torus_2d' && (() => {
            const deviceNodes = displayNodes.filter(n => !n.isSwitch)
            const { cols, rows } = getTorusGridSize(deviceNodes.length)
            if (cols < 2 && rows < 2) return null

            // 获取节点的实际显示位置（优先使用手动位置）
            const getPos = (node: { id: string; x: number; y: number }) => {
              if (isManualMode && manualPositions[node.id]) {
                return manualPositions[node.id]
              }
              return { x: node.x, y: node.y }
            }

            // 找出每行和每列的首尾节点位置
            const rowArcs: { leftX: number; leftY: number; rightX: number; rightY: number; row: number }[] = []
            const colArcs: { topX: number; topY: number; bottomX: number; bottomY: number; col: number }[] = []

            for (let r = 0; r < rows; r++) {
              const nodesInRow = deviceNodes.filter(n => n.gridRow === r).sort((a, b) => (a.gridCol || 0) - (b.gridCol || 0))
              // 只有3个以上节点时才画环绕弧（2个节点时首尾相邻，已有直线连接）
              if (nodesInRow.length >= 3) {
                const first = nodesInRow[0]
                const last = nodesInRow[nodesInRow.length - 1]
                const firstPos = getPos(first)
                const lastPos = getPos(last)
                rowArcs.push({
                  leftX: firstPos.x,
                  leftY: firstPos.y,
                  rightX: lastPos.x,
                  rightY: lastPos.y,
                  row: r
                })
              }
            }

            for (let c = 0; c < cols; c++) {
              const nodesInCol = deviceNodes.filter(n => n.gridCol === c).sort((a, b) => (a.gridRow || 0) - (b.gridRow || 0))
              // 只有3个以上节点时才画环绕弧（2个节点时首尾相邻，已有直线连接）
              if (nodesInCol.length >= 3) {
                const first = nodesInCol[0]
                const last = nodesInCol[nodesInCol.length - 1]
                const firstPos = getPos(first)
                const lastPos = getPos(last)
                colArcs.push({
                  topX: firstPos.x,
                  topY: firstPos.y,
                  bottomX: lastPos.x,
                  bottomY: lastPos.y,
                  col: c
                })
              }
            }

            // 通用弧线渲染函数：根据两点位置动态计算控制点
            const renderArc = (
              x1: number, y1: number, x2: number, y2: number,
              key: string, bulgeOffset: number
            ) => {
              const midX = (x1 + x2) / 2
              const midY = (y1 + y2) / 2
              const dx = x2 - x1
              const dy = y2 - y1
              const dist = Math.sqrt(dx * dx + dy * dy)
              // 弯曲量与距离成比例
              const bulge = dist * 0.25 + bulgeOffset * 8
              // 垂直于连线方向的单位向量（选择一个固定方向避免翻转）
              const perpX = -dy / dist
              const perpY = dx / dist
              // 控制点在连线中点的垂直方向上
              const ctrlX = midX + perpX * bulge
              const ctrlY = midY + perpY * bulge
              return (
                <path
                  key={key}
                  d={`M ${x1} ${y1} Q ${ctrlX} ${ctrlY}, ${x2} ${y2}`}
                  fill="none"
                  stroke="#999"
                  strokeWidth={1.5}
                  strokeOpacity={0.6}
                />
              )
            }

            return (
              <g>
                {/* 行环绕弧 */}
                {rowArcs.map((arc, i) => renderArc(
                  arc.leftX, arc.leftY, arc.rightX, arc.rightY,
                  `row-arc-${i}`, i
                ))}
                {/* 列环绕弧 */}
                {colArcs.map((arc, i) => renderArc(
                  arc.topX, arc.topY, arc.bottomX, arc.bottomY,
                  `col-arc-${i}`, i
                ))}
              </g>
            )
          })()}

          {/* 3D Torus：X/Y/Z三个方向的环绕弧线（只有>=3个节点才画环绕弧） */}
          {directTopology === 'torus_3d' && (() => {
            const deviceNodes = displayNodes.filter(n => !n.isSwitch)
            const { dim, layers } = getTorus3DSize(deviceNodes.length)
            if (dim < 2) return null

            // 获取节点的实际显示位置（优先使用手动位置）
            const getPos = (node: { id: string; x: number; y: number }) => {
              if (isManualMode && manualPositions[node.id]) {
                return manualPositions[node.id]
              }
              return { x: node.x, y: node.y }
            }

            // 通用弧线渲染函数
            const renderArc3D = (
              x1: number, y1: number, x2: number, y2: number,
              key: string, bulgeOffset: number
            ) => {
              const midX = (x1 + x2) / 2
              const midY = (y1 + y2) / 2
              const dx = x2 - x1
              const dy = y2 - y1
              const dist = Math.sqrt(dx * dx + dy * dy)
              if (dist < 1) return null
              const bulge = dist * 0.25 + bulgeOffset * 5
              const perpX = -dy / dist
              const perpY = dx / dist
              const ctrlX = midX + perpX * bulge
              const ctrlY = midY + perpY * bulge
              return (
                <path
                  key={key}
                  d={`M ${x1} ${y1} Q ${ctrlX} ${ctrlY}, ${x2} ${y2}`}
                  fill="none"
                  stroke="#999"
                  strokeWidth={1.5}
                  strokeOpacity={0.5}
                />
              )
            }

            const arcs: JSX.Element[] = []

            // X方向环绕弧
            for (let z = 0; z < layers; z++) {
              for (let r = 0; r < dim; r++) {
                const rowNodes = deviceNodes
                  .filter(n => n.gridZ === z && n.gridRow === r)
                  .sort((a, b) => a.gridCol! - b.gridCol!)
                if (rowNodes.length >= 3) {
                  const first = rowNodes[0]
                  const last = rowNodes[rowNodes.length - 1]
                  const firstPos = getPos(first)
                  const lastPos = getPos(last)
                  const arc = renderArc3D(firstPos.x, firstPos.y, lastPos.x, lastPos.y, `x-arc-z${z}-r${r}`, z + r)
                  if (arc) arcs.push(arc)
                }
              }
            }

            // Y方向环绕弧
            for (let z = 0; z < layers; z++) {
              for (let c = 0; c < dim; c++) {
                const colNodes = deviceNodes
                  .filter(n => n.gridZ === z && n.gridCol === c)
                  .sort((a, b) => a.gridRow! - b.gridRow!)
                if (colNodes.length >= 3) {
                  const first = colNodes[0]
                  const last = colNodes[colNodes.length - 1]
                  const firstPos = getPos(first)
                  const lastPos = getPos(last)
                  const arc = renderArc3D(firstPos.x, firstPos.y, lastPos.x, lastPos.y, `y-arc-z${z}-c${c}`, z + c)
                  if (arc) arcs.push(arc)
                }
              }
            }

            // Z方向环绕弧
            for (let r = 0; r < dim; r++) {
              for (let c = 0; c < dim; c++) {
                const depthNodes = deviceNodes
                  .filter(n => n.gridRow === r && n.gridCol === c)
                  .sort((a, b) => a.gridZ! - b.gridZ!)
                if (depthNodes.length >= 3) {
                  const first = depthNodes[0]
                  const last = depthNodes[depthNodes.length - 1]
                  const firstPos = getPos(first)
                  const lastPos = getPos(last)
                  const arc = renderArc3D(firstPos.x, firstPos.y, lastPos.x, lastPos.y, `z-arc-r${r}-c${c}`, r + c)
                  if (arc) arcs.push(arc)
                }
              }
            }

            return <g>{arcs}</g>
          })()}

          {/* 渲染连接线（非多层级模式） */}
          {!multiLevelOptions?.enabled && edges.map((edge, i) => {
            const sourcePos = nodePositions.get(edge.source)
            const targetPos = nodePositions.get(edge.target)
            if (!sourcePos || !targetPos) return null

            const sourceNode = nodes.find(n => n.id === edge.source)
            const targetNode = nodes.find(n => n.id === edge.target)

            const adjustedSourcePos = { x: sourcePos.x, y: sourcePos.y }
            const adjustedTargetPos = { x: targetPos.x, y: targetPos.y }

            // 生成唯一的连接ID
            const edgeId = `${edge.source}-${edge.target}`
            const isLinkSelected = selectedLinkId === edgeId || selectedLinkId === `${edge.target}-${edge.source}`

            const bandwidthStr = edge.bandwidth ? `${edge.bandwidth}Gbps` : ''
            const latencyStr = edge.latency ? `${edge.latency}ns` : ''
            const trafficStyle = getTrafficHeatmapStyle(edge.source, edge.target)
            const trafficStr = trafficStyle ? `流量: ${trafficStyle.trafficMb.toFixed(1)}MB, 利用率: ${(trafficStyle.utilization * 100).toFixed(0)}%` : ''
            const propsStr = [bandwidthStr, latencyStr, trafficStr].filter(Boolean).join(', ')
            const tooltipContent = `${sourceNode?.label || edge.source} ↔ ${targetNode?.label || edge.target}${propsStr ? ` (${propsStr})` : ''}`

            // 点击 link 的处理函数
            const handleLinkClick = (e: React.MouseEvent) => {
              e.stopPropagation()
              if (connectionMode !== 'view' || isManualMode) return
              if (onLinkClick) {
                onLinkClick({
                  id: edgeId,
                  sourceId: edge.source,
                  sourceLabel: sourceNode?.label || edge.source,
                  sourceType: sourceNode?.type || 'unknown',
                  targetId: edge.target,
                  targetLabel: targetNode?.label || edge.target,
                  targetType: targetNode?.type || 'unknown',
                  bandwidth: edge.bandwidth,
                  latency: edge.latency,
                  isManual: false
                })
              }
            }

            // 判断是否是 Torus 环绕连接
            const sourceGridRow = sourceNode?.gridRow
            const sourceGridCol = sourceNode?.gridCol
            const sourceGridZ = sourceNode?.gridZ
            const targetGridRow = targetNode?.gridRow
            const targetGridCol = targetNode?.gridCol
            const targetGridZ = targetNode?.gridZ

            // 检测环绕连接
            const deviceNodes = nodes.filter(n => !n.isSwitch)

            if (directTopology === 'torus_2d') {
              const { cols, rows } = getTorusGridSize(deviceNodes.length)
              const isHorizontalWrap = sourceGridRow === targetGridRow &&
                Math.abs((sourceGridCol || 0) - (targetGridCol || 0)) === cols - 1
              const isVerticalWrap = sourceGridCol === targetGridCol &&
                Math.abs((sourceGridRow || 0) - (targetGridRow || 0)) === rows - 1
              if (isHorizontalWrap || isVerticalWrap) {
                return null  // 2D Torus 环绕连接由弧线表示
              }
            }

            if (directTopology === 'torus_3d') {
              void getTorus3DSize(deviceNodes.length)  // 保留方法调用但不使用返回值
              const sameZ = sourceGridZ === targetGridZ
              const sameRow = sourceGridRow === targetGridRow
              const sameCol = sourceGridCol === targetGridCol

              // 只有该方向节点数>=3时，环绕连接才由弧线表示，否则直接画线
              // X方向环绕（同层同行，列差为dim-1，且该行节点数>=3）
              const rowNodes = deviceNodes.filter(n => n.gridZ === sourceGridZ && n.gridRow === sourceGridRow)
              const isXWrap = sameZ && sameRow && rowNodes.length >= 3 &&
                Math.abs((sourceGridCol || 0) - (targetGridCol || 0)) === rowNodes.length - 1

              // Y方向环绕（同层同列，行差为dim-1，且该列节点数>=3）
              const colNodes = deviceNodes.filter(n => n.gridZ === sourceGridZ && n.gridCol === sourceGridCol)
              const isYWrap = sameZ && sameCol && colNodes.length >= 3 &&
                Math.abs((sourceGridRow || 0) - (targetGridRow || 0)) === colNodes.length - 1

              // Z方向环绕（同行同列，Z差为layers-1，且该深度>=3）
              const depthNodes = deviceNodes.filter(n => n.gridRow === sourceGridRow && n.gridCol === sourceGridCol)
              const isZWrap = sameRow && sameCol && depthNodes.length >= 3 &&
                Math.abs((sourceGridZ || 0) - (targetGridZ || 0)) === depthNodes.length - 1

              if (isXWrap || isYWrap || isZWrap) {
                return null  // 3D Torus 环绕连接由弧线表示
              }
            }

            // 2D FullMesh：非相邻节点使用曲线连接
            if (directTopology === 'full_mesh_2d') {
              const sameRow = sourceGridRow === targetGridRow
              const sameCol = sourceGridCol === targetGridCol
              const colDiff = Math.abs((sourceGridCol || 0) - (targetGridCol || 0))
              const rowDiff = Math.abs((sourceGridRow || 0) - (targetGridRow || 0))

              // 同行非相邻（列差>1）或同列非相邻（行差>1）使用曲线
              if ((sameRow && colDiff > 1) || (sameCol && rowDiff > 1)) {
                const midX = (sourcePos.x + targetPos.x) / 2
                const midY = (sourcePos.y + targetPos.y) / 2
                const dx = targetPos.x - sourcePos.x
                const dy = targetPos.y - sourcePos.y
                const dist = Math.sqrt(dx * dx + dy * dy)

                // 使用垂直于连线方向的控制点，确保曲线正确连接两端
                const bulge = dist * 0.25 + (sourceGridRow || 0) * 5
                const perpX = -dy / dist
                const perpY = dx / dist
                const ctrlX = midX + perpX * bulge
                const ctrlY = midY + perpY * bulge

                const pathD = `M ${sourcePos.x} ${sourcePos.y} Q ${ctrlX} ${ctrlY}, ${targetPos.x} ${targetPos.y}`
                return (
                  <g key={`edge-${i}`}>
                    {/* 透明触发层 */}
                    <path
                      d={pathD}
                      fill="none"
                      stroke="transparent"
                      strokeWidth={16}
                      style={{ cursor: 'pointer' }}
                      onClick={handleLinkClick}
                      onMouseEnter={(e) => {
                        if (connectionMode !== 'view' || isManualMode) return
                        const rect = svgRef.current?.getBoundingClientRect()
                        if (rect) {
                          setTooltip({
                            x: e.clientX - rect.left,
                            y: e.clientY - rect.top + 20,
                            content: tooltipContent,
                          })
                        }
                      }}
                      onMouseLeave={() => (connectionMode === 'view' && !isManualMode) && setTooltip(null)}
                    />
                    {/* 可见曲线 - Switch连接用蓝色，节点直连用灰色，选中时绿色高亮 */}
                    <path
                      d={pathD}
                      fill="none"
                      stroke={isLinkSelected ? '#52c41a' : (edge.isSwitch ? '#1890ff' : '#b0b0b0')}
                      strokeWidth={isLinkSelected ? 3 : (edge.isSwitch ? 2 : 1.5)}
                      strokeOpacity={isLinkSelected ? 1 : 0.7}
                      style={{ pointerEvents: 'none', filter: isLinkSelected ? 'drop-shadow(0 0 4px #52c41a)' : 'none' }}
                    />
                  </g>
                )
              }
            }

            // 多层级连接样式计算
            const getMultiLevelEdgeStyle = () => {
              // 热力图模式优先：如果有流量分析结果，使用热力图颜色
              const trafficStyle = getTrafficHeatmapStyle(edge.source, edge.target)
              if (trafficStyle && !isLinkSelected) {
                return {
                  stroke: trafficStyle.stroke,
                  strokeWidth: trafficStyle.strokeWidth,
                  strokeDasharray: undefined as string | undefined,
                }
              }
              if (!edge.connectionType) {
                // 非多层级模式，使用原有逻辑
                return {
                  stroke: isLinkSelected ? '#52c41a' : (edge.isSwitch ? '#1890ff' : '#b0b0b0'),
                  strokeWidth: isLinkSelected ? 3 : (edge.isSwitch ? 2 : 1.5),
                  strokeDasharray: undefined as string | undefined,
                }
              }
              // 多层级模式
              switch (edge.connectionType) {
                case 'intra_upper':
                  return { stroke: isLinkSelected ? '#52c41a' : '#1890ff', strokeWidth: isLinkSelected ? 3 : 2, strokeDasharray: undefined }
                case 'intra_lower':
                  return { stroke: isLinkSelected ? '#52c41a' : '#52c41a', strokeWidth: isLinkSelected ? 3 : 1.5, strokeDasharray: undefined }
                case 'inter_level':
                  return { stroke: isLinkSelected ? '#52c41a' : '#faad14', strokeWidth: isLinkSelected ? 3 : 1.5, strokeDasharray: '6,3' }
                default:
                  return { stroke: '#b0b0b0', strokeWidth: 1.5, strokeDasharray: undefined }
              }
            }
            const edgeStyle = getMultiLevelEdgeStyle()

            // 普通直线连接 - 使用中心点，节点会遮盖线的端点
            return (
              <g key={`edge-${i}`} style={{ transition: 'transform 0.3s ease-out' }}>
                {/* 透明触发层 - 增大点击区域 */}
                <line
                  x1={adjustedSourcePos.x}
                  y1={adjustedSourcePos.y}
                  x2={adjustedTargetPos.x}
                  y2={adjustedTargetPos.y}
                  stroke="transparent"
                  strokeWidth={16}
                  style={{ cursor: 'pointer' }}
                  onClick={handleLinkClick}
                  onMouseEnter={(e) => {
                    if (connectionMode !== 'view' || isManualMode) return
                    const rect = svgRef.current?.getBoundingClientRect()
                    if (rect) {
                      setTooltip({
                        x: e.clientX - rect.left,
                        y: e.clientY - rect.top + 20,
                        content: tooltipContent,
                      })
                    }
                  }}
                  onMouseLeave={() => (connectionMode === 'view' && !isManualMode) && setTooltip(null)}
                />
                {/* 可见线条 */}
                <line
                  x1={adjustedSourcePos.x}
                  y1={adjustedSourcePos.y}
                  x2={adjustedTargetPos.x}
                  y2={adjustedTargetPos.y}
                  stroke={edgeStyle.stroke}
                  strokeWidth={edgeStyle.strokeWidth}
                  strokeDasharray={edgeStyle.strokeDasharray}
                  strokeOpacity={isLinkSelected ? 1 : 0.7}
                  style={{ pointerEvents: 'none', filter: isLinkSelected ? 'drop-shadow(0 0 4px #52c41a)' : 'none' }}
                />
              </g>
            )
          })}

          {/* 渲染手动连接线（非多层级模式） */}
          {!multiLevelOptions?.enabled && manualConnections
            .filter(mc => mc.hierarchy_level === getCurrentHierarchyLevel())
            .map((conn) => {
              const sourcePos = nodePositions.get(conn.source)
              const targetPos = nodePositions.get(conn.target)
              if (!sourcePos || !targetPos) return null

              const sourceNode = nodes.find(n => n.id === conn.source)
              const targetNode = nodes.find(n => n.id === conn.target)

              // 多层级视图：计算 yOffset（容器悬停/选中时的位移）
              let sourceYOffset = 0
              let targetYOffset = 0
              if (multiLevelOptions?.enabled) {
                const liftDistance = 150
                // 从节点ID提取父容器索引作为 zLayer
                const getZLayer = (nodeId: string): number => {
                  const parts = nodeId.split('/')
                  if (parts.length >= 2) {
                    const parentPart = parts[parts.length - 2]
                    const match = parentPart.match(/_(\d+)$/)
                    return match ? parseInt(match[1], 10) : 0
                  }
                  return 0
                }
                // 计算 selectedLayerIndex
                let selectedLayerIdx: number | null = null
                if (selectedNodeId) {
                  const selectedContainer = displayNodes.find(n => n.isContainer && n.id === selectedNodeId)
                  if (selectedContainer) {
                    selectedLayerIdx = selectedContainer.zLayer ?? null
                  } else {
                    // 查找包含选中节点的容器
                    for (const n of displayNodes) {
                      if (n.isContainer && n.singleLevelData?.nodes.some(sn => sn.id === selectedNodeId)) {
                        selectedLayerIdx = n.zLayer ?? null
                        break
                      }
                    }
                  }
                }
                const activeLayerIdx = hoveredLayerIndex ?? selectedLayerIdx
                const sourceZLayer = getZLayer(conn.source)
                const targetZLayer = getZLayer(conn.target)
                if (activeLayerIdx !== null) {
                  if (sourceZLayer < activeLayerIdx) sourceYOffset = -liftDistance
                  if (targetZLayer < activeLayerIdx) targetYOffset = -liftDistance
                }
              }
              // 应用 yOffset
              const adjustedSourcePos = { x: sourcePos.x, y: sourcePos.y + sourceYOffset }
              const adjustedTargetPos = { x: targetPos.x, y: targetPos.y + targetYOffset }
              const manualTooltip = `${sourceNode?.label || conn.source} ↔ ${targetNode?.label || conn.target} (手动)`

              // 手动连接的ID和选中状态
              const manualEdgeId = `${conn.source}-${conn.target}`
              const isManualLinkSelected = selectedLinkId === manualEdgeId || selectedLinkId === `${conn.target}-${conn.source}`

              // 手动连接的点击处理
              const handleManualLinkClick = (e: React.MouseEvent) => {
                e.stopPropagation()
                if (connectionMode !== 'view' || isManualMode) return
                if (onLinkClick) {
                  // 构建详细的节点标签（从节点ID路径中提取）
                  // 例如 "pod_0/rack_0/board_4" -> "rack_0/board_4"
                  const getDetailedLabel = (nodeId: string, fallbackLabel: string) => {
                    const parts = nodeId.split('/')
                    if (parts.length >= 2) {
                      // 取最后两部分作为详细标签
                      return parts.slice(-2).join('/')
                    }
                    return fallbackLabel
                  }
                  onLinkClick({
                    id: manualEdgeId,
                    sourceId: conn.source,
                    sourceLabel: getDetailedLabel(conn.source, sourceNode?.label || conn.source),
                    sourceType: sourceNode?.type || conn.source.split('/').pop()?.split('_')[0] || 'unknown',
                    targetId: conn.target,
                    targetLabel: getDetailedLabel(conn.target, targetNode?.label || conn.target),
                    targetType: targetNode?.type || conn.target.split('/').pop()?.split('_')[0] || 'unknown',
                    isManual: true
                  })
                }
              }

              // 判断是否跨容器连接（从节点ID路径中提取父容器索引）
              const getParentIndex = (nodeId: string): number => {
                const parts = nodeId.split('/')
                if (parts.length >= 2) {
                  const parentPart = parts[parts.length - 2]
                  const match = parentPart.match(/_(\d+)$/)
                  return match ? parseInt(match[1], 10) : 0
                }
                return 0
              }
              const sourceParentIdx = getParentIndex(conn.source)
              const targetParentIdx = getParentIndex(conn.target)
              const indexDiff = Math.abs(sourceParentIdx - targetParentIdx)
              const isCrossContainer = indexDiff > 1

              // 跨容器连接使用曲线
              if (isCrossContainer) {
                const midX = (adjustedSourcePos.x + adjustedTargetPos.x) / 2
                const midY = (adjustedSourcePos.y + adjustedTargetPos.y) / 2
                const dx = adjustedTargetPos.x - adjustedSourcePos.x
                const dy = adjustedTargetPos.y - adjustedSourcePos.y
                const dist = Math.sqrt(dx * dx + dy * dy)
                const bulge = dist * 0.2 * indexDiff
                const perpX = -dy / dist
                const perpY = dx / dist
                const ctrlX = midX + perpX * bulge
                const ctrlY = midY + perpY * bulge
                const pathD = `M ${adjustedSourcePos.x} ${adjustedSourcePos.y} Q ${ctrlX} ${ctrlY}, ${adjustedTargetPos.x} ${adjustedTargetPos.y}`
                const strokeColor = isManualLinkSelected ? '#52c41a' : '#722ed1'  // 跨容器用紫色
                const strokeWidth = isManualLinkSelected ? 3 : 2

                return (
                  <g key={`manual-${conn.id}`} style={{ transition: 'all 0.3s ease-out' }}>
                    <path d={pathD} fill="none" stroke="transparent" strokeWidth={16}
                      style={{ cursor: 'pointer' }} onClick={handleManualLinkClick}
                      onMouseEnter={(e) => {
                        if (connectionMode !== 'view' || isManualMode) return
                        const rect = svgRef.current?.getBoundingClientRect()
                        if (rect) {
                          setTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top + 20, content: manualTooltip })
                        }
                      }}
                      onMouseLeave={() => (connectionMode === 'view' && !isManualMode) && setTooltip(null)}
                    />
                    <path d={pathD} fill="none" stroke={strokeColor} strokeWidth={strokeWidth}
                      strokeOpacity={isManualLinkSelected ? 1 : 0.8}
                      style={{ pointerEvents: 'none', filter: isManualLinkSelected ? 'drop-shadow(0 0 4px #52c41a)' : 'none' }}
                    />
                  </g>
                )
              }

              return (
                <g key={`manual-${conn.id}`} style={{ transition: 'all 0.3s ease-out' }}>
                  {/* 透明触发层 */}
                  <line
                    x1={adjustedSourcePos.x}
                    y1={adjustedSourcePos.y}
                    x2={adjustedTargetPos.x}
                    y2={adjustedTargetPos.y}
                    stroke="transparent"
                    strokeWidth={16}
                    style={{ cursor: 'pointer' }}
                    onClick={handleManualLinkClick}
                    onMouseEnter={(e) => {
                      if (connectionMode !== 'view' || isManualMode) return
                      const rect = svgRef.current?.getBoundingClientRect()
                      if (rect) {
                        setTooltip({
                          x: e.clientX - rect.left,
                          y: e.clientY - rect.top + 20,
                          content: manualTooltip,
                        })
                      }
                    }}
                    onMouseLeave={() => (connectionMode === 'view' && !isManualMode) && setTooltip(null)}
                  />
                  {/* 可见线条 - 编辑模式绿色虚线，普通模式与自动连接一致，选中时绿色高亮 */}
                  <line
                    x1={adjustedSourcePos.x}
                    y1={adjustedSourcePos.y}
                    x2={adjustedTargetPos.x}
                    y2={adjustedTargetPos.y}
                    stroke={isManualLinkSelected ? '#52c41a' : (connectionMode !== 'view' ? '#52c41a' : '#b0b0b0')}
                    strokeWidth={isManualLinkSelected ? 3 : (connectionMode !== 'view' ? 2.5 : 1.5)}
                    strokeOpacity={isManualLinkSelected ? 1 : (connectionMode !== 'view' ? 1 : 0.6)}
                    strokeDasharray={connectionMode !== 'view' ? '8,4' : undefined}
                    style={{ pointerEvents: 'none', filter: isManualLinkSelected ? 'drop-shadow(0 0 4px #52c41a)' : 'none' }}
                  />
                </g>
              )
            })}

          {/* 渲染节点（非多层级模式） */}
          {!multiLevelOptions?.enabled && displayNodes.map((node) => {
            // 容器节点不渲染
            if (node.isContainer) {
              return null
            }
            const isSwitch = node.isSwitch
            // 计算连接数
            const nodeConnections = edges.filter(e => e.source === node.id || e.target === node.id)
            // 简化的tooltip内容
            const nodeTooltip = isSwitch
              ? `${node.label} · ${node.subType?.toUpperCase() || 'SWITCH'} · ${nodeConnections.length}连接`
              : `${node.label} · ${node.type.toUpperCase()} · ${nodeConnections.length}连接`
            const isSourceSelected = selectedNodes.has(node.id)
            const isTargetSelected = targetNodes.has(node.id)
            const isDragging = draggingNode === node.id
            // 判断节点是否被点击选中
            const isNodeSelected = selectedNodeId === node.id
            // 判断节点是否是选中 link 的两端
            const isLinkEndpoint = selectedLinkId && (
              selectedLinkId.startsWith(node.id + '-') ||
              selectedLinkId.endsWith('-' + node.id)
            )
            const isHovered = hoveredNodeId === node.id && connectionMode === 'view' && !isManualMode && !isDragging
            // 节点高亮：点击选中、hover 或者是选中 link 的端点
            const shouldHighlight = isNodeSelected || isHovered || isLinkEndpoint

            return (
              <g
                key={node.id}
                transform={`translate(${node.x}, ${node.y}) scale(${nodeScale})`}
                style={{
                  cursor: isManualMode ? 'move' : connectionMode !== 'view' ? 'crosshair' : 'pointer',
                  opacity: isDragging ? 0.7 : 1,
                  filter: shouldHighlight
                    ? 'drop-shadow(0 0 8px rgba(37, 99, 235, 0.6)) drop-shadow(0 0 16px rgba(37, 99, 235, 0.3))'
                    : 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))',
                  transition: 'filter 0.15s ease, opacity 0.15s ease',
                }}
                onMouseDown={(e) => handleDragStart(node.id, e)}
                onClick={(e) => {
                  // 选择源节点模式
                  if (connectionMode === 'select_source' || connectionMode === 'select' || connectionMode === 'connect') {
                    const currentSet = new Set(selectedNodes)
                    if (e.ctrlKey || e.metaKey) {
                      // Ctrl+点击：切换选中状态
                      if (currentSet.has(node.id)) {
                        currentSet.delete(node.id)
                      } else {
                        currentSet.add(node.id)
                      }
                    } else {
                      // 普通点击：切换单个选中
                      if (currentSet.has(node.id)) {
                        currentSet.delete(node.id)
                      } else {
                        currentSet.add(node.id)
                      }
                    }
                    onSelectedNodesChange?.(currentSet)
                  } else if (connectionMode === 'select_target') {
                    // 选择目标节点模式
                    const currentSet = new Set(targetNodes)
                    if (e.ctrlKey || e.metaKey) {
                      // Ctrl+点击：切换选中状态
                      if (currentSet.has(node.id)) {
                        currentSet.delete(node.id)
                      } else {
                        currentSet.add(node.id)
                      }
                    } else {
                      // 普通点击：切换单个选中
                      if (currentSet.has(node.id)) {
                        currentSet.delete(node.id)
                      } else {
                        currentSet.add(node.id)
                      }
                    }
                    onTargetNodesChange?.(currentSet)
                  } else {
                    // 普通查看模式
                    if (onNodeClick) {
                      const connections = nodeConnections.map(e => {
                        const otherId = e.source === node.id ? e.target : e.source
                        const otherNode = displayNodes.find(n => n.id === otherId)
                        return {
                          id: otherId,
                          label: otherNode?.label || otherId,
                          bandwidth: e.bandwidth,
                          latency: e.latency
                        }
                      })
                      onNodeClick({
                        id: node.id,
                        label: node.label,
                        type: node.type,
                        subType: node.subType,
                        connections,
                        portInfo: node.portInfo
                      })
                    }
                  }
                }}
                onDoubleClick={() => {
                  if (connectionMode === 'view' && onNodeDoubleClick && currentLevel !== 'board' && !isSwitch) {
                    onNodeDoubleClick(node.id, node.type)
                  }
                }}
                onMouseEnter={(e) => {
                  setHoveredNodeId(node.id)
                  if (connectionMode !== 'view' || isManualMode) return  // 连线模式或手动布局不显示悬停提示
                  const rect = svgRef.current?.getBoundingClientRect()
                  if (rect) {
                    setTooltip({
                      x: e.clientX - rect.left,
                      y: e.clientY - rect.top + 25,  // 显示在节点下方
                      content: nodeTooltip,
                    })
                  }
                }}
                onMouseLeave={() => {
                  setHoveredNodeId(null)
                  if (connectionMode === 'view' && !isManualMode) setTooltip(null)
                }}
              >
                {/* 源节点选中状态边框（绿色） */}
                {isSourceSelected && (
                  <rect
                    x={-38}
                    y={-30}
                    width={76}
                    height={60}
                    fill="none"
                    stroke="#52c41a"
                    strokeWidth={3}
                    strokeDasharray="6,3"
                    rx={8}
                  />
                )}
                {/* 目标节点选中状态边框（蓝色） */}
                {isTargetSelected && (
                  <rect
                    x={-38}
                    y={-30}
                    width={76}
                    height={60}
                    fill="none"
                    stroke="#1890ff"
                    strokeWidth={3}
                    strokeDasharray="6,3"
                    rx={8}
                  />
                )}
                {/* 根据节点类型渲染不同形状 */}
                {isSwitch ? (
                  /* Switch: 网络交换机形状 - 扁平矩形带端口和指示灯 */
                  <g>
                    {/* 主体外壳 */}
                    <rect x={-36} y={-14} width={72} height={28} rx={3} fill={node.color} stroke="#fff" strokeWidth={2} />
                    {/* 前面板凹槽 */}
                    <rect x={-32} y={-10} width={64} height={16} rx={2} fill="rgba(0,0,0,0.15)" />
                    {/* 端口组 - 左侧 */}
                    <rect x={-28} y={-6} width={6} height={8} rx={1} fill="rgba(255,255,255,0.5)" />
                    <rect x={-20} y={-6} width={6} height={8} rx={1} fill="rgba(255,255,255,0.5)" />
                    <rect x={-12} y={-6} width={6} height={8} rx={1} fill="rgba(255,255,255,0.5)" />
                    <rect x={-4} y={-6} width={6} height={8} rx={1} fill="rgba(255,255,255,0.5)" />
                    {/* 端口组 - 右侧 */}
                    <rect x={6} y={-6} width={6} height={8} rx={1} fill="rgba(255,255,255,0.5)" />
                    <rect x={14} y={-6} width={6} height={8} rx={1} fill="rgba(255,255,255,0.5)" />
                    <rect x={22} y={-6} width={6} height={8} rx={1} fill="rgba(255,255,255,0.5)" />
                    {/* 状态指示灯 */}
                    <circle cx={-28} cy={8} r={2} fill="#4ade80" />
                    <circle cx={-22} cy={8} r={2} fill="#4ade80" />
                    <circle cx={-16} cy={8} r={2} fill="#4ade80" />
                    <circle cx={-10} cy={8} r={2} fill="#fbbf24" />
                    {/* 品牌标识区 */}
                    <rect x={10} y={4} width={18} height={6} rx={1} fill="rgba(255,255,255,0.2)" />
                  </g>
                ) : node.type === 'pod' ? (
                  /* Pod: 数据中心/机房形状 - 带屋顶的建筑 */
                  <g>
                    {/* 主体建筑 */}
                    <rect x={-28} y={-12} width={56} height={32} rx={3} fill={node.color} stroke="#fff" strokeWidth={2} />
                    {/* 屋顶 */}
                    <polygon points="-32,-12 0,-24 32,-12" fill={node.color} stroke="#fff" strokeWidth={2} />
                    {/* 窗户装饰 */}
                    <rect x={-20} y={-4} width={8} height={8} rx={1} fill="rgba(255,255,255,0.3)" />
                    <rect x={-6} y={-4} width={8} height={8} rx={1} fill="rgba(255,255,255,0.3)" />
                    <rect x={8} y={-4} width={8} height={8} rx={1} fill="rgba(255,255,255,0.3)" />
                    {/* 门 */}
                    <rect x={-5} y={8} width={10} height={12} rx={1} fill="rgba(255,255,255,0.4)" />
                  </g>
                ) : node.type === 'rack' ? (
                  /* Rack: 机柜形状 - 竖长矩形带分隔线 */
                  <g>
                    <rect x={-18} y={-28} width={36} height={56} rx={3} fill={node.color} stroke="#fff" strokeWidth={2} />
                    {/* 机柜层分隔 */}
                    <line x1={-14} y1={-16} x2={14} y2={-16} stroke="rgba(255,255,255,0.3)" strokeWidth={1} />
                    <line x1={-14} y1={-4} x2={14} y2={-4} stroke="rgba(255,255,255,0.3)" strokeWidth={1} />
                    <line x1={-14} y1={8} x2={14} y2={8} stroke="rgba(255,255,255,0.3)" strokeWidth={1} />
                    <line x1={-14} y1={20} x2={14} y2={20} stroke="rgba(255,255,255,0.3)" strokeWidth={1} />
                    {/* 指示灯 */}
                    <circle cx={10} cy={-22} r={2} fill="#4ade80" />
                    <circle cx={10} cy={-10} r={2} fill="#4ade80" />
                    <circle cx={10} cy={2} r={2} fill="#4ade80" />
                    <circle cx={10} cy={14} r={2} fill="#fbbf24" />
                  </g>
                ) : node.type === 'board' ? (
                  /* Board: 电路板形状 - 横向矩形带电路纹理 */
                  <g>
                    <rect x={-32} y={-18} width={64} height={36} rx={2} fill={node.color} stroke="#fff" strokeWidth={2} />
                    {/* 电路纹理 */}
                    <path d="M-24,-10 L-24,-2 L-16,-2 L-16,6 L-8,6" stroke="rgba(255,255,255,0.25)" strokeWidth={1.5} fill="none" />
                    <path d="M8,-10 L8,0 L16,0 L16,8 L24,8" stroke="rgba(255,255,255,0.25)" strokeWidth={1.5} fill="none" />
                    {/* 芯片槽位 */}
                    <rect x={-8} y={-8} width={16} height={16} rx={1} fill="rgba(0,0,0,0.2)" stroke="rgba(255,255,255,0.3)" strokeWidth={1} />
                    {/* 连接点 */}
                    <circle cx={-26} cy={0} r={3} fill="rgba(255,255,255,0.4)" />
                    <circle cx={26} cy={0} r={3} fill="rgba(255,255,255,0.4)" />
                  </g>
                ) : (node.type === 'npu' || node.type === 'cpu') ? (
                  /* Chip: 芯片形状 - 方形带引脚 */
                  <g>
                    {/* 芯片主体 */}
                    <rect x={-20} y={-20} width={40} height={40} rx={2} fill={node.color} stroke="#fff" strokeWidth={2} />
                    {/* 引脚 - 上 */}
                    <rect x={-12} y={-26} width={4} height={6} fill={node.color} />
                    <rect x={-2} y={-26} width={4} height={6} fill={node.color} />
                    <rect x={8} y={-26} width={4} height={6} fill={node.color} />
                    {/* 引脚 - 下 */}
                    <rect x={-12} y={20} width={4} height={6} fill={node.color} />
                    <rect x={-2} y={20} width={4} height={6} fill={node.color} />
                    <rect x={8} y={20} width={4} height={6} fill={node.color} />
                    {/* 引脚 - 左 */}
                    <rect x={-26} y={-12} width={6} height={4} fill={node.color} />
                    <rect x={-26} y={-2} width={6} height={4} fill={node.color} />
                    <rect x={-26} y={8} width={6} height={4} fill={node.color} />
                    {/* 引脚 - 右 */}
                    <rect x={20} y={-12} width={6} height={4} fill={node.color} />
                    <rect x={20} y={-2} width={6} height={4} fill={node.color} />
                    <rect x={20} y={8} width={6} height={4} fill={node.color} />
                    {/* 芯片内核标识 */}
                    <rect x={-10} y={-10} width={20} height={20} rx={1} fill="rgba(255,255,255,0.15)" />
                  </g>
                ) : (
                  /* 默认: 圆角矩形 */
                  <rect
                    x={-25}
                    y={-18}
                    width={50}
                    height={36}
                    rx={6}
                    fill={node.color}
                    stroke="#fff"
                    strokeWidth={2}
                  />
                )}
                {/* 节点标签 */}
                <text
                  y={node.type === 'rack' ? 0 : (node.type === 'pod' ? 6 : 4)}
                  fontSize={isSwitch ? 8 : (node.type === 'pod' || node.type === 'rack' ? 9 : 10)}
                  fill="#fff"
                  textAnchor="middle"
                  fontWeight="bold"
                  style={{ pointerEvents: 'none' }}
                >
                  {node.label.length > 8 ? node.label.substring(0, 8) + '..' : node.label}
                </text>
              </g>
            )
          })}
        </svg>

        {/* 图例 */}
        <div style={{
          position: 'absolute',
          bottom: 16,
          left: 16,
          background: '#fff',
          padding: '8px 14px',
          borderRadius: 8,
          border: '1px solid rgba(0, 0, 0, 0.08)',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.04)',
          fontSize: 12,
          fontFamily: "'JetBrains Mono', monospace",
        }}>
          <Text type="secondary">
            节点: {nodes.length} | 连接: {edges.length}
          </Text>
        </div>

        {/* 悬停提示 */}
        {tooltip && (
          <div style={{
            position: 'absolute',
            left: tooltip.x,
            top: tooltip.y,
            transform: 'translateX(-50%)',
            background: '#171717',
            color: '#fff',
            padding: '6px 10px',
            borderRadius: 6,
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
            fontSize: 11,
            fontFamily: "'JetBrains Mono', monospace",
            whiteSpace: 'nowrap',
            pointerEvents: 'none',
            zIndex: 1000,
          }}>
            {tooltip.content}
          </div>
        )}
      </div>
  )

  // 嵌入模式：直接渲染内容
  if (embedded) {
    return (
      <div style={{ width: '100%', height: '100%' }}>
        {graphContent}
      </div>
    )
  }

  // 弹窗模式
  return (
    <Modal
      title={toolbar}
      open={visible}
      onCancel={onClose}
      footer={null}
      width={900}
      styles={{ body: { padding: 0 } }}
    >
      {graphContent}
    </Modal>
  )
}
