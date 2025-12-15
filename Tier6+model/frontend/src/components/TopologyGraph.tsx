import React, { useMemo, useRef, useState, useEffect, useCallback } from 'react'
import { Modal, Button, Space, Typography, Breadcrumb, Segmented, Tooltip, Checkbox } from 'antd'
import { ZoomInOutlined, ZoomOutOutlined, ReloadOutlined, UndoOutlined, RedoOutlined } from '@ant-design/icons'
import {
  HierarchicalTopology,
  PodConfig,
  RackConfig,
  BoardConfig,
  CHIP_TYPE_COLORS,
  SWITCH_LAYER_COLORS,
  ManualConnection,
  ConnectionMode,
  HierarchyLevel,
  LayoutType,
} from '../types'

const { Text } = Typography

// 根据板卡U高度区分颜色
const BOARD_U_COLORS: Record<number, string> = {
  1: '#13c2c2',  // 1U - 青色
  2: '#722ed1',  // 2U - 紫色
  4: '#eb2f96',  // 4U - 洋红色
}

interface BreadcrumbItem {
  level: string
  id: string
  label: string
}

// 节点详细信息
export interface NodeDetail {
  id: string
  label: string
  type: string
  subType?: string
  connections: { id: string; label: string; bandwidth?: number; latency?: number }[]
  portInfo?: { uplink: number; downlink: number; inter: number }
}

interface TopologyGraphProps {
  visible: boolean
  onClose: () => void
  topology: HierarchicalTopology | null
  currentLevel: 'datacenter' | 'pod' | 'rack' | 'board' | 'chip'
  currentPod?: PodConfig | null
  currentRack?: RackConfig | null
  currentBoard?: BoardConfig | null
  onNodeDoubleClick?: (nodeId: string, nodeType: string) => void
  onNodeClick?: (nodeDetail: NodeDetail | null) => void
  onNavigateBack?: () => void
  onBreadcrumbClick?: (index: number) => void
  breadcrumbs?: BreadcrumbItem[]
  canGoBack?: boolean
  embedded?: boolean  // 嵌入模式（非弹窗）
  // 编辑连接相关
  connectionMode?: ConnectionMode
  selectedNodes?: Set<string>  // 源节点集合
  onSelectedNodesChange?: (nodes: Set<string>) => void
  targetNodes?: Set<string>  // 目标节点集合
  onTargetNodesChange?: (nodes: Set<string>) => void
  sourceNode?: string | null  // 保留兼容
  onSourceNodeChange?: (nodeId: string | null) => void
  onManualConnect?: (sourceId: string, targetId: string, level: HierarchyLevel) => void
  manualConnections?: ManualConnection[]
  onDeleteManualConnection?: (connectionId: string) => void
  onDeleteConnection?: (source: string, target: string) => void  // 删除任意连接（包括自动生成的）
  layoutType?: LayoutType  // 布局类型
  onLayoutTypeChange?: (type: LayoutType) => void  // 布局类型变更回调
}

interface Node {
  id: string
  label: string
  type: string
  subType?: string  // Switch的层级，如 "leaf", "spine"
  isSwitch?: boolean
  x: number
  y: number
  color: string
  portInfo?: {
    uplink: number
    downlink: number
    inter: number
  }
  // Torus布局的网格位置
  gridRow?: number
  gridCol?: number
  gridZ?: number  // 3D Torus的Z层
}

interface Edge {
  source: string
  target: string
  bandwidth?: number
  latency?: number  // 延迟 (ns)
  isSwitch?: boolean  // 是否为Switch连接
}

// 布局算法：圆形布局
function circleLayout(nodes: Node[], centerX: number, centerY: number, radius: number): Node[] {
  const count = nodes.length
  // 只有一个节点时，放在中心
  if (count === 1) {
    return [{ ...nodes[0], x: centerX, y: centerY }]
  }
  return nodes.map((node, i) => ({
    ...node,
    x: centerX + radius * Math.cos((2 * Math.PI * i) / count - Math.PI / 2),
    y: centerY + radius * Math.sin((2 * Math.PI * i) / count - Math.PI / 2),
  }))
}

// 布局算法：环形拓扑布局（用于ring连接）
function ringLayout(nodes: Node[], centerX: number, centerY: number, radius: number): Node[] {
  const count = nodes.length
  if (count === 1) {
    return [{ ...nodes[0], x: centerX, y: centerY }]
  }
  return nodes.map((node, i) => ({
    ...node,
    x: centerX + radius * Math.cos((2 * Math.PI * i) / count - Math.PI / 2),
    y: centerY + radius * Math.sin((2 * Math.PI * i) / count - Math.PI / 2),
  }))
}

// 布局算法：2D Torus/网格布局（用于torus_2d和grid连接）
// 标准Torus可视化：节点排成规则网格，环绕边画在外围
function torusLayout(nodes: Node[], width: number, height: number, padding: number = 120): Node[] {
  const count = nodes.length
  if (count === 1) {
    return [{ ...nodes[0], x: width / 2, y: height / 2 }]
  }
  // 计算最佳的行列数，尽量接近正方形
  const cols = Math.ceil(Math.sqrt(count))
  const rows = Math.ceil(count / cols)

  // 留出较大边距给环绕连接线
  const innerWidth = width - padding * 2
  const innerHeight = height - padding * 2
  const spacingX = cols > 1 ? innerWidth / (cols - 1) : 0
  const spacingY = rows > 1 ? innerHeight / (rows - 1) : 0

  // 居中偏移
  const offsetX = cols === 1 ? width / 2 : padding
  const offsetY = rows === 1 ? height / 2 : padding

  return nodes.map((node, i) => ({
    ...node,
    x: offsetX + (i % cols) * spacingX,
    y: offsetY + Math.floor(i / cols) * spacingY,
    // 存储网格位置信息用于连接线计算
    gridRow: Math.floor(i / cols),
    gridCol: i % cols,
  }))
}

// 计算Torus网格的行列数
function getTorusGridSize(count: number): { cols: number; rows: number } {
  const cols = Math.ceil(Math.sqrt(count))
  const rows = Math.ceil(count / cols)
  return { cols, rows }
}

// 3D Torus专用布局：等轴测投影，呈现3D立方体效果
function torus3DLayout(nodes: Node[], width: number, height: number, _padding: number = 100): Node[] {
  const count = nodes.length
  if (count <= 1) {
    return nodes.map(n => ({ ...n, x: width / 2, y: height / 2, gridRow: 0, gridCol: 0, gridZ: 0 }))
  }

  // 计算3D维度（尽量接近立方体）
  const dim = Math.max(2, Math.ceil(Math.pow(count, 1 / 3)))
  const nodesPerLayer = dim * dim

  // 等轴测投影参数
  const centerX = width / 2
  const centerY = height / 2
  const spacingX = 140  // X方向间距
  const spacingY = 120  // Y方向间距（垂直）
  const spacingZ = 90   // Z方向间距（深度，斜向）

  return nodes.map((node, i) => {
    const z = Math.floor(i / nodesPerLayer)
    const inLayerIndex = i % nodesPerLayer
    const row = Math.floor(inLayerIndex / dim)  // Y轴（上下）
    const col = inLayerIndex % dim              // X轴（左右）

    // 等轴测投影：
    // X轴向右，Y轴向下，Z轴向右上方（模拟深度）
    const x = centerX + (col - (dim - 1) / 2) * spacingX + (z - (dim - 1) / 2) * spacingZ * 0.6
    const y = centerY + (row - (dim - 1) / 2) * spacingY - (z - (dim - 1) / 2) * spacingZ * 0.5

    return {
      ...node,
      x,
      y,
      gridRow: row,
      gridCol: col,
      gridZ: z,
    }
  })
}

// 计算3D Torus的维度
function getTorus3DSize(count: number): { dim: number; layers: number } {
  const dim = Math.max(2, Math.ceil(Math.pow(count, 1 / 3)))
  const layers = Math.ceil(count / (dim * dim))
  return { dim, layers }
}

// 根据直连拓扑类型选择最佳布局
function getLayoutForTopology(
  topologyType: string,
  nodes: Node[],
  width: number,
  height: number
): Node[] {
  const centerX = width / 2
  const centerY = height / 2
  const radius = Math.min(width, height) * 0.35

  switch (topologyType) {
    case 'ring':
      return ringLayout(nodes, centerX, centerY, radius)
    case 'torus_2d':
      return torusLayout(nodes, width, height)
    case 'torus_3d':
      return torus3DLayout(nodes, width, height)
    case 'full_mesh_2d':
      // 2D FullMesh使用网格布局（行列全连接）
      return torusLayout(nodes, width, height)
    case 'full_mesh':
      // 全连接用圆形布局最清晰
      return circleLayout(nodes, centerX, centerY, radius)
    case 'none':
    default:
      // 无连接或默认用圆形
      return circleLayout(nodes, centerX, centerY, radius)
  }
}

// 布局算法：分层布局（用于显示Switch层级，设备节点排成一排）
function hierarchicalLayout(nodes: Node[], width: number, height: number): Node[] {
  // 按类型分组
  const switchNodes = nodes.filter(n => n.isSwitch)
  const deviceNodes = nodes.filter(n => !n.isSwitch)

  // 如果没有Switch，设备节点居中显示
  if (switchNodes.length === 0) {
    const centerY = height / 2
    if (deviceNodes.length === 1) {
      return [{ ...deviceNodes[0], x: width / 2, y: centerY }]
    }
    const spacing = width / (deviceNodes.length + 1)
    return deviceNodes.map((node, i) => ({
      ...node,
      x: spacing * (i + 1),
      y: centerY,
    }))
  }

  // Switch按subType分层
  const switchLayers: Record<string, Node[]> = {}
  switchNodes.forEach(n => {
    const layer = n.subType || 'default'
    if (!switchLayers[layer]) switchLayers[layer] = []
    switchLayers[layer].push(n)
  })

  // 层级顺序：device在最下面，然后是leaf, spine, core
  const layerOrder = ['leaf', 'spine', 'core']
  const sortedLayers = Object.keys(switchLayers).sort((a, b) => {
    const aIdx = layerOrder.indexOf(a)
    const bIdx = layerOrder.indexOf(b)
    return (aIdx === -1 ? 999 : aIdx) - (bIdx === -1 ? 999 : bIdx)
  })

  const totalLayers = sortedLayers.length + (deviceNodes.length > 0 ? 1 : 0)
  const layerSpacing = 100 // 每层之间的间距
  const totalHeight = (totalLayers - 1) * layerSpacing
  const startY = (height + totalHeight) / 2 // 垂直居中的起始Y（最底层）

  const result: Node[] = []

  // 设备节点在最底层
  if (deviceNodes.length > 0) {
    const y = startY
    const spacing = width / (deviceNodes.length + 1)
    deviceNodes.forEach((node, i) => {
      result.push({ ...node, x: spacing * (i + 1), y })
    })
  }

  // Switch节点按层级向上排列（在设备上方）
  sortedLayers.forEach((layer, layerIdx) => {
    const layerNodes = switchLayers[layer]
    const y = startY - layerSpacing * (layerIdx + (deviceNodes.length > 0 ? 1 : 0))
    const spacing = width / (layerNodes.length + 1)
    layerNodes.forEach((node, i) => {
      result.push({ ...node, x: spacing * (i + 1), y })
    })
  })

  return result
}

// 布局算法：混合布局（设备节点按拓扑排列，Switch节点在上方中央）
// 用于同时有Switch和节点直连的场景
function hybridLayout(
  nodes: Node[],
  width: number,
  height: number,
  directTopology: string
): Node[] {
  const switchNodes = nodes.filter(n => n.isSwitch)
  const deviceNodes = nodes.filter(n => !n.isSwitch)

  // 如果没有Switch，使用普通拓扑布局
  if (switchNodes.length === 0) {
    return getLayoutForTopology(directTopology, deviceNodes, width, height)
  }

  // Switch层数决定Switch区域高度
  const switchLayers: Record<string, Node[]> = {}
  switchNodes.forEach(n => {
    const layer = n.subType || 'default'
    if (!switchLayers[layer]) switchLayers[layer] = []
    switchLayers[layer].push(n)
  })
  const switchLayerCount = Object.keys(switchLayers).length

  // 动态计算区域划分：Switch区域更紧凑
  const switchLayerHeight = 50  // 每层Switch的高度
  const switchAreaHeight = switchLayerCount * switchLayerHeight
  const switchAreaTop = 60  // Switch起始位置（留出顶部空间）
  const gapBetween = 40  // Switch和设备之间的间隙

  // 设备节点区域
  const deviceAreaTop = switchAreaTop + switchAreaHeight + gapBetween
  const deviceAreaHeight = height - deviceAreaTop - 30  // 底部留30px

  const result: Node[] = []

  // 1. 设备节点按拓扑类型布局（在下方区域）
  const centerX = width / 2
  const centerY = deviceAreaTop + deviceAreaHeight / 2
  const radius = Math.min(width * 0.4, deviceAreaHeight * 0.45)

  let layoutedDevices: Node[]
  switch (directTopology) {
    case 'ring':
      layoutedDevices = ringLayout(deviceNodes, centerX, centerY, radius)
      break
    case 'torus_2d':
      layoutedDevices = torusLayout(deviceNodes, width, deviceAreaHeight, 80)
      layoutedDevices = layoutedDevices.map(n => ({ ...n, y: n.y + deviceAreaTop }))
      break
    case 'torus_3d':
      layoutedDevices = torus3DLayout(deviceNodes, width, deviceAreaHeight, 60)
      layoutedDevices = layoutedDevices.map(n => ({ ...n, y: n.y + deviceAreaTop - 30 }))
      break
    case 'full_mesh_2d':
      layoutedDevices = torusLayout(deviceNodes, width, deviceAreaHeight, 80)
      layoutedDevices = layoutedDevices.map(n => ({ ...n, y: n.y + deviceAreaTop }))
      break
    case 'full_mesh':
    default:
      layoutedDevices = circleLayout(deviceNodes, centerX, centerY, radius)
      break
  }
  result.push(...layoutedDevices)

  // 2. Switch节点按层级排列（在上方区域）
  const layerOrder = ['leaf', 'spine', 'core']
  const sortedLayers = Object.keys(switchLayers).sort((a, b) => {
    const aIdx = layerOrder.indexOf(a)
    const bIdx = layerOrder.indexOf(b)
    return (aIdx === -1 ? 999 : aIdx) - (bIdx === -1 ? 999 : bIdx)
  })

  sortedLayers.forEach((layer, layerIdx) => {
    const layerNodes = switchLayers[layer]
    const y = switchAreaTop + layerIdx * switchLayerHeight
    const spacing = width / (layerNodes.length + 1)
    layerNodes.forEach((node, i) => {
      result.push({ ...node, x: spacing * (i + 1), y })
    })
  })

  return result
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
  sourceNode = null,
  onSourceNodeChange,
  onManualConnect,
  manualConnections = [],
  onDeleteManualConnection,
  onDeleteConnection,
  layoutType = 'auto',
  onLayoutTypeChange,
}) => {
  void _onNavigateBack
  void _canGoBack
  const svgRef = useRef<SVGSVGElement>(null)
  const [zoom, setZoom] = useState(1)
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null)
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null)

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

    let nodeList: Node[] = []
    let edgeList: Edge[] = []
    let graphTitle = ''

    const width = 800
    const height = 600

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
        .filter(s => s.hierarchy_level === 'inter_rack')
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
        .filter(s => s.hierarchy_level === 'inter_board' && rackIds.has(s.parent_id))
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
  }, [topology, currentLevel, currentPod, currentRack, currentBoard, layoutType])

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

      {/* 右上角布局选择器悬浮框 */}
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
          style={{ display: 'block' }}
          onMouseMove={handleDragMove}
          onMouseUp={handleDragEnd}
          onMouseLeave={handleDragEnd}
        >
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

          {/* 渲染连接线 */}
          {edges.map((edge, i) => {
            const sourcePos = nodePositions.get(edge.source)
            const targetPos = nodePositions.get(edge.target)
            if (!sourcePos || !targetPos) return null

            const sourceNode = nodes.find(n => n.id === edge.source)
            const targetNode = nodes.find(n => n.id === edge.target)

            const bandwidthStr = edge.bandwidth ? `${edge.bandwidth}Gbps` : ''
            const latencyStr = edge.latency ? `${edge.latency}ns` : ''
            const propsStr = [bandwidthStr, latencyStr].filter(Boolean).join(', ')
            const tooltipContent = `${sourceNode?.label || edge.source} ↔ ${targetNode?.label || edge.target}${propsStr ? ` (${propsStr})` : ''}`

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
                    {/* 可见曲线 - Switch连接用蓝色，节点直连用灰色 */}
                    <path
                      d={pathD}
                      fill="none"
                      stroke={edge.isSwitch ? '#1890ff' : '#b0b0b0'}
                      strokeWidth={edge.isSwitch ? 2 : 1.5}
                      strokeOpacity={0.7}
                      style={{ pointerEvents: 'none' }}
                    />
                  </g>
                )
              }
            }

            // 普通直线连接 - 使用中心点，节点会遮盖线的端点
            return (
              <g key={`edge-${i}`}>
                {/* 透明触发层 - 增大点击区域 */}
                <line
                  x1={sourcePos.x}
                  y1={sourcePos.y}
                  x2={targetPos.x}
                  y2={targetPos.y}
                  stroke="transparent"
                  strokeWidth={16}
                  style={{ cursor: 'pointer' }}
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
                {/* 可见线条 - Switch连接用蓝色，节点直连用灰色 */}
                <line
                  x1={sourcePos.x}
                  y1={sourcePos.y}
                  x2={targetPos.x}
                  y2={targetPos.y}
                  stroke={edge.isSwitch ? '#1890ff' : '#b0b0b0'}
                  strokeWidth={edge.isSwitch ? 2 : 1.5}
                  strokeOpacity={0.7}
                  style={{ pointerEvents: 'none' }}
                />
              </g>
            )
          })}

          {/* 渲染手动连接线 */}
          {manualConnections
            .filter(mc => mc.hierarchy_level === getCurrentHierarchyLevel())
            .map((conn) => {
              const sourcePos = nodePositions.get(conn.source)
              const targetPos = nodePositions.get(conn.target)
              if (!sourcePos || !targetPos) return null

              const sourceNode = nodes.find(n => n.id === conn.source)
              const targetNode = nodes.find(n => n.id === conn.target)
              const manualTooltip = `${sourceNode?.label || conn.source} ↔ ${targetNode?.label || conn.target} (手动)`

              return (
                <g key={`manual-${conn.id}`}>
                  {/* 透明触发层 */}
                  <line
                    x1={sourcePos.x}
                    y1={sourcePos.y}
                    x2={targetPos.x}
                    y2={targetPos.y}
                    stroke="transparent"
                    strokeWidth={16}
                    style={{ cursor: 'pointer' }}
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
                  {/* 可见线条 - 编辑模式绿色虚线，普通模式与自动连接一致 */}
                  <line
                    x1={sourcePos.x}
                    y1={sourcePos.y}
                    x2={targetPos.x}
                    y2={targetPos.y}
                    stroke={connectionMode !== 'view' ? '#52c41a' : '#b0b0b0'}
                    strokeWidth={connectionMode !== 'view' ? 2.5 : 1.5}
                    strokeOpacity={connectionMode !== 'view' ? 1 : 0.6}
                    strokeDasharray={connectionMode !== 'view' ? '8,4' : undefined}
                    style={{ pointerEvents: 'none' }}
                  />
                </g>
              )
            })}

          {/* 渲染节点 */}
          {displayNodes.map((node) => {
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
            const isHovered = hoveredNodeId === node.id && connectionMode === 'view' && !isManualMode && !isDragging
            return (
              <g
                key={node.id}
                transform={`translate(${node.x}, ${node.y}) scale(${nodeScale})`}
                style={{
                  cursor: isManualMode ? 'move' : connectionMode !== 'view' ? 'crosshair' : 'pointer',
                  opacity: isDragging ? 0.7 : 1,
                  filter: isHovered ? 'drop-shadow(0 0 8px rgba(37, 99, 235, 0.5)) drop-shadow(0 0 16px rgba(37, 99, 235, 0.25))' : 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))',
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
      bodyStyle={{ padding: 0 }}
    >
      {graphContent}
    </Modal>
  )
}
