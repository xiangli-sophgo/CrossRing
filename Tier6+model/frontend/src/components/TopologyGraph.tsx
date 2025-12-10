import React, { useMemo, useRef, useState } from 'react'
import { Modal, Button, Space, Typography, Breadcrumb } from 'antd'
import { ZoomInOutlined, ZoomOutOutlined, HomeOutlined } from '@ant-design/icons'
import {
  HierarchicalTopology,
  PodConfig,
  RackConfig,
  BoardConfig,
  CHIP_TYPE_COLORS,
  SWITCH_LAYER_COLORS,
} from '../types'

const { Text } = Typography

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
  connections: { id: string; label: string; bandwidth?: number }[]
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
  const spacingZ = 80   // Z方向间距（深度，斜向）

  return nodes.map((node, i) => {
    const z = Math.floor(i / nodesPerLayer)
    const inLayerIndex = i % nodesPerLayer
    const row = Math.floor(inLayerIndex / dim)  // Y轴（上下）
    const col = inLayerIndex % dim              // X轴（左右）

    // 等轴测投影：
    // X轴向右，Y轴向下，Z轴向右上方（模拟深度）
    const x = centerX + (col - (dim - 1) / 2) * spacingX + (z - (dim - 1) / 2) * spacingZ * 0.5
    const y = centerY + (row - (dim - 1) / 2) * spacingY - (z - (dim - 1) / 2) * spacingZ * 0.4

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
    case 'hw_full_mesh':
      // HW FullMesh使用网格布局（行列全连接）
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

// 布局算法：分层布局（用于显示Switch层级）
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
}) => {
  void _onNavigateBack
  void _canGoBack
  const svgRef = useRef<SVGSVGElement>(null)
  const [zoom, setZoom] = useState(1)
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null)

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
        const dcSwitches = topology.switches.filter(s => s.hierarchy_level === 'datacenter')
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

      // Pod间连接和Switch连接
      edgeList = topology.connections
        .filter(c => {
          // Pod间直连
          if (c.source.startsWith('pod_') && !c.source.includes('/')) return true
          // Switch连接（数据中心层）
          if (c.type === 'switch') {
            const isSourceDc = c.source.startsWith('leaf_') || c.source.startsWith('spine_') || c.source.startsWith('core_')
            const isTargetDc = c.target.startsWith('leaf_') || c.target.startsWith('spine_') || c.target.startsWith('core_')
            return isSourceDc || isTargetDc || c.source.startsWith('pod_') || c.target.startsWith('pod_')
          }
          return false
        })
        .map(c => ({
          source: c.source,
          target: c.target,
          bandwidth: c.bandwidth,
        }))

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
          s.hierarchy_level === 'pod' && s.parent_id === currentPod.id
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

      // Rack间连接和Switch连接
      const rackIds = new Set(currentPod.racks.map(r => r.id))
      const podSwitchIds = new Set(
        (topology.switches || [])
          .filter(s => s.hierarchy_level === 'pod' && s.parent_id === currentPod.id)
          .map(s => s.id)
      )
      edgeList = topology.connections
        .filter(c => {
          const sourceInPod = rackIds.has(c.source) || podSwitchIds.has(c.source)
          const targetInPod = rackIds.has(c.target) || podSwitchIds.has(c.target)
          return sourceInPod && targetInPod
        })
        .map(c => ({
          source: c.source,
          target: c.target,
          bandwidth: c.bandwidth,
        }))

    } else if (currentLevel === 'rack' && currentRack) {
      // Rack层：显示所有Board和Rack层Switch
      graphTitle = `${currentRack.label} - Board拓扑`
      nodeList = currentRack.boards.map((board) => ({
        id: board.id,
        label: board.label,
        type: 'board',
        x: 0,
        y: 0,
        color: '#722ed1',
      }))

      // 添加Rack层Switch
      if (topology.switches) {
        const rackSwitches = topology.switches.filter(s =>
          s.hierarchy_level === 'rack' && s.parent_id === currentRack.id
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
          .filter(s => s.hierarchy_level === 'rack' && s.parent_id === currentRack.id)
          .map(s => s.id)
      )
      edgeList = topology.connections
        .filter(c => {
          const sourceInRack = boardIds.has(c.source) || rackSwitchIds.has(c.source)
          const targetInRack = boardIds.has(c.target) || rackSwitchIds.has(c.target)
          return sourceInRack && targetInRack
        })
        .map(c => ({
          source: c.source,
          target: c.target,
          bandwidth: c.bandwidth,
        }))

    } else if (currentLevel === 'board' && currentBoard) {
      // Board层：显示所有Chip
      graphTitle = `${currentBoard.label} - Chip拓扑`
      nodeList = currentBoard.chips.map((chip) => ({
        id: chip.id,
        label: chip.label || chip.type.toUpperCase(),
        type: chip.type,
        x: 0,
        y: 0,
        color: CHIP_TYPE_COLORS[chip.type] || '#666',
      }))

      // Chip间连接
      const chipIds = new Set(currentBoard.chips.map(c => c.id))
      edgeList = topology.connections
        .filter(c => chipIds.has(c.source) && chipIds.has(c.target))
        .map(c => ({
          source: c.source,
          target: c.target,
          bandwidth: c.bandwidth,
        }))
    }

    // 获取当前层级的直连拓扑类型
    let directTopology = 'full_mesh'
    if (topology.switch_config) {
      if (currentLevel === 'datacenter') {
        const dcConfig = topology.switch_config.datacenter_level
        if (!dcConfig?.enabled) {
          directTopology = dcConfig?.direct_topology || 'full_mesh'
        }
      } else if (currentLevel === 'pod') {
        const podConfig = topology.switch_config.pod_level
        if (!podConfig?.enabled) {
          directTopology = podConfig?.direct_topology || 'full_mesh'
        }
      } else if (currentLevel === 'rack') {
        const rackConfig = topology.switch_config.rack_level
        if (!rackConfig?.enabled) {
          directTopology = rackConfig?.direct_topology || 'full_mesh'
        }
      }
    }

    // 应用布局
    const hasSwitches = nodeList.some(n => n.isSwitch)
    if (hasSwitches) {
      // 有Switch时强制使用分层布局，确保Switch在上方
      nodeList = hierarchicalLayout(nodeList, width, height)
    } else {
      // 无Switch时根据直连拓扑类型选择布局
      nodeList = getLayoutForTopology(directTopology, nodeList, width, height)
    }

    return { nodes: nodeList, edges: edgeList, title: graphTitle, directTopology }
  }, [topology, currentLevel, currentPod, currentRack, currentBoard])

  // 根据节点数量计算缩放系数
  const nodeScale = useMemo(() => {
    const deviceNodes = nodes.filter(n => !n.isSwitch)
    const count = deviceNodes.length
    if (count <= 4) return 1
    if (count <= 8) return 0.85
    if (count <= 16) return 0.7
    if (count <= 32) return 0.55
    if (count <= 64) return 0.45
    return 0.35
  }, [nodes])

  // 创建节点位置映射
  const nodePositions = useMemo(() => {
    const map = new Map<string, { x: number; y: number }>()
    nodes.forEach(node => {
      map.set(node.id, { x: node.x, y: node.y })
    })
    return map
  }, [nodes])

  const handleZoomIn = () => setZoom(z => Math.min(z + 0.2, 2))
  const handleZoomOut = () => setZoom(z => Math.max(z - 0.2, 0.5))

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
                {index === 0 ? <><HomeOutlined style={{ marginRight: 4 }} />{item.label}</> : item.label}
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
      height: embedded ? 'calc(100% - 50px)' : 650,
      overflow: 'hidden',
      background: '#fafafa',
      position: 'relative',
    }}>
        <svg
          ref={svgRef}
          width="100%"
          height="100%"
          viewBox={`${400 - 400/zoom} ${300 - 300/zoom} ${800 / zoom} ${600 / zoom}`}
          style={{ display: 'block' }}
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
          </defs>

          {/* 2D Torus：渲染环绕连接的半椭圆弧 */}
          {directTopology === 'torus_2d' && (() => {
            const deviceNodes = nodes.filter(n => !n.isSwitch)
            const { cols, rows } = getTorusGridSize(deviceNodes.length)
            if (cols < 2 && rows < 2) return null

            // 找出每行和每列的首尾节点位置
            const rowArcs: { y: number; leftX: number; rightX: number; row: number }[] = []
            const colArcs: { x: number; topY: number; bottomY: number; col: number }[] = []

            for (let r = 0; r < rows; r++) {
              const nodesInRow = deviceNodes.filter(n => n.gridRow === r).sort((a, b) => a.x - b.x)
              if (nodesInRow.length >= 2) {
                rowArcs.push({
                  y: nodesInRow[0].y,
                  leftX: nodesInRow[0].x,
                  rightX: nodesInRow[nodesInRow.length - 1].x,
                  row: r
                })
              }
            }

            for (let c = 0; c < cols; c++) {
              const nodesInCol = deviceNodes.filter(n => n.gridCol === c).sort((a, b) => a.y - b.y)
              if (nodesInCol.length >= 2) {
                colArcs.push({
                  x: nodesInCol[0].x,
                  topY: nodesInCol[0].y,
                  bottomY: nodesInCol[nodesInCol.length - 1].y,
                  col: c
                })
              }
            }

            return (
              <g>
                {/* 行环绕弧 - 曲度根据宽度比例调整 */}
                {rowArcs.map((arc, i) => {
                  const width = arc.rightX - arc.leftX
                  // 曲度与宽度成比例，基础值 + 行号偏移避免重叠
                  const bulge = width * 0.06 + i * 6
                  const bottomY = arc.y + bulge
                  // 控制点向中间靠拢
                  const ctrl1X = arc.leftX + width * 0.15
                  const ctrl2X = arc.rightX - width * 0.15
                  return (
                    <path
                      key={`row-arc-${i}`}
                      d={`M ${arc.leftX} ${arc.y} C ${ctrl1X} ${bottomY}, ${ctrl2X} ${bottomY}, ${arc.rightX} ${arc.y}`}
                      fill="none"
                      stroke="#999"
                      strokeWidth={1.5}
                      strokeOpacity={0.6}
                    />
                  )
                })}
                {/* 列环绕弧 - 曲度根据高度比例调整 */}
                {colArcs.map((arc, i) => {
                  const height = arc.bottomY - arc.topY
                  // 曲度与高度成比例
                  const bulge = height * 0.10 + i * 6
                  const leftX = arc.x - bulge
                  // 控制点向中间靠拢
                  const ctrl1Y = arc.topY + height * 0.15
                  const ctrl2Y = arc.bottomY - height * 0.15
                  return (
                    <path
                      key={`col-arc-${i}`}
                      d={`M ${arc.x} ${arc.topY} C ${leftX} ${ctrl1Y}, ${leftX} ${ctrl2Y}, ${arc.x} ${arc.bottomY}`}
                      fill="none"
                      stroke="#999"
                      strokeWidth={1.5}
                      strokeOpacity={0.6}
                    />
                  )
                })}
              </g>
            )
          })()}

          {/* 3D Torus：X/Y/Z三个方向的环绕弧线（只有>=3个节点才画环绕弧） */}
          {directTopology === 'torus_3d' && (() => {
            const deviceNodes = nodes.filter(n => !n.isSwitch)
            const { dim, layers } = getTorus3DSize(deviceNodes.length)
            if (dim < 2) return null

            const arcs: JSX.Element[] = []

            // X方向环绕弧（连接同层同行的首尾节点col=0和col=dim-1）
            // 只有行内节点数>=3才画环绕弧
            for (let z = 0; z < layers; z++) {
              for (let r = 0; r < dim; r++) {
                const rowNodes = deviceNodes
                  .filter(n => n.gridZ === z && n.gridRow === r)
                  .sort((a, b) => a.gridCol! - b.gridCol!)
                if (rowNodes.length >= 3) {
                  const first = rowNodes[0]  // col=0
                  const last = rowNodes[rowNodes.length - 1]  // col=dim-1
                  // 在下方画U型弧连接首尾（两端弯曲，中间平缓）
                  const bulge = 30 + z * 10 + r * 8
                  const bottomY = Math.max(first.y, last.y) + bulge
                  // 控制点向中间靠拢，让中间更平
                  const width = Math.abs(last.x - first.x)
                  const ctrl1X = first.x + width * 0.15
                  const ctrl2X = last.x - width * 0.15
                  arcs.push(
                    <path
                      key={`x-arc-z${z}-r${r}`}
                      d={`M ${first.x} ${first.y} C ${ctrl1X} ${bottomY}, ${ctrl2X} ${bottomY}, ${last.x} ${last.y}`}
                      fill="none"
                      stroke="#999"
                      strokeWidth={1.5}
                      strokeOpacity={0.5}
                    />
                  )
                }
              }
            }

            // Y方向环绕弧（连接同层同列的首尾节点row=0和row=dim-1）
            // 只有列内节点数>=3才画环绕弧
            for (let z = 0; z < layers; z++) {
              for (let c = 0; c < dim; c++) {
                const colNodes = deviceNodes
                  .filter(n => n.gridZ === z && n.gridCol === c)
                  .sort((a, b) => a.gridRow! - b.gridRow!)
                if (colNodes.length >= 3) {
                  const first = colNodes[0]  // row=0
                  const last = colNodes[colNodes.length - 1]  // row=dim-1
                  // 在左侧画U型弧连接首尾
                  const bulge = 30 + z * 10 + c * 8
                  const leftX = Math.min(first.x, last.x) - bulge
                  // 控制点向中间靠拢，让中间更平
                  const height = Math.abs(last.y - first.y)
                  const ctrl1Y = first.y + height * 0.15
                  const ctrl2Y = last.y - height * 0.15
                  arcs.push(
                    <path
                      key={`y-arc-z${z}-c${c}`}
                      d={`M ${first.x} ${first.y} C ${leftX} ${ctrl1Y}, ${leftX} ${ctrl2Y}, ${last.x} ${last.y}`}
                      fill="none"
                      stroke="#999"
                      strokeWidth={1.5}
                      strokeOpacity={0.5}
                    />
                  )
                }
              }
            }

            // Z方向环绕弧（连接同行同列的首尾节点z=0和z=layers-1）
            // 只有深度>=3才画环绕弧
            for (let r = 0; r < dim; r++) {
              for (let c = 0; c < dim; c++) {
                const depthNodes = deviceNodes
                  .filter(n => n.gridRow === r && n.gridCol === c)
                  .sort((a, b) => a.gridZ! - b.gridZ!)
                if (depthNodes.length >= 3) {
                  const first = depthNodes[0]  // z=0
                  const last = depthNodes[depthNodes.length - 1]  // z=layers-1
                  // 在下方画U型弧连接首尾，Z方向曲度较小
                  const bulge = 5 + r * 6 + c * 5
                  const bottomY = Math.max(first.y, last.y) + bulge
                  // 控制点向中间靠拢，让中间更平
                  const width = Math.abs(last.x - first.x)
                  const ctrl1X = first.x + width * 0.2
                  const ctrl2X = last.x - width * 0.2
                  arcs.push(
                    <path
                      key={`z-arc-r${r}-c${c}`}
                      d={`M ${first.x} ${first.y} C ${ctrl1X} ${bottomY}, ${ctrl2X} ${bottomY}, ${last.x} ${last.y}`}
                      fill="none"
                      stroke="#999"
                      strokeWidth={1.5}
                      strokeOpacity={0.5}
                    />
                  )
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

            // 计算连接线的起点和终点（考虑节点半径和缩放系数）
            const dx = targetPos.x - sourcePos.x
            const dy = targetPos.y - sourcePos.y
            const dist = Math.sqrt(dx * dx + dy * dy)
            const nodeRadius = 25 * nodeScale  // 缩放后的节点半径

            const sourceNode = nodes.find(n => n.id === edge.source)
            const targetNode = nodes.find(n => n.id === edge.target)
            const tooltipContent = `${sourceNode?.label || edge.source} ↔ ${targetNode?.label || edge.target}${edge.bandwidth ? ` (${edge.bandwidth}Gbps)` : ''}`

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

            // HW FullMesh：非相邻节点使用曲线连接
            if (directTopology === 'hw_full_mesh') {
              const sameRow = sourceGridRow === targetGridRow
              const sameCol = sourceGridCol === targetGridCol
              const colDiff = Math.abs((sourceGridCol || 0) - (targetGridCol || 0))
              const rowDiff = Math.abs((sourceGridRow || 0) - (targetGridRow || 0))

              // 同行非相邻（列差>1）或同列非相邻（行差>1）使用曲线
              if ((sameRow && colDiff > 1) || (sameCol && rowDiff > 1)) {
                const midX = (sourcePos.x + targetPos.x) / 2
                const midY = (sourcePos.y + targetPos.y) / 2

                // 同行连接：曲线向下凸出
                if (sameRow) {
                  const width = Math.abs(targetPos.x - sourcePos.x)
                  const bulge = width * 0.15 + (sourceGridRow || 0) * 8
                  const bottomY = midY + bulge
                  const ctrl1X = sourcePos.x + width * 0.2
                  const ctrl2X = targetPos.x - width * 0.2
                  const startX = Math.min(sourcePos.x, targetPos.x)
                  const endX = Math.max(sourcePos.x, targetPos.x)

                  return (
                    <path
                      key={`edge-${i}`}
                      d={`M ${startX} ${sourcePos.y} C ${ctrl1X} ${bottomY}, ${ctrl2X} ${bottomY}, ${endX} ${sourcePos.y}`}
                      fill="none"
                      stroke="#e74c3c"
                      strokeWidth={1.5}
                      strokeOpacity={0.7}
                    />
                  )
                }

                // 同列连接：曲线向左凸出
                if (sameCol) {
                  const height = Math.abs(targetPos.y - sourcePos.y)
                  const bulge = height * 0.15 + (sourceGridCol || 0) * 8
                  const leftX = midX - bulge
                  const ctrl1Y = sourcePos.y + height * 0.2
                  const ctrl2Y = targetPos.y - height * 0.2
                  const startY = Math.min(sourcePos.y, targetPos.y)
                  const endY = Math.max(sourcePos.y, targetPos.y)

                  return (
                    <path
                      key={`edge-${i}`}
                      d={`M ${sourcePos.x} ${startY} C ${leftX} ${ctrl1Y}, ${leftX} ${ctrl2Y}, ${sourcePos.x} ${endY}`}
                      fill="none"
                      stroke="#3498db"
                      strokeWidth={1.5}
                      strokeOpacity={0.7}
                    />
                  )
                }
              }
            }

            // 普通直线连接
            const offsetX = dist > 0 ? (dx / dist) * nodeRadius : 0
            const offsetY = dist > 0 ? (dy / dist) * nodeRadius : 0

            return (
              <g key={`edge-${i}`}>
                <line
                  x1={sourcePos.x + offsetX}
                  y1={sourcePos.y + offsetY}
                  x2={targetPos.x - offsetX}
                  y2={targetPos.y - offsetY}
                  stroke="#b0b0b0"
                  strokeWidth={4}
                  strokeOpacity={0}
                  style={{ cursor: 'pointer' }}
                  onMouseEnter={(e) => {
                    const rect = svgRef.current?.getBoundingClientRect()
                    if (rect) {
                      setTooltip({
                        x: e.clientX - rect.left,
                        y: e.clientY - rect.top - 30,
                        content: tooltipContent,
                      })
                    }
                  }}
                  onMouseLeave={() => setTooltip(null)}
                />
                <line
                  x1={sourcePos.x + offsetX}
                  y1={sourcePos.y + offsetY}
                  x2={targetPos.x - offsetX}
                  y2={targetPos.y - offsetY}
                  stroke="#b0b0b0"
                  strokeWidth={1.5}
                  strokeOpacity={0.6}
                  style={{ pointerEvents: 'none' }}
                />
              </g>
            )
          })}

          {/* 渲染节点 */}
          {nodes.map((node) => {
            const isSwitch = node.isSwitch
            const portInfoText = node.portInfo
              ? `上行:${node.portInfo.uplink} 下行:${node.portInfo.downlink} 互联:${node.portInfo.inter}`
              : ''
            // 计算连接信息
            const nodeConnections = edges.filter(e => e.source === node.id || e.target === node.id)
            const connectedNodes = nodeConnections.map(e => {
              const otherId = e.source === node.id ? e.target : e.source
              const otherNode = nodes.find(n => n.id === otherId)
              return otherNode?.label || otherId
            })
            const connectionInfo = nodeConnections.length > 0
              ? `连接数: ${nodeConnections.length}\n连接到: ${connectedNodes.slice(0, 5).join(', ')}${connectedNodes.length > 5 ? '...' : ''}`
              : '无连接'
            const nodeTooltip = isSwitch
              ? `${node.label} (${node.subType?.toUpperCase() || 'SWITCH'})\n${portInfoText}\n${connectionInfo}`
              : `${node.label} (${node.type.toUpperCase()})\n${connectionInfo}`
            return (
              <g
                key={node.id}
                transform={`translate(${node.x}, ${node.y}) scale(${nodeScale})`}
                style={{ cursor: 'pointer' }}
                onClick={() => {
                  if (onNodeClick) {
                    const connections = nodeConnections.map(e => {
                      const otherId = e.source === node.id ? e.target : e.source
                      const otherNode = nodes.find(n => n.id === otherId)
                      return {
                        id: otherId,
                        label: otherNode?.label || otherId,
                        bandwidth: e.bandwidth
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
                }}
                onDoubleClick={() => {
                  if (onNodeDoubleClick && currentLevel !== 'board' && !isSwitch) {
                    onNodeDoubleClick(node.id, node.type)
                  }
                }}
                onMouseEnter={(e) => {
                  const rect = svgRef.current?.getBoundingClientRect()
                  if (rect) {
                    setTooltip({
                      x: e.clientX - rect.left,
                      y: e.clientY - rect.top - 35,
                      content: nodeTooltip,
                    })
                  }
                }}
                onMouseLeave={() => setTooltip(null)}
              >
                {/* 根据节点类型渲染不同形状 */}
                {isSwitch ? (
                  /* Switch: 网络交换机形状 - 扁平矩形带端口和指示灯 */
                  <g style={{ filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.15))' }}>
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
                  <g style={{ cursor: 'pointer', filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.15))' }}>
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
                  <g style={{ cursor: 'pointer', filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.15))' }}>
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
                  <g style={{ cursor: 'pointer', filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.15))' }}>
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
                  <g style={{ cursor: 'default', filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.15))' }}>
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
                    style={{ cursor: currentLevel !== 'board' ? 'pointer' : 'default', filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.15))' }}
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
          bottom: 10,
          left: 10,
          background: 'rgba(255,255,255,0.9)',
          padding: '8px 12px',
          borderRadius: 4,
          fontSize: 12,
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
            background: 'rgba(0, 0, 0, 0.85)',
            color: '#fff',
            padding: '8px 12px',
            borderRadius: 4,
            fontSize: 12,
            whiteSpace: 'pre-line',
            pointerEvents: 'none',
            zIndex: 1000,
            maxWidth: 300,
            textAlign: 'left',
          }}>
            {tooltip.content}
          </div>
        )}
      </div>
  )

  // 嵌入模式：直接渲染内容
  if (embedded) {
    return (
      <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
        {toolbar}
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
