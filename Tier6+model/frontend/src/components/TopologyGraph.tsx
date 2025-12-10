import React, { useMemo, useRef, useEffect, useState } from 'react'
import { Modal, Button, Select, Space, Typography } from 'antd'
import { FullscreenOutlined, ZoomInOutlined, ZoomOutOutlined } from '@ant-design/icons'
import {
  HierarchicalTopology,
  PodConfig,
  RackConfig,
  BoardConfig,
  ConnectionConfig,
  CHIP_TYPE_COLORS,
  SwitchInstance,
  SWITCH_LAYER_COLORS,
} from '../types'

const { Text } = Typography

interface TopologyGraphProps {
  visible: boolean
  onClose: () => void
  topology: HierarchicalTopology | null
  currentLevel: 'datacenter' | 'pod' | 'rack' | 'board' | 'chip'
  currentPod?: PodConfig | null
  currentRack?: RackConfig | null
  currentBoard?: BoardConfig | null
  onNodeDoubleClick?: (nodeId: string, nodeType: string) => void
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
}

interface Edge {
  source: string
  target: string
  bandwidth?: number
}

// 布局算法：圆形布局
function circleLayout(nodes: Node[], centerX: number, centerY: number, radius: number): Node[] {
  const count = nodes.length
  return nodes.map((node, i) => ({
    ...node,
    x: centerX + radius * Math.cos((2 * Math.PI * i) / count - Math.PI / 2),
    y: centerY + radius * Math.sin((2 * Math.PI * i) / count - Math.PI / 2),
  }))
}

// 布局算法：网格布局
function gridLayout(nodes: Node[], startX: number, startY: number, cols: number, spacingX: number, spacingY: number): Node[] {
  return nodes.map((node, i) => ({
    ...node,
    x: startX + (i % cols) * spacingX,
    y: startY + Math.floor(i / cols) * spacingY,
  }))
}

// 布局算法：分层布局（用于显示Switch层级）
function hierarchicalLayout(nodes: Node[], width: number, height: number): Node[] {
  // 按类型分组
  const switchNodes = nodes.filter(n => n.isSwitch)
  const deviceNodes = nodes.filter(n => !n.isSwitch)

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
  const layerHeight = height / (totalLayers + 1)

  const result: Node[] = []

  // 设备节点在最底层
  if (deviceNodes.length > 0) {
    const y = height - layerHeight * 0.5
    const spacing = width / (deviceNodes.length + 1)
    deviceNodes.forEach((node, i) => {
      result.push({ ...node, x: spacing * (i + 1), y })
    })
  }

  // Switch节点按层级向上排列
  sortedLayers.forEach((layer, layerIdx) => {
    const layerNodes = switchLayers[layer]
    const y = height - layerHeight * (layerIdx + 1.5 + (deviceNodes.length > 0 ? 1 : 0))
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
}) => {
  const svgRef = useRef<SVGSVGElement>(null)
  const [zoom, setZoom] = useState(1)
  const [layout, setLayout] = useState<'circle' | 'grid' | 'hierarchical'>('circle')
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null)

  // 根据当前层级生成节点和边
  const { nodes, edges, title } = useMemo(() => {
    if (!topology) return { nodes: [], edges: [], title: '' }

    let nodeList: Node[] = []
    let edgeList: Edge[] = []
    let graphTitle = ''

    const width = 800
    const height = 600
    const centerX = width / 2
    const centerY = height / 2

    if (currentLevel === 'datacenter') {
      // 数据中心层：显示所有Pod和数据中心层Switch
      graphTitle = '数据中心拓扑'
      nodeList = topology.pods.map((pod, i) => ({
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
      nodeList = currentPod.racks.map((rack, i) => ({
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
      nodeList = currentRack.boards.map((board, i) => ({
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
      nodeList = currentBoard.chips.map((chip, i) => ({
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

    // 应用布局
    const hasSwitches = nodeList.some(n => n.isSwitch)
    if (layout === 'hierarchical' || (hasSwitches && layout !== 'grid')) {
      // 有Switch时默认使用分层布局
      nodeList = hierarchicalLayout(nodeList, width, height)
    } else if (layout === 'circle') {
      const radius = Math.min(width, height) * 0.35
      nodeList = circleLayout(nodeList, centerX, centerY, radius)
    } else {
      const cols = Math.ceil(Math.sqrt(nodeList.length))
      const spacingX = width / (cols + 1)
      const spacingY = height / (Math.ceil(nodeList.length / cols) + 1)
      nodeList = gridLayout(nodeList, spacingX, spacingY, cols, spacingX, spacingY)
    }

    return { nodes: nodeList, edges: edgeList, title: graphTitle }
  }, [topology, currentLevel, currentPod, currentRack, currentBoard, layout])

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

  return (
    <Modal
      title={
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingRight: 30 }}>
          <span>{title || '抽象拓扑图'}</span>
          <Space>
            <Select
              size="small"
              value={layout}
              onChange={setLayout}
              style={{ width: 100 }}
              options={[
                { value: 'circle', label: '圆形布局' },
                { value: 'grid', label: '网格布局' },
                { value: 'hierarchical', label: '分层布局' },
              ]}
            />
            <Button size="small" icon={<ZoomOutOutlined />} onClick={handleZoomOut} />
            <Button size="small" icon={<ZoomInOutlined />} onClick={handleZoomIn} />
          </Space>
        </div>
      }
      open={visible}
      onCancel={onClose}
      footer={null}
      width={900}
      bodyStyle={{ padding: 0 }}
    >
      <div style={{
        width: '100%',
        height: 650,
        overflow: 'hidden',
        background: '#fafafa',
        position: 'relative',
      }}>
        <svg
          ref={svgRef}
          width="100%"
          height="100%"
          viewBox={`0 0 ${800 / zoom} ${600 / zoom}`}
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

          {/* 渲染连接线 */}
          {edges.map((edge, i) => {
            const sourcePos = nodePositions.get(edge.source)
            const targetPos = nodePositions.get(edge.target)
            if (!sourcePos || !targetPos) return null

            // 计算连接线的起点和终点（考虑节点半径）
            const dx = targetPos.x - sourcePos.x
            const dy = targetPos.y - sourcePos.y
            const dist = Math.sqrt(dx * dx + dy * dy)
            const nodeRadius = 25
            const offsetX = (dx / dist) * nodeRadius
            const offsetY = (dy / dist) * nodeRadius

            const sourceNode = nodes.find(n => n.id === edge.source)
            const targetNode = nodes.find(n => n.id === edge.target)
            const tooltipContent = `${sourceNode?.label || edge.source} ↔ ${targetNode?.label || edge.target}${edge.bandwidth ? ` (${edge.bandwidth}Gbps)` : ''}`

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
          {nodes.map((node, i) => {
            const isSwitch = node.isSwitch
            const portInfoText = node.portInfo
              ? `上行:${node.portInfo.uplink} 下行:${node.portInfo.downlink} 互联:${node.portInfo.inter}`
              : ''
            const nodeTooltip = isSwitch
              ? `${node.label} (${node.subType?.toUpperCase() || 'SWITCH'})\n${portInfoText}`
              : `${node.label} (${node.type.toUpperCase()})`
            return (
              <g
                key={node.id}
                transform={`translate(${node.x}, ${node.y})`}
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
                {/* Switch节点使用菱形 */}
                {isSwitch ? (
                  <polygon
                    points="0,-25 25,0 0,25 -25,0"
                    fill={node.color}
                    stroke="#fff"
                    strokeWidth={2}
                    style={{ filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.2))' }}
                  />
                ) : (
                  /* 普通节点使用圆形 */
                  <circle
                    r={25}
                    fill={node.color}
                    stroke="#fff"
                    strokeWidth={2}
                    style={{ cursor: currentLevel !== 'board' ? 'pointer' : 'default', filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.2))' }}
                  />
                )}
                {/* 节点标签 */}
                <text
                  y={4}
                  fontSize={isSwitch ? 8 : 10}
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
            padding: '6px 12px',
            borderRadius: 4,
            fontSize: 12,
            whiteSpace: 'nowrap',
            pointerEvents: 'none',
            zIndex: 1000,
          }}>
            {tooltip.content}
          </div>
        )}
      </div>
    </Modal>
  )
}
