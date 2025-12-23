/**
 * TopologyGraph 子组件集合
 * 包含: ManualConnectionLine, ControlPanel, EdgeRenderer, LevelPairSelector
 */
import React from 'react'
import { Segmented, Tooltip, Checkbox, Button, Typography } from 'antd'
import { UndoOutlined, RedoOutlined, ReloadOutlined } from '@ant-design/icons'
import { LayoutType, AdjacentLevelPair, LEVEL_PAIR_NAMES } from '../../types'
import { Node, Edge, LinkDetail, MultiLevelViewOptions } from './shared'

const { Text } = Typography

// ==========================================
// ManualConnectionLine - 手动连接线组件
// ==========================================

export interface AnimatedManualConnectionProps {
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

export const ManualConnectionLine: React.FC<AnimatedManualConnectionProps> = ({
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

  const getSegments = () => {
    if (!isCrossContainer || !containers || containers.length === 0) {
      return [{ from: sourcePos, to: targetPos, isDashed: false }]
    }

    const minZ = Math.min(sourcePos.zLayer, targetPos.zLayer)
    const maxZ = Math.max(sourcePos.zLayer, targetPos.zLayer)

    const passedContainers = containers
      .filter(c => c.zLayer >= minZ && c.zLayer <= maxZ)
      .sort((a, b) => a.zLayer - b.zLayer)

    if (passedContainers.length <= 2) {
      return [{ from: sourcePos, to: targetPos, isDashed: false }]
    }

    const segments: Array<{ from: { x: number; y: number }; to: { x: number; y: number }; isDashed: boolean }> = []

    const getX = (y: number) => {
      if (Math.abs(targetPos.y - sourcePos.y) < 0.001) return sourcePos.x
      const t = (y - sourcePos.y) / (targetPos.y - sourcePos.y)
      return sourcePos.x + t * (targetPos.x - sourcePos.x)
    }

    const startY = Math.min(sourcePos.y, targetPos.y)
    const endY = Math.max(sourcePos.y, targetPos.y)
    const isSourceAbove = sourcePos.y < targetPos.y

    const boundaryYs: Array<{ y: number; zLayer: number; isTop: boolean }> = []
    for (const c of passedContainers) {
      boundaryYs.push({ y: c.bounds.y, zLayer: c.zLayer, isTop: true })
      boundaryYs.push({ y: c.bounds.y + c.bounds.height, zLayer: c.zLayer, isTop: false })
    }
    boundaryYs.sort((a, b) => a.y - b.y)

    let currentY = startY
    let lastPoint: { x: number; y: number } = isSourceAbove
      ? { x: sourcePos.x, y: sourcePos.y }
      : { x: targetPos.x, y: targetPos.y }

    for (const boundary of boundaryYs) {
      if (boundary.y <= startY || boundary.y >= endY) continue

      const x = getX(boundary.y)
      const nextPoint = { x, y: boundary.y }

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

    const segments = getSegments()
    const hasMiddleSegments = segments.some(s => s.isDashed)

    if (!hasMiddleSegments) {
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

    return (
      <g style={transitionStyle}>
        <path
          d={`M ${sourcePos.x} ${sourcePos.y} Q ${ctrlX} ${ctrlY}, ${targetPos.x} ${targetPos.y}`}
          fill="none" stroke="transparent" strokeWidth={16}
          style={{ cursor: 'pointer' }} onClick={onClick}
        />
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

// ==========================================
// ControlPanel - 控制面板组件
// ==========================================

type ViewLevel = 'datacenter' | 'pod' | 'rack' | 'board' | 'chip'

export interface ControlPanelProps {
  multiLevelOptions?: MultiLevelViewOptions
  onMultiLevelOptionsChange?: (options: MultiLevelViewOptions) => void
  currentLevel: ViewLevel
  layoutType: LayoutType
  onLayoutTypeChange?: (type: LayoutType) => void
  isForceMode: boolean
  isForceSimulating: boolean
  isManualMode: boolean
  setIsManualMode: (value: boolean) => void
  manualPositions: Record<string, { x: number; y: number }>
  historyIndex: number
  historyLength: number
  onUndo: () => void
  onRedo: () => void
  onReset: () => void
  onLayoutChange?: () => void
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  multiLevelOptions,
  onMultiLevelOptionsChange,
  currentLevel,
  layoutType,
  onLayoutTypeChange,
  isForceMode,
  isForceSimulating,
  isManualMode,
  setIsManualMode,
  manualPositions,
  historyIndex,
  historyLength,
  onUndo,
  onRedo,
  onReset,
  onLayoutChange,
}) => {
  return (
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
          value={multiLevelOptions?.enabled ? 'multi' : 'single'}
          onChange={(value) => {
            if (onMultiLevelOptionsChange) {
              if (value === 'multi') {
                let levelPair: 'datacenter_pod' | 'pod_rack' | 'rack_board' | 'board_chip' = multiLevelOptions?.levelPair || 'datacenter_pod'
                if (currentLevel === 'datacenter') {
                  levelPair = 'datacenter_pod'
                } else if (currentLevel === 'pod') {
                  levelPair = 'pod_rack'
                } else if (currentLevel === 'rack') {
                  levelPair = 'rack_board'
                } else if (currentLevel === 'board') {
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
        <Segmented
          size="small"
          className="topology-layout-segmented"
          value={layoutType}
          onChange={(value) => {
            onLayoutTypeChange?.(value as LayoutType)
            onLayoutChange?.()
          }}
          options={[
            { label: '自动', value: 'auto' },
            { label: '环形', value: 'circle' },
            { label: '网格', value: 'grid' },
            { label: '力导向', value: 'force' },
          ]}
        />
        {isForceMode && (
          <Tooltip title={isForceSimulating ? '物理模拟进行中，可直接拖拽节点' : '物理模拟已稳定'}>
            <span style={{
              fontSize: 11,
              color: isForceSimulating ? '#52c41a' : '#8c8c8c',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
            }}>
              <span style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                backgroundColor: isForceSimulating ? '#52c41a' : '#d9d9d9',
                animation: isForceSimulating ? 'pulse 1s infinite' : 'none',
              }} />
              {isForceSimulating ? '模拟中' : '已稳定'}
            </span>
          </Tooltip>
        )}
        <div style={{ borderLeft: '1px solid rgba(0, 0, 0, 0.08)', height: 20 }} />
        <Checkbox
          checked={isManualMode}
          onChange={(e) => setIsManualMode(e.target.checked)}
          disabled={multiLevelOptions?.enabled || isForceMode}
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
                onClick={onUndo}
                disabled={historyIndex < 0}
              />
            </Tooltip>
            <Tooltip title="重做 (Ctrl+Y)">
              <Button
                type="text"
                size="small"
                icon={<RedoOutlined />}
                onClick={onRedo}
                disabled={historyIndex >= historyLength - 1}
              />
            </Tooltip>
            {Object.keys(manualPositions).length > 0 && (
              <Tooltip title="重置布局">
                <Button
                  type="text"
                  size="small"
                  icon={<ReloadOutlined />}
                  onClick={onReset}
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
  )
}

// ==========================================
// EdgeRenderer - 边渲染组件
// ==========================================

export interface EdgeRendererProps {
  edges: Edge[]
  nodes: Node[]
  nodePositions: Map<string, { x: number; y: number }>
  zoom: number
  selectedLinkId: string | null
  connectionMode: 'view' | 'select' | 'connect' | 'select_source' | 'select_target'
  isManualMode: boolean
  onLinkClick?: (link: LinkDetail | null) => void
  setTooltip: (tooltip: { x: number; y: number; content: string } | null) => void
  svgRef: React.RefObject<SVGSVGElement>
  getTrafficHeatmapStyle: (source: string, target: string) => { stroke: string; strokeWidth: number; utilization: number; trafficMb: number } | null
  directTopology?: string
}

export const renderExternalEdge = (
  edge: Edge,
  index: number,
  props: EdgeRendererProps
): React.ReactNode => {
  const { nodes, nodePositions, zoom, selectedLinkId, connectionMode, onLinkClick } = props

  let sourcePos = nodePositions.get(edge.source)
  if (!sourcePos) {
    const viewBoxWidth = 800 / zoom
    const viewBoxHeight = 600 / zoom
    sourcePos = { x: viewBoxWidth / 2, y: viewBoxHeight / 2 }
  }
  const sourceNode = nodes.find(n => n.id === edge.source)

  const viewBoxHeight = 600 / zoom
  const anchorX = sourcePos.x
  const anchorY = edge.externalDirection === 'upper' ? -20 : viewBoxHeight + 20

  const midX = (sourcePos.x + anchorX) / 2
  const midY = (sourcePos.y + anchorY) / 2
  const bulgeDir = edge.externalDirection === 'upper' ? -1 : 1
  const bulge = Math.abs(sourcePos.y - anchorY) * 0.3
  const ctrlX = midX
  const ctrlY = midY + bulgeDir * bulge

  const pathD = `M ${sourcePos.x} ${sourcePos.y} Q ${ctrlX} ${ctrlY}, ${anchorX} ${anchorY}`
  const shadowPathD = `M ${sourcePos.x + 2} ${sourcePos.y + 3} Q ${ctrlX + 2} ${ctrlY + 3}, ${anchorX + 2} ${anchorY + 3}`

  const edgeId = `${edge.source}-external-${edge.externalNodeId}`
  const isLinkSelected = selectedLinkId === edgeId

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (connectionMode !== 'view') return
    const externalId = edge.externalNodeId || ''
    const externalParts = externalId.split('/')
    const lastPart = externalParts[externalParts.length - 1] || ''
    let externalType = 'unknown'
    if (lastPart.startsWith('board')) externalType = 'board'
    else if (lastPart.startsWith('rack')) externalType = 'rack'
    else if (lastPart.startsWith('pod')) externalType = 'pod'
    else if (lastPart.includes('switch')) externalType = 'switch'
    const dirArrow = edge.externalDirection === 'upper' ? '↑' : '↓'
    onLinkClick?.({
      id: edgeId,
      sourceId: edge.source,
      sourceLabel: sourceNode?.label || edge.source,
      sourceType: sourceNode?.type || 'unknown',
      targetId: externalId,
      targetLabel: `${dirArrow} ${edge.externalNodeLabel || '外部节点'}`,
      targetType: externalType,
      bandwidth: edge.bandwidth,
      latency: edge.latency,
      isManual: false
    })
  }

  return (
    <g key={`external-edge-${index}`} style={{ transition: 'all 0.3s ease-out' }}>
      <path d={shadowPathD} fill="none" stroke="#000" strokeWidth={2}
        strokeOpacity={0.1} style={{ pointerEvents: 'none' }} />
      <path d={pathD} fill="none" stroke="transparent" strokeWidth={12}
        style={{ cursor: 'pointer' }} onClick={handleClick} />
      <path d={pathD} fill="none" stroke={isLinkSelected ? '#52c41a' : '#faad14'}
        strokeWidth={isLinkSelected ? 2.5 : 2}
        strokeOpacity={isLinkSelected ? 1 : 0.8}
        strokeDasharray="8,4,2,4"
        style={{ pointerEvents: 'none' }} />
      <g transform={`translate(${anchorX}, ${anchorY})`}>
        <circle r={6} fill={isLinkSelected ? '#52c41a' : '#faad14'} opacity={0.9} />
        <text y={edge.externalDirection === 'upper' ? -10 : 16} textAnchor="middle"
          fontSize={9} fill="#666" fontWeight={500}>
          {edge.externalNodeLabel || '上层'}
        </text>
      </g>
      {isLinkSelected && (
        <path d={pathD} fill="none" stroke="#52c41a" strokeWidth={4}
          strokeOpacity={0.2} style={{ pointerEvents: 'none', filter: 'blur(3px)' }} />
      )}
    </g>
  )
}

export const renderIndirectEdge = (
  edge: Edge,
  index: number,
  props: EdgeRendererProps
): React.ReactNode => {
  const { nodes, nodePositions, selectedLinkId, connectionMode, onLinkClick } = props

  const sourcePos = nodePositions.get(edge.source)
  const targetPos = nodePositions.get(edge.target)
  if (!sourcePos || !targetPos) return null

  const sourceNode = nodes.find(n => n.id === edge.source)
  const targetNode = nodes.find(n => n.id === edge.target)

  const dist = Math.sqrt(
    Math.pow(targetPos.x - sourcePos.x, 2) + Math.pow(targetPos.y - sourcePos.y, 2)
  )
  const midX = (sourcePos.x + targetPos.x) / 2
  const midY = (sourcePos.y + targetPos.y) / 2
  const bulge = Math.max(dist * 0.4, 60)
  const ctrlX = midX
  const ctrlY = midY - bulge

  const pathD = `M ${sourcePos.x} ${sourcePos.y} Q ${ctrlX} ${ctrlY}, ${targetPos.x} ${targetPos.y}`
  const shadowPathD = `M ${sourcePos.x + 2} ${sourcePos.y + 4} Q ${ctrlX + 2} ${ctrlY + 4}, ${targetPos.x + 2} ${targetPos.y + 4}`

  const edgeId = `${edge.source}-indirect-${edge.target}`
  const isLinkSelected = selectedLinkId === edgeId

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (connectionMode !== 'view') return
    onLinkClick?.({
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

  return (
    <g key={`indirect-edge-${index}`} style={{ transition: 'all 0.3s ease-out' }}>
      <path d={shadowPathD} fill="none" stroke="#000" strokeWidth={1.5}
        strokeOpacity={0.08} style={{ pointerEvents: 'none' }} />
      <path d={pathD} fill="none" stroke="transparent" strokeWidth={12}
        style={{ cursor: 'pointer' }} onClick={handleClick} />
      <path d={pathD} fill="none" stroke={isLinkSelected ? '#52c41a' : '#722ed1'}
        strokeWidth={isLinkSelected ? 2 : 1.5}
        strokeOpacity={isLinkSelected ? 0.9 : 0.5}
        strokeDasharray="3,3"
        style={{ pointerEvents: 'none' }} />
      <g transform={`translate(${ctrlX}, ${ctrlY})`}>
        <circle r={5} fill={isLinkSelected ? '#52c41a' : '#722ed1'} opacity={0.7} />
        <text y={-8} textAnchor="middle" fontSize={8} fill="#666" fontWeight={500}>
          via {edge.viaNodeLabel || '上层'}
        </text>
      </g>
      {isLinkSelected && (
        <path d={pathD} fill="none" stroke="#52c41a" strokeWidth={3}
          strokeOpacity={0.2} style={{ pointerEvents: 'none', filter: 'blur(3px)' }} />
      )}
    </g>
  )
}

export const getEdgeStyle = (
  edge: Edge,
  isLinkSelected: boolean,
  getTrafficHeatmapStyle: EdgeRendererProps['getTrafficHeatmapStyle']
): { stroke: string; strokeWidth: number; strokeDasharray?: string } => {
  const trafficStyle = getTrafficHeatmapStyle(edge.source, edge.target)
  if (trafficStyle && !isLinkSelected) {
    return {
      stroke: trafficStyle.stroke,
      strokeWidth: trafficStyle.strokeWidth,
    }
  }
  if (!edge.connectionType) {
    return {
      stroke: isLinkSelected ? '#52c41a' : (edge.isSwitch ? '#1890ff' : '#b0b0b0'),
      strokeWidth: isLinkSelected ? 3 : (edge.isSwitch ? 2 : 1.5),
    }
  }
  switch (edge.connectionType) {
    case 'intra_upper':
      return { stroke: isLinkSelected ? '#52c41a' : '#1890ff', strokeWidth: isLinkSelected ? 3 : 2 }
    case 'intra_lower':
      return { stroke: isLinkSelected ? '#52c41a' : '#52c41a', strokeWidth: isLinkSelected ? 3 : 1.5 }
    case 'inter_level':
      return { stroke: isLinkSelected ? '#52c41a' : '#faad14', strokeWidth: isLinkSelected ? 3 : 1.5, strokeDasharray: '6,3' }
    default:
      return { stroke: '#b0b0b0', strokeWidth: 1.5 }
  }
}

// ==========================================
// LevelPairSelector - 层级选择组件
// ==========================================

interface LevelPairSelectorProps {
  value: AdjacentLevelPair | null
  onChange: (pair: AdjacentLevelPair | null) => void
  disabled?: boolean
  currentLevel: string
  hasCurrentPod: boolean
  hasCurrentRack: boolean
  hasCurrentBoard: boolean
}

function getAvailableOptions(
  currentLevel: string,
  hasCurrentPod: boolean,
  hasCurrentRack: boolean,
  hasCurrentBoard: boolean
): { label: string; value: string; disabled?: boolean }[] {
  const options: { label: string; value: string; disabled?: boolean }[] = [
    { label: '单层级', value: 'single' },
  ]

  if (currentLevel === 'datacenter') {
    options.push({ label: LEVEL_PAIR_NAMES.datacenter_pod, value: 'datacenter_pod' })
  }

  if (currentLevel === 'pod' || (currentLevel === 'datacenter' && hasCurrentPod)) {
    options.push({
      label: LEVEL_PAIR_NAMES.pod_rack,
      value: 'pod_rack',
      disabled: currentLevel === 'datacenter' && !hasCurrentPod,
    })
  }

  if (currentLevel === 'rack' || (currentLevel === 'pod' && hasCurrentRack)) {
    options.push({
      label: LEVEL_PAIR_NAMES.rack_board,
      value: 'rack_board',
      disabled: currentLevel === 'pod' && !hasCurrentRack,
    })
  }

  if (currentLevel === 'rack' && hasCurrentBoard) {
    options.push({
      label: LEVEL_PAIR_NAMES.board_chip,
      value: 'board_chip',
    })
  }

  if (currentLevel === 'board' && hasCurrentRack) {
    if (!options.some(o => o.value === 'rack_board')) {
      options.push({
        label: LEVEL_PAIR_NAMES.rack_board,
        value: 'rack_board',
      })
    }
  }

  return options
}

export const LevelPairSelector: React.FC<LevelPairSelectorProps> = ({
  value,
  onChange,
  disabled = false,
  currentLevel,
  hasCurrentPod,
  hasCurrentRack,
  hasCurrentBoard,
}) => {
  const options = getAvailableOptions(currentLevel, hasCurrentPod, hasCurrentRack, hasCurrentBoard)

  const handleChange = (val: string | number) => {
    if (val === 'single') {
      onChange(null)
    } else {
      onChange(val as AdjacentLevelPair)
    }
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <Text style={{ fontSize: 12, color: '#666', whiteSpace: 'nowrap' }}>视图模式:</Text>
      <Segmented
        size="small"
        options={options}
        value={value || 'single'}
        onChange={handleChange}
        disabled={disabled}
      />
    </div>
  )
}
