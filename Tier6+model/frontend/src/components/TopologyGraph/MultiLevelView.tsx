import React from 'react'
import { Node, Edge, LayoutType, ManualConnection } from './shared'
import { ManualConnectionLine } from './components'
import { HierarchyLevel } from '../../types'

// 多层级视图渲染所需的props
export interface MultiLevelViewProps {
  // 数据
  displayNodes: Node[]
  edges: Edge[]
  manualConnections: ManualConnection[]
  // 状态
  zoom: number
  selectedNodeId: string | null
  selectedLinkId: string | null
  hoveredLayerIndex: number | null
  setHoveredLayerIndex: (idx: number | null) => void
  // 展开/收缩动画
  expandingContainer: { id: string; type: string } | null
  collapsingContainer: { id: string; type: string } | null
  collapseAnimationStarted: boolean
  // 动画完成回调
  onExpandAnimationEnd?: (nodeId: string, nodeType: string) => void
  // 回调
  connectionMode: 'view' | 'select' | 'connect' | 'select_source' | 'select_target'
  onNodeClick?: (node: any) => void
  onLinkClick?: (link: any) => void
  onNodeDoubleClick?: (nodeId: string, nodeType: string) => void
  // 其他
  layoutType: LayoutType
  renderNode: (node: Node, options: { keyPrefix: string; scale?: number; isSelected?: boolean; onClick?: () => void; useMultiLevelConfig?: boolean }) => JSX.Element
  getCurrentHierarchyLevel: () => HierarchyLevel
  // 拖拽相关
  handleDragMove: (e: React.MouseEvent) => void
  handleDragEnd: () => void
  // 选中节点
  selectedNodes: Set<string>
  targetNodes: Set<string>
  onSelectedNodesChange?: (nodes: Set<string>) => void
  onTargetNodesChange?: (nodes: Set<string>) => void
}

export const MultiLevelView: React.FC<MultiLevelViewProps> = ({
  displayNodes,
  edges,
  manualConnections,
  zoom,
  selectedNodeId,
  selectedLinkId,
  hoveredLayerIndex,
  setHoveredLayerIndex,
  expandingContainer,
  collapsingContainer,
  collapseAnimationStarted,
  onExpandAnimationEnd,
  connectionMode,
  onNodeClick,
  onLinkClick,
  onNodeDoubleClick,
  layoutType,
  renderNode,
  getCurrentHierarchyLevel,
  handleDragMove,
  handleDragEnd,
  selectedNodes,
  targetNodes,
  onSelectedNodesChange,
  onTargetNodesChange,
}) => {
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
    for (const container of containers) {
      if (container.singleLevelData?.nodes.some(n => n.id === selectedNodeId)) {
        selectedLayerIndex = container.zLayer ?? null
        break
      }
    }
  }

  // 获取所有inter_level边（用于跨层级连线渲染）
  const interLevelEdges = edges.filter(e => e.connectionType === 'inter_level')
  void interLevelEdges // 在简化版本中暂未使用
  const baseSkewAngle = -30
  const skewTan = Math.tan(baseSkewAngle * Math.PI / 180)

  // 查找节点所在的容器及其zLayer和索引（用于跨层级连线）
  const getNodeContainerInfo = (nodeId: string) => {
    for (const container of containers) {
      if (container.singleLevelData?.nodes.some(n => n.id === nodeId)) {
        const idParts = container.id.split('/')
        const lastPart = idParts[idParts.length - 1]
        const indexMatch = lastPart.match(/_(\d+)$/)
        const containerIndex = indexMatch ? parseInt(indexMatch[1], 10) : 0
        return { container, zLayer: container.zLayer ?? 0, containerIndex }
      }
    }
    return null
  }
  void getNodeContainerInfo // 在简化版本中暂未使用

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

  // 视口中心
  const viewportCenterX = 400
  const viewportCenterY = 300

  // 获取当前层级的手动连接
  const currentManualConnections = manualConnections.filter(mc => mc.hierarchy_level === getCurrentHierarchyLevel())

  // 渲染跨容器手动连接
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

    // 检查节点是否属于某个容器
    const isNodeInContainer = (nodeId: string, containerId: string): boolean => {
      const container = containers.find(c => c.id === containerId)
      if (!container || !container.singleLevelData) return false
      return container.singleLevelData.nodes.some((n: any) => n.id === nodeId)
    }

    // 获取选中的容器ID（如果选中的是容器）
    const selectedContainerId = containers.find(c => c.id === selectedNodeId)?.id || null
    // 获取悬停的容器ID
    const hoveredContainerId = hoveredLayerIndex !== null
      ? containers.find(c => c.zLayer === hoveredLayerIndex)?.id || null
      : null
    // 默认容器：zLayer 最小的容器（最上面的容器）
    const defaultContainerId = containers.length > 0
  ? containers.reduce((min, c) => (c.zLayer ?? Infinity) < (min.zLayer ?? Infinity) ? c : min).id
  : null
    // 当前活跃的容器（悬停优先，其次选中，最后默认）
    const activeContainerId = hoveredContainerId || selectedContainerId || defaultContainerId

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
      // 多层级视图中所有连接都使用曲线显示
      const isCrossContainer = true

      const manualEdgeId = `${conn.source}-${conn.target}`
      const isLinkSelected = selectedLinkId === manualEdgeId || selectedLinkId === `${conn.target}-${conn.source}`

      // 检查连线是否与活跃的容器（悬停或选中）相关
      let linkOpacity = 1
      if (activeContainerId && !isLinkSelected) {
        const sourceInActive = isNodeInContainer(conn.source, activeContainerId)
        const targetInActive = isNodeInContainer(conn.target, activeContainerId)
        // 如果两端节点都不属于活跃的容器，降低透明度
        if (!sourceInActive && !targetInActive) {
          linkOpacity = 0.2
        }
      }

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
        <g key={`manual-conn-${conn.id}`} style={{ opacity: linkOpacity, transition: 'opacity 0.2s ease' }}>
          <ManualConnectionLine
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
        </g>
      )
    })
  }

  // 渲染容器
  const containersRendered = containers.map(containerNode => {
    const bounds = containerNode.containerBounds!
    const zLayer = containerNode.zLayer ?? 0
    const isExpanding = expandingContainer?.id === containerNode.id
    const isOtherExpanding = expandingContainer !== null && !isExpanding
    const isCollapsing = collapsingContainer?.id === containerNode.id
    const isOtherCollapsing = collapsingContainer !== null && !isCollapsing

    const activeLayerIndex = hoveredLayerIndex ?? selectedLayerIndex
    // 获取活跃容器（悬停或选中的容器）
    const activeContainer = hoveredLayerIndex !== null
      ? containers.find(c => c.zLayer === hoveredLayerIndex)
      : containers.find(c => c.id === selectedNodeId)

    let yOffset = 0
    let layerOpacity = 1
    // 当前容器是否需要变透明（上方的非活跃容器）
    const shouldDimContainer = activeLayerIndex !== null && zLayer < activeLayerIndex && !expandingContainer && !collapsingContainer
    if (shouldDimContainer) {
      yOffset = -liftDistance
      layerOpacity = 0.15
    }

    // 获取与活跃容器有连线的节点ID集合（这些节点不变透明）
    const getConnectedNodeIds = (): Set<string> => {
      if (!activeContainer?.singleLevelData) return new Set()
      const activeNodeIds = new Set(activeContainer.singleLevelData.nodes.map((n: any) => n.id))
      const connectedIds = new Set<string>()
      currentManualConnections.forEach(conn => {
        if (activeNodeIds.has(conn.source)) {
          connectedIds.add(conn.target)
        }
        if (activeNodeIds.has(conn.target)) {
          connectedIds.add(conn.source)
        }
      })
      return connectedIds
    }
    const connectedNodeIds = activeContainer ? getConnectedNodeIds() : new Set<string>()

    if (isOtherExpanding) {
      layerOpacity = 0
    }

    if (isOtherCollapsing) {
      if (!collapseAnimationStarted) {
        layerOpacity = 0
      } else {
        layerOpacity = 1
      }
    }

    const x = bounds.x
    const w = bounds.width
    const h = bounds.height

    // 展开动画计算
    let animX = x
    let animY = bounds.y + yOffset
    let animW = w
    let animH = h
    let animSkewAngle = baseSkewAngle
    let animOpacity = layerOpacity

    if (isExpanding) {
      const viewportWidth = 800 / zoom
      const viewportHeight = 600 / zoom
      animX = viewportCenterX - viewportWidth / 2
      animY = viewportCenterY - viewportHeight / 2
      animW = viewportWidth
      animH = viewportHeight
      animSkewAngle = 0
      animOpacity = 1
    }

    if (isCollapsing) {
      if (!collapseAnimationStarted) {
        const viewportWidth = 800 / zoom
        const viewportHeight = 600 / zoom
        animX = viewportCenterX - viewportWidth / 2
        animY = viewportCenterY - viewportHeight / 2
        animW = viewportWidth
        animH = viewportHeight
        animSkewAngle = 0
        animOpacity = 1
      } else {
        animX = x
        animY = bounds.y + yOffset
        animW = w
        animH = h
        animSkewAngle = baseSkewAngle
        animOpacity = layerOpacity
      }
    }

    const isSelected = selectedNodeId === containerNode.id
    const isHovered = activeLayerIndex === zLayer

    // 获取容器在当前层级的边
    const layerEdges = edges.filter(e => {
      const sourceInContainer = containerNode.singleLevelData?.nodes.some(n => n.id === e.source)
      const targetInContainer = containerNode.singleLevelData?.nodes.some(n => n.id === e.target)
      return sourceInContainer || targetInContainer
    })

    // 获取容器内的节点（不包括Switch）
    const layerNodes = containerNode.singleLevelData?.nodes.filter(n => !n.isSwitch) || []

    // 动画时使用整体透明度，普通悬停时不设置整体透明度（节点单独控制）
    const isAnimating = expandingContainer || collapsingContainer
    const containerGroupOpacity = isAnimating ? animOpacity : 1

    return (
      <g
        key={containerNode.id}
        style={{
          transition: isAnimating
            ? 'transform 0.5s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.5s ease'
            : 'transform 0.3s ease',
          opacity: containerGroupOpacity,
        }}
        transform={`translate(0, ${animY - bounds.y})`}
        onMouseEnter={() => !expandingContainer && !collapsingContainer && setHoveredLayerIndex(zLayer)}
        onMouseLeave={() => {
          // 在连接模式下不重置 hoveredLayerIndex，避免层级跳动
          if (connectionMode !== 'view') return
          if (!expandingContainer && !collapsingContainer) {
            setHoveredLayerIndex(null)
          }
        }}
        onTransitionEnd={(e) => {
          // 只在展开动画完成时触发（检查 transform 属性避免重复触发）
          if (isExpanding && e.propertyName === 'transform' && onExpandAnimationEnd) {
            onExpandAnimationEnd(containerNode.id, containerNode.type)
          }
        }}
      >
        {/* 容器背景 */}
        <g
          style={{
            transition: expandingContainer || collapsingContainer
              ? 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)'
              : 'all 0.3s ease',
            transformOrigin: `${animX + animW / 2}px ${bounds.y + animH / 2}px`,
            transform: `skewX(${animSkewAngle}deg)`,
          }}
        >
          <rect
            x={animX}
            y={bounds.y}
            width={animW}
            height={animH}
            fill={isHovered ? '#f8fafc' : 'white'}
            stroke={isSelected ? '#2563eb' : isHovered ? '#6b7280' : '#e5e7eb'}
            strokeWidth={isSelected ? 2 : 1}
            rx={8}
            style={{
              cursor: connectionMode === 'view' ? 'pointer' : 'default',
              opacity: shouldDimContainer ? 0.15 : 1,
              filter: isSelected
                ? 'drop-shadow(0 0 8px rgba(37, 99, 235, 0.4))'
                : isHovered
                  ? 'drop-shadow(0 4px 12px rgba(0, 0, 0, 0.15))'
                  : 'drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1))',
              transition: 'filter 0.2s ease, fill 0.2s ease, stroke 0.2s ease, opacity 0.2s ease',
              // 保持 pointerEvents 以支持悬停检测，点击逻辑在处理器中过滤
              pointerEvents: 'auto',
            }}
            onClick={(e) => {
              e.stopPropagation()
              if (connectionMode !== 'view') return
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
            onDoubleClick={(e) => {
              e.stopPropagation()
              onNodeDoubleClick?.(containerNode.id, containerNode.type)
            }}
          />
        </g>

        {/* 跨层级边渲染 - 简化版本，完整版本在主文件中 */}
        {/* 容器内部内容渲染 - 简化版本 */}
        <g
          style={{
            transition: expandingContainer || collapsingContainer
              ? 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)'
              : 'all 0.3s ease',
            transformOrigin: `${animX + animW / 2}px ${bounds.y + animH / 2}px`,
            transform: `skewX(${animSkewAngle}deg)`,
          }}
        >
          {/* 容器标签 */}
          <text
            x={animX + 10}
            y={bounds.y + animH - 5}
            fontSize={10}
            fill="#666"
            style={{ pointerEvents: 'none', opacity: shouldDimContainer ? 0.15 : 1, transition: 'opacity 0.2s ease' }}
          >
            {containerNode.label}
          </text>

          {/* 嵌套SVG渲染单层级内容 */}
          {containerNode.singleLevelData && (() => {
            const { nodes: slNodes, edges: slEdges, viewBox, switchPanelWidth } = containerNode.singleLevelData
            const slSwitchPanelWidth = switchPanelWidth ?? 0
            const slSwitchNodes = slNodes.filter(n => n.isSwitch && n.inSwitchPanel)
            const labelHeight = 10
            const sidePadding = 10
            const topPadding = 10
            const svgWidth = animW - sidePadding * 2
            const svgHeight = animH - labelHeight - topPadding
            const svgX = animX + sidePadding
            const svgY = bounds.y + topPadding

            const deviceCount = slNodes.filter(n => !n.isSwitch).length
            const slNodeScale = deviceCount > 20 ? 1.2 : deviceCount > 10 ? 1.2 : 1.4

            return (
              <svg
                x={svgX}
                y={svgY}
                width={svgWidth}
                height={svgHeight}
                viewBox={`0 0 ${viewBox.width} ${viewBox.height}`}
                preserveAspectRatio="xMidYMid meet"
                overflow="hidden"
                style={{ pointerEvents: 'all' }}
                onMouseMove={handleDragMove}
                onMouseUp={handleDragEnd}
                onMouseLeave={handleDragEnd}
              >
                {/* 透明背景 - 使用 rgba 确保能响应点击事件 */}
                <rect
                  x={0}
                  y={0}
                  width={viewBox.width}
                  height={viewBox.height}
                  fill="rgba(0,0,0,0)"
                  style={{ pointerEvents: connectionMode === 'view' ? 'all' : 'none', cursor: connectionMode !== 'view' ? 'crosshair' : 'pointer' }}
                  onClick={(e) => {
                    e.stopPropagation()
                    // 只在 view 模式下响应背景点击
                    if (onNodeClick) {
                      onNodeClick({
                        id: containerNode.id,
                        label: containerNode.label,
                        type: containerNode.type,
                        subType: containerNode.type,
                        connections: [],
                      })
                    }
                    onLinkClick?.(null)
                  }}
                />

                {/* Switch面板 */}
                {slSwitchPanelWidth > 0 && slSwitchNodes.length > 0 && (
                  <g className="container-switch-panel" style={{ opacity: shouldDimContainer ? 0.15 : 1, transition: 'opacity 0.2s ease' }}>
                    {/* Switch之间的连线 */}
                    {(() => {
                      const switchIds = new Set(slSwitchNodes.map(n => n.id))
                      const switchInternalEdges = slEdges.filter(e =>
                        switchIds.has(e.source) && switchIds.has(e.target)
                      )
                      return switchInternalEdges.map((edge, idx) => {
                        const sourceNode = slSwitchNodes.find(n => n.id === edge.source)
                        const targetNode = slSwitchNodes.find(n => n.id === edge.target)
                        if (!sourceNode || !targetNode) return null

                        const upperNode = sourceNode.y < targetNode.y ? sourceNode : targetNode
                        const lowerNode = sourceNode.y < targetNode.y ? targetNode : sourceNode

                        const midY = (upperNode.y + lowerNode.y) / 2
                        const pathD = `M ${upperNode.x} ${upperNode.y + 10}
                                       L ${upperNode.x} ${midY}
                                       L ${lowerNode.x} ${midY}
                                       L ${lowerNode.x} ${lowerNode.y - 10}`

                        return (
                          <path
                            key={`sl-sw-edge-${idx}`}
                            d={pathD}
                            fill="none"
                            stroke="#1890ff"
                            strokeWidth={1.5}
                            strokeOpacity={0.6}
                          />
                        )
                      })
                    })()}
                    {/* Switch节点 */}
                    {slSwitchNodes.map(swNode => renderNode(swNode, {
                      keyPrefix: 'sl-sw',
                      scale: 1.8,
                      isSelected: selectedNodeId === swNode.id,
                      useMultiLevelConfig: true,
                      onClick: () => {
                        if (connectionMode === 'view' && onNodeClick) {
                          const swConnections = slEdges
                            .filter(edge => edge.source === swNode.id || edge.target === swNode.id)
                            .map(edge => {
                              const otherId = edge.source === swNode.id ? edge.target : edge.source
                              const otherNode = slNodes.find(n => n.id === otherId)
                              return { id: otherId, label: otherNode?.label || otherId, bandwidth: edge.bandwidth, latency: edge.latency }
                            })
                          onNodeClick({
                            id: swNode.id,
                            label: swNode.label,
                            type: swNode.type,
                            subType: swNode.subType,
                            connections: swConnections,
                          })
                        }
                      }
                    }))}
                  </g>
                )}

                {/* 边 */}
                {slEdges.map((edge, i) => {
                  const sourceNode = slNodes.find(n => n.id === edge.source)
                  const targetNode = slNodes.find(n => n.id === edge.target)
                  if (!sourceNode || !targetNode) return null
                  if (sourceNode.inSwitchPanel && targetNode.inSwitchPanel) return null

                  const edgeId = `${edge.source}-${edge.target}`
                  const isLinkSelected = selectedLinkId === edgeId || selectedLinkId === `${edge.target}-${edge.source}`
                  // 如果边的两端节点都有跨层级连线，则边保持不透明
                  const sourceConnected = connectedNodeIds.has(edge.source)
                  const targetConnected = connectedNodeIds.has(edge.target)
                  const edgeOpacity = shouldDimContainer && !sourceConnected && !targetConnected ? 0.15 : 0.6

                  return (
                    <line
                      key={`sl-edge-${i}`}
                      x1={sourceNode.x}
                      y1={sourceNode.y}
                      x2={targetNode.x}
                      y2={targetNode.y}
                      stroke={isLinkSelected ? '#2563eb' : '#b0b0b0'}
                      strokeWidth={isLinkSelected ? 2 : 1}
                      strokeOpacity={edgeOpacity}
                      style={{ cursor: 'pointer', transition: 'stroke-opacity 0.2s ease' }}
                      onClick={(e) => {
                        e.stopPropagation()
                        if (connectionMode !== 'view') return
                        onLinkClick?.({
                          id: edgeId,
                          sourceId: edge.source,
                          sourceLabel: sourceNode.label,
                          sourceType: sourceNode.type,
                          targetId: edge.target,
                          targetLabel: targetNode.label,
                          targetType: targetNode.type,
                          bandwidth: edge.bandwidth,
                          latency: edge.latency,
                          isManual: false
                        })
                      }}
                    />
                  )
                })}

                {/* 设备节点 */}
                {slNodes.filter(n => !n.isSwitch && !n.inSwitchPanel).map(node => {
                  const isNodeSelected = selectedNodeId === node.id
                  const isSourceSelected = selectedNodes.has(node.id)
                  const isTargetSelected = targetNodes.has(node.id)
                  // 如果容器需要变透明，但该节点有跨层级连线，则节点保持不透明
                  const isConnectedNode = connectedNodeIds.has(node.id)
                  const nodeOpacity = shouldDimContainer && !isConnectedNode ? 0.15 : 1
                  // 当前层级是否是活跃层级（可以选择节点）
                  const isActiveLayer = !shouldDimContainer
                  // 在连接模式下，只有活跃层级的节点才能响应点击
                  const canSelect = connectionMode === 'view' || isActiveLayer

                  return (
                    <g
                      key={node.id}
                      transform={`translate(${node.x}, ${node.y}) scale(${slNodeScale})`}
                      style={{
                        cursor: !canSelect ? 'default' : connectionMode !== 'view' ? 'crosshair' : 'pointer',
                        opacity: nodeOpacity,
                        filter: isNodeSelected
                          ? 'drop-shadow(0 0 8px rgba(37, 99, 235, 0.6)) drop-shadow(0 0 16px rgba(37, 99, 235, 0.3))'
                          : 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))',
                        transition: 'filter 0.15s ease, opacity 0.2s ease',
                        pointerEvents: canSelect ? 'all' : 'none',
                      }}
                      onMouseDown={(e) => {
                        e.stopPropagation()
                        e.preventDefault()
                        // 非活跃层级不处理选择
                        if (!isActiveLayer) return
                        // 在连接模式下使用 mousedown 处理选择（因为 click 被嵌套 SVG 的事件干扰）
                        if (connectionMode === 'select_source' || connectionMode === 'select' || connectionMode === 'connect') {
                          const currentSet = new Set(selectedNodes)
                          if (currentSet.has(node.id)) {
                            currentSet.delete(node.id)
                          } else {
                            currentSet.add(node.id)
                          }
                          onSelectedNodesChange?.(currentSet)
                        } else if (connectionMode === 'select_target') {
                          const currentSet = new Set(targetNodes)
                          if (currentSet.has(node.id)) {
                            currentSet.delete(node.id)
                          } else {
                            currentSet.add(node.id)
                          }
                          onTargetNodesChange?.(currentSet)
                        }
                      }}
                      onClick={(e) => {
                        e.stopPropagation()
                        // view 模式下使用 click 显示节点信息
                        if (connectionMode === 'view' && onNodeClick) {
                          const nodeConnections = slEdges
                            .filter(edge => edge.source === node.id || edge.target === node.id)
                            .map(edge => {
                              const otherId = edge.source === node.id ? edge.target : edge.source
                              const otherNode = slNodes.find(n => n.id === otherId)
                              return { id: otherId, label: otherNode?.label || otherId, bandwidth: edge.bandwidth, latency: edge.latency }
                            })
                          onNodeClick({
                            id: node.id,
                            label: node.label,
                            type: node.type,
                            subType: node.subType,
                            connections: nodeConnections,
                          })
                        }
                      }}
                      onDoubleClick={(e) => {
                        e.stopPropagation()
                        // 双击容器内节点进入下一层
                        onNodeDoubleClick?.(node.id, node.type)
                      }}
                    >
                      {/* 选中指示器（连接模式）- 发光矩形效果 */}
                      {(isSourceSelected || isTargetSelected) && (() => {
                        // 根据节点类型获取尺寸（多层级视图使用的尺寸）
                        const nodeType = node.isSwitch ? 'switch' : node.type.toLowerCase()
                        const sizeMap: Record<string, { w: number; h: number }> = {
                          switch: { w: 61, h: 24 },
                          pod: { w: 56, h: 32 },
                          rack: { w: 36, h: 56 },
                          board: { w: 64, h: 36 },
                          chip: { w: 40, h: 40 },
                          default: { w: 50, h: 36 },
                        }
                        const size = sizeMap[nodeType] || sizeMap.default
                        const padding = 6
                        const color = isSourceSelected ? '#2563eb' : '#10b981'
                        const glowColor = isSourceSelected ? 'rgba(37, 99, 235, 0.5)' : 'rgba(16, 185, 129, 0.5)'
                        return (
                          <rect
                            x={-(size.w / 2 + padding)}
                            y={-(size.h / 2 + padding)}
                            width={size.w + padding * 2}
                            height={size.h + padding * 2}
                            rx={6}
                            ry={6}
                            fill="none"
                            stroke={color}
                            strokeWidth={2.5}
                            style={{
                              filter: `drop-shadow(0 0 8px ${color}) drop-shadow(0 0 16px ${glowColor})`,
                            }}
                          />
                        )
                      })()}
                      {renderNode({ ...node, x: 0, y: 0 }, {
                        keyPrefix: 'sl-node',
                        scale: 1,
                        isSelected: isNodeSelected,
                        useMultiLevelConfig: true,
                        onClick: () => {}
                      })}
                    </g>
                  )
                })}
              </svg>
            )
          })()}
        </g>
      </g>
    )
  })

  // 上层Switch节点（容器外的Switch）
  const upperSwitchNodes = displayNodes.filter(n => n.isSwitch && n.inSwitchPanel)

  // 渲染上层Switch面板
  const renderUpperSwitchPanel = () => {
    if (upperSwitchNodes.length === 0) return null

    return (
      <g className="upper-switch-panel">
        {/* Switch之间的连线 */}
        {(() => {
          const switchIds = new Set(upperSwitchNodes.map(n => n.id))
          const switchInternalEdges = edges.filter(e =>
            switchIds.has(e.source) && switchIds.has(e.target)
          )
          return switchInternalEdges.map((edge, idx) => {
            const sourceNode = upperSwitchNodes.find(n => n.id === edge.source)
            const targetNode = upperSwitchNodes.find(n => n.id === edge.target)
            if (!sourceNode || !targetNode) return null

            const upperNode = sourceNode.y < targetNode.y ? sourceNode : targetNode
            const lowerNode = sourceNode.y < targetNode.y ? targetNode : sourceNode

            const midY = (upperNode.y + lowerNode.y) / 2
            const pathD = `M ${upperNode.x} ${upperNode.y + 12}
                           L ${upperNode.x} ${midY}
                           L ${lowerNode.x} ${midY}
                           L ${lowerNode.x} ${lowerNode.y - 12}`

            return (
              <path
                key={`upper-sw-edge-${idx}`}
                d={pathD}
                fill="none"
                stroke="#1890ff"
                strokeWidth={1.5}
                strokeOpacity={0.6}
              />
            )
          })
        })()}
        {/* Switch节点 */}
        {upperSwitchNodes.map(swNode => renderNode(swNode, {
          keyPrefix: 'upper-sw',
          scale: 1,
          isSelected: selectedNodeId === swNode.id,
          useMultiLevelConfig: true,
          onClick: () => {
            if (connectionMode === 'view' && onNodeClick) {
              const swConnections = edges
                .filter(edge => edge.source === swNode.id || edge.target === swNode.id)
                .map(edge => {
                  const otherId = edge.source === swNode.id ? edge.target : edge.source
                  const otherNode = displayNodes.find(n => n.id === otherId)
                  return { id: otherId, label: otherNode?.label || otherId, bandwidth: edge.bandwidth, latency: edge.latency }
                })
              onNodeClick({
                id: swNode.id,
                label: swNode.label,
                type: swNode.type,
                subType: swNode.subType,
                connections: swConnections,
              })
            }
          }
        }))}
      </g>
    )
  }

  return (
    <>
      {/* 上层Switch面板 */}
      {renderUpperSwitchPanel()}
      {containersRendered}
      {/* 跨容器手动连接在最上层 */}
      {renderManualConnectionsOverlay()}
    </>
  )
}
