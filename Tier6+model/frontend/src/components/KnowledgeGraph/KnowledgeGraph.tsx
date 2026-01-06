/**
 * 知识网络可视化组件
 * 使用D3力导向布局展示名词及其关系
 */
import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { Input, Tag, Button, Spin, Typography } from 'antd'
import { SearchOutlined, ReloadOutlined } from '@ant-design/icons'
import * as d3Force from 'd3-force'
import {
  KnowledgeRelation,
  KnowledgeGraphData,
  ForceKnowledgeNode,
  KnowledgeCategory,
  CATEGORY_COLORS,
  CATEGORY_NAMES,
  RELATION_STYLES,
} from './types'
import { useWorkbench } from '../../contexts/WorkbenchContext'
import knowledgeData from '../../data/knowledge-graph.json'

const { Text } = Typography

// 力导向布局参数
const FORCE_CONFIG = {
  chargeStrength: -400,
  linkDistance: 100,
  linkStrength: 0.3,
  collisionRadius: 35,
  centerStrength: 0.02,
  alphaDecay: 0.02,
  velocityDecay: 0.4,
}

// 节点半径
const NODE_RADIUS = 24

export const KnowledgeGraph: React.FC = () => {
  const { ui } = useWorkbench()
  const {
    knowledgeSelectedNode: selectedNode,
    knowledgeVisibleCategories: visibleCategories,
    knowledgeNodes,
    knowledgeInitialized,
    setKnowledgeSelectedNode: setSelectedNode,
    setKnowledgeVisibleCategories: setVisibleCategories,
    setKnowledgeNodes,
    setKnowledgeInitialized,
    resetKnowledgeCategories,
  } = ui

  // 本地状态（仅用于渲染触发）
  const [, forceUpdate] = useState(0)
  const [relations, setRelations] = useState<KnowledgeRelation[]>([])
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [loading, setLoading] = useState(true)

  // Refs
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const simulationRef = useRef<d3Force.Simulation<ForceKnowledgeNode, d3Force.SimulationLinkDatum<ForceKnowledgeNode>> | null>(null)

  // 视口状态
  const [viewBox, setViewBox] = useState({ x: 0, y: 0, width: 1200, height: 800 })
  const [isPanning, setIsPanning] = useState(false)
  const panStartRef = useRef({ x: 0, y: 0, viewX: 0, viewY: 0 })

  // 加载数据（只在首次加载时初始化，避免切换视图重复计算）
  useEffect(() => {
    if (knowledgeInitialized) {
      setLoading(false)
      return
    }
    const data = knowledgeData as KnowledgeGraphData

    // 初始化节点位置
    const centerX = viewBox.width / 2
    const centerY = viewBox.height / 2
    const initialNodes: ForceKnowledgeNode[] = data.nodes.map((node) => ({
      ...node,
      category: node.category as KnowledgeCategory,
      x: centerX + (Math.random() - 0.5) * 200,
      y: centerY + (Math.random() - 0.5) * 200,
    }))

    setKnowledgeNodes(initialNodes)
    setRelations(data.relations)
    setKnowledgeInitialized(true)
    setLoading(false)
  }, [knowledgeInitialized, viewBox.width, viewBox.height, setKnowledgeNodes, setKnowledgeInitialized])

  // 初始化力导向模拟（只在节点/关系/过滤变化时重新计算，不受缩放影响）
  useEffect(() => {
    if (knowledgeNodes.length === 0) return

    // 过滤可见节点
    const visibleNodeIds = new Set(
      knowledgeNodes.filter(n => visibleCategories.has(n.category)).map(n => n.id)
    )
    const visibleRelations = relations.filter(
      r => visibleNodeIds.has(r.source) && visibleNodeIds.has(r.target)
    )

    // 使用固定的中心点（初始viewBox尺寸）
    const centerX = 600
    const centerY = 400

    // 创建力模拟
    const simulation = d3Force.forceSimulation<ForceKnowledgeNode>(knowledgeNodes)
      .force('charge', d3Force.forceManyBody<ForceKnowledgeNode>()
        .strength(FORCE_CONFIG.chargeStrength)
        .distanceMax(400)
      )
      .force('link', d3Force.forceLink<ForceKnowledgeNode, d3Force.SimulationLinkDatum<ForceKnowledgeNode>>(
        visibleRelations.map(r => ({ source: r.source, target: r.target }))
      )
        .id(d => d.id)
        .distance(FORCE_CONFIG.linkDistance)
        .strength(FORCE_CONFIG.linkStrength)
      )
      .force('center', d3Force.forceCenter(centerX, centerY)
        .strength(FORCE_CONFIG.centerStrength)
      )
      .force('collision', d3Force.forceCollide<ForceKnowledgeNode>()
        .radius(FORCE_CONFIG.collisionRadius)
        .strength(0.8)
      )
      .alphaDecay(FORCE_CONFIG.alphaDecay)
      .velocityDecay(FORCE_CONFIG.velocityDecay)

    simulation.on('tick', () => {
      forceUpdate(n => n + 1)
    })

    simulationRef.current = simulation

    return () => {
      simulation.stop()
    }
  }, [knowledgeNodes.length, relations, visibleCategories])

  // 获取相邻节点
  const getAdjacentNodeIds = useCallback((nodeId: string): Set<string> => {
    const adjacent = new Set<string>()
    relations.forEach(r => {
      if (r.source === nodeId) adjacent.add(r.target)
      if (r.target === nodeId) adjacent.add(r.source)
    })
    return adjacent
  }, [relations])

  // 搜索匹配
  const matchedNodeIds = useMemo(() => {
    if (!searchQuery.trim()) return null
    const query = searchQuery.toLowerCase()
    const matched = new Set<string>()
    knowledgeNodes.forEach(node => {
      if (
        node.name.toLowerCase().includes(query) ||
        node.fullName?.toLowerCase().includes(query) ||
        node.definition.toLowerCase().includes(query) ||
        node.aliases?.some(a => a.toLowerCase().includes(query))
      ) {
        matched.add(node.id)
      }
    })
    return matched
  }, [searchQuery, knowledgeNodes])

  // 节点点击
  const handleNodeClick = useCallback((node: ForceKnowledgeNode) => {
    setSelectedNode(selectedNode?.id === node.id ? null : node)
  }, [selectedNode, setSelectedNode])

  // 画布拖拽
  const handlePanStart = useCallback((e: React.MouseEvent) => {
    setIsPanning(true)
    panStartRef.current = {
      x: e.clientX,
      y: e.clientY,
      viewX: viewBox.x,
      viewY: viewBox.y,
    }
  }, [viewBox])

  const handlePanMove = useCallback((e: React.MouseEvent) => {
    if (!isPanning) return
    const dx = (e.clientX - panStartRef.current.x) * (viewBox.width / (containerRef.current?.clientWidth || 1))
    const dy = (e.clientY - panStartRef.current.y) * (viewBox.height / (containerRef.current?.clientHeight || 1))
    setViewBox(prev => ({
      ...prev,
      x: panStartRef.current.viewX - dx,
      y: panStartRef.current.viewY - dy,
    }))
  }, [isPanning, viewBox.width, viewBox.height])

  const handlePanEnd = useCallback(() => {
    setIsPanning(false)
  }, [])

  // 滚轮缩放
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    const scaleFactor = e.deltaY > 0 ? 1.1 : 0.9
    const rect = containerRef.current?.getBoundingClientRect()
    if (!rect) return

    const mouseX = (e.clientX - rect.left) / rect.width
    const mouseY = (e.clientY - rect.top) / rect.height

    setViewBox(prev => {
      const newWidth = prev.width * scaleFactor
      const newHeight = prev.height * scaleFactor
      const newX = prev.x + (prev.width - newWidth) * mouseX
      const newY = prev.y + (prev.height - newHeight) * mouseY
      return { x: newX, y: newY, width: newWidth, height: newHeight }
    })
  }, [])

  // 分类过滤切换 (普通点击切换，Ctrl+点击只显示该分类)
  const handleCategoryClick = useCallback((category: KnowledgeCategory, ctrlKey: boolean) => {
    if (ctrlKey) {
      // Ctrl+点击: 只显示该分类
      setVisibleCategories(new Set([category]))
    } else {
      // 普通点击: 切换该分类
      const next = new Set(visibleCategories)
      if (next.has(category)) {
        next.delete(category)
      } else {
        next.add(category)
      }
      setVisibleCategories(next)
    }
  }, [visibleCategories, setVisibleCategories])

  // 渲染边
  const renderEdges = () => {
    const nodeMap = new Map(knowledgeNodes.map(n => [n.id, n]))
    return relations.map((rel, i) => {
      const source = nodeMap.get(rel.source)
      const target = nodeMap.get(rel.target)
      if (!source || !target) return null
      if (!visibleCategories.has(source.category) || !visibleCategories.has(target.category)) return null

      const style = RELATION_STYLES[rel.type] || RELATION_STYLES.related_to
      const isHighlighted = selectedNode && (rel.source === selectedNode.id || rel.target === selectedNode.id)
      const isFiltered = matchedNodeIds && (!matchedNodeIds.has(rel.source) || !matchedNodeIds.has(rel.target))

      return (
        <line
          key={`edge-${i}`}
          x1={source.x}
          y1={source.y}
          x2={target.x}
          y2={target.y}
          stroke={style.stroke}
          strokeWidth={isHighlighted ? 2.5 : 1.5}
          opacity={isFiltered ? 0.15 : isHighlighted ? 1 : 0.6}
        />
      )
    })
  }

  // 渲染节点
  const renderNodes = () => {
    return knowledgeNodes.map(node => {
      if (!visibleCategories.has(node.category)) return null

      const color = CATEGORY_COLORS[node.category]
      const isSelected = selectedNode?.id === node.id
      const isHovered = hoveredNode === node.id
      const isAdjacent = selectedNode ? getAdjacentNodeIds(selectedNode.id).has(node.id) : false
      const isMatched = matchedNodeIds ? matchedNodeIds.has(node.id) : true
      const isFiltered = matchedNodeIds && !isMatched

      return (
        <g
          key={node.id}
          transform={`translate(${node.x}, ${node.y})`}
          style={{ cursor: 'pointer' }}
          onClick={() => handleNodeClick(node)}
          onMouseEnter={() => setHoveredNode(node.id)}
          onMouseLeave={() => setHoveredNode(null)}
        >
          {/* 选中/高亮外圈 */}
          {(isSelected || isAdjacent) && (
            <circle
              r={NODE_RADIUS + 6}
              fill="none"
              stroke={color}
              strokeWidth={3}
              strokeOpacity={isSelected ? 0.6 : 0.3}
            />
          )}

          {/* 主圆形 */}
          <circle
            r={NODE_RADIUS}
            fill={isFiltered ? `${color}40` : color}
            stroke="#fff"
            strokeWidth={2}
            style={{
              filter: isHovered ? `drop-shadow(0 0 8px ${color})` : 'none',
              opacity: isFiltered ? 0.4 : 1,
              transition: 'filter 0.2s, opacity 0.2s',
            }}
          />

          {/* 名称标签 */}
          <text
            y={4}
            textAnchor="middle"
            fill="#fff"
            fontSize={11}
            fontWeight={600}
            style={{ pointerEvents: 'none', userSelect: 'none' }}
          >
            {node.name.length > 5 ? node.name.slice(0, 4) + '..' : node.name}
          </text>

          {/* Hover Tooltip */}
          {isHovered && (
            <g transform="translate(0, -40)">
              <rect
                x={-60}
                y={-12}
                width={120}
                height={24}
                rx={4}
                fill="rgba(0,0,0,0.85)"
              />
              <text
                textAnchor="middle"
                fill="#fff"
                fontSize={12}
                y={4}
              >
                {node.fullName || node.name}
              </text>
            </g>
          )}
        </g>
      )
    })
  }

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <Spin size="large" />
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#fafafa' }}>
      {/* 工具栏 */}
      <div style={{ padding: '12px 16px', borderBottom: '1px solid #e5e5e5', background: '#fff', display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
        {/* 搜索框 */}
        <Input
          placeholder="搜索名词..."
          prefix={<SearchOutlined style={{ color: '#666' }} />}
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          allowClear
          style={{ width: 200 }}
        />

        {/* 分类过滤 (Ctrl+点击只显示该分类) */}
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', flex: 1 }}>
          {Object.entries(CATEGORY_NAMES).map(([key, name]) => {
            const category = key as KnowledgeCategory
            const isActive = visibleCategories.has(category)
            const count = knowledgeNodes.filter(n => n.category === category).length
            if (count === 0) return null
            return (
              <Tag
                key={category}
                color={isActive ? CATEGORY_COLORS[category] : undefined}
                style={{
                  cursor: 'pointer',
                  opacity: isActive ? 1 : 0.5,
                  borderColor: CATEGORY_COLORS[category],
                }}
                onClick={(e) => handleCategoryClick(category, e.ctrlKey || e.metaKey)}
              >
                {name} ({count})
              </Tag>
            )
          })}
        </div>

        {/* 全部显示按钮 */}
        {visibleCategories.size < 8 && (
          <Button
            size="small"
            icon={<ReloadOutlined />}
            onClick={resetKnowledgeCategories}
          >
            全部显示
          </Button>
        )}

        {/* 统计信息 */}
        <Text type="secondary" style={{ fontSize: 12 }}>
          {knowledgeNodes.filter(n => visibleCategories.has(n.category)).length} 个节点 · {relations.length} 条关系
        </Text>
      </div>

      {/* 画布 */}
      <div
        ref={containerRef}
        style={{ flex: 1, overflow: 'hidden', position: 'relative' }}
        onMouseDown={handlePanStart}
        onMouseMove={handlePanMove}
        onMouseUp={handlePanEnd}
        onMouseLeave={handlePanEnd}
        onWheel={handleWheel}
      >
        <svg
          ref={svgRef}
          width="100%"
          height="100%"
          viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`}
          style={{ cursor: isPanning ? 'grabbing' : 'grab' }}
        >
          {/* 背景 */}
          <rect
            x={viewBox.x - 1000}
            y={viewBox.y - 1000}
            width={viewBox.width + 2000}
            height={viewBox.height + 2000}
            fill="#fafafa"
          />

          {/* 边 */}
          <g>{renderEdges()}</g>

          {/* 节点 */}
          <g>{renderNodes()}</g>
        </svg>
      </div>
    </div>
  )
}
