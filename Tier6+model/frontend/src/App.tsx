import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react'
import { Layout, Typography, Spin, message, Segmented, Card, Descriptions, Tag, Collapse } from 'antd'
import { Scene3D } from './components/Scene3D'
import { ConfigPanel } from './components/ConfigPanel'
import { TopologyGraph, NodeDetail, LinkDetail } from './components/TopologyGraph'
import { HierarchicalTopology, ManualConnectionConfig, ManualConnection, ConnectionMode, HierarchyLevel, LayoutType, MultiLevelViewOptions, TrafficConfigItem, TrafficAnalysisResult } from './types'
import { getTopology, generateTopology, getLevelConnectionDefaults } from './api/topology'
import { useViewNavigation } from './hooks/useViewNavigation'

const { Header, Sider, Content } = Layout
const { Title } = Typography

// localStorage缓存key（与ConfigPanel保持一致）
const CONFIG_CACHE_KEY = 'tier6_topology_config_cache'
const SIDER_WIDTH_KEY = 'tier6_sider_width_cache'

// 默认和限制值
const DEFAULT_SIDER_WIDTH = 400
const MIN_SIDER_WIDTH = 280
const MAX_SIDER_WIDTH = 600

const App: React.FC = () => {
  const [topology, setTopology] = useState<HierarchicalTopology | null>(null)
  const [loading, setLoading] = useState(true)  // 初始加载状态

  // 视图模式：3d 或 topology
  const [viewMode, setViewMode] = useState<'3d' | 'topology'>('topology')

  // 侧边栏宽度（从localStorage加载）
  const [siderWidth, setSiderWidth] = useState(() => {
    const cached = localStorage.getItem(SIDER_WIDTH_KEY)
    return cached ? Math.max(MIN_SIDER_WIDTH, Math.min(MAX_SIDER_WIDTH, parseInt(cached, 10))) : DEFAULT_SIDER_WIDTH
  })

  // 拖拽状态
  const [isDragging, setIsDragging] = useState(false)
  const dragStartX = useRef(0)
  const dragStartWidth = useRef(0)

  // 视图导航状态
  const navigation = useViewNavigation(topology)

  // 选中的节点详情
  const [selectedNode, setSelectedNode] = useState<NodeDetail | null>(null)

  // 选中的连接详情
  const [selectedLink, setSelectedLink] = useState<LinkDetail | null>(null)

  // 聚焦的层级配置（点击容器时切换）
  const [focusedLevel, setFocusedLevel] = useState<'datacenter' | 'pod' | 'rack' | 'board' | null>(null)

  // 各层级连接的默认参数（从后端加载，初始为空）
  const [_levelConnectionDefaults, _setLevelConnectionDefaults] = useState<{
    datacenter: { bandwidth: number; latency: number }
    pod: { bandwidth: number; latency: number }
    rack: { bandwidth: number; latency: number }
    board: { bandwidth: number; latency: number }
  } | null>(null)

  // 手动连接状态 (从缓存加载)
  const [manualConnectionConfig, setManualConnectionConfig] = useState<ManualConnectionConfig>(() => {
    try {
      const cachedStr = localStorage.getItem(CONFIG_CACHE_KEY)
      if (cachedStr) {
        const cached = JSON.parse(cachedStr)
        if (cached.manualConnectionConfig) {
          return cached.manualConnectionConfig
        }
      }
    } catch (error) {
      console.error('加载缓存手动连接配置失败:', error)
    }
    return { enabled: false, mode: 'append', connections: [] }
  })

  // 从后端加载默认配置，并设置到 manualConnectionConfig
  useEffect(() => {
    getLevelConnectionDefaults().then((defaults) => {
      _setLevelConnectionDefaults(defaults)
      // 设置 manualConnectionConfig 的 level_defaults（保留用户自定义的值）
      setManualConnectionConfig((prev) => ({
        ...prev,
        level_defaults: {
          ...defaults,
          ...prev.level_defaults,
        },
      }))
    }).catch((error) => {
      console.error('获取层级连接默认配置失败:', error)
    })
  }, [])
  const [connectionMode, setConnectionMode] = useState<ConnectionMode>('view')
  const [selectedNodes, setSelectedNodes] = useState<Set<string>>(new Set())  // 源节点集合
  const [targetNodes, setTargetNodes] = useState<Set<string>>(new Set())  // 目标节点集合
  const [sourceNode, setSourceNode] = useState<string | null>(null)  // 保留兼容
  const [layoutType, setLayoutType] = useState<LayoutType>('auto')  // 布局类型

  // 多层级视图选项
  const [multiLevelOptions, setMultiLevelOptions] = useState<MultiLevelViewOptions>({
    enabled: false,
    levelPair: 'pod_rack',
    expandedContainers: new Set(),
  })

  // LLM流量分析状态 (多配置)
  const [trafficConfigs, setTrafficConfigs] = useState<TrafficConfigItem[]>([])
  const [trafficAnalysisResult, setTrafficAnalysisResult] = useState<TrafficAnalysisResult | null>(null)

  // 加载拓扑数据（优先使用缓存配置生成）
  const loadTopology = useCallback(async () => {
    setLoading(true)
    try {
      // 检查是否有缓存配置
      const cachedStr = localStorage.getItem(CONFIG_CACHE_KEY)
      if (cachedStr) {
        const cached = JSON.parse(cachedStr)
        // 使用缓存配置生成拓扑
        const data = await generateTopology({
          pod_count: cached.podCount,
          racks_per_pod: cached.racksPerPod,
          board_configs: cached.boardConfigs,
          switch_config: cached.switchConfig,
          manual_connections: cached.manualConnectionConfig,
        })
        setTopology(data)
      } else {
        // 没有缓存，使用默认配置
        const data = await getTopology()
        setTopology(data)
      }
    } catch (error) {
      console.error('加载拓扑失败:', error)
      message.error('加载拓扑数据失败')
    } finally {
      setLoading(false)
    }
  }, [])

  // 重新生成拓扑
  const handleGenerate = useCallback(async (config: {
    pod_count: number
    racks_per_pod: number
    board_configs: {
      u1: { count: number; chips: { npu: number; cpu: number } }
      u2: { count: number; chips: { npu: number; cpu: number } }
      u4: { count: number; chips: { npu: number; cpu: number } }
    }
    rack_config?: {
      total_u: number
      boards: Array<{
        id: string
        name: string
        u_height: number
        count: number
        chips: Array<{ name: string; count: number }>
      }>
    }
    switch_config?: any
    manual_connections?: ManualConnectionConfig
  }) => {
    try {
      const data = await generateTopology(config)
      setTopology(data)
    } catch (error) {
      console.error('生成拓扑失败:', error)
      message.error('生成拓扑失败')
    }
  }, [])

  // 初始加载
  useEffect(() => {
    loadTopology()
  }, [loadTopology])

  // 全局键盘快捷键处理（在3D和拓扑视图都生效）
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // 如果正在输入框中则忽略
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return
      }

      // Esc / Backspace - 返回上一级
      if (e.key === 'Escape' || e.key === 'Backspace') {
        e.preventDefault()
        if (navigation.canGoBack) {
          navigation.navigateBack()
        }
        return
      }

      // 左方向键 - 历史后退
      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        if (navigation.canGoHistoryBack) {
          navigation.navigateHistoryBack()
        }
        return
      }

      // 右方向键 - 历史前进
      if (e.key === 'ArrowRight') {
        e.preventDefault()
        if (navigation.canGoHistoryForward) {
          navigation.navigateHistoryForward()
        }
        return
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [navigation])

  // 手动连接处理函数
  const handleManualConnectionConfigChange = useCallback((config: ManualConnectionConfig) => {
    setManualConnectionConfig(config)
    // 禁用时重置编辑状态
    if (!config.enabled) {
      setConnectionMode('view')
      setSelectedNodes(new Set())
      setTargetNodes(new Set())
      setSourceNode(null)
    }
  }, [])

  const handleConnectionModeChange = useCallback((mode: ConnectionMode) => {
    setConnectionMode(mode)
    if (mode === 'view') {
      setSelectedNodes(new Set())
      setTargetNodes(new Set())
      setSourceNode(null)
    } else if (mode === 'select_source') {
      setTargetNodes(new Set())
      setSourceNode(null)
    }
  }, [])

  const handleSelectedNodesChange = useCallback((nodes: Set<string>) => {
    setSelectedNodes(nodes)
  }, [])

  const handleTargetNodesChange = useCallback((nodes: Set<string>) => {
    setTargetNodes(nodes)
  }, [])

  const handleSourceNodeChange = useCallback((nodeId: string | null) => {
    setSourceNode(nodeId)
  }, [])

  const handleManualConnect = useCallback((sourceId: string, targetId: string, level: HierarchyLevel) => {
    // 检查是否已存在手动连接（双向检查）
    const existsManual = manualConnectionConfig.connections.some(c =>
      (c.source === sourceId && c.target === targetId) ||
      (c.source === targetId && c.target === sourceId)
    )
    if (existsManual) {
      message.warning(`手动连接已存在: ${sourceId} ↔ ${targetId}`)
      return
    }

    // 检查是否已存在自动连接（双向检查）
    const existsAuto = topology?.connections?.some(c =>
      (c.source === sourceId && c.target === targetId) ||
      (c.source === targetId && c.target === sourceId)
    )
    if (existsAuto) {
      message.warning(`自动连接已存在: ${sourceId} ↔ ${targetId}`)
      return
    }

    const newConnection: ManualConnection = {
      id: `manual_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      source: sourceId,
      target: targetId,
      hierarchy_level: level,
      created_at: new Date().toISOString(),
    }
    setManualConnectionConfig(prev => ({
      ...prev,
      connections: [...prev.connections, newConnection],
    }))
    message.success(`已添加连接: ${sourceId} ↔ ${targetId}`)
  }, [manualConnectionConfig.connections, topology?.connections])

  // 批量连接：源节点集合 × 目标节点集合
  const handleBatchConnect = useCallback((level: HierarchyLevel) => {
    if (selectedNodes.size === 0 || targetNodes.size === 0) {
      message.warning('请先选择源节点和目标节点')
      return
    }

    let addedCount = 0
    const newConnections: ManualConnection[] = []

    selectedNodes.forEach(sourceId => {
      targetNodes.forEach(targetId => {
        if (sourceId === targetId) return

        // 检查是否已存在
        const existsManual = manualConnectionConfig.connections.some(c =>
          (c.source === sourceId && c.target === targetId) ||
          (c.source === targetId && c.target === sourceId)
        )
        const existsAuto = topology?.connections?.some(c =>
          (c.source === sourceId && c.target === targetId) ||
          (c.source === targetId && c.target === sourceId)
        )
        const existsNew = newConnections.some(c =>
          (c.source === sourceId && c.target === targetId) ||
          (c.source === targetId && c.target === sourceId)
        )

        if (!existsManual && !existsAuto && !existsNew) {
          // 新建连接不设置带宽/延迟，使用层级默认值
          newConnections.push({
            id: `manual_${Date.now()}_${Math.random().toString(36).substr(2, 9)}_${addedCount}`,
            source: sourceId,
            target: targetId,
            hierarchy_level: level,
            created_at: new Date().toISOString(),
          })
          addedCount++
        }
      })
    })

    if (newConnections.length > 0) {
      setManualConnectionConfig(prev => ({
        ...prev,
        connections: [...prev.connections, ...newConnections],
      }))
      message.success(`已添加 ${newConnections.length} 条连接`)
    } else {
      message.warning('所有连接已存在')
    }

    // 清空选中
    setSelectedNodes(new Set())
    setTargetNodes(new Set())
    setConnectionMode('select_source')
  }, [selectedNodes, targetNodes, manualConnectionConfig.connections, topology?.connections])

  const handleDeleteManualConnection = useCallback((connectionId: string) => {
    setManualConnectionConfig(prev => ({
      ...prev,
      connections: prev.connections.filter(c => c.id !== connectionId),
    }))
    message.success('已删除连接')
  }, [])

  // 删除任意连接（自动或手动）
  const handleDeleteConnection = useCallback((source: string, target: string) => {
    // 先检查是否是手动连接
    const manualConn = manualConnectionConfig.connections.find(c =>
      (c.source === source && c.target === target) ||
      (c.source === target && c.target === source)
    )
    if (manualConn) {
      // 删除手动连接
      setManualConnectionConfig(prev => ({
        ...prev,
        connections: prev.connections.filter(c => c.id !== manualConn.id),
      }))
      message.success('已删除手动连接')
    } else {
      // 自动连接：直接从topology.connections中删除
      setTopology(prev => {
        if (!prev) return prev
        return {
          ...prev,
          connections: prev.connections.filter(c =>
            !((c.source === source && c.target === target) ||
              (c.source === target && c.target === source))
          ),
        }
      })
      message.success('已删除连接')
    }
  }, [manualConnectionConfig.connections])

  // 更新连接参数（带宽/延迟）
  const handleUpdateConnectionParams = useCallback((source: string, target: string, bandwidth?: number, latency?: number) => {
    // 先检查是否是手动连接
    const manualConn = manualConnectionConfig.connections.find(c =>
      (c.source === source && c.target === target) ||
      (c.source === target && c.target === source)
    )
    if (manualConn) {
      // 更新手动连接参数
      setManualConnectionConfig(prev => ({
        ...prev,
        connections: prev.connections.map(c =>
          c.id === manualConn.id ? { ...c, bandwidth, latency } : c
        ),
      }))
    } else {
      // 更新自动连接参数
      setTopology(prev => {
        if (!prev) return prev
        return {
          ...prev,
          connections: prev.connections.map(c => {
            if ((c.source === source && c.target === target) ||
                (c.source === target && c.target === source)) {
              return { ...c, bandwidth, latency }
            }
            return c
          }),
        }
      })
    }
  }, [manualConnectionConfig.connections])

  // 拖拽处理
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true)
    dragStartX.current = e.clientX
    dragStartWidth.current = siderWidth
    e.preventDefault()
  }, [siderWidth])

  useEffect(() => {
    if (!isDragging) return

    const handleMouseMove = (e: MouseEvent) => {
      const delta = e.clientX - dragStartX.current
      const newWidth = Math.max(MIN_SIDER_WIDTH, Math.min(MAX_SIDER_WIDTH, dragStartWidth.current + delta))
      setSiderWidth(newWidth)
    }

    const handleMouseUp = () => {
      setIsDragging(false)
      // 保存到localStorage
      localStorage.setItem(SIDER_WIDTH_KEY, siderWidth.toString())
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isDragging, siderWidth])

  // 获取当前视图层级
  const getCurrentLevel = () => {
    if (navigation.currentBoard) return 'board'
    if (navigation.currentRack) return 'rack'
    if (navigation.currentPod) return 'pod'
    return 'datacenter'
  }

  // 跳转到芯片视图（用于流量分析）
  const handleNavigateToChips = useCallback(() => {
    if (!topology || topology.pods.length === 0) return
    const firstPod = topology.pods[0]
    if (firstPod.racks.length === 0) return
    const firstRack = firstPod.racks[0]
    if (firstRack.boards.length === 0) return
    const firstBoard = firstRack.boards[0]
    // 直接导航到 Board 视图（显示 Chip）
    navigation.navigateToBoard(firstPod.id, firstRack.id, firstBoard.id)
    // 切换到拓扑视图
    setViewMode('topology')
    // 关闭多层级视图以便看到热力图
    setMultiLevelOptions(prev => ({ ...prev, enabled: false }))
  }, [topology, navigation])

  // 处理3D视图节点选择，转换为NodeDetail格式
  const handleScene3DNodeSelect = useCallback((
    nodeType: 'pod' | 'rack' | 'board' | 'chip' | 'switch',
    nodeId: string,
    label: string,
    _info: Record<string, string | number>,
    subType?: string
  ) => {
    // 查找与该节点相关的连接
    const connections: { id: string; label: string; bandwidth?: number }[] = []
    if (topology?.connections) {
      topology.connections.forEach(conn => {
        if (conn.source === nodeId) {
          connections.push({ id: conn.target, label: `→ ${conn.target}`, bandwidth: conn.bandwidth })
        } else if (conn.target === nodeId) {
          connections.push({ id: conn.source, label: `← ${conn.source}`, bandwidth: conn.bandwidth })
        }
      })
    }

    // 转换为NodeDetail格式
    const nodeDetail: NodeDetail = {
      id: nodeId,
      label,
      type: nodeType,
      subType,
      connections,
    }

    setSelectedNode(nodeDetail)
  }, [topology])

  // 计算当前视图的连接
  const currentViewConnections = useMemo(() => {
    if (!topology) return []
    const currentLevel = getCurrentLevel()

    if (currentLevel === 'datacenter') {
      // Datacenter层：Pod间连接
      const podIds = new Set(topology.pods.map(p => p.id))
      const dcSwitchIds = new Set(
        (topology.switches || [])
          .filter(s => s.hierarchy_level === 'inter_pod')
          .map(s => s.id)
      )
      return topology.connections.filter(c => {
        const sourceInDc = podIds.has(c.source) || dcSwitchIds.has(c.source)
        const targetInDc = podIds.has(c.target) || dcSwitchIds.has(c.target)
        return sourceInDc && targetInDc
      })
    } else if (currentLevel === 'pod' && navigation.currentPod) {
      // Pod层：Rack间连接
      const rackIds = new Set(navigation.currentPod.racks.map(r => r.id))
      const podSwitchIds = new Set(
        (topology.switches || [])
          .filter(s => s.hierarchy_level === 'inter_rack' && s.parent_id === navigation.currentPod!.id)
          .map(s => s.id)
      )
      return topology.connections.filter(c => {
        const sourceInPod = rackIds.has(c.source) || podSwitchIds.has(c.source)
        const targetInPod = rackIds.has(c.target) || podSwitchIds.has(c.target)
        return sourceInPod && targetInPod
      })
    } else if (currentLevel === 'rack' && navigation.currentRack) {
      // Rack层：Board间连接
      const boardIds = new Set(navigation.currentRack.boards.map(b => b.id))
      const rackSwitchIds = new Set(
        (topology.switches || [])
          .filter(s => s.hierarchy_level === 'inter_board' && s.parent_id === navigation.currentRack!.id)
          .map(s => s.id)
      )
      return topology.connections.filter(c => {
        const sourceInRack = boardIds.has(c.source) || rackSwitchIds.has(c.source)
        const targetInRack = boardIds.has(c.target) || rackSwitchIds.has(c.target)
        return sourceInRack && targetInRack
      })
    } else if (currentLevel === 'board' && navigation.currentBoard) {
      // Board层：Chip间连接
      const chipIds = new Set(navigation.currentBoard.chips.map(c => c.id))
      return topology.connections.filter(c =>
        chipIds.has(c.source) && chipIds.has(c.target)
      )
    }
    return []
  }, [topology, navigation.currentPod, navigation.currentRack, navigation.currentBoard])

  return (
    <Layout style={{ height: '100vh' }}>
      <Header style={{
        background: '#FFFFFF',
        borderBottom: '1px solid #E5E5E5',
        padding: '0 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        height: 56,
        boxShadow: '0 1px 2px rgba(0, 0, 0, 0.04)',
        position: 'relative',
        zIndex: 100,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 32,
            height: 32,
            borderRadius: 8,
            background: 'linear-gradient(135deg, #5E6AD2 0%, #7C3AED 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 2px 6px rgba(94, 106, 210, 0.3)',
          }}>
            <span style={{ color: '#fff', fontSize: 13, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>T6</span>
          </div>
          <Title level={4} style={{ color: '#1A1A1A', margin: 0, fontSize: 16, fontWeight: 600 }}>
            Tier6+ 互联建模
          </Title>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <Segmented
            value={viewMode}
            onChange={(v) => setViewMode(v as '3d' | 'topology')}
            options={[
              { value: '3d', label: '3D视图' },
              { value: 'topology', label: '拓扑图' },
            ]}
          />
          <span style={{ color: '#999999', fontSize: 12 }}>v{__APP_VERSION__}</span>
        </div>
      </Header>

      <Layout>
        <Sider
          width={siderWidth}
          style={{
            background: '#EFEFEF',
            padding: 16,
            overflow: 'auto',
            position: 'relative',
            borderRight: '1px solid #E5E5E5',
            boxShadow: '1px 0 3px rgba(0, 0, 0, 0.04)',
          }}
        >
          <ConfigPanel
            topology={topology}
            onGenerate={handleGenerate}
            loading={loading}
            currentLevel={getCurrentLevel()}
            manualConnectionConfig={manualConnectionConfig}
            onManualConnectionConfigChange={handleManualConnectionConfigChange}
            connectionMode={connectionMode}
            onConnectionModeChange={handleConnectionModeChange}
            selectedNodes={selectedNodes}
            onSelectedNodesChange={handleSelectedNodesChange}
            targetNodes={targetNodes}
            onTargetNodesChange={handleTargetNodesChange}
            onBatchConnect={handleBatchConnect}
            onDeleteManualConnection={handleDeleteManualConnection}
            currentViewConnections={currentViewConnections}
            onDeleteConnection={handleDeleteConnection}
            onUpdateConnectionParams={handleUpdateConnectionParams}
            layoutType={layoutType}
            onLayoutTypeChange={setLayoutType}
            viewMode={viewMode}
            focusedLevel={focusedLevel}
            trafficConfigs={trafficConfigs}
            onTrafficConfigsChange={setTrafficConfigs}
            trafficAnalysisResult={trafficAnalysisResult}
            onTrafficAnalysisResultChange={setTrafficAnalysisResult}
            onNavigateToChips={handleNavigateToChips}
          />

          {/* 节点详情卡片 */}
          {selectedNode && (
            <Card
              title={`节点详情: ${selectedNode.label}`}
              size="small"
              style={{ marginTop: 16 }}
              extra={<a onClick={() => setSelectedNode(null)}>关闭</a>}
            >
              <Descriptions column={1} size="small">
                <Descriptions.Item label="ID">{selectedNode.id}</Descriptions.Item>
                <Descriptions.Item label="类型">
                  <Tag color={selectedNode.type === 'switch' ? 'blue' : 'green'}>
                    {selectedNode.subType?.toUpperCase() || selectedNode.type.toUpperCase()}
                  </Tag>
                </Descriptions.Item>
                {selectedNode.portInfo && (
                  <Descriptions.Item label="端口">
                    上行: {selectedNode.portInfo.uplink} | 下行: {selectedNode.portInfo.downlink} | 互联: {selectedNode.portInfo.inter}
                  </Descriptions.Item>
                )}
                <Descriptions.Item label="连接数">{selectedNode.connections.length}</Descriptions.Item>
              </Descriptions>
              {selectedNode.connections.length > 0 && (
                <Collapse
                  size="small"
                  style={{ marginTop: 8 }}
                  items={[{
                    key: 'connections',
                    label: `连接列表 (${selectedNode.connections.length})`,
                    children: (
                      <div style={{ maxHeight: 150, overflow: 'auto' }}>
                        {selectedNode.connections.map((conn, idx) => (
                          <div key={idx} style={{ fontSize: 12, padding: '2px 0', borderBottom: '1px solid #f0f0f0' }}>
                            {conn.label}
                            {conn.bandwidth && <span style={{ color: '#999', marginLeft: 8 }}>{conn.bandwidth}Gbps</span>}
                          </div>
                        ))}
                      </div>
                    ),
                  }]}
                />
              )}
            </Card>
          )}

          {/* 连接详情卡片 */}
          {selectedLink && (
            <Card
              title="连接详情"
              size="small"
              style={{ marginTop: 16 }}
              extra={<a onClick={() => setSelectedLink(null)}>关闭</a>}
            >
              <Descriptions column={1} size="small">
                <Descriptions.Item label="源节点">
                  <Tag color="green">{selectedLink.sourceLabel}</Tag>
                  <span style={{ color: '#999', marginLeft: 4, fontSize: 12 }}>({selectedLink.sourceType.toUpperCase()})</span>
                </Descriptions.Item>
                <Descriptions.Item label="目标节点">
                  <Tag color="blue">{selectedLink.targetLabel}</Tag>
                  <span style={{ color: '#999', marginLeft: 4, fontSize: 12 }}>({selectedLink.targetType.toUpperCase()})</span>
                </Descriptions.Item>
                {selectedLink.bandwidth && (
                  <Descriptions.Item label="带宽">{selectedLink.bandwidth} Gbps</Descriptions.Item>
                )}
                {selectedLink.latency && (
                  <Descriptions.Item label="延迟">{selectedLink.latency} ns</Descriptions.Item>
                )}
                <Descriptions.Item label="类型">
                  <Tag color={selectedLink.isManual ? 'orange' : 'default'}>
                    {selectedLink.isManual ? '手动连接' : '自动连接'}
                  </Tag>
                </Descriptions.Item>
              </Descriptions>
            </Card>
          )}

          {/* 拖拽手柄 */}
          <div
            onMouseDown={handleMouseDown}
            style={{
              position: 'absolute',
              top: 0,
              right: 0,
              width: 4,
              height: '100%',
              cursor: 'col-resize',
              background: isDragging ? '#4f46e5' : 'transparent',
              transition: 'background 0.15s',
              zIndex: 10,
            }}
            onMouseEnter={(e) => {
              if (!isDragging) {
                (e.target as HTMLElement).style.background = '#e2e8f0'
              }
            }}
            onMouseLeave={(e) => {
              if (!isDragging) {
                (e.target as HTMLElement).style.background = 'transparent'
              }
            }}
          />
        </Sider>

        <Content style={{ position: 'relative', background: '#ffffff' }}>
          {loading && !topology ? (
            <div style={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              height: '100%',
            }}>
              <Spin size="large" tip="加载中..." />
            </div>
          ) : viewMode === '3d' ? (
            <Scene3D
              topology={topology}
              viewState={navigation.viewState}
              breadcrumbs={navigation.breadcrumbs}
              currentPod={navigation.currentPod}
              currentRack={navigation.currentRack}
              currentBoard={navigation.currentBoard}
              onNavigate={navigation.navigateTo}
              onNavigateToPod={navigation.navigateToPod}
              onNavigateToRack={navigation.navigateToRack}
              onNavigateBack={navigation.navigateBack}
              onBreadcrumbClick={navigation.navigateToBreadcrumb}
              canGoBack={navigation.canGoBack}
              onNodeSelect={handleScene3DNodeSelect}
            />
          ) : (
            <TopologyGraph
              visible={true}
              onClose={() => setViewMode('3d')}
              topology={topology}
              currentLevel={getCurrentLevel()}
              currentPod={navigation.currentPod}
              currentRack={navigation.currentRack}
              currentBoard={navigation.currentBoard}
              onNodeDoubleClick={(nodeId, nodeType) => {
                // 双击进入时切换到单层级视图，保留当前布局设置
                if (multiLevelOptions.enabled) {
                  setMultiLevelOptions(prev => ({ ...prev, enabled: false }))
                  // 不重置 layoutType，保留用户选择的布局
                }
                if (nodeType === 'pod') {
                  navigation.navigateToPod(nodeId)
                } else if (nodeType === 'rack' && navigation.currentPod) {
                  navigation.navigateToRack(navigation.currentPod.id, nodeId)
                } else if (nodeType === 'board') {
                  navigation.navigateTo(nodeId)
                }
              }}
              onNodeClick={(node) => {
                setSelectedNode(node)
                if (node) {
                  setSelectedLink(null)  // 点击节点时清除选中的连接
                  // 如果点击的是容器（subType 是层级类型），切换左侧层级配置
                  const levelTypes = ['datacenter', 'pod', 'rack', 'board']
                  if (node.subType && levelTypes.includes(node.subType)) {
                    setFocusedLevel(node.subType as 'datacenter' | 'pod' | 'rack' | 'board')
                  } else {
                    setFocusedLevel(null)
                  }
                } else {
                  setFocusedLevel(null)
                }
              }}
              onLinkClick={(link) => {
                setSelectedLink(link)
                if (link) setSelectedNode(null)  // 点击连接时清除选中的节点
              }}
              selectedNodeId={selectedNode?.id || null}
              selectedLinkId={selectedLink?.id || null}
              onNavigateBack={() => {
                // 导航返回时切换到单层级视图
                if (multiLevelOptions.enabled) {
                  setMultiLevelOptions(prev => ({ ...prev, enabled: false }))
                }
                navigation.navigateBack()
              }}
              onBreadcrumbClick={(index) => {
                // 面包屑导航时切换到单层级视图
                if (multiLevelOptions.enabled) {
                  setMultiLevelOptions(prev => ({ ...prev, enabled: false }))
                }
                navigation.navigateToBreadcrumb(index)
              }}
              breadcrumbs={navigation.breadcrumbs}
              canGoBack={navigation.canGoBack}
              embedded={true}
              connectionMode={connectionMode}
              selectedNodes={selectedNodes}
              onSelectedNodesChange={handleSelectedNodesChange}
              targetNodes={targetNodes}
              onTargetNodesChange={handleTargetNodesChange}
              sourceNode={sourceNode}
              onSourceNodeChange={handleSourceNodeChange}
              onManualConnect={handleManualConnect}
              manualConnections={manualConnectionConfig.connections}
              onDeleteManualConnection={handleDeleteManualConnection}
              onDeleteConnection={handleDeleteConnection}
              layoutType={layoutType}
              onLayoutTypeChange={setLayoutType}
              multiLevelOptions={multiLevelOptions}
              onMultiLevelOptionsChange={setMultiLevelOptions}
              trafficAnalysisResult={trafficAnalysisResult}
            />
          )}
        </Content>
      </Layout>
    </Layout>
  )
}

export default App
