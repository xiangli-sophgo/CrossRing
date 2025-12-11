import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react'
import { Layout, Typography, Spin, message, Segmented, Card, Descriptions, Tag, Collapse } from 'antd'
import { AppstoreOutlined, NodeIndexOutlined } from '@ant-design/icons'
import { Scene3D } from './components/Scene3D'
import { ConfigPanel } from './components/ConfigPanel'
import { TopologyGraph, NodeDetail } from './components/TopologyGraph'
import { HierarchicalTopology, ManualConnectionConfig, ManualConnection, ConnectionMode, HierarchyLevel } from './types'
import { getTopology, generateTopology } from './api/topology'
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

  // 手动连接状态
  const [manualConnectionConfig, setManualConnectionConfig] = useState<ManualConnectionConfig>({
    enabled: false,
    mode: 'append',
    connections: [],
  })
  const [connectionMode, setConnectionMode] = useState<ConnectionMode>('view')
  const [selectedNodes, setSelectedNodes] = useState<Set<string>>(new Set())
  const [sourceNode, setSourceNode] = useState<string | null>(null)

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

  // 手动连接处理函数
  const handleManualConnectionConfigChange = useCallback((config: ManualConnectionConfig) => {
    setManualConnectionConfig(config)
    // 禁用时重置编辑状态
    if (!config.enabled) {
      setConnectionMode('view')
      setSelectedNodes(new Set())
      setSourceNode(null)
    }
  }, [])

  const handleConnectionModeChange = useCallback((mode: ConnectionMode) => {
    setConnectionMode(mode)
    if (mode === 'view') {
      setSelectedNodes(new Set())
      setSourceNode(null)
    } else if (mode === 'select') {
      setSourceNode(null)
    }
  }, [])

  const handleSelectedNodesChange = useCallback((nodes: Set<string>) => {
    setSelectedNodes(nodes)
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

  // 计算当前视图的连接
  const currentViewConnections = useMemo(() => {
    if (!topology) return []
    const currentLevel = getCurrentLevel()

    if (currentLevel === 'datacenter') {
      // Datacenter层：Pod间连接
      const podIds = new Set(topology.pods.map(p => p.id))
      const dcSwitchIds = new Set(
        (topology.switches || [])
          .filter(s => s.hierarchy_level === 'datacenter')
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
          .filter(s => s.hierarchy_level === 'pod' && s.parent_id === navigation.currentPod!.id)
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
          .filter(s => s.hierarchy_level === 'rack' && s.parent_id === navigation.currentRack!.id)
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
        background: 'linear-gradient(135deg, #1890ff 0%, #722ed1 100%)',
        padding: '0 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <Title level={4} style={{ color: '#fff', margin: 0 }}>
          Tier6+互联拓扑
        </Title>
        <Segmented
          value={viewMode}
          onChange={(v) => setViewMode(v as '3d' | 'topology')}
          options={[
            { value: '3d', label: '3D视图' },
            { value: 'topology', label: '拓扑图' },
          ]}
          style={{ background: 'rgba(255,255,255,0.2)' }}
        />
      </Header>

      <Layout>
        <Sider
          width={siderWidth}
          style={{
            background: '#fff',
            padding: 16,
            overflow: 'auto',
            position: 'relative',
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
            onDeleteManualConnection={handleDeleteManualConnection}
            currentViewConnections={currentViewConnections}
            onDeleteConnection={handleDeleteConnection}
          />

          {/* 节点详情卡片 */}
          {selectedNode && (
            <Card
              title={`节点详情: ${selectedNode.label}`}
              size="small"
              style={{ marginTop: 16, background: '#f8f9fa', border: '1px solid #e9ecef' }}
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

          {/* 拖拽手柄 */}
          <div
            onMouseDown={handleMouseDown}
            style={{
              position: 'absolute',
              top: 0,
              right: 0,
              width: 6,
              height: '100%',
              cursor: 'col-resize',
              background: isDragging ? '#1890ff' : 'transparent',
              transition: 'background 0.2s',
              zIndex: 10,
            }}
            onMouseEnter={(e) => {
              if (!isDragging) {
                (e.target as HTMLElement).style.background = '#e0e0e0'
              }
            }}
            onMouseLeave={(e) => {
              if (!isDragging) {
                (e.target as HTMLElement).style.background = 'transparent'
              }
            }}
          />
        </Sider>

        <Content style={{ position: 'relative' }}>
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
                if (nodeType === 'pod') {
                  navigation.navigateToPod(nodeId)
                } else if (nodeType === 'rack' && navigation.currentPod) {
                  navigation.navigateToRack(navigation.currentPod.id, nodeId)
                } else if (nodeType === 'board') {
                  navigation.navigateTo(nodeId)
                }
              }}
              onNodeClick={setSelectedNode}
              onNavigateBack={navigation.navigateBack}
              onBreadcrumbClick={navigation.navigateToBreadcrumb}
              breadcrumbs={navigation.breadcrumbs}
              canGoBack={navigation.canGoBack}
              embedded={true}
              connectionMode={connectionMode}
              selectedNodes={selectedNodes}
              onSelectedNodesChange={handleSelectedNodesChange}
              sourceNode={sourceNode}
              onSourceNodeChange={handleSourceNodeChange}
              onManualConnect={handleManualConnect}
              manualConnections={manualConnectionConfig.connections}
              onDeleteManualConnection={handleDeleteManualConnection}
              onDeleteConnection={handleDeleteConnection}
            />
          )}
        </Content>
      </Layout>
    </Layout>
  )
}

export default App
