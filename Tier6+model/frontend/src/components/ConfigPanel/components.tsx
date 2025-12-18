import React from 'react'
import {
  Typography,
  Button,
  Space,
  InputNumber,
  Collapse,
  Input,
  Switch,
  Select,
  Checkbox,
  Divider,
  Radio,
  Tooltip,
  Tag,
  Card,
} from 'antd'
import {
  DeleteOutlined,
  PlusOutlined,
  MinusCircleOutlined,
  UndoOutlined,
  QuestionCircleOutlined,
} from '@ant-design/icons'
import {
  HierarchyLevelSwitchConfig, SwitchTypeConfig, SwitchLayerConfig,
  ManualConnectionConfig, ConnectionMode, SwitchConnectionMode, HierarchyLevel,
  LevelConnectionDefaults, TrafficConfigItem, TrafficAnalysisResult,
  DEFAULT_TRAFFIC_CONFIG_ITEM, HierarchicalTopology, CollectiveType,
  ParallelismType, PARALLELISM_NAMES, PARALLELISM_COLORS,
} from '../../types'

const { Text } = Typography

// 集合操作详细说明
const COLLECTIVE_DESCRIPTIONS: Record<CollectiveType, { pattern: string; formula: string; desc: string }> = {
  AllReduce: {
    pattern: 'Ring',
    formula: '2×(n-1)/n × M',
    desc: '每个节点贡献数据，规约后所有节点得到相同结果',
  },
  AllGather: {
    pattern: 'Ring',
    formula: '(n-1)/n × M',
    desc: '收集所有节点数据，每个节点获得完整数据集',
  },
  ReduceScatter: {
    pattern: 'Ring',
    formula: '(n-1)/n × M',
    desc: '规约后将结果分散到各节点',
  },
  AllToAll: {
    pattern: '全交换',
    formula: '(n-1)/n × M',
    desc: '每个节点向所有其他节点发送不同数据',
  },
  P2P: {
    pattern: '点对点',
    formula: 'M',
    desc: '相邻节点间单向传输',
  },
}

// 各并行策略的典型操作选项
const PARALLELISM_COLLECTIVE_OPTIONS: Record<string, Array<{ value: CollectiveType; label: string; recommended?: boolean }>> = {
  DP: [
    { value: 'AllReduce', label: 'AllReduce', recommended: true },
    { value: 'ReduceScatter', label: 'ReduceScatter' },
    { value: 'AllGather', label: 'AllGather' },
  ],
  TP: [
    { value: 'AllReduce', label: 'AllReduce', recommended: true },
    { value: 'AllGather', label: 'AllGather' },
    { value: 'ReduceScatter', label: 'ReduceScatter' },
  ],
  PP: [
    { value: 'P2P', label: 'P2P', recommended: true },
  ],
  EP: [
    { value: 'AllToAll', label: 'AllToAll', recommended: true },
  ],
  SP: [
    { value: 'AllGather', label: 'AllGather', recommended: true },
    { value: 'ReduceScatter', label: 'ReduceScatter' },
  ],
}

// ============================================
// Switch层级配置子组件
// ============================================

interface SwitchLevelConfigProps {
  levelKey: string
  config: HierarchyLevelSwitchConfig
  switchTypes: SwitchTypeConfig[]
  onChange: (config: HierarchyLevelSwitchConfig) => void
  configRowStyle: React.CSSProperties
  viewMode?: '3d' | 'topology'
}

export const SwitchLevelConfig: React.FC<SwitchLevelConfigProps> = ({
  levelKey,
  config,
  switchTypes,
  onChange,
  configRowStyle,
  viewMode,
}) => {
  const updateLayer = (index: number, field: keyof SwitchLayerConfig, value: any) => {
    const newLayers = [...config.layers]
    newLayers[index] = { ...newLayers[index], [field]: value }
    onChange({ ...config, layers: newLayers })
  }

  const addLayer = () => {
    const newLayer: SwitchLayerConfig = {
      layer_name: config.layers.length === 0 ? 'leaf' : 'spine',
      switch_type_id: switchTypes[0]?.id || '',
      count: 2,
      inter_connect: false,
    }
    onChange({ ...config, layers: [...config.layers, newLayer] })
  }

  const removeLayer = (index: number) => {
    const newLayers = config.layers.filter((_, i) => i !== index)
    onChange({ ...config, layers: newLayers })
  }

  // 直连拓扑类型选项
  const directTopologyOptions = [
    { value: 'none', label: '无连接' },
    { value: 'full_mesh', label: '全连接 (Full Mesh)' },
    { value: 'full_mesh_2d', label: '2D FullMesh (行列全连接)' },
    { value: 'ring', label: '环形 (Ring)' },
    { value: 'torus_2d', label: '2D Torus' },
    { value: 'torus_3d', label: '3D Torus' },
  ]

  return (
    <div>
      {/* 启用开关 */}
      <div style={configRowStyle}>
        <Text>启用Switch</Text>
        <Switch
          size="small"
          checked={config.enabled}
          onChange={(checked) => onChange({ ...config, enabled: checked })}
        />
      </div>

      {/* 不启用Switch时显示直连拓扑选项 */}
      {!config.enabled && (
        <div style={configRowStyle}>
          <Text>直连拓扑</Text>
          <Select
            size="small"
            value={config.direct_topology || 'none'}
            onChange={(v) => onChange({ ...config, direct_topology: v })}
            style={{ width: 150 }}
            options={directTopologyOptions}
          />
        </div>
      )}

      {config.enabled && (
        <>
          {/* 保留节点直连 */}
          <div style={configRowStyle}>
            <Text>保留节点直连</Text>
            <Switch
              size="small"
              checked={config.keep_direct_topology || false}
              onChange={(checked) => onChange({ ...config, keep_direct_topology: checked })}
            />
          </div>

          {/* 保留直连时选择拓扑类型 */}
          {config.keep_direct_topology && (
            <div style={configRowStyle}>
              <Text>直连拓扑</Text>
              <Select
                size="small"
                value={config.direct_topology || 'full_mesh'}
                onChange={(v) => onChange({ ...config, direct_topology: v })}
                style={{ width: 150 }}
                options={directTopologyOptions}
              />
            </div>
          )}

          {/* 连接模式 */}
          <div style={configRowStyle}>
            <Text>Switch连接模式</Text>
            <Select
              size="small"
              value={config.connection_mode || 'full_mesh'}
              onChange={(v: SwitchConnectionMode) => onChange({ ...config, connection_mode: v })}
              style={{ width: 120 }}
              options={[
                { value: 'full_mesh', label: '全连接' },
                { value: 'custom', label: '自定义' },
              ]}
            />
          </div>

          {/* 自定义模式：节点连接Switch数 */}
          {config.connection_mode === 'custom' && (
            <div style={configRowStyle}>
              <Text>节点连接Switch数</Text>
              <InputNumber
                min={1}
                max={config.layers[0]?.count || 1}
                size="small"
                value={Math.min(config.downlink_redundancy || 1, config.layers[0]?.count || 1)}
                onChange={(v) => onChange({ ...config, downlink_redundancy: v || 1 })}
                style={{ width: 60 }}
              />
            </div>
          )}

          {/* Switch 3D显示配置（仅rack层级且3D视图时显示） */}
          {levelKey === 'inter_board' && viewMode === '3d' && (
            <>
              <div style={configRowStyle}>
                <Text>Switch位置</Text>
                <Radio.Group
                  size="small"
                  value={config.switch_position || 'top'}
                  onChange={(e) => onChange({ ...config, switch_position: e.target.value })}
                >
                  <Radio.Button value="top">顶部</Radio.Button>
                  <Radio.Button value="middle">中间</Radio.Button>
                  <Radio.Button value="bottom">底部</Radio.Button>
                </Radio.Group>
              </div>
              <div style={configRowStyle}>
                <Text>Switch高度</Text>
                <Radio.Group
                  size="small"
                  value={config.switch_u_height || 1}
                  onChange={(e) => onChange({ ...config, switch_u_height: e.target.value })}
                >
                  <Radio.Button value={1}>1U</Radio.Button>
                  <Radio.Button value={2}>2U</Radio.Button>
                  <Radio.Button value={4}>4U</Radio.Button>
                </Radio.Group>
              </div>
            </>
          )}

          <Divider style={{ margin: '8px 0' }} />

          {/* Switch层列表 */}
          <Text type="secondary" style={{ fontSize: 11 }}>Switch层配置 (从下到上)</Text>
          {config.layers.map((layer, index) => (
            <div key={index} style={{ marginTop: 8, padding: 8, background: '#f5f5f5', borderRadius: 4 }}>
              {/* 第一行：层名称和删除按钮 */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Text style={{ fontSize: 11, color: '#666' }}>层名称</Text>
                  <Input
                    size="small"
                    placeholder="如 leaf, spine"
                    value={layer.layer_name}
                    onChange={(e) => updateLayer(index, 'layer_name', e.target.value)}
                    style={{ width: 80 }}
                  />
                </div>
                <Button
                  type="text"
                  danger
                  size="small"
                  icon={<MinusCircleOutlined />}
                  onClick={() => removeLayer(index)}
                />
              </div>
              {/* 第二行：Switch类型和数量 */}
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
                <Select
                  size="small"
                  value={layer.switch_type_id}
                  onChange={(v) => updateLayer(index, 'switch_type_id', v)}
                  style={{ flex: 1 }}
                  options={switchTypes.map(t => ({ value: t.id, label: `${t.name} (${t.port_count}口)` }))}
                />
                <Text style={{ fontSize: 11, color: '#666' }}>×</Text>
                <InputNumber
                  size="small"
                  min={1}
                  max={16}
                  value={layer.count}
                  onChange={(v) => updateLayer(index, 'count', v || 1)}
                  style={{ width: 60 }}
                />
                <Text style={{ fontSize: 11, color: '#666' }}>台</Text>
              </div>
              {/* 第三行：同层互联选项 */}
              <Checkbox
                checked={layer.inter_connect}
                onChange={(e) => updateLayer(index, 'inter_connect', e.target.checked)}
              >
                <Text style={{ fontSize: 11 }}>同层互联</Text>
              </Checkbox>
            </div>
          ))}

          <Button
            type="dashed"
            size="small"
            icon={<PlusOutlined />}
            onClick={addLayer}
            style={{ marginTop: 8, width: '100%' }}
          >
            添加Switch层
          </Button>
        </>
      )}

    </div>
  )
}

// ============================================
// 连接编辑子组件
// ============================================

interface ConnectionEditPanelProps {
  manualConnectionConfig?: ManualConnectionConfig
  onManualConnectionConfigChange?: (config: ManualConnectionConfig) => void
  connectionMode?: ConnectionMode
  onConnectionModeChange?: (mode: ConnectionMode) => void
  selectedNodes?: Set<string>
  onSelectedNodesChange?: (nodes: Set<string>) => void
  targetNodes?: Set<string>
  onTargetNodesChange?: (nodes: Set<string>) => void
  onBatchConnect?: (level: HierarchyLevel) => void
  onDeleteManualConnection?: (id: string) => void
  currentViewConnections?: Array<{ source: string; target: string; type?: string; bandwidth?: number; latency?: number }>
  onDeleteConnection?: (source: string, target: string) => void
  onUpdateConnectionParams?: (source: string, target: string, bandwidth?: number, latency?: number) => void
  configRowStyle: React.CSSProperties
  currentLevel?: string
}

export const ConnectionEditPanel: React.FC<ConnectionEditPanelProps> = ({
  manualConnectionConfig,
  onManualConnectionConfigChange,
  connectionMode = 'view',
  onConnectionModeChange,
  selectedNodes = new Set<string>(),
  onSelectedNodesChange,
  targetNodes = new Set<string>(),
  onTargetNodesChange,
  onBatchConnect,
  onDeleteManualConnection,
  currentViewConnections = [],
  onDeleteConnection,
  onUpdateConnectionParams,
  configRowStyle: _configRowStyle,
  currentLevel = 'datacenter',
}) => {
  void _configRowStyle
  // 获取当前层级
  const getCurrentHierarchyLevel = (): HierarchyLevel => {
    switch (currentLevel) {
      case 'datacenter': return 'datacenter'
      case 'pod': return 'pod'
      case 'rack': return 'rack'
      case 'board': return 'board'
      default: return 'datacenter'
    }
  }
  // 获取当前层级的默认参数
  const levelKey = currentLevel as 'datacenter' | 'pod' | 'rack' | 'board'
  const currentDefaults = manualConnectionConfig?.level_defaults?.[levelKey] || {}

  // 更新层级默认参数
  const updateLevelDefaults = (defaults: LevelConnectionDefaults) => {
    if (!onManualConnectionConfigChange) return
    const newConfig: ManualConnectionConfig = {
      ...(manualConnectionConfig || { enabled: true, mode: 'append', connections: [] }),
      level_defaults: {
        ...(manualConnectionConfig?.level_defaults || {}),
        [levelKey]: defaults,
      },
    }
    onManualConnectionConfigChange(newConfig)
  }

  // 更新手动连接的参数
  const updateManualConnectionParams = (connId: string, bandwidth?: number, latency?: number) => {
    if (!onManualConnectionConfigChange || !manualConnectionConfig) return
    const newConnections = manualConnectionConfig.connections.map(conn => {
      if (conn.id === connId) {
        return { ...conn, bandwidth, latency }
      }
      return conn
    })
    onManualConnectionConfigChange({
      ...manualConnectionConfig,
      connections: newConnections,
    })
  }

  return (
    <div style={{
      padding: 14,
      background: '#f5f5f5',
      borderRadius: 10,
      border: '1px solid rgba(0, 0, 0, 0.06)',
    }}>
      <Text strong style={{ display: 'block', marginBottom: 10, color: '#171717' }}>连接编辑</Text>

      {/* 层级默认带宽/延迟配置 */}
      <div style={{
        marginBottom: 12,
        padding: 10,
        background: '#fff',
        borderRadius: 6,
        border: '1px solid #e8e8e8',
      }}>
        <div style={{ marginBottom: 8 }}>
          <Text style={{ fontSize: 12, color: '#333', fontWeight: 500 }}>层级默认参数</Text>
          <Text style={{ fontSize: 11, color: '#999', marginLeft: 8 }}>新建连接时自动应用</Text>
        </div>
        <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <Text style={{ fontSize: 12 }}>带宽:</Text>
            <InputNumber
              size="small"
              min={0}
              value={currentDefaults.bandwidth}
              onChange={(v) => updateLevelDefaults({ ...currentDefaults, bandwidth: v || undefined })}
              style={{ width: 80 }}
              placeholder="未设置"
            />
            <Text style={{ fontSize: 11, color: '#999' }}>Gbps</Text>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <Text style={{ fontSize: 12 }}>延迟:</Text>
            <InputNumber
              size="small"
              min={0}
              value={currentDefaults.latency}
              onChange={(v) => updateLevelDefaults({ ...currentDefaults, latency: v || undefined })}
              style={{ width: 80 }}
              placeholder="未设置"
            />
            <Text style={{ fontSize: 11, color: '#999' }}>ns</Text>
          </div>
        </div>
      </div>

      {/* 编辑模式按钮 */}
      <div style={{ marginBottom: 12 }}>
        {connectionMode === 'view' ? (
          <Button
            type="default"
            onClick={() => onConnectionModeChange?.('select_source')}
          >
            编辑连接
          </Button>
        ) : (
          <Space>
            <Button
              type={connectionMode === 'select_source' ? 'primary' : 'default'}
              onClick={() => onConnectionModeChange?.('select_source')}
            >
              选源节点
            </Button>
            <Button
              type={connectionMode === 'select_target' ? 'primary' : 'default'}
              onClick={() => onConnectionModeChange?.('select_target')}
              disabled={selectedNodes.size === 0}
            >
              选目标节点
            </Button>
            <Button onClick={() => onConnectionModeChange?.('view')}>
              退出
            </Button>
          </Space>
        )}
      </div>

      {connectionMode !== 'view' && (
        <div style={{
          padding: 12,
          background: 'rgba(37, 99, 235, 0.04)',
          borderRadius: 8,
          marginBottom: 12,
          border: '1px solid rgba(37, 99, 235, 0.1)',
        }}>
          <Text style={{ fontSize: 12, display: 'block', marginBottom: 6, color: '#525252' }}>
            <strong>操作说明：</strong>
          </Text>
          <Text style={{ fontSize: 12, display: 'block', color: connectionMode === 'select_source' ? '#2563eb' : '#525252' }}>
            1. 点击图中节点选为源节点（绿色框）
          </Text>
          <Text style={{ fontSize: 12, display: 'block', color: connectionMode === 'select_target' ? '#2563eb' : '#525252' }}>
            2. 切换到"选目标节点"，点击选择目标（蓝色框）
          </Text>
          <Text style={{ fontSize: 12, display: 'block', color: '#525252' }}>
            3. 点击下方"确认连接"按钮完成
          </Text>
        </div>
      )}

      {/* 选中状态显示 */}
      {(selectedNodes.size > 0 || targetNodes.size > 0) && (
        <div style={{
          marginBottom: 12,
          padding: 12,
          background: 'rgba(5, 150, 105, 0.04)',
          borderRadius: 8,
          border: '1px solid rgba(5, 150, 105, 0.1)',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <Text style={{ fontSize: 14 }}>
              <strong>源节点: {selectedNodes.size} 个</strong>
              {selectedNodes.size > 0 && (
                <span style={{ fontSize: 12, color: '#666', marginLeft: 8 }}>
                  ({Array.from(selectedNodes).slice(0, 3).join(', ')}{selectedNodes.size > 3 ? '...' : ''})
                </span>
              )}
            </Text>
            {selectedNodes.size > 0 && (
              <Button size="small" type="link" onClick={() => onSelectedNodesChange?.(new Set())}>清空</Button>
            )}
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Text style={{ fontSize: 14 }}>
              <strong>目标节点: {targetNodes.size} 个</strong>
              {targetNodes.size > 0 && (
                <span style={{ fontSize: 12, color: '#666', marginLeft: 8 }}>
                  ({Array.from(targetNodes).slice(0, 3).join(', ')}{targetNodes.size > 3 ? '...' : ''})
                </span>
              )}
            </Text>
            {targetNodes.size > 0 && (
              <Button size="small" type="link" onClick={() => onTargetNodesChange?.(new Set())}>清空</Button>
            )}
          </div>
          {selectedNodes.size > 0 && targetNodes.size > 0 && (() => {
            // 计算实际会创建的连接数（排除已存在的连接）
            let newCount = 0
            let existCount = 0
            selectedNodes.forEach(sourceId => {
              targetNodes.forEach(targetId => {
                if (sourceId === targetId) return
                // 检查是否已存在于当前视图连接中
                const existsInView = currentViewConnections.some(c =>
                  (c.source === sourceId && c.target === targetId) ||
                  (c.source === targetId && c.target === sourceId)
                )
                // 检查是否已存在于手动连接中
                const existsManual = manualConnectionConfig?.connections?.some(c =>
                  (c.source === sourceId && c.target === targetId) ||
                  (c.source === targetId && c.target === sourceId)
                )
                if (existsInView || existsManual) {
                  existCount++
                } else {
                  newCount++
                }
              })
            })
            return (
              <Button
                type="primary"
                style={{ marginTop: 12, width: '100%' }}
                onClick={() => onBatchConnect?.(getCurrentHierarchyLevel())}
                disabled={newCount === 0}
              >
                确认连接（{newCount} 条新连接{existCount > 0 ? `，${existCount} 条已存在` : ''}）
              </Button>
            )
          })()}
        </div>
      )}

      {/* 手动添加的连接列表 */}
      {(() => {
        // 过滤当前层级的手动连接
        const currentLevelConnections = manualConnectionConfig?.connections?.filter(
          conn => conn.hierarchy_level === currentLevel
        ) || []
        return (
          <Collapse
            size="small"
            style={{ marginTop: 8 }}
            items={[{
              key: 'manual',
              label: <span style={{ fontSize: 14 }}>手动连接 ({currentLevelConnections.length})</span>,
              children: (
                <div style={{ maxHeight: 240, overflow: 'auto' }}>
                  {currentLevelConnections.map((conn) => {
                    // 判断是否使用默认值（值为空）
                    const useDefaultBandwidth = conn.bandwidth === undefined || conn.bandwidth === null
                    const useDefaultLatency = conn.latency === undefined || conn.latency === null
                    const hasCustom = !useDefaultBandwidth || !useDefaultLatency
                    // 显示值：空值时显示默认值
                    const displayBandwidth = useDefaultBandwidth ? currentDefaults.bandwidth : conn.bandwidth
                    const displayLatency = useDefaultLatency ? currentDefaults.latency : conn.latency
                    return (
                  <div
                    key={conn.id}
                    style={{
                      padding: 10,
                      background: 'rgba(5, 150, 105, 0.04)',
                      marginBottom: 8,
                      borderRadius: 8,
                      border: '1px solid rgba(5, 150, 105, 0.1)',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <Text code style={{ fontSize: 14 }}>{conn.source}</Text>
                        <Text style={{ margin: '0 6px', fontSize: 14 }}>↔</Text>
                        <Text code style={{ fontSize: 14 }}>{conn.target}</Text>
                      </div>
                      <Space size={4}>
                        {hasCustom && (
                          <Button
                            type="text"
                            size="small"
                            icon={<UndoOutlined />}
                            title="重置为默认"
                            onClick={() => updateManualConnectionParams(conn.id, undefined, undefined)}
                            style={{ color: '#999' }}
                          />
                        )}
                        <Button
                          type="text"
                          danger
                          size="small"
                          icon={<DeleteOutlined />}
                          onClick={() => onDeleteManualConnection?.(conn.id)}
                        />
                      </Space>
                    </div>
                    <div style={{ marginTop: 8, display: 'flex', gap: 12, alignItems: 'center' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        <Text style={{ fontSize: 11, color: useDefaultBandwidth ? '#999' : '#333' }}>带宽:</Text>
                        <InputNumber
                          size="small"
                          min={0}
                          value={displayBandwidth}
                          onChange={(v) => updateManualConnectionParams(conn.id, v ?? undefined, conn.latency)}
                          style={{ width: 80, color: useDefaultBandwidth ? '#999' : undefined }}
                          placeholder="Gbps"
                        />
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        <Text style={{ fontSize: 11, color: useDefaultLatency ? '#999' : '#333' }}>延迟:</Text>
                        <InputNumber
                          size="small"
                          min={0}
                          value={displayLatency}
                          onChange={(v) => updateManualConnectionParams(conn.id, conn.bandwidth, v ?? undefined)}
                          style={{ width: 80, color: useDefaultLatency ? '#999' : undefined }}
                          placeholder="ns"
                        />
                      </div>
                    </div>
                  </div>
                )
                  })}
                  {currentLevelConnections.length === 0 && (
                    <Text type="secondary" style={{ fontSize: 13 }}>暂无手动连接</Text>
                  )}
                </div>
              ),
            }, {
          key: 'current',
          label: <span style={{ fontSize: 14 }}>当前连接 ({currentViewConnections.length})</span>,
          children: (
            <div style={{ maxHeight: 240, overflow: 'auto' }}>
              {currentViewConnections.map((conn, idx) => {
                // 判断是否使用默认值（值为空）
                const useDefaultBandwidth = conn.bandwidth === undefined || conn.bandwidth === null
                const useDefaultLatency = conn.latency === undefined || conn.latency === null
                const hasCustom = !useDefaultBandwidth || !useDefaultLatency
                // 显示值：空值时显示默认值
                const displayBandwidth = useDefaultBandwidth ? currentDefaults.bandwidth : conn.bandwidth
                const displayLatency = useDefaultLatency ? currentDefaults.latency : conn.latency
                return (
                  <div
                    key={`auto-${idx}`}
                    style={{
                      padding: 10,
                      background: '#fff',
                      marginBottom: 8,
                      borderRadius: 8,
                      border: '1px solid rgba(0, 0, 0, 0.06)',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div style={{ flex: 1 }}>
                        <Text code style={{ fontSize: 14 }}>{conn.source}</Text>
                        <Text style={{ margin: '0 6px', fontSize: 14 }}>↔</Text>
                        <Text code style={{ fontSize: 14 }}>{conn.target}</Text>
                      </div>
                      <Space size={4}>
                        {hasCustom && (
                          <Button
                            type="text"
                            size="small"
                            icon={<UndoOutlined />}
                            title="重置为默认"
                            onClick={() => onUpdateConnectionParams?.(conn.source, conn.target, undefined, undefined)}
                            style={{ color: '#999' }}
                          />
                        )}
                        <Button
                          type="text"
                          danger
                          size="small"
                          icon={<DeleteOutlined />}
                          onClick={() => onDeleteConnection?.(conn.source, conn.target)}
                        />
                      </Space>
                    </div>
                    <div style={{ marginTop: 8, display: 'flex', gap: 12, alignItems: 'center' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        <Text style={{ fontSize: 11, color: useDefaultBandwidth ? '#999' : '#333' }}>带宽:</Text>
                        <InputNumber
                          size="small"
                          min={0}
                          value={displayBandwidth}
                          onChange={(v) => onUpdateConnectionParams?.(conn.source, conn.target, v ?? undefined, conn.latency)}
                          style={{ width: 80, color: useDefaultBandwidth ? '#999' : undefined }}
                          placeholder="Gbps"
                        />
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        <Text style={{ fontSize: 11, color: useDefaultLatency ? '#999' : '#333' }}>延迟:</Text>
                        <InputNumber
                          size="small"
                          min={0}
                          value={displayLatency}
                          onChange={(v) => onUpdateConnectionParams?.(conn.source, conn.target, conn.bandwidth, v ?? undefined)}
                          style={{ width: 80, color: useDefaultLatency ? '#999' : undefined }}
                          placeholder="ns"
                        />
                      </div>
                    </div>
                  </div>
                )
              })}
              {currentViewConnections.length === 0 && (
                <Text type="secondary" style={{ fontSize: 13 }}>暂无连接</Text>
              )}
            </div>
          ),
        }]}
      />
        )
      })()}
    </div>
  )
}

// ============================================
// LLM流量分析配置子组件 (多配置版本)
// ============================================

interface TrafficAnalysisPanelProps {
  configs: TrafficConfigItem[]
  onConfigsChange: (configs: TrafficConfigItem[]) => void
  onRunAnalysis: () => void
  analysisResult: TrafficAnalysisResult | null
  totalChips: number
  configRowStyle: React.CSSProperties
  onNavigateToChips?: () => void
  topology: HierarchicalTopology | null
}

// 生成唯一ID
const generateId = () => Math.random().toString(36).substr(2, 9)

// 计算配置范围内的 Chip 数量
const getChipsInScope = (
  config: TrafficConfigItem,
  topology: HierarchicalTopology | null,
  totalChips: number
): number => {
  if (!topology) return 0
  const { chip_scope, scope_pod_ids, scope_rack_ids, scope_board_ids, scope_chip_ids } = config

  if (chip_scope === 'all') return totalChips

  if (chip_scope === 'pod' && scope_pod_ids && scope_pod_ids.length > 0) {
    let count = 0
    for (const podId of scope_pod_ids) {
      const pod = topology.pods.find(p => p.id === podId)
      if (pod) {
        count += pod.racks.reduce((sum, r) =>
          sum + r.boards.reduce((s, b) => s + b.chips.length, 0), 0)
      }
    }
    return count
  }

  if (chip_scope === 'rack' && scope_rack_ids && scope_rack_ids.length > 0) {
    const rackIdSet = new Set(scope_rack_ids)
    let count = 0
    for (const pod of topology.pods) {
      for (const rack of pod.racks) {
        if (rackIdSet.has(rack.id)) {
          count += rack.boards.reduce((sum, b) => sum + b.chips.length, 0)
        }
      }
    }
    return count
  }

  if (chip_scope === 'board' && scope_board_ids && scope_board_ids.length > 0) {
    const boardIdSet = new Set(scope_board_ids)
    let count = 0
    for (const pod of topology.pods) {
      for (const rack of pod.racks) {
        for (const board of rack.boards) {
          if (boardIdSet.has(board.id)) {
            count += board.chips.length
          }
        }
      }
    }
    return count
  }

  if (chip_scope === 'chip' && scope_chip_ids) {
    return scope_chip_ids.length
  }

  return totalChips
}

export const TrafficAnalysisPanel: React.FC<TrafficAnalysisPanelProps> = ({
  configs,
  onConfigsChange,
  onRunAnalysis,
  analysisResult,
  totalChips,
  configRowStyle,
  onNavigateToChips,
  topology,
}) => {
  // 当前编辑区的配置
  const [currentConfig, setCurrentConfig] = React.useState<TrafficConfigItem>({
    ...DEFAULT_TRAFFIC_CONFIG_ITEM,
    id: generateId(),
    name: `配置 1`,
  })
  // 正在编辑的已有配置ID（null表示新建模式）
  const [editingExistingId, setEditingExistingId] = React.useState<string | null>(null)

  // 更新当前编辑配置
  const handleUpdateCurrentConfig = (updates: Partial<TrafficConfigItem>) => {
    setCurrentConfig(prev => ({ ...prev, ...updates }))
  }

  // 添加配置到列表
  const handleAddConfig = () => {
    if (editingExistingId) {
      // 更新已有配置
      onConfigsChange(configs.map(c => c.id === editingExistingId ? currentConfig : c))
      setEditingExistingId(null)
    } else {
      // 添加新配置
      onConfigsChange([...configs, currentConfig])
    }
    // 重置为新配置
    setCurrentConfig({
      ...DEFAULT_TRAFFIC_CONFIG_ITEM,
      id: generateId(),
      name: `配置 ${configs.length + (editingExistingId ? 1 : 2)}`,
    })
  }

  // 双击加载已有配置到编辑区
  const handleLoadConfig = (config: TrafficConfigItem) => {
    setCurrentConfig({ ...config })
    setEditingExistingId(config.id)
  }

  // 删除配置
  const handleDeleteConfig = (id: string) => {
    onConfigsChange(configs.filter(c => c.id !== id))
    if (editingExistingId === id) {
      setEditingExistingId(null)
      setCurrentConfig({
        ...DEFAULT_TRAFFIC_CONFIG_ITEM,
        id: generateId(),
        name: `配置 ${configs.length}`,
      })
    }
  }

  // 清空所有配置
  const handleClearAll = () => {
    onConfigsChange([])
    setEditingExistingId(null)
    setCurrentConfig({
      ...DEFAULT_TRAFFIC_CONFIG_ITEM,
      id: generateId(),
      name: `配置 1`,
    })
  }

  // 取消编辑（恢复新建模式）
  const handleCancelEdit = () => {
    setEditingExistingId(null)
    setCurrentConfig({
      ...DEFAULT_TRAFFIC_CONFIG_ITEM,
      id: generateId(),
      name: `配置 ${configs.length + 1}`,
    })
  }

  // 检查所有配置是否有效
  const allValid = configs.length > 0 && configs.every(c => {
    const chipsInScope = getChipsInScope(c, topology, totalChips)
    const needsSelection = c.chip_scope !== 'all' && (
      (c.chip_scope === 'pod' && (!c.scope_pod_ids || c.scope_pod_ids.length === 0)) ||
      (c.chip_scope === 'rack' && (!c.scope_rack_ids || c.scope_rack_ids.length === 0)) ||
      (c.chip_scope === 'board' && (!c.scope_board_ids || c.scope_board_ids.length === 0)) ||
      (c.chip_scope === 'chip' && (!c.scope_chip_ids || c.scope_chip_ids.length === 0))
    )
    return !needsSelection && c.size <= chipsInScope && c.size >= 2
  })

  // 当前编辑配置是否有效
  const currentConfigValid = (() => {
    const chipsInScope = getChipsInScope(currentConfig, topology, totalChips)
    const needsSelection = currentConfig.chip_scope !== 'all' && (
      (currentConfig.chip_scope === 'pod' && (!currentConfig.scope_pod_ids || currentConfig.scope_pod_ids.length === 0)) ||
      (currentConfig.chip_scope === 'rack' && (!currentConfig.scope_rack_ids || currentConfig.scope_rack_ids.length === 0)) ||
      (currentConfig.chip_scope === 'board' && (!currentConfig.scope_board_ids || currentConfig.scope_board_ids.length === 0)) ||
      (currentConfig.chip_scope === 'chip' && (!currentConfig.scope_chip_ids || currentConfig.scope_chip_ids.length === 0))
    )
    return !needsSelection && currentConfig.size <= chipsInScope && currentConfig.size >= 2
  })()

  return (
    <div>
      {/* 配置编辑区 - 始终显示 */}
      <div style={{ padding: 12, background: '#fafafa', borderRadius: 8, marginBottom: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <Text strong style={{ fontSize: 12 }}>
            {editingExistingId ? '编辑配置' : '新建配置'}
          </Text>
          {editingExistingId && (
            <Button type="link" size="small" onClick={handleCancelEdit} style={{ padding: 0, fontSize: 11 }}>
              取消编辑
            </Button>
          )}
        </div>

        {/* 名称 */}
        <div style={configRowStyle}>
          <Text style={{ fontSize: 12 }}>名称</Text>
          <Input
            size="small"
            value={currentConfig.name}
            onChange={(e) => handleUpdateCurrentConfig({ name: e.target.value })}
            style={{ width: 130 }}
          />
        </div>

        {/* 并行类型 */}
        <div style={configRowStyle}>
          <Text style={{ fontSize: 12 }}>并行类型</Text>
          <Select
            size="small"
            value={currentConfig.parallelism}
            onChange={(v: ParallelismType) => {
              const defaultCollective = PARALLELISM_COLLECTIVE_OPTIONS[v]?.[0]?.value || 'AllReduce'
              handleUpdateCurrentConfig({ parallelism: v, collective: defaultCollective })
            }}
            style={{ width: 130 }}
            options={(['DP', 'TP', 'PP', 'EP', 'SP'] as ParallelismType[]).map(p => ({
              value: p,
              label: PARALLELISM_NAMES[p],
            }))}
          />
        </div>

        {/* 集合操作 */}
        <div style={configRowStyle}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <Text style={{ fontSize: 12 }}>集合操作</Text>
            <Tooltip title={
              <div>
                <div><b>模式:</b> {COLLECTIVE_DESCRIPTIONS[currentConfig.collective].pattern}</div>
                <div><b>流量:</b> {COLLECTIVE_DESCRIPTIONS[currentConfig.collective].formula}</div>
                <div>{COLLECTIVE_DESCRIPTIONS[currentConfig.collective].desc}</div>
              </div>
            }>
              <QuestionCircleOutlined style={{ fontSize: 11, color: '#999', cursor: 'help' }} />
            </Tooltip>
          </div>
          <Select
            size="small"
            value={currentConfig.collective}
            onChange={(v: CollectiveType) => handleUpdateCurrentConfig({ collective: v })}
            style={{ width: 130 }}
            options={PARALLELISM_COLLECTIVE_OPTIONS[currentConfig.parallelism] || []}
          />
        </div>

        {/* 并行度 */}
        <div style={configRowStyle}>
          <Text style={{ fontSize: 12 }}>并行度</Text>
          <InputNumber
            size="small"
            min={2}
            max={64}
            value={currentConfig.size}
            onChange={(v) => handleUpdateCurrentConfig({ size: v || 2 })}
            style={{ width: 130 }}
          />
        </div>

        {/* 消息大小 */}
        <div style={configRowStyle}>
          <Text style={{ fontSize: 12 }}>消息大小 (MB)</Text>
          <InputNumber
            size="small"
            min={1}
            max={10000}
            value={currentConfig.message_size_mb}
            onChange={(v) => handleUpdateCurrentConfig({ message_size_mb: v || 100 })}
            style={{ width: 130 }}
          />
        </div>

        {/* Chip 范围 */}
        <div style={{ marginTop: 8 }}>
          <div style={configRowStyle}>
            <Text style={{ fontSize: 12 }}>Chip 范围</Text>
            <Select
              size="small"
              value={currentConfig.chip_scope}
              onChange={(v) => handleUpdateCurrentConfig({
                chip_scope: v,
                scope_pod_ids: [],
                scope_rack_ids: [],
                scope_board_ids: [],
                scope_chip_ids: [],
              })}
              style={{ width: 130 }}
              options={[
                { value: 'all', label: '全部' },
                { value: 'pod', label: 'Pod' },
                { value: 'rack', label: 'Rack' },
                { value: 'board', label: 'Board' },
                { value: 'chip', label: 'Chip' },
              ]}
            />
          </div>

          {currentConfig.chip_scope === 'pod' && topology && (
            <Select
              size="small"
              mode="multiple"
              value={currentConfig.scope_pod_ids || []}
              onChange={(v) => handleUpdateCurrentConfig({ scope_pod_ids: v })}
              style={{ width: '100%', marginTop: 4 }}
              placeholder="选择 Pod..."
              options={topology.pods.map(p => ({ value: p.id, label: p.label || p.id }))}
            />
          )}
          {currentConfig.chip_scope === 'rack' && topology && (
            <Select
              size="small"
              mode="multiple"
              value={currentConfig.scope_rack_ids || []}
              onChange={(v) => handleUpdateCurrentConfig({ scope_rack_ids: v })}
              style={{ width: '100%', marginTop: 4 }}
              placeholder="选择 Rack..."
              options={topology.pods.flatMap(p =>
                p.racks.map(r => ({ value: r.id, label: `${p.label || p.id} / ${r.label || r.id}` }))
              )}
            />
          )}
          {currentConfig.chip_scope === 'board' && topology && (
            <Select
              size="small"
              mode="multiple"
              value={currentConfig.scope_board_ids || []}
              onChange={(v) => handleUpdateCurrentConfig({ scope_board_ids: v })}
              style={{ width: '100%', marginTop: 4 }}
              placeholder="选择 Board..."
              options={topology.pods.flatMap(p =>
                p.racks.flatMap(r =>
                  r.boards.map(b => ({ value: b.id, label: `${p.label || p.id} / ${r.label || r.id} / ${b.label || b.id}` }))
                )
              )}
            />
          )}
          {currentConfig.chip_scope === 'chip' && topology && (
            <Select
              size="small"
              mode="multiple"
              value={currentConfig.scope_chip_ids || []}
              onChange={(v) => handleUpdateCurrentConfig({ scope_chip_ids: v })}
              style={{ width: '100%', marginTop: 4 }}
              placeholder="选择 Chip..."
              options={topology.pods.flatMap(p =>
                p.racks.flatMap(r =>
                  r.boards.flatMap(b =>
                    b.chips.map(c => ({ value: c.id, label: `${p.label || p.id} / ${r.label || r.id} / ${b.label || b.id} / ${c.label || c.id}` }))
                  )
                )
              )}
            />
          )}

          <Text type="secondary" style={{ fontSize: 11, display: 'block', marginTop: 4 }}>
            需要 {currentConfig.size} 个 Chip，范围内有 {getChipsInScope(currentConfig, topology, totalChips)} 个
          </Text>
        </div>

        {/* 添加/更新按钮 */}
        <Button
          type="primary"
          size="small"
          icon={editingExistingId ? undefined : <PlusOutlined />}
          onClick={handleAddConfig}
          disabled={!currentConfigValid}
          style={{ width: '100%', marginTop: 12 }}
        >
          {editingExistingId ? '更新配置' : '添加配置'}
        </Button>
      </div>

      {/* 已添加的配置列表 */}
      {configs.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <Text type="secondary" style={{ fontSize: 11, display: 'block', marginBottom: 4 }}>
            已添加 {configs.length} 个配置（双击编辑）
          </Text>
          {configs.map(config => {
            const chipsInScope = getChipsInScope(config, topology, totalChips)
            const isValid = config.size <= chipsInScope && config.size >= 2
            const isEditing = editingExistingId === config.id

            return (
              <Card
                key={config.id}
                size="small"
                style={{
                  marginBottom: 8,
                  border: isEditing ? '1px solid #1890ff' : '1px solid #d9d9d9',
                  cursor: 'pointer',
                }}
                bodyStyle={{ padding: 8 }}
                onDoubleClick={() => handleLoadConfig(config)}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <Tag color={PARALLELISM_COLORS[config.parallelism]}>{config.parallelism}</Tag>
                    <Text style={{ fontSize: 12 }}>{config.name}</Text>
                    <Text type="secondary" style={{ fontSize: 11 }}>×{config.size}</Text>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    {isEditing && <Tag color="processing" style={{ fontSize: 10 }}>编辑中</Tag>}
                    {!isValid && <Tag color="error" style={{ fontSize: 10 }}>无效</Tag>}
                    <Button
                      type="text"
                      size="small"
                      danger
                      icon={<DeleteOutlined />}
                      onClick={(e) => { e.stopPropagation(); handleDeleteConfig(config.id) }}
                    />
                  </div>
                </div>
                <div style={{ marginTop: 4, fontSize: 11, color: '#666' }}>
                  {config.collective} | {config.message_size_mb}MB | {config.chip_scope === 'all' ? '全部Chip' : config.chip_scope}
                </div>
              </Card>
            )
          })}
        </div>
      )}

      {/* 操作按钮 */}
      <Space style={{ width: '100%' }} direction="vertical">
        <Button
          type="primary"
          size="small"
          block
          disabled={!allValid}
          onClick={onRunAnalysis}
        >
          运行分析 ({configs.length} 个配置)
        </Button>
        {configs.length > 0 && (
          <Button
            size="small"
            block
            danger
            onClick={handleClearAll}
          >
            清空所有配置
          </Button>
        )}
      </Space>

      {/* 分析结果 */}
      {analysisResult && (
        <div style={{ marginTop: 12, padding: 10, background: '#e6f7ff', borderRadius: 8 }}>
          <Text strong style={{ fontSize: 12 }}>分析结果</Text>
          <div style={{ marginTop: 8, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
            <div>
              <Text type="secondary" style={{ fontSize: 11 }}>总流量</Text>
              <div style={{ fontSize: 14, fontWeight: 500 }}>
                {analysisResult.summary.total_traffic_mb.toFixed(1)} MB
              </div>
            </div>
            <div>
              <Text type="secondary" style={{ fontSize: 11 }}>最大链路流量</Text>
              <div style={{ fontSize: 14, fontWeight: 500 }}>
                {analysisResult.summary.max_link_traffic_mb.toFixed(1)} MB
              </div>
            </div>
            <div>
              <Text type="secondary" style={{ fontSize: 11 }}>平均带宽利用率</Text>
              <div style={{ fontSize: 14, fontWeight: 500 }}>
                {(analysisResult.summary.avg_bandwidth_utilization * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <Text type="secondary" style={{ fontSize: 11 }}>瓶颈链路</Text>
              <div style={{ fontSize: 14, fontWeight: 500, color: analysisResult.summary.bottleneck_links.length > 0 ? '#f5222d' : '#52c41a' }}>
                {analysisResult.summary.bottleneck_links.length} 条
              </div>
            </div>
          </div>

          {/* 各配置流量贡献 */}
          {analysisResult.summary.config_contributions && analysisResult.summary.config_contributions.length > 0 && (
            <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid #91d5ff' }}>
              <Text type="secondary" style={{ fontSize: 11, display: 'block', marginBottom: 4 }}>配置贡献</Text>
              {analysisResult.summary.config_contributions.map(c => (
                <div key={c.config_id} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11 }}>
                  <Text>{c.config_name}</Text>
                  <Text type="secondary">{c.traffic_mb.toFixed(1)} MB ({(c.percentage * 100).toFixed(0)}%)</Text>
                </div>
              ))}
            </div>
          )}

          {/* 通信组统计 */}
          <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid #91d5ff' }}>
            <Text type="secondary" style={{ fontSize: 11 }}>
              通信组: {(['DP', 'TP', 'PP', 'EP', 'SP'] as const)
                .map(type => {
                  const count = analysisResult.groups.filter(g => g.type === type).length
                  return count > 0 ? `${type}×${count}` : null
                })
                .filter(Boolean)
                .join(', ')}
            </Text>
          </div>

          {onNavigateToChips && (
            <Button
              type="link"
              size="small"
              onClick={onNavigateToChips}
              style={{ marginTop: 8, padding: 0 }}
            >
              查看芯片拓扑 →
            </Button>
          )}
        </div>
      )}
    </div>
  )
}
