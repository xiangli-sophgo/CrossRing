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
} from 'antd'
import {
  DeleteOutlined,
  PlusOutlined,
  MinusCircleOutlined,
} from '@ant-design/icons'
import {
  HierarchyLevelSwitchConfig, SwitchTypeConfig, SwitchLayerConfig,
  ManualConnectionConfig, ConnectionMode, SwitchConnectionMode, HierarchyLevel
} from '../../types'

const { Text } = Typography

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
  configRowStyle: React.CSSProperties
  currentLevel?: string
}

export const ConnectionEditPanel: React.FC<ConnectionEditPanelProps> = ({
  manualConnectionConfig,
  onManualConnectionConfigChange: _onManualConnectionConfigChange,
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
  configRowStyle: _configRowStyle,
  currentLevel = 'datacenter',
}) => {
  void _onManualConnectionConfigChange
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
  return (
    <div style={{
      padding: 14,
      background: '#f5f5f5',
      borderRadius: 10,
      border: '1px solid rgba(0, 0, 0, 0.06)',
    }}>
      <Text strong style={{ display: 'block', marginBottom: 10, color: '#171717' }}>连接编辑</Text>

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
      <Collapse
        size="small"
        style={{ marginTop: 8 }}
        items={[{
          key: 'manual',
          label: <span style={{ fontSize: 14 }}>手动连接 ({manualConnectionConfig?.connections?.length || 0})</span>,
          children: (
            <div style={{ maxHeight: 180, overflow: 'auto' }}>
              {manualConnectionConfig?.connections?.map((conn) => (
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
                        <Button
                          type="text"
                          danger
                          size="small"
                          icon={<DeleteOutlined />}
                          onClick={() => onDeleteManualConnection?.(conn.id)}
                        />
                      </div>
                      {(conn.bandwidth || conn.latency) && (
                        <div style={{ marginTop: 6, fontSize: 13, color: '#666' }}>
                          {conn.bandwidth && <span style={{ marginRight: 12 }}>带宽: {conn.bandwidth}Gbps</span>}
                          {conn.latency && <span>延迟: {conn.latency}ns</span>}
                        </div>
                      )}
                    </div>
                  ))}
                  {(!manualConnectionConfig?.connections || manualConnectionConfig.connections.length === 0) && (
                    <Text type="secondary" style={{ fontSize: 13 }}>暂无手动连接</Text>
                  )}
                </div>
              ),
            }, {
              key: 'current',
              label: <span style={{ fontSize: 14 }}>当前连接 ({currentViewConnections.length})</span>,
              children: (
                <div style={{ maxHeight: 180, overflow: 'auto' }}>
                  {currentViewConnections.map((conn, idx) => (
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
                        <Button
                          type="text"
                          danger
                          size="small"
                          icon={<DeleteOutlined />}
                          onClick={() => onDeleteConnection?.(conn.source, conn.target)}
                        />
                      </div>
                      {(conn.bandwidth || conn.latency) && (
                        <div style={{ marginTop: 6, fontSize: 13, color: '#666' }}>
                          {conn.bandwidth && <span style={{ marginRight: 12 }}>带宽: {conn.bandwidth}Gbps</span>}
                          {conn.latency && <span>延迟: {conn.latency}ns</span>}
                        </div>
                      )}
                    </div>
                  ))}
                  {currentViewConnections.length === 0 && (
                    <Text type="secondary" style={{ fontSize: 13 }}>暂无连接</Text>
                  )}
                </div>
              ),
            }]}
          />
    </div>
  )
}
