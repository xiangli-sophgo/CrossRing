import React, { useState, useEffect, useRef } from 'react'
import {
  Typography,
  Button,
  Space,
  Card,
  Statistic,
  Row,
  Col,
  InputNumber,
  Collapse,
  Modal,
  Input,
  message,
  Popconfirm,
  Switch,
  Select,
  Checkbox,
  Tabs,
  Divider,
  Radio,
} from 'antd'
import {
  ClusterOutlined,
  DatabaseOutlined,
  SaveOutlined,
  FolderOpenOutlined,
  DeleteOutlined,
  PlusOutlined,
  MinusCircleOutlined,
  ApartmentOutlined,
} from '@ant-design/icons'
import Icon from '@ant-design/icons'

// 自定义芯片图标 - 带引脚的芯片，中心白色
const ChipSvg = () => (
  <svg viewBox="0 0 100 100" width="1em" height="1em" fill="currentColor">
    {/* 芯片主体 */}
    <rect x="20" y="20" width="60" height="60" rx="4" fill="currentColor"/>
    {/* 中心白色区域 */}
    <rect x="30" y="30" width="40" height="40" rx="2" fill="white"/>
    {/* 上方引脚 */}
    <rect x="28" y="8" width="8" height="12" fill="currentColor"/>
    <rect x="46" y="8" width="8" height="12" fill="currentColor"/>
    <rect x="64" y="8" width="8" height="12" fill="currentColor"/>
    {/* 下方引脚 */}
    <rect x="28" y="80" width="8" height="12" fill="currentColor"/>
    <rect x="46" y="80" width="8" height="12" fill="currentColor"/>
    <rect x="64" y="80" width="8" height="12" fill="currentColor"/>
    {/* 左侧引脚 */}
    <rect x="8" y="28" width="12" height="8" fill="currentColor"/>
    <rect x="8" y="46" width="12" height="8" fill="currentColor"/>
    <rect x="8" y="64" width="12" height="8" fill="currentColor"/>
    {/* 右侧引脚 */}
    <rect x="80" y="28" width="12" height="8" fill="currentColor"/>
    <rect x="80" y="46" width="12" height="8" fill="currentColor"/>
    <rect x="80" y="64" width="12" height="8" fill="currentColor"/>
  </svg>
)
const ChipIcon = () => <Icon component={ChipSvg} />

// 自定义PCB板卡图标 - 带芯片和电路线（线框风格）
const BoardSvg = () => (
  <svg viewBox="0 0 100 80" width="1em" height="1em" fill="currentColor">
    {/* PCB主体边框 */}
    <rect x="5" y="5" width="90" height="70" rx="3" fill="none" stroke="currentColor" strokeWidth="3"/>
    {/* 芯片1 - 左上 */}
    <rect x="12" y="12" width="18" height="18" rx="2" fill="currentColor"/>
    {/* 芯片2 - 右上 */}
    <rect x="70" y="12" width="18" height="18" rx="2" fill="currentColor"/>
    {/* 芯片3 - 中下 */}
    <rect x="38" y="45" width="24" height="20" rx="2" fill="currentColor"/>
    {/* 电路线 */}
    <path d="M30 21 L70 21" stroke="currentColor" strokeWidth="2" fill="none"/>
    <path d="M12 40 L38 40 L38 50" stroke="currentColor" strokeWidth="2" fill="none"/>
    <path d="M62 55 L88 55 L88 30" stroke="currentColor" strokeWidth="2" fill="none"/>
    <path d="M21 30 L21 45 L38 45" stroke="currentColor" strokeWidth="2" fill="none"/>
    <path d="M79 30 L79 40 L62 40 L62 45" stroke="currentColor" strokeWidth="2" fill="none"/>
    {/* 连接点 */}
    <circle cx="10" cy="40" r="3" fill="currentColor"/>
    <circle cx="90" cy="40" r="3" fill="currentColor"/>
    <circle cx="50" cy="10" r="3" fill="currentColor"/>
    <circle cx="50" cy="70" r="3" fill="currentColor"/>
  </svg>
)
const BoardIcon = () => <Icon component={BoardSvg} />
import {
  HierarchicalTopology, CHIP_TYPE_COLORS, CHIP_TYPE_NAMES, ChipType,
  GlobalSwitchConfig, SwitchTypeConfig, SwitchLayerConfig, HierarchyLevelSwitchConfig,
  ManualConnectionConfig, ConnectionMode, SwitchConnectionMode, HierarchyLevel, LayoutType
} from '../types'
import { listConfigs, saveConfig, deleteConfig, SavedConfig } from '../api/topology'

const { Text } = Typography

interface ChipCounts {
  npu: number
  cpu: number
}

interface BoardTypeConfig {
  count: number
  chips: ChipCounts
}

interface BoardConfigs {
  u1: BoardTypeConfig
  u2: BoardTypeConfig
  u4: BoardTypeConfig
}

// 新的灵活板卡配置
interface FlexBoardChipConfig {
  name: string      // Chip名称，如 "NPU", "CPU", "GPU"
  count: number     // 数量
}

interface FlexBoardConfig {
  id: string            // 唯一ID
  name: string          // 板卡名称
  u_height: number      // U高度 (1-10)
  count: number         // 板卡数量
  chips: FlexBoardChipConfig[]  // Chip配置列表
}

interface RackConfig {
  total_u: number              // Rack总U数，默认42
  boards: FlexBoardConfig[]    // 板卡配置列表
}

// Switch 3D显示配置
export interface SwitchDisplayConfig {
  position: 'top' | 'middle' | 'bottom'
  uHeight: number
}

interface ConfigPanelProps {
  topology: HierarchicalTopology | null
  onGenerate: (config: {
    pod_count: number
    racks_per_pod: number
    board_configs: BoardConfigs
    rack_config?: RackConfig
    switch_config?: GlobalSwitchConfig
    manual_connections?: ManualConnectionConfig
  }) => void
  loading: boolean
  currentLevel?: 'datacenter' | 'pod' | 'rack' | 'board'
  // 编辑连接相关
  manualConnectionConfig?: ManualConnectionConfig
  onManualConnectionConfigChange?: (config: ManualConnectionConfig) => void
  connectionMode?: ConnectionMode
  onConnectionModeChange?: (mode: ConnectionMode) => void
  selectedNodes?: Set<string>
  onSelectedNodesChange?: (nodes: Set<string>) => void
  targetNodes?: Set<string>
  onTargetNodesChange?: (nodes: Set<string>) => void
  onBatchConnect?: (level: HierarchyLevel) => void
  onDeleteManualConnection?: (connectionId: string) => void
  currentViewConnections?: { source: string; target: string; type?: string; bandwidth?: number; latency?: number }[]  // 当前视图的连接
  onDeleteConnection?: (source: string, target: string) => void  // 删除连接
  // 布局相关
  layoutType?: LayoutType
  onLayoutTypeChange?: (type: LayoutType) => void
  viewMode?: '3d' | 'topology'
  // Switch 3D显示配置
  switchDisplayConfig?: SwitchDisplayConfig
  onSwitchDisplayConfigChange?: (config: SwitchDisplayConfig) => void
}

// localStorage缓存key
const CONFIG_CACHE_KEY = 'tier6_topology_config_cache'

// 默认配置
const DEFAULT_BOARD_CONFIGS: BoardConfigs = {
  u1: { count: 0, chips: { npu: 2, cpu: 0 } },
  u2: { count: 8, chips: { npu: 8, cpu: 0 } },
  u4: { count: 0, chips: { npu: 16, cpu: 2 } },
}

// 默认Rack配置
const DEFAULT_RACK_CONFIG: RackConfig = {
  total_u: 42,
  boards: [
    { id: 'board_1', name: '计算板卡', u_height: 2, count: 8, chips: [{ name: 'NPU', count: 8 }] },
  ],
}

// 默认Switch配置
const DEFAULT_SWITCH_CONFIG: GlobalSwitchConfig = {
  switch_types: [
    { id: 'leaf_48', name: '48端口Leaf交换机', port_count: 48 },
    { id: 'leaf_72', name: '72端口Leaf交换机', port_count: 72 },
    { id: 'spine_128', name: '128端口Spine交换机', port_count: 128 },
    { id: 'core_512', name: '512端口核心交换机', port_count: 512 },
  ],
  datacenter_level: { enabled: false, layers: [], downlink_redundancy: 1, connect_to_upper_level: true, keep_direct_topology: false },
  pod_level: { enabled: false, layers: [], downlink_redundancy: 1, connect_to_upper_level: true, keep_direct_topology: false },
  rack_level: { enabled: false, layers: [], downlink_redundancy: 1, connect_to_upper_level: true, switch_position: 'top', switch_u_height: 1, keep_direct_topology: false },
  board_level: { enabled: false, layers: [], downlink_redundancy: 1, connect_to_upper_level: true, keep_direct_topology: false },
}

// 从localStorage加载缓存配置
const loadCachedConfig = () => {
  try {
    const cached = localStorage.getItem(CONFIG_CACHE_KEY)
    if (cached) {
      return JSON.parse(cached)
    }
  } catch (error) {
    console.error('加载缓存配置失败:', error)
  }
  return null
}

// 保存配置到localStorage
const saveCachedConfig = (config: {
  podCount: number
  racksPerPod: number
  boardConfigs: BoardConfigs
  rackConfig?: RackConfig
  switchConfig?: GlobalSwitchConfig
  manualConnectionConfig?: ManualConnectionConfig
}) => {
  try {
    localStorage.setItem(CONFIG_CACHE_KEY, JSON.stringify(config))
  } catch (error) {
    console.error('缓存配置失败:', error)
  }
}

// Switch层级配置子组件
interface SwitchLevelConfigProps {
  levelKey: string
  config: HierarchyLevelSwitchConfig
  switchTypes: SwitchTypeConfig[]
  onChange: (config: HierarchyLevelSwitchConfig) => void
  configRowStyle: React.CSSProperties
  viewMode?: '3d' | 'topology'
}

const SwitchLevelConfig: React.FC<SwitchLevelConfigProps> = ({
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
          {levelKey === 'rack_level' && viewMode === '3d' && (
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

// 连接编辑子组件
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

const ConnectionEditPanel: React.FC<ConnectionEditPanelProps> = ({
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
  configRowStyle,
  currentLevel = 'datacenter',
}) => {
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

export const ConfigPanel: React.FC<ConfigPanelProps> = ({
  topology,
  onGenerate,
  currentLevel = 'datacenter',
  // 手动连线相关
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
  layoutType = 'auto',
  onLayoutTypeChange,
  viewMode = 'topology',
}) => {
  // 从缓存加载初始配置
  const cachedConfig = loadCachedConfig()

  // Pod层级配置
  const [podCount, setPodCount] = useState(cachedConfig?.podCount ?? 1)

  // Rack层级配置
  const [racksPerPod, setRacksPerPod] = useState(cachedConfig?.racksPerPod ?? 4)

  // Board配置（按U高度分类，每种类型有独立的chip配置）- 旧格式，保持兼容
  const [boardConfigs, setBoardConfigs] = useState<BoardConfigs>(
    cachedConfig?.boardConfigs ?? DEFAULT_BOARD_CONFIGS
  )

  // 新的灵活Rack配置
  const [rackConfig, setRackConfig] = useState<RackConfig>(
    cachedConfig?.rackConfig ?? DEFAULT_RACK_CONFIG
  )

  // Rack配置编辑模式
  const [rackEditMode, setRackEditMode] = useState(false)

  // Switch配置（深度合并默认值以兼容旧缓存）
  const [switchConfig, setSwitchConfig] = useState<GlobalSwitchConfig>(() => {
    if (cachedConfig?.switchConfig) {
      // 深度合并各层级配置，确保新字段有默认值
      const merged = { ...DEFAULT_SWITCH_CONFIG }
      if (cachedConfig.switchConfig.switch_types) {
        merged.switch_types = cachedConfig.switchConfig.switch_types
      }
      // 合并各层级配置，过滤掉无效字段
      const mergeLevel = (defaultLevel: any, cachedLevel: any) => {
        if (!cachedLevel) return defaultLevel
        return {
          enabled: cachedLevel.enabled ?? defaultLevel.enabled,
          layers: cachedLevel.layers ?? defaultLevel.layers,
          downlink_redundancy: cachedLevel.downlink_redundancy ?? defaultLevel.downlink_redundancy,
          connect_to_upper_level: cachedLevel.connect_to_upper_level ?? defaultLevel.connect_to_upper_level,
          direct_topology: cachedLevel.direct_topology ?? defaultLevel.direct_topology,
          keep_direct_topology: cachedLevel.keep_direct_topology ?? defaultLevel.keep_direct_topology,
          connection_mode: cachedLevel.connection_mode ?? defaultLevel.connection_mode,
          group_config: cachedLevel.group_config ?? defaultLevel.group_config,
          custom_connections: cachedLevel.custom_connections ?? defaultLevel.custom_connections,
          // 只保留正确的字段名
          switch_position: cachedLevel.switch_position ?? defaultLevel.switch_position,
          switch_u_height: cachedLevel.switch_u_height ?? defaultLevel.switch_u_height,
        }
      }
      merged.datacenter_level = mergeLevel(DEFAULT_SWITCH_CONFIG.datacenter_level, cachedConfig.switchConfig.datacenter_level)
      merged.pod_level = mergeLevel(DEFAULT_SWITCH_CONFIG.pod_level, cachedConfig.switchConfig.pod_level)
      merged.rack_level = mergeLevel(DEFAULT_SWITCH_CONFIG.rack_level, cachedConfig.switchConfig.rack_level)
      merged.board_level = mergeLevel(DEFAULT_SWITCH_CONFIG.board_level, cachedConfig.switchConfig.board_level)
      return merged
    }
    return DEFAULT_SWITCH_CONFIG
  })

  // 保存/加载配置状态
  const [savedConfigs, setSavedConfigs] = useState<SavedConfig[]>([])
  const [saveModalOpen, setSaveModalOpen] = useState(false)
  const [loadModalOpen, setLoadModalOpen] = useState(false)
  const [configName, setConfigName] = useState('')
  const [configDesc, setConfigDesc] = useState('')

  // 加载配置列表
  const loadConfigList = async () => {
    try {
      const configs = await listConfigs()
      setSavedConfigs(configs)
    } catch (error) {
      console.error('加载配置列表失败:', error)
    }
  }

  useEffect(() => {
    loadConfigList()
  }, [])

  // 配置变化时自动保存到localStorage
  useEffect(() => {
    saveCachedConfig({ podCount, racksPerPod, boardConfigs, rackConfig, switchConfig, manualConnectionConfig })
  }, [podCount, racksPerPod, boardConfigs, rackConfig, switchConfig, manualConnectionConfig])

  // 配置变化时自动生成拓扑（防抖500ms）
  const isFirstRender = useRef(true)
  useEffect(() => {
    // 跳过首次渲染（避免页面加载时重复生成）
    if (isFirstRender.current) {
      isFirstRender.current = false
      return
    }

    const timer = setTimeout(() => {
      onGenerate({
        pod_count: podCount,
        racks_per_pod: racksPerPod,
        board_configs: boardConfigs,
        rack_config: rackConfig,
        switch_config: switchConfig,
        manual_connections: manualConnectionConfig,
      })
    }, 500)

    return () => clearTimeout(timer)
  }, [podCount, racksPerPod, boardConfigs, rackConfig, switchConfig, manualConnectionConfig, onGenerate])

  // 保存当前配置
  const handleSaveConfig = async () => {
    if (!configName.trim()) {
      message.error('请输入配置名称')
      return
    }
    try {
      await saveConfig({
        name: configName.trim(),
        description: configDesc.trim() || undefined,
        pod_count: podCount,
        racks_per_pod: racksPerPod,
        board_configs: boardConfigs,
      })
      message.success('配置保存成功')
      setSaveModalOpen(false)
      setConfigName('')
      setConfigDesc('')
      loadConfigList()
    } catch (error) {
      console.error('保存配置失败:', error)
      message.error('保存配置失败')
    }
  }

  // 加载指定配置
  const handleLoadConfig = (config: SavedConfig) => {
    setPodCount(config.pod_count)
    setRacksPerPod(config.racks_per_pod)
    setBoardConfigs(config.board_configs)
    setLoadModalOpen(false)
    message.success(`已加载配置: ${config.name}`)
  }

  // 删除配置
  const handleDeleteConfig = async (name: string) => {
    try {
      await deleteConfig(name)
      message.success('配置已删除')
      loadConfigList()
    } catch (error) {
      console.error('删除配置失败:', error)
      message.error('删除配置失败')
    }
  }

  // 计算总占用U数
  const totalUsedU = boardConfigs.u1.count * 1 + boardConfigs.u2.count * 2 + boardConfigs.u4.count * 4

  // 计算统计数据
  const stats = {
    pods: topology?.pods.length || 0,
    racks: topology?.pods.reduce((sum, p) => sum + p.racks.length, 0) || 0,
    boards: topology?.pods.reduce((sum, p) =>
      sum + p.racks.reduce((s, r) => s + r.boards.length, 0), 0) || 0,
    chips: topology?.pods.reduce((sum, p) =>
      sum + p.racks.reduce((s, r) =>
        s + r.boards.reduce((b, board) => b + board.chips.length, 0), 0), 0) || 0,
    switches: topology?.switches?.length || 0,
  }

  const updateBoardCount = (uSize: keyof BoardConfigs, value: number | null) => {
    setBoardConfigs(prev => ({
      ...prev,
      [uSize]: { ...prev[uSize], count: value || 0 }
    }))
  }

  const updateBoardChip = (uSize: keyof BoardConfigs, chipType: keyof ChipCounts, value: number | null) => {
    setBoardConfigs(prev => ({
      ...prev,
      [uSize]: {
        ...prev[uSize],
        chips: { ...prev[uSize].chips, [chipType]: value || 0 }
      }
    }))
  }

  // 配置项样式
  const configRowStyle: React.CSSProperties = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  }

  // 层级配置Tab key
  const [layerTabKey, setLayerTabKey] = useState<string>(currentLevel === 'datacenter' ? 'datacenter' : currentLevel)

  // 当右边层级变化时，同步层级配置Tab
  useEffect(() => {
    setLayerTabKey(currentLevel === 'datacenter' ? 'datacenter' : currentLevel)
  }, [currentLevel])

  // 汇总信息
  const summaryText = topology
    ? `${stats.pods}Pod ${stats.racks}Rack ${stats.boards}Board ${stats.chips}Chip`
    : '未生成'

  // 拓扑配置内容（统计信息）
  const topologyConfigContent = (
    <Row gutter={[8, 8]}>
      <Col span={12}>
        <Statistic
          title="Pods"
          value={stats.pods}
          prefix={<ClusterOutlined />}
          valueStyle={{ fontSize: 16 }}
        />
      </Col>
      <Col span={12}>
        <Statistic
          title="Racks"
          value={stats.racks}
          prefix={<DatabaseOutlined />}
          valueStyle={{ fontSize: 16 }}
        />
      </Col>
      <Col span={12}>
        <Statistic
          title="Boards"
          value={stats.boards}
          prefix={<BoardIcon />}
          valueStyle={{ fontSize: 16 }}
        />
      </Col>
      <Col span={12}>
        <Statistic
          title="Chips"
          value={stats.chips}
          prefix={<ChipIcon />}
          valueStyle={{ fontSize: 16 }}
        />
      </Col>
      {stats.switches > 0 && (
        <Col span={24}>
          <Statistic
            title="Switches"
            value={stats.switches}
            prefix={<ApartmentOutlined />}
            valueStyle={{ fontSize: 16 }}
          />
        </Col>
      )}
    </Row>
  )


  // 层级配置内容（节点配置 + Switch连接配置）
  const layerConfigContent = (
    <Tabs
      size="small"
      type="card"
      activeKey={layerTabKey}
      onChange={setLayerTabKey}
      items={[
        {
          key: 'datacenter',
          label: '数据中心层',
          children: (
            <div>
              {/* Pod数量配置 */}
              <div style={{
                marginBottom: 12,
                padding: 14,
                background: '#f5f5f5',
                borderRadius: 10,
                border: '1px solid rgba(0, 0, 0, 0.06)',
              }}>
                <Text strong style={{ display: 'block', marginBottom: 10, color: '#171717' }}>节点配置</Text>
                <div style={configRowStyle}>
                  <Text>Pod 数量</Text>
                  <InputNumber
                    min={1}
                    max={10}
                    value={podCount}
                    onChange={(v) => setPodCount(v || 1)}
                    size="small"
                    style={{ width: 80 }}
                  />
                </div>
              </div>
              {/* Pod间连接配置 */}
              <div style={{
                padding: 14,
                background: '#f5f5f5',
                borderRadius: 10,
                border: '1px solid rgba(0, 0, 0, 0.06)',
              }}>
                <Text strong style={{ display: 'block', marginBottom: 10, color: '#171717' }}>连接配置</Text>
                <SwitchLevelConfig
                  levelKey="datacenter_level"
                  config={switchConfig.datacenter_level}
                  switchTypes={switchConfig.switch_types}
                  onChange={(newConfig) => setSwitchConfig(prev => ({ ...prev, datacenter_level: newConfig }))}
                  configRowStyle={configRowStyle}
                />
              </div>
              {/* 连接编辑（当前层级时显示） */}
              {currentLevel === 'datacenter' && (
                <div style={{ marginTop: 12 }}>
                  <ConnectionEditPanel
                    manualConnectionConfig={manualConnectionConfig}
                    onManualConnectionConfigChange={onManualConnectionConfigChange}
                    connectionMode={connectionMode}
                    onConnectionModeChange={onConnectionModeChange}
                    selectedNodes={selectedNodes}
                    onSelectedNodesChange={onSelectedNodesChange}
                    targetNodes={targetNodes}
                    onTargetNodesChange={onTargetNodesChange}
                    onBatchConnect={onBatchConnect}
                    onDeleteManualConnection={onDeleteManualConnection}
                    currentViewConnections={currentViewConnections}
                    onDeleteConnection={onDeleteConnection}
                    configRowStyle={configRowStyle}
                    currentLevel={currentLevel}
                  />
                </div>
              )}
            </div>
          ),
        },
        {
          key: 'pod',
          label: 'Pod层',
          children: (
            <div>
              {/* Rack数量配置 */}
              <div style={{
                marginBottom: 12,
                padding: 14,
                background: '#f5f5f5',
                borderRadius: 10,
                border: '1px solid rgba(0, 0, 0, 0.06)',
              }}>
                <Text strong style={{ display: 'block', marginBottom: 10, color: '#171717' }}>节点配置</Text>
                <div style={configRowStyle}>
                  <Text>每Pod机柜数</Text>
                  <InputNumber
                    min={1}
                    max={64}
                    value={racksPerPod}
                    onChange={(v) => setRacksPerPod(v || 1)}
                    size="small"
                    style={{ width: 80 }}
                  />
                </div>
              </div>
              {/* Rack间连接配置 */}
              <div style={{
                padding: 14,
                background: '#f5f5f5',
                borderRadius: 10,
                border: '1px solid rgba(0, 0, 0, 0.06)',
              }}>
                <Text strong style={{ display: 'block', marginBottom: 10, color: '#171717' }}>连接配置</Text>
                <SwitchLevelConfig
                  levelKey="pod_level"
                  config={switchConfig.pod_level}
                  switchTypes={switchConfig.switch_types}
                  onChange={(newConfig) => setSwitchConfig(prev => ({ ...prev, pod_level: newConfig }))}
                  configRowStyle={configRowStyle}
                />
              </div>
              {/* 连接编辑（当前层级时显示） */}
              {currentLevel === 'pod' && (
                <div style={{ marginTop: 12 }}>
                  <ConnectionEditPanel
                    manualConnectionConfig={manualConnectionConfig}
                    onManualConnectionConfigChange={onManualConnectionConfigChange}
                    connectionMode={connectionMode}
                    onConnectionModeChange={onConnectionModeChange}
                    selectedNodes={selectedNodes}
                    onSelectedNodesChange={onSelectedNodesChange}
                    targetNodes={targetNodes}
                    onTargetNodesChange={onTargetNodesChange}
                    onBatchConnect={onBatchConnect}
                    onDeleteManualConnection={onDeleteManualConnection}
                    currentViewConnections={currentViewConnections}
                    onDeleteConnection={onDeleteConnection}
                    configRowStyle={configRowStyle}
                    currentLevel={currentLevel}
                  />
                </div>
              )}
            </div>
          ),
        },
        {
          key: 'rack',
          label: 'Rack层',
          children: (
            <div>
              {/* Board配置 */}
              <div style={{
                marginBottom: 12,
                padding: 14,
                background: 'linear-gradient(135deg, rgba(248, 250, 252, 0.8) 0%, rgba(241, 245, 249, 0.8) 100%)',
                borderRadius: 12,
                border: '1px solid rgba(0, 0, 0, 0.04)',
              }}>
                {/* 标题和编辑开关 */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <Text strong>节点配置</Text>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Text type="secondary" style={{ fontSize: 11 }}>编辑</Text>
                    <Switch
                      size="small"
                      checked={rackEditMode}
                      onChange={setRackEditMode}
                    />
                  </div>
                </div>

                {/* 汇总信息 */}
                {(() => {
                  const usedU = rackConfig.boards.reduce((sum, b) => sum + b.u_height * (b.count || 1), 0)
                  const totalBoards = rackConfig.boards.reduce((sum, b) => sum + (b.count || 1), 0)
                  const totalChips = rackConfig.boards.reduce((sum, b) => sum + (b.count || 1) * b.chips.reduce((s, c) => s + c.count, 0), 0)
                  const isOverflow = usedU > rackConfig.total_u
                  return (
                    <div style={{ marginBottom: 8, fontSize: 12, color: '#666' }}>
                      <span>容量: <Text strong>{rackConfig.total_u}U</Text></span>
                      <span style={{ margin: '0 8px', color: '#d9d9d9' }}>|</span>
                      <span>已用: <Text strong type={isOverflow ? 'danger' : undefined}>{usedU}U</Text></span>
                      <span style={{ margin: '0 8px', color: '#d9d9d9' }}>|</span>
                      <span>板卡: <Text strong>{totalBoards}</Text></span>
                      <span style={{ margin: '0 8px', color: '#d9d9d9' }}>|</span>
                      <span>芯片: <Text strong>{totalChips}</Text></span>
                    </div>
                  )
                })()}

                {/* 编辑模式：Rack容量 */}
                {rackEditMode && (
                  <div style={configRowStyle}>
                    <Text>Rack容量</Text>
                    <InputNumber
                      min={10}
                      max={60}
                      value={rackConfig.total_u}
                      onChange={(v) => setRackConfig(prev => ({ ...prev, total_u: v || 42 }))}
                      size="small"
                      style={{ width: 70 }}
                      addonAfter="U"
                    />
                  </div>
                )}

                {/* 板卡列表 */}
                <div style={{ marginTop: 8 }}>
                  {rackConfig.boards.map((board, boardIndex) => (
                    <div key={board.id} style={{ marginBottom: 6, padding: '6px 10px', background: '#fff', borderRadius: 4, border: '1px solid #e8e8e8' }}>
                      {rackEditMode ? (
                        /* 编辑模式 */
                        <>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <Text style={{ fontSize: 12, whiteSpace: 'nowrap' }}>名称:</Text>
                              <Input
                                size="small"
                                value={board.name}
                                onChange={(e) => {
                                  const newBoards = [...rackConfig.boards]
                                  newBoards[boardIndex] = { ...newBoards[boardIndex], name: e.target.value }
                                  setRackConfig(prev => ({ ...prev, boards: newBoards }))
                                }}
                                style={{ width: 120 }}
                              />
                              <Text style={{ fontSize: 12, marginLeft: 8, whiteSpace: 'nowrap' }}>高度:</Text>
                              <InputNumber
                                size="small"
                                min={1}
                                max={10}
                                value={board.u_height}
                                onChange={(v) => {
                                  const newBoards = [...rackConfig.boards]
                                  newBoards[boardIndex] = { ...newBoards[boardIndex], u_height: v || 1 }
                                  setRackConfig(prev => ({ ...prev, boards: newBoards }))
                                }}
                                style={{ width: 70 }}
                                addonAfter="U"
                              />
                              <Text style={{ fontSize: 12, marginLeft: 8, whiteSpace: 'nowrap' }}>数量:</Text>
                              <InputNumber
                                size="small"
                                min={0}
                                max={42}
                                value={board.count || 1}
                                onChange={(v) => {
                                  const newBoards = [...rackConfig.boards]
                                  newBoards[boardIndex] = { ...newBoards[boardIndex], count: v || 0 }
                                  setRackConfig(prev => ({ ...prev, boards: newBoards }))
                                }}
                                style={{ width: 60 }}
                              />
                            </div>
                            <Button
                              type="text"
                              danger
                              size="small"
                              icon={<MinusCircleOutlined />}
                              onClick={() => {
                                const newBoards = rackConfig.boards.filter((_, i) => i !== boardIndex)
                                setRackConfig(prev => ({ ...prev, boards: newBoards }))
                              }}
                              disabled={rackConfig.boards.length <= 1}
                            />
                          </div>
                        </>
                      ) : (
                        /* 展示模式 */
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Text style={{ fontSize: 13 }}>{board.name} ×{board.count || 1}</Text>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                            <Text type="secondary" style={{ fontSize: 12 }}>{board.u_height}U</Text>
                            <Text type="secondary" style={{ fontSize: 12 }}>
                              {board.chips.map(c => `${c.name}×${c.count}`).join(' ')}
                            </Text>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                {/* 编辑模式：添加板卡按钮 */}
                {rackEditMode && (
                  <Button
                    type="dashed"
                    size="small"
                    icon={<PlusOutlined />}
                    onClick={() => {
                      const newBoard: FlexBoardConfig = {
                        id: `board_${Date.now()}`,
                        name: '新板卡',
                        u_height: 2,
                        count: 1,
                        chips: [{ name: 'NPU', count: 8 }],
                      }
                      setRackConfig(prev => ({ ...prev, boards: [...prev.boards, newBoard] }))
                    }}
                    style={{ width: '100%', marginTop: 4 }}
                  >
                    添加板卡类型
                  </Button>
                )}
              </div>

              {/* Board间连接配置 */}
              <div style={{
                padding: 14,
                background: '#f5f5f5',
                borderRadius: 10,
                border: '1px solid rgba(0, 0, 0, 0.06)',
              }}>
                <Text strong style={{ display: 'block', marginBottom: 10, color: '#171717' }}>连接配置</Text>
                <SwitchLevelConfig
                  levelKey="rack_level"
                  config={switchConfig.rack_level}
                  switchTypes={switchConfig.switch_types}
                  onChange={(newConfig) => setSwitchConfig(prev => ({ ...prev, rack_level: newConfig }))}
                  configRowStyle={configRowStyle}
                  viewMode={viewMode}
                />
              </div>
              {/* 连接编辑（当前层级时显示） */}
              {currentLevel === 'rack' && (
                <div style={{ marginTop: 12 }}>
                  <ConnectionEditPanel
                    manualConnectionConfig={manualConnectionConfig}
                    onManualConnectionConfigChange={onManualConnectionConfigChange}
                    connectionMode={connectionMode}
                    onConnectionModeChange={onConnectionModeChange}
                    selectedNodes={selectedNodes}
                    onSelectedNodesChange={onSelectedNodesChange}
                    targetNodes={targetNodes}
                    onTargetNodesChange={onTargetNodesChange}
                    onBatchConnect={onBatchConnect}
                    onDeleteManualConnection={onDeleteManualConnection}
                    currentViewConnections={currentViewConnections}
                    onDeleteConnection={onDeleteConnection}
                    configRowStyle={configRowStyle}
                    currentLevel={currentLevel}
                  />
                </div>
              )}
            </div>
          ),
        },
        {
          key: 'board',
          label: 'Board层',
          children: (
            <div>
              {/* 芯片配置 */}
              <div style={{
                marginBottom: 12,
                padding: 14,
                background: '#f5f5f5',
                borderRadius: 10,
                border: '1px solid rgba(0, 0, 0, 0.06)',
              }}>
                <Text strong style={{ display: 'block', marginBottom: 10, color: '#171717' }}>芯片配置</Text>
                <Text type="secondary" style={{ fontSize: 11, marginBottom: 10, display: 'block' }}>
                  为每种板卡类型配置芯片
                </Text>
                {rackConfig.boards.map((board, boardIndex) => (
                  <div key={board.id} style={{ marginBottom: 10, padding: '8px 10px', background: '#fff', borderRadius: 6, border: '1px solid #e8e8e8' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                      <Text strong style={{ fontSize: 12 }}>{board.name}</Text>
                      <Button
                        type="dashed"
                        size="small"
                        icon={<PlusOutlined />}
                        onClick={() => {
                          const newBoards = [...rackConfig.boards]
                          const newChips = [...newBoards[boardIndex].chips, { name: 'CPU', count: 2 }]
                          newBoards[boardIndex] = { ...newBoards[boardIndex], chips: newChips }
                          setRackConfig(prev => ({ ...prev, boards: newBoards }))
                        }}
                      >
                        添加芯片
                      </Button>
                    </div>
                    {board.chips.map((chip, chipIndex) => (
                      <div key={chipIndex} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                        <Text style={{ fontSize: 12, whiteSpace: 'nowrap' }}>类型:</Text>
                        <Input
                          size="small"
                          value={chip.name}
                          onChange={(e) => {
                            const newBoards = [...rackConfig.boards]
                            const newChips = [...newBoards[boardIndex].chips]
                            newChips[chipIndex] = { ...newChips[chipIndex], name: e.target.value }
                            newBoards[boardIndex] = { ...newBoards[boardIndex], chips: newChips }
                            setRackConfig(prev => ({ ...prev, boards: newBoards }))
                          }}
                          style={{ width: 80 }}
                        />
                        <Text style={{ fontSize: 12, whiteSpace: 'nowrap' }}>数量:</Text>
                        <InputNumber
                          size="small"
                          min={1}
                          max={64}
                          value={chip.count}
                          onChange={(v) => {
                            const newBoards = [...rackConfig.boards]
                            const newChips = [...newBoards[boardIndex].chips]
                            newChips[chipIndex] = { ...newChips[chipIndex], count: v || 1 }
                            newBoards[boardIndex] = { ...newBoards[boardIndex], chips: newChips }
                            setRackConfig(prev => ({ ...prev, boards: newBoards }))
                          }}
                          style={{ width: 60 }}
                        />
                        <Button
                          type="text"
                          danger
                          size="small"
                          icon={<MinusCircleOutlined />}
                          onClick={() => {
                            const newBoards = [...rackConfig.boards]
                            const newChips = newBoards[boardIndex].chips.filter((_, i) => i !== chipIndex)
                            newBoards[boardIndex] = { ...newBoards[boardIndex], chips: newChips }
                            setRackConfig(prev => ({ ...prev, boards: newBoards }))
                          }}
                          disabled={board.chips.length <= 1}
                        />
                      </div>
                    ))}
                  </div>
                ))}
              </div>

              {/* Chip间连接配置 */}
              <div style={{
                padding: 14,
                background: '#f5f5f5',
                borderRadius: 10,
                border: '1px solid rgba(0, 0, 0, 0.06)',
              }}>
                <Text strong style={{ display: 'block', marginBottom: 10, color: '#171717' }}>连接配置</Text>
                <SwitchLevelConfig
                  levelKey="board_level"
                  config={switchConfig.board_level}
                  switchTypes={switchConfig.switch_types}
                  onChange={(newConfig) => setSwitchConfig(prev => ({ ...prev, board_level: newConfig }))}
                  configRowStyle={configRowStyle}
                />
              </div>
              {/* 连接编辑（当前层级时显示） */}
              {currentLevel === 'board' && (
                <div style={{ marginTop: 12 }}>
                  <ConnectionEditPanel
                    manualConnectionConfig={manualConnectionConfig}
                    onManualConnectionConfigChange={onManualConnectionConfigChange}
                    connectionMode={connectionMode}
                    onConnectionModeChange={onConnectionModeChange}
                    selectedNodes={selectedNodes}
                    onSelectedNodesChange={onSelectedNodesChange}
                    targetNodes={targetNodes}
                    onTargetNodesChange={onTargetNodesChange}
                    onBatchConnect={onBatchConnect}
                    onDeleteManualConnection={onDeleteManualConnection}
                    currentViewConnections={currentViewConnections}
                    onDeleteConnection={onDeleteConnection}
                    configRowStyle={configRowStyle}
                    currentLevel={currentLevel}
                  />
                </div>
              )}
            </div>
          ),
        },
      ]}
    />
  )

  // Switch配置内容（只有Switch类型定义）
  const switchConfigContent = (
    <div>
      <Text type="secondary" style={{ fontSize: 11, display: 'block', marginBottom: 8 }}>
        定义可用的Switch型号，在各层级的连接配置中使用
      </Text>
      {switchConfig.switch_types.map((swType, index) => (
        <div key={swType.id} style={{ marginBottom: 8, padding: 8, background: '#f5f5f5', borderRadius: 4 }}>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <Input
              size="small"
              placeholder="名称"
              value={swType.name}
              onChange={(e) => {
                const newTypes = [...switchConfig.switch_types]
                newTypes[index] = { ...newTypes[index], name: e.target.value }
                setSwitchConfig(prev => ({ ...prev, switch_types: newTypes }))
              }}
              style={{ flex: 1 }}
            />
            <InputNumber
              size="small"
              min={8}
              max={1024}
              value={swType.port_count}
              onChange={(v) => {
                const newTypes = [...switchConfig.switch_types]
                newTypes[index] = { ...newTypes[index], port_count: v || 48 }
                setSwitchConfig(prev => ({ ...prev, switch_types: newTypes }))
              }}
              style={{ width: 100 }}
              suffix="端口"
            />
            <Button
              type="text"
              danger
              size="small"
              icon={<MinusCircleOutlined />}
              disabled={switchConfig.switch_types.length <= 1}
              onClick={() => {
                const newTypes = switchConfig.switch_types.filter((_, i) => i !== index)
                setSwitchConfig(prev => ({ ...prev, switch_types: newTypes }))
              }}
            />
          </div>
        </div>
      ))}
      <Button
        type="dashed"
        size="small"
        icon={<PlusOutlined />}
        onClick={() => {
          const newId = `switch_${Date.now()}`
          const newTypes = [...switchConfig.switch_types, { id: newId, name: '新Switch', port_count: 48 }]
          setSwitchConfig(prev => ({ ...prev, switch_types: newTypes }))
        }}
        style={{ width: '100%' }}
      >
        添加Switch类型
      </Button>
    </div>
  )

  const collapseItems = [
    {
      key: 'topology',
      label: (
        <span>
          <Text strong>拓扑汇总</Text>
          <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
            | {summaryText}
          </Text>
        </span>
      ),
      children: topologyConfigContent,
    },
    {
      key: 'layers',
      label: <Text strong>层级配置</Text>,
      children: layerConfigContent,
    },
    {
      key: 'switch',
      label: <Text strong>Switch配置</Text>,
      children: switchConfigContent,
    },
  ]

  return (
    <div>
      {/* 折叠面板 */}
      <Collapse
        items={collapseItems}
        defaultActiveKey={['layers']}
        size="small"
      />

      {/* 保存/加载配置按钮 */}
      <Row gutter={8} style={{ marginTop: 16 }}>
        <Col span={12}>
          <Button
            block
            icon={<SaveOutlined />}
            onClick={() => setSaveModalOpen(true)}
          >
            保存配置
          </Button>
        </Col>
        <Col span={12}>
          <Button
            block
            icon={<FolderOpenOutlined />}
            onClick={() => {
              loadConfigList()
              setLoadModalOpen(true)
            }}
          >
            加载配置
          </Button>
        </Col>
      </Row>

      {/* 保存配置模态框 */}
      <Modal
        title="保存配置"
        open={saveModalOpen}
        onOk={handleSaveConfig}
        onCancel={() => {
          setSaveModalOpen(false)
          setConfigName('')
          setConfigDesc('')
        }}
        okText="保存"
        cancelText="取消"
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <Text>配置名称 *</Text>
            <Input
              placeholder="输入配置名称"
              value={configName}
              onChange={(e) => setConfigName(e.target.value)}
              style={{ marginTop: 4 }}
            />
          </div>
          <div>
            <Text>描述 (可选)</Text>
            <Input.TextArea
              placeholder="输入配置描述"
              value={configDesc}
              onChange={(e) => setConfigDesc(e.target.value)}
              rows={2}
              style={{ marginTop: 4 }}
            />
          </div>
          {savedConfigs.some(c => c.name === configName.trim()) && (
            <Text type="warning" style={{ fontSize: 12 }}>
              同名配置已存在，保存将覆盖原配置
            </Text>
          )}
        </Space>
      </Modal>

      {/* 加载配置模态框 */}
      <Modal
        title="加载配置"
        open={loadModalOpen}
        onCancel={() => setLoadModalOpen(false)}
        footer={null}
        width={480}
      >
        {savedConfigs.length === 0 ? (
          <Text type="secondary">暂无保存的配置</Text>
        ) : (
          <Space direction="vertical" style={{ width: '100%' }}>
            {savedConfigs.map(config => (
              <Card
                key={config.name}
                size="small"
                style={{ cursor: 'pointer' }}
                hoverable
                onClick={() => handleLoadConfig(config)}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div style={{ flex: 1 }}>
                    <Text strong>{config.name}</Text>
                    {config.description && (
                      <div><Text type="secondary" style={{ fontSize: 12 }}>{config.description}</Text></div>
                    )}
                    <div style={{ marginTop: 4 }}>
                      <Text type="secondary" style={{ fontSize: 11 }}>
                        Pod:{config.pod_count} | Rack:{config.racks_per_pod} |
                        1U:{config.board_configs.u1.count} 2U:{config.board_configs.u2.count} 4U:{config.board_configs.u4.count}
                      </Text>
                    </div>
                  </div>
                  <Popconfirm
                    title="确定删除此配置？"
                    onConfirm={(e) => {
                      e?.stopPropagation()
                      handleDeleteConfig(config.name)
                    }}
                    onCancel={(e) => e?.stopPropagation()}
                    okText="删除"
                    cancelText="取消"
                  >
                    <Button
                      type="text"
                      danger
                      size="small"
                      icon={<DeleteOutlined />}
                      onClick={(e) => e.stopPropagation()}
                    />
                  </Popconfirm>
                </div>
              </Card>
            ))}
          </Space>
        )}
      </Modal>
    </div>
  )
}
