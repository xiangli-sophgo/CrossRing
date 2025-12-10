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
} from 'antd'
import {
  ClusterOutlined,
  DatabaseOutlined,
  SaveOutlined,
  FolderOpenOutlined,
  DeleteOutlined,
  ApiOutlined,
  PlusOutlined,
  MinusCircleOutlined,
} from '@ant-design/icons'
import Icon from '@ant-design/icons'

// 自定义芯片图标 - UXWing专业图标
const ChipSvg = () => (
  <svg viewBox="0 0 122.88 122.88" width="1em" height="1em" fill="currentColor">
    <path d="M28.7,122.88h11.03v-13.4H28.7V122.88L28.7,122.88z M22.67,19.51h74.76c2.56,0,4.66,2.09,4.66,4.66v75.01 c0,2.56-2.1,4.66-4.66,4.66l-74.76,0c-2.56,0-4.66-2.1-4.66-4.66V24.16C18.01,21.6,20.1,19.51,22.67,19.51L22.67,19.51L22.67,19.51 z M42.35,41.29h35.38c1.55,0,2.81,1.27,2.81,2.81v35.12c0,1.55-1.27,2.81-2.81,2.81H42.35c-1.55,0-2.81-1.27-2.81-2.81V44.1 C39.54,42.56,40.8,41.29,42.35,41.29L42.35,41.29z M122.88,65.62v9.16h-13.4v-9.16H122.88L122.88,65.62z M122.88,48.1v9.16l-13.4,0 V48.1L122.88,48.1L122.88,48.1L122.88,48.1z M122.88,83.15v11.03h-13.4V83.15H122.88L122.88,83.15z M122.88,28.7v11.03h-13.4V28.7 H122.88L122.88,28.7z M0,65.62v9.16h13.4v-9.16H0L0,65.62z M0,48.1v9.16l13.4,0V48.1L0,48.1L0,48.1z M0,83.15v11.03h13.4V83.15H0 L0,83.15z M0,28.7v11.03h13.4V28.7H0L0,28.7z M65.62,0h9.16v13.4h-9.16V0L65.62,0L65.62,0z M48.1,0h9.16v13.4H48.1V0L48.1,0L48.1,0 z M83.15,0h11.03v13.4H83.15V0L83.15,0L83.15,0z M28.7,0h11.03v13.4H28.7V0L28.7,0L28.7,0z M65.62,122.88h9.16v-13.4h-9.16V122.88 L65.62,122.88z M48.1,122.88h9.16v-13.4H48.1V122.88L48.1,122.88z M83.15,122.88h11.03v-13.4H83.15V122.88L83.15,122.88z"/>
  </svg>
)
const ChipIcon = () => <Icon component={ChipSvg} />

// 自定义PCB板卡图标 - UXWing主板图标
const BoardSvg = () => (
  <svg viewBox="0 0 122.88 117.61" width="1em" height="1em" fill="currentColor">
    <path d="M71.39,103.48h3.64v-6.8h-3.64V103.48L71.39,103.48L71.39,103.48z M6.03,0h110.81c1.65,0,3.16,0.68,4.25,1.77 c1.1,1.1,1.78,2.61,1.78,4.26v105.54c0,1.66-0.68,3.17-1.77,4.27c-1.09,1.09-2.6,1.77-4.26,1.77H6.03c-1.66,0-3.17-0.68-4.26-1.77 S0,113.23,0,111.57V6.03c0-1.65,0.68-3.16,1.78-4.26C2.88,0.68,4.38,0,6.03,0L6.03,0z M115.35,7.53H7.53v46.73h10.68v-4.04 c-0.1-0.04-0.19-0.1-0.27-0.15c-0.16-0.1-0.31-0.22-0.44-0.34l-0.01-0.02c-0.23-0.22-0.41-0.5-0.54-0.8 c-0.12-0.3-0.19-0.63-0.19-0.97c0-0.35,0.07-0.67,0.19-0.97l0,0c0.13-0.31,0.32-0.59,0.55-0.84c0.23-0.23,0.51-0.42,0.82-0.55 c0.31-0.12,0.63-0.19,0.97-0.19c0.35,0,0.67,0.07,0.97,0.19c0.31,0.13,0.59,0.32,0.83,0.55c0.23,0.24,0.42,0.51,0.55,0.82 l0.01,0.02c0.12,0.29,0.19,0.62,0.19,0.95c0,0.34-0.07,0.67-0.19,0.97c-0.13,0.31-0.32,0.59-0.55,0.82l-0.02,0.02 c-0.12,0.11-0.24,0.22-0.38,0.31c-0.08,0.05-0.15,0.1-0.23,0.13v5.21c0,0.31-0.13,0.6-0.33,0.8c-0.21,0.2-0.49,0.33-0.8,0.33H7.53 v3.4h16.61l0,0c0.04-0.09,0.09-0.17,0.14-0.25l0.01-0.01c0.1-0.15,0.21-0.28,0.33-0.4c0.24-0.23,0.51-0.42,0.83-0.55l0.02-0.01 c0.3-0.12,0.62-0.19,0.95-0.19c0.34,0,0.67,0.07,0.97,0.19c0.31,0.13,0.6,0.32,0.82,0.55c0.23,0.23,0.42,0.51,0.55,0.82 c0.12,0.31,0.19,0.63,0.19,0.97c0,0.35-0.07,0.67-0.19,0.97c-0.13,0.31-0.32,0.59-0.55,0.82c-0.24,0.23-0.51,0.42-0.82,0.55 l-0.02,0.01c-0.3,0.12-0.62,0.19-0.95,0.19c-0.35,0-0.67-0.07-0.97-0.19c-0.31-0.13-0.59-0.32-0.83-0.55 c-0.12-0.12-0.24-0.26-0.33-0.42c-0.05-0.08-0.1-0.17-0.14-0.25H7.53v3.92h18.76l0,0c0.31,0,0.59,0.13,0.8,0.33 c0.21,0.2,0.33,0.48,0.33,0.8v8.39c0.1,0.04,0.21,0.1,0.31,0.16c0.17,0.11,0.34,0.23,0.48,0.38c0.23,0.24,0.42,0.51,0.55,0.82 l0.01,0.02c0.12,0.3,0.18,0.62,0.18,0.95c0,0.34-0.07,0.67-0.19,0.97c-0.13,0.31-0.32,0.59-0.55,0.84 c-0.23,0.23-0.51,0.42-0.82,0.55c-0.31,0.12-0.63,0.19-0.97,0.19c-0.34,0-0.67-0.07-0.97-0.19c-0.31-0.13-0.59-0.32-0.82-0.55 c-0.23-0.23-0.42-0.51-0.55-0.82c-0.12-0.3-0.19-0.63-0.19-0.97c0-0.35,0.07-0.67,0.19-0.97c0.13-0.31,0.32-0.6,0.55-0.83 c0.11-0.11,0.22-0.2,0.35-0.29c0.06-0.04,0.13-0.08,0.19-0.12v-7.38H7.53v3.2h9.89l0,0c0.31,0,0.59,0.13,0.79,0.33 c0.21,0.21,0.33,0.49,0.33,0.8v4.49c0.08,0.04,0.16,0.08,0.24,0.13c0.15,0.1,0.29,0.21,0.41,0.33l0.02,0.02 c0.22,0.23,0.4,0.51,0.53,0.81c0.12,0.3,0.19,0.63,0.19,0.97c0,0.35-0.07,0.67-0.19,0.97l-0.01,0.02 c-0.13,0.31-0.31,0.58-0.54,0.81l-0.02,0.01c-0.23,0.23-0.51,0.41-0.81,0.54C18.07,81.93,17.74,82,17.4,82 c-0.34,0-0.67-0.07-0.97-0.19c-0.31-0.13-0.6-0.32-0.82-0.55l-0.02-0.02c-0.22-0.23-0.4-0.51-0.53-0.81 c-0.12-0.31-0.19-0.63-0.19-0.97c0-0.34,0.07-0.66,0.19-0.96c0.13-0.31,0.32-0.6,0.55-0.82l0,0c0.13-0.14,0.27-0.25,0.43-0.35 c0.08-0.06,0.17-0.1,0.26-0.15v-3.34H7.53v36.24h107.82V7.53L115.35,7.53z"/>
  </svg>
)
const BoardIcon = () => <Icon component={BoardSvg} />
import {
  HierarchicalTopology, CHIP_TYPE_COLORS, CHIP_TYPE_NAMES, ChipType,
  GlobalSwitchConfig, SwitchTypeConfig, SwitchLayerConfig, HierarchyLevelSwitchConfig,
  SWITCH_LAYER_COLORS
} from '../types'
import { listConfigs, saveConfig, deleteConfig, SavedConfig } from '../api/topology'

const { Text, Title } = Typography

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

interface ConfigPanelProps {
  topology: HierarchicalTopology | null
  onGenerate: (config: {
    pod_count: number
    racks_per_pod: number
    board_configs: BoardConfigs
    switch_config?: GlobalSwitchConfig
  }) => void
  loading: boolean
}

// localStorage缓存key
const CONFIG_CACHE_KEY = 'tier6_topology_config_cache'

// 默认配置
const DEFAULT_BOARD_CONFIGS: BoardConfigs = {
  u1: { count: 0, chips: { npu: 2, cpu: 0 } },
  u2: { count: 8, chips: { npu: 8, cpu: 0 } },
  u4: { count: 0, chips: { npu: 16, cpu: 2 } },
}

// 默认Switch配置
const DEFAULT_SWITCH_CONFIG: GlobalSwitchConfig = {
  switch_types: [
    { id: 'leaf_48', name: '48端口Leaf交换机', port_count: 48 },
    { id: 'leaf_72', name: '72端口Leaf交换机', port_count: 72 },
    { id: 'spine_128', name: '128端口Spine交换机', port_count: 128 },
    { id: 'core_512', name: '512端口核心交换机', port_count: 512 },
  ],
  datacenter_level: { enabled: false, layers: [], downlink_redundancy: 1, connect_to_upper_level: true },
  pod_level: { enabled: false, layers: [], downlink_redundancy: 1, connect_to_upper_level: true },
  rack_level: { enabled: false, layers: [], downlink_redundancy: 1, connect_to_upper_level: true },
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
}

const SwitchLevelConfig: React.FC<SwitchLevelConfigProps> = ({
  levelKey,
  config,
  switchTypes,
  onChange,
  configRowStyle,
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

      {config.enabled && (
        <>
          {/* 冗余度配置 */}
          <div style={configRowStyle}>
            <Text>冗余连接数</Text>
            <InputNumber
              min={1}
              max={4}
              size="small"
              value={config.downlink_redundancy}
              onChange={(v) => onChange({ ...config, downlink_redundancy: v || 1 })}
              style={{ width: 60 }}
            />
          </div>

          {/* 连接到上层 */}
          <div style={configRowStyle}>
            <Text>连接到上层Switch</Text>
            <Switch
              size="small"
              checked={config.connect_to_upper_level}
              onChange={(checked) => onChange({ ...config, connect_to_upper_level: checked })}
            />
          </div>

          <Divider style={{ margin: '8px 0' }} />

          {/* Switch层列表 */}
          <Text type="secondary" style={{ fontSize: 11 }}>Switch层配置 (从下到上)</Text>
          {config.layers.map((layer, index) => (
            <div key={index} style={{ marginTop: 8, padding: 8, background: '#f5f5f5', borderRadius: 4 }}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 4 }}>
                <Input
                  size="small"
                  placeholder="层名称"
                  value={layer.layer_name}
                  onChange={(e) => updateLayer(index, 'layer_name', e.target.value)}
                  style={{ width: 60 }}
                />
                <Select
                  size="small"
                  value={layer.switch_type_id}
                  onChange={(v) => updateLayer(index, 'switch_type_id', v)}
                  style={{ width: 130 }}
                  options={switchTypes.map(t => ({ value: t.id, label: `${t.name} (${t.port_count}口)` }))}
                />
                <InputNumber
                  size="small"
                  min={1}
                  max={16}
                  value={layer.count}
                  onChange={(v) => updateLayer(index, 'count', v || 1)}
                  style={{ width: 50 }}
                  placeholder="数量"
                />
                <Button
                  type="text"
                  danger
                  size="small"
                  icon={<MinusCircleOutlined />}
                  onClick={() => removeLayer(index)}
                />
              </div>
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

export const ConfigPanel: React.FC<ConfigPanelProps> = ({
  topology,
  onGenerate,
}) => {
  // 从缓存加载初始配置
  const cachedConfig = loadCachedConfig()

  // Pod层级配置
  const [podCount, setPodCount] = useState(cachedConfig?.podCount ?? 1)

  // Rack层级配置
  const [racksPerPod, setRacksPerPod] = useState(cachedConfig?.racksPerPod ?? 4)

  // Board配置（按U高度分类，每种类型有独立的chip配置）
  const [boardConfigs, setBoardConfigs] = useState<BoardConfigs>(
    cachedConfig?.boardConfigs ?? DEFAULT_BOARD_CONFIGS
  )

  // Switch配置
  const [switchConfig, setSwitchConfig] = useState<GlobalSwitchConfig>(
    cachedConfig?.switchConfig ?? DEFAULT_SWITCH_CONFIG
  )

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
    saveCachedConfig({ podCount, racksPerPod, boardConfigs, switchConfig })
  }, [podCount, racksPerPod, boardConfigs, switchConfig])

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
        switch_config: switchConfig,
      })
    }, 500)

    return () => clearTimeout(timer)
  }, [podCount, racksPerPod, boardConfigs, switchConfig, onGenerate])

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

  const collapseItems = [
    {
      key: 'pod',
      label: <Text strong><ClusterOutlined /> Pod 配置</Text>,
      children: (
        <div>
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
      ),
    },
    {
      key: 'rack',
      label: <Text strong><DatabaseOutlined /> Rack 配置</Text>,
      children: (
        <div>
          <div style={configRowStyle}>
            <Text>每Pod机柜数</Text>
            <InputNumber
              min={1}
              max={20}
              value={racksPerPod}
              onChange={(v) => setRacksPerPod(v || 1)}
              size="small"
              style={{ width: 80 }}
            />
          </div>
        </div>
      ),
    },
    {
      key: 'board',
      label: <Text strong><BoardIcon /> Board 配置</Text>,
      children: (
        <div>
          <Text type="secondary" style={{ fontSize: 12, marginBottom: 12, display: 'block' }}>
            每机柜板卡配置 (已用 {totalUsedU}/42U)
          </Text>

          {/* 1U 板卡配置 */}
          <div style={{ marginBottom: 12, padding: 8, background: '#f5f5f5', borderRadius: 4 }}>
            <div style={configRowStyle}>
              <Text strong style={{ color: '#4a5568' }}>1U 板卡</Text>
              <InputNumber
                min={0}
                max={42}
                value={boardConfigs.u1.count}
                onChange={(v) => updateBoardCount('u1', v)}
                size="small"
                style={{ width: 60 }}
              />
            </div>
            <div style={{ display: 'flex', gap: 12, marginTop: 4 }}>
              {(Object.keys(CHIP_TYPE_COLORS) as ChipType[]).map(type => (
                <div key={type} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <div style={{ width: 8, height: 8, borderRadius: 2, background: CHIP_TYPE_COLORS[type] }} />
                  <Text style={{ fontSize: 11 }}>{CHIP_TYPE_NAMES[type]}</Text>
                  <InputNumber
                    min={0}
                    max={32}
                    value={boardConfigs.u1.chips[type]}
                    onChange={(v) => updateBoardChip('u1', type, v)}
                    size="small"
                    style={{ width: 50 }}
                  />
                </div>
              ))}
            </div>
          </div>

          {/* 2U 板卡配置 */}
          <div style={{ marginBottom: 12, padding: 8, background: '#e6f0ff', borderRadius: 4 }}>
            <div style={configRowStyle}>
              <Text strong style={{ color: '#2c5282' }}>2U 板卡</Text>
              <InputNumber
                min={0}
                max={21}
                value={boardConfigs.u2.count}
                onChange={(v) => updateBoardCount('u2', v)}
                size="small"
                style={{ width: 60 }}
              />
            </div>
            <div style={{ display: 'flex', gap: 12, marginTop: 4 }}>
              {(Object.keys(CHIP_TYPE_COLORS) as ChipType[]).map(type => (
                <div key={type} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <div style={{ width: 8, height: 8, borderRadius: 2, background: CHIP_TYPE_COLORS[type] }} />
                  <Text style={{ fontSize: 11 }}>{CHIP_TYPE_NAMES[type]}</Text>
                  <InputNumber
                    min={0}
                    max={32}
                    value={boardConfigs.u2.chips[type]}
                    onChange={(v) => updateBoardChip('u2', type, v)}
                    size="small"
                    style={{ width: 50 }}
                  />
                </div>
              ))}
            </div>
          </div>

          {/* 4U 板卡配置 */}
          <div style={{ marginBottom: 8, padding: 8, background: '#f3e8ff', borderRadius: 4 }}>
            <div style={configRowStyle}>
              <Text strong style={{ color: '#553c9a' }}>4U 板卡</Text>
              <InputNumber
                min={0}
                max={10}
                value={boardConfigs.u4.count}
                onChange={(v) => updateBoardCount('u4', v)}
                size="small"
                style={{ width: 60 }}
              />
            </div>
            <div style={{ display: 'flex', gap: 12, marginTop: 4 }}>
              {(Object.keys(CHIP_TYPE_COLORS) as ChipType[]).map(type => (
                <div key={type} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <div style={{ width: 8, height: 8, borderRadius: 2, background: CHIP_TYPE_COLORS[type] }} />
                  <Text style={{ fontSize: 11 }}>{CHIP_TYPE_NAMES[type]}</Text>
                  <InputNumber
                    min={0}
                    max={32}
                    value={boardConfigs.u4.chips[type]}
                    onChange={(v) => updateBoardChip('u4', type, v)}
                    size="small"
                    style={{ width: 50 }}
                  />
                </div>
              ))}
            </div>
          </div>

          {totalUsedU > 42 && (
            <Text type="danger" style={{ fontSize: 12 }}>
              超出机柜容量！
            </Text>
          )}
        </div>
      ),
    },
    {
      key: 'switch',
      label: <Text strong><ApiOutlined /> Switch 配置</Text>,
      children: (
        <div>
          <Text type="secondary" style={{ fontSize: 12, marginBottom: 12, display: 'block' }}>
            配置各层级间的Switch连接（Board-Chip层不使用Switch）
          </Text>

          <Tabs
            size="small"
            items={[
              {
                key: 'rack',
                label: 'Rack层(Board间)',
                children: (
                  <SwitchLevelConfig
                    levelKey="rack_level"
                    config={switchConfig.rack_level}
                    switchTypes={switchConfig.switch_types}
                    onChange={(newConfig) => setSwitchConfig(prev => ({ ...prev, rack_level: newConfig }))}
                    configRowStyle={configRowStyle}
                  />
                ),
              },
              {
                key: 'pod',
                label: 'Pod层(Rack间)',
                children: (
                  <SwitchLevelConfig
                    levelKey="pod_level"
                    config={switchConfig.pod_level}
                    switchTypes={switchConfig.switch_types}
                    onChange={(newConfig) => setSwitchConfig(prev => ({ ...prev, pod_level: newConfig }))}
                    configRowStyle={configRowStyle}
                  />
                ),
              },
              {
                key: 'datacenter',
                label: '数据中心层(Pod间)',
                children: (
                  <SwitchLevelConfig
                    levelKey="datacenter_level"
                    config={switchConfig.datacenter_level}
                    switchTypes={switchConfig.switch_types}
                    onChange={(newConfig) => setSwitchConfig(prev => ({ ...prev, datacenter_level: newConfig }))}
                    configRowStyle={configRowStyle}
                  />
                ),
              },
            ]}
          />
        </div>
      ),
    },
  ]

  return (
    <div>
      <Title level={5} style={{ marginBottom: 16 }}>拓扑配置</Title>

      {/* 统计信息 */}
      <Card size="small" style={{ marginBottom: 16 }}>
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
                prefix={<ApiOutlined />}
                valueStyle={{ fontSize: 16 }}
              />
            </Col>
          )}
        </Row>
      </Card>

      {/* 分层级配置 */}
      <Collapse
        items={collapseItems}
        defaultActiveKey={['pod', 'rack', 'board', 'switch']}
        size="small"
        style={{ marginBottom: 16 }}
      />

      {/* 操作按钮 */}
      <Space style={{ width: '100%' }} direction="vertical">
        <Row gutter={8}>
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
      </Space>

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
