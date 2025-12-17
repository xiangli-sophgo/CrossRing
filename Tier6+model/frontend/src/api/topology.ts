import axios from 'axios'
import {
  HierarchicalTopology,
  PodConfig,
  RackConfig,
  BoardConfig,
  ConnectionConfig,
  GlobalSwitchConfig,
  ManualConnectionConfig,
  ManualConnection,
} from '../types'

const api = axios.create({
  baseURL: '/api',
})

// 获取完整拓扑数据
export async function getTopology(): Promise<HierarchicalTopology> {
  const response = await api.get('/topology')
  return response.data
}

// 生成新的拓扑
export async function generateTopology(config: {
  pod_count?: number
  racks_per_pod?: number
  board_configs?: {
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
  switch_config?: GlobalSwitchConfig
  manual_connections?: ManualConnectionConfig
}): Promise<HierarchicalTopology> {
  const response = await api.post('/topology/generate', config)
  return response.data
}

// 获取指定Pod
export async function getPod(podId: string): Promise<PodConfig> {
  const response = await api.get(`/topology/pod/${podId}`)
  return response.data
}

// 获取指定Rack
export async function getRack(rackId: string): Promise<RackConfig> {
  const response = await api.get(`/topology/rack/${rackId}`)
  return response.data
}

// 获取指定Board
export async function getBoard(boardId: string): Promise<BoardConfig> {
  const response = await api.get(`/topology/board/${boardId}`)
  return response.data
}

// 获取连接数据
export async function getConnections(
  level?: string,
  parentId?: string
): Promise<ConnectionConfig[]> {
  const params: Record<string, string> = {}
  if (level) params.level = level
  if (parentId) params.parent_id = parentId
  const response = await api.get('/topology/connections', { params })
  return response.data
}

// 获取Chip类型配置
export async function getChipTypes(): Promise<{
  types: { id: string; name: string; color: string }[]
}> {
  const response = await api.get('/config/chip-types')
  return response.data
}

// 获取Rack尺寸配置
export async function getRackDimensions(): Promise<{
  width: number
  depth: number
  u_height: number
  total_u: number
  full_height: number
}> {
  const response = await api.get('/config/rack-dimensions')
  return response.data
}

// 获取各层级连接的默认带宽和延迟配置
export async function getLevelConnectionDefaults(): Promise<{
  datacenter: { bandwidth: number; latency: number }
  pod: { bandwidth: number; latency: number }
  rack: { bandwidth: number; latency: number }
  board: { bandwidth: number; latency: number }
}> {
  const response = await api.get('/config/level-connection-defaults')
  return response.data
}

// ============================================
// 配置保存/加载 API
// ============================================

export interface SavedConfig {
  name: string
  description?: string
  pod_count: number
  racks_per_pod: number
  board_configs: {
    u1: { count: number; chips: { npu: number; cpu: number } }
    u2: { count: number; chips: { npu: number; cpu: number } }
    u4: { count: number; chips: { npu: number; cpu: number } }
  }
  created_at?: string
  updated_at?: string
}

// 获取所有保存的配置
export async function listConfigs(): Promise<SavedConfig[]> {
  const response = await api.get('/configs')
  return response.data
}

// 获取指定配置
export async function getConfig(name: string): Promise<SavedConfig> {
  const response = await api.get(`/configs/${encodeURIComponent(name)}`)
  return response.data
}

// 保存配置
export async function saveConfig(config: SavedConfig): Promise<SavedConfig> {
  const response = await api.post('/configs', config)
  return response.data
}

// 删除配置
export async function deleteConfig(name: string): Promise<void> {
  await api.delete(`/configs/${encodeURIComponent(name)}`)
}

// ============================================
// 手动连接 API
// ============================================

// 获取手动连接配置
export async function getManualConnections(): Promise<ManualConnectionConfig> {
  const response = await api.get('/manual-connections')
  return response.data
}

// 保存手动连接配置
export async function saveManualConnections(config: ManualConnectionConfig): Promise<ManualConnectionConfig> {
  const response = await api.post('/manual-connections', config)
  return response.data
}

// 添加单个手动连接
export async function addManualConnection(connection: ManualConnection): Promise<ManualConnectionConfig> {
  const response = await api.post('/manual-connections/add', connection)
  return response.data
}

// 删除单个手动连接
export async function deleteManualConnection(connectionId: string): Promise<void> {
  await api.delete(`/manual-connections/${encodeURIComponent(connectionId)}`)
}

// 清空手动连接（可按层级清空）
export async function clearManualConnections(hierarchyLevel?: string): Promise<void> {
  const params: Record<string, string> = {}
  if (hierarchyLevel) params.hierarchy_level = hierarchyLevel
  await api.delete('/manual-connections', { params })
}
