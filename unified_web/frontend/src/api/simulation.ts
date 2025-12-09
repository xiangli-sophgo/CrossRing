/**
 * 仿真执行 API 客户端
 */
import apiClient from './client'

export interface SimulationRequest {
  mode: 'kcin' | 'dcin'
  topology: string
  config_path?: string
  config_overrides?: Record<string, any>
  // DCIN模式下的DIE拓扑配置
  die_config_path?: string
  die_config_overrides?: Record<string, any>
  traffic_source: 'file' | 'generate'
  traffic_files: string[]
  traffic_path?: string
  max_time: number
  save_to_db: boolean
  experiment_name?: string
  experiment_description?: string
  max_workers?: number  // 并行进程数
}

export interface TaskResponse {
  task_id: string
  status: string
  message: string
}

export interface SimDetails {
  file_index: number
  total_files: number
  current_file: string
  sim_progress: number
  current_time: number
  max_time: number
  req_count: number
  total_req: number
  recv_flits: number
  total_flits: number
  trans_flits: number  // 网络在途flit数
}

export interface TaskStatus {
  task_id: string
  status: string
  progress: number
  current_file: string
  message: string
  error: string | null
  results: Record<string, any> | null
  sim_details: SimDetails | null
  created_at: string
  started_at: string | null
  completed_at: string | null
}

export interface ConfigOption {
  name: string
  path: string
}

export interface TrafficFileInfo {
  name: string
  path: string
  size: number
}

export interface TrafficFilesResponse {
  current_path: string
  files: TrafficFileInfo[]
  directories: string[]
}

// 树形结构节点
export interface TrafficTreeNode {
  key: string
  title: string
  isLeaf: boolean
  children?: TrafficTreeNode[]
  path?: string
  size?: number
  format?: 'kcin' | 'dcin' | 'unknown'
}

export interface TrafficFilesTreeResponse {
  tree: TrafficTreeNode[]
}

export interface TaskHistoryItem {
  task_id: string
  mode: string
  topology: string
  status: string
  progress: number
  message: string
  created_at: string
  completed_at: string | null
  traffic_files: string[]
  experiment_name: string | null
  results: Record<string, any> | null
}

// 启动仿真
export const runSimulation = async (request: SimulationRequest): Promise<TaskResponse> => {
  const response = await apiClient.post('/api/simulation/run', request)
  return response.data
}

// 获取任务状态
export const getTaskStatus = async (taskId: string): Promise<TaskStatus> => {
  const response = await apiClient.get(`/api/simulation/status/${taskId}`)
  return response.data
}

// 取消任务
export const cancelTask = async (taskId: string): Promise<{ success: boolean; message: string }> => {
  const response = await apiClient.post(`/api/simulation/cancel/${taskId}`)
  return response.data
}

// 删除任务
export const deleteTask = async (taskId: string): Promise<{ success: boolean; message: string }> => {
  const response = await apiClient.delete(`/api/simulation/${taskId}`)
  return response.data
}

// 运行中的任务信息
export interface RunningTaskItem {
  task_id: string
  mode: string
  topology: string
  status: string
  progress: number
  message: string
  current_file: string
  created_at: string
  started_at: string | null
  traffic_files: string[]
  experiment_name: string | null
  sim_details: SimDetails | null
}

// 获取运行中的任务列表
export const getRunningTasks = async (): Promise<{ tasks: RunningTaskItem[] }> => {
  const response = await apiClient.get('/api/simulation/running')
  return response.data
}

// 获取历史任务
export const getTaskHistory = async (limit: number = 20): Promise<{ tasks: TaskHistoryItem[] }> => {
  const response = await apiClient.get(`/api/simulation/history?limit=${limit}`)
  return response.data
}

// 清空历史任务
export const clearTaskHistory = async (): Promise<{ success: boolean; message: string }> => {
  const response = await apiClient.delete('/api/simulation/history')
  return response.data
}

// 获取可用配置
export const getConfigs = async (): Promise<{ kcin: ConfigOption[]; dcin: ConfigOption[] }> => {
  const response = await apiClient.get('/api/simulation/configs')
  return response.data
}

// 获取流量文件列表
export const getTrafficFiles = async (path: string = ''): Promise<TrafficFilesResponse> => {
  const response = await apiClient.get(`/api/simulation/traffic-files?path=${encodeURIComponent(path)}`)
  return response.data
}

// 获取流量文件树形结构
export const getTrafficFilesTree = async (mode?: 'kcin' | 'dcin'): Promise<TrafficFilesTreeResponse> => {
  const params = mode ? `?mode=${mode}` : ''
  const response = await apiClient.get(`/api/simulation/traffic-files-tree${params}`)
  return response.data
}

// 流量文件内容响应
export interface TrafficFileContentResponse {
  content: string[]
  total_lines: number
  truncated: boolean
  file_name: string
  file_size: number
}

// 获取流量文件内容
export const getTrafficFileContent = async (filePath: string, maxLines: number = 100): Promise<TrafficFileContentResponse> => {
  const response = await apiClient.get(`/api/simulation/traffic-file-content/${filePath}?max_lines=${maxLines}`)
  return response.data
}

// 获取配置文件内容
export const getConfigContent = async (configPath: string): Promise<Record<string, any>> => {
  const response = await apiClient.get(`/api/simulation/config/${configPath}`)
  return response.data
}

// 保存配置文件
export const saveConfigContent = async (
  configPath: string,
  content: Record<string, any>,
  saveAs?: string
): Promise<{ success: boolean; message: string; filename?: string }> => {
  const response = await apiClient.post(`/api/simulation/config/${configPath}`, { content, save_as: saveAs })
  return response.data
}
