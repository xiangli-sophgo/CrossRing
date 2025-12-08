/**
 * 仿真执行 API 客户端
 */
import axios from 'axios'

const api = axios.create({
  baseURL: '/api/simulation',
})

export interface SimulationRequest {
  mode: 'kcin' | 'dcin'
  topology: string
  config_path?: string
  traffic_source: 'file' | 'generate'
  traffic_files: string[]
  traffic_path?: string
  max_time: number
  save_to_db: boolean
  experiment_name?: string
  result_granularity: 'per_file' | 'per_batch'
}

export interface TaskResponse {
  task_id: string
  status: string
  message: string
}

export interface TaskStatus {
  task_id: string
  status: string
  progress: number
  current_file: string
  message: string
  error: string | null
  results: Record<string, any> | null
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
}

// 启动仿真
export const runSimulation = async (request: SimulationRequest): Promise<TaskResponse> => {
  const response = await api.post('/run', request)
  return response.data
}

// 获取任务状态
export const getTaskStatus = async (taskId: string): Promise<TaskStatus> => {
  const response = await api.get(`/status/${taskId}`)
  return response.data
}

// 取消任务
export const cancelTask = async (taskId: string): Promise<{ success: boolean; message: string }> => {
  const response = await api.post(`/cancel/${taskId}`)
  return response.data
}

// 删除任务
export const deleteTask = async (taskId: string): Promise<{ success: boolean; message: string }> => {
  const response = await api.delete(`/${taskId}`)
  return response.data
}

// 获取历史任务
export const getTaskHistory = async (limit: number = 20): Promise<{ tasks: TaskHistoryItem[] }> => {
  const response = await api.get(`/history?limit=${limit}`)
  return response.data
}

// 获取可用配置
export const getConfigs = async (): Promise<{ kcin: ConfigOption[]; dcin: ConfigOption[] }> => {
  const response = await api.get('/configs')
  return response.data
}

// 获取流量文件列表
export const getTrafficFiles = async (path: string = ''): Promise<TrafficFilesResponse> => {
  const response = await api.get(`/traffic-files?path=${encodeURIComponent(path)}`)
  return response.data
}
