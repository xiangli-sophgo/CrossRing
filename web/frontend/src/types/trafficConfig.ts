// 流量配置相关类型定义

export interface TrafficConfig {
  id: string
  topology: string
  mode: 'noc' | 'd2d'
  source_ip: string
  target_ip: string
  speed_gbps: number
  burst_length: number
  request_type: 'R' | 'W'
  end_time_ns: number
  created_at: string
  source_die?: number
  target_die?: number
}

export interface TrafficConfigCreate {
  topology: string
  mode: 'noc' | 'd2d'
  source_ip: string
  target_ip: string
  speed_gbps: number
  burst_length: number
  request_type: 'R' | 'W'
  end_time_ns: number
}

export interface BatchTrafficConfigCreate {
  topology: string
  mode: 'noc' | 'd2d'
  source_ips: string[]
  target_ips: string[]
  speed_gbps: number
  burst_length: number
  request_type: 'R' | 'W'
  end_time_ns: number
  source_die?: number
  target_die?: number
}

export interface TrafficConfigResponse {
  success: boolean
  message: string
  config?: TrafficConfig
}

export interface TrafficConfigListResponse {
  topology: string
  mode: string
  configs: TrafficConfig[]
  total: number
}

export interface TrafficGenerateRequest {
  topology: string
  mode: string
  split_by_source: boolean
  random_seed: number
  filename?: string
}

export interface TrafficGenerateResponse {
  success: boolean
  message: string
  file_path?: string
  total_lines: number
  file_size: number
  generation_time_ms: number
}
