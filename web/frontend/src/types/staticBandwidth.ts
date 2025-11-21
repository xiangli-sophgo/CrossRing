// 静态带宽分析相关类型定义

export interface BandwidthStatistics {
  max_bandwidth: number
  sum_bandwidth: number
  avg_bandwidth: number
  num_active_links: number
}

export interface BandwidthComputeRequest {
  topology: string
  mode: string
  routing_type: 'XY' | 'YX'
}

export interface BandwidthComputeResponse {
  success: boolean
  message: string
  link_bandwidth: Record<string, number>  // key格式: 'x1,y1-x2,y2'
  statistics: BandwidthStatistics
}

// 用于前端处理的链路带宽数据结构
export interface LinkBandwidth {
  src: { col: number; row: number }
  dst: { col: number; row: number }
  bandwidth: number
}
