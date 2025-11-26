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
  dcin_config_file?: string
}

// DCIN布局信息
export interface DCINLayoutInfo {
  die_positions: Record<string, [number, number]>  // {die_id: [x, y]}
  die_rotations: Record<string, number>            // {die_id: rotation}
  dcin_connections: [number, number, number, number][]  // [[src_die, src_node, dst_die, dst_node], ...]
  num_dies: number
}

// 流量信息（用于链路带宽组成）
export interface FlowInfo {
  src_node: number
  src_ip: string
  dst_node: number
  dst_ip: string
  bandwidth: number
  req_type: string
}

export interface BandwidthComputeResponse {
  success: boolean
  message: string
  mode: 'kcin' | 'dcin'
  // KCIN模式: Record<string, number> (key格式: 'x1,y1-x2,y2')
  // DCIN模式: Record<string, Record<string, number>> (key格式: {die_id: {link_key: bw}})
  link_bandwidth: Record<string, number> | Record<string, Record<string, number>>
  // 链路带宽组成
  link_composition?: Record<string, FlowInfo[]>
  // DCIN跨Die链路带宽 (key格式: '{src_die}-{src_node}-{dst_die}-{dst_node}')
  dcin_link_bandwidth?: Record<string, number>
  // KCIN模式: BandwidthStatistics
  // DCIN模式: Record<string, BandwidthStatistics> (key格式: {die_id: statistics})
  statistics: BandwidthStatistics | Record<string, BandwidthStatistics>
  // DCIN布局信息（仅DCIN模式）
  dcin_layout?: DCINLayoutInfo
}

// 用于前端处理的链路带宽数据结构
export interface LinkBandwidth {
  src: { col: number; row: number }
  dst: { col: number; row: number }
  bandwidth: number
}
