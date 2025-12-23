/**
 * 波形数据 API 客户端
 */

import apiClient from './client';
import type { TopologyData } from '@/types/topology';

// ==================== 类型定义 ====================

export interface WaveformEvent {
  stage: string;          // 阶段名: "l2h", "iq_out", "h_ring", "rb", "v_ring", "eq"
  start_ns: number;       // 开始时间(ns)
  end_ns: number;         // 结束时间(ns)
}

export interface WaveformSignal {
  name: string;           // 信号名: "Pkt_123.REQ", "Pkt_123.D0", "Pkt_123.RSP"
  packet_id: number;      // 请求ID
  flit_type: 'req' | 'data' | 'rsp';  // flit类型
  flit_id?: number;       // flit序号（data flit用）
  events: WaveformEvent[];  // 各阶段事件
}

export interface WaveformResponse {
  time_range: {
    start_ns: number;
    end_ns: number;
  };
  signals: WaveformSignal[];
  stages: string[];       // 所有阶段名称列表
}

export interface PacketInfo {
  packet_id: number;
  req_type: 'read' | 'write';
  source_node: number;
  source_type: string;
  dest_node: number;
  dest_type: string;
  start_time_ns: number;
  end_time_ns: number;
  latency_ns: number;
  cmd_latency_ns?: number;
  data_latency_ns?: number;
  transaction_latency_ns?: number;
}

export interface PacketListResponse {
  packets: PacketInfo[];
  total: number;
  page: number;
  page_size: number;
}

export interface WaveformCheckResponse {
  available: boolean;
  message?: string;
  parquet_dir?: string;
  stats?: {
    total_packets: number;
    total_flits: number;
    read_packets: number;
    write_packets: number;
    time_range_ns: {
      start: number;
      end: number;
    };
  };
}

// ==================== API 函数 ====================

/**
 * 获取波形数据
 */
export const getWaveformData = async (
  experimentId: number,
  resultId: number,
  options?: {
    packetIds?: number[];
    timeStart?: number;
    timeEnd?: number;
    maxPackets?: number;
  }
): Promise<WaveformResponse> => {
  const params: Record<string, string | number> = {};

  if (options?.packetIds && options.packetIds.length > 0) {
    params.packet_ids = options.packetIds.join(',');
  }
  if (options?.timeStart !== undefined) {
    params.time_start = options.timeStart;
  }
  if (options?.timeEnd !== undefined) {
    params.time_end = options.timeEnd;
  }
  if (options?.maxPackets !== undefined) {
    params.max_packets = options.maxPackets;
  }

  const response = await apiClient.get(
    `/api/experiments/${experimentId}/results/${resultId}/waveform`,
    { params }
  );
  return response.data;
};

/**
 * 获取请求列表
 */
export const getPacketList = async (
  experimentId: number,
  resultId: number,
  options?: {
    page?: number;
    pageSize?: number;
    reqType?: 'read' | 'write';
    sortBy?: string;
    order?: 'asc' | 'desc';
  }
): Promise<PacketListResponse> => {
  const params: Record<string, string | number> = {
    page: options?.page ?? 1,
    page_size: options?.pageSize ?? 100,
    sort_by: options?.sortBy ?? 'start_time_ns',
    order: options?.order ?? 'asc',
  };

  if (options?.reqType) {
    params.req_type = options.reqType;
  }

  const response = await apiClient.get(
    `/api/experiments/${experimentId}/results/${resultId}/packets`,
    { params }
  );
  return response.data;
};

/**
 * 检查波形数据是否可用
 */
export const checkWaveformData = async (
  experimentId: number,
  resultId: number
): Promise<WaveformCheckResponse> => {
  const response = await apiClient.get(
    `/api/experiments/${experimentId}/results/${resultId}/waveform/check`
  );
  return response.data;
};

// ==================== 常量 ====================

/**
 * 阶段颜色映射
 */
export const STAGE_COLORS: Record<string, string> = {
  IP_inject: '#91d5ff',   // 浅蓝 - IP注入
  IQ: '#1890ff',          // 蓝 - Inject Queue
  Link: '#52c41a',        // 绿 - 链路传输
  RB: '#faad14',          // 黄 - Ring Bridge
  EQ: '#f5222d',          // 红 - Eject Queue
  IP_eject: '#ff7a45',    // 橙 - IP弹出
};

/**
 * 阶段中文名映射
 */
export const STAGE_NAMES: Record<string, string> = {
  IP_inject: 'IP Inject',
  IQ: 'IQ',
  Link: 'Link',
  RB: 'RB',
  EQ: 'EQ',
  IP_eject: 'IP Eject',
};

/**
 * flit类型颜色
 */
export const FLIT_TYPE_COLORS: Record<string, string> = {
  req: '#722ed1',     // 紫色 - 请求
  data: '#13c2c2',    // 青色 - 数据
  rsp: '#eb2f96',     // 粉色 - 响应
};

// ==================== 拓扑数据 ====================

// 重新导出类型供外部使用
export type { TopologyData, TopologyNode, TopologyEdge } from '@/types/topology';

/**
 * 获取拓扑数据
 */
export const getTopologyData = async (
  experimentId: number,
  resultId: number
): Promise<TopologyData> => {
  const response = await apiClient.get(
    `/api/experiments/${experimentId}/results/${resultId}/topology`
  );
  return response.data;
};

// ==================== 活跃IP数据 ====================

export interface ActiveIPsResponse {
  active_ips: Record<number, string[]>;  // {node_id: [ip_type1, ip_type2, ...]}
}

/**
 * 获取实验中活跃的IP列表
 * 从requests.parquet中提取所有涉及的节点和IP类型
 */
export const getActiveIPs = async (
  experimentId: number,
  resultId: number
): Promise<ActiveIPsResponse> => {
  const response = await apiClient.get(
    `/api/experiments/${experimentId}/results/${resultId}/active-ips`
  );
  return response.data;
};
