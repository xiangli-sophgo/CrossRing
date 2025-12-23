import apiClient from './client';

export interface FIFOEvent {
  enter_ns: number;
  leave_ns: number;
  flit_id: string;
}

export interface FIFOSignal {
  name: string;
  node_id: number;
  fifo_type: string;
  events: FIFOEvent[];
}

export interface FIFOWaveformResponse {
  time_range: { start_ns: number; end_ns: number };
  signals: FIFOSignal[];
  available_fifos: string[];
}

export const getFIFOWaveform = async (
  experimentId: number,
  resultId: number,
  nodeId: number,
  fifoTypes: string[],
  flitTypesFilter?: string[],
  options?: {
    timeStart?: number;
    timeEnd?: number;
    expandRspSignals?: string[];
    expandReqSignals?: string[];
    expandDataSignals?: string[];
  }
): Promise<FIFOWaveformResponse> => {
  const params: Record<string, string | number> = {
    node_id: nodeId,
    fifo_types: fifoTypes.join(','),
  };
  if (flitTypesFilter && flitTypesFilter.length > 0) {
    params.flit_types_filter = flitTypesFilter.join(',');
  }
  if (options?.expandRspSignals && options.expandRspSignals.length > 0) {
    params.expand_rsp_signals = options.expandRspSignals.join(',');
  }
  if (options?.expandReqSignals && options.expandReqSignals.length > 0) {
    params.expand_req_signals = options.expandReqSignals.join(',');
  }
  if (options?.expandDataSignals && options.expandDataSignals.length > 0) {
    params.expand_data_signals = options.expandDataSignals.join(',');
  }
  if (options?.timeStart !== undefined) params.time_start = options.timeStart;
  if (options?.timeEnd !== undefined) params.time_end = options.timeEnd;

  const response = await apiClient.get(
    `/api/experiments/${experimentId}/results/${resultId}/fifo-waveform`,
    { params }
  );
  return response.data;
};

export const getAvailableFIFOs = async (
  experimentId: number,
  resultId: number,
  nodeId: number
): Promise<string[]> => {
  const response = await apiClient.get(
    `/api/experiments/${experimentId}/results/${resultId}/fifo-waveform/available`,
    { params: { node_id: nodeId } }
  );
  return response.data.fifos;
};

export const FIFO_COLORS: Record<string, string> = {
  IQ_TR: '#1890ff',
  IQ_TL: '#13c2c2',
  IQ_TU: '#52c41a',
  IQ_TD: '#faad14',
  IQ_EQ: '#722ed1',
  IQ_CH: '#8c8c8c',
  RB_TR: '#f5222d',
  RB_TL: '#fa8c16',
  RB_TU: '#eb2f96',
  RB_TD: '#a0d911',
  RB_EQ: '#2f54eb',
  EQ_TU: '#fa541c',
  EQ_TD: '#fadb14',
  EQ_CH: '#13c2c2',
};

// flit 类型颜色（req/data/rsp 基础颜色）
export const FLIT_TYPE_COLORS: Record<string, string> = {
  req: '#52c41a',     // 绿色 - 请求
  data: '#1890ff',    // 蓝色 - 数据
  rsp: '#eb2f96',     // 粉色 - 响应（默认）
};

// rsp_type 类型颜色（CHI 协议响应类型）
export const RSP_TYPE_COLORS: Record<string, string> = {
  CompData: '#722ed1',    // 紫色 - 带数据完成响应
  Comp: '#eb2f96',        // 粉色 - 完成响应
  DBIDResp: '#fa8c16',    // 橙色 - 数据缓冲区ID响应
  RetryAck: '#f5222d',    // 红色 - 重试确认
  RspSepData: '#13c2c2',  // 青色 - 分离数据响应
  SnpResp: '#a0d911',     // 黄绿色 - Snoop响应
  SnpRespData: '#fadb14', // 黄色 - 带数据Snoop响应
};

/**
 * 获取信号颜色（支持信号名解析）
 * 信号名格式：Node_{id}.{fifo_type}.{flit_type} 或 Node_{id}.{fifo_type}.rsp.{rsp_type}
 */
export const getSignalColor = (signalName: string): string => {
  // 解析信号名：Node_5.IQ_TR.req 或 Node_5.IQ_TR.rsp.CompData
  const parts = signalName.split('.');

  // 至少需要 Node_X.FIFO.flit_type
  if (parts.length >= 3) {
    const flit_type = parts[2];

    // rsp 类型：检查是否有 rsp_type
    if (flit_type === 'rsp' && parts.length >= 4) {
      const rsp_type = parts[3];
      if (RSP_TYPE_COLORS[rsp_type]) {
        return RSP_TYPE_COLORS[rsp_type];
      }
      // 未知的 rsp_type 使用默认 rsp 颜色
      return FLIT_TYPE_COLORS['rsp'];
    }

    // req/data/rsp 类型
    if (FLIT_TYPE_COLORS[flit_type]) {
      return FLIT_TYPE_COLORS[flit_type];
    }
  }

  // 回退到 FIFO 类型颜色
  return getFIFOColor(signalName);
};

// 获取 FIFO 颜色（支持 IP 通道如 IQ_CH_G0）
export const getFIFOColor = (fifoType: string): string => {
  // 直接匹配
  if (FIFO_COLORS[fifoType]) {
    return FIFO_COLORS[fifoType];
  }
  // IP 通道：IQ_CH_G0 -> IQ_CH
  if (fifoType.startsWith('IQ_CH_')) {
    return FIFO_COLORS['IQ_CH'] || '#8c8c8c';
  }
  if (fifoType.startsWith('EQ_CH_')) {
    return FIFO_COLORS['EQ_CH'] || '#13c2c2';
  }
  return '#999';
};
