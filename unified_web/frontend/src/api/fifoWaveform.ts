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
  options?: { timeStart?: number; timeEnd?: number }
): Promise<FIFOWaveformResponse> => {
  const params: Record<string, string | number> = {
    node_id: nodeId,
    fifo_types: fifoTypes.join(','),
  };
  if (flitTypesFilter && flitTypesFilter.length > 0) {
    params.flit_types_filter = flitTypesFilter.join(',');
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
