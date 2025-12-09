import apiClient from './client'
import type { BandwidthComputeRequest, BandwidthComputeResponse } from '../types/staticBandwidth'

/**
 * 获取可用的DCIN配置文件列表
 */
export const getDCINConfigs = async (): Promise<{
  configs: Array<{ filename: string; num_dies: number; connections: number }>
}> => {
  const response = await apiClient.get('/api/traffic/bandwidth/dcin-configs')
  return response.data
}

/**
 * 计算静态链路带宽
 */
export const computeStaticBandwidth = async (
  request: BandwidthComputeRequest
): Promise<BandwidthComputeResponse> => {
  const response = await apiClient.post('/api/traffic/bandwidth/compute', request)
  return response.data
}
