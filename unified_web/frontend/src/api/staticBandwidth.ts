import axios from 'axios'
import type { BandwidthComputeRequest, BandwidthComputeResponse } from '../types/staticBandwidth'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8002'

const client = axios.create({
  baseURL: API_BASE_URL,
})

/**
 * 获取可用的DCIN配置文件列表
 */
export const getDCINConfigs = async (): Promise<{
  configs: Array<{ filename: string; num_dies: number; connections: number }>
}> => {
  const response = await client.get('/api/traffic/bandwidth/dcin-configs')
  return response.data
}

/**
 * 计算静态链路带宽
 */
export const computeStaticBandwidth = async (
  request: BandwidthComputeRequest
): Promise<BandwidthComputeResponse> => {
  const response = await client.post('/api/traffic/bandwidth/compute', request)
  return response.data
}
