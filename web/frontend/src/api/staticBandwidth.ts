import axios from 'axios'
import type { BandwidthComputeRequest, BandwidthComputeResponse } from '../types/staticBandwidth'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const client = axios.create({
  baseURL: API_BASE_URL,
})

/**
 * 计算静态链路带宽
 */
export const computeStaticBandwidth = async (
  request: BandwidthComputeRequest
): Promise<BandwidthComputeResponse> => {
  const response = await client.post('/api/traffic/bandwidth/compute', request)
  return response.data
}
