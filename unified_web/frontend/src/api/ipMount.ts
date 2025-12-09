import apiClient from './client'
import type {
  IPMountRequest,
  BatchMountRequest,
  IPMountResponse,
  IPMountListResponse
} from '../types/ipMount'

export const mountIP = async (request: IPMountRequest): Promise<IPMountResponse> => {
  const response = await apiClient.post('/api/ip-mount/', request)
  return response.data
}

export const batchMountIP = async (request: BatchMountRequest): Promise<IPMountResponse> => {
  const response = await apiClient.post('/api/ip-mount/batch', request)
  return response.data
}

export const getMounts = async (topology: string): Promise<IPMountListResponse> => {
  const response = await apiClient.get(`/api/ip-mount/${topology}`)
  return response.data
}

export const deleteMount = async (topology: string, nodeId: number, ipType?: string) => {
  const url = ipType
    ? `/api/ip-mount/${topology}/nodes/${nodeId}?ip_type=${ipType}`
    : `/api/ip-mount/${topology}/nodes/${nodeId}`
  const response = await apiClient.delete(url)
  return response.data
}

export const clearAllMounts = async (topology: string) => {
  const response = await apiClient.delete(`/api/ip-mount/${topology}`)
  return response.data
}

export const saveMountsToFile = async (topology: string, filename: string) => {
  const response = await apiClient.post(`/api/ip-mount/${topology}/save`, { filename })
  return response.data
}

export const listMountFiles = async () => {
  const response = await apiClient.get('/api/ip-mount/files/list')
  return response.data
}

export const loadMountsFromFile = async (topology: string, filename: string) => {
  const response = await apiClient.post(`/api/ip-mount/${topology}/load`, { filename })
  return response.data
}
