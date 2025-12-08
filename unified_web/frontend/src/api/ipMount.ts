import axios from 'axios'
import type {
  IPMountRequest,
  BatchMountRequest,
  IPMountResponse,
  IPMountListResponse
} from '../types/ipMount'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8002'

const client = axios.create({
  baseURL: API_BASE_URL,
})

export const mountIP = async (request: IPMountRequest): Promise<IPMountResponse> => {
  const response = await client.post('/api/ip-mount/', request)
  return response.data
}

export const batchMountIP = async (request: BatchMountRequest): Promise<IPMountResponse> => {
  const response = await client.post('/api/ip-mount/batch', request)
  return response.data
}

export const getMounts = async (topology: string): Promise<IPMountListResponse> => {
  const response = await client.get(`/api/ip-mount/${topology}`)
  return response.data
}

export const deleteMount = async (topology: string, nodeId: number, ipType?: string) => {
  const url = ipType
    ? `/api/ip-mount/${topology}/nodes/${nodeId}?ip_type=${ipType}`
    : `/api/ip-mount/${topology}/nodes/${nodeId}`
  const response = await client.delete(url)
  return response.data
}

export const clearAllMounts = async (topology: string) => {
  const response = await client.delete(`/api/ip-mount/${topology}`)
  return response.data
}

export const saveMountsToFile = async (topology: string, filename: string) => {
  const response = await client.post(`/api/ip-mount/${topology}/save`, { filename })
  return response.data
}

export const listMountFiles = async () => {
  const response = await client.get('/api/ip-mount/files/list')
  return response.data
}

export const loadMountsFromFile = async (topology: string, filename: string) => {
  const response = await client.post(`/api/ip-mount/${topology}/load`, { filename })
  return response.data
}
