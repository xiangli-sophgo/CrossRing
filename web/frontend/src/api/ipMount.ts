import axios from 'axios'
import type {
  IPMountRequest,
  BatchMountRequest,
  IPMountResponse,
  IPMountListResponse
} from '../types/ipMount'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

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

export const deleteMount = async (topology: string, nodeId: number) => {
  const response = await client.delete(`/api/ip-mount/${topology}/nodes/${nodeId}`)
  return response.data
}

export const clearAllMounts = async (topology: string) => {
  const response = await client.delete(`/api/ip-mount/${topology}`)
  return response.data
}
