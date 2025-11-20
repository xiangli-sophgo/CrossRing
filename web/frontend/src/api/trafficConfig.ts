import axios from 'axios'
import type {
  TrafficConfigCreate,
  BatchTrafficConfigCreate,
  TrafficConfigResponse,
  TrafficConfigListResponse,
  TrafficConfig,
  TrafficGenerateRequest,
  TrafficGenerateResponse
} from '../types/trafficConfig'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const client = axios.create({
  baseURL: API_BASE_URL,
})

export const createTrafficConfig = async (request: TrafficConfigCreate): Promise<TrafficConfigResponse> => {
  const response = await client.post('/api/traffic/config/', request)
  return response.data
}

export const createBatchTrafficConfig = async (request: BatchTrafficConfigCreate): Promise<TrafficConfigResponse> => {
  const response = await client.post('/api/traffic/config/batch', request)
  return response.data
}

export const getTrafficConfigs = async (topology: string, mode: string): Promise<TrafficConfigListResponse> => {
  const response = await client.get(`/api/traffic/config/${topology}/${mode}`)
  return response.data
}

export const getTrafficConfig = async (topology: string, mode: string, configId: string): Promise<TrafficConfig> => {
  const response = await client.get(`/api/traffic/config/${topology}/${mode}/${configId}`)
  return response.data
}

export const updateTrafficConfig = async (
  topology: string,
  mode: string,
  configId: string,
  request: TrafficConfigCreate
): Promise<TrafficConfigResponse> => {
  const response = await client.put(`/api/traffic/config/${topology}/${mode}/${configId}`, request)
  return response.data
}

export const deleteTrafficConfig = async (topology: string, mode: string, configId: string) => {
  const response = await client.delete(`/api/traffic/config/${topology}/${mode}/${configId}`)
  return response.data
}

export const clearAllConfigs = async (topology: string, mode: string) => {
  const response = await client.delete(`/api/traffic/config/${topology}/${mode}`)
  return response.data
}

export const generateTraffic = async (request: TrafficGenerateRequest): Promise<TrafficGenerateResponse> => {
  const response = await client.post('/api/traffic/generate/', request)
  return response.data
}

export const downloadTrafficFile = (filename: string) => {
  window.open(`${API_BASE_URL}/api/traffic/generate/download/${filename}`, '_blank')
}

export const listGeneratedFiles = async () => {
  const response = await client.get('/api/traffic/generate/list')
  return response.data
}

export const listConfigFiles = async () => {
  const response = await client.get('/api/traffic/config/files/list')
  return response.data
}

export const loadConfigsFromFile = async (
  topology: string,
  mode: string,
  filename: string,
  loadMode: 'replace' | 'append'
) => {
  const response = await client.post(`/api/traffic/config/${topology}/${mode}/load`, {
    filename,
    mode: loadMode
  })
  return response.data
}

export const saveConfigsToFile = async (topology: string, mode: string, filename: string) => {
  const response = await client.post(`/api/traffic/config/${topology}/${mode}/save`, { filename })
  return response.data
}
