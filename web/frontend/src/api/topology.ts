import axios from 'axios'
import type { TopologyData, TopologyInfo } from '../types/topology'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const client = axios.create({
  baseURL: API_BASE_URL,
})

export const getAvailableTopologies = async (): Promise<{ topologies: TopologyInfo[] }> => {
  const response = await client.get('/api/topology/')
  return response.data
}

export const getTopology = async (topoType: string): Promise<TopologyData> => {
  const response = await client.get(`/api/topology/${topoType}`)
  return response.data
}

export const getNodeInfo = async (topoType: string, nodeId: number) => {
  const response = await client.get(`/api/topology/${topoType}/nodes/${nodeId}`)
  return response.data
}
