import axios from 'axios'
import { LevelConfig, TopologyData } from '../types'

const api = axios.create({
  baseURL: '/api',
})

// 生成拓扑数据
export async function generateTopology(
  levels: LevelConfig[],
  showInterLevel: boolean = true
): Promise<TopologyData> {
  const response = await api.post('/topology/generate', {
    levels,
    show_inter_level: showInterLevel,
    layout: 'circular',
  })
  return response.data
}

// 获取预设配置
export async function getPresets() {
  const response = await api.get('/topology/presets')
  return response.data.presets
}

// 获取颜色配置
export async function getColors() {
  const response = await api.get('/topology/colors')
  return response.data
}
