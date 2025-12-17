/**
 * 仿真页面状态管理 - 缓存配置和流量文件数据
 */

import { create } from 'zustand'
import type { ConfigOption, TrafficTreeNode } from '../api/simulation'

interface SimulationState {
  // 配置选项缓存
  configs: { kcin: ConfigOption[]; dcin: ConfigOption[] }
  configsLoaded: boolean

  // 流量文件树缓存
  trafficTree: TrafficTreeNode[]
  trafficTreeMode: 'kcin' | 'dcin' | null
  trafficTreeLoaded: boolean

  // Actions
  setConfigs: (configs: { kcin: ConfigOption[]; dcin: ConfigOption[] }) => void
  setTrafficTree: (tree: TrafficTreeNode[], mode: 'kcin' | 'dcin') => void
  clearTrafficTree: () => void
}

export const useSimulationStore = create<SimulationState>((set) => ({
  // 初始状态
  configs: { kcin: [], dcin: [] },
  configsLoaded: false,
  trafficTree: [],
  trafficTreeMode: null,
  trafficTreeLoaded: false,

  // Actions
  setConfigs: (configs) => set({ configs, configsLoaded: true }),
  setTrafficTree: (tree, mode) => set({ trafficTree: tree, trafficTreeMode: mode, trafficTreeLoaded: true }),
  clearTrafficTree: () => set({ trafficTree: [], trafficTreeMode: null, trafficTreeLoaded: false }),
}))
