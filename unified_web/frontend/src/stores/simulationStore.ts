/**
 * 仿真页面状态管理 - 缓存配置和流量文件数据
 */

import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { ConfigOption, TrafficTreeNode } from '../api/simulation'

interface FormValues {
  mode?: 'kcin' | 'dcin'
  rows?: number
  cols?: number
  config_path?: string
  die_config_path?: string
  max_time?: number
  max_workers?: number
  save_to_db?: boolean
  experiment_name?: string
  experiment_description?: string
}

interface SimulationState {
  // 配置选项缓存
  configs: { kcin: ConfigOption[]; dcin: ConfigOption[] }
  configsLoaded: boolean

  // 流量文件树缓存
  trafficTree: TrafficTreeNode[]
  trafficTreeMode: 'kcin' | 'dcin' | null
  trafficTreeLoaded: boolean

  // 用户编辑的配置值缓存
  configValues: Record<string, any>
  dieConfigValues: Record<string, any>
  // 原始配置值（用于对比修改）
  originalConfigValues: Record<string, any>
  originalDieConfigValues: Record<string, any>
  selectedFiles: string[]
  formValues: FormValues

  // Actions
  setConfigs: (configs: { kcin: ConfigOption[]; dcin: ConfigOption[] }) => void
  setTrafficTree: (tree: TrafficTreeNode[], mode: 'kcin' | 'dcin') => void
  clearTrafficTree: () => void
  setConfigValues: (values: Record<string, any>) => void
  updateConfigValue: (key: string, value: any) => void
  setDieConfigValues: (values: Record<string, any>) => void
  updateDieConfigValue: (key: string, value: any) => void
  setOriginalConfigValues: (values: Record<string, any>) => void
  setOriginalDieConfigValues: (values: Record<string, any>) => void
  setSelectedFiles: (files: string[]) => void
  setFormValues: (values: FormValues) => void
  updateFormValue: <K extends keyof FormValues>(key: K, value: FormValues[K]) => void
}

export const useSimulationStore = create<SimulationState>()(
  persist(
    (set) => ({
      // 初始状态
      configs: { kcin: [], dcin: [] },
      configsLoaded: false,
      trafficTree: [],
      trafficTreeMode: null,
      trafficTreeLoaded: false,
      configValues: {},
      dieConfigValues: {},
      originalConfigValues: {},
      originalDieConfigValues: {},
      selectedFiles: [],
      formValues: { mode: 'kcin', rows: 5, cols: 4, max_time: 6000, max_workers: 8, save_to_db: true },

      // Actions
      setConfigs: (configs) => set({ configs, configsLoaded: true }),
      setTrafficTree: (tree, mode) => set({ trafficTree: tree, trafficTreeMode: mode, trafficTreeLoaded: true }),
      clearTrafficTree: () => set({ trafficTree: [], trafficTreeMode: null, trafficTreeLoaded: false }),
      setConfigValues: (values) => set({ configValues: values }),
      updateConfigValue: (key, value) => set((state) => ({ configValues: { ...state.configValues, [key]: value } })),
      setDieConfigValues: (values) => set({ dieConfigValues: values }),
      updateDieConfigValue: (key, value) => set((state) => ({ dieConfigValues: { ...state.dieConfigValues, [key]: value } })),
      setOriginalConfigValues: (values) => set({ originalConfigValues: values }),
      setOriginalDieConfigValues: (values) => set({ originalDieConfigValues: values }),
      setSelectedFiles: (files) => set({ selectedFiles: files }),
      setFormValues: (values) => set({ formValues: values }),
      updateFormValue: (key, value) => set((state) => ({ formValues: { ...state.formValues, [key]: value } })),
    }),
    {
      name: 'simulation-store',
      // 只持久化用户编辑的配置值和表单值，不持久化缓存数据
      partialize: (state) => ({
        configValues: state.configValues,
        dieConfigValues: state.dieConfigValues,
        originalConfigValues: state.originalConfigValues,
        originalDieConfigValues: state.originalDieConfigValues,
        selectedFiles: state.selectedFiles,
        formValues: state.formValues,
      }),
    }
  )
)
