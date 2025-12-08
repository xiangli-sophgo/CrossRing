import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface LayoutState {
  // 单Die拓扑图容器尺寸
  singleDieWidth: number
  singleDieHeight: number
  // 多Die拓扑图容器尺寸
  multiDieWidth: number
  multiDieHeight: number
  // 设置单Die拓扑图尺寸
  setSingleDieSize: (width: number, height: number) => void
  // 设置多Die拓扑图尺寸
  setMultiDieSize: (width: number, height: number) => void
  // 重置为默认值
  resetLayout: () => void
}

const DEFAULT_SINGLE_DIE_WIDTH = 600
const DEFAULT_SINGLE_DIE_HEIGHT = 500
const DEFAULT_MULTI_DIE_WIDTH = 800
const DEFAULT_MULTI_DIE_HEIGHT = 600

export const useLayoutStore = create<LayoutState>()(
  persist(
    (set) => ({
      singleDieWidth: DEFAULT_SINGLE_DIE_WIDTH,
      singleDieHeight: DEFAULT_SINGLE_DIE_HEIGHT,
      multiDieWidth: DEFAULT_MULTI_DIE_WIDTH,
      multiDieHeight: DEFAULT_MULTI_DIE_HEIGHT,
      setSingleDieSize: (width, height) => set({ singleDieWidth: width, singleDieHeight: height }),
      setMultiDieSize: (width, height) => set({ multiDieWidth: width, multiDieHeight: height }),
      resetLayout: () => set({
        singleDieWidth: DEFAULT_SINGLE_DIE_WIDTH,
        singleDieHeight: DEFAULT_SINGLE_DIE_HEIGHT,
        multiDieWidth: DEFAULT_MULTI_DIE_WIDTH,
        multiDieHeight: DEFAULT_MULTI_DIE_HEIGHT,
      }),
    }),
    {
      name: 'crossring-layout-storage',
      version: 2,
    }
  )
)
