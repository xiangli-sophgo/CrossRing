/**
 * 实验状态管理
 */

import { create } from 'zustand';
import type { Experiment, Statistics, FilterCondition } from '../types';

interface ExperimentState {
  // 实验列表
  experiments: Experiment[];
  loading: boolean;
  error: string | null;

  // 当前实验
  currentExperiment: Experiment | null;
  currentStatistics: Statistics | null;

  // 筛选条件
  filters: FilterCondition;

  // 选中的实验（用于对比）
  selectedExperimentIds: number[];

  // Actions
  setExperiments: (experiments: Experiment[]) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setCurrentExperiment: (experiment: Experiment | null) => void;
  setCurrentStatistics: (statistics: Statistics | null) => void;
  setFilters: (filters: FilterCondition) => void;
  updateFilter: (paramName: string, range: [number, number] | null) => void;
  clearFilters: () => void;
  toggleExperimentSelection: (id: number) => void;
  clearSelection: () => void;
}

export const useExperimentStore = create<ExperimentState>((set) => ({
  // 初始状态
  experiments: [],
  loading: false,
  error: null,
  currentExperiment: null,
  currentStatistics: null,
  filters: {},
  selectedExperimentIds: [],

  // Actions
  setExperiments: (experiments) => set({ experiments }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  setCurrentExperiment: (experiment) => set({ currentExperiment: experiment }),
  setCurrentStatistics: (statistics) => set({ currentStatistics: statistics }),
  setFilters: (filters) => set({ filters }),

  updateFilter: (paramName, range) =>
    set((state) => {
      const newFilters = { ...state.filters };
      if (range === null) {
        delete newFilters[paramName];
      } else {
        newFilters[paramName] = range;
      }
      return { filters: newFilters };
    }),

  clearFilters: () => set({ filters: {} }),

  toggleExperimentSelection: (id) =>
    set((state) => {
      const newSelection = state.selectedExperimentIds.includes(id)
        ? state.selectedExperimentIds.filter((i) => i !== id)
        : [...state.selectedExperimentIds, id];
      return { selectedExperimentIds: newSelection };
    }),

  clearSelection: () => set({ selectedExperimentIds: [] }),
}));
