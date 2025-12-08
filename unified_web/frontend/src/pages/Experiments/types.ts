/**
 * 类型定义
 */

// 实验类型
export type ExperimentType = 'kcin' | 'dcin';

export interface Experiment {
  id: number;
  name: string;
  experiment_type: ExperimentType;
  created_at: string | null;
  description: string | null;
  config_path: string | null;
  topo_type: string | null;
  traffic_files: string[] | null;
  traffic_weights: number[] | null;
  simulation_time: number | null;
  n_repeats: number | null;
  n_jobs: number | null;
  status: string | null;
  total_combinations: number | null;
  completed_combinations: number | null;
  best_performance: number | null;
  git_commit: string | null;
  notes: string | null;
}

// 仿真结果类型（使用动态 JSON 参数）
export interface SimulationResult {
  id: number;
  experiment_id: number;
  created_at: string | null;
  performance: number;
  config_params: Record<string, number | string>;
  result_details: Record<string, unknown>;
  result_html?: string;
  result_files?: string[];
  error?: string;
}

// 分页响应类型
export interface ResultsPageResponse {
  results: SimulationResult[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

// 统计信息类型
export interface Statistics {
  experiment: Experiment;
  result_count: number;
  performance_distribution: PerformanceDistribution;
  param_keys: string[];
}

// 性能分布类型
export interface PerformanceDistribution {
  min: number;
  max: number;
  mean: number;
  count: number;
  histogram: [number, number, number][];
}

// 敏感性分析数据点
export interface SensitivityDataPoint {
  value: number;
  mean_performance: number;
  min_performance: number;
  max_performance: number;
  count: number;
}

// 敏感性分析响应
export interface SensitivityResponse {
  parameter: string;
  data: SensitivityDataPoint[];
}

// 筛选条件类型
export interface FilterCondition {
  [paramName: string]: [number, number];
}

// 按数据流对比响应类型
export interface TrafficCompareData {
  traffic_files: string[];
  experiments: Array<{ id: number; name: string }>;
  param_keys: string[];
  data: Array<Record<string, string | number | null>>;
}

