/**
 * API 客户端
 */

import axios from 'axios';
import type {
  Experiment,
  ExperimentType,
  ResultsPageResponse,
  Statistics,
  SensitivityResponse,
  FilterCondition,
  TrafficCompareData,
} from '@/types';

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
});

// ==================== 实验管理 ====================

export const getExperiments = async (
  status?: string,
  experimentType?: ExperimentType
): Promise<Experiment[]> => {
  const params: Record<string, string> = {};
  if (status) params.status = status;
  if (experimentType) params.experiment_type = experimentType;
  const response = await api.get('/experiments', { params });
  return response.data;
};

export const getExperiment = async (id: number): Promise<Experiment> => {
  const response = await api.get(`/experiments/${id}`);
  return response.data;
};

export const createExperiment = async (data: {
  name: string;
  experiment_type?: ExperimentType;
  description?: string;
  topo_type?: string;
}): Promise<Experiment> => {
  const response = await api.post('/experiments', data);
  return response.data;
};

export const updateExperiment = async (
  id: number,
  data: { name?: string; description?: string; notes?: string }
): Promise<Experiment> => {
  const response = await api.put(`/experiments/${id}`, data);
  return response.data;
};

export const deleteExperiment = async (id: number): Promise<void> => {
  await api.delete(`/experiments/${id}`);
};

export const deleteExperimentsBatch = async (
  experimentIds: number[]
): Promise<{ success: boolean; message: string; deleted_count: number }> => {
  const response = await api.post('/experiments/batch-delete', {
    experiment_ids: experimentIds,
  });
  return response.data;
};

// ==================== 结果查询 ====================

export const getResults = async (
  experimentId: number,
  page: number = 1,
  pageSize: number = 100,
  sortBy: string = 'performance',
  order: 'asc' | 'desc' = 'desc',
  filters?: FilterCondition
): Promise<ResultsPageResponse> => {
  const params: Record<string, string | number> = {
    page,
    page_size: pageSize,
    sort_by: sortBy,
    order,
  };
  if (filters) {
    params.filters = JSON.stringify(filters);
  }
  const response = await api.get(`/experiments/${experimentId}/results`, { params });
  return response.data;
};

export const getBestResults = async (
  experimentId: number,
  limit: number = 10
): Promise<{ results: Record<string, unknown>[]; count: number }> => {
  const response = await api.get(`/experiments/${experimentId}/best`, {
    params: { limit },
  });
  return response.data;
};

export const getStatistics = async (experimentId: number): Promise<Statistics> => {
  const response = await api.get(`/experiments/${experimentId}/stats`);
  return response.data;
};

export const getParamKeys = async (
  experimentId: number
): Promise<{ param_keys: string[]; count: number }> => {
  const response = await api.get(`/experiments/${experimentId}/param-keys`);
  return response.data;
};

export interface TrafficStat {
  traffic_name: string;
  count: number;
  avg_performance: number;
  max_performance: number;
  min_performance: number;
}

export const getTrafficStats = async (
  experimentId: number
): Promise<{ experiment_id: number; total_results: number; traffic_stats: TrafficStat[] }> => {
  const response = await api.get(`/experiments/${experimentId}/traffic-stats`);
  return response.data;
};

export const getDistribution = async (
  experimentId: number,
  bins: number = 50
): Promise<{
  min: number;
  max: number;
  mean: number;
  count: number;
  histogram: [number, number, number][];
}> => {
  const response = await api.get(`/experiments/${experimentId}/distribution`, {
    params: { bins },
  });
  return response.data;
};

// ==================== 分析 ====================

export const getParameterSensitivity = async (
  experimentId: number,
  parameter: string,
  metric?: string
): Promise<SensitivityResponse> => {
  const response = await api.get(
    `/experiments/${experimentId}/sensitivity/${parameter}`,
    { params: metric ? { metric } : undefined }
  );
  return response.data;
};

export const getAllSensitivity = async (
  experimentId: number,
  metric?: string
): Promise<{ experiment_id: number; metric: string; parameters: Record<string, SensitivityResponse> }> => {
  const response = await api.get(`/experiments/${experimentId}/sensitivity`, {
    params: metric ? { metric } : undefined,
  });
  return response.data;
};

export interface ParameterInfluence {
  name: string;
  influence: number;
  between_variance: number;
  mean_range: number;
  value_count: number;
}

export interface InfluenceResponse {
  experiment_id: number;
  metric: string;
  total_variance: number;
  parameters: ParameterInfluence[];
}

export const getParameterInfluence = async (
  experimentId: number,
  metric?: string
): Promise<InfluenceResponse> => {
  const response = await api.get(`/experiments/${experimentId}/influence`, {
    params: metric ? { metric } : undefined,
  });
  return response.data;
};

export const compareExperiments = async (
  experimentIds: number[]
): Promise<{
  experiments: Experiment[];
  best_configs: Record<string, unknown>[];
}> => {
  const response = await api.post('/compare', { experiment_ids: experimentIds });
  return response.data;
};

export const compareByTraffic = async (
  experimentIds: number[]
): Promise<TrafficCompareData> => {
  const response = await api.post('/compare/traffic', { experiment_ids: experimentIds });
  return response.data;
};

export const getParameterHeatmap = async (
  experimentId: number,
  paramX: string,
  paramY: string,
  metric?: string
): Promise<{
  param_x: string;
  param_y: string;
  metric: string;
  x_values: number[];
  y_values: number[];
  data: { [key: string]: number | string }[];
}> => {
  const response = await api.get(`/experiments/${experimentId}/heatmap`, {
    params: { param_x: paramX, param_y: paramY, ...(metric && { metric }) },
  });
  return response.data;
};

// ==================== 导出 ====================

export interface ExportInfo {
  experiments_count: number;
  results_count: number;
  database_size: number;
  is_selective: boolean;
}

export const getExportInfo = async (
  experimentIds?: number[]
): Promise<ExportInfo> => {
  const params: Record<string, string> = {};
  if (experimentIds && experimentIds.length > 0) {
    params.experiment_ids = experimentIds.join(',');
  }
  const response = await api.get('/export/info', { params });
  return response.data;
};

export const downloadPackage = (
  experimentIds?: number[],
  includeFrontend: boolean = true,
  includeBackend: boolean = true
): string => {
  const params = new URLSearchParams();
  if (experimentIds && experimentIds.length > 0) {
    params.append('experiment_ids', experimentIds.join(','));
  }
  params.append('include_frontend', String(includeFrontend));
  params.append('include_backend', String(includeBackend));
  return `/api/export/download?${params.toString()}`;
};

export const buildExecutablePackage = (experimentIds?: number[]): string => {
  const params = new URLSearchParams();
  if (experimentIds && experimentIds.length > 0) {
    params.append('experiment_ids', experimentIds.join(','));
  }
  return `/api/export/build?${params.toString()}`;
};

// ==================== 实验导出/导入 ====================

/** 导出实验信息预估 */
export interface ExperimentExportInfo {
  experiments_count: number;
  results_count: number;
  estimated_size: number;
}

/** 获取导出预估信息 */
export const getExperimentExportInfo = async (
  experimentIds: number[]
): Promise<ExperimentExportInfo> => {
  const response = await api.get('/export/experiment/info', {
    params: { experiment_ids: experimentIds.join(',') },
  });
  return response.data;
};

/** 导出实验为ZIP包 */
export const exportExperiment = (experimentIds: number[]): string => {
  const params = new URLSearchParams();
  params.append('experiment_ids', experimentIds.join(','));
  return `/api/export/experiment?${params.toString()}`;
};

/** 导入包中的实验信息 */
export interface ImportExperimentInfo {
  id: number;
  name: string;
  experiment_type: string;
  results_count: number;
  conflict: boolean;
  existing_id?: number;
}

/** 检查导入包的响应 */
export interface CheckImportResult {
  valid: boolean;
  error?: string;
  package_info?: {
    version: string;
    export_time: string;
    source_platform: string;
  };
  experiments?: ImportExperimentInfo[];
  temp_file_id?: string;
}

/** 检查导入包 */
export const checkImportPackage = async (file: File): Promise<CheckImportResult> => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/import/experiment/check', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 60000, // 上传可能需要更长时间
  });
  return response.data;
};

/** 导入配置项 */
export interface ImportConfigItem {
  original_id: number;
  action: 'rename' | 'overwrite' | 'skip';
  new_name?: string;
}

/** 导入结果 */
export interface ImportResult {
  success: boolean;
  imported: Array<{
    original_id: number;
    new_id: number;
    name: string;
    results_count: number;
  }>;
  skipped: number[];
  errors: Array<{ original_id: number; error: string }>;
}

/** 执行导入 */
export const executeImport = async (
  tempFileId: string,
  importConfig: ImportConfigItem[]
): Promise<ImportResult> => {
  const response = await api.post('/import/experiment/execute', {
    temp_file_id: tempFileId,
    import_config: importConfig,
  });
  return response.data;
};

// ==================== 结果文件操作 ====================

export const getResultHtmlUrl = (resultId: number, experimentId: number): string => {
  return `/api/results/${resultId}/html?experiment_id=${experimentId}`;
};

export const openLocalFile = async (path: string): Promise<{ success: boolean; message: string }> => {
  const response = await api.post('/open-file', { path });
  return response.data;
};

export const openFileDirectory = async (path: string): Promise<{ success: boolean; message: string }> => {
  const response = await api.post('/open-directory', { path });
  return response.data;
};

export interface ResultFileInfo {
  id: number;
  file_name: string;
  file_path: string;
  mime_type: string;
  file_size: number;
}

export const getResultFiles = async (
  resultId: number,
  resultType: string
): Promise<ResultFileInfo[]> => {
  const response = await api.get(`/results/${resultId}/files`, {
    params: { result_type: resultType },
  });
  return response.data;
};

export const getFileDownloadUrl = (fileId: number): string => {
  return `/api/files/${fileId}`;
};

export const getFileViewUrl = (fileId: number): string => {
  return `/api/files/${fileId}/view`;
};

export const deleteResult = async (
  resultId: number,
  experimentId: number
): Promise<{ success: boolean; message: string }> => {
  const response = await api.delete(`/results/${resultId}`, {
    params: { experiment_id: experimentId },
  });
  return response.data;
};

export const deleteResultsBatch = async (
  experimentId: number,
  resultIds: number[]
): Promise<{ success: boolean; message: string; deleted_count: number }> => {
  const response = await api.post(`/experiments/${experimentId}/results/batch-delete`, {
    result_ids: resultIds,
  });
  return response.data;
};

// ==================== 分析图表配置 ====================

export interface ChartConfig {
  id: number;
  name: string;
  chart_type: 'line' | 'heatmap' | 'sensitivity';
  config: {
    params?: string[];
    metric?: string;
    [key: string]: unknown;
  };
  sort_order: number;
  created_at?: string;
  updated_at?: string;
}

export const getAnalysisCharts = async (experimentId: number): Promise<ChartConfig[]> => {
  const response = await api.get(`/experiments/${experimentId}/charts`);
  return response.data;
};

export const addAnalysisChart = async (
  experimentId: number,
  data: {
    name: string;
    chart_type: 'line' | 'heatmap' | 'sensitivity';
    config: Record<string, unknown>;
    sort_order?: number;
  }
): Promise<ChartConfig> => {
  const response = await api.post(`/experiments/${experimentId}/charts`, data);
  return response.data;
};

export const updateAnalysisChart = async (
  experimentId: number,
  chartId: number,
  data: {
    name?: string;
    config?: Record<string, unknown>;
    sort_order?: number;
  }
): Promise<ChartConfig> => {
  const response = await api.put(`/experiments/${experimentId}/charts/${chartId}`, data);
  return response.data;
};

export const deleteAnalysisChart = async (
  experimentId: number,
  chartId: number
): Promise<{ success: boolean }> => {
  const response = await api.delete(`/experiments/${experimentId}/charts/${chartId}`);
  return response.data;
};

export default api;
