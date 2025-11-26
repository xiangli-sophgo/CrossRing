/**
 * 参数分类工具
 * 将仿真结果参数分为配置参数和结果统计两组
 */

// 配置参数关键词模式
const CONFIG_PARAM_PATTERNS = [
  /^TOPO_TYPE$/,
  /^FLIT_SIZE$/,
  /^BURST$/,
  /FREQUENCY/,
  /FIFO_DEPTH/,
  /LATENCY/,
  /_SIZE$/,
  /Etag/i,
  /ITag/i,
  /BW_LIMIT$/,
  /^ENABLE_/,
  /^ORDERING_/,
  /CROSSRING_VERSION/,
  /SLICE_PER_LINK/,
  /_RW_GAP$/,
  /IN_ORDER_/,
  /TAG_NUM/,
  /^RN_/,
  /^SN_/,
  /^UNIFIED_/,
];

// 结果统计关键词模式
const RESULT_STAT_PATTERNS = [
  /^带宽/,
  /^平均带宽/,
  /^总和带宽/,
  /延迟/,
  /完成时间/,
  /^绕环/,
  /^保序阻止/,
  /环次数/,
  /等待周期/,
  /重试次数/,
  /ETag.*num/i,
  /ITag.*num/i,
  /flit数/,
  /^模型类型$/,
  /^拓扑类型$/,
  /^数据流名称$/,
];

/**
 * 判断参数是否为配置参数
 */
function isConfigParam(key: string): boolean {
  return CONFIG_PARAM_PATTERNS.some((pattern) => pattern.test(key));
}

/**
 * 判断参数是否为结果统计
 */
function isResultStat(key: string): boolean {
  return RESULT_STAT_PATTERNS.some((pattern) => pattern.test(key));
}

/**
 * 分类参数
 */
export function classifyParams(params: Record<string, unknown>): {
  configParams: Record<string, unknown>;
  resultStats: Record<string, unknown>;
} {
  const configParams: Record<string, unknown> = {};
  const resultStats: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(params)) {
    // 跳过复杂对象（如 port_averages、circling_eject_stats 等）
    if (typeof value === 'object' && value !== null) {
      continue;
    }

    if (isResultStat(key)) {
      resultStats[key] = value;
    } else if (isConfigParam(key)) {
      configParams[key] = value;
    } else {
      // 默认归类为配置参数
      configParams[key] = value;
    }
  }

  return { configParams, resultStats };
}

/**
 * 格式化参数值
 */
export function formatParamValue(value: unknown): string {
  if (value === null || value === undefined) {
    return '-';
  }
  if (typeof value === 'number') {
    // 检查是否为浮点数
    if (!Number.isInteger(value)) {
      return value.toFixed(4);
    }
    return value.toString();
  }
  if (typeof value === 'boolean') {
    return value ? '是' : '否';
  }
  return String(value);
}
