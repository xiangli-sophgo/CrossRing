/**
 * 参数分类工具
 * 将仿真结果参数分为配置参数和结果统计两组
 */

// 列分类定义
export interface ParamCategory {
  key: string;
  label: string;
  patterns: RegExp[];
}

// 列分类配置
export const PARAM_CATEGORIES: ParamCategory[] = [
  {
    key: 'basic',
    label: '基础信息',
    patterns: [/^模型类型$/, /^拓扑类型$/, /^数据流名称$/, /^TOPO_TYPE$/, /^FLIT_SIZE$/, /^BURST$/, /^NETWORK_FREQUENCY$/],
  },
  {
    key: 'fifo',
    label: 'FIFO配置',
    patterns: [/FIFO_DEPTH/],
  },
  {
    key: 'etag',
    label: 'ETag配置',
    patterns: [/^TL_Etag/, /^TR_Etag/, /^TU_Etag/, /^TD_Etag/, /ETag_BOTHSIDE/],
  },
  {
    key: 'itag',
    label: 'ITag配置',
    patterns: [/^ITag_/],
  },
  {
    key: 'latency_config',
    label: '延迟配置',
    patterns: [/LATENCY/],
  },
  {
    key: 'resource',
    label: '资源配置',
    patterns: [/^RN_/, /^SN_/, /^UNIFIED_/, /_SIZE$/, /TRACKER/, /SLICE_PER_LINK/],
  },
  {
    key: 'bw_limit',
    label: '带宽限制',
    patterns: [/_BW_LIMIT$/, /_RW_GAP$/],
  },
  {
    key: 'bw_result',
    label: '带宽结果',
    patterns: [/^带宽/, /^平均带宽/, /^总和带宽/],
  },
  {
    key: 'latency_result',
    label: '延迟结果',
    patterns: [/延迟/],
  },
  {
    key: 'circling',
    label: '绕环统计',
    patterns: [/^绕环/, /环次数/, /^保序阻止/],
  },
  {
    key: 'other',
    label: '其他统计',
    patterns: [/完成时间/, /重试次数/, /flit数/, /等待周期/, /ETag.*num/i, /ITag.*num/i],
  },
];

/**
 * 将参数键分类到各个类别
 * @param paramKeys 所有参数键列表
 * @returns 分类映射 { categoryKey: [keys] }
 */
export function classifyParamKeys(paramKeys: string[]): Record<string, string[]> {
  const result: Record<string, string[]> = {};
  const classified = new Set<string>();

  // 初始化所有分类
  for (const category of PARAM_CATEGORIES) {
    result[category.key] = [];
  }
  result['uncategorized'] = [];

  // 对每个参数键进行分类
  for (const key of paramKeys) {
    let found = false;
    for (const category of PARAM_CATEGORIES) {
      if (category.patterns.some(pattern => pattern.test(key))) {
        result[category.key].push(key);
        classified.add(key);
        found = true;
        break;
      }
    }
    if (!found) {
      result['uncategorized'].push(key);
    }
  }

  return result;
}

/**
 * 获取分类选项（用于下拉选择）
 * @param paramKeys 所有参数键列表
 * @returns 分类选项数组
 */
export function getCategoryOptions(paramKeys: string[]): { value: string; label: string; count: number }[] {
  const classified = classifyParamKeys(paramKeys);
  const options: { value: string; label: string; count: number }[] = [];

  for (const category of PARAM_CATEGORIES) {
    const count = classified[category.key]?.length || 0;
    if (count > 0) {
      options.push({
        value: category.key,
        label: `${category.label} (${count})`,
        count,
      });
    }
  }

  // 添加未分类
  const uncategorizedCount = classified['uncategorized']?.length || 0;
  if (uncategorizedCount > 0) {
    options.push({
      value: 'uncategorized',
      label: `未分类 (${uncategorizedCount})`,
      count: uncategorizedCount,
    });
  }

  return options;
}

/**
 * 根据选中的分类获取要显示的参数键
 * @param paramKeys 所有参数键列表
 * @param selectedCategories 选中的分类键列表
 * @returns 要显示的参数键列表
 */
export function getVisibleParamKeys(paramKeys: string[], selectedCategories: string[]): string[] {
  if (selectedCategories.length === 0) {
    return paramKeys;
  }

  const classified = classifyParamKeys(paramKeys);
  const visibleKeys: string[] = [];

  for (const category of selectedCategories) {
    if (classified[category]) {
      visibleKeys.push(...classified[category]);
    }
  }

  return visibleKeys;
}

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
