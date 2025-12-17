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

// 重要统计的模式（KCIN和DCIN通用）
const IMPORTANT_PATTERNS = [
  /^模型类型$/, /^拓扑类型$/, /^数据流名称$/, /^Die数量$/, /^仿真周期$/,
  // 带宽：{IP}_带宽 或 {IP}_{操作}_带宽（包括D2D_RN, D2D_SN）
  /^[A-Z0-9_]+_带宽$/, /^[A-Z0-9_]+_(读|写)_带宽$/, /^总带宽$/,
  // 带宽汇总统计
  /^带宽_加权$/, /^带宽_平均加权$/,
  // 请求数
  /^总读请求数$/, /^总写请求数$/, /^总请求数$/,
  /^跨Die/, /^读重试数$/, /^写重试数$/,
  // 绕环比例（整体省略方向）
  /^绕环_(横向|纵向)_比例$/, /^绕环_比例$/,
  // Tag统计
  /^ITag_(横向|纵向)$/, /^RB_ETag_T[01]$/, /^EQ_ETag_T[01]$/,
  // 延迟统计（使用中文：命令延迟、数据延迟、事务延迟，排除P95和P99）
  /^命令延迟(?!.*[Pp]9)/, /^数据延迟(?!.*[Pp]9)/, /^事务延迟(?!.*[Pp]9)/,
  // 注意：TOPO_TYPE, FLIT_SIZE, BURST, NETWORK_FREQUENCY 归类为配置参数
];

// 结果统计的模式（用于列分类的优先匹配）
const RESULT_PATTERNS = [
  /[Pp]9\d/, // P95, P99, p95, p99 等百分位数据优先匹配到结果统计
  /延迟/, /完成时间/, /^绕环/, /^保序阻止/, /环次数/, /等待周期/,
  /重试/, /ETag.*num/i, /ITag.*num/i, /flit数/,
  /_带宽_read$/, /_带宽_write$/, /_带宽_total$/,
  /^D2D.*请求数$/,  // D2D读请求数, D2D写请求数, D2D总请求数
  /带宽/, /请求数/,
];

// 配置参数的模式（用于列分类，精确匹配config_whitelist中的参数）
const CONFIG_PATTERNS_FOR_COLUMN = [
  /^TOPO_TYPE$/,
  /^FLIT_SIZE$/,
  /^BURST$/,
  /^NETWORK_FREQUENCY$/,
  /^SLICE_PER_LINK_/,
  /^RN_RDB_SIZE$/, /^RN_WDB_SIZE$/,
  /^SN_DDR_RDB_SIZE$/, /^SN_DDR_WDB_SIZE$/,
  /^SN_L2M_RDB_SIZE$/, /^SN_L2M_WDB_SIZE$/,
  /^UNIFIED_RW_TRACKER$/,
  /LATENCY_original$/,
  /FIFO_DEPTH$/,
  /Etag_T\d_UE_MAX$/,
  /^ETag_BOTHSIDE_UPGRADE$/,
  /^ETAG_T1_ENABLED$/,
  /^ITag_TRIGGER_Th_/,
  /^ITag_MAX_NUM_/,
  /^ENABLE_CROSSPOINT_CONFLICT_CHECK$/,
  /^ORDERING_/,
  /BW_LIMIT$/,
  /^IN_ORDER_/,
];

/**
 * 对Die内部列进行二级分类
 * @param columnName 完整的列名（包含Die{n}_前缀）
 * @returns 分类：important | result | config
 */
export function classifyDieColumn(columnName: string): 'important' | 'result' | 'config' {
  // 去掉Die{n}_前缀后判断
  const nameWithoutPrefix = columnName.replace(/^Die\d+_/, '');

  if (IMPORTANT_PATTERNS.some(p => p.test(nameWithoutPrefix))) {
    return 'important';
  }
  if (CONFIG_PATTERNS_FOR_COLUMN.some(p => p.test(nameWithoutPrefix))) {
    return 'config';
  }
  // 不是配置参数的默认归类为结果统计
  return 'result';
}

// Die层级分类结果的类型定义
export interface DieHierarchy {
  important: string[];
  result: string[];
  config: string[];
}

export interface HierarchicalClassification {
  important: string[];
  result: string[];
  config: string[];
  [key: `die${number}`]: DieHierarchy;
}

/**
 * 带层级结构的参数键分类
 * @param paramKeys 所有参数键列表
 * @returns 层级分类结果
 */
export function classifyParamKeysWithHierarchy(paramKeys: string[]): HierarchicalClassification {
  const result: HierarchicalClassification = {
    important: [],
    result: [],
    config: [],
  };

  // 动态初始化Die分组（支持0-9）
  for (let i = 0; i <= 9; i++) {
    result[`die${i}`] = {
      important: [],
      result: [],
      config: [],
    };
  }

  for (const key of paramKeys) {
    // 检查是否是Die列
    const dieMatch = key.match(/^Die(\d+)_/);
    if (dieMatch) {
      const dieNum = parseInt(dieMatch[1], 10);
      const subCategory = classifyDieColumn(key);
      if (result[`die${dieNum}`]) {
        result[`die${dieNum}`][subCategory].push(key);
      }
    } else if (IMPORTANT_PATTERNS.some(p => p.test(key))) {
      result.important.push(key);
    } else if (CONFIG_PATTERNS_FOR_COLUMN.some(p => p.test(key))) {
      result.config.push(key);
    } else {
      // 不是配置参数的默认归类为结果统计
      result.result.push(key);
    }
  }

  return result;
}

// 列分类配置（简化为：重要统计、Die分组、配置参数、结果统计）
export const PARAM_CATEGORIES: ParamCategory[] = [
  {
    key: 'important',
    label: '重要统计',
    patterns: IMPORTANT_PATTERNS,
  },
  {
    key: 'die0',
    label: 'Die0统计',
    patterns: [/^Die0_/],
  },
  {
    key: 'die1',
    label: 'Die1统计',
    patterns: [/^Die1_/],
  },
  {
    key: 'die2',
    label: 'Die2统计',
    patterns: [/^Die2_/],
  },
  {
    key: 'die3',
    label: 'Die3统计',
    patterns: [/^Die3_/],
  },
  {
    key: 'config',
    label: '配置参数',
    patterns: CONFIG_PATTERNS_FOR_COLUMN,
  },
  {
    key: 'result',
    label: '结果统计',
    patterns: [/.*/], // 匹配所有剩余参数（不是配置参数的都是结果统计）
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

// 配置参数关键词模式（仅匹配 config_whitelist 中定义的参数）
const CONFIG_PARAM_PATTERNS = [
  /^TOPO_TYPE$/,
  /^FLIT_SIZE$/,
  /^BURST$/,
  /^NETWORK_FREQUENCY$/,
  /^SLICE_PER_LINK_/,
  // Resource configuration
  /^RN_RDB_SIZE$/, /^RN_WDB_SIZE$/,
  /^SN_DDR_RDB_SIZE$/, /^SN_DDR_WDB_SIZE$/,
  /^SN_L2M_RDB_SIZE$/, /^SN_L2M_WDB_SIZE$/,
  /^UNIFIED_RW_TRACKER$/,
  // Latency configuration
  /LATENCY_original$/,
  // FIFO depths
  /FIFO_DEPTH$/,
  // ETag configuration (配置参数以_MAX结尾)
  /Etag_T\d_UE_MAX$/,
  /^ETag_BOTHSIDE_UPGRADE$/,
  /^ETAG_T1_ENABLED$/,
  // ITag configuration
  /^ITag_TRIGGER_Th_/,
  /^ITag_MAX_NUM_/,
  // Feature switches
  /^ENABLE_CROSSPOINT_CONFLICT_CHECK$/,
  /^ORDERING_/,
  // Bandwidth limits
  /BW_LIMIT$/,
  // Other configurations
  /^IN_ORDER_/,
];

// 结果统计关键词模式
const RESULT_STAT_PATTERNS = [
  /带宽/,
  /延迟/,
  /完成时间/,
  /^绕环/,
  /^保序阻止/,
  /环次数/,
  /等待周期/,
  /重试/,
  /ETag.*num/i,
  /ITag.*num/i,
  /flit数/,
  /^模型类型$/,
  /^拓扑类型$/,
  /^数据流名称$/,
  /^仿真周期$/,
  /请求数/,
  /^跨Die/,
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

    if (isConfigParam(key)) {
      configParams[key] = value;
    } else {
      // 不是配置参数的都归类为结果统计
      resultStats[key] = value;
    }
  }

  return { configParams, resultStats };
}

/**
 * 格式化参数值
 * @param value 参数值
 * @param key 参数键名（可选，用于判断是否为比例类数据）
 */
export function formatParamValue(value: unknown, key?: string): string {
  if (value === null || value === undefined) {
    return '-';
  }
  if (typeof value === 'number') {
    // 检查是否为比例类数据（包含"比例"或"ratio"的键名，且值在0-1之间）
    if (key && (key.includes('比例') || key.toLowerCase().includes('ratio')) && value >= 0 && value <= 1) {
      return `${(value * 100).toFixed(2)}%`;
    }
    // 检查是否为浮点数
    if (!Number.isInteger(value)) {
      return value.toFixed(4);
    }
    return value.toString();
  }
  if (typeof value === 'boolean') {
    return value ? '是' : '否';
  }
  // 数据流名称只显示最后的文件名
  if (key && (key === '数据流名称' || key === 'file_name' || key === 'TRAFFIC_FILE' || key === 'traffic_file')) {
    const strValue = String(value);
    if (strValue.includes('/') || strValue.includes('\\')) {
      return strValue.split('/').pop()?.split('\\').pop() || strValue;
    }
  }
  return String(value);
}
