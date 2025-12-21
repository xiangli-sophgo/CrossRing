/**
 * Simulation 页面辅助模块 - 类型、常量、工具函数
 */
import type { TaskStatus, TaskHistoryItem } from '@/api/simulation'

// ============== 类型定义 ==============

// 参数遍历配置类型
export interface SweepParam {
  key: string           // 参数名
  start: number         // 起始值
  end: number           // 结束值
  step: number          // 步长
  values: number[]      // 计算得到的值列表
  bindGroupId?: string  // 绑定组ID（相同ID的参数同步遍历）
}

// 参数遍历进度
export interface SweepProgress {
  total: number
  completed: number
  running: number
  pending: number
  failed: number
}

// 按批次分组的任务
export interface GroupedTask {
  key: string
  experiment_name: string
  mode: string
  topology: string
  created_at: string
  status: string
  tasks: TaskHistoryItem[]
  completed_count: number
  failed_count: number
  total_count: number
  experiment_id?: number
  errors: string[]  // 收集组内失败任务的错误信息
  is_sweep: boolean  // 是否为参数遍历
}

// 保存的遍历配置
export interface SavedSweepConfig {
  name: string
  params: SweepParam[]
}

// ============== 常量定义 ==============

// 绑定组背景颜色
export const BIND_GROUP_COLORS: Record<string, string> = {
  'A': '#e6f7ff',  // 浅蓝
  'B': '#f6ffed',  // 浅绿
  'C': '#fff7e6',  // 浅橙
  'D': '#f9f0ff',  // 浅紫
  'E': '#fff1f0',  // 浅红
  'F': '#e6fffb',  // 浅青
  'G': '#fcffe6',  // 浅黄绿
  'H': '#fff0f6',  // 浅粉
}

// 配置参数描述映射
export const CONFIG_TOOLTIPS: Record<string, string> = {
  // Basic Parameters
  FLIT_SIZE: 'Size of a single flit in bits',
  BURST: 'Number of flits per burst transfer',
  NETWORK_FREQUENCY: 'Network operating frequency in GHz',
  SLICE_PER_LINK_HORIZONTAL: 'Time slices per horizontal link',
  SLICE_PER_LINK_VERTICAL: 'Time slices per vertical link',
  // Buffer Size
  RN_RDB_SIZE: 'RN read data buffer size',
  RN_WDB_SIZE: 'RN write data buffer size',
  SN_DDR_RDB_SIZE: 'SN DDR read data buffer size',
  SN_DDR_WDB_SIZE: 'SN DDR write data buffer size',
  SN_L2M_RDB_SIZE: 'SN L2M read data buffer size',
  SN_L2M_WDB_SIZE: 'SN L2M write data buffer size',
  UNIFIED_RW_TRACKER: 'Enable unified read/write tracker pool (true) or separate pools (false)',
  // Latency
  DDR_R_LATENCY: 'DDR read latency in cycles',
  DDR_R_LATENCY_VAR: 'DDR read latency variance',
  DDR_W_LATENCY: 'DDR write latency in cycles',
  L2M_R_LATENCY: 'L2M read latency in cycles',
  L2M_W_LATENCY: 'L2M write latency in cycles',
  SN_TRACKER_RELEASE_LATENCY: 'SN tracker release latency in ns',
  SN_PROCESSING_LATENCY: 'SN processing latency in ns',
  RN_PROCESSING_LATENCY: 'RN processing latency in ns',
  // FIFO Depth
  IQ_CH_FIFO_DEPTH: 'Injection queue channel FIFO depth',
  EQ_CH_FIFO_DEPTH: 'Ejection queue channel FIFO depth',
  IQ_OUT_FIFO_DEPTH_HORIZONTAL: 'IQ output FIFO depth for horizontal direction',
  IQ_OUT_FIFO_DEPTH_VERTICAL: 'IQ output FIFO depth for vertical direction',
  IQ_OUT_FIFO_DEPTH_EQ: 'IQ output FIFO depth for ejection queue',
  RB_OUT_FIFO_DEPTH: 'Ring buffer output FIFO depth',
  RB_IN_FIFO_DEPTH: 'Ring buffer input FIFO depth',
  EQ_IN_FIFO_DEPTH: 'Ejection queue input FIFO depth',
  IP_L2H_FIFO_DEPTH: 'IP low-to-high frequency FIFO depth',
  IP_H2L_H_FIFO_DEPTH: 'IP high-to-low frequency high side FIFO depth',
  IP_H2L_L_FIFO_DEPTH: 'IP high-to-low frequency low side FIFO depth',
  // ETag Config
  TL_Etag_T2_UE_MAX: 'Max T2 upgrade entries for TL direction',
  TL_Etag_T1_UE_MAX: 'Max T1 upgrade entries for TL direction',
  TR_Etag_T2_UE_MAX: 'Max T2 upgrade entries for TR direction',
  TU_Etag_T2_UE_MAX: 'Max T2 upgrade entries for TU direction',
  TU_Etag_T1_UE_MAX: 'Max T1 upgrade entries for TU direction',
  TD_Etag_T2_UE_MAX: 'Max T2 upgrade entries for TD direction',
  ETAG_T1_ENABLED: 'Enable 3-level ETag mode (T2→T1→T0), otherwise 2-level (T2→T0)',
  ETAG_BOTHSIDE_UPGRADE: 'Enable ETag upgrade on both sides',
  // ITag Config
  ITag_TRIGGER_Th_H: 'ITag trigger threshold for horizontal direction',
  ITag_TRIGGER_Th_V: 'ITag trigger threshold for vertical direction',
  ITag_MAX_Num_H: 'Max ITag number for horizontal direction',
  ITag_MAX_Num_V: 'Max ITag number for vertical direction',
  // Bandwidth Limit
  GDMA_BW_LIMIT: 'GDMA bandwidth limit in GB/s',
  SDMA_BW_LIMIT: 'SDMA bandwidth limit in GB/s',
  CDMA_BW_LIMIT: 'CDMA bandwidth limit in GB/s',
  DDR_BW_LIMIT: 'DDR bandwidth limit in GB/s',
  L2M_BW_LIMIT: 'L2M bandwidth limit in GB/s',
  // Feature Switches
  ENABLE_CROSSPOINT_CONFLICT_CHECK: 'Enable crosspoint conflict checking',
  ORDERING_PRESERVATION_MODE: '0=Disabled, 1=Single side (TL/TU), 2=Both sides (whitelist), 3=Dynamic (src-dest based)',
  ORDERING_ETAG_UPGRADE_MODE: '0=Upgrade ETag only on resource failure, 1=Also upgrade on ordering failure',
  ORDERING_GRANULARITY: '0=IP level ordering, 1=Node level ordering',
  REVERSE_DIRECTION_ENABLED: 'Enable reverse direction flow control when normal direction is congested',
  REVERSE_DIRECTION_THRESHOLD: 'Threshold ratio (0.0-1.0) for triggering reverse direction flow',
  // Allowed Source Nodes (双侧下环方向配置)
  TL_ALLOWED_SOURCE_NODES: 'Node IDs allowed to eject to TL (left) direction',
  TR_ALLOWED_SOURCE_NODES: 'Node IDs allowed to eject to TR (right) direction',
  TU_ALLOWED_SOURCE_NODES: 'Node IDs allowed to eject to TU (up) direction',
  TD_ALLOWED_SOURCE_NODES: 'Node IDs allowed to eject to TD (down) direction',
  // Arbitration
  ARBITRATION_TYPE: 'Arbitration algorithm type for queue selection',
  ARBITRATION_ITERATIONS: 'Number of iterations for iSLIP algorithm',
  ARBITRATION_WEIGHT_STRATEGY: 'Weight calculation strategy: queue_length, fixed, priority',
  // D2D Config (DCIN模式)
  NUM_DIES: 'Number of dies in the system',
  D2D_ENABLED: 'Enable D2D (Die-to-Die) communication',
  D2D_AR_LATENCY: 'D2D address read channel latency (ns)',
  D2D_R_LATENCY: 'D2D read data channel latency (ns)',
  D2D_AW_LATENCY: 'D2D address write channel latency (ns)',
  D2D_W_LATENCY: 'D2D write data channel latency (ns)',
  D2D_B_LATENCY: 'D2D write response channel latency (ns)',
  D2D_RN_BW_LIMIT: 'D2D RN bandwidth limit (GB/s)',
  D2D_SN_BW_LIMIT: 'D2D SN bandwidth limit (GB/s)',
  D2D_AXI_BANDWIDTH: 'D2D AXI channel bandwidth limit (GB/s)',
  D2D_RN_RDB_SIZE: 'D2D RN read data buffer size',
  D2D_RN_WDB_SIZE: 'D2D RN write data buffer size',
  D2D_SN_RDB_SIZE: 'D2D SN read data buffer size',
  D2D_SN_WDB_SIZE: 'D2D SN write data buffer size',
  D2D_CONNECTIONS: 'D2D connections: [src_die, src_node, dst_die, dst_node]',
}

// ============== 工具函数 ==============

// 计算参数值列表
export function calculateSweepValues(start: number, end: number, step: number): number[] {
  if (step <= 0 || start > end) return [start]
  const values: number[] = []
  for (let v = start; v <= end + step * 0.001; v += step) {
    values.push(Math.round(v * 1000) / 1000)
  }
  return values
}

// 生成笛卡尔积组合
export function generateCombinations(sweepParams: SweepParam[]): Record<string, number>[] {
  if (sweepParams.length === 0) return []
  const combinations: Record<string, number>[] = []

  function generate(index: number, current: Record<string, number>) {
    if (index >= sweepParams.length) {
      combinations.push({ ...current })
      return
    }
    const param = sweepParams[index]
    for (const value of param.values) {
      generate(index + 1, { ...current, [param.key]: value })
    }
  }
  generate(0, {})
  return combinations
}

// sleep辅助函数
export const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

// 获取当前步骤
export const getCurrentStep = (currentTask: TaskStatus | null, selectedFiles: string[]): number => {
  if (!currentTask) {
    return selectedFiles.length > 0 ? 1 : 0
  }
  if (currentTask.status === 'running') return 2
  if (['completed', 'failed', 'cancelled'].includes(currentTask.status)) return 3
  return 1
}

// 获取步骤状态
export const getStepStatus = (stepIndex: number, currentStep: number, taskStatus?: string): 'wait' | 'process' | 'finish' | 'error' => {
  if (stepIndex < currentStep) return 'finish'
  if (stepIndex === currentStep) {
    if (stepIndex === 3 && taskStatus === 'failed') return 'error'
    if (stepIndex === 2) return 'process'
    return 'process'
  }
  return 'wait'
}

// 计算运行时间
export const getElapsedTime = (startTime: number | null): string => {
  if (!startTime) return '0秒'
  const elapsed = Math.floor((Date.now() - startTime) / 1000)
  if (elapsed < 60) return `${elapsed}秒`
  const mins = Math.floor(elapsed / 60)
  const secs = elapsed % 60
  return `${mins}分${secs}秒`
}

// 从描述中提取 batch ID
function extractBatchId(description: string | null): string | null {
  if (!description) return null
  const match = description.match(/\[batch:(\d+)\]/)
  return match ? match[1] : null
}

// 按批次分组历史任务
export function groupTaskHistory(taskHistory: TaskHistoryItem[]): GroupedTask[] {
  const groups: Record<string, GroupedTask> = {}

  taskHistory.forEach(task => {
    // 检查是否为参数遍历（通过 description 中的 batch ID 判断）
    const batchId = extractBatchId(task.experiment_description)
    const isSweep = batchId !== null

    let groupKey: string
    if (isSweep) {
      // 参数遍历：用 batch ID 分组
      groupKey = `sweep_${batchId}_${task.experiment_name || '未命名'}`
    } else {
      // 普通任务：每个任务单独一组
      groupKey = `single_${task.task_id}`
    }

    if (!groups[groupKey]) {
      groups[groupKey] = {
        key: groupKey,
        experiment_name: task.experiment_name || '未命名实验',
        mode: task.mode,
        topology: task.topology,
        created_at: task.created_at,
        status: 'completed',
        tasks: [],
        completed_count: 0,
        failed_count: 0,
        total_count: 0,
        experiment_id: task.results?.experiment_id,
        errors: [],
        is_sweep: isSweep,
      }
    }

    groups[groupKey].tasks.push(task)
    groups[groupKey].total_count++

    if (task.status === 'completed') {
      groups[groupKey].completed_count++
    } else if (task.status === 'failed') {
      groups[groupKey].failed_count++
      groups[groupKey].status = 'failed'
      // 收集错误信息
      if (task.error) {
        groups[groupKey].errors.push(task.error)
      }
    } else if (['running', 'pending'].includes(task.status)) {
      groups[groupKey].status = 'running'
    }

    if (!groups[groupKey].experiment_id && task.results?.experiment_id) {
      groups[groupKey].experiment_id = task.results.experiment_id
    }
  })

  return Object.values(groups).sort((a, b) =>
    new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  )
}

// ============== 参数绑定相关函数 ==============

// 获取下一个可用的绑定组ID
export function getNextBindGroupId(existingGroups: string[]): string {
  const allIds = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
  for (const id of allIds) {
    if (!existingGroups.includes(id)) return id
  }
  return `G${existingGroups.length + 1}`
}

// 验证绑定配置
export function validateBindings(sweepParams: SweepParam[]): string[] {
  const errors: string[] = []
  const bindGroups: Map<string, SweepParam[]> = new Map()

  for (const param of sweepParams) {
    if (param.bindGroupId) {
      const group = bindGroups.get(param.bindGroupId) || []
      group.push(param)
      bindGroups.set(param.bindGroupId, group)
    }
  }

  for (const [groupId, params] of bindGroups) {
    if (params.length < 2) {
      errors.push(`绑定组 ${groupId} 至少需要2个参数`)
      continue
    }
    const counts = params.map(p => p.values.length)
    if (!counts.every(c => c === counts[0])) {
      const details = params.map(p => `${p.key}(${p.values.length})`).join(', ')
      errors.push(`绑定组 ${groupId} 参数值数量不一致: ${details}`)
    }
  }
  return errors
}

// 计算带绑定的总组合数
export function calculateTotalCombinationsWithBinding(sweepParams: SweepParam[]): number {
  if (sweepParams.length === 0) return 0
  const counted = new Set<string>()
  let total = 1

  for (const param of sweepParams) {
    if (param.bindGroupId) {
      if (!counted.has(param.bindGroupId)) {
        counted.add(param.bindGroupId)
        total *= param.values.length
      }
    } else {
      total *= param.values.length
    }
  }
  return total
}

// 生成带绑定的参数组合
export function generateCombinationsWithBinding(sweepParams: SweepParam[]): Record<string, number>[] {
  if (sweepParams.length === 0) return []

  // 分组：绑定组 vs 独立参数
  const bindGroups: Map<string, SweepParam[]> = new Map()
  const independentParams: SweepParam[] = []

  for (const param of sweepParams) {
    if (param.bindGroupId) {
      const group = bindGroups.get(param.bindGroupId) || []
      group.push(param)
      bindGroups.set(param.bindGroupId, group)
    } else {
      independentParams.push(param)
    }
  }

  // 构建所有"单元"的值列表
  const allUnits: Record<string, number>[][] = []

  // 绑定组：zip成单元
  for (const params of bindGroups.values()) {
    const unitValues: Record<string, number>[] = []
    const len = params[0].values.length
    for (let i = 0; i < len; i++) {
      const combo: Record<string, number> = {}
      for (const p of params) {
        combo[p.key] = p.values[i]
      }
      unitValues.push(combo)
    }
    allUnits.push(unitValues)
  }

  // 独立参数：每个参数是一个单元
  for (const param of independentParams) {
    allUnits.push(param.values.map(v => ({ [param.key]: v })))
  }

  // 对所有单元做笛卡尔积
  const combinations: Record<string, number>[] = []
  function cartesian(idx: number, current: Record<string, number>) {
    if (idx >= allUnits.length) {
      combinations.push({ ...current })
      return
    }
    for (const val of allUnits[idx]) {
      cartesian(idx + 1, { ...current, ...val })
    }
  }
  cartesian(0, {})
  return combinations
}
