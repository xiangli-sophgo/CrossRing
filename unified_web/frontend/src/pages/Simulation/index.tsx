/**
 * 仿真执行页面
 */
import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Typography,
  Card,
  Form,
  Select,
  Input,
  InputNumber,
  Button,
  Space,
  Switch,
  Radio,
  Tag,
  message,
  Row,
  Col,
  Collapse,
  Modal,
  AutoComplete,
  Spin,
} from 'antd'
import {
  PlayCircleOutlined,
  ReloadOutlined,
  SettingOutlined,
  FileTextOutlined,
  SaveOutlined,
  RocketOutlined,
} from '@ant-design/icons'
import { primaryColor } from '@/theme/colors'
import {
  runSimulation,
  getTaskStatus,
  cancelTask,
  deleteTask,
  getGroupedHistory,
  getRunningTasks,
  getConfigs,
  getTrafficFilesTree,
  getTrafficFileContent,
  getConfigContent,
  saveConfigContent,
  clearTaskHistory,
  type SimulationRequest,
  type TaskStatus,
  type ConfigOption,
  type TrafficTreeNode,
  type TrafficFileContentResponse,
  type GroupedTaskItem,
} from '@/api/simulation'
import { getExperiments } from '@/api/experiments'
import type { Experiment } from '@/types'
import { useSimulationStore } from '@/stores/simulationStore'
import { useExperimentStore } from '@/stores/experimentStore'

// 导入拆分的组件
import {
  KCINConfigPanel,
  DCINConfigPanel,
  SweepConfigPanel,
  TaskStatusCard,
  TaskHistoryTable,
  TrafficFileTree,
} from './components'

// 导入类型和工具函数
import type { SweepParam, SavedSweepConfig } from './helpers'
import {
  calculateSweepValues,
  generateCombinationsWithBinding,
  validateBindings,
  calculateTotalCombinationsWithBinding,
  sleep,
} from './helpers'
import { useGlobalTaskWebSocket, type GlobalTaskUpdate } from '@/hooks/useGlobalTaskWebSocket'

const { Text } = Typography
const { Option } = Select

const Simulation: React.FC = () => {
  const navigate = useNavigate()
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  // 使用全局store缓存配置和流量文件
  const {
    configs,
    configsLoaded,
    setConfigs,
    trafficTree,
    trafficTreeMode,
    trafficTreeLoaded,
    setTrafficTree,
    configValues,
    setConfigValues,
    dieConfigValues,
    setDieConfigValues,
    originalConfigValues,
    setOriginalConfigValues,
    originalDieConfigValues,
    setOriginalDieConfigValues,
    selectedFiles,
    setSelectedFiles,
    formValues,
    setFormValues,
  } = useSimulationStore()
  const [loadingFiles, setLoadingFiles] = useState(false)
  const [expandedKeys, setExpandedKeys] = useState<string[]>([])

  // 运行中的任务列表（支持多任务）
  const [runningTasks, setRunningTasks] = useState<Map<string, { task: TaskStatus; startTime: number }>>(new Map())
  const [groupedTaskHistory, setGroupedTaskHistory] = useState<GroupedTaskItem[]>([])
  const [loadingHistory, setLoadingHistory] = useState(false)

  // 配置编辑加载状态
  const [loadingConfig, setLoadingConfig] = useState(false)
  // DCIN模式下的DIE拓扑配置加载状态
  const [loadingDieConfig, setLoadingDieConfig] = useState(false)
  // 另存为对话框状态
  const [saveAsModalVisible, setSaveAsModalVisible] = useState(false)
  const [saveAsType, setSaveAsType] = useState<'main' | 'die'>('main')
  const [saveAsName, setSaveAsName] = useState('')

  // 流量文件预览状态
  const [previewModalVisible, setPreviewModalVisible] = useState(false)
  const [previewContent, setPreviewContent] = useState<TrafficFileContentResponse | null>(null)
  const [loadingPreview, setLoadingPreview] = useState(false)

  // 已有实验列表（用于实验名称自动完成）
  const [existingExperiments, setExistingExperiments] = useState<Experiment[]>([])
  // 选中已有实验时的描述（用于追加模式）
  const [selectedExperimentDesc, setSelectedExperimentDesc] = useState<string | null>(null)

  // 参数遍历配置（用于UI配置，生成combinations后作为请求参数传递）
  const [sweepParams, setSweepParams] = useState<SweepParam[]>([])
  const [savedSweepConfigs, setSavedSweepConfigs] = useState<SavedSweepConfig[]>([])
  const [sweepConfigName, setSweepConfigName] = useState('')

  const pollIntervalsRef = useRef<Map<string, ReturnType<typeof setInterval>>>(new Map())
  const runningTasksRef = useRef<Map<string, { task: TaskStatus; startTime: number }>>(new Map())

  // 同步表单值到 store
  const syncFormToStore = useCallback(() => {
    const values = form.getFieldsValue()
    setFormValues({
      mode: values.mode,
      rows: values.rows,
      cols: values.cols,
      config_path: values.config_path,
      die_config_path: values.die_config_path,
      max_time: values.max_time,
      max_workers: values.max_workers,
      save_to_db: values.save_to_db,
      experiment_name: values.experiment_name,
      experiment_description: values.experiment_description,
    })
  }, [form, setFormValues])

  // 恢复运行中的任务
  const restoreRunningTask = async () => {
    try {
      const data = await getRunningTasks()
      if (data.tasks.length === 0) return

      const newRunningTasks = new Map<string, { task: TaskStatus; startTime: number }>()
      const statusPromises = data.tasks.map(async (taskItem) => {
        try {
          const fullStatus = await getTaskStatus(taskItem.task_id)
          return { taskItem, fullStatus, success: true }
        } catch {
          return { taskItem, fullStatus: null, success: false }
        }
      })
      const results = await Promise.all(statusPromises)
      for (const { taskItem, fullStatus, success } of results) {
        if (success && fullStatus && (fullStatus.status === 'running' || fullStatus.status === 'pending')) {
          const startTime = taskItem.started_at
            ? new Date(taskItem.started_at).getTime()
            : new Date(taskItem.created_at).getTime()
          newRunningTasks.set(taskItem.task_id, { task: fullStatus, startTime })
        }
      }
      setRunningTasks(newRunningTasks)
      if (newRunningTasks.size > 0) {
        message.info(`已恢复 ${newRunningTasks.size} 个运行中的任务`)
      }
    } catch (error) {
      console.error('恢复运行中任务失败:', error)
    }
  }

  // 同步 runningTasks 到 ref，以便在 WebSocket 回调中访问最新值
  useEffect(() => {
    runningTasksRef.current = runningTasks
  }, [runningTasks])

  // 全局 WebSocket 处理：接收所有任务状态变化
  const handleGlobalTaskUpdate = useCallback((update: GlobalTaskUpdate) => {
    console.log('[DEBUG] handleGlobalTaskUpdate:', update.task_id, 'progress:', update.progress, 'has in ref:', runningTasksRef.current.has(update.task_id))
    // 更新运行中任务卡片（如果存在）
    if (runningTasksRef.current.has(update.task_id)) {
      console.log('[DEBUG] Updating task in runningTasks')
      setRunningTasks(prev => {
        const newMap = new Map(prev)
        const existing = newMap.get(update.task_id)
        if (existing) {
          newMap.set(update.task_id, {
            ...existing,
            task: {
              ...existing.task,
              status: update.status as TaskStatus['status'],
              progress: update.progress,
              message: update.message,
              current_file: update.current_file || existing.task.current_file,
              error: update.error ?? existing.task.error,
              sim_details: update.sim_details ?? existing.task.sim_details,
            }
          })
        }
        return newMap
      })
    }

    // 任务完成时的处理
    if (['completed', 'failed', 'cancelled'].includes(update.status)) {
      // 停止该任务的轮询（避免重复通知）
      const interval = pollIntervalsRef.current.get(update.task_id)
      if (interval) {
        clearInterval(interval)
        pollIntervalsRef.current.delete(update.task_id)
      }

      // 刷新历史列表
      loadHistory()

      // 通知结果管理页面需要刷新
      if (update.status === 'completed') {
        useExperimentStore.getState().setNeedsRefresh(true)
      }

      // 显示完成消息
      const expName = update.experiment_name || update.task_id.slice(0, 8)
      if (update.status === 'completed') {
        message.success(`任务完成: ${expName}`)
      } else if (update.status === 'failed') {
        message.error(`任务失败: ${expName}`)
      }

      // 延迟移除任务卡片（失败任务需手动关闭）
      if (update.status !== 'failed') {
        setTimeout(() => {
          setRunningTasks(prev => {
            const newMap = new Map(prev)
            newMap.delete(update.task_id)
            return newMap
          })
        }, 3000)
      }
    }
  }, [])

  // 使用全局 WebSocket 订阅
  useGlobalTaskWebSocket({
    onTaskUpdate: handleGlobalTaskUpdate,
  })

  // 加载配置和历史
  useEffect(() => {
    const initPage = async () => {
      // 从 store 恢复表单值
      if (formValues.mode || formValues.config_path) {
        form.setFieldsValue(formValues)
      }

      // 并行执行所有初始化操作，每个操作独立处理错误
      const tasks: Promise<void>[] = []

      // 加载配置列表（只在未加载时）
      if (!configsLoaded) {
        tasks.push(loadConfigs())
      } else if (Object.keys(configValues).length === 0) {
        // 配置列表已有缓存，但没有配置值，加载默认配置
        const defaultConfig = configs.kcin.find((c: ConfigOption) => c.name === '5x4')
        if (defaultConfig && !form.getFieldValue('config_path')) {
          form.setFieldsValue({ config_path: defaultConfig.path })
          tasks.push(loadConfigContent(defaultConfig.path))
        }
      }
      // 如果 store 中已有 configValues，不重新加载

      // 加载流量文件（只在未加载或模式不匹配时）
      const currentMode = formValues.mode || 'kcin'
      if (!trafficTreeLoaded || trafficTreeMode !== currentMode) {
        tasks.push(loadTrafficFilesTree(currentMode))
      }

      // 加载历史和实验列表
      tasks.push(loadHistory())
      tasks.push(loadExistingExperiments())
      tasks.push(restoreRunningTask())

      // 等待所有任务完成（每个任务已独立处理错误）
      await Promise.allSettled(tasks)
    }

    initPage()

    return () => {
      // 清除所有任务轮询
      pollIntervalsRef.current.forEach(interval => clearInterval(interval))
      pollIntervalsRef.current.clear()
    }
  }, [])

  // 加载已有实验列表
  const loadExistingExperiments = async () => {
    try {
      const experiments = await getExperiments()
      setExistingExperiments(experiments)
    } catch (error) {
      console.error('加载实验列表失败:', error)
    }
  }

  const loadConfigs = async () => {
    try {
      const data = await getConfigs()
      setConfigs(data)
      // 优先使用上次选择的配置，否则使用默认的 5x4
      const lastConfigPath = formValues.config_path
      const lastConfig = lastConfigPath ? data.kcin.find((c: ConfigOption) => c.path === lastConfigPath) : null
      const defaultConfig = lastConfig || data.kcin.find((c: ConfigOption) => c.name === '5x4')
      if (defaultConfig) {
        setTimeout(() => {
          form.setFieldsValue({ config_path: defaultConfig.path })
        }, 0)
        loadConfigContent(defaultConfig.path)
      }
    } catch (error) {
      console.error('加载配置失败:', error)
    }
  }

  const loadHistory = async () => {
    setLoadingHistory(true)
    try {
      const data = await getGroupedHistory(50)
      setGroupedTaskHistory(data.groups)
    } catch (error) {
      console.error('加载历史失败:', error)
    } finally {
      setLoadingHistory(false)
    }
  }

  const loadTrafficFilesTree = async (filterMode?: 'kcin' | 'dcin', forceRefresh?: boolean) => {
    // 如果缓存有效且不强制刷新，直接返回
    if (!forceRefresh && trafficTreeLoaded && trafficTreeMode === filterMode) {
      return
    }
    setLoadingFiles(true)
    try {
      const data = await getTrafficFilesTree(filterMode)
      setTrafficTree(data.tree, filterMode || 'kcin')
    } catch (error) {
      console.error('加载流量文件失败:', error)
    } finally {
      setLoadingFiles(false)
    }
  }

  const loadConfigContent = async (configPath: string) => {
    if (!configPath) {
      setConfigValues({})
      setOriginalConfigValues({})
      return
    }
    setLoadingConfig(true)
    try {
      const content = await getConfigContent(configPath)
      setConfigValues(content)
      setOriginalConfigValues(content)  // 同时保存原始配置
    } catch (error) {
      console.error('加载配置内容失败:', error)
      message.error('加载配置内容失败')
      setConfigValues({})
      setOriginalConfigValues({})
    } finally {
      setLoadingConfig(false)
    }
  }

  const loadDieConfigContent = async (configPath: string) => {
    if (!configPath) {
      setDieConfigValues({})
      setOriginalDieConfigValues({})
      return
    }
    setLoadingDieConfig(true)
    try {
      const content = await getConfigContent(configPath)
      setDieConfigValues(content)
      setOriginalDieConfigValues(content)  // 同时保存原始配置
    } catch (error) {
      console.error('加载DIE配置内容失败:', error)
      message.error('加载DIE配置内容失败')
      setDieConfigValues({})
      setOriginalDieConfigValues({})
    } finally {
      setLoadingDieConfig(false)
    }
  }

  const updateConfigValue = useSimulationStore((state) => state.updateConfigValue)
  const updateDieConfigValue = useSimulationStore((state) => state.updateDieConfigValue)

  const handlePreviewFile = async (filePath: string) => {
    setLoadingPreview(true)
    setPreviewModalVisible(true)
    try {
      const data = await getTrafficFileContent(filePath, 200)
      setPreviewContent(data)
    } catch (error) {
      console.error('加载文件内容失败:', error)
      message.error('加载文件内容失败')
      setPreviewModalVisible(false)
    } finally {
      setLoadingPreview(false)
    }
  }

  const handleSubmit = async (values: any) => {
    if (selectedFiles.length === 0) {
      message.warning('请选择流量文件')
      return
    }
    if (values.mode === 'dcin' && !values.die_config_path) {
      message.warning('请选择KCIN配置文件')
      return
    }

    setLoading(true)
    try {
      const topology = `${values.rows}x${values.cols}`
      // 生成实验描述：优先使用用户输入，否则自动生成；如果是已有实验则追加
      const autoDesc = generateDefaultDescription()
      let finalDescription: string
      if (values.experiment_description) {
        finalDescription = values.experiment_description
      } else if (selectedExperimentDesc) {
        finalDescription = `${selectedExperimentDesc}\n${autoDesc}`
      } else {
        finalDescription = autoDesc
      }
      const request: SimulationRequest = {
        mode: values.mode,
        topology: topology,
        config_path: values.config_path,
        config_overrides: Object.keys(configValues).length > 0 ? configValues : undefined,
        die_config_path: values.mode === 'dcin' ? values.die_config_path : undefined,
        die_config_overrides: values.mode === 'dcin' && Object.keys(dieConfigValues).length > 0 ? dieConfigValues : undefined,
        traffic_source: 'file',
        traffic_files: selectedFiles,
        max_time: values.max_time,
        save_to_db: values.save_to_db,
        experiment_name: values.experiment_name,
        experiment_description: finalDescription,
        max_workers: values.max_workers,
      }
      const response = await runSimulation(request)
      message.success(`仿真任务已创建: ${response.task_id}`)
      // 添加到运行任务列表
      const initialStatus: TaskStatus = {
        task_id: response.task_id,
        status: 'pending',
        progress: 0,
        current_file: '',
        message: '任务已创建',
        error: null,
        results: null,
        sim_details: null,
        created_at: new Date().toISOString(),
        started_at: null,
        completed_at: null,
        experiment_name: values.experiment_name,
      }
      setRunningTasks(prev => {
        const newMap = new Map(prev)
        newMap.set(response.task_id, { task: initialStatus, startTime: Date.now() })
        return newMap
      })
      // 短延迟后获取真实状态，避免显示"等待中"
      setTimeout(async () => {
        try {
          const realStatus = await getTaskStatus(response.task_id)
          setRunningTasks(prev => {
            const newMap = new Map(prev)
            const existing = newMap.get(response.task_id)
            if (existing) {
              newMap.set(response.task_id, { ...existing, task: realStatus })
            }
            return newMap
          })
        } catch (e) {
          console.error('获取任务状态失败:', e)
        }
      }, 500)
    } catch (error: any) {
      message.error(error.response?.data?.detail || '启动仿真失败')
    } finally {
      setLoading(false)
    }
  }

  const startPolling = (taskId: string) => {
    // 如果已经在轮询这个任务，不重复创建
    if (pollIntervalsRef.current.has(taskId)) return

    let messageShown = false  // 防止重复显示消息
    let errorCount = 0  // 连续错误计数

    const poll = async () => {
      try {
        const status = await getTaskStatus(taskId)
        errorCount = 0  // 重置错误计数
        setRunningTasks(prev => {
          const newMap = new Map(prev)
          const existing = newMap.get(taskId)
          if (existing) {
            newMap.set(taskId, { ...existing, task: status })
          }
          return newMap
        })

        if (['completed', 'failed', 'cancelled'].includes(status.status)) {
          // 停止该任务的轮询
          const interval = pollIntervalsRef.current.get(taskId)
          if (interval) {
            clearInterval(interval)
            pollIntervalsRef.current.delete(taskId)
          }
          // 从运行任务列表中移除（失败任务需手动关闭，不自动消失）
          if (status.status !== 'failed') {
            setTimeout(() => {
              setRunningTasks(prev => {
                const newMap = new Map(prev)
                newMap.delete(taskId)
                return newMap
              })
            }, 3000)
          }
          loadHistory()
          if (!messageShown) {
            messageShown = true
            const expName = status.experiment_name || taskId.slice(0, 8)
            if (status.status === 'completed') message.success(`任务完成: ${expName}`)
            else if (status.status === 'failed') message.error(`任务失败: ${expName}，详情请查看任务卡片`)
          }
        }
      } catch (error: any) {
        // 404 表示任务已不存在，立即停止轮询
        if (error.response?.status === 404) {
          const interval = pollIntervalsRef.current.get(taskId)
          if (interval) {
            clearInterval(interval)
            pollIntervalsRef.current.delete(taskId)
          }
          setRunningTasks(prev => {
            const newMap = new Map(prev)
            newMap.delete(taskId)
            return newMap
          })
          loadHistory()
          return
        }

        errorCount++
        console.error(`获取任务状态失败 (${errorCount}/3):`, error)
        // 连续3次失败后停止轮询该任务
        if (errorCount >= 3) {
          const interval = pollIntervalsRef.current.get(taskId)
          if (interval) {
            clearInterval(interval)
            pollIntervalsRef.current.delete(taskId)
          }
          setRunningTasks(prev => {
            const newMap = new Map(prev)
            newMap.delete(taskId)
            return newMap
          })
          message.warning(`任务 ${taskId.slice(0, 8)} 状态获取失败，已停止轮询`)
        }
      }
    }

    poll()
    const interval = setInterval(poll, 1000)
    pollIntervalsRef.current.set(taskId, interval)
  }

  const handleCancel = async (taskId: string) => {
    try {
      await cancelTask(taskId)
      message.info('任务已取消')
      // 停止轮询
      const interval = pollIntervalsRef.current.get(taskId)
      if (interval) {
        clearInterval(interval)
        pollIntervalsRef.current.delete(taskId)
      }
      // 从列表移除
      setRunningTasks(prev => {
        const newMap = new Map(prev)
        newMap.delete(taskId)
        return newMap
      })
      loadHistory()
    } catch (error) {
      message.error('取消任务失败')
    }
  }

  // 参数遍历相关函数
  const addSweepParam = (key: string) => {
    const currentValue = configValues[key] || 0
    const newParam: SweepParam = {
      key,
      start: currentValue,
      end: currentValue,
      step: 1,
      values: [currentValue]
    }
    setSweepParams([...sweepParams, newParam])
  }

  const updateSweepParam = (index: number, field: 'start' | 'end' | 'step', value: number | null) => {
    const newParams = [...sweepParams]
    const param = newParams[index]
    if (value !== null) {
      param[field] = value
      param.values = calculateSweepValues(param.start, param.end, param.step)
    }
    setSweepParams(newParams)
  }

  const removeSweepParam = (index: number) => {
    setSweepParams(sweepParams.filter((_, i) => i !== index))
  }

  const updateBindGroup = (index: number, groupId: string | undefined) => {
    const newParams = [...sweepParams]
    newParams[index] = { ...newParams[index], bindGroupId: groupId }
    setSweepParams(newParams)
  }

  // 生成默认实验描述
  const generateDefaultDescription = useCallback(() => {
    const now = new Date()
    const timestamp = `${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')} ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`

    // 获取配置文件名
    const configPath = form.getFieldValue('config_path') || ''
    const dieConfigPath = form.getFieldValue('die_config_path') || ''
    const mode = form.getFieldValue('mode') || 'kcin'

    // 提取配置文件名（去除路径和扩展名）
    const getConfigName = (path: string) => {
      if (!path) return ''
      const filename = path.split('/').pop() || path
      return filename.replace(/\.(yaml|json)$/, '')
    }

    const mainConfigName = getConfigName(configPath)
    const dieConfigName = getConfigName(dieConfigPath)

    // 构建配置文件信息
    let configInfo = ''
    if (mode === 'dcin') {
      configInfo = `配置: ${mainConfigName} (Die配置: ${dieConfigName})`
    } else {
      configInfo = `配置: ${mainConfigName}`
    }

    // 找出修改的参数
    const changedParams = Object.entries(configValues)
      .filter(([k, v]) => originalConfigValues[k] !== v)
      .map(([k, v]) => `${k}=${v}`)

    // 如果是参数遍历，显示遍历范围
    let paramStr: string
    if (sweepParams.length > 0) {
      paramStr = `参数遍历: ${sweepParams.map(p => `${p.key}[${p.start}→${p.end}]`).join(', ')}`
    } else if (changedParams.length > 0) {
      paramStr = `修改: ${changedParams.join(', ')}`
    } else {
      paramStr = '默认配置'
    }

    // 简化流量文件显示
    const fileNames = selectedFiles.map(f => f.split('/').pop() || f)
    const filesStr = fileNames.length <= 3
      ? fileNames.join(', ')
      : `${fileNames.slice(0, 2).join(', ')} 等${fileNames.length}个`

    return `[${timestamp}] ${configInfo} - ${paramStr} | ${filesStr}`
  }, [configValues, originalConfigValues, sweepParams, selectedFiles, form])

  const saveSweepConfig = (name: string) => {
    if (!name.trim()) {
      message.warning('请输入配置名称')
      return
    }
    if (sweepParams.length === 0) {
      message.warning('没有可保存的遍历参数')
      return
    }
    const configs = [...savedSweepConfigs]
    const existingIndex = configs.findIndex(c => c.name === name)
    if (existingIndex >= 0) {
      configs[existingIndex] = { name, params: sweepParams }
    } else {
      configs.push({ name, params: sweepParams })
    }
    setSavedSweepConfigs(configs)
    localStorage.setItem('sweepConfigs', JSON.stringify(configs))
    setSweepConfigName('')
    message.success(`遍历配置 "${name}" 已保存`)
  }

  const loadSweepConfig = (name: string) => {
    const config = savedSweepConfigs.find(c => c.name === name)
    if (config) {
      setSweepParams(config.params)
      message.success(`已加载配置 "${name}"`)
    }
  }

  const deleteSweepConfig = (name: string) => {
    const configs = savedSweepConfigs.filter(c => c.name !== name)
    setSavedSweepConfigs(configs)
    localStorage.setItem('sweepConfigs', JSON.stringify(configs))
    message.success(`配置 "${name}" 已删除`)
  }

  // 初始化时加载保存的遍历配置
  useEffect(() => {
    const saved = localStorage.getItem('sweepConfigs')
    if (saved) {
      try {
        setSavedSweepConfigs(JSON.parse(saved))
      } catch {
        // ignore
      }
    }
  }, [])

  // 计算存在的绑定组
  const existingBindGroups = useMemo(() => {
    const groups = new Set<string>()
    sweepParams.forEach(p => { if (p.bindGroupId) groups.add(p.bindGroupId) })
    return Array.from(groups).sort()
  }, [sweepParams])

  // 验证绑定配置
  const bindingErrors = useMemo(() => validateBindings(sweepParams), [sweepParams])

  // 计算总组合数（支持绑定）
  const totalCombinations = sweepParams.length > 0
    ? calculateTotalCombinationsWithBinding(sweepParams)
    : 0

  const availableParams = Object.keys(configValues).filter(key => {
    const value = configValues[key]
    return typeof value === 'number' && !sweepParams.find(p => p.key === key)
  })

  const handleSweepSubmit = async (values: any) => {
    if (selectedFiles.length === 0) {
      message.warning('请选择流量文件')
      return
    }

    // 验证绑定配置
    if (bindingErrors.length > 0) {
      message.error('绑定配置有误: ' + bindingErrors.join('; '))
      return
    }

    const combinations = generateCombinationsWithBinding(sweepParams)
    if (combinations.length === 0) {
      message.warning('请配置遍历参数')
      return
    }

    if (combinations.length > 100) {
      Modal.confirm({
        title: '组合数量较多',
        content: `将生成 ${combinations.length} 组参数组合，确定继续？`,
        onOk: () => {
          // 不返回 Promise，让 Modal 立即关闭
          executeSweep(values, combinations)
        }
      })
      return
    }

    await executeSweep(values, combinations)
  }

  const executeSweep = async (values: any, combinations: Record<string, number>[]) => {
    // 创建单个任务，包含所有参数组合（和并行执行逻辑统一）
    const topology = `${values.rows}x${values.cols}`
    const totalUnits = selectedFiles.length * combinations.length
    const experimentName = values.experiment_name || `参数遍历_${new Date().toLocaleString()}`

    // 生成实验描述：优先使用用户输入，否则自动生成；如果是已有实验则追加
    const autoDesc = generateDefaultDescription()
    let finalDescription: string
    if (values.experiment_description) {
      finalDescription = values.experiment_description
    } else if (selectedExperimentDesc) {
      finalDescription = `${selectedExperimentDesc}\n${autoDesc}`
    } else {
      finalDescription = autoDesc
    }

    const request: SimulationRequest = {
      mode: values.mode,
      topology,
      config_path: values.config_path,
      config_overrides: configValues,
      die_config_path: values.mode === 'dcin' ? values.die_config_path : undefined,
      die_config_overrides: values.mode === 'dcin' && Object.keys(dieConfigValues).length > 0 ? dieConfigValues : undefined,
      traffic_source: 'file',
      traffic_files: selectedFiles,
      max_time: values.max_time,
      save_to_db: values.save_to_db,
      experiment_name: experimentName,
      experiment_description: finalDescription,
      max_workers: values.max_workers,
      sweep_combinations: combinations,
    }

    try {
      const response = await runSimulation(request)

      // 创建初始状态（和 handleSubmit 相同）
      const initialStatus: TaskStatus = {
        task_id: response.task_id,
        status: 'pending',
        progress: 0,
        current_file: '',
        message: response.message,
        error: null,
        results: null,
        sim_details: null,
        created_at: new Date().toISOString(),
        started_at: null,
        completed_at: null,
        experiment_name: experimentName,
      }

      // 添加到 runningTasks
      setRunningTasks(prev => {
        const newMap = new Map(prev)
        newMap.set(response.task_id, { task: initialStatus, startTime: Date.now() })
        return newMap
      })

      // 短延迟后获取真实状态，避免显示"等待中"
      setTimeout(async () => {
        try {
          const realStatus = await getTaskStatus(response.task_id)
          setRunningTasks(prev => {
            const newMap = new Map(prev)
            const existing = newMap.get(response.task_id)
            if (existing) {
              newMap.set(response.task_id, { ...existing, task: realStatus })
            }
            return newMap
          })
        } catch (e) {
          console.error('获取任务状态失败:', e)
        }
      }, 500)

      message.success(`参数遍历任务已创建: ${totalUnits} 个执行单元`)
    } catch (error: any) {
      message.error(`任务创建失败: ${error.response?.data?.detail || error.message}`)
    }
  }

  const handleDeleteGroupedTask = async (record: GroupedTaskItem) => {
    try {
      for (const taskId of record.task_ids) {
        await deleteTask(taskId)
      }
      message.success(`已删除 ${record.task_ids.length} 个任务`)
      loadHistory()
    } catch (error: any) {
      message.error(error.response?.data?.detail || '删除失败')
    }
  }

  const handleClearAllHistory = async () => {
    try {
      await clearTaskHistory()
      message.success('已清除所有历史任务')
      loadHistory()
    } catch (error: any) {
      message.error(error.response?.data?.detail || '清除失败')
    }
  }

  const mode = Form.useWatch('mode', form) || 'kcin'

  return (
    <div>
      <Row gutter={24} align="stretch">
        {/* 左侧：配置表单 */}
        <Col xs={24} lg={12}>
          <Card
            title={
              <Space>
                <SettingOutlined style={{ color: primaryColor }} />
                <span>仿真配置</span>
              </Space>
            }
            style={{ marginBottom: 24 }}
          >
            <Form
              form={form}
              layout="vertical"
              onValuesChange={syncFormToStore}
              initialValues={{
                mode: 'kcin',
                rows: 5,
                cols: 4,
                max_time: 6000,
                max_workers: 8,
                save_to_db: true,
              }}
              onFinish={handleSubmit}
            >
              <Row gutter={24}>
                <Col span={12}>
                  <Form.Item name="mode" label="仿真模式" rules={[{ required: true }]}>
                    <Radio.Group onChange={(e) => {
                      const newMode = e.target.value as 'kcin' | 'dcin'
                      if (newMode === 'kcin') {
                        const defaultConfig = configs.kcin.find(c => c.name === '5x4')
                        if (defaultConfig) {
                          form.setFieldsValue({ config_path: defaultConfig.path, rows: 5, cols: 4, die_config_path: undefined })
                          loadConfigContent(defaultConfig.path)
                          setDieConfigValues({})
                        }
                      } else {
                        const defaultDcinConfig = configs.dcin.find(c => c.path.includes('4die'))
                        const defaultKcinConfig = configs.kcin.find(c => c.name === '5x4')
                        if (defaultDcinConfig) {
                          form.setFieldsValue({ config_path: defaultDcinConfig.path, rows: 5, cols: 4 })
                          loadConfigContent(defaultDcinConfig.path)
                        }
                        if (defaultKcinConfig) {
                          form.setFieldsValue({ die_config_path: defaultKcinConfig.path })
                          loadDieConfigContent(defaultKcinConfig.path)
                        }
                      }
                      setSelectedFiles([])
                      loadTrafficFilesTree(newMode)
                    }}>
                      <Radio.Button value="kcin">KCIN</Radio.Button>
                      <Radio.Button value="dcin">DCIN</Radio.Button>
                    </Radio.Group>
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="拓扑配置" required>
                    <Space>
                      <Form.Item name="rows" noStyle rules={[{ required: true, message: '请输入行数' }]}>
                        <InputNumber min={1} max={20} placeholder="行" style={{ width: 70 }} />
                      </Form.Item>
                      <Text type="secondary">×</Text>
                      <Form.Item name="cols" noStyle rules={[{ required: true, message: '请输入列数' }]}>
                        <InputNumber min={1} max={20} placeholder="列" style={{ width: 70 }} />
                      </Form.Item>
                    </Space>
                  </Form.Item>
                </Col>
              </Row>

              <Form.Item
                name="config_path"
                label={mode === 'dcin' ? 'DCIN配置文件' : 'KCIN配置文件'}
                rules={[{ required: true, message: mode === 'dcin' ? '请选择DCIN配置文件' : '请选择KCIN配置文件' }]}
              >
                <Select
                  placeholder={mode === 'dcin' ? '请选择DCIN配置文件' : '请选择KCIN配置文件'}
                  onChange={loadConfigContent}
                  popupMatchSelectWidth={false}
                >
                  {(mode === 'kcin' ? configs.kcin : configs.dcin).map((c) => (
                    <Option key={c.path} value={c.path} title={c.name}>{c.name}</Option>
                  ))}
                </Select>
              </Form.Item>

              {/* 配置编辑面板 */}
              {Object.keys(configValues).length > 0 && (
                <Spin spinning={loadingConfig}>
                  {mode === 'dcin' ? (
                    <DCINConfigPanel
                      configValues={configValues}
                      updateConfigValue={updateConfigValue}
                    />
                  ) : (
                    <KCINConfigPanel
                      configValues={configValues}
                      updateConfigValue={updateConfigValue}
                    />
                  )}
                  <Space style={{ marginBottom: 16 }}>
                    <Button
                      size="small"
                      icon={<SaveOutlined />}
                      onClick={async () => {
                        const configPath = form.getFieldValue('config_path')
                        if (!configPath) {
                          message.warning('请先选择配置文件')
                          return
                        }
                        try {
                          await saveConfigContent(configPath, configValues)
                          message.success('配置已保存')
                        } catch (e: any) {
                          message.error(`保存失败: ${e.response?.data?.detail || e.message || '未知错误'}`)
                        }
                      }}
                    >
                      保存
                    </Button>
                    <Button
                      size="small"
                      onClick={() => {
                        setSaveAsType('main')
                        setSaveAsName(mode === 'dcin' ? 'dcin_' : 'kcin_')
                        setSaveAsModalVisible(true)
                      }}
                    >
                      另存为
                    </Button>
                    <Button
                      size="small"
                      icon={<ReloadOutlined />}
                      onClick={() => loadConfigContent(form.getFieldValue('config_path'))}
                    >
                      重置
                    </Button>
                  </Space>
                </Spin>
              )}

              {/* DCIN模式下的KCIN配置选择 */}
              {mode === 'dcin' && (
                <>
                  <Form.Item
                    name="die_config_path"
                    label="KCIN配置文件 (单Die拓扑)"
                    rules={[{ required: true, message: '请选择KCIN配置文件' }]}
                  >
                    <Select placeholder="请选择KCIN配置文件" onChange={loadDieConfigContent} popupMatchSelectWidth={false}>
                      {configs.kcin.map((c) => (
                        <Option key={c.path} value={c.path} title={c.name}>{c.name}</Option>
                      ))}
                    </Select>
                  </Form.Item>

                  {Object.keys(dieConfigValues).length > 0 && (
                    <Spin spinning={loadingDieConfig}>
                      <KCINConfigPanel
                        configValues={dieConfigValues}
                        updateConfigValue={updateDieConfigValue}
                      />
                      <Space style={{ marginBottom: 16 }}>
                        <Button
                          size="small"
                          icon={<SaveOutlined />}
                          onClick={async () => {
                            const configPath = form.getFieldValue('die_config_path')
                            if (!configPath) {
                              message.warning('请先选择配置文件')
                              return
                            }
                            try {
                              await saveConfigContent(configPath, dieConfigValues)
                              message.success('配置已保存')
                            } catch (e: any) {
                              message.error(`保存失败: ${e.response?.data?.detail || e.message || '未知错误'}`)
                            }
                          }}
                        >
                          保存
                        </Button>
                        <Button
                          size="small"
                          onClick={() => {
                            setSaveAsType('die')
                            setSaveAsName('kcin_')
                            setSaveAsModalVisible(true)
                          }}
                        >
                          另存为
                        </Button>
                        <Button
                          size="small"
                          icon={<ReloadOutlined />}
                          onClick={() => loadDieConfigContent(form.getFieldValue('die_config_path'))}
                        >
                          重置
                        </Button>
                      </Space>
                    </Spin>
                  )}
                </>
              )}

              {/* 参数遍历配置 */}
              <SweepConfigPanel
                sweepParams={sweepParams}
                configValues={configValues}
                selectedFilesCount={selectedFiles.length}
                availableParams={availableParams}
                totalCombinations={totalCombinations}
                savedSweepConfigs={savedSweepConfigs}
                sweepConfigName={sweepConfigName}
                existingBindGroups={existingBindGroups}
                bindingErrors={bindingErrors}
                onAddSweepParam={addSweepParam}
                onUpdateSweepParam={updateSweepParam}
                onRemoveSweepParam={removeSweepParam}
                onUpdateBindGroup={updateBindGroup}
                onSaveSweepConfig={saveSweepConfig}
                onLoadSweepConfig={loadSweepConfig}
                onDeleteSweepConfig={deleteSweepConfig}
                onSweepConfigNameChange={setSweepConfigName}
              />

              {/* 仿真设置 */}
              <Collapse
                size="small"
                style={{ marginBottom: 16 }}
                defaultActiveKey={['sim_settings']}
                items={[{
                  key: 'sim_settings',
                  label: '仿真设置',
                  children: (
                    <>
                      <Row gutter={[16, 8]}>
                        <Col span={8}>
                          <Form.Item name="max_time" label="最大仿真时间 (ns)">
                            <InputNumber min={100} max={100000} style={{ width: '100%' }} />
                          </Form.Item>
                        </Col>
                        <Col span={8}>
                          <Form.Item name="max_workers" label="并行数">
                            <InputNumber min={1} max={8} style={{ width: '100%' }} />
                          </Form.Item>
                        </Col>
                        <Col span={8}>
                          <Form.Item name="save_to_db" label="保存到数据库" valuePropName="checked">
                            <Switch />
                          </Form.Item>
                        </Col>
                      </Row>
                      <Form.Item name="experiment_name" label="实验名称" style={{ marginBottom: 8 }}>
                        <AutoComplete
                          placeholder="输入或选择已有实验名称"
                          options={existingExperiments.map(e => ({ value: e.name }))}
                          filterOption={(inputValue, option) =>
                            option?.value.toLowerCase().includes(inputValue.toLowerCase()) || false
                          }
                          onSelect={(value: string) => {
                            // 选择已有实验时，获取其描述用于追加
                            const exp = existingExperiments.find(e => e.name === value)
                            setSelectedExperimentDesc(exp?.description || null)
                          }}
                          onChange={(value: string) => {
                            // 用户手动输入时，检查是否匹配已有实验
                            const exp = existingExperiments.find(e => e.name === value)
                            setSelectedExperimentDesc(exp?.description || null)
                          }}
                        />
                      </Form.Item>
                      <Form.Item name="experiment_description" label="实验描述" style={{ marginBottom: 0 }}>
                        <Input.TextArea rows={2} placeholder="可选的实验描述" />
                      </Form.Item>
                    </>
                  ),
                }]}
              />

              {/* 提交按钮 */}
              <Form.Item style={{ marginBottom: 0 }}>
                <Space>
                  {sweepParams.length > 0 ? (
                    <Button
                      type="primary"
                      icon={<RocketOutlined />}
                      onClick={() => handleSweepSubmit(form.getFieldsValue())}
                      loading={loading}
                      disabled={sweepParams.length === 0 || bindingErrors.length > 0}
                      size="large"
                      style={{ height: 44 }}
                    >
                      {loading ? '批量执行中...' : `批量执行 (${totalCombinations}组)`}
                    </Button>
                  ) : (
                    <Button
                      type="primary"
                      htmlType="submit"
                      icon={<PlayCircleOutlined />}
                      loading={loading}
                      size="large"
                      style={{ height: 44 }}
                    >
                      开始仿真
                    </Button>
                  )}
                </Space>
              </Form.Item>
            </Form>
          </Card>
        </Col>

        {/* 右侧：文件选择和任务执行 */}
        <Col xs={24} lg={12}>
          <TrafficFileTree
            trafficTree={trafficTree}
            selectedFiles={selectedFiles}
            expandedKeys={expandedKeys}
            loading={loadingFiles}
            onSelect={setSelectedFiles}
            onExpandedKeysChange={setExpandedKeys}
            onRefresh={() => loadTrafficFilesTree(mode as 'kcin' | 'dcin', true)}
            onPreviewFile={handlePreviewFile}
          />

          {/* 运行中的任务卡片列表 */}
          {Array.from(runningTasks.entries()).map(([taskId, { task, startTime }]) => (
            <TaskStatusCard
              key={taskId}
              currentTask={task}
              startTime={startTime}
              onCancel={task.status === 'running' ? () => handleCancel(taskId) : undefined}
              onClose={task.status === 'failed' ? () => {
                setRunningTasks(prev => {
                  const newMap = new Map(prev)
                  newMap.delete(taskId)
                  return newMap
                })
              } : undefined}
            />
          ))}
        </Col>
      </Row>

      {/* 历史任务 */}
      <TaskHistoryTable
        groupedTaskHistory={groupedTaskHistory}
        loading={loadingHistory}
        onRefresh={loadHistory}
        onViewResult={(experimentId) => navigate(`/experiments/${experimentId}`)}
        onDelete={handleDeleteGroupedTask}
        onClearAll={handleClearAllHistory}
      />

      {/* 另存为模态框 */}
      <Modal
        title={saveAsType === 'main' ? (mode === 'dcin' ? '另存DCIN配置' : '另存KCIN配置') : '另存KCIN配置'}
        open={saveAsModalVisible}
        onCancel={() => setSaveAsModalVisible(false)}
        onOk={async () => {
          if (!saveAsName.trim()) {
            message.warning('请输入文件名')
            return
          }
          if (!/^[a-zA-Z0-9_-]+$/.test(saveAsName)) {
            message.warning('文件名只能包含字母、数字、下划线和连字符')
            return
          }
          try {
            const configPath = saveAsType === 'main'
              ? form.getFieldValue('config_path')
              : form.getFieldValue('die_config_path')
            const content = saveAsType === 'main' ? configValues : dieConfigValues
            const result = await saveConfigContent(configPath || 'default.yaml', content, saveAsName)
            message.success(`配置已另存为: ${result.filename}`)
            setSaveAsModalVisible(false)
            loadConfigs()
          } catch (e: any) {
            message.error(`保存失败: ${e.response?.data?.detail || e.message || '未知错误'}`)
          }
        }}
        okText="保存"
        cancelText="取消"
      >
        <div style={{ marginBottom: 16 }}>
          <div style={{ marginBottom: 8 }}>新文件名 (不含扩展名)</div>
          <Input
            placeholder={mode === 'dcin' && saveAsType === 'main' ? 'dcin_my_config' : 'kcin_my_config'}
            value={saveAsName}
            onChange={(e) => setSaveAsName(e.target.value)}
            suffix=".yaml"
          />
          <div style={{ marginTop: 8, color: '#888', fontSize: 12 }}>
            提示: 文件名只能包含字母、数字、下划线和连字符
          </div>
        </div>
      </Modal>

      {/* 流量文件预览模态框 */}
      <Modal
        title={
          <Space>
            <FileTextOutlined />
            <span>流量文件预览</span>
            {previewContent && <Tag color="blue">{previewContent.file_name}</Tag>}
          </Space>
        }
        open={previewModalVisible}
        onCancel={() => {
          setPreviewModalVisible(false)
          setPreviewContent(null)
        }}
        footer={null}
        width={800}
      >
        <Spin spinning={loadingPreview}>
          {previewContent && (
            <>
              <div style={{ marginBottom: 12 }}>
                <Space>
                  <Text type="secondary">文件大小: {(previewContent.file_size / 1024).toFixed(1)} KB</Text>
                  <Text type="secondary">总行数: {previewContent.total_lines}</Text>
                  {previewContent.truncated && <Tag color="warning">仅显示前 200 行</Tag>}
                </Space>
              </div>
              <div
                style={{
                  backgroundColor: '#f5f5f5',
                  padding: 12,
                  borderRadius: 4,
                  maxHeight: 500,
                  overflow: 'auto',
                  fontFamily: 'monospace',
                  fontSize: 12,
                  lineHeight: 1.6,
                }}
              >
                {previewContent.content.map((line, index) => (
                  <div key={index} style={{ display: 'flex' }}>
                    <span style={{ color: '#999', width: 50, textAlign: 'right', marginRight: 12, userSelect: 'none' }}>
                      {index + 1}
                    </span>
                    <span style={{ whiteSpace: 'pre' }}>{line || ' '}</span>
                  </div>
                ))}
              </div>
            </>
          )}
        </Spin>
      </Modal>
    </div>
  )
}

export default Simulation
