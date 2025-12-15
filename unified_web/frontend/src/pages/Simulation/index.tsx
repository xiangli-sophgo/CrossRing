/**
 * 仿真执行页面
 */
import React, { useState, useEffect, useRef } from 'react'
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
  StopOutlined,
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
  getTaskHistory,
  getRunningTasks,
  getConfigs,
  getTrafficFilesTree,
  getTrafficFileContent,
  getConfigContent,
  saveConfigContent,
  type SimulationRequest,
  type TaskStatus,
  type TaskHistoryItem,
  type ConfigOption,
  type TrafficTreeNode,
  type TrafficFileContentResponse,
} from '@/api/simulation'
import { getExperiments } from '@/api/experiments'
import type { Experiment } from '@/types'

// 导入拆分的组件
import {
  KCINConfigPanel,
  DCINConfigPanel,
  SweepConfigPanel,
  TaskStatusCard,
  SweepProgressCard,
  TaskHistoryTable,
  TrafficFileTree,
} from './components'

// 导入类型和工具函数
import type { SweepParam, SweepProgress, SavedSweepConfig } from './helpers'
import { calculateSweepValues, generateCombinations, sleep, groupTaskHistory } from './helpers'

const { Text } = Typography
const { Option } = Select

const Simulation: React.FC = () => {
  const navigate = useNavigate()
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [configs, setConfigs] = useState<{ kcin: ConfigOption[]; dcin: ConfigOption[] }>({ kcin: [], dcin: [] })
  const [trafficTree, setTrafficTree] = useState<TrafficTreeNode[]>([])
  const [selectedFiles, setSelectedFiles] = useState<string[]>([])
  const [loadingFiles, setLoadingFiles] = useState(false)
  const [expandedKeys, setExpandedKeys] = useState<string[]>([])

  // 当前任务状态
  const [currentTask, setCurrentTask] = useState<TaskStatus | null>(null)
  const [taskHistory, setTaskHistory] = useState<TaskHistoryItem[]>([])
  const [loadingHistory, setLoadingHistory] = useState(false)
  const [startTime, setStartTime] = useState<number | null>(null)

  // 配置编辑状态
  const [configValues, setConfigValues] = useState<Record<string, any>>({})
  const [loadingConfig, setLoadingConfig] = useState(false)
  // DCIN模式下的DIE拓扑配置
  const [dieConfigValues, setDieConfigValues] = useState<Record<string, any>>({})
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

  // 参数遍历状态
  const [sweepParams, setSweepParams] = useState<SweepParam[]>([])
  const [sweepTaskIds, setSweepTaskIds] = useState<string[]>([])
  const [sweepProgress, setSweepProgress] = useState<SweepProgress>({ total: 0, completed: 0, running: 0, failed: 0 })
  const [sweepRunning, setSweepRunning] = useState(false)
  const [savedSweepConfigs, setSavedSweepConfigs] = useState<SavedSweepConfig[]>([])
  const [sweepConfigName, setSweepConfigName] = useState('')
  const [sweepRestoring, setSweepRestoring] = useState(false)

  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const sweepPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // 恢复运行中的任务
  const restoreRunningTask = async () => {
    try {
      const data = await getRunningTasks()
      if (data.tasks.length > 0) {
        const latestTask = data.tasks[0]
        const fullStatus = await getTaskStatus(latestTask.task_id)
        setCurrentTask(fullStatus)
        if (latestTask.started_at) {
          setStartTime(new Date(latestTask.started_at).getTime())
        } else {
          setStartTime(new Date(latestTask.created_at).getTime())
        }
        startPolling(latestTask.task_id)
        message.info('已恢复运行中的任务')
      }
    } catch (error) {
      console.error('恢复运行中任务失败:', error)
    }
  }

  // 加载配置和历史
  useEffect(() => {
    loadConfigs()
    loadHistory()
    loadExistingExperiments()
    loadTrafficFilesTree('kcin')
    restoreRunningTask()
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
      if (sweepPollRef.current) clearInterval(sweepPollRef.current)
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
      const defaultConfig = data.kcin.find((c: ConfigOption) => c.path.includes('topo_5x4'))
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
      const data = await getTaskHistory(10)
      setTaskHistory(data.tasks)
    } catch (error) {
      console.error('加载历史失败:', error)
    } finally {
      setLoadingHistory(false)
    }
  }

  const loadTrafficFilesTree = async (filterMode?: 'kcin' | 'dcin') => {
    setLoadingFiles(true)
    try {
      const data = await getTrafficFilesTree(filterMode)
      setTrafficTree(data.tree)
    } catch (error) {
      console.error('加载流量文件失败:', error)
    } finally {
      setLoadingFiles(false)
    }
  }

  const loadConfigContent = async (configPath: string) => {
    if (!configPath) {
      setConfigValues({})
      return
    }
    setLoadingConfig(true)
    try {
      const content = await getConfigContent(configPath)
      setConfigValues(content)
    } catch (error) {
      console.error('加载配置内容失败:', error)
      message.error('加载配置内容失败')
      setConfigValues({})
    } finally {
      setLoadingConfig(false)
    }
  }

  const loadDieConfigContent = async (configPath: string) => {
    if (!configPath) {
      setDieConfigValues({})
      return
    }
    setLoadingDieConfig(true)
    try {
      const content = await getConfigContent(configPath)
      setDieConfigValues(content)
    } catch (error) {
      console.error('加载DIE配置内容失败:', error)
      message.error('加载DIE配置内容失败')
      setDieConfigValues({})
    } finally {
      setLoadingDieConfig(false)
    }
  }

  const updateConfigValue = (key: string, value: any) => {
    setConfigValues(prev => ({ ...prev, [key]: value }))
  }

  const updateDieConfigValue = (key: string, value: any) => {
    setDieConfigValues(prev => ({ ...prev, [key]: value }))
  }

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
        experiment_description: values.experiment_description,
        max_workers: values.max_workers,
      }
      const response = await runSimulation(request)
      message.success(`仿真任务已创建: ${response.task_id}`)
      setStartTime(Date.now())
      startPolling(response.task_id)
    } catch (error: any) {
      message.error(error.response?.data?.detail || '启动仿真失败')
    } finally {
      setLoading(false)
    }
  }

  const startPolling = (taskId: string) => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)

    const poll = async () => {
      try {
        const status = await getTaskStatus(taskId)
        setCurrentTask(status)
        if (['completed', 'failed', 'cancelled'].includes(status.status)) {
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
          loadHistory()
          if (status.status === 'completed') message.success('仿真完成')
          else if (status.status === 'failed') message.error('仿真失败')
        }
      } catch (error) {
        console.error('获取任务状态失败:', error)
      }
    }

    poll()
    pollIntervalRef.current = setInterval(poll, 1000)
  }

  const handleCancel = async () => {
    if (!currentTask) return
    try {
      await cancelTask(currentTask.task_id)
      message.info('任务已取消')
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
      setCurrentTask(null)
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

  const totalCombinations = sweepParams.length > 0
    ? sweepParams.reduce((acc, param) => acc * param.values.length, 1)
    : 0

  const availableParams = Object.keys(configValues).filter(key => {
    const value = configValues[key]
    return typeof value === 'number' && !sweepParams.find(p => p.key === key)
  })

  // 批量任务状态轮询
  const startSweepPolling = (taskIds: string[]) => {
    if (sweepPollRef.current) clearInterval(sweepPollRef.current)

    const poll = async () => {
      let completed = 0
      let running = 0
      let failed = 0

      for (const taskId of taskIds) {
        try {
          const status = await getTaskStatus(taskId)
          if (status.status === 'completed') completed++
          else if (['running', 'pending'].includes(status.status)) running++
          else failed++
        } catch {
          failed++
        }
      }

      setSweepProgress({ total: taskIds.length, completed, running, failed })

      if (completed + failed >= taskIds.length) {
        if (sweepPollRef.current) {
          clearInterval(sweepPollRef.current)
          sweepPollRef.current = null
        }
        setSweepRunning(false)
        sessionStorage.removeItem('sweepTaskIds')
        message.success(`参数遍历完成: ${completed} 成功, ${failed} 失败`)
        loadHistory()
      }
    }

    poll()
    sweepPollRef.current = setInterval(poll, 3000)
  }

  // 恢复参数遍历任务
  useEffect(() => {
    const restoreSweepTasks = async () => {
      const savedTaskIds = sessionStorage.getItem('sweepTaskIds')
      if (!savedTaskIds) return

      try {
        const taskIds: string[] = JSON.parse(savedTaskIds)
        if (taskIds.length === 0) return

        setSweepRestoring(true)
        setSweepTaskIds(taskIds)

        let completed = 0
        let running = 0
        let failed = 0

        for (const taskId of taskIds) {
          try {
            const status = await getTaskStatus(taskId)
            if (status.status === 'completed') completed++
            else if (['running', 'pending'].includes(status.status)) running++
            else failed++
          } catch {
            failed++
          }
        }

        setSweepProgress({ total: taskIds.length, completed, running, failed })

        if (running > 0) {
          setSweepRunning(true)
          startSweepPolling(taskIds)
          message.info('已恢复参数遍历任务')
        } else {
          sessionStorage.removeItem('sweepTaskIds')
        }

        setSweepRestoring(false)
      } catch (error) {
        console.error('恢复参数遍历任务失败:', error)
        sessionStorage.removeItem('sweepTaskIds')
        setSweepRestoring(false)
      }
    }

    restoreSweepTasks()
  }, [])

  const handleSweepSubmit = async (values: any) => {
    if (selectedFiles.length === 0) {
      message.warning('请选择流量文件')
      return
    }

    const combinations = generateCombinations(sweepParams)
    if (combinations.length === 0) {
      message.warning('请配置遍历参数')
      return
    }

    if (combinations.length > 100) {
      Modal.confirm({
        title: '组合数量较多',
        content: `将生成 ${combinations.length} 组参数组合，确定继续？`,
        onOk: () => executeSweep(values, combinations)
      })
      return
    }

    await executeSweep(values, combinations)
  }

  const executeSweep = async (values: any, combinations: Record<string, number>[]) => {
    setSweepRunning(true)
    setSweepTaskIds([])
    setSweepProgress({ total: combinations.length, completed: 0, running: 0, failed: 0 })

    const topology = `${values.rows}x${values.cols}`
    const batchId = Date.now().toString()
    const experimentName = values.experiment_name || `参数遍历_${new Date().toLocaleString()}`
    const taskIds: string[] = []

    for (let i = 0; i < combinations.length; i++) {
      const combo = combinations[i]
      const mergedOverrides = { ...configValues, ...combo }

      const request: SimulationRequest = {
        mode: values.mode,
        topology,
        config_path: values.config_path,
        config_overrides: mergedOverrides,
        die_config_path: values.mode === 'dcin' ? values.die_config_path : undefined,
        die_config_overrides: values.mode === 'dcin' && Object.keys(dieConfigValues).length > 0 ? dieConfigValues : undefined,
        traffic_source: 'file',
        traffic_files: selectedFiles,
        max_time: values.max_time,
        save_to_db: values.save_to_db,
        experiment_name: experimentName,
        experiment_description: `[batch:${batchId}] 参数组合 ${i + 1}/${combinations.length}: ${JSON.stringify(combo)}`,
        max_workers: values.max_workers,
      }

      try {
        const response = await runSimulation(request)
        taskIds.push(response.task_id)
        setSweepTaskIds([...taskIds])
        sessionStorage.setItem('sweepTaskIds', JSON.stringify(taskIds))
      } catch (error: any) {
        message.error(`任务 ${i + 1} 创建失败: ${error.response?.data?.detail || error.message}`)
      }

      if ((i + 1) % 5 === 0) {
        await sleep(500)
      }
    }

    if (taskIds.length > 0) {
      message.success(`已创建 ${taskIds.length} 个仿真任务`)
      startSweepPolling(taskIds)
    } else {
      setSweepRunning(false)
      sessionStorage.removeItem('sweepTaskIds')
    }
  }

  const handleDeleteGroupedTask = async (record: GroupedTask) => {
    try {
      for (const task of record.tasks) {
        await deleteTask(task.task_id)
      }
      message.success(`已删除 ${record.tasks.length} 个任务`)
      loadHistory()
    } catch (error: any) {
      message.error(error.response?.data?.detail || '删除失败')
    }
  }

  const mode = Form.useWatch('mode', form) || 'kcin'
  const groupedTaskHistory = groupTaskHistory(taskHistory)

  return (
    <div>
      <Row gutter={24} align="stretch">
        {/* 左侧：配置表单 */}
        <Col xs={24} lg={12} style={{ display: 'flex', flexDirection: 'column' }}>
          <Card
            title={
              <Space>
                <SettingOutlined style={{ color: primaryColor }} />
                <span>仿真配置</span>
              </Space>
            }
            style={{ marginBottom: 24, flex: 1 }}
          >
            <Form
              form={form}
              layout="vertical"
              initialValues={{
                mode: 'kcin',
                rows: 5,
                cols: 4,
                max_time: 6000,
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
                        const defaultConfig = configs.kcin.find(c => c.path.includes('topo_5x4'))
                        if (defaultConfig) {
                          form.setFieldsValue({ config_path: defaultConfig.path, rows: 5, cols: 4, die_config_path: undefined })
                          loadConfigContent(defaultConfig.path)
                          setDieConfigValues({})
                        }
                      } else {
                        const defaultDcinConfig = configs.dcin.find(c => c.path.includes('4die'))
                        const defaultKcinConfig = configs.kcin.find(c => c.path.includes('topo_5x4'))
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
                        <InputNumber min={1} max={20} placeholder="行" style={{ width: 70 }} onChange={(rows) => {
                          const cols = form.getFieldValue('cols')
                          if (rows && cols) {
                            const topoName = `${rows}x${cols}`
                            const configList = mode === 'kcin' ? configs.kcin : configs.dcin
                            const matchedConfig = configList.find(c => c.name === topoName || c.path.includes(topoName))
                            if (matchedConfig) {
                              form.setFieldValue('config_path', matchedConfig.path)
                              loadConfigContent(matchedConfig.path)
                            }
                          }
                        }} />
                      </Form.Item>
                      <Text type="secondary">×</Text>
                      <Form.Item name="cols" noStyle rules={[{ required: true, message: '请输入列数' }]}>
                        <InputNumber min={1} max={20} placeholder="列" style={{ width: 70 }} onChange={(cols) => {
                          const rows = form.getFieldValue('rows')
                          if (rows && cols) {
                            const topoName = `${rows}x${cols}`
                            const configList = mode === 'kcin' ? configs.kcin : configs.dcin
                            const matchedConfig = configList.find(c => c.name === topoName || c.path.includes(topoName))
                            if (matchedConfig) {
                              form.setFieldValue('config_path', matchedConfig.path)
                              loadConfigContent(matchedConfig.path)
                            }
                          }
                        }} />
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
                >
                  {(mode === 'kcin' ? configs.kcin : configs.dcin).map((c) => (
                    <Option key={c.path} value={c.path}>{c.name}</Option>
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
                      onClick={() => {
                        setSaveAsType('main')
                        setSaveAsName('')
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
                    <Select placeholder="请选择KCIN配置文件" onChange={loadDieConfigContent}>
                      {configs.kcin.map((c) => (
                        <Option key={c.path} value={c.path}>{c.name}</Option>
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
                          onClick={() => {
                            setSaveAsType('die')
                            setSaveAsName('')
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
                onAddSweepParam={addSweepParam}
                onUpdateSweepParam={updateSweepParam}
                onRemoveSweepParam={removeSweepParam}
                onSaveSweepConfig={saveSweepConfig}
                onLoadSweepConfig={loadSweepConfig}
                onDeleteSweepConfig={deleteSweepConfig}
                onSweepConfigNameChange={setSweepConfigName}
              />

              {/* 仿真设置 */}
              <Collapse
                size="small"
                style={{ marginBottom: 16 }}
                items={[{
                  key: 'sim_settings',
                  label: '仿真设置',
                  children: (
                    <Row gutter={[16, 8]}>
                      <Col span={8}>
                        <Form.Item name="max_time" label="最大仿真时间 (ns)">
                          <InputNumber min={100} max={100000} style={{ width: '100%' }} />
                        </Form.Item>
                      </Col>
                      <Col span={8}>
                        <Form.Item name="max_workers" label="并行数">
                          <InputNumber min={1} max={16} placeholder="留空使用默认" style={{ width: '100%' }} />
                        </Form.Item>
                      </Col>
                      <Col span={8}>
                        <Form.Item name="save_to_db" label="保存到数据库" valuePropName="checked">
                          <Switch />
                        </Form.Item>
                      </Col>
                    </Row>
                  ),
                }]}
              />

              {/* 实验信息 */}
              <Collapse
                size="small"
                style={{ marginBottom: 16 }}
                items={[{
                  key: 'experiment_info',
                  label: '实验信息',
                  children: (
                    <>
                      <Form.Item name="experiment_name" label="实验名称">
                        <AutoComplete
                          placeholder="输入或选择已有实验名称"
                          options={existingExperiments.map(e => ({ value: e.name }))}
                          filterOption={(inputValue, option) =>
                            option?.value.toLowerCase().includes(inputValue.toLowerCase()) || false
                          }
                        />
                      </Form.Item>
                      <Form.Item name="experiment_description" label="实验描述">
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
                      loading={sweepRunning}
                      disabled={currentTask?.status === 'running' || sweepParams.length === 0}
                      size="large"
                      style={{ height: 44 }}
                    >
                      {sweepRunning ? '批量执行中...' : `批量执行 (${totalCombinations}组)`}
                    </Button>
                  ) : (
                    <Button
                      type="primary"
                      htmlType="submit"
                      icon={<PlayCircleOutlined />}
                      loading={loading}
                      disabled={currentTask?.status === 'running'}
                      size="large"
                      style={{ height: 44 }}
                    >
                      开始仿真
                    </Button>
                  )}
                  {currentTask?.status === 'running' && (
                    <Button icon={<StopOutlined />} onClick={handleCancel} danger size="large" style={{ height: 44 }}>
                      取消任务
                    </Button>
                  )}
                </Space>
              </Form.Item>
            </Form>
          </Card>

          {/* 参数遍历进度卡片 */}
          {(sweepTaskIds.length > 0 || sweepRestoring) && (
            <SweepProgressCard
              sweepProgress={sweepProgress}
              sweepRunning={sweepRunning}
              sweepRestoring={sweepRestoring}
              onViewResults={() => navigate('/experiments')}
            />
          )}

          {/* 当前任务状态卡片 */}
          {currentTask && (
            <TaskStatusCard
              currentTask={currentTask}
              startTime={startTime}
            />
          )}
        </Col>

        {/* 右侧：文件选择 */}
        <Col xs={24} lg={12} style={{ display: 'flex', flexDirection: 'column' }}>
          <TrafficFileTree
            trafficTree={trafficTree}
            selectedFiles={selectedFiles}
            expandedKeys={expandedKeys}
            loading={loadingFiles}
            onSelect={setSelectedFiles}
            onExpandedKeysChange={setExpandedKeys}
            onRefresh={() => loadTrafficFilesTree(mode as 'kcin' | 'dcin')}
            onPreviewFile={handlePreviewFile}
          />
        </Col>
      </Row>

      {/* 历史任务 */}
      <TaskHistoryTable
        groupedTaskHistory={groupedTaskHistory}
        loading={loadingHistory}
        onRefresh={loadHistory}
        onViewResult={(experimentId) => navigate(`/experiments/${experimentId}`)}
        onDelete={handleDeleteGroupedTask}
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
            placeholder={mode === 'dcin' && saveAsType === 'main' ? 'dcin_my_config' : 'topo_my_config'}
            value={saveAsName}
            onChange={(e) => setSaveAsName(e.target.value)}
            addonAfter=".yaml"
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
