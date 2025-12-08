/**
 * 仿真执行页面
 */
import React, { useState, useEffect, useRef } from 'react'
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
  Table,
  Tag,
  Progress,
  message,
  Breadcrumb,
  Empty,
  Spin,
  Alert,
  Steps,
  Row,
  Col,
  Statistic,
  Divider,
  List,
  Collapse,
  Tooltip,
} from 'antd'
import {
  PlayCircleOutlined,
  StopOutlined,
  FolderOpenOutlined,
  ReloadOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  LoadingOutlined,
  SettingOutlined,
  FileTextOutlined,
  ThunderboltOutlined,
  TrophyOutlined,
  ClockCircleOutlined,
  SyncOutlined,
  HistoryOutlined,
  RocketOutlined,
  SaveOutlined,
} from '@ant-design/icons'
import { primaryColor, successColor, warningColor, errorColor } from '@/theme/colors'
import {
  runSimulation,
  getTaskStatus,
  cancelTask,
  getTaskHistory,
  getConfigs,
  getTrafficFiles,
  getConfigContent,
  saveConfigContent,
  type SimulationRequest,
  type TaskStatus,
  type TaskHistoryItem,
  type ConfigOption,
  type TrafficFileInfo,
} from '@/api/simulation'

const { Title, Text } = Typography
const { Option } = Select

// 配置参数描述映射
const CONFIG_TOOLTIPS: Record<string, string> = {
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
  ETag_BOTHSIDE_UPGRADE: 'Enable ETag upgrade on both sides',
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
  CROSSRING_VERSION: 'CrossRing architecture version (V1 or V2)',
  ENABLE_CROSSPOINT_CONFLICT_CHECK: 'Enable crosspoint conflict checking',
  ORDERING_PRESERVATION_MODE: '0=Disabled, 1=Single side (TL/TU), 2=Both sides (whitelist), 3=Dynamic (src-dest based)',
  ORDERING_ETAG_UPGRADE_MODE: '0=Upgrade ETag only on resource failure, 1=Also upgrade on ordering failure',
  ORDERING_GRANULARITY: '0=IP level ordering, 1=Node level ordering',
  REVERSE_DIRECTION_ENABLED: 'Enable reverse direction flow control when normal direction is congested',
  REVERSE_DIRECTION_THRESHOLD: 'Threshold ratio (0.0-1.0) for triggering reverse direction flow',
  // Tag Config
  RB_ONLY_TAG_NUM_HORIZONTAL: 'Ring buffer only tag number for horizontal direction',
  RB_ONLY_TAG_NUM_VERTICAL: 'Ring buffer only tag number for vertical direction',
  // Allowed Source Nodes (双侧下环方向配置)
  TL_ALLOWED_SOURCE_NODES: 'Node IDs allowed to eject to TL (left) direction',
  TR_ALLOWED_SOURCE_NODES: 'Node IDs allowed to eject to TR (right) direction',
  TU_ALLOWED_SOURCE_NODES: 'Node IDs allowed to eject to TU (up) direction',
  TD_ALLOWED_SOURCE_NODES: 'Node IDs allowed to eject to TD (down) direction',
  // Arbitration
  ARBITRATION_TYPE: 'Arbitration algorithm type for queue selection',
  ARBITRATION_ITERATIONS: 'Number of iterations for iSLIP algorithm',
  ARBITRATION_WEIGHT_STRATEGY: 'Weight calculation strategy: queue_length, fixed, priority',
}

// 带Tooltip的配置项标签
const ConfigLabel: React.FC<{ name: string }> = ({ name }) => (
  <Tooltip title={CONFIG_TOOLTIPS[name] || name}>
    <Text type="secondary" style={{ cursor: 'help' }}>{name}</Text>
  </Tooltip>
)

// 获取当前步骤
const getCurrentStep = (currentTask: TaskStatus | null, selectedFiles: string[]): number => {
  if (!currentTask) {
    return selectedFiles.length > 0 ? 1 : 0
  }
  if (currentTask.status === 'running') return 2
  if (['completed', 'failed', 'cancelled'].includes(currentTask.status)) return 3
  return 1
}

// 获取步骤状态
const getStepStatus = (stepIndex: number, currentStep: number, taskStatus?: string): 'wait' | 'process' | 'finish' | 'error' => {
  if (stepIndex < currentStep) return 'finish'
  if (stepIndex === currentStep) {
    if (stepIndex === 3 && taskStatus === 'failed') return 'error'
    if (stepIndex === 2) return 'process'
    return 'process'
  }
  return 'wait'
}

const Simulation: React.FC = () => {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [configs, setConfigs] = useState<{ kcin: ConfigOption[]; dcin: ConfigOption[] }>({ kcin: [], dcin: [] })
  const [trafficFiles, setTrafficFiles] = useState<TrafficFileInfo[]>([])
  const [directories, setDirectories] = useState<string[]>([])
  const [currentPath, setCurrentPath] = useState('')
  const [selectedFiles, setSelectedFiles] = useState<string[]>([])
  const [loadingFiles, setLoadingFiles] = useState(false)

  // 当前任务状态
  const [currentTask, setCurrentTask] = useState<TaskStatus | null>(null)
  const [taskHistory, setTaskHistory] = useState<TaskHistoryItem[]>([])
  const [loadingHistory, setLoadingHistory] = useState(false)
  const [startTime, setStartTime] = useState<number | null>(null)

  // 配置编辑状态
  const [configValues, setConfigValues] = useState<Record<string, any>>({})
  const [loadingConfig, setLoadingConfig] = useState(false)

  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null)

  // 加载配置和历史
  useEffect(() => {
    loadConfigs()
    loadHistory()
    loadTrafficFiles('')
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
      }
    }
  }, [])

  const loadConfigs = async () => {
    try {
      const data = await getConfigs()
      setConfigs(data)
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

  const loadTrafficFiles = async (path: string) => {
    setLoadingFiles(true)
    try {
      const data = await getTrafficFiles(path)
      setTrafficFiles(data.files)
      setDirectories(data.directories)
      setCurrentPath(data.current_path)
    } catch (error) {
      console.error('加载流量文件失败:', error)
    } finally {
      setLoadingFiles(false)
    }
  }

  // 加载配置文件内容
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

  // 更新单个配置值
  const updateConfigValue = (key: string, value: any) => {
    setConfigValues(prev => ({ ...prev, [key]: value }))
  }

  const navigateToDir = (dir: string) => {
    const newPath = currentPath ? `${currentPath}/${dir}` : dir
    loadTrafficFiles(newPath)
    setSelectedFiles([])
  }

  const navigateBack = () => {
    const parts = currentPath.split('/')
    parts.pop()
    const newPath = parts.join('/')
    loadTrafficFiles(newPath)
    setSelectedFiles([])
  }

  // 提交仿真任务
  const handleSubmit = async (values: any) => {
    if (selectedFiles.length === 0) {
      message.warning('请选择流量文件')
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
        traffic_source: 'file',
        traffic_files: selectedFiles,
        traffic_path: currentPath,
        max_time: values.max_time,
        save_to_db: values.save_to_db,
        experiment_name: values.experiment_name,
        result_granularity: values.result_granularity,
      }

      const response = await runSimulation(request)
      message.success(`仿真任务已创建: ${response.task_id}`)
      setStartTime(Date.now())

      // 开始轮询任务状态
      startPolling(response.task_id)
    } catch (error: any) {
      message.error(error.response?.data?.detail || '启动仿真失败')
    } finally {
      setLoading(false)
    }
  }

  // 轮询任务状态
  const startPolling = (taskId: string) => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current)
    }

    const poll = async () => {
      try {
        const status = await getTaskStatus(taskId)
        setCurrentTask(status)

        if (['completed', 'failed', 'cancelled'].includes(status.status)) {
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current)
          }
          loadHistory()
          if (status.status === 'completed') {
            message.success('仿真完成')
          } else if (status.status === 'failed') {
            message.error('仿真失败')
          }
        }
      } catch (error) {
        console.error('获取任务状态失败:', error)
      }
    }

    poll()
    pollIntervalRef.current = setInterval(poll, 2000)
  }

  // 取消任务
  const handleCancel = async () => {
    if (!currentTask) return
    try {
      await cancelTask(currentTask.task_id)
      message.info('任务已取消')
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
      }
      setCurrentTask(null)
      loadHistory()
    } catch (error) {
      message.error('取消任务失败')
    }
  }

  const getStatusTag = (status: string) => {
    const statusMap: Record<string, { color: string; text: string; icon: React.ReactNode }> = {
      pending: { color: 'default', text: '等待中', icon: <ClockCircleOutlined /> },
      running: { color: 'processing', text: '运行中', icon: <SyncOutlined spin /> },
      completed: { color: 'success', text: '已完成', icon: <CheckCircleOutlined /> },
      failed: { color: 'error', text: '失败', icon: <CloseCircleOutlined /> },
      cancelled: { color: 'warning', text: '已取消', icon: <StopOutlined /> },
    }
    const { color, text, icon } = statusMap[status] || { color: 'default', text: status, icon: null }
    return <Tag color={color} icon={icon}>{text}</Tag>
  }

  const mode = Form.useWatch('mode', form) || 'kcin'
  const currentStep = getCurrentStep(currentTask, selectedFiles)

  // 计算运行时间
  const getElapsedTime = () => {
    if (!startTime) return '0秒'
    const elapsed = Math.floor((Date.now() - startTime) / 1000)
    if (elapsed < 60) return `${elapsed}秒`
    const mins = Math.floor(elapsed / 60)
    const secs = elapsed % 60
    return `${mins}分${secs}秒`
  }

  return (
    <div>
      {/* 流程步骤指示器 */}
      <Card style={{ marginBottom: 24 }}>
        <Steps
          current={currentStep}
          items={[
            {
              title: '配置参数',
              description: '设置仿真模式和拓扑',
              icon: <SettingOutlined />,
              status: getStepStatus(0, currentStep, currentTask?.status),
            },
            {
              title: '选择文件',
              description: `已选${selectedFiles.length}个文件`,
              icon: <FileTextOutlined />,
              status: getStepStatus(1, currentStep, currentTask?.status),
            },
            {
              title: '执行仿真',
              description: currentTask?.status === 'running' ? `${currentTask.progress}%` : '运行仿真任务',
              icon: currentTask?.status === 'running' ? <LoadingOutlined /> : <ThunderboltOutlined />,
              status: getStepStatus(2, currentStep, currentTask?.status),
            },
            {
              title: '查看结果',
              description: currentTask?.status === 'completed' ? '仿真完成' :
                          currentTask?.status === 'failed' ? '仿真失败' : '等待完成',
              icon: <TrophyOutlined />,
              status: getStepStatus(3, currentStep, currentTask?.status),
            },
          ]}
        />
      </Card>

      <Row gutter={24}>
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
            initialValues={{
              mode: 'kcin',
              rows: 5,
              cols: 4,
              max_time: 6000,
              save_to_db: true,
              result_granularity: 'per_file',
            }}
            onFinish={handleSubmit}
          >
            <Row gutter={24}>
              <Col span={12}>
                <Form.Item name="mode" label="仿真模式" rules={[{ required: true }]}>
                  <Radio.Group>
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

            <Form.Item name="config_path" label="配置文件" rules={[{ required: true, message: '请选择配置文件' }]}>
              <Select placeholder="请选择配置文件" onChange={loadConfigContent}>
                {(mode === 'kcin' ? configs.kcin : configs.dcin).map((c) => (
                  <Option key={c.path} value={c.path}>
                    {c.name}
                  </Option>
                ))}
              </Select>
            </Form.Item>

            {/* 配置编辑面板 */}
            {Object.keys(configValues).length > 0 && (
              <Spin spinning={loadingConfig}>
                <Collapse
                  size="small"
                  style={{ marginBottom: 16 }}
                  items={[
                    {
                      key: 'basic',
                      label: 'Basic Parameters',
                      children: (
                        <Row gutter={[16, 8]}>
                          {configValues.FLIT_SIZE !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="FLIT_SIZE" /></div>
                              <InputNumber value={configValues.FLIT_SIZE} onChange={(v) => updateConfigValue('FLIT_SIZE', v)} min={1} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.BURST !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="BURST" /></div>
                              <InputNumber value={configValues.BURST} onChange={(v) => updateConfigValue('BURST', v)} min={1} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.NETWORK_FREQUENCY !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="NETWORK_FREQUENCY" /></div>
                              <InputNumber value={configValues.NETWORK_FREQUENCY} onChange={(v) => updateConfigValue('NETWORK_FREQUENCY', v)} min={1} style={{ width: '100%' }} />
                            </Col>
                          )}
                        </Row>
                      ),
                    },
                    {
                      key: 'buffer',
                      label: 'Buffer Size',
                      children: (
                        <Row gutter={[16, 8]}>
                          {configValues.RN_RDB_SIZE !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="RN_RDB_SIZE" /></div>
                              <InputNumber value={configValues.RN_RDB_SIZE} onChange={(v) => updateConfigValue('RN_RDB_SIZE', v)} min={1} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.RN_WDB_SIZE !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="RN_WDB_SIZE" /></div>
                              <InputNumber value={configValues.RN_WDB_SIZE} onChange={(v) => updateConfigValue('RN_WDB_SIZE', v)} min={1} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.SN_DDR_RDB_SIZE !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="SN_DDR_RDB_SIZE" /></div>
                              <InputNumber value={configValues.SN_DDR_RDB_SIZE} onChange={(v) => updateConfigValue('SN_DDR_RDB_SIZE', v)} min={1} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.SN_DDR_WDB_SIZE !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="SN_DDR_WDB_SIZE" /></div>
                              <InputNumber value={configValues.SN_DDR_WDB_SIZE} onChange={(v) => updateConfigValue('SN_DDR_WDB_SIZE', v)} min={1} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.SN_L2M_RDB_SIZE !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="SN_L2M_RDB_SIZE" /></div>
                              <InputNumber value={configValues.SN_L2M_RDB_SIZE} onChange={(v) => updateConfigValue('SN_L2M_RDB_SIZE', v)} min={1} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.SN_L2M_WDB_SIZE !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="SN_L2M_WDB_SIZE" /></div>
                              <InputNumber value={configValues.SN_L2M_WDB_SIZE} onChange={(v) => updateConfigValue('SN_L2M_WDB_SIZE', v)} min={1} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.UNIFIED_RW_TRACKER !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="UNIFIED_RW_TRACKER" /></div>
                              <Switch checked={configValues.UNIFIED_RW_TRACKER} onChange={(v) => updateConfigValue('UNIFIED_RW_TRACKER', v)} />
                            </Col>
                          )}
                        </Row>
                      ),
                    },
                    {
                      key: 'kcin',
                      label: 'KCIN Config',
                      children: (
                        <Row gutter={[16, 8]}>
                          {/* Slice Per Link */}
                          {configValues.SLICE_PER_LINK_HORIZONTAL !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="SLICE_PER_LINK_HORIZONTAL" /></div>
                              <InputNumber value={configValues.SLICE_PER_LINK_HORIZONTAL} onChange={(v) => updateConfigValue('SLICE_PER_LINK_HORIZONTAL', v)} min={1} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.SLICE_PER_LINK_VERTICAL !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="SLICE_PER_LINK_VERTICAL" /></div>
                              <InputNumber value={configValues.SLICE_PER_LINK_VERTICAL} onChange={(v) => updateConfigValue('SLICE_PER_LINK_VERTICAL', v)} min={1} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {/* FIFO Depth */}
                          {configValues.IQ_CH_FIFO_DEPTH !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="IQ_CH_FIFO_DEPTH" /></div>
                              <InputNumber value={configValues.IQ_CH_FIFO_DEPTH} onChange={(v) => updateConfigValue('IQ_CH_FIFO_DEPTH', v)} min={1} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.EQ_CH_FIFO_DEPTH !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="EQ_CH_FIFO_DEPTH" /></div>
                              <InputNumber value={configValues.EQ_CH_FIFO_DEPTH} onChange={(v) => updateConfigValue('EQ_CH_FIFO_DEPTH', v)} min={1} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.IQ_OUT_FIFO_DEPTH_HORIZONTAL !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="IQ_OUT_FIFO_DEPTH_HORIZONTAL" /></div>
                              <InputNumber value={configValues.IQ_OUT_FIFO_DEPTH_HORIZONTAL} onChange={(v) => updateConfigValue('IQ_OUT_FIFO_DEPTH_HORIZONTAL', v)} min={1} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.IQ_OUT_FIFO_DEPTH_VERTICAL !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="IQ_OUT_FIFO_DEPTH_VERTICAL" /></div>
                              <InputNumber value={configValues.IQ_OUT_FIFO_DEPTH_VERTICAL} onChange={(v) => updateConfigValue('IQ_OUT_FIFO_DEPTH_VERTICAL', v)} min={1} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.IQ_OUT_FIFO_DEPTH_EQ !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="IQ_OUT_FIFO_DEPTH_EQ" /></div>
                              <InputNumber value={configValues.IQ_OUT_FIFO_DEPTH_EQ} onChange={(v) => updateConfigValue('IQ_OUT_FIFO_DEPTH_EQ', v)} min={1} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.RB_OUT_FIFO_DEPTH !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="RB_OUT_FIFO_DEPTH" /></div>
                              <InputNumber value={configValues.RB_OUT_FIFO_DEPTH} onChange={(v) => updateConfigValue('RB_OUT_FIFO_DEPTH', v)} min={1} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.RB_IN_FIFO_DEPTH !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="RB_IN_FIFO_DEPTH" /></div>
                              <InputNumber value={configValues.RB_IN_FIFO_DEPTH} onChange={(v) => updateConfigValue('RB_IN_FIFO_DEPTH', v)} min={1} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.EQ_IN_FIFO_DEPTH !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="EQ_IN_FIFO_DEPTH" /></div>
                              <InputNumber value={configValues.EQ_IN_FIFO_DEPTH} onChange={(v) => updateConfigValue('EQ_IN_FIFO_DEPTH', v)} min={1} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.IP_L2H_FIFO_DEPTH !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="IP_L2H_FIFO_DEPTH" /></div>
                              <InputNumber value={configValues.IP_L2H_FIFO_DEPTH} onChange={(v) => updateConfigValue('IP_L2H_FIFO_DEPTH', v)} min={1} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.IP_H2L_H_FIFO_DEPTH !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="IP_H2L_H_FIFO_DEPTH" /></div>
                              <InputNumber value={configValues.IP_H2L_H_FIFO_DEPTH} onChange={(v) => updateConfigValue('IP_H2L_H_FIFO_DEPTH', v)} min={1} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.IP_H2L_L_FIFO_DEPTH !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="IP_H2L_L_FIFO_DEPTH" /></div>
                              <InputNumber value={configValues.IP_H2L_L_FIFO_DEPTH} onChange={(v) => updateConfigValue('IP_H2L_L_FIFO_DEPTH', v)} min={1} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {/* ETag Config */}
                          {configValues.TL_Etag_T2_UE_MAX !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="TL_Etag_T2_UE_MAX" /></div>
                              <InputNumber value={configValues.TL_Etag_T2_UE_MAX} onChange={(v) => updateConfigValue('TL_Etag_T2_UE_MAX', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.TL_Etag_T1_UE_MAX !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="TL_Etag_T1_UE_MAX" /></div>
                              <InputNumber value={configValues.TL_Etag_T1_UE_MAX} onChange={(v) => updateConfigValue('TL_Etag_T1_UE_MAX', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.TR_Etag_T2_UE_MAX !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="TR_Etag_T2_UE_MAX" /></div>
                              <InputNumber value={configValues.TR_Etag_T2_UE_MAX} onChange={(v) => updateConfigValue('TR_Etag_T2_UE_MAX', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.TU_Etag_T2_UE_MAX !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="TU_Etag_T2_UE_MAX" /></div>
                              <InputNumber value={configValues.TU_Etag_T2_UE_MAX} onChange={(v) => updateConfigValue('TU_Etag_T2_UE_MAX', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.TU_Etag_T1_UE_MAX !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="TU_Etag_T1_UE_MAX" /></div>
                              <InputNumber value={configValues.TU_Etag_T1_UE_MAX} onChange={(v) => updateConfigValue('TU_Etag_T1_UE_MAX', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.TD_Etag_T2_UE_MAX !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="TD_Etag_T2_UE_MAX" /></div>
                              <InputNumber value={configValues.TD_Etag_T2_UE_MAX} onChange={(v) => updateConfigValue('TD_Etag_T2_UE_MAX', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.ETAG_T1_ENABLED !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="ETAG_T1_ENABLED" /></div>
                              <Switch checked={!!configValues.ETAG_T1_ENABLED} onChange={(v) => updateConfigValue('ETAG_T1_ENABLED', v ? 1 : 0)} />
                            </Col>
                          )}
                          {configValues.ETag_BOTHSIDE_UPGRADE !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="ETag_BOTHSIDE_UPGRADE" /></div>
                              <Switch checked={!!configValues.ETag_BOTHSIDE_UPGRADE} onChange={(v) => updateConfigValue('ETag_BOTHSIDE_UPGRADE', v ? 1 : 0)} />
                            </Col>
                          )}
                          {/* ITag Config */}
                          {configValues.ITag_TRIGGER_Th_H !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="ITag_TRIGGER_Th_H" /></div>
                              <InputNumber value={configValues.ITag_TRIGGER_Th_H} onChange={(v) => updateConfigValue('ITag_TRIGGER_Th_H', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.ITag_TRIGGER_Th_V !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="ITag_TRIGGER_Th_V" /></div>
                              <InputNumber value={configValues.ITag_TRIGGER_Th_V} onChange={(v) => updateConfigValue('ITag_TRIGGER_Th_V', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.ITag_MAX_Num_H !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="ITag_MAX_Num_H" /></div>
                              <InputNumber value={configValues.ITag_MAX_Num_H} onChange={(v) => updateConfigValue('ITag_MAX_Num_H', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.ITag_MAX_Num_V !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="ITag_MAX_Num_V" /></div>
                              <InputNumber value={configValues.ITag_MAX_Num_V} onChange={(v) => updateConfigValue('ITag_MAX_Num_V', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {/* Latency */}
                          {configValues.DDR_R_LATENCY !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="DDR_R_LATENCY" /></div>
                              <InputNumber value={configValues.DDR_R_LATENCY} onChange={(v) => updateConfigValue('DDR_R_LATENCY', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.DDR_R_LATENCY_VAR !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="DDR_R_LATENCY_VAR" /></div>
                              <InputNumber value={configValues.DDR_R_LATENCY_VAR} onChange={(v) => updateConfigValue('DDR_R_LATENCY_VAR', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.DDR_W_LATENCY !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="DDR_W_LATENCY" /></div>
                              <InputNumber value={configValues.DDR_W_LATENCY} onChange={(v) => updateConfigValue('DDR_W_LATENCY', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.L2M_R_LATENCY !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="L2M_R_LATENCY" /></div>
                              <InputNumber value={configValues.L2M_R_LATENCY} onChange={(v) => updateConfigValue('L2M_R_LATENCY', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.L2M_W_LATENCY !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="L2M_W_LATENCY" /></div>
                              <InputNumber value={configValues.L2M_W_LATENCY} onChange={(v) => updateConfigValue('L2M_W_LATENCY', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.SN_TRACKER_RELEASE_LATENCY !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="SN_TRACKER_RELEASE_LATENCY" /></div>
                              <InputNumber value={configValues.SN_TRACKER_RELEASE_LATENCY} onChange={(v) => updateConfigValue('SN_TRACKER_RELEASE_LATENCY', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.SN_PROCESSING_LATENCY !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="SN_PROCESSING_LATENCY" /></div>
                              <InputNumber value={configValues.SN_PROCESSING_LATENCY} onChange={(v) => updateConfigValue('SN_PROCESSING_LATENCY', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.RN_PROCESSING_LATENCY !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="RN_PROCESSING_LATENCY" /></div>
                              <InputNumber value={configValues.RN_PROCESSING_LATENCY} onChange={(v) => updateConfigValue('RN_PROCESSING_LATENCY', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {/* Bandwidth Limit */}
                          {configValues.GDMA_BW_LIMIT !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="GDMA_BW_LIMIT" /></div>
                              <InputNumber value={configValues.GDMA_BW_LIMIT} onChange={(v) => updateConfigValue('GDMA_BW_LIMIT', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.SDMA_BW_LIMIT !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="SDMA_BW_LIMIT" /></div>
                              <InputNumber value={configValues.SDMA_BW_LIMIT} onChange={(v) => updateConfigValue('SDMA_BW_LIMIT', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.CDMA_BW_LIMIT !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="CDMA_BW_LIMIT" /></div>
                              <InputNumber value={configValues.CDMA_BW_LIMIT} onChange={(v) => updateConfigValue('CDMA_BW_LIMIT', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.DDR_BW_LIMIT !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="DDR_BW_LIMIT" /></div>
                              <InputNumber value={configValues.DDR_BW_LIMIT} onChange={(v) => updateConfigValue('DDR_BW_LIMIT', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                          {configValues.L2M_BW_LIMIT !== undefined && (
                            <Col span={8}>
                              <div style={{ marginBottom: 4 }}><ConfigLabel name="L2M_BW_LIMIT" /></div>
                              <InputNumber value={configValues.L2M_BW_LIMIT} onChange={(v) => updateConfigValue('L2M_BW_LIMIT', v)} min={0} style={{ width: '100%' }} />
                            </Col>
                          )}
                        </Row>
                      ),
                    },
                    {
                      key: 'features',
                      label: 'Feature Config',
                      children: (
                        <div>
                          {/* Version */}
                          {configValues.CROSSRING_VERSION !== undefined && (
                            <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                              <Col span={10}><ConfigLabel name="CROSSRING_VERSION" /></Col>
                              <Col span={14}>
                                <Select value={configValues.CROSSRING_VERSION} onChange={(v) => updateConfigValue('CROSSRING_VERSION', v)} style={{ width: 120 }}>
                                  <Option value="V1">V1</Option>
                                  <Option value="V2">V2</Option>
                                </Select>
                              </Col>
                            </Row>
                          )}
                          {/* Tag - V2 only */}
                          {configValues.CROSSRING_VERSION === 'V2' && configValues.RB_ONLY_TAG_NUM_HORIZONTAL !== undefined && (
                            <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                              <Col span={10}><ConfigLabel name="RB_ONLY_TAG_NUM_HORIZONTAL" /></Col>
                              <Col span={14}>
                                <InputNumber value={configValues.RB_ONLY_TAG_NUM_HORIZONTAL} onChange={(v) => updateConfigValue('RB_ONLY_TAG_NUM_HORIZONTAL', v)} min={0} style={{ width: 120 }} />
                              </Col>
                            </Row>
                          )}
                          {configValues.CROSSRING_VERSION === 'V2' && configValues.RB_ONLY_TAG_NUM_VERTICAL !== undefined && (
                            <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                              <Col span={10}><ConfigLabel name="RB_ONLY_TAG_NUM_VERTICAL" /></Col>
                              <Col span={14}>
                                <InputNumber value={configValues.RB_ONLY_TAG_NUM_VERTICAL} onChange={(v) => updateConfigValue('RB_ONLY_TAG_NUM_VERTICAL', v)} min={0} style={{ width: 120 }} />
                              </Col>
                            </Row>
                          )}
                          {/* Ordering */}
                          {configValues.ORDERING_PRESERVATION_MODE !== undefined && (
                            <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                              <Col span={10}><ConfigLabel name="ORDERING_PRESERVATION_MODE" /></Col>
                              <Col span={14}>
                                <Select value={configValues.ORDERING_PRESERVATION_MODE} onChange={(v) => updateConfigValue('ORDERING_PRESERVATION_MODE', v)} style={{ width: 160 }}>
                                  <Option value={0}>0 - Disabled</Option>
                                  <Option value={1}>1 - Single Side</Option>
                                  <Option value={2}>2 - Both Sides</Option>
                                  <Option value={3}>3 - Dynamic</Option>
                                </Select>
                              </Col>
                            </Row>
                          )}
                          {configValues.ORDERING_ETAG_UPGRADE_MODE !== undefined && (
                            <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                              <Col span={10}><ConfigLabel name="ORDERING_ETAG_UPGRADE_MODE" /></Col>
                              <Col span={14}>
                                <Select value={configValues.ORDERING_ETAG_UPGRADE_MODE} onChange={(v) => updateConfigValue('ORDERING_ETAG_UPGRADE_MODE', v)} style={{ width: 160 }}>
                                  <Option value={0}>0 - Resource Only</Option>
                                  <Option value={1}>1 - Include Ordering</Option>
                                </Select>
                              </Col>
                            </Row>
                          )}
                          {configValues.ORDERING_GRANULARITY !== undefined && (
                            <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                              <Col span={10}><ConfigLabel name="ORDERING_GRANULARITY" /></Col>
                              <Col span={14}>
                                <Select value={configValues.ORDERING_GRANULARITY} onChange={(v) => updateConfigValue('ORDERING_GRANULARITY', v)} style={{ width: 160 }}>
                                  <Option value={0}>0 - IP Level</Option>
                                  <Option value={1}>1 - Node Level</Option>
                                </Select>
                              </Col>
                            </Row>
                          )}
                          {/* Allowed Source Nodes - 双侧下环方向配置 (仅 ORDERING_PRESERVATION_MODE === 2 时显示) */}
                          {configValues.ORDERING_PRESERVATION_MODE === 2 && (
                            <>
                              {configValues.TL_ALLOWED_SOURCE_NODES !== undefined && (
                                <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                                  <Col span={10}><ConfigLabel name="TL_ALLOWED_SOURCE_NODES" /></Col>
                                  <Col span={14}>
                                    <Input value={Array.isArray(configValues.TL_ALLOWED_SOURCE_NODES) ? configValues.TL_ALLOWED_SOURCE_NODES.join(', ') : configValues.TL_ALLOWED_SOURCE_NODES} onChange={(e) => updateConfigValue('TL_ALLOWED_SOURCE_NODES', e.target.value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n)))} placeholder="e.g. 2,3,6,7" style={{ width: '100%' }} />
                                  </Col>
                                </Row>
                              )}
                              {configValues.TR_ALLOWED_SOURCE_NODES !== undefined && (
                                <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                                  <Col span={10}><ConfigLabel name="TR_ALLOWED_SOURCE_NODES" /></Col>
                                  <Col span={14}>
                                    <Input value={Array.isArray(configValues.TR_ALLOWED_SOURCE_NODES) ? configValues.TR_ALLOWED_SOURCE_NODES.join(', ') : configValues.TR_ALLOWED_SOURCE_NODES} onChange={(e) => updateConfigValue('TR_ALLOWED_SOURCE_NODES', e.target.value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n)))} placeholder="e.g. 0,1,4,5" style={{ width: '100%' }} />
                                  </Col>
                                </Row>
                              )}
                              {configValues.TU_ALLOWED_SOURCE_NODES !== undefined && (
                                <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                                  <Col span={10}><ConfigLabel name="TU_ALLOWED_SOURCE_NODES" /></Col>
                                  <Col span={14}>
                                    <Input value={Array.isArray(configValues.TU_ALLOWED_SOURCE_NODES) ? configValues.TU_ALLOWED_SOURCE_NODES.join(', ') : configValues.TU_ALLOWED_SOURCE_NODES} onChange={(e) => updateConfigValue('TU_ALLOWED_SOURCE_NODES', e.target.value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n)))} placeholder="e.g. 8,9,10,11" style={{ width: '100%' }} />
                                  </Col>
                                </Row>
                              )}
                              {configValues.TD_ALLOWED_SOURCE_NODES !== undefined && (
                                <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                                  <Col span={10}><ConfigLabel name="TD_ALLOWED_SOURCE_NODES" /></Col>
                                  <Col span={14}>
                                    <Input value={Array.isArray(configValues.TD_ALLOWED_SOURCE_NODES) ? configValues.TD_ALLOWED_SOURCE_NODES.join(', ') : configValues.TD_ALLOWED_SOURCE_NODES} onChange={(e) => updateConfigValue('TD_ALLOWED_SOURCE_NODES', e.target.value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n)))} placeholder="e.g. 0,1,2,3" style={{ width: '100%' }} />
                                  </Col>
                                </Row>
                              )}
                            </>
                          )}
                          {/* Reverse Direction */}
                          {configValues.REVERSE_DIRECTION_ENABLED !== undefined && (
                            <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                              <Col span={10}><ConfigLabel name="REVERSE_DIRECTION_ENABLED" /></Col>
                              <Col span={14}>
                                <Switch checked={!!configValues.REVERSE_DIRECTION_ENABLED} onChange={(v) => updateConfigValue('REVERSE_DIRECTION_ENABLED', v ? 1 : 0)} />
                              </Col>
                            </Row>
                          )}
                          {configValues.REVERSE_DIRECTION_THRESHOLD !== undefined && (
                            <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                              <Col span={10}><ConfigLabel name="REVERSE_DIRECTION_THRESHOLD" /></Col>
                              <Col span={14}>
                                <InputNumber value={configValues.REVERSE_DIRECTION_THRESHOLD} onChange={(v) => updateConfigValue('REVERSE_DIRECTION_THRESHOLD', v)} min={0} max={1} step={0.05} style={{ width: 120 }} />
                              </Col>
                            </Row>
                          )}
                          {/* Arbitration */}
                          {configValues.arbitration?.default?.type !== undefined && (
                            <>
                              <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                                <Col span={10}><ConfigLabel name="ARBITRATION_TYPE" /></Col>
                                <Col span={14}>
                                  <Select
                                    value={configValues.arbitration.default.type}
                                    onChange={(v) => {
                                      // 根据类型设置或清理参数
                                      const newDefault: any = { type: v }
                                      if (v === 'islip') {
                                        newDefault.iterations = configValues.arbitration.default.iterations || 1
                                        newDefault.weight_strategy = configValues.arbitration.default.weight_strategy || 'queue_length'
                                      }
                                      // 其他类型不需要额外参数
                                      updateConfigValue('arbitration', { ...configValues.arbitration, default: newDefault })
                                    }}
                                    style={{ width: 160 }}
                                  >
                                    <Option value="round_robin">Round Robin</Option>
                                    <Option value="islip">iSLIP</Option>
                                    <Option value="weighted">Weighted</Option>
                                    <Option value="priority">Priority</Option>
                                    <Option value="dynamic">Dynamic</Option>
                                    <Option value="random">Random</Option>
                                  </Select>
                                </Col>
                              </Row>
                              {/* iSLIP 专用参数 */}
                              {configValues.arbitration.default.type === 'islip' && (
                                <>
                                  <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                                    <Col span={10}><ConfigLabel name="ARBITRATION_ITERATIONS" /></Col>
                                    <Col span={14}>
                                      <InputNumber
                                        value={configValues.arbitration.default.iterations || 1}
                                        onChange={(v) => updateConfigValue('arbitration', { ...configValues.arbitration, default: { ...configValues.arbitration.default, iterations: v } })}
                                        min={1}
                                        max={10}
                                        style={{ width: 120 }}
                                      />
                                    </Col>
                                  </Row>
                                  <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                                    <Col span={10}><ConfigLabel name="ARBITRATION_WEIGHT_STRATEGY" /></Col>
                                    <Col span={14}>
                                      <Select
                                        value={configValues.arbitration.default.weight_strategy || 'queue_length'}
                                        onChange={(v) => updateConfigValue('arbitration', { ...configValues.arbitration, default: { ...configValues.arbitration.default, weight_strategy: v } })}
                                        style={{ width: 160 }}
                                      >
                                        <Option value="queue_length">Queue Length</Option>
                                        <Option value="fixed">Fixed</Option>
                                        <Option value="priority">Priority</Option>
                                      </Select>
                                    </Col>
                                  </Row>
                                </>
                              )}
                            </>
                          )}
                        </div>
                      ),
                    },
                  ]}
                />
                <div style={{ marginTop: 12, textAlign: 'right' }}>
                  <Button
                    icon={<SaveOutlined />}
                    onClick={async () => {
                      const configPath = form.getFieldValue('config_path')
                      if (!configPath) {
                        message.warning('请先选择配置文件')
                        return
                      }
                      try {
                        await saveConfigContent(configPath, configValues)
                        message.success('配置保存成功')
                      } catch (e: any) {
                        message.error(`保存失败: ${e.message || '未知错误'}`)
                      }
                    }}
                  >
                    保存配置
                  </Button>
                </div>
              </Spin>
            )}

            <Form.Item name="max_time" label="最大仿真时间 (ns)" rules={[{ required: true }]}>
              <InputNumber min={1000} max={100000} step={1000} style={{ width: '100%' }} />
            </Form.Item>

            <Form.Item name="experiment_name" label="实验名称">
              <Input placeholder="可选，用于数据库记录" />
            </Form.Item>

            <Form.Item name="save_to_db" label="保存到数据库" valuePropName="checked">
              <Switch />
            </Form.Item>

            <Form.Item name="result_granularity" label="结果粒度">
              <Radio.Group>
                <Radio value="per_file">每文件一条</Radio>
                <Radio value="per_batch">每批次一条</Radio>
              </Radio.Group>
            </Form.Item>

            <Form.Item style={{ marginBottom: 0, marginTop: 8 }}>
              <Space size="middle">
                <Button
                  type="primary"
                  htmlType="submit"
                  icon={<RocketOutlined />}
                  loading={loading}
                  disabled={currentTask?.status === 'running'}
                  size="large"
                  style={{
                    height: 44,
                    paddingLeft: 24,
                    paddingRight: 24,
                    background: currentTask?.status === 'running' ? undefined : `linear-gradient(135deg, ${primaryColor} 0%, #4096ff 100%)`,
                  }}
                >
                  开始仿真
                </Button>
                {currentTask?.status === 'running' && (
                  <Button icon={<StopOutlined />} onClick={handleCancel} danger size="large" style={{ height: 44 }}>
                    取消任务
                  </Button>
                )}
              </Space>
            </Form.Item>
          </Form>
        </Card>

        {/* 当前任务状态卡片 */}
        {currentTask && (
          <Card
            title={
              <Space>
                <ThunderboltOutlined style={{ color: currentTask.status === 'running' ? warningColor : successColor }} />
                <span>任务状态</span>
                {getStatusTag(currentTask.status)}
              </Space>
            }
            style={{ marginBottom: 24 }}
          >
            <Row gutter={[24, 16]}>
              <Col span={8}>
                <Statistic
                  title="任务进度"
                  value={currentTask.progress}
                  suffix="%"
                  valueStyle={{ color: currentTask.status === 'failed' ? errorColor : primaryColor }}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="已处理文件"
                  value={currentTask.current_file ? Math.ceil((currentTask.progress / 100) * selectedFiles.length) : 0}
                  suffix={`/ ${selectedFiles.length}`}
                  valueStyle={{ color: primaryColor }}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="运行时间"
                  value={getElapsedTime()}
                  valueStyle={{ color: primaryColor }}
                />
              </Col>
            </Row>

            <Divider style={{ margin: '16px 0' }} />

            <Progress
              percent={currentTask.progress}
              status={currentTask.status === 'failed' ? 'exception' : currentTask.status === 'completed' ? 'success' : 'active'}
              strokeColor={currentTask.status === 'running' ? { from: primaryColor, to: '#4096ff' } : undefined}
              style={{ marginBottom: 12 }}
            />

            <Space direction="vertical" style={{ width: '100%' }} size={8}>
              <div>
                <Text type="secondary">任务ID: </Text>
                <Text code copyable={{ text: currentTask.task_id }}>{currentTask.task_id}</Text>
              </div>
              {currentTask.current_file && (
                <div>
                  <Text type="secondary">当前文件: </Text>
                  <Text>{currentTask.current_file}</Text>
                </div>
              )}
              {currentTask.message && (
                <div>
                  <Text type="secondary">状态信息: </Text>
                  <Text>{currentTask.message}</Text>
                </div>
              )}
              {currentTask.error && (
                <Alert type="error" message={currentTask.error} showIcon style={{ marginTop: 8 }} />
              )}
            </Space>
          </Card>
        )}
        </Col>

        {/* 右侧：文件选择 */}
        <Col xs={24} lg={12}>
          <Card
            title={
              <Space>
                <FileTextOutlined style={{ color: primaryColor }} />
                <span>流量文件</span>
                <Tag color={selectedFiles.length > 0 ? 'blue' : 'default'}>{selectedFiles.length} 个已选</Tag>
              </Space>
            }
            extra={
              <Button icon={<ReloadOutlined />} onClick={() => loadTrafficFiles(currentPath)} size="small">
                刷新
              </Button>
            }
            style={{ marginBottom: 24 }}
          >
            <Breadcrumb style={{ marginBottom: 12 }}>
              <Breadcrumb.Item>
                <a onClick={() => loadTrafficFiles('')}>traffic</a>
              </Breadcrumb.Item>
              {currentPath.split('/').filter(Boolean).map((part, index, arr) => (
                <Breadcrumb.Item key={index}>
                  <a onClick={() => loadTrafficFiles(arr.slice(0, index + 1).join('/'))}>
                    {part}
                  </a>
                </Breadcrumb.Item>
              ))}
            </Breadcrumb>

            <Spin spinning={loadingFiles}>
              {/* 目录列表 */}
              {directories.length > 0 && (
                <div style={{ marginBottom: 12 }}>
                  <Space wrap>
                    {currentPath && (
                      <Button size="small" onClick={navigateBack}>
                        ..
                      </Button>
                    )}
                    {directories.map((dir) => (
                      <Button
                        key={dir}
                        size="small"
                        icon={<FolderOpenOutlined />}
                        onClick={() => navigateToDir(dir)}
                      >
                        {dir}
                      </Button>
                    ))}
                  </Space>
                </div>
              )}

              {/* 文件列表 */}
              <Table
                dataSource={trafficFiles}
                rowKey="name"
                size="small"
                pagination={{ pageSize: 5 }}
                rowSelection={{
                  selectedRowKeys: selectedFiles,
                  onChange: (keys) => setSelectedFiles(keys as string[]),
                }}
                columns={[
                  { title: '文件名', dataIndex: 'name', ellipsis: true },
                  {
                    title: '大小',
                    dataIndex: 'size',
                    width: 80,
                    render: (size: number) => `${(size / 1024).toFixed(1)} KB`,
                  },
                ]}
                locale={{ emptyText: <Empty description="无流量文件" image={Empty.PRESENTED_IMAGE_SIMPLE} /> }}
              />
            </Spin>
          </Card>
        </Col>
      </Row>

      {/* 历史任务 */}
      <Card
        title={
          <Space>
            <HistoryOutlined style={{ color: primaryColor }} />
            <span>历史任务</span>
          </Space>
        }
        extra={
          <Button icon={<ReloadOutlined />} onClick={loadHistory} size="small">
            刷新
          </Button>
        }
      >
        <Table
          dataSource={taskHistory}
          rowKey="task_id"
          loading={loadingHistory}
          pagination={{ pageSize: 5, showSizeChanger: false }}
          columns={[
            {
              title: '任务ID',
              dataIndex: 'task_id',
              width: 120,
              render: (id: string) => <Text code style={{ fontSize: 12 }}>{id}</Text>
            },
            {
              title: '模式',
              dataIndex: 'mode',
              width: 100,
              render: (mode: string) => (
                <Tag color={mode === 'kcin' ? 'blue' : 'purple'}>{mode.toUpperCase()}</Tag>
              ),
            },
            {
              title: '拓扑',
              dataIndex: 'topology',
              width: 80,
              render: (topo: string) => <Tag>{topo}</Tag>
            },
            {
              title: '状态',
              dataIndex: 'status',
              width: 110,
              render: getStatusTag
            },
            {
              title: '消息',
              dataIndex: 'message',
              ellipsis: true,
              render: (msg: string) => <Text type="secondary">{msg || '-'}</Text>
            },
            {
              title: '创建时间',
              dataIndex: 'created_at',
              width: 170,
              render: (time: string) => (
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {new Date(time).toLocaleString()}
                </Text>
              ),
            },
          ]}
          locale={{ emptyText: <Empty description="暂无历史任务" image={Empty.PRESENTED_IMAGE_SIMPLE} /> }}
        />
      </Card>
    </div>
  )
}

export default Simulation
