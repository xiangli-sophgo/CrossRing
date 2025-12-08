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
} from 'antd'
import {
  PlayCircleOutlined,
  StopOutlined,
  FolderOpenOutlined,
  ReloadOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  LoadingOutlined,
} from '@ant-design/icons'
import {
  runSimulation,
  getTaskStatus,
  cancelTask,
  getTaskHistory,
  getConfigs,
  getTrafficFiles,
  type SimulationRequest,
  type TaskStatus,
  type TaskHistoryItem,
  type ConfigOption,
  type TrafficFileInfo,
} from '@/api/simulation'

const { Title, Text } = Typography
const { Option } = Select

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
      const request: SimulationRequest = {
        mode: values.mode,
        topology: values.topology,
        config_path: values.config_path,
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
      pending: { color: 'default', text: '等待中', icon: <LoadingOutlined /> },
      running: { color: 'processing', text: '运行中', icon: <LoadingOutlined spin /> },
      completed: { color: 'success', text: '已完成', icon: <CheckCircleOutlined /> },
      failed: { color: 'error', text: '失败', icon: <CloseCircleOutlined /> },
      cancelled: { color: 'warning', text: '已取消', icon: <StopOutlined /> },
    }
    const { color, text, icon } = statusMap[status] || { color: 'default', text: status, icon: null }
    return <Tag color={color} icon={icon}>{text}</Tag>
  }

  const mode = Form.useWatch('mode', form) || 'kcin'

  return (
    <div>
      <Title level={4}>仿真执行</Title>

      <div style={{ display: 'flex', gap: 24 }}>
        {/* 左侧：配置表单 */}
        <Card title="仿真配置" style={{ flex: 1 }}>
          <Form
            form={form}
            layout="vertical"
            initialValues={{
              mode: 'kcin',
              topology: '5x4',
              max_time: 6000,
              save_to_db: true,
              result_granularity: 'per_file',
            }}
            onFinish={handleSubmit}
          >
            <Form.Item name="mode" label="仿真模式" rules={[{ required: true }]}>
              <Radio.Group>
                <Radio.Button value="kcin">KCIN (单Die)</Radio.Button>
                <Radio.Button value="dcin">DCIN (多Die)</Radio.Button>
              </Radio.Group>
            </Form.Item>

            <Form.Item name="topology" label="拓扑类型" rules={[{ required: true }]}>
              <Select>
                <Option value="3x3">3x3</Option>
                <Option value="4x4">4x4</Option>
                <Option value="5x2">5x2</Option>
                <Option value="5x4">5x4</Option>
                <Option value="6x5">6x5</Option>
                <Option value="8x8">8x8</Option>
              </Select>
            </Form.Item>

            <Form.Item name="config_path" label="配置文件">
              <Select allowClear placeholder="使用默认配置">
                {(mode === 'kcin' ? configs.kcin : configs.dcin).map((c) => (
                  <Option key={c.path} value={c.path}>
                    {c.name}
                  </Option>
                ))}
              </Select>
            </Form.Item>

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

            <Form.Item>
              <Space>
                <Button
                  type="primary"
                  htmlType="submit"
                  icon={<PlayCircleOutlined />}
                  loading={loading}
                  disabled={currentTask?.status === 'running'}
                >
                  开始仿真
                </Button>
                {currentTask?.status === 'running' && (
                  <Button icon={<StopOutlined />} onClick={handleCancel} danger>
                    取消
                  </Button>
                )}
              </Space>
            </Form.Item>
          </Form>
        </Card>

        {/* 右侧：文件选择和状态 */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* 流量文件选择 */}
          <Card
            title="流量文件"
            extra={
              <Button icon={<ReloadOutlined />} onClick={() => loadTrafficFiles(currentPath)} size="small">
                刷新
              </Button>
            }
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
                locale={{ emptyText: <Empty description="无流量文件" /> }}
              />
              <Text type="secondary">已选择 {selectedFiles.length} 个文件</Text>
            </Spin>
          </Card>

          {/* 当前任务状态 */}
          {currentTask && (
            <Card title="当前任务">
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <Text strong>任务ID: </Text>
                  <Text code>{currentTask.task_id}</Text>
                  {getStatusTag(currentTask.status)}
                </div>
                <Progress percent={currentTask.progress} status={currentTask.status === 'failed' ? 'exception' : undefined} />
                <Text type="secondary">{currentTask.message}</Text>
                {currentTask.current_file && (
                  <Text>当前文件: {currentTask.current_file}</Text>
                )}
                {currentTask.error && (
                  <Alert type="error" message={currentTask.error} showIcon />
                )}
              </Space>
            </Card>
          )}
        </div>
      </div>

      {/* 历史任务 */}
      <Card title="历史任务" style={{ marginTop: 24 }}>
        <Table
          dataSource={taskHistory}
          rowKey="task_id"
          loading={loadingHistory}
          pagination={{ pageSize: 5 }}
          columns={[
            { title: '任务ID', dataIndex: 'task_id', width: 100, render: (id: string) => <Text code>{id}</Text> },
            {
              title: '模式',
              dataIndex: 'mode',
              width: 80,
              render: (mode: string) => (
                <Tag color={mode === 'kcin' ? 'blue' : 'purple'}>{mode.toUpperCase()}</Tag>
              ),
            },
            { title: '拓扑', dataIndex: 'topology', width: 80 },
            { title: '状态', dataIndex: 'status', width: 100, render: getStatusTag },
            { title: '消息', dataIndex: 'message', ellipsis: true },
            {
              title: '创建时间',
              dataIndex: 'created_at',
              width: 180,
              render: (time: string) => new Date(time).toLocaleString(),
            },
          ]}
          locale={{ emptyText: <Empty description="暂无历史任务" /> }}
        />
      </Card>
    </div>
  )
}

export default Simulation
