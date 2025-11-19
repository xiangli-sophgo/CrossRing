import { useState, useEffect } from 'react'
import {
  Card,
  Form,
  Input,
  InputNumber,
  Button,
  Table,
  Space,
  message,
  Popconfirm,
  Tag,
  Select,
  Row,
  Col,
  Divider
} from 'antd'
import {
  SettingOutlined,
  DeleteOutlined,
  ClearOutlined,
  ThunderboltOutlined
} from '@ant-design/icons'
import type { TrafficConfig } from '../../types/trafficConfig'
import {
  createTrafficConfig,
  getTrafficConfigs,
  deleteTrafficConfig,
  clearAllConfigs,
  generateTraffic,
  downloadTrafficFile
} from '../../api/trafficConfig'

const { Option } = Select

interface TrafficConfigPanelProps {
  topology: string
  mode: 'noc' | 'd2d'
}

const TrafficConfigPanel: React.FC<TrafficConfigPanelProps> = ({ topology, mode }) => {
  const [form] = Form.useForm()
  const [configs, setConfigs] = useState<TrafficConfig[]>([])
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [generating, setGenerating] = useState(false)

  // 加载流量配置
  const loadConfigs = async () => {
    setRefreshing(true)
    try {
      const data = await getTrafficConfigs(topology, mode)
      setConfigs(data.configs)
    } catch (error) {
      message.error('加载流量配置失败')
      console.error(error)
    } finally {
      setRefreshing(false)
    }
  }

  useEffect(() => {
    loadConfigs()
  }, [topology, mode])

  // 创建配置
  const handleCreate = async (values: any) => {
    setLoading(true)
    try {
      await createTrafficConfig({
        ...values,
        topology,
        mode
      })
      message.success('流量配置创建成功')
      form.resetFields()
      loadConfigs()
    } catch (error: any) {
      message.error(error.response?.data?.detail || '创建失败')
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  // 删除配置
  const handleDelete = async (configId: string) => {
    try {
      await deleteTrafficConfig(topology, mode, configId)
      message.success('删除成功')
      loadConfigs()
    } catch (error) {
      message.error('删除失败')
      console.error(error)
    }
  }

  // 清空所有配置
  const handleClearAll = async () => {
    try {
      await clearAllConfigs(topology, mode)
      message.success('已清空所有流量配置')
      loadConfigs()
    } catch (error) {
      message.error('清空失败')
      console.error(error)
    }
  }

  // 生成流量
  const handleGenerate = async () => {
    if (configs.length === 0) {
      message.warning('请先添加流量配置')
      return
    }

    setGenerating(true)
    try {
      const result = await generateTraffic({
        topology,
        mode,
        split_by_source: false,
        random_seed: 42
      })

      message.success(
        `流量生成成功！共 ${result.total_lines} 行，耗时 ${result.generation_time_ms.toFixed(0)}ms`
      )

      // 自动下载文件
      if (result.file_path) {
        const filename = result.file_path.split('/').pop()
        if (filename) {
          downloadTrafficFile(filename)
        }
      }
    } catch (error: any) {
      message.error(error.response?.data?.detail || '流量生成失败')
      console.error(error)
    } finally {
      setGenerating(false)
    }
  }

  const columns = [
    {
      title: '源IP',
      dataIndex: 'source_ip',
      key: 'source_ip',
      width: 120,
      render: (ip: string) => <Tag color="blue">{ip}</Tag>
    },
    {
      title: '目标IP',
      dataIndex: 'target_ip',
      key: 'target_ip',
      width: 120,
      render: (ip: string) => <Tag color="green">{ip}</Tag>
    },
    {
      title: '速度',
      dataIndex: 'speed_gbps',
      key: 'speed_gbps',
      width: 100,
      render: (speed: number) => `${speed} GB/s`
    },
    {
      title: 'Burst',
      dataIndex: 'burst_length',
      key: 'burst_length',
      width: 80
    },
    {
      title: '类型',
      dataIndex: 'request_type',
      key: 'request_type',
      width: 80,
      render: (type: string) => (
        <Tag color={type === 'R' ? 'cyan' : 'orange'}>{type}</Tag>
      )
    },
    {
      title: '结束时间',
      dataIndex: 'end_time_ns',
      key: 'end_time_ns',
      width: 120,
      render: (time: number) => `${time} ns`
    },
    {
      title: '操作',
      key: 'action',
      width: 100,
      render: (record: TrafficConfig) => (
        <Popconfirm
          title="确认删除此配置？"
          onConfirm={() => handleDelete(record.id)}
          okText="确认"
          cancelText="取消"
        >
          <Button type="link" danger icon={<DeleteOutlined />} size="small">
            删除
          </Button>
        </Popconfirm>
      )
    }
  ]

  return (
    <Card
      title={
        <Space>
          <SettingOutlined />
          <span>流量配置</span>
          <Tag color="purple">{configs.length} 个配置</Tag>
          <Tag color={mode === 'noc' ? 'blue' : 'orange'}>{mode.toUpperCase()} 模式</Tag>
        </Space>
      }
      extra={
        <Space>
          <Button size="small" onClick={loadConfigs} loading={refreshing}>
            刷新
          </Button>
          <Popconfirm
            title="确认清空所有流量配置？"
            onConfirm={handleClearAll}
            okText="确认"
            cancelText="取消"
          >
            <Button size="small" danger icon={<ClearOutlined />}>
              清空
            </Button>
          </Popconfirm>
          <Button
            type="primary"
            icon={<ThunderboltOutlined />}
            onClick={handleGenerate}
            loading={generating}
            disabled={configs.length === 0}
          >
            生成流量
          </Button>
        </Space>
      }
    >
      <Card title="添加流量配置" size="small" style={{ marginBottom: 16 }}>
        <Form form={form} layout="inline" onFinish={handleCreate}>
          <Row gutter={16} style={{ width: '100%' }}>
            <Col span={4}>
              <Form.Item
                name="source_ip"
                rules={[{ required: true, message: '请输入源IP' }]}
              >
                <Input placeholder="源IP (如 gdma_0)" />
              </Form.Item>
            </Col>
            <Col span={4}>
              <Form.Item
                name="target_ip"
                rules={[{ required: true, message: '请输入目标IP' }]}
              >
                <Input placeholder="目标IP (如 ddr_0)" />
              </Form.Item>
            </Col>
            <Col span={3}>
              <Form.Item
                name="speed_gbps"
                rules={[{ required: true, message: '请输入速度' }]}
                initialValue={128}
              >
                <InputNumber
                  placeholder="速度 (GB/s)"
                  min={0.1}
                  max={128}
                  step={0.1}
                  style={{ width: '100%' }}
                />
              </Form.Item>
            </Col>
            <Col span={3}>
              <Form.Item
                name="burst_length"
                rules={[{ required: true, message: '请输入Burst' }]}
                initialValue={4}
              >
                <InputNumber
                  placeholder="Burst长度"
                  min={1}
                  max={16}
                  style={{ width: '100%' }}
                />
              </Form.Item>
            </Col>
            <Col span={3}>
              <Form.Item
                name="request_type"
                rules={[{ required: true, message: '请选择类型' }]}
                initialValue="R"
              >
                <Select placeholder="请求类型">
                  <Option value="R">读 (R)</Option>
                  <Option value="W">写 (W)</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={4}>
              <Form.Item
                name="end_time_ns"
                rules={[{ required: true, message: '请输入结束时间' }]}
                initialValue={6000}
              >
                <InputNumber
                  placeholder="结束时间 (ns)"
                  min={100}
                  max={100000}
                  style={{ width: '100%' }}
                />
              </Form.Item>
            </Col>
            <Col span={3}>
              <Form.Item>
                <Button type="primary" htmlType="submit" loading={loading} block>
                  添加
                </Button>
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Card>

      <Table
        columns={columns}
        dataSource={configs}
        rowKey="id"
        pagination={{ pageSize: 10 }}
        loading={refreshing}
        size="small"
      />
    </Card>
  )
}

export default TrafficConfigPanel
