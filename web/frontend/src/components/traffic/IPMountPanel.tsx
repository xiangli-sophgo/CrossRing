import { useState, useEffect, useRef } from 'react'
import {
  Card,
  Form,
  Input,
  Button,
  Table,
  Space,
  message,
  Popconfirm,
  Tag,
  Row,
  Col,
  Upload
} from 'antd'
import {
  ApiOutlined,
  DeleteOutlined,
  ClearOutlined,
  DownloadOutlined,
  UploadOutlined
} from '@ant-design/icons'
import type { IPMount } from '../../types/ipMount'
import {
  mountIP,
  batchMountIP,
  getMounts,
  deleteMount,
  clearAllMounts
} from '../../api/ipMount'

interface IPMountPanelProps {
  topology: string
  onMountsChange?: () => void
}

const IPMountPanel: React.FC<IPMountPanelProps> = ({ topology, onMountsChange }) => {
  const [form] = Form.useForm()
  const [mounts, setMounts] = useState<IPMount[]>([])
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // 加载已挂载的IP
  const loadMounts = async () => {
    setRefreshing(true)
    try {
      const data = await getMounts(topology)
      setMounts(data.mounts)
    } catch (error) {
      message.error('加载IP挂载失败')
      console.error(error)
    } finally {
      setRefreshing(false)
    }
  }

  useEffect(() => {
    loadMounts()
  }, [topology])

  // 统一挂载处理 - 自动判断单个或批量模式
  const handleMount = async (values: any) => {
    setLoading(true)
    try {
      const nodeRange = values.node_range.trim()
      const ipType = values.ip_type.trim()

      // 判断是单个挂载还是批量挂载
      // 如果ip_type包含下划线和数字，认为是单个挂载
      // 如果不包含，认为是批量挂载的前缀
      if (ipType.includes('_') && /\d+$/.test(ipType)) {
        // 单个挂载模式：IP类型包含编号（如 gdma_0）
        // 解析节点范围
        const nodeIds: number[] = []
        const parts = nodeRange.split(',')
        for (const part of parts) {
          const trimmed = part.trim()
          if (trimmed.includes('-')) {
            const [start, end] = trimmed.split('-').map(x => parseInt(x.trim()))
            for (let i = start; i <= end; i++) {
              nodeIds.push(i)
            }
          } else {
            nodeIds.push(parseInt(trimmed))
          }
        }

        await mountIP({
          node_ids: [...new Set(nodeIds)], // 去重
          ip_type: ipType,
          topology
        })
      } else {
        // 批量挂载模式：IP类型是前缀（如 gdma）
        await batchMountIP({
          node_range: nodeRange,
          ip_type_prefix: ipType,
          topology
        })
      }

      message.success('IP挂载成功')
      form.resetFields()
      loadMounts()
      onMountsChange?.()
    } catch (error: any) {
      message.error(error.response?.data?.detail || 'IP挂载失败')
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  // 删除指定IP类型的所有挂载
  const handleDeleteByIPType = async (ipType: string) => {
    try {
      // 找出该IP类型的所有节点
      const nodesToDelete = mounts.filter(m => m.ip_type === ipType)

      // 逐个删除
      for (const mount of nodesToDelete) {
        await deleteMount(topology, mount.node_id)
      }

      message.success(`已删除 ${ipType} 的所有挂载`)
      loadMounts()
      onMountsChange?.()
    } catch (error) {
      message.error('删除失败')
      console.error(error)
    }
  }

  // 清空所有挂载
  const handleClearAll = async () => {
    try {
      await clearAllMounts(topology)
      message.success('已清空所有IP挂载')
      loadMounts()
      onMountsChange?.()
    } catch (error) {
      message.error('清空失败')
      console.error(error)
    }
  }

  // 导出配置
  const handleExport = () => {
    if (mounts.length === 0) {
      message.warning('当前没有IP挂载配置可导出')
      return
    }

    const config = {
      topology,
      mounts: mounts.map(m => ({
        node_id: m.node_id,
        ip_type: m.ip_type
      }))
    }

    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `ip_mounts_${topology}_${new Date().getTime()}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    message.success('配置已导出')
  }

  // 导入配置
  const handleImport = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = async (e) => {
      try {
        const config = JSON.parse(e.target?.result as string)

        if (!config.topology || !config.mounts) {
          throw new Error('配置文件格式错误')
        }

        if (config.topology !== topology) {
          message.warning(`配置文件的拓扑类型 (${config.topology}) 与当前拓扑 (${topology}) 不匹配`)
          return
        }

        // 清空现有配置
        await clearAllMounts(topology)

        // 逐个挂载
        for (const mount of config.mounts) {
          await mountIP({
            node_ids: [mount.node_id],
            ip_type: mount.ip_type,
            topology
          })
        }

        message.success(`成功导入 ${config.mounts.length} 个IP挂载`)
        loadMounts()
        onMountsChange?.()
      } catch (error: any) {
        message.error(`导入失败: ${error.message}`)
        console.error(error)
      }
    }
    reader.readAsText(file)

    // 重置文件输入
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  // 按IP类型分组挂载信息
  const groupedMounts = mounts.reduce((acc, mount) => {
    if (!acc[mount.ip_type]) {
      acc[mount.ip_type] = []
    }
    acc[mount.ip_type].push(mount.node_id)
    return acc
  }, {} as Record<string, number[]>)

  const groupedData = Object.entries(groupedMounts).map(([ipType, nodeIds]) => ({
    ip_type: ipType,
    node_ids: nodeIds.sort((a, b) => a - b),
  }))

  const columns = [
    {
      title: 'IP类型',
      dataIndex: 'ip_type',
      key: 'ip_type',
      width: 120,
      render: (type: string) => <Tag color="green">{type}</Tag>
    },
    {
      title: '挂载节点',
      dataIndex: 'node_ids',
      key: 'node_ids',
      render: (nodeIds: number[]) => (
        <Space size={[4, 4]} wrap>
          {nodeIds.map(id => (
            <Tag key={id} color="blue">{id}</Tag>
          ))}
        </Space>
      )
    },
    {
      title: '操作',
      key: 'action',
      width: 100,
      render: (record: any) => (
        <Popconfirm
          title={`确认删除 ${record.ip_type} 的所有挂载？`}
          onConfirm={() => handleDeleteByIPType(record.ip_type)}
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
          <ApiOutlined />
          <span>IP节点挂载</span>
          <Tag color="purple">{mounts.length} 个挂载</Tag>
        </Space>
      }
      extra={
        <Space>
          <Button size="small" onClick={loadMounts} loading={refreshing}>
            刷新
          </Button>
          <Button
            size="small"
            icon={<DownloadOutlined />}
            onClick={handleExport}
            disabled={mounts.length === 0}
          >
            导出
          </Button>
          <Button
            size="small"
            icon={<UploadOutlined />}
            onClick={() => fileInputRef.current?.click()}
          >
            导入
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            style={{ display: 'none' }}
            onChange={handleImport}
          />
          <Popconfirm
            title="确认清空所有IP挂载？"
            onConfirm={handleClearAll}
            okText="确认"
            cancelText="取消"
          >
            <Button size="small" danger icon={<ClearOutlined />}>
              清空
            </Button>
          </Popconfirm>
        </Space>
      }
    >
      <Card size="small" style={{ marginBottom: 16 }}>
        <Form form={form} layout="inline" onFinish={handleMount}>
          <Form.Item
            name="node_range"
            rules={[{ required: true, message: '请输入节点范围' }]}
            style={{ width: '35%' }}
            tooltip="支持: 0-3 (范围), 1,3,5 (逗号), 0-3,5,7-9 (混合)"
          >
            <Input placeholder="节点范围 (如: 0-3,5,7-9)" />
          </Form.Item>
          <Form.Item
            name="ip_type"
            rules={[{ required: true, message: '请输入IP类型' }]}
            style={{ width: '35%' }}
            tooltip="单个挂载输入完整类型(如gdma_0)，批量挂载输入前缀(如gdma)"
          >
            <Input placeholder="IP类型/前缀 (如: gdma_0 或 gdma)" />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={loading}>
              挂载IP
            </Button>
          </Form.Item>
        </Form>
      </Card>

      <Table
        columns={columns}
        dataSource={groupedData}
        rowKey="ip_type"
        pagination={false}
        loading={refreshing}
        size="small"
      />
    </Card>
  )
}

export default IPMountPanel
