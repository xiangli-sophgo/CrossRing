import React, { useEffect, useState } from 'react'
import { Card, Row, Col, Statistic, Button, Space, Typography, List, Tag, message } from 'antd'
import {
  ThunderboltOutlined,
  ExperimentOutlined,
  FileTextOutlined,
  PlayCircleOutlined,
  HistoryOutlined,
} from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'

const { Title, Text } = Typography

interface TaskSummary {
  task_id: string
  mode: string
  topology: string
  status: string
  progress: number
  message: string
  created_at: string
  completed_at: string | null
}

const Dashboard: React.FC = () => {
  const navigate = useNavigate()
  const [recentTasks, setRecentTasks] = useState<TaskSummary[]>([])
  const [stats, setStats] = useState({
    totalExperiments: 0,
    runningTasks: 0,
    completedToday: 0,
  })
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    loadDashboardData()
  }, [])

  const loadDashboardData = async () => {
    setLoading(true)
    try {
      // 加载最近任务
      const tasksRes = await axios.get('/api/simulation/history?limit=5')
      setRecentTasks(tasksRes.data.tasks || [])

      // 计算统计数据
      const tasks = tasksRes.data.tasks || []
      const running = tasks.filter((t: TaskSummary) => t.status === 'running').length
      const today = new Date().toDateString()
      const completedToday = tasks.filter((t: TaskSummary) =>
        t.completed_at && new Date(t.completed_at).toDateString() === today
      ).length

      setStats({
        totalExperiments: tasks.length,
        runningTasks: running,
        completedToday: completedToday,
      })
    } catch (error) {
      console.error('加载仪表盘数据失败:', error)
    } finally {
      setLoading(false)
    }
  }

  const getStatusTag = (status: string) => {
    const statusMap: Record<string, { color: string; text: string }> = {
      pending: { color: 'default', text: '等待中' },
      running: { color: 'processing', text: '运行中' },
      completed: { color: 'success', text: '已完成' },
      failed: { color: 'error', text: '失败' },
      cancelled: { color: 'warning', text: '已取消' },
    }
    const { color, text } = statusMap[status] || { color: 'default', text: status }
    return <Tag color={color}>{text}</Tag>
  }

  return (
    <div>
      <Title level={4}>仪表盘</Title>
      <Text type="secondary" style={{ marginBottom: 24, display: 'block' }}>
        CrossRing 一体化仿真平台 - 快速开始
      </Text>

      {/* 统计卡片 */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Card>
            <Statistic
              title="历史任务"
              value={stats.totalExperiments}
              prefix={<ExperimentOutlined />}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="正在运行"
              value={stats.runningTasks}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: stats.runningTasks > 0 ? '#1890ff' : undefined }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="今日完成"
              value={stats.completedToday}
              prefix={<FileTextOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 快速操作 */}
      <Card title="快速操作" style={{ marginBottom: 24 }}>
        <Space size="large">
          <Button
            type="primary"
            size="large"
            icon={<PlayCircleOutlined />}
            onClick={() => navigate('/simulation')}
          >
            开始仿真
          </Button>
          <Button
            size="large"
            icon={<FileTextOutlined />}
            onClick={() => navigate('/traffic')}
          >
            配置流量
          </Button>
          <Button
            size="large"
            icon={<ExperimentOutlined />}
            onClick={() => navigate('/experiments')}
          >
            查看实验
          </Button>
        </Space>
      </Card>

      {/* 最近任务 */}
      <Card
        title={
          <Space>
            <HistoryOutlined />
            <span>最近任务</span>
          </Space>
        }
        extra={
          <Button type="link" onClick={() => navigate('/simulation')}>
            查看全部
          </Button>
        }
      >
        <List
          loading={loading}
          dataSource={recentTasks}
          locale={{ emptyText: '暂无任务记录' }}
          renderItem={(task) => (
            <List.Item
              actions={[
                getStatusTag(task.status),
                <Text type="secondary" key="time">
                  {new Date(task.created_at).toLocaleString()}
                </Text>,
              ]}
            >
              <List.Item.Meta
                title={
                  <Space>
                    <Tag color={task.mode === 'kcin' ? 'blue' : 'purple'}>
                      {task.mode.toUpperCase()}
                    </Tag>
                    <span>拓扑: {task.topology}</span>
                  </Space>
                }
                description={task.message || '无描述'}
              />
            </List.Item>
          )}
        />
      </Card>
    </div>
  )
}

export default Dashboard
