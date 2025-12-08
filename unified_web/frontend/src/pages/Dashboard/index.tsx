import React, { useEffect, useState } from 'react'
import { Card, Row, Col, Button, Space, Typography, List, Tag, Progress, Skeleton } from 'antd'
import {
  ThunderboltOutlined,
  ExperimentOutlined,
  FileTextOutlined,
  PlayCircleOutlined,
  HistoryOutlined,
  SettingOutlined,
  RocketOutlined,
  DatabaseOutlined,
  ArrowRightOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  SyncOutlined,
} from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import { primaryColor, successColor, warningColor, errorColor } from '@/theme/colors'

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

// 统计卡片组件
interface StatCardProps {
  title: string
  value: number
  icon: React.ReactNode
  color: string
  bgColor: string
  suffix?: string
  onClick?: () => void
}

const StatCard: React.FC<StatCardProps> = ({ title, value, icon, color, bgColor, suffix, onClick }) => (
  <Card
    className="stat-card"
    onClick={onClick}
    style={{ cursor: onClick ? 'pointer' : 'default' }}
    styles={{ body: { padding: 20 } }}
  >
    <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16 }}>
      <div
        style={{
          width: 56,
          height: 56,
          borderRadius: 12,
          backgroundColor: bgColor,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: color,
          fontSize: 24,
        }}
      >
        {icon}
      </div>
      <div style={{ flex: 1 }}>
        <Text type="secondary" style={{ fontSize: 14 }}>{title}</Text>
        <div style={{ marginTop: 4 }}>
          <span style={{ fontSize: 28, fontWeight: 600, color }}>{value}</span>
          {suffix && <span style={{ fontSize: 14, color: 'rgba(0,0,0,0.45)', marginLeft: 4 }}>{suffix}</span>}
        </div>
      </div>
    </div>
  </Card>
)

// 快速操作按钮组件
interface QuickActionProps {
  icon: React.ReactNode
  title: string
  description: string
  color: string
  onClick: () => void
}

const QuickAction: React.FC<QuickActionProps> = ({ icon, title, description, color, onClick }) => (
  <Card
    hoverable
    onClick={onClick}
    style={{ textAlign: 'center', borderRadius: 12 }}
    styles={{ body: { padding: '24px 16px' } }}
  >
    <div
      style={{
        width: 48,
        height: 48,
        borderRadius: 12,
        background: `linear-gradient(135deg, ${color} 0%, ${color}cc 100%)`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        margin: '0 auto 12px',
        color: '#fff',
        fontSize: 22,
      }}
    >
      {icon}
    </div>
    <Title level={5} style={{ marginBottom: 4 }}>{title}</Title>
    <Text type="secondary" style={{ fontSize: 12 }}>{description}</Text>
  </Card>
)

const Dashboard: React.FC = () => {
  const navigate = useNavigate()
  const [recentTasks, setRecentTasks] = useState<TaskSummary[]>([])
  const [stats, setStats] = useState({
    totalExperiments: 0,
    runningTasks: 0,
    completedToday: 0,
    totalResults: 0,
  })
  const [loading, setLoading] = useState(true)

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

      // 尝试获取实验数量
      let totalResults = 0
      try {
        const expRes = await axios.get('/api/experiments')
        totalResults = expRes.data?.length || 0
      } catch {
        // 忽略错误
      }

      setStats({
        totalExperiments: tasks.length,
        runningTasks: running,
        completedToday: completedToday,
        totalResults: totalResults,
      })
    } catch (error) {
      console.error('加载仪表盘数据失败:', error)
    } finally {
      setLoading(false)
    }
  }

  const getStatusConfig = (status: string) => {
    const statusMap: Record<string, { color: string; text: string; icon: React.ReactNode }> = {
      pending: { color: 'default', text: '等待中', icon: <ClockCircleOutlined /> },
      running: { color: 'processing', text: '运行中', icon: <SyncOutlined spin /> },
      completed: { color: 'success', text: '已完成', icon: <CheckCircleOutlined /> },
      failed: { color: 'error', text: '失败', icon: <ExperimentOutlined /> },
      cancelled: { color: 'warning', text: '已取消', icon: <ExperimentOutlined /> },
    }
    return statusMap[status] || { color: 'default', text: status, icon: null }
  }

  return (
    <div>
      {/* 欢迎区域 */}
      <div style={{ marginBottom: 24 }}>
        <Title level={3} style={{ marginBottom: 4 }}>
          欢迎使用仿真一体化平台
        </Title>
      </div>

      {/* 统计卡片 */}
      {loading ? (
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          {[1, 2, 3, 4].map(i => (
            <Col xs={24} sm={12} lg={6} key={i}>
              <Card><Skeleton active paragraph={{ rows: 1 }} /></Card>
            </Col>
          ))}
        </Row>
      ) : (
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={24} sm={12} lg={6}>
            <StatCard
              title="历史任务"
              value={stats.totalExperiments}
              icon={<HistoryOutlined />}
              color={primaryColor}
              bgColor="#e6f4ff"
              suffix="个"
            />
          </Col>
          <Col xs={24} sm={12} lg={6}>
            <StatCard
              title="正在运行"
              value={stats.runningTasks}
              icon={<ThunderboltOutlined />}
              color={stats.runningTasks > 0 ? warningColor : '#8c8c8c'}
              bgColor={stats.runningTasks > 0 ? '#fffbe6' : '#f5f5f5'}
              suffix="个"
              onClick={() => navigate('/simulation')}
            />
          </Col>
          <Col xs={24} sm={12} lg={6}>
            <StatCard
              title="今日完成"
              value={stats.completedToday}
              icon={<CheckCircleOutlined />}
              color={successColor}
              bgColor="#f6ffed"
              suffix="个"
            />
          </Col>
          <Col xs={24} sm={12} lg={6}>
            <StatCard
              title="实验记录"
              value={stats.totalResults}
              icon={<DatabaseOutlined />}
              color="#722ed1"
              bgColor="#f9f0ff"
              suffix="条"
              onClick={() => navigate('/experiments')}
            />
          </Col>
        </Row>
      )}

      {/* 快速操作 */}
      <Card
        title={
          <Space>
            <RocketOutlined style={{ color: primaryColor }} />
            <span>快速操作</span>
          </Space>
        }
        style={{ marginBottom: 24 }}
        styles={{ body: { padding: 16 } }}
      >
        <Row gutter={[16, 16]}>
          <Col xs={12} sm={6}>
            <QuickAction
              icon={<PlayCircleOutlined />}
              title="开始仿真"
              description="运行KCIN/DCIN仿真"
              color={primaryColor}
              onClick={() => navigate('/simulation')}
            />
          </Col>
          <Col xs={12} sm={6}>
            <QuickAction
              icon={<SettingOutlined />}
              title="流量配置"
              description="配置IP和数据流"
              color="#52c41a"
              onClick={() => navigate('/traffic')}
            />
          </Col>
          <Col xs={12} sm={6}>
            <QuickAction
              icon={<ExperimentOutlined />}
              title="结果管理"
              description="查看历史实验"
              color="#722ed1"
              onClick={() => navigate('/experiments')}
            />
          </Col>
          <Col xs={12} sm={6}>
            <QuickAction
              icon={<FileTextOutlined />}
              title="结果分析"
              description="性能数据分析"
              color="#fa8c16"
              onClick={() => navigate('/experiments')}
            />
          </Col>
        </Row>
      </Card>

      {/* 最近任务 */}
      <Card
        title={
          <Space>
            <HistoryOutlined style={{ color: primaryColor }} />
            <span>最近任务</span>
          </Space>
        }
        extra={
          <Button
            type="link"
            onClick={() => navigate('/simulation')}
            style={{ padding: 0 }}
          >
            查看全部 <ArrowRightOutlined />
          </Button>
        }
      >
        {loading ? (
          <Skeleton active paragraph={{ rows: 4 }} />
        ) : recentTasks.length === 0 ? (
          <div className="empty-state">
            <ExperimentOutlined style={{ fontSize: 48, color: '#d9d9d9', marginBottom: 16 }} />
            <Text type="secondary">暂无任务记录</Text>
            <Button
              type="primary"
              style={{ marginTop: 16 }}
              onClick={() => navigate('/simulation')}
            >
              创建第一个仿真任务
            </Button>
          </div>
        ) : (
          <List
            dataSource={recentTasks}
            renderItem={(task) => {
              const statusConfig = getStatusConfig(task.status)
              return (
                <List.Item
                  style={{
                    padding: '16px 0',
                    borderRadius: 8,
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                      <Tag color={task.mode === 'kcin' ? 'blue' : 'purple'}>
                        {task.mode.toUpperCase()}
                      </Tag>
                      <Text strong>拓扑 {task.topology}</Text>
                      <Tag icon={statusConfig.icon} color={statusConfig.color}>
                        {statusConfig.text}
                      </Tag>
                    </div>
                    {task.status === 'running' && (
                      <Progress
                        percent={task.progress}
                        size="small"
                        style={{ maxWidth: 300, marginBottom: 4 }}
                      />
                    )}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {task.message || '无描述'}
                      </Text>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {new Date(task.created_at).toLocaleString()}
                      </Text>
                    </div>
                  </div>
                </List.Item>
              )
            }}
          />
        )}
      </Card>
    </div>
  )
}

export default Dashboard
