/**
 * 状态卡片组件集合 - TaskStatusCard
 */
import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Statistic, Progress, Space, Typography, Divider, Alert, Tag, Button } from 'antd'
import {
  ThunderboltOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ClockCircleOutlined,
  SyncOutlined,
  StopOutlined,
  CloseOutlined,
} from '@ant-design/icons'
import { primaryColor, successColor, warningColor, errorColor } from '@/theme/colors'
import type { TaskStatus } from '@/api/simulation'
import { getElapsedTime } from '../helpers'

const { Text } = Typography

// ============== TaskStatusCard ==============

interface TaskStatusCardProps {
  currentTask: TaskStatus
  startTime: number | null
  onCancel?: () => void
  onClose?: () => void
}

// 获取状态标签
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

export const TaskStatusCard: React.FC<TaskStatusCardProps> = ({
  currentTask,
  startTime,
  onCancel,
  onClose,
}) => {
  const [elapsedTime, setElapsedTime] = useState(getElapsedTime(startTime))

  // 每秒更新运行时间
  useEffect(() => {
    if (currentTask.status !== 'running') {
      setElapsedTime(getElapsedTime(startTime))
      return
    }

    const timer = setInterval(() => {
      setElapsedTime(getElapsedTime(startTime))
    }, 1000)

    return () => clearInterval(timer)
  }, [startTime, currentTask.status])

  const taskName = currentTask.experiment_name || currentTask.task_id.slice(0, 8)
  return (
    <Card
      title={
        <Space>
          <ThunderboltOutlined style={{ color: currentTask.status === 'running' ? warningColor : successColor }} />
          <span>{taskName}</span>
          {getStatusTag(currentTask.status)}
        </Space>
      }
      extra={
        <Space>
          {onCancel && (
            <Button
              icon={<StopOutlined />}
              danger
              size="small"
              onClick={onCancel}
            >
              取消
            </Button>
          )}
          {onClose && (
            <Button
              icon={<CloseOutlined />}
              size="small"
              onClick={onClose}
            >
              关闭
            </Button>
          )}
        </Space>
      }
      style={{ marginBottom: 16 }}
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
            title="文件进度"
            value={currentTask.sim_details?.file_index || 0}
            suffix={`/ ${currentTask.sim_details?.total_files || 0}`}
            valueStyle={{ color: primaryColor }}
          />
        </Col>
        <Col span={8}>
          <Statistic
            title="运行时间"
            value={elapsedTime}
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

      {currentTask.sim_details?.current_file && (
        <div style={{ marginBottom: 12 }}>
          <Text type="secondary">当前文件: </Text>
          <Text strong>{currentTask.sim_details.current_file}</Text>
        </div>
      )}

      {currentTask.sim_details && currentTask.status === 'running' &&
       !currentTask.sim_details.current_file?.includes('并行执行中') && (
        <Row gutter={[16, 16]}>
          <Col span={6}>
            <Statistic
              title="仿真时间"
              value={currentTask.sim_details.current_time}
              suffix={`/ ${currentTask.sim_details.max_time} ns`}
              valueStyle={{ fontSize: 16 }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="请求数"
              value={currentTask.sim_details.req_count}
              suffix={`/ ${currentTask.sim_details.total_req}`}
              valueStyle={{ fontSize: 16 }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="接收数据"
              value={currentTask.sim_details.recv_flits}
              suffix={`/ ${currentTask.sim_details.total_flits} flits`}
              valueStyle={{ fontSize: 16 }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="传输中"
              value={currentTask.sim_details.trans_flits}
              suffix="flits"
              valueStyle={{ fontSize: 16 }}
            />
          </Col>
        </Row>
      )}

      <Divider style={{ margin: '16px 0' }} />

      <Space direction="vertical" style={{ width: '100%' }} size={4}>
        <div>
          <Text type="secondary">任务ID: </Text>
          <Text code copyable={{ text: currentTask.task_id }}>{currentTask.task_id}</Text>
        </div>
        {currentTask.error && (
          <Alert type="error" message={currentTask.error} showIcon style={{ marginTop: 8 }} />
        )}
      </Space>
    </Card>
  )
}
