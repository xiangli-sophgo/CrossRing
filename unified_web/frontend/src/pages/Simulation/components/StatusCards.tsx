/**
 * 状态卡片组件集合 - TaskStatusCard
 */
import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Statistic, Progress, Space, Typography, Divider, Alert, Tag, Button, Modal, List } from 'antd'
import {
  ThunderboltOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ClockCircleOutlined,
  SyncOutlined,
  StopOutlined,
  CloseOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons'
import { primaryColor, successColor, warningColor, errorColor } from '@/theme/colors'
import type { TaskStatus, TaskProgress } from '@/api/simulation'
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
  const [showProgressModal, setShowProgressModal] = useState(false)

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

  // 处理进度条点击
  const handleProgressClick = () => {
    if (currentTask.sim_details?.is_parallel && currentTask.sim_details?.tasks_progress) {
      setShowProgressModal(true)
    }
  }

  // 计算任务统计
  const getTaskStats = () => {
    if (!currentTask.sim_details?.tasks_progress) {
      return { running: 0, completed: 0, pending: 0 }
    }
    const tasks = currentTask.sim_details.tasks_progress
    return {
      running: tasks.filter(t => t.status === 'running').length,
      completed: tasks.filter(t => t.status === 'completed').length,
      pending: tasks.filter(t => t.status === 'pending').length,
    }
  }

  const taskName = currentTask.experiment_name || `任务 ${currentTask.task_id.slice(0, 8)}`
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
          {currentTask.progress === 100 &&
           currentTask.status === 'running' &&
           !currentTask.sim_details?.is_parallel ? (
            <Statistic
              title="任务进度"
              value="结果处理中"
              prefix={<SyncOutlined spin />}
              valueStyle={{ color: primaryColor, fontSize: 24 }}
            />
          ) : (
            <Statistic
              title="任务进度"
              value={currentTask.progress}
              suffix="%"
              valueStyle={{ color: currentTask.status === 'failed' ? errorColor : primaryColor }}
            />
          )}
        </Col>
        {currentTask.sim_details?.is_parallel && (
          <Col span={8}>
            <Statistic
              title="子任务"
              value={currentTask.sim_details?.file_index || 0}
              suffix={`/ ${currentTask.sim_details?.total_files || 0}`}
              valueStyle={{ color: primaryColor }}
            />
          </Col>
        )}
        <Col span={8}>
          <Statistic
            title="运行时间"
            value={elapsedTime}
            valueStyle={{ color: primaryColor }}
          />
        </Col>
      </Row>

      <Divider style={{ margin: '16px 0' }} />

      {/* 进度条（并行任务可点击查看详情） */}
      <div style={{ position: 'relative' }}>
        <div
          style={{
            cursor: currentTask.sim_details?.is_parallel && currentTask.sim_details?.tasks_progress ? 'pointer' : 'default'
          }}
          onClick={handleProgressClick}
        >
          <Progress
            percent={currentTask.progress}
            status={currentTask.status === 'failed' ? 'exception' : currentTask.status === 'completed' ? 'success' : 'active'}
            strokeColor={currentTask.status === 'running' ? { from: primaryColor, to: '#4096ff' } : undefined}
            style={{ marginBottom: 12 }}
          />
        </div>
      </div>

      {/* 串行任务显示当前文件 */}
      {!currentTask.sim_details?.is_parallel &&
       currentTask.sim_details?.current_file &&
       !currentTask.sim_details?.processing_stage && (
        <div style={{ marginBottom: 12 }}>
          <Text type="secondary">当前文件: </Text>
          <Text strong>{currentTask.sim_details.current_file}</Text>
        </div>
      )}

      {currentTask.sim_details && currentTask.status === 'running' &&
       !currentTask.sim_details.is_parallel && (
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
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <Text type="secondary">任务ID: </Text>
            <Text code copyable={{ text: currentTask.task_id }}>{currentTask.task_id}</Text>
          </div>
          {currentTask.sim_details?.is_parallel && currentTask.sim_details?.tasks_progress && (
            <Text type="secondary" style={{ fontSize: 12 }}>
              <InfoCircleOutlined /> 点击进度条查看各任务详情
            </Text>
          )}
        </div>
        {currentTask.error && (
          <Alert type="error" message={currentTask.error} showIcon style={{ marginTop: 8 }} />
        )}
      </Space>

      {/* 并行任务详情 Modal */}
      <Modal
        title="并行任务进度详情"
        open={showProgressModal}
        onCancel={() => setShowProgressModal(false)}
        footer={null}
        width={600}
      >
        {currentTask.sim_details?.tasks_progress && (() => {
          const stats = getTaskStats()
          const runningTasks = currentTask.sim_details.tasks_progress.filter(t => t.status === 'running')

          return (
            <>
              {/* 统计信息 */}
              <div style={{ marginBottom: 16, padding: 12, background: '#f5f5f5', borderRadius: 4 }}>
                <Row gutter={16}>
                  <Col span={8}>
                    <Statistic
                      title="运行中"
                      value={stats.running}
                      valueStyle={{ color: primaryColor, fontSize: 20 }}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="已完成"
                      value={stats.completed}
                      valueStyle={{ color: successColor, fontSize: 20 }}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="等待中"
                      value={stats.pending}
                      valueStyle={{ color: warningColor, fontSize: 20 }}
                    />
                  </Col>
                </Row>
              </div>

              {/* 运行中的任务列表 */}
              {runningTasks.length > 0 ? (
                <List
                  dataSource={runningTasks}
                  renderItem={(task) => (
                    <List.Item style={{ padding: '12px 0', borderBottom: '1px solid #f0f0f0' }}>
                      <div style={{ width: '100%' }}>
                        <div style={{ marginBottom: 8 }}>
                          <Text strong style={{ color: primaryColor }}>
                            任务 {task.task_index + 1}:
                          </Text>{' '}
                          <Text>{task.traffic_file}</Text>
                        </div>
                        <Progress
                          percent={task.progress}
                          status="active"
                          strokeColor={{ from: primaryColor, to: '#4096ff' }}
                        />
                        <div style={{ marginTop: 4, fontSize: 12, color: '#999' }}>
                          仿真时间: {task.current_time} / {task.max_time} ns
                        </div>
                      </div>
                    </List.Item>
                  )}
                />
              ) : (
                <div style={{ textAlign: 'center', padding: 20, color: '#999' }}>
                  暂无运行中的任务
                </div>
              )}
            </>
          )
        })()}
      </Modal>
    </Card>
  )
}
