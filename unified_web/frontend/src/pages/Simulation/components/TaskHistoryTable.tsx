/**
 * 历史任务表格组件
 */
import React from 'react'
import { Card, Table, Space, Button, Tag, Typography, Empty } from 'antd'
import {
  HistoryOutlined,
  ReloadOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ClockCircleOutlined,
  SyncOutlined,
  StopOutlined,
} from '@ant-design/icons'
import { primaryColor } from '@/theme/colors'
import type { GroupedTask } from '../types'

const { Text } = Typography

interface TaskHistoryTableProps {
  groupedTaskHistory: GroupedTask[]
  loading: boolean
  onRefresh: () => void
  onViewResult: (experimentId: number) => void
  onDelete: (tasks: GroupedTask) => Promise<void>
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

export const TaskHistoryTable: React.FC<TaskHistoryTableProps> = ({
  groupedTaskHistory,
  loading,
  onRefresh,
  onViewResult,
  onDelete,
}) => {
  return (
    <Card
      title={
        <Space>
          <HistoryOutlined style={{ color: primaryColor }} />
          <span>历史任务</span>
        </Space>
      }
      extra={
        <Button icon={<ReloadOutlined />} onClick={onRefresh} size="small">
          刷新
        </Button>
      }
    >
      <Table
        dataSource={groupedTaskHistory}
        rowKey="key"
        loading={loading}
        pagination={{ pageSize: 5, showSizeChanger: false }}
        columns={[
          {
            title: '实验名称',
            dataIndex: 'experiment_name',
            width: 200,
            render: (name: string, record: GroupedTask) => (
              <Space direction="vertical" size={0}>
                <Text strong>{name || '未命名实验'}</Text>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  <Tag color={record.mode === 'kcin' ? 'blue' : 'purple'} style={{ marginRight: 4 }}>
                    {record.mode.toUpperCase()}
                  </Tag>
                  {record.topology}
                  {record.total_count > 1 && (
                    <Tag color="cyan" style={{ marginLeft: 4 }}>
                      参数遍历
                    </Tag>
                  )}
                </Text>
              </Space>
            ),
          },
          {
            title: '任务数',
            width: 100,
            render: (_: any, record: GroupedTask) => (
              <Text>
                {record.completed_count}/{record.total_count}
                {record.failed_count > 0 && (
                  <Text type="danger" style={{ marginLeft: 4 }}>
                    ({record.failed_count}失败)
                  </Text>
                )}
              </Text>
            ),
          },
          {
            title: '状态',
            dataIndex: 'status',
            width: 100,
            render: getStatusTag
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
          {
            title: '操作',
            width: 150,
            render: (_: any, record: GroupedTask) => (
              <Space>
                {record.status === 'completed' && record.experiment_id && (
                  <Button
                    type="link"
                    size="small"
                    onClick={() => onViewResult(record.experiment_id!)}
                  >
                    查看结果
                  </Button>
                )}
                <Button
                  type="link"
                  size="small"
                  danger
                  onClick={() => onDelete(record)}
                >
                  删除
                </Button>
              </Space>
            ),
          },
        ]}
        locale={{ emptyText: <Empty description="暂无历史任务" image={Empty.PRESENTED_IMAGE_SIMPLE} /> }}
      />
    </Card>
  )
}
