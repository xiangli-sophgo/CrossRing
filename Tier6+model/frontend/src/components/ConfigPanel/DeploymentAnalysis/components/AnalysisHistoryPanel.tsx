/**
 * 分析历史记录面板
 *
 * 功能：
 * - 表格形式展示历史记录
 * - 点击查看配置详情
 * - 支持导出 JSON/CSV
 * - 支持搜索和筛选
 */

import React, { useState, useMemo } from 'react'
import {
  Typography,
  Table,
  Button,
  Modal,
  Tag,
  Input,
  Space,
  Popconfirm,
  message,
  Descriptions,
  Tabs,
  Tooltip,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import {
  DeleteOutlined,
  DownloadOutlined,
  EyeOutlined,
  SearchOutlined,
  ClearOutlined,
  FileTextOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import {
  LLMModelConfig,
  InferenceConfig,
  HardwareConfig,
  ParallelismStrategy,
  PlanAnalysisResult,
} from '../../../../utils/llmDeployment/types'
import { colors } from '../ConfigSelectors'

const { Text, Title } = Typography

// ============================================
// 类型定义
// ============================================

export interface AnalysisHistoryItem {
  id: string
  timestamp: number
  modelName: string
  parallelism: ParallelismStrategy
  score: number
  ttft: number
  tpot: number
  throughput: number
  mfu: number
  mbu: number
  cost: number | null
  chips: number
  result: PlanAnalysisResult
  /** 自动搜索模式下的 Top-K 方案 */
  topKPlans?: PlanAnalysisResult[]
  /** 搜索模式: manual=手动, auto=自动搜索 */
  searchMode?: 'manual' | 'auto'
  modelConfig: LLMModelConfig
  inferenceConfig: InferenceConfig
  hardwareConfig: HardwareConfig
}

interface AnalysisHistoryPanelProps {
  history: AnalysisHistoryItem[]
  onLoad: (item: AnalysisHistoryItem) => void
  onDelete: (id: string) => void
  onClear: () => void
}

// ============================================
// 配置详情模态框
// ============================================

interface ConfigDetailModalProps {
  item: AnalysisHistoryItem | null
  visible: boolean
  onClose: () => void
  onLoad: (item: AnalysisHistoryItem) => void
}

const ConfigDetailModal: React.FC<ConfigDetailModalProps> = ({
  item,
  visible,
  onClose,
  onLoad,
}) => {
  if (!item) return null

  const { modelConfig, inferenceConfig, hardwareConfig, result, parallelism } = item

  const tabItems = [
    {
      key: 'overview',
      label: '概览',
      children: (
        <div>
          <Descriptions column={2} size="small" bordered>
            <Descriptions.Item label="模型" span={2}>
              <Text strong>{modelConfig.model_name}</Text>
            </Descriptions.Item>
            <Descriptions.Item label="参数量">
              {(modelConfig.hidden_size * modelConfig.num_layers * 12 / 1e9).toFixed(1)}B
            </Descriptions.Item>
            <Descriptions.Item label="层数">
              {modelConfig.num_layers}
            </Descriptions.Item>
            <Descriptions.Item label="并行策略" span={2}>
              <Space>
                <Tag color="blue">DP={parallelism.dp}</Tag>
                <Tag color="green">TP={parallelism.tp}</Tag>
                <Tag color="orange">PP={parallelism.pp}</Tag>
                {parallelism.ep > 1 && <Tag color="purple">EP={parallelism.ep}</Tag>}
              </Space>
            </Descriptions.Item>
            <Descriptions.Item label="综合评分">
              <Text strong style={{ color: colors.primary, fontSize: 18 }}>
                {item.score.toFixed(1)}
              </Text>
            </Descriptions.Item>
            <Descriptions.Item label="芯片数">
              {item.chips} 个
            </Descriptions.Item>
          </Descriptions>

          <div style={{ marginTop: 16 }}>
            <Title level={5} style={{ marginBottom: 12 }}>性能指标</Title>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
              <div style={{ padding: 12, background: '#f5f5f5', borderRadius: 8, textAlign: 'center' }}>
                <div style={{ fontSize: 11, color: '#666' }}>TTFT</div>
                <div style={{ fontSize: 18, fontWeight: 600, color: '#1890ff' }}>{item.ttft.toFixed(1)} ms</div>
              </div>
              <div style={{ padding: 12, background: '#f5f5f5', borderRadius: 8, textAlign: 'center' }}>
                <div style={{ fontSize: 11, color: '#666' }}>TPOT</div>
                <div style={{ fontSize: 18, fontWeight: 600, color: '#13c2c2' }}>{item.tpot.toFixed(2)} ms</div>
              </div>
              <div style={{ padding: 12, background: '#f5f5f5', borderRadius: 8, textAlign: 'center' }}>
                <div style={{ fontSize: 11, color: '#666' }}>吞吐量</div>
                <div style={{ fontSize: 18, fontWeight: 600, color: '#52c41a' }}>{item.throughput.toFixed(0)} tok/s</div>
              </div>
              <div style={{ padding: 12, background: '#f5f5f5', borderRadius: 8, textAlign: 'center' }}>
                <div style={{ fontSize: 11, color: '#666' }}>MFU</div>
                <div style={{ fontSize: 18, fontWeight: 600, color: '#faad14' }}>{(item.mfu * 100).toFixed(1)}%</div>
              </div>
              <div style={{ padding: 12, background: '#f5f5f5', borderRadius: 8, textAlign: 'center' }}>
                <div style={{ fontSize: 11, color: '#666' }}>MBU</div>
                <div style={{ fontSize: 18, fontWeight: 600, color: '#722ed1' }}>{(item.mbu * 100).toFixed(1)}%</div>
              </div>
              <div style={{ padding: 12, background: '#f5f5f5', borderRadius: 8, textAlign: 'center' }}>
                <div style={{ fontSize: 11, color: '#666' }}>成本</div>
                <div style={{ fontSize: 18, fontWeight: 600, color: '#fa541c' }}>
                  {item.cost !== null ? `$${item.cost.toFixed(4)}/M` : '-'}
                </div>
              </div>
            </div>
          </div>
        </div>
      ),
    },
    {
      key: 'model',
      label: '模型配置',
      children: (
        <Descriptions column={2} size="small" bordered>
          <Descriptions.Item label="模型名称" span={2}>
            {modelConfig.model_name}
          </Descriptions.Item>
          <Descriptions.Item label="模型类型">
            {modelConfig.model_type === 'moe' ? 'MoE' : 'Dense'}
          </Descriptions.Item>
          <Descriptions.Item label="数据类型">
            {modelConfig.dtype}
          </Descriptions.Item>
          <Descriptions.Item label="隐藏维度">
            {modelConfig.hidden_size}
          </Descriptions.Item>
          <Descriptions.Item label="中间维度">
            {modelConfig.intermediate_size}
          </Descriptions.Item>
          <Descriptions.Item label="层数">
            {modelConfig.num_layers}
          </Descriptions.Item>
          <Descriptions.Item label="注意力头数">
            {modelConfig.num_attention_heads}
          </Descriptions.Item>
          <Descriptions.Item label="KV头数">
            {modelConfig.num_kv_heads}
          </Descriptions.Item>
          <Descriptions.Item label="词表大小">
            {modelConfig.vocab_size.toLocaleString()}
          </Descriptions.Item>
          {modelConfig.model_type === 'moe' && modelConfig.moe_config && (
            <>
              <Descriptions.Item label="专家数">
                {modelConfig.moe_config.num_experts}
              </Descriptions.Item>
              <Descriptions.Item label="激活专家">
                {modelConfig.moe_config.num_experts_per_tok}
              </Descriptions.Item>
            </>
          )}
        </Descriptions>
      ),
    },
    {
      key: 'inference',
      label: '推理配置',
      children: (
        <Descriptions column={2} size="small" bordered>
          <Descriptions.Item label="批次大小">
            {inferenceConfig.batch_size}
          </Descriptions.Item>
          <Descriptions.Item label="输入长度">
            {inferenceConfig.input_seq_length}
          </Descriptions.Item>
          <Descriptions.Item label="输出长度">
            {inferenceConfig.output_seq_length}
          </Descriptions.Item>
          <Descriptions.Item label="最大序列长度">
            {inferenceConfig.max_seq_length}
          </Descriptions.Item>
          {inferenceConfig.num_micro_batches && (
            <Descriptions.Item label="Micro-batch数">
              {inferenceConfig.num_micro_batches}
            </Descriptions.Item>
          )}
        </Descriptions>
      ),
    },
    {
      key: 'hardware',
      label: '硬件配置',
      children: (
        <Descriptions column={2} size="small" bordered>
          <Descriptions.Item label="芯片类型" span={2}>
            {hardwareConfig.chip.chip_type}
          </Descriptions.Item>
          <Descriptions.Item label="算力">
            {hardwareConfig.chip.compute_tflops_fp16} TFLOPs
          </Descriptions.Item>
          <Descriptions.Item label="显存">
            {hardwareConfig.chip.memory_gb} GB
          </Descriptions.Item>
          <Descriptions.Item label="带宽">
            {hardwareConfig.chip.memory_bandwidth_gbps} GB/s
          </Descriptions.Item>
          <Descriptions.Item label="芯片数/节点">
            {hardwareConfig.node.chips_per_node}
          </Descriptions.Item>
          <Descriptions.Item label="节点数">
            {hardwareConfig.cluster.num_nodes}
          </Descriptions.Item>
          <Descriptions.Item label="总芯片数">
            {hardwareConfig.node.chips_per_node * hardwareConfig.cluster.num_nodes}
          </Descriptions.Item>
          <Descriptions.Item label="节点内带宽">
            {hardwareConfig.node.intra_node_bandwidth_gbps} Gbps
          </Descriptions.Item>
          <Descriptions.Item label="节点间带宽">
            {hardwareConfig.cluster.inter_node_bandwidth_gbps} Gbps
          </Descriptions.Item>
        </Descriptions>
      ),
    },
    {
      key: 'result',
      label: '详细结果',
      children: (
        <Descriptions column={2} size="small" bordered>
          <Descriptions.Item label="显存利用率" span={2}>
            {(result.memory.memory_utilization * 100).toFixed(1)}%
            ({result.memory.total_per_chip_gb.toFixed(1)} / {hardwareConfig.chip.memory_gb} GB)
          </Descriptions.Item>
          <Descriptions.Item label="模型显存">
            {result.memory.model_memory_gb.toFixed(2)} GB
          </Descriptions.Item>
          <Descriptions.Item label="KV Cache">
            {result.memory.kv_cache_memory_gb.toFixed(2)} GB
          </Descriptions.Item>
          <Descriptions.Item label="激活显存">
            {result.memory.activation_memory_gb.toFixed(2)} GB
          </Descriptions.Item>
          <Descriptions.Item label="显存充足">
            {result.memory.is_memory_sufficient ? '是' : '否'}
          </Descriptions.Item>
          <Descriptions.Item label="Prefill计算" span={1}>
            {result.latency.prefill_compute_latency_ms.toFixed(2)} ms
          </Descriptions.Item>
          <Descriptions.Item label="Prefill通信">
            {result.latency.prefill_comm_latency_ms.toFixed(2)} ms
          </Descriptions.Item>
          <Descriptions.Item label="Decode计算">
            {result.latency.decode_compute_latency_ms.toFixed(3)} ms
          </Descriptions.Item>
          <Descriptions.Item label="Decode通信">
            {result.latency.decode_comm_latency_ms.toFixed(3)} ms
          </Descriptions.Item>
          <Descriptions.Item label="瓶颈类型">
            <Tag color="warning">{result.latency.bottleneck_type}</Tag>
          </Descriptions.Item>
          <Descriptions.Item label="流水线气泡比">
            {(result.latency.pipeline_bubble_ratio * 100).toFixed(1)}%
          </Descriptions.Item>
        </Descriptions>
      ),
    },
  ]

  // 如果有 topKPlans，添加候选方案 Tab
  if (item.topKPlans && item.topKPlans.length > 1) {
    tabItems.push({
      key: 'candidates',
      label: `候选方案 (${item.topKPlans.length})`,
      children: (
        <div>
          <div style={{ marginBottom: 12, fontSize: 12, color: '#666' }}>
            自动搜索找到的 Top-{item.topKPlans.length} 候选方案，按综合评分排序
          </div>
          <Table
            size="small"
            dataSource={item.topKPlans.map((plan, index) => ({
              ...plan,
              key: plan.plan.plan_id,
              rank: index + 1,
            }))}
            columns={[
              {
                title: '#',
                dataIndex: 'rank',
                key: 'rank',
                width: 40,
                render: (rank: number) => (
                  <Text strong style={{ color: rank === 1 ? colors.primary : '#666' }}>
                    {rank}
                  </Text>
                ),
              },
              {
                title: '并行策略',
                key: 'parallelism',
                width: 140,
                render: (_: unknown, plan: PlanAnalysisResult) => {
                  const p = plan.plan.parallelism
                  return (
                    <Space size={2}>
                      <Tag color="blue" style={{ fontSize: 10, padding: '0 4px' }}>DP{p.dp}</Tag>
                      <Tag color="green" style={{ fontSize: 10, padding: '0 4px' }}>TP{p.tp}</Tag>
                      <Tag color="orange" style={{ fontSize: 10, padding: '0 4px' }}>PP{p.pp}</Tag>
                      {p.ep > 1 && <Tag color="purple" style={{ fontSize: 10, padding: '0 4px' }}>EP{p.ep}</Tag>}
                    </Space>
                  )
                },
              },
              {
                title: '评分',
                key: 'score',
                width: 60,
                align: 'center' as const,
                render: (_: unknown, plan: PlanAnalysisResult) => (
                  <Text strong style={{ color: colors.primary }}>
                    {plan.score.overall_score.toFixed(1)}
                  </Text>
                ),
              },
              {
                title: 'TTFT',
                key: 'ttft',
                width: 70,
                align: 'right' as const,
                render: (_: unknown, plan: PlanAnalysisResult) => (
                  <span style={{ fontSize: 11 }}>
                    {plan.latency.prefill_total_latency_ms.toFixed(1)} ms
                  </span>
                ),
              },
              {
                title: 'TPOT',
                key: 'tpot',
                width: 70,
                align: 'right' as const,
                render: (_: unknown, plan: PlanAnalysisResult) => (
                  <span style={{ fontSize: 11 }}>
                    {plan.latency.decode_per_token_latency_ms.toFixed(2)} ms
                  </span>
                ),
              },
              {
                title: '吞吐',
                key: 'throughput',
                width: 70,
                align: 'right' as const,
                render: (_: unknown, plan: PlanAnalysisResult) => (
                  <span style={{ fontSize: 11 }}>
                    {plan.throughput.tokens_per_second.toFixed(0)}
                  </span>
                ),
              },
              {
                title: 'MFU',
                key: 'mfu',
                width: 60,
                align: 'right' as const,
                render: (_: unknown, plan: PlanAnalysisResult) => (
                  <span style={{ fontSize: 11 }}>
                    {(plan.throughput.model_flops_utilization * 100).toFixed(1)}%
                  </span>
                ),
              },
            ]}
            pagination={false}
            scroll={{ y: 300 }}
          />
        </div>
      ),
    })
  }

  return (
    <Modal
      title={
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <FileTextOutlined style={{ color: colors.primary }} />
          <span>配置详情</span>
          <Tag color="blue">{new Date(item.timestamp).toLocaleString()}</Tag>
        </div>
      }
      open={visible}
      onCancel={onClose}
      width={800}
      footer={[
        <Button key="close" onClick={onClose}>
          关闭
        </Button>,
        <Button
          key="load"
          type="primary"
          icon={<ReloadOutlined />}
          onClick={() => {
            onLoad(item)
            onClose()
          }}
        >
          加载此配置
        </Button>,
      ]}
    >
      <Tabs items={tabItems} />
    </Modal>
  )
}

// ============================================
// 主面板组件
// ============================================

export const AnalysisHistoryPanel: React.FC<AnalysisHistoryPanelProps> = ({
  history,
  onLoad,
  onDelete,
  onClear,
}) => {
  const [searchText, setSearchText] = useState('')
  const [selectedItem, setSelectedItem] = useState<AnalysisHistoryItem | null>(null)
  const [detailVisible, setDetailVisible] = useState(false)

  // 过滤历史记录
  const filteredHistory = useMemo(() => {
    if (!searchText.trim()) return history
    const lower = searchText.toLowerCase()
    return history.filter(item =>
      item.modelName.toLowerCase().includes(lower) ||
      item.hardwareConfig.chip.chip_type.toLowerCase().includes(lower)
    )
  }, [history, searchText])

  // 导出为 JSON
  const handleExportJSON = () => {
    const data = history.map(item => ({
      timestamp: new Date(item.timestamp).toISOString(),
      model: item.modelName,
      parallelism: `DP${item.parallelism.dp}-TP${item.parallelism.tp}-PP${item.parallelism.pp}${item.parallelism.ep > 1 ? `-EP${item.parallelism.ep}` : ''}`,
      chips: item.chips,
      score: item.score,
      ttft_ms: item.ttft,
      tpot_ms: item.tpot,
      throughput_tps: item.throughput,
      mfu_percent: item.mfu * 100,
      mbu_percent: item.mbu * 100,
      cost_per_m_tokens: item.cost,
      modelConfig: item.modelConfig,
      inferenceConfig: item.inferenceConfig,
      hardwareConfig: item.hardwareConfig,
    }))
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `llm-deployment-analysis-${new Date().toISOString().slice(0, 10)}.json`
    a.click()
    URL.revokeObjectURL(url)
    message.success('已导出 JSON 文件')
  }

  // 导出为 CSV
  const handleExportCSV = () => {
    const headers = [
      '时间', '模型', '并行策略', '芯片数', '评分',
      'TTFT(ms)', 'TPOT(ms)', '吞吐量(tok/s)', 'MFU(%)', 'MBU(%)', '成本($/M)',
      '芯片类型', '显存(GB)', '算力(TFLOPs)',
      '批次', '输入长度', '输出长度'
    ]
    const rows = history.map(item => [
      new Date(item.timestamp).toLocaleString(),
      item.modelName,
      `DP${item.parallelism.dp}-TP${item.parallelism.tp}-PP${item.parallelism.pp}${item.parallelism.ep > 1 ? `-EP${item.parallelism.ep}` : ''}`,
      item.chips,
      item.score.toFixed(2),
      item.ttft.toFixed(2),
      item.tpot.toFixed(3),
      item.throughput.toFixed(0),
      (item.mfu * 100).toFixed(2),
      (item.mbu * 100).toFixed(2),
      item.cost !== null ? item.cost.toFixed(4) : '',
      item.hardwareConfig.chip.chip_type,
      item.hardwareConfig.chip.memory_gb,
      item.hardwareConfig.chip.compute_tflops_fp16,
      item.inferenceConfig.batch_size,
      item.inferenceConfig.input_seq_length,
      item.inferenceConfig.output_seq_length,
    ])
    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n')
    const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `llm-deployment-analysis-${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
    URL.revokeObjectURL(url)
    message.success('已导出 CSV 文件')
  }

  // 表格列定义
  const columns: ColumnsType<AnalysisHistoryItem> = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 90,
      render: (ts: number) => {
        const d = new Date(ts)
        return (
          <span style={{ fontSize: 11, color: '#666' }}>
            {d.getMonth() + 1}/{d.getDate()} {d.getHours().toString().padStart(2, '0')}:{d.getMinutes().toString().padStart(2, '0')}
          </span>
        )
      },
      sorter: (a, b) => b.timestamp - a.timestamp,
      defaultSortOrder: 'ascend',
    },
    {
      title: '模型',
      dataIndex: 'modelName',
      key: 'modelName',
      width: 120,
      ellipsis: true,
      render: (name: string) => (
        <Tooltip title={name}>
          <Text strong style={{ fontSize: 12 }}>{name}</Text>
        </Tooltip>
      ),
    },
    {
      title: '并行策略',
      key: 'parallelism',
      width: 110,
      render: (_, record) => {
        const p = record.parallelism
        return (
          <span style={{ fontSize: 10 }}>
            <Tag color="blue" style={{ marginRight: 2, padding: '0 4px' }}>DP{p.dp}</Tag>
            <Tag color="green" style={{ marginRight: 2, padding: '0 4px' }}>TP{p.tp}</Tag>
            <Tag color="orange" style={{ marginRight: 2, padding: '0 4px' }}>PP{p.pp}</Tag>
            {p.ep > 1 && <Tag color="purple" style={{ padding: '0 4px' }}>EP{p.ep}</Tag>}
          </span>
        )
      },
    },
    {
      title: '芯片',
      dataIndex: 'chips',
      key: 'chips',
      width: 50,
      align: 'center',
      sorter: (a, b) => a.chips - b.chips,
    },
    {
      title: '模式',
      key: 'searchMode',
      width: 60,
      align: 'center',
      render: (_, record) => (
        <Tooltip title={record.topKPlans ? `含 ${record.topKPlans.length} 个候选方案` : '手动配置'}>
          <Tag
            color={record.searchMode === 'auto' ? 'blue' : 'default'}
            style={{ fontSize: 10, padding: '0 4px' }}
          >
            {record.searchMode === 'auto' ? `自动(${record.topKPlans?.length || 1})` : '手动'}
          </Tag>
        </Tooltip>
      ),
    },
    {
      title: '评分',
      dataIndex: 'score',
      key: 'score',
      width: 60,
      align: 'center',
      render: (score: number) => (
        <Text strong style={{ color: colors.primary, fontSize: 14 }}>{score.toFixed(1)}</Text>
      ),
      sorter: (a, b) => b.score - a.score,
    },
    {
      title: 'TTFT',
      dataIndex: 'ttft',
      key: 'ttft',
      width: 70,
      align: 'right',
      render: (v: number) => <span style={{ fontSize: 11 }}>{v.toFixed(1)}ms</span>,
      sorter: (a, b) => a.ttft - b.ttft,
    },
    {
      title: 'TPOT',
      dataIndex: 'tpot',
      key: 'tpot',
      width: 65,
      align: 'right',
      render: (v: number) => <span style={{ fontSize: 11 }}>{v.toFixed(2)}ms</span>,
      sorter: (a, b) => a.tpot - b.tpot,
    },
    {
      title: '吞吐',
      dataIndex: 'throughput',
      key: 'throughput',
      width: 65,
      align: 'right',
      render: (v: number) => <span style={{ fontSize: 11 }}>{v.toFixed(0)}</span>,
      sorter: (a, b) => b.throughput - a.throughput,
    },
    {
      title: '操作',
      key: 'action',
      width: 80,
      align: 'center',
      render: (_, record) => (
        <Space size={0}>
          <Tooltip title="查看详情">
            <Button
              type="text"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => {
                setSelectedItem(record)
                setDetailVisible(true)
              }}
            />
          </Tooltip>
          <Popconfirm
            title="确定删除此记录？"
            onConfirm={() => onDelete(record.id)}
            okText="删除"
            cancelText="取消"
          >
            <Button type="text" size="small" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ]

  if (history.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: 24, color: '#999' }}>
        暂无历史记录
      </div>
    )
  }

  return (
    <div>
      {/* 工具栏 */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
        <Input
          placeholder="搜索模型或芯片类型..."
          prefix={<SearchOutlined style={{ color: '#bfbfbf' }} />}
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          allowClear
          style={{ width: 200 }}
          size="small"
        />
        <Space>
          <Button
            size="small"
            icon={<DownloadOutlined />}
            onClick={handleExportJSON}
          >
            JSON
          </Button>
          <Button
            size="small"
            icon={<DownloadOutlined />}
            onClick={handleExportCSV}
          >
            CSV
          </Button>
          <Popconfirm
            title="确定清空所有历史记录？"
            onConfirm={onClear}
            okText="清空"
            cancelText="取消"
          >
            <Button size="small" danger icon={<ClearOutlined />}>
              清空
            </Button>
          </Popconfirm>
        </Space>
      </div>

      {/* 表格 */}
      <Table
        columns={columns}
        dataSource={filteredHistory}
        rowKey="id"
        size="small"
        pagination={{
          pageSize: 10,
          showSizeChanger: false,
          showTotal: (total) => `共 ${total} 条`,
        }}
        scroll={{ x: 800 }}
        onRow={(record) => ({
          onClick: () => {
            setSelectedItem(record)
            setDetailVisible(true)
          },
          style: { cursor: 'pointer' },
        })}
      />

      {/* 配置详情模态框 */}
      <ConfigDetailModal
        item={selectedItem}
        visible={detailVisible}
        onClose={() => setDetailVisible(false)}
        onLoad={onLoad}
      />
    </div>
  )
}

export default AnalysisHistoryPanel
