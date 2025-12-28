/**
 * 分析结果展示组件
 *
 * - 首页显示历史记录列表
 * - 点击历史记录查看详情
 * - 支持返回历史记录列表
 */

import React, { useState, useCallback } from 'react'
import {
  Typography,
  Progress,
  Spin,
  Tag,
  Tooltip,
  Button,
  Table,
  Popconfirm,
  Empty,
} from 'antd'
import {
  InfoCircleOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  HistoryOutlined,
  ArrowLeftOutlined,
  DeleteOutlined,
  ClearOutlined,
  ExportOutlined,
} from '@ant-design/icons'
import { PlanAnalysisResult, HardwareConfig, LLMModelConfig } from '../../../utils/llmDeployment/types'
import { AnalysisHistoryItem, AnalysisViewMode } from '../shared'
import { colors } from './ConfigSelectors'
import { MetricDetailCard } from './components/MetricDetailCard'
import { HeroKPIPanel } from './charts'

const { Text } = Typography

// ============================================
// 历史记录列表组件
// ============================================

interface HistoryListProps {
  history: AnalysisHistoryItem[]
  onLoad: (item: AnalysisHistoryItem) => void
  onDelete: (id: string) => void
  onClear: () => void
}

const HistoryList: React.FC<HistoryListProps> = ({
  history,
  onLoad,
  onDelete,
  onClear,
}) => {
  // 导出JSON
  const handleExportJSON = () => {
    const blob = new Blob([JSON.stringify(history, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `llm-deployment-history-${new Date().toISOString().split('T')[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (history.length === 0) {
    return (
      <Empty
        image={Empty.PRESENTED_IMAGE_SIMPLE}
        description="暂无历史记录"
        style={{ padding: '40px 0' }}
      >
        <Text type="secondary" style={{ fontSize: 12 }}>
          点击左侧"运行分析"开始第一次分析
        </Text>
      </Empty>
    )
  }

  const columns = [
    {
      title: '模型',
      dataIndex: 'modelName',
      key: 'model',
      width: 120,
      render: (name: string) => (
        <Text strong style={{ fontSize: 13 }}>{name}</Text>
      ),
    },
    {
      title: '并行策略',
      key: 'parallelism',
      width: 140,
      render: (_: unknown, record: AnalysisHistoryItem) => (
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
          <Tag color="blue" style={{ fontSize: 10, margin: 0 }}>DP{record.parallelism.dp}</Tag>
          <Tag color="green" style={{ fontSize: 10, margin: 0 }}>TP{record.parallelism.tp}</Tag>
          <Tag color="orange" style={{ fontSize: 10, margin: 0 }}>PP{record.parallelism.pp}</Tag>
          {record.parallelism.ep > 1 && (
            <Tag color="purple" style={{ fontSize: 10, margin: 0 }}>EP{record.parallelism.ep}</Tag>
          )}
        </div>
      ),
    },
    {
      title: '评分',
      dataIndex: 'score',
      key: 'score',
      width: 70,
      render: (score: number) => (
        <Text strong style={{ color: score >= 70 ? colors.success : score >= 50 ? colors.warning : colors.error }}>
          {score.toFixed(1)}
        </Text>
      ),
    },
    {
      title: 'TTFT',
      dataIndex: 'ttft',
      key: 'ttft',
      width: 80,
      render: (v: number) => `${v.toFixed(1)}ms`,
    },
    {
      title: '吞吐',
      dataIndex: 'throughput',
      key: 'throughput',
      width: 90,
      render: (v: number) => `${v.toFixed(0)} tok/s`,
    },
    {
      title: '芯片',
      dataIndex: 'chips',
      key: 'chips',
      width: 60,
      render: (v: number) => v,
    },
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'time',
      width: 100,
      render: (ts: number) => {
        const date = new Date(ts)
        return (
          <Text type="secondary" style={{ fontSize: 11 }}>
            {date.toLocaleDateString()} {date.toLocaleTimeString().slice(0, 5)}
          </Text>
        )
      },
    },
    {
      title: '',
      key: 'actions',
      width: 40,
      render: (_: unknown, record: AnalysisHistoryItem) => (
        <Popconfirm
          title="删除此记录？"
          onConfirm={(e) => {
            e?.stopPropagation()
            onDelete(record.id)
          }}
          okText="删除"
          cancelText="取消"
        >
          <Button
            type="text"
            size="small"
            icon={<DeleteOutlined />}
            onClick={(e) => e.stopPropagation()}
            style={{ color: '#999' }}
          />
        </Popconfirm>
      ),
    },
  ]

  return (
    <div>
      {/* 标题栏 */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 16,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <HistoryOutlined style={{ fontSize: 18, color: colors.primary }} />
          <Text strong style={{ fontSize: 16 }}>历史记录</Text>
          <Tag color="default">{history.length}</Tag>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <Button
            size="small"
            icon={<ExportOutlined />}
            onClick={handleExportJSON}
          >
            导出
          </Button>
          <Popconfirm
            title="清空所有历史记录？"
            onConfirm={onClear}
            okText="清空"
            cancelText="取消"
          >
            <Button size="small" icon={<ClearOutlined />} danger>
              清空
            </Button>
          </Popconfirm>
        </div>
      </div>

      {/* 历史记录表格 */}
      <Table
        dataSource={history}
        columns={columns}
        rowKey="id"
        size="small"
        pagination={{ pageSize: 10, showSizeChanger: false }}
        onRow={(record) => ({
          onClick: () => onLoad(record),
          style: { cursor: 'pointer' },
        })}
        style={{ marginTop: 8 }}
      />

      <div style={{
        marginTop: 12,
        padding: '8px 12px',
        background: '#f5f5f5',
        borderRadius: 6,
        fontSize: 12,
        color: '#666',
        textAlign: 'center',
      }}>
        💡 点击行查看详细分析结果
      </div>
    </div>
  )
}

// ============================================
// 分析结果展示组件
// ============================================

interface AnalysisResultDisplayProps {
  result: PlanAnalysisResult | null
  topKPlans: PlanAnalysisResult[]
  loading: boolean
  onSelectPlan?: (plan: PlanAnalysisResult) => void
  searchStats?: { evaluated: number; feasible: number; timeMs: number } | null
  errorMsg?: string | null
  // 视图模式（从父组件传入）
  viewMode?: AnalysisViewMode
  onViewModeChange?: (mode: AnalysisViewMode) => void
  // 历史记录相关
  history?: AnalysisHistoryItem[]
  onLoadFromHistory?: (item: AnalysisHistoryItem) => void
  onDeleteHistory?: (id: string) => void
  onClearHistory?: () => void
  // 详情视图功能按钮
  canMapToTopology?: boolean
  onMapToTopology?: () => void
  onClearTraffic?: () => void
  // HeroKPIPanel 需要的数据
  hardware?: HardwareConfig
  model?: LLMModelConfig
}

type MetricType = 'ttft' | 'tpot' | 'throughput' | 'mfu' | 'mbu' | 'cost' | 'percentiles' | 'bottleneck' | 'e2e' | 'chips' | null

export const AnalysisResultDisplay: React.FC<AnalysisResultDisplayProps> = ({
  result,
  topKPlans,
  loading,
  onSelectPlan,
  searchStats,
  errorMsg,
  viewMode = 'history',
  onViewModeChange,
  history = [],
  onLoadFromHistory,
  onDeleteHistory,
  onClearHistory,
  canMapToTopology,
  onMapToTopology,
  onClearTraffic,
}) => {
  const [selectedMetric, setSelectedMetric] = useState<MetricType>(null)

  // 从历史记录加载（父组件会自动切换到详情视图）
  const handleLoadFromHistory = useCallback((item: AnalysisHistoryItem) => {
    onLoadFromHistory?.(item)
  }, [onLoadFromHistory])

  // 返回历史列表
  const handleBackToHistory = useCallback(() => {
    onViewModeChange?.('history')
  }, [onViewModeChange])

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 40 }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>
          <Text type="secondary">正在搜索最优方案...</Text>
        </div>
      </div>
    )
  }

  if (errorMsg) {
    return (
      <div style={{ padding: 16 }}>
        <div style={{ textAlign: 'center', padding: 20, background: '#fff2f0', borderRadius: 8, border: '1px solid #ffccc7' }}>
          <WarningOutlined style={{ fontSize: 24, color: '#ff4d4f', marginBottom: 8 }} />
          <div style={{ color: '#ff4d4f', fontWeight: 500 }}>{errorMsg}</div>
        </div>
        {searchStats && (
          <div style={{ marginTop: 12, padding: 8, background: '#f5f5f5', borderRadius: 6 }}>
            <Text type="secondary" style={{ fontSize: 11 }}>
              搜索统计: 评估 {searchStats.evaluated} 个方案，{searchStats.feasible} 个可行，耗时 {searchStats.timeMs.toFixed(0)}ms
            </Text>
          </div>
        )}
      </div>
    )
  }

  // 历史列表视图
  if (viewMode === 'history') {
    return (
      <div style={{ padding: 4 }}>
        {/* 如果有已查看的结果，显示返回按钮 */}
        {result && (
          <Button
            type="link"
            icon={<ArrowLeftOutlined />}
            onClick={() => onViewModeChange?.('detail')}
            style={{ marginBottom: 12, padding: 0 }}
          >
            返回当前分析
          </Button>
        )}
        <HistoryList
          history={history}
          onLoad={handleLoadFromHistory}
          onDelete={onDeleteHistory || (() => {})}
          onClear={onClearHistory || (() => {})}
        />
      </div>
    )
  }

  // 详情视图但没有结果（回退到历史列表）
  if (!result) {
    return (
      <div style={{ padding: 4 }}>
        <HistoryList
          history={history}
          onLoad={handleLoadFromHistory}
          onDelete={onDeleteHistory || (() => {})}
          onClear={onClearHistory || (() => {})}
        />
      </div>
    )
  }

  const { plan, memory, latency, throughput, score, suggestions, is_feasible, infeasibility_reason } = result

  // 指标卡片样式
  const metricCardStyle = (isSelected: boolean): React.CSSProperties => ({
    padding: '14px 12px',
    background: isSelected ? colors.primaryLight : '#fff',
    borderRadius: 10,
    cursor: 'pointer',
    border: isSelected ? `2px solid ${colors.primary}` : `1px solid ${colors.border}`,
    transition: 'all 0.2s ease',
    boxShadow: isSelected ? `0 2px 8px rgba(94, 106, 210, 0.15)` : '0 1px 2px rgba(0, 0, 0, 0.04)',
  })

  // 并行策略标签样式
  const parallelTagStyle: React.CSSProperties = {
    background: colors.primaryLight,
    color: colors.primary,
    border: 'none',
    borderRadius: 4,
    fontWeight: 500,
    fontSize: 11,
    padding: '2px 8px',
  }

  return (
    <div>
      {/* 顶部操作栏：返回按钮 + 功能按钮 */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 16,
      }}>
        <Button
          type="link"
          icon={<ArrowLeftOutlined />}
          onClick={handleBackToHistory}
          style={{ padding: 0, fontSize: 13 }}
        >
          返回历史记录
        </Button>
        {/* 映射按钮组 */}
        {canMapToTopology && (
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={onMapToTopology}
              style={{
                padding: '6px 12px',
                background: '#5E6AD2',
                color: '#fff',
                border: 'none',
                borderRadius: 6,
                cursor: 'pointer',
                fontSize: 12,
                fontWeight: 500,
              }}
            >
              映射到拓扑
            </button>
            <button
              onClick={onClearTraffic}
              style={{
                padding: '6px 10px',
                background: '#fff',
                color: '#666',
                border: '1px solid #d9d9d9',
                borderRadius: 6,
                cursor: 'pointer',
                fontSize: 12,
              }}
            >
              清除
            </button>
          </div>
        )}
      </div>

      {/* Hero KPI 面板 */}
      <div style={{ marginBottom: 16 }}>
        <HeroKPIPanel
          result={result}
          selectedMetric={null}
          onMetricClick={() => {}}
        />
      </div>

      {/* 方案概览 */}
      <div style={{
        padding: 14,
        background: is_feasible ? '#fff' : colors.errorLight,
        borderRadius: 12,
        marginBottom: 14,
        border: `1px solid ${is_feasible ? colors.border : '#ffccc7'}`,
        boxShadow: '0 2px 6px rgba(0, 0, 0, 0.04)',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <Text style={{ fontSize: 11, color: colors.textSecondary, display: 'block', marginBottom: 6 }}>并行策略</Text>
            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
              <Tag style={parallelTagStyle}>DP={plan.parallelism.dp}</Tag>
              <Tag style={parallelTagStyle}>TP={plan.parallelism.tp}</Tag>
              <Tag style={parallelTagStyle}>PP={plan.parallelism.pp}</Tag>
              {plan.parallelism.ep > 1 && <Tag style={parallelTagStyle}>EP={plan.parallelism.ep}</Tag>}
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'flex-end' }}>
              {is_feasible ? (
                <CheckCircleOutlined style={{ color: colors.success, fontSize: 16 }} />
              ) : (
                <Tooltip title={infeasibility_reason}>
                  <WarningOutlined style={{ color: colors.error, fontSize: 16 }} />
                </Tooltip>
              )}
              <Text strong style={{ fontSize: 22, color: is_feasible ? colors.success : colors.error, lineHeight: 1 }}>
                {score.overall_score.toFixed(1)}
              </Text>
            </div>
            <Text style={{ fontSize: 10, color: colors.textSecondary }}>综合评分</Text>
          </div>
        </div>
      </div>

      {/* 关键指标 - 3x3网格 */}
      <div style={{ marginBottom: 14 }}>
        <Text style={{ fontSize: 11, color: colors.textSecondary, display: 'block', marginBottom: 8 }}>关键指标 (点击查看详情)</Text>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
          {/* TTFT */}
          <div
            style={metricCardStyle(selectedMetric === 'ttft')}
            onClick={() => setSelectedMetric(selectedMetric === 'ttft' ? null : 'ttft')}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <Text style={{ fontSize: 10, color: colors.textSecondary }}>TTFT</Text>
              <InfoCircleOutlined style={{ fontSize: 9, color: selectedMetric === 'ttft' ? colors.primary : '#ccc' }} />
            </div>
            <div style={{ fontSize: 16, fontWeight: 600, color: colors.text, marginTop: 2 }}>
              {latency.prefill_total_latency_ms.toFixed(1)} <span style={{ fontSize: 10, fontWeight: 400, color: colors.textSecondary }}>ms</span>
            </div>
          </div>
          {/* TPOT */}
          <div
            style={metricCardStyle(selectedMetric === 'tpot')}
            onClick={() => setSelectedMetric(selectedMetric === 'tpot' ? null : 'tpot')}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <Text style={{ fontSize: 10, color: colors.textSecondary }}>TPOT</Text>
              <InfoCircleOutlined style={{ fontSize: 9, color: selectedMetric === 'tpot' ? colors.primary : '#ccc' }} />
            </div>
            <div style={{ fontSize: 16, fontWeight: 600, color: colors.text, marginTop: 2 }}>
              {latency.decode_per_token_latency_ms.toFixed(2)} <span style={{ fontSize: 10, fontWeight: 400, color: colors.textSecondary }}>ms</span>
            </div>
          </div>
          {/* Throughput */}
          <div
            style={metricCardStyle(selectedMetric === 'throughput')}
            onClick={() => setSelectedMetric(selectedMetric === 'throughput' ? null : 'throughput')}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <Text style={{ fontSize: 10, color: colors.textSecondary }}>吞吐量</Text>
              <InfoCircleOutlined style={{ fontSize: 9, color: selectedMetric === 'throughput' ? colors.primary : '#ccc' }} />
            </div>
            <div style={{ fontSize: 16, fontWeight: 600, color: colors.text, marginTop: 2 }}>
              {throughput.tokens_per_second.toFixed(0)} <span style={{ fontSize: 10, fontWeight: 400, color: colors.textSecondary }}>tok/s</span>
            </div>
          </div>
          {/* MFU */}
          <div
            style={metricCardStyle(selectedMetric === 'mfu')}
            onClick={() => setSelectedMetric(selectedMetric === 'mfu' ? null : 'mfu')}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <Text style={{ fontSize: 10, color: colors.textSecondary }}>MFU</Text>
              <InfoCircleOutlined style={{ fontSize: 9, color: selectedMetric === 'mfu' ? colors.primary : '#ccc' }} />
            </div>
            <div style={{ fontSize: 16, fontWeight: 600, color: colors.text, marginTop: 2 }}>
              {(throughput.model_flops_utilization * 100).toFixed(1)} <span style={{ fontSize: 10, fontWeight: 400, color: colors.textSecondary }}>%</span>
            </div>
          </div>
          {/* MBU */}
          <div
            style={metricCardStyle(selectedMetric === 'mbu')}
            onClick={() => setSelectedMetric(selectedMetric === 'mbu' ? null : 'mbu')}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <Text style={{ fontSize: 10, color: colors.textSecondary }}>MBU</Text>
              <InfoCircleOutlined style={{ fontSize: 9, color: selectedMetric === 'mbu' ? colors.primary : '#ccc' }} />
            </div>
            <div style={{ fontSize: 16, fontWeight: 600, color: colors.text, marginTop: 2 }}>
              {(throughput.memory_bandwidth_utilization * 100).toFixed(1)} <span style={{ fontSize: 10, fontWeight: 400, color: colors.textSecondary }}>%</span>
            </div>
          </div>
          {/* P99 延迟 */}
          <div
            style={metricCardStyle(selectedMetric === 'percentiles')}
            onClick={() => setSelectedMetric(selectedMetric === 'percentiles' ? null : 'percentiles')}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <Text style={{ fontSize: 10, color: colors.textSecondary }}>P99</Text>
              <InfoCircleOutlined style={{ fontSize: 9, color: selectedMetric === 'percentiles' ? colors.primary : '#ccc' }} />
            </div>
            <div style={{ fontSize: 16, fontWeight: 600, color: latency.ttft_percentiles && latency.ttft_percentiles.p99 > 450 ? colors.error : colors.text, marginTop: 2 }}>
              {latency.ttft_percentiles ? latency.ttft_percentiles.p99.toFixed(0) : '-'} <span style={{ fontSize: 10, fontWeight: 400, color: colors.textSecondary }}>ms</span>
            </div>
          </div>
          {/* 成本 */}
          <div
            style={metricCardStyle(selectedMetric === 'cost')}
            onClick={() => setSelectedMetric(selectedMetric === 'cost' ? null : 'cost')}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <Text style={{ fontSize: 10, color: colors.textSecondary }}>成本</Text>
              <InfoCircleOutlined style={{ fontSize: 9, color: selectedMetric === 'cost' ? colors.primary : '#ccc' }} />
            </div>
            <div style={{ fontSize: 16, fontWeight: 600, color: colors.text, marginTop: 2 }}>
              ${result.cost ? result.cost.cost_per_million_tokens.toFixed(3) : '-'} <span style={{ fontSize: 10, fontWeight: 400, color: colors.textSecondary }}>/M</span>
            </div>
          </div>
          {/* E2E 延迟 */}
          <div
            style={metricCardStyle(selectedMetric === 'e2e')}
            onClick={() => setSelectedMetric(selectedMetric === 'e2e' ? null : 'e2e')}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <Text style={{ fontSize: 10, color: colors.textSecondary }}>E2E</Text>
              <InfoCircleOutlined style={{ fontSize: 9, color: selectedMetric === 'e2e' ? colors.primary : '#ccc' }} />
            </div>
            <div style={{ fontSize: 16, fontWeight: 600, color: colors.text, marginTop: 2 }}>
              {(latency.end_to_end_latency_ms / 1000).toFixed(2)} <span style={{ fontSize: 10, fontWeight: 400, color: colors.textSecondary }}>s</span>
            </div>
          </div>
          {/* 芯片数 */}
          <div
            style={metricCardStyle(selectedMetric === 'chips')}
            onClick={() => setSelectedMetric(selectedMetric === 'chips' ? null : 'chips')}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <Text style={{ fontSize: 10, color: colors.textSecondary }}>芯片数</Text>
              <InfoCircleOutlined style={{ fontSize: 9, color: selectedMetric === 'chips' ? colors.primary : '#ccc' }} />
            </div>
            <div style={{ fontSize: 16, fontWeight: 600, color: colors.text, marginTop: 2 }}>
              {plan.total_chips} <span style={{ fontSize: 10, fontWeight: 400, color: colors.textSecondary }}>chips</span>
            </div>
          </div>
        </div>
      </div>

      {/* 指标详情展示 */}
      {selectedMetric && selectedMetric !== 'bottleneck' && (
        <MetricDetailCard metric={selectedMetric} result={result} />
      )}

      {/* 显存利用 */}
      <div style={{
        padding: 12,
        background: '#fff',
        borderRadius: 10,
        marginBottom: 14,
        border: `1px solid ${colors.border}`,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <Text style={{ fontSize: 11, color: colors.textSecondary }}>显存利用</Text>
          <Text style={{ fontSize: 13, fontWeight: 500, color: colors.text }}>
            {memory.total_per_chip_gb.toFixed(1)} <span style={{ color: colors.textSecondary, fontWeight: 400 }}>/ 80 GB</span>
          </Text>
        </div>
        <Progress
          percent={memory.memory_utilization * 100}
          status={memory.is_memory_sufficient ? 'normal' : 'exception'}
          size="small"
          strokeColor={memory.is_memory_sufficient ? colors.primary : colors.error}
          trailColor={colors.borderLight}
          format={(p) => <span style={{ fontSize: 11, color: colors.textSecondary }}>{p?.toFixed(0)}%</span>}
        />
        <div style={{ display: 'flex', gap: 12, marginTop: 8, fontSize: 10, color: colors.textSecondary }}>
          <span>模型: {memory.model_memory_gb.toFixed(1)}G</span>
          <span>KV Cache: {memory.kv_cache_memory_gb.toFixed(1)}G</span>
          <span>激活: {memory.activation_memory_gb.toFixed(1)}G</span>
        </div>
      </div>

      {/* 瓶颈分析 */}
      <div
        style={{
          padding: 12,
          background: selectedMetric === 'bottleneck' ? colors.warningLight : '#fff',
          borderRadius: 10,
          marginBottom: 14,
          cursor: 'pointer',
          border: selectedMetric === 'bottleneck' ? `2px solid ${colors.warning}` : `1px solid ${colors.border}`,
          transition: 'all 0.2s ease',
        }}
        onClick={() => setSelectedMetric(selectedMetric === 'bottleneck' ? null : 'bottleneck')}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
          <Text style={{ fontSize: 11, color: colors.textSecondary }}>性能瓶颈</Text>
          <InfoCircleOutlined style={{ fontSize: 10, color: selectedMetric === 'bottleneck' ? colors.warning : '#ccc' }} />
        </div>
        <Text strong style={{ fontSize: 13, color: colors.text }}>{latency.bottleneck_type}</Text>
        <Text style={{ fontSize: 11, color: colors.textSecondary, marginTop: 4, display: 'block' }}>{latency.bottleneck_details}</Text>
      </div>

      {/* 瓶颈详情展示 */}
      {selectedMetric === 'bottleneck' && (
        <MetricDetailCard metric="bottleneck" result={result} />
      )}

      {/* 优化建议 */}
      {suggestions.length > 0 && (
        <div style={{
          padding: 12,
          background: '#fff',
          borderRadius: 10,
          marginBottom: 14,
          border: `1px solid ${colors.border}`,
        }}>
          <Text style={{ fontSize: 11, color: colors.textSecondary, display: 'block', marginBottom: 10 }}>优化建议</Text>
          {suggestions.slice(0, 3).map((s, i) => (
            <div key={i} style={{
              padding: 10,
              background: colors.background,
              borderRadius: 8,
              marginBottom: i < 2 ? 8 : 0,
              borderLeft: `3px solid ${s.priority <= 2 ? colors.error : s.priority <= 3 ? colors.warning : colors.primary}`,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <Text style={{ fontSize: 12, color: colors.text, flex: 1 }}>{s.description}</Text>
                <Tag
                  style={{
                    fontSize: 9,
                    padding: '0 6px',
                    borderRadius: 4,
                    border: 'none',
                    background: s.priority <= 2 ? colors.errorLight : s.priority <= 3 ? colors.warningLight : colors.primaryLight,
                    color: s.priority <= 2 ? colors.error : s.priority <= 3 ? colors.warning : colors.primary,
                    marginLeft: 8,
                  }}
                >
                  P{s.priority}
                </Tag>
              </div>
              <Text style={{ fontSize: 10, color: colors.textSecondary, marginTop: 4, display: 'block' }}>预期: {s.expected_improvement}</Text>
            </div>
          ))}
        </div>
      )}

      {/* Top-K 方案列表 */}
      {topKPlans.length > 1 && (
        <div style={{
          padding: 12,
          background: '#fff',
          borderRadius: 10,
          marginBottom: 14,
          border: `1px solid ${colors.border}`,
        }}>
          <Text style={{ fontSize: 11, color: colors.textSecondary, display: 'block', marginBottom: 10 }}>
            候选方案 ({topKPlans.length}个)
          </Text>
          <div style={{ maxHeight: 180, overflow: 'auto' }}>
            {topKPlans.map((p, i) => {
              const isSelected = p.plan.plan_id === result?.plan.plan_id
              return (
                <div
                  key={p.plan.plan_id}
                  onClick={() => onSelectPlan?.(p)}
                  style={{
                    padding: 10,
                    background: isSelected ? colors.primaryLight : colors.background,
                    borderRadius: 8,
                    marginBottom: 6,
                    cursor: 'pointer',
                    border: isSelected ? `1px solid ${colors.primary}` : `1px solid ${colors.borderLight}`,
                    transition: 'all 0.2s ease',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={{
                        fontSize: 11,
                        fontWeight: 600,
                        color: isSelected ? colors.primary : colors.textSecondary,
                        minWidth: 20,
                      }}>
                        #{i + 1}
                      </span>
                      <div style={{ display: 'flex', gap: 3 }}>
                        <span style={{ fontSize: 10, color: colors.textSecondary }}>DP{p.plan.parallelism.dp}</span>
                        <span style={{ fontSize: 10, color: colors.textSecondary }}>·</span>
                        <span style={{ fontSize: 10, color: colors.textSecondary }}>TP{p.plan.parallelism.tp}</span>
                        <span style={{ fontSize: 10, color: colors.textSecondary }}>·</span>
                        <span style={{ fontSize: 10, color: colors.textSecondary }}>PP{p.plan.parallelism.pp}</span>
                        {p.plan.parallelism.ep > 1 && (
                          <>
                            <span style={{ fontSize: 10, color: colors.textSecondary }}>·</span>
                            <span style={{ fontSize: 10, color: colors.textSecondary }}>EP{p.plan.parallelism.ep}</span>
                          </>
                        )}
                      </div>
                    </div>
                    <Text style={{ fontSize: 14, fontWeight: 600, color: isSelected ? colors.primary : colors.text }}>
                      {p.score.overall_score.toFixed(1)}
                    </Text>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, fontSize: 10, color: colors.textSecondary }}>
                    <span>{p.latency.prefill_total_latency_ms.toFixed(1)}ms</span>
                    <span>{p.throughput.tokens_per_second.toFixed(0)} tok/s</span>
                    <span>{(p.throughput.model_flops_utilization * 100).toFixed(1)}%</span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* 搜索统计 */}
      {searchStats && (
        <div style={{
          padding: '8px 12px',
          background: colors.background,
          borderRadius: 8,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          <Text style={{ fontSize: 10, color: colors.textSecondary }}>
            搜索统计
          </Text>
          <div style={{ display: 'flex', gap: 12, fontSize: 10, color: colors.textSecondary }}>
            <span>评估 <b style={{ color: colors.text }}>{searchStats.evaluated}</b></span>
            <span>可行 <b style={{ color: colors.success }}>{searchStats.feasible}</b></span>
            <span>耗时 <b style={{ color: colors.text }}>{searchStats.timeMs.toFixed(0)}ms</b></span>
          </div>
        </div>
      )}
    </div>
  )
}

export default AnalysisResultDisplay
