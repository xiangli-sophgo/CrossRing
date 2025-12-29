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
  Collapse,
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
  ThunderboltOutlined,
  DashboardOutlined,
  ClockCircleOutlined,
  AimOutlined,
  DownOutlined,
  UpOutlined,
} from '@ant-design/icons'
import { PlanAnalysisResult, HardwareConfig, LLMModelConfig, InferenceConfig, DEFAULT_SCORE_WEIGHTS } from '../../../utils/llmDeployment/types'
import { AnalysisHistoryItem, AnalysisViewMode } from '../shared'
import { colors } from './ConfigSelectors'
import { MetricDetailCard } from './components/MetricDetailCard'
import { ModelInfoCard } from './components/ModelInfoCard'
import { ParallelismInfo, ParallelismCard, type ParallelismType } from './components/ParallelismInfo'

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
  inference?: InferenceConfig
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
  model,
  inference,
}) => {
  const [selectedMetric, setSelectedMetric] = useState<MetricType>(null)
  const [showScoreDetails, setShowScoreDetails] = useState(false)
  const [selectedParallelism, setSelectedParallelism] = useState<ParallelismType | null>(null)

  // 各章节折叠状态
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    deployment: true,
    model: true,
    performance: true,
    bottleneck: true,
    suggestions: true,
    candidates: true,
  })

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }))
  }

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
            返回分析详情
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

  // 章节标题样式
  const sectionTitleStyle: React.CSSProperties = {
    fontSize: 15,
    fontWeight: 600,
    color: colors.text,
    marginBottom: 12,
    paddingBottom: 8,
    borderBottom: `1px solid ${colors.borderLight}`,
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  }

  // 章节容器样式（与 ChartsPanel 保持一致）
  const sectionStyle: React.CSSProperties = {
    background: '#fff',
    borderRadius: 12,
    padding: 20,
    marginBottom: 16,
    boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
  }

  // 指标卡片样式
  const metricCardStyle = (isSelected: boolean): React.CSSProperties => ({
    padding: '12px 10px',
    background: isSelected ? colors.primaryLight : '#fff',
    borderRadius: 8,
    cursor: 'pointer',
    border: isSelected ? `2px solid ${colors.primary}` : `1px solid ${colors.border}`,
    transition: 'all 0.2s ease',
    boxShadow: isSelected ? `0 2px 8px rgba(94, 106, 210, 0.15)` : '0 1px 2px rgba(0, 0, 0, 0.04)',
  })

  return (
    <div>
      {/* 顶部导航栏 */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 14,
      }}>
        {model && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <Tag color="blue" style={{ fontSize: 12, margin: 0 }}>{model.model_name}</Tag>
            {is_feasible ? (
              <Tag color="success" style={{ fontSize: 11, margin: 0 }}>{score.overall_score.toFixed(1)}分</Tag>
            ) : (
              <Tag color="error" style={{ fontSize: 11, margin: 0 }}>不可行</Tag>
            )}
          </div>
        )}
        <Button
          type="text"
          size="small"
          icon={<ArrowLeftOutlined />}
          onClick={handleBackToHistory}
          style={{ fontSize: 12, color: colors.textSecondary }}
        >
          历史记录
        </Button>
      </div>

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* 一、部署方案 */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      <div style={sectionStyle}>
        <div
          style={{ ...sectionTitleStyle, cursor: 'pointer' }}
          onClick={() => toggleSection('deployment')}
        >
          部署方案
          <span style={{ marginLeft: 'auto' }}>
            {expandedSections.deployment ? <UpOutlined style={{ fontSize: 12 }} /> : <DownOutlined style={{ fontSize: 12 }} />}
          </span>
        </div>
        {/* 部署方案内容 */}
        {expandedSections.deployment && (
        <>
          {/* 顶部：并行策略卡片 + 综合评分 */}
          <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
            {/* 并行策略卡片 */}
            <div style={{ flex: 1, display: 'flex', gap: 8 }}>
              <ParallelismCard
                type="dp"
                value={plan.parallelism.dp}
                selected={selectedParallelism === 'dp'}
                onClick={() => setSelectedParallelism(selectedParallelism === 'dp' ? null : 'dp')}
              />
              <ParallelismCard
                type="tp"
                value={plan.parallelism.tp}
                selected={selectedParallelism === 'tp'}
                onClick={() => setSelectedParallelism(selectedParallelism === 'tp' ? null : 'tp')}
              />
              <ParallelismCard
                type="pp"
                value={plan.parallelism.pp}
                selected={selectedParallelism === 'pp'}
                onClick={() => setSelectedParallelism(selectedParallelism === 'pp' ? null : 'pp')}
              />
              {plan.parallelism.ep > 1 && (
                <ParallelismCard
                  type="ep"
                  value={plan.parallelism.ep}
                  selected={selectedParallelism === 'ep'}
                  onClick={() => setSelectedParallelism(selectedParallelism === 'ep' ? null : 'ep')}
                />
              )}
              {plan.parallelism.sp > 1 && (
                <ParallelismCard
                  type="sp"
                  value={plan.parallelism.sp}
                  selected={selectedParallelism === 'sp'}
                  onClick={() => setSelectedParallelism(selectedParallelism === 'sp' ? null : 'sp')}
                />
              )}
            </div>

            {/* 综合评分 */}
            <div
              style={{
                flex: '0 0 100px',
                padding: '8px 12px',
                background: is_feasible ? '#f6ffed' : '#fff2f0',
                border: `1.5px solid ${is_feasible ? '#b7eb8f' : '#ffccc7'}`,
                borderRadius: 8,
                textAlign: 'center',
                cursor: 'pointer',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
              }}
              onClick={() => setShowScoreDetails(!showScoreDetails)}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}>
                {is_feasible ? (
                  <CheckCircleOutlined style={{ color: colors.success, fontSize: 14 }} />
                ) : (
                  <Tooltip title={infeasibility_reason}>
                    <WarningOutlined style={{ color: colors.error, fontSize: 14 }} />
                  </Tooltip>
                )}
                <Text strong style={{ fontSize: 22, color: is_feasible ? colors.success : colors.error, lineHeight: 1 }}>
                  {score.overall_score.toFixed(1)}
                </Text>
              </div>
              <div style={{ fontSize: 10, color: colors.textSecondary, marginTop: 4 }}>
                综合评分 {showScoreDetails ? '▲' : '▼'}
              </div>
            </div>
          </div>

          {/* 芯片数和搜索统计 */}
          <div style={{ fontSize: 11, color: colors.textSecondary, marginBottom: 12 }}>
            <span>总芯片数: <b style={{ color: colors.text }}>{plan.total_chips}</b></span>
            {searchStats && (
              <span style={{ marginLeft: 16 }}>
                搜索空间: {searchStats.evaluated} 方案 · {searchStats.feasible} 可行 · {searchStats.timeMs.toFixed(0)}ms
              </span>
            )}
            <span style={{ marginLeft: 16, color: '#bbb' }}>点击策略卡片查看详情</span>
          </div>

          {/* 并行策略详细介绍 */}
          {selectedParallelism && (
            <div style={{
              marginBottom: 12,
              padding: 16,
              background: '#fafafa',
              borderRadius: 8,
              border: '1px solid #f0f0f0',
            }}>
              <ParallelismInfo type={selectedParallelism} />
            </div>
          )}

          {/* 评分详情展开区域 */}
          {showScoreDetails && (
            <div style={{
              marginTop: 12,
              paddingTop: 12,
              borderTop: `1px dashed ${colors.borderLight}`,
            }}>
              {/* 各项得分 */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, marginBottom: 12 }}>
                <div style={{ textAlign: 'center', padding: 8, background: '#f0f5ff', borderRadius: 6 }}>
                  <ClockCircleOutlined style={{ color: '#1890ff', fontSize: 14 }} />
                  <div style={{ fontSize: 16, fontWeight: 600, color: '#1890ff', margin: '4px 0' }}>
                    {score.latency_score.toFixed(0)}
                  </div>
                  <div style={{ fontSize: 10, color: colors.textSecondary }}>延迟 {(DEFAULT_SCORE_WEIGHTS.latency * 100).toFixed(0)}%</div>
                </div>
                <div style={{ textAlign: 'center', padding: 8, background: '#f6ffed', borderRadius: 6 }}>
                  <ThunderboltOutlined style={{ color: '#52c41a', fontSize: 14 }} />
                  <div style={{ fontSize: 16, fontWeight: 600, color: '#52c41a', margin: '4px 0' }}>
                    {score.throughput_score.toFixed(0)}
                  </div>
                  <div style={{ fontSize: 10, color: colors.textSecondary }}>吞吐 {(DEFAULT_SCORE_WEIGHTS.throughput * 100).toFixed(0)}%</div>
                </div>
                <div style={{ textAlign: 'center', padding: 8, background: '#fff7e6', borderRadius: 6 }}>
                  <DashboardOutlined style={{ color: '#faad14', fontSize: 14 }} />
                  <div style={{ fontSize: 16, fontWeight: 600, color: '#faad14', margin: '4px 0' }}>
                    {score.efficiency_score.toFixed(0)}
                  </div>
                  <div style={{ fontSize: 10, color: colors.textSecondary }}>效率 {(DEFAULT_SCORE_WEIGHTS.efficiency * 100).toFixed(0)}%</div>
                </div>
                <div style={{ textAlign: 'center', padding: 8, background: '#f9f0ff', borderRadius: 6 }}>
                  <AimOutlined style={{ color: '#722ed1', fontSize: 14 }} />
                  <div style={{ fontSize: 16, fontWeight: 600, color: '#722ed1', margin: '4px 0' }}>
                    {score.balance_score.toFixed(0)}
                  </div>
                  <div style={{ fontSize: 10, color: colors.textSecondary }}>均衡 {(DEFAULT_SCORE_WEIGHTS.balance * 100).toFixed(0)}%</div>
                </div>
              </div>

              {/* 评分规则说明 */}
              <Collapse
                size="small"
                style={{ background: '#fafafa', borderRadius: 6 }}
                items={[{
                  key: 'rules',
                  label: <Text style={{ fontSize: 12 }}>评分规则说明</Text>,
                  children: (
                    <div style={{ fontSize: 12, color: colors.textSecondary }}>
                      <div style={{ marginBottom: 8 }}>
                        <Text strong style={{ color: '#1890ff' }}>延迟评分：</Text>
                        <span>TTFT &lt; 100ms → 100分，TTFT &gt; 1000ms → 0分</span>
                      </div>
                      <div style={{ marginBottom: 8 }}>
                        <Text strong style={{ color: '#52c41a' }}>吞吐评分：</Text>
                        <span>MFU ≥ 50% → 100分，线性计算</span>
                      </div>
                      <div style={{ marginBottom: 8 }}>
                        <Text strong style={{ color: '#faad14' }}>效率评分：</Text>
                        <span>计算和显存利用率综合评估</span>
                      </div>
                      <div style={{ marginBottom: 8 }}>
                        <Text strong style={{ color: '#722ed1' }}>均衡评分：</Text>
                        <span>TP/PP/EP 均匀切分时得分高</span>
                      </div>
                      <div style={{
                        marginTop: 8,
                        padding: 8,
                        background: '#e6f7ff',
                        borderRadius: 4,
                        fontFamily: 'monospace',
                      }}>
                        综合 = {(DEFAULT_SCORE_WEIGHTS.latency * 100).toFixed(0)}%×延迟 + {(DEFAULT_SCORE_WEIGHTS.throughput * 100).toFixed(0)}%×吞吐 + {(DEFAULT_SCORE_WEIGHTS.efficiency * 100).toFixed(0)}%×效率 + {(DEFAULT_SCORE_WEIGHTS.balance * 100).toFixed(0)}%×均衡
                      </div>
                    </div>
                  ),
                }]}
              />
            </div>
          )}

          {/* 拓扑映射操作 */}
          {canMapToTopology && (
            <div style={{
              marginTop: 12,
              paddingTop: 12,
              borderTop: `1px dashed ${colors.borderLight}`,
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}>
              <Text style={{ fontSize: 11, color: colors.textSecondary }}>
                将并行策略映射到拓扑视图，查看通信流量分布
              </Text>
              <div style={{ display: 'flex', gap: 6 }}>
                <Button
                  size="small"
                  type="primary"
                  onClick={onMapToTopology}
                  style={{ fontSize: 11 }}
                >
                  映射到拓扑
                </Button>
                <Button
                  size="small"
                  onClick={onClearTraffic}
                  style={{ fontSize: 11 }}
                >
                  清除映射
                </Button>
              </div>
            </div>
          )}
        </>
        )}
      </div>

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* 二、模型架构 */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      {model && (
        <div style={sectionStyle}>
          <div
            style={{ ...sectionTitleStyle, cursor: 'pointer' }}
            onClick={() => toggleSection('model')}
          >
            模型架构
            <span style={{ marginLeft: 'auto' }}>
              {expandedSections.model ? <UpOutlined style={{ fontSize: 12 }} /> : <DownOutlined style={{ fontSize: 12 }} />}
            </span>
          </div>
          {expandedSections.model && <ModelInfoCard model={model} inference={inference} />}
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* 三、性能分析 */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      <div style={sectionStyle}>
        <div
          style={{ ...sectionTitleStyle, cursor: 'pointer' }}
          onClick={() => toggleSection('performance')}
        >
          性能分析
          <span style={{ marginLeft: 'auto' }}>
            {expandedSections.performance ? <UpOutlined style={{ fontSize: 12 }} /> : <DownOutlined style={{ fontSize: 12 }} />}
          </span>
        </div>

        {expandedSections.performance && (
        <>
        {/* 延迟指标 */}
        <Text style={{ fontSize: 11, color: colors.textSecondary, display: 'block', marginBottom: 6 }}>延迟</Text>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 6, marginBottom: 10 }}>
          <div style={metricCardStyle(selectedMetric === 'ttft')} onClick={() => setSelectedMetric(selectedMetric === 'ttft' ? null : 'ttft')}>
            <Text style={{ fontSize: 10, color: colors.textSecondary }}>TTFT</Text>
            <div style={{ fontSize: 15, fontWeight: 600, color: colors.text, marginTop: 2 }}>
              {latency.prefill_total_latency_ms.toFixed(1)} <span style={{ fontSize: 9, fontWeight: 400, color: colors.textSecondary }}>ms</span>
            </div>
          </div>
          <div style={metricCardStyle(selectedMetric === 'tpot')} onClick={() => setSelectedMetric(selectedMetric === 'tpot' ? null : 'tpot')}>
            <Text style={{ fontSize: 10, color: colors.textSecondary }}>TPOT</Text>
            <div style={{ fontSize: 15, fontWeight: 600, color: colors.text, marginTop: 2 }}>
              {latency.decode_per_token_latency_ms.toFixed(2)} <span style={{ fontSize: 9, fontWeight: 400, color: colors.textSecondary }}>ms</span>
            </div>
          </div>
          <div style={metricCardStyle(selectedMetric === 'e2e')} onClick={() => setSelectedMetric(selectedMetric === 'e2e' ? null : 'e2e')}>
            <Text style={{ fontSize: 10, color: colors.textSecondary }}>E2E</Text>
            <div style={{ fontSize: 15, fontWeight: 600, color: colors.text, marginTop: 2 }}>
              {(latency.end_to_end_latency_ms / 1000).toFixed(2)} <span style={{ fontSize: 9, fontWeight: 400, color: colors.textSecondary }}>s</span>
            </div>
          </div>
          <div style={metricCardStyle(selectedMetric === 'percentiles')} onClick={() => setSelectedMetric(selectedMetric === 'percentiles' ? null : 'percentiles')}>
            <Text style={{ fontSize: 10, color: colors.textSecondary }}>P99</Text>
            <div style={{ fontSize: 15, fontWeight: 600, color: latency.ttft_percentiles && latency.ttft_percentiles.p99 > 450 ? colors.error : colors.text, marginTop: 2 }}>
              {latency.ttft_percentiles ? latency.ttft_percentiles.p99.toFixed(0) : '-'} <span style={{ fontSize: 9, fontWeight: 400, color: colors.textSecondary }}>ms</span>
            </div>
          </div>
        </div>

        {/* 吞吐与效率 */}
        <Text style={{ fontSize: 11, color: colors.textSecondary, display: 'block', marginBottom: 6 }}>吞吐与效率</Text>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6, marginBottom: 10 }}>
          <div style={metricCardStyle(selectedMetric === 'throughput')} onClick={() => setSelectedMetric(selectedMetric === 'throughput' ? null : 'throughput')}>
            <Text style={{ fontSize: 10, color: colors.textSecondary }}>吞吐量</Text>
            <div style={{ fontSize: 15, fontWeight: 600, color: colors.text, marginTop: 2 }}>
              {throughput.tokens_per_second.toFixed(0)} <span style={{ fontSize: 9, fontWeight: 400, color: colors.textSecondary }}>tok/s</span>
            </div>
          </div>
          <div style={metricCardStyle(selectedMetric === 'mfu')} onClick={() => setSelectedMetric(selectedMetric === 'mfu' ? null : 'mfu')}>
            <Text style={{ fontSize: 10, color: colors.textSecondary }}>MFU</Text>
            <div style={{ fontSize: 15, fontWeight: 600, color: colors.text, marginTop: 2 }}>
              {(throughput.model_flops_utilization * 100).toFixed(1)} <span style={{ fontSize: 9, fontWeight: 400, color: colors.textSecondary }}>%</span>
            </div>
          </div>
          <div style={metricCardStyle(selectedMetric === 'mbu')} onClick={() => setSelectedMetric(selectedMetric === 'mbu' ? null : 'mbu')}>
            <Text style={{ fontSize: 10, color: colors.textSecondary }}>MBU</Text>
            <div style={{ fontSize: 15, fontWeight: 600, color: colors.text, marginTop: 2 }}>
              {(throughput.memory_bandwidth_utilization * 100).toFixed(1)} <span style={{ fontSize: 9, fontWeight: 400, color: colors.textSecondary }}>%</span>
            </div>
          </div>
        </div>

        {/* 资源利用 */}
        <Text style={{ fontSize: 11, color: colors.textSecondary, display: 'block', marginBottom: 6 }}>资源利用</Text>
        <div style={{
          padding: 12,
          background: '#fff',
          borderRadius: 8,
          marginBottom: 8,
          border: `1px solid ${colors.border}`,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <Text style={{ fontSize: 11, color: colors.textSecondary }}>显存</Text>
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
        <div
          style={metricCardStyle(selectedMetric === 'cost')}
          onClick={() => setSelectedMetric(selectedMetric === 'cost' ? null : 'cost')}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Text style={{ fontSize: 11, color: colors.textSecondary }}>推理成本</Text>
            <div style={{ fontSize: 15, fontWeight: 600, color: colors.text }}>
              ${result.cost ? result.cost.cost_per_million_tokens.toFixed(3) : '-'} <span style={{ fontSize: 10, fontWeight: 400, color: colors.textSecondary }}>/M tokens</span>
            </div>
          </div>
        </div>
        </>
        )}
      </div>

      {/* 指标详情展示 */}
      {selectedMetric && selectedMetric !== 'bottleneck' && (
        <MetricDetailCard metric={selectedMetric} result={result} />
      )}

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* 七、瓶颈与优化（四~六在 ChartsPanel 中渲染） */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      <div style={sectionStyle}>
        <div
          style={{ ...sectionTitleStyle, cursor: 'pointer' }}
          onClick={() => toggleSection('bottleneck')}
        >
          瓶颈分析
          <span style={{ marginLeft: 'auto' }}>
            {expandedSections.bottleneck ? <UpOutlined style={{ fontSize: 12 }} /> : <DownOutlined style={{ fontSize: 12 }} />}
          </span>
        </div>

        {/* 瓶颈 */}
        {expandedSections.bottleneck && (
        <div
          style={{
            padding: 12,
            background: selectedMetric === 'bottleneck' ? colors.warningLight : '#fff',
            borderRadius: 8,
            cursor: 'pointer',
            border: selectedMetric === 'bottleneck' ? `2px solid ${colors.warning}` : `1px solid ${colors.border}`,
            transition: 'all 0.2s ease',
          }}
          onClick={() => setSelectedMetric(selectedMetric === 'bottleneck' ? null : 'bottleneck')}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
            <Text strong style={{ fontSize: 13, color: colors.text }}>{latency.bottleneck_type}</Text>
            <InfoCircleOutlined style={{ fontSize: 12, color: selectedMetric === 'bottleneck' ? colors.warning : '#ccc' }} />
          </div>
          <Text style={{ fontSize: 11, color: colors.textSecondary }}>{latency.bottleneck_details}</Text>
        </div>
        )}
      </div>

      {/* 瓶颈详情展示 */}
      {selectedMetric === 'bottleneck' && (
        <MetricDetailCard metric="bottleneck" result={result} />
      )}

      {/* 优化建议 */}
      {suggestions.length > 0 && (
        <div style={sectionStyle}>
          <div
            style={{ ...sectionTitleStyle, cursor: 'pointer' }}
            onClick={() => toggleSection('suggestions')}
          >
            优化建议
            <span style={{ marginLeft: 'auto' }}>
              {expandedSections.suggestions ? <UpOutlined style={{ fontSize: 12 }} /> : <DownOutlined style={{ fontSize: 12 }} />}
            </span>
          </div>
          {expandedSections.suggestions && suggestions.slice(0, 3).map((s, i) => (
            <div key={i} style={{
              padding: 10,
              background: '#fff',
              borderRadius: 8,
              marginBottom: 8,
              borderLeft: `3px solid ${s.priority <= 2 ? colors.error : s.priority <= 3 ? colors.warning : colors.primary}`,
              border: `1px solid ${colors.border}`,
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

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* 八、候选方案 */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      {topKPlans.length > 1 && (
        <div style={sectionStyle}>
          <div
            style={{ ...sectionTitleStyle, cursor: 'pointer' }}
            onClick={() => toggleSection('candidates')}
          >
            候选方案
            <Tag color="default" style={{ marginLeft: 8, fontSize: 10 }}>{topKPlans.length}个</Tag>
            <span style={{ marginLeft: 'auto' }}>
              {expandedSections.candidates ? <UpOutlined style={{ fontSize: 12 }} /> : <DownOutlined style={{ fontSize: 12 }} />}
            </span>
          </div>
          {expandedSections.candidates && (
          <div style={{ maxHeight: 200, overflow: 'auto' }}>
            {topKPlans.map((p, i) => {
              const isSelected = p.plan.plan_id === result?.plan.plan_id
              return (
                <div
                  key={p.plan.plan_id}
                  onClick={() => onSelectPlan?.(p)}
                  style={{
                    padding: 10,
                    background: isSelected ? colors.primaryLight : '#fff',
                    borderRadius: 8,
                    marginBottom: 6,
                    cursor: 'pointer',
                    border: isSelected ? `2px solid ${colors.primary}` : `1px solid ${colors.border}`,
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
          )}
        </div>
      )}

    </div>
  )
}

export default AnalysisResultDisplay
