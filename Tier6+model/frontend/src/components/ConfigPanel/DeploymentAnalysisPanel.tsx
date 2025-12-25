/**
 * LLM 部署分析面板
 *
 * 提供模型配置、推理配置、硬件配置、并行策略配置和分析结果展示
 */

import React, { useState, useCallback } from 'react'
import {
  Typography,
  Button,
  InputNumber,
  Collapse,
  Select,
  Radio,
  Progress,
  Spin,
  Tag,
  Tooltip,
} from 'antd'
import {
  PlayCircleOutlined,
  SearchOutlined,
  InfoCircleOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  ApiOutlined,
  HeatMapOutlined,
} from '@ant-design/icons'
import {
  LLMModelConfig,
  InferenceConfig,
  HardwareConfig,
  ParallelismStrategy,
  PlanAnalysisResult,
  SearchConstraints,
  TopologyTrafficResult,
  ScoreWeights,
  DEFAULT_SCORE_WEIGHTS,
} from '../../utils/llmDeployment/types'
import { HierarchicalTopology } from '../../types'
import {
  MODEL_PRESETS,
  INFERENCE_PRESETS,
  getModelList,
  getChipList,
  getModelPreset,
  getInferencePreset,
  createHardwareConfig,
} from '../../utils/llmDeployment/presets'
import {
  analyzePlan,
} from '../../utils/llmDeployment/planAnalyzer'
import {
  searchWithFixedChips,
} from '../../utils/llmDeployment/planSearcher'
import {
  analyzeTopologyTraffic,
} from '../../utils/llmDeployment/trafficMapper'
import {
  extractChipGroupsFromConfig,
  generateHardwareConfig,
  ChipGroupInfo,
  TopologyHardwareSummary,
} from '../../utils/llmDeployment/topologyHardwareExtractor'
import { RackConfig } from './shared'

const { Text } = Typography
const { Panel } = Collapse

// ============================================
// 设计系统 - 样式常量
// ============================================

// 主色调
const colors = {
  primary: '#5E6AD2',
  primaryLight: '#E8EAFC',
  success: '#52c41a',
  successLight: '#f6ffed',
  warning: '#faad14',
  warningLight: '#fffbe6',
  error: '#ff4d4f',
  errorLight: '#fff2f0',
  border: '#E5E5E5',
  borderLight: '#F0F0F0',
  background: '#FAFAFA',
  backgroundDark: '#F5F5F5',
  text: '#1A1A1A',
  textSecondary: '#666666',
}

// 配置行样式
const configRowStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginBottom: 10,
}

// Section卡片样式
const sectionCardStyle: React.CSSProperties = {
  background: '#fff',
  borderRadius: 10,
  padding: 16,
  marginBottom: 12,
  boxShadow: '0 1px 3px rgba(0, 0, 0, 0.06)',
  border: `1px solid ${colors.borderLight}`,
}

// Section标题样式
const sectionTitleStyle: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 600,
  color: colors.text,
  marginBottom: 12,
  display: 'flex',
  alignItems: 'center',
  gap: 6,
}

const sectionStyle: React.CSSProperties = {
  marginBottom: 12,
}

// ============================================
// 自定义模型存储
// ============================================

const CUSTOM_MODELS_KEY = 'llm_custom_models'

function loadCustomModels(): Record<string, LLMModelConfig> {
  try {
    const data = localStorage.getItem(CUSTOM_MODELS_KEY)
    return data ? JSON.parse(data) : {}
  } catch {
    return {}
  }
}

function saveCustomModels(models: Record<string, LLMModelConfig>) {
  localStorage.setItem(CUSTOM_MODELS_KEY, JSON.stringify(models))
}

// ============================================
// 模型配置选择器
// ============================================

interface ModelConfigSelectorProps {
  value: LLMModelConfig
  onChange: (config: LLMModelConfig) => void
}

const ModelConfigSelector: React.FC<ModelConfigSelectorProps> = ({ value, onChange }) => {
  const [presetId, setPresetId] = useState<string>('deepseek-v3')
  const [editMode, setEditMode] = useState<boolean>(false)
  const [editBackup, setEditBackup] = useState<LLMModelConfig | null>(null)
  const [customModels, setCustomModels] = useState<Record<string, LLMModelConfig>>(loadCustomModels)
  const [saveModalVisible, setSaveModalVisible] = useState(false)
  const [saveName, setSaveName] = useState('')
  const modelList = getModelList()

  const handlePresetChange = (id: string) => {
    setPresetId(id)
    // 先检查自定义模型
    if (customModels[id]) {
      onChange({ ...customModels[id] })
    } else {
      onChange(getModelPreset(id))
    }
  }

  const handleSaveModel = () => {
    if (!saveName.trim()) return
    const customId = `custom_${saveName.trim().toLowerCase().replace(/\s+/g, '_')}`
    const newModels = { ...customModels, [customId]: { ...value, model_name: saveName.trim() } }
    setCustomModels(newModels)
    saveCustomModels(newModels)
    setPresetId(customId)
    setSaveModalVisible(false)
    setSaveName('')
  }

  const handleDeleteCustomModel = (id: string) => {
    const newModels = { ...customModels }
    delete newModels[id]
    setCustomModels(newModels)
    saveCustomModels(newModels)
    if (presetId === id) {
      setPresetId('deepseek-v3')
      onChange(getModelPreset('deepseek-v3'))
    }
  }

  // 合并预设模型和自定义模型列表
  const allModelOptions = [
    ...Object.entries(customModels).map(([id, config]) => ({
      value: id,
      label: `[自定义] ${config.model_name}`,
      isCustom: true,
    })),
    ...modelList.map(m => ({
      value: m.id,
      label: m.params ? `${m.name} (${m.params})` : m.name,
      isCustom: false,
    })),
  ]

  const updateField = <K extends keyof LLMModelConfig>(field: K, val: LLMModelConfig[K]) => {
    onChange({ ...value, [field]: val })
  }

  const updateMoeField = <K extends keyof NonNullable<LLMModelConfig['moe_config']>>(
    field: K,
    val: NonNullable<LLMModelConfig['moe_config']>[K]
  ) => {
    if (value.moe_config) {
      onChange({ ...value, moe_config: { ...value.moe_config, [field]: val } })
    }
  }

  const paramRowStyle: React.CSSProperties = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  }

  const isCustomModel = presetId.startsWith('custom_')

  // 估算参数量（返回参数量和计算公式）
  const estimateParams = () => {
    const H = value.hidden_size
    const L = value.num_layers
    const V = value.vocab_size
    const I = value.intermediate_size

    // 各部分参数量
    const embedding = V * H
    const attention = 4 * H * H * L  // Q, K, V, O 投影
    const layerNorm = 2 * H * L

    let ffn: number
    let formula: string
    let breakdown: string

    if (value.model_type === 'moe' && value.moe_config) {
      const E = value.moe_config.num_experts
      const S = value.moe_config.num_shared_experts || 0
      // 使用 expert_intermediate_size（如有），否则使用 intermediate_size
      const expertI = value.moe_config.expert_intermediate_size || I
      ffn = 3 * H * expertI * L * (E + S)
      const router = E * H * L  // 路由器参数
      const total = embedding + attention + ffn + layerNorm + router
      const billions = total / 1e9
      const result = billions >= 1 ? `${billions.toFixed(1)}B` : `${(total / 1e6).toFixed(0)}M`

      formula = `Embedding(V×H) + Attention(4×H²×L) + MoE_FFN(3×H×I'×L×E) + LayerNorm(2×H×L) + Router(E×H×L)`
      breakdown = [
        `Embedding: ${(embedding / 1e9).toFixed(2)}B`,
        `Attention: ${(attention / 1e9).toFixed(2)}B`,
        `MoE FFN: ${(ffn / 1e9).toFixed(2)}B (${E}+${S}专家, I'=${expertI})`,
        `LayerNorm: ${(layerNorm / 1e6).toFixed(1)}M`,
        `Router: ${(router / 1e6).toFixed(1)}M`,
        `─────────`,
        `总计: ${result}`,
      ].join('\n')

      return { value: result, formula, breakdown }
    } else {
      ffn = 3 * H * I * L  // gate, up, down (SwiGLU)
      const total = embedding + attention + ffn + layerNorm
      const billions = total / 1e9
      const result = billions >= 1 ? `${billions.toFixed(1)}B` : `${(total / 1e6).toFixed(0)}M`

      formula = `Embedding(V×H) + Attention(4×H²×L) + FFN(3×H×I×L) + LayerNorm(2×H×L)`
      breakdown = [
        `Embedding: ${(embedding / 1e9).toFixed(2)}B`,
        `Attention: ${(attention / 1e9).toFixed(2)}B`,
        `FFN: ${(ffn / 1e9).toFixed(2)}B`,
        `LayerNorm: ${(layerNorm / 1e6).toFixed(1)}M`,
        `─────────`,
        `总计: ${result}`,
      ].join('\n')

      return { value: result, formula, breakdown }
    }
  }

  return (
    <div>
      <div style={configRowStyle}>
        <Text>模型选择</Text>
        <Select
          size="small"
          value={presetId}
          onChange={handlePresetChange}
          style={{ width: '100%', maxWidth: 280 }}
          options={allModelOptions}
          optionRender={(option) => (
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>{option.label}</span>
              {option.data.isCustom && (
                <Button
                  type="text"
                  size="small"
                  danger
                  onClick={(e) => {
                    e.stopPropagation()
                    handleDeleteCustomModel(option.value as string)
                  }}
                  style={{ padding: '0 4px', height: 20, fontSize: 11 }}
                >
                  删除
                </Button>
              )}
            </div>
          )}
        />
      </div>

      {editMode ? (
        <div style={{ padding: 8, background: '#f5f5f5', borderRadius: 6, fontSize: 12 }}>
          {/* 基础参数 */}
          <div style={paramRowStyle}>
            <Tooltip title="自定义模型显示名称"><Text style={{ fontSize: 11, cursor: 'help' }}>模型名称</Text></Tooltip>
            <input
              value={value.model_name}
              onChange={(e) => updateField('model_name', e.target.value)}
              style={{ width: 180, fontSize: 11, padding: '2px 6px', border: '1px solid #d9d9d9', borderRadius: 4 }}
            />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Dense: 标准密集模型; MoE: 混合专家稀疏模型"><Text style={{ fontSize: 11, cursor: 'help' }}>模型类型</Text></Tooltip>
            <Select
              size="small"
              value={value.model_type}
              onChange={(v) => {
                if (v === 'moe' && !value.moe_config) {
                  onChange({ ...value, model_type: v, moe_config: { num_experts: 8, num_experts_per_tok: 2, expert_capacity_factor: 1.25 } })
                } else {
                  updateField('model_type', v)
                }
              }}
              style={{ width: 80 }}
              options={[
                { value: 'dense', label: 'Dense' },
                { value: 'moe', label: 'MoE' },
              ]}
            />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Hidden Size: 每个token的向量表示维度"><Text style={{ fontSize: 11, cursor: 'help' }}>隐藏层维度</Text></Tooltip>
            <InputNumber size="small" min={64} max={65536} value={value.hidden_size}
              onChange={(v) => updateField('hidden_size', v || 4096)} style={{ width: 80 }} />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Num Layers: Transformer层数"><Text style={{ fontSize: 11, cursor: 'help' }}>层数</Text></Tooltip>
            <InputNumber size="small" min={1} max={256} value={value.num_layers}
              onChange={(v) => updateField('num_layers', v || 32)} style={{ width: 80 }} />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Num Attention Heads: 多头注意力的Query头数"><Text style={{ fontSize: 11, cursor: 'help' }}>注意力头数</Text></Tooltip>
            <InputNumber size="small" min={1} max={256} value={value.num_attention_heads}
              onChange={(v) => updateField('num_attention_heads', v || 32)} style={{ width: 80 }} />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Num KV Heads: Key-Value头数，GQA时小于注意力头数"><Text style={{ fontSize: 11, cursor: 'help' }}>KV头数</Text></Tooltip>
            <InputNumber size="small" min={1} max={256} value={value.num_kv_heads}
              onChange={(v) => updateField('num_kv_heads', v || 8)} style={{ width: 80 }} />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Intermediate Size: FFN层中间维度，通常是hidden_size的2.5-4倍"><Text style={{ fontSize: 11, cursor: 'help' }}>FFN维度</Text></Tooltip>
            <InputNumber size="small" min={64} max={131072} value={value.intermediate_size}
              onChange={(v) => updateField('intermediate_size', v || 11008)} style={{ width: 80 }} />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Vocab Size: 词表大小，影响Embedding层参数量"><Text style={{ fontSize: 11, cursor: 'help' }}>词表大小</Text></Tooltip>
            <InputNumber size="small" min={1000} max={500000} value={value.vocab_size}
              onChange={(v) => updateField('vocab_size', v || 32000)} style={{ width: 80 }} />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="数据精度: 影响显存占用和计算速度"><Text style={{ fontSize: 11, cursor: 'help' }}>精度</Text></Tooltip>
            <Select size="small" value={value.dtype} onChange={(v) => updateField('dtype', v)} style={{ width: 80 }}
              options={[
                { value: 'fp32', label: 'FP32' },
                { value: 'fp16', label: 'FP16' },
                { value: 'bf16', label: 'BF16' },
                { value: 'int8', label: 'INT8' },
                { value: 'int4', label: 'INT4' },
              ]}
            />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Max Sequence Length: 模型支持的最大上下文长度"><Text style={{ fontSize: 11, cursor: 'help' }}>最大序列长度</Text></Tooltip>
            <InputNumber size="small" min={128} max={1048576} value={value.max_seq_length}
              onChange={(v) => updateField('max_seq_length', v || 4096)} style={{ width: 80 }} />
          </div>

          {/* MoE 参数 */}
          {value.model_type === 'moe' && value.moe_config && (
            <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid #e8e8e8' }}>
              <Tag color="purple" style={{ marginBottom: 6 }}>MoE 参数</Tag>
              <div style={paramRowStyle}>
                <Tooltip title="Num Experts: FFN层的专家总数"><Text style={{ fontSize: 11, cursor: 'help' }}>专家数量</Text></Tooltip>
                <InputNumber size="small" min={2} max={1024} value={value.moe_config.num_experts}
                  onChange={(v) => updateMoeField('num_experts', v || 8)} style={{ width: 80 }} />
              </div>
              <div style={paramRowStyle}>
                <Tooltip title="Top-K: 每个token激活的专家数量"><Text style={{ fontSize: 11, cursor: 'help' }}>激活专家数</Text></Tooltip>
                <InputNumber size="small" min={1} max={64} value={value.moe_config.num_experts_per_tok}
                  onChange={(v) => updateMoeField('num_experts_per_tok', v || 2)} style={{ width: 80 }} />
              </div>
              <div style={paramRowStyle}>
                <Tooltip title="Shared Experts: 所有token共享的专家数量（DeepSeek特有）"><Text style={{ fontSize: 11, cursor: 'help' }}>共享专家数</Text></Tooltip>
                <InputNumber size="small" min={0} max={16} value={value.moe_config.num_shared_experts || 0}
                  onChange={(v) => updateMoeField('num_shared_experts', v || 0)} style={{ width: 80 }} />
              </div>
              <div style={paramRowStyle}>
                <Tooltip title="Expert FFN Size: 每个专家的FFN中间维度（不设置则使用上方的FFN维度）"><Text style={{ fontSize: 11, cursor: 'help' }}>专家FFN维度</Text></Tooltip>
                <InputNumber size="small" min={64} max={65536} value={value.moe_config.expert_intermediate_size}
                  onChange={(v) => updateMoeField('expert_intermediate_size', v || undefined)} style={{ width: 80 }}
                  placeholder="同FFN" />
              </div>
            </div>
          )}
          {/* 编辑模式下也显示估算参数量 */}
          <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid #e8e8e8' }}>
            <Tooltip title={<pre style={{ margin: 0, fontSize: 11, whiteSpace: 'pre-wrap' }}>{estimateParams().breakdown}</pre>}>
              <Text type="secondary" style={{ cursor: 'help' }}>估算参数量: <b>{estimateParams().value}</b></Text>
            </Tooltip>
          </div>
        </div>
      ) : (
        <div style={{ padding: 8, background: '#f5f5f5', borderRadius: 6, fontSize: 12 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
            <Tooltip title="Hidden Size: 每个token的向量表示维度，决定模型的表示能力">
              <Text type="secondary" style={{ cursor: 'help' }}>隐藏层: {value.hidden_size}</Text>
            </Tooltip>
            <Tooltip title="Num Layers: Transformer层数，层数越多模型越深">
              <Text type="secondary" style={{ cursor: 'help' }}>层数: {value.num_layers}</Text>
            </Tooltip>
            <Tooltip title="Num Attention Heads: 多头注意力的头数，用于并行计算不同的注意力模式">
              <Text type="secondary" style={{ cursor: 'help' }}>注意力头: {value.num_attention_heads}</Text>
            </Tooltip>
            <Tooltip title="Num KV Heads: Key-Value头数，GQA/MQA时小于注意力头数可减少KV Cache">
              <Text type="secondary" style={{ cursor: 'help' }}>KV头: {value.num_kv_heads}</Text>
            </Tooltip>
            <Tooltip title="Intermediate Size: FFN层的中间维度，通常是隐藏层的2.5-4倍">
              <Text type="secondary" style={{ cursor: 'help' }}>FFN: {value.intermediate_size}</Text>
            </Tooltip>
            <Tooltip title="数据精度: FP32/FP16/BF16/INT8/INT4，精度越低显存占用越少">
              <Text type="secondary" style={{ cursor: 'help' }}>精度: {value.dtype.toUpperCase()}</Text>
            </Tooltip>
          </div>
          {value.model_type === 'moe' && value.moe_config && (
            <div style={{ marginTop: 4, paddingTop: 4, borderTop: '1px solid #e8e8e8' }}>
              <Tooltip title="Mixture of Experts: 稀疏激活架构，每次只激活部分专家，提高模型容量的同时控制计算量">
                <Tag color="purple" style={{ cursor: 'help' }}>MoE</Tag>
              </Tooltip>
              <Tooltip title={`总共${value.moe_config.num_experts}个专家，每个token激活${value.moe_config.num_experts_per_tok}个`}>
                <Text type="secondary" style={{ fontSize: 11, cursor: 'help' }}>
                  {value.moe_config.num_experts}专家 × {value.moe_config.num_experts_per_tok}激活
                  {value.moe_config.num_shared_experts ? ` + ${value.moe_config.num_shared_experts}共享` : ''}
                </Text>
              </Tooltip>
            </div>
          )}
          <div style={{ marginTop: 6, paddingTop: 6, borderTop: '1px solid #e8e8e8', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Tooltip title={<pre style={{ margin: 0, fontSize: 11, whiteSpace: 'pre-wrap' }}>{estimateParams().breakdown}</pre>}>
              <Text type="secondary" style={{ cursor: 'help' }}>估算参数量: <b>{estimateParams().value}</b></Text>
            </Tooltip>
          </div>
        </div>
      )}

      {/* 操作按钮 */}
      <div style={{ display: 'flex', gap: 4, marginTop: 8 }}>
        {editMode && (
          <Button
            size="small"
            onClick={() => {
              if (editBackup) {
                onChange(editBackup)
              }
              setEditMode(false)
              setEditBackup(null)
            }}
            style={{ flex: 1 }}
          >
            取消
          </Button>
        )}
        <Button
          size="small"
          type={editMode ? 'primary' : 'default'}
          onClick={() => {
            if (!editMode) {
              setEditBackup({ ...value })
            }
            setEditMode(!editMode)
          }}
          style={{ flex: 1 }}
        >
          {editMode ? '完成编辑' : '编辑模型参数'}
        </Button>
        {isCustomModel && (
          <Button
            size="small"
            type="primary"
            onClick={() => {
              // 直接覆盖保存当前自定义模型
              const newModels = { ...customModels, [presetId]: { ...value } }
              setCustomModels(newModels)
              saveCustomModels(newModels)
            }}
            style={{ flex: 1 }}
          >
            保存
          </Button>
        )}
        {editMode && (
          <Button
            size="small"
            onClick={() => {
              setSaveName(value.model_name)
              setSaveModalVisible(true)
            }}
            style={{ flex: 1 }}
          >
            另存为
          </Button>
        )}
        {isCustomModel && (
          <Button
            size="small"
            danger
            onClick={() => handleDeleteCustomModel(presetId)}
          >
            删除
          </Button>
        )}
      </div>

      {/* 保存弹窗 */}
      {saveModalVisible && (
        <div style={{
          padding: 12,
          background: '#fff',
          border: '1px solid #d9d9d9',
          borderRadius: 6,
          marginTop: 8,
        }}>
          <div style={{ marginBottom: 8 }}>
            <Text style={{ fontSize: 12 }}>输入自定义模型名称：</Text>
          </div>
          <input
            value={saveName}
            onChange={(e) => setSaveName(e.target.value)}
            placeholder="如: My-Custom-Model"
            style={{
              width: '100%',
              padding: '6px 8px',
              border: '1px solid #d9d9d9',
              borderRadius: 4,
              marginBottom: 8,
            }}
            onKeyDown={(e) => e.key === 'Enter' && handleSaveModel()}
          />
          <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
            <Button size="small" onClick={() => setSaveModalVisible(false)}>取消</Button>
            <Button size="small" type="primary" onClick={handleSaveModel}>保存</Button>
          </div>
        </div>
      )}
    </div>
  )
}

// ============================================
// 推理配置选择器
// ============================================

interface InferenceConfigSelectorProps {
  value: InferenceConfig
  onChange: (config: InferenceConfig) => void
}

const CUSTOM_INFERENCE_KEY = 'llm_custom_inference'

function loadCustomInference(): Record<string, InferenceConfig & { name: string }> {
  try {
    const data = localStorage.getItem(CUSTOM_INFERENCE_KEY)
    return data ? JSON.parse(data) : {}
  } catch {
    return {}
  }
}

function saveCustomInference(configs: Record<string, InferenceConfig & { name: string }>) {
  localStorage.setItem(CUSTOM_INFERENCE_KEY, JSON.stringify(configs))
}

const InferenceConfigSelector: React.FC<InferenceConfigSelectorProps> = ({ value, onChange }) => {
  const [presetId, setPresetId] = useState<string>('standard')
  const [customConfigs, setCustomConfigs] = useState(loadCustomInference)
  const [saveModalVisible, setSaveModalVisible] = useState(false)
  const [saveName, setSaveName] = useState('')

  const presetOptions = [
    { value: 'low-latency', label: '低延迟交互', isCustom: false },
    { value: 'standard', label: '标准推理', isCustom: false },
    { value: 'high-throughput', label: '高吞吐批处理', isCustom: false },
    { value: 'long-context', label: '长上下文', isCustom: false },
    { value: 'code-gen', label: '代码生成', isCustom: false },
  ]

  const allOptions = [
    ...Object.entries(customConfigs).map(([id, cfg]) => ({
      value: id,
      label: `[自定义] ${cfg.name}`,
      isCustom: true,
    })),
    ...presetOptions,
  ]

  const handlePresetChange = (id: string) => {
    setPresetId(id)
    if (customConfigs[id]) {
      const { name, ...config } = customConfigs[id]
      onChange(config)
    } else {
      onChange(getInferencePreset(id))
    }
  }

  const handleSave = () => {
    if (!saveName.trim()) return
    const customId = `custom_inf_${saveName.trim().toLowerCase().replace(/\s+/g, '_')}`
    const newConfigs = { ...customConfigs, [customId]: { ...value, name: saveName.trim() } }
    setCustomConfigs(newConfigs)
    saveCustomInference(newConfigs)
    setPresetId(customId)
    setSaveModalVisible(false)
    setSaveName('')
  }

  const handleDelete = (id: string) => {
    const newConfigs = { ...customConfigs }
    delete newConfigs[id]
    setCustomConfigs(newConfigs)
    saveCustomInference(newConfigs)
    if (presetId === id) {
      setPresetId('standard')
      onChange(getInferencePreset('standard'))
    }
  }

  const isCustom = presetId.startsWith('custom_inf_')

  return (
    <div>
      <div style={configRowStyle}>
        <Text>场景预设</Text>
        <Select
          size="small"
          value={presetId}
          onChange={handlePresetChange}
          style={{ width: 160 }}
          options={allOptions}
          optionRender={(option) => (
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>{option.label}</span>
              {option.data.isCustom && (
                <Button
                  type="text"
                  size="small"
                  danger
                  onClick={(e) => {
                    e.stopPropagation()
                    handleDelete(option.value as string)
                  }}
                  style={{ padding: '0 4px', height: 20, fontSize: 11 }}
                >
                  删除
                </Button>
              )}
            </div>
          )}
        />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 8 }}>
        <div style={configRowStyle}>
          <Tooltip title="Batch Size: 同时处理的请求数量，影响吞吐量和延迟">
            <Text style={{ fontSize: 12, cursor: 'help' }}>Batch Size</Text>
          </Tooltip>
          <InputNumber
            size="small"
            min={1}
            max={128}
            value={value.batch_size}
            onChange={(v) => onChange({ ...value, batch_size: v || 1 })}
            style={{ width: 70 }}
          />
        </div>
        <div style={configRowStyle}>
          <Tooltip title="Input Length: 输入提示词的token数量（如问题、上下文）">
            <Text style={{ fontSize: 12, cursor: 'help' }}>Input Length</Text>
          </Tooltip>
          <InputNumber
            size="small"
            min={1}
            max={131072}
            value={value.input_seq_length}
            onChange={(v) => onChange({ ...value, input_seq_length: v || 512 })}
            style={{ width: 70 }}
          />
        </div>
        <div style={configRowStyle}>
          <Tooltip title="Output Length: 生成输出的token数量（如回答、代码）">
            <Text style={{ fontSize: 12, cursor: 'help' }}>Output Length</Text>
          </Tooltip>
          <InputNumber
            size="small"
            min={1}
            max={32768}
            value={value.output_seq_length}
            onChange={(v) => onChange({ ...value, output_seq_length: v || 256 })}
            style={{ width: 70 }}
          />
        </div>
        <div style={configRowStyle}>
          <Tooltip title="Max Sequence Length: KV Cache预分配的最大长度，通常≥输入+输出">
            <Text style={{ fontSize: 12, cursor: 'help' }}>Max Seq Length</Text>
          </Tooltip>
          <InputNumber
            size="small"
            min={value.input_seq_length + value.output_seq_length}
            max={131072}
            value={value.max_seq_length}
            onChange={(v) => onChange({ ...value, max_seq_length: v || 768 })}
            style={{ width: 70 }}
          />
        </div>
      </div>

      {/* 保存按钮 */}
      <div style={{ display: 'flex', gap: 4, marginTop: 8 }}>
        {isCustom && (
          <Button
            size="small"
            type="primary"
            onClick={() => {
              const newConfigs = { ...customConfigs, [presetId]: { ...value, name: customConfigs[presetId].name } }
              setCustomConfigs(newConfigs)
              saveCustomInference(newConfigs)
            }}
            style={{ flex: 1 }}
          >
            保存
          </Button>
        )}
        <Button
          size="small"
          onClick={() => {
            setSaveName('')
            setSaveModalVisible(true)
          }}
          style={{ flex: 1 }}
        >
          另存为
        </Button>
      </div>

      {/* 保存弹窗 */}
      {saveModalVisible && (
        <div style={{
          padding: 12,
          background: '#fff',
          border: '1px solid #d9d9d9',
          borderRadius: 6,
          marginTop: 8,
        }}>
          <div style={{ marginBottom: 8 }}>
            <Text style={{ fontSize: 12 }}>输入配置名称：</Text>
          </div>
          <input
            value={saveName}
            onChange={(e) => setSaveName(e.target.value)}
            placeholder="如: RAG场景"
            style={{
              width: '100%',
              padding: '6px 8px',
              border: '1px solid #d9d9d9',
              borderRadius: 4,
              marginBottom: 8,
            }}
            onKeyDown={(e) => e.key === 'Enter' && handleSave()}
          />
          <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
            <Button size="small" onClick={() => setSaveModalVisible(false)}>取消</Button>
            <Button size="small" type="primary" onClick={handleSave}>保存</Button>
          </div>
        </div>
      )}
    </div>
  )
}

// ============================================
// 硬件配置选择器
// ============================================

interface HardwareConfigSelectorProps {
  value: HardwareConfig
  onChange: (config: HardwareConfig) => void
}

const HardwareConfigSelector: React.FC<HardwareConfigSelectorProps> = ({ value, onChange }) => {
  const [chipId, setChipId] = useState<string>('h100-sxm')
  const [nodeId, setNodeId] = useState<string>('dgx-h100')
  const [numNodes, setNumNodes] = useState<number>(1)

  const chipList = getChipList()
  const nodeOptions = [
    { value: 'dgx-h100', label: 'DGX H100 (8卡 NVLink)' },
    { value: 'dgx-a100', label: 'DGX A100 (8卡 NVLink)' },
    { value: 'pcie-8gpu', label: '通用 PCIe (8卡)' },
  ]

  const handleConfigChange = useCallback((newChipId: string, newNodeId: string, newNumNodes: number) => {
    const config = createHardwareConfig(newChipId, newNodeId, newNumNodes, 400)
    onChange(config)
  }, [onChange])

  const handleChipChange = (id: string) => {
    setChipId(id)
    handleConfigChange(id, nodeId, numNodes)
  }

  const handleNodeChange = (id: string) => {
    setNodeId(id)
    handleConfigChange(chipId, id, numNodes)
  }

  const handleNumNodesChange = (n: number) => {
    setNumNodes(n)
    handleConfigChange(chipId, nodeId, n)
  }

  const totalChips = value.node.chips_per_node * value.cluster.num_nodes

  return (
    <div>
      <div style={configRowStyle}>
        <Text>芯片类型</Text>
        <Select
          size="small"
          value={chipId}
          onChange={handleChipChange}
          style={{ width: 140 }}
          options={chipList.map(c => ({
            value: c.id,
            label: `${c.name}`,
          }))}
        />
      </div>
      <div style={configRowStyle}>
        <Text>节点类型</Text>
        <Select
          size="small"
          value={nodeId}
          onChange={handleNodeChange}
          style={{ width: 160 }}
          options={nodeOptions}
        />
      </div>
      <div style={configRowStyle}>
        <Text>节点数量</Text>
        <InputNumber
          size="small"
          min={1}
          max={64}
          value={numNodes}
          onChange={(v) => handleNumNodesChange(v || 1)}
          style={{ width: 70 }}
        />
      </div>
      <div style={{ padding: 8, background: '#f0f5ff', borderRadius: 6, fontSize: 12, marginTop: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <Text type="secondary">总芯片数: <b>{totalChips}</b></Text>
          <Text type="secondary">显存: <b>{value.chip.memory_gb}GB</b>/卡</Text>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
          <Text type="secondary">算力: {value.chip.compute_tflops_fp16} TFLOPs</Text>
          <Text type="secondary">节点内: {value.node.intra_node_bandwidth_gbps} GB/s</Text>
        </div>
      </div>
    </div>
  )
}

// ============================================
// 并行策略配置
// ============================================

// 优化目标对应的预设权重
const OPTIMIZATION_TARGET_WEIGHTS: Record<string, ScoreWeights> = {
  latency: { latency: 0.7, throughput: 0.15, efficiency: 0.1, balance: 0.05 },
  throughput: { latency: 0.1, throughput: 0.7, efficiency: 0.15, balance: 0.05 },
  efficiency: { latency: 0.1, throughput: 0.2, efficiency: 0.6, balance: 0.1 },
  balanced: { ...DEFAULT_SCORE_WEIGHTS },
}

// 自定义权重存储
const CUSTOM_WEIGHTS_KEY = 'llm_custom_score_weights'

function loadCustomWeights(): ScoreWeights | null {
  try {
    const data = localStorage.getItem(CUSTOM_WEIGHTS_KEY)
    return data ? JSON.parse(data) : null
  } catch {
    return null
  }
}

function saveCustomWeights(weights: ScoreWeights) {
  localStorage.setItem(CUSTOM_WEIGHTS_KEY, JSON.stringify(weights))
}

interface ParallelismConfigPanelProps {
  mode: 'manual' | 'auto'
  onModeChange: (mode: 'manual' | 'auto') => void
  manualStrategy: ParallelismStrategy
  onManualStrategyChange: (strategy: ParallelismStrategy) => void
  searchConstraints: SearchConstraints
  onSearchConstraintsChange: (constraints: SearchConstraints) => void
  maxChips: number
  scoreWeights: ScoreWeights
  onScoreWeightsChange: (weights: ScoreWeights) => void
}

const ParallelismConfigPanel: React.FC<ParallelismConfigPanelProps> = ({
  mode,
  onModeChange,
  manualStrategy,
  onManualStrategyChange,
  searchConstraints,
  onSearchConstraintsChange,
  maxChips,
  scoreWeights,
  onScoreWeightsChange,
}) => {
  const totalParallelism = manualStrategy.dp * manualStrategy.tp * manualStrategy.pp * manualStrategy.ep
  const [showWeights, setShowWeights] = React.useState(false)
  const [hasCustomWeights, setHasCustomWeights] = React.useState(() => loadCustomWeights() !== null)

  // 计算权重总和
  const weightSum = scoreWeights.latency + scoreWeights.throughput + scoreWeights.efficiency + scoreWeights.balance

  // 当前优化目标
  const currentTarget = (searchConstraints as any).optimization_target || 'balanced'

  // 优化目标变化时，更新权重
  const handleTargetChange = (target: string) => {
    onSearchConstraintsChange({ ...searchConstraints, optimization_target: target as any })
    // 如果不是自定义，使用预设权重
    if (target !== 'custom') {
      onScoreWeightsChange(OPTIMIZATION_TARGET_WEIGHTS[target] || DEFAULT_SCORE_WEIGHTS)
    } else {
      // 使用保存的自定义权重
      const saved = loadCustomWeights()
      if (saved) {
        onScoreWeightsChange(saved)
      }
    }
  }

  // 保存当前权重为自定义
  const handleSaveCustomWeights = () => {
    saveCustomWeights(scoreWeights)
    setHasCustomWeights(true)
  }

  return (
    <div>
      <div style={{ marginBottom: 12 }}>
        <Radio.Group
          size="small"
          value={mode}
          onChange={(e) => onModeChange(e.target.value)}
          buttonStyle="solid"
        >
          <Radio.Button value="manual">手动指定</Radio.Button>
          <Radio.Button value="auto">自动搜索</Radio.Button>
        </Radio.Group>
      </div>

      {mode === 'manual' ? (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 4 }}>
            {(['dp', 'tp', 'pp', 'ep', 'sp'] as const).map((key) => (
              <div key={key} style={{ textAlign: 'center' }}>
                <Text style={{ fontSize: 11, display: 'block' }}>{key.toUpperCase()}</Text>
                <InputNumber
                  size="small"
                  min={1}
                  max={64}
                  value={manualStrategy[key]}
                  onChange={(v) => onManualStrategyChange({ ...manualStrategy, [key]: v || 1 })}
                  style={{ width: '100%' }}
                />
              </div>
            ))}
          </div>
          <div style={{ marginTop: 8, textAlign: 'center' }}>
            <Text type={totalParallelism > maxChips ? 'danger' : 'secondary'} style={{ fontSize: 12 }}>
              总并行度: {totalParallelism} / {maxChips} 芯片
            </Text>
          </div>
        </div>
      ) : (
        <div>
          <div style={configRowStyle}>
            <Text style={{ fontSize: 12 }}>最大芯片数</Text>
            <InputNumber
              size="small"
              min={1}
              max={1024}
              value={searchConstraints.max_chips || maxChips}
              onChange={(v) => onSearchConstraintsChange({ ...searchConstraints, max_chips: v || maxChips })}
              style={{ width: 80 }}
            />
          </div>
          <div style={configRowStyle}>
            <Text style={{ fontSize: 12 }}>优化目标</Text>
            <Select
              size="small"
              value={currentTarget}
              onChange={handleTargetChange}
              style={{ width: 110 }}
              options={[
                { value: 'latency', label: '低延迟' },
                { value: 'throughput', label: '高吞吐' },
                { value: 'efficiency', label: '高效率' },
                { value: 'balanced', label: '均衡' },
                ...(hasCustomWeights ? [{ value: 'custom', label: '自定义' }] : []),
              ]}
            />
          </div>

          {/* 评分权重配置 - 仅在自动搜索模式显示 */}
          <div style={{ marginTop: 12 }}>
            <div
              style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
              onClick={() => setShowWeights(!showWeights)}
            >
              <Text style={{ fontSize: 12 }}>评分权重</Text>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <Text type="secondary" style={{ fontSize: 10 }}>
                  延迟:{scoreWeights.latency} 吞吐:{scoreWeights.throughput}
                </Text>
                <Text type="secondary" style={{ fontSize: 11 }}>{showWeights ? '▲' : '▼'}</Text>
              </div>
            </div>

            {showWeights && (
              <div style={{ marginTop: 8, padding: 8, background: '#fafafa', borderRadius: 6 }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                  <div>
                    <Tooltip title="首Token延迟(TTFT)的重要性权重">
                      <Text style={{ fontSize: 11, cursor: 'help' }}>延迟</Text>
                    </Tooltip>
                    <InputNumber
                      size="small"
                      min={0}
                      max={1}
                      step={0.05}
                      value={scoreWeights.latency}
                      onChange={(v) => onScoreWeightsChange({ ...scoreWeights, latency: v || 0 })}
                      style={{ width: '100%' }}
                    />
                  </div>
                  <div>
                    <Tooltip title="Token吞吐量和MFU的重要性权重">
                      <Text style={{ fontSize: 11, cursor: 'help' }}>吞吐</Text>
                    </Tooltip>
                    <InputNumber
                      size="small"
                      min={0}
                      max={1}
                      step={0.05}
                      value={scoreWeights.throughput}
                      onChange={(v) => onScoreWeightsChange({ ...scoreWeights, throughput: v || 0 })}
                      style={{ width: '100%' }}
                    />
                  </div>
                  <div>
                    <Tooltip title="计算和显存利用率的重要性权重">
                      <Text style={{ fontSize: 11, cursor: 'help' }}>效率</Text>
                    </Tooltip>
                    <InputNumber
                      size="small"
                      min={0}
                      max={1}
                      step={0.05}
                      value={scoreWeights.efficiency}
                      onChange={(v) => onScoreWeightsChange({ ...scoreWeights, efficiency: v || 0 })}
                      style={{ width: '100%' }}
                    />
                  </div>
                  <div>
                    <Tooltip title="负载均衡程度的重要性权重">
                      <Text style={{ fontSize: 11, cursor: 'help' }}>均衡</Text>
                    </Tooltip>
                    <InputNumber
                      size="small"
                      min={0}
                      max={1}
                      step={0.05}
                      value={scoreWeights.balance}
                      onChange={(v) => onScoreWeightsChange({ ...scoreWeights, balance: v || 0 })}
                      style={{ width: '100%' }}
                    />
                  </div>
                </div>
                <div style={{ marginTop: 6, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Text type={Math.abs(weightSum - 1) > 0.01 ? 'danger' : 'secondary'} style={{ fontSize: 10 }}>
                    权重总和: {weightSum.toFixed(2)} {Math.abs(weightSum - 1) > 0.01 && '(建议=1.0)'}
                  </Text>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <Button
                      size="small"
                      type="link"
                      style={{ fontSize: 10, padding: 0 }}
                      onClick={handleSaveCustomWeights}
                    >
                      保存为自定义
                    </Button>
                    <Button
                      size="small"
                      type="link"
                      style={{ fontSize: 10, padding: 0 }}
                      onClick={() => onScoreWeightsChange(OPTIMIZATION_TARGET_WEIGHTS[currentTarget] || DEFAULT_SCORE_WEIGHTS)}
                    >
                      重置
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ============================================
// 分析结果展示
// ============================================

interface AnalysisResultDisplayProps {
  result: PlanAnalysisResult | null
  topKPlans: PlanAnalysisResult[]
  loading: boolean
  onSelectPlan?: (plan: PlanAnalysisResult) => void
  searchStats?: { evaluated: number; feasible: number; timeMs: number } | null
  errorMsg?: string | null
}

type MetricType = 'ttft' | 'tpot' | 'throughput' | 'mfu' | 'bottleneck' | null

const AnalysisResultDisplay: React.FC<AnalysisResultDisplayProps> = ({ result, topKPlans, loading, onSelectPlan, searchStats, errorMsg }) => {
  const [selectedMetric, setSelectedMetric] = React.useState<MetricType>(null)

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 20 }}>
        <Spin />
        <div style={{ marginTop: 8 }}>
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

  if (!result) {
    return (
      <div style={{ textAlign: 'center', padding: 20, color: '#999' }}>
        <InfoCircleOutlined style={{ fontSize: 24, marginBottom: 8 }} />
        <div>点击"运行分析"查看结果</div>
      </div>
    )
  }

  const { plan, memory, latency, throughput, score, suggestions, is_feasible, infeasibility_reason } = result

  // 分数组件 - 用于显示数学分数（使用CSS Grid确保横线和内容等宽）
  const Fraction: React.FC<{ top: React.ReactNode; bottom: React.ReactNode; large?: boolean }> = ({ top, bottom, large }) => (
    <span style={{
      display: 'inline-grid',
      gridTemplateRows: 'auto auto auto',
      justifyItems: 'center',
      alignItems: 'center',
      verticalAlign: 'middle',
      margin: '0 8px',
      lineHeight: 1.4,
    }}>
      <span style={{
        padding: '0 6px 4px 6px',
        fontSize: large ? 18 : 15,
        textAlign: 'center',
        whiteSpace: 'nowrap',
      }}>{top}</span>
      <span style={{
        width: '100%',
        height: 2,
        background: '#333',
      }} />
      <span style={{
        padding: '4px 6px 0 6px',
        fontSize: large ? 18 : 15,
        textAlign: 'center',
        whiteSpace: 'nowrap',
      }}>{bottom}</span>
    </span>
  )

  // 生成指标详情
  const getMetricDetail = (metric: MetricType) => {
    if (!metric) return null

    // 详情卡片样式 - 与整体设计一致
    const cardStyle: React.CSSProperties = {
      marginTop: 10,
      padding: 14,
      background: colors.primaryLight,
      borderRadius: 10,
      border: `1px solid ${colors.primary}`,
      fontSize: 13,
    }

    const definitionStyle: React.CSSProperties = {
      background: '#fff',
      padding: '10px 12px',
      borderRadius: 8,
      marginBottom: 12,
      borderLeft: `3px solid ${colors.primary}`,
      fontSize: 12,
      color: colors.text,
    }

    const formulaStyle: React.CSSProperties = {
      fontFamily: '"Times New Roman", "CMU Serif", Georgia, serif',
      fontSize: 18,
      fontStyle: 'italic',
      background: '#fff',
      padding: '14px 16px',
      borderRadius: 8,
      textAlign: 'center' as const,
      marginBottom: 12,
      border: `1px solid ${colors.border}`,
      lineHeight: 1.8,
      color: colors.text,
    }

    const valueRowStyle: React.CSSProperties = {
      display: 'flex',
      justifyContent: 'space-between',
      marginBottom: 4,
      fontSize: 13,
    }

    const subFormulaStyle: React.CSSProperties = {
      fontSize: 13,
      color: colors.textSecondary,
      marginLeft: 16,
      marginBottom: 6,
      fontFamily: '"Times New Roman", Georgia, serif',
      fontStyle: 'italic',
      lineHeight: 1.6,
    }

    // 变量定义样式
    const varDefStyle: React.CSSProperties = {
      fontSize: 11,
      color: colors.textSecondary,
      background: '#fff',
      padding: '10px 12px',
      borderRadius: 8,
      marginBottom: 12,
      lineHeight: 1.8,
    }

    switch (metric) {
      case 'ttft':
        return (
          <div style={cardStyle}>
            <Text strong style={{ fontSize: 14, display: 'block', marginBottom: 10, color: colors.primary }}>TTFT (Time To First Token)</Text>
            <div style={definitionStyle}>
              <Text>
                <strong>定义：</strong>从请求发送到生成第一个输出Token的延迟时间，即Prefill阶段的总耗时。
                是衡量LLM响应速度的关键指标，直接影响用户体验。
              </Text>
            </div>
            <div style={formulaStyle}>
              TTFT = <Fraction top={<>max(T<sub>compute</sub>, T<sub>comm</sub>)</>} bottom="1 − β" large />
            </div>
            <div style={varDefStyle}>
              <div><b>变量说明：</b></div>
              <div>• <b>T<sub>compute</sub></b> - 计算延迟：处理输入序列所需的计算时间</div>
              <div>• <b>T<sub>comm</sub></b> - 通信延迟：TP/PP并行时芯片间AllReduce等通信的时间</div>
              <div>• <b>β</b> - 流水线气泡比：PP并行时因流水线填充/排空产生的空闲比例</div>
              <div>• <b>FLOPs</b> - Prefill阶段处理整个输入序列的总浮点运算量</div>
              <div>• <b>Peak</b> - 单芯片FP16峰值算力 (TFLOPs)</div>
              <div>• <b>Utilization</b> - 硬件利用率，实际计算效率与峰值的比值（通常0.5-0.7）</div>
              <div>• <b>TP</b> - 张量并行度：将模型权重切分到多个芯片</div>
              <div>• <b>PP</b> - 流水线并行度：将模型层切分到多个芯片</div>
              <div>• <b>Data Size</b> - 单次通信的数据量 (bytes)</div>
              <div>• <b>Bandwidth</b> - 芯片间互联带宽 (GB/s)</div>
              <div>• <b>Startup</b> - 通信启动延迟 (μs)</div>
              <div>• <b>Micro-batches</b> - 微批次数量，用于填充流水线</div>
            </div>
            <div style={{ marginTop: 12, fontSize: 14 }}>
              <div style={subFormulaStyle}>
                T<sub>compute</sub> = <Fraction top="FLOPs" bottom="Peak × Utilization × TP × PP" /> = <b>{latency.prefill_compute_latency_ms.toFixed(2)} ms</b>
              </div>
              <div style={subFormulaStyle}>
                T<sub>comm</sub> = Σ(<Fraction top="Data Size" bottom="Bandwidth" /> + Startup) = <b>{latency.prefill_comm_latency_ms.toFixed(2)} ms</b>
              </div>
              <div style={subFormulaStyle}>
                β = <Fraction top="PP − 1" bottom="Micro-batches + PP − 1" /> = <b>{(latency.pipeline_bubble_ratio * 100).toFixed(1)}%</b>
              </div>
              <div style={{ borderTop: '1px solid #d9d9d9', marginTop: 12, paddingTop: 12 }}>
                <div style={valueRowStyle}>
                  <Text strong style={{ fontSize: 15 }}>TTFT =</Text>
                  <Text strong style={{ fontSize: 16, color: '#1890ff' }}>{latency.prefill_total_latency_ms.toFixed(2)} ms</Text>
                </div>
              </div>
            </div>
          </div>
        )
      case 'tpot':
        return (
          <div style={cardStyle}>
            <Text strong style={{ fontSize: 14, display: 'block', marginBottom: 10, color: colors.primary }}>TPOT (Time Per Output Token)</Text>
            <div style={definitionStyle}>
              <Text>
                <strong>定义：</strong>Decode阶段生成每个输出Token的平均延迟。
                决定了文本生成的"打字速度"，TPOT越小，用户看到的文本流越快。
              </Text>
            </div>
            <div style={formulaStyle}>
              TPOT = max(T<sub>compute</sub>, T<sub>memory</sub>) + T<sub>comm</sub>
            </div>
            <div style={varDefStyle}>
              <div><b>变量说明：</b></div>
              <div>• <b>T<sub>compute</sub></b> - 计算延迟：生成单个Token的矩阵运算时间</div>
              <div>• <b>T<sub>memory</sub></b> - 访存延迟：从显存读取权重和KV Cache的时间（Decode阶段通常是瓶颈）</div>
              <div>• <b>T<sub>comm</sub></b> - 通信延迟：TP并行时的AllReduce通信时间</div>
              <div>• <b>FLOPs per Token</b> - 生成单个Token所需的浮点运算量</div>
              <div>• <b>Peak</b> - 单芯片FP16峰值算力 (TFLOPs)</div>
              <div>• <b>Utilization</b> - 硬件利用率（通常0.5-0.7）</div>
              <div>• <b>Weights</b> - 模型权重大小 (GB)，按TP切分</div>
              <div>• <b>KV Cache</b> - 键值缓存：2 × layers × kv_heads × head_dim × seq_len × bytes</div>
              <div>• <b>Memory Bandwidth</b> - 显存带宽，如H100为3.35 TB/s</div>
            </div>
            <div style={{ marginTop: 12, fontSize: 14 }}>
              <div style={subFormulaStyle}>
                T<sub>compute</sub> = <Fraction top="FLOPs per Token" bottom="Peak × Utilization" /> = <b>{latency.decode_compute_latency_ms.toFixed(3)} ms</b>
              </div>
              <div style={subFormulaStyle}>
                T<sub>memory</sub> = <Fraction top="Weights + KV Cache" bottom="Memory Bandwidth" /> = <b>~{(memory.model_memory_gb / 3.35).toFixed(2)} ms</b>
              </div>
              <div style={subFormulaStyle}>
                T<sub>comm</sub> = <b>{latency.decode_comm_latency_ms.toFixed(3)} ms</b>
              </div>
              <div style={{ borderTop: '1px solid #d9d9d9', marginTop: 12, paddingTop: 12 }}>
                <div style={valueRowStyle}>
                  <Text strong style={{ fontSize: 15 }}>TPOT =</Text>
                  <Text strong style={{ fontSize: 16, color: '#1890ff' }}>{latency.decode_per_token_latency_ms.toFixed(3)} ms</Text>
                </div>
              </div>
            </div>
          </div>
        )
      case 'throughput':
        const e2eLatency = latency.end_to_end_latency_ms
        return (
          <div style={cardStyle}>
            <Text strong style={{ fontSize: 14, display: 'block', marginBottom: 10, color: colors.primary }}>Throughput (Token/s)</Text>
            <div style={definitionStyle}>
              <Text>
                <strong>定义：</strong>每秒生成的Token数量(tokens/s)。
                衡量系统处理能力的核心指标，吞吐量越高，单位时间能服务的请求越多，成本效益越好。
              </Text>
            </div>
            <div style={formulaStyle}>
              Throughput = <Fraction top="Batch Size × Output Length" bottom="T_e2e" large />
            </div>
            <div style={varDefStyle}>
              <div><b>变量说明：</b></div>
              <div>• <b>Batch Size</b> - 批次大小：同时处理的请求数量（由DP并行度决定）</div>
              <div>• <b>Output Length</b> - 输出长度：每个请求生成的Token数量</div>
              <div>• <b>T<sub>e2e</sub></b> - 端到端延迟：从请求开始到所有Token生成完成的总时间 (ms)</div>
              <div>• <b>TTFT</b> - 首Token延迟：Prefill阶段处理输入的时间</div>
              <div>• <b>TPOT</b> - 每Token延迟：Decode阶段生成每个Token的平均时间</div>
              <div>• <b>Throughput<sub>max</sub></b> - 理论最大吞吐量：假设无通信开销时的上限</div>
              <div>• <b>Peak FLOPs</b> - 集群总峰值算力 = Num Chips × Chip TFLOPs × 10¹²</div>
              <div>• <b>FLOPs per Token</b> - 生成单Token的浮点运算量（约2×参数量）</div>
            </div>
            <div style={{ marginTop: 12, fontSize: 14 }}>
              <div style={subFormulaStyle}>
                T<sub>e2e</sub> = TTFT + TPOT × Output Length = <b>{e2eLatency.toFixed(1)} ms</b>
              </div>
              <div style={subFormulaStyle}>
                Throughput<sub>max</sub> = <Fraction top="Peak FLOPs" bottom="FLOPs per Token" /> = <b>{throughput.theoretical_max_throughput.toFixed(0)} tok/s</b>
              </div>
              <div style={{ borderTop: '1px solid #d9d9d9', marginTop: 12, paddingTop: 12 }}>
                <div style={valueRowStyle}>
                  <Text strong style={{ fontSize: 15 }}>Throughput =</Text>
                  <Text strong style={{ fontSize: 16, color: '#1890ff' }}>{throughput.tokens_per_second.toFixed(0)} tok/s</Text>
                </div>
                <div style={valueRowStyle}>
                  <Text type="secondary">Request Throughput:</Text>
                  <Text strong>{throughput.requests_per_second.toFixed(2)} req/s</Text>
                </div>
              </div>
            </div>
          </div>
        )
      case 'mfu':
        return (
          <div style={cardStyle}>
            <Text strong style={{ fontSize: 14, display: 'block', marginBottom: 10, color: colors.primary }}>MFU (Model FLOPs Utilization)</Text>
            <div style={definitionStyle}>
              <Text>
                <strong>定义：</strong>实际用于模型计算的算力占硬件峰值算力的比例。
                MFU越高说明硬件利用越充分，是衡量部署效率的关键指标。
              </Text>
            </div>
            <div style={formulaStyle}>
              MFU = <Fraction top="Achieved FLOPs" bottom="Peak FLOPs" large /> × 100%
            </div>
            <div style={varDefStyle}>
              <div><b>变量说明：</b></div>
              <div>• <b>Achieved FLOPs</b> - 实际算力：实际用于模型前向计算的浮点运算量/秒</div>
              <div>• <b>Peak FLOPs</b> - 峰值算力：硬件理论最大FP16浮点运算能力</div>
              <div>• <b>Throughput</b> - 实际Token吞吐量 (tokens/s)</div>
              <div>• <b>FLOPs per Token</b> - 生成单Token的有效计算量 ≈ 2 × 模型参数量</div>
              <div>• <b>Num Chips</b> - 芯片总数 = DP × TP × PP × EP</div>
              <div>• <b>Chip TFLOPs</b> - 单芯片FP16峰值算力，如H100 SXM为1979 TFLOPs</div>
              <div>• <b>10¹²</b> - TFLOPs到FLOPs的换算系数</div>
            </div>
            <div style={{ marginTop: 12, fontSize: 14 }}>
              <div style={subFormulaStyle}>
                Achieved = Throughput × FLOPs per Token = <b>{(throughput.tokens_per_second * throughput.theoretical_max_throughput / (throughput.theoretical_max_throughput || 1)).toExponential(2)} FLOPs/s</b>
              </div>
              <div style={subFormulaStyle}>
                Peak = Num Chips × Chip TFLOPs × 10¹² = <b>{plan.total_chips} chips × Peak</b>
              </div>
              <div style={{ borderTop: '1px solid #d9d9d9', marginTop: 12, paddingTop: 12 }}>
                <div style={valueRowStyle}>
                  <Text strong style={{ fontSize: 15 }}>MFU =</Text>
                  <Text strong style={{ fontSize: 16, color: '#1890ff' }}>{(throughput.model_flops_utilization * 100).toFixed(2)}%</Text>
                </div>
                <div style={{ fontSize: 13, color: '#666', marginTop: 8 }}>
                  参考值: Prefill阶段 40-60%, Decode阶段 20-40%
                </div>
              </div>
            </div>
          </div>
        )
      case 'bottleneck':
        const bottleneckDescMap: Record<string, { name: string; desc: string; solution: string }> = {
          compute: {
            name: '计算瓶颈',
            desc: '算力不足，GPU计算单元成为限制因素',
            solution: '增加TP并行度，或使用更强算力的芯片',
          },
          memory: {
            name: '访存瓶颈',
            desc: '显存带宽不足，数据读取速度限制了计算',
            solution: '减小batch size，或使用更高带宽的芯片',
          },
          communication: {
            name: '通信瓶颈',
            desc: '芯片间通信延迟过高，集合通信成为限制因素',
            solution: '减小TP/PP并行度，或使用更高带宽的互联',
          },
          pipeline_bubble: {
            name: '流水线气泡',
            desc: '流水线并行导致的空闲时间过长',
            solution: '增加micro-batch数量，或减小PP并行度',
          },
        }
        const bottleneckInfo = bottleneckDescMap[latency.bottleneck_type] || { name: '未知', desc: '', solution: '' }

        // 瓶颈分析使用warning色调
        const bottleneckCardStyle: React.CSSProperties = {
          ...cardStyle,
          background: colors.warningLight,
          border: `1px solid ${colors.warning}`,
        }

        return (
          <div style={bottleneckCardStyle}>
            <Text strong style={{ fontSize: 14, display: 'block', marginBottom: 10, color: colors.warning }}>瓶颈分析详情</Text>

            {/* 瓶颈类型标签 */}
            <div style={{ marginBottom: 12 }}>
              <Tag style={{
                background: colors.warning,
                color: '#fff',
                border: 'none',
                borderRadius: 4,
                fontSize: 11,
                padding: '2px 8px',
              }}>
                {bottleneckInfo.name}
              </Tag>
            </div>

            {/* 瓶颈原因和详情 */}
            <div style={{ background: '#fff', borderRadius: 8, padding: 10, marginBottom: 10 }}>
              <div style={{ marginBottom: 8 }}>
                <Text style={{ fontSize: 11, color: colors.textSecondary }}>瓶颈原因</Text>
                <div style={{ fontSize: 12, marginTop: 4, color: colors.text }}>{bottleneckInfo.desc}</div>
              </div>
              <div>
                <Text style={{ fontSize: 11, color: colors.textSecondary }}>详细信息</Text>
                <div style={{ fontSize: 12, marginTop: 4, color: colors.text }}>{latency.bottleneck_details}</div>
              </div>
            </div>

            {/* 优化建议 */}
            <div style={{ background: '#fff', borderRadius: 8, padding: 10, marginBottom: 10, borderLeft: `3px solid ${colors.primary}` }}>
              <Text style={{ fontSize: 11, color: colors.textSecondary }}>优化建议</Text>
              <div style={{ fontSize: 12, marginTop: 4, color: colors.primary }}>{bottleneckInfo.solution}</div>
            </div>

            {/* 延迟分解 */}
            <div style={{ background: '#fff', borderRadius: 8, padding: 10 }}>
              <Text style={{ fontSize: 11, display: 'block', marginBottom: 8, color: colors.textSecondary }}>延迟分解</Text>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11 }}>
                  <Text style={{ color: colors.textSecondary }}>Prefill 计算</Text>
                  <Text style={{ color: colors.text }}>{latency.prefill_compute_latency_ms.toFixed(2)} ms</Text>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11 }}>
                  <Text style={{ color: colors.textSecondary }}>Prefill 通信</Text>
                  <Text style={{ color: colors.text }}>{latency.prefill_comm_latency_ms.toFixed(2)} ms</Text>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11 }}>
                  <Text style={{ color: colors.textSecondary }}>Decode 计算</Text>
                  <Text style={{ color: colors.text }}>{latency.decode_compute_latency_ms.toFixed(3)} ms</Text>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11 }}>
                  <Text style={{ color: colors.textSecondary }}>Decode 通信</Text>
                  <Text style={{ color: colors.text }}>{latency.decode_comm_latency_ms.toFixed(3)} ms</Text>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, gridColumn: 'span 2' }}>
                  <Text style={{ color: colors.textSecondary }}>流水线气泡比</Text>
                  <Text style={{ color: colors.text }}>{(latency.pipeline_bubble_ratio * 100).toFixed(1)}%</Text>
                </div>
              </div>
            </div>
          </div>
        )
      default:
        return null
    }
  }

  // 指标卡片样式 - 更精致的悬浮效果
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
      {/* 方案概览 - 更精致的卡片设计 */}
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

      {/* 关键指标 - 2x2网格布局 */}
      <div style={{ marginBottom: 14 }}>
        <Text style={{ fontSize: 11, color: colors.textSecondary, display: 'block', marginBottom: 8 }}>关键指标</Text>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <div
            style={metricCardStyle(selectedMetric === 'ttft')}
            onClick={() => setSelectedMetric(selectedMetric === 'ttft' ? null : 'ttft')}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <Text style={{ fontSize: 11, color: colors.textSecondary }}>TTFT</Text>
              <InfoCircleOutlined style={{ fontSize: 10, color: selectedMetric === 'ttft' ? colors.primary : '#ccc' }} />
            </div>
            <div style={{ fontSize: 18, fontWeight: 600, color: colors.text, marginTop: 4 }}>
              {latency.prefill_total_latency_ms.toFixed(1)} <span style={{ fontSize: 12, fontWeight: 400, color: colors.textSecondary }}>ms</span>
            </div>
          </div>
          <div
            style={metricCardStyle(selectedMetric === 'tpot')}
            onClick={() => setSelectedMetric(selectedMetric === 'tpot' ? null : 'tpot')}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <Text style={{ fontSize: 11, color: colors.textSecondary }}>TPOT</Text>
              <InfoCircleOutlined style={{ fontSize: 10, color: selectedMetric === 'tpot' ? colors.primary : '#ccc' }} />
            </div>
            <div style={{ fontSize: 18, fontWeight: 600, color: colors.text, marginTop: 4 }}>
              {latency.decode_per_token_latency_ms.toFixed(2)} <span style={{ fontSize: 12, fontWeight: 400, color: colors.textSecondary }}>ms</span>
            </div>
          </div>
          <div
            style={metricCardStyle(selectedMetric === 'throughput')}
            onClick={() => setSelectedMetric(selectedMetric === 'throughput' ? null : 'throughput')}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <Text style={{ fontSize: 11, color: colors.textSecondary }}>Throughput</Text>
              <InfoCircleOutlined style={{ fontSize: 10, color: selectedMetric === 'throughput' ? colors.primary : '#ccc' }} />
            </div>
            <div style={{ fontSize: 18, fontWeight: 600, color: colors.text, marginTop: 4 }}>
              {throughput.tokens_per_second.toFixed(0)} <span style={{ fontSize: 12, fontWeight: 400, color: colors.textSecondary }}>tok/s</span>
            </div>
          </div>
          <div
            style={metricCardStyle(selectedMetric === 'mfu')}
            onClick={() => setSelectedMetric(selectedMetric === 'mfu' ? null : 'mfu')}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <Text style={{ fontSize: 11, color: colors.textSecondary }}>MFU</Text>
              <InfoCircleOutlined style={{ fontSize: 10, color: selectedMetric === 'mfu' ? colors.primary : '#ccc' }} />
            </div>
            <div style={{ fontSize: 18, fontWeight: 600, color: colors.text, marginTop: 4 }}>
              {(throughput.model_flops_utilization * 100).toFixed(1)} <span style={{ fontSize: 12, fontWeight: 400, color: colors.textSecondary }}>%</span>
            </div>
          </div>
        </div>
      </div>

      {/* 指标详情展示 (瓶颈详情在瓶颈卡片下方显示) */}
      {selectedMetric && selectedMetric !== 'bottleneck' && getMetricDetail(selectedMetric)}

      {/* 显存利用 - 更精致的进度条设计 */}
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
        {/* 显存分解 */}
        <div style={{ display: 'flex', gap: 12, marginTop: 8, fontSize: 10, color: colors.textSecondary }}>
          <span>模型: {memory.model_memory_gb.toFixed(1)}G</span>
          <span>KV Cache: {memory.kv_cache_memory_gb.toFixed(1)}G</span>
          <span>激活: {memory.activation_memory_gb.toFixed(1)}G</span>
        </div>
      </div>

      {/* 瓶颈分析 - 更精致的卡片设计 */}
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
      {selectedMetric === 'bottleneck' && getMetricDetail('bottleneck')}

      {/* 优化建议 - 更清晰的列表设计 */}
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

      {/* Top-K 方案列表 - 更精致的设计 */}
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

      {/* 搜索统计 - 更精致的设计 */}
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

// ============================================
// 主面板组件
// ============================================

interface DeploymentAnalysisPanelProps {
  topology?: HierarchicalTopology | null
  onTrafficResultChange?: (result: TopologyTrafficResult | null) => void
  rackConfig?: RackConfig
  podCount?: number
  racksPerPod?: number
}

export const DeploymentAnalysisPanel: React.FC<DeploymentAnalysisPanelProps> = ({
  topology,
  onTrafficResultChange,
  rackConfig,
  podCount = 1,
  racksPerPod = 1,
}) => {
  // 模型配置状态
  const [modelConfig, setModelConfig] = useState<LLMModelConfig>(
    MODEL_PRESETS['deepseek-v3']
  )

  // 推理配置状态
  const [inferenceConfig, setInferenceConfig] = useState<InferenceConfig>(
    INFERENCE_PRESETS['standard']
  )

  // 硬件配置来源: 'topology' 使用拓扑配置, 'manual' 手动配置
  const [hardwareSource, setHardwareSource] = useState<'topology' | 'manual'>('topology')

  // 从拓扑配置提取的芯片组
  const [chipGroups, setChipGroups] = useState<ChipGroupInfo[]>([])
  const [selectedChipType, setSelectedChipType] = useState<string | undefined>()

  // 硬件配置状态
  const [hardwareConfig, setHardwareConfig] = useState<HardwareConfig>(() =>
    createHardwareConfig('h100-sxm', 'dgx-h100', 1, 400)
  )

  // 序列化 rackConfig 用于深度比较
  const rackConfigJson = React.useMemo(() =>
    rackConfig ? JSON.stringify(rackConfig) : '',
    [rackConfig]
  )

  // 当拓扑配置变化时，提取芯片组信息并更新硬件配置
  React.useEffect(() => {
    if (!rackConfig || rackConfig.boards.length === 0) {
      setChipGroups([])
      return
    }

    const groups = extractChipGroupsFromConfig(rackConfig.boards)
    setChipGroups(groups)

    // 默认选择第一个芯片类型
    if (groups.length > 0 && !selectedChipType) {
      setSelectedChipType(groups[0].presetId || groups[0].chipType)
    }

    // 如果使用拓扑配置模式，立即更新硬件配置
    if (hardwareSource === 'topology' && groups.length > 0) {
      const currentSelectedType = selectedChipType || groups[0].presetId || groups[0].chipType
      const summary: TopologyHardwareSummary = {
        chipGroups: groups.map(g => ({
          ...g,
          boardCount: g.boardCount * podCount * racksPerPod,
          totalCount: g.totalCount * podCount * racksPerPod,
        })),
        totalPods: podCount,
        totalRacks: podCount * racksPerPod,
        totalBoards: rackConfig.boards.reduce((sum, b) => sum + b.count, 0) * podCount * racksPerPod,
        totalChips: groups.reduce((sum, g) => sum + g.totalCount, 0) * podCount * racksPerPod,
        intraNodeBandwidthGbps: 900,  // 默认值，后续从连接配置读取
        interNodeBandwidthGbps: 400,
        intraNodeLatencyUs: 1,
        interNodeLatencyUs: 2,
      }
      const config = generateHardwareConfig(summary, currentSelectedType)
      if (config) {
        setHardwareConfig(config)
      }
    }
  }, [rackConfigJson, hardwareSource, podCount, racksPerPod]) // 使用 JSON 字符串作为依赖

  // 当选择的芯片类型变化时，更新硬件配置
  React.useEffect(() => {
    if (hardwareSource === 'topology' && rackConfig && chipGroups.length > 0 && selectedChipType) {
      const summary: TopologyHardwareSummary = {
        chipGroups: chipGroups.map(g => ({
          ...g,
          boardCount: g.boardCount * podCount * racksPerPod,
          totalCount: g.totalCount * podCount * racksPerPod,
        })),
        totalPods: podCount,
        totalRacks: podCount * racksPerPod,
        totalBoards: rackConfig.boards.reduce((sum, b) => sum + b.count, 0) * podCount * racksPerPod,
        totalChips: chipGroups.reduce((sum, g) => sum + g.totalCount, 0) * podCount * racksPerPod,
        intraNodeBandwidthGbps: 900,
        interNodeBandwidthGbps: 400,
        intraNodeLatencyUs: 1,
        interNodeLatencyUs: 2,
      }
      const config = generateHardwareConfig(summary, selectedChipType)
      if (config) {
        setHardwareConfig(config)
      }
    }
  }, [selectedChipType])

  // 并行策略状态
  const [parallelismMode, setParallelismMode] = useState<'manual' | 'auto'>('manual')
  const [manualStrategy, setManualStrategy] = useState<ParallelismStrategy>({
    dp: 1, tp: 8, pp: 1, ep: 1, sp: 1,
  })
  const [searchConstraints, setSearchConstraints] = useState<SearchConstraints>({
    max_chips: 8,
    tp_within_node: true,
  })

  // 评分权重
  const [scoreWeights, setScoreWeights] = useState<ScoreWeights>({ ...DEFAULT_SCORE_WEIGHTS })

  // 分析结果状态
  const [analysisResult, setAnalysisResult] = useState<PlanAnalysisResult | null>(null)
  const [topKPlans, setTopKPlans] = useState<PlanAnalysisResult[]>([])
  const [searchStats, setSearchStats] = useState<{ evaluated: number; feasible: number; timeMs: number } | null>(null)
  const [loading, setLoading] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  // 计算最大可用芯片数
  const maxChips = hardwareConfig.node.chips_per_node * hardwareConfig.cluster.num_nodes

  // 运行分析
  const handleRunAnalysis = useCallback(async () => {
    // 清除之前的结果
    setAnalysisResult(null)
    setTopKPlans([])
    setSearchStats(null)
    setErrorMsg(null)
    setLoading(true)
    try {
      if (parallelismMode === 'manual') {
        // 手动模式：直接分析指定策略（不使用权重，显示原始评分）
        const result = analyzePlan(modelConfig, inferenceConfig, manualStrategy, hardwareConfig)
        setAnalysisResult(result)
        setTopKPlans([result])
      } else {
        // 自动模式：搜索最优方案（使用自定义权重）
        const searchResult = searchWithFixedChips(
          modelConfig,
          inferenceConfig,
          hardwareConfig,
          searchConstraints.max_chips || maxChips,
          'balanced', // 始终使用 balanced，让权重决定优化方向
          scoreWeights
        )
        if (searchResult.top_k_plans.length > 0) {
          setAnalysisResult(searchResult.optimal_plan)
          setTopKPlans(searchResult.top_k_plans)
          setSearchStats({
            evaluated: searchResult.search_stats.evaluated_count,
            feasible: searchResult.search_stats.feasible_count,
            timeMs: searchResult.search_stats.search_time_ms,
          })
        } else {
          setErrorMsg(`未找到可行方案 (已评估 ${searchResult.search_stats.evaluated_count} 个方案)`)
          setSearchStats({
            evaluated: searchResult.search_stats.evaluated_count,
            feasible: 0,
            timeMs: searchResult.search_stats.search_time_ms,
          })
        }
      }
    } catch (error) {
      console.error('分析失败:', error)
      const msg = error instanceof Error ? error.message : '未知错误'
      setErrorMsg(`搜索失败: ${msg}`)
    } finally {
      setLoading(false)
    }
  }, [modelConfig, inferenceConfig, hardwareConfig, parallelismMode, manualStrategy, searchConstraints, maxChips, scoreWeights])

  return (
    <div style={{ padding: 0 }}>
      {/* 模型配置 */}
      <div style={sectionCardStyle}>
        <div style={sectionTitleStyle}>模型配置</div>
        <ModelConfigSelector value={modelConfig} onChange={setModelConfig} />
      </div>

      {/* 推理配置 */}
      <div style={sectionCardStyle}>
        <div style={sectionTitleStyle}>推理配置</div>
        <InferenceConfigSelector value={inferenceConfig} onChange={setInferenceConfig} />
      </div>

      {/* 硬件配置 */}
      <div style={sectionCardStyle}>
        <div style={sectionTitleStyle}>硬件配置</div>
        {/* 配置来源选择 */}
        <div style={{ marginBottom: 12 }}>
          <Radio.Group
            size="small"
            value={hardwareSource}
            onChange={(e) => setHardwareSource(e.target.value)}
            buttonStyle="solid"
          >
            <Radio.Button value="topology">使用拓扑配置</Radio.Button>
            <Radio.Button value="manual">手动配置</Radio.Button>
          </Radio.Group>
        </div>

        {hardwareSource === 'topology' ? (
          <div>
            {chipGroups.length === 0 ? (
              <div style={{ padding: 12, background: colors.warningLight, borderRadius: 8, border: '1px solid #ffd591' }}>
                <Text type="warning">
                  <WarningOutlined style={{ marginRight: 6 }} />
                  请先在「Board层级」中配置芯片类型
                </Text>
              </div>
            ) : (
              <>
                {chipGroups.length > 1 && (
                  <div style={configRowStyle}>
                    <Text>分析芯片类型</Text>
                    <Select
                      size="small"
                      value={selectedChipType}
                      onChange={setSelectedChipType}
                      style={{ width: 140 }}
                      options={chipGroups.map(g => ({
                        value: g.presetId || g.chipType,
                        label: `${g.chipType} (${g.totalCount * podCount * racksPerPod}个)`,
                      }))}
                    />
                  </div>
                )}

                <div style={{ padding: 10, background: colors.successLight, borderRadius: 8, fontSize: 12, border: '1px solid #b7eb8f' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                    <Text><CheckCircleOutlined style={{ color: colors.success, marginRight: 4 }} />芯片: <b>{hardwareConfig.chip.chip_type}</b></Text>
                    <Text>共 <b>{hardwareConfig.node.chips_per_node * hardwareConfig.cluster.num_nodes}</b> 个</Text>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 12px', color: colors.textSecondary }}>
                    <span>节点数: {hardwareConfig.cluster.num_nodes}</span>
                    <span>每节点: {hardwareConfig.node.chips_per_node} 个</span>
                    <span>算力: {hardwareConfig.chip.compute_tflops_fp16} TFLOPs</span>
                    <span>显存: {hardwareConfig.chip.memory_gb}GB</span>
                  </div>
                </div>
              </>
            )}
          </div>
        ) : (
          <HardwareConfigSelector value={hardwareConfig} onChange={setHardwareConfig} />
        )}
      </div>

      {/* 并行策略 */}
      <div style={sectionCardStyle}>
        <div style={sectionTitleStyle}>并行策略</div>
        <ParallelismConfigPanel
          mode={parallelismMode}
          onModeChange={setParallelismMode}
          manualStrategy={manualStrategy}
          onManualStrategyChange={setManualStrategy}
          searchConstraints={searchConstraints}
          onSearchConstraintsChange={setSearchConstraints}
          maxChips={maxChips}
          scoreWeights={scoreWeights}
          onScoreWeightsChange={setScoreWeights}
        />
      </div>

      {/* 运行按钮 */}
      <Button
        type="primary"
        icon={parallelismMode === 'auto' ? <SearchOutlined /> : <PlayCircleOutlined />}
        onClick={handleRunAnalysis}
        loading={loading}
        block
        size="large"
        style={{
          marginBottom: 16,
          height: 44,
          borderRadius: 8,
          background: colors.primary,
          boxShadow: '0 2px 8px rgba(94, 106, 210, 0.3)',
        }}
      >
        {parallelismMode === 'auto' ? '搜索最优方案' : '运行分析'}
      </Button>

      {/* 分析结果 */}
      <div style={{
        ...sectionCardStyle,
        background: colors.background,
        border: `1px solid ${colors.border}`,
      }}>
        <div style={sectionTitleStyle}>分析结果</div>
        <AnalysisResultDisplay
          result={analysisResult}
          topKPlans={topKPlans}
          loading={loading}
          onSelectPlan={(plan) => setAnalysisResult(plan)}
          searchStats={searchStats}
          errorMsg={errorMsg}
        />

        {/* 映射到拓扑按钮 */}
        {analysisResult && topology && onTrafficResultChange && (
          <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
            <Button
              icon={<HeatMapOutlined />}
              onClick={() => {
                try {
                  const strategy = analysisResult.plan.parallelism
                  const trafficResult = analyzeTopologyTraffic(
                    topology,
                    strategy,
                    analysisResult.communication
                  )
                  onTrafficResultChange(trafficResult)
                } catch (error) {
                  console.error('流量映射失败:', error)
                  onTrafficResultChange(null)
                }
              }}
              style={{ flex: 1 }}
            >
              映射到拓扑热力图
            </Button>
            <Button
              onClick={() => onTrafficResultChange(null)}
              type="text"
            >
              清除
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}

export default DeploymentAnalysisPanel
