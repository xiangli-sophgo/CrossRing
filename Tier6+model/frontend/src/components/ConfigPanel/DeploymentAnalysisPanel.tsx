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

const { Text } = Typography
const { Panel } = Collapse

// ============================================
// 样式常量
// ============================================

const configRowStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginBottom: 8,
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
            <Text style={{ fontSize: 12, cursor: 'help' }}>批次大小</Text>
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
          <Tooltip title="Input Sequence Length: 输入提示词的token数量（如问题、上下文）">
            <Text style={{ fontSize: 12, cursor: 'help' }}>输入长度</Text>
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
          <Tooltip title="Output Sequence Length: 生成输出的token数量（如回答、代码）">
            <Text style={{ fontSize: 12, cursor: 'help' }}>输出长度</Text>
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
            <Text style={{ fontSize: 12, cursor: 'help' }}>最大序列</Text>
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

interface ParallelismConfigPanelProps {
  mode: 'manual' | 'auto'
  onModeChange: (mode: 'manual' | 'auto') => void
  manualStrategy: ParallelismStrategy
  onManualStrategyChange: (strategy: ParallelismStrategy) => void
  searchConstraints: SearchConstraints
  onSearchConstraintsChange: (constraints: SearchConstraints) => void
  maxChips: number
}

const ParallelismConfigPanel: React.FC<ParallelismConfigPanelProps> = ({
  mode,
  onModeChange,
  manualStrategy,
  onManualStrategyChange,
  searchConstraints,
  onSearchConstraintsChange,
  maxChips,
}) => {
  const totalParallelism = manualStrategy.dp * manualStrategy.tp * manualStrategy.pp * manualStrategy.ep

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
              value={(searchConstraints as any).optimization_target || 'balanced'}
              onChange={(v) => onSearchConstraintsChange({ ...searchConstraints, optimization_target: v } as any)}
              style={{ width: 100 }}
              options={[
                { value: 'latency', label: '低延迟' },
                { value: 'throughput', label: '高吞吐' },
                { value: 'efficiency', label: '高效率' },
                { value: 'balanced', label: '均衡' },
              ]}
            />
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
  loading: boolean
}

const AnalysisResultDisplay: React.FC<AnalysisResultDisplayProps> = ({ result, loading }) => {
  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 20 }}>
        <Spin />
        <div style={{ marginTop: 8 }}>
          <Text type="secondary">正在分析...</Text>
        </div>
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

  // 瓶颈类型图标
  const bottleneckIcon = {
    compute: <ThunderboltOutlined />,
    memory: <DatabaseOutlined />,
    communication: <ApiOutlined />,
    pipeline_bubble: <WarningOutlined />,
  }[latency.bottleneck_type]

  return (
    <div>
      {/* 方案概览 */}
      <div style={{
        padding: 12,
        background: is_feasible ? '#f6ffed' : '#fff2f0',
        borderRadius: 8,
        marginBottom: 12,
        border: `1px solid ${is_feasible ? '#b7eb8f' : '#ffccc7'}`,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <Text strong>方案: </Text>
            <Tag>DP={plan.parallelism.dp}</Tag>
            <Tag>TP={plan.parallelism.tp}</Tag>
            <Tag>PP={plan.parallelism.pp}</Tag>
            {plan.parallelism.ep > 1 && <Tag>EP={plan.parallelism.ep}</Tag>}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {is_feasible ? (
              <CheckCircleOutlined style={{ color: '#52c41a', fontSize: 18 }} />
            ) : (
              <Tooltip title={infeasibility_reason}>
                <WarningOutlined style={{ color: '#ff4d4f', fontSize: 18 }} />
              </Tooltip>
            )}
            <Text strong style={{ fontSize: 18, color: is_feasible ? '#52c41a' : '#ff4d4f' }}>
              {score.overall_score.toFixed(1)}
            </Text>
          </div>
        </div>
      </div>

      {/* 关键指标 */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 12 }}>
        <div style={{ padding: 10, background: '#f5f5f5', borderRadius: 6 }}>
          <Text type="secondary" style={{ fontSize: 11 }}>首Token延迟 (TTFT)</Text>
          <div style={{ fontSize: 16, fontWeight: 600 }}>
            {latency.prefill_total_latency_ms.toFixed(1)} ms
          </div>
        </div>
        <div style={{ padding: 10, background: '#f5f5f5', borderRadius: 6 }}>
          <Text type="secondary" style={{ fontSize: 11 }}>每Token延迟 (TPOT)</Text>
          <div style={{ fontSize: 16, fontWeight: 600 }}>
            {latency.decode_per_token_latency_ms.toFixed(2)} ms
          </div>
        </div>
        <div style={{ padding: 10, background: '#f5f5f5', borderRadius: 6 }}>
          <Text type="secondary" style={{ fontSize: 11 }}>Token吞吐量</Text>
          <div style={{ fontSize: 16, fontWeight: 600 }}>
            {throughput.tokens_per_second.toFixed(0)} tok/s
          </div>
        </div>
        <div style={{ padding: 10, background: '#f5f5f5', borderRadius: 6 }}>
          <Text type="secondary" style={{ fontSize: 11 }}>MFU</Text>
          <div style={{ fontSize: 16, fontWeight: 600 }}>
            {(throughput.model_flops_utilization * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* 显存利用 */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
          <Text style={{ fontSize: 12 }}>显存利用</Text>
          <Text style={{ fontSize: 12 }}>
            {memory.total_per_chip_gb.toFixed(1)} / {result.plan.total_chips > 0 ? '80' : '-'} GB
          </Text>
        </div>
        <Progress
          percent={memory.memory_utilization * 100}
          status={memory.is_memory_sufficient ? 'normal' : 'exception'}
          size="small"
          format={(p) => `${p?.toFixed(0)}%`}
        />
      </div>

      {/* 瓶颈分析 */}
      <div style={{ padding: 10, background: '#fff7e6', borderRadius: 6, marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
          {bottleneckIcon}
          <Text strong style={{ fontSize: 12 }}>瓶颈: {latency.bottleneck_type}</Text>
        </div>
        <Text style={{ fontSize: 11, color: '#666' }}>{latency.bottleneck_details}</Text>
      </div>

      {/* 优化建议 */}
      {suggestions.length > 0 && (
        <div>
          <Text strong style={{ fontSize: 12, display: 'block', marginBottom: 6 }}>优化建议</Text>
          {suggestions.slice(0, 3).map((s, i) => (
            <div key={i} style={{
              padding: 8,
              background: '#f9f9f9',
              borderRadius: 4,
              marginBottom: 4,
              fontSize: 11,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text>{s.description}</Text>
                <Tag color={s.priority <= 2 ? 'red' : s.priority <= 3 ? 'orange' : 'default'} style={{ fontSize: 10 }}>
                  P{s.priority}
                </Tag>
              </div>
              <Text type="secondary" style={{ fontSize: 10 }}>预期: {s.expected_improvement}</Text>
            </div>
          ))}
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
}

export const DeploymentAnalysisPanel: React.FC<DeploymentAnalysisPanelProps> = ({
  topology,
  onTrafficResultChange,
}) => {
  // 模型配置状态
  const [modelConfig, setModelConfig] = useState<LLMModelConfig>(
    MODEL_PRESETS['deepseek-v3']
  )

  // 推理配置状态
  const [inferenceConfig, setInferenceConfig] = useState<InferenceConfig>(
    INFERENCE_PRESETS['standard']
  )

  // 硬件配置状态
  const [hardwareConfig, setHardwareConfig] = useState<HardwareConfig>(() =>
    createHardwareConfig('h100-sxm', 'dgx-h100', 1, 400)
  )

  // 并行策略状态
  const [parallelismMode, setParallelismMode] = useState<'manual' | 'auto'>('manual')
  const [manualStrategy, setManualStrategy] = useState<ParallelismStrategy>({
    dp: 1, tp: 8, pp: 1, ep: 1, sp: 1,
  })
  const [searchConstraints, setSearchConstraints] = useState<SearchConstraints>({
    max_chips: 8,
    tp_within_node: true,
  })

  // 分析结果状态
  const [analysisResult, setAnalysisResult] = useState<PlanAnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)

  // 计算最大可用芯片数
  const maxChips = hardwareConfig.node.chips_per_node * hardwareConfig.cluster.num_nodes

  // 运行分析
  const handleRunAnalysis = useCallback(async () => {
    setLoading(true)
    try {
      if (parallelismMode === 'manual') {
        // 手动模式：直接分析指定策略
        const result = analyzePlan(modelConfig, inferenceConfig, manualStrategy, hardwareConfig)
        setAnalysisResult(result)
      } else {
        // 自动模式：搜索最优方案
        const searchResult = searchWithFixedChips(
          modelConfig,
          inferenceConfig,
          hardwareConfig,
          searchConstraints.max_chips || maxChips,
          (searchConstraints as any).optimization_target || 'balanced'
        )
        if (searchResult.top_k_plans.length > 0) {
          setAnalysisResult(searchResult.optimal_plan)
        } else {
          setAnalysisResult(null)
        }
      }
    } catch (error) {
      console.error('分析失败:', error)
    } finally {
      setLoading(false)
    }
  }, [modelConfig, inferenceConfig, hardwareConfig, parallelismMode, manualStrategy, searchConstraints, maxChips])

  return (
    <div style={{ padding: 0 }}>
      <Collapse
        defaultActiveKey={['model', 'inference', 'hardware', 'parallelism']}
        size="small"
        style={{ background: 'transparent', border: 'none' }}
      >
        <Panel
          header={<Text strong style={{ fontSize: 13 }}>模型配置</Text>}
          key="model"
          style={sectionStyle}
        >
          <ModelConfigSelector value={modelConfig} onChange={setModelConfig} />
        </Panel>

        <Panel
          header={<Text strong style={{ fontSize: 13 }}>推理配置</Text>}
          key="inference"
          style={sectionStyle}
        >
          <InferenceConfigSelector value={inferenceConfig} onChange={setInferenceConfig} />
        </Panel>

        <Panel
          header={<Text strong style={{ fontSize: 13 }}>硬件配置</Text>}
          key="hardware"
          style={sectionStyle}
        >
          <HardwareConfigSelector value={hardwareConfig} onChange={setHardwareConfig} />
        </Panel>

        <Panel
          header={<Text strong style={{ fontSize: 13 }}>并行策略</Text>}
          key="parallelism"
          style={sectionStyle}
        >
          <ParallelismConfigPanel
            mode={parallelismMode}
            onModeChange={setParallelismMode}
            manualStrategy={manualStrategy}
            onManualStrategyChange={setManualStrategy}
            searchConstraints={searchConstraints}
            onSearchConstraintsChange={setSearchConstraints}
            maxChips={maxChips}
          />
        </Panel>
      </Collapse>

      {/* 运行按钮 */}
      <Button
        type="primary"
        icon={parallelismMode === 'auto' ? <SearchOutlined /> : <PlayCircleOutlined />}
        onClick={handleRunAnalysis}
        loading={loading}
        block
        style={{ marginTop: 12, marginBottom: 12 }}
      >
        {parallelismMode === 'auto' ? '搜索最优方案' : '运行分析'}
      </Button>

      {/* 分析结果 */}
      <Collapse
        defaultActiveKey={['result']}
        size="small"
        style={{ background: 'transparent', border: 'none' }}
      >
        <Panel
          header={<Text strong style={{ fontSize: 13 }}>分析结果</Text>}
          key="result"
        >
          <AnalysisResultDisplay result={analysisResult} loading={loading} />

          {/* 映射到拓扑按钮 */}
          {analysisResult && topology && onTrafficResultChange && (
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
              block
              style={{ marginTop: 12 }}
            >
              映射到拓扑热力图
            </Button>
          )}

          {/* 清除热力图按钮 */}
          {analysisResult && topology && onTrafficResultChange && (
            <Button
              onClick={() => onTrafficResultChange(null)}
              block
              style={{ marginTop: 8 }}
              type="text"
              size="small"
            >
              清除热力图
            </Button>
          )}
        </Panel>
      </Collapse>
    </div>
  )
}

export default DeploymentAnalysisPanel
