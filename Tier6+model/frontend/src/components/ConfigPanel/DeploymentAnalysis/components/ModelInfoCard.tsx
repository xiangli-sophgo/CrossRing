/**
 * 模型架构可视化卡片
 * 左右布局：左边架构图，右边详情面板
 */

import React, { useState, useMemo } from 'react'
import { Tag } from 'antd'
import { LLMModelConfig, InferenceConfig } from '../../../../utils/llmDeployment/types'

interface ModelInfoCardProps {
  model: LLMModelConfig
  inference?: InferenceConfig
}

// 浅色配色 - 与整体风格搭配
const COLORS = {
  embedding: { bg: '#e6f4ff', border: '#91caff', text: '#0958d9' },
  attention: { bg: '#f9f0ff', border: '#d3adf7', text: '#722ed1' },
  ffn: { bg: '#f6ffed', border: '#b7eb8f', text: '#389e0d' },
  moe: { bg: '#fff0f6', border: '#ffadd2', text: '#c41d7f' },
  output: { bg: '#fff7e6', border: '#ffd591', text: '#d46b08' },
  wire: '#d9d9d9',
  wireActive: '#1677ff',
  text: '#262626',
  textSecondary: '#8c8c8c',
  bg: '#fafafa',
}

// 格式化数字
const formatNum = (n: number): string => {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`
  return n.toString()
}

// FLOPs 计算
const calculateFLOPs = (model: LLMModelConfig, inference?: InferenceConfig) => {
  const B = inference?.batch_size || 1
  const S = inference?.input_seq_length || 1024
  const H = model.hidden_size
  const I = model.intermediate_size
  const L = model.num_layers
  const n_h = model.num_attention_heads
  const n_kv = model.num_kv_heads
  const d_h = H / n_h
  const V = model.vocab_size

  const qkvProj = 2 * B * S * H * (H + 2 * (n_kv * d_h))
  const attnScore = 2 * B * n_h * S * S * d_h
  const attnOut = 2 * B * S * H * H
  const attnTotal = qkvProj + attnScore + attnOut

  let ffnTotal = 2 * 2 * B * S * H * I + 2 * B * S * I * H

  if (model.model_type === 'moe' && model.moe_config) {
    const expertI = model.moe_config.expert_intermediate_size || I
    const topK = model.moe_config.num_experts_per_tok
    const shared = model.moe_config.num_shared_experts || 0
    ffnTotal = (topK + shared) * (2 * 2 * B * S * H * expertI + 2 * B * S * expertI * H)
    ffnTotal += 2 * B * S * H * model.moe_config.num_experts
  }

  const embFLOPs = 2 * B * S * V * H
  const outFLOPs = 2 * B * S * H * V

  return {
    attention: attnTotal,
    ffn: ffnTotal,
    perLayer: attnTotal + ffnTotal,
    embedding: embFLOPs,
    output: outFLOPs,
    total: embFLOPs + L * (attnTotal + ffnTotal) + outFLOPs,
  }
}

// 参数量计算
const calculateParams = (model: LLMModelConfig) => {
  const H = model.hidden_size
  const I = model.intermediate_size
  const L = model.num_layers
  const V = model.vocab_size
  const n_kv = model.num_kv_heads
  const d_h = H / model.num_attention_heads

  const embParams = V * H
  const attnParams = H * H + 2 * (n_kv * d_h) * H + H * H

  let ffnParams = 3 * H * I
  if (model.model_type === 'moe' && model.moe_config) {
    const E = model.moe_config.num_experts
    const S = model.moe_config.num_shared_experts || 0
    const expertI = model.moe_config.expert_intermediate_size || I
    ffnParams = (E + S) * 3 * H * expertI + H * E
  }

  const outParams = H * V

  return {
    embedding: embParams,
    attention: attnParams * L,
    ffn: ffnParams * L,
    output: outParams,
    total: embParams + L * (attnParams + ffnParams) + outParams,
  }
}

// 两列参数网格
const ParamGrid: React.FC<{ items: { label: string; value: string | number }[] }> = ({ items }) => (
  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 16px', marginBottom: 8 }}>
    {items.map((item, i) => (
      <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, padding: '2px 0' }}>
        <span style={{ color: COLORS.textSecondary }}>{item.label}:</span>
        <span style={{ fontWeight: 500, color: COLORS.text, fontFamily: 'ui-monospace, monospace' }}>{item.value}</span>
      </div>
    ))}
  </div>
)

// 详情面板
const DetailSection: React.FC<{ title: string; color: typeof COLORS.embedding; children: React.ReactNode }> = ({ title, color, children }) => (
  <div style={{ marginBottom: 12 }}>
    <div style={{
      fontSize: 13,
      fontWeight: 600,
      color: color.text,
      marginBottom: 6,
      paddingBottom: 4,
      borderBottom: `2px solid ${color.border}`,
    }}>
      {title}
    </div>
    {children}
  </div>
)

export const ModelInfoCard: React.FC<ModelInfoCardProps> = ({ model, inference }) => {
  const [selectedBlock, setSelectedBlock] = useState<string>('overview')

  const isMoE = model.model_type === 'moe' && model.moe_config
  const isMLA = model.attention_type === 'mla' && model.mla_config
  const params = useMemo(() => calculateParams(model), [model])
  const flops = useMemo(() => calculateFLOPs(model, inference), [model, inference])

  const H = model.hidden_size
  const I = model.intermediate_size
  const n_h = model.num_attention_heads
  const n_kv = model.num_kv_heads
  const d_h = H / n_h

  // SVG 尺寸 - 根据内容调整高度
  const svgWidth = 500
  const svgHeight = isMoE ? 625 : 565
  const centerX = svgWidth / 2

  // 块样式
  const getBlockStyle = (key: string, color: typeof COLORS.embedding) => ({
    fill: color.bg,
    stroke: selectedBlock === key ? COLORS.wireActive : color.border,
    strokeWidth: selectedBlock === key ? 2 : 1,
    cursor: 'pointer',
  })

  // 操作步骤组件 - 更详细的说明
  const StepList: React.FC<{ items: { name: string; desc: string; detail?: string }[] }> = ({ items }) => (
    <div style={{ marginTop: 8, padding: 10, background: '#fafafa', borderRadius: 4, fontSize: 11 }}>
      <div style={{ fontWeight: 600, color: COLORS.text, marginBottom: 8 }}>操作流程</div>
      {items.map((item, i) => (
        <div key={i} style={{ marginBottom: 6, paddingLeft: 4 }}>
          <div style={{ display: 'flex', alignItems: 'flex-start' }}>
            <span style={{ color: '#1677ff', fontWeight: 600, minWidth: 20 }}>{i + 1}.</span>
            <div>
              <b style={{ color: COLORS.text }}>{item.name}</b>
              <span style={{ color: COLORS.textSecondary }}>：{item.desc}</span>
              {item.detail && <div style={{ color: '#999', marginTop: 2, fontSize: 10 }}>{item.detail}</div>}
            </div>
          </div>
        </div>
      ))}
    </div>
  )

  // 详情内容
  const detailContent: Record<string, React.ReactNode> = {
    embedding: (
      <DetailSection title="Embedding Layer" color={COLORS.embedding}>
        <div style={{ fontSize: 12, color: COLORS.textSecondary, marginBottom: 8, lineHeight: 1.5 }}>
          将离散的 Token ID 映射为连续的高维向量表示，是模型理解文本的第一步。
        </div>
        <ParamGrid items={[
          { label: '词表大小', value: formatNum(model.vocab_size) },
          { label: '隐藏维度', value: formatNum(H) },
          { label: '位置编码', value: 'RoPE' },
          { label: '参数量', value: formatNum(params.embedding) },
        ]} />
        <StepList items={[
          { name: 'Token Embedding', desc: '查表映射', detail: `输入 Token ID，从 ${formatNum(model.vocab_size)}×${formatNum(H)} 的嵌入矩阵中查找对应的 ${formatNum(H)} 维向量` },
          { name: 'RoPE 位置编码', desc: '旋转位置编码', detail: '通过旋转变换将位置信息编码到向量中，使模型能够区分不同位置的 Token' },
        ]} />
      </DetailSection>
    ),
    attention: (
      <DetailSection title={`${isMLA ? 'MLA' : model.attention_type?.toUpperCase() || 'GQA'} Attention`} color={COLORS.attention}>
        <div style={{ fontSize: 12, color: COLORS.textSecondary, marginBottom: 8, lineHeight: 1.5 }}>
          {isMLA
            ? 'Multi-head Latent Attention：DeepSeek 独创的注意力机制，通过低秩压缩大幅减少 KV Cache 显存占用。'
            : '自注意力机制：让每个位置能够关注序列中所有其他位置，捕获长距离依赖关系。'}
        </div>
        <ParamGrid items={[
          { label: '注意力头', value: n_h },
          { label: 'KV 头', value: n_kv },
          { label: '头维度', value: d_h },
          { label: '参数量/层', value: formatNum(params.attention / model.num_layers) },
          ...(isMLA && model.mla_config ? [
            { label: 'Q LoRA', value: model.mla_config.q_lora_rank },
            { label: 'KV LoRA', value: model.mla_config.kv_lora_rank },
            { label: 'KV 压缩比', value: `${Math.round(H / model.mla_config.kv_lora_rank)}×` },
          ] : []),
        ]} />
        {isMLA ? (
          <StepList items={[
            { name: 'RMSNorm', desc: '层归一化', detail: 'Root Mean Square Layer Normalization，对输入进行归一化，稳定训练过程' },
            { name: 'Q LoRA 投影', desc: '低秩 Q 生成', detail: `先 Down 投影 (${formatNum(H)}→${model.mla_config?.q_lora_rank})，再 Up 投影生成 Q，减少计算量` },
            { name: 'KV 压缩', desc: `${Math.round(H / (model.mla_config?.kv_lora_rank || 512))}× 压缩`, detail: `将 ${formatNum(H)} 维压缩到 ${model.mla_config?.kv_lora_rank} 维，大幅减少 KV Cache 显存` },
            { name: 'Attention 计算', desc: 'Q @ K^T → Softmax → @ V', detail: '计算 Query 和 Key 的相似度，Softmax 归一化后加权 Value' },
            { name: 'V 解压 + Output', desc: '恢复维度并投影', detail: `从 ${model.mla_config?.kv_lora_rank} 维解压回 ${formatNum(H)} 维，然后线性投影输出` },
            { name: '+ Residual', desc: '残差连接', detail: '将输出与原始输入相加，帮助梯度流动，防止深层网络退化' },
          ]} />
        ) : (
          <StepList items={[
            { name: 'RMSNorm', desc: '层归一化', detail: '对输入进行归一化，稳定训练过程' },
            { name: 'QKV 投影', desc: '生成 Q/K/V', detail: `通过三个线性变换生成 Query、Key、Value 向量` },
            { name: 'Attention', desc: '注意力计算', detail: 'Q @ K^T / √d → Softmax → @ V，计算位置间的关联' },
            { name: 'Output 投影', desc: '多头拼接输出', detail: '将多个注意力头的输出拼接后线性投影' },
            { name: '+ Residual', desc: '残差连接', detail: '与输入相加，防止梯度消失' },
          ]} />
        )}
      </DetailSection>
    ),
    ffn: (
      <DetailSection title="Feed-Forward Network" color={COLORS.ffn}>
        <div style={{ fontSize: 12, color: COLORS.textSecondary, marginBottom: 8, lineHeight: 1.5 }}>
          前馈网络：对每个位置独立进行非线性变换，是 Transformer 中存储知识的主要组件。
        </div>
        <ParamGrid items={[
          { label: '隐藏维度', value: formatNum(H) },
          { label: '中间维度', value: formatNum(I) },
          { label: '扩展倍数', value: `${(I / H).toFixed(1)}×` },
          { label: '激活函数', value: 'SwiGLU' },
          { label: '参数量/层', value: formatNum(params.ffn / model.num_layers) },
        ]} />
        <StepList items={[
          { name: 'RMSNorm', desc: '层归一化', detail: '对 Attention 输出进行归一化' },
          { name: 'Gate 投影', desc: `${formatNum(H)}→${formatNum(I)}`, detail: '门控分支，决定信息通过的比例' },
          { name: 'Up 投影', desc: `${formatNum(H)}→${formatNum(I)}`, detail: '数值分支，承载实际的特征变换' },
          { name: 'SiLU ⊙ 门控', desc: '门控激活', detail: 'SiLU(Gate) × Up，SiLU 是平滑的激活函数，门控机制增强表达能力' },
          { name: 'Down 投影', desc: `${formatNum(I)}→${formatNum(H)}`, detail: '将扩展的维度降回原始维度' },
          { name: '+ Residual', desc: '残差连接', detail: '与 FFN 输入相加，保持信息流通' },
        ]} />
      </DetailSection>
    ),
    moe: model.moe_config && (
      <DetailSection title="Mixture of Experts (MoE)" color={COLORS.moe}>
        <div style={{ fontSize: 12, color: COLORS.textSecondary, marginBottom: 8, lineHeight: 1.5 }}>
          稀疏专家混合：每个 Token 只激活部分专家，以较低计算量实现超大模型容量。
        </div>
        {model.model_name?.toLowerCase().includes('deepseek') && (
          <div style={{ background: '#fff0f6', border: '1px solid #ffadd2', borderRadius: 4, padding: '6px 8px', marginBottom: 8, fontSize: 11 }}>
            <b style={{ color: COLORS.moe.text }}>DeepSeek 层分布：</b>
            <span style={{ color: COLORS.textSecondary }}>Layer 0-2 使用 Dense FFN，Layer 3-{model.num_layers - 1} 使用 MoE</span>
          </div>
        )}
        <ParamGrid items={[
          { label: '专家总数', value: model.moe_config.num_experts },
          { label: '激活专家', value: `Top-${model.moe_config.num_experts_per_tok}` },
          { label: '共享专家', value: model.moe_config.num_shared_experts || 0 },
          { label: '专家维度', value: formatNum(model.moe_config.expert_intermediate_size || I) },
          { label: '参数量/层', value: formatNum(params.ffn / model.num_layers) },
        ]} />
        <StepList items={[
          { name: 'RMSNorm', desc: '层归一化', detail: '对 Attention 输出进行归一化' },
          { name: 'Router 路由', desc: '计算专家分数', detail: `将输入通过路由网络，计算对 ${model.moe_config.num_experts} 个专家的亲和度分数` },
          { name: 'Top-K 选择', desc: `选择 ${model.moe_config.num_experts_per_tok} 个专家`, detail: '每个 Token 只选择分数最高的几个专家，实现稀疏计算' },
          { name: 'AllToAll Dispatch', desc: '分布式 Token 分发', detail: '在多 GPU 环境下，将 Token 发送到对应专家所在的设备' },
          { name: 'Expert FFN', desc: '专家计算', detail: `每个被选中的专家独立执行 SwiGLU FFN (${formatNum(H)}→${formatNum(model.moe_config.expert_intermediate_size || I)}→${formatNum(H)})` },
          { name: 'Shared Expert', desc: '共享专家计算', detail: `${model.moe_config.num_shared_experts || 0} 个共享专家处理所有 Token，提供通用特征` },
          { name: 'AllToAll Combine', desc: '收集专家输出', detail: '将各专家的计算结果收集回原始设备' },
          { name: 'Sum + Residual', desc: '加权求和 + 残差', detail: '按路由分数加权求和专家输出，加上共享专家输出，再与输入残差连接' },
        ]} />
      </DetailSection>
    ),
    output: (
      <DetailSection title="LM Head (Output)" color={COLORS.output}>
        <div style={{ fontSize: 12, color: COLORS.textSecondary, marginBottom: 8, lineHeight: 1.5 }}>
          语言模型头：将最终隐藏状态映射到词表空间，预测下一个 Token 的概率分布。
        </div>
        <ParamGrid items={[
          { label: '输入维度', value: formatNum(H) },
          { label: '输出维度', value: formatNum(model.vocab_size) },
          { label: '权重共享', value: '是' },
          { label: '参数量', value: formatNum(params.output) },
        ]} />
        <StepList items={[
          { name: 'Final RMSNorm', desc: '最终归一化', detail: '对最后一层 Transformer 的输出进行归一化，确保数值稳定' },
          { name: '线性投影', desc: `${formatNum(H)}→${formatNum(model.vocab_size)}`, detail: '通过与 Embedding 矩阵共享的权重，将隐藏状态映射到词表空间' },
          { name: 'Softmax', desc: '概率分布', detail: '将 logits 转换为概率分布，选择概率最高的 Token 作为输出' },
        ]} />
      </DetailSection>
    ),
    // 整体流程概览（默认视图）
    overview: (
      <DetailSection title="模型架构概览" color={{ bg: '#e6f7ff', border: '#91d5ff', text: '#0050b3' }}>
        <div style={{ fontSize: 12, color: COLORS.textSecondary, marginBottom: 10, lineHeight: 1.6 }}>
          {model.model_name} 是一个 {model.num_layers} 层的大型语言模型，采用 {isMLA ? 'MLA (Multi-head Latent Attention)' : 'GQA (Grouped Query Attention)'} 注意力机制
          {isMoE && `和 MoE (Mixture of Experts) 稀疏架构`}。
        </div>
        <ParamGrid items={[
          { label: '总参数量', value: formatNum(params.total) },
          { label: '隐藏维度', value: formatNum(H) },
          { label: '层数', value: model.num_layers },
          { label: '词表大小', value: formatNum(model.vocab_size) },
          { label: '注意力头', value: n_h },
          { label: 'KV 头', value: n_kv },
          ...(isMoE && model.moe_config ? [
            { label: '专家数', value: model.moe_config.num_experts },
            { label: '激活专家', value: model.moe_config.num_experts_per_tok },
          ] : []),
        ]} />
        <StepList items={[
          { name: 'Embedding', desc: '词嵌入层', detail: `将 Token ID 映射为 ${formatNum(H)} 维向量，加入 RoPE 位置编码` },
          { name: 'Transformer ×' + model.num_layers, desc: '核心计算层', detail: `每层包含 ${isMLA ? 'MLA' : 'Attention'} 和 ${isMoE ? 'MoE' : 'FFN'}，使用 Pre-LN 架构` },
          { name: 'Final RMSNorm', desc: '输出归一化', detail: '对最后一层输出进行 RMSNorm 归一化' },
          { name: 'LM Head', desc: '语言模型头', detail: `映射到 ${formatNum(model.vocab_size)} 词表空间，预测下一个 Token` },
        ]} />
        <div style={{ marginTop: 12, padding: 8, background: '#f0f5ff', borderRadius: 4, fontSize: 11, color: '#1d39c4' }}>
          💡 点击左侧架构图中的各个模块，查看详细参数和操作流程
        </div>
      </DetailSection>
    ),
    // Transformer 层说明
    transformer: (
      <DetailSection title="Transformer Layer" color={{ bg: '#f0f0f0', border: '#d9d9d9', text: '#595959' }}>
        <div style={{ fontSize: 12, color: COLORS.textSecondary, marginBottom: 8, lineHeight: 1.5 }}>
          Transformer 层是模型的核心组件，由注意力机制和前馈网络组成，共 {model.num_layers} 层堆叠。
        </div>
        <ParamGrid items={[
          { label: '层数', value: model.num_layers },
          { label: '隐藏维度', value: formatNum(H) },
          { label: '注意力类型', value: isMLA ? 'MLA' : (model.attention_type?.toUpperCase() || 'GQA') },
          { label: 'FFN 类型', value: isMoE ? 'MoE' : 'Dense' },
        ]} />
        <StepList items={[
          { name: 'Pre-LN 架构', desc: '归一化在前', detail: '每个子层前先做 RMSNorm，比 Post-LN 更稳定' },
          { name: 'Attention 子层', desc: '自注意力机制', detail: `${isMLA ? 'MLA' : 'GQA'} 注意力，捕获序列中的依赖关系` },
          { name: 'FFN 子层', desc: isMoE ? 'MoE 稀疏计算' : 'SwiGLU FFN', detail: isMoE ? '稀疏专家混合，大容量低计算' : '全连接前馈网络，存储知识' },
          { name: 'Residual 连接', desc: '残差连接', detail: '每个子层都有残差连接，x + SubLayer(x)，帮助梯度流动' },
        ]} />
      </DetailSection>
    ),
  }

  // 头部信息
  const headerContent = (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: 14, fontWeight: 600, color: COLORS.text }}>{model.model_name}</span>
        <Tag color="blue" style={{ margin: 0 }}>{model.attention_type?.toUpperCase() || 'GQA'}</Tag>
        {isMoE && <Tag color="magenta" style={{ margin: 0 }}>MoE</Tag>}
      </div>
      <div style={{ display: 'flex', gap: 16, fontSize: 12, color: COLORS.textSecondary }}>
        <span><b style={{ color: '#1677ff' }}>{formatNum(params.total)}</b> Params</span>
        <span><b style={{ color: '#52c41a' }}>{formatNum(flops.total)}</b> FLOPs</span>
        <span>{model.num_layers} Layers</span>
      </div>
    </div>
  )

  const cardContent = (
    <div style={{ display: 'flex', gap: 24 }}>
      {/* 左侧：架构图 - 占更大比例 */}
      <div style={{ flex: '0 0 55%', minWidth: 0 }}>
        <svg
          width="100%"
          height={svgHeight}
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          style={{ display: 'block' }}
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            <marker id="arrow" markerWidth="12" markerHeight="10" refX="6" refY="5" orient="auto" markerUnits="userSpaceOnUse">
              <polygon points="0 0, 12 5, 0 10" fill={COLORS.wire} />
            </marker>
          </defs>

          {/* 背景点击区域 - 点击空白处返回整体流程 */}
          <rect
            x={0} y={0} width={svgWidth} height={svgHeight}
            fill="transparent"
            onClick={() => setSelectedBlock('overview')}
            style={{ cursor: selectedBlock !== 'overview' ? 'pointer' : 'default' }}
          />

          {/* Input */}
          <text x={centerX} y={24} textAnchor="middle" fontSize={15} fontWeight={500} fill={COLORS.text}>
            Input [{inference?.batch_size || 'B'}, {inference ? formatNum(inference.input_seq_length) : 'S'}]
          </text>

          {/* Arrow - 线段 + 三角形 */}
          <line x1={centerX} y1={30} x2={centerX} y2={46} stroke={COLORS.wire} strokeWidth={2} />
          <polygon points={`${centerX},54 ${centerX - 6},44 ${centerX + 6},44`} fill={COLORS.wire} />

          {/* Embedding */}
          <g onClick={(e) => { e.stopPropagation(); setSelectedBlock('embedding') }} style={{ cursor: 'pointer' }}>
            <rect x={centerX - 130} y={54} width={260} height={54} rx={6} {...getBlockStyle('embedding', COLORS.embedding)} />
            <text x={centerX} y={78} textAnchor="middle" fontSize={16} fontWeight={600} fill={COLORS.embedding.text}>
              Embedding
            </text>
            <text x={centerX} y={98} textAnchor="middle" fontSize={13} fill={COLORS.textSecondary}>
              {formatNum(model.vocab_size)} × {formatNum(H)}
            </text>
          </g>

          {/* Arrow */}
          <line x1={centerX} y1={108} x2={centerX} y2={124} stroke={COLORS.wire} strokeWidth={2} />
          <polygon points={`${centerX},132 ${centerX - 6},122 ${centerX + 6},122`} fill={COLORS.wire} />

          {/* Transformer Layer Box */}
          <rect x={20} y={132} width={svgWidth - 40} height={isMoE ? 350 : 290} rx={8} fill="none" stroke={COLORS.wire} strokeWidth={1.5} strokeDasharray="6,3" />
          <text x={35} y={156} fontSize={14} fontWeight={500} fill={COLORS.textSecondary}>
            Transformer × {model.num_layers}
          </text>

          {/* Attention - MLA 或标准 GQA */}
          <g onClick={(e) => { e.stopPropagation(); setSelectedBlock('attention') }} style={{ cursor: 'pointer' }}>
            <rect x={35} y={168} width={200} height={isMoE ? 300 : 230} rx={6} {...getBlockStyle('attention', COLORS.attention)} />
            {/* 标题 */}
            <text x={135} y={188} textAnchor="middle" fontSize={15} fontWeight={600} fill={COLORS.attention.text}>
              {isMLA ? 'MLA' : model.attention_type?.toUpperCase() || 'GQA'}
            </text>
            {/* Pre-LN: RMSNorm */}
            <rect x={53} y={198} width={164} height={22} rx={3} fill="#fafafa" stroke={COLORS.wire} strokeWidth={1} />
            <text x={135} y={213} textAnchor="middle" fontSize={11} fill={COLORS.textSecondary}>RMSNorm</text>

            {isMLA && model.mla_config ? (
              /* MLA 完整流程 - 分叉数据流，间距加大适配 300 高度 */
              <>
                {/* RMSNorm 后分叉箭头 - 左边Q路径(中心100)，右边KV路径(中心186) */}
                <line x1={100} y1={220} x2={100} y2={230} stroke={COLORS.wire} strokeWidth={1.5} />
                <polygon points="100,238 96,228 104,228" fill={COLORS.wire} />
                <line x1={186} y1={220} x2={186} y2={230} stroke={COLORS.wire} strokeWidth={1.5} />
                <polygon points="186,238 182,228 190,228" fill={COLORS.wire} />

                {/* Q LoRA: 低秩Q生成 */}
                <g transform="translate(53, 238)">
                  <rect width={94} height={60} rx={4} fill="#fff" stroke={COLORS.attention.border} strokeWidth={1.5} />
                  <text x={47} y={20} textAnchor="middle" fontSize={11} fontWeight={500} fill={COLORS.attention.text}>Q LoRA</text>
                  <text x={47} y={38} textAnchor="middle" fontSize={10} fill={COLORS.textSecondary}>低秩压缩</text>
                  <text x={47} y={52} textAnchor="middle" fontSize={10} fill={COLORS.textSecondary}>生成 Q</text>
                </g>

                {/* KV 压缩 */}
                <g transform="translate(155, 238)">
                  <rect width={62} height={60} rx={4} fill={COLORS.attention.bg} stroke={COLORS.attention.border} strokeWidth={2} />
                  <text x={31} y={18} textAnchor="middle" fontSize={11} fontWeight={600} fill={COLORS.attention.text}>KV</text>
                  <text x={31} y={36} textAnchor="middle" fontSize={10} fill={COLORS.attention.text}>压缩</text>
                  <text x={31} y={52} textAnchor="middle" fontSize={10} fill={COLORS.attention.text}>{Math.round(H / (model.mla_config.kv_lora_rank || 512))}×</text>
                </g>

                {/* 汇合箭头 - Q和KV汇合到Attention */}
                <line x1={100} y1={298} x2={100} y2={312} stroke={COLORS.wire} strokeWidth={1.5} />
                <line x1={186} y1={298} x2={186} y2={312} stroke={COLORS.wire} strokeWidth={1.5} />
                <line x1={100} y1={312} x2={186} y2={312} stroke={COLORS.wire} strokeWidth={1.5} />
                <line x1={135} y1={312} x2={135} y2={322} stroke={COLORS.wire} strokeWidth={1.5} />
                <polygon points="135,330 131,320 139,320" fill={COLORS.wire} />

                {/* Attention */}
                <g transform="translate(53, 330)">
                  <rect width={164} height={30} rx={4} fill="#fff" stroke={COLORS.attention.border} strokeWidth={1.5} />
                  <text x={82} y={20} textAnchor="middle" fontSize={12} fill={COLORS.textSecondary}>Attention (QKᵀ)</text>
                </g>

                {/* 垂直流动箭头 */}
                <line x1={135} y1={360} x2={135} y2={372} stroke={COLORS.wire} strokeWidth={1.5} />
                <polygon points="135,380 131,370 139,370" fill={COLORS.wire} />

                {/* V 解压 + Output */}
                <g transform="translate(53, 380)">
                  <rect width={164} height={30} rx={4} fill="#fff" stroke={COLORS.attention.border} strokeWidth={1.5} />
                  <text x={82} y={20} textAnchor="middle" fontSize={12} fill={COLORS.attention.text}>V 解压 + Output</text>
                </g>

                {/* 垂直流动箭头 */}
                <line x1={135} y1={410} x2={135} y2={422} stroke={COLORS.wire} strokeWidth={1.5} />
                <polygon points="135,430 131,420 139,420" fill={COLORS.wire} />

                {/* Residual Add */}
                <g transform="translate(53, 430)">
                  <rect width={164} height={26} rx={3} fill="#fafafa" stroke={COLORS.wire} strokeWidth={1} />
                  <text x={82} y={17} textAnchor="middle" fontSize={11} fill={COLORS.textSecondary}>+ Residual</text>
                </g>

              </>
            ) : (
              /* 标准 GQA/MHA */
              <>
                {/* Q K V */}
                <g transform="translate(50, 235)">
                  {['Q', 'K', 'V'].map((label, i) => (
                    <g key={label} transform={`translate(${i * 55}, 0)`}>
                      <rect width={50} height={34} rx={4} fill="#fff" stroke={COLORS.attention.border} strokeWidth={1.5} />
                      <text x={25} y={22} textAnchor="middle" fontSize={14} fontWeight={500} fill={COLORS.attention.text}>{label}</text>
                    </g>
                  ))}
                </g>
                <g transform="translate(50, 279)">
                  <rect width={160} height={34} rx={4} fill="#fff" stroke={COLORS.attention.border} strokeWidth={1.5} />
                  <text x={80} y={22} textAnchor="middle" fontSize={13} fill={COLORS.textSecondary}>Dot-Product Attn</text>
                </g>
                <g transform="translate(55, 323)">
                  <rect width={150} height={32} rx={4} fill="#fff" stroke={COLORS.attention.border} strokeWidth={1.5} />
                  <text x={75} y={21} textAnchor="middle" fontSize={13} fill={COLORS.attention.text}>Output Proj</text>
                </g>
                {/* Residual Add */}
                <g transform="translate(50, 365)">
                  <rect width={160} height={24} rx={3} fill="#fafafa" stroke={COLORS.wire} strokeWidth={1} />
                  <text x={80} y={16} textAnchor="middle" fontSize={11} fill={COLORS.textSecondary}>+ Residual</text>
                </g>
              </>
            )}
          </g>

          {/* Arrow between Attention and FFN - 线段 + 三角形 */}
          <line x1={235} y1={isMoE ? 330 : 300} x2={250} y2={isMoE ? 330 : 300} stroke={COLORS.wire} strokeWidth={2} />
          <polygon points={`260,${isMoE ? 330 : 300} 250,${isMoE ? 324 : 294} 250,${isMoE ? 336 : 306}`} fill={COLORS.wire} />

          {/* FFN / MoE */}
          {isMoE && model.moe_config ? (
            <g onClick={(e) => { e.stopPropagation(); setSelectedBlock('moe') }} style={{ cursor: 'pointer' }}>
              <rect x={260} y={168} width={200} height={300} rx={6} {...getBlockStyle('moe', COLORS.moe)} />
              {/* 标题 */}
              <text x={360} y={188} textAnchor="middle" fontSize={15} fontWeight={600} fill={COLORS.moe.text}>
                MoE FFN
              </text>
              {/* Pre-LN: RMSNorm */}
              <rect x={278} y={198} width={164} height={22} rx={3} fill="#fafafa" stroke={COLORS.wire} strokeWidth={1} />
              <text x={360} y={213} textAnchor="middle" fontSize={11} fill={COLORS.textSecondary}>RMSNorm</text>

              {/* Router */}
              <g transform="translate(278, 228)">
                <rect width={170} height={28} rx={4} fill="#fff" stroke={COLORS.moe.border} strokeWidth={1.5} />
                <text x={85} y={19} textAnchor="middle" fontSize={12} fill={COLORS.moe.text}>
                  Router → Top-{model.moe_config.num_experts_per_tok}
                </text>
              </g>

              {/* 垂直流动箭头 */}
              <line x1={360} y1={256} x2={360} y2={262} stroke={COLORS.wire} strokeWidth={1.5} />
              <polygon points="360,268 356,260 364,260" fill={COLORS.wire} />

              {/* AllToAll Dispatch */}
              <g transform="translate(278, 268)">
                <rect width={164} height={24} rx={4} fill={COLORS.bg} stroke={COLORS.wire} strokeDasharray="4,2" strokeWidth={1.5} />
                <text x={82} y={16} textAnchor="middle" fontSize={11} fill={COLORS.textSecondary}>AllToAll Dispatch</text>
              </g>

              {/* 垂直流动箭头 */}
              <line x1={360} y1={292} x2={360} y2={298} stroke={COLORS.wire} strokeWidth={1.5} />
              <polygon points="360,304 356,296 364,296" fill={COLORS.wire} />

              {/* Routed Experts + Shared Expert 并排 */}
              <g transform="translate(278, 304)">
                {/* Routed Experts */}
                <rect width={108} height={70} rx={4} fill="#fff" stroke={COLORS.ffn.border} strokeWidth={1.5} />
                <text x={54} y={15} textAnchor="middle" fontSize={11} fontWeight={600} fill={COLORS.ffn.text}>Routed ×{model.moe_config.num_experts_per_tok}</text>
                <g transform="translate(6, 20)">
                  {[0, 1, 2, 3].map((i) => (
                    <rect key={i} x={(i % 2) * 48} y={Math.floor(i / 2) * 24} width={44} height={20} rx={3}
                      fill={COLORS.ffn.bg} stroke={COLORS.ffn.border} />
                  ))}
                  <text x={22} y={14} textAnchor="middle" fontSize={10} fill={COLORS.ffn.text}>E₁</text>
                  <text x={70} y={14} textAnchor="middle" fontSize={10} fill={COLORS.ffn.text}>E₂</text>
                  <text x={22} y={38} textAnchor="middle" fontSize={10} fill={COLORS.textSecondary}>...</text>
                  <text x={70} y={38} textAnchor="middle" fontSize={10} fill={COLORS.ffn.text}>E₈</text>
                </g>

                {/* Shared Expert */}
                {(model.moe_config.num_shared_experts || 0) > 0 && (
                  <g transform="translate(112, 0)">
                    <rect width={52} height={70} rx={4} fill="#fff" stroke={COLORS.attention.border} strokeWidth={1.5} />
                    <text x={26} y={15} textAnchor="middle" fontSize={10} fontWeight={600} fill={COLORS.attention.text}>Shared</text>
                    <text x={26} y={30} textAnchor="middle" fontSize={10} fill={COLORS.attention.text}>×{model.moe_config.num_shared_experts}</text>
                    <rect x={6} y={38} width={40} height={26} rx={3} fill={COLORS.attention.bg} stroke={COLORS.attention.border} />
                    <text x={26} y={55} textAnchor="middle" fontSize={10} fill={COLORS.attention.text}>FFN</text>
                  </g>
                )}
              </g>

              {/* 垂直流动箭头 */}
              <line x1={360} y1={374} x2={360} y2={380} stroke={COLORS.wire} strokeWidth={1.5} />
              <polygon points="360,386 356,378 364,378" fill={COLORS.wire} />

              {/* AllToAll Combine */}
              <g transform="translate(278, 386)">
                <rect width={164} height={24} rx={4} fill={COLORS.bg} stroke={COLORS.wire} strokeDasharray="4,2" strokeWidth={1.5} />
                <text x={82} y={16} textAnchor="middle" fontSize={11} fill={COLORS.textSecondary}>AllToAll Combine</text>
              </g>

              {/* 垂直流动箭头 */}
              <line x1={360} y1={410} x2={360} y2={416} stroke={COLORS.wire} strokeWidth={1.5} />
              <polygon points="360,422 356,414 364,414" fill={COLORS.wire} />

              {/* Sum + Residual */}
              <g transform="translate(278, 422)">
                <rect width={164} height={24} rx={3} fill="#fafafa" stroke={COLORS.wire} strokeWidth={1} />
                <text x={82} y={16} textAnchor="middle" fontSize={11} fill={COLORS.textSecondary}>Sum + Residual</text>
              </g>
            </g>
          ) : (
            <g onClick={(e) => { e.stopPropagation(); setSelectedBlock('ffn') }} style={{ cursor: 'pointer' }}>
              <rect x={260} y={168} width={200} height={230} rx={6} {...getBlockStyle('ffn', COLORS.ffn)} />
              {/* 标题 */}
              <text x={360} y={188} textAnchor="middle" fontSize={15} fontWeight={600} fill={COLORS.ffn.text}>
                FFN (SwiGLU)
              </text>
              {/* Pre-LN: RMSNorm */}
              <rect x={278} y={198} width={164} height={22} rx={3} fill="#fafafa" stroke={COLORS.wire} strokeWidth={1} />
              <text x={360} y={213} textAnchor="middle" fontSize={11} fill={COLORS.textSecondary}>RMSNorm</text>

              {/* 垂直流动箭头 */}
              <line x1={360} y1={220} x2={360} y2={226} stroke={COLORS.wire} strokeWidth={1.5} />
              <polygon points="360,232 356,224 364,224" fill={COLORS.wire} />

              <g transform="translate(278, 232)">
                <rect width={78} height={32} rx={4} fill="#fff" stroke={COLORS.ffn.border} strokeWidth={1.5} />
                <text x={39} y={21} textAnchor="middle" fontSize={12} fill={COLORS.ffn.text}>Gate</text>
              </g>
              <g transform="translate(364, 232)">
                <rect width={78} height={32} rx={4} fill="#fff" stroke={COLORS.ffn.border} strokeWidth={1.5} />
                <text x={39} y={21} textAnchor="middle" fontSize={12} fill={COLORS.ffn.text}>Up</text>
              </g>

              {/* 垂直流动箭头 */}
              <line x1={360} y1={264} x2={360} y2={270} stroke={COLORS.wire} strokeWidth={1.5} />
              <polygon points="360,276 356,268 364,268" fill={COLORS.wire} />

              <text x={360} y={290} textAnchor="middle" fontSize={13} fill={COLORS.textSecondary}>SiLU ⊙</text>

              {/* 垂直流动箭头 */}
              <line x1={360} y1={296} x2={360} y2={302} stroke={COLORS.wire} strokeWidth={1.5} />
              <polygon points="360,308 356,300 364,300" fill={COLORS.wire} />

              <g transform="translate(295, 308)">
                <rect width={130} height={32} rx={4} fill="#fff" stroke={COLORS.ffn.border} strokeWidth={1.5} />
                <text x={65} y={21} textAnchor="middle" fontSize={12} fill={COLORS.ffn.text}>Down</text>
              </g>

              {/* 垂直流动箭头 */}
              <line x1={360} y1={340} x2={360} y2={346} stroke={COLORS.wire} strokeWidth={1.5} />
              <polygon points="360,352 356,344 364,344" fill={COLORS.wire} />

              {/* Residual Add */}
              <g transform="translate(278, 352)">
                <rect width={164} height={24} rx={3} fill="#fafafa" stroke={COLORS.wire} strokeWidth={1} />
                <text x={82} y={16} textAnchor="middle" fontSize={11} fill={COLORS.textSecondary}>+ Residual</text>
              </g>
              <text x={360} y={392} textAnchor="middle" fontSize={12} fill={COLORS.textSecondary}>
                {formatNum(H)} → {formatNum(I)} → {formatNum(H)}
              </text>
            </g>
          )}

          {/* Arrow: Transformer → Final RMSNorm */}
          <line x1={centerX} y1={isMoE ? 482 : 422} x2={centerX} y2={isMoE ? 498 : 438} stroke={COLORS.wire} strokeWidth={2} />
          <polygon points={`${centerX},${isMoE ? 506 : 446} ${centerX - 6},${isMoE ? 496 : 436} ${centerX + 6},${isMoE ? 496 : 436}`} fill={COLORS.wire} />

          {/* Final RMSNorm */}
          <g transform={`translate(${centerX - 80}, ${isMoE ? 506 : 446})`}>
            <rect width={160} height={26} rx={4} fill="#fafafa" stroke={COLORS.wire} strokeWidth={1} />
            <text x={80} y={18} textAnchor="middle" fontSize={12} fill={COLORS.textSecondary}>Final RMSNorm</text>
          </g>

          {/* Arrow: Final RMSNorm → LM Head */}
          <line x1={centerX} y1={isMoE ? 532 : 472} x2={centerX} y2={isMoE ? 548 : 488} stroke={COLORS.wire} strokeWidth={2} />
          <polygon points={`${centerX},${isMoE ? 556 : 496} ${centerX - 6},${isMoE ? 546 : 486} ${centerX + 6},${isMoE ? 546 : 486}`} fill={COLORS.wire} />

          {/* Output */}
          <g onClick={(e) => { e.stopPropagation(); setSelectedBlock('output') }} style={{ cursor: 'pointer' }}>
            <rect x={centerX - 110} y={isMoE ? 556 : 496} width={220} height={54} rx={6} {...getBlockStyle('output', COLORS.output)} />
            <text x={centerX} y={isMoE ? 582 : 520} textAnchor="middle" fontSize={15} fontWeight={600} fill={COLORS.output.text}>
              LM Head
            </text>
            <text x={centerX} y={isMoE ? 600 : 538} textAnchor="middle" fontSize={12} fill={COLORS.textSecondary}>
              {formatNum(H)} → {formatNum(model.vocab_size)}
            </text>
          </g>
        </svg>
      </div>

      {/* 右侧：详情面板 */}
      <div style={{ flex: '1 1 45%', minWidth: 0, padding: '0 8px' }}>
        {detailContent[selectedBlock] || detailContent.overview}

        {/* 推理配置 */}
        {inference && (
          <div style={{
            marginTop: 12,
            padding: '8px 10px',
            background: '#f6ffed',
            borderRadius: 4,
            border: '1px solid #b7eb8f',
            fontSize: 11,
            color: '#389e0d',
          }}>
            <span style={{ fontWeight: 600 }}>推理配置：</span>
            <span style={{ marginLeft: 8 }}>Batch={inference.batch_size}</span>
            <span style={{ marginLeft: 8 }}>Input={formatNum(inference.input_seq_length)}</span>
            <span style={{ marginLeft: 8 }}>Output={formatNum(inference.output_seq_length)}</span>
          </div>
        )}
      </div>
    </div>
  )

  return (
    <div>
      {headerContent}
      {cardContent}
    </div>
  )
}

export default ModelInfoCard
