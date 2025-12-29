/**
 * 并行策略介绍组件
 * 参考 NVIDIA Megatron-LM 和 NeMo 的专业可视化风格
 */

import React from 'react'

export type ParallelismType = 'dp' | 'tp' | 'pp' | 'ep' | 'sp'

interface ParallelismInfoProps {
  type: ParallelismType
}

// 统一配色 - 所有策略使用相同色调
const COLORS: Record<ParallelismType, { primary: string; light: string; dark: string }> = {
  dp: { primary: '#1890ff', light: '#e6f7ff', dark: '#0050b3' },
  tp: { primary: '#1890ff', light: '#e6f7ff', dark: '#0050b3' },
  pp: { primary: '#1890ff', light: '#e6f7ff', dark: '#0050b3' },
  ep: { primary: '#1890ff', light: '#e6f7ff', dark: '#0050b3' },
  sp: { primary: '#1890ff', light: '#e6f7ff', dark: '#0050b3' },
}

const INFO: Record<ParallelismType, {
  name: string
  fullName: string
  shortDesc: string
  definition: string
  keyPoints: string[]
  communication: string
  bestFor: string
}> = {
  dp: {
    name: 'DP',
    fullName: 'Data Parallelism',
    shortDesc: '数据并行',
    definition: '每个 GPU 持有完整模型副本，将 Batch 切分后并行处理，通过 AllReduce 同步梯度。',
    keyPoints: ['模型完整复制', 'Batch 切分', '通信频率低'],
    communication: 'AllReduce (每 Step)',
    bestFor: '模型可放入单 GPU，需要扩展吞吐量',
  },
  tp: {
    name: 'TP',
    fullName: 'Tensor Parallelism',
    shortDesc: '张量并行',
    definition: '将权重矩阵按列/行切分到多个 GPU，每层计算后需要 AllReduce/AllGather 同步。',
    keyPoints: ['矩阵列切分', '每层通信', '节点内高效'],
    communication: 'AllReduce (每层 2-4 次)',
    bestFor: '单层参数过大，需要 NVLink 高速互联',
  },
  pp: {
    name: 'PP',
    fullName: 'Pipeline Parallelism',
    shortDesc: '流水线并行',
    definition: '将模型按层分成多个 Stage，分配到不同 GPU，数据以流水线方式依次处理。',
    keyPoints: ['层间切分', '点对点通信', '存在 Bubble'],
    communication: 'P2P (每 Micro-batch)',
    bestFor: '跨节点扩展，模型层数多',
  },
  ep: {
    name: 'EP',
    fullName: 'Expert Parallelism',
    shortDesc: '专家并行',
    definition: 'MoE 专用，将专家网络分布到不同 GPU，通过 AllToAll 路由 Token 到对应专家。',
    keyPoints: ['专家分布', 'AllToAll 路由', '稀疏激活'],
    communication: 'AllToAll (每 MoE 层)',
    bestFor: 'MoE 模型，大规模专家扩展',
  },
  sp: {
    name: 'SP',
    fullName: 'Sequence Parallelism',
    shortDesc: '序列并行',
    definition: '沿序列维度切分激活值，在 LayerNorm/Dropout 处与 TP 配合，减少激活显存。',
    keyPoints: ['序列切分', '与 TP 配合', '减少激活显存'],
    communication: 'AllGather/ReduceScatter',
    bestFor: '长序列推理，激活显存受限',
  },
}

// ============================================
// SVG 图示 - 统一配色，简洁专业
// ============================================

// 统一色彩
const C = {
  primary: '#1890ff',
  primaryLight: '#e6f7ff',
  border: '#d9d9d9',
  text: '#262626',
  textSec: '#8c8c8c',
  bg: '#fafafa',
}

const DiagramDP: React.FC = () => (
  <svg width="280" height="140" viewBox="0 0 280 140">
    {/* 顶部：Global Batch */}
    <rect x="40" y="10" width="200" height="26" rx="4" fill={C.primaryLight} stroke={C.primary} strokeWidth="1.5" />
    <text x="140" y="27" textAnchor="middle" fontSize="12" fill={C.primary} fontWeight="600">Global Batch</text>

    {/* 分发箭头 */}
    <path d="M90 38 L55 55" stroke={C.border} strokeWidth="1.5" />
    <path d="M140 38 L140 55" stroke={C.border} strokeWidth="1.5" />
    <path d="M190 38 L225 55" stroke={C.border} strokeWidth="1.5" />

    {/* GPU 方块 */}
    {[0, 1, 2].map(i => (
      <g key={i} transform={`translate(${20 + i * 90}, 58)`}>
        <rect width="70" height="72" rx="5" fill="#fff" stroke={C.border} strokeWidth="1.5" />
        <text x="35" y="16" textAnchor="middle" fontSize="11" fill={C.text} fontWeight="600">GPU {i}</text>
        <line x1="0" y1="22" x2="70" y2="22" stroke={C.border} strokeWidth="1" />
        {/* 模型 */}
        <rect x="8" y="28" width="54" height="16" rx="3" fill={C.bg} stroke={C.border} strokeWidth="1" />
        <text x="35" y="39" textAnchor="middle" fontSize="9" fill={C.textSec}>Model (完整)</text>
        {/* 数据 */}
        <rect x="8" y="48" width="54" height="16" rx="3" fill={C.primaryLight} stroke={C.primary} strokeWidth="1" />
        <text x="35" y="59" textAnchor="middle" fontSize="9" fill={C.primary}>Data 1/3</text>
      </g>
    ))}
  </svg>
)

const DiagramTP: React.FC = () => (
  <svg width="280" height="140" viewBox="0 0 280 140">
    {/* 顶部：Weight Matrix */}
    <rect x="40" y="10" width="200" height="28" rx="4" fill={C.primaryLight} stroke={C.primary} strokeWidth="1.5" />
    <text x="140" y="28" textAnchor="middle" fontSize="12" fill={C.primary} fontWeight="600">Weight [H × 4H]</text>

    {/* 分发箭头 */}
    <path d="M90 40 L55 58" stroke={C.border} strokeWidth="1.5" />
    <path d="M140 40 L140 58" stroke={C.border} strokeWidth="1.5" />
    <path d="M190 40 L225 58" stroke={C.border} strokeWidth="1.5" />

    {/* GPU 方块 */}
    {[0, 1, 2].map(i => (
      <g key={i} transform={`translate(${20 + i * 90}, 62)`}>
        <rect width="70" height="68" rx="5" fill="#fff" stroke={C.border} strokeWidth="1.5" />
        <text x="35" y="16" textAnchor="middle" fontSize="11" fill={C.text} fontWeight="600">GPU {i}</text>
        <line x1="0" y1="22" x2="70" y2="22" stroke={C.border} strokeWidth="1" />
        {/* 权重分片 */}
        <rect x="12" y="30" width="46" height="30" rx="3" fill={C.primaryLight} stroke={C.primary} strokeWidth="1.5" />
        <text x="35" y="49" textAnchor="middle" fontSize="11" fill={C.primary} fontWeight="600">W[{i}]</text>
      </g>
    ))}
  </svg>
)

const DiagramPP: React.FC = () => (
  <svg width="280" height="140" viewBox="0 0 280 140">
    {/* Pipeline Stages */}
    {[0, 1, 2].map(i => (
      <g key={i} transform={`translate(${15 + i * 90}, 10)`}>
        <rect width="75" height="120" rx="5" fill="#fff" stroke={C.border} strokeWidth="1.5" />
        <text x="37" y="18" textAnchor="middle" fontSize="11" fill={C.text} fontWeight="600">Stage {i}</text>
        <line x1="0" y1="24" x2="75" y2="24" stroke={C.border} strokeWidth="1" />
        <text x="37" y="40" textAnchor="middle" fontSize="10" fill={C.textSec}>GPU {i}</text>
        {/* 层 - 带标签 */}
        {[0, 1, 2, 3].map(j => (
          <g key={j}>
            <rect x="8" y={48 + j * 17} width="59" height="14" rx="2" fill={C.primaryLight} stroke={C.primary} strokeWidth="1" />
            <text x="37" y={58 + j * 17} textAnchor="middle" fontSize="8" fill={C.primary}>L{i * 4 + j}</text>
          </g>
        ))}
        {/* 箭头 */}
        {i < 2 && (
          <g transform="translate(78, 70)">
            <line x1="0" y1="0" x2="8" y2="0" stroke={C.textSec} strokeWidth="2" />
            <polygon points="8,0 4,-4 4,4" fill={C.textSec} />
          </g>
        )}
      </g>
    ))}
  </svg>
)

const DiagramEP: React.FC = () => (
  <svg width="280" height="140" viewBox="0 0 280 140">
    {/* Router */}
    <rect x="80" y="8" width="120" height="24" rx="4" fill={C.primaryLight} stroke={C.primary} strokeWidth="1.5" />
    <text x="140" y="24" textAnchor="middle" fontSize="11" fill={C.primary} fontWeight="600">Router</text>

    {/* AllToAll */}
    <rect x="90" y="38" width="100" height="18" rx="3" fill={C.bg} stroke={C.border} strokeWidth="1" />
    <text x="140" y="50" textAnchor="middle" fontSize="10" fill={C.textSec} fontWeight="500">AllToAll</text>

    {/* GPU + Experts */}
    {[0, 1, 2].map(i => (
      <g key={i} transform={`translate(${20 + i * 90}, 62)`}>
        <rect width="70" height="70" rx="5" fill="#fff" stroke={C.border} strokeWidth="1.5" />
        <text x="35" y="14" textAnchor="middle" fontSize="10" fill={C.text} fontWeight="600">GPU {i}</text>
        <line x1="0" y1="20" x2="70" y2="20" stroke={C.border} strokeWidth="1" />
        {/* 专家 */}
        <rect x="6" y="26" width="26" height="36" rx="3" fill={C.primaryLight} stroke={C.primary} strokeWidth="1" />
        <text x="19" y="48" textAnchor="middle" fontSize="10" fill={C.primary} fontWeight="500">E{i*2}</text>
        <rect x="38" y="26" width="26" height="36" rx="3" fill={C.primaryLight} stroke={C.primary} strokeWidth="1" />
        <text x="51" y="48" textAnchor="middle" fontSize="10" fill={C.primary} fontWeight="500">E{i*2+1}</text>
      </g>
    ))}
  </svg>
)

const DiagramSP: React.FC = () => (
  <svg width="280" height="140" viewBox="0 0 280 140">
    {/* 顶部：Sequence */}
    <rect x="40" y="10" width="200" height="26" rx="4" fill={C.primaryLight} stroke={C.primary} strokeWidth="1.5" />
    <text x="140" y="27" textAnchor="middle" fontSize="12" fill={C.primary} fontWeight="600">Sequence [B, S, H]</text>

    {/* 分发箭头 */}
    <path d="M90 38 L55 55" stroke={C.border} strokeWidth="1.5" />
    <path d="M140 38 L140 55" stroke={C.border} strokeWidth="1.5" />
    <path d="M190 38 L225 55" stroke={C.border} strokeWidth="1.5" />

    {/* GPU 方块 */}
    {[0, 1, 2].map(i => (
      <g key={i} transform={`translate(${20 + i * 90}, 58)`}>
        <rect width="70" height="72" rx="5" fill="#fff" stroke={C.border} strokeWidth="1.5" />
        <text x="35" y="16" textAnchor="middle" fontSize="11" fill={C.text} fontWeight="600">GPU {i}</text>
        <line x1="0" y1="22" x2="70" y2="22" stroke={C.border} strokeWidth="1" />
        {/* 序列分片 */}
        <rect x="8" y="28" width="54" height="16" rx="3" fill={C.primaryLight} stroke={C.primary} strokeWidth="1" />
        <text x="35" y="39" textAnchor="middle" fontSize="9" fill={C.primary}>Seq[{i}::3]</text>
        {/* TP 分片 */}
        <rect x="8" y="48" width="54" height="16" rx="3" fill={C.bg} stroke={C.border} strokeWidth="1" />
        <text x="35" y="59" textAnchor="middle" fontSize="9" fill={C.textSec}>+ TP Shard</text>
      </g>
    ))}
  </svg>
)

const DIAGRAMS: Record<ParallelismType, React.FC> = {
  dp: DiagramDP,
  tp: DiagramTP,
  pp: DiagramPP,
  ep: DiagramEP,
  sp: DiagramSP,
}

// ============================================
// 主组件 - 简洁的左右布局
// ============================================

export const ParallelismInfo: React.FC<ParallelismInfoProps> = ({ type }) => {
  const info = INFO[type]
  const color = COLORS[type]
  const Diagram = DIAGRAMS[type]

  return (
    <div style={{
      display: 'flex',
      gap: 20,
      padding: 16,
      background: color.light,
      borderRadius: 8,
      border: `1px solid ${color.primary}33`,
    }}>
      {/* 左侧：图示 */}
      <div style={{
        flex: '0 0 280px',
        background: '#fff',
        borderRadius: 6,
        padding: 8,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <Diagram />
      </div>

      {/* 右侧：说明 */}
      <div style={{ flex: 1, minWidth: 0 }}>
        {/* 标题 */}
        <div style={{ marginBottom: 10 }}>
          <span style={{ fontSize: 16, fontWeight: 700, color: color.dark }}>
            {info.fullName}
          </span>
          <span style={{ fontSize: 13, color: '#8c8c8c', marginLeft: 8 }}>
            {info.shortDesc}
          </span>
        </div>

        {/* 定义 */}
        <div style={{ fontSize: 13, color: '#595959', lineHeight: 1.7, marginBottom: 12 }}>
          {info.definition}
        </div>

        {/* 关键特点 - 横向标签 */}
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
          {info.keyPoints.map((point, i) => (
            <span key={i} style={{
              padding: '3px 10px',
              background: '#fff',
              border: `1px solid ${color.primary}66`,
              borderRadius: 12,
              fontSize: 12,
              color: color.dark,
            }}>
              {point}
            </span>
          ))}
        </div>

        {/* 通信和适用场景 */}
        <div style={{ display: 'flex', gap: 16, fontSize: 12 }}>
          <div>
            <span style={{ color: '#8c8c8c' }}>通信: </span>
            <span style={{ color: '#262626', fontWeight: 500 }}>{info.communication}</span>
          </div>
          <div style={{ flex: 1 }}>
            <span style={{ color: '#8c8c8c' }}>适用: </span>
            <span style={{ color: '#262626' }}>{info.bestFor}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

// ============================================
// 并行策略卡片组件
// ============================================

interface ParallelismCardProps {
  type: ParallelismType
  value: number
  selected: boolean
  onClick: () => void
}

export const ParallelismCard: React.FC<ParallelismCardProps> = ({ type, value, selected, onClick }) => {
  const info = INFO[type]
  const color = COLORS[type]

  return (
    <div
      onClick={onClick}
      style={{
        flex: 1,
        minWidth: 90,
        padding: '8px 12px',
        background: selected ? color.light : '#fff',
        border: `1.5px solid ${selected ? color.primary : '#e8e8e8'}`,
        borderRadius: 8,
        cursor: 'pointer',
        transition: 'all 0.2s',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
      }}
    >
      <div style={{
        fontSize: 28,
        fontWeight: 700,
        color: selected ? color.primary : '#262626',
        lineHeight: 1,
        minWidth: 32,
        textAlign: 'center',
      }}>
        {value}
      </div>
      <div style={{ flex: 1 }}>
        <div style={{
          fontSize: 16,
          fontWeight: 700,
          color: selected ? color.primary : '#262626',
          lineHeight: 1.2,
        }}>
          {info.name}
        </div>
        <div style={{
          fontSize: 11,
          color: selected ? color.primary : '#8c8c8c',
          marginTop: 2,
        }}>
          {info.shortDesc}
        </div>
      </div>
    </div>
  )
}

export default ParallelismInfo
