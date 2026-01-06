/**
 * 知识图谱类型定义
 */

// 知识分类 (8个核心分组)
export type KnowledgeCategory =
  | 'hardware'           // 硬件 (GPU, NPU, TPU, HBM, Server, Rack, Pod)
  | 'interconnect'       // 互联 (NVLink, PCIe, InfiniBand, CXL, 拓扑)
  | 'parallel'           // 并行 (TP, SP, EP, DP, PP, Scale Up/Out)
  | 'communication'      // 通信 (AllReduce, RDMA, NCCL, MPI, 集合通信)
  | 'model'              // 模型 (Transformer, Attention, MoE, KV Cache)
  | 'inference'          // 推理 (Prefill, Decode, FlashAttention, 量化)
  | 'protocol'           // 协议 (ECC, FEC, PFC, CBFC, 编码调制)
  | 'system'             // 系统 (地址空间, 虚拟化, RAS, 事务)

// 分类颜色映射
export const CATEGORY_COLORS: Record<KnowledgeCategory, string> = {
  hardware: '#059669',           // 绿色 - 硬件基础
  interconnect: '#0284C7',       // 天蓝 - 互联技术
  parallel: '#7C3AED',           // 紫色 - 并行策略
  communication: '#D97706',      // 橙色 - 通信
  model: '#C026D3',              // 洋红 - 模型架构
  inference: '#DB2777',          // 粉色 - 推理优化
  protocol: '#4F46E5',           // 靛蓝 - 协议
  system: '#0891B2',             // 青色 - 系统
}

// 分类中文名称
export const CATEGORY_NAMES: Record<KnowledgeCategory, string> = {
  hardware: '硬件',
  interconnect: '互联',
  parallel: '并行',
  communication: '通信',
  model: '模型',
  inference: '推理',
  protocol: '协议',
  system: '系统',
}

// 知识节点
export interface KnowledgeNode {
  id: string
  name: string
  fullName?: string
  definition: string
  category: KnowledgeCategory
  source?: string
  notes?: string
  aliases?: string[]
}

// 关系类型
export type RelationType = 'related_to' | 'belongs_to' | 'depends_on' | 'contrasts_with'

// 关系样式（统一实线）
export const RELATION_STYLES: Record<RelationType, { stroke: string }> = {
  related_to: { stroke: '#94A3B8' },
  belongs_to: { stroke: '#94A3B8' },
  depends_on: { stroke: '#94A3B8' },
  contrasts_with: { stroke: '#94A3B8' },
}

// 知识关系
export interface KnowledgeRelation {
  source: string
  target: string
  type: RelationType
  description?: string
}

// 知识图谱数据
export interface KnowledgeGraphData {
  nodes: KnowledgeNode[]
  relations: KnowledgeRelation[]
  metadata?: {
    version: string
    nodeCount: number
    relationCount: number
  }
}

// 力导向节点（扩展）
export interface ForceKnowledgeNode extends KnowledgeNode {
  x: number
  y: number
  vx?: number
  vy?: number
  fx?: number | null
  fy?: number | null
  index?: number
}
