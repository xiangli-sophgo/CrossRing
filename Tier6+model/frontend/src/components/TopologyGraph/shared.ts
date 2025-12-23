import {
  HierarchicalTopology,
  PodConfig,
  RackConfig,
  BoardConfig,
  ManualConnection,
  ConnectionMode,
  HierarchyLevel,
  LayoutType,
  MultiLevelViewOptions,
  TrafficAnalysisResult,
} from '../../types'

// 重新导出供子组件使用的类型
export type { HierarchyLevel, LayoutType, MultiLevelViewOptions }

// 根据板卡U高度区分颜色
export const BOARD_U_COLORS: Record<number, string> = {
  1: '#13c2c2',  // 1U - 青色
  2: '#722ed1',  // 2U - 紫色
  4: '#eb2f96',  // 4U - 洋红色
}

export interface BreadcrumbItem {
  level: string
  id: string
  label: string
}

// 节点详细信息
export interface NodeDetail {
  id: string
  label: string
  type: string
  subType?: string
  connections: { id: string; label: string; bandwidth?: number; latency?: number }[]
  portInfo?: { uplink: number; downlink: number; inter: number }
}

// 连接详细信息
export interface LinkDetail {
  id: string  // source-target 格式
  sourceId: string
  sourceLabel: string
  sourceType: string
  targetId: string
  targetLabel: string
  targetType: string
  bandwidth?: number
  latency?: number
  isManual?: boolean
}

export interface TopologyGraphProps {
  visible: boolean
  onClose: () => void
  topology: HierarchicalTopology | null
  currentLevel: 'datacenter' | 'pod' | 'rack' | 'board' | 'chip'
  currentPod?: PodConfig | null
  currentRack?: RackConfig | null
  currentBoard?: BoardConfig | null
  onNodeDoubleClick?: (nodeId: string, nodeType: string) => void
  onNodeClick?: (nodeDetail: NodeDetail | null) => void
  onLinkClick?: (linkDetail: LinkDetail | null) => void
  selectedNodeId?: string | null  // 当前选中的节点ID
  selectedLinkId?: string | null  // 当前选中的连接ID
  onNavigateBack?: () => void
  onBreadcrumbClick?: (index: number) => void
  breadcrumbs?: BreadcrumbItem[]
  canGoBack?: boolean
  embedded?: boolean  // 嵌入模式（非弹窗）
  // 编辑连接相关
  connectionMode?: ConnectionMode
  selectedNodes?: Set<string>  // 源节点集合
  onSelectedNodesChange?: (nodes: Set<string>) => void
  targetNodes?: Set<string>  // 目标节点集合
  onTargetNodesChange?: (nodes: Set<string>) => void
  sourceNode?: string | null  // 保留兼容
  onSourceNodeChange?: (nodeId: string | null) => void
  onManualConnect?: (sourceId: string, targetId: string, level: HierarchyLevel) => void
  manualConnections?: ManualConnection[]
  onDeleteManualConnection?: (connectionId: string) => void
  onDeleteConnection?: (source: string, target: string) => void  // 删除任意连接（包括自动生成的）
  layoutType?: LayoutType  // 布局类型
  onLayoutTypeChange?: (type: LayoutType) => void  // 布局类型变更回调
  // 多层级视图相关
  multiLevelOptions?: MultiLevelViewOptions
  onMultiLevelOptionsChange?: (options: MultiLevelViewOptions) => void
  // 流量分析热力图
  trafficAnalysisResult?: TrafficAnalysisResult | null
}

export interface Node {
  id: string
  label: string
  type: string
  subType?: string  // Switch的层级，如 "leaf", "spine"
  isSwitch?: boolean
  x: number
  y: number
  color: string
  portInfo?: {
    uplink: number
    downlink: number
    inter: number
  }
  // Torus布局的网格位置
  gridRow?: number
  gridCol?: number
  gridZ?: number  // 3D Torus的Z层
  uHeight?: number  // Board的U高度
  // 多层级视图属性
  parentId?: string              // 父节点ID（下层节点使用）
  hierarchyLevel?: HierarchyLevel // 所属层级
  isContainer?: boolean          // 是否为容器节点
  zLayer?: number                // Z层 (0=下层, 1=上层)
  containerBounds?: {            // 容器边界
    x: number
    y: number
    width: number
    height: number
  }
  // 多层级模式：容器内的单层级完整布局数据（用于展开动画）
  singleLevelData?: {
    nodes: Node[]
    edges: Edge[]
    viewBox: { width: number; height: number }
    scale: number  // 从单层级视图到容器内视图的缩放比例
    directTopology?: string  // 布局类型，用于判断是否需要曲线连接
  }
}

export interface Edge {
  source: string
  target: string
  bandwidth?: number
  latency?: number  // 延迟 (ns)
  isSwitch?: boolean  // 是否为Switch连接
  // 多层级视图属性
  connectionType?: 'intra_upper' | 'intra_lower' | 'inter_level'  // 连接类型
  // 单层级视图：跨层级连接属性
  isExternal?: boolean  // 是否连接到当前层级之外
  externalDirection?: 'upper' | 'lower'  // 外部连接方向（上层/下层）
  externalNodeId?: string  // 外部节点ID
  externalNodeLabel?: string  // 外部节点标签
  // 间接连接属性（通过上层Switch）
  isIndirect?: boolean  // 是否为间接连接
  viaNodeId?: string  // 中转节点ID
  viaNodeLabel?: string  // 中转节点标签
}
