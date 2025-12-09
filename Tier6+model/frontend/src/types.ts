// 层级配置
export interface LevelConfig {
  level: string
  count: number
  topology: string
  visible: boolean
}

// 节点数据
export interface NodeData {
  id: string
  level: string
  position: [number, number, number]
  color: string
}

// 边数据
export interface EdgeData {
  source: string
  target: string
  type: 'intra_level' | 'inter_level'
}

// 拓扑数据
export interface TopologyData {
  nodes: NodeData[]
  edges: EdgeData[]
}

// 层级名称映射
export const LEVEL_NAMES: Record<string, string> = {
  die: 'Die (晶粒)',
  chip: 'Chip (芯片)',
  board: 'Board (电路板)',
  server: 'Server (服务器)',
  pod: 'Pod (机柜)',
}

// 拓扑类型
export const TOPOLOGY_TYPES = [
  { value: 'mesh', label: 'Mesh (全连接)' },
  { value: 'all_to_all', label: 'All-to-All (分组全连接)' },
  { value: 'ring', label: 'Ring (环形)' },
]

// 层级颜色
export const LEVEL_COLORS: Record<string, string> = {
  die: '#f5222d',
  chip: '#722ed1',
  board: '#52c41a',
  server: '#1890ff',
  pod: '#fa8c16',
}
