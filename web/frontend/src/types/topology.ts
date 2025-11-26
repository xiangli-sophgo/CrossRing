// 拓扑相关类型定义

export interface TopologyNode {
  id: number
  label: string
  row: number
  col: number
  x: number
  y: number
}

export interface TopologyEdge {
  source: number
  target: number
  direction: 'horizontal' | 'vertical'
  type: 'row_link' | 'col_link'
}

export interface TopologyData {
  type: string
  rows: number
  cols: number
  total_nodes: number
  nodes: TopologyNode[]
  edges: TopologyEdge[]
  metadata: {
    row_links: number
    col_links: number
    total_links: number
  }
}

export interface TopologyInfo {
  type: string
  rows: number
  cols: number
  nodes: number
}
