import type { TopologyData, TopologyNode, TopologyEdge } from '../types/topology'

export function generateTopology(topoType: string): TopologyData {
  const [rows, cols] = topoType.split('x').map(Number)

  // 生成节点
  const nodes: TopologyNode[] = []
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const nodeId = row * cols + col
      nodes.push({
        id: nodeId,
        label: `Node ${nodeId}`,
        row,
        col,
        x: col * 100,
        y: row * 100,
      })
    }
  }

  // 生成边（行链 + 列链）
  const edges: TopologyEdge[] = []

  // 行链
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols - 1; col++) {
      edges.push({
        source: row * cols + col,
        target: row * cols + col + 1,
        direction: 'horizontal',
        type: 'row_link',
      })
    }
  }

  // 列链
  for (let col = 0; col < cols; col++) {
    for (let row = 0; row < rows - 1; row++) {
      edges.push({
        source: row * cols + col,
        target: (row + 1) * cols + col,
        direction: 'vertical',
        type: 'col_link',
      })
    }
  }

  return {
    type: topoType,
    rows,
    cols,
    total_nodes: rows * cols,
    nodes,
    edges,
    metadata: {
      row_links: rows * (cols - 1),
      col_links: cols * (rows - 1),
      total_links: edges.length * 2,
    },
  }
}

export function getNodeNeighbors(nodeId: number, rows: number, cols: number): number[] {
  const row = Math.floor(nodeId / cols)
  const col = nodeId % cols
  const neighbors: number[] = []

  if (col > 0) neighbors.push(row * cols + col - 1)
  if (col < cols - 1) neighbors.push(row * cols + col + 1)
  if (row > 0) neighbors.push((row - 1) * cols + col)
  if (row < rows - 1) neighbors.push((row + 1) * cols + col)

  return neighbors
}

export interface NodeInfo {
  node_id: number
  position: { row: number; col: number }
  label: string
  neighbors: number[]
  degree: number
  topology: string
  die_id?: number
}

export function getNodeInfoLocal(topoType: string, nodeId: number, dieId?: number): NodeInfo {
  const [rows, cols] = topoType.split('x').map(Number)
  const row = Math.floor(nodeId / cols)
  const col = nodeId % cols
  const neighbors = getNodeNeighbors(nodeId, rows, cols)

  return {
    node_id: nodeId,
    position: { row, col },
    label: `Node ${nodeId}`,
    neighbors,
    degree: neighbors.length,
    topology: topoType,
    die_id: dieId,
  }
}
