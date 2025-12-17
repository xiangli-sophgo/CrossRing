import { Node } from './shared'

// ============================================
// 等轴测投影工具函数
// ============================================

// 等轴测投影参数
const ISO_ANGLE = Math.PI / 6  // 30度
const ISO_SCALE_X = Math.cos(ISO_ANGLE)  // X轴缩放 ≈ 0.866
const ISO_SCALE_Y = Math.sin(ISO_ANGLE)  // Y轴偏移 ≈ 0.5

// 等轴测坐标转换：将3D坐标(x, y, z)投影到2D
export function isoProject(x: number, y: number, z: number): { px: number; py: number } {
  const px = (x - z) * ISO_SCALE_X
  const py = (x + z) * ISO_SCALE_Y - y
  return { px, py }
}

// 容器边界
export interface ContainerBounds {
  x: number
  y: number
  width: number
  height: number
}

// 堆叠布局选项
export interface StackedLayoutOptions {
  layerGap: number           // 层间垂直距离
  containerPadding: number   // 容器内边距
  innerNodeScale: number     // 内部节点缩放比例
  upperNodeSize: number      // 上层节点大小
  lowerNodeSize: number      // 下层节点大小
}

// 堆叠布局结果
export interface StackedLayoutResult {
  upperNodes: Node[]
  lowerNodes: Node[]
  containerBounds: Map<string, ContainerBounds>
}

// 默认堆叠布局选项
const DEFAULT_STACKED_OPTIONS: StackedLayoutOptions = {
  layerGap: 120,
  containerPadding: 30,
  innerNodeScale: 0.6,
  upperNodeSize: 60,
  lowerNodeSize: 25,
}

// 多层堆叠布局 - Z轴垂直堆叠效果
// 每个上层节点展开为一个容器，内部显示下层节点的拓扑
// 多个容器在Z轴方向垂直堆叠，形成卡片堆叠效果，支持悬停抬起交互
export function isometricStackedLayout(
  upperNodes: Node[],
  lowerNodesMap: Map<string, Node[]>,
  width: number,
  height: number,
  options: Partial<StackedLayoutOptions> = {}
): StackedLayoutResult {
  const opts = { ...DEFAULT_STACKED_OPTIONS, ...options }
  const { containerPadding, innerNodeScale } = opts

  // 如果没有上层节点，返回空结果
  if (upperNodes.length === 0) {
    return { upperNodes: [], lowerNodes: [], containerBounds: new Map() }
  }

  const containerBounds = new Map<string, ContainerBounds>()
  const layoutedLower: Node[] = []
  const layoutedUpper: Node[] = []

  // 计算上层节点数量来确定布局（在Z方向堆叠）
  const upperCount = upperNodes.length

  // 每个容器的基础大小
  const containerWidth = Math.min(width * 0.85, 600)
  const containerHeight = 200  // 固定高度

  // 书本堆叠参数
  const bookThickness = 30   // 每本书的"厚度"（层间露出的高度）
  const depth3D = 20         // 3D深度（顶面和侧面的高度）

  // 计算整体堆叠的高度
  const totalStackHeight = containerHeight + (upperCount - 1) * bookThickness + depth3D

  // 计算起始位置，使整体居中
  // zLayer=0 在最上面（Y最小），zLayer越大越在下面（Y越大）
  const baseX = width / 2
  const baseY = (height - totalStackHeight) / 2 + depth3D + containerHeight / 2

  // 计算每个容器（展开的上层节点）及其内部的下层节点
  upperNodes.forEach((upperNode, idx) => {
    // 书本堆叠：idx=0 的 zLayer=0 在最上面
    const zIndex = idx

    // 容器中心位置 - zLayer=0在最上面（Y最小），zLayer越大越在下面（Y越大）
    const containerCenterX = baseX
    const containerCenterY = baseY + zIndex * bookThickness

    // 容器边界（作为展开的上层节点）
    const bounds: ContainerBounds = {
      x: containerCenterX - containerWidth / 2,
      y: containerCenterY - containerHeight / 2,
      width: containerWidth,
      height: containerHeight,
    }
    containerBounds.set(upperNode.id, bounds)

    // 上层节点作为容器
    layoutedUpper.push({
      ...upperNode,
      x: containerCenterX,
      y: containerCenterY,
      isContainer: true,
      zLayer: zIndex,
      containerBounds: bounds,
    })

    // 获取下层子节点
    const children = lowerNodesMap.get(upperNode.id) || []
    const childCount = children.length

    if (childCount === 0) return

    // 布局下层子节点（在容器内使用网格布局）
    const childCols = Math.ceil(Math.sqrt(childCount))
    const childRows = Math.ceil(childCount / childCols)
    const innerWidth = (containerWidth - containerPadding * 2) * innerNodeScale
    const innerHeight = (containerHeight - containerPadding * 2) * innerNodeScale
    const childSpacingX = childCols > 1 ? innerWidth / (childCols - 1) : 0
    const childSpacingY = childRows > 1 ? innerHeight / (childRows - 1) : 0

    children.forEach((child, i) => {
      const childCol = i % childCols
      const childRow = Math.floor(i / childCols)

      // 子节点在容器内的相对位置
      const relX = childCols === 1 ? 0 : (childCol - (childCols - 1) / 2) * childSpacingX
      const relY = childRows === 1 ? 0 : (childRow - (childRows - 1) / 2) * childSpacingY

      layoutedLower.push({
        ...child,
        x: containerCenterX + relX,
        y: containerCenterY + relY,
        parentId: upperNode.id,
        zLayer: zIndex,  // 与父容器相同的zLayer
      })
    })
  })

  return { upperNodes: layoutedUpper, lowerNodes: layoutedLower, containerBounds }
}

// 布局算法：圆形布局
export function circleLayout(nodes: Node[], centerX: number, centerY: number, radius: number): Node[] {
  const count = nodes.length
  // 只有一个节点时，放在中心
  if (count === 1) {
    return [{ ...nodes[0], x: centerX, y: centerY }]
  }
  return nodes.map((node, i) => ({
    ...node,
    x: centerX + radius * Math.cos((2 * Math.PI * i) / count - Math.PI / 2),
    y: centerY + radius * Math.sin((2 * Math.PI * i) / count - Math.PI / 2),
  }))
}

// 布局算法：环形拓扑布局（用于ring连接）
export function ringLayout(nodes: Node[], centerX: number, centerY: number, radius: number): Node[] {
  const count = nodes.length
  if (count === 1) {
    return [{ ...nodes[0], x: centerX, y: centerY }]
  }
  return nodes.map((node, i) => ({
    ...node,
    x: centerX + radius * Math.cos((2 * Math.PI * i) / count - Math.PI / 2),
    y: centerY + radius * Math.sin((2 * Math.PI * i) / count - Math.PI / 2),
  }))
}

// 布局算法：2D Torus/网格布局（用于torus_2d和grid连接）
// 标准Torus可视化：节点排成规则网格，环绕边画在外围
export function torusLayout(nodes: Node[], width: number, height: number, padding: number = 120): Node[] {
  const count = nodes.length
  if (count === 1) {
    return [{ ...nodes[0], x: width / 2, y: height / 2 }]
  }
  // 计算最佳的行列数，尽量接近正方形
  const cols = Math.ceil(Math.sqrt(count))
  const rows = Math.ceil(count / cols)

  // 留出较大边距给环绕连接线
  const innerWidth = width - padding * 2
  const innerHeight = height - padding * 2
  const spacingX = cols > 1 ? innerWidth / (cols - 1) : 0
  const spacingY = rows > 1 ? innerHeight / (rows - 1) : 0

  // 居中偏移
  const offsetX = cols === 1 ? width / 2 : padding
  const offsetY = rows === 1 ? height / 2 : padding

  return nodes.map((node, i) => ({
    ...node,
    x: offsetX + (i % cols) * spacingX,
    y: offsetY + Math.floor(i / cols) * spacingY,
    // 存储网格位置信息用于连接线计算
    gridRow: Math.floor(i / cols),
    gridCol: i % cols,
  }))
}

// 计算Torus网格的行列数
export function getTorusGridSize(count: number): { cols: number; rows: number } {
  const cols = Math.ceil(Math.sqrt(count))
  const rows = Math.ceil(count / cols)
  return { cols, rows }
}

// 3D Torus专用布局：等轴测投影，呈现3D立方体效果
export function torus3DLayout(nodes: Node[], width: number, height: number, _padding: number = 100): Node[] {
  const count = nodes.length
  if (count <= 1) {
    return nodes.map(n => ({ ...n, x: width / 2, y: height / 2, gridRow: 0, gridCol: 0, gridZ: 0 }))
  }

  // 计算3D维度（尽量接近立方体）
  const dim = Math.max(2, Math.ceil(Math.pow(count, 1 / 3)))
  const nodesPerLayer = dim * dim

  // 等轴测投影参数
  const centerX = width / 2
  const centerY = height / 2
  const spacingX = 140  // X方向间距
  const spacingY = 120  // Y方向间距（垂直）
  const spacingZ = 90   // Z方向间距（深度，斜向）

  return nodes.map((node, i) => {
    const z = Math.floor(i / nodesPerLayer)
    const inLayerIndex = i % nodesPerLayer
    const row = Math.floor(inLayerIndex / dim)  // Y轴（上下）
    const col = inLayerIndex % dim              // X轴（左右）

    // 等轴测投影：
    // X轴向右，Y轴向下，Z轴向右上方（模拟深度）
    const x = centerX + (col - (dim - 1) / 2) * spacingX + (z - (dim - 1) / 2) * spacingZ * 0.6
    const y = centerY + (row - (dim - 1) / 2) * spacingY - (z - (dim - 1) / 2) * spacingZ * 0.5

    return {
      ...node,
      x,
      y,
      gridRow: row,
      gridCol: col,
      gridZ: z,
    }
  })
}

// 计算3D Torus的维度
export function getTorus3DSize(count: number): { dim: number; layers: number } {
  const dim = Math.max(2, Math.ceil(Math.pow(count, 1 / 3)))
  const layers = Math.ceil(count / (dim * dim))
  return { dim, layers }
}

// 根据直连拓扑类型选择最佳布局
export function getLayoutForTopology(
  topologyType: string,
  nodes: Node[],
  width: number,
  height: number
): Node[] {
  const centerX = width / 2
  const centerY = height / 2
  const radius = Math.min(width, height) * 0.35

  switch (topologyType) {
    case 'ring':
      return ringLayout(nodes, centerX, centerY, radius)
    case 'torus_2d':
      return torusLayout(nodes, width, height)
    case 'torus_3d':
      return torus3DLayout(nodes, width, height)
    case 'full_mesh_2d':
      // 2D FullMesh使用网格布局（行列全连接）
      return torusLayout(nodes, width, height)
    case 'full_mesh':
      // 全连接用圆形布局最清晰
      return circleLayout(nodes, centerX, centerY, radius)
    case 'none':
    default:
      // 无连接或默认用圆形
      return circleLayout(nodes, centerX, centerY, radius)
  }
}

// 布局算法：分层布局（用于显示Switch层级，设备节点排成一排）
export function hierarchicalLayout(nodes: Node[], width: number, height: number): Node[] {
  // 按类型分组
  const switchNodes = nodes.filter(n => n.isSwitch)
  const deviceNodes = nodes.filter(n => !n.isSwitch)

  // 如果没有Switch，设备节点居中显示
  if (switchNodes.length === 0) {
    const centerY = height / 2
    if (deviceNodes.length === 1) {
      return [{ ...deviceNodes[0], x: width / 2, y: centerY }]
    }
    const spacing = width / (deviceNodes.length + 1)
    return deviceNodes.map((node, i) => ({
      ...node,
      x: spacing * (i + 1),
      y: centerY,
    }))
  }

  // Switch按subType分层
  const switchLayers: Record<string, Node[]> = {}
  switchNodes.forEach(n => {
    const layer = n.subType || 'default'
    if (!switchLayers[layer]) switchLayers[layer] = []
    switchLayers[layer].push(n)
  })

  // 层级顺序：device在最下面，然后是leaf, spine, core
  const layerOrder = ['leaf', 'spine', 'core']
  const sortedLayers = Object.keys(switchLayers).sort((a, b) => {
    const aIdx = layerOrder.indexOf(a)
    const bIdx = layerOrder.indexOf(b)
    return (aIdx === -1 ? 999 : aIdx) - (bIdx === -1 ? 999 : bIdx)
  })

  const totalLayers = sortedLayers.length + (deviceNodes.length > 0 ? 1 : 0)
  const layerSpacing = 100 // 每层之间的间距
  const totalHeight = (totalLayers - 1) * layerSpacing
  const startY = (height + totalHeight) / 2 // 垂直居中的起始Y（最底层）

  const result: Node[] = []

  // 设备节点在最底层
  if (deviceNodes.length > 0) {
    const y = startY
    const spacing = width / (deviceNodes.length + 1)
    deviceNodes.forEach((node, i) => {
      result.push({ ...node, x: spacing * (i + 1), y })
    })
  }

  // Switch节点按层级向上排列（在设备上方）
  sortedLayers.forEach((layer, layerIdx) => {
    const layerNodes = switchLayers[layer]
    const y = startY - layerSpacing * (layerIdx + (deviceNodes.length > 0 ? 1 : 0))
    const spacing = width / (layerNodes.length + 1)
    layerNodes.forEach((node, i) => {
      result.push({ ...node, x: spacing * (i + 1), y })
    })
  })

  return result
}

// 布局算法：混合布局（设备节点按拓扑排列，Switch节点在上方中央）
// 用于同时有Switch和节点直连的场景
export function hybridLayout(
  nodes: Node[],
  width: number,
  height: number,
  directTopology: string
): Node[] {
  const switchNodes = nodes.filter(n => n.isSwitch)
  const deviceNodes = nodes.filter(n => !n.isSwitch)

  // 如果没有Switch，使用普通拓扑布局
  if (switchNodes.length === 0) {
    return getLayoutForTopology(directTopology, deviceNodes, width, height)
  }

  // Switch层数决定Switch区域高度
  const switchLayers: Record<string, Node[]> = {}
  switchNodes.forEach(n => {
    const layer = n.subType || 'default'
    if (!switchLayers[layer]) switchLayers[layer] = []
    switchLayers[layer].push(n)
  })
  const switchLayerCount = Object.keys(switchLayers).length

  // 动态计算区域划分：Switch区域更紧凑
  const switchLayerHeight = 50  // 每层Switch的高度
  const switchAreaHeight = switchLayerCount * switchLayerHeight
  const switchAreaTop = 60  // Switch起始位置（留出顶部空间）
  const gapBetween = 40  // Switch和设备之间的间隙

  // 设备节点区域
  const deviceAreaTop = switchAreaTop + switchAreaHeight + gapBetween
  const deviceAreaHeight = height - deviceAreaTop - 30  // 底部留30px

  const result: Node[] = []

  // 1. 设备节点按拓扑类型布局（在下方区域）
  const centerX = width / 2
  const centerY = deviceAreaTop + deviceAreaHeight / 2
  const radius = Math.min(width * 0.4, deviceAreaHeight * 0.45)

  let layoutedDevices: Node[]
  switch (directTopology) {
    case 'ring':
      layoutedDevices = ringLayout(deviceNodes, centerX, centerY, radius)
      break
    case 'torus_2d':
      layoutedDevices = torusLayout(deviceNodes, width, deviceAreaHeight, 80)
      layoutedDevices = layoutedDevices.map(n => ({ ...n, y: n.y + deviceAreaTop }))
      break
    case 'torus_3d':
      layoutedDevices = torus3DLayout(deviceNodes, width, deviceAreaHeight, 60)
      layoutedDevices = layoutedDevices.map(n => ({ ...n, y: n.y + deviceAreaTop - 30 }))
      break
    case 'full_mesh_2d':
      layoutedDevices = torusLayout(deviceNodes, width, deviceAreaHeight, 80)
      layoutedDevices = layoutedDevices.map(n => ({ ...n, y: n.y + deviceAreaTop }))
      break
    case 'full_mesh':
    default:
      layoutedDevices = circleLayout(deviceNodes, centerX, centerY, radius)
      break
  }
  result.push(...layoutedDevices)

  // 2. Switch节点按层级排列（在上方区域）
  const layerOrder = ['leaf', 'spine', 'core']
  const sortedLayers = Object.keys(switchLayers).sort((a, b) => {
    const aIdx = layerOrder.indexOf(a)
    const bIdx = layerOrder.indexOf(b)
    return (aIdx === -1 ? 999 : aIdx) - (bIdx === -1 ? 999 : bIdx)
  })

  sortedLayers.forEach((layer, layerIdx) => {
    const layerNodes = switchLayers[layer]
    const y = switchAreaTop + layerIdx * switchLayerHeight
    const spacing = width / (layerNodes.length + 1)
    layerNodes.forEach((node, i) => {
      result.push({ ...node, x: spacing * (i + 1), y })
    })
  })

  return result
}
