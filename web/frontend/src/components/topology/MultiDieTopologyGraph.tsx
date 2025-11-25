import { useEffect, useRef, useMemo, forwardRef, useImperativeHandle } from 'react'
import CytoscapeComponent from 'react-cytoscapejs'
import Cytoscape from 'cytoscape'
import { Card, Space, Tag, Button, Row, Col, message } from 'antd'
import { ZoomInOutlined, ZoomOutOutlined, AimOutlined, SaveOutlined } from '@ant-design/icons'
import type { TopologyData } from '../../types/topology'
import type { IPMount } from '../../types/ipMount'
import type { D2DLayoutInfo } from '../../types/staticBandwidth'
import { useLayoutStore } from '../../store/layoutStore'

interface MultiDieTopologyGraphProps {
  data: TopologyData | null
  mounts: IPMount[]
  loading?: boolean
  // D2D模式: {die_id: {link_key: bw}}
  linkBandwidth?: Record<string, Record<string, number>>
  // D2D跨Die链路带宽: {'{src_die}-{src_node}-{dst_die}-{dst_node}': bw}
  d2dLinkBandwidth?: Record<string, number>
  linkComposition?: Record<string, any[]>
  d2dLayout: D2DLayoutInfo
  onNodeClick?: (nodeId: number, dieId?: number) => void
  onLinkClick?: (linkKey: string, composition: any[]) => void
  onWidthChange?: (width: number) => void
}

// 节点间距
const NODE_SPACING = 150
// Die间隙（基础值，会根据布局调整）
const DIE_GAP_BASE = 100

// 计算旋转后的节点位置
const calculateRotatedPosition = (
  nodeId: number,
  cols: number,
  rows: number,
  rotation: number
): { col: number; row: number } => {
  const origRow = Math.floor(nodeId / cols)
  const origCol = nodeId % cols

  let newRow: number, newCol: number
  if (rotation === 0 || Math.abs(rotation) === 360) {
    newRow = origRow
    newCol = origCol
  } else if (Math.abs(rotation) === 90) {
    newRow = origCol
    newCol = rows - 1 - origRow
  } else if (Math.abs(rotation) === 180) {
    newRow = rows - 1 - origRow
    newCol = cols - 1 - origCol
  } else if (Math.abs(rotation) === 270) {
    newRow = cols - 1 - origCol
    newCol = origRow
  } else {
    newRow = origRow
    newCol = origCol
  }

  return { col: newCol, row: newRow }
}

// 获取旋转后的Die尺寸
const getRotatedDieSize = (
  cols: number,
  rows: number,
  rotation: number
): { width: number; height: number } => {
  if (rotation === 90 || rotation === 270) {
    return { width: rows * NODE_SPACING, height: cols * NODE_SPACING }
  }
  return { width: cols * NODE_SPACING, height: rows * NODE_SPACING }
}

const MultiDieTopologyGraph = forwardRef<{ saveLayout: () => void }, MultiDieTopologyGraphProps>(({
  data, mounts, loading, linkBandwidth, d2dLinkBandwidth, linkComposition, d2dLayout, onNodeClick, onLinkClick, onWidthChange
}, ref) => {
  const cyRef = useRef<Cytoscape.Core | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)

  // 从store获取保存的容器尺寸
  const { multiDieWidth, multiDieHeight, setMultiDieSize } = useLayoutStore()

  // 监听容器大小变化，自动调整画布并适应
  useEffect(() => {
    if (!containerRef.current) return
    const resizeObserver = new ResizeObserver(() => {
      if (cyRef.current) {
        cyRef.current.resize()
        setTimeout(() => {
          cyRef.current?.fit(undefined, 10)
          cyRef.current?.center()
        }, 50)
      }
      // 通知父组件宽度变化
      if (containerRef.current && onWidthChange) {
        onWidthChange(containerRef.current.offsetWidth)
      }
    })
    resizeObserver.observe(containerRef.current)
    return () => resizeObserver.disconnect()
  }, [onWidthChange])

  // 创建节点ID到挂载信息的映射
  const mountMap = useMemo(() => {
    const map = new Map<number, IPMount[]>()
    mounts.forEach(mount => {
      if (!map.has(mount.node_id)) {
        map.set(mount.node_id, [])
      }
      map.get(mount.node_id)!.push(mount)
    })
    return map
  }, [mounts])

  // IP类型到颜色
  const getIPTypeColor = (ipType: string): { bg: string; border: string } => {
    const type = ipType.split('_')[0].toLowerCase()
    switch (type) {
      case 'gdma':
      case 'sdma':
        return { bg: '#5B8FF9', border: '#3A6FD9' }
      case 'cdma':
        return { bg: '#5AD8A6', border: '#3AB886' }
      case 'npu':
        return { bg: '#722ed1', border: '#531dab' }
      case 'ddr':
      case 'l2m':
        return { bg: '#E8684A', border: '#C8482A' }
      case 'pcie':
        return { bg: '#fa8c16', border: '#d46b08' }
      case 'eth':
        return { bg: '#52c41a', border: '#389e0d' }
      case 'd2d':
        return { bg: '#13c2c2', border: '#08979c' }
      default:
        return { bg: '#eb2f96', border: '#c41d7f' }
    }
  }

  // 根据带宽获取颜色（0-256映射到浅红-深红）
  const getBandwidthColor = (bandwidth: number): string => {
    if (bandwidth === 0) return '#bfbfbf'  // 灰色 - 无流量
    const ratio = Math.min(bandwidth / 256, 1)  // 0-256 映射到 0-1
    // 从浅红(255,180,180)到深红(200,0,0)
    const r = Math.round(255 - ratio * 55)
    const g = Math.round(180 * (1 - ratio))
    const b = Math.round(180 * (1 - ratio))
    return `rgb(${r},${g},${b})`
  }

  // 根据带宽获取线宽
  const getBandwidthWidth = (bandwidth: number): number => {
    if (bandwidth === 0) return 2
    if (bandwidth < 50) return 2
    if (bandwidth < 100) return 3
    return 4
  }

  // 使用useMemo同步计算elements，确保d2dLayout变化时立即更新
  const elements = useMemo(() => {
    if (!data || !d2dLayout) return []

    const { die_positions, die_rotations, d2d_connections, num_dies } = d2dLayout
    const cols = data.cols
    const rows = data.rows

    // 判断连接类型
    const getConnType = (fromDiePos: [number, number], toDiePos: [number, number]): string => {
      const dx = Math.abs(fromDiePos[0] - toDiePos[0])
      const dy = Math.abs(fromDiePos[1] - toDiePos[1])
      if (dx === 0) return 'vertical'
      if (dy === 0) return 'horizontal'
      return 'diagonal'
    }

    // 在useEffect内部重新计算dieOffsets，确保使用最新的d2dLayout
    const computeDieOffsets = () => {
      const dieSizes: Record<string, { width: number; height: number }> = {}
      Object.keys(die_positions).forEach(dieId => {
        const rotation = die_rotations[dieId] || 0
        dieSizes[dieId] = getRotatedDieSize(cols, rows, rotation)
      })

      const maxGridX = Math.max(...Object.values(die_positions).map(pos => pos[0]))
      const maxGridY = Math.max(...Object.values(die_positions).map(pos => pos[1]))
      const DIE_GAP_X = maxGridX > 0 ? DIE_GAP_BASE : 0
      // 4个DIE时竖向间距缩小，2个DIE时使用默认间距
      const DIE_GAP_Y = maxGridY > 0 ? (num_dies >= 4 ? DIE_GAP_BASE - 280 : DIE_GAP_BASE) : 0

      const maxWidthPerCol: Record<number, number> = {}
      const maxHeightPerRow: Record<number, number> = {}
      Object.entries(die_positions).forEach(([dieId, pos]) => {
        const [gridX, gridY] = pos
        const { width, height } = dieSizes[dieId]
        maxWidthPerCol[gridX] = Math.max(maxWidthPerCol[gridX] || 0, width)
        maxHeightPerRow[gridY] = Math.max(maxHeightPerRow[gridY] || 0, height)
      })

      const offsets: Record<string, { x: number; y: number }> = {}
      Object.entries(die_positions).forEach(([dieId, pos]) => {
        const [gridX, gridY] = pos
        const flippedY = maxGridY - gridY
        let offsetX = 0
        let offsetY = 0
        for (let x = 0; x < gridX; x++) {
          offsetX += (maxWidthPerCol[x] || 0) + DIE_GAP_X
        }
        for (let y = 0; y < flippedY; y++) {
          offsetY += (maxHeightPerRow[maxGridY - y] || 0) + DIE_GAP_Y
        }
        offsets[dieId] = { x: offsetX, y: offsetY }
      })

      // 计算D2D连接对齐偏移
      const alignmentOffsets: Record<string, { x: number; y: number }> = {}
      Object.keys(die_positions).forEach(dieId => {
        alignmentOffsets[dieId] = { x: 0, y: 0 }
      })

      // 收集对齐约束
      const verticalConstraints: Record<string, { die0: number; die1: number; col0: number; col1: number; offset: number }> = {}
      const horizontalConstraints: Record<string, { die0: number; die1: number; row0: number; row1: number; offset: number }> = {}

      d2d_connections.forEach(([srcDie, srcNode, dstDie, dstNode]) => {
        const srcDiePos = die_positions[String(srcDie)]
        const dstDiePos = die_positions[String(dstDie)]
        if (!srcDiePos || !dstDiePos) return

        const connType = getConnType(srcDiePos, dstDiePos)
        const srcRotation = die_rotations[String(srcDie)] || 0
        const dstRotation = die_rotations[String(dstDie)] || 0

        const srcRotated = calculateRotatedPosition(srcNode, cols, rows, srcRotation)
        const dstRotated = calculateRotatedPosition(dstNode, cols, rows, dstRotation)

        const diePair = `${Math.min(srcDie, dstDie)}-${Math.max(srcDie, dstDie)}`

        if (connType === 'vertical') {
          const srcX = srcRotated.col * NODE_SPACING
          const dstX = dstRotated.col * NODE_SPACING
          const offsetNeeded = Math.abs(srcX - dstX)
          if (!verticalConstraints[diePair] || offsetNeeded > verticalConstraints[diePair].offset) {
            verticalConstraints[diePair] = { die0: srcDie, die1: dstDie, col0: srcRotated.col, col1: dstRotated.col, offset: offsetNeeded }
          }
        } else if (connType === 'horizontal') {
          const srcY = srcRotated.row * NODE_SPACING
          const dstY = dstRotated.row * NODE_SPACING
          const offsetNeeded = Math.abs(srcY - dstY)
          if (!horizontalConstraints[diePair] || offsetNeeded > horizontalConstraints[diePair].offset) {
            horizontalConstraints[diePair] = { die0: srcDie, die1: dstDie, row0: srcRotated.row, row1: dstRotated.row, offset: offsetNeeded }
          }
        }
      })

      // 应用垂直对齐（X方向）
      Object.values(verticalConstraints).forEach(({ die0, die1, col0, col1 }) => {
        const xDiff = (col0 - col1) * NODE_SPACING
        if (die0 === 0) {
          alignmentOffsets[String(die1)].x += xDiff
        } else if (die1 === 0) {
          alignmentOffsets[String(die0)].x -= xDiff
        } else {
          if (die0 > die1) alignmentOffsets[String(die0)].x -= xDiff
          else alignmentOffsets[String(die1)].x += xDiff
        }
      })

      // 应用水平对齐（Y方向）
      Object.values(horizontalConstraints).forEach(({ die0, die1, row0, row1 }) => {
        const yDiff = (row0 - row1) * NODE_SPACING
        if (die0 === 0) {
          alignmentOffsets[String(die1)].y += yDiff
        } else if (die1 === 0) {
          alignmentOffsets[String(die0)].y -= yDiff
        } else {
          if (die0 > die1) alignmentOffsets[String(die0)].y -= yDiff
          else alignmentOffsets[String(die1)].y += yDiff
        }
      })

      // 合并基础偏移和对齐偏移
      Object.keys(offsets).forEach(dieId => {
        offsets[dieId].x += alignmentOffsets[dieId]?.x || 0
        offsets[dieId].y += alignmentOffsets[dieId]?.y || 0
      })

      // 确保所有偏移量为非负值
      const minX = Math.min(...Object.values(offsets).map(o => o.x))
      const minY = Math.min(...Object.values(offsets).map(o => o.y))
      if (minX < 0 || minY < 0) {
        Object.keys(offsets).forEach(dieId => {
          if (minX < 0) offsets[dieId].x -= minX
          if (minY < 0) offsets[dieId].y -= minY
        })
      }
      return offsets
    }

    const dieOffsets = computeDieOffsets()

    const nodes: any[] = []
    const edges: any[] = []

    // 为每个Die绘制节点
    for (let dieIdNum = 0; dieIdNum < num_dies; dieIdNum++) {
      const dieId = String(dieIdNum)
      const rotation = die_rotations[dieId] ?? die_rotations[dieIdNum] ?? 0
      const offset = dieOffsets[dieId] ?? { x: 0, y: 0 }

      // 绘制Die标签
      const dieSize = getRotatedDieSize(cols, rows, rotation)
      const diePos = die_positions[dieId] || [0, 0]
      const maxGridX = Math.max(...Object.values(die_positions).map(pos => pos[0]))
      const isRightColumn = diePos[0] > 0 && diePos[0] === maxGridX

      nodes.push({
        data: {
          id: `die-label-${dieId}`,
          label: `Die ${dieId}`,
          dieId: dieIdNum
        },
        position: {
          x: isRightColumn
            ? offset.x + dieSize.width + NODE_SPACING / 2 + 0
            : offset.x + NODE_SPACING / 2 - 100,
          y: offset.y + dieSize.height / 2
        },
        classes: 'die-label'
      })

      // 绘制节点
      data.nodes.forEach(node => {
        const rotatedPos = calculateRotatedPosition(node.id, cols, rows, rotation)
        const nodeMounts = mountMap.get(node.id)
        const hasMounts = nodeMounts && nodeMounts.length > 0

        const x = rotatedPos.col * NODE_SPACING + offset.x + NODE_SPACING / 2
        const y = rotatedPos.row * NODE_SPACING + offset.y + NODE_SPACING / 2

        // 主节点（背景）
        nodes.push({
          data: {
            id: `die-${dieId}-node-${node.id}`,
            label: '',  // 不显示标签，由单独的文本层显示
            nodeId: node.id,
            dieId: dieIdNum,
            mounted: hasMounts
          },
          position: { x, y },
          classes: hasMounts ? 'mounted-node' : 'unmounted'
        })

        // IP块
        if (nodeMounts && nodeMounts.length > 0) {
          // 按IP类型分组
          const ipByType: Record<string, typeof nodeMounts> = {}
          nodeMounts.forEach(mount => {
            const baseType = mount.ip_type.split('_')[0].toLowerCase()
            if (!ipByType[baseType]) ipByType[baseType] = []
            ipByType[baseType].push(mount)
          })

          // 按优先级排序IP类型
          const priorityTypes = ['gdma', 'ddr', 'sdma', 'cdma', 'npu', 'pcie', 'eth', 'l2m', 'd2d']
          const sortedTypes = Object.keys(ipByType).sort((a, b) => {
            const aIdx = priorityTypes.indexOf(a)
            const bIdx = priorityTypes.indexOf(b)
            const aPriority = aIdx >= 0 ? aIdx : priorityTypes.length
            const bPriority = bIdx >= 0 ? bIdx : priorityTypes.length
            return aPriority - bPriority
          })

          // 计算布局参数
          const numRows = sortedTypes.length
          const maxColsInRow = Math.max(...sortedTypes.map(t => ipByType[t].length))

          // 动态计算IP块大小：根据行数和列数调整，确保不超出节点框(70x70)
          const nodeBoxSize = 70
          const minGap = 2  // 最小间隙
          // 根据最大维度计算合适的IP块大小
          const maxDim = Math.max(numRows, maxColsInRow)
          let ipSize: number
          if (maxDim <= 1) {
            ipSize = 32  // 单个IP时放大
          } else if (maxDim === 2) {
            ipSize = 26  // 2个时适中
          } else {
            // 3个及以上：动态计算以适应节点框
            ipSize = Math.floor((nodeBoxSize - minGap * (maxDim + 1)) / maxDim)
            ipSize = Math.max(ipSize, 16)  // 最小16px
          }

          const spacing = ipSize + minGap  // 间距

          // 计算总高度，用于垂直居中
          const totalHeight = numRows * spacing
          const startY = -totalHeight / 2+ spacing / 2

          // 绘制每行IP块
          sortedTypes.forEach((baseType, rowIdx) => {
            const rowMounts = ipByType[baseType]
            const numCols = rowMounts.length
            // 该行水平居中
            const totalWidth = numCols * spacing
            const startX = -totalWidth / 2 + spacing / 2

            rowMounts.forEach((mount, colIdx) => {
              const colors = getIPTypeColor(mount.ip_type)
              const parts = mount.ip_type.split('_')
              const typeAbbr = parts[0].charAt(0).toUpperCase()
              const number = parts[1] || '0'
              const shortLabel = `${typeAbbr}${number}`

              const ipOffsetX = startX + colIdx * spacing
              const ipOffsetY = startY + rowIdx * spacing

              nodes.push({
                data: {
                  id: `die-${dieId}-ip-${node.id}-${baseType}-${colIdx}`,
                  label: shortLabel,
                  ipType: mount.ip_type,
                  bgColor: colors.bg,
                  borderColor: colors.border,
                  ipSize: ipSize,
                  parentNodeId: node.id,
                  dieId: dieId
                },
                position: { x: x + ipOffsetX, y: y + ipOffsetY },
                classes: 'ip-block',
                locked: true,
                selectable: false
              })
            })
          })
        }

        // 添加节点编号文本层（最上层）
        nodes.push({
          data: {
            id: `die-${dieId}-label-${node.id}`,
            label: `${node.id}`,
            nodeId: node.id,
          },
          position: { x, y },
          classes: 'node-label',
          locked: true,
          grabbable: false,
          selectable: false
        })
      })

      // 绘制Die内部链路
      const dieBandwidth = linkBandwidth?.[dieId] || {}
      data.edges.forEach((edge, idx) => {
        // 计算原始坐标（用于查询后端带宽数据）
        const srcOrigRow = Math.floor(edge.source / cols)
        const srcOrigCol = edge.source % cols
        const dstOrigRow = Math.floor(edge.target / cols)
        const dstOrigCol = edge.target % cols

        // 计算旋转后的坐标（用于确定渲染方向）
        const srcRotated = calculateRotatedPosition(edge.source, cols, rows, rotation)
        const dstRotated = calculateRotatedPosition(edge.target, cols, rows, rotation)

        // 根据旋转后的视觉位置计算方向
        const actualDirection = srcRotated.row === dstRotated.row ? 'horizontal' : 'vertical'

        // 确定偏移符号：视觉上水平时向右的在下，视觉上垂直时向下的在左
        const isForwardPositive = actualDirection === 'horizontal'
          ? (dstRotated.col > srcRotated.col)
          : (dstRotated.row > srcRotated.row)

        // 正向 - 使用原始坐标查询带宽（包含die_id以区分不同Die的相同位置链路）
        const fwdKey = `${dieId}-${srcOrigCol},${srcOrigRow}-${dstOrigCol},${dstOrigRow}`
        const fwdBw = dieBandwidth[`${srcOrigCol},${srcOrigRow}-${dstOrigCol},${dstOrigRow}`] || 0
        edges.push({
          data: {
            id: `die-${dieId}-edge-${idx}-fwd`,
            source: `die-${dieId}-node-${edge.source}`,
            target: `die-${dieId}-node-${edge.target}`,
            direction: actualDirection,
            offset: isForwardPositive ? 1 : -1,
            bandwidth: fwdBw,
            bandwidthColor: getBandwidthColor(fwdBw),
            bandwidthWidth: getBandwidthWidth(fwdBw),
            linkKey: fwdKey,
            label: fwdBw > 0 ? fwdBw.toFixed(1) : ''
          },
          classes: 'internal-edge'
        })

        // 反向 - 使用原始坐标查询带宽（包含die_id以区分不同Die的相同位置链路）
        const bwdKey = `${dieId}-${dstOrigCol},${dstOrigRow}-${srcOrigCol},${srcOrigRow}`
        const bwdBw = dieBandwidth[`${dstOrigCol},${dstOrigRow}-${srcOrigCol},${srcOrigRow}`] || 0
        edges.push({
          data: {
            id: `die-${dieId}-edge-${idx}-bwd`,
            source: `die-${dieId}-node-${edge.target}`,
            target: `die-${dieId}-node-${edge.source}`,
            direction: actualDirection,
            offset: isForwardPositive ? -1 : 1,
            bandwidth: bwdBw,
            bandwidthColor: getBandwidthColor(bwdBw),
            bandwidthWidth: getBandwidthWidth(bwdBw),
            linkKey: bwdKey,
            label: bwdBw > 0 ? bwdBw.toFixed(1) : ''
          },
          classes: 'internal-edge'
        })
      })
    }

    // 绘制D2D跨Die连接线（双向）
    d2d_connections.forEach((conn, idx) => {
      const [srcDie, srcNode, dstDie, dstNode] = conn
      const srcDieStr = String(srcDie)
      const dstDieStr = String(dstDie)

      // 使用本地计算的dieOffsets
      const srcRotation = die_rotations[srcDieStr] || 0
      const dstRotation = die_rotations[dstDieStr] || 0
      const srcOffset = dieOffsets[srcDieStr] || { x: 0, y: 0 }
      const dstOffset = dieOffsets[dstDieStr] || { x: 0, y: 0 }

      const srcRotated = calculateRotatedPosition(srcNode, cols, rows, srcRotation)
      const dstRotated = calculateRotatedPosition(dstNode, cols, rows, dstRotation)

      // 计算实际屏幕位置
      const srcX = srcRotated.col * NODE_SPACING + srcOffset.x
      const srcY = srcRotated.row * NODE_SPACING + srcOffset.y
      const dstX = dstRotated.col * NODE_SPACING + dstOffset.x
      const dstY = dstRotated.row * NODE_SPACING + dstOffset.y

      // 根据Die网格位置判断方向（统一使用getConnType）
      const srcDiePos = die_positions[srcDieStr]
      const dstDiePos = die_positions[dstDieStr]
      const d2dDirection = getConnType(srcDiePos, dstDiePos)

      // 确定偏移符号：水平方向时向右的在下，垂直方向时向下的在左，对角线使用水平方向判断
      const isForwardPositive = d2dDirection === 'horizontal'
        ? (dstX > srcX)
        : d2dDirection === 'vertical'
          ? (dstY > srcY)
          : (dstX > srcX)  // 对角线使用水平方向判断

      // 生成linkKey用于点击事件（带 d2d- 前缀）
      const fwdLinkKey = `d2d-${srcDie}-${srcNode}-${dstDie}-${dstNode}`
      const bwdLinkKey = `d2d-${dstDie}-${dstNode}-${srcDie}-${srcNode}`

      // 生成用于查询带宽的 key（不带前缀，与后端返回格式一致）
      const fwdBandwidthKey = `${srcDie}-${srcNode}-${dstDie}-${dstNode}`
      const bwdBandwidthKey = `${dstDie}-${dstNode}-${srcDie}-${srcNode}`

      // 从后端获取D2D链路带宽
      const fwdBandwidth = d2dLinkBandwidth?.[fwdBandwidthKey] || 0
      const bwdBandwidth = d2dLinkBandwidth?.[bwdBandwidthKey] || 0

      // 正向
      edges.push({
        data: {
          id: `d2d-conn-${idx}-fwd`,
          source: `die-${srcDie}-node-${srcNode}`,
          target: `die-${dstDie}-node-${dstNode}`,
          d2dConnection: true,
          direction: d2dDirection,
          offset: isForwardPositive ? 1 : -1,
          bandwidth: fwdBandwidth,
          bandwidthColor: getBandwidthColor(fwdBandwidth),
          bandwidthWidth: getBandwidthWidth(fwdBandwidth),
          linkKey: fwdLinkKey,
          label: fwdBandwidth > 0 ? fwdBandwidth.toFixed(1) : ''
        },
        classes: 'd2d-edge'
      })
      // 反向
      edges.push({
        data: {
          id: `d2d-conn-${idx}-bwd`,
          source: `die-${dstDie}-node-${dstNode}`,
          target: `die-${srcDie}-node-${srcNode}`,
          d2dConnection: true,
          direction: d2dDirection,
          offset: isForwardPositive ? -1 : 1,
          bandwidth: bwdBandwidth,
          bandwidthColor: getBandwidthColor(bwdBandwidth),
          bandwidthWidth: getBandwidthWidth(bwdBandwidth),
          linkKey: bwdLinkKey,
          label: bwdBandwidth > 0 ? bwdBandwidth.toFixed(1) : ''
        },
        classes: 'd2d-edge'
      })
    })

    return [...nodes, ...edges]
  }, [data, d2dLayout, linkBandwidth, d2dLinkBandwidth, mountMap])

  const stylesheet: any[] = [
    // Die标签样式
    {
      selector: '.die-label',
      style: {
        'shape': 'rectangle',
        'width': 1,
        'height': 1,
        'background-opacity': 0,
        'border-width': 0,
        'label': 'data(label)',
        'text-valign': 'center',
        'text-halign': 'center',
        'font-size': 30,
        'font-weight': 'bold',
        'color': '#1890ff',
        'z-index': 100
      }
    },
    // 节点样式（背景）
    {
      selector: 'node.mounted-node, node.unmounted',
      style: {
        'shape': 'rectangle',
        'width': 70,
        'height': 70,
        'label': '',  // 不显示标签
        'background-color': '#d9d9d9',
        'border-width': 2,
        'border-color': '#bfbfbf',
        'z-index': 1
      }
    },
    // 节点编号文本层（最上层）
    {
      selector: '.node-label',
      style: {
        'shape': 'rectangle',
        'width': 1,
        'height': 1,
        'background-opacity': 0,
        'border-width': 0,
        'label': 'data(label)',
        'text-valign': 'center',
        'text-halign': 'center',
        'color': '#000000',
        'font-size': '23px',
        'font-weight': 'bold',
        'text-outline-width': 2,
        'text-outline-color': '#ffffff',
        'z-index': 100
      }
    },
    // IP块样式（动态大小）
    {
      selector: '.ip-block',
      style: {
        'shape': 'rectangle',
        'width': (ele: any) => ele.data('ipSize') || 24,
        'height': (ele: any) => ele.data('ipSize') || 24,
        'label': 'data(label)',
        'text-valign': 'center',
        'text-halign': 'center',
        'background-color': 'data(bgColor)',
        'border-width': 1,
        'border-color': 'data(borderColor)',
        'color': '#fff',
        'font-size': (ele: any) => {
          const size = ele.data('ipSize') || 24
          return size >= 28 ? '12px' : '10px'
        },
        'font-weight': 'bold',
        'z-index': 15
      }
    },
    // 普通边样式（Die内部链路）
    {
      selector: '.internal-edge',
      style: {
        'width': 'data(bandwidthWidth)',
        'line-color': 'data(bandwidthColor)',
        'target-arrow-color': 'data(bandwidthColor)',
        'target-arrow-shape': 'triangle',
        'curve-style': 'straight',
        'source-distance-from-node': 36,
        'target-distance-from-node': 36,
        'source-endpoint': (ele: any) => {
          const offset = ele.data('offset') || 0
          const direction = ele.data('direction')
          if (direction === 'horizontal') {
            return offset > 0 ? '0 8' : '0 -8'
          } else {
            return offset > 0 ? '-8 0' : '8 0'
          }
        },
        'target-endpoint': (ele: any) => {
          const offset = ele.data('offset') || 0
          const direction = ele.data('direction')
          if (direction === 'horizontal') {
            return offset > 0 ? '0 8' : '0 -8'
          } else {
            return offset > 0 ? '-8 0' : '8 0'
          }
        },
        'arrow-scale': 1.0,
        'opacity': 0.8,
        'z-index': 5,
        // 带宽标签样式
        'label': 'data(label)',
        'font-size': 16,
        'color': '#d32029',
        'text-background-color': '#fff',
        'text-background-opacity': 0.8,
        'text-background-padding': '2px',
        'text-margin-x': (ele: any) => {
          const offset = ele.data('offset') || 0
          const direction = ele.data('direction')
          if (direction === 'vertical') {
            const label = ele.data('label') || ''
            const textLength = label.length
            const textWidth = textLength * 7.2
            const baseOffset = 8
            const dynamicOffset = baseOffset + textWidth / 2
            return offset > 0 ? -dynamicOffset : dynamicOffset
          }
          return 0
        },
        'text-margin-y': (ele: any) => {
          const offset = ele.data('offset') || 0
          const direction = ele.data('direction')
          if (direction === 'horizontal') {
            return offset > 0 ? 12 : -12
          }
          return 0
        }
      }
    },
    // D2D跨Die连接样式（双向粗虚线箭头）
    {
      selector: '.d2d-edge',
      style: {
        'width': 'data(bandwidthWidth)',
        'line-color': 'data(bandwidthColor)',
        'target-arrow-color': 'data(bandwidthColor)',
        'target-arrow-shape': 'triangle',
        'curve-style': 'straight',
        'line-style': 'dashed',
        'line-dash-pattern': [10, 5],
        'source-distance-from-node': 36,
        'target-distance-from-node': 36,
        'source-endpoint': (ele: any) => {
          const offset = ele.data('offset') || 0
          const direction = ele.data('direction')
          if (direction === 'diagonal') {
            // 对角线：使用统一方向计算垂直向量（避免正反向重叠）
            const source = ele.source().position()
            const target = ele.target().position()
            let dx = target.x - source.x
            let dy = target.y - source.y
            // 统一使用从左到右的方向作为参考
            if (dx < 0) {
              dx = -dx
              dy = -dy
            }
            const length = Math.sqrt(dx * dx + dy * dy)
            const perpX = (-dy / length) * 10
            const perpY = (dx / length) * 10
            return offset > 0 ? `${perpX} ${perpY}` : `${-perpX} ${-perpY}`
          } else if (direction === 'horizontal') {
            return offset > 0 ? '0 8' : '0 -8'
          } else {
            return offset > 0 ? '-8 0' : '8 0'
          }
        },
        'target-endpoint': (ele: any) => {
          const offset = ele.data('offset') || 0
          const direction = ele.data('direction')
          if (direction === 'diagonal') {
            // 对角线：使用统一方向计算垂直向量（避免正反向重叠）
            const source = ele.source().position()
            const target = ele.target().position()
            let dx = target.x - source.x
            let dy = target.y - source.y
            // 统一使用从左到右的方向作为参考
            if (dx < 0) {
              dx = -dx
              dy = -dy
            }
            const length = Math.sqrt(dx * dx + dy * dy)
            const perpX = (-dy / length) * 10
            const perpY = (dx / length) * 10
            return offset > 0 ? `${perpX} ${perpY}` : `${-perpX} ${-perpY}`
          } else if (direction === 'horizontal') {
            return offset > 0 ? '0 8' : '0 -8'
          } else {
            return offset > 0 ? '-8 0' : '8 0'
          }
        },
        'arrow-scale': 1.9,
        'opacity': 0.9,
        'z-index': 20,
        // 带宽标签样式
        'label': 'data(label)',
        'font-size': 20,
        'color': '#d32029',
        'text-background-color': '#fff',
        'text-background-opacity': 0.8,
        'text-background-padding': '2px',
        // 标签自动跟随边的方向旋转
        'text-rotation': 'autorotate',
        'text-margin-x': (ele: any) => {
          const offset = ele.data('offset') || 0
          const direction = ele.data('direction')

          if (direction === 'diagonal') {
            // 对角线：
            const source = ele.source().position()
            const target = ele.target().position()
            const dx = target.x - source.x
            const dy = target.y - source.y
            const length = Math.sqrt(dx * dx + dy * dy)
            // 标签总是向target方向（箭头头部）移动1/3
            const shiftAlongEdge = length / 3
            const shiftX = (dx / length) * shiftAlongEdge
            // 垂直于边的偏移：向右的箭头标签在下，向左的箭头标签在上
            const perpShift = (dx > 0 ? offset : -offset) * 15
            const perpX = -(dy / length) * perpShift
            return shiftX + perpX
          } else if (direction === 'vertical') {
            return offset > 0 ? -16 : 16
          }
          return 0
        },
        'text-margin-y': (ele: any) => {
          const offset = ele.data('offset') || 0
          const direction = ele.data('direction')

          if (direction === 'diagonal') {
            // 对角线：
            const source = ele.source().position()
            const target = ele.target().position()
            const dx = target.x - source.x
            const dy = target.y - source.y
            const length = Math.sqrt(dx * dx + dy * dy)
            // 标签总是向target方向（箭头头部）移动1/3
            const shiftAlongEdge = length / 3
            const shiftY = (dy / length) * shiftAlongEdge
            // 垂直于边的偏移：向右的箭头标签在下，向左的箭头标签在上
            const perpShift = (dx > 0 ? offset : -offset) * 15
            const perpY = (dx / length) * perpShift
            return shiftY + perpY
          } else if (direction === 'horizontal') {
            return offset > 0 ? 14 : -14
          }
          return 0
        }
      }
    }
  ]

  const handleCyInit = (cy: Cytoscape.Core) => {
    cyRef.current = cy

    setTimeout(() => {
      cy.fit(undefined, 20)
      cy.center()
    }, 100)

    // 节点点击事件
    cy.on('tap', 'node.mounted-node, node.unmounted', (evt) => {
      const nodeId = evt.target.data('nodeId')
      const dieId = evt.target.data('dieId')
      if (nodeId !== undefined && onNodeClick) {
        onNodeClick(nodeId, dieId)
      }
    })

    // IP块点击事件
    cy.on('tap', '.ip-block', (evt) => {
      const parentNodeId = evt.target.data('parentNodeId')
      const dieId = evt.target.data('dieId')
      if (parentNodeId !== undefined && onNodeClick) {
        onNodeClick(parentNodeId, dieId)
      }
    })

    // Die内部链路点击事件
    cy.on('tap', 'edge.internal-edge', (evt) => {
      const linkKey = evt.target.data('linkKey')
      if (linkKey && onLinkClick) {
        const composition = linkComposition?.[linkKey] || []
        onLinkClick(linkKey, composition)
      }
    })

    // D2D跨Die链路点击事件
    cy.on('tap', 'edge.d2d-edge', (evt) => {
      const linkKey = evt.target.data('linkKey')
      if (linkKey && onLinkClick) {
        // 去除 "d2d-" 前缀用于查询 linkComposition
        const queryKey = linkKey.replace(/^d2d-/, '')
        const composition = linkComposition?.[queryKey] || []
        onLinkClick(linkKey, composition)  // 传递完整linkKey用于显示
      }
    })

    // 空白点击清除选择
    cy.on('tap', (evt) => {
      if (evt.target === cy && onLinkClick) {
        onLinkClick('', [])
      }
    })
  }

  const handleZoomIn = () => cyRef.current?.zoom(cyRef.current.zoom() * 1.2)
  const handleZoomOut = () => cyRef.current?.zoom(cyRef.current.zoom() / 1.2)
  const handleFit = () => {
    cyRef.current?.fit(undefined, 20)
    cyRef.current?.center()
  }
  const handleSaveLayout = () => {
    if (containerRef.current) {
      const width = containerRef.current.offsetWidth
      const height = containerRef.current.offsetHeight
      setMultiDieSize(width, height)
      message.success('布局已保存')
    }
  }

  // 暴露saveLayout方法给父组件
  useImperativeHandle(ref, () => ({
    saveLayout: handleSaveLayout
  }))

  if (!data) {
    return (
      <Card>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 'calc(100vh - 250px)', minHeight: 750 }}>
          <span style={{ color: '#8c8c8c' }}>请选择拓扑类型</span>
        </div>
      </Card>
    )
  }

  return (
    <Card style={{ width: 'fit-content' }}>
      <Row style={{ marginBottom: 16 }} align="middle">
        <Col span={12}>
          <Space>
            <span>D2D拓扑</span>
            <Tag color="blue">{d2dLayout.num_dies} Dies</Tag>
            <Tag color="cyan">{d2dLayout.d2d_connections.length} D2D连接</Tag>
          </Space>
        </Col>
        <Col span={12} style={{ textAlign: 'right' }}>
          <Space>
            <Button size="small" icon={<ZoomInOutlined />} onClick={handleZoomIn}>放大</Button>
            <Button size="small" icon={<ZoomOutOutlined />} onClick={handleZoomOut}>缩小</Button>
            <Button size="small" icon={<AimOutlined />} onClick={handleFit}>适应</Button>
          </Space>
        </Col>
      </Row>
      <div
        ref={containerRef}
        style={{
          width: multiDieWidth,
          height: multiDieHeight,
          minWidth: '30vw',
          maxWidth: '60vw',
          border: '1px solid #d9d9d9',
          borderRadius: 4,
          background: '#fafafa',
          resize: 'both',
          overflow: 'hidden'
        }}
      >
        <CytoscapeComponent
          key={`cy-${JSON.stringify(d2dLayout)}-${mounts.length}-${JSON.stringify(mounts.map(m => m.node_id + m.ip_type))}`}
          elements={elements}
          style={{ width: '100%', height: '100%' }}
          stylesheet={stylesheet}
          cy={handleCyInit}
          layout={{ name: 'preset' }}
          userZoomingEnabled={true}
          userPanningEnabled={false}
          autoungrabify={true}
          boxSelectionEnabled={false}
        />
      </div>

      {/* 图例 */}
      <Card title="图例" size="small" style={{ marginTop: 16 }}>
        <Row gutter={[16, 8]} justify="center">
          {(() => {
            const ipTypeColors: Record<string, { bg: string; border: string; label: string }> = {
              'gdma': { bg: '#5B8FF9', border: '#3A6FD9', label: 'GDMA' },
              'sdma': { bg: '#5B8FF9', border: '#3A6FD9', label: 'SDMA' },
              'cdma': { bg: '#5AD8A6', border: '#3AB886', label: 'CDMA' },
              'npu': { bg: '#722ed1', border: '#531dab', label: 'NPU' },
              'ddr': { bg: '#E8684A', border: '#C8482A', label: 'DDR' },
              'l2m': { bg: '#E8684A', border: '#C8482A', label: 'L2M' },
              'pcie': { bg: '#fa8c16', border: '#d46b08', label: 'PCIe' },
              'eth': { bg: '#52c41a', border: '#389e0d', label: 'ETH' },
              'd2d': { bg: '#13c2c2', border: '#08979c', label: 'D2D' },
              'other': { bg: '#eb2f96', border: '#c41d7f', label: '其他' }
            }

            // 统计已挂载的IP类型
            const mountedTypes = new Set<string>()
            mounts.forEach(mount => {
              const type = mount.ip_type.split('_')[0].toLowerCase()
              if (ipTypeColors[type]) {
                mountedTypes.add(type)
              } else {
                mountedTypes.add('other')
              }
            })

            // 生成图例项
            return Array.from(mountedTypes).map(type => {
              const config = ipTypeColors[type]
              return (
                <Col span={4} key={type}>
                  <Space>
                    <div style={{
                      width: 24,
                      height: 24,
                      backgroundColor: config.bg,
                      border: `2px solid ${config.border}`,
                      borderRadius: 4
                    }}></div>
                    <span>{config.label}</span>
                  </Space>
                </Col>
              )
            })
          })()}
        </Row>
      </Card>
    </Card>
  )
})

MultiDieTopologyGraph.displayName = 'MultiDieTopologyGraph'

export default MultiDieTopologyGraph
