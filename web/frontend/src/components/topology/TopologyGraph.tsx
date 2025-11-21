import { useEffect, useRef, useState } from 'react'
import CytoscapeComponent from 'react-cytoscapejs'
import Cytoscape from 'cytoscape'
import { Card, Space, Tag, Button, Row, Col, Select } from 'antd'
import { ZoomInOutlined, ZoomOutOutlined, AimOutlined, ReloadOutlined } from '@ant-design/icons'
import type { TopologyData } from '../../types/topology'
import type { IPMount } from '../../types/ipMount'
import type { FlowInfo } from '../../types/staticBandwidth'

interface TopologyGraphProps {
  data: TopologyData | null
  mounts: IPMount[]
  loading?: boolean
  onNodeClick?: (nodeId: number) => void
  // NoC模式: Record<string, number>
  // D2D模式: Record<string, Record<string, number>>
  linkBandwidth?: Record<string, number> | Record<string, Record<string, number>>
  linkComposition?: Record<string, FlowInfo[]>
  bandwidthMode?: 'noc' | 'd2d'
  selectedDie?: number
  onDieChange?: (dieId: number) => void
  onLinkClick?: (linkKey: string, composition: FlowInfo[]) => void
  onWidthChange?: (width: number) => void
}

const TopologyGraph: React.FC<TopologyGraphProps> = ({
  data, mounts, loading, onNodeClick, linkBandwidth, linkComposition,
  bandwidthMode = 'noc', selectedDie = 0, onDieChange, onLinkClick, onWidthChange
}) => {
  const [elements, setElements] = useState<any[]>([])
  const [selectedNode, setSelectedNode] = useState<number | null>(null)
  const cyRef = useRef<Cytoscape.Core | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)

  // 监听容器大小变化，自动重新fit
  useEffect(() => {
    if (!containerRef.current) return
    const resizeObserver = new ResizeObserver(() => {
      if (cyRef.current) {
        cyRef.current.resize()
        setTimeout(() => {
          cyRef.current?.fit(undefined, 50)
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

  // 创建节点ID到挂载信息列表的映射（一个节点可以有多个IP）
  const mountMap = new Map<number, IPMount[]>()
  mounts.forEach(mount => {
    if (!mountMap.has(mount.node_id)) {
      mountMap.set(mount.node_id, [])
    }
    mountMap.get(mount.node_id)!.push(mount)
  })

  // IP类型到颜色的映射
  const getIPTypeColor = (ipType: string): { bg: string; border: string } => {
    const type = ipType.split('_')[0].toLowerCase()
    switch (type) {
      case 'gdma':
        return { bg: '#5B8FF9', border: '#3A6FD9' }  // 蓝色
      case 'sdma':
        return { bg: '#5B8FF9', border: '#3A6FD9' }  // 蓝色
      case 'cdma':
        return { bg: '#5AD8A6', border: '#3AB886' }  // 绿色
      case 'npu':
        return { bg: '#722ed1', border: '#531dab' }  // 紫色
      case 'ddr':
        return { bg: '#E8684A', border: '#C8482A' }  // 红色
      case 'l2m':
        return { bg: '#E8684A', border: '#C8482A' }  // 红色
      case 'pcie':
        return { bg: '#fa8c16', border: '#d46b08' }  // 橙色
      case 'eth':
        return { bg: '#52c41a', border: '#389e0d' }  // 绿色
      default:
        return { bg: '#eb2f96', border: '#c41d7f' }  // 粉红色
    }
  }

  // 获取节点的主要CSS类（基于第一个IP）
  const getIPTypeClass = (ipMounts: IPMount[] | undefined): string => {
    if (!ipMounts || ipMounts.length === 0) return 'unmounted'
    const ipType = ipMounts[0].ip_type
    const type = ipType.split('_')[0].toLowerCase()
    switch (type) {
      case 'gdma':
        return 'ip-gdma'
      case 'npu':
        return 'ip-npu'
      case 'ddr':
        return 'ip-ddr'
      case 'pcie':
        return 'ip-pcie'
      case 'eth':
        return 'ip-eth'
      default:
        return 'ip-other'
    }
  }

  // 节点ID转坐标：CrossRing格式为 row*cols + col
  const nodeIdToPos = (nodeId: number, cols: number): {col: number, row: number} => {
    return {
      col: nodeId % cols,
      row: Math.floor(nodeId / cols)
    }
  }

  // 获取链路带宽
  const getLinkBandwidth = (sourceId: number, targetId: number): number => {
    if (!linkBandwidth || !data) return 0
    const cols = data.cols
    const srcPos = nodeIdToPos(sourceId, cols)
    const dstPos = nodeIdToPos(targetId, cols)
    const key = `${srcPos.col},${srcPos.row}-${dstPos.col},${dstPos.row}`

    if (bandwidthMode === 'd2d') {
      // D2D模式：从选中的Die获取带宽
      const dieBandwidth = (linkBandwidth as Record<string, Record<string, number>>)[String(selectedDie)]
      return dieBandwidth ? (dieBandwidth[key] || 0) : 0
    } else {
      // NoC模式：直接获取带宽
      return (linkBandwidth as Record<string, number>)[key] || 0
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

  useEffect(() => {
    if (!data) return

    // 转换为Cytoscape格式，使用独立节点显示IP块（不使用复合节点）
    const nodes: any[] = []

    data.nodes.forEach(node => {
      const nodeMounts = mountMap.get(node.id)
      const hasMounts = nodeMounts && nodeMounts.length > 0

      // 背景节点（方框）
      nodes.push({
        data: {
          id: `node-${node.id}`,
          label: `${node.id}`,  // 始终显示节点ID
          nodeId: node.id,
          row: node.row,
          col: node.col,
          mounted: hasMounts,
        },
        position: {
          x: node.col * 150 + 75,
          y: node.row * 150 + 75,
        },
        classes: hasMounts ? 'mounted-node' : 'unmounted',
        style: {
          'z-index': 1,
          'text-events': 'yes',
        }
      })

      // 为每个IP创建独立节点（IP块）- z-index较高
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

        // 动态计算IP块大小：根据行数和列数调整
        const baseSize = 24
        const maxSize = 28
        const minSize = 18
        let ipSize = baseSize
        if (numRows <= 2 && maxColsInRow <= 2) {
          ipSize = maxSize  // IP少时放大
        } else if (numRows >= 4 || maxColsInRow >= 4) {
          ipSize = minSize  // IP多时缩小
        }

        const spacing = ipSize + 4  // 间距

        // 计算总高度，用于垂直居中
        const totalHeight = numRows * spacing
        const startY = -totalHeight / 2 + spacing / 2

        // 计算IP块位置基准
        const baseX = node.col * 150 + 75
        const baseY = node.row * 150 + 75

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

            const offsetX = startX + colIdx * spacing
            const offsetY = startY + rowIdx * spacing

            nodes.push({
              data: {
                id: `ip-${node.id}-${baseType}-${colIdx}`,
                label: shortLabel,
                ipType: mount.ip_type,
                parentNodeId: node.id,
                bgColor: colors.bg,
                borderColor: colors.border,
                ipSize: ipSize
              },
              position: {
                x: baseX + offsetX,
                y: baseY + offsetY,
              },
              classes: 'ip-block',
              locked: true,
              selectable: false,
              style: {
                'z-index': 10
              }
            })
          })
        })

        // 添加节点编号文本层（最上层）
        nodes.push({
          data: {
            id: `label-${node.id}`,
            label: `${node.id}`,
            parentNodeId: node.id,
          },
          position: {
            x: node.col * 150 + 75,
            y: node.row * 150 + 75,
          },
          classes: 'node-label',
          locked: true,
          grabbable: false,
          selectable: false,
          style: {
            'z-index': 100
          }
        })
      }
    })

    // 为每条边创建两条分开的线（双向）
    const edges: any[] = []
    data.edges.forEach((edge, idx) => {
      const cols = data.cols
      // 正向边 - 偏移到一侧
      const srcPos = nodeIdToPos(edge.source, cols)
      const dstPos = nodeIdToPos(edge.target, cols)
      const forwardLinkKey = `${srcPos.col},${srcPos.row}-${dstPos.col},${dstPos.row}`
      const forwardBandwidth = getLinkBandwidth(edge.source, edge.target)
      edges.push({
        data: {
          id: `edge-${idx}-forward`,
          source: `node-${edge.source}`,
          target: `node-${edge.target}`,
          direction: edge.direction,
          type: edge.type,
          offset: 1,  // 正向偏移
          bandwidth: forwardBandwidth,
          bandwidthColor: getBandwidthColor(forwardBandwidth),
          bandwidthWidth: getBandwidthWidth(forwardBandwidth),
          label: forwardBandwidth > 0 ? forwardBandwidth.toFixed(1) : '',
          linkKey: forwardLinkKey
        }
      })
      // 反向边 - 偏移到另一侧
      const backwardLinkKey = `${dstPos.col},${dstPos.row}-${srcPos.col},${srcPos.row}`
      const backwardBandwidth = getLinkBandwidth(edge.target, edge.source)
      edges.push({
        data: {
          id: `edge-${idx}-backward`,
          source: `node-${edge.target}`,
          target: `node-${edge.source}`,
          direction: edge.direction,
          type: edge.type,
          offset: -1,  // 反向偏移
          bandwidth: backwardBandwidth,
          bandwidthColor: getBandwidthColor(backwardBandwidth),
          bandwidthWidth: getBandwidthWidth(backwardBandwidth),
          label: backwardBandwidth > 0 ? backwardBandwidth.toFixed(1) : '',
          linkKey: backwardLinkKey
        }
      })
    })

    setElements([...nodes, ...edges])
  }, [data, mounts, linkBandwidth, bandwidthMode, selectedDie])

  const stylesheet: any[] = [
    // 背景节点样式 - 固定大小的正方形容器(统一浅灰色)
    {
      selector: 'node.mounted-node, node.unmounted',
      style: {
        'shape': 'rectangle',
        'width': 70,
        'height': 70,
        'label': 'data(label)',
        'text-valign': 'center',
        'text-halign': 'center',
        'background-color': '#d9d9d9',
        'border-width': 2,
        'border-color': '#bfbfbf',
        'color': '#000000',
        'font-size': '14px',
        'font-weight': 'bold',
        'text-outline-width': 2,
        'text-outline-color': '#ffffff',
        'z-index': 20,  // 文字层级最高
      }
    },
    // IP块样式 - 独立节点（动态大小）
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
        'text-outline-width': 0,
        'z-index': 10,
        'events': 'yes',
      }
    },
    // 节点编号文本层 - 最上层显示
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
        'font-size': '14px',
        'font-weight': 'bold',
        'text-outline-width': 2,
        'text-outline-color': '#ffffff',
        'z-index': 100,
        'events': 'no',  // 不响应事件
      }
    },
    // 选中节点（仅背景节点）
    {
      selector: 'node.mounted-node:selected, node.unmounted:selected',
      style: {
        'background-color': '#ff4d4f',
        'border-width': 4,
        'border-color': '#ff7875',
        'text-outline-color': '#ff4d4f',
        'width': 80,
        'height': 80,
      }
    },
    // 高亮节点（点击后）
    {
      selector: 'node.highlighted',
      style: {
        'border-width': 4,
        'border-color': '#ffd700',
      }
    },
    // 暗化节点
    {
      selector: 'node.dimmed',
      style: {
        'opacity': 0.3,
      }
    },
    // 基础边样式 - 使用endpoint实现平行分开的直线箭头
    {
      selector: 'edge',
      style: {
        'width': 'data(bandwidthWidth)',
        'line-color': 'data(bandwidthColor)',
        'target-arrow-color': 'data(bandwidthColor)',
        'target-arrow-shape': 'triangle',
        'target-arrow-fill': 'filled',
        'curve-style': 'straight',
        'edge-distances': 'node-position',
        'source-distance-from-node': 36,
        'target-distance-from-node': 36,
        'label': 'data(label)',
        'font-size': 12,
        'color': '#d32029',
        'text-background-color': '#fff',
        'text-background-opacity': 0.8,
        'text-background-padding': '2px',
        'text-margin-x': (ele: any) => {
          const offset = ele.data('offset') || 0
          const direction = ele.data('direction')
          if (direction === 'vertical') {
            // 垂直链路：根据文本长度动态左右偏移
            const label = ele.data('label') || ''
            const textLength = label.length
            const textWidth = textLength * 7.2  // 估算文本宽度
            const baseOffset = 4
            const dynamicOffset = baseOffset + textWidth / 2
            return offset > 0 ? -dynamicOffset : dynamicOffset
          }
          return 0
        },
        'text-margin-y': (ele: any) => {
          const offset = ele.data('offset') || 0
          const direction = ele.data('direction')
          if (direction === 'horizontal') {
            // 水平链路：固定上下偏移
            return offset > 0 ? -10 : 10
          }
          return 0
        },
        'source-endpoint': (ele: any) => {
          const offset = ele.data('offset') || 0
          const direction = ele.data('direction')
          if (direction === 'horizontal') {
            // 水平连接：上下偏移
            return offset > 0 ? '0 -8' : '0 8'
          } else {
            // 垂直连接：左右偏移
            return offset > 0 ? '-8 0' : '8 0'
          }
        },
        'target-endpoint': (ele: any) => {
          const offset = ele.data('offset') || 0
          const direction = ele.data('direction')
          if (direction === 'horizontal') {
            return offset > 0 ? '0 -8' : '0 8'
          } else {
            return offset > 0 ? '-8 0' : '8 0'
          }
        },
        'arrow-scale': 1.2,
        'opacity': 0.8,
      }
    },
    // 高亮边
    {
      selector: 'edge.highlighted',
      style: {
        'width': 3,
        'opacity': 1,
        'line-color': '#faad14',
        'target-arrow-color': '#faad14',
      }
    },
    // 暗化边
    {
      selector: 'edge.dimmed',
      style: {
        'opacity': 0.15,
      }
    },
  ]

  const handleCyInit = (cy: Cytoscape.Core) => {
    cyRef.current = cy

    // 初始化后自动居中显示
    setTimeout(() => {
      cy.fit(undefined, 50)
      cy.center()
    }, 100)

    // 添加背景节点点击事件
    cy.on('tap', 'node.mounted-node, node.unmounted', (evt) => {
      const node = evt.target
      const nodeId = node.data('nodeId')

      setSelectedNode(nodeId)

      if (onNodeClick) {
        onNodeClick(nodeId)
      }
    })

    // 点击IP块时，也触发对应节点的点击事件
    cy.on('tap', '.ip-block', (evt) => {
      const ipNode = evt.target
      const parentNodeId = ipNode.data('parentNodeId')

      setSelectedNode(parentNodeId)

      if (onNodeClick) {
        onNodeClick(parentNodeId)
      }
    })

    // 点击边时显示带宽组成
    cy.on('tap', 'edge', (evt) => {
      const edge = evt.target
      const linkKey = edge.data('linkKey')
      if (onLinkClick) {
        if (linkKey && linkComposition && linkComposition[linkKey]) {
          onLinkClick(linkKey, linkComposition[linkKey])
        } else {
          onLinkClick('', [])
        }
      }
    })

    // 点击空白处
    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        setSelectedNode(null)
        if (onLinkClick) {
          onLinkClick('', [])
        }
      }
    })
  }

  const handleZoomIn = () => {
    if (cyRef.current) {
      cyRef.current.zoom(cyRef.current.zoom() * 1.2)
    }
  }

  const handleZoomOut = () => {
    if (cyRef.current) {
      cyRef.current.zoom(cyRef.current.zoom() / 1.2)
    }
  }

  const handleFit = () => {
    if (cyRef.current) {
      cyRef.current.fit(undefined, 50)
      cyRef.current.center()
    }
  }

  const handleReset = () => {
    if (cyRef.current && data) {
      setSelectedNode(null)

      // 重置节点位置
      data.nodes.forEach(node => {
        const cyNode = cyRef.current!.getElementById(`node-${node.id}`)
        cyNode.position({
          x: node.col * 150 + 75,
          y: node.row * 150 + 75,
        })
      })

      // 重置缩放和位置并居中
      cyRef.current.fit(undefined, 50)
      cyRef.current.center()
    }
  }

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
    <>
      <Card style={{ width: 'fit-content', minWidth: 500 }}>
        <Row style={{ marginBottom: 16 }} align="middle">
          <Col span={12}>
            <Space>
              <span>拓扑: {data.type}</span>
              <Tag color="blue">{data.total_nodes} 节点</Tag>
              <Tag color="green">{data.metadata.total_links} 链路</Tag>
              <Tag color="orange">{mounts.length} 已挂载</Tag>
            </Space>
          </Col>
          <Col span={12} style={{ textAlign: 'right' }}>
            <Space>
              {bandwidthMode === 'd2d' && linkBandwidth && (
                <Space>
                  <span>Die:</span>
                  <Select
                    value={selectedDie}
                    onChange={onDieChange}
                    size="small"
                    style={{ width: 80 }}
                  >
                    {Object.keys(linkBandwidth).map(dieId => (
                      <Select.Option key={dieId} value={Number(dieId)}>
                        Die {dieId}
                      </Select.Option>
                    ))}
                  </Select>
                </Space>
              )}
              <Button size="small" icon={<ZoomInOutlined />} onClick={handleZoomIn}>放大</Button>
              <Button size="small" icon={<ZoomOutOutlined />} onClick={handleZoomOut}>缩小</Button>
              <Button size="small" icon={<AimOutlined />} onClick={handleFit}>适应</Button>
              <Button size="small" icon={<ReloadOutlined />} onClick={handleReset}>重置</Button>
            </Space>
          </Col>
        </Row>
        <div ref={containerRef} style={{ height: 'calc(100vh - 200px)', minHeight: 400, maxWidth: 'calc(100vw - 900px)', border: '1px solid #d9d9d9', borderRadius: 4, background: '#fafafa', resize: 'both', overflow: 'hidden' }}>
          <CytoscapeComponent
            elements={elements}
            style={{ width: '100%', height: '100%' }}
            stylesheet={stylesheet}
            cy={handleCyInit}
            layout={{
              name: 'preset',
            }}
            userZoomingEnabled={true}
            userPanningEnabled={false}
            autoungrabify={true}
            boxSelectionEnabled={false}
            autolock={false}
            autounselectify={false}
          />
        </div>
      </Card>

      {/* 图例 */}
      <Card title="图例" size="small">
        <Row gutter={[16, 8]} justify="center">
          {/* 动态显示已挂载的IP类型 */}
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
                      width: 30,
                      height: 30,
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

    </>
  )
}

export default TopologyGraph
