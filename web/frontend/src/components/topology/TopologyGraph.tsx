import { useEffect, useRef, useState } from 'react'
import CytoscapeComponent from 'react-cytoscapejs'
import Cytoscape from 'cytoscape'
import { Card, Space, Tag, Button, Row, Col } from 'antd'
import { ZoomInOutlined, ZoomOutOutlined, AimOutlined, ReloadOutlined } from '@ant-design/icons'
import type { TopologyData } from '../../types/topology'
import type { IPMount } from '../../types/ipMount'

interface TopologyGraphProps {
  data: TopologyData | null
  mounts: IPMount[]
  loading?: boolean
  onNodeClick?: (nodeId: number) => void
}

const TopologyGraph: React.FC<TopologyGraphProps> = ({ data, mounts, loading, onNodeClick }) => {
  const [elements, setElements] = useState<any[]>([])
  const [selectedNode, setSelectedNode] = useState<number | null>(null)
  const cyRef = useRef<Cytoscape.Core | null>(null)

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
        // IP类型分类：RN类型在第一排，SN类型在第二排
        const rnTypes = ['gdma', 'sdma', 'cdma', 'npu', 'pcie', 'eth']
        const snTypes = ['ddr', 'l2m']

        // 对IP进行分类和排序
        const sortedMounts = [...nodeMounts].sort((a, b) => {
          const aType = a.ip_type.split('_')[0].toLowerCase()
          const bType = b.ip_type.split('_')[0].toLowerCase()
          const aNum = parseInt(a.ip_type.split('_')[1] || '0')
          const bNum = parseInt(b.ip_type.split('_')[1] || '0')

          const aIsRN = rnTypes.includes(aType)
          const bIsRN = rnTypes.includes(bType)

          // 先按类型分组（RN在前，SN在后）
          if (aIsRN && !bIsRN) return -1
          if (!aIsRN && bIsRN) return 1

          // 同组内按编号排序
          return aNum - bNum
        })

        // 分别统计RN和SN的数量
        let rnCount = 0
        let snCount = 0

        sortedMounts.forEach((mount, idx) => {
          const colors = getIPTypeColor(mount.ip_type)
          // 提取IP类型缩写：gdma_0 -> G0, ddr_1 -> D1
          const parts = mount.ip_type.split('_')
          const typeAbbr = parts[0].charAt(0).toUpperCase()
          const number = parts[1] || '0'
          const shortLabel = `${typeAbbr}${number}`

          // 判断IP类型
          const ipType = mount.ip_type.split('_')[0].toLowerCase()
          const isRN = rnTypes.includes(ipType)

          // 计算IP块位置（在父节点内部）
          const baseX = node.col * 150 + 75
          const baseY = node.row * 150 + 75

          let offsetX, offsetY
          if (isRN) {
            // RN类型：第一排（上方）
            offsetX = (rnCount % 2) * 28 - 14  // 左右排列
            offsetY = -Math.floor(rnCount / 2) * 28 - 14  // 从上往下
            rnCount++
          } else {
            // SN类型：第二排（下方）
            offsetX = (snCount % 2) * 28 - 14  // 左右排列
            offsetY = Math.floor(snCount / 2) * 28 + 14  // 从中间往下
            snCount++
          }

          nodes.push({
            data: {
              id: `ip-${node.id}-${idx}`,
              label: shortLabel,
              ipType: mount.ip_type,
              parentNodeId: node.id,  // 记录父节点ID但不设置parent
              bgColor: colors.bg,
              borderColor: colors.border,
            },
            position: {
              x: baseX + offsetX,
              y: baseY + offsetY,
            },
            classes: 'ip-block',
            locked: true,  // 锁定IP块位置
            selectable: false,  // 不可选中
            style: {
              'z-index': 10
            }
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
      // 正向边 - 偏移到一侧
      edges.push({
        data: {
          id: `edge-${idx}-forward`,
          source: `node-${edge.source}`,
          target: `node-${edge.target}`,
          direction: edge.direction,
          type: edge.type,
          offset: 1,  // 正向偏移
        }
      })
      // 反向边 - 偏移到另一侧
      edges.push({
        data: {
          id: `edge-${idx}-backward`,
          source: `node-${edge.target}`,
          target: `node-${edge.source}`,
          direction: edge.direction,
          type: edge.type,
          offset: -1,  // 反向偏移
        }
      })
    })

    setElements([...nodes, ...edges])
  }, [data, mounts])

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
    // IP块样式 - 独立节点
    {
      selector: '.ip-block',
      style: {
        'shape': 'rectangle',
        'width': 24,
        'height': 24,
        'label': 'data(label)',
        'text-valign': 'center',
        'text-halign': 'center',
        'background-color': 'data(bgColor)',
        'border-width': 1,
        'border-color': 'data(borderColor)',
        'color': '#fff',
        'font-size': '9px',
        'font-weight': 'bold',
        'text-outline-width': 0,
        'z-index': 10,
        'events': 'yes',  // 允许事件
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
    // 基础边样式 - 使用taxi实现平行直线箭头
    {
      selector: 'edge',
      style: {
        'width': 2,
        'line-color': '#bfbfbf',
        'target-arrow-color': '#bfbfbf',
        'target-arrow-shape': 'triangle',
        'curve-style': 'taxi',
        'taxi-direction': (ele: any) => {
          return ele.data('direction') === 'horizontal' ? 'horizontal' : 'vertical'
        },
        'taxi-turn': (ele: any) => {
          const offset = ele.data('offset') || 0
          return offset * 8 + 'px'
        },
        'arrow-scale': 1.0,
        'opacity': 0.7,
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

      highlightNode(cy, nodeId)
      setSelectedNode(nodeId)

      if (onNodeClick) {
        onNodeClick(nodeId)
      }
    })

    // 点击IP块时，也触发对应节点的点击事件
    cy.on('tap', '.ip-block', (evt) => {
      const ipNode = evt.target
      const parentNodeId = ipNode.data('parentNodeId')

      highlightNode(cy, parentNodeId)
      setSelectedNode(parentNodeId)

      if (onNodeClick) {
        onNodeClick(parentNodeId)
      }
    })

    // 点击空白处取消高亮
    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        clearHighlight(cy)
        setSelectedNode(null)
      }
    })
  }

  const highlightNode = (cy: Cytoscape.Core, nodeId: number) => {
    // 重置所有样式
    cy.nodes().removeClass('highlighted dimmed')
    cy.edges().removeClass('highlighted dimmed')

    const selectedNode = cy.getElementById(`node-${nodeId}`)

    // 只高亮选中节点，不显示邻居节点
    selectedNode.addClass('highlighted')

    // 暗化其他所有节点和边
    cy.nodes().difference(selectedNode).addClass('dimmed')
    cy.edges().addClass('dimmed')
  }

  const clearHighlight = (cy: Cytoscape.Core) => {
    cy.nodes().removeClass('highlighted dimmed')
    cy.edges().removeClass('highlighted dimmed')
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
      // 清除高亮
      clearHighlight(cyRef.current)
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
      <Card>
        <Row style={{ marginBottom: 16 }} align="middle">
          <Col span={12}>
            <Space>
              <span>拓扑: {data.type}</span>
              <Tag color="blue">{data.total_nodes} 节点</Tag>
              <Tag color="green">{data.metadata.total_links} 链路</Tag>
              <Tag color="orange">{mounts.length} 已挂载</Tag>
              {selectedNode !== null && (
                <Tag color="purple">已选中节点 {selectedNode}</Tag>
              )}
            </Space>
          </Col>
          <Col span={12} style={{ textAlign: 'right' }}>
            <Space>
              <Button size="small" icon={<ZoomInOutlined />} onClick={handleZoomIn}>放大</Button>
              <Button size="small" icon={<ZoomOutOutlined />} onClick={handleZoomOut}>缩小</Button>
              <Button size="small" icon={<AimOutlined />} onClick={handleFit}>适应</Button>
              <Button size="small" icon={<ReloadOutlined />} onClick={handleReset}>重置</Button>
            </Space>
          </Col>
        </Row>
        <div style={{ height: 'calc(100vh - 310px)', minHeight: 740, border: '1px solid #d9d9d9', borderRadius: 4, background: '#fafafa' }}>
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
        <Row gutter={[16, 8]}>
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
