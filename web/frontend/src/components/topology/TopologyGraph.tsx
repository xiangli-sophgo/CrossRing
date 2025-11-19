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
        return { bg: '#1890ff', border: '#096dd9' }
      case 'npu':
        return { bg: '#722ed1', border: '#531dab' }
      case 'ddr':
        return { bg: '#13c2c2', border: '#08979c' }
      case 'pcie':
        return { bg: '#fa8c16', border: '#d46b08' }
      case 'eth':
        return { bg: '#52c41a', border: '#389e0d' }
      default:
        return { bg: '#eb2f96', border: '#c41d7f' }
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

      // 背景节点（方框）- z-index较低
      nodes.push({
        data: {
          id: `node-${node.id}`,
          label: hasMounts ? '' : `${node.id}`,  // 有IP时不显示节点ID
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
          'z-index': 1
        }
      })

      // 为每个IP创建独立节点（IP块）- z-index较高
      if (nodeMounts && nodeMounts.length > 0) {
        nodeMounts.forEach((mount, idx) => {
          const colors = getIPTypeColor(mount.ip_type)
          // 提取IP类型缩写：gdma_0 -> G0, ddr_1 -> D1
          const parts = mount.ip_type.split('_')
          const typeAbbr = parts[0].charAt(0).toUpperCase()
          const number = parts[1] || '0'
          const shortLabel = `${typeAbbr}${number}`

          // 计算IP块位置（在父节点内部）
          const baseX = node.col * 150 + 75
          const baseY = node.row * 150 + 75
          const offsetX = (idx % 2) * 28 - 14  // 左右排列，间距28
          const offsetY = Math.floor(idx / 2) * 28 - 14  // 上下排列

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
            style: {
              'z-index': 10
            }
          })
        })
      }
    })

    // 为每条边创建两条分开的线（双向）
    const edges: any[] = []
    data.edges.forEach((edge, idx) => {
      // 正向边
      edges.push({
        data: {
          id: `edge-${idx}-forward`,
          source: `node-${edge.source}`,
          target: `node-${edge.target}`,
          direction: edge.direction,
          type: edge.type,
        }
      })
      // 反向边
      edges.push({
        data: {
          id: `edge-${idx}-backward`,
          source: `node-${edge.target}`,
          target: `node-${edge.source}`,
          direction: edge.direction,
          type: edge.type,
        }
      })
    })

    setElements([...nodes, ...edges])
  }, [data, mounts])

  const stylesheet: any[] = [
    // 背景节点样式 - 固定大小的正方形容器
    {
      selector: 'node.mounted-node, node.unmounted',
      style: {
        'shape': 'rectangle',
        'width': 70,
        'height': 70,
        'label': 'data(label)',
        'text-valign': 'bottom',
        'text-halign': 'center',
        'text-margin-y': 5,
        'background-color': '#f5f5f5',
        'border-width': 2,
        'border-color': '#8c8c8c',
        'color': '#595959',
        'font-size': '12px',
        'font-weight': 'bold',
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
    // 已挂载的父节点
    {
      selector: '.mounted-node',
      style: {
        'background-color': '#fafafa',
        'border-color': '#d9d9d9',
      }
    },
    // 未挂载节点
    {
      selector: 'node.unmounted',
      style: {
        'background-color': '#d9d9d9',
        'border-color': '#bfbfbf',
      }
    },
    // 选中节点
    {
      selector: 'node:selected',
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
    // 基础边样式 - 两条分开的平行直线箭头
    {
      selector: 'edge',
      style: {
        'width': 2,
        'line-color': '#bfbfbf',
        'target-arrow-color': '#bfbfbf',
        'target-arrow-shape': 'triangle',
        'curve-style': 'segments',
        'segment-distances': (ele: any) => {
          const sourceId = ele.data('source')
          const targetId = ele.data('target')
          const isForward = sourceId.localeCompare(targetId) < 0
          // 正向边向一侧偏移,反向边向另一侧偏移
          return isForward ? [8] : [-8]
        },
        'segment-weights': [0.5],
        'arrow-scale': 1.0,
        'opacity': 0.7,
        'source-endpoint': 'outside-to-node',
        'target-endpoint': 'outside-to-node',
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

      // 重置缩放和位置
      cyRef.current.fit(undefined, 50)
    }
  }

  if (!data) {
    return (
      <Card style={{ height: 600 }}>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 550 }}>
          <span style={{ color: '#8c8c8c' }}>请选择拓扑类型</span>
        </div>
      </Card>
    )
  }

  return (
    <>
      <Card style={{ height: 600 }}>
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
        <div style={{ height: 540, border: '1px solid #d9d9d9', borderRadius: 4, background: '#fafafa' }}>
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
          {/* 未挂载节点 - 始终显示 */}
          <Col span={4}>
            <Space>
              <div style={{
                width: 40,
                height: 30,
                backgroundColor: '#d9d9d9',
                border: '2px solid #bfbfbf',
                borderRadius: 4,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 10,
                color: '#fff',
                fontWeight: 'bold'
              }}>0</div>
              <span>未挂载</span>
            </Space>
          </Col>

          {/* 动态显示已挂载的IP类型 */}
          {(() => {
            const ipTypeColors: Record<string, { bg: string; border: string; label: string }> = {
              'gdma': { bg: '#1890ff', border: '#096dd9', label: 'GDMA' },
              'npu': { bg: '#722ed1', border: '#531dab', label: 'NPU' },
              'ddr': { bg: '#13c2c2', border: '#08979c', label: 'DDR' },
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
                      width: 40,
                      height: 30,
                      backgroundColor: config.bg,
                      border: `2px solid ${config.border}`,
                      borderRadius: 4,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: 9,
                      color: '#fff',
                      fontWeight: 'bold'
                    }}>{config.label}</div>
                    <span>{config.label}</span>
                  </Space>
                </Col>
              )
            })
          })()}

          {/* 链路类型 - 始终显示 */}
          <Col span={6}>
            <Space>
              <div style={{
                width: 40,
                height: 6,
                backgroundColor: '#bfbfbf',
                borderRadius: 2,
              }}></div>
              <span>双向链路</span>
            </Space>
          </Col>
          <Col span={6}>
            <Space>
              <div style={{
                width: 40,
                height: 6,
                backgroundColor: '#faad14',
                borderRadius: 2,
              }}></div>
              <span>高亮链路</span>
            </Space>
          </Col>
        </Row>
      </Card>
    </>
  )
}

export default TopologyGraph
