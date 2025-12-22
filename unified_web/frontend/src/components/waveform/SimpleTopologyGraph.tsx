/**
 * SimpleTopologyGraph - 简化版拓扑图，用于波形查看器
 * 只显示节点和IP，不显示连线
 */

import { useEffect, useRef, useMemo } from 'react'
import CytoscapeComponent from 'react-cytoscapejs'
import Cytoscape from 'cytoscape'
import type { TopologyData } from '@/types/topology'
import { getIPTypeColor as getThemeIPTypeColor } from '@/theme/colors'

interface SimpleTopologyGraphProps {
  data: TopologyData | null
  activeIPs: Record<number, string[]>  // {node_id: [ip_type1, ip_type2, ...]}
  selectedNode: number | null
  onNodeClick: (nodeId: number) => void
}

export default function SimpleTopologyGraph({
  data,
  activeIPs,
  selectedNode,
  onNodeClick,
}: SimpleTopologyGraphProps) {
  const cyRef = useRef<Cytoscape.Core | null>(null)

  // 获取IP类型颜色
  const getIPTypeColor = (ipType: string) => {
    return getThemeIPTypeColor(ipType)
  }

  // 节点ID转坐标
  const nodeIdToPos = (nodeId: number, cols: number): { col: number; row: number } => {
    return {
      col: nodeId % cols,
      row: Math.floor(nodeId / cols),
    }
  }

  // 构建Cytoscape元素
  const elements = useMemo(() => {
    if (!data) return []

    const nodes: any[] = []
    const nodeSpacing = 120  // 节点间距
    const nodeSize = 90      // 节点大小

    data.nodes.forEach(node => {
      const nodeIPs = activeIPs[node.id] || []
      const hasMounts = nodeIPs.length > 0

      // 背景节点
      nodes.push({
        data: {
          id: `node-${node.id}`,
          label: hasMounts ? '' : `${node.id}`,
          nodeId: node.id,
          row: node.row,
          col: node.col,
          mounted: hasMounts,
        },
        position: {
          x: node.col * nodeSpacing + nodeSpacing / 2,
          y: node.row * nodeSpacing + nodeSpacing / 2,
        },
        classes: hasMounts ? 'mounted-node' : 'unmounted',
      })

      // IP块
      if (hasMounts) {
        // 按IP类型分组
        const ipByType: Record<string, string[]> = {}
        nodeIPs.forEach(ipType => {
          const baseType = ipType.split('_')[0].toLowerCase()
          if (!ipByType[baseType]) ipByType[baseType] = []
          ipByType[baseType].push(ipType)
        })

        // 排序
        const priorityTypes = ['gdma', 'ddr', 'sdma', 'cdma', 'npu', 'pcie', 'eth', 'l2m', 'dcin']
        const sortedTypes = Object.keys(ipByType).sort((a, b) => {
          const aIdx = priorityTypes.indexOf(a)
          const bIdx = priorityTypes.indexOf(b)
          return (aIdx >= 0 ? aIdx : 99) - (bIdx >= 0 ? bIdx : 99)
        })

        // 计算IP块大小
        const numRows = sortedTypes.length
        const maxColsInRow = Math.max(...sortedTypes.map(t => ipByType[t].length))
        const maxDim = Math.max(numRows, maxColsInRow)
        let ipSize = maxDim <= 1 ? 40 : maxDim === 2 ? 32 : Math.max(20, Math.floor((nodeSize - 4 * (maxDim + 1)) / maxDim))
        const spacing = ipSize + 4

        const totalHeight = numRows * spacing
        const startY = -totalHeight / 2 + spacing / 2
        const baseX = node.col * nodeSpacing + nodeSpacing / 2
        const baseY = node.row * nodeSpacing + nodeSpacing / 2

        sortedTypes.forEach((baseType, rowIdx) => {
          const rowIPs = ipByType[baseType]
          const numCols = rowIPs.length
          const totalWidth = numCols * spacing
          const startX = -totalWidth / 2 + spacing / 2

          rowIPs.forEach((ipType, colIdx) => {
            const colors = getIPTypeColor(ipType)
            const parts = ipType.split('_')
            const typeAbbr = parts[0].charAt(0).toUpperCase()
            const number = parts[1] || '0'
            const shortLabel = `${typeAbbr}${number}`

            nodes.push({
              data: {
                id: `ip-${node.id}-${ipType}`,
                label: shortLabel,
                ipType: ipType,
                parentNodeId: node.id,
                bgColor: colors.bg,
                borderColor: colors.border,
                ipSize: ipSize,
              },
              position: {
                x: baseX + startX + colIdx * spacing,
                y: baseY + startY + rowIdx * spacing,
              },
              classes: 'ip-block',
              locked: true,
              selectable: false,
            })
          })
        })

        // 节点编号层
        nodes.push({
          data: {
            id: `label-${node.id}`,
            label: `${node.id}`,
            parentNodeId: node.id,
          },
          position: {
            x: baseX,
            y: baseY,
          },
          classes: 'node-label',
          locked: true,
          grabbable: false,
          selectable: false,
        })
      }
    })

    return nodes
  }, [data, activeIPs])

  // 样式表
  const stylesheet: any[] = [
    {
      selector: 'node.mounted-node, node.unmounted',
      style: {
        'shape': 'rectangle',
        'width': 90,
        'height': 90,
        'label': 'data(label)',
        'text-valign': 'center',
        'text-halign': 'center',
        'background-color': '#d9d9d9',
        'border-width': 2,
        'border-color': '#bfbfbf',
        'color': '#000000',
        'font-size': '16px',
        'font-weight': 'bold',
        'text-outline-width': 2,
        'text-outline-color': '#ffffff',
        'z-index': 1,
      },
    },
    {
      selector: '.ip-block',
      style: {
        'shape': 'rectangle',
        'width': (ele: any) => ele.data('ipSize') || 28,
        'height': (ele: any) => ele.data('ipSize') || 28,
        'label': 'data(label)',
        'text-valign': 'center',
        'text-halign': 'center',
        'background-color': 'data(bgColor)',
        'border-width': 1,
        'border-color': 'data(borderColor)',
        'color': '#fff',
        'font-size': (ele: any) => {
          const size = ele.data('ipSize') || 28
          return size >= 32 ? '12px' : '10px'
        },
        'font-weight': 'bold',
        'z-index': 15,
      },
    },
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
        'font-size': '16px',
        'font-weight': 'bold',
        'text-outline-width': 2,
        'text-outline-color': '#ffffff',
        'z-index': 100,
        'events': 'no',
      },
    },
    {
      selector: 'node.mounted-node:selected, node.unmounted:selected',
      style: {
        'border-width': 3,
        'border-color': '#1890ff',
      },
    },
  ]

  // 初始化Cytoscape
  const handleCyInit = (cy: Cytoscape.Core) => {
    cyRef.current = cy

    // 节点点击事件
    cy.on('tap', 'node.mounted-node, node.unmounted', (evt) => {
      const node = evt.target
      const nodeId = node.data('nodeId')
      onNodeClick(nodeId)
    })

    // IP块点击事件
    cy.on('tap', '.ip-block', (evt) => {
      const ipNode = evt.target
      const parentNodeId = ipNode.data('parentNodeId')
      onNodeClick(parentNodeId)

      cy.$(':selected').unselect()
      cy.$(`#node-${parentNodeId}`).select()
    })

    // 初始fit
    setTimeout(() => {
      cy.fit(undefined, 20)
    }, 100)
  }

  // 选中节点高亮
  useEffect(() => {
    if (!cyRef.current) return
    const cy = cyRef.current

    cy.$(':selected').unselect()
    if (selectedNode !== null) {
      const nodeElement = cy.$(`#node-${selectedNode}`)
      if (nodeElement.length > 0) {
        nodeElement.select()
      }
    }
  }, [selectedNode])

  if (!data) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <span style={{ color: '#999' }}>无拓扑数据</span>
      </div>
    )
  }

  return (
    <div style={{ width: '100%', height: '100%', border: '1px solid #d9d9d9', borderRadius: 4, background: '#fafafa' }}>
      <CytoscapeComponent
        key={`simple-topo-${data.type}-${Object.keys(activeIPs).length}`}
        elements={elements}
        style={{ width: '100%', height: '100%' }}
        stylesheet={stylesheet}
        cy={handleCyInit}
        layout={{ name: 'preset' }}
        userZoomingEnabled={false}
        userPanningEnabled={true}
        autoungrabify={true}
        boxSelectionEnabled={false}
      />
    </div>
  )
}
