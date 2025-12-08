import { Card, Descriptions, Tag, Space } from 'antd'
import { NodeIndexOutlined } from '@ant-design/icons'
import type { IPMount } from '../../types/ipMount'

interface NodeInfo {
  node_id: number
  position: { row: number; col: number }
  label: string
  neighbors: number[]
  degree: number
  topology: string
  die_id?: number
}

interface NodeInfoPanelProps {
  nodeInfo: NodeInfo | null
  loading?: boolean
  mounts: IPMount[]
}

const NodeInfoPanel: React.FC<NodeInfoPanelProps> = ({ nodeInfo, loading, mounts }) => {
  if (!nodeInfo) {
    return (
      <Card
        title={
          <Space>
            <NodeIndexOutlined />
            <span>节点信息</span>
          </Space>
        }
        size="small"
      >
        <div style={{ textAlign: 'center', padding: '20px 0', color: '#8c8c8c' }}>
          点击拓扑图中的节点查看详细信息
        </div>
      </Card>
    )
  }

  // 获取当前节点的IP挂载
  const nodeMounts = mounts.filter(mount => mount.node_id === nodeInfo.node_id)

  // IP类型颜色映射
  const getIPColor = (ipType: string): string => {
    const type = ipType.split('_')[0].toLowerCase()
    switch (type) {
      case 'gdma':
      case 'sdma':
        return 'blue'
      case 'cdma':
        return 'green'
      case 'ddr':
      case 'l2m':
        return 'red'
      case 'npu':
        return 'purple'
      case 'pcie':
        return 'orange'
      case 'eth':
        return 'cyan'
      default:
        return 'magenta'
    }
  }

  // 从拓扑类型解析总行数，用于坐标转换
  const topoMatch = nodeInfo.topology.match(/^(\d+)x(\d+)$/)
  const totalRows = topoMatch ? parseInt(topoMatch[1]) : 0

  // 转换坐标：以左下角为原点(0,0)
  const xCoord = nodeInfo.position.col
  const yCoord = totalRows > 0 ? (totalRows - 1 - nodeInfo.position.row) : nodeInfo.position.row

  return (
    <Card
      title={
        <Space>
          <NodeIndexOutlined />
          <span>{nodeInfo.die_id !== undefined ? `Die${nodeInfo.die_id} 节点${nodeInfo.node_id}` : `节点 ${nodeInfo.node_id}`} 详细信息</span>
        </Space>
      }
      size="small"
      loading={loading}
    >
      <Descriptions column={3} size="small" bordered>
        <Descriptions.Item label="节点编号">
          {nodeInfo.node_id}
        </Descriptions.Item>
        <Descriptions.Item label="X坐标">
          {xCoord}
        </Descriptions.Item>
        <Descriptions.Item label="Y坐标">
          {yCoord}
        </Descriptions.Item>
        <Descriptions.Item label="挂载IP" span={3}>
          {nodeMounts.length > 0 ? (
            <Space wrap>
              {nodeMounts.map((mount, idx) => (
                <Tag key={idx} color={getIPColor(mount.ip_type)}>
                  {mount.ip_type}
                </Tag>
              ))}
            </Space>
          ) : (
            <span style={{ color: '#8c8c8c' }}>未挂载</span>
          )}
        </Descriptions.Item>
        <Descriptions.Item label="邻居节点" span={3}>
          {nodeInfo.neighbors.join(', ')}
        </Descriptions.Item>
      </Descriptions>
    </Card>
  )
}

export default NodeInfoPanel
