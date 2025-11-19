import { Card, Descriptions, Tag, Space } from 'antd'
import { NodeIndexOutlined } from '@ant-design/icons'

interface NodeInfo {
  node_id: number
  position: { row: number; col: number }
  label: string
  neighbors: number[]
  degree: number
  topology: string
}

interface NodeInfoPanelProps {
  nodeInfo: NodeInfo | null
  loading?: boolean
}

const NodeInfoPanel: React.FC<NodeInfoPanelProps> = ({ nodeInfo, loading }) => {
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

  return (
    <Card
      title={
        <Space>
          <NodeIndexOutlined />
          <span>节点 {nodeInfo.node_id} 详细信息</span>
        </Space>
      }
      size="small"
      loading={loading}
    >
      <Descriptions column={2} size="small" bordered>
        <Descriptions.Item label="节点ID" span={2}>
          <Tag color="blue">{nodeInfo.node_id}</Tag>
        </Descriptions.Item>
        <Descriptions.Item label="行位置">
          {nodeInfo.position.row}
        </Descriptions.Item>
        <Descriptions.Item label="列位置">
          {nodeInfo.position.col}
        </Descriptions.Item>
        <Descriptions.Item label="节点度数" span={2}>
          <Tag color="green">{nodeInfo.degree}</Tag>
        </Descriptions.Item>
        <Descriptions.Item label="邻居节点" span={2}>
          <Space wrap>
            {nodeInfo.neighbors.map(neighbor => (
              <Tag key={neighbor} color="cyan">{neighbor}</Tag>
            ))}
          </Space>
        </Descriptions.Item>
      </Descriptions>
    </Card>
  )
}

export default NodeInfoPanel
