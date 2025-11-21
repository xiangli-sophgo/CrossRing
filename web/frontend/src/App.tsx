import { useState, useEffect } from 'react'
import { Layout, Typography, Card, Space, message, Select, Row, Col } from 'antd'
import { AppstoreOutlined } from '@ant-design/icons'
import TopologyGraph from './components/topology/TopologyGraph'
import NodeInfoPanel from './components/topology/NodeInfoPanel'
import IPMountPanel from './components/traffic/IPMountPanel'
import TrafficConfigPanel from './components/traffic/TrafficConfigPanel'
import { getAvailableTopologies, getTopology, getNodeInfo } from './api/topology'
import { getMounts } from './api/ipMount'
import type { TopologyData, TopologyInfo } from './types/topology'
import type { IPMount } from './types/ipMount'
import type { BandwidthComputeResponse } from './types/staticBandwidth'

const { Header, Content } = Layout
const { Title, Text } = Typography
const { Option } = Select

interface NodeInfo {
  node_id: number
  position: { row: number; col: number }
  label: string
  neighbors: number[]
  degree: number
  topology: string
}

function App() {
  const [topologies, setTopologies] = useState<TopologyInfo[]>([])
  const [selectedTopo, setSelectedTopo] = useState<string>('5x4')
  const [topoData, setTopoData] = useState<TopologyData | null>(null)
  const [topoLoading, setTopoLoading] = useState(false)
  const [nodeInfo, setNodeInfo] = useState<NodeInfo | null>(null)
  const [nodeInfoLoading, setNodeInfoLoading] = useState(false)
  const [trafficMode, setTrafficMode] = useState<'noc' | 'd2d'>('noc')
  const [mounts, setMounts] = useState<IPMount[]>([])
  const [mountsVersion, setMountsVersion] = useState(0)
  const [linkBandwidth, setLinkBandwidth] = useState<Record<string, number> | Record<string, Record<string, number>>>({})
  const [bandwidthMode, setBandwidthMode] = useState<'noc' | 'd2d'>('noc')
  const [selectedDie, setSelectedDie] = useState<number>(0)

  // 加载可用拓扑类型
  const loadTopologies = async () => {
    try {
      const data = await getAvailableTopologies()
      setTopologies(data.topologies)
    } catch (error) {
      message.error('加载拓扑类型失败')
      console.error(error)
    }
  }

  // 加载指定拓扑数据
  const loadTopology = async (topoType: string) => {
    setTopoLoading(true)
    try {
      const data = await getTopology(topoType)
      setTopoData(data)
      message.success(`已加载 ${topoType} 拓扑`)
    } catch (error) {
      message.error('加载拓扑数据失败')
      console.error(error)
    } finally {
      setTopoLoading(false)
    }
  }

  // 加载IP挂载
  const loadMounts = async () => {
    try {
      const data = await getMounts(selectedTopo)
      setMounts(data.mounts)
      setMountsVersion(prev => prev + 1)
    } catch (error) {
      console.error('加载IP挂载失败:', error)
    }
  }

  // 处理节点点击
  const handleNodeClick = async (nodeId: number) => {
    setNodeInfoLoading(true)
    try {
      const info = await getNodeInfo(selectedTopo, nodeId)
      setNodeInfo(info)
      console.log('节点信息:', info)
    } catch (error) {
      message.error('获取节点信息失败')
      console.error('获取节点信息失败:', error)
    } finally {
      setNodeInfoLoading(false)
    }
  }

  // 处理带宽计算完成
  const handleBandwidthComputed = (data: BandwidthComputeResponse) => {
    setLinkBandwidth(data.link_bandwidth)
    setBandwidthMode(data.mode)
    // D2D模式时重置选中的Die为0
    if (data.mode === 'd2d') {
      setSelectedDie(0)
    }
  }

  // 初始化
  useEffect(() => {
    loadTopologies()
  }, [])

  // 自动加载默认拓扑和IP挂载
  useEffect(() => {
    if (selectedTopo) {
      loadTopology(selectedTopo)
      loadMounts()
    }
  }, [selectedTopo])

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{
        background: '#001529',
        padding: '0 50px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <Title level={3} style={{ color: 'white', margin: 0 }}>
          <AppstoreOutlined /> NoC Web 工具
        </Title>
        <Space>
          <Text style={{ color: '#8c8c8c' }}>选择拓扑:</Text>
          <Select
            value={selectedTopo}
            onChange={setSelectedTopo}
            style={{ width: 120 }}
          >
            {topologies.map(t => (
              <Option key={t.type} value={t.type}>
                {t.type}
              </Option>
            ))}
          </Select>
        </Space>
      </Header>

      <Content style={{ padding: '24px 50px' }}>
        <Row gutter={24}>
          {/* 左侧：拓扑可视化 */}
          <Col span={10}>
            <Space direction="vertical" size="large" style={{ width: '100%' }}>
              {/* 拓扑图 */}
              <TopologyGraph
                data={topoData}
                mounts={mounts}
                loading={topoLoading}
                onNodeClick={handleNodeClick}
                linkBandwidth={linkBandwidth}
                bandwidthMode={bandwidthMode}
                selectedDie={selectedDie}
                onDieChange={setSelectedDie}
              />



              {/* 节点信息 */}
              {topoData && (
                <NodeInfoPanel nodeInfo={nodeInfo} loading={nodeInfoLoading} mounts={mounts} />
              )}
            </Space>
          </Col>

          {/* 右侧：IP挂载和流量配置 */}
          <Col span={14}>
            {topoData && (
              <Space direction="vertical" size="large" style={{ width: '100%' }}>
                <IPMountPanel
                  topology={selectedTopo}
                  onMountsChange={loadMounts}
                />

                <Card title="数据流配置与生成" size="small">
                  <Space direction="vertical" style={{ width: '100%' }} size="middle">
                    <Space>
                      <Text>流量模式:</Text>
                      <Select value={trafficMode} onChange={setTrafficMode} style={{ width: 120 }}>
                        <Option value="noc">NoC 模式</Option>
                        <Option value="d2d">D2D 模式</Option>
                      </Select>
                    </Space>
                    <TrafficConfigPanel
                      topology={selectedTopo}
                      mode={trafficMode}
                      mountsVersion={mountsVersion}
                      onBandwidthComputed={handleBandwidthComputed}
                    />
                  </Space>
                </Card>
              </Space>
            )}
          </Col>
        </Row>
      </Content>
    </Layout>
  )
}

export default App
