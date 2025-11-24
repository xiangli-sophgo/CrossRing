import { useState, useEffect, useCallback, useRef } from 'react'
import { Layout, Typography, Card, Space, message, Select, Table, Tag, Button } from 'antd'
import { AppstoreOutlined, SaveOutlined } from '@ant-design/icons'
import TopologyGraph from './components/topology/TopologyGraph'
import MultiDieTopologyGraph from './components/topology/MultiDieTopologyGraph'
import NodeInfoPanel from './components/topology/NodeInfoPanel'
import IPMountPanel from './components/traffic/IPMountPanel'
import TrafficConfigPanel from './components/traffic/TrafficConfigPanel'
import { getAvailableTopologies, getTopology, getNodeInfo } from './api/topology'
import { getMounts } from './api/ipMount'
import type { TopologyData, TopologyInfo } from './types/topology'
import type { IPMount } from './types/ipMount'
import type { BandwidthComputeResponse, D2DLayoutInfo, FlowInfo } from './types/staticBandwidth'

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
  die_id?: number  // D2D模式下的Die编号
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
  const [linkComposition, setLinkComposition] = useState<Record<string, any[]>>({})
  const [bandwidthMode, setBandwidthMode] = useState<'noc' | 'd2d'>('noc')
  const [selectedDie, setSelectedDie] = useState<number>(0)
  const [d2dLayout, setD2dLayout] = useState<D2DLayoutInfo | null>(null)
  const [selectedLinkKey, setSelectedLinkKey] = useState<string>('')
  const [selectedLinkFlows, setSelectedLinkFlows] = useState<FlowInfo[]>([])
  const [d2dLinkBandwidth, setD2dLinkBandwidth] = useState<Record<string, number>>({})
  const topoGraphRef = useRef<{ saveLayout: () => void }>(null)
  const multiDieTopoGraphRef = useRef<{ saveLayout: () => void }>(null)

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
  const handleNodeClick = async (nodeId: number, dieId?: number) => {
    setNodeInfoLoading(true)
    try {
      const info = await getNodeInfo(selectedTopo, nodeId)
      // D2D模式下添加die_id
      if (dieId !== undefined) {
        info.die_id = dieId
      }
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
    setLinkComposition(data.link_composition || {})
    setBandwidthMode(data.mode)
    setSelectedLinkKey('')
    setSelectedLinkFlows([])
    // D2D模式时重置选中的Die为0，并设置布局
    if (data.mode === 'd2d') {
      setSelectedDie(0)
      setD2dLayout(data.d2d_layout || null)
      setD2dLinkBandwidth(data.d2d_link_bandwidth || {})
    } else {
      setD2dLayout(null)
      setD2dLinkBandwidth({})
    }
  }

  // 处理链路点击
  const handleLinkClick = (linkKey: string, composition: FlowInfo[]) => {
    setSelectedLinkKey(linkKey)
    setSelectedLinkFlows(composition)
    if (!linkKey) {
      setNodeInfo(null)
    }
  }

  // 保存布局
  const handleSaveLayout = () => {
    if (bandwidthMode === 'd2d' && multiDieTopoGraphRef.current) {
      multiDieTopoGraphRef.current.saveLayout()
    } else if (topoGraphRef.current) {
      topoGraphRef.current.saveLayout()
    } else {
      message.warning('请先加载拓扑图')
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
        <Space>
          <Title level={3} style={{ color: 'white', margin: 0 }}>
            <AppstoreOutlined /> NoC Web 工具
          </Title>
          <Button
            type="primary"
            icon={<SaveOutlined />}
            onClick={handleSaveLayout}
            style={{ marginLeft: 16 }}
          >
            保存布局
          </Button>
        </Space>
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
        <div style={{ display: 'flex', width: '100%', gap: 24, overflow: 'hidden' }}>
          {/* 左侧：拓扑可视化 */}
          <div style={{ flexShrink: 0, overflow: 'visible' }}>
            <Space direction="vertical" size="large" style={{ width: '100%' }}>
              {/* 拓扑图 - D2D模式显示多Die视图 */}
              {bandwidthMode === 'd2d' && d2dLayout ? (
                <MultiDieTopologyGraph
                  ref={multiDieTopoGraphRef}
                  data={topoData}
                  mounts={mounts}
                  loading={topoLoading}
                  linkBandwidth={linkBandwidth as Record<string, Record<string, number>>}
                  d2dLinkBandwidth={d2dLinkBandwidth}
                  linkComposition={linkComposition}
                  d2dLayout={d2dLayout}
                  onNodeClick={handleNodeClick}
                  onLinkClick={handleLinkClick}
                />
              ) : (
                <TopologyGraph
                  ref={topoGraphRef}
                  data={topoData}
                  mounts={mounts}
                  loading={topoLoading}
                  onNodeClick={handleNodeClick}
                  linkBandwidth={linkBandwidth}
                  linkComposition={linkComposition}
                  bandwidthMode={bandwidthMode}
                  selectedDie={selectedDie}
                  onDieChange={setSelectedDie}
                  onLinkClick={handleLinkClick}
                />
              )}

              {/* 节点信息 */}
              {topoData && nodeInfo && (
                <NodeInfoPanel nodeInfo={nodeInfo} loading={nodeInfoLoading} mounts={mounts} />
              )}

              {/* 链路带宽组成 */}
              {selectedLinkKey && selectedLinkFlows.length > 0 && topoData && (
                <Card title={`链路带宽组成 (节点${(() => {
                  const [src, dst] = selectedLinkKey.split('-')
                  const [srcCol, srcRow] = src.split(',').map(Number)
                  const [dstCol, dstRow] = dst.split(',').map(Number)
                  const srcNode = srcRow * topoData.cols + srcCol
                  const dstNode = dstRow * topoData.cols + dstCol
                  return `${srcNode} → ${dstNode}`
                })()})`} size="small" style={{ maxWidth: '100%' }}>
                  <Table
                    dataSource={selectedLinkFlows.map((flow, idx) => ({ ...flow, key: idx }))}
                    columns={[
                      { title: '源', dataIndex: 'src_node', width: 40, align: 'center' as const },
                      { title: '源IP', dataIndex: 'src_ip', width: 70, align: 'center' as const, render: (ip: string) => <Tag color="blue" style={{ fontSize: 10 }}>{ip}</Tag> },
                      { title: '目标', dataIndex: 'dst_node', width: 40, align: 'center' as const },
                      { title: '目标IP', dataIndex: 'dst_ip', width: 70, align: 'center' as const, render: (ip: string) => <Tag color="green" style={{ fontSize: 10 }}>{ip}</Tag> },
                      { title: 'GB/s', dataIndex: 'bandwidth', width: 50, align: 'center' as const, render: (bw: number) => bw.toFixed(1) },
                      { title: '类型', dataIndex: 'req_type', width: 40, align: 'center' as const, render: (t: string) => <Tag color={t === 'R' ? 'cyan' : 'orange'}>{t}</Tag> },
                    ]}
                    size="small"
                    pagination={false}
                    scroll={{ y: 200, x: 'max-content' }}
                  />
                </Card>
              )}
            </Space>
          </div>

          {/* 右侧：IP挂载和流量配置 */}
          <div style={{ flex: 1, minWidth: '300px', overflow: 'auto' }}>
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
          </div>
        </div>
      </Content>
    </Layout>
  )
}

export default App
