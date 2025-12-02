import { useState, useEffect, useRef } from 'react'
import { Layout, Typography, Card, Space, message, Select, Table, Tag, Button, InputNumber } from 'antd'
import { AppstoreOutlined, SaveOutlined } from '@ant-design/icons'
import TopologyGraph from './components/topology/TopologyGraph'
import MultiDieTopologyGraph from './components/topology/MultiDieTopologyGraph'
import NodeInfoPanel from './components/topology/NodeInfoPanel'
import IPMountPanel from './components/traffic/IPMountPanel'
import TrafficConfigPanel from './components/traffic/TrafficConfigPanel'
import { generateTopology, getNodeInfoLocal } from './utils/topology'
import { getMounts } from './api/ipMount'
import type { TopologyData } from './types/topology'
import type { IPMount } from './types/ipMount'
import type { BandwidthComputeResponse, DCINLayoutInfo, FlowInfo } from './types/staticBandwidth'

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
  die_id?: number  // DCIN模式下的Die编号
}

function App() {
  const [rows, setRows] = useState<number>(5)
  const [cols, setCols] = useState<number>(4)
  const selectedTopo = `${rows}x${cols}`
  const [topoData, setTopoData] = useState<TopologyData | null>(null)
  const [topoLoading, setTopoLoading] = useState(false)
  const [nodeInfo, setNodeInfo] = useState<NodeInfo | null>(null)
  const [nodeInfoLoading, setNodeInfoLoading] = useState(false)
  const [trafficMode, setTrafficMode] = useState<'kcin' | 'dcin'>('kcin')
  const [mounts, setMounts] = useState<IPMount[]>([])
  const [mountsVersion, setMountsVersion] = useState(0)
  const [linkBandwidth, setLinkBandwidth] = useState<Record<string, number> | Record<string, Record<string, number>>>({})
  const [linkComposition, setLinkComposition] = useState<Record<string, any[]>>({})
  const [bandwidthMode, setBandwidthMode] = useState<'kcin' | 'dcin'>('kcin')
  const [selectedDie, setSelectedDie] = useState<number>(0)
  const [dcinLayout, setDcinLayout] = useState<DCINLayoutInfo | null>(null)
  const [selectedLinkKey, setSelectedLinkKey] = useState<string>('')
  const [selectedLinkFlows, setSelectedLinkFlows] = useState<FlowInfo[]>([])
  const [dcinLinkBandwidth, setDcinLinkBandwidth] = useState<Record<string, number>>({})
  const [activeIPsByDie, setActiveIPsByDie] = useState<Map<number, Set<string>>>(new Map())
  const [showOnlyActiveIPs, setShowOnlyActiveIPs] = useState(false)
  const topoGraphRef = useRef<{ saveLayout: () => void }>(null)
  const multiDieTopoGraphRef = useRef<{ saveLayout: () => void }>(null)

  // 加载指定拓扑数据
  const loadTopology = (topoType: string) => {
    setTopoLoading(true)
    const data = generateTopology(topoType)
    setTopoData(data)
    setTopoLoading(false)
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
  const handleNodeClick = (nodeId: number, dieId?: number) => {
    const info = getNodeInfoLocal(selectedTopo, nodeId, dieId)
    setNodeInfo(info)
  }

  // 从link_composition中按Die提取使用的IP
  const extractActiveIPsByDie = (composition: Record<string, FlowInfo[]>): Map<number, Set<string>> => {
    const result = new Map<number, Set<string>>()
    Object.values(composition).forEach(flows => {
      flows.forEach(flow => {
        // 源IP
        const srcDie = flow.src_die ?? 0
        if (!result.has(srcDie)) result.set(srcDie, new Set())
        result.get(srcDie)!.add(`${flow.src_node}-${flow.src_ip}`)

        // 目标IP
        const dstDie = flow.dst_die ?? 0
        if (!result.has(dstDie)) result.set(dstDie, new Set())
        result.get(dstDie)!.add(`${flow.dst_node}-${flow.dst_ip}`)
      })
    })
    return result
  }

  // 处理带宽计算完成
  const handleBandwidthComputed = (data: BandwidthComputeResponse) => {
    setLinkBandwidth(data.link_bandwidth)
    setLinkComposition(data.link_composition || {})
    setBandwidthMode(data.mode)
    setSelectedLinkKey('')
    setSelectedLinkFlows([])

    // 按Die提取使用中的IP并默认开启过滤
    const composition = data.link_composition || {}
    const ipsByDie = extractActiveIPsByDie(composition)
    setActiveIPsByDie(ipsByDie)
    // 检查是否有任何活跃IP
    let hasActiveIPs = false
    ipsByDie.forEach(ips => { if (ips.size > 0) hasActiveIPs = true })
    setShowOnlyActiveIPs(hasActiveIPs)

    // DCIN模式时重置选中的Die为0，并设置布局
    if (data.mode === 'dcin') {
      setSelectedDie(0)
      setDcinLayout(data.dcin_layout || null)
      setDcinLinkBandwidth(data.dcin_link_bandwidth || {})
    } else {
      setDcinLayout(null)
      setDcinLinkBandwidth({})
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
    if (bandwidthMode === 'dcin' && multiDieTopoGraphRef.current) {
      multiDieTopoGraphRef.current.saveLayout()
    } else if (topoGraphRef.current) {
      topoGraphRef.current.saveLayout()
    } else {
      message.warning('请先加载拓扑图')
    }
  }

  // 当行列变化时自动加载拓扑和IP挂载
  useEffect(() => {
    if (rows > 0 && cols > 0) {
      loadTopology(selectedTopo)
      loadMounts()
    }
  }, [rows, cols])

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
            <AppstoreOutlined /> KCIN Web 工具
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
          <Text style={{ color: '#8c8c8c' }}>行:</Text>
          <InputNumber
            min={1}
            max={20}
            value={rows}
            onChange={(val) => val && setRows(val)}
            style={{ width: 70 }}
          />
          <Text style={{ color: '#8c8c8c' }}>列:</Text>
          <InputNumber
            min={1}
            max={20}
            value={cols}
            onChange={(val) => val && setCols(val)}
            style={{ width: 70 }}
          />
        </Space>
      </Header>

      <Content style={{ padding: '24px 50px' }}>
        <div style={{ display: 'flex', width: '100%', gap: 24, overflow: 'hidden' }}>
          {/* 左侧：拓扑可视化 */}
          <div style={{ flexShrink: 0, overflow: 'visible' }}>
            <Space direction="vertical" size="large" style={{ width: '100%' }}>
              {/* 拓扑图 - DCIN模式显示多Die视图 */}
              {bandwidthMode === 'dcin' && dcinLayout ? (
                <MultiDieTopologyGraph
                  ref={multiDieTopoGraphRef}
                  data={topoData}
                  mounts={mounts}
                  loading={topoLoading}
                  linkBandwidth={linkBandwidth as Record<string, Record<string, number>>}
                  dcinLinkBandwidth={dcinLinkBandwidth}
                  linkComposition={linkComposition}
                  dcinLayout={dcinLayout}
                  onNodeClick={handleNodeClick}
                  onLinkClick={handleLinkClick}
                  activeIPsByDie={activeIPsByDie}
                  showOnlyActiveIPs={showOnlyActiveIPs}
                  onShowOnlyActiveIPsChange={setShowOnlyActiveIPs}
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
                  activeIPs={activeIPsByDie.get(0)}
                  showOnlyActiveIPs={showOnlyActiveIPs}
                  onShowOnlyActiveIPsChange={setShowOnlyActiveIPs}
                />
              )}

              {/* 节点信息 */}
              {topoData && nodeInfo && (
                <NodeInfoPanel nodeInfo={nodeInfo} loading={nodeInfoLoading} mounts={mounts} />
              )}

              {/* 链路带宽组成 */}
              {selectedLinkKey && selectedLinkFlows.length > 0 && topoData && (
                <Card title={`链路带宽组成 (${(() => {
                  // DCIN链路格式: dcin-{src_die}-{src_node}-{dst_die}-{dst_node}
                  if (selectedLinkKey.startsWith('dcin-')) {
                    const parts = selectedLinkKey.split('-')
                    if (parts.length === 5) {
                      const srcDie = parts[1], srcNode = parts[2], dstDie = parts[3], dstNode = parts[4]
                      return `Die${srcDie}-节点${srcNode} → Die${dstDie}-节点${dstNode}`
                    }
                  }
                  // DIE内链路格式: {die_id}-{col},{row}-{col},{row}
                  const parts = selectedLinkKey.split('-')
                  if (parts.length === 3 && parts[1].includes(',') && parts[2].includes(',')) {
                    const dieId = parts[0]
                    const [srcCol, srcRow] = parts[1].split(',').map(Number)
                    const [dstCol, dstRow] = parts[2].split(',').map(Number)
                    const srcNode = srcRow * topoData.cols + srcCol
                    const dstNode = dstRow * topoData.cols + dstCol
                    return `Die${dieId}-节点${srcNode} → 节点${dstNode}`
                  }
                  return selectedLinkKey
                })()})`} size="small" style={{ maxWidth: '100%' }}>
                  <Table
                    dataSource={selectedLinkFlows.map((flow, idx) => ({ ...flow, key: idx }))}
                    columns={[
                      { title: '源DIE', dataIndex: 'src_die', width: 50, align: 'center' as const, render: (die: number | null | undefined) => die !== null && die !== undefined ? <Tag color="purple">{die}</Tag> : <span style={{ color: '#999' }}>-</span> },
                      { title: '源', dataIndex: 'src_node', width: 40, align: 'center' as const },
                      { title: '源IP', dataIndex: 'src_ip', width: 70, align: 'center' as const, render: (ip: string) => <Tag color="blue" style={{ fontSize: 10 }}>{ip}</Tag> },
                      { title: '目标DIE', dataIndex: 'dst_die', width: 50, align: 'center' as const, render: (die: number | null | undefined) => die !== null && die !== undefined ? <Tag color="purple">{die}</Tag> : <span style={{ color: '#999' }}>-</span> },
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
                        <Option value="kcin">KCIN 模式</Option>
                        <Option value="dcin">DCIN 模式</Option>
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
