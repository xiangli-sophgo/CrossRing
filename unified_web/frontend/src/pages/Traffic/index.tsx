/**
 * 流量配置页面 - 基于 tool_web 的功能
 */
import { useState, useEffect, useRef } from 'react'
import { Typography, Card, Space, message, Select, Table, Tag, Button, InputNumber, Tabs, Divider } from 'antd'
import {
  SaveOutlined,
  SettingOutlined,
  NodeIndexOutlined,
  ApiOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import { primaryColor, successColor } from '@/theme/colors'
import TopologyGraph from '@/components/topology/TopologyGraph'
import MultiDieTopologyGraph from '@/components/topology/MultiDieTopologyGraph'
import NodeInfoPanel from '@/components/topology/NodeInfoPanel'
import IPMountPanel from '@/components/traffic/IPMountPanel'
import TrafficConfigPanel from '@/components/traffic/TrafficConfigPanel'
import { generateTopology, getNodeInfoLocal } from '@/utils/topology'
import { getMounts } from '@/api/ipMount'
import type { TopologyData } from '@/types/topology'
import type { IPMount } from '@/types/ipMount'
import type { BandwidthComputeResponse, DCINLayoutInfo, FlowInfo } from '@/types/staticBandwidth'

const { Title, Text } = Typography
const { Option } = Select

interface NodeInfo {
  node_id: number
  position: { row: number; col: number }
  label: string
  neighbors: number[]
  degree: number
  topology: string
  die_id?: number
}

const TrafficConfig: React.FC = () => {
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

  const loadTopology = (topoType: string) => {
    setTopoLoading(true)
    const data = generateTopology(topoType)
    setTopoData(data)
    setTopoLoading(false)
  }

  const loadMounts = async () => {
    try {
      const data = await getMounts(selectedTopo)
      setMounts(data.mounts)
      setMountsVersion(prev => prev + 1)
    } catch (error) {
      console.error('加载IP挂载失败:', error)
    }
  }

  const handleNodeClick = (nodeId: number, dieId?: number) => {
    const info = getNodeInfoLocal(selectedTopo, nodeId, dieId)
    setNodeInfo(info)
  }

  const extractActiveIPsByDie = (composition: Record<string, FlowInfo[]>): Map<number, Set<string>> => {
    const result = new Map<number, Set<string>>()
    Object.values(composition).forEach(flows => {
      flows.forEach(flow => {
        const srcDie = flow.src_die ?? 0
        if (!result.has(srcDie)) result.set(srcDie, new Set())
        result.get(srcDie)!.add(`${flow.src_node}-${flow.src_ip}`)

        const dstDie = flow.dst_die ?? 0
        if (!result.has(dstDie)) result.set(dstDie, new Set())
        result.get(dstDie)!.add(`${flow.dst_node}-${flow.dst_ip}`)
      })
    })
    return result
  }

  const handleBandwidthComputed = (data: BandwidthComputeResponse) => {
    setLinkBandwidth(data.link_bandwidth)
    setLinkComposition(data.link_composition || {})
    setBandwidthMode(data.mode)
    setSelectedLinkKey('')
    setSelectedLinkFlows([])

    const composition = data.link_composition || {}
    const ipsByDie = extractActiveIPsByDie(composition)
    setActiveIPsByDie(ipsByDie)
    let hasActiveIPs = false
    ipsByDie.forEach(ips => { if (ips.size > 0) hasActiveIPs = true })
    setShowOnlyActiveIPs(hasActiveIPs)

    if (data.mode === 'dcin') {
      setSelectedDie(0)
      setDcinLayout(data.dcin_layout || null)
      setDcinLinkBandwidth(data.dcin_link_bandwidth || {})
    } else {
      setDcinLayout(null)
      setDcinLinkBandwidth({})
    }
  }

  const handleLinkClick = (linkKey: string, composition: FlowInfo[]) => {
    setSelectedLinkKey(linkKey)
    setSelectedLinkFlows(composition)
    if (!linkKey) {
      setNodeInfo(null)
    }
  }

  const handleSaveLayout = () => {
    if (bandwidthMode === 'dcin' && multiDieTopoGraphRef.current) {
      multiDieTopoGraphRef.current.saveLayout()
    } else if (topoGraphRef.current) {
      topoGraphRef.current.saveLayout()
    } else {
      message.warning('请先加载拓扑图')
    }
  }

  useEffect(() => {
    if (rows > 0 && cols > 0) {
      loadTopology(selectedTopo)
      loadMounts()
    }
  }, [rows, cols])

  return (
    <div>
      {/* 顶部配置栏 */}
      <Card style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Space size="large">
            <Space>
              <SettingOutlined style={{ color: primaryColor, fontSize: 16 }} />
              <Text strong>拓扑配置</Text>
            </Space>
            <Divider type="vertical" style={{ height: 24 }} />
            <Space>
              <Text type="secondary">行数:</Text>
              <InputNumber min={1} max={20} value={rows} onChange={(val) => val && setRows(val)} style={{ width: 70 }} />
            </Space>
            <Space>
              <Text type="secondary">列数:</Text>
              <InputNumber min={1} max={20} value={cols} onChange={(val) => val && setCols(val)} style={{ width: 70 }} />
            </Space>
            <Tag color="blue" style={{ fontSize: 14, padding: '4px 12px' }}>
              {selectedTopo} 拓扑
            </Tag>
          </Space>
          <Space>
            <Text type="secondary">流量模式:</Text>
            <Select value={trafficMode} onChange={setTrafficMode} style={{ width: 100 }}>
              <Option value="kcin">KCIN</Option>
              <Option value="dcin">DCIN</Option>
            </Select>
            <Button type="primary" icon={<SaveOutlined />} onClick={handleSaveLayout}>
              保存布局
            </Button>
          </Space>
        </div>
      </Card>

      <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start' }}>
        {/* 左侧：拓扑可视化 */}
        <div style={{ flexShrink: 0 }}>
          <Space direction="vertical" size={16}>
            <Card
              title={
                <Space>
                  <NodeIndexOutlined style={{ color: primaryColor }} />
                  <span>拓扑视图</span>
                </Space>
              }
              styles={{ body: { padding: 12 } }}
            >
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
            </Card>

            {topoData && nodeInfo && (
              <NodeInfoPanel nodeInfo={nodeInfo} loading={nodeInfoLoading} mounts={mounts} />
            )}

            {selectedLinkKey && selectedLinkFlows.length > 0 && topoData && (
              <Card
                title={
                  <Space>
                    <ApiOutlined style={{ color: successColor }} />
                    <span>链路带宽详情</span>
                    <Tag color="blue">{selectedLinkKey}</Tag>
                  </Space>
                }
                size="small"
              >
                <Table
                  dataSource={selectedLinkFlows.map((flow, idx) => ({ ...flow, key: idx }))}
                  columns={[
                    { title: '源DIE', dataIndex: 'src_die', width: 70, align: 'center' as const, render: (die: number | null | undefined) => die !== null && die !== undefined ? <Tag color="purple">{die}</Tag> : <Text type="secondary">-</Text> },
                    { title: '源节点', dataIndex: 'src_node', width: 70, align: 'center' as const, render: (v: number) => v ?? '-' },
                    { title: '源IP', dataIndex: 'src_ip', width: 100, align: 'center' as const, render: (ip: string) => ip ? <Tag color="blue">{ip}</Tag> : '-' },
                    { title: '目标DIE', dataIndex: 'dst_die', width: 70, align: 'center' as const, render: (die: number | null | undefined) => die !== null && die !== undefined ? <Tag color="purple">{die}</Tag> : <Text type="secondary">-</Text> },
                    { title: '目标节点', dataIndex: 'dst_node', width: 70, align: 'center' as const, render: (v: number) => v ?? '-' },
                    { title: '目标IP', dataIndex: 'dst_ip', width: 100, align: 'center' as const, render: (ip: string) => ip ? <Tag color="green">{ip}</Tag> : '-' },
                    { title: '带宽', dataIndex: 'bandwidth', width: 80, align: 'center' as const, render: (bw: number) => bw !== undefined ? <Text strong>{bw.toFixed(1)}</Text> : '-' },
                    { title: '类型', dataIndex: 'req_type', width: 60, align: 'center' as const, render: (t: string) => t ? <Tag color={t === 'R' ? 'cyan' : 'orange'}>{t}</Tag> : '-' },
                  ]}
                  size="small"
                  pagination={false}
                  scroll={{ x: 620 }}
                />
              </Card>
            )}
          </Space>
        </div>

        {/* 右侧：配置面板 */}
        <div style={{ flex: 1, minWidth: 300 }}>
          {topoData && (
            <Tabs
              defaultActiveKey="mount"
              type="card"
              items={[
                {
                  key: 'mount',
                  label: (
                    <Space>
                      <ApiOutlined />
                      <span>IP挂载</span>
                    </Space>
                  ),
                  children: <IPMountPanel topology={selectedTopo} onMountsChange={loadMounts} />,
                },
                {
                  key: 'traffic',
                  label: (
                    <Space>
                      <ThunderboltOutlined />
                      <span>流量配置</span>
                    </Space>
                  ),
                  children: (
                    <TrafficConfigPanel
                      topology={selectedTopo}
                      mode={trafficMode}
                      mountsVersion={mountsVersion}
                      onBandwidthComputed={handleBandwidthComputed}
                    />
                  ),
                },
              ]}
            />
          )}
        </div>
      </div>
    </div>
  )
}

export default TrafficConfig
