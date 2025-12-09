import React, { useState, useCallback } from 'react'
import { Layout, Typography } from 'antd'
import { ConfigPanel } from './components/ConfigPanel'
import { Scene3D } from './components/Scene3D'
import { LevelConfig, TopologyData } from './types'
import { generateTopology } from './api/topology'

const { Header, Sider, Content } = Layout
const { Title } = Typography

// 默认配置
const defaultLevels: LevelConfig[] = [
  { level: 'die', count: 4, topology: 'mesh', visible: true },
  { level: 'chip', count: 2, topology: 'mesh', visible: true },
  { level: 'board', count: 2, topology: 'mesh', visible: true },
  { level: 'server', count: 2, topology: 'mesh', visible: true },
  { level: 'pod', count: 1, topology: 'mesh', visible: true },
]

const App: React.FC = () => {
  const [levels, setLevels] = useState<LevelConfig[]>(defaultLevels)
  const [showInterLevel, setShowInterLevel] = useState(true)
  const [topologyData, setTopologyData] = useState<TopologyData | null>(null)
  const [loading, setLoading] = useState(false)

  // 加载拓扑数据
  const loadTopology = useCallback(async () => {
    setLoading(true)
    try {
      const data = await generateTopology(levels, showInterLevel)
      setTopologyData(data)
    } catch (error) {
      console.error('加载拓扑失败:', error)
    } finally {
      setLoading(false)
    }
  }, [levels, showInterLevel])

  // 初始加载
  React.useEffect(() => {
    loadTopology()
  }, [loadTopology])

  // 配置变更处理
  const handleConfigChange = (newLevels: LevelConfig[]) => {
    setLevels(newLevels)
  }

  const handleInterLevelChange = (show: boolean) => {
    setShowInterLevel(show)
  }

  const handleReset = () => {
    setLevels(defaultLevels)
    setShowInterLevel(true)
  }

  return (
    <Layout style={{ height: '100vh' }}>
      <Header style={{
        background: 'linear-gradient(135deg, #1890ff 0%, #722ed1 100%)',
        padding: '0 24px',
        display: 'flex',
        alignItems: 'center',
      }}>
        <Title level={4} style={{ color: '#fff', margin: 0 }}>
          Tier6+ 3D 拓扑配置器
        </Title>
      </Header>

      <Layout>
        <Sider
          width={320}
          style={{
            background: '#fff',
            padding: 16,
            overflow: 'auto',
          }}
        >
          <ConfigPanel
            levels={levels}
            showInterLevel={showInterLevel}
            onConfigChange={handleConfigChange}
            onInterLevelChange={handleInterLevelChange}
            onReset={handleReset}
          />
        </Sider>

        <Content style={{ position: 'relative' }}>
          <Scene3D
            topologyData={topologyData}
            loading={loading}
          />
        </Content>
      </Layout>
    </Layout>
  )
}

export default App
