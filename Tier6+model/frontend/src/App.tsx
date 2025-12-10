import React, { useState, useCallback, useEffect } from 'react'
import { Layout, Typography, Spin, message } from 'antd'
import { Scene3D } from './components/Scene3D'
import { ConfigPanel } from './components/ConfigPanel'
import { HierarchicalTopology } from './types'
import { getTopology, generateTopology } from './api/topology'
import { useViewNavigation } from './hooks/useViewNavigation'

const { Header, Sider, Content } = Layout
const { Title } = Typography

// localStorage缓存key（与ConfigPanel保持一致）
const CONFIG_CACHE_KEY = 'tier6_topology_config_cache'

const App: React.FC = () => {
  const [topology, setTopology] = useState<HierarchicalTopology | null>(null)
  const [loading, setLoading] = useState(true)  // 初始加载状态

  // 视图导航状态
  const navigation = useViewNavigation(topology)

  // 加载拓扑数据（优先使用缓存配置生成）
  const loadTopology = useCallback(async () => {
    setLoading(true)
    try {
      // 检查是否有缓存配置
      const cachedStr = localStorage.getItem(CONFIG_CACHE_KEY)
      if (cachedStr) {
        const cached = JSON.parse(cachedStr)
        // 使用缓存配置生成拓扑
        const data = await generateTopology({
          pod_count: cached.podCount,
          racks_per_pod: cached.racksPerPod,
          board_configs: cached.boardConfigs,
        })
        setTopology(data)
      } else {
        // 没有缓存，使用默认配置
        const data = await getTopology()
        setTopology(data)
      }
    } catch (error) {
      console.error('加载拓扑失败:', error)
      message.error('加载拓扑数据失败')
    } finally {
      setLoading(false)
    }
  }, [])

  // 重新生成拓扑
  const handleGenerate = useCallback(async (config: {
    pod_count: number
    racks_per_pod: number
    board_configs: {
      u1: { count: number; chips: { npu: number; cpu: number } }
      u2: { count: number; chips: { npu: number; cpu: number } }
      u4: { count: number; chips: { npu: number; cpu: number } }
    }
  }) => {
    try {
      const data = await generateTopology(config)
      setTopology(data)
    } catch (error) {
      console.error('生成拓扑失败:', error)
      message.error('生成拓扑失败')
    }
  }, [])

  // 初始加载
  useEffect(() => {
    loadTopology()
  }, [loadTopology])

  return (
    <Layout style={{ height: '100vh' }}>
      <Header style={{
        background: 'linear-gradient(135deg, #1890ff 0%, #722ed1 100%)',
        padding: '0 24px',
        display: 'flex',
        alignItems: 'center',
      }}>
        <Title level={4} style={{ color: '#fff', margin: 0 }}>
          Tier6+互联拓扑
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
            topology={topology}
            onGenerate={handleGenerate}
            loading={loading}
          />
        </Sider>

        <Content style={{ position: 'relative' }}>
          {loading && !topology ? (
            <div style={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              height: '100%',
            }}>
              <Spin size="large" tip="加载中..." />
            </div>
          ) : (
            <>
              <Scene3D
                topology={topology}
                viewState={navigation.viewState}
                breadcrumbs={navigation.breadcrumbs}
                currentPod={navigation.currentPod}
                currentRack={navigation.currentRack}
                currentBoard={navigation.currentBoard}
                onNavigate={navigation.navigateTo}
                onNavigateToPod={navigation.navigateToPod}
                onNavigateToRack={navigation.navigateToRack}
                onNavigateBack={navigation.navigateBack}
                onBreadcrumbClick={navigation.navigateToBreadcrumb}
                canGoBack={navigation.canGoBack}
              />
            </>
          )}
        </Content>
      </Layout>
    </Layout>
  )
}

export default App
