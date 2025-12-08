import React, { useState } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Layout, Menu, Typography, Button, Tooltip } from 'antd'
import {
  DashboardOutlined,
  ThunderboltOutlined,
  ExperimentOutlined,
  SettingOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  GithubOutlined,
  LineChartOutlined,
} from '@ant-design/icons'
import { useNavigate, useLocation } from 'react-router-dom'
import { primaryColor } from './theme/colors'

// 页面组件
import Dashboard from './pages/Dashboard'
import TrafficConfig from './pages/Traffic'
import Simulation from './pages/Simulation'
import ExperimentList from './pages/Experiments/ExperimentList'
import ExperimentDetail from './pages/Experiments/ExperimentDetail'
import Analysis from './pages/Analysis'

const { Header, Sider, Content } = Layout
const { Title, Text } = Typography

// 侧边栏菜单项
const menuItems = [
  {
    key: '/',
    icon: <DashboardOutlined />,
    label: '概览',
  },
  {
    key: '/traffic',
    icon: <SettingOutlined />,
    label: '流量配置',
  },
  {
    key: '/simulation',
    icon: <ThunderboltOutlined />,
    label: '仿真执行',
  },
  {
    key: '/experiments',
    icon: <ExperimentOutlined />,
    label: '结果管理',
  },
  {
    key: '/analysis',
    icon: <LineChartOutlined />,
    label: '结果分析',
  },
]

// 主布局组件
const MainLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const navigate = useNavigate()
  const location = useLocation()
  const [collapsed, setCollapsed] = useState(false)

  // 获取当前选中的菜单项
  const getSelectedKey = () => {
    const path = location.pathname
    if (path === '/') return '/'
    if (path.startsWith('/traffic')) return '/traffic'
    if (path.startsWith('/simulation')) return '/simulation'
    if (path.startsWith('/experiments')) return '/experiments'
    if (path.startsWith('/analysis')) return '/analysis'
    return '/'
  }

  // 获取页面标题
  const getPageTitle = () => {
    const path = location.pathname
    if (path === '/') return '概览'
    if (path.startsWith('/traffic')) return '流量配置'
    if (path.startsWith('/simulation')) return '仿真执行'
    if (path.startsWith('/experiments')) {
      if (path.includes('/experiments/')) return '实验详情'
      return '结果管理'
    }
    if (path.startsWith('/analysis')) return '结果分析'
    return '仿真一体化平台'
  }

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        collapsible
        collapsed={collapsed}
        onCollapse={setCollapsed}
        trigger={null}
        width={220}
        collapsedWidth={80}
        style={{
          background: '#fff',
          borderRight: '1px solid #f0f0f0',
          position: 'fixed',
          left: 0,
          top: 0,
          bottom: 0,
          zIndex: 100,
          overflow: 'auto',
        }}
      >
        {/* Logo区域 */}
        <div
          style={{
            height: 64,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            borderBottom: '1px solid #f0f0f0',
            padding: collapsed ? '0 8px' : '0 16px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {!collapsed && (
              <Title level={5} style={{ margin: 0, color: primaryColor, whiteSpace: 'nowrap' }}>
                仿真一体化平台
              </Title>
            )}
            {collapsed && (
              <div
                style={{
                  width: 32,
                  height: 32,
                  borderRadius: 8,
                  background: `linear-gradient(135deg, ${primaryColor} 0%, #4096ff 100%)`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#fff',
                  fontWeight: 700,
                  fontSize: 14,
                }}
              >
                仿
              </div>
            )}
          </div>
        </div>

        {/* 菜单 */}
        <Menu
          mode="inline"
          selectedKeys={[getSelectedKey()]}
          style={{
            height: 'calc(100% - 128px)',
            borderRight: 0,
            padding: '8px 0',
          }}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
        />

        {/* 底部信息 */}
        <div
          style={{
            height: 64,
            borderTop: '1px solid #f0f0f0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: collapsed ? '0 8px' : '0 16px',
          }}
        >
          {!collapsed ? (
            <Text type="secondary" style={{ fontSize: 12 }}>
              版本 1.0.0
            </Text>
          ) : (
            <Tooltip title="版本 1.0.0">
              <Text type="secondary" style={{ fontSize: 12 }}>v1</Text>
            </Tooltip>
          )}
        </div>
      </Sider>

      <Layout style={{ marginLeft: collapsed ? 80 : 220, transition: 'margin-left 0.2s' }}>
        {/* 顶部栏 */}
        <Header
          style={{
            background: '#fff',
            padding: '0 24px',
            borderBottom: '1px solid #f0f0f0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            position: 'sticky',
            top: 0,
            zIndex: 99,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <Button
              type="text"
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setCollapsed(!collapsed)}
              style={{ fontSize: 16, width: 40, height: 40 }}
            />
            <Title level={4} style={{ margin: 0 }}>
              {getPageTitle()}
            </Title>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Tooltip title="查看源码">
              <Button
                type="text"
                icon={<GithubOutlined />}
                onClick={() => window.open('https://github.com/xiangli-sophgo/CrossRing', '_blank')}
              />
            </Tooltip>
          </div>
        </Header>

        {/* 内容区 */}
        <Content
          style={{
            margin: 24,
            padding: 24,
            background: '#fff',
            borderRadius: 12,
            minHeight: 'calc(100vh - 64px - 48px)',
            boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.03), 0 1px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px 0 rgba(0, 0, 0, 0.02)',
          }}
        >
          {children}
        </Content>
      </Layout>
    </Layout>
  )
}

// 应用主组件
const App: React.FC = () => {
  return (
    <BrowserRouter>
      <MainLayout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/traffic" element={<TrafficConfig />} />
          <Route path="/simulation" element={<Simulation />} />
          <Route path="/experiments" element={<ExperimentList />} />
          <Route path="/experiments/:id" element={<ExperimentDetail />} />
          <Route path="/analysis" element={<Analysis />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </MainLayout>
    </BrowserRouter>
  )
}

export default App
