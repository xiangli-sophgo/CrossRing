import React, { useState } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AliveScope, KeepAlive } from 'react-activation'
import { Layout, Menu, Typography, Button, Tooltip } from 'antd'
import {
  DashboardOutlined,
  ThunderboltOutlined,
  ExperimentOutlined,
  SettingOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  LineChartOutlined,
} from '@ant-design/icons'
import { useNavigate, useLocation } from 'react-router-dom'
import {
  primaryColor,
  bgLayout,
  bgSider,
  bgContainer,
  bgElevated,
  borderColor,
  textColor,
  textColorSecondary,
} from './theme/colors'

// 页面组件
import Dashboard from './pages/Dashboard'
import TrafficConfig from './pages/Traffic'
import Simulation from './pages/Simulation'
import ExperimentList from './pages/Experiments/ExperimentList'
import ExperimentDetail from './pages/Experiments/ExperimentDetail'
import CompareView from './pages/Experiments/CompareView'
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
    if (path.startsWith('/experiments') || path.startsWith('/compare')) return '/experiments'
    if (path.startsWith('/analysis')) return '/analysis'
    return '/'
  }

  // 获取页面标题
  const getPageTitle = () => {
    const path = location.pathname
    if (path === '/') return '概览'
    if (path.startsWith('/traffic')) return '流量配置'
    if (path.startsWith('/simulation')) return '仿真执行'
    if (path.startsWith('/compare')) return '实验对比'
    if (path.startsWith('/experiments')) {
      if (path.includes('/experiments/')) return '实验详情'
      return '结果管理'
    }
    if (path.startsWith('/analysis')) return '结果分析'
    return '仿真平台'
  }

  // 处理菜单点击
  const handleMenuClick = ({ key }: { key: string }) => {
    // 如果点击"结果管理"且当前在分析页面，尝试返回来源页面
    if (key === '/experiments' && location.pathname === '/analysis') {
      const searchParams = new URLSearchParams(location.search)
      const fromPath = searchParams.get('from')
      if (fromPath && fromPath.startsWith('/experiments/')) {
        // 如果来源是实验详情页，返回详情页
        navigate(fromPath)
        return
      }
    }
    // 其他情况正常导航
    navigate(key)
  }

  return (
    <Layout style={{ minHeight: '100vh', background: bgLayout }}>
      <Sider
        collapsible
        collapsed={collapsed}
        onCollapse={setCollapsed}
        trigger={null}
        width={220}
        collapsedWidth={80}
        style={{
          background: bgSider,
          borderRight: `1px solid ${borderColor}`,
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
            borderBottom: `1px solid ${borderColor}`,
            padding: collapsed ? '0 8px' : '0 16px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            {/* Logo图标 */}
            <div
              style={{
                width: 34,
                height: 34,
                borderRadius: 10,
                background: `linear-gradient(135deg, ${primaryColor} 0%, #7C3AED 100%)`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 2px 6px rgba(94, 106, 210, 0.3)',
                flexShrink: 0,
              }}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                <path
                  d="M12 2L2 7L12 12L22 7L12 2Z"
                  stroke="#fff"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M2 17L12 22L22 17"
                  stroke="#fff"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M2 12L12 17L22 12"
                  stroke="#fff"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
            {!collapsed && (
              <div style={{ overflow: 'hidden' }}>
                <Title
                  level={5}
                  style={{
                    margin: 0,
                    color: textColor,
                    whiteSpace: 'nowrap',
                    fontSize: 14,
                    fontWeight: 600,
                    letterSpacing: '-0.3px',
                  }}
                >
                  Simulation Platform
                </Title>
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
            padding: '12px 8px',
            background: 'transparent',
          }}
          items={menuItems}
          onClick={handleMenuClick}
        />

        {/* 底部信息 */}
        <div
          style={{
            height: 64,
            borderTop: `1px solid ${borderColor}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: collapsed ? '0 8px' : '0 16px',
          }}
        >
          {!collapsed ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  background: '#10b981',
                  boxShadow: '0 0 6px rgba(16, 185, 129, 0.5)',
                }}
              />
              <Text style={{ fontSize: 12, color: textColorSecondary }}>
                v{__APP_VERSION__}
              </Text>
            </div>
          ) : (
            <Tooltip title={`版本 ${__APP_VERSION__}`}>
              <div
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  background: '#10b981',
                  boxShadow: '0 0 6px rgba(16, 185, 129, 0.5)',
                }}
              />
            </Tooltip>
          )}
        </div>
      </Sider>

      <Layout style={{ marginLeft: collapsed ? 80 : 220, transition: 'margin-left 0.2s', background: 'transparent' }}>
        {/* 顶部栏 */}
        <Header
          style={{
            background: bgElevated,
            padding: '0 24px',
            borderBottom: `1px solid ${borderColor}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            position: 'sticky',
            top: 0,
            zIndex: 99,
            boxShadow: '0 1px 2px rgba(0, 0, 0, 0.04)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <Button
              type="text"
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setCollapsed(!collapsed)}
              style={{ fontSize: 16, width: 40, height: 40, color: textColorSecondary }}
            />
            <Title level={4} style={{ margin: 0, color: textColor, fontWeight: 600 }}>
              {getPageTitle()}
            </Title>
          </div>
        </Header>

        {/* 内容区 */}
        <Content
          style={{
            margin: 0,
            padding: 24,
            background: bgContainer,
            minHeight: 'calc(100vh - 64px)',
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
      <AliveScope>
        <MainLayout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/traffic" element={<KeepAlive cacheKey="traffic"><TrafficConfig /></KeepAlive>} />
            <Route path="/simulation" element={<KeepAlive cacheKey="simulation"><Simulation /></KeepAlive>} />
            <Route path="/experiments" element={<ExperimentList />} />
            <Route path="/experiments/:id" element={<ExperimentDetail />} />
            <Route path="/compare" element={<CompareView />} />
            <Route path="/analysis" element={<KeepAlive cacheKey="analysis"><Analysis /></KeepAlive>} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </MainLayout>
      </AliveScope>
    </BrowserRouter>
  )
}

export default App
