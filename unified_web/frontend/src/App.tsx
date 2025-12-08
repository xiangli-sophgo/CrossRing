import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Layout, Menu, Typography } from 'antd'
import {
  DashboardOutlined,
  ThunderboltOutlined,
  ExperimentOutlined,
  BarChartOutlined,
  SettingOutlined,
} from '@ant-design/icons'
import { useNavigate, useLocation } from 'react-router-dom'

// 页面组件
import Dashboard from './pages/Dashboard'
import TrafficConfig from './pages/Traffic'
import Simulation from './pages/Simulation'
import ExperimentList from './pages/Experiments/ExperimentList'
import ExperimentDetail from './pages/Experiments/ExperimentDetail'

const { Header, Sider, Content } = Layout
const { Title } = Typography

// 侧边栏菜单项
const menuItems = [
  {
    key: '/',
    icon: <DashboardOutlined />,
    label: '仪表盘',
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
    label: '实验管理',
  },
]

// 主布局组件
const MainLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const navigate = useNavigate()
  const location = useLocation()

  // 获取当前选中的菜单项
  const getSelectedKey = () => {
    const path = location.pathname
    if (path === '/') return '/'
    if (path.startsWith('/traffic')) return '/traffic'
    if (path.startsWith('/simulation')) return '/simulation'
    if (path.startsWith('/experiments')) return '/experiments'
    return '/'
  }

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        width={200}
        style={{
          background: '#fff',
          borderRight: '1px solid #f0f0f0',
        }}
      >
        <div
          style={{
            height: 64,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            borderBottom: '1px solid #f0f0f0',
          }}
        >
          <Title level={4} style={{ margin: 0, color: '#1890ff' }}>
            CrossRing
          </Title>
        </div>
        <Menu
          mode="inline"
          selectedKeys={[getSelectedKey()]}
          style={{ height: 'calc(100% - 64px)', borderRight: 0 }}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
        />
      </Sider>
      <Layout>
        <Header
          style={{
            background: '#fff',
            padding: '0 24px',
            borderBottom: '1px solid #f0f0f0',
            display: 'flex',
            alignItems: 'center',
          }}
        >
          <Title level={4} style={{ margin: 0 }}>
            一体化仿真平台
          </Title>
        </Header>
        <Content
          style={{
            margin: '24px',
            padding: '24px',
            background: '#fff',
            borderRadius: '8px',
            minHeight: 280,
            overflow: 'auto',
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
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </MainLayout>
    </BrowserRouter>
  )
}

export default App
