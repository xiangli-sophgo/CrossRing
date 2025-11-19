# CrossRing Web Frontend

React + TypeScript + Vite 前端应用

## 🚀 快速启动

```bash
# 安装pnpm（如未安装）
npm install -g pnpm

# 安装依赖
pnpm install

# 启动开发服务器
pnpm dev
```

访问 http://localhost:3000

## 📦 依赖

核心依赖：
- `react@18.3.1` - UI框架
- `typescript@5.6.3` - 类型系统
- `vite@5.4.10` - 构建工具
- `antd@5.21.4` - UI组件库
- `cytoscape@3.30.2` - 拓扑可视化
- `echarts@5.5.1` - 数据图表
- `zustand@4.5.5` - 状态管理
- `axios@1.7.7` - HTTP客户端

## 📁 目录结构

```
frontend/src/
├── main.tsx              # React应用入口
├── App.tsx               # 主应用组件
├── components/           # UI组件
│   ├── topology/        # 拓扑相关组件
│   ├── config/          # 配置相关组件
│   ├── traffic/         # 流量相关组件
│   └── common/          # 通用组件
├── pages/               # 页面组件
├── store/               # Zustand状态管理
├── api/                 # API客户端
├── types/               # TypeScript类型定义
├── utils/               # 工具函数
└── styles/              # 样式文件
```

## 🔧 开发

### 创建新组件

```typescript
// src/components/example/MyComponent.tsx
import { FC } from 'react'
import { Card } from 'antd'

interface MyComponentProps {
  title: string
}

export const MyComponent: FC<MyComponentProps> = ({ title }) => {
  return (
    <Card title={title}>
      <p>Hello Component</p>
    </Card>
  )
}
```

### 状态管理（Zustand）

```typescript
// src/store/exampleStore.ts
import { create } from 'zustand'

interface ExampleState {
  count: number
  increment: () => void
}

export const useExampleStore = create<ExampleState>((set) => ({
  count: 0,
  increment: () => set((state) => ({ count: state.count + 1 })),
}))

// 在组件中使用
import { useExampleStore } from '@/store/exampleStore'

function MyComponent() {
  const { count, increment } = useExampleStore()
  return <button onClick={increment}>Count: {count}</button>
}
```

### API调用

```typescript
// src/api/topology.ts
import axios from 'axios'

const client = axios.create({
  baseURL: 'http://localhost:8000',
})

export const getTopology = async (type: string) => {
  const response = await client.get(`/api/topology/${type}`)
  return response.data
}

// 在组件中使用
import { useEffect, useState } from 'react'
import { getTopology } from '@/api/topology'

function TopologyView() {
  const [data, setData] = useState(null)

  useEffect(() => {
    getTopology('5x4').then(setData)
  }, [])

  return <div>{JSON.stringify(data)}</div>
}
```

## 🎨 样式

使用Ant Design主题：

```typescript
// 在App.tsx中配置主题
import { ConfigProvider } from 'antd'

const theme = {
  token: {
    colorPrimary: '#1890ff',
  },
}

function App() {
  return (
    <ConfigProvider theme={theme}>
      {/* 应用内容 */}
    </ConfigProvider>
  )
}
```

## 📝 脚本命令

```bash
# 开发模式（热更新）
pnpm dev

# 类型检查
pnpm build  # 会先执行tsc检查类型

# 预览生产构建
pnpm build
pnpm preview

# 代码检查
pnpm lint
```

## 🧪 测试

```bash
# 运行测试（待实现）
pnpm test

# 测试覆盖率
pnpm test:coverage
```

## 🏗️ 构建

```bash
# 生产构建
pnpm build

# 输出目录: dist/
```

构建产物：
- `dist/index.html` - 入口HTML
- `dist/assets/` - JS/CSS/图片等资源

## 🔍 调试

### React DevTools

安装浏览器扩展：
- [Chrome扩展](https://chrome.google.com/webstore/detail/react-developer-tools/fmkadmapgofadopljbjfkapdkoienihi)
- [Firefox扩展](https://addons.mozilla.org/en-US/firefox/addon/react-devtools/)

### Vite调试

开发模式下自动启用Source Map，可在浏览器中直接调试TypeScript源码。

## 📖 推荐阅读

- [React官方文档](https://react.dev/)
- [TypeScript手册](https://www.typescriptlang.org/docs/)
- [Vite指南](https://vitejs.dev/guide/)
- [Ant Design组件](https://ant.design/components/overview/)
- [Zustand文档](https://github.com/pmndrs/zustand)
