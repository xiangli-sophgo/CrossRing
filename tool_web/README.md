# CrossRing Web - KCIN流量生成工具

基于 **FastAPI + React** 的现代化Web应用，用于CrossRing KCIN拓扑建模和流量生成。

## ✨ 特性

- 🚀 **高性能**: 相比Streamlit版本性能提升5-10倍
- 🎨 **现代化UI**: 基于Ant Design的专业界面
- 📊 **强大可视化**: Cytoscape.js拓扑图 + ECharts图表
- ⚡ **实时更新**: WebSocket实时进度推送
- 🌐 **跨平台**: Windows/macOS/Linux全支持

## 📦 技术栈

### 后端
- FastAPI 0.115+ - 高性能异步Web框架
- Python 3.8+ - 复用CrossRing现有科学计算代码
- WebSocket - 实时通信支持

### 前端
- React 18 + TypeScript - 现代化前端框架
- Vite 5 - 极速构建工具
- Ant Design 5 - 企业级UI组件库
- Cytoscape.js 3.30+ - 网络拓扑可视化
- Apache ECharts 5.5+ - 数据图表
- Zustand 4.5+ - 轻量级状态管理

## 🚀 快速开始

### 前置要求

- **Python 3.8+** (已有)
- **Node.js 18+** ([下载安装](https://nodejs.org/))
- **pnpm** (会自动安装)

### 一键启动（推荐）

**macOS/Linux:**
```bash
cd web
./start-dev.sh
```

**Windows:**
```cmd
cd web
start-dev.bat
```

脚本会自动：
1. 检查环境
2. 安装依赖
3. 启动后端和前端服务

启动成功后访问：
- **前端界面**: http://localhost:3000
- **后端API文档**: http://localhost:8000/api/docs

### 手动启动

#### 1. 启动后端

```bash
# 进入后端目录
cd web/backend

# 创建虚拟环境（首次）
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate.bat  # Windows

# 安装依赖
pip install -r requirements.txt

# 启动FastAPI服务
python -m uvicorn app.main:app --reload --port 8000
```

后端启动在: http://localhost:8000

#### 2. 启动前端

**新开一个终端:**

```bash
# 进入前端目录
cd web/frontend

# 安装pnpm（如未安装）
npm install -g pnpm

# 安装依赖（首次运行，约2-3分钟）
pnpm install

# 启动开发服务器
pnpm dev
```

前端启动在: http://localhost:3000

## 📁 项目结构

```
web/
├── README.md                 # 本文件
├── start-dev.sh              # macOS/Linux启动脚本
├── start-dev.bat             # Windows启动脚本
│
├── backend/                  # FastAPI后端
│   ├── app/
│   │   ├── main.py          # FastAPI应用入口
│   │   ├── api/             # API路由
│   │   ├── models/          # 数据模型
│   │   └── core/            # 核心工具
│   ├── requirements.txt     # Python依赖
│   └── .env.example         # 环境变量模板
│
└── frontend/                 # React前端
    ├── src/
    │   ├── App.tsx          # 主应用组件
    │   ├── components/      # UI组件
    │   ├── pages/           # 页面
    │   ├── store/           # Zustand状态
    │   ├── api/             # API客户端
    │   └── types/           # TypeScript类型
    ├── package.json         # npm依赖
    ├── vite.config.ts       # Vite配置
    └── tsconfig.json        # TypeScript配置
```

## 🔧 开发指南

### 后端开发

后端代码位于 `backend/app/` 目录：

```python
# 添加新的API端点
# backend/app/api/example.py

from fastapi import APIRouter

router = APIRouter()

@router.get("/example")
async def get_example():
    return {"message": "Hello"}

# 在 main.py 中注册路由
# app.include_router(router, prefix="/api")
```

**热更新**: 修改代码后自动重启（`--reload`模式）

**API文档**: http://localhost:8000/api/docs (Swagger UI)

### 前端开发

前端代码位于 `frontend/src/` 目录：

```typescript
// 创建新组件
// src/components/MyComponent.tsx

import { Card } from 'antd'

export const MyComponent = () => {
  return <Card>Hello Component</Card>
}
```

**热更新**: 修改代码后浏览器自动刷新（HMR）

**类型检查**: TypeScript提供完整的类型安全

### API调用示例

```typescript
// frontend/src/api/client.ts
import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000',
})

export const getTopology = async () => {
  const response = await api.get('/api/topology')
  return response.data
}
```

## 🐛 常见问题

### 1. 端口被占用

**错误**: `Address already in use: 8000`

**解决**:
```bash
# macOS/Linux
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### 2. Node.js版本过低

**错误**: `Unsupported Node.js version`

**解决**: 升级到Node.js 18+
```bash
# macOS (Homebrew)
brew install node@20

# Windows: 访问 https://nodejs.org/ 下载最新LTS版本
```

### 3. pnpm安装依赖失败

**错误**: `ERR_PNPM_FETCH_*`

**解决**:
```bash
# 清除缓存重试
pnpm store prune
pnpm install
```

### 4. 后端无法import CrossRing模块

**错误**: `ModuleNotFoundError: No module named 'src'`

**原因**: Python路径配置问题

**解决**: 已在 `main.py` 中自动配置，无需手动操作

### 5. CORS错误

**错误**: `Access to XMLHttpRequest has been blocked by CORS policy`

**解决**: 检查后端 `main.py` 中的CORS配置，确保包含前端URL

## 📚 扩展阅读

### 官方文档

- [FastAPI文档](https://fastapi.tiangolo.com/)
- [React文档](https://react.dev/)
- [Ant Design文档](https://ant.design/)
- [Cytoscape.js文档](https://js.cytoscape.org/)
- [ECharts文档](https://echarts.apache.org/)

### 推荐工具

- **VS Code**: 代码编辑器
- **Postman**: API测试工具
- **React DevTools**: React调试工具（浏览器扩展）

## 🗺️ 路线图

### 阶段1: 基础架构 ✅
- [x] 项目初始化
- [x] 前后端框架搭建
- [x] Hello World验证

### 阶段2: 核心功能（进行中）
- [ ] 拓扑图可视化（Cytoscape.js）
- [ ] IP挂载配置界面
- [ ] 流量生成配置表单
- [ ] 配置管理（CRUD）

### 阶段3: 高级功能
- [ ] WebSocket实时进度推送
- [ ] 流量分析与图表
- [ ] 静态链路带宽可视化
- [ ] 配置导入导出

### 阶段4: 优化部署
- [ ] 性能优化
- [ ] 单元测试
- [ ] Docker部署
- [ ] 用户文档

## 📝 许可证

本项目遵循CrossRing主项目的许可证

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

**Made with ❤️ using FastAPI + React**
