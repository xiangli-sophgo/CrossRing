# 仿真一体化平台

集成流量配置、仿真执行、结果管理的统一Web平台。

## 功能模块

- **流量配置**: IP挂载、流量生成、带宽分析 (来自 tool_web)
- **仿真执行**: KCIN/DCIN仿真、实时进度、批量执行 (新增)
- **实验管理**: 结果查询、数据分析、导出 (来自 result_db_web)

## 快速启动

### Windows
```batch
cd unified_web
start.bat
```

### Linux/macOS
```bash
cd unified_web
chmod +x start.sh
./start.sh
```

## 访问地址

| 服务 | 地址 |
|------|------|
| 前端界面 | http://localhost:3002 |
| 后端API | http://localhost:8002 |
| API文档 | http://localhost:8002/api/docs |

## 目录结构

```
unified_web/
├── backend/          # FastAPI后端
│   ├── app/
│   │   ├── api/      # API路由
│   │   ├── core/     # 仿真引擎
│   │   └── models/   # 数据模型
│   └── requirements.txt
├── frontend/         # React前端
│   ├── src/
│   │   ├── pages/    # 页面组件
│   │   ├── components/ # 通用组件
│   │   └── api/      # API客户端
│   └── package.json
├── start.sh          # Linux/macOS启动脚本
├── start.bat         # Windows启动脚本
└── README.md
```

## 技术栈

**后端**:
- FastAPI
- SQLAlchemy
- WebSocket (实时进度)

**前端**:
- React 18 + TypeScript
- Ant Design
- Vite
- Cytoscape.js (拓扑可视化)

## API端点

### 流量配置
- `/api/ip-mount` - IP挂载管理
- `/api/traffic/config` - 流量配置
- `/api/traffic/generate` - 流量生成
- `/api/traffic/bandwidth` - 带宽分析

### 仿真执行
- `POST /api/simulation/run` - 启动仿真
- `GET /api/simulation/status/{id}` - 查询状态
- `POST /api/simulation/cancel/{id}` - 取消任务
- `GET /api/simulation/history` - 历史记录
- `WS /api/simulation/ws/{id}` - 实时进度

### 实验管理
- `/api/experiments` - 实验CRUD
- `/api/results` - 结果查询
- `/api/analysis` - 数据分析
- `/api/export` - 导出功能
