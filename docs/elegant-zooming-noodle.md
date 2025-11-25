# 参数优化结果管理与可视化系统设计文档

> 文档路径建议: `docs/optimization_result_manager_design.md`

## 1. 背景与目标

### 1.1 当前问题

现有的 `optimize_fifo_exhaustive.py` 脚本存在以下限制：

- 结果保存为CSV文件，查询和筛选不便
- 无法管理多次实验，难以进行跨实验对比
- 可视化依赖matplotlib静态图表，交互性差
- 数据量大时(>10万组合)，CSV加载缓慢

### 1.2 设计目标

构建一个完整的参数优化结果管理系统：

- **方便添加**: 仿真结果实时写入数据库
- **方便记录**: 支持多实验管理、实验描述/调试信息
- **方便展示**: 交互式Web界面，支持筛选、图表、对比

### 1.3 核心需求

| 需求     | 说明                        |
| -------- | --------------------------- |
| 数据规模 | >10万参数组合               |
| 实验管理 | 支持多实验，跨实验对比      |
| 性能指标 | 主要关注带宽(BW)            |
| 部署方式 | 独立部署，不影响现有web系统 |
| 实验命名 | 用户手动输入实验名称        |

## 2. 技术选型

| 层级     | 技术               | 选型理由                                             |
| -------- | ------------------ | ---------------------------------------------------- |
| 数据库   | SQLite             | 轻量级、无需安装服务、单文件便于备份、支持百万级记录 |
| ORM      | SQLAlchemy         | Python标准ORM、支持多数据库切换、查询构建方便        |
| 后端     | FastAPI            | 项目已有使用经验、性能优秀、自动生成API文档          |
| 前端     | React + TypeScript | 项目已有前端架构、组件化开发                         |
| UI组件   | Ant Design         | 项目已有依赖、企业级组件库                           |
| 图表     | ECharts            | 项目已有依赖、大数据量渲染性能好                     |
| 状态管理 | Zustand            | 项目已有使用、轻量级                                 |

## 3. 系统架构

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户界面层                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ 实验列表页   │  │ 实验详情页   │  │  对比视图   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│         React + TypeScript + Ant Design + ECharts               │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP/REST
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        后端API层                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ 实验管理API  │  │ 结果查询API  │  │  分析API    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                      FastAPI                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        数据库层                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  src/database/                                          │    │
│  │  ├── models.py      # ORM模型                           │    │
│  │  ├── database.py    # 数据库操作                        │    │
│  │  └── manager.py     # 业务逻辑                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                      SQLAlchemy + SQLite                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              ../Result/Database/optimization.db                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 目录结构

```
CrossRing/
├── src/
│   └── database/                      # 数据库模块 (新建)
│       ├── __init__.py
│       ├── models.py                  # SQLAlchemy ORM模型
│       ├── database.py                # 数据库连接和CRUD操作
│       └── manager.py                 # 业务逻辑层(导入/统计/分析)
│
├── optimization_web/                  # 独立Web应用 (新建)
│   ├── backend/
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py               # FastAPI入口
│   │   │   ├── config.py             # 配置
│   │   │   └── api/
│   │   │       ├── __init__.py
│   │   │       ├── experiments.py    # 实验管理API
│   │   │       ├── results.py        # 结果查询API
│   │   │       └── analysis.py       # 分析API
│   │   └── requirements.txt
│   │
│   ├── frontend/
│   │   ├── src/
│   │   │   ├── App.tsx
│   │   │   ├── main.tsx
│   │   │   ├── pages/
│   │   │   │   ├── ExperimentList.tsx
│   │   │   │   ├── ExperimentDetail.tsx
│   │   │   │   └── CompareView.tsx
│   │   │   ├── components/
│   │   │   │   ├── ParameterFilter.tsx
│   │   │   │   ├── PerformanceChart.tsx
│   │   │   │   └── ResultTable.tsx
│   │   │   └── stores/
│   │   │       └── experimentStore.ts
│   │   ├── package.json
│   │   └── vite.config.ts
│   │
│   ├── start.bat                     # Windows一键启动
│   ├── start.sh                      # Linux一键启动
│   └── README.md
│
├── scripts/
│   └── optimize_fifo_exhaustive.py   # 修改: 集成数据库写入
│
└── ../Result/Database/               # 数据库存储位置
    └── optimization.db
```

## 4. 数据库设计

### 4.1 ER图

```
┌─────────────────────┐       1:N      ┌─────────────────────────┐
│    experiments      │───────────────▶│  optimization_results   │
├─────────────────────┤                ├─────────────────────────┤
│ id (PK)             │                │ id (PK)                 │
│ name (UNIQUE)       │                │ experiment_id (FK)      │
│ created_at          │                │ RB_IN_FIFO_DEPTH        │
│ description         │                │ EQ_IN_FIFO_DEPTH        │
│ config_path         │                │ TL_Etag_T1_UE_MAX       │
│ topo_type           │                │ TL_Etag_T2_UE_MAX       │
│ traffic_files       │                │ TR_Etag_T2_UE_MAX       │
│ traffic_weights     │                │ ... (其他参数)          │
│ simulation_time     │                │ performance             │
│ status              │                │ created_at              │
│ total_combinations  │                └─────────────────────────┘
│ completed_count     │
│ best_performance    │
└─────────────────────┘
```

### 4.2 SQL Schema

```sql
-- 实验元数据表
CREATE TABLE experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,                    -- 用户输入的实验名称
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT,                             -- 实验描述/调试信息

    -- 仿真配置
    config_path TEXT,                             -- 配置文件路径
    topo_type TEXT,                               -- 拓扑类型 (如 "5x4")
    traffic_files TEXT,                           -- traffic文件列表 (JSON数组)
    traffic_weights TEXT,                         -- traffic权重 (JSON数组)
    simulation_time INTEGER,                      -- 仿真时间
    n_repeats INTEGER DEFAULT 1,                  -- 重复次数
    n_jobs INTEGER,                               -- 并行作业数

    -- 状态追踪
    status TEXT DEFAULT 'running',                -- running/completed/failed/interrupted
    total_combinations INTEGER,                   -- 总参数组合数
    completed_combinations INTEGER DEFAULT 0,     -- 已完成组合数
    best_performance REAL,                        -- 最佳性能 (GB/s)

    -- 元数据
    git_commit TEXT,                              -- Git commit hash (可选)
    notes TEXT                                    -- 额外备注
);

-- 优化结果表 (核心数据表)
CREATE TABLE optimization_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- ========== FIFO参数 ==========
    RB_IN_FIFO_DEPTH INTEGER,
    EQ_IN_FIFO_DEPTH INTEGER,
    IQ_CH_FIFO_DEPTH INTEGER,
    EQ_CH_FIFO_DEPTH INTEGER,
    IQ_OUT_FIFO_DEPTH_HORIZONTAL INTEGER,
    IQ_OUT_FIFO_DEPTH_VERTICAL INTEGER,
    IQ_OUT_FIFO_DEPTH_EQ INTEGER,
    RB_OUT_FIFO_DEPTH INTEGER,

    -- ========== ETag参数 ==========
    TL_Etag_T1_UE_MAX INTEGER,
    TL_Etag_T2_UE_MAX INTEGER,
    TR_Etag_T2_UE_MAX INTEGER,
    TU_Etag_T1_UE_MAX INTEGER,
    TU_Etag_T2_UE_MAX INTEGER,
    TD_Etag_T2_UE_MAX INTEGER,

    -- ========== 性能指标 ==========
    performance REAL NOT NULL,                    -- 主要性能指标 (加权带宽 GB/s)

    -- ========== 扩展字段 ==========
    extra_data TEXT,                              -- JSON格式的额外数据
    error TEXT,                                   -- 错误信息 (如果有)

    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);

-- ========== 索引优化 (针对>10万数据量) ==========
CREATE INDEX idx_results_experiment ON optimization_results(experiment_id);
CREATE INDEX idx_results_performance ON optimization_results(performance DESC);
CREATE INDEX idx_results_rb_in ON optimization_results(RB_IN_FIFO_DEPTH);
CREATE INDEX idx_results_tl_t1 ON optimization_results(TL_Etag_T1_UE_MAX);
CREATE INDEX idx_experiments_status ON experiments(status);
CREATE INDEX idx_experiments_created ON experiments(created_at DESC);
```

### 4.3 设计说明

1. **参数列设计**: 虽然可以使用JSON存储动态参数，但为了查询性能，常用参数作为独立列存储
2. **级联删除**: 删除实验时自动删除所有相关结果
3. **索引策略**: 针对高频查询字段建立索引，特别是性能排序和参数筛选

## API设计

### 实验管理

```
GET  /api/experiments                  # 获取实验列表
POST /api/experiments                  # 创建实验
GET  /api/experiments/{id}             # 获取实验详情
PUT  /api/experiments/{id}             # 更新实验(描述等)
DELETE /api/experiments/{id}           # 删除实验

POST /api/experiments/import           # 从CSV导入
```

### 结果查询

```
GET /api/experiments/{id}/results      # 获取结果(分页)
    ?page=1&page_size=100
    &sort_by=performance&order=desc
    &filters={param_name: [min, max]}

GET /api/experiments/{id}/stats        # 获取统计信息
GET /api/experiments/{id}/best         # 获取最佳配置
GET /api/experiments/{id}/distribution # 获取性能分布
```

### 分析

```
GET /api/experiments/{id}/sensitivity  # 参数敏感性分析
POST /api/compare                      # 多实验对比
    body: {experiment_ids: [1, 2, 3]}
```

## 前端页面设计

### 1. 实验列表页 (`/`)

```
+----------------------------------------------------------+
|  参数优化结果管理系统                                      |
+----------------------------------------------------------+
|  [导入CSV] [刷新]                         [搜索...]       |
+----------------------------------------------------------+
| 名称        | 创建时间      | 状态   | 组合数  | 最佳性能 |
|-------------|--------------|--------|---------|----------|
| exp_fifo_v1 | 2024-11-25   | 完成   | 120,000 | 156.3 GB/s|
| exp_etag_v2 | 2024-11-24   | 运行中 | 80,000  | 142.1 GB/s|
+----------------------------------------------------------+
```

### 2. 实验详情页 (`/experiments/{id}`)

```
+----------------------------------------------------------+
|  exp_fifo_v1                          [编辑描述] [导出]   |
|  创建: 2024-11-25 | 组合数: 120,000 | 最佳: 156.3 GB/s   |
+----------------------------------------------------------+
|  [Tab: 参数分析] [性能分布] [最优配置] [原始数据]         |
+----------------------------------------------------------+
|                                                          |
|  +---------------------+  +----------------------------+ |
|  | 参数筛选器          |  | 性能分布图 (ECharts)       | |
|  | RB_IN: [2] - [20]   |  |         ___                | |
|  | TL_T1: [2] - [19]   |  |        /   \               | |
|  | ...                 |  |       /     \              | |
|  | [应用筛选]          |  |  ____/       \____         | |
|  +---------------------+  +----------------------------+ |
|                                                          |
|  +----------------------------------------------------+ |
|  | 数据表格 (虚拟滚动)                      [导出CSV]  | |
|  | RB_IN | TL_T1 | TL_T2 | ... | Performance          | |
|  |   16  |  15   |   8   | ... | 156.3 GB/s           | |
|  |   14  |  14   |   7   | ... | 155.8 GB/s           | |
|  +----------------------------------------------------+ |
+----------------------------------------------------------+
```

### 3. 对比视图 (`/compare`)

```
+----------------------------------------------------------+
|  实验对比                                                 |
+----------------------------------------------------------+
|  选择实验: [exp_fifo_v1 ✓] [exp_etag_v2 ✓] [exp_v3 □]    |
+----------------------------------------------------------+
|                                                          |
|  +----------------------------------------------------+ |
|  | 性能分布对比                                        | |
|  |      ___                                           | |
|  |     /   \     exp_fifo_v1 (蓝)                     | |
|  |    /  _  \    exp_etag_v2 (橙)                     | |
|  |   / _/ \_ \                                        | |
|  +----------------------------------------------------+ |
|                                                          |
|  +----------------------------------------------------+ |
|  | 最优配置对比                                        | |
|  | 参数        | exp_fifo_v1 | exp_etag_v2 | 差异     | |
|  | RB_IN       |     16      |     14      |   +2     | |
|  | Performance | 156.3 GB/s  | 148.2 GB/s  | +8.1     | |
|  +----------------------------------------------------+ |
+----------------------------------------------------------+
```

## 实现步骤

### Phase 1: 数据库层 (优先级: 高)

1. **创建 `src/database/` 模块**

   - `models.py`: 定义SQLAlchemy ORM模型
   - `database.py`: 数据库连接、表创建、CRUD操作
   - `manager.py`: 业务逻辑封装（CSV导入、统计分析等）
2. **修改 `optimize_fifo_exhaustive.py`**

   - 启动时要求用户输入实验名称
   - 创建实验记录
   - 仿真结果实时写入数据库（同时保留CSV备份）
   - 完成时更新实验状态

### Phase 2: 后端API (优先级: 高)

3. **创建 `optimization_web/backend/`**
   - 复制现有web/backend的结构作为模板
   - 实现实验管理API
   - 实现结果查询API（带分页和筛选）
   - 实现统计分析API

### Phase 3: 前端界面 (优先级: 中)

4. **创建 `optimization_web/frontend/`**
   - 基于现有前端模板创建
   - 实现实验列表页
   - 实现实验详情页（参数筛选、图表、表格）
   - 实现CSV导入功能

### Phase 4: 高级功能 (优先级: 低)

5. **实现对比视图**
6. **实现参数敏感性分析图表**
7. **性能优化（大数据量场景）**

## 需要修改的文件

| 文件                                    | 操作 | 说明              |
| --------------------------------------- | ---- | ----------------- |
| `src/database/__init__.py`            | 新建 | 模块初始化        |
| `src/database/models.py`              | 新建 | ORM模型           |
| `src/database/database.py`            | 新建 | 数据库操作        |
| `src/database/manager.py`             | 新建 | 业务逻辑          |
| `scripts/optimize_fifo_exhaustive.py` | 修改 | 集成数据库写入    |
| `optimization_web/`                   | 新建 | 完整的Web应用目录 |

## 关键依赖

后端新增:

```
sqlalchemy>=2.0.0
```

前端复用现有依赖:

```
react, antd, echarts, axios, zustand
```

## 启动方式

```bash
# 后端
cd optimization_web/backend
python -m uvicorn app.main:app --port 8001

# 前端
cd optimization_web/frontend
pnpm dev

# 或一键启动
optimization_web/start.bat
```

## 注意事项

1. **数据库路径**: `../Result/Database/optimization.db`，需要确保目录存在
2. **实验名称**: 用户手动输入，用于区分不同实验
3. **CSV兼容**: 保留原有CSV输出，数据库作为增强功能
4. **大数据量**: 使用虚拟滚动表格，后端分页查询，避免一次加载全部数据
