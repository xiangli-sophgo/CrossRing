# 仿真结果数据库

交互式Web界面，用于管理和分析仿真实验结果。

## 功能特性

- **实验管理**: 查看、创建、删除实验
- **CSV导入**: 从历史CSV文件导入实验数据
- **结果查询**: 分页浏览、参数筛选、排序
- **性能分析**: 性能分布图、参数敏感性分析
- **实验对比**: 多实验对比视图

## 快速启动

### Linux/macOS

```bash
cd result_db_web
chmod +x start.sh
./start.sh
```

### Windows

```batch
cd result_db_web
start.bat
```

### 手动启动

**后端:**
```bash
cd result_db_web/backend
pip install -r requirements.txt
PYTHONPATH=../.. python -m uvicorn app.main:app --port 8001 --reload
```

**前端:**
```bash
cd result_db_web/frontend
pnpm install  # 或 npm install
pnpm dev      # 或 npm run dev
```

## 访问地址

- 前端界面: http://localhost:3000
- 后端API: http://localhost:8001
- API文档: http://localhost:8001/docs

## 目录结构

```
result_db_web/
├── backend/                    # FastAPI后端
│   ├── app/
│   │   ├── main.py            # 应用入口
│   │   ├── config.py          # 配置
│   │   └── api/               # API路由
│   │       ├── experiments.py # 实验管理
│   │       ├── results.py     # 结果查询
│   │       └── analysis.py    # 分析功能
│   └── requirements.txt
│
├── frontend/                   # React前端
│   ├── src/
│   │   ├── pages/             # 页面组件
│   │   ├── components/        # 通用组件
│   │   ├── api/               # API客户端
│   │   ├── stores/            # 状态管理
│   │   └── types/             # 类型定义
│   └── package.json
│
├── start.sh                    # Linux启动脚本
├── start.bat                   # Windows启动脚本
└── README.md
```

## API端点

### 实验管理
- `GET /api/experiments` - 获取实验列表
- `POST /api/experiments` - 创建实验
- `GET /api/experiments/{id}` - 获取实验详情
- `DELETE /api/experiments/{id}` - 删除实验
- `POST /api/experiments/import` - CSV导入

### 结果查询
- `GET /api/experiments/{id}/results` - 分页获取结果
- `GET /api/experiments/{id}/best` - 获取最佳配置
- `GET /api/experiments/{id}/stats` - 获取统计信息
- `GET /api/experiments/{id}/distribution` - 获取性能分布

### 分析功能
- `GET /api/experiments/{id}/sensitivity/{param}` - 参数敏感性
- `POST /api/compare` - 多实验对比

## 数据库

使用SQLite存储，位置: `../Result/Database/optimization.db`

## 与优化脚本集成

运行 `scripts/optimize_fifo_exhaustive.py` 时会自动将结果写入数据库：

```bash
cd scripts
python optimize_fifo_exhaustive.py
# 输入实验名称后，结果会同时保存到CSV和数据库
```
