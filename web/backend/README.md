# CrossRing Web Backend

FastAPI后端API服务

## 🚀 快速启动

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate.bat  # Windows

# 安装依赖
pip install -r requirements.txt

# 启动服务
python -m uvicorn app.main:app --reload --port 8000
```

访问 http://localhost:8000/api/docs 查看API文档

## 📦 依赖

核心依赖：
- `fastapi==0.115.0` - Web框架
- `uvicorn==0.30.6` - ASGI服务器
- `pydantic==2.9.2` - 数据验证
- `websockets==13.1` - WebSocket支持

## 📁 目录结构

```
backend/
├── app/
│   ├── main.py           # FastAPI应用入口
│   ├── api/              # API路由层
│   │   ├── topology.py   # 拓扑相关API
│   │   ├── config.py     # 配置管理API
│   │   ├── traffic.py    # 流量生成API
│   │   └── websocket.py  # WebSocket连接
│   ├── models/           # Pydantic数据模型
│   │   ├── topology.py
│   │   ├── config.py
│   │   └── traffic.py
│   └── core/             # 核心工具
│       └── deps.py       # 依赖注入
├── requirements.txt      # Python依赖
└── .env.example          # 环境变量模板
```

## 🔧 开发

### 添加新的API端点

1. 在 `app/api/` 创建新的路由文件
2. 定义API端点
3. 在 `main.py` 注册路由

示例:
```python
# app/api/example.py
from fastapi import APIRouter

router = APIRouter(prefix="/api/example", tags=["example"])

@router.get("/")
async def get_example():
    return {"message": "Hello"}

# app/main.py
from app.api import example
app.include_router(example.router)
```

### 复用CrossRing核心代码

```python
# 在 main.py 中已自动添加CrossRing项目路径
import sys
from pathlib import Path
CROSSRING_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(CROSSRING_ROOT))

# 现在可以直接import
from src.traffic_process.traffic_gene.generation_engine import GenerationEngine
from src.utils.CrossRingConfig import CrossRingConfig
```

## 🧪 测试

```bash
# 运行测试（待实现）
pytest

# 测试单个文件
pytest tests/test_api.py
```

## 📖 API文档

启动服务后访问：
- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

## 🔍 健康检查

```bash
curl http://localhost:8000/api/health
```

返回:
```json
{
  "status": "healthy",
  "service": "crossring-web-api"
}
```
