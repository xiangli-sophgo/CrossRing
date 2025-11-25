"""
仿真结果数据库 - FastAPI入口
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import CORS_ORIGINS, API_PREFIX
from .api import experiments, results, analysis

# 创建FastAPI应用
app = FastAPI(
    title="仿真结果数据库",
    description="仿真实验结果的管理、查询和分析API",
    version="1.0.0",
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(experiments.router, prefix=API_PREFIX, tags=["实验管理"])
app.include_router(results.router, prefix=API_PREFIX, tags=["结果查询"])
app.include_router(analysis.router, prefix=API_PREFIX, tags=["分析"])


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "仿真结果数据库API",
        "docs": "/docs",
        "version": "1.0.0",
    }


@app.get(f"{API_PREFIX}/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
