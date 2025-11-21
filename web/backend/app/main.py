"""
CrossRing Web Backend - FastAPI应用入口
"""
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 将CrossRing项目根目录添加到Python路径,以便import src模块
CROSSRING_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(CROSSRING_ROOT))

# 导入API路由
from app.api import topology, ip_mount, traffic_config, traffic_generate, static_bandwidth

app = FastAPI(
    title="CrossRing Web API",
    description="CrossRing NoC流量生成工具的Web API接口",
    version="1.0.0",
    docs_url="/api/docs",  # Swagger UI
    redoc_url="/api/redoc",  # ReDoc
)

# CORS配置 - 允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React开发服务器
        "http://localhost:3001",  # Vite开发服务器(备用端口)
        "http://localhost:3002",  # Vite开发服务器(备用端口2)
        "http://localhost:5173",  # Vite默认端口
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """根路径 - 健康检查"""
    return {
        "status": "ok",
        "message": "CrossRing Web API is running",
        "version": "1.0.0",
        "docs": "/api/docs"
    }


@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "service": "crossring-web-api"
    }


@app.get("/api/hello")
async def hello_world():
    """Hello World测试端点"""
    return {
        "message": "Hello from CrossRing Web API!",
        "framework": "FastAPI",
        "python_version": sys.version
    }


# 注册路由
app.include_router(topology.router)
app.include_router(ip_mount.router)
app.include_router(traffic_config.router)
app.include_router(traffic_generate.router)
app.include_router(static_bandwidth.router)


# 启动消息
@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print("🚀 CrossRing Web API 已启动")
    print(f"📁 CrossRing根目录: {CROSSRING_ROOT}")
    print("📖 API文档: http://localhost:8000/api/docs")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
