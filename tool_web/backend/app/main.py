"""
CrossRing Web Backend - FastAPI应用入口
"""
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 导入配置
from app.config import BASE_DIR, FRONTEND_DIST_DIR, ensure_dirs

# 将CrossRing项目根目录添加到Python路径,以便import src模块
CROSSRING_ROOT = BASE_DIR
sys.path.insert(0, str(CROSSRING_ROOT))

# 确保必要目录存在
ensure_dirs()

# 导入API路由
from app.api import ip_mount, traffic_config, traffic_generate, static_bandwidth

app = FastAPI(
    title="CrossRing Web API",
    description="CrossRing KCIN流量生成工具的Web API接口",
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
    """根路径 - 返回前端页面或健康检查"""
    if FRONTEND_DIST_DIR.exists() and (FRONTEND_DIST_DIR / "index.html").exists():
        return FileResponse(FRONTEND_DIST_DIR / "index.html")
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
app.include_router(ip_mount.router)
app.include_router(traffic_config.router)
app.include_router(traffic_generate.router)
app.include_router(static_bandwidth.router)

# 挂载前端静态文件 (放在API路由之后)
if FRONTEND_DIST_DIR.exists():
    # 静态资源
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")), name="assets")

    # 前端入口 - 所有非API路由返回index.html
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """服务前端页面"""
        # API路由不处理
        if full_path.startswith("api/"):
            return {"error": "Not found"}

        # 尝试返回静态文件
        file_path = FRONTEND_DIST_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)

        # 其他都返回index.html (SPA路由)
        return FileResponse(FRONTEND_DIST_DIR / "index.html")


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
    import webbrowser
    import threading
    import socket

    def find_free_port(start_port=8000, max_tries=10):
        """查找可用端口"""
        for i in range(max_tries):
            port = start_port + i
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', port))
                    return port
            except OSError:
                continue
        return start_port

    # 打包模式下自动打开浏览器
    if getattr(sys, 'frozen', False):
        port = find_free_port(8000)

        def open_browser():
            import time
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{port}")

        threading.Thread(target=open_browser, daemon=True).start()

        print(f"服务启动在端口: {port}")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
    else:
        # 开发模式
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
