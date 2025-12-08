"""
CrossRing 一体化仿真平台 - FastAPI应用入口
合并 tool_web 和 result_db_web 的功能
"""

import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 导入配置
from app.config import (
    BASE_DIR,
    FRONTEND_DIST_DIR,
    CORS_ORIGINS,
    API_PREFIX,
    API_PORT,
    ensure_dirs,
)

# 将CrossRing项目根目录添加到Python路径,以便import src模块
CROSSRING_ROOT = BASE_DIR
sys.path.insert(0, str(CROSSRING_ROOT))

# 确保必要目录存在
ensure_dirs()

# 导入API路由
# 来自 tool_web 的路由
from app.api import ip_mount, traffic_config, traffic_generate, static_bandwidth
# 来自 result_db_web 的路由
from app.api import experiments, results, analysis, export
# 新增的仿真路由
from app.api import simulation

app = FastAPI(
    title="CrossRing 一体化仿真平台",
    description="集成流量配置、仿真执行、结果管理的统一平台",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS配置 - 允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 简化配置，允许所有来源
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
        "message": "CrossRing 一体化仿真平台 API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "modules": {
            "traffic": "流量配置与生成",
            "simulation": "仿真执行",
            "experiments": "实验管理",
            "results": "结果查询",
            "analysis": "数据分析",
        }
    }


@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "service": "crossring-unified-platform"
    }


# ==================== 注册路由 ====================

# 流量配置相关 (来自 tool_web)
app.include_router(ip_mount.router, tags=["IP挂载"])
app.include_router(traffic_config.router, tags=["流量配置"])
app.include_router(traffic_generate.router, tags=["流量生成"])
app.include_router(static_bandwidth.router, tags=["带宽分析"])

# 仿真执行 (新增)
app.include_router(simulation.router, prefix=API_PREFIX, tags=["仿真执行"])

# 实验和结果管理 (来自 result_db_web)
app.include_router(experiments.router, prefix=API_PREFIX, tags=["实验管理"])
app.include_router(results.router, prefix=API_PREFIX, tags=["结果查询"])
app.include_router(analysis.router, prefix=API_PREFIX, tags=["数据分析"])
app.include_router(export.router, prefix=API_PREFIX, tags=["导出"])


# ==================== 前端静态文件服务 ====================

if FRONTEND_DIST_DIR.exists():
    # 静态资源
    assets_dir = FRONTEND_DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

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


# ==================== 启动事件 ====================

@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print("🚀 CrossRing 一体化仿真平台已启动")
    print(f"📁 项目根目录: {CROSSRING_ROOT}")
    print(f"📖 API文档: http://localhost:{API_PORT}/api/docs")
    print("=" * 60)
    print("功能模块:")
    print("  📊 流量配置: /api/ip-mount, /api/traffic")
    print("  🔬 仿真执行: /api/simulation")
    print("  📈 实验管理: /api/experiments")
    print("  📉 结果分析: /api/analysis")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    import socket

    def find_free_port(start_port=API_PORT, max_tries=10):
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

    port = find_free_port(API_PORT)

    # 打包模式下自动打开浏览器
    if getattr(sys, 'frozen', False):
        def open_browser():
            import time
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{port}")

        threading.Thread(target=open_browser, daemon=True).start()

    print(f"服务启动在端口: {port}")
    uvicorn.run(
        app if getattr(sys, 'frozen', False) else "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=not getattr(sys, 'frozen', False),
        log_level="info"
    )
