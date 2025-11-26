"""
仿真结果数据库 - FastAPI入口
"""

import sys
from pathlib import Path

# 开发模式下添加项目根目录到路径
if not getattr(sys, 'frozen', False):
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .config import CORS_ORIGINS, API_PREFIX, FRONTEND_DIST_DIR, BASE_DIR

from .api import experiments, results, analysis, export

# 创建FastAPI应用
app = FastAPI(
    title="仿真结果数据库",
    description="仿真实验结果的管理、查询和分析API",
    version="1.0.0",
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 打包后允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(experiments.router, prefix=API_PREFIX, tags=["实验管理"])
app.include_router(results.router, prefix=API_PREFIX, tags=["结果查询"])
app.include_router(analysis.router, prefix=API_PREFIX, tags=["分析"])
app.include_router(export.router, prefix=API_PREFIX, tags=["导出"])


@app.get(f"{API_PREFIX}/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


# 根路径
@app.get("/")
async def root():
    """根路径 - 返回前端页面或API信息"""
    if FRONTEND_DIST_DIR.exists() and (FRONTEND_DIST_DIR / "index.html").exists():
        return FileResponse(FRONTEND_DIST_DIR / "index.html")
    return {
        "status": "ok",
        "message": "仿真结果数据库API",
        "version": "1.0.0",
        "docs": "/docs"
    }


# 挂载前端静态文件 (放在API路由之后)
if FRONTEND_DIST_DIR.exists():
    # 静态资源
    assets_dir = FRONTEND_DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    # 前端入口 - 所有非API路由返回index.html
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """服务SPA前端"""
        # API路由不处理
        if full_path.startswith("api/"):
            return {"error": "Not found"}

        # 尝试返回静态文件
        file_path = FRONTEND_DIST_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)

        # 其他都返回index.html (SPA路由)
        return FileResponse(FRONTEND_DIST_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    import socket

    def find_free_port(start_port=8001, max_tries=10):
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

    port = find_free_port(8001)

    # 打包模式下自动打开浏览器
    if getattr(sys, 'frozen', False):
        def open_browser():
            import time
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{port}")

        threading.Thread(target=open_browser, daemon=True).start()

    print(f"服务启动在端口: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
