"""
仿真结果数据库 - 配置
"""

import sys
from pathlib import Path


def get_base_dir() -> Path:
    """
    获取应用根目录

    - 开发模式: CrossRing 项目根目录
    - 打包模式: exe 所在目录
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后
        return Path(sys.executable).parent
    else:
        # 开发模式: result_db_web/backend/app/config.py -> CrossRing根目录
        return Path(__file__).parent.parent.parent.parent


# 基础目录
BASE_DIR = get_base_dir()

# 开发模式路径
BACKEND_DIR = Path(__file__).parent.parent
RESULT_DB_WEB_DIR = BACKEND_DIR.parent
PROJECT_ROOT = RESULT_DB_WEB_DIR.parent

# 数据库路径
if getattr(sys, 'frozen', False):
    # 打包模式: 数据库在 exe 同级 data 目录
    DATABASE_DIR = BASE_DIR / "data"
    DATABASE_PATH = DATABASE_DIR / "results.db"
else:
    # 开发模式
    RESULT_DIR = PROJECT_ROOT.parent / "Result"
    DATABASE_DIR = RESULT_DIR / "Database"
    DATABASE_PATH = DATABASE_DIR / "results.db"

# 确保目录存在
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

# 前端静态文件目录
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"

# API配置
API_PREFIX = "/api"
API_PORT = 8001

# CORS配置
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3002",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
