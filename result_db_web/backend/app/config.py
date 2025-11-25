"""
仿真结果数据库 - 配置
"""

import os
from pathlib import Path

# 获取项目根目录
BACKEND_DIR = Path(__file__).parent.parent
OPTIMIZATION_WEB_DIR = BACKEND_DIR.parent
PROJECT_ROOT = OPTIMIZATION_WEB_DIR.parent

# 数据库路径
RESULT_DIR = PROJECT_ROOT.parent / "Result"
DATABASE_DIR = RESULT_DIR / "Database"
DATABASE_PATH = DATABASE_DIR / "optimization.db"

# 确保目录存在
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

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
