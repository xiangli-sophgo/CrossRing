"""
仿真一体化平台 - 统一配置
合并 tool_web 和 result_db_web 的配置
"""

import sys
from pathlib import Path


def get_base_dir() -> Path:
    """
    获取应用根目录

    - 开发模式: CrossRing 项目根目录
    - 打包模式: exe 所在目录
    """
    if getattr(sys, "frozen", False):
        # PyInstaller 打包后
        return Path(sys.executable).parent
    else:
        # 开发模式: unified_web/backend/app/config.py -> CrossRing根目录
        return Path(__file__).parent.parent.parent.parent


# 基础目录
BASE_DIR = get_base_dir()

# ==================== 配置目录 (来自 tool_web) ====================
CONFIG_DIR = BASE_DIR / "config"
IP_MOUNTS_DIR = CONFIG_DIR / "ip_mounts"
TRAFFIC_CONFIGS_DIR = CONFIG_DIR / "traffic_configs"
TOPOLOGIES_DIR = CONFIG_DIR / "topologies"

# 输出目录
TRAFFIC_OUTPUT_DIR = BASE_DIR / "traffic"

# ==================== 数据库配置 (来自 result_db_web) ====================
if getattr(sys, "frozen", False):
    # 打包模式: 数据库在 exe 同级 data 目录
    DATABASE_DIR = BASE_DIR / "data"
    DATABASE_PATH = DATABASE_DIR / "results.db"
else:
    # 开发模式
    RESULT_DIR = BASE_DIR.parent / "Result"
    DATABASE_DIR = RESULT_DIR / "Database"
    DATABASE_PATH = DATABASE_DIR / "simulation.db"

# ==================== 前端静态文件目录 ====================
FRONTEND_DIST_DIR = BASE_DIR / "unified_web" / "frontend" / "dist"

# ==================== API配置 ====================
API_PREFIX = "/api"
API_PORT = 8002  # 使用新端口，避免与现有项目冲突

# ==================== CORS配置 ====================
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3002",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:3002",
    "http://127.0.0.1:5173",
]


def ensure_dirs():
    """确保必要目录存在"""
    for dir_path in [
        IP_MOUNTS_DIR,
        TRAFFIC_CONFIGS_DIR,
        TOPOLOGIES_DIR,
        TRAFFIC_OUTPUT_DIR,
        DATABASE_DIR,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
