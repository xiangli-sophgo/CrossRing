"""
路径配置模块 - 支持开发模式和打包模式
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
        # 开发模式: web/backend/app/config.py -> CrossRing根目录
        return Path(__file__).parent.parent.parent.parent


# 基础目录
BASE_DIR = get_base_dir()

# 配置目录
CONFIG_DIR = BASE_DIR / "config"
IP_MOUNTS_DIR = CONFIG_DIR / "ip_mounts"
TRAFFIC_CONFIGS_DIR = CONFIG_DIR / "traffic_configs"
TOPOLOGIES_DIR = CONFIG_DIR / "topologies"

# 输出目录
TRAFFIC_OUTPUT_DIR = BASE_DIR / "traffic"

# 前端静态文件目录
FRONTEND_DIST_DIR = BASE_DIR / "web" / "frontend" / "dist"

# 确保目录存在
def ensure_dirs():
    """确保必要目录存在"""
    for dir_path in [IP_MOUNTS_DIR, TRAFFIC_CONFIGS_DIR, TOPOLOGIES_DIR, TRAFFIC_OUTPUT_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
