"""
CrossRing Web 打包脚本

使用方法:
    python build.py

打包结果:
    dist/CrossRing-Web/
        CrossRing-Web.exe
        config/
        traffic/
        tool_web/frontend/dist/
        src/
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
TOOL_WEB_DIR = Path(__file__).parent  # tool_web 目录
BACKEND_DIR = TOOL_WEB_DIR / "backend"
FRONTEND_DIR = TOOL_WEB_DIR / "frontend"
DIST_DIR = TOOL_WEB_DIR / "dist"


def clean_dist():
    """清理旧的打包文件"""
    if DIST_DIR.exists():
        print("清理旧的打包文件...")
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir(parents=True, exist_ok=True)


def build_frontend():
    """打包前端"""
    print("=" * 50)
    print("打包前端...")
    os.chdir(FRONTEND_DIR)
    result = subprocess.run(["npm", "run", "build"], shell=True)
    if result.returncode != 0:
        print("前端打包失败")
        sys.exit(1)
    print("前端打包完成")


def build_backend():
    """使用 PyInstaller 打包后端"""
    print("=" * 50)
    print("打包后端...")
    os.chdir(BACKEND_DIR)

    # PyInstaller 命令 - 使用 python -m 方式调用
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=CrossRing-Web",
        "--onedir",  # 打包为目录模式
        "--noconfirm",
        "--clean",
        f"--distpath={DIST_DIR}",
        f"--workpath={TOOL_WEB_DIR / 'build'}",
        f"--specpath={TOOL_WEB_DIR}",
        # 添加隐式导入
        "--hidden-import=uvicorn.logging",
        "--hidden-import=uvicorn.loops",
        "--hidden-import=uvicorn.loops.auto",
        "--hidden-import=uvicorn.protocols",
        "--hidden-import=uvicorn.protocols.http",
        "--hidden-import=uvicorn.protocols.http.auto",
        "--hidden-import=uvicorn.protocols.websockets",
        "--hidden-import=uvicorn.protocols.websockets.auto",
        "--hidden-import=uvicorn.lifespan",
        "--hidden-import=uvicorn.lifespan.on",
        "--hidden-import=uvicorn.lifespan.off",
        "--hidden-import=app.api.topology",
        "--hidden-import=app.api.ip_mount",
        "--hidden-import=app.api.traffic_config",
        "--hidden-import=app.api.traffic_generate",
        "--hidden-import=app.api.static_bandwidth",
        "--hidden-import=app.config",
        # 收集数据文件
        f"--add-data={PROJECT_ROOT / 'src'};src",
        "app/main.py",
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("后端打包失败")
        sys.exit(1)
    print("后端打包完成")


def copy_resources():
    """复制必要资源文件"""
    print("=" * 50)
    print("复制资源文件...")

    app_dir = DIST_DIR / "CrossRing-Web"

    # 复制前端 dist
    frontend_dest = app_dir / "tool_web" / "frontend" / "dist"
    frontend_dest.mkdir(parents=True, exist_ok=True)
    if (FRONTEND_DIR / "dist").exists():
        shutil.copytree(FRONTEND_DIR / "dist", frontend_dest, dirs_exist_ok=True)
        print(f"  前端文件 -> {frontend_dest}")

    # 复制 config 目录
    config_dest = app_dir / "config"
    config_src = PROJECT_ROOT / "config"
    if config_src.exists():
        shutil.copytree(config_src, config_dest, dirs_exist_ok=True)
        print(f"  配置文件 -> {config_dest}")

    # 创建 traffic 输出目录
    traffic_dest = app_dir / "traffic"
    traffic_dest.mkdir(parents=True, exist_ok=True)
    print(f"  流量输出目录 -> {traffic_dest}")

    print("资源复制完成")


def create_launcher():
    """创建启动脚本"""
    print("=" * 50)
    print("创建启动脚本...")

    app_dir = DIST_DIR / "CrossRing-Web"

    # Windows 启动脚本
    bat_content = '''@echo off
chcp 65001 >nul
echo ============================================
echo CrossRing Web Tool Starting...
echo ============================================
echo.
echo Browser: http://localhost:8000
echo.
echo Press Ctrl+C to stop
echo ============================================
CrossRing-Web.exe
pause
'''
    (app_dir / "start.bat").write_text(bat_content, encoding='utf-8')
    print("  start.bat 创建完成")


def main():
    print("=" * 50)
    print("CrossRing Web 打包工具")
    print("=" * 50)

    # 检查依赖
    try:
        import PyInstaller
    except ImportError:
        print("请先安装 PyInstaller: pip install pyinstaller")
        sys.exit(1)

    clean_dist()
    build_frontend()
    build_backend()
    copy_resources()
    create_launcher()

    print("=" * 50)
    print("打包完成!")
    print(f"输出目录: {DIST_DIR / 'CrossRing-Web'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
