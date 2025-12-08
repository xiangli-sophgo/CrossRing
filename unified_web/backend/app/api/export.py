"""
数据库打包导出API
"""

import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.database import ResultManager

router = APIRouter()

# 获取数据库管理器
db_manager = ResultManager()

# 项目路径
BACKEND_DIR = Path(__file__).parent.parent.parent
RESULT_DB_WEB_DIR = BACKEND_DIR.parent
PROJECT_ROOT = RESULT_DB_WEB_DIR.parent
RESULT_DIR = PROJECT_ROOT.parent / "Result"
DATABASE_DIR = RESULT_DIR / "Database"
DATABASE_PATH = DATABASE_DIR / "results.db"


class ExportRequest(BaseModel):
    """导出请求"""
    experiment_ids: Optional[List[int]] = None  # None表示全量导出
    include_frontend: bool = True
    include_backend: bool = True


class ExportInfo(BaseModel):
    """导出信息"""
    filename: str
    size: int
    experiments_count: int
    results_count: int
    created_at: str


def get_experiments_stats(experiment_ids: Optional[List[int]] = None) -> tuple:
    """获取实验统计信息"""
    experiments = db_manager.list_experiments()

    if experiment_ids:
        experiments = [e for e in experiments if e["id"] in experiment_ids]

    total_results = 0
    for exp in experiments:
        results = db_manager.get_results(exp["id"])
        total_results += results["total"]

    return len(experiments), total_results


def create_selective_database(experiment_ids: List[int], output_path: Path) -> None:
    """创建只包含特定实验的数据库副本"""
    import sqlite3

    # 复制原数据库
    shutil.copy2(DATABASE_PATH, output_path)

    # 删除不需要的实验
    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()

    # 获取所有实验ID
    cursor.execute("SELECT id FROM experiments")
    all_ids = [row[0] for row in cursor.fetchall()]

    # 删除不需要的实验数据
    ids_to_delete = [id for id in all_ids if id not in experiment_ids]

    if ids_to_delete:
        placeholders = ",".join("?" * len(ids_to_delete))

        # 删除 kcin_results
        cursor.execute(
            f"DELETE FROM kcin_results WHERE experiment_id IN ({placeholders})",
            ids_to_delete
        )

        # 删除 dcin_results
        cursor.execute(
            f"DELETE FROM dcin_results WHERE experiment_id IN ({placeholders})",
            ids_to_delete
        )

        # 删除 experiments
        cursor.execute(
            f"DELETE FROM experiments WHERE id IN ({placeholders})",
            ids_to_delete
        )

    conn.commit()

    # 压缩数据库
    cursor.execute("VACUUM")

    conn.close()


@router.get("/export/info")
async def get_export_info(
    experiment_ids: Optional[str] = Query(None, description="逗号分隔的实验ID列表，为空表示全量")
):
    """
    获取导出信息预览

    - experiment_ids: 逗号分隔的实验ID，为空表示全量导出
    """
    ids = None
    if experiment_ids:
        ids = [int(id.strip()) for id in experiment_ids.split(",")]

    exp_count, result_count = get_experiments_stats(ids)

    return {
        "experiments_count": exp_count,
        "results_count": result_count,
        "database_size": DATABASE_PATH.stat().st_size if DATABASE_PATH.exists() else 0,
        "is_selective": ids is not None,
    }


@router.get("/export/download")
async def download_package(
    experiment_ids: Optional[str] = Query(None, description="逗号分隔的实验ID列表"),
    include_frontend: bool = Query(True, description="是否包含前端代码"),
    include_backend: bool = Query(True, description="是否包含后端代码"),
):
    """
    下载打包文件

    - experiment_ids: 逗号分隔的实验ID，为空表示全量导出
    - include_frontend: 是否包含前端代码
    - include_backend: 是否包含后端代码
    """
    ids = None
    if experiment_ids:
        ids = [int(id.strip()) for id in experiment_ids.split(",")]

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if ids:
        zip_name = f"result_db_selective_{timestamp}.zip"
    else:
        zip_name = f"result_db_full_{timestamp}.zip"

    zip_path = Path(temp_dir) / zip_name

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # 打包数据库
            if ids:
                # 选择性导出：创建只包含特定实验的数据库
                selective_db_path = Path(temp_dir) / "results.db"
                create_selective_database(ids, selective_db_path)
                zipf.write(selective_db_path, "data/results.db")
            else:
                # 全量导出
                if DATABASE_PATH.exists():
                    zipf.write(DATABASE_PATH, "data/results.db")

            # 打包后端代码
            if include_backend:
                backend_dir = RESULT_DB_WEB_DIR / "backend"
                for root, dirs, files in os.walk(backend_dir):
                    # 跳过 __pycache__
                    dirs[:] = [d for d in dirs if d != "__pycache__"]

                    for file in files:
                        if file.endswith((".py", ".txt", ".md")):
                            file_path = Path(root) / file
                            arcname = "backend" / file_path.relative_to(backend_dir)
                            zipf.write(file_path, arcname)

            # 打包前端代码（使用构建后的dist）
            if include_frontend:
                frontend_dist = RESULT_DB_WEB_DIR / "frontend" / "dist"
                if frontend_dist.exists():
                    for root, dirs, files in os.walk(frontend_dist):
                        for file in files:
                            file_path = Path(root) / file
                            arcname = "frontend/dist" / file_path.relative_to(frontend_dist)
                            zipf.write(file_path, arcname)

                # 同时包含源代码
                frontend_src = RESULT_DB_WEB_DIR / "frontend" / "src"
                if frontend_src.exists():
                    for root, dirs, files in os.walk(frontend_src):
                        dirs[:] = [d for d in dirs if d != "node_modules"]
                        for file in files:
                            file_path = Path(root) / file
                            arcname = "frontend/src" / file_path.relative_to(frontend_src)
                            zipf.write(file_path, arcname)

                # 包含配置文件
                for config_file in ["package.json", "vite.config.ts", "tsconfig.json", "index.html"]:
                    config_path = RESULT_DB_WEB_DIR / "frontend" / config_file
                    if config_path.exists():
                        zipf.write(config_path, f"frontend/{config_file}")

            # 添加README
            readme_content = f"""# 仿真结果数据库导出包

导出时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
导出类型: {"选择性导出" if ids else "全量导出"}

## 包含内容

- data/results.db: SQLite数据库文件
{"- backend/: 后端Python代码" if include_backend else ""}
{"- frontend/: 前端React代码" if include_frontend else ""}

## 使用方法

1. 解压此文件
2. 将 data/results.db 放到 Result/Database/ 目录
3. 启动后端: cd backend && pip install -r requirements.txt && python -m uvicorn app.main:app --port 8001
4. 启动前端: cd frontend && pnpm install && pnpm dev

## 导出的实验

{"实验ID: " + ", ".join(map(str, ids)) if ids else "全部实验"}
"""
            zipf.writestr("README.md", readme_content)

        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=zip_name,
            background=None,  # 同步发送
        )

    except Exception as e:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"打包失败: {str(e)}")


@router.get("/export/build")
async def build_executable_package(
    experiment_ids: Optional[str] = Query(None, description="逗号分隔的实验ID列表"),
):
    """
    构建可执行打包文件（包含exe、前端、数据库）

    - experiment_ids: 逗号分隔的实验ID，为空表示全量导出
    """
    import subprocess
    import sys

    ids = None
    if experiment_ids:
        ids = [int(id.strip()) for id in experiment_ids.split(",")]

    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"ResultDB_{timestamp}.zip"
    zip_path = temp_dir / zip_name

    try:
        # 打包目录
        app_dir = temp_dir / "ResultDB"
        app_dir.mkdir(parents=True, exist_ok=True)

        # 1. 构建前端
        frontend_dir = RESULT_DB_WEB_DIR / "frontend"
        frontend_dist = frontend_dir / "dist"

        # 运行 pnpm build
        result = subprocess.run(
            ["pnpm", "build"],
            cwd=frontend_dir,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"前端构建失败: {result.stderr}")

        # 复制前端 dist
        if frontend_dist.exists():
            shutil.copytree(frontend_dist, app_dir / "frontend" / "dist")

        # 2. 使用 PyInstaller 打包后端
        backend_dir = RESULT_DB_WEB_DIR / "backend"
        pyinstaller_dist = temp_dir / "pyinstaller_dist"

        # 创建入口脚本
        entry_script = temp_dir / "run_server.py"
        entry_script.write_text('''
import sys
import os
import webbrowser
import threading
import socket

# 设置工作目录
if getattr(sys, 'frozen', False):
    base_path = os.path.dirname(sys.executable)
    os.chdir(base_path)

import uvicorn
from app.main import app


def find_free_port(start_port=8001, max_tries=10):
    """Find available port"""
    for i in range(max_tries):
        port = start_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    return start_port


if __name__ == "__main__":
    port = find_free_port(8001)

    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=open_browser, daemon=True).start()

    print(f"Server starting on port: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
''', encoding='utf-8')

        # PyInstaller 命令
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--name=ResultDB",
            "--onedir",
            "--noconfirm",
            "--clean",
            f"--distpath={pyinstaller_dist}",
            f"--workpath={temp_dir / 'build'}",
            f"--specpath={temp_dir}",
            # 添加路径
            f"--paths={backend_dir}",
            f"--paths={PROJECT_ROOT}",
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
            "--hidden-import=app.main",
            "--hidden-import=app.api.experiments",
            "--hidden-import=app.api.results",
            "--hidden-import=app.api.analysis",
            "--hidden-import=app.api.export",
            "--hidden-import=app.config",
            "--hidden-import=src.database",
            "--hidden-import=src.database.manager",
            "--hidden-import=src.database.database",
            "--hidden-import=src.database.models",
            # 收集数据文件
            f"--add-data={PROJECT_ROOT / 'src'};src",
            f"--add-data={backend_dir / 'app'};app",
            str(entry_script),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"后端打包失败: {result.stderr}")

        # 复制 PyInstaller 输出到 app_dir
        pyinstaller_output = pyinstaller_dist / "ResultDB"
        if pyinstaller_output.exists():
            for item in pyinstaller_output.iterdir():
                if item.is_dir():
                    shutil.copytree(item, app_dir / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, app_dir / item.name)

        # 3. 创建数据库
        data_dir = app_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        if ids:
            # 选择性导出
            db_path = data_dir / "results.db"
            create_selective_database(ids, db_path)
        else:
            # 全量导出
            if DATABASE_PATH.exists():
                shutil.copy2(DATABASE_PATH, data_dir / "results.db")

        # 4. 创建启动脚本
        bat_content = '''@echo off
chcp 65001 >nul
echo ============================================
echo Result Database Starting...
echo ============================================
echo.
echo Browser will open automatically
echo Press Ctrl+C to stop
echo ============================================
ResultDB.exe
pause
'''
        (app_dir / "start.bat").write_text(bat_content, encoding='utf-8')

        # 5. 创建 README
        readme_content = f"""# 仿真结果数据库可执行包

打包时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
包含实验: {"选择性导出 (ID: " + ", ".join(map(str, ids)) + ")" if ids else "全部实验"}

## 使用方法

1. 双击 start.bat
2. 浏览器访问 http://localhost:8001

## 文件说明

- ResultDB.exe: 后端可执行文件
- start.bat: 启动脚本
- frontend/dist/: 前端静态文件
- data/results.db: 数据库文件
"""
        (app_dir / "README.md").write_text(readme_content, encoding='utf-8')

        # 6. 打包为 zip
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(app_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(temp_dir)
                    zipf.write(file_path, arcname)

        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=zip_name,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"打包失败: {str(e)}")


@router.post("/import/package")
async def import_package():
    """
    导入打包文件（TODO: 实现导入功能）
    """
    raise HTTPException(status_code=501, detail="导入功能开发中")
