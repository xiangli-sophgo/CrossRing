@echo off
chcp 65001 > nul
title CrossRing 一体化仿真平台

echo ========================================
echo   CrossRing 一体化仿真平台
echo ========================================

set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

:: 启动后端
echo 启动后端服务...
cd backend

:: 检查虚拟环境
if not exist "venv" (
    echo 创建Python虚拟环境...
    python -m venv venv
)

call venv\Scripts\activate.bat
pip install -r requirements.txt -q

:: 设置PYTHONPATH并启动后端
set PYTHONPATH=%SCRIPT_DIR%..
start "CrossRing Backend" cmd /k "cd /d %SCRIPT_DIR%backend && call venv\Scripts\activate.bat && python -m uvicorn app.main:app --port 8002 --reload"

cd ..

:: 启动前端
echo 启动前端服务...
cd frontend

:: 检查node_modules
if not exist "node_modules" (
    echo 安装前端依赖...
    call pnpm install
)

start "CrossRing Frontend" cmd /k "cd /d %SCRIPT_DIR%frontend && pnpm run dev"

echo.
echo ========================================
echo 服务已启动:
echo   前端: http://localhost:3002
echo   后端: http://localhost:8002
echo   API文档: http://localhost:8002/api/docs
echo ========================================
echo 关闭此窗口不会停止服务
echo 请手动关闭 Backend 和 Frontend 窗口来停止服务
echo.

:: 等待几秒后自动打开浏览器
timeout /t 3 /nobreak > nul
start http://localhost:3002

pause
