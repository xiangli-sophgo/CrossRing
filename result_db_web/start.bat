@echo off
chcp 65001 >nul
setlocal

echo ==================================
echo 仿真结果数据库
echo ==================================

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

:: 安装后端依赖
echo 检查后端依赖...
cd /d "%SCRIPT_DIR%backend"
pip install -r requirements.txt -q

:: 安装前端依赖
echo 检查前端依赖...
cd /d "%SCRIPT_DIR%frontend"
call pnpm install --silent 2>nul || call npm install --silent

:: 启动后端
echo 启动后端服务 (端口 8001)...
cd /d "%SCRIPT_DIR%backend"
start "Backend" cmd /k "set PYTHONPATH=%PROJECT_ROOT% && python -m uvicorn app.main:app --port 8001 --reload"

:: 等待后端启动
timeout /t 3 /nobreak >nul

:: 启动前端
echo 启动前端服务 (端口 3000)...
cd /d "%SCRIPT_DIR%frontend"
start "Frontend" cmd /k "pnpm dev 2>nul || npm run dev"

echo.
echo ==================================
echo 服务已启动:
echo   后端API: http://localhost:8001
echo   前端界面: http://localhost:3000
echo   API文档: http://localhost:8001/docs
echo ==================================
echo 关闭此窗口不会停止服务
echo 要停止服务，请关闭 Backend 和 Frontend 窗口

pause
