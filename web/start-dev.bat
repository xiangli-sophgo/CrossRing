@echo off
REM CrossRing Web开发环境启动脚本 (Windows)
REM 不使用虚拟环境，直接使用系统Python

echo =========================================
echo   CrossRing Web 开发环境启动
echo =========================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到 Python
    echo 请先安装 Python 3.8+
    pause
    exit /b 1
)

REM 检查Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到 Node.js
    echo 请先安装 Node.js 18+
    pause
    exit /b 1
)

REM 检查pnpm
pnpm --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  未找到 pnpm，正在安装...
    npm install -g pnpm
)

echo ✅ 环境检查通过
echo.

REM 安装后端依赖
echo 📦 检查后端依赖...
cd backend
pip install -q -r requirements.txt
echo ✅ 后端依赖已安装
cd ..

REM 安装前端依赖
echo 📦 检查前端依赖...
cd frontend
if not exist "node_modules" (
    echo 安装前端依赖（首次可能需要几分钟）...
    pnpm install
)
echo ✅ 前端依赖已安装
cd ..

echo.
echo =========================================
echo   🚀 启动服务
echo =========================================
echo 后端API: http://localhost:8000
echo 前端界面: http://localhost:3000
echo API文档: http://localhost:8000/api/docs
echo.
echo 关闭窗口可停止服务
echo =========================================
echo.

REM 启动后端（新窗口）
start "CrossRing Backend" cmd /k "cd backend && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

REM 等待后端启动
timeout /t 3 /nobreak >nul

REM 启动前端（新窗口）
start "CrossRing Frontend" cmd /k "cd frontend && pnpm dev"

echo.
echo ✅ 服务已在新窗口中启动
echo.
pause
