#!/bin/bash

# 仿真结果数据库启动脚本

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "=================================="
echo "仿真结果数据库"
echo "=================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python"
    exit 1
fi

# 检查Node环境
if ! command -v pnpm &> /dev/null && ! command -v npm &> /dev/null; then
    echo "错误: 未找到pnpm或npm"
    exit 1
fi

# 安装后端依赖
echo "检查后端依赖..."
cd "$SCRIPT_DIR/backend"
pip install -r requirements.txt -q

# 安装前端依赖
echo "检查前端依赖..."
cd "$SCRIPT_DIR/frontend"
if command -v pnpm &> /dev/null; then
    pnpm install --silent
else
    npm install --silent
fi

# 启动后端
echo "启动后端服务 (端口 8001)..."
cd "$SCRIPT_DIR/backend"
PYTHONPATH="$PROJECT_ROOT" python -m uvicorn app.main:app --port 8001 --reload &
BACKEND_PID=$!

# 等待后端启动
sleep 2

# 启动前端
echo "启动前端服务 (端口 3000)..."
cd "$SCRIPT_DIR/frontend"
if command -v pnpm &> /dev/null; then
    pnpm dev &
else
    npm run dev &
fi
FRONTEND_PID=$!

echo ""
echo "=================================="
echo "服务已启动:"
echo "  后端API: http://localhost:8001"
echo "  前端界面: http://localhost:3000"
echo "  API文档: http://localhost:8001/docs"
echo "=================================="
echo "按 Ctrl+C 停止所有服务"

# 等待中断信号
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
