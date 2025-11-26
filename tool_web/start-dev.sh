#!/bin/bash
# CrossRing Web开发环境启动脚本 (macOS/Linux)
# 不使用虚拟环境，直接使用系统Python

echo "========================================="
echo "  CrossRing Web 开发环境启动"
echo "========================================="
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 Python 3"
    echo "请先安装 Python 3.8+"
    exit 1
fi

# 检查Node.js
if ! command -v node &> /dev/null; then
    echo "❌ 错误: 未找到 Node.js"
    echo "请先安装 Node.js 18+"
    exit 1
fi

# 检查pnpm
if ! command -v pnpm &> /dev/null; then
    echo "⚠️  未找到 pnpm，正在安装..."
    npm install -g pnpm
fi

echo "✅ 环境检查通过"
echo ""

# 安装后端依赖
echo "📦 检查后端依赖..."
cd backend
pip3 install -q -r requirements.txt
echo "✅ 后端依赖已安装"
cd ..

# 安装前端依赖
echo "📦 检查前端依赖..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "安装前端依赖（首次可能需要几分钟）..."
    pnpm install
fi
echo "✅ 前端依赖已安装"
cd ..

echo ""
echo "========================================="
echo "  🚀 启动服务"
echo "========================================="
echo "后端API: http://localhost:8000"
echo "前端界面: http://localhost:3000"
echo "API文档: http://localhost:8000/api/docs"
echo ""
echo "按 Ctrl+C 停止所有服务"
echo "========================================="
echo ""

# 捕获退出信号，终止所有子进程
trap 'kill $(jobs -p) 2>/dev/null' EXIT INT TERM

# 启动后端
cd backend
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# 等待后端启动
sleep 2

# 启动前端
cd ../frontend
pnpm dev &
FRONTEND_PID=$!

# 等待用户中断
wait
