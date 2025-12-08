#!/bin/bash
# CrossRing 一体化仿真平台 - 启动脚本 (Linux/macOS)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  CrossRing 一体化仿真平台"
echo "========================================"

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3"
    exit 1
fi

# 检查Node/pnpm
if ! command -v pnpm &> /dev/null; then
    echo "提示: 未找到 pnpm, 尝试使用 npm"
    NPM_CMD="npm"
else
    NPM_CMD="pnpm"
fi

# 启动后端
echo "启动后端服务..."
cd backend
if [ ! -d "venv" ]; then
    echo "创建Python虚拟环境..."
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt -q

# 后台启动后端
PYTHONPATH="$SCRIPT_DIR/.." python -m uvicorn app.main:app --port 8002 --reload &
BACKEND_PID=$!
echo "后端PID: $BACKEND_PID"

cd ..

# 启动前端
echo "启动前端服务..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "安装前端依赖..."
    $NPM_CMD install
fi

$NPM_CMD run dev &
FRONTEND_PID=$!
echo "前端PID: $FRONTEND_PID"

echo ""
echo "========================================"
echo "服务已启动:"
echo "  前端: http://localhost:3002"
echo "  后端: http://localhost:8002"
echo "  API文档: http://localhost:8002/api/docs"
echo "========================================"
echo "按 Ctrl+C 停止所有服务"

# 等待退出信号
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
