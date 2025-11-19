# 🚀 CrossRing Web 快速开始

3步启动，5分钟运行！

---

## ⚡ 最快路径（使用VS Code Tasks）

### 步骤1: 安装依赖（仅首次）

1. 在VS Code中打开CrossRing项目
2. 按 `Cmd+Shift+P` (macOS) 或 `Ctrl+Shift+P` (Windows)
3. 输入 `Tasks: Run Task`
4. 选择 **`CrossRing Web: Setup`**
5. 等待5-10分钟完成安装

### 步骤2: 启动服务

1. 按 `Cmd+Shift+P`
2. 输入 `Tasks: Run Task`
3. 选择 **`CrossRing Web: Start All`**

### 步骤3: 访问应用

打开浏览器访问: **http://localhost:3000**

🎉 完成！

---

## 🛠️ 传统启动方式

### macOS/Linux

```bash
# 进入web目录
cd /Users/lixiang/Documents/工作/code/CrossRing/web

# 一键启动
./start-dev.sh
```

### Windows

```cmd
cd CrossRing\web
start-dev.bat
```

---

## 📋 详细步骤（第一次使用）

### 前置条件

确认已安装：
- ✅ Python 3.8+ (已有)
- ⚠️ Node.js 18+ ([安装指南](./INSTALL.md))

检查命令：
```bash
python3 --version  # 应显示3.8+
node --version     # 应显示18+
```

### 安装Node.js（如未安装）

**macOS (Homebrew):**
```bash
brew install node@20
```

**Windows:**
访问 https://nodejs.org/ 下载安装

**验证:**
```bash
node --version  # v20.x.x
```

### 启动应用

#### 方式1: 一键脚本（推荐）

```bash
cd web
./start-dev.sh  # macOS/Linux
# 或
start-dev.bat   # Windows
```

首次运行会自动安装所有依赖，约5-10分钟。

#### 方式2: VS Code Tasks

参见 [VSCODE_TASKS.md](./VSCODE_TASKS.md)

#### 方式3: 手动启动

**终端1 - 后端:**
```bash
cd web/backend
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate.bat
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

**终端2 - 前端:**
```bash
cd web/frontend
npm install -g pnpm
pnpm install
pnpm dev
```

---

## 🌐 访问地址

启动成功后访问:

| 服务 | 地址 | 说明 |
|------|------|------|
| 🎨 **前端界面** | http://localhost:3000 | React应用 |
| 🔧 **后端API** | http://localhost:8000 | FastAPI |
| 📖 **API文档** | http://localhost:8000/api/docs | Swagger UI |

---

## 🛑 停止服务

### VS Code
点击终端右上角的 🗑️ (垃圾桶图标)

### 脚本方式
按 `Ctrl+C`

### 强制停止
```bash
# macOS/Linux
lsof -ti:8000 | xargs kill -9  # 停止后端
lsof -ti:3000 | xargs kill -9  # 停止前端

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

---

## 🐛 遇到问题？

### Q1: Node.js未安装

**症状:** `node: command not found`

**解决:** 参考 [INSTALL.md](./INSTALL.md) 安装Node.js

---

### Q2: 端口被占用

**症状:** `Address already in use: 8000`

**解决:**
```bash
lsof -ti:8000 | xargs kill -9  # macOS/Linux
```

---

### Q3: pnpm未安装

**症状:** `pnpm: command not found`

**解决:**
```bash
npm install -g pnpm
```

---

### Q4: 依赖安装失败

**症状:** 安装过程报错

**解决:**
```bash
# 清理缓存重试
cd web/backend
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cd ../frontend
rm -rf node_modules pnpm-lock.yaml
pnpm install
```

---

## 📖 更多文档

- 📘 [完整文档](./README.md) - 详细功能说明
- 🔧 [安装指南](./INSTALL.md) - 各平台安装教程
- 💻 [VS Code Tasks](./VSCODE_TASKS.md) - VS Code集成使用
- 🐍 [后端文档](./backend/README.md) - FastAPI开发
- ⚛️ [前端文档](./frontend/README.md) - React开发

---

## ✅ 验证清单

启动成功的标志:

- [ ] 后端输出: `🚀 CrossRing Web API 已启动`
- [ ] 前端输出: `Local: http://localhost:3000/`
- [ ] 浏览器能访问 http://localhost:3000
- [ ] 前端显示 "CrossRing Web" 标题
- [ ] "后端API状态" 卡片显示绿色 ✅

全部勾选 = 成功运行！

---

## 🎯 下一步

成功启动后，可以：

1. 📊 查看拓扑可视化（开发中）
2. ⚙️ 配置IP挂载和流量参数（开发中）
3. 📈 生成和分析流量数据（开发中）

---

## 💡 提示

- **热更新**: 修改代码后自动刷新，无需重启
- **API文档**: 访问 http://localhost:8000/api/docs 查看所有API
- **VS Code**: 使用Tasks功能一键启动（最方便）
- **性能**: 首次启动较慢（加载依赖），后续启动秒开

---

**有问题？** 查看 [INSTALL.md](./INSTALL.md) 的故障排查部分，或联系开发团队。

**Ready to go? 🚀**
```bash
cd web && ./start-dev.sh
```
