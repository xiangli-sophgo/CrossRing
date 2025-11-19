# VS Code Tasks 使用指南

已在 `.vscode/tasks.json` 中添加了CrossRing Web的启动任务。

## 🎯 可用任务

### 1. **CrossRing Web: Setup** (首次运行)

安装所有依赖（后端+前端）

**使用方法:**
1. 按 `Cmd+Shift+P` (macOS) 或 `Ctrl+Shift+P` (Windows/Linux)
2. 输入 `Tasks: Run Task`
3. 选择 `CrossRing Web: Setup`

**功能:**
- 创建Python虚拟环境
- 安装后端依赖 (~1-2分钟)
- 安装前端依赖 (~3-5分钟)

**仅需运行一次！**

---

### 2. **CrossRing Web: Start All** (推荐)

同时启动前后端服务

**使用方法:**
1. 按 `Cmd+Shift+P` (macOS) 或 `Ctrl+Shift+P` (Windows/Linux)
2. 输入 `Tasks: Run Task`
3. 选择 `CrossRing Web: Start All`

**效果:**
- 后端启动在 http://localhost:8000
- 前端启动在 http://localhost:3000
- 两个服务显示在不同的终端面板中

**快捷键设置（可选）:**

在 `.vscode/keybindings.json` 中添加:
```json
{
  "key": "cmd+shift+w",
  "command": "workbench.action.tasks.runTask",
  "args": "CrossRing Web: Start All"
}
```

---

### 3. **CrossRing Web: Start Backend** (仅后端)

只启动后端API服务

**使用场景:**
- 单独测试后端API
- 使用Postman/curl测试
- 前端已在其他地方运行

**访问:**
- API文档: http://localhost:8000/api/docs

---

### 4. **CrossRing Web: Start Frontend** (仅前端)

只启动前端开发服务器

**使用场景:**
- 后端已在其他地方运行
- 单独调试前端代码

**访问:**
- 前端界面: http://localhost:3000

---

## 📋 完整工作流程

### 首次使用

```
1. 运行 "CrossRing Web: Setup"
   → 等待依赖安装完成 (5-10分钟)

2. 运行 "CrossRing Web: Start All"
   → 前后端同时启动

3. 打开浏览器访问 http://localhost:3000
```

### 日常开发

```
1. 打开VS Code

2. Cmd+Shift+P → "CrossRing Web: Start All"

3. 开始开发（代码修改会自动热更新）

4. 停止: 点击终端右上角的垃圾桶图标
```

---

## 🔧 任务特性

### 后台运行
- 设置了 `"isBackground": true`
- 任务会持续运行，监听文件变化

### 热更新
- **后端**: 修改Python代码后自动重启
- **前端**: 修改React代码后浏览器自动刷新

### 终端面板
- 使用 `"panel": "dedicated"`
- 后端和前端各占一个专用面板
- 属于同一个 `"crossring-web"` 组

### 并行启动
- `"dependsOrder": "parallel"`
- 前后端同时启动，不阻塞

---

## 🛑 停止服务

### 方法1: VS Code终端
点击终端右上角的 🗑️ (垃圾桶图标)

### 方法2: 命令面板
1. `Cmd+Shift+P`
2. 输入 `Tasks: Terminate Task`
3. 选择要停止的任务

### 方法3: 命令行
```bash
# 杀掉所有相关进程
lsof -ti:8000 | xargs kill -9  # 后端
lsof -ti:3000 | xargs kill -9  # 前端
```

---

## 🐛 故障排查

### 问题1: "venv/bin/activate: No such file"

**原因:** 未运行Setup任务

**解决:** 先运行 `CrossRing Web: Setup`

---

### 问题2: "pnpm: command not found"

**原因:** pnpm未安装

**解决:**
```bash
npm install -g pnpm
```

---

### 问题3: 端口被占用

**症状:** `Address already in use`

**解决:**
```bash
# macOS/Linux
lsof -ti:8000 | xargs kill -9
lsof -ti:3000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

---

### 问题4: Python虚拟环境激活失败

**解决:** 手动创建虚拟环境
```bash
cd web/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 💡 高级配置

### 自定义端口

编辑 `.vscode/tasks.json`:

```json
// 后端端口
"command": "... --port 9000"

// 前端端口 (编辑 web/frontend/vite.config.ts)
server: {
  port: 4000
}
```

### 添加环境变量

```json
{
  "label": "CrossRing Web: Start Backend",
  "options": {
    "env": {
      "API_PORT": "8000",
      "LOG_LEVEL": "DEBUG"
    }
  }
}
```

### 修改Python路径

如果使用不同的Python版本:

```json
{
  "label": "CrossRing Web: Start Backend",
  "command": "source web/backend/venv/bin/activate && /usr/local/bin/python3.11 -m uvicorn ..."
}
```

---

## 📖 相关文档

- 主文档: `web/README.md`
- 安装指南: `web/INSTALL.md`
- 后端文档: `web/backend/README.md`
- 前端文档: `web/frontend/README.md`

---

## 🎨 VS Code 推荐扩展

安装这些扩展以获得更好的开发体验:

```json
{
  "recommendations": [
    "ms-python.python",              // Python支持
    "ms-python.vscode-pylance",      // Python类型检查
    "dbaeumer.vscode-eslint",        // JavaScript/TypeScript检查
    "esbenp.prettier-vscode",        // 代码格式化
    "dsznajder.es7-react-js-snippets" // React代码片段
  ]
}
```

保存到 `.vscode/extensions.json` 即可。

---

**快速开始:** 运行 `CrossRing Web: Setup` → `CrossRing Web: Start All` → 访问 http://localhost:3000 🚀
