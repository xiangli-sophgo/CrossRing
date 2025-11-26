# CrossRing Web 安装指南

## 🎯 安装流程概览

```
1. 安装Node.js (如未安装)
2. 进入web目录
3. 运行启动脚本
4. 等待自动安装依赖
5. 访问应用
```

预计首次安装时间: **5-10分钟**

---

## 📋 详细步骤

### 步骤1: 检查Python环境

CrossRing项目已有Python环境，无需额外安装。

```bash
python3 --version
# 应显示 Python 3.8+
```

### 步骤2: 安装Node.js

#### macOS

**方法1: Homebrew (推荐)**
```bash
# 安装Homebrew (如未安装)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装Node.js
brew install node@20
```

**方法2: 官方安装包**
1. 访问 https://nodejs.org/
2. 下载 **LTS版本** (推荐20.x)
3. 运行安装程序

**方法3: nvm (版本管理器)**
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 20
nvm use 20
```

#### Windows

**方法1: 官方安装包 (推荐)**
1. 访问 https://nodejs.org/zh-cn/
2. 下载 **LTS版本** (推荐20.x)
3. 运行 `.msi` 安装程序
4. 安装时勾选 **"Add to PATH"**

**方法2: Chocolatey**
```powershell
# 以管理员身份运行PowerShell
choco install nodejs-lts
```

**方法3: Scoop**
```powershell
scoop install nodejs-lts
```

#### Linux (Ubuntu/Debian)

```bash
# 使用NodeSource仓库
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# 或使用nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 20
```

**验证安装:**
```bash
node --version   # 应显示 v20.x.x
npm --version    # 应显示 10.x.x
```

### 步骤3: 启动应用

#### macOS/Linux

```bash
cd /Users/lixiang/Documents/工作/code/CrossRing/tool_web
./start-dev.sh
```

脚本会自动：
- ✅ 检查环境
- ✅ 安装pnpm
- ✅ 创建Python虚拟环境
- ✅ 安装后端依赖 (~50MB, 1-2分钟)
- ✅ 安装前端依赖 (~200MB, 3-5分钟)
- ✅ 启动后端和前端服务

#### Windows

```cmd
cd C:\...\CrossRing\web
start-dev.bat
```

脚本会弹出两个新窗口分别运行后端和前端。

### 步骤4: 访问应用

**后端启动成功标志:**
```
🚀 CrossRing Web API 已启动
📁 CrossRing根目录: /Users/lixiang/Documents/工作/code/CrossRing
📖 API文档: http://localhost:8000/api/docs
```

**前端启动成功标志:**
```
VITE v5.4.10  ready in 1234 ms

➜  Local:   http://localhost:3000/
➜  Network: http://192.168.x.x:3000/
```

**访问链接:**
- 前端应用: http://localhost:3000
- 后端API: http://localhost:8000
- API文档: http://localhost:8000/api/docs

---

## 🐛 常见问题排查

### Q1: Node.js安装后命令无效

**症状:** 输入 `node --version` 提示 `command not found`

**原因:** 环境变量未配置

**解决:**

**macOS/Linux:**
```bash
# 检查Node.js安装路径
which node

# 如果没有输出，添加到PATH
# 在 ~/.zshrc 或 ~/.bash_profile 添加:
export PATH="/usr/local/bin:$PATH"

# 重新加载配置
source ~/.zshrc  # 或 source ~/.bash_profile
```

**Windows:**
1. 搜索"环境变量"
2. 编辑系统环境变量
3. 确认 `Path` 中包含Node.js路径 (如 `C:\Program Files\nodejs\`)
4. 重启终端

### Q2: pnpm安装失败

**症状:** `npm install -g pnpm` 报权限错误

**解决:**

**macOS/Linux:**
```bash
# 使用sudo
sudo npm install -g pnpm

# 或配置npm全局目录
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
export PATH=~/.npm-global/bin:$PATH
npm install -g pnpm
```

**Windows:**
以管理员身份运行PowerShell，然后执行安装命令。

### Q3: 端口被占用

**症状:**
```
Error: listen EADDRINUSE: address already in use :::8000
```

**解决:**

**macOS/Linux:**
```bash
# 查找占用端口的进程
lsof -ti:8000

# 终止进程
lsof -ti:8000 | xargs kill -9
```

**Windows:**
```cmd
# 查找占用端口的进程
netstat -ano | findstr :8000

# 终止进程（替换<PID>为实际进程ID）
taskkill /PID <PID> /F
```

### Q4: Python虚拟环境创建失败

**症状:** `python3 -m venv venv` 报错

**原因:** 缺少venv模块

**解决:**

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-venv
```

**macOS:**
```bash
# 通常自带venv，如果不行重装Python
brew reinstall python@3
```

### Q5: 前端依赖安装慢

**症状:** `pnpm install` 卡住或很慢

**原因:** 网络问题或npm源慢

**解决:**
```bash
# 使用国内镜像（淘宝）
pnpm config set registry https://registry.npmmirror.com

# 重新安装
rm -rf node_modules pnpm-lock.yaml
pnpm install
```

### Q6: 前端无法连接后端

**症状:** 浏览器控制台报CORS错误

**原因:** 后端未启动或CORS配置问题

**解决:**
1. 确认后端已启动: `curl http://localhost:8000`
2. 检查 `backend/app/main.py` 的CORS配置
3. 确保前端URL在允许列表中

---

## 🔄 卸载和清理

### 清理所有依赖和虚拟环境

```bash
cd /Users/lixiang/Documents/工作/code/CrossRing/tool_web

# 清理后端
rm -rf backend/venv
rm -rf backend/__pycache__
rm -rf backend/app/__pycache__

# 清理前端
rm -rf frontend/node_modules
rm -rf frontend/dist
rm -f frontend/pnpm-lock.yaml

# 保留源代码和配置
```

### 完全卸载Node.js

**macOS (Homebrew):**
```bash
brew uninstall node
```

**Windows:**
通过 "添加或删除程序" 卸载Node.js

---

## 📞 获取帮助

如果遇到无法解决的问题：

1. 查看详细错误信息
2. 检查 `web/README.md` 的FAQ部分
3. 访问官方文档:
   - FastAPI: https://fastapi.tiangolo.com/
   - React: https://react.dev/
   - Node.js: https://nodejs.org/docs/

---

## ✅ 安装成功检查清单

- [ ] Python 3.8+ 已安装
- [ ] Node.js 18+ 已安装
- [ ] pnpm 已安装
- [ ] 后端依赖已安装
- [ ] 前端依赖已安装
- [ ] 后端服务已启动 (http://localhost:8000)
- [ ] 前端服务已启动 (http://localhost:3000)
- [ ] 浏览器能访问前端界面
- [ ] 前端能连接后端API

全部勾选 = 安装成功！🎉
