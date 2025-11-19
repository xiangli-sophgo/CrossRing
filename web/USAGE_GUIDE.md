# CrossRing Web 流量生成工具使用指南

## 🚀 快速开始

### 1. 启动服务

在 `web/` 目录下执行：

```bash
# macOS/Linux
./start-dev.sh

# Windows
start-dev.bat
```

服务启动后：
- **后端 API**: http://localhost:8000
- **前端界面**: http://localhost:3002 (端口可能自动调整)
- **API 文档**: http://localhost:8000/api/docs

### 2. VS Code 快捷启动

1. 打开 VS Code 命令面板 (`Cmd+Shift+P` / `Ctrl+Shift+P`)
2. 选择 `Tasks: Run Task`
3. 选择 `CrossRing Web: Start All`

## 📋 使用流程

### 第一步：选择拓扑

1. 在页面顶部选择拓扑类型（如 `5x4`）
2. 系统会自动加载并可视化拓扑结构
3. 可以点击节点查看详细信息

### 第二步：挂载 IP 节点

在 **IP节点挂载** 面板中：

#### 单个挂载
- **节点ID**: 输入要挂载的节点ID（支持多个，用逗号分隔）
  - 示例：`0,1,2`
- **IP类型**: 输入IP类型，格式为 `类型_编号`
  - 示例：`gdma_0`、`npu_0`、`ddr_0`

#### 批量挂载
- **节点范围**: 支持两种格式
  - 范围格式：`0-3` （表示节点 0, 1, 2, 3）
  - 逗号分隔：`1,3,5` （表示节点 1, 3, 5）
- **IP类型前缀**: 输入IP类型前缀（如 `gdma`）
  - 系统会自动为每个节点分配递增编号（`gdma_0`, `gdma_1`, ...）

#### 查看和管理
- 表格中显示所有已挂载的IP
- 可以删除单个挂载
- 可以清空所有挂载

### 第三步：配置流量

在 **流量配置** 面板中：

1. **选择流量模式**
   - **NoC 模式**: 片内通信
   - **D2D 模式**: 跨Die通信

2. **添加流量配置**
   - **源IP**: 已挂载的源IP类型（如 `gdma_0`）
   - **目标IP**: 已挂载的目标IP类型（如 `ddr_0`）
   - **速度**: 流量速度，单位 GB/s（范围：0.1 - 128）
   - **Burst长度**: 每次传输的数据块大小（范围：1 - 16）
   - **请求类型**:
     - `R` (读)
     - `W` (写)
   - **结束时间**: 流量生成的结束时间，单位 ns（范围：100 - 100000）

3. **查看配置列表**
   - 表格显示所有已添加的流量配置
   - 可以删除单个配置
   - 可以清空所有配置

### 第四步：生成流量

1. 确保已添加至少一个流量配置
2. 点击 **生成流量** 按钮
3. 系统会：
   - 根据配置生成流量数据
   - 显示生成进度和耗时
   - 自动下载生成的流量文件

## 📊 输出文件格式

### NoC 模式（7字段）
```csv
timestamp,src_pos,src_type,dst_pos,dst_type,req_type,burst
0,(0,0),gdma_0,(1,1),ddr_0,R,4
1280,(0,0),gdma_0,(1,2),ddr_1,W,8
```

**字段说明**:
- `timestamp`: 时间戳（ns）
- `src_pos`: 源节点坐标
- `src_type`: 源IP类型
- `dst_pos`: 目标节点坐标
- `dst_type`: 目标IP类型
- `req_type`: 请求类型（R/W）
- `burst`: Burst长度

### D2D 模式（9字段）
```csv
timestamp,src_pos,src_type,dst_pos,dst_type,req_type,burst,src_die,dst_die
0,(0,0),gdma_0,(1,1),ddr_0,R,4,0,1
```

额外字段：
- `src_die`: 源Die ID
- `dst_die`: 目标Die ID

## 🔧 高级功能

### API 接口

系统提供完整的 RESTful API，访问 http://localhost:8000/api/docs 查看详细文档。

主要接口：

#### IP 挂载
- `POST /api/ip-mount/` - 挂载IP
- `POST /api/ip-mount/batch` - 批量挂载
- `GET /api/ip-mount/{topology}` - 获取挂载列表
- `DELETE /api/ip-mount/{topology}/nodes/{node_id}` - 删除挂载

#### 流量配置
- `POST /api/traffic/config/` - 创建配置
- `POST /api/traffic/config/batch` - 批量创建
- `GET /api/traffic/config/{topology}/{mode}` - 获取配置列表
- `DELETE /api/traffic/config/{topology}/{mode}/{config_id}` - 删除配置

#### 流量生成
- `POST /api/traffic/generate/` - 生成流量
- `GET /api/traffic/generate/download/{filename}` - 下载文件
- `GET /api/traffic/generate/list` - 列出已生成文件

### 配置持久化

所有配置自动保存到 `web/backend/data/` 目录：
- IP挂载配置：`data/ip_mounts/{topology}.json`
- 流量配置：`data/traffic_configs/{topology}_{mode}.json`
- 生成的流量文件：`data/generated_traffic/`

## 💡 使用技巧

1. **批量操作**
   - 批量挂载IP可以快速配置大量节点
   - 批量流量配置可以为多个IP对创建相同的流量模式

2. **配置复用**
   - 配置会自动保存，切换拓扑后会自动加载对应配置
   - 可以为不同拓扑维护独立的配置集

3. **流量验证**
   - 系统会自动验证IP是否已挂载
   - 会检查参数范围的合法性
   - 错误信息会清晰提示问题所在

4. **性能优化**
   - 使用批量操作可以减少网络请求
   - 生成的流量文件按时间和源位置排序
   - 支持大规模流量生成（数十万行）

## ❓ 常见问题

### Q: 生成流量时提示 "源IP未挂载"？
A: 请先在 IP节点挂载 面板中挂载相应的IP。

### Q: 如何修改已有的流量配置？
A: 目前需要删除旧配置，然后添加新配置。

### Q: 生成的文件保存在哪里？
A: 保存在 `web/backend/data/generated_traffic/` 目录，并会自动触发浏览器下载。

### Q: 可以同时运行多个流量生成吗？
A: 可以，每次生成都会创建带时间戳的独立文件。

### Q: 如何清空所有配置？
A: 每个面板都有 "清空" 按钮，可以一键清除所有配置。

## 📚 相关文档

- [安装指南](INSTALL.md)
- [快速开始](QUICKSTART.md)
- [VS Code Tasks](VSCODE_TASKS.md)
- [API 文档](http://localhost:8000/api/docs)

## 🐛 问题反馈

如遇到问题，请检查：
1. 后端服务是否正常运行（http://localhost:8000/）
2. 浏览器控制台是否有错误信息
3. 查看 `web/backend/data/` 目录权限是否正确
