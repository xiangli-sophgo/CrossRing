# Traffic Gene - CrossRing 流量生成工具

## 📋 简介

Traffic Gene 是 CrossRing 项目的流量生成工具包,提供单Die和D2D(Die-to-Die)两种流量生成模式,支持Web界面和编程API两种使用方式。

### 核心功能

- ✅ **单Die流量生成** - 7字段格式流量文件生成
- ✅ **D2D流量生成** - 9字段格式跨Die流量生成
- ✅ **流量拆分** - 按源IP自动拆分流量文件
- ✅ **节点选择** - 支持拓扑图点击和节点ID输入两种方式
- ✅ **实时预估** - 生成前统计预估(请求数、带宽等)
- ✅ **结果分析** - 可视化图表分析(时间序列、热力图等)

---

## 🚀 快速开始

### 方式1: Web界面(推荐)

#### 启动命令

```bash
cd C:\Users\xiang\Documents\code\CrossRing
streamlit run scripts/tools/traffic_gene_web.py
```

#### 使用流程

1. **选择流量模式** - 侧边栏选择"单Die"或"D2D"模式
2. **配置拓扑** - 选择拓扑类型(5x4/4x4)和仿真时长
3. **选择节点**
   - 方式1: 在拓扑图上点击节点
   - 方式2: 在节点ID输入框输入(如: `6,7,26-27`)
4. **配置流量参数**
   - D2D模式: 额外配置源/目标Die编号
   - 输入IP类型、带宽、Burst长度、请求类型
5. **添加配置** - 点击"添加配置"按钮
6. **生成流量文件**
   - 可选: 勾选"拆分流量文件"选项
   - 点击"生成流量文件"按钮
7. **查看结果** - 查看统计图表和分析结果

---

### 方式2: 编程API

#### 单Die流量生成

```python
from src.traffic_process.traffic_gene.generation_engine import (
    generate_traffic_from_configs
)

# 配置列表
configs = [{
    "src_map": {"gdma_0": [6, 7]},  # 源IP映射
    "dst_map": {"ddr_0": [12, 13]},  # 目标IP映射
    "speed": 46.08,                  # 带宽 (GB/s)
    "burst": 4,                      # Burst长度
    "req_type": "R"                  # 请求类型 ("R"或"W")
}]

# 生成流量
file_path, df = generate_traffic_from_configs(
    configs=configs,
    end_time=6000,
    output_file="traffic/output.txt",
    return_dataframe=True
)

print(f"文件已生成: {file_path}")
print(f"数据预览:\n{df.head()}")
```

#### D2D流量生成

```python
from src.traffic_process.traffic_gene.generation_engine import (
    generate_d2d_traffic_from_configs
)

# D2D配置列表
configs = [{
    "src_die": 0,                    # 源Die编号
    "dst_die": 1,                    # 目标Die编号
    "src_map": {"gdma_0": [6]},
    "dst_map": {"ddr_0": [12]},
    "speed": 128.0,
    "burst": 4,
    "req_type": "R"
}]

# 生成D2D流量
file_path, df = generate_d2d_traffic_from_configs(
    configs=configs,
    end_time=6000,
    output_file="traffic/d2d_output.txt",
    return_dataframe=True
)
```

#### 流量拆分

```python
from src.traffic_process.traffic_gene.generation_engine import (
    split_traffic_by_source
)

# 拆分流量文件
result = split_traffic_by_source(
    input_file="traffic/output.txt",
    output_dir="traffic/split_output",
    num_col=4,   # 拓扑列数
    num_row=5,   # 拓扑行数
    verbose=True
)

print(f"拆分完成: {result['total_sources']} 个源IP")
print(f"输出目录: {result['output_dir']}")

# 查看拆分文件列表
for file_info in result['files']:
    print(f"  {file_info['filename']}: {file_info['count']} 条请求")
```

#### 节点ID解析

```python
from src.traffic_process.traffic_gene.topology_visualizer import (
    TopologyVisualizer
)

visualizer = TopologyVisualizer(topo_type="5x4")

# 解析节点ID输入
node_ids = visualizer.parse_node_ids("6,7,26-27")
print(f"解析结果: {node_ids}")  # [6, 7, 26, 27]

# 支持范围表达式
node_ids = visualizer.parse_node_ids("6-10,12")
print(f"解析结果: {node_ids}")  # [6, 7, 8, 9, 10, 12]
```

---

## 📊 输出格式

### 单Die格式(7字段)

```
inject_time, src_node, src_ip, dst_node, dst_ip, req_type, burst_length
```

**示例:**
```
0,6,gdma_0,12,ddr_0,R,4
160,7,gdma_0,13,ddr_0,R,4
```

### D2D格式(9字段)

```
inject_time, src_die, src_node, src_ip, dst_die, dst_node, dst_ip, req_type, burst_length
```

**示例:**
```
0,0,6,gdma_0,1,12,ddr_0,R,4
160,0,7,gdma_0,1,13,ddr_0,R,4
```

### 拆分格式(目标坐标格式)

```
inject_time, (p{dst_ip_index},x{x},y{y}), req_type, burst_length
```

**示例:**
```
0,(p0,x0,y3),R,4
160,(p0,x1,y3),R,4
```

**说明:**
- `p{dst_ip_index}`: 目标IP索引(从IP名称提取,如 `ddr_0` → `p0`)
- `x{x},y{y}`: 目标节点坐标(左下角为原点)

---

## 🔧 配置参数

### 拓扑配置

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `topo_type` | str | 拓扑类型 | `"5x4"` |
| `num_col` | int | 拓扑列数 | 4 |
| `num_row` | int | 拓扑行数 | 5 |

**支持拓扑:**
- `5x4` - 5行4列(20个节点)
- `4x4` - 4行4列(16个节点)

### 流量参数

| 参数 | 类型 | 说明 | 范围 |
|------|------|------|------|
| `speed` | float | 带宽 (GB/s) | 0.1 ~ 128.0 |
| `burst` | int | Burst长度 | 1, 2, 4, 8, 16 |
| `req_type` | str | 请求类型 | "R" (读) 或 "W" (写) |
| `end_time` | int | 仿真时长 (ns) | 100 ~ 100000 |

### IP映射(5x4拓扑)

| IP类型 | 节点位置 |
|--------|----------|
| `gdma` | [6, 7, 26, 27] |
| `ddr`  | [12, 13, 32, 33] |
| `l2m`  | [18, 19, 38, 39] |
| `sdma` | [0, 1, 20, 21] |
| `cdma` | [14, 15, 34] |

---

## 📁 目录结构

```
src/traffic_process/traffic_gene/
├── __init__.py                 # 包初始化
├── config_manager.py           # 配置管理
├── generation_engine.py        # 生成引擎(含拆分功能)
├── topology_visualizer.py      # 拓扑可视化(含节点ID解析)
├── traffic_analyzer.py         # 流量分析
└── README.md                   # 本文档

scripts/tools/
└── traffic_gene_web.py         # Web应用入口

src/traffic_process/
└── split_traffic.py            # 拆分工具(命令行版,向后兼容)
```

---

## 💡 使用示例

### 示例1: 基本流量生成

```python
from src.traffic_process.traffic_gene.generation_engine import (
    generate_traffic_from_configs,
    get_default_ip_mappings
)

# 获取默认IP映射
ip_mappings = get_default_ip_mappings("5x4")

# 配置: GDMA → DDR 读请求
configs = [{
    "src_map": {"gdma_0": ip_mappings["gdma"]},
    "dst_map": {"ddr_0": ip_mappings["ddr"]},
    "speed": 46.08,
    "burst": 4,
    "req_type": "R"
}]

# 生成流量
generate_traffic_from_configs(
    configs=configs,
    end_time=6000,
    output_file="traffic/gdma_to_ddr_read.txt"
)
```

### 示例2: 多配置合并生成

```python
configs = [
    # 读请求: GDMA → DDR
    {
        "src_map": {"gdma_0": [6, 7]},
        "dst_map": {"ddr_0": [12, 13]},
        "speed": 46.08,
        "burst": 4,
        "req_type": "R"
    },
    # 写请求: GDMA → L2M
    {
        "src_map": {"gdma_1": [26, 27]},
        "dst_map": {"l2m_0": [18, 19]},
        "speed": 32.0,
        "burst": 8,
        "req_type": "W"
    }
]

# 合并生成
generate_traffic_from_configs(
    configs=configs,
    end_time=12000,
    output_file="traffic/mixed_traffic.txt"
)
```

### 示例3: 生成并自动拆分

```python
from src.traffic_process.traffic_gene.generation_engine import (
    generate_traffic_from_configs,
    split_traffic_by_source
)

# 1. 生成流量
file_path, _ = generate_traffic_from_configs(
    configs=configs,
    end_time=6000,
    output_file="traffic/output.txt",
    return_dataframe=False
)

# 2. 自动拆分
result = split_traffic_by_source(
    input_file=file_path,
    output_dir="traffic/split_output",
    num_col=4,
    num_row=5
)

print(f"拆分完成,共 {result['total_sources']} 个源IP文件")
```

### 示例4: D2D跨Die流量

```python
from src.traffic_process.traffic_gene.generation_engine import (
    generate_d2d_traffic_from_configs
)

# 跨Die流量: Die0 GDMA → Die1 DDR
configs = [{
    "src_die": 0,
    "dst_die": 1,
    "src_map": {"gdma_0": [6, 7]},
    "dst_map": {"ddr_0": [12, 13]},
    "speed": 128.0,
    "burst": 4,
    "req_type": "R"
}]

generate_d2d_traffic_from_configs(
    configs=configs,
    end_time=6000,
    output_file="traffic/d2d_cross_die.txt"
)
```

---

## ⚙️ 节点ID输入格式

支持以下格式:

| 格式 | 说明 | 示例 | 解析结果 |
|------|------|------|----------|
| 单个节点 | 单个节点ID | `6` | `[6]` |
| 逗号分隔 | 多个节点ID | `6,7,26,27` | `[6, 7, 26, 27]` |
| 范围表达 | 节点ID范围 | `6-7` | `[6, 7]` |
| 混合格式 | 逗号+范围混合 | `6-7,26-27` | `[6, 7, 26, 27]` |
| 复杂混合 | 多种格式混合 | `6,8-10,12` | `[6, 8, 9, 10, 12]` |

**错误处理:**
- 节点ID超出范围 → 抛出 `ValueError`
- 范围起始大于结束 → 抛出 `ValueError`
- 格式错误 → 抛出 `ValueError`

---

## 🎨 可视化功能

### Web界面图表

1. **时间序列图** - 请求数随时间变化趋势
2. **读写分布饼图** - 读/写请求占比
3. **流量热力图** - 源-目标节点流量分布
4. **带宽分布柱状图** - 各配置带宽分布

### 统计指标

- 总请求数 / 读请求数 / 写请求数
- 时间范围(起止时间)
- 唯一源/目标节点数
- 平均Burst长度

---

## 🔍 常见问题

### Q1: 拆分功能支持D2D格式吗?

**A:** 目前拆分功能仅支持单Die格式(7字段),不支持D2D格式(9字段)。Web界面会在D2D模式下自动禁用拆分选项。

### Q2: 节点ID输入支持空格吗?

**A:** 支持。解析器会自动去除空格,如 `6, 7, 26 - 27` 与 `6,7,26-27` 等效。

### Q3: 如何修改默认IP映射?

**A:** 编辑 `generation_engine.py` 中的 `get_default_ip_mappings()` 函数:

```python
def get_default_ip_mappings(topo_type="5x4"):
    if topo_type == "5x4":
        return {
            "gdma": [6, 7, 26, 27],  # 修改这里
            "ddr": [12, 13, 32, 33],
            # ...
        }
```

### Q4: 生成的流量是随机的吗?

**A:** 时间点是均匀分布的,但目标节点是从目标映射中随机选择的。可以通过 `random_seed` 参数控制随机性:

```python
generate_traffic_from_configs(
    configs=configs,
    end_time=6000,
    random_seed=42,  # 固定种子,结果可复现
    output_file="traffic/output.txt"
)
```

### Q5: 如何计算请求数?

**A:** 请求数计算公式:

```
每个周期请求数 = speed * duration / (total_bandwidth * burst)
总请求数 = 每个周期请求数 × 源节点数 × (end_time / duration)
```

其中:
- `speed`: 配置带宽 (GB/s)
- `duration`: 时间窗口 (默认1280ns)
- `total_bandwidth`: 总带宽基准 (默认128 GB/s)
- `burst`: Burst长度

---

## 📝 更新日志

### v2.0.0 (2025-01-18)

**重大变更:**
- 🔄 模块重命名: `web_modules` → `traffic_gene`
- 🔄 Web入口重命名: `traffic_gen_web.py` → `traffic_gene_web.py`

**新增功能:**
- ✨ 集成流量拆分功能到 `generation_engine.py`
- ✨ 节点ID解析功能(支持范围表达式)
- ✨ D2D流量生成支持(9字段格式)
- ✨ Web界面流量模式切换(单Die/D2D)
- ✨ Web界面可选流量拆分
- ✨ 拆分结果展示

**优化改进:**
- 🎨 节点选择支持双模式(点击+输入)
- 📊 配置预估显示Die信息(D2D模式)
- 📝 完善文档和使用说明

**向后兼容:**
- ✅ 保留 `split_traffic.py` 命令行工具
- ✅ 所有导入路径已更新

---

## 📞 支持

如有问题或建议,请提交Issue到项目仓库。

**相关文档:**
- [CrossRing 项目文档](../../README.md)
- [D2D通信设计文档](../../../docs/ordering_preservation_design_CN.md)
- [流量处理文档](../README.md)

---

**最后更新:** 2025-01-18
