# CrossRing 流量生成可视化工具设计文档

## 1. 项目概述

### 1.1 项目背景
当前CrossRing项目使用`generate_data.py`和`generate_d2d_data.py`脚本生成流量文件,存在以下痛点:
- 配置参数硬编码在Python代码中,修改不便
- 节点ID需要手动查阅拓扑文档,容易出错
- 缺少配置验证机制,错误配置难以提前发现
- 生成结果无法直观预览,需要额外工具分析
- 不适合非开发人员使用

### 1.2 项目目标
开发一个**Web可视化流量生成工具**,实现:
- ✅ 零代码编辑,全图形化配置界面
- ✅ 交互式拓扑可视化,点击选择节点
- ✅ 实时参数验证与预估统计
- ✅ 生成结果即时可视化分析
- ✅ 一键生成并下载流量文件

### 1.3 技术选型

| 技术组件 | 选择方案 | 理由 |
|---------|---------|------|
| Web框架 | Streamlit | 纯Python,无需前端开发,自带交互组件,适合快速原型 |
| 可视化库 | Plotly + Matplotlib | Plotly提供交互式图表,Matplotlib绘制拓扑网格 |
| 数据处理 | Pandas | 流量数据的读取、处理、统计分析 |
| 核心逻辑 | 复用现有脚本 | 封装`generate_data.py`和`generate_d2d_data.py`为可调用函数 |

---

## 2. 功能模块设计

### 2.1 模块架构

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Web 界面                      │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ 拓扑可视化  │  │ 流量配置    │  │ 预览统计    │    │
│  │   模块      │  │  管理模块   │  │   模块      │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
├─────────────────────────────────────────────────────────┤
│              流量生成核心引擎 (封装现有逻辑)              │
│  ┌─────────────────────┐  ┌─────────────────────┐      │
│  │ generate_data.py    │  │ generate_d2d_data.py │      │
│  │  (单Die流量生成)    │  │  (D2D多Die流量生成)  │      │
│  └─────────────────────┘  └─────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

### 2.2 核心功能模块

#### 模块1: 拓扑可视化与节点选择

**功能描述:**
- 绘制NoC网格拓扑(支持5x4、4x4等配置)
- 在网格上标注IP位置(GDMA/DDR/L2M/SDMA/CDMA等)
- 交互式节点选择(点击选中/取消,支持多选)
- D2D模式下并排展示多个Die,可视化旋转映射关系

**UI组件:**
```python
# 拓扑类型选择
topo_type = st.selectbox("拓扑类型", ["5x4", "4x4"])

# 模式选择
mode = st.radio("生成模式", ["单Die流量", "D2D多Die流量"])

# 拓扑网格绘制区域
fig = draw_topology_grid(topo_type, selected_nodes, ip_mapping)
st.pyplot(fig)

# 节点选择交互
selected_src = st.multiselect("源节点", node_list)
selected_dst = st.multiselect("目标节点", node_list)
```

**实现要点:**
- 使用Matplotlib绘制网格,不同IP类型用不同颜色标注
- 通过点击事件捕获选中节点(或用multiselect作为替代方案)
- 实时更新选中节点的高亮显示
- D2D模式下展示Die间的映射关系箭头

**数据结构:**
```python
# IP位置映射 (从配置文件加载)
ip_mapping = {
    "gdma": [6, 7, 26, 27],
    "ddr": [12, 13, 32, 33],
    "l2m": [18, 19, 38, 39],
    # ...
}

# 用户选择状态
selected_nodes = {
    "src": [6, 7],      # 源节点列表
    "dst": [12, 13],    # 目标节点列表
    "src_type": "gdma_0",
    "dst_type": "ddr_0",
}
```

---

#### 模块2: 流量配置管理

**功能描述:**
- 图形化配置流量参数(源/目标IP、带宽、burst、请求类型)
- 支持多个流量配置的添加/删除/编辑/复制
- 提供IP预设模板快速选择
- 参数合法性验证(节点范围、带宽范围等)

**UI组件:**
```python
# 配置表单区域
with st.expander("添加新配置", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("源配置")
        src_ip_type = st.text_input("源IP类型", "gdma_0")
        src_nodes = st.multiselect("源节点", range(num_nodes))

    with col2:
        st.subheader("目标配置")
        dst_ip_type = st.text_input("目标IP类型", "ddr_0")
        dst_nodes = st.multiselect("目标节点", range(num_nodes))

    bandwidth = st.slider("带宽 (GB/s)", 0.0, 128.0, 46.08, step=0.01)
    burst = st.number_input("Burst长度", min_value=1, max_value=16, value=4)
    req_type = st.radio("请求类型", ["R", "W"])

    if st.button("添加配置"):
        add_traffic_config(...)

# 配置列表展示
st.subheader("流量配置列表")
for i, config in enumerate(st.session_state.configs):
    col1, col2, col3, col4 = st.columns([3, 3, 1, 1])
    col1.write(f"{config['src_ip_type']} → {config['dst_ip_type']}")
    col2.write(f"{config['bandwidth']} GB/s, Burst={config['burst']}")
    if col3.button("编辑", key=f"edit_{i}"):
        load_config_to_form(config)
    if col4.button("删除", key=f"del_{i}"):
        delete_config(i)
```

**配置模板示例:**
```python
# 预设模板
templates = {
    "所有GDMA→所有DDR (读)": {
        "src_map": {"gdma_0": [6, 7, 26, 27]},
        "dst_map": {"ddr_0": [12, 13, 32, 33]},
        "req_type": "R",
        "bandwidth": 46.08,
        "burst": 4,
    },
    "混合读写负载": [
        # 读配置
        {...},
        # 写配置
        {...},
    ],
}

template = st.selectbox("选择预设模板", list(templates.keys()))
if st.button("应用模板"):
    load_template(templates[template])
```

**参数验证逻辑:**
```python
def validate_config(config):
    errors = []

    # 节点范围检查
    if any(n >= num_nodes for n in config['src_nodes']):
        errors.append(f"源节点ID超出范围 [0, {num_nodes-1}]")

    # 带宽合法性检查
    if config['bandwidth'] <= 0:
        errors.append("带宽必须大于0")

    # Burst长度检查
    if config['burst'] not in [1, 2, 4, 8, 16]:
        errors.append("Burst长度建议为2的幂次")

    # IP类型一致性检查 (可选)
    # ...

    return errors
```

---

#### 模块3: 流量预览与统计

**功能描述:**
- **生成前预估**: 根据配置计算预期请求数、时间分布、节点负载
- **生成后预览**:
  - 时间序列图: 展示请求随时间的分布
  - 源-目标热力图: 显示节点间流量强度
  - 统计表: 总请求数、读写比例、节点使用率
- **数据表格**: 前100条流量数据的详细预览

**生成前预估:**
```python
def estimate_traffic(config, end_time):
    """预估流量统计"""
    duration = 1280  # ns
    total_bandwidth = 128  # GB/s

    # 计算每个时间窗口的传输次数
    transfers_per_window = config['bandwidth'] * duration / (total_bandwidth * config['burst'])

    # 计算总请求数
    num_windows = end_time / duration
    total_requests = transfers_per_window * len(config['src_nodes']) * num_windows

    return {
        "total_requests": int(total_requests),
        "requests_per_ns": total_requests / end_time,
        "src_node_load": total_requests / len(config['src_nodes']),
        "dst_node_load": total_requests / len(config['dst_nodes']),
    }

# UI展示
st.subheader("预估统计")
stats = estimate_traffic(config, end_time)
col1, col2, col3 = st.columns(3)
col1.metric("预计总请求数", f"{stats['total_requests']:,}")
col2.metric("平均请求频率", f"{stats['requests_per_ns']:.2f} req/ns")
col3.metric("源节点平均负载", f"{stats['src_node_load']:.0f} req/node")
```

**生成后可视化:**
```python
import plotly.express as px
import plotly.graph_objects as go

# 1. 时间序列图
def plot_time_series(traffic_df):
    # 按时间窗口统计请求数
    time_bins = pd.cut(traffic_df['timestamp'], bins=50)
    counts = traffic_df.groupby(time_bins).size()

    fig = px.line(x=counts.index.categories.mid, y=counts.values,
                  labels={'x': '时间 (ns)', 'y': '请求数'},
                  title='流量时间分布')
    return fig

st.plotly_chart(plot_time_series(df))

# 2. 源-目标热力图
def plot_heatmap(traffic_df):
    # 统计源-目标对的请求数
    matrix = traffic_df.groupby(['src_pos', 'dst_pos']).size().unstack(fill_value=0)

    fig = px.imshow(matrix,
                    labels=dict(x="目标节点", y="源节点", color="请求数"),
                    title='节点间流量热力图',
                    aspect="auto")
    return fig

st.plotly_chart(plot_heatmap(df))

# 3. 统计表格
st.subheader("统计摘要")
summary = {
    "总请求数": len(df),
    "读请求": len(df[df['req_type'] == 'R']),
    "写请求": len(df[df['req_type'] == 'W']),
    "时间范围": f"{df['timestamp'].min()} - {df['timestamp'].max()} ns",
    "涉及源节点": df['src_pos'].nunique(),
    "涉及目标节点": df['dst_pos'].nunique(),
}
st.table(pd.DataFrame([summary]))

# 4. 数据表格预览
st.subheader("数据预览 (前100条)")
st.dataframe(df.head(100))
```

---

#### 模块4: 一键生成与导出

**功能描述:**
- 验证所有配置的合法性
- 调用核心生成引擎生成流量文件
- 自动生成文件名或允许自定义
- 提供下载按钮,支持浏览器直接下载

**UI组件:**
```python
st.subheader("生成流量文件")

col1, col2 = st.columns([3, 1])
with col1:
    output_filename = st.text_input(
        "输出文件名",
        value=f"traffic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
with col2:
    end_time = st.number_input("仿真时长 (ns)", min_value=100, value=6000)

if st.button("🚀 生成流量文件", type="primary"):
    # 1. 验证所有配置
    all_errors = []
    for i, config in enumerate(st.session_state.configs):
        errors = validate_config(config)
        if errors:
            all_errors.append(f"配置{i+1}: " + ", ".join(errors))

    if all_errors:
        st.error("配置错误:\n" + "\n".join(all_errors))
        st.stop()

    # 2. 调用生成引擎
    with st.spinner("正在生成流量文件..."):
        if mode == "单Die流量":
            output_path = generate_single_die_traffic(
                st.session_state.configs,
                end_time,
                output_filename
            )
        else:
            output_path = generate_d2d_traffic(
                st.session_state.d2d_configs,
                end_time,
                output_filename
            )

    st.success(f"✅ 流量文件生成成功: {output_path}")

    # 3. 提供下载
    with open(output_path, 'r') as f:
        st.download_button(
            label="📥 下载流量文件",
            data=f.read(),
            file_name=output_filename,
            mime="text/plain"
        )

    # 4. 加载并可视化结果
    df = load_traffic_file(output_path)
    st.session_state.generated_traffic = df

    # 展示统计和图表
    plot_time_series(df)
    plot_heatmap(df)
```

---

### 2.3 D2D多Die流量特殊处理

**D2D模式下的额外功能:**

1. **Die数量与拓扑配置**
```python
num_dies = st.number_input("Die数量", min_value=2, max_value=4, value=2)
die_topo = st.selectbox("Die拓扑", ["5x4", "4x4"])
```

2. **Die旋转映射可视化**
```python
# 展示Die1相对Die0的旋转角度
rotation = st.selectbox("Die1旋转角度", [0, 90, 180, 270])

# 计算并展示映射关系
mapping = get_rotated_node_mapping(rows=5, cols=4, rotation=rotation)

# 可视化: 两个Die并排展示,用箭头连接对应节点
fig = plot_die_rotation_mapping(mapping, rotation)
st.pyplot(fig)
```

3. **流量模式选择**
```python
traffic_mode = st.radio("流量模式", ["cross_die", "same_die", "mixed"])

if traffic_mode == "mixed":
    cross_die_ratio = st.slider("跨Die流量比例", 0.0, 1.0, 0.5, step=0.1)
```

4. **源/目标Die选择**
```python
col1, col2 = st.columns(2)
with col1:
    src_die = st.selectbox("源Die", range(num_dies))
with col2:
    dst_die = st.selectbox("目标Die", range(num_dies))

# 根据选择的Die更新拓扑显示
plot_die_topology(src_die, highlight_nodes=selected_src)
plot_die_topology(dst_die, highlight_nodes=selected_dst)
```

---

## 3. 技术实现细节

### 3.1 文件结构

```
scripts/tools/
├── traffic_gen_web.py              # Streamlit主程序入口
├── web_modules/
│   ├── __init__.py
│   ├── topology_visualizer.py      # 拓扑可视化模块
│   ├── config_manager.py           # 配置管理模块
│   ├── traffic_analyzer.py         # 流量分析模块
│   └── generation_engine.py        # 生成引擎封装
├── generate_data.py                # 现有脚本 (重构为可调用函数)
└── generate_d2d_data.py            # 现有脚本 (重构为可调用函数)
```

### 3.2 核心代码重构

#### 重构 `generate_data.py`

**原代码问题:**
- 直接执行生成,没有函数封装
- 配置硬编码在示例函数中

**重构方案:**
```python
# generate_data.py

def generate_traffic_from_configs(
    configs: List[Dict],
    end_time: int,
    output_file: str,
    random_seed: int = 42
) -> str:
    """
    从配置列表生成流量文件

    Args:
        configs: 流量配置列表,每个配置包含:
            - src_map: {"ip_type": [node_list]}
            - dst_map: {"ip_type": [node_list]}
            - speed: 带宽 (GB/s)
            - burst: burst长度
            - req_type: "R" 或 "W"
        end_time: 仿真结束时间 (ns)
        output_file: 输出文件路径
        random_seed: 随机种子

    Returns:
        生成的文件路径
    """
    random.seed(random_seed)

    # 原有的生成逻辑
    all_requests = []
    for config in configs:
        requests = generate_single_config(**config, end_time=end_time)
        all_requests.extend(requests)

    # 排序并写入文件
    all_requests.sort(key=lambda x: x[0])
    with open(output_file, 'w') as f:
        for req in all_requests:
            f.write(','.join(map(str, req)) + '\n')

    return output_file

# 保留原有示例函数供测试
def generate_example_traffic():
    configs = [...]  # 原有配置
    generate_traffic_from_configs(configs, END_TIME=6000, OUTPUT_FILE="...")
```

#### 重构 `generate_d2d_data.py`

**重构方案:**
```python
# generate_d2d_data.py

class D2DTrafficGenerator:
    """D2D流量生成器 (保持现有类结构)"""

    def generate_from_configs(
        self,
        filename: str,
        traffic_configs: List[Dict],
        traffic_mode: str = "cross_die",
        end_time: int = 6000,
        random_seed: int = 42,
        **kwargs
    ) -> str:
        """
        从配置列表生成D2D流量文件

        Args:
            filename: 输出文件路径
            traffic_configs: 流量配置列表
            traffic_mode: "cross_die" | "same_die" | "mixed"
            end_time: 仿真时长
            random_seed: 随机种子
            **kwargs: mixed模式下的额外参数 (cross_die_ratio等)

        Returns:
            生成的文件路径
        """
        # 调用现有的 generate_traffic_file 方法
        return self.generate_traffic_file(
            filename=filename,
            traffic_configs=traffic_configs,
            traffic_mode=traffic_mode,
            end_time=end_time,
            random_seed=random_seed,
            **kwargs
        )
```

### 3.3 Streamlit 状态管理

**使用 `st.session_state` 管理全局状态:**

```python
# traffic_gen_web.py

def init_session_state():
    """初始化会话状态"""
    if 'configs' not in st.session_state:
        st.session_state.configs = []

    if 'selected_src_nodes' not in st.session_state:
        st.session_state.selected_src_nodes = []

    if 'selected_dst_nodes' not in st.session_state:
        st.session_state.selected_dst_nodes = []

    if 'generated_traffic' not in st.session_state:
        st.session_state.generated_traffic = None

    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = "单Die流量"

# 主程序入口
def main():
    st.set_page_config(page_title="CrossRing 流量生成器", layout="wide")
    init_session_state()

    st.title("🚦 CrossRing 流量生成可视化工具")

    # 模式选择
    mode = st.sidebar.radio("生成模式", ["单Die流量", "D2D多Die流量"])
    st.session_state.current_mode = mode

    # 根据模式渲染不同界面
    if mode == "单Die流量":
        render_single_die_mode()
    else:
        render_d2d_mode()

if __name__ == "__main__":
    main()
```

### 3.4 拓扑可视化实现

**方案1: 静态Matplotlib图 + multiselect组件**
```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_topology_grid(topo_type, ip_mapping, selected_src, selected_dst):
    """绘制NoC拓扑网格"""
    rows, cols = map(int, topo_type.split('x'))

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制网格
    for i in range(rows):
        for j in range(cols):
            node_id = i * cols + j

            # 确定节点颜色
            color = 'white'
            if node_id in selected_src:
                color = 'lightblue'
            elif node_id in selected_dst:
                color = 'lightcoral'

            # 检查是否是IP节点
            for ip_type, nodes in ip_mapping.items():
                if node_id in nodes:
                    color = get_ip_color(ip_type)
                    break

            # 绘制节点方块
            rect = patches.Rectangle((j, rows-1-i), 1, 1,
                                     linewidth=1, edgecolor='black',
                                     facecolor=color)
            ax.add_patch(rect)

            # 添加节点ID标签
            ax.text(j+0.5, rows-1-i+0.5, str(node_id),
                   ha='center', va='center', fontsize=10)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')

    return fig

def get_ip_color(ip_type):
    """IP类型颜色映射"""
    colors = {
        'gdma': '#FFD700',  # 金色
        'ddr': '#87CEEB',   # 天蓝色
        'l2m': '#98FB98',   # 浅绿色
        'sdma': '#FFB6C1',  # 浅粉色
        'cdma': '#DDA0DD',  # 梅红色
    }
    # 提取IP基础类型 (去掉_0, _1等后缀)
    base_type = ip_type.split('_')[0]
    return colors.get(base_type, 'white')

# 在Streamlit中使用
fig = draw_topology_grid("5x4", ip_mapping,
                         st.session_state.selected_src_nodes,
                         st.session_state.selected_dst_nodes)
st.pyplot(fig)

# 使用multiselect进行节点选择
st.session_state.selected_src_nodes = st.multiselect(
    "选择源节点",
    range(20),  # 5x4=20个节点
    default=st.session_state.selected_src_nodes
)
```

**方案2: Plotly交互式图 (进阶版)**
```python
import plotly.graph_objects as go

def draw_interactive_topology(topo_type, ip_mapping):
    """绘制可交互的拓扑图"""
    rows, cols = map(int, topo_type.split('x'))

    # 准备节点数据
    node_x, node_y, node_text, node_colors = [], [], [], []

    for i in range(rows):
        for j in range(cols):
            node_id = i * cols + j
            node_x.append(j)
            node_y.append(rows - 1 - i)
            node_text.append(f"Node {node_id}")

            # 确定颜色
            color = 'lightgray'
            for ip_type, nodes in ip_mapping.items():
                if node_id in nodes:
                    color = get_ip_color(ip_type)
                    node_text[-1] = f"Node {node_id}<br>{ip_type}"
                    break
            node_colors.append(color)

    # 创建Plotly图
    fig = go.Figure(data=[go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=30, color=node_colors, line=dict(width=2, color='black')),
        text=[str(i) for i in range(len(node_x))],
        textposition="middle center",
        hovertext=node_text,
        hoverinfo="text",
    )])

    fig.update_layout(
        title="NoC拓扑结构",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=600, height=500,
        hovermode='closest'
    )

    return fig

st.plotly_chart(draw_interactive_topology("5x4", ip_mapping))
```

**推荐方案**: 先使用**方案1**(Matplotlib + multiselect),简单快速实现;后续有需求再升级到**方案2**的交互式图。

---

## 4. 实施计划

### 4.1 开发阶段

#### 阶段1: 基础框架搭建 (1-2天)
**目标**: 完成Streamlit基本界面和项目结构

**任务清单**:
- [x] 创建文件结构(`traffic_gen_web.py`及`web_modules/`)
- [x] 搭建Streamlit主程序框架
- [x] 实现模式选择(单Die/D2D)切换
- [x] 实现基础拓扑网格绘制(Matplotlib)
- [x] 完成会话状态管理(`st.session_state`初始化)

**验收标准**:
- 可运行`streamlit run traffic_gen_web.py`
- 界面显示拓扑网格(静态,无交互)
- 模式切换正常工作

---

#### 阶段2: 单Die流量配置功能 (2-3天)
**目标**: 完成单Die流量的完整配置→生成→预览流程

**任务清单**:
- [x] 实现配置表单UI(源/目标IP、带宽、burst等)
- [x] 实现配置列表管理(添加/删除/编辑)
- [x] 重构`generate_data.py`为可调用函数
- [x] 连接配置表单与生成引擎
- [x] 实现参数验证逻辑
- [x] 实现生成前预估统计

**验收标准**:
- 可通过表单添加多个流量配置
- 点击"生成"按钮能生成正确的流量文件
- 参数错误时显示友好提示
- 预估统计数据准确

---

#### 阶段3: 节点选择与可视化优化 (1-2天)
**目标**: 优化节点选择体验,完善拓扑可视化

**任务清单**:
- [x] 实现IP位置映射加载(从配置或硬编码)
- [x] 在拓扑图上标注IP类型(不同颜色)
- [x] 实现节点选择后的拓扑高亮显示
- [x] 添加IP预设模板功能
- [x] (可选) 升级为Plotly交互式图

**验收标准**:
- 拓扑图清晰展示所有IP位置
- 选中节点后拓扑图实时更新高亮
- 模板功能可一键填充配置

---

#### 阶段4: 流量结果分析与预览 (2-3天)
**目标**: 完成生成后的可视化分析功能

**任务清单**:
- [x] 实现流量文件解析(读取生成的txt文件)
- [x] 绘制时间序列图(请求随时间分布)
- [x] 绘制源-目标热力图
- [x] 生成统计表格(总请求数、读写比例等)
- [x] 实现数据表格预览(前100条)
- [x] 添加文件下载功能

**验收标准**:
- 生成后立即显示可视化图表
- 图表交互流畅(缩放、hover等)
- 统计数据准确
- 可下载生成的流量文件

---

#### 阶段5: D2D多Die流量支持 (2-3天)
**目标**: 扩展支持D2D多Die流量生成

**任务清单**:
- [x] 实现多Die拓扑并排展示
- [x] 添加Die数量/旋转角度配置
- [x] 可视化Die旋转映射关系
- [x] 重构`generate_d2d_data.py`为可调用函数
- [x] 实现D2D配置表单(源/目标Die选择)
- [x] 添加流量模式选择(cross_die/same_die/mixed)
- [x] 连接D2D生成引擎

**验收标准**:
- 可配置并生成跨Die流量
- Die旋转映射可视化清晰
- 生成的D2D流量文件格式正确

---

#### 阶段6: 优化与完善 (1-2天)
**目标**: 提升用户体验,完善文档

**任务清单**:
- [x] 添加侧边栏使用说明/帮助文档
- [x] 优化界面布局和样式
- [x] 添加错误处理和异常提示
- [x] 性能优化(大数据量处理)
- [x] 编写用户使用手册(markdown)
- [x] 测试各种边界情况

**验收标准**:
- 界面美观,操作流畅
- 异常情况有友好提示
- 使用文档清晰完整
- 经过多场景测试无明显bug

---

### 4.2 总体时间估算

| 阶段 | 预计工作量 | 说明 |
|-----|----------|------|
| 阶段1: 基础框架 | 1-2天 | Streamlit基础搭建较快 |
| 阶段2: 单Die功能 | 2-3天 | 核心功能,需仔细测试 |
| 阶段3: 节点选择优化 | 1-2天 | 可视化细节打磨 |
| 阶段4: 结果分析 | 2-3天 | 图表开发工作量较大 |
| 阶段5: D2D支持 | 2-3天 | 复用单Die逻辑可加速 |
| 阶段6: 优化完善 | 1-2天 | 迭代优化 |
| **总计** | **9-15天** | 根据实际进度调整 |

### 4.3 里程碑

- **MVP版本** (阶段1+2+3): 完成单Die流量的完整功能,可投入使用
- **完整版本** (阶段1-5): 支持D2D多Die流量
- **优化版本** (阶段1-6): 用户体验打磨,文档完善

---

## 5. 依赖与环境

### 5.1 Python依赖

```bash
# 核心依赖
pip install streamlit>=1.28.0
pip install plotly>=5.17.0
pip install pandas>=2.0.0
pip install matplotlib>=3.7.0
pip install numpy>=1.24.0

# 可选依赖 (项目已有)
# scipy, networkx, seaborn等
```

### 5.2 启动方式

```bash
# 方式1: 直接启动
cd /Users/lixiang/Documents/工作/code/CrossRing
streamlit run scripts/tools/traffic_gen_web.py

# 方式2: 指定端口
streamlit run scripts/tools/traffic_gen_web.py --server.port 8502

# 方式3: 自动打开浏览器
streamlit run scripts/tools/traffic_gen_web.py --server.headless false
```

### 5.3 访问地址

- 本地访问: `http://localhost:8501`
- 局域网访问: `http://<your-ip>:8501` (需配置`--server.address 0.0.0.0`)

---

## 6. 未来扩展方向

### 6.1 短期优化 (v1.1)
- [ ] 配置导入/导出功能(JSON格式)
- [ ] 配置版本历史管理
- [ ] 更多流量模式模板
- [ ] 批量生成多个场景

### 6.2 中期扩展 (v2.0)
- [ ] 集成仿真运行功能(一键生成+仿真)
- [ ] 仿真结果可视化对比
- [ ] 流量回放动画(时间轴动画展示)
- [ ] 多拓扑类型支持(Mesh、Torus等)

### 6.3 长期愿景 (v3.0)
- [ ] 在线协作功能(多人共享配置)
- [ ] AI辅助配置推荐(基于历史数据)
- [ ] 流量生成策略优化(遗传算法等)
- [ ] 与其他NoC工具集成

---

## 7. 风险与挑战

### 7.1 技术风险

| 风险项 | 影响 | 缓解措施 |
|-------|------|---------|
| Streamlit性能限制 | 大数据量时界面卡顿 | 使用分页/懒加载,限制预览数据量 |
| 拓扑图交互复杂 | 节点选择体验不佳 | 先用multiselect替代,后续升级交互式图 |
| 现有代码耦合度高 | 重构困难 | 最小化修改,优先封装而非重写 |
| 跨平台兼容性 | Windows/Mac/Linux差异 | 使用相对路径,测试多平台 |

### 7.2 时间风险

- **风险**: 功能需求膨胀导致延期
- **缓解**: 严格按阶段交付,MVP优先

### 7.3 用户接受度风险

- **风险**: 用户习惯命令行方式,不愿切换
- **缓解**: 提供CLI模式兼容,Web工具作为补充而非替代

---

## 8. 成功标准

### 8.1 功能完整性
- ✅ 支持单Die和D2D多Die流量生成
- ✅ 可视化拓扑和节点选择
- ✅ 配置管理和参数验证
- ✅ 生成结果可视化分析

### 8.2 易用性
- ✅ 零代码编辑,全图形化操作
- ✅ 新手10分钟内上手
- ✅ 配置错误时有清晰提示

### 8.3 正确性
- ✅ 生成的流量文件与原脚本结果一致
- ✅ 参数验证准确无误
- ✅ 统计数据与实际吻合

### 8.4 性能
- ✅ 界面响应时间 < 2秒
- ✅ 生成10000条流量 < 5秒
- ✅ 可视化渲染 < 3秒

---

## 9. 附录

### 9.1 参考资料
- Streamlit官方文档: https://docs.streamlit.io
- Plotly官方文档: https://plotly.com/python/
- 现有脚本: `generate_data.py`, `generate_d2d_data.py`

### 9.2 关键设计决策

| 决策点 | 选择 | 理由 |
|-------|------|------|
| Web框架 | Streamlit | 纯Python,快速开发,适合数据可视化 |
| 节点选择方式 | multiselect → Plotly点击 | 先简单实现,后续优化 |
| 配置存储 | session_state (内存) → JSON文件 | MVP阶段用内存,后续持久化 |
| 生成引擎 | 封装现有脚本 | 避免重复开发,保证一致性 |

### 9.3 常见问题

**Q: 为什么选择Streamlit而不是Flask/Django?**
A: Streamlit专为数据科学和可视化设计,无需前端代码,开发效率高,适合快速原型。

**Q: 生成的流量文件存储在哪里?**
A: 默认存储在`traffic/`目录,用户可通过界面自定义路径,也可直接下载。

**Q: 是否支持命令行参数启动?**
A: 暂不支持,但可通过URL参数传递配置(Streamlit的query_params功能)。

**Q: 如何处理大规模流量(百万级请求)?**
A: 使用分批生成 + 懒加载预览,避免一次性加载全部数据。

---

## 10. 总结

本设计文档提出了一个**全图形化、零代码**的CrossRing流量生成可视化工具,通过Web界面解决了现有脚本配置繁琐、易出错、不直观的问题。

**核心价值**:
1. **降低使用门槛**: 非开发人员也能轻松配置流量
2. **减少配置错误**: 可视化选择节点,实时验证参数
3. **提升开发效率**: 即时预览结果,快速迭代测试
4. **保持一致性**: 复用现有生成逻辑,确保输出正确

**实施建议**:
- 采用**MVP迭代**方式,先完成单Die核心功能
- 每个阶段交付可用版本,及时收集反馈
- 保持代码模块化,便于后续扩展

该工具将成为CrossRing项目工作流中的重要一环,显著提升流量生成的效率和准确性。
