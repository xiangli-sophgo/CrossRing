# FIFO使用率热力图功能总结

## 功能概述

成功实现了FIFO使用率交互式热力图功能，用于可视化分析CrossRing NoC中各节点FIFO缓冲区的使用情况。

## 实现文件

### 核心模块
- `src/core/fifo_heatmap_visualizer.py` - FIFO热力图可视化核心模块
  - `FIFOUtilizationCollector` - 数据收集器类
  - `FIFOHeatmapVisualizer` - 可视化器类
  - `create_fifo_heatmap()` - 便捷函数

### 集成修改
- `src/core/d2d_model.py` - D2D模型集成
  - 添加`fifo_utilization_heatmap`参数到`setup_result_analysis()`
  - 在结果处理流程中自动生成热力图

- `src/core/base_model.py` - 基类修改
  - 添加`plot_fifo_heatmap`参数到`setup_result_analysis()`

- `src/core/base_model_v2.py` - V2基类修改
  - 添加`plot_fifo_heatmap`参数到`setup_result_analysis()`

### 测试和文档
- `test/test_fifo_heatmap.py` - 测试脚本
- `scripts/example_fifo_heatmap.py` - 使用示例
- `docs/FIFO_Heatmap_Usage.md` - 使用说明
- `docs/FIFO_Heatmap_Integration.md` - 集成说明
- `docs/FIFO_Heatmap_Summary.md` - 本文档

## 关键特性

### 1. 可视化方式
- ✅ Plotly网格热力图（非散点图）
- ✅ 每个节点一个颜色块
- ✅ 颜色映射：蓝色(低) → 绿色(中) → 黄色(高) → 红色(满)

### 2. 交互功能
- ✅ 下拉菜单选择FIFO类型
- ✅ 按钮切换平均/峰值使用率
- ✅ 鼠标悬停显示详细信息
- ✅ 支持缩放和平移

### 3. 支持的FIFO类型
- **IQ**: 注入队列 (TL/TR/TU/TD/EQ方向)
- **RB**: Ring Buffer
- **EQ**: 下环队列
- **IQ_CH**: IP通道缓冲 (gdma/sdma/cdma/ddr/l2m)
- **EQ_CH**: 下环通道缓冲

### 4. 数据统计
- 平均使用率：整个仿真期间的平均FIFO深度
- 峰值使用率：仿真期间的最大FIFO深度

## 使用方法

### 方法1: D2D模型中启用（推荐）

```python
from src.core.d2d_model import D2DModel

sim_model = D2DModel(config)

# 在setup_result_analysis中启用
sim_model.setup_result_analysis(
    fifo_utilization_heatmap=True,  # 启用FIFO热力图
    save_figures=True,
    save_dir="../Result"
)

sim_model.run()
sim_model.process_results()  # 自动生成热力图
```

### 方法2: 独立调用

```python
from src.core.fifo_heatmap_visualizer import create_fifo_heatmap

fifo_heatmap_path = create_fifo_heatmap(
    dies=dies,
    config=config,
    total_cycles=total_cycles,
    save_path="fifo_heatmap.html"
)
```

## API参数

### setup_result_analysis() 参数

**D2D模型** (`src/core/d2d_model.py`):
```python
def setup_result_analysis(
    self,
    flow_graph: bool = False,
    ip_bandwidth_heatmap: bool = False,
    fifo_utilization_heatmap: bool = False,  # ← 新增
    save_figures: bool = True,
    export_d2d_requests_csv: bool = True,
    export_ip_bandwidth_csv: bool = True,
    save_dir: str = "",
    heatmap_mode: str = "total"
)
```

**基类** (`src/core/base_model.py` 和 `base_model_v2.py`):
```python
def setup_result_analysis(
    self,
    result_save_path: str = "",
    results_fig_save_path: str = "",
    plot_flow_fig: bool = False,
    plot_RN_BW_fig: bool = False,
    plot_fifo_heatmap: bool = False  # ← 新增
)
```

### create_fifo_heatmap() 参数

```python
def create_fifo_heatmap(
    dies: Dict,                      # Die字典 {die_id: die_model}
    config,                          # 配置对象
    total_cycles: int,               # 网络总周期数
    die_layout: Optional[Dict] = None,      # Die布局
    die_rotations: Optional[Dict] = None,   # Die旋转
    save_path: Optional[str] = None         # HTML保存路径
) -> Optional[str]
```

## 数据流程

```
仿真运行
    ↓
Network.update_fifo_stats_after_move() (每周期自动调用)
    ↓
累积到 fifo_depth_sum 和 fifo_max_depth
    ↓
FIFOUtilizationCollector.collect_from_dies()
    ↓
计算平均和峰值使用率
    ↓
FIFOHeatmapVisualizer.create_interactive_heatmap()
    ↓
生成HTML文件 (fifo_utilization_heatmap.html)
```

## 输出文件

### 文件格式
- **格式**: HTML
- **大小**: 约100-500KB
- **依赖**: Plotly CDN (需要网络连接)

### 文件位置
- D2D模型: `{save_dir}/fifo_utilization_heatmap.html`
- 独立调用: 用户指定的`save_path`

### 查看方式
直接在浏览器中打开HTML文件即可查看交互式热力图

## 技术实现

### 数据收集
- 使用Network对象的`fifo_depth_sum`和`fifo_max_depth`字典
- 统计周期数从`simulation_end_cycle / NETWORK_FREQUENCY`获取

### 可视化技术
- **库**: Plotly (go.Heatmap)
- **布局**: 网格热力图，每个节点一个单元格
- **交互**: updatemenus实现下拉菜单和按钮切换

### 关键修复
1. ✅ 从Scatter改为Heatmap (网格状而非散点)
2. ✅ 修复标题与下拉菜单重叠 (调整y位置和margin)
3. ✅ 修复下拉菜单无法切换 (将所有traces添加到图形)

## 性能考虑

- **内存**: 统计数据在整个仿真期间累积
- **生成时间**: 通常<1秒
- **文件大小**: 100-500KB
- **浏览器**: 现代浏览器均支持

## 测试状态

- ✅ 基础功能测试通过 (`test/test_fifo_heatmap.py`)
- ✅ 下拉菜单切换正常
- ✅ 模式按钮切换正常
- ✅ 鼠标悬停信息显示正常
- ✅ 布局无重叠问题

## 已知限制

1. **需要网络连接**: HTML使用CDN加载Plotly
2. **单Die优化**: 当前为多Die设计，单Die显示可能过大
3. **模式切换**: 切换模式时始终显示第一个FIFO类型（简化实现）

## 未来改进

1. 支持离线版本 (内嵌Plotly.js)
2. 优化单Die显示布局
3. 改进模式切换逻辑（记住当前选择的FIFO类型）
4. 添加时间轴滑块（显示不同时刻的使用率快照）
5. 支持导出为PDF或PNG

## 兼容性

- **Python**: 3.8+
- **依赖**: numpy, plotly
- **浏览器**: Chrome, Firefox, Edge, Safari (现代版本)

## 维护建议

1. 保持与IP带宽热力图的API一致性
2. 定期更新文档
3. 添加更多测试用例
4. 收集用户反馈持续优化

## 相关链接

- [使用说明](FIFO_Heatmap_Usage.md)
- [集成说明](FIFO_Heatmap_Integration.md)
- [测试脚本](../test/test_fifo_heatmap.py)
- [示例脚本](../scripts/example_fifo_heatmap.py)
