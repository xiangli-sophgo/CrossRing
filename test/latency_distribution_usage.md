# 延迟分布图功能使用说明

## 功能概述

延迟分布图模块为 CrossRing 项目提供了全面的延迟分析可视化能力,支持 NoC 和 D2D 两种场景。

### 支持的图表类型

1. **直方图 (Histogram)** - 显示延迟值的频率分布
2. **累积分布函数 (CDF)** - 显示小于等于某个延迟值的请求百分比
3. **箱线图 (Box Plot)** - 显示延迟的四分位数、中位数和异常值
4. **概率密度函数 (PDF)** - 平滑的概率密度曲线

### 延迟维度

每种图表类型都支持三个延迟维度:

- **命令延迟 (cmd_latency)** - 从发送请求到收到响应的延迟
- **数据延迟 (data_latency)** - 数据传输的延迟
- **事务总延迟 (transaction_latency)** - 整个事务从开始到结束的总延迟

### 请求类型分类

所有图表都区分不同的请求类型:

- **读请求** (蓝色)
- **写请求** (红色)
- **混合** (绿色)

## 自动集成

延迟分布图已自动集成到现有的分析流程中:

### NoC 分析

在 `SingleDieAnalyzer.analyze_all_bandwidth()` 中,延迟分布图会自动生成并添加到集成 HTML 报告中。

### D2D 分析

在 `D2DAnalyzer.analyze_d2d_results()` 中,D2D 延迟分布图会自动生成。

## 手动使用示例

```python
from src.analysis.latency_distribution_plotter import LatencyDistributionPlotter

# 假设已经有延迟统计数据
# latency_stats = {...}  # 包含 cmd/data/trans 的延迟值

# 创建绘图器
plotter = LatencyDistributionPlotter(latency_stats, title_prefix="NoC")

# 生成单个图表
hist_fig = plotter.plot_histogram(return_fig=True)
cdf_fig = plotter.plot_cdf(return_fig=True)
box_fig = plotter.plot_box(return_fig=True)
pdf_fig = plotter.plot_pdf(return_fig=True)

# 保存为 HTML 文件
hist_fig.write_html("latency_histogram.html")
cdf_fig.write_html("latency_cdf.html")
box_fig.write_html("latency_box.html")
pdf_fig.write_html("latency_pdf.html")

# 或者直接显示图表
plotter.plot_histogram(return_fig=False)  # 在浏览器中打开
```

## 数据格式要求

延迟统计数据必须包含 `values` 字段:

```python
latency_stats = {
    "cmd": {
        "read": {
            "sum": float,
            "max": float,
            "count": int,
            "values": [float, ...],  # 原始延迟值列表
            "p95": float,
            "p99": float,
        },
        "write": {...},
        "mixed": {...},
    },
    "data": {...},
    "trans": {...},
}
```

## 重要修改

### LatencyStatsCollector 修改

`LatencyStatsCollector._finalize_latency_stats()` 方法已修改为默认保留原始延迟值:

```python
def _finalize_latency_stats(stats: Dict, keep_raw_values: bool = True) -> Dict:
    # ...
    if not keep_raw_values:
        del stats[category][req_type]["values"]
    # ...
```

如果不需要生成延迟分布图,可以设置 `keep_raw_values=False` 以节省内存。

## 测试

运行测试脚本验证功能:

```bash
python test/test_latency_distribution.py
```

测试会生成 8 个 HTML 文件到 `test_output/` 目录:

- `noc_latency_histogram.html`
- `noc_latency_cdf.html`
- `noc_latency_box.html`
- `noc_latency_pdf.html`
- `d2d_latency_histogram.html`
- `d2d_latency_cdf.html`
- `d2d_latency_box.html`
- `d2d_latency_pdf.html`

## 图表特性

### 直方图
- 50 个分箱
- 半透明叠加显示读/写/混合请求
- 悬停显示延迟值和计数

### CDF 图
- 显示累积概率曲线
- 添加 P95 和 P99 参考线
- 悬停显示延迟值、百分位和统计信息

### 箱线图
- 显示四分位数和异常值
- 包含均值和标准差
- 便于对比不同请求类型

### PDF 图
- 使用核密度估计平滑曲线
- 区域填充显示概率密度
- 展示延迟分布的整体形态

## 性能考虑

- 延迟分布图依赖原始延迟值列表,会增加内存使用
- 对于大规模仿真(>10万请求),建议在分析完成后及时清理 `values` 字段
- 图表生成时间与数据点数量成正比

## 未来改进建议

1. 添加延迟分布的统计测试(如 K-S 检验)
2. 支持延迟分布的参数拟合(如正态分布、对数正态分布)
3. 添加延迟异常值检测和标注
4. 支持多个仿真结果的延迟分布对比
5. 添加延迟分布的文本摘要生成
