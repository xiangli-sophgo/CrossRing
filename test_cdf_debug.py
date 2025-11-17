"""
测试CDF显示问题
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 生成测试数据
values = np.random.normal(100, 20, 500)
sorted_values = np.sort(values)
cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

# 方法1: 使用make_subplots的secondary_y
print("测试方法1: make_subplots with secondary_y=True")
fig1 = make_subplots(
    rows=1, cols=1,
    specs=[[{"secondary_y": True}]]
)

fig1.add_trace(
    go.Histogram(x=values, name="直方图"),
    row=1, col=1,
    secondary_y=False
)

fig1.add_trace(
    go.Scatter(x=sorted_values, y=cdf, name="CDF", line=dict(color="red", width=3)),
    row=1, col=1,
    secondary_y=True
)

fig1.update_yaxes(title_text="频次", secondary_y=False, row=1, col=1)
fig1.update_yaxes(title_text="累积概率", range=[0, 1.05], secondary_y=True, row=1, col=1)

fig1.write_html("/Users/lixiang/Documents/工作/code/CrossRing/test_output/debug_method1.html")
print("方法1已保存: test_output/debug_method1.html")

# 检查trace信息
print(f"方法1 - Trace数量: {len(fig1.data)}")
for i, trace in enumerate(fig1.data):
    print(f"  Trace {i}: type={trace.type}, yaxis={getattr(trace, 'yaxis', 'default')}")

# 检查layout中的yaxis
print("方法1 - Layout中的yaxis:")
for key in dir(fig1.layout):
    if key.startswith('yaxis'):
        yaxis = getattr(fig1.layout, key)
        print(f"  {key}: side={getattr(yaxis, 'side', None)}, overlaying={getattr(yaxis, 'overlaying', None)}")

print("\n" + "="*60 + "\n")

# 方法2: 手动配置双Y轴
print("测试方法2: 手动配置dual Y-axis")
fig2 = go.Figure()

fig2.add_trace(
    go.Histogram(x=values, name="直方图", yaxis="y")
)

fig2.add_trace(
    go.Scatter(
        x=sorted_values,
        y=cdf,
        name="CDF",
        line=dict(color="red", width=3),
        yaxis="y2"
    )
)

fig2.update_layout(
    yaxis=dict(title="频次"),
    yaxis2=dict(
        title="累积概率",
        overlaying="y",
        side="right",
        range=[0, 1.05]
    )
)

fig2.write_html("/Users/lixiang/Documents/工作/code/CrossRing/test_output/debug_method2.html")
print("方法2已保存: test_output/debug_method2.html")

print(f"方法2 - Trace数量: {len(fig2.data)}")
for i, trace in enumerate(fig2.data):
    print(f"  Trace {i}: type={trace.type}, yaxis={getattr(trace, 'yaxis', 'default')}")
