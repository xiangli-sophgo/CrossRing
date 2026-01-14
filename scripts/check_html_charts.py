#!/usr/bin/env python3
"""检查HTML中的图表数量"""

import os
import glob

# 查找最新的result_analysis.html
result_dirs = glob.glob("../Result*/**/result_analysis.html", recursive=True)
if not result_dirs:
    print("❌ 没有找到result_analysis.html文件")
    print("请先运行仿真生成结果")
    exit(1)

# 按修改时间排序，取最新的
result_dirs.sort(key=os.path.getmtime, reverse=True)
html_file = result_dirs[0]

print(f"检查文件: {html_file}")
print(f"修改时间: {os.path.getmtime(html_file)}")

# 读取HTML内容
with open(html_file, 'r', encoding='utf-8') as f:
    html_content = f.read()

# 检查图表标题
chart_titles = [
    "延迟分布",
    "延迟时序-窗口平均",
    "延迟时序-散点图",
    "延迟时序-热力图",
    "FIFO",
    "流量图",
    "RN带宽"
]

print("\n=== 图表检查 ===")
for title in chart_titles:
    count = html_content.count(title)
    status = "✅" if count > 0 else "❌"
    print(f"{status} {title}: 出现{count}次")

# 检查plotly div数量（每个图表应该有一个div）
plotly_div_count = html_content.count('class="plotly-graph-div"')
print(f"\n总共有 {plotly_div_count} 个Plotly图表div")

# 检查time_value_pairs关键字
if "time_value_pairs" in html_content:
    print("\n⚠️  HTML中包含'time_value_pairs'字符串（调试信息泄露）")
