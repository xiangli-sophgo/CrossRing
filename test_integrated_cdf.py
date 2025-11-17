"""
测试integrated_visualizer方式生成的HTML中CDF是否显示
"""
import sys
sys.path.insert(0, '.')
from src.analysis.latency_distribution_plotter import LatencyDistributionPlotter
from src.analysis.integrated_visualizer import create_integrated_report
import numpy as np

np.random.seed(42)
values = np.random.normal(100, 20, 500).tolist()

latency_stats = {
    'cmd': {'mixed': {'values': values, 'p95': np.percentile(values, 95), 'p99': np.percentile(values, 99)}},
    'data': {'mixed': {'values': values, 'p95': np.percentile(values, 95), 'p99': np.percentile(values, 99)}},
    'trans': {'mixed': {'values': values, 'p95': np.percentile(values, 95), 'p99': np.percentile(values, 99)}}
}

plotter = LatencyDistributionPlotter(latency_stats, 'Test')
fig = plotter.plot_histogram_with_cdf(return_fig=True)

# 使用integrated_visualizer生成
charts_config = [("测试延迟分布", fig, None)]
output_path = create_integrated_report(
    charts_config=charts_config,
    save_path='test_output/integrated_report_test.html',
    show_result_analysis=False
)

print(f"已生成: {output_path}")
print("请打开该文件查看CDF是否显示")
