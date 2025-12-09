"""
Tier6+ 报告生成器

整合所有可视化组件生成完整的 HTML 报告
"""

import os
from typing import Dict, List, Optional
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .hierarchy_graph import HierarchyGraphRenderer
from .latency_chart import LatencyBreakdownChart
from .bandwidth_chart import BandwidthBottleneckChart
from .scaling_chart import ScalingAnalysisChart


class Tier6ReportGenerator:
    """Tier6+ 报告生成器"""

    def __init__(self):
        self.hierarchy_renderer = HierarchyGraphRenderer()
        self.latency_chart = LatencyBreakdownChart()
        self.bandwidth_chart = BandwidthBottleneckChart()
        self.scaling_chart = ScalingAnalysisChart()

    def generate_report(
        self,
        analysis_results: Dict,
        hierarchy_data: Dict,
        scaling_data: Optional[Dict] = None,
        output_path: str = "tier6_report.html",
        title: str = "Tier6+ 多层级网络分析报告",
    ):
        """
        生成完整的 HTML 报告

        Args:
            analysis_results: 分析结果 (来自 Tier6Analyzer.analyze())
            hierarchy_data: 层级结构数据
            scaling_data: 规模扩展数据 (可选)
            output_path: 输出文件路径
            title: 报告标题
        """
        # 生成各个图表
        figures = []

        # 1. 层级结构图
        fig_hierarchy = self.hierarchy_renderer.render(hierarchy_data, style="treemap")
        figures.append(("层级结构 (Treemap)", fig_hierarchy))

        fig_sunburst = self.hierarchy_renderer.render(hierarchy_data, style="sunburst")
        figures.append(("层级结构 (Sunburst)", fig_sunburst))

        # 2. 延迟分解图
        latency_breakdown = analysis_results.get("latency_breakdown", {})
        if latency_breakdown:
            fig_latency = self.latency_chart.render(latency_breakdown, style="combined")
            figures.append(("延迟分解", fig_latency))

            fig_waterfall = self.latency_chart.render(latency_breakdown, style="waterfall")
            figures.append(("延迟瀑布图", fig_waterfall))

        # 3. 带宽分析
        bottleneck = analysis_results.get("bottleneck")
        if bottleneck:
            # 简化的带宽数据
            bandwidth_data = {
                bottleneck["location"]: {
                    "utilization": bottleneck["utilization"],
                    "theoretical_bandwidth_gbps": bottleneck["bandwidth_gbps"],
                    "effective_bandwidth_gbps": bottleneck["bandwidth_gbps"] * (1 - bottleneck["utilization"] * 0.1),
                }
            }
            fig_bw = self.bandwidth_chart.render(bandwidth_data, style="bar")
            figures.append(("带宽利用率", fig_bw))

        # 4. 规模扩展分析
        if scaling_data:
            fig_scaling = self.scaling_chart.render(scaling_data, style="combined")
            figures.append(("规模扩展分析", fig_scaling))

        # 生成 HTML
        html_content = self._generate_html(
            title=title,
            figures=figures,
            analysis_results=analysis_results,
        )

        # 保存文件
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"报告已生成: {output_path}")
        return output_path

    def _generate_html(
        self,
        title: str,
        figures: List[tuple],
        analysis_results: Dict,
    ) -> str:
        """生成 HTML 内容"""

        # 生成图表 HTML
        charts_html = ""
        for i, (chart_title, fig) in enumerate(figures):
            chart_id = f"chart_{i}"
            fig_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=chart_id)
            charts_html += f"""
            <div class="chart-container">
                <h3>{chart_title}</h3>
                {fig_html}
            </div>
            """

        # 生成摘要表格
        summary_html = self._generate_summary_table(analysis_results)

        # 完整 HTML
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        header {{
            background: linear-gradient(135deg, #1890ff, #722ed1);
            color: white;
            padding: 30px;
            text-align: center;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        header h1 {{
            font-size: 2em;
            margin-bottom: 10px;
        }}
        header p {{
            opacity: 0.9;
        }}
        .summary-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .summary-card h2 {{
            color: #1890ff;
            margin-bottom: 15px;
            border-bottom: 2px solid #1890ff;
            padding-bottom: 10px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .summary-item {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }}
        .summary-item .label {{
            color: #666;
            font-size: 0.9em;
        }}
        .summary-item .value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #1890ff;
        }}
        .summary-item.warning .value {{
            color: #fa8c16;
        }}
        .summary-item.critical .value {{
            color: #f5222d;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .chart-container h3 {{
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #fafafa;
            font-weight: 600;
        }}
        .level-tag {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            color: white;
        }}
        .level-die {{ background: #f5222d; }}
        .level-chip {{ background: #722ed1; }}
        .level-board {{ background: #fa8c16; }}
        .level-server {{ background: #52c41a; }}
        .level-pod {{ background: #1890ff; }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 0.9em;
        }}
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }}
        .tab {{
            padding: 10px 20px;
            background: #e8e8e8;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 1em;
        }}
        .tab.active {{
            background: #1890ff;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        {summary_html}

        <div class="charts-section">
            {charts_html}
        </div>

        <footer>
            Tier6+ 多层级网络建模框架 | CrossRing Project
        </footer>
    </div>
</body>
</html>
        """

        return html

    def _generate_summary_table(self, analysis_results: Dict) -> str:
        """生成摘要表格 HTML"""
        total_latency = analysis_results.get("total_latency_ns", 0)
        bottleneck = analysis_results.get("bottleneck")
        traffic = analysis_results.get("traffic_summary", {})

        bottleneck_class = ""
        if bottleneck:
            if bottleneck["utilization"] > 0.9:
                bottleneck_class = "critical"
            elif bottleneck["utilization"] > 0.7:
                bottleneck_class = "warning"

        # 延迟分解表格
        latency_rows = ""
        for level, data in analysis_results.get("latency_breakdown", {}).items():
            total = data.get("total_ns", 0)
            pct = total / total_latency * 100 if total_latency > 0 else 0
            latency_rows += f"""
            <tr>
                <td><span class="level-tag level-{level}">{level.upper()}</span></td>
                <td>{data.get('propagation_ns', 0):.2f}</td>
                <td>{data.get('queuing_ns', 0):.2f}</td>
                <td>{data.get('processing_ns', 0):.2f}</td>
                <td>{total:.2f}</td>
                <td>{pct:.1f}%</td>
            </tr>
            """

        html = f"""
        <div class="summary-card">
            <h2>分析摘要</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="label">总延迟</div>
                    <div class="value">{total_latency:.2f} ns</div>
                </div>
                <div class="summary-item">
                    <div class="label">流量数</div>
                    <div class="value">{traffic.get('total_flows', 0)}</div>
                </div>
                <div class="summary-item">
                    <div class="label">总带宽</div>
                    <div class="value">{traffic.get('total_bandwidth_gbps', 0):.1f} GB/s</div>
                </div>
                <div class="summary-item {bottleneck_class}">
                    <div class="label">瓶颈位置</div>
                    <div class="value">{bottleneck['location'] if bottleneck else '无'}</div>
                </div>
                <div class="summary-item {bottleneck_class}">
                    <div class="label">瓶颈利用率</div>
                    <div class="value">{bottleneck['utilization']*100:.1f}%</div>
                </div> if bottleneck else ''
            </div>
        </div>

        <div class="summary-card">
            <h2>延迟分解详情</h2>
            <table>
                <thead>
                    <tr>
                        <th>层级</th>
                        <th>传播延迟 (ns)</th>
                        <th>排队延迟 (ns)</th>
                        <th>处理延迟 (ns)</th>
                        <th>总延迟 (ns)</th>
                        <th>占比</th>
                    </tr>
                </thead>
                <tbody>
                    {latency_rows}
                </tbody>
            </table>
        </div>
        """

        return html

    def generate_quick_report(
        self,
        analyzer,
        traffic_flows,
        output_path: str = "tier6_quick_report.html",
    ):
        """
        快速生成报告

        Args:
            analyzer: Tier6Analyzer 实例
            traffic_flows: 流量流列表
            output_path: 输出路径
        """
        # 执行分析
        analysis_results = analyzer.analyze(traffic_flows)
        hierarchy_data = analyzer.get_hierarchy_structure()
        scaling_data = analyzer.analyze_scaling(traffic_flows)

        # 生成报告
        return self.generate_report(
            analysis_results=analysis_results,
            hierarchy_data=hierarchy_data,
            scaling_data=scaling_data,
            output_path=output_path,
        )
