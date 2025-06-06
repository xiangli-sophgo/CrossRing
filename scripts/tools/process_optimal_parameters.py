import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from datetime import datetime
import itertools
import warnings
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

warnings.filterwarnings("ignore")

# 设置图表样式 - 必须在字体设置前
sns.set_style("whitegrid")

# 设置字体
plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]  # Times New Roman用于英文，SimHei用于中文
plt.rcParams["axes.unicode_minus"] = False


class SimpleBandwidthAnalyzer:
    def __init__(self, csv_file_path, analysis_targets=["case1", "case2", "weighted"]):
        """
        初始化分析器

        Args:
            csv_file_path (str): CSV文件路径
            analysis_targets (list): 要分析的目标，可选: 'case1', 'case2', 'weighted'
                                   例如: ['case1'] 只分析Case1
                                        ['weighted'] 只分析加权带宽
                                        ['case1', 'case2'] 分析两个Case但不分析加权
        """
        self.csv_file_path = csv_file_path
        self.analysis_targets = analysis_targets
        self.data = None
        self.target_params = ["TL_Etag_T2_UE_MAX", "TL_Etag_T1_UE_MAX", "TR_Etag_T2_UE_MAX", "RB_IN_FIFO_DEPTH", "TU_Etag_T2_UE_MAX", "TU_Etag_T1_UE_MAX", "TD_Etag_T2_UE_MAX", "EQ_IN_FIFO_DEPTH"]
        self.bandwidth_cols = ["Total_sum_BW_mean_traffic_2260E_case1", "Total_sum_BW_mean_traffic_2260E_case2"]
        self.correlations = None

        print(f"分析目标: {', '.join(analysis_targets)}")

    def load_data(self):
        """加载和预处理数据"""
        try:
            self.data = pd.read_csv(self.csv_file_path)
            print(f"数据加载成功！数据形状: {self.data.shape}")

            # 如果存在加权带宽列，直接使用；否则创建简单平均
            if "Total_sum_BW_weighted_mean" in self.data.columns:
                self.weighted_bw_col = "Total_sum_BW_weighted_mean"
                print("发现加权带宽列：Total_sum_BW_weighted_mean")
            else:
                # 如果没有加权带宽，计算简单平均
                self.data["weighted_bw"] = (self.data[self.bandwidth_cols[0]] + self.data[self.bandwidth_cols[1]]) / 2
                self.weighted_bw_col = "weighted_bw"
                print("创建简单平均加权带宽")

            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False

    def calculate_correlations(self):
        """计算参数与带宽的相关性"""
        correlations = {}

        # 与Case1的相关性
        case1_corr = {}
        for param in self.target_params:
            if param in self.data.columns:
                corr = self.data[param].corr(self.data[self.bandwidth_cols[0]])
                case1_corr[param] = corr

        # 与Case2的相关性
        case2_corr = {}
        for param in self.target_params:
            if param in self.data.columns:
                corr = self.data[param].corr(self.data[self.bandwidth_cols[1]])
                case2_corr[param] = corr

        # 与加权带宽的相关性
        weighted_corr = {}
        for param in self.target_params:
            if param in self.data.columns:
                corr = self.data[param].corr(self.data[self.weighted_bw_col])
                weighted_corr[param] = corr

        correlations["case1"] = case1_corr
        correlations["case2"] = case2_corr
        correlations["weighted"] = weighted_corr

        self.correlations = correlations
        return correlations

    def figure_to_html(self, fig):
        """将plotly图表转换为HTML字符串"""
        return pyo.plot(fig, output_type="div", include_plotlyjs=True)

    def create_correlation_chart(self):
        """创建交互式相关性对比图"""
        # 只选择存在的参数
        existing_params = [param for param in self.target_params if param in self.data.columns]

        # 缩短参数名用于显示
        param_labels = [param.replace("_UE_MAX", "").replace("_DEPTH", "") for param in existing_params]

        fig = go.Figure()
        colors = ["rgba(135, 206, 235, 0.8)", "rgba(240, 128, 128, 0.8)", "rgba(144, 238, 144, 0.8)"]

        if "case1" in self.analysis_targets:
            case1_corrs = [self.correlations["case1"][param] for param in existing_params]
            fig.add_trace(
                go.Bar(
                    x=param_labels,
                    y=case1_corrs,
                    name="Case1相关性",
                    marker_color=colors[0],
                    text=[f"{corr:.3f}" for corr in case1_corrs],
                    textposition="outside",
                    hovertemplate="参数: %{x}<br>相关系数: %{y:.3f}<extra></extra>",
                )
            )

        if "case2" in self.analysis_targets:
            case2_corrs = [self.correlations["case2"][param] for param in existing_params]
            fig.add_trace(
                go.Bar(
                    x=param_labels,
                    y=case2_corrs,
                    name="Case2相关性",
                    marker_color=colors[1],
                    text=[f"{corr:.3f}" for corr in case2_corrs],
                    textposition="outside",
                    hovertemplate="参数: %{x}<br>相关系数: %{y:.3f}<extra></extra>",
                )
            )

        if "weighted" in self.analysis_targets:
            weighted_corrs = [self.correlations["weighted"][param] for param in existing_params]
            fig.add_trace(
                go.Bar(
                    x=param_labels,
                    y=weighted_corrs,
                    name="加权带宽相关性",
                    marker_color=colors[2],
                    text=[f"{corr:.3f}" for corr in weighted_corrs],
                    textposition="outside",
                    hovertemplate="参数: %{x}<br>相关系数: %{y:.3f}<extra></extra>",
                )
            )

        fig.update_layout(
            # title="参数与带宽相关性对比",
            xaxis_title="参数",
            yaxis_title="相关系数",
            barmode="group",
            hovermode="x unified",
            font=dict(family="Times New Roman"),
            showlegend=True,
            height=500,
            xaxis=dict(tickangle=45),
            yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="black"),
        )

        return self.figure_to_html(fig)

    def create_bandwidth_comparison(self):
        """创建交互式带宽对比图"""
        # 根据分析目标确定子图数量
        subplot_count = 0
        subplot_titles = []

        if "case1" in self.analysis_targets or "case2" in self.analysis_targets:
            subplot_count += 1
            subplot_titles.append("带宽分布对比")
        if "weighted" in self.analysis_targets:
            subplot_count += 1
            subplot_titles.append("加权带宽分布")

        if subplot_count == 0:
            return None

        fig = make_subplots(rows=1, cols=subplot_count, subplot_titles=subplot_titles, horizontal_spacing=0.1)

        col_idx = 1

        # Case1和Case2分布对比（如果需要）
        if "case1" in self.analysis_targets or "case2" in self.analysis_targets:
            if "case1" in self.analysis_targets:
                fig.add_trace(
                    go.Histogram(
                        x=self.data[self.bandwidth_cols[0]],
                        name="Case1",
                        opacity=0.7,
                        nbinsx=20,
                        marker_color="rgba(135, 206, 235, 0.7)",
                        histnorm="probability density",
                        hovertemplate="带宽: %{x:.1f} Mbps<br>密度: %{y:.4f}<extra></extra>",
                    ),
                    row=1,
                    col=col_idx,
                )

            if "case2" in self.analysis_targets:
                fig.add_trace(
                    go.Histogram(
                        x=self.data[self.bandwidth_cols[1]],
                        name="Case2",
                        opacity=0.7,
                        nbinsx=20,
                        marker_color="rgba(240, 128, 128, 0.7)",
                        histnorm="probability density",
                        hovertemplate="带宽: %{x:.1f} Mbps<br>密度: %{y:.4f}<extra></extra>",
                    ),
                    row=1,
                    col=col_idx,
                )

            fig.update_xaxes(title_text="带宽 (Mbps)", row=1, col=col_idx)
            fig.update_yaxes(title_text="密度", row=1, col=col_idx)
            col_idx += 1

        # 加权带宽分布（如果需要）
        if "weighted" in self.analysis_targets:
            mean_val = self.data[self.weighted_bw_col].mean()

            fig.add_trace(
                go.Histogram(
                    x=self.data[self.weighted_bw_col],
                    name="加权带宽",
                    opacity=0.7,
                    nbinsx=20,
                    marker_color="rgba(147, 112, 219, 0.7)",
                    histnorm="probability density",
                    hovertemplate="带宽: %{x:.1f} Mbps<br>密度: %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=col_idx,
            )

            # 添加平均值线
            fig.add_vline(x=mean_val, line_dash="dash", line_color="red", annotation_text=f"平均值: {mean_val:.1f} Mbps", row=1, col=col_idx)

            fig.update_xaxes(title_text="加权带宽 (Mbps)", row=1, col=col_idx)
            fig.update_yaxes(title_text="密度", row=1, col=col_idx)

        fig.update_layout(
            # title="带宽分布分析",
            barmode="overlay",
            font=dict(family="Times New Roman"),
            height=500,
            showlegend=True,
        )

        return self.figure_to_html(fig)

    def create_heatmap(self, param_x, param_y, bandwidth_metric, title):
        """创建交互式热力图"""
        try:
            if param_x not in self.data.columns or param_y not in self.data.columns:
                return None

            pivot_table = self.data.pivot_table(index=param_y, columns=param_x, values=bandwidth_metric, aggfunc="mean")

            if pivot_table.empty or pivot_table.shape[0] < 2 or pivot_table.shape[1] < 2:
                return None

            # 创建文本注释矩阵
            text_annotations = []
            for i in range(len(pivot_table.index)):
                row_text = []
                for j in range(len(pivot_table.columns)):
                    value = pivot_table.iloc[i, j]
                    if pd.isna(value):
                        row_text.append("")
                    else:
                        row_text.append(f"{value:.1f}")
                text_annotations.append(row_text)

            fig = go.Figure(
                data=go.Heatmap(
                    z=pivot_table.values,
                    x=pivot_table.columns,
                    y=pivot_table.index,
                    colorscale="Viridis_r",
                    text=text_annotations,
                    texttemplate="%{text}",
                    textfont={"size": 9, "color": "white"},
                    hovertemplate=f"{param_x}: %{{x}}<br>{param_y}: %{{y}}<br>带宽: %{{z:.1f}} Mbps<extra></extra>",
                    colorbar=dict(title="带宽 (Mbps)"),
                )
            )

            fig.update_layout(
                # title=f"{title}<br>{param_y} vs {param_x}",
                xaxis_title=param_x,
                yaxis_title=param_y,
                font=dict(family="Times New Roman"),
                height=500,
                width=600,
            )

            # 反转y轴，使得纵坐标从下到上是从小到大
            fig.update_yaxes(autorange="reversed")

            return self.figure_to_html(fig)

        except Exception as e:
            print(f"热力图生成失败 {param_x} vs {param_y}: {e}")
            return None

    def create_all_parameter_heatmaps(self):
        """创建有意义的参数组合热力图"""
        # 只选择存在的参数
        existing_params = [param for param in self.target_params if param in self.data.columns]

        print(f"找到参数: {existing_params}")

        # 根据功能分组设计有意义的参数组合
        meaningful_pairs = []

        # TL/TR/RB_IN 组合 - 这四个参数之间的所有两两组合
        tl_tr_rb_group = ["TL_Etag_T2_UE_MAX", "TL_Etag_T1_UE_MAX", "TR_Etag_T2_UE_MAX", "RB_IN_FIFO_DEPTH"]
        tl_tr_rb_existing = [param for param in tl_tr_rb_group if param in existing_params]

        print(f"\nTL/TR/RB_IN 组参数: {tl_tr_rb_existing}")
        tl_tr_rb_pairs = list(itertools.combinations(tl_tr_rb_existing, 2))
        meaningful_pairs.extend(tl_tr_rb_pairs)
        print(f"TL/TR/RB_IN 组合数: {len(tl_tr_rb_pairs)}")
        for pair in tl_tr_rb_pairs:
            print(f"  - {pair[0]} vs {pair[1]}")

        # TU/TD/EQ_IN 组合 - 这四个参数之间的所有两两组合
        tu_td_eq_group = ["TU_Etag_T2_UE_MAX", "TU_Etag_T1_UE_MAX", "TD_Etag_T2_UE_MAX", "EQ_IN_FIFO_DEPTH"]
        tu_td_eq_existing = [param for param in tu_td_eq_group if param in existing_params]

        print(f"\nTU/TD/EQ_IN 组参数: {tu_td_eq_existing}")
        tu_td_eq_pairs = list(itertools.combinations(tu_td_eq_existing, 2))
        meaningful_pairs.extend(tu_td_eq_pairs)
        print(f"TU/TD/EQ_IN 组合数: {len(tu_td_eq_pairs)}")
        for pair in tu_td_eq_pairs:
            print(f"  - {pair[0]} vs {pair[1]}")

        print(f"\n总共设计了 {len(meaningful_pairs)} 个有意义的参数组合")

        heatmaps = []

        # 根据分析目标生成热力图
        if "case1" in self.analysis_targets:
            print("\n生成Case1热力图...")
            for i, (param_x, param_y) in enumerate(meaningful_pairs):
                heatmap = self.create_heatmap(param_x, param_y, self.bandwidth_cols[0], "Case1 带宽分布")
                if heatmap:
                    heatmaps.append(("Case1", f"{param_y} vs {param_x}", heatmap))
                    print(f"  ✓ Case1 {i+1}/{len(meaningful_pairs)}: {param_x} vs {param_y}")

        if "case2" in self.analysis_targets:
            print("\n生成Case2热力图...")
            for i, (param_x, param_y) in enumerate(meaningful_pairs):
                heatmap = self.create_heatmap(param_x, param_y, self.bandwidth_cols[1], "Case2 带宽分布")
                if heatmap:
                    heatmaps.append(("Case2", f"{param_y} vs {param_x}", heatmap))
                    print(f"  ✓ Case2 {i+1}/{len(meaningful_pairs)}: {param_x} vs {param_y}")

        if "weighted" in self.analysis_targets:
            print("\n生成加权带宽热力图...")
            for i, (param_x, param_y) in enumerate(meaningful_pairs):
                heatmap = self.create_heatmap(param_x, param_y, self.weighted_bw_col, "加权带宽分布")
                if heatmap:
                    heatmaps.append(("加权带宽", f"{param_y} vs {param_x}", heatmap))
                    print(f"  ✓ 加权带宽 {i+1}/{len(meaningful_pairs)}: {param_x} vs {param_y}")

        return heatmaps

    def generate_simple_html(self, output_filename="bandwidth_charts.html"):
        """生成简单的HTML文件，只展示图表"""
        print("生成相关性图表...")
        correlation_chart = self.create_correlation_chart()

        print("生成带宽对比图...")
        bandwidth_chart = self.create_bandwidth_comparison()

        print("生成所有热力图...")
        heatmaps = self.create_all_parameter_heatmaps()

        # 简单的HTML模板
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>带宽分析图表</title>
    <style>
        body {{ 
            font-family: "Times New Roman", Arial, sans-serif; 
            margin: 20px; 
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
        }}
        h1, h2, h3 {{ 
            color: #333; 
            font-family: "Times New Roman", serif;
        }}
        .chart {{ 
            margin: 30px 0; 
            text-align: center; 
        }}
        .heatmap-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        .heatmap-item {{
            text-align: center;
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 5px;
            background: white;
        }}
        .plotly-chart {{
            width: 100%;
            height: auto;
        }}
        .summary {{ 
            background: #e8f4fd; 
            padding: 15px; 
            margin: 20px 0; 
            border-radius: 5px;
            border-left: 4px solid #007acc;
        }}
        .section {{
            margin: 40px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>参数与带宽关系分析</h1>
        <p>文件: {self.csv_file_path}</p>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h3>数据概要</h3>
            <p><strong>总样本数:</strong> {len(self.data)}</p>"""

        # 根据分析目标显示对应的统计信息
        if "case1" in self.analysis_targets:
            html_content += f"""
            <p><strong>Case1 平均带宽:</strong> {self.data[self.bandwidth_cols[0]].mean():.2f} Mbps</p>"""

        if "case2" in self.analysis_targets:
            html_content += f"""
            <p><strong>Case2 平均带宽:</strong> {self.data[self.bandwidth_cols[1]].mean():.2f} Mbps</p>"""

        if "weighted" in self.analysis_targets:
            html_content += f"""
            <p><strong>加权带宽平均值:</strong> {self.data[self.weighted_bw_col].mean():.2f} Mbps</p>"""

        html_content += """
        </div>
        
        <div class="section">
            <h2>1. 参数与带宽相关性分析</h2>
            <div class="chart">
                <div class="plotly-chart">{}</div>
            </div>
            <p><strong>说明:</strong> 正值表示正相关，负值表示负相关。绝对值越大影响越大。图表支持缩放、平移和悬停查看详细数据。</p>
        </div>""".format(
            correlation_chart
        )

        # 只有在有分布图时才显示带宽分布部分
        if bandwidth_chart:
            html_content += f"""
        
        <div class="section">
            <h2>2. 带宽分布分析</h2>
            <div class="chart">
                <div class="plotly-chart">{bandwidth_chart}</div>
            </div>
            <p><strong>说明:</strong> 展示带宽的分布情况。图表支持交互操作，可以缩放查看细节。</p>
        </div>"""
            section_num = 3
        else:
            section_num = 2

        html_content += f"""
        
        <div class="section">
            <h2>{section_num}. 参数两两组合热力图</h2>
            <p><strong>说明:</strong> 每行显示两张图，展示不同参数组合下的实际带宽值。颜色越深表示带宽越高，方块上的数字显示具体带宽值。图表支持缩放和悬停查看详细信息。</p>
        """

        # 按类型分组热力图，只显示选中的分析目标
        target_heatmaps = {}
        if "case1" in self.analysis_targets:
            target_heatmaps["Case1"] = [hm for hm in heatmaps if hm[0] == "Case1"]
        if "case2" in self.analysis_targets:
            target_heatmaps["Case2"] = [hm for hm in heatmaps if hm[0] == "Case2"]
        if "weighted" in self.analysis_targets:
            target_heatmaps["加权带宽"] = [hm for hm in heatmaps if hm[0] == "加权带宽"]

        # 生成热力图HTML
        subsection_count = 1
        for section_name, section_heatmaps in target_heatmaps.items():
            if section_heatmaps:
                html_content += f"""
            <h3>{section_num}.{subsection_count} {section_name} 带宽热力图</h3>
            <div class="heatmap-grid">
            """
                for _, param_desc, heatmap_html in section_heatmaps:
                    html_content += f"""
                <div class="heatmap-item">
                    <h4>{param_desc}</h4>
                    <div class="plotly-chart">{heatmap_html}</div>
                </div>
                    """
                html_content += "</div>"
                subsection_count += 1

        html_content += """
        </div>
    </div>
</body>
</html>
        """

        # 保存HTML文件
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"HTML文件已生成: {output_filename}")
        print(f"总共生成了 {len(heatmaps)} 个热力图")
        return output_filename

    def run_analysis(self, output_filename="bandwidth_charts.html"):
        """运行分析并生成HTML"""
        print("开始分析...")

        if not self.load_data():
            return None

        print("计算相关性...")
        self.calculate_correlations()

        print("生成HTML文件...")
        html_file = self.generate_simple_html(output_filename)

        print("分析完成！")
        return html_file


# 使用示例
if __name__ == "__main__":
    # 分析指定文件
    analyzer = SimpleBandwidthAnalyzer(r"../../Result/FOP/2260E_ETag_multi_traffic_0603_All.csv", analysis_targets=["case2"])

    # 生成HTML报告
    html_report = analyzer.run_analysis("all_parameter_analysis.html")

    if html_report:
        print(f"\n✅ HTML文件已生成: {html_report}")

        # 可选：自动打开浏览器
        import webbrowser

        try:
            webbrowser.open(html_report)
            print("🌐 已在浏览器中打开")
        except:
            print("⚠️ 请手动打开HTML文件")
    else:
        print("❌ 生成失败")


# 快速使用函数
def generate_all_charts(csv_file_path, output_name="all_charts.html"):
    """快速生成所有参数组合的热力图"""
    analyzer = SimpleBandwidthAnalyzer(csv_file_path)
    return analyzer.run_analysis(output_name)
