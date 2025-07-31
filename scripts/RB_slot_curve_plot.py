import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]  # 支持中文显示
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


def load_and_analyze_data(csv_file):
    """
    加载CSV数据并进行分析
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    print("数据基本信息:")
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print("\n前5行数据:")
    print(df.head())

    # 检查关键列的数据范围
    horizontal_values = sorted(df["RB_ONLY_TAG_NUM_HORIZONTAL"].unique())
    vertical_values = sorted(df["RB_ONLY_TAG_NUM_VERTICAL"].unique())

    print(f"\nHORIZONTAL的唯一值: {horizontal_values}")
    print(f"VERTICAL的唯一值: {vertical_values}")
    print(f"HORIZONTAL最大值: {max(horizontal_values)}")
    print(f"VERTICAL最大值: {max(vertical_values)}")

    return df


def analyze_horizontal_impact(df):
    """
    分析HORIZONTAL变量的影响，对每个H值找出所有V值中的最大性能
    """
    print(f"\n分析HORIZONTAL影响时，对每个H值找出所有V值中的最大性能")

    # 获取所有HORIZONTAL的唯一值
    horizontal_values = sorted(df["RB_ONLY_TAG_NUM_HORIZONTAL"].unique())
    print(f"HORIZONTAL值范围: {horizontal_values}")

    results = []

    for h_val in horizontal_values:
        # 筛选当前HORIZONTAL值的所有数据
        h_data = df[df["RB_ONLY_TAG_NUM_HORIZONTAL"] == h_val].copy()

        if len(h_data) == 0:
            continue

        # 对于每个VERTICAL值，计算R和W的均值
        v_stats = h_data.groupby("RB_ONLY_TAG_NUM_VERTICAL").agg({"mixed_avg_weighted_bw_mean_R_5x2": "mean", "mixed_avg_weighted_bw_mean_W_5x2": "mean"}).reset_index()

        # 找出R和W的最大值及对应的V值
        max_r_idx = v_stats["mixed_avg_weighted_bw_mean_R_5x2"].idxmax()
        max_w_idx = v_stats["mixed_avg_weighted_bw_mean_W_5x2"].idxmax()

        max_r_value = v_stats.loc[max_r_idx, "mixed_avg_weighted_bw_mean_R_5x2"]
        max_w_value = v_stats.loc[max_w_idx, "mixed_avg_weighted_bw_mean_W_5x2"]
        best_v_for_r = v_stats.loc[max_r_idx, "RB_ONLY_TAG_NUM_VERTICAL"]
        best_v_for_w = v_stats.loc[max_w_idx, "RB_ONLY_TAG_NUM_VERTICAL"]

        results.append(
            {
                "RB_ONLY_TAG_NUM_HORIZONTAL": h_val,
                "R_max": max_r_value,
                "W_max": max_w_value,
                "best_V_for_R": best_v_for_r,
                "best_V_for_W": best_v_for_w,
                "total_data_points": len(h_data),
                "V_values_tested": len(v_stats),
            }
        )

    horizontal_stats = pd.DataFrame(results)

    print("HORIZONTAL分析结果预览:")
    print(horizontal_stats[["RB_ONLY_TAG_NUM_HORIZONTAL", "R_max", "W_max", "best_V_for_R", "best_V_for_W"]].head())

    return horizontal_stats


def analyze_vertical_impact(df):
    """
    分析VERTICAL变量的影响，对每个V值找出所有H值中的最大性能
    """
    print(f"\n分析VERTICAL影响时，对每个V值找出所有H值中的最大性能")

    # 获取所有VERTICAL的唯一值
    vertical_values = sorted(df["RB_ONLY_TAG_NUM_VERTICAL"].unique())
    print(f"VERTICAL值范围: {vertical_values}")

    results = []

    for v_val in vertical_values:
        # 筛选当前VERTICAL值的所有数据
        v_data = df[df["RB_ONLY_TAG_NUM_VERTICAL"] == v_val].copy()

        if len(v_data) == 0:
            continue

        # 对于每个HORIZONTAL值，计算R和W的均值
        h_stats = v_data.groupby("RB_ONLY_TAG_NUM_HORIZONTAL").agg({"mixed_avg_weighted_bw_mean_R_5x2": "mean", "mixed_avg_weighted_bw_mean_W_5x2": "mean"}).reset_index()

        # 找出R和W的最大值及对应的H值
        max_r_idx = h_stats["mixed_avg_weighted_bw_mean_R_5x2"].idxmax()
        max_w_idx = h_stats["mixed_avg_weighted_bw_mean_W_5x2"].idxmax()

        max_r_value = h_stats.loc[max_r_idx, "mixed_avg_weighted_bw_mean_R_5x2"]
        max_w_value = h_stats.loc[max_w_idx, "mixed_avg_weighted_bw_mean_W_5x2"]
        best_h_for_r = h_stats.loc[max_r_idx, "RB_ONLY_TAG_NUM_HORIZONTAL"]
        best_h_for_w = h_stats.loc[max_w_idx, "RB_ONLY_TAG_NUM_HORIZONTAL"]

        results.append(
            {
                "RB_ONLY_TAG_NUM_VERTICAL": v_val,
                "R_max": max_r_value,
                "W_max": max_w_value,
                "best_H_for_R": best_h_for_r,
                "best_H_for_W": best_h_for_w,
                "total_data_points": len(v_data),
                "H_values_tested": len(h_stats),
            }
        )

    vertical_stats = pd.DataFrame(results)

    print("VERTICAL分析结果预览:")
    print(vertical_stats[["RB_ONLY_TAG_NUM_VERTICAL", "R_max", "W_max", "best_H_for_R", "best_H_for_W"]].head())

    return vertical_stats


def plot_horizontal_analysis(horizontal_stats):
    """
    绘制HORIZONTAL变量影响的图表（每个H值对应所有V值中的最大性能）
    """
    if horizontal_stats.empty:
        print("无法绘制HORIZONTAL分析图表：数据为空")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 绘制R指标最大值
    ax1.plot(horizontal_stats["RB_ONLY_TAG_NUM_HORIZONTAL"], horizontal_stats["R_max"], marker="o", linewidth=2, markersize=6, color="#2E86AB", label="读带宽")
    ax1.set_xlabel("横向环 bubble slot数量")
    ax1.set_ylabel("mixed_avg_weighted_bw_mean_R_5x2 (最大值)")
    ax1.set_title("横向环对R指标的影响 (每个横向环对应所有纵向环中最大值)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 绘制W指标最大值
    ax2.plot(horizontal_stats["RB_ONLY_TAG_NUM_HORIZONTAL"], horizontal_stats["W_max"], marker="s", linewidth=2, markersize=6, color="#F18F01", label="写带宽")
    ax2.set_xlabel("横向环 bubble slot数量")
    ax2.set_ylabel("mixed_avg_weighted_bw_mean_W_5x2 (最大值)")
    ax2.set_title("横向环对W指标的影响 (每个横向环对应所有纵向环中最大值)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 绘制达到R最大值时的最佳V值
    ax3.scatter(horizontal_stats["RB_ONLY_TAG_NUM_HORIZONTAL"], horizontal_stats["best_V_for_R"], s=60, color="#2E86AB", alpha=0.7, label="最佳V值(R)")
    ax3.set_xlabel("横向环 bubble slot数量")
    ax3.set_ylabel("最佳纵向环 bubble slot数量 (对于R指标)")
    ax3.set_title("达到R最大值时的最佳纵向环配置", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 绘制达到W最大值时的最佳V值
    ax4.scatter(horizontal_stats["RB_ONLY_TAG_NUM_HORIZONTAL"], horizontal_stats["best_V_for_W"], s=60, color="#F18F01", alpha=0.7, label="最佳V值(W)")
    ax4.set_xlabel("横向环 bubble slot数量")
    ax4.set_ylabel("最佳纵向环 bubble slot数量 (对于W指标)")
    ax4.set_title("达到W最大值时的最佳纵向环配置", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.show()


def plot_vertical_analysis(vertical_stats):
    """
    绘制VERTICAL变量影响的图表（每个V值对应所有H值中的最大性能）
    """
    if vertical_stats.empty:
        print("无法绘制VERTICAL分析图表：数据为空")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 绘制R指标最大值
    ax1.plot(vertical_stats["RB_ONLY_TAG_NUM_VERTICAL"], vertical_stats["R_max"], marker="o", linewidth=2, markersize=6, color="#A23B72", label="读带宽")
    ax1.set_xlabel("纵向环 bubble slot数量")
    ax1.set_ylabel("mixed_avg_weighted_bw_mean_R_5x2 (最大值)")
    ax1.set_title("纵向环对R指标的影响 (每个纵向环对应所有横向环中最大值)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 绘制W指标最大值
    ax2.plot(vertical_stats["RB_ONLY_TAG_NUM_VERTICAL"], vertical_stats["W_max"], marker="s", linewidth=2, markersize=6, color="#C73E1D", label="写带宽")
    ax2.set_xlabel("纵向环 bubble slot数量")
    ax2.set_ylabel("mixed_avg_weighted_bw_mean_W_5x2 (最大值)")
    ax2.set_title("纵向环对W指标的影响 (每个纵向环对应所有横向环中最大值)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 绘制达到R最大值时的最佳H值
    ax3.scatter(vertical_stats["RB_ONLY_TAG_NUM_VERTICAL"], vertical_stats["best_H_for_R"], s=60, color="#A23B72", alpha=0.7, label="最佳H值(R)")
    ax3.set_xlabel("纵向环 bubble slot数量")
    ax3.set_ylabel("最佳横向环 bubble slot数量 (对于R指标)")
    ax3.set_title("达到R最大值时的最佳横向环配置", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 绘制达到W最大值时的最佳H值
    ax4.scatter(vertical_stats["RB_ONLY_TAG_NUM_VERTICAL"], vertical_stats["best_H_for_W"], s=60, color="#C73E1D", alpha=0.7, label="最佳H值(W)")
    ax4.set_xlabel("纵向环 bubble slot数量")
    ax4.set_ylabel("最佳横向环 bubble slot数量 (对于W指标)")
    ax4.set_title("达到W最大值时的最佳横向环配置", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.show()


def plot_combined_analysis(horizontal_stats, vertical_stats):
    """
    绘制综合对比分析图表
    """
    if horizontal_stats.empty or vertical_stats.empty:
        print("无法绘制综合分析图表：数据不完整")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # HORIZONTAL对R的影响（使用最大值）
    if not horizontal_stats.empty:
        ax1.plot(horizontal_stats["RB_ONLY_TAG_NUM_HORIZONTAL"], horizontal_stats["R_max"], marker="o", linewidth=2, color="#2E86AB")
        ax1.set_title(f"横向环 Bubble Slot 数量与读带宽关系曲线", fontweight="bold")
        ax1.set_xlabel("横向环 Bubble Slot 数量")
        ax1.set_ylabel("读带宽")
        ax1.grid(True, alpha=0.3)

    # HORIZONTAL对W的影响（使用最大值）
    if not horizontal_stats.empty:
        ax2.plot(horizontal_stats["RB_ONLY_TAG_NUM_HORIZONTAL"], horizontal_stats["W_max"], marker="s", linewidth=2, color="#F18F01")
        ax2.set_title(f"横向环 Bubble Slot 数量与写带宽关系曲线", fontweight="bold")
        ax2.set_xlabel("横向环 Bubble Slot 数量")
        ax2.set_ylabel("写带宽")
        ax2.grid(True, alpha=0.3)

    # VERTICAL对R的影响（使用最大值）
    if not vertical_stats.empty:
        ax3.plot(vertical_stats["RB_ONLY_TAG_NUM_VERTICAL"], vertical_stats["R_max"], marker="o", linewidth=2, color="#A23B72")
        ax3.set_title(f"纵向环 Bubble Slot 数量与读带宽关系曲线", fontweight="bold")
        ax3.set_xlabel("纵向环 Bubble Slot 数量")
        ax3.set_ylabel("读带宽")
        ax3.grid(True, alpha=0.3)

    # VERTICAL对W的影响（使用最大值）
    if not vertical_stats.empty:
        ax4.plot(vertical_stats["RB_ONLY_TAG_NUM_VERTICAL"], vertical_stats["W_max"], marker="s", linewidth=2, color="#C73E1D")
        ax4.set_title(f"纵向环 Bubble Slot 数量与写带宽关系曲线", fontweight="bold")
        ax4.set_xlabel("纵向环 Bubble Slot 数量")
        ax4.set_ylabel("写带宽")
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_heatmaps(df):
    """
    绘制二维热力图：HORIZONTAL vs VERTICAL 对应的读带宽和写带宽
    """
    if df.empty:
        print("无法绘制热力图：数据为空")
        return

    # 过滤数据，只显示纵向环≤25的数据
    df_filtered = df[df["RB_ONLY_TAG_NUM_VERTICAL"] <= 25].copy()

    # 创建数据透视表
    read_pivot = df_filtered.pivot_table(values="mixed_avg_weighted_bw_mean_R_5x2", index="RB_ONLY_TAG_NUM_VERTICAL", columns="RB_ONLY_TAG_NUM_HORIZONTAL", aggfunc="mean")

    write_pivot = df_filtered.pivot_table(values="mixed_avg_weighted_bw_mean_W_5x2", index="RB_ONLY_TAG_NUM_VERTICAL", columns="RB_ONLY_TAG_NUM_HORIZONTAL", aggfunc="mean")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 计算读带宽的最大值和范围，适度扩大颜色变化范围
    read_max = read_pivot.max().max()
    read_min = read_pivot.min().min()
    # 设置颜色范围：从最大值的70%到最大值，平衡颜色变化和高性能区域聚焦
    read_vmin = read_max * 0.70
    read_vmax = read_max

    # 读带宽热力图
    im1 = ax1.imshow(read_pivot.values, cmap="YlGnBu", aspect="auto", origin="lower", vmin=read_vmin, vmax=read_vmax)
    ax1.set_title(f"读带宽热力图", fontsize=14, fontweight="bold", pad=20)
    ax1.set_xlabel("横向环 Bubble Slot 数量", fontsize=12)
    ax1.set_ylabel("纵向环 Bubble Slot 数量", fontsize=12)

    # 设置x轴刻度和标签
    x_ticks = range(len(read_pivot.columns))
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(read_pivot.columns.astype(int))

    # 设置y轴刻度和标签
    y_ticks = range(len(read_pivot.index))
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(read_pivot.index.astype(int))

    # 添加颜色条
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label("读带宽", rotation=270, labelpad=15)

    # 只在高性能区域添加数值标注（最大值的90%以上）
    read_threshold = read_max * 0.90
    for i in range(len(read_pivot.index)):
        for j in range(len(read_pivot.columns)):
            if not pd.isna(read_pivot.iloc[i, j]):
                value = read_pivot.iloc[i, j]
                # 只有当数值超过90%最大值时才显示数字
                if value >= read_threshold:
                    # 根据数值在颜色范围中的位置选择文字颜色
                    text_color = "white" if value > read_vmin + (read_vmax - read_vmin) * 0.8 else "white"
                    text = ax1.text(j, i, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

    # 计算写带宽的最大值和范围，适度扩大颜色变化范围
    write_max = write_pivot.max().max()
    write_min = write_pivot.min().min()
    # 设置颜色范围：从最大值的70%到最大值，平衡颜色变化和高性能区域聚焦
    write_vmin = write_max * 0.70
    write_vmax = write_max

    # 写带宽热力图
    im2 = ax2.imshow(write_pivot.values, cmap="YlGnBu", aspect="auto", origin="lower", vmin=write_vmin, vmax=write_vmax)
    ax2.set_title(f"写带宽热力图", fontsize=14, fontweight="bold", pad=20)
    ax2.set_xlabel("横向环 Bubble Slot 数量", fontsize=12)
    ax2.set_ylabel("纵向环 Bubble Slot 数量", fontsize=12)

    # 设置x轴刻度和标签
    x_ticks = range(len(write_pivot.columns))
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(write_pivot.columns.astype(int))

    # 设置y轴刻度和标签
    y_ticks = range(len(write_pivot.index))
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(write_pivot.index.astype(int))

    # 添加颜色条
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label("写带宽", rotation=270, labelpad=15)

    # 只在高性能区域添加数值标注（最大值的90%以上）
    write_threshold = write_max * 0.90
    for i in range(len(write_pivot.index)):
        for j in range(len(write_pivot.columns)):
            if not pd.isna(write_pivot.iloc[i, j]):
                value = write_pivot.iloc[i, j]
                # 只有当数值超过90%最大值时才显示数字
                if value >= write_threshold:
                    # 根据数值在颜色范围中的位置选择文字颜色
                    text_color = "white" if value > write_vmin + (write_vmax - write_vmin) * 0.8 else "white"
                    text = ax2.text(j, i, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

    # 找出最优配置并在图上标记
    read_max_pos = read_pivot.stack().idxmax()
    write_max_pos = write_pivot.stack().idxmax()

    # # 在读带宽热力图上标记最优点
    # read_optimal_v = list(read_pivot.index).index(read_max_pos[0])
    # read_optimal_h = list(read_pivot.columns).index(read_max_pos[1])
    # ax1.scatter(read_optimal_h, read_optimal_v, c="red", s=200, marker="*", edgecolors="white", linewidth=2, label=f"最优点({read_max_pos[1]},{read_max_pos[0]})")
    # ax1.legend(loc="upper right")

    # # 在写带宽热力图上标记最优点
    # write_optimal_v = list(write_pivot.index).index(write_max_pos[0])
    # write_optimal_h = list(write_pivot.columns).index(write_max_pos[1])
    # ax2.scatter(write_optimal_h, write_optimal_v, c="red", s=200, marker="*", edgecolors="white", linewidth=2, label=f"最优点({write_max_pos[1]},{write_max_pos[0]})")
    # ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    # 输出热力图数据摘要
    print("\n热力图数据摘要:")
    print(f"读带宽完整范围: {read_pivot.min().min():.2f} - {read_pivot.max().max():.2f}")
    print(f"读带宽显示范围: {read_vmin:.2f} - {read_vmax:.2f} (最大值的70%-100%)")
    print(f"写带宽完整范围: {write_pivot.min().min():.2f} - {write_pivot.max().max():.2f}")
    print(f"写带宽显示范围: {write_vmin:.2f} - {write_vmax:.2f} (最大值的70%-100%)")
    print(f"\n数值标注策略: 只在性能最高的区域（>90%最大值）显示具体数值")

    print(f"\n最优配置:")
    print(f"读带宽最大值: {read_pivot.loc[read_max_pos]:.2f} (纵向环={read_max_pos[0]}, 横向环={read_max_pos[1]})")
    print(f"写带宽最大值: {write_pivot.loc[write_max_pos]:.2f} (纵向环={write_max_pos[0]}, 横向环={write_max_pos[1]})")


def generate_summary_report(df, horizontal_stats, vertical_stats):
    """
    生成分析摘要报告
    """
    print("=" * 60)
    print("RB TAG数量优化分析报告 (固定另一变量为最大值)")
    print("=" * 60)

    # 基本统计信息
    print(f"\n数据概览:")
    print(f"- 总样本数: {len(df)}")
    print(f"- HORIZONTAL范围: {df['RB_ONLY_TAG_NUM_HORIZONTAL'].min()} - {df['RB_ONLY_TAG_NUM_HORIZONTAL'].max()}")
    print(f"- VERTICAL范围: {df['RB_ONLY_TAG_NUM_VERTICAL'].min()} - {df['RB_ONLY_TAG_NUM_VERTICAL'].max()}")

    # HORIZONTAL分析结果
    if not horizontal_stats.empty:
        print(f"\nHORIZONTAL变量分析 (每个H值对应所有V值中的最大性能):")
        best_h_r = horizontal_stats.loc[horizontal_stats["R_max"].idxmax()]
        best_h_w = horizontal_stats.loc[horizontal_stats["W_max"].idxmax()]
        print(f"- 分析样本数: {horizontal_stats['total_data_points'].sum()}")
        print(f"- R指标最优HORIZONTAL值: {best_h_r['RB_ONLY_TAG_NUM_HORIZONTAL']} (最大值: {best_h_r['R_max']:.4f}, 最佳V: {best_h_r['best_V_for_R']})")
        print(f"- W指标最优HORIZONTAL值: {best_h_w['RB_ONLY_TAG_NUM_HORIZONTAL']} (最大值: {best_h_w['W_max']:.4f}, 最佳V: {best_h_w['best_V_for_W']})")
    else:
        print(f"\nHORIZONTAL变量分析: 无可用数据")

    # VERTICAL分析结果
    if not vertical_stats.empty:
        print(f"\nVERTICAL变量分析 (每个V值对应所有H值中的最大性能):")
        best_v_r = vertical_stats.loc[vertical_stats["R_max"].idxmax()]
        best_v_w = vertical_stats.loc[vertical_stats["W_max"].idxmax()]
        print(f"- 分析样本数: {vertical_stats['total_data_points'].sum()}")
        print(f"- R指标最优VERTICAL值: {best_v_r['RB_ONLY_TAG_NUM_VERTICAL']} (最大值: {best_v_r['R_max']:.4f}, 最佳H: {best_v_r['best_H_for_R']})")
        print(f"- W指标最优VERTICAL值: {best_v_w['RB_ONLY_TAG_NUM_VERTICAL']} (最大值: {best_v_w['W_max']:.4f}, 最佳H: {best_v_w['best_H_for_W']})")
    else:
        print(f"\nVERTICAL变量分析: 无可用数据")

    # 数据分布检查
    print(f"\n数据分布检查:")
    max_h = df["RB_ONLY_TAG_NUM_HORIZONTAL"].max()
    max_v = df["RB_ONLY_TAG_NUM_VERTICAL"].max()
    data_at_max_h = len(df[df["RB_ONLY_TAG_NUM_HORIZONTAL"] == max_h])
    data_at_max_v = len(df[df["RB_ONLY_TAG_NUM_VERTICAL"] == max_v])
    print(f"- HORIZONTAL最大值({max_h})处的数据点数: {data_at_max_h}")
    print(f"- VERTICAL最大值({max_v})处的数据点数: {data_at_max_v}")


def main():
    """
    主函数
    """
    # 文件路径
    csv_file = f"../Result/RB_Tag_Num_Optimization/parameter_range_analysis_0729_2328.csv"

    try:
        # 1. 加载和基本分析
        print("正在加载数据...")
        df = load_and_analyze_data(csv_file)

        # 2. 分析HORIZONTAL变量影响（固定VERTICAL为最大值）
        print("\n正在分析HORIZONTAL变量影响（固定VERTICAL为最大值）...")
        horizontal_stats = analyze_horizontal_impact(df)

        # 3. 分析VERTICAL变量影响（固定HORIZONTAL为最大值）
        print("\n正在分析VERTICAL变量影响（固定HORIZONTAL为最大值）...")
        vertical_stats = analyze_vertical_impact(df)

        # 4. 绘制图表
        print("\n正在生成图表...")
        # plot_horizontal_analysis(horizontal_stats)
        # plot_vertical_analysis(vertical_stats)
        plot_combined_analysis(horizontal_stats, vertical_stats)
        plot_heatmaps(df)

        # 5. 生成报告
        generate_summary_report(df, horizontal_stats, vertical_stats)

        print(f"\n分析完成！")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_file}")
        print("请确保CSV文件在当前目录下")
    except Exception as e:
        print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()
