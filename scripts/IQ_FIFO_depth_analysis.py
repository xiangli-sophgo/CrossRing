#!/usr/bin/env python3
"""
IQ FIFO 深度分析脚本
分析 IQ_OUT_FIFO_DEPTH_VERTICAL 和 IQ_OUT_FIFO_DEPTH_EQ 对性能的影响
生成热力图和曲线图
"""

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
    vertical_values = sorted(df["IQ_OUT_FIFO_DEPTH_VERTICAL"].unique())
    eq_values = sorted(df["IQ_OUT_FIFO_DEPTH_EQ"].unique())

    print(f"\nIQ_OUT_FIFO_DEPTH_VERTICAL的唯一值: {vertical_values}")
    print(f"IQ_OUT_FIFO_DEPTH_EQ的唯一值: {eq_values}")
    print(f"VERTICAL最大值: {max(vertical_values)}")
    print(f"EQ最大值: {max(eq_values)}")

    return df


def analyze_vertical_impact(df):
    """
    分析VERTICAL变量的影响，对每个VERTICAL值找出所有EQ值中的最大带宽性能
    """
    print(f"\n分析VERTICAL影响时，对每个VERTICAL值找出所有EQ值中的最大带宽")

    # 获取所有VERTICAL的唯一值
    vertical_values = sorted(df["IQ_OUT_FIFO_DEPTH_VERTICAL"].unique())
    print(f"VERTICAL值范围: {vertical_values}")

    results = []

    for v_val in vertical_values:
        # 筛选当前VERTICAL值的所有数据
        v_data = df[df["IQ_OUT_FIFO_DEPTH_VERTICAL"] == v_val].copy()

        if len(v_data) == 0:
            continue

        # 对于每个EQ值，计算带宽的均值
        eq_stats = v_data.groupby("IQ_OUT_FIFO_DEPTH_EQ").agg({"bw_mean_LLama2_AllReduce": "mean"}).reset_index()

        # 找出带宽的最大值及对应的EQ值
        max_bw_idx = eq_stats["bw_mean_LLama2_AllReduce"].idxmax()
        max_bw_value = eq_stats.loc[max_bw_idx, "bw_mean_LLama2_AllReduce"]
        best_eq_for_bw = eq_stats.loc[max_bw_idx, "IQ_OUT_FIFO_DEPTH_EQ"]

        results.append(
            {
                "IQ_OUT_FIFO_DEPTH_VERTICAL": v_val,
                "bw_max": max_bw_value,
                "best_EQ_for_bw": best_eq_for_bw,
                "total_data_points": len(v_data),
                "EQ_values_tested": len(eq_stats),
            }
        )

    vertical_stats = pd.DataFrame(results)

    print("VERTICAL分析结果预览:")
    print(vertical_stats[["IQ_OUT_FIFO_DEPTH_VERTICAL", "bw_max", "best_EQ_for_bw"]].head())

    return vertical_stats


def analyze_eq_impact(df):
    """
    分析EQ变量的影响，对每个EQ值找出所有VERTICAL值中的最大性能
    """
    print(f"\n分析EQ影响时，对每个EQ值找出所有VERTICAL值中的最大性能")

    # 获取所有EQ的唯一值
    eq_values = sorted(df["IQ_OUT_FIFO_DEPTH_EQ"].unique())
    print(f"EQ值范围: {eq_values}")

    results = []

    for eq_val in eq_values:
        # 筛选当前EQ值的所有数据
        eq_data = df[df["IQ_OUT_FIFO_DEPTH_EQ"] == eq_val].copy()

        if len(eq_data) == 0:
            continue

        # 对于每个VERTICAL值，计算性能指标的均值
        v_stats = (
            eq_data.groupby("IQ_OUT_FIFO_DEPTH_VERTICAL")
            .agg({"total_weighted_bw_mean_LLama2_AllReduce": "mean", "mixed_weighted_bw_mean_LLama2_AllReduce": "mean", "bw_mean_LLama2_AllReduce": "mean"})
            .reset_index()
        )

        # 找出各指标的最大值及对应的VERTICAL值
        max_total_idx = v_stats["total_weighted_bw_mean_LLama2_AllReduce"].idxmax()
        max_mixed_idx = v_stats["mixed_weighted_bw_mean_LLama2_AllReduce"].idxmax()
        max_bw_idx = v_stats["bw_mean_LLama2_AllReduce"].idxmax()

        max_total_value = v_stats.loc[max_total_idx, "total_weighted_bw_mean_LLama2_AllReduce"]
        max_mixed_value = v_stats.loc[max_mixed_idx, "mixed_weighted_bw_mean_LLama2_AllReduce"]
        max_bw_value = v_stats.loc[max_bw_idx, "bw_mean_LLama2_AllReduce"]

        best_v_for_total = v_stats.loc[max_total_idx, "IQ_OUT_FIFO_DEPTH_VERTICAL"]
        best_v_for_mixed = v_stats.loc[max_mixed_idx, "IQ_OUT_FIFO_DEPTH_VERTICAL"]
        best_v_for_bw = v_stats.loc[max_bw_idx, "IQ_OUT_FIFO_DEPTH_VERTICAL"]

        results.append(
            {
                "IQ_OUT_FIFO_DEPTH_EQ": eq_val,
                "total_weighted_bw_max": max_total_value,
                "mixed_weighted_bw_max": max_mixed_value,
                "bw_max": max_bw_value,
                "best_V_for_total": best_v_for_total,
                "best_V_for_mixed": best_v_for_mixed,
                "best_V_for_bw": best_v_for_bw,
                "total_data_points": len(eq_data),
                "V_values_tested": len(v_stats),
            }
        )

    eq_stats = pd.DataFrame(results)

    print("EQ分析结果预览:")
    print(eq_stats[["IQ_OUT_FIFO_DEPTH_EQ", "total_weighted_bw_max", "mixed_weighted_bw_max", "bw_max"]].head())

    return eq_stats


def plot_combined_analysis(vertical_stats, eq_stats):
    """
    绘制综合对比分析图表
    """
    if vertical_stats.empty or eq_stats.empty:
        print("无法绘制综合分析图表：数据不完整")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # VERTICAL对总加权带宽的影响
    if not vertical_stats.empty:
        ax1.plot(vertical_stats["IQ_OUT_FIFO_DEPTH_VERTICAL"], vertical_stats["bw_max"], marker="o", linewidth=2, markersize=8, color="#2E86AB")
        ax1.set_title(f"垂直方向FIFO深度与总加权带宽关系曲线", fontweight="bold", fontsize=14)
        ax1.set_xlabel("IQ_OUT_FIFO_DEPTH_VERTICAL")
        ax1.set_ylabel("总加权带宽 (最大值)")
        ax1.grid(True, alpha=0.3)

    # VERTICAL对混合加权带宽的影响
    if not vertical_stats.empty:
        ax2.plot(vertical_stats["IQ_OUT_FIFO_DEPTH_VERTICAL"], vertical_stats["bw_max"], marker="s", linewidth=2, markersize=8, color="#F18F01")
        ax2.set_title(f"垂直方向FIFO深度与混合加权带宽关系曲线", fontweight="bold", fontsize=14)
        ax2.set_xlabel("IQ_OUT_FIFO_DEPTH_VERTICAL")
        ax2.set_ylabel("混合加权带宽 (最大值)")
        ax2.grid(True, alpha=0.3)

    # EQ对总加权带宽的影响
    if not eq_stats.empty:
        ax3.plot(eq_stats["IQ_OUT_FIFO_DEPTH_EQ"], eq_stats["bw_max"], marker="o", linewidth=2, markersize=8, color="#A23B72")
        ax3.set_title(f"EQ FIFO深度与总加权带宽关系曲线", fontweight="bold", fontsize=14)
        ax3.set_xlabel("IQ_OUT_FIFO_DEPTH_EQ")
        ax3.set_ylabel("总加权带宽 (最大值)")
        ax3.grid(True, alpha=0.3)

    # EQ对混合加权带宽的影响
    if not eq_stats.empty:
        ax4.plot(eq_stats["IQ_OUT_FIFO_DEPTH_EQ"], eq_stats["bw_max"], marker="s", linewidth=2, markersize=8, color="#C73E1D")
        ax4.set_title(f"EQ FIFO深度与混合加权带宽关系曲线", fontweight="bold", fontsize=14)
        ax4.set_xlabel("IQ_OUT_FIFO_DEPTH_EQ")
        ax4.set_ylabel("混合加权带宽 (最大值)")
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_heatmaps(df):
    """
    绘制二维热力图：VERTICAL vs EQ 对应的不同性能指标
    """
    if df.empty:
        print("无法绘制热力图：数据为空")
        return

    # 创建数据透视表
    total_weighted_pivot = df.pivot_table(values="total_weighted_bw_mean_LLama2_AllReduce", index="IQ_OUT_FIFO_DEPTH_VERTICAL", columns="IQ_OUT_FIFO_DEPTH_EQ", aggfunc="mean")

    mixed_weighted_pivot = df.pivot_table(values="mixed_weighted_bw_mean_LLama2_AllReduce", index="IQ_OUT_FIFO_DEPTH_VERTICAL", columns="IQ_OUT_FIFO_DEPTH_EQ", aggfunc="mean")

    bw_pivot = df.pivot_table(values="bw_mean_LLama2_AllReduce", index="IQ_OUT_FIFO_DEPTH_VERTICAL", columns="IQ_OUT_FIFO_DEPTH_EQ", aggfunc="mean")

    # 创建三个子图
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 1, figsize=(20, 6))
    fig, ax3 = plt.subplots(1, 1, figsize=(20, 6))

    # # 1. 总加权带宽热力图
    # total_max = total_weighted_pivot.max().max()
    # total_min = total_weighted_pivot.min().min()
    # total_vmin = total_max * 0.70
    # total_vmax = total_max

    # im1 = ax1.imshow(total_weighted_pivot.values, cmap="YlOrRd", aspect="auto", origin="lower", vmin=total_vmin, vmax=total_vmax)
    # ax1.set_title(f"总加权带宽热力图", fontsize=14, fontweight="bold", pad=20)
    # ax1.set_xlabel("IQ_OUT_FIFO_DEPTH_EQ", fontsize=12)
    # ax1.set_ylabel("IQ_OUT_FIFO_DEPTH_VERTICAL", fontsize=12)

    # # 设置坐标轴
    # x_ticks = range(len(total_weighted_pivot.columns))
    # ax1.set_xticks(x_ticks)
    # ax1.set_xticklabels(total_weighted_pivot.columns.astype(int))
    # y_ticks = range(len(total_weighted_pivot.index))
    # ax1.set_yticks(y_ticks)
    # ax1.set_yticklabels(total_weighted_pivot.index.astype(int))

    # # 添加颜色条
    # cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    # cbar1.set_label("总加权带宽", rotation=270, labelpad=15)

    # # 添加数值标注（高性能区域）
    # total_threshold = total_max * 0.90
    # for i in range(len(total_weighted_pivot.index)):
    #     for j in range(len(total_weighted_pivot.columns)):
    #         if not pd.isna(total_weighted_pivot.iloc[i, j]):
    #             value = total_weighted_pivot.iloc[i, j]
    #             if value >= total_threshold:
    #                 text_color = "white" if value > total_vmin + (total_vmax - total_vmin) * 0.8 else "black"
    #                 ax1.text(j, i, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

    # # 2. 混合加权带宽热力图
    # mixed_max = mixed_weighted_pivot.max().max()
    # mixed_min = mixed_weighted_pivot.min().min()
    # mixed_vmin = mixed_max * 0.70
    # mixed_vmax = mixed_max

    # im2 = ax2.imshow(mixed_weighted_pivot.values, cmap="YlGnBu", aspect="auto", origin="lower", vmin=mixed_vmin, vmax=mixed_vmax)
    # ax2.set_title(f"混合加权带宽热力图", fontsize=14, fontweight="bold", pad=20)
    # ax2.set_xlabel("IQ_OUT_FIFO_DEPTH_EQ", fontsize=12)
    # ax2.set_ylabel("IQ_OUT_FIFO_DEPTH_VERTICAL", fontsize=12)

    # # 设置坐标轴
    # x_ticks = range(len(mixed_weighted_pivot.columns))
    # ax2.set_xticks(x_ticks)
    # ax2.set_xticklabels(mixed_weighted_pivot.columns.astype(int))
    # y_ticks = range(len(mixed_weighted_pivot.index))
    # ax2.set_yticks(y_ticks)
    # ax2.set_yticklabels(mixed_weighted_pivot.index.astype(int))

    # # 添加颜色条
    # cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    # cbar2.set_label("混合加权带宽", rotation=270, labelpad=15)

    # # 添加数值标注（高性能区域）
    # mixed_threshold = mixed_max * 0.90
    # for i in range(len(mixed_weighted_pivot.index)):
    #     for j in range(len(mixed_weighted_pivot.columns)):
    #         if not pd.isna(mixed_weighted_pivot.iloc[i, j]):
    #             value = mixed_weighted_pivot.iloc[i, j]
    #             if value >= mixed_threshold:
    #                 text_color = "white" if value > mixed_vmin + (mixed_vmax - mixed_vmin) * 0.8 else "black"
    #                 ax2.text(j, i, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

    # 3. 基础带宽热力图
    bw_max = bw_pivot.max().max()
    bw_min = bw_pivot.min().min()
    bw_vmin = bw_max * 0.70
    bw_vmax = bw_max

    im3 = ax3.imshow(bw_pivot.values, cmap="YlGnBu", aspect="auto", origin="lower", vmin=bw_vmin, vmax=bw_vmax)
    ax3.set_title(f"基础带宽热力图", fontsize=14, fontweight="bold", pad=20)
    ax3.set_xlabel("IQ_OUT_FIFO_DEPTH_EQ", fontsize=12)
    ax3.set_ylabel("IQ_OUT_FIFO_DEPTH_VERTICAL", fontsize=12)

    # 设置坐标轴
    x_ticks = range(len(bw_pivot.columns))
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(bw_pivot.columns.astype(int))
    y_ticks = range(len(bw_pivot.index))
    ax3.set_yticks(y_ticks)
    ax3.set_yticklabels(bw_pivot.index.astype(int))

    # 添加颜色条
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label("基础带宽", rotation=270, labelpad=15)

    # 添加数值标注（高性能区域）
    bw_threshold = bw_max * 0.90
    for i in range(len(bw_pivot.index)):
        for j in range(len(bw_pivot.columns)):
            if not pd.isna(bw_pivot.iloc[i, j]):
                value = bw_pivot.iloc[i, j]
                if value >= bw_threshold:
                    text_color = "white" if value > bw_vmin + (bw_vmax - bw_vmin) * 0.8 else "black"
                    ax3.text(j, i, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.show()

    # 输出热力图数据摘要
    print("\n热力图数据摘要:")
    print(f"总加权带宽完整范围: {total_weighted_pivot.min().min():.2f} - {total_weighted_pivot.max().max():.2f}")
    print(f"混合加权带宽完整范围: {mixed_weighted_pivot.min().min():.2f} - {mixed_weighted_pivot.max().max():.2f}")
    print(f"基础带宽完整范围: {bw_pivot.min().min():.2f} - {bw_pivot.max().max():.2f}")

    # 找出最优配置
    total_max_pos = total_weighted_pivot.stack().idxmax()
    mixed_max_pos = mixed_weighted_pivot.stack().idxmax()
    bw_max_pos = bw_pivot.stack().idxmax()

    print(f"\n最优配置:")
    print(f"总加权带宽最大值: {total_weighted_pivot.loc[total_max_pos]:.2f} (VERTICAL={total_max_pos[0]}, EQ={total_max_pos[1]})")
    print(f"混合加权带宽最大值: {mixed_weighted_pivot.loc[mixed_max_pos]:.2f} (VERTICAL={mixed_max_pos[0]}, EQ={mixed_max_pos[1]})")
    print(f"基础带宽最大值: {bw_pivot.loc[bw_max_pos]:.2f} (VERTICAL={bw_max_pos[0]}, EQ={bw_max_pos[1]})")


def generate_summary_report(df, vertical_stats, eq_stats):
    """
    生成分析摘要报告
    """
    print("=" * 70)
    print("IQ FIFO深度优化分析报告")
    print("=" * 70)

    # 基本统计信息
    print(f"\n数据概览:")
    print(f"- 总样本数: {len(df)}")
    print(f"- IQ_OUT_FIFO_DEPTH_VERTICAL范围: {df['IQ_OUT_FIFO_DEPTH_VERTICAL'].min()} - {df['IQ_OUT_FIFO_DEPTH_VERTICAL'].max()}")
    print(f"- IQ_OUT_FIFO_DEPTH_EQ范围: {df['IQ_OUT_FIFO_DEPTH_EQ'].min()} - {df['IQ_OUT_FIFO_DEPTH_EQ'].max()}")

    # VERTICAL分析结果
    if not vertical_stats.empty:
        print(f"\nVERTICAL变量分析 (每个VERTICAL值对应所有EQ值中的最大性能):")
        best_v_total = vertical_stats.loc[vertical_stats["total_weighted_bw_max"].idxmax()]
        best_v_mixed = vertical_stats.loc[vertical_stats["mixed_weighted_bw_max"].idxmax()]
        print(f"- 总加权带宽最优VERTICAL值: {best_v_total['IQ_OUT_FIFO_DEPTH_VERTICAL']} (最大值: {best_v_total['total_weighted_bw_max']:.4f}, 最佳EQ: {best_v_total['best_EQ_for_total']})")
        print(f"- 混合加权带宽最优VERTICAL值: {best_v_mixed['IQ_OUT_FIFO_DEPTH_VERTICAL']} (最大值: {best_v_mixed['mixed_weighted_bw_max']:.4f}, 最佳EQ: {best_v_mixed['best_EQ_for_mixed']})")

    # EQ分析结果
    if not eq_stats.empty:
        print(f"\nEQ变量分析 (每个EQ值对应所有VERTICAL值中的最大性能):")
        best_eq_total = eq_stats.loc[eq_stats["total_weighted_bw_max"].idxmax()]
        best_eq_mixed = eq_stats.loc[eq_stats["mixed_weighted_bw_max"].idxmax()]
        print(f"- 总加权带宽最优EQ值: {best_eq_total['IQ_OUT_FIFO_DEPTH_EQ']} (最大值: {best_eq_total['total_weighted_bw_max']:.4f}, 最佳VERTICAL: {best_eq_total['best_V_for_total']})")
        print(f"- 混合加权带宽最优EQ值: {best_eq_mixed['IQ_OUT_FIFO_DEPTH_EQ']} (最大值: {best_eq_mixed['mixed_weighted_bw_max']:.4f}, 最佳VERTICAL: {best_eq_mixed['best_V_for_mixed']})")

    # 整体最优配置
    print(f"\n整体最优配置分析:")
    best_overall = df.loc[df["total_weighted_bw_mean_LLama2_AllReduce"].idxmax()]
    print(f"- 总加权带宽全局最优: VERTICAL={best_overall['IQ_OUT_FIFO_DEPTH_VERTICAL']}, EQ={best_overall['IQ_OUT_FIFO_DEPTH_EQ']}, 值={best_overall['total_weighted_bw_mean_LLama2_AllReduce']:.4f}")

    best_mixed = df.loc[df["mixed_weighted_bw_mean_LLama2_AllReduce"].idxmax()]
    print(f"- 混合加权带宽全局最优: VERTICAL={best_mixed['IQ_OUT_FIFO_DEPTH_VERTICAL']}, EQ={best_mixed['IQ_OUT_FIFO_DEPTH_EQ']}, 值={best_mixed['mixed_weighted_bw_mean_LLama2_AllReduce']:.4f}")


def main():
    """
    主函数
    """
    # 文件路径
    csv_file = "C:/Users/xiang/Documents/code/CrossRing/Result/ParameterTraversal/IQ_OUT_FIFO_DEPTH_VERTICAL_IQ_OUT_FIFO_DEPTH_EQ_0731_1407/final_results.csv"

    try:
        # 1. 加载和基本分析
        print("正在加载数据...")
        df = load_and_analyze_data(csv_file)

        # 2. 分析VERTICAL变量影响
        print("\n正在分析VERTICAL变量影响...")
        vertical_stats = analyze_vertical_impact(df)

        # 3. 分析EQ变量影响
        print("\n正在分析EQ变量影响...")
        eq_stats = analyze_eq_impact(df)

        # 4. 绘制图表
        print("\n正在生成图表...")
        plot_combined_analysis(vertical_stats, eq_stats)
        plot_heatmaps(df)

        # 5. 生成报告
        # generate_summary_report(df, vertical_stats, eq_stats)

        print(f"\n分析完成！")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_file}")
        print("请确保CSV文件路径正确")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback

        print("详细错误信息:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
