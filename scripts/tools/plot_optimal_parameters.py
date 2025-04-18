import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 读取CSV文件
# file_root = r"../../Result/Params_csv/"
# data_file_name = r"RN_Tracker_OSTD_Results_0225.csv"
# data = pd.read_csv(file_root + data_file_name)

# # 定义不同的拓扑
# topologies = ["4x9", "9x4", "5x4", "4x5"]
# topologies = ["4x9"]

# topo = topologies[0]

# # 创建一个图形
# plt.figure(figsize=(15, 10))

# # 遍历每种拓扑
# for topo in topologies:
#     # for unified_rn_w in range(24, 72, 8):
#     # 筛选出当前拓扑的数据
#     topo_data = data[data["Topo"] == topo]

#     # 统一rn_w_tracker_outstanding
#     # unified_rn_w = topo_data["rn_w_tracker_outstanding"].iloc[0]
#     unified_rn_w = 48
#     # x_name = "ro_tracker_ostd"
#     # y_name = "share_tracker_ostd"
#     x_name = "rn_r_tracker_outstanding"
#     y_name = "rn_w_tracker_outstanding"

#     filtered_data = topo_data[topo_data[x_name] == unified_rn_w]
#     filtered_data = filtered_data.sort_values(by=y_name)
#     plt.plot(
#         filtered_data[y_name],
#         filtered_data["ReadBandWidth"] * 1 + 1 * filtered_data["WriteBandWidth"],
#         label=f"Topo {topo}",
#     )

#     # filtered_data = topo_data[topo_data[y_name] == unified_rn_w]
#     # filtered_data = filtered_data.sort_values(by=x_name)
#     # plt.plot(
#     #     filtered_data[x_name],
#     #     filtered_data["ReadBandWidth"] * 1 + 1 * filtered_data["WriteBandWidth"],
#     #     label=f"Topo {topo}",
#     # )

# # 添加图例和标签
# plt.title(f"RN_W_Tracker_Outstanding={unified_rn_w}", fontsize=16)
# plt.xlabel("RN_R_Tracker_Outstanding", fontsize=16)
# plt.ylabel("Read Bandwidth", fontsize=16)
# # plt.title(f"RN_R_Tracker_Outstanding={unified_rn_r}", fontsize=16)
# # plt.xlabel("RN_W_Tracker_Outstanding", fontsize=16)
# # plt.ylabel("Write Bandwidth", fontsize=16)
# plt.legend()
# plt.grid()
# plt.show()

# 热力图
# 创建一个图形
# import os


# 自定义格式化函数
def format_func(x):
    if x >= 1e6:
        return f"{x/1e6:.3f}M"  # 显示为百万
    elif x >= 1e3:
        return f"{x/1e3:.2f}K"  # 显示为千
    else:
        return f"{x:.1f}"  # 保留一位小数


file_root = r"../../Result/Params_csv/"

# data_file_name = r"SN_Tracker_OSTD_Results_459_fixed_time_interval.csv"
# data_file_name = r"RB_IN_OUT_FIFO_459_0303_2_fixed_time_interval.csv"
# data_file_name = r"ITag_0411.csv"
data_file_name = r"Spare_core_32_shared_0418.csv"
# data_file_name = r"ITag_0416.csv"
# data_file_name = r"Spare_core_0410_16_core_128GB_32_shared2.csv"
# data_file_name = r"Spare_core_0410_Traffic_R_W_64GB_32_shared.csv"
topologies = [
    # "4x9",
    # "9x4",
    "5x4",
    # "4x5",
]

# data_file_name = r"RB_IN_OUT_FIFO_3x3_0303_2.csv"
# data_file_name = r"RB_IN_OUT_FIFO_3x3_0303_2_fixed_time_interval.csv"
# data_file_name = r"LBN_459_0303_fixed_time_interval.csv"
# data_file_name = r"LBN_3x3_0303_fixed_time_interval.csv"
# data_file_name = r"inject_eject_queue_length_3x3_0228_fixed_time_interval.csv"
# topologies = ["3x3"]

data = pd.read_csv(file_root + data_file_name)

# 定义不同的拓扑
# topo = topologies[0]

# show_value = "read_BW"
# show_value = "write_BW"
# show_value = "Total_BW"
# show_value = "ITag_h_num"
# show_value = "ITag_v_num"
# show_value = "R_finish_time"
show_value = "W_finish_time"
# show_value = "R_tail_latency"
# show_value = "W_tail_latency"
# show_value = "EQ_ETag_T0_num"
# show_value = "gdma-R-L2M_thoughput"
# show_value = "sdma-W-L2M_thoughput"
# show_value = "sdma-R-DDR_thoughput"
# show_value = "data_cir_h_num"
# show_value = "data_cir_v_num"
# show_value = "read_retry_num"
# show_value = "write_retry_num"
# show_value = "write_latency_max"
# show_value = "data_wait_cycle_h_num"
# x_name = "ro_tracker_ostd"
# y_name = "share_tracker_ostd"
# x_name = "rn_r_tracker_ostd"
# y_name = "rn_w_tracker_ostd"
# x_name = "inject_queue_length"
# y_name = "eject_queue_length"
# x_name = "RB_IN_FIFO"
# y_name = "RB_OUT_FIFO"
# x_name = "SliceNum"
# y_name = "model_type"
# y_name = "Topo"
# x_name = "TL_Etag_T2_UE_MAX"
# y_name = "TL_Etag_T1_UE_MAX"
# z_name = "TR_Etag_T2_UE_MAX"
# x_name = "TU_Etag_T2_UE_MAX"
# y_name = "TU_Etag_T1_UE_MAX"
# z_name = "TD_Etag_T2_UE_MAX"
# x_name = "ITag_Trigger_Th_H"
# y_name = "ITag_Trigger_Th_V"
# z_name = "ITag_Max_Num_H"
x_name = "fail_core_num"
y_name = "spare_core_row"
# model_type = "REQ_RSP"
# model_type = "Packet_Base"
# model_type = "Feature"

Both_side_ETag_upgrade = 1

rate_plot = 0
log_data = 0
save_images = 0
reverse_cmap = 0
plot_type = 0


# 设置 vmax 和 vmin
# if show_value in ["ReadBandWidth", "WriteBandWidth"]:
#     vmax = 64
#     vmin = 0
# else:
#     vmax = 128
#     vmin = 64

if save_images:
    output_dir = f"../../Result/Plt_results/{x_name}_{y_name}_{model_type}_{'both' if Both_side_ETag_upgrade else 'singe'}/"
    os.makedirs(output_dir, exist_ok=True)

for topo in topologies:
    # 筛选出当前拓扑的数据
    if "model_type" not in data.columns:
        # 如果没有该列,可以选择跳过过滤或采取其他操作
        # topo_data = data[(data["Topo"] == topo) & (data["rn_r_tracker_ostd"] > 16) & (data["rn_w_tracker_ostd"] > 16)]
        topo_data = data[(data["topo_type"] == topo) & (data["Both_side_ETag_upgrade"] == Both_side_ETag_upgrade)]
    else:
        # topo_data = data[(data["Topo"] == topo) & (data["rn_r_tracker_ostd"] > 16) & (data["rn_w_tracker_ostd"] > 16)]
        topo_data = data.loc[(data["topo_type"] == topo) & (data["Both_side_ETag_upgrade"] == Both_side_ETag_upgrade)]
        # topo_data = data.loc[(data["Topo"] == topo)]

    if log_data:
        topo_data[show_value] = np.log(topo_data[show_value] + 0.1)

    # 创建数据透视表
    # pivot_table = topo_data.pivot_table(index=y_name, columns=x_name, values=show_value, aggfunc="first")
    pivot_table = topo_data.pivot_table(index=y_name, columns=x_name, values=show_value, aggfunc="mean")
    # pivot_table = pivot_table.iloc[::2, ::2]

    # # 计算行均值和列均值
    # row_means = pivot_table.mean(axis=1)  # 每行的均值
    # col_means = pivot_table.mean(axis=0)  # 每列的均值

    # # 在数据透视表中插入空行和空列
    # pivot_table_with_means = pivot_table.copy()
    # pivot_table_with_means[" "] = np.nan  # 插入空列
    # pivot_table_with_means.loc[" "] = np.nan  # 插入空行

    # # 添加行均值和列均值
    # pivot_table_with_means["Mean"] = row_means  # 添加列：行均值
    # col_means["Mean"] = np.nan  # 在列均值的最后一项补充 NaN
    # pivot_table_with_means.loc["Mean"] = col_means  # 添加行：列均值

    cmap = "YlGnBu"
    if show_value in ["FinishCycle", "data_cir_h_num", "data_cir_v_num"] or reverse_cmap:
        cmap += "_r"
    #  计算每个点的变化率
    if rate_plot:
        dx, dy = np.gradient(pivot_table.fillna(0).values)  # 计算梯度
        change_rate = np.sqrt(dx**2 + dy**2)  # 计算综合变化率

        # 绘制变化率的热力图
        # plt.figure(figsize=(12, 10))  # 调整图像大小
        ax = sns.heatmap(
            change_rate,
            cmap="YlGnBu_r",  # 配色方案
            annot=np.vectorize(format_func)(change_rate),
            fmt="",  # 格式化
            center=change_rate.mean(),  # 将颜色映射中心值设为数据的平均值
            annot_kws={"size": 12},
            vmax=change_rate.max() * 1.05,
            vmin=change_rate.min() * 0.95,
            linewidths=0.5,  # 网格线宽度
            linecolor="white",  # 网格线颜色
        )
        ax.invert_yaxis()

        # 添加标题和轴标签
        plt.title(f"Change Rate Heatmap of {show_value} for Topo {topo}", fontsize=16, pad=20)
        plt.xlabel(x_name, fontsize=14)
        plt.ylabel(y_name, fontsize=14)

        # 调整刻度字体大小
        plt.xticks(fontsize=12, rotation=0)
        plt.yticks(fontsize=12)

        # 保存或显示图像
        if save_images:
            heatmap_filename = os.path.join(output_dir, f"{topo}_{show_value}_change_rate.png")
            plt.tight_layout()  # 调整布局
            plt.savefig(heatmap_filename, bbox_inches="tight")  # 保存图片
            plt.close()
        else:
            plt.tight_layout()  # 调整布局
            plt.show()

    # 绘制热力图
    # plt.figure(figsize=(12, 10))  # 调整图像大小
    if plot_type == 0:
        ax = sns.heatmap(
            pivot_table,
            cmap=cmap,  # 配色方案
            # annot=True,  # 显示数值
            annot=np.vectorize(format_func)(pivot_table),  # 使用自定义格式化
            fmt="",  # 使用科学计数法,保留一位小数
            # cbar_kws={"label": show_value},  # 颜色条标签
            annot_kws={"size": 12},  # 数值字体大小
            center=pivot_table.mean().mean(),  # 将颜色映射中心值设为数据的平均值
            vmax=pivot_table.max().max() * 1.005,
            vmin=pivot_table.min().min() * 0.995,
            # vmax=pivot_table.max().max() * 1,
            # vmin=pivot_table.min().min() * 1,
            linewidths=0.5,  # 网格线宽度
            linecolor="white",  # 网格线颜色
        )
        ax.invert_yaxis()

        # 添加标题和轴标签
        plt.title(f"Heatmap of {show_value} for Topo {topo}", fontsize=16, pad=20)
        plt.xlabel(x_name, fontsize=14)
        plt.ylabel(y_name, fontsize=14)

        # 调整刻度字体大小
        plt.xticks(fontsize=12, rotation=0)
        plt.yticks(fontsize=12)

        # 保存或显示图像
        if save_images:
            heatmap_filename = os.path.join(output_dir, f"{topo}_{show_value}.png")
            plt.tight_layout()  # 调整布局
            plt.savefig(heatmap_filename, bbox_inches="tight")  # 保存图片
            plt.close()
        else:
            plt.tight_layout()  # 调整布局
            plt.show()
    elif plot_type == 1:
        data = data.loc[(data["Both_side_ETag_upgrade"] == Both_side_ETag_upgrade)]
        unique_z_names = data[z_name].unique()

        # 创建子图
        fig, axes = plt.subplots(nrows=1, ncols=len(unique_z_names), figsize=(18, 3))
        vmin = data[show_value].min()
        vmax = data[show_value].max()
        # vmin = 95
        # vmax = 103

        # 遍历每个 z_name 值
        for i, z_value in enumerate(unique_z_names):
            # 筛选出当前 z_name 的数据
            z_data = data[data[z_name] == z_value]

            # 创建数据透视表
            pivot_table = z_data.pivot_table(index=y_name, columns=x_name, values=show_value, aggfunc="first")

            # 绘制热力图
            sns.heatmap(
                pivot_table,
                cmap=cmap,
                annot=np.vectorize(format_func)(pivot_table),
                fmt="",
                ax=axes[i],
                linewidths=0.5,
                # linecolor="white",
                vmin=vmin,  # 设置数据范围
                vmax=vmax,
            )
            axes[i].invert_yaxis()
            axes[i].set_title(f"{z_name[:-1]}={z_value}")
            axes[i].set_xlabel(x_name)
            axes[i].set_ylabel(y_name)
        fig.suptitle(f"Heatmap of {show_value} for Topo {topo}", fontsize=16)
        if save_images:
            heatmap_filename = os.path.join(output_dir, f"{topo}_{show_value}.png")
            plt.tight_layout()  # 调整布局
            plt.savefig(heatmap_filename, bbox_inches="tight")  # 保存图片
            plt.close()
        else:
            plt.tight_layout()  # 调整布局
            plt.show()
    elif plot_type == 2:
        data = data.loc[(data["Both_side_ETag_upgrade"] == Both_side_ETag_upgrade)]
        unique_z_names = data[z_name].unique()

        # 创建子图
        vmin = data[show_value].min() - 0.5
        vmax = data[show_value].max() + 0.5
        for i, z_value in enumerate(unique_z_names):
            # 筛选出当前 z_name 的数据
            z_data = data[data[z_name] == z_value]

            # 创建数据透视表
            pivot_table = z_data.pivot_table(index=y_name, columns=x_name, values=show_value, aggfunc="first")

            # 独立创建一个图和子图
            fig_single, ax_single = plt.subplots(figsize=(12, 8))

            # 绘制热力图
            sns.heatmap(
                pivot_table,
                cmap=cmap,
                annot=np.vectorize(format_func)(pivot_table),
                fmt="",
                ax=ax_single,
                linewidths=0.5,
                # linecolor="white",
                vmin=vmin,
                vmax=vmax,
            )

            ax_single.invert_yaxis()
            ax_single.set_title(f"{z_name[:-1]}={z_value}")
            ax_single.set_xlabel(x_name)
            ax_single.set_ylabel(y_name)

            fig_single.suptitle(f"Heatmap of {show_value} for Topo {topo}", fontsize=16)
            plt.tight_layout()

            if save_images:
                heatmap_filename = os.path.join(output_dir, f"{topo}_{show_value}_{z_value}.png")
                plt.savefig(heatmap_filename, bbox_inches="tight")
                plt.close(fig_single)
            else:
                plt.show()

# # 等高线图
# # 创建一个图形
# for topo in topologies:
#     # 筛选出当前拓扑的数据
#     topo_data = data[data["Topo"] == topo]

#     # 使用透视表来准备等高线图数据
#     pivot_table = topo_data.pivot_table(
#         index="rn_r_tracker_outstanding", columns="rn_w_tracker_outstanding", values="TotalBandWidth", aggfunc="first"
#     )  # 直接使用第一个值

#     # 绘制等高线图
#     plt.figure(figsize=(10, 8))
#     plt.contourf(pivot_table.columns, pivot_table.index, pivot_table.values, cmap="YlGnBu")
#     plt.colorbar(label="Read Bandwidth")
#     plt.title(f"Contour Plot of Read Bandwidth for Topo {topo}")
#     plt.xlabel("RN_W_Tracker_Outstanding")
#     plt.ylabel("RN_R_Tracker_Outstanding")
#     plt.show()

# from mpl_toolkits.mplot3d import Axes3D

# # 创建一个三维图
# fig = plt.figure(figsize=(15, 10))
# ax = fig.add_subplot(111, projection="3d")

# for topo in topologies:
#     # 筛选出当前拓扑的数据
#     topo_data = data[data["Topo"] == topo]

#     # 直接读取对应的值
#     ax.scatter(
#         topo_data["rn_r_tracker_outstanding"],
#         topo_data["rn_w_tracker_outstanding"],
#         topo_data["ReadBandWidth"],
#         label=f"{topo} Read Bandwidth",
#         alpha=0.6,
#     )

#     ax.scatter(
#         topo_data["rn_r_tracker_outstanding"],
#         topo_data["rn_w_tracker_outstanding"],
#         topo_data["WriteBandWidth"],
#         label=f"{topo} Write Bandwidth",
#         alpha=0.6,
#     )

# ax.set_xlabel("RN_R_Tracker_Outstanding")
# ax.set_ylabel("RN_W_Tracker_Outstanding")
# ax.set_zlabel("Bandwidth")
# ax.set_title("3D Plot of Read and Write Bandwidths")
# ax.legend()
# plt.show()
