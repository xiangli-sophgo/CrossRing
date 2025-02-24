import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import os

# 读取CSV文件
file_root = r"../../Result/Params_csv/"
data_file_name = r"RN_Tracker_OSTD_Results_copy.csv"
data = pd.read_csv(file_root + data_file_name)

# 定义不同的拓扑
topologies = ["4x9", "9x4", "5x4", "4x5"]
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

# data = pd.read_csv(r"../Params_csv/RN_R_W_Results.csv")

# # 定义不同的拓扑
# topologies = ["4x9", "9x4", "5x4", "4x5"]
# topologies = ["9x4"]

# topo = topologies[0]

# show_value = "ReadBandWidth"
# show_value = "WriteBandWidth"
show_value = "TotalBandWidth"
# show_value = "FinishCycle"
# show_value = "cir_h_total"
# show_value = "cir_v_total"
# x_name = "ro_tracker_ostd"
# y_name = "share_tracker_ostd"
x_name = "rn_r_tracker_ostd"
y_name = "rn_w_tracker_ostd"

log_data = 0
save_images = 0

if show_value in ["ReadBandWidth", "WriteBandWidth"]:
    vmax = 64
    vmin = 0
else:
    vmax = 128
    vmin = 64
if save_images:
    output_dir = f"../../Result/Plt_results/{x_name}_{y_name}/"
    os.makedirs(output_dir, exist_ok=True)

for topo in topologies:
    # 筛选出当前拓扑的数据
    # topo_data = data[(data["Topo"] == topo) & (data["FinishTime"] != 60000)]
    topo_data = data[data["Topo"] == topo]
    if log_data:
        topo_data[show_value] = np.log(topo_data[show_value] + 0.1)

    # 使用透视表来准备热图数据
    pivot_table = topo_data.pivot_table(index=x_name, columns=y_name, values=show_value, aggfunc="first")  # 直接使用第一个值

    # 绘制热图
    plt.figure(figsize=(10, 8))
    if show_value in ["FinishCycle", "cir_h_total"]:
        sns.heatmap(pivot_table, cmap="YlGnBu_r", annot=True, fmt=".1f", cbar_kws={"label": show_value})
    else:
        sns.heatmap(pivot_table, cmap="YlGnBu", annot=True, fmt=".1f", cbar_kws={"label": show_value})

    plt.title(f"Heatmap of {show_value} for Topo {topo}")
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    if save_images:
        heatmap_filename = output_dir + f"{topo}_{show_value}.png"
        plt.savefig(heatmap_filename)
        plt.close()
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
