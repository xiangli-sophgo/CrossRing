import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 读取CSV文件
data = pd.read_csv(r"all_result_overall.csv")

# 定义不同的拓扑
topologies = ["4x9", "9x4", "5x4", '4x5']

# 创建一个图形
plt.figure(figsize=(15, 10))

# 遍历每种拓扑
for topo in topologies:
    # 筛选出当前拓扑的数据
    topo_data = data[data["Topo"] == topo]

    # 统一rn_w_tracker_outstanding
    # unified_rn_w = topo_data["rn_w_tracker_outstanding"].iloc[0]
    unified_rn_w = 64
    # unified_rn_r = 48
    x_name = 'ro_tracker_ostd'
    y_name = 'share_tracker_ostd'

    # 筛选出相同的rn_w_tracker_outstanding
    filtered_data = topo_data[topo_data[y_name] == unified_rn_w]
    # filtered_data = topo_data[topo_data["rn_r_tracker_outstanding"] == unified_rn_r]

    # 绘制rn_r_tracker_outstanding与ReadBandWidth的关系
    plt.bar(
        filtered_data[x_name], filtered_data["ReadBandWidth"] + filtered_data["WriteBandWidth"], label=f"Topo {topo}",
    )

# 添加图例和标签
plt.title(f"RN_W_Tracker_Outstanding={unified_rn_w}", fontsize=16)
plt.xlabel("RN_R_Tracker_Outstanding", fontsize=16)
plt.ylabel("Read Bandwidth", fontsize=16)
# plt.title(f"RN_R_Tracker_Outstanding={unified_rn_r}", fontsize=16)
# plt.xlabel("RN_W_Tracker_Outstanding", fontsize=16)
# plt.ylabel("Write Bandwidth", fontsize=16)
plt.legend()
plt.grid()
plt.show()

# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from dash import Dash, dcc, html, Input, Output

# # 读取CSV文件

# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from dash import Dash, dcc, html, Input, Output

# # 读取CSV文件
# data = pd.read_csv(r"..\Result\all_result_overall.csv")


# # 创建Dash应用
# app = Dash(__name__)

# # 获取唯一的rn_r_tracker_outstanding和rn_w_tracker_outstanding值
# rn_r_values = data["rn_r_tracker_outstanding"].unique()
# rn_w_values = data["rn_w_tracker_outstanding"].unique()

# # 创建应用布局
# app.layout = html.Div(
#     [
#         dcc.Dropdown(
#             id="rn_r_dropdown",
#             options=[{"label": str(value), "value": value} for value in rn_r_values],
#             value=rn_r_values[0],  # 默认选择第一个值
#             clearable=False,
#         ),
#         dcc.Dropdown(
#             id="rn_w_dropdown",
#             options=[{"label": str(value), "value": value} for value in rn_w_values],
#             value=rn_w_values[0],  # 默认选择第一个值
#             clearable=False,
#         ),
#         dcc.Graph(id="3d_surface_graph"),
#     ]
# )


# # 回调函数更新图形
# @app.callback(Output("3d_surface_graph", "figure"), Input("rn_r_dropdown", "value"), Input("rn_w_dropdown", "value"))
# def update_graph(selected_rn_r, selected_rn_w):
#     # 创建网格
#     X, Y = np.meshgrid(rn_r_values, rn_w_values)

#     # 创建Z矩阵
#     Z = np.zeros(X.shape)

#     for i in range(len(rn_r_values)):
#         for j in range(len(rn_w_values)):
#             z_values = data[(data["rn_r_tracker_outstanding"] == rn_r_values[i]) & (data["rn_w_tracker_outstanding"] == rn_w_values[j])][
#                 "ReadBandWidth"
#             ]
#             # if not z_values.empty:
#             # Z[i, j] = z_values.mean()  # 取平均值作为表面高度

#     # 创建曲面图
#     surface = go.Surface(z=Z, x=X, y=Y, opacity=0.2, name="Surface")

#     # 创建对应的曲线
#     line_data_r = data[data["rn_r_tracker_outstanding"] == selected_rn_r]
#     line_data_w = data[data["rn_w_tracker_outstanding"] == selected_rn_w]

#     # 绘制选择的曲线
#     curve_r = go.Scatter3d(
#         x=[selected_rn_r] * len(line_data_w),
#         y=line_data_w["rn_w_tracker_outstanding"],
#         z=line_data_w["ReadBandWidth"],
#         mode="lines+markers",
#         name=f"Curve for RN_R = {selected_rn_r}",
#         line=dict(color="blue", width=4),
#     )

#     curve_w = go.Scatter3d(
#         x=line_data_r["rn_r_tracker_outstanding"],
#         y=[selected_rn_w] * len(line_data_r),
#         z=line_data_r["ReadBandWidth"],
#         mode="lines+markers",
#         name=f"Curve for RN_W = {selected_rn_w}",
#         line=dict(color="red", width=4),
#     )

#     # 创建图形
#     fig = go.Figure(data=[surface, curve_r, curve_w])

#     # 更新图形布局
#     fig.update_layout(
#         title="3D Surface Plot with Selected Curve",
#         scene=dict(xaxis_title="RN_R_Tracker_Outstanding", yaxis_title="RN_W_Tracker_Outstanding", zaxis_title="Read Bandwidth"),
#         autosize=True,
#     )

#     return fig


# # 运行应用
# if __name__ == "__main__":
#     app.run_server(debug=True)
