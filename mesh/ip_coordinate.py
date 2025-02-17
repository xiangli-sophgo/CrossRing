"""
文件用于生成ip坐标和slice
"""

import json

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

x_length = config["x_length"]
y_length = config["y_length"]
x_range = range(x_length)
y_range = range(y_length)
# 每个节点都挂了ddr
ddr = [(x, y) for x in x_range for y in y_range]
gdma = [(x, y) for x in x_range for y in y_range]
cdma = [(x, y) for x in x_range for y in y_range]

ip = {"ddr": ddr, "cdma": cdma, "gdma": gdma}
# LBN为链路上的slice数
LBN = config["LBN"]
slice_x = [LBN for _ in x_range]
slice_y = [LBN for _ in y_range]

enter_mesh_slice = config["enter_mesh_slice"]
leave_mesh_slice = config["leave_mesh_slice"]
