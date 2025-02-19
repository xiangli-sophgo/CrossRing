import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 定义区间和间隔
intervals = [0, 48, 64, 128]
gaps = [4, 1, 4]  # 对应于0-48, 48-64, 64-128的间隔

# 生成数据点
data_points = []

# 处理每个区间
for i in range(len(intervals) - 1):
    start = intervals[i]
    end = intervals[i + 1]
    gap = gaps[i]

    # 生成数据点
    points = np.arange(start, end, gap)
    data_points.extend(points)

# 将数据点转换为NumPy数组
data_points = np.array(data_points)

# 生成一些与数据点相关的z值（例如，使用某个函数）
z_values = np.sin(data_points / 10)  # 示例函数

# 创建网格
grid_x, grid_y = np.mgrid[0:128:100j, -1:1:100j]  # y轴可以是任意值，这里设置为-1到1
grid_z = griddata(data_points, z_values, (grid_x[:, 0], grid_y[:, 0]), method="cubic")

# 绘制热力图
plt.imshow(grid_z.T, extent=(0, 128, -1, 1), origin="lower", cmap="viridis")
plt.colorbar(label="Z values")
plt.title("Heatmap with Non-uniform Intervals")
plt.xlabel("X")
plt.ylabel("Y (arbitrary)")
plt.show()
