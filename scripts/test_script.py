def generate_ip_positions(rows, cols, zero_rows=None, zero_cols=None):
    # 创建一个矩阵,初始值为1
    matrix = [[1 for _ in range(cols)] for _ in range(rows)]

    # 将指定的行设置为0
    if zero_rows:
        for row in zero_rows:
            if 0 <= row < rows:
                for col in range(cols):
                    matrix[row][col] = 0

    # 将指定的列设置为0
    if zero_cols:
        for col in zero_cols:
            if 0 <= col < cols:
                for row in range(rows):
                    matrix[row][col] = 0

    # 收集所有元素为1的编号
    indices = []
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 1:
                index = r * cols + c
                indices.append(index)
    # assert len(indices) == self.num_ips, f"Expected {self.num_ips} indices, but got {len(indices)}."
    return indices


from scipy.optimize import linear_sum_assignment
import numpy as np


def distance(p1, p2):
    # return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1])


def assign_nearest_spare(failed_gdma, spare_cores):
    """
    为损坏核心分配备用核心,优先级为：
    1. 同列备用核心优先
    2. 同列中更靠近网络中心的优先
    3. 非同列时选择最靠近中心的备用核心
    """
    num_failed = len(failed_gdma)
    num_spare = len(spare_cores)

    if num_spare < num_failed:
        return []

    def decode(code):
        row = code // 4 // 2
        col = code % 4
        return (col, 4 - row)

    original_spare_cores = spare_cores.copy()
    failed_gdma = [decode(code) for code in failed_gdma]
    spare_cores = [decode(code) for code in spare_cores]

    # 计算每个备用核心的中心性分数（曼哈顿距离到中心点）
    network_center = (1.5, 2)  # 5x4 Mesh的中心坐标近似
    center_scores = {spare: abs(spare[0] - network_center[0]) + abs(spare[1] - network_center[1]) for spare in spare_cores}

    # 构造优先级矩阵
    cost_matrix = np.zeros((num_failed, num_spare))
    for i, gdma in enumerate(failed_gdma):
        for j, spare in enumerate(spare_cores):
            cost_matrix[i][j] = center_scores[spare] + distance(gdma, spare) * 1000

    # 匈牙利算法寻找最小总成本分配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return [original_spare_cores[j] for _, j in sorted(zip(row_ind, col_ind))]


# spare_core_row = 2
# if spare_core_row % 2 == 0:
#     change_ddr_pos = 1
# else:
#     change_ddr_pos = 0
# # if change_ddr_pos:
# normal_core_row = [i for i in range(10) if i % 2 == 0]
# core_poses = generate_ip_positions(10, 4, normal_core_row, [])
# remove_core = [i for i in range(4 * (10 - 1 - spare_core_row), 4 * (10 - spare_core_row))]
# print(core_poses, remove_core)
# normal_core_row.append((9 - (spare_core_row + 1) // 2))
# print(normal_core_row)
# # print(generate_ip_positions(5, 4, normal_core_row, []))
# print(generate_ip_positions(10, 4, normal_core_row, []))

a = assign_nearest_spare([4, 6], [28, 29, 30, 31])
print(a)
