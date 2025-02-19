def generate_matrix_and_get_indices(rows, cols, zero_rows=None, zero_cols=None):
    # 创建一个矩阵，初始值为1
    matrix = [[1 for _ in range(cols)] for _ in range(rows)]

    # 将指定的行设置为0
    if zero_rows is not None:
        for row in zero_rows:
            if 0 <= row < rows:
                for col in range(cols):
                    matrix[row][col] = 0

    # 将指定的列设置为0
    if zero_cols is not None:
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

    return indices


# 示例使用
rows = 128 // 8
cols = 8

# 设置奇数行为0（1, 3, 5, 7）
zero_rows = [i for i in range(rows) if i % 2 == 0]  # 生成奇数行索引
zero_cols = []  # 不设置任何列为0

result = generate_matrix_and_get_indices(rows, cols, zero_rows, zero_cols)
print(result)
