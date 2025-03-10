import pandas as pd

# 假设数据流文件名为 'data.txt'
# file_path = "../../traffic/output_All_reduce/step5_data_merge/TPS009-Llama2-70B-S4K-O1-W8A8-B128-LMEM2M-AllReduce_Trace.txt"
file_path = "../../traffic/output_All_reduce/step5_data_merge/TPS009-Llama2-70B-S4K-O1-W8A8-B128-LMEM2M-AllReduce_Trace.txt"

# 读取数据流文件
# 这里假设数据是以逗号分隔的
data = pd.read_csv(file_path, header=None)

# 选择前面的数字和 "ddr" 列
# 这里假设前面的数字在第二列（索引为1），"ddr" 在第五列（索引为4）
numbers = data[1]

# 统计每个数字出现的次数
count_series = numbers.value_counts()

# 将统计结果转换为 DataFrame
count_df = count_series.reset_index()
count_df.columns = ["Number", "Count"]

# 按 Number 列排序
count_df["Number"] = count_df["Number"].astype(int)  # 确保 Number 列为整数类型
count_df = count_df.sort_values(by="Number").reset_index(drop=True)

# 设置显示选项以确保完整输出
pd.set_option("display.max_rows", None)  # 显示所有行
pd.set_option("display.max_columns", None)  # 显示所有

# 打印统计结果
print(count_df)

# 如果需要将结果保存到文件，可以使用：
# count_df.to_csv('ddr_counts.csv', index=False)
