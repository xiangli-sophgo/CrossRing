import csv
from collections import defaultdict

# 初始化统计字典
statistics = defaultdict(lambda: defaultdict(int))

# 读取CSV文件（这里用字符串模拟，实际使用时应替换为文件路径）
with open(r"../Result/CrossRing/SCM/REQ_RSP/Spare_core_0403_to320_2_0/5x4/demo45/Result_R.txt") as f:
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        src_id = int(row[1])
        des_id = int(row[3])
        statistics[src_id][des_id] += 1

    # 排序输出
    print("Src_ID -> {Des_ID: Count} 统计结果:")
    for src_id in sorted(statistics.keys()):  # 按 src_id 升序排序
        des_counts = statistics[src_id]
        # 按 des_id 升序排序
        sorted_des_counts = dict(sorted(des_counts.items()))
        print(f"{src_id} -> {sorted_des_counts}")
