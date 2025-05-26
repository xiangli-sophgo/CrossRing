import csv


def modify_id(num):
    num = int(num)
    if 0 <= num <= 15:
        return f"{num}_0"
    elif 16 <= num <= 31:
        return f"{num - 16}_1"
    else:
        # 如果编号不在0-31范围内，保持原样或根据需求处理
        return str(num)


input_file = r"../../traffic/output_v8_32_no_map/step5_data_merge/TPS009-Llama2-70B-S4K-O1-W8A8-B128-LMEM2M-AllReduce_Trace.txt"
output_file = r"../../traffic/output_v8_32_no_map/step5_data_merge/TPS009-Llama2-70B-S4K-O1-W8A8-B128-LMEM2M-AllReduce_Trace_group_map.txt"

with open(input_file, mode="r", newline="") as infile, open(output_file, mode="w", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        # row示例: ['36', '0', 'gdma', '1', 'ddr', 'W', '4']
        # 修改gdma编号（第2列）和ddr编号（第4列）
        # 注意gdma编号在第二个数字，即row[1]
        # ddr编号在第四个数字，即row[3]

        # 修改gdma编号
        gdma_num = int(row[1])
        if 0 <= gdma_num <= 15:
            row[2] = row[2] + "_0"
        elif 16 <= gdma_num <= 31:
            row[2] = row[2] + "_1"
            row[1] = str(gdma_num - 16)

        # 修改ddr编号
        ddr_num = int(row[3])
        if 0 <= ddr_num <= 15:
            row[4] = row[4] + "_0"
        elif 16 <= ddr_num <= 31:
            row[4] = row[4] + "_1"
            row[3] = str(ddr_num - 16)

        writer.writerow(row)

print(f"处理完成，结果保存在 {output_file}")
