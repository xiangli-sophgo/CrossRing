import csv
import argparse

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="Convert a TXT file to CSV format.")
parser.add_argument("input_file", type=str, help="The input TXT file to convert")
# parser.add_argument("output_file", type=str, help="The output CSV file")

# 解析命令行参数
args = parser.parse_args()
args.output_file = args.input_file[:-3] + "csv"

# 读取 TXT 文件并写入 CSV 文件
with open(args.input_file, "r", encoding="utf-8") as txt_file:
    # 读取 TXT 文件内容
    lines = txt_file.readlines()

    # 打开 CSV 文件进行写入
    with open(args.output_file, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)

        # 遍历每一行并写入 CSV 文件
        for line in lines:
            # 去除行末的换行符，并按逗号分割
            row = line.strip().split("\t")
            # 写入 CSV 文件
            csv_writer.writerow(row)

print(f"成功将 {args.input_file} 转换为 {args.output_file}")
