#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模块：batch_modify_trace

提供批量修改 trace 文件中 gdma 和 ddr 编号的函数，供其他脚本调用。
"""

import os
import csv
from typing import Tuple


def split_id_suffix(num_str: str) -> Tuple[int, str]:
    """
    将编号拆分为（新的编号, 后缀）：
      - 0–15   → (原编号, '0')
      - 16–31  → (原编号-16, '1')
      - 其他   → (原编号, '')
    """
    num = int(num_str)
    if 0 <= num <= 15:
        return num, "0"
    elif 16 <= num <= 31:
        return num - 16, "1"
    else:
        return num, ""


def modify_row(row: list) -> list:
    """
    修改一行 CSV 数据中的 gdma（cols 1/2）和 ddr（cols 3/4）：
      - 调整编号（col1, col3）
      - 根据原编号范围给名称（col2, col4）追加后缀
    """
    # gdma
    new_id, suffix = split_id_suffix(row[1])
    row[1] = str(new_id)
    if suffix:
        row[2] = f"{row[2]}_{suffix}"

    # ddr
    new_id, suffix = split_id_suffix(row[3])
    row[3] = str(new_id)
    if suffix:
        row[4] = f"{row[4]}_{suffix}"

    return row


def process_file(in_path: str, out_path: str, delimiter: str = ",") -> None:
    """
    读取 in_path 文件，按行调用 modify_row 处理后写入 out_path。
    自动创建输出目录。
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(in_path, "r", newline="") as fin, open(out_path, "w", newline="") as fout:
        reader = csv.reader(fin, delimiter=delimiter)
        writer = csv.writer(fout, delimiter=delimiter)
        for row in reader:
            if not row:
                continue
            writer.writerow(modify_row(row))


def main(input_dir: str, output_dir: str, ext: str = ".txt", delimiter: str = ",") -> None:
    """
    批量遍历 input_dir 下所有以 ext 结尾的文件，修改后保存到 output_dir。
    保留子目录结构。
    参数：
      - input_dir:  源文件夹路径
      - output_dir: 目标文件夹路径（不存在则创建）
      - ext:        文件扩展名过滤，默认 ".txt"
      - delimiter:  CSV 分隔符，默认 ","
    """
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        rel = os.path.relpath(root, input_dir)
        for fname in files:
            if not fname.endswith(ext):
                continue
            in_path = os.path.join(root, fname)
            out_sub = os.path.join(output_dir, rel)
            out_path = os.path.join(out_sub, fname)
            process_file(in_path, out_path, delimiter=delimiter)


# 如果作为脚本直接运行，可以在此处调用示例（可选）
if __name__ == "__main__":
    # 示例：直接运行时修改下面两个路径
    main(input_dir="../../traffic/output_v8_32_no_map/step5_data_merge", output_dir="../../traffic/output_v8_32_no_map/step5_data_merge_mod", ext=".txt", delimiter=",")
    print("批量修改完成。")
