#!/usr/bin/env python3
"""
数据流文件按源IP拆分工具

功能:
- 读取单Die格式traffic文件(7字段CSV)
- 按(src_node, src_ip)组合分组所有请求
- 为每个源IP创建独立输出文件

使用方式:
    python split_traffic.py <input_file> [output_dir]

输入格式:
    inject_time, src_node, src_ip, dst_node, dst_ip, req_type, burst_length

输出:
    输出目录: 输入文件名（不含扩展名）
    每个源IP一个文件，名称为: master_p{src_ip_index}_x{dst_x}_y{dst_y}.txt
    输出格式: inject_time,(p{dst_ip_index},x{x},y{y}),req_type,burst_length

    坐标映射(5x4拓扑，左下角为原点):
        node_id -> (x=node_id%4, y=4-node_id//4)
        节点0(左上角) -> (x=0, y=4)
        节点4 -> (x=0, y=3)
"""

import os
import sys
from collections import defaultdict
from pathlib import Path


def node_id_to_xy(node_id, num_col=4, num_row=5):
    """
    将节点ID转换为(x,y)坐标（左下角为原点，节点编号从左上角开始）

    Args:
        node_id: 节点ID（从左上角开始编号，行优先）
        num_col: 网络列数，默认4(5x4拓扑)
        num_row: 网络行数，默认5(5x4拓扑)

    Returns:
        (x, y)坐标元组，左下角为原点

    示例(5x4):
        节点0(左上角) -> (0, 4)
        节点1 -> (1, 4)
        节点4 -> (0, 3)
    """
    node_id = int(node_id)
    row_from_top = node_id // num_col  # 从顶部数的行号
    x = node_id % num_col
    y = num_row - 1 - row_from_top  # 转换为从底部数的y坐标
    return x, y


def extract_ip_index(ip_name):
    """
    从IP名称提取索引号

    Args:
        ip_name: IP名称，如 "gdma_0", "ddr_1"

    Returns:
        索引号字符串，如 "0", "1"
    """
    if '_' in ip_name:
        return ip_name.split('_')[-1]
    return "0"  # 默认索引


def split_traffic_by_source(input_file, output_dir=None, num_col=4, num_row=5):
    """
    按源IP拆分traffic文件

    Args:
        input_file: 输入traffic文件路径
        output_dir: 输出目录路径，默认为输入文件同目录下的split_output文件夹
        num_col: 网络列数，默认4(5x4拓扑)
        num_row: 网络行数，默认5(5x4拓扑)
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    # 确定输出目录（如果未指定，使用输入文件名作为输出目录名）
    if output_dir is None:
        # 使用输入文件名（不含扩展名）作为输出目录名
        output_dir = input_path.parent / input_path.stem
    else:
        output_dir = Path(output_dir)

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集每个源IP的请求
    traffic_by_source = defaultdict(list)

    print(f"读取文件: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # 跳过空行和注释
            if not line or line.startswith("#"):
                continue

            # 解析CSV行
            parts = [p.strip() for p in line.split(",")]

            if len(parts) != 7:
                print(f"警告: 第{line_num}行格式错误，应为7个字段，实际为{len(parts)}个，跳过该行")
                continue

            # 提取源节点和源IP
            src_node = parts[1]
            src_ip = parts[2]

            # 按(src_node, src_ip)分组
            key = (src_node, src_ip)
            traffic_by_source[key].append(line)

    # 写入拆分后的文件
    print(f"\n找到 {len(traffic_by_source)} 个不同的源IP")

    for (src_node, src_ip), lines in sorted(traffic_by_source.items()):
        # 提取第一行的目标信息作为文件名参考
        first_line_parts = lines[0].split(",")
        dst_node = first_line_parts[3].strip()
        dst_ip = first_line_parts[4].strip()

        # 提取源IP和目标IP的索引号
        src_ip_index = extract_ip_index(src_ip)
        dst_ip_index = extract_ip_index(dst_ip)

        # 将目标节点转换为(x,y)坐标
        dst_x, dst_y = node_id_to_xy(dst_node, num_col, num_row)

        # 文件名格式: master_p{src_ip_index}_x{dst_x}_y{dst_y}.txt
        filename = f"master_p{src_ip_index}_x{dst_x}_y{dst_y}.txt"

        # 直接在输出目录中创建文件，不创建子目录
        output_file = output_dir / filename

        with open(output_file, "w", encoding="utf-8") as f:
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                inject_time = parts[0]
                dst_node_id = parts[3]
                dst_ip_name = parts[4]
                req_type = parts[5]
                burst_length = parts[6]

                # 提取目标IP索引
                target_ip_index = extract_ip_index(dst_ip_name)

                # 将目标节点转换为(x,y)坐标
                x, y = node_id_to_xy(dst_node_id, num_col, num_row)

                # 格式: inject_time,(p{target_ip_index},x{x},y{y}),req_type,burst_length
                formatted_line = f"{inject_time},(p{target_ip_index},x{x},y{y}),{req_type},{burst_length}"
                f.write(formatted_line + "\n")

        print(f"  {filename}: {len(lines)} 条请求 -> {output_file}")

    print(f"\n拆分完成! 输出目录: {output_dir}")


def main():
    if len(sys.argv) < 2:
        print("用法: python split_traffic.py <input_file> [output_dir]")
        print("\n示例:")
        print("  python split_traffic.py test_data/test1.txt")
        print("  python split_traffic.py test_data/test1.txt ./output")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        split_traffic_by_source(input_file, output_dir)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
