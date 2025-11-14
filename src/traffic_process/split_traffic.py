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
    每个(src_node, src_ip)组合一个文件: node{src_node}_{src_ip}.txt
"""

import os
import sys
from collections import defaultdict
from pathlib import Path


def split_traffic_by_source(input_file, output_dir=None):
    """
    按源IP拆分traffic文件

    Args:
        input_file: 输入traffic文件路径
        output_dir: 输出目录路径，默认为输入文件同目录下的split_output文件夹
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    # 确定输出目录
    if output_dir is None:
        output_dir = input_path.parent / "split_output"
    else:
        output_dir = Path(output_dir)

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集每个源IP的请求
    traffic_by_source = defaultdict(list)

    print(f"读取文件: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue

            # 解析CSV行
            parts = [p.strip() for p in line.split(',')]

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
        output_file = output_dir / f"node{src_node}_{src_ip}.txt"

        with open(output_file, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')

        print(f"  node{src_node}_{src_ip}: {len(lines)} 条请求 -> {output_file}")

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
