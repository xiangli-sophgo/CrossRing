#!/usr/bin/env python3
import argparse
import sys
import json
import configparser
import os
from typing import List, Tuple, Dict, Any


# 配置变量 - 可以通过这些变量控制默认行为
DEFAULT_CONFIG = {"input_files": [], "output_file": "merged_traffic.txt", "mode": "serial", "interval": 30, "config_file": None}

# 全局配置变量
CONFIG = DEFAULT_CONFIG.copy()


def load_config_from_file(config_file: str) -> Dict[str, Any]:
    """从配置文件加载配置"""
    config = {}

    if config_file.endswith(".json"):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading JSON config file {config_file}: {e}")
            sys.exit(1)

    elif config_file.endswith(".ini") or config_file.endswith(".cfg"):
        try:
            parser = configparser.ConfigParser()
            parser.read(config_file)

            if "traffic_merge" in parser:
                section = parser["traffic_merge"]
                config = {
                    "input_files": section.get("input_files", "").split(",") if section.get("input_files") else [],
                    "output_file": section.get("output_file", DEFAULT_CONFIG["output_file"]),
                    "mode": section.get("mode", DEFAULT_CONFIG["mode"]),
                    "interval": section.getint("interval", DEFAULT_CONFIG["interval"]),
                }
                # 清理空字符串
                config["input_files"] = [f.strip() for f in config["input_files"] if f.strip()]
        except (FileNotFoundError, configparser.Error) as e:
            print(f"Error reading INI config file {config_file}: {e}")
            sys.exit(1)

    else:
        print(f"Error: Unsupported config file format: {config_file}")
        sys.exit(1)

    return config


def set_config_variables(**kwargs):
    """设置配置变量"""
    global CONFIG
    for key, value in kwargs.items():
        if key in CONFIG:
            CONFIG[key] = value
        else:
            print(f"Warning: Unknown config key: {key}")


def get_config() -> Dict[str, Any]:
    """获取当前配置"""
    return CONFIG.copy()


def parse_traffic_line(line: str) -> Tuple[int, int, str, int, str, str, int]:
    """解析traffic数据行"""
    parts = line.strip().split(",")
    if len(parts) != 7:
        raise ValueError(f"Invalid line format: {line}")

    time = int(parts[0])
    gdma_node = int(parts[1])
    gdma_channel = parts[2]
    ddr_node = int(parts[3])
    ddr_channel = parts[4]
    request_type = parts[5]
    burst = int(parts[6])

    return (time, gdma_node, gdma_channel, ddr_node, ddr_channel, request_type, burst)


def format_traffic_line(data: Tuple[int, int, str, int, str, str, int]) -> str:
    """格式化traffic数据行"""
    return f"{data[0]},{data[1]},{data[2]},{data[3]},{data[4]},{data[5]},{data[6]}"


def read_traffic_file(filename: str) -> List[Tuple[int, int, str, int, str, str, int]]:
    """读取traffic文件"""
    traffic_data = []
    try:
        with open(filename, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                try:
                    data = parse_traffic_line(line)
                    traffic_data.append(data)
                except ValueError as e:
                    print(f"Error parsing line {line_num} in {filename}: {e}")
                    sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        sys.exit(1)

    return traffic_data


def write_traffic_file(filename: str, traffic_data: List[Tuple[int, int, str, int, str, str, int]]):
    """写入traffic文件"""
    try:
        with open(filename, "w") as f:
            for data in traffic_data:
                f.write(format_traffic_line(data) + "\n")
        print(f"Output written to {filename}")
    except IOError as e:
        print(f"Error writing to {filename}: {e}")
        sys.exit(1)


def get_max_time(traffic_data: List[Tuple[int, int, str, int, str, str, int]]) -> int:
    """获取traffic数据中的最大时间戳"""
    if not traffic_data:
        return 0
    return max(data[0] for data in traffic_data)


def serial_concat(traffic_files: List[str], interval: int) -> List[Tuple[int, int, str, int, str, str, int]]:
    """串行拼接多个traffic文件"""
    result = []
    current_offset = 0

    for i, filename in enumerate(traffic_files):
        print(f"Processing file {i+1}/{len(traffic_files)}: {filename}")
        traffic_data = read_traffic_file(filename)

        if not traffic_data:
            print(f"Warning: {filename} is empty, skipping")
            continue

        # 调整时间戳
        adjusted_data = []
        for data in traffic_data:
            new_time = data[0] + current_offset
            adjusted_data.append((new_time, data[1], data[2], data[3], data[4], data[5], data[6]))

        result.extend(adjusted_data)

        # 更新下一个traffic的时间偏移量
        if i < len(traffic_files) - 1:  # 不是最后一个文件
            max_time = get_max_time(adjusted_data)
            current_offset = max_time + interval
            print(f"  Max time: {max_time}, Next offset: {current_offset}")

    return result


def parallel_merge(traffic_files: List[str]) -> List[Tuple[int, int, str, int, str, str, int]]:
    """并行合并多个traffic文件（按时间排序）"""
    all_data = []

    for filename in traffic_files:
        print(f"Reading file: {filename}")
        traffic_data = read_traffic_file(filename)
        all_data.extend(traffic_data)

    # 按时间戳排序
    all_data.sort(key=lambda x: x[0])
    return all_data


def create_sample_config_files():
    """创建示例配置文件"""
    # JSON配置文件示例
    json_config = {"input_files": ["traffic1.txt", "traffic2.txt", "traffic3.txt"], "output_file": "merged_output.txt", "mode": "serial", "interval": 30}

    with open("config_example.json", "w") as f:
        json.dump(json_config, f, indent=2)

    # INI配置文件示例
    ini_content = """[traffic_merge]
                    input_files = traffic1.txt, traffic2.txt, traffic3.txt
                    output_file = merged_output.txt
                    mode = serial
                    interval = 50
                    """

    with open("config_example.ini", "w") as f:
        f.write(ini_content)

    print("Created sample config files: config_example.json, config_example.ini")


def process_traffic_files(input_files: List[str] = None, output_file: str = None, mode: str = None, interval: int = None) -> None:
    """处理traffic文件的主要函数"""
    # 使用传入的参数或配置变量
    config = get_config()

    files = input_files or config["input_files"]
    output = output_file or config["output_file"]
    process_mode = mode or config["mode"]
    time_interval = interval or config["interval"]

    if not files:
        print("Error: 没有指定输入文件")
        return

    if len(files) < 2:
        print("Error: 至少需要2个输入文件")
        return

    print(f"模式: {process_mode}")
    print(f"输入文件: {files}")
    print(f"输出文件: {output}")

    if process_mode == "serial":
        print(f"间隔时间: {time_interval}ns")
        result = serial_concat(files, time_interval)
    else:  # parallel
        result = parallel_merge(files)

    if result:
        print(f"总共处理了 {len(result)} 条记录")
        write_traffic_file(output, result)
    else:
        print("Warning: 没有数据输出")


def main():
    parser = argparse.ArgumentParser(description="Traffic拼接和合并工具")
    parser.add_argument("mode", nargs="?", choices=["serial", "parallel"], help="处理模式: serial(串行拼接) 或 parallel(并行合并)")
    parser.add_argument("-i", "--input", nargs="+", help="输入的traffic文件列表")
    parser.add_argument("-o", "--output", help="输出文件名")
    parser.add_argument("--interval", type=int, help="串行拼接时的间隔时间(ns), 默认30ns")
    parser.add_argument("-c", "--config", help="配置文件路径 (支持.json, .ini, .cfg)")
    parser.add_argument("--create-config", action="store_true", help="创建示例配置文件")

    args = parser.parse_args()

    # # 创建示例配置文件
    # if args.create_config:
    #     create_sample_config_files()
    #     return

    # # 加载配置文件
    # if args.config:
    #     if os.path.exists(args.config):
    #         file_config = load_config_from_file(args.config)
    #         set_config_variables(**file_config)
    #         print(f"Loaded config from: {args.config}")
    #     else:
    #         print(f"Error: Config file {args.config} not found")
    #         sys.exit(1)

    # # 命令行参数覆盖配置文件参数
    # if args.input:
    #     set_config_variables(input_files=args.input)
    # if args.output:
    #     set_config_variables(output_file=args.output)
    # if args.mode:
    #     set_config_variables(mode=args.mode)
    # if args.interval is not None:
    #     set_config_variables(interval=args.interval)

    set_config_variables(
        input_files=[
            r"../../traffic/DeepSeek_0616/step6_ch_map/MLP_MoE.txt",
        ]
        * 9,
        output_file=r"../../traffic/DeepSeek_0616/step6_ch_map/MLP_merge.txt",
        mode="serial",
        # mode="parallel",
        interval=0,
    )

    # 处理文件
    process_traffic_files()


# 使用示例函数
def example_usage():
    """使用示例"""
    print("=== 使用示例 ===")

    # 1. 通过变量设置配置
    set_config_variables(
        input_files=[
            r"../../traffic/DeepSeek_0616/step6_ch_map/MLP_MoE.txt",
            r"../../traffic/DeepSeek_0616/step6_ch_map/MLP_MoE.txt",
        ],
        output_file=r"../../traffic/DeepSeek_0616/step6_ch_map/MLP_merge.txt",
        mode="serial",
        interval=0,
    )

    # 2. 直接调用处理函数
    # process_traffic_files()

    # 3. 或者传入特定参数
    # process_traffic_files(
    #     input_files=['a.txt', 'b.txt'],
    #     output_file='custom.txt',
    #     mode='parallel'
    # )


if __name__ == "__main__":
    main()
