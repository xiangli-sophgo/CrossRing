"""
流量生成引擎 - 封装generate_data.py为可调用函数

提供Web界面可调用的流量生成API,支持返回DataFrame和文件双输出
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path


class TrafficGenerator:
    """流量生成器,计算发送时间点"""

    def __init__(self, total_bandwidth=128):
        """
        :param total_bandwidth: 总带宽基准值(GB/s)
        """
        self.total_bandwidth = total_bandwidth

    def calculate_time_points(self, speed, burst, duration=1280):
        """
        计算发送时间点序列

        :param speed: 实际速度 (GB/s)
        :param burst: burst长度
        :param duration: 时间窗口持续时间 (ns)
        :return: 时间点列表
        """
        # 计算总传输次数
        total_transfers_float = speed * duration / (self.total_bandwidth * burst)
        total_transfers = max(1, round(total_transfers_float))

        if total_transfers == 0:
            return []

        # 均匀分布时间点
        time_points = [int(i * duration / total_transfers) for i in range(total_transfers)]
        return time_points


def generate_traffic_from_config(config, end_time):
    """
    根据单个配置生成流量数据

    :param config: 配置字典,包含:
        - src_map: 源IP映射 {"ip_type": [positions]}
        - dst_map: 目标IP映射 {"ip_type": [positions]}
        - speed: 带宽 (GB/s)
        - burst: burst长度
        - req_type: 请求类型 ("R" 或 "W")
    :param end_time: 结束时间
    :return: 流量数据列表
    """
    generator = TrafficGenerator()

    # 获取配置参数
    src_map = config["src_map"]
    dst_map = config["dst_map"]
    speed = config["speed"]
    burst = config["burst"]
    req_type = config["req_type"]

    # 扁平化源和目标映射
    src_items = [(ip_type, pos) for ip_type, positions in src_map.items() for pos in positions]
    dst_items = [(ip_type, pos) for ip_type, positions in dst_map.items() for pos in positions]

    if not src_items or not dst_items:
        return []

    # 计算时间点模式
    duration = 1280  # 读写时间窗口
    time_pattern = generator.calculate_time_points(speed, burst, duration)

    data = []
    cycle = 0

    while True:
        base_time = cycle * duration

        # 检查是否超过结束时间
        if base_time >= end_time:
            break

        # 为每个时间点生成流量
        for t in time_pattern:
            for src_type, src_pos in src_items:
                # 随机选择目标
                dst_type, dst_pos = random.choice(dst_items)

                timestamp = base_time + t
                if timestamp < end_time:
                    data.append((timestamp, src_pos, src_type, dst_pos, dst_type, req_type, burst))

        cycle += 1

    return data


def generate_traffic_from_configs(
    configs,
    end_time,
    output_file=None,
    random_seed=42,
    return_dataframe=True
):
    """
    根据多个配置生成并合并流量数据

    :param configs: 配置列表
    :param end_time: 结束时间
    :param output_file: 输出文件路径(可选,为None则不写文件)
    :param random_seed: 随机种子
    :param return_dataframe: 是否返回DataFrame
    :return: (file_path, dataframe) 或 file_path 或 dataframe
    """
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)

    all_data = []

    # 为每个配置生成数据
    for config in configs:
        data = generate_traffic_from_config(config, end_time)
        all_data.extend(data)

    # 按时间和源位置排序
    all_data.sort(key=lambda x: (x[0], x[1]))

    # 写入文件
    file_path = None
    if output_file:
        # 确保目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            for timestamp, src_pos, src_type, dst_pos, dst_type, req_type, burst in all_data:
                f.write(f"{timestamp},{src_pos},{src_type},{dst_pos},{dst_type},{req_type},{burst}\n")
        file_path = str(output_path.absolute())

    # 转换为DataFrame
    df = None
    if return_dataframe:
        df = pd.DataFrame(
            all_data,
            columns=["timestamp", "src_pos", "src_type", "dst_pos", "dst_type", "req_type", "burst"]
        )

    # 返回结果
    if output_file and return_dataframe:
        return file_path, df
    elif output_file:
        return file_path
    else:
        return df


def get_default_ip_mappings(topo_type="5x4"):
    """
    获取默认IP位置映射

    :param topo_type: 拓扑类型 ("5x4" 或 "4x4")
    :return: IP映射字典
    """
    if topo_type == "5x4":
        return {
            "gdma": [6, 7, 26, 27],
            "ddr": [12, 13, 32, 33],
            "l2m": [18, 19, 38, 39],
            "sdma": [0, 1, 20, 21],
            "cdma": [14, 15, 34],
        }
    elif topo_type == "4x4":
        return {
            "gdma": [0, 1, 2, 3],
            "ddr": [12, 13, 14, 15],
            "l2m": [8, 9, 10, 11],
            "sdma": [4, 5, 6, 7],
            "cdma": [14, 15],
        }
    else:
        raise ValueError(f"不支持的拓扑类型: {topo_type}")
