"""
流量生成引擎 - 封装generate_data.py为可调用函数

提供Web界面可调用的流量生成API,支持返回DataFrame和文件双输出
集成流量拆分功能
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


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
    end_time=None,
    output_file=None,
    random_seed=42,
    return_dataframe=True,
    topo_type=None,
    routing_type="XY",
    node_ips=None
):
    """
    根据多个配置生成并合并流量数据

    :param configs: 配置列表
    :param end_time: 结束时间（废弃，现使用配置中的end_time）
    :param output_file: 输出文件路径(可选,为None则不写文件)
    :param random_seed: 随机种子
    :param return_dataframe: 是否返回DataFrame
    :param topo_type: 拓扑类型（如"5x4"），用于元数据
    :param routing_type: 路由算法（"XY"或"YX"），用于静态带宽计算
    :param node_ips: 节点IP映射字典，用于静态带宽计算
    :return: (file_path, dataframe) 或 file_path 或 dataframe
    """
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)

    all_data = []

    # 为每个配置生成数据，使用配置自己的end_time
    for config in configs:
        config_end_time = config.get("end_time", end_time if end_time else 6000)
        data = generate_traffic_from_config(config, config_end_time)
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
            # 写入元数据行 - 只保留configs
            if configs:
                import json
                meta_data = {
                    "configs": [
                        {k: v for k, v in cfg.items() if not k.startswith('_')} if isinstance(cfg, dict) else cfg.__dict__
                        for cfg in configs
                    ]
                }
                meta_json = json.dumps(meta_data, separators=(',', ':'), ensure_ascii=False)
                f.write(f"# TRAFFIC_META: {meta_json}\n")

            # 写入traffic数据
            for timestamp, src_pos, src_type, dst_pos, dst_type, req_type, burst in all_data:
                f.write(f"{timestamp},{src_pos},{src_type},{dst_pos},{dst_type},{req_type},{burst}\n")
        file_path = str(output_path.absolute())

    # 转换为DataFrame,显式指定数据类型避免Arrow序列化错误
    df = None
    if return_dataframe:
        df = pd.DataFrame(
            all_data,
            columns=["timestamp", "src_pos", "src_type", "dst_pos", "dst_type", "req_type", "burst"]
        )
        # 确保数据类型正确
        df = df.astype({
            "timestamp": "int64",
            "src_pos": "int64",
            "src_type": "str",
            "dst_pos": "int64",
            "dst_type": "str",
            "req_type": "str",
            "burst": "int64"
        })

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


# ========== 流量拆分功能 (从split_traffic.py集成) ==========

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


def split_traffic_by_source(input_file, output_dir=None, num_col=4, num_row=5, verbose=True):
    """
    按源IP拆分traffic文件

    Args:
        input_file: 输入traffic文件路径
        output_dir: 输出目录路径，默认为输入文件同目录下的split_output文件夹
        num_col: 网络列数，默认4(5x4拓扑)
        num_row: 网络行数，默认5(5x4拓扑)
        verbose: 是否打印详细信息

    Returns:
        拆分结果字典: {
            "output_dir": 输出目录路径,
            "files": [{"filename": 文件名, "count": 请求数, "path": 完整路径}],
            "total_sources": 总源IP数
        }
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

    if verbose:
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
                if verbose:
                    print(f"警告: 第{line_num}行格式错误，应为7个字段，实际为{len(parts)}个，跳过该行")
                continue

            # 提取源节点和源IP
            src_node = parts[1]
            src_ip = parts[2]

            # 按(src_node, src_ip)分组 - 每个节点上的每个IP都是独立的
            key = (src_node, src_ip)
            traffic_by_source[key].append(line)

    # 写入拆分后的文件
    if verbose:
        print(f"\n找到 {len(traffic_by_source)} 个不同的源IP")

    result_files = []

    for (src_node, src_ip), lines in sorted(traffic_by_source.items()):
        # 提取源IP索引号
        src_ip_index = extract_ip_index(src_ip)

        # 将源节点转换为(x,y)坐标
        src_x, src_y = node_id_to_xy(int(src_node), num_col, num_row)

        # 提取第一行的目标信息作为文件名参考
        first_line_parts = lines[0].split(",")
        dst_node = first_line_parts[3].strip()

        # 将目标节点转换为(x,y)坐标
        dst_x, dst_y = node_id_to_xy(int(dst_node), num_col, num_row)

        # 文件名格式: master_p{src_ip_index}_x{src_x}_y{src_y}.txt
        # 注意: 这里使用源节点的坐标,而不是目标节点
        filename = f"master_p{src_ip_index}_x{src_x}_y{src_y}.txt"

        # 直接在输出目录中创建文件
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

        result_files.append({
            "filename": filename,
            "count": len(lines),
            "path": str(output_file.absolute())
        })

        if verbose:
            print(f"  {filename}: {len(lines)} 条请求 -> {output_file}")

    if verbose:
        print(f"\n拆分完成! 输出目录: {output_dir}")

    return {
        "output_dir": str(output_dir.absolute()),
        "files": result_files,
        "total_sources": len(traffic_by_source)
    }


def split_d2d_traffic_by_source(input_file, output_dir=None, num_col=4, num_row=5, verbose=True):
    """
    按源IP拆分D2D traffic文件

    Args:
        input_file: 输入D2D traffic文件路径
        output_dir: 输出目录路径
        num_col: 网络列数，默认4(5x4拓扑)
        num_row: 网络行数，默认5(5x4拓扑)
        verbose: 是否打印详细信息

    Returns:
        拆分结果字典
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    # 确定输出目录
    if output_dir is None:
        output_dir = input_path.parent / input_path.stem
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集每个源IP的请求
    traffic_by_source = defaultdict(list)

    if verbose:
        print(f"读取D2D文件: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            # D2D格式: timestamp,src_die,src_node,src_ip,dst_die,dst_node,dst_ip,req_type,burst
            parts = [p.strip() for p in line.split(",")]

            if len(parts) != 9:
                if verbose:
                    print(f"警告: 第{line_num}行格式错误，应为9个字段，实际为{len(parts)}个，跳过该行")
                continue

            # 提取源Die、源节点和源IP
            src_die = parts[1]
            src_node = parts[2]
            src_ip = parts[3]

            # 按(src_die, src_node, src_ip)分组
            key = (src_die, src_node, src_ip)
            traffic_by_source[key].append(line)

    if verbose:
        print(f"\n找到 {len(traffic_by_source)} 个不同的源IP")

    result_files = []

    for (src_die, src_node, src_ip), lines in sorted(traffic_by_source.items()):
        # 提取源IP索引号
        src_ip_index = extract_ip_index(src_ip)

        # 将源节点转换为(x,y)坐标
        src_x, src_y = node_id_to_xy(int(src_node), num_col, num_row)

        # 文件名格式: master_d{die_id}_p{src_ip_index}_x{src_x}_y{src_y}.txt
        filename = f"master_d{src_die}_p{src_ip_index}_x{src_x}_y{src_y}.txt"

        output_file = output_dir / filename

        with open(output_file, "w", encoding="utf-8") as f:
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                inject_time = parts[0]
                dst_die = parts[4]
                dst_node_id = parts[5]
                dst_ip_name = parts[6]
                req_type = parts[7]
                burst_length = parts[8]

                # 提取目标IP索引
                target_ip_index = extract_ip_index(dst_ip_name)

                # 将目标节点转换为(x,y)坐标
                x, y = node_id_to_xy(int(dst_node_id), num_col, num_row)

                # 格式: inject_time,(D{dst_die},p{target_ip_index},x{x},y{y}),req_type,burst_length
                formatted_line = f"{inject_time},(D{dst_die},p{target_ip_index},x{x},y{y}),{req_type},{burst_length}"
                f.write(formatted_line + "\n")

        result_files.append({
            "filename": filename,
            "count": len(lines),
            "path": str(output_file.absolute())
        })

        if verbose:
            print(f"  {filename}: {len(lines)} 条请求 -> {output_file}")

    if verbose:
        print(f"\n拆分完成! 输出目录: {output_dir}")

    return {
        "output_dir": str(output_dir.absolute()),
        "files": result_files,
        "total_sources": len(traffic_by_source)
    }


# ========== D2D流量生成功能 ==========

def generate_d2d_traffic_from_config(config, end_time):
    """
    根据单个D2D配置生成流量数据

    :param config: 配置字典,包含:
        - src_die: 源Die编号
        - dst_die: 目标Die编号
        - src_map: 源IP映射 {"ip_type": [positions]}
        - dst_map: 目标IP映射 {"ip_type": [positions]}
        - speed: 带宽 (GB/s)
        - burst: burst长度
        - req_type: 请求类型 ("R" 或 "W")
    :param end_time: 结束时间
    :return: D2D流量数据列表 (9字段格式)
    """
    generator = TrafficGenerator()

    # 获取配置参数
    src_die = config["src_die"]
    dst_die = config["dst_die"]
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
                    # D2D格式: inject_time, src_die, src_node, src_ip, dst_die, dst_node, dst_ip, req_type, burst_length
                    data.append((timestamp, src_die, src_pos, src_type, dst_die, dst_pos, dst_type, req_type, burst))

        cycle += 1

    return data


def generate_d2d_traffic_from_configs(
    configs,
    end_time=None,
    output_file=None,
    random_seed=42,
    return_dataframe=True,
    topo_type=None,
    routing_type="XY",
    node_ips=None,
    d2d_config=None
):
    """
    根据多个D2D配置生成并合并流量数据

    :param configs: D2D配置列表
    :param end_time: 结束时间（废弃，现使用配置中的end_time）
    :param output_file: 输出文件路径(可选,为None则不写文件)
    :param random_seed: 随机种子
    :param return_dataframe: 是否返回DataFrame
    :param topo_type: 拓扑类型（如"5x4"），用于元数据
    :param routing_type: 路由算法（"XY"或"YX"），用于静态带宽计算
    :param node_ips: 节点IP映射字典，用于静态带宽计算
    :param d2d_config: D2D配置字典（包含num_dies, d2d_connections）
    :return: (file_path, dataframe) 或 file_path 或 dataframe
    """
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)

    all_data = []

    # 为每个配置生成数据，使用配置自己的end_time
    for config in configs:
        config_end_time = config.get("end_time", end_time if end_time else 6000)
        data = generate_d2d_traffic_from_config(config, config_end_time)
        all_data.extend(data)

    # 按时间和源位置排序
    all_data.sort(key=lambda x: (x[0], x[2]))  # 按timestamp和src_node排序

    # 写入文件
    file_path = None
    if output_file:
        # 确保目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            # 写入元数据行 - 只保留configs
            if configs:
                import json
                meta_data = {
                    "configs": [
                        {k: v for k, v in cfg.items() if not k.startswith('_')} if isinstance(cfg, dict) else cfg.__dict__
                        for cfg in configs
                    ]
                }
                meta_json = json.dumps(meta_data, separators=(',', ':'), ensure_ascii=False)
                f.write(f"# TRAFFIC_META: {meta_json}\n")

            # 写入traffic数据
            for timestamp, src_die, src_pos, src_type, dst_die, dst_pos, dst_type, req_type, burst in all_data:
                f.write(f"{timestamp},{src_die},{src_pos},{src_type},{dst_die},{dst_pos},{dst_type},{req_type},{burst}\n")
        file_path = str(output_path.absolute())

    # 转换为DataFrame,显式指定数据类型避免Arrow序列化错误
    df = None
    if return_dataframe:
        df = pd.DataFrame(
            all_data,
            columns=["timestamp", "src_die", "src_pos", "src_type", "dst_die", "dst_pos", "dst_type", "req_type", "burst"]
        )
        # 确保数据类型正确
        df = df.astype({
            "timestamp": "int64",
            "src_die": "int64",
            "src_pos": "int64",
            "src_type": "str",
            "dst_die": "int64",
            "dst_pos": "int64",
            "dst_type": "str",
            "req_type": "str",
            "burst": "int64"
        })

    # 返回结果
    if output_file and return_dataframe:
        return file_path, df
    elif output_file:
        return file_path
    else:
        return df
