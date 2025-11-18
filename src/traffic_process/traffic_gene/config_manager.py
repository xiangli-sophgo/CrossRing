"""
配置管理模块 - 流量配置的验证、管理和预估

提供配置数据结构、参数验证和生成前预估统计功能
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple
import copy


@dataclass
class TrafficConfig:
    """流量配置数据类"""

    src_map: Dict[str, List[int]]  # 源IP映射 {"ip_type": [positions]}
    dst_map: Dict[str, List[int]]  # 目标IP映射 {"ip_type": [positions]}
    speed: float  # 带宽 (GB/s)
    burst: int  # burst长度
    req_type: str  # 请求类型 ("R" 或 "W")
    config_id: int = 0  # 配置ID
    src_die: int = 0  # 源Die编号 (D2D模式使用)
    dst_die: int = 0  # 目标Die编号 (D2D模式使用)

    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'TrafficConfig':
        """从字典创建配置"""
        return cls(**data)

    def get_source_count(self) -> int:
        """获取源节点总数"""
        return sum(len(positions) for positions in self.src_map.values())

    def get_destination_count(self) -> int:
        """获取目标节点总数"""
        return sum(len(positions) for positions in self.dst_map.values())

    def get_source_nodes(self) -> List[int]:
        """获取所有源节点列表"""
        nodes = []
        for positions in self.src_map.values():
            nodes.extend(positions)
        return sorted(set(nodes))

    def get_destination_nodes(self) -> List[int]:
        """获取所有目标节点列表"""
        nodes = []
        for positions in self.dst_map.values():
            nodes.extend(positions)
        return sorted(set(nodes))


class ConfigValidator:
    """配置验证器"""

    def __init__(self, num_nodes: int, total_bandwidth: float = 128.0):
        """
        :param num_nodes: 拓扑总节点数
        :param total_bandwidth: 总带宽 (GB/s)
        """
        self.num_nodes = num_nodes
        self.total_bandwidth = total_bandwidth

    def validate(self, config: TrafficConfig) -> List[str]:
        """
        验证配置的合法性

        :param config: 流量配置
        :return: 错误信息列表,空列表表示验证通过
        """
        errors = []

        # 1. 检查源映射不为空
        if not config.src_map or all(not positions for positions in config.src_map.values()):
            errors.append("源IP映射不能为空")

        # 2. 检查目标映射不为空
        if not config.dst_map or all(not positions for positions in config.dst_map.values()):
            errors.append("目标IP映射不能为空")

        # 3. 检查节点范围
        all_src_nodes = config.get_source_nodes()
        all_dst_nodes = config.get_destination_nodes()

        for node in all_src_nodes:
            if node < 0 or node >= self.num_nodes:
                errors.append(f"源节点 {node} 超出范围 [0, {self.num_nodes - 1}]")

        for node in all_dst_nodes:
            if node < 0 or node >= self.num_nodes:
                errors.append(f"目标节点 {node} 超出范围 [0, {self.num_nodes - 1}]")

        # 4. 检查带宽合法性
        if config.speed <= 0:
            errors.append("带宽必须大于0")
        elif config.speed > self.total_bandwidth:
            errors.append(f"带宽 {config.speed} GB/s 超过总带宽 {self.total_bandwidth} GB/s")

        # 5. 检查burst合法性
        valid_bursts = [1, 2, 4, 8, 16]
        if config.burst not in valid_bursts:
            errors.append(f"Burst长度应为 {valid_bursts} 之一,当前值: {config.burst}")

        # 6. 检查请求类型
        if config.req_type not in ["R", "W"]:
            errors.append(f"请求类型必须是 'R' 或 'W',当前值: {config.req_type}")

        return errors


class TrafficEstimator:
    """流量预估器"""

    def __init__(self, total_bandwidth: float = 128.0, duration: int = 1280):
        """
        :param total_bandwidth: 总带宽 (GB/s)
        :param duration: 时间窗口持续时间 (ns)
        """
        self.total_bandwidth = total_bandwidth
        self.duration = duration

    def estimate_single_config(
        self,
        config: TrafficConfig,
        end_time: int
    ) -> Dict[str, any]:
        """
        预估单个配置的流量统计

        :param config: 流量配置
        :param end_time: 结束时间 (ns)
        :return: 预估统计字典
        """
        # 计算每个时间窗口的传输次数
        total_transfers_per_window = config.speed * self.duration / (
            self.total_bandwidth * config.burst
        )
        transfers_per_window = max(1, round(total_transfers_per_window))

        # 计算时间窗口数量
        num_windows = end_time / self.duration

        # 计算总请求数
        src_count = config.get_source_count()
        total_requests = transfers_per_window * src_count * num_windows

        # 计算请求频率
        requests_per_ns = total_requests / end_time if end_time > 0 else 0

        # 计算节点平均负载
        src_node_load = total_requests / src_count if src_count > 0 else 0
        dst_count = config.get_destination_count()
        dst_node_load = total_requests / dst_count if dst_count > 0 else 0

        return {
            "total_requests": int(total_requests),
            "requests_per_ns": requests_per_ns,
            "requests_per_window": transfers_per_window,
            "num_windows": int(num_windows),
            "src_node_count": src_count,
            "dst_node_count": dst_count,
            "src_node_load": src_node_load,
            "dst_node_load": dst_node_load,
            "bandwidth_gb_s": config.speed,
            "burst_length": config.burst,
            "req_type": config.req_type,
        }

    def estimate_multiple_configs(
        self,
        configs: List[TrafficConfig],
        end_time: int
    ) -> Dict[str, any]:
        """
        预估多个配置的合并流量统计

        :param configs: 流量配置列表
        :param end_time: 结束时间 (ns)
        :return: 合并预估统计字典
        """
        if not configs:
            return {
                "total_requests": 0,
                "total_bandwidth": 0,
                "read_requests": 0,
                "write_requests": 0,
                "num_configs": 0,
                "config_stats": []
            }

        # 统计每个配置
        config_stats = []
        total_requests = 0
        read_requests = 0
        write_requests = 0
        total_bandwidth = 0
        all_src_nodes = set()
        all_dst_nodes = set()

        for config in configs:
            stats = self.estimate_single_config(config, end_time)
            config_stats.append(stats)

            total_requests += stats["total_requests"]
            total_bandwidth += config.speed

            if config.req_type == "R":
                read_requests += stats["total_requests"]
            else:
                write_requests += stats["total_requests"]

            all_src_nodes.update(config.get_source_nodes())
            all_dst_nodes.update(config.get_destination_nodes())

        return {
            "total_requests": total_requests,
            "total_bandwidth": total_bandwidth,
            "read_requests": read_requests,
            "write_requests": write_requests,
            "read_ratio": read_requests / total_requests if total_requests > 0 else 0,
            "write_ratio": write_requests / total_requests if total_requests > 0 else 0,
            "num_configs": len(configs),
            "unique_src_nodes": len(all_src_nodes),
            "unique_dst_nodes": len(all_dst_nodes),
            "avg_requests_per_ns": total_requests / end_time if end_time > 0 else 0,
            "config_stats": config_stats,
        }


class ConfigManager:
    """配置管理器"""

    def __init__(self, num_nodes: int = 40):
        """
        :param num_nodes: 拓扑总节点数
        """
        self.num_nodes = num_nodes
        self.configs: List[TrafficConfig] = []
        self.next_id = 1
        self.validator = ConfigValidator(num_nodes)
        self.estimator = TrafficEstimator()

    def add_config(self, config: TrafficConfig) -> Tuple[bool, List[str]]:
        """
        添加配置

        :param config: 流量配置
        :return: (是否成功, 错误信息列表)
        """
        # 验证配置
        errors = self.validator.validate(config)
        if errors:
            return False, errors

        # 分配ID并添加
        config.config_id = self.next_id
        self.next_id += 1
        self.configs.append(config)

        return True, []

    def remove_config(self, config_id: int) -> bool:
        """
        删除配置

        :param config_id: 配置ID
        :return: 是否成功删除
        """
        original_length = len(self.configs)
        self.configs = [c for c in self.configs if c.config_id != config_id]
        return len(self.configs) < original_length

    def get_config(self, config_id: int) -> TrafficConfig:
        """
        获取配置

        :param config_id: 配置ID
        :return: 流量配置,不存在则返回None
        """
        for config in self.configs:
            if config.config_id == config_id:
                return config
        return None

    def get_all_configs(self) -> List[TrafficConfig]:
        """获取所有配置"""
        return copy.deepcopy(self.configs)

    def clear_all(self):
        """清空所有配置"""
        self.configs = []
        self.next_id = 1

    def estimate_traffic(self, end_time: int) -> Dict[str, any]:
        """
        预估当前所有配置的流量统计

        :param end_time: 结束时间 (ns)
        :return: 预估统计字典
        """
        return self.estimator.estimate_multiple_configs(self.configs, end_time)
