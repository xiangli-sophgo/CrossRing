"""
KCIN 配置基类

提供 v1/v2 共享的基础配置参数和方法。
版本特定参数由子类添加。
"""

import json
import yaml
import os
import numpy as np
import copy
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict
from fractions import Fraction
from math import lcm, gcd


class KCINConfigBase:
    """KCIN 配置基类

    包含 v1/v2 共享的：
    - 拓扑参数
    - 基础网络参数
    - 带宽/延迟配置
    - 资源配置
    - 保序配置
    - 通用工具方法
    """

    def __init__(self, config_file: Optional[str] = None):
        """初始化基础配置

        Args:
            config_file: 配置文件路径，为 None 时使用默认配置
        """
        if config_file is None:
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            config_file = str(project_root / "config" / "topologies" / "kcin_5x4.yaml")

        self._load_config_file(config_file)

    def _load_config_file(self, config_file: str) -> Dict[str, Any]:
        """加载配置文件

        Args:
            config_file: 配置文件路径

        Returns:
            配置字典

        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置文件格式错误
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            if config_file.endswith((".yaml", ".yml")):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)

        self._apply_config(config)
        return config

    def _apply_config(self, config: Dict[str, Any]):
        """应用配置到实例属性（子类可覆盖扩展）

        Args:
            config: 配置字典
        """
        # ==================== 拓扑参数 ====================
        self.TOPO_TYPE = config.get("TOPO_TYPE", "5x4")
        self.NUM_COL, self.NUM_ROW, self.NUM_NODE = self._parse_topo_type(self.TOPO_TYPE)

        # ==================== 基础网络参数 ====================
        self.FLIT_SIZE = config.get("FLIT_SIZE", 64)
        self.BURST = config.get("BURST", 4)
        self.NETWORK_FREQUENCY = config.get("NETWORK_FREQUENCY", 2)
        self.IP_FREQUENCY = config.get("IP_FREQUENCY", 1)

        # 计算时间缩放参数
        self._calculate_time_scale()

        # KCIN 版本
        self.KCIN_VERSION = config.get("KCIN_VERSION", "v1")

        # ==================== 带宽限制 ====================
        self.GDMA_BW_LIMIT = config.get("GDMA_BW_LIMIT", 128)
        self.SDMA_BW_LIMIT = config.get("SDMA_BW_LIMIT", 128)
        self.CDMA_BW_LIMIT = config.get("CDMA_BW_LIMIT", 128)
        self.DDR_BW_LIMIT = config.get("DDR_BW_LIMIT", 128)
        self.L2M_BW_LIMIT = config.get("L2M_BW_LIMIT", 128)

        # ==================== 延迟配置（原始 ns 值）====================
        self.DDR_R_LATENCY_original = config.get("DDR_R_LATENCY", 50)
        self.DDR_R_LATENCY_VAR_original = config.get("DDR_R_LATENCY_VAR", 0)
        self.DDR_W_LATENCY_original = config.get("DDR_W_LATENCY", 10)
        self.L2M_R_LATENCY_original = config.get("L2M_R_LATENCY", 10)
        self.L2M_W_LATENCY_original = config.get("L2M_W_LATENCY", 5)
        self.SN_TRACKER_RELEASE_LATENCY_original = config.get("SN_TRACKER_RELEASE_LATENCY", 0)
        self.SN_PROCESSING_LATENCY_original = config.get("SN_PROCESSING_LATENCY", 0)
        self.RN_PROCESSING_LATENCY_original = config.get("RN_PROCESSING_LATENCY", 0)

        # ==================== 资源配置 ====================
        self._process_resource_config(config)

        # ==================== 保序配置 ====================
        self.ORDERING_PRESERVATION_MODE = config.get("ORDERING_PRESERVATION_MODE", "none")
        self.ORDERING_GRANULARITY = config.get("ORDERING_GRANULARITY", "ip")
        self.IN_ORDER_EJECTION_PAIRS = config.get("IN_ORDER_EJECTION_PAIRS", [])
        self.IN_ORDER_PACKET_CATEGORIES = config.get("IN_ORDER_PACKET_CATEGORIES", [])
        self.ORDERING_ETAG_UPGRADE_MODE = config.get("ORDERING_ETAG_UPGRADE_MODE", "none")

        # ==================== 通道配置 ====================
        self.CHANNEL_SPEC = config.get("CHANNEL_SPEC", {})
        self.CH_NAME_LIST = []

        # ==================== 仲裁配置 ====================
        self.arbitration = config.get("arbitration", {})

        # ==================== 多通道配置 ====================
        self.NETWORK_CHANNEL_CONFIG = config.get("NETWORK_CHANNEL_CONFIG", {})
        self.CHANNEL_SELECT_STRATEGY = config.get("CHANNEL_SELECT_STRATEGY", "ip_type_id_based")
        self.CHANNEL_SELECT_PARAMS = config.get("CHANNEL_SELECT_PARAMS", {})

        # ==================== 方向控制 ====================
        self.TL_ALLOWED_SOURCE_NODES = config.get("TL_ALLOWED_SOURCE_NODES", None)
        self.TR_ALLOWED_SOURCE_NODES = config.get("TR_ALLOWED_SOURCE_NODES", None)
        self.TU_ALLOWED_SOURCE_NODES = config.get("TU_ALLOWED_SOURCE_NODES", None)
        self.TD_ALLOWED_SOURCE_NODES = config.get("TD_ALLOWED_SOURCE_NODES", None)

        # ==================== 反方向流控 ====================
        self.REVERSE_DIRECTION_ENABLED = config.get("REVERSE_DIRECTION_ENABLED", False)
        self.REVERSE_DIRECTION_THRESHOLD = config.get("REVERSE_DIRECTION_THRESHOLD", 0.8)

        # ==================== IP 接口 FIFO ====================
        self.IP_L2H_FIFO_DEPTH = config.get("IP_L2H_FIFO_DEPTH", 16)
        self.IP_H2L_H_FIFO_DEPTH = config.get("IP_H2L_H_FIFO_DEPTH", 16)
        self.IP_H2L_L_FIFO_DEPTH = config.get("IP_H2L_L_FIFO_DEPTH", 16)

        # 更新延迟（转换为 cycles）
        self.update_latency()

    def _parse_topo_type(self, topo_type: str):
        """解析拓扑类型字符串

        Args:
            topo_type: 拓扑类型，格式为 "AxB"

        Returns:
            tuple: (NUM_COL, NUM_ROW, NUM_NODE)

        Raises:
            ValueError: 格式错误
        """
        if not topo_type or "x" not in topo_type:
            raise ValueError(f"TOPO_TYPE 格式错误，应为 'AxB' 格式，当前值: {topo_type}")

        parts = topo_type.split("x")
        if len(parts) != 2:
            raise ValueError(f"TOPO_TYPE 格式错误，应为 'AxB' 格式，当前值: {topo_type}")

        try:
            rows = int(parts[0])
            cols = int(parts[1])
        except ValueError:
            raise ValueError(f"TOPO_TYPE 格式错误，应为数字格式 'AxB'，当前值: {topo_type}")

        if rows <= 0 or cols <= 0:
            raise ValueError(f"TOPO_TYPE 参数必须为正整数，当前值: {topo_type}")

        num_col = cols
        num_row = rows
        num_node = num_col * num_row

        return num_col, num_row, num_node

    def _process_resource_config(self, config: Dict[str, Any]):
        """处理资源配置（databuffer 和 tracker）

        Args:
            config: 配置字典
        """
        burst = config.get("BURST", 4)

        # RN 资源
        self.RN_RDB_SIZE = self._parse_buffer_size(config.get("RN_RDB_SIZE", 64))
        self.RN_WDB_SIZE = self._parse_buffer_size(config.get("RN_WDB_SIZE", 64))
        self.RN_R_TRACKER_OSTD = self._resolve_tracker_ostd(
            config.get("RN_R_TRACKER_OSTD", "auto"), self.RN_RDB_SIZE, burst
        )
        self.RN_W_TRACKER_OSTD = self._resolve_tracker_ostd(
            config.get("RN_W_TRACKER_OSTD", "auto"), self.RN_WDB_SIZE, burst
        )

        # SN DDR 资源
        self.SN_DDR_RDB_SIZE = self._parse_buffer_size(config.get("SN_DDR_RDB_SIZE", 64))
        self.SN_DDR_WDB_SIZE = self._parse_buffer_size(config.get("SN_DDR_WDB_SIZE", 64))
        self.SN_DDR_R_TRACKER_OSTD = self._resolve_tracker_ostd(
            config.get("SN_DDR_R_TRACKER_OSTD", "auto"), self.SN_DDR_RDB_SIZE, burst
        )
        self.SN_DDR_W_TRACKER_OSTD = self._resolve_tracker_ostd(
            config.get("SN_DDR_W_TRACKER_OSTD", "auto"), self.SN_DDR_WDB_SIZE, burst
        )

        # SN L2M 资源
        self.SN_L2M_RDB_SIZE = self._parse_buffer_size(config.get("SN_L2M_RDB_SIZE", 64))
        self.SN_L2M_WDB_SIZE = self._parse_buffer_size(config.get("SN_L2M_WDB_SIZE", 64))
        self.SN_L2M_R_TRACKER_OSTD = self._resolve_tracker_ostd(
            config.get("SN_L2M_R_TRACKER_OSTD", "auto"), self.SN_L2M_RDB_SIZE, burst
        )
        self.SN_L2M_W_TRACKER_OSTD = self._resolve_tracker_ostd(
            config.get("SN_L2M_W_TRACKER_OSTD", "auto"), self.SN_L2M_WDB_SIZE, burst
        )

        # 统一 R/W tracker 模式
        self.UNIFIED_RW_TRACKER = config.get("UNIFIED_RW_TRACKER", False)
        if self.UNIFIED_RW_TRACKER:
            self.RN_W_TRACKER_OSTD = self.RN_R_TRACKER_OSTD
            self.RN_WDB_SIZE = self.RN_RDB_SIZE
            self.SN_DDR_W_TRACKER_OSTD = self.SN_DDR_R_TRACKER_OSTD
            self.SN_DDR_WDB_SIZE = self.SN_DDR_RDB_SIZE
            self.SN_L2M_W_TRACKER_OSTD = self.SN_L2M_R_TRACKER_OSTD
            self.SN_L2M_WDB_SIZE = self.SN_L2M_RDB_SIZE

    def _parse_buffer_size(self, value):
        """解析缓冲区大小参数"""
        if isinstance(value, str) and value.lower() == "auto":
            return 64  # 默认值
        return int(value)

    def _resolve_tracker_ostd(self, tracker_value, databuffer_size, burst):
        """根据 databuffer 自动计算 tracker 数量"""
        if isinstance(tracker_value, str) and tracker_value.lower() == "auto":
            return databuffer_size // burst
        return int(tracker_value)

    def _calculate_time_scale(self):
        """根据网络频率和IP频率自动计算时间缩放参数

        计算 CYCLES_PER_NS、NETWORK_SCALE、IP_SCALE，使得：
        - 所有时间转换都使用整数运算
        - 网络域操作每 NETWORK_SCALE 个仿真周期执行一次
        - IP 域操作每 IP_SCALE 个仿真周期执行一次
        """
        max_denominator = 16  # 限制分母，支持17/8、33/16等精确分数

        # 转换为分数
        net_frac = Fraction(self.NETWORK_FREQUENCY).limit_denominator(max_denominator)
        ip_frac = Fraction(self.IP_FREQUENCY).limit_denominator(max_denominator)

        # 通分
        common_denom = lcm(net_frac.denominator, ip_frac.denominator)
        net_num = net_frac.numerator * (common_denom // net_frac.denominator)
        ip_num = ip_frac.numerator * (common_denom // ip_frac.denominator)

        # 计算最小 CYCLES_PER_NS
        net_divisor = net_num // gcd(net_num, common_denom)
        ip_divisor = ip_num // gcd(ip_num, common_denom)
        self.CYCLES_PER_NS = lcm(net_divisor, ip_divisor)

        # 计算各域的缩放因子
        self.NETWORK_SCALE = self.CYCLES_PER_NS * common_denom // net_num
        self.IP_SCALE = self.CYCLES_PER_NS * common_denom // ip_num

        # 计算实际生效的频率
        self.EFFECTIVE_NETWORK_FREQ = self.CYCLES_PER_NS / self.NETWORK_SCALE
        self.EFFECTIVE_IP_FREQ = self.CYCLES_PER_NS / self.IP_SCALE

        # 精度检查
        net_error = abs(self.EFFECTIVE_NETWORK_FREQ - self.NETWORK_FREQUENCY) / self.NETWORK_FREQUENCY
        ip_error = abs(self.EFFECTIVE_IP_FREQ - self.IP_FREQUENCY) / self.IP_FREQUENCY
        if net_error > 0.02:
            print(f"警告: 网络频率 {self.NETWORK_FREQUENCY}GHz 近似为 {self.EFFECTIVE_NETWORK_FREQ:.4f}GHz")
        if ip_error > 0.02:
            print(f"警告: IP频率 {self.IP_FREQUENCY}GHz 近似为 {self.EFFECTIVE_IP_FREQ:.4f}GHz")

    def update_latency(self):
        """将延迟配置从 ns 转换为 cycles"""
        self.DDR_R_LATENCY = int(self.DDR_R_LATENCY_original * self.CYCLES_PER_NS)
        self.DDR_R_LATENCY_VAR = int(self.DDR_R_LATENCY_VAR_original * self.CYCLES_PER_NS)
        self.DDR_W_LATENCY = int(self.DDR_W_LATENCY_original * self.CYCLES_PER_NS)
        self.L2M_R_LATENCY = int(self.L2M_R_LATENCY_original * self.CYCLES_PER_NS)
        self.L2M_W_LATENCY = int(self.L2M_W_LATENCY_original * self.CYCLES_PER_NS)
        self.SN_PROCESSING_LATENCY = int(self.SN_PROCESSING_LATENCY_original * self.CYCLES_PER_NS)
        self.RN_PROCESSING_LATENCY = int(self.RN_PROCESSING_LATENCY_original * self.CYCLES_PER_NS)
        self.SN_TRACKER_RELEASE_LATENCY = int(self.SN_TRACKER_RELEASE_LATENCY_original * self.CYCLES_PER_NS)

    def update_channel_list_from_ips(self, ip_types):
        """从 IP 类型列表更新 CH_NAME_LIST

        Args:
            ip_types: IP 类型列表，如 ["gdma_0", "gdma_1", "ddr_0"]
        """
        self.CH_NAME_LIST = sorted(list(ip_types))

    def infer_channel_spec_from_ips(self, ip_types):
        """从 IP 类型列表反向推断 CHANNEL_SPEC

        Args:
            ip_types: IP 类型列表

        Returns:
            推断的 CHANNEL_SPEC 字典
        """
        channel_counts = defaultdict(set)

        for ip_type in ip_types:
            if "_" in ip_type:
                parts = ip_type.rsplit("_", 1)
                base_type = parts[0]
                try:
                    idx = int(parts[1])
                    channel_counts[base_type].add(idx)
                except ValueError:
                    pass

        channel_spec = {}
        for base_type, indices in channel_counts.items():
            if indices:
                channel_spec[base_type] = max(indices) + 1

        self.CHANNEL_SPEC = channel_spec
        return channel_spec

    def generate_ip_positions(self, zero_rows=None, zero_cols=None):
        """生成 IP 位置列表

        Args:
            zero_rows: 排除的行
            zero_cols: 排除的列

        Returns:
            有效 IP 位置列表
        """
        matrix = [[1 for _ in range(self.NUM_COL)] for _ in range(self.NUM_ROW)]

        if zero_rows:
            for row in zero_rows:
                if 0 <= row < self.NUM_ROW:
                    for col in range(self.NUM_COL):
                        matrix[row][col] = 0

        if zero_cols:
            for col in zero_cols:
                if 0 <= col < self.NUM_COL:
                    for row in range(self.NUM_ROW):
                        matrix[row][col] = 0

        indices = []
        for r in range(self.NUM_ROW):
            for c in range(self.NUM_COL):
                if matrix[r][c] == 1:
                    index = r * self.NUM_COL + c
                    indices.append(index)
        return indices

    def distance(self, p1, p2):
        """计算曼哈顿距离"""
        return np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1])

    def update_config(self, topo_type="default"):
        """更新配置，转换延迟并选择拓扑"""
        self.update_latency()
        self.topology_select(topo_type)

    def topology_select(self, topo_type="default"):
        """选择拓扑类型（子类可重写）"""
        pass

    def _make_channels(self, key_types, value_factory=None):
        """创建通道字典

        Args:
            key_types: IP类型列表
            value_factory: 值工厂函数，为 None 时使用默认 deque 工厂

        Returns:
            通道字典，格式为 {ip_type_idx: value}
        """
        if value_factory is None:
            from collections import defaultdict, deque
            # 使用 RS_IN_CH_BUFFER 作为默认深度（兼容 v1/v2）
            depth = getattr(self, 'RS_IN_CH_BUFFER', getattr(self, 'IQ_CH_FIFO_DEPTH', 4))
            value_factory = lambda: defaultdict(lambda d=depth: deque(maxlen=d))

        if not callable(value_factory):
            static_value = copy.deepcopy(value_factory)
            value_factory = lambda v=static_value: copy.deepcopy(v)

        ports = {}
        for key in key_types:
            for idx in range(self.CHANNEL_SPEC.get(key, 0)):
                ports[f"{key}_{idx}"] = value_factory() if value_factory else None
        return ports
