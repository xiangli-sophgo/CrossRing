"""
Tier6+ 配置加载器
"""

import os
from typing import Dict, Optional, Any

import yaml


class Tier6Config:
    """Tier6+ 配置管理器"""

    # 默认配置
    DEFAULT_CONFIG = {
        # Die 层级
        "die": {
            "num_nodes": 20,
            "num_cols": 4,
            "noc_link_bw_gbps": 128.0,
            "hop_latency_ns": 0.5,
            "ddr_read_latency_ns": 50,
            "l2m_read_latency_ns": 20,
        },
        # Chip 层级 (D2D)
        "chip": {
            "num_dies": 2,
            "d2d_latency_ns": 20.0,
            "d2d_bandwidth_gbps": 192.0,
        },
        # Board 层级 (C2C)
        "board": {
            "num_chips": 2,
            "c2c_latency_ns": 100.0,
            "c2c_bandwidth_gbps": 64.0,
        },
        # Server 层级 (B2B)
        "server": {
            "num_boards": 2,
            "b2b_latency_ns": 500.0,
            "b2b_bandwidth_gbps": 32.0,
        },
        # Pod 层级 (S2S)
        "pod": {
            "num_servers": 4,
            "s2s_latency_ns": 2000.0,
            "s2s_bandwidth_gbps": 100.0,
            "topology": "full_mesh",
        },
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置

        Args:
            config_path: 配置文件路径 (YAML)
        """
        self.config: Dict[str, Any] = {}
        self._load_defaults()

        if config_path:
            self._load_from_file(config_path)

    def _load_defaults(self):
        """加载默认配置"""
        self.config = self._deep_copy(self.DEFAULT_CONFIG)

    def _deep_copy(self, d: Dict) -> Dict:
        """深拷贝字典"""
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = self._deep_copy(v)
            else:
                result[k] = v
        return result

    def _load_from_file(self, config_path: str):
        """从文件加载配置"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            file_config = yaml.safe_load(f)

        if file_config:
            self._merge_config(file_config)

    def _merge_config(self, new_config: Dict):
        """合并配置（新配置覆盖默认）"""
        for key, value in new_config.items():
            if key in self.config and isinstance(value, dict):
                self.config[key].update(value)
            else:
                self.config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        d = self.config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    def get_level_config(self, level: str) -> Dict:
        """获取指定层级的完整配置"""
        if level not in self.config:
            return {}

        # 构建嵌套配置
        level_config = self._deep_copy(self.config[level])

        # 添加子层级配置
        level_order = ["die", "chip", "board", "server", "pod"]
        level_idx = level_order.index(level) if level in level_order else -1

        if level_idx > 0:
            # 递归添加子层级配置
            child_level = level_order[level_idx - 1]
            child_config_key = f"{child_level}_config"
            level_config[child_config_key] = self.get_level_config(child_level)

        return level_config

    def to_dict(self) -> Dict:
        """导出为字典"""
        return self._deep_copy(self.config)

    def save(self, path: str):
        """保存配置到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

    def __repr__(self) -> str:
        return f"Tier6Config({self.config})"


def load_config(config_path: Optional[str] = None) -> Dict:
    """
    便捷函数：加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    config = Tier6Config(config_path)
    return config.to_dict()
