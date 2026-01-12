"""
StatisticalFIFO - 带自动统计功能的 FIFO

统一 v1 和 v2 的 FIFO 统计收集，自动注册、自动命名、自动记录深度统计。
"""

from collections import deque
from typing import Dict, List, Optional, Any, Iterator


class StatisticalFIFO:
    """带自动统计功能的 FIFO"""

    # 类级别注册表 - 收集所有实例
    _registry: Dict[str, 'StatisticalFIFO'] = {}

    def __init__(self, name: str, maxlen: int, node_pos: int,
                 category: str, fifo_type: str, ip_type: str = None):
        """
        初始化 StatisticalFIFO

        Args:
            name: 唯一名称，如 "RS(3,2)_IN_ddr_0", "IQ(1,2)_CH_gdma_0"
            maxlen: 最大深度
            node_pos: 节点位置
            category: 分类 (v1: IQ/RB/EQ, v2: RS_IN/RS_OUT)
            fifo_type: 类型 (CH, TL, TR, TU, TD)
            ip_type: IP类型 (ddr_0, gdma_0 等，仅 CH 类型有)
        """
        self._fifo = deque(maxlen=maxlen)
        self.name = name
        self.maxlen = maxlen
        self.node_pos = node_pos
        self.category = category
        self.fifo_type = fifo_type
        self.ip_type = ip_type

        # 统计数据
        self.depth_sum = 0
        self.max_depth = 0
        self.flit_count = 0
        self.sample_count = 0

        # 注册到全局注册表
        StatisticalFIFO._registry[name] = self

    # ========== deque 接口方法 ==========

    def append(self, item):
        """添加元素到队尾"""
        self._fifo.append(item)
        self.flit_count += 1

    def appendleft(self, item):
        """添加元素到队首"""
        self._fifo.appendleft(item)
        self.flit_count += 1

    def popleft(self):
        """从队首弹出元素"""
        return self._fifo.popleft()

    def pop(self):
        """从队尾弹出元素"""
        return self._fifo.pop()

    def clear(self):
        """清空队列"""
        self._fifo.clear()

    def __len__(self) -> int:
        return len(self._fifo)

    def __iter__(self) -> Iterator:
        return iter(self._fifo)

    def __getitem__(self, index):
        return self._fifo[index]

    def __setitem__(self, index, value):
        self._fifo[index] = value

    def __bool__(self) -> bool:
        return len(self._fifo) > 0

    def __contains__(self, item) -> bool:
        return item in self._fifo

    # ========== 统计方法 ==========

    def sample(self):
        """每周期调用，记录当前深度"""
        depth = len(self._fifo)
        self.depth_sum += depth
        self.max_depth = max(self.max_depth, depth)
        self.sample_count += 1

    def get_utilization(self) -> float:
        """获取平均利用率"""
        if self.sample_count == 0 or self.maxlen == 0:
            return 0.0
        avg_depth = self.depth_sum / self.sample_count
        return avg_depth / self.maxlen

    def get_avg_depth(self) -> float:
        """获取平均深度"""
        if self.sample_count == 0:
            return 0.0
        return self.depth_sum / self.sample_count

    def get_stats(self) -> dict:
        """获取完整统计数据"""
        return {
            "name": self.name,
            "node_pos": self.node_pos,
            "category": self.category,
            "fifo_type": self.fifo_type,
            "ip_type": self.ip_type,
            "maxlen": self.maxlen,
            "depth_sum": self.depth_sum,
            "max_depth": self.max_depth,
            "flit_count": self.flit_count,
            "sample_count": self.sample_count,
            "avg_depth": self.get_avg_depth(),
            "utilization": self.get_utilization(),
        }

    def reset_stats(self):
        """重置统计数据"""
        self.depth_sum = 0
        self.max_depth = 0
        self.flit_count = 0
        self.sample_count = 0

    # ========== 类方法 ==========

    @classmethod
    def get_all_fifos(cls) -> Dict[str, 'StatisticalFIFO']:
        """获取所有注册的 FIFO 实例"""
        return cls._registry

    @classmethod
    def get_fifos_by_category(cls, category: str) -> Dict[str, 'StatisticalFIFO']:
        """按分类获取 FIFO 实例"""
        return {name: fifo for name, fifo in cls._registry.items()
                if fifo.category == category}

    @classmethod
    def get_fifos_by_node(cls, node_pos: int) -> Dict[str, 'StatisticalFIFO']:
        """按节点位置获取 FIFO 实例"""
        return {name: fifo for name, fifo in cls._registry.items()
                if fifo.node_pos == node_pos}

    @classmethod
    def clear_registry(cls):
        """清空注册表（新仿真开始时调用）"""
        cls._registry.clear()

    @classmethod
    def sample_all(cls):
        """采样所有注册的 FIFO"""
        for fifo in cls._registry.values():
            fifo.sample()

    @classmethod
    def reset_all_stats(cls):
        """重置所有 FIFO 的统计数据"""
        for fifo in cls._registry.values():
            fifo.reset_stats()

    @classmethod
    def get_stats_summary(cls) -> Dict[str, Dict]:
        """获取所有 FIFO 统计摘要，按 category 和 fifo_type 分组"""
        summary = {}
        for fifo in cls._registry.values():
            cat = fifo.category
            ft = fifo.fifo_type
            if cat not in summary:
                summary[cat] = {}
            if ft not in summary[cat]:
                summary[cat][ft] = {}

            if fifo.ip_type:
                # CH 类型，按 node_pos -> ip_type 存储
                if fifo.node_pos not in summary[cat][ft]:
                    summary[cat][ft][fifo.node_pos] = {}
                summary[cat][ft][fifo.node_pos][fifo.ip_type] = fifo.get_utilization()
            else:
                # 方向类型，按 node_pos 存储
                summary[cat][ft][fifo.node_pos] = fifo.get_utilization()

        return summary
