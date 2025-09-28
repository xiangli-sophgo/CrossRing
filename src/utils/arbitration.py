"""
NoC端口仲裁模块

提供多种端口仲裁算法的实现，用于解决多个数据流同时竞争网络资源的冲突问题。
包含轮询仲裁、优先级仲裁、加权公平仲裁等多种策略。

主要仲裁算法：
- RoundRobinArbiter: 标准轮询仲裁
- ConditionalRoundRobinArbiter: 条件轮询仲裁
- WeightedRoundRobinArbiter: 加权轮询仲裁
- FixedPriorityArbiter: 固定优先级仲裁
- DynamicPriorityArbiter: 动态优先级仲裁
- RandomArbiter: 随机仲裁
- TokenBucketArbiter: 令牌桶仲裁
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from collections import defaultdict
import random
import time


class BaseArbiter(ABC):
    """
    仲裁器抽象基类

    定义了所有仲裁器必须实现的基本接口，包括仲裁决策、状态重置和统计信息获取。
    """

    def __init__(self, num_ports: int):
        """
        初始化仲裁器

        参数:
            num_ports: 端口数量
        """
        if num_ports <= 0:
            raise ValueError("端口数量必须大于0")

        self.num_ports = num_ports
        self.stats = {
            'total_arbitrations': 0,
            'successful_arbitrations': 0,
            'grants_per_port': [0] * num_ports,
            'starvation_counter': [0] * num_ports
        }

    @abstractmethod
    def arbitrate(self, requests: List[bool]) -> Optional[int]:
        """
        执行仲裁决策

        参数:
            requests: 布尔列表，表示各端口的请求状态

        返回:
            获得授权的端口索引，如果无有效请求则返回None
        """
        pass

    def reset(self):
        """重置仲裁器状态"""
        self.stats = {
            'total_arbitrations': 0,
            'successful_arbitrations': 0,
            'grants_per_port': [0] * self.num_ports,
            'starvation_counter': [0] * self.num_ports
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        获取仲裁统计信息

        返回:
            包含统计信息的字典
        """
        total = self.stats['total_arbitrations']
        successful = self.stats['successful_arbitrations']

        # 计算成功率
        success_rate = successful / total if total > 0 else 0.0

        # 计算公平性指标（方差越小越公平）
        grants = self.stats['grants_per_port']
        if successful > 0:
            avg_grants = sum(grants) / len(grants)
            variance = sum((g - avg_grants) ** 2 for g in grants) / len(grants)
            fairness_index = 1.0 / (1.0 + variance) if variance > 0 else 1.0
        else:
            fairness_index = 1.0

        return {
            'total_arbitrations': total,
            'successful_arbitrations': successful,
            'success_rate': success_rate,
            'grants_per_port': grants.copy(),
            'fairness_index': fairness_index,
            'max_starvation': max(self.stats['starvation_counter']),
            'starving_ports': [i for i, count in enumerate(self.stats['starvation_counter']) if count > 10]
        }

    def _update_stats(self, requests: List[bool], granted_port: Optional[int]):
        """
        更新统计信息

        参数:
            requests: 请求状态列表
            granted_port: 获得授权的端口，None表示无授权
        """
        self.stats['total_arbitrations'] += 1

        if granted_port is not None:
            self.stats['successful_arbitrations'] += 1
            self.stats['grants_per_port'][granted_port] += 1
            # 重置获得服务端口的饥饿计数器
            self.stats['starvation_counter'][granted_port] = 0

        # 更新未获得服务端口的饥饿计数器
        for i, has_request in enumerate(requests):
            if has_request and i != granted_port:
                self.stats['starvation_counter'][i] += 1


class RoundRobinArbiter(BaseArbiter):
    """
    标准轮询仲裁器

    按照固定的循环顺序依次给予各个端口访问权限，确保所有端口都有公平的机会。
    特点：
    - 绝对公平性：长期来看每个端口获得相同的服务机会
    - 饥饿免疫：任何端口都不会被永久阻塞
    - 实现简单：硬件开销小，逻辑简单
    - 可预测性：服务间隔固定，便于性能分析
    """

    def __init__(self, num_ports: int):
        """
        初始化轮询仲裁器

        参数:
            num_ports: 端口数量
        """
        super().__init__(num_ports)
        self.current_pointer = 0

    def arbitrate(self, requests: List[bool]) -> Optional[int]:
        """
        执行轮询仲裁

        从当前指针位置开始，循环查找第一个有效请求

        参数:
            requests: 各端口的请求状态

        返回:
            获得授权的端口索引
        """
        if len(requests) != self.num_ports:
            raise ValueError(f"请求列表长度({len(requests)})与端口数量({self.num_ports})不匹配")

        if not any(requests):
            self._update_stats(requests, None)
            return None

        # 从当前指针位置开始查找第一个有效请求
        for i in range(self.num_ports):
            current_index = (self.current_pointer + i) % self.num_ports
            if requests[current_index]:
                # 找到有效请求，更新指针到下一个位置
                self.current_pointer = (current_index + 1) % self.num_ports
                self._update_stats(requests, current_index)
                return current_index

        # 理论上不会到达这里（前面已经检查了any(requests)）
        self._update_stats(requests, None)
        return None

    def reset(self):
        """重置仲裁器状态"""
        super().reset()
        self.current_pointer = 0

    def get_current_pointer(self) -> int:
        """获取当前指针位置"""
        return self.current_pointer

    def set_pointer(self, position: int):
        """
        设置指针位置

        参数:
            position: 新的指针位置
        """
        self.current_pointer = position % self.num_ports


class ConditionalRoundRobinArbiter(BaseArbiter):
    """
    条件轮询仲裁器

    只有在找到有效请求并授权后指针才前进。
    在低负载情况下可能出现不公平现象，但能够提供更好的资源利用率。
    """

    def __init__(self, num_ports: int):
        """
        初始化条件轮询仲裁器

        参数:
            num_ports: 端口数量
        """
        super().__init__(num_ports)
        self.current_pointer = 0

    def arbitrate(self, requests: List[bool]) -> Optional[int]:
        """
        执行条件轮询仲裁

        只有在成功授权时才移动指针

        参数:
            requests: 各端口的请求状态

        返回:
            获得授权的端口索引
        """
        if len(requests) != self.num_ports:
            raise ValueError(f"请求列表长度({len(requests)})与端口数量({self.num_ports})不匹配")

        if not any(requests):
            self._update_stats(requests, None)
            return None

        # 从当前指针位置开始查找第一个有效请求
        for i in range(self.num_ports):
            current_index = (self.current_pointer + i) % self.num_ports
            if requests[current_index]:
                # 找到有效请求，授权并移动指针
                self.current_pointer = (current_index + 1) % self.num_ports
                self._update_stats(requests, current_index)
                return current_index

        # 理论上不会到达这里
        self._update_stats(requests, None)
        return None

    def reset(self):
        """重置仲裁器状态"""
        super().reset()
        self.current_pointer = 0


class WeightedRoundRobinArbiter(BaseArbiter):
    """
    加权轮询仲裁器

    为不同端口分配不同的权重，高权重端口在一轮中可能获得多次服务机会。
    实现了在公平性基础上的差异化服务。
    """

    def __init__(self, num_ports: int, weights: Optional[List[int]] = None):
        """
        初始化加权轮询仲裁器

        参数:
            num_ports: 端口数量
            weights: 各端口的权重，如果为None则使用等权重
        """
        super().__init__(num_ports)

        if weights is None:
            self.weights = [1] * num_ports
        else:
            if len(weights) != num_ports:
                raise ValueError(f"权重列表长度({len(weights)})与端口数量({num_ports})不匹配")
            if any(w <= 0 for w in weights):
                raise ValueError("所有权重必须大于0")
            self.weights = weights.copy()

        # 初始化信用计数器
        self.credits = self.weights.copy()
        self.current_pointer = 0

    def arbitrate(self, requests: List[bool]) -> Optional[int]:
        """
        执行加权轮询仲裁

        使用信用机制实现权重分配

        参数:
            requests: 各端口的请求状态

        返回:
            获得授权的端口索引
        """
        if len(requests) != self.num_ports:
            raise ValueError(f"请求列表长度({len(requests)})与端口数量({self.num_ports})不匹配")

        if not any(requests):
            self._update_stats(requests, None)
            return None

        # 如果当前端口有请求且有剩余信用
        if requests[self.current_pointer] and self.credits[self.current_pointer] > 0:
            self.credits[self.current_pointer] -= 1
            self._update_stats(requests, self.current_pointer)
            return self.current_pointer

        # 寻找下一个有请求且有信用的端口
        for i in range(1, self.num_ports):
            next_index = (self.current_pointer + i) % self.num_ports
            if requests[next_index] and self.credits[next_index] > 0:
                self.credits[next_index] -= 1
                self.current_pointer = next_index
                self._update_stats(requests, next_index)
                return next_index

        # 如果没有找到，说明所有端口的信用都用完了，重新分配信用
        self._refill_credits()

        # 重新分配信用后再次尝试
        for i in range(self.num_ports):
            current_index = (self.current_pointer + i) % self.num_ports
            if requests[current_index] and self.credits[current_index] > 0:
                self.credits[current_index] -= 1
                self.current_pointer = (current_index + 1) % self.num_ports
                self._update_stats(requests, current_index)
                return current_index

        # 理论上不会到达这里
        self._update_stats(requests, None)
        return None

    def _refill_credits(self):
        """重新分配所有端口的信用"""
        self.credits = self.weights.copy()

    def set_weights(self, weights: List[int]):
        """
        设置新的权重

        参数:
            weights: 新的权重列表
        """
        if len(weights) != self.num_ports:
            raise ValueError(f"权重列表长度({len(weights)})与端口数量({self.num_ports})不匹配")
        if any(w <= 0 for w in weights):
            raise ValueError("所有权重必须大于0")

        self.weights = weights.copy()
        self.credits = self.weights.copy()

    def get_weights(self) -> List[int]:
        """获取当前权重配置"""
        return self.weights.copy()

    def get_credits(self) -> List[int]:
        """获取当前信用状态"""
        return self.credits.copy()

    def reset(self):
        """重置仲裁器状态"""
        super().reset()
        self.credits = self.weights.copy()
        self.current_pointer = 0


class FixedPriorityArbiter(BaseArbiter):
    """
    固定优先级仲裁器

    按照预定的固定优先级顺序进行仲裁，优先级高的端口总是优先获得服务。
    特点：
    - 简单实现：优先级顺序固定，逻辑简单
    - 确定性：相同输入产生相同输出
    - 可能饥饿：低优先级端口可能被永久阻塞
    - 适用场景：有明确优先级需求的系统
    """

    def __init__(self, num_ports: int, priorities: Optional[List[int]] = None):
        """
        初始化固定优先级仲裁器

        参数:
            num_ports: 端口数量
            priorities: 各端口的优先级，数值越小优先级越高，如果为None则使用端口索引作为优先级
        """
        super().__init__(num_ports)

        if priorities is None:
            # 默认优先级：端口0优先级最高
            self.priorities = list(range(num_ports))
        else:
            if len(priorities) != num_ports:
                raise ValueError(f"优先级列表长度({len(priorities)})与端口数量({num_ports})不匹配")
            self.priorities = priorities.copy()

        # 创建按优先级排序的端口索引列表
        self.priority_order = sorted(range(num_ports), key=lambda i: self.priorities[i])

    def arbitrate(self, requests: List[bool]) -> Optional[int]:
        """
        执行固定优先级仲裁

        按照优先级顺序查找第一个有效请求

        参数:
            requests: 各端口的请求状态

        返回:
            获得授权的端口索引
        """
        if len(requests) != self.num_ports:
            raise ValueError(f"请求列表长度({len(requests)})与端口数量({self.num_ports})不匹配")

        if not any(requests):
            self._update_stats(requests, None)
            return None

        # 按优先级顺序查找第一个有效请求
        for port_index in self.priority_order:
            if requests[port_index]:
                self._update_stats(requests, port_index)
                return port_index

        # 理论上不会到达这里
        self._update_stats(requests, None)
        return None

    def set_priorities(self, priorities: List[int]):
        """
        设置新的优先级

        参数:
            priorities: 新的优先级列表
        """
        if len(priorities) != self.num_ports:
            raise ValueError(f"优先级列表长度({len(priorities)})与端口数量({self.num_ports})不匹配")

        self.priorities = priorities.copy()
        self.priority_order = sorted(range(self.num_ports), key=lambda i: self.priorities[i])

    def get_priorities(self) -> List[int]:
        """获取当前优先级配置"""
        return self.priorities.copy()


class DynamicPriorityArbiter(BaseArbiter):
    """
    动态优先级仲裁器

    根据饥饿时间动态调整端口优先级，长时间未获得服务的端口优先级会逐渐提升。
    特点：
    - 饥饿预防：避免任何端口被永久阻塞
    - 动态调整：优先级根据历史服务情况变化
    - 公平性：长期来看提供相对公平的服务
    - 复杂度适中：需要维护动态优先级状态
    """

    def __init__(self, num_ports: int, base_priorities: Optional[List[int]] = None, aging_factor: float = 1.0):
        """
        初始化动态优先级仲裁器

        参数:
            num_ports: 端口数量
            base_priorities: 基础优先级，如果为None则使用相等优先级
            aging_factor: 老化因子，控制优先级提升速度
        """
        super().__init__(num_ports)

        if base_priorities is None:
            self.base_priorities = [0] * num_ports
        else:
            if len(base_priorities) != num_ports:
                raise ValueError(f"基础优先级列表长度({len(base_priorities)})与端口数量({num_ports})不匹配")
            self.base_priorities = base_priorities.copy()

        self.aging_factor = aging_factor
        # 动态优先级 = 基础优先级 - 饥饿时间 * 老化因子
        self.dynamic_priorities = self.base_priorities.copy()

    def arbitrate(self, requests: List[bool]) -> Optional[int]:
        """
        执行动态优先级仲裁

        根据当前动态优先级和请求状态进行仲裁

        参数:
            requests: 各端口的请求状态

        返回:
            获得授权的端口索引
        """
        if len(requests) != self.num_ports:
            raise ValueError(f"请求列表长度({len(requests)})与端口数量({self.num_ports})不匹配")

        if not any(requests):
            self._update_stats(requests, None)
            return None

        # 更新动态优先级（饥饿时间越长，优先级越高）
        for i in range(self.num_ports):
            self.dynamic_priorities[i] = self.base_priorities[i] - self.stats['starvation_counter'][i] * self.aging_factor

        # 在有请求的端口中找到优先级最高的（数值最小的）
        requesting_ports = [i for i in range(self.num_ports) if requests[i]]
        if not requesting_ports:
            self._update_stats(requests, None)
            return None

        # 选择优先级最高的端口
        granted_port = min(requesting_ports, key=lambda i: self.dynamic_priorities[i])
        self._update_stats(requests, granted_port)
        return granted_port

    def reset(self):
        """重置仲裁器状态"""
        super().reset()
        self.dynamic_priorities = self.base_priorities.copy()

    def get_dynamic_priorities(self) -> List[float]:
        """获取当前动态优先级"""
        return self.dynamic_priorities.copy()

    def set_aging_factor(self, aging_factor: float):
        """设置老化因子"""
        self.aging_factor = aging_factor


class RandomArbiter(BaseArbiter):
    """
    随机仲裁器

    从有请求的端口中随机选择一个进行授权。
    特点：
    - 简单实现：逻辑非常简单
    - 无偏向性：不偏向任何特定端口
    - 不可预测：服务顺序随机
    - 可能不公平：短期内可能出现不公平现象
    - 适用场景：对公平性要求不高但需要避免确定性偏向的系统
    """

    def __init__(self, num_ports: int, seed: Optional[int] = None):
        """
        初始化随机仲裁器

        参数:
            num_ports: 端口数量
            seed: 随机种子，用于确保测试的可重现性
        """
        super().__init__(num_ports)

        if seed is not None:
            random.seed(seed)

    def arbitrate(self, requests: List[bool]) -> Optional[int]:
        """
        执行随机仲裁

        从有请求的端口中随机选择一个

        参数:
            requests: 各端口的请求状态

        返回:
            获得授权的端口索引
        """
        if len(requests) != self.num_ports:
            raise ValueError(f"请求列表长度({len(requests)})与端口数量({self.num_ports})不匹配")

        if not any(requests):
            self._update_stats(requests, None)
            return None

        # 收集所有有请求的端口
        requesting_ports = [i for i in range(self.num_ports) if requests[i]]

        if not requesting_ports:
            self._update_stats(requests, None)
            return None

        # 随机选择一个端口
        granted_port = random.choice(requesting_ports)
        self._update_stats(requests, granted_port)
        return granted_port

    def set_seed(self, seed: int):
        """设置随机种子"""
        random.seed(seed)


class TokenBucketArbiter(BaseArbiter):
    """
    令牌桶仲裁器

    为每个端口维护一个令牌桶，端口获得服务需要消耗令牌，令牌以固定速率补充。
    特点：
    - 流量控制：可以控制每个端口的服务速率
    - 突发支持：允许短期突发访问
    - 长期公平：长期来看各端口获得相同的服务机会
    - 实现复杂：需要维护令牌桶状态
    """

    def __init__(self, num_ports: int, bucket_size: int = 10, refill_rate: float = 1.0,
                 port_rates: Optional[List[float]] = None):
        """
        初始化令牌桶仲裁器

        参数:
            num_ports: 端口数量
            bucket_size: 每个端口的令牌桶大小
            refill_rate: 默认令牌补充速率（令牌/周期）
            port_rates: 各端口的令牌补充速率，如果为None则使用默认速率
        """
        super().__init__(num_ports)

        self.bucket_size = bucket_size
        self.refill_rate = refill_rate

        if port_rates is None:
            self.port_rates = [refill_rate] * num_ports
        else:
            if len(port_rates) != num_ports:
                raise ValueError(f"端口速率列表长度({len(port_rates)})与端口数量({num_ports})不匹配")
            self.port_rates = port_rates.copy()

        # 初始化令牌桶（开始时桶是满的）
        self.token_buckets = [bucket_size] * num_ports
        self.last_refill_time = time.time()

    def arbitrate(self, requests: List[bool]) -> Optional[int]:
        """
        执行令牌桶仲裁

        从有请求且有令牌的端口中选择一个（使用轮询策略）

        参数:
            requests: 各端口的请求状态

        返回:
            获得授权的端口索引
        """
        if len(requests) != self.num_ports:
            raise ValueError(f"请求列表长度({len(requests)})与端口数量({self.num_ports})不匹配")

        # 补充令牌
        self._refill_tokens()

        if not any(requests):
            self._update_stats(requests, None)
            return None

        # 找到有请求且有令牌的端口
        eligible_ports = [i for i in range(self.num_ports)
                         if requests[i] and self.token_buckets[i] > 0]

        if not eligible_ports:
            self._update_stats(requests, None)
            return None

        # 使用轮询策略选择端口（也可以使用其他策略）
        granted_port = min(eligible_ports)

        # 消耗令牌
        self.token_buckets[granted_port] -= 1

        self._update_stats(requests, granted_port)
        return granted_port

    def _refill_tokens(self):
        """补充令牌"""
        current_time = time.time()
        time_elapsed = current_time - self.last_refill_time

        for i in range(self.num_ports):
            # 计算应该补充的令牌数
            tokens_to_add = self.port_rates[i] * time_elapsed
            self.token_buckets[i] = min(self.bucket_size,
                                       self.token_buckets[i] + tokens_to_add)

        self.last_refill_time = current_time

    def reset(self):
        """重置仲裁器状态"""
        super().reset()
        self.token_buckets = [self.bucket_size] * self.num_ports
        self.last_refill_time = time.time()

    def get_token_status(self) -> List[float]:
        """获取各端口的令牌数量"""
        return self.token_buckets.copy()

    def set_bucket_size(self, bucket_size: int):
        """设置令牌桶大小"""
        self.bucket_size = bucket_size

    def set_port_rates(self, port_rates: List[float]):
        """设置各端口的令牌补充速率"""
        if len(port_rates) != self.num_ports:
            raise ValueError(f"端口速率列表长度({len(port_rates)})与端口数量({self.num_ports})不匹配")
        self.port_rates = port_rates.copy()


# 工厂函数，便于创建仲裁器实例
def create_round_robin_arbiter(num_ports: int) -> RoundRobinArbiter:
    """创建标准轮询仲裁器"""
    return RoundRobinArbiter(num_ports)


def create_conditional_round_robin_arbiter(num_ports: int) -> ConditionalRoundRobinArbiter:
    """创建条件轮询仲裁器"""
    return ConditionalRoundRobinArbiter(num_ports)


def create_weighted_round_robin_arbiter(num_ports: int, weights: Optional[List[int]] = None) -> WeightedRoundRobinArbiter:
    """创建加权轮询仲裁器"""
    return WeightedRoundRobinArbiter(num_ports, weights)


def create_fixed_priority_arbiter(num_ports: int, priorities: Optional[List[int]] = None) -> FixedPriorityArbiter:
    """创建固定优先级仲裁器"""
    return FixedPriorityArbiter(num_ports, priorities)


def create_dynamic_priority_arbiter(num_ports: int, base_priorities: Optional[List[int]] = None,
                                   aging_factor: float = 1.0) -> DynamicPriorityArbiter:
    """创建动态优先级仲裁器"""
    return DynamicPriorityArbiter(num_ports, base_priorities, aging_factor)


def create_random_arbiter(num_ports: int, seed: Optional[int] = None) -> RandomArbiter:
    """创建随机仲裁器"""
    return RandomArbiter(num_ports, seed)


def create_token_bucket_arbiter(num_ports: int, bucket_size: int = 10, refill_rate: float = 1.0,
                               port_rates: Optional[List[float]] = None) -> TokenBucketArbiter:
    """创建令牌桶仲裁器"""
    return TokenBucketArbiter(num_ports, bucket_size, refill_rate, port_rates)