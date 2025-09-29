"""
CrossRing NoC 统一仲裁系统

提供多种仲裁算法的统一实现，直接操作候选对象，支持多队列管理。
设计理念：仲裁器应该直接从候选对象中选择，而不是操作布尔列表。

主要仲裁算法：
- RoundRobinArbiter: 标准轮询仲裁
- WeightedArbiter: 加权轮询仲裁
- PriorityArbiter: 固定优先级仲裁
- DynamicPriorityArbiter: 动态优先级仲裁
- RandomArbiter: 随机仲裁
- TokenBucketArbiter: 令牌桶仲裁
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Callable, Tuple
from collections import defaultdict
import random
import time


class Arbiter(ABC):
    """
    统一仲裁器基类

    所有仲裁器的核心理念：直接从候选对象中选择，支持多队列独立状态管理。
    """

    def __init__(self):
        """初始化仲裁器"""
        # 每个队列的独立状态
        self.queue_states = {}

        # 统计信息
        self.stats = defaultdict(lambda: {
            'total_arbitrations': 0,
            'successful_arbitrations': 0,
            'grants_per_slot': defaultdict(int),
            'starvation_counter': defaultdict(int)
        })

    def select(self, candidates: List[Any], queue_id: str = "default",
               is_valid: Optional[Callable[[Any], bool]] = None) -> Tuple[Optional[Any], int]:
        """
        统一接口：从候选对象中选择一个

        参数:
            candidates: 候选对象列表（任意类型）
            queue_id: 队列标识符，支持多队列独立状态管理
            is_valid: 可选的验证函数，判断候选对象是否有效

        返回:
            (选中的对象, 索引) 或 (None, -1) 如果无有效候选
        """
        if not candidates:
            return None, -1

        # 初始化或更新队列状态
        if queue_id not in self.queue_states or len(candidates) != self.queue_states[queue_id].get('size', 0):
            self.queue_states[queue_id] = self._init_queue_state(len(candidates))

        # 生成有效性掩码
        valid_mask = []
        for candidate in candidates:
            if is_valid:
                valid_mask.append(is_valid(candidate))
            else:
                # 默认：非None即有效
                valid_mask.append(candidate is not None)

        # 执行具体仲裁算法
        selected_idx = self._do_arbitrate(queue_id, valid_mask)

        # 更新统计信息
        self._update_stats(queue_id, selected_idx, valid_mask)

        if selected_idx >= 0:
            return candidates[selected_idx], selected_idx
        return None, -1

    @abstractmethod
    def _init_queue_state(self, size: int) -> Dict[str, Any]:
        """
        初始化队列状态

        参数:
            size: 候选对象数量

        返回:
            队列状态字典
        """
        pass

    @abstractmethod
    def _do_arbitrate(self, queue_id: str, valid_mask: List[bool]) -> int:
        """
        执行具体的仲裁算法

        参数:
            queue_id: 队列标识符
            valid_mask: 有效性掩码

        返回:
            选中的索引，-1表示无有效候选
        """
        pass

    def _update_stats(self, queue_id: str, selected_idx: int, valid_mask: List[bool]):
        """更新统计信息"""
        stats = self.stats[queue_id]
        stats['total_arbitrations'] += 1

        if selected_idx >= 0:
            stats['successful_arbitrations'] += 1
            stats['grants_per_slot'][selected_idx] += 1
            # 重置获得服务槽位的饥饿计数器
            stats['starvation_counter'][selected_idx] = 0

        # 更新未获得服务槽位的饥饿计数器
        for i, is_valid in enumerate(valid_mask):
            if is_valid and i != selected_idx:
                stats['starvation_counter'][i] += 1

    def get_stats(self, queue_id: str = None) -> Dict[str, Any]:
        """
        获取统计信息

        参数:
            queue_id: 指定队列，None表示获取所有队列统计

        返回:
            统计信息字典
        """
        if queue_id:
            if queue_id not in self.stats:
                return {}

            stats = self.stats[queue_id]
            total = stats['total_arbitrations']
            successful = stats['successful_arbitrations']

            # 计算成功率
            success_rate = successful / total if total > 0 else 0.0

            # 计算公平性指标
            grants = dict(stats['grants_per_slot'])
            if successful > 0 and grants:
                grant_values = list(grants.values())
                avg_grants = sum(grant_values) / len(grant_values)
                variance = sum((g - avg_grants) ** 2 for g in grant_values) / len(grant_values)
                fairness_index = 1.0 / (1.0 + variance) if variance > 0 else 1.0
            else:
                fairness_index = 1.0

            return {
                'queue_id': queue_id,
                'total_arbitrations': total,
                'successful_arbitrations': successful,
                'success_rate': success_rate,
                'grants_per_slot': grants,
                'fairness_index': fairness_index,
                'max_starvation': max(stats['starvation_counter'].values()) if stats['starvation_counter'] else 0,
                'starving_slots': [i for i, count in stats['starvation_counter'].items() if count > 10]
            }
        else:
            # 返回所有队列的汇总统计
            all_stats = {}
            for qid in self.stats:
                all_stats[qid] = self.get_stats(qid)
            return all_stats

    def reset(self, queue_id: str = None):
        """
        重置仲裁器状态

        参数:
            queue_id: 指定队列，None表示重置所有队列
        """
        if queue_id:
            if queue_id in self.queue_states:
                size = self.queue_states[queue_id].get('size', 0)
                self.queue_states[queue_id] = self._init_queue_state(size)
            if queue_id in self.stats:
                del self.stats[queue_id]
        else:
            # 重置所有队列
            for qid in list(self.queue_states.keys()):
                size = self.queue_states[qid].get('size', 0)
                self.queue_states[qid] = self._init_queue_state(size)
            self.stats.clear()


class RoundRobinArbiter(Arbiter):
    """
    标准轮询仲裁器

    按照固定的循环顺序依次给予各个候选访问权限，确保所有候选都有公平的机会。
    特点：
    - 绝对公平性：长期来看每个候选获得相同的服务机会
    - 饥饿免疫：任何候选都不会被永久阻塞
    - 实现简单：硬件开销小，逻辑简单
    - 可预测性：服务间隔固定，便于性能分析
    """

    def _init_queue_state(self, size: int) -> Dict[str, Any]:
        """初始化轮询状态"""
        return {
            'pointer': 0,
            'size': size
        }

    def _do_arbitrate(self, queue_id: str, valid_mask: List[bool]) -> int:
        """执行轮询仲裁"""
        if not any(valid_mask):
            return -1

        state = self.queue_states[queue_id]
        n = len(valid_mask)

        # 从当前指针位置开始查找第一个有效候选
        for i in range(n):
            idx = (state['pointer'] + i) % n
            if valid_mask[idx]:
                # 找到有效候选，更新指针到下一个位置
                state['pointer'] = (idx + 1) % n
                return idx

        # 无有效候选也要移动指针保证公平性
        state['pointer'] = (state['pointer'] + 1) % n
        return -1

    def get_current_pointer(self, queue_id: str = "default") -> int:
        """获取当前指针位置"""
        if queue_id in self.queue_states:
            return self.queue_states[queue_id].get('pointer', 0)
        return 0

    def set_pointer(self, position: int, queue_id: str = "default"):
        """设置指针位置"""
        if queue_id in self.queue_states:
            size = self.queue_states[queue_id].get('size', 1)
            self.queue_states[queue_id]['pointer'] = position % size


class WeightedArbiter(Arbiter):
    """
    加权轮询仲裁器

    为不同候选分配不同的权重，高权重候选在一轮中可能获得多次服务机会。
    实现了在公平性基础上的差异化服务。
    """

    def __init__(self, weights: Optional[List[int]] = None):
        """
        初始化加权轮询仲裁器

        参数:
            weights: 各候选的权重，如果为None则使用等权重
        """
        super().__init__()
        self.default_weights = weights or []

    def _init_queue_state(self, size: int) -> Dict[str, Any]:
        """初始化加权状态"""
        # 使用默认权重或等权重
        if self.default_weights and len(self.default_weights) == size:
            weights = self.default_weights[:]
        else:
            weights = [1] * size

        return {
            'weights': weights,
            'credits': weights[:],
            'pointer': 0,
            'size': size
        }

    def _do_arbitrate(self, queue_id: str, valid_mask: List[bool]) -> int:
        """执行加权轮询仲裁"""
        if not any(valid_mask):
            return -1

        state = self.queue_states[queue_id]
        n = len(valid_mask)

        # 查找有效且有信用的候选
        for i in range(n):
            idx = (state['pointer'] + i) % n
            if valid_mask[idx] and state['credits'][idx] > 0:
                state['credits'][idx] -= 1
                state['pointer'] = (idx + 1) % n

                # 检查是否需要重置信用
                if all(c == 0 for c in state['credits']):
                    state['credits'] = state['weights'][:]

                return idx

        # 所有候选都没信用了，重置信用后重试
        state['credits'] = state['weights'][:]

        # 重新查找
        for i in range(n):
            idx = (state['pointer'] + i) % n
            if valid_mask[idx] and state['credits'][idx] > 0:
                state['credits'][idx] -= 1
                state['pointer'] = (idx + 1) % n
                return idx

        return -1

    def set_weights(self, weights: List[int], queue_id: str = "default"):
        """设置新的权重"""
        if queue_id in self.queue_states:
            state = self.queue_states[queue_id]
            if len(weights) == state['size']:
                state['weights'] = weights[:]
                state['credits'] = weights[:]

    def get_weights(self, queue_id: str = "default") -> List[int]:
        """获取当前权重配置"""
        if queue_id in self.queue_states:
            return self.queue_states[queue_id]['weights'][:]
        return []


class PriorityArbiter(Arbiter):
    """
    固定优先级仲裁器

    按照预定的固定优先级顺序进行仲裁，优先级高的候选总是优先获得服务。
    特点：
    - 简单实现：优先级顺序固定，逻辑简单
    - 确定性：相同输入产生相同输出
    - 可能饥饿：低优先级候选可能被永久阻塞
    - 适用场景：有明确优先级需求的系统
    """

    def __init__(self, priorities: Optional[List[int]] = None):
        """
        初始化固定优先级仲裁器

        参数:
            priorities: 各候选的优先级，数值越小优先级越高
        """
        super().__init__()
        self.default_priorities = priorities or []

    def _init_queue_state(self, size: int) -> Dict[str, Any]:
        """初始化优先级状态"""
        # 使用默认优先级或按索引顺序
        if self.default_priorities and len(self.default_priorities) == size:
            priorities = self.default_priorities[:]
        else:
            priorities = list(range(size))

        # 创建按优先级排序的索引列表
        priority_order = sorted(range(size), key=lambda i: priorities[i])

        return {
            'priorities': priorities,
            'priority_order': priority_order,
            'size': size
        }

    def _do_arbitrate(self, queue_id: str, valid_mask: List[bool]) -> int:
        """执行优先级仲裁"""
        if not any(valid_mask):
            return -1

        state = self.queue_states[queue_id]

        # 按优先级顺序查找第一个有效候选
        for idx in state['priority_order']:
            if valid_mask[idx]:
                return idx

        return -1

    def set_priorities(self, priorities: List[int], queue_id: str = "default"):
        """设置新的优先级"""
        if queue_id in self.queue_states:
            state = self.queue_states[queue_id]
            if len(priorities) == state['size']:
                state['priorities'] = priorities[:]
                state['priority_order'] = sorted(range(state['size']), key=lambda i: priorities[i])


class DynamicPriorityArbiter(Arbiter):
    """
    动态优先级仲裁器

    根据饥饿时间动态调整候选优先级，长时间未获得服务的候选优先级会逐渐提升。
    特点：
    - 饥饿预防：避免任何候选被永久阻塞
    - 动态调整：优先级根据历史服务情况变化
    - 公平性：长期来看提供相对公平的服务
    - 复杂度适中：需要维护动态优先级状态
    """

    def __init__(self, base_priorities: Optional[List[int]] = None, aging_factor: float = 1.0):
        """
        初始化动态优先级仲裁器

        参数:
            base_priorities: 基础优先级，如果为None则使用相等优先级
            aging_factor: 老化因子，控制优先级提升速度
        """
        super().__init__()
        self.default_base_priorities = base_priorities or []
        self.aging_factor = aging_factor

    def _init_queue_state(self, size: int) -> Dict[str, Any]:
        """初始化动态优先级状态"""
        if self.default_base_priorities and len(self.default_base_priorities) == size:
            base_priorities = self.default_base_priorities[:]
        else:
            base_priorities = [0] * size

        return {
            'base_priorities': base_priorities,
            'dynamic_priorities': base_priorities[:],
            'size': size
        }

    def _do_arbitrate(self, queue_id: str, valid_mask: List[bool]) -> int:
        """执行动态优先级仲裁"""
        if not any(valid_mask):
            return -1

        state = self.queue_states[queue_id]
        stats = self.stats[queue_id]

        # 更新动态优先级（饥饿时间越长，优先级越高）
        for i in range(len(valid_mask)):
            starvation = stats['starvation_counter'].get(i, 0)
            state['dynamic_priorities'][i] = state['base_priorities'][i] - starvation * self.aging_factor

        # 在有效候选中找到优先级最高的（数值最小的）
        requesting_indices = [i for i in range(len(valid_mask)) if valid_mask[i]]
        if not requesting_indices:
            return -1

        # 选择优先级最高的候选
        granted_idx = min(requesting_indices, key=lambda i: state['dynamic_priorities'][i])
        return granted_idx

    def set_aging_factor(self, aging_factor: float):
        """设置老化因子"""
        self.aging_factor = aging_factor

    def get_dynamic_priorities(self, queue_id: str = "default") -> List[float]:
        """获取当前动态优先级"""
        if queue_id in self.queue_states:
            return self.queue_states[queue_id]['dynamic_priorities'][:]
        return []


class RandomArbiter(Arbiter):
    """
    随机仲裁器

    从有效的候选中随机选择一个进行授权。
    特点：
    - 简单实现：逻辑非常简单
    - 无偏向性：不偏向任何特定候选
    - 不可预测：服务顺序随机
    - 可能不公平：短期内可能出现不公平现象
    - 适用场景：对公平性要求不高但需要避免确定性偏向的系统
    """

    def __init__(self, seed: Optional[int] = None):
        """
        初始化随机仲裁器

        参数:
            seed: 随机种子，用于确保测试的可重现性
        """
        super().__init__()
        if seed is not None:
            random.seed(seed)

    def _init_queue_state(self, size: int) -> Dict[str, Any]:
        """初始化随机状态"""
        return {'size': size}

    def _do_arbitrate(self, queue_id: str, valid_mask: List[bool]) -> int:
        """执行随机仲裁"""
        if not any(valid_mask):
            return -1

        # 收集所有有效候选的索引
        valid_indices = [i for i, is_valid in enumerate(valid_mask) if is_valid]

        if not valid_indices:
            return -1

        # 随机选择一个
        return random.choice(valid_indices)

    def set_seed(self, seed: int):
        """设置随机种子"""
        random.seed(seed)


class MaxWeightMatchingArbiter:
    """
    最大权重匹配仲裁器（用于交换矩阵场景）

    专门处理多输入-多输出的匹配问题，如NoC中的crossbar交换矩阵仲裁。
    实现了iSLIP算法和其他常见匹配算法。

    注意：此类不继承Arbiter基类，因为它处理的是多对多匹配问题，
    而不是传统的单选仲裁问题。

    特点：
    - 高吞吐量：在理想条件下可达到100%利用率
    - 公平性：通过轮询指针防止饥饿
    - 可配置：支持多种权重计算策略
    - 迭代优化：通过多轮迭代提高匹配质量
    """

    def __init__(self, algorithm: str = "islip", iterations: int = 1,
                 weight_strategy: str = "uniform"):
        """
        初始化最大权重匹配仲裁器

        参数:
            algorithm: 匹配算法 ("islip", "lqf", "ocf", "pim")
            iterations: 迭代轮数
            weight_strategy: 权重策略 ("uniform", "queue_length", "wait_time", "hybrid")
        """
        self.algorithm = algorithm
        self.iterations = iterations
        self.weight_strategy = weight_strategy

        # 仲裁器状态
        self.input_pointers = {}  # 输入端轮询指针
        self.output_pointers = {}  # 输出端轮询指针
        self.queue_states = {}    # 队列状态

        # 统计信息
        self.stats = defaultdict(lambda: {
            'total_matches': 0,
            'successful_matches': 0,
            'average_matching_size': 0.0,
            'throughput_efficiency': 0.0,
            'input_utilization': defaultdict(int),
            'output_utilization': defaultdict(int)
        })

    def match(self, request_matrix: List[List[bool]],
              weight_matrix: Optional[List[List[float]]] = None,
              queue_id: str = "default") -> List[Tuple[int, int]]:
        """
        执行输入-输出匹配

        参数:
            request_matrix: NxM请求矩阵，request_matrix[i][j]表示输入i请求输出j
            weight_matrix: NxM权重矩阵，可选，如果未提供则使用权重策略计算
            queue_id: 队列标识符

        返回:
            匹配结果列表：[(输入索引, 输出索引), ...]
        """
        if not request_matrix or not request_matrix[0]:
            return []

        num_inputs = len(request_matrix)
        num_outputs = len(request_matrix[0])

        # 初始化或更新状态
        self._init_or_update_state(queue_id, num_inputs, num_outputs)

        # 计算权重矩阵
        if weight_matrix is None:
            weight_matrix = self._calculate_weights(request_matrix, queue_id)

        # 执行匹配算法
        if self.algorithm == "islip":
            matches = self._islip_matching(request_matrix, weight_matrix, queue_id)
        elif self.algorithm == "lqf":
            matches = self._lqf_matching(request_matrix, weight_matrix, queue_id)
        elif self.algorithm == "ocf":
            matches = self._ocf_matching(request_matrix, weight_matrix, queue_id)
        elif self.algorithm == "pim":
            matches = self._pim_matching(request_matrix, weight_matrix, queue_id)
        else:
            raise ValueError(f"不支持的匹配算法: {self.algorithm}")

        # 更新统计信息
        self._update_match_stats(queue_id, matches, request_matrix)

        return matches

    def _init_or_update_state(self, queue_id: str, num_inputs: int, num_outputs: int):
        """初始化或更新队列状态"""
        if queue_id not in self.queue_states:
            self.queue_states[queue_id] = {
                'num_inputs': num_inputs,
                'num_outputs': num_outputs,
                'queue_lengths': [[0] * num_outputs for _ in range(num_inputs)],
                'wait_times': [[0] * num_outputs for _ in range(num_inputs)],
                'last_service_time': [[0] * num_outputs for _ in range(num_inputs)]
            }
            self.input_pointers[queue_id] = [0] * num_inputs
            self.output_pointers[queue_id] = [0] * num_outputs

    def _calculate_weights(self, request_matrix: List[List[bool]],
                          queue_id: str) -> List[List[float]]:
        """根据权重策略计算权重矩阵"""
        num_inputs = len(request_matrix)
        num_outputs = len(request_matrix[0])
        weights = [[0.0] * num_outputs for _ in range(num_inputs)]

        state = self.queue_states[queue_id]

        for i in range(num_inputs):
            for j in range(num_outputs):
                if not request_matrix[i][j]:
                    continue

                if self.weight_strategy == "uniform":
                    weights[i][j] = 1.0
                elif self.weight_strategy == "queue_length":
                    weights[i][j] = float(state['queue_lengths'][i][j])
                elif self.weight_strategy == "wait_time":
                    weights[i][j] = float(state['wait_times'][i][j])
                elif self.weight_strategy == "hybrid":
                    # 队列长度 + 等待时间的混合权重
                    weights[i][j] = (state['queue_lengths'][i][j] * 0.7 +
                                   state['wait_times'][i][j] * 0.3)

        return weights

    def _islip_matching(self, request_matrix: List[List[bool]],
                       weight_matrix: List[List[float]],
                       queue_id: str) -> List[Tuple[int, int]]:
        """iSLIP算法实现"""
        num_inputs = len(request_matrix)
        num_outputs = len(request_matrix[0])
        matches = []

        # 获取轮询指针
        input_ptrs = self.input_pointers[queue_id][:]
        output_ptrs = self.output_pointers[queue_id][:]

        # 追踪已匹配的输入和输出
        matched_inputs = set()
        matched_outputs = set()

        for iteration in range(self.iterations):
            grants = {}  # output -> input
            requests = {}  # input -> [outputs]

            # 第一阶段：请求阶段
            for i in range(num_inputs):
                if i in matched_inputs:
                    continue
                requests[i] = []
                for j in range(num_outputs):
                    if request_matrix[i][j] and j not in matched_outputs:
                        requests[i].append(j)

            # 第二阶段：授权阶段（输出端轮询选择）
            for j in range(num_outputs):
                if j in matched_outputs:
                    continue

                requesting_inputs = [i for i in requests if j in requests[i]]
                if not requesting_inputs:
                    continue

                # 从输出端指针位置开始轮询
                selected_input = None
                max_weight = -1

                for k in range(num_inputs):
                    i = (output_ptrs[j] + k) % num_inputs
                    if i in requesting_inputs:
                        if weight_matrix[i][j] > max_weight:
                            max_weight = weight_matrix[i][j]
                            selected_input = i

                if selected_input is not None:
                    grants[j] = selected_input

            # 第三阶段：接受阶段（输入端轮询选择）
            for i in range(num_inputs):
                if i in matched_inputs:
                    continue

                granting_outputs = [j for j in grants if grants[j] == i]
                if not granting_outputs:
                    continue

                # 从输入端指针位置开始轮询
                selected_output = None
                max_weight = -1

                for k in range(num_outputs):
                    j = (input_ptrs[i] + k) % num_outputs
                    if j in granting_outputs:
                        if weight_matrix[i][j] > max_weight:
                            max_weight = weight_matrix[i][j]
                            selected_output = j

                if selected_output is not None:
                    matches.append((i, selected_output))
                    matched_inputs.add(i)
                    matched_outputs.add(selected_output)

                    # 更新指针（只有成功匹配时才更新）
                    input_ptrs[i] = (selected_output + 1) % num_outputs
                    output_ptrs[selected_output] = (i + 1) % num_inputs

        # 更新指针状态
        self.input_pointers[queue_id] = input_ptrs
        self.output_pointers[queue_id] = output_ptrs

        return matches

    def _lqf_matching(self, request_matrix: List[List[bool]],
                     weight_matrix: List[List[float]],
                     queue_id: str) -> List[Tuple[int, int]]:
        """最长队列优先（Longest Queue First）匹配"""
        num_inputs = len(request_matrix)
        num_outputs = len(request_matrix[0])

        # 创建带权重的请求列表
        weighted_requests = []
        for i in range(num_inputs):
            for j in range(num_outputs):
                if request_matrix[i][j]:
                    weighted_requests.append((weight_matrix[i][j], i, j))

        # 按权重降序排序
        weighted_requests.sort(reverse=True)

        matches = []
        used_inputs = set()
        used_outputs = set()

        for weight, i, j in weighted_requests:
            if i not in used_inputs and j not in used_outputs:
                matches.append((i, j))
                used_inputs.add(i)
                used_outputs.add(j)

        return matches

    def _ocf_matching(self, request_matrix: List[List[bool]],
                     weight_matrix: List[List[float]],
                     queue_id: str) -> List[Tuple[int, int]]:
        """最老数据包优先（Oldest Cell First）匹配"""
        # 实现与LQF类似，但使用等待时间作为权重
        return self._lqf_matching(request_matrix, weight_matrix, queue_id)

    def _pim_matching(self, request_matrix: List[List[bool]],
                     weight_matrix: List[List[float]],
                     queue_id: str) -> List[Tuple[int, int]]:
        """并行迭代匹配（Parallel Iterative Matching）"""
        num_inputs = len(request_matrix)
        num_outputs = len(request_matrix[0])
        matches = []

        # 随机置换以避免固定偏好
        input_order = list(range(num_inputs))
        output_order = list(range(num_outputs))
        random.shuffle(input_order)
        random.shuffle(output_order)

        matched_inputs = set()
        matched_outputs = set()

        for iteration in range(self.iterations):
            # 并行匹配
            iteration_matches = []

            for i in input_order:
                if i in matched_inputs:
                    continue

                available_outputs = [j for j in output_order
                                   if j not in matched_outputs and request_matrix[i][j]]

                if available_outputs:
                    # 选择权重最大的输出
                    best_output = max(available_outputs, key=lambda j: weight_matrix[i][j])
                    iteration_matches.append((i, best_output))

            # 解决冲突（多个输入选择同一个输出）
            output_conflicts = defaultdict(list)
            for i, j in iteration_matches:
                output_conflicts[j].append(i)

            for j, competing_inputs in output_conflicts.items():
                if len(competing_inputs) == 1:
                    i = competing_inputs[0]
                    matches.append((i, j))
                    matched_inputs.add(i)
                    matched_outputs.add(j)
                else:
                    # 选择权重最大的输入
                    winner = max(competing_inputs, key=lambda i: weight_matrix[i][j])
                    matches.append((winner, j))
                    matched_inputs.add(winner)
                    matched_outputs.add(j)

        return matches

    def _update_match_stats(self, queue_id: str, matches: List[Tuple[int, int]],
                           request_matrix: List[List[bool]]):
        """更新匹配统计信息"""
        stats = self.stats[queue_id]
        stats['total_matches'] += 1

        if matches:
            stats['successful_matches'] += 1

            # 更新平均匹配大小
            current_size = len(matches)
            prev_avg = stats['average_matching_size']
            stats['average_matching_size'] = (prev_avg * (stats['successful_matches'] - 1) +
                                           current_size) / stats['successful_matches']

            # 计算吞吐量效率
            total_requests = sum(sum(row) for row in request_matrix)
            if total_requests > 0:
                efficiency = len(matches) / total_requests
                prev_eff = stats['throughput_efficiency']
                stats['throughput_efficiency'] = (prev_eff * (stats['successful_matches'] - 1) +
                                                efficiency) / stats['successful_matches']

            # 更新端口利用率
            for i, j in matches:
                stats['input_utilization'][i] += 1
                stats['output_utilization'][j] += 1

    def get_match_stats(self, queue_id: str = None) -> Dict[str, Any]:
        """获取匹配统计信息"""
        if queue_id:
            if queue_id not in self.stats:
                return {}
            return dict(self.stats[queue_id])
        else:
            return {qid: dict(stats) for qid, stats in self.stats.items()}

    def reset(self, queue_id: str = None):
        """重置仲裁器状态"""
        if queue_id:
            if queue_id in self.queue_states:
                del self.queue_states[queue_id]
            if queue_id in self.input_pointers:
                del self.input_pointers[queue_id]
            if queue_id in self.output_pointers:
                del self.output_pointers[queue_id]
            if queue_id in self.stats:
                del self.stats[queue_id]
        else:
            self.queue_states.clear()
            self.input_pointers.clear()
            self.output_pointers.clear()
            self.stats.clear()

    def update_queue_info(self, queue_id: str, input_idx: int, output_idx: int,
                         queue_length: int = None, wait_time: int = None):
        """更新队列信息（用于权重计算）"""
        if queue_id in self.queue_states:
            state = self.queue_states[queue_id]
            if queue_length is not None:
                state['queue_lengths'][input_idx][output_idx] = queue_length
            if wait_time is not None:
                state['wait_times'][input_idx][output_idx] = wait_time


class TokenBucketArbiter(Arbiter):
    """
    令牌桶仲裁器

    为每个候选维护一个令牌桶，候选获得服务需要消耗令牌，令牌以固定速率补充。
    特点：
    - 流量控制：可以控制每个候选的服务速率
    - 突发支持：允许短期突发访问
    - 长期公平：长期来看各候选获得相应的服务机会
    - 实现复杂：需要维护令牌桶状态
    """

    def __init__(self, bucket_size: int = 10, refill_rate: float = 1.0,
                 port_rates: Optional[List[float]] = None):
        """
        初始化令牌桶仲裁器

        参数:
            bucket_size: 每个候选的令牌桶大小
            refill_rate: 默认令牌补充速率（令牌/周期）
            port_rates: 各候选的令牌补充速率，如果为None则使用默认速率
        """
        super().__init__()
        self.bucket_size = bucket_size
        self.refill_rate = refill_rate
        self.default_port_rates = port_rates or []

    def _init_queue_state(self, size: int) -> Dict[str, Any]:
        """初始化令牌桶状态"""
        # 使用默认速率或统一速率
        if self.default_port_rates and len(self.default_port_rates) == size:
            port_rates = self.default_port_rates[:]
        else:
            port_rates = [self.refill_rate] * size

        return {
            'port_rates': port_rates,
            'token_buckets': [self.bucket_size] * size,
            'last_refill_time': time.time(),
            'size': size
        }

    def _do_arbitrate(self, queue_id: str, valid_mask: List[bool]) -> int:
        """执行令牌桶仲裁"""
        if not any(valid_mask):
            return -1

        state = self.queue_states[queue_id]

        # 补充令牌
        self._refill_tokens(state)

        # 找到有效且有令牌的候选
        eligible_indices = []
        for i, is_valid in enumerate(valid_mask):
            if is_valid and state['token_buckets'][i] > 0:
                eligible_indices.append(i)

        if not eligible_indices:
            return -1

        # 使用轮询策略选择（也可以使用其他策略）
        granted_idx = min(eligible_indices)

        # 消耗令牌
        state['token_buckets'][granted_idx] -= 1

        return granted_idx

    def _refill_tokens(self, state: Dict[str, Any]):
        """补充令牌"""
        current_time = time.time()
        time_elapsed = current_time - state['last_refill_time']

        for i in range(state['size']):
            # 计算应该补充的令牌数
            tokens_to_add = state['port_rates'][i] * time_elapsed
            state['token_buckets'][i] = min(self.bucket_size,
                                          state['token_buckets'][i] + tokens_to_add)

        state['last_refill_time'] = current_time

    def get_token_status(self, queue_id: str = "default") -> List[float]:
        """获取各候选的令牌数量"""
        if queue_id in self.queue_states:
            return self.queue_states[queue_id]['token_buckets'][:]
        return []

    def set_bucket_size(self, bucket_size: int):
        """设置令牌桶大小"""
        self.bucket_size = bucket_size

    def set_port_rates(self, port_rates: List[float], queue_id: str = "default"):
        """设置各候选的令牌补充速率"""
        if queue_id in self.queue_states:
            state = self.queue_states[queue_id]
            if len(port_rates) == state['size']:
                state['port_rates'] = port_rates[:]


# 工厂函数
def create_arbiter(arbiter_type: str, **kwargs) -> Arbiter:
    """
    根据类型创建单选仲裁器的工厂函数

    参数:
        arbiter_type: 仲裁器类型
        **kwargs: 传递给仲裁器构造函数的参数

    返回:
        仲裁器实例

    抛出:
        ValueError: 如果仲裁器类型未知
    """
    arbiter_classes = {
        'round_robin': RoundRobinArbiter,
        'weighted': WeightedArbiter,
        'priority': PriorityArbiter,
        'dynamic': DynamicPriorityArbiter,
        'random': RandomArbiter,
        'token_bucket': TokenBucketArbiter,
    }

    arbiter_class = arbiter_classes.get(arbiter_type)
    if not arbiter_class:
        available_types = ', '.join(arbiter_classes.keys())
        raise ValueError(f"未知的仲裁器类型: {arbiter_type}. 可用类型: {available_types}")

    return arbiter_class(**kwargs)


def create_matching_arbiter(arbiter_type: str, **kwargs) -> MaxWeightMatchingArbiter:
    """
    根据类型创建匹配仲裁器的工厂函数

    参数:
        arbiter_type: 匹配仲裁器类型
        **kwargs: 传递给仲裁器构造函数的参数

    返回:
        匹配仲裁器实例

    抛出:
        ValueError: 如果仲裁器类型未知
    """
    matching_arbiter_classes = {
        'max_weight_matching': MaxWeightMatchingArbiter,
        'islip': MaxWeightMatchingArbiter,
        'lqf': MaxWeightMatchingArbiter,
        'ocf': MaxWeightMatchingArbiter,
        'pim': MaxWeightMatchingArbiter,
    }

    if arbiter_type not in matching_arbiter_classes:
        available_types = ', '.join(matching_arbiter_classes.keys())
        raise ValueError(f"未知的匹配仲裁器类型: {arbiter_type}. 可用类型: {available_types}")

    # 对于具体算法类型，将其作为algorithm参数传递
    if arbiter_type in ['islip', 'lqf', 'ocf', 'pim']:
        kwargs['algorithm'] = arbiter_type

    return MaxWeightMatchingArbiter(**kwargs)


def create_arbiter_from_config(config: Dict[str, Any]):
    """
    根据配置字典创建仲裁器

    参数:
        config: 配置字典，必须包含'type'字段

    返回:
        仲裁器实例（Arbiter或MaxWeightMatchingArbiter）
    """
    if 'type' not in config:
        raise ValueError("配置中缺少'type'字段")

    arbiter_type = config['type']
    # 移除type字段，其余作为参数传递
    kwargs = {k: v for k, v in config.items() if k != 'type'}

    # 检查是否是匹配仲裁器类型
    matching_types = ['max_weight_matching', 'islip', 'lqf', 'ocf', 'pim']
    if arbiter_type in matching_types:
        return create_matching_arbiter(arbiter_type, **kwargs)
    else:
        return create_arbiter(arbiter_type, **kwargs)


# 便捷创建函数
def create_round_robin_arbiter() -> RoundRobinArbiter:
    """创建标准轮询仲裁器"""
    return RoundRobinArbiter()


def create_weighted_arbiter(weights: List[int]) -> WeightedArbiter:
    """创建加权轮询仲裁器"""
    return WeightedArbiter(weights=weights)


def create_priority_arbiter(priorities: List[int]) -> PriorityArbiter:
    """创建固定优先级仲裁器"""
    return PriorityArbiter(priorities=priorities)


def create_dynamic_priority_arbiter(base_priorities: Optional[List[int]] = None,
                                   aging_factor: float = 1.0) -> DynamicPriorityArbiter:
    """创建动态优先级仲裁器"""
    return DynamicPriorityArbiter(base_priorities=base_priorities, aging_factor=aging_factor)


def create_random_arbiter(seed: Optional[int] = None) -> RandomArbiter:
    """创建随机仲裁器"""
    return RandomArbiter(seed=seed)


def create_token_bucket_arbiter(bucket_size: int = 10, refill_rate: float = 1.0,
                               port_rates: Optional[List[float]] = None) -> TokenBucketArbiter:
    """创建令牌桶仲裁器"""
    return TokenBucketArbiter(bucket_size=bucket_size, refill_rate=refill_rate, port_rates=port_rates)


def create_max_weight_matching_arbiter(algorithm: str = "islip", iterations: int = 1,
                                     weight_strategy: str = "uniform") -> MaxWeightMatchingArbiter:
    """创建最大权重匹配仲裁器"""
    return MaxWeightMatchingArbiter(algorithm=algorithm, iterations=iterations,
                                  weight_strategy=weight_strategy)