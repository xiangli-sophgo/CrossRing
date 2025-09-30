"""
CrossRing NoC 统一多对多仲裁系统

所有仲裁器都实现真正的多对多匹配，确保无资源冲突和全局优化。
设计理念：NoC中的仲裁本质上是多输入争夺多输出的匹配问题。

主要仲裁算法：
- RoundRobinArbiter: 轮询多对多匹配
- WeightedArbiter: 加权贪心匹配
- PriorityArbiter: 优先级多对多匹配
- DynamicPriorityArbiter: 动态优先级匹配
- MaxWeightMatchingArbiter: iSLIP/LQF/OCF/PIM高级匹配算法
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict
import random
import time


class Arbiter(ABC):
    """
    统一仲裁器基类

    核心理念：所有仲裁都是多对多匹配问题
    - 输入：请求矩阵 (N个输入 × M个输出)
    - 输出：匹配对列表 [(input_idx, output_idx), ...]
    - 约束：每个输入最多匹配一个输出，每个输出最多匹配一个输入
    """

    def __init__(self):
        """初始化仲裁器"""
        # 每个队列的独立状态
        self.queue_states = {}

        # 统计信息
        self.stats = defaultdict(lambda: {
            'total_matches': 0,
            'successful_matches': 0,
            'total_matched_pairs': 0,
            'input_utilization': defaultdict(int),
            'output_utilization': defaultdict(int),
            'input_starvation': defaultdict(int),
            'output_starvation': defaultdict(int)
        })

    @abstractmethod
    def match(self, request_matrix: List[List[bool]],
              weight_matrix: Optional[List[List[float]]] = None,
              queue_id: str = "default") -> List[Tuple[int, int]]:
        """
        多对多匹配（唯一核心接口）

        参数:
            request_matrix: NxM布尔矩阵，request_matrix[i][j]=True表示输入i请求输出j
            weight_matrix: NxM权重矩阵（可选），用于优先级/加权仲裁
            queue_id: 队列标识符，支持多队列独立状态管理

        返回:
            匹配对列表：[(input_idx, output_idx), ...]

        约束:
            - 每个input_idx在结果中最多出现一次
            - 每个output_idx在结果中最多出现一次
            - 只匹配request_matrix[i][j]=True的位置
        """
        pass

    @abstractmethod
    def _init_queue_state(self, num_inputs: int, num_outputs: int) -> Dict[str, Any]:
        """
        初始化队列状态

        参数:
            num_inputs: 输入数量
            num_outputs: 输出数量

        返回:
            队列状态字典
        """
        pass

    def _update_stats(self, queue_id: str, matches: List[Tuple[int, int]],
                     request_matrix: List[List[bool]]):
        """更新匹配统计信息"""
        stats = self.stats[queue_id]
        stats['total_matches'] += 1

        if matches:
            stats['successful_matches'] += 1
            stats['total_matched_pairs'] += len(matches)

            # 更新端口利用率
            matched_inputs = set()
            matched_outputs = set()
            for input_idx, output_idx in matches:
                stats['input_utilization'][input_idx] += 1
                stats['output_utilization'][output_idx] += 1
                matched_inputs.add(input_idx)
                matched_outputs.add(output_idx)

            # 更新饥饿计数器
            for i in range(len(request_matrix)):
                if any(request_matrix[i]) and i not in matched_inputs:
                    stats['input_starvation'][i] += 1
                elif i in matched_inputs:
                    stats['input_starvation'][i] = 0

            for j in range(len(request_matrix[0]) if request_matrix else 0):
                has_request = any(request_matrix[i][j] for i in range(len(request_matrix)))
                if has_request and j not in matched_outputs:
                    stats['output_starvation'][j] += 1
                elif j in matched_outputs:
                    stats['output_starvation'][j] = 0

    def get_stats(self, queue_id: str = None) -> Dict[str, Any]:
        """
        获取匹配统计信息

        参数:
            queue_id: 指定队列，None表示获取所有队列统计

        返回:
            统计信息字典
        """
        if queue_id:
            if queue_id not in self.stats:
                return {}

            stats = self.stats[queue_id]
            total = stats['total_matches']
            successful = stats['successful_matches']

            # 计算匹配成功率
            success_rate = successful / total if total > 0 else 0.0

            # 计算平均匹配数
            avg_matched_pairs = stats['total_matched_pairs'] / successful if successful > 0 else 0.0

            # 计算公平性指标（基于端口利用率）
            input_utils = dict(stats['input_utilization'])
            output_utils = dict(stats['output_utilization'])

            def calc_fairness(utils):
                if not utils:
                    return 1.0
                values = list(utils.values())
                if sum(values) == 0:
                    return 1.0
                avg = sum(values) / len(values)
                variance = sum((v - avg) ** 2 for v in values) / len(values)
                return 1.0 / (1.0 + variance) if variance > 0 else 1.0

            return {
                'queue_id': queue_id,
                'total_matches': total,
                'successful_matches': successful,
                'success_rate': success_rate,
                'average_matched_pairs': avg_matched_pairs,
                'input_utilization': input_utils,
                'output_utilization': output_utils,
                'input_fairness': calc_fairness(input_utils),
                'output_fairness': calc_fairness(output_utils),
                'max_input_starvation': max(stats['input_starvation'].values()) if stats['input_starvation'] else 0,
                'max_output_starvation': max(stats['output_starvation'].values()) if stats['output_starvation'] else 0
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
                state = self.queue_states[queue_id]
                num_inputs = state.get('num_inputs', 1)
                num_outputs = state.get('num_outputs', 1)
                self.queue_states[queue_id] = self._init_queue_state(num_inputs, num_outputs)
            if queue_id in self.stats:
                del self.stats[queue_id]
        else:
            # 重置所有队列
            for qid in list(self.queue_states.keys()):
                state = self.queue_states[qid]
                num_inputs = state.get('num_inputs', 1)
                num_outputs = state.get('num_outputs', 1)
                self.queue_states[qid] = self._init_queue_state(num_inputs, num_outputs)
            self.stats.clear()


class RoundRobinArbiter(Arbiter):
    """
    轮询多对多匹配仲裁器

    特点：
    - 双端轮询：输入和输出都维护轮询指针
    - 全局公平：长期来看每个输入和输出都获得相同的服务机会
    - 无冲突：确保每个输出只被一个输入占用
    - 饥饿免疫：任何输入/输出都不会被永久阻塞

    算法：
    1. 按输入轮询顺序处理每个输入
    2. 每个输入从其指针位置开始选择可用输出
    3. 输入和输出的指针都会推进
    """

    def _init_queue_state(self, num_inputs: int, num_outputs: int) -> Dict[str, Any]:
        """初始化轮询状态"""
        return {
            'num_inputs': num_inputs,
            'num_outputs': num_outputs,
            'input_pointers': [0] * num_inputs,   # 每个输入的轮询指针
            'output_pointers': [0] * num_outputs,  # 每个输出的轮询指针
            'input_start': 0  # 输入处理起始位置（轮询）
        }

    def match(self, request_matrix: List[List[bool]],
              weight_matrix: Optional[List[List[float]]] = None,
              queue_id: str = "default") -> List[Tuple[int, int]]:
        """执行轮询多对多匹配"""
        if not request_matrix or not request_matrix[0]:
            return []

        num_inputs = len(request_matrix)
        num_outputs = len(request_matrix[0])

        # 初始化或更新状态
        if queue_id not in self.queue_states:
            self.queue_states[queue_id] = self._init_queue_state(num_inputs, num_outputs)

        state = self.queue_states[queue_id]
        input_ptrs = state['input_pointers']
        output_ptrs = state['output_pointers']
        input_start = state['input_start']

        matches = []
        matched_outputs = set()

        # 按轮询顺序处理每个输入
        for i_offset in range(num_inputs):
            input_idx = (input_start + i_offset) % num_inputs

            # 检查该输入是否有请求
            if not any(request_matrix[input_idx]):
                continue

            # 从该输入的指针位置开始查找可用输出
            input_ptr = input_ptrs[input_idx]
            for j_offset in range(num_outputs):
                output_idx = (input_ptr + j_offset) % num_outputs

                # 检查是否可以匹配
                if (request_matrix[input_idx][output_idx] and
                    output_idx not in matched_outputs):
                    # 匹配成功
                    matches.append((input_idx, output_idx))
                    matched_outputs.add(output_idx)

                    # 推进输入和输出指针
                    input_ptrs[input_idx] = (output_idx + 1) % num_outputs
                    output_ptrs[output_idx] = (input_idx + 1) % num_inputs

                    break  # 该输入已匹配，处理下一个输入

        # 推进输入起始位置（下次从不同的输入开始）
        state['input_start'] = (input_start + 1) % num_inputs

        # 更新统计
        self._update_stats(queue_id, matches, request_matrix)

        return matches

    def get_pointers(self, queue_id: str = "default") -> Dict[str, Any]:
        """获取当前指针状态"""
        if queue_id in self.queue_states:
            state = self.queue_states[queue_id]
            return {
                'input_pointers': state['input_pointers'][:],
                'output_pointers': state['output_pointers'][:],
                'input_start': state['input_start']
            }
        return {}


class WeightedArbiter(Arbiter):
    """
    加权贪心多对多匹配仲裁器

    特点：
    - 基于权重的贪心匹配：权重高的(input,output)对优先匹配
    - 全局优化：按权重排序，从高到低依次匹配
    - 无冲突：确保每个输出只被一个输入占用

    算法：
    1. 收集所有有效的(input, output, weight)三元组
    2. 按权重降序排序
    3. 从高到低依次匹配，跳过已占用的输入/输出
    """

    def _init_queue_state(self, num_inputs: int, num_outputs: int) -> Dict[str, Any]:
        """初始化状态"""
        return {
            'num_inputs': num_inputs,
            'num_outputs': num_outputs
        }

    def match(self, request_matrix: List[List[bool]],
              weight_matrix: Optional[List[List[float]]] = None,
              queue_id: str = "default") -> List[Tuple[int, int]]:
        """执行加权贪心匹配"""
        if not request_matrix or not request_matrix[0]:
            return []

        num_inputs = len(request_matrix)
        num_outputs = len(request_matrix[0])

        # 初始化状态
        if queue_id not in self.queue_states:
            self.queue_states[queue_id] = self._init_queue_state(num_inputs, num_outputs)

        # 如果没有提供权重矩阵，使用统一权重1.0
        if weight_matrix is None:
            weight_matrix = [[1.0 if request_matrix[i][j] else 0.0
                            for j in range(num_outputs)]
                           for i in range(num_inputs)]

        # 收集所有有效的(input, output, weight)三元组
        candidates = []
        for i in range(num_inputs):
            for j in range(num_outputs):
                if request_matrix[i][j]:
                    candidates.append((weight_matrix[i][j], i, j))

        # 按权重降序排序
        candidates.sort(reverse=True)

        # 贪心匹配
        matches = []
        matched_inputs = set()
        matched_outputs = set()

        for weight, input_idx, output_idx in candidates:
            if input_idx not in matched_inputs and output_idx not in matched_outputs:
                matches.append((input_idx, output_idx))
                matched_inputs.add(input_idx)
                matched_outputs.add(output_idx)

        # 更新统计
        self._update_stats(queue_id, matches, request_matrix)

        return matches


class PriorityArbiter(Arbiter):
    """
    固定优先级多对多匹配仲裁器

    特点：
    - 输入优先级：按输入优先级顺序处理
    - 确定性：相同输入产生相同输出
    - 可能饥饿：低优先级输入可能被永久阻塞
    - 适用场景：有明确优先级需求的系统

    算法：
    1. 按输入优先级排序（优先级值越小越优先）
    2. 高优先级输入先选择输出
    3. 每个输入选择第一个可用的输出
    """

    def __init__(self, input_priorities: Optional[List[int]] = None):
        """
        初始化固定优先级仲裁器

        参数:
            input_priorities: 各输入的优先级，数值越小优先级越高
        """
        super().__init__()
        self.default_input_priorities = input_priorities or []

    def _init_queue_state(self, num_inputs: int, num_outputs: int) -> Dict[str, Any]:
        """初始化优先级状态"""
        # 使用默认优先级或按索引顺序
        if self.default_input_priorities and len(self.default_input_priorities) == num_inputs:
            input_priorities = self.default_input_priorities[:]
        else:
            input_priorities = list(range(num_inputs))

        # 创建按优先级排序的输入索引列表
        input_order = sorted(range(num_inputs), key=lambda i: input_priorities[i])

        return {
            'num_inputs': num_inputs,
            'num_outputs': num_outputs,
            'input_priorities': input_priorities,
            'input_order': input_order
        }

    def match(self, request_matrix: List[List[bool]],
              weight_matrix: Optional[List[List[float]]] = None,
              queue_id: str = "default") -> List[Tuple[int, int]]:
        """执行优先级匹配"""
        if not request_matrix or not request_matrix[0]:
            return []

        num_inputs = len(request_matrix)
        num_outputs = len(request_matrix[0])

        # 初始化状态
        if queue_id not in self.queue_states:
            self.queue_states[queue_id] = self._init_queue_state(num_inputs, num_outputs)

        state = self.queue_states[queue_id]
        input_order = state['input_order']

        matches = []
        matched_outputs = set()

        # 按优先级顺序处理每个输入
        for input_idx in input_order:
            # 检查该输入是否有请求
            if not any(request_matrix[input_idx]):
                continue

            # 找到第一个可用的输出
            for output_idx in range(num_outputs):
                if (request_matrix[input_idx][output_idx] and
                    output_idx not in matched_outputs):
                    # 匹配成功
                    matches.append((input_idx, output_idx))
                    matched_outputs.add(output_idx)
                    break  # 该输入已匹配，处理下一个输入

        # 更新统计
        self._update_stats(queue_id, matches, request_matrix)

        return matches



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




# 工厂函数
def create_arbiter(arbiter_type: str, **kwargs) -> Arbiter:
    """
    根据类型创建仲裁器的工厂函数

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

    # 检查是否是匹配仲裁器类型（iSLIP等高级算法）
    matching_types = ['max_weight_matching', 'islip', 'lqf', 'ocf', 'pim']
    if arbiter_type in matching_types:
        return create_matching_arbiter(arbiter_type, **kwargs)
    else:
        # 基础仲裁器（轮询、加权、优先级）
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




def create_max_weight_matching_arbiter(algorithm: str = "islip", iterations: int = 1,
                                     weight_strategy: str = "uniform") -> MaxWeightMatchingArbiter:
    """创建最大权重匹配仲裁器"""
    return MaxWeightMatchingArbiter(algorithm=algorithm, iterations=iterations,
                                  weight_strategy=weight_strategy)


