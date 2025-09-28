"""
仲裁器测试模块

用于验证各种仲裁算法的正确性和性能
包含压力测试、长时间运行测试和性能基准测试
"""

import sys
import os
import random
import time
import statistics
from typing import List, Dict, Any, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "utils"))

from src.utils.arbitration import (
    RoundRobinArbiter, ConditionalRoundRobinArbiter, WeightedRoundRobinArbiter,
    FixedPriorityArbiter, DynamicPriorityArbiter, RandomArbiter, TokenBucketArbiter,
    create_round_robin_arbiter, create_weighted_round_robin_arbiter,
    create_fixed_priority_arbiter, create_dynamic_priority_arbiter,
    create_random_arbiter, create_token_bucket_arbiter
)


def test_round_robin_arbiter():
    """测试标准轮询仲裁器"""
    print("=== 测试标准轮询仲裁器 ===")

    arbiter = RoundRobinArbiter(4)

    # 测试用例：所有端口都有请求
    test_cases = [
        [True, True, True, True],  # 所有端口有请求
        [True, False, True, False],  # 部分端口有请求
        [False, False, False, True],  # 只有一个端口有请求
        [False, False, False, False],  # 没有端口有请求
    ]

    print("测试序列:")
    for i, requests in enumerate(test_cases):
        granted_port = arbiter.arbitrate(requests)
        print(f"周期 {i}: 请求={requests}, 授权端口={granted_port}, 指针位置={arbiter.get_current_pointer()}")

    # 显示统计信息
    stats = arbiter.get_stats()
    print(f"\n统计信息:")
    print(f"总仲裁次数: {stats['total_arbitrations']}")
    print(f"成功仲裁次数: {stats['successful_arbitrations']}")
    print(f"成功率: {stats['success_rate']:.2%}")
    print(f"各端口授权次数: {stats['grants_per_port']}")
    print(f"公平性指标: {stats['fairness_index']:.3f}")
    print()


def test_conditional_round_robin_arbiter():
    """测试条件轮询仲裁器"""
    print("=== 测试条件轮询仲裁器 ===")

    arbiter = ConditionalRoundRobinArbiter(4)

    # 测试低负载场景下的行为差异
    test_cases = [
        [True, False, False, False],  # 只有端口0有请求
        [False, False, False, False],  # 无请求
        [True, False, False, False],  # 再次只有端口0有请求
        [False, True, False, False],  # 端口1有请求
    ]

    print("测试序列:")
    for i, requests in enumerate(test_cases):
        granted_port = arbiter.arbitrate(requests)
        print(f"周期 {i}: 请求={requests}, 授权端口={granted_port}")

    stats = arbiter.get_stats()
    print(f"\n统计信息:")
    print(f"各端口授权次数: {stats['grants_per_port']}")
    print(f"公平性指标: {stats['fairness_index']:.3f}")
    print()


def test_weighted_round_robin_arbiter():
    """测试加权轮询仲裁器"""
    print("=== 测试加权轮询仲裁器 ===")

    # 设置权重：端口0权重为3，端口1权重为2，端口2和3权重为1
    weights = [3, 2, 1, 1]
    arbiter = WeightedRoundRobinArbiter(4, weights)

    print(f"端口权重配置: {arbiter.get_weights()}")

    # 模拟多个周期，所有端口都有请求
    requests = [True, True, True, True]

    print("测试序列（所有端口都有请求）:")
    granted_sequence = []
    for i in range(10):  # 运行10个周期
        granted_port = arbiter.arbitrate(requests)
        granted_sequence.append(granted_port)
        credits = arbiter.get_credits()
        print(f"周期 {i}: 授权端口={granted_port}, 剩余信用={credits}")

    print(f"授权序列: {granted_sequence}")

    stats = arbiter.get_stats()
    print(f"\n统计信息:")
    print(f"各端口授权次数: {stats['grants_per_port']}")
    print(f"理论比例: {[w/sum(weights) for w in weights]}")
    print(f"实际比例: {[g/sum(stats['grants_per_port']) if sum(stats['grants_per_port']) > 0 else 0 for g in stats['grants_per_port']]}")
    print(f"公平性指标: {stats['fairness_index']:.3f}")
    print()


def test_fairness_comparison():
    """对比不同仲裁器的公平性"""
    print("=== 公平性对比测试 ===")

    # 创建不同的仲裁器
    rr_arbiter = RoundRobinArbiter(4)
    wrr_arbiter = WeightedRoundRobinArbiter(4, [2, 2, 1, 1])

    # 运行相同的测试序列
    test_sequence = [
        [True, True, False, False],
        [True, False, True, False],
        [False, True, True, False],
        [True, True, True, False],
        [True, False, False, True],
    ] * 4  # 重复4次

    print("运行测试序列...")
    for requests in test_sequence:
        rr_arbiter.arbitrate(requests)
        wrr_arbiter.arbitrate(requests)

    # 比较结果
    rr_stats = rr_arbiter.get_stats()
    wrr_stats = wrr_arbiter.get_stats()

    print("标准轮询仲裁器:")
    print(f"  各端口授权次数: {rr_stats['grants_per_port']}")
    print(f"  公平性指标: {rr_stats['fairness_index']:.3f}")

    print("加权轮询仲裁器 (权重[2,2,1,1]):")
    print(f"  各端口授权次数: {wrr_stats['grants_per_port']}")
    print(f"  公平性指标: {wrr_stats['fairness_index']:.3f}")
    print()


def test_fixed_priority_arbiter():
    """测试固定优先级仲裁器"""
    print("=== 测试固定优先级仲裁器 ===")

    # 设置优先级：端口0优先级最高(0)，端口3优先级最低(3)
    priorities = [0, 1, 2, 3]
    arbiter = FixedPriorityArbiter(4, priorities)

    print(f"端口优先级配置: {arbiter.get_priorities()}")

    # 测试用例
    test_cases = [
        [True, True, True, True],  # 所有端口有请求
        [False, True, True, True],  # 高优先级端口无请求
        [False, False, False, True],  # 只有最低优先级端口有请求
        [True, False, False, False],  # 只有最高优先级端口有请求
    ]

    print("测试序列:")
    for i, requests in enumerate(test_cases):
        granted_port = arbiter.arbitrate(requests)
        print(f"周期 {i}: 请求={requests}, 授权端口={granted_port}")

    stats = arbiter.get_stats()
    print(f"\n统计信息:")
    print(f"各端口授权次数: {stats['grants_per_port']}")
    print(f"公平性指标: {stats['fairness_index']:.3f}")
    print(f"最大饥饿计数: {stats['max_starvation']}")
    print()


def test_dynamic_priority_arbiter():
    """测试动态优先级仲裁器"""
    print("=== 测试动态优先级仲裁器 ===")

    # 使用相等的基础优先级
    arbiter = DynamicPriorityArbiter(4, base_priorities=[0, 0, 0, 0], aging_factor=1.0)

    # 创建不公平的请求模式：端口0频繁请求，端口3很少请求
    test_cycles = 20
    print("测试不公平请求模式下的动态调整:")

    for cycle in range(test_cycles):
        # 端口0总是请求，端口3只在某些周期请求
        requests = [True, False, False, cycle % 10 == 0]
        granted_port = arbiter.arbitrate(requests)

        if cycle % 5 == 0:  # 每5个周期显示一次状态
            dynamic_priorities = arbiter.get_dynamic_priorities()
            stats = arbiter.get_stats()
            starvation = stats.get('starvation_counter', [0] * 4)
            print(f"周期 {cycle}: 请求={requests}, 授权={granted_port}")
            print(f"  动态优先级: {[f'{p:.1f}' for p in dynamic_priorities]}")
            print(f"  饥饿计数: {starvation}")

    final_stats = arbiter.get_stats()
    print(f"\n最终统计:")
    print(f"各端口授权次数: {final_stats['grants_per_port']}")
    print(f"公平性指标: {final_stats['fairness_index']:.3f}")
    print()


def test_random_arbiter():
    """测试随机仲裁器"""
    print("=== 测试随机仲裁器 ===")

    # 使用固定种子确保测试可重现
    arbiter = RandomArbiter(4, seed=42)

    requests = [True, True, True, True]
    grants = []

    print("测试序列（固定种子=42）:")
    for i in range(10):
        granted_port = arbiter.arbitrate(requests)
        grants.append(granted_port)
        print(f"周期 {i}: 授权端口={granted_port}")

    print(f"授权序列: {grants}")

    stats = arbiter.get_stats()
    print(f"\n统计信息:")
    print(f"各端口授权次数: {stats['grants_per_port']}")
    print(f"公平性指标: {stats['fairness_index']:.3f}")

    # 测试随机性（重置种子应该产生相同序列）
    arbiter.set_seed(42)
    arbiter.reset()
    new_grants = []
    for i in range(10):
        granted_port = arbiter.arbitrate(requests)
        new_grants.append(granted_port)

    print(f"重置种子后序列: {new_grants}")
    print(f"序列一致性: {'相同' if grants == new_grants else '不同'}")
    print()


def test_token_bucket_arbiter():
    """测试令牌桶仲裁器"""
    print("=== 测试令牌桶仲裁器 ===")

    # 设置不同的端口速率
    port_rates = [2.0, 1.5, 1.0, 0.5]  # 端口0速率最高，端口3速率最低
    arbiter = TokenBucketArbiter(4, bucket_size=5, port_rates=port_rates)

    print(f"端口令牌补充速率: {port_rates}")
    print(f"令牌桶大小: 5")

    requests = [True, True, True, True]

    print("\n测试序列:")
    for i in range(15):
        tokens_before = arbiter.get_token_status()
        granted_port = arbiter.arbitrate(requests)
        tokens_after = arbiter.get_token_status()

        if i % 3 == 0:  # 每3个周期显示一次详细状态
            print(f"周期 {i}:")
            print(f"  授权前令牌: {[f'{t:.1f}' for t in tokens_before]}")
            print(f"  授权端口: {granted_port}")
            print(f"  授权后令牌: {[f'{t:.1f}' for t in tokens_after]}")

        # 添加小延迟模拟时间流逝
        time.sleep(0.01)

    stats = arbiter.get_stats()
    print(f"\n统计信息:")
    print(f"各端口授权次数: {stats['grants_per_port']}")
    print(f"期望比例（基于速率）: {[r/sum(port_rates) for r in port_rates]}")
    total_grants = sum(stats['grants_per_port'])
    if total_grants > 0:
        actual_ratios = [g/total_grants for g in stats['grants_per_port']]
        print(f"实际比例: {[f'{r:.3f}' for r in actual_ratios]}")
    print()


def test_factory_functions():
    """测试工厂函数"""
    print("=== 测试工厂函数 ===")

    # 使用工厂函数创建各种仲裁器
    rr_arbiter = create_round_robin_arbiter(3)
    wrr_arbiter = create_weighted_round_robin_arbiter(3, [2, 1, 1])
    fp_arbiter = create_fixed_priority_arbiter(3, [0, 1, 2])
    dp_arbiter = create_dynamic_priority_arbiter(3, aging_factor=0.5)
    rand_arbiter = create_random_arbiter(3, seed=123)
    tb_arbiter = create_token_bucket_arbiter(3, bucket_size=3)

    requests = [True, True, False]

    results = {
        "轮询": rr_arbiter.arbitrate(requests),
        "加权轮询": wrr_arbiter.arbitrate(requests),
        "固定优先级": fp_arbiter.arbitrate(requests),
        "动态优先级": dp_arbiter.arbitrate(requests),
        "随机": rand_arbiter.arbitrate(requests),
        "令牌桶": tb_arbiter.arbitrate(requests)
    }

    print("各种仲裁器的结果:")
    for name, result in results.items():
        print(f"{name}: {result}")
    print()


class TrafficPattern:
    """流量模式生成器"""

    @staticmethod
    def uniform_random(num_ports: int, request_rate: float) -> List[bool]:
        """均匀随机请求模式"""
        return [random.random() < request_rate for _ in range(num_ports)]

    @staticmethod
    def hotspot(num_ports: int, hotspot_port: int, hotspot_rate: float, normal_rate: float) -> List[bool]:
        """热点流量模式"""
        requests = [random.random() < normal_rate for _ in range(num_ports)]
        if hotspot_port < num_ports:
            requests[hotspot_port] = random.random() < hotspot_rate
        return requests

    @staticmethod
    def bursty(num_ports: int, burst_prob: float, burst_duration: int, current_cycle: int) -> List[bool]:
        """突发流量模式"""
        if current_cycle % burst_duration == 0:
            # 新的突发周期开始
            return [random.random() < burst_prob for _ in range(num_ports)]
        else:
            # 非突发期间，低负载
            return [random.random() < 0.1 for _ in range(num_ports)]

    @staticmethod
    def adversarial(num_ports: int, arbiter_type: str) -> List[bool]:
        """对抗性流量模式，针对特定仲裁器的最坏情况"""
        if arbiter_type == "round_robin":
            # 对轮询仲裁器：所有端口同时请求
            return [True] * num_ports
        elif arbiter_type == "weighted":
            # 对加权仲裁器：高权重端口不请求，低权重端口频繁请求
            requests = [False] * num_ports
            for i in range(num_ports // 2, num_ports):  # 假设后半部分是低权重端口
                requests[i] = True
            return requests
        else:
            return [True] * num_ports


def test_stress_scenarios():
    """压力测试场景"""
    print("=== 压力测试场景 ===")

    num_ports = 8
    test_cycles = 10000

    arbiters = {
        "轮询": RoundRobinArbiter(num_ports),
        "条件轮询": ConditionalRoundRobinArbiter(num_ports),
        "加权轮询": WeightedRoundRobinArbiter(num_ports, [4, 3, 2, 2, 1, 1, 1, 1]),
        "固定优先级": FixedPriorityArbiter(num_ports, list(range(num_ports))),
        "动态优先级": DynamicPriorityArbiter(num_ports, aging_factor=1.0),
        "随机": RandomArbiter(num_ports, seed=42)
    }

    stress_scenarios = [
        ("高负载均匀请求", lambda cycle: TrafficPattern.uniform_random(num_ports, 0.9)),
        ("热点流量", lambda cycle: TrafficPattern.hotspot(num_ports, 0, 0.95, 0.3)),
        ("突发流量", lambda cycle: TrafficPattern.bursty(num_ports, 0.8, 100, cycle)),
        ("满负载", lambda cycle: [True] * num_ports),
        ("对抗性流量", lambda cycle: TrafficPattern.adversarial(num_ports, "round_robin"))
    ]

    for scenario_name, pattern_generator in stress_scenarios:
        print(f"\n--- {scenario_name} ---")

        for arbiter_name, arbiter in arbiters.items():
            arbiter.reset()
            start_time = time.time()

            for cycle in range(test_cycles):
                requests = pattern_generator(cycle)
                arbiter.arbitrate(requests)

            execution_time = time.time() - start_time
            stats = arbiter.get_stats()

            print(f"{arbiter_name}:")
            print(f"  执行时间: {execution_time:.3f}秒")
            print(f"  平均处理时间: {execution_time/test_cycles*1000:.3f}毫秒/周期")
            print(f"  成功率: {stats['success_rate']:.3f}")
            print(f"  公平性指标: {stats['fairness_index']:.3f}")
            print(f"  最大饥饿计数: {stats['max_starvation']}")


def test_long_term_fairness():
    """长时间运行公平性测试"""
    print("\n=== 长时间运行公平性测试 ===")

    num_ports = 6
    test_cycles = 100000

    arbiters = {
        "轮询": RoundRobinArbiter(num_ports),
        "条件轮询": ConditionalRoundRobinArbiter(num_ports),
        "加权轮询": WeightedRoundRobinArbiter(num_ports, [3, 2, 2, 1, 1, 1]),
        "固定优先级": FixedPriorityArbiter(num_ports, list(range(num_ports))),
        "动态优先级": DynamicPriorityArbiter(num_ports, aging_factor=0.5),
        "随机": RandomArbiter(num_ports, seed=100)
    }

    print(f"运行{test_cycles}个周期的长时间测试...")

    for arbiter_name, arbiter in arbiters.items():
        print(f"\n--- {arbiter_name} ---")
        arbiter.reset()

        # 记录每1000周期的统计信息
        fairness_history = []
        grants_history = []

        for cycle in range(test_cycles):
            # 使用中等负载的随机请求模式
            requests = TrafficPattern.uniform_random(num_ports, 0.6)
            arbiter.arbitrate(requests)

            # 每1000周期记录一次统计信息
            if (cycle + 1) % 1000 == 0:
                stats = arbiter.get_stats()
                fairness_history.append(stats['fairness_index'])
                grants_history.append(stats['grants_per_port'].copy())

        # 最终统计
        final_stats = arbiter.get_stats()
        print(f"最终统计:")
        print(f"  总仲裁次数: {final_stats['total_arbitrations']}")
        print(f"  成功仲裁次数: {final_stats['successful_arbitrations']}")
        print(f"  最终公平性指标: {final_stats['fairness_index']:.6f}")
        print(f"  各端口授权次数: {final_stats['grants_per_port']}")
        print(f"  最大饥饿计数: {final_stats['max_starvation']}")
        print(f"  饥饿端口: {final_stats['starving_ports']}")

        # 公平性趋势分析
        if len(fairness_history) > 1:
            fairness_trend = statistics.mean(fairness_history[-10:]) - statistics.mean(fairness_history[:10])
            print(f"  公平性趋势: {fairness_trend:+.6f} (正值表示改善)")
            print(f"  公平性标准差: {statistics.stdev(fairness_history):.6f}")


def test_performance_benchmark():
    """性能基准测试"""
    print("\n=== 性能基准测试 ===")

    port_sizes = [4, 8, 16, 32, 64]
    cycles_per_test = 10000

    for num_ports in port_sizes:
        print(f"\n--- {num_ports}端口性能测试 ---")

        arbiters = {
            "轮询": RoundRobinArbiter(num_ports),
            "条件轮询": ConditionalRoundRobinArbiter(num_ports),
            "加权轮询": WeightedRoundRobinArbiter(num_ports, [i+1 for i in range(num_ports)]),
            "固定优先级": FixedPriorityArbiter(num_ports, list(range(num_ports))),
            "动态优先级": DynamicPriorityArbiter(num_ports, aging_factor=1.0),
            "随机": RandomArbiter(num_ports, seed=200)
        }

        for arbiter_name, arbiter in arbiters.items():
            # 预热
            for _ in range(100):
                requests = TrafficPattern.uniform_random(num_ports, 0.5)
                arbiter.arbitrate(requests)

            arbiter.reset()

            # 性能测试
            start_time = time.perf_counter()
            for cycle in range(cycles_per_test):
                requests = TrafficPattern.uniform_random(num_ports, 0.7)
                arbiter.arbitrate(requests)
            end_time = time.perf_counter()

            execution_time = end_time - start_time
            throughput = cycles_per_test / execution_time

            print(f"{arbiter_name}: {throughput:.0f} 仲裁/秒, {execution_time/cycles_per_test*1000000:.2f} 微秒/仲裁")


def test_starvation_analysis():
    """饥饿分析测试"""
    print("\n=== 饥饿分析测试 ===")

    num_ports = 8
    test_cycles = 50000

    arbiters = {
        "轮询": RoundRobinArbiter(num_ports),
        "条件轮询": ConditionalRoundRobinArbiter(num_ports),
        "加权轮询": WeightedRoundRobinArbiter(num_ports, [8, 4, 2, 1, 1, 1, 1, 1])
    }

    # 创建不公平的请求模式：部分端口请求频率很高
    def unfair_pattern(cycle: int) -> List[bool]:
        requests = [False] * num_ports
        # 端口0-2高频请求
        for i in range(3):
            requests[i] = random.random() < 0.8
        # 端口3-7低频请求
        for i in range(3, num_ports):
            requests[i] = random.random() < 0.2
        return requests

    for arbiter_name, arbiter in arbiters.items():
        print(f"\n--- {arbiter_name} 饥饿分析 ---")
        arbiter.reset()

        max_starvation_per_cycle = []
        starving_ports_count = []

        for cycle in range(test_cycles):
            requests = unfair_pattern(cycle)
            arbiter.arbitrate(requests)

            if cycle % 1000 == 0:
                stats = arbiter.get_stats()
                max_starvation_per_cycle.append(stats['max_starvation'])
                starving_ports_count.append(len(stats['starving_ports']))

        final_stats = arbiter.get_stats()
        print(f"最大饥饿计数: {final_stats['max_starvation']}")
        print(f"当前饥饿端口: {final_stats['starving_ports']}")
        print(f"各端口授权次数: {final_stats['grants_per_port']}")
        print(f"公平性指标: {final_stats['fairness_index']:.6f}")

        if max_starvation_per_cycle:
            print(f"平均最大饥饿计数: {statistics.mean(max_starvation_per_cycle):.1f}")
            print(f"平均饥饿端口数量: {statistics.mean(starving_ports_count):.1f}")


def test_burst_load_handling():
    """突发负载处理测试"""
    print("\n=== 突发负载处理测试 ===")

    num_ports = 6
    test_cycles = 20000

    arbiters = {
        "轮询": RoundRobinArbiter(num_ports),
        "条件轮询": ConditionalRoundRobinArbiter(num_ports),
        "加权轮询": WeightedRoundRobinArbiter(num_ports, [2, 2, 2, 1, 1, 1])
    }

    def burst_pattern(cycle: int) -> List[bool]:
        # 每500周期一个突发，突发持续50周期
        burst_period = 500
        burst_duration = 50

        if (cycle % burst_period) < burst_duration:
            # 突发期间：高负载
            return TrafficPattern.uniform_random(num_ports, 0.9)
        else:
            # 平静期间：低负载
            return TrafficPattern.uniform_random(num_ports, 0.2)

    for arbiter_name, arbiter in arbiters.items():
        print(f"\n--- {arbiter_name} 突发负载测试 ---")
        arbiter.reset()

        burst_fairness = []
        normal_fairness = []

        for cycle in range(test_cycles):
            requests = burst_pattern(cycle)
            arbiter.arbitrate(requests)

            # 分别记录突发期和平静期的公平性
            if cycle % 100 == 0:
                stats = arbiter.get_stats()
                if (cycle % 500) < 50:  # 突发期
                    burst_fairness.append(stats['fairness_index'])
                else:  # 平静期
                    normal_fairness.append(stats['fairness_index'])

        final_stats = arbiter.get_stats()
        print(f"最终统计:")
        print(f"  总成功率: {final_stats['success_rate']:.3f}")
        print(f"  最终公平性: {final_stats['fairness_index']:.6f}")
        print(f"  最大饥饿计数: {final_stats['max_starvation']}")

        if burst_fairness and normal_fairness:
            print(f"  突发期平均公平性: {statistics.mean(burst_fairness):.6f}")
            print(f"  平静期平均公平性: {statistics.mean(normal_fairness):.6f}")


def test_weighted_arbiter_stress():
    """加权仲裁器压力测试"""
    print("\n=== 加权仲裁器压力测试 ===")

    num_ports = 8
    test_cycles = 50000

    # 测试不同的权重配置
    weight_configs = [
        ("均匀权重", [1] * num_ports),
        ("递减权重", [8, 7, 6, 5, 4, 3, 2, 1]),
        ("极端权重", [10, 10, 1, 1, 1, 1, 1, 1]),
        ("随机权重", [random.randint(1, 5) for _ in range(num_ports)])
    ]

    for config_name, weights in weight_configs:
        print(f"\n--- {config_name}: {weights} ---")

        arbiter = WeightedRoundRobinArbiter(num_ports, weights)

        # 运行测试
        for cycle in range(test_cycles):
            requests = TrafficPattern.uniform_random(num_ports, 0.8)
            arbiter.arbitrate(requests)

        stats = arbiter.get_stats()

        # 计算实际权重比例
        total_grants = sum(stats['grants_per_port'])
        if total_grants > 0:
            actual_ratios = [g / total_grants for g in stats['grants_per_port']]
            expected_ratios = [w / sum(weights) for w in weights]

            print(f"期望比例: {[f'{r:.3f}' for r in expected_ratios]}")
            print(f"实际比例: {[f'{r:.3f}' for r in actual_ratios]}")

            # 计算比例误差
            ratio_errors = [abs(a - e) for a, e in zip(actual_ratios, expected_ratios)]
            print(f"最大比例误差: {max(ratio_errors):.6f}")
            print(f"平均比例误差: {statistics.mean(ratio_errors):.6f}")

        print(f"公平性指标: {stats['fairness_index']:.6f}")
        print(f"最大饥饿计数: {stats['max_starvation']}")


def run_comprehensive_tests():
    """运行综合测试套件"""
    print("NoC端口仲裁器综合测试套件")
    print("=" * 60)

    start_time = time.time()

    # 基本功能测试
    print("\n第一阶段：基本功能验证")
    test_round_robin_arbiter()
    test_conditional_round_robin_arbiter()
    test_weighted_round_robin_arbiter()
    test_fixed_priority_arbiter()
    test_dynamic_priority_arbiter()
    test_random_arbiter()
    test_token_bucket_arbiter()
    test_fairness_comparison()
    test_factory_functions()

    # 压力测试
    print("\n第二阶段：压力和性能测试")
    test_stress_scenarios()
    test_performance_benchmark()

    # 长时间测试
    print("\n第三阶段：长时间运行测试")
    test_long_term_fairness()
    test_starvation_analysis()

    # 特殊场景测试
    print("\n第四阶段：特殊场景测试")
    test_burst_load_handling()
    test_weighted_arbiter_stress()

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"所有测试完成! 总耗时: {total_time:.2f}秒")


def run_quick_tests():
    """运行快速基本测试"""
    print("NoC端口仲裁器基本测试")
    print("=" * 50)

    test_round_robin_arbiter()
    test_conditional_round_robin_arbiter()
    test_weighted_round_robin_arbiter()
    test_fixed_priority_arbiter()
    test_dynamic_priority_arbiter()
    test_random_arbiter()
    test_token_bucket_arbiter()
    test_fairness_comparison()
    test_factory_functions()

    print("基本测试完成!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='仲裁器测试套件')
    parser.add_argument('--comprehensive', '-c', action='store_true',
                       help='运行综合测试套件（包含长时间测试）')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='只运行快速基本测试')

    args = parser.parse_args()

    if args.comprehensive:
        run_comprehensive_tests()
    elif args.quick:
        run_quick_tests()
    else:
        # 默认运行综合测试
        print("运行综合测试套件（使用 --quick 或 -q 参数只运行基本测试）")
        run_comprehensive_tests()
