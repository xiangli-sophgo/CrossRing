"""
测试最大权重匹配仲裁器的功能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.arbitration import MaxWeightMatchingArbiter, create_max_weight_matching_arbiter


def test_basic_islip_matching():
    """测试基本的iSLIP匹配功能"""
    print("测试基本iSLIP匹配...")

    arbiter = MaxWeightMatchingArbiter(algorithm="islip", iterations=1)

    # 简单的3x3请求矩阵
    request_matrix = [
        [True, False, True],   # 输入0请求输出0,2
        [False, True, True],   # 输入1请求输出1,2
        [True, True, False]    # 输入2请求输出0,1
    ]

    matches = arbiter.match(request_matrix)
    print(f"匹配结果: {matches}")

    # 验证匹配结果的有效性
    assert len(matches) <= 3, "匹配数量不应超过输入或输出数量"

    used_inputs = set()
    used_outputs = set()
    for input_idx, output_idx in matches:
        assert input_idx not in used_inputs, "输入端口不应重复匹配"
        assert output_idx not in used_outputs, "输出端口不应重复匹配"
        assert request_matrix[input_idx][output_idx], "匹配应该基于有效请求"
        used_inputs.add(input_idx)
        used_outputs.add(output_idx)

    print("√ 基本iSLIP匹配测试通过")


def test_weight_strategies():
    """测试不同的权重策略"""
    print("测试权重策略...")

    arbiter = MaxWeightMatchingArbiter(algorithm="lqf", weight_strategy="queue_length")

    # 4x4请求矩阵
    request_matrix = [
        [True, True, False, False],
        [False, True, True, False],
        [True, False, True, True],
        [False, False, False, True]
    ]

    # 更新队列信息
    queue_id = "test_queue"
    arbiter._init_or_update_state(queue_id, 4, 4)

    # 设置不同的队列长度
    arbiter.update_queue_info(queue_id, 0, 0, queue_length=10)  # 高优先级
    arbiter.update_queue_info(queue_id, 0, 1, queue_length=5)
    arbiter.update_queue_info(queue_id, 1, 1, queue_length=8)
    arbiter.update_queue_info(queue_id, 1, 2, queue_length=3)
    arbiter.update_queue_info(queue_id, 2, 0, queue_length=7)
    arbiter.update_queue_info(queue_id, 2, 2, queue_length=12)  # 最高优先级
    arbiter.update_queue_info(queue_id, 2, 3, queue_length=4)
    arbiter.update_queue_info(queue_id, 3, 3, queue_length=6)

    matches = arbiter.match(request_matrix, queue_id=queue_id)
    print(f"LQF匹配结果: {matches}")

    # 验证权重最高的请求被优先匹配
    assert (2, 2) in matches, "最长队列应该被优先匹配"

    print("√ 权重策略测试通过")


def test_multiple_iterations():
    """测试多轮迭代匹配"""
    print("测试多轮迭代匹配...")

    # 单轮匹配
    arbiter_single = MaxWeightMatchingArbiter(algorithm="islip", iterations=1)

    # 多轮匹配
    arbiter_multi = MaxWeightMatchingArbiter(algorithm="islip", iterations=3)

    # 复杂的5x5请求矩阵，存在多种可能的匹配方案
    request_matrix = [
        [True, True, True, False, False],
        [True, True, False, True, False],
        [False, True, True, True, True],
        [True, False, True, True, False],
        [False, True, False, True, True]
    ]

    matches_single = arbiter_single.match(request_matrix)
    matches_multi = arbiter_multi.match(request_matrix)

    print(f"单轮匹配: {matches_single} (数量: {len(matches_single)})")
    print(f"多轮匹配: {matches_multi} (数量: {len(matches_multi)})")

    # 多轮匹配通常能找到更多的匹配
    assert len(matches_multi) >= len(matches_single), "多轮匹配应该找到至少相同数量的匹配"

    print("√ 多轮迭代测试通过")


def test_different_algorithms():
    """测试不同的匹配算法"""
    print("测试不同匹配算法...")

    algorithms = ["islip", "lqf", "ocf", "pim"]

    request_matrix = [
        [True, False, True, True],
        [True, True, False, False],
        [False, True, True, True],
        [True, True, False, True]
    ]

    for algorithm in algorithms:
        arbiter = MaxWeightMatchingArbiter(algorithm=algorithm, iterations=2)
        matches = arbiter.match(request_matrix)
        print(f"{algorithm.upper()}算法匹配结果: {matches}")

        # 验证每个算法都能产生有效匹配
        assert len(matches) > 0, f"{algorithm}算法应该产生至少一个匹配"

        # 验证匹配的有效性
        used_inputs = set()
        used_outputs = set()
        for input_idx, output_idx in matches:
            assert input_idx not in used_inputs, f"{algorithm}: 输入端口重复"
            assert output_idx not in used_outputs, f"{algorithm}: 输出端口重复"
            assert request_matrix[input_idx][output_idx], f"{algorithm}: 无效请求匹配"
            used_inputs.add(input_idx)
            used_outputs.add(output_idx)

    print("√ 不同算法测试通过")


def test_fairness_over_time():
    """测试长期公平性"""
    print("测试长期公平性...")

    arbiter = MaxWeightMatchingArbiter(algorithm="islip", iterations=1)

    # 固定的请求模式
    request_matrix = [
        [True, True, False],
        [False, True, True],
        [True, False, True]
    ]

    input_grants = [0, 0, 0]
    output_grants = [0, 0, 0]

    # 运行100轮匹配
    for round_num in range(100):
        matches = arbiter.match(request_matrix)
        for input_idx, output_idx in matches:
            input_grants[input_idx] += 1
            output_grants[output_idx] += 1

    print(f"输入端口授权次数: {input_grants}")
    print(f"输出端口授权次数: {output_grants}")

    # 检查公平性（每个端口的授权次数应该相对均匀）
    max_input_grants = max(input_grants)
    min_input_grants = min(input_grants)
    max_output_grants = max(output_grants)
    min_output_grants = min(output_grants)

    assert max_input_grants - min_input_grants <= 20, "输入端口授权应该相对公平"
    assert max_output_grants - min_output_grants <= 20, "输出端口授权应该相对公平"

    print("√ 长期公平性测试通过")


def test_statistics():
    """测试统计信息功能"""
    print("测试统计信息...")

    arbiter = MaxWeightMatchingArbiter(algorithm="islip")

    request_matrix = [
        [True, False],
        [False, True]
    ]

    # 执行几轮匹配
    for _ in range(10):
        arbiter.match(request_matrix)

    stats = arbiter.get_match_stats("default")
    print(f"统计信息: {stats}")

    assert stats['total_matches'] == 10, "总匹配次数应该为10"
    assert stats['successful_matches'] > 0, "应该有成功的匹配"
    assert stats['average_matching_size'] > 0, "平均匹配大小应该大于0"

    print("√ 统计信息测试通过")


def test_factory_functions():
    """测试工厂函数"""
    print("测试工厂函数...")

    # 测试便捷创建函数
    arbiter1 = create_max_weight_matching_arbiter("islip", 2, "queue_length")
    assert arbiter1.algorithm == "islip"
    assert arbiter1.iterations == 2
    assert arbiter1.weight_strategy == "queue_length"

    # 测试配置创建函数
    from src.utils.arbitration import create_arbiter_from_config

    config = {
        "type": "islip",
        "iterations": 3,
        "weight_strategy": "hybrid"
    }

    arbiter2 = create_arbiter_from_config(config)
    assert isinstance(arbiter2, MaxWeightMatchingArbiter)
    assert arbiter2.algorithm == "islip"
    assert arbiter2.iterations == 3
    assert arbiter2.weight_strategy == "hybrid"

    print("√ 工厂函数测试通过")


def test_edge_cases():
    """测试边界情况"""
    print("测试边界情况...")

    arbiter = MaxWeightMatchingArbiter()

    # 空请求矩阵
    matches = arbiter.match([])
    assert matches == [], "空矩阵应该返回空匹配"

    # 全False请求矩阵
    request_matrix = [
        [False, False],
        [False, False]
    ]
    matches = arbiter.match(request_matrix)
    assert matches == [], "无请求矩阵应该返回空匹配"

    # 单个请求
    request_matrix = [
        [True, False],
        [False, False]
    ]
    matches = arbiter.match(request_matrix)
    assert matches == [(0, 0)], "单请求应该被成功匹配"

    print("√ 边界情况测试通过")


if __name__ == "__main__":
    print("开始测试最大权重匹配仲裁器...")
    print("=" * 50)

    test_basic_islip_matching()
    print()

    test_weight_strategies()
    print()

    test_multiple_iterations()
    print()

    test_different_algorithms()
    print()

    test_fairness_over_time()
    print()

    test_statistics()
    print()

    test_factory_functions()
    print()

    test_edge_cases()
    print()

    print("=" * 50)
    print("√ 所有测试通过！最大权重匹配仲裁器实现正确。")