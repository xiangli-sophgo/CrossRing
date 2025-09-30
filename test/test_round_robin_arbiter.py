"""
测试RoundRobinArbiter的功能和性能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.arbitration import RoundRobinArbiter, MaxWeightMatchingArbiter
import random


def test_basic_functionality():
    """测试基本功能：验证匹配有效性"""
    print("=" * 60)
    print("测试1: 基本功能测试")
    print("=" * 60)

    arbiter = RoundRobinArbiter()

    # 测试用例1: 简单的2x2矩阵
    print("\n[测试1.1] 2x2请求矩阵")
    request_matrix = [
        [True, True],
        [True, True]
    ]
    matches = arbiter.match(request_matrix, queue_id="test1.1")
    print(f"请求矩阵:\n{request_matrix}")
    print(f"匹配结果: {matches}")

    # 验证匹配有效性
    assert len(matches) == 2, f"应该有2个匹配，实际{len(matches)}"

    used_inputs = set()
    used_outputs = set()
    for input_idx, output_idx in matches:
        assert input_idx not in used_inputs, f"输入{input_idx}重复匹配"
        assert output_idx not in used_outputs, f"输出{output_idx}重复匹配"
        assert request_matrix[input_idx][output_idx], f"无效匹配({input_idx}, {output_idx})"
        used_inputs.add(input_idx)
        used_outputs.add(output_idx)

    print("[OK] 2x2测试通过")

    # 测试用例2: 3x3矩阵，部分请求
    print("\n[测试1.2] 3x3请求矩阵（部分请求）")
    request_matrix = [
        [True, False, True],
        [False, True, True],
        [True, True, False]
    ]
    matches = arbiter.match(request_matrix, queue_id="test1.2")
    print(f"请求矩阵:")
    for row in request_matrix:
        print(f"  {row}")
    print(f"匹配结果: {matches}")

    # 验证
    used_inputs = set()
    used_outputs = set()
    for input_idx, output_idx in matches:
        assert input_idx not in used_inputs, f"输入{input_idx}重复匹配"
        assert output_idx not in used_outputs, f"输出{output_idx}重复匹配"
        assert request_matrix[input_idx][output_idx], f"无效匹配({input_idx}, {output_idx})"
        used_inputs.add(input_idx)
        used_outputs.add(output_idx)

    print(f"[OK] 3x3测试通过，匹配数量: {len(matches)}")


def test_pointer_advancement():
    """测试指针推进机制"""
    print("\n" + "=" * 60)
    print("测试2: 指针推进测试")
    print("=" * 60)

    arbiter = RoundRobinArbiter()

    # 固定的请求模式
    request_matrix = [
        [True, True, True],
        [True, True, True],
        [True, True, True]
    ]

    print("\n所有输入请求所有输出，连续5轮匹配：")
    for round_num in range(5):
        matches = arbiter.match(request_matrix, queue_id="pointer_test")
        pointers = arbiter.get_pointers(queue_id="pointer_test")

        print(f"\n第{round_num + 1}轮:")
        print(f"  匹配结果: {matches}")
        print(f"  input_start: {pointers['input_start']}")
        print(f"  input_pointers: {pointers['input_pointers']}")
        print(f"  output_pointers: {pointers['output_pointers']}")

    print("\n[观察] 检查指针推进规律")


def test_fairness():
    """测试长期公平性"""
    print("\n" + "=" * 60)
    print("测试3: 公平性测试（100轮）")
    print("=" * 60)

    arbiter = RoundRobinArbiter()

    # 固定的请求模式：所有输入都可以匹配所有输出
    request_matrix = [
        [True, True, True],
        [True, True, True],
        [True, True, True]
    ]

    input_match_count = [0, 0, 0]
    output_match_count = [0, 0, 0]

    # 运行100轮
    for _ in range(100):
        matches = arbiter.match(request_matrix, queue_id="fairness_test")
        for input_idx, output_idx in matches:
            input_match_count[input_idx] += 1
            output_match_count[output_idx] += 1

    print(f"\n输入端口匹配次数: {input_match_count}")
    print(f"输出端口匹配次数: {output_match_count}")

    # 计算公平性指标
    total_matches = sum(input_match_count)
    avg_per_port = total_matches / 3
    max_deviation_input = max(abs(count - avg_per_port) for count in input_match_count)
    max_deviation_output = max(abs(count - avg_per_port) for count in output_match_count)

    print(f"\n平均每端口匹配: {avg_per_port:.1f}")
    print(f"输入端口最大偏差: {max_deviation_input:.1f}")
    print(f"输出端口最大偏差: {max_deviation_output:.1f}")

    # 公平性判断（偏差不应超过20%）
    if max_deviation_input <= avg_per_port * 0.2 and max_deviation_output <= avg_per_port * 0.2:
        print("[OK] 公平性测试通过")
    else:
        print("[WARNING] 公平性存在问题，偏差较大")


def test_throughput_comparison():
    """测试吞吐量：对比不同算法"""
    print("\n" + "=" * 60)
    print("测试4: 吞吐量对比测试")
    print("=" * 60)

    # 测试场景：4x4满负载
    request_matrix = [
        [True, True, True, True],
        [True, True, True, True],
        [True, True, True, True],
        [True, True, True, True]
    ]

    # RoundRobinArbiter
    rr_arbiter = RoundRobinArbiter()
    rr_total = 0
    for _ in range(100):
        matches = rr_arbiter.match(request_matrix, queue_id="rr_throughput")
        rr_total += len(matches)
    rr_avg = rr_total / 100

    # iSLIP (iterations=1)
    islip_arbiter = MaxWeightMatchingArbiter(algorithm="islip", iterations=1)
    islip_total = 0
    for _ in range(100):
        matches = islip_arbiter.match(request_matrix, queue_id="islip_throughput")
        islip_total += len(matches)
    islip_avg = islip_total / 100

    # iSLIP (iterations=2)
    islip2_arbiter = MaxWeightMatchingArbiter(algorithm="islip", iterations=2)
    islip2_total = 0
    for _ in range(100):
        matches = islip2_arbiter.match(request_matrix, queue_id="islip2_throughput")
        islip2_total += len(matches)
    islip2_avg = islip2_total / 100

    print(f"\n4x4满负载，100轮平均匹配数:")
    print(f"  RoundRobin:      {rr_avg:.2f} / 4 ({rr_avg/4*100:.1f}%)")
    print(f"  iSLIP(iter=1):   {islip_avg:.2f} / 4 ({islip_avg/4*100:.1f}%)")
    print(f"  iSLIP(iter=2):   {islip2_avg:.2f} / 4 ({islip2_avg/4*100:.1f}%)")

    # 分析
    if rr_avg >= 3.9:
        print("[OK] RoundRobin吞吐量正常")
    else:
        print(f"[WARNING] RoundRobin吞吐量较低：{rr_avg:.2f} < 3.9")


def test_hotspot_contention():
    """测试热点冲突：多输入竞争少数输出"""
    print("\n" + "=" * 60)
    print("测试5: 热点冲突测试")
    print("=" * 60)

    arbiter = RoundRobinArbiter()

    # 场景：3个输入都想要输出0和输出1，输出2无人问津
    request_matrix = [
        [True, True, False],
        [True, True, False],
        [True, True, False]
    ]

    print("\n请求模式：3个输入竞争2个输出")
    print("请求矩阵:")
    for i, row in enumerate(request_matrix):
        print(f"  Input{i}: {row}")

    input_match_count = [0, 0, 0]
    output_match_count = [0, 0, 0]

    # 运行100轮
    for _ in range(100):
        matches = arbiter.match(request_matrix, queue_id="hotspot_test")
        for input_idx, output_idx in matches:
            input_match_count[input_idx] += 1
            output_match_count[output_idx] += 1

    print(f"\n100轮后统计:")
    print(f"  输入匹配次数: {input_match_count}")
    print(f"  输出匹配次数: {output_match_count}")

    # 检查是否有输入被饿死
    min_input_matches = min(input_match_count)
    max_input_matches = max(input_match_count)

    if min_input_matches == 0:
        print("[ERROR] 存在输入饥饿现象！")
    elif max_input_matches - min_input_matches > 20:
        print(f"[WARNING] 输入间不公平，差距: {max_input_matches - min_input_matches}")
    else:
        print("[OK] 热点冲突处理正常")


def test_idle_inputs():
    """测试空闲输入：部分输入无请求"""
    print("\n" + "=" * 60)
    print("测试6: 空闲输入测试")
    print("=" * 60)

    arbiter = RoundRobinArbiter()

    # 场景：输入1无任何请求
    request_matrix = [
        [True, True, True],
        [False, False, False],  # 输入1空闲
        [True, True, True]
    ]

    print("\n输入1无请求，连续5轮:")
    for round_num in range(5):
        matches = arbiter.match(request_matrix, queue_id="idle_test")
        pointers = arbiter.get_pointers(queue_id="idle_test")

        print(f"\n第{round_num + 1}轮:")
        print(f"  匹配结果: {matches}")
        print(f"  input_start: {pointers['input_start']}")
        print(f"  input_pointers: {pointers['input_pointers']}")

    print("\n[OK] 空闲输入测试完成")


def test_dynamic_requests():
    """测试动态请求模式"""
    print("\n" + "=" * 60)
    print("测试7: 动态请求模式测试")
    print("=" * 60)

    arbiter = RoundRobinArbiter()

    # 场景：请求模式每轮变化
    patterns = [
        # 模式1：对角线
        [
            [True, False, False],
            [False, True, False],
            [False, False, True]
        ],
        # 模式2：反对角线
        [
            [False, False, True],
            [False, True, False],
            [True, False, False]
        ],
        # 模式3：全请求
        [
            [True, True, True],
            [True, True, True],
            [True, True, True]
        ],
        # 模式4：稀疏请求
        [
            [True, False, True],
            [False, True, False],
            [True, False, True]
        ]
    ]

    print("\n4种不同请求模式，各运行25轮:")
    total_matches = 0

    for pattern_idx, pattern in enumerate(patterns):
        pattern_matches = 0
        for _ in range(25):
            matches = arbiter.match(pattern, queue_id="dynamic_test")
            pattern_matches += len(matches)

        avg_matches = pattern_matches / 25
        total_matches += pattern_matches
        print(f"  模式{pattern_idx + 1}: 平均 {avg_matches:.2f} 匹配/轮")

    overall_avg = total_matches / 100
    print(f"\n总体平均: {overall_avg:.2f} 匹配/轮")
    print("[OK] 动态请求测试完成")


def test_pointer_usage():
    """测试output_pointers是否被使用"""
    print("\n" + "=" * 60)
    print("测试8: 指针使用分析")
    print("=" * 60)

    arbiter = RoundRobinArbiter()

    request_matrix = [
        [True, True, True],
        [True, True, True],
        [True, True, True]
    ]

    print("\n观察output_pointers是否影响匹配结果：")
    print("\n初始状态（所有指针为0）")
    matches1 = arbiter.match(request_matrix, queue_id="pointer_usage_test")
    pointers1 = arbiter.get_pointers(queue_id="pointer_usage_test")
    print(f"  匹配: {matches1}")
    print(f"  output_pointers: {pointers1['output_pointers']}")

    # 再运行几轮
    for i in range(2, 6):
        matches = arbiter.match(request_matrix, queue_id="pointer_usage_test")
        pointers = arbiter.get_pointers(queue_id="pointer_usage_test")
        print(f"\n第{i}轮:")
        print(f"  匹配: {matches}")
        print(f"  output_pointers: {pointers['output_pointers']}")

    print("\n[观察] output_pointers在增长，但是否影响了匹配决策？")


def run_all_tests():
    """运行所有测试"""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + " " * 15 + "RoundRobinArbiter 全面测试" + " " * 15 + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)

    try:
        test_basic_functionality()
        test_pointer_advancement()
        test_fairness()
        test_throughput_comparison()
        test_hotspot_contention()
        test_idle_inputs()
        test_dynamic_requests()
        test_pointer_usage()

        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n[ERROR] 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n[ERROR] 发生异常: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()