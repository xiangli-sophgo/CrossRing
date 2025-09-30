"""
使用更接近实际NoC场景的测试用例
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.arbitration import RoundRobinArbiter, MaxWeightMatchingArbiter
import random


def test_realistic_noc_traffic():
    """模拟真实的NoC流量模式"""
    print("=" * 60)
    print("真实NoC流量模拟测试")
    print("=" * 60)

    # 场景：4个IP类型向5个方向注入（类似IQ仲裁）
    # IP类型：gdma_0, gdma_1, sdma_0, sdma_1
    # 方向：TL, TR, TU, TD, EQ
    # 流量特点：不均匀、有突发、有周期性

    rr_arbiter = RoundRobinArbiter()
    islip_arbiter = MaxWeightMatchingArbiter(algorithm="islip", iterations=2)

    # 生成1000轮真实流量
    random.seed(42)  # 固定种子确保可重现

    rr_total_matches = 0
    islip_total_matches = 0

    # 流量模式1：突发流量（前500轮）
    print("\n[场景1] 突发流量模式（500轮）")
    for cycle in range(500):
        # 模拟突发：某些cycle流量很大，某些很小
        if cycle % 10 < 7:  # 70%的时间有流量
            # 不均匀的请求矩阵
            request_matrix = [
                [random.random() < 0.8, random.random() < 0.6, random.random() < 0.4,
                 random.random() < 0.7, random.random() < 0.3],
                [random.random() < 0.6, random.random() < 0.8, random.random() < 0.5,
                 random.random() < 0.4, random.random() < 0.6],
                [random.random() < 0.5, random.random() < 0.5, random.random() < 0.8,
                 random.random() < 0.6, random.random() < 0.4],
                [random.random() < 0.4, random.random() < 0.6, random.random() < 0.6,
                 random.random() < 0.8, random.random() < 0.5]
            ]
        else:  # 30%的时间低流量
            request_matrix = [
                [random.random() < 0.2, random.random() < 0.1, random.random() < 0.1,
                 random.random() < 0.2, random.random() < 0.1],
                [random.random() < 0.1, random.random() < 0.2, random.random() < 0.1,
                 random.random() < 0.1, random.random() < 0.1],
                [random.random() < 0.1, random.random() < 0.1, random.random() < 0.2,
                 random.random() < 0.1, random.random() < 0.1],
                [random.random() < 0.1, random.random() < 0.1, random.random() < 0.1,
                 random.random() < 0.2, random.random() < 0.1]
            ]

        rr_matches = rr_arbiter.match(request_matrix, queue_id="realistic_rr")
        islip_matches = islip_arbiter.match(request_matrix, queue_id="realistic_islip")

        rr_total_matches += len(rr_matches)
        islip_total_matches += len(islip_matches)

    print(f"  RoundRobin总匹配数: {rr_total_matches}")
    print(f"  iSLIP总匹配数:      {islip_total_matches}")
    print(f"  RoundRobin平均:     {rr_total_matches/500:.2f}/轮")
    print(f"  iSLIP平均:          {islip_total_matches/500:.2f}/轮")

    # 流量模式2：周期性流量（后500轮）
    print("\n[场景2] 周期性流量模式（500轮）")
    rr_total_matches_2 = 0
    islip_total_matches_2 = 0

    for cycle in range(500):
        # 4周期的流量模式
        phase = cycle % 4

        if phase == 0:
            # 阶段0：IP0和IP1活跃
            request_matrix = [
                [True, True, True, False, False],
                [True, True, False, True, False],
                [False, False, False, False, False],
                [False, False, False, False, False]
            ]
        elif phase == 1:
            # 阶段1：IP2和IP3活跃
            request_matrix = [
                [False, False, False, False, False],
                [False, False, False, False, False],
                [True, False, True, True, False],
                [False, True, True, True, False]
            ]
        elif phase == 2:
            # 阶段2：所有IP活跃，竞争激烈
            request_matrix = [
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True]
            ]
        else:
            # 阶段3：稀疏流量
            request_matrix = [
                [True, False, False, False, False],
                [False, True, False, False, False],
                [False, False, True, False, False],
                [False, False, False, True, False]
            ]

        rr_matches = rr_arbiter.match(request_matrix, queue_id="realistic_rr_2")
        islip_matches = islip_arbiter.match(request_matrix, queue_id="realistic_islip_2")

        rr_total_matches_2 += len(rr_matches)
        islip_total_matches_2 += len(islip_matches)

    print(f"  RoundRobin总匹配数: {rr_total_matches_2}")
    print(f"  iSLIP总匹配数:      {islip_total_matches_2}")
    print(f"  RoundRobin平均:     {rr_total_matches_2/500:.2f}/轮")
    print(f"  iSLIP平均:          {islip_total_matches_2/500:.2f}/轮")

    # 总体对比
    print("\n[总体对比] 1000轮")
    rr_total = rr_total_matches + rr_total_matches_2
    islip_total = islip_total_matches + islip_total_matches_2
    print(f"  RoundRobin: {rr_total} 匹配 ({rr_total/1000:.2f}/轮)")
    print(f"  iSLIP:      {islip_total} 匹配 ({islip_total/1000:.2f}/轮)")

    diff = rr_total - islip_total
    if abs(diff) <= 50:
        print(f"\n[结论] 性能相当（差异{diff}，仅{abs(diff)/1000*100:.1f}%）")
    elif diff > 0:
        print(f"\n[结论] RoundRobin更优（多{diff}次匹配，+{diff/1000*100:.1f}%）")
    else:
        print(f"\n[结论] iSLIP更优（多{-diff}次匹配，+{-diff/1000*100:.1f}%）")


def test_pathological_case():
    """测试病态场景：可能暴露RoundRobin的弱点"""
    print("\n" + "=" * 60)
    print("病态场景测试")
    print("=" * 60)

    rr_arbiter = RoundRobinArbiter()
    islip_arbiter = MaxWeightMatchingArbiter(algorithm="islip", iterations=2)

    # 场景：所有输入都想要同一个输出（输出0），但也有备选
    # 这是最容易产生冲突的模式
    print("\n[场景] 所有输入优先请求输出0，有备选")
    request_matrix = [
        [True, True, False, False],
        [True, False, True, False],
        [True, False, False, True],
        [True, True, True, True]
    ]

    print("请求矩阵：")
    for i, row in enumerate(request_matrix):
        print(f"  Input{i}: {row}")

    rr_total = 0
    islip_total = 0

    # 运行100轮
    for _ in range(100):
        rr_matches = rr_arbiter.match(request_matrix, queue_id="pathological_rr")
        islip_matches = islip_arbiter.match(request_matrix, queue_id="pathological_islip")

        rr_total += len(rr_matches)
        islip_total += len(islip_matches)

    print(f"\n100轮结果：")
    print(f"  RoundRobin: {rr_total} 匹配 ({rr_total/100:.2f}/轮)")
    print(f"  iSLIP:      {islip_total} 匹配 ({islip_total/100:.2f}/轮)")

    if rr_total >= islip_total:
        print("[OK] RoundRobin在病态场景下表现正常")
    else:
        print(f"[WARNING] RoundRobin在病态场景下较弱（差{islip_total - rr_total}）")


if __name__ == "__main__":
    test_realistic_noc_traffic()
    test_pathological_case()
    print("\n" + "=" * 60)
    print("真实场景测试完成")
    print("=" * 60)