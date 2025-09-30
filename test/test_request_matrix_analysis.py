"""
分析实际请求矩阵的特征
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.arbitration import RoundRobinArbiter, MaxWeightMatchingArbiter


def analyze_request_patterns():
    """分析不同类型的请求矩阵模式"""
    print("=" * 60)
    print("请求矩阵模式分析")
    print("=" * 60)

    # 模式1：稀疏矩阵（每个输入最多一个输出）
    print("\n[模式1] 稀疏矩阵 - 每个IP只有一个目的地")
    sparse_matrix = [
        [False, True, False, False, False],  # gdma_0 → TL
        [True, False, False, False, False],  # gdma_1 → TR
        [False, False, False, False, True],  # sdma_0 → EQ
        [False, False, True, False, False],  # sdma_1 → TU
    ]

    print("请求矩阵:")
    print_matrix(sparse_matrix, ["gdma_0", "gdma_1", "sdma_0", "sdma_1"],
                 ["TR", "TL", "TU", "TD", "EQ"])

    # 统计
    analyze_matrix_stats(sparse_matrix, "稀疏矩阵")

    # 测试RoundRobin
    rr = RoundRobinArbiter()
    matches = rr.match(sparse_matrix, queue_id="sparse")
    print(f"RoundRobin匹配: {matches}")
    print(f"匹配数: {len(matches)}/4 ({len(matches)/4*100:.0f}%)\n")

    # 模式2：热点冲突（多个输入竞争少数输出）
    print("\n[模式2] 热点冲突 - 3个IP都要去TR方向")
    hotspot_matrix = [
        [True, False, False, False, False],   # gdma_0 → TR
        [True, False, False, False, False],   # gdma_1 → TR
        [True, False, False, False, False],   # sdma_0 → TR
        [False, True, False, False, False],   # sdma_1 → TL
    ]

    print("请求矩阵:")
    print_matrix(hotspot_matrix, ["gdma_0", "gdma_1", "sdma_0", "sdma_1"],
                 ["TR", "TL", "TU", "TD", "EQ"])

    analyze_matrix_stats(hotspot_matrix, "热点冲突")

    rr2 = RoundRobinArbiter()
    islip = MaxWeightMatchingArbiter(algorithm="islip", iterations=2)

    print("\n连续10轮观察:")
    for i in range(10):
        rr_matches = rr2.match(hotspot_matrix, queue_id="hotspot_rr")
        islip_matches = islip.match(hotspot_matrix, queue_id="hotspot_islip")
        print(f"轮{i+1}: RR={rr_matches}, iSLIP={islip_matches}")

    # 模式3：完全均衡（每个方向都有人要）
    print("\n\n[模式3] 完全均衡 - 每个方向恰好一个IP")
    balanced_matrix = [
        [True, False, False, False, False],   # gdma_0 → TR
        [False, True, False, False, False],   # gdma_1 → TL
        [False, False, True, False, False],   # sdma_0 → TU
        [False, False, False, True, False],   # sdma_1 → TD
    ]

    print("请求矩阵:")
    print_matrix(balanced_matrix, ["gdma_0", "gdma_1", "sdma_0", "sdma_1"],
                 ["TR", "TL", "TU", "TD", "EQ"])

    analyze_matrix_stats(balanced_matrix, "完全均衡")

    rr3 = RoundRobinArbiter()
    matches = rr3.match(balanced_matrix, queue_id="balanced")
    print(f"RoundRobin匹配: {matches}")
    print(f"匹配数: {len(matches)}/4 (100%)\n")

    # 模式4：极端冲突（所有IP都要同一方向）
    print("\n[模式4] 极端冲突 - 所有IP都要去TR")
    extreme_matrix = [
        [True, False, False, False, False],   # gdma_0 → TR
        [True, False, False, False, False],   # gdma_1 → TR
        [True, False, False, False, False],   # sdma_0 → TR
        [True, False, False, False, False],   # sdma_1 → TR
    ]

    print("请求矩阵:")
    print_matrix(extreme_matrix, ["gdma_0", "gdma_1", "sdma_0", "sdma_1"],
                 ["TR", "TL", "TU", "TD", "EQ"])

    analyze_matrix_stats(extreme_matrix, "极端冲突")

    print("\n100轮统计:")
    rr4 = RoundRobinArbiter()
    ip_wins = [0, 0, 0, 0]

    for _ in range(100):
        matches = rr4.match(extreme_matrix, queue_id="extreme")
        for ip_idx, _ in matches:
            ip_wins[ip_idx] += 1

    print(f"每个IP获胜次数: {ip_wins}")
    print(f"公平性: {'完美' if max(ip_wins) - min(ip_wins) <= 1 else '不均'}\n")


def print_matrix(matrix, row_labels, col_labels):
    """打印矩阵"""
    # 表头
    print(f"{'IP类型':<10} ", end="")
    for col in col_labels:
        print(f"{col:>6}", end="")
    print()
    print("-" * 50)

    # 数据行
    for i, row in enumerate(matrix):
        print(f"{row_labels[i]:<10} ", end="")
        for val in row:
            print(f"{'Y' if val else '.':>6}", end="")
        print()


def analyze_matrix_stats(matrix, name):
    """分析矩阵统计特征"""
    num_inputs = len(matrix)
    num_outputs = len(matrix[0]) if matrix else 0

    # 每个输入的请求数
    input_requests = [sum(row) for row in matrix]

    # 每个输出被多少输入请求
    output_requests = [sum(matrix[i][j] for i in range(num_inputs))
                       for j in range(num_outputs)]

    print(f"\n{name}统计:")
    print(f"  每个输入的请求数: {input_requests}")
    print(f"  每个输出的竞争度: {output_requests}")

    total_requests = sum(input_requests)
    max_output_contention = max(output_requests)

    print(f"  总请求数: {total_requests}")
    print(f"  最大输出竞争: {max_output_contention}个输入竞争同一输出")

    # 理论最大匹配数
    theoretical_max = min(sum(1 for r in input_requests if r > 0),
                         sum(1 for r in output_requests if r > 0))
    print(f"  理论最大匹配: {theoretical_max}")


def test_real_world_scenario():
    """模拟真实世界场景"""
    print("\n" + "=" * 60)
    print("真实世界场景模拟")
    print("=" * 60)

    print("\n场景：4个DMA IP，5个方向，流量不均匀")

    # 假设：
    # - 大部分流量走TR/TL（横向传输）
    # - 少数走TU/TD（垂直环）
    # - 极少走EQ（本地弹出）

    import random
    random.seed(42)

    rr = RoundRobinArbiter()
    islip = MaxWeightMatchingArbiter(algorithm="islip", iterations=2)

    direction_usage = {"TR": 0, "TL": 0, "TU": 0, "TD": 0, "EQ": 0}
    rr_total_matches = 0
    islip_total_matches = 0

    # 生成1000轮真实流量模式
    for cycle in range(1000):
        # 随机生成请求矩阵（符合真实特点）
        request_matrix = []
        for ip in range(4):
            row = [False] * 5  # [TR, TL, TU, TD, EQ]

            # 每个IP只选一个方向（符合实际）
            if random.random() < 0.9:  # 90%的时间有流量
                # 70%走横向，20%走垂直，10%走EQ
                choice = random.random()
                if choice < 0.35:
                    row[0] = True  # TR
                elif choice < 0.70:
                    row[1] = True  # TL
                elif choice < 0.85:
                    row[2] = True  # TU
                elif choice < 0.95:
                    row[3] = True  # TD
                else:
                    row[4] = True  # EQ

            request_matrix.append(row)

        # 统计方向使用
        for i, row in enumerate(request_matrix):
            for j, val in enumerate(row):
                if val:
                    direction_usage[["TR", "TL", "TU", "TD", "EQ"][j]] += 1

        # 执行仲裁
        rr_matches = rr.match(request_matrix, queue_id="real_world_rr")
        islip_matches = islip.match(request_matrix, queue_id="real_world_islip")

        rr_total_matches += len(rr_matches)
        islip_total_matches += len(islip_matches)

    print(f"\n1000轮统计:")
    print(f"  方向使用频率: {direction_usage}")
    print(f"  RoundRobin总匹配: {rr_total_matches} ({rr_total_matches/1000:.2f}/轮)")
    print(f"  iSLIP总匹配:      {islip_total_matches} ({islip_total_matches/1000:.2f}/轮)")

    diff = rr_total_matches - islip_total_matches
    if abs(diff) < 10:
        print(f"  性能: 相当 (差异{diff})")
    elif diff > 0:
        print(f"  性能: RoundRobin优 (+{diff})")
    else:
        print(f"  性能: iSLIP优 (+{-diff})")


if __name__ == "__main__":
    analyze_request_patterns()
    test_real_world_scenario()
    print("\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)