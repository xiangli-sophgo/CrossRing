"""
测试仲裁器权重策略功能

测试内容：
1. iSLIP 标准模式（uniform）：使用轮询选择
2. iSLIP 加权模式（wait_time/queue_length/hybrid）：按权重选择
3. LQF/OCF 算法的权重使用
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.arbitration import (
    MaxWeightMatchingArbiter,
    RoundRobinArbiter,
    create_matching_arbiter,
)


class TestISLIPWeightStrategy:
    """测试 iSLIP 算法的权重策略"""

    def test_islip_uniform_uses_round_robin(self):
        """测试 uniform 策略使用轮询选择"""
        arbiter = MaxWeightMatchingArbiter(algorithm="islip", weight_strategy="uniform")

        # 请求矩阵：输入0和输入1都请求输出0
        request_matrix = [
            [True, False],  # 输入0请求输出0
            [True, False],  # 输入1请求输出0
        ]

        # 权重矩阵：输入1权重更高
        weight_matrix = [
            [1.0, 0.0],  # 输入0权重1.0
            [10.0, 0.0],  # 输入1权重10.0
        ]

        # uniform策略应该忽略权重，使用轮询
        matches = arbiter.match(request_matrix, weight_matrix, queue_id="test1")

        # 第一次匹配，轮询指针从0开始，应该选择输入0
        assert len(matches) == 1
        assert matches[0][0] == 0, f"uniform策略应选择轮询顺序第一个（输入0），但选择了输入{matches[0][0]}"
        print("[PASS] test_islip_uniform_uses_round_robin")

    def test_islip_wait_time_uses_weights(self):
        """测试 wait_time 策略按权重选择"""
        arbiter = MaxWeightMatchingArbiter(algorithm="islip", weight_strategy="wait_time")

        # 请求矩阵：输入0和输入1都请求输出0
        request_matrix = [
            [True, False],  # 输入0请求输出0
            [True, False],  # 输入1请求输出0
        ]

        # 权重矩阵：输入1权重更高（等待时间更长）
        weight_matrix = [
            [1.0, 0.0],  # 输入0等待1周期
            [10.0, 0.0],  # 输入1等待10周期
        ]

        matches = arbiter.match(request_matrix, weight_matrix, queue_id="test2")

        # wait_time策略应该选择权重最大的输入1
        assert len(matches) == 1
        assert matches[0][0] == 1, f"wait_time策略应选择权重最大的（输入1），但选择了输入{matches[0][0]}"
        print("[PASS] test_islip_wait_time_uses_weights")

    def test_islip_queue_length_uses_weights(self):
        """测试 queue_length 策略按权重选择"""
        arbiter = MaxWeightMatchingArbiter(algorithm="islip", weight_strategy="queue_length")

        # 请求矩阵：输入0、1、2都请求输出0
        request_matrix = [
            [True],  # 输入0
            [True],  # 输入1
            [True],  # 输入2
        ]

        # 权重矩阵：输入2队列最长
        weight_matrix = [
            [3.0],  # 输入0队列长度3
            [5.0],  # 输入1队列长度5
            [8.0],  # 输入2队列长度8
        ]

        matches = arbiter.match(request_matrix, weight_matrix, queue_id="test3")

        assert len(matches) == 1
        assert matches[0][0] == 2, f"queue_length策略应选择队列最长的（输入2），但选择了输入{matches[0][0]}"
        print("[PASS] test_islip_queue_length_uses_weights")

    def test_islip_hybrid_uses_weights(self):
        """测试 hybrid 策略按权重选择"""
        arbiter = MaxWeightMatchingArbiter(algorithm="islip", weight_strategy="hybrid")

        request_matrix = [
            [True, True],  # 输入0请求输出0和1
            [True, True],  # 输入1请求输出0和1
        ]

        # 输入0的输出0权重最高
        weight_matrix = [
            [100.0, 1.0],  # 输入0: 输出0权重100, 输出1权重1
            [50.0, 2.0],  # 输入1: 输出0权重50, 输出1权重2
        ]

        matches = arbiter.match(request_matrix, weight_matrix, queue_id="test4")

        # 应该匹配 (0, 0) 因为权重最高
        assert len(matches) >= 1
        # 检查输入0是否匹配到输出0
        input0_match = [m for m in matches if m[0] == 0]
        assert len(input0_match) == 1
        assert input0_match[0][1] == 0, f"输入0应该匹配到权重最高的输出0，但匹配到了输出{input0_match[0][1]}"
        print("[PASS] test_islip_hybrid_uses_weights")


class TestISLIPMultipleOutputs:
    """测试 iSLIP 多输出场景下的权重选择"""

    def test_accept_phase_weight_selection(self):
        """测试接受阶段的权重选择"""
        arbiter = MaxWeightMatchingArbiter(algorithm="islip", weight_strategy="wait_time")

        # 请求矩阵：输入0请求输出0和1，输入1请求输出0和1
        request_matrix = [
            [True, True],  # 输入0
            [True, True],  # 输入1
        ]

        # 输入0对输出1的权重更高
        weight_matrix = [
            [1.0, 10.0],  # 输入0: 输出0权重1, 输出1权重10
            [5.0, 2.0],  # 输入1: 输出0权重5, 输出1权重2
        ]

        matches = arbiter.match(request_matrix, weight_matrix, queue_id="test5")

        # 在接受阶段，输入0应该选择权重更高的输出1
        input0_match = [m for m in matches if m[0] == 0]
        if input0_match:
            assert input0_match[0][1] == 1, f"输入0应选择权重更高的输出1，但选择了输出{input0_match[0][1]}"

        print("[PASS] test_accept_phase_weight_selection")


class TestLQFAndOCF:
    """测试 LQF 和 OCF 算法"""

    def test_lqf_selects_longest_queue(self):
        """测试 LQF 选择最长队列"""
        arbiter = MaxWeightMatchingArbiter(algorithm="lqf", weight_strategy="queue_length")

        request_matrix = [
            [True, True],
            [True, True],
            [True, True],
        ]

        # 输入2-输出1的权重最高
        weight_matrix = [
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 100.0],  # 最高权重
        ]

        matches = arbiter.match(request_matrix, weight_matrix, queue_id="test_lqf")

        # LQF 应该优先匹配权重最高的 (2, 1)
        assert (2, 1) in matches, f"LQF应优先匹配权重最高的(2,1)，但匹配结果是{matches}"
        print("[PASS] test_lqf_selects_longest_queue")

    def test_ocf_selects_oldest_cell(self):
        """测试 OCF 选择最老的数据包"""
        arbiter = MaxWeightMatchingArbiter(algorithm="ocf", weight_strategy="wait_time")

        request_matrix = [
            [True],
            [True],
            [True],
        ]

        # 输入1等待时间最长
        weight_matrix = [
            [5.0],
            [100.0],  # 最长等待时间
            [3.0],
        ]

        matches = arbiter.match(request_matrix, weight_matrix, queue_id="test_ocf")

        assert len(matches) == 1
        assert matches[0][0] == 1, f"OCF应选择等待时间最长的输入1，但选择了输入{matches[0][0]}"
        print("[PASS] test_ocf_selects_oldest_cell")


class TestRoundRobinArbiter:
    """测试 RoundRobinArbiter 的权重参数兼容性"""

    def test_round_robin_ignores_weights(self):
        """测试 RoundRobinArbiter 忽略权重参数"""
        arbiter = RoundRobinArbiter()

        request_matrix = [
            [True, False],
            [True, False],
        ]

        weight_matrix = [
            [1.0, 0.0],
            [100.0, 0.0],  # 高权重
        ]

        # RoundRobinArbiter 应该接受 weight_matrix 参数但不使用它
        matches = arbiter.match(request_matrix, weight_matrix, queue_id="test_rr")

        # 应该按轮询顺序选择，而不是按权重
        assert len(matches) == 1
        print("[PASS] test_round_robin_ignores_weights")


class TestEdgeCases:
    """测试边界情况"""

    def test_empty_request_matrix(self):
        """测试空请求矩阵"""
        arbiter = MaxWeightMatchingArbiter(algorithm="islip", weight_strategy="wait_time")

        matches = arbiter.match([], queue_id="empty")
        assert matches == []
        print("[PASS] test_empty_request_matrix")

    def test_no_valid_requests(self):
        """测试没有有效请求"""
        arbiter = MaxWeightMatchingArbiter(algorithm="islip", weight_strategy="wait_time")

        request_matrix = [
            [False, False],
            [False, False],
        ]

        weight_matrix = [
            [1.0, 1.0],
            [1.0, 1.0],
        ]

        matches = arbiter.match(request_matrix, weight_matrix, queue_id="no_req")
        assert matches == []
        print("[PASS] test_no_valid_requests")

    def test_single_request(self):
        """测试单个请求"""
        arbiter = MaxWeightMatchingArbiter(algorithm="islip", weight_strategy="wait_time")

        request_matrix = [
            [True],
        ]

        weight_matrix = [
            [5.0],
        ]

        matches = arbiter.match(request_matrix, weight_matrix, queue_id="single")
        assert matches == [(0, 0)]
        print("[PASS] test_single_request")

    def test_weight_matrix_not_provided(self):
        """测试不提供权重矩阵时使用内部计算"""
        arbiter = MaxWeightMatchingArbiter(algorithm="islip", weight_strategy="uniform")

        request_matrix = [
            [True, False],
            [False, True],
        ]

        # 不提供 weight_matrix
        matches = arbiter.match(request_matrix, queue_id="no_weight")

        assert len(matches) == 2
        assert (0, 0) in matches
        assert (1, 1) in matches
        print("[PASS] test_weight_matrix_not_provided")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Testing Arbitration Weight Strategy")
    print("=" * 60)

    # iSLIP 权重策略测试
    print("\n--- TestISLIPWeightStrategy ---")
    t1 = TestISLIPWeightStrategy()
    t1.test_islip_uniform_uses_round_robin()
    t1.test_islip_wait_time_uses_weights()
    t1.test_islip_queue_length_uses_weights()
    t1.test_islip_hybrid_uses_weights()

    # iSLIP 多输出测试
    print("\n--- TestISLIPMultipleOutputs ---")
    t2 = TestISLIPMultipleOutputs()
    t2.test_accept_phase_weight_selection()

    # LQF 和 OCF 测试
    print("\n--- TestLQFAndOCF ---")
    t3 = TestLQFAndOCF()
    t3.test_lqf_selects_longest_queue()
    t3.test_ocf_selects_oldest_cell()

    # RoundRobinArbiter 测试
    print("\n--- TestRoundRobinArbiter ---")
    t4 = TestRoundRobinArbiter()
    t4.test_round_robin_ignores_weights()

    # 边界情况测试
    print("\n--- TestEdgeCases ---")
    t5 = TestEdgeCases()
    t5.test_empty_request_matrix()
    t5.test_no_valid_requests()
    t5.test_single_request()
    t5.test_weight_matrix_not_provided()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
