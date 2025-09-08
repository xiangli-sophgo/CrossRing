"""
RoundRobinScheduler的单元测试。
测试轮询调度算法的核心功能、公平性和性能。
"""

import unittest
from collections import defaultdict, Counter
import time
from src.utils.components.round_robin import RoundRobinScheduler


class TestRoundRobinScheduler(unittest.TestCase):

    def setUp(self):
        """在每个测试方法前设置测试环境。"""
        self.scheduler = RoundRobinScheduler()

    def test_basic_round_robin(self):
        """测试基本轮询功能。"""
        candidates = ["A", "B", "C", "D"]
        key = "test_basic"

        # Test that we get each candidate in order
        selected_items = []
        for _ in range(8):  # Two complete rounds
            selected, idx = self.scheduler.select(key, candidates)
            selected_items.append(selected)

        # Should cycle through candidates in order
        expected = ["A", "B", "C", "D", "A", "B", "C", "D"]
        self.assertEqual(selected_items, expected)

        # Check final index position
        final_index = self.scheduler.get_current_index(key)
        self.assertEqual(final_index, 0)  # Should wrap around to 0

    def test_index_always_advances(self):
        """测试即使没有找到匹配项，索引也会前进。"""
        candidates = ["A", "B", "C"]
        key = "test_advance"

        def never_match(item):
            return False

        # Even with no matches, index should advance
        for expected_idx in [1, 2, 0, 1, 2]:  # Cycles through indices
            selected, _ = self.scheduler.select(key, candidates, never_match)
            self.assertIsNone(selected)  # No match
            current_idx = self.scheduler.get_current_index(key)
            self.assertEqual(current_idx, expected_idx)

    def test_conditional_selection(self):
        """测试带条件的选择。"""
        candidates = [1, 2, 3, 4, 5]
        key = "test_condition"

        def only_even(num):
            return num % 2 == 0

        # Should only select even numbers, but index still advances
        results = []
        for _ in range(10):
            selected, _ = self.scheduler.select(key, candidates, only_even)
            if selected is not None:
                results.append(selected)

        # Should get even numbers in cycling order
        # Starting from 1: skip 1, select 2, skip 3, select 4, skip 5, skip 1, select 2...
        expected_evens = [2, 4, 2, 4, 2]  # Pattern repeats
        self.assertEqual(results[:5], expected_evens)

    def test_fairness(self):
        """测试在多轮中所有候选项都得到公平对待。"""
        candidates = ["A", "B", "C", "D", "E"]
        key = "test_fairness"

        selection_count = Counter()
        total_selections = 100

        for _ in range(total_selections):
            selected, _ = self.scheduler.select(key, candidates)
            if selected:
                selection_count[selected] += 1

        # Each candidate should be selected exactly 20 times (100/5)
        for candidate in candidates:
            self.assertEqual(selection_count[candidate], 20)

    def test_independence(self):
        """测试不同键保持独立状态。"""
        candidates = ["X", "Y", "Z"]
        key1 = "independent_1"
        key2 = "independent_2"

        # Advance key1's scheduler
        for _ in range(5):
            self.scheduler.select(key1, candidates)

        # key2 should start fresh
        selected, _ = self.scheduler.select(key2, candidates)
        self.assertEqual(selected, "X")  # Should start from beginning

        # Verify indices are different
        idx1 = self.scheduler.get_current_index(key1)
        idx2 = self.scheduler.get_current_index(key2)
        self.assertEqual(idx1, 2)  # 5 % 3 = 2
        self.assertEqual(idx2, 1)  # Advanced once

    def test_edge_cases(self):
        """测试边界情况和边界条件。"""
        # Empty candidate list
        selected, idx = self.scheduler.select("empty", [])
        self.assertIsNone(selected)
        self.assertIsNone(idx)

        # Single candidate
        selected, idx = self.scheduler.select("single", ["ONLY"])
        self.assertEqual(selected, "ONLY")
        self.assertEqual(idx, 0)

        # Second call with single candidate
        selected, idx = self.scheduler.select("single", ["ONLY"])
        self.assertEqual(selected, "ONLY")
        self.assertEqual(idx, 0)  # Always 0 for single item

    def test_reset_functionality(self):
        """测试索引重置功能。"""
        candidates = ["A", "B", "C"]
        key = "test_reset"

        # Advance the scheduler
        for _ in range(5):
            self.scheduler.select(key, candidates)

        self.assertEqual(self.scheduler.get_current_index(key), 2)

        # Reset specific key
        self.scheduler.reset_index(key)
        self.assertEqual(self.scheduler.get_current_index(key), 0)

        # Reset all keys
        self.scheduler.select("other_key", candidates)
        self.scheduler.reset_index()  # Reset all
        self.assertEqual(self.scheduler.get_current_index(key), 0)
        self.assertEqual(self.scheduler.get_current_index("other_key"), 0)

    def test_statistics(self):
        """测试统计信息收集。"""
        candidates = ["A", "B", "C"]
        key = "test_stats"

        # Make some selections
        successful = 0
        for i in range(10):
            # Every other selection fails
            def condition(item):
                return i % 2 == 0

            selected, _ = self.scheduler.select(key, candidates, condition)
            if selected:
                successful += 1

        stats = self.scheduler.get_stats()
        self.assertEqual(stats["total_selections"], 10)
        self.assertEqual(stats["successful_selections"], 5)
        self.assertEqual(stats["success_rate"], 0.5)
        self.assertEqual(stats["queue_accesses"][key], 10)

    def test_performance_vs_deque_operations(self):
        """与传统的remove+append方法的性能对比。"""
        from collections import deque

        # Test data
        candidates = list(range(100))
        iterations = 1000

        # Test RoundRobinScheduler performance
        start_time = time.perf_counter()
        for i in range(iterations):
            key = f"perf_test_{i % 10}"  # 10 different queues
            self.scheduler.select(key, candidates)
        rr_time = time.perf_counter() - start_time

        # Test traditional deque remove+append performance
        queues = {f"deque_test_{i}": deque(candidates) for i in range(10)}

        start_time = time.perf_counter()
        for i in range(iterations):
            key = f"deque_test_{i % 10}"
            queue = queues[key]
            if queue:
                item = queue[0]  # Get first item
                queue.remove(item)  # O(n) operation
                queue.append(item)  # O(1) operation
        deque_time = time.perf_counter() - start_time

        # RoundRobinScheduler should be faster (O(1) vs O(n))
        print(f"\nPerformance comparison:")
        print(f"RoundRobinScheduler: {rr_time:.6f}s")
        print(f"Deque remove+append: {deque_time:.6f}s")
        print(f"Speedup: {deque_time / rr_time:.2f}x")

        # Assert that new method is faster (should be true for lists > ~10 items)
        self.assertLess(rr_time, deque_time, "RoundRobinScheduler should be faster than remove+append")

    def test_noc_simulation_scenario(self):
        """测试类似实际NoC使用的场景。"""
        # Simulate IQ arbitration scenario
        ip_types = ["sdma_0", "gdma_0", "cdma_0", "ddr_0", "l2m_0"]

        # Multiple IQ queues with different keys
        scenarios = [
            ("IQ_TR_14_req", ip_types),
            ("IQ_TL_14_req", ip_types),
            ("IQ_TR_14_data", ip_types),
            ("EQ_sdma_10", [0, 1, 2, 3]),  # TU, TD, IQ, RB
            ("RB_TL_5_1", [0, 1, 2, 3]),  # TL, TR, TU, TD
        ]

        results = {}

        # Simulate 100 cycles of arbitration
        for cycle in range(100):
            for key, candidates in scenarios:
                # Simulate some candidates being unavailable
                def random_availability(item):
                    # Use cycle and item to create pseudo-random availability
                    return (hash(str(item) + str(cycle)) % 3) != 0

                selected, _ = self.scheduler.select(key, candidates, random_availability)
                if selected is not None:
                    if key not in results:
                        results[key] = Counter()
                    results[key][selected] += 1

        # Verify that each scenario got some selections
        for key, candidates in scenarios:
            self.assertIn(key, results, f"Key {key} should have some selections")
            self.assertGreater(len(results[key]), 0, f"Key {key} should select some candidates")

        # Print results for manual inspection
        print("\nNoC Simulation Results:")
        for key, counter in results.items():
            print(f"{key}: {dict(counter)}")


def run_scheduler_tests():
    """便捷函数，运行所有测试。"""
    unittest.main(argv=[""], exit=False, verbosity=2)


if __name__ == "__main__":
    # 当脚本直接执行时运行测试
    unittest.main(verbosity=2)
