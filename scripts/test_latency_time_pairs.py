#!/usr/bin/env python3
"""测试time_value_pairs数据收集"""

import sys
sys.path.insert(0, '..')

from src.analysis.data_collectors import LatencyStatsCollector
from src.analysis.analyzers import RequestInfo

# 创建测试数据
test_requests = []
for i in range(10):
    req = RequestInfo(
        packet_id=i,
        start_time=i * 100,  # ns
        end_time=i * 100 + 50,  # ns
        req_type="read" if i % 2 == 0 else "write",
        source_node=0,
        dest_node=1,
        source_type="gdma_0",
        dest_type="ddr_0",
        burst_length=4,
        total_bytes=512,
        cmd_latency=10 + i,  # ns
        data_latency=20 + i,  # ns
        transaction_latency=30 + i,  # ns
    )
    test_requests.append(req)

# 测试数据收集
collector = LatencyStatsCollector()
stats = collector.calculate_latency_stats(test_requests)

# 检查结果
print("\n=== 测试结果 ===")
for cat in ["cmd", "data", "trans"]:
    tvp_count = len(stats[cat]["mixed"]["time_value_pairs"])
    print(f"{cat} time_value_pairs 数量: {tvp_count}")
    if tvp_count > 0:
        print(f"  前3个: {stats[cat]['mixed']['time_value_pairs'][:3]}")
    else:
        print("  ❌ 没有数据!")

print("\n如果上面显示数据，说明代码逻辑正确。")
print("如果没有数据，说明RequestInfo的end_time可能有问题。")
