"""
测试D2D IP挂载修复
验证每个Die只挂载自己涉及的IP，不会重复挂载所有traffic中的IP
"""

import sys
import os

# 解决Windows下的编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.traffic_ip_extractor import TrafficIPExtractor


def test_single_die_extraction():
    """测试单Die场景的IP提取（验证原有功能不受影响）"""
    print("\n=== 测试1: 单Die场景IP提取 ===")

    # 创建测试traffic内容（单Die格式）
    test_traffic = """# Single Die Traffic
0,1,gdma_0,7,ddr_0,R,4
0,2,gdma_1,7,ddr_0,W,4
"""

    # 写入临时文件
    test_file = "test_single_die.txt"
    with open(test_file, 'w') as f:
        f.write(test_traffic)

    # 提取IP（不指定die_id）
    extractor = TrafficIPExtractor()
    result = extractor.extract_from_multiple_files([test_file], die_id=None)

    print(f"提取到的IP需求: {result['required_ips']}")
    print(f"流量格式: {result['traffic_format']}")

    expected_ips = {(1, "gdma_0"), (7, "ddr_0"), (2, "gdma_1")}
    assert result['required_ips'] == expected_ips, f"期望 {expected_ips}, 实际 {result['required_ips']}"

    # 清理
    os.remove(test_file)
    print("✓ 单Die场景测试通过")


def test_d2d_extraction_with_die_filter():
    """测试D2D场景的Die级别IP过滤"""
    print("\n=== 测试2: D2D场景Die级别IP过滤 ===")

    # 创建测试traffic内容（D2D格式）
    test_traffic = """# D2D Traffic (4 Dies)
# inject_time, src_die, src_node, src_ip, dst_die, dst_node, dst_ip, req_type, burst
0,0,1,gdma_0,3,7,ddr_0,W,4
0,1,1,gdma_0,0,7,ddr_1,W,4
0,2,5,gdma_2,3,8,ddr_2,R,4
0,3,10,gdma_3,1,15,ddr_3,R,4
"""

    # 写入临时文件
    test_file = "test_d2d_traffic.txt"
    with open(test_file, 'w') as f:
        f.write(test_traffic)

    # 测试Die 0的IP提取
    print("\n--- Die 0 的IP提取 ---")
    extractor_die0 = TrafficIPExtractor()
    result_die0 = extractor_die0.extract_from_multiple_files([test_file], die_id=0)
    print(f"Die 0 IP需求: {result_die0['required_ips']}")
    expected_die0 = {(1, "gdma_0"), (7, "ddr_1")}  # src_die=0的源IP + dst_die=0的目标IP
    assert result_die0['required_ips'] == expected_die0, f"Die 0期望 {expected_die0}, 实际 {result_die0['required_ips']}"
    print("✓ Die 0 IP过滤正确")

    # 测试Die 1的IP提取
    print("\n--- Die 1 的IP提取 ---")
    extractor_die1 = TrafficIPExtractor()
    result_die1 = extractor_die1.extract_from_multiple_files([test_file], die_id=1)
    print(f"Die 1 IP需求: {result_die1['required_ips']}")
    expected_die1 = {(1, "gdma_0"), (15, "ddr_3")}  # src_die=1的源IP + dst_die=1的目标IP
    assert result_die1['required_ips'] == expected_die1, f"Die 1期望 {expected_die1}, 实际 {result_die1['required_ips']}"
    print("✓ Die 1 IP过滤正确")

    # 测试Die 2的IP提取
    print("\n--- Die 2 的IP提取 ---")
    extractor_die2 = TrafficIPExtractor()
    result_die2 = extractor_die2.extract_from_multiple_files([test_file], die_id=2)
    print(f"Die 2 IP需求: {result_die2['required_ips']}")
    expected_die2 = {(5, "gdma_2")}  # 只有src_die=2的源IP
    assert result_die2['required_ips'] == expected_die2, f"Die 2期望 {expected_die2}, 实际 {result_die2['required_ips']}"
    print("✓ Die 2 IP过滤正确")

    # 测试Die 3的IP提取
    print("\n--- Die 3 的IP提取 ---")
    extractor_die3 = TrafficIPExtractor()
    result_die3 = extractor_die3.extract_from_multiple_files([test_file], die_id=3)
    print(f"Die 3 IP需求: {result_die3['required_ips']}")
    expected_die3 = {(10, "gdma_3"), (7, "ddr_0"), (8, "ddr_2")}  # src_die=3的源IP + dst_die=3的两个目标IP
    assert result_die3['required_ips'] == expected_die3, f"Die 3期望 {expected_die3}, 实际 {result_die3['required_ips']}"
    print("✓ Die 3 IP过滤正确")

    # 清理
    os.remove(test_file)
    print("\n✓ D2D场景Die级别IP过滤测试通过")


def test_d2d_no_filter():
    """测试D2D场景不指定die_id时的行为（兼容性测试）"""
    print("\n=== 测试3: D2D场景不指定die_id（兼容性）===")

    # 创建测试traffic内容
    test_traffic = """# D2D Traffic
0,0,1,gdma_0,1,7,ddr_0,W,4
0,1,2,gdma_1,0,8,ddr_1,R,4
"""

    # 写入临时文件
    test_file = "test_d2d_no_filter.txt"
    with open(test_file, 'w') as f:
        f.write(test_traffic)

    # 不指定die_id，应该提取所有IP
    extractor = TrafficIPExtractor()
    result = extractor.extract_from_multiple_files([test_file], die_id=None)

    print(f"提取到的IP需求: {result['required_ips']}")
    expected_all = {(1, "gdma_0"), (7, "ddr_0"), (2, "gdma_1"), (8, "ddr_1")}
    assert result['required_ips'] == expected_all, f"期望 {expected_all}, 实际 {result['required_ips']}"

    # 清理
    os.remove(test_file)
    print("✓ 兼容性测试通过：不指定die_id时提取所有IP")


def test_cross_die_detection():
    """测试跨Die流量检测"""
    print("\n=== 测试4: 跨Die流量检测 ===")

    # 创建测试traffic内容（包含跨Die和Die内流量）
    test_traffic = """# Mixed Traffic
0,0,1,gdma_0,0,7,ddr_0,W,4
0,0,2,gdma_1,1,8,ddr_1,R,4
"""

    # 写入临时文件
    test_file = "test_cross_die_detection.txt"
    with open(test_file, 'w') as f:
        f.write(test_traffic)

    # 提取并检测跨Die流量
    extractor = TrafficIPExtractor()
    result = extractor.extract_from_multiple_files([test_file])

    print(f"检测到跨Die流量: {result['has_cross_die']}")
    assert result['has_cross_die'] == True, "应该检测到跨Die流量"

    # 清理
    os.remove(test_file)
    print("✓ 跨Die流量检测测试通过")


if __name__ == "__main__":
    print("=" * 60)
    print("D2D IP挂载修复验证测试")
    print("=" * 60)

    try:
        test_single_die_extraction()
        test_d2d_extraction_with_die_filter()
        test_d2d_no_filter()
        test_cross_die_detection()

        print("\n" + "=" * 60)
        print("✓✓✓ 所有测试通过！IP挂载修复验证成功 ✓✓✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
