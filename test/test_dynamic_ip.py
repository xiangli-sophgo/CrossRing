"""
测试动态IP挂载功能

验证:
1. Traffic文件解析
2. IP接口动态创建
3. CHANNEL_SPEC反向推断
4. CH_NAME_LIST更新
"""

import os
import sys
import io

# 设置UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.traffic_ip_extractor import TrafficIPExtractor
from config.config import CrossRingConfig


def test_traffic_extraction():
    """测试traffic文件IP提取"""
    print("\n=== 测试1: Traffic文件IP提取 ===")

    # 创建测试traffic文件
    test_traffic = os.path.join(project_root, "test_data", "LLama2_AllReduce.txt")

    if not os.path.exists(test_traffic):
        print(f"⚠ 测试traffic文件不存在: {test_traffic}")
        return False

    extractor = TrafficIPExtractor()
    result = extractor.extract_from_file(test_traffic)

    print(f"✓ 解析traffic文件: {test_traffic}")
    print(f"  - 提取到 {len(result['required_ips'])} 个IP接口需求")
    print(f"  - 是否有跨Die流量: {result['has_cross_die']}")
    print(f"  - Traffic格式: {result['traffic_format']}")

    # 提取唯一IP类型
    ip_types = TrafficIPExtractor.get_unique_ip_types(result['required_ips'])
    print(f"  - 唯一IP类型: {ip_types}")

    # 推断CHANNEL_SPEC
    channel_spec = TrafficIPExtractor.infer_channel_spec(ip_types)
    print(f"  - 推断CHANNEL_SPEC: {channel_spec}")

    return True


def test_config_methods():
    """测试配置类的新方法"""
    print("\n=== 测试2: Config类新方法 ===")

    config_path = os.path.join(project_root, "config", "topologies", "topo_4x5.yaml")
    config = CrossRingConfig(config_path)

    print(f"✓ 加载配置: {config_path}")
    print(f"  - 初始CHANNEL_SPEC: {config.CHANNEL_SPEC}")
    print(f"  - 初始CH_NAME_LIST长度: {len(config.CH_NAME_LIST)}")

    # 测试更新方法
    test_ip_types = ["gdma_0", "gdma_1", "ddr_0"]
    config.update_channel_list_from_ips(test_ip_types)
    print(f"\n  更新后CH_NAME_LIST: {config.CH_NAME_LIST}")

    inferred_spec = config.infer_channel_spec_from_ips(test_ip_types)
    print(f"  反向推断CHANNEL_SPEC: {inferred_spec}")

    return True


def test_integration():
    """测试完整集成流程"""
    print("\n=== 测试3: 完整集成流程 ===")

    try:
        from src.noc.REQ_RSP import REQ_RSP_model

        config_path = os.path.join(project_root, "config", "topologies", "topo_4x5.yaml")
        config = CrossRingConfig(config_path)

        print(f"✓ 创建模型实例...")
        sim = REQ_RSP_model(
            model_type="REQ_RSP",
            config=config,
            topo_type="4x5",
            verbose=0,
        )

        print(f"✓ 配置traffic调度器...")
        sim.setup_traffic_scheduler(
            traffic_file_path=os.path.join(project_root, "test_data"),
            traffic_chains=[["LLama2_AllReduce.txt"]],
        )

        print(f"✓ Traffic解析完成!")
        print(f"  - CH_NAME_LIST: {config.CH_NAME_LIST}")
        print(f"  - CHANNEL_SPEC: {config.CHANNEL_SPEC}")

        print(f"\n✓ 配置result analysis...")
        sim.setup_result_analysis(
            plot_RN_BW_fig=False,
            plot_flow_fig=False,
        )

        print(f"✓ 初始化模型...")
        sim.initial()

        print(f"✓ 模型初始化完成!")
        print(f"  - 创建的IP接口数量: {len(sim.ip_modules)}")

        # 统计每种IP类型的数量
        ip_type_counts = {}
        for (ip_type, node_id) in sim.ip_modules.keys():
            if ip_type not in ip_type_counts:
                ip_type_counts[ip_type] = 0
            ip_type_counts[ip_type] += 1

        print(f"  - IP类型统计:")
        for ip_type, count in sorted(ip_type_counts.items()):
            print(f"    • {ip_type}: {count}个")

        return True

    except Exception as e:
        print(f"✗ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("动态IP挂载功能测试")
    print("=" * 60)

    results = []

    # 测试1: Traffic提取
    results.append(("Traffic文件IP提取", test_traffic_extraction()))

    # 测试2: Config方法
    results.append(("Config类新方法", test_config_methods()))

    # 测试3: 完整集成
    results.append(("完整集成流程", test_integration()))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)
    print("\n" + ("=" * 60))
    if all_passed:
        print("✓ 所有测试通过!")
    else:
        print("✗ 部分测试失败,请检查错误信息")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
