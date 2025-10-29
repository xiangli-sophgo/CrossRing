"""
检查ORDERING_PRESERVATION_MODE和ORDERING_GRANULARITY配置是否正确生效
"""
import sys
import io
from pathlib import Path

# 设置UTF-8编码输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import CrossRingConfig

def check_config_loading():
    """检查配置文件加载"""
    print("=" * 80)
    print("配置文件加载检查")
    print("=" * 80)

    # 测试不同的拓扑配置
    topologies = ["topo_3x3", "topo_4x4", "topo_4x5", "topo_5x4"]

    for topo in topologies:
        print(f"\n【拓扑: {topo}】")
        try:
            config = CrossRingConfig(topo)
            print(f"  ORDERING_PRESERVATION_MODE: {config.ORDERING_PRESERVATION_MODE}")
            print(f"  ORDERING_GRANULARITY: {config.ORDERING_GRANULARITY}")

            # 检查是否有保序相关配置
            if hasattr(config, "IN_ORDER_PACKET_CATEGORIES"):
                print(f"  IN_ORDER_PACKET_CATEGORIES: {config.IN_ORDER_PACKET_CATEGORIES}")

            # Mode 2特有的配置
            if config.ORDERING_PRESERVATION_MODE == 2:
                if hasattr(config, "TL_ALLOWED_SOURCE_NODES"):
                    print(f"  TL_ALLOWED_SOURCE_NODES: {config.TL_ALLOWED_SOURCE_NODES}")
                    print(f"  TR_ALLOWED_SOURCE_NODES: {config.TR_ALLOWED_SOURCE_NODES}")
                    print(f"  TU_ALLOWED_SOURCE_NODES: {config.TU_ALLOWED_SOURCE_NODES}")
                    print(f"  TD_ALLOWED_SOURCE_NODES: {config.TD_ALLOWED_SOURCE_NODES}")
                else:
                    print("  ⚠️ Mode 2配置缺少方向白名单")

        except Exception as e:
            print(f"  ❌ 加载失败: {e}")


def test_granularity_effect():
    """测试粒度参数是否真的影响order_id分配"""
    print("\n" + "=" * 80)
    print("测试粒度参数实际效果")
    print("=" * 80)

    from src.utils.components.flit import Flit

    # 测试场景：同一节点的两个不同IP
    test_cases = [
        # (src_node, src_type, dest_node, dest_type, description)
        (0, "gdma_0", 4, "ddr_0", "gdma_0 -> ddr_0 (第1个请求)"),
        (0, "gdma_1", 4, "ddr_0", "gdma_1 -> ddr_0 (第1个请求)"),
        (0, "gdma_0", 4, "ddr_0", "gdma_0 -> ddr_0 (第2个请求)"),
        (0, "gdma_1", 4, "ddr_0", "gdma_1 -> ddr_0 (第2个请求)"),
    ]

    # 测试IP层级 (granularity=0)
    print("\n【IP层级 (ORDERING_GRANULARITY=0)】")
    print("预期：不同IP有独立的order_id序列")
    Flit.reset_order_ids()

    order_ids_ip = []
    for src_node, src_type, dest_node, dest_type, desc in test_cases:
        order_id = Flit.get_next_order_id(
            src_node=src_node, src_type=src_type,
            dest_node=dest_node, dest_type=dest_type,
            packet_category="REQ", granularity=0
        )
        order_ids_ip.append(order_id)
        print(f"  {desc}: order_id={order_id}")

    # 验证：gdma_0应该是[1,2], gdma_1应该是[1,2]
    if order_ids_ip[0] == 1 and order_ids_ip[1] == 1 and order_ids_ip[2] == 2 and order_ids_ip[3] == 2:
        print("  ✓ IP层级正常工作：不同IP有独立序列")
    else:
        print(f"  ❌ IP层级异常：order_id序列 = {order_ids_ip}")

    # 测试节点层级 (granularity=1)
    print("\n【节点层级 (ORDERING_GRANULARITY=1)】")
    print("预期：同节点不同IP共享order_id序列")
    Flit.reset_order_ids()

    order_ids_node = []
    for src_node, src_type, dest_node, dest_type, desc in test_cases:
        order_id = Flit.get_next_order_id(
            src_node=src_node, src_type=src_type,
            dest_node=dest_node, dest_type=dest_type,
            packet_category="REQ", granularity=1
        )
        order_ids_node.append(order_id)
        print(f"  {desc}: order_id={order_id}")

    # 验证：应该是[1,2,3,4]（共享序列）
    if order_ids_node == [1, 2, 3, 4]:
        print("  ✓ 节点层级正常工作：同节点不同IP共享序列")
    else:
        print(f"  ❌ 节点层级异常：order_id序列 = {order_ids_node}")

    # 比较两种粒度
    print("\n【对比两种粒度】")
    if order_ids_ip != order_ids_node:
        print("  ✓ 两种粒度产生不同的order_id分配，参数生效")
    else:
        print("  ❌ 两种粒度产生相同的order_id分配，参数可能未生效")


def test_network_key_construction():
    """测试Network中key构造是否受ORDERING_GRANULARITY影响"""
    print("\n" + "=" * 80)
    print("测试Network中key构造")
    print("=" * 80)

    from src.utils.components.flit import Flit

    # 创建两个flit：同节点不同IP
    flit1 = Flit(source=0, destination=4, path=[0, 4])
    flit1.source_original = 0
    flit1.destination_original = 4
    flit1.original_source_type = "gdma_0"
    flit1.source_type = "gdma_0"
    flit1.original_destination_type = "ddr_0"
    flit1.destination_type = "ddr_0"

    flit2 = Flit(source=0, destination=4, path=[0, 4])
    flit2.source_original = 0
    flit2.destination_original = 4
    flit2.original_source_type = "gdma_1"
    flit2.source_type = "gdma_1"
    flit2.original_destination_type = "ddr_0"
    flit2.destination_type = "ddr_0"

    # 模拟Network中的key构造逻辑
    def construct_network_key(flit, granularity, direction="TL"):
        src = flit.source_original if flit.source_original != -1 else flit.source
        dest = flit.destination_original if flit.destination_original != -1 else flit.destination

        if granularity == 0:  # IP层级
            src_type = flit.original_source_type if flit.original_source_type else flit.source_type
            dest_type = flit.original_destination_type if flit.original_destination_type else flit.destination_type
            return (src, src_type, dest, dest_type, direction)
        else:  # 节点层级
            return (src, dest, direction)

    print("\n【IP层级 (granularity=0)】")
    key1_ip = construct_network_key(flit1, granularity=0)
    key2_ip = construct_network_key(flit2, granularity=0)
    print(f"  gdma_0的key: {key1_ip}")
    print(f"  gdma_1的key: {key2_ip}")
    print(f"  key是否相同: {key1_ip == key2_ip}")
    if key1_ip != key2_ip:
        print("  ✓ IP层级：不同IP产生不同key")
    else:
        print("  ❌ IP层级：不同IP产生相同key（异常）")

    print("\n【节点层级 (granularity=1)】")
    key1_node = construct_network_key(flit1, granularity=1)
    key2_node = construct_network_key(flit2, granularity=1)
    print(f"  gdma_0的key: {key1_node}")
    print(f"  gdma_1的key: {key2_node}")
    print(f"  key是否相同: {key1_node == key2_node}")
    if key1_node == key2_node:
        print("  ✓ 节点层级：不同IP产生相同key")
    else:
        print("  ❌ 节点层级：不同IP产生不同key（异常）")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ORDERING_PRESERVATION_MODE和ORDERING_GRANULARITY配置诊断")
    print("=" * 80)

    try:
        # 1. 检查配置文件加载
        check_config_loading()

        # 2. 测试粒度参数实际效果
        test_granularity_effect()

        # 3. 测试Network中key构造
        test_network_key_construction()

        print("\n" + "=" * 80)
        print("诊断完成")
        print("=" * 80)
        print("\n如果所有测试都通过，说明参数配置正确。")
        print("如果你的仿真结果两个层级相同，可能是：")
        print("  1. 配置文件中ORDERING_GRANULARITY设置不正确")
        print("  2. 运行仿真时使用了错误的配置文件")
        print("  3. 代码中某处硬编码了granularity值")
        print("  4. 结果分析脚本没有区分不同粒度的结果")

    except Exception as e:
        print(f"\n❌ 诊断失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
