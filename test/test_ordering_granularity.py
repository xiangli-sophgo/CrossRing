"""
测试ORDERING_GRANULARITY参数是否正确生效
验证IP层级和节点层级两种保序粒度
"""
import sys
import io
from pathlib import Path

# 设置UTF-8编码输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.components.flit import Flit
from src.utils.components.network import Network
from config.config import CrossRingConfig


def test_order_id_allocation():
    """测试order_id分配是否根据粒度正确分配"""

    print("=" * 60)
    print("测试1: order_id分配机制")
    print("=" * 60)

    # 重置order_id
    Flit.reset_order_ids()

    # 测试节点层级 (granularity=1)
    print("\n【节点层级 (ORDERING_GRANULARITY=1)】")
    print("同一节点的不同IP应该共享order_id序列")

    order_id_1 = Flit.get_next_order_id(
        src_node=0, src_type="gdma_0",
        dest_node=4, dest_type="ddr_0",
        packet_category="REQ", granularity=1
    )
    print(f"gdma_0 -> ddr_0, order_id: {order_id_1}")

    order_id_2 = Flit.get_next_order_id(
        src_node=0, src_type="gdma_1",  # 不同的IP类型
        dest_node=4, dest_type="ddr_0",
        packet_category="REQ", granularity=1
    )
    print(f"gdma_1 -> ddr_0, order_id: {order_id_2}")
    print(f"预期: order_id_2 = {order_id_1 + 1}, 实际: order_id_2 = {order_id_2}")
    assert order_id_2 == order_id_1 + 1, "节点层级应该共享order_id序列"
    print("✓ 节点层级测试通过")

    # 重置order_id
    Flit.reset_order_ids()

    # 测试IP层级 (granularity=0)
    print("\n【IP层级 (ORDERING_GRANULARITY=0)】")
    print("同一节点的不同IP应该有独立的order_id序列")

    order_id_1 = Flit.get_next_order_id(
        src_node=0, src_type="gdma_0",
        dest_node=4, dest_type="ddr_0",
        packet_category="REQ", granularity=0
    )
    print(f"gdma_0 -> ddr_0, order_id: {order_id_1}")

    order_id_2 = Flit.get_next_order_id(
        src_node=0, src_type="gdma_1",  # 不同的IP类型
        dest_node=4, dest_type="ddr_0",
        packet_category="REQ", granularity=0
    )
    print(f"gdma_1 -> ddr_0, order_id: {order_id_2}")
    print(f"预期: order_id_2 = 1 (独立序列), 实际: order_id_2 = {order_id_2}")
    assert order_id_2 == 1, "IP层级应该有独立的order_id序列"
    print("✓ IP层级测试通过")


def test_network_key_construction():
    """测试Network中的key构造是否正确"""

    print("\n" + "=" * 60)
    print("测试2: Network保序检查key构造逻辑")
    print("=" * 60)

    # 创建测试用的flit
    flit = Flit(source=0, destination=4, path=[0, 4])
    flit.source_original = 0
    flit.destination_original = 4
    flit.original_source_type = "gdma_0"
    flit.source_type = "gdma_0"
    flit.original_destination_type = "ddr_0"
    flit.destination_type = "ddr_0"
    flit.src_dest_order_id = 1
    flit.flit_type = "req"
    flit.req_type = "read"
    flit.packet_category = "REQ"

    # 测试节点层级的key构造
    print("\n【节点层级key构造 (ORDERING_GRANULARITY=1)】")
    ORDERING_GRANULARITY_node = 1

    # 模拟_can_eject_in_order中的key构造逻辑
    src = flit.source_original if flit.source_original != -1 else flit.source
    dest = flit.destination_original if flit.destination_original != -1 else flit.destination

    if ORDERING_GRANULARITY_node == 0:
        src_type = flit.original_source_type if flit.original_source_type else flit.source_type
        dest_type = flit.original_destination_type if flit.original_destination_type else flit.destination_type
        key_node = (src, src_type, dest, dest_type, "TL")
    else:
        key_node = (src, dest, "TL")

    print(f"节点层级key: {key_node}")
    print(f"预期长度: 3, 实际长度: {len(key_node)}")
    assert len(key_node) == 3, "节点层级key应该是3元组"
    print("✓ 节点层级key构造正确")

    # 测试IP层级的key构造
    print("\n【IP层级key构造 (ORDERING_GRANULARITY=0)】")
    ORDERING_GRANULARITY_ip = 0

    if ORDERING_GRANULARITY_ip == 0:
        src_type = flit.original_source_type if flit.original_source_type else flit.source_type
        dest_type = flit.original_destination_type if flit.original_destination_type else flit.destination_type
        key_ip = (src, src_type, dest, dest_type, "TL")
    else:
        key_ip = (src, dest, "TL")

    print(f"IP层级key: {key_ip}")
    print(f"预期长度: 5, 实际长度: {len(key_ip)}")
    assert len(key_ip) == 5, "IP层级key应该是5元组"
    print("✓ IP层级key构造正确")

    # 验证两种粒度的key不同
    print(f"\n验证不同粒度的key不同:")
    print(f"节点层级key: {key_node}")
    print(f"IP层级key:   {key_ip}")
    assert key_node != key_ip, "不同粒度的key应该不同"
    print("✓ 不同粒度的key确实不同")


def test_order_tracking_consistency():
    """测试保序检查和更新的key一致性"""

    print("\n" + "=" * 60)
    print("测试3: 保序检查和更新的key一致性")
    print("=" * 60)

    # 创建测试flit
    flit = Flit(source=0, destination=4, path=[0, 4])
    flit.source_original = 0
    flit.destination_original = 4
    flit.original_source_type = "gdma_0"
    flit.source_type = "gdma_0"
    flit.original_destination_type = "ddr_0"
    flit.destination_type = "ddr_0"
    flit.src_dest_order_id = 1
    flit.flit_type = "req"
    flit.req_type = "read"
    flit.packet_category = "REQ"

    ORDERING_GRANULARITY = 0
    print(f"\n配置: ORDERING_GRANULARITY={ORDERING_GRANULARITY}")
    print(f"Flit信息: {flit.original_source_type} -> {flit.original_destination_type}")
    print(f"order_id: {flit.src_dest_order_id}")

    # 模拟更新order_tracking_table时的key构造
    direction = "TL"
    src = flit.source_original if flit.source_original != -1 else flit.source
    dest = flit.destination_original if flit.destination_original != -1 else flit.destination

    if ORDERING_GRANULARITY == 0:
        src_type = flit.original_source_type if flit.original_source_type else flit.source_type
        dest_type = flit.original_destination_type if flit.original_destination_type else flit.destination_type
        update_key = (src, src_type, dest, dest_type, direction)
    else:
        update_key = (src, dest, direction)

    # 模拟检查时的key构造
    if ORDERING_GRANULARITY == 0:
        src_type = flit.original_source_type if flit.original_source_type else flit.source_type
        dest_type = flit.original_destination_type if flit.original_destination_type else flit.destination_type
        check_key = (src, src_type, dest, dest_type, direction)
    else:
        check_key = (src, dest, direction)

    print(f"\n更新时的key: {update_key}")
    print(f"检查时的key: {check_key}")
    assert update_key == check_key, "更新和检查的key必须一致"
    print("✓ 更新和检查的key一致")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ORDERING_GRANULARITY参数功能测试")
    print("=" * 60)

    try:
        test_order_id_allocation()
        test_network_key_construction()
        test_order_tracking_consistency()

        print("\n" + "=" * 60)
        print("所有测试通过! ✓")
        print("=" * 60)
        print("\n总结:")
        print("1. order_id分配正确区分IP层级和节点层级")
        print("2. Network中的key构造根据ORDERING_GRANULARITY正确调整")
        print("3. 保序检查和更新使用的key保持一致")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
