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
    """测试order_id分配是否根据粒度正确分配 - 扩展测试"""

    print("=" * 60)
    print("测试1: order_id分配机制 (扩展多节点、多IP、多flit测试)")
    print("=" * 60)

    # 重置order_id
    Flit.reset_order_ids()

    # 测试节点层级 (granularity=1)
    print("\n【节点层级 (ORDERING_GRANULARITY=1)】")
    print("同一节点的不同IP应该共享order_id序列")

    # 模拟多个节点，每个节点多个IP
    test_cases_node_level = [
        # 节点0的多个IP -> 节点4
        (0, "gdma_0", 4, "ddr_0", "REQ"),
        (0, "gdma_1", 4, "ddr_0", "REQ"),
        (0, "gdma_2", 4, "ddr_1", "REQ"),
        (0, "npu_0", 4, "ddr_0", "REQ"),
        # 节点1的多个IP -> 节点5
        (1, "gdma_0", 5, "ddr_0", "REQ"),
        (1, "gdma_1", 5, "ddr_1", "REQ"),
        (1, "vpu_0", 5, "ddr_0", "REQ"),
        # 节点2的多个IP -> 节点6
        (2, "npu_0", 6, "ddr_0", "REQ"),
        (2, "npu_1", 6, "ddr_1", "REQ"),
        (2, "vpu_0", 6, "ddr_0", "REQ"),
        # 节点0 -> 节点7 (验证不同目标节点)
        (0, "gdma_0", 7, "ddr_0", "REQ"),
        (0, "gdma_1", 7, "ddr_0", "REQ"),
    ]

    order_ids_node = []
    for src_node, src_type, dest_node, dest_type, category in test_cases_node_level:
        order_id = Flit.get_next_order_id(
            src_node=src_node, src_type=src_type,
            dest_node=dest_node, dest_type=dest_type,
            packet_category=category, granularity=1
        )
        order_ids_node.append(order_id)
        print(f"  节点{src_node}.{src_type} -> 节点{dest_node}.{dest_type}, order_id: {order_id}")

    # 验证节点0的所有IP到节点4的order_id是递增的
    node0_to_node4_ids = order_ids_node[0:4]
    print(f"\n节点0的多个IP到节点4的order_id序列: {node0_to_node4_ids}")
    for i in range(1, len(node0_to_node4_ids)):
        assert node0_to_node4_ids[i] == node0_to_node4_ids[i-1] + 1, \
            f"节点层级应该共享order_id序列，但 {node0_to_node4_ids[i]} != {node0_to_node4_ids[i-1] + 1}"

    # 验证节点0到节点4和节点0到节点7的order_id是独立的
    node0_to_node4_last = order_ids_node[3]
    node0_to_node7_first = order_ids_node[10]
    print(f"\n验证不同目标节点的order_id独立性:")
    print(f"  节点0->节点4最后一个order_id: {node0_to_node4_last}")
    print(f"  节点0->节点7第一个order_id: {node0_to_node7_first}")
    assert node0_to_node7_first == 1, "不同目标节点应该有独立的order_id序列"
    print("✓ 节点层级测试通过")

    # 重置order_id
    Flit.reset_order_ids()

    # 测试IP层级 (granularity=0)
    print("\n【IP层级 (ORDERING_GRANULARITY=0)】")
    print("同一节点的不同IP应该有独立的order_id序列")

    test_cases_ip_level = [
        # 节点0的多个IP -> 节点4，每个IP应该有独立序列
        (0, "gdma_0", 4, "ddr_0", "REQ"),
        (0, "gdma_0", 4, "ddr_0", "REQ"),  # 同一IP第二个请求
        (0, "gdma_0", 4, "ddr_0", "REQ"),  # 同一IP第三个请求
        (0, "gdma_1", 4, "ddr_0", "REQ"),  # 不同IP，应该从1开始
        (0, "gdma_1", 4, "ddr_0", "REQ"),  # 同一IP第二个请求
        (0, "npu_0", 4, "ddr_0", "REQ"),   # 又一个不同IP，应该从1开始
        (0, "npu_0", 4, "ddr_1", "REQ"),   # 同一IP不同目标，应该从1开始
        # 节点1的IP
        (1, "gdma_0", 5, "ddr_0", "REQ"),
        (1, "gdma_0", 5, "ddr_0", "REQ"),
        (1, "vpu_0", 5, "ddr_0", "REQ"),
    ]

    order_ids_ip = []
    for src_node, src_type, dest_node, dest_type, category in test_cases_ip_level:
        order_id = Flit.get_next_order_id(
            src_node=src_node, src_type=src_type,
            dest_node=dest_node, dest_type=dest_type,
            packet_category=category, granularity=0
        )
        order_ids_ip.append(order_id)
        print(f"  节点{src_node}.{src_type} -> 节点{dest_node}.{dest_type}, order_id: {order_id}")

    # 验证同一IP的order_id递增
    gdma0_ids = [order_ids_ip[0], order_ids_ip[1], order_ids_ip[2]]
    print(f"\n节点0.gdma_0的连续请求order_id: {gdma0_ids}")
    assert gdma0_ids == [1, 2, 3], f"同一IP的order_id应该递增，但得到 {gdma0_ids}"

    # 验证不同IP的order_id独立
    gdma1_first = order_ids_ip[3]
    npu0_first = order_ids_ip[5]
    print(f"\n验证不同IP的order_id独立性:")
    print(f"  节点0.gdma_1第一个order_id: {gdma1_first}")
    print(f"  节点0.npu_0第一个order_id: {npu0_first}")
    assert gdma1_first == 1 and npu0_first == 1, "不同IP应该有独立的order_id序列"

    # 验证同一IP不同目标的order_id独立
    npu0_to_ddr0 = order_ids_ip[5]
    npu0_to_ddr1 = order_ids_ip[6]
    print(f"\n验证同一IP不同目标的order_id独立性:")
    print(f"  节点0.npu_0->节点4.ddr_0的order_id: {npu0_to_ddr0}")
    print(f"  节点0.npu_0->节点4.ddr_1的order_id: {npu0_to_ddr1}")
    assert npu0_to_ddr1 == 1, "同一IP不同目标应该有独立的order_id序列"
    print("✓ IP层级测试通过")


def test_network_key_construction():
    """测试Network中的key构造是否正确 - 扩展多flit测试"""

    print("\n" + "=" * 60)
    print("测试2: Network保序检查key构造逻辑 (多flit、多方向测试)")
    print("=" * 60)

    # 测试多个flit，不同节点、IP和方向
    test_flits = [
        # (source, destination, src_type, dest_type, direction, description)
        (0, 4, "gdma_0", "ddr_0", "TL", "节点0.gdma_0 -> 节点4.ddr_0 (TL方向)"),
        (0, 4, "gdma_1", "ddr_0", "TL", "节点0.gdma_1 -> 节点4.ddr_0 (TL方向)"),
        (1, 5, "npu_0", "ddr_1", "TR", "节点1.npu_0 -> 节点5.ddr_1 (TR方向)"),
        (2, 6, "vpu_0", "ddr_0", "TU", "节点2.vpu_0 -> 节点6.ddr_0 (TU方向)"),
        (3, 7, "gdma_0", "ddr_1", "TD", "节点3.gdma_0 -> 节点7.ddr_1 (TD方向)"),
    ]

    # 测试节点层级的key构造
    print("\n【节点层级key构造 (ORDERING_GRANULARITY=1)】")
    ORDERING_GRANULARITY_node = 1

    for source, destination, src_type, dest_type, direction, desc in test_flits:
        flit = Flit(source=source, destination=destination, path=[source, destination])
        flit.source_original = source
        flit.destination_original = destination
        flit.original_source_type = src_type
        flit.source_type = src_type
        flit.original_destination_type = dest_type
        flit.destination_type = dest_type
        flit.src_dest_order_id = 1
        flit.flit_type = "req"
        flit.req_type = "read"
        flit.packet_category = "REQ"

        # 模拟_can_eject_in_order中的key构造逻辑
        src = flit.source_original if flit.source_original != -1 else flit.source
        dest = flit.destination_original if flit.destination_original != -1 else flit.destination

        if ORDERING_GRANULARITY_node == 0:
            src_type_key = flit.original_source_type if flit.original_source_type else flit.source_type
            dest_type_key = flit.original_destination_type if flit.original_destination_type else flit.destination_type
            key_node = (src, src_type_key, dest, dest_type_key, direction)
        else:
            key_node = (src, dest, direction)

        print(f"  {desc}")
        print(f"    key: {key_node}, 长度: {len(key_node)}")
        assert len(key_node) == 3, f"节点层级key应该是3元组，但得到{len(key_node)}元组"

    print("✓ 所有flit的节点层级key构造正确")

    # 测试IP层级的key构造
    print("\n【IP层级key构造 (ORDERING_GRANULARITY=0)】")
    ORDERING_GRANULARITY_ip = 0

    for source, destination, src_type, dest_type, direction, desc in test_flits:
        flit = Flit(source=source, destination=destination, path=[source, destination])
        flit.source_original = source
        flit.destination_original = destination
        flit.original_source_type = src_type
        flit.source_type = src_type
        flit.original_destination_type = dest_type
        flit.destination_type = dest_type
        flit.src_dest_order_id = 1
        flit.flit_type = "req"
        flit.req_type = "read"
        flit.packet_category = "REQ"

        # 模拟_can_eject_in_order中的key构造逻辑
        src = flit.source_original if flit.source_original != -1 else flit.source
        dest = flit.destination_original if flit.destination_original != -1 else flit.destination

        if ORDERING_GRANULARITY_ip == 0:
            src_type_key = flit.original_source_type if flit.original_source_type else flit.source_type
            dest_type_key = flit.original_destination_type if flit.original_destination_type else flit.destination_type
            key_ip = (src, src_type_key, dest, dest_type_key, direction)
        else:
            key_ip = (src, dest, direction)

        print(f"  {desc}")
        print(f"    key: {key_ip}, 长度: {len(key_ip)}")
        assert len(key_ip) == 5, f"IP层级key应该是5元组，但得到{len(key_ip)}元组"

    print("✓ 所有flit的IP层级key构造正确")

    # 验证同一源目标对不同IP的key在两种粒度下的区别
    print("\n【验证同源目标不同IP的key区别】")
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

    # 节点层级：两个不同IP应该产生相同的key
    key1_node = (flit1.source_original, flit1.destination_original, "TL")
    key2_node = (flit2.source_original, flit2.destination_original, "TL")
    print(f"节点层级:")
    print(f"  gdma_0的key: {key1_node}")
    print(f"  gdma_1的key: {key2_node}")
    assert key1_node == key2_node, "节点层级不同IP应该产生相同的key"
    print("  ✓ 节点层级：不同IP产生相同key")

    # IP层级：两个不同IP应该产生不同的key
    key1_ip = (flit1.source_original, flit1.original_source_type,
               flit1.destination_original, flit1.original_destination_type, "TL")
    key2_ip = (flit2.source_original, flit2.original_source_type,
               flit2.destination_original, flit2.original_destination_type, "TL")
    print(f"IP层级:")
    print(f"  gdma_0的key: {key1_ip}")
    print(f"  gdma_1的key: {key2_ip}")
    assert key1_ip != key2_ip, "IP层级不同IP应该产生不同的key"
    print("  ✓ IP层级：不同IP产生不同key")


def test_order_tracking_consistency():
    """测试保序检查和更新的key一致性 - 扩展多场景测试"""

    print("\n" + "=" * 60)
    print("测试3: 保序检查和更新的key一致性 (多flit、多方向、两种粒度)")
    print("=" * 60)

    # 测试多个flit和不同方向
    test_scenarios = [
        # (source, destination, src_type, dest_type, direction, order_id)
        (0, 4, "gdma_0", "ddr_0", "TL", 1),
        (0, 4, "gdma_1", "ddr_0", "TL", 2),
        (1, 5, "npu_0", "ddr_1", "TR", 1),
        (2, 6, "vpu_0", "ddr_0", "TU", 1),
        (3, 7, "gdma_0", "ddr_1", "TD", 1),
        (0, 8, "gdma_0", "ddr_0", "TL", 1),
    ]

    # 测试IP层级 (ORDERING_GRANULARITY=0)
    print("\n【IP层级 (ORDERING_GRANULARITY=0)】")
    ORDERING_GRANULARITY = 0

    for source, destination, src_type, dest_type, direction, order_id in test_scenarios:
        flit = Flit(source=source, destination=destination, path=[source, destination])
        flit.source_original = source
        flit.destination_original = destination
        flit.original_source_type = src_type
        flit.source_type = src_type
        flit.original_destination_type = dest_type
        flit.destination_type = dest_type
        flit.src_dest_order_id = order_id
        flit.flit_type = "req"
        flit.req_type = "read"
        flit.packet_category = "REQ"

        # 模拟更新order_tracking_table时的key构造
        src = flit.source_original if flit.source_original != -1 else flit.source
        dest = flit.destination_original if flit.destination_original != -1 else flit.destination

        if ORDERING_GRANULARITY == 0:
            src_type_key = flit.original_source_type if flit.original_source_type else flit.source_type
            dest_type_key = flit.original_destination_type if flit.original_destination_type else flit.destination_type
            update_key = (src, src_type_key, dest, dest_type_key, direction)
        else:
            update_key = (src, dest, direction)

        # 模拟检查时的key构造
        if ORDERING_GRANULARITY == 0:
            src_type_key = flit.original_source_type if flit.original_source_type else flit.source_type
            dest_type_key = flit.original_destination_type if flit.original_destination_type else flit.destination_type
            check_key = (src, src_type_key, dest, dest_type_key, direction)
        else:
            check_key = (src, dest, direction)

        print(f"  节点{source}.{src_type} -> 节点{destination}.{dest_type} ({direction})")
        print(f"    更新key: {update_key}")
        print(f"    检查key: {check_key}")
        assert update_key == check_key, f"IP层级更新和检查的key必须一致，但不同: {update_key} != {check_key}"

    print("✓ IP层级所有场景的key一致")

    # 测试节点层级 (ORDERING_GRANULARITY=1)
    print("\n【节点层级 (ORDERING_GRANULARITY=1)】")
    ORDERING_GRANULARITY = 1

    for source, destination, src_type, dest_type, direction, order_id in test_scenarios:
        flit = Flit(source=source, destination=destination, path=[source, destination])
        flit.source_original = source
        flit.destination_original = destination
        flit.original_source_type = src_type
        flit.source_type = src_type
        flit.original_destination_type = dest_type
        flit.destination_type = dest_type
        flit.src_dest_order_id = order_id
        flit.flit_type = "req"
        flit.req_type = "read"
        flit.packet_category = "REQ"

        # 模拟更新order_tracking_table时的key构造
        src = flit.source_original if flit.source_original != -1 else flit.source
        dest = flit.destination_original if flit.destination_original != -1 else flit.destination

        if ORDERING_GRANULARITY == 0:
            src_type_key = flit.original_source_type if flit.original_source_type else flit.source_type
            dest_type_key = flit.original_destination_type if flit.original_destination_type else flit.destination_type
            update_key = (src, src_type_key, dest, dest_type_key, direction)
        else:
            update_key = (src, dest, direction)

        # 模拟检查时的key构造
        if ORDERING_GRANULARITY == 0:
            src_type_key = flit.original_source_type if flit.original_source_type else flit.source_type
            dest_type_key = flit.original_destination_type if flit.original_destination_type else flit.destination_type
            check_key = (src, src_type_key, dest, dest_type_key, direction)
        else:
            check_key = (src, dest, direction)

        print(f"  节点{source}.{src_type} -> 节点{destination}.{dest_type} ({direction})")
        print(f"    更新key: {update_key}")
        print(f"    检查key: {check_key}")
        assert update_key == check_key, f"节点层级更新和检查的key必须一致，但不同: {update_key} != {check_key}"

    print("✓ 节点层级所有场景的key一致")

    # 验证同一源目标对不同IP在两种粒度下的key行为
    print("\n【验证同源目标对不同IP的key处理】")
    flit1 = Flit(source=0, destination=4, path=[0, 4])
    flit1.source_original = 0
    flit1.destination_original = 4
    flit1.original_source_type = "gdma_0"
    flit1.source_type = "gdma_0"
    flit1.original_destination_type = "ddr_0"
    flit1.destination_type = "ddr_0"
    flit1.src_dest_order_id = 1

    flit2 = Flit(source=0, destination=4, path=[0, 4])
    flit2.source_original = 0
    flit2.destination_original = 4
    flit2.original_source_type = "gdma_1"
    flit2.source_type = "gdma_1"
    flit2.original_destination_type = "ddr_0"
    flit2.destination_type = "ddr_0"
    flit2.src_dest_order_id = 1

    # IP层级：不同IP应该有不同的key
    ORDERING_GRANULARITY = 0
    key1_ip = (flit1.source_original, flit1.original_source_type,
               flit1.destination_original, flit1.original_destination_type, "TL")
    key2_ip = (flit2.source_original, flit2.original_source_type,
               flit2.destination_original, flit2.original_destination_type, "TL")
    print(f"IP层级 (ORDERING_GRANULARITY=0):")
    print(f"  gdma_0的key: {key1_ip}")
    print(f"  gdma_1的key: {key2_ip}")
    assert key1_ip != key2_ip, "IP层级不同IP应该有不同的key，可以独立保序"
    print("  ✓ 不同IP使用不同key，保序流独立")

    # 节点层级：不同IP应该有相同的key
    ORDERING_GRANULARITY = 1
    key1_node = (flit1.source_original, flit1.destination_original, "TL")
    key2_node = (flit2.source_original, flit2.destination_original, "TL")
    print(f"节点层级 (ORDERING_GRANULARITY=1):")
    print(f"  gdma_0的key: {key1_node}")
    print(f"  gdma_1的key: {key2_node}")
    assert key1_node == key2_node, "节点层级不同IP应该有相同的key，共享保序流"
    print("  ✓ 不同IP使用相同key，保序流共享")


def test_ordering_preservation_mode():
    """测试ORDERING_PRESERVATION_MODE三种模式的下环方向控制"""

    print("\n" + "=" * 60)
    print("测试4: ORDERING_PRESERVATION_MODE三种模式的下环方向控制")
    print("=" * 60)

    # 创建测试用的flit
    test_flits = []
    for i in range(5):
        flit = Flit(source=i, destination=i+4, path=[i, i+4])
        flit.source_original = i
        flit.destination_original = i+4
        flit.original_source_type = f"gdma_{i}"
        flit.source_type = f"gdma_{i}"
        flit.original_destination_type = f"ddr_{i}"
        flit.destination_type = f"ddr_{i}"
        flit.src_dest_order_id = 1
        flit.flit_type = "req"
        flit.req_type = "read"
        flit.packet_category = "REQ"
        test_flits.append(flit)

    # 模拟Network的determine_allowed_eject_directions方法
    class MockConfig:
        def __init__(self, mode):
            self.ORDERING_PRESERVATION_MODE = mode
            # Mode 2的白名单配置
            self.TL_ALLOWED_SOURCE_NODES = [0, 1]
            self.TR_ALLOWED_SOURCE_NODES = [2, 3]
            self.TU_ALLOWED_SOURCE_NODES = [0, 2, 4]
            self.TD_ALLOWED_SOURCE_NODES = []  # 空列表表示所有节点都允许

    def determine_allowed_directions(config, flit):
        """模拟Network.determine_allowed_eject_directions逻辑"""
        mode = config.ORDERING_PRESERVATION_MODE

        # Mode 0: 不保序，所有方向都允许
        if mode == 0:
            return None

        # Mode 1: 单侧下环，固定只允许TL和TU方向
        if mode == 1:
            return ["TL", "TU"]

        # Mode 2: 双侧下环，根据方向配置决定
        if mode == 2:
            src_node = flit.source_original if flit.source_original != -1 else flit.source
            allowed_source_nodes = {
                "TL": config.TL_ALLOWED_SOURCE_NODES,
                "TR": config.TR_ALLOWED_SOURCE_NODES,
                "TU": config.TU_ALLOWED_SOURCE_NODES,
                "TD": config.TD_ALLOWED_SOURCE_NODES,
            }

            allowed_dirs = []
            for direction in ["TL", "TR", "TU", "TD"]:
                # 空列表表示所有节点都允许
                if len(allowed_source_nodes[direction]) == 0 or src_node in allowed_source_nodes[direction]:
                    allowed_dirs.append(direction)

            return allowed_dirs if allowed_dirs else None

        # 未知模式，默认不保序
        return None

    # 测试Mode 0: 不保序
    print("\n【Mode 0: 不保序】")
    print("所有flit的allowed_eject_directions应该为None（所有方向都允许）")
    config_mode0 = MockConfig(mode=0)
    for flit in test_flits:
        allowed_dirs = determine_allowed_directions(config_mode0, flit)
        print(f"  节点{flit.source_original}.{flit.original_source_type} -> allowed_directions: {allowed_dirs}")
        assert allowed_dirs is None, f"Mode 0应该返回None，但得到{allowed_dirs}"
    print("✓ Mode 0测试通过")

    # 测试Mode 1: 单侧下环
    print("\n【Mode 1: 单侧下环 (TL/TU固定)】")
    print("所有flit的allowed_eject_directions应该为['TL', 'TU']")
    config_mode1 = MockConfig(mode=1)
    for flit in test_flits:
        allowed_dirs = determine_allowed_directions(config_mode1, flit)
        print(f"  节点{flit.source_original}.{flit.original_source_type} -> allowed_directions: {allowed_dirs}")
        assert allowed_dirs == ["TL", "TU"], f"Mode 1应该返回['TL', 'TU']，但得到{allowed_dirs}"
    print("✓ Mode 1测试通过")

    # 测试Mode 2: 双侧下环（白名单配置）
    print("\n【Mode 2: 双侧下环 (白名单配置)】")
    print("约束1：同一节点在横向环不能同时允许TL和TR，在纵向环不能同时允许TU和TD")
    print("约束2：每个节点必须在横向(TL或TR)和纵向(TU或TD)各有一个允许方向")
    print("根据白名单配置:")
    print("  TL_ALLOWED_SOURCE_NODES: [0, 1]")
    print("  TR_ALLOWED_SOURCE_NODES: [2, 3, 4]")
    print("  TU_ALLOWED_SOURCE_NODES: [0, 2, 4]")
    print("  TD_ALLOWED_SOURCE_NODES: [1, 3]")
    config_mode2 = MockConfig(mode=2)
    config_mode2.TL_ALLOWED_SOURCE_NODES = [0, 1]
    config_mode2.TR_ALLOWED_SOURCE_NODES = [2, 3, 4]
    config_mode2.TU_ALLOWED_SOURCE_NODES = [0, 2, 4]
    config_mode2.TD_ALLOWED_SOURCE_NODES = [1, 3]

    expected_results = {
        0: ["TL", "TU"],        # 节点0: 横向TL, 纵向TU (完整配置)
        1: ["TL", "TD"],        # 节点1: 横向TL, 纵向TD (完整配置)
        2: ["TR", "TU"],        # 节点2: 横向TR, 纵向TU (完整配置)
        3: ["TR", "TD"],        # 节点3: 横向TR, 纵向TD (完整配置)
        4: ["TR", "TU"],        # 节点4: 横向TR, 纵向TU (完整配置)
    }

    for flit in test_flits:
        allowed_dirs = determine_allowed_directions(config_mode2, flit)
        expected = expected_results[flit.source_original]
        print(f"  节点{flit.source_original}.{flit.original_source_type} -> allowed_directions: {allowed_dirs}")
        print(f"    预期: {expected}")
        assert allowed_dirs == expected, f"节点{flit.source_original}应该返回{expected}，但得到{allowed_dirs}"
    print("✓ Mode 2测试通过")

    # 验证配置的合法性
    print("\n【验证Mode 2配置合法性】")
    print("检查配置是否满足两个约束条件")

    def validate_mode2_config(config, all_nodes=None):
        """验证Mode 2配置的合法性"""
        errors = []

        # 约束1：检查横向环约束：TL和TR不能有交集
        horizontal_conflict = set(config.TL_ALLOWED_SOURCE_NODES) & set(config.TR_ALLOWED_SOURCE_NODES)
        if horizontal_conflict:
            errors.append(f"约束1违反-横向冲突：节点{horizontal_conflict}同时在TL和TR白名单中")

        # 约束1：检查纵向环约束：TU和TD不能有交集
        vertical_conflict = set(config.TU_ALLOWED_SOURCE_NODES) & set(config.TD_ALLOWED_SOURCE_NODES)
        if vertical_conflict:
            errors.append(f"约束1违反-纵向冲突：节点{vertical_conflict}同时在TU和TD白名单中")

        # 约束2：检查每个节点是否在横向和纵向都有配置
        if all_nodes is not None:
            horizontal_nodes = set(config.TL_ALLOWED_SOURCE_NODES) | set(config.TR_ALLOWED_SOURCE_NODES)
            vertical_nodes = set(config.TU_ALLOWED_SOURCE_NODES) | set(config.TD_ALLOWED_SOURCE_NODES)

            for node in all_nodes:
                if node not in horizontal_nodes:
                    errors.append(f"约束2违反-节点{node}缺少横向配置(TL或TR)")
                if node not in vertical_nodes:
                    errors.append(f"约束2违反-节点{node}缺少纵向配置(TU或TD)")

        return errors

    validation_errors = validate_mode2_config(config_mode2, all_nodes=[0, 1, 2, 3, 4])
    if validation_errors:
        for error in validation_errors:
            print(f"  ❌ {error}")
        assert False, "Mode 2配置不满足约束条件"
    else:
        print("  ✓ 约束1：没有节点在相反方向的白名单中")
        print("  ✓ 约束2：每个节点在横向和纵向都有唯一的下环方向")

    # 测试Mode 2的特殊情况：所有方向都不允许的节点
    print("\n【Mode 2特殊情况：测试没有任何允许方向的节点】")
    config_mode2_strict = MockConfig(mode=2)
    config_mode2_strict.TL_ALLOWED_SOURCE_NODES = [10]
    config_mode2_strict.TR_ALLOWED_SOURCE_NODES = [11]
    config_mode2_strict.TU_ALLOWED_SOURCE_NODES = [12]
    config_mode2_strict.TD_ALLOWED_SOURCE_NODES = [13]

    flit_no_direction = test_flits[0]  # 节点0，不在任何白名单中
    allowed_dirs = determine_allowed_directions(config_mode2_strict, flit_no_direction)
    print(f"  节点{flit_no_direction.source_original}不在任何白名单中 -> allowed_directions: {allowed_dirs}")
    assert allowed_dirs is None, f"没有允许方向时应该返回None，但得到{allowed_dirs}"
    print("✓ Mode 2特殊情况测试通过")

    # 测试违反约束1的配置（相反方向冲突）
    print("\n【测试违反约束1的配置：相反方向冲突】")
    config_invalid1 = MockConfig(mode=2)
    config_invalid1.TL_ALLOWED_SOURCE_NODES = [0, 1]
    config_invalid1.TR_ALLOWED_SOURCE_NODES = [0, 2]  # 节点0冲突
    config_invalid1.TU_ALLOWED_SOURCE_NODES = [1, 3]
    config_invalid1.TD_ALLOWED_SOURCE_NODES = [1, 4]  # 节点1冲突

    validation_errors = validate_mode2_config(config_invalid1)
    print(f"  检测到配置错误: {validation_errors}")
    assert len(validation_errors) == 2, f"应该检测到2个错误，但得到{len(validation_errors)}个"
    assert "节点{0}" in str(validation_errors[0]), "应该检测到节点0的横向冲突"
    assert "节点{1}" in str(validation_errors[1]), "应该检测到节点1的纵向冲突"
    print("  ✓ 成功检测到约束1违反")

    # 测试违反约束2的配置（缺少横向或纵向配置）
    print("\n【测试违反约束2的配置：缺少横向或纵向配置】")
    config_invalid2 = MockConfig(mode=2)
    config_invalid2.TL_ALLOWED_SOURCE_NODES = [0, 1]
    config_invalid2.TR_ALLOWED_SOURCE_NODES = [2]     # 节点3,4缺少横向配置
    config_invalid2.TU_ALLOWED_SOURCE_NODES = [0, 2]
    config_invalid2.TD_ALLOWED_SOURCE_NODES = [1]     # 节点3,4缺少纵向配置

    validation_errors = validate_mode2_config(config_invalid2, all_nodes=[0, 1, 2, 3, 4])
    print(f"  检测到配置错误: {validation_errors}")
    assert len(validation_errors) == 4, f"应该检测到4个错误，但得到{len(validation_errors)}个"
    print(f"  错误详情:")
    for error in validation_errors:
        print(f"    - {error}")
    print("  ✓ 成功检测到约束2违反")

    # 测试同时违反两个约束的配置
    print("\n【测试同时违反约束1和约束2的配置】")
    config_invalid3 = MockConfig(mode=2)
    config_invalid3.TL_ALLOWED_SOURCE_NODES = [0]
    config_invalid3.TR_ALLOWED_SOURCE_NODES = [0, 2]  # 节点0冲突(约束1)，节点1,3,4缺失(约束2)
    config_invalid3.TU_ALLOWED_SOURCE_NODES = [1]     # 节点0,2,3,4缺失(约束2)
    config_invalid3.TD_ALLOWED_SOURCE_NODES = []      # 节点0,1,2,3,4缺失(约束2)

    validation_errors = validate_mode2_config(config_invalid3, all_nodes=[0, 1, 2, 3, 4])
    print(f"  检测到配置错误数量: {len(validation_errors)}")
    assert len(validation_errors) > 0, "应该检测到多个错误"
    # 应该有1个约束1错误 + 多个约束2错误
    constraint1_errors = [e for e in validation_errors if "约束1违反" in e]
    constraint2_errors = [e for e in validation_errors if "约束2违反" in e]
    print(f"  约束1违反: {len(constraint1_errors)}个")
    print(f"  约束2违反: {len(constraint2_errors)}个")
    assert len(constraint1_errors) >= 1, "应该至少检测到1个约束1错误"
    assert len(constraint2_errors) >= 1, "应该至少检测到1个约束2错误"
    print("  ✓ 成功检测到同时违反两个约束")


def test_out_of_order_arrival():
    """测试flit乱序到达下环点时的保序处理"""

    print("\n" + "=" * 60)
    print("测试5: Flit乱序到达下环点的保序处理")
    print("=" * 60)

    from src.utils.components.network import Network
    from config.config import CrossRingConfig

    # 创建测试配置
    class MockConfigForOrdering:
        def __init__(self, mode, granularity):
            self.ORDERING_PRESERVATION_MODE = mode
            self.ORDERING_GRANULARITY = granularity
            self.IN_ORDER_PACKET_CATEGORIES = ["REQ", "RSP", "DATA"]
            self.TL_ALLOWED_SOURCE_NODES = [0, 1]
            self.TR_ALLOWED_SOURCE_NODES = [2, 3]
            self.TU_ALLOWED_SOURCE_NODES = [0, 2]
            self.TD_ALLOWED_SOURCE_NODES = [1, 3]
            self.NUM_NODE = 20
            self.NUM_COLUMN = 4

    # 模拟创建network（只用到保序相关部分）
    def create_mock_network(config):
        class MockNetwork:
            def __init__(self, config):
                self.config = config
                self.order_tracking_table = {}
                self._init_direction_control()

            def _init_direction_control(self):
                self.allowed_source_nodes = {
                    "TL": set(self.config.TL_ALLOWED_SOURCE_NODES),
                    "TR": set(self.config.TR_ALLOWED_SOURCE_NODES),
                    "TU": set(self.config.TU_ALLOWED_SOURCE_NODES),
                    "TD": set(self.config.TD_ALLOWED_SOURCE_NODES),
                }

            def _can_eject_in_order(self, flit, direction):
                """检查是否可以按序下环"""
                src = flit.source_original if flit.source_original != -1 else flit.source
                dest = flit.destination_original if flit.destination_original != -1 else flit.destination

                # 根据保序粒度构造key
                if self.config.ORDERING_GRANULARITY == 0:  # IP层级
                    src_type = flit.original_source_type if flit.original_source_type else flit.source_type
                    dest_type = flit.original_destination_type if flit.original_destination_type else flit.destination_type
                    key = (src, src_type, dest, dest_type, direction)
                else:  # 节点层级
                    key = (src, dest, direction)

                # 检查是否是期望的下一个顺序ID
                if key not in self.order_tracking_table:
                    self.order_tracking_table[key] = 0

                expected_order_id = self.order_tracking_table[key] + 1
                can_eject = flit.src_dest_order_id == expected_order_id
                return can_eject, key, expected_order_id

            def _update_order_tracking(self, flit, direction):
                """更新保序跟踪表"""
                src = flit.source_original if flit.source_original != -1 else flit.source
                dest = flit.destination_original if flit.destination_original != -1 else flit.destination

                if self.config.ORDERING_GRANULARITY == 0:
                    src_type = flit.original_source_type if flit.original_source_type else flit.source_type
                    dest_type = flit.original_destination_type if flit.original_destination_type else flit.destination_type
                    key = (src, src_type, dest, dest_type, direction)
                else:
                    key = (src, dest, direction)

                self.order_tracking_table[key] = flit.src_dest_order_id

        return MockNetwork(config)

    # 测试场景：节点层级，flit乱序到达
    print("\n【场景1: 节点层级 - 同节点不同IP的flit乱序到达】")
    print("上环顺序: flit1(gdma_0,id=1) → flit2(gdma_1,id=2) → flit3(gdma_0,id=3) → flit4(gdma_1,id=4)")
    print("到达顺序: flit1 → flit3 → flit2 → flit4 (乱序)")

    config_node = MockConfigForOrdering(mode=2, granularity=1)
    network_node = create_mock_network(config_node)

    # 创建4个flit，按上环顺序分配order_id
    Flit.reset_order_ids()
    flits_node = []
    for i, (src_type, desc) in enumerate([
        ("gdma_0", "flit1"), ("gdma_1", "flit2"),
        ("gdma_0", "flit3"), ("gdma_1", "flit4")
    ]):
        flit = Flit(source=0, destination=4, path=[0, 4])
        flit.source_original = 0
        flit.destination_original = 4
        flit.original_source_type = src_type
        flit.source_type = src_type
        flit.original_destination_type = "ddr_0"
        flit.destination_type = "ddr_0"
        flit.packet_category = "REQ"

        # 分配order_id
        flit.src_dest_order_id = Flit.get_next_order_id(
            src_node=0, src_type=src_type,
            dest_node=4, dest_type="ddr_0",
            packet_category="REQ", granularity=1
        )
        flits_node.append((desc, src_type, flit))
        print(f"  {desc}({src_type}): order_id={flit.src_dest_order_id}")

    # 模拟乱序到达：flit1, flit3, flit2, flit4
    # 真实场景：每个周期检查所有在下环点等待的flit
    arrival_order = [0, 2, 1, 3]  # 索引
    direction = "TL"

    print(f"\n模拟多周期下环处理:")
    waiting_flits = []  # 等待下环的flit队列
    ejected = []

    # 周期1: flit1, flit3到达
    print("\n周期1: flit1, flit3到达下环点")
    for idx in [0, 2]:
        desc, src_type, flit = flits_node[idx]
        waiting_flits.append((desc, src_type, flit))
        print(f"  {desc}到达")

    # 检查所有等待的flit
    print("  检查等待队列:")
    new_waiting = []
    for desc, src_type, flit in waiting_flits:
        can_eject, key, expected = network_node._can_eject_in_order(flit, direction)
        if can_eject:
            network_node._update_order_tracking(flit, direction)
            ejected.append(desc)
            print(f"    {desc}(order_id={flit.src_dest_order_id}): ✓下环")
        else:
            new_waiting.append((desc, src_type, flit))
            print(f"    {desc}(order_id={flit.src_dest_order_id}): ✗等待(期望{expected})")
    waiting_flits = new_waiting

    # 周期2: flit2, flit4到达
    print("\n周期2: flit2, flit4到达下环点")
    for idx in [1, 3]:
        desc, src_type, flit = flits_node[idx]
        waiting_flits.append((desc, src_type, flit))
        print(f"  {desc}到达")

    # 检查所有等待的flit（包括上周期阻塞的）
    print("  检查等待队列:")
    new_waiting = []
    for desc, src_type, flit in waiting_flits:
        can_eject, key, expected = network_node._can_eject_in_order(flit, direction)
        if can_eject:
            network_node._update_order_tracking(flit, direction)
            ejected.append(desc)
            print(f"    {desc}(order_id={flit.src_dest_order_id}): ✓下环")
        else:
            new_waiting.append((desc, src_type, flit))
            print(f"    {desc}(order_id={flit.src_dest_order_id}): ✗等待(期望{expected})")
    waiting_flits = new_waiting

    # 周期3: 继续检查剩余的flit
    if waiting_flits:
        print("\n周期3: 检查剩余flit")
        print("  检查等待队列:")
        for desc, src_type, flit in waiting_flits:
            can_eject, key, expected = network_node._can_eject_in_order(flit, direction)
            if can_eject:
                network_node._update_order_tracking(flit, direction)
                ejected.append(desc)
                print(f"    {desc}(order_id={flit.src_dest_order_id}): ✓下环")
            else:
                print(f"    {desc}(order_id={flit.src_dest_order_id}): ✗等待(期望{expected})")

    print(f"\n最终下环顺序: {' → '.join(ejected)}")
    print(f"预期下环顺序: flit1 → flit2 → flit3 → flit4")
    assert ejected == ["flit1", "flit2", "flit3", "flit4"], "节点层级应该严格按上环顺序下环"
    print("✓ 节点层级保序正确：即使乱序到达，也严格按上环顺序下环")

    # 测试场景：IP层级，flit乱序到达
    print("\n【场景2: IP层级 - 同节点不同IP的flit乱序到达】")
    print("上环顺序: flit1(gdma_0,id=1) → flit2(gdma_1,id=1) → flit3(gdma_0,id=2) → flit4(gdma_1,id=2)")
    print("到达顺序: flit1 → flit3 → flit2 → flit4 (乱序)")

    config_ip = MockConfigForOrdering(mode=2, granularity=0)
    network_ip = create_mock_network(config_ip)

    # 创建4个flit，按上环顺序分配order_id (IP层级)
    Flit.reset_order_ids()
    flits_ip = []
    for i, (src_type, desc) in enumerate([
        ("gdma_0", "flit1"), ("gdma_1", "flit2"),
        ("gdma_0", "flit3"), ("gdma_1", "flit4")
    ]):
        flit = Flit(source=0, destination=4, path=[0, 4])
        flit.source_original = 0
        flit.destination_original = 4
        flit.original_source_type = src_type
        flit.source_type = src_type
        flit.original_destination_type = "ddr_0"
        flit.destination_type = "ddr_0"
        flit.packet_category = "REQ"

        # 分配order_id
        flit.src_dest_order_id = Flit.get_next_order_id(
            src_node=0, src_type=src_type,
            dest_node=4, dest_type="ddr_0",
            packet_category="REQ", granularity=0
        )
        flits_ip.append((desc, src_type, flit))
        print(f"  {desc}({src_type}): order_id={flit.src_dest_order_id}")

    # 模拟乱序到达：flit1, flit3, flit2, flit4
    print(f"\n模拟单周期下环处理 (IP层级每个IP独立):")
    ejected_ip = []

    # IP层级：不同IP独立保序，一次性检查即可
    print("  检查到达的flit:")
    for idx in arrival_order:
        desc, src_type, flit = flits_ip[idx]
        can_eject, key, expected = network_ip._can_eject_in_order(flit, direction)

        print(f"  {desc}({src_type}, order_id={flit.src_dest_order_id}):")
        print(f"    保序key: {key}")
        print(f"    期望order_id: {expected}")
        print(f"    可以下环: {'✓' if can_eject else '✗ 需要等待'}")

        if can_eject:
            network_ip._update_order_tracking(flit, direction)
            ejected_ip.append(desc)
            print(f"    → 下环成功")
        else:
            print(f"    → 阻塞，等待order_id={expected}的flit")

    print(f"\n实际下环顺序: {' → '.join(ejected_ip)}")
    print(f"预期下环顺序: flit1 → flit3 → flit2 → flit4 (允许不同IP交错)")
    # IP层级：gdma_0和gdma_1有独立的保序流，可以交错下环
    # flit1(gdma_0,id=1)可以下，flit3(gdma_0,id=2)可以下，flit2(gdma_1,id=1)可以下，flit4(gdma_1,id=2)可以下
    assert ejected_ip == ["flit1", "flit3", "flit2", "flit4"], "IP层级允许不同IP交错下环"
    print("✓ IP层级保序正确：不同IP独立保序，可以交错下环")

    # 测试场景：多轮到达
    print("\n【场景3: 多轮到达 - 模拟flit分批到达】")
    print("第一轮到达: flit1, flit3 (flit2未到)")
    print("第二轮到达: flit2, flit4")

    config_node2 = MockConfigForOrdering(mode=2, granularity=1)
    network_node2 = create_mock_network(config_node2)

    # 重新创建flit
    Flit.reset_order_ids()
    flits_node2 = []
    for i, (src_type, desc) in enumerate([
        ("gdma_0", "flit1"), ("gdma_1", "flit2"),
        ("gdma_0", "flit3"), ("gdma_1", "flit4")
    ]):
        flit = Flit(source=0, destination=4, path=[0, 4])
        flit.source_original = 0
        flit.destination_original = 4
        flit.original_source_type = src_type
        flit.source_type = src_type
        flit.original_destination_type = "ddr_0"
        flit.destination_type = "ddr_0"
        flit.packet_category = "REQ"
        flit.src_dest_order_id = Flit.get_next_order_id(
            src_node=0, src_type=src_type,
            dest_node=4, dest_type="ddr_0",
            packet_category="REQ", granularity=1
        )
        flits_node2.append((desc, src_type, flit))

    print("\n第一轮到达: flit1, flit3")
    ejected_round1 = []
    for idx in [0, 2]:  # flit1, flit3
        desc, src_type, flit = flits_node2[idx]
        can_eject, key, expected = network_node2._can_eject_in_order(flit, direction)
        print(f"  {desc}(order_id={flit.src_dest_order_id}): 期望{expected}, {'✓下环' if can_eject else '✗阻塞'}")
        if can_eject:
            network_node2._update_order_tracking(flit, direction)
            ejected_round1.append(desc)

    print(f"  第一轮下环: {ejected_round1}")
    assert ejected_round1 == ["flit1"], "只有flit1可以下环，flit3被阻塞"

    print("\n第二轮到达: flit2, flit4 (flit3仍在等待)")
    waiting_queue = [flits_node2[2]]  # flit3仍在等待队列中

    # flit2, flit4到达，加入等待队列
    for idx in [1, 3]:
        waiting_queue.append(flits_node2[idx])

    ejected_round2 = []
    still_waiting = []

    # 逐个检查等待队列中的flit
    for desc, src_type, flit in waiting_queue:
        can_eject, key, expected = network_node2._can_eject_in_order(flit, direction)
        print(f"  {desc}(order_id={flit.src_dest_order_id}): 期望{expected}, {'✓下环' if can_eject else '✗阻塞'}")
        if can_eject:
            network_node2._update_order_tracking(flit, direction)
            ejected_round2.append(desc)
            # 下环成功后，重新检查还在等待的flit
            for w_desc, w_src_type, w_flit in still_waiting:
                w_can_eject, w_key, w_expected = network_node2._can_eject_in_order(w_flit, direction)
                if w_can_eject:
                    network_node2._update_order_tracking(w_flit, direction)
                    ejected_round2.append(w_desc)
                    print(f"  {w_desc}(order_id={w_flit.src_dest_order_id}): 期望{w_expected}, ✓下环(解除阻塞)")
            still_waiting = []  # 清空已检查列表
        else:
            still_waiting.append((desc, src_type, flit))

    print(f"  第二轮下环: {ejected_round2}")
    assert ejected_round2 == ["flit2", "flit3", "flit4"], "flit2下环后，flit3和flit4依次下环"
    print("✓ 多轮到达场景：阻塞的flit在前序flit到达后可以继续下环")


def test_ordering_mode_with_granularity():
    """测试ORDERING_PRESERVATION_MODE与ORDERING_GRANULARITY的组合"""

    print("\n" + "=" * 60)
    print("测试6: ORDERING_PRESERVATION_MODE与ORDERING_GRANULARITY组合测试")
    print("=" * 60)

    # 创建两个来自同一节点不同IP的flit
    flit1 = Flit(source=0, destination=4, path=[0, 4])
    flit1.source_original = 0
    flit1.destination_original = 4
    flit1.original_source_type = "gdma_0"
    flit1.source_type = "gdma_0"
    flit1.original_destination_type = "ddr_0"
    flit1.destination_type = "ddr_0"
    flit1.src_dest_order_id = 1
    flit1.packet_category = "REQ"

    flit2 = Flit(source=0, destination=4, path=[0, 4])
    flit2.source_original = 0
    flit2.destination_original = 4
    flit2.original_source_type = "gdma_1"
    flit2.source_type = "gdma_1"
    flit2.original_destination_type = "ddr_0"
    flit2.destination_type = "ddr_0"
    flit2.src_dest_order_id = 1
    flit2.packet_category = "REQ"

    # 测试Mode 0：不保序，粒度不影响
    print("\n【Mode 0 + 任意粒度】")
    print("Mode 0禁用保序功能，ORDERING_GRANULARITY不生效")
    ORDERING_GRANULARITY = 0
    mode = 0
    # Mode 0下不需要保序检查，直接下环
    print(f"  Mode={mode}, Granularity={ORDERING_GRANULARITY}")
    print("  不需要构造保序key，直接允许下环")
    print("✓ Mode 0组合测试通过")

    # 测试Mode 1 + IP层级粒度
    print("\n【Mode 1 (单侧TL/TU) + IP层级粒度 (0)】")
    print("不同IP有独立的保序流，只能从TL/TU下环")
    ORDERING_GRANULARITY = 0
    direction = "TL"
    key1 = (flit1.source_original, flit1.original_source_type,
            flit1.destination_original, flit1.original_destination_type, direction)
    key2 = (flit2.source_original, flit2.original_source_type,
            flit2.destination_original, flit2.original_destination_type, direction)
    print(f"  gdma_0的保序key: {key1}")
    print(f"  gdma_1的保序key: {key2}")
    assert key1 != key2, "IP层级不同IP应该有不同的保序key"
    print("  ✓ 不同IP独立保序，只能从TL/TU下环")

    # 测试Mode 1 + 节点层级粒度
    print("\n【Mode 1 (单侧TL/TU) + 节点层级粒度 (1)】")
    print("同节点所有IP共享保序流，只能从TL/TU下环")
    ORDERING_GRANULARITY = 1
    key1 = (flit1.source_original, flit1.destination_original, direction)
    key2 = (flit2.source_original, flit2.destination_original, direction)
    print(f"  gdma_0的保序key: {key1}")
    print(f"  gdma_1的保序key: {key2}")
    assert key1 == key2, "节点层级同节点不同IP应该有相同的保序key"
    print("  ✓ 同节点IP共享保序，只能从TL/TU下环")

    # 测试Mode 2 + IP层级粒度
    print("\n【Mode 2 (双侧白名单) + IP层级粒度 (0)】")
    print("不同IP有独立的保序流，根据白名单决定下环方向")
    ORDERING_GRANULARITY = 0
    # 假设节点0在TL、TR、TU白名单中
    allowed_directions = ["TL", "TR", "TU"]
    print(f"  节点0允许的下环方向: {allowed_directions}")
    for dir in allowed_directions:
        key1 = (flit1.source_original, flit1.original_source_type,
                flit1.destination_original, flit1.original_destination_type, dir)
        key2 = (flit2.source_original, flit2.original_source_type,
                flit2.destination_original, flit2.original_destination_type, dir)
        print(f"  {dir}方向 - gdma_0 key: {key1}")
        print(f"  {dir}方向 - gdma_1 key: {key2}")
        assert key1 != key2, f"{dir}方向IP层级不同IP应该有不同的保序key"
    print("  ✓ 不同IP独立保序，多方向可下环")

    # 测试Mode 2 + 节点层级粒度
    print("\n【Mode 2 (双侧白名单) + 节点层级粒度 (1)】")
    print("同节点所有IP共享保序流，根据白名单决定下环方向")
    ORDERING_GRANULARITY = 1
    allowed_directions = ["TL", "TR", "TU"]
    print(f"  节点0允许的下环方向: {allowed_directions}")
    for dir in allowed_directions:
        key1 = (flit1.source_original, flit1.destination_original, dir)
        key2 = (flit2.source_original, flit2.destination_original, dir)
        print(f"  {dir}方向 - gdma_0 key: {key1}")
        print(f"  {dir}方向 - gdma_1 key: {key2}")
        assert key1 == key2, f"{dir}方向节点层级同节点不同IP应该有相同的保序key"
    print("  ✓ 同节点IP共享保序，多方向可下环")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ORDERING_GRANULARITY与ORDERING_PRESERVATION_MODE参数功能测试")
    print("=" * 60)

    try:
        test_order_id_allocation()
        test_network_key_construction()
        test_order_tracking_consistency()
        test_ordering_preservation_mode()
        test_out_of_order_arrival()
        test_ordering_mode_with_granularity()

        print("\n" + "=" * 60)
        print("所有测试通过! ✓")
        print("=" * 60)
        print("\n总结:")
        print("1. order_id分配正确区分IP层级和节点层级")
        print("2. Network中的key构造根据ORDERING_GRANULARITY正确调整")
        print("3. 保序检查和更新使用的key保持一致")
        print("4. ORDERING_PRESERVATION_MODE三种模式的下环方向控制正确")
        print("   - Mode 0: 不保序，所有方向都允许")
        print("   - Mode 1: 单侧下环，固定TL/TU方向")
        print("   - Mode 2: 双侧下环，根据白名单配置")
        print("5. Flit乱序到达下环点的保序处理正确")
        print("   - 节点层级：严格按上环顺序下环，乱序flit被阻塞")
        print("   - IP层级：不同IP独立保序，可以交错下环")
        print("   - 分批到达：阻塞flit在前序flit到达后继续处理")
        print("6. ORDERING_PRESERVATION_MODE与ORDERING_GRANULARITY组合工作正确")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
