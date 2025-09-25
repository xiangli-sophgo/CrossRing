#!/usr/bin/env python3
"""
测试地址hash映射的正确性
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'traffic_process'))

from traffic_processor import AddressHasher

def test_address_mapping():
    """测试地址映射功能"""
    hasher = AddressHasher()

    print("=" * 60)
    print("地址Hash映射测试")
    print("=" * 60)

    # 测试256b交织
    print("=" * 60)
    print("256b交织测试")
    print("=" * 60)
    hasher_256 = AddressHasher(interleave_size=256)

    # 测试用例：各个内存区域的边界地址
    test_cases = [
        # 32CH_shared 测试
        ("32CH_shared - 起始", 0x80000000),
        ("32CH_shared - 中间", 0xC0000000),
        ("32CH_shared - 结束", 0xffffffff),

        # 16CH_shared 区域0 测试
        ("16CH_shared[0] - 起始", 0x100000000),
        ("16CH_shared[0] - 中间", 0x200000000),
        ("16CH_shared[0] - 结束", 0x2ffffffff),

        # 16CH_shared 区域1 测试
        ("16CH_shared[1] - 起始", 0x300000000),
        ("16CH_shared[1] - 中间", 0x400000000),
        ("16CH_shared[1] - 结束", 0x4ffffffff),

        # 8CH_shared 区域0 测试
        ("8CH_shared[0] - 起始", 0x500000000),
        ("8CH_shared[0] - 结束", 0x57fffffff),

        # 8CH_shared 区域1 测试
        ("8CH_shared[1] - 起始", 0x580000000),
        ("8CH_shared[1] - 结束", 0x5ffffffff),

        # 8CH_shared 区域2 测试
        ("8CH_shared[2] - 起始", 0x600000000),
        ("8CH_shared[2] - 结束", 0x67fffffff),

        # 8CH_shared 区域3 测试
        ("8CH_shared[3] - 起始", 0x680000000),
        ("8CH_shared[3] - 结束", 0x6ffffffff),

        # Private 测试
        ("Private - 起始", 0x700000000),
        ("Private - 中间", 0x800000000),
    ]

    for desc, addr in test_cases:
        try:
            category = hasher_256.get_address_category(hex(addr))
            node, ip = hasher_256.hash_all(hex(addr))
            print(f"{desc:25} | 地址: {hex(addr):12} | 类别: {category:12} | 节点: {node:2} | IP: {ip}")
        except Exception as e:
            print(f"{desc:25} | 地址: {hex(addr):12} | 错误: {e}")

    # 测试512b交织
    print("\n" + "=" * 60)
    print("512b交织测试")
    print("=" * 60)
    hasher_512 = AddressHasher(interleave_size=512)

    # 测试几个关键地址的交织差异
    key_addresses = [
        ("32CH - 测试1", 0x80000000),
        ("32CH - 测试2", 0x80001000),
        ("16CH[0] - 测试1", 0x100000000),
        ("16CH[0] - 测试2", 0x100001000),
        ("8CH[0] - 测试1", 0x500000000),
        ("8CH[0] - 测试2", 0x500001000),
    ]

    for desc, addr in key_addresses:
        try:
            node_256, ip_256 = hasher_256.hash_all(hex(addr))
            node_512, ip_512 = hasher_512.hash_all(hex(addr))
            print(f"{desc:20} | 地址: {hex(addr):12} | 256b: 节点{node_256:2} {ip_256} | 512b: 节点{node_512:2} {ip_512}")
        except Exception as e:
            print(f"{desc:20} | 地址: {hex(addr):12} | 错误: {e}")

    print("\n" + "=" * 60)
    print("统计分布测试")
    print("=" * 60)

    # 测试32CH的分布均匀性
    print("\n32CH_shared 分布测试:")
    node_count_32ch = [0] * 16  # 16个节点
    ddr_count_32ch = [0, 0]     # 2个DDR
    test_addresses_32ch = range(0x80000000, 0x80100000, 0x1000)  # 测试1MB范围

    for addr in test_addresses_32ch:
        try:
            node, ip = hasher_256.hash_all(hex(addr))
            if 0 <= node < 16:
                node_count_32ch[node] += 1
            ddr_id = int(ip.split('_')[1])
            ddr_count_32ch[ddr_id] += 1
        except:
            continue

    print(f"测试地址数量: {len(test_addresses_32ch)}")
    print(f"节点分布: {node_count_32ch}")
    print(f"DDR分布: ddr_0={ddr_count_32ch[0]}, ddr_1={ddr_count_32ch[1]}")
    print(f"节点最小/最大: {min(node_count_32ch)} / {max(node_count_32ch)}")

    # 测试16CH区域0的分布
    print("\n16CH_shared[0] 分布测试:")
    node_count_16ch0 = [0] * 8  # 前8个节点
    ddr_count_16ch0 = [0, 0]
    test_addresses_16ch0 = range(0x100000000, 0x100100000, 0x10000)  # 测试1MB范围

    for addr in test_addresses_16ch0:
        try:
            node, ip = hasher_256.hash_all(hex(addr))
            if 0 <= node < 8:  # 区域0应该映射到节点0-7
                node_count_16ch0[node] += 1
            ddr_id = int(ip.split('_')[1])
            ddr_count_16ch0[ddr_id] += 1
        except:
            continue

    print(f"测试地址数量: {len(test_addresses_16ch0)}")
    print(f"节点分布: {node_count_16ch0}")
    print(f"DDR分布: ddr_0={ddr_count_16ch0[0]}, ddr_1={ddr_count_16ch0[1]}")
    if node_count_16ch0:
        print(f"节点最小/最大: {min([x for x in node_count_16ch0 if x > 0])} / {max(node_count_16ch0)}")

    # 测试16CH区域1的分布
    print("\n16CH_shared[1] 分布测试:")
    node_count_16ch1 = [0] * 8  # 后8个节点（8-15）
    ddr_count_16ch1 = [0, 0]
    test_addresses_16ch1 = range(0x300000000, 0x300100000, 0x10000)  # 测试1MB范围

    for addr in test_addresses_16ch1:
        try:
            node, ip = hasher_256.hash_all(hex(addr))
            if 8 <= node < 16:  # 区域1应该映射到节点8-15
                node_count_16ch1[node - 8] += 1
            ddr_id = int(ip.split('_')[1])
            ddr_count_16ch1[ddr_id] += 1
        except:
            continue

    print(f"测试地址数量: {len(test_addresses_16ch1)}")
    print(f"节点分布: {node_count_16ch1}")
    print(f"DDR分布: ddr_0={ddr_count_16ch1[0]}, ddr_1={ddr_count_16ch1[1]}")
    if node_count_16ch1:
        print(f"节点最小/最大: {min([x for x in node_count_16ch1 if x > 0])} / {max(node_count_16ch1)}")

    # 测试交织位的影响
    print("\n交织位影响测试:")
    test_addr = 0x80000000
    print(f"基地址: {hex(test_addr)}")
    for i in range(8):
        addr = test_addr + i * 0x1000
        node_256, ip_256 = hasher_256.hash_all(hex(addr))
        node_512, ip_512 = hasher_512.hash_all(hex(addr))
        print(f"  偏移{i*0x1000:04X}: 256b=节点{node_256:2} {ip_256}, 512b=节点{node_512:2} {ip_512}")

if __name__ == "__main__":
    test_address_mapping()