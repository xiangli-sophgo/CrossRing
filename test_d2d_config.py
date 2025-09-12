#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2D配置格式测试脚本
用于验证新的配置格式是否能正确解析和验证
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from config.d2d_config import D2DConfig


def test_2die_config():
    """测试2-Die配置"""
    print("=== 测试2-Die配置 ===")
    try:
        config = D2DConfig(
            die_config_file="topologies/topo_5x4.yaml",
            d2d_config_file="config/topologies/d2d_config.yaml",
        )

        print(f"NUM_DIES: {config.NUM_DIES}")
        print(f"D2D_PAIRS数量: {len(config.D2D_PAIRS)}")
        print("D2D连接对:")
        for i, pair in enumerate(config.D2D_PAIRS):
            print(f"  {i+1}. Die{pair[0]}:节点{pair[1]} <-> Die{pair[2]}:节点{pair[3]}")

        print("D2D节点位置:")
        for die_id in range(config.NUM_DIES):
            positions = config.D2D_DIE_POSITIONS.get(die_id, [])
            print(f"  Die{die_id}: {positions}")

        print("[PASS] 2-Die配置测试通过\n")
        return True

    except Exception as e:
        print(f"[FAIL] 2-Die配置测试失败: {e}\n")
        return False


def test_4die_config():
    """测试4-Die配置"""
    print("=== 测试4-Die配置 ===")
    try:
        config = D2DConfig(
            die_config_file="topologies/topo_5x4.yaml",
            d2d_config_file="config/topologies/d2d_4die_config.yaml",
        )

        print(f"NUM_DIES: {config.NUM_DIES}")
        print(f"D2D_PAIRS数量: {len(config.D2D_PAIRS)}")
        print("D2D连接对:")
        for i, pair in enumerate(config.D2D_PAIRS):
            print(f"  {i+1:2d}. Die{pair[0]}:节点{pair[1]} <-> Die{pair[2]}:节点{pair[3]}")

        print("D2D节点位置:")
        for die_id in range(config.NUM_DIES):
            positions = config.D2D_DIE_POSITIONS.get(die_id, [])
            print(f"  Die{die_id}: {positions}")

        # 验证连接的双向性
        print("验证连接双向性:")
        die_connections = {}
        for pair in config.D2D_PAIRS:
            die0, die1 = pair[0], pair[2]
            if die0 not in die_connections:
                die_connections[die0] = set()
            if die1 not in die_connections:
                die_connections[die1] = set()
            die_connections[die0].add(die1)
            die_connections[die1].add(die0)

        for die_id, connected in die_connections.items():
            print(f"  Die{die_id} 连接到: {sorted(connected)}")

        print("[PASS] 4-Die配置测试通过\n")
        return True

    except Exception as e:
        print(f"[FAIL] 4-Die配置测试失败: {e}\n")
        return False


def test_invalid_config():
    """测试无效配置的验证"""
    print("=== 测试配置验证功能 ===")

    # 创建一个无效的配置来测试验证功能
    test_config = {
        0: {
            "num_row": 5,
            "num_col": 4,
            "connections": {
                "top": {
                    "d2d_nodes": [0, 1, 2],
                    "node_to_die_mapping": {
                        0: 1,
                        1: 1,
                        # 缺少节点2的映射
                    },
                }
            },
        },
        1: {"num_row": 5, "num_col": 4, "connections": {"bottom": {"d2d_nodes": [0, 1], "node_to_die_mapping": {0: 0, 1: 0}}}},
    }

    try:
        config = D2DConfig(die_config_file="topologies/topo_5x4.yaml")
        config.NUM_DIES = 2
        config._validate_d2d_config(test_config)
        print("[FAIL] 配置验证测试失败: 应该检测到无效配置")
        return False

    except ValueError as e:
        print("[PASS] 配置验证测试通过: 成功检测到错误")
        print(f"   错误信息: {str(e)[:100]}...")
        return True
    except Exception as e:
        print(f"[ERROR] 配置验证测试异常: {e}")
        return False


def main():
    """主测试函数"""
    print("D2D配置格式测试开始\n")

    results = []
    results.append(test_2die_config())
    results.append(test_4die_config())
    results.append(test_invalid_config())

    # 总结测试结果
    passed = sum(results)
    total = len(results)

    print(f"=== 测试结果总结 ===")
    print(f"通过: {passed}/{total}")
    if passed == total:
        print("[SUCCESS] 所有测试通过!")
        return 0
    else:
        print("[WARNING] 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
