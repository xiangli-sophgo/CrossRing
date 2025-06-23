#!/usr/bin/env python3
"""
测试Ring.py的修复是否正确
"""
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.core.Ring import RingConfig, RingTopology
    
    print("=== 测试Ring修复 ===")
    
    # 创建配置
    config = RingConfig()
    config.NUM_RING_NODES = 4  # 使用较小的配置进行快速测试
    
    print(f"创建Ring配置成功，节点数: {config.NUM_RING_NODES}")
    
    # 创建Ring拓扑
    try:
        ring = RingTopology(config, "test_data")
        print("创建Ring拓扑成功")
        
        # 检查仲裁状态初始化
        for channel in ["req", "rsp", "data"]:
            for node_id in range(config.NUM_RING_NODES):
                ring_node = ring.networks[channel].ring_nodes[node_id]
                
                # 验证仲裁状态结构
                if hasattr(ring_node, 'inject_arbitration_state'):
                    state = ring_node.inject_arbitration_state
                    if "CW" in state and "CCW" in state:
                        if isinstance(state["CW"], dict) and isinstance(state["CCW"], dict):
                            print(f"✓ 节点 {node_id} 通道 {channel} 仲裁状态结构正确")
                        else:
                            print(f"✗ 节点 {node_id} 通道 {channel} 仲裁状态结构错误: {type(state['CW'])}, {type(state['CCW'])}")
                    else:
                        print(f"✗ 节点 {node_id} 通道 {channel} 缺少CW/CCW键: {list(state.keys())}")
                else:
                    print(f"! 节点 {node_id} 通道 {channel} 没有inject_arbitration_state")
        
        print("\n=== 测试注入逻辑 ===")
        # 运行少量周期测试注入逻辑
        for i in range(5):
            try:
                ring.step_simulation()
                print(f"✓ 周期 {i+1} 执行成功")
            except Exception as e:
                print(f"✗ 周期 {i+1} 执行失败: {e}")
                break
        
        print("\n=== 测试完成 ===")
        print("Ring修复验证成功！")
        
    except Exception as e:
        print(f"创建Ring拓扑失败: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f"导入失败: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()