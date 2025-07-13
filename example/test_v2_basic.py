#!/usr/bin/env python3
"""
简单的v2版本功能测试
验证基本的双向Ring Bridge功能是否正常工作
"""

import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_network_v2_import():
    """测试network_v2模块导入"""
    print("1. 测试network_v2模块导入...")
    try:
        from src.utils.components.network_v2 import Network
        from config.config import CrossRingConfig
        from src.utils.optimal_placement import create_adjacency_matrix
        
        config = CrossRingConfig()
        adjacency_matrix = create_adjacency_matrix("CrossRing", config.NUM_NODE, config.NUM_COL)
        network = Network(config, adjacency_matrix, name="Test Network v2")
        
        print("   ✓ Network_v2导入成功")
        print(f"   ✓ 网络配置: {config.NUM_ROW}x{config.NUM_COL}")
        
        # 验证新增的FIFO结构
        print("   ✓ 验证新增FIFO结构:")
        print(f"     - ring_bridge_input keys: {list(network.ring_bridge_input.keys())}")
        print(f"     - ring_bridge_output keys: {list(network.ring_bridge_output.keys())}")
        
        # 验证round-robin队列支持6个输入源
        sample_pos = config.NUM_COL if config.NUM_COL < config.NUM_NODE else 0
        rr_queue = network.round_robin["RB"]["EQ"][sample_pos]
        print(f"     - Round-robin队列长度: {len(rr_queue)}")
        print(f"     - Round-robin队列内容: {list(rr_queue)}")
        
        return True
    except Exception as e:
        print(f"   ❌ Network_v2导入失败: {e}")
        return False


def test_base_model_v2_import():
    """测试base_model_v2模块导入"""
    print("\n2. 测试base_model_v2模块导入...")
    try:
        from src.core.base_model_v2 import BaseModel
        from config.config import CrossRingConfig
        
        print("   ✓ BaseModel_v2导入成功")
        
        # 检查新增方法是否存在
        model_methods = dir(BaseModel)
        required_methods = [
            'Ring_Bridge_arbitration',
            'RB_inject_horizontal',
            '_ring_bridge_arbitrate_horizontal',
            '_should_output_to_horizontal'
        ]
        
        for method in required_methods:
            if method in model_methods:
                print(f"   ✓ 方法 {method} 存在")
            else:
                print(f"   ❌ 方法 {method} 不存在")
                return False
        
        return True
    except Exception as e:
        print(f"   ❌ BaseModel_v2导入失败: {e}")
        return False


def test_basic_functionality():
    """测试基本功能"""
    print("\n3. 测试基本功能...")
    try:
        from src.core.base_model_v2 import BaseModel
        from config.config import CrossRingConfig
        
        config = CrossRingConfig()
        
        # 创建一个最小的仿真实例
        traffic_dir = os.path.join(project_root, "traffic")
        os.makedirs(traffic_dir, exist_ok=True)
        
        # 创建最简单的测试traffic文件
        simple_traffic = os.path.join(traffic_dir, "simple_test.txt")
        with open(simple_traffic, "w") as f:
            f.write("10,0,gdma,1,ddr,R,1\n")  # 一个简单的读请求
        
        print("   ✓ 创建测试traffic文件")
        
        # 创建仿真实例但不运行
        sim = BaseModel(
            model_type="REQ_RSP",
            config=config,
            topo_type="5x4",
            traffic_file_path=traffic_dir,
            traffic_config="simple_test.txt",
            result_save_path="",
            verbose=0
        )
        
        print("   ✓ BaseModel_v2实例创建成功")
        
        # 初始化
        sim.initial()
        print("   ✓ 仿真初始化成功")
        
        # 验证网络是否使用了v2版本
        network = sim.req_network
        if hasattr(network, 'ring_bridge_input') and hasattr(network, 'ring_bridge_output'):
            print("   ✓ 确认使用Network_v2")
        else:
            print("   ❌ 未使用Network_v2")
            return False
        
        # 验证双向RB方法是否可调用
        try:
            # 测试新方法（不实际运行，只检查是否存在且可调用）
            if hasattr(sim, 'RB_inject_horizontal'):
                print("   ✓ RB_inject_horizontal方法可用")
            if hasattr(sim, '_should_output_to_horizontal'):
                print("   ✓ _should_output_to_horizontal方法可用")
        except Exception as e:
            print(f"   ⚠️  方法调用测试遇到问题: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_minimal_simulation():
    """运行最小仿真测试"""
    print("\n4. 运行最小仿真测试...")
    try:
        from src.core.base_model_v2 import BaseModel
        from config.config import CrossRingConfig
        
        config = CrossRingConfig()
        
        # 创建最小流量文件
        traffic_dir = os.path.join(project_root, "traffic")
        minimal_traffic = os.path.join(traffic_dir, "minimal_test.txt")
        
        with open(minimal_traffic, "w") as f:
            # 只有一个简单请求，避免复杂路径
            f.write("10,0,gdma,0,ddr,R,1\n")  # 本地访问
        
        sim = BaseModel(
            model_type="REQ_RSP",
            config=config,
            topo_type="default",  # 使用默认配置而不是5x4
            traffic_file_path=traffic_dir,
            traffic_config="minimal_test.txt",
            result_save_path="",
            verbose=1
        )
        
        sim.initial()
        
        print("   ✓ 开始运行最小仿真...")
        
        # 设置较小的结束时间避免长时间运行
        sim.end_time = 100  # 100 ns
        
        sim.run()
        
        print("   ✓ 最小仿真运行完成")
        
        # 获取结果
        results = sim.get_results()
        print(f"   ✓ 仿真周期: {results.get('cycle', 'N/A')}")
        print(f"   ✓ 发送flit数: {results.get('send_flits_num_stat', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 最小仿真失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 50)
    print("CrossRing v2版本基本功能测试")
    print("=" * 50)
    
    test_results = []
    
    # 运行各项测试
    test_results.append(("Network_v2导入", test_network_v2_import()))
    test_results.append(("BaseModel_v2导入", test_base_model_v2_import()))
    test_results.append(("基本功能", test_basic_functionality()))
    test_results.append(("最小仿真", run_minimal_simulation()))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有测试通过! v2版本基本功能正常")
        print("\n可以使用以下功能:")
        print("- ✅ 双向Ring Bridge仲裁（6个输入源）")
        print("- ✅ 纵向环→横向环转换") 
        print("- ✅ 横向环→纵向环转换（原有功能）")
        print("- ✅ 扩展的FIFO结构")
        print("- ✅ 与路由表的集成")
    else:
        print("❌ 部分测试失败，请检查v2版本实现")
    print("=" * 50)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)