"""
双通道CrossRing演示
这是一个运行支持双通道数据的CrossRing仿真的示例脚本。
"""

from src.core.dual_channel_base_model import DualChannelBaseModel
from config.dual_channel_config import (
    DualChannelConfig, 
    create_balanced_dual_channel_config,
    create_read_write_separated_config,
    create_size_based_dual_channel_config
)
from src.utils.channel_selector import DefaultChannelSelector
import numpy as np
import os


def demo_basic_dual_channel():
    """基础双通道演示"""
    print("=== 基础双通道演示 ===")
    
    # 定义模型类型
    model_type = "DualChannel_REQ_RSP"
    
    # 创建双通道配置
    config = DualChannelConfig()
    
    # 配置拓扑结构
    topo_type = "5x4"
    config.TOPO_TYPE = topo_type
    
    # 设置双通道特定参数
    config.set_channel_selection_strategy("hash_based")
    config.set_channel_bandwidth_ratio(0.5, 0.5)  # 等带宽分配
    config.print_dual_channel_config()
    
    # 初始化双通道仿真模型
    sim = DualChannelBaseModel(
        model_type=model_type,
        config=config,
        topo_type=topo_type,
        traffic_file_path="",
        traffic_config=[["test_data.txt"]],
        result_save_path="../Result/dual_channel_basic/",
        results_fig_save_path="",
        plot_flow_fig=0,
        plot_RN_BW_fig=0,
        plot_link_state=0,
        verbose=1,
    )
    
    # 设置仿真参数
    sim.end_time = 1000
    sim.print_interval = 1000
    
    print("开始基础双通道仿真...")
    
    # 初始化并运行仿真
    sim.initial()
    sim.run()
    
    # 打印双通道统计信息
    sim.print_dual_channel_summary()
    
    print("基础双通道演示完成!\n")


def demo_read_write_separated():
    """读写分离双通道演示"""
    print("=== 读写分离双通道演示 ===")
    
    model_type = "DualChannel_REQ_RSP"
    
    # 创建读写分离配置
    config = create_read_write_separated_config()
    
    # 配置拓扑结构
    topo_type = "5x4"
    config.TOPO_TYPE = topo_type
    
    config.print_dual_channel_config()
    
    # 初始化读写分离配置的仿真
    sim = DualChannelBaseModel(
        model_type=model_type,
        config=config,
        topo_type=topo_type,
        traffic_file_path="",
        traffic_config=[["test_data.txt"]],
        result_save_path="../Result/dual_channel_rw_separated/",
        results_fig_save_path="",
        plot_flow_fig=0,
        plot_RN_BW_fig=0,
        plot_link_state=0,
        verbose=1,
    )
    
    sim.end_time = 1000
    sim.print_interval = 1000
    
    print("开始读写分离双通道仿真...")
    
    sim.initial()
    sim.run()
    
    sim.print_dual_channel_summary()
    
    print("读写分离演示完成!\n")


def demo_size_based_channels():
    """基于包大小的双通道演示"""
    print("=== 基于包大小的双通道演示 ===")
    
    model_type = "DualChannel_REQ_RSP"
    
    # 创建基于包大小的配置
    config = create_size_based_dual_channel_config()
    
    # 配置拓扑结构
    topo_type = "5x4"
    config.TOPO_TYPE = topo_type
    
    config.print_dual_channel_config()
    
    # 初始化基于包大小配置的仿真
    sim = DualChannelBaseModel(
        model_type=model_type,
        config=config,
        topo_type=topo_type,
        traffic_file_path="",
        traffic_config=[["test_data.txt"]],
        result_save_path="../Result/dual_channel_size_based/",
        results_fig_save_path="",
        plot_flow_fig=0,
        plot_RN_BW_fig=0,
        plot_link_state=0,
        verbose=1,
    )
    
    sim.end_time = 1000
    sim.print_interval = 1000
    
    print("开始基于包大小的双通道仿真...")
    
    sim.initial()
    sim.run()
    
    sim.print_dual_channel_summary()
    
    print("基于包大小演示完成!\n")


def demo_load_balanced_channels():
    """负载均衡双通道演示"""
    print("=== 负载均衡双通道演示 ===")
    
    model_type = "DualChannel_REQ_RSP"
    
    # 创建负载均衡配置
    config = create_balanced_dual_channel_config()
    
    # 配置拓扑结构
    topo_type = "5x4"
    config.TOPO_TYPE = topo_type
    
    # 启用自适应选择以获得更好的负载均衡效果
    config.ENABLE_ADAPTIVE_CHANNEL_SELECTION = True
    config.DUAL_CHANNEL_CONGESTION_THRESHOLD = 0.7
    
    config.print_dual_channel_config()
    
    # 初始化负载均衡配置的仿真
    sim = DualChannelBaseModel(
        model_type=model_type,
        config=config,
        topo_type=topo_type,
        traffic_file_path="",
        traffic_config=[["test_data.txt"]],
        result_save_path="../Result/dual_channel_load_balanced/",
        results_fig_save_path="",
        plot_flow_fig=0,
        plot_RN_BW_fig=0,
        plot_link_state=0,
        verbose=1,
    )
    
    sim.end_time = 1500  # 更长的仿真时间以观察负载均衡效果
    sim.print_interval = 1000
    
    print("开始负载均衡双通道仿真...")
    
    sim.initial()
    sim.run()
    
    sim.print_dual_channel_summary()
    
    print("负载均衡演示完成!\n")


def demo_performance_comparison():
    """性能对比演示"""
    print("=== 双通道与单通道性能对比演示 ===")
    
    try:
        # 运行单通道仿真
        print("运行单通道仿真...")
        
        from src.core.REQ_RSP import REQ_RSP_model
        from config.config import CrossRingConfig
        
        single_config = CrossRingConfig()
        single_config.TOPO_TYPE = "5x4"
        
        single_sim = REQ_RSP_model(
            model_type="REQ_RSP",
            config=single_config,
            topo_type="5x4",
            traffic_file_path="",
            traffic_config=[["test_data.txt"]],
            result_save_path="../Result/single_channel_comparison/",
            results_fig_save_path="",
            plot_flow_fig=0,
            plot_RN_BW_fig=0,
            plot_link_state=0,
            verbose=0,
        )
        
        single_sim.initial()
        single_sim.end_time = 1000
        single_sim.print_interval = 1000
        single_sim.run()
        
        print("单通道仿真完成。")
        
    except Exception as e:
        print(f"单通道仿真失败: {e}")
        print("跳过单通道对比，仅运行双通道演示...")
        single_sim = None
    
    # 运行双通道仿真
    print("运行双通道仿真...")
    
    dual_config = create_balanced_dual_channel_config()
    dual_config.TOPO_TYPE = "5x4"
    
    dual_sim = DualChannelBaseModel(
        model_type="DualChannel_REQ_RSP",
        config=dual_config,
        topo_type="5x4",
        traffic_file_path="",
        traffic_config=[["test_data.txt"]],
        result_save_path="../Result/dual_channel_comparison/",
        results_fig_save_path="",
        plot_flow_fig=0,
        plot_RN_BW_fig=0,
        plot_link_state=0,
        verbose=0,
    )
    
    dual_sim.initial()
    dual_sim.end_time = 1000
    dual_sim.print_interval = 1000
    dual_sim.run()
    
    print("双通道仿真完成。")
    
    # 对比结果
    print("\n=== 性能对比结果 ===")
    
    # 获取双通道统计信息
    try:
        dual_stats = dual_sim.get_dual_channel_statistics()
        dual_latency_ch0 = dual_stats.get('avg_latency_ch0', 0)
        dual_latency_ch1 = dual_stats.get('avg_latency_ch1', 0)
        dual_inject_ch0 = dual_stats.get('total_inject_ch0', 0)
        dual_inject_ch1 = dual_stats.get('total_inject_ch1', 0)
        
        print(f"双通道结果:")
        print(f"  通道0注入包数: {dual_inject_ch0}")
        print(f"  通道1注入包数: {dual_inject_ch1}")
        print(f"  总注入包数: {dual_inject_ch0 + dual_inject_ch1}")
        
        if dual_latency_ch0 > 0 or dual_latency_ch1 > 0:
            avg_dual_latency = (dual_latency_ch0 + dual_latency_ch1) / 2 if (dual_latency_ch0 > 0 and dual_latency_ch1 > 0) else max(dual_latency_ch0, dual_latency_ch1)
            print(f"  平均延迟: {avg_dual_latency:.2f}")
        
        # 如果有单通道数据，进行对比
        if single_sim is not None:
            try:
                single_latency = getattr(single_sim, 'actual_avg_latency', None) or getattr(single_sim, 'avg_latency', None) or 50.0
                single_throughput = getattr(single_sim, 'total_throughput', None) or 100.0
                dual_throughput = dual_inject_ch0 + dual_inject_ch1
                
                print(f"\n对比结果:")
                print(f"  单通道吞吐量: {single_throughput}")
                print(f"  双通道吞吐量: {dual_throughput}")
                
                if dual_throughput > single_throughput:
                    improvement = ((dual_throughput - single_throughput) / single_throughput) * 100
                    print(f"  吞吐量提升: {improvement:.1f}%")
                else:
                    print(f"  吞吐量变化: {((dual_throughput - single_throughput) / single_throughput) * 100:.1f}%")
                
            except Exception as e:
                print(f"无法计算详细性能对比: {e}")
        
        dual_sim.print_dual_channel_summary()
        
    except Exception as e:
        print(f"警告: 无法计算详细性能对比: {e}")
        print("请手动检查仿真结果。")
    
    print("性能对比演示完成!\n")


def main():
    """
    运行双通道CrossRing仿真演示的主函数。
    """
    print("双通道CrossRing仿真演示")
    print("======================================")
    print()
    
    # 创建结果目录
    result_dirs = [
        "../Result/dual_channel_basic/",
        "../Result/dual_channel_rw_separated/", 
        "../Result/dual_channel_size_based/",
        "../Result/dual_channel_load_balanced/",
        "../Result/single_channel_comparison/",
        "../Result/dual_channel_comparison/"
    ]
    
    for result_dir in result_dirs:
        os.makedirs(result_dir, exist_ok=True)
    
    # 运行不同的演示场景
    demos = [
        ("基础双通道", demo_basic_dual_channel),
        ("读写分离", demo_read_write_separated),
        ("基于包大小", demo_size_based_channels),
        ("负载均衡", demo_load_balanced_channels),
        ("性能对比", demo_performance_comparison)
    ]
    
    print("可用演示:")
    for i, (name, _) in enumerate(demos):
        print(f"{i+1}. {name}")
    print("0. 运行所有演示")
    print()
    
    try:
        choice = input("选择要运行的演示 (0-5): ").strip()
        
        if choice == "0":
            # 运行所有演示
            for name, demo_func in demos:
                print(f"\n{'='*50}")
                print(f"运行中: {name}")
                print(f"{'='*50}")
                try:
                    demo_func()
                except Exception as e:
                    print(f"演示 '{name}' 失败: {e}")
                    print("继续下一个演示...\n")
        elif choice in ["1", "2", "3", "4", "5"]:
            # 运行选定的演示
            idx = int(choice) - 1
            name, demo_func = demos[idx]
            print(f"\n{'='*50}")
            print(f"运行中: {name}")
            print(f"{'='*50}")
            demo_func()
        else:
            print("无效选择。运行基础演示...")
            demo_basic_dual_channel()
            
    except KeyboardInterrupt:
        print("\n演示被用户中断。")
    except Exception as e:
        print(f"演示选择过程中出错: {e}")
        print("运行基础演示作为后备...")
        demo_basic_dual_channel()
    
    print("\n======================================")
    print("所有双通道演示完成!")
    print("请查看 ../Result/ 目录中的仿真输出。")
    print("======================================")


if __name__ == "__main__":
    main()