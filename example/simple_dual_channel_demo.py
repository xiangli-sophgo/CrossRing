"""
简单的双通道CrossRing演示
一个简化的示例脚本，用于演示双通道数据功能。
"""

from src.core.dual_channel_base_model import DualChannelBaseModel
from config.dual_channel_config import DualChannelConfig
import numpy as np


def main():
    """
    简单的双通道CrossRing仿真演示。
    """
    print("=== 简单双通道CrossRing演示 ===")
    
    # 定义模型类型
    model_type = "DualChannel_REQ_RSP"

    # 创建双通道配置
    config = DualChannelConfig()

    # 定义拓扑结构
    topo_type = "5x4"
    config.TOPO_TYPE = topo_type
    
    # 配置双通道设置
    config.DATA_DUAL_CHANNEL_ENABLED = True
    config.DATA_CHANNEL_SELECT_STRATEGY = "hash_based"  # hash_based, size_based, type_based, load_balanced
    config.DATA_CH0_BANDWIDTH_RATIO = 0.5
    config.DATA_CH1_BANDWIDTH_RATIO = 0.5
    
    # 打印配置信息
    print("双通道配置:")
    print(f"  选择策略: {config.DATA_CHANNEL_SELECT_STRATEGY}")
    print(f"  通道0带宽比例: {config.DATA_CH0_BANDWIDTH_RATIO}")
    print(f"  通道1带宽比例: {config.DATA_CH1_BANDWIDTH_RATIO}")
    print()

    # 初始化双通道仿真模型
    sim = DualChannelBaseModel(
        model_type=model_type,
        config=config,
        topo_type=topo_type,
        traffic_file_path="",
        traffic_config=[["test_data.txt"]],
        result_save_path="../Result/simple_dual_channel/",
        results_fig_save_path="",
        plot_flow_fig=0,
        plot_RN_BW_fig=0,
        plot_link_state=0,
        verbose=1,
    )

    # 初始化仿真
    sim.initial()
    
    # 设置仿真参数
    sim.end_time = 1000
    sim.print_interval = 1000

    print("开始双通道仿真...")
    
    # 运行仿真
    sim.run()
    
    # 打印结果
    print("\n=== 仿真结果 ===")
    sim.print_dual_channel_summary()
    
    print("简单双通道演示完成!")


if __name__ == "__main__":
    main()