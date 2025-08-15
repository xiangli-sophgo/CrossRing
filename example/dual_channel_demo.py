"""
双通道CrossRing演示
一个示例脚本，用于演示双通道数据功能。
"""

from src.core.dual_channel_base_model import DualChannelBaseModel
from config.dual_channel_config import DualChannelConfig
import numpy as np


def main():
    """
    双通道CrossRing仿真演示。
    """

    # 定义模型类型
    model_type = "DualChannel_REQ_RSP"

    # 创建双通道配置 - 可以基于拓扑专用配置或自定义双通道配置
    topo_type = "5x4"

    # 拓扑配置：
    config = DualChannelConfig(f"../config/topologies/topo_{topo_type}.yaml")

    # 配置双通道设置
    config.DATA_DUAL_CHANNEL_ENABLED = True
    config.DATA_CHANNEL_SELECT_STRATEGY = "ip_id_based"  # ip_id_based, target_node_based, flit_id_based

    # 打印配置信息
    print("双通道配置:")
    print(f"  选择策略: {config.DATA_CHANNEL_SELECT_STRATEGY}")
    print()

    # 初始化双通道仿真模型
    sim = DualChannelBaseModel(
        model_type=model_type,
        config=config,
        topo_type=topo_type,
        traffic_file_path=r"../traffic/0617",
        traffic_config=[["LLama2_AllReduce.txt"]],
        result_save_path="../Result/simple_dual_channel/",
        results_fig_save_path="",
        plot_flow_fig=0,
        plot_RN_BW_fig=0,
        plot_link_state=0,
        plot_start_cycle=3000,
        print_trace=0,
        show_trace_id=10651,
        verbose=1,
    )

    # 初始化仿真
    sim.initial()

    # 设置仿真参数
    sim.end_time = 6000
    sim.print_interval = 2000

    print("开始双通道仿真...")

    # 运行仿真
    sim.run()


if __name__ == "__main__":
    main()
