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

    # 创建双通道配置
    config = DualChannelConfig()

    # 定义拓扑结构
    topo_type = "5x4"
    config.TOPO_TYPE = topo_type

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
        traffic_file_path=r"",
        traffic_config=[["../test_data/test1.txt"]],
        result_save_path="../Result/simple_dual_channel/",
        results_fig_save_path="",
        plot_flow_fig=1,
        plot_RN_BW_fig=0,
        plot_link_state=0,
        print_trace=0,
        show_trace_id=0,
        verbose=1,
    )

    # 初始化仿真
    sim.initial()

    # 设置仿真参数
    sim.end_time = 1000
    sim.print_interval = 500

    print("开始双通道仿真...")

    # 运行仿真
    sim.run()

    print("简单双通道演示完成!")


if __name__ == "__main__":
    main()
