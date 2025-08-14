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
    config.ENABLE_IN_ORDER_EJECTION = 0

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

    # 调试：打印IP ID分配情况
    print("IP ID 分配情况:")
    gdma_0_count = 0
    gdma_1_count = 0
    for (ip_type, ip_pos), ip_interface in sim.ip_modules.items():
        ip_id = ip_interface.ip_id
        channel = ip_id % 2
        if ip_type == "gdma_0":
            gdma_0_count += 1
        elif ip_type == "gdma_1":
            gdma_1_count += 1
        print(f"  ({ip_type}, {ip_pos}) -> ip_id={ip_id} -> channel_{channel}")

    print(f"\n统计: gdma_0 有 {gdma_0_count} 个实例, gdma_1 有 {gdma_1_count} 个实例")
    print()

    # 设置仿真参数
    sim.end_time = 6000  # 超短时间测试，只为调试
    sim.print_interval = 2000

    print("开始双通道仿真...")

    # 运行仿真
    sim.run()


if __name__ == "__main__":
    main()
