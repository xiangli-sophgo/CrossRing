"""
D2D (Die-to-Die) 5x4拓扑演示
包含完整的结果处理和可视化功能。
"""

from src.core.d2d_model import D2D_Model
from config.d2d_config import D2DConfig


def main():
    """
    D2D 仿真演示 - 包含完整的结果处理和可视化
    """
    # 使用D2DConfig替代CrossRingConfig，获得D2D特定配置功能
    # 现在每个Die的拓扑由D2D配置文件中的topology参数指定，无需die_config_file
    config = D2DConfig(
        # d2d_config_file="../config/topologies/d2d_4die_config.yaml",
        d2d_config_file="../config/topologies/d2d_config.yaml",
    )

    # 定义拓扑结构

    print(f"配置信息:")
    print(f"  Die数量: {getattr(config, 'NUM_DIES', 2)}")
    print(f"  Die拓扑配置:")
    die_topologies = getattr(config, "DIE_TOPOLOGIES", {})
    for die_id, topology in die_topologies.items():
        print(f"    Die{die_id}: {topology}")
    print()

    # 初始化D2D仿真模型 - 启用完整功能和D2D可视化
    sim = D2D_Model(
        config=config,
        traffic_file_path=r"../test_data",
        traffic_config=[
            [
                "d2d_4die_1016.txt",
            ],
        ],
        model_type="REQ_RSP",
        result_save_path="../Result/d2d_demo/",
        results_fig_save_path="../Result/d2d_demo/figures/",
        verbose=1,
        print_d2d_trace=1,  # 启用D2D trace功能
        show_d2d_trace_id=[12],  # 自动跟踪所有活跃packet，也可以指定特定ID如[1, 2]
        d2d_trace_sleep=0.1,  # 不暂停，加快调试as
        enable_flow_graph=1,  # 是否在仿真结束后自动生成流量图
        # D2D链路状态可视化参
        plot_link_state=0,  # 启用D2D链路状态可视化 12
        plot_start_cycle=10,  # 从第100周期开始可视化
    )

    # 初始化仿真
    sim.initial()

    # 设置仿真参数
    sim.end_time = 5000  # 增加仿真时间以确保数据传输完成
    sim.print_interval = 1000

    sim.run()

    # 调用D2D结果处理
    try:
        sim.process_d2d_comprehensive_results()
    except Exception as e:
        print(f"D2D结果处理失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
