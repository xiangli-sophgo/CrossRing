"""
简单的并行模式测试 - 快速验证
"""

from src.core.d2d_model import D2D_Model
from config.d2d_config import D2DConfig

print("开始简单并行测试...")

config = D2DConfig(
    d2d_config_file="../config/topologies/d2d_config.yaml",
)

print(f"Die数量: {getattr(config, 'NUM_DIES', 2)}")

sim = D2D_Model(
    config=config,
    traffic_file_path=r"../test_data",
    traffic_config=[
        ["d2d_data_0916.txt"],
    ],
    model_type="REQ_RSP",
    result_save_path="../Result/d2d_test_simple/",
    verbose=1,
    print_d2d_trace=0,
    enable_flow_graph=0,
    plot_link_state=0,
    enable_parallel=1,  # 并行模式
)

sim.initial()
sim.end_time = 100  # 只测试100周期
sim.print_interval = 50

print("开始仿真...")
sim.run()
print("仿真完成！")
