"""测试批量优化的效果"""
import sys
import time
sys.path.insert(0, '/Users/lixiang/Documents/工作/code/CrossRing')

from src.kcin.v1.config import V1Config
from src.kcin.v1 import REQ_RSP_model

def test_batch_optimization():
    """测试批量优化"""

    # 配置路径
    config_path = "../config/topologies/kcin_5x4_2.5GHz.yaml"
    traffic_file_path = "../traffic"
    traffic_config = [["LLama2_AllReduce.txt"]]

    # 创建配置
    config = V1Config(config_path)

    print("=" * 60)
    print("批量优化测试")
    print("=" * 60)
    print(f"网络频率: {config.NETWORK_FREQUENCY}GHz")
    print(f"IP频率: {config.IP_FREQUENCY}GHz")
    print(f"CYCLES_PER_NS: {config.CYCLES_PER_NS}")
    print(f"NETWORK_SCALE: {config.NETWORK_SCALE}")
    print(f"IP_SCALE: {config.IP_SCALE}")
    print()

    # 创建模型
    sim = REQ_RSP_model(
        model_type="REQ_RSP",
        config=config,
        topo_type="5x4",
        verbose=1,
    )

    # 配置流量
    sim.setup_traffic_scheduler(
        traffic_file_path=traffic_file_path,
        traffic_chains=traffic_config
    )

    # 配置结果分析（关闭图表生成加速测试）
    sim.setup_result_analysis(
        plot_RN_BW_fig=0,
        flow_graph_interactive=0,
        fifo_utilization_heatmap=0,
        result_save_path=f"../Result/CrossRing/test_batch/",
        show_result_analysis=0,
    )

    print()
    print("=" * 60)
    print("开始仿真（100ns）")
    print("=" * 60)

    # 运行仿真
    start_time = time.perf_counter()
    sim.run_simulation(max_time=100, print_interval=500)
    end_time = time.perf_counter()

    elapsed = end_time - start_time

    print()
    print("=" * 60)
    print("仿真完成")
    print("=" * 60)
    print(f"仿真时间: {elapsed:.3f}秒")
    print(f"仿真性能: {sim.cycle / elapsed:.0f} cycles/秒")
    print(f"总仿真周期数: {sim.cycle}")

    # 计算预期的cycle数
    expected_cycles = 100 * config.CYCLES_PER_NS
    print(f"预期周期数: {expected_cycles}")

    # 计算跳过的cycle数
    skipped_ratio = (sim.SYNC_PERIOD - len(sim.active_offsets)) / sim.SYNC_PERIOD
    print(f"跳过cycle比例: {skipped_ratio*100:.1f}%")

if __name__ == "__main__":
    test_batch_optimization()
