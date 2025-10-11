"""
测试D2D进程并行化性能对比

对比串行模式和并行模式的执行时间和结果一致性
"""

import time
from src.core.d2d_model import D2D_Model
from config.d2d_config import D2DConfig


def run_simulation(enable_parallel, mode_name):
    """
    运行D2D仿真

    Args:
        enable_parallel: 是否启用并行模式
        mode_name: 模式名称（用于输出）

    Returns:
        execution_time: 执行时间（秒）
        die_stats: Die统计信息
    """
    print(f"\n{'='*60}")
    print(f"开始测试: {mode_name}")
    print(f"{'='*60}\n")

    # 创建配置
    config = D2DConfig(
        d2d_config_file="../config/topologies/d2d_config.yaml",
    )

    print(f"配置信息:")
    print(f"  Die数量: {getattr(config, 'NUM_DIES', 2)}")
    print(f"  并行模式: {enable_parallel}")

    # 初始化D2D仿真模型
    sim = D2D_Model(
        config=config,
        traffic_file_path=r"../test_data",
        traffic_config=[
            [
                "d2d_data_0916.txt",
            ],
        ],
        model_type="REQ_RSP",
        result_save_path=f"../Result/d2d_test_{mode_name}/",
        results_fig_save_path=f"../Result/d2d_test_{mode_name}/figures/",
        verbose=1,
        print_d2d_trace=0,
        show_d2d_trace_id=1,
        d2d_trace_sleep=0.1,
        enable_flow_graph=0,  # 关闭流量图以加快测试
        plot_link_state=0,
        enable_parallel=enable_parallel,
    )

    # 初始化仿真
    sim.initial()

    # 设置仿真参数
    sim.end_time = 5000
    sim.print_interval = 500

    # 测量执行时间
    print(f"\n开始仿真 ({mode_name})...")
    start_time = time.time()

    sim.run()

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\n{mode_name}完成！")
    print(f"执行时间: {execution_time:.2f} 秒")

    # 获取统计信息
    die_stats = getattr(sim, 'die_stats', {})

    return execution_time, die_stats


def main():
    """
    主测试函数 - 对比串行和并行模式
    """
    print("="*60)
    print("D2D进程并行化性能测试")
    print("="*60)

    # 测试1: 串行模式
    serial_time, serial_stats = run_simulation(
        enable_parallel=False,
        mode_name="串行模式"
    )

    # 测试2: 并行模式
    parallel_time, parallel_stats = run_simulation(
        enable_parallel=True,
        mode_name="并行模式"
    )

    # 性能对比
    print("\n" + "="*60)
    print("性能对比结果")
    print("="*60)
    print(f"\n串行模式执行时间: {serial_time:.2f} 秒")
    print(f"并行模式执行时间: {parallel_time:.2f} 秒")

    if serial_time > 0:
        speedup = serial_time / parallel_time
        improvement = (serial_time - parallel_time) / serial_time * 100
        print(f"\n加速比: {speedup:.2f}x")
        print(f"性能提升: {improvement:.1f}%")

    # 结果一致性验证
    print(f"\n{'='*60}")
    print("结果一致性验证")
    print(f"{'='*60}\n")

    if serial_stats and parallel_stats:
        print("Die统计信息对比:")
        for die_id in sorted(serial_stats.keys()):
            serial_stat = serial_stats[die_id]
            parallel_stat = parallel_stats.get(die_id, {})

            print(f"\nDie {die_id}:")
            print(f"  串行 - Flits: {serial_stat.get('total_flits', 0)}, "
                  f"Reqs: {serial_stat.get('total_reqs', 0)}")
            print(f"  并行 - Flits: {parallel_stat.get('total_flits', 0)}, "
                  f"Reqs: {parallel_stat.get('total_reqs', 0)}")
    else:
        print("注意: 部分统计信息未收集（这对于串行模式是正常的）")

    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()
