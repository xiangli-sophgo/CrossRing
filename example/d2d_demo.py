"""
D2D (Die-to-Die) CrossRing演示
一个示例脚本，用于演示跨Die通信功能。
"""

from src.core.d2d_model import D2D_Model
from config.d2d_config import D2DConfig
import numpy as np
import logging
import os


def create_d2d_traffic_file(traffic_path, d2d_config=None):
    """创建D2D测试用的traffic文件"""
    os.makedirs(traffic_path, exist_ok=True)
    
    # 创建跨Die读写请求的traffic文件
    traffic_file = os.path.join(traffic_path, "d2d_test_traffic.txt")
    
    if d2d_config:
        # 使用D2DConfig生成流量
        traffic_lines = d2d_config.generate_d2d_traffic_example(num_requests=50)
        with open(traffic_file, 'w') as f:
            for line in traffic_lines:
                f.write(line + "\n")
    else:
        # fallback到原来的生成方式
        with open(traffic_file, 'w') as f:
            # 写入header - 使用标准CrossRing流量格式，inject_time在最前面
            f.write("# D2D Traffic Test File\n")
            f.write("# Format: inject_time, src_die, src_node, src_ip, dst_die, dst_node, dst_ip, req_type, burst_length\n")
            f.write("# Example: 5, 0, 0, gdma_0, 1, 10, ddr_0, R, 4\n")
            f.write("#\n")
            
            # Die 0 -> Die 1 读请求
            f.write("100, 0, 5, gdma_0, 1, 10, ddr_0, R, 4\n")
            f.write("200, 0, 6, gdma_1, 1, 12, ddr_1, R, 8\n")
            
            # Die 0 -> Die 1 写请求
            f.write("300, 0, 7, gdma_0, 1, 15, ddr_0, W, 4\n")
            f.write("400, 0, 8, gdma_1, 1, 16, ddr_1, W, 8\n")
            
            # Die 1 -> Die 0 读请求
            f.write("500, 1, 5, gdma_0, 0, 20, ddr_0, R, 4\n")
            f.write("600, 1, 6, gdma_1, 0, 22, ddr_1, R, 8\n")
            
            # Die 1 -> Die 0 写请求
            f.write("700, 1, 7, gdma_0, 0, 25, ddr_0, W, 4\n")
            f.write("800, 1, 8, gdma_1, 0, 26, ddr_1, W, 8\n")
            
            # 更多测试用例
            for cycle in range(1000, 5000, 100):
                src_die = np.random.randint(0, 2)
                dst_die = 1 - src_die  # 另一个Die
                src_node = np.random.randint(0, 20)
                dst_node = np.random.randint(20, 40)
                req_type = np.random.choice(['R', 'W'])  # 使用R/W而非read/write
                burst_length = np.random.choice([4, 8, 16])
                src_ip = f"gdma_{np.random.randint(0, 2)}"
                dst_ip = f"ddr_{np.random.randint(0, 2)}"
                
                f.write(f"{cycle}, {src_die}, {src_node}, {src_ip}, {dst_die}, {dst_node}, {dst_ip}, {req_type}, {burst_length}\n")
    
    return "d2d_test_traffic.txt"


def main():
    """
    D2D CrossRing仿真演示。
    """
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 定义模型类型
    model_type = "REQ_RSP"
    
    # 创建D2D配置 - 使用新的YAML配置
    config = D2DConfig(
        base_config_file="../config/topologies/d2d_config.yaml",  # 使用YAML配置
        d2d_layout="horizontal"  # 可选: "horizontal" 或 "vertical"
    )
    
    # 定义拓扑结构
    topo_type = "8x9"
    config.TOPO_TYPE = topo_type
    
    # 打印D2D配置信息
    config.print_d2d_layout()
    
    # 创建traffic文件，使用新的D2DConfig
    traffic_path = "../traffic/d2d_test/"
    traffic_file = create_d2d_traffic_file(traffic_path, config)
    
    # 初始化D2D仿真模型
    sim = D2D_Model(
        config=config,
        traffic_config=[[traffic_file]],
        model_type=model_type,
        topo_type=topo_type,
        traffic_file_path=traffic_path,
        result_save_path="../Result/d2d_test/",
        results_fig_save_path="",
        plot_flow_fig=0,
        plot_RN_BW_fig=0,
        plot_link_state=0,
        plot_start_cycle=0,
        print_trace=0,
        show_trace_id=0,
        verbose=1,
    )
    
    # 初始化仿真
    sim.initial()
    
    # 设置仿真参数
    sim.end_time = 10000  # 运行10000个周期
    sim.print_interval = 1000
    
    print("开始D2D仿真...")
    print(f"将运行 {sim.end_time} 个周期")
    print()
    
    # 运行仿真
    sim.run()
    
    print("\nD2D仿真完成!")


if __name__ == "__main__":
    main()