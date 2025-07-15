#!/usr/bin/env python3
"""
CrossRing NoC Bidirectional Ring Bridge Demo
演示纵向环到横向环的双向转换功能
使用base_model_v2和network_v2实现
"""

import os
import sys
import numpy as np
import time

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config import CrossRingConfig
from src.core.base_model_v2 import BaseModel
from src.utils.components.route_table import RouteTable


def run_bidirectional_rb_demo():

    # 创建配置
    print("\n初始化配置...")
    config = CrossRingConfig()
    config.TOPO_TYPE = "5x2"
    config.NUM_COL = 2
    config.NUM_NODE = 20
    config.NUM_IP = 16
    config.BURST = 4
    config.RN_R_TRACKER_OSTD = 64
    config.RN_W_TRACKER_OSTD = 32
    config.SN_DDR_R_TRACKER_OSTD = 96
    config.SN_DDR_W_TRACKER_OSTD = 48
    config.SN_L2M_R_TRACKER_OSTD = 96
    config.SN_L2M_W_TRACKER_OSTD = 48
    config.RN_RDB_SIZE = config.RN_R_TRACKER_OSTD * config.BURST
    config.RN_WDB_SIZE = config.RN_W_TRACKER_OSTD * config.BURST
    config.SN_DDR_WDB_SIZE = config.SN_DDR_W_TRACKER_OSTD * config.BURST
    config.SN_L2M_WDB_SIZE = config.SN_L2M_W_TRACKER_OSTD * config.BURST
    config.DDR_R_LATENCY_original = 100
    config.DDR_R_LATENCY_VAR_original = 0
    config.DDR_W_LATENCY_original = 40
    config.L2M_R_LATENCY_original = 0
    config.L2M_W_LATENCY_original = 0
    config.IQ_CH_FIFO_DEPTH = 10
    config.EQ_CH_FIFO_DEPTH = 10
    config.IQ_OUT_FIFO_DEPTH = 8
    config.RB_OUT_FIFO_DEPTH = 8
    config.SN_TRACKER_RELEASE_LATENCY = 40
    config.CDMA_BW_LIMIT = 8
    config.DDR_BW_LIMIT = 102
    config.RB_ONLY_TAG_NUM_PER_RING = 6

    config.TL_Etag_T2_UE_MAX = 8
    config.TL_Etag_T1_UE_MAX = 15
    config.TR_Etag_T2_UE_MAX = 12
    config.RB_IN_FIFO_DEPTH = 16
    config.TU_Etag_T2_UE_MAX = 8
    config.TU_Etag_T1_UE_MAX = 15
    config.TD_Etag_T2_UE_MAX = 12
    config.EQ_IN_FIFO_DEPTH = 16

    config.ITag_TRIGGER_Th_H = config.ITag_TRIGGER_Th_V = 80
    config.ITag_MAX_NUM_H = config.ITag_MAX_NUM_V = 1
    config.ETag_BOTHSIDE_UPGRADE = 0
    config.SLICE_PER_LINK = 8

    config.GDMA_RW_GAP = np.inf
    config.SDMA_RW_GAP = np.inf
    config.CHANNEL_SPEC = {
        "gdma": 2,
        "sdma": 2,
        "cdma": 1,
        "ddr": 2,
        "l2m": 2,
    }

    print(f"   拓扑: {config.NUM_ROW}x{config.NUM_COL} ({config.NUM_NODE}个节点)")
    print(f"   IP节点数: {config.NUM_IP}")
    print(f"   RB输入FIFO深度: {config.RB_IN_FIFO_DEPTH}")
    print(f"   RB输出FIFO深度: {config.RB_OUT_FIFO_DEPTH}")

    # 创建仿真实例（使用v2版本）
    result_dir = os.path.join(project_root, "Result", "CrossRing_v2")
    os.makedirs(result_dir, exist_ok=True)

    sim = BaseModel(
        model_type="REQ_RSP",
        config=config,
        topo_type="5x2",
        traffic_file_path=f"../traffic/0617",
        traffic_config=[["Read_burst4_2262HBM_v2.txt"]],
        # traffic_file_path=f"../test_data/",
        # traffic_config=[["test1.txt"]],
        result_save_path=result_dir + "/",
        verbose=1,  # 启用详细输出
        print_trace=0,
        show_trace_id=0,
        plot_link_state=1,
        plot_start_time=600,
        plot_flow_fig=1,
        plot_RN_BW_fig=1,
    )

    sim.initial()
    sim.end_time = 3000
    sim.print_interval = 500

    start_time = time.time()
    sim.run()
    end_time = time.time()

    print(f"\n✓ 仿真完成! 用时: {end_time - start_time:.2f}秒")


if __name__ == "__main__":
    print("启动CrossRing v2.0:")

    results = run_bidirectional_rb_demo()
