from src.core import *
import os
from src.utils.component import Flit, Network, Node
from config.config import CrossRingConfig
import matplotlib
import numpy as np
import sys
import tracemalloc

if sys.platform == "darwin":  # macOS 的系统标识是 'darwin'
    matplotlib.use("macosx")  # 仅在 macOS 上使用该后端


def main():
    # traffic_file_path = r"../test_data/"
    traffic_file_path = r"../traffic/0617/"
    # traffic_file_path = r"../traffic/nxn_traffics"

    traffic_config = [
        [
            # r"Read_burst4_2262HBM_v2.txt",
            # r"MLP_MoE.txt",
        ]
        * 3,
        [
            # r"All2All_Combine.txt",
            r"All2All_Dispatch.txt",
            # r"R_5x2.txt"
        ],
    ]

    # model_type = "Feature"
    model_type = "REQ_RSP"
    # model_type = "Packet_Base"

    results_fig_save_path = None

    result_save_path = f"../Result/CrossRing/{model_type}/"
    # results_fig_save_path = f"../Result/Plt_IP_BW/{model_type}/"

    config_path = r"../config/config2.json"
    config = CrossRingConfig(config_path)
    if not config.TOPO_TYPE:
        # topo_type = "4x9"
        # topo_type = "9x4"
        topo_type = "5x4"  # SG2262
        # topo_type = "4x5"
        # topo_type = "4x2"
        # topo_type = "3x1"
        # topo_type = "6x5"  # SG2260
        # topo_type = "3x3"  # SG2260E
    else:
        topo_type = config.TOPO_TYPE

    config.TOPO_TYPE = topo_type

    # result_save_path = None
    # config_path = r"config.json"
    sim: BaseModel = eval(f"{model_type}_model")(
        model_type=model_type,
        config=config,
        topo_type=topo_type,
        traffic_file_path=traffic_file_path,
        traffic_config=traffic_config,
        result_save_path=result_save_path,
        results_fig_save_path=results_fig_save_path,
        plot_flow_fig=1,
        flow_fig_show_CDMA=1,
        plot_RN_BW_fig=1,
        plot_link_state=1,
        plot_start_time=20,
        print_trace=0,
        show_trace_id=7,
        show_node_id=4,
        verbose=1,
    )
    np.random.seed(617)

    if topo_type == "3x3":
        sim.config.BURST = 2
        sim.config.NUM_IP = 8
        sim.config.NUM_DDR = 16
        sim.config.NUM_L2M = 4
        sim.config.NUM_GDMA = 4
        sim.config.NUM_SDMA = 4
        sim.config.NUM_RN = 8
        sim.config.NUM_SN = 20
        sim.config.RN_R_TRACKER_OSTD = 128
        sim.config.RN_W_TRACKER_OSTD = 32
        sim.config.RN_RDB_SIZE = sim.config.RN_R_TRACKER_OSTD * sim.config.BURST
        sim.config.RN_WDB_SIZE = sim.config.RN_W_TRACKER_OSTD * sim.config.BURST
        sim.config.SN_DDR_R_TRACKER_OSTD = 32
        sim.config.SN_DDR_W_TRACKER_OSTD = 16
        sim.config.SN_L2M_R_TRACKER_OSTD = 64
        sim.config.SN_L2M_W_TRACKER_OSTD = 64
        sim.config.SN_DDR_WDB_SIZE = sim.config.SN_DDR_W_TRACKER_OSTD * sim.config.BURST
        sim.config.SN_L2M_WDB_SIZE = sim.config.SN_L2M_W_TRACKER_OSTD * sim.config.BURST
        sim.config.DDR_R_LATENCY_original = 155
        sim.config.DDR_R_LATENCY_VAR_original = 25
        sim.config.DDR_W_LATENCY_original = 16
        sim.config.L2M_R_LATENCY_original = 12
        sim.config.L2M_W_LATENCY_original = 16
        sim.config.DDR_BW_LIMIT = 76.8 / 4
        sim.config.L2M_BW_LIMIT = np.inf
        sim.config.IQ_CH_FIFO_DEPTH = 8
        sim.config.EQ_CH_FIFO_DEPTH = 8
        sim.config.IQ_OUT_FIFO_DEPTH = 8
        sim.config.RB_OUT_FIFO_DEPTH = 8
        sim.config.SLICE_PER_LINK = 8

        # sim.config.EQ_IN_FIFO_DEPTH = 8
        # sim.config.RB_IN_FIFO_DEPTH = 8
        # sim.config.TL_Etag_T2_UE_MAX = 4
        # sim.config.TL_Etag_T1_UE_MAX = 7
        # sim.config.TR_Etag_T2_UE_MAX = 5
        # sim.config.TU_Etag_T2_UE_MAX = 4
        # sim.config.TU_Etag_T1_UE_MAX = 7
        # sim.config.TD_Etag_T2_UE_MAX = 6

        sim.config.TL_Etag_T2_UE_MAX = 8
        sim.config.TL_Etag_T1_UE_MAX = 12
        sim.config.TR_Etag_T2_UE_MAX = 12
        sim.config.RB_IN_FIFO_DEPTH = 16
        sim.config.TU_Etag_T2_UE_MAX = 8
        sim.config.TU_Etag_T2_UE_MAX = 12
        sim.config.TD_Etag_T2_UE_MAX = 12
        sim.config.EQ_IN_FIFO_DEPTH = 16

        sim.config.GDMA_RW_GAP = np.inf
        # sim.config.SDMA_RW_GAP = np.inf
        sim.config.SDMA_RW_GAP = 50
        sim.config.ITag_TRIGGER_Th_H = sim.config.ITag_TRIGGER_Th_V = 80
        sim.config.ITag_MAX_NUM_H = sim.config.ITag_MAX_NUM_V = 1
        sim.config.CHANNEL_SPEC = {
            "gdma": 1,
            "sdma": 1,
            "ddr": 4,
            "l2m": 2,
        }

    elif topo_type in ["5x4", "4x5"]:
        sim.config.NUM_COL = 4
        sim.config.NUM_NODE = 40
        sim.config.BURST = 4
        sim.config.NUM_IP = 32
        sim.config.NUM_DDR = 32
        sim.config.NUM_L2M = 32
        sim.config.NUM_GDMA = 32
        sim.config.NUM_SDMA = 32
        sim.config.NUM_RN = 32
        sim.config.NUM_SN = 32
        sim.config.RN_R_TRACKER_OSTD = 64
        sim.config.RN_W_TRACKER_OSTD = 64
        sim.config.RN_RDB_SIZE = sim.config.RN_R_TRACKER_OSTD * sim.config.BURST
        sim.config.RN_WDB_SIZE = sim.config.RN_W_TRACKER_OSTD * sim.config.BURST
        sim.config.SN_DDR_R_TRACKER_OSTD = 64
        sim.config.SN_DDR_W_TRACKER_OSTD = 64
        sim.config.SN_L2M_R_TRACKER_OSTD = 64
        sim.config.SN_L2M_W_TRACKER_OSTD = 64
        sim.config.SN_DDR_WDB_SIZE = sim.config.SN_DDR_W_TRACKER_OSTD * sim.config.BURST
        sim.config.SN_L2M_WDB_SIZE = sim.config.SN_L2M_W_TRACKER_OSTD * sim.config.BURST
        sim.config.DDR_R_LATENCY_original = 40
        sim.config.DDR_R_LATENCY_VAR_original = 0
        sim.config.DDR_W_LATENCY_original = 0
        sim.config.L2M_R_LATENCY_original = 12
        sim.config.L2M_W_LATENCY_original = 16
        sim.config.IQ_CH_FIFO_DEPTH = 10
        sim.config.EQ_CH_FIFO_DEPTH = 10
        sim.config.IQ_OUT_FIFO_DEPTH = 8
        sim.config.RB_OUT_FIFO_DEPTH = 8
        sim.config.SN_TRACKER_RELEASE_LATENCY = 40
        sim.config.CDMA_BW_LIMIT = 8

        # sim.config.EQ_IN_FIFO_DEPTH = 8
        # sim.config.RB_IN_FIFO_DEPTH = 8
        # sim.config.TL_Etag_T2_UE_MAX = 4
        # sim.config.TL_Etag_T1_UE_MAX = 7
        # sim.config.TR_Etag_T2_UE_MAX = 5
        # sim.config.TU_Etag_T2_UE_MAX = 4
        # sim.config.TU_Etag_T1_UE_MAX = 7
        # sim.config.TD_Etag_T2_UE_MAX = 6

        sim.config.TL_Etag_T2_UE_MAX = 8
        sim.config.TL_Etag_T1_UE_MAX = 15
        sim.config.TR_Etag_T2_UE_MAX = 12
        sim.config.RB_IN_FIFO_DEPTH = 16
        sim.config.TU_Etag_T2_UE_MAX = 8
        sim.config.TU_Etag_T1_UE_MAX = 15
        sim.config.TD_Etag_T2_UE_MAX = 12
        sim.config.EQ_IN_FIFO_DEPTH = 16

        sim.config.ITag_TRIGGER_Th_H = sim.config.ITag_TRIGGER_Th_V = 80
        sim.config.ITag_MAX_NUM_H = sim.config.ITag_MAX_NUM_V = 1
        sim.config.ETag_BOTHSIDE_UPGRADE = 0
        sim.config.SLICE_PER_LINK = 8

        sim.config.GDMA_RW_GAP = np.inf
        sim.config.SDMA_RW_GAP = np.inf
        sim.config.CHANNEL_SPEC = {
            "gdma": 2,
            "sdma": 2,
            "cdma": 1,
            "ddr": 2,
            "l2m": 2,
        }
    elif topo_type in ["6x5"]:
        sim.config.BURST = 2
        sim.config.RN_R_TRACKER_OSTD = 64
        sim.config.RN_W_TRACKER_OSTD = 64
        sim.config.RN_RDB_SIZE = sim.config.RN_R_TRACKER_OSTD * sim.config.BURST
        sim.config.RN_WDB_SIZE = sim.config.RN_W_TRACKER_OSTD * sim.config.BURST
        sim.config.SN_DDR_R_TRACKER_OSTD = 64
        sim.config.SN_DDR_W_TRACKER_OSTD = 64
        sim.config.SN_L2M_R_TRACKER_OSTD = 64
        sim.config.SN_L2M_W_TRACKER_OSTD = 64
        sim.config.SN_DDR_WDB_SIZE = sim.config.SN_DDR_W_TRACKER_OSTD * sim.config.BURST
        sim.config.SN_L2M_WDB_SIZE = sim.config.SN_L2M_W_TRACKER_OSTD * sim.config.BURST
        sim.config.DDR_R_LATENCY_original = 40
        sim.config.DDR_R_LATENCY_VAR_original = 0
        sim.config.DDR_W_LATENCY_original = 0
        sim.config.L2M_R_LATENCY_original = 12
        sim.config.L2M_W_LATENCY_original = 16
        sim.config.IQ_CH_FIFO_DEPTH = 10
        sim.config.EQ_CH_FIFO_DEPTH = 10
        sim.config.IQ_OUT_FIFO_DEPTH = 8
        sim.config.RB_OUT_FIFO_DEPTH = 8
        sim.config.SN_TRACKER_RELEASE_LATENCY = 40
        sim.config.CDMA_BW_LIMIT = 8

        sim.config.TL_Etag_T2_UE_MAX = 8
        sim.config.TL_Etag_T1_UE_MAX = 15
        sim.config.TR_Etag_T2_UE_MAX = 12
        sim.config.RB_IN_FIFO_DEPTH = 16
        sim.config.TU_Etag_T2_UE_MAX = 8
        sim.config.TU_Etag_T1_UE_MAX = 15
        sim.config.TD_Etag_T2_UE_MAX = 12
        sim.config.EQ_IN_FIFO_DEPTH = 16

        sim.config.ITag_TRIGGER_Th_H = sim.config.ITag_TRIGGER_Th_V = 80
        sim.config.ITag_MAX_NUM_H = sim.config.ITag_MAX_NUM_V = 1
        sim.config.ETag_BOTHSIDE_UPGRADE = 0
        sim.config.SLICE_PER_LINK = 8

        sim.config.GDMA_RW_GAP = np.inf
        sim.config.SDMA_RW_GAP = np.inf
        sim.config.CHANNEL_SPEC = {
            "gdma": 2,
            "sdma": 2,
            "cdma": 1,
            "ddr": 2,
            "l2m": 2,
        }
    elif topo_type in ["4x2"]:
        sim.config.BURST = 4
        sim.config.RN_R_TRACKER_OSTD = 64
        sim.config.RN_W_TRACKER_OSTD = 32
        sim.config.RN_RDB_SIZE = sim.config.RN_R_TRACKER_OSTD * sim.config.BURST
        sim.config.RN_WDB_SIZE = sim.config.RN_W_TRACKER_OSTD * sim.config.BURST
        sim.config.SN_DDR_R_TRACKER_OSTD = 96
        sim.config.SN_DDR_W_TRACKER_OSTD = 48
        sim.config.SN_L2M_R_TRACKER_OSTD = 96
        sim.config.SN_L2M_W_TRACKER_OSTD = 48
        sim.config.SN_DDR_WDB_SIZE = sim.config.SN_DDR_W_TRACKER_OSTD * sim.config.BURST
        sim.config.SN_L2M_WDB_SIZE = sim.config.SN_L2M_W_TRACKER_OSTD * sim.config.BURST
        sim.config.DDR_R_LATENCY_original = 100
        sim.config.DDR_R_LATENCY_VAR_original = 0
        sim.config.DDR_W_LATENCY_original = 0
        sim.config.L2M_R_LATENCY_original = 12
        sim.config.L2M_W_LATENCY_original = 16
        sim.config.IQ_CH_FIFO_DEPTH = 10
        sim.config.EQ_CH_FIFO_DEPTH = 10
        sim.config.IQ_OUT_FIFO_DEPTH = 8
        sim.config.RB_OUT_FIFO_DEPTH = 8
        sim.config.SN_TRACKER_RELEASE_LATENCY = 40
        sim.config.CDMA_BW_LIMIT = 8
        sim.config.DDR_BW_LIMIT = 102

        sim.config.TL_Etag_T2_UE_MAX = 8
        sim.config.TL_Etag_T1_UE_MAX = 15
        sim.config.TR_Etag_T2_UE_MAX = 12
        sim.config.RB_IN_FIFO_DEPTH = 16
        sim.config.TU_Etag_T2_UE_MAX = 8
        sim.config.TU_Etag_T1_UE_MAX = 15
        sim.config.TD_Etag_T2_UE_MAX = 12
        sim.config.EQ_IN_FIFO_DEPTH = 16

        sim.config.ITag_TRIGGER_Th_H = sim.config.ITag_TRIGGER_Th_V = 80
        sim.config.ITag_MAX_NUM_H = sim.config.ITag_MAX_NUM_V = 1
        sim.config.ETag_BOTHSIDE_UPGRADE = 0
        sim.config.SLICE_PER_LINK = 8

        sim.config.GDMA_RW_GAP = np.inf
        sim.config.SDMA_RW_GAP = np.inf
        sim.config.CHANNEL_SPEC = {
            "gdma": 2,
            "sdma": 2,
            "cdma": 1,
            "ddr": 2,
            "l2m": 2,
        }
    else:
        rows, cols = 3, 1
        # 解析行列并计算核心数量
        num_cores = rows * cols
        # 每个核心对应两个网络节点（RN & SN）
        sim.config.NUM_NODE = num_cores * 2
        # 网络行数是核心行数的两倍，列数保持核心列数
        sim.config.NUM_ROW = rows * 2
        sim.config.NUM_COL = cols
        # IP、RN、SN、DMA 等统计基于核心数量
        sim.config.NUM_IP = num_cores
        sim.config.NUM_RN = num_cores
        sim.config.NUM_SN = num_cores
        sim.config.NUM_GDMA = num_cores
        sim.config.NUM_SDMA = num_cores
        sim.config.NUM_DDR = num_cores
        sim.config.NUM_L2M = num_cores
        sim.config.BURST = 4
        sim.config.RN_R_TRACKER_OSTD = 64
        sim.config.RN_W_TRACKER_OSTD = 64
        sim.config.RN_RDB_SIZE = sim.config.RN_R_TRACKER_OSTD * sim.config.BURST
        sim.config.RN_WDB_SIZE = sim.config.RN_W_TRACKER_OSTD * sim.config.BURST
        sim.config.SN_DDR_R_TRACKER_OSTD = 64
        sim.config.SN_DDR_W_TRACKER_OSTD = 64
        sim.config.SN_L2M_R_TRACKER_OSTD = 64
        sim.config.SN_L2M_W_TRACKER_OSTD = 64
        sim.config.SN_DDR_WDB_SIZE = sim.config.SN_DDR_W_TRACKER_OSTD * sim.config.BURST
        sim.config.SN_L2M_WDB_SIZE = sim.config.SN_L2M_W_TRACKER_OSTD * sim.config.BURST
        sim.config.DDR_R_LATENCY_original = 40
        sim.config.DDR_R_LATENCY_VAR_original = 0
        sim.config.DDR_W_LATENCY_original = 0
        sim.config.L2M_R_LATENCY_original = 12
        sim.config.L2M_W_LATENCY_original = 16
        sim.config.IQ_CH_FIFO_DEPTH = 10
        sim.config.EQ_CH_FIFO_DEPTH = 10
        sim.config.IQ_OUT_FIFO_DEPTH = 8
        sim.config.RB_OUT_FIFO_DEPTH = 8
        sim.config.SN_TRACKER_RELEASE_LATENCY = 40
        sim.config.CDMA_BW_LIMIT = 8

        # sim.config.EQ_IN_FIFO_DEPTH = 8
        # sim.config.RB_IN_FIFO_DEPTH = 8
        # sim.config.TL_Etag_T2_UE_MAX = 4
        # sim.config.TL_Etag_T1_UE_MAX = 7
        # sim.config.TR_Etag_T2_UE_MAX = 5
        # sim.config.TU_Etag_T2_UE_MAX = 4
        # sim.config.TU_Etag_T1_UE_MAX = 7
        # sim.config.TD_Etag_T2_UE_MAX = 6

        sim.config.TL_Etag_T2_UE_MAX = 8
        sim.config.TL_Etag_T1_UE_MAX = 15
        sim.config.TR_Etag_T2_UE_MAX = 12
        sim.config.RB_IN_FIFO_DEPTH = 16
        sim.config.TU_Etag_T2_UE_MAX = 8
        sim.config.TU_Etag_T1_UE_MAX = 15
        sim.config.TD_Etag_T2_UE_MAX = 12
        sim.config.EQ_IN_FIFO_DEPTH = 16

        sim.config.ITag_TRIGGER_Th_H = sim.config.ITag_TRIGGER_Th_V = 80
        sim.config.ITag_MAX_NUM_H = sim.config.ITag_MAX_NUM_V = 1
        sim.config.ETag_BOTHSIDE_UPGRADE = 0
        sim.config.SLICE_PER_LINK = 8

        sim.config.GDMA_RW_GAP = np.inf
        sim.config.SDMA_RW_GAP = np.inf
        sim.config.CHANNEL_SPEC = {
            "gdma": 2,
            "sdma": 2,
            "cdma": 1,
            "ddr": 2,
            "l2m": 2,
        }

    sim.initial()
    sim.end_time = 6000
    sim.print_interval = 2000
    sim.run()


if __name__ == "__main__":
    main()
