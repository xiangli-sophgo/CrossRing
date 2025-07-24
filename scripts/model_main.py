from src.core import *
import os
from src.utils.components import *
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
    # traffic_file_path = r"../traffic/DeepSeek_0616/step6_ch_map/"
    # traffic_file_path = r"../traffic/RW_4x2_4x4/"
    # traffic_file_path = r"../traffic/nxn_traffics"

    traffic_config = [
        [
            # r"R_5x2.txt",
            # r"W_5x2.txt",
            # r"Read_burst4_2262HBM_v2.txt",
            # r"Write_burst4_2262HBM_v2.txt",
            # r"MLP_MoE.txt",
        ]
        * 1,
        [
            # r"All2All_Combine.txt",
            # r"All2All_Dispatch.txt",
            # r"full_bw_R_4x5.txt"
            "LLama2_AllReduce.txt"
            # "test1.txt"
            # "LLama2_AttentionFC.txt"
            # "R_4x4.txt"
            # "MLA_B32.txt"
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
    config.CROSSRING_VERSION = "V1"
    if not config.TOPO_TYPE:
        # topo_type = "4x9"
        # topo_type = "9x4"
        topo_type = "5x4"  # SG2262
        # topo_type = "4x4"
        # topo_type = "5x2"
        # topo_type = "3x1"
        # topo_type = "6x5"  # SG2260
        # topo_type = "3x3"  # SG2260E
    else:
        topo_type = config.TOPO_TYPE

    if topo_type == "3x3":
        config.NUM_COL = 3
        config.NUM_NODE = 18
        config.NUM_ROW = 6
        config.BURST = 4
        config.NUM_IP = 8
        config.NUM_DDR = 16
        config.NUM_L2M = 4
        config.NUM_GDMA = 4
        config.NUM_SDMA = 4
        config.NUM_RN = 8
        config.NUM_SN = 20
        config.RN_R_TRACKER_OSTD = 128
        config.RN_W_TRACKER_OSTD = 32
        config.RN_RDB_SIZE = config.RN_R_TRACKER_OSTD * config.BURST
        config.RN_WDB_SIZE = config.RN_W_TRACKER_OSTD * config.BURST
        config.SN_DDR_R_TRACKER_OSTD = 32
        config.SN_DDR_W_TRACKER_OSTD = 16
        config.SN_L2M_R_TRACKER_OSTD = 64
        config.SN_L2M_W_TRACKER_OSTD = 64
        config.SN_DDR_WDB_SIZE = config.SN_DDR_W_TRACKER_OSTD * config.BURST
        config.SN_L2M_WDB_SIZE = config.SN_L2M_W_TRACKER_OSTD * config.BURST
        config.DDR_R_LATENCY_original = 155
        config.DDR_R_LATENCY_VAR_original = 0
        config.DDR_W_LATENCY_original = 16
        config.L2M_R_LATENCY_original = 12
        config.L2M_W_LATENCY_original = 16
        # config.DDR_BW_LIMIT = 76.8 / 4
        # config.L2M_BW_LIMIT = np.inf
        config.IQ_CH_FIFO_DEPTH = 8
        config.EQ_CH_FIFO_DEPTH = 8
        config.IQ_OUT_FIFO_DEPTH = 8
        config.RB_OUT_FIFO_DEPTH = 8
        config.SLICE_PER_LINK = 8

        # config.EQ_IN_FIFO_DEPTH = 8
        # config.RB_IN_FIFO_DEPTH = 8
        # config.TL_Etag_T2_UE_MAX = 4
        # config.TL_Etag_T1_UE_MAX = 7
        # config.TR_Etag_T2_UE_MAX = 5
        # config.TU_Etag_T2_UE_MAX = 4
        # config.TU_Etag_T1_UE_MAX = 7
        # config.TD_Etag_T2_UE_MAX = 6

        config.TL_Etag_T2_UE_MAX = 8
        config.TL_Etag_T1_UE_MAX = 12
        config.TR_Etag_T2_UE_MAX = 12
        config.RB_IN_FIFO_DEPTH = 16
        config.TU_Etag_T2_UE_MAX = 8
        config.TU_Etag_T2_UE_MAX = 12
        config.TD_Etag_T2_UE_MAX = 12
        config.EQ_IN_FIFO_DEPTH = 16

        config.GDMA_RW_GAP = np.inf
        # config.SDMA_RW_GAP = np.inf
        config.SDMA_RW_GAP = 50
        config.ITag_TRIGGER_Th_H = config.ITag_TRIGGER_Th_V = 80
        config.ITag_MAX_NUM_H = config.ITag_MAX_NUM_V = 1
        config.CHANNEL_SPEC = {
            "gdma": 1,
            "sdma": 1,
            "ddr": 4,
            "l2m": 2,
        }

    elif topo_type in ["5x4"]:
        config.NUM_COL = 4
        config.NUM_NODE = 40
        config.NUM_ROW = 10
        config.BURST = 4
        config.NUM_IP = 32
        config.NUM_DDR = 32
        config.NUM_L2M = 32
        config.NUM_GDMA = 32
        config.NUM_SDMA = 32
        config.NUM_RN = 32
        config.NUM_SN = 32
        config.RN_R_TRACKER_OSTD = 64
        config.RN_W_TRACKER_OSTD = 64
        config.SN_DDR_R_TRACKER_OSTD = 64
        config.SN_DDR_W_TRACKER_OSTD = 64
        config.SN_L2M_R_TRACKER_OSTD = 64
        config.SN_L2M_W_TRACKER_OSTD = 64
        config.RN_RDB_SIZE = config.RN_R_TRACKER_OSTD * config.BURST
        config.RN_WDB_SIZE = config.RN_W_TRACKER_OSTD * config.BURST
        config.SN_DDR_WDB_SIZE = config.SN_DDR_W_TRACKER_OSTD * config.BURST
        config.SN_L2M_WDB_SIZE = config.SN_L2M_W_TRACKER_OSTD * config.BURST
        config.DDR_R_LATENCY_original = 40
        config.DDR_R_LATENCY_VAR_original = 0
        config.DDR_W_LATENCY_original = 0
        config.L2M_R_LATENCY_original = 12
        config.L2M_W_LATENCY_original = 16
        config.IQ_CH_FIFO_DEPTH = 2
        config.EQ_CH_FIFO_DEPTH = 4
        config.IQ_OUT_FIFO_DEPTH = 8
        config.RB_OUT_FIFO_DEPTH = 8
        config.SN_TRACKER_RELEASE_LATENCY = 40
        # config.GDMA_BW_LIMIT = 16
        # config.CDMA_BW_LIMIT = 16
        # config.DDR_BW_LIMIT = 128
        config.ENABLE_CROSSPOINT_CONFLICT_CHECK = 0
        config.ENABLE_IN_ORDER_EJECTION = 0

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
    elif topo_type in ["6x5"]:
        config.BURST = 2
        config.RN_R_TRACKER_OSTD = 64
        config.RN_W_TRACKER_OSTD = 64
        config.RN_RDB_SIZE = config.RN_R_TRACKER_OSTD * config.BURST
        config.RN_WDB_SIZE = config.RN_W_TRACKER_OSTD * config.BURST
        config.SN_DDR_R_TRACKER_OSTD = 64
        config.SN_DDR_W_TRACKER_OSTD = 64
        config.SN_L2M_R_TRACKER_OSTD = 64
        config.SN_L2M_W_TRACKER_OSTD = 64
        config.SN_DDR_WDB_SIZE = config.SN_DDR_W_TRACKER_OSTD * config.BURST
        config.SN_L2M_WDB_SIZE = config.SN_L2M_W_TRACKER_OSTD * config.BURST
        config.DDR_R_LATENCY_original = 40
        config.DDR_R_LATENCY_VAR_original = 0
        config.DDR_W_LATENCY_original = 0
        config.L2M_R_LATENCY_original = 12
        config.L2M_W_LATENCY_original = 16
        config.IQ_CH_FIFO_DEPTH = 10
        config.EQ_CH_FIFO_DEPTH = 10
        config.IQ_OUT_FIFO_DEPTH = 8
        config.RB_OUT_FIFO_DEPTH = 8
        config.SN_TRACKER_RELEASE_LATENCY = 40
        config.CDMA_BW_LIMIT = 8

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
    elif topo_type in ["5x2"]:
        config.BURST = 4
        config.NUM_COL = 2
        config.RN_R_TRACKER_OSTD = 64
        config.RN_W_TRACKER_OSTD = 32
        config.RN_RDB_SIZE = config.RN_R_TRACKER_OSTD * config.BURST
        config.RN_WDB_SIZE = config.RN_W_TRACKER_OSTD * config.BURST
        config.SN_DDR_R_TRACKER_OSTD = 96
        config.SN_DDR_W_TRACKER_OSTD = 48
        config.SN_L2M_R_TRACKER_OSTD = 96
        config.SN_L2M_W_TRACKER_OSTD = 48
        config.SN_DDR_WDB_SIZE = config.SN_DDR_W_TRACKER_OSTD * config.BURST
        config.SN_L2M_WDB_SIZE = config.SN_L2M_W_TRACKER_OSTD * config.BURST
        config.DDR_R_LATENCY_original = 100
        config.DDR_R_LATENCY_VAR_original = 0
        config.DDR_W_LATENCY_original = 40
        config.L2M_R_LATENCY_original = 12
        config.L2M_W_LATENCY_original = 16
        config.IQ_CH_FIFO_DEPTH = 10
        config.EQ_CH_FIFO_DEPTH = 10
        config.IQ_OUT_FIFO_DEPTH = 8
        config.RB_OUT_FIFO_DEPTH = 8
        config.SN_TRACKER_RELEASE_LATENCY = 40
        config.CDMA_BW_LIMIT = 8
        config.DDR_BW_LIMIT = 102
        config.GDMA_BW_LIMIT = 102
        config.ENABLE_CROSSPOINT_CONFLICT_CHECK = 1

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
    elif topo_type in ["4x2", "4x4"]:
        config.BURST = 4
        config.NUM_COL = 2 if topo_type == "4x2" else 4
        config.NUM_NODE = 16 if topo_type == "4x2" else 32
        config.NUM_ROW = config.NUM_NODE // config.NUM_COL
        config.NUM_IP = 16
        config.RN_R_TRACKER_OSTD = 64
        config.RN_W_TRACKER_OSTD = 32
        config.RN_RDB_SIZE = config.RN_R_TRACKER_OSTD * config.BURST
        config.RN_WDB_SIZE = config.RN_W_TRACKER_OSTD * config.BURST
        config.NETWORK_FREQUENCY = 2
        config.SN_DDR_R_TRACKER_OSTD = 96
        config.SN_DDR_W_TRACKER_OSTD = 48
        config.SN_L2M_R_TRACKER_OSTD = 96
        config.SN_L2M_W_TRACKER_OSTD = 48
        config.SN_DDR_WDB_SIZE = config.SN_DDR_W_TRACKER_OSTD * config.BURST
        config.SN_L2M_WDB_SIZE = config.SN_L2M_W_TRACKER_OSTD * config.BURST
        config.DDR_R_LATENCY_original = 100
        config.DDR_R_LATENCY_VAR_original = 0
        config.DDR_W_LATENCY_original = 40
        config.L2M_R_LATENCY_original = 12
        config.L2M_W_LATENCY_original = 16
        config.IQ_CH_FIFO_DEPTH = 10
        config.EQ_CH_FIFO_DEPTH = 10
        config.IQ_OUT_FIFO_DEPTH = 8
        config.RB_OUT_FIFO_DEPTH = 8
        config.SN_TRACKER_RELEASE_LATENCY = 40
        config.CDMA_BW_LIMIT = 8
        # config.DDR_BW_LIMIT = 102
        # config.GDMA_BW_LIMIT = 102
        config.ENABLE_CROSSPOINT_CONFLICT_CHECK = 0

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
    else:
        rows, cols = 3, 1
        # 解析行列并计算核心数量
        num_cores = rows * cols
        # 每个核心对应两个网络节点（RN & SN）
        config.NUM_NODE = num_cores * 2
        # 网络行数是核心行数的两倍，列数保持核心列数
        config.NUM_ROW = rows * 2
        config.NUM_COL = cols
        # IP、RN、SN、DMA 等统计基于核心数量
        config.NUM_IP = num_cores
        config.NUM_RN = num_cores
        config.NUM_SN = num_cores
        config.NUM_GDMA = num_cores
        config.NUM_SDMA = num_cores
        config.NUM_DDR = num_cores
        config.NUM_L2M = num_cores
        config.BURST = 4
        config.RN_R_TRACKER_OSTD = 64
        config.RN_W_TRACKER_OSTD = 64
        config.RN_RDB_SIZE = config.RN_R_TRACKER_OSTD * config.BURST
        config.RN_WDB_SIZE = config.RN_W_TRACKER_OSTD * config.BURST
        config.SN_DDR_R_TRACKER_OSTD = 64
        config.SN_DDR_W_TRACKER_OSTD = 64
        config.SN_L2M_R_TRACKER_OSTD = 64
        config.SN_L2M_W_TRACKER_OSTD = 64
        config.SN_DDR_WDB_SIZE = config.SN_DDR_W_TRACKER_OSTD * config.BURST
        config.SN_L2M_WDB_SIZE = config.SN_L2M_W_TRACKER_OSTD * config.BURST
        config.DDR_R_LATENCY_original = 40
        config.DDR_R_LATENCY_VAR_original = 0
        config.DDR_W_LATENCY_original = 0
        config.L2M_R_LATENCY_original = 12
        config.L2M_W_LATENCY_original = 16
        config.IQ_CH_FIFO_DEPTH = 10
        config.EQ_CH_FIFO_DEPTH = 10
        config.IQ_OUT_FIFO_DEPTH = 8
        config.RB_OUT_FIFO_DEPTH = 8
        config.SN_TRACKER_RELEASE_LATENCY = 40
        config.CDMA_BW_LIMIT = 8

        # config.EQ_IN_FIFO_DEPTH = 8
        # config.RB_IN_FIFO_DEPTH = 8
        # config.TL_Etag_T2_UE_MAX = 4
        # config.TL_Etag_T1_UE_MAX = 7
        # config.TR_Etag_T2_UE_MAX = 5
        # config.TU_Etag_T2_UE_MAX = 4
        # config.TU_Etag_T1_UE_MAX = 7
        # config.TD_Etag_T2_UE_MAX = 6

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
        plot_link_state=0,
        plot_start_time=3000,
        print_trace=0,
        show_trace_id=10,
        show_node_id=4,
        verbose=1,
    )
    np.random.seed(722)

    sim.initial()
    sim.end_time = 6000
    sim.print_interval = 1000
    sim.run()


if __name__ == "__main__":
    main()
