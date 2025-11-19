"""调试flit移动"""
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.noc.REQ_RSP import REQ_RSP_model
from config.config import CrossRingConfig
import numpy as np

config_path = os.path.join(project_root, "config", "topologies", "topo_4x5.yaml")
config = CrossRingConfig(config_path)

sim = REQ_RSP_model(
    model_type="REQ_RSP",
    config=config,
    topo_type="4x5",
    verbose=0,
)

sim.setup_traffic_scheduler(
    traffic_file_path=os.path.join(project_root, "test_data"),
    traffic_chains=[["LLama2_AllReduce.txt"]],
)

sim.setup_result_analysis(plot_RN_BW_fig=False, plot_flow_fig=False)

np.random.seed(801)
sim.initial()

# 运行几个周期
sim.cycle = 0
for cycle in range(80):
    sim.cycle = cycle
    sim.step()

    # 在cycle 75附近检查
    if cycle >= 72 and cycle <= 79:
        print(f"\n=== Cycle {sim.cycle} ===")

        # 检查node 12的gdma_0
        node_id = 12
        ip_type = "gdma_0"

        # 检查IQ_channel_buffer_pre
        pre_val = sim.req_network.IQ_channel_buffer_pre[ip_type][node_id]
        print(f"IQ_channel_buffer_pre[{ip_type}][{node_id}]: {pre_val}")

        # 检查IQ_channel_buffer
        buffer_len = len(sim.req_network.IQ_channel_buffer[ip_type][node_id])
        print(f"IQ_channel_buffer[{ip_type}][{node_id}] length: {buffer_len}")

        if buffer_len > 0:
            flit = list(sim.req_network.IQ_channel_buffer[ip_type][node_id])[0]
            print(f"  First flit: {flit.packet_id}.{flit.flit_id} pos={flit.flit_position}")

        # 检查仲裁器输入
        arb_len = len(sim.req_network.IQ_arbiter_input_fifo[ip_type][node_id])
        print(f"IQ_arbiter_input_fifo[{ip_type}][{node_id}] length: {arb_len}")
