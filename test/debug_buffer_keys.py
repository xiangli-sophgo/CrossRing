"""调试buffer keys"""
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.noc.REQ_RSP import REQ_RSP_model
from config.config import CrossRingConfig

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

print(f"Traffic解析后CH_NAME_LIST: {config.CH_NAME_LIST}")
print(f"Traffic解析后CHANNEL_SPEC: {config.CHANNEL_SPEC}")

print("\n初始化模型...")
sim.initial()

print(f"\n检查req_network的buffer keys:")
print(f"IQ_channel_buffer keys: {list(sim.req_network.IQ_channel_buffer.keys())}")
print(f"IQ_channel_buffer_pre keys: {list(sim.req_network.IQ_channel_buffer_pre.keys())}")
print(f"IQ_arbiter_input_fifo keys: {list(sim.req_network.IQ_arbiter_input_fifo.keys())}")
print(f"IQ_arbiter_input_fifo_pre keys: {list(sim.req_network.IQ_arbiter_input_fifo_pre.keys())}")

print(f"\n检查一个buffer的内容:")
for ip_type in list(sim.req_network.IQ_channel_buffer.keys())[:2]:
    print(f"\nip_type={ip_type}:")
    print(f"  type: {type(sim.req_network.IQ_channel_buffer[ip_type])}")
    print(f"  内容: {dict(sim.req_network.IQ_channel_buffer[ip_type])}")
