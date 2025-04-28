from src.traffic_process import step1_flatten, step2_hash_addr2node, step5_data_merge
from src.core import *
import os
from src.utils.component import Flit, Network, Node
from config.config import SimulationConfig
import csv
import argparse


def process_traffic_data(input_path, output_path, outstanding_num):
    """Process traffic data through the pipeline"""
    # Validate outstanding_num
    assert isinstance(outstanding_num, int), "outstanding_num must be integer or out of range."
    assert outstanding_num > 0, "outstanding_num must be positive integer."
    assert outstanding_num & outstanding_num - 1 == 0, "outstanding_num must be a power of 2."

    outstanding_digit = outstanding_num.bit_length() - 1

    # Step 1: Flatten trace files
    step1_flatten.main(input_path, output_path)

    # Step 2: Hash addresses to node numbers
    hasher = step2_hash_addr2node.AddressHasher(itlv_size=outstanding_num)
    hasher.run(f"{output_path}/step1_flatten", f"{output_path}/step2_hash_addr2node")

    # Step 5: Merge data
    step5_data_merge.main(f"{output_path}/step2_hash_addr2node", output_path)


def run_simulation(config_path, traffic_path, model_type, results_file_name):
    """Run network simulation with processed data"""
    # Get all processed traffic files
    all_files = os.listdir(traffic_path)
    file_names = [file for file in all_files if file.endswith(".txt")]

    # Setup result paths
    result_save_path = f"../Result/CrossRing/{model_type}/{traffic_path.split('/')[-4]}/"
    results_fig_save_path = f"../Result/Plt_IP_BW/{model_type}/{traffic_path.split('/')[-4]}/"
    output_csv = os.path.join(r"../Result/Traffic_result_csv/", f"{results_file_name}.csv")
    os.makedirs(result_save_path, exist_ok=True)

    # Load simulation config
    config = SimulationConfig(config_path)
    if not config.topo_type:
        topo_type = "5x4"  # Default topology
    else:
        topo_type = config.topo_type
    config.topo_type = topo_type

    # Run simulation for each traffic file
    for file_name in file_names:
        sim = eval(f"{model_type}_model")(
            model_type=model_type,
            config=config,
            topo_type=topo_type,
            traffic_file_path=traffic_path,
            file_name=file_name,
            result_save_path=result_save_path + file_name[:-4] + "/",
            results_fig_save_path=results_fig_save_path,
        )
        # Set simulation parameters
        sim.config.burst = 4
        sim.config.num_ips = 32
        sim.config.num_ddr = 32
        sim.config.num_l2m = 32
        sim.config.num_gdma = 32
        sim.config.num_sdma = 32
        sim.config.num_RN = 32
        sim.config.num_SN = 32
        sim.config.rn_read_tracker_ostd = 64
        sim.config.rn_write_tracker_ostd = 64
        sim.config.rn_rdb_size = sim.config.rn_read_tracker_ostd * sim.config.burst
        sim.config.rn_wdb_size = sim.config.rn_write_tracker_ostd * sim.config.burst
        sim.config.sn_ddr_read_tracker_ostd = 128
        sim.config.sn_ddr_write_tracker_ostd = 64
        sim.config.sn_l2m_read_tracker_ostd = 64
        sim.config.sn_l2m_write_tracker_ostd = 64
        sim.config.sn_ddr_wdb_size = sim.config.sn_ddr_write_tracker_ostd * sim.config.burst
        sim.config.sn_l2m_wdb_size = sim.config.sn_l2m_write_tracker_ostd * sim.config.burst
        sim.config.ddr_R_latency_original = 150
        sim.config.ddr_R_latency_var_original = 0
        sim.config.ddr_W_latency_original = 16
        sim.config.l2m_R_latency_original = 12
        sim.config.l2m_W_latency_original = 16

        sim.initial()
        sim.print_interval = 5000
        sim.run()

        # Save results
        results = sim.get_results()
        csv_file_exists = os.path.isfile(output_csv)
        with open(output_csv, mode="a", newline="") as output_csv_file:
            writer = csv.DictWriter(output_csv_file, fieldnames=results.keys())
            if not csv_file_exists:
                writer.writeheader()
            writer.writerow(results)


def main():
    parser = argparse.ArgumentParser(description="Network Traffic Processing and Simulation")
    parser.add_argument("--raw_traffic_input", default="../traffic/original_data/DeepSeek/", help="Input traffic data path")
    parser.add_argument("--traffic_output", default="../traffic/output_DeepSeek_0427/", help="Output directory for processed data")
    parser.add_argument("--outstanding", type=int, default=512, help="Outstanding number (must be power of 2)")
    parser.add_argument("--config", default="../config/config2.json", help="Simulation config file path")
    parser.add_argument("--model", default="REQ_RSP", choices=["Feature", "REQ_RSP", "Packet_Base"], help="Simulation model type")
    parser.add_argument("--results_file_name", default="DeepSeek_0427", help="Base name for results files")
    parser.add_argument("--mode", default=1, choices=[0, 1, 2], help="Execution mode: 0 for data processing only, 1 for simulation only, 2 for both (default)")

    args = parser.parse_args()

    if args.mode in [0, 2]:
        print("Processing traffic data...")
        process_traffic_data(args.raw_traffic_input, args.traffic_output, args.outstanding)

    if args.mode in [1, 2]:
        print("Running simulation...")
        processed_data_path = f"{args.traffic_output}/step5_data_merge/"
        run_simulation(args.config, processed_data_path, args.model, args.results_file_name)


if __name__ == "__main__":
    main()
