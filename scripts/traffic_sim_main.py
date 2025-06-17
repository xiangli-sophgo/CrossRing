from src.traffic_process import step1_flatten, step2_hash_addr2node, step5_data_merge, step6_map_to_ch
from src.core import *
import os
from src.utils.component import Flit, Network, Node
from config.config import CrossRingConfig
import csv
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
import logging

# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


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

    # step 6: map to channel
    step6_map_to_ch.main(f"{output_path}/step5_data_merge", f"{output_path}/step6_ch_map")


def run_single_simulation(sim_params):
    """Run a single simulation - designed to be called in parallel"""
    (config_path, traffic_path, model_type, results_file_name, file_name, result_save_path, results_fig_save_path, output_csv) = sim_params

    try:
        print(f"Starting simulation for {file_name} on process {os.getpid()}")

        # Load simulation config
        config = CrossRingConfig(config_path)
        if not config.TOPO_TYPE:
            topo_type = "5x4"  # Default topology
        else:
            topo_type = config.TOPO_TYPE
        config.TOPO_TYPE = topo_type

        # Create simulation instance
        sim: BaseModel = eval(f"{model_type}_model")(
            model_type=model_type,
            config=config,
            topo_type=topo_type,
            traffic_file_path=traffic_path,
            traffic_config=file_name,
            result_save_path=result_save_path + file_name[:-4] + "/",
            results_fig_save_path=results_fig_save_path,
            plot_flow_fig=1,
            plot_RN_BW_fig=1,
            verbose=0,
        )

        # Set simulation parameters
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
        sim.config.DDR_R_LATENCY_original = 150
        sim.config.DDR_R_LATENCY_VAR_original = 0
        sim.config.DDR_W_LATENCY_original = 0
        sim.config.L2M_R_LATENCY_original = 12
        sim.config.L2M_W_LATENCY_original = 16
        sim.config.IQ_CH_FIFO_DEPTH = 10
        sim.config.EQ_CH_FIFO_DEPTH = 10
        sim.config.IQ_OUT_FIFO_DEPTH = 8
        sim.config.RB_IN_FIFO_DEPTH = 16
        sim.config.RB_OUT_FIFO_DEPTH = 8
        sim.config.EQ_IN_FIFO_DEPTH = 16
        sim.config.TL_Etag_T2_UE_MAX = 8
        sim.config.TL_Etag_T1_UE_MAX = 15
        sim.config.TR_Etag_T2_UE_MAX = 12
        sim.config.TU_Etag_T2_UE_MAX = 8
        sim.config.TU_Etag_T1_UE_MAX = 15
        sim.config.TD_Etag_T2_UE_MAX = 12
        sim.config.ITag_TRIGGER_Th_H = sim.config.ITag_TRIGGER_Th_V = 80
        sim.config.ITag_MAX_NUM_H = sim.config.ITag_MAX_NUM_V = 1
        sim.config.ETag_BOTHSIDE_UPGRADE = 0

        sim.config.GDMA_RW_GAP = np.inf
        sim.config.SDMA_RW_GAP = np.inf
        sim.config.CHANNEL_SPEC = {
            "gdma": 2,
            "sdma": 2,
            "ddr": 4,
            "l2m": 2,
        }

        sim.initial()
        sim.print_interval = 5000
        sim.run()

        # Get results
        results = sim.get_results()

        print(f"Completed simulation for {file_name} on process {os.getpid()}")
        return (file_name, results, output_csv)

    except Exception:
        logging.exception(f"Simulation failed for {file_name}")
        return (file_name, None, output_csv)


def save_results_to_csv(results_data):
    """Save simulation results to CSV file with thread safety"""
    file_name, results, output_csv = results_data

    if results is None:
        print(f"Skipping CSV write for {file_name} due to simulation error")
        return

    # Use portalocker for cross-platform file locking
    try:
        import portalocker

        use_portalocker = True
    except ImportError:
        print("Warning: portalocker not available, using threading.Lock instead")
        use_portalocker = False

    csv_file_exists = os.path.isfile(output_csv)

    if use_portalocker:
        with open(output_csv, mode="a", newline="") as output_csv_file:
            # Lock the file for exclusive access (cross-platform)
            portalocker.lock(output_csv_file, portalocker.LOCK_EX)

            writer = csv.DictWriter(output_csv_file, fieldnames=results.keys())
            if not csv_file_exists:
                writer.writeheader()
            writer.writerow(results)

            # File is automatically unlocked when closed
    else:
        # Fallback: use a global lock (less ideal but works)
        import threading

        if not hasattr(save_results_to_csv, "_lock"):
            save_results_to_csv._lock = threading.Lock()

        with save_results_to_csv._lock:
            with open(output_csv, mode="a", newline="") as output_csv_file:
                writer = csv.DictWriter(output_csv_file, fieldnames=results.keys())
                if not csv_file_exists:
                    writer.writeheader()
                writer.writerow(results)

    print(f"Results for {file_name} saved to CSV")


def run_simulation(config_path, traffic_path, model_type, results_file_name, max_workers=None):
    """Run network simulation with processed data using parallel processing"""

    # Get all processed traffic files
    all_files = os.listdir(traffic_path)
    file_names = [file for file in all_files if file.endswith(".txt")]

    if not file_names:
        print("No traffic files found to process!")
        return

    print(f"Found {len(file_names)} traffic files to process")

    # Setup result paths
    result_save_path = f"../Result/CrossRing/{model_type}/{results_file_name}/"
    results_fig_save_path = f"../Result/Plt_IP_BW/{model_type}/{results_file_name}/"
    output_csv = os.path.join(r"../Result/Traffic_result_csv/", f"{results_file_name}.csv")
    os.makedirs(result_save_path, exist_ok=True)

    # Ensure the CSV output directory exists
    csv_dir = os.path.dirname(output_csv)
    os.makedirs(csv_dir, exist_ok=True)

    # Determine number of workers
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    print(f"Using {max_workers} parallel workers for {len(file_names)} simulations")

    # Prepare simulation parameters for each file
    sim_params_list = []
    for file_name in file_names:
        sim_params = (config_path, traffic_path, model_type, results_file_name, file_name, result_save_path, results_fig_save_path, output_csv)
        sim_params_list.append(sim_params)

    # Run simulations in parallel
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all simulation tasks
        future_to_file = {executor.submit(run_single_simulation, params): params[4] for params in sim_params_list}

        # Process completed simulations and save results
        completed = 0
        for future in future_to_file:
            file_name = future_to_file[future]
            try:
                result_data = future.result()
                save_results_to_csv(result_data)
                completed += 1
                print(f"Progress: {completed}/{len(file_names)} simulations completed")
            except Exception:
                logging.exception(f"Error processing {file_name}")

    end_time = time.time()
    print(f"All simulations completed in {end_time - start_time:.2f} seconds")
    print(f"Result output: {csv_dir}")


def main():
    parser = argparse.ArgumentParser(description="Network Traffic Processing and Simulation")
    parser.add_argument("--raw_traffic_input", default="../traffic/original/DeepSeek3-671B-A37B-S4K-O1-W8A8-B32-Decode/", help="Input traffic data path")
    parser.add_argument("--traffic_output", default="../traffic/DeepSeek_0616", help="Output directory for processed data")
    parser.add_argument("--outstanding", type=int, default=2048, help="Outstanding number (must be power of 2)")
    parser.add_argument("--config", default="../config/config2.json", help="Simulation config file path")
    parser.add_argument("--model", default="REQ_RSP", choices=["Feature", "REQ_RSP", "Packet_Base"], help="Simulation model type")
    parser.add_argument("--results_file_name", default="DeepSeek_0617_mix_new", help="Base name for results files")
    parser.add_argument("--mode", default=1, choices=[0, 1, 2], help="Execution mode: 0 for data processing only, 1 for simulation only, 2 for both")
    # parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of parallel workers (default: number of CPU cores)")
    parser.add_argument("--max_workers", type=int, default=2, help="Maximum number of parallel workers (default: number of CPU cores)")

    args = parser.parse_args()

    if args.mode in [0, 2]:
        print("Processing traffic data...")
        process_traffic_data(args.raw_traffic_input, args.traffic_output, args.outstanding)

    if args.mode in [1, 2]:
        print("Running parallel simulations...")
        processed_data_path = f"{args.traffic_output}/step6_ch_map/"
        # processed_data_path = f"{args.traffic_output}"
        run_simulation(args.config, processed_data_path, args.model, args.results_file_name, args.max_workers)


if __name__ == "__main__":
    main()
