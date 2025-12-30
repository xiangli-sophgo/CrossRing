from src.kcin import *
from src.kcin.config import KCINConfig
import numpy as np


def main():
    """
    This is an example script to run a simple CrossRing simulation.
    """
    # Define the model type
    model_type = "REQ_RSP"

    # Define the path to the configuration file - use topology-specific YAML config
    kcin_type = "5x4"
    config = KCINConfig(f"../config/topologies/kcin_{kcin_type}.yaml")

    # Initialize the simulation model
    sim: BaseModel = eval(f"{model_type}_model")(
        model_type=model_type,
        config=config,
        kcin_type=kcin_type,
        verbose=1,
    )

    # Configure traffic scheduler
    sim.setup_traffic_scheduler(
        traffic_file_path="",
        traffic_chains=[["test_data.txt"]],
    )

    # Configure result analysis
    sim.setup_result_analysis(
        result_save_path="../Result/",
        results_fig_save_path="",
        plot_flow_fig=True,
        plot_RN_BW_fig=False,
    )

    # Configure visualization
    sim.setup_visualization(plot_link_state=False)

    # Run the simulation
    sim.run_simulation(max_time=1000, print_interval=1000)


if __name__ == "__main__":
    main()
