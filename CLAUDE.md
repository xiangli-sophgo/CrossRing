# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CrossRing is a Network-on-Chip (NoC) modeling and simulation framework focusing on CrossRing topology architectures. It's designed for chip architecture researchers and designers to analyze NoC performance under various traffic scenarios.

## Development Commands

### Installation and Setup
```bash
# Install in development mode
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

### Main Simulation Commands
```bash
# Run main simulation with default configuration
python scripts/model_main.py

# Run traffic processing simulation
python scripts/traffic_sim_main.py

# Run parameter optimization
python scripts/find_optimal_parameters.py

# Generate and process results
python scripts/result_statistic.py
```

### Configuration Management
- Primary config file: `config/config.json` or `config/config2.json`
- Modify config parameters through the `CrossRingConfig` class in `config/config.py`
- Traffic files are located in `traffic/` directory with various subdirectories for different scenarios

## Core Architecture

### Three-Tier Model Architecture
The framework supports three main simulation models:
1. **REQ_RSP**: Request-Response model (`src/core/REQ_RSP.py`)
2. **Packet_Base**: Packet-based model (`src/core/Packet_Base.py`) 
3. **Feature**: Feature-based model (in development)

### Key Components

#### Base Model (`src/core/base_model.py`)
- Central simulation engine implementing the main simulation loop
- Handles packet injection, network traversal, and statistics collection
- Supports multiple topology configurations (3x3, 4x5, 5x4, 6x5, etc.)

#### Network Components (`src/utils/component.py`)
- `Flit`: Basic data unit for network communication
- `Network`: Core network abstraction with adjacency matrix routing
- `Node`: Network node with IP interfaces and buffer management
- `IPInterface`: Interface handling IP-network interactions

#### Ring Topology (`src/core/Ring.py`)
- Dedicated Ring topology implementation
- Supports bidirectional (CW/CCW) routing with adaptive routing capabilities
- Specialized for ring-based NoC architectures

#### Traffic Processing (`src/traffic_process/`)
- Multi-stage traffic analysis pipeline
- Traffic flattening, node mapping, and data merging utilities
- Support for real traffic traces and synthetic patterns

### Configuration Patterns

#### Topology-Specific Configuration
Different topologies require specific parameter tuning:
- **3x3 (SG2260E)**: Lower burst sizes, specialized IP counts
- **4x5/5x4 (SG2262)**: Higher tracker counts, different latency settings  
- **6x5 (SG2260)**: Balanced configuration for larger topologies

#### Traffic Configuration
Traffic can be configured as:
- Single file: `traffic_config = "filename.txt"`
- Multiple files: `traffic_config = [["file1.txt", "file2.txt"]]`
- Parallel chains: `traffic_config = [["chain1.txt"], ["chain2.txt"]]`

### Result Processing and Visualization

#### Result Storage
- Results saved to `Result/` directory with timestamp subdirectories
- CSV files for quantitative analysis
- PNG visualizations for bandwidth and flow analysis

#### Visualization Types
- Flow graphs showing traffic patterns between nodes
- Bandwidth utilization curves for each IP type
- Link state visualization for debugging network congestion

## Development Patterns

### Adding New Topologies
1. Create topology-specific configuration in `scripts/model_main.py`
2. Define node count, IP distributions, and buffer sizes
3. Set appropriate tracker sizes and latency parameters
4. Configure channel specifications in `CHANNEL_SPEC`

### Traffic File Format
Traffic files use CSV format: `time,src,src_type,dst,dst_type,operation,burst_length`
- `operation`: "R" for read, "W" for write
- `src_type`/`dst_type`: IP types like "gdma_0", "ddr_1", etc.

### Performance Analysis
Use the result processing pipeline:
1. `BandwidthAnalyzer` for throughput analysis
2. `NetworkLinkVisualizer` for real-time network state
3. Custom statistics collection through `base_model.stats`

## Special Considerations

### Memory and Performance
- Large traffic files may require memory optimization
- Use `@lru_cache` decorators for expensive computations
- Traffic processing can be parallelized using the tools in `scripts/tools/`

### Debugging
- Enable verbose output: `sim.verbose = 1`
- Use Link State Visualizer: `sim.plot_link_state = 1`
- Traffic tracing: `sim.print_trace = 1`

### Optimal Parameter Finding
The framework includes optimization tools:
- Automated parameter space exploration in `scripts/find_optimal_parameters.py`
- Progress tracking with JSON files for long-running optimizations
- Multi-objective optimization support for latency/throughput trade-offs