# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CrossRing is a modeling and simulation tool for CrossRing topology in Network-on-Chip (NoC) systems. It enables chip architecture designers and researchers to model and analyze NoC performance under various traffic scenarios.

**Technology Stack:**
- Python 3.8+
- Key dependencies: numpy, pandas, matplotlib, scipy, networkx, seaborn, joblib, tqdm, optuna

## Essential Commands

```bash
# Installation (development mode)
pip install -e .

# Run basic example
cd example
python example.py

# Run main simulation
python scripts/model_main.py

# Analyze results
python scripts/result_statistic.py

# Find optimal parameters
python scripts/find_optimal_parameters.py

# Run simulations for different topologies
python scripts/tools/different_topo_runner.py
```

## High-Level Architecture

### Model-Based Design
The system uses an abstract base model pattern:
- `src/core/base_model.py` - BaseModel abstract class
- `src/core/REQ_RSP.py` - Request-Response model implementation
- `src/core/Packet_Base.py` - Packet-based communication model
- `src/core/Ring.py` - Ring topology implementation

### Component System
Network elements are modeled as components in `src/utils/components/`:
- `Flit` - Basic data unit with object pooling for performance
- `Node` - Network nodes with routing logic
- `Network` - Overall network structure
- `IPInterface` - Interface for IP blocks
- `TokenBucket` - Rate limiting component

### Configuration System
- JSON configuration files in `config/` directory
- `CrossRingConfig` class provides Python wrapper for configurations
- Key parameters: topology (nodes, columns, IPs), performance (rates, latencies, buffers), simulation (cycles, timeouts)

### Performance Optimizations
- Object pooling for Flit objects to reduce memory allocation overhead
- LRU caching for repeated computations
- Batch I/O operations for traffic file processing
- Memory optimization using `__slots__` in performance-critical classes

## Key Development Patterns

### Directory Structure
- `traffic/` - Input traffic pattern files
- `../Result/` - Simulation results (outside project root)
- `config/` - Configuration JSON files
- `test_data/` - Test traffic data
- `scripts/` - Simulation and analysis scripts

### Simulation Workflow
1. Configure topology and parameters in JSON config files
2. Prepare traffic pattern files in `traffic/` directory
3. Run simulation using scripts
4. Results automatically saved to `../Result/`
5. Use analysis scripts for visualization and statistics

### Model Selection
- REQ_RSP model: For request-response traffic patterns
- Packet_Base model: For packet-switched communication
- Feature model: For feature-specific simulations

### D2D Flit State Design

The D2D (Die-to-Die) communication uses a unified attribute design for clear state management:

#### D2D Attributes (6 total)
```python
# Transaction-level attributes (immutable throughout transaction)
d2d_origin_die      # Initiator Die ID (e.g., 0)
d2d_origin_node     # Initiator node source mapping (e.g., 36 for GDMA)
d2d_origin_type     # Initiator IP type (e.g., "gdma_0")

d2d_target_die      # Target Die ID (e.g., 1)
d2d_target_node     # Target node source mapping (e.g., 4 for DDR)
d2d_target_type     # Target IP type (e.g., "ddr_0")
```

#### Key Design Principles
1. **Unified Source Mapping**: All `d2d_*_node` attributes store source mapping positions
2. **Path Calculation**: Use `node_map(node, is_source=False)` to convert to destination mapping when calculating paths
3. **Clear Naming**: `d2d_` prefix distinguishes D2D attributes from Die-internal attributes
4. **Immutable Transaction Info**: D2D attributes remain constant throughout the transaction lifecycle

#### Usage Examples
```python
# Check if cross-die transfer needed
if hasattr(flit, "d2d_target_die") and flit.d2d_target_die != self.die_id:
    # Cross-die transfer required

# Data return to originator (Stage 6)
source = self.ip_pos  # D2D_SN source position (36)
destination = node_map(flit.d2d_origin_node, is_source=False)  # GDMA destination (32)
destination_type = flit.d2d_origin_type  # "gdma_0"
```

#### Migration from Legacy Attributes
- Replaced: `source_die_id`, `target_die_id`, `source_node_id_physical`, etc.
- New unified design eliminates confusion with `_physical` suffixes
- Consistent with 6-stage D2D flow: Request(Die0→Die1) → Data(Die1→Die0)

## Important Notes

- **Testing**: No established testing framework. `scripts/test.py` exists but no pytest/unittest setup
- **Linting**: No linting configuration. Consider using standard Python tools (flake8, black)
- **Environment**: Uses `.env` file for environment variables
- **Active Development**: Check git status for uncommitted changes in `scripts/`
- **Results Location**: Results saved to `../Result/` (parent directory of project root)
- **Traffic Processing**: Comprehensive traffic file parsing in `src/traffic_process/`
- **Visualization**: Multiple visualization tools in `scripts/tools/`