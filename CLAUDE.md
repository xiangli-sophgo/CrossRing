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

## D2D Communication Flow and Tracker Management

### 6-Stage D2D Communication Flow

#### Read Request Flow (GDMA@Die0 → DDR@Die1)

**Stage 1: GDMA → D2D_SN (Die0 Internal)**

- **Flow**: GDMA(14.g0) → D2D_SN(33.ds)
- **D2D_SN Actions**:
  - Check `sn_tracker_count["ro"]` availability
  - If sufficient: Allocate tracker, forward to Stage 2
  - If insufficient: Send `negative` response, add to `sn_req_wait` queue
- **Tracker State**: D2D_SN allocates 1 RO tracker

**Stage 2: D2D_SN → D2D_RN (Die0→Die1 AXI AR)**

- **Flow**: D2D_SN(37.ds@Die0) → D2D_RN(4.dr@Die1) via AXI AR channel
- **AXI Channel**: Address Read (AR) with configurable latency
- **Tracker State**: D2D_SN tracker remains allocated

**Stage 3: D2D_RN → DDR (Die1 Internal)**

- **Flow**: D2D_RN(4.dr) → DDR(4.dr)
- **D2D_RN Actions**:
  - Check `rn_tracker_count["read"]` and `rn_rdb_count` availability
  - If sufficient: Allocate tracker and RDB space
  - If insufficient: **Should implement backpressure** (currently drops request)
- **Tracker State**: D2D_RN allocates 1 read tracker + burst_length RDB

**Stage 4: DDR → D2D_RN (Die1 Internal)**

- **Flow**: DDR(4.dr) → D2D_RN(4.dr)
- **Data Processing**: D2D_RN collects complete burst of data flits
- **Tracker State**: D2D_RN tracker remains allocated until Stage 5 begins

**Stage 5: D2D_RN → D2D_SN (Die1→Die0 AXI R)**

- **Flow**: D2D_RN(4.dr@Die1) → D2D_SN(37.ds@Die0) via AXI R channel
- **AXI Channel**: Read Data (R) with configurable latency
- **D2D_RN Actions**: Release tracker immediately after sending to AXI
- **Tracker State**: D2D_RN releases 1 read tracker + burst_length RDB

**Stage 6: D2D_SN → GDMA (Die0 Internal)**

- **Flow**: D2D_SN(37.ds) → GDMA(14.g0)
- **D2D_SN Actions**: Forward data to original requester
- **Tracker State**: D2D_SN releases 1 RO tracker after forwarding

#### Write Request Flow (GDMA@Die0 → DDR@Die1)

**Stage 1: GDMA → D2D_SN (Die0 Internal)**

- **Flow**: GDMA(14.g0) → D2D_SN(33.ds)
- **D2D_SN Actions**:
  - Check `sn_tracker_count["share"]` and `sn_wdb_count` availability
  - If sufficient: Allocate tracker+WDB, send `datasend` response
  - If insufficient: Send `negative` response, add to `sn_req_wait` queue
- **Tracker State**: D2D_SN allocates 1 share tracker + burst_length WDB

**Stage 2: D2D_SN → D2D_RN (Die0→Die1 AXI AW+W)**

- **Flow**: D2D_SN(37.ds@Die0) → D2D_RN(4.dr@Die1) via AXI AW+W channels
- **AXI Channels**: Address Write (AW) + Write Data (W) with configurable latencies
- **Tracker State**: D2D_SN tracker remains allocated

**Stage 3: D2D_RN → DDR (Die1 Internal)**

- **Flow**: D2D_RN(4.dr) → DDR(4.dr)
- **D2D_RN Actions**:
  - Check `rn_tracker_count["write"]` and `rn_wdb_count` availability
  - Allocate tracker and WDB space
- **Tracker State**: D2D_RN allocates 1 write tracker + burst_length WDB

**Stage 4: DDR → D2D_RN (Die1 Internal)**

- **Flow**: DDR(4.dr) → D2D_RN(4.dr)
- **Response**: Write complete response
- **D2D_RN Actions**: Release tracker after receiving write complete
- **Tracker State**: D2D_RN releases 1 write tracker + burst_length WDB

**Stage 5: D2D_RN → D2D_SN (Die1→Die0 AXI B)**

- **Flow**: D2D_RN(4.dr@Die1) → D2D_SN(37.ds@Die0) via AXI B channel
- **AXI Channel**: Write Response (B) with configurable latency
- **Tracker State**: D2D_RN tracker already released

**Stage 6: D2D_SN → GDMA (Die0 Internal)**

- **Flow**: D2D_SN(37.ds) → GDMA(14.g0)
- **Response**: Write complete response to original requester
- **Tracker State**: D2D_SN releases 1 share tracker + burst_length WDB

### Retry Mechanism

#### Resource Shortage Handling

**D2D_SN Resource Shortage**:

1. **Immediate Response**: Send `negative` response to requester (GDMA)
2. **Queue Management**: Add request to `sn_req_wait[req_type]` queue
3. **GDMA Behavior**: Waits for `positive` response before retry

**Tracker Release and Retry Notification**:

1. **Resource Release**: When D2D_SN completes transaction, call `release_completed_sn_tracker()`
2. **Queue Processing**: Check `sn_req_wait` for waiting requests
3. **Retry Activation**:
   - **Write Requests**: Send `positive` response to trigger GDMA retry
   - **Read Requests**: Process directly or send `positive` (implementation dependent)

#### GDMA Retry Behavior

**On `negative` Response**:

- Mark request as `req_attr = "old"` and `req_state = "invalid"`
- Wait for `positive` response (no automatic retry)

**On `positive` Response**:

- Reactivate request: `req_state = "valid"`, `req_attr = "old"`
- Re-inject into network: `enqueue(req, "req", retry=True)`

### Critical Design Constraints

#### AXI Protocol Limitations

- **No Built-in Retry**: AXI channels cannot reject requests once accepted
- **Committed Transmission**: All AXI transactions must complete successfully
- **Resource Pre-check**: Must verify resources before entering AXI layer

#### Resource Management Strategy

- **D2D_SN Gate-keeping**: Primary resource control at source Die
- **D2D_RN Guaranteed Processing**: Should not fail if D2D_SN managed resources correctly
- **Early Resource Allocation**: Allocate at Stage 1, release at Stage 6

#### Current Implementation Issues

1. **D2D_SN Read Requests**: Currently bypasses resource check (needs fixing)
2. **D2D_RN Resource Drops**: Drops requests when resources unavailable (AXI violation)
3. **Missing Retry Flow**: Incomplete positive response mechanism for queued requests

## Important Notes

- **Testing**: No established testing framework. `scripts/test.py` exists but no pytest/unittest setup
- **Linting**: No linting configuration. Consider using standard Python tools (flake8, black)
- **Environment**: Uses `.env` file for environment variables
- **Active Development**: Check git status for uncommitted changes in `scripts/`
- **Results Location**: Results saved to `../Result/` (parent directory of project root)
- **Traffic Processing**: Comprehensive traffic file parsing in `src/traffic_process/`
- **Visualization**: Multiple visualization tools in `scripts/tools/`
- 所有的输出都使用中文
- 代码实现的时候不需要考虑兼容性
- 代码实现时，如果需要读取变量值，不要使用默认值，如果没有找到变量就直接报错
- 所有的输出都使用中文
- 测试文件都放到test文件夹中
- 不需要总是赞同我的想法，需要批判性的思考问题。
- 每次修改都要尽量少修改已有的代码，尽量使用已有的函数实现功能，不要重新实现方法。
