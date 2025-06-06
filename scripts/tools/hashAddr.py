import math
import os
import re
from pathlib import Path


def systemAddrMap(addr):
    SHARE32_BASE = 0x0400000000  # 16GB
    SHARE8_BASE = 0x0800000000  # 16GB
    PRIVATE_BASE = 0x1000000000  # 32GB

    region_bits = addr >> 35

    # 32-share region
    if region_bits == 0x0:
        wdc_id = hashAddrMap(addr, 32)
        return wdc_id

    # 8-share region
    if region_bits == 0x1:
        cluster_id = (addr - SHARE8_BASE) >> 32
        wdc_cluster_id = hashAddrMap(addr, 8)
        wdc_id = cluster_id * 8 + wdc_cluster_id
        return wdc_id

    # private region
    if region_bits >= 0x2:
        wdc_id = (addr - PRIVATE_BASE) >> 30
        return wdc_id

    return -1  # invalid addr


def hashAddrMap(addr, num_nodes, intlvSize=2048):
    paddr = addr >> int(math.log2(intlvSize))
    bits = int(math.log2(num_nodes))
    sel = 0

    idx = 0
    while paddr:
        sel ^= (paddr & 1) << (idx % bits)
        paddr >>= 1
        idx += 1

    mask = (1 << bits) - 1
    return sel & mask if sel & mask else sel


def map_node_to_coordinates(node_id):
    if not 0 <= node_id <= 31:
        raise ValueError("Node ID must be between 0 and 31")

    # Determine if we're in the upper half (y=4,3) or lower half (y=2,1)
    half = node_id // 16

    # Within each half, we have groups of 8 nodes
    group = (node_id % 16) // 8

    # Within each group of 8, we have subgroups of 4
    subgroup = (node_id % 8) // 4

    # Partition (0 or 1)
    p = subgroup

    # Position within subgroup (0-3)
    pos_in_subgroup = node_id % 4

    # x coordinate alternates between 0 and 1 in each subgroup of 4
    x = pos_in_subgroup % 2

    # y coordinate depends on position in subgroup and half
    if pos_in_subgroup < 2:
        y = 4 - (2 * half)  # First 2 in subgroup: y=4 or y=2
    else:
        y = 3 - (2 * half)  # Last 2 in subgroup: y=3 or y=1

    # For each complete group of 8, we add 2 to the x coordinate
    x += 2 * group

    return p, x, y


def process_file(input_path, output_dir, tpu_id):
    """Process a single TPU trace file"""
    # Get coordinates for this TPU (based on its ID)
    p, x, y = map_node_to_coordinates(tpu_id)
    output_filename = f"master_p{p}_x{x}_y{y}.txt"
    output_path = os.path.join(output_dir, output_filename)

    try:
        with open(input_path, "r") as infile, open(output_path, "w") as outfile:
            for line in infile:
                parts = line.strip().split(",")
                if len(parts) != 5:
                    continue

                cycle, addr, rw, _, size = parts
                try:
                    addr_int = int(addr, 16)
                except ValueError:
                    continue  # Skip invalid address lines

                # Calculate node ID (0-31)
                node_id = hashAddrMap(addr_int, 32, 2048)

                # Get p,x,y coordinates for the node
                p_node, x_node, y_node = map_node_to_coordinates(node_id)

                # Rewrite line
                new_line = f"{cycle},(p{p_node},x{x_node},y{y_node}),{rw},{size}\n"
                outfile.write(new_line)

        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def process_subfolder(subfolder_path, root_output_dir):
    """Process all TPU files in a single subfolder"""
    print(f"\nProcessing subfolder: {subfolder_path.name}")

    # Find all TPU trace files in this subfolder
    tpu_files = []

    # Look for files in 'output' subdirectory first
    output_dir = subfolder_path / "output"
    if output_dir.exists():
        search_dir = output_dir
    else:
        search_dir = subfolder_path

    # Find all matching TPU files
    for pattern in ["gmemTrace.TPU*.tdma_instance.txt", "TPU*.txt", "*TPU*.txt"]:
        for f in search_dir.glob(pattern):
            # Extract TPU ID using regex
            match = re.search(r"TPU(\d+)", f.name)
            if match:
                tpu_id = int(match.group(1))
                if 0 <= tpu_id <= 31:
                    tpu_files.append((f, tpu_id))

    if not tpu_files:
        print(f"  No valid TPU files found in {subfolder_path}")
        return 0

    # Create output directory for this subfolder
    output_dir = root_output_dir / subfolder_path.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all TPU files in this subfolder
    processed_count = 0
    for tpu_file, tpu_id in sorted(tpu_files):
        if process_file(tpu_file, output_dir, tpu_id):
            p, x, y = map_node_to_coordinates(tpu_id)
            print(f"  Processed TPU{tpu_id} -> master_p{p}_x{x}_y{y}.txt")
            processed_count += 1
        else:
            print(f"  Failed to process TPU{tpu_id}")

    print(f"  Completed: {processed_count}/{len(tpu_files)} files processed")
    return processed_count


def has_subfolders(directory):
    """Check if directory contains subfolders"""
    path = Path(directory)
    if not path.exists():
        return False

    for item in path.iterdir():
        if item.is_dir():
            return True
    return False


def process_directory(root_input_dir, root_output_dir):
    """Main processing function with automatic subfolder detection"""
    root_input_dir = Path(root_input_dir)
    root_output_dir = Path(root_output_dir)

    if not root_input_dir.exists():
        print(f"Input directory {root_input_dir} does not exist!")
        return

    print(f"Input directory: {root_input_dir}")
    print(f"Output directory: {root_output_dir}")

    # Create output root directory
    root_output_dir.mkdir(parents=True, exist_ok=True)

    # Check if input directory has subfolders
    if has_subfolders(root_input_dir):
        print(f"\nDetected subfolders in {root_input_dir}")
        print("Processing each subfolder separately...")

        # Get all subdirectories
        subfolders = [d for d in root_input_dir.iterdir() if d.is_dir()]

        if not subfolders:
            print("No subfolders found!")
            return

        total_processed = 0
        for subfolder in sorted(subfolders):
            processed = process_subfolder(subfolder, root_output_dir)
            total_processed += processed

        print(f"\n=== Summary ===")
        print(f"Total subfolders processed: {len(subfolders)}")
        print(f"Total files processed: {total_processed}")

    else:
        print(f"\nNo subfolders detected in {root_input_dir}")
        print("Processing files directly in root directory...")

        # Process files directly in root directory
        processed = process_subfolder(root_input_dir, root_output_dir.parent)
        print(f"\n=== Summary ===")
        print(f"Total files processed: {processed}")


if __name__ == "__main__":
    # Set your input and output root directories
    input_root = "DeepSeek_origin"
    output_root = "DeepSeek_remap"

    print("=== Chip2Chip Topology Mapping Tool ===")

    # Process all matching files in the directory structure
    process_directory(input_root, output_root)

    print("\nProcessing completed!")

if __name__ == "__main__":
    # Set your input and output root directories
    input_root = r"../../traffic/original/DeepSeek"
    output_root = r"../../traffic/DeepSeek_xy"

    # Process all matching files in the directory structure
    process_directory(input_root, output_root)
