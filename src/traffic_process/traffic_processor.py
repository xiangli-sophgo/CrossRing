#!/usr/bin/env python3
"""
优化的流量数据处理器 - traffic_processor.py

主要改进：
1. 使用生成器和流式处理减少内存占用
2. 并行处理多个子文件夹
3. 配置化文件模式和输出格式
4. 增强错误处理和日志
5. 直接处理避免中间复制
"""

import os
import re
import logging
import time
import math
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Generator, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import multiprocessing as mp


@dataclass
class ProcessingConfig:
    """处理配置类"""

    file_patterns: List[str] = None
    output_format: str = "txt"
    time_column: int = 0
    max_workers: int = None
    chunk_size: int = 10000
    # 处理步骤控制 - 使用列表格式 [合并, 统计, hash]
    process_steps: List[int] = None

    def __post_init__(self):
        if self.file_patterns is None:
            self.file_patterns = ["gmemTrace01.tpu_*.gdma_instance.csv", "*tdma_instance.csv"]
        if self.max_workers is None:
            self.max_workers = min(8, mp.cpu_count())
        if self.process_steps is None:
            self.process_steps = [1, 1, 1]  # 默认执行所有步骤

        # 验证 process_steps 参数
        if len(self.process_steps) != 3:
            raise ValueError("process_steps 必须是长度为3的列表，格式为 [合并, 统计, hash]")
        if not all(step in [0, 1] for step in self.process_steps):
            raise ValueError("process_steps 中的值必须为0或1")

    @property
    def enable_merge(self) -> bool:
        """是否启用数据合并"""
        return bool(self.process_steps[0])

    @property
    def enable_statistics(self) -> bool:
        """是否启用统计分析"""
        return bool(self.process_steps[1])

    @property
    def enable_hash(self) -> bool:
        """是否启用hash处理"""
        return bool(self.process_steps[2])


class AddressHasher:
    """地址哈希处理类 - 基于文档的XOR hash算法"""

    def __init__(self, interleave_size=256):
        """
        初始化地址hash器

        Args:
            interleave_size: 交织大小（位）
            - 256: 256位 = 32字节交织
            - 512: 512位 = 64字节交织
        """
        self.interleave_size = interleave_size

        # 计算交织偏移
        if interleave_size == 256:
            self.interleave_offset = 0
        elif interleave_size == 512:
            self.interleave_offset = 1
        else:
            # 其他大小（如2048）
            self.interleave_offset = 3

    def hash_all(self, addr, base=16):
        """对地址进行哈希并返回(node_id, ip_type)"""
        if isinstance(addr, str):
            addr = int(addr, base=base)

        # 根据地址范围选择hash方法
        if 0x80000000 <= addr <= 0xFFFFFFFF:
            return self.shared_32ch(addr)
        elif 0x100000000 <= addr <= 0x2FFFFFFFF:
            return self.shared_16ch(addr, cluster_id=0)
        elif 0x300000000 <= addr <= 0x4FFFFFFFF:
            return self.shared_16ch(addr, cluster_id=1)
        elif 0x500000000 <= addr <= 0x57FFFFFFF:
            return self.shared_8ch(addr, cluster_id=0)
        elif 0x580000000 <= addr <= 0x5FFFFFFFF:
            return self.shared_8ch(addr, cluster_id=1)
        elif 0x600000000 <= addr <= 0x67FFFFFFF:
            return self.shared_8ch(addr, cluster_id=2)
        elif 0x680000000 <= addr <= 0x6FFFFFFFF:
            return self.shared_8ch(addr, cluster_id=3)
        elif 0x700000000 <= addr < 0x1F00000000:
            return self.private(addr)
        else:
            raise ValueError(f"Address error: {hex(addr)}")

    def get_address_category(self, addr, base=16):
        """获取地址类别"""
        addr = int(addr, base=base)
        if 0x80000000 <= addr <= 0xFFFFFFFF:
            return "32CH_shared"
        elif (0x100000000 <= addr <= 0x2FFFFFFFF) or (0x300000000 <= addr <= 0x4FFFFFFFF):
            return "16CH_shared"
        elif (0x500000000 <= addr <= 0x57FFFFFFF) or (0x580000000 <= addr <= 0x5FFFFFFFF) or (0x600000000 <= addr <= 0x67FFFFFFF) or (0x680000000 <= addr <= 0x6FFFFFFFF):
            return "8CH_shared"
        elif 0x700000000 <= addr < 0x1F00000000:
            return "private"
        else:
            return "unknown"

    def _xor_hash(self, addr, num_bits):
        """通用XOR hash函数，从文档算法扩展"""
        select = [0] * num_bits

        # 从PA[6]开始，按num_bits间隔进行XOR
        for i in range(num_bits):
            bit_pos = 6 + i
            while bit_pos < 48:
                if addr & (1 << bit_pos):
                    select[i] ^= 1
                bit_pos += num_bits

        # 组合选择位
        result = 0
        for i in range(num_bits):
            result |= select[i] << i

        return result

    def _get_interleave_bit(self, memory_type):
        """根据内存类型和交织大小确定交织位"""
        if memory_type == "32CH":
            # 32CH使用与16CH类似的交织位
            base_bit = 11
        elif memory_type == "16CH":
            # 16CH: 256b用PA[11], 512b用PA[12]
            base_bit = 11
        elif memory_type == "8CH":
            # 8CH: 256b用PA[10], 512b用PA[11]
            base_bit = 10
        elif memory_type == "private":
            # Private可能使用固定的位
            base_bit = 10
        else:
            base_bit = 10

        # 加上交织偏移
        return base_bit + self.interleave_offset

    def _get_ip_in_node(self, addr, memory_type):
        """根据交织位选择节点内的IP (0或1)"""
        interleave_bit = self._get_interleave_bit(memory_type)
        return (addr >> interleave_bit) & 1

    def shared_32ch(self, addr):
        """处理 32通道共享内存"""
        # 5位hash选择32个逻辑通道中的一个
        channel_index = self._xor_hash(addr, 5)

        # 使用交织位选择节点内的DDR
        ddr_in_node = self._get_ip_in_node(addr, "32CH")

        # 32个逻辑通道映射到16个物理节点
        # 每个节点处理2个逻辑通道
        physical_node = channel_index % 16

        return physical_node, f"ddr_{ddr_in_node}"

    def shared_16ch(self, addr, cluster_id):
        """处理 16通道共享内存"""
        # 4位hash选择16个逻辑通道中的一个
        channel_index = self._xor_hash(addr, 4)

        # 使用交织位选择节点内的DDR
        ddr_in_node = self._get_ip_in_node(addr, "16CH")

        # 16个通道分布在8个节点，每节点2个通道
        if cluster_id == 0:
            # 前16个通道，使用节点0-7
            physical_node = channel_index // 2
        else:
            # 后16个通道，使用节点8-15
            physical_node = 8 + (channel_index // 2)

        return physical_node, f"ddr_{ddr_in_node}"

    def shared_8ch(self, addr, cluster_id):
        """处理 8通道共享内存"""
        # 3位hash选择8个逻辑通道中的一个
        channel_index = self._xor_hash(addr, 3)

        # 使用交织位选择节点内的DDR
        ddr_in_node = self._get_ip_in_node(addr, "8CH")

        # 8个通道分布在4个节点，每节点2个通道
        nodes_per_cluster = 4
        node_offset = channel_index // 2
        physical_node = (cluster_id * nodes_per_cluster) + node_offset

        return physical_node, f"ddr_{ddr_in_node}"

    def private(self, addr):
        """处理私有内存"""
        # 计算私有区域编号
        private_size = 0xC0000000  # 3GB一个区域
        region_id = (addr - 0x700000000) // private_size

        # 直接映射到节点
        physical_node = region_id % 16

        # 私有内存使用交织位选择DDR
        ddr_in_node = self._get_ip_in_node(addr, "private")

        return physical_node, f"ddr_{ddr_in_node}"

    # 保留原有的hash_addr2node方法作为兼容，但不再使用
    def hash_addr2node(self, addr, node_num):
        """旧的地址到节点的哈希函数（保留兼容）"""
        # 使用新的XOR hash算法
        sel_bits = int(math.log2(node_num))
        return self._xor_hash(addr, sel_bits)

    def ip2node(self, n):
        """IP编号到节点的映射"""
        n = int(n)
        group = n // 8
        subgroup = (n % 8) // 4
        parity = n % 2

        base_map = [[0, 16], [2, 18], [8, 24], [10, 26]]
        base = base_map[group][subgroup]
        offset = 4 * ((n % 4) // 2)
        return str(base + offset + parity)


class AddressAnalyzer:
    """地址分析类"""

    def __init__(self):
        self.hasher = AddressHasher()

    def analyze_folder_data(self, folder_data: Dict[str, List[List[str]]]) -> Dict[str, Dict]:
        """分析所有文件夹的地址统计"""
        analysis_results = {}

        for folder_name, data in folder_data.items():
            analysis_results[folder_name] = self._analyze_single_folder(folder_name, data)

        return analysis_results

    def _analyze_single_folder(self, folder_name: str, data: List[List[str]]) -> Dict:
        """分析单个文件夹的数据"""
        stats = {
            "32CH_shared_flit": 0,
            "16CH_shared_flit": 0,
            "8CH_shared_flit": 0,
            "private_flit": 0,
            "32CH_shared_req": 0,
            "16CH_shared_req": 0,
            "8CH_shared_req": 0,
            "private_req": 0,
            "read_flit": 0,
            "write_flit": 0,
            "read_req": 0,
            "write_req": 0,
            "total_req": len(data),
            "total_flit": 0,
            "max_time": 0,
        }

        for row in data:
            if len(row) >= 7:
                addr = row[3]  # 目标地址列
                time_val = int(row[0]) if row[0].isdigit() else 0
                burst_len = int(row[6]) if row[6].isdigit() else 1  # burst长度列
                req_type = row[5]  # 请求类型 (r/w)

                # 累计总flit数
                stats["total_flit"] += burst_len

                # 统计地址类别的flit数量和请求数量
                category = self.hasher.get_address_category(addr)
                if category in ["32CH_shared", "16CH_shared", "8CH_shared", "private"]:
                    stats[f"{category}_flit"] += burst_len
                    stats[f"{category}_req"] += 1

                # 统计读写类型的flit数量和请求数量
                if req_type.lower() == "r":
                    stats["read_flit"] += burst_len
                    stats["read_req"] += 1
                elif req_type.lower() == "w":
                    stats["write_flit"] += burst_len
                    stats["write_req"] += 1

                # 更新最大时间
                stats["max_time"] = max(stats["max_time"], time_val)
            else:
                raise ValueError

        # 计算基于flit数量的地址类别百分比
        if stats["total_flit"] > 0:
            for key in ["32CH_shared", "16CH_shared", "8CH_shared", "private"]:
                stats[f"{key}_flit_percent"] = stats[f"{key}_flit"] / stats["total_flit"] * 100

            # 计算读写比例
            stats["read_flit_percent"] = stats["read_flit"] / stats["total_flit"] * 100
            stats["write_flit_percent"] = stats["write_flit"] / stats["total_flit"] * 100

        return stats

    def generate_csv_report(self, analysis_results: Dict[str, Dict], output_path: Path):
        """生成CSV统计报告"""
        stats_dir = output_path / "statistics"
        stats_dir.mkdir(parents=True, exist_ok=True)

        # 生成详细统计报告
        csv_file = stats_dir / "traffic_statistics.csv"

        with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)

            # 写入表头
            headers = [
                "文件夹名称",
                "总请求数",
                "总flit数",
                "最大时间",
                "32CH_shared flit数",
                "32CH_shared flit百分比",
                "16CH_shared flit数",
                "16CH_shared flit百分比",
                "8CH_shared flit数",
                "8CH_shared flit百分比",
                "private flit数",
                "private flit百分比",
                "读 flit数",
                "读 flit百分比",
                "写 flit数",
                "写 flit百分比",
            ]
            writer.writerow(headers)

            # 写入数据
            for folder_name, stats in analysis_results.items():
                row = [
                    folder_name,
                    stats["total_req"],
                    stats["total_flit"],
                    stats["max_time"],
                    stats["32CH_shared_flit"],
                    f"{stats.get('32CH_shared_flit_percent', 0):.2f}%",
                    stats["16CH_shared_flit"],
                    f"{stats.get('16CH_shared_flit_percent', 0):.2f}%",
                    stats["8CH_shared_flit"],
                    f"{stats.get('8CH_shared_flit_percent', 0):.2f}%",
                    stats["private_flit"],
                    f"{stats.get('private_flit_percent', 0):.2f}%",
                    stats["read_flit"],
                    f"{stats.get('read_flit_percent', 0):.2f}%",
                    stats["write_flit"],
                    f"{stats.get('write_flit_percent', 0):.2f}%",
                ]
                writer.writerow(row)

        return csv_file

    def process_and_hash_data(self, folder_data: Dict[str, List[List[str]]], output_path: Path):
        """处理数据并生成hash后的文件"""
        hashed_dir = output_path / "hashed"
        hashed_dir.mkdir(parents=True, exist_ok=True)

        for folder_name, data in folder_data.items():
            hashed_file = hashed_dir / f"{folder_name}.txt"

            with open(hashed_file, "w", encoding="utf-8") as f:
                for row in data:
                    if len(row) >= 7:
                        # 获取原始数据
                        time_val = row[0]
                        tpu_num = row[1]
                        src_type = row[2]  # "gdma"
                        addr = row[3]
                        dst_type = row[4]  # "ddr"
                        req_type = row[5]  # "r"/"w"
                        burst_len = row[6]

                        # 进行地址hash得到目标节点和DDR
                        try:
                            target_node, target_ip = self.hasher.hash_all(addr)

                            # 源节点和源IP：TPU编号映射到节点和GDMA
                            tpu_id = int(tpu_num)
                            src_node = tpu_id // 2  # 源节点编号 (每节点2个GDMA)
                            src_ip = f"gdma_{tpu_id % 2}"  # "gdma_0"或"gdma_1"

                            # 写入hash后的数据: 时间,源节点,源IP,目标节点,目标IP,请求类型,burst长度
                            hashed_row = [time_val, str(src_node), src_ip, str(target_node), target_ip, req_type, burst_len]  # 源节点编号  # "gdma_0" 或 "gdma_1"  # 目标节点编号  # "ddr_0" 或 "ddr_1"
                            f.write(",".join(hashed_row) + "\n")

                        except (ValueError, AssertionError) as e:
                            # 跳过无效地址
                            continue


class FileScanner:
    """文件扫描和发现模块"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def find_target_files(self, root_path: Path) -> Dict[str, List[Path]]:
        """
        扫描目录，按子文件夹分组收集符合条件的文件

        Returns:
            Dict[子文件夹名, 文件路径列表]
        """
        files_by_folder = defaultdict(list)

        self.logger.info(f"开始扫描目录: {root_path}")

        for subfolder in root_path.iterdir():
            if not subfolder.is_dir():
                continue

            folder_name = subfolder.name
            self.logger.debug(f"处理子文件夹: {folder_name}")

            # 递归查找符合条件的文件
            target_files = self._find_files_recursive(subfolder)
            files_by_folder[folder_name].extend(target_files)

            self.logger.debug(f"在 {folder_name} 中找到 {len(target_files)} 个文件")

        total_files = sum(len(files) for files in files_by_folder.values())
        self.logger.info(f"扫描完成，共找到 {len(files_by_folder)} 个子文件夹，{total_files} 个文件")

        return dict(files_by_folder)

    def _find_files_recursive(self, directory: Path) -> List[Path]:
        """递归查找符合模式的文件"""
        target_files = []

        for pattern in self.config.file_patterns:
            # 使用 rglob 递归匹配文件
            matches = list(directory.rglob(pattern))
            target_files.extend(matches)

        # 去重
        return list(set(target_files))


class DataMerger:
    """数据合并和排序模块"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def process_files_for_folder(self, folder_name: str, file_paths: List[Path]) -> List[List[str]]:
        """处理单个文件夹中的所有文件"""
        self.logger.info(f"开始处理文件夹: {folder_name} ({len(file_paths)} 个文件)")

        all_data = []

        for file_path in file_paths:
            try:
                data = list(self._process_single_file(file_path))
                all_data.extend(data)
                self.logger.debug(f"处理文件 {file_path.name}: {len(data)} 条记录")
            except Exception as e:
                self.logger.error(f"处理文件失败 {file_path}: {e}")
                continue

        if all_data:
            # 先按时间排序，然后按TPU编号排序
            all_data.sort(key=lambda x: (int(x[0]) if x[0].isdigit() else 0, int(x[1]) if x[1].isdigit() else 0))  # 时间列（第一列）  # TPU编号列（第二列）
            self.logger.info(f"文件夹 {folder_name} 处理完成: {len(all_data)} 条记录")

        return all_data

    def _process_single_file(self, file_path: Path) -> Generator[List[str], None, None]:
        """流式处理单个文件"""
        if file_path.stat().st_size == 0:
            self.logger.warning(f"跳过空文件: {file_path}")
            return

        # 从文件名提取TPU编号
        tpu_match = re.search(r"tpu_(\d+)", file_path.name)
        if not tpu_match:
            self.logger.warning(f"无法从文件名提取TPU编号: {file_path.name}")
            return

        tpu_num = tpu_match.group(1)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                # 跳过标题行
                first_line = f.readline().strip()
                if first_line.startswith("begin_req"):
                    pass  # 标题行已跳过
                else:
                    # 如果第一行不是标题，重新定位到文件开头
                    f.seek(0)

                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    data = line.split(",")
                    if len(data) < 8:
                        self.logger.debug(f"跳过无效行 {file_path.name}:{line_num}")
                        continue

                    # 提取所需字段：时间、源节点、源IP、目标地址、目标IP、请求类型、burst
                    processed_data = [
                        data[1],  # 时间 (end_req)
                        tpu_num,  # 源节点 (TPU编号)
                        "gdma",  # 源IP类型 (固定为gdma)
                        data[4],  # 目标地址 (addr，保留原始地址)
                        "ddr",  # 目标IP类型 (固定为ddr)
                        data[5],  # 请求类型 (r/w)
                        data[7],  # burst长度 (burst_len)
                    ]

                    yield processed_data

        except Exception as e:
            self.logger.error(f"读取文件出错 {file_path}: {e}")
            raise


class DataStreamProcessor:
    """主数据流处理类"""

    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.scanner = FileScanner(self.config)
        self.merger = DataMerger(self.config)
        self._setup_logging()

    def _setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
        self.logger = logging.getLogger(__name__)

    def process_directory(self, input_path: str, output_path: str = None) -> Dict[str, any]:
        """
        处理整个目录

        Args:
            input_path: 输入目录路径
            output_path: 输出目录路径

        Returns:
            处理统计信息
        """
        start_time = time.time()
        input_path = Path(input_path)

        if output_path is None:
            output_path = Path("../output/step1_flatten_optimized")
        else:
            output_path = Path(output_path)

        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"开始处理目录: {input_path}")
        self.logger.info(f"输出目录: {output_path}")

        # 显示处理步骤配置
        step_names = ["数据合并", "统计分析", "Hash处理"]
        enabled_steps = [step_names[i] for i, enabled in enumerate(self.config.process_steps) if enabled]
        self.logger.info(f"执行步骤: {enabled_steps}")

        # 根据配置决定处理步骤
        stats = {"folders_processed": 0, "total_records": 0, "files_processed": 0, "errors": [], "all_folder_data": {}}

        # 步骤1: 数据合并（如果启用）
        if self.config.enable_merge:
            self.logger.info("执行步骤1: 数据合并...")
            files_by_folder = self.scanner.find_target_files(input_path)

            if not files_by_folder:
                self.logger.warning("没有找到任何文件")
                return {"status": "no_files", "processing_time": 0}

            stats = self._process_folders_parallel(files_by_folder, output_path)
        else:
            self.logger.info("跳过数据合并步骤")
            # 如果不合并，尝试从已有的merged文件夹读取数据用于后续处理
            merged_dir = output_path / "merged"
            if merged_dir.exists():
                stats["all_folder_data"] = self._load_existing_merged_data(merged_dir)
                self.logger.info(f"从现有合并数据中加载了 {len(stats['all_folder_data'])} 个文件夹的数据")

        # 步骤2: 统计分析（如果启用）
        if self.config.enable_statistics and stats["all_folder_data"]:
            self.logger.info("执行步骤2: 生成统计报告...")
            analyzer = AddressAnalyzer()
            analysis_results = analyzer.analyze_folder_data(stats["all_folder_data"])
            csv_file = analyzer.generate_csv_report(analysis_results, output_path)
            self.logger.info(f"统计报告已生成: {csv_file}")
        elif self.config.enable_statistics:
            self.logger.warning("无可用数据进行统计分析")

        # 步骤3: Hash处理（如果启用）
        if self.config.enable_hash and stats["all_folder_data"]:
            self.logger.info("执行步骤3: Hash处理...")
            analyzer = AddressAnalyzer()
            analyzer.process_and_hash_data(stats["all_folder_data"], output_path)
            self.logger.info("hash处理完成")
        elif self.config.enable_hash:
            self.logger.warning("无可用数据进行hash处理")

        processing_time = time.time() - start_time
        stats["processing_time"] = processing_time

        self.logger.info(f"处理完成，总耗时: {processing_time:.2f} 秒")
        self._print_summary(stats)

        return stats

    def _process_folders_parallel(self, files_by_folder: Dict[str, List[Path]], output_path: Path) -> Dict[str, any]:
        """并行处理多个文件夹"""
        stats = {"folders_processed": 0, "total_records": 0, "files_processed": 0, "errors": [], "all_folder_data": {}}

        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # 提交任务
            future_to_folder = {executor.submit(self._process_folder_task, folder_name, file_paths, output_path): folder_name for folder_name, file_paths in files_by_folder.items()}

            # 收集结果
            for future in as_completed(future_to_folder):
                folder_name = future_to_folder[future]
                try:
                    result = future.result()
                    stats["folders_processed"] += 1
                    stats["total_records"] += result["records"]
                    stats["files_processed"] += result["files"]

                    # 保存每个文件夹的数据用于后续分析
                    if result["data"]:
                        stats["all_folder_data"][result["folder_name"]] = result["data"]

                    self.logger.info(f"完成处理: {folder_name} - {result['records']} 条记录")

                except Exception as e:
                    error_msg = f"处理文件夹 {folder_name} 时出错: {e}"
                    self.logger.error(error_msg)
                    stats["errors"].append(error_msg)

        return stats

    def _load_existing_merged_data(self, merged_dir: Path) -> Dict[str, List[List[str]]]:
        """从已有的merged文件夹加载数据"""
        folder_data = {}

        for merged_file in merged_dir.glob("*.txt"):
            folder_name = merged_file.stem
            data = []

            with open(merged_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(line.split(","))

            if data:
                folder_data[folder_name] = data

        return folder_data

    def _process_folder_task(self, folder_name: str, file_paths: List[Path], output_path: Path) -> Dict[str, int]:
        """单个文件夹处理任务（用于并行执行）"""
        merger = DataMerger(self.config)

        all_data = merger.process_files_for_folder(folder_name, file_paths)

        if all_data:
            # 创建三层输出结构
            merged_dir = output_path / "merged"
            merged_dir.mkdir(parents=True, exist_ok=True)

            # 写入合并后的原始数据
            output_file = merged_dir / f"{folder_name}.{self.config.output_format}"

            with open(output_file, "w", encoding="utf-8") as f:
                for data_row in all_data:
                    f.write(",".join(data_row) + "\n")

        return {"records": len(all_data), "files": len(file_paths), "data": all_data, "folder_name": folder_name}

    def _print_summary(self, stats: Dict[str, any]):
        """打印处理摘要"""
        print("\n" + "=" * 60)
        print("处理摘要")
        print("=" * 60)
        print(f"处理的文件夹数量: {stats['folders_processed']}")
        print(f"处理的文件数量: {stats['files_processed']}")
        print(f"处理的记录总数: {stats['total_records']}")
        print(f"处理总时间: {stats['processing_time']:.2f} 秒")

        if stats["errors"]:
            print(f"错误数量: {len(stats['errors'])}")
            for error in stats["errors"]:
                print(f"  - {error}")

        print("=" * 60)


def main(input_path: str, output_path: str = None, max_workers: int = None, process_steps: List[int] = None):
    """
    主函数

    Args:
        input_path: 输入目录路径
        output_path: 输出目录路径
        max_workers: 并行工作进程数
        process_steps: 处理步骤控制列表 [合并, 统计, hash]，每个值为0或1
                      例如: [1,1,1] 执行所有步骤
                           [1,0,0] 只执行合并
                           [0,1,0] 只执行统计
                           [0,0,1] 只执行hash
    """

    # 创建配置
    config = ProcessingConfig()
    if max_workers:
        config.max_workers = max_workers
    if process_steps:
        config.process_steps = process_steps

    # 创建处理器并执行
    processor = DataStreamProcessor(config)
    stats = processor.process_directory(input_path, output_path)

    return stats


if __name__ == "__main__":
    # 输入输出路径配置
    input_path = r"c:\Users\xiang\Documents\code\CrossRing\traffic\original\TPS175-DeepSeek3-671B-A37B-S4K-O1-W8A8-B64-16share"
    output_path = r"c:\Users\xiang\Documents\code\CrossRing\traffic\DeepSeek_0922"

    # 处理步骤控制配置
    # 使用列表格式控制执行步骤: [合并, 统计, hash]
    # 每个位置的值: 1=执行, 0=跳过

    # 当前配置
    PROCESS_STEPS = [0, 0, 1]  # [合并, 统计, hash]

    step_names = ["数据合并", "统计分析", "Hash处理"]
    enabled_steps = [step_names[i] for i, enabled in enumerate(PROCESS_STEPS) if enabled]
    disabled_steps = [step_names[i] for i, enabled in enumerate(PROCESS_STEPS) if not enabled]

    print("数据流处理配置:")
    print(f"  执行步骤: {enabled_steps}")
    if disabled_steps:
        print(f"  跳过步骤: {disabled_steps}")
    print(f"  步骤配置: {PROCESS_STEPS}")
    print("-" * 50)

    # 直接运行处理
    main(input_path, output_path, max_workers=4, process_steps=PROCESS_STEPS)
