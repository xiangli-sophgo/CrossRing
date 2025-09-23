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
    """地址哈希处理类"""

    def __init__(self, itlv_size=2048):
        self.itlv_size = itlv_size
        self.itlv_digit = itlv_size.bit_length() - 1

    def hash_all(self, addr, base=16):
        """对地址进行哈希并返回节点号"""
        addr = int(addr, base=base)
        if 0x80000000 <= addr < 0x100000000:
            return self.shared_32ch(addr)
        elif 0x100000000 <= addr < 0x500000000:
            return self.shared_16ch(addr)
        elif 0x500000000 <= addr < 0x700000000:
            return self.shared_8ch(addr)
        elif 0x700000000 <= addr < 0x1F00000000:
            return self.private(addr)
        else:
            raise ValueError(f"Address error: {hex(addr)}")

    def get_address_category(self, addr, base=16):
        """获取地址类别"""
        addr = int(addr, base=base)
        if 0x80000000 <= addr < 0x100000000:
            return "32CH_shared"
        elif 0x100000000 <= addr < 0x500000000:
            return "16CH_shared"
        elif 0x500000000 <= addr < 0x700000000:
            return "8CH_shared"
        elif 0x700000000 <= addr < 0x1F00000000:
            return "private"
        else:
            return "unknown"

    def shared_32ch(self, addr):
        """处理 32通道共享内存"""
        assert 0x80000000 <= addr < 0x100000000
        res = self.hash_addr2node(addr, 32)
        assert res < 32
        return res

    def shared_16ch(self, addr):
        """处理 16通道共享内存"""
        assert 0x100000000 <= addr < 0x500000000
        shared_16ch_size = 0x200000000
        cluster_id = (addr - 0x100000000) // shared_16ch_size
        hash_value = self.hash_addr2node(addr, 16)
        return cluster_id * 16 + hash_value

    def shared_8ch(self, addr):
        """处理 8通道共享内存"""
        assert 0x500000000 <= addr < 0x700000000
        shared_8ch_size = 0x80000000
        cluster_id = (addr - 0x500000000) // shared_8ch_size
        hash_value = self.hash_addr2node(addr, 8)
        return cluster_id * 8 + hash_value

    def private(self, addr):
        """处理私有内存"""
        assert 0x700000000 <= addr < 0x1F00000000
        private_size = 0xC0000000
        region_id = (addr - 0x700000000) // private_size
        assert region_id < 32
        return region_id

    def hash_addr2node(self, addr, node_num):
        """地址到节点的哈希函数"""
        assert (node_num & (node_num - 1) == 0) and node_num != 0

        addr_valid = addr >> self.itlv_digit
        sel_bits = int(math.log2(node_num))

        sel_bit_vec = [[] for _ in range(sel_bits)]

        i = 0
        while addr_valid:
            bit = addr_valid % 2
            sel_bit_vec[i % sel_bits].append(bit)
            i += 1
            addr_valid >>= 1

        sel_bit = [0] * sel_bits
        for i in range(sel_bits):
            for bit in sel_bit_vec[i]:
                sel_bit[i] ^= bit

        sel = sum(sel_bit[i] << i for i in range(sel_bits))
        return sel

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
        stats = {"32CH_shared": 0, "16CH_shared": 0, "8CH_shared": 0, "private": 0, "total": len(data), "max_time": 0}

        for row in data:
            if len(row) >= 4:
                addr = row[3]  # 目标地址列
                time_val = int(row[0]) if row[0].isdigit() else 0

                # 统计地址类别
                category = self.hasher.get_address_category(addr)
                if category in stats:
                    stats[category] += 1

                # 更新最大时间
                stats["max_time"] = max(stats["max_time"], time_val)

        # 计算百分比
        if stats["total"] > 0:
            for key in ["32CH_shared", "16CH_shared", "8CH_shared", "private"]:
                stats[f"{key}_percent"] = stats[key] / stats["total"] * 100

        return stats

    def generate_csv_report(self, analysis_results: Dict[str, Dict], output_path: Path):
        """生成CSV统计报告"""
        stats_dir = output_path / "statistics"
        stats_dir.mkdir(parents=True, exist_ok=True)

        # 生成详细统计报告
        csv_file = stats_dir / "traffic_statistics.csv"

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # 写入表头
            headers = [
                "文件夹名称",
                "总记录数",
                "最大时间",
                "32CH_shared数量",
                "32CH_shared百分比",
                "16CH_shared数量",
                "16CH_shared百分比",
                "8CH_shared数量",
                "8CH_shared百分比",
                "private数量",
                "private百分比",
            ]
            writer.writerow(headers)

            # 写入数据
            for folder_name, stats in analysis_results.items():
                row = [
                    folder_name,
                    stats["total"],
                    stats["max_time"],
                    stats["32CH_shared"],
                    f"{stats.get('32CH_shared_percent', 0):.2f}%",
                    stats["16CH_shared"],
                    f"{stats.get('16CH_shared_percent', 0):.2f}%",
                    stats["8CH_shared"],
                    f"{stats.get('8CH_shared_percent', 0):.2f}%",
                    stats["private"],
                    f"{stats.get('private_percent', 0):.2f}%",
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
                        src_type = row[2]
                        addr = row[3]
                        dst_type = row[4]
                        req_type = row[5]
                        burst_len = row[6]

                        # 进行地址hash，不再使用IP映射
                        try:
                            hashed_node = self.hasher.hash_all(addr)
                            src_node = tpu_num                    # 直接使用TPU编号
                            dst_node = str(hashed_node)           # 直接使用hash结果

                            # 写入hash后的数据: 时间,源节点,源类型,目标节点,目标类型,请求类型,burst长度
                            hashed_row = [time_val, src_node, src_type, dst_node, dst_type, req_type, burst_len]
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
    PROCESS_STEPS = [1, 1, 1]  # [合并, 统计, hash]

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
