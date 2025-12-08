"""
仿真引擎封装 - 统一KCIN和DCIN模型的调用接口
"""

import asyncio
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
import time


class SimulationMode(str, Enum):
    """仿真模式"""
    KCIN = "kcin"
    DCIN = "dcin"


class SimulationStatus(str, Enum):
    """仿真状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SimulationResult:
    """仿真结果"""
    status: SimulationStatus
    message: str = ""
    results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_seconds: float = 0.0


class SimulationEngine:
    """
    统一仿真引擎 - 封装KCIN和DCIN模型
    支持异步执行和进度回调
    """

    def __init__(
        self,
        mode: SimulationMode,
        config_path: str,
        topology: str = "5x4",
        verbose: int = 1,
        config_overrides: Optional[Dict[str, Any]] = None,
        die_config_path: Optional[str] = None,
        die_config_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.mode = mode
        self.config_path = config_path
        self.topology = topology
        self.verbose = verbose
        self.config_overrides = config_overrides or {}
        # DCIN模式下的DIE拓扑配置
        self.die_config_path = die_config_path
        self.die_config_overrides = die_config_overrides or {}
        self.model = None
        self.config = None
        self._cancelled = False

    def _init_kcin_model(self):
        """初始化KCIN模型"""
        from config.config import CrossRingConfig
        from src.noc import REQ_RSP_model

        self.config = CrossRingConfig(self.config_path)

        # 应用配置覆盖
        for key, value in self.config_overrides.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        topo_type = self.config.TOPO_TYPE if self.config.TOPO_TYPE else self.topology

        self.model = REQ_RSP_model(
            model_type="REQ_RSP",
            config=self.config,
            topo_type=topo_type,
            verbose=self.verbose,
        )

    def _init_dcin_model(self):
        """初始化DCIN模型"""
        from config.d2d_config import D2DConfig
        from src.d2d.d2d_model import D2D_Model

        # 初始化D2D配置，支持单独指定DIE拓扑配置
        self.config = D2DConfig(
            d2d_config_file=self.config_path,
            die_config_file=self.die_config_path,
        )

        # 应用DCIN配置覆盖
        for key, value in self.config_overrides.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # 应用DIE拓扑配置覆盖
        if self.die_config_overrides and hasattr(self.config, 'die_config'):
            for key, value in self.die_config_overrides.items():
                if hasattr(self.config.die_config, key):
                    setattr(self.config.die_config, key, value)

        self.model = D2D_Model(
            config=self.config,
            model_type="REQ_RSP",
            verbose=self.verbose,
        )

    def setup(
        self,
        traffic_file_path: str,
        traffic_files: List[str],
        result_save_path: Optional[str] = None,
    ):
        """
        设置仿真参数

        Args:
            traffic_file_path: 流量文件目录
            traffic_files: 流量文件列表
            result_save_path: 结果保存路径
        """
        # 初始化模型
        if self.mode == SimulationMode.KCIN:
            self._init_kcin_model()
        else:
            self._init_dcin_model()

        # 设置流量
        traffic_chains = [traffic_files]
        self.model.setup_traffic_scheduler(
            traffic_file_path=traffic_file_path,
            traffic_chains=traffic_chains,
        )

        # 设置结果分析
        if result_save_path:
            if self.mode == SimulationMode.KCIN:
                self.model.setup_result_analysis(
                    plot_RN_BW_fig=0,
                    flow_graph_interactive=1,
                    fifo_utilization_heatmap=1,
                    result_save_path=result_save_path,
                    show_result_analysis=0,
                )
            else:
                self.model.setup_result_analysis(
                    flow_graph_interactive=1,
                    plot_rn_bw_fig=0,
                    fifo_utilization_heatmap=1,
                    show_result_analysis=0,
                    export_d2d_requests_csv=1,
                    export_ip_bandwidth_csv=1,
                )

    def run_sync(
        self,
        max_time: int = 6000,
        print_interval: int = 200,
    ) -> SimulationResult:
        """
        同步执行仿真

        Args:
            max_time: 最大仿真时间
            print_interval: 打印间隔

        Returns:
            SimulationResult
        """
        start_time = time.time()

        try:
            # 执行仿真
            self.model.run_simulation(
                max_time=max_time,
                print_interval=print_interval,
            )

            # 获取结果
            results = self._extract_results()

            duration = time.time() - start_time

            return SimulationResult(
                status=SimulationStatus.COMPLETED,
                message="仿真完成",
                results=results,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{str(e)}\n{traceback.format_exc()}"

            return SimulationResult(
                status=SimulationStatus.FAILED,
                message="仿真失败",
                error=error_msg,
                duration_seconds=duration,
            )

    async def run_async(
        self,
        max_time: int = 6000,
        print_interval: int = 200,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> SimulationResult:
        """
        异步执行仿真

        Args:
            max_time: 最大仿真时间
            print_interval: 打印间隔
            progress_callback: 进度回调函数 (current_time, max_time)

        Returns:
            SimulationResult
        """
        loop = asyncio.get_event_loop()

        # 使用线程池执行同步仿真
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(
                executor,
                lambda: self.run_sync(max_time, print_interval)
            )

        return result

    def _extract_results(self) -> Dict[str, Any]:
        """从模型中提取结果"""
        results = {}

        try:
            # 获取基本统计
            if hasattr(self.model, 'get_results'):
                model_results = self.model.get_results()
                results.update(model_results)

            # 获取延迟和带宽统计
            if hasattr(self.model, 'analyzer') and self.model.analyzer:
                analyzer = self.model.analyzer
                if hasattr(analyzer, 'get_summary'):
                    results['summary'] = analyzer.get_summary()

        except Exception as e:
            results['extraction_error'] = str(e)

        return results

    def cancel(self):
        """取消仿真"""
        self._cancelled = True
        # 注意: 当前模型实现不支持中断，这里只是标记

    def save_to_database(
        self,
        experiment_name: str,
        config_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """
        保存结果到数据库

        Args:
            experiment_name: 实验名称
            config_params: 额外的配置参数

        Returns:
            实验ID或None
        """
        try:
            if hasattr(self.model, 'save_to_database'):
                return self.model.save_to_database(experiment_name=experiment_name)
        except Exception as e:
            print(f"保存到数据库失败: {e}")

        return None


class BatchSimulationEngine:
    """
    批量仿真引擎 - 支持多个流量文件的并行仿真
    """

    def __init__(
        self,
        mode: SimulationMode,
        config_path: str,
        topology: str = "5x4",
        max_workers: int = 4,
    ):
        self.mode = mode
        self.config_path = config_path
        self.topology = topology
        self.max_workers = max_workers

    async def run_batch(
        self,
        traffic_file_path: str,
        traffic_files: List[str],
        max_time: int = 6000,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[SimulationResult]:
        """
        批量执行仿真

        Args:
            traffic_file_path: 流量文件目录
            traffic_files: 流量文件列表
            max_time: 每个仿真的最大时间
            progress_callback: 进度回调 (completed, total, current_file)

        Returns:
            SimulationResult列表
        """
        results = []
        total = len(traffic_files)

        for i, traffic_file in enumerate(traffic_files):
            if progress_callback:
                progress_callback(i, total, traffic_file)

            engine = SimulationEngine(
                mode=self.mode,
                config_path=self.config_path,
                topology=self.topology,
                verbose=0,  # 批量模式下减少输出
            )

            engine.setup(
                traffic_file_path=traffic_file_path,
                traffic_files=[traffic_file],
            )

            result = await engine.run_async(max_time=max_time)
            result.results['traffic_file'] = traffic_file
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total, traffic_file)

        return results
