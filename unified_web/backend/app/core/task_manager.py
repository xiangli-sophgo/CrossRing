"""
异步任务管理器 - 管理仿真任务的生命周期
支持任务历史持久化到JSON文件
"""

import asyncio
import uuid
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError, BrokenExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import logging

logger = logging.getLogger("unified_web.task_manager")

# 单个任务超时时间（秒）
TASK_TIMEOUT = 3600  # 1小时

# 全局最大并行 worker 数
MAX_GLOBAL_WORKERS = 4


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SimulationTask:
    """仿真任务"""
    task_id: str
    mode: str  # kcin or dcin
    topology: str
    config_path: str
    traffic_file_path: str
    traffic_files: List[str]
    max_time: int
    experiment_name: Optional[str]
    experiment_description: Optional[str]
    save_to_db: bool
    max_workers: Optional[int] = None  # 并行进程数，None表示使用CPU核心数
    config_overrides: Optional[Dict[str, Any]] = None  # 配置覆盖项
    # DCIN模式下的DIE拓扑配置
    die_config_path: Optional[str] = None
    die_config_overrides: Optional[Dict[str, Any]] = None
    # 参数遍历组合列表
    sweep_combinations: Optional[List[Dict[str, Any]]] = None

    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0  # 0-100
    current_file: str = ""
    message: str = ""
    error: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)
    sim_details: Optional[Dict[str, Any]] = None  # 结构化的仿真进度数据

    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # 运行时属性（不序列化）
    _current_engine: Any = field(default=None, repr=False)
    _executor: Any = field(default=None, repr=False)  # ProcessPoolExecutor 引用
    _futures: List[Any] = field(default_factory=list, repr=False)  # Future 列表


def _run_single_simulation(sim_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    在子进程中运行单个仿真（独立函数，用于多进程）

    Args:
        sim_params: 仿真参数字典
            - combination_overrides: 参数遍历时的配置覆盖项，会与 config_overrides 合并
            - combination_index: 参数组合索引（用于日志和结果标识）

    Returns:
        结果字典
    """
    from .simulation_engine import SimulationEngine, SimulationMode

    traffic_file = sim_params['traffic_file']
    combination_index = sim_params.get('combination_index')
    try:
        # 合并配置覆盖项: base config_overrides + combination_overrides
        config_overrides = sim_params.get('config_overrides') or {}
        combination_overrides = sim_params.get('combination_overrides') or {}
        merged_overrides = {**config_overrides, **combination_overrides}

        engine = SimulationEngine(
            mode=SimulationMode(sim_params['mode']),
            config_path=sim_params['config_path'],
            topology=sim_params['topology'],
            verbose=0,
            config_overrides=merged_overrides if merged_overrides else None,
            die_config_path=sim_params.get('die_config_path'),
            die_config_overrides=sim_params.get('die_config_overrides'),
        )

        engine.setup(
            traffic_file_path=sim_params['traffic_file_path'],
            traffic_files=[traffic_file],
            show_result_analysis=sim_params.get('show_result', False),
        )

        result = engine.run_sync(max_time=sim_params['max_time'])
        result.results['traffic_file'] = traffic_file

        # 保存到数据库（仅在成功时保存）
        experiment_id = None
        print(f"[TaskManager] save_to_db={sim_params.get('save_to_db')}, status={result.status.value}")
        if sim_params.get('save_to_db') and result.status.value == 'completed':
            exp_name = sim_params.get('experiment_name') or '仿真实验'
            print(f"[TaskManager] 准备保存到数据库: {exp_name}")
            experiment_id = engine.save_to_database(
                experiment_name=exp_name,
                description=sim_params.get('experiment_description')
            )
            print(f"[TaskManager] 保存结果: experiment_id={experiment_id}")

        return {
            'traffic_file': traffic_file,
            'combination_index': combination_index,
            'status': result.status.value,
            'duration': result.duration_seconds,
            'error': result.error,
            'experiment_id': experiment_id,
            'success': True,
        }

    except Exception as e:
        import traceback
        return {
            'traffic_file': traffic_file,
            'combination_index': combination_index,
            'status': 'failed',
            'duration': 0,
            'error': f"{str(e)}\n{traceback.format_exc()}",
            'experiment_id': None,
            'success': False,
        }


class TaskManager:
    """
    任务管理器 - 单例模式
    管理所有仿真任务的创建、执行和状态查询
    支持任务历史持久化到JSON文件
    """

    _instance = None
    _lock = threading.Lock()
    _history_file = Path(__file__).parent.parent.parent / "data" / "task_history.json"

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._tasks: Dict[str, SimulationTask] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        # 全局订阅者（接收所有任务状态变化）
        self._global_subscribers: List[asyncio.Queue] = []
        # 主事件循环引用（用于跨线程通知）
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None
        # 全局 worker 数管理
        self._active_workers: Dict[str, int] = {}  # task_id -> worker_count
        self._workers_lock = threading.Lock()
        # 从文件加载历史任务
        self._load_history()

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """设置主事件循环引用（在 FastAPI 启动时调用）"""
        self._main_loop = loop
        logger.info("TaskManager 事件循环已设置")

    def create_task(
        self,
        mode: str,
        topology: str,
        config_path: str,
        traffic_file_path: str,
        traffic_files: List[str],
        max_time: int = 6000,
        experiment_name: Optional[str] = None,
        experiment_description: Optional[str] = None,
        save_to_db: bool = True,
        max_workers: Optional[int] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        die_config_path: Optional[str] = None,
        die_config_overrides: Optional[Dict[str, Any]] = None,
        sweep_combinations: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        创建新的仿真任务

        Returns:
            task_id
        """
        task_id = str(uuid.uuid4())[:8]

        task = SimulationTask(
            task_id=task_id,
            mode=mode,
            topology=topology,
            config_path=config_path,
            traffic_file_path=traffic_file_path,
            traffic_files=traffic_files,
            max_time=max_time,
            experiment_name=experiment_name,
            experiment_description=experiment_description,
            save_to_db=save_to_db,
            max_workers=max_workers,
            config_overrides=config_overrides,
            die_config_path=die_config_path,
            die_config_overrides=die_config_overrides,
            sweep_combinations=sweep_combinations,
        )

        self._tasks[task_id] = task
        return task_id

    def get_task(self, task_id: str) -> Optional[SimulationTask]:
        """获取任务"""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[SimulationTask]:
        """获取所有任务"""
        return list(self._tasks.values())

    def get_available_workers(self) -> int:
        """获取当前可用的 worker 数"""
        with self._workers_lock:
            used = sum(self._active_workers.values())
            return max(0, MAX_GLOBAL_WORKERS - used)

    def allocate_workers(self, task_id: str, requested: int) -> int:
        """
        为任务分配 worker 数
        Returns: 实际分配的 worker 数（至少为1）
        """
        with self._workers_lock:
            available = MAX_GLOBAL_WORKERS - sum(self._active_workers.values())
            allocated = max(1, min(requested, available))
            self._active_workers[task_id] = allocated
            logger.info(f"任务 {task_id} 分配 {allocated} 个 worker（请求 {requested}，可用 {available}）")
            return allocated

    def release_workers(self, task_id: str):
        """释放任务占用的 worker"""
        with self._workers_lock:
            if task_id in self._active_workers:
                released = self._active_workers.pop(task_id)
                logger.info(f"任务 {task_id} 释放 {released} 个 worker")

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: int = None,
        message: str = None,
        current_file: str = None,
        error: str = None,
        results: Dict = None,
    ):
        """更新任务状态"""
        task = self._tasks.get(task_id)
        if not task:
            return

        task.status = status
        if progress is not None:
            task.progress = progress
        if message is not None:
            task.message = message
        if current_file is not None:
            task.current_file = current_file
        if error is not None:
            task.error = error
        if results is not None:
            task.results = results

        if status == TaskStatus.RUNNING and task.started_at is None:
            task.started_at = datetime.now()
        if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            task.completed_at = datetime.now()
            # 释放 worker
            self.release_workers(task_id)
            # 任务完成时保存历史
            self._save_history()

        # 通知订阅者（线程安全方式）
        self._notify_subscribers_sync(task_id, task, level='task_level')

    def _notify_subscribers_sync(self, task_id: str, task: SimulationTask, level: str = 'task_level'):
        """
        线程安全的同步通知方法
        从普通线程中调用，自动调度到事件循环
        """
        try:
            # 尝试获取当前运行的事件循环（在异步上下文中）
            loop = asyncio.get_running_loop()
            loop.create_task(self._notify_subscribers(task_id, task, level))
        except RuntimeError:
            # 没有运行中的事件循环（在子线程中），使用保存的主事件循环
            if self._main_loop and self._main_loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._notify_subscribers(task_id, task, level),
                    self._main_loop
                )
            else:
                logger.warning(f"无法通知订阅者（任务 {task_id}）: 主事件循环未设置或未运行")

    async def _notify_subscribers(self, task_id: str, task: SimulationTask, level: str = 'task_level'):
        """
        通知任务状态变更

        Args:
            task_id: 任务ID
            task: 任务对象
            level: 通知级别
                - 'task_level': 任务状态变化，通知全局订阅者和单任务订阅者
                - 'sim_level': 仿真详情更新，只通知单任务订阅者
        """
        # 通知单任务订阅者（所有级别）
        if task_id in self._subscribers:
            for queue in self._subscribers[task_id]:
                await queue.put(task)

        # 只有任务级别更新才通知全局订阅者
        if level == 'task_level' and self._global_subscribers:
            global_update = {
                'task_id': task.task_id,
                'status': task.status.value,
                'progress': task.progress,
                'message': task.message,
                'experiment_name': task.experiment_name,
                'current_file': task.current_file,
                'error': task.error,
            }
            for queue in self._global_subscribers:
                await queue.put(global_update)

    def subscribe(self, task_id: str) -> asyncio.Queue:
        """订阅任务状态更新"""
        if task_id not in self._subscribers:
            self._subscribers[task_id] = []
        queue = asyncio.Queue()
        self._subscribers[task_id].append(queue)
        return queue

    def subscribe_global(self) -> asyncio.Queue:
        """订阅所有任务状态更新（全局）"""
        queue = asyncio.Queue()
        self._global_subscribers.append(queue)
        return queue

    def unsubscribe_global(self, queue: asyncio.Queue):
        """取消全局订阅"""
        if queue in self._global_subscribers:
            self._global_subscribers.remove(queue)

    def unsubscribe(self, task_id: str, queue: asyncio.Queue):
        """取消订阅"""
        if task_id in self._subscribers:
            self._subscribers[task_id].remove(queue)

    async def run_task(self, task_id: str):
        """
        执行仿真任务（单执行单元串行执行带进度，多执行单元并行执行）

        执行单元 = 文件 × 参数组合 的笛卡尔积
        - 无 sweep_combinations: 每个文件是一个执行单元
        - 有 sweep_combinations: 文件数 × 组合数 = 总执行单元数
        """
        task = self._tasks.get(task_id)
        if not task:
            return

        try:
            # 构建执行单元列表 (文件 × 参数组合 笛卡尔积)
            combinations = task.sweep_combinations or [{}]  # 无组合时用空字典
            execution_units = []
            for traffic_file in task.traffic_files:
                for combo_idx, combo in enumerate(combinations):
                    execution_units.append({
                        'traffic_file': traffic_file,
                        'combination': combo,
                        'combination_index': combo_idx if task.sweep_combinations else None,
                    })

            total_units = len(execution_units)
            results_list = []
            experiment_id = None

            # 单执行单元时使用串行执行，显示详细进度
            if total_units == 1:
                await self._run_single_file_task(task_id, task, execution_units[0])
                return

            # 多执行单元时使用并行执行
            self.update_task_status(task_id, TaskStatus.RUNNING, message="正在初始化并行仿真...")

            # 准备所有仿真参数
            sim_params_list = []
            for unit in execution_units:
                sim_params = {
                    'mode': task.mode,
                    'config_path': task.config_path,
                    'topology': task.topology,
                    'config_overrides': task.config_overrides,
                    'combination_overrides': unit['combination'],
                    'combination_index': unit['combination_index'],
                    'die_config_path': task.die_config_path,
                    'die_config_overrides': task.die_config_overrides,
                    'traffic_file_path': task.traffic_file_path,
                    'traffic_file': unit['traffic_file'],
                    'max_time': task.max_time,
                    'save_to_db': task.save_to_db,
                    'experiment_name': task.experiment_name,
                    'experiment_description': task.experiment_description,
                    'show_result': not task.save_to_db,
                }
                sim_params_list.append(sim_params)

            # 确定并行进程数（考虑全局限制）
            if task.max_workers:
                requested_workers = min(task.max_workers, total_units)
            else:
                requested_workers = min(multiprocessing.cpu_count(), total_units)
            # 分配 worker（考虑其他正在运行的任务）
            max_workers = self.allocate_workers(task_id, requested_workers)

            # 更新状态
            task.sim_details = {
                "file_index": 0,
                "total_files": total_units,
                "current_file": f"并行执行中 ({max_workers} 进程)",
                "sim_progress": 0,
                "current_time": 0,
                "max_time": task.max_time,
                "req_count": 0,
                "total_req": 0,
                "recv_flits": 0,
                "total_flits": 0,
                "trans_flits": 0,
            }

            # 使用线程池在后台运行进程池（避免阻塞事件循环）
            loop = asyncio.get_event_loop()

            def run_parallel():
                completed_results = []
                timed_out_files = []
                executor = None
                try:
                    executor = ProcessPoolExecutor(max_workers=max_workers)
                    task._executor = executor  # 保存引用以便取消

                    future_to_file = {
                        executor.submit(_run_single_simulation, params): params['traffic_file']
                        for params in sim_params_list
                    }
                    task._futures = list(future_to_file.keys())  # 保存 futures 引用

                    for future in as_completed(future_to_file, timeout=TASK_TIMEOUT):
                        if task.status == TaskStatus.CANCELLED:
                            # 取消所有pending的任务
                            for f in future_to_file:
                                f.cancel()
                            logger.info(f"任务 {task.task_id} 被取消")
                            break

                        try:
                            # 单个future的超时检查
                            result = future.result(timeout=60)  # 60秒等待结果
                            completed_results.append(result)
                        except FuturesTimeoutError:
                            # 单个任务超时
                            file_name = future_to_file.get(future, 'unknown')
                            timed_out_files.append(file_name)
                            logger.warning(f"仿真任务超时: {file_name}")
                            completed_results.append({
                                'traffic_file': file_name,
                                'status': 'failed',
                                'duration': 0,
                                'error': '任务执行超时',
                                'experiment_id': None,
                                'success': False,
                            })
                        except Exception as e:
                            # 其他异常
                            file_name = future_to_file.get(future, 'unknown')
                            logger.error(f"仿真任务异常: {file_name} - {e}")
                            completed_results.append({
                                'traffic_file': file_name,
                                'status': 'failed',
                                'duration': 0,
                                'error': str(e),
                                'experiment_id': None,
                                'success': False,
                            })

                        # 更新进度
                        completed_count = len(completed_results)
                        task.progress = int((completed_count / total_units) * 100)
                        last_result = completed_results[-1] if completed_results else {}
                        # 构建当前文件显示（包含组合索引）
                        current_display = last_result.get('traffic_file', '')
                        if last_result.get('combination_index') is not None:
                            current_display = f"{current_display} [组合{last_result['combination_index']}]"
                        task.sim_details = {
                            "file_index": completed_count,
                            "total_files": total_units,
                            "current_file": f"已完成: {current_display}",
                            "sim_progress": 100,
                            "current_time": task.max_time,
                            "max_time": task.max_time,
                            "req_count": 0,
                            "total_req": 0,
                            "recv_flits": 0,
                            "total_flits": 0,
                            "trans_flits": 0,
                        }

                        # 通知订阅者（包括全局订阅者）
                        self._notify_subscribers_sync(task_id, task, level='task_level')

                        # 保存最后一个experiment_id
                        if last_result.get('experiment_id'):
                            nonlocal experiment_id
                            experiment_id = last_result['experiment_id']
                except BrokenPipeError:
                    logger.warning(f"任务 {task.task_id} 进程池管道断开，可能是父进程被终止")
                except BrokenExecutor:
                    logger.warning(f"任务 {task.task_id} 进程池执行器已损坏")
                finally:
                    # 清理 executor
                    if executor:
                        executor.shutdown(wait=False)
                    task._executor = None
                    task._futures = []

                return completed_results

            # 在线程中运行并行任务
            results_list = await loop.run_in_executor(None, run_parallel)

            # 合并结果
            combined_results = {
                'total_files': total_units,
                'completed_files': len([r for r in results_list if r['status'] == 'completed']),
                'failed_files': len([r for r in results_list if r['status'] == 'failed']),
                'experiment_id': experiment_id,
                'file_results': [
                    {
                        'file': r['traffic_file'],
                        'combination_index': r.get('combination_index'),
                        'status': r['status'],
                        'duration': r['duration'],
                        'error': r['error'],
                    }
                    for r in results_list
                ],
            }

            # 检查是否被取消
            if task.status == TaskStatus.CANCELLED:
                completed_count = len(results_list)
                combined_results['cancelled'] = True
                self.update_task_status(
                    task_id,
                    TaskStatus.CANCELLED,
                    message=f"任务已取消，已完成 {completed_count}/{total_units} 个执行单元",
                    results=combined_results,
                )
            else:
                self.update_task_status(
                    task_id,
                    TaskStatus.COMPLETED,
                    progress=100,
                    message="仿真完成",
                    results=combined_results,
                )

        except FuturesTimeoutError:
            # 整体超时
            logger.error(f"仿真任务整体超时 [{task_id}]")
            self.update_task_status(
                task_id,
                TaskStatus.FAILED,
                message="仿真任务整体超时",
                error=f"任务执行超过 {TASK_TIMEOUT} 秒",
            )
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(f"仿真任务失败 [{task_id}]: {error_msg}")
            self.update_task_status(
                task_id,
                TaskStatus.FAILED,
                message="仿真失败",
                error=error_msg,
            )

    async def _run_single_file_task(self, task_id: str, task: SimulationTask, execution_unit: Dict[str, Any]):
        """
        执行单执行单元仿真任务（串行执行，显示详细进度）

        Args:
            task_id: 任务ID
            task: 任务对象
            execution_unit: 执行单元，包含 traffic_file, combination, combination_index
        """
        from .simulation_engine import SimulationEngine, SimulationMode

        traffic_file = execution_unit['traffic_file']
        combination = execution_unit.get('combination', {})
        combination_index = execution_unit.get('combination_index')

        # 构建显示名称
        display_name = traffic_file
        if combination_index is not None:
            display_name = f"{traffic_file} [组合{combination_index}]"

        self.update_task_status(task_id, TaskStatus.RUNNING, message=f"正在仿真: {display_name}")

        # 初始化sim_details
        task.sim_details = {
            "file_index": 0,
            "total_files": 1,
            "current_file": display_name,
            "sim_progress": 0,
            "current_time": 0,
            "max_time": task.max_time,
            "req_count": 0,
            "total_req": 0,
            "recv_flits": 0,
            "total_flits": 0,
            "trans_flits": 0,
        }

        try:
            # 合并配置覆盖项: base config_overrides + combination
            config_overrides = task.config_overrides or {}
            merged_overrides = {**config_overrides, **combination}

            engine = SimulationEngine(
                mode=SimulationMode(task.mode),
                config_path=task.config_path,
                topology=task.topology,
                verbose=0,
                config_overrides=merged_overrides if merged_overrides else None,
                die_config_path=task.die_config_path,
                die_config_overrides=task.die_config_overrides,
            )

            # 设置进度回调
            def on_progress(data: dict):
                if task.status == TaskStatus.CANCELLED:
                    return
                task.sim_details.update(data)
                # 计算总体进度
                if data.get("max_time", 0) > 0:
                    task.progress = int((data.get("current_time", 0) / data["max_time"]) * 100)
                # 通知 WebSocket 订阅者（线程安全方式）
                # 使用 sim_level 级别，只通知单任务订阅者（不通知全局订阅者）
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._notify_subscribers(task_id, task, level='sim_level'))
                except RuntimeError:
                    # 没有运行中的事件循环，跳过通知（进度数据已更新到 task 对象）
                    pass

            engine.set_progress_callback(on_progress)
            task._current_engine = engine

            engine.setup(
                traffic_file_path=task.traffic_file_path,
                traffic_files=[traffic_file],
                show_result_analysis=not task.save_to_db,
            )

            result = await engine.run_async(max_time=task.max_time)
            result.results['traffic_file'] = traffic_file

            # 保存到数据库（仅在成功时保存）
            experiment_id = None
            if task.save_to_db and result.status.value == 'completed':
                exp_name = task.experiment_name or '仿真实验'
                experiment_id = engine.save_to_database(
                    experiment_name=exp_name,
                    description=task.experiment_description
                )

            # 检查是否被取消
            if task.status == TaskStatus.CANCELLED:
                self.update_task_status(
                    task_id,
                    TaskStatus.CANCELLED,
                    message="任务已取消",
                )
                return

            combined_results = {
                'total_files': 1,
                'completed_files': 1 if result.status.value == 'completed' else 0,
                'failed_files': 0 if result.status.value == 'completed' else 1,
                'experiment_id': experiment_id,
                'file_results': [{
                    'file': traffic_file,
                    'status': result.status.value,
                    'duration': result.duration_seconds,
                    'error': result.error,
                }],
            }

            self.update_task_status(
                task_id,
                TaskStatus.COMPLETED,
                progress=100,
                message="仿真完成",
                results=combined_results,
            )

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(f"单文件仿真任务失败 [{task_id}]: {error_msg}")
            self.update_task_status(
                task_id,
                TaskStatus.FAILED,
                message="仿真失败",
                error=error_msg,
            )

    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务 - 真正终止运行中的仿真进程

        对于单文件任务：调用引擎的 cancel() 方法，仿真循环会检测到取消标志并退出
        对于多文件任务：关闭 ProcessPoolExecutor 并取消所有 futures
        """
        task = self._tasks.get(task_id)
        if not task:
            return False

        if task.status != TaskStatus.RUNNING:
            return False

        # 标记任务状态为取消
        task.status = TaskStatus.CANCELLED
        logger.info(f"正在取消任务 {task_id}")

        # 单文件任务：调用引擎的取消方法
        if task._current_engine:
            try:
                task._current_engine.cancel()
                logger.info(f"任务 {task_id} 的仿真引擎已标记取消")
            except Exception as e:
                logger.error(f"取消仿真引擎失败: {e}")

        # 多文件任务：关闭进程池
        if task._executor:
            try:
                # 取消所有未完成的 futures
                for future in task._futures:
                    future.cancel()
                # 强制关闭进程池（不等待）
                task._executor.shutdown(wait=False, cancel_futures=True)
                logger.info(f"任务 {task_id} 的进程池已关闭")
            except Exception as e:
                logger.error(f"关闭进程池失败: {e}")

        # 释放 worker
        self.release_workers(task_id)

        return True

    def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False

    def get_history(self, limit: int = 20) -> List[SimulationTask]:
        """获取历史任务"""
        tasks = sorted(
            self._tasks.values(),
            key=lambda t: t.created_at,
            reverse=True,
        )
        return tasks[:limit]

    def _load_history(self):
        """从文件加载历史任务"""
        if not self._history_file.exists():
            return

        try:
            with open(self._history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for task_data in data:
                # 转换状态枚举
                task_data['status'] = TaskStatus(task_data['status'])
                # 转换时间字段
                task_data['created_at'] = datetime.fromisoformat(task_data['created_at'])
                if task_data.get('started_at'):
                    task_data['started_at'] = datetime.fromisoformat(task_data['started_at'])
                if task_data.get('completed_at'):
                    task_data['completed_at'] = datetime.fromisoformat(task_data['completed_at'])

                task = SimulationTask(**task_data)
                self._tasks[task.task_id] = task
        except Exception as e:
            print(f"加载任务历史失败: {e}")

    def _save_history(self):
        """保存任务历史到文件"""
        # 确保目录存在
        self._history_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            # 只保存已完成的任务（最近100条）
            completed_tasks = [
                t for t in self._tasks.values()
                if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            ]
            completed_tasks.sort(key=lambda t: t.created_at, reverse=True)
            tasks_to_save = completed_tasks[:100]

            data = []
            for task in tasks_to_save:
                task_dict = asdict(task)
                # 移除不可序列化的字段
                task_dict.pop('_current_engine', None)
                # 转换枚举为字符串
                task_dict['status'] = task.status.value
                # 转换时间为ISO字符串
                task_dict['created_at'] = task.created_at.isoformat()
                if task.started_at:
                    task_dict['started_at'] = task.started_at.isoformat()
                if task.completed_at:
                    task_dict['completed_at'] = task.completed_at.isoformat()
                data.append(task_dict)

            with open(self._history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存任务历史失败: {e}")

    def clear_history(self):
        """清空历史任务（保留正在运行的任务）"""
        running_tasks = {
            tid: task for tid, task in self._tasks.items()
            if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
        }
        self._tasks = running_tasks
        self._save_history()


# 全局任务管理器实例
task_manager = TaskManager()
