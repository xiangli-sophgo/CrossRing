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


def _run_single_simulation(sim_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    在子进程中运行单个仿真（独立函数，用于多进程）

    Args:
        sim_params: 仿真参数字典

    Returns:
        结果字典
    """
    from .simulation_engine import SimulationEngine, SimulationMode

    traffic_file = sim_params['traffic_file']
    try:
        engine = SimulationEngine(
            mode=SimulationMode(sim_params['mode']),
            config_path=sim_params['config_path'],
            topology=sim_params['topology'],
            verbose=0,
            config_overrides=sim_params.get('config_overrides'),
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

        # 保存到数据库（每个文件独立保存）
        experiment_id = None
        if sim_params.get('save_to_db'):
            exp_name = sim_params.get('experiment_name') or '仿真实验'
            experiment_id = engine.save_to_database(
                experiment_name=exp_name,
                description=sim_params.get('experiment_description')
            )

        return {
            'traffic_file': traffic_file,
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
        # 从文件加载历史任务
        self._load_history()

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
        )

        self._tasks[task_id] = task
        return task_id

    def get_task(self, task_id: str) -> Optional[SimulationTask]:
        """获取任务"""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[SimulationTask]:
        """获取所有任务"""
        return list(self._tasks.values())

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
            # 任务完成时保存历史
            self._save_history()

        # 通知订阅者
        asyncio.create_task(self._notify_subscribers(task_id, task))

    async def _notify_subscribers(self, task_id: str, task: SimulationTask):
        """通知任务状态变更"""
        if task_id in self._subscribers:
            for queue in self._subscribers[task_id]:
                await queue.put(task)

    def subscribe(self, task_id: str) -> asyncio.Queue:
        """订阅任务状态更新"""
        if task_id not in self._subscribers:
            self._subscribers[task_id] = []
        queue = asyncio.Queue()
        self._subscribers[task_id].append(queue)
        return queue

    def unsubscribe(self, task_id: str, queue: asyncio.Queue):
        """取消订阅"""
        if task_id in self._subscribers:
            self._subscribers[task_id].remove(queue)

    async def run_task(self, task_id: str):
        """执行仿真任务（单文件串行执行带进度，多文件并行执行）"""
        task = self._tasks.get(task_id)
        if not task:
            return

        try:
            total_files = len(task.traffic_files)
            results_list = []
            experiment_id = None

            # 单文件时使用串行执行，显示详细进度
            if total_files == 1:
                await self._run_single_file_task(task_id, task)
                return

            # 多文件时使用并行执行
            self.update_task_status(task_id, TaskStatus.RUNNING, message="正在初始化并行仿真...")

            # 准备所有仿真参数
            sim_params_list = []
            for traffic_file in task.traffic_files:
                sim_params = {
                    'mode': task.mode,
                    'config_path': task.config_path,
                    'topology': task.topology,
                    'config_overrides': task.config_overrides,
                    'die_config_path': task.die_config_path,
                    'die_config_overrides': task.die_config_overrides,
                    'traffic_file_path': task.traffic_file_path,
                    'traffic_file': traffic_file,
                    'max_time': task.max_time,
                    'save_to_db': task.save_to_db,
                    'experiment_name': task.experiment_name,
                    'experiment_description': task.experiment_description,
                    'show_result': not task.save_to_db,
                }
                sim_params_list.append(sim_params)

            # 确定并行进程数
            if task.max_workers:
                max_workers = min(task.max_workers, total_files)
            else:
                max_workers = min(multiprocessing.cpu_count(), total_files)

            # 更新状态
            task.sim_details = {
                "file_index": 0,
                "total_files": total_files,
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
                try:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        future_to_file = {
                            executor.submit(_run_single_simulation, params): params['traffic_file']
                            for params in sim_params_list
                        }

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
                            task.progress = int((completed_count / total_files) * 100)
                            last_result = completed_results[-1] if completed_results else {}
                            task.sim_details = {
                                "file_index": completed_count,
                                "total_files": total_files,
                                "current_file": f"已完成: {last_result.get('traffic_file', '')}",
                                "sim_progress": 100,
                                "current_time": task.max_time,
                                "max_time": task.max_time,
                                "req_count": 0,
                                "total_req": 0,
                                "recv_flits": 0,
                                "total_flits": 0,
                                "trans_flits": 0,
                            }

                            # 保存最后一个experiment_id
                            if last_result.get('experiment_id'):
                                nonlocal experiment_id
                                experiment_id = last_result['experiment_id']
                except BrokenPipeError:
                    logger.warning(f"任务 {task.task_id} 进程池管道断开，可能是父进程被终止")
                except BrokenExecutor:
                    logger.warning(f"任务 {task.task_id} 进程池执行器已损坏")

                return completed_results

            # 在线程中运行并行任务
            results_list = await loop.run_in_executor(None, run_parallel)

            # 合并结果
            combined_results = {
                'total_files': total_files,
                'completed_files': len([r for r in results_list if r['status'] == 'completed']),
                'failed_files': len([r for r in results_list if r['status'] == 'failed']),
                'experiment_id': experiment_id,
                'file_results': [
                    {
                        'file': r['traffic_file'],
                        'status': r['status'],
                        'duration': r['duration'],
                        'error': r['error'],
                    }
                    for r in results_list
                ],
            }

            # 检查是否被取消
            if task.status == TaskStatus.CANCELLED:
                completed_files = len(results_list)
                combined_results['cancelled'] = True
                self.update_task_status(
                    task_id,
                    TaskStatus.CANCELLED,
                    message=f"任务已取消，已完成 {completed_files}/{total_files} 个文件",
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

    async def _run_single_file_task(self, task_id: str, task: SimulationTask):
        """执行单文件仿真任务（串行执行，显示详细进度）"""
        from .simulation_engine import SimulationEngine, SimulationMode

        traffic_file = task.traffic_files[0]
        self.update_task_status(task_id, TaskStatus.RUNNING, message=f"正在仿真: {traffic_file}")

        # 初始化sim_details
        task.sim_details = {
            "file_index": 0,
            "total_files": 1,
            "current_file": traffic_file,
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
            engine = SimulationEngine(
                mode=SimulationMode(task.mode),
                config_path=task.config_path,
                topology=task.topology,
                verbose=0,
                config_overrides=task.config_overrides,
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

            engine.set_progress_callback(on_progress)
            task._current_engine = engine

            engine.setup(
                traffic_file_path=task.traffic_file_path,
                traffic_files=[traffic_file],
                show_result_analysis=not task.save_to_db,
            )

            result = await engine.run_async(max_time=task.max_time)
            result.results['traffic_file'] = traffic_file

            # 保存到数据库
            experiment_id = None
            if task.save_to_db:
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
        """取消任务"""
        task = self._tasks.get(task_id)
        if not task:
            return False

        if task.status == TaskStatus.RUNNING:
            # 标记任务状态为取消
            # 并行模式下，run_parallel 会检测到这个状态并停止提交新任务
            task.status = TaskStatus.CANCELLED

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
