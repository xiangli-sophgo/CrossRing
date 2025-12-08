"""
异步任务管理器 - 管理仿真任务的生命周期
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from enum import Enum
import threading


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
    save_to_db: bool
    result_granularity: str  # per_file or per_batch

    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0  # 0-100
    current_file: str = ""
    message: str = ""
    error: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TaskManager:
    """
    任务管理器 - 单例模式
    管理所有仿真任务的创建、执行和状态查询
    """

    _instance = None
    _lock = threading.Lock()

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

    def create_task(
        self,
        mode: str,
        topology: str,
        config_path: str,
        traffic_file_path: str,
        traffic_files: List[str],
        max_time: int = 6000,
        experiment_name: Optional[str] = None,
        save_to_db: bool = True,
        result_granularity: str = "per_file",
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
            save_to_db=save_to_db,
            result_granularity=result_granularity,
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
        """执行仿真任务"""
        from .simulation_engine import SimulationEngine, SimulationMode

        task = self._tasks.get(task_id)
        if not task:
            return

        self.update_task_status(task_id, TaskStatus.RUNNING, message="正在初始化仿真引擎...")

        try:
            mode = SimulationMode(task.mode)
            total_files = len(task.traffic_files)

            results_list = []

            for i, traffic_file in enumerate(task.traffic_files):
                progress = int((i / total_files) * 100)
                self.update_task_status(
                    task_id,
                    TaskStatus.RUNNING,
                    progress=progress,
                    current_file=traffic_file,
                    message=f"正在仿真: {traffic_file} ({i+1}/{total_files})",
                )

                engine = SimulationEngine(
                    mode=mode,
                    config_path=task.config_path,
                    topology=task.topology,
                    verbose=0,
                )

                engine.setup(
                    traffic_file_path=task.traffic_file_path,
                    traffic_files=[traffic_file],
                )

                result = await engine.run_async(max_time=task.max_time)
                result.results['traffic_file'] = traffic_file
                results_list.append(result)

                # 按文件保存到数据库
                if task.save_to_db and task.result_granularity == "per_file":
                    exp_name = f"{task.experiment_name or '仿真实验'}_{traffic_file}"
                    engine.save_to_database(experiment_name=exp_name)

            # 按批次保存到数据库
            if task.save_to_db and task.result_granularity == "per_batch":
                # 这里需要更复杂的逻辑来合并结果并保存
                pass

            # 合并结果
            combined_results = {
                'total_files': total_files,
                'completed_files': len([r for r in results_list if r.status.value == 'completed']),
                'failed_files': len([r for r in results_list if r.status.value == 'failed']),
                'file_results': [
                    {
                        'file': r.results.get('traffic_file', ''),
                        'status': r.status.value,
                        'duration': r.duration_seconds,
                        'error': r.error,
                    }
                    for r in results_list
                ],
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
            # 尝试取消正在运行的异步任务
            if task_id in self._running_tasks:
                self._running_tasks[task_id].cancel()

        self.update_task_status(task_id, TaskStatus.CANCELLED, message="任务已取消")
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


# 全局任务管理器实例
task_manager = TaskManager()
