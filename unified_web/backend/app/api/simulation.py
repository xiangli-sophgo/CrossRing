"""
仿真执行 API - 提供仿真任务的创建、执行、查询和取消
"""

import asyncio
from pathlib import Path
from typing import List, Optional, Literal
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.config import BASE_DIR, TRAFFIC_OUTPUT_DIR, TOPOLOGIES_DIR
from app.core.task_manager import task_manager, TaskStatus

router = APIRouter(prefix="/simulation")


# ==================== 请求/响应模型 ====================

class SimulationRequest(BaseModel):
    """仿真请求"""
    mode: Literal["kcin", "dcin"] = Field(..., description="仿真模式: kcin(单Die) 或 dcin(多Die)")
    topology: str = Field(default="5x4", description="拓扑类型")
    config_path: Optional[str] = Field(None, description="配置文件路径(相对于config/topologies/)")
    traffic_source: Literal["file", "generate"] = Field(default="file", description="流量来源")
    traffic_files: List[str] = Field(default=[], description="流量文件列表")
    traffic_path: Optional[str] = Field(None, description="流量文件目录(相对于traffic/)")
    max_time: int = Field(default=6000, description="最大仿真时间(ns)")
    save_to_db: bool = Field(default=True, description="是否保存到数据库")
    experiment_name: Optional[str] = Field(None, description="实验名称")
    result_granularity: Literal["per_file", "per_batch"] = Field(
        default="per_file",
        description="结果粒度: per_file(每文件一条) 或 per_batch(每批次一条)"
    )


class TaskResponse(BaseModel):
    """任务响应"""
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str
    status: str
    progress: int
    current_file: str
    message: str
    error: Optional[str]
    results: Optional[dict]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]


class ConfigOption(BaseModel):
    """配置选项"""
    name: str
    path: str


class TrafficFileInfo(BaseModel):
    """流量文件信息"""
    name: str
    path: str
    size: int


# ==================== API端点 ====================

@router.post("/run", response_model=TaskResponse)
async def run_simulation(request: SimulationRequest):
    """
    启动仿真任务
    """
    # 解析配置文件路径
    if request.config_path:
        config_path = str(TOPOLOGIES_DIR / request.config_path)
    else:
        # 根据模式和拓扑选择默认配置
        if request.mode == "kcin":
            config_path = str(TOPOLOGIES_DIR / f"topo_{request.topology}.yaml")
        else:
            config_path = str(TOPOLOGIES_DIR / f"dcin_{request.topology}_config.yaml")

    # 检查配置文件是否存在
    if not Path(config_path).exists():
        raise HTTPException(status_code=400, detail=f"配置文件不存在: {config_path}")

    # 解析流量文件路径
    if request.traffic_path:
        traffic_file_path = str(TRAFFIC_OUTPUT_DIR / request.traffic_path)
    else:
        traffic_file_path = str(TRAFFIC_OUTPUT_DIR)

    # 检查流量文件
    if not request.traffic_files:
        raise HTTPException(status_code=400, detail="请指定流量文件")

    # 创建任务
    task_id = task_manager.create_task(
        mode=request.mode,
        topology=request.topology,
        config_path=config_path,
        traffic_file_path=traffic_file_path,
        traffic_files=request.traffic_files,
        max_time=request.max_time,
        experiment_name=request.experiment_name,
        save_to_db=request.save_to_db,
        result_granularity=request.result_granularity,
    )

    # 异步执行任务
    asyncio.create_task(task_manager.run_task(task_id))

    return TaskResponse(
        task_id=task_id,
        status="pending",
        message="仿真任务已创建",
    )


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    获取任务状态
    """
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status.value,
        progress=task.progress,
        current_file=task.current_file,
        message=task.message,
        error=task.error,
        results=task.results if task.results else None,
        created_at=task.created_at.isoformat(),
        started_at=task.started_at.isoformat() if task.started_at else None,
        completed_at=task.completed_at.isoformat() if task.completed_at else None,
    )


@router.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    """
    取消任务
    """
    success = task_manager.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="任务不存在或无法取消")

    return {"success": True, "message": "任务已取消"}


@router.delete("/{task_id}")
async def delete_task(task_id: str):
    """
    删除任务
    """
    success = task_manager.delete_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="任务不存在")

    return {"success": True, "message": "任务已删除"}


@router.get("/history")
async def get_history(limit: int = 20):
    """
    获取历史任务
    """
    tasks = task_manager.get_history(limit)
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "mode": t.mode,
                "topology": t.topology,
                "status": t.status.value,
                "progress": t.progress,
                "message": t.message,
                "created_at": t.created_at.isoformat(),
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                "traffic_files": t.traffic_files,
            }
            for t in tasks
        ]
    }


@router.get("/configs")
async def list_configs():
    """
    列出可用的配置文件
    """
    kcin_configs = []
    dcin_configs = []

    if TOPOLOGIES_DIR.exists():
        for f in TOPOLOGIES_DIR.glob("*.yaml"):
            name = f.stem
            if name.startswith("topo_"):
                kcin_configs.append(ConfigOption(name=name.replace("topo_", ""), path=f.name))
            elif name.startswith("dcin_"):
                dcin_configs.append(ConfigOption(name=name, path=f.name))

    return {
        "kcin": [c.dict() for c in kcin_configs],
        "dcin": [c.dict() for c in dcin_configs],
    }


@router.get("/traffic-files")
async def list_traffic_files(path: str = ""):
    """
    列出流量文件
    """
    base_path = TRAFFIC_OUTPUT_DIR / path if path else TRAFFIC_OUTPUT_DIR

    if not base_path.exists():
        return {"files": [], "directories": []}

    files = []
    directories = []

    for item in base_path.iterdir():
        if item.is_dir():
            directories.append(item.name)
        elif item.suffix == ".txt":
            files.append(TrafficFileInfo(
                name=item.name,
                path=str(item.relative_to(TRAFFIC_OUTPUT_DIR)),
                size=item.stat().st_size,
            ).dict())

    return {
        "current_path": path,
        "files": files,
        "directories": sorted(directories),
    }


# ==================== WebSocket 实时进度 ====================

@router.websocket("/ws/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str):
    """
    WebSocket 实时进度推送
    """
    await websocket.accept()

    task = task_manager.get_task(task_id)
    if not task:
        await websocket.send_json({"error": "任务不存在"})
        await websocket.close()
        return

    # 订阅任务更新
    queue = task_manager.subscribe(task_id)

    try:
        # 发送当前状态
        await websocket.send_json({
            "task_id": task.task_id,
            "status": task.status.value,
            "progress": task.progress,
            "current_file": task.current_file,
            "message": task.message,
        })

        # 持续推送更新
        while True:
            try:
                updated_task = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json({
                    "task_id": updated_task.task_id,
                    "status": updated_task.status.value,
                    "progress": updated_task.progress,
                    "current_file": updated_task.current_file,
                    "message": updated_task.message,
                    "error": updated_task.error,
                })

                # 任务完成后退出
                if updated_task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                    break

            except asyncio.TimeoutError:
                # 发送心跳
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        pass
    finally:
        task_manager.unsubscribe(task_id, queue)
