"""
仿真执行 API - 提供仿真任务的创建、执行、查询和取消
"""

import asyncio
import yaml
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any
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
    config_overrides: Optional[Dict[str, Any]] = Field(None, description="配置覆盖项")
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
        config_overrides=request.config_overrides,
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


def _sanitize_for_json(obj):
    """将inf/nan等特殊浮点数转换为JSON兼容格式"""
    import math
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        elif math.isnan(obj):
            return "nan"
    return obj


@router.get("/config/{config_path:path}")
async def get_config_content(config_path: str):
    """
    读取配置文件内容
    """
    config_file = TOPOLOGIES_DIR / config_path
    if not config_file.exists():
        raise HTTPException(status_code=404, detail=f"配置文件不存在: {config_path}")

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
        return _sanitize_for_json(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取配置文件失败: {str(e)}")


def _format_config_with_comments(content: dict) -> str:
    """
    将配置内容格式化为带注释和分类的YAML字符串
    按照前端页面分类: Basic Parameters, Buffer Size, KCIN Config, Feature Config
    """
    lines = []

    # 配置分类定义 (按前端页面分类)
    categories = {
        "Basic Parameters": {
            "keys": ["TOPO_TYPE", "FLIT_SIZE", "BURST", "NETWORK_FREQUENCY"],
            "comments": {
                "TOPO_TYPE": "格式 AxB，自动计算拓扑参数",
            }
        },
        "Buffer Size": {
            "keys": ["RN_RDB_SIZE", "RN_WDB_SIZE", "SN_DDR_RDB_SIZE", "SN_DDR_WDB_SIZE", "SN_L2M_RDB_SIZE", "SN_L2M_WDB_SIZE", "UNIFIED_RW_TRACKER"],
            "comments": {
                "RN_RDB_SIZE": "RN读数据缓冲区",
                "RN_WDB_SIZE": "RN写数据缓冲区",
                "SN_DDR_RDB_SIZE": "SN DDR读数据缓冲区",
                "SN_DDR_WDB_SIZE": "SN DDR写数据缓冲区",
                "SN_L2M_RDB_SIZE": "SN L2M读数据缓冲区",
                "SN_L2M_WDB_SIZE": "SN L2M写数据缓冲区",
                "UNIFIED_RW_TRACKER": "true=读写共享资源池，false=读写分离",
            }
        },
        "KCIN Config": {
            "keys": [
                # Slice Per Link
                "SLICE_PER_LINK_HORIZONTAL", "SLICE_PER_LINK_VERTICAL",
                # FIFO Depth
                "IQ_CH_FIFO_DEPTH", "EQ_CH_FIFO_DEPTH", "IQ_OUT_FIFO_DEPTH_HORIZONTAL", "IQ_OUT_FIFO_DEPTH_VERTICAL",
                "IQ_OUT_FIFO_DEPTH_EQ", "RB_OUT_FIFO_DEPTH", "RB_IN_FIFO_DEPTH", "EQ_IN_FIFO_DEPTH",
                "IP_L2H_FIFO_DEPTH", "IP_H2L_H_FIFO_DEPTH", "IP_H2L_L_FIFO_DEPTH",
                # ETag
                "TL_Etag_T2_UE_MAX", "TL_Etag_T1_UE_MAX", "TR_Etag_T2_UE_MAX",
                "TU_Etag_T2_UE_MAX", "TU_Etag_T1_UE_MAX", "TD_Etag_T2_UE_MAX",
                "ETag_BOTHSIDE_UPGRADE", "ETAG_T1_ENABLED",
                # ITag
                "ITag_TRIGGER_Th_H", "ITag_TRIGGER_Th_V", "ITag_MAX_Num_H", "ITag_MAX_Num_V",
                # Latency
                "DDR_R_LATENCY", "DDR_R_LATENCY_VAR", "DDR_W_LATENCY", "L2M_R_LATENCY", "L2M_W_LATENCY",
                "SN_TRACKER_RELEASE_LATENCY", "SN_PROCESSING_LATENCY", "RN_PROCESSING_LATENCY",
                # Bandwidth Limit
                "GDMA_BW_LIMIT", "SDMA_BW_LIMIT", "CDMA_BW_LIMIT", "DDR_BW_LIMIT", "L2M_BW_LIMIT",
            ],
            "comments": {
                "ETAG_T1_ENABLED": "true=三级(T2→T1→T0), false=两级(T2→T0)",
                "SN_PROCESSING_LATENCY": "SN端处理延迟 (ns)",
                "RN_PROCESSING_LATENCY": "RN端处理延迟 (ns)",
            }
        },
        "Feature Config": {
            "keys": [
                "CROSSRING_VERSION",
                "RB_ONLY_TAG_NUM_HORIZONTAL", "RB_ONLY_TAG_NUM_VERTICAL",
                "ORDERING_PRESERVATION_MODE", "ORDERING_ETAG_UPGRADE_MODE", "ORDERING_GRANULARITY",
                "TL_ALLOWED_SOURCE_NODES", "TR_ALLOWED_SOURCE_NODES", "TU_ALLOWED_SOURCE_NODES", "TD_ALLOWED_SOURCE_NODES",
                "REVERSE_DIRECTION_ENABLED", "REVERSE_DIRECTION_THRESHOLD",
            ],
            "comments": {
                "RB_ONLY_TAG_NUM_HORIZONTAL": "仅V2生效",
                "RB_ONLY_TAG_NUM_VERTICAL": "仅V2生效",
                "ORDERING_PRESERVATION_MODE": "0=不保序, 1=单侧下环(TL/TU), 2=双侧下环(白名单), 3=动态方向",
                "ORDERING_ETAG_UPGRADE_MODE": "0=仅资源失败升级ETag, 1=保序失败也升级ETag",
                "ORDERING_GRANULARITY": "0=IP层级, 1=节点层级",
                "REVERSE_DIRECTION_ENABLED": "启用反方向流控 (0=禁用, 1=启用)",
                "REVERSE_DIRECTION_THRESHOLD": "阈值比例 (0.25=激进, 0.5=推荐, 0.75=保守)",
            }
        },
    }

    # 记录已处理的key
    processed_keys = set()

    def format_value(v):
        """格式化值"""
        if isinstance(v, bool):
            return str(v).lower()
        elif isinstance(v, str):
            if v in [".inf", "inf"] or (isinstance(v, float) and v == float('inf')):
                return ".inf"
            return f'"{v}"' if ' ' in v or ':' in v else v
        elif isinstance(v, list):
            return yaml.dump(v, default_flow_style=True, allow_unicode=True).strip()
        elif isinstance(v, dict):
            return None  # 字典单独处理
        else:
            return str(v)

    # 按分类输出
    for category_name, category_info in categories.items():
        category_keys = category_info["keys"]
        category_comments = category_info["comments"]

        # 检查该分类是否有值
        has_values = any(key in content for key in category_keys)
        if not has_values:
            continue

        lines.append(f"\n# {category_name}")

        for key in category_keys:
            if key in content:
                value = content[key]
                formatted_value = format_value(value)
                if formatted_value is not None:
                    comment = category_comments.get(key, "")
                    if comment:
                        lines.append(f"{key}: {formatted_value}  # {comment}")
                    else:
                        lines.append(f"{key}: {formatted_value}")
                    processed_keys.add(key)

    # 处理特殊的嵌套配置
    if "CHANNEL_SPEC" in content:
        lines.append("\n# 通道规格")
        lines.append("CHANNEL_SPEC:")
        for k, v in content["CHANNEL_SPEC"].items():
            lines.append(f"  {k}: {v}")
        processed_keys.add("CHANNEL_SPEC")

    if "IN_ORDER_PACKET_CATEGORIES" in content:
        lines.append("\n# 需要保序的包类型")
        lines.append("IN_ORDER_PACKET_CATEGORIES:")
        for item in content["IN_ORDER_PACKET_CATEGORIES"]:
            lines.append(f'  - "{item}"')
        processed_keys.add("IN_ORDER_PACKET_CATEGORIES")

    if "IN_ORDER_EJECTION_PAIRS" in content:
        lines.append(f"IN_ORDER_EJECTION_PAIRS: {format_value(content['IN_ORDER_EJECTION_PAIRS'])}")
        processed_keys.add("IN_ORDER_EJECTION_PAIRS")

    if "arbitration" in content:
        lines.append("\n# 仲裁器配置")
        lines.append("arbitration:")
        arb = content["arbitration"]
        if "default" in arb:
            lines.append("  default:")
            arb_type = arb["default"].get("type", "round_robin")
            lines.append(f"    type: {format_value(arb_type)}")
            # 只有 islip 类型需要额外参数
            if arb_type == "islip":
                if "iterations" in arb["default"]:
                    lines.append(f"    iterations: {arb['default']['iterations']}")
                if "weight_strategy" in arb["default"]:
                    lines.append(f"    weight_strategy: {format_value(arb['default']['weight_strategy'])}")
        processed_keys.add("arbitration")

    # 处理未分类的配置
    remaining = {k: v for k, v in content.items() if k not in processed_keys}
    if remaining:
        lines.append("\n# 其他配置")
        for key, value in remaining.items():
            formatted_value = format_value(value)
            if formatted_value is not None:
                lines.append(f"{key}: {formatted_value}")
            elif isinstance(value, dict):
                lines.append(f"{key}:")
                dumped = yaml.dump(value, default_flow_style=False, allow_unicode=True)
                for line in dumped.strip().split('\n'):
                    lines.append(f"  {line}")

    return '\n'.join(lines) + '\n'


@router.post("/config/{config_path:path}")
async def save_config_content(config_path: str, content: dict):
    """
    保存配置文件内容（带分类和注释）
    """
    config_file = TOPOLOGIES_DIR / config_path
    if not config_file.exists():
        raise HTTPException(status_code=404, detail=f"配置文件不存在: {config_path}")

    try:
        formatted_content = _format_config_with_comments(content)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(formatted_content)
        return {"success": True, "message": f"配置文件已保存: {config_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存配置文件失败: {str(e)}")


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
