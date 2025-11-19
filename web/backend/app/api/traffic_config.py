from fastapi import APIRouter, HTTPException
from typing import Dict, List
import json
from pathlib import Path
import uuid
from datetime import datetime

from app.models.traffic_config import (
    TrafficConfig,
    TrafficConfigCreate,
    TrafficConfigListResponse,
    TrafficConfigResponse,
    BatchTrafficConfigCreate,
    D2DTrafficConfig,
    D2DTrafficConfigCreate
)

router = APIRouter(prefix="/api/traffic/config", tags=["流量配置"])

# 内存存储
# 格式: {topology: {mode: {config_id: TrafficConfig}}}
traffic_configs: Dict[str, Dict[str, Dict[str, TrafficConfig]]] = {}

# 配置文件路径
CONFIG_DIR = Path(__file__).parent.parent.parent / "data" / "traffic_configs"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _get_config_path(topology: str, mode: str) -> Path:
    """获取配置文件路径"""
    return CONFIG_DIR / f"{topology}_{mode}.json"


def _load_configs(topology: str, mode: str) -> Dict[str, TrafficConfig]:
    """从文件加载流量配置"""
    config_path = _get_config_path(topology, mode)
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {k: TrafficConfig(**v) for k, v in data.items()}
    return {}


def _save_configs(topology: str, mode: str, configs: Dict[str, TrafficConfig]):
    """保存流量配置到文件"""
    config_path = _get_config_path(topology, mode)
    data = {k: v.model_dump(mode='json') for k, v in configs.items()}
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _ensure_topology_mode(topology: str, mode: str):
    """确保拓扑和模式的存储结构存在"""
    if topology not in traffic_configs:
        traffic_configs[topology] = {}
    if mode not in traffic_configs[topology]:
        traffic_configs[topology][mode] = _load_configs(topology, mode)


@router.post("/", response_model=TrafficConfigResponse)
async def create_traffic_config(request: TrafficConfigCreate):
    """
    创建单个流量配置

    支持 NoC 和 D2D 模式
    """
    _ensure_topology_mode(request.topology, request.mode)

    # 生成唯一ID
    config_id = str(uuid.uuid4())

    # 创建配置对象
    config = TrafficConfig(
        id=config_id,
        topology=request.topology,
        mode=request.mode,
        source_ip=request.source_ip,
        target_ip=request.target_ip,
        speed_gbps=request.speed_gbps,
        burst_length=request.burst_length,
        request_type=request.request_type,
        end_time_ns=request.end_time_ns,
        created_at=datetime.now()
    )

    # 保存到内存
    traffic_configs[request.topology][request.mode][config_id] = config

    # 保存到文件
    _save_configs(request.topology, request.mode, traffic_configs[request.topology][request.mode])

    return TrafficConfigResponse(
        success=True,
        message="流量配置创建成功",
        config=config
    )


@router.post("/batch", response_model=TrafficConfigResponse)
async def create_batch_traffic_config(request: BatchTrafficConfigCreate):
    """
    批量创建流量配置

    为所有源IP和目标IP的组合创建配置
    """
    _ensure_topology_mode(request.topology, request.mode)

    created_configs = []

    # 为每个源IP和目标IP组合创建配置
    for source_ip in request.source_ips:
        for target_ip in request.target_ips:
            config_id = str(uuid.uuid4())

            config = TrafficConfig(
                id=config_id,
                topology=request.topology,
                mode=request.mode,
                source_ip=source_ip,
                target_ip=target_ip,
                speed_gbps=request.speed_gbps,
                burst_length=request.burst_length,
                request_type=request.request_type,
                end_time_ns=request.end_time_ns,
                created_at=datetime.now()
            )

            traffic_configs[request.topology][request.mode][config_id] = config
            created_configs.append(config)

    # 保存到文件
    _save_configs(request.topology, request.mode, traffic_configs[request.topology][request.mode])

    return TrafficConfigResponse(
        success=True,
        message=f"成功创建 {len(created_configs)} 个流量配置",
        config=None
    )


@router.get("/{topology}/{mode}", response_model=TrafficConfigListResponse)
async def get_traffic_configs(topology: str, mode: str):
    """
    获取指定拓扑和模式的所有流量配置
    """
    _ensure_topology_mode(topology, mode)

    configs = list(traffic_configs[topology][mode].values())

    return TrafficConfigListResponse(
        topology=topology,
        mode=mode,
        configs=configs,
        total=len(configs)
    )


@router.get("/{topology}/{mode}/{config_id}", response_model=TrafficConfig)
async def get_traffic_config(topology: str, mode: str, config_id: str):
    """
    获取指定流量配置详情
    """
    _ensure_topology_mode(topology, mode)

    if config_id not in traffic_configs[topology][mode]:
        raise HTTPException(status_code=404, detail=f"配置 {config_id} 不存在")

    return traffic_configs[topology][mode][config_id]


@router.put("/{topology}/{mode}/{config_id}", response_model=TrafficConfigResponse)
async def update_traffic_config(
    topology: str,
    mode: str,
    config_id: str,
    request: TrafficConfigCreate
):
    """
    更新流量配置
    """
    _ensure_topology_mode(topology, mode)

    if config_id not in traffic_configs[topology][mode]:
        raise HTTPException(status_code=404, detail=f"配置 {config_id} 不存在")

    # 保留原ID和创建时间，更新其他字段
    old_config = traffic_configs[topology][mode][config_id]

    updated_config = TrafficConfig(
        id=config_id,
        topology=request.topology,
        mode=request.mode,
        source_ip=request.source_ip,
        target_ip=request.target_ip,
        speed_gbps=request.speed_gbps,
        burst_length=request.burst_length,
        request_type=request.request_type,
        end_time_ns=request.end_time_ns,
        created_at=old_config.created_at
    )

    traffic_configs[topology][mode][config_id] = updated_config

    # 保存到文件
    _save_configs(topology, mode, traffic_configs[topology][mode])

    return TrafficConfigResponse(
        success=True,
        message="流量配置更新成功",
        config=updated_config
    )


@router.delete("/{topology}/{mode}/{config_id}")
async def delete_traffic_config(topology: str, mode: str, config_id: str):
    """
    删除流量配置
    """
    _ensure_topology_mode(topology, mode)

    if config_id not in traffic_configs[topology][mode]:
        raise HTTPException(status_code=404, detail=f"配置 {config_id} 不存在")

    deleted_config = traffic_configs[topology][mode].pop(config_id)

    # 保存到文件
    _save_configs(topology, mode, traffic_configs[topology][mode])

    return {
        "success": True,
        "message": "流量配置删除成功",
        "deleted": deleted_config
    }


@router.delete("/{topology}/{mode}")
async def clear_all_configs(topology: str, mode: str):
    """
    清空指定拓扑和模式的所有流量配置
    """
    _ensure_topology_mode(topology, mode)

    count = len(traffic_configs[topology][mode])
    traffic_configs[topology][mode] = {}

    # 保存到文件
    _save_configs(topology, mode, {})

    return {
        "success": True,
        "message": f"成功清空 {count} 个流量配置",
        "cleared": count
    }
