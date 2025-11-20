from fastapi import APIRouter, HTTPException
from typing import Dict, List
import json
from pathlib import Path

from app.models.ip_mount import (
    IPMountRequest,
    IPMount,
    IPMountResponse,
    IPMountListResponse,
    BatchMountRequest
)

router = APIRouter(prefix="/api/ip-mount", tags=["IP挂载管理"])

# 内存存储（后续可以改为数据库或文件持久化）
# 格式: {topology: {node_id: [IPMount]}}  # 支持一个节点多个IP
ip_mounts: Dict[str, Dict[int, List[IPMount]]] = {}

# 配置文件路径 - 指向项目根目录的config/ip_mounts
CONFIG_DIR = Path(__file__).parent.parent.parent.parent.parent / "config" / "ip_mounts"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _get_mount_config_path(topology: str) -> Path:
    """获取拓扑对应的挂载配置文件路径"""
    return CONFIG_DIR / f"{topology}.json"


def _load_mounts(topology: str) -> Dict[int, List[IPMount]]:
    """从文件加载IP挂载配置"""
    config_path = _get_mount_config_path(topology)
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 兼容旧格式和新格式
            result = {}
            for k, v in data.items():
                node_id = int(k)
                if isinstance(v, list):
                    # 新格式：列表
                    result[node_id] = [IPMount(**mount) for mount in v]
                else:
                    # 旧格式：单个对象，转换为列表
                    result[node_id] = [IPMount(**v)]
            return result
    return {}


def _save_mounts(topology: str, mounts: Dict[int, List[IPMount]]):
    """保存IP挂载配置到文件"""
    config_path = _get_mount_config_path(topology)
    data = {str(k): [m.model_dump() for m in v] for k, v in mounts.items()}
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _get_node_position(topology: str, node_id: int) -> dict:
    """根据拓扑和节点ID计算节点位置"""
    rows, cols = map(int, topology.split('x'))
    if node_id >= rows * cols:
        raise ValueError(f"节点ID {node_id} 超出拓扑范围 {topology}")
    row = node_id // cols
    col = node_id % cols
    return {"row": row, "col": col}


@router.post("/", response_model=IPMountResponse)
async def mount_ip(request: IPMountRequest):
    """
    挂载IP到指定节点

    支持单个或批量挂载节点
    """
    # 加载现有配置
    if request.topology not in ip_mounts:
        ip_mounts[request.topology] = _load_mounts(request.topology)

    mounts = ip_mounts[request.topology]
    mounted = []

    # 挂载IP到节点（允许一个节点挂载多个IP）
    for node_id in request.node_ids:
        try:
            position = _get_node_position(request.topology, node_id)

            # 检查该节点是否已挂载相同的IP类型
            if node_id in mounts:
                existing_types = [m.ip_type for m in mounts[node_id]]
                if request.ip_type in existing_types:
                    raise HTTPException(
                        status_code=400,
                        detail=f"节点 {node_id} 已挂载 {request.ip_type}"
                    )

            mount = IPMount(
                node_id=node_id,
                ip_type=request.ip_type,
                topology=request.topology,
                position=position
            )

            # 添加到节点的IP列表
            if node_id not in mounts:
                mounts[node_id] = []
            mounts[node_id].append(mount)
            mounted.append(mount)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 保存配置
    _save_mounts(request.topology, mounts)

    return IPMountResponse(
        success=True,
        message=f"成功挂载 {len(mounted)} 个IP",
        mounted_ips=mounted
    )


@router.post("/batch", response_model=IPMountResponse)
async def batch_mount_ip(request: BatchMountRequest):
    """
    批量挂载IP

    支持范围语法: "0-3" 或 "1,3,5"
    自动为每个节点分配递增的IP编号
    """
    node_ids = request.get_node_ids()

    # 加载现有配置
    if request.topology not in ip_mounts:
        ip_mounts[request.topology] = _load_mounts(request.topology)

    mounts = ip_mounts[request.topology]
    mounted = []

    # 批量挂载，自动分配IP编号（允许一个节点挂载多个IP）
    for idx, node_id in enumerate(node_ids):
        try:
            position = _get_node_position(request.topology, node_id)
            ip_type = f"{request.ip_type_prefix}_{idx}"

            # 检查该节点是否已挂载相同的IP类型
            if node_id in mounts:
                existing_types = [m.ip_type for m in mounts[node_id]]
                if ip_type in existing_types:
                    raise HTTPException(
                        status_code=400,
                        detail=f"节点 {node_id} 已挂载 {ip_type}"
                    )

            mount = IPMount(
                node_id=node_id,
                ip_type=ip_type,
                topology=request.topology,
                position=position
            )

            # 添加到节点的IP列表
            if node_id not in mounts:
                mounts[node_id] = []
            mounts[node_id].append(mount)
            mounted.append(mount)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 保存配置
    _save_mounts(request.topology, mounts)

    return IPMountResponse(
        success=True,
        message=f"成功批量挂载 {len(mounted)} 个IP",
        mounted_ips=mounted
    )


@router.get("/{topology}", response_model=IPMountListResponse)
async def get_mounts(topology: str):
    """
    获取指定拓扑的所有IP挂载
    """
    # 加载配置
    if topology not in ip_mounts:
        ip_mounts[topology] = _load_mounts(topology)

    mounts = ip_mounts[topology]
    # 展平所有IP列表
    mount_list = []
    for node_mounts in mounts.values():
        mount_list.extend(node_mounts)

    return IPMountListResponse(
        topology=topology,
        mounts=mount_list,
        total=len(mount_list)
    )


@router.delete("/{topology}/nodes/{node_id}")
async def delete_mount(topology: str, node_id: int, ip_type: str = None):
    """
    删除指定节点的IP挂载

    - 如果指定ip_type，只删除该IP类型
    - 如果不指定，删除该节点的所有IP
    """
    # 加载配置
    if topology not in ip_mounts:
        ip_mounts[topology] = _load_mounts(topology)

    mounts = ip_mounts[topology]

    if node_id not in mounts:
        raise HTTPException(status_code=404, detail=f"节点 {node_id} 未挂载IP")

    if ip_type:
        # 删除指定IP类型
        original_count = len(mounts[node_id])
        mounts[node_id] = [m for m in mounts[node_id] if m.ip_type != ip_type]

        if len(mounts[node_id]) == original_count:
            raise HTTPException(status_code=404, detail=f"节点 {node_id} 未挂载 {ip_type}")

        # 如果节点没有IP了，删除该节点
        if len(mounts[node_id]) == 0:
            mounts.pop(node_id)

        deleted_info = {"node_id": node_id, "ip_type": ip_type}
    else:
        # 删除该节点的所有IP
        deleted_mounts = mounts.pop(node_id)
        deleted_info = {"node_id": node_id, "ips": [m.ip_type for m in deleted_mounts]}

    # 保存配置
    _save_mounts(topology, mounts)

    return {
        "success": True,
        "message": f"成功删除节点 {node_id} 的IP挂载",
        "deleted": deleted_info
    }


@router.delete("/{topology}")
async def clear_all_mounts(topology: str):
    """
    清空指定拓扑的所有IP挂载
    """
    # 加载配置
    if topology not in ip_mounts:
        ip_mounts[topology] = _load_mounts(topology)

    count = len(ip_mounts[topology])
    ip_mounts[topology] = {}

    # 保存配置（空配置）
    _save_mounts(topology, {})

    return {
        "success": True,
        "message": f"成功清空 {count} 个IP挂载",
        "cleared": count
    }
