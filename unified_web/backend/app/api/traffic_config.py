from fastapi import APIRouter, HTTPException
from typing import Dict, List
import json
import re
from pathlib import Path
from app.config import TRAFFIC_CONFIGS_DIR
import uuid
from datetime import datetime

from app.models.traffic_config import (
    TrafficConfig,
    TrafficConfigCreate,
    TrafficConfigListResponse,
    TrafficConfigResponse,
    BatchTrafficConfigCreate,
    DCINTrafficConfig,
    DCINTrafficConfigCreate
)

router = APIRouter(prefix="/api/traffic/config", tags=["流量配置"])

# 内存存储
# 格式: {topology: {mode: {config_id: TrafficConfig}}}
traffic_configs: Dict[str, Dict[str, Dict[str, TrafficConfig]]] = {}

# 配置文件路径
CONFIG_DIR = TRAFFIC_CONFIGS_DIR
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _validate_filename(filename: str) -> str:
    """
    验证并清理文件名，防止路径遍历攻击

    Args:
        filename: 用户提供的文件名

    Returns:
        清理后的安全文件名

    Raises:
        HTTPException: 文件名无效
    """
    # 移除路径分隔符
    clean_name = filename.replace("/", "").replace("\\", "").replace("..", "")
    # 只允许安全字符
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', clean_name):
        raise HTTPException(status_code=400, detail="文件名只能包含字母、数字、下划线、连字符和点号")
    # 确保有 .json 后缀
    if not clean_name.endswith(".json"):
        clean_name += ".json"
    return clean_name


def _validate_path(base_dir: Path, user_path: str) -> Path:
    """
    验证用户路径在允许的目录范围内

    Args:
        base_dir: 基础目录
        user_path: 用户提供的相对路径

    Returns:
        验证后的完整路径

    Raises:
        HTTPException: 路径无效或超出允许范围
    """
    try:
        full_path = (base_dir / user_path).resolve()
        full_path.relative_to(base_dir.resolve())
        return full_path
    except (ValueError, OSError):
        raise HTTPException(status_code=400, detail="无效的文件路径")


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

    支持 KCIN 和 DCIN 模式
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
        created_at=datetime.now(),
        source_die=request.source_die,
        target_die=request.target_die
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

    创建一个配置存储IP列表，而不是展开所有组合
    DCIN模式：支持多个DIE对（使用die_pairs字段）
    """
    _ensure_topology_mode(request.topology, request.mode)

    config_id = str(uuid.uuid4())

    # 处理DCIN多DIE对支持
    die_pairs_value = None
    source_die_value = None
    target_die_value = None

    if request.mode == "dcin":
        # 优先使用die_pairs（新格式）
        if request.die_pairs:
            die_pairs_value = request.die_pairs
        # 回退到旧格式（向后兼容）
        elif request.source_die is not None and request.target_die is not None:
            die_pairs_value = [[request.source_die, request.target_die]]
            source_die_value = request.source_die
            target_die_value = request.target_die

    # 创建一个配置存储IP列表
    config = TrafficConfig(
        id=config_id,
        topology=request.topology,
        mode=request.mode,
        source_ip=request.source_ips,  # 存储列表
        target_ip=request.target_ips,  # 存储列表
        speed_gbps=request.speed_gbps,
        burst_length=request.burst_length,
        request_type=request.request_type,
        end_time_ns=request.end_time_ns,
        created_at=datetime.now(),
        source_die=source_die_value,
        target_die=target_die_value,
        die_pairs=die_pairs_value
    )

    traffic_configs[request.topology][request.mode][config_id] = config

    # 保存到文件
    _save_configs(request.topology, request.mode, traffic_configs[request.topology][request.mode])

    # 计算实际组合数
    combo_count = len(request.source_ips) * len(request.target_ips)
    if die_pairs_value:
        combo_count *= len(die_pairs_value)

    message_text = f"成功创建流量配置（{len(request.source_ips)}源 × {len(request.target_ips)}目标"
    if die_pairs_value:
        message_text += f" × {len(die_pairs_value)}DIE对"
    message_text += f" = {combo_count}个组合）"

    return TrafficConfigResponse(
        success=True,
        message=message_text,
        config=config
    )


@router.get("/files/list")
async def list_config_files():
    """
    列出所有可用的流量配置文件
    """
    files = []
    if CONFIG_DIR.exists():
        for file_path in CONFIG_DIR.glob("*.json"):
            stat = file_path.stat()
            # 解析文件名获取拓扑和模式信息
            filename = file_path.stem  # 不包含.json后缀
            parts = filename.split('_')
            topology = parts[0] if len(parts) > 0 else "unknown"
            mode = parts[1] if len(parts) > 1 else "unknown"

            files.append({
                "filename": file_path.name,
                "path": str(file_path),
                "topology": topology,
                "mode": mode,
                "size": stat.st_size,
                "modified": stat.st_mtime
            })

    # 按修改时间倒序排列
    files.sort(key=lambda x: x["modified"], reverse=True)

    return {
        "files": files,
        "total": len(files),
        "directory": str(CONFIG_DIR)
    }


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

    # 处理 die_pairs 字段（优先使用请求中的值，否则保留旧值）
    die_pairs_value = request.die_pairs if request.die_pairs is not None else old_config.die_pairs
    source_die_value = request.source_die if request.source_die is not None else old_config.source_die
    target_die_value = request.target_die if request.target_die is not None else old_config.target_die

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
        created_at=old_config.created_at,
        source_die=source_die_value,
        target_die=target_die_value,
        die_pairs=die_pairs_value
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


@router.post("/{topology}/{mode}/load")
async def load_configs_from_file(topology: str, mode: str, request: dict):
    """
    从指定文件加载流量配置

    支持两种模式：
    - replace: 替换当前配置
    - append: 添加到当前配置
    """
    filename = request.get("filename", "").strip()
    load_mode = request.get("mode", "replace")  # replace 或 append

    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    # 验证文件名安全性
    filename = _validate_filename(filename)
    file_path = _validate_path(CONFIG_DIR, filename)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"文件 {filename} 不存在")

    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    _ensure_topology_mode(topology, mode)

    # 转换为内存格式
    loaded_configs = {k: TrafficConfig(**v) for k, v in data.items()}

    if load_mode == "replace":
        # 替换模式：清空当前配置
        traffic_configs[topology][mode] = loaded_configs
        message = f"成功从 {filename} 替换加载 {len(loaded_configs)} 个流量配置"
    else:
        # 添加模式：合并配置
        traffic_configs[topology][mode].update(loaded_configs)
        message = f"成功从 {filename} 添加 {len(loaded_configs)} 个流量配置"

    # 自动保存到当前拓扑模式的配置文件
    _save_configs(topology, mode, traffic_configs[topology][mode])

    return {
        "success": True,
        "message": message,
        "filename": filename,
        "count": len(loaded_configs),
        "total": len(traffic_configs[topology][mode])
    }


@router.post("/{topology}/{mode}/save")
async def save_configs_to_file(topology: str, mode: str, request: dict):
    """
    保存当前流量配置到指定文件名
    """
    filename = request.get("filename", "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    # 验证文件名安全性
    filename = _validate_filename(filename)

    _ensure_topology_mode(topology, mode)

    # 保存到指定文件名
    config_path = _validate_path(CONFIG_DIR, filename)
    configs = traffic_configs[topology][mode]
    data = {k: v.model_dump(mode='json') for k, v in configs.items()}

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    return {
        "success": True,
        "message": f"配置已保存到 {filename}",
        "filename": filename,
        "path": str(config_path),
        "count": len(configs)
    }
