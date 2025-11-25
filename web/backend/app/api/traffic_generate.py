from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from typing import List, Dict, Optional
from pathlib import Path
from app.config import TRAFFIC_OUTPUT_DIR, TOPOLOGIES_DIR
import json
import io
import zipfile
import yaml
from datetime import datetime
from pydantic import BaseModel, Field

# 导入 CrossRing 流量生成模块
from src.traffic_process.traffic_gene.generation_engine import (
    generate_traffic_from_configs,
    generate_d2d_traffic_from_configs,
    split_traffic_by_source,
    split_d2d_traffic_by_source
)

from app.api.ip_mount import _load_mounts
from app.api.traffic_config import _load_configs

router = APIRouter(prefix="/api/traffic/generate", tags=["流量生成"])

# 输出目录
OUTPUT_DIR = TRAFFIC_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class TrafficGenerateRequest(BaseModel):
    """流量生成请求"""
    topology: str = Field(..., description="拓扑类型")
    mode: str = Field(..., description="流量模式: noc 或 d2d")
    split_by_source: bool = Field(default=False, description="是否按源IP分割")
    random_seed: int = Field(default=42, description="随机种子")
    filename: Optional[str] = Field(default=None, description="自定义文件名（不含扩展名）")


class TrafficGenerateResponse(BaseModel):
    """流量生成响应"""
    success: bool
    message: str
    file_path: Optional[str] = None
    total_lines: int = 0
    file_size: int = 0
    generation_time_ms: float = 0
    split_files: Optional[List[Dict]] = None


def _build_traffic_configs_for_engine(
    topology: str,
    mode: str,
    ip_mounts: Dict,
    traffic_configs: Dict
) -> List[Dict]:
    """
    将IP挂载和流量配置转换为流量生成引擎需要的格式

    :param topology: 拓扑类型
    :param mode: 模式 (noc/d2d)
    :param ip_mounts: IP挂载字典 {node_id: [IPMount]}
    :param traffic_configs: 流量配置字典 {config_id: TrafficConfig}
    :return: 引擎配置列表
    """
    # 创建 "节点X-IP类型" -> 节点ID 的映射
    # 同时创建 IP类型 -> 节点ID列表 的映射（用于向后兼容）
    ip_label_to_node_id = {}
    ip_type_to_node_ids = {}

    for node_id, mounts in ip_mounts.items():
        # 处理列表格式
        if isinstance(mounts, list):
            for mount in mounts:
                ip_type = mount.ip_type

                # 格式: "节点X-IP类型"
                ip_label = f"节点{node_id}-{ip_type}"
                ip_label_to_node_id[ip_label] = node_id

                # 同时保存IP类型映射（向后兼容）
                if ip_type not in ip_type_to_node_ids:
                    ip_type_to_node_ids[ip_type] = []
                ip_type_to_node_ids[ip_type].append(node_id)
        else:
            # 处理旧格式（单个对象）
            mount = mounts
            ip_type = mount.ip_type

            ip_label = f"节点{node_id}-{ip_type}"
            ip_label_to_node_id[ip_label] = node_id

            if ip_type not in ip_type_to_node_ids:
                ip_type_to_node_ids[ip_type] = []
            ip_type_to_node_ids[ip_type].append(node_id)

    engine_configs = []

    # 为每个流量配置生成引擎配置
    for config_id, config in traffic_configs.items():
        src_ip = config.source_ip
        dst_ip = config.target_ip

        # 处理source_ip（支持字符串或列表）
        src_ips = [src_ip] if isinstance(src_ip, str) else src_ip
        src_map = {}

        for ip in src_ips:
            if ip in ip_label_to_node_id:
                # 新格式: "节点X-IP类型"
                node_id = ip_label_to_node_id[ip]
                ip_type = ip.split('-')[1]
                if ip_type not in src_map:
                    src_map[ip_type] = []
                src_map[ip_type].append(node_id)
            elif ip in ip_type_to_node_ids:
                # 旧格式: "IP类型"
                if ip not in src_map:
                    src_map[ip] = []
                src_map[ip].extend(ip_type_to_node_ids[ip])
            else:
                raise ValueError(f"源IP {ip} 未挂载到拓扑。请检查IP挂载配置是否包含此IP。")

        # 处理target_ip（支持字符串或列表）
        dst_ips = [dst_ip] if isinstance(dst_ip, str) else dst_ip
        dst_map = {}

        for ip in dst_ips:
            if ip in ip_label_to_node_id:
                # 新格式: "节点X-IP类型"
                node_id = ip_label_to_node_id[ip]
                ip_type = ip.split('-')[1]
                if ip_type not in dst_map:
                    dst_map[ip_type] = []
                dst_map[ip_type].append(node_id)
            elif ip in ip_type_to_node_ids:
                # 旧格式: "IP类型"
                if ip not in dst_map:
                    dst_map[ip] = []
                dst_map[ip].extend(ip_type_to_node_ids[ip])
            else:
                raise ValueError(f"目标IP {ip} 未挂载到拓扑。请检查IP挂载配置是否包含此IP。")

        # D2D模式：需要为每个DIE对生成独立的引擎配置
        if mode == "d2d":
            # 优先使用die_pairs（新格式）
            if hasattr(config, 'die_pairs') and config.die_pairs:
                # 计算每个DIE对分配的带宽：总带宽 / DIE对数量
                num_die_pairs = len(config.die_pairs)
                speed_per_pair = config.speed_gbps / num_die_pairs

                # 为每个DIE对创建一个引擎配置
                for die_pair in config.die_pairs:
                    src_die, dst_die = die_pair
                    engine_config = {
                        "src_map": src_map,
                        "dst_map": dst_map,
                        "speed": speed_per_pair,
                        "burst": config.burst_length,
                        "req_type": config.request_type,
                        "end_time": config.end_time_ns,
                        "src_die": src_die,
                        "dst_die": dst_die
                    }
                    engine_configs.append(engine_config)
            # 回退到旧格式
            elif hasattr(config, 'source_die') and config.source_die is not None:
                engine_config = {
                    "src_map": src_map,
                    "dst_map": dst_map,
                    "speed": config.speed_gbps,
                    "burst": config.burst_length,
                    "req_type": config.request_type,
                    "end_time": config.end_time_ns,
                    "src_die": config.source_die,
                    "dst_die": config.target_die
                }
                engine_configs.append(engine_config)
            else:
                raise ValueError(f"D2D配置缺少DIE信息: {config_id}")
        else:
            # NoC模式：一个配置生成一个引擎配置
            engine_config = {
                "src_map": src_map,
                "dst_map": dst_map,
                "speed": config.speed_gbps,
                "burst": config.burst_length,
                "req_type": config.request_type,
                "end_time": config.end_time_ns
            }
            engine_configs.append(engine_config)

    return engine_configs


def _load_d2d_config_for_topology(topology: str) -> Optional[Dict]:
    """
    为指定拓扑加载D2D配置

    Args:
        topology: 拓扑类型（如 "5x4"）

    Returns:
        D2D配置字典，包含 num_dies 和 d2d_connections，如果未找到则返回None
    """
    # 尝试加载拓扑特定的D2D配置
    config_file = TOPOLOGIES_DIR / f"d2d_{topology}_config.yaml"
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if config.get('D2D_ENABLED', False):
                return {
                    "num_dies": config.get('NUM_DIES', 2),
                    "d2d_connections": config.get('D2D_CONNECTIONS', [])
                }

    # 回退到默认配置
    default_config = TOPOLOGIES_DIR / "d2d_2die_config.yaml"
    if default_config.exists():
        with open(default_config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if config.get('D2D_ENABLED', False):
                return {
                    "num_dies": config.get('NUM_DIES', 2),
                    "d2d_connections": config.get('D2D_CONNECTIONS', [])
                }

    return None


@router.post("/", response_model=TrafficGenerateResponse)
async def generate_traffic(request: TrafficGenerateRequest):
    """
    生成流量文件

    根据IP挂载和流量配置生成流量数据，可选按源IP分割
    """
    start_time = datetime.now()

    # 加载IP挂载
    ip_mounts = _load_mounts(request.topology)
    if not ip_mounts:
        raise HTTPException(
            status_code=400,
            detail=f"拓扑 {request.topology} 没有IP挂载配置"
        )

    # 加载流量配置
    traffic_configs = _load_configs(request.topology, request.mode)
    if not traffic_configs:
        raise HTTPException(
            status_code=400,
            detail=f"拓扑 {request.topology} 模式 {request.mode} 没有流量配置"
        )

    # 转换为引擎配置格式
    try:
        engine_configs = _build_traffic_configs_for_engine(
            request.topology,
            request.mode,
            ip_mounts,
            traffic_configs
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 生成输出文件路径
    if request.filename:
        # 使用用户指定的文件名
        filename = request.filename.strip()
        # 确保文件名安全
        filename = filename.replace("/", "_").replace("\\", "_")
        if not filename.endswith('.txt'):
            filename += '.txt'
        output_file = OUTPUT_DIR / filename
    else:
        # 使用默认时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"traffic_{timestamp}.txt"

    # 构建元数据所需参数
    node_ips = {}
    for node_id, mounts in ip_mounts.items():
        if isinstance(mounts, list):
            node_ips[node_id] = [mount.ip_type for mount in mounts]
        else:
            node_ips[node_id] = [mounts.ip_type]

    topo_type = request.topology
    routing_type = "XY"  # 默认XY路由

    # D2D模式：加载D2D配置
    d2d_config = None
    if request.mode == "d2d":
        d2d_config = _load_d2d_config_for_topology(request.topology)

    try:
        # 调用流量生成引擎
        if request.mode == "d2d":
            file_path, df = generate_d2d_traffic_from_configs(
                configs=engine_configs,
                end_time=None,
                output_file=str(output_file),
                random_seed=request.random_seed,
                return_dataframe=True,
                topo_type=topo_type,
                routing_type=routing_type,
                node_ips=node_ips,
                d2d_config=d2d_config
            )
        else:
            file_path, df = generate_traffic_from_configs(
                configs=engine_configs,
                end_time=None,
                output_file=str(output_file),
                random_seed=request.random_seed,
                return_dataframe=True,
                topo_type=topo_type,
                routing_type=routing_type,
                node_ips=node_ips
            )

        total_lines = len(df) if df is not None else 0
        file_size = Path(file_path).stat().st_size if file_path else 0

        # 如果需要按源IP分割
        split_result = None
        if request.split_by_source:
            split_output_dir = OUTPUT_DIR / request.filename
            split_output_dir.mkdir(parents=True, exist_ok=True)

            if request.mode == "d2d":
                split_result = split_d2d_traffic_by_source(
                    input_file=file_path,
                    output_dir=str(split_output_dir)
                )
            else:
                split_result = split_traffic_by_source(
                    input_file=file_path,
                    output_dir=str(split_output_dir)
                )

            file_path = str(split_output_dir)
            file_size = 0

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"流量生成失败: {str(e)}"
        )

    # 计算耗时
    end_time = datetime.now()
    generation_time_ms = (end_time - start_time).total_seconds() * 1000

    return TrafficGenerateResponse(
        success=True,
        message=f"成功生成流量文件，共 {total_lines} 行" if not split_result else f"成功拆分流量文件为 {split_result['total_sources']} 个文件",
        file_path=file_path,
        total_lines=total_lines,
        file_size=file_size,
        generation_time_ms=generation_time_ms,
        split_files=split_result['files'] if split_result else None
    )


@router.get("/download/{filename:path}")
async def download_traffic_file(filename: str):
    """
    下载生成的流量文件（支持文件夹内文件）
    filename可以是：
    - 单个文件名
    - folder_name/master_xxx.txt (文件夹内文件)
    """
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="不能下载文件夹")

    # 确定文件类型
    if filename.endswith('.zip'):
        media_type = 'application/zip'
    else:
        media_type = 'text/csv'

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename
    )


@router.get("/list")
async def list_generated_files():
    """
    列出所有已生成的流量文件
    """
    files = []
    for file_path in OUTPUT_DIR.glob("traffic_*.csv"):
        stat = file_path.stat()
        files.append({
            "filename": file_path.name,
            "size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
        })

    # 也包含zip文件
    for file_path in OUTPUT_DIR.glob("traffic_*.zip"):
        stat = file_path.stat()
        files.append({
            "filename": file_path.name,
            "size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
        })

    # 按修改时间倒序
    files.sort(key=lambda x: x["modified_at"], reverse=True)

    return {
        "files": files,
        "total": len(files)
    }
