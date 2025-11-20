from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from typing import List, Dict, Optional
from pathlib import Path
import json
import io
import zipfile
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

# 输出目录 - 指向项目根目录的traffic文件夹
# Path: web/backend/app/api/traffic_generate.py -> 需要5个parent到达CrossRing根目录
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent.parent / "traffic"
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

        # 检查是新格式还是旧格式
        # 新格式: "节点X-IP类型"
        # 旧格式: "IP类型"
        if src_ip in ip_label_to_node_id:
            # 新格式: 单个节点-IP对
            src_node_id = ip_label_to_node_id[src_ip]
            src_ip_type = src_ip.split('-')[1]  # 提取IP类型
            src_map = {src_ip_type: [src_node_id]}
        elif src_ip in ip_type_to_node_ids:
            # 旧格式: IP类型
            src_map = {src_ip: ip_type_to_node_ids[src_ip]}
        else:
            raise ValueError(f"源IP {src_ip} 未挂载到拓扑")

        if dst_ip in ip_label_to_node_id:
            # 新格式: 单个节点-IP对
            dst_node_id = ip_label_to_node_id[dst_ip]
            dst_ip_type = dst_ip.split('-')[1]  # 提取IP类型
            dst_map = {dst_ip_type: [dst_node_id]}
        elif dst_ip in ip_type_to_node_ids:
            # 旧格式: IP类型
            dst_map = {dst_ip: ip_type_to_node_ids[dst_ip]}
        else:
            raise ValueError(f"目标IP {dst_ip} 未挂载到拓扑")

        engine_config = {
            "src_map": src_map,
            "dst_map": dst_map,
            "speed": config.speed_gbps,
            "burst": config.burst_length,
            "req_type": config.request_type,
            "end_time": config.end_time_ns
        }

        # D2D模式需要额外的die信息
        if mode == "d2d" and hasattr(config, 'source_die') and hasattr(config, 'target_die'):
            engine_config["source_die"] = config.source_die
            engine_config["target_die"] = config.target_die

        engine_configs.append(engine_config)

    return engine_configs


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

    try:
        # 调用流量生成引擎
        if request.mode == "d2d":
            file_path, df = generate_d2d_traffic_from_configs(
                configs=engine_configs,
                end_time=None,
                output_file=str(output_file),
                random_seed=request.random_seed,
                return_dataframe=True
            )
        else:
            file_path, df = generate_traffic_from_configs(
                configs=engine_configs,
                end_time=None,
                output_file=str(output_file),
                random_seed=request.random_seed,
                return_dataframe=True
            )

        total_lines = len(df) if df is not None else 0
        file_size = Path(file_path).stat().st_size if file_path else 0

        # 如果需要按源IP分割
        if request.split_by_source:
            split_output_dir = OUTPUT_DIR / f"split_{request.topology}_{request.mode}_{timestamp}"
            split_output_dir.mkdir(parents=True, exist_ok=True)

            if request.mode == "d2d":
                split_files = split_d2d_traffic_by_source(
                    input_file=file_path,
                    output_dir=str(split_output_dir)
                )
            else:
                split_files = split_traffic_by_source(
                    input_file=file_path,
                    output_dir=str(split_output_dir)
                )

            # 创建zip文件包含所有分割文件
            zip_path = OUTPUT_DIR / f"traffic_{request.topology}_{request.mode}_{timestamp}_split.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for split_file in split_files:
                    zipf.write(split_file, Path(split_file).name)

            file_path = str(zip_path)
            file_size = zip_path.stat().st_size

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
        message=f"成功生成流量文件，共 {total_lines} 行",
        file_path=file_path,
        total_lines=total_lines,
        file_size=file_size,
        generation_time_ms=generation_time_ms
    )


@router.get("/download/{filename}")
async def download_traffic_file(filename: str):
    """
    下载生成的流量文件
    """
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

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
