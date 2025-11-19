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

# 输出目录
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "generated_traffic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class TrafficGenerateRequest(BaseModel):
    """流量生成请求"""
    topology: str = Field(..., description="拓扑类型")
    mode: str = Field(..., description="流量模式: noc 或 d2d")
    split_by_source: bool = Field(default=False, description="是否按源IP分割")
    random_seed: int = Field(default=42, description="随机种子")


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
    :param ip_mounts: IP挂载字典 {node_id: IPMount}
    :param traffic_configs: 流量配置字典 {config_id: TrafficConfig}
    :return: 引擎配置列表
    """
    # 创建 IP类型 -> 节点位置 的映射
    ip_to_positions = {}
    for node_id, mount in ip_mounts.items():
        ip_type = mount.ip_type
        position = mount.position
        coord = (position["row"], position["col"])

        if ip_type not in ip_to_positions:
            ip_to_positions[ip_type] = []
        ip_to_positions[ip_type].append(coord)

    engine_configs = []

    # 为每个流量配置生成引擎配置
    for config_id, config in traffic_configs.items():
        src_ip = config.source_ip
        dst_ip = config.target_ip

        # 检查IP是否已挂载
        if src_ip not in ip_to_positions:
            raise ValueError(f"源IP {src_ip} 未挂载到拓扑")
        if dst_ip not in ip_to_positions:
            raise ValueError(f"目标IP {dst_ip} 未挂载到拓扑")

        engine_config = {
            "src_map": {src_ip: ip_to_positions[src_ip]},
            "dst_map": {dst_ip: ip_to_positions[dst_ip]},
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"traffic_{request.topology}_{request.mode}_{timestamp}.csv"

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
