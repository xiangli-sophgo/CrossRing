"""
波形数据API - 用于NoC仿真波形可视化
"""

import io
from typing import List, Optional, Dict, Any, Tuple
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import pandas as pd

from src.database import ResultManager, DatabaseManager
from app.core.logger import get_logger

logger = get_logger("waveform")

router = APIRouter()

# 获取数据库管理器
db_manager = ResultManager()


# ==================== Pydantic模型 ====================

class WaveformEvent(BaseModel):
    """波形事件（一个阶段）"""
    stage: str          # 阶段名: "l2h", "iq_out", "h_ring", "rb", "v_ring", "eq", "eject"
    start_ns: float     # 开始时间(ns)
    end_ns: float       # 结束时间(ns)


class WaveformSignal(BaseModel):
    """波形信号（一个flit的传输轨迹）"""
    name: str           # 信号名: "Pkt_123.REQ", "Pkt_123.D0", "Pkt_123.RSP"
    packet_id: int      # 请求ID
    flit_type: str      # "req" | "data" | "rsp"
    flit_id: Optional[int] = None  # flit序号（data flit用）
    events: List[WaveformEvent]    # 各阶段事件


class WaveformResponse(BaseModel):
    """波形数据响应"""
    time_range: Dict[str, float]  # {start_ns, end_ns}
    signals: List[WaveformSignal]
    stages: List[str]   # 所有阶段名称列表


class PacketInfo(BaseModel):
    """请求信息"""
    packet_id: int
    req_type: str       # "read" | "write"
    source_node: int
    source_type: str
    dest_node: int
    dest_type: str
    start_time_ns: float
    end_time_ns: float
    latency_ns: float


class PacketListResponse(BaseModel):
    """请求列表响应"""
    packets: List[PacketInfo]
    total: int
    page: int
    page_size: int


# ==================== 辅助函数 ====================

def _load_parquet_from_db(result_id: int, result_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """从数据库加载Parquet文件内容

    Args:
        result_id: 结果ID
        result_type: 结果类型 ("kcin" 或 "dcin")

    Returns:
        (requests_df, flits_df) 元组

    Raises:
        HTTPException: 如果文件不存在
    """
    db = DatabaseManager()

    # 获取requests.parquet
    requests_file = db.get_result_file_by_name(result_id, result_type, "requests.parquet")
    if not requests_file or not requests_file.file_content:
        raise HTTPException(status_code=404, detail="未找到requests.parquet文件")

    # 获取flits.parquet
    flits_file = db.get_result_file_by_name(result_id, result_type, "flits.parquet")
    if not flits_file or not flits_file.file_content:
        raise HTTPException(status_code=404, detail="未找到flits.parquet文件")

    # 从内存读取parquet
    requests_df = pd.read_parquet(io.BytesIO(requests_file.file_content))
    flits_df = pd.read_parquet(io.BytesIO(flits_file.file_content))

    return requests_df, flits_df


def _get_result_type(experiment_id: int) -> str:
    """获取实验类型"""
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")
    return experiment.get("experiment_type", "kcin")


def _build_waveform_signal(packet_id: int, flit_row, flit_type: str, flit_id: Optional[int] = None) -> WaveformSignal:
    """构建单个flit的波形信号"""
    # 信号名
    if flit_type == "req":
        name = f"Pkt_{packet_id}.REQ"
    elif flit_type == "rsp":
        name = f"Pkt_{packet_id}.RSP"
    else:
        name = f"Pkt_{packet_id}.D{flit_id}"

    # 定义阶段顺序和对应的时间戳字段
    stage_fields = [
        ("l2h", "ip_inject_ns", "iq_out_ns"),       # IP注入 -> IQ_OUT
        ("iq_out", "iq_out_ns", "h_ring_ns"),       # IQ_OUT -> 横向环
        ("h_ring", "h_ring_ns", "rb_ns"),           # 横向环 -> Ring_Bridge
        ("rb", "rb_ns", "v_ring_ns"),               # Ring_Bridge -> 纵向环
        ("v_ring", "v_ring_ns", "eq_ns"),           # 纵向环 -> EQ
        ("eq", "eq_ns", "ip_eject_ns"),             # EQ -> IP接收
    ]

    events = []
    for stage_name, start_field, end_field in stage_fields:
        start_val = flit_row.get(start_field, -1)
        end_val = flit_row.get(end_field, -1)

        # 只添加有效的事件（时间戳 >= 0）
        if start_val >= 0 and end_val >= 0:
            events.append(WaveformEvent(
                stage=stage_name,
                start_ns=float(start_val),
                end_ns=float(end_val)
            ))

    return WaveformSignal(
        name=name,
        packet_id=packet_id,
        flit_type=flit_type,
        flit_id=flit_id,
        events=events
    )


# ==================== API端点 ====================

@router.get("/experiments/{experiment_id}/results/{result_id}/waveform")
async def get_waveform_data(
    experiment_id: int,
    result_id: int,
    packet_ids: Optional[str] = Query(None, description="逗号分隔的packet ID列表"),
    time_start: Optional[float] = Query(None, description="时间范围起点(ns)"),
    time_end: Optional[float] = Query(None, description="时间范围终点(ns)"),
    max_packets: int = Query(20, ge=1, le=100, description="最大请求数"),
) -> WaveformResponse:
    """
    获取波形数据

    - packet_ids: 指定的请求ID列表（逗号分隔）
    - time_start/time_end: 时间范围过滤
    - max_packets: 不指定packet_ids时，返回的最大请求数
    """
    # 获取实验类型
    result_type = _get_result_type(experiment_id)

    # 从数据库加载parquet数据
    try:
        requests_df, flits_df = _load_parquet_from_db(result_id, result_type)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"加载parquet数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"加载波形数据失败: {str(e)}")

    # 解析packet_ids参数
    selected_packet_ids = None
    if packet_ids:
        try:
            selected_packet_ids = [int(pid.strip()) for pid in packet_ids.split(",")]
        except ValueError:
            raise HTTPException(status_code=400, detail="packet_ids格式错误，应为逗号分隔的整数")

    # 过滤requests
    filtered_requests = requests_df

    if selected_packet_ids:
        filtered_requests = filtered_requests[filtered_requests["packet_id"].isin(selected_packet_ids)]
    else:
        # 按开始时间排序，取前max_packets个
        filtered_requests = filtered_requests.sort_values("start_time_ns").head(max_packets)

    # 时间范围过滤
    if time_start is not None:
        filtered_requests = filtered_requests[filtered_requests["end_time_ns"] >= time_start]
    if time_end is not None:
        filtered_requests = filtered_requests[filtered_requests["start_time_ns"] <= time_end]

    # 获取过滤后的packet_ids
    final_packet_ids = filtered_requests["packet_id"].tolist()

    # 过滤flits
    filtered_flits = flits_df[flits_df["packet_id"].isin(final_packet_ids)]

    # 构建波形信号
    signals = []
    for _, flit_row in filtered_flits.iterrows():
        packet_id = int(flit_row["packet_id"])
        flit_type = flit_row["flit_type"]
        flit_id = int(flit_row["flit_id"]) if flit_row["flit_id"] >= 0 else None

        signal = _build_waveform_signal(packet_id, flit_row, flit_type, flit_id)
        if signal.events:  # 只添加有事件的信号
            signals.append(signal)

    # 计算时间范围
    all_times = []
    for signal in signals:
        for event in signal.events:
            all_times.append(event.start_ns)
            all_times.append(event.end_ns)

    if all_times:
        time_range = {
            "start_ns": min(all_times),
            "end_ns": max(all_times)
        }
    else:
        time_range = {"start_ns": 0, "end_ns": 0}

    # 阶段列表
    stages = ["l2h", "iq_out", "h_ring", "rb", "v_ring", "eq"]

    return WaveformResponse(
        time_range=time_range,
        signals=signals,
        stages=stages
    )


@router.get("/experiments/{experiment_id}/results/{result_id}/packets")
async def list_packets(
    experiment_id: int,
    result_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    req_type: Optional[str] = Query(None, description="请求类型过滤: read/write"),
    sort_by: str = Query("start_time_ns", description="排序字段"),
    order: str = Query("asc", regex="^(asc|desc)$", description="排序方向"),
) -> PacketListResponse:
    """
    列出可选的请求列表

    - page/page_size: 分页参数
    - req_type: 过滤请求类型
    - sort_by: 排序字段
    - order: 排序方向
    """
    # 获取实验类型
    result_type = _get_result_type(experiment_id)

    # 从数据库加载parquet数据
    try:
        requests_df, _ = _load_parquet_from_db(result_id, result_type)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"加载parquet数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"加载波形数据失败: {str(e)}")

    # 类型过滤
    if req_type:
        requests_df = requests_df[requests_df["req_type"] == req_type]

    # 排序
    ascending = order == "asc"
    if sort_by in requests_df.columns:
        requests_df = requests_df.sort_values(sort_by, ascending=ascending)

    total = len(requests_df)

    # 分页
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_df = requests_df.iloc[start_idx:end_idx]

    # 构建响应
    packets = []
    for _, row in page_df.iterrows():
        start_time = float(row["start_time_ns"])
        end_time = float(row["end_time_ns"])
        packets.append(PacketInfo(
            packet_id=int(row["packet_id"]),
            req_type=row["req_type"],
            source_node=int(row["source_node"]),
            source_type=row["source_type"],
            dest_node=int(row["dest_node"]),
            dest_type=row["dest_type"],
            start_time_ns=start_time,
            end_time_ns=end_time,
            latency_ns=end_time - start_time
        ))

    return PacketListResponse(
        packets=packets,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/experiments/{experiment_id}/results/{result_id}/waveform/check")
async def check_waveform_data(
    experiment_id: int,
    result_id: int,
) -> Dict[str, Any]:
    """
    检查波形数据是否可用

    返回是否存在parquet文件及基本统计信息
    """
    # 获取实验类型
    try:
        result_type = _get_result_type(experiment_id)
    except HTTPException:
        return {
            "available": False,
            "message": "实验不存在",
        }

    try:
        requests_df, flits_df = _load_parquet_from_db(result_id, result_type)

        return {
            "available": True,
            "stats": {
                "total_packets": len(requests_df),
                "total_flits": len(flits_df),
                "read_packets": len(requests_df[requests_df["req_type"] == "read"]),
                "write_packets": len(requests_df[requests_df["req_type"] == "write"]),
                "time_range_ns": {
                    "start": float(requests_df["start_time_ns"].min()),
                    "end": float(requests_df["end_time_ns"].max())
                }
            }
        }
    except HTTPException as e:
        return {
            "available": False,
            "message": e.detail,
        }
    except Exception as e:
        return {
            "available": False,
            "message": f"加载波形数据失败: {str(e)}",
        }
