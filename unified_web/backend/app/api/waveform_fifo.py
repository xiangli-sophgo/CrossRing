"""
FIFO端口波形数据API - 基于position_timestamps重建FIFO波形
"""

import json
import io
from typing import List, Dict, Any, Tuple
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import pandas as pd

from src.database import DatabaseManager
from app.core.logger import get_logger

logger = get_logger("waveform_fifo")

router = APIRouter()


# ==================== Pydantic模型 ====================

class FIFOEvent(BaseModel):
    """FIFO事件"""
    enter_ns: float
    leave_ns: float
    flit_id: str


class FIFOSignal(BaseModel):
    """FIFO信号"""
    name: str  # "Node_5.IQ_TR"
    node_id: int
    fifo_type: str
    events: List[Dict[str, Any]]


class FIFOWaveformResponse(BaseModel):
    """FIFO波形响应"""
    time_range: Dict[str, float]
    signals: List[FIFOSignal]
    available_fifos: List[str]


# ==================== 辅助函数 ====================

def extract_fifo_events_from_timestamps(
    position_timestamps_json: str,
    source_node: int,
    dest_node: int
) -> List[Tuple[str, int, float, float]]:
    """从 position_timestamps JSON 提取 FIFO 事件

    Args:
        position_timestamps_json: JSON字符串格式的位置时间戳
        source_node: 源节点ID
        dest_node: 目标节点ID

    Returns:
        List[(fifo_type, node_id, enter_ns, leave_ns)]
    """
    if not position_timestamps_json or position_timestamps_json == "{}":
        return []

    try:
        # 解析 JSON
        timestamps = json.loads(position_timestamps_json)
    except (json.JSONDecodeError, TypeError):
        return []

    events = []

    # 已知的 FIFO 位置模式
    # IQ: IQ_TR, IQ_TL, IQ_TU, IQ_TD, IQ_CH
    # RB: RB_TR, RB_TL, RB_TU, RB_TD, RB_EQ
    # EQ: EQ_TU, EQ_TD, EQ_CH

    # IQ 事件（在源节点）
    iq_positions = ["IQ_TR", "IQ_TL", "IQ_TU", "IQ_TD", "IQ_CH"]
    for iq_pos in iq_positions:
        if iq_pos in timestamps:
            enter_ns = timestamps.get("L2H", timestamps.get("IP_inject", -1))
            leave_ns = timestamps.get("Link", -1)

            # 如果没有 Link 时间戳，尝试使用后续位置
            if leave_ns < 0 or leave_ns <= enter_ns:
                # 检查是否直接到 RB 或 EQ
                for next_pos in ["RB_TR", "RB_TL", "RB_TU", "RB_TD", "EQ_TU", "EQ_TD", "EQ_CH"]:
                    if next_pos in timestamps and timestamps[next_pos] > enter_ns:
                        leave_ns = timestamps[next_pos]
                        break

            if enter_ns >= 0 and leave_ns >= 0 and leave_ns > enter_ns:
                events.append((iq_pos, source_node, enter_ns, leave_ns))

    # RB 事件（在中间节点，暂时简化处理）
    rb_positions = ["RB_TR", "RB_TL", "RB_TU", "RB_TD", "RB_EQ"]
    for rb_pos in rb_positions:
        if rb_pos in timestamps:
            enter_ns = timestamps[rb_pos]
            # 离开时间：下一个位置（Link 或 EQ）
            leave_ns = -1
            for next_pos in ["Link", "EQ_TU", "EQ_TD", "EQ_CH"]:
                if next_pos in timestamps and timestamps[next_pos] > enter_ns:
                    leave_ns = timestamps[next_pos]
                    break

            if enter_ns >= 0 and leave_ns >= 0:
                # TODO: 推导中间节点ID（暂时跳过）
                # events.append((rb_pos, middle_node_id, enter_ns, leave_ns))
                pass

    # EQ 事件（在目标节点）
    eq_positions = ["EQ_TU", "EQ_TD", "EQ_CH"]
    for eq_pos in eq_positions:
        if eq_pos in timestamps:
            enter_ns = timestamps[eq_pos]
            leave_ns = timestamps.get("IP_eject", -1)

            if enter_ns >= 0 and leave_ns >= 0 and leave_ns > enter_ns:
                events.append((eq_pos, dest_node, enter_ns, leave_ns))

    return events


def build_fifo_waveform_for_node(
    flits_df: pd.DataFrame,
    requests_df: pd.DataFrame,
    node_id: int,
    fifo_types: List[str],
    flit_types_filter: List[str] = None
) -> Dict[str, List[Dict]]:
    """为指定节点构建 FIFO 波形数据（基于 position_timestamps）

    Args:
        flits_df: flits DataFrame
        requests_df: requests DataFrame
        node_id: 节点ID
        fifo_types: 要查询的FIFO类型列表
        flit_types_filter: Flit类型过滤列表（如 ["req", "rsp", "data"]），None表示不过滤

    Returns:
        {
            "IQ_TR": [{"enter_ns": 10.5, "leave_ns": 12.3, "flit_id": "123.req.0", "flit_type": "req"}, ...],
            "EQ_TU": [...],
            ...
        }
    """
    # 合并 flits 和 requests 数据
    merged = flits_df.merge(
        requests_df[["packet_id", "source_node", "dest_node", "source_type", "dest_type"]],
        on="packet_id",
        how="left"
    )

    # 初始化结果
    waveform_data = {fifo_type: [] for fifo_type in fifo_types}

    # 遍历所有 flit
    for _, row in merged.iterrows():
        # 获取flit类型
        flit_type = row.get("flit_type", "")

        # flit类型过滤
        if flit_types_filter and flit_type not in flit_types_filter:
            continue

        # 从 position_timestamps 提取事件
        events = extract_fifo_events_from_timestamps(
            row.get("position_timestamps", "{}"),
            row["source_node"],
            row["dest_node"]
        )

        # 过滤：只保留目标节点的目标 FIFO 类型
        for fifo_type_name, event_node_id, enter_ns, leave_ns in events:
            if event_node_id == node_id and fifo_type_name in fifo_types:
                waveform_data[fifo_type_name].append({
                    "enter_ns": enter_ns,
                    "leave_ns": leave_ns,
                    "flit_id": f"{row['packet_id']}.{flit_type}.{row.get('flit_id', 0)}",
                    "flit_type": flit_type
                })

    # 按时间排序
    for fifo_type in waveform_data:
        waveform_data[fifo_type].sort(key=lambda x: x["enter_ns"])

    return waveform_data


def _ip_type_to_channel(ip_type: str) -> str:
    """将IP类型转换为通道名（如 gdma_0 -> G0）"""
    if not ip_type:
        return ""
    parts = ip_type.split("_")
    if len(parts) >= 2:
        type_char = parts[0][0].upper()
        num = parts[1]
        return f"{type_char}{num}"
    return ip_type[0].upper() + "0"


# ==================== API端点 ====================

def _get_result_type(experiment_id: int) -> str:
    """获取实验的结果类型"""
    # 简化实现：根据ID范围判断，或从数据库查询
    # TODO: 从数据库查询实验配置获取类型
    return "kcin"  # 默认返回 kcin


@router.get("/experiments/{experiment_id}/results/{result_id}/fifo-waveform")
async def get_fifo_waveform(
    experiment_id: int,
    result_id: int,
    node_id: int = Query(..., description="节点ID"),
    fifo_types: str = Query(..., description="FIFO类型列表(逗号分隔)"),
    flit_types_filter: str = Query(None, description="Flit类型过滤(逗号分隔，如req,rsp,data)"),
    time_start: float = Query(None, description="起始时间(ns)"),
    time_end: float = Query(None, description="结束时间(ns)"),
) -> FIFOWaveformResponse:
    """从 flits.parquet 的 position_timestamps 重建指定节点的 FIFO 波形数据"""
    result_type = _get_result_type(experiment_id)
    db = DatabaseManager()

    # 加载 flits.parquet 和 requests.parquet
    flits_file = db.get_result_file_by_name(result_id, result_type, "flits.parquet")
    requests_file = db.get_result_file_by_name(result_id, result_type, "requests.parquet")

    if not flits_file or not requests_file:
        raise HTTPException(status_code=404, detail="未找到波形数据文件")

    # 读取数据
    flits_df = pd.read_parquet(io.BytesIO(flits_file.file_content))
    requests_df = pd.read_parquet(io.BytesIO(requests_file.file_content))

    # 检查是否包含 position_timestamps 字段
    if "position_timestamps" not in flits_df.columns:
        raise HTTPException(
            status_code=400,
            detail="flits.parquet 缺少 position_timestamps 字段，请重新运行仿真"
        )

    # 解析请求的 FIFO 类型
    requested_fifos = [f.strip() for f in fifo_types.split(",")]

    # 解析flit类型过滤列表
    flit_types_list = None
    if flit_types_filter:
        flit_types_list = [f.strip() for f in flit_types_filter.split(",")]

    # 重建 FIFO 波形
    waveform_data = build_fifo_waveform_for_node(
        flits_df, requests_df, node_id, requested_fifos, flit_types_list
    )

    # 构建信号列表
    signals = []
    all_times = []
    for fifo_type, events in waveform_data.items():
        if events:  # 只添加有事件的 FIFO
            # 时间范围过滤
            filtered_events = events
            if time_start is not None:
                filtered_events = [e for e in filtered_events if e["leave_ns"] >= time_start]
            if time_end is not None:
                filtered_events = [e for e in filtered_events if e["enter_ns"] <= time_end]

            if filtered_events:
                signals.append(FIFOSignal(
                    name=f"Node_{node_id}.{fifo_type}",
                    node_id=node_id,
                    fifo_type=fifo_type,
                    events=filtered_events
                ))
                all_times.extend([e["enter_ns"] for e in filtered_events])
                all_times.extend([e["leave_ns"] for e in filtered_events])

    # 计算时间范围
    time_range = {
        "start_ns": min(all_times) if all_times else 0,
        "end_ns": max(all_times) if all_times else 0
    }

    # 可用 FIFO 列表
    available_fifos = list(waveform_data.keys())

    return FIFOWaveformResponse(
        time_range=time_range,
        signals=signals,
        available_fifos=available_fifos
    )


@router.get("/experiments/{experiment_id}/results/{result_id}/fifo-waveform/available")
async def get_available_fifos(
    experiment_id: int,
    result_id: int,
    node_id: int = Query(..., description="节点ID"),
) -> Dict[str, List[str]]:
    """获取指定节点可用的 FIFO 列表"""
    # 返回所有可能的 FIFO 类型
    all_fifos = [
        "IQ_TR", "IQ_TL", "IQ_TU", "IQ_TD", "IQ_EQ", "IQ_CH",
        "RB_TR", "RB_TL", "RB_TU", "RB_TD", "RB_EQ",
        "EQ_TU", "EQ_TD", "EQ_CH"
    ]
    return {"fifos": all_fifos}
