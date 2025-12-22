"""
FIFO端口波形数据API - 基于position_timestamps重建FIFO波形
"""

import json
from typing import List, Dict, Any, Tuple
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import pandas as pd

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
    # IP: IP_inject, L2H, H2L_H, H2L_L, IP_eject
    # IQ: IQ_TR, IQ_TL, IQ_TU, IQ_TD, IQ_CH
    # RB: RB_TR, RB_TL, RB_TU, RB_TD, RB_EQ
    # EQ: EQ_TU, EQ_TD, EQ_CH

    # ===== IP 发送端口事件（在源节点）=====
    # IP_TX: 从 IP_inject 到 L2H（或第一个 IQ）
    if "IP_inject" in timestamps:
        enter_ns = timestamps["IP_inject"]
        leave_ns = timestamps.get("L2H", -1)
        # 如果没有 L2H，尝试 IQ 位置
        if leave_ns < 0:
            for iq_pos in ["IQ_CH", "IQ_TR", "IQ_TL", "IQ_TU", "IQ_TD"]:
                if iq_pos in timestamps and timestamps[iq_pos] > enter_ns:
                    leave_ns = timestamps[iq_pos]
                    break
        if enter_ns >= 0 and leave_ns >= 0 and leave_ns > enter_ns:
            events.append(("IP_TX", source_node, enter_ns, leave_ns))

    # L2H: 从 L2H 到 IQ（发送端的 L2H 队列）
    if "L2H" in timestamps:
        enter_ns = timestamps["L2H"]
        leave_ns = -1
        for iq_pos in ["IQ_CH", "IQ_TR", "IQ_TL", "IQ_TU", "IQ_TD"]:
            if iq_pos in timestamps and timestamps[iq_pos] > enter_ns:
                leave_ns = timestamps[iq_pos]
                break
        if leave_ns < 0:
            leave_ns = timestamps.get("Link", -1)
        if enter_ns >= 0 and leave_ns >= 0 and leave_ns > enter_ns:
            events.append(("L2H", source_node, enter_ns, leave_ns))

    # ===== IQ 事件（在源节点）=====
    iq_positions = ["IQ_TR", "IQ_TL", "IQ_TU", "IQ_TD", "IQ_CH"]
    for iq_pos in iq_positions:
        if iq_pos in timestamps:
            enter_ns = timestamps[iq_pos]
            leave_ns = timestamps.get("Link", -1)

            # 如果没有 Link 时间戳，尝试使用后续位置
            if leave_ns < 0 or leave_ns <= enter_ns:
                for next_pos in ["RB_TR", "RB_TL", "RB_TU", "RB_TD", "EQ_TU", "EQ_TD", "EQ_CH"]:
                    if next_pos in timestamps and timestamps[next_pos] > enter_ns:
                        leave_ns = timestamps[next_pos]
                        break

            if enter_ns >= 0 and leave_ns >= 0 and leave_ns > enter_ns:
                events.append((iq_pos, source_node, enter_ns, leave_ns))

    # ===== RB 事件（在中间节点，暂时简化处理）=====
    rb_positions = ["RB_TR", "RB_TL", "RB_TU", "RB_TD", "RB_EQ"]
    for rb_pos in rb_positions:
        if rb_pos in timestamps:
            enter_ns = timestamps[rb_pos]
            leave_ns = -1
            for next_pos in ["Link", "EQ_TU", "EQ_TD", "EQ_CH"]:
                if next_pos in timestamps and timestamps[next_pos] > enter_ns:
                    leave_ns = timestamps[next_pos]
                    break

            if enter_ns >= 0 and leave_ns >= 0:
                # TODO: 推导中间节点ID（暂时跳过）
                pass

    # ===== EQ 事件（在目标节点）=====
    eq_positions = ["EQ_TU", "EQ_TD", "EQ_CH"]
    for eq_pos in eq_positions:
        if eq_pos in timestamps:
            enter_ns = timestamps[eq_pos]
            # 查找离开时间：H2L_H -> H2L_L -> IP_eject
            leave_ns = -1
            for next_pos in ["H2L_H", "H2L_L", "IP_eject"]:
                if next_pos in timestamps and timestamps[next_pos] > enter_ns:
                    leave_ns = timestamps[next_pos]
                    break
            if leave_ns < 0:
                leave_ns = timestamps.get("IP_eject", -1)

            if enter_ns >= 0 and leave_ns >= 0 and leave_ns > enter_ns:
                events.append((eq_pos, dest_node, enter_ns, leave_ns))

    # ===== IP 接收端口事件（在目标节点）=====
    # H2L: 从 EQ 或 H2L_H 到 H2L_L（高层到低层的转换）
    if "H2L_H" in timestamps:
        enter_ns = timestamps["H2L_H"]
        leave_ns = timestamps.get("H2L_L", timestamps.get("IP_eject", -1))
        if enter_ns >= 0 and leave_ns >= 0 and leave_ns > enter_ns:
            events.append(("H2L", dest_node, enter_ns, leave_ns))

    # IP_RX: 从 H2L_L（或 EQ）到 IP_eject（接收端的最后阶段）
    if "IP_eject" in timestamps:
        leave_ns = timestamps["IP_eject"]
        # 查找进入时间：优先 H2L_L，其次 H2L_H，再次 EQ
        enter_ns = -1
        for prev_pos in ["H2L_L", "H2L_H", "EQ_CH", "EQ_TU", "EQ_TD"]:
            if prev_pos in timestamps and timestamps[prev_pos] < leave_ns:
                enter_ns = timestamps[prev_pos]
                break
        if enter_ns >= 0 and leave_ns >= 0 and leave_ns > enter_ns:
            events.append(("IP_RX", dest_node, enter_ns, leave_ns))

    return events


def _parse_ip_channel_fifo(fifo_type: str) -> Tuple[str, str, str]:
    """解析 IP 通道 FIFO 类型

    Args:
        fifo_type: 如 "IQ_CH_G0", "EQ_CH_D1"

    Returns:
        (base_type, ip_type, channel_name)
        如 ("IQ_CH", "gdma_0", "G0") 或 ("", "", "") 如果不是 IP 通道
    """
    # 匹配 IQ_CH_X0 或 EQ_CH_X0 格式
    import re
    match = re.match(r'^(IQ_CH|EQ_CH)_([A-Z])(\d+)$', fifo_type)
    if not match:
        return ("", "", "")

    base_type = match.group(1)  # IQ_CH 或 EQ_CH
    type_char = match.group(2)  # G, D, C 等
    num = match.group(3)        # 0, 1 等

    # 映射类型字符到 IP 类型
    type_map = {
        'G': 'gdma',
        'D': 'ddr',
        'C': 'cpu',
        'P': 'pcie',
    }
    ip_prefix = type_map.get(type_char, type_char.lower())
    ip_type = f"{ip_prefix}_{num}"
    channel_name = f"{type_char}{num}"

    return (base_type, ip_type, channel_name)


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
        按 fifo_type.flit_type 分组:
        {
            "IQ_TR.req": [{"enter_ns": 10.5, "leave_ns": 12.3, "flit_id": "123.req.0", "flit_type": "req"}, ...],
            "IQ_TR.rsp": [...],
            "EQ_TU.req": [...],
            ...
        }
    """
    # 合并 flits 和 requests 数据
    merged = flits_df.merge(
        requests_df[["packet_id", "source_node", "dest_node", "source_type", "dest_type"]],
        on="packet_id",
        how="left"
    )

    # 确定要使用的 flit 类型列表
    if flit_types_filter:
        flit_types_to_use = flit_types_filter
    else:
        # 默认所有类型
        flit_types_to_use = ["req", "rsp", "data"]

    # 解析 IP 通道 FIFO 类型
    ip_channel_map = {}  # {fifo_type: (base_type, ip_type, channel_name)}
    regular_fifos = []
    for fifo_type in fifo_types:
        base_type, ip_type, channel_name = _parse_ip_channel_fifo(fifo_type)
        if base_type:
            ip_channel_map[fifo_type] = (base_type, ip_type, channel_name)
        else:
            regular_fifos.append(fifo_type)

    # 初始化结果：按 fifo_type.flit_type 分组
    waveform_data = {}
    for fifo_type in fifo_types:
        for flit_type in flit_types_to_use:
            key = f"{fifo_type}.{flit_type}"
            waveform_data[key] = []

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

        source_type = row.get("source_type", "")
        dest_type = row.get("dest_type", "")

        # 过滤：只保留目标节点的目标 FIFO 类型
        for fifo_type_name, event_node_id, enter_ns, leave_ns in events:
            if event_node_id != node_id:
                continue

            # 1. 处理常规 FIFO 类型
            if fifo_type_name in regular_fifos:
                key = f"{fifo_type_name}.{flit_type}"
                if key in waveform_data:
                    waveform_data[key].append({
                        "enter_ns": enter_ns,
                        "leave_ns": leave_ns,
                        "flit_id": f"{row['packet_id']}.{flit_type}.{row.get('flit_id', 0)}",
                        "flit_type": flit_type
                    })

            # 2. 处理 IP 通道 FIFO
            # IQ_CH_G0 -> L2H 事件，过滤 source_type=gdma_0
            # EQ_CH_G0 -> H2L 事件，过滤 dest_type=gdma_0
            for fifo_key, (base_type, ip_type, channel_name) in ip_channel_map.items():
                if base_type == "IQ_CH" and fifo_type_name == "L2H":
                    # IQ 中的 IP 通道：显示 L2H，按 source_type 过滤
                    if source_type == ip_type:
                        key = f"{fifo_key}.{flit_type}"
                        if key in waveform_data:
                            waveform_data[key].append({
                                "enter_ns": enter_ns,
                                "leave_ns": leave_ns,
                                "flit_id": f"{row['packet_id']}.{flit_type}.{row.get('flit_id', 0)}",
                                "flit_type": flit_type
                            })
                elif base_type == "EQ_CH" and fifo_type_name == "H2L":
                    # EQ 中的 IP 通道：显示 H2L，按 dest_type 过滤
                    if dest_type == ip_type:
                        key = f"{fifo_key}.{flit_type}"
                        if key in waveform_data:
                            waveform_data[key].append({
                                "enter_ns": enter_ns,
                                "leave_ns": leave_ns,
                                "flit_id": f"{row['packet_id']}.{flit_type}.{row.get('flit_id', 0)}",
                                "flit_type": flit_type
                            })

    # 按时间排序
    for key in waveform_data:
        waveform_data[key].sort(key=lambda x: x["enter_ns"])

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
    """从 waveform.parquet 的 position_timestamps 重建指定节点的 FIFO 波形数据"""
    result_type = _get_result_type(experiment_id)

    # 使用 waveform.py 中的加载函数
    from .waveform import _load_parquet_from_db
    requests_df, flits_df = _load_parquet_from_db(result_id, result_type)

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
    for key, events in waveform_data.items():
        if events:  # 只添加有事件的 FIFO
            # 时间范围过滤
            filtered_events = events
            if time_start is not None:
                filtered_events = [e for e in filtered_events if e["leave_ns"] >= time_start]
            if time_end is not None:
                filtered_events = [e for e in filtered_events if e["enter_ns"] <= time_end]

            if filtered_events:
                # key 格式: "IQ_TR.req" -> fifo_type="IQ_TR", flit_type="req"
                parts = key.rsplit(".", 1)
                base_fifo_type = parts[0] if len(parts) > 1 else key
                signals.append(FIFOSignal(
                    name=f"Node_{node_id}.{key}",
                    node_id=node_id,
                    fifo_type=base_fifo_type,  # 用于颜色映射
                    events=filtered_events
                ))
                all_times.extend([e["enter_ns"] for e in filtered_events])
                all_times.extend([e["leave_ns"] for e in filtered_events])

    # 计算时间范围（从0开始，结束时留5ns余量）
    padding_ns = 5.0
    time_range = {
        "start_ns": 0,
        "end_ns": (max(all_times) + padding_ns) if all_times else 0
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
