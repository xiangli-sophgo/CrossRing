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


def extract_fifo_events_from_timestamps(position_timestamps_json: str, source_node: int, dest_node: int, cycle_time_ns: float = 0.5) -> List[Tuple[str, int, float, float]]:
    """从 position_timestamps JSON 提取 FIFO 事件

    每个事件的持续时间为一个 cycle（触发是瞬时的）

    Args:
        position_timestamps_json: JSON字符串格式的位置时间戳
        source_node: 源节点ID
        dest_node: 目标节点ID
        cycle_time_ns: 一个cycle的时间长度（ns），默认0.5ns对应2GHz

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
    # IP: IP_TX, L2H, H2L_H, H2L_L, IP_RX
    # IQ: IQ_TR, IQ_TL, IQ_TU, IQ_TD, IQ_CH
    # RB: RB_TR, RB_TL, RB_TU, RB_TD, RB_EQ
    # EQ: EQ_TU, EQ_TD, EQ_CH

    # ===== IP 发送端口事件（在源节点）=====
    if "IP_TX" in timestamps:
        enter_ns = timestamps["IP_TX"]
        events.append(("IP_TX", source_node, enter_ns, enter_ns + cycle_time_ns))

    # L2H 事件
    if "L2H" in timestamps:
        enter_ns = timestamps["L2H"]
        events.append(("L2H", source_node, enter_ns, enter_ns + cycle_time_ns))

    # ===== IQ 事件（在源节点）=====
    iq_positions = ["IQ_TR", "IQ_TL", "IQ_TU", "IQ_TD", "IQ_CH"]
    for iq_pos in iq_positions:
        if iq_pos in timestamps:
            enter_ns = timestamps[iq_pos]
            events.append((iq_pos, source_node, enter_ns, enter_ns + cycle_time_ns))

    # ===== RB 事件（从位置名称中提取节点 ID）=====
    import re
    rb_pattern = re.compile(r"^(RB_(?:TR|TL|TU|TD|EQ))_N(\d+)$")
    for pos_name, enter_ns in timestamps.items():
        match = rb_pattern.match(pos_name)
        if match:
            rb_type = match.group(1)  # RB_TR, RB_TL 等
            rb_node_id = int(match.group(2))  # 节点 ID
            events.append((rb_type, rb_node_id, enter_ns, enter_ns + cycle_time_ns))

    # ===== EQ 事件（在目标节点）=====
    eq_positions = ["EQ_TU", "EQ_TD", "EQ_CH"]
    for eq_pos in eq_positions:
        if eq_pos in timestamps:
            enter_ns = timestamps[eq_pos]
            events.append((eq_pos, dest_node, enter_ns, enter_ns + cycle_time_ns))

    # ===== IP 接收端口事件（在目标节点）=====
    # H2L 事件
    if "H2L_H" in timestamps:
        enter_ns = timestamps["H2L_H"]
        events.append(("H2L", dest_node, enter_ns, enter_ns + cycle_time_ns))

    # IP_RX 事件
    if "IP_RX" in timestamps:
        enter_ns = timestamps["IP_RX"]
        events.append(("IP_RX", dest_node, enter_ns, enter_ns + cycle_time_ns))

    return events


def _parse_ip_channel_fifo(fifo_type: str) -> Tuple[str, str, str]:
    """解析 IP 通道 FIFO 类型

    Args:
        fifo_type: 如 "IQ_CH_G0", "EQ_CH_D1", "IP_TX_G0", "IP_RX_D1"

    Returns:
        (base_type, ip_type, channel_name)
        如 ("IQ_CH", "gdma_0", "G0") 或 ("IP_TX", "gdma_0", "G0") 或 ("", "", "") 如果不是 IP 通道
    """
    import re

    # 匹配 IQ_CH_X0, EQ_CH_X0, IP_TX_X0, IP_RX_X0 格式
    match = re.match(r"^(IQ_CH|EQ_CH|IP_TX|IP_RX)_([A-Z])(\d+)$", fifo_type)
    if not match:
        return ("", "", "")

    base_type = match.group(1)  # IQ_CH, EQ_CH, IP_TX 或 IP_RX
    type_char = match.group(2)  # G, D, C 等
    num = match.group(3)  # 0, 1 等

    # 映射类型字符到 IP 类型
    type_map = {
        "G": "gdma",
        "D": "ddr",
        "C": "cpu",
        "P": "pcie",
        "S": "sdma",
        "N": "npu",
        "E": "eth",
        "L": "l2m",
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
    flit_types_filter: List[str] = None,
    expand_rsp_fifo_types: List[str] = None,
    expand_req_fifo_types: List[str] = None,
    expand_data_fifo_types: List[str] = None,
) -> Dict[str, List[Dict]]:
    """为指定节点构建 FIFO 波形数据（基于 position_timestamps）

    Args:
        flits_df: flits DataFrame
        requests_df: requests DataFrame
        node_id: 节点ID
        fifo_types: 要查询的FIFO类型列表
        flit_types_filter: Flit类型过滤列表（如 ["req", "rsp", "data"]），None表示不过滤
        expand_rsp_fifo_types: 需要展开rsp_type的FIFO类型列表（如 ["IQ_TR", "EQ_TD"]），None表示都不展开
        expand_req_fifo_types: 需要展开req_attr的FIFO类型列表，展开为new/old
        expand_data_fifo_types: 需要展开data_id的FIFO类型列表，展开为0/1/2/3...

    Returns:
        按 fifo_type.flit_type 分组（可选择进一步细分）:
        {
            "IQ_TR.req": [...],
            "IQ_TR.req.new": [...],  # req展开时按req_attr细分
            "IQ_TR.req.old": [...],
            "IQ_TR.rsp": [...],  # 未展开时合并所有rsp_type
            "IQ_TR.rsp.CompData": [...],  # rsp展开时按rsp_type细分
            "IQ_TR.data": [...],
            "IQ_TR.data.0": [...],  # data展开时按data_id细分
            "IQ_TR.data.1": [...],
            ...
        }
    """
    expand_rsp_fifo_types = expand_rsp_fifo_types or []
    expand_req_fifo_types = expand_req_fifo_types or []
    expand_data_fifo_types = expand_data_fifo_types or []

    # 检查 flits_df 是否已包含 source_node/dest_node（新格式）
    if "source_node" in flits_df.columns and "dest_node" in flits_df.columns:
        # 新格式：flits_df 已包含 flit 级别的 source/dest
        merged = flits_df.copy()
        # 如果缺少 source_type/dest_type，从 requests_df 补充
        if "source_type" not in merged.columns or "dest_type" not in merged.columns:
            merged = merged.merge(requests_df[["packet_id", "source_type", "dest_type"]], on="packet_id", how="left")
    else:
        # 旧格式：需要从 requests_df 获取 source/dest
        merged = flits_df.merge(requests_df[["packet_id", "source_node", "dest_node", "source_type", "dest_type"]], on="packet_id", how="left")

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

    # print(f"[DEBUG] build_fifo_waveform_for_node: node_id={node_id}, fifo_types={fifo_types}")
    # print(f"[DEBUG] regular_fifos={regular_fifos}, ip_channel_map={ip_channel_map}")

    # 初始化结果：按 fifo_type.flit_type 分组（rsp 按 rsp_type 动态创建）
    waveform_data = {}
    for fifo_type in fifo_types:
        for flit_type in flit_types_to_use:
            # rsp 类型不预创建，由实际数据动态创建（因为rsp_type种类不确定）
            if flit_type != "rsp":
                key = f"{fifo_type}.{flit_type}"
                waveform_data[key] = []

    # 辅助函数：生成 key 并添加事件
    def add_event(fifo_type: str, flit_type: str, rsp_type: str, req_attr: str, data_id: int, enter_ns: float, leave_ns: float, flit_id_str: str):
        """添加事件到 waveform_data，根据展开配置决定是否细分"""
        if flit_type == "rsp":
            # 检查该 fifo_type 是否需要展开 rsp_type
            if fifo_type in expand_rsp_fifo_types and rsp_type:
                key = f"{fifo_type}.rsp.{rsp_type}"
            else:
                key = f"{fifo_type}.rsp"
        elif flit_type == "req":
            # 检查该 fifo_type 是否需要展开 req_attr
            # 只有 old/retry 类型需要展开显示，new 类型保持在主 req 行
            if fifo_type in expand_req_fifo_types and req_attr and req_attr != "new":
                key = f"{fifo_type}.req.Retry"  # old/retry 统一显示为 Retry
            else:
                key = f"{fifo_type}.req"
        elif flit_type == "data":
            # 检查该 fifo_type 是否需要展开 data_id
            if fifo_type in expand_data_fifo_types:
                key = f"{fifo_type}.data.{data_id}"
            else:
                key = f"{fifo_type}.data"
        else:
            key = f"{fifo_type}.{flit_type}"

        # 动态创建 key
        if key not in waveform_data:
            waveform_data[key] = []

        waveform_data[key].append(
            {
                "enter_ns": enter_ns,
                "leave_ns": leave_ns,
                "flit_id": flit_id_str,
                "flit_type": flit_type,
                "rsp_type": rsp_type if flit_type == "rsp" else "",
                "req_attr": req_attr if flit_type == "req" else "",
                "data_id": data_id if flit_type == "data" else -1,
            }
        )

    # 遍历所有 flit
    for _, row in merged.iterrows():
        # 获取flit类型
        flit_type = row.get("flit_type", "")

        # flit类型过滤
        if flit_types_filter and flit_type not in flit_types_filter:
            continue

        # 获取各类型特有属性
        rsp_type = row.get("rsp_type", "") if flit_type == "rsp" else ""
        req_attr = row.get("req_attr", "") if flit_type == "req" else ""
        data_id = int(row.get("flit_id", 0)) if flit_type == "data" else -1

        # 从 position_timestamps 提取事件
        events = extract_fifo_events_from_timestamps(row.get("position_timestamps", "{}"), row["source_node"], row["dest_node"])

        source_type = row.get("source_type", "")
        dest_type = row.get("dest_type", "")

        # 生成 flit_id_str 用于显示
        # req: "123" 或 "123(Retry)"
        # rsp: "123(Comp)" 或 "123(CompData)"
        # data: "123.1"
        packet_id = row["packet_id"]
        if flit_type == "req":
            if req_attr and req_attr != "new":
                flit_id_str = f"{packet_id}(Retry)"
            else:
                flit_id_str = str(packet_id)
        elif flit_type == "rsp":
            if rsp_type:
                flit_id_str = f"{packet_id}({rsp_type})"
            else:
                flit_id_str = str(packet_id)
        elif flit_type == "data":
            flit_id_str = f"{packet_id}.{data_id}"
        else:
            flit_id_str = str(packet_id)

        # 过滤：只保留目标节点的目标 FIFO 类型
        for fifo_type_name, event_node_id, enter_ns, leave_ns in events:
            if event_node_id != node_id:
                continue

            # 1. 处理常规 FIFO 类型
            if fifo_type_name in regular_fifos:
                add_event(fifo_type_name, flit_type, rsp_type, req_attr, data_id, enter_ns, leave_ns, flit_id_str)

            # 2. 处理 IP 通道 FIFO
            # IQ_CH_G0 -> L2H 事件，过滤 source_type=gdma_0
            # EQ_CH_G0 -> H2L 事件，过滤 dest_type=gdma_0
            # IP_TX_G0 -> IP_TX 事件，过滤 source_type=gdma_0
            # IP_RX_G0 -> IP_RX 事件，过滤 dest_type=gdma_0
            for fifo_key, (base_type, ip_type, channel_name) in ip_channel_map.items():
                if base_type == "IQ_CH" and fifo_type_name == "L2H":
                    # IQ 中的 IP 通道：显示 L2H，按 source_type 过滤
                    if source_type == ip_type:
                        add_event(fifo_key, flit_type, rsp_type, req_attr, data_id, enter_ns, leave_ns, flit_id_str)
                elif base_type == "EQ_CH" and fifo_type_name == "H2L":
                    # EQ 中的 IP 通道：显示 H2L，按 dest_type 过滤
                    if dest_type == ip_type:
                        add_event(fifo_key, flit_type, rsp_type, req_attr, data_id, enter_ns, leave_ns, flit_id_str)
                elif base_type == "IP_TX" and fifo_type_name == "IP_TX":
                    # IP 发送端口：按 source_type 过滤
                    # print(f"[DEBUG] IP_TX match: fifo_key={fifo_key}, ip_type={ip_type}, source_type={source_type}, event_node_id={event_node_id}")
                    if source_type == ip_type:
                        add_event(fifo_key, flit_type, rsp_type, req_attr, data_id, enter_ns, leave_ns, flit_id_str)
                        # print(f"[DEBUG] IP_TX event added: {fifo_key}")
                elif base_type == "IP_RX" and fifo_type_name == "IP_RX":
                    # IP 接收端口：按 dest_type 过滤
                    # print(f"[DEBUG] IP_RX match: fifo_key={fifo_key}, ip_type={ip_type}, dest_type={dest_type}, event_node_id={event_node_id}")
                    if dest_type == ip_type:
                        add_event(fifo_key, flit_type, rsp_type, req_attr, data_id, enter_ns, leave_ns, flit_id_str)
                        # print(f"[DEBUG] IP_RX event added: {fifo_key}")

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
    expand_rsp_signals: str = Query(None, description="需要展开rsp_type的信号(逗号分隔，如IQ_TR.rsp,EQ_TD.rsp)"),
    expand_req_signals: str = Query(None, description="需要展开req_attr的信号(逗号分隔，如IQ_TR.req)"),
    expand_data_signals: str = Query(None, description="需要展开data_id的信号(逗号分隔，如IQ_TR.data)"),
    time_start: float = Query(None, description="起始时间(ns)"),
    time_end: float = Query(None, description="结束时间(ns)"),
) -> FIFOWaveformResponse:
    """从 waveform.parquet 的 position_timestamps 重建指定节点的 FIFO 波形数据"""
    # print(f"[DEBUG] get_fifo_waveform called: node_id={node_id}, fifo_types={fifo_types}, flit_types_filter={flit_types_filter}")
    result_type = _get_result_type(experiment_id)

    # 使用 waveform.py 中的加载函数
    from .waveform import _load_parquet_from_db

    requests_df, flits_df = _load_parquet_from_db(result_id, result_type)

    # 检查是否包含 position_timestamps 字段
    if "position_timestamps" not in flits_df.columns:
        raise HTTPException(status_code=400, detail="flits.parquet 缺少 position_timestamps 字段，请重新运行仿真")

    # 解析请求的 FIFO 类型
    requested_fifos = [f.strip() for f in fifo_types.split(",")]

    # 解析flit类型过滤列表
    flit_types_list = None
    if flit_types_filter:
        flit_types_list = [f.strip() for f in flit_types_filter.split(",")]

    # 解析需要展开的信号列表
    # 格式: "IQ_TR.rsp,EQ_TD.rsp" -> ["IQ_TR", "EQ_TD"]
    def parse_expand_signals(signals_str: str, suffix: str) -> list:
        result = []
        if signals_str:
            for sig in signals_str.split(","):
                sig = sig.strip()
                if sig.endswith(suffix):
                    result.append(sig[: -len(suffix)])  # 去掉后缀
        return result

    expand_rsp_list = parse_expand_signals(expand_rsp_signals, ".rsp")
    expand_req_list = parse_expand_signals(expand_req_signals, ".req")
    expand_data_list = parse_expand_signals(expand_data_signals, ".data")

    # 重建 FIFO 波形
    waveform_data = build_fifo_waveform_for_node(flits_df, requests_df, node_id, requested_fifos, flit_types_list, expand_rsp_list, expand_req_list, expand_data_list)

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
                signals.append(FIFOSignal(name=f"Node_{node_id}.{key}", node_id=node_id, fifo_type=base_fifo_type, events=filtered_events))  # 用于颜色映射
                all_times.extend([e["enter_ns"] for e in filtered_events])
                all_times.extend([e["leave_ns"] for e in filtered_events])

    # 对信号进行排序，确保展开的子类型紧跟在父类型后面
    # 排序规则：FIFO类型 > flit类型(req/rsp/data) > 子类型
    flit_type_order = {"req": 0, "rsp": 1, "data": 2}

    def signal_sort_key(signal: FIFOSignal):
        # name 格式: "Node_X.FIFO.flit_type" 或 "Node_X.FIFO.flit_type.sub_type"
        parts = signal.name.split(".")
        if len(parts) >= 3:
            fifo_type = parts[1]
            flit_type = parts[2]
            sub_type = parts[3] if len(parts) >= 4 else ""
            # 子类型排序：空字符串（父类型）在前，其他按字母顺序
            return (fifo_type, flit_type_order.get(flit_type, 99), 0 if not sub_type else 1, sub_type)
        return (signal.name, 99, 0, "")

    signals.sort(key=signal_sort_key)

    # 计算时间范围（从0开始，结束时留5ns余量）
    padding_ns = 5.0
    time_range = {"start_ns": 0, "end_ns": (max(all_times) + padding_ns) if all_times else 0}

    # 可用 FIFO 列表
    available_fifos = list(waveform_data.keys())

    return FIFOWaveformResponse(time_range=time_range, signals=signals, available_fifos=available_fifos)


@router.get("/experiments/{experiment_id}/results/{result_id}/fifo-waveform/available")
async def get_available_fifos(
    experiment_id: int,
    result_id: int,
    node_id: int = Query(..., description="节点ID"),
) -> Dict[str, List[str]]:
    """获取指定节点可用的 FIFO 列表"""
    # 返回所有可能的 FIFO 类型
    all_fifos = ["IQ_TR", "IQ_TL", "IQ_TU", "IQ_TD", "IQ_EQ", "IQ_CH", "RB_TR", "RB_TL", "RB_TU", "RB_TD", "RB_EQ", "EQ_TU", "EQ_TD", "EQ_CH"]
    return {"fifos": all_fifos}
