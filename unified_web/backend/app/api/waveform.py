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
    cmd_latency_ns: Optional[float] = None
    data_latency_ns: Optional[float] = None
    transaction_latency_ns: Optional[float] = None


class PacketListResponse(BaseModel):
    """请求列表响应"""
    packets: List[PacketInfo]
    total: int
    page: int
    page_size: int


class TopologyNode(BaseModel):
    """拓扑节点"""
    id: int
    row: int
    col: int
    label: str


class TopologyEdge(BaseModel):
    """拓扑边"""
    source: int
    target: int
    direction: str  # 'horizontal' | 'vertical'
    type: str  # 'row_link' | 'col_link'


class TopologyMetadata(BaseModel):
    """拓扑元数据"""
    row_links: int
    col_links: int
    total_links: int


class TopologyData(BaseModel):
    """拓扑数据"""
    type: str
    rows: int
    cols: int
    total_nodes: int
    nodes: List[TopologyNode]
    edges: List[TopologyEdge]
    metadata: TopologyMetadata


# ==================== 辅助函数 ====================

def _parse_topology_type(topo_type: str) -> Tuple[int, int]:
    """解析拓扑类型字符串

    Args:
        topo_type: 拓扑类型字符串，格式 "AxB"（例如 "8x8", "5x4"）

    Returns:
        (rows, cols) 元组

    Raises:
        ValueError: 如果格式不正确
    """
    if not topo_type:
        raise ValueError("拓扑类型为空")

    parts = topo_type.lower().split('x')
    if len(parts) != 2:
        raise ValueError(f"拓扑类型格式错误: {topo_type}，应为 AxB 格式")

    try:
        rows = int(parts[0])
        cols = int(parts[1])
    except ValueError:
        raise ValueError(f"拓扑类型格式错误: {topo_type}，行列必须为整数")

    if rows <= 0 or cols <= 0:
        raise ValueError(f"拓扑类型错误: {topo_type}，行列必须为正整数")

    return rows, cols


def _generate_topology_data(topo_type: str) -> TopologyData:
    """根据拓扑类型生成拓扑数据

    Args:
        topo_type: 拓扑类型字符串，格式 "AxB"

    Returns:
        TopologyData 对象
    """
    rows, cols = _parse_topology_type(topo_type)
    total_nodes = rows * cols

    # 生成节点
    nodes = []
    for i in range(total_nodes):
        row = i // cols
        col = i % cols
        nodes.append(TopologyNode(
            id=i,
            row=row,
            col=col,
            label=str(i)
        ))

    # 生成边
    edges = []
    row_links = 0
    col_links = 0

    for i in range(total_nodes):
        row = i // cols
        col = i % cols

        # 横向边（向右）
        if col < cols - 1:
            edges.append(TopologyEdge(
                source=i,
                target=i + 1,
                direction="horizontal",
                type="row_link"
            ))
            row_links += 1

        # 纵向边（向下）
        if row < rows - 1:
            edges.append(TopologyEdge(
                source=i,
                target=i + cols,
                direction="vertical",
                type="col_link"
            ))
            col_links += 1

    # 计算元数据
    metadata = TopologyMetadata(
        row_links=row_links,
        col_links=col_links,
        total_links=row_links + col_links
    )

    return TopologyData(
        type=topo_type,
        rows=rows,
        cols=cols,
        total_nodes=total_nodes,
        nodes=nodes,
        edges=edges,
        metadata=metadata
    )


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
    import json

    # 获取响应类型和retry信息
    rsp_type = flit_row.get("rsp_type", "") if "rsp_type" in flit_row else ""
    req_attr = flit_row.get("req_attr", "") if "req_attr" in flit_row else ""

    # 调试日志
    logger.info(f"[DEBUG] packet_id={packet_id}, flit_type={flit_type}, rsp_type='{rsp_type}', req_attr='{req_attr}'")

    # 信号名
    if flit_type == "req":
        retry_suffix = "(retry)" if req_attr == "old" else ""
        name = f"{packet_id}.REQ{retry_suffix}"
    elif flit_type == "rsp":
        rsp_suffix = f"({rsp_type})" if rsp_type else ""
        name = f"{packet_id}.RSP{rsp_suffix}"
    else:
        name = f"{packet_id}.D{flit_id}"

    # 解析 position_timestamps JSON 字符串
    pos_timestamps = {}
    if "position_timestamps" in flit_row and flit_row["position_timestamps"]:
        try:
            pos_timestamps = json.loads(flit_row["position_timestamps"])
            logger.debug(f"Packet {packet_id} flit {flit_type} timestamps: {list(pos_timestamps.keys())}")
        except Exception as e:
            logger.error(f"Failed to parse position_timestamps: {e}")
            pass

    # 将所有位置按时间排序
    sorted_positions = sorted(pos_timestamps.items(), key=lambda x: x[1])

    if not sorted_positions:
        # 没有时间戳数据
        return WaveformSignal(
            name=name,
            packet_id=packet_id,
            flit_type=flit_type,
            flit_id=flit_id,
            events=[]
        )

    # 位置到阶段的映射
    def get_stage(pos_name: str) -> str:
        if pos_name == "L2H":
            return "IP_inject"
        elif pos_name.startswith("IQ"):
            return "IQ"
        elif pos_name == "Link":
            return "Link"
        elif pos_name == "RB":
            return "RB"
        elif pos_name.startswith("EQ"):
            return "EQ"
        elif pos_name in ["H2L_H", "H2L_L"]:
            return "IP_eject"
        else:
            return "Unknown"

    # 按阶段分组连续的位置
    events = []
    current_stage = None
    stage_start = None

    for i, (pos_name, pos_time) in enumerate(sorted_positions):
        stage = get_stage(pos_name)

        if stage == "Unknown":
            continue

        if current_stage is None:
            # 第一个阶段
            current_stage = stage
            stage_start = pos_time
        elif stage != current_stage:
            # 阶段切换，保存上一个阶段
            if stage_start is not None:
                events.append(WaveformEvent(
                    stage=current_stage,
                    start_ns=float(stage_start),
                    end_ns=float(pos_time)
                ))
            current_stage = stage
            stage_start = pos_time

    # 添加最后一个阶段（使用最后一个时间戳作为结束）
    if current_stage is not None and stage_start is not None and sorted_positions:
        last_time = sorted_positions[-1][1]
        events.append(WaveformEvent(
            stage=current_stage,
            start_ns=float(stage_start),
            end_ns=float(last_time)
        ))

    return WaveformSignal(
        name=name,
        packet_id=packet_id,
        flit_type=flit_type,
        flit_id=flit_id,
        events=events
    )


def _get_flit_start_time(flit_row) -> float:
    """从 position_timestamps 中提取 flit 的开始时间（优先使用L2H时间戳）"""
    import json
    pos_timestamps = {}
    if "position_timestamps" in flit_row and flit_row["position_timestamps"]:
        try:
            pos_timestamps = json.loads(flit_row["position_timestamps"])
        except:
            pass

    if pos_timestamps:
        # 优先使用L2H时间戳（进入网络的时间）
        if 'L2H' in pos_timestamps:
            return pos_timestamps['L2H']
        # 否则使用最小时间戳
        return min(pos_timestamps.values())
    return float('inf')


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
    filtered_flits = flits_df[flits_df["packet_id"].isin(final_packet_ids)].copy()

    # 计算每个 flit 的开始时间并排序
    filtered_flits['_start_time'] = filtered_flits.apply(_get_flit_start_time, axis=1)
    filtered_flits = filtered_flits.sort_values(['packet_id', '_start_time'])
    filtered_flits = filtered_flits.drop(columns=['_start_time'])

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

    # 位置列表（按flit传输顺序）
    stages = ["IP_inject", "IQ", "Link", "RB", "EQ", "IP_eject"]

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

        # 尝试读取延迟字段（负值视为无效）
        # 注意：parquet中字段名是trans_latency_ns而不是transaction_latency_ns
        cmd_latency = None
        data_latency = None
        transaction_latency = None

        if "cmd_latency_ns" in row and row["cmd_latency_ns"] is not None and row["cmd_latency_ns"] >= 0:
            cmd_latency = float(row["cmd_latency_ns"])
        if "data_latency_ns" in row and row["data_latency_ns"] is not None and row["data_latency_ns"] >= 0:
            data_latency = float(row["data_latency_ns"])

        # 先尝试transaction_latency_ns（兼容旧数据），再尝试trans_latency_ns（实际字段名）
        trans_field = "transaction_latency_ns" if "transaction_latency_ns" in row else "trans_latency_ns"
        if trans_field in row and row[trans_field] is not None and row[trans_field] >= 0:
            transaction_latency = float(row[trans_field])

        packets.append(PacketInfo(
            packet_id=int(row["packet_id"]),
            req_type=row["req_type"],
            source_node=int(row["source_node"]),
            source_type=row["source_type"],
            dest_node=int(row["dest_node"]),
            dest_type=row["dest_type"],
            start_time_ns=start_time,
            end_time_ns=end_time,
            latency_ns=end_time - start_time,
            cmd_latency_ns=cmd_latency,
            data_latency_ns=data_latency,
            transaction_latency_ns=transaction_latency,
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


@router.get("/experiments/{experiment_id}/results/{result_id}/topology")
async def get_topology_data(
    experiment_id: int,
    result_id: int,
) -> TopologyData:
    """
    获取拓扑数据用于波形查看器

    从实验配置中读取拓扑类型，生成节点和边的数据

    Returns:
        拓扑数据，包含节点列表和边列表
    """
    # 获取实验信息
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    # 获取拓扑类型
    topo_type = experiment.get("topo_type")
    if not topo_type:
        raise HTTPException(status_code=404, detail="实验未配置拓扑类型")

    # 生成拓扑数据
    try:
        topology_data = _generate_topology_data(topo_type)
        return topology_data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"拓扑类型解析失败: {str(e)}")
    except Exception as e:
        logger.error(f"生成拓扑数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成拓扑数据失败: {str(e)}")


@router.get("/experiments/{experiment_id}/results/{result_id}/active-ips")
async def get_active_ips(
    experiment_id: int,
    result_id: int,
) -> Dict[str, Any]:
    """获取实验中活跃的IP（从requests数据中提取）

    从parquet文件中提取所有请求涉及的节点和IP类型组合

    Returns:
        {"active_ips": {node_id: [ip_type1, ip_type2, ...]}}
    """
    result_type = _get_result_type(experiment_id)

    try:
        requests_df, _ = _load_parquet_from_db(result_id, result_type)
    except HTTPException:
        return {"active_ips": {}}

    # 提取source节点的IP
    source_ips = requests_df[["source_node", "source_type"]].drop_duplicates()
    source_ips.columns = ["node", "ip_type"]

    # 提取dest节点的IP
    dest_ips = requests_df[["dest_node", "dest_type"]].drop_duplicates()
    dest_ips.columns = ["node", "ip_type"]

    # 合并并去重
    all_ips = pd.concat([source_ips, dest_ips]).drop_duplicates()

    # 按节点分组
    result: Dict[int, List[str]] = {}
    for _, row in all_ips.iterrows():
        node_id = int(row["node"])
        ip_type = row["ip_type"]
        if node_id not in result:
            result[node_id] = []
        if ip_type not in result[node_id]:
            result[node_id].append(ip_type)

    return {"active_ips": result}
