from fastapi import APIRouter, HTTPException
from typing import Dict, Tuple, List
from pydantic import BaseModel, Field

# 导入 CrossRing 静态带宽分析模块
from src.traffic_process.traffic_gene.static_bandwidth_analyzer import compute_link_bandwidth

from app.api.ip_mount import _load_mounts
from app.api.traffic_config import _load_configs

router = APIRouter(prefix="/api/traffic/bandwidth", tags=["静态带宽分析"])


class BandwidthComputeRequest(BaseModel):
    """静态带宽计算请求"""
    topology: str = Field(..., description="拓扑类型")
    mode: str = Field(..., description="流量模式: noc 或 d2d")
    routing_type: str = Field(default="XY", description="路由算法: XY 或 YX")


class BandwidthStatistics(BaseModel):
    """带宽统计信息"""
    max_bandwidth: float = Field(..., description="最大链路带宽 (GB/s)")
    sum_bandwidth: float = Field(..., description="总带宽 (GB/s)")
    avg_bandwidth: float = Field(..., description="平均带宽 (GB/s)")
    num_active_links: int = Field(..., description="活动链路数量")


class BandwidthComputeResponse(BaseModel):
    """静态带宽计算响应"""
    success: bool
    message: str
    link_bandwidth: Dict[str, float] = Field(default={}, description="链路带宽字典 (key格式: 'x1,y1-x2,y2')")
    statistics: BandwidthStatistics


@router.post("/compute", response_model=BandwidthComputeResponse)
async def compute_static_bandwidth(request: BandwidthComputeRequest):
    """
    计算静态链路带宽

    基于当前的IP挂载和流量配置，计算NoC拓扑中每条链路的静态带宽。
    使用指定的路由算法(XY或YX)进行计算。
    """
    # 检查模式
    if request.mode.lower() == "d2d":
        raise HTTPException(
            status_code=400,
            detail="D2D模式暂不支持静态链路带宽计算"
        )

    # 验证路由类型
    if request.routing_type not in ["XY", "YX"]:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的路由类型: {request.routing_type}. 必须是 'XY' 或 'YX'"
        )

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

    try:
        # 转换IP挂载格式: {node_id: [IPMount]} -> {node_id: [ip_type]}
        node_ips = {}
        for node_id, mounts in ip_mounts.items():
            if isinstance(mounts, list):
                node_ips[node_id] = [mount.ip_type for mount in mounts]
            else:
                # 兼容旧格式
                node_ips[node_id] = [mounts.ip_type]

        # 创建 ip_type -> [node_ids] 的映射
        ip_to_nodes = {}
        for node_id, ip_types in node_ips.items():
            for ip_type in ip_types:
                if ip_type not in ip_to_nodes:
                    ip_to_nodes[ip_type] = []
                ip_to_nodes[ip_type].append(node_id)

        # 转换流量配置格式: TrafficConfig -> StaticBandwidthAnalyzer期望的格式
        # TrafficConfig: source_ip, target_ip (str | List[str])
        # Analyzer期望: src_map, dst_map, speed, burst, req_type, end_time
        configs_list = []
        for config in traffic_configs.values():
            # 处理source_ip
            src_ips = [config.source_ip] if isinstance(config.source_ip, str) else config.source_ip
            src_map = {}
            for ip in src_ips:
                # 提取IP类型：支持 "节点X-IP类型" 或 "IP类型" 格式
                if '-' in ip:
                    ip_type = ip.split('-')[1]  # "节点18-cdma_0" -> "cdma_0"
                else:
                    ip_type = ip  # "cdma_0" -> "cdma_0"

                if ip_type in ip_to_nodes:
                    src_map[ip_type] = ip_to_nodes[ip_type]
                else:
                    # 可能IP没有挂载，跳过
                    continue

            # 处理target_ip
            dst_ips = [config.target_ip] if isinstance(config.target_ip, str) else config.target_ip
            dst_map = {}
            for ip in dst_ips:
                # 提取IP类型：支持 "节点X-IP类型" 或 "IP类型" 格式
                if '-' in ip:
                    ip_type = ip.split('-')[1]  # "节点3-ddr_0" -> "ddr_0"
                else:
                    ip_type = ip  # "ddr_0" -> "ddr_0"

                if ip_type in ip_to_nodes:
                    dst_map[ip_type] = ip_to_nodes[ip_type]
                else:
                    continue

            # 如果有有效的源和目标，创建配置
            if src_map and dst_map:
                # 创建一个类似config_manager.TrafficConfig的对象
                class AnalyzerConfig:
                    def __init__(self, src_map, dst_map, speed, burst, req_type, end_time):
                        self.src_map = src_map
                        self.dst_map = dst_map
                        self.speed = speed
                        self.burst = burst
                        self.req_type = req_type
                        self.end_time = end_time

                analyzer_config = AnalyzerConfig(
                    src_map=src_map,
                    dst_map=dst_map,
                    speed=config.speed_gbps,
                    burst=config.burst_length,
                    req_type=config.request_type,
                    end_time=config.end_time_ns
                )
                configs_list.append(analyzer_config)

        if not configs_list:
            raise HTTPException(
                status_code=400,
                detail="没有有效的流量配置可用于带宽计算（可能IP未正确挂载）"
            )

        # 调用静态带宽分析器
        link_bandwidth_dict = compute_link_bandwidth(
            topo_type=request.topology,
            node_ips=node_ips,
            configs=configs_list,
            routing_type=request.routing_type
        )

        # 转换字典key为字符串格式 (FastAPI不支持tuple key的JSON序列化)
        # {((x1,y1), (x2,y2)): bw} -> {"x1,y1-x2,y2": bw}
        link_bandwidth_str = {}
        for (src_pos, dst_pos), bandwidth in link_bandwidth_dict.items():
            key = f"{src_pos[0]},{src_pos[1]}-{dst_pos[0]},{dst_pos[1]}"
            link_bandwidth_str[key] = bandwidth

        # 计算统计信息
        active_bandwidths = [bw for bw in link_bandwidth_dict.values() if bw > 0]

        if active_bandwidths:
            statistics = BandwidthStatistics(
                max_bandwidth=max(active_bandwidths),
                sum_bandwidth=sum(active_bandwidths),
                avg_bandwidth=sum(active_bandwidths) / len(active_bandwidths),
                num_active_links=len(active_bandwidths)
            )
        else:
            statistics = BandwidthStatistics(
                max_bandwidth=0.0,
                sum_bandwidth=0.0,
                avg_bandwidth=0.0,
                num_active_links=0
            )

        return BandwidthComputeResponse(
            success=True,
            message=f"成功计算静态链路带宽 (路由算法: {request.routing_type})",
            link_bandwidth=link_bandwidth_str,
            statistics=statistics
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"带宽计算失败: {str(e)}"
        )
