from fastapi import APIRouter, HTTPException
from typing import Dict, Tuple, List, Optional, Union
from pydantic import BaseModel, Field
from pathlib import Path
import yaml

# 导入 CrossRing 静态带宽分析模块
from src.traffic_process.traffic_gene.static_bandwidth_analyzer import (
    compute_link_bandwidth,
    compute_d2d_link_bandwidth,
)

from app.api.ip_mount import _load_mounts
from app.api.traffic_config import _load_configs

# D2D配置文件目录
D2D_CONFIG_DIR = Path(__file__).parent.parent.parent.parent.parent / "config" / "topologies"

router = APIRouter(prefix="/api/traffic/bandwidth", tags=["静态带宽分析"])


class BandwidthComputeRequest(BaseModel):
    """静态带宽计算请求"""
    topology: str = Field(..., description="拓扑类型")
    mode: str = Field(..., description="流量模式: noc 或 d2d")
    routing_type: str = Field(default="XY", description="路由算法: XY 或 YX")
    d2d_config_file: Optional[str] = Field(default=None, description="D2D配置文件名（仅D2D模式）")


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
    mode: str = Field(default="noc", description="流量模式: noc 或 d2d")
    link_bandwidth: Union[Dict[str, float], Dict[str, Dict[str, float]]] = Field(
        default={},
        description="链路带宽字典。NoC模式: {link_key: bw}; D2D模式: {die_id: {link_key: bw}}"
    )
    statistics: Union[BandwidthStatistics, Dict[str, BandwidthStatistics]] = Field(
        description="统计信息。NoC模式: 单个统计; D2D模式: {die_id: 统计}"
    )


def _load_d2d_config(config_filename: Optional[str] = None) -> Optional[Dict]:
    """加载D2D配置文件"""
    if config_filename:
        # 加载指定的配置文件
        config_file = D2D_CONFIG_DIR / config_filename
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if config.get('D2D_ENABLED', False):
                    return config
        return None

    # 未指定时，尝试加载默认配置文件
    config_files = [
        D2D_CONFIG_DIR / f"d2d_2die_config.yaml",
        D2D_CONFIG_DIR / f"d2d_4die_config.yaml",
    ]

    for config_file in config_files:
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if config.get('D2D_ENABLED', False):
                    return config

    return None


def _list_d2d_configs() -> List[Dict]:
    """列出所有可用的D2D配置文件"""
    configs = []
    if D2D_CONFIG_DIR.exists():
        for config_file in D2D_CONFIG_DIR.glob("d2d_*_config.yaml"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    if config.get('D2D_ENABLED', False):
                        configs.append({
                            "filename": config_file.name,
                            "num_dies": config.get('NUM_DIES', 2),
                            "connections": len(config.get('D2D_CONNECTIONS', []))
                        })
            except Exception:
                pass
    return configs


@router.get("/d2d-configs")
async def list_d2d_configs():
    """获取可用的D2D配置文件列表"""
    configs = _list_d2d_configs()
    return {"configs": configs}


@router.post("/compute", response_model=BandwidthComputeResponse)
async def compute_static_bandwidth(request: BandwidthComputeRequest):
    """
    计算静态链路带宽

    基于当前的IP挂载和流量配置，计算NoC拓扑中每条链路的静态带宽。
    使用指定的路由算法(XY或YX)进行计算。
    支持NoC和D2D两种模式。
    """
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
                    # "节点18-cdma_0" -> 节点ID=18, IP类型=cdma_0
                    node_id = int(ip.split('-')[0].replace('节点', ''))
                    ip_type = ip.split('-')[1]
                    if ip_type not in src_map:
                        src_map[ip_type] = []
                    src_map[ip_type].append(node_id)
                else:
                    # "cdma_0" -> 所有该类型节点
                    ip_type = ip
                    if ip_type in ip_to_nodes:
                        src_map[ip_type] = ip_to_nodes[ip_type]

            # 处理target_ip
            dst_ips = [config.target_ip] if isinstance(config.target_ip, str) else config.target_ip
            dst_map = {}
            for ip in dst_ips:
                # 提取IP类型：支持 "节点X-IP类型" 或 "IP类型" 格式
                if '-' in ip:
                    # "节点3-ddr_0" -> 节点ID=3, IP类型=ddr_0
                    node_id = int(ip.split('-')[0].replace('节点', ''))
                    ip_type = ip.split('-')[1]
                    if ip_type not in dst_map:
                        dst_map[ip_type] = []
                    dst_map[ip_type].append(node_id)
                else:
                    # "ddr_0" -> 所有该类型节点
                    ip_type = ip
                    if ip_type in ip_to_nodes:
                        dst_map[ip_type] = ip_to_nodes[ip_type]

            # 如果有有效的源和目标，创建配置
            if src_map and dst_map:
                # 创建一个类似config_manager.TrafficConfig的对象
                class AnalyzerConfig:
                    def __init__(self, src_map, dst_map, speed, burst, req_type, end_time, die_pairs=None):
                        self.src_map = src_map
                        self.dst_map = dst_map
                        self.speed = speed
                        self.burst = burst
                        self.req_type = req_type
                        self.end_time = end_time
                        self.die_pairs = die_pairs

                # D2D模式：获取die_pairs
                die_pairs = None
                if request.mode.lower() == "d2d":
                    if hasattr(config, 'die_pairs') and config.die_pairs:
                        die_pairs = config.die_pairs

                analyzer_config = AnalyzerConfig(
                    src_map=src_map,
                    dst_map=dst_map,
                    speed=config.speed_gbps,
                    burst=config.burst_length,
                    req_type=config.request_type,
                    end_time=config.end_time_ns,
                    die_pairs=die_pairs
                )
                configs_list.append(analyzer_config)

        if not configs_list:
            raise HTTPException(
                status_code=400,
                detail="没有有效的流量配置可用于带宽计算（可能IP未正确挂载）"
            )

        # 根据模式选择不同的计算方法
        if request.mode.lower() == "d2d":
            # D2D模式：加载D2D配置
            d2d_config = _load_d2d_config(request.d2d_config_file)
            if not d2d_config:
                raise HTTPException(
                    status_code=400,
                    detail="未找到D2D配置文件"
                )

            # 获取D2D连接配置
            d2d_connections = d2d_config.get('D2D_CONNECTIONS', [])
            num_dies = d2d_config.get('NUM_DIES', 2)

            # 生成双向D2D配对
            d2d_pairs = []
            for conn in d2d_connections:
                src_die, src_node, dst_die, dst_node = conn
                d2d_pairs.append((src_die, src_node, dst_die, dst_node))
                d2d_pairs.append((dst_die, dst_node, src_die, src_node))

            # 调用D2D静态带宽分析器
            die_link_bandwidth_dict = compute_d2d_link_bandwidth(
                topo_type=request.topology,
                node_ips=node_ips,
                configs=configs_list,
                d2d_pairs=d2d_pairs,
                routing_type=request.routing_type,
                num_dies=num_dies
            )

            # 转换字典格式
            link_bandwidth_str = {}
            statistics_dict = {}

            for die_id, link_bw in die_link_bandwidth_dict.items():
                die_key = str(die_id)
                link_bandwidth_str[die_key] = {}

                for (src_pos, dst_pos), bandwidth in link_bw.items():
                    key = f"{src_pos[0]},{src_pos[1]}-{dst_pos[0]},{dst_pos[1]}"
                    link_bandwidth_str[die_key][key] = bandwidth

                # 计算每个Die的统计信息
                active_bandwidths = [bw for bw in link_bw.values() if bw > 0]
                if active_bandwidths:
                    statistics_dict[die_key] = BandwidthStatistics(
                        max_bandwidth=max(active_bandwidths),
                        sum_bandwidth=sum(active_bandwidths),
                        avg_bandwidth=sum(active_bandwidths) / len(active_bandwidths),
                        num_active_links=len(active_bandwidths)
                    )
                else:
                    statistics_dict[die_key] = BandwidthStatistics(
                        max_bandwidth=0.0,
                        sum_bandwidth=0.0,
                        avg_bandwidth=0.0,
                        num_active_links=0
                    )

            return BandwidthComputeResponse(
                success=True,
                message=f"成功计算D2D静态链路带宽 (路由算法: {request.routing_type}, {num_dies}个Die)",
                mode="d2d",
                link_bandwidth=link_bandwidth_str,
                statistics=statistics_dict
            )

        else:
            # NoC模式：使用原有逻辑
            link_bandwidth_dict = compute_link_bandwidth(
                topo_type=request.topology,
                node_ips=node_ips,
                configs=configs_list,
                routing_type=request.routing_type
            )

            # 转换字典key为字符串格式
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
                mode="noc",
                link_bandwidth=link_bandwidth_str,
                statistics=statistics
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"带宽计算失败: {str(e)}"
        )
