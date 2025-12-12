"""
数据中心层级拓扑数据模型

层级结构: Pod -> Rack -> Board -> Chip
"""

from typing import List, Optional, Literal, Tuple, Dict
from pydantic import BaseModel, Field
from enum import Enum


# Chip类型
ChipType = Literal['npu', 'cpu']


class ChipConfig(BaseModel):
    """芯片配置"""
    id: str
    type: ChipType
    position: Tuple[int, int]  # (行, 列) 在Board上的位置
    label: Optional[str] = None


class BoardConfig(BaseModel):
    """板卡配置"""
    id: str
    u_position: int = Field(..., ge=1, le=42, description="起始U位 (1-42)")
    u_height: int = Field(..., ge=1, le=10, description="占用U数")
    label: str
    chips: List[ChipConfig] = []


class RackConfig(BaseModel):
    """机柜配置"""
    id: str
    position: Tuple[int, int]  # 在Pod中的网格位置 (行, 列)
    label: str
    total_u: int = Field(default=42, description="总U数")
    boards: List[BoardConfig] = []


class PodConfig(BaseModel):
    """Pod配置"""
    id: str
    label: str
    grid_size: Tuple[int, int]  # Rack排列网格 (行, 列)
    racks: List[RackConfig] = []


class ConnectionConfig(BaseModel):
    """连接配置"""
    source: str
    target: str
    type: Literal['intra', 'inter', 'switch', 'manual']  # 层内/层间/Switch/手动连接
    bandwidth: Optional[float] = None  # 带宽 (Gbps)
    latency: Optional[float] = None  # 延迟 (ns)
    connection_role: Optional[Literal['uplink', 'downlink', 'inter']] = None  # Switch连接角色
    is_manual: bool = False  # 是否为手动添加的连接


# ============================================
# 手动连接配置模型
# ============================================

# 层级类型
HierarchyLevel = Literal['datacenter', 'pod', 'rack', 'board']


class ManualConnection(BaseModel):
    """手动添加的连接"""
    id: str
    source: str
    target: str
    hierarchy_level: HierarchyLevel
    bandwidth: Optional[float] = None
    latency: Optional[float] = None  # 延迟 (ns)
    description: Optional[str] = None
    created_at: Optional[str] = None


class ManualConnectionConfig(BaseModel):
    """手动连接配置"""
    enabled: bool = False
    mode: Literal['append', 'replace'] = 'append'
    connections: List[ManualConnection] = []


# ============================================
# Switch配置模型
# ============================================

class SwitchTypeConfig(BaseModel):
    """Switch类型预定义"""
    id: str  # 类型标识，如 "leaf_72", "spine_512"
    name: str  # 显示名称，如 "72端口Leaf交换机"
    port_count: int  # 总端口数


class SwitchLayerConfig(BaseModel):
    """单层Switch配置"""
    layer_name: str  # 层名称，如 "leaf", "spine"
    switch_type_id: str  # 使用的Switch类型ID
    count: int  # 该层Switch数量
    inter_connect: bool = False  # 同层Switch是否互联


class SwitchConnectionMode(str, Enum):
    """Switch与下层节点的连接模式"""
    ROUND_ROBIN = "round_robin"  # 轮询连接（默认，节点轮流连接到不同Switch）
    FULL_MESH = "full_mesh"      # 全连接（每个节点连接到所有Switch）
    GROUP = "group"              # 分组连接（节点按组分配到不同Switch）
    CUSTOM = "custom"            # 自定义连接（手动指定连接关系）


class SwitchGroupConfig(BaseModel):
    """Switch分组连接配置"""
    # 分组数量，节点将被平均分配到各组
    group_count: int = 2
    # 每组连接到哪些Switch（索引），如果为空则自动分配
    group_switch_mapping: Optional[Dict[int, List[int]]] = None


class SwitchCustomConnection(BaseModel):
    """自定义Switch连接"""
    device_id: str  # 设备ID
    switch_indices: List[int]  # 连接到的Switch索引列表


class HierarchyLevelSwitchConfig(BaseModel):
    """某一层级的Switch配置（支持多层Switch，如Leaf-Spine）"""
    enabled: bool = False  # 是否启用该层级的Switch
    layers: List[SwitchLayerConfig] = []  # Switch层列表（从下到上，如[leaf, spine]）
    downlink_redundancy: int = 1  # 下层设备连接几个Switch（冗余度）
    connect_to_upper_level: bool = True  # 是否连接到上层的Switch
    # 无Switch时的直连拓扑类型
    direct_topology: Literal['none', 'full_mesh', 'full_mesh_2d', 'ring', 'torus_2d', 'torus_3d'] = 'none'
    # Switch与下层节点的连接模式
    connection_mode: SwitchConnectionMode = SwitchConnectionMode.ROUND_ROBIN
    # 分组连接配置（connection_mode为GROUP时使用）
    group_config: Optional[SwitchGroupConfig] = None
    # 自定义连接配置（connection_mode为CUSTOM时使用）
    custom_connections: Optional[List[SwitchCustomConnection]] = None


class GlobalSwitchConfig(BaseModel):
    """全局Switch配置"""
    switch_types: List[SwitchTypeConfig] = [
        SwitchTypeConfig(id="leaf_48", name="48端口Leaf交换机", port_count=48),
        SwitchTypeConfig(id="leaf_72", name="72端口Leaf交换机", port_count=72),
        SwitchTypeConfig(id="spine_128", name="128端口Spine交换机", port_count=128),
        SwitchTypeConfig(id="core_512", name="512端口核心交换机", port_count=512),
    ]
    datacenter_level: HierarchyLevelSwitchConfig = HierarchyLevelSwitchConfig()  # Pod间
    pod_level: HierarchyLevelSwitchConfig = HierarchyLevelSwitchConfig()  # Rack间
    rack_level: HierarchyLevelSwitchConfig = HierarchyLevelSwitchConfig()  # Board间
    board_level: HierarchyLevelSwitchConfig = HierarchyLevelSwitchConfig()  # Chip间


class SwitchInstance(BaseModel):
    """Switch实例（出现在拓扑数据中）"""
    id: str  # 唯一标识，如 "dc_spine_0", "pod_0/leaf_1"
    type_id: str  # Switch类型ID
    layer: str  # 所在层，如 "leaf", "spine"
    hierarchy_level: Literal['datacenter', 'pod', 'rack']  # 所属层级
    parent_id: Optional[str] = None  # 父节点ID（如pod_0）
    label: str  # 显示标签
    uplink_ports_used: int = 0  # 上行端口使用数
    downlink_ports_used: int = 0  # 下行端口使用数
    inter_ports_used: int = 0  # 同层互联端口使用数


class HierarchicalTopology(BaseModel):
    """完整的层级拓扑数据"""
    pods: List[PodConfig] = []
    connections: List[ConnectionConfig] = []
    switches: List[SwitchInstance] = []  # Switch实例列表
    switch_config: Optional[GlobalSwitchConfig] = None  # Switch配置（保存用）
    manual_connections: Optional[ManualConnectionConfig] = None  # 手动连接配置


# ============================================
# API请求/响应模型
# ============================================

class ChipCountConfig(BaseModel):
    """各类型芯片数量配置"""
    npu: int = Field(default=8, ge=0, le=32)
    cpu: int = Field(default=0, ge=0, le=32)


class BoardTypeConfig(BaseModel):
    """单个U高度板卡的完整配置"""
    count: int = Field(default=0, ge=0, description="板卡数量")
    chips: ChipCountConfig = Field(default_factory=lambda: ChipCountConfig(npu=8, cpu=0))


class BoardCountConfig(BaseModel):
    """各U高度板卡数量配置（兼容旧格式）"""
    u1: int = Field(default=0, ge=0, le=42, description="1U板卡数量")
    u2: int = Field(default=8, ge=0, le=21, description="2U板卡数量")
    u4: int = Field(default=0, ge=0, le=10, description="4U板卡数量")


class BoardConfigByType(BaseModel):
    """按U高度分类的板卡配置"""
    u1: BoardTypeConfig = Field(default_factory=lambda: BoardTypeConfig(count=0, chips=ChipCountConfig(npu=2, cpu=0)))
    u2: BoardTypeConfig = Field(default_factory=lambda: BoardTypeConfig(count=8, chips=ChipCountConfig(npu=8, cpu=0)))
    u4: BoardTypeConfig = Field(default_factory=lambda: BoardTypeConfig(count=0, chips=ChipCountConfig(npu=16, cpu=2)))


# ============================================
# 灵活Rack配置模型
# ============================================

class FlexBoardChipConfig(BaseModel):
    """灵活板卡中的Chip配置"""
    name: str = Field(..., description="Chip名称，如 NPU, CPU, GPU")
    count: int = Field(default=1, ge=1, le=64, description="Chip数量")


class FlexBoardConfig(BaseModel):
    """灵活板卡配置"""
    id: str = Field(..., description="板卡唯一ID")
    name: str = Field(..., description="板卡名称")
    u_height: int = Field(default=2, ge=1, le=10, description="U高度")
    count: int = Field(default=1, ge=0, le=42, description="板卡数量")
    chips: List[FlexBoardChipConfig] = Field(default_factory=list, description="Chip配置列表")


class FlexRackConfig(BaseModel):
    """灵活Rack配置"""
    total_u: int = Field(default=42, ge=10, le=60, description="Rack总U数")
    boards: List[FlexBoardConfig] = Field(default_factory=list, description="板卡配置列表")


class TopologyGenerateRequest(BaseModel):
    """拓扑生成请求"""
    pod_count: int = Field(default=1, ge=1, le=10)
    racks_per_pod: int = Field(default=4, ge=1, le=64)
    board_counts: Optional[BoardCountConfig] = None  # 旧格式，保持兼容
    board_configs: Optional[BoardConfigByType] = None  # 新格式：按U高度配置Chip
    rack_config: Optional[FlexRackConfig] = None  # 最新格式：灵活Rack配置
    chip_types: List[ChipType] = ['npu', 'cpu']
    chip_counts: Optional[ChipCountConfig] = None  # 旧格式，保持兼容
    switch_config: Optional[GlobalSwitchConfig] = None  # Switch配置
    manual_connections: Optional[ManualConnectionConfig] = None  # 手动连接配置


class ViewDataRequest(BaseModel):
    """视图数据请求"""
    level: Literal['pod', 'rack', 'board', 'chip']
    path: List[str] = []  # 路径，如 ['pod_0', 'rack_1']


class SavedConfig(BaseModel):
    """保存的配置"""
    name: str = Field(..., description="配置名称")
    description: Optional[str] = Field(default=None, description="配置描述")
    pod_count: int = Field(default=1, ge=1, le=10)
    racks_per_pod: int = Field(default=4, ge=1, le=64)
    board_configs: BoardConfigByType
    switch_config: Optional[GlobalSwitchConfig] = None  # Switch配置
    manual_connections: Optional[ManualConnectionConfig] = None  # 手动连接配置
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
