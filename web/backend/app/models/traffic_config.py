from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
from datetime import datetime


class TrafficConfigBase(BaseModel):
    """流量配置基础模型"""
    source_ip: str = Field(..., description="源IP类型，如 gdma_0")
    target_ip: str = Field(..., description="目标IP类型，如 ddr_0")
    speed_gbps: float = Field(..., gt=0, description="速度 (GB/s)，必须大于0")
    burst_length: int = Field(..., gt=0, le=16, description="Burst长度，范围1-16")
    request_type: Literal["R", "W"] = Field(..., description="请求类型：R(读) 或 W(写)")
    end_time_ns: int = Field(..., gt=0, description="结束时间 (ns)，必须大于0")

    @field_validator('source_ip', 'target_ip')
    @classmethod
    def validate_ip_format(cls, v):
        """验证IP格式：类型_编号"""
        if '_' not in v:
            raise ValueError("IP格式错误，应为 '类型_编号'，如 gdma_0")
        return v


class TrafficConfig(TrafficConfigBase):
    """流量配置完整模型（含ID和时间戳）"""
    id: str = Field(..., description="配置唯一ID")
    topology: str = Field(..., description="拓扑类型")
    mode: Literal["noc", "d2d"] = Field(default="noc", description="流量模式：noc 或 d2d")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")


class TrafficConfigCreate(TrafficConfigBase):
    """创建流量配置请求模型"""
    topology: str = Field(..., description="拓扑类型")
    mode: Literal["noc", "d2d"] = Field(default="noc", description="流量模式")


class D2DTrafficConfigBase(TrafficConfigBase):
    """D2D流量配置基础模型"""
    source_die: int = Field(..., ge=0, description="源Die ID")
    target_die: int = Field(..., ge=0, description="目标Die ID")

    @field_validator('source_die', 'target_die')
    @classmethod
    def validate_die_id(cls, v):
        """验证Die ID范围"""
        if v < 0:
            raise ValueError("Die ID 必须 >= 0")
        return v


class D2DTrafficConfig(D2DTrafficConfigBase):
    """D2D流量配置完整模型"""
    id: str
    topology: str
    mode: Literal["d2d"] = "d2d"
    created_at: datetime = Field(default_factory=datetime.now)


class D2DTrafficConfigCreate(D2DTrafficConfigBase):
    """创建D2D流量配置请求模型"""
    topology: str
    mode: Literal["d2d"] = "d2d"


class TrafficConfigListResponse(BaseModel):
    """流量配置列表响应"""
    topology: str
    mode: str
    configs: List[TrafficConfig]
    total: int


class TrafficConfigResponse(BaseModel):
    """流量配置操作响应"""
    success: bool
    message: str
    config: Optional[TrafficConfig] = None


class BatchTrafficConfigCreate(BaseModel):
    """批量流量配置创建请求"""
    topology: str
    mode: Literal["noc", "d2d"] = "noc"
    source_ips: List[str] = Field(..., description="源IP列表")
    target_ips: List[str] = Field(..., description="目标IP列表")
    speed_gbps: float = Field(..., gt=0)
    burst_length: int = Field(..., gt=0, le=16)
    request_type: Literal["R", "W"]
    end_time_ns: int = Field(..., gt=0)

    # D2D模式专用
    source_die: Optional[int] = Field(None, ge=0, description="源Die ID (仅D2D模式)")
    target_die: Optional[int] = Field(None, ge=0, description="目标Die ID (仅D2D模式)")

    @field_validator('source_ips', 'target_ips')
    @classmethod
    def validate_ip_lists(cls, v):
        """验证IP列表不为空"""
        if not v:
            raise ValueError("IP列表不能为空")
        return v
