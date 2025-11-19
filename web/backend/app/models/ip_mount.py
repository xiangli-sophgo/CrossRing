from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class IPMountRequest(BaseModel):
    """IP挂载请求模型"""
    node_ids: List[int] = Field(..., description="节点ID列表，支持单个或批量")
    ip_type: str = Field(..., description="IP类型，如 gdma_0, npu_0, ddr_0 等")
    topology: str = Field(..., description="拓扑类型，如 5x4")

    @field_validator('node_ids')
    @classmethod
    def validate_node_ids(cls, v):
        if not v:
            raise ValueError("节点ID列表不能为空")
        if len(v) != len(set(v)):
            raise ValueError("节点ID列表包含重复值")
        return v

    @field_validator('ip_type')
    @classmethod
    def validate_ip_type(cls, v):
        # 验证IP类型格式：类型_编号
        if '_' not in v:
            raise ValueError("IP类型格式错误，应为 '类型_编号'，如 gdma_0")
        return v


class IPMount(BaseModel):
    """IP挂载记录模型"""
    node_id: int
    ip_type: str
    topology: str
    position: dict = Field(default_factory=dict, description="节点位置 {row, col}")


class IPMountResponse(BaseModel):
    """IP挂载响应模型"""
    success: bool
    message: str
    mounted_ips: List[IPMount]


class IPMountListResponse(BaseModel):
    """IP挂载列表响应模型"""
    topology: str
    mounts: List[IPMount]
    total: int


class BatchMountRequest(BaseModel):
    """批量挂载请求模型"""
    node_range: str = Field(..., description="节点范围，支持混合格式：'0-3', '5,7,9', '0-3,5,7-9'")
    ip_type_prefix: str = Field(..., description="IP类型前缀，如 gdma, npu")
    topology: str = Field(..., description="拓扑类型")

    @field_validator('node_range')
    @classmethod
    def validate_node_range(cls, v):
        """验证节点范围格式，支持混合格式如 '0-3,5,7-9'"""
        try:
            # 尝试解析所有部分
            parts = v.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:
                    # 范围格式
                    range_parts = part.split('-')
                    if len(range_parts) != 2:
                        raise ValueError(f"范围格式错误: {part}")
                    start, end = int(range_parts[0]), int(range_parts[1])
                    if start >= end:
                        raise ValueError(f"范围起始值必须小于结束值: {part}")
                else:
                    # 单个数字
                    int(part)
        except ValueError as e:
            raise ValueError(f"节点范围格式错误: {str(e)}")
        return v

    def get_node_ids(self) -> List[int]:
        """解析节点范围为节点ID列表，支持混合格式"""
        node_ids = []
        parts = self.node_range.split(',')

        for part in parts:
            part = part.strip()
            if '-' in part:
                # 范围格式：0-3
                start, end = map(int, part.split('-'))
                node_ids.extend(range(start, end + 1))
            else:
                # 单个数字
                node_ids.append(int(part))

        # 去重并排序
        return sorted(list(set(node_ids)))
