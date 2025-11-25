"""
结果查询API
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.database import ResultManager

router = APIRouter()

# 获取数据库管理器
db_manager = ResultManager()


# ==================== Pydantic模型 ====================

class ResultResponse(BaseModel):
    """单条结果响应"""
    id: int
    experiment_id: int
    created_at: Optional[str]
    performance: float
    config_params: Dict[str, Any]
    result_details: Dict[str, Any]
    error: Optional[str] = None


class ResultsPageResponse(BaseModel):
    """分页结果响应"""
    results: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int
    total_pages: int


class StatisticsResponse(BaseModel):
    """统计信息响应"""
    experiment: Dict[str, Any]
    result_count: int
    performance_distribution: Dict[str, Any]
    param_keys: List[str]


class DistributionResponse(BaseModel):
    """性能分布响应"""
    min: float
    max: float
    mean: float
    count: int
    histogram: List[List[float]]


# ==================== API端点 ====================

@router.get("/experiments/{experiment_id}/results", response_model=ResultsPageResponse)
async def get_results(
    experiment_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    sort_by: str = Query("performance"),
    order: str = Query("desc", regex="^(asc|desc)$"),
):
    """
    分页获取实验结果

    - page: 页码（从1开始）
    - page_size: 每页数量（1-1000）
    - sort_by: 排序字段
    - order: 排序方向（asc/desc）
    """
    # 检查实验是否存在
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    # 获取结果
    result = db_manager.get_results(
        experiment_id=experiment_id,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        order=order,
    )

    return result


@router.get("/experiments/{experiment_id}/best")
async def get_best_results(
    experiment_id: int,
    limit: int = Query(10, ge=1, le=100),
):
    """获取最佳配置"""
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    results = db_manager.get_best_results(experiment_id, limit)
    return {"results": results, "count": len(results)}


@router.get("/experiments/{experiment_id}/stats", response_model=StatisticsResponse)
async def get_statistics(experiment_id: int):
    """获取实验统计信息"""
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    stats = db_manager.get_statistics(experiment_id)
    return stats


@router.get("/experiments/{experiment_id}/distribution")
async def get_distribution(
    experiment_id: int,
    bins: int = Query(50, ge=10, le=200),
):
    """获取性能分布数据"""
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    distribution = db_manager.db.get_performance_distribution(experiment_id, bins)
    return distribution


@router.get("/experiments/{experiment_id}/param-keys")
async def get_param_keys(experiment_id: int):
    """获取实验的配置参数键名列表（用于动态生成表格列）"""
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    param_keys = db_manager.db.get_param_keys(experiment_id)
    return {
        "param_keys": param_keys,
        "count": len(param_keys),
    }
