"""
结果查询API
"""

import os
import subprocess
import sys
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
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
    result_html: Optional[str] = None
    result_files: Optional[List[str]] = None
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


@router.get("/experiments/{experiment_id}/traffic-stats")
async def get_traffic_stats(experiment_id: int):
    """
    按数据流文件分组统计实验结果

    返回每个数据流文件的结果数量和平均性能
    """
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    # 获取所有结果
    results, total = db_manager.db.get_results(experiment_id, page=1, page_size=100000)

    # 按数据流名称分组统计
    traffic_groups = {}
    for result in results:
        config_params = result.get("config_params", {})
        # 尝试获取数据流名称字段
        traffic_name = config_params.get("数据流名称") or config_params.get("file_name") or "未知"

        if traffic_name not in traffic_groups:
            traffic_groups[traffic_name] = {
                "count": 0,
                "performances": [],
            }

        traffic_groups[traffic_name]["count"] += 1
        performance = result.get("performance", 0)
        if performance is not None:
            traffic_groups[traffic_name]["performances"].append(performance)

    # 计算统计值
    stats = []
    for traffic_name, group in traffic_groups.items():
        performances = group["performances"]
        avg_performance = sum(performances) / len(performances) if performances else 0
        max_performance = max(performances) if performances else 0
        min_performance = min(performances) if performances else 0

        stats.append({
            "traffic_name": traffic_name,
            "count": group["count"],
            "avg_performance": avg_performance,
            "max_performance": max_performance,
            "min_performance": min_performance,
        })

    # 按结果数量降序排序
    stats.sort(key=lambda x: x["count"], reverse=True)

    return {
        "experiment_id": experiment_id,
        "total_results": total,
        "traffic_stats": stats,
    }


@router.get("/results/{result_id}/html", response_class=HTMLResponse)
async def get_result_html(result_id: int, experiment_id: int):
    """
    获取结果HTML报告内容

    - result_id: 结果ID
    - experiment_id: 实验ID（用于确定结果表）
    """
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    # 获取结果
    results_data = db_manager.get_results(
        experiment_id=experiment_id,
        page=1,
        page_size=100000,
    )

    # 查找指定result_id的结果
    target_result = None
    for result in results_data["results"]:
        if result["id"] == result_id:
            target_result = result
            break

    if not target_result:
        raise HTTPException(status_code=404, detail="结果不存在")

    result_html = target_result.get("result_html")
    if not result_html:
        raise HTTPException(status_code=404, detail="该结果没有HTML报告")

    return HTMLResponse(content=result_html)


class OpenFileRequest(BaseModel):
    """打开文件请求"""
    path: str


@router.post("/open-file")
async def open_local_file(request: OpenFileRequest):
    """
    打开本地文件（调用系统默认程序）

    - path: 文件路径
    """
    file_path = request.path

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"文件不存在: {file_path}")

    try:
        if sys.platform == "win32":
            os.startfile(file_path)
        elif sys.platform == "darwin":
            subprocess.run(["open", file_path], check=True)
        else:
            subprocess.run(["xdg-open", file_path], check=True)

        return {"success": True, "message": f"已打开文件: {file_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"打开文件失败: {str(e)}")
