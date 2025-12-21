"""
分析API
"""

from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.database import ResultManager

router = APIRouter()

# 获取数据库管理器
db_manager = ResultManager()


# ==================== Pydantic模型 ====================

class SensitivityDataPoint(BaseModel):
    """敏感性分析数据点"""
    value: Any
    mean_performance: float
    min_performance: float
    max_performance: float
    count: int


class SensitivityResponse(BaseModel):
    """参数敏感性响应"""
    parameter: str
    data: List[SensitivityDataPoint]


class CompareRequest(BaseModel):
    """实验对比请求"""
    experiment_ids: List[int]


class CompareResponse(BaseModel):
    """实验对比响应"""
    experiments: List[Dict[str, Any]]
    best_configs: List[Dict[str, Any]]


class TrafficCompareResponse(BaseModel):
    """按数据流对比响应"""
    traffic_files: List[str]
    experiments: List[Dict[str, Any]]
    param_keys: List[str]
    data: List[Dict[str, Any]]


# ==================== API端点 ====================

@router.get("/experiments/{experiment_id}/sensitivity/{parameter}", response_model=SensitivityResponse)
async def get_parameter_sensitivity(
    experiment_id: int,
    parameter: str,
    metric: str = Query(None, description="性能指标名（默认使用performance，可指定config_params中的字段）"),
):
    """
    获取单个参数的敏感性分析

    返回该参数不同取值对应的平均性能、最小/最大性能和样本数
    """
    # 检查实验是否存在
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    # 检查参数是否在该实验的参数列表中
    param_keys = db_manager.db.get_param_keys(experiment_id)
    if parameter not in param_keys:
        raise HTTPException(
            status_code=400,
            detail=f"无效的参数名: {parameter}。该实验有效参数: {param_keys}",
        )

    try:
        result = db_manager.get_parameter_sensitivity(experiment_id, parameter, metric)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/experiments/{experiment_id}/sensitivity")
async def get_all_sensitivity(
    experiment_id: int,
    metric: str = Query(None, description="性能指标名（默认使用performance）"),
):
    """获取所有参数的敏感性分析"""
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    result = db_manager.get_all_parameter_sensitivity(experiment_id, metric)
    return {"experiment_id": experiment_id, "metric": metric or "performance", "parameters": result}


@router.get("/experiments/{experiment_id}/influence")
async def get_parameter_influence(
    experiment_id: int,
    metric: str = Query(None, description="性能指标名（默认使用performance）"),
):
    """
    获取各参数对性能的影响度（基于方差贡献）

    使用方差分解方法计算每个参数对性能变化的贡献比例：
    影响度 = Var(各取值的平均性能) / Var(所有性能)

    返回按影响度从高到低排序的参数列表
    """
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    result = db_manager.get_parameter_influence(experiment_id, metric)
    return {"experiment_id": experiment_id, **result}


@router.post("/compare", response_model=CompareResponse)
async def compare_experiments(request: CompareRequest):
    """
    对比多个实验

    返回各实验的基本信息和最佳配置对比
    注意：只能对比同类型的实验（都是 NoC 或都是 D2D）
    """
    if len(request.experiment_ids) < 2:
        raise HTTPException(status_code=400, detail="至少需要2个实验进行对比")

    if len(request.experiment_ids) > 10:
        raise HTTPException(status_code=400, detail="最多支持10个实验对比")

    # 验证所有实验是否存在
    for exp_id in request.experiment_ids:
        experiment = db_manager.get_experiment(exp_id)
        if not experiment:
            raise HTTPException(status_code=404, detail=f"实验不存在: ID={exp_id}")

    try:
        result = db_manager.compare_experiments(request.experiment_ids)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/experiments/{experiment_id}/heatmap")
async def get_parameter_heatmap(
    experiment_id: int,
    param_x: str = Query(..., description="X轴参数"),
    param_y: str = Query(..., description="Y轴参数"),
    metric: str = Query(None, description="性能指标名（默认使用performance，可指定config_params中的字段）"),
):
    """
    获取两个参数的性能热图数据

    返回param_x和param_y的组合对应的平均性能
    """
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    # 检查参数是否在该实验的参数列表中
    param_keys = db_manager.db.get_param_keys(experiment_id)
    if param_x not in param_keys:
        raise HTTPException(status_code=400, detail=f"无效的X轴参数: {param_x}")
    if param_y not in param_keys:
        raise HTTPException(status_code=400, detail=f"无效的Y轴参数: {param_y}")
    if param_x == param_y:
        raise HTTPException(status_code=400, detail="X轴和Y轴参数不能相同")

    # 轻量级查询，只获取 config_params 和 performance 字段
    results = db_manager.db.get_results_for_analysis(experiment_id)

    # 按参数组合分组统计
    groups = {}
    for result in results:
        config_params = result.get("config_params") or {}
        x_val = config_params.get(param_x)
        y_val = config_params.get(param_y)
        if x_val is not None and y_val is not None:
            # 获取性能值
            if metric:
                perf_val = config_params.get(metric)
                if perf_val is None or not isinstance(perf_val, (int, float)):
                    continue
            else:
                perf_val = result.get("performance")

            if perf_val is not None:
                key = (x_val, y_val)
                if key not in groups:
                    groups[key] = []
                groups[key].append(perf_val)

    data = []
    x_values = set()
    y_values = set()
    for (x_val, y_val), perfs in groups.items():
        x_values.add(x_val)
        y_values.add(y_val)
        data.append({
            param_x: x_val,
            param_y: y_val,
            "mean_performance": sum(perfs) / len(perfs),
            "min_performance": min(perfs),
            "max_performance": max(perfs),
            "count": len(perfs),
        })

    return {
        "param_x": param_x,
        "param_y": param_y,
        "metric": metric or "performance",
        "x_values": sorted(x_values, key=lambda x: x if isinstance(x, (int, float)) else 0),
        "y_values": sorted(y_values, key=lambda x: x if isinstance(x, (int, float)) else 0),
        "data": data,
    }


@router.post("/compare/traffic", response_model=TrafficCompareResponse)
async def compare_experiments_by_traffic(request: CompareRequest):
    """
    按数据流对比多个实验的性能（支持多指标）

    返回每个数据流在各实验中的平均/最大/最小性能值
    """
    if len(request.experiment_ids) < 2:
        raise HTTPException(status_code=400, detail="至少需要2个实验进行对比")

    if len(request.experiment_ids) > 10:
        raise HTTPException(status_code=400, detail="最多支持10个实验对比")

    # 验证所有实验是否存在
    for exp_id in request.experiment_ids:
        experiment = db_manager.get_experiment(exp_id)
        if not experiment:
            raise HTTPException(status_code=404, detail=f"实验不存在: ID={exp_id}")

    try:
        result = db_manager.compare_by_traffic(request.experiment_ids)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== 分析图表配置 API ====================

class ChartConfigRequest(BaseModel):
    """图表配置请求"""
    name: str
    chart_type: str  # line / heatmap / sensitivity
    config: Dict[str, Any]  # 图表配置（参数、指标等）
    sort_order: int = 0


class ChartConfigUpdateRequest(BaseModel):
    """图表配置更新请求"""
    name: str = None
    config: Dict[str, Any] = None
    sort_order: int = None


class ChartConfigResponse(BaseModel):
    """图表配置响应"""
    id: int
    name: str
    chart_type: str
    config: Dict[str, Any]
    sort_order: int
    created_at: str = None
    updated_at: str = None


@router.get("/experiments/{experiment_id}/charts", response_model=List[ChartConfigResponse])
async def get_analysis_charts(experiment_id: int):
    """获取实验的所有保存的分析图表配置"""
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    charts = db_manager.db.get_analysis_charts(experiment_id)
    return charts


@router.post("/experiments/{experiment_id}/charts", response_model=ChartConfigResponse)
async def add_analysis_chart(experiment_id: int, request: ChartConfigRequest):
    """添加分析图表配置"""
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    if request.chart_type not in ["line", "heatmap", "sensitivity"]:
        raise HTTPException(status_code=400, detail="无效的图表类型，必须是 line/heatmap/sensitivity")

    chart_id = db_manager.db.add_analysis_chart(
        experiment_id=experiment_id,
        name=request.name,
        chart_type=request.chart_type,
        config=request.config,
        sort_order=request.sort_order,
    )

    charts = db_manager.db.get_analysis_charts(experiment_id)
    return next(c for c in charts if c["id"] == chart_id)


@router.put("/experiments/{experiment_id}/charts/{chart_id}", response_model=ChartConfigResponse)
async def update_analysis_chart(experiment_id: int, chart_id: int, request: ChartConfigUpdateRequest):
    """更新分析图表配置"""
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    success = db_manager.db.update_analysis_chart(
        chart_id=chart_id,
        name=request.name,
        config=request.config,
        sort_order=request.sort_order,
    )

    if not success:
        raise HTTPException(status_code=404, detail="图表配置不存在")

    charts = db_manager.db.get_analysis_charts(experiment_id)
    return next(c for c in charts if c["id"] == chart_id)


@router.delete("/experiments/{experiment_id}/charts/{chart_id}")
async def delete_analysis_chart(experiment_id: int, chart_id: int):
    """删除分析图表配置"""
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    success = db_manager.db.delete_analysis_chart(chart_id)
    if not success:
        raise HTTPException(status_code=404, detail="图表配置不存在")

    return {"success": True}
