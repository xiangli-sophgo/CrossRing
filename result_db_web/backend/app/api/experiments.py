"""
实验管理API
"""

import os
from typing import Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from src.database import ResultManager

router = APIRouter()

# 获取数据库管理器
db_manager = ResultManager()


# ==================== Pydantic模型 ====================

class ExperimentCreate(BaseModel):
    """创建实验请求"""
    name: str
    experiment_type: str = "noc"  # "noc" 或 "d2d"
    description: Optional[str] = None
    topo_type: Optional[str] = None


class ExperimentUpdate(BaseModel):
    """更新实验请求"""
    description: Optional[str] = None
    notes: Optional[str] = None


class ExperimentResponse(BaseModel):
    """实验响应"""
    id: int
    name: str
    experiment_type: str
    created_at: Optional[str]
    description: Optional[str]
    config_path: Optional[str]
    topo_type: Optional[str]
    traffic_files: Optional[List[str]]
    traffic_weights: Optional[List[float]]
    simulation_time: Optional[int]
    n_repeats: Optional[int]
    n_jobs: Optional[int]
    status: Optional[str]
    total_combinations: Optional[int]
    completed_combinations: Optional[int]
    best_performance: Optional[float]
    git_commit: Optional[str]
    notes: Optional[str]


class ImportResponse(BaseModel):
    """CSV导入响应"""
    experiment_id: int
    imported_count: int
    errors: List[str]


# ==================== API端点 ====================

@router.get("/experiments", response_model=List[ExperimentResponse])
async def list_experiments(
    status: Optional[str] = None,
    experiment_type: Optional[str] = None,
):
    """
    获取实验列表

    - status: 筛选状态
    - experiment_type: 筛选类型 ("noc" 或 "d2d")
    """
    experiments = db_manager.list_experiments(status, experiment_type)
    return experiments


@router.post("/experiments", response_model=ExperimentResponse)
async def create_experiment(request: ExperimentCreate):
    """创建新实验"""
    try:
        experiment_id = db_manager.create_experiment(
            name=request.name,
            experiment_type=request.experiment_type,
            description=request.description,
            topo_type=request.topo_type,
        )
        experiment = db_manager.get_experiment(experiment_id)
        return experiment
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: int):
    """获取实验详情"""
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")
    return experiment


@router.put("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def update_experiment(experiment_id: int, request: ExperimentUpdate):
    """更新实验信息"""
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    db_manager.update_experiment(
        experiment_id,
        description=request.description,
        notes=request.notes,
    )

    return db_manager.get_experiment(experiment_id)


@router.delete("/experiments/{experiment_id}")
async def delete_experiment(experiment_id: int):
    """删除实验"""
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    db_manager.delete_experiment(experiment_id)
    return {"message": "实验已删除", "id": experiment_id}


@router.post("/experiments/import", response_model=ImportResponse)
async def import_from_csv(
    file: UploadFile = File(...),
    experiment_name: str = Form(...),
    experiment_type: str = Form("noc"),
    description: Optional[str] = Form(None),
    topo_type: Optional[str] = Form(None),
):
    """
    从CSV文件导入实验数据

    - file: CSV文件
    - experiment_name: 实验名称
    - experiment_type: 实验类型 ("noc" 或 "d2d")
    - description: 实验描述
    - topo_type: 拓扑类型
    """
    # 保存上传的文件
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        result = db_manager.import_from_csv(
            csv_path=tmp_path,
            experiment_name=experiment_name,
            experiment_type=experiment_type,
            description=description,
            topo_type=topo_type,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # 清理临时文件
        os.unlink(tmp_path)
