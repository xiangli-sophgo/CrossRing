"""
FastAPI 接口模块

提供 LLM 推理模拟的 REST API 接口。
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

from .simulator import run_simulation


# ============================================
# Pydantic 模型
# ============================================

class SimulationRequest(BaseModel):
    """模拟请求"""
    topology: dict[str, Any]
    model: dict[str, Any]
    inference: dict[str, Any]
    parallelism: dict[str, Any]
    hardware: dict[str, Any]
    config: dict[str, Any] | None = None


class SimulationResponse(BaseModel):
    """模拟响应"""
    ganttChart: dict[str, Any]
    stats: dict[str, Any]
    timestamp: float


# ============================================
# FastAPI 应用
# ============================================

app = FastAPI(
    title="LLM 推理模拟器 API",
    description="基于拓扑的 GPU/加速器侧精细模拟服务",
    version="1.0.0",
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "LLM 推理模拟器 API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


@app.post("/api/simulate", response_model=SimulationResponse)
async def simulate(request: SimulationRequest):
    """
    运行 LLM 推理模拟

    Args:
        request: 模拟请求，包含拓扑、模型、推理、并行策略、硬件配置

    Returns:
        模拟结果，包含甘特图数据和统计信息
    """
    try:
        result = run_simulation(
            topology_dict=request.topology,
            model_dict=request.model,
            inference_dict=request.inference,
            parallelism_dict=request.parallelism,
            hardware_dict=request.hardware,
            config_dict=request.config,
        )
        return SimulationResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模拟失败: {str(e)}")


@app.post("/api/validate")
async def validate_config(request: SimulationRequest):
    """
    验证配置是否有效

    检查：
    - 拓扑中芯片数量是否满足并行策略需求
    - 硬件配置是否完整
    - 模型配置是否合理
    """
    try:
        topology = request.topology
        parallelism = request.parallelism

        # 计算所需芯片数
        required_chips = (
            parallelism.get("dp", 1) *
            parallelism.get("tp", 1) *
            parallelism.get("pp", 1) *
            parallelism.get("ep", 1)
        )

        # 计算拓扑中的芯片数
        available_chips = 0
        for pod in topology.get("pods", []):
            for rack in pod.get("racks", []):
                for board in rack.get("boards", []):
                    available_chips += len(board.get("chips", []))

        if available_chips < required_chips:
            return {
                "valid": False,
                "error": f"芯片数量不足: 需要 {required_chips} 个，拓扑中只有 {available_chips} 个",
            }

        return {
            "valid": True,
            "required_chips": required_chips,
            "available_chips": available_chips,
        }

    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
        }
