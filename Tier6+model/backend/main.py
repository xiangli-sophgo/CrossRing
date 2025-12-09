"""
Tier6+ 3D拓扑配置工具 - 后端API
"""

import sys
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from topology import TopologyGenerator, LEVEL_COLORS, LEVEL_Z_POSITIONS

app = FastAPI(
    title="Tier6+ 3D拓扑配置器",
    description="交互式配置多层级网络拓扑",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LevelConfig(BaseModel):
    """层级配置"""
    level: str
    count: int
    topology: str = "mesh"
    visible: bool = True


class TopologyRequest(BaseModel):
    """拓扑请求"""
    levels: List[LevelConfig]
    show_inter_level: bool = True
    layout: str = "circular"


class NodeData(BaseModel):
    """节点数据"""
    id: str
    level: str
    position: List[float]
    color: str


class EdgeData(BaseModel):
    """边数据"""
    source: str
    target: str
    type: str  # intra_level 或 inter_level


class TopologyResponse(BaseModel):
    """拓扑响应"""
    nodes: List[NodeData]
    edges: List[EdgeData]


@app.get("/")
async def root():
    return {"message": "Tier6+ 3D拓扑配置器 API"}


@app.post("/api/topology/generate", response_model=TopologyResponse)
async def generate_topology(request: TopologyRequest):
    """生成3D拓扑数据"""
    generator = TopologyGenerator()
    return generator.generate(
        levels=request.levels,
        show_inter_level=request.show_inter_level,
        layout=request.layout
    )


@app.get("/api/topology/presets")
async def get_presets():
    """获取预设配置"""
    return {
        "presets": [
            {
                "name": "小型系统",
                "description": "1 Pod, 2 Servers, 4 Boards",
                "levels": [
                    {"level": "die", "count": 8, "topology": "mesh"},
                    {"level": "chip", "count": 4, "topology": "mesh"},
                    {"level": "board", "count": 4, "topology": "mesh"},
                    {"level": "server", "count": 2, "topology": "mesh"},
                    {"level": "pod", "count": 1, "topology": "mesh"},
                ]
            },
            {
                "name": "中型系统",
                "description": "2 Pods, 4 Servers, 8 Boards",
                "levels": [
                    {"level": "die", "count": 16, "topology": "mesh"},
                    {"level": "chip", "count": 8, "topology": "mesh"},
                    {"level": "board", "count": 8, "topology": "mesh"},
                    {"level": "server", "count": 4, "topology": "mesh"},
                    {"level": "pod", "count": 2, "topology": "mesh"},
                ]
            },
            {
                "name": "大型系统",
                "description": "4 Pods, 8 Servers, 16 Boards",
                "levels": [
                    {"level": "die", "count": 32, "topology": "all_to_all"},
                    {"level": "chip", "count": 16, "topology": "mesh"},
                    {"level": "board", "count": 16, "topology": "mesh"},
                    {"level": "server", "count": 8, "topology": "mesh"},
                    {"level": "pod", "count": 4, "topology": "mesh"},
                ]
            },
        ]
    }


@app.get("/api/topology/colors")
async def get_colors():
    """获取层级颜色配置"""
    return LEVEL_COLORS


@app.get("/api/topology/z-positions")
async def get_z_positions():
    """获取层级Z轴位置"""
    return LEVEL_Z_POSITIONS


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8100, reload=True)
