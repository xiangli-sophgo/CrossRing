"""
Tier6+互联拓扑 API

提供层级拓扑数据的生成和查询接口
Pod -> Rack -> Board -> Chip
"""

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import os
from datetime import datetime

from models import HierarchicalTopology, PodConfig, RackConfig, BoardConfig, ConnectionConfig, TopologyGenerateRequest, SavedConfig
from topology import HierarchicalTopologyGenerator

# 配置文件存储路径
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "saved_configs")
os.makedirs(CONFIG_DIR, exist_ok=True)


app = FastAPI(title="Tier6+互联拓扑API", description="提供Pod/Rack/Board/Chip层级拓扑的生成和查询", version="2.0.0")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局拓扑生成器实例
generator = HierarchicalTopologyGenerator()


# ============================================
# API 接口
# ============================================


@app.get("/")
async def root():
    return {"message": "Tier6+互联拓扑 API", "version": "2.0.0"}


@app.get("/api/topology", response_model=HierarchicalTopology)
async def get_topology():
    """获取完整的层级拓扑数据"""
    return generator.get_cached_topology()


@app.post("/api/topology/generate", response_model=HierarchicalTopology)
async def generate_topology(request: TopologyGenerateRequest):
    """根据配置生成新的拓扑"""
    # 转换为字典格式
    chip_counts = None
    if request.chip_counts:
        chip_counts = request.chip_counts.model_dump()

    board_counts = None
    if request.board_counts:
        board_counts = request.board_counts.model_dump()

    board_configs = None
    if request.board_configs:
        board_configs = {
            'u1': {
                'count': request.board_configs.u1.count,
                'chips': request.board_configs.u1.chips.model_dump()
            },
            'u2': {
                'count': request.board_configs.u2.count,
                'chips': request.board_configs.u2.chips.model_dump()
            },
            'u4': {
                'count': request.board_configs.u4.count,
                'chips': request.board_configs.u4.chips.model_dump()
            },
        }

    switch_config = None
    if request.switch_config:
        switch_config = request.switch_config.model_dump()

    result = generator.generate(
        pod_count=request.pod_count,
        racks_per_pod=request.racks_per_pod,
        board_counts=board_counts,
        chip_types=request.chip_types,
        chip_counts=chip_counts,
        board_configs=board_configs,
        switch_config=switch_config,
    )

    return result


@app.get("/api/topology/pod/{pod_id}", response_model=PodConfig)
async def get_pod(pod_id: str):
    """获取指定Pod的数据"""
    pod = generator.get_pod(pod_id)
    if pod is None:
        raise HTTPException(status_code=404, detail=f"Pod '{pod_id}' not found")
    return pod


@app.get("/api/topology/rack/{rack_id:path}", response_model=RackConfig)
async def get_rack(rack_id: str):
    """获取指定Rack的数据 (rack_id格式: pod_0/rack_1)"""
    rack = generator.get_rack(rack_id)
    if rack is None:
        raise HTTPException(status_code=404, detail=f"Rack '{rack_id}' not found")
    return rack


@app.get("/api/topology/board/{board_id:path}", response_model=BoardConfig)
async def get_board(board_id: str):
    """获取指定Board的数据 (board_id格式: pod_0/rack_1/board_2)"""
    board = generator.get_board(board_id)
    if board is None:
        raise HTTPException(status_code=404, detail=f"Board '{board_id}' not found")
    return board


@app.get("/api/topology/connections", response_model=List[ConnectionConfig])
async def get_connections(level: Optional[str] = None, parent_id: Optional[str] = None):
    """
    获取连接数据

    Args:
        level: 筛选层级 (rack/board/chip)
        parent_id: 父节点ID，筛选指定父节点下的连接
    """
    if level:
        return generator.get_connections_for_level(level, parent_id)
    return generator.get_cached_topology().connections


# ============================================
# 辅助接口
# ============================================


@app.get("/api/config/chip-types")
async def get_chip_types():
    """获取支持的Chip类型"""
    return {
        "types": [
            {"id": "npu", "name": "NPU", "color": "#eb2f96"},
            {"id": "cpu", "name": "CPU", "color": "#1890ff"},
        ]
    }


@app.get("/api/config/rack-dimensions")
async def get_rack_dimensions():
    """获取Rack物理尺寸配置"""
    return {
        "width": 0.6,
        "depth": 1.0,
        "u_height": 0.0445,
        "total_u": 42,
        "full_height": 42 * 0.0445,
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "version": "2.0.0"}


# ============================================
# 配置保存/加载接口
# ============================================


def _get_config_path(name: str) -> str:
    """获取配置文件路径"""
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return os.path.join(CONFIG_DIR, f"{safe_name}.json")


@app.get("/api/configs", response_model=List[SavedConfig])
async def list_configs():
    """获取所有保存的配置列表"""
    configs = []
    for filename in os.listdir(CONFIG_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(CONFIG_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    configs.append(SavedConfig(**data))
            except Exception as e:
                print(f"读取配置文件失败: {filename}, {e}")
    # 按更新时间倒序
    configs.sort(key=lambda x: x.updated_at or x.created_at or "", reverse=True)
    return configs


@app.get("/api/configs/{name}", response_model=SavedConfig)
async def get_config(name: str):
    """获取指定名称的配置"""
    filepath = _get_config_path(name)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"配置 '{name}' 不存在")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SavedConfig(**data)


@app.post("/api/configs", response_model=SavedConfig)
async def save_config(config: SavedConfig):
    """保存配置"""
    filepath = _get_config_path(config.name)
    now = datetime.now().isoformat()

    # 检查是否存在，如存在则保留创建时间
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            old_data = json.load(f)
            config.created_at = old_data.get("created_at", now)
    else:
        config.created_at = now

    config.updated_at = now

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, ensure_ascii=False, indent=2)

    print(f"配置已保存: {config.name}")
    return config


@app.delete("/api/configs/{name}")
async def delete_config(name: str):
    """删除配置"""
    filepath = _get_config_path(name)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"配置 '{name}' 不存在")
    os.remove(filepath)
    return {"message": f"配置 '{name}' 已删除"}


# ============================================
# 启动入口
# ============================================

if __name__ == "__main__":
    # 推荐使用 start.py 启动，会自动清理端口
    # 直接运行 main.py 需要手动确保端口未被占用
    uvicorn.run(app, host="0.0.0.0", port=8102, reload=False)
