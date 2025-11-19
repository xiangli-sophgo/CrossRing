"""
拓扑API路由
提供CrossRing拓扑结构数据
"""
from fastapi import APIRouter
from typing import List, Dict

router = APIRouter(prefix="/api/topology", tags=["topology"])


@router.get("/")
async def get_available_topologies():
    """获取所有可用的拓扑类型"""
    return {
        "topologies": [
            {"type": "5x4", "rows": 5, "cols": 4, "nodes": 20},
            {"type": "4x5", "rows": 4, "cols": 5, "nodes": 20},
            {"type": "4x9", "rows": 4, "cols": 9, "nodes": 36},
            {"type": "9x4", "rows": 9, "cols": 4, "nodes": 36},
        ]
    }


@router.get("/{topo_type}")
async def get_topology(topo_type: str):
    """
    获取指定类型的拓扑数据

    返回格式：
    - nodes: 节点列表 [{id, label, row, col, x, y}]
    - edges: 边列表 [{source, target, direction}]
    """
    # 解析拓扑类型
    if "x" not in topo_type:
        return {"error": "Invalid topology type. Use format like '5x4'"}

    try:
        rows, cols = map(int, topo_type.split("x"))
    except ValueError:
        return {"error": "Invalid topology type. Use format like '5x4'"}

    # 生成节点数据
    nodes = []
    for row in range(rows):
        for col in range(cols):
            node_id = row * cols + col
            nodes.append({
                "id": node_id,
                "label": f"Node {node_id}",
                "row": row,
                "col": col,
                "x": col * 100,  # 用于布局
                "y": row * 100,
            })

    # 生成边数据 (Mesh拓扑: 行链 + 列链，无环形连接)
    edges = []

    # 行链 (Links in each row, no wrap-around)
    for row in range(rows):
        for col in range(cols - 1):  # 不连接最后一列到第一列
            source = row * cols + col
            target = row * cols + (col + 1)
            edges.append({
                "source": source,
                "target": target,
                "direction": "horizontal",
                "type": "row_link"
            })

    # 列链 (Links in each column, no wrap-around)
    for col in range(cols):
        for row in range(rows - 1):  # 不连接最后一行到第一行
            source = row * cols + col
            target = (row + 1) * cols + col
            edges.append({
                "source": source,
                "target": target,
                "direction": "vertical",
                "type": "col_link"
            })

    return {
        "type": topo_type,
        "rows": rows,
        "cols": cols,
        "total_nodes": rows * cols,
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "row_links": rows * (cols - 1),
            "col_links": cols * (rows - 1),
            "total_links": len(edges)
        }
    }


@router.get("/{topo_type}/nodes/{node_id}")
async def get_node_info(topo_type: str, node_id: int):
    """获取指定节点的详细信息"""
    rows, cols = map(int, topo_type.split("x"))

    if node_id < 0 or node_id >= rows * cols:
        return {"error": "Invalid node ID"}

    row = node_id // cols
    col = node_id % cols

    # 找到相邻节点 (Mesh结构，无环形连接)
    neighbors = []

    # 行方向邻居
    if col > 0:
        left_neighbor = row * cols + (col - 1)
        neighbors.append(left_neighbor)
    if col < cols - 1:
        right_neighbor = row * cols + (col + 1)
        neighbors.append(right_neighbor)

    # 列方向邻居
    if row > 0:
        up_neighbor = (row - 1) * cols + col
        neighbors.append(up_neighbor)
    if row < rows - 1:
        down_neighbor = (row + 1) * cols + col
        neighbors.append(down_neighbor)

    return {
        "node_id": node_id,
        "position": {"row": row, "col": col},
        "label": f"Node {node_id}",
        "neighbors": neighbors,
        "degree": len(set(neighbors)),  # 去重后的邻居数
        "topology": topo_type
    }
