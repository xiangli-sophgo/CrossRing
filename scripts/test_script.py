import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np


def draw_links_grid(links, rows=5, cols=4, node_size=800, selfloop_radius=0.2):
    """
    绘制 links 字典的图，节点为方形，边从方形边缘出发。

    参数:
        links (dict): 键为 (i, j) 表示边，值为边的标签/权重。
        rows (int): 网格行数（默认 5）。
        cols (int): 网格列数（默认 4）。
        node_size (int): 节点大小（默认 800）。
        selfloop_radius (float): 自循环边的半径（默认 0.2）。
    """
    G = nx.DiGraph()

    # 添加边
    for (i, j), value in links.items():
        G.add_edge(i, j, label=value)

    # 计算节点位置（按编号顺序排列）
    pos = {}
    for node in G.nodes():
        x = node % cols
        y = node // cols
        pos[node] = (x, -y)

    # 创建图形
    fig, ax = plt.subplots()
    ax.set_aspect("equal")  # 保持方形比例

    # 方形节点参数
    square_size = np.sqrt(node_size) / 100  # 方形边长（比例）

    # 绘制方形节点
    for node, (x, y) in pos.items():
        rect = Rectangle((x - square_size / 2, y - square_size / 2), width=square_size, height=square_size, color="lightblue", ec="black", zorder=2)  # 确保节点在边的上层
        ax.add_patch(rect)
        ax.text(x, y, str(node), ha="center", va="center", fontsize=10)

    # 自定义边的绘制（避开方形边缘）
    for i, j, data in G.edges(data=True):
        x1, y1 = pos[i]
        x2, y2 = pos[j]

        if i == j:  # 自循环边
            nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], connectionstyle=f"arc3,rad={selfloop_radius}", arrows=True, arrowstyle="-|>", ax=ax)
        else:  # 普通边
            # 计算边的起点和终点（方形边缘）
            dx, dy = x2 - x1, y2 - y1
            dist = np.hypot(dx, dy)
            if dist > 0:
                dx, dy = dx / dist, dy / dist  # 单位方向向量
                start_x = x1 + dx * square_size / 2  # 从方形边缘出发
                start_y = y1 + dy * square_size / 2
                end_x = x2 - dx * square_size / 2  # 到目标方形边缘结束
                end_y = y2 - dy * square_size / 2

                # 绘制带箭头的边
                arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle="-|>", mutation_scale=15, color="black", zorder=1)  # 边在节点下层
                ax.add_patch(arrow)

    # 绘制边标签
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    plt.title("Links Graph (Square Nodes, Edge Avoidance)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# 示例用法
if __name__ == "__main__":
    # links = {
    #     (0, 1): "A",
    #     (1, 2): "B",
    #     (2, 3): "C",
    #     (3, 3): "Self",
    #     (4, 5): "D",
    #     (5, 6): "E",
    #     (6, 6): "Loop",
    # }
    # draw_links_grid(links, rows=2, cols=4, node_size=1000, selfloop_radius=0.3)
    import matplotlib.colors as mcolors

    print(mcolors.CSS4_COLORS)  # 打印所有 CSS4 颜色名
