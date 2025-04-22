import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import numpy as np


class Network_Internal_Partition_Visualizer:
    def __init__(self, network, node_id, config):
        self.network = network
        self.node_id = node_id
        self.config = config

        # 初始化图形参数
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.axis("off")

        # 初始化图形元素
        self.init_node_elements()
        self.init_link_elements()

    def init_node_elements(self):
        """初始化节点内部元素"""
        # 主节点方框
        self.ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=False, lw=2))

        # 初始化四个方向的FIFO队列
        self.fifo_artists = {
            "left": self.create_fifo_queue((-1.2, 0), "vertical"),
            "right": self.create_fifo_queue((1.2, 0), "vertical"),
            "up": self.create_fifo_queue((0, 1.2), "horizontal"),
            "local": self.create_fifo_queue((0, -1.2), "horizontal"),
        }

    def create_fifo_queue(self, position, orientation):
        """创建FIFO队列的图形元素"""
        fifo_length = self.config.IQ_OUT_FIFO_DEPTH
        elements = []

        for i in range(fifo_length):
            if orientation == "vertical":
                pos = (position[0], position[1] - i * 0.2)
            else:
                pos = (position[0] + i * 0.2, position[1])

            elements.append({"box": Rectangle((pos[0] - 0.1, pos[1] - 0.1), 0.2, 0.2, fill=True, color="white", ec="black"), "flits": []})
        return {"elements": elements, "direction": orientation, "patches": [self.ax.add_patch(e["box"]) for e in elements]}

    def init_link_elements(self):
        """初始化链路元素"""
        self.link_artists = {}
        neighbors = self.get_neighbors()

        # 为每个链路创建图形元素
        for direction, pos in neighbors.items():
            seats = self.config.seats_per_link
            self.link_artists[direction] = self.create_link_elements(direction, seats)

    def create_link_elements(self, direction, seats):
        """创建链路的flit位置标记"""
        positions = {
            "left": [(-1.5 + i * 0.3, 0) for i in range(seats)],
            "right": [(1.5 - i * 0.3, 0) for i in range(seats)],
            "up": [(0, 1.5 - i * 0.3) for i in range(seats)],
            "down": [(0, -1.5 + i * 0.3) for i in range(seats)],
        }
        return [self.ax.add_patch(Circle(pos, 0.1, color="gray", alpha=0.3)) for pos in positions[direction]]

    def get_neighbors(self):
        """获取当前节点的邻居节点"""
        # 这里需要根据实际的网络拓扑实现
        # 示例返回假数据
        return {"left": self.node_id - 1, "right": self.node_id + 1, "up": self.node_id + self.config.cols, "down": self.node_id - self.config.cols}

    def update(self, frame):
        """动画更新函数"""
        self.update_fifos()
        self.update_links()
        return []

    def update_fifos(self):
        """更新FIFO队列的可视化"""
        for direction, artist in self.fifo_artists.items():
            fifo = self.network.inject_queues[direction].get(self.node_id, [])

            for i, element in enumerate(artist["elements"]):
                # 更新颜色表示不同优先级
                if i < len(fifo):
                    flit = fifo[i]
                    color = self.get_flit_color(flit)
                    element["box"].set_facecolor(color)
                else:
                    element["box"].set_facecolor("white")

    def update_links(self):
        """更新链路状态的可视化"""
        neighbors = self.get_neighbors()
        for direction, neighbor in neighbors.items():
            link_key = (self.node_id, neighbor)
            flits = self.network.links.get(link_key, [])

            for i, patch in enumerate(self.link_artists[direction]):
                if i < len(flits) and flits[i] is not None:
                    color = self.get_flit_color(flits[i])
                    patch.set_color(color)
                    patch.set_alpha(1.0)
                else:
                    patch.set_color("gray")
                    patch.set_alpha(0.3)

    def get_flit_color(self, flit):
        """根据flit属性获取颜色"""
        # 示例颜色映射，需要根据实际数据结构调整
        if flit and hasattr(flit, "tag"):
            return {"T0": "red", "T1": "orange", "T2": "green"}.get(flit.tag, "blue")
        return "white"

    def animate(self):
        """启动动画"""
        ani = animation.FuncAnimation(self.fig, self.update, interval=1000, blit=True)
        plt.show()
