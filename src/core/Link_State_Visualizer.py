import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from collections import defaultdict, deque
import copy
import threading
import time


class NetworkLinkVisualizer:
    def __init__(self, network):
        self.network = network
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_aspect("equal")
        self.node_positions = self._calculate_layout()
        self.link_artists = {}  # 存储链路相关的静态信息
        self._colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self._color_map = {}
        self._next_color = 0
        self._draw_static_elements()

        # 播放控制参数
        self.pause_interval = 0.1        # 默认每帧暂停间隔(秒)
        self.should_stop = False         # 停止标志
        self.status_text = self.ax.text(-0.1, 1, f"Running...\nInterval: {self.pause_interval:.2f}", transform=self.ax.transAxes,
                                        fontsize=12, fontweight='bold', color='green',
                                        verticalalignment='top')
        # 绑定键盘事件:
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _calculate_layout(self):
        """根据网格计算节点位置（可调整节点间距）"""
        pos = {}
        for node in range(self.network.config.rows * self.network.config.cols):
            x, y = node % self.network.config.cols, node // self.network.config.cols
            # 为了美观，按照行列计算位置，并添加些许偏移
            if y % 2 == 1:  # 奇数行左移
                x -= 0.2
                y -= 0.6
            pos[node] = (x * 4, -y * 1.8)
        return pos

    def _draw_static_elements(self):
        """绘制静态元素：网络节点和链路（队列框架、方向箭头等）"""
        self.ax.clear()

        # 存放所有节点的 x 和 y 坐标
        xs = []
        ys = []

        # 绘制所有节点
        for node, (x, y) in self.node_positions.items():
            xs.append(x)
            ys.append(y)
            node_rect = Rectangle((x, y), 0.5, 0.5, facecolor="lightblue", edgecolor="black")
            self.ax.add_patch(node_rect)
            self.ax.text(x + 0.22, y + 0.24, f"{node}", ha="center", va="center", fontsize=12)

        # 绘制所有链路的框架，这里不再赘述
        self.link_artists.clear()
        for src, dest in self.network.links.keys():
            self._draw_link_frame(src, dest)

        # 根据节点位置自动调整显示范围
        if xs and ys:
            # 计算边界，并设定一定的补充边距
            margin_x = (max(xs) - min(xs)) * 0.1
            margin_y = (max(ys) - min(ys)) * 0.1
            self.ax.set_xlim(min(xs) - margin_x, max(xs) + margin_x + 0.5)
            self.ax.set_ylim(min(ys) - margin_y, max(ys) + margin_y + 0.5)
            
        self.ax.axis("off")
        plt.tight_layout()

    def _draw_link_frame(self, src, dest, queue_fixed_length=2):
        # 检查是否为自环链路
        is_self_loop = src == dest
        
        # 节点矩形尺寸
        node_width = 0.5
        node_height = 0.5
        half_w, half_h = node_width / 2, node_height / 2
        
        # 获取节点信息
        src_pos = self.node_positions[src]
        src_center = (src_pos[0] + half_w, src_pos[1] + half_h)
        
        if is_self_loop:
            # 判断节点是否在边界
            rows, cols = self.network.config.rows, self.network.config.cols
            row, col = src // cols, src % cols
            
            # 确定节点在哪个边界并设置相应的箭头和队列位置
            is_left_edge = col == 0 and row % 2 == 1
            is_right_edge = col == cols - 1 and row % 2 == 1
            is_top_edge = row == 0  and row % 2 == 0
            is_bottom_edge = row == rows - 2 and row % 2 == 0
            
            # 只处理边界节点，内部节点不添加自环
            if not (is_left_edge or is_right_edge or is_top_edge or is_bottom_edge):
                return
                
            # 根据边界位置设置自环方向和队列位置
            loop_offset = 0.1  # 自环与节点的距离
            queue_width = 0.2
            queue_height = queue_fixed_length/3.5
            
            # 确定箭头和队列的位置及方向
            if is_top_edge:  # 最上边，从右到左
                # src_arrow = (src_center[0] + half_w, src_center[1] + loop_offset)
                # dest_arrow = (src_center[0] - half_w, src_center[1] + loop_offset)
                queue_center = (src_center[0], src_center[1] + loop_offset + queue_height/2)
                is_horizontal = True
                is_forward = False  # 从右到左
            elif is_bottom_edge:  # 最下边，从左到右
                # src_arrow = (src_center[0] - half_w, src_center[1] - loop_offset)
                # dest_arrow = (src_center[0] + half_w, src_center[1] - loop_offset)
                queue_center = (src_center[0], src_center[1] - loop_offset - queue_height/2)
                is_horizontal = True
                is_forward = True  # 从左到右
            elif is_left_edge:  # 最左边，从上到下
                # src_arrow = (src_center[0] - loop_offset, src_center[1] + half_h)
                # dest_arrow = (src_center[0] - loop_offset, src_center[1] - half_h)
                queue_center = (src_center[0] - loop_offset*1.5 - queue_width, src_center[1])
                is_horizontal = False
                is_forward = False  # 从上到下
            elif is_right_edge:  # 最右边，从下到上
                # src_arrow = (src_center[0] + loop_offset, src_center[1] - half_h)
                # dest_arrow = (src_center[0] + loop_offset, src_center[1] + half_h)
                queue_center = (src_center[0] + loop_offset*1.5 + queue_width, src_center[1])
                is_horizontal = False
                is_forward = True  # 从下到上
            
            # 根据是水平还是垂直方向调整队列尺寸
            if is_horizontal:
                queue_width, queue_height = queue_height, queue_width
                
            # 绘制自环箭头
            # self.ax.annotate("", xy=dest_arrow, xycoords="data", xytext=src_arrow, 
                            # textcoords="data", arrowprops=dict(arrowstyle="->", color="blue", lw=2))
            
            # 绘制队列框架
            q_ll = (queue_center[0] - queue_width/2, queue_center[1] - queue_height/2)
            queue = Rectangle(q_ll, queue_width, queue_height, 
                            facecolor="white", edgecolor="black", linestyle="--")
            self.ax.add_patch(queue)
            
            # 存储链路绘制信息
            link_id = f"{src}-{dest}"
            self.link_artists[link_id] = {
                "queue_center": queue_center,
                "queue_width": queue_width,
                "queue_height": queue_height,
                "is_horizontal": is_horizontal,
                "is_forward": is_forward,
                "is_self_loop": True
            }
            return
            
        # 以下是原有的非自环链路的处理逻辑
        dest_pos = self.node_positions[dest]
        dest_center = (dest_pos[0] + half_w, dest_pos[1] + half_h)

        # 计算中心向量和距离
        dx = dest_center[0] - src_center[0]
        dy = dest_center[1] - src_center[1]
        center_distance = np.hypot(dx, dy)
        if center_distance == 0:
            return  # 避免自连接（这个检查现在实际上不会执行，因为我们已经单独处理了自环）
        dx, dy = dx / center_distance, dy / center_distance

        # 计算箭头穿过节点边界的交点
        src_arrow = (src_center[0] + dx * half_w, src_center[1] + dy * half_h)
        dest_arrow = (dest_center[0] - dx * half_w, dest_center[1] - dy * half_h)

        # 检查是否为双向链路
        bidirectional = (dest, src) in self.network.links
        extra_arrow_offset = 0.1  # 双向箭头的额外偏移量
        if bidirectional:
            # 使用简单的 id 大小比较决定偏移方向
            sign = -1 if src < dest else 1
            if abs(dx) >= abs(dy):
                # 水平布局，调整 y 坐标，使箭头上下错开
                src_arrow = (src_arrow[0], src_arrow[1] + sign * extra_arrow_offset)
                dest_arrow = (dest_arrow[0], dest_arrow[1] + sign * extra_arrow_offset)
            else:
                # 竖直布局，调整 x 坐标，使箭头左右错开
                src_arrow = (src_arrow[0] + sign * extra_arrow_offset, src_arrow[1])
                dest_arrow = (dest_arrow[0] + sign * extra_arrow_offset, dest_arrow[1])

        # 箭头中点，用于队列框架的参考位置
        arrow_mid = ((src_arrow[0] + dest_arrow[0]) / 2, (src_arrow[1] + dest_arrow[1]) / 2)

        # 根据箭头方向确定队列框架放置在箭头的哪一侧：
        # 对于水平箭头（|dx|>=|dy|）：dx<0表示箭头向左，队列放在上方；dx>0表示向右，队列放在下方。
        # 对于竖直箭头：dy>0表示箭头向上，队列放在右侧；dy<0表示向下，队列放在左侧。
        queue_offset = 0.15  # 队列框架中心与箭头中点的偏移量
        if abs(dx) >= abs(dy):
            # 水平箭头：队列偏移仅影响 y 坐标
            if dx < 0:
                queue_dx, queue_dy = 0.4, queue_offset  # 向上
            else:
                queue_dx, queue_dy = 0.4, -queue_offset  # 向下
        else:
            # 竖直箭头：队列偏移仅影响 x 坐标
            if dy > 0:
                queue_dx, queue_dy = queue_offset, -0.2  # 向右
            else:
                queue_dx, queue_dy = -queue_offset, -0.2  # 向左

        # 队列框架中心点
        queue_center = (arrow_mid[0] + queue_dx, arrow_mid[1] + queue_dy)

        # 队列框架的尺寸根据箭头方向决定
        # 对于水平箭头：宽度为 queue_fixed_length，高度固定为 0.3（横版）
        # 对于竖直箭头：宽度固定为 0.3，高度为 queue_fixed_length（竖版）
        is_horizontal = abs(dx) >= abs(dy)
        if is_horizontal:
            queue_width = queue_fixed_length
            queue_height = 0.2
        else:
            queue_width = 0.2
            queue_height = queue_fixed_length

        # 绘制队列框架矩形，中心位于 queue_center
        queue = Rectangle((queue_center[0] - queue_width / 2, queue_center[1] - queue_height / 2), queue_width, queue_height, facecolor="white", edgecolor="black", linestyle="--")
        self.ax.add_patch(queue)

        # 绘制箭头连接线，并使用 annotate 添加箭头头部
        self.ax.annotate("", xy=dest_arrow, xycoords="data", xytext=src_arrow, textcoords="data", arrowprops=dict(arrowstyle="->", color="blue", lw=2))

        # 存储链路绘制信息，可用于后续动态更新
        link_id = f"{src}-{dest}"
        self.link_artists[link_id] = {
            "queue_center": queue_center, 
            "queue_width": queue_width, 
            "queue_height": queue_height,
            "is_horizontal": is_horizontal,
            "is_forward": dx > 0 if is_horizontal else dy > 0,
            "is_self_loop": False
        }

    def update(self, network=None, expected_packet_id=0, use_highlight=False):
        """
        更新每条链路队列中 flit 的显示
        - 空位: 无填充的方形
        - flit: 有颜色的方形，颜色由 packet_id 决定
        - 支持所有方向(右、左、上、下)的链路
        - ID标签位置根据链路方向调整:
        - 向右的链路: 纵向标签在下方（数字垂直排列）
        - 向左的链路: 纵向标签在上方（数字垂直排列）
        - 向上的链路: 横向标签在右侧
        - 向下的链路: 横向标签在左侧
        """
        if self.should_stop:
            return False
        
        self.network = network
                        
        for link_id, artists_dict in self.link_artists.items():
            if "flit_artists" in artists_dict:
                for artist in artists_dict["flit_artists"]:
                    artist.remove()
            artists_dict["flit_artists"] = []

        # 遍历所有链路
        for (src, dest), flit_list in self.network.links.items():
            link_id = f"{src}-{dest}"
            if link_id not in self.link_artists:
                continue

            # 获取队列信息
            queue_info = self.link_artists[link_id]
            queue_center = queue_info["queue_center"]
            queue_width = queue_info["queue_width"]
            queue_height = queue_info["queue_height"]

            # 检查是否为自环
            is_self_loop = queue_info.get("is_self_loop", False)
            
            # 如果已经存储了链路方向，直接使用
            if "is_horizontal" in queue_info and "is_forward" in queue_info:
                is_horizontal = queue_info["is_horizontal"]
                is_forward = queue_info["is_forward"]
            else:
                # 确定链路方向（针对非自环链路的旧代码兼容）
                dx = src % self.network.config.cols - dest % self.network.config.cols
                dy = src // self.network.config.cols - dest // self.network.config.cols
                is_horizontal = abs(dx) > abs(dy)
                if is_horizontal:
                    is_forward = dx < 0  # 向右为正向
                else:
                    is_forward = dy > 0  # 向上为正向

            # 计算队列区域参数
            q_ll = (queue_center[0] - queue_width / 2, queue_center[1] - queue_height / 2)

            margin = 0.02
            flit_size = 0.15  # flit方形大小
            spacing = flit_size * 1.2  # flit之间的间距

            # 计算需要的总空间
            num_flits = len(flit_list)
            if is_horizontal:
                required_space = num_flits * spacing
                available_space = queue_width - 2 * margin
                # 如果需要的空间大于可用空间，按比例缩小间距
                spacing = available_space / num_flits
            else:
                required_space = num_flits * spacing
                available_space = queue_height - 2 * margin
                spacing = available_space / num_flits

            # 绘制所有flit位置
            flit_artists = []
            for i in range(num_flits):
                # 计算位置 - 考虑方向
                if is_horizontal:
                    x = q_ll[0] + margin + (i + 0.5) * spacing
                    y = queue_center[1]
                    # 如果是向左的链路，反转顺序
                    if not is_forward:
                        x = q_ll[0] + queue_width - margin - (i + 0.5) * spacing
                else:
                    x = queue_center[0]
                    y = q_ll[1] + margin + (i + 0.5) * spacing
                    # 如果是向下的链路，反转顺序
                    if not is_forward:
                        y = q_ll[1] + queue_height - margin - (i + 0.5) * spacing

                # 创建方形
                flit = flit_list[i]
                if flit is None:
                    continue
                facecolor = self._get_flit_color(flit, expected_packet_id=expected_packet_id, use_highlight=use_highlight)
                rect = Rectangle((x - flit_size / 2, y - flit_size / 2), flit_size, flit_size, facecolor=facecolor, edgecolor="black")

                # 添加文本标签 - 根据方向调整位置和显示方式
                label = f"{flit.packet_id}.{flit.flit_id}"

                # 2. 拼成多行标签
                if is_horizontal:
                    label = f"{flit.packet_id}\n{flit.flit_id}"
                    # 根据方向决定 y 偏移
                    if is_forward:
                        y_text = y - flit_size * 2 - 0.1
                    else:
                        y_text = y + flit_size * 2 + 0.1
                    if not is_self_loop:
                        # 一次性绘制两行
                        txt = self.ax.text(
                            x + (i - 0.2) * 0.04 * (int(is_forward)-0.5), y_text, label,
                            ha="center", va="center",
                            fontsize=9
                        )
                        flit_artists.append(txt)
                else:
                    # 垂直链路：横向标签
                    if is_forward:  # 向上的链路
                        text_x = x + flit_size * 1.1  # 右侧
                        ha = "left"
                    else:  # 向下的链路
                        text_x = x - flit_size * 1.1  # 左侧
                        ha = "right"
                    text_y = y
                    if not is_self_loop:
                        text = self.ax.text(text_x, text_y, label, ha=ha, va="center", fontsize=9)
                        flit_artists.append(text)

                self.ax.add_patch(rect)
                flit_artists.append(rect)

            # 保存该链路的图形对象
            self.link_artists[link_id]["flit_artists"] = flit_artists
        
        # 标题与排版
        self.ax.set_title(self.network.name)
        plt.tight_layout()
        plt.pause(self.pause_interval)

        return self.ax.patches
    
    def _update_status_display(self):
        """更新状态显示"""
        status = f"Running...\nInterval: {self.pause_interval:.2f}"
        color = 'green'
            
        # 更新状态文本
        self.status_text.set_text(status)
        self.status_text.set_color(color)

    def _on_key(self, event):
        key = event.key

        if key == 'up':
            # 加快 --> 缩短 pause_interval，但不跑到 0
            self.pause_interval = max(1e-3, self.pause_interval*0.75)
            self._update_status_display()
        elif key == 'down':
            # 减慢
            self.pause_interval *= 1.25
            self._update_status_display()
        elif key == 'q':
            # q 键 - 停止更新
            self.should_stop = True
        

    def _get_flit_color(self, flit, use_highlight=True, expected_packet_id=1, highlight_color=None):
        """获取颜色，支持多种PID格式：
        - 单个值 (packet_id 或 flit_id)
        - 元组 (packet_id, flit_id)
        - 字典 {'packet_id': x, 'flit_id': y}

        新增参数:
        - use_highlight: 是否启用高亮功能(默认False)
        - expected_packet_id: 期望的packet_id值
        - highlight_color: 高亮颜色(默认为红色)
        """
        # 统一提取 packet_id 作为颜色依据
        pid = {"packet_id": flit.packet_id, "flit_id": flit.flit_id}
        if isinstance(pid, tuple) and len(pid) >= 1:
            color_key = pid[0]  # 元组第一个元素作为 packet_id
        elif isinstance(pid, dict):
            color_key = pid.get("packet_id", str(pid))
        else:
            color_key = pid

        # 如果不启用高亮功能，使用原有逻辑
        if not use_highlight:
            if color_key in self._color_map:
                return self._color_map[color_key]
            c = self._colors[self._next_color % len(self._colors)]
            self._color_map[color_key] = c
            self._next_color += 1
            return c

        # 启用高亮功能时的逻辑
        if highlight_color is None:
            highlight_color = "red"  # 默认高亮红色

        # 方案1: 仅期望包高亮，其他灰色
        if flit.packet_id == expected_packet_id:
            return highlight_color
        return "grey"