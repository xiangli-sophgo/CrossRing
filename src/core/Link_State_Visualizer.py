import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from collections import defaultdict, deque
import copy
import threading
import time
from types import SimpleNamespace

# 引入节点局部 CrossRing piece 绘制函数（若存在）
from .CrossRing_Piece_Visualizer import CrossRingVisualizer


class NetworkLinkVisualizer:
    def __init__(self, network):
        self.network = network
        self.cols = network.config.cols
        # ---- Figure & Sub‑Axes ------------------------------------------------
        self.fig = plt.figure(figsize=(15, 8))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
        self.ax = self.fig.add_subplot(gs[0])  # 主网络视图
        self.piece_ax = self.fig.add_subplot(gs[1])  # 右侧 Piece 视图
        self.piece_ax.axis("off")
        self.ax.set_aspect("equal")
        self.piece_vis = CrossRingVisualizer(self.network.config, self.piece_ax)
        # 当前点击选中的节点 (None 表示未选)
        self._selected_node = None
        # 绘制主网络的静态元素
        self.node_positions = self._calculate_layout()
        self.link_artists = {}  # 存储链路相关的静态信息
        self._colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        self.paused = False
        # ============  flit‑click tracking ==============
        self.tracked_pid = None  # 当前追踪的 packet_id (None = 不追踪)
        self.rect_info_map = {}  # rect → (text_obj, packet_id)
        self.fig.canvas.mpl_connect("button_press_event", self._on_flit_click)
        # 颜色/高亮控制
        self._use_highlight = False
        self._expected_pid = 0
        # ===============  History Buffer  ====================
        # 保存最近N个周期的轻量级链路状态，便于暂停时回溯
        self.history = deque(maxlen=20)
        self._play_idx = None  # 暂停时正在浏览的 history 索引
        self._draw_static_elements()

        # 播放控制参数
        self.pause_interval = 0.2  # 默认每帧暂停间隔(秒)
        self.should_stop = False  # 停止标志
        self.status_text = self.ax.text(
            -0.1, 1, f"Running...\nInterval: {self.pause_interval:.2f}", transform=self.ax.transAxes, fontsize=12, fontweight="bold", color="green", verticalalignment="top"
        )
        # 绑定键盘事件:
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        # 鼠标点击用于显示节点局部 Piece
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    # ------------------------------------------------------------------
    #  simple palette lookup (no cache)                                 #
    # ------------------------------------------------------------------
    def _palette_color(self, pid: int):
        return self._colors[pid % len(self._colors)]

    # ------------------------------------------------------------------
    # 鼠标点击：显示选中节点的 Cross‑Ring Piece
    # ------------------------------------------------------------------
    def _on_click(self, event):
        # 若点中了 flit，则交由 flit 追踪逻辑处理，不当作节点点击
        for rect in self.rect_info_map:
            contains, _ = rect.contains(event)
            if contains:
                return
        # 仅处理左键，且在主 ax 内点击
        if event.button != 1 or event.inaxes is not self.ax:
            return

        # 找到距离最近且在阈值内的节点
        sel_node = None
        min_d = float("inf")
        for nid, (x_ll, y_ll) in self.node_positions.items():
            cx, cy = x_ll + 0.25, y_ll + 0.25  # 节点中心
            d = np.hypot(event.xdata - cx, event.ydata - cy)
            if d < min_d and d < 0.35:  # 阈值可根据布局调整
                min_d = d
                sel_node = nid

        if sel_node is None:
            return  # 点击空白
        if (sel_node // self.cols) % 2 == 0:
            sel_node += self.cols
        # 记录当前选中节点
        self._selected_node = sel_node

        # 清空右侧子图并绘制
        self.piece_ax.clear()
        self.piece_ax.axis("off")

        # 若正处于历史回溯视图，使用快照里的队列状态
        if self.paused and self._play_idx is not None and len(self.history) > self._play_idx:
            _, _, meta = self.history[self._play_idx]
            fake_net = SimpleNamespace(
                IQ_channel_buffer=meta["IQ_channel_buffer"],
                EQ_channel_buffer=meta["EQ_channel_buffer"],
                inject_queues=meta["inject_queues"],
                eject_queues=meta["eject_queues"],
                ring_bridge=meta["ring_bridge"],
                config=self.network.config,
            )
            self.piece_vis.draw_piece_for_node(sel_node, fake_net)
        else:
            self.piece_vis.draw_piece_for_node(sel_node, self.network)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # flit 点击：显示/隐藏 flit id 并追踪该 packet_id
    # ------------------------------------------------------------------
    def _on_flit_click(self, event):
        if event.button != 1 or event.inaxes is not self.ax:
            return
        # 查找点击中的 flit
        for rect, (txt, pid) in self.rect_info_map.items():
            contains, _ = rect.contains(event)
            if contains:
                # 切换追踪
                if self.tracked_pid == pid:
                    self.tracked_pid = None
                else:
                    self.tracked_pid = pid
                # 根据追踪状态更新高亮配置
                self._use_highlight = self.tracked_pid is not None
                self._expected_pid = self.tracked_pid or 0
                # 刷新右侧 Piece 视图以保持同步高亮
                self._refresh_piece_view()
                # 更新可见性
                self._update_tracked_labels()
                self.fig.canvas.draw_idle()
                break

    def _update_tracked_labels(self):
        highlight_on = self.tracked_pid is not None
        for rect, (txt, pid) in self.rect_info_map.items():
            is_target = highlight_on and pid == self.tracked_pid

            # 文本可见性
            txt.set_visible(is_target)

            # 面色高亮 / 恢复
            if highlight_on:
                rect.set_facecolor("red" if is_target else "lightgrey")
            else:
                rect.set_facecolor(self._palette_color(pid))

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

    def _draw_link_frame(self, src, dest, queue_fixed_length=1.6, seat_num=7):
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
            is_top_edge = row == 0 and row % 2 == 0
            is_bottom_edge = row == rows - 2 and row % 2 == 0

            # 只处理边界节点，内部节点不添加自环
            if not (is_left_edge or is_right_edge or is_top_edge or is_bottom_edge):
                return

            # 根据边界位置设置自环方向和队列位置
            loop_offset = 0.1  # 自环与节点的距离
            queue_width = 0.2
            queue_height = queue_fixed_length / 3.5

            # 确定箭头和队列的位置及方向
            if is_top_edge:  # 最上边，从右到左
                # src_arrow = (src_center[0] + half_w, src_center[1] + loop_offset)
                # dest_arrow = (src_center[0] - half_w, src_center[1] + loop_offset)
                queue_center = (src_center[0], src_center[1] + loop_offset + queue_height / 2)
                is_horizontal = True
                is_forward = False  # 从右到左
            elif is_bottom_edge:  # 最下边，从左到右
                # src_arrow = (src_center[0] - half_w, src_center[1] - loop_offset)
                # dest_arrow = (src_center[0] + half_w, src_center[1] - loop_offset)
                queue_center = (src_center[0], src_center[1] - loop_offset - queue_height / 2)
                is_horizontal = True
                is_forward = True  # 从左到右
            elif is_left_edge:  # 最左边，从上到下
                # src_arrow = (src_center[0] - loop_offset, src_center[1] + half_h)
                # dest_arrow = (src_center[0] - loop_offset, src_center[1] - half_h)
                queue_center = (src_center[0] - loop_offset * 1.5 - queue_width, src_center[1])
                is_horizontal = False
                is_forward = False  # 从上到下
            elif is_right_edge:  # 最右边，从下到上
                # src_arrow = (src_center[0] + loop_offset, src_center[1] - half_h)
                # dest_arrow = (src_center[0] + loop_offset, src_center[1] + half_h)
                queue_center = (src_center[0] + loop_offset * 1.5 + queue_width, src_center[1])
                is_horizontal = False
                is_forward = True  # 从下到上

            # 根据是水平还是垂直方向调整队列尺寸
            if is_horizontal:
                queue_width, queue_height = queue_height, queue_width

            # 绘制自环箭头
            # self.ax.annotate("", xy=dest_arrow, xycoords="data", xytext=src_arrow,
            # textcoords="data", arrowprops=dict(arrowstyle="->", color="blue", lw=2))

            # 绘制队列框架
            q_ll = (queue_center[0] - queue_width / 2, queue_center[1] - queue_height / 2)
            queue = Rectangle(q_ll, queue_width, queue_height, facecolor="white", edgecolor="black", linestyle="--")
            self.ax.add_patch(queue)

            # 存储链路绘制信息
            link_id = f"{src}-{dest}"
            self.link_artists[link_id] = {
                "queue_center": queue_center,
                "queue_width": queue_width,
                "queue_height": queue_height,
                "is_horizontal": is_horizontal,
                "is_forward": is_forward,
                "is_self_loop": True,
            }
            return

        # 非自环链路的处理逻辑
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
        # queue = Rectangle((queue_center[0] - queue_width / 2, queue_center[1] - queue_height / 2), queue_width, queue_height, facecolor="white", edgecolor="black", linestyle="--")
        # self.ax.add_patch(queue)

        seats = self.split_queue_into_seats(
            (queue_center[0] - queue_width / 2, queue_center[1] - queue_height / 2),
            queue_width,
            queue_height,
            seat_num,
            is_horizontal,
            facecolor="white",
            edgecolor="black",
            linestyle="--",
        )

        for seat in seats:
            self.ax.add_patch(seat)
        # 绘制箭头连接线，并使用 annotate 添加箭头头部
        self.ax.annotate("", xy=dest_arrow, xycoords="data", xytext=src_arrow, textcoords="data", arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))

        # 存储链路绘制信息，可用于后续动态更新
        link_id = f"{src}-{dest}"
        self.link_artists[link_id] = {
            "queue_center": queue_center,
            "queue_width": queue_width,
            "queue_height": queue_height,
            "is_horizontal": is_horizontal,
            "is_forward": dx > 0 if is_horizontal else dy > 0,
            "is_self_loop": False,
        }

    def split_queue_into_seats(self, q_ll, queue_width, queue_height, seat_num, is_horizontal=True, **kwargs):
        """
        将队列矩形分割成 seat_num 个小矩形，支持横向或纵向切割

        参数:
            q_ll: 队列左下角坐标 (x, y)
            queue_width: 队列总宽度
            queue_height: 队列高度
            seat_num: 座位数量
            is_horizontal:
                - True（默认）: 横向切割（沿宽度方向，生成多个等宽小矩形）
                - False: 纵向切割（沿高度方向，生成多个等高小矩形）
            **kwargs: 传递给 Rectangle 的其他参数（如 facecolor, edgecolor 等）

        返回:
            list: 包含 seat_num 个小矩形的列表
        """
        seats = []

        if is_horizontal:
            # 横向切割（沿宽度方向）
            seat_width = queue_width / seat_num
            for i in range(seat_num):
                seat_ll = (q_ll[0] + i * seat_width, q_ll[1])
                seat = Rectangle(seat_ll, seat_width, queue_height, **kwargs)
                seats.append(seat)
        else:
            # 纵向切割（沿高度方向）
            seat_height = queue_height / seat_num
            for i in range(seat_num):
                seat_ll = (q_ll[0], q_ll[1] + i * seat_height)
                seat = Rectangle(seat_ll, queue_width, seat_height, **kwargs)
                seats.append(seat)

        return seats

    def update(self, network=None, cycle=None, expected_packet_id=0, use_highlight=False):
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
        # 记录当前高亮配置, 供回溯绘制保持一致
        self._use_highlight = use_highlight
        self._expected_pid = expected_packet_id
        # 若暂停，则仅保持 GUI 响应；不推进模拟
        if self.paused:
            plt.pause(self.pause_interval)
            return self.ax.patches

        if self.should_stop:
            return False

        self.network = network
        # 记录快照到历史缓冲 (仅在提供 cycle 时)
        if cycle is not None:
            snapshot = {(src, dest): [(f.packet_id, f.flit_id) if f is not None else None for f in flits] for (src, dest), flits in self.network.links.items()}
            meta = {
                "network_name": self.network.name,
                "use_highlight": self._use_highlight,
                "expected_pid": self._expected_pid,
                # 深拷贝三类队列，便于历史回溯时还原 Piece 状态
                "IQ_channel_buffer": copy.deepcopy(self.network.IQ_channel_buffer),
                "EQ_channel_buffer": copy.deepcopy(self.network.EQ_channel_buffer),
                "inject_queues": copy.deepcopy(self.network.inject_queues),
                "eject_queues": copy.deepcopy(self.network.eject_queues),
                "ring_bridge": copy.deepcopy(getattr(self.network, "ring_bridge", {})),
            }
            self.history.append((cycle, snapshot, meta))

        self._render_snapshot({(src, dest): [(f.packet_id, f.flit_id) if f is not None else None for f in flits] for (src, dest), flits in self.network.links.items()})
        # 若已有选中节点，实时更新右侧 Piece 视图
        if self._selected_node is not None:
            self._refresh_piece_view()
        self.ax.set_title(self.network.name)
        plt.tight_layout()
        plt.pause(self.pause_interval)
        return self.ax.patches

    def _update_status_display(self):
        """更新状态显示"""
        if self.paused:
            # 保持暂停颜色 & 文本
            self.status_text.set_color("orange")
            return
        status = f"Running...\nInterval: {self.pause_interval:.2f}"
        color = "green"

        # 更新状态文本
        self.status_text.set_text(status)
        self.status_text.set_color(color)

    # ------------------------------------------------------------------
    # 刷新右侧局部 Piece 视图（实时 / 回溯自动判断）
    # ------------------------------------------------------------------
    def _refresh_piece_view(self):
        if self._selected_node is None:
            return
        # 把当前高亮信息同步给右侧 Piece 可视化器
        self.piece_vis.use_highlight = self._use_highlight
        self.piece_vis.highlight_pid = self.tracked_pid

        self.piece_ax.clear()
        self.piece_ax.axis("off")

        # 回溯模式：用保存的快照队列
        if self.paused and self._play_idx is not None and len(self.history) > self._play_idx:
            _, _, meta = self.history[self._play_idx]
            fake_net = SimpleNamespace(
                IQ_channel_buffer=meta["IQ_channel_buffer"],
                EQ_channel_buffer=meta["EQ_channel_buffer"],
                inject_queues=meta["inject_queues"],
                eject_queues=meta["eject_queues"],
                ring_bridge=meta["ring_bridge"],
                config=self.network.config,
            )
            self.piece_vis.draw_piece_for_node(self._selected_node, fake_net)
        else:  # 实时
            self.piece_vis.draw_piece_for_node(self._selected_node, self.network)

        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        key = event.key

        if key == "up":
            if not self.paused:  # 暂停时不调速
                self.pause_interval = max(1e-3, self.pause_interval * 0.75)
                self._update_status_display()
        elif key == "down":
            if not self.paused:
                self.pause_interval *= 1.25
                self._update_status_display()
        elif key == "q":
            # q 键 - 停止更新
            for art in getattr(self, "_history_artists", []):
                try:
                    art.remove()
                except Exception:
                    pass
            self.should_stop = True
        elif key == "p":
            self.paused = not self.paused
            if self.paused:
                self.status_text.set_text("Paused")
                self.status_text.set_color("orange")
                if self.paused:
                    # 进入暂停：定位到最新快照并立即绘制
                    if self.history:
                        self._play_idx = len(self.history) - 1
                        cyc, snap, meta = self.history[self._play_idx]
                        # 同步高亮 / 标题等元数据
                        self._use_highlight = meta.get("use_highlight", False)
                        self._expected_pid = meta.get("expected_pid", 0)
                        self.ax.set_title(meta.get("network_name", ""))
                        self.status_text.set_text(f"Paused\ncycle {cyc} ({self._play_idx+1}/{len(self.history)})")
                        self._draw_state(snap)
                        self._refresh_piece_view()
            else:
                self._update_status_display()
                self._play_idx = None
        elif self.paused and key in {"left", "right"}:
            if not self.history:
                return
            if self._play_idx is None:
                self._play_idx = len(self.history) - 1
            if key == "left":
                self._play_idx = max(0, self._play_idx - 1)
            else:  # "right"
                self._play_idx = min(len(self.history) - 1, self._play_idx + 1)
            cyc, snap, meta = self.history[self._play_idx]
            # 同步高亮 / 标题
            self._use_highlight = meta.get("use_highlight", False)
            self._expected_pid = meta.get("expected_pid", 0)
            self.ax.set_title(meta.get("network_name", ""))
            self.status_text.set_text(f"Paused\ncycle {cyc} ({self._play_idx+1}/{len(self.history)})")
            self._draw_state(snap)
            self._refresh_piece_view()

    def _draw_state(self, snapshot):
        self._render_snapshot(snapshot)

    def _render_snapshot(self, snapshot):
        # 重置 flit→文本映射
        self.rect_info_map.clear()
        # 清掉上一帧的 flit 图元
        for link_id, info in self.link_artists.items():
            for art in info.get("flit_artists", []):
                try:
                    art.remove()
                except Exception:
                    pass
            info["flit_artists"] = []

        margin = 0.02
        flit_size = 0.15  # 单个 flit 方块边长
        for (src, dest), flit_list in snapshot.items():
            link_id = f"{src}-{dest}"
            if link_id not in self.link_artists:
                continue

            info = self.link_artists[link_id]
            queue_center = info["queue_center"]
            queue_width = info["queue_width"]
            queue_height = info["queue_height"]
            is_horizontal = info["is_horizontal"]
            is_forward = info["is_forward"]
            is_self_loop = info.get("is_self_loop", False)

            q_ll = (queue_center[0] - queue_width / 2, queue_center[1] - queue_height / 2)

            num_seats = len(flit_list)
            if num_seats == 0:
                continue

            # 计算 seat 间距，确保所有 seat 都能显示
            if is_horizontal:
                spacing = (queue_width - 2 * margin) / num_seats
            else:
                spacing = (queue_height - 2 * margin) / num_seats

            flit_artists = []
            for i, seat in enumerate(flit_list):
                if seat is None:
                    continue  # 空位

                pid, fid = seat
                flit = SimpleNamespace(packet_id=pid, flit_id=fid)

                # ---------- 位置 ----------
                if is_horizontal:
                    x = q_ll[0] + margin + (i + 0.5) * spacing
                    y = queue_center[1]
                    if not is_forward:  # 向左
                        x = q_ll[0] + queue_width - margin - (i + 0.5) * spacing
                else:
                    x = queue_center[0]
                    y = q_ll[1] + margin + (i + 0.5) * spacing
                    if not is_forward:  # 向下
                        y = q_ll[1] + queue_height - margin - (i + 0.5) * spacing

                # ---------- 绘制矩形 ----------
                facecolor = self._get_flit_color(
                    flit,
                    use_highlight=self._use_highlight,
                    expected_packet_id=self._expected_pid,
                )
                rect = Rectangle(
                    (x - flit_size / 2, y - flit_size / 2),
                    flit_size,
                    flit_size,
                    facecolor=facecolor,
                    edgecolor="black",
                )
                self.ax.add_patch(rect)
                flit_artists.append(rect)

                # ---------- 文本标签 ----------
                label_vert = f"{pid}\n{fid}"
                label_horz = f"{pid}.{fid}"
                if is_horizontal:
                    # 标签放上下
                    y_text = y - flit_size * 2 - 0.1 if is_forward else y + flit_size * 2 + 0.1
                    txt = self.ax.text(
                        x,
                        y_text,
                        label_vert,
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
                    txt.set_visible(False)
                    self.rect_info_map[rect] = (txt, pid)
                    flit_artists.append(txt)
                else:
                    # 标签放左右
                    text_x = x + flit_size * 1.1 if is_forward else x - flit_size * 1.1
                    ha = "left" if is_forward else "right"
                    txt = self.ax.text(
                        text_x,
                        y,
                        label_horz,
                        ha=ha,
                        va="center",
                        fontsize=8,
                    )
                    txt.set_visible(False)
                    self.rect_info_map[rect] = (txt, pid)
                    flit_artists.append(txt)

            # 保存此链路新生成的图元
            info["flit_artists"] = flit_artists

        # 根据当前 tracked_pid 更新可见性
        self._update_tracked_labels()
        # 最后刷新画布
        self.fig.canvas.draw_idle()

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
        # 高亮模式：目标 flit → 红，其余 → 灰
        if use_highlight:
            hl = highlight_color or "red"
            return hl if flit.packet_id == expected_packet_id else "lightgrey"

        # 普通模式：直接取调色板色
        return self._palette_color(flit.packet_id)
