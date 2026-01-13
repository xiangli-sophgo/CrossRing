import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from collections import defaultdict, deque
import copy
import threading
from matplotlib.widgets import Button
import time
from types import SimpleNamespace


# ---------- lightweight flit proxy for snapshot rendering ----------
class _FlitProxy:
    __slots__ = ("packet_id", "flit_id", "ETag_priority", "itag_h", "itag_v")

    def __init__(self, pid, fid, etag, ih, iv):
        self.packet_id = pid
        self.flit_id = fid
        self.ETag_priority = etag
        self.itag_h = ih
        self.itag_v = iv

    def __repr__(self):
        itag = "H" if self.itag_h else ("V" if self.itag_v else "")
        return f"(pid={self.packet_id}, fid={self.flit_id}, ET={self.ETag_priority}, IT={itag})"


from src.kcin.base.config import KCINConfigBase
from src.analysis.piece_visualizer_v1 import PieceVisualizerV1
from src.analysis.piece_visualizer_v2 import PieceVisualizerV2

import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict
from matplotlib.patches import FancyArrowPatch


class NetworkLinkVisualizer:
    """网络链路可视化器 - 显示网络拓扑和节点 FIFO 状态

    PieceVisualizer 已分离到独立文件:
    - piece_visualizer_v1.py: v1架构 (IQ/EQ/RB)
    - piece_visualizer_v2.py: v2架构 (RingStation)
    """

    def __init__(self, network):
        self.network = network
        self.cols = network.config.NUM_COL
        # ---- Figure & Sub‑Axes ------------------------------------------------
        self.fig = plt.figure(figsize=(15, 10), constrained_layout=True)

        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.3, 1], left=0.02, right=0.98, top=0.95, bottom=0.08)
        self.ax = self.fig.add_subplot(gs[0])  # 主网络视图
        self.piece_ax = self.fig.add_subplot(gs[1])  # 右侧 Piece 视图
        self.piece_ax.axis("off")
        self.ax.set_aspect("equal")
        # 根据版本选择 PieceVisualizer
        if hasattr(self.network, "ring_stations"):
            # v2 架构使用 RingStation
            self.piece_vis = PieceVisualizerV2(self.network.config, self.piece_ax, highlight_callback=self._on_piece_highlight, parent=self)
        else:
            # v1 架构使用 IQ/EQ/RB
            self.piece_vis = PieceVisualizerV1(self.network.config, self.piece_ax, highlight_callback=self._on_piece_highlight, parent=self)
        # 当前点击选中的节点 (None 表示未选)
        self._selected_node = None
        # 绘制主网络的静态元素
        self.slice_per_link_horizontal = network.config.SLICE_PER_LINK_HORIZONTAL
        self.slice_per_link_vertical = network.config.SLICE_PER_LINK_VERTICAL
        self.node_positions = self._calculate_layout()
        self.link_artists = {}  # 存储链路相关的静态信息
        self._colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        # 节点大小参数（统一管理，方便调整）
        self.node_size = 0.8

        self.cycle = 0
        self.paused = False
        # ============  flit‑click tracking ==============
        self.tracked_pid = None  # 当前追踪的 packet_id (None = 不追踪)
        self.rect_info_map = {}  # rect → (text_obj, packet_id)
        self.node_pair_slots = {}  # 存储节点对的slot位置，用于双向link对齐
        self.fig.canvas.mpl_connect("button_press_event", self._on_flit_click)
        # 绑定节点点击事件
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        # 绑定窗口关闭事件
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        # 颜色/高亮控制
        self.use_highlight = False
        self.highlight_pid = 0
        # Show tags only mode (force all flit faces to light grey)
        self.show_tags_only = False
        # ===============  History Buffer  ====================
        # 支持多网络显示
        self.networks = None
        self.selected_network_index = 2
        # 为每个网络维护独立历史缓冲
        self.histories = [deque(maxlen=20) for _ in range(3)]
        self.buttons = []
        # 添加网络选择按钮
        btn_positions = [
            (0.01, 0.03, 0.07, 0.04),
            (0.10, 0.03, 0.07, 0.04),
            (0.19, 0.03, 0.07, 0.04),
        ]
        for idx, label in enumerate(["REQ", "RSP", "DATA"]):
            ax_btn = self.fig.add_axes(btn_positions[idx])
            btn = Button(ax_btn, label)
            btn.on_clicked(lambda event, i=idx: self._on_select_network(i))
            self.buttons.append(btn)

        # -------- Clear-Highlight button --------
        ax_clear = self.fig.add_axes([0.28, 0.03, 0.07, 0.04])
        self.clear_btn = Button(ax_clear, "Clear HL")
        self.clear_btn.on_clicked(self._on_clear_highlight)

        # -------- Show Tags Only button --------
        ax_tags = self.fig.add_axes([0.37, 0.03, 0.07, 0.04])
        self.tags_btn = Button(ax_tags, "Show Tags")
        self.tags_btn.on_clicked(self._on_toggle_tags)

        # -------- Hover tooltip (显示 packet_id) --------
        # (removed hover annotation and motion_notify_event)
        self._play_idx = None  # 暂停时正在浏览的 history 索引
        self._draw_static_elements()

        # 初始化按钮状态
        self._update_button_states()

        # 初始化时显示中心节点
        rows, cols = self.network.config.NUM_ROW, self.network.config.NUM_COL
        # 取中间行列
        center_row = rows // 2
        center_col = cols // 2
        center_raw = center_row * cols + center_col
        # 点击逻辑：若在偶数行（0-based）需映射到下一行
        if (center_raw // cols) % 2 == 0:
            center_sel = center_raw + cols if center_raw + cols < rows * cols else center_raw
        else:
            center_sel = center_raw
        self._selected_node = center_sel
        # 绘制初始 Piece
        self.piece_ax.clear()
        self.piece_ax.axis("off")
        self.piece_vis.draw_piece_for_node(self._selected_node, self.network)
        # 初始化时绘制高亮框（仅高亮当前选中节点）
        x_ll, y_ll = self.node_positions[self._selected_node]
        self.click_box = Rectangle((x_ll, y_ll), self.node_size, self.node_size, facecolor="none", edgecolor="red", linewidth=1.2, linestyle="--")
        self.ax.add_patch(self.click_box)
        self.fig.canvas.draw_idle()

        # 播放控制参数
        self.pause_interval = 0.4  # 默认每帧暂停间隔(秒)
        self.should_stop = False  # 停止标志
        self.status_text = self.ax.text(
            -0.1, 1, f"Running...\nInterval: {self.pause_interval:.2f}", transform=self.ax.transAxes, fontsize=12, fontweight="bold", color="green", verticalalignment="top"
        )
        # 用 PieceVisualizer 的 info_text 作为统一信息框
        self.info_text = self.piece_vis.info_text
        # store last snapshot for info_text refresh
        self.last_snapshot = {}

    # ------------------------------------------------------------
    # Helper: refresh info_text with repr of all flits with tracked_pid
    # ------------------------------------------------------------
    def _update_info_text(self):
        """右下角显示：当前 tracked_pid 的所有 flit.__repr__()"""
        pid_trk = self.tracked_pid
        if pid_trk is None:
            self.info_text.set_text("")
            return

        lines = []

        # ① link 视图：rect_info_map
        for _, flit, _ in self.rect_info_map.values():
            if getattr(flit, "packet_id", None) == pid_trk:
                lines.append(str(flit))

        # ② piece 视图：patch_info_map
        for _, flit in self.piece_vis.patch_info_map.values():
            if getattr(flit, "packet_id", None) == pid_trk:
                lines.append(str(flit))

        # 去重、保持顺序
        seen = set()
        uniq = [l for l in lines if not (l in seen or seen.add(l))]

        self.info_text.set_text("\n".join(uniq))

    # ------------------------------------------------------------------
    #  simple palette lookup (no cache)                                 #
    # ------------------------------------------------------------------
    def _palette_color(self, pid: int):
        return self._colors[pid % len(self._colors)]

    # ------------------------------------------------------------------
    # 鼠标点击：显示选中节点的 Cross‑Ring Piece
    # ------------------------------------------------------------------
    def _on_click(self, event):
        # 若点中了某个 flit patch，则交给 flit 点击逻辑，不当节点点击
        for rect in self.rect_info_map:
            contains, _ = rect.contains(event)
            if contains:
                return

        # 只处理左键，且点击在主网络视图(self.ax)上
        if event.button != 1 or event.inaxes is not self.ax:
            return

        # 检查是否点击到节点方块
        sel_node = None
        for nid, (x_ll, y_ll) in self.node_positions.items():
            # 节点方块大小
            if x_ll <= event.xdata <= x_ll + self.node_size and y_ll <= event.ydata <= y_ll + self.node_size:
                sel_node = nid
                break

        if sel_node is None:
            return

        self._selected_node = sel_node

        # 删除上一个 click_box
        if hasattr(self, "click_box"):
            try:
                self.click_box.remove()
            except Exception:
                pass

        # 重画新的高亮框（仅高亮被点击的节点）
        x_ll, y_ll = self.node_positions[sel_node]
        self.click_box = Rectangle((x_ll, y_ll), self.node_size, self.node_size, facecolor="none", edgecolor="red", linewidth=1.2, linestyle="--")
        self.ax.add_patch(self.click_box)

        # 清空并绘制右侧 Piece 视图
        self.piece_ax.clear()
        self.piece_ax.axis("off")
        if self.paused and self._play_idx is not None and len(self.histories[self.selected_network_index]) > self._play_idx:
            _, _, meta = self.histories[self.selected_network_index][self._play_idx]
            fake_net = SimpleNamespace(
                IQ_channel_buffer=meta["IQ_channel_buffer"],
                EQ_channel_buffer=meta["EQ_channel_buffer"],
                inject_queues=meta["inject_queues"],
                eject_queues=meta["eject_queues"],
                ring_bridge=meta["ring_bridge"],
                config=self.networks[self.selected_network_index].config,
                cross_point=meta["cross_point"],
                IQ_arbiter_input_fifo=meta.get("IQ_arbiter_input_fifo", {}),
                EQ_arbiter_input_fifo=meta.get("EQ_arbiter_input_fifo", {}),
                crosspoints=meta.get("crosspoints", {}),
            )
            self.piece_vis.draw_piece_for_node(self._selected_node, fake_net)
        else:
            live_net = self.networks[self.selected_network_index] if self.networks is not None else self.network
            self.piece_vis.draw_piece_for_node(self._selected_node, live_net)

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Receive packet_id selected in PieceVisualizer and toggle highlight
    # ------------------------------------------------------------------
    def _on_piece_highlight(self, pid, fid):
        if self.tracked_pid == pid:
            self.tracked_pid = None
            self.use_highlight = False
            self.highlight_pid = 0
            self.info_text.set_text("")
        else:
            self.tracked_pid = pid
            self.use_highlight = True
            self.highlight_pid = pid
            self.info_text.set_text(f"PID {pid}  FID {fid}")
        self._update_tracked_labels()
        self._refresh_piece_view()
        self.fig.canvas.draw_idle()
        # 同步piece视图的高亮状态
        self.piece_vis.sync_highlight(self.use_highlight, self.tracked_pid)

    # ------------------------------------------------------------------
    # flit 点击：显示/隐藏 flit id 并追踪该 packet_id
    # ------------------------------------------------------------------
    def _on_flit_click(self, event):
        if event.button != 1 or event.inaxes is not self.ax:
            return

        for rect, (txt, flit, tag) in self.rect_info_map.items():
            contains, _ = rect.contains(event)
            if contains:
                pid = getattr(flit, "packet_id", None)
                # 切换追踪状态
                if self.tracked_pid == pid:
                    self.tracked_pid = None
                    self.use_highlight = False
                else:
                    self.tracked_pid = pid
                    self.use_highlight = True
                    self.highlight_pid = pid

                # 同步右侧 Piece 高亮
                self.piece_vis.sync_highlight(self.use_highlight, self.tracked_pid)
                # 同步右侧 Piece 高亮并重绘 Piece 视图
                if self._selected_node is not None:
                    live_net = self.networks[self.selected_network_index] if self.networks is not None else self.network
                    self.piece_vis.draw_piece_for_node(self._selected_node, live_net)
                # 更新链路和 piece 的可见性
                self._update_tracked_labels()
                # Show info_text with tag if present
                info_str = str(flit)
                if tag is not None:
                    info_str += f"\nITag: {tag}"
                self.info_text.set_text(info_str)
                self.fig.canvas.draw_idle()
                return

        # 若点击空白 link 的 slice，仍需判断是否有 tag
        if event.inaxes == self.ax:
            for rect in self.rect_info_map:
                contains, _ = rect.contains(event)
                if contains:
                    break
            else:
                # 更鲁棒的三角形点击检测逻辑，使用 bbox.contains(event.x, event.y)
                import matplotlib.pyplot as plt

                for link_id, info in self.link_artists.items():
                    flit_artists = info.get("flit_artists", [])
                    for idx, artist in enumerate(flit_artists):
                        if not isinstance(artist, plt.Polygon):
                            continue
                        bbox = artist.get_window_extent()
                        if bbox.contains(event.x, event.y):
                            # 新架构：从artist.tag_val获取ITag信息
                            tag_val = getattr(artist, "tag_val", None)
                            if tag_val is not None:
                                self.info_text.set_text(f"ITag: {tag_val}")
                                self.fig.canvas.draw_idle()
                            return

    def _update_tracked_labels(self):
        highlight_on = self.tracked_pid is not None
        for rect, (txt, flit, _) in self.rect_info_map.items():
            pid = getattr(flit, "packet_id", None)
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
        for node in range(self.network.config.NUM_ROW * self.network.config.NUM_COL):
            x, y = node % self.network.config.NUM_COL, node // self.network.config.NUM_COL
            pos[node] = (x * 4, -y * 4)
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
            node_rect = Rectangle((x, y), self.node_size, self.node_size, facecolor="lightblue", edgecolor="black")
            self.ax.add_patch(node_rect)
            self.ax.text(x + self.node_size / 2, y + self.node_size / 2, f"{node}", ha="center", va="center", fontsize=12)

        # 绘制所有链路的框架
        self.link_artists.clear()
        for link_key in self.network.links_flow_stat.keys():
            # 处理2-tuple或3-tuple格式的link key
            src, dest = link_key[:2] if len(link_key) >= 2 else link_key
            # 根据链路类型选择正确的 slice 数量
            if abs(src - dest) == self.network.config.NUM_COL or abs(src - dest) == self.network.config.NUM_COL * 2:
                # 纵向链路
                slice_num = self.slice_per_link_vertical
            else:
                # 横向链路
                slice_num = self.slice_per_link_horizontal
            self._draw_link_frame(src, dest, slice_num=slice_num)

        # 根据节点位置自动调整显示范围
        if xs and ys:
            # 计算边界，并设定一定的补充边距
            margin_x = (max(xs) - min(xs)) * 0.1
            margin_y = (max(ys) - min(ys)) * 0.1
            self.ax.set_xlim(min(xs) - margin_x, max(xs) + margin_x + self.node_size)
            self.ax.set_ylim(min(ys) - margin_y, max(ys) + margin_y + self.node_size)

        self.ax.axis("off")
        # self.fig.tight_layout(rect=[0, 0.1, 1, 1])

    def _draw_link_frame(self, src, dest, queue_fixed_length=1.6, slice_num=7):
        # 检查是否为自环链路
        is_self_loop = src == dest

        # 节点矩形尺寸
        node_width = self.node_size
        node_height = self.node_size
        half_w, half_h = node_width / 2, node_height / 2

        # 获取节点信息
        src_pos = self.node_positions[src]
        src_center = (src_pos[0] + half_w, src_pos[1] + half_h)

        if is_self_loop:
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
        bidirectional = (dest, src) in self.network.links_flow_stat
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

        # 队列框架中心点
        queue_center = (arrow_mid[0], arrow_mid[1])

        # 队列框架的尺寸根据箭头方向决定，动态调整长度保持 slice 接近正方形
        # 期望的单个 slice 尺寸（接近正方形）
        target_slice_size = 0.4
        actual_slice_count = slice_num - 2  # 实际绘制的 slice 数量（减去首尾）

        is_horizontal = abs(dx) >= abs(dy)
        if is_horizontal:
            queue_height = 0.4
            queue_width = target_slice_size * actual_slice_count if actual_slice_count > 0 else queue_fixed_length
        else:
            queue_width = 0.4
            queue_height = target_slice_size * actual_slice_count if actual_slice_count > 0 else queue_fixed_length

        # 新实现：slices沿link方向排列，而非侧面队列框架
        # 计算垂直于link的方向向量
        perp_dx, perp_dy = -dy, dx  # 旋转90度

        # Slice参数
        slot_size = 0.2
        slot_spacing = 0.0
        side_offset = 0.25  # 距离link中心线的距离

        # 计算link的实际起止点（去除节点占用部分）
        node_radius = 0.25
        link_start_x = src_center[0] + dx * node_radius
        link_start_y = src_center[1] + dy * node_radius
        link_end_x = dest_center[0] - dx * node_radius
        link_end_y = dest_center[1] - dy * node_radius

        # Link长度
        link_length = np.hypot(link_end_x - link_start_x, link_end_y - link_start_y)

        # 计算slices排列区域（显示全部slice）
        total_length = slice_num * slot_size + (slice_num - 1) * slot_spacing
        start_offset = (link_length - total_length) / 2

        # 节点对，用于双向link对齐
        node_pair = (min(src, dest), max(src, dest))
        link_id = f"{src}-{dest}"

        # 检查是否已经为这对节点创建了slices
        if node_pair in self.node_pair_slots:
            # 复用已有位置
            existing_slots = self.node_pair_slots[node_pair]
            # 根据方向选择对应侧
            is_forward = src < dest
            target_side = "side1" if is_forward else "side2"
            target_slots = [s for s in existing_slots if s[1].startswith(target_side)]

            if not is_forward:
                # 反向link需要反转slice顺序
                target_slots = list(reversed(target_slots))

            for slot_pos, slot_id in target_slots:
                slot_x, slot_y = slot_pos
                slot = Rectangle((slot_x, slot_y), slot_size, slot_size, facecolor="white", edgecolor="gray", linewidth=0.8, linestyle="--")
                self.ax.add_patch(slot)
                self.rect_info_map[slot] = (None, None)
        else:
            # 首次创建，在link两侧都绘制
            slot_positions_list = []

            for side_name, side_sign in [("side1", 1), ("side2", -1)]:
                for i in range(slice_num):  # 显示全部slice
                    # 沿link方向的位置
                    along_dist = start_offset + i * (slot_size + slot_spacing)
                    progress = along_dist / link_length if link_length > 0 else 0

                    center_x = link_start_x + progress * (link_end_x - link_start_x)
                    center_y = link_start_y + progress * (link_end_y - link_start_y)

                    # 垂直偏移
                    slot_x = center_x + perp_dx * side_offset * side_sign - slot_size / 2
                    slot_y = center_y + perp_dy * side_offset * side_sign - slot_size / 2

                    slot = Rectangle((slot_x, slot_y), slot_size, slot_size, facecolor="white", edgecolor="gray", linewidth=0.8, linestyle="--")
                    self.ax.add_patch(slot)

                    slot_id = f"{side_name}_{i}"
                    slot_positions_list.append(((slot_x, slot_y), slot_id))
                    self.rect_info_map[slot] = (None, None)

            # 记录供反向link复用
            self.node_pair_slots[node_pair] = slot_positions_list
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

    def update(self, networks=None, cycle=None, skip_pause=False):
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
        # 检查窗口是否仍然存在
        if not plt.fignum_exists(self.fig.number):
            self.should_stop = True
            return False

        # 接收并保存多网络列表
        if networks is not None:
            self.networks = networks

        # 检查停止标志
        if self.should_stop:
            return False

        # 若暂停且非跳过暂停调用，则仅保持 GUI 响应；不推进模拟
        if self.paused and not skip_pause:
            plt.pause(self.pause_interval)
            return self.ax.patches
        self.cycle = cycle

        # 记录所有网络的历史快照
        if cycle is not None and self.networks is not None:
            for i, net in enumerate(self.networks):
                # 构建快照（存储 Flit 对象或 None）
                # 纯offset模式：从Ring.slices获取flit
                snap = self._build_snapshot_from_rings(net)
                meta = {
                    "network_name": net.name,
                    "cross_point": copy.deepcopy(net.cross_point),
                    # 新架构：ITag存储在Ring.itag中，不再使用links_tag
                }
                # v2 架构：保存 ring_stations
                if hasattr(net, "ring_stations"):
                    # 只保存每个 RingStation 的 FIFO 状态
                    rs_snapshot = {}
                    for node_id, rs in net.ring_stations.items():
                        rs_snapshot[node_id] = SimpleNamespace(
                            input_fifos={k: list(v) for k, v in rs.input_fifos.items()},
                            output_fifos={k: list(v) for k, v in rs.output_fifos.items()},
                        )
                    meta["ring_stations"] = rs_snapshot
                else:
                    # v1 架构：保存 IQ/EQ/RB
                    meta["IQ_channel_buffer"] = copy.deepcopy(net.IQ_channel_buffer)
                    meta["EQ_channel_buffer"] = copy.deepcopy(net.EQ_channel_buffer)
                    meta["inject_queues"] = copy.deepcopy(net.inject_queues)
                    meta["eject_queues"] = copy.deepcopy(net.eject_queues)
                    meta["ring_bridge"] = copy.deepcopy(net.ring_bridge)
                    meta["IQ_arbiter_input_fifo"] = copy.deepcopy(getattr(net, "IQ_arbiter_input_fifo", {}))
                    meta["EQ_arbiter_input_fifo"] = copy.deepcopy(getattr(net, "EQ_arbiter_input_fifo", {}))
                    # 保存crosspoints的cp_slices
                    crosspoints_snapshot = {}
                    for node_id, cps in net.crosspoints.items():
                        crosspoints_snapshot[node_id] = {
                            "horizontal": SimpleNamespace(cp_slices=copy.deepcopy(cps["horizontal"].cp_slices)),
                            "vertical": SimpleNamespace(cp_slices=copy.deepcopy(cps["vertical"].cp_slices)),
                        }
                    meta["crosspoints"] = crosspoints_snapshot
                    # 新架构：保存Ring的itag信息（用于ITag显示）
                    # 直接引用live Ring对象（ITag回放时使用实时网络）
                    meta["horizontal_rings"] = net.horizontal_rings
                    meta["vertical_rings"] = net.vertical_rings
                self.histories[i].append((cycle, snap, meta))

        # 渲染当前选中网络的快照
        if self.networks is not None:
            current_net = self.networks[self.selected_network_index]
            # 纯offset模式：从Ring.slices获取flit
            render_snap = self._build_snapshot_from_rings(current_net)
            self._render_snapshot(render_snap)
            # 若已有选中节点，实时更新右侧 Piece 视图
            if self._selected_node is not None:
                self._refresh_piece_view()
            self.ax.set_title(current_net.name)
        else:
            # 纯offset模式：从Ring.slices获取flit
            self._render_snapshot(self._build_snapshot_from_rings(self.network))
            # 若已有选中节点，实时更新右侧 Piece 视图
            if self._selected_node is not None:
                self._refresh_piece_view()
            self.ax.set_title(self.network.name)
        if cycle and self.cycle % 10 == 0:
            self._update_status_display()
        # 检查停止标志，避免不必要的pause
        if not skip_pause and not self.should_stop:
            plt.pause(self.pause_interval)
        return self.ax.patches

    def _update_status_display(self):
        """更新状态显示"""
        if self.paused:
            # 保持暂停颜色 & 文本
            self.status_text.set_color("orange")
            return
        status = f"Running... cycle: {self.cycle}\nInterval: {self.pause_interval:.2f}"
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
        self.piece_vis.sync_highlight(self.use_highlight, self.tracked_pid)

        self.piece_ax.clear()
        self.piece_ax.axis("off")

        # 当前网络对应的历史缓冲
        current_history = self.histories[self.selected_network_index]

        # 回溯模式：用保存的快照队列
        if self.paused and self._play_idx is not None and len(current_history) > self._play_idx:
            _, _, meta = current_history[self._play_idx]
            # v2 架构检查
            if hasattr(self.networks[self.selected_network_index], "ring_stations"):
                # v2: 需要包含 ring_stations
                fake_net = SimpleNamespace(
                    ring_stations=meta.get("ring_stations", {}),
                    cross_point=meta["cross_point"],
                    config=self.networks[self.selected_network_index].config,
                )
            else:
                fake_net = SimpleNamespace(
                    IQ_channel_buffer=meta["IQ_channel_buffer"],
                    EQ_channel_buffer=meta["EQ_channel_buffer"],
                    inject_queues=meta["inject_queues"],
                    eject_queues=meta["eject_queues"],
                    ring_bridge=meta["ring_bridge"],
                    cross_point=meta["cross_point"],
                    # 新架构：ITag存储在Ring.itag中
                    horizontal_rings=meta.get("horizontal_rings", {}),
                    vertical_rings=meta.get("vertical_rings", {}),
                    IQ_arbiter_input_fifo=meta.get("IQ_arbiter_input_fifo", {}),
                    EQ_arbiter_input_fifo=meta.get("EQ_arbiter_input_fifo", {}),
                    crosspoints=meta.get("crosspoints", {}),
                    config=self.networks[self.selected_network_index].config,
                )
            self.piece_vis.draw_piece_for_node(self._selected_node, fake_net)
        else:  # 实时
            live_net = self.networks[self.selected_network_index]
            self.piece_vis.draw_piece_for_node(self._selected_node, live_net)

        self.fig.canvas.draw_idle()

    def _on_close(self, event):
        """窗口关闭时设置停止标志"""
        self.should_stop = True
        # print("可视化窗口已关闭，仿真将继续运行...")

    def _on_key(self, event):
        key = event.key

        # 数字键1-3选择对应网络 (REQ=1, RSP=2, DATA=3)
        if key in ["1", "2", "3"]:
            network_idx = int(key) - 1
            if 0 <= network_idx < len(self.histories):
                self.selected_network_index = network_idx
                # 更新按钮状态
                self._update_button_states()
                # 刷新显示（cycle=None避免重复保存快照）
                if self.networks is not None:
                    self.update(self.networks, cycle=None, skip_pause=True)
                else:
                    self.update(None, cycle=None, skip_pause=True)
                return

        # 使用当前选中网络的历史
        current_history = self.histories[self.selected_network_index]

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
        elif key == " ":  # 空格键控制暂停/恢复
            self.paused = not self.paused
            if self.paused:
                self.status_text.set_text("Paused")
                self.status_text.set_color("orange")
                if self.paused:
                    # 进入暂停：定位到最新快照并立即绘制
                    if current_history:
                        self._play_idx = len(current_history) - 1
                        cyc, snap, meta = current_history[self._play_idx]
                        # 同步高亮 / 标题等元数据
                        # 注释掉这两行，不从历史恢复高亮状态
                        # self.use_highlight = meta.get("use_highlight", False)
                        # self.highlight_pid = meta.get("expected_pid", 0)
                        self.ax.set_title(meta.get("network_name", ""))
                        self.status_text.set_text(f"Paused\ncycle {cyc} ({self._play_idx+1}/{len(current_history)})")
                        self._draw_state(snap)
                        self._refresh_piece_view()
            else:
                self._update_status_display()
                self._play_idx = None
        elif self.paused and key in {"left", "right"}:
            if not current_history:
                return
            if self._play_idx is None:
                self._play_idx = len(current_history) - 1
            if key == "left":
                self._play_idx = max(0, self._play_idx - 1)
            else:  # "right"
                self._play_idx = min(len(current_history) - 1, self._play_idx + 1)
            cyc, snap, meta = current_history[self._play_idx]

            # 保存当前的高亮状态
            self.ax.set_title(meta.get("network_name", ""))
            self.status_text.set_text(f"Paused\ncycle {cyc} ({self._play_idx+1}/{len(current_history)})")
            self._draw_state(snap)
            self._refresh_piece_view()

    def _draw_state(self, snapshot):
        self._render_snapshot(snapshot)

    def _get_itag_from_ring(self, network, src, dest, slice_index):
        """从新架构的Ring.itag获取ITag信息

        Args:
            network: 网络对象
            src, dest: link的源和目标节点
            slice_index: slice在link中的索引

        Returns:
            tag信息 [reserver_node, reserver_dir] 或 None
        """
        if not hasattr(network, 'horizontal_rings') or not hasattr(network, 'vertical_rings'):
            return None

        # 判断是横向还是纵向link
        num_col = network.config.NUM_COL
        is_horizontal = abs(src - dest) == 1 or (src // num_col == dest // num_col)

        if is_horizontal:
            row = src // num_col
            if row not in network.horizontal_rings:
                return None
            ring = network.horizontal_rings[row]
        else:
            col = src % num_col
            if col not in network.vertical_rings:
                return None
            ring = network.vertical_rings[col]

        # 在Ring中找到对应的slice并获取slot_id
        # 遍历ring.slices找到属于这个link的slice
        link_slice_count = 0
        for s in ring.slices:
            if s.slice_type == "LINK" and s.node_id == src:
                if link_slice_count == slice_index:
                    slot_id = ring.get_slot_id_at(s.ring_index)
                    if slot_id in ring.itag:
                        info = ring.itag[slot_id]
                        return [info.get("reserver_node"), info.get("reserver_dir")]
                    return None
                link_slice_count += 1
        return None

    def _get_link_flits_from_ring(self, network, src, dest):
        """从Ring.slices获取link上的flit列表（纯offset模式）

        Args:
            network: 网络对象
            src, dest: link的源和目标节点

        Returns:
            list: link上每个slice的flit列表（可能为None）
        """
        # 判断是横向还是纵向link
        num_col = network.config.NUM_COL
        is_horizontal = abs(src - dest) == 1 or (src // num_col == dest // num_col)

        # 确定方向
        if is_horizontal:
            row = src // num_col
            if row not in network.horizontal_rings:
                return []
            ring = network.horizontal_rings[row]
            direction = "TR" if dest > src else "TL"
        else:
            col = src % num_col
            if col not in network.vertical_rings:
                return []
            ring = network.vertical_rings[col]
            direction = "TD" if dest > src else "TU"

        # 收集属于这个link的slice的flit
        flits = []
        for s in ring.slices:
            if s.slice_type == "LINK" and s.node_id == src and s.direction == direction:
                flits.append(s.flit)

        return flits

    def _build_snapshot_from_rings(self, network):
        """从Ring.slices构建快照（纯offset模式）

        Args:
            network: 网络对象

        Returns:
            dict: {(src, dest): [flit列表]}
        """
        snap = {}

        # 使用links_flow_stat的键（它与links的键相同）
        for link_key in network.links_flow_stat.keys():
            src, dest = link_key[:2]
            flits = self._get_link_flits_from_ring(network, src, dest)
            snap[(src, dest)] = flits

        return snap

    def _render_snapshot(self, snapshot):
        # 新架构：从Ring.itag获取tag信息
        current_net = self.networks[self.selected_network_index] if self.networks else self.network
        # keep snapshot for later info refresh
        self.last_snapshot = snapshot
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

        slot_size = 0.2  # slot方块边长
        flit_size = 0.2  # flit方块边长(略小于slot)

        for (src, dest), flit_list in snapshot.items():
            link_id = f"{src}-{dest}"
            if link_id not in self.link_artists:
                continue

            # 获取link的方向信息
            info = self.link_artists[link_id]
            is_horizontal = info["is_horizontal"]
            is_forward = info["is_forward"]

            # 获取slot位置
            node_pair = (min(src, dest), max(src, dest))
            if node_pair not in self.node_pair_slots:
                continue

            # 根据方向选择对应侧的slots
            all_slots = self.node_pair_slots[node_pair]
            target_side = "side2" if src < dest else "side1"
            target_slots = [s for s in all_slots if s[1].startswith(target_side)]

            if src >= dest:
                # 反向link需要反转slot顺序
                target_slots = list(reversed(target_slots))

            num_slices = len(flit_list)
            if num_slices == 0 or num_slices != len(target_slots):
                continue

            flit_artists = []
            for i, flit in enumerate(flit_list):
                if i >= len(target_slots):
                    break

                # 获取slot位置
                slot_pos, slot_id = target_slots[i]
                slot_x, slot_y = slot_pos

                # 计算flit在slot中心的位置
                x = slot_x + slot_size / 2
                y = slot_y + slot_size / 2

                # 获取tag信息（新架构：从Ring.itag获取）
                idx_slice = i  # 现在显示全部slice，索引直接对应
                tag = self._get_itag_from_ring(current_net, src, dest, idx_slice)

                if flit is None:
                    # 空slot，如果有tag画三角
                    if tag is not None:
                        t_size = flit_size * 0.6
                        triangle = plt.Polygon(
                            [
                                (x, y + t_size / 2),
                                (x - t_size / 2, y - t_size / 4),
                                (x + t_size / 2, y - t_size / 4),
                            ],
                            color="red",
                        )
                        triangle.tag_val = tag
                        self.ax.add_patch(triangle)
                        flit_artists.append(triangle)
                    continue

                # 绘制flit矩形
                face, alpha, lw, edge = self._get_flit_style(
                    flit,
                    use_highlight=self.use_highlight,
                    expected_packet_id=self.highlight_pid,
                )

                rect = Rectangle(
                    (x - flit_size / 2, y - flit_size / 2),
                    flit_size,
                    flit_size,
                    facecolor=face,
                    linewidth=lw,
                    alpha=alpha,
                    edgecolor=edge,
                )
                self.ax.add_patch(rect)
                flit_artists.append(rect)

                # 文本标签
                pid, fid = flit.packet_id, flit.flit_id
                label = f"{pid}.{fid}"

                if is_horizontal:
                    # 标签放上下
                    y_text = y - slot_size * 2 if is_forward else y + slot_size * 2
                    txt = self.ax.text(x, y_text, label, ha="center", va="center", fontsize=8)
                else:
                    # 标签放左右
                    text_x = x + slot_size * 2 if is_forward else x - slot_size * 2
                    ha = "left" if is_forward else "right"
                    txt = self.ax.text(text_x, y, label, ha=ha, va="center", fontsize=8)

                txt.set_visible(self.use_highlight and pid == self.tracked_pid)
                self.rect_info_map[rect] = (txt, flit, tag)
                flit_artists.append(txt)

                # 绘制tag三角形
                if tag is not None:
                    t_size = flit_size * 0.6
                    triangle = plt.Polygon(
                        [
                            (x, y + t_size / 2),
                            (x - t_size / 2, y - t_size / 4),
                            (x + t_size / 2, y - t_size / 4),
                        ],
                        color="red",
                    )
                    triangle.tag_val = tag
                    self.ax.add_patch(triangle)
                    flit_artists.append(triangle)

            # 保存此链路新生成的图元
            info["flit_artists"] = flit_artists

        # update info box according to current tracking
        self._update_info_text()
        # 最后刷新画布
        self.fig.canvas.draw_idle()

    _ETAG_ALPHA = {"T0": 1.0, "T1": 1.0, "T2": 0.85}  # T0  # T1  # T2
    _ETAG_LW = {"T0": 2, "T1": 2, "T2": 1}  # T0  # T1  # T2
    _ETAG_EDGE = {"T0": 2, "T1": 1, "T2": 0}
    _ITAG_ALPHA = {True: 1.0, False: 0.85}
    _ITAG_LW = {True: 1.0, False: 0}
    _ITAG_EDGE = {True: 2.0, False: 0}

    def _get_flit_style(self, flit, use_highlight=True, expected_packet_id=0, highlight_color=None):
        """
        返回 (facecolor, alpha, linewidth)
        - facecolor 仍沿用 _get_flit_color 的逻辑（高亮 / 调色板）
        - alpha / linewidth 由 flit.etag 决定
        """
        # When showing tags only, force face color to light grey
        if getattr(self, "show_tags_only", False):
            face = "lightgrey"
        else:
            face = self._get_flit_color(flit, use_highlight, expected_packet_id, highlight_color)

        etag = getattr(flit, "ETag_priority", "T2")
        itag = flit.itag_h or flit.itag_v
        alpha = max(self._ETAG_ALPHA.get(etag, 1.0), self._ITAG_ALPHA.get(itag, 1.0))
        lw = max(self._ETAG_LW.get(etag, 0), self._ITAG_LW.get(itag, 0))
        # Determine border color: yellow if ITag is present, else red if ETag indicates, otherwise black
        if flit.itag_h:
            edge_color = "yellow"
        elif flit.itag_v:
            edge_color = "blue"
        else:
            etag_edge_value = self._ETAG_EDGE.get(etag, 0)
            edge_color = "red" if etag_edge_value == 2 else "black"

        return face, alpha, lw, edge_color

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

    def _update_button_states(self):
        """更新按钮的视觉状态（高亮选中的按钮）"""
        # 更新网络类型按钮（REQ/RSP/DATA）
        for idx, btn in enumerate(self.buttons):
            if idx == self.selected_network_index:
                btn.color = "lightblue"  # 选中
                btn.hovercolor = "cornflowerblue"
            else:
                btn.color = "0.85"  # 未选中（浅灰色）
                btn.hovercolor = "0.95"
            btn.ax.set_facecolor(btn.color)

        self.fig.canvas.draw_idle()

    def _on_select_network(self, idx):
        """切换显示网络索引 idx（0/1/2）"""
        self.selected_network_index = idx
        # 更新按钮状态
        self._update_button_states()
        # 刷新显示（调用 update 渲染当前网络）
        if self.networks is not None:
            self.update(
                self.networks,
                cycle=None,
                # expected_packet_id=self.highlight_pid,
                # use_highlight=self.use_highlight,
                skip_pause=True,
            )

    # ------------------------------------------------------------------
    # (removed hover annotation and motion event handler)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Clear‑Highlight button callback

    def _on_clear_highlight(self, event):
        """清除高亮追踪状态"""
        self.tracked_pid = None
        self.use_highlight = False
        self.piece_vis.sync_highlight(False, None)
        self._update_tracked_labels()
        self.info_text.set_text("")
        self.fig.canvas.draw_idle()

    def _on_toggle_tags(self, event):
        """切换仅显示标签模式，并刷新视图"""
        self.show_tags_only = not self.show_tags_only
        # 更新按钮标签文本以反映当前状态
        if self.show_tags_only:
            self.tags_btn.label.set_text("Show Flits")
        else:
            self.tags_btn.label.set_text("Show Tags")
        # 刷新当前网络视图（cycle=None避免重复保存快照）
        if self.networks is not None:
            self.update(self.networks, cycle=None, skip_pause=True)
        else:
            self.update(None, cycle=None, skip_pause=True)

    def _on_select_network(self, idx):
        """切换显示网络索引 idx（0/1/2）"""
        self.selected_network_index = idx
        # 更新按钮状态
        self._update_button_states()
        # 刷新显示（调用 update 渲染当前网络）
        if self.networks is not None:
            self.update(
                self.networks,
                cycle=None,
                # expected_packet_id=self.highlight_pid,
                # use_highlight=self.use_highlight,
                skip_pause=True,
            )
