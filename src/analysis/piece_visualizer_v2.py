"""
PieceVisualizer v2 - 适配 RingStation 架构的节点可视化

v2 架构使用 RingStation 统一 IQ/EQ/RB 功能：
- input_fifos: ch_buffer, TL, TR, TU, TD
- output_fifos: ch_buffer, TL, TR, TU, TD
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict

from src.kcin.base.config import KCINConfigBase


class PieceVisualizerV2:
    """
    v2 架构的节点 FIFO 可视化器
    绘制单个 RingStation 模块（包含 input 和 output）+ CrossPoint
    """

    def __init__(self, config: KCINConfigBase, ax, highlight_callback=None, parent=None):
        self.highlight_callback = highlight_callback
        self.config = config
        self.cols = config.NUM_COL
        self.rows = config.NUM_ROW
        self.parent = parent

        # v2 使用 RingStation 配置
        self.rs_in_ch_depth = config.RS_IN_CH_BUFFER
        self.rs_in_fifo_depth = config.RS_IN_FIFO_DEPTH
        self.rs_out_ch_depth = config.RS_OUT_CH_BUFFER
        self.rs_out_fifo_depth = config.RS_OUT_FIFO_DEPTH

        # 几何参数
        self.square = 0.3
        self.gap = 0.02
        self.fontsize = 8
        self.slot_frame_lw = 0.4

        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
        else:
            self.ax = ax
            self.fig = ax.figure
        self.ax.axis("off")
        self.ax.set_aspect("equal")

        self._colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # 高亮设置
        self.use_highlight = False
        self.highlight_pid = None
        self.highlight_color = "red"
        self.grey_color = "lightgrey"

        # 存储 patch 和 text - RS 统一模块
        self.rs_patches, self.rs_texts = {}, {}
        self.cph_patches, self.cph_texts = {}, {}
        self.cpv_patches, self.cpv_texts = {}, {}

        # 不在初始化时绘制，等 update() 时根据实际通道名绘制
        self.patch_info_map = {}
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.info_text = self.fig.text(0.75, 0.02, "", fontsize=12, va="bottom", ha="left", wrap=True)
        self.current_highlight_flit = None
        self.node_id = 0

    def _draw_modules(self, ch_names=None):
        """绘制 RingStation + CrossPoint 模块

        布局：
        - RS左侧: Input FIFOs (通道名 + TL, TR, TU, TD) 纵向排列
        - RS右侧: Output FIFOs (通道名 + TL, TR, TU, TD) 纵向排列
        - 每个FIFO横向展开显示槽位
        """
        square = self.square
        gap = self.gap
        fontsize = self.fontsize

        # 清空 patch 字典
        self.rs_patches.clear()
        self.rs_texts.clear()
        self.cph_patches.clear()
        self.cph_texts.clear()
        self.cpv_patches.clear()
        self.cpv_texts.clear()

        # 获取通道名列表
        if ch_names is None:
            ch_names = self.config.CH_NAME_LIST if hasattr(self.config, "CH_NAME_LIST") else []

        # === RingStation 模块 ===
        # 输入侧：通道名 + 环方向
        in_ch_lanes = [f"I_{ch}" for ch in ch_names]  # 按通道名
        in_ring_lanes = ["I_TL", "I_TR", "I_TU", "I_TD"]
        in_lanes = in_ch_lanes + in_ring_lanes
        in_depths = [self.rs_in_ch_depth] * len(ch_names) + [self.rs_in_fifo_depth] * 4

        # 输出侧：通道名 + 环方向
        out_ch_lanes = [f"O_{ch}" for ch in ch_names]  # 按通道名
        out_ring_lanes = ["O_TL", "O_TR", "O_TU", "O_TD"]
        out_lanes = out_ch_lanes + out_ring_lanes
        out_depths = [self.rs_out_ch_depth] * len(ch_names) + [self.rs_out_fifo_depth] * 4

        # 计算模块尺寸
        max_in_depth = max(in_depths) if in_depths else 4
        max_out_depth = max(out_depths) if out_depths else 4
        lane_height = square + gap
        num_lanes = max(len(in_lanes), len(out_lanes))
        rs_height = num_lanes * (lane_height + 0.1) + 0.4
        rs_width = (max_in_depth + max_out_depth) * (square + gap) + 1.5

        RS_x, RS_y = 0, 0

        # 绘制 RS 边框
        box = Rectangle((RS_x, RS_y), rs_width, rs_height, fill=False)
        self.ax.add_patch(box)
        self.ax.text(RS_x + rs_width / 2, RS_y + rs_height + 0.1, "RingStation",
                     ha="center", va="bottom", fontweight="bold")

        # 左侧 Input FIFOs
        for i, (lane, depth) in enumerate(zip(in_lanes, in_depths)):
            lane_y = RS_y + rs_height - (i + 1) * (lane_height + 0.1)
            lane_x = RS_x + 0.1

            # 标签
            self.ax.text(lane_x + depth * (square + gap) + 0.05, lane_y + square / 2,
                         lane, ha="left", va="center", fontsize=fontsize)

            self.rs_patches[lane] = []
            self.rs_texts[lane] = []

            for s in range(depth):
                slot_x = lane_x + s * (square + gap)
                slot_y = lane_y
                frame = Rectangle((slot_x, slot_y), square, square,
                                   edgecolor="black", facecolor="none",
                                   linewidth=self.slot_frame_lw, linestyle="--")
                self.ax.add_patch(frame)
                inner = Rectangle((slot_x, slot_y), square, square,
                                   edgecolor="none", facecolor="none", linewidth=0)
                self.ax.add_patch(inner)
                txt = self.ax.text(slot_x + square / 2, slot_y - 0.02, "",
                                   ha="center", va="top", fontsize=fontsize - 1)
                txt.set_visible(False)
                self.rs_patches[lane].append(inner)
                self.rs_texts[lane].append(txt)

        # 右侧 Output FIFOs
        for i, (lane, depth) in enumerate(zip(out_lanes, out_depths)):
            lane_y = RS_y + rs_height - (i + 1) * (lane_height + 0.1)
            lane_x = RS_x + rs_width - 0.1 - depth * (square + gap)

            # 标签
            self.ax.text(lane_x - 0.05, lane_y + square / 2,
                         lane, ha="right", va="center", fontsize=fontsize)

            self.rs_patches[lane] = []
            self.rs_texts[lane] = []

            for s in range(depth):
                slot_x = lane_x + s * (square + gap)
                slot_y = lane_y
                frame = Rectangle((slot_x, slot_y), square, square,
                                   edgecolor="black", facecolor="none",
                                   linewidth=self.slot_frame_lw, linestyle="--")
                self.ax.add_patch(frame)
                inner = Rectangle((slot_x, slot_y), square, square,
                                   edgecolor="none", facecolor="none", linewidth=0)
                self.ax.add_patch(inner)
                txt = self.ax.text(slot_x + square / 2, slot_y - 0.02, "",
                                   ha="center", va="top", fontsize=fontsize - 1)
                txt.set_visible(False)
                self.rs_patches[lane].append(inner)
                self.rs_texts[lane].append(txt)

        # === CrossPoint Horizontal 模块 ===
        cp_square = square * 1.5
        cp_gap = gap * 2
        CPH_x = RS_x
        CPH_y = RS_y - 2.5
        cp_h_width = 4 * (cp_square + cp_gap) + 0.4
        cp_h_height = 2 * (cp_square + cp_gap) + 0.4

        box_h = Rectangle((CPH_x, CPH_y), cp_h_width, cp_h_height, fill=False)
        self.ax.add_patch(box_h)
        self.ax.text(CPH_x + cp_h_width / 2, CPH_y + cp_h_height + 0.1, "CP_H",
                     ha="center", va="bottom", fontweight="bold")

        for i, lane in enumerate(["TR", "TL"]):
            lane_y = CPH_y + cp_h_height - (i + 1) * (cp_square + cp_gap) - 0.1
            lane_x = CPH_x + 0.2

            self.ax.text(lane_x - 0.05, lane_y + cp_square / 2,
                         lane, ha="right", va="center", fontsize=fontsize)

            self.cph_patches[lane] = []
            self.cph_texts[lane] = []

            for s in range(2):
                slot_x = lane_x + s * (cp_square + cp_gap)
                slot_y = lane_y
                frame = Rectangle((slot_x, slot_y), cp_square, cp_square,
                                   edgecolor="black", facecolor="none",
                                   linewidth=self.slot_frame_lw, linestyle="--")
                self.ax.add_patch(frame)
                inner = Rectangle((slot_x, slot_y), cp_square, cp_square,
                                   edgecolor="none", facecolor="none", linewidth=0)
                self.ax.add_patch(inner)
                txt = self.ax.text(slot_x + cp_square / 2, slot_y + cp_square / 2, "",
                                   ha="center", va="center", fontsize=fontsize)
                txt.set_visible(False)
                self.cph_patches[lane].append(inner)
                self.cph_texts[lane].append(txt)

        # === CrossPoint Vertical 模块 ===
        CPV_x = RS_x + rs_width + 0.5
        CPV_y = RS_y + rs_height / 2 - 1
        cp_v_width = 2 * (cp_square + cp_gap) + 0.4
        cp_v_height = 2 * (cp_square + cp_gap) + 0.4

        box_v = Rectangle((CPV_x, CPV_y), cp_v_width, cp_v_height, fill=False)
        self.ax.add_patch(box_v)
        self.ax.text(CPV_x + cp_v_width / 2, CPV_y + cp_v_height + 0.1, "CP_V",
                     ha="center", va="bottom", fontweight="bold")

        for i, lane in enumerate(["TD", "TU"]):
            lane_x = CPV_x + 0.2 + i * (cp_square + cp_gap)
            lane_y = CPV_y + 0.2

            self.ax.text(lane_x + cp_square / 2, CPV_y + cp_v_height + 0.02,
                         lane, ha="center", va="bottom", fontsize=fontsize)

            self.cpv_patches[lane] = []
            self.cpv_texts[lane] = []

            for s in range(2):
                slot_x = lane_x
                slot_y = lane_y + s * (cp_square + cp_gap)
                frame = Rectangle((slot_x, slot_y), cp_square, cp_square,
                                   edgecolor="black", facecolor="none",
                                   linewidth=self.slot_frame_lw, linestyle="--")
                self.ax.add_patch(frame)
                inner = Rectangle((slot_x, slot_y), cp_square, cp_square,
                                   edgecolor="none", facecolor="none", linewidth=0)
                self.ax.add_patch(inner)
                txt = self.ax.text(slot_x + cp_square / 2, slot_y + cp_square / 2, "",
                                   ha="center", va="center", fontsize=fontsize)
                txt.set_visible(False)
                self.cpv_patches[lane].append(inner)
                self.cpv_texts[lane].append(txt)

        self.ax.relim()
        self.ax.autoscale_view()

    def draw_piece_for_node(self, node_id, network):
        """v1 兼容接口"""
        self.update(network, node_id)

    def update(self, network, node_id):
        """更新 FIFO 显示"""
        self.patch_info_map.clear()
        self.current_highlight_flit = None

        # 获取通道名列表
        ch_names = network.config.CH_NAME_LIST if hasattr(network.config, "CH_NAME_LIST") else []

        # 每次都重新绘制模块（因为 piece_ax.clear() 会清除所有 patches）
        self._draw_modules(ch_names)

        self.node_id = node_id

        # 获取 RingStation
        rs = network.ring_stations.get(node_id)
        if rs is None:
            return

        CP_H = network.cross_point["horizontal"]
        CP_V = network.cross_point["vertical"]

        # 获取 channel buffer 数据（由 stats_mixin 同步）
        IQ_Ch = getattr(network, "IQ_channel_buffer", {})
        EQ_Ch = getattr(network, "EQ_channel_buffer", {})

        # 调试：打印 RingStation FIFO 详情
        net_type = getattr(network, 'network_type', 'unknown')
        in_total = sum(len(v) for v in rs.input_fifos.values())
        out_total = sum(len(v) for v in rs.output_fifos.values())
        if in_total > 0 or out_total > 0:
            print(f"\n[PieceVis] node={node_id}, net_type={net_type}")
            print(f"  input_fifos:")
            for k, v in rs.input_fifos.items():
                if len(v) > 0:
                    print(f"    {k}: {[str(f) for f in v]}")
            print(f"  output_fifos:")
            for k, v in rs.output_fifos.items():
                if len(v) > 0:
                    print(f"    {k}: {[str(f) for f in v]}")

        # lane名到FIFO的映射
        for lane, patches in self.rs_patches.items():
            if lane.startswith("I_"):
                key = lane[2:]  # 去掉 I_ 前缀
                if key in ["TL", "TR", "TU", "TD"]:
                    # 环方向 FIFO - 从 RingStation 获取
                    q = list(rs.input_fifos.get(key, []))
                else:
                    # 通道 buffer - 从 IQ_channel_buffer 获取
                    q = list(IQ_Ch.get(key, {}).get(node_id, []))
            elif lane.startswith("O_"):
                key = lane[2:]  # 去掉 O_ 前缀
                if key in ["TL", "TR", "TU", "TD"]:
                    # 环方向 FIFO - 从 RingStation 获取
                    q = list(rs.output_fifos.get(key, []))
                else:
                    # 通道 buffer - 从 EQ_channel_buffer 获取
                    q = list(EQ_Ch.get(key, {}).get(node_id, []))
            else:
                continue
            self._update_patches(patches, self.rs_texts[lane], q)

        # 更新 CrossPoint Horizontal
        for lane, patches in self.cph_patches.items():
            q = CP_H.get(node_id, {}).get(lane, [])
            if lane == "TL":
                q = q[::-1]
            self._update_patches(patches, self.cph_texts[lane], q)

        # 更新 CrossPoint Vertical
        for lane, patches in self.cpv_patches.items():
            q = CP_V.get(node_id, {}).get(lane, [])
            if lane == "TD":
                q = q[::-1]
            self._update_patches(patches, self.cpv_texts[lane], q)

        # 刷新信息框
        if self.use_highlight and self.current_highlight_flit is not None:
            self.info_text.set_text(str(self.current_highlight_flit))
        elif not self.use_highlight and self.current_highlight_flit is None:
            self.info_text.set_text("")

    def _update_patches(self, patches, texts, queue):
        """更新单个 FIFO 的 patches"""
        for idx, p in enumerate(patches):
            t = texts[idx]
            if idx < len(queue):
                flit = queue[idx]
                if flit is None:
                    p.set_facecolor("none")
                    t.set_visible(False)
                    continue

                packet_id = getattr(flit, "packet_id", None)
                flit_id = getattr(flit, "flit_id", str(flit))

                face, alpha, lw, edge = self.parent._get_flit_style(
                    flit,
                    use_highlight=self.use_highlight,
                    expected_packet_id=self.highlight_pid,
                )
                p.set_facecolor(face)
                p.set_alpha(alpha)
                p.set_linewidth(lw)
                p.set_edgecolor(edge)

                info = f"{packet_id}-{flit_id}"
                t.set_text(info)
                t.set_visible(self.use_highlight and packet_id == self.highlight_pid)
                self.patch_info_map[p] = (t, flit)

                if self.use_highlight and packet_id == self.highlight_pid:
                    self.current_highlight_flit = flit
            else:
                p.set_facecolor("none")
                t.set_visible(False)
                if p in self.patch_info_map:
                    self.patch_info_map.pop(p, None)

    def _on_click(self, event):
        """点击事件处理"""
        if event.inaxes != self.ax:
            return
        for patch, (txt, flit) in self.patch_info_map.items():
            contains, _ = patch.contains(event)
            if contains:
                pid = getattr(flit, "packet_id", None)
                fid = getattr(flit, "flit_id", None)
                if self.use_highlight and pid == self.highlight_pid:
                    vis = not txt.get_visible()
                    txt.set_visible(vis)
                    if vis:
                        txt.set_zorder(patch.get_zorder() + 1)
                self.info_text.set_text(str(flit))
                self.current_highlight_flit = flit
                if self.highlight_callback:
                    try:
                        self.highlight_callback(int(pid), int(fid))
                    except Exception:
                        pass
                self.fig.canvas.draw_idle()
                break
        else:
            self.info_text.set_text("")

    def sync_highlight(self, use_highlight, highlight_pid):
        """同步高亮状态"""
        self.use_highlight = use_highlight
        self.highlight_pid = highlight_pid

        for patch, (txt, flit) in self.patch_info_map.items():
            pid = getattr(flit, "packet_id", None)
            if self.use_highlight and pid == self.highlight_pid:
                txt.set_visible(True)
            else:
                txt.set_visible(False)
        if not self.use_highlight:
            self.info_text.set_text("")
