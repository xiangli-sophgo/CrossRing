"""
PieceVisualizer v1 - 适配 IQ/EQ/RB 架构的节点可视化

v1 架构使用分离的 Inject Queue / Eject Queue / Ring Bridge 结构
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from collections import defaultdict

from src.kcin.base.config import KCINConfigBase


class PieceVisualizerV1:
    """
    v1 架构的节点 FIFO 可视化器
    绘制 Inject Queue / Eject Queue / Ring Bridge / CrossPoint
    """

    def __init__(self, config: KCINConfigBase, ax, highlight_callback=None, parent=None):
        """
        仅绘制单个节点的 Inject/Eject Queue 和 Ring Bridge FIFO。
        """
        self.highlight_callback = highlight_callback
        self.config = config
        self.cols = config.NUM_COL
        self.rows = config.NUM_ROW
        self.parent = parent
        # 提取深度
        self.IQ_depth_horizontal = config.IQ_OUT_FIFO_DEPTH_HORIZONTAL
        self.IQ_depth_vertical = config.IQ_OUT_FIFO_DEPTH_VERTICAL
        self.IQ_depth_eq = config.IQ_OUT_FIFO_DEPTH_EQ
        self.EQ_depth = config.EQ_IN_FIFO_DEPTH
        self.RB_in_depth = config.RB_IN_FIFO_DEPTH
        self.RB_out_depth = config.RB_OUT_FIFO_DEPTH
        self.slice_per_link_horizontal = config.SLICE_PER_LINK_HORIZONTAL
        self.slice_per_link_vertical = config.SLICE_PER_LINK_VERTICAL
        self.IQ_CH_depth = config.IQ_CH_FIFO_DEPTH
        self.EQ_CH_depth = config.EQ_CH_FIFO_DEPTH

        # 几何参数
        self.square = 0.3
        self.gap = 0.02
        self.fifo_gap = 0.8
        self.fontsize = 8
        self.gap_lr = 0.35
        self.gap_hv = 0.35
        self.min_depth_vis = 4
        self.text_gap = 0.1
        self.slot_frame_lw = 0.4

        height = 8
        weight = 5
        self.inject_module_size = (height, weight)
        self.eject_module_size = (weight, height)
        self.rb_module_size = (height, height)
        self.cp_module_size = (2, 5)

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

        # 存储 patch 和 text
        self.iq_patches, self.iq_texts = {}, {}
        self.eq_patches, self.eq_texts = {}, {}
        self.rb_patches, self.rb_texts = {}, {}
        self.cph_patches, self.cph_texts = {}, {}
        self.cpv_patches, self.cpv_texts = {}, {}

        self._draw_modules()

        self.patch_info_map = {}
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.info_text = self.fig.text(0.75, 0.02, "", fontsize=12, va="bottom", ha="left", wrap=True)
        self.current_highlight_flit = None
        self.node_id = 0

    def _draw_modules(self):
        ch_names = self.config.CH_NAME_LIST
        iq_arbiter_lanes = ["arb_" + ch for ch in ch_names]

        iq_config = dict(
            title="Inject Queue",
            lanes=ch_names + iq_arbiter_lanes + ["TL", "TR", "TD", "TU", "EQ"],
            depths=[self.IQ_CH_depth] * len(ch_names) + [2] * len(ch_names) + [self.IQ_depth_horizontal, self.IQ_depth_horizontal, self.IQ_depth_vertical, self.IQ_depth_vertical, self.IQ_depth_eq],
            orientations=["vertical"] * len(ch_names) + ["vertical"] * len(ch_names) + ["vertical"] * 2 + ["horizontal"] * 3,
            h_pos=["top"] * len(ch_names) + ["top2"] * len(ch_names) + ["bottom"] * 2 + ["mid"] * 3,
            v_pos=["left"] * len(ch_names) + ["left"] * len(ch_names) + ["left"] * 2 + ["right"] * 3,
            patch_dict=self.iq_patches,
            text_dict=self.iq_texts,
        )

        eq_arbiter_lanes = ["arb_TD", "arb_TU", "arb_IQ", "arb_RB"]

        eq_config = dict(
            title="Eject Queue",
            lanes=ch_names + ["TD", "TU"] + eq_arbiter_lanes,
            depths=[self.EQ_CH_depth] * len(ch_names) + [self.EQ_depth] * 2 + [2] * 4,
            orientations=["horizontal"] * len(ch_names) + ["horizontal"] * 2 + ["horizontal"] * 4,
            h_pos=["top"] * len(ch_names) + ["bottom"] * 2 + ["mid"] * 4,
            v_pos=["left"] * len(ch_names) + ["right", "right"] + ["mid"] * 4,
            patch_dict=self.eq_patches,
            text_dict=self.eq_texts,
        )

        rb_config = dict(
            title="Ring Bridge",
            lanes=["TL", "TR", "TU", "TD", "EQ"],
            depths=[self.RB_in_depth] * 2 + [self.RB_out_depth] * 3,
            orientations=["vertical", "vertical", "horizontal", "horizontal", "vertical"],
            h_pos=["bottom", "bottom", "top", "top", "top"],
            v_pos=["left", "left", "right", "right", "left"],
            patch_dict=self.rb_patches,
            text_dict=self.rb_texts,
        )

        cross_point_horizontal_config = dict(
            title="CP",
            lanes=["TR", "TL"],
            depths=[2, 2],
            orientations=["horizontal", "horizontal"],
            h_pos=["bottom", "bottom"],
            v_pos=["right", "right"],
            patch_dict=self.cph_patches,
            text_dict=self.cph_texts,
        )

        cross_point_vertical_config = dict(
            title="CP",
            lanes=["TD", "TU"],
            depths=[2, 2],
            orientations=["vertical", "vertical"],
            h_pos=["bottom", "bottom"],
            v_pos=["left", "left"],
            patch_dict=self.cpv_patches,
            text_dict=self.cpv_texts,
        )

        def make_specs(c):
            specs = []
            for ori, hp, vp, d in zip(c["orientations"], c["h_pos"], c["v_pos"], c["depths"]):
                if ori[0].upper() == "H":
                    v_group = {"left": "L", "right": "R"}.get(vp, "M")
                    h_group = {"top": "T", "bottom": "B"}.get(hp, "M")
                    specs.append(("H", h_group, v_group, d))
                else:
                    v_group = {"left": "L", "right": "R"}.get(vp, "M")
                    h_group = {"top": "T", "bottom": "B"}.get(hp, "M")
                    specs.append(("V", h_group, v_group, d))
            return specs

        w_iq, h_iq = self._calc_module_size("IQ", make_specs(iq_config))
        w_eq, h_eq = self._calc_module_size("EQ", make_specs(eq_config))
        w_rb, h_rb = self._calc_module_size("RB", make_specs(rb_config))
        rb_h = max(h_iq, h_rb)
        rb_w = max(w_eq, w_rb)
        self.inject_module_size = (rb_h, w_iq)
        self.eject_module_size = (h_eq * 1.5, rb_w)
        self.rb_module_size = (rb_h, rb_w)

        center_x, center_y = 0, 0
        spacing = 1.2
        IQ_x = center_x - self.inject_module_size[1] - spacing
        IQ_y = center_y
        RB_x = center_x
        RB_y = center_y
        EQ_x = center_x
        EQ_y = center_y + self.rb_module_size[0] + spacing
        CPH_x = center_x - (self.inject_module_size[1] - spacing) / 3
        CPH_y = center_y - self.cp_module_size[0] - spacing
        CPV_x = center_x + self.rb_module_size[1] + spacing
        CPV_y = center_y + (self.rb_module_size[0] + spacing) * 2 / 3

        self._draw_fifo_module(x=IQ_x, y=IQ_y, title=iq_config["title"],
            module_height=self.inject_module_size[0], module_width=self.inject_module_size[1],
            lanes=iq_config["lanes"], depths=iq_config["depths"],
            orientations=iq_config["orientations"], h_position=iq_config["h_pos"],
            v_position=iq_config["v_pos"], patch_dict=iq_config["patch_dict"],
            text_dict=iq_config["text_dict"], per_lane_depth=True)

        self._draw_fifo_module(x=EQ_x, y=EQ_y, title=eq_config["title"],
            module_height=self.eject_module_size[0], module_width=self.eject_module_size[1],
            lanes=eq_config["lanes"], depths=eq_config["depths"],
            orientations=eq_config["orientations"], h_position=eq_config["h_pos"],
            v_position=eq_config["v_pos"], patch_dict=eq_config["patch_dict"],
            text_dict=eq_config["text_dict"], per_lane_depth=True)

        self._draw_fifo_module(x=RB_x, y=RB_y, title=rb_config["title"],
            module_height=self.rb_module_size[0], module_width=self.rb_module_size[1],
            lanes=rb_config["lanes"], depths=rb_config["depths"],
            orientations=rb_config["orientations"], h_position=rb_config["h_pos"],
            v_position=rb_config["v_pos"], patch_dict=rb_config["patch_dict"],
            text_dict=rb_config["text_dict"], per_lane_depth=True)

        self._draw_fifo_module(x=CPH_x, y=CPH_y, title=cross_point_horizontal_config["title"],
            module_height=self.cp_module_size[0], module_width=self.cp_module_size[1],
            lanes=cross_point_horizontal_config["lanes"], depths=cross_point_horizontal_config["depths"],
            orientations=cross_point_horizontal_config["orientations"],
            h_position=cross_point_horizontal_config["h_pos"],
            v_position=cross_point_horizontal_config["v_pos"],
            patch_dict=cross_point_horizontal_config["patch_dict"],
            text_dict=cross_point_horizontal_config["text_dict"], per_lane_depth=True)

        self._draw_fifo_module(x=CPV_x, y=CPV_y, title=cross_point_vertical_config["title"],
            module_height=self.cp_module_size[1], module_width=self.cp_module_size[0],
            lanes=cross_point_vertical_config["lanes"], depths=cross_point_vertical_config["depths"],
            orientations=cross_point_vertical_config["orientations"],
            h_position=cross_point_vertical_config["h_pos"],
            v_position=cross_point_vertical_config["v_pos"],
            patch_dict=cross_point_vertical_config["patch_dict"],
            text_dict=cross_point_vertical_config["text_dict"], per_lane_depth=True)

        self.ax.relim()
        self.ax.autoscale_view()

    def _draw_fifo_module(self, x, y, title, module_height, module_width,
                          lanes, depths, patch_dict, text_dict, per_lane_depth=False,
                          orientations=None, h_position="top", v_position="left",
                          title_position="left-up"):
        square = self.square
        gap = self.gap
        fontsize = self.fontsize
        if title == "CP":
            square *= 2
            gap *= 20
            fontsize = 8

        if orientations is None:
            orientations = ["horizontal"] * len(lanes)
        elif isinstance(orientations, str):
            orientations = [orientations] * len(lanes)

        if isinstance(h_position, str):
            h_position = [h_position if ori == "horizontal" else None for ori in orientations]
        if isinstance(v_position, str):
            v_position = [v_position if ori == "vertical" else None for ori in orientations]

        if per_lane_depth:
            lane_depths = depths
        else:
            lane_depths = [depths] * len(lanes)

        box = Rectangle((x, y), module_width, module_height, fill=False)
        self.ax.add_patch(box)

        title_x = x + module_width / 2
        title_y = y + module_height + 0.05
        self.ax.text(title_x, title_y, title, ha="center", va="bottom", fontweight="bold")

        patch_dict.clear()
        text_dict.clear()

        group_map = defaultdict(list)
        for i, (ori, hpos, vpos) in enumerate(zip(orientations, h_position, v_position)):
            group_map[(ori, hpos, vpos)].append(i)

        group_idx = {}
        for group, idxs in group_map.items():
            for j, i in enumerate(idxs):
                group_idx[i] = j

        for i, (lane, orient, depth) in enumerate(zip(lanes, orientations, lane_depths)):
            hpos = h_position[i]
            vpos = v_position[i]
            idx_in_group = group_idx[i]
            group_size = len(group_map[(orient, hpos, vpos)])

            if orient == "horizontal":
                if hpos == "top":
                    lane_y = y + module_height - ((idx_in_group + 1) * self.fifo_gap) - self.gap_hv
                    text_va = "bottom"
                elif hpos == "bottom":
                    lane_y = y + (idx_in_group * self.fifo_gap) + self.gap_hv
                    text_va = "top"
                elif hpos == "mid":
                    offset = (idx_in_group - (group_size - 1) / 2) * self.fifo_gap
                    lane_y = y + module_height / 2 + offset
                    text_va = "center"
                else:
                    raise ValueError(f"Unknown h_position: {hpos}")

                if vpos == "right":
                    lane_x = x + module_width - depth * (square + gap) - self.gap_lr
                    text_x = x + module_width - depth * (square + gap) - self.gap_lr - self.text_gap
                    slot_dir = 1
                    ha = "right"
                elif vpos == "left":
                    lane_x = x + self.gap_lr
                    text_x = x + self.gap_lr + depth * (square + gap) + self.text_gap
                    slot_dir = 1
                    ha = "left"
                elif vpos == "mid" or vpos is None:
                    lane_x = x + module_width / 2 - depth * (square + gap)
                    text_x = x + module_width / 2 - depth * (square + gap) - self.text_gap
                    slot_dir = 1
                    ha = "left"
                else:
                    raise ValueError(f"Unknown v_position: {vpos}")

                if lane[:2] in ["TL", "TR", "TU", "TD", "EQ"]:
                    self.ax.text(text_x, lane_y + square / 2, lane[:2].upper(), ha=ha, va="center", fontsize=fontsize)
                elif lane.startswith("arb_"):
                    if title == "Eject Queue":
                        port_name = lane.replace("arb_", "")
                        label = port_name + "_Ar"
                        label_x = lane_x + depth * (square + gap) + self.text_gap
                        self.ax.text(label_x, lane_y + square / 2, label, ha="left", va="center", fontsize=fontsize)
                else:
                    self.ax.text(text_x, lane_y + square / 2, lane[0].upper() + lane[-1], ha=ha, va="center", fontsize=fontsize)

                patch_dict[lane] = []
                text_dict[lane] = []

                for s in range(depth):
                    slot_x = lane_x + slot_dir * s * (square + gap)
                    slot_y = lane_y
                    frame = Rectangle((slot_x, slot_y), square, square, edgecolor="black",
                                       facecolor="none", linewidth=self.slot_frame_lw, linestyle="--")
                    self.ax.add_patch(frame)
                    inner = Rectangle((slot_x, slot_y), square, square, edgecolor="none",
                                       facecolor="none", linewidth=0)
                    self.ax.add_patch(inner)
                    txt = self.ax.text(slot_x, slot_y + (square / 2 + 0.005 if hpos == "top" else -square / 2 - 0.005),
                                       "", ha="center", va=text_va, fontsize=fontsize)
                    txt.set_visible(False)
                    patch_dict[lane].append(inner)
                    text_dict[lane].append(txt)

            elif orient == "vertical":
                if vpos == "left":
                    lane_x = x + (idx_in_group * self.fifo_gap) + self.gap_lr
                    text_ha = "right"
                elif vpos == "right":
                    lane_x = x + module_width - (idx_in_group * self.fifo_gap) - self.gap_lr
                    text_ha = "left"
                elif vpos == "mid" or vpos is None:
                    offset = (idx_in_group - (group_size - 1) / 2) * self.fifo_gap
                    lane_x = x + offset
                    text_ha = "center"
                else:
                    raise ValueError(f"Unknown v_position: {vpos}")

                if hpos == "top":
                    lane_y = y + module_height - depth * (square + gap) - self.gap_hv
                    text_y = y + module_height - depth * (square + gap) - self.gap_hv - self.text_gap
                    slot_dir = 1
                    va = "top"
                elif hpos == "top2":
                    top_group_bottom = y + module_height - self.IQ_CH_depth * (square + gap) - self.gap_hv
                    lane_y = top_group_bottom - self.fifo_gap - depth * (square + gap)
                    text_y = lane_y - self.text_gap
                    slot_dir = 1
                    va = "top"
                elif hpos == "bottom":
                    lane_y = y + self.gap_hv
                    text_y = y + self.gap_hv + depth * (square + gap) + self.text_gap
                    slot_dir = 1
                    va = "bottom"
                elif hpos == "mid" or hpos is None:
                    lane_y = y + module_height / 2 - (depth / 2) * (square + gap)
                    text_y = y + module_height / 2 - (depth / 2) * (square + gap) - self.text_gap
                    slot_dir = 1
                    va = "center"
                else:
                    raise ValueError(f"Unknown h_position: {hpos}")

                if lane[:2] in ["TL", "TR", "TU", "TD", "EQ"]:
                    self.ax.text(lane_x + square / 2, text_y, lane[:2].upper(), ha="center", va=va, fontsize=fontsize)
                elif lane.startswith("arb_"):
                    if title == "Eject Queue":
                        port_name = lane.replace("arb_", "")
                        label = port_name + "_Ar"
                        self.ax.text(lane_x + square / 2, text_y, label, ha="center", va=va, fontsize=fontsize)
                else:
                    self.ax.text(lane_x + square / 2, text_y, lane[0].upper() + lane[-1], ha="center", va=va, fontsize=fontsize)

                patch_dict[lane] = []
                text_dict[lane] = []

                for s in range(depth):
                    slot_x = lane_x
                    slot_y = lane_y + slot_dir * s * (square + gap)
                    frame = Rectangle((slot_x, slot_y), square, square, edgecolor="black",
                                       facecolor="none", linewidth=self.slot_frame_lw, linestyle="--")
                    self.ax.add_patch(frame)
                    inner = Rectangle((slot_x, slot_y), square, square, edgecolor="none",
                                       facecolor="none", linewidth=0)
                    self.ax.add_patch(inner)
                    txt = self.ax.text(slot_x + (square / 2 + 0.005 if vpos == "right" else -square / 2 - 0.005),
                                       slot_y, "", ha=text_ha, va="center", fontsize=fontsize)
                    txt.set_visible(False)
                    patch_dict[lane].append(inner)
                    text_dict[lane].append(txt)

    def _calc_module_size(self, module_type, fifo_specs):
        _ = module_type
        max_depth = {k: 0 for k in ("L", "M_h", "R", "T", "M_v", "B")}
        cnt_H = {"L": 0, "M": 0, "R": 0}
        cnt_V = {"T": 0, "M": 0, "B": 0}

        for o, h_grp, v_grp, d in fifo_specs:
            if o == "H":
                g = v_grp or "M"
                key = "M_h" if g == "M" else g
                max_depth[key] = max(max_depth[key], d)
                cnt_H[g] += 1
            else:
                g = h_grp or "M"
                key = "M_v" if g == "M" else g
                max_depth[key] = max(max_depth[key], d)
                cnt_V[g] += 1

        count_H = max(cnt_H.values())
        count_V = max(cnt_V.values())

        width_slots = max_depth["L"] + max_depth["M_h"] + max_depth["R"] + count_V * 2 + 4
        height_slots = max_depth["T"] + max_depth["M_v"] + max_depth["B"] + count_H * 2 + 4

        width = width_slots * (self.square + self.gap) + 4 * self.gap_lr
        height = height_slots * (self.square + self.gap) + 4 * self.gap_hv
        return width, height

    def update(self, network, node_id):
        """更新 FIFO 显示 (v1 接口名为 draw_piece_for_node)"""
        self.draw_piece_for_node(node_id, network)

    def draw_piece_for_node(self, node_id, network):
        """更新当前节点的 FIFO 状态"""
        self.patch_info_map.clear()
        self.current_highlight_flit = None

        if len(self.ax.patches) == 0:
            self._draw_modules()

        self.node_id = node_id
        IQ = network.inject_queues
        EQ = network.eject_queues
        RB = network.ring_bridge
        IQ_Ch = network.IQ_channel_buffer
        EQ_Ch = network.EQ_channel_buffer
        CP_H = network.cross_point["horizontal"]
        CP_V = network.cross_point["vertical"]

        # Inject Queue
        for lane, patches in self.iq_patches.items():
            if lane.startswith("arb_"):
                ip_type = lane.replace("arb_", "")
                if self.node_id >= len(network.IQ_arbiter_input_fifo.get(ip_type, [])):
                    continue
                q = network.IQ_arbiter_input_fifo[ip_type][self.node_id]
            elif "_" in lane:
                ch_list = IQ_Ch.get(lane, [])
                if self.node_id >= len(ch_list):
                    continue
                q = ch_list[self.node_id]
            else:
                iq_list = IQ.get(lane, [])
                if self.node_id >= len(iq_list):
                    continue
                q = iq_list[self.node_id]
            self._update_patches(patches, self.iq_texts[lane], q)

        # Eject Queue
        for lane, patches in self.eq_patches.items():
            if lane.startswith("arb_"):
                port_name = lane.replace("arb_", "")
                if self.node_id >= len(network.EQ_arbiter_input_fifo.get(port_name, [])):
                    continue
                q = network.EQ_arbiter_input_fifo[port_name][self.node_id]
            elif "_" in lane:
                ch_list = EQ_Ch.get(lane, [])
                if self.node_id >= len(ch_list):
                    continue
                q = ch_list[self.node_id]
            else:
                eq_list = EQ.get(lane, [])
                if self.node_id >= len(eq_list):
                    continue
                q = eq_list[self.node_id]
            self._update_patches(patches, self.eq_texts[lane], q)

        # Ring Bridge
        for lane, patches in self.rb_patches.items():
            q = RB.get(lane, [])[self.node_id]
            self._update_patches(patches, self.rb_texts[lane], q)

        # CrossPoint Horizontal
        for lane, patches in self.cph_patches.items():
            q = CP_H.get(self.node_id, [])[lane]
            if lane == "TL":
                q = q[::-1]
            self._update_patches(patches, self.cph_texts[lane], q)

        # CrossPoint Vertical
        for lane, patches in self.cpv_patches.items():
            q = CP_V.get(self.node_id, [])[lane]
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
