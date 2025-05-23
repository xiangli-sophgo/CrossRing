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
from config.config import CrossRingConfig

# 引入节点局部 CrossRing piece 绘制函数（若存在）
# from .CrossRing_Piece_Visualizer import CrossRingVisualizer
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict
from matplotlib.patches import FancyArrowPatch


class NetworkLinkVisualizer:
    class PieceVisualizer:
        def __init__(self, config: CrossRingConfig, ax, highlight_callback=None):
            """
            仅绘制单个节点的 Inject/Eject Queue 和 Ring Bridge FIFO。
            参数:
            - config: 含有 FIFO 深度配置的对象，属性包括 cols, num_nodes, IQ_OUT_FIFO_DEPTH,
              EQ_IN_FIFO_DEPTH, RB_IN_FIFO_DEPTH, RB_OUT_FIFO_DEPTH
            - node_id: 要可视化的节点索引 (0 到 num_nodes-1)
            """
            self.highlight_callback = highlight_callback
            self.config = config
            self.cols = config.NUM_COL
            self.rows = config.NUM_ROW
            # 提取深度
            self.IQ_depth = config.IQ_OUT_FIFO_DEPTH
            self.EQ_depth = config.EQ_IN_FIFO_DEPTH
            self.RB_in_depth = config.RB_IN_FIFO_DEPTH
            self.RB_out_depth = config.RB_OUT_FIFO_DEPTH
            self.seats_per_link = config.SEAT_PER_LINK
            self.IQ_CH_depth = config.IQ_CH_FIFO_DEPTH
            self.EQ_CH_depth = config.EQ_CH_FIFO_DEPTH
            # 固定几何参数
            self.square = 0.3  # flit 方块边长
            self.gap = 0.04  # 相邻槽之间间距
            self.fifo_gap = 0.8  # 相邻fifo之间间隙

            # ------- layout tuning parameters (all adjustable) -------
            self.gap_lr = 0.3  # 左右内边距
            self.gap_hv = 0.3  # 上下内边距
            self.min_depth_vis = 4  # 设计最小深度 (=4)
            self.text_gap = 0.1
            # ---------------------------------------------------------

            height = 8
            weight = 5
            self.inject_module_size = (height, weight)
            self.eject_module_size = (weight, height)
            self.rb_module_size = (height, height)
            # 初始化图形
            if ax is None:
                self.fig, self.ax = plt.subplots(figsize=(8, 6))  # 增大图形尺寸
            else:
                self.ax = ax
                self.fig = ax.figure
            self.ax.axis("off")
            self.ax.set_aspect("equal")
            # 调色板
            self._colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            # ------ highlight / tracking ------
            self.use_highlight = False  # 是否启用高亮模式
            self.highlight_pid = None  # 被追踪的 packet_id
            self.highlight_color = "red"  # 追踪 flit 颜色
            self.grey_color = "lightgrey"  # 其它 flit 颜色

            # 存储 patch 和 text
            self.iq_patches, self.iq_texts = {}, {}
            self.eq_patches, self.eq_texts = {}, {}
            self.rb_patches, self.rb_texts = {}, {}
            self.lh_patches, self.cph_texts = {}, {}
            self.lv_patches, self.cpv_texts = {}, {}
            # 画出三个模块的框和 FIFO 槽
            self._draw_modules()
            # self._draw_arrows()

            # 点击显示 flit 信息
            self.patch_info_map = {}  # patch -> (text_obj, info_str)
            self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        def _draw_arrows(self):
            # TODO: not Finished
            # 1. 模块几何信息（必须与 _draw_modules 中的保持一致）
            IQ_x, IQ_y, IQ_w, IQ_h = -4, 0.0, self.inject_module_size
            EQ_x, EQ_y, EQ_w, EQ_h = 0.0, 4, self.eject_module_size
            RB_x, RB_y, RB_w, RB_h = 0.0, 0.0, self.rb_module_size

            # 2. 公共箭头样式基础
            base_style = dict(arrowstyle="-|>", color="black", lw=3.5, mutation_scale=15)

            # 3. 原来的三条箭头，改用 arc3,rad 控制圆润度
            # 箭头1: IQ 右侧上半 → EQ 下侧左半
            A = (IQ_x + IQ_w / 2, IQ_y + IQ_h * 0.25)
            B = (EQ_x - EQ_w / 2, EQ_y - EQ_h / 2)
            style = base_style.copy()
            style["connectionstyle"] = "arc3,rad=0"
            self.ax.add_patch(FancyArrowPatch(posA=A, posB=B, **style))

            # 箭头2: IQ 右侧下半 → RB 左侧中点
            C = (IQ_x + IQ_w / 2, IQ_y - IQ_h * 0.25)
            D = (RB_x - RB_w / 2, RB_y)
            style = base_style.copy()
            style["connectionstyle"] = "arc3,rad=0"
            self.ax.add_patch(FancyArrowPatch(posA=C, posB=D, **style))

            # 箭头3: IQ 底部中点 向下
            E = (IQ_x, IQ_y - IQ_h / 2)
            F = (IQ_x, IQ_y - IQ_h / 2 - 1.0)
            style = base_style.copy()
            style["connectionstyle"] = "arc3,rad=0"  # 直线
            self.ax.add_patch(FancyArrowPatch(posA=E, posB=F, **style))

            # 4. 新增四条箭头
            # 箭头4: RB 底部中点 向下
            G = (RB_x, RB_y - RB_h / 2)
            H = (RB_x, RB_y - RB_h / 2 - 1.0)
            style = base_style.copy()
            style["connectionstyle"] = "arc3,rad=0"
            self.ax.add_patch(FancyArrowPatch(posA=G, posB=H, **style))

            # 箭头5: RB 顶部中点 → EQ 底部中点
            I = (RB_x, RB_y + RB_h / 2)
            J = (EQ_x, EQ_y - EQ_h / 2)
            style = base_style.copy()
            style["connectionstyle"] = "arc3,rad=0"
            self.ax.add_patch(FancyArrowPatch(posA=I, posB=J, **style))

            # 箭头6: RB 右侧中点 向右
            K = (RB_x + RB_w / 2, RB_y)
            L = (RB_x + RB_w / 2 + 1.0, RB_y)
            style = base_style.copy()
            style["connectionstyle"] = "arc3,rad=0"
            self.ax.add_patch(FancyArrowPatch(posA=K, posB=L, **style))

            # 箭头7: EQ 右侧中点 → 向右的一段距离（指向中点）
            M = (EQ_x + EQ_w / 2, EQ_y)
            N = (EQ_x + EQ_w / 2 + 1.0, EQ_y)
            style = base_style.copy()
            style["connectionstyle"] = "arc3,rad=0"
            self.ax.add_patch(FancyArrowPatch(posA=M, posB=N, **style))

        def _draw_modules(self):
            # ---- collect fifo specs ---------------------------------
            ch_names = self.config.CH_NAME_LIST
            # ------- absolute positions (keep spacing param) ----------
            # ------------------- unified module configs ------------------- #
            iq_config = dict(
                title="Inject Queue",
                lanes=ch_names + ["TL", "TR", "EQ", "TU", "TD"],
                depths=[self.IQ_CH_depth] * len(ch_names) + [self.IQ_depth] * 5,
                orientations=["vertical"] * len(ch_names) + ["vertical"] * 2 + ["horizontal"] * 3,
                h_pos=["top"] * len(ch_names) + ["bottom"] * 2 + ["mid"] * 3,
                v_pos=["left"] * len(ch_names) + ["left"] * 2 + ["right"] * 3,
                patch_dict=self.iq_patches,
                text_dict=self.iq_texts,
            )

            eq_config = dict(
                title="Eject Queue",
                lanes=ch_names + ["TU", "TD"],
                depths=[self.EQ_CH_depth] * len(ch_names) + [self.EQ_depth] * 2,
                orientations=["horizontal"] * len(ch_names) + ["horizontal"] * 2,
                h_pos=["top"] * len(ch_names) + ["bottom"] * 2,
                v_pos=["left"] * len(ch_names) + ["right", "right"],
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

            # ---------------- compute sizes via fifo specs ---------------- #
            def make_specs(c):
                """
                Build a list of (orient, h_group, v_group, depth) for each fifo lane.
                Each spec tuple is (orient, h_group, v_group, depth), unused group is None.
                """
                specs = []
                for ori, hp, vp, d in zip(c["orientations"], c["h_pos"], c["v_pos"], c["depths"]):
                    if ori[0].upper() == "H":
                        v_group = {"left": "L", "right": "R"}.get(vp, "M")
                        h_group = {"top": "T", "bottom": "B"}.get(hp, "M")
                        specs.append(("H", h_group, v_group, d))
                    else:  # vertical
                        v_group = {"left": "L", "right": "R"}.get(vp, "M")
                        h_group = {"top": "T", "bottom": "B"}.get(hp, "M")
                        specs.append(("V", h_group, v_group, d))
                return specs

            w_iq, h_iq = self._calc_module_size("IQ", make_specs(iq_config))
            w_eq, h_eq = self._calc_module_size("EQ", make_specs(eq_config))
            w_rb, h_rb = self._calc_module_size("RB", make_specs(rb_config))
            rb_h = max(h_iq, h_rb)
            rb_w = max(w_eq, w_rb)
            # self.inject_module_size = (w_iq, rb_h)
            # self.eject_module_size = (rb_w, h_eq)
            # self.rb_module_size = (rb_w, rb_h)
            self.inject_module_size = (rb_h, w_iq)
            self.eject_module_size = (h_eq, rb_w)
            self.rb_module_size = (rb_h, rb_w)
            # self.inject_module_size = (6, 4)
            # self.eject_module_size = (4, 6)
            # self.rb_module_size = (6, 6)

            center_x, center_y = 0, 0
            spacing = 1.5
            IQ_x = center_x - self.inject_module_size[1] - spacing
            IQ_y = center_y
            RB_x = center_x
            RB_y = center_y
            EQ_x = center_x
            EQ_y = center_y + self.rb_module_size[0] + spacing

            # ---------------------- draw modules -------------------------- #
            self._draw_fifo_module(
                x=IQ_x,
                y=IQ_y,
                title=iq_config["title"],
                module_height=self.inject_module_size[0],
                module_width=self.inject_module_size[1],
                lanes=iq_config["lanes"],
                depths=iq_config["depths"],
                orientations=iq_config["orientations"],
                h_position=iq_config["h_pos"],
                v_position=iq_config["v_pos"],
                patch_dict=iq_config["patch_dict"],
                text_dict=iq_config["text_dict"],
                per_lane_depth=True,
            )

            self._draw_fifo_module(
                x=EQ_x,
                y=EQ_y,
                title=eq_config["title"],
                module_height=self.eject_module_size[0],
                module_width=self.eject_module_size[1],
                lanes=eq_config["lanes"],
                depths=eq_config["depths"],
                orientations=eq_config["orientations"],
                h_position=eq_config["h_pos"],
                v_position=eq_config["v_pos"],
                patch_dict=eq_config["patch_dict"],
                text_dict=eq_config["text_dict"],
                per_lane_depth=True,
            )

            self._draw_fifo_module(
                x=RB_x,
                y=RB_y,
                title=rb_config["title"],
                module_height=self.rb_module_size[0],
                module_width=self.rb_module_size[1],
                lanes=rb_config["lanes"],
                depths=rb_config["depths"],
                orientations=rb_config["orientations"],
                h_position=rb_config["h_pos"],
                v_position=rb_config["v_pos"],
                patch_dict=rb_config["patch_dict"],
                text_dict=rb_config["text_dict"],
                per_lane_depth=True,
            )

            self.ax.relim()
            self.ax.autoscale_view()

        def _draw_fifo_module(
            self,
            x,
            y,
            title,
            module_height,
            module_width,
            lanes,
            depths,
            patch_dict,
            text_dict,
            per_lane_depth=False,
            orientations=None,
            h_position="top",
            v_position="left",
            title_position="left-up",
        ):
            """
            支持 hpos/vpos 联合定位的 FIFO 绘制
            """
            square = self.square
            gap = self.gap

            # 处理方向参数
            if orientations is None:
                orientations = ["horizontal"] * len(lanes)
            elif isinstance(orientations, str):
                orientations = [orientations] * len(lanes)

            # 处理 h_position/v_position 支持列表
            if isinstance(h_position, str):
                h_position = [h_position if ori == "horizontal" else None for ori in orientations]
            if isinstance(v_position, str):
                v_position = [v_position if ori == "vertical" else None for ori in orientations]

            if not (len(h_position) == len(v_position) == len(lanes)):
                raise ValueError("h_position, v_position, lanes must have the same length")

            # 处理 depth
            if per_lane_depth:
                lane_depths = depths
            else:
                lane_depths = [depths] * len(lanes)

            # 绘制模块边框
            box = Rectangle((x, y), module_width, module_height, fill=False)
            self.ax.add_patch(box)

            # 模块标题
            title_x = x + module_width / 2
            title_y = y + module_height + 0.05
            self.ax.text(title_x, title_y, title, ha="center", va="bottom", fontweight="bold")

            patch_dict.clear()
            text_dict.clear()

            # 分组并组内编号
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
                    # 纵坐标由 hpos 决定
                    if hpos == "top":
                        lane_y = y + module_height - ((idx_in_group + 1) * self.fifo_gap) - self.gap_hv
                        text_va = "bottom"
                    elif hpos == "bottom":
                        lane_y = y + (idx_in_group * self.fifo_gap) + self.gap_hv
                        text_va = "top"
                    elif hpos == "mid":
                        lane_y = y + module_height / 2 + (idx_in_group - 1) * self.fifo_gap
                        text_va = "center"
                    else:
                        raise ValueError(f"Unknown h_position: {hpos}")

                    # 横坐标由 vpos 决定
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

                    self.ax.text(text_x, lane_y + square / 2, lane[0].upper() + lane[-1], ha=ha, va="center", fontsize=9)
                    patch_dict[lane] = []
                    text_dict[lane] = []

                    for s in range(depth):
                        slot_x = lane_x + slot_dir * s * (square + gap)
                        slot_y = lane_y
                        patch = Rectangle((slot_x, slot_y), square, square, edgecolor="black", facecolor="none")
                        self.ax.add_patch(patch)
                        txt = self.ax.text(slot_x, slot_y + (square / 2 + 0.005 if hpos == "top" else -square / 2 - 0.005), "", ha="center", va=text_va, fontsize=9)
                        txt.set_visible(False)  # 默认隐藏
                        patch_dict[lane].append(patch)
                        text_dict[lane].append(txt)

                elif orient == "vertical":
                    # 横坐标由 vpos 决定
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

                    # 纵坐标由 hpos 决定
                    if hpos == "top":
                        lane_y = y + module_height - depth * (square + gap) - self.gap_hv
                        text_y = y + module_height - depth * (square + gap) - self.gap_hv - self.text_gap
                        slot_dir = 1
                        va = "top"
                    elif hpos == "bottom":
                        lane_y = y + self.gap_hv
                        text_y = y + self.gap_hv + depth * (square + gap) + self.text_gap
                        slot_dir = 1
                        va = "bottom"
                    elif hpos == "mid" or hpos is None:
                        lane_y = y - (depth / 2) * (square + gap)
                        slot_dir = 1
                        va = "center"
                    else:
                        raise ValueError(f"Unknown h_position: {hpos}")

                    self.ax.text(lane_x + square / 2, text_y, lane[0].upper() + lane[-1], ha="center", va=va, fontsize=9)
                    patch_dict[lane] = []
                    text_dict[lane] = []

                    for s in range(depth):
                        slot_x = lane_x
                        slot_y = lane_y + slot_dir * s * (square + gap)
                        patch = Rectangle((slot_x, slot_y), square, square, edgecolor="black", facecolor="none")
                        self.ax.add_patch(patch)
                        txt = self.ax.text(slot_x + (square / 2 + 0.005 if vpos == "right" else -square / 2 - 0.005), slot_y, "", ha=text_ha, va="center", fontsize=9)
                        txt.set_visible(False)  # 默认隐藏
                        patch_dict[lane].append(patch)
                        text_dict[lane].append(txt)

                else:
                    raise ValueError(f"Unknown orientation: {orient}")

        def _get_color(self, flit):
            """返回矩形槽的填充颜色；支持“高亮追踪模式”。"""
            pid = getattr(flit, "packet_id", 0)

            # --- 高亮模式：目标 flit → 红，其余 → 灰 -----------------
            if self.use_highlight:
                return self.highlight_color if pid == self.highlight_pid else self.grey_color

            # --- 普通模式：按 packet_id 轮询调色板（无缓存） ----------
            return self._colors[pid % len(self._colors)]

        def draw_piece_for_node(self, node_id, network):
            """
            更新当前节点的 FIFO 状态。
            state: { 'inject': {...}, 'eject': {...}, 'ring_bridge': {...} }
            """
            # 清空旧的 patch->info 映射
            self.patch_info_map.clear()
            # --------------------------------------------------------------
            # 若外部 (Link_State_Visualizer) 清除了坐标轴，需要重新画框架
            # --------------------------------------------------------------
            if len(self.ax.patches) == 0:  # 轴内无任何图元，说明已被 clear()
                self._draw_modules()  # 重建 FIFO / RB 边框与槽
                # 如果还想显示所有 links，可取消下一行注释
                # self._draw_links_module()

            self.node_id = node_id
            IQ = network.inject_queues
            EQ = network.eject_queues
            RB = network.ring_bridge
            IQ_Ch = network.IQ_channel_buffer
            EQ_Ch = network.EQ_channel_buffer
            # Inject
            for lane, patches in self.iq_patches.items():
                if "_" in lane:
                    q = IQ_Ch.get(lane, [])[self.node_id]
                else:
                    q = IQ.get(lane, [])[self.node_id]
                for idx, p in enumerate(patches):
                    t = self.iq_texts[lane][idx]
                    if idx < len(q):
                        flit = q[idx]
                        packet_id = getattr(flit, "packet_id", None)
                        flit_id = getattr(flit, "flit_id", str(flit))

                        # 创建复合ID对象
                        pid = {"packet_id": packet_id, "flit_id": flit_id}

                        # 设置颜色（基于packet_id）和显示文本
                        p.set_facecolor(self._get_color(flit))
                        info = f"{packet_id}-{flit_id}"
                        t.set_text(info)
                        t.set_visible(self.use_highlight and packet_id == self.highlight_pid)
                        self.patch_info_map[p] = (t, packet_id, flit_id)
                    else:
                        p.set_facecolor("none")
                        t.set_visible(False)
                        if p in self.patch_info_map:
                            self.patch_info_map.pop(p, None)
            # Eject
            for lane, patches in self.eq_patches.items():
                if "_" in lane:
                    q = EQ_Ch.get(lane, [])[self.node_id - self.cols]
                else:
                    q = EQ.get(lane, [])[self.node_id - self.cols]
                for idx, p in enumerate(patches):
                    t = self.eq_texts[lane][idx]
                    if idx < len(q):
                        flit = q[idx]
                        packet_id = getattr(flit, "packet_id", None)
                        flit_id = getattr(flit, "flit_id", str(flit))

                        # 创建复合ID对象
                        pid = {"packet_id": packet_id, "flit_id": flit_id}

                        # 设置颜色（基于packet_id）和显示文本
                        p.set_facecolor(self._get_color(flit))
                        info = f"{packet_id}-{flit_id}"
                        t.set_text(info)
                        t.set_visible(self.use_highlight and packet_id == self.highlight_pid)
                        self.patch_info_map[p] = (t, packet_id, flit_id)
                    else:
                        p.set_facecolor("none")
                        t.set_visible(False)
                        if p in self.patch_info_map:
                            self.patch_info_map.pop(p, None)
            # Ring Bridge
            for lane, patches in self.rb_patches.items():
                q = RB.get(lane, [])[(self.node_id, self.node_id - self.cols)]
                for idx, p in enumerate(patches):
                    t = self.rb_texts[lane][idx]
                    if idx < len(q):
                        flit = q[idx]
                        packet_id = getattr(flit, "packet_id", None)
                        flit_id = getattr(flit, "flit_id", str(flit))

                        # 创建复合ID对象
                        pid = {"packet_id": packet_id, "flit_id": flit_id}

                        # 设置颜色（基于packet_id）和显示文本
                        p.set_facecolor(self._get_color(flit))
                        info = f"{packet_id}-{flit_id}"
                        t.set_text(info)
                        t.set_visible(self.use_highlight and packet_id == self.highlight_pid)
                        self.patch_info_map[p] = (t, packet_id, flit_id)
                    else:
                        p.set_facecolor("none")
                        t.set_visible(False)
                        if p in self.patch_info_map:
                            self.patch_info_map.pop(p, None)

            # plt.title(f"Node: {self.node_id}", fontsize=12)
            # plt.pause(0.2)

        # ------------------------------------------------------------------ #
        #  点击 flit 矩形时显示 / 隐藏文字                                      #
        # ------------------------------------------------------------------ #
        def _on_click(self, event):
            if event.inaxes != self.ax:
                return
            for patch, (txt, pid, fid) in self.patch_info_map.items():
                contains, _ = patch.contains(event)
                if contains:
                    # 切换可见性
                    vis = not txt.get_visible()
                    txt.set_visible(vis)
                    # 若即将显示，确保在最上层
                    if vis:
                        txt.set_zorder(patch.get_zorder() + 1)
                    # -> 通知父级高亮
                    if self.highlight_callback:
                        try:
                            self.highlight_callback(int(pid), int(fid))
                        except Exception:
                            pass
                    self.fig.canvas.draw_idle()
                    break

        # ------------------------------------------------------------------ #
        #  计算模块尺寸 (宽 = X 方向, 高 = Y 方向)                             #
        # ------------------------------------------------------------------ #
        def _calc_module_size(self, module_type, fifo_specs):
            """
            fifo_specs: list of tuples (orient, h_group, v_group, depth)
            - orient: 'H' or 'V'
            - h_group: for V → 'T' | 'M' | 'B', else None
            - v_group: for H → 'L' | 'M' | 'R', else None
            - depth: int
            The size is determined by the max depth in each group (per orientation), plus number of orthogonal FIFOs.
            """
            _ = module_type  # unused, retained for compatibility

            # ----- max depth per slot (L/M/R  and  T/M/B) -----------------
            max_depth = {k: 0 for k in ("L", "M_h", "R", "T", "M_v", "B")}

            # counts per side group
            cnt_H = {"L": 0, "M": 0, "R": 0}  # horizontal fifo counts by v_group
            cnt_V = {"T": 0, "M": 0, "B": 0}  # vertical   fifo counts by h_group

            for o, h_grp, v_grp, d in fifo_specs:
                if o == "H":
                    # horizontal -> depth to L/M_h/R & count into cnt_H
                    g = v_grp or "M"
                    key = "M_h" if g == "M" else g
                    max_depth[key] = max(max_depth[key], d)
                    cnt_H[g] += 1
                else:  # 'V'
                    g = h_grp or "M"
                    key = "M_v" if g == "M" else g
                    max_depth[key] = max(max_depth[key], d)
                    cnt_V[g] += 1

            # take MAX count across side groups (per requirement)
            count_H = max(cnt_H.values())  # horizontal fifo effective count
            count_V = max(cnt_V.values())  # vertical fifo effective count

            width_slots = max_depth["L"] + max_depth["M_h"] + max_depth["R"] + count_V * 2 + 4
            height_slots = max_depth["T"] + max_depth["M_v"] + max_depth["B"] + count_H * 2 + 4

            width = width_slots * (self.square + self.gap) + 4 * self.gap_lr
            height = height_slots * (self.square + self.gap) + 4 * self.gap_hv
            return width, height

    def __init__(self, network):
        self.network = network
        self.cols = network.config.NUM_COL
        # ---- Figure & Sub‑Axes ------------------------------------------------
        self.fig = plt.figure(figsize=(15, 8), constrained_layout=True)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
        self.ax = self.fig.add_subplot(gs[0])  # 主网络视图
        self.piece_ax = self.fig.add_subplot(gs[1])  # 右侧 Piece 视图
        self.piece_ax.axis("off")
        self.ax.set_aspect("equal")
        self.piece_vis = self.PieceVisualizer(self.network.config, self.piece_ax, highlight_callback=self._on_piece_highlight)
        # 当前点击选中的节点 (None 表示未选)
        self._selected_node = None
        # 绘制主网络的静态元素
        self.node_positions = self._calculate_layout()
        self.link_artists = {}  # 存储链路相关的静态信息
        self._colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        self.cycle = 0
        self.paused = False
        # ============  flit‑click tracking ==============
        self.tracked_pid = None  # 当前追踪的 packet_id (None = 不追踪)
        self.rect_info_map = {}  # rect → (text_obj, packet_id)
        self.fig.canvas.mpl_connect("button_press_event", self._on_flit_click)
        # 颜色/高亮控制
        self._use_highlight = False
        self._expected_pid = 0
        # ===============  History Buffer  ====================
        # 支持多网络显示
        self.networks = None
        self.selected_network_index = 0
        # 为每个网络维护独立历史缓冲
        self.histories = [deque(maxlen=20) for _ in range(3)]
        self.buttons = []
        # 添加网络选择按钮
        btn_positions = [(0.01, 0.03, 0.06, 0.04), (0.10, 0.03, 0.06, 0.04), (0.19, 0.03, 0.06, 0.04)]
        for idx, label in enumerate(["REQ", "RSP", "DATA"]):
            ax_btn = self.fig.add_axes(btn_positions[idx])
            btn = Button(ax_btn, label)
            btn.on_clicked(lambda event, i=idx: self._on_select_network(i))
            self.buttons.append(btn)

        # -------- Clear‑Highlight button --------
        ax_clear = self.fig.add_axes([0.28, 0.03, 0.08, 0.04])
        self.clear_btn = Button(ax_clear, "Clear HL")
        self.clear_btn.on_clicked(self._on_clear_highlight)

        # -------- Hover tooltip (显示 packet_id) --------
        # (removed hover annotation and motion_notify_event)
        self._play_idx = None  # 暂停时正在浏览的 history 索引
        self._draw_static_elements()

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
        # 初始化时绘制高亮框
        raw_center = self._selected_node
        row0 = raw_center // cols
        nodes_center = [raw_center]
        if row0 % 2 == 0 and raw_center + cols in self.node_positions:
            nodes_center.append(raw_center + cols)
        elif row0 % 2 == 1 and raw_center - cols in self.node_positions:
            nodes_center.append(raw_center - cols)
        xs0 = [self.node_positions[n][0] for n in nodes_center]
        ys0 = [self.node_positions[n][1] for n in nodes_center]
        llx0, lly0 = min(xs0), min(ys0)
        w0 = max(xs0) - min(xs0) + 0.5
        h0 = max(ys0) - min(ys0) + 0.5
        self.click_box = Rectangle((llx0, lly0), w0, h0, facecolor="none", edgecolor="red", linewidth=1.2, linestyle="--")
        self.ax.add_patch(self.click_box)
        self.fig.canvas.draw_idle()

        # 播放控制参数
        self.pause_interval = 0.2  # 默认每帧暂停间隔(秒)
        self.should_stop = False  # 停止标志
        self.status_text = self.ax.text(
            -0.1, 1, f"Running...\nInterval: {self.pause_interval:.2f}", transform=self.ax.transAxes, fontsize=12, fontweight="bold", color="green", verticalalignment="top"
        )
        self.detail_text = self.fig.text(0.70, 0.02, "", transform=self.fig.transFigure, fontsize=10, ha="left", va="bottom", color="blue")
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
        # 记录原始点击节点
        raw_node = sel_node
        if (sel_node // self.cols) % 2 == 0:
            sel_node += self.cols
        # 记录当前选中节点
        self._selected_node = sel_node

        # 清除之前的高亮框
        if hasattr(self, "click_box"):
            try:
                self.click_box.remove()
            except Exception:
                pass
        row = raw_node // self.cols
        nodes_to_box = [raw_node]
        if row % 2 == 0:
            # 偶数行，高亮下一行
            if raw_node + self.cols in self.node_positions:
                nodes_to_box.append(raw_node + self.cols)
        else:
            # 奇数行，高亮上一行
            if raw_node - self.cols in self.node_positions:
                nodes_to_box.append(raw_node - self.cols)
        # 找到所有这些节点的坐标
        xs = [self.node_positions[n][0] for n in nodes_to_box]
        ys = [self.node_positions[n][1] for n in nodes_to_box]
        # 计算包围框左下角及宽高
        llx = min(xs)
        lly = min(ys)
        width = max(xs) - min(xs) + 0.5
        height = max(ys) - min(ys) + 0.5
        # 绘制包围框
        self.click_box = Rectangle((llx, lly), width, height, facecolor="none", edgecolor="red", linewidth=1.2, linestyle="--")
        self.ax.add_patch(self.click_box)

        # 清空右侧子图并绘制
        self.piece_ax.clear()
        self.piece_ax.axis("off")

        # 当前网络对应的历史缓冲
        current_history = self.histories[self.selected_network_index]
        # 若正处于历史回溯视图，使用快照里的队列状态
        if self.paused and self._play_idx is not None and len(current_history) > self._play_idx:
            _, _, meta = current_history[self._play_idx]
            fake_net = SimpleNamespace(
                IQ_channel_buffer=meta["IQ_channel_buffer"],
                EQ_channel_buffer=meta["EQ_channel_buffer"],
                inject_queues=meta["inject_queues"],
                eject_queues=meta["eject_queues"],
                ring_bridge=meta["ring_bridge"],
                config=self.networks[self.selected_network_index].config,
            )
            self.piece_vis.draw_piece_for_node(sel_node, fake_net)
        else:
            live_net = self.networks[self.selected_network_index] if self.networks is not None else self.network
            self.piece_vis.draw_piece_for_node(sel_node, live_net)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # flit 点击：显示/隐藏 flit id 并追踪该 packet_id
    # ------------------------------------------------------------------
    def _on_flit_click(self, event):
        if event.button != 1 or event.inaxes is not self.ax:
            return
        # 查找点击中的 flit
        for rect, (txt, pid, fid) in self.rect_info_map.items():
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
                self.detail_text.set_text(f"PID {pid}  FID {fid}" if self.tracked_pid else "")
                self.fig.canvas.draw_idle()
                break

    def _update_tracked_labels(self):
        highlight_on = self.tracked_pid is not None
        for rect, (txt, pid, fid) in self.rect_info_map.items():
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
        # self.fig.tight_layout(rect=[0, 0.1, 1, 1])

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
            rows, cols = self.network.config.NUM_ROW, self.network.config.NUM_COL
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

    def update(self, networks=None, cycle=None, expected_packet_id=0, use_highlight=False, skip_pause=False):
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
        # 接收并保存多网络列表
        if networks is not None:
            self.networks = networks
        # 记录当前高亮配置, 供回溯绘制保持一致
        self._use_highlight = use_highlight
        self._expected_pid = expected_packet_id
        # 若暂停且非跳过暂停调用，则仅保持 GUI 响应；不推进模拟
        if self.paused and not skip_pause:
            plt.pause(self.pause_interval)
            return self.ax.patches

        if self.should_stop:
            return False
        self.cycle = cycle

        # 记录所有网络的历史快照
        if cycle is not None and self.networks is not None:
            for i, net in enumerate(self.networks):
                # 构建快照
                snap = {(s, d): [(f.packet_id, f.flit_id) if f is not None else None for f in flits] for (s, d), flits in net.links.items()}
                meta = {
                    "network_name": net.name,
                    "use_highlight": use_highlight,
                    "expected_pid": expected_packet_id,
                    "IQ_channel_buffer": copy.deepcopy(net.IQ_channel_buffer),
                    "EQ_channel_buffer": copy.deepcopy(net.EQ_channel_buffer),
                    "inject_queues": copy.deepcopy(net.inject_queues),
                    "eject_queues": copy.deepcopy(net.eject_queues),
                    "ring_bridge": copy.deepcopy(getattr(net, "ring_bridge", {})),
                }
                self.histories[i].append((cycle, snap, meta))

        # 渲染当前选中网络的快照
        if self.networks is not None:
            current_net = self.networks[self.selected_network_index]
            render_snap = {(s, d): [(f.packet_id, f.flit_id) if f is not None else None for f in flits] for (s, d), flits in current_net.links.items()}
            self._render_snapshot(render_snap)
            # 若已有选中节点，实时更新右侧 Piece 视图
            if self._selected_node is not None:
                self._refresh_piece_view()
            self.ax.set_title(current_net.name)
        else:
            self._render_snapshot({(src, dest): [(f.packet_id, f.flit_id) if f is not None else None for f in flits] for (src, dest), flits in self.network.links.items()})
            # 若已有选中节点，实时更新右侧 Piece 视图
            if self._selected_node is not None:
                self._refresh_piece_view()
            self.ax.set_title(self.network.name)
        if cycle and self.cycle % 10 == 0:
            self._update_status_display()
        if not skip_pause:
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
        self.piece_vis.use_highlight = self._use_highlight
        self.piece_vis.highlight_pid = self.tracked_pid

        self.piece_ax.clear()
        self.piece_ax.axis("off")

        # 当前网络对应的历史缓冲
        current_history = self.histories[self.selected_network_index]

        # 回溯模式：用保存的快照队列
        if self.paused and self._play_idx is not None and len(current_history) > self._play_idx:
            _, _, meta = current_history[self._play_idx]
            fake_net = SimpleNamespace(
                IQ_channel_buffer=meta["IQ_channel_buffer"],
                EQ_channel_buffer=meta["EQ_channel_buffer"],
                inject_queues=meta["inject_queues"],
                eject_queues=meta["eject_queues"],
                ring_bridge=meta["ring_bridge"],
                config=self.networks[self.selected_network_index].config,
            )
            self.piece_vis.draw_piece_for_node(self._selected_node, fake_net)
        else:  # 实时
            live_net = self.networks[self.selected_network_index]
            self.piece_vis.draw_piece_for_node(self._selected_node, live_net)

        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        key = event.key

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
        elif key == "p":
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
                        self._use_highlight = meta.get("use_highlight", False)
                        self._expected_pid = meta.get("expected_pid", 0)
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
            # 同步高亮 / 标题
            self._use_highlight = meta.get("use_highlight", False)
            self._expected_pid = meta.get("expected_pid", 0)
            self.ax.set_title(meta.get("network_name", ""))
            self.status_text.set_text(f"Paused\ncycle {cyc} ({self._play_idx+1}/{len(current_history)})")
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
                    self.rect_info_map[rect] = (txt, pid, fid)
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
                    self.rect_info_map[rect] = (txt, pid, fid)
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

    def _on_select_network(self, idx):
        """切换显示网络索引 idx（0/1/2）"""
        self.selected_network_index = idx
        # 刷新显示（调用 update 渲染当前网络）
        if self.networks is not None:
            self.update(
                self.networks,
                cycle=None,
                expected_packet_id=self._expected_pid,
                use_highlight=self._use_highlight,
                skip_pause=True,
            )

    # ------------------------------------------------------------------
    # (removed hover annotation and motion event handler)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Clear‑Highlight button callback
    # ------------------------------------------------------------------
    def _on_clear_highlight(self, event):
        self.tracked_pid = None
        self._use_highlight = False
        self._expected_pid = 0
        self._update_tracked_labels()
        self._refresh_piece_view()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Receive packet_id selected in PieceVisualizer and toggle highlight
    # ------------------------------------------------------------------
    def _on_piece_highlight(self, pid, fid):
        if self.tracked_pid == pid:
            self.tracked_pid = None
            self._use_highlight = False
            self._expected_pid = 0
            self.detail_text.set_text("")
        else:
            self.tracked_pid = pid
            self._use_highlight = True
            self._expected_pid = pid
            self.detail_text.set_text(f"PID {pid}  FID {fid}")
        self._update_tracked_labels()
        self._refresh_piece_view()
        self.fig.canvas.draw_idle()
