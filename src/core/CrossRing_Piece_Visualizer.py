import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict
from matplotlib.patches import FancyArrowPatch


class CrossRingVisualizer:
    def __init__(self, config, ax):
        """
        仅绘制单个节点的 Inject/Eject Queue 和 Ring Bridge FIFO。
        参数:
        - config: 含有 FIFO 深度配置的对象，属性包括 cols, num_nodes, IQ_OUT_FIFO_DEPTH,
          EQ_IN_FIFO_DEPTH, RB_IN_FIFO_DEPTH, RB_OUT_FIFO_DEPTH
        - node_id: 要可视化的节点索引 (0 到 num_nodes-1)
        """
        self.config = config
        self.cols = config.cols
        self.rows = config.rows
        # 提取深度
        self.IQ_depth = config.IQ_OUT_FIFO_DEPTH
        self.EQ_depth = config.EQ_IN_FIFO_DEPTH
        self.RB_in_depth = config.RB_IN_FIFO_DEPTH
        self.RB_out_depth = config.RB_OUT_FIFO_DEPTH
        self.seats_per_link = config.seats_per_link
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
        ch_names = self.config.channel_names
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
                    t.set_visible(False)  # 隐藏文字
                    self.patch_info_map[p] = (t, info)
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
                    t.set_visible(False)  # 隐藏文字
                    self.patch_info_map[p] = (t, info)
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
                    t.set_visible(False)  # 隐藏文字
                    self.patch_info_map[p] = (t, info)
                else:
                    p.set_facecolor("none")
                    t.set_visible(False)
                    if p in self.patch_info_map:
                        self.patch_info_map.pop(p, None)

        plt.title(f"Node: {self.node_id}", fontsize=12)
        # plt.pause(0.2)

    # ------------------------------------------------------------------ #
    #  点击 flit 矩形时显示 / 隐藏文字                                      #
    # ------------------------------------------------------------------ #
    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        for patch, (txt, info) in self.patch_info_map.items():
            contains, _ = patch.contains(event)
            if contains:
                # 切换可见性
                vis = not txt.get_visible()
                txt.set_visible(vis)
                # 若即将显示，确保在最上层
                if vis:
                    txt.set_zorder(patch.get_zorder() + 1)
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


# --------------------------------------------------------------------------- #
#  Simple debug / demo                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    """
    Quick manual test:
    $ python CrossRing_Piece_Visualizer.py
    A matplotlib window pops up showing one node's three modules with random flits.
    """
    import matplotlib
    import numpy as np
    import sys

    if sys.platform == "darwin":  # macOS 的系统标识是 'darwin'
        matplotlib.use("macosx")  # 仅在 macOS 上使用该后端
    import random

    class _DemoFlit:
        def __init__(self, pid, fid):
            self.packet_id = pid
            self.flit_id = fid

    class _DemoConfig:
        cols = 0
        rows = 4
        IQ_OUT_FIFO_DEPTH = 6
        EQ_IN_FIFO_DEPTH = 5
        RB_IN_FIFO_DEPTH = 4
        RB_OUT_FIFO_DEPTH = 4
        seats_per_link = 4
        IQ_CH_FIFO_DEPTH = 5
        EQ_CH_FIFO_DEPTH = 5
        channel_names = ["gdma_0", "ddr_0"]

    class _DemoNet:
        def __init__(self):
            self.inject_queues = defaultdict(lambda: defaultdict(list))
            self.eject_queues = defaultdict(lambda: defaultdict(list))
            self.ring_bridge = defaultdict(lambda: defaultdict(list))
            self.IQ_channel_buffer = defaultdict(lambda: defaultdict(list))
            self.EQ_channel_buffer = defaultdict(lambda: defaultdict(list))

            # populate with random flits
            for lane in ["TL", "TR", "TU", "TD", "EQ", "IQ", "RB"]:
                for idx in range(4):
                    if random.random() > 0.5:
                        self.inject_queues[lane][0].append(_DemoFlit(random.randint(0, 9), idx))
                        self.eject_queues[lane][0].append(_DemoFlit(random.randint(0, 9), idx))
                        self.ring_bridge[lane][(0, 0)].append(_DemoFlit(random.randint(0, 9), idx))
            for ch in ["gdma_0", "ddr_0"]:
                if random.random() > 0.3:
                    self.inject_queues[ch][0].append(_DemoFlit(random.randint(0, 9), 0))
                    self.eject_queues[ch][0].append(_DemoFlit(random.randint(0, 9), 0))
                    self.ring_bridge[ch][(0, 0)].append(_DemoFlit(random.randint(0, 9), 0))

            # Ensure IQ_channel_buffer / EQ_channel_buffer have lists for each node to avoid IndexError
            all_lanes = ["gdma_0", "ddr_0", "TL", "TR", "TU", "TD"]
            for lane in all_lanes:
                self.IQ_channel_buffer[lane] = [[]]  # one node (index 0) with empty fifo
                self.EQ_channel_buffer[lane] = [[]]

    cfg = _DemoConfig()
    net = _DemoNet()

    fig, ax = plt.subplots(figsize=(10, 8))
    viz = CrossRingVisualizer(cfg, ax=ax)
    viz.draw_piece_for_node(0, net)
    plt.show()
