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
        self.square = 0.17  # flit 方块边长
        self.gap = 0.02  # 相邻槽之间间距
        self.fifo_gap = 0.8  # 相邻fifo之间间隙
        height = 5
        weight = 3
        self.inject_module_size = (height, weight)
        self.eject_module_size = (weight, height)
        self.rb_module_size = (height, height)
        # 初始化图形
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 6))  # 增大图形尺寸
        else:
            self.ax = ax
        self.ax.axis("off")
        self.ax.set_aspect("equal")
        # 调色板
        self._colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self._color_map = {}
        self._next_color = 0

        # 存储 patch 和 text
        self.iq_patches, self.iq_texts = {}, {}
        self.eq_patches, self.eq_texts = {}, {}
        self.rb_patches, self.rb_texts = {}, {}
        self.lh_patches, self.cph_texts = {}, {}
        self.lv_patches, self.cpv_texts = {}, {}
        # 画出三个模块的框和 FIFO 槽
        self._draw_modules()
        # self._draw_arrows()

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
        # 仅绘制当前节点的 Inject Queue, Eject Queue, Ring Bridge
        center_x, center_y = 0, 0
        IQ_x = center_x - self.inject_module_size[0] * 1.2
        IQ_y = center_y
        EQ_x = center_x
        EQ_y = center_y + self.inject_module_size[1] * 1.5
        RB_x = center_x
        RB_y = center_y

        # Inject Queue
        self._draw_fifo_module(
            x=IQ_x,
            y=IQ_y,
            title="Inject Queue",
            lanes=self.config.channel_names + ["TL", "TR", "EQ", "TU", "TD"],
            module_height=self.inject_module_size[0],
            module_width=self.inject_module_size[1],
            depths=[self.IQ_CH_depth] * len(self.config.channel_names) + [self.IQ_depth] * 5,
            patch_dict=self.iq_patches,
            text_dict=self.iq_texts,
            per_lane_depth=True,
            orientations=["vertical"] * len(self.config.channel_names) + ["vertical"] * 2 + ["horizontal"] * 3,
            h_position=["top"] * len(self.config.channel_names) + ["bottom"] * 2 + ["mid"] * 3,
            v_position=["left"] * len(self.config.channel_names) + ["left"] * 2 + ["right"] * 3,
        )

        # Eject Queue
        self._draw_fifo_module(
            x=EQ_x,
            y=EQ_y,
            title="Eject Queue",
            lanes=["TU", "TD"],
            module_height=self.eject_module_size[0],
            module_width=self.eject_module_size[1],
            depths=self.EQ_depth,
            patch_dict=self.eq_patches,
            text_dict=self.eq_texts,
            per_lane_depth=False,
            orientations=["horizontal"] * 2,
            h_position=["top"] * 2,
            v_position=["right"] * 2,
        )

        # Ring Bridge（入 3 条，出 3 条）
        self._draw_fifo_module(
            x=RB_x,
            y=RB_y,
            title="Ring Bridge",
            lanes=["TL", "TR", "TU", "TD"],
            depths=[self.RB_in_depth] * 2 + [self.RB_out_depth] * 2,
            module_height=self.rb_module_size[0],
            module_width=self.rb_module_size[1],
            patch_dict=self.rb_patches,
            text_dict=self.rb_texts,
            per_lane_depth=True,
            orientations=["vertical"] * 2 + ["horizontal"] * 2,
            h_position=["bottom"] * 2 + ["top"] * 2,
            v_position=["left"] * 2 + ["right"] * 2,
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
        box = Rectangle((x - module_width / 2, y - module_height / 2), module_width, module_height, fill=False)
        self.ax.add_patch(box)

        # 模块标题
        title_x = x
        title_y = y + module_height / 2 + 0.02
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
                    lane_y = y + module_height / 2 - (idx_in_group * self.fifo_gap + 0.2)
                    text_va = "bottom"
                elif hpos == "bottom":
                    lane_y = y - module_height / 2 + (idx_in_group * self.fifo_gap + 0.2)
                    text_va = "top"
                elif hpos == "mid":
                    offset = (idx_in_group - (group_size - 1) / 2) * self.fifo_gap
                    lane_y = y + offset
                    text_va = "center"
                else:
                    raise ValueError(f"Unknown h_position: {hpos}")

                # 横坐标由 vpos 决定
                if vpos == "right":
                    lane_x = x + module_width - depth * (square + gap) * 1.8
                    text_x = x + module_width / 2 - depth * (square + gap) * 1.2
                    slot_dir = -1
                    ha = "right"
                elif vpos == "left":
                    lane_x = x - module_width / 2 + depth * (square + gap) * 1
                    text_x = x - module_width / 2 + depth * (square + gap) * 1.2
                    slot_dir = -1
                    ha = "left"
                elif vpos == "mid" or vpos is None:
                    lane_x = x + module_width - depth * (square + gap) * 1.2
                    text_x = x + module_width / 2 - depth * (square + gap) * 0.8
                    slot_dir = -1
                    ha = "left"
                else:
                    raise ValueError(f"Unknown v_position: {vpos}")

                self.ax.text(text_x, lane_y, lane, ha=ha, va="center", fontsize=10)
                patch_dict[lane] = []
                text_dict[lane] = []

                for s in range(depth):
                    slot_x = lane_x + slot_dir * s * (square + gap)
                    slot_y = lane_y
                    patch = Rectangle((slot_x - square / 2, slot_y - square / 2), square, square, edgecolor="black", facecolor="none")
                    self.ax.add_patch(patch)
                    txt = self.ax.text(slot_x, slot_y + (square / 2 + 0.005 if hpos == "top" else -square / 2 - 0.005), "", ha="center", va=text_va, fontsize=10)
                    patch_dict[lane].append(patch)
                    text_dict[lane].append(txt)

            elif orient == "vertical":
                # 横坐标由 vpos 决定
                if vpos == "left":
                    lane_x = x - module_width / 2 + (idx_in_group * self.fifo_gap + 0.2)
                    text_ha = "right"
                elif vpos == "right":
                    lane_x = x + module_width / 2 - (idx_in_group * self.fifo_gap + 0.2)
                    text_ha = "left"
                elif vpos == "mid" or vpos is None:
                    offset = (idx_in_group - (group_size - 1) / 2) * self.fifo_gap
                    lane_x = x + offset
                    text_ha = "center"
                else:
                    raise ValueError(f"Unknown v_position: {vpos}")

                # 纵坐标由 hpos 决定
                if hpos == "top":
                    lane_y = y + module_height / 2 - 0.02 - depth * (square + gap) - square - 0.02
                    slot_dir = -1
                    va = "top"
                elif hpos == "bottom":
                    lane_y = y - module_height / 2 + 0.02 + square
                    slot_dir = 1
                    va = "bottom"
                elif hpos == "mid" or hpos is None:
                    lane_y = y - (depth / 2) * (square + gap)
                    slot_dir = 1
                    va = "center"
                else:
                    raise ValueError(f"Unknown h_position: {hpos}")

                self.ax.text(lane_x, lane_y, lane, ha="center", va=va, fontsize=10)
                patch_dict[lane] = []
                text_dict[lane] = []

                for s in range(depth):
                    slot_x = lane_x
                    slot_y = lane_y + slot_dir * s * (square + gap)
                    patch = Rectangle((slot_x - square / 2, slot_y - square / 2), square, square, edgecolor="black", facecolor="none")
                    self.ax.add_patch(patch)
                    txt = self.ax.text(slot_x + (square / 2 + 0.005 if vpos == "right" else -square / 2 - 0.005), slot_y, "", ha=text_ha, va="center", fontsize=10)
                    patch_dict[lane].append(patch)
                    text_dict[lane].append(txt)

            else:
                raise ValueError(f"Unknown orientation: {orient}")

    def _get_color(self, flit):
        """获取颜色，支持多种PID格式：
        - 单个值 (packet_id 或 flit_id)
        - 元组 (packet_id, flit_id)
        - 字典 {'packet_id': x, 'flit_id': y}
        """
        # 统一提取 packet_id 作为颜色依据
        pid = {"packet_id": flit.packet_id, "flit_id": flit.flit_id}
        if isinstance(pid, tuple) and len(pid) >= 1:
            color_key = pid[0]  # 元组第一个元素作为 packet_id
        elif isinstance(pid, dict):
            color_key = pid.get("packet_id", str(pid))
        else:
            color_key = pid

        # if color_key in self._color_map:
        #     return self._color_map[color_key]

        c = self._colors[color_key % len(self._colors)]
        # c = self._colors[self._next_color % len(self._colors)]
        self._color_map[color_key] = c
        # self._next_color += 1
        return c

    def draw_piece_for_node(self, node_id, network):
        """
        更新当前节点的 FIFO 状态。
        state: { 'inject': {...}, 'eject': {...}, 'ring_bridge': {...} }
        """
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
        # Inject
        for lane, patches in self.iq_patches.items():
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
                    t.set_text(f"{packet_id}-{flit_id}")  # 显示格式: packet_id/flit_id
                else:
                    p.set_facecolor("none")
                    t.set_text("")
        # Eject
        for lane, patches in self.eq_patches.items():
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
                    t.set_text(f"{packet_id}-{flit_id}")  # 显示格式: packet_id/flit_id
                else:
                    p.set_facecolor("none")
                    t.set_text("")
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
                    t.set_text(f"{packet_id}-{flit_id}")  # 显示格式: packet_id/flit_id
                else:
                    p.set_facecolor("none")
                    t.set_text("")

        plt.title(f"Node: {self.node_id}", fontsize=12)
        # plt.pause(0.2)
