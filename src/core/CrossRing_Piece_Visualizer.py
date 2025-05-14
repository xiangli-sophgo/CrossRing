import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from matplotlib.patches import FancyArrowPatch


class CrossRingVisualizer:
    def __init__(self, config, node_id):
        """
        仅绘制单个节点的 Inject/Eject Queue 和 Ring Bridge FIFO。
        参数:
        - config: 含有 FIFO 深度配置的对象，属性包括 cols, num_nodes, IQ_OUT_FIFO_DEPTH,
          EQ_IN_FIFO_DEPTH, RB_IN_FIFO_DEPTH, RB_OUT_FIFO_DEPTH
        - node_id: 要可视化的节点索引 (0 到 num_nodes-1)
        """
        self.cfg = config
        self.node_id = node_id
        # 计算该节点的坐标 (暂不用于绘制位置)
        self.row = node_id // config.cols
        self.col = node_id % config.cols
        self.cols = config.cols
        self.rows = config.rows
        # 提取深度
        self.IQ_depth = config.IQ_OUT_FIFO_DEPTH
        self.EQ_depth = config.EQ_IN_FIFO_DEPTH
        self.RB_in_depth = config.RB_IN_FIFO_DEPTH
        self.RB_out_depth = config.RB_OUT_FIFO_DEPTH
        self.seats_per_link = config.seats_per_link
        # 固定几何参数
        self.square = 0.17  # flit 方块边长
        self.gap = 0.02  # 相邻槽之间间距
        self.fifo_gap = 0.5  # 相邻fifo之间间隙
        # 初始化图形
        self.fig, self.ax = plt.subplots(figsize=(12, 10))  # 增大图形尺寸
        plt.subplots_adjust(bottom=0.2)  # 为底部links模块留出空间
        self.ax.axis("off")
        self.ax.set_aspect("equal")
        # 调色板
        self._colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self._color_map = {}
        self._next_color = 0
        self.name_map = {
            "left": "TL",
            "right": "TR",
            "up_R": "IQ",
            "up_E": "TU",
            "up_I": "RB",
            "down": "TD",
            "vup": "TU",
            "vdown": "TD",
            "ring_bridge": "RB",
            "eject": "EQ",
            "local_I": "EQ",
            "local_E": "IQ",
        }
        # 存储 patch 和 text
        self.iq_patches, self.iq_texts = {}, {}
        self.eq_patches, self.eq_texts = {}, {}
        self.rb_patches, self.rb_texts = {}, {}
        self.lh_patches, self.cph_texts = {}, {}
        self.lv_patches, self.cpv_texts = {}, {}
        self.link_patches, self.link_texts = {}, {}  # 新增的link可视化存储
        # 画出三个模块的框和 FIFO 槽
        self._draw_modules()
        # self._draw_arrows()

    def _draw_links_module(self):
        """绘制所有links的模块"""
        square = self.square
        gap = self.gap

        # 确定模块位置
        links_x = -4
        links_y = -1  # 放在底部

        # 模块尺寸
        module_width = 10
        module_height = 3

        # 绘制模块边框
        box = Rectangle((links_x - module_width / 2, links_y - module_height / 2), module_width, module_height, fill=False)
        self.ax.add_patch(box)

        # 模块标题
        title_x = links_x
        title_y = links_y + module_height / 2 + 0.02
        self.ax.text(title_x, title_y, "All Links", ha="center", va="bottom", fontweight="bold")

        # 计算每个link的位置
        num_links = self.cols * self.rows * 4  # 假设每个节点有4个方向的link
        link_rows = 3
        link_cols = (num_links + link_rows - 1) // link_rows

        # 清空旧数据
        self.link_patches.clear()
        self.link_texts.clear()

        # 为每个link绘制FIFO槽
        for i in range(num_links):
            row = i // link_cols
            col = i % link_cols

            # 计算位置
            x = links_x - module_width / 2 + 0.5 + col * 1.5
            y = links_y + module_height / 2 - 0.5 - row * 0.7

            # 创建link标识
            link_name = f"Link_{i}"
            self.ax.text(x - 0.5, y, link_name, ha="right", va="center", fontsize=8)

            # 绘制FIFO槽
            self.link_patches[link_name] = []
            self.link_texts[link_name] = []

            for s in range(self.seats_per_link):
                slot_x = x + s * (square + gap)
                slot_y = y
                patch = Rectangle((slot_x - square / 2, slot_y - square / 2), square, square, edgecolor="black", facecolor="none")
                self.ax.add_patch(patch)
                txt = self.ax.text(slot_x, slot_y + square / 2 + 0.005, "", ha="center", va="bottom", size=8)
                self.link_patches[link_name].append(patch)
                self.link_texts[link_name].append(txt)

    def _draw_arrows(self):
        # 1. 模块几何信息（必须与 _draw_modules 中的保持一致）
        IQ_x, IQ_y, IQ_w, IQ_h = -3.5, 0.0, 2.5, 3.5
        EQ_x, EQ_y, EQ_w, EQ_h = 0.0, 3.5, 3.5, 2.5
        RB_x, RB_y, RB_w, RB_h = 0.0, 0.0, 3.5, 3.5

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
        IQ_x = center_x - 3.5
        IQ_y = center_y
        EQ_x = center_x
        EQ_y = center_y + 3.5
        RB_x = center_x
        RB_y = center_y

        # Inject Queue
        self._draw_fifo_module(
            x=IQ_x,
            y=IQ_y,
            title="Inject Queue",
            lanes=["TL", "TR", "EQ", "TU", "TD"],
            module_height=3.5,
            module_width=2.5,
            depths=self.IQ_depth,
            patch_dict=self.iq_patches,
            text_dict=self.iq_texts,
            per_lane_depth=False,
            orientations=["vertical"] * 2 + ["horizontal"] * 3,
        )

        # Eject Queue
        self._draw_fifo_module(
            x=EQ_x,
            y=EQ_y,
            title="Eject Queue",
            lanes=["TU", "TD"],
            module_height=2.5,
            module_width=3.5,
            depths=self.EQ_depth,
            patch_dict=self.eq_patches,
            text_dict=self.eq_texts,
            per_lane_depth=False,
            orientations=["horizontal"] * 2,
        )

        # Ring Bridge（入 3 条，出 3 条）
        self._draw_fifo_module(
            x=RB_x,
            y=RB_y,
            title="Ring Bridge",
            lanes=["TL", "TR", "TU", "TD"],
            depths=[self.RB_in_depth] * 3 + [self.RB_out_depth] * 3,
            module_height=3.5,
            module_width=3.5,
            patch_dict=self.rb_patches,
            text_dict=self.rb_texts,
            per_lane_depth=True,
            orientations=["vertical"] * 2 + ["horizontal"] * 2,
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
        绘制一个模块及其 FIFO 槽，支持横向和纵向 FIFO 的灵活布局

        参数：
        - x, y: 模块中心坐标
        - title: 模块名称
        - lanes: 列表，表示每条 FIFO 的键名
        - depths: 单个深度或列表，per_lane_depth 控制
        - patch_dict, text_dict: 存放 patch/text 对象的字典
        - per_lane_depth: 如果 True，则 depths 必须是与 lanes 等长的列表
        - orientations: None (全部相同方向) 或列表，每个元素为 'horizontal'/'vertical'
        - h_position: 横向 FIFO 的位置 ('top' 或 'bottom')
        - v_position: 纵向 FIFO 的位置 ('left' 或 'right')
        - title_position: 标题的位置 ('left-up/down' 或 'right-up/down')

        """
        square = self.square
        gap = self.gap

        # 处理方向参数
        if orientations is None:
            default_orientation = "horizontal"
            orientations = [default_orientation] * len(lanes)
        elif isinstance(orientations, str):
            orientations = [orientations] * len(lanes)

        # 分离横向和纵向的 FIFO
        h_lanes = [lane for lane, orient in zip(lanes, orientations) if orient == "horizontal"]
        v_lanes = [lane for lane, orient in zip(lanes, orientations) if orient == "vertical"]
        h_depths = [depths[i] if per_lane_depth else depths for i, orient in enumerate(orientations) if orient == "horizontal"]
        v_depths = [depths[i] if per_lane_depth else depths for i, orient in enumerate(orientations) if orient == "vertical"]

        # 绘制模块边框
        box = Rectangle((x - module_width / 2, y - module_height / 2), module_width, module_height, fill=False)
        self.ax.add_patch(box)

        # 模块标题
        title_x = x
        title_y = y + module_height / 2 + 0.02
        self.ax.text(title_x, title_y, title, ha="center", va="bottom", fontweight="bold")

        # 清空旧数据
        patch_dict.clear()
        text_dict.clear()

        # 绘制横向 FIFO
        for i, (lane, depth) in enumerate(zip(h_lanes, h_depths)):
            # 根据位置参数确定 y 坐标
            if h_position == "top":
                lane_y = y + module_height / 2 - (i * self.fifo_gap + 0.2)
                text_va = "bottom"
            else:  # bottom
                lane_y = y - module_height / 2 + (i * self.fifo_gap + 0.2)
                text_va = "top"

            lane_x = x + module_width / 2 - 0.02 - depth * (square + gap) - square - 0.02
            # self.ax.text(lane_x, lane_y, self.name_map[lane] if lane not in ["local", "up"] else self.name_map[f"{lane}_{title[0]}"], ha="right", va="center", fontsize=10)
            self.ax.text(lane_x, lane_y, lane, ha="right", va="center", fontsize=10)

            patch_dict[lane] = []
            text_dict[lane] = []

            for s in range(depth):
                slot_x = x + module_width / 2 - 0.02 - s * (square + gap) - square
                slot_y = lane_y
                patch = Rectangle((slot_x - square / 2, slot_y - square / 2), square, square, edgecolor="black", facecolor="none")
                self.ax.add_patch(patch)
                txt = self.ax.text(slot_x, slot_y + (square / 2 + 0.005 if h_position == "top" else -square / 2 - 0.005), "", ha="center", va=text_va, fontsize=10)
                patch_dict[lane].append(patch)
                text_dict[lane].append(txt)

        # 绘制纵向 FIFO
        for i, (lane, depth) in enumerate(zip(v_lanes, v_depths)):
            # 根据位置参数确定 x 坐标
            if v_position == "left":
                lane_x = x - module_width / 2 + (i * self.fifo_gap + 0.2)
                text_ha = "right"
            else:  # right
                lane_x = x + module_width / 2 - (i * self.fifo_gap + 0.2)
                text_ha = "left"

            lane_y = y - module_height / 2 + 0.1 + depth * (square + gap) + square / 2 + 0.05
            # self.ax.text(lane_x, lane_y, self.name_map[lane] if lane not in ["local", "up"] else self.name_map[f"{lane}_{title[0]}"], ha="center", va="bottom", fontsize=10)
            self.ax.text(lane_x, lane_y, lane, ha="center", va="bottom", fontsize=10)

            patch_dict[lane] = []
            text_dict[lane] = []

            for s in range(depth):
                slot_x = lane_x
                slot_y = y - module_height / 2 + 0.1 + s * (square + gap) + square / 2
                patch = Rectangle((slot_x - square / 2, slot_y - square / 2), square, square, edgecolor="black", facecolor="none")
                self.ax.add_patch(patch)
                txt = self.ax.text(slot_x + (square / 2 + 0.005 if v_position == "right" else -square / 2 - 0.005), slot_y, "", ha=text_ha, va="center", fontsize=10)
                patch_dict[lane].append(patch)
                text_dict[lane].append(txt)

    def _get_color(self, pid):
        """获取颜色，支持多种PID格式：
        - 单个值 (packet_id 或 flit_id)
        - 元组 (packet_id, flit_id)
        - 字典 {'packet_id': x, 'flit_id': y}
        """
        # 统一提取 packet_id 作为颜色依据
        if isinstance(pid, tuple) and len(pid) >= 1:
            color_key = pid[0]  # 元组第一个元素作为 packet_id
        elif isinstance(pid, dict):
            color_key = pid.get("packet_id", str(pid))
        else:
            color_key = pid

        if color_key in self._color_map:
            return self._color_map[color_key]

        c = self._colors[self._next_color % len(self._colors)]
        self._color_map[color_key] = c
        self._next_color += 1
        return c

    def update_display(self, network):
        """
        更新当前节点的 FIFO 状态。
        state: { 'inject': {...}, 'eject': {...}, 'ring_bridge': {...} }
        """
        IQ = network.inject_queues
        EQ = network.eject_queues
        RB = network.ring_bridge
        links = network.links
        # Inject
        for lane, patches in self.iq_patches.items():
            q = IQ.get(lane, [])[self.node_id]
            for idx, p in enumerate(patches):
                t = self.iq_texts[lane][idx]
                if idx < len(q):
                    item = q[idx]
                    packet_id = getattr(item, "packet_id", None)
                    flit_id = getattr(item, "flit_id", str(item))

                    # 创建复合ID对象
                    pid = {"packet_id": packet_id, "flit_id": flit_id}

                    # 设置颜色（基于packet_id）和显示文本
                    p.set_facecolor(self._get_color(pid))
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
                    item = q[idx]
                    packet_id = getattr(item, "packet_id", None)
                    flit_id = getattr(item, "flit_id", str(item))

                    # 创建复合ID对象
                    pid = {"packet_id": packet_id, "flit_id": flit_id}

                    # 设置颜色（基于packet_id）和显示文本
                    p.set_facecolor(self._get_color(pid))
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
                    item = q[idx]
                    packet_id = getattr(item, "packet_id", None)
                    flit_id = getattr(item, "flit_id", str(item))

                    # 创建复合ID对象
                    pid = {"packet_id": packet_id, "flit_id": flit_id}

                    # 设置颜色（基于packet_id）和显示文本
                    p.set_facecolor(self._get_color(pid))
                    t.set_text(f"{packet_id}-{flit_id}")  # 显示格式: packet_id/flit_id
                else:
                    p.set_facecolor("none")
                    t.set_text("")

        # Cross Ring Horizontal
        for lane, patches in self.lh_patches.items():
            if lane == "left":
                if self.node_id % self.cols == 0:
                    q = links.get((self.node_id, self.node_id), [])
                else:
                    q = links.get((self.node_id, self.node_id - 1), [])
            elif lane == "right":
                if self.node_id % self.cols == self.cols - 1:
                    q = links.get((self.node_id, self.node_id), [])
                else:
                    q = links.get((self.node_id, self.node_id + 1), [])
            for idx, p in enumerate(patches):
                t = self.cph_texts[lane][idx]
                if idx < len(q):
                    item = q[idx]
                    if item is None:
                        continue
                    packet_id = getattr(item, "packet_id", None)
                    flit_id = getattr(item, "flit_id", str(item))

                    # 创建复合ID对象
                    pid = {"packet_id": packet_id, "flit_id": flit_id}

                    # 设置颜色（基于packet_id）和显示文本
                    p.set_facecolor(self._get_color(pid))
                    t.set_text(f"{packet_id}-{flit_id}")  # 显示格式: packet_id/flit_id
                else:
                    p.set_facecolor("none")
                    t.set_text("")

        # Cross Ring Vertical
        for lane, patches in self.lv_patches.items():
            if lane == "up":
                if self.node_id // self.cols == 0:
                    q = links.get((self.node_id, self.node_id), [])
                else:
                    q = links.get((self.node_id, self.node_id - 2 * self.cols), [])
            elif lane == "down":
                if self.node_id // self.cols == self.rows:
                    q = links.get((self.node_id, self.node_id), [])
                else:
                    q = links.get((self.node_id, self.node_id + 2 * self.cols), [])
            for idx, p in enumerate(patches):
                t = self.cpv_texts[lane][idx]
                if idx < len(q):
                    item = q[idx]
                    if item is None:
                        continue
                    packet_id = getattr(item, "packet_id", None)
                    flit_id = getattr(item, "flit_id", str(item))

                    # 创建复合ID对象
                    pid = {"packet_id": packet_id, "flit_id": flit_id}

                    # 设置颜色（基于packet_id）和显示文本
                    p.set_facecolor(self._get_color(pid))
                    t.set_text(f"{packet_id}-{flit_id}")  # 显示格式: packet_id/flit_id
                else:
                    p.set_facecolor("none")
                    t.set_text("")
        plt.title(f"{network.name}, node: {self.node_id}", fontsize=12)
        plt.pause(0.2)
