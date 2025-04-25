import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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
        self.name_map = {"left": "TL", "right": "TR", "up": "TU", "down": "TD", "vup": "TU", "vdown": "TD", "ring_bridge": "RB", "eject": "EQ", "local_I": "EQ", "local_E": "IQ"}
        # 存储 patch 和 text
        self.iq_patches, self.iq_texts = {}, {}
        self.eq_patches, self.eq_texts = {}, {}
        self.rb_patches, self.rb_texts = {}, {}
        self.lh_patches, self.cph_texts = {}, {}
        self.lv_patches, self.cpv_texts = {}, {}
        self.link_patches, self.link_texts = {}, {}  # 新增的link可视化存储
        # 画出三个模块的框和 FIFO 槽
        self._draw_modules()
        self._draw_links_module()  # 新增的links模块

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
                txt = self.ax.text(slot_x, slot_y + square / 2 + 0.005, "", ha="center", va="bottom", fontsize=8)
                self.link_patches[link_name].append(patch)
                self.link_texts[link_name].append(txt)

    def _draw_modules(self):
        # 仅绘制当前节点的 Inject Queue, Eject Queue, Ring Bridge
        center_x, center_y = 0, 0
        IQ_x = center_x - 3.5
        IQ_y = center_y
        EQ_x = center_x
        EQ_y = center_y + 3.5
        RB_x = center_x
        RB_y = center_y
        LH_x = center_x - 1.2
        LH_y = center_y - 3.5
        LV_x = center_x + 3.5
        LV_y = center_y + 1.2
        # Inject Queue
        self._draw_fifo_module(
            x=IQ_x,
            y=IQ_y,
            title="Inject Queue",
            lanes=["left", "right", "up", "local"],
            module_height=3.5,
            module_width=2.5,
            depths=self.IQ_depth,
            patch_dict=self.iq_patches,
            text_dict=self.iq_texts,
            per_lane_depth=False,
            orientations=["vertical"] * 2 + ["horizontal"] * 2,
        )

        # Eject Queue
        self._draw_fifo_module(
            x=EQ_x,
            y=EQ_y,
            title="Eject Queue",
            lanes=["up", "down", "ring_bridge", "local"],
            module_height=2.5,
            module_width=3.5,
            depths=self.EQ_depth,
            patch_dict=self.eq_patches,
            text_dict=self.eq_texts,
            per_lane_depth=False,
            orientations=["vertical"] * 2 + ["horizontal"] * 2,
        )

        # Ring Bridge（入 3 条，出 3 条）
        self._draw_fifo_module(
            x=RB_x,
            y=RB_y,
            title="Ring Bridge",
            lanes=["left", "right", "up", "vup", "vdown", "eject"],
            depths=[self.RB_in_depth] * 3 + [self.RB_out_depth] * 3,
            module_height=3.5,
            module_width=3.5,
            patch_dict=self.rb_patches,
            text_dict=self.rb_texts,
            per_lane_depth=True,
            orientations=["vertical"] * 3 + ["horizontal"] * 3,
        )

        LH_depths = [self.seats_per_link if self.col != 0 else 2, self.seats_per_link if self.col + 1 != self.cols else 2]
        LH_orientations = ["horizontal" if self.col != 0 else "vertical", "horizontal" if self.col + 1 != self.cols else "vertical"]
        self._draw_fifo_module(
            x=LH_x,
            y=LH_y,
            title="Link Horizontal",
            lanes=["left", "right"],
            depths=LH_depths,
            module_height=2,
            module_width=5,
            patch_dict=self.lh_patches,
            text_dict=self.cph_texts,
            per_lane_depth=True,
            orientations=LH_orientations,
            h_position="top",
            v_position="right",
        )

        LV_depths = [self.seats_per_link if self.row != 1 else 2, self.seats_per_link if self.row + 1 != self.rows else 2]
        LV_orientations = ["vertical" if self.row != 1 else "horizontal", "vertical" if self.row + 1 != self.rows else "horizontal"]
        self._draw_fifo_module(
            x=LV_x,
            y=LV_y,
            title="Link vertical",
            lanes=["up", "down"],
            depths=LV_depths,
            module_height=5,
            module_width=2,
            patch_dict=self.lv_patches,
            text_dict=self.cpv_texts,
            per_lane_depth=True,
            orientations=LV_orientations,
            h_position="top",
            v_position="left",
        )
        self.ax.relim()
        self.ax.autoscale_view()

    def _draw_fifo_module(self, x, y, title, module_height, module_width, lanes, depths, patch_dict, text_dict, per_lane_depth=False, orientations=None, h_position="top", v_position="left"):
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
            self.ax.text(lane_x, lane_y, self.name_map[lane] if lane != "local" else self.name_map[f"{lane}_{title[0]}"], ha="right", va="center", fontsize=10)

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
            self.ax.text(lane_x, lane_y, self.name_map[lane] if lane != "local" else self.name_map[f"{lane}_{title[0]}"], ha="center", va="bottom", fontsize=10)

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

        # 更新所有links的显示
        self._update_links_display(links)
        plt.pause(0.2)

    def _update_links_display(self, links):
        """更新所有links的显示"""
        # 遍历所有links
        link_idx = 0
        for src_dest, q in links.items():
            src, dest = src_dest
            link_name = f"Link_{link_idx}"
            link_idx += 1

            if link_name not in self.link_patches:
                continue

            patches = self.link_patches[link_name]
            texts = self.link_texts[link_name]

            # 更新显示
            for idx, p in enumerate(patches):
                t = texts[idx]
                if idx < len(q):
                    item = q[idx]
                    if item is None:
                        p.set_facecolor("none")
                        t.set_text("")
                        continue

                    packet_id = getattr(item, "packet_id", None)
                    flit_id = getattr(item, "flit_id", str(item))
                    pid = {"packet_id": packet_id, "flit_id": flit_id}

                    p.set_facecolor(self._get_color(pid))
                    t.set_text(f"{packet_id}-{flit_id}")
                else:
                    p.set_facecolor("none")
                    t.set_text("")
