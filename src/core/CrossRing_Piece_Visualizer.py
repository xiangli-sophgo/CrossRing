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
        # 提取深度
        self.IQ_depth = config.IQ_OUT_FIFO_DEPTH
        self.EQ_depth = config.EQ_IN_FIFO_DEPTH
        self.RB_in_depth = config.RB_IN_FIFO_DEPTH
        self.RB_out_depth = config.RB_OUT_FIFO_DEPTH
        # 固定几何参数
        self.square = 0.08  # flit 方块边长
        self.gap = 0.02  # 相邻槽之间间距
        # 初始化图形
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
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
        # 画出三个模块的框和 FIFO 槽
        self._draw_modules()

    def _draw_modules(self):
        # 仅绘制当前节点的 Inject Queue, Eject Queue, Ring Bridge
        center_x, center_y = 0, 0
        IQ_x = center_x - 1.2
        IQ_y = center_y
        EQ_x = center_x
        EQ_y = center_y + 1.5
        RB_x = center_x
        RB_y = center_y
        # Inject Queue
        self._draw_fifo_module(
            x=IQ_x,
            y=IQ_y,
            title="Inject Queue",
            lanes=["left", "right", "up", "local"],
            # lanes=["TL", "TR", "RB", "EQ"],
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
            # lanes=["TU", "TD", "RB", "IQ"],
            depths=self.EQ_depth,
            patch_dict=self.eq_patches,
            text_dict=self.eq_texts,
            per_lane_depth=False,
            orientations=["vertical"] * 2 + ["horizontal"] * 2,
        )

        # Ring Bridge（入 3 条，出 3 条）
        lanes_rb = ["left", "right", "up", "vup", "vdown", "eject"]
        # ["TL", "TR", "IQ", "TU", "TD", "EQ"]
        depths_rb = [self.RB_in_depth] * 3 + [self.RB_out_depth] * 3
        self._draw_fifo_module(
            x=RB_x,
            y=RB_y,
            title="Ring Bridge",
            lanes=lanes_rb,
            depths=depths_rb,
            patch_dict=self.rb_patches,
            text_dict=self.rb_texts,
            per_lane_depth=True,
            orientations=["vertical"] * 3 + ["horizontal"] * 3,
        )
        self.ax.relim()
        self.ax.autoscale_view()

    def calculate_module_size(self, h_lanes, v_lanes):
        # 基础尺寸参数
        h_fifo_height = 0.3  # 每个横向FIFO的高度
        h_fifo_width = 0.2  # 每个横向FIFO的基础宽度影响
        v_fifo_height = 0.3  # 每个纵向FIFO的高度
        v_fifo_width = 0.25  # 每个纵向FIFO的宽度

        # 最小尺寸限制
        min_width = 0.6
        min_height = 0.6

        # 计算宽度（条件判断）
        if h_lanes and not v_lanes:  # 只有横向
            module_width = max(h_fifo_width * len(h_lanes), min_width)
        elif v_lanes and not h_lanes:  # 只有纵向
            module_width = max(v_fifo_width * len(v_lanes), min_width)
        else:  # 两者都有或都无（都无时结果为min_width）
            module_width = max(
                h_fifo_width * len(h_lanes) + v_fifo_width * len(v_lanes),
                min_width
            )
        
        # 计算高度（类似逻辑）
        if h_lanes and not v_lanes:
            module_height = max(h_fifo_height * len(h_lanes), min_height)
        elif v_lanes and not h_lanes:
            module_height = max(v_fifo_height * len(v_lanes), min_height)
        else:
            module_height = max(
                h_fifo_height * len(h_lanes) + (v_fifo_height * len(v_lanes) if v_lanes else 0),
                min_height
            )
        

        return module_width, module_height

    def _draw_fifo_module(self, x, y, title, lanes, depths, patch_dict, text_dict, per_lane_depth=False, orientations=None):
        """
        绘制一个模块及其 FIFO 槽，支持横向 FIFO 在上部、纵向 FIFO 在下部的混合布局

        参数：
        - x, y: 模块中心坐标
        - title: 模块名称
        - lanes: 列表，表示每条 FIFO 的键名
        - depths: 单个深度或列表，per_lane_depth 控制
        - patch_dict, text_dict: 存放 patch/text 对象的字典
        - per_lane_depth: 如果 True，则 depths 必须是与 lanes 等长的列表
        - orientations: None (全部相同方向) 或列表，每个元素为 'horizontal'/'vertical'
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

        # 计算模块尺寸
        module_width, module_height = self.calculate_module_size(h_lanes, v_lanes)
        # h_section_height = 0.3 * len(h_lanes) if h_lanes else 0
        # v_section_height = 0.9 if v_lanes else 0
        # module_height = h_section_height + v_section_height
        # module_width = max(0.8, 0.25 * len(v_lanes))  # 确保足够宽度

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

        # 绘制横向 FIFO (上部)
        for i, (lane, depth) in enumerate(zip(h_lanes, h_depths)):
            lane_x = x - module_width / 2 - 0.02
            lane_y = y + module_height / 2 - (i * 0.2 + 0.1)
            self.ax.text(lane_x, lane_y, lane, ha="right", va="center", fontsize=8)

            patch_dict[lane] = []
            text_dict[lane] = []

            for s in range(depth):
                slot_x = x - module_width / 2 + 0.02 + s * (square + gap) + square / 2
                slot_y = lane_y
                patch = Rectangle((slot_x - square / 2, slot_y - square / 2), square, square, edgecolor="black", facecolor="none")
                self.ax.add_patch(patch)
                txt = self.ax.text(slot_x, slot_y + square / 2 + 0.005, "", ha="center", va="bottom", fontsize=6)
                patch_dict[lane].append(patch)
                text_dict[lane].append(txt)

        # 绘制纵向 FIFO (下部)
        for i, (lane, depth) in enumerate(zip(v_lanes, v_depths)):
            lane_x = x - module_width / 2 + (i * 0.2 + 0.1)
            lane_y = y - module_height / 2 + 0.05  # 稍微抬高一点
            self.ax.text(lane_x, lane_y, lane, ha="center", va="bottom", fontsize=8)

            patch_dict[lane] = []
            text_dict[lane] = []

            for s in range(depth):
                slot_x = lane_x
                slot_y = y - module_height / 2 + 0.1 + s * (square + gap) + square / 2
                patch = Rectangle((slot_x - square / 2, slot_y - square / 2), square, square, edgecolor="black", facecolor="none")
                self.ax.add_patch(patch)
                txt = self.ax.text(slot_x - square / 2 - 0.005, slot_y, "", ha="right", va="center", fontsize=6)
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

    def update_display(self, state):
        """
        更新当前节点的 FIFO 状态。
        state: { 'inject': {...}, 'eject': {...}, 'ring_bridge': {...} }
        """
        IQ = state.inject_queues
        EQ = state.eject_queues
        RB = state.ring_bridge
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
                    if packet_id is not None:
                        t.set_text(f"{packet_id}/{flit_id}")  # 显示格式: packet_id/flit_id
                    else:
                        t.set_text(str(flit_id))  # 只有flit_id时直接显示
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
                    if packet_id is not None:
                        t.set_text(f"{packet_id}/{flit_id}")  # 显示格式: packet_id/flit_id
                    else:
                        t.set_text(str(flit_id))  # 只有flit_id时直接显示
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
                    if packet_id is not None:
                        t.set_text(f"{packet_id}.{flit_id}")  # 显示格式: packet_id/flit_id
                    else:
                        t.set_text(str(flit_id))  # 只有flit_id时直接显示
                else:
                    p.set_facecolor("none")
                    t.set_text("")
        plt.pause(0.2)
