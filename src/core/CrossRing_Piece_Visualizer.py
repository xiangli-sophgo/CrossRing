import matplotlib
matplotlib.use('Agg')  # 或者根据需要切换到支持 GUI 的后端
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
        # 提取深度
        self.IQ_depth = config.IQ_OUT_FIFO_DEPTH
        self.EQ_depth = config.EQ_IN_FIFO_DEPTH
        self.RB_in_depth = config.RB_IN_FIFO_DEPTH
        self.RB_out_depth = config.RB_OUT_FIFO_DEPTH
        # 固定几何参数
        self.square = 0.1  # flit 方块边长
        self.gap = 0.02    # 相邻槽之间间距
        # 初始化图形
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.axis('off')
        self.ax.set_aspect('equal')
        # 调色板
        self._colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
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
        inj_x = center_x - 0.5
        ej_x  = center_x + 0.5
        rb_x  = center_x
        # Inject Queue
        self._draw_fifo_module(
            x=inj_x, y=center_y,
            title='Inject Queue',
            lanes=['left','right','up','local'],
            depths=self.IQ_depth,
            patch_dict=self.iq_patches,
            text_dict=self.iq_texts,
            per_lane_depth=False
        )
        # Eject Queue
        self._draw_fifo_module(
            x=ej_x, y=center_y,
            title='Eject Queue',
            lanes=['up','down','rb','local'],
            depths=self.EQ_depth,
            patch_dict=self.eq_patches,
            text_dict=self.eq_texts,
            per_lane_depth=False
        )
        # Ring Bridge（入 3 条，出 3 条）
        lanes_rb   = ['in_left','in_right','in_up','out_up','out_down','out_ej']
        depths_rb  = [self.RB_in_depth]*3 + [self.RB_out_depth]*3
        self._draw_fifo_module(
            x=rb_x, y=center_y - 0.8,
            title='Ring Bridge',
            lanes=lanes_rb,
            depths=depths_rb,
            patch_dict=self.rb_patches,
            text_dict=self.rb_texts,
            per_lane_depth=True
        )
        self.ax.relim()
        self.ax.autoscale_view()

    def _draw_fifo_module(self, x, y, title, lanes, depths, patch_dict, text_dict, per_lane_depth=False):
        """
        绘制一个模块及其 FIFO 槽。
        - x, y: 模块中心坐标
        - title: 模块名称
        - lanes: 列表，表示每条 FIFO 的键名
        - depths: 单个深度或列表，per_lane_depth 控制
        - patch_dict, text_dict: 存放 patch/text 对象的字典
        - per_lane_depth: 如果 True，则 depths 必须是与 lanes 等长的列表
        """
        square = self.square
        gap    = self.gap
        # 模块边框
        width  = 0.8
        height = 0.2 * len(lanes)
        box = Rectangle((x-width/2, y-height/2), width, height, fill=False)
        self.ax.add_patch(box)
        self.ax.text(x, y + height/2 + 0.02, title, ha='center', va='bottom', fontweight='bold')
        # 清空旧数据
        patch_dict.clear(); text_dict.clear()
        # 绘制每条 lane
        for i, lane in enumerate(lanes):
            laney = y + height/2 - (i * 0.2 + 0.1)
            self.ax.text(x-width/2-0.02, laney, lane, ha='right', va='center', fontsize=8)
            depth = depths[i] if per_lane_depth else depths
            patch_dict[lane] = []
            text_dict[lane]  = []
            for s in range(depth):
                sx = x - width/2 + 0.02 + s*(square+gap) + square/2
                patch = Rectangle((sx-square/2, laney-square/2), square, square,
                                  edgecolor='black', facecolor='none')
                self.ax.add_patch(patch)
                txt = self.ax.text(sx, laney+square/2+0.005, '', ha='center', va='bottom', fontsize=6)
                patch_dict[lane].append(patch)
                text_dict[lane].append(txt)

    def _get_color(self, pid):
        if pid in self._color_map:
            return self._color_map[pid]
        c = self._colors[self._next_color % len(self._colors)]
        self._color_map[pid] = c
        self._next_color += 1
        return c

    def update_display(self, state):
        """
        更新当前节点的 FIFO 状态。
        state: { 'inject': {...}, 'eject': {...}, 'ring_bridge': {...} }
        """
        inj = state.inject_queues
        ej  = state.eject_queues
        rb  = state.ring_bridge
        # Inject
        for lane, patches in self.iq_patches.items():
            q = inj.get(lane, [])
            for idx, p in enumerate(patches):
                t = self.iq_texts[lane][idx]
                if idx < len(q):
                    pid = getattr(q[idx], 'packet_id', str(q[idx]))
                    p.set_facecolor(self._get_color(pid))
                    t.set_text(str(pid))
                else:
                    p.set_facecolor('none'); t.set_text('')
        # Eject
        for lane, patches in self.eq_patches.items():
            q = ej.get(lane, [])
            for idx, p in enumerate(patches):
                t = self.eq_texts[lane][idx]
                if idx < len(q):
                    pid = getattr(q[idx], 'packet_id', str(q[idx]))
                    p.set_facecolor(self._get_color(pid))
                    t.set_text(str(pid))
                else:
                    p.set_facecolor('none'); t.set_text('')
        # Ring Bridge
        for lane, patches in self.rb_patches.items():
            q = rb.get(lane, [])
            for idx, p in enumerate(patches):
                t = self.rb_texts[lane][idx]
                if idx < len(q):
                    pid = getattr(q[idx], 'packet_id', str(q[idx]))
                    p.set_facecolor(self._get_color(pid))
                    t.set_text(str(pid))
                else:
                    p.set_facecolor('none'); t.set_text('')
        self.fig.canvas.draw_idle()
