import json
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.widgets import Button, Slider
from collections import defaultdict

#
# ------------------ optional Tk support ------------------ #
try:
    import tkinter as tk
    from tkinter import filedialog, simpledialog
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
except ImportError:
    # macOS 或环境缺少 Tcl/Tk 时，退化到纯 matplotlib
    tk = None
    filedialog = simpledialog = None
    FigureCanvasTkAgg = NavigationToolbar2Tk = None
import matplotlib.patches as patches
from matplotlib.figure import Figure
import sys
import matplotlib

# macOS 后端处理
if sys.platform == "darwin":
    matplotlib.use("macosx")  # only switch if Tk is available


class DraggableModule:
    """可拖拽的模块类，用于处理模块的拖拽事件"""

    def __init__(self, rect, title_text, module_name, on_position_changed=None):
        self.rect = rect
        self.title_text = title_text
        self.module_name = module_name
        self.press = None
        self.background = None
        self.on_position_changed = on_position_changed
        self.connect()

    def connect(self):
        """连接事件处理器"""
        self.cidpress = self.rect.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def on_press(self, event):
        """鼠标按下事件处理"""
        if event.inaxes != self.rect.axes:
            return
        contains, _ = self.rect.contains(event)
        if not contains:
            return
        x0, y0 = self.rect.get_xy()
        self.press = x0, y0, event.xdata, event.ydata
        self.rect.set_animated(True)
        self.title_text.set_animated(True)
        self.rect.figure.canvas.draw()
        self.background = self.rect.figure.canvas.copy_from_bbox(self.rect.axes.bbox)

    def on_motion(self, event):
        """鼠标移动事件处理"""
        if self.press is None:
            return
        if event.inaxes != self.rect.axes:
            return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.rect.set_xy((x0 + dx, y0 + dy))

        # 更新标题位置
        self.title_text.set_position((x0 + dx + self.rect.get_width() / 2, y0 + dy + self.rect.get_height() / 2 + 0.5))

        self.rect.figure.canvas.restore_region(self.background)
        self.rect.axes.draw_artist(self.rect)
        self.rect.axes.draw_artist(self.title_text)
        self.rect.figure.canvas.blit(self.rect.axes.bbox)

    def on_release(self, event):
        """鼠标释放事件处理"""
        if self.press is None:
            return
        self.press = None
        self.rect.set_animated(False)
        self.title_text.set_animated(False)
        self.background = None
        self.rect.figure.canvas.draw()

        # 通知位置变化
        if self.on_position_changed:
            x, y = self.rect.get_xy()
            width, height = self.rect.get_width(), self.rect.get_height()
            self.on_position_changed(self.module_name, x, y, width, height)

    def disconnect(self):
        """断开事件连接，防止 figure 已被清空时触发 NoneType 错误"""
        if getattr(self, "cidpress", None) is None:
            return  # 已断开或从未连接

        fig = getattr(self.rect, "figure", None)
        if fig is not None and getattr(fig, "canvas", None) is not None:
            fig.canvas.mpl_disconnect(self.cidpress)
            fig.canvas.mpl_disconnect(self.cidrelease)
            fig.canvas.mpl_disconnect(self.cidmotion)

        # 避免重复调用
        self.cidpress = self.cidrelease = self.cidmotion = None


class DraggableFIFO:
    """可拖拽的FIFO类，用于处理FIFO的拖拽事件"""

    def __init__(self, patches, texts, lane_name, module_name, on_position_changed=None):
        self.patches = patches  # FIFO的所有方块
        self.texts = texts  # FIFO的所有文本
        self.lane_name = lane_name
        self.module_name = module_name
        self.press = None
        self.background = None
        self.on_position_changed = on_position_changed
        self.connect()

    def connect(self):
        """连接事件处理器"""
        self.cidpress = self.patches[0].figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.cidrelease = self.patches[0].figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.cidmotion = self.patches[0].figure.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def on_press(self, event):
        """鼠标按下事件处理"""
        if event.inaxes != self.patches[0].axes:
            return

        # 检查是否点击了任何一个FIFO方块
        for patch in self.patches:
            contains, _ = patch.contains(event)
            if contains:
                # 记录所有方块的初始位置
                self.press = []
                for p in self.patches:
                    x0, y0 = p.get_xy()
                    self.press.append((x0, y0))
                self.press.append((event.xdata, event.ydata))

                # 设置动画模式
                for p in self.patches:
                    p.set_animated(True)
                for t in self.texts:
                    if t:  # 确保文本对象存在
                        t.set_animated(True)

                self.patches[0].figure.canvas.draw()
                self.background = self.patches[0].figure.canvas.copy_from_bbox(self.patches[0].axes.bbox)
                return

    def on_motion(self, event):
        """鼠标移动事件处理"""
        if self.press is None:
            return
        if event.inaxes != self.patches[0].axes:
            return

        # 计算移动距离
        xpress, ypress = self.press[-1]
        dx = event.xdata - xpress
        dy = event.ydata - ypress

        # 更新所有方块和文本的位置
        for i, patch in enumerate(self.patches):
            x0, y0 = self.press[i]
            patch.set_xy((x0 + dx, y0 + dy))

            # 更新对应文本位置
            if i < len(self.texts) and self.texts[i]:
                self.texts[i].set_position((x0 + dx + patch.get_width() / 2, y0 + dy + patch.get_height() / 2))

        # 重绘
        self.patches[0].figure.canvas.restore_region(self.background)
        for p in self.patches:
            self.patches[0].axes.draw_artist(p)
        for t in self.texts:
            if t:
                self.patches[0].axes.draw_artist(t)
        self.patches[0].figure.canvas.blit(self.patches[0].axes.bbox)

    def on_release(self, event):
        """鼠标释放事件处理"""
        if self.press is None:
            return

        # 恢复非动画模式
        for p in self.patches:
            p.set_animated(False)
        for t in self.texts:
            if t:
                t.set_animated(False)

        self.background = None
        self.patches[0].figure.canvas.draw()

        # 通知位置变化
        if self.on_position_changed:
            positions = []
            for patch in self.patches:
                x, y = patch.get_xy()
                width, height = patch.get_width(), patch.get_height()
                positions.append((x, y, width, height))
            self.on_position_changed(self.module_name, self.lane_name, positions)

        self.press = None

    def disconnect(self):
        """断开事件连接，防止 figure 已被清空时触发 NoneType 错误"""
        if getattr(self, "cidpress", None) is None:
            return  # 已断开或从未连接

        fig = getattr(self.patches[0], "figure", None) if self.patches else None
        if fig is not None and getattr(fig, "canvas", None) is not None:
            fig.canvas.mpl_disconnect(self.cidpress)
            fig.canvas.mpl_disconnect(self.cidrelease)
            fig.canvas.mpl_disconnect(self.cidmotion)

        # 避免重复调用
        self.cidpress = self.cidrelease = self.cidmotion = None


class InteractiveCrossRingVisualizer:
    """交互式Cross Ring Piece可视化器"""

    def __init__(self, config, root=None):
        """
        初始化可视化器

        参数:
        - config: 含有FIFO深度配置的对象
        - root: tkinter根窗口，如果为None则创建新窗口
        """
        self.config = config

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
        self.fifo_gap = 0.4  # 相邻fifo之间间隙

        # 布局参数
        self.margin_small = 0.05
        self.depth_scale_left = 1.0
        self.depth_scale_mid = 1.2
        self.depth_scale_right = 1.8
        self.depth_scale_top = 1.8
        self.depth_scale_down = 1.8
        self.text_offset = 0.005
        self.text_scale_mid = 0.8

        # 模块尺寸 - 初始值，会根据FIFO深度动态调整
        self.inject_module_size = (5, 3)
        self.eject_module_size = (3, 5)
        self.rb_module_size = (5, 5)

        # 模块位置 - 初始值，可通过拖拽调整
        self.module_positions = {
            "IQ": {"x": -6, "y": 0, "width": self.inject_module_size[1], "height": self.inject_module_size[0]},
            "EQ": {"x": 0, "y": 6, "width": self.eject_module_size[0], "height": self.eject_module_size[1]},
            "RB": {"x": 0, "y": 0, "width": self.rb_module_size[1], "height": self.rb_module_size[0]},
        }

        # FIFO位置配置 - 会在绘制时动态生成
        self.fifo_positions = {"IQ": {}, "EQ": {}, "RB": {}}

        # ---------- 创建 GUI / Figure ----------
        if tk is None:
            # --- 无 Tk 环境：直接用普通 matplotlib 窗口 ---
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            self.ax.axis("off")
            self.ax.set_aspect("equal")
            self.canvas = self.fig.canvas  # ensure .canvas exists in non‑Tk mode

            # 用 matplotlib.widgets 替代滑块（仅演示 IQ 深度，其他可类推）
            from matplotlib.widgets import Slider, Button

            self.depth_axes = self.fig.add_axes([0.25, 0.03, 0.5, 0.02])
            self.iq_depth_slider = Slider(self.depth_axes, "IQ depth", 1, 20, valinit=self.IQ_depth, valstep=1)
            self.iq_depth_slider.on_changed(lambda val: self.update_iq_depth(val))

            # 保存按钮
            self.btn_ax = self.fig.add_axes([0.8, 0.03, 0.1, 0.04])
            self.save_btn = Button(self.btn_ax, "Save")
            self.save_btn.on_clicked(lambda evt: self.save_layout_cli())

        else:
            # --- 有 Tk 环境：保持原先 UI ---
            self.root = tk.Tk() if root is None else root
            self.root.title("Interactive Cross Ring Piece Visualizer")
            self.root.geometry("1200x800")

            self.fig = Figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111)
            self.ax.axis("off")
            self.ax.set_aspect("equal")

            self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
            self.toolbar.update()

            # -- control frame, buttons, sliders  (retain original code) --
            self.control_frame = tk.Frame(self.root)
            self.control_frame.pack(side=tk.BOTTOM, fill=tk.X)

            # 保存 / 加载 / 重置
            self.save_btn = tk.Button(self.control_frame, text="保存布局", command=self.save_layout)
            self.save_btn.pack(side=tk.LEFT, padx=5, pady=5)
            self.load_btn = tk.Button(self.control_frame, text="加载布局", command=self.load_layout)
            self.load_btn.pack(side=tk.LEFT, padx=5, pady=5)
            self.reset_btn = tk.Button(self.control_frame, text="重置布局", command=self.reset_layout)
            self.reset_btn.pack(side=tk.LEFT, padx=5, pady=5)

            # 深度调节区 (保持原先四个滑块)
            self.depth_frame = tk.Frame(self.control_frame)
            self.depth_frame.pack(side=tk.LEFT, padx=20, pady=5)
            # IQ深度滑块
            tk.Label(self.depth_frame, text="IQ深度:").grid(row=0, column=0)
            self.iq_depth_var = tk.IntVar(value=self.IQ_depth)
            self.iq_depth_slider = tk.Scale(self.depth_frame, from_=1, to=20, orient=tk.HORIZONTAL, variable=self.iq_depth_var, command=self.update_iq_depth)
            self.iq_depth_slider.grid(row=0, column=1)
            # EQ深度滑块
            tk.Label(self.depth_frame, text="EQ深度:").grid(row=1, column=0)
            self.eq_depth_var = tk.IntVar(value=self.EQ_depth)
            self.eq_depth_slider = tk.Scale(self.depth_frame, from_=1, to=20, orient=tk.HORIZONTAL, variable=self.eq_depth_var, command=self.update_eq_depth)
            self.eq_depth_slider.grid(row=1, column=1)
            # RB深度滑块
            tk.Label(self.depth_frame, text="RB输入深度:").grid(row=2, column=0)
            self.rb_in_depth_var = tk.IntVar(value=self.RB_in_depth)
            self.rb_in_depth_slider = tk.Scale(self.depth_frame, from_=1, to=20, orient=tk.HORIZONTAL, variable=self.rb_in_depth_var, command=self.update_rb_in_depth)
            self.rb_in_depth_slider.grid(row=2, column=1)
            tk.Label(self.depth_frame, text="RB输出深度:").grid(row=3, column=0)
            self.rb_out_depth_var = tk.IntVar(value=self.RB_out_depth)
            self.rb_out_depth_slider = tk.Scale(self.depth_frame, from_=1, to=20, orient=tk.HORIZONTAL, variable=self.rb_out_depth_var, command=self.update_rb_out_depth)
            self.rb_out_depth_slider.grid(row=3, column=1)

        # 存储patch和text
        self.iq_patches, self.iq_texts = {}, {}
        self.eq_patches, self.eq_texts = {}, {}
        self.rb_patches, self.rb_texts = {}, {}

        # 存储可拖拽对象
        self.draggable_modules = {}
        self.draggable_fifos = {}

        # 调色板
        self._colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self._color_map = {}
        self._next_color = 0

        # 绘制模块
        self._draw_modules()

        # # 绘制连接箭头
        # self._draw_arrows()

        # 更新画布
        self.canvas.draw()

    def _get_color(self, flit):
        """获取flit的颜色"""
        packet_id = getattr(flit, "packet_id", str(flit))
        if packet_id not in self._color_map:
            self._color_map[packet_id] = self._colors[self._next_color % len(self._colors)]
            self._next_color += 1
        return self._color_map[packet_id]

    def update_iq_depth(self, value):
        """更新IQ深度"""
        self.IQ_depth = int(value)
        self.redraw()

    def update_eq_depth(self, value):
        """更新EQ深度"""
        self.EQ_depth = int(value)
        self.redraw()

    def update_rb_in_depth(self, value):
        """更新RB输入深度"""
        self.RB_in_depth = int(value)
        self.redraw()

    def update_rb_out_depth(self, value):
        """更新RB输出深度"""
        self.RB_out_depth = int(value)
        self.redraw()

    def save_layout_inner(self, file_path):
        # existing JSON dump body moved here
        layout_data = {
            "module_positions": self.module_positions,
            "fifo_positions": self.fifo_positions,
            "fifo_depths": {
                "IQ_depth": self.IQ_depth,
                "EQ_depth": self.EQ_depth,
                "RB_in_depth": self.RB_in_depth,
                "RB_out_depth": self.RB_out_depth,
                "IQ_CH_depth": self.IQ_CH_depth,
                "EQ_CH_depth": self.EQ_CH_depth,
            },
            "layout_params": {
                "square": self.square,
                "gap": self.gap,
                "fifo_gap": self.fifo_gap,
                "margin_small": self.margin_small,
                "depth_scale_left": self.depth_scale_left,
                "depth_scale_mid": self.depth_scale_mid,
                "depth_scale_right": self.depth_scale_right,
                "depth_scale_top": self.depth_scale_top,
                "depth_scale_down": self.depth_scale_down,
                "text_offset": self.text_offset,
                "text_scale_mid": self.text_scale_mid,
            },
        }
        with open(file_path, "w") as f:
            json.dump(layout_data, f, indent=4)
        print(f"布局已保存到: {file_path}")

    def save_layout(self):
        if tk is None:
            print("Tk not available; use save_layout_cli() instead.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not file_path:
            return
        self.save_layout_inner(file_path)

    # -------- CLI fallback save (when no Tk) --------
    def save_layout_cli(self):
        path = "layout_cli.json"
        self.save_layout_inner(path)
        print(f"Layout saved to {path}")

    def load_layout(self):
        if tk is None:
            print("Tk not available; cannot load layout via GUI.")
            return
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not file_path:
            return

        try:
            with open(file_path, "r") as f:
                layout_data = json.load(f)

            # 加载模块位置
            if "module_positions" in layout_data:
                self.module_positions = layout_data["module_positions"]

            # 加载FIFO位置
            if "fifo_positions" in layout_data:
                self.fifo_positions = layout_data["fifo_positions"]

            # 加载FIFO深度
            if "fifo_depths" in layout_data:
                depths = layout_data["fifo_depths"]
                self.IQ_depth = depths.get("IQ_depth", self.IQ_depth)
                self.EQ_depth = depths.get("EQ_depth", self.EQ_depth)
                self.RB_in_depth = depths.get("RB_in_depth", self.RB_in_depth)
                self.RB_out_depth = depths.get("RB_out_depth", self.RB_out_depth)
                self.IQ_CH_depth = depths.get("IQ_CH_depth", self.IQ_CH_depth)
                self.EQ_CH_depth = depths.get("EQ_CH_depth", self.EQ_CH_depth)

                # 更新滑块值
                self.iq_depth_var.set(self.IQ_depth)
                self.eq_depth_var.set(self.EQ_depth)
                self.rb_in_depth_var.set(self.RB_in_depth)
                self.rb_out_depth_var.set(self.RB_out_depth)

            # 加载布局参数
            if "layout_params" in layout_data:
                params = layout_data["layout_params"]
                self.square = params.get("square", self.square)
                self.gap = params.get("gap", self.gap)
                self.fifo_gap = params.get("fifo_gap", self.fifo_gap)
                self.margin_small = params.get("margin_small", self.margin_small)
                self.depth_scale_left = params.get("depth_scale_left", self.depth_scale_left)
                self.depth_scale_mid = params.get("depth_scale_mid", self.depth_scale_mid)
                self.depth_scale_right = params.get("depth_scale_right", self.depth_scale_right)
                self.depth_scale_top = params.get("depth_scale_top", self.depth_scale_top)
                self.depth_scale_down = params.get("depth_scale_down", self.depth_scale_down)
                self.text_offset = params.get("text_offset", self.text_offset)
                self.text_scale_mid = params.get("text_scale_mid", self.text_scale_mid)

            # 重绘
            self.redraw()
            print(f"布局已从 {file_path} 加载")

        except Exception as e:
            print(f"加载布局时出错: {e}")

    def reset_layout(self):
        if tk is None:
            print("Tk not available; cannot reset layout via GUI.")
            return
        # 重置模块位置
        self.module_positions = {
            "IQ": {"x": -6, "y": 0, "width": self.inject_module_size[1], "height": self.inject_module_size[0]},
            "EQ": {"x": 0, "y": 6, "width": self.eject_module_size[0], "height": self.eject_module_size[1]},
            "RB": {"x": 0, "y": 0, "width": self.rb_module_size[1], "height": self.rb_module_size[0]},
        }

        # 重置FIFO位置
        self.fifo_positions = {"IQ": {}, "EQ": {}, "RB": {}}

        # 重绘
        self.redraw()

    def redraw(self):
        """重新绘制所有内容"""
        # 清除当前图形
        self.ax.clear()
        self.ax.axis("off")
        self.ax.set_aspect("equal")

        # 清除存储的对象
        self.iq_patches.clear()
        self.iq_texts.clear()
        self.eq_patches.clear()
        self.eq_texts.clear()
        self.rb_patches.clear()
        self.rb_texts.clear()

        # 清除可拖拽对象
        for module in self.draggable_modules.values():
            module.disconnect()
        for fifo_group in self.draggable_fifos.values():
            for fifo in fifo_group.values():
                fifo.disconnect()

        self.draggable_modules.clear()
        self.draggable_fifos.clear()

        # 重新绘制模块
        self._draw_modules()

        # 重新绘制箭头
        # self._draw_arrows()

        # 更新画布
        self.canvas.draw()

    def _draw_arrows(self):
        """绘制模块间的连接箭头"""
        # 获取模块位置
        IQ_pos = self.module_positions["IQ"]
        EQ_pos = self.module_positions["EQ"]
        RB_pos = self.module_positions["RB"]

        IQ_x, IQ_y = IQ_pos["x"], IQ_pos["y"]
        IQ_w, IQ_h = IQ_pos["width"], IQ_pos["height"]

        EQ_x, EQ_y = EQ_pos["x"], EQ_pos["y"]
        EQ_w, EQ_h = EQ_pos["width"], EQ_pos["height"]

        RB_x, RB_y = RB_pos["x"], RB_pos["y"]
        RB_w, RB_h = RB_pos["width"], RB_pos["height"]

        # 箭头样式
        base_style = dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=12)

        # 箭头1: IQ → EQ
        A = (IQ_x + IQ_w, IQ_y + IQ_h * 0.25)
        B = (EQ_x, EQ_y)
        style = base_style.copy()
        style["connectionstyle"] = "arc3,rad=0.3"
        self.ax.add_patch(FancyArrowPatch(posA=A, posB=B, **style))

        # 箭头2: IQ → RB
        C = (IQ_x + IQ_w, IQ_y - IQ_h * 0.25)
        D = (RB_x, RB_y)
        style = base_style.copy()
        style["connectionstyle"] = "arc3,rad=-0.3"
        self.ax.add_patch(FancyArrowPatch(posA=C, posB=D, **style))

        # 箭头3: RB → EQ
        E = (RB_x, RB_y + RB_h)
        F = (EQ_x, EQ_y - EQ_h)
        style = base_style.copy()
        style["connectionstyle"] = "arc3,rad=0"
        self.ax.add_patch(FancyArrowPatch(posA=E, posB=F, **style))

    def _draw_modules(self):
        """绘制所有模块"""
        # 根据FIFO深度调整模块大小
        self._adjust_module_sizes()

        # 创建FIFO配置
        L = lambda name, depth, orient, quad: {
            "name": name,
            "depth": depth,
            "orient": orient,  # 'h' or 'v'
            "quad": quad,  # 象限代码，如 'lt', 'rb' 等
        }

        # Inject Queue配置
        lanes_cfg_IQ = [
            L("TL", self.IQ_depth, "v", "lb"),
            L("TR", self.IQ_depth, "v", "rb"),
            L("TU", self.IQ_depth, "h", "lm"),
            L("TD", self.IQ_depth, "h", "lm"),
        ]

        # 添加channel buffers
        if hasattr(self.config, "channel_names"):
            for ch in self.config.channel_names:
                lanes_cfg_IQ.append(L(ch, self.IQ_CH_depth, "v", "lt"))

        # Eject Queue配置
        lanes_cfg_EQ = [
            L("TU", self.EQ_depth, "h", "lm"),
            L("TD", self.EQ_depth, "h", "lm"),
        ]

        # 添加channel buffers
        if hasattr(self.config, "channel_names"):
            for ch in self.config.channel_names:
                lanes_cfg_EQ.append(L(ch, self.EQ_CH_depth, "v", "rt"))

        # Ring Bridge配置
        lanes_cfg_RB = [
            L("TL", self.RB_in_depth, "v", "lb"),
            L("TR", self.RB_in_depth, "v", "rb"),
            L("TU", self.RB_out_depth, "h", "lm"),
            L("TD", self.RB_out_depth, "h", "lm"),
            L("RB", self.RB_out_depth, "h", "lm"),
        ]

        # 绘制三个模块
        IQ_layout = {
            "margin_small": self.margin_small,
            "depth_scale_left": self.depth_scale_left,
            "depth_scale_mid": self.depth_scale_mid,
            "depth_scale_right": self.depth_scale_right,
            "depth_scale_top": self.depth_scale_top,
            "depth_scale_down": self.depth_scale_down,
            "text_offset": self.text_offset,
            "text_scale_mid": self.text_scale_mid,
            "square": self.square,
            "gap": self.gap,
            "fifo_gap": self.fifo_gap,
        }

        EQ_layout = IQ_layout.copy()
        RB_layout = IQ_layout.copy()

        # 绘制Inject Queue
        self._draw_fifo_module(
            x=self.module_positions["IQ"]["x"],
            y=self.module_positions["IQ"]["y"],
            title="Inject Queue",
            module_width=self.module_positions["IQ"]["width"],
            module_height=self.module_positions["IQ"]["height"],
            lanes_cfg=lanes_cfg_IQ,
            patch_dict=self.iq_patches,
            text_dict=self.iq_texts,
            layout=IQ_layout,
            module_name="IQ",
        )

        # 绘制Eject Queue
        self._draw_fifo_module(
            x=self.module_positions["EQ"]["x"],
            y=self.module_positions["EQ"]["y"],
            title="Eject Queue",
            module_width=self.module_positions["EQ"]["width"],
            module_height=self.module_positions["EQ"]["height"],
            lanes_cfg=lanes_cfg_EQ,
            patch_dict=self.eq_patches,
            text_dict=self.eq_texts,
            layout=EQ_layout,
            module_name="EQ",
        )

        # 绘制Ring Bridge
        self._draw_fifo_module(
            x=self.module_positions["RB"]["x"],
            y=self.module_positions["RB"]["y"],
            title="Ring Bridge",
            module_width=self.module_positions["RB"]["width"],
            module_height=self.module_positions["RB"]["height"],
            lanes_cfg=lanes_cfg_RB,
            patch_dict=self.rb_patches,
            text_dict=self.rb_texts,
            layout=RB_layout,
            module_name="RB",
        )

        # 调整视图
        self.ax.relim()
        self.ax.autoscale_view()

    def _adjust_module_sizes(self):
        """根据FIFO深度调整模块大小"""
        # 计算基础大小
        base_width = 3
        base_height = 5

        # 根据FIFO深度线性调整
        iq_width = base_width + self.IQ_depth * 0.1
        iq_height = base_height + self.IQ_depth * 0.1

        eq_width = base_width + self.EQ_depth * 0.1
        eq_height = base_height + self.EQ_depth * 0.1

        rb_width = base_width + max(self.RB_in_depth, self.RB_out_depth) * 0.1
        rb_height = base_height + max(self.RB_in_depth, self.RB_out_depth) * 0.1

        # 更新模块尺寸
        self.inject_module_size = (iq_height, iq_width)
        self.eject_module_size = (eq_width, eq_height)
        self.rb_module_size = (rb_height, rb_width)

        # 更新模块位置字典中的尺寸
        self.module_positions["IQ"]["width"] = iq_width
        self.module_positions["IQ"]["height"] = iq_height

        self.module_positions["EQ"]["width"] = eq_width
        self.module_positions["EQ"]["height"] = eq_height

        self.module_positions["RB"]["width"] = rb_width
        self.module_positions["RB"]["height"] = rb_height

    def _draw_fifo_module(
        self,
        x: float,
        y: float,
        title: str,
        module_width: float,
        module_height: float,
        lanes_cfg: list,
        patch_dict: dict,
        text_dict: dict,
        layout: dict = None,
        module_name: str = "",
    ):
        """
        绘制单个FIFO模块

        参数:
        - x, y: 模块中心坐标
        - title: 模块标题
        - module_width, module_height: 模块宽高
        - lanes_cfg: FIFO配置列表
        - patch_dict, text_dict: 存储绘制对象的字典
        - layout: 布局参数
        - module_name: 模块名称
        """
        # 解析布局参数
        layout = layout or {}
        margin = layout.get("margin_small", self.margin_small)
        d_left = layout.get("depth_scale_left", self.depth_scale_left)
        d_mid = layout.get("depth_scale_mid", self.depth_scale_mid)
        d_right = layout.get("depth_scale_right", self.depth_scale_right)
        d_top = layout.get("depth_scale_top", self.depth_scale_top)
        d_down = layout.get("depth_scale_down", self.depth_scale_down)
        t_off = layout.get("text_offset", self.text_offset)
        t_mid = layout.get("text_scale_mid", self.text_scale_mid)
        square = layout.get("square", self.square)
        gap = layout.get("gap", self.gap)
        fifo_gap = layout.get("fifo_gap", self.fifo_gap)

        # 辅助映射
        map_h = {"t": "top", "b": "bottom", "m": "mid"}
        map_v = {"l": "left", "r": "right", "m": "mid"}

        # 解码FIFO配置
        def decode(idx_cfg):
            orient = idx_cfg["orient"]
            quad = idx_cfg["quad"]
            v_char, h_char = quad[0], quad[1]
            if orient == "h":
                hpos = map_h[h_char]
                vpos = map_v[v_char]
            else:  # orient == 'v'
                hpos = map_h[h_char]
                vpos = map_v[v_char]
            return orient, hpos, vpos

        # 构建位置列表
        orientations, h_positions, v_positions = [], [], []
        for cfg in lanes_cfg:
            o, hpos, vpos = decode(cfg)
            orientations.append(o)
            h_positions.append(hpos)
            v_positions.append(vpos)

        # 创建分组索引
        group_map = defaultdict(list)
        for i, (o, h, v) in enumerate(zip(orientations, h_positions, v_positions)):
            group_map[(o, h, v)].append(i)

        idx_in_group = {}
        for group, idxs in group_map.items():
            for local_idx, global_idx in enumerate(idxs):
                idx_in_group[global_idx] = local_idx

        # 绘制模块边框和标题
        rect = Rectangle((x - module_width / 2, y - module_height / 2), module_width, module_height, fill=False, edgecolor="black", linewidth=1.5)
        self.ax.add_patch(rect)

        title_text = self.ax.text(x, y + module_height / 2 + margin, title, ha="center", va="bottom", fontweight="bold", fontsize=12)

        # 创建可拖拽模块
        self.draggable_modules[module_name] = DraggableModule(rect, title_text, module_name, self.on_module_position_changed)

        # 初始化FIFO位置字典
        if module_name not in self.fifo_positions:
            self.fifo_positions[module_name] = {}

        # 清空当前模块的patch和text字典
        patch_dict.clear()
        text_dict.clear()

        # 如果模块名不在可拖拽FIFO字典中，初始化它
        if module_name not in self.draggable_fifos:
            self.draggable_fifos[module_name] = {}

        # 绘制每个FIFO
        for i, cfg in enumerate(lanes_cfg):
            lane = cfg["name"]
            depth = cfg["depth"]
            orient = orientations[i]
            hpos = h_positions[i]
            vpos = v_positions[i]
            gidx = idx_in_group[i]
            gsize = len(group_map[(orient, hpos, vpos)])

            # 初始化FIFO的patch和text列表
            patch_dict[lane] = []
            text_dict[lane] = []

            # 水平方向FIFO
            if orient == "h":
                # y坐标
                if hpos == "top":
                    lane_y = y + module_height / 2 - margin - gidx * fifo_gap
                    txt_va = "bottom"
                elif hpos == "bottom":
                    lane_y = y - module_height / 2 + margin + gidx * fifo_gap
                    txt_va = "top"
                else:  # "mid"
                    lane_y = y + (gidx - (gsize - 1) / 2) * fifo_gap
                    txt_va = "center"

                # x坐标
                if vpos == "right":
                    lane_x = x + module_width / 2 - depth * (square + gap) * d_right
                    text_x = x + module_width / 2 - depth * (square + gap) * d_mid
                    slot_dir, ha = -1, "right"
                elif vpos == "left":
                    lane_x = x - module_width / 2 + depth * (square + gap) * d_left
                    text_x = x - module_width / 2 + depth * (square + gap) * d_mid
                    slot_dir, ha = 1, "left"
                else:  # "mid"
                    lane_x = x - depth * (square + gap) / 2
                    text_x = x - module_width / 4
                    slot_dir, ha = 1, "center"

                # 绘制FIFO标签
                self.ax.text(text_x, lane_y, lane, ha=ha, va=txt_va, fontsize=10)

                # 绘制每个槽
                fifo_positions = []
                for s in range(depth):
                    sx = lane_x + slot_dir * s * (square + gap)
                    sy = lane_y

                    # 保存位置信息
                    fifo_positions.append((sx, sy, square, square))

                    # 绘制方块
                    rect = Rectangle((sx - square / 2, sy - square / 2), square, square, edgecolor="black", facecolor="none", linewidth=1)
                    self.ax.add_patch(rect)
                    patch_dict[lane].append(rect)

                    # 添加文本
                    txt = self.ax.text(sx, sy + (square / 2 + t_off if hpos == "top" else -square / 2 - t_off), "", ha="center", va=txt_va, fontsize=8)
                    text_dict[lane].append(txt)

                # 保存FIFO位置
                self.fifo_positions[module_name][lane] = fifo_positions

            # 垂直方向FIFO
            else:  # orient == "v"
                # x坐标
                if vpos == "left":
                    lane_x = x - module_width / 2 + margin + gidx * fifo_gap
                    txt_ha = "left"
                elif vpos == "right":
                    lane_x = x + module_width / 2 - margin - gidx * fifo_gap
                    txt_ha = "right"
                else:  # "mid"
                    lane_x = x + (gidx - (gsize - 1) / 2) * fifo_gap
                    txt_ha = "center"

                # y坐标
                if hpos == "top":
                    lane_y = y + module_height / 2 - depth * (square + gap) * d_top
                    text_y = y + module_height / 2 - depth * (square + gap) * d_mid
                    slot_dir, va = -1, "top"
                elif hpos == "bottom":
                    lane_y = y - module_height / 2 + depth * (square + gap) * d_down
                    text_y = y - module_height / 2 + depth * (square + gap) * d_mid
                    slot_dir, va = 1, "bottom"
                else:  # "mid"
                    lane_y = y - depth * (square + gap) / 2
                    text_y = y - module_height / 4
                    slot_dir, va = 1, "center"

                # 绘制FIFO标签
                self.ax.text(lane_x, text_y, lane, ha=txt_ha, va=va, fontsize=10)

                # 绘制每个槽
                fifo_positions = []
                for s in range(depth):
                    sx = lane_x
                    sy = lane_y + slot_dir * s * (square + gap)

                    # 保存位置信息
                    fifo_positions.append((sx, sy, square, square))

                    # 绘制方块
                    rect = Rectangle((sx - square / 2, sy - square / 2), square, square, edgecolor="black", facecolor="none", linewidth=1)
                    self.ax.add_patch(rect)
                    patch_dict[lane].append(rect)

                    # 添加文本
                    txt = self.ax.text(sx + (square / 2 + t_off if vpos == "left" else -square / 2 - t_off), sy, "", ha=txt_ha, va="center", fontsize=8)
                    text_dict[lane].append(txt)

                # 保存FIFO位置
                self.fifo_positions[module_name][lane] = fifo_positions

            # 创建可拖拽FIFO
            self.draggable_fifos[module_name][lane] = DraggableFIFO(patch_dict[lane], text_dict[lane], lane, module_name, self.on_fifo_position_changed)

    def on_module_position_changed(self, module_name, x, y, width, height):
        """模块位置变化回调"""
        self.module_positions[module_name] = {"x": x + width / 2, "y": y + height / 2, "width": width, "height": height}  # 转换为中心坐标

    def on_fifo_position_changed(self, module_name, lane_name, positions):
        """FIFO位置变化回调"""
        self.fifo_positions[module_name][lane_name] = positions

    def draw_piece_for_node(self, network, node_id):
        """
        根据网络状态更新可视化

        参数:
        - network: 包含FIFO状态的网络对象
        - node_id: 节点ID
        """
        self.node_id = node_id

        # 获取队列
        IQ = network.inject_queue
        EQ = network.eject_queue
        RB = network.ring_bridge

        # 更新Inject Queue
        for lane, patches in self.iq_patches.items():
            q = IQ.get(lane, [])[node_id] if hasattr(IQ, "get") else []
            for idx, p in enumerate(patches):
                t = self.iq_texts[lane][idx]
                if idx < len(q):
                    flit = q[idx]
                    packet_id = getattr(flit, "packet_id", None)
                    flit_id = getattr(flit, "flit_id", str(flit))

                    # 设置颜色和文本
                    p.set_facecolor(self._get_color(flit))
                    t.set_text(f"{packet_id}-{flit_id}")
                else:
                    p.set_facecolor("none")
                    t.set_text("")

        # 更新Eject Queue
        for lane, patches in self.eq_patches.items():
            q = EQ.get(lane, [])[node_id] if hasattr(EQ, "get") else []
            for idx, p in enumerate(patches):
                t = self.eq_texts[lane][idx]
                if idx < len(q):
                    flit = q[idx]
                    packet_id = getattr(flit, "packet_id", None)
                    flit_id = getattr(flit, "flit_id", str(flit))

                    # 设置颜色和文本
                    p.set_facecolor(self._get_color(flit))
                    t.set_text(f"{packet_id}-{flit_id}")
                else:
                    p.set_facecolor("none")
                    t.set_text("")

        # 更新Ring Bridge
        for lane, patches in self.rb_patches.items():
            q = RB.get(lane, [])[(node_id, node_id)] if hasattr(RB, "get") else []
            for idx, p in enumerate(patches):
                t = self.rb_texts[lane][idx]
                if idx < len(q):
                    flit = q[idx]
                    packet_id = getattr(flit, "packet_id", None)
                    flit_id = getattr(flit, "flit_id", str(flit))

                    # 设置颜色和文本
                    p.set_facecolor(self._get_color(flit))
                    t.set_text(f"{packet_id}-{flit_id}")
                else:
                    p.set_facecolor("none")
                    t.set_text("")

        # 设置标题
        self.ax.set_title(f"Node: {node_id}", fontsize=12)

        # 更新画布
        self.canvas.draw()

    def run(self):
        """运行主循环"""
        self.root.mainloop()


# 示例用法
if __name__ == "__main__":
    # 创建一个简单的配置对象
    class Config:
        def __init__(self):
            self.cols = 4
            self.rows = 4
            self.IQ_OUT_FIFO_DEPTH = 4
            self.EQ_IN_FIFO_DEPTH = 4
            self.RB_IN_FIFO_DEPTH = 4
            self.RB_OUT_FIFO_DEPTH = 4
            self.seats_per_link = 4
            self.IQ_CH_FIFO_DEPTH = 4
            self.EQ_CH_FIFO_DEPTH = 4
            self.channel_names = ["CH0", "CH1"]

    # 创建可视化器
    config = Config()
    visualizer = InteractiveCrossRingVisualizer(config)

    # 运行
    if tk is None:
        visualizer.redraw()
        plt.show()
    else:
        visualizer.run()
