"""
calibrate_iq_layout.py
交互式微调 Inject-Queue 布局常量 → 保存 JSON
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import sys

# macOS 后端处理
if sys.platform == "darwin":
    matplotlib.use("macosx")

# ------- 1. 创建配置对象 -------------------------------------------------
from config.config import CrossRingConfig
from src.core.CrossRing_Piece_Visualizer import CrossRingVisualizer

config_path = r"../../config/config2.json"
cfg = CrossRingConfig(config_path)

# ------- 2. 初始参数 -----------------------------------------------------
init = dict(
    margin_small=0.25,
    depth_scale_left=7.0,
    depth_scale_mid=1.5,
    depth_scale_right=1.8,
    depth_scale_top=0.2,
    depth_scale_down=0.2,
    text_offset=0.005,
    text_scale_mid=0.8,
    square=0.17,
    gap=0.02,
    fifo_gap=0.35,
)

# ------- 3. 建立画布 & 预留滑块区 ----------------------------------------
fig = plt.figure(figsize=(10, 8))
ax_vis = fig.add_axes([0.05, 0.30, 0.9, 0.65])
plt.subplots_adjust(left=0.1, bottom=0.23)

# ------- 4. 生成滑块 ------------------------------------------------------
sliders = {}
slider_specs = [
    ("margin_small", 0.0, 1.0),
    ("depth_scale_left", 0.5, 10.0),
    ("depth_scale_mid", 0.5, 10.0),
    ("depth_scale_right", 0.5, 10.0),
    ("depth_scale_top", 0.0, 2.0),
    ("depth_scale_down", 0.0, 2.0),
    ("fifo_gap", 0.05, 1.0),
]

for i, (key, vmin, vmax) in enumerate(slider_specs):
    ax_slider = plt.axes([0.1, 0.20 - 0.03 * i, 0.8, 0.02])
    sliders[key] = Slider(ax_slider, key, vmin, vmax, valinit=init[key], valstep=0.01)

ax_save = plt.axes([0.8, 0.02, 0.15, 0.04])
btn_save = Button(ax_save, "Save JSON", hovercolor="0.975")


# ------- 5. 绘制函数 ------------------------------------------------------
def redraw(event=None):
    layout = {k: (sliders[k].val if k in sliders else v) for k, v in init.items()}
    ax_vis.cla()
    ax_vis.set_aspect("equal")

    viz = CrossRingVisualizer(cfg, ax_vis)
    L = lambda name, depth, orient, quad: {"name": name, "depth": depth, "orient": orient, "quad": quad}

    lanes_cfg_IQ = [
        *[L(ch, 8, "v", "lt") for ch in cfg.CH_NAME_LIST],
        L("TL", 8, "v", "lb"),
        L("TR", 8, "v", "lb"),
        L("TU", 8, "h", "rm"),
        L("TD", 8, "h", "rm"),
        L("EQ", 8, "h", "rm"),
    ]

    viz._draw_fifo_module(
        x=-3.0,
        y=0.0,
        title="Inject Queue",
        module_height=viz.inject_module_size[0],
        module_width=viz.inject_module_size[1],
        lanes_cfg=lanes_cfg_IQ,
        patch_dict={},
        text_dict={},
        layout=layout,
    )
    fig.canvas.draw_idle()


for s in sliders.values():
    s.on_changed(redraw)

redraw()  # 初次绘制


# ------- 6. 保存回调 ------------------------------------------------------
def save_json(event):
    layout = {k: (sliders[k].val if k in sliders else v) for k, v in init.items()}
    Path("iq_layout.json").write_text(json.dumps(layout, indent=2))
    print("Layout saved to iq_layout.json:", layout)


btn_save.on_clicked(save_json)

plt.show()
