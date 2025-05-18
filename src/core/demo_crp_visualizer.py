#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross Ring Piece Visualizer演示程序
"""

# ---------------- optional Tk support ---------------- #
try:
    import tkinter as tk
except ImportError:
    tk = None

import sys
import matplotlib
from interactive_crp_visualizer import InteractiveCrossRingVisualizer
import matplotlib.pyplot as plt

# macOS 后端处理
if sys.platform == "darwin":
    matplotlib.use("macosx")  # only switch if Tk is available


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


# 创建一个简单的网络对象用于演示
class DemoNetwork:
    def __init__(self):
        # 模拟注入队列
        self.inject_queue = {"TL": {0: [DemoFlit(1, 1), DemoFlit(1, 2)]}, "TR": {0: [DemoFlit(2, 1)]}, "TU": {0: []}, "TD": {0: []}, "CH0": {0: [DemoFlit(3, 1)]}, "CH1": {0: []}}

        # 模拟弹出队列
        self.eject_queue = {"TU": {0: [DemoFlit(4, 1)]}, "TD": {0: []}, "CH0": {0: []}, "CH1": {0: [DemoFlit(5, 1)]}}

        # 模拟环桥
        self.ring_bridge = {
            "TL": {(0, 0): [DemoFlit(6, 1)]},
            "TR": {(0, 0): []},
            "TU": {(0, 0): [DemoFlit(7, 1), DemoFlit(7, 2)]},
            "TD": {(0, 0): []},
            "RB": {(0, 0): [DemoFlit(8, 1)]},
        }

    def get(self, lane, default):
        """模拟get方法"""
        return default


# 模拟Flit对象
class DemoFlit:
    def __init__(self, packet_id, flit_id):
        self.packet_id = packet_id
        self.flit_id = flit_id

    def __str__(self):
        return f"{self.packet_id}-{self.flit_id}"


# 主函数
def main():
    # 创建配置
    config = Config()

    # 创建可视化器 & 网络
    if tk is None:
        # --- 无 Tk 环境：使用纯 matplotlib ---
        visualizer = InteractiveCrossRingVisualizer(config, root=None)  # ax fallback inside
        network = DemoNetwork()
        visualizer.draw_piece_for_node(network, 0)
        plt.show()
    else:
        # --- 有 Tk 环境：保留原窗口与按钮 ---
        root = tk.Tk()
        root.title("Cross Ring Piece Visualizer Demo")

        visualizer = InteractiveCrossRingVisualizer(config, root)
        network = DemoNetwork()
        visualizer.draw_piece_for_node(network, 0)

        update_btn = tk.Button(root, text="更新演示数据", command=lambda: visualizer.draw_piece_for_node(network, 0))
        update_btn.pack(side=tk.BOTTOM, pady=10)

        root.mainloop()


if __name__ == "__main__":
    main()
