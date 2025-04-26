import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from collections import defaultdict, deque
import copy
import threading
import time
import matplotlib


class NetworkLinkVisualizer:
    def __init__(self, network):
        self.network = network
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_aspect("equal")
        self.node_positions = self._calculate_layout()
        self.link_artists = {}  # 存储链路相关的静态信息
        self._colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self._color_map = {}
        self._next_color = 0
        self._draw_static_elements()

        # 播放控制参数
        self.pause_interval = 0.1        # 默认每帧暂停间隔(秒)
        
        # 状态管理
        self.state_history = []          # 历史状态记录
        self.current_history_index = -1  # 当前显示的历史帧索引
        self.is_paused = False           # 暂停状态标志
        self.pause_event = threading.Event()  # 用于阻塞主线程的事件对象
        self.pause_event.set()           # 初始未暂停状态
        
        # 状态显示文本
        self.status_text = self.ax.text(0.02, 0.98, "Running", transform=self.ax.transAxes,
                                        fontsize=12, fontweight='bold', color='green',
                                        verticalalignment='top')
        
        # 绑定键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        # 显示帮助信息
        self._show_controls_help()

    def _calculate_layout(self):
        """根据网格计算节点位置（可调整节点间距）"""
        pos = {}
        for node in range(self.network.config.rows * self.network.config.cols):
            x, y = node % self.network.config.cols, node // self.network.config.cols
            # 为了美观，按照行列计算位置，并添加些许偏移
            if y % 2 == 1:  # 奇数行左移
                x -= 0.16
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

        # 绘制所有链路的框架
        self.link_artists.clear()
        for src, dest in self.network.links.keys():
            self._draw_link_frame(src, dest)

        # 根据节点位置自动调整显示范围
        if xs and ys:
            # 计算边界，并设定一定的补充边距
            margin_x = (max(xs) - min(xs)) * 0.1
            margin_y = (max(ys) - min(ys)) * 0.1
            self.ax.set_xlim(min(xs) - margin_x, max(xs) + margin_x + 0.3)
            self.ax.set_ylim(min(ys) - margin_y, max(ys) + margin_y + 0.3)

        self.ax.axis("off")
        plt.tight_layout()
        
        # 重新添加状态文本
        self.status_text = self.ax.text(0.02, 0.98, "Running", transform=self.ax.transAxes,
                                        fontsize=12, fontweight='bold', color='green',
                                        verticalalignment='top')

    def _draw_link_frame(self, src, dest, queue_fixed_length=2):
        # 节点矩形尺寸
        node_width = 0.5
        node_height = 0.5
        half_w, half_h = node_width / 2, node_height / 2

        # 计算节点中心（假设存储的是左下角坐标）
        src_center = (self.node_positions[src][0] + half_w, self.node_positions[src][1] + half_h)
        dest_center = (self.node_positions[dest][0] + half_w, self.node_positions[dest][1] + half_h)

        # 计算中心向量和距离
        dx = dest_center[0] - src_center[0]
        dy = dest_center[1] - src_center[1]
        center_distance = np.hypot(dx, dy)
        if center_distance == 0:
            return  # 避免自连接
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

        # 根据箭头方向确定队列框架放置在箭头的哪一侧
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
        if abs(dx) >= abs(dy):
            queue_width = queue_fixed_length
            queue_height = 0.2
        else:
            queue_width = 0.2
            queue_height = queue_fixed_length

        # 绘制队列框架矩形，中心位于 queue_center
        queue = Rectangle((queue_center[0] - queue_width / 2, queue_center[1] - queue_height / 2), queue_width, queue_height, facecolor="white", edgecolor="black", linestyle="--")
        self.ax.add_patch(queue)

        # 绘制箭头连接线，并使用 annotate 添加箭头头部
        self.ax.annotate("", xy=dest_arrow, xycoords="data", xytext=src_arrow, textcoords="data", arrowprops=dict(arrowstyle="->", color="blue", lw=2))

        # 存储链路绘制信息，可用于后续动态更新
        link_id = f"{src}-{dest}"
        self.link_artists[link_id] = {"queue_center": queue_center, "queue_width": queue_width, "queue_height": queue_height}

    def update(self, network=None, use_highlight=False):
        """
        更新每条链路队列中 flit 的显示
        - 如果在回放模式，此函数使用存储的状态，否则使用传入的network
        - 当在实时模式时，更新会被保存到历史记录中
        """
        # 检查是否在暂停状态 - 这会阻塞调用函数
        if not self.pause_event.is_set():
            print("暂停 - 等待恢复...")
        self.pause_event.wait()
        
        # 如果我们在回放历史记录模式并且索引有效
        if self.current_history_index >= 0 and self.current_history_index < len(self.state_history):
            # 使用历史记录中的状态
            links_state = self.state_history[self.current_history_index]
            self._update_display(links_state, use_highlight)
            return self.ax.patches
            
        # 实时模式: 使用当前network状态更新显示
        if network:
            self.network = network
            
        # 保存当前状态到历史记录
        links_state = self._capture_links_state()
        self.state_history.append(links_state)
        self.current_history_index = len(self.state_history) - 1
        
        # 更新显示
        self._update_display(links_state, use_highlight)
        
        return self.ax.patches
        
    def _capture_links_state(self):
        """
        捕获当前链路状态的轻量级快照，不使用深拷贝
        只保存需要渲染的关键信息
        """
        state = {}
        for (src, dest), flit_list in self.network.links.items():
            link_id = f"{src}-{dest}"
            # 只保存渲染所需的信息
            state[link_id] = []
            for flit in flit_list:
                if flit is None:
                    state[link_id].append(None)
                else:
                    # 只保存绘制所需的属性
                    state[link_id].append({
                        'packet_id': flit.packet_id,
                        'flit_id': flit.flit_id
                    })
        return state
    
    def _update_display(self, links_state, use_highlight=False):
        """
        使用存储的状态更新显示
        """
        # 清除之前的flit显示
        for link_id, artists_dict in self.link_artists.items():
            if "flit_artists" in artists_dict:
                for artist in artists_dict["flit_artists"]:
                    artist.remove()
            artists_dict["flit_artists"] = []

        # 遍历所有链路状态
        for link_id, flit_list in links_state.items():
            if link_id not in self.link_artists:
                continue
                
            # 解析源和目标节点
            src, dest = map(int, link_id.split('-'))
            
            # 获取队列信息
            queue_info = self.link_artists[link_id]
            queue_center = queue_info["queue_center"]
            queue_width = queue_info["queue_width"]
            queue_height = queue_info["queue_height"]

            # 确定链路方向
            dx = src % self.network.config.cols - dest % self.network.config.cols
            dy = src // self.network.config.cols - dest // self.network.config.cols

            # 计算队列区域参数
            q_ll = (queue_center[0] - queue_width / 2, queue_center[1] - queue_height / 2)

            # 根据链路方向确定是水平还是垂直队列
            is_horizontal = abs(dx) > abs(dy)

            # 确定队列方向(正向或反向)
            if is_horizontal:
                is_forward = dx < 0  # 向右为正向
            else:
                is_forward = dy > 0  # 向上为正向

            margin = 0.02
            flit_size = 0.15  # flit方形大小
            spacing = flit_size * 1.2  # flit之间的间距

            # 计算需要的总空间
            num_flits = len(flit_list)
            if is_horizontal:
                required_space = num_flits * spacing
                available_space = queue_width - 2 * margin
                spacing = available_space / num_flits if num_flits > 0 else spacing
            else:
                required_space = num_flits * spacing
                available_space = queue_height - 2 * margin
                spacing = available_space / num_flits if num_flits > 0 else spacing

            # 绘制所有flit位置
            flit_artists = []
            for i in range(num_flits):
                # 计算位置 - 考虑方向
                if is_horizontal:
                    x = q_ll[0] + margin + (i + 0.5) * spacing
                    y = queue_center[1]
                    # 如果是向左的链路，反转顺序
                    if not is_forward:
                        x = q_ll[0] + queue_width - margin - (i + 0.5) * spacing
                else:
                    x = queue_center[0]
                    y = q_ll[1] + margin + (i + 0.5) * spacing
                    # 如果是向下的链路，反转顺序
                    if not is_forward:
                        y = q_ll[1] + queue_height - margin - (i + 0.5) * spacing

                # 创建方形
                flit_data = flit_list[i]
                if flit_data is None:
                    continue
                
                # 创建用于颜色计算的模拟flit对象
                class FlitView:
                    def __init__(self, data):
                        if isinstance(data, dict):
                            self.packet_id = data['packet_id']
                            self.flit_id = data['flit_id']
                        else:
                            # 如果已经是FlitView或类似对象，直接使用
                            self.packet_id = data.packet_id
                            self.flit_id = data.flit_id
                
                flit = FlitView(flit_data) if isinstance(flit_data, dict) else flit_data
                facecolor = self._get_flit_color(flit, use_highlight=use_highlight)
                rect = Rectangle((x - flit_size / 2, y - flit_size / 2), flit_size, flit_size, facecolor=facecolor, edgecolor="black")

                # 添加文本标签 - 根据方向调整位置和显示方式
                label = f"{flit.packet_id}.{flit.flit_id}"

                if is_horizontal:
                    label = f"{flit.packet_id}\n{flit.flit_id}"
                    # 根据方向决定 y 偏移
                    if is_forward:
                        y_text = y - flit_size * 2 - 0.1
                    else:
                        y_text = y + flit_size * 2 + 0.1

                    # 一次性绘制两行
                    txt = self.ax.text(
                        x + (i - 0.2) * 0.04 * (int(is_forward)-0.5), y_text, label,
                        ha="center", va="center",
                        fontsize=9
                    )
                    flit_artists.append(txt)
                else:
                    # 垂直链路：横向标签
                    if is_forward:  # 向上的链路
                        text_x = x + flit_size * 1.1  # 右侧
                        ha = "left"
                    else:  # 向下的链路
                        text_x = x - flit_size * 1.1  # 左侧
                        ha = "right"
                    text_y = y
                    text = self.ax.text(text_x, text_y, label, ha=ha, va="center", fontsize=9)
                    flit_artists.append(text)

                self.ax.add_patch(rect)
                flit_artists.append(rect)

            # 保存该链路的图形对象
            self.link_artists[link_id]["flit_artists"] = flit_artists

        # 更新状态显示
        self._update_status_display()
        
        # 标题与排版
        title = f"{self.network.name}"
        if self.is_paused:
            title += f" [已暂停 - 帧 {self.current_history_index + 1}/{len(self.state_history)}]"
        self.ax.set_title(title)
        
        plt.tight_layout()
        plt.pause(self.pause_interval)

    def _update_status_display(self):
        """更新状态显示"""
        if self.is_paused:
            status = f"Pause."
            color = 'red'
        else:
            status = f"Running..."
            color = 'green'
            
        # 更新状态文本
        self.status_text.set_text(status)
        self.status_text.set_color(color)

    def _on_key(self, event):
        key = event.key

        if key == 'up':
            # 加快 --> 缩短 pause_interval，但不跑到 0
            self.pause_interval = max(1e-3, self.pause_interval*0.75)
            print(f"播放间隔：{self.pause_interval:.3f}s (速度：{1/self.pause_interval:.1f}帧/秒)")
            self._update_status_display()
        elif key == 'down':
            # 减慢
            self.pause_interval *= 1.25
            print(f"播放间隔：{self.pause_interval:.3f}s (速度：{1/self.pause_interval:.1f}帧/秒)")
            self._update_status_display()
        elif key == 'p':
            # 暂停/继续
            self._toggle_pause()
        elif key == 'left' and self.is_paused:
            # 回退一帧 (只在暂停时有效)
            self._go_to_previous_frame()
        elif key == 'right' and self.is_paused:
            # 前进一帧 (只在暂停时有效)
            self._go_to_next_frame()
        elif key == 'home' and self.is_paused:
            # 回到第一帧
            self._go_to_first_frame()
        elif key == 'end' and self.is_paused:
            # 去到最后一帧
            self._go_to_last_frame()
        elif key == 'h':
            # 显示帮助
            self._show_controls_help()
    
    def _toggle_pause(self):
        """切换暂停/继续状态"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_event.clear()  # 阻塞主线程
            print("播放暂停 - 使用左右箭头浏览帧")
        else:
            self.pause_event.set()   # 恢复主线程
            print("播放继续")
        
        # 更新显示状态和标题
        if self.current_history_index >= 0 and self.current_history_index < len(self.state_history):
            self._update_display(self.state_history[self.current_history_index])
    
    def _go_to_previous_frame(self):
        """回退到上一帧"""
        if self.current_history_index > 0:
            self.current_history_index -= 1
            self._update_display(self.state_history[self.current_history_index])
            print(f"显示帧 {self.current_history_index + 1}/{len(self.state_history)}")
    
    def _go_to_next_frame(self):
        """前进到下一帧"""
        if self.current_history_index < len(self.state_history) - 1:
            self.current_history_index += 1
            self._update_display(self.state_history[self.current_history_index])
            print(f"显示帧 {self.current_history_index + 1}/{len(self.state_history)}")
    
    def _go_to_first_frame(self):
        """前往第一帧"""
        if len(self.state_history) > 0:
            self.current_history_index = 0
            self._update_display(self.state_history[self.current_history_index])
            print(f"显示帧 {self.current_history_index + 1}/{len(self.state_history)}")
    
    def _go_to_last_frame(self):
        """前往最后一帧"""
        if len(self.state_history) > 0:
            self.current_history_index = len(self.state_history) - 1
            self._update_display(self.state_history[self.current_history_index])
            print(f"显示帧 {self.current_history_index + 1}/{len(self.state_history)}")
    
    def _show_controls_help(self):
        """显示控制帮助信息"""
        help_text = """
        --- 可视化器控制帮助 ---
        空格键: 暂停/继续播放
        左/右箭头: 前一帧/后一帧 (暂停时)
        Home/End: 第一帧/最后一帧 (暂停时)
        上/下箭头: 加速/减速播放
        H键: 显示此帮助
        """
        print(help_text)
        
        # 如果要在可视化窗口中显示帮助，可以使用以下代码
        # plt.figure(figsize=(6, 3))
        # plt.text(0.5, 0.5, help_text, ha='center', va='center', fontsize=10)
        # plt.axis('off')
        # plt.title("控制帮助")
        # plt.tight_layout()
        # plt.show(block=False)

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
        # 统一提取 packet_id 作为颜色依据
        pid = {"packet_id": flit.packet_id, "flit_id": flit.flit_id}
        if isinstance(pid, tuple) and len(pid) >= 1:
            color_key = pid[0]  # 元组第一个元素作为 packet_id
        elif isinstance(pid, dict):
            color_key = pid.get("packet_id", str(pid))
        else:
            color_key = pid

        # 如果不启用高亮功能，使用原有逻辑
        if not use_highlight:
            if color_key in self._color_map:
                return self._color_map[color_key]
            c = self._colors[self._next_color % len(self._colors)]
            self._color_map[color_key] = c
            self._next_color += 1
            return c

        # 启用高亮功能时的逻辑
        if highlight_color is None:
            highlight_color = "red"  # 默认高亮红色

        # 仅期望包高亮，其他灰色
        if flit.packet_id == expected_packet_id:
            return highlight_color
        return "grey"