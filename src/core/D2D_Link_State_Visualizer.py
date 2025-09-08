"""
D2D网络链路状态可视化器
支持多个Die和每个Die的3个网络切换（REQ、RSP、DATA）
每次显示一个Die的一个网络，通过按键切换不同Die和网络
"""

from .Link_State_Visualizer import NetworkLinkVisualizer
from collections import deque
from matplotlib.widgets import Button


class D2D_Link_State_Visualizer(NetworkLinkVisualizer):
    """D2D网络链路可视化器，支持多Die和网络切换"""
    
    def __init__(self, num_dies, initial_network=None):
        """
        初始化D2D可视化器
        
        Args:
            num_dies: Die数量（2, 4, 8等）
            initial_network: 初始网络对象（用于获取配置信息）
        """
        # 使用initial_network初始化父类
        super().__init__(initial_network)
        
        # D2D特定属性
        self.num_dies = num_dies
        self.current_die = 0  # 当前显示的Die索引
        self.current_network = 2  # 当前网络索引 (0=REQ, 1=RSP, 2=DATA)
        
        # 重新初始化以支持多Die多网络
        self._reinitialize_for_d2d()
    
    def _reinitialize_for_d2d(self):
        """重新初始化以支持D2D（多Die × 3网络）"""
        # 创建扁平的历史缓冲区数组以兼容父类
        # 索引计算：die_id * 3 + network_id
        total_networks = self.num_dies * 3
        self.histories = [deque(maxlen=20) for _ in range(total_networks)]
        
        # 同时保持二维索引的便利性
        self._die_histories = [[deque(maxlen=20) for _ in range(3)] 
                              for _ in range(self.num_dies)]
        
        # 同步两个结构
        for die_id in range(self.num_dies):
            for net_id in range(3):
                flat_idx = die_id * 3 + net_id
                self._die_histories[die_id][net_id] = self.histories[flat_idx]
        
        # 清除原有按钮
        for btn in self.buttons:
            btn.ax.remove()
        self.buttons = []
        
        # 创建网络选择按钮（REQ/RSP/DATA）
        network_btn_positions = [
            (0.01, 0.03, 0.06, 0.04),  # REQ
            (0.08, 0.03, 0.06, 0.04),  # RSP  
            (0.15, 0.03, 0.06, 0.04),  # DATA
        ]
        
        for idx, label in enumerate(["REQ", "RSP", "DATA"]):
            ax_btn = self.fig.add_axes(network_btn_positions[idx])
            btn = Button(ax_btn, label)
            btn.on_clicked(lambda event, i=idx: self._on_select_network_type(i))
            self.buttons.append(btn)
        
        # 创建Die选择按钮
        die_btn_start_x = 0.22
        die_btn_width = 0.05
        die_btn_gap = 0.01
        
        self.die_buttons = []
        for die_id in range(self.num_dies):
            btn_x = die_btn_start_x + die_id * (die_btn_width + die_btn_gap)
            ax_btn = self.fig.add_axes([btn_x, 0.03, die_btn_width, 0.04])
            btn = Button(ax_btn, f"Die{die_id}")
            btn.on_clicked(lambda event, d=die_id: self._on_select_die(d))
            self.die_buttons.append(btn)
        
        # 调整其他按钮位置
        clear_btn_x = die_btn_start_x + self.num_dies * (die_btn_width + die_btn_gap) + 0.02
        self.clear_btn.ax.set_position([clear_btn_x, 0.03, 0.07, 0.04])
        
        # 查找并调整Show Tags按钮位置
        if hasattr(self, 'tags_btn'):
            tags_btn_x = clear_btn_x + 0.08
            self.tags_btn.ax.set_position([tags_btn_x, 0.03, 0.07, 0.04])
        
        # 更新选中的网络索引（转换为单个索引）
        self.selected_network_index = self.current_die * 3 + self.current_network
        
        # 更新显示标题
        self._update_title()
    
    def _update_title(self):
        """更新显示标题"""
        network_names = ["REQ", "RSP", "DATA"]
        network_name = network_names[self.current_network]
        title = f"Die {self.current_die}/{self.num_dies-1} - {network_name} Network"
        self.ax.set_title(title, fontsize=14, fontweight='bold')
    
    def update(self, all_die_networks=None, cycle=None, skip_pause=False):
        """
        重写update方法以支持多Die网络
        
        Args:
            all_die_networks: List[List[Network]] - all_die_networks[die_id][network_type]
            cycle: 当前周期
            skip_pause: 是否跳过暂停
        """
        # 保存所有Die的网络数据
        if all_die_networks is not None:
            self.all_die_networks = all_die_networks
            
            # 选择当前显示的网络
            if (self.current_die < len(all_die_networks) and 
                self.current_network < len(all_die_networks[self.current_die])):
                current_network = all_die_networks[self.current_die][self.current_network]
                
                # 设置networks为包含所有网络的完整列表，但只更新当前选择的网络
                all_networks = []
                for die_networks in all_die_networks:
                    all_networks.extend(die_networks)
                self.networks = all_networks
                
                # 调用父类的update方法
                return super().update(all_networks, cycle, skip_pause)
        
        # 如果没有网络数据，调用父类默认处理
        return super().update(None, cycle, skip_pause)
    
    def _on_key(self, event):
        """重写按键事件处理，增加D2D特定按键功能"""
        key = event.key
        
        # 数字键1-3选择网络类型
        if key in ['1', '2', '3']:
            network_idx = int(key) - 1
            if 0 <= network_idx < 3:
                self.current_network = network_idx
                self.selected_network_index = self.current_die * 3 + self.current_network
                self._update_title()
                self._update_status_display()
                # 刷新显示
                if hasattr(self, 'all_die_networks'):
                    self.update(self.all_die_networks, cycle=self.cycle, skip_pause=True)
                return
        
        # D键切换Die（正向）
        elif key == 'd':
            self._switch_die(forward=True)
            return
        
        # Shift+D键切换Die（反向）
        elif key == 'D':  # Shift+d
            self._switch_die(forward=False)
            return
        
        # 其他按键调用父类处理
        super()._on_key(event)
    
    def _update_status_display(self):
        """重写状态显示，添加Die信息"""
        if self.paused:
            # 暂停状态
            self.status_text.set_color("orange")
            return
        
        # 添加Die信息到状态显示
        network_names = ["REQ", "RSP", "DATA"]
        network_name = network_names[self.current_network]
        status = f"Running... cycle: {self.cycle}\nDie {self.current_die} - {network_name}\nInterval: {self.pause_interval:.2f}"
        color = "green"

        # 更新状态文本
        self.status_text.set_text(status)
        self.status_text.set_color(color)
    
    def _switch_die(self, forward=True):
        """切换Die"""
        if forward:
            self.current_die = (self.current_die + 1) % self.num_dies
        else:
            self.current_die = (self.current_die - 1) % self.num_dies
        
        # 更新选中的网络索引
        self.selected_network_index = self.current_die * 3 + self.current_network
        self._update_title()
        self._update_status_display()
        
        # 刷新显示
        if hasattr(self, 'all_die_networks'):
            self.update(self.all_die_networks, cycle=self.cycle, skip_pause=True)
    
    def _on_select_network_type(self, network_type):
        """通过按钮选择网络类型"""
        self.current_network = network_type
        self.selected_network_index = self.current_die * 3 + self.current_network
        self._update_title()
        self._update_status_display()
        # 刷新显示
        if hasattr(self, 'all_die_networks'):
            self.update(self.all_die_networks, cycle=self.cycle, skip_pause=True)
    
    def _on_select_die(self, die_id):
        """通过按钮选择Die"""
        if 0 <= die_id < self.num_dies:
            self.current_die = die_id
            self.selected_network_index = self.current_die * 3 + self.current_network
            self._update_title()
            self._update_status_display()
            # 刷新显示
            if hasattr(self, 'all_die_networks'):
                self.update(self.all_die_networks, cycle=self.cycle, skip_pause=True)
    
