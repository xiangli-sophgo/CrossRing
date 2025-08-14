"""
双通道网络链路状态可视化器
继承自NetworkLinkVisualizer，支持4个网络切换（REQ、RSP、CH0、CH1）
"""

from .Link_State_Visualizer import NetworkLinkVisualizer
from collections import deque
from matplotlib.widgets import Button


class DualChannelNetworkLinkVisualizer(NetworkLinkVisualizer):
    """双通道网络链路可视化器，支持4个网络切换"""
    
    def __init__(self, config, network):
        # 直接使用传入的network，假设它是完全初始化的
        super().__init__(network)
        
        # 重新初始化以支持4个网络
        self._reinitialize_for_dual_channel()
    
    def _reinitialize_for_dual_channel(self):
        """重新初始化以支持双通道（4个网络）"""
        # 清除原有的3个网络历史缓冲区，创建4个新的
        self.histories = [deque(maxlen=20) for _ in range(4)]
        
        # 清除原有按钮
        for btn in self.buttons:
            btn.ax.remove()
        self.buttons = []
        
        # 创建新的4个网络选择按钮
        btn_positions = [
            (0.01, 0.03, 0.06, 0.04),  # REQ
            (0.08, 0.03, 0.06, 0.04),  # RSP  
            (0.15, 0.03, 0.06, 0.04),  # CH0
            (0.22, 0.03, 0.06, 0.04),  # CH1
        ]
        
        for idx, label in enumerate(["REQ", "RSP", "CH0", "CH1"]):
            ax_btn = self.fig.add_axes(btn_positions[idx])
            btn = Button(ax_btn, label)
            btn.on_clicked(lambda event, i=idx: self._on_select_network(i))
            self.buttons.append(btn)
        
        # 调整其他按钮位置（Clear HL等）向右移动以腾出空间
        self.clear_btn.ax.set_position([0.29, 0.03, 0.07, 0.04])
        
        # 查找并调整Show Tags按钮位置
        if hasattr(self, 'tags_btn'):
            self.tags_btn.ax.set_position([0.37, 0.03, 0.07, 0.04])
        
        # 默认选择第一个数据通道（索引2，即CH0）
        self.selected_network_index = 2
    
    def update(self, networks=None, cycle=None, skip_pause=False):
        """重写update方法以支持4个网络"""
        # 确保传入的是4个网络列表
        if networks is not None:
            if len(networks) == 4:
                self.networks = networks
            else:
                # 如果传入的网络数量不是4个，抛出警告但继续执行
                print(f"Warning: Expected 4 networks, got {len(networks)}")
                self.networks = networks
        
        # 调用父类的update方法，父类会处理所有的渲染逻辑
        return super().update(networks, cycle, skip_pause)
    
    def _on_key(self, event):
        """重写按键事件处理，增加数字键选择网络功能"""
        key = event.key
        
        # 数字键1-4选择对应网络
        if key in ['1', '2', '3', '4']:
            network_idx = int(key) - 1
            if 0 <= network_idx < len(self.histories):
                self.selected_network_index = network_idx
                # 刷新显示
                if self.networks is not None:
                    self.update(self.networks, cycle=self.cycle, skip_pause=True)
                else:
                    self.update(None, cycle=self.cycle, skip_pause=True)
                return
        
        # 其他按键调用父类处理
        super()._on_key(event)