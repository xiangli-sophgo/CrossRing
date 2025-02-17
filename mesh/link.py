from __future__ import annotations

from packet import Packet, Response, Request
import numpy as np
from collections import deque


class Link:
    def __init__(
        self,
        src: tuple[int, int] | str,  # src表示Link的起点坐标,形如(x,y)或ip
        dest: tuple[int, int] | str,  # dest表示Link的终点坐标,形如(x,y)或ip
        slice_num: int,  # slice_num表示Link上的slice数量,也就是buffer子列表的数量
        buffer_depth: int = 2,  # buffer_depth表示buffer深度,也就是buffer二子列表的长度
        input_capable: bool = True,  # input_capable表示buffer能否输入的符号
        output_capable: bool = False,  # output_capable表示buffer能否输出的符号
        round_robin: list[Link] = [],  # round_robin表示轮询仲裁的链路列表
        routing_name: str = "XY",
    ):
        self.src = src
        self.dest = dest
        self.slice_num = slice_num

        # buffer表示Link上的buffer,为二维列表,list的元素是Packet类型的数据包
        self.buffer = [deque() for _ in range(slice_num)]  # 使用 deque 作为缓冲区

        self.buffer_depth = buffer_depth
        self.input_capable = input_capable
        self.output_capable = output_capable
        self.round_robin = round_robin
        self.routing_name = routing_name

    def state_update(self):
        """链路状态更新,判断是否可输入和是否可输出"""
        self.input_capable = len(self.buffer[-1]) < self.buffer_depth
        self.output_capable = len(self.buffer[0]) > 0

    def inside_walk(self):
        """让链路内部的数据包往前走"""
        current_num = [-1 for _ in range(self.slice_num)]
        for i in range(self.slice_num):
            current_num[i] = len(self.buffer[i])
        for i in range(1, self.slice_num):
            if current_num[i - 1] == self.buffer_depth or current_num[i] == 0:
                continue
            p = self.buffer[i].popleft()  # 从当前slice取出数据包
            self.buffer[i - 1].append(p)  # 放入前一个slice

    def link2link(self):
        """让每一条链路获得来自它上游链路的输入"""
        if isinstance(self.src, str) or not self.input_capable:
            return
        for input_link in self.round_robin:
            if not input_link.output_capable:
                continue
            if input_link.buffer and self.route(input_link.buffer[0][0]) == self.dest:
                # 输入链路的数据包移动到输出链路
                p = input_link.buffer[0].popleft()
                p.loc = self.dest
                if type(p.loc) is str:
                    assert p.dest == self.src
                self.buffer[-1].append(p)
                # 更新round robin
                self.round_robin.remove(input_link)
                self.round_robin.append(input_link)
                # 更新输入输出状态
                self.input_capable = False
                input_link.output_capable = False
                break

    def route(self, p: Packet | Request | Response) -> tuple[int, int] | str:
        """路由函数,根据routing_name对Packet|Request|Response类型输入做路由"""
        if self.routing_name == "XY":
            return self.xy_routing(p)
        elif self.routing_name == "YX":
            return self.yx_routing(p)
        else:
            raise ValueError(f"Only support XY and YX, but got {self.routing_name}")

    def xy_routing(self, p: Packet | Request | Response) -> tuple[int, int] | str:
        """xy路由,输入为Packet|Request|Response类型的数据包,输出为下一跳的坐标或终点的类型"""
        x = p.dest[0] - p.loc[0]
        y = p.dest[1] - p.loc[1]
        if x != 0:
            x_dir = np.sign(x)
            return p.loc[0] + x_dir, p.loc[1]
        elif y != 0:
            y_dir = np.sign(y)
            return p.loc[0], p.loc[1] + y_dir
        else:
            return p.dest_type

    def yx_routing(self, p: Packet | Request | Response) -> tuple[int, int] | str:
        """yx路由,输入为Packet|Request|Response类型的数据包,输出为下一跳的坐标或终点的类型"""
        x = p.dest[0] - p.loc[0]
        y = p.dest[1] - p.loc[1]
        if y != 0:
            y_dir = np.sign(y)
            return p.loc[0], p.loc[1] + y_dir
        elif x != 0:
            x_dir = np.sign(x)
            return p.loc[0] + x_dir, p.loc[1]
        else:
            return p.dest_type
