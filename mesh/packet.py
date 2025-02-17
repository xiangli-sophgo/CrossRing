from __future__ import annotations
from typing import Optional, Tuple


class Request:
    def __init__(
        self,
        src: Tuple[int, int],  # src表示Packet的起点,形式为(x,y),对应横纵坐标
        loc: Tuple[int, int],  # loc表示Packet的当前位置,形式为(x,y),对应横纵坐标
        dest: Tuple[int, int],  # dest表示Packet的起点,形式为(x,y),对应横纵坐标
        src_type: str,  # src_type表示src的类型,目前为'ddr','cdma','gdma'中的一种
        dest_type: str,  # dest_type表示dest的类型,目前为'ddr','cdma','gdma'中的一种
        cycle_start: int,  # cycle_start记录产生时间
        cycle_end: Optional[int] = None,  # cycle_end记录到达时间
        req_type: Optional[str] = None,  # W or R
        waiting: bool = False,  # 值为True表示正在等待响应
        retry: str = "No",  # No表示正在等待响应,RetryGrant表示收到Grant但没有进入queue,Complete表示进入queue
        receive_packet_num: int = 0,  # 表示收到的数据的数量
        flit_num: Optional[int] = None,
    ):
        self.src = src
        self.loc = loc
        self.dest = dest
        self.src_type = src_type
        self.dest_type = dest_type
        self.cycle_start = cycle_start
        self.cycle_end = cycle_end
        self.req_type = req_type
        self.waiting = waiting
        self.retry = retry
        self.receive_packet_num = receive_packet_num
        self.flit_num = flit_num

    def __str__(self):
        return (
            # f"REQ: src={self.src},loc={self.loc},dest={self.dest},type={self.src_type}->{self.dest_type},"
            # f"req_type={self.req_type},cycle_start={self.cycle_start},cycle_end={self.cycle_end},"
            # f"flit_num={self.flit_num}"
            f"R"
        )

    # def _


class Response:
    def __init__(
        self,
        src: Tuple[int, int],  # src表示Packet的起点,形式为(x,y),对应横纵坐标
        loc: Tuple[int, int],  # loc表示Packet的当前位置,形式为(x,y),对应横纵坐标
        dest: Tuple[int, int],  # dest表示Packet的起点,形式为(x,y),对应横纵坐标
        src_type: str,  # src_type表示src的类型,目前为'ddr','cdma','gdma'中的一种
        dest_type: str,  # dest_type表示dest的类型,目前为'ddr','cdma','gdma'中的一种
        cycle_start: int,  # cycle_start记录产生时间
        cycle_end: Optional[int] = None,  # cycle_end记录到达时间
        rsp_type: Optional[str] = None,  # 'Read_Retry' or 'Write_RSP' or 'Write_Retry'
        retry_type: Optional[str] = None,  # 'RetryAck' or 'RetryGrant'
        req_belong: Optional[Request] = None,  # 属于哪个请求的响应
    ):
        self.src = src
        self.loc = loc
        self.dest = dest
        self.src_type = src_type
        self.dest_type = dest_type
        self.cycle_start = cycle_start
        self.cycle_end = cycle_end
        self.rsp_type = rsp_type
        self.retry_type = retry_type
        self.req_belong = req_belong

    def __str__(self):
        return (
            f"RSP:src={self.src},loc={self.loc},dest={self.dest}, ype={self.src_type}->{self.dest_type},"
            f"cycle_start={self.cycle_start},cycle_end={self.cycle_end}"
        )


class Packet:
    def __init__(
        self,
        src: Tuple[int, int],  # src表示Packet的起点,形式为(x,y),对应横纵坐标
        loc: Tuple[int, int],  # loc表示Packet的当前位置,形式为(x,y),对应横纵坐标
        dest: Tuple[int, int],  # dest表示Packet的起点,形式为(x,y),对应横纵坐标
        src_type: str,  # src_type表示src的类型,目前为'ddr','cdma','gdma'中的一种
        dest_type: str,  # dest_type表示dest的类型,目前为'ddr','cdma','gdma'中的一种
        cycle_start: int,  # cycle_start记录产生时间
        req_belong: Optional[Request] = None,  # data类型时,记录是属于哪个req和rsp
        cycle_end: Optional[int] = None,  # cycle_start记录结束时间
        islast: bool = False,  # 是否是最后一个flit
    ):
        self.src = src
        self.loc = loc
        self.dest = dest
        self.src_type = src_type
        self.dest_type = dest_type
        self.cycle_start = cycle_start
        self.cycle_end = cycle_end
        self.req_belong = req_belong
        self.islast = islast

    def __str__(self):
        return (
            f"Flit:src={self.src},loc={self.loc},dest={self.dest},type={self.src_type}->{self.dest_type},"
            f"cycle_start={self.cycle_start},cycle_end={self.cycle_end},islast={self.islast}"
        )
