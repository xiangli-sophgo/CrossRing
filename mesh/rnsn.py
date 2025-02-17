from packet import Packet, Request, Response
from link import Link
from collections import deque


class RN:
    def __init__(
        self,
        coord: tuple[int, int],  # coord表示RN所挂节点的坐标
        name: str,  # name表示RN的类型,包括cdma和gdma
        r_tracker: deque[Request] = None,  # r_tracker用于存放读请求
        r_pointer: int = 0,  # r_pointer指向的是下一个发送的请求的编号
        r_tracker_retry: int = 0,  # 值为False表示没有retry的
        rdb_output_task_num: int = 0,  # 表示rdb在弹出数据包时,剩下的任务个数
        w_tracker: deque[Request] = None,  # w_tracker用于存放写请求
        w_pointer: int = 0,  # w_pointer指向的是下一个发送的写请求的编号
        w_tracker_retry: int = 0,  # 值为False表示没有retry的
        enter_mesh_req_queue: deque[Request] = None,  # 表示上网络的请求队列
        enter_mesh_dat_queue: deque[Packet] = None,  # 表示上网络的数据队列
        config: dict = None,
    ):
        self.coord = coord
        self.name = name

        self.r_tracker = r_tracker if r_tracker is not None else deque()  # 使用 deque
        self.r_pointer = r_pointer
        self.r_tracker_retry = r_tracker_retry
        self.rdb_output_task_num = rdb_output_task_num

        self.w_tracker = w_tracker if w_tracker is not None else deque()  # 使用 deque
        self.w_pointer = w_pointer
        self.w_tracker_retry = w_tracker_retry

        self.enter_mesh_req_queue = enter_mesh_req_queue if enter_mesh_req_queue is not None else deque()  # 使用 deque
        self.enter_mesh_dat_queue = enter_mesh_dat_queue if enter_mesh_dat_queue is not None else deque()  # 使用 deque

        self.config = config if config is not None else {}

        self.interleave = self.config.get("interleave", 4)

        self.rn_r_tracker_outstanding = self.config.get("rn_r_tracker_outstanding", 64)
        self.rn_w_tracker_outstanding = self.config.get("rn_w_tracker_outstanding", 64)

        self.rdb_credit = self.rn_r_tracker_outstanding * self.interleave  # 表示rdb的剩余空位数
        self.wdb_credit = self.rn_w_tracker_outstanding * self.interleave  # 表示wdb的剩余空位数

    def rn2noc(self, req_link: Link, dat_link: Link):
        """处理从 RN 到 NoC 的 R/W 请求和数据传输。"""
        # queue to req_link
        if self.enter_mesh_req_queue and req_link.input_capable:
            req = self.enter_mesh_req_queue.popleft()  # 从 deque 中弹出请求
            req_link.buffer[-1].append(req)
            req_link.input_capable = False

        # queue to dat_link
        if self.enter_mesh_dat_queue and dat_link.input_capable:
            dat = self.enter_mesh_dat_queue.popleft()  # 从 deque 中弹出数据包
            dat_link.buffer[-1].append(dat)
            dat_link.input_capable = False

        # rn.r_tracker to queue
        if self.rdb_credit >= self.interleave:
            if self.r_tracker_retry > 0:
                self.r_tracker_retry -= 1
                for i in range(self.r_pointer):
                    req = self.r_tracker[i]
                    if req.waiting and req.retry == "Grant":
                        req.retry = "Complete"
                        self.rdb_credit -= req.flit_num
                        self.enter_mesh_req_queue.append(req)
                        break
            else:
                if self.r_pointer < self.rn_r_tracker_outstanding and self.r_pointer < len(self.r_tracker):
                    req = self.r_tracker[self.r_pointer]
                    self.rdb_credit -= req.flit_num
                    req.waiting = True
                    self.enter_mesh_req_queue.append(req)
                    self.r_pointer += 1

        # rn.w_tracker to queue
        if self.w_tracker_retry > 0:
            self.w_tracker_retry -= 1
            for i in range(self.w_pointer):
                req = self.w_tracker[i]
                if req.waiting and req.retry == "Grant":
                    req.retry = "Complete"
                    self.enter_mesh_req_queue.append(req)
                    break
        else:
            if self.w_pointer < self.rn_w_tracker_outstanding and self.w_pointer < len(self.w_tracker):
                req = self.w_tracker[self.w_pointer]
                req.waiting = True
                self.enter_mesh_req_queue.append(req)
                self.w_pointer += 1

    def noc2rn(self, rsp_link: Link, dat_link: Link, cycle):
        """处理从 NoC 到 RN 的响应。"""
        # rsp_link to rn
        if rsp_link.output_capable:
            rsp: Response = rsp_link.buffer[0].popleft()
            rsp_link.output_capable = False

            if rsp.rsp_type == "Read_Retry":
                # Read_Retry对应的响应
                if rsp.retry_type == "RetryAck":
                    rsp.req_belong.retry = "Ack"
                    self.rdb_credit += rsp.req_belong.flit_num
                else:
                    assert rsp.retry_type == "RetryGrant"
                    rsp.req_belong.retry = "Grant"
                    self.r_tracker_retry += 1
            elif rsp.rsp_type == "Write_RSP":
                # 写操作对应的响应
                self.w_tracker.remove(rsp.req_belong)
                self.w_pointer -= 1
                for _ in range(rsp.req_belong.flit_num):
                    self.enter_mesh_dat_queue.append(
                        Packet(
                            src=rsp.dest,
                            loc=rsp.dest,
                            dest=rsp.src,
                            src_type=rsp.dest_type,
                            dest_type=rsp.src_type,
                            cycle_start=cycle,
                            req_belong=rsp.req_belong,
                        )
                    )
            elif rsp.rsp_type == "Write_Retry":
                self.w_tracker_retry += 1
                assert rsp.retry_type == "RetryGrant"
                rsp.req_belong.retry = "Grant"

        # dat_link to rn,读回来数据
        if dat_link.output_capable:
            pac = dat_link.buffer[0].popleft()
            dat_link.output_capable = False
            pac.req_belong.receive_packet_num += 1

    def rn2dma(self, read_req, cycle):
        """rn到dma,rdb收齐data弹出data,弹出最后一笔data时释放tracker中的req"""
        # 如果还有输出任务在进行，减少任务计数并增加信用
        if self.rdb_output_task_num > 0:
            # print(self.rdb_output_task_num)
            self.rdb_output_task_num -= 1
            self.rdb_credit += 1
            return True

        # 遍历请求跟踪器，查找已完成的请求
        for i in range(self.r_pointer):
            req = self.r_tracker[i]
            if req.receive_packet_num == req.flit_num:
                self.rdb_output_task_num = req.flit_num
                flit_num = req.flit_num

                # 这里为什么要加 flit_num
                req.cycle_end = cycle  # + flit_num

                self.r_tracker.remove(req)
                read_req.append(req)

                self.r_pointer -= 1
                self.rdb_output_task_num -= 1
                self.rdb_credit += 1
                return True
        return False


class SN:
    def __init__(
        self,
        coord: tuple[int, int],  # coord表示RN所挂节点的坐标
        name: str,  # name表示RN的类型,包括cdma和gdma
        r_tracker: deque[Request] = None,  # r_tracker用于存放读请求
        r_retry_grant: deque[Response] = None,  # 用于记录Retry Grant
        w_tracker: deque[Request] = None,  # w_tracker用于存放写请求
        w_retry_grant: deque[Response] = None,  # 用于记录Retry Grant
        w_pointer: int = 0,  # w_pointer指向的是下一个发送的写请求的编号
        wdb_output_task_num: int = 0,  # 表示wdb在弹出数据包时, 剩下的任务个数
        enter_mesh_rsp_que: deque[Response] = None,  # 表示上网络的响应队列
        enter_mesh_dat_que: deque[Packet] = None,  # 表示上网络的dat队列
        config: dict = None,  # 配置信息
    ):
        self.coord = coord
        self.name = name

        self.r_tracker = r_tracker if r_tracker is not None else deque()  # 使用 deque
        self.r_retry_grant = r_retry_grant if r_retry_grant is not None else deque()  # 使用 deque

        self.w_tracker = w_tracker if w_tracker is not None else deque()  # 使用 deque
        self.w_retry_grant = w_retry_grant if w_retry_grant is not None else deque()  # 使用 deque

        self.w_pointer = w_pointer
        self.wdb_output_task_num = wdb_output_task_num

        self.enter_mesh_rsp_queue = enter_mesh_rsp_que if enter_mesh_rsp_que is not None else deque()  # 使用 deque
        self.enter_mesh_dat_queue = enter_mesh_dat_que if enter_mesh_dat_que is not None else deque()  # 使用 deque

        self.config = config if config is not None else {}
        self.interleave = self.config.get("interleave", 4)  # burst length 一笔req对应的data的数量

        # r_tracker_credit表示r_tracker剩余空位数
        self.r_tracker_credit = self.sn_r_tracker_outstanding = self.config.get("sn_r_tracker_outstanding", 64)
        # w_tracker_credit表示w_tracker剩余空位数
        self.w_tracker_credit = self.sn_w_tracker_outstanding = self.config.get("sn_w_tracker_outstanding", 64)
        # wdb_credit表示wdb的剩余空位数
        self.wdb_credit = self.sn_w_tracker_outstanding * self.interleave

        self.sn_r_data_buffer_size = self.sn_r_tracker_outstanding * self.interleave
        self.ddr_latency = self.config.get("ddr_latency", 150)

    def sn2noc(self, rsp_link: Link, dat_link: Link, cycle):
        # rsp_queue to rsp_link
        if self.enter_mesh_rsp_queue and rsp_link.input_capable:
            rsp: Response = self.enter_mesh_rsp_queue.popleft()
            rsp_link.buffer[-1].append(rsp)
            rsp_link.input_capable = False

        # dat_queue to dat_link
        if self.enter_mesh_dat_queue and dat_link.input_capable:
            if cycle >= self.enter_mesh_dat_queue[0].cycle_start:
                dat = self.enter_mesh_dat_queue.popleft()
                dat_link.buffer[-1].append(dat)
                dat_link.input_capable = False
                if dat.islast:
                    self.r_tracker_credit += 1
                    if self.r_retry_grant:
                        rsp = self.r_retry_grant.popleft()
                        self.enter_mesh_rsp_queue.append(rsp)

        # tracker to dat_queue
        # 先读
        if self.r_tracker:
            req = self.r_tracker.popleft()
            for _ in range(req.flit_num - 1):
                self.enter_mesh_dat_queue.append(
                    Packet(
                        src=req.dest,
                        loc=req.dest,
                        dest=req.src,
                        src_type=req.dest_type,
                        dest_type=req.src_type,
                        cycle_start=cycle + self.ddr_latency,
                        req_belong=req,
                    )
                )
            self.enter_mesh_dat_queue.append(
                Packet(
                    src=req.dest,
                    loc=req.dest,
                    dest=req.src,
                    src_type=req.dest_type,
                    dest_type=req.src_type,
                    cycle_start=cycle + self.ddr_latency,
                    req_belong=req,
                    islast=True,
                )
            )

        if self.w_pointer < len(self.w_tracker):
            req = self.w_tracker[self.w_pointer]
            self.w_pointer += 1

            rsp = Response(
                src=req.dest,
                loc=req.dest,
                dest=req.src,
                src_type=req.dest_type,
                dest_type=req.src_type,
                cycle_start=cycle,
                rsp_type="Write_RSP",
                req_belong=req,
            )
            self.enter_mesh_rsp_queue.append(rsp)

    def noc2sn(self, req_link: Link, dat_link: Link, cycle):
        """将请求和数据从 NOC 接收并处理"""
        # req_link to sn
        if req_link.output_capable:
            req = req_link.buffer[0].popleft()
            assert req.dest == self.coord
            req_link.output_capable = False

            # 处理读请求
            if req.req_type == "R":
                if req.retry == "Complete" or self.r_tracker_credit > 0:
                    if req.retry != "Complete":
                        self.r_tracker_credit -= 1
                    self.r_tracker.append(req)
                else:
                    self.r_tracker_credit -= 1
                    req.loc = req.src
                    self.enter_mesh_rsp_queue.append(
                        Response(
                            src=req.dest,
                            loc=req.dest,
                            dest=req.src,
                            src_type=req.dest_type,
                            dest_type=req.src_type,
                            cycle_start=cycle,
                            rsp_type="Read_Retry",
                            retry_type="RetryAck",
                            req_belong=req,
                        )
                    )
                    self.r_retry_grant.append(
                        Response(
                            src=req.dest,
                            loc=req.dest,
                            dest=req.src,
                            src_type=req.dest_type,
                            dest_type=req.src_type,
                            cycle_start=cycle,
                            rsp_type="Read_Retry",
                            retry_type="RetryGrant",
                            req_belong=req,
                        )
                    )

            # 处理写请求
            else:
                assert req.req_type == "W"
                if req.retry == "Complete" or self.w_tracker_credit > 0:
                    if req.retry != "Complete":
                        self.w_tracker_credit -= 1
                    self.w_tracker.append(req)
                else:
                    self.w_tracker_credit -= 1
                    req.loc = req.src
                    self.w_retry_grant.append(
                        Response(
                            src=req.dest,
                            loc=req.dest,
                            dest=req.src,
                            src_type=req.dest_type,
                            dest_type=req.src_type,
                            cycle_start=cycle,
                            rsp_type="Write_Retry",
                            retry_type="RetryGrant",
                            req_belong=req,
                        )
                    )

        # dat_link to sn
        if dat_link.output_capable:
            dat = dat_link.buffer[0].popleft()
            self.wdb_credit -= 1
            dat.req_belong.receive_packet_num += 1
            dat_link.output_capable = False

    def sn2ddr(self, write_req, cycle):
        """将写请求从 SN 发送到 DDR"""
        if self.wdb_output_task_num:
            self.wdb_output_task_num -= 1
            self.wdb_credit += 1
            return True

        for req in self.w_tracker:
            if req.receive_packet_num == req.flit_num:
                self.wdb_output_task_num = req.flit_num
                self.w_tracker.remove(req)

                req.cycle_end = cycle + req.flit_num

                write_req.append(req)
                self.w_tracker_credit += 1
                self.w_pointer -= 1

                if self.w_retry_grant:
                    req = self.w_retry_grant.popleft()
                    self.enter_mesh_rsp_queue.append(req)

                self.wdb_output_task_num -= 1
                self.wdb_credit += 1
                return True
        return False
