import heapq
import random
import csv
import matplotlib.pyplot as plt
import math
import matplotlib
import sys
import numpy as np
if sys.platform == "darwin":  # macOS 的系统标识是 'darwin'
    matplotlib.use("macosx")  # 仅在 macOS 上使用该后端


class Tracker:
    def __init__(self, limit):
        self.limit = limit
        self.count = 0

    def can_accept(self):
        return self.count < self.limit

    def inc(self):
        if self.count >= self.limit:
            raise RuntimeError("Exceeded limit")
        self.count += 1

    def dec(self):
        if self.count <= 0:
            raise RuntimeError("Underflow")
        self.count -= 1


class DMA:
    def __init__(self, id, out_limit):
        self.id = id
        self.tracker = Tracker(out_limit)


class DDR:
    def __init__(self, id, out_limit, read_lat, write_lat):
        self.id = id
        self.tracker = Tracker(out_limit)
        self.read_latency = read_lat
        self.write_latency = write_lat


class Event:
    RN_CMD = "RN_CMD"
    SN_RECEIVE_CMD = "SN_RECEIVE_CMD"
    SN_DONE = "SN_PROCESS_DONE"
    RN_DATA = "RN_RECEIVE_DATA"
    RN_ACK = "RN_RECEIVE_ACK"
    RN_SEND = "RN_SEND_DATA"
    SN_DATA = "SN_RECEIVE_DATA"

    def __init__(self, time, type, dma_id=None, ddr_id=None, cmd=None, flit_idx=None, req_id=None):
        self.time = time
        self.type = type
        self.dma_id = dma_id
        self.ddr_id = ddr_id
        self.cmd = cmd
        self.flit_idx = flit_idx
        self.req_id = req_id

    def __lt__(self, other):
        return self.time < other.time


class Request:
    """保存一次 request（命令）全过程的时间点和 flit 时间列表"""

    def __init__(self, req_id, dma_id, ddr_id, cmd, cmd_time):
        self.req_id = req_id
        self.dma_id = dma_id
        self.ddr_id = ddr_id
        self.cmd = cmd
        self.cmd_time = cmd_time
        self.sn_receive_time = None
        self.sn_send_data = []
        self.ack_time = None  # 写请求的 ACK
        self.read_flit_send = []  # SN→RN 发 flit 时间列表
        self.read_flit_recv = []  # RN 收 flit 时间列表
        self.write_flit_send = []  # RN→SN 发 flit 时间列表
        self.write_flit_recv = []  # SN 收 flit 时间列表
        self.bw = None


class NoC_Simulator:
    def __init__(
        self,
        n_dma=1,
        n_ddr=1,
        RN_ostd=4,
        SN_ostd=4,
        ddr_latency=None,
        latency_var=[0,0],
        path_delay=10,
        num_cmds=200,
        read_ratio=1.0,
        burst_length=4,
        ddr_data_bw=None,
        rn_data_bw=None,
        ddr_select_mode="round_robin"  # "round_robin" 或 "random"
    ):
        # DMA 和 SN buffer
        self.RN = [DMA(i, RN_ostd) for i in range(n_dma)]
        self.SN = []
        if ddr_latency and len(ddr_latency) != n_ddr:
            ddr_latency = ddr_latency * n_ddr
        for i in range(n_ddr):
            rl = ddr_latency[i][0] if ddr_latency else 10
            wl = ddr_latency[i][1] if ddr_latency else 10
            self.SN.append(DDR(i, SN_ostd, rl, wl))

        self.latency_var = latency_var
        # 网络参数
        self.path_delay = path_delay
        self.num_cmds = num_cmds
        self.cmds_left = num_cmds
        self.read_ratio = read_ratio
        self.burst_length = burst_length
        # 仿真状态
        self.current_time = 0.0
        self.events = []
        self.next_ddr = 0
        random.seed(0)
        self.requests = {}

        # 发命令/发数据间隔，统一用 ip_gap
        self.ip_gap = 1.0  # ns
        self.flit_size = 128  # bytes
        # DDR 令牌桶初始化
        self.ddr_rate      = ddr_data_bw or float('inf')      # 填充速率 (Bytes/ns)
        self.ddr_tokens    = [self.ddr_rate] * len(self.SN)   # 初始令牌
        self.ddr_last_time = [0.0] * len(self.SN)             # 上次更新时间
        self.sn_next_free  = [0.0] * len(self.SN)             # SN 端口下次可发时间
        # —— DDR 选择策略初始化 ——
        if ddr_select_mode not in ("round_robin", "random"):
            raise ValueError("ddr_select_mode 必须是 'round_robin' 或 'random'")
        self.ddr_select_mode = ddr_select_mode
        self.next_ddr        = 0
        self.sn_proc_next_free = [0.0] * len(self.SN)
        self.sn_cmd_gap = self.ip_gap

        # SN 处理命令开始的下次可用时间
        self.sn_receive_next_free = [0.0] * len(self.SN)
        # 并行：为每个 DMA 单独调度第一次 RN_CMD
        for dma_id in range(len(self.RN)):
            self.schedule(Event(0, Event.RN_CMD, dma_id=dma_id))

        if rn_data_bw:
            per_rn_bw = rn_data_bw / len(self.RN)
            raw_gap = self.flit_size / per_rn_bw
            self.rn_data_gap = max(self.ip_gap, math.ceil(raw_gap))
        else:
            self.rn_data_gap = self.ip_gap

        # 每个 RN 端口的下次可发时间
        self.rn_next_free = [0.0] * len(self.RN)
        # 每个 RN 端口的下次可收时间
        self.rn_recv_next_free = [0.0] * len(self.RN)

    def schedule(self, ev):
        heapq.heappush(self.events, ev)

    def all_dma_free(self):
        return all(d.tracker.count == 0 for d in self.RN)

    def run(self):
        # 第一次发命令
        # self.schedule(Event(0.0, Event.RN_CMD))

        while self.events:
            ev = heapq.heappop(self.events)
            self.current_time = ev.time

            if ev.type == Event.RN_CMD:
                self.on_rn_cmd(ev)
            elif ev.type == Event.SN_RECEIVE_CMD:
                self.handle_sn_receive(ev)
            elif ev.type == Event.SN_DONE:
                self.handle_sn_done(ev)
            elif ev.type == Event.RN_DATA:
                self.handle_rn_data(ev)
            elif ev.type == Event.RN_ACK:
                self.handle_rn_ack(ev)
            elif ev.type == Event.RN_SEND:
                self.handle_rn_send(ev)
            elif ev.type == Event.SN_DATA:
                self.handle_sn_data(ev)

            # 结束条件：所有命令发完且所有 DMA credit 回收
            if self.cmds_left == 0 and self.all_dma_free():
                break

        total_time = self.current_time
        total_bytes = self.num_cmds * 128 * self.burst_length
        bw = total_bytes / total_time
        self.report(bw, total_time)
        return bw

    def on_rn_cmd(self, ev):
        dma = self.RN[ev.dma_id]
        # 只有该 DMA 有 credit 且还有命令未发完
        if self.cmds_left > 0 and dma.tracker.can_accept():
            # 根据模式选择一个 DDR
            if self.ddr_select_mode == "random":
                ddr_id = random.randrange(len(self.SN))
            else:  # round_robin
                ddr_id = self.next_ddr
                self.next_ddr = (self.next_ddr + 1) % len(self.SN)

            ddr = self.SN[ddr_id]
            if ddr.tracker.can_accept():
                cmd = 'R' if random.random() < self.read_ratio else 'W'
                dma.tracker.inc()
                ddr.tracker.inc()
                self.cmds_left -= 1
                req_id = self.num_cmds - self.cmds_left
                req = Request(req_id, dma.id, ddr_id, cmd, ev.time)
                self.requests[req_id] = req

                # 取整到整数纳秒
                t_arr = math.ceil(ev.time + self.path_delay)
                self.schedule(Event(
                    t_arr,
                    Event.SN_RECEIVE_CMD,
                    dma_id=ev.dma_id,
                    ddr_id=ddr_id,
                    cmd=cmd,
                    req_id=req_id
                ))

        # 为同一个 DMA 调度下一次 RN_CMD
        if self.cmds_left > 0:
            t_next = math.ceil(ev.time + self.ip_gap)
            self.schedule(Event(
                t_next,
                Event.RN_CMD,
                dma_id=ev.dma_id
            ))

    # def handle_sn_receive(self, ev):
    #     self.requests[ev.req_id].sn_receive_time = ev.time
    #     ddr = self.SN[ev.ddr_id]
    #     lat = ddr.read_latency if ev.cmd == "R" else ddr.write_latency
    #     self.schedule(Event(ev.time + lat, Event.SN_DONE, dma_id=ev.dma_id, ddr_id=ev.ddr_id, cmd=ev.cmd, req_id=ev.req_id))
    def handle_sn_receive(self, ev):
        """
        SN（DDR）收到命令后：
         - 若已有 SN_ostd credit，可并行处理多条命令
         - 但每条命令的开始处理时间要至少间隔 sn_cmd_gap
        """
        ddr_id = ev.ddr_id
        ddr = self.SN[ddr_id]

        # 1. 确定开始处理时间：>=命令到达时间 & >=上次开始时间+间隔
        t_start = max(ev.time, self.sn_receive_next_free[ddr_id])
        # 2. 更新下次可开始时间
        self.sn_receive_next_free[ddr_id] = t_start + self.sn_cmd_gap

        # 3. 记录开始处理时间
        self.requests[ev.req_id].sn_receive_time = t_start

        # 4. 计算处理完成时间
        latency = ddr.read_latency + np.random.uniform(-self.latency_var[0], self.latency_var[0]) if ev.cmd == "R" else ddr.write_latency + np.random.uniform(-self.latency_var[1], self.latency_var[1])
        t_done = t_start + latency

        # 5. 调度 SN_DONE（向上取整到整数 ns）
        t_done_int = math.ceil(t_done)
        self.schedule(Event(
            t_done_int,
            Event.SN_DONE,
            dma_id=ev.dma_id,
            ddr_id=ddr_id,
            cmd=ev.cmd,
            req_id=ev.req_id
        ))

    
    def handle_sn_done(self, ev):
        r = self.requests[ev.req_id]
        ddr_id = ev.ddr_id
        self.SN[ddr_id].tracker.dec()

        send_times = []
        count = self.burst_length if ev.cmd == 'R' else 1
        for i in range(count):
            t0 = max(ev.time, self.sn_next_free[ddr_id])
            # 累积令牌
            dt = t0 - self.ddr_last_time[ddr_id]
            self.ddr_tokens[ddr_id] = min(
                self.ddr_tokens[ddr_id] + dt * self.ddr_rate,
                self.ddr_rate
            )
            self.ddr_last_time[ddr_id] = t0
            # 不足时等待
            if self.ddr_tokens[ddr_id] < self.flit_size:
                needed = self.flit_size - self.ddr_tokens[ddr_id]
                wait = needed / self.ddr_rate
                t0 += wait
                self.ddr_tokens[ddr_id] = 0.0
                self.ddr_last_time[ddr_id] = t0
            else:
                self.ddr_tokens[ddr_id] -= self.flit_size
            # 向上取整到整数纳秒
            t0 = math.ceil(t0)
            send_times.append(t0)
            arr = t0 + self.path_delay
            ev_type = Event.RN_DATA if ev.cmd == 'R' else Event.RN_ACK
            self.schedule(Event(arr, ev_type,
                                dma_id=ev.dma_id, ddr_id=ddr_id,
                                cmd=ev.cmd,
                                flit_idx=(i if ev.cmd == 'R' else None),
                                req_id=ev.req_id))
            self.sn_next_free[ddr_id] = t0

        r.sn_send_data = send_times

    def handle_rn_data(self, ev):
        # self.requests[ev.req_id].read_flit_recv.append(ev.time)
        # if ev.flit_idx == self.burst_length - 1:
        #     self.RN[ev.dma_id].tracker.dec()
        """收到一个 read-flit，保证同一 RN 口上的 flit 串行接收"""
        r = self.requests[ev.req_id]
        # 两帧最小间隔，整数 ns
        # gap = max(self.ip_gap, self.rn_data_gap)
        gap = self.ip_gap
        # 计算本帧真正的接收时刻：至少要等到上次接收完毕
        recv_t = max(ev.time, self.rn_recv_next_free[ev.dma_id])
        # 记录下来
        r.read_flit_recv.append(recv_t)
        # 更新下次可收时间
        self.rn_recv_next_free[ev.dma_id] = recv_t + gap

        # 最后一帧到达时，才释放 credit
        if ev.flit_idx == self.burst_length - 1:
            self.RN[ev.dma_id].tracker.dec()

    def handle_rn_ack(self, ev):
        self.requests[ev.req_id].ack_time = ev.time
        self.schedule(Event(ev.time, Event.RN_SEND, dma_id=ev.dma_id, ddr_id=ev.ddr_id, cmd="W", req_id=ev.req_id))

    def handle_rn_send(self, ev):
        # for i in range(self.burst_length):
        #     t = ev.time + self.path_delay + i * self.ip_gap
        #     self.requests[ev.req_id].write_flit_send.append(t)
        #     self.schedule(Event(t, Event.SN_DATA, dma_id=ev.dma_id, ddr_id=ev.ddr_id, cmd="W", flit_idx=i, req_id=ev.req_id))
        # 按照 RN 端口带宽限速发送
        r = self.requests[ev.req_id]
        # 本次最早可发时间 = max(当前事件时间, 上次发送完毕时间)
        next_free = max(ev.time, self.rn_next_free[ev.dma_id])
        gap = max(self.ip_gap, self.rn_data_gap)
        for i in range(self.burst_length):
            send_t = next_free
            r.write_flit_send.append(send_t)
            arr = send_t + self.path_delay
            self.schedule(Event(arr, Event.SN_DATA, dma_id=ev.dma_id, ddr_id=ev.ddr_id, cmd="W", flit_idx=i, req_id=ev.req_id))
            next_free += gap
        # 更新此 RN 口的下次可发时间
        self.rn_next_free[ev.dma_id] = next_free

    def handle_sn_data(self, ev):
        self.requests[ev.req_id].write_flit_recv.append(ev.time)
        if ev.flit_idx == self.burst_length - 1:
            self.RN[ev.dma_id].tracker.dec()

    def report(self, bw, total_time):
        print("\n====== SUMMARY ======")
        print(f"Total time   : {total_time:.2f} ns")
        print(f"Bandwidth    : {bw:.2f} GB/s")
        print(f"Total cmdd   : {self.num_cmds}\n")

        completion_times = []
        for r in self.requests.values():
            if r.cmd == "R":
                # 读请求：最后一个 flit 到达的时刻
                last_time = max(r.read_flit_recv)
            else:
                # 写请求：ACK 时刻
                last_time = r.ack_time
            completion_times.append(last_time)

        # 2. 排序
        completion_times.sort()

        # 3. 为每个完成点计算累计字节数
        #    假设读请求数据量 = burst_length * flit_size；写请求的 ACK 不算数据
        bytes_per_req = self.burst_length * self.flit_size
        bytes_cum = [(i + 1) * bytes_per_req for i in range(len(completion_times))]

        # 4. 计算带宽 (Bytes/ns 转 GB/s = Bytes/ns)
        times = completion_times  # 单位 ns
        bws = [bytes_cum[i] / t for i, t in enumerate(times)]

        # 5. 绘图
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(times, bws, drawstyle="steps-post")
        plt.xlabel("Completion Time (ns)")
        plt.ylabel("Bandwidth (GB/s)")
        plt.title("Cumulative Bandwidth vs Completion Time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        with open("Theoretical_Bandwidth.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["req_id", "dma_id", "ddr_id", "cmd", "cmd_time", "sn_receive_time", "sn_send_data", "ack_time", "read_flit_recv", "write_flit_recv"])
            for req_id in sorted(self.requests):
                r = self.requests[req_id]
                writer.writerow(
                    [
                        r.req_id,
                        r.dma_id,
                        r.ddr_id,
                        r.cmd,
                        f"{r.cmd_time:.2f}",
                        f"{r.sn_receive_time:.2f}" if r.sn_receive_time else "",
                        ";".join(f"{t:.2f}" for t in r.sn_send_data),
                        f"{r.ack_time:.2f}" if r.ack_time else "",
                        ";".join(f"{t:.2f}" for t in r.read_flit_recv),
                        ";".join(f"{t:.2f}" for t in r.write_flit_recv),
                    ]
                )
        
        print("=> events written to Theoretical_Bandwidth.csv")


if __name__ == "__main__":
    sim = NoC_Simulator(
        n_dma=4,
        n_ddr=16,
        RN_ostd=128,
        SN_ostd=32,
        ddr_latency=[(155, 16)],
        latency_var = [25, 0],
        path_delay=80,
        num_cmds=128 * 100,
        read_ratio=1,
        burst_length=2,
        ddr_data_bw=76.8,
        rn_data_bw=128,
        # ddr_select_mode='random'
    )
    sim.run()
