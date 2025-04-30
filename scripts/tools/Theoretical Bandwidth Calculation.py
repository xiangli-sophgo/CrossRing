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
    def __init__(self, id, ostd_rd, ostd_wr):
        self.id = id
        self.tracker_rd = Tracker(ostd_rd)
        self.tracker_wr = Tracker(ostd_wr)


class DDR:
    def __init__(self, id, ostd_rd, ostd_wr, read_lat, write_lat):
        self.id = id
        self.tracker_rd = Tracker(ostd_rd)
        self.tracker_wr = Tracker(ostd_wr)
        self.read_latency = read_lat
        self.write_latency = write_lat


class Event:
    RN_CMD = "RN_CMD"
    SN_RECEIVE_CMD = "SN_RECEIVE_CMD"
    SN_DONE = "SN_PROCESS_DONE"
    RN_DATA = "RN_RECEIVE_DATA"
    SN_RELEASE_CREDIT = "SN_RELEASE_CREDIT"  # SN 端口释放 credit 的延时事件
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
    def __init__(self, req_id, dma_id, ddr_id, cmd, cmd_time):
        self.req_id = req_id
        self.dma_id = dma_id
        self.ddr_id = ddr_id
        self.cmd = cmd
        self.cmd_time = cmd_time
        self.sn_receive_time = None
        self.sn_send_data = []
        self.rn_send_data = []
        self.ack_time = None
        self.read_flit_recv = []
        self.write_flit_recv = []
        self.bw = None


class NoC_Simulator:
    def __init__(
        self,
        n_dma=1,
        n_ddr=1,
        RN_ostd_rd=4,
        RN_ostd_wr=4,
        SN_ostd_rd=4,
        SN_ostd_wr=4,
        ddr_latency=None,
        latency_var=[0, 0],
        path_delay=10,
        num_cmds=200,
        read_ratio=1.0,
        burst_length=4,
        ddr_data_bw=None,
        release_latency=40.0,  # 写请求 SN 释放 credit 的延时 (ns)
        ddr_select_mode="round_robin",  # "round_robin" 或 "random"
    ):
        # DMA 和 SN buffer
        self.RN = [DMA(i, RN_ostd_rd, RN_ostd_wr) for i in range(n_dma)]
        self.SN = []
        if ddr_latency and len(ddr_latency) != n_ddr:
            ddr_latency = ddr_latency * n_ddr
        self.SN = []
        for i in range(n_ddr):
            rl = ddr_latency[i][0] if ddr_latency else 10
            wl = ddr_latency[i][1] if ddr_latency else 10
            self.SN.append(DDR(i, SN_ostd_rd, SN_ostd_wr, rl, wl))

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
        self.ddr_rate = ddr_data_bw or float("inf")  # 填充速率 (Bytes/ns)
        self.ddr_tokens = [self.ddr_rate] * len(self.SN)  # 初始令牌
        self.ddr_last_time = [0.0] * len(self.SN)  # 上次更新时间
        self.sn_next_free = [0.0] * len(self.SN)  # SN 端口下次可发时间
        # —— DDR 选择策略初始化 ——
        if ddr_select_mode not in ("round_robin", "random"):
            raise ValueError("ddr_select_mode 必须是 'round_robin' 或 'random'")
        self.ddr_select_mode = ddr_select_mode
        self.next_ddr = 0
        self.sn_proc_next_free = [0.0] * len(self.SN)
        self.sn_cmd_gap = self.ip_gap

        # SN 处理命令开始的下次可用时间
        self.sn_receive_next_free = [0.0] * len(self.SN)
        self.release_latency = release_latency

        # 每个 RN 端口的下次可发时间
        self.rn_next_free = [0.0] * len(self.RN)
        # 每个 RN 端口的下次可收时间
        self.rn_recv_next_free = [0.0] * len(self.RN)
        # 并行：为每个 DMA 单独调度第一次 RN_CMD
        for dma_id in range(len(self.RN)):
            self.schedule(Event(0, Event.RN_CMD, dma_id=dma_id))

    def schedule(self, ev):
        heapq.heappush(self.events, ev)

    def all_dma_free(self):
        # 所有 DMA 的读写 credit 都归零
        return all(d.tracker_rd.count == 0 and d.tracker_wr.count == 0 for d in self.RN)

    def run(self):
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
            elif ev.type == Event.SN_RELEASE_CREDIT:
                # 延迟释放 SN 写 credit
                if ev.cmd == "R":
                    self.SN[ev.ddr_id].tracker_rd.dec()
                else:
                    self.SN[ev.ddr_id].tracker_wr.dec()

            # 结束条件
            if self.cmds_left == 0 and self.all_dma_free():
                break

        total_time = self.current_time
        total_bytes = self.num_cmds * self.burst_length * 128
        bw = total_bytes / total_time
        print(f"SIM Done: time={total_time:.2f} ns, bw={bw:.2f} GB/s")
        self.report(bw, total_time)
        return bw

    def on_rn_cmd(self, ev):
        dma = self.RN[ev.dma_id]
        # 选 DDR
        if self.ddr_select_mode == "random":
            ddr_id = random.randrange(len(self.SN))
        else:
            ddr_id = self.next_ddr
            self.next_ddr = (self.next_ddr + 1) % len(self.SN)
        ddr = self.SN[ddr_id]

        if self.cmds_left > 0:
            # 决定是读还是写
            cmd = "R" if random.random() < self.read_ratio else "W"
            # 分别检查读写 credit
            tr_dma = dma.tracker_rd if cmd == "R" else dma.tracker_wr
            tr_sn = ddr.tracker_rd if cmd == "R" else ddr.tracker_wr
            if tr_dma.can_accept() and tr_sn.can_accept():
                tr_dma.inc()
                tr_sn.inc()
                self.cmds_left -= 1
                req_id = self.num_cmds - self.cmds_left
                self.requests[req_id] = Request(req_id, ev.dma_id, ddr_id, cmd, ev.time)
                t_arr = math.ceil(ev.time + self.path_delay)
                self.schedule(Event(t_arr, Event.SN_RECEIVE_CMD, dma_id=ev.dma_id, ddr_id=ddr_id, cmd=cmd, req_id=req_id))
        # 再给同一个 DMA 调度下一条命令
        if self.cmds_left > 0:
            t_next = math.ceil(ev.time + 1.0)
            self.schedule(Event(t_next, Event.RN_CMD, dma_id=ev.dma_id))

    def handle_sn_receive(self, ev):
        ddr_id = ev.ddr_id
        t_start = max(ev.time, self.sn_receive_next_free[ddr_id])
        self.sn_receive_next_free[ddr_id] = t_start + self.sn_cmd_gap
        self.requests[ev.req_id].sn_receive_time = t_start

        ddr = self.SN[ddr_id]
        if ev.cmd == "R":
            base_lat = ddr.read_latency
            var = self.latency_var[0]
        else:
            base_lat = ddr.write_latency
            var = self.latency_var[1]
        latency = base_lat + np.random.uniform(-var, var)
        t_done = math.ceil(t_start + latency)
        self.schedule(Event(t_done, Event.SN_DONE, dma_id=ev.dma_id, ddr_id=ddr_id, cmd=ev.cmd, req_id=ev.req_id))

    def handle_sn_done(self, ev):
        r = self.requests[ev.req_id]
        ddr_id = ev.ddr_id
        ddr = self.SN[ddr_id]

        # SN 端口释放 credit：读立即，写延迟
        if ev.cmd == "R":
            ddr.tracker_rd.dec()
        else:
            # 延迟 release_latency 后再 dec
            self.schedule(Event(ev.time + self.release_latency, Event.SN_RELEASE_CREDIT, ddr_id=ddr_id, cmd="W"))

        # 下面发数据或 ACK
        send_times = []
        cnt = self.burst_length if ev.cmd == "R" else 1
        for i in range(cnt):
            t0 = max(ev.time, self.sn_next_free[ddr_id])
            # token bucket refill
            if ev.cmd == "R":
                dt = t0 - self.ddr_last_time[ddr_id]
                self.ddr_tokens[ddr_id] = min(self.ddr_tokens[ddr_id] + dt * self.ddr_rate, self.ddr_rate)
                self.ddr_last_time[ddr_id] = t0
                if self.ddr_tokens[ddr_id] < 128:
                    wait = (128 - self.ddr_tokens[ddr_id]) / self.ddr_rate
                    t0 += wait
                    self.ddr_last_time[ddr_id] = t0
                    self.ddr_tokens[ddr_id] = 0.0
                else:
                    self.ddr_tokens[ddr_id] -= 128

            t_send = math.ceil(t0)
            send_times.append(t_send)
            ev_type = Event.RN_DATA if ev.cmd == "R" else Event.RN_ACK
            self.schedule(Event(t_send + self.path_delay, ev_type, dma_id=ev.dma_id, ddr_id=ddr_id, cmd=ev.cmd, flit_idx=(i if ev.cmd == "R" else None), req_id=ev.req_id))
            self.sn_next_free[ddr_id] = t_send

        r.sn_send_data = send_times

    def handle_rn_data(self, ev):
        # 读 payload 到达
        req = self.requests[ev.req_id]
        req.read_flit_recv.append(ev.time)
        # 最后一个 flit 到达后才释放 RN 读 credit
        if ev.flit_idx == self.burst_length - 1:
            self.RN[ev.dma_id].tracker_rd.dec()

    def handle_rn_ack(self, ev):
        # 写请求收到 ACK，释放读命令端 credit
        self.requests[ev.req_id].ack_time = ev.time
        # 同时 RN 需要发写数据
        self.schedule(Event(ev.time, Event.RN_SEND, dma_id=ev.dma_id, ddr_id=ev.ddr_id, cmd="W", req_id=ev.req_id))

    # def handle_rn_send(self, ev):
    #     # RN→SN 写数据 flit
    #     for i in range(self.burst_length):
    #         t = ev.time + self.path_delay + i * 1.0
    #         self.schedule(Event(t, Event.SN_DATA, dma_id=ev.dma_id, ddr_id=ev.ddr_id, cmd="W", flit_idx=i, req_id=ev.req_id))

    def handle_rn_send(self, ev):
        """
        RN→SN 写数据 flit，顺串行发送：
        - 每个 DMA port 用 rn_next_free[dma_id] 来保证 flit 之间至少间隔 ip_gap
        - flit 发送后马上 schedule 到 SN_DATA，并累加 path_delay
        """
        dma_id = ev.dma_id
        ddr_id = ev.ddr_id
        req = self.requests[ev.req_id]

        # t0 表示本次可以开始发下一个 flit 的时间
        t0 = ev.time
        for i in range(self.burst_length):
            # 等到上一次 flit 发完毕
            t0 = max(t0, self.rn_next_free[dma_id])
            # 计算发送到 NoC 上的时间点
            send_on_nic = math.ceil(t0)
            req.rn_send_data.append(send_on_nic)
            # schedule 到达 SN 的事件
            self.schedule(Event(send_on_nic + self.path_delay, Event.SN_DATA, dma_id=dma_id, ddr_id=ddr_id, cmd="W", flit_idx=i, req_id=ev.req_id))

            # 更新本 DMA port 的下次可发时间
            self.rn_next_free[dma_id] = t0 + self.ip_gap
            # 同时 t0 向前推进 ip_gap，保证下一个 flit 至少间隔 ip_gap
            t0 += self.ip_gap

    def handle_sn_data(self, ev):
        # SN 收写数据也受带宽限制
        ddr_id = ev.ddr_id
        t0 = max(ev.time, self.sn_next_free[ddr_id])
        if ev.cmd == "W":
            # RN→SN 写数据才消耗 token
            dt = t0 - self.ddr_last_time[ddr_id]
            self.ddr_tokens[ddr_id] = min(self.ddr_tokens[ddr_id] + dt * self.ddr_rate, self.ddr_rate)
            self.ddr_last_time[ddr_id] = t0
            if self.ddr_tokens[ddr_id] < self.flit_size:
                need = self.flit_size - self.ddr_tokens[ddr_id]
                wait = need / self.ddr_rate
                t0 += wait
                self.ddr_tokens[ddr_id] = 0.0
                self.ddr_last_time[ddr_id] = t0
            else:
                self.ddr_tokens[ddr_id] -= self.flit_size

        t0 = math.ceil(t0)
        self.requests[ev.req_id].write_flit_recv.append(t0)
        self.sn_next_free[ddr_id] = t0
        if ev.flit_idx == self.burst_length - 1:
            self.RN[ev.dma_id].tracker_wr.dec()

    def mean_mid_to_3quarter(self, lst):
        """
        计算 lst 从中间位置（floor(n/2)）到 3/4 位置（floor(3n/4)）之间元素的平均值。
        下标采用左闭右开区间 [start, end)。
        如果区间内没有元素，返回 None。
        """
        n = len(lst)
        if n == 0:
            return None

        # 起始位置：floor(n/2)
        start = n // 2
        # 结束位置：floor(3*n/4)
        end = (3 * n) // 4

        # 取子列表
        sub = lst[start:end]

        # 防止除以 0
        if not sub:
            return None

        # 计算平均值
        return sum(sub) / len(sub)

    def report(self, bw, total_time):

        completion_times = []
        for r in self.requests.values():
            if r.cmd == "R":
                # 读请求：最后一个 flit 到达的时刻
                last_time = max(r.read_flit_recv)
            else:
                # 写请求：
                # last_time = r.ack_time
                last_time = max(r.rn_send_data)
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

        print("\n====== SUMMARY ======")
        print(f"Total time   : {total_time:.2f} ns")
        # print(f"Bandwidth    : {max(bws):.2f} GB/s")
        print(f"Bandwidth    : {self.mean_mid_to_3quarter(bws):.2f} GB/s")
        print(f"Total cmdd   : {self.num_cmds}\n")

        # 5. 绘图
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(times, bws, drawstyle="steps-post")
        plt.xlabel("Completion Time (ns)")
        plt.ylabel("Bandwidth (GB/s)")
        # plt.title("Cumulative Bandwidth vs Completion Time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        with open("Theoretical_Bandwidth.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["req_id", "dma_id", "ddr_id", "cmd", "cmd_time", "sn_receive_time", "sn_send_data", "rn_send_data", "ack_time", "read_flit_recv", "write_flit_recv"])
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
                        ";".join(f"{t:.2f}" for t in r.rn_send_data),
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
        RN_ostd_rd=128,
        RN_ostd_wr=64,
        SN_ostd_rd=64,
        SN_ostd_wr=64,
        ddr_latency=[(155, 0)],
        latency_var=(0, 0),
        path_delay=20,
        num_cmds=128 * 100,
        read_ratio=0,
        burst_length=2,
        ddr_data_bw=76.8 / 4,
        release_latency=40.0,
    )
    sim.run()
