import heapq
import random
import csv
import matplotlib.pyplot as plt


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
    RN_ISSUE = "RN_ISSUE"
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

    def __init__(self, req_id, dma_id, ddr_id, cmd, issue_time):
        self.req_id = req_id
        self.dma_id = dma_id
        self.ddr_id = ddr_id
        self.cmd = cmd
        self.issue_time = issue_time
        self.sn_receive_time = None
        self.sn_send_data = None
        self.ack_time = None  # 写请求的 ACK
        self.read_flit_send = []  # SN→RN 发 flit 时间列表
        self.read_flit_recv = []  # RN 收 flit 时间列表
        self.write_flit_send = []  # RN→SN 发 flit 时间列表
        self.write_flit_recv = []  # SN 收 flit 时间列表


class NoC_Simulator:
    def __init__(self, n_dma=1, n_ddr=1, RN_ostd=4, SN_ostd=4, ddr_latency=None, path_delay=10, num_cmds=200, read_ratio=1.0, burst_length=4):
        self.RN = [DMA(i, RN_ostd) for i in range(n_dma)]
        self.SN = []
        for i in range(n_ddr):
            rl = ddr_latency[i][0] if ddr_latency else 10
            wl = ddr_latency[i][1] if ddr_latency else 10
            self.SN.append(DDR(i, SN_ostd, rl, wl))

        self.path_delay = path_delay
        self.num_cmds = num_cmds
        self.cmds_left = num_cmds
        self.read_ratio = read_ratio
        self.burst_length = burst_length

        self.issue_gap = 1.0  # ns
        self.flit_gap = 0.5  # ns

        self.current_time = 0.0
        self.events = []
        self.next_ddr = 0
        random.seed(0)

        # 按 request_id 保存 Request 对象
        self.requests = {}

    def schedule(self, ev):
        heapq.heappush(self.events, ev)

    def run(self):
        # schedule 第一次 issue
        self.schedule(Event(0.0, Event.RN_ISSUE))

        while self.events:
            ev = heapq.heappop(self.events)
            self.current_time = ev.time

            if ev.type == Event.RN_ISSUE:
                self.on_rn_issue(ev)
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

    def all_dma_free(self):
        return all(d.tracker.count == 0 for d in self.RN)

    def on_rn_issue(self, ev):
        if self.cmds_left > 0:
            for dma in self.RN:
                ddr_id = (dma.id + self.next_ddr) % len(self.SN)
                self.next_ddr += 1
                ddr = self.SN[ddr_id]
                if dma.tracker.can_accept() and ddr.tracker.can_accept():
                    is_rd = random.random() < self.read_ratio
                    cmd = "R" if is_rd else "W"
                    dma.tracker.inc()
                    ddr.tracker.inc()
                    self.cmds_left -= 1
                    req_id = self.num_cmds - self.cmds_left
                    # 新建 Request
                    req = Request(req_id, dma.id, ddr_id, cmd, ev.time)
                    self.requests[req_id] = req

                    # print(f"[{ev.time:6.2f}ns] ISSUE   Req#{req_id:4d}  " f"DMA={dma.id}→DDR={ddr_id} CMD={cmd}")

                    # 发到 SN
                    t_arr = ev.time + self.path_delay
                    self.schedule(Event(t_arr, Event.SN_RECEIVE_CMD, dma_id=dma.id, ddr_id=ddr_id, cmd=cmd, req_id=req_id))
                    break

        # schedule 下一次 issue
        if self.cmds_left > 0:
            self.schedule(Event(ev.time + self.issue_gap, Event.RN_ISSUE))

    def handle_sn_receive(self, ev):
        # 记录 SN 收到命令时间
        self.requests[ev.req_id].sn_receive_time = ev.time

        ddr = self.SN[ev.ddr_id]
        lat = ddr.read_latency if ev.cmd == "R" else ddr.write_latency
        self.schedule(Event(ev.time + lat, Event.SN_DONE, dma_id=ev.dma_id, ddr_id=ev.ddr_id, cmd=ev.cmd, req_id=ev.req_id))

    def handle_sn_done(self, ev):
        # 记录 SN 完成处理时间
        self.requests[ev.req_id].sn_send_data = ev.time
        ddr = self.SN[ev.ddr_id]
        ddr.tracker.dec()

        if ev.cmd == "R":
            for i in range(self.burst_length):
                t = ev.time + self.path_delay + i * self.flit_gap
                # print(f"  [SN→RN] Req#{ev.req_id:4d} SND R-flit#{i} @ {t:.2f}ns")
                self.requests[ev.req_id].read_flit_send.append(t)
                self.schedule(Event(t, Event.RN_DATA, dma_id=ev.dma_id, ddr_id=ev.ddr_id, cmd="R", flit_idx=i, req_id=ev.req_id))
        else:
            # 写命令 ACK
            t = ev.time + self.path_delay
            # print(f"  [SN→RN] Req#{ev.req_id:4d} SND W-ACK  @ {t:.2f}ns")
            self.schedule(Event(t, Event.RN_ACK, dma_id=ev.dma_id, ddr_id=ev.ddr_id, cmd="W", req_id=ev.req_id))

    def handle_rn_data(self, ev):
        # flit 到达 RN
        # print(f"  [RN]       Req#{ev.req_id:4d} RCV R-flit#{ev.flit_idx} @ {ev.time:.2f}ns")
        self.requests[ev.req_id].read_flit_recv.append(ev.time)
        # 最后一个 flit 回收一次 credit
        if ev.flit_idx == self.burst_length - 1:
            self.RN[ev.dma_id].tracker.dec()

    def handle_rn_ack(self, ev):
        # 记录 ACK 时间
        self.requests[ev.req_id].ack_time = ev.time
        # 开始写数据
        self.schedule(Event(ev.time, Event.RN_SEND, dma_id=ev.dma_id, ddr_id=ev.ddr_id, cmd="W", req_id=ev.req_id))

    def handle_rn_send(self, ev):
        for i in range(self.burst_length):
            t = ev.time + self.path_delay + i * self.flit_gap
            # print(f"  [RN→SN]   Req#{ev.req_id:4d} SND W-flit#{i} @ {t:.2f}ns")
            self.requests[ev.req_id].write_flit_send.append(t)
            self.schedule(Event(t, Event.SN_DATA, dma_id=ev.dma_id, ddr_id=ev.ddr_id, cmd="W", flit_idx=i, req_id=ev.req_id))

    def handle_sn_data(self, ev):
        # print(f"  [SN]       Req#{ev.req_id:4d} RCV W-flit#{ev.flit_idx} @ {ev.time:.2f}ns")
        self.requests[ev.req_id].write_flit_recv.append(ev.time)
        if ev.flit_idx == self.burst_length - 1:
            self.RN[ev.dma_id].tracker.dec()

    def report(self, bw, total_time):
        print("\n====== SUMMARY ======")
        print(f"Total time   : {total_time:.2f} ns")
        print(f"Bandwidth    : {bw:.2f} GB/s")
        print(f"Total issued : {self.num_cmds}\n")

        # # 绘制 Issue timeline
        # times = [req.issue_time for req in self.requests.values()]
        # issued = [req.req_id for req in self.requests.values()]
        # plt.figure(figsize=(8, 4))
        # plt.plot(times, issued, "-o")
        # plt.xlabel("Time (ns)")
        # plt.ylabel("Request ID Issued")
        # plt.title("Issue Timeline @1GHz, Link@2GHz")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

        # 新增：计算每条 Request 的完成时间和累计带宽
        compl_times = []
        bw_list = []
        for req_id in sorted(self.requests):
            r = self.requests[req_id]
            # 计算完成时间：读请求取最后一个 read_flit_recv，写请求取最后一个 write_flit_recv
            if r.cmd == "R":
                finish = r.read_flit_recv[-1]
            else:
                finish = r.write_flit_recv[-1]
            compl_times.append(finish)
            # 带宽(GB/s)：(请求编号 * 128字节 * burst_length)除以时间(ns)，再换算到GB/s
            # 1 byte/ns = 1e-9 GB/s
            bytes_total = req_id * 128 * self.burst_length
            bw_i = bytes_total / finish  # GB/s
            r.bw = bw_i
            bw_list.append(bw_i)

        # 绘制 完成时间 vs 累计带宽
        plt.figure(figsize=(8, 4))
        plt.plot(compl_times, bw_list, "-o", color="C1")
        plt.xlabel("Completion Time (ns)")
        plt.ylabel("Cumulative Bandwidth (GB/s)")
        plt.title("Bandwidth Growth vs Request Completion")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # （可选）将数据输出到 CSV
        import csv

        # 导出 CSV
        with open("Theoretical_Bandwidth.csv", "w", newline="") as f:
            writer = csv.writer(f)
            # header
            writer.writerow(["req_id", "dma_id", "ddr_id", "cmd", "issue_time", "sn_receive_time", "sn_send_data", "ack_time", "read_flit_recv", "write_flit_recv", "BW"])
            for req_id in sorted(self.requests):
                r = self.requests[req_id]
                writer.writerow(
                    [
                        r.req_id,
                        r.dma_id,
                        r.ddr_id,
                        r.cmd,
                        f"{r.issue_time:.2f}",
                        f"{r.sn_receive_time:.2f}" if r.sn_receive_time else "",
                        f"{r.sn_send_data:.2f}" if r.sn_send_data else "",
                        f"{r.ack_time:.2f}" if r.ack_time else "",
                        ";".join(f"{t:.2f}" for t in r.read_flit_recv),
                        ";".join(f"{t:.2f}" for t in r.write_flit_recv),
                        f"{r.bw}",
                    ]
                )
        print("=> events written to Theoretical_Bandwidth.csv")


if __name__ == "__main__":
    sim = NoC_Simulator(n_dma=2, n_ddr=1, RN_ostd=16, SN_ostd=32, ddr_latency=[(155, 0)] * 4, path_delay=40, num_cmds=1000, read_ratio=1, burst_length=2)
    sim.run()
