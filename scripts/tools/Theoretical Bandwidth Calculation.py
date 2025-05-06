#!/usr/bin/env python3
# ---------------------------------------------------------------
# 多端口 NoC 仿真器 (支持 GDMA/SDMA 与 DDR/L2M 互访、带宽上限、带宽曲线)
#    – 保留原始代码主体，仅最小增量扩展
# ---------------------------------------------------------------
import heapq
import random
import csv
import math
import sys
from collections import namedtuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if sys.platform == "darwin":  # macOS
    matplotlib.use("macosx")


# ===============================================================
#                    简   单   公   共   工   具
# ===============================================================
class Tracker:
    """简单的 outstanding credit 计数器"""

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
    """RN 侧 DMA 端口"""

    def __init__(self, id_, ostd_rd, ostd_wr):
        self.id = id_
        self.tracker_rd = Tracker(ostd_rd)
        self.tracker_wr = Tracker(ostd_wr)


class DDR:  # 这里同时可代表 DDR / L2M
    def __init__(self, id_, ostd_rd, ostd_wr, read_lat, write_lat):
        self.id = id_
        self.tracker_rd = Tracker(ostd_rd)
        self.tracker_wr = Tracker(ostd_wr)
        self.read_latency = read_lat
        self.write_latency = write_lat


class Event:
    """离散事件"""

    RN_CMD = "RN_CMD"
    SN_RECEIVE_CMD = "SN_RECEIVE_CMD"
    SN_DONE = "SN_PROCESS_DONE"
    RN_DATA = "RN_RECEIVE_DATA"
    SN_RELEASE_CREDIT = "SN_RELEASE_CREDIT"
    RN_ACK = "RN_RECEIVE_ACK"
    RN_SEND = "RN_SEND_DATA"
    SN_DATA = "SN_RECEIVE_DATA"

    def __init__(self, time, typ, dma_id=None, port_id=None, cmd=None, flit_idx=None, req_id=None):
        self.time = time
        self.type = typ
        self.dma_id = dma_id
        self.port_id = port_id
        self.cmd = cmd  # 'R' | 'W'
        self.flit_idx = flit_idx
        self.req_id = req_id

    def __lt__(self, other):
        return self.time < other.time


class Request:
    """收集一次 DMA 请求的完整时间线"""

    def __init__(self, req_id, dma_id, port_id, cmd, cmd_time):
        self.req_id = req_id
        self.dma_id = dma_id
        self.port_id = port_id
        self.cmd = cmd
        self.cmd_time = cmd_time
        self.sn_receive_time = None
        self.sn_send_data = []  # SN->RN 或 SN->ACK
        self.rn_send_data = []  # RN->SN 写数据
        self.ack_time = None
        self.read_flit_recv = []  # RN 端接收 payload
        self.write_flit_recv = []  # SN 端接收写数据


# ------------- 简单“枚举” -------------
DMAType = namedtuple("DMAType", ["GDMA", "SDMA"])(0, 1)
MemType = namedtuple("MemType", ["DDR", "L2M"])(0, 1)
# -------------------------------------


# ===============================================================
#                        核     心     仿     真
# ===============================================================
class NoC_Simulator:
    def __init__(
        self,
        n_gdma=1,
        n_sdma=1,
        n_ddr=1,
        n_l2m=1,
        # RN outstanding credits, 分别给 GDMA/SDMA
        RN_ostd_rd_gdma=4,
        RN_ostd_wr_gdma=4,
        RN_ostd_rd_sdma=4,
        RN_ostd_wr_sdma=4,
        # SN outstanding credits, 分别给 DDR/L2M
        SN_ostd_rd_ddr=4,
        SN_ostd_wr_ddr=4,
        SN_ostd_rd_l2m=4,
        SN_ostd_wr_l2m=4,
        # 延迟 / 带宽配置
        ddr_latency=None,  # [(read,write), ...]
        l2m_latency=None,  # 同上
        latency_var=(0, 0),  # (read_var, write_var)
        path_delay=10,
        num_cmds=200,
        burst_length=4,
        # token bucket 速率 (Byte/ns)
        ddr_data_bw=None,
        l2m_data_bw=None,
        release_latency=40.0,
        # 请求类型概率
        request_mix=None,  # {("gdma","l2m","R"):0.5, ...}
        seed=0,
        save_req_trace=0,
    ):
        random.seed(seed)
        # ---------- RN / DMA 侧 ----------
        self.dma_types = []
        self.RN = []
        for _ in range(n_gdma):
            self.RN.append(DMA(len(self.RN), ostd_rd=RN_ostd_rd_gdma, ostd_wr=RN_ostd_wr_gdma))
            self.dma_types.append(DMAType.GDMA)
        for _ in range(n_sdma):
            self.RN.append(DMA(len(self.RN), ostd_rd=RN_ostd_rd_sdma, ostd_wr=RN_ostd_wr_sdma))
            self.dma_types.append(DMAType.SDMA)

        # ---------- SN / 存储侧 ----------
        self.SN = []
        self.mem_types = []  # 与 SN 索引对应
        self.mem_num = {"ddr": n_ddr, "l2m": n_l2m}

        # DDR
        if ddr_latency and len(ddr_latency) != n_ddr:
            ddr_latency = ddr_latency * n_ddr
        for i in range(n_ddr):
            rl = ddr_latency[i][0] if ddr_latency else 10
            wl = ddr_latency[i][1] if ddr_latency else 10
            self.SN.append(DDR(len(self.SN), ostd_rd=SN_ostd_rd_ddr, ostd_wr=SN_ostd_wr_ddr, read_lat=rl, write_lat=wl))
            self.mem_types.append(MemType.DDR)

        # L2M
        if l2m_latency and len(l2m_latency) != n_l2m:
            l2m_latency = l2m_latency * n_l2m
        for i in range(n_l2m):
            rl = l2m_latency[i][0] if l2m_latency else 5
            wl = l2m_latency[i][1] if l2m_latency else 5
            self.SN.append(DDR(len(self.SN), ostd_rd=SN_ostd_rd_l2m, ostd_wr=SN_ostd_wr_l2m, read_lat=rl, write_lat=wl))
            self.mem_types.append(MemType.L2M)

        # ---------- token bucket ----------
        self.ddr_rate = ddr_data_bw or float("inf")
        self.l2m_rate = l2m_data_bw or float("inf")
        self.tokens = [self.ddr_rate if t == MemType.DDR else self.l2m_rate for t in self.mem_types]
        self.last_token_time = [0.0] * len(self.SN)

        # ---------- 其余状态 ----------
        self.latency_var = latency_var
        self.path_delay = path_delay
        self.num_cmds = num_cmds
        self.cmds_left = num_cmds
        self.burst_length = burst_length
        self.current_time = 0.0
        self.events = []
        self.next_mem = {"ddr": 0, "l2m": 0}
        self.ip_gap = 1.0
        self.flit_size = 128
        self.sn_next_free = [0.0] * len(self.SN)
        self.sn_receive_next_free = [0.0] * len(self.SN)
        self.rn_next_free = [0.0] * len(self.RN)
        self.release_latency = release_latency
        self.requests = {}
        self.request_mix = request_mix
        self.save_req_trace = save_req_trace
        if request_mix:
            # 先计算原始权重和
            total = sum(request_mix.values())
            if total <= 0:
                raise ValueError("request_mix 中的权重之和必须大于 0")

            # 如果和不为 1，就缩放到和为 1
            if abs(total - 1.0) > 1e-6:
                # 你可以打印警告或日志
                # print(f"warning: request_mix sum is {total:.6f}, will normalize it to 1.0")
                # 也可以直接静默 normalize
                for k in request_mix:
                    request_mix[k] /= total

            # 构造累加数组，方便后面做抽样
            self.mix_keys = []
            self.mix_accu = []
            accu = 0.0
            for k, p in request_mix.items():
                accu += p
                self.mix_keys.append(k)
                self.mix_accu.append(accu)
            # 理论上最后一个 accu 应该是 1.0（或非常接近）
            if abs(self.mix_accu[-1] - 1.0) > 1e-6:
                # 万一数值误差太大，再报错
                raise RuntimeError("normalize 后的 request_mix 累积和不为 1，请检查输入")

        # 初始事件：只对 mix 中出现过的 DMA Type 发第一条命令
        if self.request_mix:
            # 哪些 dma type 在 mix 里出现过
            dma_types_in_mix = set(k[0] for k in self.mix_keys)  # {'gdma', 'sdma', …}
            for dma_id, dt in enumerate(self.dma_types):
                dtype = "gdma" if dt == DMAType.GDMA else "sdma"
                if dtype in dma_types_in_mix:
                    self.schedule(Event(0.0, Event.RN_CMD, dma_id=dma_id))

    # ---------------- util ----------------
    def schedule(self, ev: Event):
        heapq.heappush(self.events, ev)

    def all_dma_free(self):
        return all(d.tracker_rd.count == 0 and d.tracker_wr.count == 0 for d in self.RN)

    # ===========================================================
    #                 主          循          环
    # ===========================================================
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
                mem = self.SN[ev.port_id]
                if ev.cmd == "R":
                    mem.tracker_rd.dec()
                else:
                    mem.tracker_wr.dec()

            if self.cmds_left == 0 and self.all_dma_free():
                break

        # total_time = self.current_time
        # total_bytes = self.num_cmds * self.burst_length * self.flit_size
        # bw = total_bytes / total_time
        # print(f"SIM DONE: {total_time:.2f} ns, Avg BW = {bw:.2f} GB/s")
        self.report()

    # ===========================================================
    #                     事     件     处     理
    # ===========================================================
    # ---------- 选择一次请求 ----------
    def choose_req(self, dma_id):
        dtype = self.dma_types[dma_id]

        if self.request_mix:
            # 按概率抽样
            while True:
                r = random.random()
                idx = next(i for i, v in enumerate(self.mix_accu) if r <= v)
                dma_str, mem_str, cmd = self.mix_keys[idx]
                need = "gdma" if dtype == DMAType.GDMA else "sdma"
                if dma_str != need:
                    continue
                port_candidates = [i for i, t in enumerate(self.mem_types) if (mem_str == "ddr" and t == MemType.DDR) or (mem_str == "l2m" and t == MemType.L2M)]
                return random.choice(port_candidates), cmd
                mem_id = self.next_mem[mem_str]
                self.next_mem[mem_str] = (self.next_mem[mem_str] + 1) % self.mem_num[mem_str]
                return mem_id, cmd
        else:
            raise ValueError(self.request_mix)

    # ---------- RN_CMD ----------
    def on_rn_cmd(self, ev: Event):
        if self.cmds_left == 0:
            return
        port_id, cmd = self.choose_req(ev.dma_id)
        dma = self.RN[ev.dma_id]
        mem = self.SN[port_id]

        tr_dma = dma.tracker_rd if cmd == "R" else dma.tracker_wr
        tr_mem = mem.tracker_rd if cmd == "R" else mem.tracker_wr

        if tr_dma.can_accept() and tr_mem.can_accept():
            tr_dma.inc()
            tr_mem.inc()
            self.cmds_left -= 1
            req_id = self.num_cmds - self.cmds_left
            self.requests[req_id] = Request(req_id, ev.dma_id, port_id, cmd, self.current_time)
            self.schedule(Event(self.current_time + self.path_delay, Event.SN_RECEIVE_CMD, dma_id=ev.dma_id, port_id=port_id, cmd=cmd, req_id=req_id))
        # 同 dma 继续
        if self.cmds_left:
            self.schedule(Event(self.current_time + 1.0, Event.RN_CMD, dma_id=ev.dma_id))

    # ---------- SN 接收到 CMD ----------
    def handle_sn_receive(self, ev: Event):
        port = ev.port_id
        t_start = max(self.current_time, self.sn_receive_next_free[port])
        self.sn_receive_next_free[port] = t_start + self.ip_gap
        self.requests[ev.req_id].sn_receive_time = t_start

        mem = self.SN[port]
        base_lat = mem.read_latency if ev.cmd == "R" else mem.write_latency
        var = self.latency_var[0] if ev.cmd == "R" else self.latency_var[1]
        latency = base_lat + random.uniform(-var, var)
        t_done = math.ceil(t_start + latency)
        self.schedule(Event(t_done, Event.SN_DONE, dma_id=ev.dma_id, port_id=port, cmd=ev.cmd, req_id=ev.req_id))

    # ---------- SN 执行完 ----------
    def handle_sn_done(self, ev: Event):
        req = self.requests[ev.req_id]
        port = ev.port_id
        mem = self.SN[port]

        # credit 释放
        if ev.cmd == "R":
            mem.tracker_rd.dec()
        else:
            self.schedule(Event(ev.time + self.release_latency, Event.SN_RELEASE_CREDIT, port_id=port, cmd="W"))

        # SN 侧发 payload(读) 或 ACK(写)
        send_cnt = self.burst_length if ev.cmd == "R" else 1
        for i in range(send_cnt):
            t0 = max(ev.time, self.sn_next_free[port])
            if ev.cmd == "R":  # 读数据出 SN
                t0 = self._consume_token(port, self.flit_size, t0)
            t_send = math.ceil(t0)
            self.sn_next_free[port] = t_send
            req.sn_send_data.append(t_send)
            etype = Event.RN_DATA if ev.cmd == "R" else Event.RN_ACK
            self.schedule(Event(t_send + self.path_delay, etype, dma_id=ev.dma_id, port_id=port, cmd=ev.cmd, flit_idx=(i if ev.cmd == "R" else None), req_id=ev.req_id))

    # ---------- RN 收到读 payload ----------
    def handle_rn_data(self, ev: Event):
        req = self.requests[ev.req_id]
        req.read_flit_recv.append(ev.time)
        if ev.flit_idx == self.burst_length - 1:
            self.RN[ev.dma_id].tracker_rd.dec()

    # ---------- RN 收到 ACK ----------
    def handle_rn_ack(self, ev: Event):
        self.requests[ev.req_id].ack_time = ev.time
        self.schedule(Event(ev.time, Event.RN_SEND, dma_id=ev.dma_id, port_id=ev.port_id, cmd="W", req_id=ev.req_id))

    # ---------- RN 发送写数据 ----------
    def handle_rn_send(self, ev: Event):
        req = self.requests[ev.req_id]
        dma_id = ev.dma_id
        port = ev.port_id
        t0 = ev.time
        for i in range(self.burst_length):
            t0 = max(t0, self.rn_next_free[dma_id])
            send_time = math.ceil(t0)
            req.rn_send_data.append(send_time)
            self.rn_next_free[dma_id] = t0 + self.ip_gap
            t0 += self.ip_gap
            self.schedule(Event(send_time + self.path_delay, Event.SN_DATA, dma_id=dma_id, port_id=port, cmd="W", flit_idx=i, req_id=ev.req_id))

    # ---------- SN 接收写数据 ----------
    def handle_sn_data(self, ev: Event):
        port = ev.port_id
        t0 = max(ev.time, self.sn_next_free[port])
        if ev.cmd == "W":
            t0 = self._consume_token(port, self.flit_size, t0)
        t_done = math.ceil(t0)
        self.sn_next_free[port] = t_done
        self.requests[ev.req_id].write_flit_recv.append(t_done)
        if ev.flit_idx == self.burst_length - 1:
            self.RN[ev.dma_id].tracker_wr.dec()

    # ---------- token bucket 工具 ----------
    def _consume_token(self, port, size, now):
        rate = self.ddr_rate if self.mem_types[port] == MemType.DDR else self.l2m_rate
        self.tokens[port] = min(self.tokens[port] + (now - self.last_token_time[port]) * rate, rate)
        self.last_token_time[port] = now

        if self.tokens[port] < size:
            wait = (size - self.tokens[port]) / rate
            now += wait
            self.tokens[port] = 0.0
            self.last_token_time[port] = now
        else:
            self.tokens[port] -= size
        return now

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
        if len(sub) == 0:
            return None

        # 计算平均值
        return sum(sub) / len(sub)

    # ===========================================================
    #                       统   计   报   告
    # ===========================================================
    def report(self):
        groups = {
            ("gdma", "R"): [],
            ("gdma", "W"): [],
            ("sdma", "R"): [],
            ("sdma", "W"): [],
        }
        byte_per_req = self.burst_length * self.flit_size

        for r in self.requests.values():
            dm = "gdma" if self.dma_types[r.dma_id] == DMAType.GDMA else "sdma"
            if r.cmd == "R":
                t_done = max(r.read_flit_recv)
            else:
                t_done = max(r.rn_send_data)
            groups[(dm, r.cmd)].append((t_done, byte_per_req))

        plt.figure(figsize=(8, 5))

        # for k, data in groups.items():
        #     if not data:
        #         continue
        #     data.sort(key=lambda x: x[0])
        #     ts = np.array([t for t, _ in data])
        #     cum = np.cumsum([b for _, b in data])
        #     bw = cum / ts

        #     # 计算前 90% 的索引位置
        #     end = int(len(ts) * 0.9)

        #     # 只画前 90% 的点
        #     plt.plot(ts[:end] / 1000, bw[:end], drawstyle="steps-post", label=f"{k[0].upper()}-{k[1]} Bandwidth")

        #     print(f"{k[0].upper()}-{k[1]} Bandwidth: " f"{bw[-1]:.2f} GB/s")
        total_bw = 0
        for k, data in groups.items():
            if not data:
                continue
            data.sort(key=lambda x: x[0])
            ts = np.array([t for t, _ in data])
            cum = np.cumsum([b for _, b in data])
            bw = cum / ts

            # 找到 ts 的 90% 分位时间点
            t85 = np.percentile(ts, 85)
            mask = ts <= t85

            plt.plot(ts[mask] / 1000, bw[mask], drawstyle="steps-post", label=f"{k[0].upper()}-{k[1]}")

            print(f"{k[0].upper()}-{k[1]} Bandwidth: " f"{bw[-1]:.2f} GB/s")
            total_bw += bw[-1]
        print(f"Total Bandwidth: {total_bw:.2f} GB/s")

        plt.xlabel("Time (us)")
        plt.ylabel("Bandwidth (GB/s)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # ------ CSV -------------
        if self.save_req_trace:
            with open("sim_trace.csv", "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["req", "dma", "type", "mem_port", "mem_type", "cmd", "cmd_t", "sn_recv_t", "sn_sends", "rn_sends", "read_recv", "write_recv"])
                for r in sorted(self.requests.values(), key=lambda x: x.req_id):
                    wr.writerow(
                        [
                            r.req_id,
                            r.dma_id,
                            "GDMA" if self.dma_types[r.dma_id] == DMAType.GDMA else "SDMA",
                            r.port_id,
                            "DDR" if self.mem_types[r.port_id] == MemType.DDR else "L2M",
                            r.cmd,
                            f"{r.cmd_time:.2f}",
                            f"{r.sn_receive_time:.2f}" if r.sn_receive_time else "",
                            ";".join(map(str, r.sn_send_data)),
                            ";".join(map(str, r.rn_send_data)),
                            ";".join(map(str, r.read_flit_recv)),
                            ";".join(map(str, r.write_flit_recv)),
                        ]
                    )
            print("=> detailed trace written to sim_trace.csv")


# ===============================================================
#                           DEMO
# ===============================================================
if __name__ == "__main__":
    # 示例：80% GDMA 读 L2M，20% SDMA 写 DDR
    mix = {
        # ("gdma", "l2m", "R"): 1,
        ("sdma", "ddr", "W"): 1,
        ("sdma", "ddr", "R"): 1,
    }

    sim = NoC_Simulator(
        n_gdma=4,
        n_sdma=4,
        n_ddr=4,
        n_l2m=4,
        RN_ostd_rd_gdma=128,
        RN_ostd_wr_gdma=32,
        RN_ostd_rd_sdma=128,
        RN_ostd_wr_sdma=32,
        SN_ostd_rd_ddr=64,
        SN_ostd_wr_ddr=64,
        SN_ostd_rd_l2m=64,
        SN_ostd_wr_l2m=64,
        ddr_latency=[(155, 16)],
        l2m_latency=[(10, 10)],
        path_delay=20,
        num_cmds=128 * 200,
        burst_length=2,
        ddr_data_bw=76.8,  # GB/s
        l2m_data_bw=128,  # GB/s
        release_latency=40.0,
        request_mix=mix,
        save_req_trace=0,
    )
    sim.run()
