#!/usr/bin/env python3
# ---------------------------------------------------------------
# 多端口 NoC 仿真器 (GDMA/SDMA ↔ DDR/L2M) – 带读写差值限制版
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
    """简单 outstanding credit 计数器"""

    def __init__(self, limit):
        self.limit = limit
        self.count = 0

    def can_accept(self):
        return self.count < self.limit

    def inc(self):
        if not self.can_accept():
            raise RuntimeError("Exceeded limit")
        self.count += 1

    def dec(self):
        if self.count == 0:
            raise RuntimeError("Underflow")
        self.count -= 1


class DMA:
    """RN 侧 DMA 端口"""

    def __init__(self, id_, ostd_rd, ostd_wr):
        self.id = id_
        self.tracker_rd = Tracker(ostd_rd)
        self.tracker_wr = Tracker(ostd_wr)


class DDR:  # 同时可代表 L2M
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
        self.sn_send_data = []  # SN→RN payload / ACK
        self.rn_send_data = []  # RN→SN 写数据
        self.ack_time = None
        self.read_flit_recv = []  # RN 收到读数据
        self.write_flit_recv = []  # SN 收到写数据


# ----------- 简易“枚举” ------------
DMAType = namedtuple("DMAType", ["GDMA", "SDMA"])(0, 1)
MemType = namedtuple("MemType", ["DDR", "L2M"])(0, 1)
# -----------------------------------


# ===============================================================
#                        核     心     仿     真
# ===============================================================
class NoC_Simulator:
    def __init__(
        self,
        # ------ DMA & MEM 规模 ------
        n_gdma=1,
        n_sdma=1,
        n_ddr=1,
        n_l2m=1,
        # ------ RN outstanding ------
        RN_ostd_rd_gdma=128,
        RN_ostd_wr_gdma=32,
        RN_ostd_rd_sdma=128,
        RN_ostd_wr_sdma=32,
        # ------ SN outstanding ------
        SN_ostd_rd_ddr=64,
        SN_ostd_wr_ddr=64,
        SN_ostd_rd_l2m=64,
        SN_ostd_wr_l2m=64,
        # ------ 延迟 / 带宽 ------
        ddr_latency=None,
        l2m_latency=None,
        latency_var=(0, 0),
        path_delay=10,
        num_cmds=200,
        burst_length=4,
        ddr_data_bw=None,  # Byte/ns
        l2m_data_bw=None,
        release_latency=40.0,
        # ------ 流量混合 ------
        request_mix=None,  # {("sdma","l2m","R"):1, ("sdma","ddr","W"):1}
        # ------ 读写差值限制 ------
        gap_rules=None,  # 列表，每条规则见下
        # seed=0,
        save_req_trace=0,
        choose_req_type=0,  # 0 for round, 1 for random
    ):
        # random.seed(seed)

        # ------------ DMA 实例 ------------
        self.dma_types = []
        self.RN = []
        for _ in range(n_gdma):
            self.RN.append(DMA(len(self.RN), RN_ostd_rd_gdma, RN_ostd_wr_gdma))
            self.dma_types.append(DMAType.GDMA)
        for _ in range(n_sdma):
            self.RN.append(DMA(len(self.RN), RN_ostd_rd_sdma, RN_ostd_wr_sdma))
            self.dma_types.append(DMAType.SDMA)

        # ------------ SN 端口 ------------
        self.SN = []
        self.mem_types = []  # 索引 → MemType
        self.mem_num = {"ddr": n_ddr, "l2m": n_l2m}

        if ddr_latency and len(ddr_latency) != n_ddr:
            ddr_latency *= n_ddr
        for i in range(n_ddr):
            rl, wl = ddr_latency[i] if ddr_latency else (10, 10)
            self.SN.append(DDR(len(self.SN), SN_ostd_rd_ddr, SN_ostd_wr_ddr, rl, wl))
            self.mem_types.append(MemType.DDR)

        if l2m_latency and len(l2m_latency) != n_l2m:
            l2m_latency *= n_l2m
        for i in range(n_l2m):
            rl, wl = l2m_latency[i] if l2m_latency else (5, 5)
            self.SN.append(DDR(len(self.SN), SN_ostd_rd_l2m, SN_ostd_wr_l2m, rl, wl))
            self.mem_types.append(MemType.L2M)

        # ------------ token‑bucket ------------
        self.ddr_rate = ddr_data_bw or float("inf")
        if isinstance(l2m_data_bw, (list, tuple)) and len(l2m_data_bw) == 2:
            self.l2m_rate_rd, self.l2m_rate_wr = l2m_data_bw
        else:
            self.l2m_rate_rd = self.l2m_rate_wr = l2m_data_bw or float("inf")

        # ------ 初始化 token-bucket 状态 ------
        # 对每个 SN 端口：
        #   DDR: tokens/shared bucket
        #   L2M: tokens_rd, tokens_wr 两套独立 bucket
        self.ddr_tokens = []
        self.ddr_last_time = []
        self.l2m_tokens_r = []
        self.l2m_last_r_time = []
        self.l2m_tokens_w = []
        self.l2m_last_w_time = []
        for t in self.mem_types:
            if t == MemType.DDR:
                # DDR 端口
                self.ddr_tokens.append(self.ddr_rate)
                self.ddr_last_time.append(0.0)
                # L2M 那两套放 dummy
                self.l2m_tokens_r.append(0.0)
                self.l2m_last_r_time.append(0.0)
                self.l2m_tokens_w.append(0.0)
                self.l2m_last_w_time.append(0.0)
            else:
                # L2M 端口
                self.ddr_tokens.append(0.0)
                self.ddr_last_time.append(0.0)
                self.l2m_tokens_r.append(self.l2m_rate_rd)
                self.l2m_last_r_time.append(0.0)
                self.l2m_tokens_w.append(self.l2m_rate_wr)
                self.l2m_last_w_time.append(0.0)

        # ------------ 其余状态 ------------
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
        self.sn_next_free_r = [0.0] * len(self.SN)
        self.sn_next_free_w = [0.0] * len(self.SN)
        self.sn_receive_next_free = [0.0] * len(self.SN)
        self.rn_next_free = [0.0] * len(self.RN)
        self.release_latency = release_latency
        self.requests = {}
        self.save_req_trace = save_req_trace
        self.choose_req_type = choose_req_type

        # ------------ 读写 gap 规则 ------------
        # 规则格式：
        # {"dma":"sdma","read":"l2m","write":"ddr","limit":256}
        self.gap_rules = gap_rules or []
        # 初始化计数器
        self.rule_counters = {}
        for rule in self.gap_rules:
            key = rule["dma"]
            self.rule_counters[key] = {"read": 0, "write": 0}

        # ------------ 处理 request_mix ------------
        self.request_mix = request_mix or {}
        if not self.request_mix:
            raise ValueError("request_mix 不能为空")
        total = sum(self.request_mix.values())
        self.mix_keys, self.mix_accu = [], []
        accu = 0.0
        for k, p in self.request_mix.items():
            accu += p / total
            self.mix_keys.append(k)
            self.mix_accu.append(accu)

        # 初始事件
        dma_types_in_mix = set(k[0] for k in self.mix_keys)
        for dma_id, dt in enumerate(self.dma_types):
            if ("gdma" if dt == DMAType.GDMA else "sdma") in dma_types_in_mix:
                self.schedule(Event(0.0, Event.RN_CMD, dma_id=dma_id))

    # ---------------- util ----------------
    def schedule(self, ev: Event):
        heapq.heappush(self.events, ev)

    def all_dma_free(self):
        return all(d.tracker_rd.count == 0 and d.tracker_wr.count == 0 for d in self.RN)

    # ------------ gap 限制判断  ------------
    def _gap_allows(self, dma_id, mem_str, cmd):
        """返回 True 表示可发送；False 表示因差值达到上限而禁止"""
        dma_str = "gdma" if self.dma_types[dma_id] == DMAType.GDMA else "sdma"

        for rule in self.gap_rules:
            if rule["dma"] != dma_str:
                continue
            key = rule["dma"]
            cnt = self.rule_counters[key]
            limit = rule["limit"]

            # 读侧判断
            if cmd == "R":
                ahead = cnt["read"] - cnt["write"]
                if ahead >= limit:  # 已领先到上限
                    return False
            # 写侧判断
            if cmd == "W":
                ahead = cnt["write"] - cnt["read"]
                if ahead >= limit:
                    return False
        return True

    # ------------ gap 计数递增  ------------
    def _gap_increment(self, dma_id, mem_str, cmd):
        dma_str = "gdma" if self.dma_types[dma_id] == DMAType.GDMA else "sdma"
        for rule in self.gap_rules:
            if rule["dma"] != dma_str:
                continue
            key = rule["dma"]
            if cmd == "R":
                self.rule_counters[key]["read"] += 1
            elif cmd == "W":
                self.rule_counters[key]["write"] += 1

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
                self.SN[ev.port_id].tracker_wr.dec()

            if self.cmds_left == 0 and self.all_dma_free():
                break

        self.report()

    # ===========================================================
    #                选择请求 & 发送  (含 gap 逻辑)
    # ===========================================================
    def choose_req(self, dma_id):
        """按 mix 抽样一次请求；返回 (port_id, cmd)"""
        dtype = self.dma_types[dma_id]
        need = "gdma" if dtype == DMAType.GDMA else "sdma"

        while True:
            r = random.random()
            idx = next(i for i, v in enumerate(self.mix_accu) if r <= v)
            dma_str, mem_str, cmd = self.mix_keys[idx]
            if dma_str != need:
                continue
            # 端口候选
            ports = [i for i, t in enumerate(self.mem_types) if (mem_str == "ddr" and t == MemType.DDR) or (mem_str == "l2m" and t == MemType.L2M)]
            if self.choose_req_type == 1:
                return random.choice(ports), cmd
            elif self.choose_req_type == 0:
                mem_id = self.next_mem[mem_str]
                self.next_mem[mem_str] = (self.next_mem[mem_str] + 1) % self.mem_num[mem_str]
                return ports[mem_id], cmd

    def on_rn_cmd(self, ev: Event):
        """尝试让某 DMA 发送一条新 CMD"""
        if self.cmds_left == 0:
            return

        port_id, cmd = self.choose_req(ev.dma_id)
        mem_str = "ddr" if self.mem_types[port_id] == MemType.DDR else "l2m"

        # gap 限制检查
        if not self._gap_allows(ev.dma_id, mem_str, cmd):
            # 达到硬限制：稍后重试
            self.schedule(Event(self.current_time + 1.0, Event.RN_CMD, dma_id=ev.dma_id))
            return

        dma = self.RN[ev.dma_id]
        mem = self.SN[port_id]
        tr_dma = dma.tracker_rd if cmd == "R" else dma.tracker_wr
        tr_mem = mem.tracker_rd if cmd == "R" else mem.tracker_wr

        # ostd 检查
        if tr_dma.can_accept() and tr_mem.can_accept():
            # gap 计数递增
            self._gap_increment(ev.dma_id, mem_str, cmd)

            tr_dma.inc()
            tr_mem.inc()
            self.cmds_left -= 1

            req_id = self.num_cmds - self.cmds_left
            self.requests[req_id] = Request(req_id, ev.dma_id, port_id, cmd, self.current_time)

            self.schedule(Event(self.current_time + self.path_delay, Event.SN_RECEIVE_CMD, dma_id=ev.dma_id, port_id=port_id, cmd=cmd, req_id=req_id))

        # 同一个 DMA 继续下一条
        if self.cmds_left:
            self.schedule(Event(self.current_time + 1.0, Event.RN_CMD, dma_id=ev.dma_id))

    # ===========================================================
    #          SN / RN 后续事件 (与原版相同，略去注释)
    # ===========================================================
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

    def handle_sn_done(self, ev: Event):
        req = self.requests[ev.req_id]
        port, cmd = ev.port_id, ev.cmd
        mem = self.SN[port]

        if cmd == "R":
            mem.tracker_rd.dec()
        else:
            self.schedule(Event(ev.time + self.release_latency, Event.SN_RELEASE_CREDIT, port_id=port, cmd="W"))

        send_cnt = self.burst_length if cmd == "R" else 1
        for i in range(send_cnt):
            # 依据端口类型 + cmd 选取相应的 next_free
            if self.mem_types[port] == MemType.L2M:
                if cmd == "R":
                    t0 = max(ev.time, self.sn_next_free_r[port])
                else:
                    t0 = max(ev.time, self.sn_next_free_w[port])
            else:
                # DDR 仍然一条流水线
                t0 = max(ev.time, self.sn_next_free_r[port])
            if cmd == "R":
                t0 = self._consume_token(port, self.flit_size, t0, cmd="R")
            t_send = math.ceil(t0)
            # 更新对应队列的 next_free
            if self.mem_types[port] == MemType.L2M:
                if cmd == "R":
                    self.sn_next_free_r[port] = t_send
                else:
                    self.sn_next_free_w[port] = t_send
            else:
                self.sn_next_free_r[port] = t_send
            req.sn_send_data.append(t_send)

            etype = Event.RN_DATA if cmd == "R" else Event.RN_ACK
            self.schedule(Event(t_send + self.path_delay, etype, dma_id=ev.dma_id, port_id=port, cmd=cmd, flit_idx=(i if cmd == "R" else None), req_id=ev.req_id))

    def handle_rn_data(self, ev: Event):
        req = self.requests[ev.req_id]
        req.read_flit_recv.append(ev.time)
        if ev.flit_idx == self.burst_length - 1:
            self.RN[ev.dma_id].tracker_rd.dec()

    def handle_rn_ack(self, ev: Event):
        self.requests[ev.req_id].ack_time = ev.time
        self.schedule(Event(ev.time, Event.RN_SEND, dma_id=ev.dma_id, port_id=ev.port_id, cmd="W", req_id=ev.req_id))

    def handle_rn_send(self, ev: Event):
        req = self.requests[ev.req_id]
        dma_id, port = ev.dma_id, ev.port_id
        t0 = ev.time
        for i in range(self.burst_length):
            t0 = max(t0, self.rn_next_free[dma_id])
            # 写数据保留 SN 侧 token‑bucket，不在 RN 扣
            send_time = math.ceil(t0)
            req.rn_send_data.append(send_time)
            self.rn_next_free[dma_id] = t0 + self.ip_gap
            t0 += self.ip_gap
            self.schedule(Event(send_time + self.path_delay, Event.SN_DATA, dma_id=dma_id, port_id=port, cmd="W", flit_idx=i, req_id=ev.req_id))

    def handle_sn_data(self, ev: Event):
        port = ev.port_id
        # 写数据回写也走写队列，读数据（如果有）走读队列
        if self.mem_types[port] == MemType.L2M:
            t0 = max(ev.time, self.sn_next_free_w[port])
        else:
            t0 = max(ev.time, self.sn_next_free_r[port])
        if ev.cmd == "W":
            t0 = self._consume_token(port, self.flit_size, t0, cmd="W")
        t_done = math.ceil(t0)
        if self.mem_types[port] == MemType.L2M:
            self.sn_next_free_w[port] = t_done
        else:
            self.sn_next_free_r[port] = t_done
        self.requests[ev.req_id].write_flit_recv.append(t_done)
        if ev.flit_idx == self.burst_length - 1:
            self.RN[ev.dma_id].tracker_wr.dec()

    # ---------- token-bucket 消耗函数 ----------
    def _consume_token(self, port, size, now, cmd):
        """
        port: SN 端口号
        size: bytes
        now: 当前时间
        cmd: "R" 或 "W"
        """
        if self.mem_types[port] == MemType.DDR:
            # DDR 端口读写共用一个 bucket
            rate = self.ddr_rate
            tokens = self.ddr_tokens
            last = self.ddr_last_time
        else:
            # L2M 端口，读写分开
            if cmd == "R":
                rate = self.l2m_rate_rd
                tokens = self.l2m_tokens_r
                last = self.l2m_last_r_time
            else:
                rate = self.l2m_rate_wr
                tokens = self.l2m_tokens_w
                last = self.l2m_last_w_time

        # Accumulate
        tokens[port] = min(tokens[port] + (now - last[port]) * rate, rate)
        last[port] = now

        # 如果不足，等待
        if tokens[port] < size:
            wait = (size - tokens[port]) / rate
            now += wait
            tokens[port] = 0.0
            last[port] = now
        else:
            tokens[port] -= size

        return now

    # ===========================================================
    #                       统   计   报   告
    # ===========================================================
    def report(self):
        groups = {("gdma", "R"): [], ("gdma", "W"): [], ("sdma", "R"): [], ("sdma", "W"): []}
        bytes_per_req = self.burst_length * self.flit_size

        for r in self.requests.values():
            dm = "gdma" if self.dma_types[r.dma_id] == DMAType.GDMA else "sdma"
            done_t = max(r.read_flit_recv) if r.cmd == "R" else max(r.rn_send_data)
            # done_t = max(r.read_flit_recv) if r.cmd == "R" else max(r.write_flit_recv)
            groups[(dm, r.cmd)].append((done_t, bytes_per_req))

        plt.figure(figsize=(8, 5))
        total_bw = 0.0
        for k, data in groups.items():
            if not data:
                continue
            data.sort(key=lambda x: x[0])
            ts = np.array([t for t, _ in data])
            cum = np.cumsum([b for _, b in data])
            bw = cum / ts
            t85 = np.percentile(ts, 85)
            mask = ts <= t85
            plt.plot(ts[mask] / 1000, bw[mask], drawstyle="steps-post", label=f"{k[0].upper()}-{k[1]}")
            total_bw += bw[-1]
            print(f"{k[0].upper()}-{k[1]} Bandwidth: {bw[-1]:.2f} GB/s")
        print(f"Total Bandwidth: {total_bw:.2f} GB/s")

        # 打印最终差值
        for rule in self.gap_rules:
            key = rule["dma"]
            cnt = self.rule_counters[key]
            diff = abs(cnt["read"] - cnt["write"])
            print(f"[GAP] {key}: read={cnt['read']} write={cnt['write']} " f"diff={diff} limit={rule['limit']}")

        plt.xlabel("Time (µs)")
        plt.ylabel("Bandwidth (GB/s)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

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
#                            DEMO
# ===============================================================
if __name__ == "__main__":
    # 例：SDMA 读 L2M 与 写 DDR 差值 ≤ 256
    mix = {
        # 性能case 1
        ("gdma", "l2m", "W"): 1,
        # ("sdma", "l2m", "R"): 1,
        # ("sdma", "ddr", "W"): 1,
        # 性能case 2
        # ("gdma", "l2m", "W"): 1,
        # ("sdma", "l2m", "R"): 1,
        # ("sdma", "ddr", "W"): 1,
    }
    gap_rules = [
        {"dma": "gdma", "limit": np.inf},
        {"dma": "sdma", "limit": 100},
    ]

    sim = NoC_Simulator(
        n_gdma=4,
        n_sdma=4,
        n_ddr=16,
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
        l2m_latency=[(12, 16)],
        latency_var=(25, 0),
        path_delay=10,
        num_cmds=128 * 300,
        burst_length=2,
        ddr_data_bw=76.8 / 4,  # GB/s
        l2m_data_bw=(128, 128),  # (read, write) GB/s
        release_latency=40.0,
        request_mix=mix,
        gap_rules=gap_rules,
        save_req_trace=0,
        choose_req_type=0,
    )
    sim.run()
