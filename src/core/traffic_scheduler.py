import os
from typing import List, Dict, Tuple, Optional, Iterator
from collections import defaultdict
import heapq


class TrafficState:
    """跟踪单个traffic的执行状态"""

    def __init__(self, traffic_id: str, chain_id: str, total_req: int, total_flit: int):
        self.traffic_id = traffic_id
        self.chain_id = chain_id
        self.total_req = total_req
        self.total_flit = total_flit
        self.injected_req = 0
        self.completed_req = 0
        self.sent_flit = 0
        self.received_flit = 0
        self.start_time = 0
        self.end_time = 0

    def is_completed(self) -> bool:
        """判断traffic是否完成"""
        return self.injected_req >= self.total_req and self.sent_flit >= self.total_flit and self.received_flit >= self.total_flit

    def update_injected_req(self):
        """更新已注入请求数"""
        self.injected_req += 1

    def update_sent_flit(self):
        """更新已发送flit数"""
        self.sent_flit += 1

    def update_received_flit(self):
        """更新已接收flit数"""
        self.received_flit += 1


class SerialChain:
    """表示一条串行的traffic链"""

    def __init__(self, chain_id: str, traffic_files: List[str]):
        self.chain_id = chain_id
        self.traffic_files = traffic_files
        self.current_index = 0
        self.chain_time_offset = 0
        self.is_completed = False
        self.start_time = 0

    def get_current_traffic_file(self) -> Optional[str]:
        """获取当前应该执行的traffic文件"""
        if self.current_index < len(self.traffic_files):
            return self.traffic_files[self.current_index]
        return None

    def advance_to_next(self):
        """推进到下一个traffic"""
        self.current_index += 1
        if self.current_index >= len(self.traffic_files):
            self.is_completed = True

    def has_next_traffic(self) -> bool:
        """检查是否还有下一个traffic"""
        return self.current_index < len(self.traffic_files)


class TrafficScheduler:
    """Traffic调度器，支持多条并行的串行链"""

    def __init__(self, config, traffic_file_path: str):
        self.config = config
        self.traffic_file_path = traffic_file_path
        self.parallel_chains: List[SerialChain] = []
        self.active_traffics: Dict[str, TrafficState] = {}
        self.pending_requests = []  # 使用堆来维护按时间排序的请求
        self.current_cycle = 0
        self.verbose = False

    def setup_parallel_chains(self, chains_config: List[List[str]]):
        """
        设置并行的串行链

        Args:
            chains_config: 链配置，例如：
                [
                    ["traffic_A.txt", "traffic_B.txt", "traffic_C.txt"],  # 链1
                    ["traffic_D.txt", "traffic_E.txt"],                   # 链2
                    ["traffic_F.txt"]                                     # 链3
                ]
        """
        self.parallel_chains.clear()
        self.active_traffics.clear()
        self.pending_requests.clear()

        for i, traffic_files in enumerate(chains_config):
            chain_id = f"chain_{i}"
            chain = SerialChain(chain_id, traffic_files)
            self.parallel_chains.append(chain)

        if self.verbose:
            print(f"Setup {len(self.parallel_chains)} parallel chains")
            for chain in self.parallel_chains:
                print(f"  {chain.chain_id}: {chain.traffic_files}")

    def setup_single_chain(self, traffic_files: List[str]):
        """设置单条串行链（向后兼容）"""
        self.setup_parallel_chains([traffic_files])

    def start_initial_traffics(self):
        """启动每条链的第一个traffic"""
        for chain in self.parallel_chains:
            if chain.has_next_traffic():
                self._start_single_traffic(chain)

    def _start_single_traffic(self, chain: SerialChain):
        """启动链中的单个traffic"""
        traffic_file = chain.get_current_traffic_file()
        if not traffic_file:
            return

        traffic_id = f"{chain.chain_id}_t{chain.current_index}"

        # 解析traffic文件
        total_req, total_flit, requests = self._parse_traffic_file(traffic_file, chain.chain_time_offset, traffic_id)

        # 创建traffic状态
        traffic_state = TrafficState(traffic_id, chain.chain_id, total_req, total_flit)
        traffic_state.start_time = chain.chain_time_offset
        self.active_traffics[traffic_id] = traffic_state

        # 将请求添加到pending队列中
        for req in requests:
            heapq.heappush(self.pending_requests, req)

        if self.verbose:
            print(f"Started traffic {traffic_file} on {chain.chain_id} at time {chain.chain_time_offset}")
            print(f"  Traffic ID: {traffic_id}, Requests: {total_req}, Flits: {total_flit}")

    def _parse_traffic_file(self, filename: str, time_offset: int, traffic_id: str) -> Tuple[int, int, List[Tuple]]:
        """
        解析traffic文件并添加时间偏移

        Returns:
            (total_req, total_flit, requests_list)
        """
        abs_path = os.path.join(self.traffic_file_path, filename)
        requests = []
        read_req = write_req = read_flit = write_flit = 0

        with open(abs_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue

                # 解析原始数据
                t, src, src_t, dst, dst_t, op, burst = parts
                t = int(t) * self.config.NETWORK_FREQUENCY + time_offset
                src, dst, burst = int(src), int(dst), int(burst)

                # 创建带traffic_id的请求元组
                req_tuple = (t, src, src_t, dst, dst_t, op, burst, traffic_id)
                requests.append(req_tuple)

                # 统计
                if op == "R":
                    read_req += 1
                    read_flit += burst
                else:
                    write_req += 1
                    write_flit += burst

        total_req = read_req + write_req
        total_flit = read_flit + write_flit

        return total_req, total_flit, requests

    def get_next_request(self, current_cycle: int) -> Optional[Tuple]:
        """
        获取下一个应该处理的请求

        Args:
            current_cycle: 当前仿真周期

        Returns:
            请求元组或None
        """
        self.current_cycle = current_cycle

        if self.pending_requests and self.pending_requests[0][0] <= current_cycle:
            return heapq.heappop(self.pending_requests)
        return None

    def has_pending_requests(self) -> bool:
        """检查是否还有等待处理的请求"""
        return len(self.pending_requests) > 0

    def update_traffic_stats(self, traffic_id: str, stat_type: str):
        """
        更新traffic统计信息

        Args:
            traffic_id: traffic标识
            stat_type: 统计类型 ('injected_req', 'sent_flit', 'received_flit')
        """
        if traffic_id in self.active_traffics:
            state = self.active_traffics[traffic_id]
            if stat_type == "injected_req":
                state.update_injected_req()
            elif stat_type == "sent_flit":
                state.update_sent_flit()
            elif stat_type == "received_flit":
                state.update_received_flit()

    def check_and_advance_chains(self, current_cycle: int) -> List[str]:
        """
        检查traffic完成情况并推进链

        Args:
            current_cycle: 当前仿真周期

        Returns:
            完成的traffic_id列表
        """
        completed_traffics = []

        # 检查已完成的traffic
        for traffic_id, state in list(self.active_traffics.items()):
            if state.is_completed():
                completed_traffics.append(traffic_id)

                # 更新对应链的状态
                chain = self._find_chain_by_id(state.chain_id)
                if chain:
                    # 记录结束时间
                    state.end_time = current_cycle // self.config.NETWORK_FREQUENCY
                    chain.chain_time_offset = state.end_time

                    # 推进到下一个traffic
                    chain.advance_to_next()

                    if self.verbose:
                        print(f"Traffic {traffic_id} completed at time {state.end_time}")

                    # 如果链还有下一个traffic，立即启动
                    if chain.has_next_traffic():
                        self._start_single_traffic(chain)
                    else:
                        if self.verbose:
                            print(f"Chain {chain.chain_id} completed at time {chain.chain_time_offset}")

                # 清理完成的traffic
                del self.active_traffics[traffic_id]

        return completed_traffics

    def _find_chain_by_id(self, chain_id: str) -> Optional[SerialChain]:
        """根据chain_id查找链"""
        for chain in self.parallel_chains:
            if chain.chain_id == chain_id:
                return chain
        return None

    def is_all_completed(self) -> bool:
        """检查是否所有链都已完成"""
        return len(self.active_traffics) == 0 and len(self.pending_requests) == 0 and all(chain.is_completed for chain in self.parallel_chains)

    def get_active_traffic_count(self) -> int:
        """获取当前活跃的traffic数量"""
        return len(self.active_traffics)

    def get_chain_status(self) -> Dict[str, Dict]:
        """获取所有链的状态信息"""
        status = {}
        for chain in self.parallel_chains:
            status[chain.chain_id] = {
                "current_index": chain.current_index,
                "total_traffics": len(chain.traffic_files),
                "current_file": chain.get_current_traffic_file(),
                "time_offset": chain.chain_time_offset,
                "is_completed": chain.is_completed,
            }
        return status

    def get_active_traffic_status(self) -> Dict[str, Dict]:
        """获取活跃traffic的状态信息"""
        status = {}
        for traffic_id, state in self.active_traffics.items():
            status[traffic_id] = {
                "chain_id": state.chain_id,
                "total_req": state.total_req,
                "injected_req": state.injected_req,
                "total_flit": state.total_flit,
                "sent_flit": state.sent_flit,
                "received_flit": state.received_flit,
                "progress": f"{state.received_flit}/{state.total_flit}",
                "is_completed": state.is_completed(),
            }
        return status

    def set_verbose(self, verbose: bool):
        """设置详细输出模式"""
        self.verbose = verbose

    def get_combined_filename(self) -> str:
        """
        生成组合的文件名（用于结果保存路径）

        Returns:
            组合的文件名，不包含扩展名
        """
        if not self.parallel_chains:
            return "no_traffic"

        chain_names = []
        for chain in self.parallel_chains:
            # 移除扩展名并连接链中的文件
            files_without_ext = [f[:-4] if f.endswith(".txt") else f for f in chain.traffic_files]
            chain_name = "-".join(files_without_ext)
            chain_names.append(chain_name)

        # 用 "_" 分隔不同的链
        return "_".join(chain_names)

    def reset(self):
        """重置调度器状态"""
        self.parallel_chains.clear()
        self.active_traffics.clear()
        self.pending_requests.clear()
        self.current_cycle = 0


# 使用示例和测试代码
if __name__ == "__main__":
    # 示例配置类
    class MockConfig:
        def __init__(self):
            self.NETWORK_FREQUENCY = 1000

    # 创建调度器
    config = MockConfig()
    scheduler = TrafficScheduler(config, "./traffic_files/")
    scheduler.set_verbose(True)

    # 设置并行链
    chains_config = [
        ["traffic_A.txt", "traffic_B.txt", "traffic_C.txt"],
        ["traffic_D.txt", "traffic_E.txt"],
        ["traffic_F.txt"],
    ]

    scheduler.setup_parallel_chains(chains_config)
    scheduler.start_initial_traffics()

    # 模拟主循环
    cycle = 0
    while not scheduler.is_all_completed():
        cycle += 1000  # 假设每次增加1000个周期

        # 检查是否有请求需要处理
        while True:
            req = scheduler.get_next_request(cycle)
            if req is None:
                break
            print(f"Processing request at cycle {cycle}: {req}")
            # 这里会调用实际的请求处理逻辑

        # 检查并推进链
        completed = scheduler.check_and_advance_chains(cycle)
        if completed:
            print(f"Completed traffics: {completed}")

        # 输出状态
        if cycle % 10000 == 0:
            print(f"\nCycle {cycle}:")
            print(f"Active traffics: {scheduler.get_active_traffic_count()}")
            print(f"Pending requests: {len(scheduler.pending_requests)}")

        # 防止无限循环
        if cycle > 100000:
            break

    print("All chains completed!")
