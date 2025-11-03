"""
D2D AXI通道带宽测试

只测试D2D_SN、D2D_RN、D2D_Sys三个模块，不涉及NoC网络和其他IP
目标：验证AXI通道能否达到配置的带宽(256GB/s)
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.components.d2d_sys import D2D_Sys
from src.utils.components.flit import Flit, _flit_pool
from config.d2d_config import D2DConfig


class MinimalConfig:
    """最小化配置，只包含D2D必需的参数"""

    def __init__(self):
        # 基础频率参数
        self.NETWORK_FREQUENCY = 2  # 2GHz
        self.FLIT_SIZE = 128  # bytes

        # AXI通道延迟 (cycles @ 2GHz)
        self.D2D_AR_LATENCY = 10
        self.D2D_R_LATENCY = 8
        self.D2D_AW_LATENCY = 10
        self.D2D_W_LATENCY = 2
        self.D2D_B_LATENCY = 8

        # AXI通道带宽 (GB/s)
        self.D2D_AR_BANDWIDTH = 256
        self.D2D_R_BANDWIDTH = 256
        self.D2D_AW_BANDWIDTH = 256
        self.D2D_W_BANDWIDTH = 256  # 重点测试写数据通道
        self.D2D_B_BANDWIDTH = 256

        # D2D数据带宽限制
        self.D2D_DATA_BW_LIMIT = 256


class MockD2DSN:
    """模拟D2D_SN - 持续生成写请求"""

    def __init__(self, d2d_sys: D2D_Sys, target_die_id: int):
        self.d2d_sys = d2d_sys
        self.target_die_id = target_die_id
        self.packet_id = 0
        self.requests_sent = 0
        self.current_cycle = 0

    def generate_write_request(self):
        """生成一个写请求(AW)和4个写数据(W)"""
        # 创建写地址请求(AW通道)
        aw_flit = _flit_pool.get_flit(source=0, destination=1, path=[0, 1])
        aw_flit.packet_id = self.packet_id
        aw_flit.req_type = "write"
        aw_flit.flit_type = "req"
        aw_flit.burst_length = 4
        aw_flit.d2d_origin_die = 0
        aw_flit.d2d_target_die = 1
        aw_flit.d2d_origin_node = 0
        aw_flit.d2d_target_node = 1
        aw_flit.d2d_origin_type = "test_sn"
        aw_flit.d2d_target_type = "test_rn"

        # 发送到D2D_Sys (AW通道)
        self.d2d_sys.enqueue_sn(aw_flit, self.target_die_id, 0)

        # 创建4个写数据flit (W通道)
        for i in range(4):
            w_flit = _flit_pool.get_flit(source=0, destination=1, path=[0, 1])
            w_flit.packet_id = self.packet_id
            w_flit.req_type = "write"
            w_flit.flit_type = "data"
            w_flit.flit_id = i
            w_flit.burst_length = 4
            w_flit.is_last_flit = (i == 3)
            w_flit.d2d_origin_die = 0
            w_flit.d2d_target_die = 1
            w_flit.d2d_origin_node = 0
            w_flit.d2d_target_node = 1
            w_flit.d2d_origin_type = "test_sn"
            w_flit.d2d_target_type = "test_rn"

            # 发送到D2D_Sys (W通道)
            self.d2d_sys.enqueue_sn(w_flit, self.target_die_id, 0)

        self.packet_id += 1
        self.requests_sent += 1


class MockD2DRN:
    """模拟D2D_RN - 立即响应write_complete"""

    def __init__(self, d2d_sys: D2D_Sys, source_die_id: int):
        self.d2d_sys = d2d_sys
        self.source_die_id = source_die_id
        self.requests_received = 0
        self.data_received = 0
        self.responses_sent = 0
        self.pending_requests = {}  # {packet_id: request_info}

    def schedule_cross_die_receive(self, flit: Flit, arrival_cycle: int):
        """接收AXI传输来的flit"""
        # 简化处理：立即处理，不考虑arrival_cycle
        self.handle_received_flit(flit)

    def handle_received_flit(self, flit: Flit):
        """处理接收到的flit"""
        packet_id = flit.packet_id

        # 检查flit类型
        if hasattr(flit, "flit_position") and flit.flit_position:
            channel = flit.flit_position.split("_")[1] if "_" in flit.flit_position else ""

            if channel == "AW":
                # 写地址请求
                self.requests_received += 1
                self.pending_requests[packet_id] = {
                    "data_count": 0,
                    "burst_length": getattr(flit, "burst_length", 4),
                    "origin_die": flit.d2d_origin_die,
                }

            elif channel == "W":
                # 写数据
                self.data_received += 1
                if packet_id in self.pending_requests:
                    self.pending_requests[packet_id]["data_count"] += 1

                    # 检查是否收齐所有数据
                    if self.pending_requests[packet_id]["data_count"] >= self.pending_requests[packet_id]["burst_length"]:
                        # 立即发送write_complete响应(B通道)
                        self.send_write_complete(packet_id)
                        del self.pending_requests[packet_id]

        # 回收AXI flit
        _flit_pool.return_flit(flit)

    def send_write_complete(self, packet_id: int):
        """发送write_complete响应(B通道)"""
        b_flit = _flit_pool.get_flit(source=1, destination=0, path=[1, 0])
        b_flit.packet_id = packet_id
        b_flit.rsp_type = "write_complete"
        b_flit.req_type = "write"
        b_flit.d2d_origin_die = 0
        b_flit.d2d_target_die = 1
        b_flit.d2d_origin_node = 0
        b_flit.d2d_target_node = 1
        b_flit.d2d_origin_type = "test_sn"
        b_flit.d2d_target_type = "test_rn"

        # 通过D2D_Sys发送(B通道)
        self.d2d_sys.enqueue_rn(b_flit, self.source_die_id, 0, channel="B")
        self.responses_sent += 1


def run_bandwidth_test(test_duration_ns=1000, max_requests=None):
    """
    运行D2D AXI带宽测试

    Args:
        test_duration_ns: 测试时长(纳秒)
        max_requests: 最大请求数(None表示不限制)
    """
    # 创建配置
    config = MinimalConfig()

    # 创建D2D_Sys (Die0节点0 -> Die1节点1)
    d2d_sys = D2D_Sys(
        node_pos=0,
        die_id=0,
        target_die_id=1,
        target_node_pos=1,
        config=config
    )

    # 创建Mock组件
    mock_sn = MockD2DSN(d2d_sys, target_die_id=1)
    mock_rn = MockD2DRN(d2d_sys, source_die_id=0)

    # 设置D2D_Sys的目标接口
    d2d_sys.target_die_interfaces = {
        1: {"rn": mock_rn, "sn": mock_sn}
    }

    # 计算总周期数 (2GHz)
    total_cycles = test_duration_ns * config.NETWORK_FREQUENCY

    print(f"开始D2D AXI带宽测试")
    print(f"配置:")
    print(f"  测试时长: {test_duration_ns}ns ({total_cycles} cycles @ {config.NETWORK_FREQUENCY}GHz)")
    print(f"  AXI_W带宽: {config.D2D_W_BANDWIDTH} GB/s")
    print(f"  Flit大小: {config.FLIT_SIZE} bytes")
    print(f"  最大请求数: {max_requests if max_requests else '无限制'}")
    print()

    # 运行仿真
    for cycle in range(total_cycles):
        # D2D_SN持续生成写请求
        if max_requests is None or mock_sn.requests_sent < max_requests:
            mock_sn.generate_write_request()

        # D2D_Sys处理传输
        d2d_sys.step(cycle)

    # 统计结果
    stats = d2d_sys.get_statistics()

    print("=" * 60)
    print("测试结果:")
    print("=" * 60)
    print(f"\nD2D_SN统计:")
    print(f"  写请求发送: {mock_sn.requests_sent}")
    print(f"  预期写数据flit: {mock_sn.requests_sent * 4}")

    print(f"\nD2D_RN统计:")
    print(f"  写请求接收: {mock_rn.requests_received}")
    print(f"  写数据接收: {mock_rn.data_received}")
    print(f"  写完成响应发送: {mock_rn.responses_sent}")

    print(f"\nAXI通道统计:")
    for channel_type in ["AW", "W", "B"]:
        channel_stats = stats["axi_channel_stats"][channel_type]
        print(f"\n  {channel_type}通道:")
        print(f"    Injected: {channel_stats['injected']}")
        print(f"    Ejected: {channel_stats['ejected']}")
        print(f"    Throttled: {channel_stats['throttled']}")

        if channel_type == "W":
            # 计算写数据通道的带宽
            total_data_bytes = channel_stats['ejected'] * config.FLIT_SIZE
            bandwidth_gbps = total_data_bytes / test_duration_ns  # GB/s
            utilization = bandwidth_gbps / config.D2D_W_BANDWIDTH * 100

            print(f"    数据量: {total_data_bytes} bytes ({total_data_bytes / 1024:.2f} KB)")
            print(f"    实际带宽: {bandwidth_gbps:.2f} GB/s")
            print(f"    理论带宽: {config.D2D_W_BANDWIDTH} GB/s")
            print(f"    利用率: {utilization:.2f}%")

    print(f"\n传输统计:")
    print(f"  RN发送: {stats['rn_transmit_count']}")
    print(f"  SN发送: {stats['sn_transmit_count']}")
    print(f"  总传输: {stats['total_transmit_count']}")

    # 分析瓶颈
    print("\n" + "=" * 60)
    print("瓶颈分析:")
    print("=" * 60)

    w_stats = stats["axi_channel_stats"]["W"]
    if w_stats["throttled"] > 0:
        print(f"⚠️  W通道被限流 {w_stats['throttled']} 次")
        print(f"   原因: TokenBucket带宽限制")

    if w_stats["ejected"] < mock_sn.requests_sent * 4:
        print(f"⚠️  部分写数据仍在传输中")
        print(f"   预期: {mock_sn.requests_sent * 4}, 实际ejected: {w_stats['ejected']}")

    # 计算理论峰值
    theoretical_flits = config.D2D_W_BANDWIDTH / config.FLIT_SIZE * test_duration_ns
    print(f"\n理论峰值: {theoretical_flits:.0f} flits ({theoretical_flits * config.FLIT_SIZE / 1024:.2f} KB)")
    print(f"实际传输: {w_stats['ejected']} flits ({w_stats['ejected'] * config.FLIT_SIZE / 1024:.2f} KB)")
    print(f"达成率: {w_stats['ejected'] / theoretical_flits * 100:.2f}%")


if __name__ == "__main__":
    # 测试1: 短时间高压测试
    print("\n" + "=" * 60)
    print("测试1: 1000ns高压测试")
    print("=" * 60)
    run_bandwidth_test(test_duration_ns=1000, max_requests=None)

    print("\n\n")

    # 测试2: 固定请求数测试
    print("=" * 60)
    print("测试2: 固定1000个请求")
    print("=" * 60)
    run_bandwidth_test(test_duration_ns=10000, max_requests=1000)
