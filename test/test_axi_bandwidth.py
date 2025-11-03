"""
AXI通道带宽测试脚本

测试目标：
1. 验证5个AXI通道（AR/R/AW/W/B）的独立性
2. 测试各通道能否达到配置的带宽上限
3. 验证2GHz频率下的实际传输能力
4. 分析TokenBucket限流情况
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.d2d_model import D2D_Model
from config.d2d_config import D2DConfig


class AXIBandwidthTester:
    """AXI带宽测试器"""

    def __init__(self, config_file):
        self.config_file = config_file
        self.config = D2DConfig(d2d_config_file=config_file)
        self.results = {}

        # 创建结果目录
        self.result_dir = project_root / "test_data" / "axi_bandwidth_test"
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def generate_traffic_file(self, scenario, traffic_file):
        """
        生成测试traffic文件

        Args:
            scenario: 测试场景名称
            traffic_file: 输出文件路径
        """
        print(f"\n生成测试场景: {scenario}")

        # 获取配置参数
        burst_length = 4  # 默认burst长度
        sim_cycles = 2000  # 默认模拟周期（缩短以加快测试）

        # GDMA在Die0的源节点位置（使用0作为示例）
        gdma_source = 0  # gdma_0在Die0的节点位置

        traffic_data = []

        if scenario == "pure_read":
            # 场景1: 纯读测试（压满AR+R通道）
            # 每10个cycle发起一个读请求
            for cycle in range(0, sim_cycles, 10):
                traffic_data.append({
                    "cycle": cycle,
                    "src_die": 0,
                    "source": gdma_source,
                    "source_type": "gdma_0",
                    "destination": 7,  # DDR在Die1的目标位置（参考示例文件）
                    "destination_type": "ddr_0",
                    "req_type": "R",  # 使用大写R
                    "burst_length": burst_length,
                    "d2d_target_die": 1
                })

        elif scenario == "pure_write":
            # 场景2: 纯写测试（压满AW+W+B通道）
            for cycle in range(0, sim_cycles, 10):
                traffic_data.append({
                    "cycle": cycle,
                    "src_die": 0,
                    "source": gdma_source,
                    "source_type": "gdma_0",
                    "destination": 7,
                    "destination_type": "ddr_0",
                    "req_type": "W",  # 使用大写W
                    "burst_length": burst_length,
                    "d2d_target_die": 1
                })

        elif scenario == "mixed_balanced":
            # 场景3: 读写均衡（50%读 + 50%写）
            for cycle in range(0, sim_cycles, 10):
                req_type = "R" if (cycle // 10) % 2 == 0 else "W"
                traffic_data.append({
                    "cycle": cycle,
                    "src_die": 0,
                    "source": gdma_source,
                    "source_type": "gdma_0",
                    "destination": 7,
                    "destination_type": "ddr_0",
                    "req_type": req_type,
                    "burst_length": burst_length,
                    "d2d_target_die": 1
                })

        elif scenario == "high_rate":
            # 场景4: 高注入率测试（每个cycle都发请求）
            for cycle in range(0, min(sim_cycles, 500)):  # 限制请求数量（缩短测试）
                req_type = "R" if cycle % 2 == 0 else "W"
                traffic_data.append({
                    "cycle": cycle,
                    "src_die": 0,
                    "source": gdma_source,
                    "source_type": "gdma_0",
                    "destination": 7,
                    "destination_type": "ddr_0",
                    "req_type": req_type,
                    "burst_length": burst_length,
                    "d2d_target_die": 1
                })

        # 写入文件（CSV格式）
        # 格式: inject_time, src_die, src_node, src_ip, dst_die, dst_node, dst_ip, req_type, burst_length
        with open(traffic_file, 'w') as f:
            f.write("# D2D Multi-Die-Pair Traffic\n")
            f.write("# Format: inject_time, src_die, src_node, src_ip, dst_die, dst_node, dst_ip, req_type, burst_length\n")
            for req in traffic_data:
                line = f"{req['cycle']}, {req.get('src_die', 0)}, {req['source']}, {req['source_type']}, {req['d2d_target_die']}, {req['destination']}, {req['destination_type']}, {req['req_type']}, {req['burst_length']}\n"
                f.write(line)

        print(f"  生成 {len(traffic_data)} 个请求")
        print(f"  保存到: {traffic_file}")

    def run_simulation(self, scenario):
        """
        运行单个测试场景

        Args:
            scenario: 测试场景名称
        """
        print(f"\n{'='*60}")
        print(f"运行测试场景: {scenario}")
        print(f"{'='*60}")

        # 生成traffic文件
        traffic_file = self.result_dir / f"traffic_{scenario}.txt"
        self.generate_traffic_file(scenario, traffic_file)

        # 创建模型
        print("\n开始模拟...")
        model = D2D_Model(
            config=self.config,
            model_type="REQ_RSP",
            result_save_path=str(self.result_dir / scenario),
            results_fig_save_path=str(self.result_dir / scenario / "figures"),
            verbose=0,
        )

        # 设置traffic调度器
        traffic_file_path = str(self.result_dir)
        traffic_chains = [[traffic_file.name]]  # 使用文件名
        model.setup_traffic_scheduler(traffic_file_path=traffic_file_path, traffic_chains=traffic_chains)

        # 关闭可视化（加快速度）
        model.setup_visualization(enable=0)

        # 运行
        model.run()

        # 收集统计数据
        stats = self.collect_statistics(model, scenario)
        self.results[scenario] = stats

        print(f"\n场景 {scenario} 完成")

        return stats

    def collect_statistics(self, model, scenario):
        """
        收集AXI通道统计数据

        Args:
            model: 模拟模型
            scenario: 场景名称
        """
        print("\n收集统计数据...")

        stats = {
            "scenario": scenario,
            "channels": {},
            "interface": {},
            "summary": {}
        }

        # 网络频率和模拟周期
        network_freq = 2.0  # GHz（从配置文件读取）
        sim_cycles = model.current_cycle if hasattr(model, 'current_cycle') else 10000
        time_ns = sim_cycles / network_freq

        stats["summary"]["sim_cycles"] = sim_cycles
        stats["summary"]["time_ns"] = time_ns
        stats["summary"]["network_freq_ghz"] = network_freq

        # 收集每个Die的D2D_Sys统计
        for die_id, die_model in model.dies.items():
            if not hasattr(die_model, "d2d_systems"):
                continue

            for pos, d2d_sys in die_model.d2d_systems.items():
                # 通道级统计
                for channel in ["AR", "R", "AW", "W", "B"]:
                    if channel not in stats["channels"]:
                        stats["channels"][channel] = {
                            "flit_count": 0,
                            "injected": 0,
                            "ejected": 0,
                            "throttled": 0,
                            "bandwidth_gbps": 0.0
                        }

                    # 累加flit计数
                    flit_count = d2d_sys.axi_channel_flit_count.get(channel, 0)
                    stats["channels"][channel]["flit_count"] += flit_count

                    # 累加通道统计
                    if hasattr(d2d_sys, "axi_channel_stats"):
                        ch_stats = d2d_sys.axi_channel_stats.get(channel, {})
                        stats["channels"][channel]["injected"] += ch_stats.get("injected", 0)
                        stats["channels"][channel]["ejected"] += ch_stats.get("ejected", 0)
                        stats["channels"][channel]["throttled"] += ch_stats.get("throttled", 0)

                # 接口级统计
                stats["interface"]["rn_transmit"] = getattr(d2d_sys, "rn_transmit_count", 0)
                stats["interface"]["sn_transmit"] = getattr(d2d_sys, "sn_transmit_count", 0)
                stats["interface"]["data_throttled"] = getattr(d2d_sys, "data_throttled_count", 0)

        # 计算带宽
        flit_size = 128  # Bytes
        for channel, ch_stats in stats["channels"].items():
            flit_count = ch_stats["flit_count"]
            if time_ns > 0:
                bandwidth = flit_count * flit_size / time_ns  # GB/s
                ch_stats["bandwidth_gbps"] = bandwidth

        # 计算总带宽
        total_bandwidth = sum(ch["bandwidth_gbps"] for ch in stats["channels"].values())
        data_bandwidth = stats["channels"]["R"]["bandwidth_gbps"] + stats["channels"]["W"]["bandwidth_gbps"]

        stats["summary"]["total_bandwidth_gbps"] = total_bandwidth
        stats["summary"]["data_bandwidth_gbps"] = data_bandwidth  # 仅R+W
        stats["summary"]["total_flits"] = sum(ch["flit_count"] for ch in stats["channels"].values())

        # 打印摘要
        print(f"\n统计摘要:")
        print(f"  模拟周期: {sim_cycles}")
        print(f"  模拟时间: {time_ns:.2f} ns")
        print(f"  总带宽: {total_bandwidth:.2f} GB/s")
        print(f"  数据带宽(R+W): {data_bandwidth:.2f} GB/s")
        print(f"\n各通道带宽:")
        for channel in ["AR", "R", "AW", "W", "B"]:
            ch = stats["channels"][channel]
            print(f"    {channel}: {ch['bandwidth_gbps']:6.2f} GB/s (flits={ch['flit_count']}, throttled={ch['throttled']})")

        return stats

    def run_all_tests(self):
        """运行所有测试场景"""
        scenarios = [
            "pure_read",      # 纯读测试
            "pure_write",     # 纯写测试
            "high_rate"       # 高注入率测试（最关键）
            # "mixed_balanced"  # 暂时跳过混合测试
        ]

        for scenario in scenarios:
            self.run_simulation(scenario)

        # 生成报告
        self.generate_report()

    def generate_report(self):
        """生成测试报告"""
        print(f"\n{'='*60}")
        print("生成测试报告")
        print(f"{'='*60}")

        # 保存原始数据
        report_file = self.result_dir / "test_results.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n原始数据已保存: {report_file}")

        # 生成汇总表格
        self.generate_summary_table()

        # 生成可视化图表
        self.generate_plots()

    def generate_summary_table(self):
        """生成汇总表格"""
        # 创建DataFrame
        rows = []
        for scenario, stats in self.results.items():
            row = {
                "场景": scenario,
                "总带宽(GB/s)": stats["summary"]["total_bandwidth_gbps"],
                "数据带宽(GB/s)": stats["summary"]["data_bandwidth_gbps"],
            }
            for channel in ["AR", "R", "AW", "W", "B"]:
                row[f"{channel}(GB/s)"] = stats["channels"][channel]["bandwidth_gbps"]
                row[f"{channel}_throttled"] = stats["channels"][channel]["throttled"]
            rows.append(row)

        df = pd.DataFrame(rows)

        # 保存CSV
        csv_file = self.result_dir / "summary_table.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"\n汇总表格已保存: {csv_file}")

        # 打印表格
        print("\n" + "="*80)
        print("测试结果汇总表")
        print("="*80)
        print(df.to_string(index=False))

    def generate_plots(self):
        """生成可视化图表"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 图1: 各场景的通道带宽对比
        ax1 = axes[0, 0]
        scenarios = list(self.results.keys())
        channels = ["AR", "R", "AW", "W", "B"]

        x = range(len(scenarios))
        width = 0.15

        for i, channel in enumerate(channels):
            bandwidths = [self.results[s]["channels"][channel]["bandwidth_gbps"] for s in scenarios]
            ax1.bar([xi + i*width for xi in x], bandwidths, width, label=channel)

        ax1.set_xlabel('测试场景')
        ax1.set_ylabel('带宽 (GB/s)')
        ax1.set_title('各场景AXI通道带宽对比')
        ax1.set_xticks([xi + width*2 for xi in x])
        ax1.set_xticklabels(scenarios, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 图2: 各场景的总带宽
        ax2 = axes[0, 1]
        total_bw = [self.results[s]["summary"]["total_bandwidth_gbps"] for s in scenarios]
        data_bw = [self.results[s]["summary"]["data_bandwidth_gbps"] for s in scenarios]

        x = range(len(scenarios))
        ax2.bar([xi - 0.2 for xi in x], total_bw, 0.4, label='总带宽(5通道)', alpha=0.8)
        ax2.bar([xi + 0.2 for xi in x], data_bw, 0.4, label='数据带宽(R+W)', alpha=0.8)
        ax2.axhline(y=256, color='r', linestyle='--', label='接口限制(256GB/s)')
        ax2.axhline(y=640, color='orange', linestyle='--', label='理论最大(640GB/s)')

        ax2.set_xlabel('测试场景')
        ax2.set_ylabel('带宽 (GB/s)')
        ax2.set_title('各场景总带宽')
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenarios, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 图3: TokenBucket限流统计
        ax3 = axes[1, 0]
        throttled_data = []
        for scenario in scenarios:
            for channel in channels:
                throttled_data.append({
                    'scenario': scenario,
                    'channel': channel,
                    'throttled': self.results[scenario]["channels"][channel]["throttled"]
                })

        df_throttled = pd.DataFrame(throttled_data)
        pivot = df_throttled.pivot(index='scenario', columns='channel', values='throttled')

        sns.heatmap(pivot, annot=True, fmt='g', cmap='YlOrRd', ax=ax3)
        ax3.set_title('TokenBucket限流次数热力图')
        ax3.set_xlabel('AXI通道')
        ax3.set_ylabel('测试场景')

        # 图4: 各通道带宽利用率
        ax4 = axes[1, 1]
        channel_config_bw = 128  # GB/s 配置的通道带宽

        utilization_data = []
        for scenario in scenarios:
            for channel in channels:
                actual_bw = self.results[scenario]["channels"][channel]["bandwidth_gbps"]
                utilization = (actual_bw / channel_config_bw) * 100
                utilization_data.append({
                    'scenario': scenario,
                    'channel': channel,
                    'utilization': utilization
                })

        df_util = pd.DataFrame(utilization_data)
        pivot_util = df_util.pivot(index='scenario', columns='channel', values='utilization')

        sns.heatmap(pivot_util, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax4, vmin=0, vmax=100)
        ax4.set_title('各通道带宽利用率 (%)')
        ax4.set_xlabel('AXI通道')
        ax4.set_ylabel('测试场景')

        plt.tight_layout()

        # 保存图表
        plot_file = self.result_dir / "bandwidth_analysis.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\n可视化图表已保存: {plot_file}")

        # 显示图表
        plt.show()


def main():
    """主函数"""
    # 使用D2D配置
    config_path = project_root / "config" / "topologies" / "d2d_2die_config.yaml"

    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return

    print("="*60)
    print("AXI通道带宽测试")
    print("="*60)
    print(f"配置文件: {config_path}")

    # 创建测试器
    tester = AXIBandwidthTester(str(config_path))

    # 运行所有测试
    tester.run_all_tests()

    print("\n" + "="*60)
    print("所有测试完成!")
    print("="*60)


if __name__ == "__main__":
    main()
