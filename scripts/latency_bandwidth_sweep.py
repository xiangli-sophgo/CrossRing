"""
延迟-带宽曲线扫描工具

通过扫描不同的带宽限制配置,生成延迟-带宽特性曲线,用于识别网络饱和点。
支持自适应带宽间隔(低带宽大间隔,高带宽小间隔)和可交互的Plotly可视化。

使用方法:
1. 修改main函数内的配置参数
2. 运行: python latency_bandwidth_sweep.py
3. 查看生成的HTML交互式图表和CSV原始数据
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.REQ_RSP import REQ_RSP_model
from config.config import CrossRingConfig
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def generate_bw_range(start, end, num_points):
    """生成带宽扫描范围,在start和end之间均匀分布num_points个点

    Args:
        start: 起始带宽值
        end: 结束带宽值
        num_points: 扫描点总数(包含start和end)

    Returns:
        list: 带宽扫描点列表

    Examples:
        generate_bw_range(50, 128, 10)  # 在50-128之间均匀分布10个点
        # 返回: [50.0, 58.67, 67.33, 76.0, 84.67, 93.33, 102.0, 110.67, 119.33, 128.0]
    """
    return np.linspace(start, end, num_points).tolist()


def _run_single_bw_point(args):
    """并行执行的单个带宽点仿真(独立函数,用于多进程)

    Args:
        args: (bw_limit, config_path, traffic_dir, traffic_file, model_type, target_ips, max_cycles)

    Returns:
        dict: 仿真结果
    """
    bw_limit, config_path, traffic_dir, traffic_file, model_type, target_ips, max_cycles = args

    try:
        # 加载配置
        config = CrossRingConfig(config_path)

        # 获取topo_type
        topo_type = config.TOPO_TYPE

        # 创建模型
        sim = REQ_RSP_model(model_type=model_type, config=config, topo_type=topo_type, verbose=0)

        # 设置traffic调度器
        sim.setup_traffic_scheduler(traffic_file_path=traffic_dir, traffic_chains=[[traffic_file]])

        # 应用流量级带宽限制(通过TrafficScheduler)
        sim.traffic_scheduler.setup_ip_bandwidth_limits(target_ips=target_ips, bw_limit=bw_limit)

        # 设置结果分析(不保存图表,只做统计)
        # 注意: result_save_path不能为空,否则process_comprehensive_results()会跳过统计计算
        import tempfile

        temp_result_dir = tempfile.mkdtemp(prefix="bw_sweep_")
        sim.setup_result_analysis(plot_flow_fig=False, plot_RN_BW_fig=False, fifo_utilization_heatmap=False, result_save_path=temp_result_dir, save_fig=False)

        # 运行仿真
        sim.run_simulation(max_time=max_cycles)

        # 获取仿真结果(这是标准方式)
        results = sim.get_results()

        # 从结果字典中提取数据(使用中文键名,不提供默认值以便发现错误)
        actual_bw = results["带宽_混合_平均加权"]
        cmd_latency_avg = results["命令延迟_混合_平均"]
        data_latency_avg = results["数据延迟_混合_平均"]
        trans_latency_avg = results["事务延迟_混合_平均"]
        cmd_latency_max = results["命令延迟_混合_最大"]
        data_latency_max = results["数据延迟_混合_最大"]
        trans_latency_max = results["事务延迟_混合_最大"]
        req_type = "mixed"
        count = results["发送读flit数"] + results["发送写flit数"]

        return {
            "bw_limit": bw_limit,
            "actual_bw": actual_bw,
            "cmd_latency_avg": cmd_latency_avg,
            "data_latency_avg": data_latency_avg,
            "trans_latency_avg": trans_latency_avg,
            "cmd_latency_max": cmd_latency_max,
            "data_latency_max": data_latency_max,
            "trans_latency_max": trans_latency_max,
            "req_type": req_type,
            "request_count": count,
        }
    except Exception as e:
        return {
            "bw_limit": bw_limit,
            "actual_bw": 0,
            "cmd_latency_avg": 0,
            "data_latency_avg": 0,
            "trans_latency_avg": 0,
            "cmd_latency_max": 0,
            "data_latency_max": 0,
            "trans_latency_max": 0,
            "req_type": "error",
            "request_count": 0,
            "error": str(e),
        }


class LatencyBandwidthSweeper:
    """延迟-带宽曲线扫描器"""

    def __init__(self, config_path, traffic_file, traffic_dir, save_dir, model_type, ip_config, max_cycles, verbose):
        """初始化扫描器

        Args:
            config_path: 拓扑配置文件路径
            traffic_file: 流量文件名
            traffic_dir: 流量文件目录
            save_dir: 结果保存目录
            model_type: 模型类型("REQ_RSP"或"Packet_Base")
            ip_config: IP带宽限制策略配置
            max_cycles: 最大仿真周期数
            verbose: 详细输出级别
        """
        self.config_path = config_path
        self.traffic_file = traffic_file
        self.traffic_dir = traffic_dir
        self.model_type = model_type
        self.ip_config = ip_config
        self.max_cycles = max_cycles
        self.verbose = verbose

        # 创建保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(save_dir) / f"sweep_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 结果存储
        self.results = []
        self.target_ips = []

        # 解析traffic获取目标IP
        if self.ip_config["mode"] == "auto":
            self.target_ips = self._parse_traffic_ips()
        else:
            self.target_ips = self.ip_config["target_ips"]

        print(f"扫描目标IP: {self.target_ips}")

    def _parse_traffic_ips(self):
        """从traffic文件自动识别涉及的IP类型

        Returns:
            List[str]: IP类型列表,如["GDMA", "DDR"]
        """
        traffic_path = Path(self.traffic_dir) / self.traffic_file
        if not traffic_path.exists():
            print(f"警告: traffic文件不存在 {traffic_path}, 使用默认IP配置")
            return ["GDMA", "DDR"]

        ip_types = set()

        try:
            with open(traffic_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    # 解析source和destination字段
                    # 格式示例: source: gdma_0, destination: ddr_0
                    source_match = re.search(r"source:\s*(\w+)_\d+", line)
                    dest_match = re.search(r"destination:\s*(\w+)_\d+", line)

                    if source_match:
                        ip_types.add(source_match.group(1).upper())
                    if dest_match:
                        ip_types.add(dest_match.group(1).upper())

        except Exception as e:
            print(f"警告: 解析traffic文件失败 {e}, 使用默认IP配置")
            return ["GDMA", "DDR"]

        if not ip_types:
            print("警告: 未从traffic文件解析到IP类型, 使用默认IP配置")
            return ["GDMA", "DDR"]

        return sorted(list(ip_types))

    def _apply_bw_limit(self, config, bw_limit):
        """应用带宽限制到配置

        Args:
            config: CrossRingConfig对象
            bw_limit: 带宽限制值(GB/s)
        """
        for ip_type in self.target_ips:
            attr_name = f"{ip_type}_BW_LIMIT"
            if hasattr(config, attr_name):
                setattr(config, attr_name, bw_limit)
            else:
                print(f"警告: 配置中不存在属性 {attr_name}")

    def _run_single_simulation(self, bw_limit):
        """运行单次仿真

        Args:
            bw_limit: 当前带宽限制值

        Returns:
            dict: 包含延迟和带宽统计的结果字典
        """
        # 加载配置
        config = CrossRingConfig(self.config_path)

        # 应用带宽限制
        self._apply_bw_limit(config, bw_limit)

        # 获取topo_type
        topo_type = config.TOPO_TYPE

        # 创建模型
        sim = REQ_RSP_model(model_type=self.model_type, config=config, topo_type=topo_type, verbose=self.verbose)

        # 设置traffic调度器
        sim.setup_traffic_scheduler(traffic_file_path=self.traffic_dir, traffic_chains=[[self.traffic_file]])

        # 设置结果分析(不保存图表,只做统计)
        # 注意: result_save_path不能为空,否则process_comprehensive_results()会跳过统计计算
        import tempfile

        temp_result_dir = tempfile.mkdtemp(prefix="bw_sweep_")
        sim.setup_result_analysis(plot_flow_fig=False, plot_RN_BW_fig=False, fifo_utilization_heatmap=False, result_save_path=temp_result_dir, save_fig=False)

        # 运行仿真
        print(f"  运行仿真: BW_LIMIT={bw_limit} GB/s", end="", flush=True)
        sim.run_simulation(max_time=self.max_cycles)
        print(f" - 完成")

        # 获取仿真结果(这是标准方式)
        results = sim.get_results()

        # 从结果字典中提取数据(使用中文键名,不提供默认值以便发现错误)
        actual_bw = results["带宽_混合_平均加权"]
        cmd_latency_avg = results["命令延迟_混合_平均"]
        data_latency_avg = results["数据延迟_混合_平均"]
        trans_latency_avg = results["事务延迟_混合_平均"]
        cmd_latency_max = results["命令延迟_混合_最大"]
        data_latency_max = results["数据延迟_混合_最大"]
        trans_latency_max = results["事务延迟_混合_最大"]
        req_type = "mixed"
        count = results["发送读flit数"] + results["发送写flit数"]

        return {
            "bw_limit": bw_limit,
            "actual_bw": actual_bw,
            "cmd_latency_avg": cmd_latency_avg,
            "data_latency_avg": data_latency_avg,
            "trans_latency_avg": trans_latency_avg,
            "cmd_latency_max": cmd_latency_max,
            "data_latency_max": data_latency_max,
            "trans_latency_max": trans_latency_max,
            "req_type": req_type,
            "request_count": count,
        }

    def run_sweep(self, bw_ranges, n_workers=None):
        """批量扫描带宽范围(支持并行)

        Args:
            bw_ranges: 带宽扫描点列表
            n_workers: 并行进程数,None表示使用CPU核心数
        """
        print(f"\n开始扫描 {len(bw_ranges)} 个带宽点...")
        print(f"带宽范围: {bw_ranges[0]} - {bw_ranges[-1]} GB/s")

        if n_workers is None:
            n_workers = multiprocessing.cpu_count()

        if n_workers == 1:
            # 串行模式
            print(f"使用串行模式\n")
            for i, bw_limit in enumerate(bw_ranges, 1):
                print(f"[{i}/{len(bw_ranges)}]", end=" ")
                try:
                    result = self._run_single_simulation(bw_limit)
                    self.results.append(result)
                except Exception as e:
                    print(f"  错误: {e}")
                    self.results.append(
                        {
                            "bw_limit": bw_limit,
                            "actual_bw": 0,
                            "cmd_latency_avg": 0,
                            "data_latency_avg": 0,
                            "trans_latency_avg": 0,
                            "cmd_latency_max": 0,
                            "data_latency_max": 0,
                            "trans_latency_max": 0,
                            "req_type": "error",
                            "request_count": 0,
                        }
                    )
        else:
            # 并行模式
            print(f"使用并行模式 ({n_workers} 个进程)\n")

            # 准备并行任务参数
            tasks = [(bw, self.config_path, self.traffic_dir, self.traffic_file, self.model_type, self.target_ips, self.max_cycles) for bw in bw_ranges]

            # 使用进程池执行
            completed = 0
            submitted = 0
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # 提交所有任务
                future_to_bw = {}
                for task in tasks:
                    bw_limit = task[0]
                    future = executor.submit(_run_single_bw_point, task)
                    future_to_bw[future] = bw_limit
                    submitted += 1
                    print(f"[提交 {submitted}/{len(bw_ranges)}] 开始运行 BW={bw_limit:.1f} GB/s 的仿真...")

                print(f"\n所有任务已提交,等待完成...\n")

                # 收集结果
                for future in as_completed(future_to_bw):
                    bw_limit = future_to_bw[future]
                    completed += 1

                    try:
                        result = future.result()
                        self.results.append(result)
                        if "error" in result:
                            print(f"[完成 {completed}/{len(bw_ranges)}] 带宽限制={bw_limit:.1f} GB/s - 错误: {result.get('error', '未知错误')}")
                        else:
                            print(f"[完成 {completed}/{len(bw_ranges)}] 带宽限制={bw_limit:.1f} GB/s - 已完成 (平均加权带宽={result['actual_bw']:.2f} GB/s)")
                    except Exception as e:
                        print(f"[完成 {completed}/{len(bw_ranges)}] 带宽限制={bw_limit:.1f} GB/s - 异常: {e}")
                        self.results.append(
                            {
                                "bw_limit": bw_limit,
                                "actual_bw": 0,
                                "cmd_latency_avg": 0,
                                "data_latency_avg": 0,
                                "trans_latency_avg": 0,
                                "cmd_latency_max": 0,
                                "data_latency_max": 0,
                                "trans_latency_max": 0,
                                "req_type": "error",
                                "request_count": 0,
                            }
                        )

            # 按bw_limit排序结果
            self.results.sort(key=lambda x: x["bw_limit"])

        print("\n扫描完成!")

    def save_results(self):
        """保存结果到CSV和JSON

        Returns:
            tuple: (csv_path, config_path)
        """
        # 保存CSV
        df = pd.DataFrame(self.results)
        csv_path = self.save_dir / "sweep_results.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n结果已保存到: {csv_path}")

        # 保存配置
        config_data = {
            "config_path": str(self.config_path),
            "traffic_file": self.traffic_file,
            "traffic_dir": str(self.traffic_dir),
            "model_type": self.model_type,
            "max_cycles": self.max_cycles,
            "target_ips": self.target_ips,
            "ip_bw_config": self.ip_config,
            "bw_ranges": [r["bw_limit"] for r in self.results],
            "timestamp": datetime.now().isoformat(),
        }

        config_path = self.save_dir / "sweep_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        print(f"配置已保存到: {config_path}")

        return csv_path, config_path

    def detect_saturation_point(self, percentile=90):
        """检测饱和点

        Args:
            percentile: 延迟增长率百分位数阈值

        Returns:
            int or None: 饱和点索引,如果未检测到则返回None
        """
        if len(self.results) < 3:
            return None

        trans_latencies = np.array([r["trans_latency_avg"] for r in self.results])
        bw_limits = np.array([r["bw_limit"] for r in self.results])

        # 计算延迟增长率
        latency_growth = np.diff(trans_latencies) / np.diff(bw_limits)

        # 找到增长率突增的点
        threshold = np.percentile(latency_growth, percentile)
        saturation_candidates = np.where(latency_growth > threshold)[0]

        if len(saturation_candidates) > 0:
            return saturation_candidates[0]

        return None

    def create_interactive_plot(self, saturation_config=None):
        """生成Plotly交互式图表

        Args:
            saturation_config: 饱和点检测配置

        Returns:
            str: HTML文件路径
        """
        if not self.results:
            print("警告: 没有结果数据,无法生成图表")
            return None

        df = pd.DataFrame(self.results)

        # 创建双Y轴图表
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # 实际带宽曲线(主Y轴)
        fig.add_trace(
            go.Scatter(
                x=df["bw_limit"],
                y=df["actual_bw"],
                mode="lines+markers",
                name="实际带宽",
                line=dict(color="blue", width=3),
                marker=dict(size=8, symbol="circle"),
                hovertemplate="<b>带宽限制</b>: %{x:.1f} GB/s<br><b>实际带宽</b>: %{y:.2f} GB/s<extra></extra>",
            ),
            secondary_y=False,
        )

        # Cmd延迟曲线(副Y轴)
        fig.add_trace(
            go.Scatter(
                x=df["bw_limit"],
                y=df["cmd_latency_avg"],
                mode="lines+markers",
                name="Cmd延迟",
                line=dict(color="green", width=2, dash="dash"),
                marker=dict(size=6, symbol="square"),
                hovertemplate="<b>Cmd延迟</b>: %{y:.1f} cycles<extra></extra>",
            ),
            secondary_y=True,
        )

        # Data延迟曲线(副Y轴)
        fig.add_trace(
            go.Scatter(
                x=df["bw_limit"],
                y=df["data_latency_avg"],
                mode="lines+markers",
                name="Data延迟",
                line=dict(color="orange", width=2, dash="dash"),
                marker=dict(size=6, symbol="triangle-up"),
                hovertemplate="<b>Data延迟</b>: %{y:.1f} cycles<extra></extra>",
            ),
            secondary_y=True,
        )

        # Transaction延迟曲线(副Y轴)
        fig.add_trace(
            go.Scatter(
                x=df["bw_limit"],
                y=df["trans_latency_avg"],
                mode="lines+markers",
                name="Transaction延迟",
                line=dict(color="red", width=2, dash="dot"),
                marker=dict(size=8, symbol="diamond"),
                hovertemplate="<b>Trans延迟</b>: %{y:.1f} cycles<extra></extra>",
            ),
            secondary_y=True,
        )

        # 检测并标注饱和点
        if saturation_config and saturation_config.get("enabled", False):
            saturation_idx = self.detect_saturation_point(percentile=saturation_config.get("percentile", 90))

            if saturation_idx is not None:
                sat_bw = df["bw_limit"].iloc[saturation_idx]
                sat_latency = df["trans_latency_avg"].iloc[saturation_idx]

                fig.add_annotation(
                    x=sat_bw,
                    y=sat_latency,
                    text=f"疑似饱和点<br>{sat_bw:.1f} GB/s",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="red",
                    ax=-50,
                    ay=-50,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="red",
                    borderwidth=2,
                    yref="y2",
                )

                print(f"\n检测到饱和点: {sat_bw:.1f} GB/s")

        # 更新布局
        fig.update_xaxes(title_text="带宽限制 (GB/s)")
        fig.update_yaxes(title_text="实际带宽 (GB/s)", secondary_y=False)
        fig.update_yaxes(title_text="延迟 (cycles)", secondary_y=True)

        fig.update_layout(
            title=dict(text=f"延迟-带宽特性曲线<br><sub>数据流: {self.traffic_file.replace('.txt', '')}</sub>", x=0.5, xanchor="center"),
            hovermode="x unified",
            width=1400,
            height=700,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1),
        )

        # 保存HTML
        html_path = self.save_dir / "latency_bandwidth_curve.html"
        fig.write_html(html_path)
        print(f"交互式图表已保存到: {html_path}")

        return str(html_path)


def main():
    """主函数"""

    # ==================== 配置区域 ====================

    # 基础配置
    CONFIG_PATH = "../config/topologies/topo_5x4.yaml"
    TRAFFIC_DIR = r"../traffic/DeepSeek_0616/step6_ch_map/"
    TRAFFIC_FILE = "LLama2_AllReduce.txt"
    SAVE_DIR = "../Result/LatencyBandwidthSweep"
    MODEL_TYPE = "REQ_RSP"  # "REQ_RSP" 或 "Packet_Base"

    # 带宽扫描范围配置(单位: GB/s)
    bw_ranges = generate_bw_range(20, 128, 40)  # 在50-128GB/s之间均匀分布10个点

    # IP带宽限制策略
    IP_BW_CONFIG = {"mode": "auto", "sync": True, "target_ips": []}  # "auto"=自动解析traffic, "manual"=手动指定  # True=所有IP同步调整, False=独立调整  # manual模式下手动指定,如["GDMA", "DDR"]

    # 仿真参数
    MAX_CYCLES = 10000
    VERBOSE = 0

    # 并行配置
    N_WORKERS = 6  # None=使用CPU核心数, 1=串行, >1=指定进程数

    # 饱和点检测配置
    SATURATION_DETECT = {"enabled": False, "percentile": 90}  # 是否检测饱和点  # 延迟增长率超过该百分位数视为饱和

    # ==================================================

    print("=" * 60)
    print("延迟-带宽曲线扫描工具")
    print("=" * 60)
    print(f"\n带宽扫描范围: {len(bw_ranges)} 个扫描点")
    print(f"扫描点: {bw_ranges}")

    # 创建扫描器
    sweeper = LatencyBandwidthSweeper(
        config_path=CONFIG_PATH, traffic_file=TRAFFIC_FILE, traffic_dir=TRAFFIC_DIR, save_dir=SAVE_DIR, model_type=MODEL_TYPE, ip_config=IP_BW_CONFIG, max_cycles=MAX_CYCLES, verbose=VERBOSE
    )

    # 运行扫描(支持并行)
    sweeper.run_sweep(bw_ranges, n_workers=N_WORKERS)

    # 保存结果
    sweeper.save_results()

    # 生成可视化
    sweeper.create_interactive_plot(saturation_config=SATURATION_DETECT)

    print("\n" + "=" * 60)
    print("扫描完成!")
    print(f"结果目录: {sweeper.save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
