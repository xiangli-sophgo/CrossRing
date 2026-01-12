"""
统计与调试相关的Mixin类
包含统计结果收集、调试输出、可视化等功能
"""

import os
import time
import inspect
import matplotlib.pyplot as plt

from src.utils.flit import Flit


class StatsMixin:
    """统计与调试功能Mixin"""

    def update_finish_time_stats(self):
        """从traffic_scheduler和result_processor获取结束时间并更新统计"""
        read_end_times = []
        write_end_times = []
        all_end_times = []

        # 从traffic_scheduler获取结束时间统计
        try:
            finish_stats = self.traffic_scheduler.get_finish_time_stats()
            if finish_stats["R_finish_time"] > 0:
                read_end_times.append(finish_stats["R_finish_time"])
                all_end_times.append(finish_stats["R_finish_time"])
            if finish_stats["W_finish_time"] > 0:
                write_end_times.append(finish_stats["W_finish_time"])
                all_end_times.append(finish_stats["W_finish_time"])
            if finish_stats["Total_finish_time"] > 0:
                all_end_times.append(finish_stats["Total_finish_time"])
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not get finish time stats from traffic_scheduler: {e}")

        # 从result_processor获取请求的结束时间
        try:
            if hasattr(self, "result_processor") and hasattr(self.result_processor, "requests"):
                for req_info in self.result_processor.requests:
                    end_time_ns = req_info.end_time // self.config.CYCLES_PER_NS
                    all_end_times.append(end_time_ns)
                    if req_info.req_type == "read":
                        read_end_times.append(end_time_ns)
                    elif req_info.req_type == "write":
                        write_end_times.append(end_time_ns)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not get finish time stats from result_processor: {e}")

        # 更新统计数据，使用当前cycle作为备选
        current_time_ns = self.cycle // self.config.CYCLES_PER_NS

        if read_end_times:
            self.R_finish_time_stat = max(read_end_times)
        else:
            self.R_finish_time_stat = current_time_ns

        if write_end_times:
            self.W_finish_time_stat = max(write_end_times)
        else:
            self.W_finish_time_stat = current_time_ns

        if all_end_times:
            self.Total_finish_time_stat = max(all_end_times)
        else:
            self.Total_finish_time_stat = current_time_ns

        if self.verbose:
            print(f"Updated finish times - Read: {self.R_finish_time_stat}ns, Write: {self.W_finish_time_stat}ns, Total: {self.Total_finish_time_stat}ns")

    def update_traffic_completion_stats(self, flit):
        """在flit完成时更新TrafficScheduler的统计"""
        # 只有当 flit 真正到达 IP_RX 状态时才更新统计
        if hasattr(flit, "traffic_id") and flit.flit_position.startswith("IP_RX"):
            self.traffic_scheduler.update_traffic_stats(flit.traffic_id, "received_flit")

    def syn_IP_stat(self):
        # 直接遍历实际创建的IP接口(动态挂载模式)
        for (ip_type, node_id), ip_interface in self.ip_modules.items():
            if self.model_type_stat == "REQ_RSP":
                self.read_retry_num_stat += ip_interface.read_retry_num_stat
                self.write_retry_num_stat += ip_interface.write_retry_num_stat
            self.req_cir_h_num_stat += ip_interface.req_cir_h_num
            self.req_cir_v_num_stat += ip_interface.req_cir_v_num
            self.rsp_cir_h_num_stat += ip_interface.rsp_cir_h_num
            self.rsp_cir_v_num_stat += ip_interface.rsp_cir_v_num
            self.data_cir_h_num_stat += ip_interface.data_cir_h_num
            self.data_cir_v_num_stat += ip_interface.data_cir_v_num
            # 反方向上环统计汇总
            self.req_reverse_h_num_stat += ip_interface.req_reverse_h_num
            self.req_reverse_v_num_stat += ip_interface.req_reverse_v_num
            self.rsp_reverse_h_num_stat += ip_interface.rsp_reverse_h_num
            self.rsp_reverse_v_num_stat += ip_interface.rsp_reverse_v_num
            self.data_reverse_h_num_stat += ip_interface.data_reverse_h_num
            self.data_reverse_v_num_stat += ip_interface.data_reverse_v_num
            self.req_wait_cycle_h_num_stat += ip_interface.req_wait_cycles_h
            self.req_wait_cycle_v_num_stat += ip_interface.req_wait_cycles_v
            self.rsp_wait_cycle_h_num_stat += ip_interface.rsp_wait_cycles_h
            self.rsp_wait_cycle_v_num_stat += ip_interface.rsp_wait_cycles_v
            self.data_wait_cycle_h_num_stat += ip_interface.data_wait_cycles_h
            self.data_wait_cycle_v_num_stat += ip_interface.data_wait_cycles_v

    def debug_func(self):
        if self.print_trace:
            self.flit_trace(self.show_trace_id)
        if self.plot_link_state:
            # 检查停止信号 - 如果已停止则完全跳过可视化
            if self.link_state_vis.should_stop:
                return
            if self.cycle < self.plot_start_cycle:
                return

            # 暂停阻塞机制
            while self.link_state_vis.paused and not self.link_state_vis.should_stop:
                plt.pause(0.05)

            # 再次检查停止信号（可能在暂停期间被设置）
            if self.link_state_vis.should_stop:
                return

            # v2 架构：同步 IP 模块的 channel buffer 到 network 对象
            self._sync_channel_buffers_for_visualization()


            try:
                self.link_state_vis.update([self.req_network, self.rsp_network, self.data_network], self.cycle)
            except Exception as e:
                # 窗口已关闭，设置停止标志
                self.link_state_vis.should_stop = True

    def _sync_channel_buffers_for_visualization(self):
        """将 IP 模块的 channel buffer 数据同步到 network 对象，供可视化使用"""
        from collections import defaultdict

        for network in [self.req_network, self.rsp_network, self.data_network]:
            # 初始化 channel buffer 结构（如果不存在）
            if not hasattr(network, "IQ_channel_buffer"):
                network.IQ_channel_buffer = defaultdict(dict)
            if not hasattr(network, "EQ_channel_buffer"):
                network.EQ_channel_buffer = defaultdict(dict)

            # 确定网络类型
            if network == self.req_network:
                net_type = "req"
            elif network == self.rsp_network:
                net_type = "rsp"
            else:
                net_type = "data"

            # 从 IP 模块同步数据
            for (ip_type, node_id), ip_interface in self.ip_modules.items():
                net_info = ip_interface.networks.get(net_type)
                if net_info is None:
                    continue

                # IQ: IP发送方向 (tx_channel_buffer_pre)
                # 使用列表包装单个 flit，以便可视化器可以迭代
                tx_pre = net_info.get("tx_channel_buffer_pre")
                network.IQ_channel_buffer[ip_type][node_id] = [tx_pre] if tx_pre is not None else []

                # EQ: IP接收方向 (rx_channel_buffer)
                rx_buf = net_info.get("rx_channel_buffer")
                network.EQ_channel_buffer[ip_type][node_id] = list(rx_buf) if rx_buf else []

    def error_log(self, flit, target_id, flit_id):
        if flit and flit.packet_id == target_id and flit.flit_id == flit_id:
            print(inspect.currentframe().f_back.f_code.co_name, self.cycle, flit)

    def flit_trace(self, packet_id):
        """打印指定 packet_id 或 packet_id 列表的调试信息"""
        if self.plot_link_state and self.link_state_vis.should_stop:
            return
        # 统一处理 packet_id（兼容单个值或列表）
        packet_ids = [packet_id] if isinstance(packet_id, (int, str)) else packet_id

        for pid in packet_ids:
            self._debug_print(self.req_network, "req", pid)
            self._debug_print(self.rsp_network, "rsp", pid)
            self._debug_print(self.data_network, "flit", pid)

    def _should_skip_waiting_flit(self, flit) -> bool:
        """判断flit是否在等待状态，不需要打印"""
        if hasattr(flit, "flit_position"):
            # IP_TX 状态算等待状态
            if flit.flit_position == "IP_TX":
                return True
            # L2H状态且还未到departure时间 = 等待状态
            if flit.flit_position == "L2H" and hasattr(flit, "departure_cycle") and flit.departure_cycle > self.cycle:
                return True
            # IP_RX状态且位置没有变化，也算等待状态
            if flit.flit_position == "IP_RX":
                # 使用外部字典跟踪flit的稳定周期（避免修改Flit类的__slots__）
                if not hasattr(self, "_flit_stable_cycles"):
                    self._flit_stable_cycles = {}

                flit_key = f"{flit.packet_id}_{flit.flit_id}"
                if flit_key in self._flit_stable_cycles:
                    if self.cycle - self._flit_stable_cycles[flit_key] > 2:  # 在IP_RX超过2个周期就跳过
                        return True
                else:
                    self._flit_stable_cycles[flit_key] = self.cycle
        return False

    def _debug_print(self, net, net_type, packet_id):
        # 取出所有 flit
        flits = net.send_flits.get(packet_id)
        if not flits:
            return

        # 如果这个 packet_id 已经标记完成，直接跳过
        packet_done_key = f"{packet_id}_{net_type}"
        if self._done_flags.get(packet_done_key, False):
            return

        # 检查是否有活跃的flit（非等待状态的flit）
        has_active_flit = any(not self._should_skip_waiting_flit(flit) for flit in flits)

        # 对于单 flit 的 Retry rsp，到达后不打印也不更新状态
        if net_type == "rsp":
            last_flit = flits[-1]
            if last_flit.rsp_type == "Retry" and len(flits) == 1 and last_flit.is_finish:
                return

        # 只有当有活跃flit时才打印
        if has_active_flit:
            # —— 到这里，说明需要打印调试信息 ——
            if self.cycle != self._last_printed_cycle:
                print(f"Cycle {self.cycle}:")  # 醒目标记当前 cycle
                self._last_printed_cycle = self.cycle  # 更新记录

            # 收集所有flit并格式化打印
            all_flits = []

            # REQ网络的flit
            req_flits = self.req_network.send_flits.get(packet_id, [])
            for flit in req_flits:
                all_flits.append(f"REQ,{flit}")

            # RSP网络的flit
            rsp_flits = self.rsp_network.send_flits.get(packet_id, [])
            for flit in rsp_flits:
                all_flits.append(f"RSP,{flit}")

            # DATA网络的flit
            data_flits = self.data_network.send_flits.get(packet_id, [])
            for flit in data_flits:
                all_flits.append(f"DATA,{flit}")

            # 打印所有flit，用 | 分隔
            if all_flits:
                print(" | ".join(all_flits) + " |")

        # —— 更新完成标记 ——
        # 检查所有 flit 是否都已到达 IP_RX 状态
        all_at_ip_rx = all(f.flit_position == "IP_RX" for f in flits)

        if net_type == "rsp":
            # 只有最后一个 DBID 到达 IP_RX 时才算完成
            last_flit = flits[-1]
            if last_flit.rsp_type == "DBID" and last_flit.flit_position == "IP_RX":
                self._done_flags[packet_done_key] = True
        else:
            # 其他网络类型，所有 flit 都到达 IP_RX 才算完成
            if all_at_ip_rx:
                self._done_flags[packet_done_key] = True

        # 只有在实际打印了信息时才执行sleep
        if has_active_flit and self.update_interval > 0:
            time.sleep(self.update_interval)

    def process_comprehensive_results(self):
        """处理综合统计结果，所有结果收集到内存（不写本地文件）"""
        self.result_processor.collect_requests_data(self, self.cycle)
        results = self.result_processor.analyze_all_bandwidth()

        # 初始化文件内容收集器
        self._result_file_contents = {}

        # 收集CSV内容
        csv_contents = self.result_processor.generate_unified_report(results, return_content=True)
        if csv_contents:
            self._result_file_contents.update(csv_contents)

        # 收集tracker使用数据
        from src.analysis.data_collectors import TrackerDataCollector

        self._tracker_collector = TrackerDataCollector()
        self._tracker_collector.collect_tracker_data(self)

        tracker_json = self._tracker_collector.save_to_json(return_content=True)
        if tracker_json:
            self._result_file_contents["tracker_data.json"] = tracker_json

        self.Total_sum_BW_stat = results["Total_sum_BW"]

        # 额外带宽统计
        read_metrics = results["network_overall"]["read"]
        write_metrics = results["network_overall"]["write"]
        # 非加权 / 加权 带宽
        self.read_unweighted_bw_stat = read_metrics.unweighted_bandwidth
        self.read_weighted_bw_stat = read_metrics.weighted_bandwidth
        self.write_unweighted_bw_stat = write_metrics.unweighted_bandwidth
        self.write_weighted_bw_stat = write_metrics.weighted_bandwidth

        # 延迟统计
        latency_stats = self.result_processor._calculate_latency_stats()

        # FIFO使用率统计
        from src.analysis.data_collectors import CircuitStatsCollector
        fifo_collector = CircuitStatsCollector()
        fifo_csv = fifo_collector.generate_fifo_usage_csv(self, return_content=True)
        if fifo_csv:
            self._result_file_contents["fifo_usage_statistics.csv"] = fifo_csv

        # 波形数据导出到Parquet（内存）
        if hasattr(self, 'request_tracker'):
            try:
                from src.analysis.exporters import ParquetExporter
                parquet_exporter = ParquetExporter(
                    network_frequency=self.config.NETWORK_FREQUENCY if hasattr(self.config, 'NETWORK_FREQUENCY') else 2.0
                )
                parquet_contents = parquet_exporter.export_waveform_data_to_bytes(self)
                if parquet_contents:
                    self._result_file_contents.update(parquet_contents)
            except Exception as e:
                if self.verbose:
                    print(f"Parquet导出错误: {e}")
        # CMD 延迟
        self.cmd_read_avg_latency_stat = (latency_stats["cmd"]["read"]["sum"] / latency_stats["cmd"]["read"]["count"]) if latency_stats["cmd"]["read"]["count"] else 0.0
        self.cmd_read_max_latency_stat = latency_stats["cmd"]["read"]["max"]
        self.cmd_read_p95_latency_stat = latency_stats["cmd"]["read"]["p95"]
        self.cmd_read_p99_latency_stat = latency_stats["cmd"]["read"]["p99"]
        self.cmd_write_avg_latency_stat = (latency_stats["cmd"]["write"]["sum"] / latency_stats["cmd"]["write"]["count"]) if latency_stats["cmd"]["write"]["count"] else 0.0
        self.cmd_write_max_latency_stat = latency_stats["cmd"]["write"]["max"]
        self.cmd_write_p95_latency_stat = latency_stats["cmd"]["write"]["p95"]
        self.cmd_write_p99_latency_stat = latency_stats["cmd"]["write"]["p99"]
        # Data 延迟
        self.data_read_avg_latency_stat = (latency_stats["data"]["read"]["sum"] / latency_stats["data"]["read"]["count"]) if latency_stats["data"]["read"]["count"] else 0.0
        self.data_read_max_latency_stat = latency_stats["data"]["read"]["max"]
        self.data_read_p95_latency_stat = latency_stats["data"]["read"]["p95"]
        self.data_read_p99_latency_stat = latency_stats["data"]["read"]["p99"]
        self.data_write_avg_latency_stat = (latency_stats["data"]["write"]["sum"] / latency_stats["data"]["write"]["count"]) if latency_stats["data"]["write"]["count"] else 0.0
        self.data_write_max_latency_stat = latency_stats["data"]["write"]["max"]
        self.data_write_p95_latency_stat = latency_stats["data"]["write"]["p95"]
        self.data_write_p99_latency_stat = latency_stats["data"]["write"]["p99"]
        # Transaction 延迟
        self.trans_read_avg_latency_stat = (latency_stats["trans"]["read"]["sum"] / latency_stats["trans"]["read"]["count"]) if latency_stats["trans"]["read"]["count"] else 0.0
        self.trans_read_max_latency_stat = latency_stats["trans"]["read"]["max"]
        self.trans_read_p95_latency_stat = latency_stats["trans"]["read"]["p95"]
        self.trans_read_p99_latency_stat = latency_stats["trans"]["read"]["p99"]
        self.trans_write_avg_latency_stat = (latency_stats["trans"]["write"]["sum"] / latency_stats["trans"]["write"]["count"]) if latency_stats["trans"]["write"]["count"] else 0.0
        self.trans_write_max_latency_stat = latency_stats["trans"]["write"]["max"]
        self.trans_write_p95_latency_stat = latency_stats["trans"]["write"]["p95"]
        self.trans_write_p99_latency_stat = latency_stats["trans"]["write"]["p99"]

        # Mixed 带宽统计
        mixed_metrics = results["network_overall"]["mixed"]
        self.mixed_unweighted_bw_stat = mixed_metrics.unweighted_bandwidth
        self.mixed_weighted_bw_stat = mixed_metrics.weighted_bandwidth
        # Total average bandwidth stats (unweighted and weighted)
        # 使用result_processor中动态计算的实际IP数量
        actual_num_ip = self.result_processor.actual_num_ip or 1  # 避免除零错误
        self.mixed_avg_unweighted_bw_stat = mixed_metrics.unweighted_bandwidth / actual_num_ip
        self.mixed_avg_weighted_bw_stat = mixed_metrics.weighted_bandwidth / actual_num_ip

        # Mixed 延迟统计
        # CMD 混合
        self.cmd_mixed_avg_latency_stat = (latency_stats["cmd"]["mixed"]["sum"] / latency_stats["cmd"]["mixed"]["count"]) if latency_stats["cmd"]["mixed"]["count"] else 0.0
        self.cmd_mixed_max_latency_stat = latency_stats["cmd"]["mixed"]["max"]
        self.cmd_mixed_p95_latency_stat = latency_stats["cmd"]["mixed"]["p95"]
        self.cmd_mixed_p99_latency_stat = latency_stats["cmd"]["mixed"]["p99"]
        # Data 混合
        self.data_mixed_avg_latency_stat = (latency_stats["data"]["mixed"]["sum"] / latency_stats["data"]["mixed"]["count"]) if latency_stats["data"]["mixed"]["count"] else 0.0
        self.data_mixed_max_latency_stat = latency_stats["data"]["mixed"]["max"]
        self.data_mixed_p95_latency_stat = latency_stats["data"]["mixed"]["p95"]
        self.data_mixed_p99_latency_stat = latency_stats["data"]["mixed"]["p99"]
        # Trans 混合
        self.trans_mixed_avg_latency_stat = (latency_stats["trans"]["mixed"]["sum"] / latency_stats["trans"]["mixed"]["count"]) if latency_stats["trans"]["mixed"]["count"] else 0.0
        self.trans_mixed_max_latency_stat = latency_stats["trans"]["mixed"]["max"]
        self.trans_mixed_p95_latency_stat = latency_stats["trans"]["mixed"]["p95"]
        self.trans_mixed_p99_latency_stat = latency_stats["trans"]["mixed"]["p99"]

        # FIFO使用率热力图 - 收集到结果处理器中
        if getattr(self, "fifo_utilization_heatmap", False):
            try:
                from src.analysis.fifo_heatmap_visualizer import create_fifo_heatmap

                # 计算总周期数(使用物理周期数,因为depth_sum在每个物理周期累加)
                total_cycles = self.cycle

                # 构造dies字典（单Die情况）
                dies = {0: self}

                # 获取Figure和JavaScript（不保存）
                fifo_fig, fifo_js = create_fifo_heatmap(
                    dies=dies, config=self.config, total_cycles=total_cycles, die_layout=None, die_rotations=None, save_path=None, show_fig=False, return_fig_and_js=True
                )

                # 将FIFO图表添加到结果处理器的图表列表中
                if not hasattr(self.result_processor, "charts_to_merge"):
                    self.result_processor.charts_to_merge = []
                # FIFO热力图插入到流量图和RN带宽之间（索引1位置）
                if len(self.result_processor.charts_to_merge) > 0:
                    self.result_processor.charts_to_merge.insert(1, ("FIFO使用率热力图", fifo_fig, fifo_js))
                else:
                    self.result_processor.charts_to_merge.append(("FIFO使用率热力图", fifo_fig, fifo_js))

            except Exception as e:
                if self.verbose:
                    print(f"警告: FIFO使用率热力图生成失败: {e}")

        # 生成集成HTML（所有图表收集完毕后，始终收集到内存）
        self._generate_integrated_visualization(save_to_db_only=True)

    def _generate_integrated_visualization(self, save_to_db_only: bool = False):
        """生成集成的可视化HTML报告

        Args:
            save_to_db_only: 如果为True，不写本地文件，只收集HTML内容
        """
        if not hasattr(self.result_processor, "charts_to_merge"):
            return

        all_charts = self.result_processor.charts_to_merge
        if not all_charts:
            return

        # 确保顺序：流量图 → FIFO热力图 → RN带宽曲线 → 延迟分布图 → 结果分析
        ordered_charts = []
        flow_chart = None
        fifo_chart = None
        rn_chart = None
        bandwidth_report = None
        latency_charts = []  # 延迟分布图列表

        for title, fig, custom_js in all_charts:
            if "流量图" in title:
                flow_chart = (title, fig, custom_js)
            elif "FIFO" in title:
                fifo_chart = (title, fig, custom_js)
            elif "结果分析" in title or "带宽分析报告" in title:
                bandwidth_report = (title, fig, custom_js)
            elif "RN" in title or "带宽曲线" in title:
                rn_chart = (title, fig, custom_js)
            elif "延迟分布" in title:
                latency_charts.append((title, fig, custom_js))

        # 按顺序添加
        if flow_chart:
            ordered_charts.append(flow_chart)
        if fifo_chart:
            ordered_charts.append(fifo_chart)
        if rn_chart:
            ordered_charts.append(rn_chart)
        # 添加所有延迟分布图
        ordered_charts.extend(latency_charts)
        if bandwidth_report:
            ordered_charts.append(bandwidth_report)

        if not ordered_charts:
            return

        try:
            from src.analysis.integrated_visualizer import create_integrated_report

            if save_to_db_only:
                # 不写文件，直接获取HTML内容
                html_content = create_integrated_report(
                    charts_config=ordered_charts,
                    save_path=None,
                    show_result_analysis=self.show_result_analysis,
                    return_content=True
                )

                if html_content:
                    # 注入tracker功能到HTML内容
                    if hasattr(self, "_result_file_contents") and "tracker_data.json" in self._result_file_contents:
                        try:
                            from src.analysis.tracker_html_injector import inject_tracker_functionality_to_content
                            tracker_json = self._result_file_contents["tracker_data.json"]
                            html_content = inject_tracker_functionality_to_content(html_content, tracker_json)
                        except Exception:
                            pass

                    # 保存HTML内容
                    self._result_html_content = html_content
                    if hasattr(self, "_result_file_contents"):
                        self._result_file_contents["result_analysis.html"] = html_content
            else:
                # 写本地文件（原有行为）
                if self.result_save_path:
                    save_path = f"{self.result_save_path}result_analysis.html"
                else:
                    return

                # 生成集成HTML
                integrated_path = create_integrated_report(charts_config=ordered_charts, save_path=save_path, show_result_analysis=self.show_result_analysis)

                if integrated_path and self.verbose:
                    print(f"结果分析报告: {integrated_path}")

                # HTML生成完成后，注入tracker交互功能
                if integrated_path and hasattr(self, "_tracker_json_path") and os.path.exists(self._tracker_json_path):
                    try:
                        from src.analysis.tracker_html_injector import inject_tracker_functionality

                        inject_tracker_functionality(integrated_path, self._tracker_json_path)
                    except Exception as e:
                        if self.verbose:
                            print(f"警告: Tracker交互功能注入失败: {e}")

        except Exception as e:
            if self.verbose:
                print(f"警告: 集成HTML生成失败: {e}")
                import traceback

                traceback.print_exc()

    def get_results(self):
        """
        Extract simulation statistics and configuration variables.

        Returns:
            dict: A combined dictionary of configuration variables and statistics.
        """
        # 需要在这里导入以避免循环导入
        from src.kcin.v2.base_model import BaseModel

        # Get all variables from the sim instance
        sim_vars = vars(self)

        # Extract statistics (ending with "_stat") and translate to Chinese
        stat_name_map = {
            "model_type": "模型类型",
            "topo_type": "拓扑类型",
            "file_name": "数据流名称",
            "send_read_flits_num": "总读flit数",
            "send_write_flits_num": "总写flit数",
            "R_finish_time": "读完成时间",
            "W_finish_time": "写完成时间",
            "Total_finish_time": "总完成时间",
            "R_tail_latency": "读尾延迟",
            "W_tail_latency": "写尾延迟",
            # CMD延迟
            "cmd_read_avg_latency": "命令延迟_读_平均",
            "cmd_read_max_latency": "命令延迟_读_最大",
            "cmd_write_avg_latency": "命令延迟_写_平均",
            "cmd_write_max_latency": "命令延迟_写_最大",
            "cmd_mixed_avg_latency": "命令延迟_平均",
            "cmd_mixed_max_latency": "命令延迟_最大",
            # Data延迟
            "data_read_avg_latency": "数据延迟_读_平均",
            "data_read_max_latency": "数据延迟_读_最大",
            "data_write_avg_latency": "数据延迟_写_平均",
            "data_write_max_latency": "数据延迟_写_最大",
            "data_mixed_avg_latency": "数据延迟_平均",
            "data_mixed_max_latency": "数据延迟_最大",
            # Transaction延迟
            "trans_read_avg_latency": "事务延迟_读_平均",
            "trans_read_max_latency": "事务延迟_读_最大",
            "trans_write_avg_latency": "事务延迟_写_平均",
            "trans_write_max_latency": "事务延迟_写_最大",
            "trans_mixed_avg_latency": "事务延迟_平均",
            "trans_mixed_max_latency": "事务延迟_最大",
            # Circling统计
            "req_cir_h_num": "请求_横向_环次数",
            "req_cir_v_num": "请求_纵向_环次数",
            "rsp_cir_h_num": "响应_横向_环次数",
            "rsp_cir_v_num": "响应_纵向_环次数",
            "data_cir_h_num": "数据_横向_环次数",
            "data_cir_v_num": "数据_纵向_环次数",
            # Wait Cycle统计
            "req_wait_cycle_h_num": "请求_横向_等待周期",
            "req_wait_cycle_v_num": "请求_纵向_等待周期",
            "rsp_wait_cycle_h_num": "响应_横向_等待周期",
            "rsp_wait_cycle_v_num": "响应_纵向_等待周期",
            "data_wait_cycle_h_num": "数据_横向_等待周期",
            "data_wait_cycle_v_num": "数据_纵向_等待周期",
            # Retry统计
            "read_retry_num": "读重试数",
            "write_retry_num": "写重试数",
            # ETag统计
            "EQ_ETag_T1_num": "EQ_ETag_T1",
            "EQ_ETag_T0_num": "EQ_ETag_T0",
            "RB_ETag_T1_num": "RB_ETag_T1",
            "RB_ETag_T0_num": "RB_ETag_T0",
            # ITag统计
            "ITag_h_num": "ITag_横向",
            "ITag_v_num": "ITag_纵向",
            # 带宽统计
            "Total_sum_BW": "总带宽",
            "read_unweighted_bw": "带宽_读_非加权",
            "read_weighted_bw": "带宽_读_加权",
            "write_unweighted_bw": "带宽_写_非加权",
            "write_weighted_bw": "带宽_写_加权",
            "mixed_unweighted_bw": "带宽_非加权",
            "mixed_weighted_bw": "带宽_加权",
            "mixed_avg_unweighted_bw": "带宽_平均非加权",
            "mixed_avg_weighted_bw": "带宽_平均加权",
            "total_unweighted_bw": "带宽_总_非加权",
            "total_weighted_bw": "带宽_总_加权",
        }

        results = {}
        for key, value in sim_vars.items():
            if key.endswith("_stat"):
                base_key = key.rsplit("_stat", 1)[0]
                chinese_key = stat_name_map.get(base_key, base_key)
                results[chinese_key] = value

        # Define config whitelist (only YAML defined parameters)
        config_whitelist = [
            # Basic parameters
            "TOPO_TYPE",
            "FLIT_SIZE",
            "SLICE_PER_LINK_HORIZONTAL",
            "SLICE_PER_LINK_VERTICAL",
            "BURST",
            "NETWORK_FREQUENCY",
            # Resource configuration
            "RN_RDB_SIZE",
            "RN_WDB_SIZE",
            "SN_DDR_RDB_SIZE",
            "SN_DDR_WDB_SIZE",
            "SN_L2M_RDB_SIZE",
            "SN_L2M_WDB_SIZE",
            "UNIFIED_RW_TRACKER",
            # Latency configuration (using original values)
            "DDR_R_LATENCY_original",
            "DDR_R_LATENCY_VAR_original",
            "DDR_W_LATENCY_original",
            "L2M_R_LATENCY_original",
            "L2M_W_LATENCY_original",
            "SN_TRACKER_RELEASE_LATENCY_original",
            "SN_PROCESSING_LATENCY_original",
            "RN_PROCESSING_LATENCY_original",
            # RingStation 配置
            "RS_IN_CH_BUFFER",
            "RS_IN_FIFO_DEPTH",
            "RS_OUT_CH_BUFFER",
            "RS_OUT_FIFO_DEPTH",
            # ETag configuration
            "TL_Etag_T1_UE_MAX",
            "TL_Etag_T2_UE_MAX",
            "TR_Etag_T2_UE_MAX",
            "TU_Etag_T1_UE_MAX",
            "TU_Etag_T2_UE_MAX",
            "TD_Etag_T2_UE_MAX",
            "ETAG_BOTHSIDE_UPGRADE",
            # ITag configuration
            "ITag_TRIGGER_Th_H",
            "ITag_TRIGGER_Th_V",
            "ITag_MAX_NUM_H",
            "ITag_MAX_NUM_V",
            # Feature switches
            "ENABLE_CROSSPOINT_CONFLICT_CHECK",
            "ETAG_T1_ENABLED",
            "ORDERING_PRESERVATION_MODE",
            "ORDERING_ETAG_UPGRADE_MODE",
            "ORDERING_GRANULARITY",
            "REVERSE_DIRECTION_ENABLED",
            "REVERSE_DIRECTION_THRESHOLD",
            # Allowed source nodes (dual-side ejection)
            "TL_ALLOWED_SOURCE_NODES",
            "TR_ALLOWED_SOURCE_NODES",
            "TU_ALLOWED_SOURCE_NODES",
            "TD_ALLOWED_SOURCE_NODES",
            # Bandwidth limits
            "GDMA_BW_LIMIT",
            "SDMA_BW_LIMIT",
            "CDMA_BW_LIMIT",
            "DDR_BW_LIMIT",
            "L2M_BW_LIMIT",
            # Other configurations
            "IN_ORDER_EJECTION_PAIRS",
            "IN_ORDER_PACKET_CATEGORIES",
            # IP frequency transformation FIFO depths
            "IP_L2H_FIFO_DEPTH",
            "IP_H2L_H_FIFO_DEPTH",
            "IP_H2L_L_FIFO_DEPTH",
        ]

        # Add selected configuration variables
        for key in config_whitelist:
            if hasattr(self.config, key):
                results[key] = getattr(self.config, key)

        # Clear flit and packet IDs (assuming these are class methods)
        Flit.clear_flit_id()
        BaseModel.reset_packet_id()

        # Add result processor analysis for port bandwidth data
        try:
            if hasattr(self, "result_processor") and self.result_processor:
                # Collect request data and analyze bandwidth
                self.result_processor.collect_requests_data(self, self.cycle)
                bandwidth_analysis = self.result_processor.analyze_all_bandwidth()

                # Include port averages in results (both original dict and expanded fields)
                if "port_averages" in bandwidth_analysis:
                    port_avg = bandwidth_analysis["port_averages"]
                    results["port_averages"] = port_avg  # Keep original dict for compatibility

                    # Expand port_averages dictionary into individual fields with Chinese names
                    # Pattern: avg_{port}_{op}_bw -> {端口}_{操作}_带宽
                    port_name_map = {
                        "avg_gdma_read_bw": "GDMA_读_带宽",
                        "avg_gdma_write_bw": "GDMA_写_带宽",
                        "avg_gdma_bw": "GDMA_带宽",
                        "avg_sdma_read_bw": "SDMA_读_带宽",
                        "avg_sdma_write_bw": "SDMA_写_带宽",
                        "avg_sdma_bw": "SDMA_带宽",
                        "avg_cdma_read_bw": "CDMA_读_带宽",
                        "avg_cdma_write_bw": "CDMA_写_带宽",
                        "avg_cdma_bw": "CDMA_带宽",
                        "avg_ddr_read_bw": "DDR_读_带宽",
                        "avg_ddr_write_bw": "DDR_写_带宽",
                        "avg_ddr_bw": "DDR_带宽",
                        "avg_l2m_read_bw": "L2M_读_带宽",
                        "avg_l2m_write_bw": "L2M_写_带宽",
                        "avg_l2m_bw": "L2M_带宽",
                    }
                    for key, value in port_avg.items():
                        chinese_key = port_name_map.get(key, key)
                        results[chinese_key] = value

                # Include other useful bandwidth metrics
                if "Total_sum_BW" in bandwidth_analysis:
                    results["总和带宽"] = bandwidth_analysis["Total_sum_BW"]

                # Include circling eject stats (both original dict and expanded fields)
                if "circling_eject_stats" in bandwidth_analysis:
                    circling_stats = bandwidth_analysis["circling_eject_stats"]
                    results["circling_eject_stats"] = circling_stats  # Keep original dict for compatibility

                    # Expand circling_eject_stats dictionary into individual fields with Chinese names
                    if "horizontal" in circling_stats:
                        results["绕环_横向_总flit数"] = circling_stats["horizontal"]["total_data_flits"]
                        results["绕环_横向_绕环flit数"] = circling_stats["horizontal"]["circling_flits"]
                        results["绕环_横向_比例"] = circling_stats["horizontal"]["circling_ratio"]

                    if "vertical" in circling_stats:
                        results["绕环_纵向_总flit数"] = circling_stats["vertical"]["total_data_flits"]
                        results["绕环_纵向_绕环flit数"] = circling_stats["vertical"]["circling_flits"]
                        results["绕环_纵向_比例"] = circling_stats["vertical"]["circling_ratio"]

                    if "overall" in circling_stats:
                        results["绕环_总flit数"] = circling_stats["overall"]["total_data_flits"]
                        results["绕环_绕环flit数"] = circling_stats["overall"]["circling_flits"]
                        results["绕环_比例"] = circling_stats["overall"]["circling_ratio"]

                # Include ordering blocked stats (both original dict and expanded fields)
                if "ordering_blocked_stats" in bandwidth_analysis:
                    ordering_stats = bandwidth_analysis["ordering_blocked_stats"]
                    results["ordering_blocked_stats"] = ordering_stats  # Keep original dict for compatibility

                    # Expand ordering_blocked_stats dictionary into individual fields
                    if "horizontal" in ordering_stats:
                        results["保序阻止_横向_总flit数"] = ordering_stats["horizontal"]["total_data_flits"]
                        results["保序阻止_横向_被阻止flit数"] = ordering_stats["horizontal"]["ordering_blocked_flits"]
                        results["保序阻止_横向_比例"] = ordering_stats["horizontal"]["ordering_blocked_ratio"]

                    if "vertical" in ordering_stats:
                        results["保序阻止_纵向_总flit数"] = ordering_stats["vertical"]["total_data_flits"]
                        results["保序阻止_纵向_被阻止flit数"] = ordering_stats["vertical"]["ordering_blocked_flits"]
                        results["保序阻止_纵向_比例"] = ordering_stats["vertical"]["ordering_blocked_ratio"]

                    if "overall" in ordering_stats:
                        results["保序阻止_总flit数"] = ordering_stats["overall"]["total_data_flits"]
                        results["保序阻止_被阻止flit数"] = ordering_stats["overall"]["ordering_blocked_flits"]
                        results["保序阻止_比例"] = ordering_stats["overall"]["ordering_blocked_ratio"]

                # 反方向上环统计
                if "reverse_inject_stats" in bandwidth_analysis:
                    reverse_stats = bandwidth_analysis["reverse_inject_stats"]
                    results["reverse_inject_stats"] = reverse_stats

                    if "horizontal" in reverse_stats:
                        results["反方向_横向_总flit数"] = reverse_stats["horizontal"]["total_data_flits"]
                        results["反方向_横向_反方向flit数"] = reverse_stats["horizontal"]["reverse_inject_flits"]
                        results["反方向_横向_比例"] = reverse_stats["horizontal"]["reverse_inject_ratio"]

                    if "vertical" in reverse_stats:
                        results["反方向_纵向_总flit数"] = reverse_stats["vertical"]["total_data_flits"]
                        results["反方向_纵向_反方向flit数"] = reverse_stats["vertical"]["reverse_inject_flits"]
                        results["反方向_纵向_比例"] = reverse_stats["vertical"]["reverse_inject_ratio"]

                    if "overall" in reverse_stats:
                        results["反方向_总flit数"] = reverse_stats["overall"]["total_data_flits"]
                        results["反方向_反方向flit数"] = reverse_stats["overall"]["reverse_inject_flits"]
                        results["反方向_比例"] = reverse_stats["overall"]["reverse_inject_ratio"]

        except Exception as e:
            if hasattr(self, "verbose") and self.verbose:
                print(f"Warning: Could not get port bandwidth analysis: {e}")
            # Set empty port_averages to avoid errors in downstream code
            results["port_averages"] = {}

        return results

    def get_performance_stats(self):
        """Get performance optimization statistics"""
        stats = {
            "simulation_cycles": self.cycle,
            "total_flits_processed": self.trans_flits_num,
            "flit_pool_stats": Flit.get_pool_stats(),
        }

        # Add I/O performance stats if available
        if hasattr(self.traffic_scheduler, "get_io_stats"):
            stats["io_performance"] = self.traffic_scheduler.get_io_stats()

        return stats

    def save_to_database(self, experiment_name=None, experiment_type="kcin", description=None):
        """保存仿真结果到数据库（所有文件内容存入数据库，不生成本地文件）

        Args:
            experiment_name: 实验名称，默认为"日常仿真_YYYY-MM-DD"
            experiment_type: 实验类型，默认为"kcin"
            description: 实验描述，默认为"日常仿真结果汇总"

        Returns:
            experiment_id: 实验ID
        """
        from datetime import datetime
        from src.database import ResultManager

        db = ResultManager()

        # 获取 traffic 文件列表
        traffic_files = []
        if hasattr(self, "traffic_scheduler") and hasattr(self.traffic_scheduler, "traffic_config"):
            for chain in self.traffic_scheduler.traffic_config:
                traffic_files.extend(chain)

        # 获取配置路径
        config_path = getattr(self.config, "config_path", None)

        # 实验名称处理：
        # - 用户指定了名称：查找已有实验并追加，或创建新实验
        # - 用户未指定名称：创建带时间戳的新实验

        if experiment_name is None:
            # 未指定名称，创建带时间戳的新实验
            now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            experiment_name = f"日常仿真_{now}"
            experiment_id = db.create_experiment(
                name=experiment_name,
                experiment_type=experiment_type,
                topo_type=self.topo_type_stat,
                config_path=config_path,
                traffic_files=traffic_files,
                description=description or "日常仿真结果汇总",
            )
        else:
            # 用户指定了名称，查找已有实验
            exp = db.get_experiment_by_name(experiment_name)
            if exp:
                experiment_id = exp["id"]
                # 如果传入了新描述，更新实验描述
                if description:
                    db.update_experiment(experiment_id, description=description)
            else:
                experiment_id = db.create_experiment(
                    name=experiment_name,
                    experiment_type=experiment_type,
                    topo_type=self.topo_type_stat,
                    config_path=config_path,
                    traffic_files=traffic_files,
                    description=description or "日常仿真结果汇总",
                )
        db.update_experiment_status(experiment_id, "completed")

        # 处理综合结果
        # 如果仿真运行时已经处理过（_result_file_contents已存在），跳过重复处理
        already_processed = hasattr(self, '_result_file_contents') and self._result_file_contents
        if not already_processed and hasattr(self, 'result_processor') and self.result_processor is not None:
            self.process_comprehensive_results()

        # 获取仿真结果
        results = self.get_results()

        # 提取性能指标（带宽）
        performance = results.get("DDR_带宽", 0)

        # 获取HTML报告内容
        result_html = getattr(self, "_result_html_content", None)

        # 获取收集到的文件内容（所有结果文件都在这里，包括Parquet）
        result_file_contents = getattr(self, "_result_file_contents", {})

        # 保存结果
        result_id = db.add_result(
            experiment_id=experiment_id,
            config_params=results,
            performance=performance,
            result_details=results,
            result_html=result_html,
            result_file_contents=result_file_contents if result_file_contents else None,
        )

        if self.verbose:
            print(f"\n结果已保存到数据库，实验: {experiment_name}, ID: {experiment_id}")

        return experiment_id
