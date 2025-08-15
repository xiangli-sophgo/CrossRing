"""
D2D_Model class for Die-to-Die simulation.
Manages multiple die instances and coordinates cross-die communication.
"""

import copy
import time
import logging
import os
from typing import Dict, List, Optional
from .base_model import BaseModel
from .d2d_traffic_scheduler import D2DTrafficScheduler
from src.utils.components.d2d_rn_interface import D2D_RN_Interface
from src.utils.components.d2d_sn_interface import D2D_SN_Interface
from config.config import CrossRingConfig


class D2D_Model:
    """
    D2D仿真主类 - 管理多Die协调
    每个Die是独立的BaseModel实例，D2D_Model负责：
    1. 创建和管理多个Die实例
    2. 设置Die间的连接关系
    3. 协调多Die的时钟同步
    """

    def __init__(self, config: CrossRingConfig, traffic_config: list, **kwargs):
        self.config = config
        self.traffic_config = traffic_config
        self.kwargs = kwargs
        self.current_cycle = 0

        # 获取Die数量，默认为2
        self.num_dies = getattr(config, "NUM_DIES", 2)

        # 存储各Die实例
        self.dies: Dict[int, BaseModel] = {}

        # 仿真参数
        self.end_time = getattr(config, "END_TIME", 10000)
        self.print_interval = getattr(config, "PRINT_INTERVAL", 1000)

        # 统计信息
        self.total_cross_die_requests = 0
        self.total_cross_die_responses = 0

        # 创建D2D专用的traffic调度器
        self.d2d_traffic_scheduler = D2DTrafficScheduler(traffic_config, self.kwargs.get("traffic_file_path", "../traffic/"), config)

        # 创建Die实例
        self._create_die_instances()

        # 设置跨Die连接
        self._setup_cross_die_connections()

    def _create_die_instances(self):
        """为每个Die创建独立的BaseModel实例"""
        for die_id in range(self.num_dies):
            die_config = self._create_die_config(die_id)

            # 创建BaseModel实例
            die_model = BaseModel(
                model_type=self.kwargs.get("model_type", "REQ_RSP"),
                config=die_config,
                topo_type=self.kwargs.get("topo_type", "8x9"),
                traffic_file_path=self.kwargs.get("traffic_file_path", "../traffic/"),
                traffic_config=self.traffic_config,
                result_save_path=self.kwargs.get("result_save_path", "../Result/"),
                results_fig_save_path=self.kwargs.get("results_fig_save_path", ""),
                plot_flow_fig=self.kwargs.get("plot_flow_fig", 0),
                plot_RN_BW_fig=self.kwargs.get("plot_RN_BW_fig", 0),
                plot_link_state=self.kwargs.get("plot_link_state", 0),
                plot_start_cycle=self.kwargs.get("plot_start_cycle", 0),
                print_trace=self.kwargs.get("print_trace", 0),
                show_trace_id=self.kwargs.get("show_trace_id", 0),
                verbose=self.kwargs.get("verbose", 1),
            )

            # 设置Die ID
            die_model.die_id = die_id

            # 初始化Die
            die_model.initial()

            # 替换或添加D2D节点
            self._add_d2d_nodes_to_die(die_model, die_id)

            self.dies[die_id] = die_model

    def _create_die_config(self, die_id: int) -> CrossRingConfig:
        """为每个Die创建独立的配置"""
        die_config = copy.deepcopy(self.config)

        # 设置Die ID
        die_config.DIE_ID = die_id

        # 获取D2D节点位置配置
        d2d_rn_position = getattr(self.config, "D2D_RN_POSITION", 35)
        d2d_sn_position = getattr(self.config, "D2D_SN_POSITION", 36)

        # 设置D2D节点位置
        die_config.D2D_RN_POSITION = d2d_rn_position
        die_config.D2D_SN_POSITION = d2d_sn_position

        # 添加D2D节点到IP列表
        if hasattr(die_config, "CH_NAME_LIST"):
            if "d2d_rn" not in die_config.CH_NAME_LIST:
                die_config.CH_NAME_LIST.append("d2d_rn")
            if "d2d_sn" not in die_config.CH_NAME_LIST:
                die_config.CH_NAME_LIST.append("d2d_sn")

        # 设置D2D节点的发送位置列表
        if hasattr(die_config, "D2D_RN_SEND_POSITION_LIST"):
            die_config.D2D_RN_SEND_POSITION_LIST = [d2d_rn_position]
        else:
            die_config.D2D_RN_SEND_POSITION_LIST = [d2d_rn_position]

        if hasattr(die_config, "D2D_SN_SEND_POSITION_LIST"):
            die_config.D2D_SN_SEND_POSITION_LIST = [d2d_sn_position]
        else:
            die_config.D2D_SN_SEND_POSITION_LIST = [d2d_sn_position]

        return die_config

    def _add_d2d_nodes_to_die(self, die_model: BaseModel, die_id: int):
        """向Die添加D2D节点"""
        config = die_model.config

        # 根据die_id确定D2D节点位置
        if die_id == 0:
            # Die 0：水平布局右边界位置（考虑node_map映射）
            # 物理位置：7, 15, 23, 31, 39（右边界）
            d2d_rn_position = 7  # 第0行右边界
            d2d_sn_position = 15  # 第1行右边界
        else:
            # Die 1：水平布局左边界位置
            # 物理位置：4, 12, 20, 28, 36（左边界源节点）
            d2d_rn_position = 4  # 第0行左边界源节点
            d2d_sn_position = 12  # 第1行左边界源节点

        # 强制添加D2D节点位置到flit_positions（绕过原有限制）
        die_model.flit_positions.add(d2d_rn_position)
        die_model.flit_positions.add(d2d_sn_position)
        die_model.flit_positions_list = list(die_model.flit_positions)

        # 创建D2D_RN节点
        die_model.ip_modules[("d2d_rn", d2d_rn_position)] = D2D_RN_Interface(
            ip_type="d2d_rn",
            ip_pos=d2d_rn_position,
            config=config,
            req_network=die_model.req_network,
            rsp_network=die_model.rsp_network,
            data_network=die_model.data_network,
            node=die_model.node,
            routes=die_model.routes,
            ip_id=0,
        )

        # 创建D2D_SN节点
        die_model.ip_modules[("d2d_sn", d2d_sn_position)] = D2D_SN_Interface(
            ip_type="d2d_sn",
            ip_pos=d2d_sn_position,
            config=config,
            req_network=die_model.req_network,
            rsp_network=die_model.rsp_network,
            data_network=die_model.data_network,
            node=die_model.node,
            routes=die_model.routes,
            ip_id=0,
        )

        # 更新配置以记录实际使用的D2D位置
        config.D2D_RN_POSITION = d2d_rn_position
        config.D2D_SN_POSITION = d2d_sn_position

    def _setup_cross_die_connections(self):
        """建立Die间的连接关系"""
        for die_id, die_model in self.dies.items():
            # 获取当前Die的D2D_RN接口
            d2d_rn_key = ("d2d_rn", die_model.config.D2D_RN_POSITION)
            if d2d_rn_key in die_model.ip_modules:
                d2d_rn = die_model.ip_modules[d2d_rn_key]

                # 连接到其他Die的D2D_SN
                for other_die_id, other_die_model in self.dies.items():
                    if other_die_id != die_id:
                        d2d_sn_key = ("d2d_sn", other_die_model.config.D2D_SN_POSITION)
                        if d2d_sn_key in other_die_model.ip_modules:
                            d2d_sn = other_die_model.ip_modules[d2d_sn_key]
                            d2d_rn.target_die_interfaces[other_die_id] = d2d_sn

    def initial(self):
        """初始化D2D仿真"""
        # Dies已在构造函数中初始化

    def run(self):
        """运行D2D仿真主循环"""
        simulation_start = time.perf_counter()

        # 主仿真循环
        while self.current_cycle < self.end_time:
            # 更新所有Die的当前周期
            for die_id, die_model in self.dies.items():
                die_model.current_cycle = self.current_cycle

                # 更新所有IP模块的当前周期
                for ip_module in die_model.ip_modules.values():
                    ip_module.current_cycle = self.current_cycle

            # 执行各Die的单周期步进
            for die_id, die_model in self.dies.items():
                # 调用Die的step方法
                self._step_die(die_model)

            # 打印进度
            if self.current_cycle % self.print_interval == 0 and self.current_cycle > 0:
                self._print_progress()

            self.current_cycle += 1

        # 仿真结束
        simulation_end = time.perf_counter()
        simulation_time = simulation_end - simulation_start

        self._print_final_statistics()

    def _step_die(self, die_model: BaseModel):
        """执行单个Die的单周期步进"""
        # 参考BaseModel的单周期更新逻辑

        # 0. 处理D2D Traffic注入
        self._process_d2d_traffic(die_model)

        # 1. IP inject 步骤
        for ip_pos in die_model.flit_positions_list:
            for ip_type in die_model.config.CH_NAME_LIST:
                ip_key = (ip_type, ip_pos)
                if ip_key in die_model.ip_modules:
                    ip_interface = die_model.ip_modules[ip_key]
                    ip_interface.inject_step(self.current_cycle)

        # 2. 网络更新步骤 - 需要实现具体的网络步进逻辑
        # 暂时跳过，因为Network类没有step方法
        # TODO: 需要根据BaseModel的实际实现来调用正确的网络更新方法
        pass

        # 3. IP eject 步骤
        for ip_pos in die_model.flit_positions_list:
            for ip_type in die_model.config.CH_NAME_LIST:
                ip_key = (ip_type, ip_pos)
                if ip_key in die_model.ip_modules:
                    ip_interface = die_model.ip_modules[ip_key]
                    ip_interface.eject_step(self.current_cycle)

        # 4. 移动pre到FIFO（参考BaseModel的move_pre_to_fifo）
        for ip_pos in die_model.flit_positions_list:
            for ip_type in die_model.config.CH_NAME_LIST:
                ip_key = (ip_type, ip_pos)
                if ip_key in die_model.ip_modules:
                    die_model.ip_modules[ip_key].move_pre_to_fifo()

        # 网络的pre到FIFO移动 - 只处理原始网络位置，不包括D2D节点
        original_positions = [pos for pos in die_model.flit_positions_list if (pos != die_model.config.D2D_RN_POSITION and pos != die_model.config.D2D_SN_POSITION)]
        for in_pos in original_positions:
            die_model._move_pre_to_queues(die_model.req_network, in_pos)
            die_model._move_pre_to_queues(die_model.rsp_network, in_pos)
            die_model._move_pre_to_queues(die_model.data_network, in_pos)

    def _process_d2d_traffic(self, die_model: BaseModel):
        """处理D2D traffic注入"""
        # 获取当前周期的D2D请求
        pending_requests = self.d2d_traffic_scheduler.get_pending_requests(self.current_cycle)

        for req_data in pending_requests:
            try:
                # 检查这个请求是否属于当前Die
                src_die = req_data[1]  # src_die字段
                if src_die != die_model.die_id:
                    continue  # 不是当前Die的请求，跳过

                # 从D2D请求创建flit（第一段路由：源节点→D2D_SN）
                flit = self.d2d_traffic_scheduler.create_d2d_flit_from_request(req_data, die_model, die_model.die_id)

                # 查找源IP接口
                source_ip_key = (flit.source_type, flit.source)
                if source_ip_key in die_model.ip_modules:
                    ip_interface = die_model.ip_modules[source_ip_key]
                    # 将flit注入到对应的IP接口
                    ip_interface.enqueue(flit, "req")

            except Exception as e:
                print(e)

    def _print_progress(self):
        """打印仿真进度"""
        progress = (self.current_cycle / self.end_time) * 100
        print(f"Cycle {self.current_cycle}/{self.end_time} ({progress:.1f}%)")

        # 打印各Die的统计信息
        for die_id, die_model in self.dies.items():
            d2d_rn_key = ("d2d_rn", die_model.config.D2D_RN_POSITION)
            d2d_sn_key = ("d2d_sn", die_model.config.D2D_SN_POSITION)

            if d2d_rn_key in die_model.ip_modules:
                rn_stats = die_model.ip_modules[d2d_rn_key].get_statistics()
                print(f"  Die {die_id} D2D_RN: sent={rn_stats['cross_die_requests_sent']}")

            if d2d_sn_key in die_model.ip_modules:
                sn_stats = die_model.ip_modules[d2d_sn_key].get_statistics()
                print(f"  Die {die_id} D2D_SN: received={sn_stats['cross_die_requests_received']}")

    def _print_final_statistics(self):
        """打印最终统计信息"""
        print("\n=== D2D Simulation Final Statistics ===")

        total_cross_die_requests = 0
        total_cross_die_responses = 0

        for die_id, die_model in self.dies.items():
            print(f"\nDie {die_id} Statistics:")

            # D2D_RN统计
            d2d_rn_key = ("d2d_rn", die_model.config.D2D_RN_POSITION)
            if d2d_rn_key in die_model.ip_modules:
                rn_stats = die_model.ip_modules[d2d_rn_key].get_statistics()
                print(f"  D2D_RN:")
                print(f"    Cross-die requests sent: {rn_stats['cross_die_requests_sent']}")
                print(f"    Cross-die responses received: {rn_stats['cross_die_responses_received']}")
                total_cross_die_requests += rn_stats["cross_die_requests_sent"]

            # D2D_SN统计
            d2d_sn_key = ("d2d_sn", die_model.config.D2D_SN_POSITION)
            if d2d_sn_key in die_model.ip_modules:
                sn_stats = die_model.ip_modules[d2d_sn_key].get_statistics()
                print(f"  D2D_SN:")
                print(f"    Cross-die requests received: {sn_stats['cross_die_requests_received']}")
                print(f"    Cross-die requests forwarded: {sn_stats['cross_die_requests_forwarded']}")
                print(f"    Cross-die responses sent: {sn_stats['cross_die_responses_sent']}")
                total_cross_die_responses += sn_stats["cross_die_responses_sent"]

        print(f"\nTotal cross-die transactions:")
        print(f"  Requests: {total_cross_die_requests}")
        print(f"  Responses: {total_cross_die_responses}")

        # 显示packet_id范围（用于验证全局唯一性）
        from src.utils.components.node import Node

        print(f"\nPacket ID Statistics:")
        print(f"  Last packet_id generated: {Node.global_packet_id}")

    def generate_combined_flow_graph(self, mode="total", save_path=None, show_cdma=True):
        """
        生成D2D双Die组合流量图

        Args:
            mode: 显示模式，支持 'utilization', 'total', 'ITag_ratio' 等
            save_path: 图片保存路径
            show_cdma: 是否显示CDMA
        """
        # 检查是否有BandwidthAnalyzer实例
        analyzers = {}
        die_networks = {}

        for die_id, die_model in self.dies.items():
            if hasattr(die_model, "result_processor") and die_model.result_processor:
                analyzers[die_id] = die_model.result_processor
                # 收集网络对象 - 根据mode选择合适的网络
                if mode == "total":
                    die_networks[die_id] = die_model.data_network  # 使用data_network显示总带宽
                else:
                    die_networks[die_id] = die_model.req_network  # 默认使用req_network

            else:
                return

        if len(analyzers) < 2:
            return

        # 使用第一个analyzer来绘制组合图
        primary_analyzer = list(analyzers.values())[0]

        # 设置保存路径
        if save_path is None:
            import time

            timestamp = int(time.time())
            save_path = f"../Result/d2d_combined_flow_{mode}_{timestamp}.png"

        # 确保primary_analyzer有必要的属性
        if not hasattr(primary_analyzer, "simulation_end_cycle"):
            primary_analyzer.simulation_end_cycle = self.current_cycle

        try:
            # 调用新的draw_d2d_flow_graph方法
            primary_analyzer.draw_d2d_flow_graph(die_networks=die_networks, config=self.config, mode=mode, save_path=save_path, show_cdma=show_cdma)

            print(f"D2D组合流量图已保存: {save_path}")

        except Exception as e:
            import traceback

            print(f"生成D2D组合流量图失败: {e}")

    def run_with_flow_visualization(self, enable_flow_graph=True, flow_mode="total"):
        """
        运行D2D仿真并生成流量可视化

        Args:
            enable_flow_graph: 是否启用流量图生成
            flow_mode: 流量图显示模式
        """
        # 先运行正常的仿真
        self.run()

        # 仿真完成后进行综合结果处理
        if enable_flow_graph:
            self.generate_combined_flow_graph(mode=flow_mode)

        # 处理D2D综合结果分析
        self.process_d2d_comprehensive_results()

    def process_d2d_comprehensive_results(self):
        """
        处理D2D综合结果分析，复用现有的结果处理方法
        """
        print("\n" + "=" * 60)
        print("D2D仿真综合结果分析")
        print("=" * 60)

        # 收集D2D专有统计信息
        d2d_stats = self._collect_d2d_statistics()

        # 1. 对每个Die调用现有的结果处理方法
        die_results = {}
        for die_id, die_model in self.dies.items():
            print(f"\n处理Die {die_id}的结果...")

            # 调用现有的综合结果处理方法
            if hasattr(die_model, "result_processor") and die_model.result_processor:
                try:
                    # 收集请求数据
                    die_model.result_processor.collect_requests_data(die_model, die_model.cycle)
                    # 分析带宽
                    die_results[die_id] = die_model.result_processor.analyze_all_bandwidth()

                    # 为每个Die生成独立的报告（保存到子目录）
                    die_result_path = os.path.join(self.kwargs.get("result_save_path", "../Result/"), f"Die_{die_id}")
                    os.makedirs(die_result_path, exist_ok=True)
                    die_model.result_processor.generate_unified_report(die_results[die_id], die_result_path)

                    # 生成FIFO使用率报告
                    die_model.result_processor.generate_fifo_usage_csv(die_model, die_result_path)

                    print(f"  Die {die_id}结果已保存到: {die_result_path}")

                except Exception as e:
                    print(f"  警告: Die {die_id}结果处理失败: {e}")
                    die_results[die_id] = None

        # 2. 输出D2D专有统计信息
        self._print_d2d_statistics(d2d_stats)

        # 3. 生成D2D组合报告（基于现有结果）
        self._generate_d2d_combined_report(die_results, d2d_stats)

        print("=" * 60)
        print("D2D综合结果分析完成")
        print("=" * 60)

    def _collect_d2d_statistics(self):
        """收集D2D专有统计信息"""
        d2d_stats = {"cross_die_requests": 0, "cross_die_responses": 0, "die_stats": {}}

        for die_id, die_model in self.dies.items():
            die_stat = {
                "read_req": die_model.read_req,
                "write_req": die_model.write_req,
                "read_flit": die_model.read_flit,
                "write_flit": die_model.write_flit,
                "total_cycles": die_model.cycle,
            }

            # D2D专有统计
            d2d_rn_key = ("d2d_rn", die_model.config.D2D_RN_POSITION)
            d2d_sn_key = ("d2d_sn", die_model.config.D2D_SN_POSITION)

            if d2d_rn_key in die_model.ip_modules:
                rn_stats = die_model.ip_modules[d2d_rn_key].get_statistics()
                die_stat["d2d_rn_sent"] = rn_stats["cross_die_requests_sent"]
                die_stat["d2d_rn_received"] = rn_stats["cross_die_responses_received"]
                d2d_stats["cross_die_requests"] += rn_stats["cross_die_requests_sent"]

            if d2d_sn_key in die_model.ip_modules:
                sn_stats = die_model.ip_modules[d2d_sn_key].get_statistics()
                die_stat["d2d_sn_received"] = sn_stats["cross_die_requests_received"]
                die_stat["d2d_sn_forwarded"] = sn_stats["cross_die_requests_forwarded"]
                die_stat["d2d_sn_responses"] = sn_stats["cross_die_responses_sent"]
                d2d_stats["cross_die_responses"] += sn_stats["cross_die_responses_sent"]

            d2d_stats["die_stats"][die_id] = die_stat

        return d2d_stats

    def _print_d2d_statistics(self, d2d_stats):
        """输出D2D专有统计信息"""
        print(f"\nD2D专有统计:")
        print(f"  跨Die请求总数: {d2d_stats['cross_die_requests']}")
        print(f"  跨Die响应总数: {d2d_stats['cross_die_responses']}")

        for die_id, stat in d2d_stats["die_stats"].items():
            print(f"\nDie {die_id} D2D统计:")
            if "d2d_rn_sent" in stat:
                print(f"  D2D_RN: 发送={stat['d2d_rn_sent']}, 接收={stat['d2d_rn_received']}")
            if "d2d_sn_received" in stat:
                print(f"  D2D_SN: 接收={stat['d2d_sn_received']}, 转发={stat['d2d_sn_forwarded']}, 响应={stat['d2d_sn_responses']}")

    def _generate_d2d_combined_report(self, die_results, d2d_stats):
        """生成D2D组合报告，基于现有的结果分析"""
        if not self.kwargs.get("result_save_path"):
            return

        result_path = self.kwargs.get("result_save_path", "../Result/")

        # 生成D2D组合统计报告
        combined_report_file = os.path.join(result_path, f"d2d_combined_report_{int(time.time())}.txt")

        with open(combined_report_file, "w", encoding="utf-8") as f:
            f.write("D2D双Die组合仿真报告\n")
            f.write("=" * 50 + "\n\n")

            # 配置信息
            f.write("仿真配置:\n")
            f.write(f"  Die数量: {self.num_dies}\n")
            f.write(f"  拓扑类型: 5x4 (每个Die)\n")
            f.write(f"  仿真周期: {self.current_cycle}\n")
            f.write(f"  网络频率: {self.config.NETWORK_FREQUENCY}\n\n")

            # D2D专有统计
            f.write("D2D通信统计:\n")
            f.write(f"  跨Die请求: {d2d_stats['cross_die_requests']}\n")
            f.write(f"  跨Die响应: {d2d_stats['cross_die_responses']}\n\n")

            # 各Die的带宽分析结果摘要
            f.write("各Die带宽分析摘要:\n")
            for die_id, results in die_results.items():
                if results:
                    f.write(f"  Die {die_id}:\n")
                    if "Total_sum_BW" in results:
                        f.write(f"    总带宽: {results['Total_sum_BW']:.2f} GB/s\n")
                    if "network_overall" in results:
                        read_bw = results["network_overall"]["read"].unweighted_bandwidth
                        write_bw = results["network_overall"]["write"].unweighted_bandwidth
                        f.write(f"    读带宽: {read_bw:.2f} GB/s, 写带宽: {write_bw:.2f} GB/s\n")
                    f.write("\n")
                else:
                    f.write(f"  Die {die_id}: 结果处理失败\n\n")

            # 分Die的D2D统计
            for die_id, stat in d2d_stats["die_stats"].items():
                f.write(f"Die {die_id} D2D详细统计:\n")
                f.write(f"  基本流量: 读请求={stat['read_req']}, 写请求={stat['write_req']}\n")
                f.write(f"  基本流量: 读flit={stat['read_flit']}, 写flit={stat['write_flit']}\n")
                if "d2d_rn_sent" in stat:
                    f.write(f"  D2D_RN: 发送={stat['d2d_rn_sent']}, 接收={stat['d2d_rn_received']}\n")
                if "d2d_sn_received" in stat:
                    f.write(f"  D2D_SN: 接收={stat['d2d_sn_received']}, 转发={stat['d2d_sn_forwarded']}, 响应={stat['d2d_sn_responses']}\n")
                f.write("\n")

        print(f"\nD2D组合报告已保存: {combined_report_file}")
