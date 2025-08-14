"""
D2D_Model class for Die-to-Die simulation.
Manages multiple die instances and coordinates cross-die communication.
"""

import copy
import time
import logging
from typing import Dict, List, Optional
from .base_model import BaseModel
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
        self.num_dies = getattr(config, 'NUM_DIES', 2)
        
        # 存储各Die实例
        self.dies: Dict[int, BaseModel] = {}
        
        # 仿真参数
        self.end_time = getattr(config, 'END_TIME', 10000)
        self.print_interval = getattr(config, 'PRINT_INTERVAL', 1000)
        
        # 统计信息
        self.total_cross_die_requests = 0
        self.total_cross_die_responses = 0
        
        logging.info(f"D2D_Model initialized with {self.num_dies} dies")
        
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
                model_type=self.kwargs.get('model_type', 'REQ_RSP'),
                config=die_config,
                topo_type=self.kwargs.get('topo_type', '8x9'),
                traffic_file_path=self.kwargs.get('traffic_file_path', '../traffic/'),
                traffic_config=self.traffic_config,
                result_save_path=self.kwargs.get('result_save_path', '../Result/'),
                results_fig_save_path=self.kwargs.get('results_fig_save_path', ''),
                plot_flow_fig=self.kwargs.get('plot_flow_fig', 0),
                plot_RN_BW_fig=self.kwargs.get('plot_RN_BW_fig', 0),
                plot_link_state=self.kwargs.get('plot_link_state', 0),
                plot_start_cycle=self.kwargs.get('plot_start_cycle', 0),
                print_trace=self.kwargs.get('print_trace', 0),
                show_trace_id=self.kwargs.get('show_trace_id', 0),
                verbose=self.kwargs.get('verbose', 1),
            )
            
            # 设置Die ID
            die_model.die_id = die_id
            
            # 初始化Die
            die_model.initial()
            
            # 替换或添加D2D节点
            self._add_d2d_nodes_to_die(die_model, die_id)
            
            self.dies[die_id] = die_model
            
            logging.info(f"Created Die {die_id} instance")
    
    def _create_die_config(self, die_id: int) -> CrossRingConfig:
        """为每个Die创建独立的配置"""
        die_config = copy.deepcopy(self.config)
        
        # 设置Die ID
        die_config.DIE_ID = die_id
        
        # 获取D2D节点位置配置
        d2d_rn_position = getattr(self.config, 'D2D_RN_POSITION', 35)
        d2d_sn_position = getattr(self.config, 'D2D_SN_POSITION', 36)
        
        # 设置D2D节点位置
        die_config.D2D_RN_POSITION = d2d_rn_position
        die_config.D2D_SN_POSITION = d2d_sn_position
        
        # 添加D2D节点到IP列表
        if hasattr(die_config, 'CH_NAME_LIST'):
            if 'd2d_rn' not in die_config.CH_NAME_LIST:
                die_config.CH_NAME_LIST.append('d2d_rn')
            if 'd2d_sn' not in die_config.CH_NAME_LIST:
                die_config.CH_NAME_LIST.append('d2d_sn')
        
        # 设置D2D节点的发送位置列表
        if hasattr(die_config, 'D2D_RN_SEND_POSITION_LIST'):
            die_config.D2D_RN_SEND_POSITION_LIST = [d2d_rn_position]
        else:
            die_config.D2D_RN_SEND_POSITION_LIST = [d2d_rn_position]
            
        if hasattr(die_config, 'D2D_SN_SEND_POSITION_LIST'):
            die_config.D2D_SN_SEND_POSITION_LIST = [d2d_sn_position]
        else:
            die_config.D2D_SN_SEND_POSITION_LIST = [d2d_sn_position]
        
        return die_config
    
    def _add_d2d_nodes_to_die(self, die_model: BaseModel, die_id: int):
        """向Die添加D2D节点"""
        config = die_model.config
        
        d2d_rn_position = config.D2D_RN_POSITION
        d2d_sn_position = config.D2D_SN_POSITION
        
        # 创建D2D_RN节点
        if d2d_rn_position in die_model.flit_positions:
            die_model.ip_modules[('d2d_rn', d2d_rn_position)] = D2D_RN_Interface(
                ip_type='d2d_rn',
                ip_pos=d2d_rn_position,
                config=config,
                req_network=die_model.req_network,
                rsp_network=die_model.rsp_network,
                data_network=die_model.data_network,
                node=die_model.node,
                routes=die_model.routes,
                ip_id=0
            )
            logging.info(f"Added D2D_RN at position {d2d_rn_position} in Die {die_id}")
        
        # 创建D2D_SN节点
        if d2d_sn_position in die_model.flit_positions:
            die_model.ip_modules[('d2d_sn', d2d_sn_position)] = D2D_SN_Interface(
                ip_type='d2d_sn',
                ip_pos=d2d_sn_position,
                config=config,
                req_network=die_model.req_network,
                rsp_network=die_model.rsp_network,
                data_network=die_model.data_network,
                node=die_model.node,
                routes=die_model.routes,
                ip_id=0
            )
            logging.info(f"Added D2D_SN at position {d2d_sn_position} in Die {die_id}")
    
    def _setup_cross_die_connections(self):
        """建立Die间的连接关系"""
        for die_id, die_model in self.dies.items():
            # 获取当前Die的D2D_RN接口
            d2d_rn_key = ('d2d_rn', die_model.config.D2D_RN_POSITION)
            if d2d_rn_key in die_model.ip_modules:
                d2d_rn = die_model.ip_modules[d2d_rn_key]
                
                # 连接到其他Die的D2D_SN
                for other_die_id, other_die_model in self.dies.items():
                    if other_die_id != die_id:
                        d2d_sn_key = ('d2d_sn', other_die_model.config.D2D_SN_POSITION)
                        if d2d_sn_key in other_die_model.ip_modules:
                            d2d_sn = other_die_model.ip_modules[d2d_sn_key]
                            d2d_rn.target_die_interfaces[other_die_id] = d2d_sn
                            logging.info(f"Connected Die {die_id} D2D_RN to Die {other_die_id} D2D_SN")
    
    def initial(self):
        """初始化D2D仿真"""
        logging.info("D2D simulation initialized")
        # Dies已在构造函数中初始化
    
    def run(self):
        """运行D2D仿真主循环"""
        simulation_start = time.perf_counter()
        
        logging.info(f"Starting D2D simulation with {self.num_dies} dies")
        logging.info(f"Simulation will run for {self.end_time} cycles")
        
        # 主仿真循环
        while self.current_cycle < self.end_time:
            # 更新所有Die的当前周期
            for die_id, die_model in self.dies.items():
                die_model.current_cycle = self.current_cycle
                
                # 更新所有IP模块的当前周期
                for ip_module in die_model.ip_modules.values():
                    ip_module.current_cycle = self.current_cycle
            
            # 执行各Die的单周期更新
            for die_id, die_model in self.dies.items():
                # 调用Die的update方法（需要在BaseModel中实现）
                self._update_die(die_model)
            
            # 打印进度
            if self.current_cycle % self.print_interval == 0 and self.current_cycle > 0:
                self._print_progress()
            
            self.current_cycle += 1
        
        # 仿真结束
        simulation_end = time.perf_counter()
        simulation_time = simulation_end - simulation_start
        
        logging.info(f"D2D simulation completed in {simulation_time:.2f} seconds")
        self._print_final_statistics()
    
    def _update_die(self, die_model: BaseModel):
        """更新单个Die的状态"""
        # 更新IP模块
        for ip_module in die_model.ip_modules.values():
            ip_module.update()
        
        # 更新网络
        die_model.req_network.update()
        die_model.rsp_network.update()
        die_model.data_network.update()
        
        # 处理traffic（如果有）
        if hasattr(die_model, 'update_traffic'):
            die_model.update_traffic()
    
    def _print_progress(self):
        """打印仿真进度"""
        progress = (self.current_cycle / self.end_time) * 100
        print(f"Cycle {self.current_cycle}/{self.end_time} ({progress:.1f}%)")
        
        # 打印各Die的统计信息
        for die_id, die_model in self.dies.items():
            d2d_rn_key = ('d2d_rn', die_model.config.D2D_RN_POSITION)
            d2d_sn_key = ('d2d_sn', die_model.config.D2D_SN_POSITION)
            
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
            d2d_rn_key = ('d2d_rn', die_model.config.D2D_RN_POSITION)
            if d2d_rn_key in die_model.ip_modules:
                rn_stats = die_model.ip_modules[d2d_rn_key].get_statistics()
                print(f"  D2D_RN:")
                print(f"    Cross-die requests sent: {rn_stats['cross_die_requests_sent']}")
                print(f"    Cross-die responses received: {rn_stats['cross_die_responses_received']}")
                total_cross_die_requests += rn_stats['cross_die_requests_sent']
            
            # D2D_SN统计
            d2d_sn_key = ('d2d_sn', die_model.config.D2D_SN_POSITION)
            if d2d_sn_key in die_model.ip_modules:
                sn_stats = die_model.ip_modules[d2d_sn_key].get_statistics()
                print(f"  D2D_SN:")
                print(f"    Cross-die requests received: {sn_stats['cross_die_requests_received']}")
                print(f"    Cross-die requests forwarded: {sn_stats['cross_die_requests_forwarded']}")
                print(f"    Cross-die responses sent: {sn_stats['cross_die_responses_sent']}")
                total_cross_die_responses += sn_stats['cross_die_responses_sent']
        
        print(f"\nTotal cross-die transactions:")
        print(f"  Requests: {total_cross_die_requests}")
        print(f"  Responses: {total_cross_die_responses}")