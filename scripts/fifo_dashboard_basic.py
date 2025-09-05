"""
FIFO参数优化 - 基础交互式可视化仪表板

基于全遍历结果数据，提供简洁直观的参数调整和性能查看功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
import ast
from typing import Dict, List, Optional
import dash
from dash import dcc, html, Input, Output, State, callback_context
from dash.dependencies import ALL
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

class FIFODashboard:
    """FIFO参数优化交互式仪表板"""
    
    def __init__(self, csv_path: str):
        """
        初始化仪表板
        
        Args:
            csv_path: parameter_performance.csv文件路径
        """
        self.csv_path = csv_path
        self.df = None
        self.param_names = []
        self.param_ranges = {}
        self.param_defaults = {}
        self.valid_combinations = set()  # 存储有效的参数组合
        self.param_config_index = {}  # 参数值到配置的索引 {(param_name, param_value): [configs]}
        
        # 初始化Dash应用
        self.app = dash.Dash(__name__)
        self.app.title = "FIFO参数优化仪表板"
        
        # 加载数据
        self.load_data()
        
        # 设置布局和回调
        self.setup_layout()
        self.setup_callbacks()
    
    def load_data(self):
        """加载和处理CSV数据"""
        print(f"正在加载数据: {self.csv_path}")
        
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"数据加载成功，共 {len(self.df)} 条记录")
            
            # 解析参数组合字符串
            self.df['combination_dict'] = self.df['full_combination'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            
            # 获取所有参数名和范围
            all_combinations = []
            for combo_dict in self.df['combination_dict']:
                all_combinations.append(combo_dict)
                # 构建有效组合索引（使用排序后的参数组合作为key）
                combo_key = tuple(sorted(combo_dict.items()))
                self.valid_combinations.add(combo_key)
            
            # 分析参数范围
            param_values = {}
            for combo in all_combinations:
                for param, value in combo.items():
                    if param not in param_values:
                        param_values[param] = []
                    param_values[param].append(value)
            
            # 设置参数信息
            self.param_names = sorted(param_values.keys())
            for param in self.param_names:
                values = sorted(set(param_values[param]))
                self.param_ranges[param] = {'min': min(values), 'max': max(values), 'values': values}
                self.param_defaults[param] = values[len(values)//2]  # 取中间值作为默认值
            
            print(f"检测到参数: {self.param_names}")
            
            # 建立参数配置索引
            self._build_param_config_index()
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            raise
    
    def _build_param_config_index(self):
        """建立参数值到配置的索引，优化查询性能"""
        print("建立参数配置索引...")
        
        for _, row in self.df.iterrows():
            combo_dict = row['combination_dict']
            performance = row['performance']
            
            # 为每个参数值建立索引
            for param_name, param_value in combo_dict.items():
                key = (param_name, param_value)
                if key not in self.param_config_index:
                    self.param_config_index[key] = []
                
                # 存储完整配置和性能
                self.param_config_index[key].append({
                    'config': combo_dict.copy(),
                    'performance': performance
                })
        
        print(f"索引建立完成，共 {len(self.param_config_index)} 个索引项")
    
    def get_performance(self, params: Dict) -> Optional[float]:
        """
        根据参数组合查询性能
        
        Args:
            params: 参数字典
            
        Returns:
            性能值，如果未找到返回None
        """
        # 查找匹配的记录
        matches = self.df[self.df['combination_dict'].apply(
            lambda x: all(x.get(k) == v for k, v in params.items())
        )]
        
        if not matches.empty:
            return matches['performance'].iloc[0]
        return None
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计信息"""
        return {
            'min': self.df['performance'].min(),
            'max': self.df['performance'].max(),
            'mean': self.df['performance'].mean(),
            'std': self.df['performance'].std(),
            'count': len(self.df)
        }
    
    def get_best_configuration(self) -> Dict:
        """
        获取真正的最优配置（性能最高的完整配置）
        
        Returns:
            最优配置的参数字典
        """
        # 找到性能最高的记录
        best_idx = self.df['performance'].idxmax()
        best_record = self.df.loc[best_idx]
        
        return best_record['combination_dict'].copy()
    
    def is_valid_combination(self, params: Dict) -> bool:
        """
        检查参数组合是否存在于数据集中
        
        Args:
            params: 参数字典
            
        Returns:
            True如果组合存在，False否则
        """
        combo_key = tuple(sorted(params.items()))
        return combo_key in self.valid_combinations
    
    def find_similar_configuration(self, params: Dict, max_suggestions: int = 3) -> List[Dict]:
        """
        找到与给定参数最相似的有效配置
        
        Args:
            params: 目标参数字典
            max_suggestions: 最大推荐数量
            
        Returns:
            相似配置列表，按相似度排序
        """
        if self.is_valid_combination(params):
            return [params]  # 如果本身就有效，直接返回
        
        # 计算与所有有效组合的距离
        suggestions = []
        for combo_key in self.valid_combinations:
            combo_dict = dict(combo_key)
            
            # 计算汉明距离（不同参数值的数量）
            distance = sum(1 for param in self.param_names 
                          if combo_dict.get(param) != params.get(param))
            
            # 获取该组合的性能
            performance = self.get_performance(combo_dict)
            if performance is not None:
                suggestions.append({
                    'params': combo_dict,
                    'distance': distance,
                    'performance': performance
                })
        
        # 按距离排序，距离相同时按性能排序
        suggestions.sort(key=lambda x: (x['distance'], -x['performance']))
        
        return [s['params'] for s in suggestions[:max_suggestions]]
    
    def check_constraints_detailed(self, params: Dict) -> Dict:
        """
        检查参数组合的约束并返回详细原因
        
        Args:
            params: 参数字典
            
        Returns:
            {'valid': bool, 'reasons': [reasons]}
        """
        reasons = []
        is_valid = True
        
        # 基于原始optimize_fifo_exhaustive.py的约束逻辑
        # ETag参数约束检查
        etag_constraints = {
            'TL_Etag_T2_UE_MAX': {'related_fifo': 'RB_IN_FIFO_DEPTH', 'constraint': 'less_than_fifo'},
            'TL_Etag_T1_UE_MAX': {'related_fifo': 'RB_IN_FIFO_DEPTH', 'constraint': 'less_than_fifo_and_greater_than_t2', 'corresponding_t2': 'TL_Etag_T2_UE_MAX'},
            'TR_Etag_T2_UE_MAX': {'related_fifo': 'RB_IN_FIFO_DEPTH', 'constraint': 'less_than_fifo'},
            'TU_Etag_T2_UE_MAX': {'related_fifo': 'EQ_IN_FIFO_DEPTH', 'constraint': 'less_than_fifo'},
            'TU_Etag_T1_UE_MAX': {'related_fifo': 'EQ_IN_FIFO_DEPTH', 'constraint': 'less_than_fifo_and_greater_than_t2', 'corresponding_t2': 'TU_Etag_T2_UE_MAX'},
            'TD_Etag_T2_UE_MAX': {'related_fifo': 'EQ_IN_FIFO_DEPTH', 'constraint': 'less_than_fifo'},
        }
        
        for param_name, param_value in params.items():
            if param_name in etag_constraints:
                constraint_info = etag_constraints[param_name]
                
                # 检查是否小于相关FIFO参数
                if 'less_than_fifo' in constraint_info['constraint']:
                    related_fifo = constraint_info['related_fifo']
                    if related_fifo in params:
                        fifo_value = params[related_fifo]
                        if param_value >= fifo_value:
                            reasons.append(f"{param_name}({param_value}) 必须小于 {related_fifo}({fifo_value})")
                            is_valid = False
                
                # 检查T1是否大于T2
                if 'greater_than_t2' in constraint_info['constraint']:
                    corresponding_t2 = constraint_info['corresponding_t2']
                    if corresponding_t2 in params:
                        t2_value = params[corresponding_t2]
                        if param_value <= t2_value:
                            reasons.append(f"{param_name}({param_value}) 必须大于 {corresponding_t2}({t2_value})")
                            is_valid = False
        
        return {'valid': is_valid, 'reasons': reasons}
    
    def get_configurations_for_param_value(self, param_name: str, param_value: int) -> List[Dict]:
        """
        获取特定参数值下的所有配置
        
        Args:
            param_name: 参数名
            param_value: 参数值
            
        Returns:
            配置列表，按性能排序
        """
        key = (param_name, param_value)
        if key not in self.param_config_index:
            return []
        
        # 按性能排序
        configs = self.param_config_index[key]
        sorted_configs = sorted(configs, key=lambda x: x['performance'], reverse=True)
        
        return sorted_configs
    
    def get_parameter_impact(self) -> Dict:
        """计算每个参数的详细影响力分析"""
        param_impact = {}
        
        for param in self.param_names:
            # 按参数值分组计算性能统计
            param_perf = self.df[self.df['param_name'] == param].groupby('param_value')['performance'].agg([
                'mean', 'min', 'max', 'std', 'count'
            ]).reset_index()
            
            if not param_perf.empty:
                # 计算总体影响力指标
                perf_range = param_perf['mean'].max() - param_perf['mean'].min()
                overall_mean = self.df['performance'].mean()
                impact_percentage = (perf_range / overall_mean) * 100 if overall_mean > 0 else 0
                
                # 整理详细数据用于绘图
                param_impact[param] = {
                    # 总体指标
                    'range': perf_range,
                    'impact_percentage': impact_percentage,
                    'min_perf': param_perf['mean'].min(),
                    'max_perf': param_perf['mean'].max(),
                    'best_value': param_perf.loc[param_perf['mean'].idxmax(), 'param_value'],
                    'worst_value': param_perf.loc[param_perf['mean'].idxmin(), 'param_value'],
                    
                    # 详细数据用于绘图
                    'values': param_perf['param_value'].tolist(),
                    'mean_performance': param_perf['mean'].tolist(),
                    'min_performance': param_perf['min'].tolist(),
                    'max_performance': param_perf['max'].tolist(),
                    'std_performance': param_perf['std'].tolist(),
                    'sample_counts': param_perf['count'].tolist()
                }
        
        return param_impact
    
    def setup_layout(self):
        """设置页面布局"""
        
        # 获取性能统计
        perf_stats = self.get_performance_stats()
        param_impact = self.get_parameter_impact()
        
        self.app.layout = html.Div([
            # 标题
            html.H1("FIFO参数优化交互式仪表板", 
                   style={'text-align': 'center', 'margin-bottom': '20px'}),
            
            # 主要内容区域
            html.Div([
                # 左侧：参数控制面板
                html.Div([
                    html.H3("参数控制", style={'margin-bottom': '20px'}),
                    
                    # 性能显示区
                    html.Div([
                        html.H4("当前性能"),
                        html.Div(id='current-performance', 
                                style={'font-size': '48px', 'font-weight': 'bold', 'color': '#2E8B57',
                                      'text-align': 'center', 'margin': '10px 0'}),
                        html.Div(id='performance-rank',
                                style={'text-align': 'center', 'color': '#666', 'margin-bottom': '20px'})
                    ], style={'background-color': '#f8f9fa', 'padding': '15px', 'border-radius': '5px',
                             'margin-bottom': '20px'}),
                    
                    # 参数滑块
                    html.Div(id='parameter-sliders'),
                    
                    # 快速操作按钮
                    html.Div([
                        html.Button("最优配置", id='btn-optimal', n_clicks=0,
                                  style={'margin': '5px', 'padding': '8px 16px'}),
                        html.Button("默认配置", id='btn-default', n_clicks=0,
                                  style={'margin': '5px', 'padding': '8px 16px'}),
                    ], style={'margin-top': '20px'})
                    
                ], style={'width': '30%', 'float': 'left', 'padding': '0 20px'}),
                
                # 右侧：可视化图表区域（主要展示区）
                html.Div([
                    # 参数影响力图表（小提琴图）
                    html.Div([
                        html.H4("参数性能分布分析", style={'color': '#2c3e50', 'margin-bottom': '15px'}),
                        dcc.Graph(id='parameter-impact-chart', style={'height': '800px'})
                    ], style={'margin-bottom': '20px'}),
                    
                    # 性能分布图（核密度图）
                    html.Div([
                        html.H4("整体性能分布", style={'color': '#2c3e50', 'margin-bottom': '15px'}),
                        dcc.Graph(id='performance-distribution', style={'height': '250px'})
                    ])
                    
                ], style={'width': '65%', 'float': 'right', 'padding': '0 20px'})
                
            ], style={'overflow': 'hidden'}),
            
            # 隐藏的存储组件
            dcc.Store(id='current-params', data=self.param_defaults),
            dcc.Store(id='perf-stats', data=perf_stats),
            dcc.Store(id='param-impact', data=param_impact)
            
        ], style={'font-family': 'Arial, sans-serif', 'margin': '20px', 'background-color': '#f8f9fa'})
        
        # 动态生成参数滑块
        slider_components = []
        for param in self.param_names:
            param_range = self.param_ranges[param]
            best_value = param_impact.get(param, {}).get('best_value', self.param_defaults[param])
            
            slider_components.append(
                html.Div([
                    html.Label(f"{param}:", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                    html.Div([
                        html.Span(f"当前: ", style={'font-size': '14px'}),
                        html.Span(id=f'value-{param}', 
                                 style={'font-weight': 'bold', 'color': '#007bff', 'font-size': '16px'}),
                        html.Span(f" (最优: {best_value})", 
                                 style={'color': '#28a745', 'font-size': '12px', 'margin-left': '10px'})
                    ], style={'margin-bottom': '5px'}),
                    dcc.Slider(
                        id=f'slider-{param}',
                        min=param_range['min'],
                        max=param_range['max'],
                        value=self.param_defaults[param],
                        step=1,
                        marks={v: str(v) for v in param_range['values'][::max(1, len(param_range['values'])//5)]},
                        tooltip={"placement": "bottom", "always_visible": False}
                    )
                ], style={'margin-bottom': '25px'})
            )
        
        # 更新滑块容器
        self.app.callback(
            Output('parameter-sliders', 'children'),
            Input('perf-stats', 'data')  # 触发器，确保在数据加载后执行
        )(lambda _: slider_components)
    
    def setup_callbacks(self):
        """设置回调函数"""
        
        # 参数滑块值显示回调
        for param in self.param_names:
            @self.app.callback(
                Output(f'value-{param}', 'children'),
                Input(f'slider-{param}', 'value')
            )
            def update_param_value(value, param=param):
                return str(value)
        
        
        # 主要回调：更新性能和图表
        @self.app.callback(
            [Output('current-performance', 'children'),
             Output('current-performance', 'style'),
             Output('performance-rank', 'children'),
             Output('current-params', 'data'),
             Output('parameter-impact-chart', 'figure'),
             Output('performance-distribution', 'figure')],
            [Input(f'slider-{param}', 'value') for param in self.param_names] +
            [Input('btn-optimal', 'n_clicks'),
             Input('btn-default', 'n_clicks')],
            [State('current-params', 'data'),
             State('perf-stats', 'data'),
             State('param-impact', 'data')]
        )
        def update_dashboard(*args):
            # 解析输入参数
            slider_values = args[:len(self.param_names)]
            btn_optimal_clicks = args[len(self.param_names)]
            btn_default_clicks = args[len(self.param_names) + 1]
            current_params = args[len(self.param_names) + 2]
            perf_stats = args[len(self.param_names) + 3]
            param_impact = args[len(self.param_names) + 4]
            
            # 检查是否点击了按钮
            ctx = callback_context
            if ctx.triggered:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                
                if button_id == 'btn-optimal':
                    # 设置为真实的最优配置
                    current_params = self.get_best_configuration()
                elif button_id == 'btn-default':
                    # 恢复默认配置
                    current_params = self.param_defaults.copy()
                else:
                    # 滑块变化
                    current_params = dict(zip(self.param_names, slider_values))
            else:
                current_params = dict(zip(self.param_names, slider_values))
            
            # 查询当前配置的性能
            performance = self.get_performance(current_params)
            
            if performance is not None:
                perf_text = f"{performance:.2f}"
                perf_style = {'font-size': '48px', 'font-weight': 'bold', 'color': '#2E8B57',
                             'text-align': 'center', 'margin': '10px 0'}
                
                # 计算排名
                better_count = len(self.df[self.df['performance'] > performance])
                total_count = len(self.df)
                rank = better_count + 1
                rank_text = f"排名: {rank}/{total_count} ({100-better_count/total_count*100:.1f}%)"
            else:
                perf_text = "无数据"
                perf_style = {'font-size': '48px', 'font-weight': 'bold', 'color': '#dc3545',
                             'text-align': 'center', 'margin': '10px 0'}
                
                # 分析约束违反原因
                constraint_check = self.check_constraints_detailed(current_params)
                if not constraint_check['valid'] and constraint_check['reasons']:
                    reasons_text = "约束违反: " + "; ".join(constraint_check['reasons'])
                    rank_text = reasons_text
                else:
                    # 如果通过约束检查但仍不存在，可能是数据集不完整
                    rank_text = "当前参数组合在数据集中不存在（可能未被测试）"
            
            # 生成参数影响力图表
            impact_fig = self.create_parameter_impact_chart(param_impact, current_params)
            
            # 生成性能分布图表
            dist_fig = self.create_performance_distribution_chart(performance, perf_stats)
            
            return perf_text, perf_style, rank_text, current_params, impact_fig, dist_fig
        
        # 按钮点击后更新滑块位置
        @self.app.callback(
            [Output(f'slider-{param}', 'value') for param in self.param_names],
            [Input('btn-optimal', 'n_clicks'),
             Input('btn-default', 'n_clicks')]
        )
        def update_sliders(btn_optimal_clicks, btn_default_clicks):
            ctx = callback_context
            if not ctx.triggered:
                return [self.param_defaults[param] for param in self.param_names]
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == 'btn-optimal':
                # 使用真实的最优配置
                best_config = self.get_best_configuration()
                return [best_config.get(param, self.param_defaults[param]) 
                       for param in self.param_names]
            elif button_id == 'btn-default':
                return [self.param_defaults[param] for param in self.param_names]
            else:
                return [self.param_defaults[param] for param in self.param_names]
    
    def create_parameter_impact_chart(self, param_impact: Dict, current_params: Dict = None):
        """创建参数性能分位数带图"""
        if not param_impact:
            return go.Figure()
        
        # 按影响力排序参数
        params = list(param_impact.keys())
        param_ranges = [(p, param_impact[p]['range']) for p in params]
        param_ranges.sort(key=lambda x: x[1], reverse=True)
        
        # 使用垂直排列，每个参数一行
        n_params = len(params)
        
        # 创建子图，垂直布局
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=n_params, cols=1,
            subplot_titles=[f"{p} (影响力: {param_impact[p]['impact_percentage']:.1f}%)" 
                          for p, _ in param_ranges],
            vertical_spacing=0.08,
            shared_xaxes=False
        )
        
        # 为每个参数创建分位数带图
        for i, (param, _) in enumerate(param_ranges):
            row = i + 1
            
            param_data = param_impact[param]
            values = param_data['values']
            mean_perf = param_data['mean_performance']
            
            # 计算每个参数值的统计数据
            x_values = []
            means = []
            medians = []
            q25_values = []
            q75_values = []
            q10_values = []
            q90_values = []
            min_values = []
            max_values = []
            
            for val in values:
                configs = self.get_configurations_for_param_value(param, val)
                performances = [c['performance'] for c in configs]
                
                if performances:
                    import numpy as np
                    x_values.append(val)
                    means.append(np.mean(performances))
                    medians.append(np.median(performances))
                    q25_values.append(np.percentile(performances, 25))
                    q75_values.append(np.percentile(performances, 75))
                    q10_values.append(np.percentile(performances, 10))
                    q90_values.append(np.percentile(performances, 90))
                    min_values.append(np.min(performances))
                    max_values.append(np.max(performances))
            
            if x_values:  # 确保有数据
                # 1. 添加极值范围填充（P10-P90）
                fig.add_trace(
                    go.Scatter(
                        x=x_values + x_values[::-1],
                        y=q90_values + q10_values[::-1],
                        fill='toself',
                        fillcolor='rgba(135,206,235,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='P10-P90区间' if i == 0 else '',
                        showlegend=(i == 0),
                        hoverinfo='skip'
                    ), row=row, col=1
                )
                
                # 2. 添加四分位数范围填充（P25-P75）
                fig.add_trace(
                    go.Scatter(
                        x=x_values + x_values[::-1],
                        y=q75_values + q25_values[::-1],
                        fill='toself',
                        fillcolor='rgba(70,130,180,0.3)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='P25-P75区间' if i == 0 else '',
                        showlegend=(i == 0),
                        hoverinfo='skip'
                    ), row=row, col=1
                )
                
                # 3. 添加中位数曲线（主要趋势线）
                fig.add_trace(
                    go.Scatter(
                        x=x_values, y=medians,
                        mode='lines+markers',
                        line=dict(color='rgb(70,130,180)', width=3),
                        marker=dict(size=6, color='rgb(70,130,180)'),
                        name='中位数趋势' if i == 0 else '',
                        showlegend=(i == 0),
                        hovertemplate=f'<b>{param}=%{{x}}</b><br>中位数: %{{y:.2f}}<extra></extra>'
                    ), row=row, col=1
                )
                
                # 4. 添加平均值曲线（虚线）
                fig.add_trace(
                    go.Scatter(
                        x=x_values, y=means,
                        mode='lines+markers',
                        line=dict(color='rgb(255,140,0)', width=2, dash='dash'),
                        marker=dict(size=4, color='rgb(255,140,0)'),
                        name='平均值趋势' if i == 0 else '',
                        showlegend=(i == 0),
                        hovertemplate=f'<b>{param}=%{{x}}</b><br>平均值: %{{y:.2f}}<extra></extra>'
                    ), row=row, col=1
                )
                
                # 5. 添加箱线图元素（每个参数值位置）
                for j, val in enumerate(x_values):
                    configs = self.get_configurations_for_param_value(param, val)
                    performances = [c['performance'] for c in configs]
                    
                    if len(performances) > 1:  # 只有多个数据点才显示箱线图
                        # 小型箱线图
                        fig.add_trace(
                            go.Box(
                                x=[val] * len(performances),
                                y=performances,
                                width=0.3,
                                name='',
                                showlegend=False,
                                marker_color='rgba(70,130,180,0.5)',
                                line_color='rgb(70,130,180)',
                                hovertemplate=f'<b>{param}={val}</b><br>' +
                                            f'性能: %{{y:.2f}}<br>' +
                                            f'配置数: {len(performances)}<extra></extra>'
                            ), row=row, col=1
                        )
                
                # 6. 标注最优值
                best_value = param_data['best_value']
                if best_value in x_values:
                    best_idx = x_values.index(best_value)
                    best_median = medians[best_idx]
                    fig.add_trace(
                        go.Scatter(
                            x=[best_value], y=[best_median],
                            mode='markers',
                            marker=dict(color='red', size=15, symbol='star', 
                                       line=dict(color='darkred', width=2)),
                            name='最优值' if i == 0 else '',
                            showlegend=(i == 0),
                            hovertemplate=f'<b>最优: {param}={best_value}</b><br>中位数: {best_median:.2f}<extra></extra>'
                        ), row=row, col=1
                    )
                
                # 7. 标注当前值
                if current_params and param in current_params:
                    current_value = current_params[param]
                    if current_value in x_values:
                        current_idx = x_values.index(current_value)
                        current_median = medians[current_idx]
                        fig.add_trace(
                            go.Scatter(
                                x=[current_value], y=[current_median],
                                mode='markers',
                                marker=dict(color='orange', size=12, symbol='diamond', 
                                           line=dict(color='darkorange', width=2)),
                                name='当前值' if i == 0 else '',
                                showlegend=(i == 0),
                                hovertemplate=f'<b>当前: {param}={current_value}</b><br>中位数: {current_median:.2f}<extra></extra>'
                            ), row=row, col=1
                        )
            
            # 更新子图坐标轴
            fig.update_xaxes(title_text=param, row=row, col=1)
            fig.update_yaxes(title_text="性能 (GB/s)", row=row, col=1)
        
        # 更新整体布局
        fig.update_layout(
            title='参数性能趋势分析（分位数带图）',
            height=400 * n_params,  # 每个参数400px高度
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=80, b=50)
        )
        
        return fig
    
    def create_performance_distribution_chart(self, current_performance: Optional[float], perf_stats: Dict):
        """创建性能分布核密度图"""
        fig = go.Figure()
        
        performances = self.df['performance'].values
        
        # 核密度估计
        from scipy.stats import gaussian_kde
        import numpy as np
        
        # 创建核密度估计器
        kde = gaussian_kde(performances)
        
        # 生成平滑的x值用于绘制密度曲线
        x_range = np.linspace(performances.min(), performances.max(), 200)
        density = kde(x_range)
        
        # 绘制核密度曲线
        fig.add_trace(go.Scatter(
            x=x_range,
            y=density,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(135,206,235,0.3)',
            line=dict(color='rgb(70,130,180)', width=2),
            name='性能密度分布',
            hovertemplate='性能: %{x:.2f}<br>密度: %{y:.4f}<extra></extra>'
        ))
        
        # 添加实际数据点（底部的strip plot）
        fig.add_trace(go.Scatter(
            x=performances,
            y=np.zeros(len(performances)),  # 底部位置
            mode='markers',
            marker=dict(
                color='rgba(70,130,180,0.6)',
                size=3,
                line=dict(width=0)
            ),
            name='实际配置',
            showlegend=False,
            hovertemplate='配置性能: %{x:.2f}<extra></extra>'
        ))
        
        # 计算并标记关键统计点
        percentiles = np.percentile(performances, [25, 50, 75, 90, 95])
        percentile_labels = ['P25', 'P50(中位数)', 'P75', 'P90', 'P95']
        colors = ['lightgreen', 'green', 'orange', 'red', 'darkred']
        
        for i, (perc, label, color) in enumerate(zip(percentiles, percentile_labels, colors)):
            # 在曲线上找到对应的密度值
            density_at_perc = kde([perc])[0]
            
            fig.add_trace(go.Scatter(
                x=[perc], y=[density_at_perc],
                mode='markers+text',
                marker=dict(color=color, size=8, symbol='circle'),
                text=[label],
                textposition='top center',
                showlegend=False,
                hovertemplate=f'<b>{label}</b><br>性能: {perc:.2f}<extra></extra>'
            ))
            
            # 添加垂直虚线
            fig.add_vline(
                x=perc,
                line_dash="dot",
                line_color=color,
                line_width=1,
                opacity=0.7
            )
        
        # 标记当前性能
        if current_performance is not None:
            current_density = kde([current_performance])[0]
            fig.add_trace(go.Scatter(
                x=[current_performance], y=[current_density],
                mode='markers+text',
                marker=dict(color='red', size=12, symbol='diamond', 
                           line=dict(color='darkred', width=2)),
                text=['当前配置'],
                textposition='top center',
                showlegend=False,
                hovertemplate=f'<b>当前配置</b><br>性能: {current_performance:.2f}<extra></extra>'
            ))
            
            fig.add_vline(
                x=current_performance,
                line_dash="solid",
                line_color="red",
                line_width=3,
                opacity=0.8
            )
        
        # 添加性能区间背景色
        max_density = density.max()
        
        # 优秀区间（P90以上）
        excellent_start = percentiles[3]  # P90
        excellent_end = performances.max()
        fig.add_shape(
            type="rect",
            x0=excellent_start, x1=excellent_end,
            y0=0, y1=max_density,
            fillcolor="rgba(0,255,0,0.1)",
            line=dict(width=0),
            layer="below"
        )
        
        # 良好区间（P75-P90）
        good_start = percentiles[2]  # P75
        good_end = percentiles[3]   # P90
        fig.add_shape(
            type="rect",
            x0=good_start, x1=good_end,
            y0=0, y1=max_density,
            fillcolor="rgba(255,255,0,0.1)",
            line=dict(width=0),
            layer="below"
        )
        
        fig.update_layout(
            title='整体性能分布分析（核密度图）',
            xaxis_title='性能 (GB/s)',
            yaxis_title='概率密度',
            height=350,
            showlegend=False,
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            plot_bgcolor='white'
        )
        
        return fig
    
    def run(self, host='127.0.0.1', port=8050, debug=False):
        """启动仪表板"""
        print(f"启动FIFO参数优化仪表板...")
        print(f"请在浏览器中打开: http://{host}:{port}")
        print(f"按 Ctrl+C 停止服务")
        
        try:
            # 优先尝试新版本的run方法
            self.app.run(host=host, port=port, debug=debug)
        except AttributeError:
            # 如果没有run方法，使用旧版本的run_server
            self.app.run_server(host=host, port=port, debug=debug)


def main():
    """主函数"""
    # 数据文件路径 - 请根据实际情况修改
    csv_path = "../Result/FIFO_Exhaustive/FIFO_Exhaustive_0904_1630/raw_data/parameter_performance.csv"
    
    try:
        # 检查文件是否存在
        if not os.path.exists(csv_path):
            print(f"错误: 找不到数据文件 {csv_path}")
            print("请确保已运行optimize_fifo_exhaustive.py生成数据")
            return
        
        # 创建并启动仪表板
        dashboard = FIFODashboard(csv_path)
        dashboard.run(debug=False)
        
    except Exception as e:
        print(f"启动失败: {e}")
        return


if __name__ == "__main__":
    main()