"""
LLM 推理模拟器核心

实现基于拓扑的 GPU/加速器侧精细模拟，包括：
- 数据搬运阶段（PCIe传输、HBM存储、权重加载）
- 推理计算阶段（细化为Attention/FFN/LayerNorm子操作）
- 结果收集阶段（HBM读取、PCIe回传）
"""
from __future__ import annotations

import time
from typing import Any
from dataclasses import dataclass, field

from .types import (
    LLMModelConfig, InferenceConfig, ParallelismStrategy,
    HardwareConfig, HierarchicalTopology,
    ChipHardwareConfig, NodeConfig, ClusterConfig,
    SimulationResult, SimulationStats, PhaseTimeStats,
    GanttTaskType, InferencePhase,
    get_bytes_per_element,
    MLAConfig,
)
from .topology import TopologyParser
from .latency import (
    calc_pcie_h2d_latency, calc_pcie_d2h_latency,
    calc_weight_load_latency, calc_embedding_latency,
    calc_layernorm_latency, calc_lm_head_latency,
    calc_attention_qkv_latency, calc_attention_score_latency,
    calc_attention_softmax_latency, calc_attention_output_latency,
    calc_ffn_gate_latency, calc_ffn_up_latency, calc_ffn_down_latency,
    calc_kv_cache_read_latency, calc_kv_cache_write_latency,
    calc_tp_allreduce_latency, calc_pp_p2p_latency,
    # MLA 专用
    calc_mla_q_projection_latency, calc_mla_kv_compression_latency,
    calc_mla_attention_score_latency, calc_mla_output_latency,
    calc_mla_kv_cache_read_latency, calc_mla_kv_cache_write_latency,
    # Kernel Fusion
    calc_fused_layernorm_qkv_latency, calc_fused_ffn_gate_up_latency,
    calc_single_layer_latency_fused, OVERLAP_COEFFICIENTS,
)
from .gantt import GanttChartBuilder, convert_to_frontend_format


@dataclass
class SimulationConfig:
    """模拟配置"""
    max_simulated_tokens: int = 16
    enable_data_transfer: bool = True
    enable_detailed_ops: bool = True
    enable_kv_cache: bool = True
    enable_overlap: bool = True
    # 新增: Kernel Fusion 和 MLA 优化
    enable_fusion: bool = True      # 启用 Kernel Fusion 优化
    enable_comm_overlap: bool = True  # 启用计算-通信重叠


@dataclass
class ChipState:
    """芯片状态"""
    chip_id: str
    pp_stage: int
    tp_rank: int
    current_time: float = 0.0
    compute_idle_at: float = 0.0
    network_idle_at: float = 0.0


class LLMInferenceSimulator:
    """LLM 推理模拟器"""

    def __init__(
        self,
        topology_dict: dict[str, Any],
        model: LLMModelConfig,
        inference: InferenceConfig,
        parallelism: ParallelismStrategy,
        hardware: HardwareConfig,
        config: SimulationConfig | None = None,
    ):
        """
        初始化模拟器

        Args:
            topology_dict: 前端拓扑配置
            model: 模型配置
            inference: 推理配置
            parallelism: 并行策略
            hardware: 硬件配置
            config: 模拟配置
        """
        self.model = model
        self.inference = inference
        self.parallelism = parallelism
        self.hardware = hardware
        self.config = config or SimulationConfig()

        # 解析拓扑
        self.topo_parser = TopologyParser(topology_dict, hardware)
        self.interconnect = self.topo_parser.build_interconnect_graph()
        self.group_assignment = self.topo_parser.map_parallelism(parallelism)

        # 获取 TP 组的链路参数
        if self.group_assignment.tp_groups and len(self.group_assignment.tp_groups[0]) > 1:
            self.tp_bandwidth, self.tp_latency = self.topo_parser.get_link_params_for_group(
                self.group_assignment.tp_groups[0], 'allreduce'
            )
        else:
            self.tp_bandwidth = hardware.node.intra_node_bandwidth_gbps
            self.tp_latency = hardware.node.intra_node_latency_us

        # 获取 PP 组的链路参数
        if self.group_assignment.pp_groups and len(self.group_assignment.pp_groups[0]) > 1:
            self.pp_bandwidth, self.pp_latency = self.topo_parser.get_link_params_for_group(
                self.group_assignment.pp_groups[0], 'p2p'
            )
        else:
            self.pp_bandwidth = hardware.cluster.inter_node_bandwidth_gbps
            self.pp_latency = hardware.cluster.inter_node_latency_us

        # 甘特图构建器
        self.gantt_builder = GanttChartBuilder(parallelism)

        # 芯片状态
        self.chip_states: dict[str, ChipState] = {}
        self._init_chip_states()

        # 统计
        self.prefill_stats = PhaseTimeStats()
        self.decode_stats = PhaseTimeStats()

    def _init_chip_states(self):
        """初始化芯片状态"""
        for assignment in self.group_assignment.assignments:
            self.chip_states[assignment.chip_id] = ChipState(
                chip_id=assignment.chip_id,
                pp_stage=assignment.pp_rank,
                tp_rank=assignment.tp_rank,
            )

    def simulate(self) -> SimulationResult:
        """
        运行完整模拟

        Returns:
            模拟结果
        """
        start_time = time.time()

        current_time = 0.0

        # 阶段1: 数据搬运 (H2D)
        if self.config.enable_data_transfer:
            current_time = self._simulate_data_transfer_h2d(current_time)

        # 阶段2: Prefill 推理
        prefill_end_time = self._simulate_prefill(current_time)
        phase_transition = prefill_end_time

        # 阶段3: Decode 推理
        decode_end_time = self._simulate_decode(prefill_end_time)

        # 阶段4: 数据收集 (D2H)
        if self.config.enable_data_transfer:
            final_time = self._simulate_data_transfer_d2h(decode_end_time)
        else:
            final_time = decode_end_time

        # 构建甘特图
        gantt_data = self.gantt_builder.build(phase_transition=phase_transition)

        # 计算统计信息
        stats = self._compute_stats(final_time)

        return SimulationResult(
            gantt_chart=gantt_data,
            stats=stats,
            timestamp=time.time(),
        )

    def _simulate_data_transfer_h2d(self, start_time: float) -> float:
        """模拟 Host to Device 数据传输"""
        # 计算输入数据大小
        bytes_per_elem = get_bytes_per_element(self.model.dtype)
        input_size_gb = (
            self.inference.batch_size * self.inference.input_seq_length *
            self.model.hidden_size * bytes_per_elem
        ) / (1024 ** 3)

        # PCIe 传输延迟
        pcie_latency = calc_pcie_h2d_latency(input_size_gb, self.hardware)

        # 为第一个 PP stage 的所有芯片添加传输任务
        for chip_id, state in self.chip_states.items():
            if state.pp_stage == 0:
                self.gantt_builder.add_task(
                    name="PCIe H2D",
                    start=start_time,
                    end=start_time + pcie_latency,
                    task_type=GanttTaskType.PCIE_H2D,
                    phase=InferencePhase.PREFILL,
                    chip_id=chip_id,
                    pp_stage=0,
                )
                state.compute_idle_at = start_time + pcie_latency

        return start_time + pcie_latency

    def _simulate_data_transfer_d2h(self, start_time: float) -> float:
        """模拟 Device to Host 数据传输"""
        # 计算输出数据大小 (logits)
        bytes_per_elem = get_bytes_per_element(self.model.dtype)
        output_size_gb = (
            self.inference.batch_size * self.model.vocab_size * bytes_per_elem
        ) / (1024 ** 3)

        # PCIe 传输延迟
        pcie_latency = calc_pcie_d2h_latency(output_size_gb, self.hardware)

        # 为最后一个 PP stage 的所有芯片添加传输任务
        last_stage = self.parallelism.pp - 1
        for chip_id, state in self.chip_states.items():
            if state.pp_stage == last_stage:
                self.gantt_builder.add_task(
                    name="PCIe D2H",
                    start=start_time,
                    end=start_time + pcie_latency,
                    task_type=GanttTaskType.PCIE_D2H,
                    phase=InferencePhase.DECODE,
                    chip_id=chip_id,
                    pp_stage=last_stage,
                )

        return start_time + pcie_latency

    def _simulate_prefill(self, start_time: float) -> float:
        """模拟 Prefill 阶段"""
        num_tokens = self.inference.input_seq_length
        context_length = self.inference.input_seq_length

        # 每个 PP stage 处理的层数
        layers_per_stage = self.model.num_layers // self.parallelism.pp

        # 为每个 PP stage 模拟
        stage_times = [start_time] * self.parallelism.pp

        for layer in range(self.model.num_layers):
            pp_stage = layer // layers_per_stage
            if pp_stage >= self.parallelism.pp:
                pp_stage = self.parallelism.pp - 1

            layer_in_stage = layer % layers_per_stage

            # 获取该 stage 的第一个芯片
            chip_id = self._get_chip_for_stage(pp_stage)
            current_time = stage_times[pp_stage]

            # PP 前向传递等待上一个 stage
            if pp_stage > 0 and layer_in_stage == 0:
                prev_stage_end = stage_times[pp_stage - 1]
                if prev_stage_end > current_time:
                    # 添加气泡
                    bubble_duration = prev_stage_end - current_time
                    self.gantt_builder.add_bubble(
                        start=current_time,
                        duration=bubble_duration,
                        phase=InferencePhase.PREFILL,
                        chip_id=chip_id,
                        pp_stage=pp_stage,
                    )
                    current_time = prev_stage_end

                    # PP P2P 通信
                    pp_comm_latency = self._calc_pp_comm_latency(num_tokens)
                    self.gantt_builder.add_comm_task(
                        task_type=GanttTaskType.PP_COMM,
                        start=current_time,
                        duration=pp_comm_latency,
                        phase=InferencePhase.PREFILL,
                        chip_id=chip_id,
                        pp_stage=pp_stage,
                        layer_index=layer,
                    )
                    current_time += pp_comm_latency

            # 模拟单层
            current_time = self._simulate_single_layer(
                current_time=current_time,
                layer_index=layer,
                num_tokens=num_tokens,
                context_length=context_length,
                phase=InferencePhase.PREFILL,
                chip_id=chip_id,
                pp_stage=pp_stage,
            )

            stage_times[pp_stage] = current_time

        # Embedding (在第一层之前) 和 LM Head (在最后一层之后) 已包含在层计算中
        # 返回最后一个 stage 的结束时间
        prefill_end = max(stage_times)

        # 更新统计
        self.prefill_stats.total_time = prefill_end - start_time

        return prefill_end

    def _simulate_decode(self, start_time: float) -> float:
        """模拟 Decode 阶段"""
        current_time = start_time
        num_tokens_to_simulate = min(
            self.config.max_simulated_tokens,
            self.inference.output_seq_length
        )

        layers_per_stage = self.model.num_layers // self.parallelism.pp

        for token_idx in range(num_tokens_to_simulate):
            context_length = self.inference.input_seq_length + token_idx + 1
            stage_times = [current_time] * self.parallelism.pp

            for layer in range(self.model.num_layers):
                pp_stage = layer // layers_per_stage
                if pp_stage >= self.parallelism.pp:
                    pp_stage = self.parallelism.pp - 1

                layer_in_stage = layer % layers_per_stage
                chip_id = self._get_chip_for_stage(pp_stage)
                layer_start = stage_times[pp_stage]

                # PP 等待
                if pp_stage > 0 and layer_in_stage == 0:
                    prev_end = stage_times[pp_stage - 1]
                    if prev_end > layer_start:
                        bubble = prev_end - layer_start
                        self.gantt_builder.add_bubble(
                            start=layer_start,
                            duration=bubble,
                            phase=InferencePhase.DECODE,
                            chip_id=chip_id,
                            pp_stage=pp_stage,
                        )
                        layer_start = prev_end

                        pp_comm = self._calc_pp_comm_latency(1)
                        self.gantt_builder.add_comm_task(
                            task_type=GanttTaskType.PP_COMM,
                            start=layer_start,
                            duration=pp_comm,
                            phase=InferencePhase.DECODE,
                            chip_id=chip_id,
                            pp_stage=pp_stage,
                            layer_index=layer,
                            token_index=token_idx,
                        )
                        layer_start += pp_comm

                # 模拟单层 (Decode: 1 token)
                layer_end = self._simulate_single_layer(
                    current_time=layer_start,
                    layer_index=layer,
                    num_tokens=1,
                    context_length=context_length,
                    phase=InferencePhase.DECODE,
                    chip_id=chip_id,
                    pp_stage=pp_stage,
                    token_index=token_idx,
                )

                stage_times[pp_stage] = layer_end

            current_time = max(stage_times)

        # 更新统计
        self.decode_stats.total_time = current_time - start_time

        return current_time

    def _simulate_single_layer(
        self,
        current_time: float,
        layer_index: int,
        num_tokens: int,
        context_length: int,
        phase: InferencePhase,
        chip_id: str,
        pp_stage: int,
        token_index: int | None = None,
    ) -> float:
        """模拟单层 Transformer"""

        # 检查是否使用 MLA
        use_mla = self.model.attention_type == 'mla' and self.model.mla_config is not None

        if self.config.enable_detailed_ops:
            # 细粒度模拟

            # LayerNorm 1
            ln1_latency = calc_layernorm_latency(
                self.model, self.inference, self.parallelism, self.hardware, num_tokens
            )
            self.gantt_builder.add_compute_task(
                GanttTaskType.LAYERNORM, current_time, ln1_latency,
                phase, chip_id, pp_stage, layer_index, token_index
            )
            current_time += ln1_latency

            # Attention QKV (MLA 使用专用计算)
            if use_mla:
                # MLA: Q 投影 + KV 压缩
                qkv_latency = calc_mla_q_projection_latency(
                    self.model, self.inference, self.parallelism, self.hardware, num_tokens
                )
                kv_compress_latency = calc_mla_kv_compression_latency(
                    self.model, self.inference, self.parallelism, self.hardware, num_tokens
                )
                qkv_latency += kv_compress_latency
            else:
                qkv_latency = calc_attention_qkv_latency(
                    self.model, self.inference, self.parallelism, self.hardware, num_tokens
                )
            self.gantt_builder.add_compute_task(
                GanttTaskType.ATTENTION_QKV, current_time, qkv_latency,
                phase, chip_id, pp_stage, layer_index, token_index
            )
            current_time += qkv_latency

            # KV Cache 读取 (Decode 阶段)
            if self.config.enable_kv_cache and phase == InferencePhase.DECODE:
                if use_mla:
                    # MLA: 压缩后的 KV Cache (~32x 小)
                    kv_read_latency = calc_mla_kv_cache_read_latency(
                        self.model, self.inference, self.parallelism, self.hardware, context_length
                    )
                else:
                    kv_read_latency = calc_kv_cache_read_latency(
                        self.model, self.inference, self.parallelism, self.hardware, context_length
                    )
                self.gantt_builder.add_compute_task(
                    GanttTaskType.KV_CACHE_READ, current_time, kv_read_latency,
                    phase, chip_id, pp_stage, layer_index, token_index
                )
                current_time += kv_read_latency

            # Attention Score
            if use_mla:
                # MLA: 在压缩空间计算 Score
                score_latency = calc_mla_attention_score_latency(
                    self.model, self.inference, self.parallelism, self.hardware, num_tokens, context_length
                )
            else:
                score_latency = calc_attention_score_latency(
                    self.model, self.inference, self.parallelism, self.hardware, num_tokens, context_length
                )
            self.gantt_builder.add_compute_task(
                GanttTaskType.ATTENTION_SCORE, current_time, score_latency,
                phase, chip_id, pp_stage, layer_index, token_index
            )
            current_time += score_latency

            # Softmax (MLA 和标准 Attention 相同)
            softmax_latency = calc_attention_softmax_latency(
                self.model, self.inference, self.parallelism, self.hardware, num_tokens, context_length
            )
            self.gantt_builder.add_compute_task(
                GanttTaskType.ATTENTION_SOFTMAX, current_time, softmax_latency,
                phase, chip_id, pp_stage, layer_index, token_index
            )
            current_time += softmax_latency

            # Attention Output
            if use_mla:
                # MLA: V 解压缩 + Softmax@V + Output 投影
                output_latency = calc_mla_output_latency(
                    self.model, self.inference, self.parallelism, self.hardware, num_tokens, context_length
                )
            else:
                output_latency = calc_attention_output_latency(
                    self.model, self.inference, self.parallelism, self.hardware, num_tokens, context_length
                )
            self.gantt_builder.add_compute_task(
                GanttTaskType.ATTENTION_OUTPUT, current_time, output_latency,
                phase, chip_id, pp_stage, layer_index, token_index
            )
            current_time += output_latency

            # TP AllReduce (Attention)
            if self.parallelism.tp > 1:
                tp_comm_latency = self._calc_tp_allreduce_latency(num_tokens)
                self.gantt_builder.add_comm_task(
                    GanttTaskType.TP_COMM, current_time, tp_comm_latency,
                    phase, chip_id, pp_stage, layer_index, token_index
                )
                current_time += tp_comm_latency

            # KV Cache 写入
            if self.config.enable_kv_cache:
                if use_mla:
                    # MLA: 压缩后的 KV Cache 写入 (~32x 小)
                    kv_write_latency = calc_mla_kv_cache_write_latency(
                        self.model, self.inference, self.parallelism, self.hardware, num_tokens
                    )
                else:
                    kv_write_latency = calc_kv_cache_write_latency(
                        self.model, self.inference, self.parallelism, self.hardware, num_tokens
                    )
                self.gantt_builder.add_compute_task(
                    GanttTaskType.KV_CACHE_WRITE, current_time, kv_write_latency,
                    phase, chip_id, pp_stage, layer_index, token_index
                )
                current_time += kv_write_latency

            # LayerNorm 2
            ln2_latency = calc_layernorm_latency(
                self.model, self.inference, self.parallelism, self.hardware, num_tokens
            )
            self.gantt_builder.add_compute_task(
                GanttTaskType.LAYERNORM, current_time, ln2_latency,
                phase, chip_id, pp_stage, layer_index, token_index
            )
            current_time += ln2_latency

            # FFN Gate
            gate_latency = calc_ffn_gate_latency(
                self.model, self.inference, self.parallelism, self.hardware, num_tokens
            )
            self.gantt_builder.add_compute_task(
                GanttTaskType.FFN_GATE, current_time, gate_latency,
                phase, chip_id, pp_stage, layer_index, token_index
            )
            current_time += gate_latency

            # FFN Up
            up_latency = calc_ffn_up_latency(
                self.model, self.inference, self.parallelism, self.hardware, num_tokens
            )
            self.gantt_builder.add_compute_task(
                GanttTaskType.FFN_UP, current_time, up_latency,
                phase, chip_id, pp_stage, layer_index, token_index
            )
            current_time += up_latency

            # FFN Down
            down_latency = calc_ffn_down_latency(
                self.model, self.inference, self.parallelism, self.hardware, num_tokens
            )
            self.gantt_builder.add_compute_task(
                GanttTaskType.FFN_DOWN, current_time, down_latency,
                phase, chip_id, pp_stage, layer_index, token_index
            )
            current_time += down_latency

            # TP AllReduce (FFN)
            if self.parallelism.tp > 1:
                tp_comm_latency = self._calc_tp_allreduce_latency(num_tokens)
                self.gantt_builder.add_comm_task(
                    GanttTaskType.TP_COMM, current_time, tp_comm_latency,
                    phase, chip_id, pp_stage, layer_index, token_index
                )
                current_time += tp_comm_latency

        else:
            # 粗粒度模拟 - 整层计算
            layer_latency = self._calc_layer_latency_coarse(num_tokens, context_length)
            self.gantt_builder.add_compute_task(
                GanttTaskType.COMPUTE, current_time, layer_latency,
                phase, chip_id, pp_stage, layer_index, token_index
            )
            current_time += layer_latency

            # TP 通信
            if self.parallelism.tp > 1:
                tp_comm = self._calc_tp_allreduce_latency(num_tokens) * 2  # Attn + FFN
                self.gantt_builder.add_comm_task(
                    GanttTaskType.TP_COMM, current_time, tp_comm,
                    phase, chip_id, pp_stage, layer_index, token_index
                )
                current_time += tp_comm

        return current_time

    def _calc_tp_allreduce_latency(self, num_tokens: int) -> float:
        """计算 TP AllReduce 延迟"""
        bytes_per_elem = get_bytes_per_element(self.model.dtype)
        data_size_gb = (
            self.inference.batch_size * num_tokens * self.model.hidden_size * bytes_per_elem
        ) / (1024 ** 3)

        return calc_tp_allreduce_latency(
            data_size_gb, self.tp_bandwidth, self.tp_latency, self.parallelism.tp
        )

    def _calc_pp_comm_latency(self, num_tokens: int) -> float:
        """计算 PP P2P 通信延迟"""
        bytes_per_elem = get_bytes_per_element(self.model.dtype)
        data_size_gb = (
            self.inference.batch_size * num_tokens * self.model.hidden_size * bytes_per_elem
        ) / (1024 ** 3)

        return calc_pp_p2p_latency(data_size_gb, self.pp_bandwidth, self.pp_latency)

    def _calc_layer_latency_coarse(self, num_tokens: int, context_length: int) -> float:
        """粗粒度计算单层延迟"""
        # 简化计算：主要是 Attention 和 FFN
        attn_latency = (
            calc_attention_qkv_latency(self.model, self.inference, self.parallelism, self.hardware, num_tokens) +
            calc_attention_score_latency(self.model, self.inference, self.parallelism, self.hardware, num_tokens, context_length) +
            calc_attention_output_latency(self.model, self.inference, self.parallelism, self.hardware, num_tokens, context_length)
        )
        ffn_latency = (
            calc_ffn_gate_latency(self.model, self.inference, self.parallelism, self.hardware, num_tokens) +
            calc_ffn_up_latency(self.model, self.inference, self.parallelism, self.hardware, num_tokens) +
            calc_ffn_down_latency(self.model, self.inference, self.parallelism, self.hardware, num_tokens)
        )
        return attn_latency + ffn_latency

    def _get_chip_for_stage(self, pp_stage: int) -> str:
        """获取指定 PP stage 的第一个芯片ID"""
        for assignment in self.group_assignment.assignments:
            if assignment.pp_rank == pp_stage:
                return assignment.chip_id
        raise ValueError(f"找不到 PP stage {pp_stage} 的芯片")

    def _compute_stats(self, total_time: float) -> SimulationStats:
        """计算统计信息"""
        # TTFT = Prefill 总时间
        ttft = self.prefill_stats.total_time

        # 平均 TPOT
        num_decode_tokens = min(self.config.max_simulated_tokens, self.inference.output_seq_length)
        avg_tpot = self.decode_stats.total_time / num_decode_tokens if num_decode_tokens > 0 else 0.0

        # 计算 MFU (简化版本)
        bytes_per_elem = get_bytes_per_element(self.model.dtype)

        # Prefill 阶段 MFU
        # MFU = 实际 FLOPs/s / 峰值 FLOPs/s
        # 注意: prefill_flops 是单个 DP 副本的 FLOPs (不需要乘 DP)
        # peak_tflops 应该是单个 DP 副本使用的芯片总算力 (tp * pp)
        prefill_flops = self._calc_total_flops(self.inference.input_seq_length)
        prefill_mfu = 0.0
        if self.prefill_stats.total_time > 0:
            # 时间单位: ms -> s
            time_s = self.prefill_stats.total_time / 1000
            achieved_tflops = (prefill_flops / 1e12) / time_s

            # 单 DP 副本的峰值算力 (tp * pp 个芯片)
            # 注意: 不乘 dp，因为每个 dp 副本独立计算相同 FLOPs
            chips_per_replica = self.parallelism.tp * self.parallelism.pp
            peak_tflops = self.hardware.chip.compute_tflops_fp16 * chips_per_replica

            prefill_mfu = achieved_tflops / peak_tflops

        # Decode 阶段 MBU (内存带宽利用率)
        # MBU = 实际带宽需求 / 峰值带宽
        # 实际带宽需求 = (模型权重 + KV Cache) / TPOT
        decode_mbu = 0.0
        if num_decode_tokens > 0 and avg_tpot > 0:
            # 模型权重大小
            model_size_gb = self._calc_model_size_gb()

            # KV Cache 大小 (平均 context 长度)
            avg_context = self.inference.input_seq_length + num_decode_tokens // 2
            kv_cache_gb = self._calc_kv_cache_size_gb(avg_context)

            # 总数据量
            data_read_gb = model_size_gb + kv_cache_gb

            # 实际带宽需求 (GB/s)
            required_bandwidth = data_read_gb / (avg_tpot / 1000)

            # 峰值带宽 (考虑 HBM 效率 85%)
            peak_bandwidth = self.hardware.chip.memory_bandwidth_gbps * 0.85
            decode_mbu = required_bandwidth / peak_bandwidth

        return SimulationStats(
            prefill=self.prefill_stats,
            decode=self.decode_stats,
            total_run_time=total_time,
            simulated_tokens=num_decode_tokens,
            ttft=ttft,
            avg_tpot=avg_tpot,
            dynamic_mfu=min(prefill_mfu, 1.0),
            dynamic_mbu=min(decode_mbu, 1.0),
            max_pp_bubble_ratio=0.0,  # TODO: 计算气泡比
            total_events=len(self.gantt_builder.tasks),
        )

    def _calc_total_flops(self, seq_length: int) -> float:
        """
        计算总 FLOPs

        标准 Transformer FLOPs 计算:
        - QKV Projection: 2 * B * S * H * (H + 2 * kv_heads * head_dim)  (考虑 GQA)
        - Attention Score: 2 * B * n_heads * S * S * head_dim
        - Attention Output: 2 * B * n_heads * S * S * head_dim + 2 * B * S * H * H
        - FFN: 3 * 2 * B * S * H * I (gate, up, down)
        - LM Head: 2 * B * S * H * V

        简化公式: 约等于 2 * num_params * seq_length
        """
        B = self.inference.batch_size
        S = seq_length
        H = self.model.hidden_size
        L = self.model.num_layers
        I = self.model.intermediate_size
        V = self.model.vocab_size
        n_heads = self.model.num_attention_heads
        kv_heads = self.model.num_kv_heads
        head_dim = H // n_heads

        # QKV Projection (考虑 GQA)
        qkv_flops = 2 * B * S * H * (H + 2 * kv_heads * head_dim) * L

        # Attention Score: Q @ K^T
        score_flops = 2 * B * n_heads * S * S * head_dim * L

        # Attention Output: Softmax @ V + Output Projection
        output_flops = (2 * B * n_heads * S * S * head_dim + 2 * B * S * H * H) * L

        # FFN: gate, up, down
        ffn_flops = 2 * B * S * H * I * 3 * L

        # LM Head
        lm_head_flops = 2 * B * S * H * V

        return qkv_flops + score_flops + output_flops + ffn_flops + lm_head_flops

    def _calc_model_size_gb(self) -> float:
        """计算模型大小 (GB)"""
        bytes_per_elem = get_bytes_per_element(self.model.dtype)
        H = self.model.hidden_size
        L = self.model.num_layers
        I = self.model.intermediate_size
        V = self.model.vocab_size

        # Attention weights: (Q, K, V, O) per layer
        attn_params = 4 * H * H * L

        # FFN weights: (gate, up, down) per layer
        ffn_params = 3 * H * I * L

        # Embedding (LM Head 通常与 Embedding 共享权重)
        embed_params = V * H

        total_params = attn_params + ffn_params + embed_params
        return (total_params * bytes_per_elem) / (1024 ** 3)

    def _calc_kv_cache_size_gb(self, context_length: int) -> float:
        """计算 KV Cache 大小 (GB)"""
        bytes_per_elem = get_bytes_per_element(self.model.dtype)
        B = self.inference.batch_size
        H = self.model.hidden_size
        L = self.model.num_layers
        num_heads = self.model.num_attention_heads
        num_kv_heads = self.model.num_kv_heads
        head_dim = H // num_heads

        # KV Cache: 2 (K+V) × batch × context × kv_heads × head_dim × layers
        kv_cache_bytes = 2 * B * context_length * num_kv_heads * head_dim * L * bytes_per_elem
        return kv_cache_bytes / (1024 ** 3)


def run_simulation(
    topology_dict: dict[str, Any],
    model_dict: dict[str, Any],
    inference_dict: dict[str, Any],
    parallelism_dict: dict[str, Any],
    hardware_dict: dict[str, Any],
    config_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    运行模拟的入口函数

    Args:
        topology_dict: 拓扑配置
        model_dict: 模型配置
        inference_dict: 推理配置
        parallelism_dict: 并行策略
        hardware_dict: 硬件配置
        config_dict: 模拟配置

    Returns:
        模拟结果字典
    """
    # 解析 MLA 配置 (DeepSeek V3/R1)
    mla_config = None
    mla_dict = model_dict.get("mla_config")
    if mla_dict:
        mla_config = MLAConfig(
            kv_lora_rank=mla_dict["kv_lora_rank"],
            q_lora_rank=mla_dict["q_lora_rank"],
            qk_nope_head_dim=mla_dict["qk_nope_head_dim"],
            qk_rope_head_dim=mla_dict["qk_rope_head_dim"],
            v_head_dim=mla_dict["v_head_dim"],
        )

    # 解析配置
    model = LLMModelConfig(
        model_name=model_dict.get("model_name", "Unknown"),
        model_type=model_dict.get("model_type", "dense"),
        hidden_size=model_dict["hidden_size"],
        num_layers=model_dict["num_layers"],
        num_attention_heads=model_dict["num_attention_heads"],
        num_kv_heads=model_dict.get("num_kv_heads", model_dict["num_attention_heads"]),
        intermediate_size=model_dict["intermediate_size"],
        vocab_size=model_dict.get("vocab_size", 32000),
        dtype=model_dict.get("dtype", "fp16"),
        max_seq_length=model_dict.get("max_seq_length", 4096),
        attention_type=model_dict.get("attention_type", "gqa"),
        mla_config=mla_config,
    )

    inference = InferenceConfig(
        batch_size=inference_dict["batch_size"],
        input_seq_length=inference_dict["input_seq_length"],
        output_seq_length=inference_dict["output_seq_length"],
        max_seq_length=inference_dict.get("max_seq_length", 4096),
    )

    parallelism = ParallelismStrategy(
        dp=parallelism_dict.get("dp", 1),
        tp=parallelism_dict.get("tp", 1),
        pp=parallelism_dict.get("pp", 1),
        ep=parallelism_dict.get("ep", 1),
        sp=parallelism_dict.get("sp", 1),
    )

    chip_hw = hardware_dict.get("chip", {})
    node_hw = hardware_dict.get("node", {})
    cluster_hw = hardware_dict.get("cluster", {})

    hardware = HardwareConfig(
        chip=ChipHardwareConfig(
            chip_type=chip_hw.get("chip_type", "H100-SXM"),
            compute_tflops_fp16=chip_hw.get("compute_tflops_fp16", 989),
            memory_gb=chip_hw.get("memory_gb", 80),
            memory_bandwidth_gbps=chip_hw.get("memory_bandwidth_gbps", 3350),
            pcie_bandwidth_gbps=chip_hw.get("pcie_bandwidth_gbps", 64),
            pcie_latency_us=chip_hw.get("pcie_latency_us", 1),
            hbm_random_access_latency_ns=chip_hw.get("hbm_random_access_latency_ns", 100),
        ),
        node=NodeConfig(
            chips_per_node=node_hw.get("chips_per_node", 8),
            intra_node_bandwidth_gbps=node_hw.get("intra_node_bandwidth_gbps", 900),
            intra_node_latency_us=node_hw.get("intra_node_latency_us", 1),
        ),
        cluster=ClusterConfig(
            num_nodes=cluster_hw.get("num_nodes", 1),
            inter_node_bandwidth_gbps=cluster_hw.get("inter_node_bandwidth_gbps", 400),
            inter_node_latency_us=cluster_hw.get("inter_node_latency_us", 2),
        ),
    )

    config = SimulationConfig(
        max_simulated_tokens=config_dict.get("maxSimulatedTokens", 16) if config_dict else 16,
        enable_data_transfer=config_dict.get("enableDataTransferSimulation", True) if config_dict else True,
        enable_detailed_ops=config_dict.get("enableDetailedTransformerOps", True) if config_dict else True,
        enable_kv_cache=config_dict.get("enableKVCacheAccessSimulation", True) if config_dict else True,
    )

    # 运行模拟
    simulator = LLMInferenceSimulator(
        topology_dict=topology_dict,
        model=model,
        inference=inference,
        parallelism=parallelism,
        hardware=hardware,
        config=config,
    )

    result = simulator.simulate()

    # 转换为前端格式
    from .gantt import convert_to_frontend_format

    return {
        "ganttChart": convert_to_frontend_format(result.gantt_chart),
        "stats": {
            "prefill": {
                "computeTime": result.stats.prefill.compute_time,
                "commTime": result.stats.prefill.comm_time,
                "bubbleTime": result.stats.prefill.bubble_time,
                "overlapTime": result.stats.prefill.overlap_time,
                "totalTime": result.stats.prefill.total_time,
                "computeEfficiency": result.stats.prefill.compute_efficiency,
            },
            "decode": {
                "computeTime": result.stats.decode.compute_time,
                "commTime": result.stats.decode.comm_time,
                "bubbleTime": result.stats.decode.bubble_time,
                "overlapTime": result.stats.decode.overlap_time,
                "totalTime": result.stats.decode.total_time,
                "computeEfficiency": result.stats.decode.compute_efficiency,
            },
            "totalRunTime": result.stats.total_run_time,
            "simulatedTokens": result.stats.simulated_tokens,
            "ttft": result.stats.ttft,
            "avgTpot": result.stats.avg_tpot,
            "dynamicMfu": result.stats.dynamic_mfu,
            "dynamicMbu": result.stats.dynamic_mbu,
            "maxPPBubbleRatio": result.stats.max_pp_bubble_ratio,
            "totalEvents": result.stats.total_events,
        },
        "timestamp": result.timestamp,
    }
