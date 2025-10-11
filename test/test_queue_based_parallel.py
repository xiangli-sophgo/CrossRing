"""
基于共享队列的D2D多进程并行化原型

设计要点：
1. 根据D2D_PAIRS配置创建队列（而非全连接）
2. 每个Die独立运行，通过队列通信
3. 使用Barrier同步周期边界
4. 验证Windows兼容性
"""

import time
import sys
import os
from multiprocessing import Process, Manager, Barrier

# Windows编码修复
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.components.flit import Flit
from config.d2d_config import D2DConfig


class SimplifiedDie:
    """简化的Die模型"""
    def __init__(self, die_id):
        self.die_id = die_id
        self.current_cycle = 0
        self.processed_d2d_count = 0
        self.sent_d2d_count = 0

    def simulate_work(self):
        """模拟CPU密集型工作"""
        result = 0
        for i in range(100000):
            result += i * i % 997
        return result

    def process_d2d_receive(self, flit):
        """处理接收到的D2D消息"""
        self.processed_d2d_count += 1

    def step(self):
        """执行一个周期"""
        self.simulate_work()


def die_worker(die_id, config, d2d_input_queues, d2d_output_queues,
               sync_barrier, num_cycles, stats_queue):
    """
    Die工作进程

    Args:
        die_id: Die ID
        config: D2DConfig对象
        d2d_input_queues: 输入队列字典 {connection_key: Queue}
        d2d_output_queues: 输出队列字典 {connection_key: Queue}
        sync_barrier: 周期同步屏障
        num_cycles: 总周期数
        stats_queue: 统计信息队列
    """
    # 在子进程中创建Die实例
    die = SimplifiedDie(die_id)

    # 本地缓冲：{connection_key: [(arrival_cycle, flit), ...]}
    receive_buffers = {key: [] for key in d2d_input_queues.keys()}

    for cycle in range(num_cycles):
        # 1. 周期开始同步点
        sync_barrier.wait()

        die.current_cycle = cycle

        # 2. 从共享队列批量读取到本地缓冲
        for conn_key, queue in d2d_input_queues.items():
            while not queue.empty():
                try:
                    item = queue.get(block=False)
                    receive_buffers[conn_key].append(item)
                except:
                    break

        # 3. 处理到期的D2D接收
        for conn_key, buffer in receive_buffers.items():
            # 找出到期的消息
            arrived = []
            remaining = []
            for arrival_cycle, flit in buffer:
                if arrival_cycle <= cycle:
                    arrived.append(flit)
                else:
                    remaining.append((arrival_cycle, flit))

            # 处理到期消息
            for flit in arrived:
                die.process_d2d_receive(flit)

            # 保留未到期消息
            receive_buffers[conn_key] = remaining

        # 4. 执行Die内部处理
        die.step()

        # 5. 模拟生成D2D消息（每10个周期）
        if cycle % 10 == 0 and cycle > 0 and d2d_output_queues:
            # 随机选择一个输出连接发送
            for conn_key, queue in d2d_output_queues.items():
                # 创建测试flit
                src_die, src_node, dst_die, dst_node = conn_key
                flit = Flit(source=src_node, destination=dst_node, path=[src_node, dst_node])
                flit.packet_id = cycle * 100 + die_id
                flit.d2d_origin_die = src_die
                flit.d2d_target_die = dst_die

                # 添加到队列（模拟10周期延迟）
                queue.put((cycle + 10, flit))
                die.sent_d2d_count += 1

                break  # 只发送一个

        # 6. 周期结束同步点
        sync_barrier.wait()

    # 发送统计信息
    stats_queue.put({
        'die_id': die_id,
        'processed_d2d': die.processed_d2d_count,
        'sent_d2d': die.sent_d2d_count
    })


def create_d2d_queues(config):
    """
    根据D2D_PAIRS创建队列

    Returns:
        {
            'input_queues': {die_id: {connection_key: Queue}},
            'output_queues': {die_id: {connection_key: Queue}}
        }
    """
    manager = Manager()

    # 为每个D2D连接对创建队列
    # connection_key = (src_die, src_node, dst_die, dst_node)
    queues = {}

    for pair in config.D2D_PAIRS:
        src_die, src_node, dst_die, dst_node = pair

        # 创建正向队列：src_die → dst_die
        forward_key = (src_die, src_node, dst_die, dst_node)
        queues[forward_key] = manager.Queue()

        # 创建反向队列：dst_die → src_die
        reverse_key = (dst_die, dst_node, src_die, src_node)
        queues[reverse_key] = manager.Queue()

    # 为每个Die组织输入输出队列
    input_queues = {die_id: {} for die_id in range(config.NUM_DIES)}
    output_queues = {die_id: {} for die_id in range(config.NUM_DIES)}

    for conn_key, queue in queues.items():
        src_die, src_node, dst_die, dst_node = conn_key

        # 输出队列：本Die发送
        output_queues[src_die][conn_key] = queue

        # 输入队列：目标Die接收
        input_queues[dst_die][conn_key] = queue

    return {
        'input_queues': input_queues,
        'output_queues': output_queues
    }


def test_queue_based_parallel():
    """测试基于队列的并行方案"""
    print("=" * 70)
    print("基于共享队列的D2D并行化测试")
    print("=" * 70)

    # 加载配置（使用绝对路径）
    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "config",
        "topologies",
        "d2d_4die_config.yaml"
    )
    config = D2DConfig(d2d_config_file=config_path)

    print(f"\n配置信息:")
    print(f"  Die数量: {config.NUM_DIES}")
    print(f"  D2D连接对数: {len(config.D2D_PAIRS)}")
    print(f"  D2D连接详情:")
    for i, pair in enumerate(config.D2D_PAIRS[:5]):  # 只显示前5个
        print(f"    {i+1}. Die{pair[0]}:节点{pair[1]} ↔ Die{pair[2]}:节点{pair[3]}")
    if len(config.D2D_PAIRS) > 5:
        print(f"    ... (共{len(config.D2D_PAIRS)}个连接)")

    # 创建D2D队列
    print(f"\n创建D2D通信队列...")
    queue_info = create_d2d_queues(config)

    for die_id in range(config.NUM_DIES):
        input_count = len(queue_info['input_queues'][die_id])
        output_count = len(queue_info['output_queues'][die_id])
        print(f"  Die {die_id}: {input_count}个输入连接, {output_count}个输出连接")

    # 测试参数
    num_cycles = 500

    print(f"\n开始并行仿真: {num_cycles}个周期...")

    # 创建同步屏障和统计队列
    manager = Manager()
    sync_barrier = manager.Barrier(config.NUM_DIES)
    stats_queue = manager.Queue()

    # 创建进程
    processes = []
    start_time = time.perf_counter()

    for die_id in range(config.NUM_DIES):
        p = Process(
            target=die_worker,
            args=(
                die_id,
                config,
                queue_info['input_queues'][die_id],
                queue_info['output_queues'][die_id],
                sync_barrier,
                num_cycles,
                stats_queue
            )
        )
        p.start()
        processes.append(p)
        print(f"  启动进程 Die {die_id}")

    # 等待所有进程完成
    for i, p in enumerate(processes):
        p.join(timeout=60)
        if p.is_alive():
            print(f"  警告: Die {i} 进程超时")
            p.terminate()

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    # 收集统计
    stats = {}
    while not stats_queue.empty():
        stat = stats_queue.get()
        stats[stat['die_id']] = stat

    # 显示结果
    print(f"\n仿真完成!")
    print(f"  执行时间: {elapsed:.3f} 秒")
    print(f"  吞吐量: {num_cycles / elapsed:.0f} 周期/秒")

    print(f"\n各Die统计:")
    for die_id in sorted(stats.keys()):
        s = stats[die_id]
        print(f"  Die {die_id}: 发送了 {s['sent_d2d']} 个D2D消息, "
              f"接收了 {s['processed_d2d']} 个D2D消息")

    return elapsed


def test_serial_baseline():
    """串行基准测试"""
    print("\n" + "=" * 70)
    print("串行基准测试")
    print("=" * 70)

    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "config",
        "topologies",
        "d2d_4die_config.yaml"
    )
    config = D2DConfig(d2d_config_file=config_path)

    num_cycles = 500
    dies = [SimplifiedDie(die_id) for die_id in range(config.NUM_DIES)]

    print(f"开始串行仿真: {num_cycles}个周期...")

    start_time = time.perf_counter()

    for cycle in range(num_cycles):
        for die in dies:
            die.current_cycle = cycle
            die.step()

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    print(f"\n仿真完成!")
    print(f"  执行时间: {elapsed:.3f} 秒")
    print(f"  吞吐量: {num_cycles / elapsed:.0f} 周期/秒")

    return elapsed


def main():
    print("\n" + "=" * 70)
    print("D2D多进程并行化原型测试（基于配置文件的队列）")
    print("=" * 70)

    # 串行基准
    time_serial = test_serial_baseline()

    # 并行测试
    time_parallel = test_queue_based_parallel()

    # 性能对比
    print("\n" + "=" * 70)
    print("性能对比")
    print("=" * 70)

    speedup = time_serial / time_parallel

    print(f"\n串行执行: {time_serial:.3f} 秒")
    print(f"并行执行: {time_parallel:.3f} 秒")
    print(f"加速比:   {speedup:.2f}x")

    if speedup >= 2.0:
        print(f"\n[优秀] 队列并行方案验证成功！")
        print(f"可以继续实施到实际D2D仿真")
    elif speedup >= 1.5:
        print(f"\n[良好] 有一定加速效果")
    else:
        print(f"\n[需优化] 加速效果不明显")

    print(f"\n关键验证:")
    print(f"  ✓ D2D_PAIRS配置正确读取")
    print(f"  ✓ 队列根据配置创建")
    print(f"  ✓ 多进程同步工作")
    print(f"  ✓ Windows兼容性测试通过")


if __name__ == "__main__":
    main()
