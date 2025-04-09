import numpy as np
import random


def generate_data(topo, end_time, file_name, sdma_pos, gdma_pos, ddr_pos, l2m_pos, speed, burst, flow_type=0, mix_ratios=None):
    """
    flow_type:
        0: 32-shared (默认)
        1: 8-shared 分组
        2: private (自己发给自己)
        3: 混合模式 (需要指定mix_ratios)
    mix_ratios: 当flow_type=3时使用，格式为 {0: ratio1, 1: ratio2, 2: ratio3}
    """
    f = open(file_name, "w", encoding="utf-8")
    data_all = []

    def generate_entries(src_pos, src_type, dest_type, dest_pos, operation, burst, flow_type):
        time = [128 // (speed[burst] // burst) * i for i in range(speed[burst] // burst)]

        if flow_type == 1:  # 8-shared分组模式
            for group_start in range(0, len(src_pos), 8):
                group_end = min(group_start + 8, len(src_pos))
                current_src_pos = src_pos[group_start:group_end]
                current_dest_pos = dest_pos[group_start:group_end]

                for src in current_src_pos:
                    rand_index = []
                    for i in range(end_time):
                        for j in time:
                            if not rand_index:
                                rand_index = random.sample(range(len(current_dest_pos)), len(current_dest_pos))
                            index = rand_index.pop()
                            data_all.append(f"{128 * i + j},{src},{src_type},{current_dest_pos[index]},{dest_type},{operation},{burst}\n")

        elif flow_type == 2:  # private模式 (自己发给自己)
            for src in src_pos:
                for i in range(end_time):
                    for j in time:
                        # 找到对应的dest位置 (src可能不在dest_pos中)
                        dest = src if src in dest_pos else dest_pos[src % len(dest_pos)]
                        data_all.append(f"{128 * i + j},{src},{src_type},{dest},{dest_type},{operation},{burst}\n")

        else:  # 原始32-shared模式
            for src in src_pos:
                rand_index = []
                for i in range(end_time):
                    for j in time:
                        if not rand_index:
                            rand_index = random.sample(range(len(dest_pos)), len(dest_pos))
                        index = rand_index.pop()
                        data_all.append(f"{128 * i + j},{src},{src_type},{dest_pos[index]},{dest_type},{operation},{burst}\n")

    def generate_mixed_entries(src_pos, src_type, dest_type, dest_pos, operation, burst, ratios):
        """混合模式生成函数"""
        time = [128 // (speed[burst] // burst) * i for i in range(speed[burst] // burst)]
        total_entries = len(src_pos) * end_time * len(time)

        # 计算每种模式需要的条目数
        counts = {k: int(total_entries * v) for k, v in ratios.items()}
        remaining = total_entries - sum(counts.values())
        counts[max(ratios, key=ratios.get)] += remaining  # 将剩余条目分配给比例最大的模式

        # 为每个src分配模式
        mode_assignments = []
        for mode, count in counts.items():
            mode_assignments.extend([mode] * count)
        random.shuffle(mode_assignments)

        # 按分配的模式生成数据
        for src in src_pos:
            mode_iter = iter(mode_assignments)
            for i in range(end_time):
                for j in time:
                    mode = next(mode_iter)
                    if mode == 0:  # 32-shared
                        dest = random.choice(dest_pos)
                    elif mode == 1:  # 8-shared
                        group = src // 8
                        group_start = group * 8
                        group_end = min(group_start + 8, len(dest_pos))
                        dest = random.choice(dest_pos[group_start:group_end])
                    else:  # private
                        dest = src if src in dest_pos else dest_pos[src % len(dest_pos)]

                    data_all.append(f"{128 * i + j},{src},{src_type},{dest},{dest_type},{operation},{burst}\n")

    if topo in ["4x9", "9x4", "4x5", "5x4"]:
        if flow_type == 3:  # 混合模式
            if not mix_ratios:
                mix_ratios = {0: 0.4, 1: 0.4, 2: 0.2}  # 默认比例
            generate_mixed_entries(sdma_pos, "gdma", "ddr", ddr_pos, "R", burst, mix_ratios)
            generate_mixed_entries(sdma_pos, "gdma", "ddr", l2m_pos, "W", burst, mix_ratios)
        else:
            generate_entries(sdma_pos, "gdma", "ddr", ddr_pos, "R", burst, flow_type)
            generate_entries(sdma_pos, "gdma", "ddr", l2m_pos, "W", burst, flow_type)

    elif topo == "3x3":
        if flow_type == 3:  # 混合模式
            if not mix_ratios:
                mix_ratios = {0: 0.4, 1: 0.4, 2: 0.2}  # 默认比例
            generate_mixed_entries(sdma_pos, "sdma", "ddr", ddr_pos, "R", burst, mix_ratios)
            generate_mixed_entries(sdma_pos, "sdma", "l2m", l2m_pos, "W", burst, mix_ratios)
            generate_mixed_entries(gdma_pos, "gdma", "l2m", l2m_pos, "R", burst, mix_ratios)
        else:
            generate_entries(sdma_pos, "sdma", "ddr", ddr_pos, "R", burst, flow_type)
            generate_entries(sdma_pos, "sdma", "l2m", l2m_pos, "W", burst, flow_type)
            generate_entries(gdma_pos, "gdma", "l2m", l2m_pos, "R", burst, flow_type)

    sorted_data = sorted(data_all, key=lambda x: int(x.strip().split(",")[0]))
    f.writelines(sorted_data)
    f.close()


def main():
    # 示例参数配置
    np.random.seed(409)
    topo = "5x4"
    end_time = 128
    file_name = "../../test_data/demo_54_64_8_shared.txt"

    sdma_pos = range(32)
    gdma_pos = range(32)
    ddr_pos = range(32)
    l2m_pos = range(32)

    # sdma_pos = [0, 2, 6, 8]
    # gdma_pos = [0, 2, 6, 8]
    # ddr_pos = [0, 2, 3, 3, 5, 5, 6, 8]
    # l2m_pos = [0, 0, 1, 1, 7, 7, 8, 8]

    speed = {1: 128, 2: 68, 4: 64}
    burst = 4

    # 调用生成数据的函数
    custom_ratios = {0: 0.4, 1: 0.4, 2: 0.2}
    generate_data(topo, end_time, file_name, sdma_pos, gdma_pos, ddr_pos, l2m_pos, speed, burst, flow_type=1, mix_ratios=custom_ratios)


if __name__ == "__main__":
    main()
