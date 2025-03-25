import numpy as np
import random


def generate_data(topo, end_time, file_name, sdma_pos, gdma_pos, ddr_pos, l2m_pos, speed, burst):
    f = open(file_name, "w", encoding="utf-8")
    data_all = []

    def generate_entries(src_pos, src_type, dest_type, dest_pos, operation, burst):
        time = [128 // (speed[burst] // burst) * i for i in range(speed[burst] // burst)]
        for src in src_pos:
            rand_index = []
            for i in range(end_time):
                for j in time:
                    if not rand_index:
                        rand_index = random.sample(range(len(dest_pos)), len(dest_pos))
                    index = rand_index.pop()
                    data_all.append(f"{128 * i + j},{src},{src_type},{dest_pos[index]},{dest_type},{operation},{burst}\n")

    if topo in ["4x9", "9x4", "4x5", "5x4"]:
        generate_entries(sdma_pos, "sdma", "ddr", ddr_pos, "R", burst)
        generate_entries(sdma_pos, "sdma", "ddr", l2m_pos, "W", burst)

    elif topo == "3x3":
        generate_entries(sdma_pos, "sdma", "ddr", ddr_pos, "R", burst)
        generate_entries(sdma_pos, "sdma", "l2m", l2m_pos, "W", burst)
        generate_entries(gdma_pos, "gdma", "l2m", l2m_pos, "R", burst)

    sorted_data = sorted(data_all, key=lambda x: int(x.strip().split(",")[0]))
    f.writelines(sorted_data)
    f.close()


def main():
    # 示例参数配置
    np.random.seed(12)
    topo = "5x4"
    end_time = 64
    file_name = "../../test_data/demo45.txt"

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
    generate_data(topo, end_time, file_name, sdma_pos, gdma_pos, ddr_pos, l2m_pos, speed, burst)


if __name__ == "__main__":
    main()
