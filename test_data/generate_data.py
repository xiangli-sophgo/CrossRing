import numpy as np
import random

topo = "3x3"

if topo in ["4x9", "9x4", "4x5", "5x4"]:
    f = open("../test_data/demo_459.txt", "w", encoding="utf-8")
    end_time = 64
    sdma_pos = range(32)
    gdma_pos = range(32)
    ddr_pos = range(32)
    l2m_pos = range(32)
    rn_num = 32
    np.random.seed(12)
    burst = 4
    speed = {1: 128, 2: 68, 4: 64}

    time = [128 // (speed[burst] // burst) * i for i in range(speed[burst] // burst)]
    # print(time)
    data_all = []

    for src in sdma_pos:
        rand_index = []
        for i in range(end_time):
            for j in time:
                if not rand_index:
                    rand_index = random.sample(range(rn_num), rn_num)
                index = rand_index.pop()
                data_all.append(f"{128 * i + j},{src},sdma,{ddr_pos[index]},ddr,R,{burst}\n")

    for src in sdma_pos:
        rand_index = []
        for i in range(end_time):
            for j in time:
                if not rand_index:
                    rand_index = random.sample(range(rn_num), rn_num)
                index = rand_index.pop()
                data_all.append(f"{128 * i + j},{src},sdma,{l2m_pos[index]},ddr,W,{burst}\n")

    sorted_data = sorted(data_all, key=lambda x: int(x.strip().split(",")[0]))
    f.writelines(sorted_data)
    f.close()


elif topo == "3x3":
    # 3x3
    f = open("../test_data/demo_3x3.txt", "w", encoding="utf-8")
    end_time = 128
    sdma_pos = [0, 2, 6, 8]
    gdma_pos = [0, 2, 6, 8]
    ddr_pos = [0, 2, 3, 3, 5, 5, 6, 8]
    l2m_pos = [0, 0, 1, 1, 7, 7, 8, 8]
    np.random.seed(12)
    burst = 2
    speed = {1: 128, 2: 68, 4: 128}

    time = [128 // (speed[burst] // burst) * i for i in range(speed[burst] // burst)]
    # print(time)
    data_all = []

    for src in sdma_pos:
        rand_index = []
        for i in range(end_time):
            for j in time:
                if not rand_index:
                    rand_index = random.sample(range(8), 8)
                index = rand_index.pop()
                data_all.append(f"{128 * i + j},{src},sdma,{ddr_pos[index]},ddr,R,{burst}\n")

    for src in sdma_pos:
        rand_index = []
        for i in range(end_time):
            for j in time:
                if not rand_index:
                    rand_index = random.sample(range(8), 8)
                index = rand_index.pop()
                data_all.append(f"{128 * i + j},{src},sdma,{l2m_pos[index]},l2m,W,{burst}\n")

    time = [128 // (128 // burst) * i for i in range(128 // burst)]

    for src in gdma_pos:
        rand_index = []
        for i in range(end_time):
            for j in time:
                if not rand_index:
                    rand_index = random.sample(range(8), 8)
                index = rand_index.pop()
                data_all.append(f"{128 * i + j},{src},gdma,{l2m_pos[index]},l2m,R,{burst}\n")

    sorted_data = sorted(data_all, key=lambda x: int(x.strip().split(",")[0]))
    f.writelines(sorted_data)
    f.close()

# # 4x9/9x4/4x5/5x4
# f = open("../test_data/demo3.txt", "w", encoding="utf-8")
# end = 64
# m = 8
# np.random.seed(12)
# for i in range(end):
#     # print(f"{1 * i + 1},2,sdma,3,ddr,R,4", file=f)
#     for j in range(32):
#         # rand_dest = np.random.choice(range(32), replace=False)
#         rand_dest = np.random.choice(range(32), size=32, replace=False).tolist()
#         rand_src = np.random.choice(range(32), size=32, replace=False).tolist()
#         # rand_src = 4
#         for k in range(32):
#             # rand_dest = 10
#             print(f"{i* 32 * m + (j + 1) * m},{rand_src[k]},gdma,{rand_dest[k]},ddr,{'R'},{random.randint(4, 4)}", file=f)
#             print(f"{i* 32 * m + (j + 1) * m},{rand_src[k]},gdma,{rand_dest[k]},ddr,{'W'},{random.randint(4, 4)}", file=f)

# for i in range(end):
#     # print(f"{1 * i + 1},2,sdma,3,ddr,R,4", file=f)
#     # for j in range(4):
#     # rand_dest = np.random.choice(range(32), replace=False)
#     rand_dest_index = np.random.choice(range(8), replace=False).tolist()
#     rand_src_index = np.random.choice(range(8), replace=False).tolist()
#     # rand_src = 4
#     for k in range(8):
#         # src = rand_src[k]
#         # dest = sdma_pos[j]  # 当前SDMA位置
#         # rand_dest = 10
#         if rand_dest_index[k] >= 4:
#             print(f"{i* 32 * m + (j + 1) * m},{sdma_pos[rand_dest_index[k]]},sdma,{rand_dest[k]},ddr,{'R'},{random.randint(4, 4)}", file=f)
#         # print(f"{i* 32 * m + (j + 1) * m},{rand_src[k]},sdma,{rand_dest[k]},l2m,{'W'},{random.randint(4, 4)}", file=f)

# print(f"{i * 1 + 1},{j},gdma,{rand_dest},ddr,{'W'},4", file=f)

# for j in range(32):
#     src_id = j % 16
#     rand_dest = np.random.choice(rwrite_after_read# , replace=False) % 16
#     if j < 16:
#         print(f"{i * 1 + 1},{src_id},{random.choice(['gdma', 'sdma'])},{rand_dest},ddr,{'R'},4", file=f)
# print(f"{i * 1 + 1},{src_id},sdma,{rand_dest},ddr,{'R'},4", file=f)
# print(f"{i * 1 + 1},{j},gdma,{rand_dest},ddr,{'W'},4", file=f)
# else:
#     print(f"{i * 1 + 1},{src_id},sdma,{rand_dest},ddr,{'R'},4", file=f)
# print(f"{i * 1 + 1},{j},sdma,{rand_dest},ddr,{'W'},4", file=f)
# f.close()
