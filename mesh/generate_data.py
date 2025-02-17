import numpy as np
import random

f = open("demo3.txt", "w", encoding="utf-8")
end = 1
for i in range(end):
    # print(f"{1 * i + 1},2,cdma,3,ddr,R,4", file=f)
    for j in range(32):
        rand_src = np.random.choice(range(32), replace=False)
        # rand_src = 4
        for k in range(8):
            # rand_dest = np.random.choice(range(32), replace=False)
            rand_dest = 2
            print(f"{i * 2 + 1},{rand_src},gdma,{rand_dest},ddr,{'R'},{random.randint(1, 1)}", file=f)
            # print(f"{i * 2 + 1},{rand_src},gdma,{rand_dest},ddr,{'W'},{random.randint(1, 4)}", file=f)
    # print(f"{i * 1 + 1},{j},gdma,{rand_dest},ddr,{'W'},4", file=f)

    # for j in range(32):
    #     src_id = j % 16
    #     rand_dest = np.random.choice(range(32), replace=False) % 16
    #     if j < 16:
    #         print(f"{i * 1 + 1},{src_id},{random.choice(['gdma', 'cdma'])},{rand_dest},ddr,{'R'},4", file=f)
    # print(f"{i * 1 + 1},{src_id},cdma,{rand_dest},ddr,{'R'},4", file=f)
    # print(f"{i * 1 + 1},{j},gdma,{rand_dest},ddr,{'W'},4", file=f)
    # else:
    #     print(f"{i * 1 + 1},{src_id},cdma,{rand_dest},ddr,{'R'},4", file=f)
    # print(f"{i * 1 + 1},{j},cdma,{rand_dest},ddr,{'W'},4", file=f)
f.close()
