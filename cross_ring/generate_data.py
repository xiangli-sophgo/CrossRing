import numpy as np
import random

f = open("demo3.txt", "w", encoding="utf-8")
end = 64
m = 8
np.random.seed(12)
for i in range(end):
    # print(f"{1 * i + 1},2,sdma,3,ddr,R,4", file=f)
    for j in range(32):
        # rand_dest = np.random.choice(range(32), replace=False)
        rand_dest = np.random.choice(range(32), size=32, replace=False).tolist()
        rand_src = np.random.choice(range(32), size=32, replace=False).tolist()
        # rand_src = 4
        for k in range(32):
            # rand_dest = 10
            print(f"{i* 32 * m + (j + 1) * m},{rand_src[k]},gdma,{rand_dest[k]},ddr,{'R'},{random.randint(4, 4)}", file=f)
            # print(f"{i + 1},{rand_src},gdma,{rand_dest},ddr,{'W'},{random.randint(4, 4)}", file=f)

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
f.close()
