import math
import os


def shared_8(addr, base=16):
    addr = int(addr, base=base)
    if 0x400000000 <= addr < 0x406000000:
        return 0
    if 0x4a0000000 <= addr < 0x4a6000000:
        return 1
    if 0x540000000 <= addr < 0x546000000:
        return 2
    if 0x5e0000000 <= addr < 0x5e6000000:
        return 3
    if 0x680000000 <= addr < 0x686000000:
        return 4
    if 0x720000000 <= addr < 0x726000000:
        return 5
    if 0x7c0000000 <= addr < 0x7c6000000:
        return 6
    if 0x860000000 <= addr < 0x866000000:
        return 7
    else:
        return 'not 8-shared'


def hash_addr2node(addr, node_num, outstanding_digit, base=16):
    # 输入字符串格式16进制地址addr,节点数量node_num
    # 输出hash结果(0到node_num-1)
    # 确保 nodes 是 2 的幂
    assert (node_num & (node_num - 1) == 0) and node_num != 0, "nodes 必须是 2 的幂"
    addr = int(addr, base=base)
    # 计算目标路由器
    addr_valid = addr >> outstanding_digit
    sel_bits = int(math.log2(node_num))

    # 初始化选择位向量
    sel_bit_vec = [[] for _ in range(sel_bits)]

    i = 0
    while addr_valid:
        bit = addr_valid % 2
        sel_bit_vec[i % sel_bits].append(bit)
        i += 1
        addr_valid >>= 1

    # 计算选择位
    sel_bit = [0] * sel_bits
    for i in range(sel_bits):
        for bit in sel_bit_vec[i]:
            sel_bit[i] ^= bit

    # 计算最终的选择值
    sel = 0
    for i in range(sel_bits):
        sel |= (sel_bit[i] << i)

    return sel


address = '0x4a0300000'
cluster = shared_8(address)
hash_value = hash_addr2node(address,8,9)
print(f'cluster={cluster},hash_value={hash_value}')