import math
import os
import re


class AddressHasher:
    def __init__(self, itlv_size=2048):
        self.itlv_size = itlv_size
        self.itlv_digit = itlv_size.bit_length() - 1
        self.shared_32ch_count = 0
        self.shared_16ch_count = 0
        self.shared_8ch_count = 0
        self.private_count = 0
        self.request_end_time = -1

    def hash_all(self, addr, base=16):
        addr = int(addr, base=base)
        if 0x80000000 <= addr < 0x100000000:
            self.shared_32ch_count += 1  # 统计 32通道 shared
            return self.shared_32ch(addr)
        elif 0x100000000 <= addr < 0x500000000:
            self.shared_16ch_count += 1  # 统计 16通道 shared
            return self.shared_16ch(addr)
        elif 0x500000000 <= addr < 0x700000000:
            self.shared_8ch_count += 1  # 统计 8通道 shared
            return self.shared_8ch(addr)
        elif 0x700000000 <= addr < 0x1F00000000:
            self.private_count += 1  # 统计 private
            return self.private(addr)
        else:
            raise ValueError(f"Address error: {hex(addr)}")

    def shared_32ch(self, addr):
        """处理 32通道共享内存: 0x80000000 - 0xffffffff (2GB)"""
        assert 0x80000000 <= addr < 0x100000000, f"Address {hex(addr)} not in 32CH range"
        res = self.hash_addr2node(addr, 32)
        assert res < 32
        return res

    def shared_16ch(self, addr):
        """处理 16通道共享内存: 0x100000000 - 0x4ffffffff (16GB, 2个8GB区域)"""
        assert 0x100000000 <= addr < 0x500000000, f"Address {hex(addr)} not in 16CH range"
        shared_16ch_size = 0x200000000  # 8GB per region
        cluster_id = (addr - 0x100000000) // shared_16ch_size
        hash_value = self.hash_addr2node(addr, 16)
        return cluster_id * 16 + hash_value

    def shared_8ch(self, addr):
        """处理 8通道共享内存: 0x500000000 - 0x6ffffffff (8GB, 4个2GB区域)"""
        assert 0x500000000 <= addr < 0x700000000, f"Address {hex(addr)} not in 8CH range"
        shared_8ch_size = 0x80000000  # 2GB per region
        cluster_id = (addr - 0x500000000) // shared_8ch_size
        hash_value = self.hash_addr2node(addr, 8)
        return cluster_id * 8 + hash_value

    def private(self, addr):
        """处理私有内存: 0x700000000 - 0x1effffffff (96GB, 32个3GB区域)"""
        assert 0x700000000 <= addr < 0x1F00000000, f"Address {hex(addr)} not in private range"
        private_size = 0xC0000000  # 3GB per region
        region_id = (addr - 0x700000000) // private_size
        assert region_id < 32, f"Private region ID {region_id} exceeds 32 regions"
        return region_id

    def hash_addr2node(self, addr, node_num):
        # 输入字符串格式16进制地址addr,节点数量node_num
        # 输出hash结果(0到node_num-1)
        # 确保 nodes 是 2 的幂
        assert (node_num & (node_num - 1) == 0) and node_num != 0, "nodes must be a power of 2"

        # 计算目标路由器
        addr_valid = addr >> self.itlv_digit
        sel_bits = int(math.log2(node_num))

        # 初始化选择位向量
        sel_bit_vec = [[] for _ in range(sel_bits)]

        i = 0
        while addr_valid:
            bit = addr_valid % 2
            sel_bit_vec[i % sel_bits].append(bit)
            i += 1
            addr_valid >>= 1

        # 计算最终的选择值
        sel_bit = [0] * sel_bits
        for i in range(sel_bits):
            for bit in sel_bit_vec[i]:
                sel_bit[i] ^= bit

        # 计算最终选择值
        sel = sum(sel_bit[i] << i for i in range(sel_bits))
        return sel

    # def ip2node(self, n):
    #     """Input ddr/dma number and return the corresponding node number"""
    #     n = int(n)
    #     # return str(n)
    #     # ip2node_map = {0: 0, 4: 8, 8: 4, 12: 12, 16: 20, 20: 28, 24: 36, 28: 44, 32: 52, 36: 60, 40: 48, 44: 56, 48: 32, 52: 40, 56: 16, 60: 24}
    #     # ip2node_map = {0: 0, 4: 8, 8: 4, 12: 12, 16: 16, 20: 24, 24: 20, 28: 28}
    #     ip2node_map = {0: 0, 4: 8, 8: 4, 12: 12, 16: 16, 20: 24, 24: 20, 28: 28}
    #     remaining = n % 4
    #     n_1 = ip2node_map[n - remaining] + remaining
    #     ip2node_map_2 = {0: 0, 2: 4, 4: 2, 6: 6, 8: 8, 10: 12, 12: 10, 14: 14, 16: 16, 18: 20, 20: 18, 22: 22, 24: 24, 26: 28, 28: 26, 30: 30}  # 每 2 个一组映射
    #     remaining = n_1 % 2
    #     return str(ip2node_map_2[n_1 - remaining] + remaining)

    def ip2node(self, n):
        """
        32 core 映射到 2x2 cluster
        """
        n = int(n)
        group = n // 8
        subgroup = (n % 8) // 4
        parity = n % 2

        base_map = [[0, 16], [2, 18], [8, 24], [10, 26]]  # 组0  # 组1  # 组2  # 组3
        base = base_map[group][subgroup]
        offset = 4 * ((n % 4) // 2)  # 偏移量为0或4
        return str(base + offset + parity)

    def process_file(self, file_path, input_folder, output_folder, file_name):
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # 如果存在标题行则跳过
        if lines and lines[0].startswith("begin_req"):
            lines = lines[1:]

        # 处理每一行
        processed_lines = []
        # 从文件名提取TPU编号 (例如: gmemTrace01.tpu_0.gdma_instance.csv -> 0)
        tpu_match = re.search(r"tpu_(\d+)", file_name)
        if not tpu_match:
            raise ValueError(f"无法从文件名中提取TPU编号: {file_name}")
        tpu_num = tpu_match.group(1)

        for line in lines:
            data = line.strip().split(",")  # 假设数据以逗号分隔
            if len(data) < 8:
                continue  # 跳过无效行

            processed_data = [
                data[1],  # end_req (请求结束时间)
                self.ip2node(tpu_num),  # 源节点
                "gdma",  # 源类型 (固定为gdma)
                self.ip2node(self.hash_all(addr=data[4])),  # 目标节点 (地址hash结果)
                "ddr",  # 目标类型 (固定为ddr)
                data[5],  # r/w (请求类型)
                data[7],  # burst_len (burst长度)
            ]
            processed_lines.append(",".join(processed_data))  # 转换回逗号分隔的字符串

        # 使用最后一行的end_req更新请求结束时间
        if lines:
            self.request_end_time = max(self.request_end_time, int(lines[-1].strip().split(",")[1]))

        # 确保输出目录存在
        output_file_path = os.path.join(output_folder, os.path.relpath(file_path, start=input_folder))
        # 将输出文件扩展名改为.txt
        output_file_path = os.path.splitext(output_file_path)[0] + ".txt"
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        # 将处理后的数据写入新文件
        with open(output_file_path, "w", encoding="utf-8") as file:
            for line in processed_lines:
                file.write(line + "\n")

    def process_folder(self, input_folder, output_folder):
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                # 检查文件是否匹配TPU模式
                if not re.search(r"tpu_\d+", file):
                    print(f"跳过非TPU文件: {file}")
                    continue

                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) == 0:
                    print(f"跳过空文件: {file}")
                    continue
                self.process_file(file_path, input_folder, output_folder, file)

            # 输出当前目录的统计信息
            if not files:
                continue
            total_request = self.shared_32ch_count + self.shared_16ch_count + self.shared_8ch_count + self.private_count
            print(f"Directory: {root[27:]}")
            print(f"32CH shared requests: {self.shared_32ch_count}, {self.shared_32ch_count / total_request:.4f}")
            print(f"16CH shared requests: {self.shared_16ch_count}, {self.shared_16ch_count / total_request:.4f}")
            print(f"8CH shared requests: {self.shared_8ch_count}, {self.shared_8ch_count / total_request:.4f}")
            print(f"Private requests: {self.private_count}, {self.private_count / total_request:.4f}")
            print(f"Total requests: {total_request}")
            print(f"Request end time: {self.request_end_time}")
            self.shared_32ch_count = 0
            self.shared_16ch_count = 0
            self.shared_8ch_count = 0
            self.private_count = 0

    def run(self, input_folder, output_folder):
        self.process_folder(input_folder, output_folder)
        print("Processing has been completed.")


# 主要执行部分
if __name__ == "__main__":
    hasher = AddressHasher()

    input_folder = r"../../traffic/DeepSeek_0918/step1_flatten"
    output_folder = "../output/step2_hash_addr2node"
    hasher.run(input_folder, output_folder)
