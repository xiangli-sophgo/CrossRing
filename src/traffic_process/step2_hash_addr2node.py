import math
import os
import re


class AddressHasher:
    def __init__(self, itlv_size=512):
        self.itlv_size = itlv_size
        self.itlv_digit = itlv_size.bit_length() - 1
        self.shared_64_count = 0
        self.shared_8_count = 0
        self.private_count = 0
        self.request_end_time = -1

    def hash_all(self, addr, base=16):
        addr = int(addr, base=base)
        if 0x04_0000_0000 <= addr < 0x08_0000_0000:
            self.shared_64_count += 1  # 统计 64 shared
            return self.shared_64(addr)
        elif 0x08_0000_0000 <= addr < 0x10_0000_0000:
            self.shared_8_count += 1  # 统计 8 shared
            return self.shared_8(addr)
        elif 0x10_0000_0000 <= addr < 0x20_0000_0000:
            self.private_count += 1  # 统计 private
            return self.private(addr)
        else:
            raise ValueError(f"Address error: {hex(addr)}")

    def shared_64(self, addr):
        assert 0x04_0000_0000 <= addr < 0x08_0000_0000
        res = self.hash_addr2node(addr, 32)
        assert res < 32
        return res

    def shared_8(self, addr):
        assert 0x08_0000_0000 <= addr < 0x10_0000_0000
        shared_8_size = 0x01_0000_0000
        cluster_id = (addr - 0x08_0000_0000) // shared_8_size
        hash_value = self.hash_addr2node(addr, 8)
        return cluster_id * 8 + hash_value

    def private(self, addr):
        assert 0x10_0000_0000 <= addr < 0x20_0000_0000
        shared_8_size = 0x00_4000_0000
        return (addr - 0x10_0000_0000) // shared_8_size

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

        # Calculate final selection value
        sel = sum(sel_bit[i] << i for i in range(sel_bits))
        return sel

    def ip2node(self, n):
        """Input ddr/dma number and return the corresponding node number"""
        n = int(n)
        # ip2node_map = {0: 0, 4: 8, 8: 4, 12: 12, 16: 20, 20: 28, 24: 36, 28: 44, 32: 52, 36: 60, 40: 48, 44: 56, 48: 32, 52: 40, 56: 16, 60: 24}
        # ip2node_map = {0: 0, 4: 8, 8: 4, 12: 12, 16: 16, 20: 24, 24: 20, 28: 28}
        ip2node_map = {0: 0, 4: 8, 8: 4, 12: 12, 16: 16, 20: 24, 24: 20, 28: 28}
        remaining = n % 4
        n_1 = ip2node_map[n - remaining] + remaining
        ip2node_map_2 = {0: 0, 2: 4, 4: 2, 6: 6, 8: 8, 10: 12, 12: 10, 14: 14, 16: 16, 18: 20, 20: 18, 22: 22, 24: 24, 26: 28, 28: 26, 30: 30}  # 每 2 个一组映射
        remaining = n_1 % 2
        return str(ip2node_map_2[n_1 - remaining] + remaining)

    def process_file(self, file_path, input_folder, output_folder, file_name):
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # Process each line
        processed_lines = []
        dma = "gdma" if file_name.startswith("gmemTrace.TPU") else "cdma"
        for line in lines:
            data = line.strip().split(",")  # Assume data is comma-separated
            processed_data = [
                data[0],
                self.ip2node(re.findall(r"\d+", file_name)[0]),
                dma,
                self.ip2node(self.hash_all(addr=data[1])),
                "ddr",
                data[2],
                data[-1],
            ]
            processed_lines.append(",".join(processed_data))  # Convert back to comma-separated string

        self.request_end_time = max(self.request_end_time, int(lines[-1].strip().split(",")[0]))

        # Ensure output directory exists
        output_file_path = os.path.join(output_folder, os.path.relpath(file_path, start=input_folder))
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        # Write processed data to new file
        with open(output_file_path, "w", encoding="utf-8") as file:
            for line in processed_lines:
                file.write(line + "\n")

    def process_folder(self, input_folder, output_folder):
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                numbers = re.findall(r"\d+", file)
                assert len(numbers) == 1
                num = int(numbers[0])
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) == 0:
                    continue
                self.process_file(file_path, input_folder, output_folder, file)

            # Output statistics for the current directory
            if not files:
                continue
            total_request = self.shared_64_count + self.shared_8_count + self.private_count
            print(f"Directory: {root[27:]}")
            print(f"64 shared requests: {self.shared_64_count}, {self.shared_64_count / total_request}")
            print(f"8 shared requests: {self.shared_8_count}, {self.shared_8_count / total_request}")
            print(f"Private requests: {self.private_count}, {self.private_count / total_request}")
            print(f"Total requests: {self.shared_64_count + self.shared_8_count + self.private_count}")
            print(f"Request end time: {self.request_end_time}")
            self.shared_64_count = 0
            self.shared_8_count = 0
            self.private_count = 0

    def run(self, input_folder, output_folder):
        self.process_folder(input_folder, output_folder)
        print("Processing has been completed.")


# Main execution
if __name__ == "__main__":
    hasher = AddressHasher()
    input_folder = "../output-v7-32/step1_flatten"
    output_folder = "../output-v7-32/step2_hash_addr2node"
    hasher.run(input_folder, output_folder)
