import os


def core32_map(input_file_path, output_file_path):
    mesh_mapping = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        8: 8,
        9: 9,
        10: 10,
        11: 11,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        12: 12,
        13: 13,
        14: 14,
        15: 15,
        20: 16,
        21: 17,
        22: 18,
        23: 19,
        28: 24,
        29: 25,
        30: 26,
        31: 27,
        36: 20,
        37: 21,
        38: 22,
        39: 23,
        44: 28,
        45: 29,
        46: 30,
        47: 31,
    }
    # 定义映射关系
    cross_ring_mapping = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        8: 4,
        9: 5,
        10: 6,
        11: 7,
        4: 8,
        5: 9,
        6: 10,
        7: 11,
        12: 12,
        13: 13,
        14: 14,
        15: 15,
        20: 16,
        21: 17,
        22: 18,
        23: 19,
        28: 20,
        29: 21,
        30: 22,
        31: 23,
        36: 24,
        37: 25,
        38: 26,
        39: 27,
        44: 28,
        45: 29,
        46: 30,
        47: 31,
    }

    # 从输入文件中读取数据
    with open(input_file_path, "r") as file:
        lines = file.readlines()

    # 进行映射
    mapped_lines = []
    for line in lines:
        data = line.strip().split(",")
        if len(data) > 1 and data[1].isdigit() and data[3].isdigit():
            mapped_value_src = cross_ring_mapping.get(int(data[1]), None)
            mapped_value_dest = cross_ring_mapping.get(int(data[3]), None)
            if mapped_value_src and mapped_value_dest:
                data[1] = str(mapped_value_src)
                data[3] = str(mapped_value_dest)

        mapped_lines.append(",".join(data))

    # file_name = os.path.basename(input_file_path) + "_Trace.txt"
    # output_file = os.path.join(output_file_path, file_name)

    # 将映射后的数据写入到输出文件中
    with open(output_file_path, "w") as file:
        for line in mapped_lines:
            file.write(line + "\n")

    return mapped_lines


def main(directory_path, output_directory):
    # 遍历指定目录中的所有文件
    os.makedirs(output_directory, exist_ok=True)
    for filename in os.listdir(directory_path):
        input_file_path = os.path.join(directory_path, filename)

        # 只处理文件,跳过子目录
        if os.path.isfile(input_file_path):
            # 定义输出文件路径
            output_file_path = os.path.join(output_directory, filename)

            # 调用映射函数
            core32_map(input_file_path, output_file_path)
            print(f"Processed {input_file_path} -> {output_file_path}")


if __name__ == "__main__":
    directory_path = r"../traffic/output-v7-32/step5_data_merge"
    output_directory = r"../traffic/output-v7-32/step6_32core_map"
    main(directory_path, output_directory)
