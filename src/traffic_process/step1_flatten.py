import os
import shutil
import re


def flatten_directory(directory):
    """展平目录结构，消除只有一个子目录的中间层"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"目录不存在,已自动创建: {directory}")
    elif not os.path.isdir(directory):
        raise NotADirectoryError(f"路径不是目录: {directory}")

    for dir_name in os.listdir(directory):
        path = os.path.join(directory, dir_name)
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path, topdown=False):
                if len(dirs) == 1 and not files:
                    single_dir = dirs[0]
                    single_dir_path = os.path.join(root, single_dir)

                    # 移动内容
                    for item in os.listdir(single_dir_path):
                        src = os.path.join(single_dir_path, item)
                        dst = os.path.join(root, item)
                        shutil.move(src, dst)

                    # 删除空目录
                    os.rmdir(single_dir_path)


def copy_files_with_suffix(input_path, output_path):
    """从输入路径复制指定后缀的文件到输出路径，跳过output子目录"""
    if not os.path.exists(input_path):
        print("输入路径不存在")
        return

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 遍历输入路径的第一层子目录
    for item in os.listdir(input_path):
        subfolder_path = os.path.join(input_path, item)

        # 只处理目录
        if not os.path.isdir(subfolder_path):
            continue

        print(f"处理子文件夹: {item}")

        # 查找该子文件夹下的output目录
        output_dir = subfolder_path

        # 创建对应的输出目录（以子文件夹名称命名）
        dest_subfolder = os.path.join(output_path, item)
        if not os.path.exists(dest_subfolder):
            os.makedirs(dest_subfolder)

        # 在output目录中查找符合条件的文件
        files_copied = 0
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                # 检查文件后缀
                if file.startswith("gmemTrace") or file.endswith("tdma_instance.csv"):
                    # 构造完整的文件路径
                    full_file_path = os.path.join(root, file)

                    # 计算相对于output目录的路径
                    relative_path = os.path.relpath(root, output_dir)

                    # 构建目标目录
                    if relative_path == ".":
                        # 文件直接在output目录下
                        dest_dir = dest_subfolder
                    else:
                        # 文件在output的子目录中
                        dest_dir = os.path.join(dest_subfolder, relative_path)

                    if not os.path.exists(dest_dir):
                        os.makedirs(dest_dir)

                    # 构建目标文件路径
                    dest_file_path = os.path.join(dest_dir, file)

                    # 复制文件
                    shutil.copy2(full_file_path, dest_file_path)
                    # print(f"  复制文件：{file} -> {dest_file_path}")
                    files_copied += 1


def merge_and_sort_files_by_folder(input_folder, output_folder):
    """
    将每个子文件夹中的所有CSV文件合并成一个文件，文件名为文件夹名称
    提取格式：时间、源节点、源IP、目标地址、目标IP、请求类型、burst
    """
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历第一层子文件夹
    for item in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, item)

        # 只处理目录
        if not os.path.isdir(subfolder_path):
            continue

        print(f"处理文件夹: {item}")
        all_data = []

        # 遍历该文件夹中的所有文件
        for root, _, files in os.walk(subfolder_path):
            for file in files:
                # 只处理匹配的文件
                if not (file.startswith("gmemTrace") or file.endswith("tdma_instance.csv")):
                    continue

                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) == 0:
                    print(f"  跳过空文件: {file}")
                    continue

                # 从文件名提取TPU编号
                tpu_match = re.search(r"tpu_(\d+)", file)
                if not tpu_match:
                    print(f"  警告: 无法从文件名中提取TPU编号: {file}")
                    continue
                tpu_num = tpu_match.group(1)

                print(f"  处理文件: {file}")

                # 读取文件内容
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                # 跳过标题行
                if lines and lines[0].startswith("begin_req"):
                    lines = lines[1:]

                # 处理每一行数据
                for line in lines:
                    data = line.strip().split(",")
                    if len(data) < 8:
                        continue  # 跳过无效行

                    # 提取所需字段：时间、源节点、源IP、目标地址、目标IP、请求类型、burst
                    processed_data = [
                        data[1],  # 时间 (end_req)
                        tpu_num,  # 源节点 (TPU编号)
                        "gdma",  # 源IP类型 (固定为gdma)
                        data[4],  # 目标地址 (addr，保留原始地址)
                        "ddr",  # 目标IP类型 (固定为ddr)
                        data[5],  # 请求类型 (r/w)
                        data[7],  # burst长度 (burst_len)
                    ]
                    all_data.append(processed_data)

        if all_data:
            # 按第一列（时间）进行数值排序
            all_data.sort(key=lambda x: int(x[0]))

            # 输出文件名为文件夹名称
            output_file = os.path.join(output_folder, f"{item}.txt")

            # 写入合并后的数据
            with open(output_file, "w", encoding="utf-8") as f:
                for data in all_data:
                    f.write(",".join(data) + "\n")

            print(f"  合并完成，共处理 {len(all_data)} 条记录，输出文件: {output_file}")
        else:
            print(f"  文件夹 {item} 中没有找到有效数据")


def main(input_path, output_path=None):
    """主函数"""
    if output_path is None:
        output_path = "../output"

    output_path = os.path.join(output_path, "step1_flatten")

    # 第一步：复制文件并展平目录
    copy_files_with_suffix(input_path, output_path)
    flatten_directory(output_path)

    # 第二步：将每个文件夹中的文件合并成一个文件，文件名为文件夹名称
    merged_output_folder = os.path.join(output_path, "../merged")
    merge_and_sort_files_by_folder(output_path, merged_output_folder)

    print("step1_flatten has been completed.")


if __name__ == "__main__":
    input_path = r"../../traffic/DeepSeek_0918/step1_flatten/"
    main(input_path, output_path="../output")
