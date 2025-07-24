import os
import shutil


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
        output_dir = os.path.join(subfolder_path, "output")

        if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
            print(f"  警告: 未找到output目录 - {output_dir}")
            continue

        # 创建对应的输出目录（以子文件夹名称命名）
        dest_subfolder = os.path.join(output_path, item)
        if not os.path.exists(dest_subfolder):
            os.makedirs(dest_subfolder)

        # 在output目录中查找符合条件的文件
        files_copied = 0
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                # 检查文件后缀
                if file.startswith("gmemTrace.CDMA") or file.endswith("tdma_instance.txt"):
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

        # print(f"  从 {item} 复制了 {files_copied} 个文件")


def main(input_path, output_path=None):
    """主函数"""
    if output_path is None:
        output_path = "../output"

    output_path = os.path.join(output_path, "step1_flatten")

    # print(f"输入路径: {input_path}")
    # print(f"输出路径: {output_path}")

    copy_files_with_suffix(input_path, output_path)
    flatten_directory(output_path)

    print("step1_flatten has been completed.")


if __name__ == "__main__":
    input_path = "Results_LlaMa2-70B-Patterns-v5"
    main(input_path, output_path="../output")
