import os

folder_path_8shared = "../output-v6-Ins_SG2262_LLaMa_MLP_8cluster_64core/step2_hash_addr2node"
folder_path_all = "../output-v6-Ins_SG2262_LLaMa_MLP_8cluster_64core/step1_flatten"


def data_count(input_folder):
    res = [0, 0]
    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
            max_line = len(lines)
            for i in range(max_line):
                data = lines[i].strip().split(",")
                t = int(data[0])
                if t > res[1]:
                    res[1] = t
                if data[-1] != 0:
                    res[0] += 1

    return res[0], res[1]


for subfolder in os.listdir(folder_path_8shared):
    path1 = os.path.join(folder_path_8shared, subfolder)
    path2 = os.path.join(folder_path_all, subfolder)
    shared_8, max_time1 = data_count(path1)
    all, max_time2 = data_count(path2)
    assert all % 2 == 0
    print(f"{subfolder[4:]}: max_time={max_time1, max_time2}\n" f"8-shared = {shared_8}, all = {all // 2}, 8-share/all = {2 * shared_8 / all:.3f}")
