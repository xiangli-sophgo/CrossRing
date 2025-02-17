import os


def read_txt_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    return [line.strip() for line in lines]


def write_txt_file(file_path, data):
    with open(file_path, "w") as file:
        for line in data:
            file.write(line + "\n")


def merge_and_sort_files(folder_path, output_path=None):
    all_data = []

    # Traverse the directory
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                file_data = read_txt_file(file_path)
                all_data.extend(file_data)

    # Sort data based on the first column (integer values)
    all_data.sort(key=lambda x: int(x.split(",")[0]))

    # Write sorted data to a new file
    file_name = os.path.basename(folder_path) + "_Trace.txt"
    output_path = output_path if output_path else "../output"
    output_file = os.path.join(output_path + "/step5_data_merge", file_name)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    write_txt_file(output_file, all_data)


# Example usage
def main(folder_path=None, output_path=None):
    # folder_path = "../output/step2_hash_addr2node"
    folder_path = "../output-v7-32/step2_hash_addr2node" if not folder_path else folder_path
    for subfolder in os.listdir(folder_path):
        subfolder = os.path.join(folder_path, subfolder)
        merge_and_sort_files(subfolder, output_path)

    print("step5_data_merge has been completed.")


if __name__ == "__main__":
    main()
