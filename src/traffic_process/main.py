import step1_flatten
import step2_hash_addr2node

# import step3_data_reduced
# import step4_add_4N
import step5_data_merge
import step6_core32_map

# import AddPacketId


# path为输入Trace的文件夹名称,path和代码在同一路径

path = "../traffic/data/V8-32/2M"
# path += "ins_SG2262_Ring_all_reduce_8cluster_all2all"
output_path = "../traffic/output-v8-32/2M"
# # outstanding_num必须为2的幂
outstanding_num = 512
assert isinstance(outstanding_num, int), "outstanding_num must be integer or out of range."
assert outstanding_num > 0, "outstanding_num must be positive integer."
assert outstanding_num & outstanding_num - 1 == 0, "outstanding_num must be a power of 2."
#
# # 根据outstanding_num值,做hash需要多少位运算
outstanding_digit = outstanding_num.bit_length() - 1
# print(outstanding_digit)

# 1.压平Trace文件夹,只保留NoC所需要的文件,并简化目录结构,输出文件夹为step1_flatten
# step1_flatten.main(path, output_path)


# 2.将Trace中的地址转化为节点编号
hasher = step2_hash_addr2node.AddressHasher()
hasher.run(output_path + "/step1_flatten", output_path + "/step2_hash_addr2node")


# step3_data_reduced.main()


# step4_add_4N.main()


step5_data_merge.main(output_path + "/step2_hash_addr2node", output_path)

step6_core32_map.main(output_path + "/step5_data_merge", output_path + "/step6_32core_map")


# AddPacketId.main()
