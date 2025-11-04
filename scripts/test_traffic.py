# 快速生成一个读请求测试文件
with open("CrossRing/traffic/d2d_test_read.csv", "w") as f:
    f.write("traffic_id,source_die,source_node,target_die,target_node,req_type,burst_length,start_time_ns,interval_ns,count\n")
    f.write("1,0,0,1,3,read,4,1,0,1\n")
print("生成读请求测试文件")
