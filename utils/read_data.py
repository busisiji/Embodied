import pickle
import os


# 假设 DATA_BUFFER_PATH 已定义为你的数据文件路径
def read_data_from_pkl():
    start_index = 0
    end_index = 5
    path = "../data/data_buffer.pkl"
    if os.path.exists(path):
        with open(path, "rb") as data_file:
            data_dict = pickle.load(data_file)
            data_buffer = data_dict["data_buffer"]  # 获取经验回放缓冲区数据
            iters = data_dict["iters"]  # 获取迭代次数

        print(f"迭代次数: {iters}")
        print(f"数据缓冲区大小: {len(data_buffer)} 条")
        print(f"数据缓冲区内容 ({start_index}-{end_index}条):")
        for i, entry in enumerate(data_buffer):
            if i < start_index:
                continue
            if i >= end_index:
                break
            print(entry)  # 输出每条数据的前5项以查看样本
        print("...")
    else:
        print("数据文件不存在，请先运行数据收集程序生成数据。")

read_data_from_pkl()