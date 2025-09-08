import datetime
import os
import pickle
from datetime import time
from pathlib import Path

import h5py
import numpy as np

from src.cchessAI.parameters import DATA_PATH, MODELS, MODEL_PATH, DATA_BUFFER_PATH, DATA_SELFPLAY, MODEL_USER_PATH, \
    DATA_USER_PATH


def merge_pkl_files(input_paths, output_path):
    """
    合并多个 .pkl 文件为一个文件。

    Args:
        input_paths (list): 多个 .pkl 文件路径列表
        output_path (str): 合并后的输出文件路径
    """
    merged_data = {"data_buffer": [], "iters": 0}
    total_iters = 0

    for path in input_paths:
        if not os.path.exists(path):
            print(f"[警告] 文件不存在: {path}")
            continue
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                merged_data["data_buffer"].extend(data.get("data_buffer", []))
                merged_data["iters"] += data.get("iters", 0)
                total_iters += data.get("iters", 0)
            print(f"[INFO] 已读取: {path}")
        except Exception as e:
            print(f"[错误] 加载文件失败 {path}: {e}")
    print(f"[INFO] 正在合并 {len(merged_data['data_buffer'])} 条数据，局数{total_iters}")
    with open(output_path, 'wb') as f:
        pickle.dump(merged_data, f)
    print(f"[INFO] 合并完成，共处理 {total_iters} 局数据，已保存至: {output_path}")

def pkl_to_hdf5(pkl_path, h5_path):
    """
    将 .pkl 文件转换为 .h5 文件格式。

    Args:
        pkl_path (str): 输入的 .pkl 文件路径
        h5_path (str): 输出的 .h5 文件路径
    """
    if not os.path.exists(pkl_path):
        print(f"[错误] 文件不存在: {pkl_path}")
        return

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        play_data = data.get("data_buffer", [])
        states, mcts_probs, winners = zip(*[
            (s, mp, w) for s, mp, w in play_data
        ])

        states = np.array(states, dtype=np.float32)
        mcts_probs = np.array(mcts_probs, dtype=np.float32)
        winners = np.array(winners, dtype=np.float64)

        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset('states', data=states, compression='gzip')
            hf.create_dataset('mcts_probs', data=mcts_probs, compression='gzip')
            hf.create_dataset('winners', data=winners, compression='gzip')

        print(f"[INFO] 成功将 {pkl_path} 转换为 {h5_path}")
    except Exception as e:
        print(f"[错误] 转换失败: {e}")

def hdf5_to_pkl(h5_path, pkl_path):
    """
    将 .h5 文件转换为 .pkl 文件格式。

    Args:
        h5_path (str): 输入的 .h5 文件路径
        pkl_path (str): 输出的 .pkl 文件路径
    """
    if not os.path.exists(h5_path):
        print(f"[错误] 文件不存在: {h5_path}")
        return

    try:
        with h5py.File(h5_path, 'r') as hf:
            states = hf['states'][:]
            mcts_probs = hf['mcts_probs'][:]
            winners = hf['winners'][:]

        play_data = list(zip(states, mcts_probs, winners))
        data = {
            "data_buffer": play_data,
            "iters": len(states)
        }

        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"[INFO] 成功将 {h5_path} 转换为 {pkl_path}")
    except Exception as e:
        print(f"[错误] 转换失败: {e}")
def get_latest_model_state():
    """获取最新的模型路径"""
    try:
        # 优先使用 training_state.pkl 中的模型信息
        state_path = os.path.join(MODELS, "training_state.pkl")
        if os.path.exists(state_path):
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
                train_num = state.get('train_num', 0)

            # 查找匹配 batch 的模型文件
            for model_file in os.listdir(MODELS):
                if f"batch{train_num}" in model_file and model_file.endswith(".pkl"):
                    return os.path.join(MODELS, model_file)

        # 如果没有找到匹配的模型，尝试查找最新的 ONNX 模型
        onnx_dir = os.path.join(MODELS, "onnx")
        if os.path.exists(onnx_dir):
            model_files = [f for f in os.listdir(onnx_dir) if f.endswith(".onnx")]
            if model_files:
                latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(onnx_dir, x)))
                return os.path.join(onnx_dir, latest_model)

        # 最后尝试查找所有 .pkl 模型
        model_files = [f for f in os.listdir(MODELS) if f.endswith(".pkl")]
        if model_files:
            latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(MODELS, x)))
            return os.path.join(MODELS, latest_model)

    except Exception as e:
        print(f"[ERROR] 获取最新模型失败: {e}")
    return None
def get_latest_model(usefiles=[".pkl", ".onnx",'trt']):
    """获取最新的模型路径"""
    try:
        model_dir = MODELS
        model_files = []

        # 递归查找所有 .pkl 和 .onnx 模型，包括子目录
        for root, dirs, files in os.walk(model_dir):
            for ext in usefiles:
                model_files.extend([os.path.join(root, file) for file in files if file.endswith(ext) and 'current_policy' in file])

        if not model_files:
            return MODEL_PATH  # 没有模型则使用默认路径

        # 按修改时间排序取最新模型
        latest_model = max(
            model_files,
            key=lambda x: os.path.getmtime(x)
        )
        return latest_model

    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] 获取最新模型失败: {e}")
        return MODEL_PATH
def get_latest_data():
    data_dir = Path(DATA_SELFPLAY)
    data_files = sorted(
        [f for f in os.listdir(data_dir) if f.startswith("data_") and f.endswith(".pkl")],
        key=lambda x: os.path.getmtime(os.path.join(data_dir, x)),
        reverse=True
    )
    if data_files:
        return os.path.join(data_dir, data_files[0])
    else:
        return None

def add_data(latest_file,data_buffer,iters):
    merged_data = []
    total_iters = iters
    merged_data.extend(data_buffer)
    if latest_file and os.path.exists(latest_file):
        try:
            with open(latest_file, "rb") as f:
                data = pickle.load(f)
                loaded_data = data.get("data_buffer", [])
                merged_data.extend(loaded_data)
                total_iters += data.get("iters", 0)
                print(
                    f"[{time.strftime('%H:%M:%S')}] 已加载历史数据文件: {latest_file}, 共 {len(loaded_data)} 条样本, 局数{total_iters}")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 加载历史数据文件失败：{e}")

    # 保存最终合并后的数据
    with open(latest_file, "wb") as f:
        pickle.dump({
            "data_buffer": merged_data,
            "iters": total_iters
        }, f)
    return latest_file
def add_latest_data(data_buffer, iters):
    # 加载已有数据（如果存在）
    data_dir = Path(DATA_SELFPLAY)
    data_files = sorted(
        [f for f in os.listdir(data_dir) if f.startswith("data_") and f.endswith(".pkl")],
        key=lambda x: os.path.getmtime(os.path.join(data_dir, x)),
        reverse=True
    )
    merged_data = []
    total_iters = iters
    merged_data.extend(data_buffer)
    if data_files:
        latest_file = os.path.join(data_dir, data_files[0])
        try:
            return add_data(latest_file,merged_data,total_iters)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 加载历史数据文件失败：{e}")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] 未找到任何历史数据文件")

def is_file_exists(init_model):
    init_model_dir = os.path.dirname(init_model)
    if init_model_dir and not os.path.exists(init_model_dir):
        os.makedirs(init_model_dir, exist_ok=True)

def init_user_file(user_id: str):
    user_path = os.path.join(MODEL_USER_PATH, "user_" + user_id)
    if not os.path.exists(user_path):
        os.makedirs(user_path, exist_ok=True)
    if not os.path.exists(os.path.join(user_path, 'pkl')):
        os.makedirs(os.path.join(user_path, 'pkl'), exist_ok=True)
    if not os.path.exists(os.path.join(user_path, 'onnx')):
        os.makedirs(os.path.join(user_path, 'onnx'), exist_ok=True)
    if not os.path.exists(os.path.join(user_path, 'trt')):
        os.makedirs(os.path.join(user_path, 'trt'), exist_ok=True)
    user_path = os.path.join(DATA_USER_PATH, "user_" + user_id)
    if not os.path.exists(user_path):
        os.makedirs(user_path, exist_ok=True)
    collect_path = os.path.join(user_path, "collect")
    if not os.path.exists(collect_path):
        os.makedirs(collect_path, exist_ok=True)


if __name__ == "__main__":
    # 示例用法
    input_paths = [os.path.join(DATA_PATH, "_m_/data_buffer.pkl"),os.path.join(DATA_PATH, "t/data_buffer.pkl")]
    output_path = os.path.join(DATA_PATH, "data_buffer.pkl")
    merge_pkl_files(input_paths, output_path)