import os
import sys
import time
import numpy as np
import pandas as pd

from src.cchessAI.core.net import PolicyValueNet

# 添加项目路径到 PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cchess import Board


def test_model_comparisons(model_pkl_path, model_onnx_path, model_trt_path):
    # 初始化三个模型
    print("Loading models...")
    pnet = PolicyValueNet(model_file=model_pkl_path, use_gpu=True)   # 移除了 model_type
    onet = PolicyValueNet(model_file=model_onnx_path, use_gpu=True)  # 移除了 model_type
    tnet = PolicyValueNet(model_file=model_trt_path, use_gpu=True)   # 移除了 model_type

    # 构造初始棋盘
    board = Board()
    print("\nInitial FEN:", board.fen())

    def run_inference(model, model_name):
        start_time = time.time()
        act_probs, value = model.policy_value_fn(board)
        end_time = time.time()

        # 转换为 DataFrame 并排序
        df = pd.DataFrame(act_probs, columns=["move_id", "prob"])
        df = df.sort_values(by="prob", ascending=False).head(10)

        print(f"\n[{model_name}] Inference Time: {(end_time - start_time)*1000:.2f} ms")
        print(f"Top 10 moves:")
        print(df)
        print(f"Value: {value.item():.4f}")  # 使用 .item() 将 numpy 标量转为 float

        return df['move_id'].tolist(), df['prob'].tolist(), value.item()  # 使用 .item() 转换为 Python float

    print("\n=== PyTorch (pkl) ===")
    pkl_moves, pkl_probs, pkl_val = run_inference(pnet, "PyTorch")

    print("\n=== ONNX ===")
    onnx_moves, onnx_probs, onnx_val = run_inference(onet, "ONNX")

    print("\n=== TensorRT ===")
    trt_moves, trt_probs, trt_val = run_inference(tnet, "TensorRT")

    # 对比输出
    print("\n=== Output Comparison ===")
    print(f"Top Move (PyTorch): {pkl_moves[0]} | Prob: {pkl_probs[0]:.4f} | Value: {pkl_val:.4f}")
    print(f"Top Move (ONNX):     {onnx_moves[0]} | Prob: {onnx_probs[0]:.4f} | Value: {onnx_val:.4f}")
    print(f"Top Move (TensorRT): {trt_moves[0]} | Prob: {trt_probs[0]:.4f} | Value: {trt_val:.4f}")

    # 差异分析
    print("\n=== Difference Analysis ===")
    prob_diff = abs(pkl_probs[0] - trt_probs[0])
    val_diff = abs(pkl_val - trt_val)
    print(f"Top Move Probability Diff: {prob_diff:.6f}")
    print(f"Value Diff: {val_diff:.6f}")

    # 相似度分析
    common_moves = set(pkl_moves[:5]).intersection(set(trt_moves[:5]))
    print(f"Common Top 5 Moves: {common_moves}")
    print(f"Overlap Count: {len(common_moves)} / 5")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare PyTorch, ONNX and TensorRT model outputs")
    parser.add_argument("--pkl", default="models/current_policy.pkl", type=str)
    parser.add_argument("--onnx", default="models/current_policy.onnx", type=str)
    parser.add_argument("--trt", default="models/current_policy.trt", type=str)

    args = parser.parse_args()

    test_model_comparisons(args.pkl, args.onnx, args.trt)
