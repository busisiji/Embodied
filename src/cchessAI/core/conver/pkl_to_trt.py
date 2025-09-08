import os
import sys
import argparse

import numpy as np
import onnx
from onnx import numpy_helper

from src.cchessAI.parameters import MODELS

# 添加项目根目录到 PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.cchessAI.core.net import PolicyValueNet


def pkl_to_onnx(pkl_model_path, onnx_model_path):
    """
    将 PyTorch .pkl 模型导出为 ONNX 模型
    """
    print(f"[INFO] Exporting PyTorch model to ONNX: {pkl_model_path} -> {onnx_model_path}")
    policy_value_net = PolicyValueNet(model_file=pkl_model_path, use_gpu=True)
    policy_value_net.policy_value_net.eval()
    policy_value_net.export_to_onnx(onnx_model_path)

def convert_int64_to_int32(onnx_model_path, output_model_path):
    model = onnx.load(onnx_model_path)

    for initializer in model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.INT64:
            print(f"Converting {initializer.name} from INT64 to INT32")
            arr = numpy_helper.to_array(initializer)
            arr = arr.astype(np.int32)
            new_initializer = numpy_helper.from_array(arr, initializer.name)
            initializer.CopyFrom(new_initializer)

    onnx.save(model, output_model_path)
    print(f"Model saved to {output_model_path}")
def build_tensorrt_engine(onnx_model_path, trt_engine_path, max_batch_size=1):
    """
    使用 convert_onnx_to_trt.py 中的函数构建 TensorRT 引擎
    """
    from src.cchessAI.core.conver.convert_onnx_to_trt import build_engine
    print(f"[INFO] Building TensorRT engine from ONNX: {onnx_model_path} -> {trt_engine_path}")
    build_engine(onnx_model_path, trt_engine_path, max_batch_size=max_batch_size)


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch (.pkl) model to TensorRT (.trt) via ONNX")
    parser.add_argument("--pkl", type=str,default=os.path.join(MODELS, "current_policy_batch7500_202507172325.pkl"),
                        help="Path to the input PyTorch .pkl model file")
    parser.add_argument("--onnx", type=str, default=os.path.join(MODELS,"onnx/current_policy_batch7500_202507172325.onnx"),
                        help="Path to save the intermediate ONNX model file (optional)")
    parser.add_argument("--trt", type=str, default=os.path.join(MODELS, "trt/current_policy_batch7500_202507172325.trt"),
                        help="Path to save the output TensorRT engine (.trt) file")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Max batch size for the TensorRT engine")

    args = parser.parse_args()

    # 自动命名中间 ONNX 文件（如果未指定）
    if args.onnx is None:
        onnx_filename = os.path.splitext(os.path.basename(args.pkl))[0] + ".onnx"
        args.onnx = os.path.join(os.path.dirname(args.pkl), onnx_filename)
        # 步骤 1: 导出 ONNX
    pkl_to_onnx(args.pkl, args.onnx)
    convert_int64_to_int32(args.onnx, args.onnx)

    # 步骤 2: 构建 TensorRT 引擎
    build_tensorrt_engine(args.onnx, args.trt, max_batch_size=args.batch_size)

    print(f"\n✅ Conversion completed:")
    print(f"ONNX Model saved to: {args.onnx}")
    print(f"TensorRT Engine saved to: {args.trt}")


if __name__ == "__main__":
    main()
