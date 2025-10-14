# fruit_to_onnx_converter.py

import argparse
import os
from pathlib import Path

from src.cchessYolo.chess_detection_trainer import ChessPieceDetectorSeparate
# 当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 当前文件所在的目录
current_dir = os.path.dirname(current_file_path)
# 上一层目录
parent_dir = os.path.dirname(current_dir)

# 引入棋子检测器中的转换方法

def convert_fruit_model_to_onnx(pt_model_path, output_onnx_path=None, imgsz=640):
    """
    将水果检测模型(.pt)转换为ONNX格式

    Args:
        pt_model_path (str): .pt模型文件路径
        output_onnx_path (str): 输出ONNX文件路径，默认为None则自动生成
        imgsz (int): 输入图像尺寸，默认640
    """
    # 创建一个临时检测器实例用于转换
    detector = ChessPieceDetectorSeparate(model_path=pt_model_path)

    # 执行转换
    onnx_path = detector.convert_to_onnx(
        output_path=output_onnx_path,
        imgsz=imgsz,
        dynamic=False  # 根据需求调整是否使用动态尺寸
    )

    if onnx_path:
        print(f"✅ 水果模型已成功转换为ONNX格式: {onnx_path}")
    else:
        print("❌ 模型转换失败")

    return onnx_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_model', type=str, default=os.path.join(current_dir ,'runs/obb/fruit_obb_detection3/weights/best.pt'),
                        help='水果检测模型路径 (.pt)')
    parser.add_argument('--output_onnx', type=str, default=None,
                        help='输出ONNX模型路径')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸')

    args = parser.parse_args()

    convert_fruit_model_to_onnx(
        pt_model_path=args.pt_model,
        output_onnx_path=args.output_onnx,
        imgsz=args.imgsz
    )
