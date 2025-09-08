import sys
import os
import tensorrt as trt

# 添加项目路径到 PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def build_engine(onnx_file_path, engine_file_path, max_batch_size=1):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        # 解析 ONNX 模型
        with open(onnx_file_path, "rb") as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # 配置构建器
        config = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        config.set_shape(input_tensor.name, (1, 15, 10, 9), (1, 15, 10, 9), (max_batch_size, 15, 10, 9))

        # 构建配置（启用FP16）
        builder_config = builder.create_builder_config()
        builder_config.add_optimization_profile(config)
        builder_config.set_flag(trt.BuilderFlag.FP16)
        builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        # 构建引擎（新版 API）
        engine = builder.build_serialized_network(network, builder_config)

        # 保存引擎文件
        if engine:
            if not os.path.exists(os.path.dirname(engine_file_path)):
                os.makedirs(os.path.dirname(engine_file_path))
            with open(engine_file_path, "wb") as f:
                f.write(engine)
            print(f"[INFO] TensorRT Engine saved to {engine_file_path}")
            return engine
        else:
            print("[ERROR] Failed to build TensorRT engine.")
            return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert ONNX model to TensorRT engine")
    parser.add_argument("--onnx", default='models/current_policy.onnx', type=str, help="Path to the ONNX model file")
    parser.add_argument("--trt", default="models/current_policy.trt", type=str, help="Path to save the TensorRT engine file")
    parser.add_argument("--batch-size", type=int, default=1, help="Max batch size for the engine")

    args = parser.parse_args()

    build_engine(args.onnx, args.trt, max_batch_size=args.batch_size)
