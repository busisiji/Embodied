import onnx
import onnxruntime as ort
import numpy as np

# 加载模型
onnx_model = onnx.load("../models/current_policy.onnx")
onnx.checker.check_model(onnx_model)

# 创建输入
dummy_input = np.random.rand(1, 15, 10, 9).astype(np.float32)

# 推理测试
ort_session = ort.InferenceSession("../models/current_policy.onnx")
outputs = ort_session.run(None, {"input": dummy_input})
print("✅ ONNX 推理成功:", outputs[0].shape, outputs[1].shape)
