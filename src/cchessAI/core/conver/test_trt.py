import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# 初始化 TensorRT 日志和运行时
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open("models/trt/model.trt", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# 创建执行上下文
context = engine.create_execution_context()

# 获取绑定信息
inputs, outputs, bindings, stream = [], [], [], cuda.Stream()

for i in range(engine.num_bindings):
    binding_name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(binding_name)
    is_input = mode == trt.TensorIOMode.INPUT

    print(f"Binding {i}: {binding_name}, Is Input: {is_input}")

    # ✅ 使用 get_tensor_shape 来判断是否为动态模型
    dims = context.get_tensor_shape(binding_name)

    if any(dim == -1 or dim == 0 for dim in dims):
        print(f"检测到动态维度: {binding_name} -> {dims}")
        # 动态模型需手动设置输入维度
        context.set_input_shape(binding_name, (1, 15, 10, 9))  # batchhan, cnels, height, width
        # 更新 dims
        dims = context.get_tensor_shape(binding_name)

    dtype = trt.nptype(engine.get_tensor_dtype(binding_name))
    size = trt.volume(dims)
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)

    context.set_tensor_address(binding_name, device_mem)
    bindings.append(int(device_mem))

    if is_input:
        inputs.append({'name': binding_name, 'host': host_mem, 'device': device_mem, 'dims': dims})
    else:
        outputs.append({'name': binding_name, 'host': host_mem, 'device': device_mem, 'dims': dims})

# 构造输入数据
input_data = np.random.rand(1, 15, 10, 9).astype(np.float16)

# 执行推理
np.copyto(inputs[0]['host'], input_data.ravel())
cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)

# 异步执行推理
context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

# 将输出从设备复制回主机
for out in outputs:
    cuda.memcpy_dtoh_async(out['host'], out['device'], stream)

stream.synchronize()

# 输出结果
print("\n推理完成，输出结果如下：")
for out in outputs:
    output_array = out['host'].reshape(out['dims'])  # ✅ 使用自己的 dims
    print(f"{out['name']}: {output_array}")
