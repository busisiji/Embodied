import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
from torch import nn


import numpy as np
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils import checkpoint

from src.cchessAI import cchess
from src.cchessAI.parameters import IS_WINDOW
from utils.tools import move_action2move_id, decode_board

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda") if CUDA else torch.device("cpu")


# 构建残差块
class ResBlockCheck(nn.Module):
    def __init__(self, num_filters=256):
        super(ResBlockCheck, self).__init__()
        self.conv1 = nn.Conv2d(
            num_filters, num_filters, kernel_size=(3, 3), stride=(1, 1), padding=1
        )
        self.conv1_bn = nn.BatchNorm2d(
            num_filters,
        )
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(
            num_filters, num_filters, kernel_size=(3, 3), stride=(1, 1), padding=1
        )
        self.conv2_bn = nn.BatchNorm2d(
            num_filters,
        )
        self.conv2_act = nn.ReLU()

    def forward(self, x):
        return checkpoint.checkpoint(self._forward, x)

    def _forward(self, x):
        y = self.conv1(x)
        y = self.conv1_bn(y)
        y = self.conv1_act(y)
        y = self.conv2(y)
        y = self.conv2_bn(y)
        y = x + y
        return self.conv2_act(y)

class ResBlock(nn.Module):
    def __init__(self, num_filters=256):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            num_filters, num_filters, kernel_size=(3, 3), stride=(1, 1), padding=1
        )
        self.conv1_bn = nn.BatchNorm2d(
            num_filters,
        )
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(
            num_filters, num_filters, kernel_size=(3, 3), stride=(1, 1), padding=1
        )
        self.conv2_bn = nn.BatchNorm2d(
            num_filters,
        )
        self.conv2_act = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv1_bn(y)
        y = self.conv1_act(y)
        y = self.conv2(y)
        y = self.conv2_bn(y)
        y = x + y
        return self.conv2_act(y)

# 构建骨干网络, 输入: N, 15, 10, 9 --> N, C, H, W
class Net(nn.Module):
    def __init__(
        self, num_channels=256, num_res_blocks=40
    ):  # 40 ResBlock为AlphaZero的默认值
        super(Net, self).__init__()
        # 全局特征
        # self.global_conv = nn.Conv2D(in_channels=15, out_channels=512, kernel_size=(10, 9))
        # self.global_bn = nn.BatchNorm2D(512)
        # 初始化特征
        self.conv_block = nn.Conv2d(
            15, num_channels, kernel_size=(3, 3), stride=(1, 1), padding=1
        )
        self.conv_block_bn = nn.BatchNorm2d(
            num_channels,
        )
        self.conv_block_act = nn.ReLU()
        # 残差块抽取特征
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)]
        )
        # 策略头
        self.policy_conv = nn.Conv2d(
            num_channels, 16, kernel_size=(1, 1), stride=(1, 1)
        )
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_act = nn.ReLU()
        self.policy_fc = nn.Linear(16 * 10 * 9, 2086)
        # 价值头
        self.value_conv = nn.Conv2d(num_channels, 8, kernel_size=(1, 1), stride=(1, 1))
        self.value_bn = nn.BatchNorm2d(8)
        self.value_act1 = nn.ReLU()
        self.value_fc1 = nn.Linear(8 * 10 * 9, 256)
        self.value_act2 = nn.ReLU()
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # 公共头
        x = self.conv_block(x)
        x = self.conv_block_bn(x)
        x = self.conv_block_act(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        # 策略头
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_act(policy)
        policy = torch.reshape(policy, [-1, 16 * 10 * 9])
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        # 价值头
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.value_act1(value)
        value = torch.reshape(value, [-1, 8 * 10 * 9])
        value = self.value_fc1(value)
        value = self.value_act1(value)
        value = self.value_fc2(value)
        value = F.tanh(value)

        return policy, value


class PolicyValueNet(object):
    def __init__(self, model_file=None, use_gpu=True, device="cuda"):
        self.use_gpu = use_gpu
        self.l2_const = 2e-3  # l2 正则化
        self.device = device
        self.model_file = model_file
        self.model_type = 'pkl'
        ext = None

        # 根据设备类型设置正确的数据类型和混合精度配置
        if torch.cuda.is_available() and use_gpu:
            self.device = torch.device("cuda")
            self.stream = torch.cuda.Stream()
            if IS_WINDOW:
                self.scaler = torch.amp.GradScaler('cuda')  # Windows
            else:
                self.scaler = torch.amp.GradScaler('cuda')  # Linux也使用正确的设备
        else:
            self.device = torch.device("cpu")
            self.stream = None
            # CPU环境下不使用混合精度或使用CPU版本
            try:
                self.scaler = torch.amp.GradScaler('cpu')
            except:
                self.scaler = None  # 如果不支持则禁用

        # 自动判断模型类型
        if model_file:
            ext = os.path.splitext(model_file)[1].lower()

        try:
            import tensorrt as trt
            import pycuda.autoinit
            import pycuda.driver as cuda
            TENSORRT_AVAILABLE = True
        except ImportError:
            TENSORRT_AVAILABLE = False
            if ext == ".trt":
                print("TensorRT未安装，无法加载TRT模型，回退到PyTorch模型")
                ext = None  # 强制使用PyTorch模型

        if ext == ".onnx":
            import onnxruntime as ort
            # 启用 CUDA 执行提供者
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
            self.ort_session = ort.InferenceSession(model_file, providers=providers)
            self.onnx_input_name = self.ort_session.get_inputs()[0].name
            self.model_type = 'onnx'
            print(f"[INFO] ONNX 使用的执行提供者: {self.ort_session.get_providers()}")
        elif ext == ".trt" and TENSORRT_AVAILABLE:
            self.trt_logger = trt.Logger(trt.Logger.WARNING)
            with open(model_file, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.use_trt = True
            self.model_type = 'trt'
            self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        else:
            # PyTorch模型初始化
            self.model_type = 'pkl'
            self.policy_value_net = Net().to(self.device)
            self.optimizer = torch.optim.Adam(
                params=self.policy_value_net.parameters(),
                lr=1e-3,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.l2_const,
            )

            if ext and model_file and os.path.exists(model_file):
                try:
                    if self.use_gpu:
                        self.policy_value_net.load_state_dict(torch.load(model_file, weights_only=True))
                    else:
                        # 当CUDA不可用时，将模型映射到CPU
                        self.policy_value_net.load_state_dict(torch.load(model_file, map_location='cpu', weights_only=True))
                    print(f"[INFO] 成功加载模型: {model_file}")
                except Exception as e:
                    print(f"[WARN] 加载模型失败: {e}，使用随机初始化")



    # 输入一个批次的状态，输出一个批次的动作概率和状态价值
    def policy_value(self, state_batch):
        if self.model_type == 'onnx':  # 如果是ONNX模型
            state_batch = state_batch.astype(np.float32)
            outputs = self.ort_session.run(None, {self.onnx_input_name: state_batch})
            act_probs = np.exp(outputs[0])  # 转换logits为概率
            value = outputs[1]
            return act_probs, value
        else:
            # 原有的PyTorch推理保持不变
            self.policy_value_net.eval()
            state_batch = torch.tensor(state_batch).to(self.device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                log_act_probs, value = self.policy_value_net(state_batch)
            # 将结果移回 CPU 并转换为 numpy
            log_act_probs, value = log_act_probs.cpu().detach().numpy(), value.cpu().detach().numpy()
            act_probs = np.exp(log_act_probs)
            return act_probs, value




    # 输入棋盘，返回每个合法动作的（动作，概率）元组列表，以及棋盘状态的分数
    # act_probs：动作概率分布，用于指导 MCTS 的扩展；
    # value：当前局面的静态评估值，用于 playout 结束时的叶节点估值。 输出范围为 [-1, 1]，代表当前玩家的胜率估计
    def policy_value_fn(self, board):
        # 获取合法动作列表
        legal_positions = [
            move_action2move_id[cchess.Move.uci(move)]
            for move in list(board.legal_moves)
        ]
        current_state = decode_board(board)

        # 根据设备类型选择合适的数据类型
        if str(self.device) == 'cuda' and torch.cuda.is_available():
            current_state = np.ascontiguousarray(
                current_state.reshape(-1, 15, 10, 9)
            ).astype("float16")
        else:
            current_state = np.ascontiguousarray(
                current_state.reshape(-1, 15, 10, 9)
            ).astype("float32")

        if self.model_type == 'pkl':
            self.policy_value_net.eval()
            if self.stream is not None and str(self.device) == 'cuda':
                with torch.cuda.stream(self.stream):
                    current_state = torch.as_tensor(current_state).to(
                        self.device, non_blocking=True
                    )
                    # 根据设备类型决定是否使用混合精度
                    if str(self.device) == 'cuda' and torch.cuda.is_available():
                        with autocast('cuda'):  # GPU使用混合精度
                            log_act_probs, value = self.policy_value_net(current_state)
                    else:
                        # CPU直接推理
                        log_act_probs, value = self.policy_value_net(current_state)

                    log_act_probs, value = log_act_probs.to(
                        "cpu", non_blocking=True
                    ), value.to("cpu", non_blocking=True)
                torch.cuda.current_stream().wait_stream(self.stream)
            else:
                current_state = torch.as_tensor(current_state).to(self.device)

                # 根据设备类型决定是否使用混合精度
                if str(self.device) == 'cuda' and torch.cuda.is_available():
                    with autocast('cuda'):  # GPU使用混合精度
                        log_act_probs, value = self.policy_value_net(current_state)
                else:
                    # CPU直接推理，不使用混合精度
                    log_act_probs, value = self.policy_value_net(current_state)

                log_act_probs, value = log_act_probs.cpu(), value.cpu()

            # 修复BFloat16转换问题
            # 首先确保在CPU上并处理可能的BFloat16类型
            log_act_probs_cpu = log_act_probs.cpu()
            value_cpu = value.cpu()

            # 处理可能的BFloat16类型
            if log_act_probs_cpu.dtype == torch.bfloat16:
                log_act_probs_cpu = log_act_probs_cpu.float()

            if value_cpu.dtype == torch.bfloat16:
                value_cpu = value_cpu.float()

            # 安全地转换为numpy数组
            try:
                # 使用float32而不是float16以提高兼容性
                act_probs_np = log_act_probs_cpu.detach().numpy().astype(np.float32).flatten()
                act_probs = np.exp(act_probs_np)
                value_scalar = value_cpu.item()
            except Exception as e:
                print(f"数据类型转换错误: {e}")
                print(f"log_act_probs dtype: {log_act_probs_cpu.dtype}")
                print(f"value dtype: {value_cpu.dtype}")
                # 回退到安全的转换方法
                act_probs_np = log_act_probs_cpu.detach().float().numpy().flatten()
                act_probs = np.exp(act_probs_np)
                value_scalar = float(value_cpu.float())

            # 过滤合法动作
            act_probs = zip(legal_positions, act_probs[legal_positions])
            act_probs = list(act_probs)

            if not act_probs:
                # 如果没有合法动作，返回均匀分布
                act_probs = list(zip(legal_positions, [1.0/len(legal_positions)] * len(legal_positions)))

            # 归一化
            total_probs = sum(prob for _, prob in act_probs)
            if total_probs > 0:
                act_probs = [(action, prob/total_probs) for action, prob in act_probs]
            else:
                act_probs = [(action, 1.0/len(act_probs)) for action, _ in act_probs]

            return act_probs, value_scalar

        elif self.model_type == 'onnx':
            current_state = current_state.astype(np.float32)
            inputs = {self.onnx_input_name: current_state}
            outputs = self.ort_session.run(None, inputs)

            act_probs = np.exp(outputs[0].astype(np.float32).flatten())
            act_probs = zip(legal_positions, act_probs[legal_positions])
            value = outputs[1].flatten()[0]
            return act_probs, value
        elif self.model_type == 'trt':
            # 使用 TRT 推理
            policy, value = self.trt_predict(current_state)

            act_probs = np.exp(policy.astype(np.float32))
            act_probs = zip(legal_positions, act_probs[legal_positions])
            return act_probs, value



    def allocate_buffers(self):
        import pycuda.autoinit
        import pycuda.driver as cuda
        import tensorrt as trt

        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_type = self.engine.get_tensor_dtype(tensor_name)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            dtype = trt.nptype(tensor_type)

            size = trt.volume(tensor_shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            self.context.set_tensor_address(tensor_name, device_mem)

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append({
                    "name": tensor_name,
                    "host": host_mem,
                    "device": device_mem,
                    "shape": tensor_shape,
                    "dtype": dtype
                })
            else:
                outputs.append({
                    "name": tensor_name,
                    "host": host_mem,
                    "device": device_mem,
                    "shape": tensor_shape,
                    "dtype": dtype
                })

        return inputs, outputs, bindings, stream

    def trt_predict(self, current_state):
        import pycuda.driver as cuda

        # 设置输入数据
        np.copyto(self.inputs[0]['host'], current_state.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # 设置输入形状（动态维度）
        self.context.set_input_shape(self.inputs[0]['name'], current_state.shape)

        # 设置张量地址（每次推理前必须调用）
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], inp['device'])
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], out['device'])

        # 执行推理
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # 将输出从设备复制回主机
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        # 同步流
        self.stream.synchronize()

        # 提取输出
        policy = self.outputs[0]['host'].reshape(-1)
        value = self.outputs[1]['host'][0]

        return policy, value




    # 保存模型
    def save_model(self, model_file):
        torch.save(self.policy_value_net.state_dict(), model_file)

    # 执行一步训练
    def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
        state_batch = state_batch.astype(np.float16)

        self.policy_value_net.train()
        state_batch = torch.tensor(state_batch).to(self.device)
        mcts_probs = torch.tensor(mcts_probs).to(self.device)
        winner_batch = torch.tensor(winner_batch).to(self.device)

        self.optimizer.zero_grad()

        with autocast(str(DEVICE)):
            log_act_probs, value = self.policy_value_net(state_batch)
            value = torch.reshape(value, shape=[-1])
            value_loss = F.mse_loss(input=value, target=winner_batch)
            policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))
            loss = value_loss + policy_loss
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1)
            )
        return loss.item(), entropy.item()

    def export_to_onnx(self, onnx_model_path, dummy_input=None):
        self.policy_value_net.eval().to(torch.float32)  # 确保整个模型是 float32
        if dummy_input is None:
            # 创建一个虚拟输入
            dummy_input = torch.randn(1, 15, 10, 9).to(self.device)

        torch.onnx.export(
            self.policy_value_net,
            dummy_input,
            onnx_model_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['policy_output', 'value_output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'policy_output': {0: 'batch_size'},
                'value_output': {0: 'batch_size'}
            }
        )

        print(f"[INFO] ONNX 模型已导出至 {onnx_model_path}")
