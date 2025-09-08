# 训练模型
import os
import sys
import time
import pickle
import random
import numpy as np
import re
import torch
from concurrent.futures import ThreadPoolExecutor
from collections import deque

from src.cchessAI.core.conver.pkl_to_trt import convert_int64_to_int32, build_tensorrt_engine

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.data_utils import CompressionUtil, DataReader
from src.cchessAI import cchess
from src.cchessAI.core.game import Game
from src.cchessAI.core.mcts import MCTS_AI
from src.cchessAI.core.net import PolicyValueNet
from src.cchessAI.parameters import IS_UPNEW, BATCH_SIZE, EPOCHS, KL_TARG, CHECK_FREQ, GAME_BATCH_NUM, BUFFER_SIZE, \
    MODELS, DATA_BUFFER_PATH, PLAYOUT, C_PUCT
from utils.file_utils import get_latest_model

# 在设置CUDA内存限制前检查CUDA是否可用
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.9)  # 限制最大使用 90% 显存
else:
    print("CUDA不可用，使用CPU进行训练")
class TrainPipeline:
    def __init__(self, init_model=None):
        """初始化训练管道"""
        # 初始化训练参数
        self.init_model = init_model
        self.board = cchess.Board()
        self.game = Game(self.board)
        self.n_playout = PLAYOUT
        self.c_puct = C_PUCT
        self.learning_rate = 1e-3
        self.lr_multiplier = 1
        self.temp = 1.0
        self.batch_size = BATCH_SIZE
        self.epochs = EPOCHS
        self.kl_targ = KL_TARG
        self.check_freq = CHECK_FREQ
        self.game_batch_num = GAME_BATCH_NUM
        self.names = []
        self.train_num = 0
        self.best_win_ratio = 0.0
        self.buffer_size = BUFFER_SIZE
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.reader = None

        # 创建模型保存目录
        os.makedirs(MODELS, exist_ok=True)

        # 初始化策略价值网络
        if self.init_model:
            try:
                self.policy_value_net = PolicyValueNet(model_file=self.init_model)
                self.train_num = self.extract_batch_number(self.init_model)
                print(f"[{time.strftime('%H:%M:%S')}] 已加载上次最终模型 {self.train_num}批次训练")
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] 模型路径不存在，从零开始训练")
                self.policy_value_net = PolicyValueNet()
        else:
            if IS_UPNEW:
                self.init_model = self.resume_training_state()
            print(f"[{time.strftime('%H:%M:%S')}] 从零开始训练")
            self.policy_value_net = PolicyValueNet()

    def extract_batch_number(self, model_path):
        """从模型路径中提取 batch 轮数"""
        filename = os.path.basename(model_path)
        match = re.search(r"batch(\d+)_", filename)
        return int(match.group(1)) if match else 0

    def cleanup_models(self, n=1):
        """保留最新的n个模型，其余删除"""
        if len(self.names) <= n:
            return

        for model_path in self.names[:-n]:
            if os.path.exists(model_path):
                try:
                    os.remove(model_path)
                    print(f"[{time.strftime('%H:%M:%S')}] 已删除模型: {model_path}")
                except Exception as e:
                    print(f"[{time.strftime('%H:%M:%S')}] 删除失败: {model_path}, 错误: {e}")

    def policy_evaluate(self, num_games=5):
        """使用当前策略网络与纯MCTS玩家进行对局，计算胜率"""
        current_player = MCTS_AI(
            policy_value_fn=self.policy_value_net.policy_value_fn,
            n_playout=400
        )
        opponent_player = MCTS_AI(
            policy_value_fn=self.policy_value_net.policy_value_fn,
            n_playout=200
        )

        def play_game(game):
            winner, _ = game.start_play(
                current_player, opponent_player, is_shown=False
            )
            return 1 if winner == cchess.RED else 0

        win_count = 0
        with ThreadPoolExecutor(max_workers=num_games) as executor:
            futures = [
                executor.submit(play_game, Game(cchess.Board()))
                for _ in range(num_games)
            ]
            for future in futures:
                win_count += future.result()

        return win_count / num_games

    def policy_update(self, i=None):
        """更新策略价值网络"""
        if len(self.data_buffer) < self.batch_size:
            raise ValueError("数据缓冲区不足一个批次大小")

        mini_batch = random.sample(self.data_buffer, self.batch_size)
        if self.reader and self.reader.data_format in ['pkl']:
            mini_batch = [
                CompressionUtil.recovery_state_mcts_prob(data)
                for data in mini_batch
            ]

        # 准备训练数据
        state_batch = np.array([data[0] for data in mini_batch]).astype("float32")
        mcts_probs_batch = np.array([data[1] for data in mini_batch]).astype("float32")
        winner_batch = np.array([data[2] for data in mini_batch]).astype("float32")

        # 获取旧策略和价值
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)


        # 执行训练步骤
        for epoch in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learning_rate * self.lr_multiplier,
            )
            # 获取新策略和价值
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            # 计算KL散度
            kl = np.mean(np.sum(
                old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                axis=1
            ))

            # 如果KL散度很差，则提前终止
            if kl > self.kl_targ * 4:
                break

        # 自适应调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        # 计算解释方差
        def safe_explained_variance(y_true, y_pred):
            var_true = np.var(y_true)
            if var_true == 0 or np.isnan(var_true):
                return 0.0
            return 1 - np.var(y_true - y_pred) / var_true

        explained_var_old = safe_explained_variance(winner_batch, old_v.flatten())
        explained_var_new = safe_explained_variance(winner_batch, new_v.flatten())


        self.train_num += 1

        # 打印训练信息
        print((
            f"[{time.strftime('%H:%M:%S')}] kl:{kl:.5f},"
            f"第{self.train_num}轮"
            f"lr_multiplier:{self.lr_multiplier:.3f},"
            f"loss:{loss:.3f},"
            f"entropy:{entropy:.3f},"
            f"explained_var_old:{explained_var_old:.9f},"
            f"explained_var_new:{explained_var_new:.9f}"
        ))
        torch.cuda.empty_cache()  # 清理缓存
        return loss, entropy


    def handle_interruption(self):
        """处理中断事件，保存当前训练状态，并导出ONNX模型"""
        try:
            # 1. 保存主模型
            interrupted_model = os.path.join(
                MODELS,
                f"current_policy_batch{self.train_num}_{time.strftime('%Y%m%d%H%M')}.pkl"
            )
            self.policy_value_net.save_model(interrupted_model)
            self.names.append(interrupted_model)

            print(f"[INFO] 中断处理完成。模型已保存至：{interrupted_model}")

            # 2. 保存训练状态，包含最后保存的模型路径
            state = {
                'train_num': self.train_num,
                'lr_multiplier': self.lr_multiplier,
                'best_win_ratio': self.best_win_ratio,
                'data_buffer_size': len(self.data_buffer),
                'timestamp': time.time(),
                'last_model_path': interrupted_model
            }

            state_path = os.path.join(MODELS, "training_state.pkl")
            with open(state_path, 'wb') as f:
                pickle.dump(state, f)

            # 3. 导出 ONNX 模型
            onnx_path = self.export_model_to_onnx(interrupted_model)
            # 4.导出trt模型
            self.export_model_to_trt(onnx_path)


        except Exception as save_error:
            print(f"[ERROR] 中断处理失败: {str(save_error)}")


    def export_model_to_onnx(self, model_path):
        """
        将指定的 .pkl 模型导出为 ONNX 格式。

        参数：
            model_path: 模型路径（.pkl）
        """
        try:
            # 加载模型
            policy_value_net = PolicyValueNet(model_file=model_path, use_gpu=True)

            # 构建输出路径
            onnx_filename = os.path.basename(model_path).replace(".pkl", ".onnx")
            onnx_path = os.path.join(MODELS, "onnx", onnx_filename)
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

            # 导出 ONNX 模型
            policy_value_net.export_to_onnx(onnx_path)
            print(f"[{time.strftime('%H:%M:%S')}] 模型已成功导出至 {onnx_path}")
            return onnx_path
        except Exception as e:
            print(f"[ERROR] 导出 ONNX 模型失败: {e}")

    def export_model_to_trt(self,onnx_path):
        """
        将指定的 .onnx 模型导出为 trt 格式。

        参数：
            model_path: 模型路径（.pkl）
        """
        try:
            # 构建输出路径
            trt_filename = os.path.basename(onnx_path).replace(".onnx", ".trt")
            trt_path = os.path.join(MODELS, "trt", trt_filename)
            os.makedirs(os.path.dirname(trt_path), exist_ok=True)

            convert_int64_to_int32(onnx_path, onnx_path)

            # 导出 trt 模型
            build_tensorrt_engine(onnx_path,trt_path, max_batch_size = 1)
            print(f"[{time.strftime('%H:%M:%S')}] 模型已成功导出至 {trt_path}")
            return trt_path
        except Exception as e:
            print(f"[ERROR] 导出 ONNX 模型失败: {e}")

    def resume_training_state(self):
        """
        恢复训练状态，并返回中断时最后保存的模型路径。

        返回：
            str: 最后保存的模型路径，如果没有找到则返回 None
        """
        state_path = os.path.join(MODELS, "training_state.pkl")
        if os.path.exists(state_path):
            try:
                with open(state_path, 'rb') as f:
                    state = pickle.load(f)

                self.train_num = state.get('train_num', self.train_num)
                self.lr_multiplier = state.get('lr_multiplier', self.lr_multiplier)
                self.best_win_ratio = state.get('best_win_ratio', self.best_win_ratio)

                print(f"[INFO] 成功恢复训练状态，继续从第{self.train_num}轮开始")

                # 优先从状态文件中获取模型路径
                last_model_path = state.get('last_model_path')
                if last_model_path and os.path.exists(last_model_path):
                    print(f"[INFO] 上次保存的模型路径为：{last_model_path}")
                    return last_model_path

                # 如果状态文件中没有，则尝试从 names 中获取
                if self.names:
                    last_model_path = self.names[-1]
                    if os.path.exists(last_model_path):
                        print(f"[INFO] 通过 self.names 找到最新模型：{last_model_path}")
                        return last_model_path

                # 如果 self.names 也为空或无效，则查找目录中的最新模型
                latest_model = get_latest_model([".pkl"])
                if latest_model:
                    print(f"[INFO] 通过文件系统找到最新模型：{latest_model}")
                    return latest_model

                print("[WARNING] 未找到任何已保存的模型文件")
                return None

            except Exception as e:
                print(f"[ERROR] 恢复训练状态失败: {str(e)}")
                return None
        else:
            print(f"[INFO] 状态文件 {state_path} 不存在，尝试查找最新模型文件...")

            # 如果没有状态文件，直接查找最新模型
            latest_model = get_latest_model([".pkl"])
            if latest_model:
                print(f"[INFO] 找到最新模型：{latest_model}")
                return latest_model

            print("[WARNING] 未找到任何训练状态或模型文件")
            return None

    def run(self):
        """开始训练"""
        try:
            self.reader = DataReader(DATA_BUFFER_PATH)

            # 尝试加载数据
            retry_count = 0
            max_retries = 5
            nums = 0

            while retry_count < max_retries:
                try:
                    print(f"[{time.strftime('%H:%M:%S')}] 正在加载训练数据...")
                    all_data = []
                    for chunk, iters in self.reader.read_in_chunks(chunk_size=10000):
                        # 数据压缩处理
                        if self.reader.data_format == 'hdf5':
                            chunk = CompressionUtil.zip_states_mcts_prob(chunk)
                        all_data.extend(chunk)
                        del chunk

                        if iters > nums:
                            nums = iters
                        if nums >= self.buffer_size:
                            break

                    self.data_buffer = deque(all_data)
                    print(f"[{time.strftime('%H:%M:%S')}] 成功加载 {nums}局{len(self.data_buffer)} 条训练样本")
                    break

                except Exception as e:
                    retry_count += 1
                    print(f"[{time.strftime('%H:%M:%S')}] 数据加载失败: {e}，第 {retry_count}/{max_retries} 次重试")
                    time.sleep(10)

            if retry_count >= max_retries:
                raise RuntimeError("数据加载失败，请检查 DATA_BUFFER_PATH 路径或数据格式")

            # 开始训练循环
            print(f"[{time.strftime('%H:%M:%S')}] 开始训练流程")
            for i in range(self.game_batch_num):
                try:
                    loss, entropy = self.policy_update(i)
                    print(f"[{time.strftime('%H:%M:%S')}] 第 {i + 1} 轮训练完成，loss={loss:.3f}, entropy={entropy:.3f}")
                    if (i + 1) % 1000 == 0 or (i + 1) % self.check_freq == 0:
                        name = os.path.join(MODELS,
                                            f"current_policy_batch{self.train_num}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.pkl")
                        os.makedirs(os.path.dirname(name), exist_ok=True)
                        self.policy_value_net.save_model(name)
                        self.names.append(name)
                        self.cleanup_models()
                    torch.cuda.empty_cache()  # 已有：主循环中清理
                except KeyboardInterrupt:
                    print("\n[INFO] 检测到用户中断，正在保存当前模型...")
                    self.handle_interruption()
                    break
                except Exception as e:
                    print(f"[ERROR] 训练过程中发生异常: {str(e)}")
                    # self.handle_interruption()
                    raise

            print(f"[{time.strftime('%H:%M:%S')}] 训练已完成")
            self.handle_interruption()

        except Exception as e:
            print(f"[CRITICAL] 致命错误: {str(e)}")
            # 可以添加发送警报或日志记录


if __name__ == "__main__":
    training_pipeline = TrainPipeline(
        init_model=os.path.join(MODELS, "current_policy_batch7634_202507231110.pkl"))
    training_pipeline.run()
