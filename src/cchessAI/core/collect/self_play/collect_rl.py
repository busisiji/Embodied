# src/cchessAI/collect/self_play/collect_rl.py
# 自我学习采集数据
import os
import sys

from utils.file_utils import add_data

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（cchess）
project_root = os.path.dirname((os.path.dirname(current_dir)))

# 如果不在 PYTHONPATH 中，则加入
if project_root not in sys.path:
    print(f"[INFO] 正在加入 {project_root}...")
    sys.path.insert(0, project_root)

import cchess
import time
import copy
import argparse
from collections import deque
from src.cchessAI.core.net import PolicyValueNet
from src.cchessAI.core.mcts import MCTS_AI
from src.cchessAI.core.game import Game
from src.cchessAI.parameters import BUFFER_SIZE
from utils.tools import (
    move_id2move_action,
    move_action2move_id,
    zip_state_mcts_prob,
    flip,
)


# 定义整个对弈收集数据流程
class CollectPipeline():
    def __init__(self, init_model=None,data_path=None, n_playout=1200,c_puct=5.0, temp=1.0):
        self.board = cchess.Board()
        self.game = Game(self.board)
        # 对弈参数

        self.init_model = init_model
        self.data_path = data_path
        self.temp = temp  # 温度
        self.n_playout = n_playout  # 每次移动的模拟次数
        self.c_puct = c_puct
        self.buffer_size = BUFFER_SIZE  # 经验池大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0
        self.mcts_ai = None
        self.policy_value_net = None  # 延迟初始化

    # 从主体加载模型
    def load_model(self):
        if self.policy_value_net is None:  # 仅初始化一次
            try:
                if self.init_model:
                    self.policy_value_net = PolicyValueNet(model_file=self.init_model)
                    print(f"[{time.strftime('%H:%M:%S')}] 已加载最新模型")
                else:
                    self.policy_value_net = PolicyValueNet()
                    print(f"[{time.strftime('%H:%M:%S')}] 已加载初始模型")
            except:
                self.policy_value_net = PolicyValueNet()
                print(f"[{time.strftime('%H:%M:%S')}] 已加载初始模型")
            self.mcts_ai = MCTS_AI(
                self.policy_value_net.policy_value_fn,
                c_puct=self.c_puct,
                n_playout=self.n_playout,
                is_selfplay=True,
            )
        else:
            print(f"[{time.strftime('%H:%M:%S')}] 模型已是最新，无需重新加载")

    def mirror_data(self, play_data):
        """左右对称变换，扩充数据集一倍，加速一倍训练速度"""
        mirror_data = []
        # 棋盘形状 [15, 10, 9], 走子概率，赢家
        for state, mcts_prob, winner in play_data:
            # 原始数据
            mirror_data.append(zip_state_mcts_prob((state, mcts_prob, winner)))
            # 水平翻转后的数据
            state_flip = state.transpose([1, 2, 0])[:, ::-1, :].transpose([2, 0, 1])
            mcts_prob_flip = copy.deepcopy(mcts_prob)
            for i in range(len(mcts_prob_flip)):
                # 水平翻转后，走子概率也需要翻转
                mcts_prob_flip[i] = mcts_prob[
                    move_action2move_id[flip(move_id2move_action[i])]
                ]
            mirror_data.append(
                zip_state_mcts_prob((state_flip, mcts_prob_flip, winner))
            )
        return mirror_data

    def collect_data(self, n_games=1, is_shown=False):
        """收集自我对弈的数据"""
        for i in range(n_games):
            self.load_model()  # 从本体处加载最新模型
            winner, play_data = self.game.start_self_play(
                self.mcts_ai, is_shown=is_shown, temp=self.temp
            )  # 开始自我对弈，传递温度参数
            play_data = list(play_data)  # 转换为列表
            self.episode_len = len(play_data)  # 记录每盘对局长度

            # 增加数据
            play_data = self.mirror_data(play_data)
            self.data_buffer.extend(play_data)
            self.iters += 1

            add_data(self.data_path,self.data_buffer, self.iters)

        return self.iters

    def run(self, n_games=1, is_shown=False):
        """开始收集数据"""
        try:
            total_games = 0
            while total_games < n_games:
                batch_size = min(n_games - total_games, 10)  # 每批最多10局
                iters = self.collect_data(n_games=batch_size, is_shown=is_shown)
                total_games += batch_size
                print(
                    f"[{time.strftime('%H:%M:%S')}] 已完成 {total_games}/{n_games} 局, batch i: {iters}, episode_len: {self.episode_len}"
                )
        except KeyboardInterrupt:
            print(f"\n\r[{time.strftime('%H:%M:%S')}] quit")


if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="收集中国象棋自对弈数据")
    parser.add_argument(
        "--show", action="store_true", default=False, help="是否显示棋盘对弈过程"
    )
    parser.add_argument("--init-model", type=str, default=None,
                        help="初始模型路径（可以是.pkl或.onnx或trt格式）")
    parser.add_argument("--data-path", type=str, default=None,
                        help="初始数据文件路径")
    parser.add_argument(
        "--game-count", type=int, default=1, help="对弈局数"
    )
    parser.add_argument(
        "--temp", type=float, default=1.0, help="温度参数"
    )
    parser.add_argument(
        "--cpuct", type=float, default=5.0, help="UCT探索参数"
    )
    parser.add_argument(
        "--nplayout", type=int, default=1200, help="每次移动的模拟次数"
    )

    args = parser.parse_args()    # 处理 init_model 参数，将字符串 "None" 转换为真正的 None
    if args.init_model == "None":
        args.init_model = None

    # 处理 data_path 参数，将字符串 "None" 转换为真正的 None
    if args.data_path == "None":
        args.data_path = None


    # 创建数据收集管道实例
    collecting_pipeline = CollectPipeline(init_model=args.init_model,data_path=args.data_path, n_playout=args.nplayout,c_puct=args.cpuct, temp=args.temp)
    collecting_pipeline.run(n_games=args.game_count, is_shown=args.show)
