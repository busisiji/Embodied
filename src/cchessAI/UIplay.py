# UIplay.py - 优化版人机对弈
# 获取当前文件所在目录
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（cchess）
project_root = os.path.dirname(current_dir)

# 如果不在 PYTHONPATH 中，则加入
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.game import Game
import cchess
from core.mcts import MCTS_AI
from utils.tools import move_id2move_action, move_action2move_id
from core.net import PolicyValueNet


class Human:
    """人类玩家类"""
    agent = "HUMAN"
    def get_action(self, board):
        try:
            user_input = input('请输入走法 (例如: e6e9): ')
            uci_move = move_action2move_id.get(user_input)
            if uci_move is None:
                print("无效的走法，请重新输入。")
                return self.get_action(board)
            uci_str = move_id2move_action[uci_move]
            move = cchess.Move.from_uci(uci_str)

            # 检查是否是合法走法
            if move not in board.legal_moves:
                print(f"非法走法: {user_input}，请重试！")
                return self.get_action(board)

            return uci_move

        except Exception as e:
            print(f"输入错误，请重试。错误: {e}")
            return self.get_action(board)


    def set_player_idx(self, idx):
        """设置玩家标识（红方/黑方）"""
        self.player_idx = idx


def run():
    # 加载模型
    policy_value_net = PolicyValueNet(model_file='models/admin/trt/current_policy_batch7483_202507170806.trt', use_gpu=True)

    # 创建 MCTS 玩家
    mcts_player = MCTS_AI(policy_value_net.policy_value_fn,c_puct=3, n_playout=1200)
    # 创建人类玩家
    human = Human()

    # 初始化棋盘和游戏
    board = cchess.Board()
    game = Game(board)

    # # 开始人机对战（人类先手）
    # game.start_play(player1=human, player0=mcts_player, is_shown=True)
    # 开始人机对战（机器先手）
    game.start_play_with_mixed_strategy(player0=human, player1=mcts_player, is_shown=True,use_computerplay=True)

if __name__ == '__main__':
    run()