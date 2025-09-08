import argparse
import os
import sys
import time
from src.cchessAI.core.game import Game
from src.cchessAI.core.mcts import MCTS_AI
from src.cchessAI.core.net import PolicyValueNet
import cchess

def evaluate_model(model_path, num_games=10):
    """
    评估模型性能
    """
    print(f"[{time.strftime('%H:%M:%S')}] 开始评估模型: {model_path}")
    print(f"[{time.strftime('%H:%M:%S')}] 对局数: {num_games}")

    # 加载模型
    try:
        policy_value_net = PolicyValueNet(model_file=model_path)
        print(f"[{time.strftime('%H:%M:%S')}] 模型加载成功")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] 模型加载失败: {e}")
        return None

    # 创建两个使用相同模型的AI玩家
    player1 = MCTS_AI(policy_value_fn=policy_value_net.policy_value_fn, c_puct=5, n_playout=400)
    player2 = MCTS_AI(policy_value_fn=policy_value_net.policy_value_fn, c_puct=5, n_playout=400)

    # 进行对局
    red_wins = 0
    black_wins = 0

    for i in range(num_games):
        print(f"[{time.strftime('%H:%M:%S')}] 开始第 {i+1}/{num_games} 局")

        # 创建新游戏
        board = cchess.Board()
        game = Game(board)

        # 开始对局
        winner, _ = game.start_play(player1=player1, player2=player2, is_shown=True)

        if winner == cchess.RED:
            red_wins += 1
            print(f"[{time.strftime('%H:%M:%S')}] 第 {i+1} 局结果: 红方胜")
        else:
            black_wins += 1
            print(f"[{time.strftime('%H:%M:%S')}] 第 {i+1} 局结果: 黑方胜")

        print(f"[{time.strftime('%H:%M:%S')}] 当前比分 - 红方: {red_wins}, 黑方: {black_wins}")

    # 输出最终结果
    print(f"[{time.strftime('%H:%M:%S')}] 评估完成")
    print(f"[{time.strftime('%H:%M:%S')}] 总对局数: {num_games}")
    print(f"[{time.strftime('%H:%M:%S')}] 红方胜率: {red_wins/num_games*100:.2f}%")
    print(f"[{time.strftime('%H:%M:%S')}] 黑方胜率: {black_wins/num_games*100:.2f}%")

    return {
        "total_games": num_games,
        "red_wins": red_wins,
        "black_wins": black_wins,
        "red_win_rate": red_wins/num_games,
        "black_win_rate": black_wins/num_games
    }

def main():
    parser = argparse.ArgumentParser(description="模型评估工具")
    parser.add_argument("--model-path", type=str, required=True, help="模型文件路径")
    parser.add_argument("--num-games", type=int, default=10, help="评估对局数")

    args = parser.parse_args()

    # 执行评估
    result = evaluate_model(args.model_path, args.num_games)

    if result:
        print(f"[{time.strftime('%H:%M:%S')}] 评估结果: {result}")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] 评估失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
