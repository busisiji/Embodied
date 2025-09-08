# 加载模型并进行自我对弈评估
import time
import cchess
from mcts import MCTS_AI
from core.game import Game
from core.net import PolicyValueNet


def evaluate_model(model_path, num_games=10):
    """
    加载模型并进行自我对弈评估
    :param model_path: 模型路径
    :param num_games: 对弈次数
    :return: 胜率、平局率
    """
    print(f"[{time.strftime('%H:%M:%S')}] 正在加载模型：{model_path}")
    policy_value_net = PolicyValueNet(model_file=model_path)

    # 创建两个 AI 玩家，使用相同的模型和不同模拟次数
    player1 = MCTS_AI(policy_value_fn=policy_value_net.policy_value_fn, n_playout=1200)
    player2 = MCTS_AI(policy_value_fn=policy_value_net.policy_value_fn, n_playout=800)

    game = Game(cchess.Board())

    win_count = 0
    tie_count = 0
    completed_games = 0

    try:
        for i in range(num_games):
            print(f"[{time.strftime('%H:%M:%S')}] 开始第 {i+1} 局对弈")
            winner = game.start_play(player1, player2, is_shown=False)  # 设置为 False 不显示棋盘

            if winner == cchess.RED:
                win_count += 1
            elif winner == -1:
                tie_count += 1
            completed_games += 1

            print(f"[{time.strftime('%H:%M:%S')}] 第 {i+1} 局结果：{'红方胜' if winner == cchess.RED else '黑方胜' if winner == cchess.BLACK else '平局'}")

    except KeyboardInterrupt:
        print("\n\n[评估中断] 用户已请求终止对弈。正在汇总已完成的对弈结果...\n")

    finally:
        completed_games = i  # 已完成的对弈数量
        if completed_games > 0:
            win_rate = win_count / completed_games
            tie_rate = tie_count / completed_games
            print(f"[部分评估结果] 已完成 {completed_games} 局")
            print(f"胜率: {win_rate:.2%}, 平局率: {tie_rate:.2%}\n")
        else:
            print("[无有效对弈记录]\n")

    return win_count / num_games if num_games > 0 else 0, tie_count / num_games if num_games > 0 else 0


if __name__ == "__main__":
    # 指定要评估的模型路径
    model_path = "../models/current_policy_batch300_2025-06-27_14-03-44.pkl"

    # 运行评估
    win_rate, tie_rate = evaluate_model(model_path, num_games=100)

    # 输出最终结果
    print(f"模型路径: {model_path}")
    print(f"胜率: {win_rate:.2%}")
    print(f"平局率: {tie_rate:.2%}")
