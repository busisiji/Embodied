# 获取当前文件所在目录
import os
import sys

from src.cchessAI.core.mcts import MCTS_AI

current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（cchess）
project_root = os.path.dirname(current_dir)

# 如果不在 PYTHONPATH 中，则加入
if project_root not in sys.path:
    print(f"[INFO] 正在加入 {project_root}...")
    sys.path.insert(0, project_root)
import threading

import cchess
import cchess.svg
import time
import numpy as np
from IPython.display import display, SVG

from parameters import PLAYOUT, C_PUCT, TEMP, IS_DYNAMIC_PLAYOUT
from utils.tools import decode_board, move_id2move_action, is_tie
from core.frontend import create_window_visualization


class Game(object):
    """
    在cchess.Board类基础上定义Game类, 用于启动并控制一整局对局的完整流程,
    收集对局过程中的数据，以及进行棋盘的展示
    """

    def __init__(self, board, port=8000):
        self.board = board
        self.port = port
        self.chess_window = create_window_visualization(port=self.port)

        # 动态 playout 参数
        if IS_DYNAMIC_PLAYOUT:
            # self.c_puct_schedule = [
            #     ((0, 30), (200, C_PUCT if C_PUCT > 0 else 5),(TEMP if TEMP >= 1 else 2)),
            #     ((30, 60), (400, C_PUCT-1 if C_PUCT > 1 else 3),(TEMP/10 if TEMP >= 1 else 1)),
            #     ((60, 120), (800, C_PUCT-2 if C_PUCT > 2 else 1),(TEMP/100 if TEMP >= 1 else 1e-2)),
            #     ((120, float('inf')), (1200, C_PUCT if C_PUCT > 0 else 3),(TEMP/1000 if TEMP >= 1 else 1e-3)),
            # ]
            self.c_puct_schedule = [
                ((0, 30), (PLAYOUT-400 if PLAYOUT > 500 else 400, C_PUCT if C_PUCT > 0 else 3),(TEMP if TEMP >= 1 else 1)),
                ((30, 60), (PLAYOUT-200 if PLAYOUT > 400 else 400, C_PUCT-1 if C_PUCT > 1 else 2),(TEMP/10 if TEMP >= 1 else 1e-1)),
                ((60, 120), (PLAYOUT if PLAYOUT >= 800 else 800, C_PUCT-2 if C_PUCT > 2 else 1),(TEMP/100 if TEMP >= 1 else 1e-2)),
                ((120, float('inf')), (PLAYOUT if PLAYOUT >= 1200 else 1200, C_PUCT if C_PUCT > 0 else 3),(TEMP/1000 if TEMP >= 1 else 1e-3)),
            ]
        else:
            self.c_puct_schedule = [
                ((0, float('inf')),(PLAYOUT, C_PUCT), (TEMP))
            ]

    # 可视化棋盘
    def graphic(self, board):
        """print(
            f"[{time.strftime('%H:%M:%S')}] player1 take: ",
            "RED" if cchess.RED else "BLACK",
        )
        print(
            f"[{time.strftime('%H:%M:%S')}] player0 take: ",
            "BLACK" if cchess.RED else "RED",
        )"""
        svg = cchess.svg.board(
            board,
            size=600,
            coordinates=True,
            axes_type=0,
            checkers=board.checkers(),
            lastmove=board.peek() if len(board.move_stack) > 0 else None,
            orientation=cchess.RED,
        )
        # 获取当前玩家

        current_player = "红方" if board.turn == cchess.RED else "黑方"
        status_text = f"当前走子: {current_player} - 步数: {len(board.move_stack)}"

        # 尝试在窗口中显示
        try:
            window = self.chess_window()
            if window:
                window.update_board(svg, status_text)
                # 给窗口一点时间更新
                time.sleep(0.1)
            else:
                # 如果窗口创建失败，回退到终端显示
                display(SVG(svg))
        except ImportError:
            # 如果无法导入窗口函数，回退到终端显示
            display(SVG(svg))
    # 异步显示棋盘
    def async_graphic_update(self, board):
        thread = threading.Thread(target=self.graphic, args=(board,))
        thread.start()
    # 用于人机对战，人人对战等
    def start_play(self, player1, player0, is_shown=True):
        """
        启动一场对局

        Args:
            player1: 玩家1(红方)
            player0: 玩家0(黑方)
            先手玩家1
            is_shown: 是否显示棋盘

        Returns:
            winner: 获胜方, True (cchess.RED) 或 False (cchess.BLACK) 或 None (平局)
        """

        # 初始化棋盘
        self.board = cchess.Board()

        # 设置玩家(默认玩家1先手)
        player1.set_player_idx(1)
        player0.set_player_idx(0)
        players = {cchess.RED: player1, cchess.BLACK: player0}

        # 显示初始棋盘
        if is_shown:
            self.graphic(self.board)

        # 开始游戏循环
        while True:
            player_in_turn = players[self.board.turn]
            move = player_in_turn.get_action(self.board)

            # 执行移动
            self.board.push(move)

            # 更新显示
            if is_shown:
                self.graphic(self.board)

            # 检查游戏是否结束
            if self.board.is_game_over():
                outcome = self.board.outcome()
                if outcome.winner is not None:
                    winner = outcome.winner
                    if is_shown:
                        winner_name = "RED" if winner == cchess.RED else "BLACK"
                        print(
                            f"[{time.strftime('%H:%M:%S')}] 游戏结束. 赢家是: {winner_name}"
                        )
                else:
                    winner = -1
                    if is_shown:
                        print(f"[{time.strftime('%H:%M:%S')}] 游戏结束. 平局")
                return winner

    def get_c_puct_by_step(self, step):
        return next(
            ((n_playout, c_puct, temp) for (low, high), (n_playout, c_puct), temp in self.c_puct_schedule if
             low <= step < high),
            (PLAYOUT, C_PUCT, TEMP)
        )
    def reset_mcts(self, policy_value_net, n_playout=None, c_puct=None):
        """重置MCTS"""
        n_playout = n_playout or PLAYOUT
        c_puct = c_puct or C_PUCT
        return MCTS_AI(
            policy_value_net.policy_value_fn,
            c_puct=c_puct,
            n_playout=n_playout,
            is_selfplay=True
        )
    # 使用蒙特卡洛树搜索开始自我对弈，存储游戏状态（状态，蒙特卡洛落子概率，胜负手）三元组用于神经网络训练
    def start_self_play(self, policy_value_net, is_shown=False, temp=1e-3, pid=None, board=None):
        """
        开始自我对弈，用于收集训练数据

        Args:
            player: 自我对弈的玩家(MCTS_AI)
            is_shown: 是否显示棋盘
            temp: 温度参数，控制探索度

        Returns:
            winner: 获胜方
            play_data: 包含(state, mcts_prob, winner)的元组列表，用于训练
        """
        # 初始化棋盘
        self.board = cchess.Board()

        # 初始化数据收集列表
        states, mcts_probs, current_players = [], [], []

        # 开始自我对弈
        move_count = 0
        # 初始化玩家
        n_playout, c_puct, temp = self.get_c_puct_by_step(move_count)
        player = self.reset_mcts(policy_value_net, n_playout, c_puct)

        while True:
            try:
                move_count += 1
                if move_count % 30 == 0:
                    n_playout, c_puct, temp = self.get_c_puct_by_step(move_count)
                # 只在必要时更新参数
                if n_playout != player.mcts.n_playout or abs(c_puct - player.mcts.c_puct) > 1e-6:
                    player.mcts.n_playout = n_playout
                    player.mcts.c_puct = c_puct

                # 每1步输出一次耗时
                if move_count % 1 == 0 or move_count == 1:
                    start_time = time.time()
                    move, move_probs = player.get_action(
                        self.board, temp=temp, return_prob=True
                    )
                    print(
                        f"[{time.strftime('%H:%M:%S')}][PID={pid}] 第{move_count}步耗时: {time.time() - start_time:.2f}秒"
                    )
                else:
                    move, move_probs = player.get_action(
                        self.board, temp=temp, return_prob=True
                    )

                # 保存自我对弈的数据
                current_state = decode_board(self.board)
                states.append(current_state)
                mcts_probs.append(move_probs)
                current_players.append(self.board.turn)

                # 执行一步落子
                self.board.push(cchess.Move.from_uci(move_id2move_action[move]))

                # 显示当前棋盘状态
                if is_shown:
                    # self.async_graphic_update(self.board)
                    self.graphic(self.board)

                # 检查游戏是否结束
                if self.board.is_game_over() or is_tie(self.board):
                    # 处理游戏结束情况
                    outcome = self.board.outcome() if self.board.is_game_over() else None

                    # 初始化胜负信息
                    winner_z = np.zeros(len(current_players))

                    if outcome and outcome.winner is not None:
                        winner = outcome.winner
                        # 根据胜方设置奖励
                        for i, player_id in enumerate(current_players):
                            winner_z[i] = 1.0 if player_id == winner else -1.0

                        if is_shown:
                            winner_name = "RED" if winner == cchess.RED else "BLACK"
                            print(
                                f"[{time.strftime('%H:%M:%S')}][PID={pid}] 游戏结束. 赢家是: {winner_name}"
                            )
                    else:
                        # 平局情况
                        winner = -1
                        if is_shown:
                            print(f"[{time.strftime('%H:%M:%S')}][PID={pid}] 游戏结束. 平局")

                    # 重置蒙特卡洛根节点
                    player.reset_player()

                    # 返回胜方和游戏数据
                    return winner, zip(states, mcts_probs, winner_z)
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}][PID={pid}] 错误: {e}")
                raise e