# 修改 game.py 文件中的 Game 类

import os
import sys

from src.cchessAG import constants, my_chess
from src.cchessAG.chinachess import MainGame
from src.cchessAG.computer import movedeep
from src.cchessAG.my_game import my_game

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（cchess）
project_root = os.path.dirname(current_dir)

# 如果不在 PYTHONPATH 中，则加入
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cchess
import cchess.svg
import time
import numpy as np
from IPython.display import display, SVG

from core.frontend import get_chess_window, create_window_visualization
from utils.tools import decode_board, move_id2move_action, is_tie


class Game(object):
    """
    在cchess.Board类基础上定义Game类, 用于启动并控制一整局对局的完整流程,
    收集对局过程中的数据，以及进行棋盘的展示
    """

    def __init__(self, board, window_id=None):
        self.board = board
        self.window_id = window_id
        self.window = None
        if window_id:
            self._create_window(window_id)

    # 添加创建窗口的方法
    def _create_window(self, window_id):
        """为游戏创建专用窗口"""
        # 使用固定端口映射避免冲突
        base_port = 8000
        port = base_port + hash(window_id) % 1000 + 1  # 简单的端口分配策略

        window_creator = create_window_visualization(host="127.0.0.1", port=port)
        self.window = window_creator()
        if hasattr(self.window, 'start'):
            self.window.start(window_id)
    def get_window_url(self):
        """
        返回游戏窗口的 URL
        """
        try:
            window = self.window if self.window else get_chess_window()
            host = window.host
            port = window.port
            if host in ["0.0.0.0", "127.0.0.1", "localhost"]:
                import socket
                # 优先获取eth0接口的IP地址
                actual_ip = self._get_eth0_ip()
                if not actual_ip:
                    # 如果没有eth0，则获取默认网络接口的IP
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(("8.8.8.8", 80))
                    actual_ip = s.getsockname()[0]
                    s.close()
                host = actual_ip if actual_ip else host
            return f"http://{host}:{port}"
        except:
            print("无法获取窗口 URL")
            return None

    def _get_eth0_ip(self):
        """
        获取eth0网络接口的IP地址
        """
        try:
            import netifaces
            # 获取eth0接口的IPv4地址
            if 'eth0' in netifaces.interfaces():
                addresses = netifaces.ifaddresses('eth0')
                if netifaces.AF_INET in addresses:
                    ip = addresses[netifaces.AF_INET][0]['addr']
                    # 验证是否为有效的IPv4地址
                    if ip and not ip.startswith('127.'):
                        return ip
        except ImportError:
            print("未安装netifaces库，使用备用方法获取IP")
            import subprocess
            import re
            result = subprocess.run(['ifconfig', 'eth0'], capture_output=True, text=True)
            if result.returncode == 0:
                # 使用正则表达式匹配IP地址
                ip_match = re.search(r'inet\s+(\d+\.\d+\.\d+\.\d+)', result.stdout)
                if ip_match:
                    ip = ip_match.group(1)
                    # 验证是否为有效的IPv4地址且不是回环地址
                    if ip and not ip.startswith('127.'):
                        return ip
        except Exception as e:
            print(f"获取eth0 IP地址时出错: {e}")
            pass
        return None

    def graphic(self, board):
        """可视化棋盘"""
        svg = cchess.svg.board(
            board,
            size=600,
            coordinates=True,
            axes_type=0,
            checkers=board.checkers(),
            lastmove=board.peek() if len(board.move_stack) > 0 else None,
            orientation=cchess.RED,
        )

        current_player = "红方" if board.turn == cchess.RED else "黑方"
        status_text = f"当前走子: {current_player} - 步数: {len(board.move_stack)}"

        try:
            # 优先使用游戏实例的专用窗口
            window = self.window if self.window else get_chess_window()
            if window:
                window.update_board(svg, status_text)
                time.sleep(0.1)
            else:
                display(SVG(svg))
        except ImportError:
            display(SVG(svg))


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
            current_turn = "红方" if self.board.turn == cchess.RED else "黑方"
            print(f"[{time.strftime('%H:%M:%S')}] 当前轮到: {current_turn}")

            player_in_turn = players[self.board.turn]

            move = player_in_turn.get_action(self.board)
            print(f"[{time.strftime('%H:%M:%S')}] 动作已获取: {move_id2move_action[move]}")

            # 执行移动
            self.board.push(cchess.Move.from_uci(move_id2move_action[move]))

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
                        status_text = f"游戏结束，赢家是：{winner_name}"
                        print(
                            f"[{time.strftime('%H:%M:%S')}] {status_text}"
                        )
                        try:
                            window = self.window if self.window else get_chess_window()
                            if window:
                                window.update_status(status_text)
                                time.sleep(0.5)  # 给窗口一点时间更新
                        except ImportError:
                            print(status_text)
                else:
                    winner = -1
                    if is_shown:
                        status_text = "游戏结束. 平局"
                        print(f"[{time.strftime('%H:%M:%S')}] {status_text}")
                        try:
                            window = self.window if self.window else get_chess_window()
                            if window:
                                window.update_status(status_text)
                                time.sleep(0.5)  # 给窗口一点时间更新
                        except ImportError:
                            print(status_text)

                return winner

    # 使用蒙特卡洛树搜索开始自我对弈，存储游戏状态（状态，蒙特卡洛落子概率，胜负手）三元组用于神经网络训练
    def start_self_play(self, player, is_shown=False, temp=1e-3):
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

        while True:
            move_count += 1

            # 每20步输出一次耗时
            if move_count % 1 == 0:
                start_time = time.time()
                move, move_probs = player.get_action(
                    self.board, temp=temp, return_prob=True
                )
                print(
                    f"[{time.strftime('%H:%M:%S')}] 第{move_count}步耗时: {time.time() - start_time:.2f}秒"
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
                            f"[{time.strftime('%H:%M:%S')}] 游戏结束. 赢家是: {winner_name}"
                        )
                else:
                    # 平局情况
                    winner = -1
                    if is_shown:
                        print(f"[{time.strftime('%H:%M:%S')}] 游戏结束. 平局")

                # 重置蒙特卡洛根节点
                player.reset_player()

                # 返回胜方和游戏数据
                return winner, zip(states, mcts_probs, winner_z)


    def start_play_with_mixed_strategy(self, player1, player0, is_shown=True, use_computerplay=True):
        """
        启动一场对局，并结合 MCTS 与 ComputerPlay 的混合策略

        Args:
            player1: 玩家1(红方)
            player0: 玩家0(黑方)
            is_shown: 是否显示棋盘
            use_computerplay: 是否启用 ComputerPlay 作为辅助策略

        Returns:
            winner: 获胜方, True (cchess.RED) 或 False (cchess.BLACK) 或 None (平局)
        """
        move_uci = ''
        # 初始化棋盘
        self.board = cchess.Board()

        # 设置玩家(默认玩家1先手)
        player1.set_player_idx(1)
        player0.set_player_idx(0)
        players = {cchess.RED: player1, cchess.BLACK: player0}

        mgInit = my_game()  # 初始化 my_game 实例
        maingame = MainGame()
        maingame.piecesInit()
        # 显示初始棋盘
        if is_shown:
            self.graphic(self.board)

        # 开始游戏循环
        while True:
            current_turn = "红方" if self.board.turn == cchess.RED else "黑方"
            print(f"[{time.strftime('%H:%M:%S')}] 当前轮到: {current_turn}")

            player_in_turn = players[self.board.turn]

            # move = player_in_turn.get_action(self.board)
            if not use_computerplay or player_in_turn.agent == 'HUMAN':
                move = player_in_turn.get_action(self.board)
                move_uci = move_id2move_action[move]

                print(f"[{time.strftime('%H:%M:%S')}] 动作已获取: {move_uci}")

            # 如果启用 ComputerPlay 并且当前是 AI 下棋
            if use_computerplay and player_in_turn.agent == 'AI' :
                from_x, from_y, to_x, to_y = uci_to_coordinates(move_uci)
                move_uci = get_best_move_with_computer_play(maingame,self.board,from_x, from_y, to_x, to_y)
                if move_uci:
                    print(f"[INFO] ComputerPlay 推荐走法: {move_uci}")

            # 执行移动
            self.board.push(cchess.Move.from_uci(move_uci))


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
                        status_text = f"游戏结束，赢家是：{winner_name}"
                        print(f"[{time.strftime('%H:%M:%S')}] {status_text}")
                        try:
                            window = self.window if self.window else get_chess_window()
                            if window:
                                window.update_status(status_text)
                                time.sleep(0.5)  # 给窗口一点时间更新
                        except ImportError:
                            print(status_text)
                else:
                    winner = -1
                    if is_shown:
                        status_text = "游戏结束. 平局"
                        print(f"[{time.strftime('%H:%M:%S')}] {status_text}")
                        try:
                            window = self.window if self.window else get_chess_window()
                            if window:
                                window.update_status(status_text)
                                time.sleep(0.5)  # 给窗口一点时间更新
                        except ImportError:
                            print(status_text)

                return winner
                
    def close_window(self):
        """关闭游戏窗口"""
        if self.window:
            self.window.stop()
            self.window = None

def uci_to_coordinates(move_str):
    """
    将代数记谱法（如 c3c4）转换为以左上角为 (0, 0) 的坐标表示。

    Args:
        move_str (str): 代数记谱法的移动字符串，如 'c3c4'。

    Returns:
        tuple: 起始坐标的 (x1, y1) 和目标坐标的 (x2, y2)，以左上角为原点。
    """
    try:
        # 解析输入字符串
        start_col = move_str[0]  # 起点列
        start_row = move_str[1]  # 起点行
        end_col = move_str[2]    # 目标列
        end_row = move_str[3]    # 目标行

        # 转换为数字坐标
        x1 = ord(start_col) - ord('a')  # 列字母转为数字（a=0, b=1, ..., h=7）
        y1 = 9 - int(start_row)         # 行号转换为从上到下的坐标（1->7, 2->6, ..., 8->0）
        x2 = ord(end_col) - ord('a')
        y2 = 9 - int(end_row)

        return x1, y1, x2, y2
    except:
        return 0,9,0,9

# 在 /home/jetson/Desktop/Embodied/src/cchessAI/core/game.py 文件中修改 get_best_move_with_computer_play 函数

def get_best_move_with_computer_play(maingame, board, from_x, from_y, to_x, to_y):
    """
    结合当前棋盘状态，使用 computer.getPlayInfo 获取最佳动作
    注意：只计算移动，不执行移动
    """
    try:
        if board.turn == cchess.RED:
            best_move_info = movedeep(
                MainGame.piecesList,
                0,
                constants.player1Color,
                from_x,
                9-from_y,
                to_x,
                9-to_y,
                maingame.mgInit,
                False
            )
        else:
            best_move_info = movedeep(
                MainGame.piecesList,
                1,
                constants.player2Color,
                from_x,
                from_y,
                to_x,
                to_y,
                maingame.mgInit,
                False
            )
        if best_move_info:
            x1, y1, x2, y2 = 8 - best_move_info.from_x, best_move_info.from_y, 8 - best_move_info.to_x, best_move_info.to_y
            if board.turn == cchess.RED:
                move_str = f"{chr(97 + x1)}{y1}{chr(97 + x2)}{y2}"
            else:
                move_str = f"{chr(97 + x1)}{9-y1}{chr(97 + x2)}{9-y2}"

            return move_str
        else:
            return None
    except Exception as e:
        print(f"[ERROR] 获取 ComputerPlay 动作失败: {e}")
        return None


def execute_computer_move(maingame,board, move_uci):
    """
    在确定采用ComputerPlay推荐的移动后，更新游戏状态

    Args:
        maingame: 主游戏对象
        move_uci: UCI格式的移动字符串
    """
    try:
        # 解析UCI移动字符串为内部表示
        from_col = ord(move_uci[0]) - ord('a')
        from_row = int(move_uci[1])
        to_col = ord(move_uci[2]) - ord('a')
        to_row = int(move_uci[3])

        # 转换为ComputerPlay坐标系
        if board.turn != cchess.RED:
            # 黑方视角转换
            s = my_chess.step(8 - from_col, 9 - from_row, 8 - to_col, 9 - to_row)
        else:
            # 红方视角转换
            s = my_chess.step(8 - from_col, from_row, 8 - to_col, to_row)

        # 执行移动
        maingame.mgInit.move_to(s)
        print(f"已更新游戏状态，执行移动: {s}")

    except Exception as e:
        print(f"[ERROR] 执行ComputerPlay移动失败: {e}")

