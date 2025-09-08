# csv_to_pkl.py
import argparse
import os
import sys

from src.cchessAI.core.crawl.crawl_data_csv import moves_add_gameinfo

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（cchess）
project_root = os.path.dirname(os.path.dirname(current_dir))
print(project_root)
# 如果不在 PYTHONPATH 中，则加入
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.data_utils import DataWriter
from utils.tools import move_action2move_id
import pandas as pd
import cchess  # 引入中国象棋模块
from collections import defaultdict, deque

from parameters import BUFFER_SIZE
from utils.tools import decode_board
from parameters import DATA_PATH
# UCI 格式棋盘索引对照表
"""
81:a9 82:b9 83:c9 84:d9 85:e9 86:f9 87:g9 88:h9 89:i9
72:a8 73:b8 74:c8 75:d8 76:e8 77:f8 78:g8 79:h8 80:i8
63:a7 64:b7 65:c7 66:d7 67:e7 68:f7 69:g7 70:h7 71:i7
54:a6 55:b6 56:c6 57:d6 58:e6 59:f6 60:g6 61:h6 62:i6
45:a5 46:b5 47:c5 48:d5 49:e5 50:f5 51:g5 52:h5 53:i5
36:a4 37:b4 38:c4 39:d4 40:e4 41:f4 42:g4 43:h4 44:i4
27:a3 28:b3 29:c3 30:d3 31:e3 32:f3 33:g3 34:h3 35:i3
18:a2 19:b2 20:c2 21:d2 22:e2 23:f2 24:g2 25:h2 26:i2
 9:a1 10:b1 11:c1 12:d1 13:e1 14:f1 15:g1 16:h1 17:i1
 0:a0  1:b0  2:c0  3:d0  4:e0  5:f0  6:g0  7:h0  8:i0
"""

"""
红方从右到左算列数
例如：C2+5 表示第8列的炮向上移动5行
k6+1 表示第6列的将向下移动1行
"""


class ChessDataLoader:
    def __init__(self):
        """
        初始化棋子类型和映射关系。
        红黑双方棋子符号定义（如 'R' 表示红车，'r' 表示黑车）。
        """

        self.ele_red = "B"  # 象（红）
        self.ele_black = 'b'  # 象（黑）
        self.horse_red = 'N'  # 马（红）
        self.horse_black = 'n'  # 马（黑）

        # 棋子符号映射表：用于转换不同表示方式的棋子字符
        self.piece_map = {
            'r': 'r', 'R': 'R',  # 车
            self.horse_black: 'n', self.horse_red: 'N',  # 马
            self.ele_black: 'b', self.ele_red: 'B',  # 象
            'a': 'a', 'A': 'A',  # 士
            'k': 'k', 'K': 'K',  # 将
            'c': 'c', 'C': 'C',  # 炮
            'p': 'p', 'P': 'P',  # 兵
        }

        self.states = []
        self.mcts_probs = []
        self.winners = []
        self.buffer_size = BUFFER_SIZE  # 经验池大小
        self.data_buffer = deque(maxlen=self.buffer_size)


    def get_red_col(self, col):
        """
        红方列号从右侧开始计算，此函数将标准列号转换为红方视角的列号。

        Args:
            col (int): 标准列号（从左侧开始计数）

        Returns:
            int: 红方视角的列号
        """
        return 8 - col

    def find_pieces_in_same_column(self, board: cchess.Board, piece_char: str) -> dict:
        """
        查找指定棋子类型在棋盘上是否出现在同一列。

        Args:
            board (cchess.Board): 当前的棋盘状态
            piece_char (str): 要查找的棋子符号（如 'r', 'n', 'k' 等）

        Returns:
            dict: 键为列号，值为该列上的此棋子所在位置列表
        """
        column_pieces = defaultdict(list)

        for square in range(90):  # 中国象棋棋盘总共有 90 个格子（10行9列）
            piece = board.piece_at(square)
            if piece and piece.symbol() == piece_char:
                col = cchess.square_column(square)
                column_pieces[col].append(square)

        return {col: squares for col, squares in column_pieces.items() if len(squares) > 1}

    def notation_to_uci(self, notation: str, side: str, board: cchess.Board) -> (str, object):
        """
        将中文象棋记谱法动作字符串转换为 UCI 格式的动作字符串。

        Args:
            notation (str): 如 "C2.5", "H2+3"
            side (str): 操作方（'red' 或 'black'）
            board (cchess.Board): 当前棋盘状态

        Returns:
            tuple: 包含 UCI 格式的动作字符串（如 "h2h9"）和对应的 Move 对象
        """
        same_col = None  # 是否出现同列相同棋子
        same_col_piece = []  # 同列相同棋子的信息存储列表

        piece_char = notation[0]  # 获取当前动作的棋子字符
        if piece_char not in self.piece_map:
            raise ValueError(f"未知棋子类型: {piece_char}")

        wxf_move = notation[1:]  # 去掉棋子字符后的动作描述部分
        legal_moves = list(board.legal_moves)  # 获取当前棋盘下的所有合法动作
        # if 'N-+4' == notation:
        #     print("测试")
        # 处理同列相同棋子的情况
        same_columns = self.find_pieces_in_same_column(board, self.piece_map[piece_char])
        if notation[1] in ['-', '+'] and same_columns:
            same_col = notation[1]
            sorted_squares = sorted(
                list(same_columns.values())[0],
                key=lambda sq: cchess.square_row(sq)
            )
            selected_square = sorted_squares[0] if notation[2] == '-' else sorted_squares[-1]
            from_col = cchess.square_column(selected_square)
            wxf_move = str(from_col) + wxf_move[1:]
        else:
            from_col = int(wxf_move[0]) - 1

        # 判断目标列号
        if '.' in wxf_move or piece_char.upper() in [self.horse_red, self.ele_red, 'A']:
            to_col = int(wxf_move[-1]) - 1
        else:
            to_col = from_col

        # 如果是红方操作，需要调整列号
        if side == 'red':
            if not same_col:
                from_col = self.get_red_col(from_col)
            if '.' in wxf_move or piece_char.upper() in [self.horse_red, self.ele_red, 'A']:
                to_col = self.get_red_col(to_col)
                wxf_move = str(from_col) + wxf_move[1] + str(to_col)
            else:
                to_col = from_col
                wxf_move = str(from_col) + wxf_move[1:]

        # 处理平移动作（如 C2.5）
        if '.' in wxf_move:
            for move in legal_moves:
                from_square, to_square = move.from_square, move.to_square
                from_name, to_name = cchess.SQUARE_NAMES[from_square], cchess.SQUARE_NAMES[to_square]
                from_col_sq = cchess.square_column(from_square)
                from_row_sq = cchess.square_row(from_square)
                to_col_sq = cchess.square_column(to_square)

                if (
                    from_col_sq == from_col and
                    to_col_sq == to_col and
                    self.piece_map[piece_char] == board.piece_at(from_square).symbol()
                ):
                    if not same_col:
                        return f"{from_name}{to_name}",move
                    else:
                        same_col_piece.append({from_row_sq:["{from_name}{to_name}",move]})


        # 处理进/退动作（如 H2+3）
        elif '+' in wxf_move or '-' in wxf_move:
            is_special_piece = piece_char.upper() in [self.horse_red, self.ele_red, 'A']
            for move in legal_moves:
                from_square, to_square = move.from_square, move.to_square
                from_name, to_name = cchess.SQUARE_NAMES[from_square], cchess.SQUARE_NAMES[to_square]
                from_row_sq = cchess.square_row(from_square)
                from_col_sq = cchess.square_column(from_square)
                to_row_sq = cchess.square_row(to_square)
                to_col_sq = cchess.square_column(to_square)

                if from_col != from_col_sq or to_col != to_col_sq:
                    continue
                if self.piece_map[piece_char] != board.piece_at(from_square).symbol():
                    continue

                if is_special_piece:
                    if (side == 'red' and '+' in wxf_move and from_row_sq > to_row_sq) or \
                       (side == 'red' and '-' in wxf_move and from_row_sq < to_row_sq) or \
                       (side == 'black' and '+' in wxf_move and from_row_sq < to_row_sq) or \
                       (side == 'black' and '-' in wxf_move and from_row_sq > to_row_sq):
                        continue
                    # 检查行是否在合理范围内
                    if from_col_sq == from_col and to_col == to_col_sq:
                        if self.piece_map[piece_char] == board.piece_at(from_square).symbol():
                            if not same_col:
                                return f"{from_name}{to_name}", move
                            else:
                                same_col_piece.append({from_row_sq: (f"{from_name}{to_name}", move)})
                else:
                    step = int(wxf_move[-1])
                    if side == 'red':
                        if ('+' in wxf_move and to_row_sq == from_row_sq + step) or \
                           ('-' in wxf_move and to_row_sq == from_row_sq - step):
                            if not same_col:
                                return f"{from_name}{to_name}", move
                            else:
                                same_col_piece.append({from_row_sq: (f"{from_name}{to_name}", move)})

                    else:
                        if ('+' in wxf_move and to_row_sq == from_row_sq - step) or \
                           ('-' in wxf_move and to_row_sq == from_row_sq + step):
                            if not same_col:
                                return f"{from_name}{to_name}", move
                            else:
                                same_col_piece.append({from_row_sq: (f"{from_name}{to_name}", move)})

        # 多个相同棋子处理
        if same_col_piece:
            values = same_col_piece[-1].values() if (
                (side == 'red' and same_col == '-') or (side == 'black' and same_col == '+')
            ) else same_col_piece[0].values()

            for key, move in values:
                return key, move

        return "", None



    def load_game_data(self, moves_file, gameinfo_file, data_path):
        """
        加载并解析棋局数据，按自我对弈格式保存到 DATA_BUFFER_PATH。
        根据文件后缀自动选择使用HDF5或pickle格式进行存储。

        Args:
            moves_file (str): 棋局步骤文件路径
            gameinfo_file (str): 棟局胜负信息文件路径
            data_path (str): 输出文件路径 (.h5/.pkl)
        """
        # 读取 CSV 数据
        moves_df = pd.read_csv(moves_file)
        gameinfo_df = pd.read_csv(gameinfo_file)

        # 构建游戏结果字典
        game_results = dict(zip(gameinfo_df['gameID'], gameinfo_df['winner']))
        grouped_moves = moves_df.groupby('gameID')

        print(f"[INFO] 开始加载棋局数据，目标文件: {data_path}")

        # 初始化 DataWriter
        writer = DataWriter(data_path)

        iters = 0
        chunk_size = 10
        play_data_all = []

        for game_id, group in grouped_moves:
            processed = self._process_single_game(game_id, group, game_results)
            if not processed:
                continue

            states, mcts_probs, winner_z = processed
            play_data = list(zip(states, mcts_probs, winner_z))
            # if writer.data_format == 'pkl':
            #     play_data = mirror_data(play_data)
            play_data_all.extend(play_data)
            # else:
            #     play_data_all = play_data
            iters += 1

            # 批量写入
            if iters % chunk_size == 0:
                writer.write(play_data_all,iters)
                play_data_all.clear()
                print(f"[INFO] 已处理 {iters} 局，当前文件大小: "
                      f"{os.path.getsize(data_path) / (1024 * 1024):.2f} MB")

        # 写入剩余数据
        if play_data_all:
            writer.write(play_data_all,iters)

        print(f"[INFO] 所有棋局已成功追加写入至 {data_path}，共处理 {iters} 局数据")


    def _process_single_game(self, game_id, group, game_results):
        """
        处理单个游戏对局的核心逻辑

        Args:
            game_id: 游戏ID
            group: 分组的游戏数据
            game_results: 游戏结果字典

        Returns:
            tuple: (states, mcts_probs, winner_z) 或 None（如果处理失败）
        """
        import numpy as np

        board = cchess.Board()
        winner = game_results.get(game_id, None)
        group = group.sort_values(by=['turn', 'side'], ascending=[True, False])

        print(f"[INFO] 解析棋局: {game_id}")

        states, mcts_probs, current_players = [], [], []

        for _, row in group.iterrows():
            move_str, side, turn = row['move'], row['side'], row['turn']
            move_str = move_str[0].upper() + move_str[1:] if side == 'red' else move_str[0].lower() + move_str[1:]

            try:
                move_uci, move = self.notation_to_uci(move_str, side, board)
                if move is None:
                    print(f"[警告] 无法找到合法动作: {move_uci}")
                    return None
            except Exception as e:
                print(f"[警告] 非法动作格式: {move_str}, 错误: {e}")
                return None

            if not board.is_legal(move):
                print(f"[警告] 非法动作: {move_uci}")
                return None

            current_state = decode_board(board)
            states.append(current_state)
            prob = np.zeros(2086, dtype=np.float32)

            uci_move = move.uci()
            move_idx = move_action2move_id.get(uci_move, -1)
            if move_idx != -1:
                prob[move_idx] = 1.0
            else:
                print(f"[警告] 动作 {uci_move} 未找到对应的 move_id")

            mcts_probs.append(prob)
            current_players.append(board.turn)
            board.push(move)

        outcome = board.outcome()

        # 只要将军的棋谱
        # if outcome is None:
        #     print(f"[警告] 棋局 {game_id} 跳过，非完整棋局")
        #     return

        if outcome is None:
            if side == 'red':
                winner_z = np.array([1.0 for player in current_players])
            elif side == 'black':
                winner_z = np.array([-1.0 for player in current_players])
        else:
            winner_z = np.array([
                1.0 if player == outcome.winner else -1.0
                for player in current_players
            ]) if outcome.winner is not None else np.zeros(len(current_players))
        print(f"[INFO] 胜方: {winner}/{side},")
        return states, mcts_probs, winner_z



def remove_duplicate_rows(file_path, subset=['gameID', 'turn', 'side']):
    """
    删除指定 CSV 文件中 gameID、turn、side 列组合重复的行，只保留第一次出现的记录。

    参数:
        file_path (str): CSV 文件路径
        subset (list): 去重依据的列名列表

    返回:
        bool: 是否进行了修改
    """

    df = pd.read_csv(file_path)

    # 检查是否有重复行
    duplicates = df.duplicated(subset=subset, keep='first')

    if not duplicates.any():
        print(f"[INFO] {file_path} 中没有重复的 {subset} 行。")
        return False

    # 删除重复行
    df_cleaned = df[~duplicates]

    # 保存到原文件
    df_cleaned.to_csv(file_path, index=False)
    print(f"[INFO] 已删除 {sum(duplicates)} 行重复的 {subset} 数据，并保存至 {file_path}")
    return True

if __name__ == "__main__":
    # 设置参数解析器
    parser = argparse.ArgumentParser(description='运行棋谱抓取脚本')
    parser.add_argument('--owner', type=str, default='n')
    parser.add_argument('--moves', type=str, default='moves2')
    parser.add_argument('--gameinfo', type=str, default='gameinfo2')
    parser.add_argument('--outfile', type=str, default="data_buffer2.pkl")


    args = parser.parse_args()

    folder_path = os.path.join(DATA_PATH, args.owner)
    moves_file = os.path.join(folder_path, args.moves+".csv")
    gameinfo_file = os.path.join(folder_path, args.gameinfo+".csv")
    out_file = os.path.join(folder_path, args.outfile)

    if not os.path.exists(gameinfo_file):
        moves_add_gameinfo(moves_file, gameinfo_file)
    # 预处理：删除重复的 gameID、turn、side 组合行
    remove_duplicate_rows(moves_file, subset=['gameID', 'turn', 'side'])
    remove_duplicate_rows(gameinfo_file, subset=['gameID'])

    chess = ChessDataLoader()
    data_buffer = chess.load_game_data(moves_file, gameinfo_file , out_file)
