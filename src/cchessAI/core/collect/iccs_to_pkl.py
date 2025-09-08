import argparse
import os

from src.cchessAI.parameters import DATA_PATH
from utils.data_utils import mirror_data, DataWriter
from utils.tools import decode_board, move_action2move_id

ICCS_HEADER_TEMPLATE = """[Game "Chinese Chess"]
[Event "{event}"]
[Site "{site}"]
[Date "{date}"]
[Round "{round}"]
[RedTeam "{red_team}"]
[Red "{red_player}"]
[BlackTeam "{black_team}"]
[Black "{black_player}"]
[Result "{result}"]
[Opening "{opening}"]
[FEN "{fen}"]
[Format "ICCS"]
"""


def iccs_to_uci(move_str):
    """
    将 ICCS 格式的动作字符串转换为 UCI 格式。

    Args:
        move_str (str): ICCS 格式的动作字符串，如 "C3-C4"

    Returns:
        str: UCI 格式的动作字符串，如 "c3c4"
    """
    if '-' in move_str:
        from_pos, to_pos = move_str.split('-')
        return from_pos.lower() + to_pos.lower()
    else:
        return move_str.lower()


def format_moves(moves):
    """
    将动作列表按 ICCS 格式分组输出。

    Args:
        moves (list): 动作字符串列表

    Returns:
        str: 格式化后的棋步文本
    """
    lines = []
    for i in range(0, len(moves), 2):
        line_number = i // 2 + 1
        red_move = moves[i] if i < len(moves) else ""
        black_move = moves[i + 1] if i + 1 < len(moves) else ""
        lines.append(f"{line_number}. {red_move} {black_move}")
    return "\n".join(lines)


def write_iccs_file(game_id, moves_list, output_path):
    """
    将单个棋局写入 ICCS 格式文件。

    Args:
        game_id (str): 棋局 ID
        moves_list (list): 包含 UCI 格式的动作列表
        output_path (str): 输出文件路径
    """
    header = ICCS_HEADER_TEMPLATE.format(
        event="自动生成比赛",
        site="-",
        date="2025-04-05",
        round="-",
        red_team="AI Team",
        red_player="AI Red",
        black_team="AI Team",
        black_player="AI Black",
        result="*",
        opening="-",
        fen="rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    )

    formatted_moves = format_moves(moves_list)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write("\n")
        f.write(formatted_moves)
        f.write("\n")
        f.write("1-0\n")

    print(f"[INFO] 已保存 ICCS 文件至 {output_path}")


def parse_iccs_file(file_path):
    """
    解析 ICCS 文件，按棋局分组提取动作列表。

    Args:
        file_path (str): ICCS 文件路径

    Returns:
        list: 每个元素是一个 tuple，包含 (game_id, moves_list)
    """
    games = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_game_moves = []
    game_id = "game_1"
    game_count = 1

    for line in lines:
        line = line.strip()
        if line.startswith("[Game"):
            if current_game_moves:
                # 保存上一局
                games.append((game_id, current_game_moves))
                current_game_moves = []
            game_id = f"game_{game_count}"
            game_count += 1
        elif line and line[0].isdigit():
            parts = line.split()
            if len(parts) >= 2:
                red_move = parts[1].strip()
                black_move = parts[2].strip() if len(parts) > 2 else ""
                if red_move and '-' in red_move:
                    current_game_moves.append(red_move)
                if black_move and '-' in black_move:
                    current_game_moves.append(black_move)

    if current_game_moves:
        games.append((game_id, current_game_moves))

    return games


def generate_training_data_for_game(moves):
    """
    根据动作序列生成训练数据 (state, mcts_prob, winner)，仅当整局合法时返回数据。

    Args:
        moves (list): ICCS 格式的动作字符串列表

    Returns:
        list or None: 合法则返回训练数据；否则返回 None
    """
    import numpy as np
    import cchess

    board = cchess.Board()
    training_data = []

    for i, iccs_move in enumerate(moves):
        try:
            # 将 ICCS 动作转换为 UCI 格式
            uci_move = iccs_to_uci(iccs_move)

            state = decode_board(board)

            # 创建 MCTS 概率向量
            mcts_prob = np.zeros(2086, dtype=np.float32)
            move_idx = move_action2move_id.get(uci_move, -1)
            if move_idx != -1:
                mcts_prob[move_idx] = 1.0

            # 假设胜者为最后一步走子方
            winner = 1.0 if i % 2 == 0 else -1.0

            training_data.append((state, mcts_prob, winner))

            # 验证动作合法性
            move = board.parse_uci(uci_move)
            if board.is_legal(move):
                board.push(move)
            else:
                print(f"[警告] 非法动作: {uci_move}，整局已被跳过")
                return None  # 整局跳过

        except Exception as e:
            print(f"[警告] 处理动作失败: {e}，整局已被跳过")
            return None  # 整局跳过

    return training_data
def parse_iccs_file_in_batches(file_path, batch_size=100):
    """
    分批解析 ICCS 文件，每次返回一个批次的棋局数据。

    Args:
        file_path (str): ICCS 文件路径
        batch_size (int): 每批次包含的棋局数量

    Yields:
        list: 批次内的棋局列表，每个元素是 (game_id, moves_list)
    """
    current_game_moves = []
    game_id = "game_1"
    game_count = 1
    batch = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("[Game"):
                if current_game_moves:
                    batch.append((game_id, current_game_moves))
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                    game_count += 1
                    game_id = f"game_{game_count}"
                    current_game_moves = []
            elif line and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 2:
                    red_move = parts[1].strip()
                    black_move = parts[2].strip() if len(parts) > 2 else ""
                    if '-' in red_move:
                        current_game_moves.append(red_move)
                    if '-' in black_move:
                        current_game_moves.append(black_move)

        # 处理最后一批
        if current_game_moves:
            batch.append((game_id, current_game_moves))
        if batch:
            yield batch



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ICCS to PKL 转换器')
    parser.add_argument('--owner', type=str, default='_m_',
                        help='棋盘所有者，用于生成棋盘文件路径')
    parser.add_argument('--input', type=str, default='dpxq-99813games.pgns',
                        help='输入的 ICCS 文件路径')
    parser.add_argument('--output', type=str, default='data_buffer_iccs.h5',
                        help='输出的 PKL 文件路径')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='每批次处理的棋局数量')
    parser.add_argument('--parallel', default=False, help='启用多线程写入')
    args = parser.parse_args()

    folder_path = os.path.join(DATA_PATH, 'ICCS')
    input_file = os.path.join(folder_path, args.input)
    out_file = os.path.join(folder_path, args.output)

    # 初始化 DataWriter
    writer = DataWriter(out_file)

    iters = 0
    for idx, batch_games in enumerate(parse_iccs_file_in_batches(input_file, args.batch_size)):
        print(f"[INFO] 正在处理并写入第 {idx + 1} 批次，共 {len(batch_games)} 局棋局")
        batch_data = []

        for game_idx, (game_id, moves) in enumerate(batch_games):
            print(f"  - 处理第 {game_idx + 1} 局棋局: {game_id}")
            game_data = generate_training_data_for_game(moves)
            if not game_data:
                print(f"    第 {game_idx + 1} 局跳过")
                continue
            if writer.data_format == 'pkl':
                play_data = mirror_data(game_data)
            else:
                play_data = game_data
            batch_data.extend(play_data)
            print(f"    已生成 {len(game_data)} 条训练样本")
            iters += 1

        # 使用 DataWriter 写入当前批次数据
        if args.parallel:
            writer.bulk_write_parallel(batch_data,iters)
        else:
            writer.write(batch_data, iters=iters)

    print(f"[INFO] 数据已全部写入至 {out_file}")

