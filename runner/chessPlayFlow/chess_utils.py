import asyncio
import os
import time

import numpy as np

from parameters import WORLD_POINTS_R, WORLD_POINTS_RCV, WORLD_POINTS_B, CHESS_POINTS_R, CHESS_POINTS_RCV_H, \
    CHESS_POINTS_B, CHESS_POINTS_RCV_L, RCV_CAMERA, POINT_DOWN, POINT_RCV_DOWN, RED_CAMERA, BLACK_CAMERA, PIECE_SIZE, \
    IO_QI, RCV_H_LAY
from src.cchessAI import cchess
from src.cchessAI.core.game import uci_to_coordinates, get_best_move_with_computer_play, execute_computer_move
from src.cchessYolo.detect_chess_box import calculate_4x4_collection_positions
from utils.calibrationManager import multi_camera_pixel_to_world, chess_to_world_position, get_area_center
from utils.tools import move_id2move_action


class ChessPlayFlowUtils():
    def __init__(self, parent):
        self.parent = parent
    # 棋盘
    def infer_human_move(self, old_positions, new_positions):
        """
        通过比较棋盘前后的变化推断人类的走法

        Args:
            old_positions: 移动前的棋盘状态
            new_positions: 移动后的棋盘状态

        Returns:
            str: UCI格式的移动字符串，如果无法推断则返回None
        """
        # 找到不同的位置
        diff_positions = []
        for row in range(10):
            for col in range(9):
                if old_positions[row][col] != new_positions[row][col]:
                    diff_positions.append((row, col, old_positions[row][col], new_positions[row][col]))

        # 分析差异以确定移动
        diff_count = len(diff_positions)

        if diff_count == 0:
            asyncio.run(self.parent.speak_cchess("没有识别到变化"))
            return None

        elif diff_count == 1:
            return self._handle_single_diff(diff_positions[0])

        elif diff_count == 2:
            return self._handle_double_diff(diff_positions, old_positions, new_positions)

        else:  # diff_count >= 3
            return self._handle_multiple_diff(diff_positions, old_positions, new_positions)

    def compare_chessboard_positions(self, old_positions, new_positions):
        """
        对比两个棋盘状态的差异

        Args:
            old_positions: 之前的棋盘状态
            new_positions: 当前识别的棋盘状态

        Returns:
            list: 差异列表，包含位置和变化详情
        """


        differences = []

        for row in range(10):
            for col in range(9):
                # 检查游戏状态，处理暂停和投降
                surrendered, paused = self.parent.check_game_state()
                if surrendered or self.parent.surrendered:
                    return []


                old_piece = old_positions[row][col]
                new_piece = new_positions[row][col]
                old_piece = self.parent.piece_map[old_piece] if old_piece in self.parent.piece_map else old_piece
                new_piece = self.parent.piece_map[new_piece] if new_piece in self.parent.piece_map else new_piece

                if old_piece != new_piece:
                    differences.append({
                        'row': row,
                        'col': col,
                        'from': old_piece,
                        'to': new_piece,
                        'type': self.get_difference_type(old_piece, new_piece)
                    })
        if differences:
            print( "棋盘没有正确布局:",differences)
            asyncio.run(self.parent.speak_cchess(f"棋盘没有正确布局"))
            if len(differences) <= 3:
                for diff in differences:
                    if diff['type'] == 'placed':
                        asyncio.run(self.parent.speak_cchess(f"第{diff['row']+1}行,第{diff['col']+1}列的{diff['to']}多余"))
                    elif diff['type'] == 'removed':
                        asyncio.run(self.parent.speak_cchess(f"第{diff['row']+1}行,第{diff['col']+1}列的{diff['from']}缺少"))
                    elif diff['type'] == 'replaced':
                        asyncio.run(self.parent.speak_cchess(f"第{diff['row']+1}行,第{diff['col']+1}列的{diff['from']}被替换为了{diff['to']}"))
        return differences

    def get_difference_type(self, old_piece, new_piece):
        """
        判断差异类型

        Args:
            old_piece: 原棋子
            new_piece: 新棋子

        Returns:
            str: 差异类型
        """
        if old_piece == '.' and new_piece != '.':
            return 'placed'  # 放置棋子
        elif old_piece != '.' and new_piece == '.':
            return 'removed'  # 移除棋子
        elif old_piece != '.' and new_piece != '.' and old_piece != new_piece:
            return 'replaced'  # 替换棋子
        else:
            return None

    def check_all_pieces_initial_position(self, tolerance=10):
        """
        检查初始状态下所有棋子是否在正确位置上

        Args:
            tolerance: 偏差容忍度(mm)

        Returns:
            bool: True表示所有棋子都在正确位置上，False表示有偏差
        """
        print("🔍 检查初始棋子位置...")
        pieces_with_deviation = []
        er_points = []

        # 遍历所有棋子位置
        for piece_key, (pixel_x, pixel_y) in self.parent.piece_pixel_positions.items():
            # 检查游戏状态，处理暂停和投降
            surrendered, paused = self.parent.check_game_state()
            if surrendered or self.parent.surrendered:
                return True  # 直接返回True以避免阻塞


            # 解析棋子位置
            row_idx = int(piece_key[0])
            row = 9 - row_idx
            col = int(piece_key[1])
            point_key = self.parent.chess_positions[row_idx][col]
            point_type = self.parent.piece_map[point_key] if point_key in self.parent.piece_map else point_key

            # 使用通用函数计算偏差
            deviation_data = self.parent.cMove._calculate_piece_deviation(row, col, pixel_x, pixel_y, tolerance)

            # 如果偏差超过容忍度，记录下来并报告
            if deviation_data['is_deviation_exceeded']:
                pieces_with_deviation.append({
                    'position': (row, col),
                    'world_position': deviation_data['world_position'],
                    'standard_position': deviation_data['standard_position'],
                    'deviation_x': deviation_data['deviation_x'],
                    'deviation_y': deviation_data['deviation_y'],
                    'distance': deviation_data['distance']
                })
                print(
                    f"⚠️ ({row_idx + 1},{col + 1})的{point_type}偏离标准位置X方向{abs(deviation_data['world_position'][0] - deviation_data['standard_position'][0]):.2f}mm，Y方向{abs(deviation_data['world_position'][1] - deviation_data['standard_position'][1]):.2f}mm，超过{tolerance}mm阈值")
                er_points.append([row_idx + 1, col + 1])

                if self.parent.args.use_api:
                    # 报告偏移棋子信息
                    self.parent.report_piece_deviation(
                        row_idx,
                        col,
                        deviation_data['deviation_x'],
                        deviation_data['deviation_y'],
                        deviation_data['distance']
                    )

        # 如果有偏差的棋子，报告详细信息
        if pieces_with_deviation:
            print(f"❌ 发现{len(pieces_with_deviation)}个棋子位置不正确")
            asyncio.run(self.parent.speak_cchess(f"发现{len(pieces_with_deviation)}个棋子偏离标准位置"))
            for i in range(len(pieces_with_deviation)):
                point_type = self.parent.piece_map[
                    self.parent.chess_positions[er_points[i][0] - 1][er_points[i][1] - 1]]
                asyncio.run(self.parent.speak_cchess(f"第{er_points[i][0]}行,第{er_points[i][1]}列的{point_type}"))
                if i > 3:
                    break
            return False
        else:
            print("✅ 所有棋子都在正确位置上")
            # asyncio.run(self.parent.speak_cchess("所有棋子位置正确"))
            return True

    # 算法
    def is_in_check(self, board, side):
        """
        检查指定方是否被将军

        Args:
            board: 棋盘对象
            side: 检查的方('red'或'black')

        Returns:
            bool: 是否被将军
        """
        return board.is_check()
    def is_king_captured_by_move(self, move_uci, positions):
        """
        通过检查移动后的位置是否为k或K来判断是否吃掉了将军

        Args:
            move_uci: 移动的UCI表示 (例如: "a1a2")
            positions: 当前棋盘位置

        Returns:
            tuple: (is_captured, king_side) 如果吃掉将军返回(True, 'red'/'black')，否则返回(False, None)
        """
        if not move_uci or len(move_uci) != 4:
            return False, None

        # 解析目标位置
        to_col = ord(move_uci[2]) - ord('a')  # 0-8
        to_row = int(move_uci[3])             # 0-9

        # 转换为数组索引
        to_row_idx = 9 - to_row  # 转换为数组行索引

        # 检查目标位置的棋子
        if 0 <= to_row_idx < 10 and 0 <= to_col < 9:
            target_piece = positions[to_row_idx][to_col]
            # 检查是否为对方的将/帅
            if target_piece == 'k':
                return True, 'black'  # 吃掉了黑方将
            elif target_piece == 'K':
                return True, 'red'    # 吃掉了红方帅

        return False, None
    def calculate_next_move(self):
        """
        计算下一步棋，确保移动在合法范围内
        """
        print("🧠 AI计算下一步...")

        # 获取所有合法移动
        legal_moves = list(self.parent.board.legal_moves)
        print(f"_legal_mo_covesunt: {len(legal_moves)}")

        if not legal_moves:
            print("❌ 没有合法的移动")
            asyncio.run(self.parent.speak_cchess("没有合法的移动，游戏结束"))
            return None

        max_attempts = 5  # 最大尝试次数
        move_uci = None

        # if self.parent.args.use_ag and len(self.parent.move_history)%8!=7:
        if self.parent.args.use_ag:
            for attempt in range(max_attempts):
                try:
                    from_x, from_y, to_x, to_y = uci_to_coordinates(self.parent.move_uci)

                    # 将耗时的计算放到独立线程中执行
                    def computer_play_task():
                        return get_best_move_with_computer_play(self.parent.maingame, self.parent.board, from_x, from_y, to_x, to_y)

                    # 使用事件来同步等待计算结果
                    import threading
                    result_container = [None]  # 用于在线程间传递结果
                    calculation_event = threading.Event()

                    def run_calculation():
                        result_container[0] = computer_play_task()
                        calculation_event.set()

                    calculation_thread = threading.Thread(target=run_calculation, daemon=True)
                    calculation_thread.start()

                    # 等待计算完成，同时定期检查游戏状态
                    while not calculation_event.is_set():
                        # 检查游戏是否已结束或暂停
                        surrendered, paused = self.parent.check_game_state()
                        if surrendered or self.parent.surrendered:
                            return None

                        time.sleep(0.01)  # 短暂等待

                    move_uci = result_container[0]

                    if move_uci:
                        # 检查计算出的移动是否在合法移动列表中
                        if move_uci in [move.uci() for move in legal_moves]:
                            print(f"✅ AI决定走: {move_uci} (合法移动)")
                            break
                        else:
                            move_uci = None
                            print(f"⚠️ 第{attempt + 1}次尝试计算出的移动 {move_uci} 不在合法移动列表中")
                    else:
                        print(f"⚠️ 第{attempt + 1}次尝试未获得有效移动，重新计算...")
                        # 等待时也检查游戏状态
                        for _ in range(100):  # 1秒分成100个0.01秒
                            surrendered, paused = self.parent.check_game_state()
                            if surrendered or self.parent.surrendered:
                                return None

                            time.sleep(0.01)

                except Exception as e:
                    print(f"⚠️ 第{attempt + 1}次尝试出错: {e}")
                    if attempt < max_attempts - 1:
                        # 出错后等待再重试，同时检查游戏状态
                        for _ in range(100):  # 1秒分成100个0.01秒
                            surrendered, paused = self.parent.check_game_state()
                            if surrendered or self.parent.surrendered:
                                return None

                            time.sleep(0.01)
                    continue

        # 如果经过多次尝试仍未获得合法移动，则从合法移动列表中选择
        if not move_uci and legal_moves:
            try:
                asyncio.run(self.parent.speak_cchess("AI切换为复杂运算，请稍等"))

                # 将耗时的AI计算放到独立线程中执行
                def ai_calculation_task():
                    try:
                        move_id = self.parent.mcts_player.get_action(self.parent.board)
                        return move_id2move_action[move_id]
                    except Exception as e:
                        print(f"AI计算出错: {e}")
                        return None

                # 使用事件来同步等待AI计算结果
                import threading
                result_container = [None]  # 用于在线程间传递结果
                calculation_event = threading.Event()

                def run_calculation():
                    result_container[0] = ai_calculation_task()
                    calculation_event.set()

                calculation_thread = threading.Thread(target=run_calculation, daemon=True)
                calculation_thread.start()

                # 等待AI计算完成，同时定期检查游戏状态
                while not calculation_event.is_set():
                    # 检查游戏是否已结束或暂停
                    surrendered, paused = self.parent.check_game_state()
                    if surrendered or self.parent.surrendered:
                        return None

                    time.sleep(0.1)  # 短暂等待

                move_uci = result_container[0]

                if move_uci:
                    move_mg = self.uci_to_mg_coords(move_uci)
                    # 执行移动到MainGame并保存历史信息
                    self.parent.maingame.mgInit.move_to(move_mg)

            except Exception as e:
                selected_move = legal_moves[0]
                move_uci = selected_move.uci()
                print(f"🔄 最终选择第一个合法移动: {move_uci}")

        if not move_uci:
            print("❌ AI无法计算出有效移动")
            asyncio.run(self.parent.speak_cchess("无法计算出有效移动，机器人投降"))
            self.parent.gama_over('player')
            print(self.parent.board.unicode())
            return None

        execute_computer_move(self.parent.maingame,self.parent.board,move_uci)
        return move_uci

    def find_check_move(self):
        """
        优先寻找能吃掉对方将军的移动
        """
        print("🧠 寻找能吃掉对方将军的移动...")

        # 获取所有合法移动
        legal_moves = list(self.parent.board.legal_moves)

        # 首先寻找能直接吃掉对方将军的移动
        for move in legal_moves:
            # 检查这个移动是否是吃子移动
            if self.parent.board.is_capture(move):
                # 获取目标位置的棋子
                target_piece = self.parent.board.piece_at(move.to_square)
                # 检查目标位置是否是对方的将/帅
                if target_piece and target_piece.piece_type == cchess.KING:
                    move_uci = move.uci()
                    print(f"✅ 找到能吃掉对方将军的移动: {move_uci}")
                    return move_uci

        # 如果没有能直接吃掉将军的移动，则使用原来的AI计算
        print("⚠️ 没有找到能直接吃掉将军的移动，使用默认AI计算...")
        from_x, from_y, to_x, to_y = uci_to_coordinates(self.parent.move_uci)
        move_uci = get_best_move_with_computer_play(self.parent.maingame, self.parent.board, from_x, from_y, to_x, to_y)

        print(f"✅ AI决定走: {move_uci}")
        return move_uci

    # 棋谱格式转换
    def _create_uci_move(self, from_row, from_col, to_row, to_col, old_positions):
        """
        创建UCI格式的移动字符串

        Args:
            from_row: 起点行
            from_col: 起点列
            to_row: 终点行
            to_col: 终点列
            old_positions: 移动前的棋盘状态

        Returns:
            str: UCI格式的移动字符串
        """
        from_row_char = chr(ord('a') + from_col)
        to_row_char = chr(ord('a') + to_col)
        move_uci = f"{from_row_char}{9-from_row}{to_row_char}{9-to_row}"

        if self.parent.args.use_api:
            # 报告人类移动
            chinese_notation = self.uci_to_chinese_notation(move_uci, old_positions)
            self.parent.report_move("human", move_uci, chinese_notation)

        return move_uci

    def uci_to_chinese_notation(self, move_uci, previous_positions=None):
        """
        将UCI格式的移动转换为中文象棋记谱法

        输入坐标系：x轴从左到右为a-i，y轴从下到上为0-9
        输出：标准中文象棋记谱法，如 "马八进七"

        Args:
            move_uci: UCI格式移动，如 "b0c2"
            previous_positions: 当前棋盘状态，用于确定棋子类型

        Returns:
            str: 中文象棋记谱法，如 "马八进七"
        """
        if not move_uci or len(move_uci) != 4:
            return move_uci

        # 解析UCI格式 (x轴从左到右为a-i，y轴从下到上为0-9)
        from_col = ord(move_uci[0]) - ord('a')  # 0-8 (a-i)
        from_row = int(move_uci[1])             # 0-9 (0-9 从下到上)
        to_col = ord(move_uci[2]) - ord('a')    # 0-8 (a-i)
        to_row = int(move_uci[3])               # 0-9 (0-9 从下到上)

        # 获取棋子类型
        piece_type = '?'
        piece_char = '?'
        if previous_positions:
            # 将行列转换为数组索引 (棋盘数组是10x9)
            to_row_idx = 9 - to_row  # 转换为数组行索引 (0-9 从上到下)
            if 0 <= to_row_idx < 10 and 0 <= from_col < 9:
                piece_char = previous_positions[to_row_idx][to_col]
                if piece_char in self.parent.piece_map:
                    piece_type = self.parent.piece_map[piece_char]

        # 判断是红方还是黑方的棋子（根据棋子是否为大写）
        is_red_piece = piece_char.isupper() if 'piece_char' in locals() else True

        # 列名映射
        # 红方视角：从右到左为九到一
        red_col_names = ['九', '八', '七', '六', '五', '四', '三', '二', '一']
        # 黑方视角：从左到右为1到9
        black_col_names = ['一', '二', '三', '四', '五', '六', '七', '八', '九']

        # 根据棋子方选择列名映射
        col_names = red_col_names if is_red_piece else black_col_names

        # 计算移动方向和距离
        row_diff = to_row - from_row  # 正数表示向上，负数表示向下
        col_diff = to_col - from_col  # 正数表示向右，负数表示向左

        # 确定方向描述（需要根据棋子方调整方向判断）
        if is_red_piece:
            # 红方视角：数值增加是向上（向对方阵地），数值减少是向下（向自己阵地）
            forward = row_diff > 0  # 向上为前进
        else:
            # 黑方视角：数值增加是向下（向对方阵地），数值减少是向上（向自己阵地）
            forward = row_diff < 0  # 向下为前进

        # 确定方向描述
        if from_col == to_col:  # 同列移动（进/退）
            if (is_red_piece and row_diff > 0) or (not is_red_piece and row_diff < 0):  # 向对方阵地移动
                direction = '进'
            else:  # 向自己阵地移动
                direction = '退'
            # 对于马、象、士等走斜线的棋子，同行同列移动实际是斜向移动
            if piece_type in ['马', '象', '相', '士', '仕']:
                distance = col_names[to_col]
            else:
                distance = int(abs(row_diff)) if piece_type not in ['马', '象', '相', '士', '仕'] else col_names[to_col]
                distance = black_col_names[distance-1]
        elif from_row == to_row:  # 同行移动（平）
            direction = '平'
            distance = col_names[to_col]
        else:  # 斜向移动（马、象等）
            if (is_red_piece and row_diff > 0) or (not is_red_piece and row_diff < 0):  # 向对方阵地移动
                direction = '进'
            else:  # 向自己阵地移动
                direction = '退'
            distance = col_names[to_col]

        # 特殊处理马、象、士的移动表示
        if piece_type in ['马', '象', '相', '士', '仕']:
            # 这些棋子的移动距离用目标位置的列名表示
            distance = col_names[to_col]

        return f"{piece_type}{col_names[from_col]}{direction}{distance}"

    def uci_to_mg_coords(self, uci):
        """
        将UCI格式的移动转换为MainGame坐标

        Args:
            uci: UCI格式的移动

        Returns:
            tuple: 转换后的MainGame坐标 (from_x, from_y, to_x, to_y)
        """
        from_col = ord(uci[0]) - ord('a')
        from_row = int(uci[1])
        to_col = ord(uci[2]) - ord('a')
        to_row = int(uci[3])

        # 转换为MainGame坐标系 (镜像处理)
        mg_from_x = 8 - from_col
        mg_to_x = 8 - to_col
        mg_from_y = 9 - from_row
        mg_to_y = 9 - to_row
        # 创建移动步骤
        from src.cchessAG import my_chess
        move_mg = my_chess.step(mg_from_x, mg_from_y, mg_to_x, mg_to_y)
        print(f"已创建移动步骤: {move_mg}")
        return move_mg

    def parse_chinese_notation(self, chinese_move):
        """
        解析中文象棋记谱法，如"炮七平四"转UCI格式

        Args:
            chinese_move: 中文象棋记谱法字符串

        Returns:
            str: UCI格式的移动字符串，如果无法解析则返回None
        """
        chinese_move = chinese_move.replace(" ", "").replace("\t", "").strip()
        if not chinese_move:
            return None

        if chinese_move[0] == '四':
            chinese_move = '士'+chinese_move[1:]

        # 中文记谱法到棋子字符的映射
        piece_map_chinese = {
            '车': 'r', '馬': 'n', '象': 'b', '士': 'a', '將': 'k', '炮': 'c', '卒': 'p',  # 黑方
            '車': 'R', '马': 'N', '相': 'B', '仕': 'A', '帅': 'K', '砲': 'C', '兵': 'P'   # 红方
        }

        # 棋子类型分类
        straight_moving_pieces = ['车', '車', '炮', '砲', '兵', '卒']  # 直线移动棋子
        diagonal_moving_pieces = ['马', '馬', '象', '相', '士', '仕']   # 斜线移动棋子

        # 列名映射
        red_col_names = ['九', '八', '七', '六', '五', '四', '三', '二', '一']
        black_col_names = ['一', '二', '三', '四', '五', '六', '七', '八', '九']

        # 方向映射
        directions = ['进', '退', '平']

        # 方位词映射
        position_words = ['前', '后', '中']

        # 判断是否包含方位词
        has_position_word = chinese_move[0] in position_words

        if has_position_word:
            if len(chinese_move) != 4:
                return None
            position_word = chinese_move[0]  # 前/后/中
            piece_char = chinese_move[1]     # 棋子类型
            start_col_char = chinese_move[2] # 起始列
            direction_char = chinese_move[3] # 方向
        else:
            if len(chinese_move) != 4:
                return None
            position_word = None
            piece_char = chinese_move[0]     # 棋子类型
            start_col_char = chinese_move[1] # 起始列
            direction_char = chinese_move[2] # 方向

        if piece_char not in piece_map_chinese.keys():
            return None

        piece_symbol = piece_map_chinese[piece_char]
        piece_type = piece_char  # 棋子类型用于判断处理方式

        # 确定当前行棋方的列名映射
        if self.parent.side == 'red':  # 红方
            col_names = red_col_names
            piece_symbol = piece_symbol.upper()
        else:  # 黑方
            col_names = black_col_names
            piece_symbol = piece_symbol.lower()

        if start_col_char not in col_names:
            return None

        start_col = col_names.index(start_col_char)

        # 查找棋盘上符合条件的棋子
        candidate_pieces = []

        # 遍历棋盘找到所有该类型的棋子
        for row in range(10):
            for col in range(9):
                if self.parent.previous_positions[row][col] == piece_symbol:
                    # 检查是否在正确的列上
                    if self.parent.side == 'red':
                        display_col = red_col_names[col]
                    else:
                        display_col = black_col_names[col]

                    if display_col == start_col_char:
                        candidate_pieces.append((row, col))

        if not candidate_pieces:
            return None

        # 根据方位词筛选棋子
        selected_piece = None
        if has_position_word and len(candidate_pieces) > 1:
            # 按照行号排序
            if self.parent.side == 'red':
                # 红方：行号越大越靠前
                candidate_pieces.sort(key=lambda x: x[0], reverse=True)
            else:
                # 黑方：行号越小越靠前
                candidate_pieces.sort(key=lambda x: x[0])

            if position_word == '前':
                selected_piece = candidate_pieces[0]
            elif position_word == '后':
                selected_piece = candidate_pieces[-1]
            elif position_word == '中':
                if len(candidate_pieces) >= 3:
                    selected_piece = candidate_pieces[1]  # 选择中间的棋子
                else:
                    return None  # 没有足够的棋子来定义"中"
        elif len(candidate_pieces) == 1:
            selected_piece = candidate_pieces[0]
        else:
            # 如果没有方位词但有多个候选棋子，需要进一步筛选
            if self.parent.side == 'red':
                # 红方，选择行数大的（更靠近对方的）
                selected_piece = max(candidate_pieces, key=lambda x: x[0])
            else:
                # 黑方，选择行数小的（更靠近对方的）
                selected_piece = min(candidate_pieces, key=lambda x: x[0])

        if not selected_piece:
            return None

        from_row, from_col = selected_piece

        # 提取目标位置字符（根据方向字符的位置确定）
        if has_position_word:
            target_char_index = 3  # "前炮进四" 中的 "四"
        else:
            target_char_index = 3  # "炮七平八" 中的 "八"

        if len(chinese_move) <= target_char_index:
            return None

        target_char = chinese_move[target_char_index]

        # 计算目标位置
        if direction_char == '平':  # 平移
            to_row = from_row
            if target_char in col_names:
                to_col = col_names.index(target_char)
            else:
                return None

        elif direction_char == '进':  # 前进
            # 根据棋子类型判断处理方式
            if piece_type in straight_moving_pieces:
                # 直线移动棋子优先判断步数，其次为列名
                try:
                    steps = black_col_names.index(target_char) + 1 # 数字表示步数
                    if self.parent.side == 'red':
                        to_row = from_row - steps
                    else:
                        to_row = from_row + steps
                    to_col = from_col  # 直线移动列不变
                except ValueError:
                    # 如果不是数字，则是列名
                    if target_char in col_names:
                        to_col = col_names.index(target_char)
                        to_row = from_row  # 平移到目标列
                    else:
                        return None
            else:  # 斜线移动棋子只能是列名
                if target_char in col_names:
                    to_col = col_names.index(target_char)
                    col_step = abs(to_col - from_col)
                    # 对于斜线移动棋子，需要根据棋子类型计算行位置
                    if piece_type in ['马', '馬']:
                        if self.parent.side == 'red':
                            to_row = from_row - 2 if col_step==1 else from_row - 1
                        else:
                            to_row = from_row + 2 if col_step==1 else from_row + 1
                    elif piece_type in ['象', '相']:
                        # 象走"田"字
                        if self.parent.side == 'red':
                            to_row = from_row - 2
                        else:
                            to_row = from_row + 2
                    elif piece_type in ['士', '仕']:
                        # 士走斜线
                        if self.parent.side == 'red':
                            to_row = from_row - 1
                        else:
                            to_row = from_row + 1
                else:
                    return None

        elif direction_char == '退':  # 后退
            # 根据棋子类型判断处理方式
            if piece_type in straight_moving_pieces:
                # 直线移动棋子优先判断步数，其次为列名
                try:
                    steps = black_col_names.index(target_char) + 1
                    if self.parent.side == 'red':
                        to_row = from_row + steps  # 红方后退是行数增加
                    else:
                        to_row = from_row - steps  # 黑方后退是行数减少
                    to_col = from_col  # 直线移动列不变
                except ValueError:
                    # 如果不是数字，则是列名
                    if target_char in col_names:
                        to_col = col_names.index(target_char)
                        to_row = from_row  # 平移到目标列
                    else:
                        return None
            else:  # 斜线移动棋子只能是列名
                if target_char in col_names:
                    to_col = col_names.index(target_char)
                    col_step = abs(to_col - from_col)
                    # 对于斜线移动棋子，需要根据棋子类型计算行位置
                    if piece_type in ['马', '馬']:
                        # 马走"日"字
                        if self.parent.side == 'red':
                            to_row = from_row + 2 if col_step==1 else from_row + 1
                        else:
                            to_row = from_row - 2 if col_step==1 else from_row - 1
                    elif piece_type in ['象', '相']:
                        # 象走"田"字
                        if self.parent.side == 'red':
                            to_row = from_row + 2
                        else:
                            to_row = from_row - 2
                    elif piece_type in ['士', '仕']:
                        # 士走斜线
                        if self.parent.side == 'red':
                            to_row = from_row + 1
                        else:
                            to_row = from_row - 1
                else:
                    return None
        else:
            return None  # 无效的方向字符

        # 检查目标位置是否有效
        if to_row < 0 or to_row > 9 or to_col < 0 or to_col > 8:
            return None

        # 转换为UCI格式 (数组索引转棋盘坐标)
        from_col_uci = chr(ord('a') + from_col)
        from_row_uci = 9 - from_row
        to_col_uci = chr(ord('a') + to_col)
        to_row_uci = 9 - to_row

        return f"{from_col_uci}{from_row_uci}{to_col_uci}{to_row_uci}"


    # 人类移动
    def _handle_single_diff(self, diff_position):
        """
        处理只有一个位置发生变化的情况

        Args:
            diff_position: 差异位置信息 (row, col, old_piece, new_piece)

        Returns:
            None: 无法构成有效移动
        """
        row, col, old_piece, new_piece = diff_position

        # 将行号转换为棋盘表示法 (0-9 -> 0-9)
        display_row = 9 - row
        # 将列号转换为字母表示法 (0-8 -> a-i)
        display_col = chr(ord('a') + col)

        # 生成中文记谱法位置描述
        col_names = ['一', '二', '三', '四', '五', '六', '七', '八', '九']
        row_names = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
        col_name = col_names[col]
        row_name = row_names[row]

        print(f"🔍 检测到1个位置发生变化:")
        print(f"   位置{display_col}{display_row}: '{old_piece}' -> '{new_piece}'")

        # 语音播报变化信息
        speech_text = f"只检测第{row_name}行第{col_name}列发生变化，从'{old_piece}'变为'{new_piece}'。"
        asyncio.run(self.parent.speak_cchess(speech_text))

        # 无法构成有效移动，返回None
        return None

    def _handle_double_diff(self, diff_positions, old_positions, new_positions):
        """
        处理两个位置发生变化的情况（标准移动）

        Args:
            diff_positions: 差异位置列表
            old_positions: 移动前的棋盘状态
            new_positions: 移动后的棋盘状态

        Returns:
            str: UCI格式的移动字符串，如果无法推断则返回None
        """
        pos1, pos2 = diff_positions[0], diff_positions[1]

        # 判断哪个位置是起点，哪个是终点
        # 情况1: pos1是起点(有棋子离开)，pos2是终点(空位被占据或被吃子)
        if pos1[2] != '.' and pos2[2] == '.':
            from_row, from_col = pos1[0], pos1[1]
            to_row, to_col = pos2[0], pos2[1]
        # 情况2: pos2是起点(有棋子离开)，pos1是终点(空位被占据或被吃子)
        elif pos1[2] == '.' and pos2[2] != '.':
            from_row, from_col = pos2[0], pos2[1]
            to_row, to_col = pos1[0], pos1[1]
        else:
            # 其他情况，可能有吃子
            # 简化处理：假定非空位置是目标位置
            if pos1[3] != '.' and pos2[3] != '.':
                # 两个位置都有棋子，无法判断
                return None
            elif pos1[3] != '.':
                # pos1是终点
                return self._find_move_start_position(pos1, old_positions, new_positions)
            else:
                # pos2是终点
                return self._find_move_start_position(pos2, old_positions, new_positions)

        # 转换为UCI格式
        return self._create_uci_move(from_row, from_col, to_row, to_col, old_positions)

    def _find_move_start_position(self, target_pos, old_positions, new_positions):
        """
        根据目标位置查找移动的起始位置

        Args:
            target_pos: 目标位置信息
            old_positions: 移动前的棋盘状态
            new_positions: 移动后的棋盘状态

        Returns:
            str: UCI格式的移动字符串，如果无法推断则返回None
        """
        target_piece = new_positions[target_pos[0]][target_pos[1]]
        to_row, to_col = target_pos[0], target_pos[1]

        from_pos = None
        for r in range(10):
            for c in range(9):
                if old_positions[r][c] == target_piece and new_positions[r][c] == '.':
                    from_pos = (r, c)
                    break
            if from_pos:
                break

        if from_pos:
            from_row, from_col = from_pos
            return self._create_uci_move(from_row, from_col, to_row, to_col, old_positions)
        else:
            return None

    def _handle_multiple_diff(self, diff_positions, old_positions, new_positions):
        """
        处理三个或更多位置发生变化的情况

        Args:
            diff_positions: 差异位置列表
            old_positions: 移动前的棋盘状态
            new_positions: 移动后的棋盘状态

        Returns:
            str: UCI格式的移动字符串，如果无法推断则返回None
        """
        diff_count = len(diff_positions)
        print(f"🔍 检测到{diff_count}个位置发生变化:")

        # if diff_count == 3:
        #     return self._handle_triple_diff(diff_positions, old_positions)
        # else:
        asyncio.run(self.parent.speak_cchess(f"有{diff_count}个位置变化，请检查棋盘状态"))
        return self._handle_complex_diff(diff_positions)

    def _handle_triple_diff(self, diff_positions, old_positions):
        """
        处理三个位置发生变化的情况

        Args:
            diff_positions: 差异位置列表
            old_positions: 移动前的棋盘状态

        Returns:
            str: UCI格式的移动字符串，如果无法推断则返回None
        """
        # 分析三个位置的变化，尝试找出合理的移动组合
        # 查找移动的起点和终点
        from_pos = None
        to_pos = None
        changed_pos = None

        # 寻找典型的移动模式：一个棋子离开(.), 一个棋子到达(新棋子)
        for pos in diff_positions:
            row, col, old_piece, new_piece = pos
            if old_piece != '.' and new_piece == '.':  # 棋子离开的位置
                from_pos = pos
            elif old_piece == '.' and new_piece != '.':  # 棋子到达的位置
                to_pos = pos
            else:  # 其他变化(如棋子类型改变)
                changed_pos = pos

        if changed_pos and changed_pos[3] == '.':
            changed_row, changed_col, old_changed_piece, new_changed_piece = changed_pos

            # 生成中文坐标
            row_names = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
            col_names = ['一', '二', '三', '四', '五', '六', '七', '八', '九']
            changed_chinese_col = col_names[changed_col]
            changed_chinese_row = row_names[changed_row]

            print(f"⚠️ 第3个位置变为'.', 用户可能违规")
            asyncio.run(self.parent.speak_cchess(f"第{changed_chinese_row}行,第{changed_chinese_col}列的{self.parent.piece_map.get(old_changed_piece, old_changed_piece)}棋子不见了"))
            return None  # 返回None表示无法推断有效移动

        # 如果找到了明确的起点和终点
        if from_pos and to_pos:
            from_row, from_col, old_from_piece, new_from_piece = from_pos
            to_row, to_col, old_to_piece, new_to_piece = to_pos

            # 将行列转换为显示坐标
            from_display_row = 9 - from_row
            from_display_col = chr(ord('a') + from_col)
            to_display_row = 9 - to_row
            to_display_col = chr(ord('a') + to_col)

            # 生成中文坐标
            row_names = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
            col_names = ['一', '二', '三', '四', '五', '六', '七', '八', '九']
            from_chinese_col = col_names[from_col]
            from_chinese_row = str(from_display_row)
            to_chinese_col = col_names[to_col]
            to_chinese_row = str(to_display_row)

            print(f"   位置{from_display_col}{from_display_row}: '{old_from_piece}' -> '{new_from_piece}'")
            print(f"   位置{to_display_col}{to_display_row}: '{old_to_piece}' -> '{new_to_piece}'")

            # 如果还有第三个位置变化，可能是识别错误
            if changed_pos:
                changed_row, changed_col, old_changed_piece, new_changed_piece = changed_pos
                changed_display_row = 9 - changed_row + 1
                changed_display_col = chr(ord('a') + changed_col)
                print(f"   位置可能误识别{changed_display_col}{changed_display_row}: '{old_changed_piece}' -> '{new_changed_piece}'")

            # 正常的移动情况
            # speech_text = f"检测到从{from_chinese_col}{from_chinese_row}移动到{to_chinese_col}{to_chinese_row}"
            # asyncio.run(self.parent.speak_cchess(speech_text))

            # 构造UCI移动字符串
            move_uci = f"{from_display_col}{from_display_row}{to_display_col}{to_display_row}"

            if self.parent.args.use_api:
                # 报告人类移动
                chinese_notation = self.uci_to_chinese_notation(move_uci, old_positions)
                self.parent.report_move("human", move_uci, chinese_notation)

            return move_uci
        else:
            return self._handle_complex_diff(diff_positions)

    def _handle_complex_diff(self, diff_positions):
        """
        处理复杂情况（超过3个位置发生变化）

        Args:
            diff_positions: 差异位置列表

        Returns:
            None: 无法准确推断移动
        """

        for i, diff in enumerate(diff_positions):
            row, col, old_piece, new_piece = diff
            # 将行号转换为棋盘表示法 (0-9 -> 0-9)
            display_row = 9 - row
            # 将列号转换为字母表示法 (0-8 -> a-i)
            display_col = chr(ord('a') + col)

            print(f"位置{display_col}{display_row}: '{old_piece}' -> '{new_piece}'")
            if old_piece in self.parent.piece_map and new_piece in self.parent.piece_map:
                old_piece = self.parent.piece_map[old_piece]
                new_piece = self.parent.piece_map[new_piece]
                speech_text = f"第{10-display_row}行第{col+1}列的{old_piece}检测成了{new_piece}"
                asyncio.run(self.parent.speak_cchess(speech_text))
        # 无法准确推断移动
        return None