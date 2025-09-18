# 工具函数模块
import time
import copy
from src.cchessAI import cchess
from src.cchessAI.core.game import uci_to_coordinates, get_best_move_with_computer_play, \
    execute_computer_move
from src.cchess_runner.chess_play_flow_branch import ChessPlayFlowBranch
from utils.calibrationManager import chess_to_world_position
from utils.tools import move_id2move_action


class ChessPlayFlowUtils(ChessPlayFlowBranch):
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
            self.speak("没有识别到变化")
            return None

        elif diff_count == 1:
            return self._handle_single_diff(diff_positions[0])

        elif diff_count == 2:
            return self._handle_double_diff(diff_positions, old_positions, new_positions)

        else:  # diff_count >= 3
            return self._handle_multiple_diff(diff_positions, old_positions, new_positions)

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
        self.speak(speech_text)

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

        if diff_count == 3:
            return self._handle_triple_diff(diff_positions, old_positions)
        else:
            self.speak(f"有{diff_count}个位置变化，请检查棋盘状态")
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
            self.speak(
                f"第{changed_chinese_row}行,第{changed_chinese_col}列的{self.piece_map.get(old_changed_piece, old_changed_piece)}棋子不见了")
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
            # self.speak(speech_text)

            # 构造UCI移动字符串
            move_uci = f"{from_display_col}{from_display_row}{to_display_col}{to_display_row}"

            if self.args.use_api:
                # 报告人类移动
                chinese_notation = self.uci_to_chinese_notation(move_uci, old_positions)
                self.report_move("human", move_uci, chinese_notation)

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
        speech_text = f"检测到{len(diff_positions)}个位置发生变化："

        for i, diff in enumerate(diff_positions):
            row, col, old_piece, new_piece = diff
            # 将行号转换为棋盘表示法 (0-9 -> 0-9)
            display_row = 9 - row
            # 将列号转换为字母表示法 (0-8 -> a-i)
            display_col = chr(ord('a') + col)

            print(f"   位置{display_col}{display_row}: '{old_piece}' -> '{new_piece}'")
            speech_text += (f"位"
                            f"置{display_col}{display_row}从'{old_piece}'变为'{new_piece}'。")

        # 无法准确推断移动
        return None

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

        if self.args.use_api:
            # 报告人类移动
            chinese_notation = self.uci_to_chinese_notation(move_uci, old_positions)
            self.report_move("human", move_uci, chinese_notation)

        return move_uci

    def update_chess_positions_after_move(self, move_uci):
        """
        根据移动UCI更新chess_positions状态
        """
        # 解析移动
        from_col= ord(move_uci[0]) - ord('a')
        from_row= int(move_uci[1])
        to_col=  ord(move_uci[2]) - ord('a')
        to_row= int(move_uci[3])

        # 将行列转换为数组索引 (棋盘坐标到数组索引)
        from_row_idx = 9 - from_row
        from_col_idx = from_col
        to_row_idx = 9 - to_row
        to_col_idx = to_col

        # 移动棋子
        piece = self.previous_positions[from_row_idx][from_col_idx]
        self.previous_positions[to_row_idx][to_col_idx] = piece
        self.previous_positions[from_row_idx][from_col_idx] = '.'

    def uci_to_chinese_notation(self, move_uci, chess_positions=None):
        """
        将UCI格式的移动转换为中文象棋记谱法

        输入坐标系：x轴从左到右为a-i，y轴从下到上为0-9
        输出：标准中文象棋记谱法，如 "马八进七"

        Args:
            move_uci: UCI格式移动，如 "b0c2"
            chess_positions: 当前棋盘状态，用于确定棋子类型

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
        if chess_positions:
            # 将行列转换为数组索引 (棋盘数组是10x9)
            to_row_idx = 9 - to_row  # 转换为数组行索引 (0-9 从上到下)
            if 0 <= to_row_idx < 10 and 0 <= from_col < 9:
                piece_char = chess_positions[to_row_idx][to_col]
                if piece_char in self.piece_map:
                    piece_type = self.piece_map[piece_char]

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

    def unicode_to_chess_positions(self, unicode_board):
        """
        将unicode棋盘表示转换为chess_positions格式

        Args:
            unicode_board: self.board.unicode()的输出

        Returns:
            list: 10x9的二维数组，表示棋盘状态
        """
        # 初始化空棋盘
        chess_positions = [['.' for _ in range(9)] for _ in range(10)]

        # 棋子映射字典（从显示字符到内部表示）
        unicode_piece_map = {
            '车': 'r', '馬': 'n', '象': 'b', '士': 'a', '將': 'k', '炮': 'c', '卒': 'p',  # 黑方
            '車': 'R', '马': 'N', '相': 'B', '仕': 'A', '帅': 'K', '砲': 'C', '兵': 'P'   # 红方
        }

        # 按行解析unicode棋盘
        lines = unicode_board.strip().split('\n')

        # 跳过第一行和最后一行（坐标标记），处理中间10行
        for i in range(1, 11):
            line = lines[i].strip()
            # 跳过行号和最后的行号
            row_content = line[2:-1]  # 去掉行号和最后的行号

            # 解析每一列
            for j in range(9):
                # 检查索引是否在有效范围内
                char_index = j * 2
                if char_index < len(row_content):
                    char = row_content[char_index]  # 每个棋子字符之间有一个空格
                    if char in unicode_piece_map:
                        # 转换为数组坐标系 (第0行对应棋盘第9行)
                        chess_positions[10-i][j] = unicode_piece_map[char]
                # '.' 保持不变

        return chess_positions

    def calculate_next_move(self):
        """
        计算下一步棋，确保移动在合法范围内
        """
        print("🧠 AI计算下一步...")

        # 获取所有合法移动
        legal_moves = list(self.board.legal_moves)
        print(f"_legal_mo_covesunt: {len(legal_moves)}")

        if not legal_moves:
            print("❌ 没有合法的移动")
            self.speak("没有合法的移动，游戏结束")
            return None

        max_attempts = 5  # 最大尝试次数
        move_uci = None
        selected_move = None

        for attempt in range(max_attempts):
            try:
                # 使用MCTS计算下一步
                # move_id = self.mcts_player.get_action(self.board)
                # move_uci = move_id2move_action[move_id]
                from_x, from_y, to_x, to_y = uci_to_coordinates(self.move_uci)
                move_uci = get_best_move_with_computer_play(self.maingame, self.board, from_x, from_y, to_x, to_y)

                if move_uci:
                    # 检查计算出的移动是否在合法移动列表中
                    calculated_move = cchess.Move.from_uci(move_uci)
                    if move_uci in [move.uci() for move in legal_moves]:
                        selected_move = calculated_move
                        print(f"✅ AI决定走: {move_uci} (合法移动)")
                        break
                    else:
                        print(f"⚠️ 第{attempt + 1}次尝试计算出的移动 {move_uci} 不在合法移动列表中")
                else:
                    print(f"⚠️ 第{attempt + 1}次尝试未获得有效移动，重新计算...")
                    time.sleep(1)  # 短暂等待后重试

            except Exception as e:
                print(f"⚠️ 第{attempt + 1}次尝试出错: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(1)  # 出错后等待再重试
                continue

        # 如果经过多次尝试仍未获得合法移动，则从合法移动列表中选择
        if not selected_move and legal_moves:
            try:
                self.speak("AI切换为复杂运算，请稍等")
                move_id = self.mcts_player.get_action(self.board)
                move_uci = move_id2move_action[move_id]
            except Exception as e:
                selected_move = legal_moves[0]
                move_uci = selected_move.uci()
                print(f"🔄 最终选择第一个合法移动: {move_uci}")

        if not selected_move:
            print("❌ AI无法计算出有效移动")
            self.speak("无法计算出有效移动，机器人投降")
            self.gama_over('player')
            print(self.board.unicode())
            if hasattr(self, 'move_uci'):
                print(self.move_uci)
            return None

        execute_computer_move(self.maingame,self.board,move_uci)
        return move_uci

    def find_check_move(self):
        """
        优先寻找能吃掉对方将军的移动，确保移动在合法范围内
        """
        print("🧠 寻找能吃掉对方将军的移动...")

        # 获取所有合法移动
        legal_moves = list(self.board.legal_moves)

        # 首先寻找能直接吃掉对方将军的移动
        for move in legal_moves:
            # 检查这个移动是否是吃子移动
            if self.board.is_capture(move):
                # 获取目标位置的棋子
                target_piece = self.board.piece_at(move.to_square)
                # 检查目标位置是否是对方的将/帅
                if target_piece and target_piece.piece_type == cchess.KING:
                    move_uci = move.uci()
                    print(f"✅ 找到能吃掉对方将军的移动: {move_uci}")
                    return move_uci

        # 如果没有能直接吃掉将军的移动，则使用原来的AI计算
        print("⚠️ 没有找到能直接吃掉将军的移动，使用默认AI计算...")

        max_attempts = 3
        move_uci = None

        for attempt in range(max_attempts):
            try:
                from_x, from_y, to_x, to_y = uci_to_coordinates(self.move_uci) if self.move_uci else (4, 0, 4, 1)
                move_uci = get_best_move_with_computer_play(self.maingame, self.board, from_x, from_y, to_x, to_y)

                # 验证计算出的移动是否合法
                if move_uci:
                    calculated_move = cchess.Move.from_uci(move_uci)
                    if calculated_move in legal_moves:
                        print(f"✅ AI决定走: {move_uci} (合法移动)")
                        return move_uci
                    else:
                        print(f"⚠️ 计算出的移动 {move_uci} 不合法，重新计算...")
                        time.sleep(0.5)
                else:
                    time.sleep(0.5)
            except Exception as e:
                print(f"⚠️ 计算出错: {e}")
                time.sleep(0.5)

        # 如果AI计算失败，从合法移动中选择一个
        if legal_moves:
            selected_move = legal_moves[0]
            move_uci = selected_move.uci()
            print(f"🔄 选择第一个合法移动: {move_uci}")
            return move_uci

        print("❌ 无法找到合法移动")
        return None
