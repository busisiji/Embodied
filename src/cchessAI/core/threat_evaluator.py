from src.cchessAI import cchess
from utils.tools import move_id2move_action

# 棋子价值表，用于衡量吃子/被吃的优先级
PIECE_VALUES = {
    'K': 100,  # 帅/将
    'A': 10,   # 士
    'B': 10,   # 象
    'N': 30,   # 马
    'R': 90,   # 车
    'C': 45,   # 炮
    'P': 10,   # 兵/卒
}


class ThreatEvaluator:
    """
    威胁评估器：提供棋局中各个位置的威胁值计算、吃子奖励与威胁惩罚机制。
    支持多步威胁链分析、保护者检测、威胁缓存等高级功能。
    """

    def __init__(self):
        """初始化威胁缓存"""
        self.threat_cache = {}  # 缓存威胁值以避免重复计算

    def _get_piece_value(self, piece):
        """
        获取指定棋子的价值

        参数:
            piece (cchess.Piece): 棋盘上的棋子对象

        返回:
            int: 棋子对应的价值数值（如车为 90）
        """
        if not piece:
            return 0
        symbol = piece.symbol().upper()[0]
        return PIECE_VALUES.get(symbol, 0)

    def is_protected(self, board, square, cache=None):
        """
        判断某位置上的棋子是否受到己方保护。

        方法：模拟敌方在该位置放置一个棋子，看是否有己方走法能回吃。

        参数:
            board (cchess.Board): 当前棋局状态
            square (int): 棋盘上的数字编号 (0~89)
            cache (dict): 可选缓存字典，用于加速多次调用

        返回:
            bool: 是否受到保护
        """
        if cache is not None:
            key = (square, board.turn)
            if key in cache:
                return cache[key]

        temp_board = board.copy()
        original_piece = temp_board.piece_at(square)
        if not original_piece:
            if cache is not None:
                cache[key] = False
            return False

        enemy_color = not temp_board.turn
        temp_board.remove_piece_at(square)
        temp_board.set_piece_at(square, cchess.Piece(cchess.PAWN, enemy_color))

        is_protected_flag = False
        for move in temp_board.legal_moves:
            if move.to_square == square:
                attacker = temp_board.piece_at(move.from_square)
                if attacker and attacker.color == temp_board.turn:
                    is_protected_flag = True
                    break

        if cache is not None:
            cache[key] = is_protected_flag
        return is_protected_flag

    def get_all_protectors(self, board, square):
        """
        获取所有可以保护目标位置的己方棋子。

        参数:
            board (cchess.Board): 当前棋局状态
            square (int): 棋盘上的位置编号

        返回:
            List[dict]: 保护者列表，每个元素包含 from_square 和 piece 类型
        """
        temp_board = board.copy()
        original_piece = temp_board.piece_at(square)

        if original_piece is None:
            return []

        enemy_color = not temp_board.turn
        temp_board.remove_piece_at(square)
        temp_board.set_piece_at(square, cchess.Piece(cchess.PAWN, enemy_color))

        protectors = []
        for move in temp_board.legal_moves:
            if move.to_square == square:
                attacker = temp_board.piece_at(move.from_square)
                if attacker and attacker.color == temp_board.turn:
                    protectors.append({
                        'from_square': move.from_square,
                        'piece': attacker.symbol()
                    })

        return protectors

    def simulate_piece_captured(self, board, square):
        """
        模拟某位置上的棋子被吃掉后的局面。

        参数:
            board (cchess.Board): 当前棋局状态
            square (int): 被吃掉棋子的位置编号

        返回:
            cchess.Board: 已模拟吃掉该棋子的新棋盘
        """
        temp_board = board.copy()
        temp_board.remove_piece_at(square)
        return temp_board

    def analyze_threat_chain(self, board, attacker_value, target_square, max_depth=5):
        """
        分析从攻击者到目标棋子，再到其保护者的完整威胁链。

        参数:
            board (cchess.Board): 当前棋局状态
            attacker_value (int): 攻击者的棋子价值
            target_square (int): 被攻击的目标棋子所在位置
            max_depth (int): 最大威胁链深度

        返回:
            float: 威胁值（负数表示有威胁）
        """
        if max_depth <= 0:
            return -attacker_value * 0.01

        # 获取目标棋子
        target_piece = board.piece_at(target_square)
        if not target_piece:
            return 0.0

        target_value = self._get_piece_value(target_piece)

        # 检查目标棋子是否受到保护
        protectors = self.get_all_protectors(board, target_square)
        if not protectors:
            return -(attacker_value - target_value) * 0.01

        min_threat = float('inf')

        for protector in protectors:
            protector_square = protector['from_square']
            protector_piece = board.piece_at(protector_square)
            if not protector_piece:
                continue

            protector_value = self._get_piece_value(protector_piece)

            # 判断该保护者是否受到威胁
            protector_threat = self.predict_n_step_threat(board, protector_square, threat_depth=1)

            if protector_threat >= 0:
                # 保护者安全 → 整个威胁链无效
                current_threat = 0.0
            else:
                # 保护者也受到威胁 → 继续向下分析威胁链
                current_threat = self.analyze_threat_chain(
                    board,
                    protector_value,
                    protector_square,
                    max_depth - 1
                )

            # 计算当前威胁链的威胁值（攻击者 vs 目标棋子）
            direct_threat = (attacker_value - target_value) * 0.01

            # 总威胁为当前 + 下一级威胁（取更小值）
            combined_threat = direct_threat + current_threat * 0.8
            min_threat = min(min_threat, combined_threat)

        return min_threat

    def _evaluate_direct_capture(self, board, square, attacker_value):
        """
        评估敌方攻击者对受保护己方棋子的威胁值（不考虑保护者）。

        参数:
            board (cchess.Board): 当前棋局状态
            square (int): 被攻击的棋盘位置编号
            attacker_value (int): 攻击者的棋子价值

        返回:
            float: 威胁值（负数表示有威胁）
        """
        target_piece = board.piece_at(square)
        if not target_piece:
            return 0.0

        target_value = self._get_piece_value(target_piece)

        if attacker_value <= target_value:
            # 如果攻击者价值小于等于被吃棋子，认为攻击划算
            return -(attacker_value - target_value) * 0.01
        else:
            # 否则认为攻击者更强，该位置仍有潜在威胁
            return -target_value * 0.01

    def predict_n_step_threat(self, board, square, threat_depth=1):
        """
        使用缓存机制预测某个位置在未来 N 步内是否会被吃掉。
        现在会正确切换到敌方视角进行威胁判断，并使用 is_protected 判断保护机制。

        参数:
            board (cchess.Board): 当前棋局状态
            square (int): 棋盘上的位置编号
            threat_depth (int): 最大威胁分析深度

        返回:
            float: 威胁值（负数表示有威胁）
        """
        fen = board.board_fen()
        key = (fen, square, threat_depth)
        # 使用缓存
        # if key in self.threat_cache:
        #     return self.threat_cache[key]

        target_piece = board.piece_at(square)
        if not target_piece:
            self.threat_cache[key] = 0.0
            return 0.0

        target_symbol = target_piece.symbol().upper()[0]
        target_value = PIECE_VALUES.get(target_symbol, 0)
        threat_value = 0.0

        enemy_color = not board.turn
        enemy_board = board.copy()
        enemy_board.turn = enemy_color
        enemy_board.clear_stack()

        # 遍历敌方视角下所有合法的移动（即敌方可能执行的攻击动作）
        for enemy_move in enemy_board.legal_moves:
            # 如果该移动的目标位置是当前分析的位置 [square](file://E:\现有文件\工作\工程\新人工智能实训室\apiserver\src\cchessAI\cchess\__init__.py#L213-L215)，说明这是一个对该位置棋子的威胁
            if enemy_move.to_square == square:
                # 获取攻击者的起始位置上的棋子对象
                attacker = enemy_board.piece_at(enemy_move.from_square)
                # 如果攻击者存在，则继续分析
                if attacker:
                    # 获取攻击者棋子的符号（如'R'表示车），并提取其价值用于后续计算
                    attacker_symbol = attacker.symbol().upper()[0]
                    attacker_value = PIECE_VALUES.get(attacker_symbol, 0)

                    # 判断目标位置的棋子是否受到己方保护
                    if self.is_protected(board, square):
                        # 获取所有可能的保护者
                        protectors = self.get_all_protectors(board, square)

                        if protectors:
                            # 调用威胁链分析函数，找出最小威胁路径
                            threat_value += self.analyze_threat_chain(board, attacker_value, square)
                    else:
                        # 如果目标位置未受保护，则直接加上负的威胁值（根据攻击者价值）
                        threat_value += -attacker_value * 0.01
        self.threat_cache[key] = threat_value
        return threat_value

    def get_threatened_pieces(self, board, threat_depth=1):
        """
        获取当前棋局中所有受到威胁的己方棋子及其威胁值。

        返回:
            dict: {square: threat_value} 键为棋子所在位置，值为其威胁值
        """
        threatened = {}
        for square in board.piece_map().keys():
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                threat_value = self.predict_n_step_threat(board, square, threat_depth)
                if threat_value < 0:
                    threatened[square] = threat_value
        return threatened

    def calculate_eat_rewards_and_threats(self, board, acts, threat_depth=1):
        eat_rewards = []
        threat_penalties = []

        # 获取当前棋盘下所有受威胁的己方棋子及其威胁值
        current_threatened = self.get_threatened_pieces(board, threat_depth)
        current_threat_total = sum(current_threatened.values())

        piece_value_cache = {}

        def get_piece_value(sq):
            if sq in piece_value_cache:
                return piece_value_cache[sq]
            piece = board.piece_at(sq)
            value = self._get_piece_value(piece)
            piece_value_cache[sq] = value
            return value

        for act in acts:
            move_str = move_id2move_action[act]
            move = cchess.Move.from_uci(move_str)
            from_square = move.from_square
            to_square = move.to_square

            captured_value = get_piece_value(to_square)
            eat_reward = captured_value * 0.01

            # 模拟走子
            temp_board = board.copy()
            temp_board.push(move)

            # 强制切换回己方视角进行威胁评估
            temp_board.turn = board.turn

            # 获取模拟后棋盘中所有受威胁的己方棋子（排除刚移动的棋子）
            new_threatened = self.get_threatened_pieces(temp_board, threat_depth)
            # 排除掉刚刚移动的棋子所在的位置（to_square）的威胁值
            new_threatened.pop(to_square, None)
            new_threat_total = sum(new_threatened.values())

            # 单独获取移动棋子在新位置的威胁值
            moving_piece_threat = self.predict_n_step_threat(temp_board, to_square, threat_depth)

            # 计算新的惩罚值：原威胁值 + 移动棋子的新威胁值 - 原威胁值
            threat_diff = (new_threat_total + moving_piece_threat) - current_threat_total
            threat_penalty = threat_diff

            eat_rewards.append(eat_reward)
            threat_penalties.append(threat_penalty)

        return eat_rewards, threat_penalties