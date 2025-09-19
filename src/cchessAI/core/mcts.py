# 核心算法
import time
import numpy as np

from src.cchessAI import cchess
from src.cchessAI.core.threat_evaluator import ThreatEvaluator
from src.cchessAI.parameters import IS_DEBUG, C_PUCT, EPS, ALPHA
from utils.tools import move_id2move_action, is_tie, softmax


class Node:
    """
    蒙特卡罗树中的节点，表示一个游戏状态。
    """

    def __init__(self, parent=None, prob=None):
        self.parent = parent  # 父节点
        self.children = {}   # 子节点字典 {action: Node}
        self.value = 0       # 当前节点的价值估计 Q
        self.visits = 0      # 访问次数 N
        self.prob = prob     # 先验概率 P（来自策略网络）

    def is_leaf(self):
        """判断是否是叶子节点"""
        return self.children == {}

    def is_root(self):
        """判断是否是根节点"""
        return self.parent is None

    def expand(self, action_priors):
        """
        根据策略网络的输出扩展子节点
        :param action_priors: (action, prior_probability) 列表
        """
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = Node(parent=self, prob=prob)

    def puct_value(self, c_puct=C_PUCT):
        """
        使用 PUCT 公式计算当前节点的价值
        """
        if self.visits == 0:
            return float("inf")
        q_value = self.value
        u_value = c_puct * self.prob * np.sqrt(self.parent.visits) / (1 + self.visits)
        return q_value + u_value

    def select(self, c_puct):
        """
        选择最优子节点
        :return: (action, next_node)
        """
        return max(self.children.items(), key=lambda node: node[1].puct_value(c_puct))

    def update(self, leaf_value):
        """
        更新当前节点的访问次数和价值估计
        visits: 节点的访问次数,代表了搜索的“置信度”
        value：当前局面的静态评估值，用于 playout 结束时的叶节点估值。 输出范围为 [-1, 1]，代表当前玩家的胜率估计
        """
        self.visits += 1
        self.value += (leaf_value - self.value) / self.visits

    def update_recursive(self, leaf_value):
        """
        从叶节点反向更新所有祖先节点
        """
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)


class MCTS():
    """
    蒙特卡罗树搜索主体类
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=400):
        """
        :param policy_value_fn: 输入棋盘返回 (action_probs, value) 的函数
        :param c_puct: 探索常数
        :param n_playout: 每次搜索的模拟次数
        """
        self.root = Node(prob=1.0)
        self.policy = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.cache = {}  # FEN 缓存，避免重复计算相同局面
        self.threat_cache = {}  # 新增威胁缓存


    def _playout(self, board):
        node = self.root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self.c_puct)
            board.push(cchess.Move.from_uci(move_id2move_action[action]))

        fen = board.fen()
        if fen in self.cache:
            leaf_value = self.cache[fen]
            action_probs, _ = self.policy(board)  # 重用缓存值
        else:
            action_probs, leaf_value = self.policy(board)
            self.cache[fen] = leaf_value

        if not board.is_game_over() and not is_tie(board):
            node.expand(action_probs)
        elif board.is_game_over():
            winner = cchess.RED if board.outcome().winner else cchess.BLACK
            leaf_value = 1.0 if winner == board.turn else -1.0
        else:
            leaf_value = 0.0

        node.update_recursive(-leaf_value)
    def get_top_n_within_threshold(self, vi_logits, threshold=2.0):
        """
        获取与最大值差距不超过 threshold 的前 n 个元素的索引

        参数:
            vi_logits (np.array): 访问概率对数数组
            n (int): 最多返回多少个索引
            threshold (float): 与最大值的最大允许差值

        返回:
            List[int]: 满足条件的前 n 个索引
        """
        # 获取最大值
        max_logit = np.max(vi_logits)

        # 找出所有满足条件的索引
        candidates = [i for i, val in enumerate(vi_logits) if max_logit - val <= threshold]

        # 按值从高到低排序这些候选索引
        sorted_candidates = sorted(candidates, key=lambda i: vi_logits[i], reverse=True)

        # 取前 n 个
        return sorted_candidates

    def get_move_probs(self, board, temp=1e-3, is_evaluator=False):
        total_time = 0.0
        playout_start_time = time.time()

        for i in range(self.n_playout):
            start_time = time.time()
            board_copy = board.copy()
            self._playout(board_copy)
            elapsed = time.time() - start_time
            total_time += elapsed

        playout_total_time = time.time() - playout_start_time

        if IS_DEBUG:
            print(f"[INFO] 总模拟耗时：{total_time:.4f} 秒，平均每次：{total_time / self.n_playout:.6f} 秒")
            print(f"[INFO] 模拟循环总耗时（包含循环开销）：{playout_total_time:.4f} 秒")

        processing_start_time = time.time()

        act_visits_start_time = time.time()
        act_visits = [(act, node.visits) for act, node in self.root.children.items()]
        act_visits_time = time.time() - act_visits_start_time

        if act_visits:
            acts, visits = zip(*act_visits)
            visits = np.array(visits)

            log_start_time = time.time()
            vi_logits = np.log(visits + 1e-10)
            log_time = time.time() - log_start_time

            m = 4  # 奖惩权重
            logits = vi_logits

            if is_evaluator:
                # 使用辅助算法 + 缓存机制
                threat_start_time = time.time()

                # 获取访问次数最高的前5个动作索引
                vitop_5_indices_start = time.time()
                vitop_5_indices = self.get_top_n_within_threshold(vi_logits, threshold=0.9*m)
                vitop_5_indices_time = time.time() - vitop_5_indices_start

                if vitop_5_indices:
                    vitop_5_acts = [acts[i] for i in vitop_5_indices]

                    # 只对这5个动作进行威胁与奖励计算
                    evaluator_start_time = time.time()
                    evaluator = ThreatEvaluator()
                    eat_rewards_top5, threat_penalties_top5 = evaluator.calculate_eat_rewards_and_threats(board, vitop_5_acts,
                                                                                                          threat_depth=1)
                    evaluator_time = time.time() - evaluator_start_time

                    # 构造一个完整的 eat_rewards/threat_penalties 列表，其余为0
                    array_construction_start_time = time.time()
                    eat_rewards = np.zeros_like(vi_logits)
                    threat_penalties = np.zeros_like(vi_logits)
                    for idx_in_all, top5_idx in enumerate(vitop_5_indices):
                        eat_rewards[top5_idx] = eat_rewards_top5[idx_in_all]
                        threat_penalties[top5_idx] = threat_penalties_top5[idx_in_all]
                    array_construction_time = time.time() - array_construction_start_time

                    threat_total_time = time.time() - threat_start_time
                    print(f"[INFO] 辅助算法耗时：{threat_total_time:.4f} 秒")
                    print(f"  - 获取Top动作耗时: {vitop_5_indices_time:.4f} 秒")
                    print(f"  - 威胁评估耗时: {evaluator_time:.4f} 秒")
                    print(f"  - 数组构建耗时: {array_construction_time:.4f} 秒")

                    logits = logits + (eat_rewards + threat_penalties) * m
                else:
                    print("[INFO] 未找到Top动作，跳过威胁评估")
            else:
                threat_total_time = 0

            softmax_start_time = time.time()
            logits = 1.0 / temp * logits
            act_probs = softmax(logits)
            softmax_time = time.time() - softmax_start_time

            # 可视化 top5 动作
            sort_start_time = time.time()
            indexed_actions = list(enumerate(zip(acts, act_probs, logits)))
            sorted_with_indices = sorted(indexed_actions, key=lambda x: x[1][2], reverse=True)
            sort_time = time.time() - sort_start_time

            processing_total_time = time.time() - processing_start_time

            if IS_DEBUG:
                print("[DEBUG] 处理阶段各步骤耗时分析:")
                print(f"  - 获取动作访问次数: {act_visits_time:.4f} 秒")
                print(f"  - 对数计算: {log_time:.4f} 秒")
                print(f"  - Softmax计算: {softmax_time:.4f} 秒")
                print(f"  - 排序: {sort_time:.4f} 秒")
                print(f"  - 威胁评估总耗时: {threat_total_time:.4f} 秒")
                print(f"  - 处理阶段总耗时: {processing_total_time:.4f} 秒")
                print(f"  - 整体函数总耗时: {playout_total_time + processing_total_time:.4f} 秒")

                print("[DEBUG] Top 5 动作及原始索引:")
                for idx, (act, prob, logit) in sorted_with_indices[:5]:
                    if is_evaluator and 'eat_rewards' in locals() and 'threat_penalties' in locals():
                        print(f"Index {idx}: 落子： {move_id2move_action[act]} 访问次数：{visits[idx]} 访问概率：{np.round(vi_logits[idx],2)} 吃子奖励: {eat_rewards[idx]}, 威胁惩罚：{np.round(threat_penalties[idx],2)},奖惩系数{m}， 概率: {prob:.4f}")
                    else:
                        print(f"Index {idx}: 落子： {move_id2move_action[act]} 访问次数：{visits[idx]} 访问概率：{np.round(vi_logits[idx],2)}, 概率: {prob:.4f}")

            return acts, act_probs
        else:
            # 处理没有子节点的情况
            processing_total_time = time.time() - processing_start_time
            if IS_DEBUG:
                print(f"[WARN] 没有可用的动作，处理耗时: {processing_total_time:.4f} 秒")
            return [], []



    def update_with_move(self, last_move):
        """
        更新树结构以反映最新一步落子
        """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = Node(prob=1.0)


class MCTS_AI():
    """
    基于 MCTS 的 AI 玩家接口
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=400, is_selfplay=False):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self.is_selfplay = is_selfplay
        self.agent = "AI"

    def set_player_idx(self, idx):
        self.player = idx

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=False):
        """
        获取 AI 动作,从当前棋局的所有合法动作中，按照其对应概率进行加权随机选择一个动作作为 AI 的下一步。
        """
        if IS_DEBUG:
            print(f"[AI] 开始思考第 {len(board.move_stack)} 步...")

        start_time = time.time()
        acts, probs = self.mcts.get_move_probs(board, temp)
        if IS_DEBUG:
            print(f"[AI] 思考结束，耗时 {time.time() - start_time:.2f} 秒")

        move_probs = np.zeros(2086)
        move_probs[list(acts)] = probs

        if self.is_selfplay:
            # 自我对弈时添加 Dirichlet 噪声增强探索
            move = np.random.choice(acts, p=(1 - EPS) * probs + EPS * np.random.dirichlet(ALPHA * np.ones(len(probs))))
            self.mcts.update_with_move(move)
        else:
            move = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)

        if return_prob:
            return move, move_probs
        else:
            return move
