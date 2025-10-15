# file: /media/jetson/KESU/V10.14/Embodied/runner/chessPlayFlow_runner.py
import argparse
import asyncio
import base64
import copy
import logging
import threading
import time
import os
import sys
from datetime import datetime

import cv2
import numpy as np

from runner.chessPlayFlow.chess_branch import ChessPlayFlowBranch
from runner.chessPlayFlow.chess_camera import ChessPlayFlowCamera
from runner.chessPlayFlow.chess_move import ChessPlayFlowMove
from runner.chessPlayFlow.chess_utils import ChessPlayFlowUtils
from src.cchessAI.parameters import MODELS
from parameters import RED_CAMERA, BLACK_CAMERA, RCV_CAMERA, WORLD_POINTS_R, WORLD_POINTS_B, SRC_RCV_POINTS, \
    DST_RCV_POINTS, IO_SIDE, IO_START, IO_STOP, IO_RESET
from src.cchessAG.chinachess import MainGame
from src.cchessAI import cchess

# 添加项目路径到PYTHONPATH
dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.cchessAI.game import Game
from src.cchessAI.mcts import MCTS_AI
from src.cchessAI.net import PolicyValueNet
# 从YOLO模块导入
from src.cchessYolo.chess_detection_trainer import ChessPieceDetectorSeparate


# 中文棋谱坐标系 ：
# 1 2 3 4 5 6 7 8 9
# 2
# 3
# 4
# 5
#                      五
#                      四
#                      三
#                      二
# 九 八 七 六 五 四 三 二 一

# 简谱坐标系
# 9
# 8
# 7
# 6
# 5
# 4
# 3
# 2
# 1
# 0/a b c d e f h i

# 数组坐标系
# 0
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9/0 1 2 3 4 5 6 7 8

class ChessPlayFlowInit():
    def __init__(self, args):
        self.args = args
        self.urController = None # 机械人控制器
        self.detector = None # 棋子检测器
        self.board = cchess.Board()
        self.game = Game(self.board)
        self.move_history =  [] # 存储历史移动
        self.mcts_player = None
        self.human_player = None
        self.side = 'red'  # 开始棋子方为红方
        self.point_home = self.args.red_camera_position # 红黑拍照点
        self.pipeline = None # 相机
        self.chessboard_image = None # 棋盘图片
        self.human_move_by_voice =  False # 是否使用语音控制落子
        self.is_playing = False # 是否人类正在落子
        self.box_center = [RCV_CAMERA[0],RCV_CAMERA[1]] # 棋盒中心点

        # 状态管理
        self.game_state = 'start' # 状态管理
        self.surrendered = False  # 添加投降标志
        self._game_paused = False  # 添加游戏暂停标志
        self._stop_event = threading.Event()  # 添加停止事件，用于立即停止所有操作

        # 棋盘状态
        self.sac_nums = 0 # 吃子数量
        self.move_uci = ''                                  # 棋子移动 使用简谱坐标系
        # 棋子映射字典
        self.piece_map = {
            'r': '车', 'n': '马', 'b': '象', 'a': '士', 'k': '将', 'c': '炮', 'p': '卒',  # 黑方
            'R': '車', 'N': '馬', 'B': '相', 'A': '仕', 'K': '帥', 'C': '砲', 'P': '兵'   # 红方
        }
        self.his_chessboard = {} # 历史棋盘
        # 识别的棋盘
        self.chess_positions = [                            # 使用数组坐标系
            ['r', 'n', 'b', 'a', 'k', 'a', 'b', 'n', 'r'],  # 0行 黑方
            ['.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 1行
            ['.', 'c', '.', '.', '.', '.', '.', 'c', '.'],  # 2行
            ['p', '.', 'p', '.', 'p', '.', 'p', '.', 'p'],  # 3行
            ['.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 4行
            ['.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 5行
            ['P', '.', 'P', '.', 'P', '.', 'P', '.', 'P'],  # 6行 红方
            ['.', 'C', '.', '.', '.', '.', '.', 'C', '.'],  # 7行
            ['.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 8行
            ['R', 'N', 'B', 'A', 'K', 'A', 'B', 'N', 'R']   # 9行
        ]
        # 初始化棋子像素坐标存储
        self.piece_pixel_positions = {}                 # '行列'：像素坐标   数组坐标系
        # 当前的棋盘状态
        self.previous_positions = self.chess_positions  # 使用数组坐标系

        # 计算透视变换矩阵
        self.m_rcv = cv2.getPerspectiveTransform(SRC_RCV_POINTS, DST_RCV_POINTS)

        # 初始化MainGame
        self.maingame = MainGame()
        self.maingame.piecesInit()

        # 初始化日志器
        self.logger = logging.getLogger(f"ChessPlayFlow-{os.getpid()}")
        self.logger.setLevel(logging.DEBUG)

        # 避免重复添加处理器
        if not self.logger.handlers:
            # 创建日志目录
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)

            # 创建文件处理器
            file_handler = logging.FileHandler(
                os.path.join(log_dir, f'chess_play_flow_{os.getpid()}.log'),
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)

            # 创建控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # 创建格式器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # 添加处理器到日志器
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

        self.cCamera = ChessPlayFlowCamera(self)
        self.cUtils = ChessPlayFlowUtils(self)
        self.cMove = ChessPlayFlowMove(self)
        self.cBranch = ChessPlayFlowBranch(self)


    # 状态管理
    def pause_thread(self,seconds=1):

        time.sleep(seconds)

        print("暂停结束")
    def check_game_state(self,is_wait=True):
        """
        统一检查游戏状态，用于快速响应投降或暂停
        Returns:
            tuple: (should_stop, should_pause)
        """
        # 检查是否收到停止信号
        if self._stop_event.is_set() or self.surrendered:
            return True, False
        if is_wait:
            self.wait_while_paused()
        return self.surrendered, self._game_paused

    def wait_while_paused(self):
        """
        等待游戏从暂停状态恢复
        """
        while self._game_paused and not self.surrendered and not self._stop_event.is_set():
            time.sleep(0.1)  # 短暂休眠避免过度占用CPU
        return self.surrendered or self._stop_event.is_set()

    def set_surrendered(self):
        """认输"""
        self.surrendered = True
        self._stop_event.set()  # 设置停止事件
        time.sleep(3)
        if self.urController:
            self.urController.hll()


    # 语音播报
    async def speak_cchess(self, text):
        """
        使用统一的TTS管理器进行异步语音播报

        Args:
            text: 要播报的文本
        """
        # 检查是否启用语音
        if not self.args.enable_voice:
            return
        try:
            from manager.manager import system_manager
            # 尝试异步调用
            await system_manager.speak_async(text)
        except Exception as e:
            print(f"⚠️ 语音播报失败: {e}")
            # 不中断程序执行
            pass

    # 初始化
    def initialize_chessboard_points(self):
        """
        初始化棋盘所有点位坐标
        根据WORLD_POINTS_R和WORLD_POINTS_B创建存储棋盘所有点位的参数
        """
        print("_INITIALIZING_CHESSBOARD_POINTS_...")

        # 初始化红方和黑方的棋盘点位字典
        self.red_board_points = {}
        self.black_board_points = {}

        red_top_left = np.array(WORLD_POINTS_R[1])      # 右上
        red_top_right = np.array(WORLD_POINTS_R[2])     # 右下
        red_bottom_left = np.array(WORLD_POINTS_R[0])   # 左上
        red_bottom_right = np.array(WORLD_POINTS_R[3])  # 左下

        # 计算红方区域的棋盘点位 (0-4行)
        for row in range(5):  # 0-4行对应红方
            for col in range(9):  # 0-8列
                # 计算在红方区域中的相对位置
                # 行从上到下: 0->1, 4->0
                u = col / 8.0  # 列比例 0-1
                v = row / 4.0  # 行比例 0-1

                # 顶部线性插值
                top_point = red_top_left + u * (red_top_right - red_top_left)
                # 底部线性插值
                bottom_point = red_bottom_left + u * (red_bottom_right - red_bottom_left)
                # 垂直插值
                point = bottom_point + v * (top_point - bottom_point)

                # 存储为 (行, 列) 格式
                self.red_board_points[(row, col)] = tuple(point)

        black_top_left = np.array(WORLD_POINTS_B[1])      # 右上
        black_top_right = np.array(WORLD_POINTS_B[2])     # 右下
        black_bottom_left = np.array(WORLD_POINTS_B[0])   # 左上
        black_bottom_right = np.array(WORLD_POINTS_B[3])  # 左下

        # 计算黑方区域的棋盘点位 (5-9行)
        for row in range(5, 10):  # 5-9行对应黑方
            for col in range(9):  # 0-8列
                # 计算在黑方区域中的相对位置
                # 行从上到下: 5->0, 9->1
                u = col / 8.0  # 列比例 0-1
                v = (row - 5) / 4.0  # 行比例 0-1 (转换为0-4范围再归一化)

                # 顶部线性插值
                top_point = black_top_left + u * (black_top_right - black_top_left)
                # 底部线性插值
                bottom_point = black_bottom_left + u * (black_bottom_right - black_bottom_left)
                # 垂直插值
                point = bottom_point + v * (top_point - bottom_point)

                # 存储为 (行, 列) 格式
                self.black_board_points[(row, col)] = tuple(point)

        # 合并所有棋盘点位到一个字典中
        self.chessboard_points = {}
        self.chessboard_points.update(self.red_board_points)
        self.chessboard_points.update(self.black_board_points)

        print(f"✅ 棋盘点位初始化完成")
        print(f"   红方点位数量: {len(self.red_board_points)}")
        print(f"   黑方点位数量: {len(self.black_board_points)}")
        print(f"   总点位数量: {len(self.chessboard_points)}")


class ChessPlayFlow(ChessPlayFlowInit):

    # 主函数
    def set_side(self):
        if self.side == 'red':
            self.side = 'black'
        else:
            self.side = 'red'
    def _init_play_game(self):
        self.his_chessboard = {} # 历史棋盘
        # 识别的棋盘
        self.chess_positions = [                            # 使用数组坐标系
            ['r', 'n', 'b', 'a', 'k', 'a', 'b', 'n', 'r'],  # 0行 黑方
            ['.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 1行
            ['.', 'c', '.', '.', '.', '.', '.', 'c', '.'],  # 2行
            ['p', '.', 'p', '.', 'p', '.', 'p', '.', 'p'],  # 3行
            ['.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 4行
            ['.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 5行
            ['P', '.', 'P', '.', 'P', '.', 'P', '.', 'P'],  # 6行 红方
            ['.', 'C', '.', '.', '.', '.', '.', 'C', '.'],  # 7行
            ['.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 8行
            ['R', 'N', 'B', 'A', 'K', 'A', 'B', 'N', 'R']   # 9行
        ]
        self.previous_positions = self.chess_positions # 现在的棋盘
        self.move_history = []
        self.board = cchess.Board()
        self.game = Game(self.board)
        self.surrendered = False
        self.captured_pieces_history = {}  # 记录被吃的棋子信息
        self.is_undo = False  # 添加悔棋标志
        self.move_count = 0
        self.move_uci = ''

        # 初始化MainGame
        self.maingame.restart_game()

        # 显示初始棋盘
        if self.args.show_board:
            self.game.graphic(self.board)

    # 语音事件
    def handle_voice_command(self, keywords, full_text):
        """
        处理语音命令 - 支持象棋移动命令的专用识别
        """
        print(f"识别到语音命令: {full_text}")

        # 游戏控制命令
        if "帮助" in full_text:
            asyncio.run(self.speak_cchess("您可以使用语音控制游戏，说开始、结束、悔棋等命令"))
            return

        if '开始' in full_text:
            self.play_game()
            return
        if '停止' in full_text or '暂停' in full_text:
            self._handle_stop_game()
            return
        elif '启动' in full_text or '继续' in full_text:
            self._handle_start_game()
            return
        elif '复位' in full_text or '重启' in full_text:
            self._handle_reset_board()
            return

        if self.game_state == 'start':
            # 添加收子关键字相关回调事件
            if "收子" in full_text:
                asyncio.run(self.speak_cchess("执行收子"))
                try:
                    # 调用收子方法
                    self.cBranch.collect_pieces_at_end()
                except Exception as e:
                    asyncio.run(self.speak_cchess("收子操作失败"))
                    print(f"收子操作失败: {e}")
            # 添加布局关键字相关回调事件
            elif "布局" in full_text or "摆子" in full_text:
                asyncio.run(self.speak_cchess("执行初始布局操作"))
                try:
                    # 调用布局方法
                    self.cBranch.setup_initial_board()
                except Exception as e:
                    asyncio.run(self.speak_cchess("布局操作失败"))
                    print(f"布局操作失败: {e}")

        if self.game_state == 'running':
            if "认输" in full_text or "投降" in full_text:
                asyncio.run(self.speak_cchess("执行认输"))
                self.set_surrendered()
                return
            if not self.is_playing:
                # asyncio.run(self.speak_cchess("还没轮到您的回合"))
                return
            if "悔棋" in full_text or "会七" in full_text:
                asyncio.run(self.speak_cchess("执行悔棋"))
                # 设置悔棋标志
                self.cBranch.undo_move()
                return

            # 如果当前不是机器人回合，且不是语音控制移动状态
            if self.side != self.args.robot_side and not self.human_move_by_voice:
                # 检查是否是象棋移动命令（包含棋子名称）
                piece_chars = ['进','退','平','车', '马', '炮', '象', '相', '士', '仕', '将', '帅', '兵', '卒']

                # 检查文本是否包含棋子字符
                if any(piece in full_text for piece in piece_chars):
                    # 解析中文记谱法
                    start_time = time.time()
                    chinese_notation = full_text.strip()
                    move_uci = self.cUtils.parse_chinese_notation(chinese_notation)
                    time_1 = time.time()
                    print("解析中文记谱法", time_1 - start_time)

                    if not move_uci:
                        return

                    # 执行移动
                    success = self.cMove.execute_updata_move(move_uci)
                    if success:
                        # 语音移动成功后设置标志以退出人类回合
                        self.human_move_by_voice = True
                    else:
                        asyncio.run(self.speak_cchess("非法移动，无法执行"))
                        return

                    print(f"语音命令执行移动: {chinese_notation} -> {move_uci}")
                    asyncio.run(self.speak_cchess(f"执行移动 {chinese_notation}"))
    # 初始化
    def initialize(self):
        """初始化所有组件"""
        print("🔧 开始初始化...")
        from manager.manager import system_manager
        try:
            # 语音识别器初始化
            if system_manager.speech_recognizer:
                chess_keywords = ['开始', '停止', '暂停', '启动', '继续', '复位', '重启', '认输', '投降', '悔棋', '收子', '布局', '摆子']
                system_manager.add_keywords(chess_keywords)
                system_manager.register_keyword_callback("象棋", self.handle_voice_command)

            # 相机初始化
            if not system_manager.camera_manager or not system_manager.camera_manager.running:
                raise Exception("相机初始化失败")

            # 机械臂初始化
            print("🤖 获取机械臂实例...")
            self.urController = system_manager.dobot_controller

            if not self.urController or not self.urController.is_connected():
                raise Exception("机械臂未连接或连接失败")

            system_manager.speak_sync("机械臂连接成功")
            self.urController.set_speed(0.8)
            self.urController.run_point_j(self.args.red_camera_position)
            self.urController.hll()

            # 模型初始化
            print("👁️ 初始化棋子识别模型...")
            system_manager.speak_sync("正在加载识别模型")
            self.detector = ChessPieceDetectorSeparate(model_path=self.args.yolo_model_path)

            # 对弈模型初始化
            print("🧠 初始化对弈模型...")
            system_manager.speak_sync("正在加载对弈模型")
            self._initialize_mcts_player()

            # 初始化棋盘点位
            self.initialize_chessboard_points()

            # 注册IO回调
            self._register_io_callbacks(system_manager)

            system_manager.speak_sync("系统初始化完成")

        except Exception as e:
            system_manager.speak_sync(f"初始化失败: {str(e)}")
            raise


    def _initialize_mcts_player(self):
        """初始化MCTS对弈模型"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.args.use_gpu:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.init()

                policy_value_net = PolicyValueNet(
                    model=self.args.play_model_file,
                    use_gpu=self.args.use_gpu
                )
                self.mcts_player = MCTS_AI(
                    policy_value_net.policy_value_fn,
                    c_puct=self.args.cpuct,
                    n_playout=self.args.nplayout
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception("对弈模型初始化失败")
                time.sleep(2)

    def _register_io_callbacks(self, system_manager):
        """注册IO回调函数"""
        callbacks = {
            "start": self._handle_start_game,
            "stop": self._handle_stop_game,
            "reset": self._handle_reset_board
        }

        for io_type, callback in callbacks.items():
            system_manager.register_io_callback(io_type, callback)

        system_manager.start_io_monitoring()
        print("🔔 IO回调注册完成")

    def _handle_start_game(self):
        """
        处理启动游戏事件
        """
        # 只有在暂停状态下才能启动
        if hasattr(self, '_game_paused') and self._game_paused:
            print("🚀 继续对弈游戏")
            try:
                if self._game_paused:
                    self._game_paused = False

                    # 如果有暂停的移动操作，继续执行
                    if hasattr(self, '_paused_move') and self._paused_move:
                        move_uci = self._paused_move
                        self._paused_move = None
                        self.cMove.execute_move(move_uci)

                    asyncio.run(self.speak_cchess("游戏继续"))

            except Exception as e:
                print(f"❌ 启动游戏失败: {e}")
        else:
            print("ℹ️ 游戏未处于暂停状态，无需继续")

    def _handle_stop_game(self):
        """
        处理停止游戏事件
        """
        print("✋ 停止对弈游戏")
        try:
            if not self._game_paused:
                self._game_paused = True

                # 停止机械臂当前所有动作
                self.urController.pause()

                asyncio.run(self.speak_cchess("游戏已暂停"))

        except Exception as e:
            print(f"❌ 停止游戏失败: {e}")

    def _handle_reset_board(self):
        """
        处理复位棋盘事件
        """
        print("🔄 复位棋盘到初始状态")
        try:
            # 检查停止灯是否为暗（即游戏是否在运行）
            # 如果游戏正在进行中，则不执行复位
            if not (hasattr(self, '_game_paused') and self._game_paused):
                print("ℹ️ 游戏正在运行，无法执行复位操作")
                return

            # 执行棋盘还原成初始状态
            self.cBranch.collect_pieces_at_end()
            self.cBranch.setup_initial_board()

            asyncio.run(self.speak_cchess("棋盘已复位"))

            # 重置游戏状态
            self._game_paused = False
            if hasattr(self, '_paused_move'):
                self._paused_move = None

        except Exception as e:
            print(f"❌ 棋盘复位失败: {e}")

    def play_game(self):
        """
        执行完整对弈流程
        """
        # 重置停止事件
        self._stop_event.clear()

        try:
            print("🎮 开始象棋对弈...")
            self.game_state = "running"
            self.urController.hll(IO_START, [IO_START, IO_STOP, IO_RESET])
            asyncio.run(self.speak_cchess("开始对弈，请等待指示灯为绿色再落子"))

            self._init_play_game()

            # 主游戏循环
            while not self.board.is_game_over() and not self._stop_event.is_set():
                # 更频繁地检查游戏状态
                if self.surrendered or self._stop_event.is_set():
                    break

                self.move_count += 1
                print(f"\n--- 第 {self.move_count} 回合 ---")

                if self.move_count == 1:
                    self.board = cchess.Board()

                # 判断当前回合
                is_robot_turn = (self.move_count + (0 if self.args.robot_side == 'red' else 1)) % 2 == 1

                # 初始状态检查
                if self.move_count == 1:
                    asyncio.run(self.speak_cchess("正在检查棋盘初始状态，请稍等"))

                    # 优化初始棋盘识别，增加中断检查
                    if not self._quick_check_and_adjust_initial_board():
                        continue

                if is_robot_turn:
                    # 机器人回合处理
                    if not self._handle_robot_turn():
                        break
                else:
                    # 人类回合处理
                    if not self._handle_human_turn():
                        break

            # 游戏结束处理
            self._handle_game_end()

        except Exception as e:
            self.report_error(str(e))
        finally:
            self._stop_event.set()

    def _quick_check_and_adjust_initial_board(self):
        """
        快速检查并调整初始棋盘状态
        """
        max_attempts = 3
        for attempt in range(max_attempts):
            # 检查是否需要停止
            if self.surrendered or self._stop_event.is_set():
                return False

            self.cCamera.recognize_chessboard()

            # 快速检查位置差异
            differences = self.cUtils.compare_chessboard_positions(
                self.previous_positions,
                self.chess_positions
            )

            if not differences:
                # 位置正确，检查精确位置
                if self.cUtils.check_all_pieces_initial_position(tolerance=15):
                    return True
                else:
                    # 位置不精确，等待调整
                    asyncio.run(self.speak_cchess(f"棋子位置不精确，请调整"))
                    self.cMove.wait_for_player_adjustment()
            else:
                # 位置错误，等待调整
                asyncio.run(self.speak_cchess(f"棋盘状态不正确，请调整"))
                self.cMove.wait_for_player_adjustment()

            # 检查是否需要停止
            if self.surrendered or self._stop_event.is_set():
                return False

        return False

    def _handle_robot_turn(self):
        """
        处理机器人回合
        """
        # 检查游戏状态
        if self.surrendered or self._stop_event.is_set():
            return False

        self.urController.hll(5)  # 红灯
        print(f"🤖 机器人回合")
        self.is_playing = False
        asyncio.run(self.speak_cchess("轮到机器人回合，请稍等"))

        # 显示当前棋盘
        if self.args.show_board:
            self.game.graphic(self.board)

        # 计算下一步 - 增加中断检查
        move_uci = self.cUtils.calculate_next_move()
        if not move_uci or self._stop_event.is_set() or self.surrendered:
            return False

        # 执行移动到棋盘对象
        move = cchess.Move.from_uci(move_uci)
        if move not in self.board.legal_moves:
            asyncio.run(self.speak_cchess("机器人无法执行该移动"))
            self.gama_over()
            return False

        self.board.push(move)

        # 执行物理移动
        self.cMove.execute_move(move_uci)
        if self._stop_event.is_set() or self.surrendered:
            return False

        self.move_history.append(move_uci)

        print(f"当前{self.side}方")
        self.set_side()
        print(f"当前{self.side}方")

        # 检查是否将军
        if self.cUtils.is_in_check(self.board, self.side):
            asyncio.run(self.speak_cchess("请注意，您已被将军！"))

        self.cMove.updat_previous_positions_after_move(move_uci)
        chinese_notation = self.cUtils.uci_to_chinese_notation(move_uci, self.previous_positions)
        asyncio.run(self.speak_cchess(f"机器人已走子，{chinese_notation}"))

        # 显示更新后的棋盘
        if self.args.show_board:
            self.game.graphic(self.board)

        print(chinese_notation)
        return True

    def _handle_human_turn(self):
        """
        处理人类回合
        """
        print("👤 人类回合")
        self.urController.hll(4)  # 绿灯
        asyncio.run(self.speak_cchess("轮到您的回合，请落子"))
        print("⏳ 等待人类落子完成信号...")

        # 等待人类落子完成信号 - 更高效的等待方式
        if not self._wait_for_human_move():
            return False

        self.is_playing = False

        # 再次检查游戏状态
        if self.surrendered or self._stop_event.is_set():
            return False

        if self.human_move_by_voice:
            self.human_move_by_voice = False
            return True

        if self.is_undo:
            self.is_undo = False
            return True

        if self._stop_event.is_set():
            return False

        # 复位信号
        self.urController.hll(5)  # 红灯
        self.io_side = self.urController.get_di(IO_SIDE)
        print("✅ 检测到人类落子完成信号")
        asyncio.run(self.speak_cchess("您已落子，请稍等"))

        # 识别当前棋盘状态以更新棋盘
        print("🔍 识别棋盘以更新状态...")
        self.his_chessboard[self.move_count - 1] = copy.deepcopy(self.previous_positions)

        # 优化棋盘识别过程
        if not self._recognize_and_infer_human_move():
            return True  # 继续游戏循环

        if self.move_uci:
            print(f"✅ 人类推测走子: {self.move_uci}")
            move = cchess.Move.from_uci(self.move_uci)
            if move in self.board.legal_moves:
                # 检查是否吃掉了机器人的将军
                is_captured, king_side = self.cUtils.is_king_captured_by_move(
                    self.move_uci, self.previous_positions
                )
                # 如果吃掉的是机器人的将/帅
                if is_captured and king_side == self.args.robot_side:
                    self.gama_over('player')  # 人类玩家获胜
                    asyncio.run(self.speak_cchess('吃掉了机器人的将军！'))
                    return False  # 结束游戏

                self.board.push(move)

            else:
                # 检查是否被将军且无法解除将军状态
                if self.cUtils.is_in_check(self.board, self.args.robot_side):
                    # 移动无效，执行空移动
                    self.board.push(cchess.Move.null())

                    # 检查是否存在能吃掉将军的移动
                    move_uci = self.cUtils.find_check_move()

                    # 检查这个移动是否真的是吃掉将军的移动
                    move = cchess.Move.from_uci(move_uci)
                    if move in self.board.legal_moves:
                        # 检查目标位置是否是对方的将/帅
                        target_piece = self.board.piece_at(move.to_square)
                        if target_piece and target_piece.piece_type == cchess.KING:
                            # 确实是吃掉将军的移动，执行它
                            self.cMove.execute_move(move_uci)
                            asyncio.run(self.speak_cchess(f"很遗憾，您输了！"))
                            time.sleep(5)
                            return False  # 结束游戏

                else:
                    asyncio.run(self.speak_cchess("您违规了，请重新走子"))
                    self.move_count = self.move_count - 1
                    self.urController.hll(4)  # 绿灯
                    return True
        else:
            print("错误！无法推断人类的移动")
            asyncio.run(self.speak_cchess("无法检测到走棋，请重新落子"))
            self.urController.hll(4)  # 绿灯
            self.move_count = self.move_count - 1
            return True

        # 显示更新后的棋盘
        if self.args.show_board:
            self.game.graphic(self.board)

        # 落子完成
        self.cMove.updat_previous_positions_after_move(self.move_uci)
        print(f"✅ 人类走法已应用: {self.move_uci}")
        chinese_notation = self.cUtils.uci_to_chinese_notation(self.move_uci, self.previous_positions)
        asyncio.run(self.speak_cchess(f"您已走子，{chinese_notation}"))
        print(chinese_notation)

        self.move_history.append(self.move_uci)
        self.his_chessboard[self.move_count] = copy.deepcopy(self.previous_positions)

        self.set_side()
        return True

    def _wait_for_human_move(self):
        """
        等待人类移动完成 - 优化版本
        """
        check_interval = 0.1  # 缩短检查间隔以提高响应速度
        while 1:
            # 更频繁地检查游戏状态
            if self.surrendered or self._stop_event.is_set():
                return False

            # 检查IO信号
            if self.urController.get_di(IO_SIDE, is_log=False):
                return True

            self.is_playing = True

            # 更短的等待时间
            time.sleep(check_interval)

            if self.human_move_by_voice or self.is_undo:
                return True

            # 检查是否需要停止
            if self.surrendered or self._stop_event.is_set():
                return False

    def _recognize_and_infer_human_move(self):
        """
        识别并推断人类移动
        """
        max_attempts = 5
        for i in range(max_attempts):
            # 检查游戏状态
            if self.surrendered or self._stop_event.is_set():
                return False

            # 识别棋盘
            positions = self.cCamera.recognize_chessboard(i > 0)  # 第一次不移动相机

            # 推断人类的移动
            self.move_uci = self.cUtils.infer_human_move(
                self.his_chessboard[self.move_count - 1],
                positions
            )

            if self.move_uci:
                return True

            # 短暂等待后重试
            time.sleep(0.5)

        return False

    def _handle_game_end(self):
        """
        处理游戏结束
        """
        if self.surrendered or self._stop_event.is_set():
            self.gama_over('surrender')
        else:
            # 正常游戏结束
            outcome = self.board.outcome()
            if outcome is not None:
                winner = "red" if outcome.winner == cchess.RED else "black"
                print(f"获胜方是{winner}")
                if winner == self.args.robot_side:
                    asyncio.run(self.speak_cchess("您已被将死！"))
                    self.gama_over('dobot')
                else:
                    self.gama_over('player')
            else:
                self.gama_over('平局')

    def gama_over(self, winner='player'):
        self.urController.hll()
        game_result = ""
        result_text = ""
        self.game_state = 'start'
        if winner == 'player':
            print(f'恭喜您获得胜利！')
            asyncio.run(self.speak_cchess(f"恭喜您获得胜利！"))
            game_result = "玩家胜利"
            result_text = "player_win"
        elif winner == 'dobot':
            print(f'很遗憾，您输了！')
            asyncio.run(self.speak_cchess(f"很遗憾，您输了！"))
            game_result = "机器人胜利"
            result_text = "robot_win"
        elif winner == 'surrender':
            print(f'您已投降！')
            asyncio.run(self.speak_cchess(f"您已投降！"))
            game_result = "玩家投降"
            result_text = "player_surrender"
        else:
            print("🤝 游戏结束，平局")
            asyncio.run(self.speak_cchess(f"游戏结束，平局"))
            game_result = "平局"
            result_text = "draw"

        # 保存对局到CSV文件（除非是投降）
        if winner != 'surrender':
            self.save_game_to_csv(game_result)

        time.sleep(3)


    # 保存
    def save_game_to_csv(self, game_result):
        """
        保存对局记录到CSV文件

        Args:
            game_result: 游戏结果描述
        """
        import csv
        from datetime import datetime
        import os

        # 创建保存目录
        game_records_dir = os.path.join(dir, "game_records")
        os.makedirs(game_records_dir, exist_ok=True)

        # 生成时间戳和对局编号
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        game_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        game_id = f"game_{game_timestamp}"

        # 文件路径
        moves_filename = os.path.join(game_records_dir, "chess_moves.csv")
        summary_filename = os.path.join(game_records_dir, "chess_summary.csv")

        try:
            # 保存详细对局记录
            with open(moves_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # 如果文件为空，写入表头
                if os.path.getsize(moves_filename) == 0:
                    writer.writerow(["对局编号", "回合数", "UCI移动", "中文记谱", "玩家", "记录时间"])

                # 写入每步棋记录
                for i, move in enumerate(self.move_history):
                    # 根据回合数判断是哪方走的棋
                    player = self.args.robot_side if (i + (1 if self.args.robot_side == 'red' else 0)) % 2 == 1 else (
                        'black' if self.args.robot_side == 'red' else 'red')

                    # 转换中文记谱
                    chinese_notation = ""
                    try:
                        if i < len(self.his_chessboard):
                            chinese_notation = self.cUtils.uci_to_chinese_notation(move, self.his_chessboard[i])
                    except:
                        chinese_notation = "未知"

                    writer.writerow([game_id, i + 1, move, chinese_notation, player, timestamp])

            print(f"💾 对局详细记录已保存至: {moves_filename}")
            self.logger.info(f"对局详细记录已保存至: {moves_filename}")

        except Exception as e:
            error_msg = f"保存对局详细记录失败: {e}"
            print(f"⚠️ {error_msg}")
            self.logger.error(error_msg)

        try:
            # 保存对局摘要
            with open(summary_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # 如果文件为空，写入表头
                if os.path.getsize(summary_filename) == 0:
                    writer.writerow(["对局编号", "游戏结果", "总回合数", "记录时间"])

                # 写入对局摘要
                writer.writerow([game_id, game_result, len(self.move_history), timestamp])

            print(f"💾 对局摘要已保存至: {summary_filename}")
            self.logger.info(f"对局摘要已保存至: {summary_filename}")

        except Exception as e:
            error_msg = f"保存对局摘要失败: {e}"
            print(f"⚠️ {error_msg}")
            self.logger.error(error_msg)
    async def save_recognition_result_with_detections(self, red_image=None, red_detections=None, black_image=None, black_detections=None, chess_result=None, move_count=None):
        """
        异步保存带检测框的识别结果图像

        Args:
            red_image: 红方半区原始图像
            red_detections: 红方半区检测结果 (Results对象)
            black_image: 黑方半区原始图像
            black_detections: 黑方半区检测结果 (Results对象)
            chess_result: 棋盘识别结果
        """
        import cv2
        from copy import deepcopy
        import asyncio
        import os

        # 创建结果目录
        result_dir = self.args.result_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if not move_count:
            move_count = self.move_count

        async def save_detections(image, detections, prefix, color):
            """通用保存检测结果函数"""
            if image is not None and detections is not None:
                image_with_detections = deepcopy(image)

                # 从Results对象中提取边界框信息
                boxes = detections[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())

                        # 绘制边界框和标签
                        cv2.rectangle(image_with_detections, (x1, y1), (x2, y2), color, 2)
                        label = f"{prefix}:{cls} {conf:.2f}"
                        cv2.putText(image_with_detections, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # 保存带检测框的图像
                detected_path = os.path.join(result_dir, f"{prefix.lower()}_side_detected{move_count}.jpg")
                cv2.imwrite(detected_path, image_with_detections)
                print(f"💾 {prefix}方检测结果已保存至: {detected_path}")

        async def save_chessboard_layout():
            """异步保存棋盘布局图"""
            if chess_result:
                # 可视化完整的棋盘布局
                self.chessboard_image = self.cMove.visualize_chessboard(chess_result)
                chessboard_path = os.path.join(result_dir, f"chessboard_layout.jpg")
                cv2.imwrite(chessboard_path, self.chessboard_image)
                # 报告棋盘识别结果给web端
                if self.args.use_api:
                    self.report_board_recognition_result()

                print(f"💾 棋盘布局图已保存至: {chessboard_path}")

        # 并发执行保存操作
        await asyncio.gather(
            save_detections(red_image, red_detections, "Red", (0, 255, 0)),
            save_detections(black_image, black_detections, "Black", (255, 0, 0)),
            save_chessboard_layout()
        )

    # 清理
    def cleanup(self):
        """
        清理资源
        """
        try:
            self.surrendered = True

            # 断开机械臂
            try:
                if self.urController:
                    self.urController.hll()
                    print("🔌 断开机械臂连接...")
                    self.urController.disconnect()
            except Exception as e:
                print(f"⚠️ 断开机械臂连接时出错: {e}")

            # 清理相机窗口（但不释放相机资源，由system_manager管理）
            from manager.manager import system_manager
            if system_manager.camera_manager:
                system_manager.camera_manager.cleanup_camera_windows()

            # 关闭OpenCV窗口
            if self.args.show_camera:
                cv2.destroyAllWindows()

            print("✅ 清理完成")
            system_manager.speak_sync("结束运行")
        except Exception as e:
            print(f"⚠️ 清理时出错: {e}")

    # 报告
    def report_piece_deviation(self, row, col, deviation_x, deviation_y, distance):
        """
        报告棋子偏移信息

        Args:
            row: 行号
            col: 列号
            deviation_x: X方向偏移(mm)
            deviation_y: Y方向偏移(mm)
            distance: 总偏移距离(mm)
        """
        # 发送偏移报警到游戏服务
        try:
            from api.services.chess_game_service import chess_game_service
            if hasattr(chess_game_service, 'game_events') and chess_game_service.game_events:
                chess_game_service.game_events.put({
                    "type": "error",
                    "scene": "chess/deviation",
                    "data" : {
                        "position": {"row": row, "col": col},
                        "deviation": {
                            "x": deviation_x,
                            "y": deviation_y,
                            "distance": distance
                        },
                    },
                    "timestamp": datetime.now().isoformat(),
                    "message": f"第{row + 1}行,第{col + 1}列棋子偏离标准位置{distance:.2f}mm"
                })
        except Exception as e:
            print(f"发送偏移报警失败: {e}")

    def report_move(self, player, move_uci, chinese_notation):
        """
        报告棋子移动信息

        Args:
            player: 玩家 ("human" 或 "robot")
            move_uci: UCI格式移动
            chinese_notation: 中文记谱法
        """
        # 发送移动信息到游戏服务
        try:
            from api.services.chess_game_service import chess_game_service
            if hasattr(chess_game_service, 'game_events') and chess_game_service.game_events:
                chess_game_service.game_events.put({
                    "type": "info",
                    "scene": "chess/move",
                    'data':{
                        "player": player,
                        "uci": move_uci,
                        "chinese": chinese_notation
                    },
                    "timestamp": datetime.now().isoformat(),
                    "message": f"{player}走棋: {chinese_notation} ({move_uci})"
                })
        except Exception as e:
            print(f"发送移动信息失败: {e}")

    def report_board_recognition_result(self):
        """
        报告棋盘识别结果图像信息
        """
        # 发送棋盘识别结果到游戏服务
        try:
            from api.services.chess_game_service import chess_game_service
            if hasattr(chess_game_service, 'game_events') and chess_game_service.game_events:

                # 将图像编码为JPEG格式
                if self.chessboard_image is not None:
                    jpg_as_text = ''
                    success, buffer = cv2.imencode('.jpg', self.chessboard_image)
                    if success:
                        # 将 buffer 转换为 bytes
                        buffer_bytes = buffer.tobytes()
                        jpg_as_text = base64.b64encode(buffer_bytes).decode('utf-8')

                    chess_game_service.game_events.put({
                        "type": "info",
                        "scene": "chess/recognition",
                        "data": {
                            "image_data": jpg_as_text,
                        },
                        "timestamp": datetime.now().isoformat(),
                        "message": "棋盘识别结果已更新"
                    })
        except Exception as e:
            print(f"发送棋盘识别结果失败: {e}")

    def report_error(self, error_msg):
        """
        报告错误信息并记录日志

        Args:
            error_msg: 错误信息
        """
        # 记录错误日志
        self.logger.error(f"人机对弈错误: {error_msg}")

        # 发送错误信息到游戏服务
        try:
            from api.services.chess_game_service import chess_game_service
            if hasattr(chess_game_service, 'game_events') and chess_game_service.game_events:
                error_data = {
                    "type": "error",
                    "scene": "chess/error",
                    "data": {},
                    "timestamp": datetime.now().isoformat(),
                    "message": error_msg
                }
                chess_game_service.game_events.put(error_data)
        except Exception as e:
            self.logger.error(f"发送错误信息失败: {e}")
    def __del__(self):
        """
        析构函数，确保资源正确释放
        """
        try:
            self.cCamera.cleanup_camera_windows()
        except:
            pass
def create_parser():
    """创建参数解析器"""
    parser = argparse.ArgumentParser(description='象棋自动对弈系统')

    # 显示和保存参数
    parser.add_argument('--use_api', default=False, help='是否使用api')
    parser.add_argument('--use_ag', default=True, help='是否使用固定算法辅助')
    parser.add_argument('--show_camera', default=False, action='store_true', help='是否显示相机实时画面')
    parser.add_argument('--show_board',  default=False, action='store_true', help='是否在窗口中显示棋局')
    parser.add_argument('--save_recognition_results', default=True, action='store_true', help='是否保存识别结果')
    parser.add_argument('--result_dir', type=str, default='chess_play_results',
                        help='结果保存目录')

    # 语音
    parser.add_argument('--enable_voice', default=True, action='store_true', help='是否启用语音提示')
    parser.add_argument('--voice_rate', type=int, default=0, help='语音语速，语速稍慢(-10)，音调较高(20)，音量适中(90)')
    parser.add_argument('--voice_volume', type=int, default=0, help='语音音量')
    parser.add_argument('--voice_pitch', type=int, default=0, help='语音音调')

    # 机械臂相关参数
    parser.add_argument('--robot_ip', type=str, default='192.168.5.1', help='机械臂IP地址')
    parser.add_argument('--robot_port', type=int, default=30003, help='机械臂移动控制端口')
    parser.add_argument('--robot_dashboard_port', type=int, default=29999, help='机械臂控制面板端口')
    parser.add_argument('--robot_feed_port', type=int, default=30005, help='机械臂反馈端口')

    # 模型路径参数
    parser.add_argument('--yolo_model_path', type=str,
                        default=os.path.join(dir, '../src/cchessYolo/runs/detect/chess_piece_detection_separate3/weights/best.onnx'),
                        help='YOLO棋子检测模型路径')
    parser.add_argument('--play_model_file', type=str,
                        default=os.path.join(MODELS, 'onnx/current_policy_batch7661_202507241306.onnx'),
                        help='对弈模型文件路径')
    # 相机位置参数
    parser.add_argument('--red_camera_position', type=float, nargs=6,
                        default=RED_CAMERA,
                        help='红方拍摄吸子位置 [x, y, z, rx, ry, rz]')
    parser.add_argument('--black_camera_position', type=float, nargs=6,
                        default=BLACK_CAMERA,
                        help='黑方拍摄位置 [x, y, z, rx, ry, rz]')
    parser.add_argument('--black_position', type=float, nargs=6,
                        default=[BLACK_CAMERA[0],BLACK_CAMERA[1],BLACK_CAMERA[2],RED_CAMERA[3],RED_CAMERA[4],RED_CAMERA[5]],
                        help='黑方吸子位置 [x, y, z, rx, ry, rz]')
    # 其他参数
    parser.add_argument('--robot_side', type=str, default='black', help='机器人执子方')
    parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--nplayout', type=int, default=400, help='MCTS模拟次数')
    parser.add_argument('--cpuct', type=float, default=5.0, help='MCTS参数')
    parser.add_argument('--conf', type=float, default=0.45, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.25, help='IOU阈值')

    return parser



def main():
    parser = create_parser()
    args = parser.parse_args()

    # 创建对弈流程对象
    chess_flow = ChessPlayFlow(args)

    try:
        # 初始化
        chess_flow.initialize()

        # 收局
        # chess_flow.cBranch.collect_pieces_at_end()

        # 布局
        # chess_flow.cBranch.setup_initial_board()

        # 开始对弈
        chess_flow.play_game()

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断程序")
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        # import traceback
        # traceback.print_exc()
        chess_flow.report_error(str(e))
    finally:
        # 清理资源
        chess_flow.cleanup()

if __name__ == "__main__":
    main()
