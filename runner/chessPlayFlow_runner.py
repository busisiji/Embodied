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

from api.services.tts_service import  speak_await, speak_async, _init_voice_async_loop
from initialization_manager import  initialize_components, voice_loop
from runner.chessPlayFlow.chess_branch import ChessPlayFlowBranch
from runner.chessPlayFlow.chess_camera import ChessPlayFlowCamera
from runner.chessPlayFlow.chess_move import ChessPlayFlowMove
from runner.chessPlayFlow.chess_utils import ChessPlayFlowUtils
from src.speech.speech_service import initialize_speech_recognizer, get_speech_recognizer
from dobot.dobot_control import connect_and_check_speed
from parameters import RED_CAMERA, BLACK_CAMERA, RCV_CAMERA, WORLD_POINTS_R, WORLD_POINTS_B, SRC_RCV_POINTS, \
    DST_RCV_POINTS, IO_SIDE, IO_START, IO_STOP, IO_RESET
from src.cchessAG.chinachess import MainGame
from src.cchessAI import cchess

# 添加项目路径到PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.cchessAI.core.game import Game
from src.cchessAI.core.mcts import MCTS_AI
from src.cchessAI.core.net import PolicyValueNet
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
        self.surrendered = False  # 添加投降标志
        self._game_paused = False  # 添加游戏暂停标志
        self.human_move_by_voice =  False # 是否使用语音控制落子
        self.is_playing = False # 是否人类正在落子
        self.box_center = [RCV_CAMERA[0],RCV_CAMERA[1]] # 棋盒中心点

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

        # 添加IO监控线程相关属性
        self.io_monitor_thread = None
        self.io_monitoring = False

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
    def set_surrendered(self):
        """认输"""
        self.surrendered = True
        time.sleep(3)
        self.urController.hll()


    # 语音
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
            # 尝试异步调用
            if voice_loop:
                await speak_async(text)
            else:
                # 如果没有事件循环，直接调用同步方法
                await speak_await(text)
        except Exception as e:
            print(f"⚠️ 语音播报失败: {e}")
            # 不中断程序执行
            pass
    def handle_voice_command(self, keywords, full_text):
        """
        处理语音命令 - 支持象棋移动命令的专用识别
        """
        print(f"识别到语音命令: {full_text}")
        if not self.is_playing:
            # asyncio.run(self.speak_cchess("还没轮到您的回合"))
            return None
        speech_recognizer = get_speech_recognizer()
        # 游戏控制命令
        if "悔棋" in full_text or "会 七" in full_text:
            asyncio.run(self.speak_cchess("执行悔棋"))
            # 设置悔棋标志
            self.cBranch.undo_move()
            return None

        elif "帮助" in full_text:
            asyncio.run(self.speak_cchess("您可以使用语音控制游戏，说开始、结束、悔棋等命令"))
            return None

        elif "认输" in full_text or "投降" in full_text:
            asyncio.run(self.speak_cchess("执行认输"))
            self.set_surrendered()
            return None
        # 添加收子关键字相关回调事件
        elif "收子" in full_text:
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

        # 如果当前不是机器人回合，且不是语音控制移动状态
        elif self.side != self.args.robot_side and not self.human_move_by_voice:
            # 检查是否是象棋移动命令（包含棋子名称）
            piece_chars = ['进','退','平','车', '马', '炮', '象', '相', '士', '仕', '将', '帅', '兵', '卒']

            # 检查文本是否包含棋子字符
            if any(piece in full_text for piece in piece_chars):
                # 获取语音识别器实例
                if speech_recognizer:

                    # 解析中文记谱法
                    start_time = time.time()
                    chinese_notation = full_text.strip()
                    move_uci = self.cUtils.parse_chinese_notation(chinese_notation)
                    time_1 = time.time()
                    print("解析中文记谱法", time_1 - start_time)

                    if not move_uci:
                        return False

                    # 执行移动
                    success = self.cMove.execute_updata_move(move_uci)
                    if success:
                        # 语音移动成功后设置标志以退出人类回合
                        self.human_move_by_voice = True
                    else:
                        asyncio.run(self.speak_cchess("非法移动，无法执行"))
                        return False

                    print(f"语音命令执行移动: {chinese_notation} -> {move_uci}")
                    asyncio.run(self.speak_cchess(f"执行移动 {chinese_notation}"))
            else:
                return  False
        else:
            return False

    # 初始化
    def initialize(self):
        """
        初始化所有组件
        """
        print("🔧 开始初始化...")
        # 初始化语音引擎
        try:
            _init_voice_async_loop()
        except Exception as e:
            print(f"⚠️ 语音引擎初始化失败: {e}")
            self.voice_engine = None
        # 初始化语音识别器
        try:
            speech_recognizer = get_speech_recognizer()
            if initialize_speech_recognizer(
            ):
                if speech_recognizer:
                    asyncio.run(speech_recognizer.start_listening())
                print("语音识别初始化并启动成功")
                asyncio.run(self.speak_cchess("语音识别器初始化完成"))
        except Exception as e:
            print(f"⚠️ 语音识别器初始化异常: {e}")
            speech_recognizer = None
        else:
            # 设置语音识别器的回调函数
            speech_recognizer.callback = self.handle_voice_command

        # 2. 初始化相机
        print("📷 初始化相机...")
        # self.speak("正在初始化相机")
        self.init_camera()
        self.cCamera.setup_camera_windows()
        if self.pipeline is None:
            asyncio.run(self.speak_cchess("相机初始化失败,请检查相机连接"))

        # 1. 连接机械臂
        print("🤖 连接机械臂...")
        # asyncio.run(self.speak_cchess("正在连接机械)臂")
        try:
            self.urController = connect_and_check_speed(
                ip=self.args.robot_ip,
                port=self.args.robot_port,
                dashboard_port=self.args.robot_dashboard_port,
                feed_port=self.args.robot_feed_port,
            )
        except Exception as e:
            print(f"⚠️ 连接机械臂失败: {e}")
            asyncio.run(self.speak_cchess("连接机械臂失败"))
            raise Exception(f"机械臂连接失败{e}")

        if not self.urController:
            asyncio.run(self.speak_cchess("机械臂连接失败"))
            raise Exception("机械臂连接失败")

        if not self.urController.is_connected():
            asyncio.run(self.speak_cchess("机械臂连接失败"))
            raise Exception("机械臂连接失败")

        asyncio.run(self.speak_cchess("机械臂连接成功"))
        self.urController.set_speed(0.8)
        # 移动到初始位置
        self.urController.run_point_j(self.args.red_camera_position)
        self.urController.hll()

        # 2. 打开识别模型 (使用 YOLO 检测器)
        print("👁️ 初始化棋子识别模型...")
        asyncio.run(self.speak_cchess("正在加载识别模型"))
        try:
            self.detector = ChessPieceDetectorSeparate(
                model_path=self.args.yolo_model_path
            )
        except Exception as e:
            print(f"⚠️识别模型初始化失败: {e}")
            asyncio.run(self.speak_cchess("识别模型初始化失败"))
            raise Exception("识别模型初始化失败")

        # 3. 打开对弈模型
        print("🧠 初始化对弈模型...")
        asyncio.run(self.speak_cchess("正在加载对弈模型"))
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 在每次尝试前清理可能的CUDA状态
                if self.args.use_gpu:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                policy_value_net = PolicyValueNet(
                    model_file=self.args.play_model_file,
                    use_gpu=self.args.use_gpu
                )
                self.mcts_player = MCTS_AI(
                    policy_value_net.policy_value_fn,
                    c_puct=self.args.cpuct,
                    n_playout=self.args.nplayout
                )
                break  # 成功初始化则跳出循环
            except Exception as e:
                print(f"⚠️ 对弈模型初始化失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    asyncio.run(self.speak_cchess("对弈模型初始化失败"))
                    raise Exception("对弈模型初始化失败")
                time.sleep(2)  # 等待后重试

        # 5. 初始化棋盘
        self.initialize_chessboard_points()

        # 显示初始棋盘
        if self.args.show_board:
            self.game.graphic(self.board)

        # 启动IO监控线程
        self.start_io_monitoring()

        asyncio.run(self.speak_cchess("系统初始化完成"))

    def init_camera(self):
        """
        初始化RealSense相机（支持彩色和深度流）
        """
        try:
            import pyrealsense2 as rs

            # 如果已有pipeline，先停止并释放
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except:
                    pass
                self.pipeline = None

            self.pipeline = rs.pipeline()
            config = rs.config()

            # 启用彩色流
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
            # 启用深度流
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)

            # 启动相机
            profile = self.pipeline.start(config)

            sensors = profile.get_device().query_sensors()
            for sensor in sensors:
                if sensor.get_info(rs.camera_info.name) == "RGB Camera":
                    print("Setting RGB Camera default parameters...")
                    # 设置默认参数
                    if sensor.supports(rs.option.exposure):
                        # sensor.set_option(rs.option.exposure, 300) # 曝光时间（单位：微秒）
                        sensor.set_option(rs.option.auto_exposure_priority, True)
                    # ✅ 启用自动对焦
                    if sensor.supports(rs.option.enable_auto_exposure):
                        sensor.set_option(rs.option.enable_auto_exposure, True)
                    if sensor.supports(rs.option.sharpness):
                        sensor.set_option(rs.option.sharpness, 100)
                    # 启用 Decimation Filter（降采样滤镜）
                    if sensor.supports(rs.option.filter_magnitude):
                        sensor.set_option(rs.option.filter_magnitude, 1)

            # 等待相机稳定
            # time.sleep(2)
            print("✅ 相机初始化完成（支持深度信息）")
            return True

        except Exception as e:
            print(f"⚠️ 相机初始化失败: {e}")
            self.pipeline = None
            return False


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

    # 线程
    def start_io_monitoring(self):
        """
        启动IO监控线程，监控启动/停止/复位按钮
        """
        # 初始化时停止灯亮，启动和复位灯暗

        self.urController.hll(IO_STOP,[IO_START,IO_STOP,IO_RESET]) # 停止灯亮

        self.io_monitoring = True
        self.io_monitor_thread = threading.Thread(target=self._monitor_io_buttons)
        self.io_monitor_thread.daemon = True
        self.io_monitor_thread.start()
        print("🔔 IO监控线程已启动")

    def _monitor_io_buttons(self):
        """
        监控IO按钮的线程函数
        """
        last_states = {IO_START: 0, IO_STOP: 0, IO_RESET: 0}

        while self.io_monitoring:
            try:
                # 检查启动按钮
                start_state = self.urController.get_di(IO_START, is_log=False)
                if start_state == 1 and last_states[IO_START] == 0:
                    print("🎮 检测到启动信号")
                    self._handle_start_game()
                last_states[IO_START] = start_state

                # 检查停止按钮
                stop_state = self.urController.get_di(IO_STOP, is_log=False)
                if stop_state == 1 and last_states[IO_STOP] == 0:
                    print("⏹️ 检测到停止信号")
                    self._handle_stop_game()
                last_states[IO_STOP] = stop_state

                # 检查复位按钮
                reset_state = self.urController.get_di(IO_RESET, is_log=False)
                if reset_state == 1 and last_states[IO_RESET] == 0:
                    print("🔄 检测到复位信号")
                    self._handle_reset_board()
                last_states[IO_RESET] = reset_state

                time.sleep(0.1)  # 100ms检查一次

            except Exception as e:
                print(f"⚠️ IO监控线程异常: {e}")
                time.sleep(1)

    def _handle_start_game(self):
        """
        处理启动游戏事件
        """
        # 只有在暂停状态下才能启动
        if hasattr(self, '_game_paused') and self._game_paused:
            print("🚀 继续对弈游戏")
            try:
                # 设置启动灯亮，其他灯暗
                self.urController.hll(IO_START,[IO_START,IO_STOP,IO_RESET])  # 启动灯亮，停止灯暗
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
            # 设置停止灯亮，其他灯暗
            self.urController.hll(IO_STOP,[IO_START,IO_STOP,IO_RESET])  # 启动灯暗，停止灯亮
            self._game_paused = True

            # 停止机械臂当前所有动作
            self.urController.dashboard.StopScript()

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

            # 设置复位灯闪烁，其他灯暗
            self.urController.hll()  # 所有灯先暗

            # 启动复位灯闪烁线程
            def blink_reset_light():
                for i in range(10):  # 最多闪烁10次
                    if not self._resetting:
                        break
                    self.urController.set_do(IO_RESET, 1)
                    time.sleep(0.5)
                    self.urController.set_do(IO_RESET, 0)
                    time.sleep(0.5)

            self._resetting = True
            blink_thread = threading.Thread(target=blink_reset_light)
            blink_thread.daemon = True
            blink_thread.start()

            # 执行棋盘还原成初始状态
            self.cBranch.collect_pieces_at_end()
            self.cBranch.setup_initial_board()

            # 复位完成，停止闪烁
            self._resetting = False
            blink_thread.join(timeout=1)

            # 设置复位灯亮，其他灯暗
            self.urController.hll(IO_RESET,[IO_START,IO_STOP,IO_RESET])
            asyncio.run(self.speak_cchess("棋盘已复位"))

            # 重置游戏状态
            self._game_paused = False
            if hasattr(self, '_paused_move'):
                self._paused_move = None

        except Exception as e:
            print(f"❌ 棋盘复位失败: {e}")
            self._resetting = False


    def stop_io_monitoring(self):
        """
        停止IO监控线程
        """
        self.io_monitoring = False
        if self.io_monitor_thread and self.io_monitor_thread.is_alive():
            self.io_monitor_thread.join(timeout=2)
        print("🔕 IO监控线程已停止")

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
    def play_game(self):
        """
        执行完整对弈流程（修改版）
        """
        try:
            print("🎮 开始象棋对弈...")
            self.urController.hll(IO_START,[IO_START,IO_STOP,IO_RESET])
            asyncio.run(self.speak_cchess("开始对弈，请等待指示灯为绿色再落子"))

            self._init_play_game()
            # 修改循环条件，添加投降检查
            while not self.board.is_game_over() and not self.surrendered:
                if self.surrendered:
                    return

                self.move_count += 1
                print(f"\n--- 第 {self.move_count} 回合 ---")
                if self.move_count == 1:
                    self.board = cchess.Board()
                # 判断当前回合
                is_robot_turn = (self.move_count + (0 if self.args.robot_side == 'red' else 1)) % 2 == 1

                # 1. 识别当前棋盘状态
                if self.move_count == 1:
                    asyncio.run(self.speak_cchess("正在检查棋盘初始状态，请稍等"))
                    self.cCamera.recognize_chessboard()

                    # 检查初始棋子位置
                    while  self.cUtils.compare_chessboard_positions(self.previous_positions, self.chess_positions):
                        if self.surrendered:
                            return
                        # 如果棋子位置不正确，等待玩家调整
                        self.cMove.wait_for_player_adjustment()
                    while not self.cUtils.check_all_pieces_initial_position():
                        if self.surrendered:
                            return
                        # 如果棋子位置不正确，等待玩家调整
                        self.cMove.wait_for_player_adjustment()

                if is_robot_turn:

                    self.urController.hll(5)  # 红灯
                    print(f"🤖 机器人回合")
                    self.is_playing = False
                    asyncio.run(self.speak_cchess("轮到机器人回合，请稍等"))

                    # 3. 显示当前棋盘
                    if self.args.show_board:
                        self.game.graphic(self.board)

                    # 4. 计算下一步
                    move_uci = self.cUtils.calculate_next_move()
                    if not move_uci:
                        return

                    # 6. 执行移动到棋盘对象
                    move = cchess.Move.from_uci(move_uci)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                    else:
                        asyncio.run(self.speak_cchess("机器人无法执行该移动"))
                        self.gama_over()
                        return

                    # 5. 执行移动
                    self.cMove.execute_move(move_uci)
                    self.move_history.append(move_uci)

                    print(f"当前{self.side}方")
                    self.set_side()
                    print(f"当前{self.side}方")


                    # 检查是否将军
                    if self.cUtils.is_in_check(self.board,self.side):
                        asyncio.run(self.speak_cchess("请注意，您已被将军！"))

                    self.cMove.updat_previous_positions_after_move(move_uci)
                    chinese_notation = self.cUtils.uci_to_chinese_notation(move_uci, self.previous_positions)
                    asyncio.run(self.speak_cchess(f"机器人已走子，{chinese_notation}"))

                    # 7. 显示更新后的棋盘
                    if self.args.show_board:
                        self.game.graphic(self.board)

                    print(chinese_notation)

                else:
                    print("👤 人类回合")
                    self.urController.hll(4)  # 绿灯
                    asyncio.run(self.speak_cchess("轮到您的回合，请落子"))
                    print("⏳ 等待人类落子完成信号...")

                    # 修改等待逻辑，添加投降检查
                    while not self.urController.get_di(IO_SIDE, is_log=False) and not self.surrendered:
                        self.is_playing = True
                        time.sleep(0.5)
                        if self.human_move_by_voice:
                            break
                        if self.surrendered:
                            return
                        if self.is_undo:
                            break
                    self.is_playing = False
                    if self.human_move_by_voice:
                        self.human_move_by_voice = False
                        continue
                    if self.is_undo:
                        self.is_undo = False
                        continue
                        # 检查是否投降
                    if self.surrendered:
                        self.gama_over('surrender')
                        return

                    # 复位信号
                    self.urController.hll(5)  # 红灯
                    self.io_side = self.urController.get_di(IO_SIDE)
                    print("✅ 检测到人类落子完成信号")
                    asyncio.run(self.speak_cchess("您已落子，请稍等"))

                    # 识别当前棋盘状态以更新棋盘
                    print("🔍 识别棋盘以更新状态...")
                    self.his_chessboard[self.move_count-1] = copy.deepcopy(self.previous_positions)
                    # old_positions = self.previous_positions
                    # if self.move_count == 1:
                    #     old_positions = self.chess_positions
                    for i in range(10):
                        if i > 0:
                            positions = self.cCamera.recognize_chessboard(True)
                        else:
                            positions = self.cCamera.recognize_chessboard(True)
                        # 推断人类的移动
                        self.move_uci = self.cUtils.infer_human_move(self.his_chessboard[self.move_count-1], positions)
                        if self.move_uci:
                            break
                    if self.move_uci:
                        print(f"✅ 人类推测走子: {self.move_uci}")
                        move = cchess.Move.from_uci(self.move_uci)
                        if move in self.board.legal_moves:
                            # 检查是否吃掉了机器人的将军
                            is_captured, king_side = self.cUtils.is_king_captured_by_move(self.move_uci, self.previous_positions)
                            # 如果吃掉的是机器人的将/帅
                            if is_captured and king_side == self.args.robot_side:
                                self.gama_over('player')  # 人类玩家获胜
                                asyncio.run(self.speak_cchess('吃掉了机器人的将军！'))
                                return  # 结束游戏

                            self.board.push(move)

                        else:
                            # 检查是否被将军且无法解除将军状态
                            if self.cUtils.is_in_check(self.board,self.args.robot_side):
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
                                        # asyncio.run(self.speak_cchess("将军！吃掉你的将帅！"))
                                        asyncio.run(self.speak_cchess(f"很遗憾，您输了！"))
                                        time.sleep(20)
                                        return  # 结束游戏

                            else:
                                asyncio.run(self.speak_cchess("您违规了，请重新走子"))
                                self.move_count = self.move_count - 1
                                self.urController.hll(4)  # 绿灯
                                continue
                    else:
                        print("错误！无法推断人类的移动")
                        asyncio.run(self.speak_cchess("无法检测到走棋，请重新落子"))
                        self.urController.hll(4)  # 绿灯
                        self.move_count = self.move_count - 1
                        continue

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


            # 游戏结束
            if self.board.is_game_over() or self.surrendered:
                # 如果是投降结束的游戏
                if self.surrendered:
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
                            self.gama_over()
                    else:
                        self.gama_over('平局')
        except Exception as e:
            self.report_error(str(e))
    def gama_over(self, winner='player'):
        self.urController.hll()
        game_result = ""
        result_text = ""

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
        game_records_dir = os.path.join(dir,"game_records")
        os.makedirs(game_records_dir, exist_ok=True)

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        game_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 生成对局编号
        game_id = f"game_{game_timestamp}"

        # 保存详细对局记录
        moves_filename = os.path.join(game_records_dir, "chess_moves.csv")
        summary_filename = os.path.join(game_records_dir, "chess_summary.csv")

        # 准备详细对局数据
        moves_data = []
        # 添加表头（如果文件不存在）
        if not os.path.exists(moves_filename):
            moves_data.append(["对局编号", "回合数", "UCI移动", "中文记谱", "玩家", "记录时间"])

        # 添加每步棋记录
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

            moves_data.append([game_id, i + 1, move, chinese_notation, player, timestamp])

        # 保存详细对局记录到CSV文件
        try:
            with open(moves_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(moves_data)
            print(f"💾 对局详细记录已保存至: {moves_filename}")
            self.logger.info(f"对局详细记录已保存至: {moves_filename}")

        except Exception as e:
            error_msg = f"保存对局详细记录失败: {e}"
            print(f"⚠️ {error_msg}")
            self.logger.error(error_msg)

        # 准备对局摘要数据
        summary_data = []
        # 添加表头（如果文件不存在）
        if not os.path.exists(summary_filename):
            summary_data.append(["对局编号", "游戏结果", "总回合数", "记录时间"])

        # 添加对局摘要
        summary_data.append([game_id, game_result, len(self.move_history), timestamp])

        # 保存对局摘要到CSV文件
        try:
            with open(summary_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(summary_data)
            print(f"💾 对局摘要已保存至: {summary_filename}")
            self.logger.info(f"对局摘要已保存至: {summary_filename}")

        except Exception as e:
            error_msg = f"保存对局摘要失败: {e}"
            print(f"⚠️ {error_msg}")
            self.logger.error(error_msg)
    async def save_recognition_result_with_detections(self, red_image=None, red_detections=None, black_image=None, black_detections=None,chess_result=None,move_count=None):
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

        # 创建结果目录
        result_dir = self.args.result_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        if not move_count:
            move_count = self.move_count
        async def save_red_detections():
            """异步保存红方检测结果"""
            if red_image is not None and red_detections is not None:
                red_image_with_detections = deepcopy(red_image)

                # 从Results对象中提取边界框信息
                boxes = red_detections[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())

                        # 绘制边界框
                        cv2.rectangle(red_image_with_detections, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # 添加标签
                        label = f"Red:{cls} {conf:.2f}"
                        cv2.putText(red_image_with_detections, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 保存带检测框的红方图像
                red_detected_path = os.path.join(result_dir,f"red_side_detected{move_count}.jpg")
                cv2.imwrite(red_detected_path, red_image_with_detections)
                print(f"💾 红方检测结果已保存至: {red_detected_path}")

        async def save_black_detections():
            """异步保存黑方检测结果"""
            if black_image is not None and black_detections is not None:
                black_image_with_detections = deepcopy(black_image)

                # 从Results对象中提取边界框信息
                boxes = black_detections[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())

                        # 绘制边界框
                        cv2.rectangle(black_image_with_detections, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        # 添加标签
                        label = f"Black:{cls} {conf:.2f}"
                        cv2.putText(black_image_with_detections, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # 保存带检测框的黑方图像
                black_detected_path = os.path.join(result_dir, f"black_side_detected{move_count}.jpg")
                cv2.imwrite(black_detected_path, black_image_with_detections)
                print(f"💾 黑方检测结果已保存至: {black_detected_path}")

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
            save_red_detections(),
            save_black_detections(),
            save_chessboard_layout()
        )

    # 清理
    def clear_cache(self):
        """
        清理缓存，释放内存
        """
        try:
            # 清理Python垃圾回收
            import gc
            gc.collect()

            # 如果使用了torch，清理GPU缓存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass

            print("✅ 缓存清理完成")

        except Exception as e:
            print(f"⚠️ 缓存清理时出错: {e}")


    def cleanup(self):
        """
        清理资源
        """
        try:
            self.surrendered = True
            # 停止IO监控
            self.stop_io_monitoring()

            # 断开机械臂
            try:
                if self.urController:
                    self.urController.hll()
                    print("🔌 断开机械臂连接...")
                    self.urController.disconnect()
            except Exception as e:
                print(f"⚠️ 断开机械臂连接时出错: {e}")

            # 清理相机
            self.cCamera.cleanup_camera_windows()
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                print("📷 关闭相机...")
                self.pipeline.stop()
                self.pipeline = None


            # 关闭OpenCV窗口
            if self.args.show_camera:
                cv2.destroyAllWindows()


            print("✅ 清理完成")
            asyncio.run(self.speak_cchess("结束运行"))
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
                        default='../src/cchessYolo/runs/detect/chess_piece_detection_separate/weights/best.onnx',
                        help='YOLO棋子检测模型路径')
    parser.add_argument('--play_model_file', type=str,
                        default='../src/cchessAI/models/admin/trt/current_policy_batch7483_202507170806.trt',
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
        asyncio.run(initialize_components())

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
