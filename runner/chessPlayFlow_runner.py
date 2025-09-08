# file: /media/jetson/KESU/code/Embodied/src/cchessAI/chessPlayFlow_runner.py

import argparse
import asyncio
import copy
import logging
import queue
import threading
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np

from api.services.tts_service import TTSManager
from utils.calibrationManager import chess_to_world_position, calculate_perspective_transform_matrices, \
    multi_camera_pixel_to_world, get_area_center
from dobot.dobot_control import connect_and_check_speed
from parameters import RED_CAMERA, BLACK_CAMERA, POINT_DOWN, IO_QI, IO_SIDE, \
    WORLD_POINTS_RCV, POINT_RCV_DOWN, RCV_CAMERA, SAC_CAMERA, SRC_SAC_POINTS, SRC_RCV_POINTS, \
    DST_SAC_POINTS, DST_RCV_POINTS, RCV_H_LAY, SAC_H_LAY, POINT_SAC_DOWN, CHESS_POINTS_R, CHESS_POINTS_B, \
    WORLD_POINTS_R, WORLD_POINTS_B, CHESS_POINTS_RCV_H, CHESS_POINTS_RCV_L
from src.cchessAG.chinachess import MainGame
from src.cchessAI import cchess
from src.cchessYolo.detect_chess_box import select_corner_circles, order_points, calculate_box_corners, \
    calculate_4x4_collection_positions
from utils.corrected import get_corrected_chessboard_points, correct_chessboard_to_square
from utils.tools import move_id2move_action

# 添加项目路径到PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.cchessAI.core.game import Game, algebraic_to_coordinates, get_best_move_with_computer_play, \
    execute_computer_move
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

class ChessPlayFlow():
    def __init__(self, args):
        self.args = args
        self.urController = None
        self.detector = None
        self.board = cchess.Board()
        self.game = Game(self.board)
        self.board = cchess.Board()
        self.move_history =  []
        self.mcts_player = None
        self.human_player = None
        self.side = 'red'  # 开始棋子方为红方
        self.point_home = self.args.red_camera_position
        self.voice_engine_type = "edge"
        self.pipeline = None
        self.chessboard_image = None
        self.surrendered = False  # 添加投降标志

        # 棋盘状态
        self.is_chesk = False
        self.sac_nums = 0
        self.move_uci = ''                                  # 棋子移动 使用简谱坐标系
        # 棋子映射字典
        self.piece_map = {
            'r': '车', 'n': '马', 'b': '象', 'a': '士', 'k': '将', 'c': '炮', 'p': '卒',  # 黑方
            'R': '車', 'N': '馬', 'B': '相', 'A': '仕', 'K': '帥', 'C': '砲', 'P': '兵'   # 红方
        }
        self.his_chessboard = {} # 历史棋盘
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
        self.previous_positions = self.chess_positions  # 使用数组坐标系 # 识别棋盘

        # 初始化语音进程管理列表
        self.voice_executor = ThreadPoolExecutor(max_workers=2)
        self.voice_loop = None
        self.voice_thread = None
        self._init_voice_async_loop()

        # 语音队列和状态管理
        self.speech_queue = queue.Queue()  # 语音播报队列
        self.is_speaking = False  # 当前是否正在播报
        self.speech_lock = threading.Lock()  # 语音播报锁
        self.speech_thread = None  # 语音处理线程
        self.voice_cache = {}  # 添加语音缓存
        self.voice_cache_dir = "voice_cache"
        os.makedirs(self.voice_cache_dir, exist_ok=True)

        # 计算透视变换矩阵
        self.m_sac = cv2.getPerspectiveTransform(SRC_SAC_POINTS, DST_SAC_POINTS)
        self.m_rcv = cv2.getPerspectiveTransform(SRC_RCV_POINTS, DST_RCV_POINTS)
        self.rcv_h_lay = RCV_H_LAY # 收子分层高度
        self.sac_v_lay = SAC_H_LAY # 弃子分层高度

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

    # 初始化
    def initialize(self):
        """
        初始化所有组件
        """
        print("🔧 开始初始化...")

        # 初始化语音引擎
        try:
            # 初始化统一的TTS管理器
            self.tts_manager = TTSManager()
            self.speak("开始初始化系统")
        except Exception as e:
            print(f"⚠️ 语音引擎初始化失败: {e}")
            self.voice_engine = None


        # 1. 连接机械臂
        print("🤖 连接机械臂...")
        # self.speak("正在连接机械臂")
        try:
            self.urController = connect_and_check_speed(
                ip=self.args.robot_ip,
                port=self.args.robot_port,
                dashboard_port=self.args.robot_dashboard_port,
                feed_port=self.args.robot_feed_port,
            )
        except Exception as e:
            print(f"⚠️ 连接机械臂失败: {e}")
            self.speak("连接机械臂失败")
            raise Exception(f"机械臂连接失败{e}")

        if not self.urController:
            self.speak("机械臂连接失败")
            raise Exception("机械臂连接失败")

        if not self.urController.is_connected():
            self.speak("机械臂连接失败")
            raise Exception("机械臂连接失败")

        self.speak("机械臂连接成功")
        self.urController.set_speed(0.8)
        # 移动到初始位置
        self.urController.run_point_j(self.args.red_camera_position)
        self.urController.hll()
        # 2. 初始化相机
        print("📷 初始化相机...")
        # self.speak("正在初始化相机")
        self.init_camera()
        if self.pipeline is None:
            self.speak("相机初始化失败,请检查相机连接")

        # 3. 打开识别模型 (使用 YOLO 检测器)
        print("👁️ 初始化棋子识别模型...")
        self.speak("正在加载识别模型")
        try:
            self.detector = ChessPieceDetectorSeparate(
                model_path=self.args.yolo_model_path
            )
        except Exception as e:
            print(f"⚠️识别模型初始化失败: {e}")
            self.speak("识别模型初始化失败")
            raise Exception("识别模型初始化失败")

        # 4. 打开对弈模型
        print("🧠 初始化对弈模型...")
        self.speak("正在加载对弈模型")
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
                    self.speak("对弈模型初始化失败")
                    raise Exception("对弈模型初始化失败")
                time.sleep(2)  # 等待后重试

        # 5. 初始化棋盘
        self.initialize_chessboard_points()

        # 显示初始棋盘
        if self.args.show_board:
            self.game.graphic(self.board)

        self.speak("系统初始化完成")

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


    def _init_voice_async_loop(self):
        """
        初始化异步语音播报的事件循环
        """

        def run_loop():
            self.voice_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.voice_loop)
            self.voice_loop.run_forever()

        self.voice_thread = threading.Thread(target=run_loop, daemon=True)
        self.voice_thread.start()

        # 等待循环初始化完成
        while self.voice_loop is None:
            time.sleep(0.01)
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

    # 相机
    def capture_stable_image(self, num_frames=5, is_chessboard=False):
        """
        捕获稳定的图像和深度信息（通过多帧平均减少噪声）

        Args:
            num_frames: captured帧数用于平均
            is_chessboard: 是否为棋盘图像，需要进行畸变矫正

        Returns:
            tuple: (稳定图像, 深度帧)
        """
        max_retry_attempts = 100  # 最大重试次数
        retry_count = 0

        while retry_count < max_retry_attempts:
            if self.surrendered:
                return

            if not hasattr(self, 'pipeline') or self.pipeline is None:
                # 尝试重新初始化相机
                self.speak("相机未连接，正在重新连接相机")
                self.init_camera()

                if not hasattr(self, 'pipeline') or self.pipeline is None:
                    retry_count += 1
                    self.speak(f"相机连接失败，{retry_count}秒后重试")
                    time.sleep(5)
                    continue

            try:
                frames_list = []
                depth_frames_list = []

                # 捕获多帧图像
                for i in range(num_frames):
                    frames = self.pipeline.wait_for_frames(timeout_ms=5000)  # 设置超时时间
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()

                    if color_frame and depth_frame:
                        frame = np.asanyarray(color_frame.get_data())
                        frames_list.append(frame)
                        depth_frames_list.append(depth_frame)
                    else:
                        continue

                    # 短暂等待
                    time.sleep(0.1)

                if not frames_list:
                    raise Exception("无法捕获有效图像帧")

                # 如果只捕获到一帧，直接返回
                if len(frames_list) == 1:
                    result_frame = frames_list[0]
                    latest_depth_frame = depth_frames_list[0]
                else:
                    # 多帧平均以减少噪声（仅对彩色图像）
                    result_frame = np.mean(frames_list, axis=0).astype(np.uint8)
                    # 使用最新的深度帧
                    latest_depth_frame = depth_frames_list[-1]

                world_r = WORLD_POINTS_R
                world_b = WORLD_POINTS_B
                world_rcv = WORLD_POINTS_RCV
                self.chess_r = CHESS_POINTS_R
                self.chess_b = CHESS_POINTS_B
                self.chess_rcv_h = CHESS_POINTS_RCV_H
                self.chess_rcv_l = CHESS_POINTS_RCV_L

                # 畸变矫正
                if is_chessboard:
                    self.chess_r, self.m_R = get_corrected_chessboard_points(CHESS_POINTS_R)
                    self.chess_b, self.m_B = get_corrected_chessboard_points(CHESS_POINTS_B)
                    self.chess_rcv_h, self.m_RCV_h = get_corrected_chessboard_points(CHESS_POINTS_RCV_H)
                    self.chess_rcv_l, self.m_RCV_l = get_corrected_chessboard_points(CHESS_POINTS_RCV_L)

                    if self.side == 'red':
                        result_frame, _ = correct_chessboard_to_square(result_frame, CHESS_POINTS_R, self.m_R)
                    else:
                        result_frame, _ = correct_chessboard_to_square(result_frame, CHESS_POINTS_B, self.m_B)

                self.forward_matrix_r, self.inverse_matrix_r = calculate_perspective_transform_matrices(world_r, self.chess_r)
                self.forward_matrix_b, self.inverse_matrix_b = calculate_perspective_transform_matrices(world_b, self.chess_b)
                self.forward_matrix_rcv_h, self.inverse_matrix_rcv_h = calculate_perspective_transform_matrices(world_rcv, self.chess_rcv_h)
                self.forward_matrix_rcv_l, self.inverse_matrix_rcv_l = calculate_perspective_transform_matrices(world_rcv, self.chess_rcv_l)
                if retry_count > 0:
                    self.speak(f"相机图像获取成功")

                return result_frame, latest_depth_frame

            except Exception as e:

                retry_count += 1
                error_msg = f"捕获图像失败，第{retry_count}次重试"
                print(f"⚠️ {error_msg}: {e}")
                self.speak(error_msg)

                # 如果达到最大重试次数，停止重试
                if retry_count >= max_retry_attempts:
                    self.speak("已达最大重试次数，无法获取图像")
                    break

                # 等待一段时间后重试
                time.sleep(3)

                # 尝试重新初始化相机
                self.speak("正在重新初始化相机")
                self.pipeline = None
                self.init_camera()

        # 如果所有重试都失败，返回None
        self.speak("无法捕获稳定图像，请检查相机连接")
        return None, None
    # 识别
    def detect_chess_box(self, max_attempts=20):
        """
        识别棋盒位置，支持3个或4个圆角标记

        Args:
            max_attempts: 最大尝试次数

        Returns:
            list: 棋盒角点坐标列表，如果无法识别则返回None
        """
        print("🔍 寻找棋盒位置...")
        chess_box_points = None

        for attempt in range(max_attempts):
            print(f"🔍 尝试识别棋盒位置 {attempt + 1}/{max_attempts}...")
            # 捕获图像
            rcv_image, rcv_depth = self.capture_stable_image()
            if rcv_image is None:
                print("⚠️ 无法捕获收子区图像")
                continue

            # 使用霍夫圆检测来识别棋盒的圆形标记
            gray = cv2.cvtColor(rcv_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)

            # 霍夫圆检测
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=50,
                param2=30,
                minRadius=10,
                maxRadius=50
            )

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")

                # 检查是否检测到至少3个圆
                if len(circles) >= 3:
                    # 如果检测到4个或更多圆，选择4个最可能的角点圆
                    if len(circles) >= 4:
                        selected_circles = select_corner_circles(circles)
                        if len(selected_circles) >= 3:  # 至少需要3个圆
                            # 按顺序排列圆心点
                            centers = [(int(circle[0]), int(circle[1])) for circle in selected_circles[:4]]
                            ordered_centers = order_points(np.array(centers))

                            # 计算平均半径
                            radii = [int(circle[2]) for circle in selected_circles[:4]]
                            avg_radius = int(np.mean(radii))

                            # 计算棋盒的实际角点
                            chess_box_points = calculate_box_corners(ordered_centers, avg_radius)
                    else:
                        # 只检测到3个圆的情况
                        centers = [(int(circle[0]), int(circle[1])) for circle in circles]
                        # 简单按x,y坐标排序
                        centers.sort(key=lambda c: (c[0], c[1]))

                        # 估算第4个点来构成矩形
                        if len(centers) == 3:
                            # 基于3个点估算第4个点
                            # 假设这3个点形成一个直角三角形，计算第4个点
                            pts = np.array(centers)
                            # 计算距离矩阵找最远的两个点作为对角点
                            distances = np.sqrt(((pts[:, None] - pts)**2).sum(axis=2))
                            i, j = np.unravel_index(distances.argmax(), distances.shape)

                            # 第4个点为其他两点的对称点
                            missing_point = (int(pts[i][0] + pts[j][0] - pts[6 - i - j][0]),
                                             int(pts[i][1] + pts[j][1] - pts[6 - i - j][1]))
                            centers.append(missing_point)

                            ordered_centers = order_points(np.array(centers))
                            avg_radius = int(np.mean([int(circle[2]) for circle in circles]))
                            chess_box_points = calculate_box_corners(ordered_centers, avg_radius)

                    if chess_box_points is not None and len(chess_box_points) >= 4:
                        break

            time.sleep(0.5)

        return chess_box_points

    def recognize_chessboard(self,is_run_red=False):
        """
        识别整个棋盘状态 (使用 YOLO 检测器，包含高度信息)
        """
        print("🔍 开始识别棋盘...")

        if self.surrendered:
            return

        # 创建结果目录
        if self.args.save_recognition_results:
            result_dir = self.args.result_dir
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

        # 初始化棋子像素坐标存储
        self.piece_pixel_positions = {}

        # 识别红方半区
        print("🔴 识别红方半区...")
        if is_run_red:
#             self.urController.set_speed(0.8)
            self.urController.run_point_j(self.args.red_camera_position)
            # time.sleep(3)  # 等待稳定

        # 多次捕获取最佳图像和深度信息
        red_image, red_depth = self.capture_stable_image()
        if red_image is None:
            print("⚠️ 无法捕获红方图像")
            return self.chess_positions

        if self.args.show_camera:
            cv2.imshow("Red Side", red_image)
            cv2.waitKey(1)

        # 识别红方半区棋子 (使用 YOLO，包含高度信息)
        red_result, red_detections,points_center = self.detector.extract_chessboard_layout_with_height(
            red_image, self.chess_r,half_board="red",
            conf_threshold=self.args.conf,
            iou_threshold=self.args.iou
        )


        if points_center:
            self.piece_pixel_positions.update(points_center)

        # 识别黑方半区
        print("⚫ 识别黑方半区...")
#         self.urController.set_speed(0.8)
        self.urController.run_point_j(self.args.black_camera_position)
        # time.sleep(3)  # 等待稳定
#         self.urController.set_speed(0.5)

        # 多次捕获取最佳图像和深度信息
        black_image, black_depth = self.capture_stable_image()
        if black_image is None:
            print("⚠️ 无法捕获黑方图像")
            return self.chess_positions

        if self.args.show_camera:
            cv2.imshow("Black Side", black_image)
            cv2.waitKey(1)

        # 识别黑方半区棋子 (使用 YOLO，包含高度信息)
        black_result, black_detections,points_center = self.detector.extract_chessboard_layout_with_height(
            black_image, self.chess_b,half_board="black",
            conf_threshold=self.args.conf,
            iou_threshold=self.args.iou
        )

        if points_center:
            self.piece_pixel_positions.update(points_center)

        # 合并结果 (黑方在0-4行，红方在5-9行，且红方需要倒置)
        chess_result = [['.' for _ in range(9)] for _ in range(10)]

        # 黑方半区放在棋盘的0-4行
        for row in range(5):  # 黑方半区 0-4行
            for col in range(9):
                chess_result[row][col] = black_result[row][col]

        # 红方半区放在棋盘的5-9行，并进行倒置处理
        for row in range(5):  # 红方半区原始为0-4行
            for col in range(9):
                # 红方需要倒置，所以(0,0)变成(9,8)
                chess_result[9-row][8-col] = red_result[row][col]
        self.chess_positions = chess_result


        # 保存识别结果（包括可视化检测结果）
        if self.args.save_recognition_results:
            self.save_recognition_result_with_detections(
                chess_result, red_image, red_detections, black_image, black_detections
            )

        print("✅ 棋盘识别完成")
        return chess_result

    # 移动
    def move_home(self,from_col=0,type='poi'):
        if from_col >=5:
            if type=='cam':
                self.point_home = self.args.black_camera_position
            elif type=='poi':
                self.point_home = self.args.black_position
        else:
            self.point_home = self.args.red_camera_position
        self.urController.point_o = self.point_home
        self.urController.run_point_j(self.point_home)

    def point_move(self,from_point,to_point,home_row=[0,0]):
        """移动棋子"""
        from_x_world, from_y_world, pick_height = from_point
        to_x_world, to_y_world, place_height = to_point
        from_row , to_row = home_row

        # 移动到起始位置上方 (使用安全高度) 到不了角落点的上方
#         self.urController.set_speed(0.8)
        self.move_home(from_row)
        # time.sleep(3)

        # 降低到吸取高度
        print("👇 降低到吸取高度")
#         self.urController.set_speed(0.5)
        self.urController.move_to(from_x_world, from_y_world, pick_height+15, use_safety=False)
        # time.sleep(1)
        self.urController.move_to(from_x_world, from_y_world, pick_height, use_safety=False)
#         time.sleep(1)

        # 吸取棋子
        print("🫳 吸取棋子")
        self.urController.set_do(IO_QI, 1)  # 吸合
#         time.sleep(1)
        self.urController.move_to(from_x_world, from_y_world, pick_height+15, use_safety=False)
#         time.sleep(1)

        # 抬起棋子到安全高度
        print("👆 抬起棋子到安全高度")
#         self.urController.set_speed(0.8)
        self.move_home(from_row)
#         time.sleep(1)


        # 移动到目标位置上方（使用安全高度）
        print(f"📍 移动到目标位置上方: ({to_x_world}, {to_y_world})")
        self.move_home(to_row)
#         time.sleep(1)

        # 降低到放置高度
        print("👇 降低到放置高度")
#         self.urController.set_speed(0.5)
        self.urController.move_to(to_x_world, to_y_world, POINT_RCV_DOWN[2])
#         time.sleep(1)

        self.urController.move_to(to_x_world, to_y_world, place_height)
#         time.sleep(1)

        # 放置棋子
        print("🤲 放置棋子")
        self.urController.set_do(IO_QI, 0)  # 释放
#         time.sleep(1)
        self.urController.move_to(to_x_world, to_y_world, POINT_RCV_DOWN[2])
#         time.sleep(1)

    def execute_move(self, move_uci):
        """
        执行移动操作前检查目标位置及周围位置的偏差

        Args:
            move_uci: 移动的UCI表示
        """
        print(f"🦾 执行移动: {move_uci}")
        pick_height = POINT_DOWN[0]

        # 解析移动 (UCI格式: 列行列表行) 简谱坐标系
        from_col = ord(move_uci[0]) - ord('a')  # 0-8 (a-i)
        from_row = int(move_uci[1])  # 0-9 (0-9)
        to_col = ord(move_uci[2]) - ord('a')  # 0-8 (a-i)
        to_row = int(move_uci[3])  # 0-9 (0-9)

        # 转换为数组行索引 数组坐标系
        from_row_idx = 9 - from_row
        to_row_idx = 9 - to_row

        # 检查目标位置及周围位置的偏差，如果有偏差超过容忍度则不断重新检查直到没有偏差为止
        print("🔍 检查目标位置及周围棋子位置偏差...")
        while not self.check_target_position_and_surroundings(from_row,from_col,to_row, to_col):
            if self.surrendered:
                return

            self.wait_for_player_adjustment()

                # 检查是否投降
            if self.surrendered:
                self.gama_over('surrender')
                return

        # 将棋盘坐标转换为世界坐标
        # 使用存储的像素坐标来提高精度
        piece_key = f"{from_row_idx}{from_col}"  # 使用数组索引
        if piece_key in self.piece_pixel_positions:
            # 使用之前识别的精确像素坐标
            pixel_x, pixel_y = self.piece_pixel_positions[piece_key]

            # 根据半区类型转换为世界坐标
            if from_row <= 4:  # 判断是红方还是黑方半区
                from_x_world, from_y_world = multi_camera_pixel_to_world(pixel_x, pixel_y, self.inverse_matrix_r,
                                                                         "RED_CAMERA")
            else:
                from_x_world, from_y_world = multi_camera_pixel_to_world(pixel_x, pixel_y, self.inverse_matrix_b,
                                                                         "BLACK_CAMERA")
            print('像素坐标：', pixel_x, pixel_y)
        else:
            # 如果没有存储的像素坐标，则使用原来的计算方法作为备选
            if from_row <= 4:
                half_board = 'red'
            else:
                half_board = 'black'
            from_x_world, from_y_world = chess_to_world_position(from_col, from_row, half_board)

        # 目标位置世界坐标转换
        if to_row <= 4:
            half_board = 'red'
        else:
            half_board = 'black'
        to_x_world, to_y_world = chess_to_world_position(to_col, to_row, half_board)
        print('世界坐标：', from_x_world, from_y_world, " to ", to_x_world, to_y_world)

        # 检查目标位置是否有棋子（即将被吃掉）
        target_piece_key = f"{to_row_idx}{to_col}"
        if self.chess_positions[to_row_idx][to_col] != '.':
            captured_piece = self.chess_positions[to_row_idx][to_col]
            print(f"⚔️ 吃掉棋子: {self.piece_map[captured_piece]}")

            # 记录被吃的棋子信息，用于悔棋时恢复
            self.captured_pieces_history[target_piece_key] = {
                'piece': captured_piece,
                'move': move_uci,
                'position': (to_row_idx, to_col)
            }

            # 移动被吃的棋子到弃子区
            self.move_piece_to_area(to_row_idx, to_col)

        # 移动棋子
        self.point_move([from_x_world, from_y_world, pick_height],
                        [to_x_world, to_y_world, pick_height],
                        [from_row, to_row])

        # 回到初始位置
        print("🏠 返回初始位置")
#         self.urController.set_speed(0.5)
        self.move_home()
        print("✅ 移动执行完成")

        if self.args.use_api:
            # 报告机器人移动
            chinese_notation = self.uci_to_chinese_notation(move_uci, self.chess_positions)
            self.report_move("robot", move_uci, chinese_notation)

    def move_piece_to_area(self, row, col):
        """
        移动被吃的棋子到弃子区域的空位

        Args:
            row: 棋子所在行
            col: 棜子所在列
        """
        pick_height = POINT_DOWN[0]
        piece_key = f"{row}{col}"
        pixel_x, pixel_y = self.piece_pixel_positions[piece_key]

        # 根据半区类型转换为世界坐标
        camera_type = "RED_CAMERA" if (9-row) <= 4 else "BLACK_CAMERA"
        inverse_matrix = self.inverse_matrix_r if  (9-row) <= 4 else self.inverse_matrix_b
        from_x_world, from_y_world = multi_camera_pixel_to_world(pixel_x, pixel_y,inverse_matrix, camera_type)
        print('像素坐标：', pixel_x, pixel_y)

        # 计算弃子区域偏移位置
        side = 30
        offset_map = {
            1: (-side, +side),
            2: (+side, +side),
            3: (+side, -side),
            4: (-side, -side)
        }
        mod = self.sac_nums % 5
        if mod in offset_map:
            dx, dy = offset_map[mod]
            sac_camera = [SAC_CAMERA[0] + dx, SAC_CAMERA[1] + dy] + SAC_CAMERA[2:]
        else:
            sac_camera = SAC_CAMERA

        to_x_world, to_y_world = sac_camera[0], sac_camera[1]
        place_height = POINT_SAC_DOWN[1]

        # 使用 point_move 函数执行移动操作
        self.point_move(
            [from_x_world, from_y_world, pick_height],
            [to_x_world, to_y_world, place_height],
            [9-row, row]  # home_row 参数，用于控制 move_home 的行为
        )

        # 复位到标准弃子区域中心点上方
#         self.urController.set_speed(0.5)
        self.urController.run_point_j(SAC_CAMERA)
        self.sac_nums += 1

    # 棋盘
    def visualize_chessboard(self, chess_result):
        """
        可视化棋盘布局

        Args:
            chess_result: 棋盘状态二维数组

        Returns:
            numpy数组: 可视化的棋盘图像
        """
        # 创建一个空白图像 (500x500 pixels)
        board_size = 500
        cell_size = board_size // 10  # 每个格子的大小
        img = np.ones((board_size, board_size, 3), dtype=np.uint8) * 255  # 白色背景

        # 绘制棋盘网格
        for i in range(11):  # 10行+1
            # 横线
            cv2.line(img, (0, i * cell_size), (9 * cell_size, i * cell_size), (0, 0, 0), 1)
            if i < 10:  # 竖线
                cv2.line(img, (i * cell_size, 0), (i * cell_size, 10 * cell_size), (0, 0, 0), 1)

        # 绘制九宫格斜线
        # 红方九宫格
        cv2.line(img, (3 * cell_size, 0), (5 * cell_size, 2 * cell_size), (0, 0, 0), 1)
        cv2.line(img, (5 * cell_size, 0), (3 * cell_size, 2 * cell_size), (0, 0, 0), 1)

        # 黑方九宫格
        cv2.line(img, (3 * cell_size, 7 * cell_size), (5 * cell_size, 9 * cell_size), (0, 0, 0), 1)
        cv2.line(img, (5 * cell_size, 7 * cell_size), (3 * cell_size, 9 * cell_size), (0, 0, 0), 1)


        # 在对应位置绘制棋子
        for row in range(10):
            for col in range(9):
                piece = chess_result[row][col]
                if piece != '.':
                    # 计算棋子中心位置
                    center_x = int(col * cell_size + cell_size // 2)
                    center_y = int(row * cell_size + cell_size // 2)

                    # 绘制棋子圆形
                    is_red = piece.isupper()  # 大写为红方
                    color = (0, 0, 255) if is_red else (0, 0, 0)  # 红方用红色，黑方用黑色
                    cv2.circle(img, (center_x, center_y), cell_size // 2 - 5, color, -1)
                    cv2.circle(img, (center_x, center_y), cell_size // 2 - 5, (0, 0, 0), 2)

                    # 绘制棋子文字
                    # text = piece_map.get(piece, piece)
                    text = piece
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    text_x = center_x - text_size[0] // 2
                    text_y = center_y + text_size[1] // 2
                    text_color = (255, 255, 255) if is_red else (255, 255, 255)  # 白色文字
                    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        return img
    def _calculate_piece_deviation(self, row, col, pixel_x, pixel_y,tolerance=10):
        """
        计算单个棋子位置偏差的通用函数

        Args:
            row: 棋子行号 (0-9) 简谱坐标系
            col: 棋子列号 (0-8) 简谱坐标系
            pixel_x: 棋子像素坐标x
            pixel_y: 棋子像素坐标y

        Returns:
            dict: 包含实际位置、标准位置和偏差信息的字典
        """
        # 转换为世界坐标
        if row <= 4:  # 红方区域(0-4行)
            x_world, y_world = multi_camera_pixel_to_world(pixel_x, pixel_y, self.inverse_matrix_r, "RED_CAMERA")
            half_board = "red"
        else:  # 黑方区域(5-9行)
            x_world, y_world = multi_camera_pixel_to_world(pixel_x, pixel_y, self.inverse_matrix_b, "BLACK_CAMERA")
            half_board = "black"

        # 计算标准位置的世界坐标
        standard_x, standard_y = chess_to_world_position(col, row, half_board)

        # 计算偏差距离
        cx = round(x_world - standard_x,2)
        cy = round(y_world - standard_y,2)
        distance = np.sqrt((x_world - standard_x)**2 + (y_world - standard_y)**2)
        is_deviation_exceeded = distance > tolerance
        # if row == 0:
        #     print('测试',col,distance,x_world - standard_x,y_world - standard_y)
        #     if y_world - standard_y >= 1:
        #         is_deviation_exceeded = True
        return {
            'world_position': (x_world, y_world),
            'standard_position': (standard_x, standard_y),
            'deviation_x':cx,
            'deviation_y':cy,
            'distance':distance,
            'is_deviation_exceeded': is_deviation_exceeded
        }

    def check_target_position_and_surroundings(self, from_row, from_col, target_row, target_col, tolerance=40):
        """
        检查目标位置及周围位置的棋子是否偏离标准位置，以及棋子之间距离是否过近

        Args:
            from_row: 起始行 (0-9) 简谱坐标系
            from_col: 起始列 (0-8) 简谱坐标系
            target_row: 目标行 (0-9) 简谱坐标系
            target_col: 目标列 (0-8) 简谱坐标系
            tolerance: 偏差容忍度(mm)

        Returns:
            bool: True表示没有问题，False表示存在问题
        """
        # 定义要检查的位置：目标位置及其周围8个位置
        surrounding_positions = [
            (target_row, target_col + 1),  # 上方
            (target_row - 1, target_col),  # 左侧
            (target_row + 1, target_col),  # 右侧
            (target_row, target_col - 1),  # 下方
        ]

        # 从检查位置中移除起始位置（如果存在）
        if (from_row, from_col) in surrounding_positions:
            surrounding_positions.remove((from_row, from_col))

        # 收集所有相关位置的棋子世界坐标
        piece_world_positions = {}

        # 先收集目标位置和周围位置的棋子世界坐标
        for row, col in surrounding_positions:
            row_idx = 9 - row
            piece_key = f"{row_idx}{col}"
            # 检查该位置是否有棋子
            if piece_key in self.piece_pixel_positions:
                # 获取当前棋子的实际位置
                pixel_x, pixel_y = self.piece_pixel_positions[piece_key]

                # 转换为世界坐标
                if row <= 4:  # 红方区域(0-4行)
                    x_world, y_world = multi_camera_pixel_to_world(pixel_x, pixel_y, self.inverse_matrix_r,
                                                                   "RED_CAMERA")
                else:  # 黑方区域(5-9行)
                    x_world, y_world = multi_camera_pixel_to_world(pixel_x, pixel_y, self.inverse_matrix_b,
                                                                   "BLACK_CAMERA")

                piece_world_positions[(row, col)] = (x_world, y_world)

                # # 使用通用函数计算偏差
                # deviation_data = self._calculate_piece_deviation(row, col, pixel_x, pixel_y, tolerance)
                # deviation_info[(row, col)] = deviation_data
                #
                # # 如果偏差超过容忍度，给出警告
                # if deviation_data['is_deviation_exceeded']:
                #     print(
                #         f"⚠️ 棋子({row_idx+1},{col+1})偏离标准位置X方向{abs(deviation_data['world_position'][0] - deviation_data['standard_position'][0]):.2f}mm，Y方向{abs(deviation_data['world_position'][1] - deviation_data['standard_position'][1]):.2f}mm，超过{tolerance}mm阈值")
                #     self.speak(
                #         f"第{row_idx+1}行,第{col+1}列的棋子偏离标准位置")

        # 检查目标位置与周围棋子之间的距离，防止落子时碰撞
        # 目标位置世界坐标转换
        if target_row <= 4:
            half_board = 'red'
        else:
            half_board = 'black'
        x_world, y_world = chess_to_world_position(target_col, target_row, half_board)

        # 检查与周围棋子的距离
        for row, col in piece_world_positions.keys():
            neighbor_x, neighbor_y = piece_world_positions[(row, col)]
            # 计算与周围棋子的距离
            distance = np.sqrt((x_world - neighbor_x) ** 2 + (y_world - neighbor_y) ** 2)

            # 如果最近的棋子距离小于容忍度，发出警告并报告
            if distance < tolerance:
                row_idx = 9 - row
                point_type = self.piece_map[self.chess_positions[row_idx][col]]
                print(f"⚠️ 第({row_idx + 1},{col + 1})的{point_type}距离过近: {distance:.2f}mm，可能造成碰撞")

                if target_row > row :
                    text = f"请将第{row_idx + 1}行,第{col + 1}列的{point_type}向下移动"
                elif target_row < row :
                    text = f"请将第{row_idx + 1}行,第{col + 1}列的{point_type}向上移动"
                elif target_col > col :
                    text = f"请将第{row_idx + 1}行,第{col + 1}列的{point_type}向左移动"
                elif target_col < col :
                    text = f"请将第{row_idx + 1}行,第{col + 1}列的{point_type}向右移动"

                if self.args.use_api:
                    # 报告偏移信息
                    deviation_x = abs(x_world - neighbor_x)
                    deviation_y = abs(y_world - neighbor_y)
                    self.report_piece_deviation(row_idx, col, deviation_x, deviation_y, distance)

                self.speak(text)

                return False

        return True
    def check_all_pieces_initial_position(self, tolerance=10):
        """
        检查初始状态下所有棋子是否在正确位置上

        Args:
            tolerance: 偏差容忍度(mm)

        Returns:
            bool: True表示所有棋子都在正确位置上，False表示有偏差
        """
        print("🔍 检查初始棋子位置...")
        self.speak("正在检查棋子是否偏移")

        pieces_with_deviation = []
        er_points = []

        # 遍历所有棋子位置
        for piece_key, (pixel_x, pixel_y) in self.piece_pixel_positions.items():
            # 解析棋子位置
            row_idx = int(piece_key[0])
            row = 9 - row_idx
            col = int(piece_key[1])
            point_type = self.piece_map[self.chess_positions[row_idx][col]]

            # 使用通用函数计算偏差
            deviation_data = self._calculate_piece_deviation(row, col, pixel_x, pixel_y, tolerance)

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

                if self.args.use_api:
                    # 报告偏移棋子信息
                    self.report_piece_deviation(
                        row_idx,
                        col,
                        deviation_data['deviation_x'],
                        deviation_data['deviation_y'],
                        deviation_data['distance']
                    )

        # 如果有偏差的棋子，报告详细信息
        if pieces_with_deviation:
            print(f"❌ 发现{len(pieces_with_deviation)}个棋子位置不正确")
            self.speak(f"发现{len(pieces_with_deviation)}个棋子偏离标准位置")
            for i in range(len(pieces_with_deviation)):
                point_type = self.piece_map[self.chess_positions[er_points[i][0] - 1][er_points[i][1] - 1]]
                self.speak(f"第{er_points[i][0]}行,第{er_points[i][1]}列的{point_type}")
                if i > 3:
                     break
            return False
        else:
            print("✅ 所有棋子都在正确位置上")
            # self.speak("所有棋子位置正确")
            return True
    def wait_for_player_adjustment(self):
        """
        等待玩家调整棋子位置
        """
        # print("⏳ 等待玩家调整棋子位置...")
        # # self.speak("请调整棋子位置")
        # # 等待一段时间让玩家有时间调整
        # time.sleep(12)
        # print("🔍 重新检测棋盘...")
        # # 重新识别棋盘状态
        # self.recognize_chessboard(True)
        # # self.speak("正在重新检测棋盘")

        while not self.urController.get_di(IO_SIDE, is_log=False) and not self.surrendered:
            time.sleep(0.5)
            if self.surrendered:
                return


    # 分支函数
    # 收棋
    def collect_pieces_at_end(self):
        """
        收局函数：识别棋盒位置，然后将所有棋子按颜色分类放入棋盒
        """
        try:
            print("🧹 开始收局...")
            self.speak("开始收局")

            while 1:
#                 self.urController.set_speed(0.8)
                self.urController.run_point_j(RCV_CAMERA)
                # time.sleep(3)

                # 1. 识别棋盒位置（支持3或4个圆）
                chess_box_points = self.detect_chess_box(max_attempts=20)

                # 如果无法识别到棋盒位置，则报错
                if chess_box_points is None:
                    print("无法识别棋盒位置")
                    self.speak("无法识别棋盒位置")
                    time.sleep(10)
                    return

                print("✅ 成功识别棋盒位置")
                self.speak("成功识别棋盒位置")

                chess_box_points = np.array([[point[0]+40,point[1]+40] for point in chess_box_points])

                # 转换为世界坐标检查尺寸 注意镜像翻转
                world_corner_0 = multi_camera_pixel_to_world(chess_box_points[2][0], chess_box_points[2][1], self.inverse_matrix_r, "RCV_CAMERA") # 棋盒左上角
                world_corner_1 = multi_camera_pixel_to_world(chess_box_points[3][0], chess_box_points[3][1], self.inverse_matrix_r,  "RCV_CAMERA") # 棋盒右上角
                world_corner_2 = multi_camera_pixel_to_world(chess_box_points[0][0], chess_box_points[0][1], self.inverse_matrix_r, "RCV_CAMERA") # 棋盒右下角
                world_corner_3 = multi_camera_pixel_to_world(chess_box_points[1][0], chess_box_points[1][1], self.inverse_matrix_r, "RCV_CAMERA") # 棋盒左下角

                cx = 0
                cy = 0
                topLeft = world_corner_0[0]  , world_corner_0[1]
                topRight = world_corner_1[0]  , world_corner_1[1]
                bottomRight = world_corner_2[0]  , world_corner_2[1]
                bottomLeft = world_corner_3[0] , world_corner_3[1]
                chess_box_points = [topLeft, topRight, bottomRight, bottomLeft]

                if not self.urController.is_point_reachable(bottomLeft[0], bottomLeft[1], POINT_RCV_DOWN[1] + 20):
                    print("机械臂无法到达棋盒，请重新放置到靠近机械臂的位置！")
                    self.speak("机械臂无法到达棋盒，请重新放置到靠近机械臂的位置！")
                    raise ValueError("机械臂无法到达棋盒，请将棋盒放置到靠近机械臂的位置")

                # 计算4x4网格的世界坐标位置
                collection_positions = calculate_4x4_collection_positions(chess_box_points)
                print('棋盒坐标：', topLeft, topRight, bottomRight, bottomLeft)

                world_width = np.linalg.norm(np.array(topRight) - np.array(topLeft))
                world_height = np.linalg.norm(np.array(topLeft) - np.array(bottomLeft))

                # # 检查每个格子是否大于PIECE_SIZE
                # min_size = PIECE_SIZE * 3 * 0
                #
                # if min_size > world_width or min_size > world_height:
                #     print('棋盒格子尺寸不足')
                #     self.speak(
                #         f"❌ 棋盒格子尺寸不足，需要大于{min_size}mm，当前尺寸: {world_width:.2f}mm x {world_height:.2f}mm")
                #     raise ValueError("棋盒格子尺寸不足")
                print(f"✅ 棋盒尺寸检查通过，格子尺寸: {world_width:.2f}mm x {world_height:.2f}mm")

                # 3. 识别红方棋子并移动到棋盒下层
                print("🔴 开始收集红方棋子...")
                self.speak("开始收集红方棋子")
                self.collect_half_board_pieces("red", collection_positions)

                # 4. 识别黑方棋子并移动到棋盒上层
                print("⚫ 开始收集黑方棋子...")
                self.speak("开始收集黑方棋子")
                self.collect_half_board_pieces("black", collection_positions)

                print("✅ 收局完成")
                self.speak("收局完成")
                time.sleep(5)
                return
        except Exception as e:
            print(e)
            self.speak("收局失败")
            time.sleep(5)
    def collect_half_board_pieces(self, side, collection_positions):
        """
        收集指定颜色的棋子到棋盒

        Args:
            side: 收集棋子颜色("red"或"black")
            collection_positions: 收集位置列表
        """
        pick_height = POINT_DOWN[0]
        place_height = POINT_RCV_DOWN[0] if side == "red" else POINT_RCV_DOWN[1]  # red放底层，black放上层

        # 根据side决定要收集的棋子类型（大写为红方，小写为黑方）
        if side == "red":
            # 收集所有红方棋子（大写字母）
            target_class_names = ['R', 'N', 'B', 'A', 'K', 'C', 'P']
        else:
            # 收集所有黑方棋子（小写字母）
            target_class_names = ['r', 'n', 'b', 'a', 'k', 'c', 'p']

        # 1. 处理红方半区
        print(f"🔍 在红方半区寻找{side}方棋子...")
        red_piece_positions = self._collect_pieces_from_half_board(
            RED_CAMERA, "RED_CAMERA", target_class_names)

        black_piece_positions = self._collect_pieces_from_half_board(
            BLACK_CAMERA, "BLACK_CAMERA", target_class_names)

        if len(red_piece_positions) + len(black_piece_positions) != 16:
            print(f"⚠️ 棋子数量不足,只有{len(red_piece_positions) + len(black_piece_positions)}")
            self.speak("棋子数量不足16个,无法步棋")
            raise ValueError("棋子数量不足,无法步棋")

        # 按从左到右、从上到下的顺序排序
        red_piece_positions.sort(key=lambda p: (p[1], p[0]))  # 按y坐标升序，x坐标升序

        # 立即移动红方半区识别到的棋子到棋盒
        position_index = 16 - len(red_piece_positions) - len(black_piece_positions)
        print(f"🚚 开始移动红方半区识别到的{side}方棋子...")
        for x_world, y_world in red_piece_positions:
            if position_index >= len(collection_positions):
                print("⚠️ 棋盒位置不足")
                raise ValueError("棋盒位置不足")

            # 目标位置
            target_x, target_y = collection_positions[position_index]

            self.point_move(
                [x_world, y_world, pick_height],
                [target_x, target_y, place_height],  # 根据side决定放置高度
                [0, 0]  # home_row 参数，控制 move_home 行为
            )

            position_index += 1
            print(f"✅ 将{side}方棋子从({x_world:.1f},{y_world:.1f})放置到棋盒位置({position_index}/{len(red_piece_positions)})")

        print(f"✅ 完成移动红方半区{side}方棋子，共移动{position_index}个")

        # 2. 处理黑方半区
        print(f"🔍 在黑方半区寻找{side}方棋子...")

        # 按从左到右、从上到下的顺序排序
        black_piece_positions.sort(key=lambda p: (p[1], p[0]))  # 按y坐标升序，x坐标升序

        # 移动黑方半区识别到的棋子到棋盒
        print(f"🚚 开始移动黑方半区识别到的{side}方棋子...")
        for x_world, y_world in black_piece_positions:
            if position_index >= len(collection_positions):
                print("⚠️ 棋盒位置不足")
                break

            # 目标位置
            target_x, target_y = collection_positions[position_index]

            self.point_move(
                [x_world, y_world, pick_height],
                [target_x, target_y, place_height],  # 根据side决定放置高度
                [9, 9]  # home_row 参数，控制 move_home 行为
            )

            position_index += 1
            print(f"✅ 将{side}方棋子从({x_world:.1f},{y_world:.1f})放置到棋盒位置({position_index}/{len(black_piece_positions)})")

        print(f"✅ 完成收集{side}方棋子，共收集{position_index}个")

    def _collect_pieces_from_half_board(self, camera_position, camera_type, target_class_names):
        """
        从指定半区收集目标棋子

        Args:
            camera_position: 相机位置
            camera_type: 相机类型 ("RED_CAMERA" 或 "BLACK_CAMERA")
            target_class_names: 目标棋子类型

        Returns:
            list: 棋子位置列表 [(x_world, y_world, row), ...]
        """
        piece_positions = []
        if camera_type == "RED_CAMERA":
            inverse_matrix = self.inverse_matrix_r
        else:
            inverse_matrix = self.inverse_matrix_b

        # 移动到拍照点
        self.urController.run_point_j(camera_position)
        # time.sleep(3)

        # 捕获图像
        image, depth = self.capture_stable_image(is_chessboard=False)
        if image is None:
            print(f"⚠️ 无法捕获{camera_type}图像")
            return piece_positions

        # 使用YOLO检测器识别棋子
        objects_info = self.detector.detect_objects_with_height(
            image, depth,
            conf_threshold=self.args.conf,
            iou_threshold=self.args.iou
        )

        # 筛选出目标颜色的棋子
        for object_info in objects_info:
            if object_info['class_name'] in target_class_names:
                pixel_x, pixel_y = object_info['center']
                x_world, y_world = multi_camera_pixel_to_world(pixel_x, pixel_y,inverse_matrix, camera_type)
                piece_positions.append((x_world, y_world))

        return piece_positions

    # 布局
    def setup_initial_board(self):
        """
        布局函数：从收子区取出棋子并按初始布局放到棋盘上
        先处理上层的黑方棋子，再处理下层的红方棋子
        """
        try:
            print("🎯 开始初始布局...")
            self.speak("开始初始布局")

            # 定义中国象棋初始布局 (从上到下，从左到右)
            # 黑方在上半区(0-4行)，红方在下半区(5-9行)
            initial_layout = [
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

            # 1. 处理上层黑方棋子
            print("⚫ 处理上层黑方棋子...")
            self.speak("正在布置黑方棋子")
            for i in range(20):
                if self.setup_half_board_pieces("black", initial_layout):
                    break
                time.sleep(10)

            # 2. 处理下层红方棋子
            print("🔴 处理下层红方棋子...")
            self.speak("正在布置红方棋子")
            for i in range(20):
                if self.setup_half_board_pieces("red", initial_layout):
                    break
                time.sleep(10)

            print("✅ 初始布局完成")
        except Exception as e:
            print(f"❌ 初始布局异常: {str(e)}")
            self.speak("初始布局异常")
            raise e
    def setup_half_board_pieces(self, side, target_layout):
        """
        布置半区棋子，确保棋子类型与目标位置匹配

        Args:
            side: 棋子方("red"或"black")
            target_layout: 目标布局
        """
        # 移动到收子区拍照点
#         self.urController.set_speed(0.8)
        self.urController.run_point_j(RCV_CAMERA)
        # time.sleep(3)
        # 捕获图像和深度信息
        rcv_image, rcv_depth = self.capture_stable_image(is_chessboard=False)
        if rcv_image is None:
            print("⚠️ 无法捕获收子区图像")
            return

        inverse_matrix = self.inverse_matrix_rcv_h if side == "black" else self.inverse_matrix_rcv_l

        # 使用YOLO检测器识别收子区的棋子（包含高度信息）
        objects_info = self.detector.detect_objects_with_height(
            rcv_image, rcv_depth,
            conf_threshold=self.args.conf,
            iou_threshold=self.args.iou,
            mat=self.m_rcv
        )

        # 确定要处理的行范围和层
        if side == "black":
            rows = range(5, 10)
            # 上层棋子高度小于RCV_H_LAY
            is_target_layer = lambda h: h and h < RCV_H_LAY
            layer_name = "上层"
            target_class_names = ['r', 'n', 'b', 'a', 'k', 'c', 'p']  # 黑方棋子类型
        else:
            rows = range(0, 5)
            # 下层棋子高度大于等于RCV_H_LAY
            is_target_layer = lambda h: h and h >= RCV_H_LAY
            layer_name = "下层"
            target_class_names = ['R', 'N', 'B', 'A', 'K', 'C', 'P']  # 红方棋子类型

        pick_height = POINT_RCV_DOWN[1] if side == "black" else POINT_RCV_DOWN[0]
        print(f"📦 从收子区{layer_name}取{side}方棋子")

        # 创建棋子列表，按目标布局顺序排列
        target_pieces = []
        for row in rows:
            for col in range(9):
                piece = target_layout[9-row][col]
                if piece != '.' and piece in target_class_names:
                    target_pieces.append((row, col, piece))
        # 棋子棋盒位置列表
        available_pieces = {}

        # 从objects_info中提取棋子位置信息并按类型分类
        if objects_info:
            for i, obj_info in enumerate(objects_info):
                class_name = obj_info['class_name']
                # 检查是否为目标颜色的棋子
                if class_name not in target_class_names:
                    continue

                # 获取边界框中心点
                center_x, center_y = obj_info['center']

                # 根据高度判断是否为目标层
                height = obj_info.get('height', None)

                # 直接将像素坐标转换为世界坐标
                x_world, y_world = multi_camera_pixel_to_world(
                    center_x, center_y, inverse_matrix)

                # 按棋子类型分类存储
                if class_name not in available_pieces:
                    available_pieces[class_name] = []
                available_pieces[class_name].append((x_world, y_world, height))

        # 对每种类型的棋子按位置排序
        for piece_type in available_pieces:
            available_pieces[piece_type].sort(key=lambda p: (p[1], p[0]))  # 按y坐标升序，x坐标升序

        # 检查是否所有棋子都齐全
        required_pieces_count = {}
        for _, _, piece in target_pieces:
            required_pieces_count[piece] = required_pieces_count.get(piece, 0) + 1

        available_pieces_count = {}
        for piece_type, pieces in available_pieces.items():
            available_pieces_count[piece_type] = len(pieces)

        # 检查棋子是否完整
        is_complete = True
        missing_pieces = []

        for piece_type, required_count in required_pieces_count.items():
            available_count = available_pieces_count.get(piece_type, 0)
            if available_count < required_count:
                is_complete = False
                missing_pieces.append(f"{self.piece_map[piece_type]}缺少{required_count - available_count}个")

        if not is_complete:
            print(f"⚠️ {side}方棋子不完整: {', '.join(missing_pieces)}")
            self.speak(f"{side}方棋子{', '.join(missing_pieces)}")
            return  # 如果棋子不完整，直接返回不执行布置
        else:
            total_count = sum(available_pieces_count.values())
            print(f"✅ {side}方{total_count}个棋子齐全，开始布置")

        # 移动棋子到棋盘正确位置
        piece_counters = {piece: 0 for piece in target_class_names}  # 为每种棋子类型维护计数器

        for i, (target_row, target_col, target_piece) in enumerate(target_pieces):
            # 获取对应类型的下一个可用棋子
            if target_piece not in available_pieces or piece_counters[target_piece] >= len(available_pieces[target_piece]):
                print(f"⚠️ {layer_name}{side}方缺少棋子{target_piece}")
                continue

            # 获取该类型棋子的下一个可用实例
            piece_index = piece_counters[target_piece]
            x_world, y_world, piece_height = available_pieces[target_piece][piece_index]
            piece_counters[target_piece] += 1  # 增加该类型棋子的计数器

            # 计算目标位置世界坐标
            x_world_target, y_world_target = chess_to_world_position(target_col, target_row, side)
            place_height = POINT_DOWN[0]  # 放置高度

            rcv_center_x, rcv_center_y = get_area_center(CHESS_POINTS_RCV_H)
            rcv_world_x, rcv_world_y = multi_camera_pixel_to_world(
                    rcv_center_x, rcv_center_y, inverse_matrix)
            print(f"📥 将{side}方棋子{target_piece}从收子区放置到位置({target_row},{target_col})")

            # 移动到收子区拍照点
#             self.urController.set_speed(0.8)
            # self.urController.run_point_j(RCV_CAMERA)

            # 移动到中心点
            self.urController.move_to(rcv_world_x, rcv_world_y, pick_height + 50)

            # 移动到棋子上方
            self.urController.move_to(x_world, y_world, pick_height+20)
#             time.sleep(1)

            # 降低到吸取高度
#             self.urController.set_speed(0.5)
            self.urController.move_to(x_world, y_world, pick_height)
#             time.sleep(1)

            # 吸取棋子
            self.urController.set_do(IO_QI, 1)  # 吸合
#             time.sleep(1)

            # 抬起棋子到安全高度
#             self.urController.set_speed(0.8)
            self.urController.move_to(x_world, y_world, pick_height+20)
#             time.sleep(1)

            # 移动到中心点
            self.urController.move_to(rcv_world_x, rcv_world_y, pick_height+50)
#             time.sleep(2)

            # 移动到棋盘上方
            col = 9 if side == "black" else 0
            self.move_home(col)

            # 移动到目标位置上方
            self.urController.move_to(x_world_target, y_world_target, place_height+20)
#             time.sleep(1)

            # 降低到放置高度
#             self.urController.set_speed(0.5)
            self.urController.move_to(x_world_target, y_world_target, place_height+5)
#             time.sleep(1)

            # 放置棋子
            self.urController.set_do(IO_QI, 0)
#             time.sleep(1)
            self.urController.move_to(x_world_target, y_world_target, place_height+20)

            # 抬起机械臂到安全高度
#             self.urController.set_speed(0.8)
            self.move_home(col)
#             time.sleep(1)

            print(f"✅ {side}方棋子{target_piece}已放置到位置({target_row},{target_col})")
        return True

    def undo_move(self, steps=2):
        """
        悔棋函数，将棋盘状态还原到前n步

        Args:
            steps: 要悔棋的步数，默认为1步
        """
        try:
            if self.side == self.args.robot_side:
                print(f"⚠️ 当前棋子方为 {self.side}，无法悔棋")
                self.speak(f"机器人正在落子，无法悔棋")
                raise Exception("机器人正在落子，无法悔棋")
            print(f"↩️ 执行悔棋，回到 {steps} 步前的状态")
            self.speak(f"正在执行悔棋")
            self.urController.hll(f_5=1)  # 红灯
            # 检查是否有足够的历史记录
            if len(self.move_history) < steps:
                print(f"⚠️ 没有足够的移动历史，当前只有 {len(self.move_history)} 步")
                self.speak("没有足够的移动历史")
                return False

            # 从移动历史中获取要撤销的移动
            moves_to_undo = self.move_history[-steps:]
            print(f".undo_move 将撤销的移动: {moves_to_undo}")

            # 逐步撤销移动
            for i in range(steps):
                move_uci = moves_to_undo[-(i+1)]  # 从最后一步开始撤销
                print(f"撤销移动: {move_uci}")

                # 解析UCI格式移动
                from_col = ord(move_uci[0]) - ord('a')  # 0-8 (a-i)
                from_row = int(move_uci[1])             # 0-9 (0-9 从下到上)
                to_col = ord(move_uci[2]) - ord('a')    # 0-8 (a-i)
                to_row = int(move_uci[3])               # 0-9 (0-9 从下到上)

                # 转换为数组索引
                from_row_idx = 9 - from_row
                to_row_idx = 9 - to_row

                # 检查目标位置是否有被吃的棋子需要恢复
                target_piece_key = f"{to_row_idx}{to_col}"
                if target_piece_key in self.captured_pieces_history:
                    # 恢复被吃的棋子
                    captured_info = self.captured_pieces_history[target_piece_key]
                    print(f"发现被吃的棋子需要恢复: {captured_info}")
                    self.speak(f"请将被吃的{self.piece_map[captured_info['piece']]}放回棋盘")

                    # 等待用户放回棋子
                    self.wait_for_player_adjustment()


                # 物理上将棋子移回原位
                self._move_piece_back(from_row, from_col, to_row, to_col)

            # 更新移动历史
            self.move_history = self.move_history[:-steps]

            # 更新全局变量 move_count 和 side
            self.move_count = len(self.move_history)
            self._update_side_after_undo()

            # 更新棋盘状态
            self._revert_board_state(steps)

            # 更新MainGame棋盘状态
            self._revert_maingame_state(steps)

            # 7. 显示更新后的棋盘
            if self.args.show_board:
                self.game.graphic(self.board)

            print(f"✅ 悔棋完成，已回到 {steps} 步前的状态")
            self.speak("悔棋完成")
            self.is_undo = True
            return True
        except Exception as e:
            print(f"❌ 悔棋异常: {str(e)}")
            raise e

    def _revert_maingame_state(self, steps):
        """
        还原MainGame的棋盘状态

        Args:
            steps: 要还原的步数
        """
        print(f"🔄 还原MainGame棋盘状态，撤销 {steps} 步")

        # 重新初始化MainGame状态
        self.maingame.restart_game()

        # 重新应用未被撤销的移动到MainGame
        moves_to_keep = self.move_history
        for move_uci in moves_to_keep:
            try:
                # 将UCI移动转换为MainGame坐标
                from_col = ord(move_uci[0]) - ord('a')
                from_row = int(move_uci[1])
                to_col = ord(move_uci[2]) - ord('a')
                to_row = int(move_uci[3])

                # 转换为MainGame坐标系 (镜像处理)
                mg_from_x = 8 - from_col
                mg_to_x = 8 - to_col
                mg_from_y = 9 - from_row
                mg_to_y = 9 - to_row

                # 创建移动步骤
                from src.cchessAG import my_chess
                s = my_chess.step(mg_from_x, mg_from_y, mg_to_x, mg_to_y)
                print(f"已创建移动步骤: {s}")

                # 执行移动到MainGame并保存历史信息
                self.maingame.mgInit.move_to(s)
                print(f"MainGame重新应用移动: {move_uci} -> ({mg_from_x},{mg_from_y}) to ({mg_to_x},{mg_to_y})")

            except Exception as e:
                print(f"MainGame应用移动 {move_uci} 时出错: {e}")

    def _update_side_after_undo(self):
        """
        悔棋后更新当前回合方
        """
        # 根据已走步数和机器人执子方来确定当前回合方
        is_robot_turn = (self.move_count + (0 if self.args.robot_side == 'red' else 1)) % 2 == 1
        if not is_robot_turn:
            self.side = self.args.robot_side
        else:
            self.side = 'black' if self.args.robot_side == 'red' else 'red'
        print(f"🔄 悔棋后更新当前回合方为: {self.side}")

    def _move_piece_back(self, from_row, from_col, to_row, to_col):
        """
        物理上将棋子从目标位置移回起始位置

        Args:
            from_row, from_col: 起始位置
            to_row, to_col: 目标位置
        """
        print(f"🔄 物理移动棋子从 ({to_row},{to_col}) 回到 ({from_row},{from_col})")

        pick_height = POINT_DOWN[0]

        # 计算世界坐标
        # 起始位置（现在是目标位置）
        if to_row <= 4:
            half_board = 'red'
            from_x_world, from_y_world = chess_to_world_position(to_col, to_row, half_board)
        else:
            half_board = 'black'
            from_x_world, from_y_world = chess_to_world_position(to_col, to_row, half_board)

        # 目标位置（现在是起始位置）
        if from_row <= 4:
            half_board = 'red'
            to_x_world, to_y_world = chess_to_world_position(from_col, from_row, half_board)
        else:
            half_board = 'black'
            to_x_world, to_y_world = chess_to_world_position(from_col, from_row, half_board)

        print(f'世界坐标：{from_x_world}, {from_y_world} -> {to_x_world}, {to_y_world}')

        # 执行移动
        self.point_move(
            [from_x_world, from_y_world, pick_height],
            [to_x_world, to_y_world, pick_height],
            [to_row, from_row]
        )

        # 回到初始位置
        print("🏠 返回初始位置")
#         self.urController.set_speed(0.5)
        self.move_home()

    def _revert_board_state(self, steps):
        """
        还原棋盘逻辑状态

        Args:
            steps: 要还原的步数
        """
        print(f"🔄 还原棋盘逻辑状态，撤销 {steps} 步")

        # 重新初始化棋盘
        self.board = cchess.Board()

        # 重新应用未被撤销的移动
        moves_to_keep = self.move_history
        for move_uci in moves_to_keep:
            try:
                move = cchess.Move.from_uci(move_uci)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    print(f"重新应用移动: {move_uci}")
            except Exception as e:
                print(f"应用移动 {move_uci} 时出错: {e}")

        # 更新棋盘位置状态
        self.previous_positions = self.his_chessboard[self.move_count]

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
                from_x, from_y, to_x, to_y = algebraic_to_coordinates(self.move_uci)
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
                from_x, from_y, to_x, to_y = algebraic_to_coordinates(self.move_uci) if self.move_uci else (4, 0, 4, 1)
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


    def find_check_move(self):
        """
        优先寻找能吃掉对方将军的移动
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
        from_x, from_y, to_x, to_y = algebraic_to_coordinates(self.move_uci)
        move_uci = get_best_move_with_computer_play(self.maingame, self.board, from_x, from_y, to_x, to_y)

        print(f"✅ AI决定走: {move_uci}")
        return move_uci


    # 主函数
    def set_side(self):
        if self.side == 'red':
            self.side = 'black'
        else:
            self.side = 'red'
    def _init_play_game(self):
        self.his_chessboard = {} # 历史棋盘
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
        self.previous_positions = self.chess_positions
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
            self.speak("开始对弈，请等待指示灯为绿色再落子")
            self.voice_engine_type = "edge"

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
                # if self.move_count == 1:
                #     self.recognize_chessboard()

                    # # 检查初始棋子位置
                    # while not self.check_all_pieces_initial_position():
                    #     if self.surrendered:
                    #         return
                    #
                    #     # 如果棋子位置不正确，等待玩家调整
                    #     self.wait_for_player_adjustment()

                if is_robot_turn:
                    self.urController.hll(f_5=1)  # 红灯
                    print(f"🤖 机器人回合")
                    self.speak("轮到机器人回合，请稍等")

                    # 3. 显示当前棋盘
                    if self.args.show_board:
                        self.game.graphic(self.board)

                    # 4. 计算下一步
                    move_uci = self.calculate_next_move()

                    # 6. 执行移动到棋盘对象
                    move = cchess.Move.from_uci(move_uci)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                    else:
                        self.speak("机器人无法执行该移动")
                        self.gama_over()
                        return

                    # 5. 执行移动
                    self.execute_move(move_uci)
                    self.move_history.append(move_uci)

                    print(f"当前{self.side}方")
                    self.set_side()
                    print(f"当前{self.side}方")


                    # 检查是否将军
                    if self.is_in_check(self.board,self.side):
                        self.speak("请注意，您已被将军！")

                    self.update_chess_positions_after_move(move_uci)
                    chinese_notation = self.uci_to_chinese_notation(move_uci, self.previous_positions)
                    self.speak(f"机器人已走子，{chinese_notation}")

                    # 7. 显示更新后的棋盘
                    if self.args.show_board:
                        self.game.graphic(self.board)

                    print(chinese_notation)

                else:
                    print("👤 人类回合")
                    self.urController.hll(f_4=1)  # 绿灯
                    self.speak("轮到您的回合，请落子")
                    print("⏳ 等待人类落子完成信号...")

                    # 修改等待逻辑，添加投降检查
                    while not self.urController.get_di(IO_SIDE, is_log=False) and not self.surrendered:
                        time.sleep(0.5)
                        if self.surrendered:
                            return
                        if self.is_undo:
                            break
                    if self.is_undo:
                        self.is_undo = False
                        continue
                        # 检查是否投降
                    if self.surrendered:
                        self.gama_over('surrender')
                        return

                    # 复位信号
                    self.urController.hll(f_5=1)  # 红灯
                    self.io_side = self.urController.get_di(IO_SIDE)
                    print("✅ 检测到人类落子完成信号")
                    self.speak("您已落子，请稍等")

                    # 识别当前棋盘状态以更新棋盘
                    print("🔍 识别棋盘以更新状态...")
                    self.his_chessboard[self.move_count-1] = copy.deepcopy(self.previous_positions)
                    # old_positions = self.previous_positions
                    # if self.move_count == 1:
                    #     old_positions = self.chess_positions
                    for i in range(10):
                        if i > 0:
                            positions = self.recognize_chessboard(True)
                        else:
                            positions = self.recognize_chessboard(True)
                        # 推断人类的移动
                        self.move_uci = self.infer_human_move(self.his_chessboard[self.move_count-1], positions)
                        if self.move_uci:
                            break
                    if self.move_uci:
                        print(f"✅ 人类推测走子: {self.move_uci}")
                        move = cchess.Move.from_uci(self.move_uci)
                        if move in self.board.legal_moves:
                            # 检查是否吃掉了机器人的将军
                            is_captured, king_side = self.is_king_captured_by_move(self.move_uci, self.previous_positions)
                            # 如果吃掉的是机器人的将/帅
                            if is_captured and king_side == self.args.robot_side:
                                self.gama_over('player')  # 人类玩家获胜
                                self.speak('吃掉了机器人的将军！')
                                return  # 结束游戏

                            self.board.push(move)

                        else:
                            # 检查是否被将军且无法解除将军状态
                            if self.is_in_check(self.board,self.args.robot_side):
                                # 移动无效，执行空移动
                                self.board.push(cchess.Move.null())

                                # 检查是否存在能吃掉将军的移动
                                move_uci = self.find_check_move()

                                # 检查这个移动是否真的是吃掉将军的移动
                                move = cchess.Move.from_uci(move_uci)
                                if move in self.board.legal_moves:
                                    # 检查目标位置是否是对方的将/帅
                                    target_piece = self.board.piece_at(move.to_square)
                                    if target_piece and target_piece.piece_type == cchess.KING:
                                        # 确实是吃掉将军的移动，执行它
                                        self.execute_move(move_uci)
                                        # self.speak("将军！吃掉你的将帅！")
                                        self.speak(f"很遗憾，您输了！")
                                        time.sleep(20)
                                        return  # 结束游戏

                            else:
                                self.speak("您违规了，请重新走子")
                                self.move_count = self.move_count - 1
                                self.urController.hll(f_4=1)  # 绿灯
                                continue
                    else:
                        print("错误！无法推断人类的移动")
                        self.speak("无法检测到走棋，请重新落子")
                        self.urController.hll(f_4=1)  # 绿灯
                        self.move_count = self.move_count - 1
                        continue

                    # 显示更新后的棋盘
                    if self.args.show_board:
                        self.game.graphic(self.board)

                    # 落子完成
                    self.update_chess_positions_after_move(self.move_uci)
                    print(f"✅ 人类走法已应用: {self.move_uci}")
                    chinese_notation = self.uci_to_chinese_notation(self.move_uci, self.previous_positions)
                    self.speak(f"您已走子，{chinese_notation}")
                    print(chinese_notation)

                    self.move_history.append(self.move_uci)
                    self.his_chessboard[self.move_count] = copy.deepcopy(self.previous_positions)

                    self.set_side()
                # 短暂等待以便观察
                #             time.sleep(1)
                # self.clear_cache()


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
                            self.speak("您已被将死！")
                            self.gama_over('dobot')
                        else:
                            self.gama_over()
                    else:
                        self.gama_over('平局')
        except Exception as e:
            self.report_error(str(e))
    def gama_over(self,winner='player'):
        self.urController.hll()
        if winner == 'player':
            print(f'恭喜您获得胜利！')
            self.speak(f"恭喜您获得胜利！")
        elif winner == 'dobot':
            print(f'很遗憾，您输了！')
            self.speak(f"很遗憾，您输了！")
        elif winner == 'surrender':
            print(f'您已投降！')
            self.speak(f"您已投降！")
        else:
            print("🤝 游戏结束，平局")
            self.speak(f"游戏结束，平局")
        time.sleep(3)

    # 保存

    async def save_recognition_result_with_detections(self, chess_result, red_image, red_detections, black_image, black_detections):
        """
        异步保存带检测框的识别结果图像

        Args:
            chess_result: 棋盘识别结果
            red_image: 红方半区原始图像
            red_detections: 红方半区检测结果 (Results对象)
            black_image: 黑方半区原始图像
            black_detections: 黑方半区检测结果 (Results对象)
        """
        import cv2
        from copy import deepcopy
        import asyncio

        # 创建结果目录
        result_dir = self.args.result_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

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
                red_detected_path = os.path.join(result_dir,f"red_side_detected{self.move_count}.jpg")
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
                black_detected_path = os.path.join(result_dir, f"black_side_detected{self.move_count}.jpg")
                cv2.imwrite(black_detected_path, black_image_with_detections)
                print(f"💾 黑方检测结果已保存至: {black_detected_path}")

        async def save_chessboard_layout():
            """异步保存棋盘布局图"""
            # 可视化完整的棋盘布局
            self.chessboard_image = self.visualize_chessboard(chess_result)
            chessboard_path = os.path.join(result_dir, f"chessboard_layout.jpg")
            cv2.imwrite(chessboard_path, self.chessboard_image)
            # 报告棋盘识别结果给web端
            if self.args.use_api:
                self.report_board_recognition_result(self.chessboard_image)

            print(f"💾 棋盘布局图已保存至: {chessboard_path}")

        # 并发执行保存操作
        await asyncio.gather(
            save_red_detections(),
            save_black_detections(),
            save_chessboard_layout()
        )

    # 语音
    def speak(self, text):
        """
        使用统一的TTS管理器进行异步语音播报

        Args:
            text: 要播报的文本
        """
        # 检查是否启用语音
        if not self.args.enable_voice:
            return

        try:
            print(f"📢 语音播报: {text}")
            # 使用异步方式调用TTS管理器播报文本
            if hasattr(self, 'tts_manager') and self.tts_manager:
                # 提交到线程池异步执行
                async def async_speak():
                    await self.tts_manager.speak_async(text)
                asyncio.run(async_speak())
                time.sleep(1)
            else:
                print("⚠️ TTS管理器未初始化")
        except Exception as e:
            print(f"⚠️ 语音播报失败: {e}")

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

    def set_surrendered(self):
        self.surrendered = True
        time.sleep(3)
        self.urController.hll()
    def cleanup(self):
        """
        清理资源
        """
        try:
            self.surrendered = True

            # 断开机械臂
            if self.urController:
                self.urController.hll()
                print("🔌 断开机械臂连接...")
                self.urController.disconnect()

            # 清理相机
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                print("📷 关闭相机...")
                self.pipeline.stop()
                self.pipeline = None


            # 关闭OpenCV窗口
            if self.args.show_camera:
                cv2.destroyAllWindows()


            print("✅ 清理完成")
            self.speak("结束运行")
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
    def report_board_recognition_result(self, chessboard_image):
        """
        报告棋盘识别结果图像信息

        Args:
            chessboard_image: 识别后的棋盘图像(numpy数组)
        """
        # 发送棋盘识别结果到游戏服务
        try:
            from api.services.chess_game_service import chess_game_service
            if hasattr(chess_game_service, 'game_events') and chess_game_service.game_events:
                # 将图像转换为base64编码以便通过JSON传输
                import base64
                import cv2
                import numpy as np

                # 将图像编码为JPEG格式
                if chessboard_image is not None:
                    _, buffer = cv2.imencode('.jpg', chessboard_image)
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

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

def create_parser():
    """创建参数解析器"""
    parser = argparse.ArgumentParser(description='象棋自动对弈系统')

    # 显示和保存参数
    parser.add_argument('--use_api', default=False, help='是否使用api')
    parser.add_argument('--show_camera', default=False, action='store_true', help='是否显示相机实时画面')
    parser.add_argument('--show_board',  default=False, action='store_true', help='是否在窗口中显示棋局')
    parser.add_argument('--save_recognition_results', default=False, action='store_true', help='是否保存识别结果')
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
                        default='../src/cchessYolo/runs/detect/chess_piece_detection_separate5/weights/best.pt',
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
        chess_flow.initialize()

        # 收局
        # chess_flow.collect_pieces_at_end()

        # 布局
        # chess_flow.setup_initial_board()

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
