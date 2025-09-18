# 初始化功能模块
import asyncio
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from dobot.dobot_control import connect_and_check_speed
from parameters import WORLD_POINTS_R, WORLD_POINTS_B, CHESS_POINTS_R, CHESS_POINTS_B, \
    CHESS_POINTS_RCV_H, CHESS_POINTS_RCV_L, WORLD_POINTS_RCV
from api.services.tts_service import TTSManager,tts_manager
from src.cchessAI.core.mcts import MCTS_AI
from src.cchessYolo.detect_chess_box import select_corner_circles, order_points, calculate_box_corners
from src.cchess_runner.chess_play_flow_base import ChessPlayFlowBase
from src.speech.speech_service import initialize_speech_recognizer, start_listening, get_speech_recognizer
from utils.calibrationManager import calculate_perspective_transform_matrices
from utils.corrected import get_corrected_chessboard_points, correct_chessboard_to_square



class ChessPlayFlowInit(ChessPlayFlowBase):
    def initialize(self):
        """
        初始化所有组件
        """
        print("🔧 开始初始化...")

        # 初始化语音引擎
        try:
            # 初始化统一的TTS管理器
            self.tts_manager = tts_manager
            self.speak("开始初始化系统")
        except Exception as e:
            print(f"⚠️ 语音引擎初始化失败: {e}")
            self.voice_engine = None

        # 初始化语音识别器
        try:
            if initialize_speech_recognizer(
            ):
                self.speech_recognizer = get_speech_recognizer()
                if self.speech_recognizer:
                    # 获取当前事件循环并创建任务
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        # 如果没有运行中的事件循环，创建一个新的
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    loop.create_task(self.speech_recognizer.start_listening())
                print("语音识别初始化并启动成功")
                self.speak("语音识别器初始化完成")
            else:
                print(f"⚠️ 语音识别器初始化异常: {e}")
                self.speech_recognizer = None
        except Exception as e:
            print(f"⚠️ 语音识别器初始化异常: {e}")
            self.speech_recognizer = None

        self.speak("系统初始化完成")

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
            from src.cchessYolo.chess_detection_trainer import ChessPieceDetectorSeparate
            self.detector = ChessPieceDetectorSeparate(
                model_path=self.args.yolo_model_path
            )
        except Exception as e:
            print(f"⚠️识别模型初始化失败: {e}")
            self.speak("识别模型初始化失败")
            raise Exception(f"识别模型初始化失败{self.args.yolo_model_path}")

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

                from src.cchessAI.core.net import PolicyValueNet
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
