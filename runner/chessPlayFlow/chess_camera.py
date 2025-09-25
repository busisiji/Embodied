import asyncio
import os
import threading
import time

import cv2
import numpy as np

from parameters import WORLD_POINTS_R, WORLD_POINTS_RCV, WORLD_POINTS_B, CHESS_POINTS_R, CHESS_POINTS_RCV_H, \
    CHESS_POINTS_B, CHESS_POINTS_RCV_L
from src.cchessYolo.detect_chess_box import select_corner_circles, order_points, calculate_box_corners
from utils.calibrationManager import calculate_perspective_transform_matrices
from utils.corrected import get_corrected_chessboard_points, correct_chessboard_to_square


class ChessPlayFlowCamera():
    def __init__(self, parent):
        self.parent = parent
    # 相机
    def setup_camera_windows(self):
        """
        初始化相机显示窗口
        """
        if self.parent.args.show_camera:
            try:
                # 先清理可能存在的窗口
                cv2.destroyAllWindows()
                # 创建新窗口
                cv2.namedWindow("camera", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)
            except cv2.error as e:
                print(f"⚠️ 创建窗口时出错: {e}")
                self.parent.args.show_camera = False

    def update_camera_display(self, image, window_name="camera"):
        """
        更新相机显示
        """
        if self.parent.args.show_camera and image is not None:
            try:
                # 检查窗口是否存在
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    # 如果窗口不存在，重新创建
                    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)

                # 显示图像
                cv2.startWindowThread()
                cv2.imshow(window_name, image)

                # 使用1ms等待，检查按键事件
                key = cv2.waitKey(1) & 0xFF

                # 检查是否按下ESC键(27)或窗口被关闭
                if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:  # ESC键
                    print("ESC键被按下，关闭显示窗口")
                    cv2.destroyAllWindows()
                    self.parent.args.show_camera = False  # 禁用后续显示

            except cv2.error as e:
                print(f"⚠️ 更新显示时出错: {e}")

    def cleanup_camera_windows(self):
        """
        清理相机窗口
        """
        try:
            if self.parent.args.show_camera:
                cv2.destroyAllWindows()
        except:
            pass

    def capture_stable_image(self, num_frames=5, is_chessboard=True):
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
            if self.parent.pipeline is None:
                # 尝试重新初始化相机
                asyncio.run(self.parent.speak_cchess("相机未连接，正在重新连接相机"))
                self.parent.init_camera()

                if self.parent.pipeline is None:
                    retry_count += 1
                    asyncio.run(self.parent.speak_cchess(f"相机连接失败"))
                    # 使用更短的等待时间，并定期检查游戏状态
                    for _ in range(50):  # 5秒分成50个0.1秒
                        surrendered, paused = self.parent.check_game_state()
                        if surrendered:
                            return None, None
                        time.sleep(0.1)
                    continue

            try:
                frames_list = []
                depth_frames_list = []

                # 捕获多帧图像
                for i in range(num_frames):
                    # 定期检查游戏状态
                    if self.parent.surrendered:
                        return None, None

                    frames = self.parent.pipeline.wait_for_frames(timeout_ms=5000)  # 设置超时时间
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()

                    if color_frame and depth_frame:
                        frame = np.asanyarray(color_frame.get_data())
                        frames_list.append(frame)
                        depth_frames_list.append(depth_frame)
                    else:
                        continue

                    # 短暂等待，也定期检查游戏状态
                    for _ in range(10):  # 0.1秒分成10个0.01秒
                        if self.parent.surrendered:
                            return None, None
                        time.sleep(0.01)

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
                self.parent.chess_r = CHESS_POINTS_R
                self.parent.chess_b = CHESS_POINTS_B
                self.parent.chess_rcv_h = CHESS_POINTS_RCV_H
                self.parent.chess_rcv_l = CHESS_POINTS_RCV_L

                # 畸变矫正
                if is_chessboard:
                    self.parent.chess_r, self.parent.m_R = get_corrected_chessboard_points(CHESS_POINTS_R)
                    self.parent.chess_b, self.parent.m_B = get_corrected_chessboard_points(CHESS_POINTS_B)
                    self.parent.chess_rcv_h, self.parent.m_RCV_h = get_corrected_chessboard_points(CHESS_POINTS_RCV_H)
                    self.parent.chess_rcv_l, self.parent.m_RCV_l = get_corrected_chessboard_points(CHESS_POINTS_RCV_L)

                    if self.parent.side == 'red':
                        result_frame, _ = correct_chessboard_to_square(result_frame, CHESS_POINTS_R, self.parent.m_R)
                    else:
                        result_frame, _ = correct_chessboard_to_square(result_frame, CHESS_POINTS_B, self.parent.m_B)

                self.parent.forward_matrix_r, self.parent.inverse_matrix_r = calculate_perspective_transform_matrices(world_r, self.parent.chess_r)
                self.parent.forward_matrix_b, self.parent.inverse_matrix_b = calculate_perspective_transform_matrices(world_b, self.parent.chess_b)
                self.parent.forward_matrix_rcv_h, self.parent.inverse_matrix_rcv_h = calculate_perspective_transform_matrices(world_rcv, self.parent.chess_rcv_h)
                self.parent.forward_matrix_rcv_l, self.parent.inverse_matrix_rcv_l = calculate_perspective_transform_matrices(world_rcv, self.parent.chess_rcv_l)
                if retry_count > 0:
                    asyncio.run(self.parent.speak_cchess(f"相机图像获取成功"))

                return result_frame, latest_depth_frame

            except Exception as e:
                # 定期检查游戏状态
                if self.parent.surrendered:
                    return None, None

                retry_count += 1
                error_msg = f"捕获图像失败，第{retry_count}次重试"
                print(f"⚠️ {error_msg}: {e}")
                asyncio.run(self.parent.speak_cchess(error_msg))

                # 如果达到最大重试次数，停止重试
                if retry_count >= max_retry_attempts:
                    asyncio.run(self.parent.speak_cchess("已达最大重试次数，无法获取图像"))
                    break

                # 等待一段时间后重试，也定期检查游戏状态
                for _ in range(30):  # 3秒分成30个0.1秒
                    if self.parent.surrendered:
                        return None, None
                    time.sleep(0.1)

                # 尝试重新初始化相机
                asyncio.run(self.parent.speak_cchess("正在重新初始化相机"))
                self.parent.pipeline = None
                self.parent.init_camera()

        # 如果所有重试都失败，返回None
        asyncio.run(self.parent.speak_cchess("无法捕获稳定图像，请检查相机连接"))
        return None, None

    # 识别
    def detect_chess_box(self, max_attempts=10):
        """
        识别棋盒位置，只支持检测4个圆角标记

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

            # 创建用于显示的图像副本
            display_image = rcv_image.copy()

            # 使用霍夫圆检测来识别棋盒的圆形标记
            gray = cv2.cvtColor(rcv_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)

            # 使用霍夫圆检测查找圆形贴纸
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,  # 圆心之间的最小距离
                param1=50,  # Canny边缘检测的高阈值
                param2=50,  # 累积阈值，越小检测到的圆越多
                minRadius=20,  # 最小半径
                maxRadius=40  # 最大半径
            )

            # 在图像上绘制检测到的圆
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    # 绘制圆
                    cv2.circle(display_image, (x, y), r, (0, 255, 0), 2)
                    # 绘制圆心
                    cv2.circle(display_image, (x, y), 2, (0, 0, 255), 3)


            # 只有检测到恰好4个圆时才继续处理
            if circles is not None and len(circles) == 4:
                # 选择4个角点圆
                selected_circles = select_corner_circles(circles)

                # 按顺序排列圆心点
                centers = [(int(circle[0]), int(circle[1])) for circle in selected_circles[:4]]
                ordered_centers = order_points(np.array(centers))

                # 计算平均半径
                radii = [int(circle[2]) for circle in selected_circles[:4]]
                avg_radius = int(np.mean(radii))

                # 计算棋盒的实际角点
                chess_box_points = calculate_box_corners(ordered_centers, avg_radius)

                # 如果成功计算了棋盒角点，在图像上绘制角点
                if chess_box_points is not None and len(chess_box_points) >= 4:
                    # 绘制棋盒角点
                    for i, point in enumerate(chess_box_points):
                        x, y = int(point[0]), int(point[1])
                        # 绘制角点
                        cv2.circle(display_image, (x, y), 5, (255, 0, 0), -1)
                        # 添加角点标签
                        cv2.putText(display_image, f"{i}", (x+10, y+10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    # 绘制棋盒边界框
                    for i in range(4):
                        pt1 = tuple(map(int, chess_box_points[i]))
                        pt2 = tuple(map(int, chess_box_points[(i+1)%4]))
                        cv2.line(display_image, pt1, pt2, (0, 255, 255), 2)

                    print("✅ 成功检测到4个圆角标记")
                    self.update_camera_display(display_image)
                    break
            else:
                if circles is not None:
                    print(f"🔍 检测到{len(circles)}个圆，需要恰好4个圆")
                else:
                    print("🔍 未检测到任何圆形标记")
        return chess_box_points

    def recognize_chessboard(self, is_run_red=False, half_board=None):
        """
        识别整个棋盘状态 (使用 YOLO 检测器，包含高度信息)
        """
        print("🔍 开始识别棋盘...")

        # 检查游戏状态
        surrendered, paused = self.parent.check_game_state()
        if surrendered:
            return

        # 创建结果目录
        if self.parent.args.save_recognition_results:
            result_dir = self.parent.args.result_dir
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

        # 识别红方半区
        if not half_board or half_board == "red":
            print("🔴 识别红方半区...")
            if is_run_red:
                self.parent.urController.run_point_j(self.parent.args.red_camera_position)

            # 多次捕获取最佳图像和深度信息
            time.sleep(0.5)
            red_image, red_depth = self.capture_stable_image()
            if red_image is None:
                print("⚠️ 无法捕获红方图像")
                return self.parent.chess_positions
            self.update_camera_display(red_image)

            # 识别红方半区棋子 (使用 YOLO，包含高度信息)
            # 将耗时的YOLO识别过程放到独立线程中执行
            def red_detection_task():
                return self.parent.detector.extract_chessboard_layout_with_height(
                    red_image, self.parent.chess_r, half_board="red",
                    conf_threshold=self.parent.args.conf,
                    iou_threshold=self.parent.args.iou
                )

            # 使用事件来同步等待识别结果
            import threading
            result_container = [None]  # 用于在线程间传递结果
            detection_event = threading.Event()

            def run_detection():
                result_container[0] = red_detection_task()
                detection_event.set()

            detection_thread = threading.Thread(target=run_detection, daemon=True)
            detection_thread.start()

            # 等待识别完成，同时定期检查游戏状态
            while not detection_event.is_set():
                if self.parent.surrendered:
                    return self.parent.chess_positions
                time.sleep(0.01)  # 短暂等待

            self.red_result, red_detections, points_center = result_container[0]

            if points_center:
                self.parent.piece_pixel_positions.update(points_center)
            else:
                asyncio.run(self.parent.speak_cchess("识别不到棋子"))

        if not half_board or half_board == "black":
            # 识别黑方半区
            print("⚫ 识别黑方半区...")
            self.parent.urController.run_point_j(self.parent.args.black_camera_position)

            # 多次捕获取最佳图像和深度信息
            time.sleep(0.5)
            black_image, black_depth = self.capture_stable_image()
            if black_image is None:
                print("⚠️ 无法捕获黑方图像")
                return self.parent.chess_positions

            self.update_camera_display(black_image)

            # 识别黑方半区棋子 (使用 YOLO，包含高度信息)
            # 将耗时的YOLO识别过程放到独立线程中执行
            def black_detection_task():
                return self.parent.detector.extract_chessboard_layout_with_height(
                    black_image, self.parent.chess_b, half_board="black",
                    conf_threshold=self.parent.args.conf,
                    iou_threshold=self.parent.args.iou
                )

            # 使用事件来同步等待识别结果
            import threading
            result_container = [None]  # 用于在线程间传递结果
            detection_event = threading.Event()

            def run_detection():
                result_container[0] = black_detection_task()
                detection_event.set()

            detection_thread = threading.Thread(target=run_detection, daemon=True)
            detection_thread.start()

            # 等待识别完成，同时定期检查游戏状态
            while not detection_event.is_set():
                if self.parent.surrendered:
                    return self.parent.chess_positions
                time.sleep(0.01)  # 短暂等待

            self.black_result, black_detections, points_center = result_container[0]

            if points_center:
                self.parent.piece_pixel_positions.update(points_center)
            else:
                asyncio.run(self.parent.speak_cchess("识别不到棋子"))

        # 合并结果 (黑方在0-4行，红方在5-9行，且红方需要倒置)
        chess_result = [['.' for _ in range(9)] for _ in range(10)]

        # 黑方半区放在棋盘的0-4行
        for row in range(5):  # 黑方半区 0-4行
            for col in range(9):
                chess_result[row][col] = self.black_result[row][col]

        # 红方半区放在棋盘的5-9行，并进行倒置处理
        for row in range(5):  # 红方半区原始为0-4行
            for col in range(9):
                # 红方需要倒置，所以(0,0)变成(9,8)
                chess_result[9-row][8-col] = self.red_result[row][col]
        self.parent.chess_positions = chess_result


        # 保存识别结果（包括可视化检测结果）
        if self.parent.args.save_recognition_results :
            if not half_board:
                asyncio.run(self.parent.save_recognition_result_with_detections(
                    red_image, red_detections, black_image, black_detections,chess_result
                ))
            elif half_board == "red":
                asyncio.run(self.parent.save_recognition_result_with_detections(
                    red_image=red_image, red_detections=red_detections,chess_result=chess_result
                ))
            elif half_board == "black":
                asyncio.run(self.parent.save_recognition_result_with_detections(
                    black_image=black_image, black_detections=black_detections,chess_result=chess_result
                ))

        print("✅ 棋盘识别完成")
        return chess_result
