# file: /media/jetson/KESU/V10.14/Embodied/runner/chessPlayFlow/chess_camera.py

import asyncio
import os
import threading
import time
from typing import Optional, Tuple, List

import cv2
import numpy as np

from src.cchessYolo.detect_chess_box import select_corner_circles, order_points, calculate_box_corners
from manager.manager import system_manager  # 导入系统管理器以访问camera_manager


class ChessPlayFlowCamera:
    """
    象棋对弈流程中的相机管理模块
    负责棋盘识别、图像捕获和视觉处理相关功能
    """

    def __init__(self, parent):
        """
        初始化相机管理器

        Args:
            parent: 父级对象（ChessPlayFlow）
        """
        self.parent = parent
        self._recognition_lock = threading.Lock()  # 识别任务锁，防止并发识别

    def capture_stable_image(self, num_frames: int = 5) -> Tuple[Optional[np.ndarray], Optional]:
        """
        通过camera_manager捕获稳定的图像

        Args:
            num_frames: 用于平均的帧数

        Returns:
            tuple: (图像, 深度帧) 或 (None, None) 如果失败
        """
        if not system_manager.camera_manager or not system_manager.camera_manager.running:
            print("⚠️ 相机未初始化")
            return None, None

        # 使用camera_manager.capture_stable_image获取稳定图像
        return system_manager.camera_manager.capture_stable_image(num_frames=num_frames)

    def update_camera_display(self, image: np.ndarray) -> None:
        """
        通过camera_manager更新相机显示

        Args:
            image: 要显示的图像
        """
        if system_manager.camera_manager and image is not None:
            system_manager.camera_manager.update_camera_display(image)

    def detect_chess_box(self, max_attempts: int = 10) -> Optional[List]:
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
            # 检查是否需要停止
            if self.parent.surrendered or self.parent._stop_event.is_set():
                return None

            # 捕获图像
            rcv_image, rcv_depth = self.capture_stable_image()
            if rcv_image is None:
                print("⚠️ 无法捕获收子区图像")
                time.sleep(0.5)  # 等待一段时间再重试
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

                time.sleep(0.5)  # 等待一段时间再重试

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

    def cleanup_camera_windows(self) -> None:
        """
        清理相机窗口资源
        """
        try:
            if system_manager.camera_manager:
                system_manager.camera_manager.cleanup_camera_windows()
        except Exception as e:
            print(f"⚠️ 清理相机窗口时出错: {e}")
