# 修改后的 calibrateCamera_runner.py

import tkinter as tk
from tkinter import messagebox
import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime
from PIL import Image, ImageTk

from utils.calibrationManager import pixel_to_world, calculate_perspective_transform_matrices, multi_camera_pixel_to_world
from utils.corrected import correct_chessboard_to_square
from parameters import CHESS_POINTS_R, WORLD_POINTS_R
from src.cchessYolo.chess_detection_trainer import ChessPieceDetectorSeparate
from src.cchessYolo.detect_chess_box import select_corner_circles, order_points, calculate_box_corners
dir = os.path.dirname(os.path.abspath(__file__))
# ================== 配置参数 ==================
SQUARE_SIZE_MM = 12.5         # 棋盘格大小（单位：毫米）
CHESSBOARD_SHAPE = (7, 7)      # 内部角点数量（对应 4x4 棋盘格）
MAX_IMAGES = 100                # 最大采集图像数量
AUTO_CAPTURE_INTERVAL = 100   # 自动拍照间隔（毫秒）默认 10s
SAVE_DIR = os.path.join(dir, "calibration/images")
OUTPUT_DIR = os.path.join(dir, "calibration/output")
WIDTH = 1280
HEIGHT = 720
FPS = 6

class CalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RealSense 自动拍照 + 标定工具")

        self.result_dir = "validation_results"

        # 初始化相机管道
        self.pipeline = None
        self.config = None
        self.running = True

        # 存储数据
        self.captured = 0
        self.auto_capturing = False
        self.countdown = 0
        self.mouse_x = 0
        self.mouse_y = 0
        self.show_mouse_coords = False
        self.apply_correction = False  # 是否应用实时矫正
        self.mtx = None  # 相机矩阵
        self.dist = None  # 畸变系数
        self.M = None
        self.chess_box_points = None

        self.detector = ChessPieceDetectorSeparate(os.path.join(dir,'../src/cchessYolo/fruitYolo/runs/obb/fruit_obb_detection4/weights/best.pt')
        )

    def init(self):
        # 创建 UI
        self.create_ui()

        # 初始化摄像头
        self.init_camera()

        # 启动主循环
        self.root.after(10, self.update_frame)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close_window)

        _, self.inverse_matrix = calculate_perspective_transform_matrices(WORLD_POINTS_R, CHESS_POINTS_R)

    def create_ui(self):
        """创建 GUI 界面"""
        self.frame = tk.Frame(self.root)
        self.frame.pack(padx=10, pady=10)

        self.label = tk.Label(self.frame, text=" 实时预览")
        self.label.pack()

        self.canvas = tk.Canvas(self.frame, width=WIDTH if WIDTH <= 1280 else 1280, height=HEIGHT if HEIGHT <= 720 else 720)
        self.canvas.pack()

        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Enter>", lambda e: setattr(self, "show_mouse_coords", True))
        self.canvas.bind("<Leave>", lambda e: setattr(self, "show_mouse_coords", False))

        # 新增坐标显示标签
        self.coord_frame = tk.Frame(self.frame)
        self.coord_frame.pack(pady=5)

        self.coord_label = tk.Label(self.coord_frame, text="️ 鼠标坐标: (0, 0)", fg="black")
        self.coord_label.pack(side=tk.RIGHT, padx=5)

        self.world_coord_label = tk.Label(self.coord_frame, text=" 世界坐标: 未标定", fg="red")
        self.world_coord_label.pack(side=tk.RIGHT, padx=5)

        self.btn_frame = tk.Frame(self.frame)
        self.btn_frame.pack(pady=10)

        self.manual_button = tk.Button(self.btn_frame, text=" 拍照", command=self.toggle_manual_mode)
        self.manual_button.pack(side=tk.LEFT, padx=5)

        self.auto_button = tk.Button(self.btn_frame, text="⏱️ 自动拍照", command=self.toggle_auto_mode)
        self.auto_button.pack(side=tk.LEFT, padx=5)


        # self.calibrate_button = tk.Button(self.btn_frame, text=" 畸变矫正", command=self.apply_perspective_correction)
        # self.calibrate_button.pack(side=tk.LEFT, padx=5)

        # 新增按钮，控制是否实时应用矫正
        self.correct_toggle_button = tk.Button(self.btn_frame, text=" 实时矫正", command=self.toggle_correction)
        self.correct_toggle_button.pack(side=tk.LEFT, padx=5)

        self.detect_chess_box_button = tk.Button(self.btn_frame, text=" 识别收棋盒", command=self.detect_chess_box)
        self.detect_chess_box_button.pack(side=tk.LEFT, padx=5)

        # 新增识别棋子按钮
        self.detect_chess_button = tk.Button(self.btn_frame, text=" 识别棋子", command=self.detect_chess_pieces)
        self.detect_chess_button.pack(side=tk.LEFT, padx=5)

        self.squares_label = tk.Label(self.frame, text="未识别到棋格...", fg="red")
        self.squares_label.pack()

        self.hand_eye_calibration_button = tk.Button(self.btn_frame, text=" 手眼标定", command=self.hand_eye_calibration)
        self.hand_eye_calibration_button.pack(side=tk.LEFT, padx=5)

        self.toggle_label = tk.Label(self.frame, text="实时矫正 已禁用", fg="blue")
        self.toggle_label.pack()

        self.status_label = tk.Label(self.frame, text="状态：等待开始...", fg="blue")
        self.status_label.pack()

        self.quit_button = tk.Button(self.btn_frame, text=" 退出", command=self.stop_app)
        self.quit_button.pack(side=tk.LEFT, padx=5)

    def init_camera(self):
        """初始化深度相机"""
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # 启用彩色流
        self.config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        # 启用深度流
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, FPS)

        profile = self.pipeline.start(self.config)

    def save_image(self):
        """触发保存原始帧"""
        if not hasattr(self, 'original_frame'):
            return
        if self.captured >= MAX_IMAGES:
            messagebox.showinfo("提示", f"已达到最大拍摄数量 {MAX_IMAGES} 张。")
            self.auto_capturing = False
            self.manual_button.config(state=tk.NORMAL)
            return

        # 检查是否只保存有棋盘格的图像
        gray = cv2.cvtColor(self.original_frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SHAPE, None)

            # 创建目录
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

            # 生成文件名并保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, f"RS_{timestamp}.jpg")
        cv2.imwrite(filename, self.original_frame)
        self.captured += 1
        self.status_label.config(text=f"已拍摄：{self.captured} 张图像", fg="green")

    def toggle_manual_mode(self):
        """切换到手动模式"""
        if self.auto_capturing:
            self.auto_capturing = False
            self.manual_button.config(state=tk.NORMAL)
        self.save_image()

    def toggle_auto_mode(self):
        """切换到自动模式"""
        if not self.auto_capturing:
            self.auto_capturing = True
            self.manual_button.config(state=tk.DISABLED)
            self.countdown = AUTO_CAPTURE_INTERVAL
            self.status_label.config(text=f"状态：已切换到自动拍照模式（{AUTO_CAPTURE_INTERVAL // 1000}s/张）", fg="green")
        else:
            self.auto_capturing = False
            self.manual_button.config(state=tk.NORMAL)
            self.status_label.config(text="状态：已切换到手动拍照模式", fg="green")

    def hand_eye_calibration(self):
        """检测带十字的九点坐标进行手眼标定"""
        if not hasattr(self, 'current_frame'):
            self.status_label.config(text="⚠️ 未获取到图像数据", fg="red")
            return

        img = self.current_frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 检测九点坐标
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=9, qualityLevel=0.01, minDistance=50)

        if corners is None or len(corners) < 9:
            self.status_label.config(text="❌ 未检测到足够的九点坐标", fg="red")
            return

        # 对角点进行排序，从左到右，从上到下
        corners = corners.reshape(-1, 2)
        # 先按y坐标排序(从上到下)，再按x坐标排序(从左到右)
        sorted_indices = np.lexsort((corners[:, 0], corners[:, 1]))
        corners = corners[sorted_indices]

        # 重新排列为3x3网格格式
        # 将排序后的点重新排列成3行，每行按x坐标排序
        row1 = sorted(corners[0:3], key=lambda p: p[0])
        row2 = sorted(corners[3:6], key=lambda p: p[0])
        row3 = sorted(corners[6:9], key=lambda p: p[0])
        corners = np.array(row1 + row2 + row3)

        # 在图像上绘制检测到的角点并标注"第几点"
        for i, corner in enumerate(corners):
            x, y = corner.astype(int)
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img, f"{i + 1}", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示结果
        cv2.imshow("Hand-Eye Calibration", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 输出检测结果
        self.status_label.config(text="✅ 手眼标定完成", fg="green")
        print("检测到的九点坐标（从左到右，从上到下）：")
        for i, corner in enumerate(corners):
            x, y = corner.astype(int)
            print(f"点 {i+1}: ({x}, {y})")

    def detect_chess_pieces(self):
        """识别棋盘上的棋子位置和高度"""
        if not hasattr(self, 'original_frame') or not hasattr(self, 'depth_frame'):
            self.status_label.config(text="⚠️ 未获取到图像数据", fg="red")
            return

        # 使用新添加的函数检测物体和高度信息
        objects_info, _ = self.detector.detect_objects_with_height(
            self.original_frame,
            None,
            conf_threshold=0.5,
            iou_threshold=0.4
        )

        # 创建一个可视化图像用于显示结果
        result_image = self.original_frame.copy()

        # 绘制检测到的物体信息
        for obj in objects_info:
            # 获取边界框坐标
            x1, y1, x2, y2 = obj['bbox']
            class_name = obj['class_name']
            confidence = obj['confidence']
            height = obj['height']

            # 绘制边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 添加标签
            label = f'{class_name} {confidence:.2f}'
            cv2.putText(result_image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 显示世界坐标
            # wx,wy = pixel_to_world((x2+x1)/2,(y2+y1)/2)

            wx,wy = multi_camera_pixel_to_world((x2+x1)/2,(y2+y1)/2,self.inverse_matrix,"RED_CAMERA")
            xy_text = f'XY: {(x2+x1)/2:.0f} {(y2+y1)/2:.0f}'
            wxy_text = f'WXY: {wx:.0f} {wy:.0f}'
            cv2.putText(result_image, xy_text, (x1-20, y2 -40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.putText(result_image, wxy_text, (x1-40, y2 -20 ),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


            # 显示高度信息（如果可用）
            if height is not None:
                height_text = f'H: {height:.3f}m'
                cv2.putText(result_image, height_text, (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 显示结果（只显示一帧，2秒后自动关闭）
        cv2.imshow("Object Detection with Height", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 输出检测结果摘要
        detected_count = len(objects_info)
        self.status_label.config(text=f"✅ 物体检测完成 - 检测到 {detected_count} 个物体", fg="green")

        # 打印详细检测信息
        print(f"\n检测到 {detected_count} 个物体:")
        for i, obj in enumerate(objects_info):
            x, y = obj['center']
            height_info = f"{obj['height']:.3f}m" if obj['height'] is not None else "N/A"
            print(f"物体 {i+1}: {obj['class_name']} - 置信度: {obj['confidence']:.2f}, "
                  f"中心位置: ({x}, {y}), 高度: {height_info}")

    def apply_perspective_correction(self):
        """检测棋盘格并进行透视矫正，保存相机矩阵和畸变系数"""
        """
        从指定文件夹中的图像中计算相机矩阵和畸变系数
        :param image_folder: 图像文件夹路径
        """
        objp = np.zeros((CHESSBOARD_SHAPE[0] * CHESSBOARD_SHAPE[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHESSBOARD_SHAPE[0], 0:CHESSBOARD_SHAPE[1]].T.reshape(-1, 2)
        objp *= SQUARE_SIZE_MM

        objpoints = []  # 3D点
        imgpoints = []  # 2D图像点

        images = [os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR) if f.endswith(".jpg") or f.endswith(".png")]

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 寻找棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SHAPE, None)

            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        if len(objpoints) == 0:
            self.status_label.config(text="⚠️ 未找到有效棋盘格图像", fg="red")
            return

        # 计算相机矩阵和畸变系数
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        if ret:
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            np.savez(os.path.join(OUTPUT_DIR, "camera_params.npz"), mtx=mtx, dist=dist)
            self.status_label.config(text="✅ 从图像计算并保存标定矩阵成功", fg="green")
        else:
            self.status_label.config(text="⚠️ 相机标定失败", fg="red")


    def load_calibration_data(self):
        """
        从文件加载相机矩阵、畸变系数和透视变换矩阵
        """
        calibration_file = os.path.join(OUTPUT_DIR, "camera_params.npz")

        if not os.path.exists(calibration_file):
            self.status_label.config(text="⚠️ 未找到标定文件，请先进行标定", fg="red")
            return False

        try:
            data = np.load(calibration_file)
            self.mtx = data['mtx']
            self.dist = data['dist']
            if 'M' in data.files:
                self.M = data['M']
            self.status_label.config(text="✅ 成功加载相机标定数据", fg="green")
            return True
        except Exception as e:
            self.status_label.config(text=f"⚠️ 标定数据加载失败: {e}", fg="red")
            return False

    def toggle_correction(self):
        """切换是否实时应用矫正"""
        self.load_calibration_data()
        if self.mtx is None or self.dist is None:
            self.toggle_label.config(text="⚠️ 未加载相机矫正矩阵，请先进行标定", fg="red")
            self.apply_correction = False
            return

        self.apply_correction = not self.apply_correction
        status = "已启用" if self.apply_correction else "已禁用"
        self.toggle_label.config(text=f" 实时矫正 {status}", fg="green" if self.apply_correction else "red")


    def on_mouse_move(self, event):
        """鼠标移动事件，更新坐标显示"""
        self.mouse_x = event.x
        self.mouse_y = event.y

        if self.show_mouse_coords:
            # 更新鼠标相机坐标
            self.coord_label.config(text=f"️ 鼠标坐标: ({self.mouse_x}, {self.mouse_y})")
            wx,wy = pixel_to_world(self.mouse_x, self.mouse_y)
            self.world_coord_label.config(text=f" 世界坐标: ({wx}, {wy})")

    def stop_app(self):
        """停止程序"""
        self.running = False
        self.pipeline.stop()
        self.root.destroy()

    def on_close_window(self):
        """窗口关闭时执行清理"""
        if messagebox.askokcancel("退出", "是否要关闭程序并释放相机资源？"):
            self.running = False
            self.pipeline.stop()
            self.root.destroy()

    def camera_to_world_coordinates(self, x, y):
        """
        使用透视变换矩阵 M 将相机坐标转换为世界坐标
        :param x: 相机坐标 x
        :param y: 相机坐标 y
        :return: 转换后的世界坐标 (wx, wy)
        """
        if self.M is None:
            return None, None

        # 构造齐次坐标
        point_camera = np.array([[x, y]], dtype=np.float32)
        point_homogeneous = cv2.perspectiveTransform(point_camera.reshape(1, -1, 2), self.M)

        wx, wy = point_homogeneous[0][0]
        return int(wx), int(wy)
    def detect_chess_box(self):
        """使用圆形检测识别棋盒（基于四个圆形贴纸）"""
        if not hasattr(self, 'current_frame'):
            self.status_label.config(text="⚠️ 未获取到图像数据", fg="red")
            return

        # 复制原始图像用于处理
        img = self.current_frame.copy()

        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # 使用霍夫圆检测查找圆形贴纸
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,  # 圆心之间的最小距离
            param1=50,   # Canny边缘检测的高阈值
            param2=50,   # 累积阈值，越小检测到的圆越多
            minRadius=10,  # 最小半径
            maxRadius=40   # 最大半径
        )

        if circles is None:
            self.chess_box_points = []
            self.status_label.config(text="❌ 未检测到圆形标记", fg="red")
            return

        # 转换圆形坐标为整数
        circles = np.round(circles[0, :]).astype("int")

        # 如果检测到的圆形少于4个，无法构成四边形
        if len(circles) < 4:
            self.chess_box_points = []
            self.status_label.config(text=f"❌ 检测到的圆形标记不足4个 (检测到{len(circles)}个)", fg="red")
            return

        # 从检测到的圆形中选择最可能的4个角点圆形
        selected_circles = select_corner_circles(circles)

        if len(selected_circles) == 4:
            # 提取圆心和半径
            centers = [(int(circle[0]), int(circle[1])) for circle in selected_circles]
            radii = [int(circle[2]) for circle in selected_circles]

            # 计算平均半径
            avg_radius = int(np.mean(radii))

            # 按照顺序排列圆心点（左上，右上，右下，左下）
            ordered_centers = order_points(np.array(centers))

            # 根据圆形位置计算棋盒的实际角点（需要向外偏移一个半径）
            actual_corners = calculate_box_corners(ordered_centers, avg_radius)

            # 保存角点
            self.chess_box_points = [(int(point[0]), int(point[1])) for point in actual_corners]

            # 显示信息
            info_text = f"✅ 检测到4个圆形标记 (平均半径: {avg_radius}px):\n"
            corner_names = ["左上", "右上", "右下", "左下"]
            for i, (center, corner) in enumerate(zip(ordered_centers, actual_corners)):
                info_text += f"  {corner_names[i]}: 圆心({int(center[0])}, {int(center[1])}) -> 角点({int(corner[0])}, {int(corner[1])})\n"

            self.status_label.config(text="✅ 成功检测到棋盒4个圆形标记", fg="green")
            print(info_text)

            # 在图像上绘制检测到的圆形和计算出的角点
            for i, (center, corner) in enumerate(zip(ordered_centers, actual_corners)):
                # 绘制检测到的圆形
                cv2.circle(self.current_frame, (int(center[0]), int(center[1])), avg_radius, (0, 255, 0), 2)
                # 绘制圆心
                cv2.circle(self.current_frame, (int(center[0]), int(center[1])), 3, (0, 0, 255), -1)
                # 绘制计算出的角点
                cv2.circle(self.current_frame, (int(corner[0]), int(corner[1])), 5, (255, 0, 0), -1)
        else:
            self.chess_box_points = []
            self.status_label.config(text=f"❌ 无法确定4个角点 (找到{len(selected_circles)}个合适圆形)", fg="red")

    def update_frame(self):
        """更新视频帧到 canvas"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()


        if color_frame and depth_frame:
            # 保存彩色帧和深度帧用于拍照
            self.original_frame = np.asanyarray(color_frame.get_data())
            self.depth_frame = depth_frame

            # 应用实时矫正
            if self.apply_correction and self.M is not None:
                # 应用透视矫正
                # self.current_frame = cv2.warpPerspective(self.original_frame, self.M, (1280, 720))
                self.current_frame,_ = correct_chessboard_to_square(self.original_frame,CHESS_POINTS_R,self.inverse_matrix)
                self.original_frame = self.current_frame
            elif self.apply_correction and self.mtx is not None and self.dist is not None:
                # 回退到畸变矫正
                h, w = self.original_frame.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
                self.current_frame = cv2.undistort(self.original_frame, self.mtx, self.dist, None, newcameramtx)
                self.original_frame = self.current_frame

            else:
                self.current_frame = self.original_frame.copy()

            # 绘制收棋盒角点
            if self.chess_box_points:
                # 每4个点为一个棋盒
                for i in range(0, len(self.chess_box_points), 4):
                    # 为每个棋盒使用不同颜色
                    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255)]
                    color = colors[(i // 4) % len(colors)]

                    # 绘制棋盒的四个角点
                    points = []
                    for j in range(4):
                        if i + j < len(self.chess_box_points):
                            point = self.chess_box_points[i + j]
                            points.append(point)
                            cv2.circle(self.current_frame, point, 5, color, -1)
                            cv2.putText(self.current_frame, f"{j + 1}", (point[0] + 10, point[1] + 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # 绘制矩形边框
                    if len(points) == 4:
                        points = np.array(points, np.int32)
                        cv2.polylines(self.current_frame, [points], True, color, 2)

            # 自动拍照逻辑
            if self.auto_capturing:
                if self.countdown <= 0:
                    if self.captured < MAX_IMAGES:
                        self.save_image()  # 调用拍照保存函数
                        self.countdown = AUTO_CAPTURE_INTERVAL  # 重置倒计时
                    else:
                        # 达到最大拍照数量，停止自动拍照
                        self.auto_capturing = False
                        self.manual_button.config(state=tk.NORMAL)
                        self.status_label.config(text=f"已达到最大拍摄数量 {MAX_IMAGES} 张", fg="orange")
                else:
                    self.countdown -= 15  # 减少15毫秒（大约是update_frame的调用间隔）

            # # 绘制倒计时
            # if self.auto_capturing and self.countdown > 0:
            #     countdown_text = f"倒计时: {self.countdown // 1000}s"
            #     cv2.putText(self.current_frame, countdown_text, (20, 50),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 判断图像是否大于 1280x720 并进行缩放
            display_frame = self.current_frame.copy()
            if display_frame.shape[1] > 1280 or display_frame.shape[0] > 720:
                display_frame = cv2.resize(display_frame, (1280, 720), interpolation=cv2.INTER_AREA)

            # 检测并绘制棋盘格
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SHAPE, None)

            if ret:
                # 绘制棋盘格角点
                cv2.drawChessboardCorners(display_frame, CHESSBOARD_SHAPE, corners, ret)

                # 可选：绘制棋盘格网格连接线
                corners = corners.reshape(-1, 2)
                for i in range(CHESSBOARD_SHAPE[1]):  # 遍历每行
                    for j in range(CHESSBOARD_SHAPE[0]):  # 遍历每列
                        if j < CHESSBOARD_SHAPE[0] - 1:
                            start_point = tuple(corners[i * CHESSBOARD_SHAPE[0] + j].astype(int))
                            end_point = tuple(corners[i * CHESSBOARD_SHAPE[0] + j + 1].astype(int))
                            cv2.line(display_frame, start_point, end_point, (0, 255, 0), 1)
                        if i < CHESSBOARD_SHAPE[1] - 1:
                            start_point = tuple(corners[i * CHESSBOARD_SHAPE[0] + j].astype(int))
                            end_point = tuple(corners[(i + 1) * CHESSBOARD_SHAPE[0] + j].astype(int))
                            cv2.line(display_frame, start_point, end_point, (0, 255, 0), 1)

            # 转换为 Tkinter 图像格式并显示
            img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.photo = img_tk
            self.canvas.create_image(0, 0, image=img_tk, anchor='nw')


        self.root.after(15, self.update_frame)



if __name__ == "__main__":
    root = tk.Tk()
    app = CalibrationApp(root)
    app.init()
    root.mainloop()
