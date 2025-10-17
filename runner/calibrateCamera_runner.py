import tkinter as tk
from tkinter import messagebox
import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime
from PIL import Image, ImageTk

from src.cchessYolo.fruit_yolo_obb_trainer import FruitOBBTrainer
from utils.calibrationManager import pixel_to_world, calculate_perspective_transform_matrices, multi_camera_pixel_to_world
from utils.corrected import correct_chessboard_to_square
from parameters import CHESS_POINTS_R, WORLD_POINTS_R
from src.cchessYolo.chess_detection_trainer import ChessPieceDetectorSeparate
from src.cchessYolo.detect_chess_box import select_corner_circles, order_points, calculate_box_corners
from manager.camera_manager import CameraManager

dir = os.path.dirname(os.path.abspath(__file__))

# ================== 配置参数 ==================
SQUARE_SIZE_MM = 12.5         # 棋盘格大小（单位：毫米）
CHESSBOARD_SHAPE = (7, 7)     # 内部角点数量（对应 4x4 棋盘格）
MAX_IMAGES = 100              # 最大采集图像数量
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

        # 初始化相机管理器
        self.camera_manager = CameraManager(WIDTH, HEIGHT, FPS)
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



    def init(self):
        """初始化应用"""
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

        self.label = tk.Label(self.frame, text="实时预览")
        self.label.pack()

        self.canvas = tk.Canvas(self.frame, width=WIDTH if WIDTH <= 1280 else 1280,
                               height=HEIGHT if HEIGHT <= 720 else 720)
        self.canvas.pack()

        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Enter>", lambda e: setattr(self, "show_mouse_coords", True))
        self.canvas.bind("<Leave>", lambda e: setattr(self, "show_mouse_coords", False))

        # 新增坐标显示标签
        self.coord_frame = tk.Frame(self.frame)
        self.coord_frame.pack(pady=5)

        self.coord_label = tk.Label(self.coord_frame, text="鼠标坐标: (0, 0)", fg="black")
        self.coord_label.pack(side=tk.RIGHT, padx=5)

        self.world_coord_label = tk.Label(self.coord_frame, text="世界坐标: 未标定", fg="red")
        self.world_coord_label.pack(side=tk.RIGHT, padx=5)

        self.btn_frame = tk.Frame(self.frame)
        self.btn_frame.pack(pady=10)

        # 创建按钮
        buttons = [
            ("拍照", self.toggle_manual_mode),
            ("⏱️ 自动拍照", self.toggle_auto_mode),
            ("实时矫正", self.toggle_correction),
            ("识别收棋盒", self.detect_chess_box),
            ("识别正方形", self.detect_squares),
            ("识别棋子", self.detect_chess_pieces),
            ("识别水果", self.detect_fruits),
            ("手眼标定", self.hand_eye_calibration),
            ("退出", self.stop_app)
        ]

        for i, (text, command) in enumerate(buttons):
            btn = tk.Button(self.btn_frame, text=text, command=command)
            btn.pack(side=tk.LEFT, padx=5)

        self.squares_label = tk.Label(self.frame, text="未识别到棋格...", fg="red")
        self.squares_label.pack()

        self.hand_eye_calibration_button = tk.Button(self.btn_frame, text="手眼标定", command=self.hand_eye_calibration)

        self.toggle_label = tk.Label(self.frame, text="实时矫正 已禁用", fg="blue")
        self.toggle_label.pack()

        self.status_label = tk.Label(self.frame, text="状态：等待开始...", fg="blue")
        self.status_label.pack()

    def init_camera(self):
        """初始化深度相机"""
        if not self.camera_manager.initialize_camera():
            messagebox.showerror("错误", "相机初始化失败")
            self.running = False

    def save_image(self):
        """触发保存原始帧"""
        if not hasattr(self, 'original_frame'):
            return
        if self.captured >= MAX_IMAGES:
            messagebox.showinfo("提示", f"已达到最大拍摄数量 {MAX_IMAGES} 张。")
            self.auto_capturing = False
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
        self.save_image()

    def toggle_auto_mode(self):
        """切换到自动模式"""
        if not self.auto_capturing:
            self.auto_capturing = True
            self.countdown = AUTO_CAPTURE_INTERVAL
            self.status_label.config(text=f"状态：已切换到自动拍照模式（{AUTO_CAPTURE_INTERVAL // 1000}s/张）", fg="green")
        else:
            self.auto_capturing = False
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

    def detect_squares(self):
        """检测图像中的正方形并输出四角和中心坐标，并在画面中绘制"""
        if not hasattr(self, 'current_frame'):
            self.status_label.config(text="⚠️ 未获取到图像数据", fg="red")
            return

        # 复制当前帧用于处理
        img = self.current_frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 预处理：高斯模糊和边缘检测
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        # 查找轮廓
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        squares = []

        # 遍历所有轮廓，筛选正方形
        for contour in contours:
            # 计算轮廓面积，过滤太小的轮廓
            area = cv2.contourArea(contour)
            if area < 1000:  # 可根据需要调整最小面积
                continue

            # 近似轮廓为多边形
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 如果多边形有4个顶点，可能是正方形
            if len(approx) == 4:
                # 检查是否为凸四边形
                if cv2.isContourConvex(approx):
                    # 计算边长
                    sides = []
                    points = approx.reshape(4, 2)
                    for i in range(4):
                        p1 = points[i]
                        p2 = points[(i + 1) % 4]
                        length = np.linalg.norm(p1 - p2)
                        sides.append(length)

                    # 检查四条边是否近似相等（正方形特性）
                    avg_side = np.mean(sides)
                    side_var = np.var(sides)

                    # 如果边长差异较小，则认为是正方形
                    if side_var < (avg_side * 0.3) ** 2:  # 允许30%的边长差异
                        squares.append(approx)

        # 保存检测到的正方形以便在update_frame中绘制
        self.detected_squares = squares

        # 输出坐标信息到控制台
        if len(squares) > 0:
            print(f"\n检测到 {len(squares)} 个正方形:")

            for i, square in enumerate(squares):
                points = square.reshape(4, 2)

                # 按照顺序排列四个角点（左上，右上，右下，左下）
                ordered_points = self.order_square_points(points)

                # 计算中心点
                center = np.mean(ordered_points, axis=0).astype(int)

                # 输出坐标信息
                print(f"正方形 {i + 1}:")
                corner_names = ["左上", "右上", "右下", "左下"]
                for j, (name, point) in enumerate(zip(corner_names, ordered_points)):
                    print(f"  {name}角点: ({int(point[0])}, {int(point[1])})")
                print(f"  中心点: ({int(center[0])}, {int(center[1])})")
                print()

            self.status_label.config(text=f"✅ 检测完成 - 找到 {len(squares)} 个正方形", fg="green")
        else:
            self.status_label.config(text="❌ 未检测到正方形", fg="red")

    def order_square_points(self, points):
        """
        按照顺序排列正方形的四个角点：左上，右上，右下，左下
        """
        # 计算中心点
        center = np.mean(points, axis=0)

        # 根据相对于中心点的位置对点进行排序
        top_points = []
        bottom_points = []

        for point in points:
            if point[1] < center[1]:
                top_points.append(point)
            else:
                bottom_points.append(point)

        # 对顶部和底部点分别按x坐标排序
        top_left = min(top_points, key=lambda p: p[0])
        top_right = max(top_points, key=lambda p: p[0])
        bottom_right = max(bottom_points, key=lambda p: p[0])
        bottom_left = min(bottom_points, key=lambda p: p[0])

        return np.array([top_left, top_right, bottom_right, bottom_left])

    def detect_chess_pieces(self):
        """识别棋盘上的棋子位置和高度"""
        self._detect_objects("chess")

    def detect_fruits(self):
        """识别图像中的水果"""
        self._detect_objects("fruit")

    def _detect_objects(self, object_type="chess"):
        """通用物体检测方法"""
        if not hasattr(self, 'original_frame') or not hasattr(self, 'depth_frame'):
            self.status_label.config(text="⚠️ 未获取到图像数据", fg="red")
            return

        # 根据物体类型选择正确的模型和检测器
        if object_type == "chess":
            # 象棋检测使用专门的象棋模型
            model_path = os.path.join(dir, '../src/cchessYolo/runs/detect/chess_piece_detection_separate/weights/best.pt')
            self.detector = ChessPieceDetectorSeparate(model_path)
        else:  # fruit
            # 水果检测使用OBB模型
            model_path = os.path.join(dir, '../src/cchessYolo/runs/obb/fruit_obb_detection4/weights/best.pt')
            self.detector = FruitOBBTrainer(model_path)

        # 使用相应的方法检测物体和高度信息
        if object_type == "chess":
            objects_info, _ = self.detector.detect_objects_with_height(
                self.original_frame,
                None,
                conf_threshold=0.5,
                iou_threshold=0.4
            )
        else:  # fruit
            # 对于水果检测，使用FruitOBBTrainer的predict方法
            results, detections = self.detector.predict(
                source=self.original_frame,
                conf=0.7,
                iou=0.45,
                save=False
            )
            # 将OBB检测结果转换为统一格式
            objects_info = self._convert_obb_detections(detections)

        # 创建一个可视化图像用于显示结果
        result_image = self.original_frame.copy()

        # 绘制检测到的物体信息
        for obj in objects_info:
            # 获取边界框坐标
            if 'bbox' in obj:  # 常规边界框
                x1, y1, x2, y2 = obj['bbox']
            elif 'bbox_vertices' in obj:  # OBB边界框（使用包围矩形）
                vertices = obj['bbox_vertices']
                x_coords = [vertices[i] for i in range(0, 8, 2)]
                y_coords = [vertices[i] for i in range(1, 8, 2)]
                x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
            else:
                # 默认值
                x1, y1, x2, y2 = obj.get('x1', 0), obj.get('y1', 0), obj.get('x2', 0), obj.get('y2', 0)

            class_name = obj['class_name']
            confidence = obj['confidence']
            height = obj.get('height', None)

            # 绘制边界框
            cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # 添加标签
            label = f'{class_name} {confidence:.2f}'
            cv2.putText(result_image, label, (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 显示世界坐标
            wx, wy = multi_camera_pixel_to_world((x2+x1)/2, (y2+y1)/2, self.inverse_matrix, "RED_CAMERA")
            xy_text = f'XY: {(x2+x1)/2:.0f} {(y2+y1)/2:.0f}'
            wxy_text = f'WXY: {wx:.0f} {wy:.0f}'
            cv2.putText(result_image, xy_text, (int(x1)-20, int(y2) -40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(result_image, wxy_text, (int(x1)-40, int(y2) -20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # 显示高度信息（如果可用）
            if height is not None:
                height_text = f'H: {height:.3f}m'
                cv2.putText(result_image, height_text, (int(x1), int(y2) + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 显示结果（只显示一帧，2秒后自动关闭）
        window_title = "Chess Piece Detection with Height" if object_type == "chess" else "Fruit Detection"
        cv2.imshow(window_title, result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 输出检测结果摘要
        detected_count = len(objects_info)
        status_text = f"✅ 象棋检测完成 - 检测到 {detected_count} 个棋子" if object_type == "chess" else f"✅ 水果检测完成 - 检测到 {detected_count} 个水果"
        self.status_label.config(text=status_text, fg="green")

        # 打印详细检测信息
        obj_name = "棋子" if object_type == "chess" else "水果"
        print(f"\n检测到 {detected_count} 个{obj_name}:")
        for i, obj in enumerate(objects_info):
            x, y = obj.get('center', (0, 0))
            height_info = f"{obj.get('height', 'N/A'):.3f}m" if obj.get('height') is not None else "N/A"
            print(f"{obj_name} {i+1}: {obj['class_name']} - 置信度: {obj['confidence']:.2f}, "
                  f"中心位置: ({x}, {y}), 高度: {height_info}")

    def _convert_obb_detections(self, detections):
        """
        将OBB检测结果转换为统一格式
        """
        converted = []
        for det in detections:
            # 提取中心点
            center_x, center_y = det['center']

            # 计算边界框坐标（如果存在bbox_vertices则使用包围矩形）
            if 'bbox_vertices' in det and len(det['bbox_vertices']) >= 8:
                vertices = det['bbox_vertices']
                x_coords = [vertices[i] for i in range(0, 8, 2)]
                y_coords = [vertices[i] for i in range(1, 8, 2)]
                x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
            elif 'bbox' in det and len(det['bbox']) >= 4:
                x1, y1, x2, y2 = det['bbox']
            else:
                # 默认值
                x1, y1, x2, y2 = center_x - 10, center_y - 10, center_x + 10, center_y + 10

            converted.append({
                'class_id': det['class_id'],
                'class_name': det['class_name'],
                'confidence': det['confidence'],
                'center': (int(center_x), int(center_y)),
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'height': None  # OBB检测不直接提供高度信息
            })
        return converted

    def apply_perspective_correction(self):
        """检测棋盘格并进行透视矫正，保存相机矩阵和畸变系数"""
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
            np.savez(os.path.join(OUTPUT_DIR, "RED_CAMERA","camera_params.npz"), mtx=mtx, dist=dist)
            self.status_label.config(text="✅ 从图像计算并保存标定矩阵成功", fg="green")
        else:
            self.status_label.config(text="⚠️ 相机标定失败", fg="red")

    def load_calibration_data(self):
        """
        从文件加载相机矩阵、畸变系数和透视变换矩阵
        """
        calibration_file = os.path.join(OUTPUT_DIR, "RED_CAMERA","camera_params.npz")

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
        self.toggle_label.config(text=f"实时矫正 {status}", fg="green" if self.apply_correction else "red")


    def on_mouse_move(self, event):
        """鼠标移动事件，更新坐标显示"""
        self.mouse_x = event.x
        self.mouse_y = event.y

        if self.show_mouse_coords:
            # 更新鼠标相机坐标
            coord_text = f"鼠标坐标: ({self.mouse_x}, {self.mouse_y})"

            # 获取深度值（如果可用）
            depth_text = ""
            if hasattr(self, 'depth_frame') and self.depth_frame is not None:
                # 注意：需要将显示坐标转换回原始图像坐标（如果进行了缩放）
                try:
                    # 获取深度值（单位：米）
                    depth_value = self.depth_frame.get_distance(self.mouse_x, self.mouse_y)
                    if depth_value and depth_value > 0:
                        depth_text = f", 深度: {depth_value:.3f}m"
                except:
                    pass

            self.coord_label.config(text=coord_text + depth_text)

            # 世界坐标转换保持不变
            wx, wy = pixel_to_world(self.mouse_x, self.mouse_y)
            self.world_coord_label.config(text=f"世界坐标: ({wx}, {wy})")

    def stop_app(self):
        """停止程序"""
        self.running = False
        self.camera_manager.release_camera()
        self.root.destroy()

    def on_close_window(self):
        """窗口关闭时执行清理"""
        if messagebox.askokcancel("退出", "是否要关闭程序并释放相机资源？"):
            self.running = False
            self.camera_manager.release_camera()
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
        if not self.running:
            return

        # 使用 CameraManager 获取帧
        image, depth_frame = self.camera_manager.get_frame()

        if image is not None and depth_frame is not None:
            # 保存彩色帧和深度帧用于拍照
            self.original_frame = image
            self.depth_frame = depth_frame

            # 应用实时矫正
            if self.apply_correction and self.M is not None:
                # 应用透视矫正
                self.current_frame, _ = correct_chessboard_to_square(self.original_frame, CHESS_POINTS_R, self.inverse_matrix)
                self.original_frame = self.current_frame
            elif self.apply_correction and self.mtx is not None and self.dist is not None:
                # 回退到畸变矫正
                h, w = self.original_frame.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
                self.current_frame = cv2.undistort(self.original_frame, self.mtx, self.dist, None, newcameramtx)
                self.original_frame = self.current_frame
            else:
                self.current_frame = self.original_frame.copy()

            # 绘制检测到的正方形
            if hasattr(self, 'detected_squares') and self.detected_squares:
                for i, square in enumerate(self.detected_squares):
                    points = square.reshape(4, 2)

                    # 按照顺序排列四个角点（左上，右上，右下，左下）
                    ordered_points = self.order_square_points(points)

                    # 计算中心点
                    center = np.mean(ordered_points, axis=0).astype(int)

                    # 绘制正方形轮廓
                    cv2.drawContours(self.current_frame, [square], -1, (0, 255, 0), 2)

                    # 绘制四个角点
                    for j, point in enumerate(ordered_points):
                        x, y = int(point[0]), int(point[1])
                        cv2.circle(self.current_frame, (x, y), 5, (0, 0, 255), -1)
                        cv2.putText(self.current_frame, f"{j + 1}", (x + 10, y + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    # 绘制中心点
                    cv2.circle(self.current_frame, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)
                    cv2.putText(self.current_frame, "C", (int(center[0]) + 10, int(center[1]) + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

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
                        self.status_label.config(text=f"已达到最大拍摄数量 {MAX_IMAGES} 张", fg="orange")
                else:
                    self.countdown -= 15  # 减少15毫秒（大约是update_frame的调用间隔）

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
