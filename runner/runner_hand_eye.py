import os
import cv2
import numpy as np
from math import *
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys

from dobot.dobot_control import URController
from parameters import FRUIT_CAMERA,RED_CAMERA,IO_QI


# 添加机械臂控制模块导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

dir = os.path.dirname(os.path.abspath(__file__))

class HandEyeCalibration:
    def __init__(self,filename = 'camera_params.npz'):
        # 相机内参矩阵 (示例参数，实际使用时需要相机标定获得)
        self.K = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype=np.float64)

        # 畸变参数
        self.distortion = np.array([[0, 0, 0.0, 0.0, 0]])
        self.filename = os.path.join(dir, 'calibration', 'output', filename)

        # 相机外参矩阵 (初始为单位矩阵)
        self.R_camera2base = np.eye(3, dtype=np.float64)  # 旋转矩阵
        self.T_camera2base = np.zeros((3, 1), dtype=np.float64)  # 平移向量

        # 棋盘格参数
        self.target_x_number = 5  # 棋盘格内角点x方向数量
        self.target_y_number = 8  # 棋盘格内角点y方向数量
        self.board_size = (self.target_x_number, self.target_y_number)

        # 初始化时尝试加载相机参数
        self.load_camera_parameters()

    def save_camera_parameters(self, filename=None):
        """
        保存相机参数到文件

        Args:
            filename: 保存的文件名，如果为None则使用实例的filename属性
        """
        # 如果没有提供文件名，使用实例属性
        if filename is None:
            filename = self.filename

        # 确保目录存在
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # 准备保存的数据
        save_data = {
            'mtx': self.K,
            'dist': self.distortion,
            'r_mtx': self.R_camera2base,
            'vector': self.T_camera2base
        }

        # 如果存在外部参数和投影矩阵，则也保存它们
        if hasattr(self, 'external') and self.external is not None:
            save_data['external'] = self.external
        if hasattr(self, 'projection_matrix') and self.projection_matrix is not None:
            save_data['projection_matrix'] = self.projection_matrix
        if hasattr(self, 's_arr') and self.s_arr is not None:
            save_data['s_arr'] = self.s_arr

        np.savez(filename, **save_data)
        print(f"相机参数已保存到 {filename}")

    def load_camera_parameters(self,filename=None):
        """
        从文件加载相机参数

        Args:
            filename: 保存的文件名
        """
        if filename is None:
            filename = self.filename

        if os.path.exists(filename):
            try:
                data = np.load(filename, allow_pickle=True)
                self.K = data['mtx']
                self.distortion = data['dist']

                # 如果文件中包含外参矩阵，则加载它们
                if 'r_mtx' in data:
                    self.R_camera2base = data['r_mtx']
                if 'vector' in data:
                    self.T_camera2base = data['vector']

                # 如果文件中包含外部参数和投影矩阵，则加载它们
                if 'external' in data:
                    self.external = data['external']
                if 'projection_matrix' in data:
                    self.projection_matrix = data['projection_matrix']
                if 's_arr' in data:
                    self.s_arr = data['s_arr']

                print(f"相机参数已从 {filename} 加载")
                return True
            except Exception as e:
                print(f"加载相机参数时出错: {e}")
                return False
        else:
            print(f"未找到相机参数文件 {filename}")
            return False

    def draw_chessboard_corners(self, img):
        """输入图像，输出绘制棋格角点的图像

        Args:
            img: 输入的图像数组

        Returns:
            img_with_corners: 绘制了检测到的棋盘格角点的图像
            corners: 棋盘格角点坐标，如果没有检测到则返回None
        """
        try:
            # 检测棋盘格角点
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(
                gray,
                (self.target_x_number, self.target_y_number),
                None
            )

            if ret:
                # 亚像素精度优化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # 绘制角点
                img_with_corners = img.copy()
                cv2.drawChessboardCorners(
                    img_with_corners,
                    (self.target_x_number, self.target_y_number),
                    corners,
                    ret
                )
                return img_with_corners, corners
            else:
                return img, None

        except Exception as e:
            # 如果处理过程中出现错误，返回原图和None
            return img, None

    def calculate_extrinsics_zhang(self, world_points_list, pixel_points_list):
        """
        使用张正友标定法计算相机外参

        Args:
            world_points_list: 世界坐标点列表 (每项为 Nx3 数组)
            pixel_points_list: 对应的像素坐标点列表 (每项为 Nx2 数组)

        Returns:
            R: 旋转矩阵
            T: 平移向量
        """
        if len(world_points_list) != len(pixel_points_list):
            raise ValueError("世界坐标点和像素坐标点数量不匹配")

        if len(world_points_list) < 3:
            raise ValueError("至少需要3组对应点进行标定")

        # 将所有点组合成一个大数组
        all_world_points = np.vstack(world_points_list)
        all_pixel_points = np.vstack(pixel_points_list)

        # 使用 solvePnP 计算外参
        success, rvec, tvec = cv2.solvePnP(
            all_world_points.astype(np.float32),
            all_pixel_points.astype(np.float32),
            self.K.astype(np.float32),
            self.distortion.astype(np.float32),
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            raise RuntimeError("外参标定失败")

        # 将旋转向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)

        # 计算缩放因子s_arr
        self.s_arr = np.array([0], dtype=np.float32)
        total_points_used = len(all_world_points)

        # 计算投影矩阵
        external_matrix = np.column_stack((R, tvec))
        projection_matrix = self.K.dot(external_matrix)

        # 计算每个点的缩放因子
        for i in range(total_points_used):
            XYZ1 = np.array([[all_world_points[i, 0], all_world_points[i, 1], all_world_points[i, 2], 1]], dtype=np.float32)
            XYZ1 = XYZ1.T
            suv1 = projection_matrix.dot(XYZ1)
            s = suv1[2, 0]
            self.s_arr = np.array([s / total_points_used + self.s_arr[0]], dtype=np.float32)

        # 保存外参矩阵和投影矩阵
        self.R_camera2base = R
        self.T_camera2base = tvec
        self.external = external_matrix
        self.projection_matrix = projection_matrix

        return R, tvec

    def pixel_to_world(self, pixel_x, pixel_y):
        """将像素坐标转换为世界坐标"""
        if self.R_camera2base is None or self.T_camera2base is None:
            raise ValueError("请先执行手眼标定")

        # 获取缩放因子
        scalingfactor = self.s_arr[0] if hasattr(self, 's_arr') else 1.0

        # 计算相机内参矩阵的逆
        inverse_K = np.linalg.inv(self.K)

        # 计算旋转矩阵的逆
        inverse_R = np.linalg.inv(self.R_camera2base)

        # 构建像素坐标齐次坐标
        uv_1 = np.array([[pixel_x, pixel_y, 1]], dtype=np.float32)
        uv_1 = uv_1.T

        # 缩放像素坐标
        suv_1 = scalingfactor * uv_1

        # 转换到相机坐标系
        xyz_c = inverse_K.dot(suv_1)

        # 减去平移向量
        xyz_c = xyz_c - self.T_camera2base

        # 转换到世界坐标系
        XYZ = inverse_R.dot(xyz_c)

        # 返回世界坐标XY值
        world_x, world_y = XYZ[0, 0], XYZ[1, 0]

        return world_x, world_y


    def generate_chessboard_image(self, output_path='calibration/output/chessboard.png',
                                 board_size=(7, 7), square_size=100, dpi=300):
        """
        生成黑白棋盘格图像

        Args:
            output_path: 输出图像路径
            board_size: 棋盘格尺寸(内角点数)
            square_size: 每个格子的像素大小
            dpi: 图像分辨率
        """
        # 计算图像尺寸
        width = board_size[0] * square_size + 1 * square_size
        height = board_size[1] * square_size + 1 * square_size

        # 创建白色背景图像
        img = np.ones((height, width), dtype=np.uint8) * 255

        # 绘制黑白棋盘格
        for i in range(board_size[1] + 1):
            for j in range(board_size[0] + 1):
                if (i + j) % 2 == 0:
                    start_y = i * square_size + square_size
                    end_y = start_y + square_size
                    start_x = j * square_size + square_size
                    end_x = start_x + square_size
                    img[start_y:end_y, start_x:end_x] = 0  # 黑色方块

        # 确保输出目录存在
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # 保存图像
        cv2.imwrite(output_path, img)
        print(f"棋盘格图像已保存到 {output_path}")
        return img

    def generate_calibration_points_image(self, image_path, output_path='calibration/output/calibration_points.png'):
        """
        在图像上绘制九点标定的点位

        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径
        """
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 绘制标定点
        for i, point_data in enumerate(self.zhang_points_data):
            if point_data and 'pixel_x' in point_data and 'pixel_y' in point_data:
                # 获取像素坐标
                x = int(point_data['pixel_x'])
                y = int(point_data['pixel_y'])

                # 绘制点位
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # 红色实心圆
                cv2.circle(img, (x, y), 10, (0, 255, 0), 2)   # 绿色圆环

                # 添加点编号
                cv2.putText(img, f"P{i+1}", (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 确保输出目录存在
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # 保存图像
        cv2.imwrite(output_path, img)
        print(f"标定点图像已保存到 {output_path}")
        return img

    def pixel_to_world_nine_points(self, pixel_x, pixel_y):
        """
        使用九点标定数据将像素坐标转换为世界坐标

        Args:
            pixel_x: 像素x坐标
            pixel_y: 像素y坐标

        Returns:
            tuple: (world_x, world_y) 世界坐标
        """
        # 这个方法需要从外部获取九点标定数据
        # 由于这个类没有直接访问GUI中的九点标定数据，我们需要通过其他方式传递
        raise NotImplementedError("需要从GUI传递九点标定数据")
class HandEyeCalibrationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("手眼标定系统")
        self.root.geometry("1200x800")

        # 配置按钮样式（可选）
        style = ttk.Style()
        style.configure("Red.TButton", foreground="red")

        self.calibrator = HandEyeCalibration()
        self.current_image = None
        self.calibration_data = []
        self.image_files = []              # 新增
        self.current_image_index = -1       # 新增
        self.robot_speed = tk.DoubleVar(value=50.0)  # 默认速度50%
        self.suction_state = False  # 吸取状态：False=释放状态，True=吸取状态

        # 机械臂控制器
        self.robot_controller = None
        self.robot_connected = False
        # 像素坐标转世界坐标相关属性
        self.pixel_x_var = tk.StringVar(value="0")
        self.pixel_y_var = tk.StringVar(value="0")
        self.world_x_var = tk.StringVar(value="0.0")
        self.world_y_var = tk.StringVar(value="0.0")

        # 张正友标定点数据 (初始化为9个空点)
        self.zhang_points_data = [{} for _ in range(9)]
        self.selected_point_index = 0

        self.setup_ui()
        # 初始化后更新界面显示
        self.update_camera_params_display()
        # 自动加载保存的标定点数据
        self.load_zhang_points_from_json()

    def setup_ui(self):
        """设置用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建包含折叠按钮和控制面板的容器
        control_wrapper = ttk.Frame(main_frame)
        control_wrapper.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 控制面板标题和折叠按钮 - 放在wrapper容器中，而不是control_container中
        control_header = ttk.Frame(control_wrapper)
        control_header.pack(fill=tk.X)

        control_title = ttk.Label(control_header, text="控制面板", font=('Arial', 12, 'bold'))
        control_title.pack(side=tk.LEFT)

        self.collapse_button = ttk.Button(control_header, text="◀", width=3, command=self.toggle_control_panel)
        self.collapse_button.pack(side=tk.RIGHT)

        # 左侧控制面板容器
        self.control_container = ttk.Frame(control_wrapper)
        self.control_container.pack(side=tk.LEFT, fill=tk.Y)

        # 创建可滚动的控制面板
        self.setup_scrollable_control_panel()

        # 右侧显示区域
        self.display_frame = ttk.Frame(main_frame)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 图像显示
        self.image_label = ttk.Label(self.display_frame, text="请加载图像")
        self.image_label.pack(pady=10)

        # 结果显示
        result_frame = ttk.LabelFrame(self.display_frame, text="标定结果")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.result_text = tk.Text(result_frame, height=15)
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_scrollable_control_panel(self):
        """设置可滚动的控制面板"""
        # 创建Canvas和滚动条
        self.control_canvas = tk.Canvas(self.control_container, width=300)
        self.control_scrollbar = ttk.Scrollbar(self.control_container, orient="vertical", command=self.control_canvas.yview)
        self.control_scrollable_frame = ttk.Frame(self.control_canvas)

        # 配置滚动区域
        self.control_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.control_canvas.configure(
                scrollregion=self.control_canvas.bbox("all")
            )
        )

        self.control_canvas.create_window((0, 0), window=self.control_scrollable_frame, anchor="nw", width=300)
        self.control_canvas.configure(yscrollcommand=self.control_scrollbar.set)

        self.control_canvas.pack(side="left", fill="both", expand=True)
        self.control_scrollbar.pack(side="right", fill="y")

        # 绑定鼠标滚轮事件
        self.control_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.control_scrollable_frame.bind("<MouseWheel>", self._on_mousewheel)

        # 创建控制面板内容
        self.create_control_panel_content()

    def _on_mousewheel(self, event):
        """处理鼠标滚轮事件"""
        self.control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def create_control_panel_content(self):
        """创建控制面板内容"""
        # 控制面板框架
        control_frame = ttk.LabelFrame(self.control_scrollable_frame, text="控制面板")
        control_frame.pack(fill=tk.BOTH, expand=True)

        # 添加文件名设置区域
        filename_frame = ttk.LabelFrame(control_frame, text="参数文件设置")
        filename_frame.pack(fill=tk.X, pady=5)

        ttk.Label(filename_frame, text="参数文件名:").grid(row=0, column=0, sticky=tk.W)
        self.filename_var = tk.StringVar(value="camera_params.npz")
        filename_entry = ttk.Entry(filename_frame, textvariable=self.filename_var, width=25)
        filename_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
        # 更新按钮，用于重新加载参数
        ttk.Button(filename_frame, text="重新加载", command=self.reload_camera_parameters).grid(row=0, column=2, padx=5)

        # 机械臂连接设置
        robot_frame = ttk.LabelFrame(control_frame, text="机械臂控制")
        robot_frame.pack(fill=tk.X, pady=5)

        ttk.Label(robot_frame, text="IP地址:").grid(row=0, column=0, sticky=tk.W)
        self.robot_ip_var = tk.StringVar(value="192.168.5.1")
        ttk.Entry(robot_frame, textvariable=self.robot_ip_var, width=15).grid(row=0, column=1)

        self.connect_button = ttk.Button(robot_frame, text="连接机械臂", command=self.toggle_connect)
        self.connect_button.grid(row=0, column=4, columnspan=2, pady=5)

        self.robot_status_label = ttk.Label(robot_frame, text="未连接")
        self.robot_status_label.grid(row=1, column=4, columnspan=2)

        ttk.Label(robot_frame, text="速度(%):").grid(row=1, column=0, sticky=tk.W)
        speed_scale = ttk.Scale(robot_frame, from_=1, to=100, orient=tk.HORIZONTAL,
                                variable=self.robot_speed, command=self.update_speed_display)
        speed_scale.grid(row=1, column=1, sticky=tk.EW, padx=5)

        self.speed_display = ttk.Label(robot_frame, text="50%")
        self.speed_display.grid(row=1, column=2)

        # 移动控制
        move_frame = ttk.LabelFrame(control_frame, text="移动控制")
        move_frame.pack(fill=tk.X, pady=5)
        suction_frame = ttk.LabelFrame(control_frame, text="吸放控制")
        suction_frame.pack(fill=tk.X, pady=5)

        ttk.Button(move_frame, text="移动到世界坐标", command=self.move_to_world_coordinate).pack(fill=tk.X, pady=2)
        ttk.Button(move_frame, text="回家", command=self.move_home).pack(fill=tk.X, pady=2)
        self.suction_button = ttk.Button(suction_frame, text="吸取", command=self.toggle_suction)
        self.suction_button.pack(fill=tk.X, padx=5, pady=5)
        # 相机内参设置
        intrinsic_frame = ttk.LabelFrame(control_frame, text="相机内参")
        intrinsic_frame.pack(fill=tk.X, pady=5)

        ttk.Label(intrinsic_frame, text="焦距fx:").grid(row=0, column=0, sticky=tk.W)
        self.fx_var = tk.StringVar(value="0")
        ttk.Entry(intrinsic_frame, textvariable=self.fx_var, width=10).grid(row=0, column=1)

        ttk.Label(intrinsic_frame, text="焦距fy:").grid(row=1, column=0, sticky=tk.W)
        self.fy_var = tk.StringVar(value="0")
        ttk.Entry(intrinsic_frame, textvariable=self.fy_var, width=10).grid(row=1, column=1)

        ttk.Label(intrinsic_frame, text="主点cx:").grid(row=2, column=0, sticky=tk.W)
        self.cx_var = tk.StringVar(value="0")
        ttk.Entry(intrinsic_frame, textvariable=self.cx_var, width=10).grid(row=2, column=1)

        ttk.Label(intrinsic_frame, text="主点cy:").grid(row=3, column=0, sticky=tk.W)
        self.cy_var = tk.StringVar(value="0")
        ttk.Entry(intrinsic_frame, textvariable=self.cy_var, width=10).grid(row=3, column=1)

        # 相机外参设置 (可隐藏)
        self.extrinsic_frame_container = ttk.LabelFrame(control_frame, text="相机外参")
        self.extrinsic_frame_container.pack(fill=tk.X, pady=5)

        extrinsic_header = ttk.Frame(self.extrinsic_frame_container)
        extrinsic_header.pack(fill=tk.X)

        extrinsic_title = ttk.Label(extrinsic_header, text="相机外参")
        extrinsic_title.pack(side=tk.LEFT)

        self.extrinsic_toggle_btn = ttk.Button(extrinsic_header, text="隐藏", width=5, command=lambda: self.toggle_collapsible_section('extrinsic'))
        self.extrinsic_toggle_btn.pack(side=tk.RIGHT)

        self.extrinsic_content_frame = ttk.Frame(self.extrinsic_frame_container)
        self.extrinsic_content_frame.pack(fill=tk.X, pady=5)

        ttk.Label(self.extrinsic_content_frame, text="旋转矩阵 R:").grid(row=0, column=0, sticky=tk.W)
        self.r11_var = tk.StringVar(value="1.0")
        ttk.Entry(self.extrinsic_content_frame, textvariable=self.r11_var, width=8).grid(row=0, column=1)
        self.r12_var = tk.StringVar(value="0.0")
        ttk.Entry(self.extrinsic_content_frame, textvariable=self.r12_var, width=8).grid(row=0, column=2)
        self.r13_var = tk.StringVar(value="0.0")
        ttk.Entry(self.extrinsic_content_frame, textvariable=self.r13_var, width=8).grid(row=0, column=3)

        self.r21_var = tk.StringVar(value="0.0")
        ttk.Entry(self.extrinsic_content_frame, textvariable=self.r21_var, width=8).grid(row=1, column=1)
        self.r22_var = tk.StringVar(value="1.0")
        ttk.Entry(self.extrinsic_content_frame, textvariable=self.r22_var, width=8).grid(row=1, column=2)
        self.r23_var = tk.StringVar(value="0.0")
        ttk.Entry(self.extrinsic_content_frame, textvariable=self.r23_var, width=8).grid(row=1, column=3)

        self.r31_var = tk.StringVar(value="0.0")
        ttk.Entry(self.extrinsic_content_frame, textvariable=self.r31_var, width=8).grid(row=2, column=1)
        self.r32_var = tk.StringVar(value="0.0")
        ttk.Entry(self.extrinsic_content_frame, textvariable=self.r32_var, width=8).grid(row=2, column=2)
        self.r33_var = tk.StringVar(value="1.0")
        ttk.Entry(self.extrinsic_content_frame, textvariable=self.r33_var, width=8).grid(row=2, column=3)

        ttk.Label(self.extrinsic_content_frame, text="平移向量 T:").grid(row=3, column=0, sticky=tk.W)
        self.t1_var = tk.StringVar(value="0.0")
        ttk.Entry(self.extrinsic_content_frame, textvariable=self.t1_var, width=8).grid(row=3, column=1)
        self.t2_var = tk.StringVar(value="0.0")
        ttk.Entry(self.extrinsic_content_frame, textvariable=self.t2_var, width=8).grid(row=3, column=2)
        self.t3_var = tk.StringVar(value="0.0")
        ttk.Entry(self.extrinsic_content_frame, textvariable=self.t3_var, width=8).grid(row=3, column=3)

        # 棋盘格参数
        board_frame = ttk.LabelFrame(control_frame, text="棋盘格参数")
        board_frame.pack(fill=tk.X, pady=5)

        ttk.Label(board_frame, text="X方向角点数:").grid(row=0, column=0, sticky=tk.W)
        self.board_x_var = tk.StringVar(value=str(self.calibrator.target_x_number))
        ttk.Entry(board_frame, textvariable=self.board_x_var, width=10).grid(row=0, column=1)

        ttk.Label(board_frame, text="Y方向角点数:").grid(row=1, column=0, sticky=tk.W)
        self.board_y_var = tk.StringVar(value=str(self.calibrator.target_y_number))
        ttk.Entry(board_frame, textvariable=self.board_y_var, width=10).grid(row=1, column=1)

        # 机械臂位姿显示（可控制）(可隐藏)
        self.pose_frame_container = ttk.LabelFrame(control_frame, text="机械臂位姿 (mm, deg)")
        self.pose_frame_container.pack(fill=tk.X, pady=5)

        pose_header = ttk.Frame(self.pose_frame_container)
        pose_header.pack(fill=tk.X)

        pose_title = ttk.Label(pose_header, text="机械臂位姿 (mm, deg)")
        pose_title.pack(side=tk.LEFT)

        self.pose_toggle_btn = ttk.Button(pose_header, text="隐藏", width=5, command=lambda: self.toggle_collapsible_section('pose'))
        self.pose_toggle_btn.pack(side=tk.RIGHT)

        self.pose_content_frame = ttk.Frame(self.pose_frame_container)
        self.pose_content_frame.pack(fill=tk.X, pady=5)

        pose_labels = ["X:", "Y:", "Z:", "RX:", "RY:", "RZ:"]
        self.pose_vars = []
        self.pose_labels_widgets = []

        for i, label in enumerate(pose_labels):
            frame = ttk.Frame(self.pose_content_frame)
            frame.grid(row=i, column=0, columnspan=3, sticky=tk.W, pady=2)

            ttk.Label(frame, text=label, width=3).pack(side=tk.LEFT)

            var = tk.StringVar(value="0.0")
            label_widget = ttk.Label(frame, textvariable=var, width=10, relief=tk.SUNKEN, anchor=tk.E)
            label_widget.pack(side=tk.LEFT, padx=(0, 5))

            dec_btn = ttk.Button(frame, text="-", width=3)
            dec_btn.pack(side=tk.LEFT, padx=(0, 2))

            axis_mapping = {0: "X-", 1: "Y-", 2: "Z-", 3: "Rx-", 4: "Ry-", 5: "Rz-"}
            axis_id = axis_mapping.get(i, f"Axis{i}-")

            dec_btn.bind("<ButtonPress-1>", lambda e, axis=axis_id: self.start_jog(axis))
            dec_btn.bind("<ButtonRelease-1>", lambda e: self.stop_jog())

            inc_btn = ttk.Button(frame, text="+", width=3)
            inc_btn.pack(side=tk.LEFT)

            axis_mapping = {0: "X+", 1: "Y+", 2: "Z+", 3: "Rx+", 4: "Ry+", 5: "Rz+"}
            axis_id = axis_mapping.get(i, f"Axis{i}+")

            inc_btn.bind("<ButtonPress-1>", lambda e, axis=axis_id: self.start_jog(axis))
            inc_btn.bind("<ButtonRelease-1>", lambda e: self.stop_jog())

            self.pose_vars.append(var)
            self.pose_labels_widgets.append(label_widget)

        # 像素坐标转换区域
        pixel_frame = ttk.LabelFrame(control_frame, text="像素坐标转世界坐标")
        pixel_frame.pack(fill=tk.X, pady=5)

        # 添加转换方式选择下拉框
        ttk.Label(pixel_frame, text="转换方式:").grid(row=0, column=0, sticky=tk.W)
        self.conversion_method_var = tk.StringVar(value="matrix")
        conversion_method_combo = ttk.Combobox(pixel_frame, textvariable=self.conversion_method_var,
                                              values=["matrix", "nine_points"], state="readonly", width=12)
        conversion_method_combo.grid(row=0, column=1, columnspan=2, padx=5, sticky=tk.W)
        conversion_method_combo.set("matrix")  # 默认选择矩阵转换

        ttk.Label(pixel_frame, text="像素:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(pixel_frame, textvariable=self.pixel_x_var, width=10).grid(row=1, column=1, padx=5)
        ttk.Entry(pixel_frame, textvariable=self.pixel_y_var, width=10).grid(row=1, column=2, padx=5)

        ttk.Button(pixel_frame, text="转换", command=self.convert_pixel_to_world).grid(row=1, column=3, padx=5)

        ttk.Label(pixel_frame, text="世界:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(pixel_frame, textvariable=self.world_x_var, width=10).grid(row=2, column=1, padx=5)
        ttk.Entry(pixel_frame, textvariable=self.world_y_var, width=10).grid(row=2, column=2, padx=5)

        # 九点标定区域 (可隐藏)
        self.zhang_frame_container = ttk.LabelFrame(control_frame, text="九点标定")
        self.zhang_frame_container.pack(fill=tk.X, pady=5)

        zhang_header = ttk.Frame(self.zhang_frame_container)
        zhang_header.pack(fill=tk.X)

        zhang_title = ttk.Label(zhang_header, text="九点标定")
        zhang_title.pack(side=tk.LEFT)

        self.zhang_toggle_btn = ttk.Button(zhang_header, text="隐藏", width=5, command=lambda: self.toggle_collapsible_section('zhang'))
        self.zhang_toggle_btn.pack(side=tk.RIGHT)

        self.zhang_content_frame = ttk.Frame(self.zhang_frame_container)
        self.zhang_content_frame.pack(fill=tk.X, pady=5)

        point_selection_frame = ttk.Frame(self.zhang_content_frame)
        point_selection_frame.pack(fill=tk.X, pady=2)

        ttk.Label(point_selection_frame, text="标定点:").pack(side=tk.LEFT)

        self.point_selector = ttk.Combobox(point_selection_frame, width=5, state="readonly")
        self.point_selector['values'] = [f"点{i+1}" for i in range(9)]
        self.point_selector.current(0)
        self.point_selector.pack(side=tk.LEFT, padx=5)
        self.point_selector.bind("<<ComboboxSelected>>", self.on_point_selected)

        point_btn_frame = ttk.Frame(point_selection_frame)
        point_btn_frame.pack(side=tk.RIGHT)

        ttk.Button(point_btn_frame, text="删除", command=self.delete_current_point).pack(side=tk.LEFT, padx=2)
        ttk.Button(point_btn_frame, text="清除所有", command=self.clear_zhang_points).pack(side=tk.LEFT, padx=2)

        coord_input_frame = ttk.Frame(self.zhang_content_frame)
        coord_input_frame.pack(fill=tk.X, pady=5)

        # 像素坐标输入 - 第一层
        pixel_coord_frame = ttk.LabelFrame(coord_input_frame, text="像素坐标")
        pixel_coord_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(pixel_coord_frame, text="X:").grid(row=0, column=0, sticky=tk.W)
        self.pixel_x_entry = ttk.Entry(pixel_coord_frame, width=10)
        self.pixel_x_entry.grid(row=0, column=1, padx=5)

        ttk.Label(pixel_coord_frame, text="Y:").grid(row=0, column=2, sticky=tk.W)
        self.pixel_y_entry = ttk.Entry(pixel_coord_frame, width=10)
        self.pixel_y_entry.grid(row=0, column=3, padx=5)

        # 世界坐标输入 - 第二层
        world_coord_frame = ttk.LabelFrame(coord_input_frame, text="世界坐标")
        world_coord_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(world_coord_frame, text="X:").grid(row=0, column=0, sticky=tk.W)
        self.world_x_entry = ttk.Entry(world_coord_frame, width=10)
        self.world_x_entry.grid(row=0, column=1, padx=5)

        ttk.Label(world_coord_frame, text="Y:").grid(row=0, column=2, sticky=tk.W)
        self.world_y_entry = ttk.Entry(world_coord_frame, width=10)
        self.world_y_entry.grid(row=0, column=3, padx=5)


        button_frame = ttk.Frame(self.zhang_content_frame)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(button_frame, text="检测角点", command=self.detect_chessboard_corners).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(button_frame, text="更新点", command=self.update_calibration_point).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)

        points_list_frame = ttk.Frame(self.zhang_content_frame)
        points_list_frame.pack(fill=tk.X, pady=2)

        ttk.Label(points_list_frame, text="标定点列表:").pack(anchor=tk.W)
        self.points_listbox = tk.Listbox(points_list_frame, height=6)
        self.points_listbox.pack(fill=tk.X)
        self.points_listbox.bind('<<ListboxSelect>>', self.on_point_listbox_select)

        ttk.Button(self.zhang_content_frame, text="执行外参标定", command=self.perform_extrinsic_calibration).pack(fill=tk.X, pady=5)


        # 操作按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="加载图像", command=self.load_image).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="加载文件夹", command=self.load_folder).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="执行内参标定", command=self.calibrate).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="生成棋盘格图像", command=self.generate_chessboard_image_gui).pack(fill=tk.X,                                                                                                        pady=2)
        ttk.Button(button_frame, text="生成标定点图像", command=self.generate_calibration_points_image_gui).pack(
            fill=tk.X, pady=2)

        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(fill=tk.X, pady=5)

        ttk.Button(nav_frame, text="上一张", command=self.prev_image).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.image_count_label = ttk.Label(nav_frame, text="0/0")
        self.image_count_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="下一张", command=self.next_image).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)

        data_frame = ttk.LabelFrame(control_frame, text="标定数据")
        data_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.data_listbox = tk.Listbox(data_frame, height=8)
        self.data_listbox.pack(fill=tk.BOTH, expand=True)

    def reload_camera_parameters(self):
        """重新加载相机参数"""
        filename = self.filename_var.get()
        if filename:
            # 更新calibrator的文件名
            self.calibrator.filename = os.path.join(dir, 'calibration', 'output', filename)
            # 重新加载参数
            if self.calibrator.load_camera_parameters():
                # 更新界面显示
                self.update_camera_params_display()
                self.result_text.insert(tk.END, f"已重新加载参数文件: {filename}\n")
            else:
                self.result_text.insert(tk.END, f"加载参数文件失败: {filename}\n")
            self.result_text.see(tk.END)

    def toggle_control_panel(self):
        """切换控制面板显示状态"""
        if hasattr(self, 'control_container'):
            if self.control_container.winfo_viewable():
                # 隐藏控制面板
                self.control_container.pack_forget()
                self.display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                self.collapse_button.config(text="▶")
            else:
                # 显示控制面板
                self.control_container.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 10))
                self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
                self.collapse_button.config(text="◀")

    def toggle_collapsible_section(self, section_name):
        """切换可折叠区域的显示/隐藏状态"""
        if section_name == 'extrinsic':
            if self.extrinsic_content_frame.winfo_viewable():
                self.extrinsic_content_frame.pack_forget()
                self.extrinsic_toggle_btn.config(text="显示")
            else:
                self.extrinsic_content_frame.pack(fill=tk.X, pady=5)
                self.extrinsic_toggle_btn.config(text="隐藏")

        elif section_name == 'pose':
            if self.pose_content_frame.winfo_viewable():
                self.pose_content_frame.pack_forget()
                self.pose_toggle_btn.config(text="显示")
            else:
                self.pose_content_frame.pack(fill=tk.X, pady=5)
                self.pose_toggle_btn.config(text="隐藏")

        elif section_name == 'zhang':
            if self.zhang_content_frame.winfo_viewable():
                self.zhang_content_frame.pack_forget()
                self.zhang_toggle_btn.config(text="显示")
            else:
                self.zhang_content_frame.pack(fill=tk.X, pady=5)
                self.zhang_toggle_btn.config(text="隐藏")

        # 更新滚动区域
        self.control_canvas.update_idletasks()
        self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))

    def toggle_connect(self):
        """切换连接状态"""
        if self.robot_connected:
            self.disconnect_robot()
        else:
            self.connect_robot()

    def connect_robot(self):
        """连接机械臂"""
        try:
            ip = self.robot_ip_var.get()
            self.robot_controller = URController(ip=ip)
            if self.robot_controller.is_connected():
                self.robot_connected = True
                self.connect_button.config(text="断开机械臂", style="Red.TButton")  # 修改按钮文本
                self.robot_status_label.config(text="已连接", foreground="green")
                self.result_text.insert(tk.END, f"机械臂连接成功: {ip}\n")
                # 启动位姿更新定时器
                self.start_pose_update_timer()
            else:
                self.robot_status_label.config(text="连接失败", foreground="red")
                self.result_text.insert(tk.END, f"机械臂连接失败: {ip}\n")
        except Exception as e:
            self.robot_status_label.config(text="连接错误", foreground="red")
            self.result_text.insert(tk.END, f"连接机械臂时出错: {str(e)}\n")
        self.result_text.see(tk.END)

    def disconnect_robot(self):
        """断开机械臂连接"""
        try:
            if self.robot_controller:
                self.robot_controller.disconnect()
                self.robot_controller = None

            self.robot_connected = False
            self.connect_button.config(text="连接机械臂", style="")  # 恢复按钮文本
            self.robot_status_label.config(text="未连接", foreground="black")
            self.result_text.insert(tk.END, "机械臂已断开连接\n")
        except Exception as e:
            self.result_text.insert(tk.END, f"断开连接时出错: {str(e)}\n")
        self.result_text.see(tk.END)

    def start_pose_update_timer(self):
        """启动位姿更新定时器"""
        if self.robot_connected:
            self.update_robot_pose_display()
            self.root.after(1000, self.start_pose_update_timer)  # 每秒更新一次

    def update_robot_pose_display(self):
        """更新机械臂位姿显示"""
        if self.robot_connected and self.robot_controller:
            try:
                # 获取机械臂当前位置
                current_pos = self.robot_controller.get_current_position()
                if current_pos is not None:
                    # 更新界面显示的实际位姿
                    for i in range(min(6, len(current_pos))):
                        self.pose_vars[i].set(str(round(current_pos[i], 3)))
            except Exception as e:
                self.result_text.insert(tk.END, f"获取机械臂位姿时出错: {str(e)}\n")
                self.result_text.see(tk.END)

    def start_jog(self, axis_id):
        """开始点动"""
        if not self.robot_connected:
            messagebox.showerror("错误", "请先连接机械臂")
            return

        try:
            # 设置速度
            speed = int(self.robot_speed.get())
            self.robot_controller.set_speed(speed / 100.0)

            # 发送点动指令
            self.robot_controller.move_jog(axis_id)
            self.result_text.insert(tk.END, f"开始点动: {axis_id} (速度: {speed}%)\n")
            self.result_text.see(tk.END)
        except Exception as e:
            messagebox.showerror("错误", f"点动失败: {str(e)}")

    def stop_jog(self):
        """停止点动"""
        if not self.robot_connected:
            return

        try:
            # 发送停止点动指令（通常用空字符串或特定指令）
            self.robot_controller.move_jog("")
            self.result_text.insert(tk.END, "停止点动\n")
            self.result_text.see(tk.END)
        except Exception as e:
            self.result_text.insert(tk.END, f"停止点动失败: {str(e)}\n")
            self.result_text.see(tk.END)

    def decrease_pose(self, index):
        """减少指定轴的位姿值"""
        self._change_pose_jog(index, "-")

    def increase_pose(self, index):
        """增加指定轴的位姿值"""
        self._change_pose_jog(index, "+")

    def _change_pose_jog(self, index, direction):
        """通过点动改变位姿值"""
        if not self.robot_connected:
            messagebox.showerror("错误", "请先连接机械臂")
            return

        # 定义轴ID映射关系
        axis_mapping = {
            0: "X",   # X轴
            1: "Y",   # Y轴
            2: "Z",   # Z轴
            3: "Rx",  # Rx轴
            4: "Ry",  # Ry轴
            5: "Rz"   # Rz轴
        }

        try:
            # 构造点动指令 (例如 "X+", "Y-", "J1+"等)
            axis_id = f"{axis_mapping.get(index, 'X')}{direction}"

            # 发送点动指令
            self.start_jog(axis_id)

            # 注意：这里不需要等待，因为点动是持续动作直到发送停止指令

        except Exception as e:
            messagebox.showerror("错误", f"点动失败: {str(e)}")

    # 添加速度显示更新方法
    def update_speed_display(self, value):
        """更新速度显示"""
        speed = int(float(value))
        self.speed_display.config(text=f"{speed}%")

    # 修改 move_to_current_pose 方法，在移动前设置速度
    def move_to_current_pose(self):
        """移动到当前输入的位姿"""
        if not self.robot_connected:
            messagebox.showerror("错误", "请先连接机械臂")
            return

        try:
            # 设置速度
            speed = int(self.robot_speed.get())
            self.robot_controller.set_speed(speed / 100.0)

            # 获取当前显示的位姿值
            pose = [float(var.get()) for var in self.pose_vars]
            x, y, z, rx, ry, rz = pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]

            # 移动机械臂到指定位置
            success = self.robot_controller.move_to(x, y, z)
            if success:
                self.result_text.insert(tk.END, f"机械臂已移动到: X={x}, Y={y}, Z={z} (速度: {speed}%)\n")
            else:
                self.result_text.insert(tk.END, f"机械臂移动失败\n")
        except Exception as e:
            messagebox.showerror("错误", f"移动机械臂时出错: {str(e)}")
        self.result_text.see(tk.END)

    # 修改 move_home 方法，在移动前设置速度
    def move_home(self):
        """机械臂回家"""
        if not self.robot_connected:
            messagebox.showerror("错误", "请先连接机械臂")
            return

        try:
            # 设置速度
            speed = int(self.robot_speed.get())
            self.robot_controller.set_speed(speed / 100.0)

            self.robot_controller.run_point_j(RED_CAMERA)
            self.result_text.insert(tk.END, f"机械臂已回家 (速度: {speed}%)\n")
        except Exception as e:
            messagebox.showerror("错误", f"机械臂回家时出错: {str(e)}")
        self.result_text.see(tk.END)

    def toggle_suction(self):
        """切换吸放状态"""
        if not self.robot_connected:
            messagebox.showerror("错误", "请先连接机械臂")
            return

        try:
            if not self.suction_state:
                # 当前为释放状态，执行吸取
                self.robot_controller.set_do(IO_QI, 1)  # 设置IO_QI为1，吸取
                self.suction_button.config(text="释放", style="Red.TButton")  # 更改按钮文本和样式
                self.suction_state = True
                self.result_text.insert(tk.END, "发送吸取指令\n")
            else:
                # 当前为吸取状态，执行释放
                self.robot_controller.set_do(IO_QI, 0)  # 设置IO_QI为0，释放
                self.suction_button.config(text="吸取", style="")  # 恢复按钮文本和样式
                self.suction_state = False
                self.result_text.insert(tk.END, "发送释放指令\n")

            self.result_text.see(tk.END)
        except Exception as e:
            messagebox.showerror("错误", f"吸放控制失败: {str(e)}")

    def load_image(self):
        """加载图像"""
        default_path = os.path.expanduser(os.path.join(dir,'calibration/images'))  # 使用用户图片目录作为默认路径
        filename = filedialog.askopenfilename(
            title="选择图像文件",
            initialdir=default_path,  # 添加默认路径
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp")]
        )

        if filename:
            # 显示检测到的角点
            self.current_image = cv2.imread(filename)
            if self.current_image is not None:
                # 显示图像
                self.display_image(self.current_image)
                self.result_text.insert(tk.END, f"已加载图像: {filename}\n")

                self.image_files.append(filename)
                self.current_image_index += 1
                # 更新图像计数显示
                self.image_count_label.config(text=f"{len(self.image_files)}/{len(self.image_files)}")

    def load_folder(self):
        """加载文件夹中的所有图像"""
        default_path = os.path.expanduser(os.path.join(dir,'calibration/images'))  # 使用用户图片目录作为默认路径

        folder_path = filedialog.askdirectory( initialdir=default_path, title="选择包含图像的文件夹")

        if folder_path:
            # 支持的图像格式
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

            # 获取文件夹中所有图像文件
            image_files = []
            for file in os.listdir(folder_path):
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(folder_path, file))

            if not image_files:
                messagebox.showwarning("警告", "文件夹中未找到图像文件")
                return

            # 加载所有图像路径
            self.image_files = image_files
            self.current_image_index = 0

            # 加载第一张图像
            self.load_image_from_list()

            self.result_text.insert(tk.END, f"已加载文件夹: {folder_path}\n")
            self.result_text.insert(tk.END, f"共找到 {len(image_files)} 张图像\n")


    def load_image_from_list(self):
        """从图像列表中加载当前图像"""
        if hasattr(self, 'image_files') and self.image_files:
            if 0 <= self.current_image_index < len(self.image_files):
                image_path = self.image_files[self.current_image_index]
                self.current_image = cv2.imread(image_path)
                if self.current_image is not None:
                    self.display_image(self.current_image)
                    self.result_text.insert(tk.END,
                        f"已加载图像 ({self.current_image_index + 1}/{len(self.image_files)}): {os.path.basename(image_path)}\n")
                    self.result_text.see(tk.END)

                    # 更新图像计数显示
                    self.image_count_label.config(text=f"{self.current_image_index + 1}/{len(self.image_files)}")


    def next_image(self):
        """加载下一张图像"""
        if hasattr(self, 'image_files') and self.image_files:
            if self.current_image_index < len(self.image_files) - 1:
                self.current_image_index += 1
                self.load_image_from_list()

    def prev_image(self):
        """加载上一张图像"""
        if hasattr(self, 'image_files') and self.image_files:
            if self.current_image_index > 0:
                self.current_image_index -= 1
                self.load_image_from_list()

    def display_image(self, img):
        """显示图像"""
        img, _ = self.calibrator.draw_chessboard_corners(img)

        # 调整图像大小以适应显示
        height, width = img.shape[:2]
        max_width, max_height = 600, 400

        if width > max_width or height > max_height:
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_resized = cv2.resize(img, (new_width, new_height))
        else:
            img_resized = img.copy()

        # 转换为RGB格式
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        self.image_label.configure(image=img_tk, text="")
        self.image_label.image = img_tk

    # 内参
    def calibrate(self):
        """执行相机内参标定"""
        if not hasattr(self, 'image_files') or not self.image_files:
            messagebox.showerror("错误", "请先加载图像文件夹")
            return

        # 缓存 calibrator 属性引用
        calib = self.calibrator

        # 更新棋盘格参数
        try:
            calib.target_x_number = int(self.board_x_var.get())
            calib.target_y_number = int(self.board_y_var.get())
            calib.board_size = (calib.target_x_number, calib.target_y_number)
        except ValueError:
            messagebox.showerror("错误", "棋盘格参数必须是有效的数字")
            return

        # 准备标定数据
        object_points = []  # 世界坐标系中的点
        image_points = []   # 图像坐标系中的点

        # 创建世界坐标系中的棋盘格角点坐标
        total_points = calib.target_x_number * calib.target_y_number
        objp = np.zeros((total_points, 3), np.float32)
        objp[:, :2] = np.mgrid[0:calib.target_x_number, 0:calib.target_y_number].T.reshape(-1, 2)

        successful_images = 0
        self.result_text.insert(tk.END, "开始相机内参标定...\n")

        gray_shape = None  # 初始化 gray_shape 防止未定义错误

        # 遍历所有图像进行角点检测
        for idx, image_path in enumerate(self.image_files):
            img = cv2.imread(image_path)
            if img is None:
                self.result_text.insert(tk.END, f"图像 {idx+1}: 加载失败\n")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if gray_shape is None:
                gray_shape = gray.shape[::-1]

            # 查找棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, calib.board_size, None)

            if ret:
                # 精确检测角点位置
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                if refined_corners is not None:
                    corners = refined_corners

                # 添加点到标定数据中
                object_points.append(objp.copy())  # 使用 copy() 避免共享内存问题
                image_points.append(corners)
                successful_images += 1

                self.result_text.insert(tk.END, f"图像 {idx+1}: 检测到角点\n")
            else:
                self.result_text.insert(tk.END, f"图像 {idx+1}: 未检测到角点\n")

        self.result_text.see(tk.END)

        if successful_images < 3:
            messagebox.showerror("标定失败", f"成功检测角点的图像少于3张 ({successful_images} 张)，无法进行标定")
            return

        # 执行相机标定
        try:
            if gray_shape is None:
                raise RuntimeError("未能从有效图像中提取尺寸信息")

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                object_points, image_points, gray_shape, None, None)

            # 基本验证返回值有效性
            if mtx is None or dist is None:
                raise RuntimeError("OpenCV 标定返回无效结果")

            # 保存标定结果
            calib.K = mtx
            calib.distortion = dist

            # 更新UI中的相机参数显示
            self.fx_var.set(f"{mtx[0, 0]:.3f}")
            self.fy_var.set(f"{mtx[1, 1]:.3f}")
            self.cx_var.set(f"{mtx[0, 2]:.3f}")
            self.cy_var.set(f"{mtx[1, 2]:.3f}")

            # 显示标定结果
            self.result_text.insert(tk.END, "\n相机内参标定完成!\n")
            self.result_text.insert(tk.END, f"使用了 {successful_images} 张图像进行标定\n")
            self.result_text.insert(tk.END, f"重投影误差: {ret:.3f} pixels\n\n")
            self.result_text.insert(tk.END, f"相机内参矩阵 K:\n{mtx}\n\n")
            self.result_text.insert(tk.END, f"畸变系数:\n{dist}\n")
            self.result_text.see(tk.END)

            calib.save_camera_parameters(os.path.join(dir, 'calibration', 'output', self.filename_var.get()))

            messagebox.showinfo("标定完成", f"相机内参标定完成!\n重投影误差: {ret:.3f} pixels")

        except (RuntimeError, cv2.error) as specific_error:
            messagebox.showerror("标定失败", f"标定过程中出现错误: {str(specific_error)}")
            self.result_text.insert(tk.END, f"标定失败: {str(specific_error)}\n")
            self.result_text.see(tk.END)
        except Exception as general_error:
            messagebox.showerror("未知错误", f"发生未预期错误: {str(general_error)}")
            self.result_text.insert(tk.END, f"未知错误: {str(general_error)}\n")
            self.result_text.see(tk.END)

    def update_camera_params_display(self):
        """更新相机参数显示"""
        # 更新内参显示
        self.fx_var.set(str(round(self.calibrator.K[0, 0], 3)))
        self.fy_var.set(str(round(self.calibrator.K[1, 1], 3)))
        self.cx_var.set(str(round(self.calibrator.K[0, 2], 3)))
        self.cy_var.set(str(round(self.calibrator.K[1, 2], 3)))

        # 更新外参显示
        if hasattr(self.calibrator, 'R_camera2base') and self.calibrator.R_camera2base is not None:
            R = self.calibrator.R_camera2base
            T = self.calibrator.T_camera2base

            self.r11_var.set(str(round(R[0, 0], 3)))
            self.r12_var.set(str(round(R[0, 1], 3)))
            self.r13_var.set(str(round(R[0, 2], 3)))
            self.r21_var.set(str(round(R[1, 0], 3)))
            self.r22_var.set(str(round(R[1, 1], 3)))
            self.r23_var.set(str(round(R[1, 2], 3)))
            self.r31_var.set(str(round(R[2, 0], 3)))
            self.r32_var.set(str(round(R[2, 1], 3)))
            self.r33_var.set(str(round(R[2, 2], 3)))

            self.t1_var.set(str(round(T[0, 0], 3)))
            self.t2_var.set(str(round(T[1, 0], 3)))
            self.t3_var.set(str(round(T[2, 0], 3)))

    def convert_pixel_to_world(self):
        """将像素坐标转换为世界坐标"""
        try:
            pixel_x = float(self.pixel_x_var.get())
            pixel_y = float(self.pixel_y_var.get())

            # 根据选择的转换方式执行不同的转换方法
            conversion_method = self.conversion_method_var.get()

            if conversion_method == "matrix":
                # 使用矩阵转换方法
                world_x, world_y = self.calibrator.pixel_to_world(pixel_x, pixel_y)
            elif conversion_method == "nine_points":
                # 使用九点标定转换方法
                world_x, world_y = self.pixel_to_world_nine_points(pixel_x, pixel_y)
            else:
                raise ValueError(f"未知的转换方式: {conversion_method}")

            # 更新显示
            self.world_x_var.set(f"{world_x:.3f}")
            self.world_y_var.set(f"{world_y:.3f}")

            self.result_text.insert(tk.END, f"像素坐标({pixel_x}, {pixel_y}) -> 世界坐标({world_x:.3f}, {world_y:.3f}) [{conversion_method}]\n")
            self.result_text.see(tk.END)

        except ValueError as e:
            messagebox.showerror("错误", f"坐标转换失败: {str(e)}")
        except Exception as e:
            messagebox.showerror("错误", f"发生错误: {str(e)}")

    def move_to_world_coordinate(self):
        """移动到指定的世界坐标"""
        if not self.robot_connected:
            messagebox.showerror("错误", "请先连接机械臂")
            return

        try:
            # 获取当前Z坐标
            current_z = float(self.pose_vars[2].get())  # Z坐标

            # 获取世界坐标
            world_x = float(self.world_x_var.get())
            world_y = float(self.world_y_var.get())

            # 设置速度
            speed = int(self.robot_speed.get())
            self.robot_controller.set_speed(speed / 100.0)

            # 移动机械臂到指定位置
            success = self.robot_controller.move_to(world_x, world_y, current_z)
            if success:
                self.result_text.insert(tk.END, f"机械臂已移动到世界坐标: X={world_x:.3f}, Y={world_y:.3f}, Z={current_z:.3f} (速度: {speed}%)\n")

                # 更新位姿显示
                self.pose_vars[0].set(f"{world_x:.3f}")
                self.pose_vars[1].set(f"{world_y:.3f}")
            else:
                self.result_text.insert(tk.END, f"机械臂移动失败\n")

        except ValueError:
            messagebox.showerror("错误", "请输入有效的坐标值")
        except Exception as e:
            messagebox.showerror("错误", f"移动机械臂时出错: {str(e)}")

        self.result_text.see(tk.END)

    def on_point_selected(self, event=None):
        """当选择标定点时调用"""
        point_index = self.point_selector.current()
        self.load_point_data(point_index)

    def load_point_data(self, point_index):
        """加载指定标定点的数据显示"""
        if 0 <= point_index < len(self.zhang_points_data):
            point_data = self.zhang_points_data[point_index]

            # 如果有点数据，加载到输入框
            if 'pixel_x' in point_data and 'pixel_y' in point_data:
                self.pixel_x_entry.delete(0, tk.END)
                self.pixel_x_entry.insert(0, str(point_data['pixel_x']))
                self.pixel_y_entry.delete(0, tk.END)
                self.pixel_y_entry.insert(0, str(point_data['pixel_y']))

            if 'world_x' in point_data and 'world_y' in point_data:
                self.world_x_entry.delete(0, tk.END)
                self.world_x_entry.insert(0, str(point_data['world_x']))
                self.world_y_entry.delete(0, tk.END)
                self.world_y_entry.insert(0, str(point_data['world_y']))
        else:
            # 清空输入框
            self.pixel_x_entry.delete(0, tk.END)
            self.pixel_y_entry.delete(0, tk.END)
            self.world_x_entry.delete(0, tk.END)
            self.world_y_entry.delete(0, tk.END)

    def detect_chessboard_corners(self):
        """检测当前图像的棋盘格角点（修改为检测九点标定中的九个点）"""
        if not self.current_image_index >= 0 or self.current_image is None:
            messagebox.showerror("错误", "请先加载图像")
            return

        try:
            # 在当前图像中查找角点
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(
                gray,
                self.calibrator.board_size,
                None
            )

            if not ret:
                messagebox.showerror("错误", "当前图像未检测到棋盘格角点")
                return

            # 亚像素精化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # 确保有9个点（3x3网格）
            if len(corners) < 9:
                messagebox.showerror("错误", "检测到的角点少于9个")
                return

            # 使用3x3网格的9个点
            # 从棋盘格角点中选择9个点（按照3x3网格排列）
            grid_points = []
            board_w = self.calibrator.target_x_number
            board_h = self.calibrator.target_y_number

            # 选择3x3网格的角点索引
            indices = []
            for i in [0, board_h//2, board_h-1]:
                for j in [0, board_w//2, board_w-1]:
                    index = i * board_w + j
                    indices.append(index)

            # 提取9个点
            for idx in indices:
                if idx < len(corners):
                    grid_points.append(corners[idx][0])

            if len(grid_points) != 9:
                messagebox.showerror("错误", "无法提取9个标定点")
                return

            # 更新所有9个标定点的像素坐标
            for i, point in enumerate(grid_points):
                # 更新所有9个点的像素坐标
                if i < len(self.zhang_points_data):
                    self.zhang_points_data[i]['pixel_x'] = round(point[0], 2)
                    self.zhang_points_data[i]['pixel_y'] = round(point[1], 2)

            # 更新当前选中点的显示
            current_point_index = self.point_selector.current()
            self.load_point_data(current_point_index)

            # 更新列表显示
            self.update_points_listbox()

            # 在图像上绘制这9个点
            img_with_points = self.current_image.copy()
            for i, point in enumerate(grid_points):
                x, y = int(point[0]), int(point[1])
                # 绘制点位
                cv2.circle(img_with_points, (x, y), 5, (0, 0, 255), -1)  # 红色实心圆
                cv2.circle(img_with_points, (x, y), 10, (0, 255, 0), 2)   # 绿色圆环
                # 添加点编号
                cv2.putText(img_with_points, f"P{i+1}", (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # 显示图像
            self.display_image(img_with_points)

            self.result_text.insert(tk.END, f"检测到棋盘格角点，已更新9个标定点的像素坐标\n")
            self.result_text.see(tk.END)

            # 自动保存到JSON文件
            self.save_zhang_points_to_json()

        except Exception as e:
            messagebox.showerror("错误", f"角点检测失败: {str(e)}")

    def save_zhang_points_to_json(self, filename='calibration/output/zhang_points.json'):
        """
        保存张正友九点标定数据到JSON文件

        Args:
            filename: 保存的文件名
        """
        try:
            # 确保目录存在
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            # 准备要保存的数据
            data_to_save = []
            for i, point_data in enumerate(self.zhang_points_data):
                if point_data:  # 只保存非空的点数据
                    point_entry = {
                        'index': i,
                        'pixel_x': point_data.get('pixel_x'),
                        'pixel_y': point_data.get('pixel_y'),
                        'world_x': point_data.get('world_x'),
                        'world_y': point_data.get('world_y')
                    }
                    data_to_save.append(point_entry)

            # 保存到文件
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=4)

            self.result_text.insert(tk.END, f"标定点数据已保存到 {filename}\n")
            self.result_text.see(tk.END)

        except Exception as e:
            messagebox.showerror("保存失败", f"保存标定点数据时出错: {str(e)}")

    def load_zhang_points_from_json(self, filename='calibration/output/zhang_points.json'):
        """
        从JSON文件加载张正友九点标定数据

        Args:
            filename: 加载的文件名
        """
        try:
            if not os.path.exists(filename):
                self.result_text.insert(tk.END, f"未找到标定点数据文件 {filename}\n")
                self.result_text.see(tk.END)
                return False

            with open(filename, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)

            # 重置当前标定点数据
            self.zhang_points_data = [{} for _ in range(9)]

            # 加载数据
            for point_entry in loaded_data:
                index = point_entry.get('index', 0)
                if 0 <= index < 9:  # 确保索引有效
                    self.zhang_points_data[index] = {
                        'pixel_x': point_entry.get('pixel_x'),
                        'pixel_y': point_entry.get('pixel_y'),
                        'world_x': point_entry.get('world_x'),
                        'world_y': point_entry.get('world_y')
                    }

            # 更新界面显示
            self.update_points_listbox()
            # 加载当前选中的点数据到输入框
            self.load_point_data(self.point_selector.current())

            self.result_text.insert(tk.END, f"标定点数据已从 {filename} 加载\n")
            self.result_text.see(tk.END)
            return True

        except Exception as e:
            messagebox.showerror("加载失败", f"加载标定点数据时出错: {str(e)}")
            return False

    def update_calibration_point(self):
        """更新当前标定点数据"""
        try:
            point_index = self.point_selector.current()

            # 获取输入的坐标值
            pixel_x = float(self.pixel_x_entry.get())
            pixel_y = float(self.pixel_y_entry.get())
            world_x = float(self.world_x_entry.get())
            world_y = float(self.world_y_entry.get())

            # 如果该点已存在，更新数据；否则创建新点
            if point_index < len(self.zhang_points_data):
                self.zhang_points_data[point_index].update({
                    'pixel_x': pixel_x,
                    'pixel_y': pixel_y,
                    'world_x': world_x,
                    'world_y': world_y
                })
            else:
                # 扩展列表到所需大小
                while len(self.zhang_points_data) <= point_index:
                    self.zhang_points_data.append({})

                self.zhang_points_data[point_index] = {
                    'pixel_x': pixel_x,
                    'pixel_y': pixel_y,
                    'world_x': world_x,
                    'world_y': world_y
                }

            # 更新列表显示
            self.update_points_listbox()

            self.result_text.insert(tk.END, f"更新标定点 {point_index+1}: 像素({pixel_x:.2f}, {pixel_y:.2f}) -> 世界({world_x:.2f}, {world_y:.2f})\n")
            self.result_text.see(tk.END)

            # 自动保存到JSON文件
            self.save_zhang_points_to_json()

            # 自动跳到下一个点，除非是第9个点
            if point_index <= 8:  # 0-8 对应点1-9，第9个点的索引是8
                if point_index < 8:
                    next_point_index = point_index + 1
                else:
                    next_point_index = 0
                self.point_selector.current(next_point_index)
                self.load_point_data(next_point_index)
                self.result_text.insert(tk.END, f"自动跳转到标定点 {next_point_index+1}\n")
                self.result_text.see(tk.END)

        except ValueError:
            messagebox.showerror("错误", "请输入有效的坐标值")

    def delete_current_point(self):
        """删除当前标定点"""
        point_index = self.point_selector.current()

        if 0 <= point_index < len(self.zhang_points_data):
            self.zhang_points_data[point_index] = {}

            # 更新显示
            self.load_point_data(point_index)
            self.update_points_listbox()

            self.result_text.insert(tk.END, f"已删除标定点 {point_index+1}\n")
            self.result_text.see(tk.END)

    def update_points_listbox(self):
        """更新标定点列表显示"""
        self.points_listbox.delete(0, tk.END)

        for i, point_data in enumerate(self.zhang_points_data):
            if point_data and ('pixel_x' in point_data or 'world_x' in point_data):
                px = point_data.get('pixel_x', 'N/A')
                py = point_data.get('pixel_y', 'N/A')
                wx = point_data.get('world_x', 'N/A')
                wy = point_data.get('world_y', 'N/A')
                self.points_listbox.insert(tk.END, f"点{i+1}: 像素({px}, {py}) -> 世界({wx}, {wy})")
            else:
                self.points_listbox.insert(tk.END, f"点{i+1}: 未设置")

    def on_point_listbox_select(self, event):
        """当在列表中选择标定点时"""
        selection = self.points_listbox.curselection()
        if selection:
            point_index = selection[0]
            if point_index < 9:
                self.point_selector.current(point_index)
                self.load_point_data(point_index)

    def clear_zhang_points(self):
        """清除所有张正友标定点"""
        self.zhang_points_data = [{} for _ in range(9)]  # 保持9个空点
        self.points_listbox.delete(0, tk.END)

        # 重新填充空点
        for i in range(9):
            self.points_listbox.insert(tk.END, f"点{i+1}: 未设置")

        # 重置选择器
        self.point_selector.current(0)
        self.load_point_data(0)

        self.result_text.insert(tk.END, "已清除所有标定点\n")
        self.result_text.see(tk.END)

    def perform_extrinsic_calibration(self):
        """执行外参标定"""
        # 过滤出有效标定点
        valid_points = [point for point in self.zhang_points_data if point and
                       'pixel_x' in point and 'world_x' in point]

        if len(valid_points) < 3:
            messagebox.showerror("错误", f"至少需要3个有效标定点，当前有{len(valid_points)}个")
            return

        try:
            # 准备标定数据
            world_points_list = []
            pixel_points_list = []

            for point_data in valid_points:
                # 创建单个点的像素坐标数组
                pixel_point = np.array([[point_data['pixel_x'], point_data['pixel_y']]], dtype=np.float32)
                pixel_points_list.append(pixel_point)

                # 创建单个点的世界坐标数组 (Z=0)
                world_point = np.array([[point_data['world_x'], point_data['world_y'], 0]], dtype=np.float32)
                world_points_list.append(world_point)

            # 执行外参标定
            R, T = self.calibrator.calculate_extrinsics_zhang(world_points_list, pixel_points_list)

            # 保存外参
            self.calibrator.R_camera2base = R
            self.calibrator.T_camera2base = T

            # 更新界面显示
            self.update_camera_params_display()

            self.result_text.insert(tk.END, "外参标定完成!\n")
            self.result_text.insert(tk.END, f"旋转矩阵 R:\n{R}\n")
            self.result_text.insert(tk.END, f"平移向量 T:\n{T}\n")
            self.result_text.see(tk.END)

            # 保存外参到文件
            self.calibrator.save_camera_parameters(os.path.join(dir, 'calibration', 'output', self.filename_var.get()))

            messagebox.showinfo("标定完成", "外参标定已完成!")

        except Exception as e:
            messagebox.showerror("标定失败", f"外参标定失败: {str(e)}")
            self.result_text.insert(tk.END, f"外参标定失败: {str(e)}\n")
            self.result_text.see(tk.END)

    def generate_chessboard_image_gui(self):
        """生成棋盘格图像的GUI方法"""
        try:
            # 获取棋盘格参数
            board_x = int(self.board_x_var.get())
            board_y = int(self.board_y_var.get())

            # 生成图像
            img = self.calibrator.generate_chessboard_image(
                board_size=(board_x, board_y)
            )

            # 显示图像
            self.display_image(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
            self.result_text.insert(tk.END, f"已生成 {board_x}x{board_y} 棋盘格图像\n")
            self.result_text.see(tk.END)

        except Exception as e:
            messagebox.showerror("错误", f"生成棋盘格图像失败: {str(e)}")


    def generate_calibration_points_image_gui(self):
        """生成九点标定图案的GUI方法"""
        try:
            # 获取棋盘格参数
            board_x = int(self.board_x_var.get())
            board_y = int(self.board_y_var.get())

            # 生成九点标定图案
            # 创建一个空白图像(黑色背景)
            img_size = 800
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

            # 计算九个点的位置(3x3网格)
            margin = 100
            grid_size = img_size - 2 * margin
            step = grid_size // 2  # 3个点分成两段

            points = []
            for i in range(3):
                for j in range(3):
                    x = margin + j * step
                    y = margin + i * step
                    points.append((x, y))

            # 在图像上绘制九个点
            for i, (x, y) in enumerate(points):
                # 绘制大圆(绿色)
                cv2.circle(img, (x, y), 15, (0, 255, 0), 3)
                # 绘制小圆(红色实心)
                cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
                # 添加点编号
                cv2.putText(img, f"P{i+1}", (x+20, y-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 显示图像
            self.display_image(img)
            self.result_text.insert(tk.END, "已生成九点标定图像\n")
            self.result_text.see(tk.END)

            # 保存图像到文件
            output_path = 'calibration/output/nine_point_calibration.png'
            directory = os.path.dirname(output_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            cv2.imwrite(output_path, img)
            self.result_text.insert(tk.END, f"九点标定图像已保存到 {output_path}\n")

        except Exception as e:
            messagebox.showerror("错误", f"生成九点标定图像失败: {str(e)}")

    def pixel_to_world_nine_points(self, pixel_x, pixel_y):
        """
        使用九点标定数据将像素坐标转换为世界坐标

        Args:
            pixel_x: 像素x坐标
            pixel_y: 像素y坐标

        Returns:
            tuple: (world_x, world_y) 世界坐标
        """
        # 收集有效的九点标定数据
        valid_points = [point for point in self.zhang_points_data
                        if point and 'pixel_x' in point and 'world_x' in point]

        if len(valid_points) < 4:
            raise ValueError(f"至少需要4个有效标定点进行插值，当前只有{len(valid_points)}个")

        # 提取像素坐标和世界坐标
        pixel_coords = np.array([[point['pixel_x'], point['pixel_y']] for point in valid_points])
        world_coords = np.array([[point['world_x'], point['world_y']] for point in valid_points])

        # 使用scipy的griddata进行插值
        try:
            from scipy.interpolate import griddata
            world_x, world_y = griddata(
                pixel_coords,
                world_coords,
                (pixel_x, pixel_y),
                method='linear'
            )
            return world_x, world_y
        except Exception as e:
            # 如果线性插值失败，尝试使用最近邻插值
            try:
                world_x, world_y = griddata(
                    pixel_coords,
                    world_coords,
                    (pixel_x, pixel_y),
                    method='nearest'
                )
                return world_x, world_y
            except Exception:
                raise RuntimeError(f"无法使用九点标定数据进行坐标转换: {str(e)}")

def pixel_to_world_coordinates(pixel_x, pixel_y,camera_type='RED_CAMERA'):
    """
    将像素坐标转换为世界坐标

    Args:
        pixel_x (float): 像素x坐标
        pixel_y (float): 像素y坐标

    Returns:
        tuple: (world_x, world_y) 世界坐标
    """
    if camera_type=='RED_CAMERA':
        npz_path='camera_params_r.npz'
    elif camera_type=='BLUE_CAMERA':
        npz_path='camera_params_b.npz'

    calibrator = HandEyeCalibration(npz_path)

    # 步骤1: 去除畸变
    # 构造像素点数组
    pixel_points = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)

    # 使用undistortPoints去除畸变
    undistorted_points = cv2.undistortPoints(
        pixel_points,
        calibrator.K,
        calibrator.distortion,
        None,
        calibrator.K  # 使用内参矩阵作为矫正矩阵，保持坐标在同一坐标系下
    )

    # 获取去畸变后的像素坐标
    undistorted_x, undistorted_y = undistorted_points[0][0]

    # 步骤2: 使用外参将去畸变后的像素坐标转换为世界坐标
    # 这里复用HandEyeCalibration类中的pixel_to_world方法
    world_x, world_y = calibrator.pixel_to_world(undistorted_x, undistorted_y)

    return round(world_x,2), round(world_y,2)


def main():
    root = tk.Tk()
    app = HandEyeCalibrationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
    # print(pixel_to_world_coordinates(1082,398,'camera_params_r.npz'))