# file: handeye_runner.py (GUI部分)
import os
from datetime import datetime

import cv2
import numpy as np
from math import *
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import sys

from dobot.dobot_control import URController
from parameters import FRUIT_CAMERA,RED_CAMERA,IO_QI
import pyrealsense2 as rs

from runner.handeye_service import HandEyeCalibrationService

# 添加机械臂控制模块导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

dir = os.path.dirname(os.path.abspath(__file__))

class HandEyeCalibration:
    def __init__(self,camera_params = 'RED_CAMERA'):
        # 相机内参矩阵 (示例参数，实际使用时需要相机标定获得)
        self.K = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype=np.float64)

        # 畸变参数
        self.distortion = np.array([[0, 0, 0.0, 0.0, 0]])
        self.filedir = os.path.join(dir, 'calibration', 'output')
        self.filepath = os.path.join(self.filedir,camera_params, 'camera_params.npz')

        # 相机外参矩阵 (初始为单位矩阵)
        self.R_camera2base = np.eye(3, dtype=np.float64)  # 旋转矩阵
        self.T_camera2base = np.zeros((3, 1), dtype=np.float64)  # 平移向量

        # 棋盘格参数
        self.target_x_number = 11  # 棋盘格内角点x方向数量
        self.target_y_number = 8  # 棋盘格内角点y方向数量
        self.board_size = (self.target_x_number, self.target_y_number)

        self.load_camera_parameters()

    def save_camera_parameters(self, filepath=None):
        """
        保存相机参数到文件
        """
        # 如果没有提供文件名，使用实例属性
        if filepath is None:
            filepath = self.filepath

        # 确保目录存在
        directory = os.path.dirname(filepath)
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

        np.savez(filepath, **save_data)
        print(f"相机参数已保存到 {filepath}")

    def load_camera_parameters(self,filepath=None):
        """
        从文件加载相机参数

        Args:
            filename: 保存的文件名
        """
        if filepath is None:
            filepath = self.filepath

        if os.path.exists(filepath):
            try:
                data = np.load(filepath, allow_pickle=True)
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

                print(f"相机参数已从 {filepath} 加载")
                return True
            except Exception as e:
                print(f"加载相机参数时出错: {e}")
                return False
        else:
            print(f"未找到相机参数文件 {filepath}")
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

    def pixel_to_world_3d_realsense(self, pixel_x, pixel_y, depth_value, intrinsics, depth_scale):
        """
        使用RealSense SDK进行3D坐标转换

        Args:
            pixel_x: 像素x坐标
            pixel_y: 像素y坐标
            depth_value: 深度值
            intrinsics: RealSense相机内参
            depth_scale: 深度比例因子

        Returns:
            tuple: (world_x, world_y, world_z) 世界坐标
        """
        if self.R_camera2base is None or self.T_camera2base is None:
            raise ValueError("请先执行手眼标定")

        # 使用标定好的内参矩阵self.K替代intrinsics
        # 将像素坐标反投影到相机坐标系
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        # 将像素坐标和深度值转换为相机坐标系下的3D点
        z = depth_value * depth_scale
        x = (pixel_x - cx) * z / fx
        y = (pixel_y - cy) * z / fy
        # 使用RealSense SDK反投影像素到3D点
        # point_3d = rs.rs2_deproject_pixel_to_point(
        #     intrinsics,
        #     [pixel_x, pixel_y],
        #     depth_value * depth_scale
        # )
        # 相机坐标系下的3D点
        camera_coords = np.array([x, y, z])

        # 应用外参变换到世界坐标系
        camera_coords_reshaped = camera_coords.reshape(3, 1)
        world_coords = self.R_camera2base @ camera_coords_reshaped + self.T_camera2base

        return world_coords[0, 0], world_coords[1, 0], world_coords[2, 0]


class HandEyeCalibrationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("手眼标定系统")
        self.root.geometry("1200x800")

        # 配置按钮样式（可选）
        style = ttk.Style()
        style.configure("Red.TButton", foreground="red")

        self.calibrator = HandEyeCalibration()
        self.service = HandEyeCalibrationService(self.calibrator)

        self.current_image = None
        self.calibration_data = []
        self.image_files = []              # 新增
        self.current_image_index = -1       # 新增
        self.robot_speed = tk.DoubleVar(value=50.0)  # 默认速度50%
        self.suction_state = False  # 吸取状态：False=释放状态，True=吸取状态

        # 机械臂控制器
        self.robot_controller = None
        self.robot_connected = False
        # 相机控制相关属性
        self.camera_manager = None
        self.camera_connected = False

        # 像素坐标转世界坐标相关属性
        self.pixel_x_var = tk.StringVar(value="0")
        self.pixel_y_var = tk.StringVar(value="0")
        self.world_x_var = tk.StringVar(value="0.0")
        self.world_y_var = tk.StringVar(value="0.0")

        # 张正友标定点数据 (初始化为9个空点)
        self.selected_point_index = 0

        self.setup_ui()
        # 初始化后更新界面显示
        self.update_camera_params_display()
        # 自动加载保存的标定点数据
        self.reload_camera_parameters()

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

        ttk.Label(filename_frame, text="保存目录:").grid(row=0, column=0, sticky=tk.W)
        self.filename_var = tk.StringVar(value="RED_CAMERA")

        # 修改为Combobox并绑定事件
        self.filename_combo = ttk.Combobox(filename_frame, textvariable=self.filename_var, width=15, state="normal")
        self.filename_combo.grid(row=0, column=1, padx=5, sticky=tk.EW)
        self.filename_combo.bind('<FocusOut>', self.update_filename_options)  # 失去焦点时更新选项
        self.filename_combo.bind('<Return>', self.update_filename_options)  # 回车时更新选项
        self.update_filename_options()  # 初始化选项

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

        # 相机控制区域
        camera_frame = ttk.LabelFrame(control_frame, text="相机控制")
        camera_frame.pack(fill=tk.X, pady=5)

        ttk.Label(camera_frame, text="相机状态:").grid(row=0, column=0, sticky=tk.W)
        self.camera_status_var = tk.StringVar(value="未连接")
        self.camera_status_label = ttk.Label(camera_frame, textvariable=self.camera_status_var)
        self.camera_status_label.grid(row=0, column=1, sticky=tk.W)

        # 创建一个框架来容纳连接和拍照按钮
        camera_button_frame = ttk.Frame(camera_frame)
        camera_button_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=tk.EW)

        self.connect_camera_button = ttk.Button(camera_button_frame, text="连接相机", command=self.toggle_camera_connection)
        self.connect_camera_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.capture_button = ttk.Button(camera_button_frame, text="拍照", command=self.capture_image, state=tk.DISABLED)
        self.capture_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # 配置列权重使按钮能正确拉伸
        camera_button_frame.columnconfigure(0, weight=1)
        camera_button_frame.columnconfigure(1, weight=1)


        # 移动控制
        move_frame = ttk.LabelFrame(control_frame, text="移动控制")
        move_frame.pack(fill=tk.X, pady=5)
        suction_frame = ttk.LabelFrame(control_frame, text="吸放控制")
        suction_frame.pack(fill=tk.X, pady=5)
        home_frame = ttk.LabelFrame(control_frame, text="回家点设置")
        home_frame.pack(fill=tk.X, pady=5)
        ttk.Label(home_frame, text="回家点:").pack(side=tk.LEFT)
        self.home_point_var = tk.StringVar(value="RED_CAMERA")
        self.home_point_combo = ttk.Combobox(home_frame, textvariable=self.home_point_var,
                                             values=["RED_CAMERA", "FRUIT_CAMERA"], state="readonly", width=15)
        self.home_point_combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(move_frame, text="移动到世界坐标", command=self.move_to_world_coordinate).pack(fill=tk.X, pady=2)
        ttk.Button(move_frame, text="回家", command=self.move_home).pack(fill=tk.X, pady=2)
        ttk.Button(move_frame, text="清除报警", command=self.clear_robot_alarm).pack(fill=tk.X, pady=2)

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
        self.pose_frame_container = ttk.LabelFrame(control_frame, text="机械臂位姿")
        self.pose_frame_container.pack(fill=tk.X, pady=5)

        pose_header = ttk.Frame(self.pose_frame_container)
        pose_header.pack(fill=tk.X)

        pose_title = ttk.Label(pose_header, text="机械臂位姿")
        pose_title.pack(side=tk.LEFT)

        self.pose_toggle_btn = ttk.Button(pose_header, text="隐藏", width=5, command=lambda: self.toggle_collapsible_section('pose'))
        self.pose_toggle_btn.pack(side=tk.RIGHT)

        self.pose_content_frame = ttk.Frame(self.pose_frame_container)
        self.pose_content_frame.pack(fill=tk.X, pady=5)

        # 创建笛卡尔坐标和关节坐标两个子框架，每个都可以单独隐藏
        # 笛卡尔坐标框架
        self.cartesian_container = ttk.LabelFrame(self.pose_content_frame, text="笛卡尔坐标 (mm, deg)")
        self.cartesian_container.pack(fill=tk.X, pady=5)

        cartesian_header = ttk.Frame(self.cartesian_container)
        cartesian_header.pack(fill=tk.X)

        cartesian_title = ttk.Label(cartesian_header, text="笛卡尔坐标 (mm, deg)")
        cartesian_title.pack(side=tk.LEFT)

        self.cartesian_toggle_btn = ttk.Button(cartesian_header, text="隐藏", width=5,
                                              command=self.toggle_cartesian_section)
        self.cartesian_toggle_btn.pack(side=tk.RIGHT)

        self.cartesian_content_frame = ttk.Frame(self.cartesian_container)
        self.cartesian_content_frame.pack(fill=tk.X, pady=5)

        # 关节坐标框架
        self.joint_container = ttk.LabelFrame(self.pose_content_frame, text="关节坐标 (deg)")
        self.joint_container.pack(fill=tk.X, pady=5)

        joint_header = ttk.Frame(self.joint_container)
        joint_header.pack(fill=tk.X)

        joint_title = ttk.Label(joint_header, text="关节坐标 (deg)")
        joint_title.pack(side=tk.LEFT)

        self.joint_toggle_btn = ttk.Button(joint_header, text="隐藏", width=5,
                                          command=self.toggle_joint_section)
        self.joint_toggle_btn.pack(side=tk.RIGHT)

        self.joint_content_frame = ttk.Frame(self.joint_container)
        self.joint_content_frame.pack(fill=tk.X, pady=5)

        # 笛卡尔坐标显示 (X, Y, Z, RX, RY, RZ) - 每行一个
        cartesian_labels = ["X:", "Y:", "Z:", "RX:", "RY:", "RZ:"]
        self.cartesian_vars = []
        self.cartesian_labels_widgets = []

        for i, label in enumerate(cartesian_labels):
            frame = ttk.Frame(self.cartesian_content_frame)
            frame.pack(fill=tk.X, pady=2, padx=5)

            ttk.Label(frame, text=label, width=3).pack(side=tk.LEFT)

            var = tk.StringVar(value="0.0")
            label_widget = ttk.Label(frame, textvariable=var, width=12, relief=tk.SUNKEN, anchor=tk.E)
            label_widget.pack(side=tk.LEFT, padx=(0, 5))

            # 添加点动控制按钮
            dec_btn = ttk.Button(frame, text="-", width=3)
            dec_btn.pack(side=tk.LEFT, padx=(0, 2))

            axis_mapping_minus = {0: "X-", 1: "Y-", 2: "Z-", 3: "Rx-", 4: "Ry-", 5: "Rz-"}
            axis_id_minus = axis_mapping_minus.get(i, f"Axis{i}-")

            dec_btn.bind("<ButtonPress-1>", lambda e, axis=axis_id_minus: self.start_jog(axis))
            dec_btn.bind("<ButtonRelease-1>", lambda e: self.stop_jog())

            inc_btn = ttk.Button(frame, text="+", width=3)
            inc_btn.pack(side=tk.LEFT)

            axis_mapping_plus = {0: "X+", 1: "Y+", 2: "Z+", 3: "Rx+", 4: "Ry+", 5: "Rz+"}
            axis_id_plus = axis_mapping_plus.get(i, f"Axis{i}+")

            inc_btn.bind("<ButtonPress-1>", lambda e, axis=axis_id_plus: self.start_jog(axis))
            inc_btn.bind("<ButtonRelease-1>", lambda e: self.stop_jog())

            self.cartesian_vars.append(var)
            self.cartesian_labels_widgets.append(label_widget)

        # 关节坐标显示 (J1, J2, J3, J4, J5, J6) - 每行一个
        joint_labels = ["J1:", "J2:", "J3:", "J4:", "J5:", "J6:"]
        self.joint_vars = []
        self.joint_labels_widgets = []

        for i, label in enumerate(joint_labels):
            frame = ttk.Frame(self.joint_content_frame)
            frame.pack(fill=tk.X, pady=2, padx=5)

            ttk.Label(frame, text=label, width=3).pack(side=tk.LEFT)

            var = tk.StringVar(value="0.0")
            label_widget = ttk.Label(frame, textvariable=var, width=12, relief=tk.SUNKEN, anchor=tk.E)
            label_widget.pack(side=tk.LEFT, padx=(0, 5))

            # 添加点动控制按钮
            dec_btn = ttk.Button(frame, text="-", width=3)
            dec_btn.pack(side=tk.LEFT, padx=(0, 2))

            axis_id_minus = f"J{i+1}-"
            dec_btn.bind("<ButtonPress-1>", lambda e, axis=axis_id_minus: self.start_jog(axis))
            dec_btn.bind("<ButtonRelease-1>", lambda e: self.stop_jog())

            inc_btn = ttk.Button(frame, text="+", width=3)
            inc_btn.pack(side=tk.LEFT)

            axis_id_plus = f"J{i+1}+"
            inc_btn.bind("<ButtonPress-1>", lambda e, axis=axis_id_plus: self.start_jog(axis))
            inc_btn.bind("<ButtonRelease-1>", lambda e: self.stop_jog())

            self.joint_vars.append(var)
            self.joint_labels_widgets.append(label_widget)



        # 像素坐标转换区域
        pixel_frame = ttk.LabelFrame(control_frame, text="像素坐标转世界坐标")
        pixel_frame.pack(fill=tk.X, pady=5)

        # 添加转换方式选择下拉框
        ttk.Label(pixel_frame, text="转换方式:").grid(row=0, column=0, sticky=tk.W)
        self.conversion_method_var = tk.StringVar(value="matrix")
        conversion_method_combo = ttk.Combobox(pixel_frame, textvariable=self.conversion_method_var,
                                              values=["matrix", "nine_points", "3d_with_depth"],
                                              state="readonly", width=12)
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

    def update_filename_options(self, event=None):
        """更新文件名下拉选项"""
        if os.path.exists(self.calibrator.filedir):
            # 获取目录中所有文件夹
            folders = [f for f in os.listdir(self.calibrator.filedir)
                      if os.path.isdir(os.path.join(self.calibrator.filedir, f))]
            # 设置下拉选项
            self.filename_combo['values'] = folders
            # 如果当前值不在选项中且不为空，则添加到选项中
            current_value = self.filename_var.get()
            if current_value and current_value not in folders:
                folders.append(current_value)
                self.filename_combo['values'] = folders

    def reload_camera_parameters(self):
        """重新加载相机参数"""
        filepath = os.path.join(self.calibrator.filedir,self.filename_var.get(),'camera_params.npz')
        if filepath:
            # 更新calibrator的文件名
            self.calibrator.filepath = filepath
            # 重新加载参数
            self.load_zhang_points_from_json()
            if self.calibrator.load_camera_parameters():
                # 更新界面显示
                self.update_camera_params_display()
                self.result_text.insert(tk.END, f"已重新加载参数文件: {filepath}\n")
            else:
                self.result_text.insert(tk.END, f"加载参数文件失败: {filepath}\n")
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

    def toggle_cartesian_section(self):
        """切换笛卡尔坐标区域的显示/隐藏状态"""
        if self.cartesian_content_frame.winfo_viewable():
            self.cartesian_content_frame.pack_forget()
            self.cartesian_toggle_btn.config(text="显示")
        else:
            self.cartesian_content_frame.pack(fill=tk.X, pady=5)
            self.cartesian_toggle_btn.config(text="隐藏")

        # 更新滚动区域
        self.control_canvas.update_idletasks()
        self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))

    def toggle_joint_section(self):
        """切换关节坐标区域的显示/隐藏状态"""
        if self.joint_content_frame.winfo_viewable():
            self.joint_content_frame.pack_forget()
            self.joint_toggle_btn.config(text="显示")
        else:
            self.joint_content_frame.pack(fill=tk.X, pady=5)
            self.joint_toggle_btn.config(text="隐藏")

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

    def clear_robot_alarm(self):
        """清除机械臂报警"""
        if not self.robot_connected:
            messagebox.showerror("错误", "请先连接机械臂")
            return

        try:
            if self.robot_controller.clear_alarm():
                self.result_text.insert(tk.END, "✅ 机械臂报警已清除\n")
                messagebox.showinfo("提示", "机械臂报警已清除")
            else:
                self.result_text.insert(tk.END, "❌ 机械臂报警清除失败\n")
                messagebox.showerror("错误", "机械臂报警清除失败")
        except Exception as e:
            self.result_text.insert(tk.END, f"❌ 清除报警时出错: {str(e)}\n")
            messagebox.showerror("错误", f"清除报警时出错: {str(e)}")
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
                # 获取机械臂当前位置（笛卡尔坐标）
                current_pos = self.robot_controller.get_current_position()
                if current_pos is not None:
                    # 更新界面显示的笛卡尔坐标
                    for i in range(min(6, len(current_pos))):
                        if i < len(self.cartesian_vars):
                            self.cartesian_vars[i].set(str(round(current_pos[i], 3)))

                # 获取机械臂当前关节位置
                current_joint_pos = self.robot_controller.get_current_joint_position()
                if current_joint_pos is not None:
                    # 更新界面显示的关节坐标
                    for i in range(min(6, len(current_joint_pos))):
                        if i < len(self.joint_vars):
                            self.joint_vars[i].set(str(round(current_joint_pos[i], 3)))
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

    def move_to_current_pose(self):
        """移动到当前输入的位姿"""
        if not self.robot_connected:
            messagebox.showerror("错误", "请先连接机械臂")
            return

        try:
            # 设置速度
            speed = int(self.robot_speed.get())
            self.robot_controller.set_speed(speed / 100.0)

            # 获取当前显示的笛卡尔坐标值
            pose = [float(var.get()) for var in self.cartesian_vars]
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


    def move_home(self):
        """机械臂回家"""
        if not self.robot_connected:
            messagebox.showerror("错误", "请先连接机械臂")
            return

        try:
            # 设置速度
            speed = int(self.robot_speed.get())
            self.robot_controller.set_speed(speed / 100.0)

            # 根据选择的回家点执行相应操作
            home_point = self.home_point_var.get()
            if home_point == "RED_CAMERA":
                self.robot_controller.run_point_j(RED_CAMERA)
            elif home_point == "FRUIT_CAMERA":
                self.robot_controller.run_point_j(FRUIT_CAMERA)

            self.result_text.insert(tk.END, f"机械臂已回家到 {home_point} (速度: {speed}%)\n")
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

        try:
            # 更新棋盘格参数
            target_x_number = int(self.board_x_var.get())
            target_y_number = int(self.board_y_var.get())
            self.service.update_chessboard_params(target_x_number, target_y_number)

            # 执行标定
            result = self.service.calibrate_intrinsic(self.image_files)

            # 更新UI中的相机参数显示
            self.fx_var.set(f"{result['mtx'][0, 0]:.3f}")
            self.fy_var.set(f"{result['mtx'][1, 1]:.3f}")
            self.cx_var.set(f"{result['mtx'][0, 2]:.3f}")
            self.cy_var.set(f"{result['mtx'][1, 2]:.3f}")

            # 显示标定结果
            self.result_text.insert(tk.END, "\n相机内参标定完成!\n")
            self.result_text.insert(tk.END, f"使用了 {result['successful_images']} 张图像进行标定\n")
            self.result_text.insert(tk.END, f"重投影误差: {result['reprojection_error']:.3f} pixels\n\n")
            self.result_text.insert(tk.END, f"相机内参矩阵 K:\n{result['mtx']}\n\n")
            self.result_text.insert(tk.END, f"畸变系数:\n{result['dist']}\n")

            # 显示处理详情
            for idx, status in result['processed_images']:
                self.result_text.insert(tk.END, f"图像 {idx}: {status}\n")

            self.result_text.see(tk.END)

            filepath = os.path.join(self.calibrator.filedir,self.filename_var.get(),'camera_params.npz')
            # 确保目录存在
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            self.calibrator.save_camera_parameters(filepath)

            messagebox.showinfo("标定完成", f"相机内参标定完成!\n重投影误差: {result['reprojection_error']:.3f} pixels")

        except ValueError as e:
            messagebox.showerror("标定失败", f"标定过程中出现错误: {str(e)}")
            self.result_text.insert(tk.END, f"标定失败: {str(e)}\n")
            self.result_text.see(tk.END)
        except Exception as e:
            messagebox.showerror("未知错误", f"发生未预期错误: {str(e)}")
            self.result_text.insert(tk.END, f"未知错误: {str(e)}\n")
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
                world_x, world_y = self.service.pixel_to_world_matrix(pixel_x, pixel_y)
                self.world_x_var.set(f"{world_x:.3f}")
                self.world_y_var.set(f"{world_y:.3f}")
                self.result_text.insert(tk.END, f"像素坐标({pixel_x}, {pixel_y}) -> 世界坐标({world_x:.3f}, {world_y:.3f}) [{conversion_method}]\n")

            elif conversion_method == "nine_points":
                # 使用九点标定转换方法
                world_x, world_y = self.service.pixel_to_world_nine_points(pixel_x, pixel_y)
                self.world_x_var.set(f"{world_x:.3f}")
                self.world_y_var.set(f"{world_y:.3f}")
                self.result_text.insert(tk.END, f"像素坐标({pixel_x}, {pixel_y}) -> 世界坐标({world_x:.3f}, {world_y:.3f}) [{conversion_method}]\n")

            elif conversion_method == "3d_with_depth":
                # 使用3D深度转换方法
                self.convert_pixel_to_world_3d()

            else:
                raise ValueError(f"未知的转换方式: {conversion_method}")

            self.result_text.see(tk.END)

        except ValueError as e:
            print(e)
            messagebox.showerror("错误", f"坐标转换失败: {str(e)}")
        except Exception as e:
            messagebox.showerror("错误", f"发生错误: {str(e)}")
    def convert_pixel_to_world_3d(self):
        """将像素坐标和深度转换为3D世界坐标"""
        try:
            pixel_x = float(self.pixel_x_var.get())
            pixel_y = float(self.pixel_y_var.get())

            if not self.camera_connected or not self.camera_manager:
                messagebox.showerror("错误", "请先连接相机以获取深度信息")
                return

            # 从相机管理器获取帧和内参
            color_frame, depth_frame = self.camera_manager.get_frame()
            if depth_frame is not None:
                # 获取深度值
                depth_data = np.asanyarray(depth_frame.get_data())
                depth_value = depth_data[int(pixel_y), int(pixel_x)]

                # 获取相机内参和深度比例
                depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                # 使用 CameraManager 中保存的深度比例
                depth_scale = self.camera_manager.depth_scale

                # 使用RealSense SDK进行坐标转换
                world_x, world_y, world_z = self.calibrator.pixel_to_world_3d_realsense(
                    pixel_x, pixel_y, depth_value, depth_intrinsics, depth_scale
                )

                # 更新显示
                self.world_x_var.set(f"{world_x:.3f}")
                self.world_y_var.set(f"{world_y:.3f}")

                self.result_text.insert(tk.END,
                                        f"3D坐标转换(RealSense): 像素({pixel_x}, {pixel_y}) 深度{depth_value*depth_scale} -> "
                                        f"世界坐标({world_x:.3f}, {world_y:.3f}, {world_z:.3f})\n")
                self.result_text.see(tk.END)
            else:
                messagebox.showerror("错误", "无法获取深度信息")

        except Exception as e:
            messagebox.showerror("错误", f"3D坐标转换失败: {str(e)}")
    def move_to_world_coordinate(self):
        """移动到指定的世界坐标"""
        if not self.robot_connected:
            messagebox.showerror("错误", "请先连接机械臂")
            return

        try:
            # 获取当前Z坐标
            current_z = float(self.cartesian_vars[2].get())  # Z坐标

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
                self.cartesian_vars[0].set(f"{world_x:.3f}")
                self.cartesian_vars[1].set(f"{world_y:.3f}")
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
        # 清空输入框
        self.pixel_x_entry.delete(0, tk.END)
        self.pixel_y_entry.delete(0, tk.END)
        self.world_x_entry.delete(0, tk.END)
        self.world_y_entry.delete(0, tk.END)

        zhang_points_data = self.service.get_calibration_points()
        if 0 <= point_index < len(zhang_points_data):
            point_data = zhang_points_data[point_index]

            # 如果有点数据，加载到输入框
            if 'pixel_x' in point_data and 'pixel_y' in point_data:
                self.pixel_x_entry.insert(0, str(point_data['pixel_x']))
                self.pixel_y_entry.insert(0, str(point_data['pixel_y']))

            if 'world_x' in point_data and 'world_y' in point_data:
                self.world_x_entry.insert(0, str(point_data['world_x']))
                self.world_y_entry.insert(0, str(point_data['world_y']))


    def detect_chessboard_corners(self):
        """检测当前图像的棋盘格角点（修改为检测九点标定中的九个点）"""
        if not self.current_image_index >= 0 or self.current_image is None:
            messagebox.showerror("错误", "请先加载图像")
            return

        try:
            # 获取棋盘格尺寸
            board_x = int(self.board_x_var.get())
            board_y = int(self.board_y_var.get())
            board_size = (board_x, board_y)

            # 检测角点
            grid_points = self.service.detect_calibration_points(self.current_image, board_size)

            # 更新所有9个标定点的像素坐标
            for i, point in enumerate(grid_points):
                self.service.update_calibration_point(
                    i,
                    round(point[0], 2),
                    round(point[1], 2),
                    None,
                    None
                )

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

    def save_zhang_points_to_json(self):
        """
        保存张正友九点标定数据到JSON文件
        """
        try:
            filepath = os.path.join(self.calibrator.filedir,self.filename_var.get(),'saved_points.json')
            saved_path = self.service.save_points_to_json(filepath)
            self.result_text.insert(tk.END, f"标定点数据已保存到 {saved_path}\n")
            self.result_text.see(tk.END)

        except Exception as e:
            messagebox.showerror("保存失败", f"保存标定点数据时出错: {str(e)}")

    def load_zhang_points_from_json(self):
        """
        从JSON文件加载张正友九点标定数据
        """
        try:
            filepath = os.path.join(self.calibrator.filedir,self.filename_var.get(),'saved_points.json')

            if self.service.load_points_from_json(filepath):
                # 更新界面显示
                self.update_points_listbox()
                # 加载当前选中的点数据到输入框
                self.load_point_data(self.point_selector.current())

                self.result_text.insert(tk.END, f"标定点数据已从 {filepath} 加载\n")
                self.result_text.see(tk.END)
                return True
            else:
                self.result_text.insert(tk.END, f"未找到标定点数据文件 {filepath}\n")
                self.result_text.see(tk.END)
                return False

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

            # 更新标定点
            self.service.update_calibration_point(point_index, pixel_x, pixel_y, world_x, world_y)

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
        # 清空输入框
        self.pixel_x_entry.delete(0, tk.END)
        self.pixel_y_entry.delete(0, tk.END)
        self.world_x_entry.delete(0, tk.END)
        self.world_y_entry.delete(0, tk.END)

        if self.service.delete_calibration_point(point_index):
            # 更新显示
            self.load_point_data(point_index)
            self.update_points_listbox()

            self.result_text.insert(tk.END, f"已删除标定点 {point_index+1}\n")
            self.result_text.see(tk.END)

    def update_points_listbox(self):
        """更新标定点列表显示"""
        self.points_listbox.delete(0, tk.END)
        zhang_points_data = self.service.get_calibration_points()

        for i, point_data in enumerate(zhang_points_data):
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
        self.service.clear_all_calibration_points()
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
        try:
            # 执行外参标定
            result = self.service.calibrate_extrinsic()

            # 更新界面显示
            self.update_camera_params_display()

            self.result_text.insert(tk.END, "外参标定完成!\n")
            self.result_text.insert(tk.END, f"旋转矩阵 R:\n{result['rotation_matrix']}\n")
            self.result_text.insert(tk.END, f"平移向量 T:\n{result['translation_vector']}\n")
            self.result_text.see(tk.END)

            # 保存外参到文件
            filepath = os.path.join(self.calibrator.filedir,self.filename_var.get(),'camera_params.npz')
            # 确保目录存在
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            self.calibrator.save_camera_parameters(filepath)

            messagebox.showinfo("标定完成", "外参标定已完成!")

        except ValueError as e:
            messagebox.showerror("错误", f"外参标定失败: {str(e)}")
            self.result_text.insert(tk.END, f"外参标定失败: {str(e)}\n")
            self.result_text.see(tk.END)
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
            filepath = os.path.join(self.calibrator.filedir,self.filename_var.get(),'nine_point_calibration.png')
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            cv2.imwrite(filepath, img)
            self.result_text.insert(tk.END, f"九点标定图像已保存到 {filepath}\n")

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
        return self.service.pixel_to_world_nine_points(pixel_x, pixel_y)


    def toggle_camera_connection(self):
        """切换相机连接状态"""
        if self.camera_connected:
            self.disconnect_camera()
        else:
            self.connect_camera()

    def connect_camera(self):
        """连接RealSense相机"""
        try:
            from manager.camera_manager import CameraManager
            self.camera_manager = CameraManager()

            if self.camera_manager.initialize_camera():
                self.camera_connected = True
                self.connect_camera_button.config(text="断开相机", style="Red.TButton")  # 添加红色样式
                self.camera_status_var.set("已连接")
                self.capture_button.config(state=tk.NORMAL)
                self.result_text.insert(tk.END, "相机连接成功\n")
            else:
                self.result_text.insert(tk.END, "相机连接失败\n")

        except Exception as e:
            self.result_text.insert(tk.END, f"连接相机时出错: {str(e)}\n")
        self.result_text.see(tk.END)

    def disconnect_camera(self):
        """断开相机连接"""
        try:
            if self.camera_manager:
                self.camera_manager.release_camera()
                self.camera_manager = None

            self.camera_connected = False
            self.connect_camera_button.config(text="连接相机", style="")  # 恢复默认样式
            self.camera_status_var.set("未连接")
            self.capture_button.config(state=tk.DISABLED)
            self.result_text.insert(tk.END, "相机已断开连接\n")

        except Exception as e:
            self.result_text.insert(tk.END, f"断开相机连接时出错: {str(e)}\n")
        self.result_text.see(tk.END)

    def capture_image(self):
        """拍照并显示"""
        if not self.camera_connected or not self.camera_manager:
            messagebox.showerror("错误", "请先连接相机")
            return

        try:
            # 捕获图像
            image, depth_frame = self.camera_manager.get_frame()

            if image is not None:
                # 如果需要处理深度数据，也需要转换
                if depth_frame is not None:
                    depth_data = np.asanyarray(depth_frame.get_data())
                    # 现在可以安全地使用 depth_data[y, x] 访问深度值

                # 保存图像到指定路径
                images_dir = os.path.join(dir, "calibration/images")
                if not os.path.exists(images_dir):
                    os.makedirs(images_dir)

                # 生成唯一文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                filepath = os.path.join(images_dir, filename)

                # 保存图像
                cv2.imwrite(filepath, image)

                # 加载并显示图像
                self.current_image = image
                self.image_files.append(filepath)
                self.current_image_index = len(self.image_files) - 1

                self.display_image(image)
                self.image_count_label.config(text=f"{len(self.image_files)}/{len(self.image_files)}")
                self.result_text.insert(tk.END, f"拍照成功，图像已保存到: {filepath}\n")
            else:
                self.result_text.insert(tk.END, "拍照失败，未能获取图像\n")

        except Exception as e:
            self.result_text.insert(tk.END, f"拍照时出错: {str(e)}\n")
        self.result_text.see(tk.END)


    def __del__(self):
        """析构函数，确保释放资源"""
        if self.camera_connected:
            self.disconnect_camera()


def calculate_height_compensation(top_pixel, top_depth, bottom_depth=0.415, center_pixel=(640, 360)):
    """
    根据深度差异与透视关系，将顶部像素坐标转换为对应底部像素坐标

    参数:
        top_pixel (tuple): 顶部像素坐标 (x, y)
        top_depth (float): 顶部深度值
        bottom_depth (float): 目标底部深度值
        center_pixel (tuple): 图像中心像素坐标，默认(600, 350)

    返回:
        tuple: 对应的底部像素坐标 (x, y)
    """
    x_top, y_top = top_pixel
    cx, cy = center_pixel

    # 深度比例因子
    depth_ratio = top_depth / bottom_depth

    # 基于相似三角形的缩放
    dx = (x_top - cx) * depth_ratio + cx
    dy = (y_top - cy) * depth_ratio + cy

    return (dx, dy)
def pixel_to_world_coordinates(pixel_x, pixel_y,camera_type='RED_CAMERA'):
    """
    将像素坐标转换为世界坐标

    Args:
        pixel_x (float): 像素x坐标
        pixel_y (float): 像素y坐标

    Returns:
        tuple: (world_x, world_y) 世界坐标
    """
    filedir = os.path.join(dir, 'calibration', 'output')
    filepath = os.path.join(filedir, camera_type, 'camera_params.npz')

    calibrator = HandEyeCalibration(camera_type)

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
    world_x, world_y = calibrator.pixel_to_world(undistorted_x, undistorted_y)

    return round(world_x,2), round(world_y,2)


def main():
    root = tk.Tk()
    app = HandEyeCalibrationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
    # print(pixel_to_world_coordinates(1082,398,'camera_params_r.npz'))
