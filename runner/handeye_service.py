# file: handeye_service.py
import os
import cv2
import numpy as np
import json
from scipy.interpolate import griddata

class CalibrationPoint:
    """标定点数据传输对象"""
    def __init__(self, index=0, pixel_x=None, pixel_y=None, world_x=None, world_y=None):
        self.index = index
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        self.world_x = world_x
        self.world_y = world_y

class HandEyeCalibrationService:
    """手眼标定业务逻辑服务类"""

    def __init__(self, calibrator):
        self.calibrator = calibrator
        self.zhang_points_data = [{} for _ in range(9)]

    def update_chessboard_params(self, target_x_number, target_y_number):
        """更新棋盘格参数"""
        self.calibrator.target_x_number = target_x_number
        self.calibrator.target_y_number = target_y_number
        self.calibrator.board_size = (target_x_number, target_y_number)

    def calibrate_intrinsic(self, image_files):
        """执行相机内参标定"""
        # 更新棋盘格参数已在GUI中处理

        # 准备标定数据
        object_points = []  # 世界坐标系中的点
        image_points = []   # 图像坐标系中的点

        # 创建世界坐标系中的棋盘格角点坐标
        total_points = self.calibrator.target_x_number * self.calibrator.target_y_number
        objp = np.zeros((total_points, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.calibrator.target_x_number, 0:self.calibrator.target_y_number].T.reshape(-1, 2)

        successful_images = 0
        gray_shape = None

        # 遍历所有图像进行角点检测
        processed_images = []
        for idx, image_path in enumerate(image_files):
            img = cv2.imread(image_path)
            if img is None:
                processed_images.append((idx+1, "加载失败"))
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if gray_shape is None:
                gray_shape = gray.shape[::-1]

            # 查找棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, self.calibrator.board_size, None)

            if ret:
                # 精确检测角点位置
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                if refined_corners is not None:
                    corners = refined_corners

                # 添加点到标定数据中
                object_points.append(objp.copy())
                image_points.append(corners)
                successful_images += 1
                processed_images.append((idx+1, "检测到角点"))
            else:
                processed_images.append((idx+1, "未检测到角点"))

        if successful_images < 3:
            raise ValueError(f"成功检测角点的图像少于3张 ({successful_images} 张)，无法进行标定")

        if gray_shape is None:
            raise RuntimeError("未能从有效图像中提取尺寸信息")

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, gray_shape, None, None)

        if mtx is None or dist is None:
            raise RuntimeError("OpenCV 标定返回无效结果")

        # 保存标定结果
        self.calibrator.K = mtx
        self.calibrator.distortion = dist

        return {
            'success': True,
            'reprojection_error': ret,
            'successful_images': successful_images,
            'mtx': mtx,
            'dist': dist,
            'processed_images': processed_images
        }

    def detect_calibration_points(self, image, board_size):
        """检测图像中的标定角点"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        if not ret:
            raise ValueError("当前图像未检测到棋盘格角点")

        # 亚像素精化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 确保有9个点（3x3网格）
        if len(corners) < 9:
            raise ValueError("检测到的角点少于9个")

        # 使用3x3网格的9个点
        grid_points = []
        board_w, board_h = board_size

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
            raise ValueError("无法提取9个标定点")

        return grid_points

    def update_calibration_point(self, point_index, pixel_x, pixel_y, world_x, world_y):
        """更新标定点数据"""
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

        return True

    def delete_calibration_point(self, point_index):
        """删除标定点"""
        if 0 <= point_index < len(self.zhang_points_data):
            self.zhang_points_data[point_index] = {}
            return True
        return False

    def clear_all_calibration_points(self):
        """清除所有标定点"""
        self.zhang_points_data = [{} for _ in range(9)]
        return True

    def get_calibration_points(self):
        """获取所有标定点数据"""
        return self.zhang_points_data

    def set_calibration_points(self, points_data):
        """设置标定点数据"""
        self.zhang_points_data = points_data

    def calibrate_extrinsic(self):
        """执行外参标定"""
        # 过滤出有效标定点
        valid_points = [point for point in self.zhang_points_data if point and
                       'pixel_x' in point and 'world_x' in point]

        if len(valid_points) < 3:
            raise ValueError(f"至少需要3个有效标定点，当前有{len(valid_points)}个")

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

        return {
            'rotation_matrix': R,
            'translation_vector': T
        }

    def pixel_to_world_matrix(self, pixel_x, pixel_y):
        """使用矩阵方法将像素坐标转换为世界坐标"""
        return self.calibrator.pixel_to_world(pixel_x, pixel_y)

    def pixel_to_world_nine_points(self, pixel_x, pixel_y):
        """使用九点标定数据将像素坐标转换为世界坐标"""
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
            world_x, world_y = griddata(
                pixel_coords,
                world_coords,
                (pixel_x, pixel_y),
                method='linear'
            )
            return world_x, world_y
        except Exception:
            # 如果线性插值失败，尝试使用最近邻插值
            try:
                world_x, world_y = griddata(
                    pixel_coords,
                    world_coords,
                    (pixel_x, pixel_y),
                    method='nearest'
                )
                return world_x, world_y
            except Exception as e:
                raise RuntimeError(f"无法使用九点标定数据进行坐标转换: {str(e)}")

    def save_points_to_json(self, filepath):
        """保存标定点数据到JSON文件"""
        # 确保目录存在
        directory = os.path.dirname(filepath)
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
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)

        return filepath

    def load_points_from_json(self, filepath):
        """从JSON文件加载标定点数据"""
        if not os.path.exists(filepath):
            return False

        with open(filepath, 'r', encoding='utf-8') as f:
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

        return True
