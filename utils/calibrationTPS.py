# 在文件开头添加scipy导入
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve

from parameters import CHESS_POINTS_R, WORLD_POINTS_R, CHESS_POINTS_B, WORLD_POINTS_B, WORLD_POINTS_RCV, \
    CHESS_POINTS_RCV_L


class TPSTransform():
    """
    Thin Plate Spline 变换类
    """
    def __init__(self, control_points_src, control_points_dst):
        """
        初始化TPS变换

        Args:
            control_points_src: 源控制点坐标列表 [(x1, y1), (x2, y2), ...]
            control_points_dst: 目标控制点坐标列表 [(x1, y1), (x2, y2), ...]
        """
        self.src_points = np.array(control_points_src, dtype=np.float32)
        self.dst_points = np.array(control_points_dst, dtype=np.float32)
        self._compute_tps_parameters()

    def _tps_basis(self, r):
        """
        TPS基函数: phi(r) = r^2 * log(r)
        """
        return np.where(r == 0, 0, r * r * np.log(r + 1e-10))

    def _compute_tps_parameters(self):
        """
        计算TPS变换参数
        """
        n = len(self.src_points)

        # 构建K矩阵 (径向基函数值)
        dist_matrix = cdist(self.src_points, self.src_points)
        K = self._tps_basis(dist_matrix)

        # 构建P矩阵 (仿射部分)
        P = np.ones((n, 3))
        P[:, 1:] = self.src_points

        # 构建L矩阵
        L = np.zeros((n + 3, n + 3))
        L[:n, :n] = K
        L[:n, n:] = P
        L[n:, :n] = P.T

        # 构建目标点矩阵
        target_matrix = np.zeros((n + 3, 2))
        target_matrix[:n, :] = self.dst_points

        # 求解参数
        try:
            params = solve(L, target_matrix)
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用最小二乘法
            params = np.linalg.lstsq(L, target_matrix, rcond=None)[0]

        self.weights = params[:n, :]
        self.affine = params[n:, :]

    def transform(self, points):
        """
        应用TPS变换

        Args:
            points: 待变换点坐标 [(x1, y1), (x2, y2), ...] 或 (x, y)

        Returns:
            变换后的点坐标
        """
        points = np.array(points, dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        # 计算到控制点的距离
        dist_matrix = cdist(points, self.src_points)
        K = self._tps_basis(dist_matrix)

        # 构建P矩阵 (仿射部分)
        P = np.ones((len(points), 3))
        P[:, 1:] = points

        # 应用变换
        result = np.dot(K, self.weights) + np.dot(P, self.affine)
        return result if len(result) > 1 else result[0]

# 创建TPS变换实例
TPS_R_PIXEL_TO_WORLD = TPSTransform(CHESS_POINTS_R, WORLD_POINTS_R)
TPS_R_WORLD_TO_PIXEL = TPSTransform(WORLD_POINTS_R, CHESS_POINTS_R)
TPS_B_PIXEL_TO_WORLD = TPSTransform(CHESS_POINTS_B, WORLD_POINTS_B)
TPS_B_WORLD_TO_PIXEL = TPSTransform(WORLD_POINTS_B, CHESS_POINTS_B)
TPS_RCV_PIXEL_TO_WORLD = TPSTransform(CHESS_POINTS_RCV_L, WORLD_POINTS_RCV)
TPS_RCV_WORLD_TO_PIXEL = TPSTransform(WORLD_POINTS_RCV, CHESS_POINTS_RCV_L)


def pixel_to_world_tps(u, v, tps_transform=TPS_R_PIXEL_TO_WORLD):
    """
    使用TPS变换将像素坐标转换为世界坐标

    Args:
        u, v: 像素坐标
        tps_transform: TPS变换实例

    Returns:
        tuple: (world_x, world_y) 世界坐标
    """
    world_coords = tps_transform.transform([u, v])
    return round(float(world_coords[0]), 2), round(float(world_coords[1]), 2)


def world_to_pixel_tps(x, y, tps_transform=TPS_R_WORLD_TO_PIXEL):
    """
    使用TPS变换将世界坐标转换为像素坐标

    Args:
        x, y: 世界坐标
        tps_transform: TPS变换实例

    Returns:
        tuple: (pixel_u, pixel_v) 像素坐标
    """
    pixel_coords = tps_transform.transform([x, y])
    return int(round(pixel_coords[0])), int(round(pixel_coords[1]))

