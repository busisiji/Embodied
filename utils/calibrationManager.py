import cv2
import numpy as np

from parameters import (
    WORLD_POINTS_R, CHESS_POINTS_R,
    WORLD_POINTS_B, CHESS_POINTS_B,
    RCV_CAMERA, RED_CAMERA, SAC_CAMERA, BLACK_CAMERA,
    CHESS_POINTS_RCV_H, BOARD_SAC
)
from utils.calibrationTPS import world_to_pixel_tps, TPS_R_WORLD_TO_PIXEL, pixel_to_world_tps, TPS_R_PIXEL_TO_WORLD, \
    TPS_B_PIXEL_TO_WORLD, TPS_B_WORLD_TO_PIXEL, TPS_RCV_PIXEL_TO_WORLD


def apply_perspective_correction(m, x, y):
    """
    应用透视矫正到坐标点

    Args:
        x: 原始x坐标
        y: 原始y坐标

    Returns:
        矫正后的坐标 (corrected_x, corrected_y)
    """
    # 构造齐次坐标
    point = np.array([[x, y]], dtype=np.float32)
    # 应用透视变换
    corrected_point = cv2.perspectiveTransform(point.reshape(1, -1, 2), m)
    corrected_x, corrected_y = corrected_point[0][0]
    return int(corrected_x), int(corrected_y)
def calculate_perspective_transform_matrices(world_points, chess_points):
    """
    使用标定数据计算透视变换矩阵
    返回世界坐标到像素坐标的变换矩阵和逆变换矩阵

    Args:
        world_points: 世界坐标点列表，长度为4
        chess_points: 棋盘坐标点列表，长度为4

    Returns:
        tuple: (forward_matrix, inverse_matrix) 正向和逆向变换矩阵
    """
    # 将点转换为numpy数组格式
    src_points = np.array(world_points, dtype=np.float32)
    dst_points = np.array(chess_points, dtype=np.float32)

    # 4点标定 - 使用 getPerspectiveTransform
    forward_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    inverse_matrix = cv2.getPerspectiveTransform(dst_points, src_points)

    return forward_matrix, inverse_matrix

# 计算红方和黑方的透视变换矩阵
FORWARD_MATRIX_R, INVERSE_MATRIX_R = calculate_perspective_transform_matrices(WORLD_POINTS_R, CHESS_POINTS_R)
FORWARD_MATRIX_B, INVERSE_MATRIX_B = calculate_perspective_transform_matrices(WORLD_POINTS_B, CHESS_POINTS_B)


def pixel_to_world(u, v, matrix=None, use_tps=True, camera_type="RED_CAMERA"):
    """
    将像素坐标转换为世界坐标

    Args:
        u, v: 像素坐标
        matrix: 逆变换矩阵
        use_tps: 是否使用TPS变换
        camera_type: 相机类型 ("RED" 或 "BLACK")

    Returns:
        tuple: (world_x, world_y) 世界坐标
    """
    if use_tps:
        if camera_type == "RCV_CAMERA":
            return pixel_to_world_tps(u, v, TPS_RCV_PIXEL_TO_WORLD)
        elif camera_type != "BLACK_CAMERA":
            return pixel_to_world_tps(u, v, TPS_R_PIXEL_TO_WORLD)
        else:  # BLACK
            return pixel_to_world_tps(u, v, TPS_B_PIXEL_TO_WORLD)
    else:
        if matrix is None:
            matrix = INVERSE_MATRIX_R

        # 构造齐次坐标
        point = np.array([[[u, v]]], dtype=np.float32)

        # 应用透视变换
        transformed_point = cv2.perspectiveTransform(point, matrix)

        # 返回世界坐标
        return round(float(transformed_point[0][0][0]), 2), round(float(transformed_point[0][0][1]), 2)


def world_to_pixel(x, y, matrix=FORWARD_MATRIX_R, use_tps=True, camera_type="RED"):
    """
    将世界坐标转换为像素坐标

    Args:
        x, y: 世界坐标
        matrix: 正向变换矩阵
        use_tps: 是否使用TPS变换
        camera_type: 相机类型 ("RED" 或 "BLACK")

    Returns:
        tuple: (pixel_u, pixel_v) 像素坐标
    """
    if use_tps:
        if camera_type == "RED":
            return world_to_pixel_tps(x, y, TPS_R_WORLD_TO_PIXEL)
        else:  # BLACK
            return world_to_pixel_tps(x, y, TPS_B_WORLD_TO_PIXEL)
    else:
        # 构造齐次坐标
        point = np.array([[[x, y]]], dtype=np.float32)

        # 应用透视变换
        transformed_point = cv2.perspectiveTransform(point, matrix)

        # 返回像素坐标
        return (int(transformed_point[0][0][0]), int(transformed_point[0][0][1]))


def pixel_to_grid(pixel_x, pixel_y, corner_points):
    """
    将像素坐标映射到4x4网格的位置

    Args:
        pixel_x: 像素x坐标
        pixel_y: 像素y坐标
        corner_points: 区域的四个角点坐标 [(左上), (右上), (右下), (左下)]

    Returns:
        tuple: (grid_x, grid_y) 网格坐标，范围为0-3
    """
    # 检查角点是否定义
    if not corner_points or len(corner_points) != 4:
        print("⚠️ 区域角点未定义或定义不完整，使用默认值")
        # 使用默认值
        corner_points = [(0, 0), (100, 0), (100, 100), (0, 100)]

    # 提取边界坐标
    left = min(corner_points[0][0], corner_points[3][0])
    right = max(corner_points[1][0], corner_points[2][0])
    top = min(corner_points[0][1], corner_points[1][1])
    bottom = max(corner_points[2][1], corner_points[3][1])

    # 计算每个网格单元的宽度和高度
    grid_width = (right - left) / 4
    grid_height = (bottom - top) / 4

    # 将像素坐标转换为网格坐标
    grid_x = int((pixel_x - left) / grid_width)
    grid_y = int((pixel_y - top) / grid_height)

    # 确保坐标在有效范围内
    grid_x = max(0, min(3, grid_x))
    grid_y = max(0, min(3, grid_y))

    return grid_x, grid_y

def pixel_to_sacrifice_grid(pixel_x, pixel_y):
    """
    将像素坐标映射到弃子区4x4网格的位置

    Args:
        pixel_x: 像素x坐标
        pixel_y: 像素y坐标

    Returns:
        tuple: (grid_x, grid_y) 网格坐标，范围为0-3
    """
    grid_x, grid_y = pixel_to_grid(pixel_x, pixel_y, BOARD_SAC)
    return 4 - grid_x, 4 - grid_y

def pixel_to_recovery_grid(pixel_x, pixel_y):
    """
    将像素坐标映射到收子区4x4网格的位置

    Args:
        pixel_x: 像素x坐标
        pixel_y: 像素y坐标

    Returns:
        tuple: (grid_x, grid_y) 网格坐标，范围为0-3
    """
    return pixel_to_grid(pixel_x, pixel_y, CHESS_POINTS_RCV_H)



def chess_to_world_position(col, row, half_board="red"):
    """
    将棋盘坐标转换为世界坐标

    :param col: 棋盘列坐标 (0-8)
    :param row: 棋盘行坐标 (red0-4 black5-9)
    :param half_board: 半区类型 ("red" 或 "black")
    :return: (world_x, world_y) 世界坐标
    """
    # 根据half_board参数选择使用哪一组世界坐标
    if half_board == "red":
        # 红方半盘的世界坐标
        world_coords = WORLD_POINTS_R
    else:  # black
        # 黑方半盘的世界坐标
        world_coords = WORLD_POINTS_B
        row = row - 5

    # 解析四个角点的世界坐标
    top_left_world = np.array(world_coords[0])      # 左上
    top_right_world = np.array(world_coords[1])     # 右上
    bottom_right_world = np.array(world_coords[2])  # 右下
    bottom_left_world = np.array(world_coords[3])   # 左下

    # 确保行列在有效范围内
    row = max(0, min(4, row))
    col = max(0, min(8, col))

    # 镜头翻转
    if half_board == "red":
        col = 8 - col
        row = 4 - row

    # 计算在棋盘中的相对位置比例
    col_ratio = col / 8.0  # 列比例 (0-8 列)
    row_ratio = (1 - row / 4.0)  # 行比例 (0-4 行)

    # 使用双线性插值计算世界坐标
    # 先在上下边上插值
    top_world = top_left_world + col_ratio * (top_right_world - top_left_world)
    bottom_world = bottom_left_world + col_ratio * (bottom_right_world - bottom_left_world)

    # 再在垂直方向上插值
    world_pos = top_world + row_ratio * (bottom_world - top_world)

    return tuple(world_pos)
def camera_to_chess_position( camera_x, camera_y, chess_ponts):
    """
    将相机坐标转换为棋盘坐标

    :param camera_x: 相机x坐标
    :param camera_y: 相机y坐标
    :param chess_ponts: 棋盘四点
    :return: (row, col) 棋盘行列坐标
    """
    try:
        top_left = np.array(chess_ponts[0])
        top_right = np.array(chess_ponts[1])
        bottom_left = np.array(chess_ponts[2])
        bottom_right = np.array(chess_ponts[3])

    except ImportError:
        return None, None

    # 定义棋盘区域边界
    board_x1 = int(min(top_left[0], bottom_left[0]))
    board_y1 = int(min(top_left[1], top_right[1]))
    board_x2 = int(max(top_right[0], bottom_right[0]))
    board_y2 = int(max(bottom_left[1], bottom_right[1]))

    # 计算棋盘格子的尺寸
    cell_width = (board_x2 - board_x1) / 8  # 9列，所以有8个间隔
    cell_height = (board_y2 - board_y1) / 4  # 5行，所以有4个间隔


    # 检查坐标是否在棋盘区域内
    if not (board_x1 - cell_width <= camera_x <= board_x2 + cell_width
            and board_y1 - cell_height <= camera_y <= board_y2 + cell_height/2):
        return None  # 坐标不在棋盘区域内

    # 将图像坐标转换为棋盘坐标
    # 相对于棋盘区域的坐标
    rel_x = camera_x - board_x1
    rel_y = camera_y - board_y1
    # 转换为棋盘格子坐标
    col = int(round(rel_x / cell_width))
    row = int(round(rel_y / cell_height))

    if row < 0 or row > 4 or col <0 or col >8:
        return  None

    return (row, col)

def multi_camera_pixel_to_world(pixel_x, pixel_y, inverse_matrix, camera_type="RED_CAMERA"):
    """
    将任意拍照点的像素坐标转换为世界坐标
    流程：其他拍照点像素 -> RED_CAMERA像素 -> 世界坐标

    Args:
        pixel_x: 像素x坐标
        pixel_y: 像素y坐标
        inverse_matrix: 逆变换矩阵
        camera_type: 拍照点类型 ("RED_CAMERA", "BLACK_CAMERA", "RCV_CAMERA", "SAC_CAMERA")

    Returns:
        tuple: (world_x, world_y) 世界坐标
    """
    # RED_CAMERA像素坐标转换为世界坐标
    world_x, world_y = pixel_to_world(pixel_x, pixel_y, inverse_matrix, camera_type=camera_type)

    # 如果是RED_CAMERA或BLACK_CAMERA，直接返回
    if camera_type in ["RED_CAMERA", "BLACK_CAMERA","RCV_CAMERA"]:
        return world_x, world_y

    # 计算其他相机与RED_CAMERA的坐标偏移
    camera_offsets = {
        "RCV_CAMERA": (RED_CAMERA[0] - RCV_CAMERA[0], RED_CAMERA[1] - RCV_CAMERA[1]),
        "SAC_CAMERA": (RED_CAMERA[0] - SAC_CAMERA[0], RED_CAMERA[1] - SAC_CAMERA[1])
    }

    if camera_type in camera_offsets:
        delta_x, delta_y = camera_offsets[camera_type]
        return round(world_x - delta_x, 2), round(world_y - delta_y, 2)

    return world_x, world_y

def get_area_center(points):
    """
    计算四边形区域的中心点

    Args:
        points: 四个角点的坐标列表 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
               顺序为: 左上, 右上, 右下, 左下

    Returns:
        tuple: 中心点坐标 (center_x, center_y)
    """
    # 计算四个点的平均值
    center_x = sum([point[0] for point in points]) / 4
    center_y = sum([point[1] for point in points]) / 4

    return (center_x, center_y)
