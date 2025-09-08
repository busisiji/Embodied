import cv2
import numpy as np

from parameters import FOUR_WORLD_SAC, CHESS_POINTS_RCV_H, BOARD_SAC, WORLD_POINTS_R, CHESS_POINTS_R, RCV_CAMERA, RED_CAMERA, \
    SAC_CAMERA, BLACK_CAMERA, WORLD_POINTS_B, CHESS_POINTS_B

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
        world_points: 世界坐标点列表，长度为4或9
        chess_points: 棋盘坐标点列表，长度为4或9

    Returns:
        tuple: (forward_matrix, inverse_matrix) 正向和逆向变换矩阵
    """
    # 将点转换为numpy数组格式
    src_points = np.array(world_points, dtype=np.float32)
    dst_points = np.array(chess_points, dtype=np.float32)

    # 根据点的数量选择不同的标定方法
    if len(world_points) == 4 and len(chess_points) == 4:
        # 4点标定 - 使用 getPerspectiveTransform
        forward_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        inverse_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
    elif len(world_points) == 9 and len(chess_points) == 9:
        # 9点标定 - 使用 findHomography
        forward_matrix, _ = cv2.findHomography(src_points, dst_points)
        inverse_matrix, _ = cv2.findHomography(dst_points, src_points)
    else:
        raise ValueError("world_points和chess_points的长度必须都是4或都是9")

    return forward_matrix, inverse_matrix

FORWARD_MATRIX_R, INVERSE_MATRIX_R = calculate_perspective_transform_matrices(WORLD_POINTS_R,CHESS_POINTS_R)
FORWARD_MATRIX_B, INVERSE_MATRIX_B = calculate_perspective_transform_matrices(WORLD_POINTS_B,CHESS_POINTS_B)

def world_to_pixel(x, y, matrix=None):
    """
    将世界坐标转换为像素坐标

    Args:
        x, y: 世界坐标
        matrix: 可选的变换矩阵，如果不提供则使用默认计算的矩阵

    Returns:
        tuple: (pixel_x, pixel_y) 像素坐标
    """
    if matrix is None:
        matrix = FORWARD_MATRIX_R

    # 构造齐次坐标
    point = np.array([[[x, y]]], dtype=np.float32)

    # 应用透视变换
    transformed_point = cv2.perspectiveTransform(point, matrix)

    # 返回像素坐标
    return (int(transformed_point[0][0][0]), int(transformed_point[0][0][1]))

def pixel_to_world(u, v, matrix=None):
    """
    将像素坐标转换为世界坐标

    Args:
        u, v: 像素坐标
        matrix: 可选的逆变换矩阵，如果不提供则使用默认计算的矩阵

    Returns:
        tuple: (world_x, world_y) 世界坐标
    """
    if matrix is None:
        matrix = INVERSE_MATRIX_R

    # 构造齐次坐标
    point = np.array([[[u, v]]], dtype=np.float32)

    # 应用透视变换
    transformed_point = cv2.perspectiveTransform(point, matrix)

    # 返回世界坐标
    return round(float(transformed_point[0][0][0]),2) , round(float(transformed_point[0][0][1]),2)

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

    # 提取左上角和右下角坐标作为边界
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
    return 4-grid_x, 4-grid_y

def pixel_to_recovery_grid(pixel_x, pixel_y):
    """
    将像素坐标映射到收子区4x4网格的位置

    Args:
        pixel_x: 像素x坐标
        pixel_y: 像素y坐标

    Returns:
        tuple: (grid_x, grid_y) 网格坐标，范围为0-3
    """

    grid_x, grid_y = pixel_to_grid(pixel_x, pixel_y, CHESS_POINTS_RCV_H)
    return grid_x, grid_y


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

def chess_to_world_position(col, row, half_board="red"):
    """
    将棋盘坐标转换为世界坐标

    :param col: 棋盘列坐标 (0-8)
    :param row: 棋盘行坐标 (red0-4 black5-9)
    :param half_board: 半区类型 ("red" 或 "black")
    :return: (world_x, world_y) 世界坐标
    """
    try:

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

    except ImportError:
        return None,None

    # 确保行列在有效范围内
    row = max(0, min(4, row))
    col = max(0, min(8, col))
    # 镜头翻转
    if half_board == "red":
        col = 8 - col
        row = 4 - row

    # 计算在棋盘中的相对位置比例
    col_ratio = col / 8.0  # 列比例 (0-8 列)
    row_ratio = (1-row / 4.0)  # 行比例 (0-4 行)

    # 使用双线性插值计算世界坐标
    # 先在上下边上插值
    top_world = top_left_world + col_ratio * (top_right_world - top_left_world)
    bottom_world = bottom_left_world + col_ratio * (bottom_right_world - bottom_left_world)

    # 再在垂直方向上插值
    world_pos = top_world + row_ratio * (bottom_world - top_world)

    return tuple(world_pos)

def chess_to_pixel_position(col, row, half_board="red"):
    """
    将棋盘坐标转换为像素坐标

    :param col: 棋盘列坐标 (0-8)
    :param row: 棋盘行坐标 (0-4)
    :param half_board: 半区类型 ("red" 或 "black")
    :return: (world_x, world_y) 像素坐标
    """
    try:

        # 根据half_board参数选择使用哪一组世界坐标
        if half_board == "red":
            # 红方半盘的世界坐标
            world_coords = CHESS_POINTS_R
        else:  # black
            # 黑方半盘的世界坐标
            world_coords = CHESS_POINTS_B
            row = row - 5

        # 解析四个角点的世界坐标
        top_left_world = np.array(world_coords[0])      # 左上
        top_right_world = np.array(world_coords[1])     # 右上
        bottom_right_world = np.array(world_coords[2])  # 右下
        bottom_left_world = np.array(world_coords[3])   # 左下

    except ImportError:
        return None,None

    # 确保行列在有效范围内
    row = max(0, min(4, row))
    col = max(0, min(8, col))
    # 镜头翻转
    if half_board == "red":
        col = 8 - col
        row = 4 - row

    # 计算在棋盘中的相对位置比例
    col_ratio = col / 8.0  # 列比例 (0-8 列)
    row_ratio = (1-row / 4.0)  # 行比例 (0-4 行)

    # 使用双线性插值计算世界坐标
    # 先在上下边上插值
    top_world = top_left_world + col_ratio * (top_right_world - top_left_world)
    bottom_world = bottom_left_world + col_ratio * (bottom_right_world - bottom_left_world)

    # 再在垂直方向上插值
    world_pos = top_world + row_ratio * (bottom_world - top_world)

    return tuple(world_pos)

def multi_camera_pixel_to_world(pixel_x, pixel_y, inverse_matrix,camera_type="RED_CAMERA"):
    """
    将任意拍照点的像素坐标转换为世界坐标
    流程：其他拍照点像素 -> RED_CAMERA像素 -> 世界坐标

    Args:
        pixel_x: 像素x坐标
        pixel_y: 像素y坐标
        camera_type: 拍照点类型 ("RED_CAMERA", "BLACK_CAMERA", "RCV_CAMERA", "SAC_CAMERA")

    Returns:
        tuple: (world_x, world_y) 世界坐标
    """
    # RED_CAMERA像素坐标转换为世界坐标
    world_x, world_y = pixel_to_world(pixel_x, pixel_y, inverse_matrix)

    # 获取两个拍照点的坐标差异
    if camera_type == "RED_CAMERA":
        return world_x, world_y
    elif camera_type == "RCV_CAMERA":
        # 计算坐标偏移
        delta_x = RED_CAMERA[0] - RCV_CAMERA[0]
        delta_y = RED_CAMERA[1] - RCV_CAMERA[1]
    elif camera_type == "SAC_CAMERA":
        # 计算坐标偏移
        delta_x = RED_CAMERA[0] - SAC_CAMERA[0]
        delta_y = RED_CAMERA[1] - SAC_CAMERA[1]
    elif camera_type == "BLACK_CAMERA":
        return world_x, world_y
    return round(world_x - delta_x, 2), round(world_y - delta_y, 2)
def pixel_to_world_3d(u, v, height, camera_matrix, rotation_matrix, translation_vector):
    """
    使用相机参数将像素坐标和高度转换为世界坐标

    Args:
        u, v: 像素坐标
        height: 物体高度
        camera_matrix: 相机内参矩阵
        rotation_matrix: 旋转矩阵
        translation_vector: 平移向量

    Returns:
        tuple: (world_x, world_y) 世界坐标
    """
    # 构建像素坐标齐次表示
    pixel_point = np.array([u, v, 1.0])

    # 使用相机内参矩阵的逆矩阵
    inv_camera_matrix = np.linalg.inv(camera_matrix)

    # 将像素坐标转换为相机坐标系下的射线方向
    ray_direction = inv_camera_matrix @ pixel_point

    # 计算在指定高度下的交点
    # 假设世界坐标系Z轴向上，地面为Z=0平面
    if abs(ray_direction[2]) > 1e-6:  # 避免除零
        scale = height / ray_direction[2]
        camera_coords = ray_direction * scale

        # 转换到世界坐标系
        world_point = rotation_matrix.T @ (camera_coords - translation_vector)
        return world_point[0], world_point[1]

    return 0, 0  # 错误情况

def get_area_center(points):
    """
    计算四边形区域的中心点

    Args:
        points: 四个角点的坐标列表 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
               顺序为: 左上, 右上, 右下, 左下

    Returns:
        tuple: 中心点坐标 (center_x, center_y)
    """
    # 方法1: 计算四个点的平均值
    center_x = sum([point[0] for point in points]) / 4
    center_y = sum([point[1] for point in points]) / 4

    return (center_x, center_y)

def _convert_rcv_pixel_to_world_with_height(pixel_x, pixel_y,mat, side='red'):
    """
    根据高度信息将收子区像素坐标转换为世界坐标
    上层使用原始矩阵，下层根据高度偏移计算

    Args:
        pixel_x: 像素x坐标
        pixel_y: 像素y坐标
        height: 棋子高度信息

    Returns:
        tuple: (world_x, world_y) 世界坐标
    """
    # 先使用上层的矩阵转换
    world_x, world_y = multi_camera_pixel_to_world(
        pixel_x, pixel_y, mat)

    # 如果高度信息表明是下层棋子，则进行偏移调整
    if side=='black':
        lower_layer_offset_x = 5  # 下层x轴偏移量
        lower_layer_offset_y = 5  # 下层y轴偏移量

        world_x += lower_layer_offset_x
        world_y += lower_layer_offset_y

    return world_x, world_y
