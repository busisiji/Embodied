import cv2
import numpy as np

from parameters import PIECE_SIZE, CHESSBOX_DEVX, CHESSBOX_DEVY


def select_corner_circles( circles):
    """
    从检测到的圆形中选择最可能的4个角点圆形用于收局
    """
    if len(circles) == 4:
        return circles

    # 寻找最接近矩形的4个圆
    best_combination = []
    best_score = float('inf')

    from itertools import combinations
    for combination in combinations(circles, 4):
        points = np.array([[circle[0], circle[1]] for circle in combination])
        score = rectangle_similarity_score(points)
        if score < best_score:
            best_score = score
            best_combination = combination

    if best_combination:
        return np.array(best_combination)

    # 如果没有找到接近矩形的组合，则选择构成最大面积凸四边形的4个圆
    max_area = 0
    largest_combination = []

    for combination in combinations(circles, 4):
        points = np.array([[circle[0], circle[1]] for circle in combination])
        if is_convex_quadrilateral(points):
            area = polygon_area(points)
            if area > max_area:
                max_area = area
                largest_combination = combination

    if largest_combination:
        return np.array(largest_combination)

    # 如果仍然没有找到，则选择距离图像中心最远的4个
    return np.array(circles[:4]) if len(circles) >= 4 else circles


def order_points( pts):
    """
    按照左上、右上、右下、左下的顺序排列四个点
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下

    return rect


def calculate_box_corners( ordered_centers, radius):
    """
    根据圆形中心点和半径计算棋盒的实际角点
    """
    corners = []

    # 左上角点
    lt_center = ordered_centers[0]
    dx = radius
    dy = radius
    corners.append((lt_center[0] - dx, lt_center[1] - dy))

    # 右上角点
    rt_center = ordered_centers[1]
    corners.append((rt_center[0] + dx, rt_center[1] - dy))

    # 右下角点
    rb_center = ordered_centers[2]
    corners.append((rb_center[0] + dx, rb_center[1] + dy))

    # 左下角点
    lb_center = ordered_centers[3]
    corners.append((lb_center[0] - dx, lb_center[1] + dy))

    return np.array(corners)


def is_convex_quadrilateral( points):
    """
    检查四个点是否能构成凸四边形
    """
    if len(points) != 4:
        return False

    ordered = order_points(points)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    signs = []
    for i in range(4):
        o = ordered[i]
        a = ordered[(i + 1) % 4]
        b = ordered[(i + 2) % 4]
        signs.append(cross(o, a, b))

    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


def polygon_area( points):
    """
    计算多边形面积（Shoelace formula）
    """
    n = len(points)
    if n < 3:
        return 0

    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]

    return abs(area) / 2


def rectangle_similarity_score( points):
    """
    计算四边形接近矩形的分数（分数越低越接近矩形）
    """
    if len(points) != 4:
        return float('inf')

    ordered = order_points(points)

    # 计算四条边的长度
    edges = []
    for i in range(4):
        p1 = ordered[i]
        p2 = ordered[(i + 1) % 4]
        length = np.linalg.norm(p2 - p1)
        edges.append(length)

    # 计算边长差异分数
    avg_edge = np.mean(edges)
    edge_variance = np.std(edges) / avg_edge if avg_edge > 0 else float('inf')

    # 计算对角线差异分数
    diag1 = np.linalg.norm(ordered[0] - ordered[2])
    diag2 = np.linalg.norm(ordered[1] - ordered[3])
    diag_diff = abs(diag1 - diag2) / max(diag1, diag2) if max(diag1, diag2) > 0 else float('inf')

    # 计算角度差异分数
    def angle_between_vectors(v1, v2):
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        return np.arccos(np.clip(cos_angle, -1.0, 1.0))

    angle_scores = []
    for i in range(4):
        v1 = ordered[(i + 1) % 4] - ordered[i]
        v2 = ordered[(i + 2) % 4] - ordered[(i + 1) % 4]
        angle = angle_between_vectors(v1, v2)
        angle_deg = np.degrees(angle)
        # 90度的偏差分数
        angle_scores.append(abs(angle_deg - 90) / 90)

    avg_angle_score = np.mean(angle_scores)

    # 综合分数
    total_score = edge_variance * 0.4 + diag_diff * 0.3 + avg_angle_score * 0.3

    return total_score

def calculate_4x4_collection_positions(corner_points, type='vec'):
    """
    先将四角点转为世界坐标，再将该四边形划分为16个区域，得到每个区域的中心点collection_positions(从左到右，从上到下)

    Args:
        corner_points: 棋盒的四个角点坐标 [(x,y), (x,y), (x,y), (x,y)]
                      顺序为: 左上, 右上, 右下, 左下
        type: 计算方式，'grid'表示网格划分，'lt'表示从左上角点开始依次排列

    Returns:
        list: 4x4网格的中心点世界坐标列表 (从左到右，从上到下)
    """
    if len(corner_points) != 4:
        print("⚠️ 角点数量不正确，应为4个")
        return []

    # 提取四个角点的世界坐标
    topLeft, topRight, bottomRight, bottomLeft = corner_points

    # 检查四边形尺寸是否至少为 PIECE_SIZE * 4
    # 计算宽度（上下两边）
    top_width = np.linalg.norm(np.array(topRight) - np.array(topLeft))
    bottom_width = np.linalg.norm(np.array(bottomRight) - np.array(bottomLeft))

    # 计算高度（左右两边）
    left_height = np.linalg.norm(np.array(bottomLeft) - np.array(topLeft))
    right_height = np.linalg.norm(np.array(bottomRight) - np.array(topRight))

    min_width = min(top_width, bottom_width)
    min_height = min(left_height, right_height)

    # 最小尺寸要求
    min_required_size = PIECE_SIZE * 4

    # 如果不满足条件，则调整角点位置
    if min_width < min_required_size or min_height < min_required_size:
        print(f"⚠️ 四边形尺寸过小: 宽度={min_width:.2f}, 高度={min_height:.2f}, 最小要求={min_required_size}")
        print("正在调整四边形尺寸...")

        # 计算中心点
        center_x = (topLeft[0] + topRight[0] + bottomRight[0] + bottomLeft[0]) / 4
        center_y = (topLeft[1] + topRight[1] + bottomRight[1] + bottomLeft[1]) / 4
        center = np.array([center_x, center_y])

        # 计算原始向量
        top_vector = np.array(topRight) - np.array(topLeft)
        left_vector = np.array(bottomLeft) - np.array(topLeft)

        # 调整向量长度以满足最小尺寸要求
        if min_width < min_required_size:
            # 调整水平向量
            top_norm = np.linalg.norm(top_vector)
            if top_norm > 0:
                top_vector = top_vector / top_norm * min_required_size

        if min_height < min_required_size:
            # 调整垂直向量
            left_norm = np.linalg.norm(left_vector)
            if left_norm > 0:
                left_vector = left_vector / left_norm * min_required_size

        # 重新计算角点位置
        topLeft = center - top_vector/2 - left_vector/2
        topRight = center + top_vector/2 - left_vector/2
        bottomRight = center + top_vector/2 + left_vector/2
        bottomLeft = center - top_vector/2 + left_vector/2

        # 转换为元组格式
        topLeft = tuple(topLeft)
        topRight = tuple(topRight)
        bottomRight = tuple(bottomRight)
        bottomLeft = tuple(bottomLeft)

        print("四边形尺寸已调整以满足最小要求")

    collection_positions = []

    if type == 'vec':
        # 使用向量方法计算，考虑棋盘可能的倾斜角度
        # 计算x轴和y轴的单位向量（考虑倾斜角度）
        # x方向向量：从左上到右上
        x_vector = np.array([topRight[0] - topLeft[0], topRight[1] - topLeft[1]])
        # y方向向量：从左上到左下
        y_vector = np.array([bottomLeft[0] - topLeft[0], bottomLeft[1] - topLeft[1]])

        # 计算每一步的向量（整个棋盘分为4份）
        x_step = x_vector / 4.0
        y_step = y_vector / 4.0

        # 从上到下，从左到右遍历16个格子中心点
        # 每个格子中心点位于格子的中心，需要加上半步长
        for row in range(4):
            for col in range(4):
                # 计算每个格子中心点的位置
                # 起始点是topLeft，加上相应步数和半个步长以定位到格子中心
                center_x = round(topLeft[0] + (col + 0.5) * x_step[0] + (row + 0.5) * y_step[0], 2)
                center_y = round(topLeft[1] + (col + 0.5) * x_step[1] + (row + 0.5) * y_step[1], 2)
                collection_positions.append((center_x, center_y))
    else:
        # 使用双线性插值方法计算
        # 从上到下，从左到右遍历16个格子中心点
        for row in range(4):
            for col in range(4):
                # 计算在该格子中的相对位置 (0.0 到 1.0)
                # u 代表水平方向 (列)，v 代表垂直方向 (行)
                # 中心点位置为 (i+0.5)/4 形式
                u = (col + 0.5) / 4.0  # 列方向比例 0.125, 0.375, 0.625, 0.875
                v = (row + 0.5) / 4.0  # 行方向比例 0.125, 0.375, 0.625, 0.875

                # 使用双线性插值计算格子中心点的世界坐标
                # 先计算上下两条边上的点
                top_x = topLeft[0] + u * (topRight[0] - topLeft[0])
                top_y = topLeft[1] + u * (topRight[1] - topLeft[1])

                bottom_x = bottomLeft[0] + u * (bottomRight[0] - bottomLeft[0])
                bottom_y = bottomLeft[1] + u * (bottomRight[1] - bottomLeft[1])

                # 再在上下边之间进行插值
                center_x = round(top_x + v * (bottom_x - top_x), 2)
                center_y = round(top_y + v * (bottom_y - top_y), 2)

                collection_positions.append((center_x, center_y))
    return collection_positions
