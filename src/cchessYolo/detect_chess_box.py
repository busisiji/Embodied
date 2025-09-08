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
    # 对bottomRight和bottomLeft的x坐标进行调整（减去2）
    bottomRight = (bottomRight[0] - 2, bottomRight[1])
    bottomLeft = (bottomLeft[0] - 2, bottomLeft[1])

    collection_positions = []
    size = PIECE_SIZE
    # 从左上角点开始，按PIECE_SIZE依次排列
    base_x, base_y = topLeft
    base_x = base_x + CHESSBOX_DEVX
    base_y = base_y + CHESSBOX_DEVY

    if type == 'lt':
        for col in range(4):
            for row in range(4):
                # 每个点相对于左上角点向右和向下偏移n个PIECE_SIZE
                center_x = base_x + row * size
                center_y = base_y - col * size
                collection_positions.append((center_x, center_y))
    elif type == 'vec':
        # 每格中心距离为size，以左上角为基准点，考虑xy的倾斜角度
        # 计算x轴和y轴的单位向量（考虑倾斜角度）
        # x方向向量：从左上到右上
        x_vector = np.array([topRight[0] - topLeft[0], topRight[1] - topLeft[1]])
        # y方向向量：从左上到左下
        y_vector = np.array([bottomLeft[0] - topLeft[0], bottomLeft[1] - topLeft[1]])

        # 归一化向量并乘以size作为步长
        x_distance = np.linalg.norm(x_vector)
        y_distance = np.linalg.norm(y_vector)

        if x_distance > 0:
            x_unit_vector = x_vector / x_distance * size
        else:
            x_unit_vector = np.array([size, 0])

        if y_distance > 0:
            y_unit_vector = y_vector / y_distance * size
        else:
            y_unit_vector = np.array([0, size])

        # 从上到下，从左到右遍历16个格子
        for row in range(4):
            for col in range(4):
                # 计算每个格子中心点的位置
                # 考虑到xy轴可能的倾斜角度
                center_x = base_x + col * x_unit_vector[0] + row * y_unit_vector[0]
                center_y = base_y + col * x_unit_vector[1] + row * y_unit_vector[1]
                collection_positions.append((center_x, center_y))
    else:
        # 2. 将四边形划分为4x4网格，计算每个区域的中心点
        # 从上到下，从左到右遍历16个格子
        for row in range(4):
            for col in range(4):
                # 计算在该格子中的相对位置 (0.0 到 1.0)
                # u 代表水平方向 (列)，v 代表垂直方向 (行)
                u = col / 3.0 if 3.0 > 0 else 0  # 列方向比例 0, 1/3, 2/3, 1
                v = row / 3.0 if 3.0 > 0 else 0  # 行方向比例 0, 1/3, 2/3, 1

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
