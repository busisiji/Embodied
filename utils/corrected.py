import numpy as np
import cv2

def _calculate_corrected_points(src_points):
    """
    计算透视矫正后的目标点坐标和变换矩阵

    Args:
        src_points: 原始四点坐标 numpy数组

    Returns:
        tuple: (dst_points, M) 目标点坐标和透视变换矩阵
    """
    # 计算原始四边形的边长
    top_width = np.sqrt(((src_points[1][0] - src_points[0][0]) ** 2) +
                       ((src_points[1][1] - src_points[0][1]) ** 2))
    bottom_width = np.sqrt(((src_points[2][0] - src_points[3][0]) ** 2) +
                          ((src_points[2][1] - src_points[3][1]) ** 2))

    left_height = np.sqrt(((src_points[3][0] - src_points[0][0]) ** 2) +
                         ((src_points[3][1] - src_points[0][1]) ** 2))
    right_height = np.sqrt(((src_points[2][0] - src_points[1][0]) ** 2) +
                          ((src_points[2][1] - src_points[1][1]) ** 2))

    # 取平均宽度和高度
    avg_width = (top_width + bottom_width) / 2
    avg_height = (left_height + right_height) / 2

    # 计算目标矩形的左上角点（选择原始四边形的最左上角点作为参考）
    min_x = np.min(src_points[:, 0])
    min_y = np.min(src_points[:, 1])

    # 定义目标点（保持原始尺寸和比例）
    dst_points = np.float32([
        [min_x, min_y],                           # 左上
        [min_x + avg_width, min_y],               # 右上
        [min_x + avg_width, min_y + avg_height],  # 右下
        [min_x, min_y + avg_height]               # 左下
    ])

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    return dst_points, M

def correct_chessboard_to_square(current_frame, points, M):
    """
    以最小幅度矫正棋盘四点透视，保持尽可能接近原始图像的比例和尺寸
    """
    try:
        # 定义源点（原始四角点）
        src_points = np.float32(points)

        # 计算目标点和变换矩阵
        dst_points, M = _calculate_corrected_points(src_points)

        # 获取原始图像尺寸
        h, w = current_frame.shape[:2]

        # 应用透视变换，保持原始图像尺寸
        corrected_img = cv2.warpPerspective(current_frame, M, (w, h))

        return corrected_img, M
    except Exception as e:
        print(f"透视矫正出错: {e}")
        # 出错时返回原图
        return current_frame.copy(), None

def get_corrected_chessboard_points(original_points):
    """
    输入原始四点坐标，输出经过最小幅度透视矫正后的四点坐标

    Args:
        original_points: 原始四点坐标列表，格式为 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                        顺序应为: 左上, 右上, 右下, 左下

    Returns:
        tuple: (corrected_points, M) 矫正后的四点坐标列表和透视变换矩阵
    """
    # # 将列表转换为numpy数组
    src_points = np.float32(original_points)

    # 计算目标点和变换矩阵
    dst_points, M = _calculate_corrected_points(src_points)

    # 转换为整数坐标并返回
    corrected_points = [(int(point[0]), int(point[1])) for point in dst_points]

    return corrected_points, M


def transform_point_perspective(point, M):
    """
    将单个点从原始坐标系转换到矫正后的坐标系

    Args:
        point: 原始坐标点 (x, y)
        M: 透视变换矩阵

    Returns:
        tuple: 矫正后的坐标点 (x, y)
    """
    # 构造齐次坐标
    point_array = np.array([[[point[0], point[1]]]], dtype=np.float32)

    # 应用透视变换
    transformed_point = cv2.perspectiveTransform(point_array, M)

    # 返回矫正后的坐标
    return (int(transformed_point[0][0][0]), int(transformed_point[0][0][1]))

# 示例用法
if __name__ == "__main__":
    # 使用默认的 CHESS_POINTS_R 进行演示
    original_points = [(137, 151), (1206, 140), (1188, 606), (135, 624)]

    print("原始坐标点:")
    corner_names = ["左上", "右上", "右下", "左下"]
    for i, point in enumerate(original_points):
        print(f"  {corner_names[i]}: ({point[0]}, {point[1]})")

    # 获取矫正后的坐标点
    corrected_points, m = get_corrected_chessboard_points(original_points)

    print("\n矫正后的坐标点:")
    for i, point in enumerate(corrected_points):
        print(f"  {corner_names[i]}: ({point[0]}, {point[1]})")

    # 示例：转换单个点
    test_point = (137, 151)  # 假设这是棋盘上的一个点
    corrected_test_point = transform_point_perspective(test_point, m)
    print(f"\n点 {test_point} 矫正后为 {corrected_test_point}")

