import os
import json
import random
import shutil
from pathlib import Path

# 定义类别映射
classes = {
    "peach": 0,
    "y-apple": 1,
    "g-apple": 2,
    "g-grape": 3,
    "persimmon": 4,
    "apple": 5,
    "orange": 6,
    "citrus": 7
}

def convert_rotation_box_to_yolo_obb(points, img_width, img_height):
    """
    将旋转框的4个点坐标转换为YOLO OBB格式
    YOLO OBB格式: class_id cx cy w h angle
    """
    # 提取4个点坐标
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]

    # 计算中心点
    cx = (x1 + x2 + x3 + x4) / 4
    cy = (y1 + y2 + y3 + y4) / 4

    # 计算宽度和高度(近似)
    w = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    h = ((x4 - x1)**2 + (y4 - y1)**2)**0.5

    # 计算角度(弧度)
    angle = math.atan2(y2 - y1, x2 - x1)

    # 归一化坐标
    cx /= img_width
    cy /= img_height
    w /= img_width
    h /= img_height

    return cx, cy, w, h, angle

def convert_json_to_yolo_obb(json_file, output_dir, img_width=1280, img_height=720):
    """
    将单个JSON文件转换为YOLO OBB格式 (9列格式)
    YOLO OBB格式: class_index x1 y1 x2 y2 x3 y3 x4 y4
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 获取文件名(不含扩展名)
    base_name = Path(json_file).stem
    output_file = os.path.join(output_dir, f"{base_name}.txt")

    with open(output_file, 'w') as f:
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']

            if label in classes and len(points) >= 4:
                class_id = classes[label]

                # 提取4个点并归一化
                normalized_points = []
                for i in range(4):  # 只取前4个点
                    x_norm = points[i][0] / img_width
                    y_norm = points[i][1] / img_height
                    normalized_points.extend([x_norm, y_norm])

                # 写入YOLO OBB格式: class_index x1 y1 x2 y2 x3 y3 x4 y4 (9列)
                f.write(f"{class_id} " + " ".join(f"{p:.6f}" for p in normalized_points) + "\n")




def create_dataset_split(data_dir, output_dir, train_ratio=0.7):
    """
    创建数据集划分: 训练集和验证集
    """
    # 创建目录结构
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)

    # 获取所有JSON文件
    json_files = list(Path(data_dir).glob("*.json"))

    # 随机打乱文件列表
    random.shuffle(json_files)

    # 计算训练集大小
    train_size = int(len(json_files) * train_ratio)

    # 划分训练集和验证集
    train_files = json_files[:train_size]
    val_files = json_files[train_size:]

    # 处理训练集
    for json_file in train_files:
        # 转换标注文件
        convert_json_to_yolo_obb(json_file, os.path.join(output_dir, "labels", "train"))

        # 复制图像文件(假设图像文件与JSON文件同名)
        img_file = json_file.with_suffix(".jpg")
        if img_file.exists():
            shutil.copy(img_file, os.path.join(output_dir, "images", "train"))

    # 处理验证集
    for json_file in val_files:
        # 转换标注文件
        convert_json_to_yolo_obb(json_file, os.path.join(output_dir, "labels", "val"))

        # 复制图像文件
        img_file = json_file.with_suffix(".jpg")
        if img_file.exists():
            shutil.copy(img_file, os.path.join(output_dir, "images", "val"))

    # 创建数据集配置文件
    create_dataset_config(output_dir, classes)

    print(f"数据集划分完成:")
    print(f"训练集: {len(train_files)} 张图像")
    print(f"验证集: {len(val_files)} 张图像")

def create_dataset_config(output_dir, classes):
    """
    创建YOLO数据集配置文件
    """
    config_content = f"""
path: {output_dir}  # 数据集根目录
train: images/train  # 训练集图像路径
val: images/val  # 验证集图像路径

# 类别数量
nc: {len(classes)}

# 类别名称
names: {list(classes.keys())}
"""

    with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
        f.write(config_content.strip())

# 主程序
if __name__ == "__main__":
    import math

    # 设置输入输出路径
    data_dir = "E:/现有文件/工作/工程/新人工智能实训室/代码/V10.10/Embodied/src/cchessYolo/data/fruit"
    output_dir = "E:/现有文件/工作/工程/新人工智能实训室/代码/V10.10/Embodied/src/cchessYolo/data/fruit"

    # 设置随机种子以确保结果可复现
    random.seed(42)

    # 创建数据集划分
    create_dataset_split(data_dir, output_dir, train_ratio=0.7)
