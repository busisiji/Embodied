# split_dataset.py
import os
import shutil
import random
from pathlib import Path
import json

def create_dataset_structure():
    """创建数据集目录结构（按期望格式）"""
    dirs = [
        'dataset/images/train',
        'dataset/images/val',
        'dataset/labels/train',
        'dataset/labels/val'
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 创建目录: {dir_path}")

def get_image_from_json(json_path):
    """根据JSON文件名获取对应的图像文件名"""
    # 假设图像文件与JSON文件同名，但扩展名不同
    base_name = Path(json_path).stem
    # 这里可以根据实际图像格式修改扩展名
    possible_extensions = ['.jpg', '.jpeg', '.png']

    for ext in possible_extensions:
        image_path = Path(json_path).parent / (base_name + ext)
        if image_path.exists():
            return str(image_path)
    return None

def convert_json_to_yolo(json_path, txt_path, img_width=1280, img_height=720):
    """将JSON标注转换为YOLO格式"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 如果未指定图像尺寸，则从JSON数据中获取
    if img_width == 1280 and img_height == 720:  # 默认值
        img_width = data.get('imageWidth', img_width)
        img_height = data.get('imageHeight', img_height)

    yolo_lines = []
    # 中国象棋棋子类别映射
    class_mapping = {
        "A": 0,  # 红仕
        "B": 1,  # 红相
        "C": 2,  # 红炮
        "K": 3,  # 红帅
        "N": 4,  # 红马
        "P": 5,  # 红兵
        "R": 6,  # 红车
        "a": 7,  # 黑仕
        "b": 8,  # 黑相
        "c": 9,  # 黑炮
        "k": 10, # 黑将
        "n": 11, # 黑马
        "p": 12, # 黑卒
        "r": 13  # 黑车
    }

    for shape in data.get('shapes', []):
        label = shape.get('label')
        if label not in class_mapping:
            print(f"⚠️  未知标签 '{label}'，跳过")
            continue

        class_id = class_mapping[label]
        points = shape.get('points', [])

        if not points:
            continue

        # 根据形状类型计算边界框
        if shape.get('shape_type') == 'rectangle':
            # 矩形格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
        else:
            # 对于其他形状（如多边形），使用所有点的边界框
            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

        # 转换为YOLO格式: center_x, center_y, width, height (归一化)
        center_x = ((x_min + x_max) / 2) / img_width
        center_y = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

    with open(txt_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

def split_dataset(json_dir='dataset', train_ratio=0.8, val_ratio=0.2):
    """划分数据集（只包含训练集和验证集）"""
    # 创建目录结构
    create_dataset_structure()

    # 获取所有JSON文件
    json_files = list(Path(json_dir).glob('*.json'))
    if not json_files:
        print("❌ 未找到JSON标注文件")
        return

    print(f"📊 找到 {len(json_files)} 个JSON标注文件")

    # 随机打乱文件列表
    random.shuffle(json_files)

    # 计算划分索引
    train_end = int(len(json_files) * train_ratio)

    # 划分数据集
    train_files = json_files[:train_end]
    val_files = json_files[train_end:]

    print(f"📈 数据集划分:")
    print(f"   训练集: {len(train_files)} 个文件")
    print(f"   验证集: {len(val_files)} 个文件")

    # 处理训练集
    for json_file in train_files:
        process_file(json_file, 'train')

    # 处理验证集
    for json_file in val_files:
        process_file(json_file, 'val')

    print("✅ 数据集划分完成!")

def process_file(json_file, dataset_type):
    """处理单个文件"""
    # 获取对应图像文件
    image_file = get_image_from_json(str(json_file))
    if not image_file:
        print(f"⚠️  未找到 {json_file} 对应的图像文件，跳过")
        return

    # 复制图像文件到正确位置
    dest_image = f'dataset/images/{dataset_type}/{Path(image_file).name}'
    shutil.copy2(image_file, dest_image)

    # 转换并保存标注文件到正确位置
    dest_label = f'dataset/labels/{dataset_type}/{json_file.stem}.txt'
    convert_json_to_yolo(str(json_file), dest_label)

    print(f"📄 处理文件: {json_file.name} -> {dataset_type}")

if __name__ == '__main__':
    # 设置随机种子以确保可重复性
    random.seed(42)

    # 执行数据集划分
    split_dataset()
