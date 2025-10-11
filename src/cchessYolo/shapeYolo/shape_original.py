import json
import os
from PIL import Image, ImageEnhance
import numpy as np
import random
from typing import List, Tuple

dir = os.path.dirname(os.path.abspath(__file__))

# 配置路径 - 支持多个原始图像
data_dir = os.path.join(dir, "data/images")  # 原始数据目录
output_dir = os.path.join(dir, "data/images/train")

# 创建基础目录结构
os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

# 为每个类别创建子目录
classes = ['red_triangle', 'blue_circle', 'yellow_pentagon', 'green_rectangle']
for cls in classes:
    os.makedirs(os.path.join(output_dir, 'images', cls), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', cls), exist_ok=True)

# 类别映射（与 data_shape.yaml 一致）
class_map = {
    'red_triangle': 0,
    'blue_circle': 1,
    'yellow_pentagon': 2,
    'green_rectangle': 3
}

# 图像增强函数 - 增强更多样化
def augment_image(img, idx, prefix=""):
    augmented_images = []
    # 原图
    augmented_images.append((img, f"{prefix}_{idx}"))

    # 亮度调整 (更多级别)
    enhancer = ImageEnhance.Brightness(img)
    for i, factor in enumerate([0.7, 0.85, 1.15, 1.3]):
        augmented_images.append((enhancer.enhance(factor), f"{prefix}_{idx}_bright_{i}"))

    # 对比度调整
    enhancer = ImageEnhance.Contrast(img)
    for i, factor in enumerate([0.7, 0.85, 1.15, 1.3]):
        augmented_images.append((enhancer.enhance(factor), f"{prefix}_{idx}_contrast_{i}"))

    # 颜色饱和度调整
    enhancer = ImageEnhance.Color(img)
    for i, factor in enumerate([0.3, 0.6, 1.4, 1.7]):
        augmented_images.append((enhancer.enhance(factor), f"{prefix}_{idx}_color_{i}"))

    # 旋转增强 (更细粒度)
    for angle in range(0, 360, 30):  # 每30度旋转一次
        if angle == 0:  # 跳过0度（原图已包含）
            continue
        rotated_img = img.rotate(angle, expand=True)
        augmented_images.append((rotated_img, f"{prefix}_{idx}_rot_{angle}"))

        # 为部分旋转图像添加亮度变化
        if angle % 90 == 0:  # 仅对90度倍数的旋转图像添加亮度变化
            enhancer = ImageEnhance.Brightness(rotated_img)
            augmented_images.append((enhancer.enhance(0.8), f"{prefix}_{idx}_rot_{angle}_bright_low"))
            augmented_images.append((enhancer.enhance(1.2), f"{prefix}_{idx}_rot_{angle}_bright_high"))

    # 添加噪声增强
    def add_noise(img, strength=20):
        img_array = np.array(img)
        noise = np.random.normal(0, strength, img_array.shape)
        noisy_img = img_array + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    # 为原图添加不同强度的噪声
    for i, strength in enumerate([10, 25, 40]):
        noisy_img = add_noise(img, strength)
        augmented_images.append((noisy_img, f"{prefix}_{idx}_noise_{i}"))

    return augmented_images

# 更多样的裁剪增强
def crop_with_margin(img, x, y, w, h, margin_factor=0.1):
    """
    在目标周围添加边距进行裁剪，增加背景信息
    """
    width, height = img.size
    margin_w = int(w * margin_factor)
    margin_h = int(h * margin_factor)

    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(width, x + w + margin_w)
    y2 = min(height, y + h + margin_h)

    return img.crop((x1, y1, x2, y2)), (x - x1) / (x2 - x1), (y - y1) / (y2 - y1), w / (x2 - x1), h / (y2 - y1)

# 处理单个JSON文件
def process_json_file(json_path, image_path, file_prefix=""):
    # 加载图像
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"无法加载图像 {image_path}: {e}")
        return 0, {}

    # 加载 JSON 数据
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"无法加载JSON文件 {json_path}: {e}")
        return 0, {}

    total_samples = 0
    class_counts = {cls: 0 for cls in classes}

    for idx, shape in enumerate(data['shapes']):
        label = shape['label']
        if label not in classes:
            continue

        points = np.array(shape['points'], dtype=np.int32)
        shape_type = shape['shape_type']

        # 获取包围框
        if shape_type == 'rectangle' or shape_type == 'polygon':
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            x, y, w, h = int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
        elif shape_type == 'circle':
            center = points[0]
            edge = points[1]
            radius = int(np.linalg.norm(np.array(center) - np.array(edge)))
            x, y = int(center[0] - radius), int(center[1] - radius)
            w, h = 2 * radius, 2 * radius
        else:
            continue

        # 基本裁剪
        try:
            cropped_img = image.crop((x, y, x + w, y + h))
        except Exception as e:
            print(f"裁剪图像时出错: {e}")
            continue

        # 生成基础样本
        img_filename = f"{label}{file_prefix}_{idx}.jpg"
        cropped_img.save(os.path.join(output_dir, 'images', label, img_filename))

        # 生成对应的 YOLO 标签文件
        label_file = os.path.join(output_dir, 'labels', label, img_filename.replace('.jpg', '.txt'))
        with open(label_file, 'w') as f:
            # 归一化坐标 (对于裁剪后的图像，目标占据整个图像)
            cx = 0.5  # 中心点 x 归一化
            cy = 0.5  # 中心点 y 归一化
            bw = 1.0  # 宽度归一化
            bh = 1.0  # 高度归一化
            cls_idx = class_map[label]
            f.write(f"{cls_idx} {cx} {cy} {bw} {bh}\n")

        class_counts[label] += 1
        total_samples += 1

        # 获取图像增强版本
        augmented_images = augment_image(cropped_img, idx, file_prefix)

        # 保存增强图像
        for aug_img, suffix in augmented_images:
            img_filename = f"{label}{suffix}.jpg"
            try:
                aug_img.save(os.path.join(output_dir, 'images', label, img_filename))

                # 生成对应的 YOLO 标签文件
                label_file = os.path.join(output_dir, 'labels', label, img_filename.replace('.jpg', '.txt'))
                with open(label_file, 'w') as f:
                    # 归一化坐标 (对于裁剪后的图像，目标占据整个图像)
                    cx = 0.5  # 中心点 x 归一化
                    cy = 0.5  # 中心点 y 归一化
                    bw = 1.0  # 宽度归一化
                    bh = 1.0  # 高度归一化
                    cls_idx = class_map[label]
                    f.write(f"{cls_idx} {cx} {cy} {bw} {bh}\n")

                class_counts[label] += 1
                total_samples += 1
            except Exception as e:
                print(f"保存增强图像时出错 {img_filename}: {e}")

        # 带边距的裁剪增强
        try:
            margin_cropped_img, norm_x, norm_y, norm_w, norm_h = crop_with_margin(image, x, y, w, h)
            img_filename = f"{label}{file_prefix}_{idx}_margin.jpg"
            margin_cropped_img.save(os.path.join(output_dir, 'images', label, img_filename))

            # 生成对应的 YOLO 标签文件
            label_file = os.path.join(output_dir, 'labels', label, img_filename.replace('.jpg', '.txt'))
            with open(label_file, 'w') as f:
                cls_idx = class_map[label]
                f.write(f"{cls_idx} {norm_x + norm_w/2} {norm_y + norm_h/2} {norm_w} {norm_h}\n")

            class_counts[label] += 1
            total_samples += 1
        except Exception as e:
            print(f"带边距裁剪时出错: {e}")

    return total_samples, class_counts

# 主处理函数
def main():
    # 查找所有原始数据文件
    if not os.path.exists(data_dir):
        print(f"原始数据目录不存在: {data_dir}")
        return

    total_samples = 0
    overall_class_counts = {cls: 0 for cls in classes}

    # 查找所有JSON文件
    json_files = []
    for file in os.listdir(data_dir):
        if file.endswith('.json'):
            json_files.append(file)

    if not json_files:
        # 如果没有找到JSON文件，尝试使用原来的单个文件
        json_path = os.path.join(dir, "data/images/camera_capture_20250904_172816.json")
        image_path = os.path.join(dir, "data/images/camera_capture_20250904_172816.jpg")
        if os.path.exists(json_path) and os.path.exists(image_path):
            print("处理单个原始文件...")
            samples, class_counts = process_json_file(json_path, image_path, "_orig")
            total_samples += samples
            for cls in classes:
                overall_class_counts[cls] += class_counts.get(cls, 0)
        else:
            print("未找到任何原始数据文件")
            return
    else:
        # 处理所有找到的JSON文件
        print(f"找到 {len(json_files)} 个JSON文件，开始处理...")
        for json_file in json_files:
            base_name = json_file.replace('.json', '')
            json_path = os.path.join(data_dir, json_file)
            image_path = os.path.join(data_dir, base_name + '.jpg')

            # 检查对应的图像文件是否存在
            if not os.path.exists(image_path):
                # 尝试其他可能的图像扩展名
                for ext in ['.png', '.jpeg']:
                    alt_image_path = os.path.join(data_dir, base_name + ext)
                    if os.path.exists(alt_image_path):
                        image_path = alt_image_path
                        break

            if os.path.exists(image_path):
                print(f"处理文件: {json_file}")
                samples, class_counts = process_json_file(json_path, image_path, f"_{base_name}")
                total_samples += samples
                for cls in classes:
                    overall_class_counts[cls] += class_counts.get(cls, 0)
            else:
                print(f"找不到对应的图像文件: {image_path}")

    print("✅ 数据增强完成！")
    print("生成目录", output_dir)
    print(f"共生成 {total_samples} 个样本。")
    for cls, count in overall_class_counts.items():
        print(f"  {cls}: {count} 个样本")

if __name__ == "__main__":
    main()
