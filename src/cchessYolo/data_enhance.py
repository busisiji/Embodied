import xml.etree.ElementTree as ET
import cv2
import os
import numpy as np
from PIL import Image, ImageEnhance


# 1. 解析标注文件
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        objects.append({
            'name': name,
            'bbox_2d': [xmin, ymin, xmax, ymax]
        })
    return objects


# 2. 读取图像
def read_image(image_path):
    return cv2.imread(image_path)


# 3. 截取棋子
def crop_objects(image, objects):
    cropped_images = {}
    for obj in objects:
        name = obj['name']
        bbox_2d = obj['bbox_2d']
        cropped = image[bbox_2d[1]:bbox_2d[3], bbox_2d[0]:bbox_2d[2]]
        if name not in cropped_images:
            cropped_images[name] = []
        cropped_images[name].append(cropped)
    return cropped_images


# 4. 数据增强
def augment_image(image):
    # 转换为PIL Image
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 旋转
    angle = np.random.randint(-10, 10)
    image = image.rotate(angle)

    # 翻转
    if np.random.rand() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 缩放
    scale = np.random.uniform(0.8, 1.2)
    width, height = image.size
    new_size = (int(width * scale), int(height * scale))
    image = image.resize(new_size, Image.LANCZOS)

    # 亮度调整
    enhancer = ImageEnhance.Brightness(image)
    factor = np.random.uniform(0.8, 1.2)
    image = enhancer.enhance(factor)

    # 添加噪声
    image = np.array(image)
    noise = np.random.normal(0, 20, image.shape).astype(np.float64)
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    image = cv2.add(image, noise)

    return image


# 5. 保存增强后的图像
def save_augmented_images(cropped_images, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for name, images in cropped_images.items():
        class_dir = os.path.join(output_dir, name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        for i, image in enumerate(images):
            for j in range(5):  # 每张图片生成5张增强图片
                augmented = augment_image(image)
                save_path = os.path.join(class_dir, f'{name}_{i}_{j}.png')
                cv2.imwrite(save_path, cv2.cvtColor(np.array(augmented), cv2.COLOR_RGB2BGR))


# 主函数
def main(image_path, xml_path, output_dir):
    # 解析XML
    objects = parse_xml(xml_path)

    # 读取图像
    image = read_image(image_path)

    # 截取棋子
    cropped_images = crop_objects(image, objects)

    # 保存增强后的图像
    save_augmented_images(cropped_images, output_dir)


# 使用示例
image_path = 'RS_20250730_104247.jpg'
xml_path = 'RS_20250730_104247.xml'
output_dir = 'data/augmented_chess_pieces'
main(image_path, xml_path, output_dir)
