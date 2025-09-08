import os
import shutil
import random
import glob

# 获取当前目录
dir = os.path.dirname(os.path.abspath(__file__))

# 定义路径
train_dir = os.path.join(dir, "data/images/train")
val_dir = os.path.join(dir, "data/images/val")
test_dir = os.path.join(dir, "data/images/test")

# 创建验证和测试目录
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 为验证集和测试集创建子目录
classes = ['red_triangle', 'blue_circle', 'yellow_pentagon', 'green_rectangle']
for cls in classes:
    os.makedirs(os.path.join(val_dir, 'images', cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'labels', cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'images', cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'labels', cls), exist_ok=True)

# 统计每个类别的样本数量
print("数据集划分信息:")
total_moved = 0

for cls in classes:
    # 获取该类别的所有图像文件
    image_files = glob.glob(os.path.join(train_dir, 'images', cls, "*.jpg"))

    # 随机打乱文件列表
    random.shuffle(image_files)

    # 计算划分数量
    total_count = len(image_files)
    val_count = max(1, int(total_count * 0.1))  # 10%作为验证集
    test_count = max(1, int(total_count * 0.1))  # 10%作为测试集
    train_count = total_count - val_count - test_count  # 剩余作为训练集

    print(f"{cls}: 总计{total_count}张, 训练集{train_count}张, 验证集{val_count}张, 测试集{test_count}张")

    # 移动验证集数据
    for i, img_path in enumerate(image_files[:val_count]):
        filename = os.path.basename(img_path)
        # 移动图像
        shutil.move(img_path, os.path.join(val_dir, 'images', cls, filename))
        # 移动对应的标签文件
        label_filename = filename.replace('.jpg', '.txt')
        label_path = os.path.join(train_dir, 'labels', cls, label_filename)
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(val_dir, 'labels', cls, label_filename))
        total_moved += 1

    # 移动测试集数据
    for i, img_path in enumerate(image_files[val_count:val_count+test_count]):
        filename = os.path.basename(img_path)
        # 移动图像
        shutil.move(img_path, os.path.join(test_dir, 'images', cls, filename))
        # 移动对应的标签文件
        label_filename = filename.replace('.jpg', '.txt')
        label_path = os.path.join(train_dir, 'labels', cls, label_filename)
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(test_dir, 'labels', cls, label_filename))
        total_moved += 1

print(f"\n✅ 数据集划分完成！共移动了 {total_moved} 个文件。")

# 创建新的数据配置文件
data_yaml_content = f"""path: {os.path.join(dir, "data/images")}
train: train
val: val
test: test

# Classes
names:
  0: red_triangle
  1: blue_circle
  2: yellow_pentagon
  3: green_rectangle
"""

# 保存数据配置文件
data_yaml_path = os.path.join(dir, "data_shape_complete.yaml")
with open(data_yaml_path, 'w', encoding='utf-8') as f:
    f.write(data_yaml_content)

print(f"数据配置文件已保存到: {data_yaml_path}")
print("\n现在您的数据集已经完整，包含:")
print("- 训练集 (train): 80% 数据")
print("- 验证集 (val): 10% 数据")
print("- 测试集 (test): 10% 数据")
