import os
import shutil
import glob
import random
from ultralytics import YOLO

def validate_dataset_structure(base_dir, classes):
    """
    验证数据集结构是否正确
    """
    print("正在验证数据集结构...")

    splits = ['train', 'val', 'test']
    valid = True

    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            print(f"❌ 缺少 {split} 目录: {split_dir}")
            valid = False
            continue

        # 检查每个类别目录
        for cls in classes:
            img_dir = os.path.join(split_dir, 'images', cls)
            label_dir = os.path.join(split_dir, 'labels', cls)

            if not os.path.exists(img_dir):
                print(f"❌ 缺少图像目录: {img_dir}")
                valid = False

            if not os.path.exists(label_dir):
                print(f"❌ 缺少标签目录: {label_dir}")
                valid = False

            # 检查文件数量是否匹配
            if os.path.exists(img_dir) and os.path.exists(label_dir):
                img_files = set([f.replace('.jpg', '') for f in os.listdir(img_dir) if f.endswith('.jpg')])
                label_files = set([f.replace('.txt', '') for f in os.listdir(label_dir) if f.endswith('.txt')])

                if img_files != label_files:
                    print(f"⚠️  {split}/{cls} 图像和标签文件数量不匹配")
                    print(f"   图像: {len(img_files)}, 标签: {len(label_files)}")

    if valid:
        print("✅ 数据集结构验证通过")
    else:
        print("❌ 数据集结构验证失败")

    return valid

def count_dataset_samples(base_dir, classes):
    """
    统计数据集中每个类别的样本数量
    """
    print("\n数据集统计信息:")
    splits = ['train', 'val', 'test']

    for split in splits:
        print(f"\n{split.upper()} SET:")
        split_total = 0
        split_dir = os.path.join(base_dir, split)

        for cls in classes:
            img_dir = os.path.join(split_dir, 'images', cls)
            if os.path.exists(img_dir):
                count = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
                print(f"  {cls}: {count} 张图片")
                split_total += count
            else:
                print(f"  {cls}: 0 张图片")

        print(f"  总计: {split_total} 张图片")

def validate_model_performance(metrics, min_map50=0.5, min_map50_95=0.3):
    """
    验证模型性能是否达到基本要求
    """
    print("\n模型性能验证:")

    map50 = metrics.box.map50 if metrics.box.map50 is not None else 0
    map50_95 = metrics.box.map if metrics.box.map is not None else 0

    print(f"  mAP50: {map50:.4f} (要求: ≥{min_map50})")
    print(f"  mAP50-95: {map50_95:.4f} (要求: ≥{min_map50_95})")

    if map50 >= min_map50 and map50_95 >= min_map50_95:
        print("✅ 模型性能验证通过")
        return True
    else:
        print("⚠️ 模型性能未达到基本要求")
        return False

def validate_model_files(model_path):
    """
    验证模型文件是否存在且有效
    """
    print("\n模型文件验证:")

    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        print(f"✅ 模型文件存在，大小: {file_size/1024/1024:.2f} MB")
        return True
    else:
        print("❌ 模型文件不存在")
        return False

def main():
    # 获取当前目录
    dir = os.path.dirname(os.path.abspath(__file__))

    # 定义路径
    train_dir = os.path.join(dir, "data/images/train")
    val_dir = os.path.join(dir, "data/images/val")
    test_dir = os.path.join(dir, "data/images/test")
    base_data_dir = os.path.join(dir, "data/images")

    # 检查训练数据是否存在
    if not os.path.exists(train_dir):
        print("错误: 训练数据目录不存在!")
        print("请先运行数据生成脚本生成训练数据")
        return

    # 检查训练数据是否包含各类别子目录
    classes = ['red_triangle', 'blue_circle', 'yellow_pentagon', 'green_rectangle']
    train_has_classes = all(os.path.exists(os.path.join(train_dir, 'images', cls)) for cls in classes)
    if not train_has_classes:
        print("错误: 训练数据目录结构不完整!")
        return

    # 创建验证和测试目录
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 为验证集和测试集创建子目录
    for cls in classes:
        os.makedirs(os.path.join(val_dir, 'images', cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, 'labels', cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'images', cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'labels', cls), exist_ok=True)

    # 统计并划分数据集
    print("正在划分数据集...")
    total_moved = 0

    for cls in classes:
        # 获取该类别的所有图像文件
        image_files = glob.glob(os.path.join(train_dir, 'images', cls, "*.jpg"))

        if not image_files:
            print(f"警告: 类别 {cls} 没有找到图像文件")
            continue

        # 随机打乱文件列表
        random.shuffle(image_files)

        # 计算划分数量
        total_count = len(image_files)
        val_count = max(1, int(total_count * 0.1))  # 10%作为验证集
        test_count = max(1, int(total_count * 0.1))  # 10%作为测试集

        print(f"{cls}: 总计{total_count}张图片")

        # 移动验证集数据
        for img_path in image_files[:val_count]:
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
        for img_path in image_files[val_count:val_count+test_count]:
            filename = os.path.basename(img_path)
            # 移动图像
            shutil.move(img_path, os.path.join(test_dir, 'images', cls, filename))
            # 移动对应的标签文件
            label_filename = filename.replace('.jpg', '.txt')
            label_path = os.path.join(train_dir, 'labels', cls, label_filename)
            if os.path.exists(label_path):
                shutil.move(label_path, os.path.join(test_dir, 'labels', cls, label_filename))
            total_moved += 1

    print(f"✅ 数据集划分完成！共移动了 {total_moved} 个文件。")

    # 创建数据配置文件 (使用相对路径)
    data_yaml_content = """path: .
train: data/images/train
val: data/images/val
test: data/images/test

# Classes
names:
  0: red_triangle
  1: blue_circle
  2: yellow_pentagon
  3: green_rectangle
"""

    # 保存数据配置文件
    data_yaml_path = os.path.join(dir, "data_shape_training.yaml")
    with open(data_yaml_path, 'w', encoding='utf-8') as f:
        f.write(data_yaml_content)

    print(f"数据配置文件已保存到: {data_yaml_path}")

    # 验证数据集结构
    if not validate_dataset_structure(base_data_dir, classes):
        print("数据集结构验证失败，停止训练")
        return

    # 统计数据集样本
    count_dataset_samples(base_data_dir, classes)

    # 检查数据集完整性
    print("\n检查数据集完整性...")
    required_paths = [
        os.path.join("data/images/train"),
        os.path.join("data/images/val"),
        os.path.join("data/images/test")
    ]

    missing_paths = []
    for path in required_paths:
        full_path = os.path.join(dir, path)
        if not os.path.exists(full_path):
            missing_paths.append(full_path)

    if missing_paths:
        print("错误: 以下目录不存在:")
        for path in missing_paths:
            print(f"  - {path}")
        return

    print("✅ 数据集完整性检查通过")

    # 加载预训练的 YOLOv8 模型
    print("\n正在加载预训练模型...")
    try:
        model = YOLO('yolov8n.pt')  # 使用 nano 版本
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"错误: 模型加载失败: {e}")
        return

    print("开始训练模型...")

    # 训练模型
    try:
        results = model.train(
            data=data_yaml_path,      # 使用我们创建的数据配置文件
            epochs=50,                # 训练轮数 (减少到50以便快速测试)
            imgsz=640,                # 图像大小
            batch=8,                  # 批次大小 (减少以适应较小的数据集)
            name='color_shape_detection',  # 实验名称
            patience=5,               # 早停轮数
            verbose=True              # 显示详细信息
        )
        print("✅ 训练完成！")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        return

    # 加载训练好的模型
    best_model_path = os.path.join('runs', 'detect', 'color_shape_detection', 'weights', 'best.pt')
    final_model_path = os.path.join('runs', 'detect', 'color_shape_detection', 'weights', 'last.pt')

    if os.path.exists(best_model_path):
        trained_model = YOLO(best_model_path)
        print("✅ 成功加载最佳模型")
        model_to_use = best_model_path
    elif os.path.exists(final_model_path):
        trained_model = YOLO(final_model_path)
        print("✅ 成功加载最终模型")
        model_to_use = final_model_path
    else:
        trained_model = model
        print("⚠️ 未找到训练好的模型，使用原始模型")
        model_to_use = 'yolov8n.pt'

    # 在测试集上评估模型
    print("\n在测试集上评估模型...")
    try:
        metrics = trained_model.val(data=data_yaml_path, split='test')
        print(f"测试集 mAP50: {metrics.box.map50:.4f}")
        print(f"测试集 mAP50-95: {metrics.box.map:.4f}")

        # 验证模型性能
        validate_model_performance(metrics)
    except Exception as e:
        print(f"评估过程中出现错误: {e}")

    # 保存模型
    output_model_path = os.path.join(dir, "color_shape_model.pt")
    try:
        trained_model.save(output_model_path)
        print(f"✅ 模型已保存到: {output_model_path}")

        # 验证保存的模型文件
        validate_model_files(output_model_path)
    except Exception as e:
        print(f"保存模型时出现错误: {e}")

    print("\n🎉 模型训练和评估完成！")
    # 在训练完成后添加验证代码
    print("\n检查训练结果...")
    if results:
        print(f"训练轮数: {results.epoch+1}")
        print(f"最终mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"最终mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")


if __name__ == '__main__':
    main()
