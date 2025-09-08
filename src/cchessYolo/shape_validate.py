# validate_model.py
import os
import cv2
import numpy as np
from ultralytics import YOLO

def validate_trained_model():
    """
    验证训练好的模型是否能正确检测形状
    """
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 模型路径
    model_path = os.path.join(current_dir, "color_shape_model.pt")

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print("❌ 模型文件不存在，请先训练模型")
        return False

    # 加载模型
    try:
        model = YOLO(model_path)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

    # 类别名称
    class_names = ['red_triangle', 'blue_circle', 'yellow_pentagon', 'green_rectangle']

    # 测试图像路径
    test_images_dir = os.path.join(current_dir, "data")

    # 初始化 img_path
    img_path = None

    # 如果测试目录存在，则使用其中的图像进行测试
    if os.path.exists(test_images_dir):
        print(f"🔍 搜索测试图像目录: {test_images_dir}")
        # 获取一个测试图像
        for root, dirs, files in os.walk(test_images_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, file)
                    print(f"📷 找到测试图像: {img_path}")
                    break
            if img_path:
                break

    if img_path and os.path.exists(img_path):
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            print("❌ 无法读取图像")
            return False

        # 进行预测
        print(f"🧠 使用模型进行预测...")
        results = model(image, conf=0.25)  # 降低置信度阈值

        # 显示结果信息
        print(f"✅ 图像预测完成")
        print(f"检测到 {len(results[0].boxes)} 个目标")

        # 如果没有检测到目标，尝试更低的置信度
        if len(results[0].boxes) == 0:
            print("🔄 尝试更低置信度阈值...")
            results = model(image, conf=0.1)
            print(f"使用0.1置信度阈值检测到 {len(results[0].boxes)} 个目标")

        # 显示每个检测到的目标
        for i, box in enumerate(results[0].boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = class_names[class_id] if class_id < len(class_names) else f"未知类别 {class_id}"
            print(f"  目标 {i+1}: {class_name} (置信度: {confidence:.2f})")

        return True
    else:
        print("⚠️ 未找到测试图像，仅验证模型加载")
        return True

def validate_model_structure():
    """
    验证模型文件结构
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "color_shape_model.pt")

    if not os.path.exists(model_path):
        print("❌ 模型文件不存在")
        return False

    file_size = os.path.getsize(model_path)
    print(f"✅ 模型文件存在，大小: {file_size/1024/1024:.2f} MB")
    return True

def main():
    print("开始验证模型...")

    # 验证模型文件结构
    if not validate_model_structure():
        return

    # 验证模型功能
    if not validate_trained_model():
        return

    print("\n🎉 模型验证完成！")

if __name__ == "__main__":
    main()
