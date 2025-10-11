# detect_shapes.py
import cv2
import torch
from ultralytics import YOLO
import numpy as np

class ShapesDetector:
    def __init__(self, model_path='runs/detect/shapes_detection/weights/best.pt'):
        """
        初始化形状检测器

        Args:
            model_path: 训练好的模型路径
        """
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        # 类别名称
        self.class_names = ["red_triangle", "blue_circle", "yellow_pentagon", "green_rectangle"]

        # 为每种形状分配不同颜色
        self.colors = [
            (0, 0, 255),    # red_triangle - 红色
            (255, 0, 0),    # blue_circle - 蓝色
            (0, 255, 255),  # yellow_pentagon - 黄色
            (0, 255, 0)     # green_rectangle - 绿色
        ]

    def detect(self, image_path, conf_threshold=0.5, iou_threshold=0.45, save_path='result.jpg'):
        """
        检测图像中的形状物体

        Args:
            image_path: 输入图像路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            save_path: 结果保存路径
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 执行检测
        results = self.model(image, conf=conf_threshold, iou=iou_threshold)

        # 可视化检测结果
        vis_image = self.visualize_detections(image, results)

        # 保存结果图像
        cv2.imwrite(save_path, vis_image)
        print(f"✅ 检测结果已保存至: {save_path}")

        return results

    def visualize_detections(self, image, results):
        """
        可视化检测结果

        Args:
            image: 输入图像
            results: 检测结果
        """
        # 复制图像用于绘制
        img_vis = image.copy()

        # 获取检测结果
        boxes = results[0].boxes if hasattr(results, '__len__') and len(results) > 0 else None

        if boxes is not None and len(boxes) > 0:
            # 遍历每个检测到的物体
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                # 绘制边界框
                color = self.colors[cls]
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)

                # 绘制标签
                label = f'{self.class_names[cls]} {conf:.2f}'
                cv2.putText(img_vis, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img_vis

# 使用示例
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='runs/detect/shapes_detection/weights/best.pt',
                        help='模型路径')
    parser.add_argument('--image_path', type=str, required=True,
                        help='输入图像路径')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='置信度阈值')
    parser.add_argument('--iou_threshold', type=float, default=0.45,
                        help='IOU阈值')
    parser.add_argument('--save_path', type=str, default='result.jpg',
                        help='结果保存路径')

    args = parser.parse_args()

    # 创建检测器实例
    detector = ShapesDetector(args.model_path)

    # 执行检测
    detector.detect(
        image_path=args.image_path,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        save_path=args.save_path
    )
