# fruit_yolo_obb_trainer.py

import argparse
import os
import math
from pathlib import Path
from ultralytics import YOLO

# 当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 当前文件所在的目录
current_dir = os.path.dirname(current_file_path)
# 上一层目录
parent_dir = os.path.dirname(current_dir)

class FruitOBBTrainer:
    def __init__(self, model_path='yolov8s-obb.pt'):
        """
        初始化水果OBB检测训练器
        """
        self.model_path = model_path
        self.classes = {
            "peach": 0,
            "y-apple": 1,
            "g-apple": 2,
            "g-grape": 3,
            "persimmon": 4,
            "apple": 5,
            "orange": 6,
            "citrus": 7
        }

    def create_data_yaml(self, data_dir, output_path):
        """
        创建YOLO OBB训练所需的数据配置文件

        Args:
            data_dir (str): 数据集根目录
            output_path (str): yaml配置文件输出路径
        """
        yaml_content = f"""
path: {os.path.abspath(data_dir)}
train: images/train
val: images/val

# 类别数量
nc: {len(self.classes)}

# 类别名称
names: {list(self.classes.keys())}
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content.strip())

        print(f"✅ 数据配置文件已创建: {output_path}")
        return output_path

    def train(self, data_yaml, epochs=100, imgsz=640, batch_size=16, device=''):
        """
        训练YOLO OBB模型

        Args:
            data_yaml (str): 数据配置文件路径
            epochs (int): 训练轮数
            imgsz (int): 输入图像尺寸
            batch_size (int): 批次大小
            device (str): 训练设备 ('cpu', '0', '1', etc.)
        """
        # 加载模型
        model = YOLO(self.model_path)

        # 设置训练参数
        train_params = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'name': 'fruit_obb_detection'
        }

        # 如果指定了设备，则添加到参数中
        if device:
            train_params['device'] = device

        # 开始训练
        print(" 开始训练YOLO OBB模型...")
        model.train(**train_params)

        print("✅ 训练完成!")
        return model

    def validate(self, model=None, data_yaml=None, device=''):
        """
        验证模型性能

        Args:
            model: 训练好的模型对象或路径
            data_yaml (str): 数据配置文件路径
            device (str): 验证设备
        """
        if isinstance(model, str):
            model = YOLO(model)
        elif model is None:
            model = YOLO(self.model_path)

        # 执行验证
        print(" 正在验证模型...")
        metrics = model.val(data=data_yaml, device=device)

        # 输出验证结果
        print(" 验证结果:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")

        return metrics

    def predict(self, source, model_path=None, conf=0.25, iou=0.8, save=True, save_dir='runs/obb_predict'):
        """
        使用训练好的模型进行预测

        Args:
            source (str): 输入源（图像或视频路径）
            model_path (str): 模型路径
            conf (float): 置信度阈值
            save (bool): 是否保存结果
            save_dir (str): 结果保存目录
        """
        # 加载模型
        if model_path is None:
            model_path = self.model_path
        model = YOLO(model_path)

        # 执行预测
        print(f" 正在对 {source} 进行预测...")
        results = model.predict(
            source=source,
            conf=conf,
            iou=iou,
            save=save,
            project=save_dir,
            name='predict',
            exist_ok=True
        )

        # 提取并输出详细检测信息
        detections = []
        class_names = list(self.classes.keys())

        for result in results:
            # 优先使用OBB信息
            if hasattr(result, 'obb') and result.obb is not None:
                # 处理OBB检测结果
                for i, obb in enumerate(result.obb):
                    # 获取类别索引和名称
                    class_id = int(obb.cls.item()) if hasattr(obb.cls, 'item') else int(obb.cls)
                    class_name = class_names[class_id] if class_id < len(class_names) else f"unknown_{class_id}"

                    # 获取置信度
                    confidence = obb.conf.item() if hasattr(obb.conf, 'item') else float(obb.conf)

                    # 获取OBB特有的信息
                    # 中心点坐标和角度
                    center_x, center_y, angle = 0.0, 0.0, 0.0
                    if hasattr(obb, 'xywhr') and obb.xywhr is not None:
                        xywhr = obb.xywhr[0] if len(obb.xywhr) > 0 else obb.xywhr
                        if hasattr(xywhr[0], 'item'):
                            center_x = float(xywhr[0].item())
                            center_y = float(xywhr[1].item())
                            angle = float(xywhr[4].item()) if len(xywhr) > 4 else 0.0
                        else:
                            center_x = float(xywhr[0])
                            center_y = float(xywhr[1])
                            angle = float(xywhr[4]) if len(xywhr) > 4 else 0.0

                    # 获取边界框顶点坐标（OBB特有）
                    bbox_vertices = []
                    if hasattr(obb, 'xyxyxyxy') and obb.xyxyxyxy is not None:
                        xyxyxyxy = obb.xyxyxyxy[0] if len(obb.xyxyxyxy) > 0 else obb.xyxyxyxy
                        # 正确处理张量数据
                        try:
                            # 如果是张量，转换为numpy数组
                            if hasattr(xyxyxyxy, 'cpu'):
                                xyxyxyxy = xyxyxyxy.cpu().numpy()

                            # 如果是多维数组，展平处理
                            if hasattr(xyxyxyxy, 'flatten'):
                                xyxyxyxy = xyxyxyxy.flatten()

                            # 转换为列表格式
                            if hasattr(xyxyxyxy, 'tolist'):
                                xyxyxyxy = xyxyxyxy.tolist()

                            # 确保我们有8个值（4个点的x,y坐标）
                            if len(xyxyxyxy) >= 8:
                                # 只取前8个值
                                for j in range(8):
                                    val = xyxyxyxy[j]
                                    # 如果还是数组，取第一个元素
                                    if hasattr(val, '__len__') and len(val) > 0:
                                        val = val[0]
                                    bbox_vertices.append(round(float(val), 2))
                            else:
                                # 如果不足8个值，用0填充
                                for j in range(8):
                                    if j < len(xyxyxyxy):
                                        val = xyxyxyxy[j]
                                        if hasattr(val, '__len__') and len(val) > 0:
                                            val = val[0]
                                        bbox_vertices.append(round(float(val), 2))
                                    else:
                                        bbox_vertices.append(0.0)
                        except Exception as e:
                            # 出现异常时，使用默认值
                            bbox_vertices = [0.0] * 8
                            print(f"处理边界框顶点时出错: {e}")

                    # 构建检测信息字典
                    detection_info = {
                        "index": i,
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(confidence, 4),
                        "center": [round(center_x, 2), round(center_y, 2)],
                        "bbox_vertices": bbox_vertices,
                        "angle": round(angle, 2)
                    }

                    detections.append(detection_info)

                    # 打印检测信息
                    print(f"检测 {i+1}: 类别={class_name}({class_id}), "
                          f"置信度={detection_info['confidence']}, "
                          f"中心点=({detection_info['center'][0]}, {detection_info['center'][1]}), "
                          f"角度={detection_info['angle']}°")
            else:
                # 如果没有OBB信息，回退到常规边界框处理
                boxes = result.boxes if hasattr(result, 'boxes') else None
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # 获取类别索引和名称
                        class_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
                        class_name = class_names[class_id] if class_id < len(class_names) else f"unknown_{class_id}"

                        # 获取置信度
                        confidence = box.conf.item() if hasattr(box.conf, 'item') else float(box.conf)

                        # 获取中心点坐标
                        center_x, center_y = 0.0, 0.0
                        if hasattr(box, 'xywh') and box.xywh is not None:
                            xywh = box.xywh[0] if len(box.xywh) > 0 else box.xywh
                            if hasattr(xywh[0], 'item'):
                                center_x = float(xywh[0].item())
                                center_y = float(xywh[1].item())
                            else:
                                center_x = float(xywh[0])
                                center_y = float(xywh[1])

                        # 获取边界框坐标
                        bbox = [0, 0, 0, 0]
                        if hasattr(box, 'xyxy') and box.xyxy is not None:
                            xyxy = box.xyxy[0] if len(box.xyxy) > 0 else box.xyxy
                            if hasattr(xyxy[0], 'item'):
                                bbox = [
                                    round(float(xyxy[0].item()), 2),
                                    round(float(xyxy[1].item()), 2),
                                    round(float(xyxy[2].item()), 2),
                                    round(float(xyxy[3].item()), 2)
                                ]
                            else:
                                bbox = [
                                    round(float(xyxy[0]), 2),
                                    round(float(xyxy[1]), 2),
                                    round(float(xyxy[2]), 2),
                                    round(float(xyxy[3]), 2)
                                ]

                        # 构建检测信息字典
                        detection_info = {
                            "index": i,
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": round(confidence, 4),
                            "center": [round(center_x, 2), round(center_y, 2)],
                            "bbox": bbox,
                            "angle": 0.0  # 常规边界框没有角度信息
                        }

                        detections.append(detection_info)

                        # 打印检测信息
                        print(f"检测 {i+1}: 类别={class_name}({class_id}), "
                              f"置信度={detection_info['confidence']}, "
                              f"中心点=({detection_info['center'][0]}, {detection_info['center'][1]}), "
                              f"边界框={detection_info['bbox']}, "
                              f"角度={detection_info['angle']}°")

        print(f"✅ 预测完成，共检测到 {len(detections)} 个目标")
        print(f"结果已保存至 {save_dir}/predict")
        return results, detections

def main():
    parser = argparse.ArgumentParser(description='YOLO OBB水果检测训练器')
    # parser.add_argument('--model', type=str, default=os.path.join(parent_dir ,'yolov8s-obb.pt'),
    #                     help='预训练模型路径')
    parser.add_argument('--model', type=str, default=os.path.join(current_dir ,'runs/obb/fruit_obb_detection4/weights/best.pt'),
                        help='预训练模型路径')
    parser.add_argument('--data_dir', type=str,default=os.path.join(parent_dir ,'yaml') ,
                        help='数据集根目录')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸')
    parser.add_argument('--batch', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--device', type=str, default='0',
                        help='训练设备 (e.g., "0" for GPU 0, "cpu" for CPU)')
    parser.add_argument('--task', type=str, choices=['train', 'val', 'predict'],
                        default='predict', help='任务类型')
    parser.add_argument('--source', type=str,default='/home/jetson/code/V10.10/Embodied/runner/calibration/images/RS_20251014_105504.jpg', help='预测时的输入源路径')
    args = parser.parse_args()

    # 创建训练器实例
    trainer = FruitOBBTrainer(model_path=args.model)

    if args.task == 'train':
        # 创建数据配置文件
        data_yaml = os.path.join(args.data_dir, 'data_fruit.yaml')
        if not os.path.exists(data_yaml):
            trainer.create_data_yaml(args.data_dir, data_yaml)

        # 开始训练
        trainer.train(
            data_yaml=data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch_size=args.batch,
            device=args.device
        )

    elif args.task == 'val':
        # 验证模型
        data_yaml = os.path.join(args.data_dir, 'dataset.yaml')
        trainer.validate(
            data_yaml=data_yaml,
            device=args.device
        )

    elif args.task == 'predict':
        if not args.source:
            raise ValueError("预测任务需要提供 --source 参数")

        # 执行预测
        trainer.predict(
            source=args.source,
            conf=0.25,
            iou=0.45,
            save=True
        )

if __name__ == '__main__':
    main()
