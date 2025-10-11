import argparse
import random
import shutil

import onnx
import pyrealsense2
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import os
import torch

# 动态导入TensorRT相关库（如果可用）
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("⚠️  TensorRT未安装，将使用标准PyTorch模型")

# 动态导入ONNX Runtime相关库（如果可用）
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  ONNX Runtime未安装，将使用标准PyTorch模型")

from utils.calibrationManager import apply_perspective_correction, camera_to_chess_position
dir = os.path.dirname(os.path.abspath(__file__))
def split_dataset(images_dir, output_dir, train_ratio=0.8):
    """
    将Images目录中的jpg和json文件划分为训练集和验证集

    Args:
        images_dir (str): 包含图像和标注文件的源目录路径
        output_dir (str): 输出目录路径
        train_ratio (float): 训练集比例，默认为0.8
    """
    # 创建目录结构
    train_images_dir = Path(output_dir) / "images" / "train"
    train_labels_dir = Path(output_dir) / "labels" / "train"
    val_images_dir = Path(output_dir) / "images" / "val"
    val_labels_dir = Path(output_dir) / "labels" / "val"

    for directory in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # 获取所有jpg文件
    images_dir = Path(images_dir)
    image_files = list(images_dir.glob("*.jpg"))

    # 随机打乱文件列表
    random.shuffle(image_files)

    # 计算训练集数量
    train_count = int(len(image_files) * train_ratio)

    # 划分训练集和验证集
    train_files = image_files[:train_count]
    val_files = image_files[train_count:]

    # 处理训练集文件
    for img_file in train_files:
        # 复制图像文件
        shutil.copy(img_file, train_images_dir / img_file.name)

        # 复制对应的json标注文件
        json_file = img_file.with_suffix('.json')
        if json_file.exists():
            # 将.json文件复制为.txt格式的YOLO标签文件
            convert_json_to_txt(json_file, train_labels_dir / (img_file.stem + '.txt'))

    # 处理验证集文件
    for img_file in val_files:
        # 复制图像文件
        shutil.copy(img_file, val_images_dir / img_file.name)

        # 复制对应的json标注文件
        json_file = img_file.with_suffix('.json')
        if json_file.exists():
            # 将.json文件复制为.txt格式的YOLO标签文件
            convert_json_to_txt(json_file, val_labels_dir / (img_file.stem + '.txt'))

    print(f"数据集划分完成:")
    print(f"  训练集: {len(train_files)} 个样本")
    print(f"  验证集: {len(val_files)} 个样本")
    print(f"  输出目录: {output_dir}")

def convert_json_to_txt(json_file, txt_file):
    """
    将LabelMe格式的JSON标注文件转换为YOLO格式的TXT标签文件

    Args:
        json_file (Path): JSON文件路径
        txt_file (Path): 输出TXT文件路径
    """
    import json

    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取图像尺寸
    img_width = data['imageWidth']
    img_height = data['imageHeight']

    # 类别名称映射到索引
    class_names = ["A", "B", "C", "K", "N", "P", "R", "a", "b", "c", "k", "n", "p", "r"]
    class_mapping = {name: idx for idx, name in enumerate(class_names)}

    # 处理shapes中的标签
    lines = []
    if 'shapes' in data:
        for shape in data['shapes']:
            label = shape.get('label', '')
            if label in class_mapping:
                # 获取边界框坐标
                points = shape['points']
                x_coords = [point[0] for point in points]
                y_coords = [point[1] for point in points]

                # 计算边界框
                x_min = min(x_coords)
                y_min = min(y_coords)
                x_max = max(x_coords)
                y_max = max(y_coords)

                # 转换为YOLO格式 (center_x, center_y, width, height)
                center_x = ((x_min + x_max) / 2) / img_width
                center_y = ((y_min + y_max) / 2) / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height

                # 添加到行列表
                class_idx = class_mapping[label]
                lines.append(f"{class_idx} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

    # 写入TXT文件
    with open(txt_file, 'w') as f:
        f.writelines(lines)

class ChessPieceDetectorSeparate():
    def __init__(self, model_path='yolov8s.pt'):
        """
        初始化棋子检测器 - 红黑棋子分别识别
        支持.pt、.trt/.engine和.onnx格式
        """
        self.model_path = model_path
        self.is_trt_model = model_path.endswith(('.trt', '.engine'))
        self.is_onnx_model = model_path.endswith('.onnx')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 使用设备: {self.device}")

        if self.is_trt_model and TENSORRT_AVAILABLE:
            print("🔧 加载TensorRT优化模型...")
            self.model = self._load_trt_model(model_path)
        elif self.is_onnx_model and ONNX_AVAILABLE:
            print("🔧 加载ONNX模型...")
            self.model = self._load_onnx_model(model_path)
        else:
            print("🔧 加载标准YOLO模型...")
            self.model = YOLO(model_path)
            # 将模型移动到指定设备
            self.model.to(self.device)

        # 红黑双方各7种棋子
        self.class_names = ["A","B", "C", "K", "N", "P", "R", "a", "b", "c", "k", "n", "p", "r"]

        # 为每种棋子分配不同颜色
        self.colors = [
            # 红方棋子 - 红色系
            (0, 0, 255),    # red_general - 纯红
            (0, 50, 255),   # red_advisor - 橙红
            (0, 100, 255),  # red_elephant - 橙色
            (0, 150, 255),  # red_horse - 橙黄
            (0, 200, 255),  # red_chariot - 黄橙
            (0, 255, 255),  # red_cannon - 黄色
            (100, 255, 255),# red_soldier - 浅黄

            # 黑方棋子 - 蓝色系/黑色系
            (255, 0, 0),    # black_general - 蓝色
            (255, 50, 0),   # black_advisor - 深蓝
            (255, 100, 0),  # black_elephant - 靛蓝
            (255, 150, 0),  # black_horse - 紫蓝
            (255, 200, 0),  # black_chariot - 青蓝
            (255, 255, 0),  # black_cannon - 青色
            (128, 128, 0)   # black_soldier - 灰色
        ]


    def _load_onnx_model(self, onnx_path):
        """
        加载ONNX模型
        """
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime未安装，无法加载ONNX模型")

        try:
            # 根据CUDA可用性选择执行提供者
            providers = []
            if torch.cuda.is_available():
                # 尝试不同的CUDA提供者
                cuda_providers = [
                    'CUDAExecutionProvider',
                    'TensorrtExecutionProvider',  # 如果安装了TensorRT
                    'CPUExecutionProvider'
                ]
                # 检查哪些提供者可用
                available_providers = ort.get_available_providers()
                for provider in cuda_providers:
                    if provider in available_providers:
                        providers.append(provider)
                        break  # 只使用第一个可用的提供者
            else:
                providers = ['CPUExecutionProvider']

            print(f"🔧 使用ONNX提供者: {providers}")

            # 创建会话选项
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.intra_op_num_threads = 4  # 限制线程数以减少内存使用

            self.onnx_session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=providers
            )
            print("✅ ONNX模型加载成功")
            return None
        except Exception as e:
            print(f"⚠️  ONNX模型加载失败: {e}")
            print("🔄 回退到标准PyTorch模型")
            self.is_onnx_model = False
            # 尝试找到对应的.pt文件
            pt_path = onnx_path.replace('.onnx', '.pt')
            if not os.path.exists(pt_path):
                # 如果没有.pt文件，使用默认的yolov8s.pt
                pt_path = 'yolov8s.pt'
            model = YOLO(pt_path)
            model.to(self.device)
            return model

    def _onnx_inference(self, image_path,conf_threshold,iou_threshold):
        """
        使用ONNX模型执行推理
        """
        try:
            # 直接使用YOLO加载ONNX模型进行推理
            model = YOLO(self.model_path)
            results = model(
                image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=640,
                save=False,
                show=False,
                device='cpu' if not torch.cuda.is_available() else 0
            )
            return  results
        except Exception as e:
            print(f"⚠️  ONNX推理失败，回退到标准方法: {e}")

    def convert_to_onnx(self, output_path=None, imgsz=640, dynamic=False):
        """
        将YOLO模型转换为ONNX格式

        Args:
            output_path: 输出ONNX模型路径
            imgsz: 输入图像尺寸
            dynamic: 是否使用动态输入尺寸
        """
        if not self.model_path.endswith('.pt'):
            print("❌  只有PyTorch模型(.pt)可以转换为ONNX格式")
            return None

        if output_path is None:
            # 保存在与.pt文件相同的路径下
            model_name = Path(self.model_path).stem
            output_dir = Path(self.model_path).parent
            output_path = str(output_dir / f"{model_name}.onnx")

        try:
            # 使用ultralytics提供的导出功能
            model = YOLO(self.model_path)

            # 导出为ONNX格式，添加更多兼容性选项
            export_args = {
                'format': 'onnx',
                'imgsz': imgsz,
                'dynamic': False,  # 根据__main__中的设置改为False
                'simplify': True,
                'opset': 17,  # 使用与__main__中相同的opset版本
                'device': 0 if torch.cuda.is_available() else 'cpu'
            }

            # 如果是CPU环境，添加额外的兼容性选项
            if not torch.cuda.is_available():
                export_args['opset'] = 11  # 更低的opset版本
                export_args['half'] = False  # 禁用半精度

            model.export(**export_args)

            # 重命名生成的文件
            generated_path = self.model_path.replace('.pt', '.onnx')
            if os.path.exists(generated_path):
                os.rename(generated_path, output_path)

            # 设置ONNX IR版本
            self.set_onnx_ir_version(output_path)

            print(f"✅ 模型已成功转换为ONNX格式: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ 模型转换失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def set_onnx_ir_version(self, onnx_path, ir_version=8):
        """
        设置ONNX模型的IR版本

        Args:
            onnx_path: ONNX模型路径
            ir_version: 目标IR版本
        """
        try:
            model = onnx.load(onnx_path)
            model.ir_version = ir_version
            onnx.save(model, onnx_path)
            print(f"✅ ONNX IR版本已设置为: {ir_version}")
        except Exception as e:
            print(f"⚠️  设置ONNX IR版本失败: {e}")

    def train(self, data_yaml='yaml/data.yaml', epochs=100, imgsz=640):
        """
        训练棋子检测模型
        """
        # 确保使用的是.pt模型进行训练
        train_model_path = self.model_path
        if self.model_path.endswith(('.trt', '.engine', '.onnx')):
            train_model_path = self._get_pt_model_path()

        model = YOLO(train_model_path)
        # 根据CUDA可用性选择设备
        device = 0 if torch.cuda.is_available() else 'cpu'
        print(f"设备: {device}")
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=16,
            device=device,
            name='chess_piece_detection_separate'
        )

    def _get_pt_model_path(self):
        """
        获取对应的.pt模型路径
        """
        if self.model_path.endswith(('.trt', '.engine')):
            return self.model_path.replace('.trt', '.pt').replace('.engine', '.pt')
        elif self.model_path.endswith('.onnx'):
            return self.model_path.replace('.onnx', '.pt')
        return self.model_path

    def _run_inference(self, image_path,image, conf_threshold=0.5, iou_threshold=0.25):
        """
        根据模型类型执行推理
        """
        if self.is_trt_model and TENSORRT_AVAILABLE:
            return
        elif self.is_onnx_model and ONNX_AVAILABLE:
            return self._onnx_inference(image_path, conf_threshold=0.5, iou_threshold=0.25)
        else:
            return self.model(image, conf=conf_threshold, iou=iou_threshold)

    def _filter_duplicate_detections(self, boxes, iou_threshold=0.5):
        """
        过滤重复检测框
        """
        if len(boxes) <= 1:
            return boxes

        # 按置信度排序
        confidences = [float(box.conf[0].cpu().numpy()) for box in boxes]
        sorted_indices = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)

        keep_boxes = []
        used_indices = set()

        for i in sorted_indices:
            if i in used_indices:
                continue

            current_box = boxes[i]
            keep_boxes.append(current_box)
            used_indices.add(i)

            # 计算与当前框的IOU，标记重叠的低置信度框
            for j in sorted_indices:
                if j in used_indices:
                    continue

                iou = self._calculate_iou(current_box.xyxy[0].cpu().numpy(),
                                        boxes[j].xyxy[0].cpu().numpy())
                if iou > iou_threshold:
                    used_indices.add(j)

        return keep_boxes

    def _calculate_iou(self, box1, box2):
        """
        计算两个边界框的IOU
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def detect(self, image_path, conf_threshold=0.3, iou_threshold=0.45, save_path='result.jpg'):
        """
        检测图像中的棋子并保存结果图片
        根据__main__部分优化推理流程

        :param image_path: 输入图像路径
        :param conf_threshold: 置信度阈值
        :param iou_threshold: IOU 阈值
        :param save_path: 保存结果图像的路径
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        # 执行检测
        results = self._run_inference(image_path,image, conf_threshold, iou_threshold)

        # 可视化检测结果
        vis_image = self.visualize_detections(image, results)

        # 保存结果图像
        cv2.imwrite(save_path, vis_image)
        print(f"✅ 检测结果已保存至: {save_path}")

    def visualize_detections(self, image, results):
        """
        可视化检测结果（增加去重逻辑）
        """
        # 复制图像用于绘制
        img_vis = image.copy()

        # 获取检测结果
        boxes = None
        if hasattr(results, '__len__') and len(results) > 0:
            boxes = results[0].boxes if hasattr(results[0], 'boxes') else None
        else:
            boxes = results[0].boxes if hasattr(results, '__len__') and hasattr(results[0], 'boxes') else None

        if boxes is not None and len(boxes) > 0:
            # 添加去重逻辑：对相同类别的重叠框进行聚类
            filtered_boxes = self._filter_duplicate_detections(boxes)

            # 遍历每个检测到的棋子
            for box in filtered_boxes:
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

    def extract_chessboard_layout_with_height(self, image, chess_points, half_board="red",
                                              conf_threshold=0.5, iou_threshold=0.4):
        """
        从图像中提取棋盘布局

        :param image: 输入图像
        :param chess_points: 棋盘关键点
        :param half_board: 半区类型 ("red" 或 "black")
        :param conf_threshold: 置信度阈值
        :param iou_threshold: IOU阈值
        :return: 棋盘布局矩阵和检测结果
        """
        # 处理空图像情况
        if image is None:
            empty_layout = [['.' for _ in range(9)] for _ in range(5)]
            return empty_layout, None, {}

        # 执行检测
        results = self._run_inference(None, image, conf_threshold, iou_threshold)

        # 初始化5x9棋盘（半个棋盘）
        chess_layout = [['.' for _ in range(9)] for _ in range(5)]
        points_center = {}

        # 获取检测结果
        boxes = results[0].boxes if hasattr(results, '__len__') and hasattr(results[0], 'boxes') else None

        if boxes is None or len(boxes) == 0:
            return chess_layout, results, points_center

        # 提取所有有效检测
        detections = []
        for box in boxes:
            # 获取边界框坐标和类别
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())

            # 计算棋子中心点
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # 将图像坐标转换为棋盘坐标
            chess_pos = camera_to_chess_position(center_x, center_y, chess_points)

            if chess_pos is not None:
                detections.append({
                    'box': box,
                    'chess_pos': chess_pos,
                    'conf': conf,
                    'cls': cls,
                    'center': (center_x, center_y),
                })

        if not detections:
            return chess_layout, results, points_center

        # 按照棋盘位置对检测结果进行分组，并保留置信度最高的
        position_detections = {}
        for detection in detections:
            row, col = detection['chess_pos']
            position_key = (row, col)

            if position_key not in position_detections:
                position_detections[position_key] = detection
            elif detection['conf'] > position_detections[position_key]['conf']:
                position_detections[position_key] = detection

        # 填充棋盘布局
        for (row, col), detection in position_detections.items():
            cls = detection['cls']
            chess_layout[row][col] = self.class_names[cls]

            # 根据半区类型确定坐标标记方式
            coord_key = f"{9-row}{8-col}" if half_board == 'red' else f"{row}{col}"
            points_center[coord_key] = detection['center']

        return chess_layout, results, points_center


    def detect_objects_with_height(self, image, depth_frame=None, conf_threshold=0.5, iou_threshold=0.4, mat=None):
        """
        检测图像中的物体并获取其位置和高度信息，不涉及棋盘逻辑

        :param image: 输入图像
        :param depth_frame: 深度帧数据（可选）
        :param conf_threshold: 置信度阈值
        :param iou_threshold: IOU阈值
        :return: 物体检测结果列表，包含类别、边界框坐标和高度信息
        """
        if image is None:
            return [], None

        # 执行检测
        results = self._run_inference(None, image, conf_threshold, iou_threshold)

        # 获取检测结果
        boxes = results[0].boxes if hasattr(results, '__len__') and hasattr(results[0], 'boxes') else None

        if boxes is None or len(boxes) == 0:
            return [], results

        # 提取所有检测框信息
        detections = []
        for box in boxes:
            # 获取边界框坐标和类别
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())

            # 计算物体中心点
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # 创建物体信息字典
            detection = {
                'box': box,
                'class_id': cls,
                'class_name': self.class_names[cls] if cls < len(self.class_names) else f"unknown_{cls}",
                'bbox': (x1, y1, x2, y2),
                'center': (center_x, center_y),
                'confidence': conf
            }

            # 获取高度信息（如果提供了深度帧）
            if depth_frame is not None:
                try:
                    # 获取该点的深度值
                    if mat is not None:
                        x, y = apply_perspective_correction(mat, center_x, center_y)
                    else:
                        x, y = center_x, center_y

                    depth_value = depth_frame.get_distance(x, y)
                    if depth_value == 0:
                        depth_value = depth_frame.get_distance(x+5, y+5)

                    depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                    camera_xyz = pyrealsense2.rs2_deproject_pixel_to_point(
                        depth_intrinsics,
                        [float(center_x), float(center_y)],
                        depth_value
                    )
                    camera_xyz = np.round(np.array(camera_xyz), 3)
                    detection['height'] = camera_xyz[2]
                except Exception as e:
                    print(f"Error processing depth information: {e}")
                    detection['height'] = None
            else:
                detection['height'] = None

            detections.append(detection)

        # 按置信度排序
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        # 去重逻辑：保留置信度最高的检测结果
        filtered_detections = []
        used_boxes = []

        for detection in detections:
            current_box = detection['bbox']
            is_overlap = any(
                self._calculate_iou(
                    [current_box[0], current_box[1], current_box[2], current_box[3]],
                    [used_box[0], used_box[1], used_box[2], used_box[3]]
                ) > iou_threshold for used_box in used_boxes
            )

            # 如果不重叠，则保留该检测结果
            if not is_overlap:
                filtered_detections.append(detection)
                used_boxes.append(current_box)

        return filtered_detections, results



# 在主程序中使用
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(dir,'runs/detect/chess_piece_detection_separate/weights/best.pt'),
                        help='模型路径 (.pt 或 .trt/.engine 或 .onnx)')
    parser.add_argument('--convert_to', default='onnx', action='store_true',
                        help='将.pt模型转换为TensorRT/onnx格式')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸')
    parser.add_argument('--conf_threshold', type=float, default=0.45,
                        help='置信度阈值')
    parser.add_argument('--iou_threshold', type=float, default=0.25,
                        help='IOU阈值')

    args = parser.parse_args()

    # # 划分数据集
    # split_dataset(
    #     images_dir="E:/现有文件/工作/工程/新人工智能实训室/代码/Embodied/src/cchessYolo/dataset/images",
    #     output_dir="E:/现有文件/工作/工程/新人工智能实训室/代码/Embodied/src/cchessYolo/dataset",
    #     train_ratio=0.8
    # )

    # 创建检测器实例
    detector = ChessPieceDetectorSeparate(args.model_path)
    detector.train()
    #
    # # 如果需要转换模型为TensorRT
    # if args.convert_to=='trt' and args.model_path.endswith('.pt'):
    #     trt_model_path = detector.convert_to_trt(imgsz=args.imgsz)
    #     if trt_model_path:
    #         print(f"✅ 模型转换成功: {trt_model_path}")
    #         detector = ChessPieceDetectorSeparate(trt_model_path)
    #     else:
    #         print("❌ 模型转换失败")
    # #
    # 如果需要转换模型为ONNX
    if args.convert_to=='onnx' and args.model_path.endswith('.pt'):
        onnx_model_path = detector.convert_to_onnx(imgsz=args.imgsz)
        if onnx_model_path:
            print(f"✅ 模型转换成功: {onnx_model_path}")
            detector = ChessPieceDetectorSeparate(onnx_model_path)
        else:
            print("❌ 模型转换失败")

    # # 执行检测
    # detector.detect(
    #     "RS_20250913_114917.jpg",
    #     conf_threshold=args.conf_threshold,
    #     iou_threshold=args.iou_threshold,
    #     save_path="result_with_keypoints.jpg"
    # )
