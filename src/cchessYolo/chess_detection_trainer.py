import argparse
import shutil

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


    def _load_trt_model(self, engine_path):
        """
        加载TensorRT引擎模型
        """
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT库未安装，无法加载TensorRT模型")

        try:
            # 创建TensorRT运行时
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, 'rb') as f:
                runtime = trt.Runtime(TRT_LOGGER)
                self.engine = runtime.deserialize_cuda_engine(f.read())

            # 创建执行上下文
            self.context = self.engine.create_execution_context()

            # 分配GPU缓冲区
            self.inputs = []
            self.outputs = []
            self.bindings = []

            for binding in self.engine:
                binding_idx = self.engine.get_binding_index(binding)
                if binding_idx == -1:
                    print(f"❌ 未找到绑定: {binding}")
                    continue

                # 获取绑定维度
                dims = self.engine.get_binding_shape(binding_idx)
                size = trt.volume(dims) * self.engine.max_batch_size * np.dtype(np.float32).itemsize

                # 分配GPU内存
                gpu_mem = cuda.mem_alloc(size)
                self.bindings.append(int(gpu_mem))

                if self.engine.binding_is_input(binding_idx):
                    self.inputs.append({'mem': gpu_mem, 'shape': dims, 'name': binding})
                else:
                    self.outputs.append({'mem': gpu_mem, 'shape': dims, 'name': binding})

            print("✅ TensorRT模型加载成功")
            return None  # TRT模型不需要保存为self.model
        except Exception as e:
            print(f"⚠️  TensorRT模型加载失败: {e}")
            print("🔄 回退到标准PyTorch模型")
            self.is_trt_model = False
            model = YOLO(self.model_path.replace('.trt', '.pt').replace('.engine', '.pt'))
            model.to(self.device)
            return model

    def _load_onnx_model(self, onnx_path):
        """
        加载ONNX模型
        """
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime未安装，无法加载ONNX模型")

        try:
            # 根据CUDA可用性选择执行提供者
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
            print("✅ ONNX模型加载成功")
            return None
        except Exception as e:
            print(f"⚠️  ONNX模型加载失败: {e}")
            print("🔄 回退到标准PyTorch模型")
            self.is_onnx_model = False
            model = YOLO(onnx_path.replace('.onnx', '.pt'))
            model.to(self.device)
            return model

    def _trt_inference(self, image):
        """
        使用TensorRT引擎执行推理
        """
        try:
            # 预处理图像
            input_shape = self.inputs[0]['shape']
            processed_img = self._preprocess_image(image, input_shape)

            # 将输入数据复制到GPU
            cuda.memcpy_htod(self.inputs[0]['mem'], processed_img)

            # 执行推理
            self.context.execute_v2(bindings=self.bindings)

            # 从GPU获取输出
            results = []
            for output in self.outputs:
                output_data = np.empty(output['shape'], dtype=np.float32)
                cuda.memcpy_dtoh(output_data, output['mem'])
                results.append(output_data)

            # 后处理结果（这里需要根据具体模型输出格式进行调整）
            return self._postprocess_results(results, image.shape)

        except Exception as e:
            print(f"⚠️  TensorRT推理失败: {e}")
            # 回退到标准YOLO推理
            model = YOLO(self.model_path.replace('.trt', '.pt').replace('.engine', '.pt'))
            model.to(self.device)
            return model(image)

    def _onnx_inference(self, image):
        """
        使用ONNX模型执行推理
        """
        try:
            # 预处理图像
            input_name = self.onnx_session.get_inputs()[0].name
            input_shape = self.onnx_session.get_inputs()[0].shape

            # 如果是动态输入形状，使用640作为默认尺寸
            if isinstance(input_shape[2], str) or isinstance(input_shape[3], str):
                input_shape = [input_shape[0], input_shape[1], 640, 640]

            processed_img = self._preprocess_image(image, input_shape)

            # 执行推理
            outputs = self.onnx_session.run(None, {input_name: processed_img.astype(np.float32)})

            # 后处理结果
            return self._postprocess_onnx_results(outputs, image.shape)

        except Exception as e:
            print(f"⚠️  ONNX推理失败: {e}")
            # 回退到标准YOLO推理
            model = YOLO(self.model_path.replace('.onnx', '.pt'))
            model.to(self.device)
            return model(image)

    def _preprocess_image(self, image, input_shape):
        """
        预处理图像以适配模型输入
        """
        # 调整图像大小
        resized = cv2.resize(image, (input_shape[2], input_shape[1]))

        # 转换颜色空间 (BGR to RGB)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 归一化
        normalized = rgb.astype(np.float32) / 255.0

        # 转换为CHW格式
        chw = np.transpose(normalized, (2, 0, 1))

        # 添加批次维度
        batched = np.expand_dims(chw, axis=0)

        # 转换为连续内存布局
        return np.ascontiguousarray(batched)

    def _postprocess_results(self, trt_outputs, original_shape):
        """
        后处理TensorRT输出结果
        注意：这个实现需要根据具体模型输出格式进行调整
        """
        # 这里是一个简化的示例实现
        # 实际应用中需要根据模型的具体输出格式进行解析
        # 为保持兼容性，我们暂时回退到标准YOLO模型
        print("⚠️  TensorRT后处理未完全实现，回退到标准模型")
        standard_model = YOLO(self.model_path.replace('.trt', '.pt').replace('.engine', '.pt'))
        standard_model.to(self.device)
        return standard_model(np.zeros(original_shape, dtype=np.uint8))

    def _postprocess_onnx_results(self, onnx_outputs, original_shape):
        """
        后处理ONNX输出结果
        """
        # 这里需要根据具体的ONNX模型输出格式进行调整
        # 为保持兼容性，暂时回退到标准YOLO模型
        print("⚠️  ONNX后处理未完全实现，回退到标准模型")
        standard_model = YOLO(self.model_path.replace('.onnx', '.pt'))
        standard_model.to(self.device)
        return standard_model(np.zeros(original_shape, dtype=np.uint8))

    def convert_to_trt(self, output_path=None, imgsz=640, half=True):
        """
        将YOLO模型转换为TensorRT格式

        Args:
            output_path: 输出TensorRT模型路径
            imgsz: 输入图像尺寸
            half: 是否使用FP16精度
        """
        if not self.model_path.endswith('.pt'):
            print("❌  只有PyTorch模型(.pt)可以转换为TensorRT格式")
            return None

        if output_path is None:
            model_name = Path(self.model_path).stem
            output_path = f"{model_name}.trt"

        try:
            # 使用ultralytics提供的导出功能
            model = YOLO(self.model_path)

            # 导出为TensorRT格式
            model.export(
                format='engine',
                imgsz=imgsz,
                half=half,  # 使用FP16精度
                device=0 if torch.cuda.is_available() else None,  # 使用GPU 0或CPU
                verbose=True
            )

            # 重命名生成的文件
            generated_path = self.model_path.replace('.pt', '.engine')
            if os.path.exists(generated_path):
                shutil.move(generated_path, output_path)

            print(f"✅ 模型已成功转换为TensorRT格式: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ 模型转换失败: {e}")
            import traceback
            traceback.print_exc()
            return None

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
            model_name = Path(self.model_path).stem
            output_path = f"{model_name}.onnx"

        try:
            # 使用ultralytics提供的导出功能
            model = YOLO(self.model_path)

            # 导出为ONNX格式
            model.export(
                format='onnx',
                imgsz=imgsz,
                dynamic=dynamic,  # 是否支持动态输入尺寸
                simplify=True,    # 简化模型
                device=0 if torch.cuda.is_available() else 'cpu'
            )

            # 重命名生成的文件
            generated_path = self.model_path.replace('.pt', '.onnx')
            if os.path.exists(generated_path):
                os.rename(generated_path, output_path)

            print(f"✅ 模型已成功转换为ONNX格式: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ 模型转换失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def train(self, data_yaml='yaml/data.yaml', epochs=300, imgsz=640):
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

    def _run_inference(self, image, conf_threshold=0.5, iou_threshold=0.25):
        """
        根据模型类型执行推理
        """
        if self.is_trt_model and TENSORRT_AVAILABLE:
            return self._trt_inference(image)
        elif self.is_onnx_model and ONNX_AVAILABLE:
            return self._onnx_inference(image)
        else:
            return self.model(image, conf=conf_threshold, iou=iou_threshold)

    def detect(self, image_path, conf_threshold=0.5, iou_threshold=0.8, save_path='result.jpg'):
        """
        检测图像中的棋子并保存结果图片
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
        results = self._run_inference(image, conf_threshold, iou_threshold)

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

    def save_image_with_keypoints(self, image_path, output_path, half_board="red", conf_threshold=0.5, iou_threshold=0.4):
        """
        检测图像中的关键点并保存带关键点的图像

        :param image_path: 输入图像路径
        :param output_path: 输出图像保存路径
        :param half_board: 半区类型 ("red" 或 "black")
        :param conf_threshold: 置信度阈值
        :param iou_threshold: IOU阈值
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 复制图像用于绘制
        img_vis = image.copy()

        try:
            # 导入参数文件中的固定坐标
            from parameters import CHESS_BOARD_R, CHESS_BOARD_B

            # 根据half_board参数选择使用哪一组固定坐标
            if half_board == "red":
                # 使用红方半盘的四个角点坐标
                fixed_points = CHESS_BOARD_R
            else:  # black
                # 使用黑方半盘的四个角点坐标
                fixed_points = CHESS_BOARD_B

            # 绘制固定的关键点
            for i, (x, y) in enumerate(fixed_points):
                # 绘制关键点（使用不同颜色区分）
                cv2.circle(img_vis, (int(x), int(y)), 8, (0, 255, 0), -1)  # 绿色圆点
                # 添加关键点编号
                cv2.putText(img_vis, str(i), (int(x)+10, int(y)+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 绘制棋盘边界框
            pts = np.array(fixed_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img_vis, [pts], True, (255, 0, 0), 2)

        except ImportError:
            print("⚠️  无法导入参数文件，跳过固定关键点绘制")

        # 保存结果图像
        cv2.imwrite(output_path, img_vis)
        print(f"✅ 带关键点的图像已保存至: {output_path}")
    def extract_chessboard_layout_with_height(self, image, chess_points, half_board="red",
                                              conf_threshold=0.5, iou_threshold=0.4):
        """
        从图像中提取棋盘布局

        :param image: 输入图像
        :param half_board: 半区类型 ("red" 或 "black")
        :param conf_threshold: 置信度阈值
        :param iou_threshold: IOU阈值
        :return: 棋盘布局矩阵和检测结果
        """
        if image is None:
            # 如果没有图像，返回空棋盘
            empty_layout = [['.' for _ in range(9)] for _ in range(5)]
            return empty_layout, None, {}

        # 执行检测
        results = self._run_inference(image, conf_threshold, iou_threshold)

        # 初始化5x9棋盘（半个棋盘）
        chess_layout = [['.' for _ in range(9)] for _ in range(5)]
        points_center = {}
        # 存储棋子高度信息
        piece_heights = {}

        # 获取检测结果
        boxes = results[0].boxes if hasattr(results, '__len__') and hasattr(results[0], 'boxes') else None

        if boxes is not None and len(boxes) > 0:
            # 提取所有检测框信息用于去重
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
                if half_board == 'red':
                    chess_pos = camera_to_chess_position(center_x, center_y, chess_points)
                else:
                    chess_pos = camera_to_chess_position(center_x, center_y, chess_points)

                if chess_pos is not None:
                    detections.append({
                        'box': box,
                        'chess_pos': chess_pos,
                        'conf': conf,
                        'cls': cls,
                        'center': (center_x, center_y),
                    })

            # 按照棋盘位置对检测结果进行分组
            position_detections = {}
            for detection in detections:
                row, col = detection['chess_pos']
                position_key = (row, col)

                if position_key not in position_detections:
                    position_detections[position_key] = []
                position_detections[position_key].append(detection)

            # 对每个位置的多个检测结果进行去重，保留置信度最高的
            filtered_detections = []
            for position_key, detections_list in position_detections.items():
                # 按置信度排序，保留最高的
                best_detection = max(detections_list, key=lambda x: x['conf'])
                filtered_detections.append(best_detection)

            # 遍历去重后的检测结果
            for detection in filtered_detections:
                row, col = detection['chess_pos']
                cls = detection['cls']
                # 将检测到的棋子放入棋盘布局
                chess_layout[row][col] = self.class_names[cls]
                if half_board == 'red':
                    points_center[f"{9-row}{8-col}"] = detection['center']
                else:
                    points_center[f"{row}{col}"] = detection['center']

        return chess_layout, results,points_center

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
            return []

        # 执行检测
        results = self._run_inference(image, conf_threshold, iou_threshold)

        # 存储检测到的物体信息
        objects_info = []

        # 获取检测结果
        boxes = results[0].boxes if hasattr(results, '__len__') and hasattr(results[0], 'boxes') else None

        if boxes is not None and len(boxes) > 0:
            # 遍历每个检测到的物体
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
                        # detection['center'] = (camera_xyz[0], camera_xyz[1])
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
                is_overlap = False

                # 检查当前检测框是否与已保留的检测框重叠
                for used_box in used_boxes:
                    iou = self._calculate_iou(
                        [current_box[0], current_box[1], current_box[2], current_box[3]],
                        [used_box[0], used_box[1], used_box[2], used_box[3]]
                    )
                    if iou > iou_threshold:
                        is_overlap = True
                        break

                # 如果不重叠，则保留该检测结果
                if not is_overlap:
                    filtered_detections.append(detection)
                    used_boxes.append(current_box)

            objects_info = filtered_detections

        return objects_info



# 在主程序中使用
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='runs/detect/chess_piece_detection_separate7/weights/best.pt',
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

    # 创建检测器实例
    detector = ChessPieceDetectorSeparate()
    detector.train()

    # # 如果需要转换模型为TensorRT
    # if args.convert_to=='trt' and args.model_path.endswith('.pt'):
    #     trt_model_path = detector.convert_to_trt(imgsz=args.imgsz)
    #     if trt_model_path:
    #         print(f"✅ 模型转换成功: {trt_model_path}")
    #         detector = ChessPieceDetectorSeparate(trt_model_path)
    #     else:
    #         print("❌ 模型转换失败")
    #
    # # 如果需要转换模型为ONNX
    # elif args.convert_to=='onnx' and args.model_path.endswith('.pt'):
    #     onnx_model_path = detector.convert_to_onnx(imgsz=args.imgsz)
    #     if onnx_model_path:
    #         print(f"✅ 模型转换成功: {onnx_model_path}")
    #         detector = ChessPieceDetectorSeparate(onnx_model_path)
    #     else:
    #         print("❌ 模型转换失败")
    #
    # # 执行检测
    # detector.detect(
    #     "data/images/val/RS_20250730_111927.jpg",
    #     conf_threshold=args.conf_threshold,
    #     iou_threshold=args.iou_threshold,
    #     save_path="result_with_keypoints.jpg"
    # )
