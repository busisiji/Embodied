# train_yolo.py
import time

from ultralytics import YOLO
import cv2
import numpy as np

def train_yolo_model():
    """
    使用YOLOv8训练模型识别四种几何形状
    """
    # 加载预训练的YOLOv8s模型
    model = YOLO('yolo11n.pt')

    # 训练配置
    results = model.train(
        data='yaml/data_shape.yaml',  # 数据集配置文件路径
        epochs=300,                   # 训练轮数
        imgsz=640,                    # 输入图像尺寸
        batch=16,                     # 批次大小
        name='cchess_shape_model'     # 保存的模型名称
    )

    return results

def validate_model():
    """
    验证训练好的模型
    """
    # 加载训练好的模型
    model = YOLO('runs/detect/cchess_shape_model/weights/best.pt')

    # 在验证集上评估模型，设置IoU阈值为0.9
    metrics = model.val(iou=0.9)

    return metrics

def detect_shapes(image_path):
    """
    使用训练好的模型检测图像中的几何形状

    Args:
        image_path (str): 图像路径
    """
    # 加载训练好的模型
    model = YOLO('runs/detect/cchess_shape_model/weights/best.pt')

    # 进行检测，设置IoU阈值为0.9
    results = model(image_path, iou=0.24, conf=0.8)

    # 手动处理显示，显示置信度但不显示类别标签
    for result in results:
        # 获取原始图像
        img = result.orig_img.copy()

        # 获取检测框坐标
        boxes = result.boxes

        if boxes is not None:
            # 为每个检测框绘制边界框和置信度
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                # 绘制边界框
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # # 显示置信度但不显示类别标签
                # cv2.putText(img, f'{confidence:.2f}', (x1, y1 - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 2)

        # 显示结果
        cv2.imshow('Detection Result', img)
        cv2.waitKey(0)  # 等待按键
        cv2.destroyAllWindows()

        # 保存检测结果
        cv2.imwrite('result.jpg', img)

    return results


def validate_trt_model(image_path):
    """
    使用TensorRT模型验证检测结果

    Args:
        image_path (str): 图像路径
    """
    # 注意：直接使用PyTorch无法加载TensorRT模型
    # 需要使用TensorRT Python API或通过YOLO接口加载
    try:
        # 方法1: 通过YOLO接口加载TensorRT模型
        model = YOLO('runs/detect/cchess_shape_model/weights/best.engine')
        # 指定使用TensorRT推理（如果已导出）
        results = model(image_path, imgsz=640, device='cuda')  # 如果有GPU

        # 处理结果
        for result in results:
            img = result.orig_img.copy()
            boxes = result.boxes

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())

                    # 绘制边界框
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 显示结果
            cv2.imshow('TRT Detection Result', img)
            cv2.waitKey(0)
            cv2.imwrite('trt_result.jpg', img)
            cv2.destroyAllWindows()

        return results
    except Exception as e:
        print(f"TensorRT模型验证出错: {e}")
        return None

if __name__ == '__main__':
    # print("开始训练YOLO模型...")
    # train_results = train_yolo_model()
    # print("模型训练完成!")
    #
    # print("验证模型性能...")
    # val_metrics = validate_model()
    # print("验证完成!")

    # # 如果需要检测示例图像，取消下面的注释
    st =time.time()
    validate_trt_model('test.png')
    print("检测完成!",time.time()-st)