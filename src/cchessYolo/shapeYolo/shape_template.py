# template_matching_pil.py
import numpy as np
import os
from typing import List, Tuple
import json
from PIL import Image, ImageDraw, ImageFont
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class TemplateMatcher:
    """
    基于PIL和numpy的模板匹配类，用于在原始图像中查找模板位置（不使用OpenCV）
    支持多模板、多旋转角度、多线程处理
    """

    def __init__(self):
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.templates_dir = os.path.join(self.dir, "data/images/train/images")
        self.source_images_dir = os.path.join(self.dir, "data/images")
        self.results_dir = os.path.join(self.dir, "data/matching_results")
        os.makedirs(self.results_dir, exist_ok=True)

    def load_template(self, template_path: str) -> np.ndarray:
        """
        加载模板图像

        Args:
            template_path: 模板图像路径

        Returns:
            模板图像数组
        """
        try:
            template = Image.open(template_path)
            # 转换为numpy数组
            template_array = np.array(template)
            return template_array
        except Exception as e:
            raise FileNotFoundError(f"无法加载模板图像: {template_path}, 错误: {e}")

    def load_source_image(self, image_path: str) -> np.ndarray:
        """
        加载原始图像

        Args:
            image_path: 原始图像路径

        Returns:
            原始图像数组
        """
        try:
            image = Image.open(image_path)
            # 转换为numpy数组
            image_array = np.array(image)
            return image_array
        except Exception as e:
            raise FileNotFoundError(f"无法加载原始图像: {image_path}, 错误: {e}")

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        标准化图像数据到0-1范围

        Args:
            image: 输入图像数组

        Returns:
            标准化后的图像数组
        """
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image.astype(np.float32)

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        旋转图像

        Args:
            image: 输入图像数组
            angle: 旋转角度（度）

        Returns:
            旋转后的图像数组
        """
        pil_image = Image.fromarray(np.uint8(image * 255) if image.dtype == np.float32 else image)
        # 使用 bilinear 插值并保持原始尺寸
        rotated = pil_image.rotate(angle, expand=False, resample=Image.BILINEAR)
        return np.array(rotated)

    def match_template_with_rotation(self, source_image: np.ndarray, template: np.ndarray,
                                   threshold: float = 0.8, rotations: List[float] = None) -> List[Tuple[int, int, int, int, float, float]]:
        """
        在原始图像中匹配模板（使用归一化互相关），支持旋转匹配

        Args:
            source_image: 原始图像
            template: 模板图像
            threshold: 匹配阈值
            rotations: 旋转角度列表

        Returns:
            匹配结果列表，每个元素为(x, y, w, h, score, angle)
        """
        if rotations is None:
            rotations = [0]  # 默认不旋转

        all_matches = []

        # 标准化源图像
        source_norm = self.normalize_image(source_image)

        # 如果是彩色图像，转换为灰度图像
        if len(source_norm.shape) == 3:
            source_gray = np.dot(source_norm[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            source_gray = source_norm

        source_h, source_w = source_gray.shape[:2]

        for angle in rotations:
            # 旋转模板
            if angle == 0:
                template_rotated = template
            else:
                template_rotated = self.rotate_image(template, angle)

            # 标准化旋转后的模板
            template_norm = self.normalize_image(template_rotated)

            # 如果是彩色图像，转换为灰度图像
            if len(template_norm.shape) == 3:
                template_gray = np.dot(template_norm[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                template_gray = template_norm

            # 模板尺寸
            template_h, template_w = template_gray.shape[:2]

            # 如果模板大于源图像，则无法匹配
            if template_h > source_h or template_w > source_w:
                continue

            # 计算模板的均值和标准差
            template_mean = np.mean(template_gray)
            template_std = np.std(template_gray)

            # 如果模板标准差为0（全相同像素），则特殊处理
            if template_std == 0:
                continue

            # 归一化模板
            template_normalized = (template_gray - template_mean) / template_std

            # 滑动窗口进行匹配
            for y in range(source_h - template_h + 1):
                for x in range(source_w - template_w + 1):
                    # 提取源图像中的区域
                    region = source_gray[y:y+template_h, x:x+template_w]

                    # 计算区域的均值和标准差
                    region_mean = np.mean(region)
                    region_std = np.std(region)

                    # 避免除零错误
                    if region_std == 0:
                        continue

                    # 归一化区域
                    region_normalized = (region - region_mean) / region_std

                    # 计算归一化互相关(NCC)
                    ncc = np.sum(template_normalized * region_normalized) / (template_h * template_w)

                    # 如果匹配度超过阈值，则记录
                    if ncc >= threshold:
                        all_matches.append((x, y, template_w, template_h, ncc, angle))

        return all_matches

    def non_max_suppression_rotated(self, boxes: List[Tuple], overlap_threshold: float) -> List[Tuple]:
        """
        非极大值抑制，去除重复的匹配框（支持旋转角度）

        Args:
            boxes: 匹配框列表 [(x, y, w, h, score, angle), ...]
            overlap_threshold: 重叠阈值

        Returns:
            抑制后的匹配框列表
        """
        if len(boxes) == 0:
            return []

        # 提取坐标和分数
        boxes_array = np.array(boxes)
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 0] + boxes_array[:, 2]
        y2 = boxes_array[:, 1] + boxes_array[:, 3]
        scores = boxes_array[:, 4]

        # 计算面积
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # 按分数排序（降序）
        indices = np.argsort(scores)[::-1]
        keep = []

        while len(indices) > 0:
            # 保留当前最高分数的框
            current = indices[0]
            keep.append(current)

            if len(indices) == 1:
                break

            # 计算与其他框的重叠
            current_box = boxes_array[current]
            other_boxes = boxes_array[indices[1:]]

            # 计算交集
            xx1 = np.maximum(current_box[0], other_boxes[:, 0])
            yy1 = np.maximum(current_box[1], other_boxes[:, 1])
            xx2 = np.minimum(current_box[0] + current_box[2], other_boxes[:, 0] + other_boxes[:, 2])
            yy2 = np.minimum(current_box[1] + current_box[3], other_boxes[:, 1] + other_boxes[:, 3])

            # 计算交集面积
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            intersection = w * h

            # 计算IoU
            iou = intersection / (areas[current] + other_boxes[:, 2] * other_boxes[:, 3] - intersection)

            # 保留IoU小于阈值的框
            # 修复索引越界问题
            remaining_indices = np.where(iou <= overlap_threshold)[0]
            if len(remaining_indices) > 0:
                indices = indices[1:][remaining_indices]
            else:
                indices = np.array([], dtype=int)

        return [boxes[i] for i in keep] if keep else []

    def draw_matches(self, image_array: np.ndarray, matches: List[Tuple],
                    template_name: str) -> Image.Image:
        """
        在图像上绘制匹配结果

        Args:
            image_array: 原始图像数组
            matches: 匹配结果
            template_name: 模板名称

        Returns:
            绘制后的PIL图像
        """
        # 将numpy数组转换为PIL图像
        if image_array.dtype == np.float32:
            # 如果是浮点数，先转换到0-255范围
            image_array = (image_array * 255).astype(np.uint8)

        result_image = Image.fromarray(image_array)
        draw = ImageDraw.Draw(result_image)

        # 尝试加载字体，如果失败则使用默认字体
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except:
            font = ImageFont.load_default()

        for (x, y, w, h, score, angle) in matches:
            # 绘制矩形框（绿色）
            draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=2)
            # 添加标签
            label = f"{template_name}: {score:.2f} ({angle}°)"
            draw.text((x, y - 15), label, fill=(0, 255, 0), font=font)

        return result_image

    def save_results(self, image: Image.Image, matches: List[Tuple],
                    output_path: str, annotations_path: str = None):
        """
        保存匹配结果图像和标注文件

        Args:
            image: 结果图像(PIL格式)
            matches: 匹配结果
            output_path: 输出图像路径
            annotations_path: 标注文件路径
        """
        # 保存图像
        image.save(output_path)

        # 保存标注文件（YOLO格式）
        if annotations_path and matches:
            width, height = image.size
            with open(annotations_path, 'w') as f:
                for (x, y, w, h, score, angle) in matches:
                    # 转换为YOLO格式（中心点坐标和宽高，相对值）
                    cx = (x + w/2) / width
                    cy = (y + h/2) / height
                    nw = w / width
                    nh = h / height
                    # 假设所有模板都属于同一类别，实际应用中需要根据模板类型确定类别索引
                    class_id = 0
                    f.write(f"{class_id} {cx} {cy} {nw} {nh}\n")

    def process_single_template(self, source_image_path: str, template_path: str,
                              output_path: str = None, annotations_path: str = None,
                              threshold: float = 0.8, rotations: List[float] = None) -> List[Tuple]:
        """
        处理单个模板匹配任务

        Args:
            source_image_path: 原始图像路径
            template_path: 模板图像路径
            output_path: 结果图像保存路径
            annotations_path: 标注文件保存路径
            threshold: 匹配阈值
            rotations: 旋转角度列表

        Returns:
            匹配结果列表
        """
        if rotations is None:
            rotations = [0, 45, 90, 135, 180, 225, 270, 315]  # 默认旋转角度

        # 加载图像
        source_image = self.load_source_image(source_image_path)
        template = self.load_template(template_path)
        print(f"Processing template: {template_path}")

        # 获取模板名称
        template_name = os.path.basename(template_path).split('.')[0]

        # 执行匹配
        matches = self.match_template_with_rotation(source_image, template, threshold=threshold, rotations=rotations)

        # 去除重复匹配（非极大值抑制）
        if matches:
            matches = self.non_max_suppression_rotated(matches, overlap_threshold=0.5)

        if output_path:
            # 绘制结果
            result_image = self.draw_matches(source_image, matches, template_name)

            # 保存结果
            self.save_results(result_image, matches, output_path, annotations_path)

        return matches

    def _process_template_file(self, source_image: np.ndarray, template_path: str,
                              class_dir: str, threshold: float, rotations: List[float]):
        """
        处理单个模板文件的内部方法，用于多线程处理

        Args:
            source_image: 原始图像数组
            template_path: 模板文件路径
            class_dir: 类别目录名
            threshold: 匹配阈值
            rotations: 旋转角度列表

        Returns:
            (class_dir, matches) 元组
        """
        try:
            template = self.load_template(template_path)
            matches = self.match_template_with_rotation(source_image, template, threshold=threshold, rotations=rotations)
            template_file = os.path.basename(template_path)
            extended_matches = [(m[0], m[1], m[2], m[3], m[4], m[5], template_file) for m in matches]
            return class_dir, extended_matches
        except Exception as e:
            print(f"处理模板 {template_path} 时出错: {e}")
            return class_dir, []

    def batch_process(self, source_image_path: str, templates_dir: str = None,
                     output_dir: str = None, threshold: float = 0.8,
                     rotations: List[float] = None, max_workers: int = 4):
        """
        批量处理多个模板匹配（支持多线程）

        Args:
            source_image_path: 原始图像路径
            templates_dir: 模板目录路径
            output_dir: 结果保存目录
            threshold: 匹配阈值
            rotations: 旋转角度列表
            max_workers: 最大线程数
        """
        if rotations is None:
            rotations = [0, 45, 90, 135, 180, 225, 270, 315]  # 默认旋转角度

        if templates_dir is None:
            templates_dir = self.templates_dir

        if output_dir is None:
            output_dir = self.results_dir

        # 获取所有模板类别目录
        try:
            class_dirs = [d for d in os.listdir(templates_dir)
                         if os.path.isdir(os.path.join(templates_dir, d))]
        except:
            print("未找到模板目录或模板目录为空")
            return {}

        # 加载原始图像
        source_image = self.load_source_image(source_image_path)
        image_name = os.path.basename(source_image_path).split('.')[0]

        all_matches = {}

        # 使用线程池处理模板匹配
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_template = {}

            for class_dir in class_dirs:
                class_path = os.path.join(templates_dir, class_dir)
                try:
                    template_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                except:
                    continue

                # 限制每个类别最多使用3个模板以提高效率
                for template_file in template_files[:3]:
                    template_path = os.path.join(class_path, template_file)
                    future = executor.submit(
                        self._process_template_file,
                        source_image,
                        template_path,
                        class_dir,
                        threshold,
                        rotations
                    )
                    future_to_template[future] = (class_dir, template_file)

            # 收集结果
            class_matches = {}
            for future in as_completed(future_to_template):
                class_dir, matches = future.result()
                if class_dir not in class_matches:
                    class_matches[class_dir] = []
                class_matches[class_dir].extend(matches)

        # 对每个类别的所有匹配结果进行非极大值抑制
        for class_dir, matches in class_matches.items():
            if matches:
                # 提取坐标和分数
                boxes = [(x, y, w, h, score, angle) for x, y, w, h, score, angle, _ in matches]
                suppressed_boxes = self.non_max_suppression_rotated(boxes, overlap_threshold=0.3)

                # 过滤出被保留的匹配
                final_matches = []
                for box in suppressed_boxes:
                    for match in matches:
                        if (match[0] == box[0] and match[1] == box[1] and
                            match[2] == box[2] and match[3] == box[3] and
                            match[5] == box[5]):  # 也检查角度
                            final_matches.append(match)
                            break

                all_matches[class_dir] = final_matches

        # 绘制所有匹配结果
        if image_name.endswith(('.jpg', '.jpeg', '.png')):
            base_image_name = image_name
        else:
            base_image_name = image_name.split('.')[0]

        result_image = self.draw_batch_matches(source_image, all_matches)

        # 保存结果
        output_image_path = os.path.join(output_dir, f"{base_image_name}_matches.jpg")
        result_image.save(output_image_path)

        # 保存JSON格式的结果
        results_json = {
            'source_image': source_image_path,
            'matches': {}
        }

        for class_name, matches in all_matches.items():
            results_json['matches'][class_name] = [
                {
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'confidence': float(score),
                    'angle': float(angle),
                    'template': template_name
                }
                for x, y, w, h, score, angle, template_name in matches
            ]

        json_path = os.path.join(output_dir, f"{base_image_name}_matches.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, ensure_ascii=False, indent=2)

        print(f"批量处理完成，结果已保存至: {output_image_path}")
        print(f"详细结果已保存至: {json_path}")

        return all_matches

    def draw_batch_matches(self, image_array: np.ndarray,
                          all_matches: dict) -> Image.Image:
        """
        绘制批量匹配结果

        Args:
            image_array: 原始图像数组
            all_matches: 所有匹配结果

        Returns:
            绘制后的PIL图像
        """
        # 将numpy数组转换为PIL图像
        if image_array.dtype == np.float32:
            # 如果是浮点数，先转换到0-255范围
            image_array = (image_array * 255).astype(np.uint8)

        result_image = Image.fromarray(image_array)
        draw = ImageDraw.Draw(result_image)

        # 尝试加载字体，如果失败则使用默认字体
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except:
            font = ImageFont.load_default()

        # 不同类别的颜色
        colors = {
            'red_triangle': (255, 0, 0),        # 红色
            'blue_circle': (0, 0, 255),         # 蓝色
            'yellow_pentagon': (255, 255, 0),   # 黄色
            'green_rectangle': (0, 255, 0)      # 绿色
        }

        for class_name, matches in all_matches.items():
            color = colors.get(class_name, (255, 255, 255))  # 默认白色
            for (x, y, w, h, score, angle, template_name) in matches:
                # 绘制矩形框
                draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
                # 添加标签
                label = f"{class_name}: {score:.2f} ({angle}°)"
                draw.text((x, y - 15), label, fill=color, font=font)

        return result_image

def main():
    """
    主函数 - 演示模板匹配功能
    """
    matcher = TemplateMatcher()

    # 示例：处理单个图像
    try:
        source_images = [f for f in os.listdir(matcher.source_images_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and
                        not f.startswith('camera_capture_20250904_172816')]
    except:
        print("未找到原始图像目录")
        return

    if not source_images:
        print("未找到可处理的原始图像")
        return

    # 选择第一张图像进行处理
    source_image_path = os.path.join(matcher.source_images_dir, source_images[0])
    print(f"处理图像: {source_image_path}")

    # 执行批量模板匹配
    try:
        results = matcher.batch_process(source_image_path, max_workers=4)

        # 输出结果统计
        total_matches = sum(len(matches) for matches in results.values())
        print(f"\n匹配完成，共找到 {total_matches} 个匹配:")
        for class_name, matches in results.items():
            print(f"  {class_name}: {len(matches)} 个")
    except Exception as e:
        print(f"处理过程中出错: {e}")

if __name__ == "__main__":
    main()
