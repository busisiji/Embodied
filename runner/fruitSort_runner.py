# file: /media/jetson/KESU/V10.14/Embodied/runner/fruitSort_runner.py

import asyncio
import cv2
import time
import threading
from manager.manager import system_manager
from src.cchessYolo.fruit_yolo_obb_trainer import FruitOBBTrainer

class FruitSortingApp:
    """
    水果分拣应用程序
    """

    def __init__(self):
        self.system_manager = system_manager
        # 初始化水果检测器
        model_path = 'E:/现有文件/工作/工程/新人工智能实训室/代码/V10.10/Embodied/src/cchessYolo/fruitYolo/runs/obb/fruit_obb_detection4/weights/best.pt'
        self.fruit_detector = FruitOBBTrainer(model_path=model_path)
        # 水果类别映射
        self.fruit_classes = {
            "黄梨": "peach",
            "黄苹果": "y-apple",
            "绿苹果": "g-apple",
            "绿葡萄": "g-grape",
            "番茄": "persimmon",
            "苹果": "apple",
            "橙子": "orange",
            "柑橘": "citrus",
            "红苹果": "apple"  # 红苹果也映射到apple类别
        }
        # 提取所有水果关键字
        self.fruit_keywords = list(self.fruit_classes.keys())

        # 注册关键字和回调函数
        if self.system_manager.speech_recognizer:
            self.system_manager.add_keywords(self.fruit_keywords)
            self.system_manager.register_keyword_callback("水果分拣", self._speech_callback)

        # 任务控制相关
        self.sorting_active = False
        self.sorting_paused = False
        self.sorting_thread = None
        self._stop_event = threading.Event()

        # 注册IO回调函数
        self.system_manager.register_io_callback("start", self._handle_start_button)
        self.system_manager.register_io_callback("stop", self._handle_stop_button)
        self.system_manager.register_io_callback("reset", self._handle_reset_button)

    async def start_sorting(self):
        """开始分拣"""
        # 确保系统已初始化
        if not hasattr(self.system_manager, '_initialized') or not self.system_manager._initialized:
            self.system_manager.initialize()

        await self.system_manager.speak_async("水果分拣系统已启动")
        print("水果分拣系统已启动")

    def _speech_callback(self, keywords, full_text):
        """语音识别回调函数"""
        print(f"识别到关键词: {keywords}, 完整文本: {full_text}")

        # 处理水果分拣相关命令
        if "开始" in keywords:
            asyncio.run(self.system_manager.speak_async("开始分拣"))
            self._start_sorting_process()
        elif "停止" in keywords:
            asyncio.run(self.system_manager.speak_async("停止分拣"))
            self._stop_sorting_process()
        else:
            # 检查是否提到了具体的水果
            for chinese_name, english_name in self.fruit_classes.items():
                if chinese_name in keywords:
                    self.detect_specific_fruit(english_name, chinese_name)
                    break

    def _handle_start_button(self):
        """处理启动按钮事件"""
        print("🎮 检测到启动按钮按下")
        if self.sorting_paused:
            self._resume_sorting_process()
        elif not self.sorting_active:
            self._start_sorting_process()

    def _handle_stop_button(self):
        """处理停止按钮事件"""
        print("⏹️ 检测到停止按钮按下")
        if self.sorting_active:
            self._pause_sorting_process()

    def _handle_reset_button(self):
        """处理复位按钮事件"""
        print("🔄 检测到复位按钮按下")
        self._stop_sorting_process()
        # 可以添加其他复位逻辑

    def _start_sorting_process(self):
        """开始分拣过程"""
        if self.sorting_active:
            print("⚠️ 分拣过程已在运行")
            return

        self.sorting_active = True
        self.sorting_paused = False
        self._stop_event.clear()

        self.sorting_thread = threading.Thread(target=self._sorting_worker, daemon=True)
        self.sorting_thread.start()

        asyncio.run(self.system_manager.speak_async("开始分拣过程"))

    def _pause_sorting_process(self):
        """暂停分拣过程"""
        if not self.sorting_active:
            print("⚠️ 分拣过程未在运行")
            return

        self.sorting_paused = True
        asyncio.run(self.system_manager.speak_async("分拣过程已暂停"))

    def _resume_sorting_process(self):
        """恢复分拣过程"""
        if not self.sorting_active or not self.sorting_paused:
            print("⚠️ 分拣过程未处于暂停状态")
            return

        self.sorting_paused = False
        asyncio.run(self.system_manager.speak_async("分拣过程已恢复"))

    def _stop_sorting_process(self):
        """停止分拣过程"""
        if not self.sorting_active:
            print("⚠️ 分拣过程未在运行")
            return

        self.sorting_active = False
        self.sorting_paused = False
        self._stop_event.set()

        asyncio.run(self.system_manager.speak_async("分拣过程已停止"))

    def _sorting_worker(self):
        """分拣工作线程"""
        print("📦 分拣工作线程已启动")

        while self.sorting_active and not self._stop_event.is_set():
            # 如果暂停，则等待
            while self.sorting_paused and not self._stop_event.is_set():
                time.sleep(0.1)

            if self._stop_event.is_set():
                break

            try:
                # 执行分拣逻辑
                self._perform_sorting_step()
                time.sleep(0.1)  # 控制处理频率
            except Exception as e:
                print(f"❌ 分拣过程中出错: {e}")
                asyncio.run(self.system_manager.speak_async("分拣过程中出现错误"))

        print("📦 分拣工作线程已停止")

    def _perform_sorting_step(self):
        """执行单步分拣操作"""
        # 这里实现具体的分拣逻辑
        # 例如：获取图像、检测水果、控制机械臂移动等
        print("🔄 执行分拣步骤...")

        # 模拟耗时操作
        for i in range(100):  # 模拟1秒的耗时操作
            if self._stop_event.is_set() or self.sorting_paused:
                break
            time.sleep(0.01)

    def detect_fruit(self, color=None, shape=None):
        """检测水果"""
        # 使用相机获取图像并分析水果
        frame = self.system_manager.get_camera_frame()
        if frame is not None:
            # 分析水果的颜色和形状
            pass
        return None

    def detect_specific_fruit(self, fruit_name, chinese_name):
        """
        检测指定类型的水果

        Args:
            fruit_name (str): 水果英文名称
            chinese_name (str): 水果中文名称
        """
        # 获取相机帧
        frame = self.system_manager.get_camera_frame()
        if frame is None:
            print("无法获取相机帧")
            asyncio.run(self.system_manager.speak_async(f"无法获取相机帧，无法检测{chinese_name}"))
            return None

        # 保存临时图像文件用于检测
        temp_image_path = "temp_frame.jpg"
        cv2.imwrite(temp_image_path, frame)

        # 使用训练好的模型进行预测
        results, detections = self.fruit_detector.predict(
            source=temp_image_path,
            conf=0.25,
            iou=0.45,
            save=False  # 不保存结果
        )

        # 筛选出指定类型的水果
        target_detections = [
            det for det in detections
            if det["class_name"] == fruit_name
        ]

        if not target_detections:
            print(f"未检测到 {chinese_name}")
            asyncio.run(self.system_manager.speak_async(f"未检测到{chinese_name}"))
            return []

        # 输出检测到的水果信息
        print(f"\n检测到 {len(target_detections)} 个 {chinese_name}:")
        speak_text = f"检测到{len(target_detections)}个{chinese_name}"
        asyncio.run(self.system_manager.speak_async(speak_text))

        for i, detection in enumerate(target_detections):
            print(f"  {chinese_name} {i+1}:")
            print(f"    置信度: {detection['confidence']}")
            print(f"    中心点: ({detection['center'][0]}, {detection['center'][1]})")

            if 'bbox_vertices' in detection and detection['bbox_vertices']:
                print(f"    边界框顶点: {detection['bbox_vertices']}")
            elif 'bbox' in detection and detection['bbox']:
                print(f"    边界框: {detection['bbox']}")

            print(f"    角度: {detection['angle']}°")

        return target_detections
