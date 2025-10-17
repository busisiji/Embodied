# file: /media/jetson/KESU/V10.14/Embodied/runner/fruitSort_runner.py
import argparse
import asyncio
import os
import cv2
import time
import threading
import numpy as np

from manager.manager import system_manager
from runner.handeye_runner import pixel_to_world_coordinates
from src.cchessYolo.fruit_yolo_obb_trainer import FruitOBBTrainer
from parameters import FRUIT_CAMERA, FRUIT_A_POINT, IO_QI, FRUIT_GRID_POINT

current_dir = os.path.dirname(os.path.abspath(__file__))

class FruitSortingApp:
    """
    水果分拣应用程序
    """

    def __init__(self):
        self.system_manager = system_manager
        if not self.system_manager.camera_manager or not self.system_manager.dobot_controller:
            self.system_manager.initialize()
        # 统一使用 urController 来操作机械臂
        self.urController = self.system_manager.dobot_controller
        # 初始化水果检测器
        model_path = os.path.join(current_dir, '../src/cchessYolo/runs/obb/fruit_obb_detection4/weights/best.pt')
        self.fruit_detector = FruitOBBTrainer(model_path=model_path)
        # 水果类别映射
        self.\
            fruit_classes = {
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

        self.grid_points = self._calculate_fruit_grid_points()

        # 注册关键字和回调函数
        if self.system_manager.speech_recognizer:
            self.system_manager.add_keywords(self.fruit_keywords)
            self.system_manager.register_keyword_callback("水果分拣", self._speech_callback)

        # 抓取点
        self.point_up = 350
        self.point_down = 250

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
        elif "复位" in keywords:
            asyncio.run(self.system_manager.speak_async("系统复位"))
            self._reset_system()
        else:
            # 检查是否是抓取命令
            if "抓取" in keywords and ("到" in keywords or "至" in keywords):
                self._handle_pick_and_place_command(keywords)
            else:
                # 检查是否提到了具体的水果
                for chinese_name, english_name in self.fruit_classes.items():
                    if chinese_name in keywords:
                        self.detect_specific_fruit(english_name, chinese_name)
                        break

    def _handle_pick_and_place_command(self, keywords):
        """处理抓取命令"""
        # 解析命令格式: 抓取{x}到{y}点
        command_text = "".join(keywords) if isinstance(keywords, list) else keywords

        # 提取水果名称和目标位置
        fruit_name = None
        target_position = None

        # 查找水果名称
        for chinese_name in self.fruit_classes.keys():
            if chinese_name in command_text:
                fruit_name = chinese_name
                break

        # 查找目标位置
        positions = {
            "一点": 1, "二点": 2, "三点": 3,
            "四点": 4, "五点": 5, "六点": 6,
            "七点": 7, "八点": 8, "九点": 9,
            "一": 1, "二": 2, "三": 3,
            "四": 4, "五": 5, "六": 6,
            "七": 7, "八": 8, "九": 9,
            "A点": "A", "B点": "B", "a点": "A", "b点": "B"
        }

        for pos_text, pos_value in positions.items():
            if pos_text in command_text:
                target_position = pos_value
                break

        if fruit_name and target_position:
            asyncio.run(self.system_manager.speak_async(f"开始抓取{fruit_name}到{target_position}点"))
            self._pick_and_place_fruit(fruit_name, target_position)
        else:
            asyncio.run(self.system_manager.speak_async("无法识别抓取命令，请重新表述"))

    def _pick_fruit_at_position(self, world_coords):
        """
        在指定世界坐标位置抓取水果

        Args:
            world_coords (tuple): 水果的世界坐标 (x, y)

        Returns:
            bool: 抓取成功返回True，否则返回False
        """
        try:
            # 移动到水果上方
            pick_up_position = [world_coords[0], world_coords[1], self.point_up]
            self._move_arm_to_position(pick_up_position)

            # 打开爪子
            self._gripper_action("open")

            # 下降到抓取高度
            pick_down_position = [world_coords[0], world_coords[1], self.point_down]
            self._move_arm_to_position(pick_down_position)

            # 执行抓取动作
            self._gripper_action("grab")

            # 提升到安全高度
            self._move_arm_to_position(pick_up_position)

            return True
        except Exception as e:
            print(f"抓取水果时出错: {e}")
            return False

    def _place_fruit_at_position(self, target_coord):
        """
        在指定位置放置水果

        Args:
            target_coord (tuple): 目标位置的世界坐标 (x, y)

        Returns:
            bool: 放置成功返回True，否则返回False
        """
        try:
            # 移动到目标点上方
            place_up_position = [target_coord[0], target_coord[1], self.point_up]
            self._move_arm_to_position(place_up_position)

            # 下降到放置高度
            place_down_position = [target_coord[0], target_coord[1], self.point_down]
            self._move_arm_to_position(place_down_position)

            # 执行放置动作
            self._gripper_action("open")

            # 提升到安全高度
            self._move_arm_to_position(place_up_position)

            return True
        except Exception as e:
            print(f"放置水果时出错: {e}")
            return False

    def _move_fruit_from_position_to_target(self, world_coords, target_coord):
        """
        将水果从一个位置移动到另一个位置

        Args:
            world_coords (tuple): 水果当前位置的世界坐标 (x, y)
            target_coord (tuple): 目标位置的世界坐标 (x, y)

        Returns:
            bool: 移动成功返回True，否则返回False
        """
        # 抓取水果
        if not self._pick_fruit_at_position(world_coords):
            return False

        # 放置到目标位置
        if not self._place_fruit_at_position(target_coord):
            return False

        return True

    def _pick_and_place_fruit(self, fruit_name, target_position):
        """
        抓取水果并放置到指定位置

        Args:
            fruit_name (str): 水果名称
            target_position (int or str): 目标位置(1-9或A/B)
        """
        try:
            # 0. 移动到拍照点
            self._gripper_action("grab") # 防止遮挡
            self.urController.run_point_j(FRUIT_CAMERA)

            # 1. 检测指定水果
            detections = self.detect_specific_fruit(self.fruit_classes[fruit_name], fruit_name)
            if not detections:
                asyncio.run(self.system_manager.speak_async(f"未找到{fruit_name}"))
                return False

            # 取第一个检测到的水果
            target_fruit = detections[0]
            x, y = target_fruit['center']

            # 2. 转换像素坐标到世界坐标
            world_coords = self._pixel_to_world(x, y)
            if not world_coords:
                asyncio.run(self.system_manager.speak_async("坐标转换失败"))
                return False

            # 3. 检查目标位置是否已有其他水果，如果有则先移走
            if isinstance(target_position, int) and 1 <= target_position <= 9:
                # 检查目标位置是否有其他水果，传递检测结果
                if self._is_position_occupied(target_position, detections):
                    print(f"目标位置{target_position}已有水果，先将其移走")
                    # 将目标位置的水果移到FRUIT_A_POINT
                    self._move_fruit_from_position_to_a_point(target_position)

            # 4. 移动水果到目标位置
            target_coord = self._get_target_coordinates(target_position)
            if not target_coord:
                asyncio.run(self.system_manager.speak_async("无效的目标位置"))
                return False
            target_world_coord = self._pixel_to_world(target_coord[0], target_coord[1])
            success = self._move_fruit_from_position_to_target(world_coords, target_world_coord)
            if not success:
                asyncio.run(self.system_manager.speak_async("移动水果过程中出现错误"))
                return False

            asyncio.run(self.system_manager.speak_async(f"已将{fruit_name}放置到{target_position}点"))
            return True

        except Exception as e:
            print(f"抓取过程中出错: {e}")
            asyncio.run(self.system_manager.speak_async("抓取过程中出现错误"))
            return False

    def _is_position_occupied(self, position, current_detections=None):
        """
        检查指定九宫格位置是否已被占用

        Args:
            position (int): 九宫格位置(1-9)
            current_detections (list): 当前检测到的水果列表，用于排除正在处理的水果

        Returns:
            bool: 如果位置被占用返回True，否则返回False
        """
        try:
            # 检查位置参数有效性
            if not isinstance(position, int) or position < 1 or position > 9:
                print(f"无效的位置参数: {position}")
                return False

            # 获取目标位置坐标
            if position not in self.grid_points:
                print(f"位置{position}未在网格点中定义")
                return False

            target_coord = self.grid_points[position]

            # 检测所有水果
            all_detections = self.detect_all_fruits()
            if not all_detections:
                print(f"位置{position}未被占用（未检测到任何水果）")
                return False

            # 位置判断阈值(mm)
            position_threshold = 30

            # 遍历所有检测到的水果
            for fruit_type, detections in all_detections.items():
                for detection in detections:
                    # 如果提供了当前检测结果，排除正在处理的水果
                    if current_detections and detection in current_detections:
                        continue

                    x, y = detection['center']
                    # 转换像素坐标到世界坐标
                    world_coords = self._pixel_to_world(x, y)
                    if world_coords:
                        world_x, world_y = world_coords
                        # 计算与目标位置的距离
                        distance = ((world_x - target_coord[0])**2 +
                                   (world_y - target_coord[1])**2)**0.5
                        if distance < position_threshold:
                            print(f"位置{position}已被{fruit_type}占用，距离: {distance:.2f}mm")
                            return True

            print(f"位置{position}未被占用")
            return False

        except Exception as e:
            print(f"检查位置占用状态时出错: {e}")
            # 出错时保守返回True，避免碰撞
            return True

    def _move_fruit_from_position_to_a_point(self, position):
        """
        将指定位置的水果移动到A点

        Args:
            position (int): 九宫格位置(1-9)
        """
        try:
            # 获取位置坐标
            if position not in self.grid_points:
                print(f"无效的位置: {position}")
                return False

            position_coord = self.grid_points[position]

            # 抓取水果
            pick_success = self._pick_fruit_at_position(position_coord)
            if not pick_success:
                print(f"在位置{position}抓取水果失败")
                return False

            # 移动到A点上方
            a_point_up_position = [FRUIT_A_POINT[0], FRUIT_A_POINT[1], self.point_up]
            self._move_arm_to_position(a_point_up_position)

            # 执行放置动作
            self._gripper_action("open")

            # 提升到安全高度
            self._move_arm_to_position(a_point_up_position)

            print(f"已将位置{position}的水果移至A点")
            return True

        except Exception as e:
            print(f"移动水果过程中出错: {e}")
            return False


    # 坐标转换
    def _pixel_to_world(self, pixel_x, pixel_y):
        """
        将像素坐标转换为世界坐标

        Args:
            pixel_x (float): 像素X坐标
            pixel_y (float): 像素Y坐标

        Returns:
            tuple: 世界坐标(x, y)或None
        """
        try:
            world_x, world_y = pixel_to_world_coordinates(pixel_x, pixel_y, 'FRUIT_CAMERA')
            return world_x, world_y
        except Exception as e:
            print(f"坐标转换出错: {e}")
            return None

    # IO事件
    def _reset_system(self):
        """系统复位"""
        try:
            # 添加复位逻辑
            # 例如：机械臂回到初始位置，清理状态等
            self._stop_sorting_process()
            # 使用 urController 执行复位操作
            self.urController.move_home()
            asyncio.run(self.system_manager.speak_async("系统已复位"))
            print("系统已复位")
        except Exception as e:
            print(f"复位过程中出错: {e}")
            asyncio.run(self.system_manager.speak_async("复位过程中出现错误"))

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
        self._reset_system()

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
                time.sleep(0.5)  # 控制处理频率
            except Exception as e:
                print(f"❌ 分拣过程中出错: {e}")
                asyncio.run(self.system_manager.speak_async("分拣过程中出现错误"))

        print("📦 分拣工作线程已停止")

    def _perform_sorting_step(self):
        """执行单步分拣操作"""
        print("🔄 执行分拣步骤...")

        # 1. 检测所有水果
        all_detections = self.detect_all_fruits()

        # 2. 如果没有检测到水果，直接返回
        if not all_detections or sum(len(dets) for dets in all_detections.values()) == 0:
            print("未检测到任何水果")
            return

        # 3. 遍历检测到的水果，按类别进行分拣
        fruit_count = 0
        for english_name, detections in all_detections.items():
            # 根据英文名找到对应的中文名
            chinese_name = next((cn for cn, en in self.fruit_classes.items() if en == english_name), None)
            if not chinese_name:
                continue

            # 为每个检测到的水果执行分拣操作
            for i, detection in enumerate(detections):
                if self._stop_event.is_set() or self.sorting_paused:
                    return

                # 获取水果中心点坐标
                x, y = detection['center']

                # 转换像素坐标到世界坐标
                world_coords = self._pixel_to_world(x, y)
                if not world_coords:
                    print(f"坐标转换失败，跳过{chinese_name}")
                    continue

                # 计算目标位置（根据水果类别分配到不同的九宫格位置）
                target_position = self._get_target_position_for_fruit(english_name, i)

                # 获取目标坐标
                target_coord = self.grid_points.get(target_position, self.grid_points[1])

                try:
                    # 移动水果
                    success = self._move_fruit_from_position_to_target(world_coords, target_coord)
                    if success:
                        print(f"已将{chinese_name}放置到位置{target_position}")
                        fruit_count += 1
                    else:
                        print(f"移动{chinese_name}失败")
                        asyncio.run(self.system_manager.speak_async("机械臂操作出错"))

                    # 短暂等待
                    time.sleep(0.5)

                except Exception as e:
                    print(f"机械臂操作出错: {e}")
                    asyncio.run(self.system_manager.speak_async("机械臂操作出错"))

        if fruit_count > 0:
            asyncio.run(self.system_manager.speak_async(f"已完成{fruit_count}个水果的分拣"))
        else:
            print("未分拣任何水果")

    def _get_target_position_for_fruit(self, fruit_type, index):
        """
        根据水果类型和索引确定目标位置

        Args:
            fruit_type (str): 水果英文类型
            index (int): 同类型水果的索引

        Returns:
            int: 目标九宫格位置(1-9)
        """
        # 根据水果类型分配到不同的区域
        position_mapping = {
            "apple": 1,      # 苹果区域
            "y-apple": 1,    # 黄苹果区域
            "g-apple": 1,    # 绿苹果区域
            "orange": 2,     # 橙子区域
            "citrus": 2,     # 柑橘区域
            "peach": 3,      # 桃子区域
            "persimmon": 4,  # 番茄区域
            "g-grape": 5,    # 绿葡萄区域
        }

        base_position = position_mapping.get(fruit_type, 6)  # 默认区域6

        # 在基础位置附近分配具体位置，避免重叠
        target_position = base_position + index
        return min(target_position, 9)

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
        frame_data = self.system_manager.get_camera_frame()

        # 注意：get_camera_frame 返回的是 (图像, 深度帧) 元组
        if isinstance(frame_data, tuple):
            frame = frame_data[0]  # 取第一个元素作为图像
        else:
            frame = frame_data

        # 加强帧检查
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
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

    def detect_all_fruits(self):
        """
        检测所有类型的水果
        """
        # 获取相机帧
        frame_data = self.system_manager.get_camera_frame()

        # 注意：get_camera_frame 返回的是 (图像, 深度帧) 元组
        if isinstance(frame_data, tuple):
            frame = frame_data[0]  # 取第一个元素作为图像
        else:
            frame = frame_data

        # 加强帧检查
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            print("无法获取有效的相机帧")
            asyncio.run(self.system_manager.speak_async("无法获取相机帧，无法检测水果"))
            return {}

        # 保存临时图像文件用于检测
        temp_image_path = "temp_frame.jpg"
        cv2.imwrite(temp_image_path, frame)

        # 使用训练好的模型进行预测
        results, detections = self.fruit_detector.predict(
            source=temp_image_path,
            conf=0.6,
            iou=0.45,
            save_dir=os.path.join(current_dir, '../src/cchessYolo/runs/obb_predict'),
            save=True,
        )

        # 按类别组织检测结果
        all_detections = {}
        for det in detections:
            class_name = det["class_name"]
            if class_name not in all_detections:
                all_detections[class_name] = []
            all_detections[class_name].append(det)

        # 输出检测到的水果信息
        total_count = sum(len(dets) for dets in all_detections.values())
        print(f"\n总共检测到 {total_count} 个水果:")
        speak_text = f"总共检测到{total_count}个水果"

        for english_name, chinese_name in self.fruit_classes.items():
            count = len(all_detections.get(english_name, []))
            if count > 0:
                print(f"  {chinese_name}: {count} 个")
                speak_text += f"，{chinese_name}{count}个"

        asyncio.run(self.system_manager.speak_async(speak_text))
        return all_detections

    def _calculate_fruit_grid_points(self):
        """
        根据FRUIT_BOX_POINT四角点计算九宫格1-9格的中心点
        FRUIT_BOX_POINT四角分别是9, 7, 1, 3位置的角点

        Returns:
            dict: 包含1-9位置中心点坐标的字典
        """
        # 获取水果盒四角点坐标 (右下, 左下, 左上, 右上)
        # 这四个点分别对应九宫格中的位置9, 7, 1, 3
        # 九宫格位置排列:
        # 1 | 2 | 3
        # ----------
        # 4 | 5 | 6
        # ----------
        # 7 | 8 | 9
        try:
            top_left = FRUIT_GRID_POINT[2]      # 位置1
            top_right = FRUIT_GRID_POINT[3]     # 位置3
            bottom_right = FRUIT_GRID_POINT[0]  # 位置9
            bottom_left = FRUIT_GRID_POINT[1]   # 位置7
        except ImportError:
            # 如果没有找到FRUIT_BOX_POINT，则使用默认值占位
            top_left = (100, 100)      # 位置1
            top_right = (300, 100)     # 位置3
            bottom_right = (300, 300)  # 位置9
            bottom_left = (100, 300)   # 位置7

        # 计算水平和垂直方向的间距
        # 水平间距
        step = (top_right[0] - top_left[0]) / 3  # 每个九宫格的大小
        step_x = step / 2
        step_y = step / 2
        # 计算各个位置的中心点
        grid_points = {}

        # 第一行 (位置1, 2, 3)
        grid_points[1] = (
            round(top_left[0] + step_x * 1, 2),
            round(top_left[1] + step_y * 1, 2)
        )
        grid_points[2] = (
            round(top_left[0] + step_x * 3, 2),
            round(top_left[1] + step_y * 1, 2)
        )
        grid_points[3] = (
            round(top_left[0] + step_x * 5, 2),
            round(top_left[1] + step_y * 1, 2)
        )

        # 第二行 (位置4, 5, 6)
        grid_points[4] = (
            round(top_left[0] + step_x * 1, 2),
            round(top_left[1] + step_y * 3, 2)
        )
        grid_points[5] = (
            round(top_left[0] + step_x * 3, 2),
            round(top_left[1] + step_y * 3, 2)
        )
        grid_points[6] = (
            round(top_left[0] + step_x * 5, 2),
            round(top_left[1] + step_y * 3, 2)
        )

        # 第三行 (位置7, 8, 9)
        grid_points[7] = (
            round(top_left[0] + step_x * 1, 2),
            round(top_left[1] + step_y * 5, 2)
        )
        grid_points[8] = (
            round(top_left[0] + step_x * 3, 2),
            round(top_left[1] + step_y * 5, 2)
        )
        grid_points[9] = (
            round(top_left[0] + step_x * 5, 2),
            round(top_left[1] + step_y * 5, 2)
        )

        return grid_points

    def _move_arm_to_position(self, position):
        """
        移动机械臂到指定位置

        Args:
            position (list): [x, y, z] 位置坐标
        """
        self.urController.move_to(position[0], position[1], position[2])

    def _gripper_action(self, action):
        """
        控制夹爪动作

        Args:
            action (str): "grab" 抓取, "open" 释放
        """
        if action == "grab":
            self.urController.set_do(IO_QI, 1)  # 吸合
        elif action == "open":
            self.urController.set_do(IO_QI, 0)  # 释放

    def _get_target_coordinates(self, target_position):
        """
        获取目标位置坐标

        Args:
            target_position (int or str): 目标位置

        Returns:
            tuple: 目标坐标(x, y)或None
        """
        if isinstance(target_position, int) and 1 <= target_position <= 9:
            # 移动到九宫格位置
            return self.grid_points[target_position]
        elif target_position in ["A", "B"]:
            # 移动到A点或B点
            if target_position == "A":
                return FRUIT_A_POINT[:2]  # 取XY坐标
            else:
                # 如果有B点需要定义
                return FRUIT_A_POINT[:2]  # 临时使用A点
        return None


def main():
    """主函数入口"""
    parser = argparse.ArgumentParser(description='水果分拣系统')
    parser.add_argument('--model-path', type=str,
                        default='E:/现有文件/工作/工程/新人工智能实训室/代码/V10.10/Embodied/src/cchessYolo/fruitYolo/runs/obb/fruit_obb_detection4/weights/best.pt',
                        help='水果检测模型路径')
    args = parser.parse_args()

    # 创建水果分拣应用实例
    app = FruitSortingApp()

    # 如果需要启动分拣
    print("启动水果分拣系统...")
    asyncio.run(app.start_sorting())
    time.sleep(10)
    app._pick_and_place_fruit('苹果',7)
    app.system_manager.cleanup()
    return app

if __name__ == "__main__":
    app = main()
