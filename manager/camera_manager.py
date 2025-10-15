# camera_manager.py
import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import time
from typing import Optional, Tuple

class CameraManager:
    """
    相机管理器，专门用于 Intel RealSense D435i 相机
    """

    def __init__(self, width=1280, height=720, fps=6):
        """
        初始化相机管理器

        Args:
            width: 图像宽度
            height: 图像高度
            fps: 帧率
        """
        self.width = width
        self.height = height
        self.fps = fps

        # 相机相关属性
        self.pipeline = None
        self.config = None
        self.running = False

        # 图像显示相关
        self.show_camera = True
        self.window_name = "camera"

    def initialize_camera(self) -> bool:
        """
        初始化 RealSense D435i 相机

        Returns:
            bool: 初始化是否成功
        """
        try:
            # 创建 pipeline
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # 配置彩色流和深度流
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, self.fps)

            # 启动相机
            profile = self.pipeline.start(self.config)
            self.running = True

            print("✅ 相机初始化成功")
            return True

        except Exception as e:
            print(f"⚠️ 相机初始化失败: {e}")
            self.running = False
            return False

    def setup_camera_windows(self):
        """
        初始化相机显示窗口
        """
        if self.show_camera:
            try:
                # 先清理可能存在的窗口
                cv2.destroyAllWindows()
                # 创建新窗口
                cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)
            except cv2.error as e:
                print(f"⚠️ 创建窗口时出错: {e}")
                self.show_camera = False

    def update_camera_display(self, image):
        """
        更新相机显示

        Args:
            image: 要显示的图像
        """
        if self.show_camera and image is not None:
            try:
                # 检查窗口是否存在
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    # 如果窗口不存在，重新创建
                    cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)

                # 显示图像
                cv2.startWindowThread()
                cv2.imshow(self.window_name, image)

                # 使用1ms等待，检查按键事件
                key = cv2.waitKey(1) & 0xFF

                # 检查是否按下ESC键(27)或窗口被关闭
                # getWindowProperty返回-1表示窗口已被关闭
                if key == 27 or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:  # ESC键或窗口关闭
                    print("ESC键被按下或窗口已关闭，关闭显示窗口")
                    cv2.destroyAllWindows()
                    self.show_camera = False  # 禁用后续显示

            except cv2.error as e:
                print(f"⚠️ 更新显示时出错: {e}")

    def cleanup_camera_windows(self):
        """
        清理相机窗口
        """
        try:
            if self.show_camera:
                cv2.destroyAllWindows()
        except:
            pass

    def get_frame(self):
        """
        获取单帧图像和深度信息

        Returns:
            tuple: (图像, 深度帧) 或 (None, None) 如果失败
        """
        return self.capture_stable_image(num_frames=1)

    def capture_stable_image(self, num_frames=5, max_retries=3) -> Tuple[Optional[np.ndarray], Optional[rs.depth_frame]]:
        """
        捕获稳定的图像和深度信息（通过多帧平均减少噪声）

        Args:
            num_frames: captured帧数用于平均
            max_retries: 最大重试次数

        Returns:
            tuple: (稳定图像, 深度帧) 或 (None, None) 如果失败
        """
        for attempt in range(max_retries):
            if not self.running or self.pipeline is None:
                print("⚠️ 相机未初始化")
                if attempt < max_retries - 1:  # 不是最后一次尝试
                    print(f"🔄 尝试重新初始化相机 ({attempt + 1}/{max_retries})")
                    if self.initialize_camera():
                        print("✅ 相机重新初始化成功")
                    else:
                        print("❌ 相机重新初始化失败")
                        time.sleep(1)  # 等待1秒后重试
                        continue
                else:
                    return None, None

            try:
                frames_list = []
                depth_frames_list = []

                # 捕获多帧图像
                for i in range(num_frames):
                    frames = self.pipeline.wait_for_frames(timeout_ms=5000)  # 设置超时时间
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()

                    if color_frame and depth_frame:
                        frame = np.asanyarray(color_frame.get_data())
                        frames_list.append(frame)
                        depth_frames_list.append(depth_frame)
                    else:
                        continue

                if not frames_list:
                    raise Exception("无法捕获有效图像帧")

                # 如果只捕获到一帧，直接返回
                if len(frames_list) == 1:
                    result_frame = frames_list[0]
                    latest_depth_frame = depth_frames_list[0]
                else:
                    # 多帧平均以减少噪声（仅对彩色图像）
                    result_frame = np.mean(frames_list, axis=0).astype(np.uint8)
                    # 使用最新的深度帧
                    latest_depth_frame = depth_frames_list[-1]

                return result_frame, latest_depth_frame

            except Exception as e:
                print(f"⚠️ 捕获图像失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:  # 不是最后一次尝试
                    # 释放当前相机资源
                    self.release_camera()
                    time.sleep(1)  # 等待1秒后重试
                    # 重新初始化相机
                    self.initialize_camera()
                else:
                    return None, None

        return None, None


    def release_camera(self):
        """
        释放相机资源
        """
        self.running = False
        if self.pipeline:
            try:
                self.pipeline.stop()
                print("✅ 相机资源已释放")
            except Exception as e:
                print(f"⚠️ 释放相机资源时出错: {e}")


if __name__ == "__main__":
    # 创建相机管理器实例
    camera_manager = CameraManager()

    try:
        # 初始化相机
        if not camera_manager.initialize_camera():
            print("❌ 无法初始化相机")
            exit(1)

        # 设置显示窗口
        camera_manager.setup_camera_windows()

        print("📸 开始捕获图像，按 ESC 键退出...")

        # 主循环
        while camera_manager.running:
            # 捕获稳定图像
            image, depth_frame = camera_manager.capture_stable_image()

            if image is not None:
                # 显示图像
                camera_manager.update_camera_display(image)

                # 可以在这里添加图像处理逻辑
                # print(f"📷 捕获到图像，尺寸: {image.shape}")
            else:
                print("⚠️ 未能捕获到图像")
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断程序")
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
    finally:
        # 清理资源
        camera_manager.cleanup_camera_windows()
        camera_manager.release_camera()
        print("🔚 程序结束")
