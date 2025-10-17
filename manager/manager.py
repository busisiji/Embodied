# core/manager/system_manager.py
import asyncio
import threading
import time
from typing import Optional, Callable, Dict, List

from dobot.dobot_control import URController
from manager.tts_manager import TTSManager
from manager.speech_manager import SpeechManager
from manager.camera_manager import CameraManager

class SystemManager():
    """
    系统核心管理器，负责管理TTS、语音识别、相机和IO监控等核心组件
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self.tts_manager: Optional[TTSManager] = None
        self.speech_recognizer: Optional[SpeechManager] = None
        self.camera_manager: Optional[CameraManager] = None
        self.dobot_controller: Optional[URController] =  None

        self.speech_recognizer: Optional[SpeechManager] = None
        self.keyword_callbacks = {}  # 存储关键字和回调函数的映射

        # 异步事件循环
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None

        # IO监控相关
        self.io_monitoring = False
        self.io_monitor_thread: Optional[threading.Thread] = None
        self.io_callbacks: Dict[str, List[Callable]] = {
            "start": [],
            "stop": [],
            "reset": []
        }
        self.io_states = {"start": 0, "stop": 0, "reset": 0}
        self.dobot_controller = None  # 用于获取IO状态的机械臂控制器

        self._initialized = True


    def initialize(self,no_init=[]):
        """初始化所有核心组件"""
        try:
            if 'dobot' not in no_init:
                # 初始化机械臂
                self.initialize_dobot()
        except Exception as e:
            print(f"初始化机械臂时出错: {e}")

        try:
            if 'tts' not in no_init:
                # 初始化TTS管理器
                self.tts_manager = TTSManager()
        except Exception as e:
            print(f"初始化TTS管理器时出错: {e}")

        try:
            if 'speech' not in no_init:
                # 初始化语音识别器
                self.speech_recognizer = SpeechManager(self)
                self.speech_recognizer.start_listening(self._speech_callback)
        except Exception as e:
            print(f"初始化语音识别器时出错: {e}")

        try:
            if 'camera' not in no_init:
                # 初始化相机管理器
                self.camera_manager = CameraManager()
                # 启动相机
                if not self.camera_manager.initialize_camera():
                    print("⚠️ 相机启动失败")
        except Exception as e:
            print(f"初始化相机管理器时出错: {e}")

        # 初始化异步事件循环
        self._init_async_loop()

        # 注册默认IO回调函数
        self._register_default_io_callbacks()

        # 启动IO监控
        self.start_io_monitoring()

        print("✅ 系统核心组件初始化完成")

    def _init_async_loop(self):
        """初始化异步事件循环"""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()

        # 等待循环初始化完成
        while self.loop is None:
            pass

    # 机械臂相关
    def initialize_dobot(self, ip='192.168.5.1', port=30003, dashboard_port=29999, feed_port=30005):
        """初始化机械臂"""
        try:
            from dobot.dobot_control import connect_and_check_speed
            self.dobot_controller = connect_and_check_speed(
                ip=ip,
                port=port,
                dashboard_port=dashboard_port,
                feed_port=feed_port,
            )
            if self.dobot_controller and self.dobot_controller.is_connected():
                print("✅ 机械臂连接成功")
                return True
            else:
                print("❌ 机械臂连接失败")
                return False
        except Exception as e:
            print(f"初始化机械臂时出错: {e}")
            return False

    # 语音播报相关
    async def speak_async(self, text: str):
        """异步语音播报"""
        if self.tts_manager:
            await self.tts_manager.speak_async(text)

    def speak_sync(self, text: str):
        """同步语音播报"""
        if self.tts_manager and self.loop:
            future = asyncio.run_coroutine_threadsafe(
                self.tts_manager.speak_async(text),
                self.loop
            )
            future.result()  # 等待执行完成

    # 相机相关
    def get_camera_frame(self):
        """获取相机帧数据"""
        if self.camera_manager:
            return self.camera_manager.get_frame()
        return None

    # 语音识别相关
    def start_speech_recognition(self, callback=None):
        """开始语音识别"""
        if self.speech_recognizer:
            self.speech_recognizer.start_listening(callback)

    def _speech_callback(self, keywords, full_text):
        """统一的语音识别回调函数"""
        # 处理唤醒词
        if any(wake_word in full_text for wake_word in self.speech_recognizer.wake_words):
            asyncio.run(self.speak_async("我在"))
            return
        # 根据关键字调用相应的回调函数
        for keyword, callback in self.keyword_callbacks.items():
            if keyword in full_text:
                callback(keywords, full_text)

    def register_keyword_callback(self, keyword, callback):
        """注册关键字和回调函数"""
        self.keyword_callbacks[keyword] = callback

    def add_keywords(self, keywords):
        """添加关键字到语音识别器"""
        if self.speech_recognizer:
            self.speech_recognizer.add_keywords(keywords)

    def stop_speech_recognition(self):
        """停止语音识别"""
        if self.speech_recognizer:
            self.speech_recognizer.stop_listening()

    # IO监控相关
    def start_io_monitoring(self):
        """
        启动IO监控线程
        """
        if self.io_monitor_thread and self.io_monitor_thread.is_alive():
            print("⚠️ IO监控线程已在运行")
            return

        self.io_monitoring = True
        self.io_monitor_thread = threading.Thread(target=self._monitor_io_buttons, daemon=True)
        self.io_monitor_thread.start()
        print("🔔 IO监控线程已启动")

    def register_io_callback(self, io_type: str, callback: Callable):
        """
        注册IO回调函数
        @param io_type: IO类型 ("start", "stop", "reset")
        @param callback: 回调函数
        """
        if io_type in self.io_callbacks:
            self.io_callbacks[io_type].append(callback)
            print(f"✅ 已注册 {io_type} IO回调函数")

    def _default_start_callback(self):
        """默认的启动IO回调函数"""
        try:
            if self.dobot_controller:
                # 设置启动灯亮，其他灯暗
                self.dobot_controller.hll(1, [1, 2, 3])  # IO_START=1, IO_STOP=2, IO_RESET=3
                self.dobot_controller.resume()
                print("✅ 机械臂脚本已恢复")
                asyncio.run(self.speak_async("游戏继续"))
        except Exception as e:
            print(f"⚠️ 默认启动回调函数执行出错: {e}")

    def _default_stop_callback(self):
        """默认的停止IO回调函数"""
        try:
            if self.dobot_controller:
                # 设置停止灯亮，其他灯暗
                self.dobot_controller.hll(2, [1, 2, 3])  # IO_STOP=2, IO_START=1, IO_RESET=3
                self.dobot_controller.pause()
                print("✋ 游戏已暂停")
                asyncio.run(self.speak_async("游戏已暂停"))
        except Exception as e:
            print(f"⚠️ 默认停止回调函数执行出错: {e}")

    def _default_reset_callback(self):
        """默认的复位IO回调函数"""
        try:
            if self.dobot_controller:
                # 设置复位灯闪烁，其他灯暗
                self.dobot_controller.hll()  # 所有灯先暗

                # 启动复位灯闪烁线程
                def blink_reset_light():
                    for i in range(10):  # 最多闪烁10次
                        if not hasattr(self, '_resetting') or not self._resetting:
                            break
                        self.dobot_controller.set_do(3, 1)  # IO_RESET=3
                        time.sleep(0.5)
                        self.dobot_controller.set_do(3, 0)  # IO_RESET=3
                        time.sleep(0.5)

                self._resetting = True
                blink_thread = threading.Thread(target=blink_reset_light)
                blink_thread.daemon = True
                blink_thread.start()

                time.sleep(3)  # 等待复位操作完成

                # 复位完成，停止闪烁
                self._resetting = False
                blink_thread.join(timeout=1)

                # 设置复位灯亮，其他灯暗
                self.dobot_controller.hll(3, [1, 2, 3])  # IO_RESET=3, IO_START=1, IO_STOP=2
                asyncio.run(self.speak_async("系统已复位"))
        except Exception as e:
            print(f"⚠️ 默认复位回调函数执行出错: {e}")
        finally:
            if hasattr(self, '_resetting') and self._resetting:
                self._resetting = False

    def _register_default_io_callbacks(self):
        """注册默认的IO回调函数"""
        self.register_io_callback("start", self._default_start_callback)
        self.register_io_callback("stop", self._default_stop_callback)
        self.register_io_callback("reset", self._default_reset_callback)
        print("✅ 默认IO回调函数注册完成")

    def _monitor_io_buttons(self):
        """
        监控IO按钮的线程函数
        """
        # IO编号定义
        IO_START = 1  # 启动按钮IO编号
        IO_STOP = 2   # 停止按钮IO编号
        IO_RESET = 3  # 复位按钮IO编号

        last_states = {"start": 0, "stop": 0, "reset": 0}

        while self.io_monitoring:
            try:
                if not self.dobot_controller:
                    time.sleep(0.1)
                    continue

                # 获取当前IO状态
                try:
                    result = self.dobot_controller.get_dis(IO_START, IO_STOP, IO_RESET)
                    if len(result) >= 3:
                        start_state, stop_state, reset_state = result[0], result[1], result[2]
                    else:
                        time.sleep(0.1)
                        continue
                except Exception as e:
                    print(f"⚠️ 获取IO状态失败: {e}")
                    time.sleep(0.1)
                    continue

                # 检查启动按钮
                if start_state == 1 and last_states["start"] == 0:
                    print("🎮 检测到启动信号")
                    # 调用所有注册的启动回调函数
                    for callback in self.io_callbacks["start"]:
                        try:
                            callback()
                        except Exception as e:
                            print(f"⚠️ 启动回调函数执行出错: {e}")

                # 检查停止按钮
                if stop_state == 1 and last_states["stop"] == 0:
                    print("⏹️ 检测到停止信号")
                    # 调用所有注册的停止回调函数
                    for callback in self.io_callbacks["stop"]:
                        try:
                            callback()
                        except Exception as e:
                            print(f"⚠️ 停止回调函数执行出错: {e}")

                # 检查复位按钮
                if reset_state == 1 and last_states["reset"] == 0:
                    print("🔄 检测到复位信号")
                    # 调用所有注册的复位回调函数
                    for callback in self.io_callbacks["reset"]:
                        try:
                            callback()
                        except Exception as e:
                            print(f"⚠️ 复位回调函数执行出错: {e}")

                # 更新状态
                last_states["start"] = start_state
                last_states["stop"] = stop_state
                last_states["reset"] = reset_state

                time.sleep(0.01)  # 10ms检查一次，确保低延迟

            except Exception as e:
                print(f"⚠️ IO监控线程异常: {e}")
                time.sleep(0.1)

    def stop_io_monitoring(self):
        """
        停止IO监控线程
        """
        self.io_monitoring = False
        if self.io_monitor_thread and self.io_monitor_thread.is_alive():
            self.io_monitor_thread.join(timeout=2)
        print("🔕 IO监控线程已停止")

    # 退出清理
    def cleanup(self):
        """清理资源"""
        self.stop_io_monitoring()

        if self.speech_recognizer:
            self.speech_recognizer.stop_listening()

        if self.camera_manager:
            self.camera_manager.release_camera()

        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)


# 全局管理器实例
system_manager = SystemManager()
