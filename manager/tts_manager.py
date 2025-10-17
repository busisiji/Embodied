# core/manager/tts_manager.py
import asyncio
import hashlib
import os
import platform
import threading
import time
import socket
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import aiohttp

from src.tts_utils.edgeTTS import EdgeTTSWrapper
from src.tts_utils.ekhoTTS import EkhoTTS
dir = os.path.dirname(os.path.abspath(__file__))
class TTSManager:
    """
    统一的TTS管理器，根据系统环境和网络状态选择合适的TTS引擎
    """

    def __init__(self):
        """
        初始化TTS管理器
        """
        self.system = platform.system().lower()
        self.tts_engine: Optional[EdgeTTSWrapper] = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.speech_recognizer = None

        # 语音队列相关属性
        self.speech_queue: Optional[asyncio.Queue] = None
        self.speech_task: Optional[asyncio.Task] = None
        self.is_speaking = False

        self.was_listening = False
        self.is_network_available = True
        self.tts_cache_dir = os.path.join(dir, "../src/tts_utils/tts_cache")

        self._initialize_tts()

        # 自动启动语音队列处理器
        self._start_speech_queue_processor()


    def _initialize_tts(self):
        """
        初始化TTS引擎
        """
        print(f"系统类型: {self.system}")
        try:
            self.ekho_tts = None
            self.tts_engine = EdgeTTSWrapper(tts_cache_dir=self.tts_cache_dir)
            print("✅ 初始化 Edge TTS 引擎")
        except Exception as e:
            print(f"⚠️ Edge TTS 初始化失败: {e}")
            self.tts_engine = None

    def set_speech_recognizer(self, recognizer):
        """
        设置语音识别器引用，用于在播报时暂停和恢复识别

        Args:
            recognizer: 语音识别器实例
        """
        self.speech_recognizer = recognizer

    async def speak_async(self, text: str):
        """
        将文本添加到语音播报队列中

        Args:
            text (str): 要播报的文本
        """
        # 确保队列已初始化
        if self.speech_queue is None:
            self.speech_queue = asyncio.Queue()

        # 将播报任务添加到队列
        print(f"添加到语音队列: {text}")
        await self.speech_queue.put(text)

    async def _process_speech_queue(self):
        """
        处理语音播报队列中的任务
        """
        print("✅ 语音队列处理器已启动")
        # 确保队列已初始化
        if self.speech_queue is None:
            self.speech_queue = asyncio.Queue()

        while True:
            try:
                # 从队列中获取播报文本
                text = await self.speech_queue.get()

                # 设置正在播报标志
                self.is_speaking = True

                try:
                    # 执行实际的播报逻辑
                    await self._speak_text(text)
                except Exception as e:
                    print(f"语音播报出错: {e}")
                finally:
                    # 标记任务完成
                    self.speech_queue.task_done()
                    self.is_speaking = False

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"处理语音队列时出错: {e}")

    def _start_speech_queue_processor(self):
        """
        启动语音队列处理任务
        """
        try:
            # 如果任务已存在且正在运行，直接返回
            if self.speech_task and not self.speech_task.done():
                print("✅ 语音队列处理器已在运行")
                return True

            # 获取事件循环
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # 如果没有运行中的事件循环，创建新任务
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            self.speech_task = loop.create_task(self._process_speech_queue())
            print("✅ 语音队列处理器已启动")
            return True
        except Exception as e:
            print(f"❌ 启动语音队列处理器失败: {e}")
            return False

    async def _play_cached_audio_if_exists(self, text):
        """
        检查是否有缓存的音频文件，如果有则直接播放

        Args:
            text (str): 要播报的文本

        Returns:
            bool: 是否找到并播放了缓存音频
        """
        try:
            # filename = hashlib.md5(text.encode('utf-8')).hexdigest() + ".wav"
            filename = f"{text}.wav"
            cache_file_path = os.path.join(self.tts_cache_dir, filename)

            if os.path.exists(cache_file_path):
                print(f"播放缓存音频: {cache_file_path}")

                # 直接播放缓存的音频文件
                if self.system == "windows":
                    # Windows系统播放音频文件
                    import subprocess
                    process = await asyncio.create_subprocess_exec("powershell", "-c", f"Start-Process -FilePath '{cache_file_path}'")
                    await process.communicate()
                else:
                    # Linux系统播放音频文件
                    import subprocess
                    process = await asyncio.create_subprocess_exec("cvlc", "--play-and-exit", cache_file_path)
                    await process.communicate()


                return True
        except Exception as e:
            print(f"播放缓存音频失败: {e}")

        return False

    async def _check_network_async(self):
        """
        异步检查网络连接状态

        Returns:
            bool: 网络是否可用
        """
        try:
            # 使用aiohttp异步检查网络连接
            async with aiohttp.ClientSession() as session:
                async with session.get("http://www.baidu.com", timeout=aiohttp.ClientTimeout(total=3)) as response:
                    if response.status == 200:
                        self.is_network_available = True
                        return True
        except:
            pass

        try:
            # 备用检查
            async with aiohttp.ClientSession() as session:
                async with session.get("http://8.8.8.8", timeout=aiohttp.ClientTimeout(total=3)) as response:
                    if response.status == 200:
                        self.is_network_available = True
                        return True
        except:
            pass

        self.is_network_available = False
        return False

    async def _speak_text(self, text: str):
        """
        实际执行文本播报的逻辑

        Args:
            text (str): 要播报的文本
        """
        # 暂停语音识别
        if self.speech_recognizer and hasattr(self.speech_recognizer, 'is_listening'):
            self.was_listening = self.speech_recognizer.is_listening
            if self.was_listening:
                self.speech_recognizer.pause_listening()

        # 异步等待一段时间确保识别器已暂停
        await asyncio.sleep(0.1)

        try:
            # 首先检查是否有缓存的音频文件
            if not await self._play_cached_audio_if_exists(text):
                # 异步检查网络状态
                await self._check_network_async()
                print(f"网络状态: {'可用' if self.is_network_available else '不可用'}")

                # Windows系统始终使用Edge TTS
                if self.system == "windows" and self.tts_engine:
                    await self._speak_with_edge_tts_async(text)
                # Linux系统根据网络状态选择TTS引擎
                elif self.system == "linux":
                    if self.is_network_available:
                        # 网络可用时使用Edge TTS
                        await self._speak_with_edge_tts_async(text)
                    else:
                        # 网络不可用时使用Ekho TTS
                        await self._fallback_to_ekho_async(text)
                else:
                    # 其他系统使用Ekho TTS作为备选方案
                    await self._fallback_to_ekho_async(text)
        except Exception as e:
            print(f"TTS执行异常: {e}")
        finally:
            # 更新TTS时间戳
            if self.speech_recognizer:
                self.speech_recognizer.last_tts_time = time.time()

            # 播报完成后恢复语音识别（如果之前是开启状态）
            if self.was_listening and self.speech_recognizer:
                self.speech_recognizer.resume_listening()

    async def _speak_with_edge_tts_async(self, text: str):
        """
        异步使用Edge TTS引擎播报文本，并缓存音频文件

        Args:
            text (str): 要播报的文本
        """
        try:
            # 在线程池中运行阻塞的TTS调用
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self.tts_engine.speak, text)
        except Exception as e:
            print(f"❌ Edge TTS 播报失败: {e}")
            # 回退到Ekho TTS
            await self._fallback_to_ekho_async(text)

    async def _fallback_to_ekho_async(self, text: str):
        """
        异步回退到Ekho TTS引擎

        Args:
            text (str): 要播报的文本
        """
        try:
            print("🔄 使用 Ekho TTS 引擎")
            if not self.ekho_tts:
                self.ekho_tts = EkhoTTS()

            # 使用 asyncio.create_subprocess_exec 异步执行命令
            cmd = ["ekho", text]
            process = await asyncio.create_subprocess_exec(*cmd)
            await process.communicate()  # 等待进程完成
        except Exception as e:
            print(f"❌ Ekho TTS 播报失败: {e}")
            print("⚠️ 无法使用任何TTS引擎播报文本")

    def __del__(self):
        """
        析构函数，关闭线程池和队列任务
        """
        if hasattr(self, 'speech_task') and self.speech_task:
            self.speech_task.cancel()

        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

# # 创建全局TTS管理器实例
# tts_manager = TTSManager()

# 在文件末尾添加以下测试代码
if __name__ == "__main__":
    async def main():
        # 创建TTS管理器实例
        tts_manager = TTSManager()

        # 测试网络检测
        print("正在检测网络状态...")
        await tts_manager._check_network_async()
        print(f"网络状态: {'可用' if tts_manager.is_network_available else '不可用'}")

        # 测试不同情况下的TTS播报

        # 测试1: 基本文本播报
        test_text1 = "你好，这是TTS管理器测试。"
        print(f"\n正在添加到队列: {test_text1}")
        await tts_manager.speak_async(test_text1)
        await asyncio.sleep(2)  # 等待播报完成

        # 测试2: 多个文本连续播报
        texts = [
            "这是第一个句子。",
            "这是第二个句子。",
            "这是第三个句子。"
        ]

        print("\n测试连续播报:")
        for text in texts:
            print(f"添加到队列: {text}")
            await tts_manager.speak_async(text)

        await asyncio.sleep(5)  # 等待所有播报完成

        # 测试3: 特殊字符播报
        test_text3 = "今天的日期是2024年1月1日，星期一。"
        print(f"\n正在播报: {test_text3}")
        await tts_manager.speak_async(test_text3)
        await asyncio.sleep(2)  # 等待播报完成

        print("\n所有测试完成!")
        await asyncio.sleep(10)  # 等待播报完成


    # 运行异步测试
    asyncio.run(main())
