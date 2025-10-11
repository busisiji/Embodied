# /home/sx/code/graphiCcolor/services/tts_service.py
import asyncio
import os
import platform
import sys
import threading
import time
import socket

import aiohttp
from concurrent.futures import ThreadPoolExecutor

from src.tts_utils.edgeTTS import EdgeTTSWrapper
from src.tts_utils.ekhoTTS import EkhoTTS


voice_loop,voice_thread = None,None
class TTSManager:
    """
    统一的TTS管理器，根据系统环境和网络状态选择合适的TTS引擎
    """

    def __init__(self):
        """
        初始化TTS管理器
        """
        self.system = platform.system().lower()
        self.tts_engine = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.speech_recognizer = None
        self.piper_available = False

        # 添加语音队列相关属性
        self.speech_queue = None
        self.speech_task = None
        self.is_speaking = False

        self._initialize_tts()
        self.was_listening = False
        self.is_network_available = True
        # asyncio.run(self._check_network_async())
        self.tts_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src/tts_utils/tts_cache")
    def _check_network(self):
        """
        检查网络连接状态

        Returns:
            bool: 网络是否可用
        """
        try:
            # 尝试连接到公共DNS服务器
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            pass
        try:
            # 尝试连接到公共网站
            socket.create_connection(("www.baidu.com", 80), timeout=3)
            return True
        except OSError:
            pass
        return False

    def _test_piper(self):
        """
        测试 Piper TTS 是否可用

        Returns:
            bool: Piper 是否可用
        """
        try:
            # 尝试导入 piper 相关模块
            from piper import PiperVoice
            import wave

            # 尝试加载模型文件
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "../../src/tts_utils/tts_model/zh_CN-huayan-medium.onnx")

            if os.path.exists(model_path):
                # 尝试加载模型
                voice = PiperVoice.load(model_path)
                if voice:
                    print("✅ Piper TTS 可用")
                    return True
            else:
                print(f"❌ Piper 模型文件不存在{model_path}")
        except Exception as e:
            print(f"❌ Piper TTS 不可用: {e}")

        return False

    def _initialize_tts(self):
        """
        初始化TTS引擎（不依赖网络状态）
        """
        print(f"系统类型: {self.system}")

        if self.system == "windows":
            # Windows系统使用Edge TTS
            try:
                self.tts_engine = EdgeTTSWrapper()
                print("✅ 初始化 Edge TTS 引擎")
            except Exception as e:
                print(f"⚠️ Edge TTS 初始化失败: {e}")
                self.tts_engine = None
        else:
            # Linux或其他系统测试 Piper 可用性
            print("ℹ️  测试 Piper TTS 可用性...")
            self.piper_available = self._test_piper()

            if not self.piper_available:
                print("⚠️  Piper TTS 不可用，将使用 Ekho TTS 作为离线语音引擎")

            self.tts_engine = None

    async def _play_cached_audio_if_exists(self, text):
        """
        检查是否有缓存的音频文件，如果有则直接播放

        Args:
            text (str): 要播报的文本

        Returns:
            bool: 是否找到并播放了缓存音频
        """
        try:
            # 根据文本生成文件名（这里简单地用文本作为文件名，实际可能需要更复杂的处理）
            filename = f"{text}.wav"
            cache_file_path = os.path.join(self.tts_cache_dir, filename)

            if os.path.exists(cache_file_path):
                # 如果找到缓存文件，先尝试使用Edge TTS播报
                if self.is_network_available and (self.system == "windows" or (self.system == "linux" and self.is_network_available)):
                    try:
                        print(f"使用Edge TTS播报缓存内容: {text}")
                        await self._speak_with_edge_tts_async(text)
                        return True
                    except Exception as e:
                        print(f"Edge TTS播报失败，回退到本地播放: {e}")

                # 如果Edge TTS不可用或失败，则直接播放缓存的音频文件
                print(f"播放缓存音频: {cache_file_path}")

                # 使用系统命令或音频库播放缓存的音频文件
                # 这里需要根据实际使用的音频播放库进行调整
                import subprocess
                process = await asyncio.create_subprocess_exec("aplay", cache_file_path)  # Linux示例
                # Windows可以使用 'powershell' 和 ' MediaPlayer.MediaPlayer' 或其他方式
                await process.communicate()

                return True
        except Exception as e:
            print(f"播放缓存音频失败: {e}")

        return False

    def set_speech_recognizer(self, recognizer):
        """
        设置语音识别器引用，用于在播报时暂停和恢复识别

        Args:
            recognizer: 语音识别器实例
        """
        self.speech_recognizer = recognizer

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

    async def speak_async(self, text):
        """
        将文本添加到语音播报队列中

        Args:
            text (str): 要播报的文本
        """
        # 确保队列已初始化
        if self.speech_queue is None:
            self.speech_queue = asyncio.Queue()

        # 将播报任务添加到队列
        await self.speech_queue.put(text)

    async def _process_speech_queue(self):
        """
        处理语音播报队列中的任务
        """
        print("处理语音播报队列中的任务")
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

    async def _speak_text(self, text):
        """
        实际执行文本播报的逻辑

        Args:
            text (str): 要播报的文本
        """
        if self.speech_recognizer and hasattr(self.speech_recognizer, 'is_listening'):
            self.was_listening = self.speech_recognizer.is_listening
            # 暂停语音识别
            if self.was_listening:
                self.speech_recognizer.pause_listening()

        # 异步等待一段时间确保识别器已暂停
        await asyncio.sleep(0.1)

        try:
            if not await self._play_cached_audio_if_exists(text):
                # 异步检查网络状态
                asyncio.create_task(self._check_network_async())
                print(f"网络状态: {'可用' if self.is_network_available else '不可用'}")

                if self.system == "windows":
                    # Windows系统始终使用Edge TTS
                    await self._speak_with_edge_tts_async(text)
                elif self.system == "linux" and self.is_network_available:
                    # Linux系统且网络可用时使用Edge TTS
                    await self._speak_with_edge_tts_async(text)
                elif self.piper_available:
                    # 使用Piper TTS
                    await self._fallback_to_piper_async(text)
                else:
                    # 使用 Ekho TTS 作为最后的备选方案
                    await self._fallback_to_ekho_async(text)
        except Exception as e:
            print(f"TTS执行异常: {e}")
            if self.speech_recognizer:
                self.speech_recognizer.resume_listening()
        finally:
            # 更新TTS时间戳
            if self.speech_recognizer:
                self.speech_recognizer.last_tts_time = time.time()

            # 播报完成后恢复语音识别（如果之前是开启状态）
            if self.was_listening and self.speech_recognizer:
                # 异步等待播报完全结束并增加额外延迟
                # await asyncio.sleep(0.5)  # 异步延迟，确保音频播放完成
                self.speech_recognizer.resume_listening()

    async def _speak_with_edge_tts_async(self, text):
        """
        异步使用Edge TTS引擎播报文本

        Args:
            text (str): 要播报的文本
        """
        try:
            # 如果还没有初始化Edge TTS，则初始化
            if not self.tts_engine:
                self.tts_engine = EdgeTTSWrapper()

            # 在线程池中运行阻塞的TTS调用
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self.tts_engine.speak, text)
        except Exception as e:
            print(f"❌ Edge TTS 播报失败: {e}")
            # 回退到Piper TTS 或 Ekho TTS
            if self.piper_available:
                await self._fallback_to_piper_async(text)
            else:
                await self._fallback_to_ekho_async(text)

    async def _fallback_to_piper_async(self, text):
        """
        异步回退到Piper TTS引擎

        Args:
            text (str): 要播报的文本
        """
        try:
            # 在线程池中运行阻塞的Piper TTS调用
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._synthesize_and_play_blocking,
                text
            )
        except Exception as e:
            print(f"❌ Piper TTS 播报失败: {e}")
            # 如果 Piper 失败，尝试使用 Ekho
            await self._fallback_to_ekho_async(text)

    def _synthesize_and_play_blocking(self, text):
        """
        在线程中运行的阻塞式Piper TTS调用
        """
        try:
            from src.tts_utils.piperTTS import synthesize_and_play
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            print("🔄 使用 Piper TTS 引擎")
            synthesize_and_play(text)
        except Exception as e:
            raise e


    async def _ekho_speak_async(self, text):
        """
        异步执行Ekho TTS调用
        """
        try:
            print("🔄 使用 Ekho TTS 引擎")
            ekho_tts = EkhoTTS()

            # 使用 asyncio.create_subprocess_exec 异步执行命令
            cmd = ["ekho", text]
            process = await asyncio.create_subprocess_exec(*cmd)
            await process.communicate()  # 等待进程完成
        except Exception as e:
            raise e

    async def _fallback_to_ekho_async(self, text):
        """
        异步回退到Ekho TTS引擎

        Args:
            text (str): 要播报的文本
        """
        try:
            # 直接await异步方法，无需使用线程池
            await self._ekho_speak_async(text)
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

# 创建全局TTS管理器实例
tts_manager = TTSManager()
def _init_voice_async_loop():
    """
    初始化异步语音播报的事件循环
    """
    if voice_loop is not None:
        print("语音事件循环已存在，无需重复初始化")
        return False
    def run_loop():
        try:
            global voice_loop,voice_thread,camera
            voice_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(voice_loop)

            # 在事件循环启动后立即启动TTS处理器
            tts_manager._start_speech_queue_processor()

            # 运行事件循环
            voice_loop.run_forever()
            print("✅ 语音事件循环已启动")
        except Exception as e:
            print(f"⚠️ 语音事件循环启动失败: {e}")

    voice_thread = threading.Thread(target=run_loop, daemon=True)
    voice_thread.start()

    # 等待循环初始化完成
    while voice_loop is None:
        time.sleep(0.01)
async def speak_async(text: str):
    try:
        # 在异步事件循环中执行
        asyncio.run_coroutine_threadsafe(
            tts_manager.speak_async(text),
            voice_loop
        )
    except:
        print("无法使用TTS播放文本")

async def speak_await(text: str):
    """
    同步文本转语音播放（不按隊列）
    """
    global tts_manager, last_tts_time

    if tts_manager:
        try:
            # 使用阻塞方式播放音频
            await tts_manager._speak_text(text)
            last_tts_time = time.time()
        except Exception as e:
            print(f"TTS播放错误: {e}")
    else:
        print("TTS管理器未初始化")


# 测试代码
if __name__ == "__main__":
    async def main():
        # 创建TTS管理器实例

        # 启动处理任务
        tts_manager._start_speech_queue_processor()

        # 测试文本
        test_text = "你好，这是统一TTS管理器的异步播报测试。"
        print(f"\n正在异步播报: {test_text}")
        await tts_manager.speak_async(test_text)
        print("异步播报完成")

        await asyncio.sleep(0.1)

        # 测试文本
        test_text = "断网测试"
        print(f"\n正在异步播报: {test_text}")
        await tts_manager.speak_async(test_text)
        print("异步播报完成")

        await asyncio.sleep(0.1)

        # 测试文本
        test_text = "联网测试"
        print(f"\n正在异步播报: {test_text}")
        await tts_manager.speak_async(test_text)
        print("异步播报完成")

    # 运行异步测试
    asyncio.run(main())