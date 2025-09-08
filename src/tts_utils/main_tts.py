import os
import platform
import socket
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from src.tts_utils.edgeTTS import EdgeTTSWrapper


class TTSManager():
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
        self._initialize_tts()

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
            # Linux或其他系统先不初始化特定引擎
            print("ℹ️  等待运行时决定使用哪个TTS引擎")
            self.tts_engine = None

    def speak(self, text):
        """
        根据当前状态播报文本（同步方法）

        Args:
            text (str): 要播报的文本
        """
        # 每次播报前检测网络状态
        is_network_available = self._check_network()
        print(f"网络状态: {'可用' if is_network_available else '不可用'}")

        if self.system == "windows":
            # Windows系统始终使用Edge TTS
            self._speak_with_edge_tts(text)
        elif self.system == "linux" and is_network_available:
            # Linux系统且网络可用时使用Edge TTS
            self._speak_with_edge_tts(text)
        else:
            # 其他情况使用Piper TTS
            self._fallback_to_piper(text)

    async def speak_async(self, text):
        """
        根据当前状态异步播报文本

        Args:
            text (str): 要播报的文本
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self.speak, text)

    def _speak_with_edge_tts(self, text):
        """
        使用Edge TTS引擎播报文本

        Args:
            text (str): 要播报的文本
        """
        try:
            # 如果还没有初始化Edge TTS，则初始化
            if not self.tts_engine:
                self.tts_engine = EdgeTTSWrapper()
            self.tts_engine.speak(text)
        except Exception as e:
            print(f"❌ Edge TTS 播报失败: {e}")
            # 回退到Piper TTS
            self._fallback_to_piper(text)

    def _fallback_to_piper(self, text):
        """
        回退到Piper TTS引擎

        Args:
            text (str): 要播报的文本
        """
        try:
            # 动态导入piperTTS模块
            import sys
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from piperTTS import synthesize_and_play

            print("🔄 使用 Piper TTS 引擎")
            synthesize_and_play(text)
        except Exception as e:
            print(f"❌ Piper TTS 播报失败: {e}")
            print("⚠️ 无法使用任何TTS引擎播报文本")

    def __del__(self):
        """
        析构函数，关闭线程池
        """
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

# 测试代码
if __name__ == "__main__":
    # 创建TTS管理器实例
    tts_manager = TTSManager()

    async def async_speak():
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
    asyncio.run(async_speak())
