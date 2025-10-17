# src/tts_utils/edgeTTS.py
import asyncio
import time

import edge_tts
import os
import tempfile

class EdgeTTSWrapper():
    """
    Edge TTS 引擎包装器，用于将文本转换为语音
    """

    def __init__(self, voice="zh-CN-XiaoxiaoNeural", rate="+0%", volume="+0%",tts_cache_dir= None):
        """
        初始化 Edge TTS 引擎

        Args:
            voice (str): 语音类型，默认为中文女声
            rate (str): 语速，"+0%" 为正常速度
            volume (str): 音量，"+0%" 为正常音量
        """
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.tts_cache_dir = tts_cache_dir

        self.debug = True

    def speak(self, text):
        """
        将文本转换为语音并播放

        Args:
            text (str): 要转换为语音的文本
        """
        # 使用 asyncio.run 来运行异步函数
        asyncio.run(self._speak_async(text))

    async def _speak_async(self, text):
        """
        异步将文本转换为语音并播放

        Args:
            text (str): 要转换为语音的文本
        """
        try:
            start_time = time.time()  # 记录开始时间

            if self.tts_cache_dir:
                filename = f"{text}.wav"
                cache_file_path = os.path.join(self.tts_cache_dir, filename)
            else:
                # 创建临时文件来存储音频数据
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                    cache_file_path = temp_file.name

            if self.debug:
                step_time = time.time()
                print(f"[EdgeTTS] 文件路径准备耗时: {step_time - start_time:.3f}s")

            # 使用 edge-tts 生成音频
            communicate = edge_tts.Communicate(
                text,
                self.voice,
                rate=self.rate,
                volume=self.volume
            )

            if self.debug:
                step_time2 = time.time()
                print(f"[EdgeTTS] TTS对象创建耗时: {step_time2 - step_time:.3f}s")

            # 保存音频到临时文件
            save_task = asyncio.create_task(self._save_audio_background(communicate, cache_file_path))
            if self.debug:
                step_time3 = time.time()
                print(f"[EdgeTTS] 音频保存耗时: {step_time3 - step_time2:.3f}s")

            # 播放音频文件（Linux系统使用mpg123，Windows可以使用默认播放器）
            if os.name == "nt":  # Windows
                os.startfile(cache_file_path)
            else:  # Linux/Mac
                os.system(f"mpg123 '{cache_file_path}' >/dev/null 2>&1")

            if self.debug:
                step_time4 = time.time()
                print(f"[EdgeTTS] 音频播放命令耗时: {step_time4 - step_time3:.3f}s")

            await asyncio.sleep(0.1)  # 确保播放开始
            if not self.tts_cache_dir:
                os.remove(cache_file_path)
                if self.debug:
                    step_time5 = time.time()
                    print(f"[EdgeTTS] 临时文件清理耗时: {step_time5 - step_time4:.3f}s")

            if self.debug:
                total_time = time.time()
                print(f"[EdgeTTS] 总耗时: {total_time - start_time:.3f}s")

        except Exception as e:
            raise Exception(f"Edge TTS 播报失败: {str(e)}")

    # 完全异步保存，不阻塞主流程
    async def _save_audio_background(self, communicate, cache_file_path):
        try:
            await communicate.save(cache_file_path)
        except Exception as e:
            print(f"[EdgeTTS] 音频保存失败: {str(e)}")
    def set_voice(self, voice):
        """
        设置语音类型

        Args:
            voice (str): 语音类型
        """
        self.voice = voice

    def set_rate(self, rate):
        """
        设置语速

        Args:
            rate (str): 语速百分比
        """
        self.rate = rate

    def set_volume(self, volume):
        """
        设置音量

        Args:
            volume (str): 音量百分比
        """
        self.volume = volume
