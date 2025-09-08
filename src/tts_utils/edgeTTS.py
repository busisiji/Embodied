# 在 edgeTTS.py 文件中修改 EdgeTTSWrapper 类

import edge_tts
import asyncio
import subprocess
import hashlib
import os
from pathlib import Path

# 添加缓存目录
CACHE_DIR = Path.home() / ".edge_tts_cache"
CACHE_DIR.mkdir(exist_ok=True)

class EdgeTTSWrapper():
    """
    Edge TTS 语音合成工具类
    将文本转换为语音并播放
    """

    # 可用的中文语音选项
    CHINESE_VOICES = {
        "zh-CN-XiaoxiaoNeural": "中文女声（推荐）",
        "zh-CN-YunyangNeural": "中文男声",
        "zh-HK-HiuGaaiNeural": "粤语女声",
        "zh-TW-HsiaoChenNeural": "台湾国语女声"
    }

    def __init__(self, voice="zh-CN-XiaoxiaoNeural"):
        """
        初始化 Edge TTS 播报器

        Args:
            voice: 语音名称，默认使用中文女声
        """
        self.selected_voice = voice
        self.rate = '+0%'
        self.volume = '+0%'
        self.pitch = '+0Hz'
        self.player_cmd = None
        self._find_player()


    def _find_player(self):
        """查找可用的音频播放器"""
        for cmd in ['ffplay', 'mpg123', 'mpg321']:
            if subprocess.call(f"which {cmd}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                self.player_cmd = cmd
                break

        if not self.player_cmd:
            raise Exception("未找到可用的音频播放器，请安装 ffplay、mpg123 或 mpg321")

    def set_voice_parameters(self, rate=None, volume=None, pitch=None):
        """
        设置语音参数

        Args:
            rate: 语速，例如 '+10%'、'-20%'
            volume: 音量，例如 '+0%'、'-10%'
            pitch: 音调，例如 '+50Hz'、'-30Hz'
        """
        if rate is not None:
            self.rate = rate
        if volume is not None:
            self.volume = volume
        if pitch is not None:
            self.pitch = pitch

    def set_voice(self, voice):
        """
        设置语音

        Args:
            voice: 语音名称
        """
        if voice in self.CHINESE_VOICES:
            self.selected_voice = voice
        else:
            raise ValueError(f"不支持的语音: {voice}")

    def preprocess_text(self, text):
        """
        优化的文本预处理
        """
        # 移除多余空格但保留基本停顿
        text = ' '.join(text.split())

        # 简化标点处理，避免过多停顿
        text = text.replace('，', ',')
        text = text.replace('。', '.')
        text = text.replace('！', '!')
        text = text.replace('？', '?')

        # 限制文本长度以提高生成速度
        if len(text) > 100:
            # 对于长文本，可以考虑分段处理
            pass

        return text.strip()

    def _is_audio_device_available(self):
        """
        检查音频设备是否可用

        Returns:
            bool: 音频设备是否可用
        """
        try:
            # 重新检测播放器命令（以防设备插拔后路径变化）
            self._find_player()

            if not self.player_cmd:
                return False

            # 尝试使用播放器检测音频设备
            if self.player_cmd == 'ffplay':
                # 使用 aplay 检查 ALSA 设备（如果可用）
                result = subprocess.run(['aplay', '-l'],
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
                return result.returncode == 0
            elif self.player_cmd in ['mpg123', 'mpg321']:
                # 对于 mpg 播放器，尝试简单测试
                result = subprocess.run([self.player_cmd, '--help'],
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
                return result.returncode == 0

        except Exception:
            # 如果检测命令失败，回退到基本检查
            try:
                # 检查是否存在音频设备
                result = subprocess.run(['ls', '/dev/snd/'],
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
                return result.returncode == 0
            except:
                pass

        return True  # 默认返回 True，避免阻止播放

    async def _async_speak(self, text, max_retries=3):
        """
        异步播放 TTS 音频（带重试机制和缓存）

        Args:
            text: 要播报的文本
            max_retries: 最大重试次数
        """
        # 生成文本的哈希值用于缓存
        text_hash = hashlib.md5(f"{text}_{self.selected_voice}_{self.rate}_{self.volume}_{self.pitch}".encode()).hexdigest()
        cached_file = CACHE_DIR / f"{text_hash}.mp3"

        # 检查是否有缓存
        if cached_file.exists():
            await self._async_play_audio(str(cached_file))
            return

        for attempt in range(max_retries):
            try:
                # 使用缓存文件路径而不是临时文件
                temp_filename = str(cached_file)

                communicate = edge_tts.Communicate(
                    text=text,
                    voice=self.selected_voice,
                    rate=self.rate,
                    volume=self.volume,
                    pitch=self.pitch
                )

                await communicate.save(temp_filename)
                await self._async_play_audio(temp_filename)

                return  # 成功则返回
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 指数退避
                else:
                    # 清理可能损坏的缓存文件
                    try:
                        if cached_file.exists():
                            cached_file.unlink()
                    except:
                        pass
                    raise e


    def _play_cached_audio(self, audio_file):
        """
        播放缓存的音频文件
        """
        try:
            if self.player_cmd == 'ffplay':
                cmd = ['ffplay', '-autoexit', '-nodisp', '-loglevel', 'quiet', audio_file]
            elif self.player_cmd in ['mpg123', 'mpg321']:
                cmd = [self.player_cmd, '-q', audio_file]

            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"⚠️ 缓存音频播放失败: {e}")
    async def _async_play_audio(self, audio_file, max_retries=3):
        """
        异步播放音频文件（优化版）
        """
        try:
            # 检查音频设备是否可用
            for attempt in range(max_retries):
                if not self._is_audio_device_available():
                    if attempt < max_retries - 1:
                        print(f"⚠️ 音频设备未就绪，等待中... ({attempt + 1}/{max_retries})")
                        await asyncio.sleep(2 ** attempt)  # 指数退避
                        continue
                    else:
                        print("❌ 音频设备长时间未就绪，播放失败")
                        raise Exception("音频设备未就绪")

            # 构建播放命令，添加参数以提高播放速度
            if self.player_cmd == 'ffplay':
                cmd = [
                    'ffplay',
                    '-autoexit',
                    '-nodisp',
                    '-loglevel', 'quiet',
                    '-af', 'atempo=1.2',  # 音频加速播放
                    audio_file
                ]
            elif self.player_cmd in ['mpg123', 'mpg321']:
                cmd = [self.player_cmd, '-q', '--speed', '1.2', audio_file]
            else:
                return

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            await process.wait()

        except Exception as e:
            print(f"⚠️ 音频播放失败: {e}")

    def speak(self, text):
        """
        播报文本（同步方法）

        Args:
            text: 要播报的文本
        """
        processed_text = self.preprocess_text(text)
        asyncio.run(self._async_speak(processed_text))

    async def speak_async(self, text):
        """
        异步播报文本

        Args:
            text: 要播报的文本
        """
        processed_text = self.preprocess_text(text)
        await self._async_speak(processed_text)

    def speak_with_params(self, text, rate=None, volume=None, pitch=None):
        """
        使用特定参数播报文本

        Args:
            text: 要播报的文本
            rate: 语速
            volume: 音量
            pitch: 音调
        """
        # 保存当前参数
        old_rate = self.rate
        old_volume = self.volume
        old_pitch = self.pitch

        # 设置新参数
        if rate is not None:
            self.rate = rate
        if volume is not None:
            self.volume = volume
        if pitch is not None:
            self.pitch = pitch

        # 播报
        self.speak(text)

        # 恢复原参数
        self.rate = old_rate
        self.volume = old_volume
        self.pitch = old_pitch

    async def speak_with_params_async(self, text, rate=None, volume=None, pitch=None):
        """
        使用特定参数异步播报文本

        Args:
            text: 要播报的文本
            rate: 语速
            volume: 音量
            pitch: 音调
        """
        # 保存当前参数
        old_rate = self.rate
        old_volume = self.volume
        old_pitch = self.pitch

        # 设置新参数
        if rate is not None:
            self.rate = rate
        if volume is not None:
            self.volume = volume
        if pitch is not None:
            self.pitch = pitch

        # 异步播报
        await self.speak_async(text)

        # 恢复原参数
        self.rate = old_rate
        self.volume = old_volume
        self.pitch = old_pitch

    @classmethod
    def list_voices(cls):
        """
        列出所有支持的中文语音
        """
        print("可用的中文语音选项：")
        for voice, description in cls.CHINESE_VOICES.items():
            print(f"  {voice}: {description}")


# 在 edgeTTS.py 文件末尾添加测试代码
if __name__ == "__main__":
    # 创建 EdgeTTSWrapper 实例，使用默认的甜美女声
    tts = EdgeTTSWrapper(voice="zh-CN-XiaoxiaoNeural")

    print(tts._is_audio_device_available())

    # 测试甜美女声
    print("正在测试甜美女声...")
    test_text = "你好，我是小晓，这是甜美女声测试。"
    tts.speak(test_text)
    #
    # # 测试不同的语音参数
    # print("正在测试带有参数的甜美女声...")
    # tts.speak_with_params(
    #     "你好，这是调整了语速、音量和音调的甜美女声测试。",
    #     rate="+15%",
    #     volume="+10%",
    #     pitch="+20Hz"
    # )
