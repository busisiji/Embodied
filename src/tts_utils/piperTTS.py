import wave
import os
import threading

from piper import PiperVoice
current_dir = os.path.dirname(os.path.abspath(__file__))
def play_audio_file(file_path):
    """
    在不同平台上播放音频文件
    """
    import platform
    import subprocess

    system = platform.system().lower()

    try:
        if system == "windows":
            os.startfile(file_path)
        elif system == "darwin":
            result = subprocess.run(["afplay", file_path], capture_output=True)
            if result.returncode != 0:
                print(f"播放失败: {result.stderr.decode()}")
        elif system == "linux":
            # 尝试多种播放方式
            players = [
                ["aplay", "-f", "cd", file_path],
                ["paplay", file_path],
                ["ffplay", "-nodisp", "-autoexit", file_path],
            ]

            played = False
            for player in players:
                try:
                    result = subprocess.run(player, capture_output=True, timeout=10)
                    if result.returncode == 0:
                        played = True
                        break
                    else:
                        print(f"使用 {player[0]} 播放失败: {result.stderr.decode()}")
                except subprocess.TimeoutExpired:
                    print(f"{player[0]} 播放超时")
                except FileNotFoundError:
                    print(f"{player[0]} 未安装")

            if not played:
                print("所有音频播放器都失败了，请检查音频设备配置")
    except Exception as e:
        print(f"播放音频时出错: {e}")


def synthesize_and_play(text, cache_dir="tts_cache"):
    """
    合成语音并播放，同时保存到缓存文件

    Args:
        text (str): 要合成的文本
        cache_dir (str): 缓存目录路径
    """
    # 创建缓存目录
    cache_dir = os.path.join(current_dir, cache_dir)
    if not os.path.exists(cache_dir):
        os.makedirs( cache_dir)

    # 生成基于文本的文件名
    filename = "".join(c for c in text if c.isalnum() or c in (' ', '-', '_')).rstrip()
    if not filename:
        import hashlib
        filename = hashlib.md5(text.encode()).hexdigest()[:10]

    # 限制文件名长度
    filename = filename[:20]
    cache_file = os.path.join(cache_dir, f"{filename}.wav")
    # 检查缓存文件是否存在
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        print(f"从缓存播放: {cache_file}")
        play_audio_file(cache_file)
        return cache_file

    # 加载模型
    voice = PiperVoice.load(os.path.join(current_dir,"zh_CN-huayan-medium.onnx"))

    # 生成音频 - 使用正确的 synthesize 方法
    with wave.open(cache_file, "wb") as wav_file:
        # 设置音频参数
        wav_file.setframerate(voice.config.sample_rate)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setnchannels(1)  # mono

        # 直接合成音频数据
        voice.synthesize(text, wav_file)


    # 播放音频（在单独线程中播放以避免阻塞）
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        play_thread = threading.Thread(target=play_audio_file, args=(cache_file,))
        play_thread.daemon = True
        play_thread.start()
    else:
        print("错误：音频文件为空或不存在")
        return None

    return cache_file
def check_audio_devices():
    """
    检查可用的音频设备
    """
    import subprocess
    try:
        # 检查 ALSA 设备
        result = subprocess.run(["aplay", "-l"], capture_output=True, text=True)
        if result.returncode == 0:
            print("可用的音频播放设备:")
            print(result.stdout)
        else:
            print("无法列出音频设备:", result.stderr)
    except Exception as e:
        print(f"检查音频设备时出错: {e}")


def main():
    # 测试文本
    test_texts = [
        "欢迎来到语音合成的世界！",
        "你好，这是一个 Piper TTS 测试。",
        "今天天气怎么样？",
    ]
    # # 检查音频设备
    # check_audio_devices()

    print("Piper TTS 直接播报并生成缓存文件")
    print("=" * 50)

    for i, text in enumerate(test_texts, 1):
        print(f"\n处理文本 {i}: {text}")
        cache_file = synthesize_and_play(text)
        if cache_file and os.path.exists(cache_file):
            size = os.path.getsize(cache_file)
            print(f"文件大小: {size} 字节")
        # # 等待播放完成（简单地通过输入等待）
        # input("按回车键继续到下一个文本...")

    print("\n所有文本处理完成！")


if __name__ == "__main__":
    main()
