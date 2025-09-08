import subprocess
import os
import time


def espeak_text_to_speech(text):
    """使用espeak进行离线TTS"""
    try:
        # 检查espeak是否可用
        if subprocess.call("which espeak-ng", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            cmd = ['espeak-ng', '-v', 'zh', '-s', '150', '-p', '50', '-a', '100']
            player_cmd = 'espeak-ng'
        elif subprocess.call("which espeak", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            cmd = ['espeak', '-v', 'zh', '-s', '150', '-p', '50', '-a', '100']
            player_cmd = 'espeak'
        else:
            raise Exception("未找到espeak或espeak-ng，请先安装")

        print(f"使用播放器: {player_cmd}")
        print("开始播放语音...")

        start_time = time.time()
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL
        )

        process.communicate(input=text.encode('utf-8'))
        process.wait()

        elapsed_time = time.time() - start_time
        print(f"✓ 语音播放完成！总用时: {elapsed_time:.2f}s")

    except Exception as e:
        print(f"✗ espeak播放失败: {e}")

if __name__ == '__main__':
    text = "这是一个测试文本，请忽略。"
    espeak_text_to_speech(text)