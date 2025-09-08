#!/usr/bin/env python3
import subprocess
import os
import tempfile
import time

class EkhoTTS:
    def __init__(self):
        """初始化 ekho TTS 类"""
        self.check_ekho()

    def check_ekho(self):
        """检查 ekho 是否已安装"""
        try:
            result = subprocess.run(["ekho", "--help"],
                                  capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception("ekho 命令执行失败")
        except FileNotFoundError:
            raise Exception("未找到 ekho 命令，请确保已正确安装 ekho")

    def speak(self, text, sync=True):
        """
        播报文本

        Args:
            text (str): 要播报的文本
            sync (bool): 是否同步播放（等待播放完成）
        """
        cmd = ["ekho", text]
        if sync:
            subprocess.run(cmd)
        else:
            subprocess.Popen(cmd)

    def save_to_file(self, text, filename, format="wav"):
        """
        将文本保存为音频文件

        Args:
            text (str): 要转换的文本
            filename (str): 输出文件名
            format (str): 音频格式 (wav, mp3, ogg)
        """
        cmd = ["ekho", "-o", filename, text]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"音频已保存到: {filename}")
            return True
        else:
            print(f"保存失败: {result.stderr}")
            return False

    def speak_with_options(self, text, speed=0, pitch=0, volume=100,
                          voice="Mandarin", sync=True):
        """
        带参数的文本播报

        Args:
            text (str): 要播报的文本
            speed (int): 语速 (-50 到 100)
            pitch (int): 音调 (-100 到 100)
            volume (int): 音量 (0 到 100)
            voice (str): 语音类型 (mandarin, cantonese, tibetan 等)
            sync (bool): 是否同步播放
        """
        cmd = ["ekho"]

        # 添加参数
        cmd.extend(["-s", str(speed)])      # 语速
        cmd.extend(["-p", str(pitch)])      # 音调
        cmd.extend(["-a", str(volume)])     # 音量
        cmd.extend(["-v", voice])           # 语音类型


        cmd.append(text)

        # 执行命令
        if sync:
            subprocess.run(cmd)
        else:
            subprocess.Popen(cmd)

    def get_voices(self):
        """获取支持的语音列表"""
        # 运行 ekho --help 来获取帮助信息
        result = subprocess.run(["ekho", "--help"],
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            voices = []
            for line in lines:
                if '-l, --language' in line or 'mandarin' in line or 'cantonese' in line:
                    voices.append(line.strip())
            return voices
        return []

    def get_pinyin(self, text):
        """获取文本的拼音"""
        cmd = ["ekho", "-l", "pinyin", text]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"获取拼音失败: {result.stderr}")
            return None

    def speak_with_background_music(self, text, music_file=None):
        """带背景音乐的语音播报（需要sox等工具）"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_voice:
            voice_file = tmp_voice.name

        # 生成语音
        self.save_to_file(text, voice_file)

        if music_file and os.path.exists(music_file):
            # 混合语音和背景音乐（需要sox）
            mixed_file = "mixed_output.wav"
            try:
                subprocess.run([
                    "sox", "-m", voice_file, music_file, mixed_file
                ])
                # 播放混合音频
                subprocess.run(["play", mixed_file])  # 需要安装sox
                # 清理临时文件
                os.unlink(voice_file)
                os.unlink(mixed_file)
            except FileNotFoundError:
                print("需要安装sox工具来混合音频")
                # 直接播放语音
                subprocess.run(["play", voice_file])
                os.unlink(voice_file)
        else:
            # 直接播放语音
            subprocess.run(["play", voice_file])  # 需要安装sox
            os.unlink(voice_file)
    def speak_with_retry(self, text, max_retries=3, **kwargs):
        """
        带重试机制的语音播报

        Args:
            text (str): 要播报的文本
            max_retries (int): 最大重试次数
            **kwargs: 其他参数传递给 speak_with_options
        """
        for attempt in range(max_retries):
            try:
                self.speak_with_options(text, **kwargs)
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"语音播报失败，第{attempt + 1}次重试: {e}")
                    time.sleep(0.5)
                else:
                    print(f"语音播报最终失败: {e}")
                    return False
    def preprocess_text(self, text):
        """
        预处理文本以提高语音播报质量

        Args:
            text (str): 原始文本

        Returns:
            str: 处理后的文本
        """
        # 移除多余空格
        text = ' '.join(text.split())

        # 处理标点符号，添加适当停顿
        text = text.replace('，', '， ')
        text = text.replace('。', '。 ')
        text = text.replace('！', '！ ')
        text = text.replace('？', '？ ')

        return text

    def speak_natural(self, text, sync=True):
        """
        自然语音播报（自动优化文本和参数）

        Args:
            text (str): 要播报的文本
            sync (bool): 是否同步播放
        """
        # 预处理文本
        processed_text = self.preprocess_text(text)

        # 使用优化参数播报
        self.speak_with_options(
            processed_text,
            speed=-5,      # 稍慢语速
            pitch=20,      # 适中音调
            volume=90,     # 适中音量
            voice="Mandarin",
            sync=sync
        )
    def speak_with_enhanced_quality(self, text, speed=-5, pitch=20, volume=90,
                                   voice="Mandarin", sample_rate=22050, sync=True):
        """
        高质量语音播报（增强版）

        Args:
            text (str): 要播报的文本
            speed (int): 语速 (-50 到 100)
            pitch (int): 音调 (-100 到 100)
            volume (int): 音量 (0 到 100)
            voice (str): 语音类型
            sample_rate (int): 采样率 (默认22050Hz)
            sync (bool): 是否同步播放
        """
        cmd = ["ekho"]

        # 添加参数
        cmd.extend(["-s", str(speed)])           # 语速
        cmd.extend(["-p", str(pitch)])           # 音调
        cmd.extend(["-a", str(volume)])          # 音量
        cmd.extend(["-v", voice])                # 语音类型
        cmd.extend(["--samplerate", str(sample_rate)])  # 采样率

        cmd.append(text)

        # 执行命令
        if sync:
            subprocess.run(cmd)
        else:
            subprocess.Popen(cmd)
    def preprocess_speech_text(self, text):
        """
        预处理语音文本以提高播报质量，将阿拉伯数字转换为中文数字

        Args:
            text: 原始文本

        Returns:
            str: 处理后的文本
        """

        # 中文数字映射
        num_map = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
        unit_map = ['', '十', '百', '千']
        big_unit_map = ['', '万', '亿']

        def convert_number(num_str):
            """将数字字符串转换为中文读法"""
            if not num_str.isdigit():
                return num_str

            num = int(num_str)
            if num == 0:
                return '零'

            # 处理简单情况：0-9
            if num < 10:
                return num_map[num]

            # 处理10-99
            if num < 100:
                if num == 10:
                    return '十'
                elif num < 20:
                    return '十' + num_map[num % 10]
                else:
                    return num_map[num // 10] + '十' + (num_map[num % 10] if num % 10 != 0 else '')

            # 更大的数字使用简单替换方式
            result = ''
            for digit in num_str:
                result += num_map[int(digit)]
            return result

        # 使用正则表达式查找并替换所有数字
        import re
        result = re.sub(r'\d+', lambda m: convert_number(m.group()), text)

        return result

    def preprocess_text_enhanced(self, text):
        """
        增强型文本预处理以提高语音播报质量

        Args:
            text (str): 原始文本

        Returns:
            str: 处理后的文本
        """
        # 移除多余空格
        text = ' '.join(text.split())

        # 更好的标点处理
        # 在标点符号后添加适当的停顿（使用空格实现停顿效果）
        text = text.replace('，', '， ')      # 在中文逗号后添加空格
        text = text.replace('。', '。 ')      # 在句号后添加空格
        text = text.replace('！', '！ ')      # 在感叹号后添加空格
        text = text.replace('？', '？ ')      # 在问号后添加空格
        text = text.replace('；', '； ')      # 在分号后添加空格
        text = text.replace('：', '： ')      # 在冒号后添加空格

        # 处理长句，添加中途停顿
        sentences = text.split('。 ')
        processed_sentences = []
        for sentence in sentences:
            if len(sentence) > 30:  # 对长句进行分割
                # 简单按逗号分割长句
                parts = sentence.split('， ')
                if len(parts) > 1:
                    sentence = '， '.join(parts[:-1]) + '， ' + parts[-1]
            processed_sentences.append(sentence)

        text = '。 '.join(processed_sentences)

        text = self.preprocess_speech_text(text)
        return text


    def speak_with_emotion(self, text, emotion="happy"):
        """
        带情感的语音播报

        Args:
            text (str): 要播报的文本
            emotion (str): 情感类型 (happy, sad, angry 等，取决于ekho支持)
        """
        # 根据情感调整参数
        emotion_params = {
            "happy": {"speed": 5, "pitch": 30, "volume": 95},
            "sad": {"speed": -15, "pitch": -10, "volume": 70},
            "angry": {"speed": 10, "pitch": 40, "volume": 100},
            "calm": {"speed": -10, "pitch": 10, "volume": 80}
        }

        params = emotion_params.get(emotion, emotion_params["happy"])

        self.speak_with_options(
            text,
            speed=params["speed"],
            pitch=params["pitch"],
            volume=params["volume"],
            voice="Mandarin"
        )

def demo():
    """演示EkhoTTS功能（高质量版）"""
    try:
        # 创建EkhoTTS实例
        tts = EkhoTTS()
        print("✓ 成功初始化 ekho TTS")
        test_text = "这是一个高质量语音播报的测试。"
        test_text = tts.preprocess_text_enhanced(test_text)

        # 测试高质量语音播报
        # print("\n1. 测试高质量语音播报...")
        # tts.speak_natural("你好，世界！欢迎使用 ekho 文本转语音系统。")

        # 测试不同语音参数
        print("\n2. 测试不同语音参数...")
        print("自然语速和音调:")
        tts.speak_with_options(test_text, speed=-5, pitch=-20, volume=90)

        print("快速高音调:")
        tts.speak_with_options("这是快速高音调的语音。", speed=10, pitch=-35, volume=95)

        print("慢速低音调:")
        tts.speak_with_options("这是慢速低音调的语音。", speed=-20, pitch=-5, volume=80)

        # 测试情感语音
        print("\n3. 测试情感语音...")
        print("开心语气:")
        tts.speak_with_emotion("今天天气真好，我很开心！", "happy")

        print("平静语气:")
        tts.speak_with_emotion("请保持冷静，慢慢来。", "calm")

        # 测试保存到文件
        print("\n4. 测试保存到高质量音频文件...")
        tts.save_to_file("今天天气真不错，适合出去散步。", "high_quality_output.wav")

        print("\n✓ 所有测试完成！")

    except Exception as e:
        print(f"✗ 测试过程中出现错误: {e}")

def interactive_test():
    """交互式测试"""
    try:
        tts = EkhoTTS()
        print("=== Ekho TTS 交互式测试 ===")
        print("输入 'quit' 退出程序")

        while True:
            text = input("\n请输入要播报的文本: ").strip()
            if text.lower() == 'quit':
                break

            if not text:
                continue

            print("选择操作:")
            print("1. 普通播报")
            print("2. 自定义参数播报")
            print("3. 保存到文件")
            print("4. 获取拼音")

            choice = input("请选择操作 (1-4): ").strip()

            if choice == "1":
                tts.speak(text)
            elif choice == "2":
                try:
                    speed = int(input("语速 (-50到100, 默认0): ") or "0")
                    pitch = int(input("音调 (-100到100, 默认0): ") or "0")
                    volume = int(input("音量 (0到100, 默认100): ") or "100")
                    voice = input("语音类型 (mandarin/cantonese/tibetan, 默认mandarin): ") or "mandarin"
                    tts.speak_with_options(text, speed, pitch, volume, voice)
                except ValueError:
                    print("参数输入错误，使用默认参数播报")
                    tts.speak(text)
            elif choice == "3":
                filename = input("文件名 (默认output.wav): ") or "output.wav"
                tts.save_to_file(text, filename)
            elif choice == "4":
                pinyin = tts.get_pinyin(text)
                if pinyin:
                    print(f"拼音: {pinyin}")
            else:
                print("无效选择，执行普通播报")
                tts.speak(text)

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序出现错误: {e}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_test()
    else:
        demo()


