# core/manager/speech_manager.py
import asyncio
import json
import time
import queue
import threading
from typing import List, Callable, Optional
import os

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

from manager.speech_model_update import suggest_vocabulary_for_chess

current_dir = os.path.dirname(os.path.abspath(__file__))

class SpeechManager():
    """
    语音识别器
    """

    def __init__(self,
                 parent = None,
                 keywords: List[str] = None,
                 wake_words: List[str] = ['小助手'],
                 callback: Optional[Callable] = None,
                 model_path: str = os.path.join(current_dir, "speech_model")):
        """
        初始化语音识别器

        Args:
            keywords: 需要识别的关键词列表
            wake_words: 唤醒词列表
            callback: 识别到关键词后的回调函数
            model_path: 离线模型路径
        """
        self.system_manager = parent
        if self.system_manager:
            self.tts_manager = parent.tts_manager
        else:
            self.tts_manager = None

        self.callback = callback
        self.is_listening = False
        self.is_awake = False
        self.wake_timeout = 180  # 唤醒后保持活跃的时间（秒）
        self.last_wake_time = 0
        self._paused = False  # 添加暂停状态
        self.data_bytes = None

        # 音频参数
        self.sample_rate = 16000
        self.block_duration = 0.3  # 每块音频的时长（秒）
        self.block_size = int(self.sample_rate * self.block_duration)

        # 音频队列 - 增加最大尺寸限制，避免内存占用过高
        self.audio_queue = queue.Queue(maxsize=100)

        # 加载模型
        self.model = self._load_model(model_path)

        # 事件驱动机制
        self.pause_event = threading.Event()
        self.resume_event = threading.Event()
        self.stop_event = threading.Event()

        # 设置关键词和唤醒词
        self.keywords = [kw.lower() for kw in (keywords or [])]
        self.wake_words = [ww.lower() for ww in (wake_words or [])]
        vocab_words = suggest_vocabulary_for_chess()

        # 初始化识别器
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate, json.dumps(vocab_words, ensure_ascii=False))
        self.recognizer.SetWords(True)


        print("✅ 语音识别器初始化完成")
        if self.wake_words:
            print(f"唤醒词: {', '.join(self.wake_words)}")
        if self.keywords:
            print(f"关键词: {', '.join(self.keywords)}")

        # 将语音识别器设置到TTS管理器中
        if self.tts_manager:
            self.tts_manager.set_speech_recognizer(self)

        self.last_tts_time = 0  # 记录最后一次TTS时间
        self.tts_cooldown = 0.0  # TTS结束后2秒内忽略识别结果
    # 在 SpeechManager 类中添加以下方法
    def add_keywords(self, new_keywords: List[str]):
        """
        动态添加关键字并去重

        Args:
            new_keywords: 新的关键字列表
        """
        # 添加新关键字并去重
        for keyword in new_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in self.keywords:
                self.keywords.append(keyword_lower)

        # 更新识别器的词汇表
        if self.keywords:
            vocab_words = suggest_vocabulary_for_chess()
            # 合并基础词汇和新增关键字
            all_words = list(set(vocab_words + self.keywords))
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate, json.dumps(all_words, ensure_ascii=False))
            self.recognizer.SetWords(True)

        print(f"更新后关键词: {', '.join(self.keywords)}")

    def _load_model(self, model_path: str):
        """
        加载Vosk模型

        Args:
            model_path: 模型路径

        Returns:
            加载的模型对象
        """
        try:
            # 检查模型是否存在
            if not os.path.exists(model_path):
                print(f"未找到离线语音识别模型{model_path}，正在下载...")
                self._download_model(model_path)

            model = Model(model_path)
            print("✅ 离线语音识别模型加载成功")
            return model
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("请确保已下载中文语音识别模型")
            raise e

    def _download_model(self, model_path: str):
        """
        下载中文语音识别模型（提示用户手动下载）
        """
        print("=== 离线语音识别模型下载说明 ===")
        print("1. 请访问: https://alphacephei.com/vosk/models")
        print("2. 下载中文模型: vosk-model-small-cn-0.22 或 vosk-model-cn-0.22")
        print("3. 解压后将模型文件夹重命名为 'model' 并放在当前目录")
        print("4. 或者指定模型路径")
        print("\n示例下载命令:")
        print("wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip")
        print("unzip vosk-model-small-cn-0.22.zip")
        print("mv vosk-model-small-cn-0.22 model")
        raise Exception("请手动下载并指定模型路径")

    def audio_callback(self, indata, frames, time_info, status):
        """
        音频回调函数 - 事件驱动版本
        """
        if status:
            print(f"音频状态: {status}")

        try:
            # 尝试非阻塞方式放入队列
            self.audio_queue.put_nowait(indata.copy())
        except queue.Full:
            # 如果队列已满，移除最旧的元素并添加新的
            try:
                self.audio_queue.get_nowait()  # 移除最旧的元素
                self.audio_queue.put_nowait(indata.copy())  # 添加新的元素
            except:
                # 如果还是失败，就阻塞等待
                self.audio_queue.put(indata.copy())

    def check_wake_state(self, is_wait=True):
        """
        检查唤醒状态，如果超时则重置
        """
        if not is_wait or (self.is_awake and (time.time() - self.last_wake_time) > self.wake_timeout):
            self.is_awake = False
            print("唤醒状态已超时，进入休眠模式")
            asyncio.run(self.system_manager.speak_async("我睡了"))

    def process_text(self, text: str):
        """
        处理识别到的文本

        Args:
            text: 识别到的文本
        """
        start_time = time.time()  # 记录开始时间

        # 检查是否在TTS冷却期内
        if time.time() - self.last_tts_time < self.tts_cooldown:
            print(f"在TTS冷却期内，忽略识别结果: {text}")
            return

        text = text.lower()
        print(f"识别到语音: {text}")

        # 检查唤醒词
        wake_check_start = time.time()
        for wake_word in self.wake_words:
            if wake_word in text:
                self.is_awake = True
                self.last_wake_time = time.time()
                print(f"已被唤醒: {wake_word}")

                # 性能监控
                wake_found_time = time.time()
                wake_check_time = wake_found_time - wake_check_start
                print(f"唤醒词匹配耗时: {wake_check_time*1000:.2f}ms")

                # 异步执行唤醒回调
                self._async_wake_callback(wake_word)

                callback_start_time = time.time()
                callback_scheduling_time = callback_start_time - wake_found_time
                total_time = time.time() - start_time
                print(f"回调调度耗时: {callback_scheduling_time*1000:.2f}ms, 总耗时: {total_time*1000:.2f}ms")
                return
        wake_check_time = time.time() - wake_check_start
        print(f"完整唤醒词检查耗时: {wake_check_time*1000:.2f}ms")

        # 如果处于唤醒状态，检查关键词
        keyword_check_start = time.time()
        if self.is_awake:
            matched_keywords = [kw for kw in self.keywords if kw in text]
            print(f"匹配的关键词: {matched_keywords}")
            keyword_check_time = time.time() - keyword_check_start

            if self.callback:
                callback_start = time.time()
                self.callback(matched_keywords, text)
                callback_time = time.time() - callback_start
                total_time = time.time() - start_time
                print(f"关键词检查耗时: {keyword_check_time*1000:.2f}ms, 回调执行耗时: {callback_time*1000:.2f}ms, 总耗时: {total_time*1000:.2f}ms")
            return

        # 如果不使用唤醒模式，直接检查关键词
        if not self.wake_words:  # 没有设置唤醒词时
            matched_keywords = [kw for kw in self.keywords if kw in text]
            keyword_check_time = time.time() - keyword_check_start

            if self.callback:
                callback_start = time.time()
                self.callback(matched_keywords, text)
                callback_time = time.time() - callback_start
                total_time = time.time() - start_time
                print(f"关键词检查耗时: {keyword_check_time*1000:.2f}ms, 回调执行耗时: {callback_time*1000:.2f}ms, 总耗时: {total_time*1000:.2f}ms")

        total_time = time.time() - start_time
        print(f"文本处理总耗时: {total_time*1000:.2f}ms")

    def _async_wake_callback(self, wake_word: str):
        """
        异步执行唤醒回调函数
        """
        try:
            print(f"系统被唤醒: {wake_word}")
            if self.tts_manager:
                asyncio.run(self.system_manager.speak_async("我在"))
        except Exception as e:
            print(f"异步唤醒回调执行错误: {e}")

    def start_listening(self, callback=None):
        """
        开始监听（非阻塞方式，事件驱动）
        """
        if callback:
            self.callback = callback

        if self.is_listening:
            return

        self.is_listening = True
        self._paused = False

        # 重置事件状态
        self.pause_event.clear()
        self.resume_event.set()
        self.stop_event.clear()

        # 添加性能监控变量
        self.recognition_times = []  # 存储最近的识别时间，用于性能分析
        self.data_get_times = []  # 存储数据获取时间，用于性能分析

        def listen_thread():
            print("✅ 开始离线语音监听...")
            try:
                with sd.InputStream(
                        samplerate=self.sample_rate,
                        blocksize=self.block_size,
                        dtype=np.int16,
                        channels=1,
                        callback=self.audio_callback
                ):
                    while self.is_listening and not self.stop_event.is_set():
                        # 检查唤醒状态
                        start_time = time.time()
                        self.check_wake_state()
                        check_wake_time = time.time() - start_time

                        # 处理音频数据 - 事件驱动方式
                        try:
                            data_start_time = time.time()

                            # 使用非阻塞方式获取数据
                            try:
                                data = self.audio_queue.get_nowait()
                            except queue.Empty:
                                # 使用事件驱动的等待机制
                                if self.stop_event.wait(0.01):  # 等待10ms或直到停止
                                    break
                                continue

                            data_get_time = time.time() - data_start_time

                            # 记录数据获取时间用于性能分析
                            self.data_get_times.append(data_get_time)
                            if len(self.data_get_times) > 10:  # 保持最近10次记录
                                self.data_get_times.pop(0)

                            # 转换为bytes
                            self.data_bytes = data.tobytes()

                            # 检查暂停状态 - 使用事件驱动
                            if self._paused:
                                print("已暂停，正在等待...")
                                # 等待恢复事件或停止事件
                                if self.resume_event.wait(0.1) or self.stop_event.wait(0.1):
                                    continue
                                else:
                                    continue

                            # 识别音频
                            recognition_start_time = time.time()
                            if self.recognizer.AcceptWaveform(self.data_bytes):
                                result = json.loads(self.recognizer.Result())
                                recognition_time = time.time() - recognition_start_time

                                # 记录识别时间用于性能分析
                                self.recognition_times.append(recognition_time)
                                if len(self.recognition_times) > 10:  # 保持最近10次记录
                                    self.recognition_times.pop(0)

                                avg_recognition_time = sum(self.recognition_times) / len(
                                    self.recognition_times) if self.recognition_times else 0

                                print(f"处理时间统计 - 唤醒检查: {check_wake_time * 1000:.2f}ms, "
                                      f"数据获取: {data_get_time * 1000:.2f}ms, "
                                      f"语音识别: {recognition_time * 1000:.2f}ms (平均: {avg_recognition_time * 1000:.2f}ms)")
                                if 'text' in result and result['text'].strip():
                                    process_start_time = time.time()
                                    self.process_text(result['text'].strip())
                                    process_time = time.time() - process_start_time
                                    print(f"文本处理耗时: {process_time * 1000:.2f}ms")
                            else:
                                # 部分结果（可选处理）
                                partial_result = json.loads(self.recognizer.PartialResult())
                                recognition_time = time.time() - recognition_start_time

                                # 记录识别时间用于性能分析
                                self.recognition_times.append(recognition_time)
                                if len(self.recognition_times) > 10:  # 保持最近10次记录
                                    self.recognition_times.pop(0)

                                avg_recognition_time = sum(self.recognition_times) / len(
                                    self.recognition_times) if self.recognition_times else 0

                                # 只有在部分结果包含文字且长度大于1时才输出日志
                                if ('partial' in partial_result and
                                        partial_result['partial'].strip() and
                                        len(partial_result['partial'].strip()) > 1):
                                    print(f"处理时间统计 - 唤醒检查: {check_wake_time * 1000:.2f}ms, "
                                          f"数据获取: {data_get_time * 1000:.2f}ms, "
                                          f"部分识别: {recognition_time * 1000:.2f}ms (平均: {avg_recognition_time * 1000:.2f}ms)")

                        except queue.Empty:
                            # 队列为空时短暂休眠，避免CPU占用过高
                            if self.stop_event.wait(0.005):  # 等待5ms或直到停止
                                break
                            continue
                        except Exception as e:
                            print(f"音频处理错误: {e}")
                            if self.stop_event.wait(0.1):  # 等待100ms或直到停止
                                break

            except Exception as e:
                print(f"音频流错误: {e}")
            finally:
                print("语音监听已停止")

        # 启动监听线程
        self.listen_thread = threading.Thread(target=listen_thread, daemon=True)
        self.listen_thread.start()

    def stop_listening(self):
        """
        停止监听（事件驱动方式）
        """
        self.is_listening = False
        self.is_awake = False
        self._paused = False
        self.stop_event.set()

        # 清理事件状态
        self.pause_event.clear()
        self.resume_event.set()

        if hasattr(self, 'listen_thread'):
            self.listen_thread.join(timeout=1)

    def pause_listening(self):
        """
        暂停语音监听（事件驱动方式）
        """
        self._paused = True
        self.pause_event.set()
        self.resume_event.clear()
        print("语音识别已暂停")

    def resume_listening(self):
        """
        恢复语音监听并清空缓存（事件驱动方式）
        """
        # 清空音频队列缓冲区
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        self._paused = False
        self.pause_event.clear()
        self.resume_event.set()
        print("语音识别已恢复")

# # 全局语音识别器实例
# speech_recognizer: Optional[SpeechManager] = None
