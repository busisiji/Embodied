# speech_service.py
import asyncio
import json
import time
from datetime import datetime

import aiohttp
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import queue
import threading
from typing import List, Callable, Optional
import sys
import os

from api.services.tts_service import tts_manager, speak_await, speak_async
from src.speech.model_update import suggest_vocabulary_for_chess

current_dir = os.path.dirname(os.path.abspath(__file__))


class OfflineKeywordRecognizer():
    """
    基于Vosk的离线关键词识别器
    """

    def __init__(self, keywords: List[str],
                 wake_words: List[str] = None,
                 callback: Optional[Callable] = None,
                 model_path: str = os.path.join(current_dir, "speech_model")):
        """
        初始化离线关键词识别器

        Args:
            keywords: 需要识别的关键词列表
            wake_words: 唤醒词列表
            callback: 识别到关键词后的回调函数
            model_path: 离线模型路径
        """
        self.callback = callback
        self.is_listening = False
        self.is_awake = False
        self.wake_timeout = 180  # 唤醒后保持活跃的时间（秒）
        self.last_wake_time = 0
        self._paused = False  # 添加暂停状态
        self.data_bytes = None

        # 添加设备状态和启动状态
        self.start_state = "停止"  # 启动状态: 停止/启动
        self.device_state = "空闲"  # 设备状态: 空闲/出库/停止装配/入库

        # 音频参数
        self.sample_rate = 16000
        self.block_duration = 0.3  # 每块音频的时长（秒）
        self.block_size = int(self.sample_rate * self.block_duration)

        # 音频队列 - 增加最大尺寸限制，避免内存占用过高
        self.audio_queue = queue.Queue()

        # 加载模型
        self.model = self._load_model(model_path)

        self.update_keywords(keywords, wake_words)

        print("离线语音识别器初始化完成")
        print(f"唤醒词: {', '.join(self.wake_words)}")
        print(f"关键词: {', '.join(self.keywords)}")

        # 将语音识别器设置到TTS管理器中
        if tts_manager:
            tts_manager.set_speech_recognizer(self)

        self.last_tts_time = 0  # 记录最后一次TTS时间
        self.tts_cooldown = 0.0  # TTS结束后2秒内忽略识别结果
    def update_keywords(self, keywords, wake_words = None):
        """
        更新识别器的关键词和唤醒词

        Args:
            keywords: 新的关键词列表
            wake_words: 新的唤醒词列表
        """
        self.keywords = [kw.lower() for kw in keywords]
        if wake_words:
            self.wake_words = [ww.lower() for ww in wake_words]

        # 更新识别器的词汇表
        vocab_words = suggest_vocabulary_for_chess() if "棋" in keywords or any(piece in str(keywords) for piece in ['车', '马', '炮', '象', '士', '将', '帅', '兵', '卒']) else keywords
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate, json.dumps(vocab_words, ensure_ascii=False))
        self.recognizer.SetWords(True)

        print(f"已更新关键词: {', '.join(self.keywords)}")
        if self.wake_words:
            print(f"已更新唤醒词: {', '.join(self.wake_words)}")

    def re_recognize_with_vocabulary(self, vocabulary_type="chess"):
        """
        使用指定词汇表重新识别最近的音频数据

        Args:
            vocabulary_type: 词汇表类型 ("chess" 或 "default")

        Returns:
            str: 重新识别的文本结果
        """
        # 清空当前识别器的结果缓存
        self.recognizer.FinalResult()

        # 根据类型设置对应的词汇表
        if vocabulary_type == "chess":
            chess_keywords = [
                # 棋子名称
                "车", "马", "炮", "象", "相", "士", "仕", "将", "帅", "兵", "卒",
                # 棋盘坐标 (列)
                "一", "二", "三", "四", "五", "六", "七", "八", "九",
                # 移动方向
                "进", "退", "平",
                # 位置描述
                "前", "后", "中",
                # 数字
                "1", "2", "3", "4", "5", "6", "7", "8", "9"
            ]
            vocab_words = chess_keywords
        else:
            # 使用默认词汇表
            vocab_words = self.keywords

        # 更新识别器的词汇表
        self.update_keywords(vocab_words)

        print(f"已切换到{vocabulary_type}词汇表进行重新识别")

        # 进行识别
        if self.recognizer.AcceptWaveform(self.data_bytes):
            result = json.loads(self.recognizer.Result())
            if 'text' in result and result['text'].strip():
                recognized_text = result['text'].strip()
                print(f"重新识别结果: {recognized_text}")
                return recognized_text

        return ""

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
            print("离线语音识别模型加载成功")
            return model
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("请确保已下载中文语音识别模型")
            raise  e

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

    def audio_callback(self, indata, frames, time, status):
        """
        音频回调函数 - 优化版本
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
    def check_wake_state(self,is_wait=True):
        """
        检查唤醒状态，如果超时则重置
        """
        import time
        if not is_wait or (self.is_awake and (time.time() - self.last_wake_time) > self.wake_timeout):
            self.is_awake = False
            print("唤醒状态已超时，进入休眠模式")
            # 发送休眠WebSocket消息
            # asyncio.run(send_message('sleep'))
            asyncio.run(tts_manager._speak_text("我睡了"))

    def process_text(self, text: str):
        """
        处理识别到的文本

        Args:
            text: 识别到的文本
        """
        import time
        start_time = time.time()  # 记录开始时间

        # 检查是否在TTS冷却期内
        if time.time() - self.last_tts_time < self.tts_cooldown:
            print(f"在TTS冷却期内，忽略识别结果: {text}")
            return

        text = text.lower()
        print(f"识别到语音: {text}")

        # 检查唤醒词
        # if not self.is_awake:
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
                result = self.callback(matched_keywords, text)
                # 一次匹配成功，则立即进入休眠模式
                # if result:
                #     self.check_wake_state(False)
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
                result = self.callback(matched_keywords, text)
                # 一次匹配成功，则立即进入休眠模式
                # if result:
                #     self.check_wake_state(False)
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
            asyncio.run(tts_manager._speak_text("我在"))
        except Exception as e:
            print(f"异步唤醒回调执行错误: {e}")


    async def start_listening(self):
        """
        开始监听（非阻塞方式）
        """
        if self.is_listening:
            return

        self.is_listening = True
        self._paused = False
        # 添加性能监控变量
        self.recognition_times = []  # 存储最近的识别时间，用于性能分析
        self.data_get_times = []     # 存储数据获取时间，用于性能分析

        def listen_thread():
            print("开始离线语音监听...")
            try:
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    dtype=np.int16,
                    channels=1,
                    callback=self.audio_callback
                ):
                    while self.is_listening:
                        # 检查唤醒状态
                        start_time = time.time()
                        self.check_wake_state()
                        check_wake_time = time.time() - start_time

                        # 处理音频数据 - 优化数据获取
                        try:
                            data_start_time = time.time()

                            # 使用非阻塞方式获取数据，减少等待时间
                            try:
                                data = self.audio_queue.get_nowait()
                            except queue.Empty:
                                # 如果队列为空，短暂等待后重试
                                data = self.audio_queue.get(timeout=0.01)  # 减少超时时间

                            data_get_time = time.time() - data_start_time

                            # 记录数据获取时间用于性能分析
                            self.data_get_times.append(data_get_time)
                            if len(self.data_get_times) > 10:  # 保持最近10次记录
                                self.data_get_times.pop(0)

                            avg_data_get_time = sum(self.data_get_times) / len(self.data_get_times) if self.data_get_times else 0

                            # 转换为bytes
                            self.data_bytes = data.tobytes()

                            # print(f"数据获取耗时: {data_get_time*1000:.2f}ms (平均: {avg_data_get_time*1000:.2f}ms)")

                            # 如果被暂停，则跳过处理
                            if self._paused:
                                print("已暂停，正在等待...")
                                time.sleep(1)
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

                                avg_recognition_time = sum(self.recognition_times) / len(self.recognition_times) if self.recognition_times else 0

                                print(f"处理时间统计 - 唤醒检查: {check_wake_time*1000:.2f}ms, "
                                      f"数据获取: {data_get_time*1000:.2f}ms, "
                                      f"语音识别: {recognition_time*1000:.2f}ms (平均: {avg_recognition_time*1000:.2f}ms)")
                                if 'text' in result and result['text'].strip():
                                    process_start_time = time.time()
                                    self.process_text(result['text'].strip())
                                    process_time = time.time() - process_start_time
                                    print(f"文本处理耗时: {process_time*1000:.2f}ms")
                            else:
                                # 部分结果（可选处理）
                                partial_result = json.loads(self.recognizer.PartialResult())
                                recognition_time = time.time() - recognition_start_time

                                # 记录识别时间用于性能分析
                                self.recognition_times.append(recognition_time)
                                if len(self.recognition_times) > 10:  # 保持最近10次记录
                                    self.recognition_times.pop(0)

                                avg_recognition_time = sum(self.recognition_times) / len(self.recognition_times) if self.recognition_times else 0

                                # 只有在部分结果包含文字且长度大于1时才输出日志
                                if ('partial' in partial_result and
                                    partial_result['partial'].strip() and
                                    len(partial_result['partial'].strip()) > 1):
                                    print(f"处理时间统计 - 唤醒检查: {check_wake_time*1000:.2f}ms, "
                                          f"数据获取: {data_get_time*1000:.2f}ms, "
                                          f"部分识别: {recognition_time*1000:.2f}ms (平均: {avg_recognition_time*1000:.2f}ms)")

                        except queue.Empty:
                            # 队列为空时短暂休眠，避免CPU占用过高
                            time.sleep(0.005)  # 5ms
                            continue
                        except Exception as e:
                            print(f"音频处理错误: {e}")

            except Exception as e:
                print(f"音频流错误: {e}")
            finally:
                print("语音监听已停止")

        # 启动监听线程
        self.listen_thread = threading.Thread(target=listen_thread, daemon=True)
        self.listen_thread.start()

    def stop_listening(self):
        """
        停止监听
        """
        self.is_listening = False
        self.is_awake = False
        self._paused = False
        if hasattr(self, 'listen_thread'):
            self.listen_thread.join(timeout=1)

    def pause_listening(self):
        """
        暂停语音监听
        """
        self._paused = True
        print("语音识别已暂停")


    def resume_listening(self):
        """
        恢复语音监听并清空缓存
        """
        # 清空音频队列缓冲区
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        self._paused = False
        print("语音识别已恢复")


    def recognize_once(self):
        """
        单次识别（阻塞方式）
        """
        print("请说话...")
        try:
            # 录制音频
            duration = 5  # 录制时长（秒）
            audio_data = sd.rec(
                int(self.sample_rate * duration),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.int16
            )
            sd.wait()  # 等待录制完成

            # 识别音频
            audio_bytes = audio_data.tobytes()
            if self.recognizer.AcceptWaveform(audio_bytes):
                result = json.loads(self.recognizer.Result())
                if 'text' in result and result['text'].strip():
                    text = result['text'].strip()
                    print(f"识别到语音: {text}")
                    return text
            else:
                partial_result = json.loads(self.recognizer.PartialResult())
                if 'partial' in partial_result and partial_result['partial'].strip():
                    text = partial_result['partial'].strip()
                    print(f"识别到语音: {text}")
                    return text

        except Exception as e:
            print(f"单次识别错误: {e}")

        return ""

# 用于存储语音识别器实例
speech_recognizer: Optional[OfflineKeywordRecognizer] = None

def initialize_speech_recognizer():
    """
    初始化语音识别器，从words.txt文件读取关键词
    """
    global speech_recognizer
    try:
        # 检查是否已经初始化
        if speech_recognizer is not None:
            print("语音识别器已经初始化，无需重复初始化")
            return False

        # 读取words.txt文件中的关键词
        words_file_path = os.path.join(current_dir, "speech_model/words.txt")
        keywords = []

        if os.path.exists(words_file_path):
            with open(words_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith('#'):  # 跳过空行和注释行
                        keywords.append(word)
        else:
            print(f"警告: 找不到词汇文件 {words_file_path}，使用默认关键词")
            # 象棋相关关键词列表（如果文件不存在时使用）
            keywords = [
                # 唤醒词
                "小助手", "你好",

                # 游戏控制
                "开始", "重新开始", "结束", "暂停", "继续",
                "布局", "布棋", "收棋", "收局", "收子",
                "悔棋", "认输", "投降", "退出", "帮助",
                "上一步", "下一步", "第一步", "最后一步",

                # 棋子名称
                "车", "马", "炮", "象", "相", "士", "仕",
                "将", "帅", "兵", "卒", "車", "馬", "砲",
                "紅方", "黑方", "红方", "黑方",

                # 棋盘坐标 (列)
                "一", "二", "三", "四", "五", "六", "七", "八", "九",
                "1", "2", "3", "4", "5", "6", "7", "8", "9",

                # 棋盘坐标 (行)
                "前", "后", "中", "上", "下", "左", "右",

                # 象棋术语
                "吃子", "将军", "照将", "胜负", "平局",
                "长将", "长杀", "长捉", "捉子", "兑子",
                "弃子", "献子", "等着", "先手", "后手",
                "开局", "中局", "残局", "杀棋", "困毙",

                # 移动指令
                "进", "退", "平", "走", "移动", "跳", "飞",
                "打", "吃", "捉", "献", "兑", "拦", "堵",

                # 难度设置
                "简单", "中等", "困难", "初级", "中级", "高级",
                "难度", "级别", "等级",

                # 游戏状态
                "思考中", "等待", "轮到", "回合", "时间",
                "超时", "违规", "重新走", "无效",

                # 语音反馈
                "我在", "好的", "明白", "确认", "取消",
                "是的", "不是", "可以", "不可以",

                # 常见干扰词（容易被误识别的词语）
                # 日常用语
                "什么", "怎么", "为什么", "可以", "不能",
                "现在", "这里", "那里", "这个", "那个",
                "一下", "一下下", "一点点", "很多", "一些",

                # 数字
                "零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
                "百", "千", "万", "亿",

                # 量词
                "个", "只", "台", "套", "件", "块", "片", "张", "条", "根",

                # 方位词
                "上", "下", "左", "右", "前", "后", "里", "外", "内", "中",

                # 动词
                "是", "有", "在", "做", "看", "听", "说", "想", "要", "会",

                # 形容词
                "好", "坏", "大", "小", "多", "少", "快", "慢", "高", "低",

                # 语气词
                "啊", "哦", "嗯", "呃", "啦", "吧", "吗", "呢", "了", "的",

                # 常见名词
                "东西", "地方", "时候", "时间", "问题", "情况", "方法", "工作",
            ]

        # 初始化语音识别器但不启动监听
        speech_recognizer = OfflineKeywordRecognizer(
            keywords=keywords,
            wake_words=["小助手"],
            callback=command_callback,
        )
        print("/语音识别器初始化完成")
        return True
    except Exception as e:
        print(f"语音识别器初始化失败: {e}")
        speech_recognizer = None
        return False


def cleanup_speech_recognizer():
    """
    清理语音识别器资源
    """
    global speech_recognizer
    if speech_recognizer and speech_recognizer.is_listening:
        speech_recognizer.stop_listening()
        print("/语音识别器已停止")



async def handle_detection_command(shape=None, color=None):
    """
    处理检测命令 - 按照指定规则合并颜色和形状

    组合规则:
    1. shape为圆,color为None -> 蓝色圆
    2. shape为None,color为红 -> 红色三角形
    3. shape为五边形,color为黄 -> 黄色五边形
    4. shape为矩形,color为绿 -> 绿色矩形

    Args:
        shape: 形状参数 (circle, triangle, rectangle, pentagon)
        color: 颜色参数 (red, blue, green, yellow)
    """
    start_time = time.time()
    # 颜色和形状的映射
    color_chinese_map = {
        "red": "红色", "blue": "蓝色", "green": "绿色", "yellow": "黄色"
    }

    shape_chinese_map = {
        "circle": "圆形", "rectangle": "矩形", "triangle": "三角形",
        "pentagon": "五边形"
    }

    # 根据规则组合颜色和形状
    target_color = color
    target_shape = shape

    # 规则1: shape为圆,color为None -> 蓝色圆
    if shape == "circle"or color == "blue":
        target_color = "blue"
        target_shape = "circle"

    # 规则2: shape为None,color为红 -> 红色三角形
    elif shape== "triangle" or  color == "red":
        target_color = "red"
        target_shape = "triangle"

    # 规则3: shape为五边形,color为黄 -> 黄色五边形
    elif shape == "pentagon" or color == "yellow":
        target_color = "yellow"
        target_shape = "pentagon"

    # 规则4: shape为矩形,color为绿 -> 绿色矩形
    elif shape == "rectangle"  or color == "green":
        target_color = "green"
        target_shape = "rectangle"

    # 如果不匹配任何规则，则使用默认值
    else:
        # 如果只有颜色，使用默认形状(圆形)
        if color is not None and shape is None:
            target_shape = "circle"
        # 如果只有形状，使用默认颜色(蓝色)
        elif shape is not None and color is None:
            target_color = "blue"
        # 如果都没有，则使用默认值
        elif shape is None and color is None:
            target_color = "blue"
            target_shape = "circle"

    try:
        time_speak_sync = time.time()
        # 在开始网络请求前先进行语音反馈
        msg = f"正在查找{color_chinese_map.get(target_color)}{shape_chinese_map.get(target_shape)}"
        # 使用 create_task 在后台执行 TTS，不阻塞当前函数
        tts_task = asyncio.create_task(speak_await(msg))
        print(f"正在执行检测命令: {msg}",time.time()-time_speak_sync)

        # 调用摄像头API进行检测
        url = "http://localhost:8000/camera/get_world_position"
        params = {}

        if target_color:
            params["color"] = target_color
        if target_shape:
            params["shape"] = target_shape

        # 设置更短的超时时间
        timeout = aiohttp.ClientTimeout(total=6)  # 3秒超时

        # 使用 aiohttp 进行异步 HTTP 请求
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.get(url, params=params) as response:
                    result = await response.json()

                    if "error" in result:
                        asyncio.create_task(speak_await(f"未找到{color_chinese_map.get(target_color)}{shape_chinese_map.get(target_shape)}"))
                    else:
                        objects = result.get("objects", [])
                        if len(objects) == 0:
                            asyncio.create_task(speak_await(f"未找到{color_chinese_map.get(target_color)}{shape_chinese_map.get(target_shape)}"))
                        elif len(objects) >= 1:
                            obj = objects[0]
                            msg = f"正在抓取{color_chinese_map.get(target_color)}{shape_chinese_map.get(target_shape)}"
                            asyncio.create_task(speak_await(msg))

            except asyncio.TimeoutError:
                print("请求超时")
                asyncio.create_task(speak_await("查找超时，请重试"))
            except aiohttp.ClientError as e:
                print(f"HTTP请求错误: {e}")
                asyncio.create_task(speak_await("网络请求失败，请检查网络连接"))

        print(f"检测命令总耗时: {time.time() - start_time:.3f}秒")

    except Exception as e:
        print(f"检测命令执行错误: {e}")
        asyncio.create_task(speak_await("执行检测命令时发生错误"))

def command_callback(keywords: List[str], full_text: str):
    """
    象棋命令回调函数 - 支持象棋相关命令
    """
    print(f"执行命令: {keywords}")



def get_speech_recognizer():
    """获取全局语音识别器实例"""
    global speech_recognizer
    return speech_recognizer

def start_listening():
    """
    开始语音识别监听
    """
    global speech_recognizer
    if speech_recognizer and not speech_recognizer.is_listening:
        speech_recognizer.start_listening()

def stop_listening():
    """
    停止语音识别监听
    """
    global speech_recognizer
    if speech_recognizer and speech_recognizer.is_listening:
        speech_recognizer.stop_listening()


def recognize_once():
    """
    单次语音识别
    """
    global speech_recognizer
    if speech_recognizer:
        return speech_recognizer.recognize_once()
    return ""

def is_listening():
    """
    检查是否正在监听
    """
    global speech_recognizer
    if speech_recognizer:
        return speech_recognizer.is_listening
    return False

def is_awake():
    """
    检查是否处于唤醒状态
    """
    global speech_recognizer
    if speech_recognizer and hasattr(speech_recognizer, 'is_awake'):
        return speech_recognizer.is_awake
    return False

if __name__ == "__main__":
    """
    主函数
    """

    async def main():
        try:
            # 在异步环境中初始化组件
            initialize_speech_recognizer()
            tts_manager._start_speech_queue_processor()

            global speech_recognizer
            if speech_recognizer:
                await speech_recognizer.start_listening()
                # 使用create_task确保TTS任务被正确调度
                asyncio.create_task(speak_await("开始监听"))
                print("语音识别已启动，按 Ctrl+C 退出...")
            else:
                print("语音识别器初始化失败")
                return

            # 保持事件循环运行
            while True:
                await asyncio.sleep(1)

        except Exception as e:
            print(f"程序运行出错: {e}")
        finally:
            # 清理资源
            if speech_recognizer:
                speech_recognizer.stop_listening()
            print("语音识别已停止")

    # 运行异步主函数
    try:
        print("正在启动语音识别服务...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序已退出")
    except Exception as e:
        print(f"程序启动失败: {e}")

