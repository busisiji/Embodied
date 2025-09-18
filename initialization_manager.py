# initialization_manager.py
import logging
import threading
import time
import asyncio

from api.services.tts_service import speak_await, tts_manager, voice_loop, _init_voice_async_loop
from api.utils.websocket_utils import send_error_notification_sync
from src.speech.speech_service import initialize_speech_recognizer, get_speech_recognizer

# 设置日志
logger = logging.getLogger(__name__)

async def initialize_components():
    """
    初始化所有组件
    """
    # 初始化TTS
    try:
        from api.services.tts_service import  tts_manager

        if not tts_manager.speech_task or tts_manager.speech_task.done():
            # tts_manager._start_speech_queue_processor()
            _init_voice_async_loop()
        logger.info("TTS初始化成功")
        print("TTS初始化成功")
        await speak_await("语音播报已启动")
    except Exception as e:
        logger.error(f"TTS初始化失败: {e}")
        print(f"TTS初始化失败: {e}")
        # 使用统一错误处理函数发送错误信息
        send_error_notification_sync(f"语音播报初始化失败: {str(e)}", "error")

    # 初始化语音识别
    try:
        # 使用 speech_service.py 中的初始化函数
        if initialize_speech_recognizer(
        ):
            # 获取已创建的识别器实例
            speech_recognizer = get_speech_recognizer()

            # 启动监听
            if speech_recognizer:
                await speech_recognizer.start_listening()

            logger.info("语音识别初始化并启动成功")
            print("语音识别初始化并启动成功")
            await speak_await("语音识别已启动")
        else:
            logger.error(f"语音识别初始化失败{str(e)}")
            await speak_await("语音识别初始化失败")
            send_error_notification_sync(f"语音识别初始化失败{str(e)}", "error")

    except ImportError:
        logger.warning("未找到语音识别模块，语音识别功能将不可用")
        print("未找到语音识别模块，语音识别功能将不可用")
        await speak_await("未找到语音识别模块")
        send_error_notification_sync("未找到语音识别模块，语音识别功能将不可用", "error")
    except Exception as e:
        logger.error(f"语音识别初始化失败: {e}")
        await speak_await("语音识别初始化失败")
        print(f"语音识别初始化失败: {e}")
        send_error_notification_sync(f"语音识别初始化失败: {str(e)}", "error")

    # await speak_sync("初始化完成"))
    # await asyncio.sleep(5)

async def cleanup_components():
    """
    清理所有组件
    """
    pass
    # # 关闭相机
    # if camera:
    #     try:
    #         camera.stop_camera()
    #         logger.info("相机已关闭")
    #         await speak_await("相机已关闭")
    #     except Exception as e:
    #         logger.error(f"关闭相机时出错: {e}")
    #         await speak_await("关闭相机时出错")
    #         send_error_notification_sync(f"关闭相机时出错: {str(e)}", "error")

if __name__ == "__main__":
    asyncio.run(initialize_components())