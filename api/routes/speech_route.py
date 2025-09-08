# speech_route.py
from fastapi import APIRouter, BackgroundTasks
from typing import List
from pydantic import BaseModel
from services.speech_service import (
    start_listening,
    stop_listening,
    recognize_once,
    is_listening,
    is_awake,
    get_speech_recognizer
)

router = APIRouter()

class SpeechConfig(BaseModel):
    keywords: List[str] = ["打开", "关闭", "播放", "停止", "音乐", "灯光", "电视", "开始", "结束"]
    wake_words: List[str] = ["小助手", "你好", "嘿", "小灵"]
    model_path: str = "model"

class SpeechCommand(BaseModel):
    command: str
#
# @router.post("/start")
# async def start_speech_recognition(background_tasks: BackgroundTasks):
#     """
#     开始语音识别监听
#     """
#     if not get_speech_recognizer():
#         return {"error": "语音识别器未初始化"}
#
#     if is_listening():
#         return {"message": "语音识别已在运行中"}
#
#     try:
#         # 在后台线程中启动监听
#         background_tasks.add_task(start_listening)
#         return {"message": "语音识别已启动"}
#     except Exception as e:
#         return {"error": f"启动语音识别失败: {str(e)}"}
#
# @router.post("/stop")
# async def stop_speech_recognition():
#     """
#     停止语音识别监听
#     """
#     if not get_speech_recognizer():
#         return {"error": "语音识别器未初始化"}
#
#     if not is_listening():
#         return {"message": "语音识别未在运行"}
#
#     try:
#         stop_listening()
#         return {"message": "语音识别已停止"}
#     except Exception as e:
#         return {"error": f"停止语音识别失败: {str(e)}"}

@router.get("/status")
async def get_speech_status():
    """
    获取语音识别状态
    """
    recognizer = get_speech_recognizer()
    if not recognizer:
        return {"status": "未初始化"}

    return {
        "status": "运行中" if is_listening() else "已停止",
        "awake": is_awake(),
        "model_loaded": hasattr(recognizer, 'model') and recognizer.model is not None
    }

@router.post("/recognize")
async def recognize_speech_once_api():
    """
    单次语音识别
    """
    if not get_speech_recognizer():
        return {"error": "语音识别器未初始化"}

    try:
        result = recognize_once()
        return {"text": result}
    except Exception as e:
        return {"error": f"语音识别失败: {str(e)}"}

@router.post("/configure")
async def configure_speech_recognition(config: SpeechConfig):
    """
    配置语音识别器
    """
    try:
        # 注意：这里需要重新设计，因为初始化函数在服务中
        return {"message": "语音识别器配置功能需要重新实现"}
    except Exception as e:
        return {"error": f"配置语音识别器失败: {str(e)}"}
