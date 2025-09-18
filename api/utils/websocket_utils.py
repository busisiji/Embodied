# api/utils/websocket_utils.py
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# 延迟导入，避免循环导入
def get_websocket_manager():
    from api.services.websocket_service import websocket_manager
    return websocket_manager

async def send_error_notification(process_type: str, process_id: str, error_message: str):
    """
    通过WebSocket发送错误通知的通用方法
    """
    try:
        websocket_manager = get_websocket_manager()
        error_notification = {
            "type": "process_error",
            "process_type": process_type,
            "process_id": process_id,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }

        await websocket_manager.broadcast(error_notification)
    except Exception as e:
        logger.error(f"通过WebSocket发送错误通知失败: {str(e)}")

def send_error_notification_sync(process_type: str,error_message: str='', process_id: str = None):
    """
    同步版本的错误通知发送方法
    """
    try:
        websocket_manager = get_websocket_manager()
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(send_error_notification(process_type, process_id, error_message))
        else:
            loop.run_until_complete(send_error_notification(process_type, process_id, error_message))
    except RuntimeError:
        try:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            new_loop.run_until_complete(send_error_notification(process_type, process_id, error_message))
        except Exception as e:
            logger.warning(f"创建新事件循环失败: {str(e)}")
    except Exception as e:
        logger.warning(f"发送WebSocket消息失败: {str(e)}")
