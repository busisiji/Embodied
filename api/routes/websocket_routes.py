# api/routes/websocket_routes.py
from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional

from starlette.websockets import WebSocket, WebSocketDisconnect

from api.services.model_service import logger
from api.services.websocket_service import websocket_manager
from api.models.user_model import User
from api.services.auth_service import get_current_user_from_request

router = APIRouter(prefix="/ws", tags=["报错管理"])

@router.websocket("/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket端点，用于实时接收进程状态更新和错误通知
    """
    # 必须首先接受WebSocket连接
    await websocket.accept()

    # 连接成功后，将客户端添加到管理器中
    await websocket_manager.connect(websocket, client_id)

    try:
        while True:
            # 接收客户端消息（可选）
            data = await websocket.receive_text()
            # 这里可以处理客户端发送的消息
            await websocket.send_text(f"收到消息: {data}")
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, client_id)
        await websocket_manager.broadcast({
            "type": "client_disconnected",
            "client_id": client_id,
            "message": f"客户端 {client_id} 已断开连接"
        })
    except Exception as e:
        # 确保在任何异常情况下都正确断开连接
        websocket_manager.disconnect(websocket, client_id)
        logger.error(f"WebSocket处理错误: {str(e)}")
