# api/services/websocket_service.py

import asyncio
import json
import logging
from typing import Dict, Set, Any, Optional
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)

class WebSocketManager:
    """
    WebSocket连接管理器 (增强版错误处理)
    """
    def __init__(self):
        # 存储所有活跃的WebSocket连接
        self.active_connections: Dict[str, Set[Any]] = {}
        # 存储每个用户/客户端的连接
        self.client_connections: Dict[str, Any] = {}
        # 存储连接错误信息
        self.connection_errors: Dict[str, list] = {}

    async def connect(self, websocket, client_id: str):
        """
        建立WebSocket连接
        """
        try:
            if client_id not in self.active_connections:
                self.active_connections[client_id] = set()
                self.connection_errors[client_id] = []

            self.active_connections[client_id].add(websocket)
            self.client_connections[client_id] = websocket

            logger.info(f"客户端 {client_id} 已连接，当前连接数: {len(self.active_connections[client_id])}")

            # 发送连接确认消息
            await self._safe_send(websocket, json.dumps({
                "type": "connection_established",
                "client_id": client_id,
                "timestamp": datetime.now().isoformat(),
                "message": f"欢迎客户端 {client_id} 连接成功"
            }, ensure_ascii=False), client_id)

        except Exception as e:
            error_msg = f"WebSocket连接失败: {str(e)}"
            logger.error(error_msg)
            self._record_connection_error(client_id, error_msg)
            # 重新抛出异常以便上层处理
            raise

    def disconnect(self, websocket, client_id: str):
        """
        断开WebSocket连接
        """
        try:
            disconnected = False
            if client_id in self.active_connections:
                if websocket in self.active_connections[client_id]:
                    self.active_connections[client_id].discard(websocket)
                    disconnected = True
                if not self.active_connections[client_id]:
                    del self.active_connections[client_id]

                if client_id in self.client_connections:
                    del self.client_connections[client_id]

            if disconnected:
                logger.info(f"客户端 {client_id} 已断开连接")
            else:
                logger.debug(f"客户端 {client_id} 连接已不存在，无需断开")

        except Exception as e:
            error_msg = f"WebSocket断开连接失败: {str(e)}"
            logger.error(error_msg)
            self._record_connection_error(client_id, error_msg)

    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        """
        向特定客户端发送消息
        """
        try:
            if client_id in self.client_connections:
                websocket = self.client_connections[client_id]
                await self._safe_send(websocket, json.dumps(message, ensure_ascii=False), client_id)
                logger.debug(f"向客户端 {client_id} 发送消息: {message}")
        except Exception as e:
            error_msg = f"向客户端 {client_id} 发送消息失败: {str(e)}"
            logger.error(error_msg)
            self._record_connection_error(client_id, error_msg)
            # 如果发送失败，尝试清理连接
            if client_id in self.client_connections:
                websocket = self.client_connections[client_id]
                self.disconnect(websocket, client_id)

    async def broadcast(self, message: Dict[str, Any]):
        """
        向所有连接的客户端广播消息
        """
        try:
            if not self.active_connections:
                logger.debug("没有活跃的WebSocket连接")
                return

            disconnected_clients = []
            failed_clients = []

            # 向所有客户端发送消息
            for client_id, connections in self.active_connections.items():
                try:
                    for websocket in list(connections):  # 创建副本以避免修改时迭代
                        await self._safe_send(websocket, json.dumps(message, ensure_ascii=False), client_id)
                except Exception as e:
                    error_msg = f"向客户端 {client_id} 发送消息失败: {str(e)}"
                    logger.warning(error_msg)
                    self._record_connection_error(client_id, error_msg)
                    failed_clients.append(client_id)

            # 清理完全断开的连接
            for client_id in disconnected_clients:
                if client_id in self.active_connections:
                    del self.active_connections[client_id]
                if client_id in self.client_connections:
                    del self.client_connections[client_id]

        except Exception as e:
            error_msg = f"广播消息失败: {str(e)}"
            logger.error(error_msg)
            # 记录广播错误但不抛出，避免影响主流程

    def get_active_clients(self) -> int:
        """
        获取活跃客户端数量
        """
        return len(self.active_connections)

    def get_connection_errors(self, client_id: str) -> list:
        """
        获取特定客户端的连接错误历史
        """
        return self.connection_errors.get(client_id, [])

    def clear_connection_errors(self, client_id: str):
        """
        清除特定客户端的错误历史
        """
        if client_id in self.connection_errors:
            self.connection_errors[client_id].clear()

    async def _safe_send(self, websocket, message: str, client_id: str):
        """
        安全地发送消息，处理连接异常
        """
        try:
            # 检查连接是否仍然活跃
            if hasattr(websocket, 'application_state'):
                # FastAPI WebSocket
                if websocket.application_state.value != 1:  # CONNECTED state
                    logger.warning(f"客户端 {client_id} 连接已断开，准备清理")
                    self.disconnect(websocket, client_id)
                    return
            else:
                # 其他类型的WebSocket实现
                if not hasattr(websocket, 'send_text'):
                    raise ValueError("WebSocket对象不支持send_text方法")

            await websocket.send_text(message)
        except Exception as e:
            error_msg = f"向客户端 {client_id} 发送消息时发生异常: {str(e)}"
            logger.error(error_msg)
            self._record_connection_error(client_id, error_msg)
            # 标记连接已断开并清理
            self.disconnect(websocket, client_id)
            raise

    def _record_connection_error(self, client_id: str, error_msg: str):
        """
        记录连接错误信息
        """
        try:
            if client_id not in self.connection_errors:
                self.connection_errors[client_id] = []

            error_record = {
                "timestamp": datetime.now().isoformat(),
                "error": error_msg,
                "traceback": traceback.format_exc() if logger.level == logging.DEBUG else None
            }
            self.connection_errors[client_id].append(error_record)

            # 限制错误记录数量，避免内存泄漏
            if len(self.connection_errors[client_id]) > 100:
                self.connection_errors[client_id] = self.connection_errors[client_id][-50:]

        except Exception as e:
            logger.error(f"记录连接错误失败: {str(e)}")

# 创建全局WebSocket管理器实例
websocket_manager = WebSocketManager()
