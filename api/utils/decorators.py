# api/utils/decorators.py
import asyncio
import functools
import logging
from typing import Optional, Any
from datetime import datetime

from pydantic import ValidationError
from fastapi import HTTPException as FastAPIHTTPException
from starlette.responses import JSONResponse

from api.utils.websocket_utils import send_error_notification_sync
from api.services.auth_service import AuthError

logger = logging.getLogger(__name__)

def handle_service_exceptions(process_type: str = "system", include_websocket_notify: bool = False):
    """
    统一服务层异常处理装饰器

    Args:
        process_type: 进程类型，用于WebSocket通知分类
        include_websocket_notify: 是否发送WebSocket通知
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 记录错误日志
                error_msg = f"执行 {func.__name__} 失败: {str(e)}"
                logger.error(error_msg, exc_info=True)

                # 发送WebSocket错误通知（如果启用）
                if include_websocket_notify:
                    try:
                        send_error_notification_sync(process_type,  error_msg)
                    except Exception as ws_error:
                        logger.warning(f"发送WebSocket通知失败: {str(ws_error)}")

                # 重新抛出异常让上层处理
                raise
        return wrapper
    return decorator


def handle_route_exceptions(process_type: str = "api"):
    """
    统一路由层异常处理装饰器（已禁用）

    Args:
        process_type: 进程类型，用于WebSocket通知分类
    """
    def decorator(func):
        # 直接返回原函数，不添加任何处理逻辑
        return func
    return decorator


def format_response():
    """
    统一响应格式处理装饰器
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            return _format_response_result(result)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return _format_response_result(result)

        # 根据函数是否为异步函数选择包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

async def _handle_result(result, func, process_type, **kwargs):
    """处理异步函数结果"""
    return _process_result_common(result, func, process_type, **kwargs)

def _handle_result_sync(result, func, process_type, **kwargs):
    """处理同步函数结果"""
    return _process_result_common(result, func, process_type, **kwargs)

def _process_result_common(result, func, process_type, **kwargs):
    """处理结果的公共逻辑"""
    # 检查是否有AuthError参数
    for arg_name, arg_value in kwargs.items():
        if 'user' in arg_name.lower() and isinstance(arg_value, AuthError):
            # 记录错误日志
            error_msg = arg_value.message
            logger.error(error_msg)

            # 发送WebSocket错误通知
            try:
                send_error_notification_sync(process_type,  error_msg)
            except Exception as ws_error:
                logger.warning(f"发送WebSocket通知失败: {str(ws_error)}")

            # 返回统一格式的错误响应
            return JSONResponse(
                status_code=401,
                content={
                    "message": error_msg,
                    "data": None,
                    "timestamp": datetime.now().isoformat(),
                    "success": False
                }
            )

    # 检查结果是否为错误响应
    if isinstance(result, dict) and result.get('code', 200) != 200:
        result['success'] = False
        if 'timestamp' not in result:
            result['timestamp'] = datetime.now().isoformat()
        return JSONResponse(
            status_code=result['code'] if result['code'] != 200 else 200,
            content=result
        )

    return result

def _handle_exception(e, func, process_type):
    """处理异常的公共逻辑"""
    if isinstance(e, ValidationError):
        # 处理Pydantic验证错误
        error_msg = f"请求参数验证失败: {str(e)}"
        logger.error(error_msg, exc_info=True)

        # 发送WebSocket错误通知
        try:
            send_error_notification_sync(process_type,  error_msg)
        except Exception as ws_error:
            logger.warning(f"发送WebSocket通知失败: {str(ws_error)}")

        # 返回统一格式的错误响应
        return JSONResponse(
            status_code=422,
            content={
                "message": "请求参数验证失败",
                "data": None,
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
        )
    elif isinstance(e, FastAPIHTTPException):
        # 记录错误日志
        error_msg = f"路由 {func.__name__} 执行失败: {str(e.detail)}"
        logger.error(error_msg)

        # 发送WebSocket错误通知
        try:
            send_error_notification_sync(process_type,  error_msg)
        except Exception as ws_error:
            logger.warning(f"发送WebSocket通知失败: {str(ws_error)}")

        # 返回统一格式的错误响应
        return JSONResponse(
            status_code=e.status_code,
            content={
                "message": str(e.detail),
                "data": None,
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
        )
    else:
        # 记录错误日志
        error_msg = f"路由 {func.__name__} 执行失败: {str(e)}"
        logger.error(error_msg, exc_info=True)

        # 发送WebSocket错误通知
        try:
            send_error_notification_sync(process_type,  error_msg)
        except Exception as ws_error:
            logger.warning(f"发送WebSocket通知失败: {str(ws_error)}")

        # 返回统一格式的错误响应
        return JSONResponse(
            status_code=500,
            content={
                "message": "Internal server error",
                "data": None,
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
        )


def _format_response_result(result):
    """
    格式化响应结果
    """
    # 如果已经是字典且包含data字段，则认为已经是格式化过的响应
    if isinstance(result, dict) and 'message' in result:
        # 添加时间戳字段（如果还没有的话）
        if 'timestamp' not in result:
            result['timestamp'] = datetime.now().isoformat()
        # 添加success字段
        if 'success' not in result:
            result['success'] = result.get('code', 200) == 200
        if 'data' not in result:
            result['data'] = None
        return result
    # 否则包装成统一格式
    return {
        "message": "success",
        "data": result,
        "timestamp": datetime.now().isoformat(),
        "success": True
    }
