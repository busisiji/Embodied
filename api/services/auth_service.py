# api/services/auth_service.py
import hashlib
import secrets
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from fastapi import Request, HTTPException, status, Query, Header
from fastapi.security import HTTPBearer
from google.oauth2.id_token import verify_token

from api.models.user_model import User
from api.utils.exceptions import ResourceNotFoundException, InvalidInputException
from api.utils.websocket_utils import send_error_notification_sync
from test.user_api import get_user_by_id

# 配置日志
logger = logging.getLogger(__name__)

# 简单的内存会话存储（生产环境中应使用Redis等）
sessions = {}

# 添加安全验证方案
security = HTTPBearer()

class AuthError:
    """认证错误标识类"""
    def __init__(self, message: str):
        self.message = message

    def __bool__(self):
        # 使该对象在布尔上下文中为False
        return False

# 创建全局的认证错误实例
MISSING_TOKEN = AuthError("缺少认证令牌")
INVALID_TOKEN = AuthError("无效或过期的会话令牌")

def hash_password(password: str, salt: str = None) -> tuple:
    """
    对密码进行哈希处理

    Args:
        password: 原始密码
        salt: 盐值，如果未提供则生成新的

    Returns:
        (hashed_password, salt) 元组
    """
    if salt is None:
        salt = secrets.token_hex(16)

    # 使用SHA-256和盐值对密码进行哈希
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return pwdhash.hex(), salt

def verify_password(stored_password: str, salt: str, provided_password: str) -> bool:
    """
    验证提供的密码是否与存储的密码匹配

    Args:
        stored_password: 存储的哈希密码
        salt: 盐值
        provided_password: 提供的密码

    Returns:
        bool: 密码是否匹配
    """
    pwdhash, _ = hash_password(provided_password, salt)
    return pwdhash == stored_password

def generate_session_token() -> str:
    """
    生成会话令牌

    Returns:
        str: 会话令牌
    """
    return secrets.token_urlsafe(32)

def create_session(user_id: str) -> str:
    """
    为用户创建会话

    Args:
        user_id: 用户ID

    Returns:
        str: 会话令牌
    """
    token = generate_session_token()
    sessions[token] = {
        "user_id": user_id,
        "created_at": datetime.now(),
        "expires_at": datetime.now() + timedelta(days=30)  # 30天过期
    }
    return token

def get_session_user(token: str) -> Optional[Dict[str, Any]]:
    """
    根据会话令牌获取用户信息

    Args:
        token: 会话令牌

    Returns:
        Optional[Dict]: 用户信息或None
    """
    session = sessions.get(token)
    if not session:
        return None

    # 检查是否过期
    if datetime.now() > session["expires_at"]:
        # 删除过期会话
        del sessions[token]
        return None

    return session

def destroy_session(token: str) -> bool:
    """
    销毁会话

    Args:
        token: 会话令牌

    Returns:
        bool: 是否成功销毁
    """
    if token in sessions:
        del sessions[token]
        return True
    return False

def register_user(name: str, password: str, permission: str = "student") -> Dict[str, Any]:
    """
    注册新用户

    Args:
        name: 用户名
        password: 密码
        permission: 权限

    Returns:
        Dict: 用户信息和会话令牌
    """
    try:
        if not name or not password:
            error_msg = "用户名和密码为必填项"
            logger.warning(error_msg)
            send_error_notification_sync("auth_service", None, error_msg)
            raise InvalidInputException("Name and password are required")

        # 生成用户ID (user_ + 时间戳)
        user_id = f"user_{int(datetime.now().timestamp() * 1000)}"

        # 检查用户是否已存在
        if User.get_or_none(User.name == name):
            error_msg = f"用户名 {name} 已存在"
            logger.warning(error_msg)
            send_error_notification_sync("auth_service", None, error_msg)
            raise InvalidInputException(f"Username {name} already exists")

        # 哈希密码
        hashed_password, salt = hash_password(password)

        # 创建用户
        user = User.create(
            user_id=user_id,
            name=name,
            permission=permission,
            password=hashed_password + ":" + salt  # 存储哈希密码和盐值
        )

        # 创建会话
        token = create_session(user_id)

        return {
            "user": {
                "user_id": user.user_id,
                "name": user.name,
                "permission": user.permission
            },
            "token": token
        }
    except InvalidInputException:
        raise
    except Exception as e:
        error_msg = f"注册用户失败: {str(e)}"
        logger.error(error_msg)
        send_error_notification_sync("auth_service", None, error_msg)
        raise

def login_user(name: str, password: str) -> Dict[str, Any]:
    """
    用户登录

    Args:
        name: 用户名
        password: 密码

    Returns:
        Dict: 用户信息和会话令牌
    """
    try:
        if not name or not password:
            error_msg = "用户名和密码为必填项"
            logger.warning(error_msg)
            send_error_notification_sync("auth_service", None, error_msg)
            raise InvalidInputException("Name and password are required")

        # 查找用户
        user = User.get_or_none(User.name == name)
        if not user:
            error_msg = f"用户 {name} 不存在"
            logger.warning(error_msg)
            send_error_notification_sync("auth_service", None, error_msg)
            raise ResourceNotFoundException("User")

        # 验证密码
        if ":" in user.password:
            stored_password, salt = user.password.split(":")
            if not verify_password(stored_password, salt, password):
                error_msg = f"密码错误"
                logger.warning(error_msg)
                send_error_notification_sync("auth_service", None, error_msg)
                raise InvalidInputException("Invalid password")
        else:
            # 兼容旧数据格式
            error_msg = f"用户密码格式错误"
            logger.error(error_msg)
            send_error_notification_sync("auth_service", None, error_msg)
            raise InvalidInputException("Invalid password format")

        # 创建会话
        token = create_session(user.user_id)

        return {
            "user": {
                "user_id": user.user_id,
                "name": user.name,
                "permission": user.permission
            },
            "token": token
        }
    except (ResourceNotFoundException, InvalidInputException):
        raise
    except Exception as e:
        error_msg = f"用户登录失败: {str(e)}"
        logger.error(error_msg)
        send_error_notification_sync("auth_service", None, error_msg)
        raise

def logout_user(token: str) -> Dict[str, str]:
    """
    用户登出

    Args:
        token: 会话令牌

    Returns:
        Dict: 操作结果
    """
    try:
        if destroy_session(token):
            return {"message": "成功登出"}
        else:
            return {"message": "会话不存在或已过期"}
    except Exception as e:
        error_msg = f"用户登出失败: {str(e)}"
        logger.error(error_msg)
        send_error_notification_sync("auth_service", None, error_msg)
        raise



def get_current_user(token: str):
    """
    获取当前登录用户

    Args:
        token: 会话令牌

    Returns:
        用户对象
    """
    user_session = get_session_user(token)
    if not user_session:
        return None

    try:
        user = User.get_or_none(User.user_id == user_session["user_id"])
        return user
    except Exception as e:
        logger.error(f"获取用户信息失败: {str(e)}")
        return None

# def get_current_user_from_request(request: Request):
#     """
#     从请求中获取当前用户（依赖函数）
#     """
#     # 首先尝试从Cookie获取token
#     token = request.cookies.get("session_token")
#
#     # 如果Cookie中没有token，则从查询参数获取
#     if not token:
#         token = request.query_params.get("token")
#
#     if not token:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="缺少认证令牌",
#         )
#
#     user = get_current_user(token)
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="无效或过期的会话令牌",
#         )
#     return user



async def get_token_from_request(request: Request) -> str:
    """
    从请求的不同位置获取token
    """
    token = None

    # 1. 首先尝试从Authorization请求头获取 (推荐方式)
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]  # 移除 "Bearer " 前缀

    # 2. 如果请求头中没有，尝试从查询参数获取
    if not token:
        token = request.query_params.get("token")

    # 3. 如果查询参数中没有，尝试从表单数据获取
    if not token:
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
            form_data = await request.form()
            token = form_data.get("token")

    # 4. 如果还没有，尝试从JSON请求体获取（注意：这会消费请求体）
    if not token:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body = await request.json()
                token = body.get("token") if isinstance(body, dict) else None
            except:
                # 如果解析JSON失败，忽略错误
                pass

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少认证令牌",
        )

    return token



async def get_current_user_from_request(request: Request):
    """
    从请求中获取token并验证用户身份
    """
    token = await get_token_from_request(request)

    user_session = get_session_user(token)
    if not user_session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效或过期的会话令牌",
        )

    try:
        user = User.get_or_none(User.user_id == user_session["user_id"])
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户不存在",
            )
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取用户信息失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="验证用户身份时发生错误",
        )


# 添加获取所有活动会话的函数
def get_all_active_sessions():
    """
    获取所有活动会话（未过期的会话）
    """
    active_sessions = {}
    current_time = datetime.now()
    for token, session in sessions.items():
        if current_time <= session["expires_at"]:
            active_sessions[token] = session
    return active_sessions

# 添加检查token是否处于登录状态的函数
def is_token_logged_in(token: str) -> bool:
    """
    检查token是否处于登录状态（存在且未过期）
    """
    session = sessions.get(token)
    if not session:
        return False

    # 检查是否过期
    if datetime.now() > session["expires_at"]:
        return False

    return True

# 添加强制删除所有过期会话的函数
def cleanup_expired_sessions():
    """
    清理所有过期会话
    """
    current_time = datetime.now()
    expired_tokens = [
        token for token, session in sessions.items()
        if current_time > session["expires_at"]
    ]

    for token in expired_tokens:
        del sessions[token]

    return len(expired_tokens)

def get_user_token(user_id: str) -> Optional[str]:
    """
    根据用户ID获取其对应的token

    Args:
        user_id: 用户ID

    Returns:
        str: 用户的认证token，如果未找到则返回None
    """
    for token, session in sessions.items():
        if session.get("user_id") == user_id:
            # 检查会话是否过期
            if datetime.now() <= session["expires_at"]:
                return token
    return None

def get_user_tokens(user_id: str) -> List[str]:
    """
    根据用户ID获取其对应的所有token（一个用户可能有多个会话）

    Args:
        user_id: 用户ID

    Returns:
        List[str]: 用户的所有认证token列表
    """
    user_tokens = []
    for token, session in sessions.items():
        if session.get("user_id") == user_id:
            # 检查会话是否过期
            if datetime.now() <= session["expires_at"]:
                user_tokens.append(token)
    return user_tokens


def logout_user_all_sessions(user_id: str) -> Dict[str, str]:
    """
    登出用户的所有会话

    Args:
        user_id: 用户ID

    Returns:
        Dict: 操作结果
    """
    try:
        # 获取用户的所有token
        user_tokens = get_user_tokens(user_id)

        # 登出所有会话
        logout_count = 0
        for token in user_tokens:
            if destroy_session(token):
                logout_count += 1

        return {"message": f"成功登出 {logout_count} 个会话"}
    except Exception as e:
        error_msg = f"用户登出所有会话失败: {str(e)}"
        logger.error(error_msg)
        send_error_notification_sync("auth_service", None, error_msg)
        raise
