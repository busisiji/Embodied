# api/routes/user_routes.py
# 该模块定义了 /users/ 路由下的所有 RESTful 接口，使用标准 HTTP 方法设计。
from typing import Optional
from fastapi import APIRouter, HTTPException, status, Query, Request, Depends

from pydantic import BaseModel

from api.models.user_model import User
from api.services.auth_service import get_current_user_from_request, destroy_session, get_all_active_sessions, \
    get_user_token, get_user_tokens
from api.services.user_service import (
    get_user_by_id,
    create_user,
    update_user,
    delete_user, search_users, get_all_users
)
from api.utils import logger
from api.utils.decorators import handle_route_exceptions, format_response
from api.utils.response_utils import format_response_data

# 定义数据模型
class UserCreate(BaseModel):
    name: str
    permission: str = "student"

class UserUpdate(BaseModel):
    name: Optional[str] = None
    password: Optional[str] = None
    permission: str = "student"  # 保留但不使用于普通用户更新


router = APIRouter(prefix="/users", tags=["Users"])

def remove_sensitive_fields(user_data):
    """
    移除用户数据中的敏感字段
    """
    if isinstance(user_data, dict):
        # 创建副本避免修改原始数据
        safe_data = user_data.copy()
        # 移除敏感字段
        sensitive_fields = ['user_id', 'password']
        for field in sensitive_fields:
            safe_data.pop(field, None)
        return safe_data
    return user_data

# 在函数中使用
@router.get("/")
@handle_route_exceptions("user_service")
@format_response()
def read_users(current_user: User = Depends(get_current_user_from_request)):
    """
    获取所有用户列表（需要登录）
    """
    # 只有管理员可以查看所有用户
    if current_user.permission != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有管理员可以查看所有用户",
        )
    users = get_all_users()
    return format_response_data(users, remove_sensitive_fields)

@router.get("/me")
@handle_route_exceptions("user_service")
@format_response()
def read_current_user(current_user: User = Depends(get_current_user_from_request)):
    """
    获取当前登录用户信息
    """
    if hasattr(current_user, '__data__'):
        user_data = current_user.__data__
    elif isinstance(current_user, dict):
        user_data = current_user
    elif hasattr(current_user, '__dict__'):
        user_data = current_user.__dict__
    else:
        user_data = current_user
    # 移除敏感字段
    safe_user_data = remove_sensitive_fields(user_data)
    token = get_user_tokens(current_user.user_id)
    safe_user_data['token'] = token
    return [safe_user_data]

@router.get("/login")
@handle_route_exceptions("user_service")
@format_response()
def get_logged_in_users(current_user: User = Depends(get_current_user_from_request)):
    """
    获取所有当前登录的用户及其token（仅管理员）
    """
    if current_user.permission != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有管理员可以查看所有登录用户",
        )

    # 获取所有活动会话
    active_sessions = get_all_active_sessions()

    # 获取所有登录用户的信息
    logged_in_users = []
    for token, session in active_sessions.items():
        try:
            user = User.get_or_none(User.user_id == session["user_id"])
            if user:
                user_data = {
                    "user_id": user.user_id,
                    "name": user.name,
                    "permission": user.permission,
                    "token": token,
                    "login_time": session.get("created_at").isoformat() if session.get("created_at") else None,
                    "expires_at": session.get("expires_at").isoformat() if session.get("expires_at") else None
                }
                logged_in_users.append(user_data)
        except Exception as e:
            logger.error(f"获取用户信息失败: {str(e)}")
            continue

    return logged_in_users
@router.put("/me")
@handle_route_exceptions("user_service")
@format_response()
def update_current_user(user: UserUpdate, current_user: User = Depends(get_current_user_from_request)):
    """
    更新当前登录用户信息（仅限名称和密码）
    """
    # 准备更新参数
    update_params = {
        "user_id": current_user.user_id,
        "name": user.name
    }

    # 如果提供了密码，则包含在更新参数中
    if user.password:
        update_params["password"] = user.password

    # 调用更新函数（服务层应该负责密码加密）
    current_user = update_user(**update_params)
    return format_response_data(current_user, remove_sensitive_fields)

@router.delete("/me")
@handle_route_exceptions("user_service")
@format_response()
def delete_current_user(current_user: User = Depends(get_current_user_from_request)):
    """
    删除当前登录用户
    """
    result = delete_user(current_user.user_id)
    return [{"name": current_user.name,"permission": current_user.permission}]

