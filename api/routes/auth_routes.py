# api/routes/auth_routes.py
from fastapi import APIRouter, status, Response, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from api.models.user_model import User
from api.services.auth_service import register_user, login_user, logout_user, get_current_user_from_request, \
    get_session_user, get_user_tokens, destroy_session, logout_user_all_sessions, sessions, get_all_active_sessions
from api.utils import logger
from api.utils.decorators import format_response, handle_route_exceptions
from api.utils.response_utils import format_response_data

router = APIRouter(prefix="/auth", tags=["认证"])

class UserRegister(BaseModel):
    username: str
    password: str
    permission: str = "student"

class UserLogin(BaseModel):
    username: str
    password: str

class TokenData(BaseModel):
    token: str

@router.post("/register", status_code=status.HTTP_201_CREATED)
@handle_route_exceptions("auth_service")
@format_response()
def register(user: UserRegister):
    """
    用户注册（注册后需要登录才能使用其他API）
    """
    result = register_user(
        name=user.username,
        password=user.password,
        permission=user.permission
    )
    result = result["user"]
    return format_response_data([{
        "user_id": result.get("user_id") or result.get("id"),
        "username": result.get("name"),
        "permission": result.get("permission"),
    }])

@router.post("/login", status_code=status.HTTP_200_OK)
@handle_route_exceptions("auth_service")
@format_response()
def login(user: UserLogin):
    """
    用户登录
    """
    result = login_user(
        name=user.username,
        password=user.password
    )
    
    # 返回token而不是设置cookie
    user_info = result["user"]
    return format_response_data([{
        "token": result["token"],
        "user_id": user_info.get("user_id") or user_info.get("id"),
        "username": user_info.get("name"),
        "permission": user_info.get("permission")
    }])




# 添加用户登出自己的会话接口
@router.post("/logout/me", status_code=status.HTTP_200_OK)
@handle_route_exceptions("auth_service")
@format_response()
def logout_current_user(current_user: User = Depends(get_current_user_from_request)):
    """
    当前用户登出所有会话
    """
    # 获取用户的所有token
    user_tokens = get_user_tokens(current_user.user_id)

    # 登出所有会话
    logout_count = 0
    for token in user_tokens:
        if destroy_session(token):
            logout_count += 1

# 添加登出所有用户会话的端点
@router.post("/logout/all", status_code=status.HTTP_200_OK)
@handle_route_exceptions("auth_service")
@format_response()
def logout_all_users(current_user: User = Depends(get_current_user_from_request)):
    """
    登出所有用户的所有会话（仅管理员）
    """
    # 检查权限：只有管理员可以执行此操作
    if current_user.permission != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有管理员可以登出所有用户",
        )

    try:
        # 获取所有活动会话的数量
        active_sessions = get_all_active_sessions()
        session_count = len(active_sessions)

        # 销毁所有会话
        cleared_count = 0
        for token in list(sessions.keys()):  # 使用list()避免在迭代时修改字典
            if destroy_session(token):
                cleared_count += 1

        return format_response_data({
            "message": f"成功登出所有用户，共清理 {cleared_count} 个会话",
            "cleared_sessions": cleared_count,
            "total_sessions": session_count
        })
    except Exception as e:
        error_msg = f"登出所有用户失败: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登出所有用户时发生错误",
        )

# 添加管理员强制登出用户所有会话的接口
@router.post("/logout/{user_id}", status_code=status.HTTP_200_OK)
@handle_route_exceptions("auth_service")
@format_response()
def logout_user_all(user_id: str, current_user: User = Depends(get_current_user_from_request)):
    """
    管理员强制登出用户的所有会话
    """
    # 检查权限：只有管理员可以强制登出其他用户
    if current_user.permission != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有管理员可以强制登出用户",
        )

    # 检查用户是否存在
    user = User.get_or_none(User.user_id == user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在",
        )

    # 登出用户的所有会话
    result = logout_user_all_sessions(user_id)
    return format_response_data([result])

