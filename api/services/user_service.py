# api/services/user_service.py
# 这一层封装了与用户相关的业务逻辑，实现接口与数据访问的解耦。
import logging
from typing import Optional, List

from api.models.user_model import User
from api.services.auth_service import hash_password
from api.utils.exceptions import ResourceNotFoundException, InvalidInputException
from api.utils.websocket_utils import send_error_notification_sync

# 配置日志
logger = logging.getLogger(__name__)
# api/services/user_service.py
from api.utils.decorators import handle_service_exceptions

@handle_service_exceptions("user_service")
def get_all_users(limit: Optional[int] = None, offset: Optional[int] = None) -> List[User]:
    """
    获取所有用户列表，支持分页
    """
    query = User.select()
    if offset is not None:
        query = query.offset(offset)
    if limit is not None:
        query = query.limit(limit)
    return list(query)

@handle_service_exceptions("user_service")
def get_user_by_id(user_id: str):
    user = User.get_or_none(User.user_id == user_id)
    if not user:
        raise ResourceNotFoundException("User")
    return user

@handle_service_exceptions("user_service")
def search_users(name: Optional[str] = None, permission: Optional[str] = None) -> List[User]:
    """
    根据条件搜索用户
    """
    query = User.select()

    if name:
        query = query.where(User.name.contains(name))

    if permission:
        query = query.where(User.permission == permission)

    return list(query)

@handle_service_exceptions("user_service")
def create_user(user_id: str, name: str, permission: str = "student"):
    if not user_id or not name:
        raise InvalidInputException("User ID and name are required")
    return User.create(
        user_id=user_id,
        name=name,
        permission=permission
    )

@handle_service_exceptions("user_service")
def update_user(user_id: str, name: str = None, password: str = None):
    """
    更新用户信息

    Args:
        user_id: 用户ID
        name: 新用户名（可选）
        password: 新密码（可选）

    Returns:
        更新后的用户对象
    """
    try:
        # 获取用户
        user = User.get_or_none(User.user_id == user_id)
        if not user:
            raise ResourceNotFoundException("User")

        # 更新用户名（如果提供）
        if name is not None:
            user.name = name

        # 更新密码（如果提供）
        if password is not None:
            # 对新密码进行哈希处理
            hashed_password, salt = hash_password(password)
            user.password = hashed_password + ":" + salt

        # 保存更新
        user.save()

        return user
    except Exception as e:
        logger.error(f"更新用户失败: {str(e)}")
        send_error_notification_sync("user_service", None, f"更新用户失败: {str(e)}")
        raise


@handle_service_exceptions("user_service")
def delete_user(user_id: str):
    user = get_user_by_id(user_id)
    user.delete_instance()
    return {"detail": "User deleted successfully"}
