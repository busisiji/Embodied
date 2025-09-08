# routes/db/crud.py
from typing import List, Optional, Type, TypeVar
from api.db.database import db
from peewee import Model, DoesNotExist, IntegrityError

T = TypeVar('T', bound=Model)

class DatabaseCRUD:
    """
    通用数据库CRUD操作类
    提供基础的增删改查功能
    """

    @staticmethod
    def create(model: Type[T], **kwargs) -> Optional[T]:
        """
        创建新记录

        Args:
            model: Peewee模型类
            **kwargs: 要创建的字段值

        Returns:
            创建的模型实例或None（如果失败）
        """
        try:
            with db.atomic():
                instance = model.create(**kwargs)
                return instance
        except IntegrityError as e:
            print(f"创建记录失败: {e}")
            return None

    @staticmethod
    def get_by_id(model: Type[T], record_id: int) -> Optional[T]:
        """
        根据ID获取记录

        Args:
            model: Peewee模型类
            record_id: 记录ID

        Returns:
            模型实例或None（如果不存在）
        """
        try:
            return model.get_by_id(record_id)
        except DoesNotExist:
            return None

    @staticmethod
    def get_all(model: Type[T]) -> List[T]:
        """
        获取所有记录

        Args:
            model: Peewee模型类

        Returns:
            包含所有记录的列表
        """
        try:
            return list(model.select())
        except Exception as e:
            print(f"查询所有记录失败: {e}")
            return []

    @staticmethod
    def update(model: Type[T], record_id: int, **kwargs) -> bool:
        """
        更新记录

        Args:
            model: Peewee模型类
            record_id: 要更新的记录ID
            **kwargs: 要更新的字段值

        Returns:
            更新是否成功
        """
        try:
            with db.atomic():
                query = model.update(**kwargs).where(model.id == record_id)
                return query.execute() > 0
        except IntegrityError as e:
            print(f"更新记录失败: {e}")
            return False

    @staticmethod
    def delete(model: Type[T], record_id: int) -> bool:
        """
        删除记录

        Args:
            model: Peewee模型类
            record_id: 要删除的记录ID

        Returns:
            删除是否成功
        """
        try:
            with db.atomic():
                query = model.delete().where(model.id == record_id)
                return query.execute() > 0
        except Exception as e:
            print(f"删除记录失败: {e}")
            return False

    @staticmethod
    def get_by_field(model: Type[T], field_name: str, field_value) -> Optional[T]:
        """
        根据字段值获取记录

        Args:
            model: Peewee模型类
            field_name: 字段名
            field_value: 字段值

        Returns:
            匹配的模型实例或None
        """
        try:
            field = getattr(model, field_name)
            return model.get(field == field_value)
        except (DoesNotExist, AttributeError):
            return None

    @staticmethod
    def filter_by_field(model: Type[T], field_name: str, field_value) -> List[T]:
        """
        根据字段值筛选记录

        Args:
            model: Peewee模型类
            field_name: 字段名
            field_value: 字段值

        Returns:
            匹配的模型实例列表
        """
        try:
            field = getattr(model, field_name)
            return list(model.select().where(field == field_value))
        except AttributeError:
            return []

# 使用示例（以User模型为例）:
"""
from routes.models.user import User
from routes.db.crud import DatabaseCRUD

# 创建用户
user = DatabaseCRUD.create(User, username="testuser", email="test@example.com")

# 获取用户
user = DatabaseCRUD.get_by_id(User, 1)

# 获取所有用户
users = DatabaseCRUD.get_all(User)

# 根据用户名获取用户
user = DatabaseCRUD.get_by_field(User, "username", "testuser")

# 更新用户
success = DatabaseCRUD.update(User, 1, email="newemail@example.com")

# 删除用户
success = DatabaseCRUD.delete(User, 1)
"""
