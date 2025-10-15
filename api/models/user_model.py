# api/models/user_model.py
from peewee import Model, CharField, DateTimeField
from datetime import datetime
from api.db.database import db

def get_admin_user_id():
    """
    获取admin用户的ID
    """
    try:
        # 方法1: 通过用户名查找admin用户
        # admin_user = User.get(User.name == 'admin')
        # return admin_user.user_id
        return "user_1759917156490"
    except User.DoesNotExist:
        # 如果admin用户不存在，可能需要先创建
        return 'admin'

class BaseModel(Model):
    class Meta:
        database = db

class User(BaseModel):
    user_id = CharField(unique=True, max_length=100)
    name = CharField(max_length=50)
    # 使用 choices 限制权限值
    PERMISSION_CHOICES = [
        ('student', 'Student'),
        ('teacher', 'Teacher'),
        ('admin', 'Administrator'),
    ]
    permission = CharField(max_length=10, choices=PERMISSION_CHOICES, default='student')
    password = CharField(max_length=512)  # 存储加密后的密码和盐值
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

    def save(self, *args, **kwargs):
        # 每次保存时更新 updated_at 字段
        self.updated_at = datetime.now()
        return super(User, self).save(*args, **kwargs)

    class Meta:
        table_name = 'users'
