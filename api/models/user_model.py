# api/models/user_model.py
# 使用 Peewee ORM 定义用户数据模型。
from peewee import Model, CharField, DateTimeField
from datetime import datetime
from api.db.database import db

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
