# api/models/chess_file_models.py
from peewee import *
from api.db.database import db
from datetime import datetime

class DataFile(db.Model):
    """数据文件模型"""
    user_id = CharField(max_length=50, help_text="用户ID")
    game_count = IntegerField(default=0, help_text="对弈局数")
    data_length = IntegerField(default=0, help_text="数据长度")
    file_path = CharField(max_length=255, help_text="文件路径")
    type = CharField(max_length=50, default="collect", help_text="文件类型")  # 新增字段
    created_at = DateTimeField(help_text="创建时间")
    updated_at = DateTimeField(help_text="更新时间")

    class Meta:
        database = db
        table_name = "play_data_files"

    @classmethod
    def get_latest_for_user(cls, user_id):
        """获取用户最新的数据文件"""
        try:
            return cls.select().where(cls.user_id == user_id).order_by(cls.created_at.desc()).first()
        except cls.DoesNotExist:
            return None

    @classmethod
    def get_all_for_user(cls, user_id):
        """获取用户的所有数据文件"""
        try:
            return cls.select().where(cls.user_id == user_id).order_by(cls.created_at.desc())
        except cls.DoesNotExist:
            return []

class ModelFile(db.Model):
    """模型文件模型"""
    user_id = CharField(max_length=50, help_text="用户ID")
    training_epochs = IntegerField(default=0, help_text="训练轮次")
    file_path = CharField(max_length=255, help_text="文件路径")
    type = CharField(max_length=50, help_text="文件类型")
    created_at = DateTimeField(help_text="创建时间")
    updated_at = DateTimeField(help_text="更新时间")

    class Meta:
        database = db
        table_name = "play_model_files"

    @classmethod
    def get_latest_for_user(cls, user_id):
        """获取用户最新的模型文件"""
        try:
            return cls.select().where(cls.user_id == user_id).order_by(cls.created_at.desc()).first()
        except cls.DoesNotExist:
            return None

    @classmethod
    def get_all_for_user(cls, user_id):
        """获取用户的所有模型文件"""
        try:
            return cls.select().where(cls.user_id == user_id).order_by(cls.created_at.desc())
        except cls.DoesNotExist:
            return []
