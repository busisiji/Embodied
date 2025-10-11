# api/models/chess_config_model.py
# 使用 Peewee ORM 定义象棋对弈参数数据模型。

from peewee import Model, CharField, IntegerField, FloatField, ForeignKeyField, DateTimeField
from datetime import datetime

from api.models.user_model import User
from api.db.database import db

class BaseModel(Model):
    class Meta:
        database = db

class ChessGameConfig(BaseModel):
    """象棋对弈参数配置模型"""
    # 关联用户ID
    user = ForeignKeyField(User, backref='chess_configs', on_delete='CASCADE')

    # 机器人执子方
    robot_side = CharField(max_length=10, null=True)  # "red", "black", or null

    # 模型路径参数
    play_model_file = CharField(max_length=255, null=True)
    yolo_model_path = CharField(max_length=255, null=True)

    # 对弈参数
    nplayout = IntegerField(null=True)  # 蒙特卡洛树搜索次数
    cpuct = FloatField(null=True)       # MCTS探索参数

    # 识别参数
    conf = FloatField(null=True)        # 置信度阈值
    iou = FloatField(null=True)         # IOU阈值

    # 语音参数
    voice_rate = IntegerField(null=True)
    voice_volume = IntegerField(null=True)
    voice_pitch = IntegerField(null=True)

    # 时间戳
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

    def save(self, *args, **kwargs):
        # 每次保存时更新 updated_at 字段
        self.updated_at = datetime.now()
        return super(ChessGameConfig, self).save(*args, **kwargs)

    @classmethod
    def get_default_config(cls, user):
        """获取用户的默认配置，如果没有则创建一个"""
        config, created = cls.get_or_create(
            user=user,
            defaults={
                'robot_side': 'black',
                'play_model_file': None,
                'yolo_model_path': None,
                'nplayout': 400,
                'cpuct': 1.0,
                'conf': 0.45,
                'iou': 0.25,
                'voice_rate': 0,
                'voice_volume': 0,
                'voice_pitch': 0,
            }
        )
        return config

    class Meta:
        table_name = 'chess_game_configs'
        # 确保每个用户只能有一条配置记录
        indexes = (
            (('user',), True),
        )
