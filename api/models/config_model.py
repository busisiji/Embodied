from peewee import Model, CharField, TextField, DateTimeField
from api.db.database import db
from datetime import datetime

class BaseModel(Model):
    class Meta:
        database = db

class ConfigTable(BaseModel):
    """配置表定义"""
    table_name = CharField(unique=True, max_length=100)  # 配置表名
    description = TextField(null=True)  # 表描述
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = 'config_tables'

class ConfigField(BaseModel):
    """配置表字段定义"""
    config_table = CharField(max_length=100)  # 关联的配置表名
    field_name = CharField(max_length=100)    # 字段名
    field_type = CharField(max_length=50)     # 字段类型 (string, integer, boolean, etc.)
    is_required = CharField(default='false')  # 是否必填 (true/false)
    default_value = TextField(null=True)      # 默认值
    description = TextField(null=True)        # 字段描述
    created_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = 'config_fields'
        indexes = (
            (('config_table', 'field_name'), True),  # 联合唯一索引
        )

class ConfigData(BaseModel):
    """配置数据"""
    user_id = CharField(max_length=100) # 用户ID
    config_table = CharField(max_length=100)  # 关联的配置表名
    config_key = CharField(max_length=100)    # 配置键名
    config_value = TextField()                # 配置值 (JSON格式存储)
    description = TextField(null=True)        # 配置描述
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = 'config_data'
        indexes = (
            (('config_table', 'config_key'), True),  # 联合唯一索引
        )
