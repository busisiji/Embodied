# api/models/database_model.py
import os

from api.models.chess_file_models import DataFile, ModelFile
from api.db.database import db
from datetime import datetime
import logging

# 导入WebSocket管理器
from api.utils.websocket_utils import send_error_notification_sync

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        """初始化数据库管理器"""
        self.init_database()


    def init_database(self):
        """初始化数据库和表结构"""
        try:
            with db:
                db.create_tables([DataFile, ModelFile], safe=True)
        except Exception as e:
            error_msg = f"初始化数据库失败: {str(e)}"
            logger.error(error_msg)
            send_error_notification_sync("database", None, error_msg)
            raise Exception(error_msg)


    def add_data_file(self, user_id, game_count, data_length, file_path, file_type="collect"):
        """添加数据文件记录"""
        try:
            data_file = DataFile.create(
                user_id=user_id,
                game_count=game_count,
                data_length=data_length,
                file_path=file_path,
                type=file_type,  # 新增字段
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            return data_file.id
        except Exception as e:
            error_msg = f"添加数据文件记录失败: {str(e)}"
            logger.error(error_msg)
            send_error_notification_sync("database",None, error_msg)
            raise Exception(error_msg)

    def update_data_file(self, data_id, game_count=None, data_length=None):
        """更新数据文件记录"""
        try:
            data_file = DataFile.get_by_id(data_id)
            if game_count is not None:
                data_file.game_count = game_count
            if data_length is not None:
                data_file.data_length = data_length
            data_file.updated_at = datetime.now()
            data_file.save()
        except DataFile.DoesNotExist:
            error_msg = f"数据文件记录 {data_id} 不存在"
            logger.error(error_msg)
            send_error_notification_sync("database", None, error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"更新数据文件记录失败: {str(e)}"
            logger.error(error_msg)
            send_error_notification_sync("database", None, error_msg)
            raise Exception(error_msg)

    def add_model_file(self, user_id, training_epochs, file_path, file_type="models"):
        """添加模型文件记录"""
        try:
            model_file = ModelFile.create(
                user_id=user_id,
                training_epochs=training_epochs,
                file_path=file_path,
                type=file_type,  # 新增字段
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            return model_file.id
        except Exception as e:
            error_msg = f"添加模型文件记录失败: {str(e)}"
            logger.error(error_msg)
            send_error_notification_sync("database", None, error_msg)
            raise Exception(error_msg)

    def get_latest_data_file(self, user_id):
        """获取最新的数据文件"""
        try:
            data_file = DataFile.get_latest_for_user(user_id)
            if data_file:
                return {
                    'id': data_file.id,
                    'user_id': data_file.user_id,
                    'game_count': data_file.game_count,
                    'data_length': data_file.data_length,
                    'file_path': data_file.file_path,
                    'type': data_file.type,  # 新增字段
                    'created_at': data_file.created_at.isoformat() if hasattr(data_file.created_at, 'isoformat') else str(data_file.created_at),
                    'updated_at': data_file.updated_at.isoformat() if hasattr(data_file.updated_at, 'isoformat') else str(data_file.updated_at)
                }
            return None
        except Exception as e:
            error_msg = f"获取最新数据文件失败: {str(e)}"
            logger.error(error_msg)
            send_error_notification_sync("database", None, error_msg)
            raise Exception(error_msg)

    def get_latest_model_file(self, user_id):
        """获取最新的模型文件"""
        try:
            model_file = ModelFile.get_latest_for_user(user_id)
            if model_file:
                return {
                    'id': model_file.id,
                    'user_id': model_file.user_id,
                    'training_epochs': model_file.training_epochs,
                    'file_path': model_file.file_path,
                    'type': model_file.type,  # 新增字段
                    'created_at': model_file.created_at.isoformat() if hasattr(model_file.created_at, 'isoformat') else str(model_file.created_at),
                    'updated_at': model_file.updated_at.isoformat() if hasattr(model_file.updated_at, 'isoformat') else str(model_file.updated_at)
                }
            return None
        except Exception as e:
            error_msg = f"获取最新模型文件失败: {str(e)}"
            logger.error(error_msg)
            send_error_notification_sync("database", None, error_msg)
            raise Exception(error_msg)

    def get_all_data_files(self, user_id):
        """获取用户的所有数据文件记录"""
        try:
            data_files = DataFile.get_all_for_user(user_id)
            return [
                {
                    'id': data_file.id,
                    'user_id': data_file.user_id,
                    'game_count': data_file.game_count,
                    'data_length': data_file.data_length,
                    'file_path': data_file.file_path,
                    'file_name': data_file.name if hasattr(data_file, 'file_name') else (os.path.basename(data_file.file_path) if data_file.file_path else ''),  # 添加 name 字段
                    'type': data_file.type,  # 新增字段
                    'created_at': data_file.created_at.isoformat() if hasattr(data_file.created_at, 'isoformat') else str(data_file.created_at),
                    'updated_at': data_file.updated_at.isoformat() if hasattr(data_file.updated_at, 'isoformat') else str(data_file.updated_at)
                }
                for data_file in data_files
            ]
        except Exception as e:
            error_msg = f"获取用户数据文件列表失败: {str(e)}"
            logger.error(error_msg)
            send_error_notification_sync("database", None, error_msg)
            raise Exception(error_msg)

    def get_all_model_files(self, user_id):
        """获取用户的所有模型文件记录"""
        try:
            model_files = ModelFile.get_all_for_user(user_id)
            return [
                {
                    'id': model_file.id,
                    'user_id': model_file.user_id,
                    'training_epochs': model_file.training_epochs,
                    'file_path': model_file.file_path,
                    'file_name': model_file.name if hasattr(model_file, 'file_name') else (os.path.basename(model_file.file_path) if model_file.file_path else ''),  # 添加 name 字段
                    'type': model_file.type,  # 新增字段
                    'created_at': model_file.created_at.isoformat() if hasattr(model_file.created_at, 'isoformat') else str(model_file.created_at),
                    'updated_at': model_file.updated_at.isoformat() if hasattr(model_file.updated_at, 'isoformat') else str(model_file.updated_at)
                }
                for model_file in model_files
            ]
        except Exception as e:
            error_msg = f"获取用户模型文件列表失败: {str(e)}"
            logger.error(error_msg)
            send_error_notification_sync("database", None, error_msg)
            raise Exception(error_msg)
