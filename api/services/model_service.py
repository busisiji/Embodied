# api/services/model_service.py
import os
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from api.models.chess_file_models import ModelFile, DataFile
from api.models.database_model import DatabaseManager
from api.models.user_model import User
from api.services.auth_service import login_user, register_user
from api.services.user_service import create_user
from api.utils.websocket_utils import send_error_notification_sync
from src.cchessAI.parameters import MODELS, DATA_SELFPLAY, MODEL_USER_PATH, DATA_USER_PATH

# 导入WebSocket管理器
from api.utils.decorators import handle_service_exceptions

# 配置日志
logger = logging.getLogger(__name__)



class ModelService:
    db_manager = DatabaseManager()  # 添加数据库管理器实例

    @staticmethod
    @handle_service_exceptions("model_service")
    def list_models() -> List[Dict[str, Any]]:
        """
        列出所有可用的模型
        """
        models = []
        if os.path.exists(MODEL_USER_PATH):
            # 使用 os.walk 遍历所有子目录
            for root, dirs, files in os.walk(MODEL_USER_PATH):
                for file in files:
                    if file.endswith((".pkl", ".onnx", ".trt")) and file != 'training_state.pkl':
                        file_path = os.path.join(root, file)
                        # 根据路径分离出 user_id 和 type 字段
                        relative_path = os.path.relpath(root, MODEL_USER_PATH)

                        if relative_path == ".":
                            # 根目录下的模型文件，user_id 设为 "system"，type 设为 "root"
                            user_id = "system"
                            file_type = "root"
                        else:
                            # 解析路径，格式应为 {user_id}\{type}
                            path_parts = relative_path.split(os.sep)
                            if len(path_parts) >= 2:
                                user_part = path_parts[0]  # 应该是 {user_id} 格式
                                user_id = user_part
                                file_type = os.sep.join(path_parts[1:])  # 剩余部分作为 type
                            elif len(path_parts) == 1:
                                # 只有用户目录
                                user_part = path_parts[0]
                                user_id = user_part
                                file_type = "root"

                        models.append({
                            "file_name": file,
                            "file_path": file_path,
                            "user_id": user_id,
                            "type": file_type,
                            "size": os.path.getsize(file_path),
                            "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                        })
        return models

    @staticmethod
    @handle_service_exceptions("model_service")
    def list_data() -> List[Dict[str, Any]]:
        """
        列出所有采集的数据文件
        """
        data_files = []
        if os.path.exists(DATA_USER_PATH):
            # 使用 os.walk 遍历所有子目录
            for root, dirs, files in os.walk(DATA_USER_PATH):
                for file in files:
                    if file.endswith(".pkl"):
                        file_path = os.path.join(root, file)
                        # 根据路径分离出 user_id 和 type 字段
                        relative_path = os.path.relpath(root, DATA_USER_PATH)

                        if relative_path == ".":
                            # 根目录下没有文件，这种情况理论上不会出现
                            user_id = "unknown"
                            file_type = "root"
                        else:
                            # 解析路径，格式应为 {user_id}\{type}
                            path_parts = relative_path.split(os.sep)
                            if len(path_parts) >= 2:
                                user_part = path_parts[0]  # 应该是{user_id} 格式
                                user_id = user_part
                                file_type = os.sep.join(path_parts[1:])  # 剩余部分作为 type
                            elif len(path_parts) == 1:
                                # 只有用户目录
                                user_part = path_parts[0]
                                user_id = user_part
                                file_type = "root"

                        data_files.append({
                            "file_name": file,
                            "file_path": file_path,
                            "user_id": user_id,
                            "type": file_type,
                            "size": os.path.getsize(file_path),
                            "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                        })
        return data_files

    # 为其他方法也添加装饰器
    @staticmethod
    @handle_service_exceptions("model_service")
    def list_user_data(user_id: str) -> List[Dict[str, Any]]:
        """
        列出指定用户的所有数据文件
        """
        data_files = []
        user_data_path = os.path.join(DATA_USER_PATH, f"{user_id}")
        if os.path.exists(user_data_path):
            # 遍历所有子目录和文件
            for root, dirs, files in os.walk(user_data_path):
                for file in files:
                    if file.endswith(".pkl"):
                        file_path = os.path.join(root, file)
                        # 获取相对路径以确定type
                        relative_path = os.path.relpath(root, user_data_path)
                        # 根据上级目录名称设置type字段
                        if relative_path == ".":
                            file_type = "root"
                        else:
                            file_type = relative_path

                        data_files.append({
                            "file_name": file,  # 添加 file_name 字段
                            "file_path": file_path,
                            "type": file_type,
                            "size": os.path.getsize(file_path),
                            "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                        })
        return data_files


    @staticmethod
    @handle_service_exceptions("model_service")
    def list_user_models(user_id: str) -> List[Dict[str, Any]]:
        """
        列出指定用户的所有模型文件
        """
        model_files = []
        user_model_path = os.path.join(MODEL_USER_PATH, f"{user_id}")
        if os.path.exists(user_model_path):
            # 遍历所有子目录和文件
            for root, dirs, files in os.walk(user_model_path):
                for file in files:
                    if file.endswith((".pkl", ".onnx", ".trt")) and file != 'training_state.pkl':
                        file_path = os.path.join(root, file)
                        # 获取相对路径以确定type
                        relative_path = os.path.relpath(root, user_model_path)
                        # 根据上级目录名称设置type字段
                        if relative_path == ".":
                            file_type = "root"
                        else:
                            file_type = relative_path

                        model_files.append({
                            "file_name": file,  # 添加 file_name 字段
                            "file_path": file_path,
                            "type": file_type,
                            "size": os.path.getsize(file_path),
                            "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                        })
        return model_files


    @staticmethod
    @handle_service_exceptions("model_service")
    def get_user_data_file(user_id: str, file_name: str) -> Dict[str, Any]:
        """
        获取指定用户的数据文件详细信息
        """
        user_data_path = os.path.join(DATA_USER_PATH, f"{user_id}", file_name)

        if not os.path.exists(user_data_path):
            error_msg = f"数据文件 {file_name} 不存在"
            logger.error(error_msg)
            send_error_notification_sync("system", None, error_msg)
            raise Exception(error_msg)

        if not user_data_path.endswith(".pkl"):
            error_msg = "文件格式不正确，只支持 .pkl 文件"
            logger.error(error_msg)
            send_error_notification_sync("system", None, error_msg)
            raise Exception(error_msg)

        # 获取文件详细信息
        file_stats = os.stat(user_data_path)

        # 尝试读取文件内容以获取对弈局数和数据长度
        game_count = 0
        data_length = 0
        try:
            import pickle
            with open(user_data_path, 'rb') as f:
                data = pickle.load(f)
                game_count = data.get('iters', 0) if isinstance(data, dict) else 0
                data_buffer = data.get('data_buffer', []) if isinstance(data, dict) else []
                data_length = len(data_buffer)
        except Exception as e:
            logger.warning(f"无法读取数据文件内容: {str(e)}")

        return {
            "file_name": file_name,  # 添加 file_name 字段
            "file_path": user_data_path,
            "size": file_stats.st_size,
            "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "game_count": game_count,
            "data_length": data_length
        }

    @staticmethod
    @handle_service_exceptions("model_service")
    def get_user_model_file(user_id: str, file_name: str) -> Dict[str, Any]:
        """
        获取指定用户的模型文件详细信息
        """
        # 根据文件后缀确定子目录
        file_extension = os.path.splitext(file_name)[1]
        if file_extension in ['.pkl', '.onnx', '.trt']:
            subdir = file_extension[1:]  # 去掉点号
        else:
            subdir = "pkl"  # 默认使用pkl目录

        user_model_path = os.path.join(MODEL_USER_PATH, f"{user_id}", subdir, file_name)

        if not os.path.exists(user_model_path):
            error_msg = f"模型文件 {file_name} 不存在"
            logger.error(error_msg)
            send_error_notification_sync("system", None, error_msg)
            raise Exception(error_msg)

        # 获取文件详细信息
        file_stats = os.stat(user_model_path)

        return {
            "file_name": file_name,  # 添加 file_name 字段
            "file_path": user_model_path,
            "size": file_stats.st_size,
            "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "extension": file_extension
        }

    @staticmethod
    @handle_service_exceptions("model_service")
    def sync_user_data_files_to_db(user_id: str) -> Dict[str, Any]:
        """
        同步用户数据文件到数据库，根据上级目录名称设置type字段
        同步包括添加新文件和删除本地不存在的文件记录
        """
        # 检查用户是否存在于数据库中
        try:
            user = User.get_or_none(User.name == user_id)
            if not user:
                error_msg = f"用户 {user_id} 不存在于数据库中"
                logger.error(error_msg)
                send_error_notification_sync("system", None, error_msg)
                raise Exception(error_msg)
        except Exception as e:
            error_msg = f"检查用户 {user_id} 存在性失败: {str(e)}"
            logger.error(error_msg)
            send_error_notification_sync("system", None, error_msg)
            raise Exception(error_msg)

        synced_count = 0
        removed_count = 0
        existing_count = 0
        user_data_path = os.path.join(DATA_USER_PATH, f"user_{user_id}")

        # 记录已存在的文件路径和新同步的文件路径
        existing_files = []
        new_synced_files = []
        removed_files = []

        if not os.path.exists(user_data_path):
            error_msg = f"用户 {user_id} 的数据目录不存在"
            logger.error(error_msg)
            send_error_notification_sync("system", None, error_msg)
            raise Exception(error_msg)

        # 获取数据库中已有的数据文件记录
        try:
            db_data_files = ModelService.db_manager.get_all_data_files(user_id)
            db_file_paths = {file['file_path']: file for file in db_data_files}
            existing_files = list(db_file_paths.keys())
            existing_count = len(existing_files)
        except Exception as e:
            logger.warning(f"获取数据库中已有的数据文件记录失败: {str(e)}")
            db_file_paths = {}

        # 收集本地所有数据文件路径
        local_file_paths = set()
        # 遍历本地文件
        for root, dirs, files in os.walk(user_data_path):
            for file in files:
                if file.endswith(".pkl"):
                    file_path = os.path.join(root, file)
                    local_file_paths.add(file_path)

        # 删除数据库中存在但本地不存在的文件记录
        for db_file_path in db_file_paths:
            if db_file_path not in local_file_paths:
                try:
                    # 从数据库中删除该文件记录
                    data_file_id = db_file_paths[db_file_path]['id']
                    DataFile.delete_by_id(data_file_id)
                    removed_files.append(db_file_path)
                    removed_count += 1
                    logger.info(f"已从数据库删除不存在的本地文件记录: {db_file_path}")
                except Exception as e:
                    error_msg = f"从数据库删除文件记录失败 {db_file_path}: {str(e)}"
                    logger.error(error_msg)
                    send_error_notification_sync("system", None, error_msg)

        # 添加本地存在但数据库中没有的文件记录
        for root, dirs, files in os.walk(user_data_path):
            for file in files:
                if file.endswith(".pkl"):
                    file_path = os.path.join(root, file)

                    # 如果文件已在数据库中，跳过
                    if file_path in db_file_paths:
                        continue
                    else:
                        # 记录新文件
                        new_synced_files.append(file_path)

                    # 根据上级目录名称确定type字段
                    parent_dir_name = os.path.basename(root)
                    file_type = parent_dir_name if parent_dir_name else "collect"

                    # 读取文件信息
                    game_count = 0
                    data_length = 0
                    try:
                        import pickle
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                            game_count = data.get('iters', 0) if isinstance(data, dict) else 0
                            data_buffer = data.get('data_buffer', []) if isinstance(data, dict) else []
                            data_length = len(data_buffer)
                    except Exception as e:
                        logger.warning(f"无法读取数据文件 {file_path} 内容: {str(e)}")

                    # 添加到数据库
                    try:
                        ModelService.db_manager.add_data_file(
                            user_id=user_id,
                            game_count=game_count,
                            data_length=data_length,
                            file_path=file_path,
                            file_type=file_type  # 根据上级目录名称设置type
                        )
                        synced_count += 1
                        logger.info(f"已同步数据文件到数据库: {file_path} (type: {file_type})")
                    except Exception as e:
                        error_msg = f"同步数据文件到数据库失败 {file_path}: {str(e)}"
                        logger.error(error_msg)
                        send_error_notification_sync("system", None, error_msg)

        return {
            "status": "success",
            "message": f"成功同步数据文件到数据库: 新增{synced_count}个, 删除{removed_count}个",
            "existing_count": existing_count,
            "synced_count": synced_count,
            "removed_count": removed_count,
            "existing_files": existing_files,
            "new_synced_files": new_synced_files,
            "removed_files": removed_files
        }

    @staticmethod
    @handle_service_exceptions("model_service")
    def sync_user_model_files_to_db(user_id: str) -> Dict[str, Any]:
        """
        同步用户模型文件到数据库
        同步包括添加新文件和删除本地不存在的文件记录
        """
        # 检查用户是否存在于数据库中
        try:
            user = User.get_or_none(User.name == user_id)
            if not user:
                error_msg = f"用户 {user_id} 不存在于数据库中"
                logger.error(error_msg)
                send_error_notification_sync("system", None, error_msg)
                raise Exception(error_msg)
        except Exception as e:
            error_msg = f"检查用户 {user_id} 存在性失败: {str(e)}"
            logger.error(error_msg)
            send_error_notification_sync("system", None, error_msg)
            raise Exception(error_msg)

        synced_count = 0
        removed_count = 0
        existing_count = 0
        user_model_path = os.path.join(MODEL_USER_PATH, f"user_{user_id}")

        # 记录已存在的文件路径和新同步的文件路径
        existing_files = []
        new_synced_files = []
        removed_files = []

        if not os.path.exists(user_model_path):
            error_msg = f"用户 {user_id} 的模型目录不存在"
            logger.error(error_msg)
            send_error_notification_sync("system", None, error_msg)
            raise Exception(error_msg)

        # 获取数据库中已有的模型文件记录
        try:
            db_model_files = ModelService.db_manager.get_all_model_files(user_id)
            db_file_paths = {file['file_path']: file for file in db_model_files}
            existing_files = list(db_file_paths.keys())
            existing_count = len(existing_files)
        except Exception as e:
            logger.warning(f"获取数据库中已有的模型文件记录失败: {str(e)}")
            db_file_paths = {}

        # 收集本地所有模型文件路径
        local_file_paths = set()
        # 遍历本地模型文件
        for root, dirs, files in os.walk(user_model_path):
            for file in files:
                if file.endswith((".pkl", ".onnx", ".trt")):
                    file_path = os.path.join(root, file)
                    local_file_paths.add(file_path)

        # 删除数据库中存在但本地不存在的文件记录
        for db_file_path in db_file_paths:
            if db_file_path not in local_file_paths:
                try:
                    # 从数据库中删除该文件记录
                    model_file_id = db_file_paths[db_file_path]['id']
                    ModelFile.delete_by_id(model_file_id)
                    removed_files.append(db_file_path)
                    removed_count += 1
                    logger.info(f"已从数据库删除不存在的本地文件记录: {db_file_path}")
                except Exception as e:
                    error_msg = f"从数据库删除文件记录失败 {db_file_path}: {str(e)}"
                    logger.error(error_msg)
                    send_error_notification_sync("system", None, error_msg)

        # 添加本地存在但数据库中没有的文件记录
        for root, dirs, files in os.walk(user_model_path):
            for file in files:
                if file.endswith((".pkl", ".onnx", ".trt")):
                    file_path = os.path.join(root, file)

                    # 如果文件已在数据库中，跳过
                    if file_path in db_file_paths:
                        continue
                    else:
                        # 记录新文件
                        new_synced_files.append(file_path)

                    # 根据上级目录名称确定type字段
                    parent_dir_name = os.path.basename(root)
                    file_type = parent_dir_name if parent_dir_name else "models"

                    # 确定训练轮次（从文件名中提取，如果没有则默认为0）
                    training_epochs = 0
                    try:
                        # 尝试从文件名中提取轮次信息
                        # 例如: model_epoch_5.pkl -> 5
                        import re
                        epoch_match = re.search(r'epoch_(\d+)', file)
                        if epoch_match:
                            training_epochs = int(epoch_match.group(1))
                    except Exception:
                        training_epochs = 0

                    # 添加到数据库
                    try:
                        ModelService.db_manager.add_model_file(
                            user_id=user_id,
                            training_epochs=training_epochs,
                            file_path=file_path,
                            file_type=file_type  # 添加type字段
                        )
                        synced_count += 1
                        logger.info(f"已同步模型文件到数据库: {file_path} (type: {file_type})")
                    except Exception as e:
                        error_msg = f"同步模型文件到数据库失败 {file_path}: {str(e)}"
                        logger.error(error_msg)
                        send_error_notification_sync("system", None, error_msg)

        return {
            "status": "success",
            "message": f"成功同步模型文件到数据库: 新增{synced_count}个, 删除{removed_count}个",
            "existing_count": existing_count,
            "synced_count": synced_count,
            "removed_count": removed_count,
            "existing_files": existing_files,
            "new_synced_files": new_synced_files,
            "removed_files": removed_files
        }


    @staticmethod
    @handle_service_exceptions("model_service")
    def sync_all_users_files_to_db() -> Dict[str, Any]:
        """
        同步所有用户的数据和模型文件到数据库（只处理数据库User表中存在的用户）
        """
        # 获取所有用户目录
        all_users_data = []
        all_users_model = []

        # 处理数据文件目录
        if os.path.exists(DATA_USER_PATH):
            for item in os.listdir(DATA_USER_PATH):
                item_path = os.path.join(DATA_USER_PATH, item)
                if os.path.isdir(item_path) and item.startswith("user_"):
                    user_id = item
                    all_users_data.append(user_id)

        # 处理模型文件目录
        if os.path.exists(MODEL_USER_PATH):
            for item in os.listdir(MODEL_USER_PATH):
                item_path = os.path.join(MODEL_USER_PATH, item)
                if os.path.isdir(item_path) and item.startswith("user_"):
                    user_id = item
                    all_users_model.append(user_id)

        # 合并所有用户ID并去重
        all_user_ids = list(set(all_users_data + all_users_model))

        # 只处理数据库User表中存在的用户
        existing_user_ids = set()
        try:
            # 获取数据库中所有存在的用户ID
            existing_users = User.select()
            existing_user_ids = {user.user_id for user in existing_users}
        except Exception as e:
            logger.error(f"获取数据库用户列表失败: {str(e)}")
            send_error_notification_sync("system", None, f"获取数据库用户列表失败: {str(e)}")
            raise Exception(f"获取数据库用户列表失败: {str(e)}")

        # 过滤出数据库中存在的用户
        valid_user_ids = [user_id for user_id in all_user_ids if user_id in existing_user_ids]

        # 记录被跳过的用户（在文件系统中存在但在数据库中不存在）
        skipped_users = [user_id for user_id in all_user_ids if user_id not in existing_user_ids]

        # 同步每个有效的用户数据
        results = []
        for user_id in valid_user_ids:
            try:
                user_result = ModelService.sync_all_userid_files_to_db(user_id)
                results.append({
                    "user_id": user_id,
                    "status": "success",
                    "result": user_result
                })
            except Exception as e:
                results.append({
                    "user_id": user_id,
                    "status": "error",
                    "error": str(e)
                })
                logger.error(f"同步用户 {user_id} 文件失败: {str(e)}")
                send_error_notification_sync("system", None, f"同步用户 {user_id} 文件失败: {str(e)}")

        return {
            "processed_count": len(valid_user_ids),
            "skipped_count": len(skipped_users),
            "skipped_users": skipped_users,
            "results": results
        }


        # 删除数据库中存在但本地不存在的文件记录
        for db_file_path in db_file_paths:
            if db_file_path not in local_file_paths:
                try:
                    # 从数据库中删除该文件记录
                    model_file_id = db_file_paths[db_file_path]['id']
                    ModelFile.delete_by_id(model_file_id)
                    removed_files.append(db_file_path)
                    removed_count += 1
                    logger.info(f"已从数据库删除不存在的本地文件记录: {db_file_path}")
                except Exception as e:
                    error_msg = f"从数据库删除文件记录失败 {db_file_path}: {str(e)}"
                    logger.error(error_msg)
                    send_error_notification_sync("system", None, error_msg)

        # 添加本地存在但数据库中没有的文件记录
        for root, dirs, files in os.walk(user_model_path):
            for file in files:
                if file.endswith((".pkl", ".onnx", ".trt")):
                    file_path = os.path.join(root, file)

                    # 如果文件已在数据库中，跳过
                    if file_path in db_file_paths:
                        continue
                    else:
                        # 记录新文件
                        new_synced_files.append(file_path)

                    # 根据上级目录名称确定type字段
                    parent_dir_name = os.path.basename(root)
                    file_type = parent_dir_name if parent_dir_name else "models"

                    # 确定训练轮次（从文件名中提取，如果没有则默认为0）
                    training_epochs = 0
                    try:
                        # 尝试从文件名中提取轮次信息
                        # 例如: model_epoch_5.pkl -> 5
                        import re
                        epoch_match = re.search(r'epoch_(\d+)', file)
                        if epoch_match:
                            training_epochs = int(epoch_match.group(1))
                    except Exception:
                        training_epochs = 0

                    # 添加到数据库
                    try:
                        ModelService.db_manager.add_model_file(
                            user_id=user_id,
                            training_epochs=training_epochs,
                            file_path=file_path,
                            file_type=file_type  # 添加type字段
                        )
                        synced_count += 1
                        logger.info(f"已同步模型文件到数据库: {file_path} (type: {file_type})")
                    except Exception as e:
                        error_msg = f"同步模型文件到数据库失败 {file_path}: {str(e)}"
                        logger.error(error_msg)
                        send_error_notification_sync("system", None, error_msg)

        return {
            "existing_count": existing_count,
            "synced_count": synced_count,
            "removed_count": removed_count,
            "existing_files": existing_files,
            "new_synced_files": new_synced_files,
            "removed_files": removed_files
        }
    @staticmethod
    @handle_service_exceptions("model_service")
    def sync_all_users_files_to_db() -> Dict[str, Any]:
        """
        同步所有用户的数据和模型文件到数据库
        """
        # 获取所有用户目录
        all_users_data = []
        all_users_model = []

        # 处理数据文件目录
        if os.path.exists(DATA_USER_PATH):
            for item in os.listdir(DATA_USER_PATH):
                item_path = os.path.join(DATA_USER_PATH, item)
                if os.path.isdir(item_path) and item.startswith("user_"):
                    user_id = item
                    all_users_data.append(user_id)

        # 处理模型文件目录
        if os.path.exists(MODEL_USER_PATH):
            for item in os.listdir(MODEL_USER_PATH):
                item_path = os.path.join(MODEL_USER_PATH, item)
                if os.path.isdir(item_path) and item.startswith("user_"):
                    user_id = item
                    all_users_model.append(user_id)

        # 合并所有用户ID
        all_user_ids = list(set(all_users_data + all_users_model))

        # 同步每个用户的数据
        results = []
        for user_id in all_user_ids:
            try:
                user_result = ModelService.sync_all_userid_files_to_db(user_id)
                results.append({
                    "user_id": user_id,
                    "status": "success",
                    "result": user_result
                })
            except Exception as e:
                results.append({
                    "user_id": user_id,
                    "status": "error",
                    "error": str(e)
                })
                logger.error(f"同步用户 {user_id} 文件失败: {str(e)}")
                send_error_notification_sync("system", None, f"同步用户 {user_id} 文件失败: {str(e)}")

        return {
            "results": results
            }

    # api/services/model_service.py

    @staticmethod
    @handle_service_exceptions("model_service")
    def sync_all_userid_files_to_db(user_id: str) -> Dict[str, Any]:
        """
        同步用户所有数据和模型文件到数据库
        """
        # 创建用户
        try:
            user_name = user_id.split("user_")[1]
            if not user_name:
                return
            result = register_user(
                name=user_name,
                password='123456',
                permission="student"
            )
        except :
            return
        # result = result["user"]

        # 同步数据文件
        data_result = ModelService.sync_user_data_files_to_db(user_name)

        # 同步模型文件
        model_result = ModelService.sync_user_model_files_to_db(user_name)

        return {
            "data_files": {
                "synced_count": data_result["synced_count"],
                "removed_count": data_result["removed_count"],
                "existing_count": data_result["existing_count"],
                "existing_files": data_result["existing_files"],
                "new_synced_files": data_result["new_synced_files"],
                "removed_files": data_result["removed_files"]
            },
            "model_files": {
                "synced_count": model_result["synced_count"],
                "removed_count": model_result["removed_count"],
                "existing_count": model_result["existing_count"],
                "existing_files": model_result["existing_files"],
                "new_synced_files": model_result["new_synced_files"],
                "removed_files": model_result["removed_files"]
            }
        }


    @staticmethod
    @handle_service_exceptions("model_service")
    def delete_data_file(user_id: str, file_name: str) -> Dict[str, Any]:
        """
        删除指定用户的数据文件（包括本地文件和数据库记录）
        """
        # 查找文件路径
        user_data_path = os.path.join(DATA_USER_PATH, f"{user_id}")
        file_path = None

        # 遍历用户目录查找文件
        for root, dirs, files in os.walk(user_data_path):
            if file_name in files:
                file_path = os.path.join(root, file_name)
                break

        if not file_path or not os.path.exists(file_path):
            error_msg = f"数据文件 {file_name} 不存在"
            logger.error(error_msg)
            send_error_notification_sync("system", None, error_msg)
            raise Exception(error_msg)

        # 删除本地文件
        try:
            os.remove(file_path)
        except Exception as e:
            error_msg = f"删除本地文件失败 {file_path}: {str(e)}"
            logger.error(error_msg)
            send_error_notification_sync("system", None, error_msg)
            raise Exception(error_msg)

        # 删除数据库记录
        try:
            # 先查找数据库中的记录
            db_data_files = ModelService.db_manager.get_all_data_files(user_id)
            for db_file in db_data_files:
                if db_file['file_path'] == file_path:
                    DataFile.delete_by_id(db_file['id'])
                    break
        except Exception as e:
            error_msg = f"删除数据库记录失败: {str(e)}"
            logger.warning(error_msg)
            send_error_notification_sync("system", None, error_msg)

        return {
            "file_path": file_path
        }

    @staticmethod
    @handle_service_exceptions("model_service")
    def delete_model_file(user_id: str, file_name: str) -> Dict[str, Any]:
        """
        删除指定用户的模型文件（包括本地文件和数据库记录）
        """
        # 根据文件后缀确定子目录
        file_extension = os.path.splitext(file_name)[1]
        if file_extension in ['.pkl', '.onnx', '.trt']:
            subdir = file_extension[1:]  # 去掉点号
        else:
            subdir = "pkl"  # 默认使用pkl目录

        file_path = os.path.join(MODEL_USER_PATH, f"{user_id}", subdir, file_name)

        if not os.path.exists(file_path):
            error_msg = f"模型文件 {file_name} 不存在"
            logger.error(error_msg)
            send_error_notification_sync("system", None, error_msg)
            raise Exception(error_msg)

        # 删除本地文件
        try:
            os.remove(file_path)
        except Exception as e:
            error_msg = f"删除本地文件失败 {file_path}: {str(e)}"
            logger.error(error_msg)
            send_error_notification_sync("system", None, error_msg)
            raise Exception(error_msg)

        # 删除数据库记录
        try:
            # 先查找数据库中的记录
            db_model_files = ModelService.db_manager.get_all_model_files(user_id)
            for db_file in db_model_files:
                if db_file['file_path'] == file_path:
                    ModelFile.delete_by_id(db_file['id'])
                    break
        except Exception as e:
            error_msg = f"删除数据库记录失败: {str(e)}"
            logger.warning(error_msg)
            send_error_notification_sync("system", None, error_msg)

        return {
            "file_path": file_path
        }

    @staticmethod
    @handle_service_exceptions("model_service")
    def update_data_file(user_id: str, file_name: str, new_file_name: str = None,
                         new_type: str = None) -> Dict[str, Any]:
        """
        更新指定用户的数据文件（重命名文件和更新数据库记录）
        """
        # 查找原文件路径
        user_data_path = os.path.join(DATA_USER_PATH, f"{user_id}")
        old_file_path = None

        # 遍历用户目录查找文件
        for root, dirs, files in os.walk(user_data_path):
            if file_name in files:
                old_file_path = os.path.join(root, file_name)
                old_root = root
                break

        if not old_file_path or not os.path.exists(old_file_path):
            error_msg = f"数据文件 {file_name} 不存在"
            logger.error(error_msg)
            send_error_notification_sync("system", None, error_msg)
            raise Exception(error_msg)

        # 确定新文件路径
        if new_type:
            new_file_path = os.path.join(user_data_path, new_type, new_file_name or file_name)
        else:
            new_file_path = os.path.join(old_root, new_file_name or file_name)

        # 创建目标目录（如果不存在）
        try:
            os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        except Exception as e:
            error_msg = f"创建目录失败 {os.path.dirname(new_file_path)}: {str(e)}"
            logger.error(error_msg)
            send_error_notification_sync("system", None, error_msg)
            raise Exception(error_msg)

        # 重命名文件
        try:
            os.rename(old_file_path, new_file_path)
        except Exception as e:
            error_msg = f"重命名文件失败 {old_file_path} -> {new_file_path}: {str(e)}"
            logger.error(error_msg)
            send_error_notification_sync("system", None, error_msg)
            raise Exception(error_msg)

        # 更新数据库记录
        try:
            # 先查找数据库中的记录
            db_data_files = ModelService.db_manager.get_all_data_files(user_id)
            for db_file in db_data_files:
                if db_file['file_path'] == old_file_path:
                    # 更新记录
                    data_file = DataFile.get_by_id(db_file['id'])
                    data_file.file_path = new_file_path
                    if new_type:
                        data_file.file_type = new_type
                    data_file.save()
                    break
        except Exception as e:
            error_msg = f"更新数据库记录失败: {str(e)}"
            logger.warning(error_msg)
            send_error_notification_sync("system", None, error_msg)

        return {
            "old_file_path": old_file_path,
            "new_file_path": new_file_path
        }

    @staticmethod
    @handle_service_exceptions("model_service")
    def update_model_file(user_id: str, file_name: str, new_file_name: str = None,
                          new_training_epochs: int = None) -> Dict[str, Any]:
        """
        更新指定用户的模型文件（重命名文件和更新数据库记录）
        """
        # 根据文件后缀确定原文件路径
        file_extension = os.path.splitext(file_name)[1]
        if file_extension in ['.pkl', '.onnx', '.trt']:
            subdir = file_extension[1:]  # 去掉点号
        else:
            subdir = "pkl"  # 默认使用pkl目录

        old_file_path = os.path.join(MODEL_USER_PATH, f"{user_id}", subdir, file_name)

        if not os.path.exists(old_file_path):
            error_msg = f"模型文件 {file_name} 不存在"
            logger.error(error_msg)
            send_error_notification_sync("system", None, error_msg)
            raise Exception(error_msg)

        # 确定新文件名
        actual_new_file_name = new_file_name or file_name

        # 根据新文件后缀确定新子目录
        new_file_extension = os.path.splitext(actual_new_file_name)[1]
        if new_file_extension in ['.pkl', '.onnx', '.trt']:
            new_subdir = new_file_extension[1:]  # 去掉点号
        else:
            new_subdir = subdir  # 保持原目录

        new_file_path = os.path.join(MODEL_USER_PATH, f"{user_id}", new_subdir, actual_new_file_name)

        # 创建目标目录（如果不存在）
        try:
            os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        except Exception as e:
            error_msg = f"创建目录失败 {os.path.dirname(new_file_path)}: {str(e)}"
            logger.error(error_msg)
            send_error_notification_sync("system", None, error_msg)
            raise Exception(error_msg)

        # 重命名文件
        try:
            os.rename(old_file_path, new_file_path)
        except Exception as e:
            error_msg = f"重命名文件失败 {old_file_path} -> {new_file_path}: {str(e)}"
            logger.error(error_msg)
            send_error_notification_sync("system", None, error_msg)
            raise Exception(error_msg)

        # 更新数据库记录
        try:
            # 先查找数据库中的记录
            db_model_files = ModelService.db_manager.get_all_model_files(user_id)
            for db_file in db_model_files:
                if db_file['file_path'] == old_file_path:
                    # 更新记录
                    model_file = ModelFile.get_by_id(db_file['id'])
                    model_file.file_path = new_file_path
                    if new_training_epochs is not None:
                        model_file.training_epochs = new_training_epochs
                    model_file.save()
                    break
        except Exception as e:
            error_msg = f"更新数据库记录失败: {str(e)}"
            logger.warning(error_msg)
            send_error_notification_sync("system", None, error_msg)

        return {
            "old_file_path": old_file_path,
            "new_file_path": new_file_path
        }
