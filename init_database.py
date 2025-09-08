# init_database.py
"""
数据库初始化脚本
用于初始化所有表结构并创建默认配置表
"""

from api.models.user_model import User
from api.models.config_model import ConfigTable, ConfigField, ConfigData
from api.models.chess_file_models import DataFile, ModelFile
from api.db.database import db
from datetime import datetime

from api.services.model_service import ModelService


def init_all_tables():
    """
    初始化所有数据库表
    """
    print("正在初始化数据库表...")

    # 创建所有表结构
    with db:
        db.create_tables([
            User,
            ConfigTable,
            ConfigField,
            ConfigData,
            DataFile,   # 添加数据文件表
            ModelFile,  # 添加模型文件表
        ], safe=True)

    print("数据库表初始化完成!")


def create_default_config_tables():
    """
    创建默认配置表结构
    """
    print("正在创建默认配置表...")

    # 创建对弈数据配置表
    try:
        # 创建对弈配置表
        chess_config_table, created = ConfigTable.get_or_create(
            table_name="chess_game_config",
            defaults={
                "description": "对弈数据配置表，包含对弈相关参数配置",
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
        )

        if created:
            print("  - 创建对弈配置表: chess_game_config")

            # 添加对弈配置字段
            config_fields = [
                {
                    "field_name": "data_file_path",
                    "field_type": "string",
                    "is_required": True,
                    "default_value": "./data/chess_games/",
                    "description": "数据文件存储路径"
                },
                {
                    "field_name": "use_multithreading",
                    "field_type": "boolean",
                    "is_required": True,
                    "default_value": "false",
                    "description": "是否使用多线程"
                },
                {
                    "field_name": "thread_count",
                    "field_type": "integer",
                    "is_required": False,
                    "default_value": "4",
                    "description": "多线程数"
                },
                {
                    "field_name": "game_count",
                    "field_type": "integer",
                    "is_required": True,
                    "default_value": "100",
                    "description": "对弈局数"
                },
                {
                    "field_name": "total_rounds",
                    "field_type": "integer",
                    "is_required": True,
                    "default_value": "10",
                    "description": "总轮数"
                },
                {
                    "field_name": "timeout_seconds",
                    "field_type": "integer",
                    "is_required": False,
                    "default_value": "30",
                    "description": "每步超时时间(秒)"
                },
                {
                    "field_name": "save_interval",
                    "field_type": "integer",
                    "is_required": False,
                    "default_value": "10",
                    "description": "保存间隔(局)"
                },
                # 添加自动化训练相关字段
                {
                    "field_name": "model_update_interval",
                    "field_type": "integer",
                    "is_required": True,
                    "default_value": "60",
                    "description": "模型更新间隔(分钟)"
                },
                {
                    "field_name": "temp",
                    "field_type": "float",
                    "is_required": True,
                    "default_value": "1.0",
                    "description": "温度参数"
                },
                {
                    "field_name": "cpuct",
                    "field_type": "float",
                    "is_required": True,
                    "default_value": "5.0",
                    "description": "CPUCT参数"
                },
                {
                    "field_name": "collect_mode",
                    "field_type": "string",
                    "is_required": True,
                    "default_value": "multi_thread",
                    "description": "采集模式(multi_thread/single_gpu)"
                },
                {
                    "field_name": "use_gpu",
                    "field_type": "boolean",
                    "is_required": True,
                    "default_value": "true",
                    "description": "是否使用GPU"
                },
                # 添加状态跟踪字段
                {
                    "field_name": "last_training_start",
                    "field_type": "string",
                    "is_required": False,
                    "default_value": None,
                    "description": "上次训练开始时间"
                },
                {
                    "field_name": "last_training_end",
                    "field_type": "string",
                    "is_required": False,
                    "default_value": None,
                    "description": "上次训练结束时间"
                },
                {
                    "field_name": "last_selfplay_complete",
                    "field_type": "string",
                    "is_required": False,
                    "default_value": None,
                    "description": "上次自我对弈完成时间"
                },
                {
                    "field_name": "next_training_time",
                    "field_type": "string",
                    "is_required": False,
                    "default_value": None,
                    "description": "下次训练计划时间"
                }
            ]

            for field_data in config_fields:
                ConfigField.create(
                    config_table="chess_game_config",
                    **field_data
                )
                print(f"    - 添加字段: {field_data['field_name']}")
        else:
            print("  - 对弈配置表已存在: chess_game_config")

            # 检查并添加新增的字段（用于更新已存在的表结构）
            existing_fields = [field.field_name for field in ConfigField.select().where(
                ConfigField.config_table == "chess_game_config"
            )]

            new_fields = [
                {
                    "field_name": "model_update_interval",
                    "field_type": "integer",
                    "is_required": True,
                    "default_value": "60",
                    "description": "模型更新间隔(分钟)"
                },
                {
                    "field_name": "temp",
                    "field_type": "float",
                    "is_required": True,
                    "default_value": "1.0",
                    "description": "温度参数"
                },
                {
                    "field_name": "cpuct",
                    "field_type": "float",
                    "is_required": True,
                    "default_value": "5.0",
                    "description": "CPUCT参数"
                },
                {
                    "field_name": "collect_mode",
                    "field_type": "string",
                    "is_required": True,
                    "default_value": "multi_thread",
                    "description": "采集模式(multi_thread/single_gpu)"
                },
                {
                    "field_name": "use_gpu",
                    "field_type": "boolean",
                    "is_required": True,
                    "default_value": "true",
                    "description": "是否使用GPU"
                },
                {
                    "field_name": "last_training_start",
                    "field_type": "string",
                    "is_required": False,
                    "default_value": None,
                    "description": "上次训练开始时间"
                },
                {
                    "field_name": "last_training_end",
                    "field_type": "string",
                    "is_required": False,
                    "default_value": None,
                    "description": "上次训练结束时间"
                },
                {
                    "field_name": "last_selfplay_complete",
                    "field_type": "string",
                    "is_required": False,
                    "default_value": None,
                    "description": "上次自我对弈完成时间"
                },
                {
                    "field_name": "next_training_time",
                    "field_type": "string",
                    "is_required": False,
                    "default_value": None,
                    "description": "下次训练计划时间"
                }
            ]

            for field_data in new_fields:
                if field_data['field_name'] not in existing_fields:
                    ConfigField.create(
                        config_table="chess_game_config",
                        **field_data
                    )
                    print(f"    - 添加新字段: {field_data['field_name']}")

        # 创建训练配置表
        train_config_table, created = ConfigTable.get_or_create(
            table_name="training_config",
            defaults={
                "description": "训练配置表，包含模型训练相关参数",
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
        )

        if created:
            print("  - 创建训练配置表: training_config")

            # 添加训练配置字段
            training_fields = [
                {
                    "field_name": "model_save_path",
                    "field_type": "string",
                    "is_required": True,
                    "default_value": "./models/",
                    "description": "模型保存路径"
                },
                {
                    "field_name": "learning_rate",
                    "field_type": "float",
                    "is_required": True,
                    "default_value": "0.001",
                    "description": "学习率"
                },
                {
                    "field_name": "batch_size",
                    "field_type": "integer",
                    "is_required": True,
                    "default_value": "32",
                    "description": "批次大小"
                },
                {
                    "field_name": "epochs",
                    "field_type": "integer",
                    "is_required": True,
                    "default_value": "50",
                    "description": "训练轮数"
                },
                {
                    "field_name": "validation_split",
                    "field_type": "float",
                    "is_required": False,
                    "default_value": "0.2",
                    "description": "验证集比例"
                }
            ]

            for field_data in training_fields:
                ConfigField.create(
                    config_table="training_config",
                    **field_data
                )
                print(f"    - 添加字段: {field_data['field_name']}")
        else:
            print("  - 训练配置表已存在: training_config")

        # 创建系统配置表
        system_config_table, created = ConfigTable.get_or_create(
            table_name="system_config",
            defaults={
                "description": "系统配置表，包含系统级参数配置",
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
        )

        if created:
            print("  - 创建系统配置表: system_config")

            # 添加系统配置字段
            system_fields = [
                {
                    "field_name": "log_level",
                    "field_type": "string",
                    "is_required": True,
                    "default_value": "INFO",
                    "description": "日志级别"
                },
                {
                    "field_name": "max_concurrent_games",
                    "field_type": "integer",
                    "is_required": False,
                    "default_value": "8",
                    "description": "最大并发对弈数"
                },
                {
                    "field_name": "backup_enabled",
                    "field_type": "boolean",
                    "is_required": True,
                    "default_value": "true",
                    "description": "是否启用数据备份"
                }
            ]

            for field_data in system_fields:
                ConfigField.create(
                    config_table="system_config",
                    **field_data
                )
                print(f"    - 添加字段: {field_data['field_name']}")
        else:
            print("  - 系统配置表已存在: system_config")

    except Exception as e:
        print(f"创建默认配置表时出错: {str(e)}")
        raise

def set_default_config_values(user_id):
    """
    设置默认配置值
    """
    print("正在设置默认配置值...")

    try:
        user_id = user_id
        # 设置对弈配置默认值
        chess_config_data = [
            {
                "config_table": "chess_game_config",
                "config_key": "data_file_path",
                "config_value": "./data/chess_games/",
                "description": "默认数据文件存储路径"
            },
            {
                "config_table": "chess_game_config",
                "config_key": "use_multithreading",
                "config_value": "false",
                "description": "默认不使用多线程"
            },
            {
                "config_table": "chess_game_config",
                "config_key": "thread_count",
                "config_value": "4",
                "description": "默认线程数"
            },
            {
                "config_table": "chess_game_config",
                "config_key": "game_count",
                "config_value": "100",
                "description": "默认对弈局数"
            },
            {
                "config_table": "chess_game_config",
                "config_key": "total_rounds",
                "config_value": "10",
                "description": "默认总轮数"
            },
            # 添加新的默认配置值
            {
                "config_table": "chess_game_config",
                "config_key": "model_update_interval",
                "config_value": "60",
                "description": "默认模型更新间隔(分钟)"
            },
            {
                "config_table": "chess_game_config",
                "config_key": "temp",
                "config_value": "1.0",
                "description": "默认温度参数"
            },
            {
                "config_table": "chess_game_config",
                "config_key": "cpuct",
                "config_value": "5.0",
                "description": "默认CPUCT参数"
            },
            {
                "config_table": "chess_game_config",
                "config_key": "collect_mode",
                "config_value": "multi_thread",
                "description": "默认采集模式"
            },
            {
                "config_table": "chess_game_config",
                "config_key": "use_gpu",
                "config_value": "true",
                "description": "默认使用GPU"
            }
        ]

        for data in chess_config_data:
            ConfigData.get_or_create(
                config_table=data["config_table"],
                config_key=data["config_key"],
                defaults={
                    "user_id": user_id,
                    "config_value": data["config_value"],
                    "description": data["description"],
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }
            )

        print("  - 对弈配置默认值设置完成")

        # 设置训练配置默认值
        training_config_data = [
            {
                "config_table": "training_config",
                "config_key": "model_save_path",
                "config_value": "./models/",
                "description": "默认模型保存路径"
            },
            {
                "config_table": "training_config",
                "config_key": "learning_rate",
                "config_value": "0.001",
                "description": "默认学习率"
            },
            {
                "config_table": "training_config",
                "config_key": "batch_size",
                "config_value": "32",
                "description": "默认批次大小"
            }
        ]

        for data in training_config_data:
            ConfigData.get_or_create(
                config_table=data["config_table"],
                config_key=data["config_key"],
                defaults={
                    "user_id": user_id,
                    "config_value": data["config_value"],
                    "description": data["description"],
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }
            )

        print("  - 训练配置默认值设置完成")

    except Exception as e:
        print(f"设置默认配置值时出错: {str(e)}")
        raise


def create_chess_file_tables():
    """
    创建棋谱数据文件和模型文件表
    """
    print("正在创建棋谱文件表...")

    try:
        # 创建数据文件表
        data_file_table, created = ConfigTable.get_or_create(
            table_name="data_files_meta",
            defaults={
                "description": "数据文件元信息表，记录用户的数据文件信息",
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
        )

        if created:
            print("  - 创建数据文件元信息表: data_files_meta")

            # 添加数据文件表字段定义
            data_file_fields = [
                {
                    "field_name": "user_id",
                    "field_type": "string",
                    "is_required": True,
                    "default_value": None,
                    "description": "用户ID"
                },
                {
                    "field_name": "game_count",
                    "field_type": "integer",
                    "is_required": True,
                    "default_value": "0",
                    "description": "对弈局数"
                },
                {
                    "field_name": "data_length",
                    "field_type": "integer",
                    "is_required": True,
                    "default_value": "0",
                    "description": "数据长度"
                },
                {
                    "field_name": "file_path",
                    "field_type": "string",
                    "is_required": True,
                    "default_value": None,
                    "description": "文件路径"
                }
            ]

            for field_data in data_file_fields:
                ConfigField.create(
                    config_table="data_files_meta",
                    **field_data
                )
                print(f"    - 添加字段: {field_data['field_name']}")
        else:
            print("  - 数据文件元信息表已存在: data_files_meta")

        # 创建模型文件表
        model_file_table, created = ConfigTable.get_or_create(
            table_name="model_files_meta",
            defaults={
                "description": "模型文件元信息表，记录用户的模型文件信息",
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
        )

        if created:
            print("  - 创建模型文件元信息表: model_files_meta")

            # 添加模型文件表字段定义
            model_file_fields = [
                {
                    "field_name": "user_id",
                    "field_type": "string",
                    "is_required": True,
                    "default_value": None,
                    "description": "用户ID"
                },
                {
                    "field_name": "training_epochs",
                    "field_type": "integer",
                    "is_required": True,
                    "default_value": "0",
                    "description": "训练轮次"
                },
                {
                    "field_name": "file_path",
                    "field_type": "string",
                    "is_required": True,
                    "default_value": None,
                    "description": "文件路径"
                }
            ]

            for field_data in model_file_fields:
                ConfigField.create(
                    config_table="model_files_meta",
                    **field_data
                )
                print(f"    - 添加字段: {field_data['field_name']}")
        else:
            print("  - 模型文件元信息表已存在: model_files_meta")

    except Exception as e:
        print(f"创建棋谱文件表时出错: {str(e)}")
        raise


def create_admin_user():
    """
    创建默认管理员账户
    """
    print("正在创建默认管理员账户...")

    try:
        # 检查是否已存在admin用户
        existing_user = User.get_or_none(User.name == "admin")
        if existing_user:
            print("  - 管理员账户已存在")
            return

        # 生成用户ID (user_ + 时间戳)
        from datetime import datetime
        user_id = f"user_{int(datetime.now().timestamp() * 1000)}"

        # 导入密码哈希函数
        from api.services.auth_service import hash_password

        # 哈希密码
        hashed_password, salt = hash_password("admin")

        # 创建管理员用户
        user = User.create(
            user_id=user_id,
            name="admin",
            permission="admin",
            password=hashed_password + ":" + salt
        )

        print("  - 成功创建管理员账户 (用户名: admin, 密码: admin)")
        return user_id

    except Exception as e:
        print(f"创建管理员账户时出错: {str(e)}")
        raise
def init_database():
    """
    初始化完整数据库
    """
    print("=" * 50)
    print("开始初始化数据库...")
    print("=" * 50)

    try:
        # 1. 初始化所有表
        init_all_tables()

        # 2. 创建棋谱文件表
        create_chess_file_tables()

        # 3. 创建管理员账户
        user_id = create_admin_user()

        # 4. 创建默认配置表结构
        create_default_config_tables()

        # 5. 设置默认配置值
        set_default_config_values(user_id)

        # 6. 同步本地文件和数据库
        # ModelService.sync_all_users_files_to_db()

        print("=" * 50)
        print("数据库初始化完成!")
        print("=" * 50)

    except Exception as e:
        print(f"数据库初始化失败: {str(e)}")
        raise


if __name__ == "__main__":
    init_database()
