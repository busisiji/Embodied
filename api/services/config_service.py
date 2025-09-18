# api/services/config_service.py
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from api.models.config_model import ConfigTable, ConfigField, ConfigData
from api.utils.exceptions import ResourceNotFoundException, InvalidInputException
from api.utils.websocket_utils import send_error_notification_sync

# 配置日志
logger = logging.getLogger(__name__)
# api/services/config_service.py
from api.utils.decorators import handle_service_exceptions

# 为各个函数添加装饰器
@handle_service_exceptions("config_service")
def get_all_config_tables():
    """获取所有配置表"""
    return list(ConfigTable.select())

@handle_service_exceptions("config_service")
def get_config_table_by_name(table_name: str):
    """根据表名获取配置表"""
    table = ConfigTable.get_or_none(ConfigTable.table_name == table_name)
    if not table:
        raise ResourceNotFoundException(f"Config table '{table_name}'")
    return table

@handle_service_exceptions("config_service")
def create_config_table(table_name: str, description: str = None):
    """创建配置表定义"""
    if not table_name:
        raise InvalidInputException("Table name is required")

    # 检查是否已存在
    existing = ConfigTable.get_or_none(ConfigTable.table_name == table_name)
    if existing:
        raise InvalidInputException(f"Config table '{table_name}' already exists")

    return ConfigTable.create(
        table_name=table_name,
        description=description
    )

# 为其他函数也添加装饰器
@handle_service_exceptions("config_service")
def update_config_table(table_name: str, description: str):
    table = get_config_table_by_name(table_name)
    table.description = description
    table.updated_at = datetime.now()
    table.save()
    return table

@handle_service_exceptions("config_service")
def delete_config_table(table_name: str):
    table = get_config_table_by_name(table_name)

    # 删除相关字段定义
    ConfigField.delete().where(ConfigField.config_table == table_name).execute()

    # 删除相关配置数据
    ConfigData.delete().where(ConfigData.config_table == table_name).execute()

    # 删除表定义
    table.delete_instance()
    return {"detail": f"Config table '{table_name}' deleted successfully"}

@handle_service_exceptions("config_service")
def get_config_fields(table_name: str):
    get_config_table_by_name(table_name)  # 验证表是否存在
    return list(ConfigField.select().where(ConfigField.config_table == table_name))
@handle_service_exceptions("config_service")
def update_config_field(table_name: str, field_name: str, **kwargs):
    field = ConfigField.get_or_none(
        ConfigField.config_table == table_name,
        ConfigField.field_name == field_name
    )
    if not field:
        error_msg = f"字段 '{field_name}' 在表 '{table_name}' 中不存在"
        logger.warning(error_msg)
        send_error_notification_sync("config_service", error_msg)
        raise ResourceNotFoundException(f"Field '{field_name}' in table '{table_name}'")

    for key, value in kwargs.items():
        if hasattr(field, key):
            if key == 'is_required':
                setattr(field, key, 'true' if value else 'false')
            else:
                setattr(field, key, value)

    field.save()
    return field
@handle_service_exceptions("config_service")
def update_config_field_comprehensive(table_name: str, field_name: str,
                                      new_field_name: str = None, **kwargs):
    # 验证配置表是否存在
    get_config_table_by_name(table_name)

    # 验证字段是否存在
    field = ConfigField.get_or_none(
        ConfigField.config_table == table_name,
        ConfigField.field_name == field_name
    )
    if not field:
        error_msg = f"字段 '{field_name}' 在表 '{table_name}' 中不存在"
        logger.warning(error_msg)
        send_error_notification_sync("config_service", error_msg)
        raise ResourceNotFoundException(f"Field '{field_name}' not found in table '{table_name}'")

    # 处理字段名更新
    field_name_changed = False
    if new_field_name and new_field_name != field_name:
        # 验证新字段名是否已存在
        existing = ConfigField.get_or_none(
            ConfigField.config_table == table_name,
            ConfigField.field_name == new_field_name
        )
        if existing:
            error_msg = f"字段 '{new_field_name}' 已存在于表 '{table_name}' 中"
            logger.warning(error_msg)
            send_error_notification_sync("config_service", error_msg)
            raise InvalidInputException(f"Field '{new_field_name}' already exists in table '{table_name}'")

        # 更新字段定义中的字段名
        field.field_name = new_field_name
        field_name_changed = True

    # 处理其他字段属性更新
    for key, value in kwargs.items():
        if hasattr(field, key):
            if key == 'is_required':
                setattr(field, key, 'true' if value else 'false')
            else:
                setattr(field, key, value)

    # 保存字段更新
    field.save()

    # 如果字段名改变了，同步更新配置数据
    if field_name_changed:
        ConfigData.update(config_key=new_field_name).where(
            ConfigData.config_table == table_name,
            ConfigData.config_key == field_name
        ).execute()

    return field

@handle_service_exceptions("config_service")
def set_config_data(table_name: str, config_key: str, config_value, description: str = None):
    get_config_table_by_name(table_name)  # 验证表是否存在

    # 验证字段是否存在
    field = ConfigField.get_or_none(
        ConfigField.config_table == table_name,
        ConfigField.field_name == config_key
    )
    if not field:
        error_msg = f"字段 '{config_key}' 未在表 '{table_name}' 中定义"
        logger.warning(error_msg)
        send_error_notification_sync("config_service", error_msg)
        raise InvalidInputException(f"Field '{config_key}' is not defined in table '{table_name}'")

    # 验证数据类型是否匹配
    if not _validate_data_type(config_value, field.field_type):
        error_msg = f"值类型与字段 '{config_key}' 的类型 '{field.field_type}' 不匹配"
        logger.warning(error_msg)
        send_error_notification_sync("config_service", error_msg)
        raise InvalidInputException(
            f"Value type does not match field type '{field.field_type}' for field '{config_key}'")

    # 序列化值为JSON字符串
    if isinstance(config_value, (dict, list)):
        value_str = json.dumps(config_value)
    else:
        value_str = str(config_value)

    # 检查是否已存在
    existing = ConfigData.get_or_none(
        ConfigData.config_table == table_name,
        ConfigData.config_key == config_key
    )

    if existing:
        existing.config_value = value_str
        existing.description = description
        existing.updated_at = datetime.now()
        existing.save()
        return existing
    else:
        return ConfigData.create(
            config_table=table_name,
            config_key=config_key,
            config_value=value_str,
            description=description
        )
def _validate_data_type(value, field_type: str) -> bool:
    """
    验证数据类型是否与字段定义匹配

    Args:
        value: 要验证的值
        field_type: 字段定义的类型

    Returns:
        bool: 是否匹配
    """
    type_mapping = {
        'string': str,
        'int': int,
        'integer': int,
        'float': (int, float),
        'double': (int, float),
        'boolean': bool,
        'bool': bool
    }

    if field_type in type_mapping:
        expected_type = type_mapping[field_type]
        # 对于数值类型，允许整数赋值给浮点字段
        if field_type in ['float', 'double'] and isinstance(value, int):
            return True
        # 对于布尔类型，也接受字符串形式的true/false
        if field_type in ['boolean', 'bool'] and isinstance(value, str):
            return value.lower() in ['true', 'false', '1', '0']
        return isinstance(value, expected_type)
    elif field_type == 'json':
        # JSON类型接受字典或列表
        return isinstance(value, (dict, list))
    else:
        # 未知类型默认接受
        return True
@handle_service_exceptions("config_service")
def add_config_field(table_name: str, field_name: str, field_type: str,
                    is_required: bool = False, default_value: str = None,
                    description: str = None):
    get_config_table_by_name(table_name)  # 验证表是否存在

    # 检查字段是否已存在
    existing = ConfigField.get_or_none(
        ConfigField.config_table == table_name,
        ConfigField.field_name == field_name
    )
    if existing:
        error_msg = f"字段 '{field_name}' 已存在于表 '{table_name}' 中"
        logger.warning(error_msg)
        send_error_notification_sync("config_service", error_msg)
        raise InvalidInputException(f"Field '{field_name}' already exists in table '{table_name}'")

    field = ConfigField.create(
        config_table=table_name,
        field_name=field_name,
        field_type=field_type,
        is_required='true' if is_required else 'false',
        default_value=default_value,
        description=description
    )

    # 如果字段有默认值，自动创建对应的配置数据
    if default_value is not None:
        try:
            # 尝试解析默认值为正确类型
            parsed_value = _parse_default_value(default_value, field_type)
            set_config_data(table_name, field_name, parsed_value, f"Default value for {field_name}")
        except Exception as e:
            warning_msg = f"为字段 {field_name} 设置默认值失败: {e}"
            logger.warning(warning_msg)

    return field
def _parse_default_value(default_value: str, field_type: str):
    """解析默认值为正确的数据类型"""
    if field_type in ['boolean', 'bool']:
        return default_value.lower() in ['true', '1']
    elif field_type in ['int', 'integer']:
        return int(default_value)
    elif field_type in ['float', 'double']:
        return float(default_value)
    else:
        return default_value
@handle_service_exceptions("config_service")
def delete_config_field(table_name: str, field_name: str):
    field = ConfigField.get_or_none(
        ConfigField.config_table == table_name,
        ConfigField.field_name == field_name
    )
    if not field:
        error_msg = f"字段 '{field_name}' 在表 '{table_name}' 中不存在"
        logger.warning(error_msg)
        send_error_notification_sync("config_service", error_msg)
        raise ResourceNotFoundException(f"Field '{field_name}' in table '{table_name}'")

    # 删除相关的配置数据
    ConfigData.delete().where(
        ConfigData.config_table == table_name,
        ConfigData.config_key == field_name
    ).execute()

    field.delete_instance()
    return {"detail": f"Field '{field_name}' deleted from table '{table_name}'"}

@handle_service_exceptions("config_service")
def get_config_data(table_name: str):
    get_config_table_by_name(table_name)  # 验证表是否存在

    # 只返回已定义字段的数据
    defined_fields = [field.field_name for field in ConfigField.select().where(ConfigField.config_table == table_name)]

    data_list = list(ConfigData.select().where(
        ConfigData.config_table == table_name,
        ConfigData.config_key.in_(defined_fields)
    ))

    # 将配置值解析为适当的数据类型
    result = []
    for data in data_list:
        try:
            parsed_value = json.loads(data.config_value)
        except:
            parsed_value = data.config_value

        result.append({
            'config_key': data.config_key,
            'config_value': parsed_value,
            'description': data.description,
            'created_at': data.created_at,
            'updated_at': data.updated_at
        })

    return result
@handle_service_exceptions("config_service")
def delete_config_data(table_name: str, config_key: str):
    data = ConfigData.get_or_none(
        ConfigData.config_table == table_name,
        ConfigData.config_key == config_key
    )
    if not data:
        error_msg = f"配置数据 '{config_key}' 在表 '{table_name}' 中不存在"
        logger.warning(error_msg)
        send_error_notification_sync("config_service", error_msg)
        raise ResourceNotFoundException(f"Config data '{config_key}' in table '{table_name}'")

    data.delete_instance()
    return {"detail": f"Config data '{config_key}' deleted from table '{table_name}'"}

@handle_service_exceptions("config_service")
def get_user_config_data(table_name: str, user_id: str):
    get_config_table_by_name(table_name)  # 验证表是否存在

    # 只返回已定义字段的数据
    defined_fields = [field.field_name for field in ConfigField.select().where(ConfigField.config_table == table_name)]

    data_list = list(ConfigData.select().where(
        ConfigData.config_table == table_name,
        ConfigData.user_id == user_id,
        ConfigData.config_key.in_(defined_fields)
    ))

    # 将配置值解析为适当的数据类型
    result = []
    for data in data_list:
        try:
            parsed_value = json.loads(data.config_value)
        except:
            parsed_value = data.config_value

        result.append({
            'user_id': data.user_id,
            'config_key': data.config_key,
            'config_value': parsed_value,
            'description': data.description,
            'created_at': data.created_at,
            'updated_at': data.updated_at
        })

    return result

@handle_service_exceptions("config_service")
def get_user_config_values(table_name: str, user_id: str):
    """
    获取指定用户在某个配置表中的所有配置值

    Args:
        table_name: 配置表名
        user_id: 用户ID

    Returns:
        dict: 配置键值对
    """
    get_config_table_by_name(table_name)  # 验证表是否存在

    # 只返回已定义字段的数据
    defined_fields = [field.field_name for field in ConfigField.select().where(ConfigField.config_table == table_name)]

    data_list = list(ConfigData.select().where(
        ConfigData.config_table == table_name,
        ConfigData.user_id == user_id,
        ConfigData.config_key.in_(defined_fields)
    ))

    # 将配置值解析为适当的数据类型并组织成字典
    result = {}
    for data in data_list:
        try:
            parsed_value = json.loads(data.config_value)
        except:
            parsed_value = data.config_value
        result[data.config_key] = parsed_value

    return result

@handle_service_exceptions("config_service")
def set_user_config_values(table_name: str, user_id: str, config_values: Dict[str, Any]):
    """
    设置指定用户在某个配置表中的多个配置值

    Args:
        table_name: 配置表名
        user_id: 用户ID
        config_values: 配置键值对字典

    Returns:
        dict: 更新结果
    """
    get_config_table_by_name(table_name)  # 验证表是否存在

    # 验证字段是否存在并设置值
    results = {}
    for config_key, config_value in config_values.items():
        try:
            # 验证字段是否存在
            field = ConfigField.get_or_none(
                ConfigField.config_table == table_name,
                ConfigField.field_name == config_key
            )
            if not field:
                raise InvalidInputException(f"Field '{config_key}' is not defined in table '{table_name}'")

            # 验证数据类型是否匹配
            if not _validate_data_type(config_value, field.field_type):
                raise InvalidInputException(
                    f"Value type does not match field type '{field.field_type}' for field '{config_key}'")

            # 序列化值为JSON字符串
            if isinstance(config_value, (dict, list)):
                value_str = json.dumps(config_value)
            else:
                value_str = str(config_value)

            # 检查是否已存在
            existing = ConfigData.get_or_none(
                ConfigData.config_table == table_name,
                ConfigData.config_key == config_key,
                ConfigData.user_id == user_id
            )

            if existing:
                existing.config_value = value_str
                existing.updated_at = datetime.now()
                existing.save()
                results[config_key] = "updated"
            else:
                ConfigData.create(
                    user_id=user_id,
                    config_table=table_name,
                    config_key=config_key,
                    config_value=value_str
                )
                results[config_key] = "created"
        except Exception as e:
            results[config_key] = f"error: {str(e)}"

    return results

@handle_service_exceptions("config_service")
def reset_user_config_to_defaults_all(table_name: str, user_id: str):
    """
    将指定用户在某个配置表中的配置值重置为默认值

    Args:
        table_name: 配置表名
        user_id: 用户ID

    Returns:
        dict: 重置结果
    """
    get_config_table_by_name(table_name)  # 验证表是否存在

    # 获取表中所有定义的字段
    fields = list(ConfigField.select().where(ConfigField.config_table == table_name))

    results = {}
    for field in fields:
        try:
            if field.default_value is not None:
                # 解析默认值
                parsed_default_value = _parse_default_value(field.default_value, field.field_type)

                # 序列化值为JSON字符串
                if isinstance(parsed_default_value, (dict, list)):
                    value_str = json.dumps(parsed_default_value)
                else:
                    value_str = str(parsed_default_value)

                # 检查是否已存在该用户的配置数据
                existing = ConfigData.get_or_none(
                    ConfigData.config_table == table_name,
                    ConfigData.config_key == field.field_name,
                    ConfigData.user_id == user_id
                )

                if existing:
                    existing.config_value = value_str
                    existing.updated_at = datetime.now()
                    existing.save()
                    results[field.field_name] = "reset to default"
                else:
                    ConfigData.create(
                        user_id=user_id,
                        config_table=table_name,
                        config_key=field.field_name,
                        config_value=value_str
                    )
                    results[field.field_name] = "set to default"
            else:
                # 如果没有默认值，删除用户现有的配置（如果存在）
                ConfigData.delete().where(
                    ConfigData.config_table == table_name,
                    ConfigData.config_key == field.field_name,
                    ConfigData.user_id == user_id
                ).execute()
                results[field.field_name] = "cleared (no default value)"
        except Exception as e:
            results[field.field_name] = f"error: {str(e)}"

    return results

@handle_service_exceptions("config_service")
def reset_user_config_to_defaults(table_name: str, user_id: str, fields: Optional[List[str]] = None):
    """
    将指定用户在某个配置表中的配置值重置为默认值

    Args:
        table_name: 配置表名
        user_id: 用户ID
        fields: 可选的字段列表，如果提供则只重置指定字段，否则重置所有字段

    Returns:
        dict: 重置结果
    """
    get_config_table_by_name(table_name)  # 验证表是否存在

    # 获取表中所有定义的字段或指定的字段
    if fields:
        field_query = ConfigField.select().where(
            ConfigField.config_table == table_name,
            ConfigField.field_name.in_(fields)
        )
        # 验证指定的字段是否存在
        existing_fields = [f.field_name for f in field_query]
        missing_fields = set(fields) - set(existing_fields)
        if missing_fields:
            raise InvalidInputException(f"Fields not found in table '{table_name}': {missing_fields}")
    else:
        # 如果未指定字段，则选择所有字段
        field_query = ConfigField.select().where(ConfigField.config_table == table_name)

    fields_to_process = list(field_query)

    results = {}
    for field in fields_to_process:
        try:
            if field.default_value is not None:
                # 解析默认值
                parsed_default_value = _parse_default_value(field.default_value, field.field_type)

                # 序列化值为JSON字符串
                if isinstance(parsed_default_value, (dict, list)):
                    value_str = json.dumps(parsed_default_value)
                else:
                    value_str = str(parsed_default_value)

                # 检查是否已存在该用户的配置数据
                existing = ConfigData.get_or_none(
                    ConfigData.config_table == table_name,
                    ConfigData.config_key == field.field_name,
                    ConfigData.user_id == user_id
                )

                if existing:
                    existing.config_value = value_str
                    existing.updated_at = datetime.now()
                    existing.save()
                    results[field.field_name] = "reset to default"
                else:
                    ConfigData.create(
                        user_id=user_id,
                        config_table=table_name,
                        config_key=field.field_name,
                        config_value=value_str
                    )
                    results[field.field_name] = "set to default"
            else:
                # 如果没有默认值，删除用户现有的配置（如果存在）
                ConfigData.delete().where(
                    ConfigData.config_table == table_name,
                    ConfigData.config_key == field.field_name,
                    ConfigData.user_id == user_id
                ).execute()
                results[field.field_name] = "cleared (no default value)"
        except Exception as e:
            results[field.field_name] = f"error: {str(e)}"

    return results
