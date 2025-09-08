# api/routes/config_routes.py
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Union

from api.services.config_service import (
    get_all_config_tables,
    get_config_table_by_name,
    create_config_table,
    update_config_table,
    delete_config_table,
    get_config_fields,
    add_config_field,
    update_config_field,
    delete_config_field,
    get_config_data,
    set_config_data,
    delete_config_data, update_config_field_comprehensive, get_user_config_data, set_user_config_values,
    get_user_config_values, reset_user_config_to_defaults, reset_user_config_to_defaults_all
)
from api.utils.decorators import handle_route_exceptions, format_response
from api.models.user_model import User
from api.services.auth_service import get_current_user_from_request
from api.utils.response_utils import format_response_data

# 定义数据模型
class ConfigTableCreate(BaseModel):
    table_name: str
    description: Optional[str] = None

class ConfigTableUpdate(BaseModel):
    description: str

class ConfigFieldCreate(BaseModel):
    field_name: str
    field_type: str
    is_required: bool = False
    default_value: Optional[str] = None
    description: Optional[str] = None
class ConfigFieldComprehensiveUpdate(BaseModel):
    new_field_name: Optional[str] = None
    field_type: Optional[str] = None
    is_required: Optional[bool] = None
    default_value: Optional[str] = None
    description: Optional[str] = None
class ConfigFieldUpdate(BaseModel):
    field_type: Optional[str] = None
    is_required: Optional[bool] = None
    default_value: Optional[str] = None
    description: Optional[str] = None

class ConfigDataSet(BaseModel):
    config_key: str
    config_value: Union[str, int, float, bool, Dict[str, Any], List[Any]]
    description: Optional[str] = None

class ConfigValuesSet(BaseModel):
    config_values: Dict[str, Union[str, int, float, bool, Dict[str, Any], List[Any]]]
class ResetConfigFields(BaseModel):
    fields: Optional[List[str]] = None  # 指定要重置的字段列表，如果为None则重置所有字段



router = APIRouter(prefix="/config-tables", tags=["Config Tables"])

# 配置表管理接口
@router.get("/")
@handle_route_exceptions("config_service")
@format_response()
def read_config_tables(current_user: User = Depends(get_current_user_from_request)):
    """获取所有配置表"""
    tables = get_all_config_tables()
    return format_response_data(tables)

@router.post("/", status_code=status.HTTP_201_CREATED)
@handle_route_exceptions("config_service")
@format_response()
def create_config_table_endpoint(table: ConfigTableCreate,
                                current_user: User = Depends(get_current_user_from_request)):
    """创建配置表（仅管理员）"""
    if current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="只有管理员可以创建配置表"
        )
    table_obj = create_config_table(
        table_name=table.table_name,
        description=table.description
    )
    return format_response_data(table_obj)

@router.get("/{table_name}")
@handle_route_exceptions("config_service")
@format_response()
def read_config_table(table_name: str,
                     current_user: User = Depends(get_current_user_from_request)):
    """获取配置表信息"""
    table = get_config_table_by_name(table_name)
    return format_response_data(table)

@router.put("/{table_name}")
@handle_route_exceptions("config_service")
@format_response()
def update_config_table_endpoint(table_name: str, table: ConfigTableUpdate,
                                current_user: User = Depends(get_current_user_from_request)):
    """更新配置表（仅管理员）"""
    if current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="只有管理员可以更新配置表"
        )
    updated_table = update_config_table(
        table_name=table_name,
        description=table.description
    )
    return format_response_data(updated_table)

@router.delete("/{table_name}")
@handle_route_exceptions("config_service")
@format_response()
def delete_config_table_endpoint(table_name: str,
                                current_user: User = Depends(get_current_user_from_request)):
    """删除配置表（仅管理员）"""
    if current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="只有管理员可以删除配置表"
        )
    result = delete_config_table(table_name)
    return [{"deleted": result}]

# 配置字段管理接口
@router.get("/{table_name}/fields")
@handle_route_exceptions("config_service")
@format_response()
def read_config_fields(table_name: str,
                      current_user: User = Depends(get_current_user_from_request)):
    """获取配置表的所有字段定义"""
    fields = get_config_fields(table_name)
    return format_response_data(fields)

@router.post("/{table_name}/fields", status_code=status.HTTP_201_CREATED)
@handle_route_exceptions("config_service")
@format_response()
def add_config_field_endpoint(table_name: str, field: ConfigFieldCreate,
                             current_user: User = Depends(get_current_user_from_request)):
    """为配置表添加字段定义（仅管理员）"""
    if current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="只有管理员可以添加字段定义"
        )
    field_obj = add_config_field(
        table_name=table_name,
        field_name=field.field_name,
        field_type=field.field_type,
        is_required=field.is_required,
        default_value=field.default_value,
        description=field.description
    )
    return format_response_data(field_obj)

@router.patch("/{table_name}/fields/{field_name}")
@handle_route_exceptions("config_service")
@format_response()
def update_config_field_comprehensive_endpoint(table_name: str, field_name: str,
                                               field_update: ConfigFieldComprehensiveUpdate,
                                               current_user: User = Depends(get_current_user_from_request)):
    """综合更新字段定义（仅管理员）"""
    if current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="只有管理员可以更新字段定义"
        )

    # 准备更新数据
    update_kwargs = {}
    if field_update.field_type is not None:
        update_kwargs['field_type'] = field_update.field_type
    if field_update.is_required is not None:
        update_kwargs['is_required'] = field_update.is_required
    if field_update.default_value is not None:
        update_kwargs['default_value'] = field_update.default_value
    if field_update.description is not None:
        update_kwargs['description'] = field_update.description

    # 调用综合更新函数
    field = update_config_field_comprehensive(
        table_name=table_name,
        field_name=field_name,
        new_field_name=field_update.new_field_name,
        **update_kwargs
    )

    return format_response_data(field)

@router.delete("/{table_name}/fields/{field_name}")
@handle_route_exceptions("config_service")
@format_response()
def delete_config_field_endpoint(table_name: str, field_name: str,
                                current_user: User = Depends(get_current_user_from_request)):
    """删除字段定义（仅管理员）"""
    if current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="只有管理员可以删除字段定义"
        )
    result = delete_config_field(table_name, field_name)
    return [{"deleted": result}]


@router.get("/{table_name}/data/{user_id}")
@handle_route_exceptions("config_service")
@format_response()
def read_user_config_data(table_name: str, user_id: str,
                          current_user: User = Depends(get_current_user_from_request)):
    """获取指定用户在配置表中的所有数据"""
    # 管理员可以查看任何用户的数据，普通用户只能查看自己的数据
    if current_user.permission != "admin" and current_user.id != user_id:
        raise HTTPException(
            status_code=403,
            detail="只能查看自己的配置数据"
        )

    data = get_user_config_data(table_name, user_id)
    return format_response_data(data)

@router.post("/{table_name}/data", status_code=status.HTTP_201_CREATED)
@handle_route_exceptions("config_service")
@format_response()
def set_config_data_endpoint(table_name: str, data: ConfigDataSet,
                            current_user: User = Depends(get_current_user_from_request)):
    """设置配置数据"""
    data_obj = set_config_data(
        table_name=table_name,
        config_key=data.config_key,
        config_value=data.config_value,
        description=data.description
    )
    return format_response_data(data_obj)


@router.get("/{table_name}/data/{user_id}/values")
@handle_route_exceptions("config_service")
@format_response()
def read_user_config_values(table_name: str, user_id: str,
                           current_user: User = Depends(get_current_user_from_request)):
    """获取指定用户在配置表中的所有配置值"""
    # 管理员可以查看任何用户的数据，普通用户只能查看自己的数据
    if current_user.permission != "admin" and current_user.id != user_id:
        raise HTTPException(
            status_code=403,
            detail="只能查看自己的配置数据"
        )

    values = get_user_config_values(table_name, user_id)
    return format_response_data(values)

@router.post("/{table_name}/data/{user_id}/values")
@handle_route_exceptions("config_service")
@format_response()
def set_user_config_values_endpoint(table_name: str, user_id: str, data: ConfigValuesSet,
                                   current_user: User = Depends(get_current_user_from_request)):
    """设置指定用户在配置表中的多个配置值"""
    # 管理员可以设置任何用户的数据，普通用户只能设置自己的数据
    if current_user.permission != "admin" and current_user.id != user_id:
        raise HTTPException(
            status_code=403,
            detail="只能设置自己的配置数据"
        )

    result = set_user_config_values(table_name, user_id, data.config_values)
    return format_response_data(result)

@router.post("/{table_name}/data/{user_id}/reset-defaults")
@handle_route_exceptions("config_service")
@format_response()
def reset_user_config_to_defaults_endpoint(table_name: str, user_id: str,
                                          reset_fields: ResetConfigFields = None,
                                          current_user: User = Depends(get_current_user_from_request)):
    """将指定用户在配置表中的配置值重置为默认值"""
    # 管理员可以重置任何用户的数据，普通用户只能重置自己的数据
    if current_user.permission != "admin" and current_user.id != user_id:
        raise HTTPException(
            status_code=403,
            detail="只能重置自己的配置数据"
        )

    # 获取要重置的字段列表
    fields_to_reset = reset_fields.fields if reset_fields else None

    result = reset_user_config_to_defaults(table_name, user_id, fields_to_reset)
    return format_response_data(result)

@router.post("/{table_name}/data/{user_id}/reset-defaults/all")
@handle_route_exceptions("config_service")
@format_response()
def reset_user_config_to_defaults_endpoints(table_name: str, user_id: str,
                                          current_user: User = Depends(get_current_user_from_request)):
    """将指定用户在配置表中的配置值重置为默认值"""
    # 管理员可以重置任何用户的数据，普通用户只能重置自己的数据
    if current_user.permission != "admin" and current_user.id != user_id:
        raise HTTPException(
            status_code=403,
            detail="只能重置自己的配置数据"
        )

    result = reset_user_config_to_defaults_all(table_name, user_id)
    return format_response_data(result)


