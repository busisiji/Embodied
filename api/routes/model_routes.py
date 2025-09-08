# api/routes/model_routes.py
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional

from api.services.model_service import ModelService
from api.utils.decorators import handle_route_exceptions, format_response
from api.models.user_model import User
from api.services.auth_service import get_current_user_from_request
from api.utils.response_utils import format_response_data

router = APIRouter(prefix="/paly_cchess", tags=["文件管理"])

class UpdateDataFileRequest(BaseModel):
    new_file_name: Optional[str] = None
    new_type: Optional[str] = None

class UpdateModelFileRequest(BaseModel):
    new_file_name: Optional[str] = None
    new_training_epochs: Optional[int] = None

@router.get("/models/list")
@handle_route_exceptions("model_service")
@format_response()
async def list_models(current_user: User = Depends(get_current_user_from_request)):
    """
    列出所有可用的模型
    """
    result = ModelService.list_models()
    return format_response_data(result)

@router.get("/data/list")
@handle_route_exceptions("model_service")
@format_response()
async def list_data(current_user: User = Depends(get_current_user_from_request)):
    """
    列出所有采集的数据文件
    """
    result = ModelService.list_data()
    return format_response_data(result)

@router.get("/data/list/{user_id}")
@handle_route_exceptions("model_service")
@format_response()
async def list_user_data(user_id: str,
                        current_user: User = Depends(get_current_user_from_request)):
    """
    列出指定用户的所有数据文件
    """
    # 检查权限：用户只能查看自己的文件，管理员可以查看所有用户文件
    if current_user.user_id != user_id and current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="没有权限查看该用户的数据文件"
        )
    result = ModelService.list_user_data(user_id)
    return format_response_data(result)

@router.get("/models/list/{user_id}")
@handle_route_exceptions("model_service")
@format_response()
async def list_user_models(user_id: str,
                          current_user: User = Depends(get_current_user_from_request)):
    """
    列出指定用户的所有模型文件
    """
    # 检查权限：用户只能查看自己的文件，管理员可以查看所有用户文件
    if current_user.user_id != user_id and current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="没有权限查看该用户的模型文件"
        )
    result = ModelService.list_user_models(user_id)
    return format_response_data(result)

@router.get("/data/{user_id}/{file_name}")
@handle_route_exceptions("model_service")
@format_response()
async def get_user_data_file(user_id: str, file_name: str,
                            current_user: User = Depends(get_current_user_from_request)):
    """
    获取指定用户的数据文件详细信息
    """
    # 检查权限
    if current_user.user_id != user_id and current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="没有权限查看该用户的数据文件"
        )
    result = ModelService.get_user_data_file(user_id, file_name)
    return format_response_data(result)

@router.get("/models/{user_id}/{file_name}")
@handle_route_exceptions("model_service")
@format_response()
async def get_user_model_file(user_id: str, file_name: str,
                             current_user: User = Depends(get_current_user_from_request)):
    """
    获取指定用户的模型文件详细信息
    """
    # 检查权限
    if current_user.user_id != user_id and current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="没有权限查看该用户的模型文件"
        )
    result = ModelService.get_user_model_file(user_id, file_name)
    return format_response_data(result)

@router.post("/sync/data/{user_id}")
@handle_route_exceptions("model_service")
@format_response()
async def sync_user_data_files_to_db(user_id: str,
                                    current_user: User = Depends(get_current_user_from_request)):
    """
    同步用户数据文件到数据库
    """
    # 检查权限
    if current_user.user_id != user_id and current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="没有权限同步该用户的数据文件"
        )
    result = ModelService.sync_user_data_files_to_db(user_id)
    return [{"sync_result": str(result)}]

@router.post("/sync/models/{user_id}")
@handle_route_exceptions("model_service")
@format_response()
async def sync_user_model_files_to_db(user_id: str,
                                     current_user: User = Depends(get_current_user_from_request)):
    """
    同步用户模型文件到数据库
    """
    # 检查权限
    if current_user.user_id != user_id and current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="没有权限同步该用户的模型文件"
        )
    result = ModelService.sync_user_model_files_to_db(user_id)
    return [{"sync_result": str(result)}]

@router.post("/sync/all")
@handle_route_exceptions("model_service")
@format_response()
async def sync_all_users_files_to_db(current_user: User = Depends(get_current_user_from_request)):
    """
    同步所有用户的数据和模型文件到数据库（仅管理员）
    """
    if current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="只有管理员可以同步所有用户文件"
        )
    result = ModelService.sync_all_users_files_to_db()
    return [{"sync_result": str(result)}]

@router.post("/sync/all/{user_id}")
@handle_route_exceptions("model_service")
@format_response()
async def sync_all_user_files_to_db(user_id: str,
                                   current_user: User = Depends(get_current_user_from_request)):
    """
    同步用户所有数据和模型文件到数据库
    """
    # 检查权限
    if current_user.user_id != user_id and current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="没有权限同步该用户的所有文件"
        )
    result = ModelService.sync_all_user_files_to_db(user_id)
    return [{"sync_result": str(result)}]

@router.get("/db/data/{user_id}")
@handle_route_exceptions("model_service")
@format_response()
async def get_user_data_files_from_db(user_id: str,
                                     current_user: User = Depends(get_current_user_from_request)):
    """
    从数据库获取指定用户的所有数据文件记录
    """
    # 检查权限
    if current_user.user_id != user_id and current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="没有权限查看该用户的数据文件记录"
        )
    result = ModelService.db_manager.get_all_data_files(user_id)
    return format_response_data(result)

@router.get("/db/models/{user_id}")
@handle_route_exceptions("model_service")
@format_response()
async def get_user_model_files_from_db(user_id: str,
                                      current_user: User = Depends(get_current_user_from_request)):
    """
    从数据库获取指定用户的所有模型文件记录
    """
    # 检查权限
    if current_user.user_id != user_id and current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="没有权限查看该用户的模型文件记录"
        )
    result = ModelService.db_manager.get_all_model_files(user_id)
    return format_response_data(result)

@router.delete("/data/{user_id}/{file_name}")
@handle_route_exceptions("model_service")
@format_response()
async def delete_data_file(user_id: str, file_name: str,
                          current_user: User = Depends(get_current_user_from_request)):
    """
    删除指定用户的数据文件（包括本地文件和数据库记录）
    """
    # 检查权限
    if current_user.user_id != user_id and current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="没有权限删除该用户的数据文件"
        )
    result = ModelService.delete_data_file(user_id, file_name)
    return [{"deleted": str(result)}]

@router.delete("/models/{user_id}/{file_name}")
@handle_route_exceptions("model_service")
@format_response()
async def delete_model_file(user_id: str, file_name: str,
                           current_user: User = Depends(get_current_user_from_request)):
    """
    删除指定用户的模型文件（包括本地文件和数据库记录）
    """
    # 检查权限
    if current_user.user_id != user_id and current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="没有权限删除该用户的模型文件"
        )
    result = ModelService.delete_model_file(user_id, file_name)
    return [{"deleted": str(result)}]

@router.put("/data/{user_id}/{file_name}")
@handle_route_exceptions("model_service")
@format_response()
async def update_data_file(user_id: str, file_name: str, request: UpdateDataFileRequest,
                          current_user: User = Depends(get_current_user_from_request)):
    """
    更新指定用户的数据文件（重命名文件和更新数据库记录）
    """
    # 检查权限
    if current_user.user_id != user_id and current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="没有权限更新该用户的数据文件"
        )
    result = ModelService.update_data_file(
        user_id,
        file_name,
        request.new_file_name,
        request.new_type
    )
    return [{"updated": str(result)}]

@router.put("/models/{user_id}/{file_name}")
@handle_route_exceptions("model_service")
@format_response()
async def update_model_file(user_id: str, file_name: str, request: UpdateModelFileRequest,
                           current_user: User = Depends(get_current_user_from_request)):
    """
    更新指定用户的模型文件（重命名文件和更新数据库记录）
    """
    # 检查权限
    if current_user.user_id != user_id and current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="没有权限更新该用户的模型文件"
        )
    result = ModelService.update_model_file(
        user_id,
        file_name,
        request.new_file_name,
        request.new_training_epochs
    )
    return [{"updated": str(result)}]

@router.get("/health")
@handle_route_exceptions("model_service")
@format_response()
async def health_check(current_user: User = Depends(get_current_user_from_request)):
    """
    健康检查端点
    """
    return [{
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "model_service"
    }]
