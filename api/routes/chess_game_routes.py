# api/routes/chess_game_routes.py

from fastapi import APIRouter, Depends, status, HTTPException  # 添加status导入
from pydantic import BaseModel
from typing import Optional

from api.models.chess_config_model import ChessGameConfig
from api.services.chess_game_service import chess_game_service
from api.models.user_model import User
from api.services.auth_service import get_current_user_from_request
from api.utils.decorators import handle_route_exceptions, format_response
from api.utils.response_utils import format_response_data
from src.cchessAI.core.frontend import get_active_window_ports

router = APIRouter(prefix="/chess-game", tags=["象棋对弈"])

class InitializeRequest(BaseModel):
    """
    初始化游戏请求参数
    """
    user_name: str  # 用户名，用于获取对应的配置

class SpeakRequest(BaseModel):
    """
    语音播报请求参数
    """
    text: str  # 要播报的文本
class MoveRequest(BaseModel):
    """
    移动请求参数
    """
    move: str  # UCI格式的移动，如 "a1a2"
# 添加获取窗口端口的请求模型
class WindowPortsResponse(BaseModel):
    """
    窗口端口响应模型
    """
    active_windows: dict  # 活跃窗口 {window_id: port}
    success: bool = True


@router.post("/initialize", status_code=status.HTTP_200_OK)
@handle_route_exceptions("chess_game")
@format_response()
def initialize_game(current_user: User = Depends(get_current_user_from_request)):
    """
    初始化象棋对弈游戏，根据用户名获取配置

    Args:
        request: 包含用户名的请求对象
        current_user: 当前登录用户
    """
    # 验证用户身份
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户未认证"
        )

    # 获取用户的配置
    config = ChessGameConfig.get_default_config(current_user.user_id)

    # 构造传递给service的参数字典
    init_params = {
        'robot_side': config.robot_side,
        'play_model_file': config.play_model_file,
        'yolo_model_path': config.yolo_model_path,
        'nplayout': config.nplayout,
        'cpuct': config.cpuct,
        'conf': config.conf,
        'iou': config.iou,
        'use_api': True,
        'use_gpu': True,
        'show_board': True,
        'save_recognition_results': None,
        'robot_ip': None,
        'voice_rate': None,
        'voice_volume': None,
        'voice_pitch': None
    }

    result = chess_game_service.initialize_game(**init_params)
    return format_response_data(result)

@router.post("/start", status_code=status.HTTP_200_OK)
@handle_route_exceptions("chess_game")
@format_response()
def start_game(current_user: User = Depends(get_current_user_from_request)):
    """
    开始对弈
    """
    result = chess_game_service.start_game()
    return format_response_data(result)

@router.get("/status", status_code=status.HTTP_200_OK)
@handle_route_exceptions("chess_game")
@format_response()
def get_game_status(current_user: User = Depends(get_current_user_from_request)):
    """
    获取游戏状态
    """
    result = chess_game_service.get_game_status()
    return format_response_data(result)

@router.get("/board-recognition", status_code=status.HTTP_200_OK)
@handle_route_exceptions("chess_game")
@format_response()
def get_board_recognition(current_user: User = Depends(get_current_user_from_request)):
    """
    获取棋盘识别结果
    """
    result = chess_game_service.get_board_recognition_result()
    return format_response_data(result)

@router.post("/surrender", status_code=status.HTTP_200_OK)
@handle_route_exceptions("chess_game")
@format_response()
def surrender(current_user: User = Depends(get_current_user_from_request)):
    """
    投降
    """
    result = chess_game_service.surrender()
    return format_response_data(result)

@router.post("/setup-board", status_code=status.HTTP_200_OK)
@handle_route_exceptions("chess_game")
@format_response()
def setup_initial_board(current_user: User = Depends(get_current_user_from_request)):
    """
    布局初始棋盘
    """
    result = chess_game_service.setup_initial_board()
    return format_response_data(result)

@router.post("/collect-pieces", status_code=status.HTTP_200_OK)
@handle_route_exceptions("chess_game")
@format_response()
def collect_pieces(current_user: User = Depends(get_current_user_from_request)):
    """
    收局
    """
    result = chess_game_service.collect_pieces()
    return format_response_data(result)

@router.post("/undo", status_code=status.HTTP_200_OK)
@handle_route_exceptions("chess_game")
@format_response()
def undo_move(steps: int = 2, current_user: User = Depends(get_current_user_from_request)):
    """
    悔棋

    Args:
        steps: 悔棋步数，默认为1步
    """
    result = chess_game_service.undo_move(steps)
    return format_response_data(result)



@router.post("/stop", status_code=status.HTTP_200_OK)
@handle_route_exceptions("chess_game")
@format_response()
def stop_game(current_user: User = Depends(get_current_user_from_request)):
    """
    停止对弈
    """
    result = chess_game_service.stop_game()
    return format_response_data(result)


@router.post("/speak", status_code=status.HTTP_200_OK)
@handle_route_exceptions("chess_game")
@format_response()
def speak_text(request: SpeakRequest,
               current_user: User = Depends(get_current_user_from_request)):
    """
    语音播报文本

    Args:
        request: 包含要播报文本的请求对象
    """
    result = chess_game_service.speak_text(request.text)
    return format_response_data(result)

# 添加获取窗口端口的API端点
@router.get("/window-ports", response_model=WindowPortsResponse, status_code=status.HTTP_200_OK)
@handle_route_exceptions("chess_game")
@format_response()
def get_window_ports(current_user: User = Depends(get_current_user_from_request)):
    """
    获取当前仍在使用的所有窗口的端口
    """
    try:
        active_windows = get_active_window_ports()
        return format_response_data({
            "active_windows": active_windows
        })
    except Exception as e:
        return format_response_data({
            "success": False,
            "message": f"获取窗口端口信息失败: {str(e)}"
        }, success=False)