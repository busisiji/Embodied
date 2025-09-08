# api/routes/process_routes.py
from datetime import datetime
import asyncio

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from starlette.websockets import WebSocket, WebSocketDisconnect

from api.services.process_service import ProcessService
from api.services.websocket_service import websocket_manager
from api.services.process_service import logger
from api.utils.decorators import handle_route_exceptions, format_response
from api.utils.websocket_utils import send_error_notification_sync
from api.models.user_model import User
from api.services.auth_service import get_current_user_from_request
from api.utils.response_utils import format_response_data

router = APIRouter(prefix="/process", tags=["进程管理"])
class CrawlRequest(BaseModel):
    """
    爬虫采集请求数据模型
    """
    owner: str = "n"  # 棋谱类型: t-顶级, m-大师, _m_-比赛, o-其他, u-棋友, n-网络比赛
    start_id: int = 1
    end_id: int = 100
    mode: str = "append"  # append 或 overwrite
    id_mode: bool = False  # True表示具体gameID范围，False表示页码范围

class SelfPlayRequest(BaseModel):
    init_model: Optional[str] = None
    data_path: Optional[str] = None
    workers: int = 4
    use_gpu: bool = True
    nplayout: int = 1200
    temp: float = 1.0
    cpuct: float = 5.0
    game_count: int = 100

class TrainingRequest(BaseModel):
    init_model: Optional[str] = None
    data_path: Optional[str] = None
    epochs: int = 5
    batch_size: int = 512

class AutoTrainingRequest(BaseModel):
    init_model: Optional[str] = None
    interval_minutes: int = 60
    workers: int = 4
    use_gpu: bool = True
    collect_mode: str = "multi_thread"
    temp: float = 1.0
    cpuct: float = 5.0
    self_play_only: bool = False

# 在文件顶部的 import 区域添加新的数据模型
class HumanMoveRequest(BaseModel):
    game_id: str
    move: str  # 用户输入的走法，例如 "e6e9"

class StartGameRequest(BaseModel):
    init_model: Optional[str] = None
    use_gpu: bool = True
    nplayout: int = 1200
    cpuct: float = 5.0
    human_first: bool = False  # 是否人类先手

class PhysicalGameRequest(BaseModel):
    """
    实机人机对弈请求数据模型
    """
    init_model: Optional[str] = None
    use_gpu: bool = True
    nplayout: int = 1200
    cpuct: float = 5.0
    robot_side: str = "red"  # 机器人执子方 ('red' 或 'black')
    yolo_model_path: Optional[str] = None  # YOLO模型路径
    pick_height: float = 100.0  # 棋子吸取高度
    place_height: float = 110.0  # 棋子放置高度

@router.post("/selfplay/start/{user_id}")
@handle_route_exceptions("selfplay")
@format_response()
def start_self_play(user_id: str, request: SelfPlayRequest,
                         current_user: User = Depends(get_current_user_from_request)):
    """
    启动自我对弈数据采集
    """
    # 验证用户权限
    if current_user.user_id != user_id and current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="没有权限执行此操作"
        )

    result = ProcessService.start_self_play(
        user_id=user_id,
        workers=request.workers,
        init_model=request.init_model,
        data_path=request.data_path,
        use_gpu=request.use_gpu,
        nplayout=request.nplayout,
        temp=request.temp,
        cpuct=request.cpuct,
        game_count=request.game_count
    )
    return format_response_data(result)

@router.post("/training/start")
@handle_route_exceptions("training")
@format_response()
def start_training(request: TrainingRequest,
                        current_user: User = Depends(get_current_user_from_request)):
    """
    启动模型训练
    """
    result = ProcessService.start_training(
        init_model=request.init_model,
        data_path=request.data_path,
        epochs=request.epochs,
        batch_size=request.batch_size
    )
    return format_response_data(result)

@router.post("/auto-training/start")
@handle_route_exceptions("auto_training")
@format_response()
def start_auto_training(request: AutoTrainingRequest,
                             current_user: User = Depends(get_current_user_from_request)):
    """
    启动自动训练流程（自我对弈+模型训练）
    """
    result = ProcessService.start_auto_training(
        init_model=request.init_model,
        interval_minutes=request.interval_minutes,
        workers=request.workers,
        use_gpu=request.use_gpu,
        collect_mode=request.collect_mode,
        temp=request.temp,
        cpuct=request.cpuct,
        self_play_only=request.self_play_only
    )
    return format_response_data(result)

@router.post("/evaluation/start")
@handle_route_exceptions("evaluation")
@format_response()
def start_evaluation(model_path: str,
                          current_user: User = Depends(get_current_user_from_request)):
    """
    启动模型评估
    """
    result = ProcessService.start_evaluation(model_path)
    return format_response_data(result)

@router.get("/status/{process_id}")
@handle_route_exceptions("process")
@format_response()
def get_process_status(process_id: str,
                            current_user: User = Depends(get_current_user_from_request)):
    """
    获取进程状态
    """
    result = ProcessService.get_process_status(process_id)
    return format_response_data(result)

@router.get("/logs/{process_id}")
@handle_route_exceptions("process")
@format_response()
def get_process_logs(process_id: str, limit: int = 100,
                          current_user: User = Depends(get_current_user_from_request)):
    """
    获取进程的日志输出
    """
    result = ProcessService.get_process_logs(process_id, limit)
    return format_response_data(result)

@router.post("/stop/{process_id}")
@handle_route_exceptions("process")
@format_response()
def stop_process(process_id: str,
                      current_user: User = Depends(get_current_user_from_request)):
    """
    停止进程
    """
    result = ProcessService.stop_process(process_id)
    return [{"stopped": str(result)}]

@router.post("/stop-all")
@handle_route_exceptions("process")
@format_response()
def stop_all_processes(current_user: User = Depends(get_current_user_from_request)):
    """
    停止所有进程（仅管理员）
    """
    if current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="只有管理员可以停止所有进程"
        )
    result = ProcessService.stop_all_processes()
    return [{"stopped": str(result)}]

@router.get("/list")
@handle_route_exceptions("process")
@format_response()
def list_processes(current_user: User = Depends(get_current_user_from_request)):
    """
    列出所有运行中的进程
    """
    result = ProcessService.list_processes()
    return format_response_data(result)

@router.get("/errors/{process_id}")
@handle_route_exceptions("process")
@format_response()
def get_process_errors(process_id: str,
                            current_user: User = Depends(get_current_user_from_request)):
    """
    获取进程的错误信息
    """
    result = ProcessService.get_process_errors(process_id)
    return format_response_data(result)

@router.get("/health")
@handle_route_exceptions("process")
@format_response()
def health_check(current_user: User = Depends(get_current_user_from_request)):
    """
    健康检查端点
    """
    return [{
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "process_service"
    }]


@router.post("/game/move")
@handle_route_exceptions("game")
@format_response()
def make_human_move(request: HumanMoveRequest,
                          current_user: User = Depends(get_current_user_from_request)):
    """
    用户走一步棋
    """
    result = ProcessService.make_human_move(
        game_id=request.game_id,
        move=request.move
    )
    return format_response_data(result)


@router.post("/crawl/start")
@handle_route_exceptions("crawl")
@format_response()
def start_crawl(request: CrawlRequest,
                      current_user: User = Depends(get_current_user_from_request)):
    """
    启动棋谱数据爬虫采集
    """
    if current_user.permission != "admin":
        raise HTTPException(
            status_code=403,
            detail="只有管理员可以启动数据爬虫"
        )

    result = ProcessService.start_crawl(
        owner=request.owner,
        start_id=request.start_id,
        end_id=request.end_id,
        mode=request.mode,
        id_mode=request.id_mode
    )
    return format_response_data(result)