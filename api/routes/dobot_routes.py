# api/routes/dobot_routes.py
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Union, Any

from api.models.user_model import User
from api.services.auth_service import get_current_user_from_request
from api.utils.decorators import handle_route_exceptions, format_response
from api.utils.response_utils import format_response_data
from dobot.dobot_control import URController

router = APIRouter(prefix="/dobot", tags=["机械臂控制"])

# 全局机械臂控制器实例
dobot_controller = None

class ConnectRequest(BaseModel):
    ip: Optional[str] = "192.168.5.1"
    port: Optional[int] = 30003
    dashboard_port: Optional[int] = 29999
    feed_port: Optional[int] = 30004
    acceleration: Optional[float] = 0.5
    velocity: Optional[float] = 0.3
class GetParamsRequest(BaseModel):
    params: List[str]  # 要获取的参数列表

class SetParamsRequest(BaseModel):
    params: Dict[str, Union[int, float, str, bool]]  # 要设置的参数字典

class ParamResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
class PointRequest(BaseModel):
    x: float
    y: float
    z: Optional[float] = None

class MoveRequest(BaseModel):
    point: List[float]  # 6个值 [x, y, z, rx, ry, rz]

class JointMoveRequest(BaseModel):
    joints: List[float]  # 6个关节角度值

class IORequest(BaseModel):
    index: int
    value: int

class PositionResponse(BaseModel):
    position: List[float]
    message: str

class StatusResponse(BaseModel):
    connected: bool
    enabled: Optional[bool]
    error_status: Optional[str]
    message: str

class AlarmResponse(BaseModel):
    has_alarm: bool
    error_code: Optional[str]
    message: str


# 添加新的请求模型
class JogStartRequest(BaseModel):
    axis_id: str  # 点动轴，如 "J1+", "X-", 等
    coord_type: Optional[int] = 0  # 坐标系类型 (0: 关节坐标系, 1: 用户坐标系, 2: 工具坐标系)
    user: Optional[int] = 0  # 用户坐标系索引
    tool: Optional[int] = 0  # 工具坐标系索引


# 在路由文件中添加以下API端点
def get_dobot_controller():
    """获取机械臂控制器实例"""
    global dobot_controller
    if dobot_controller is None:
        dobot_controller = URController()
    return dobot_controller

@router.post("/connect", status_code=status.HTTP_200_OK)
@handle_route_exceptions("dobot")
@format_response()
async def connect_dobot(
    request: ConnectRequest = None,
    current_user: User = Depends(get_current_user_from_request)
):
    """
    连接机械臂
    仅管理员可以操作
    """
    global dobot_controller

    # 如果没有提供请求参数，使用默认值
    if request is None:
        request = ConnectRequest()

    try:
        # 如果控制器已经存在且参数不同，则重新创建
        if (dobot_controller is not None and
            (dobot_controller.ip != request.ip or
             dobot_controller.port != request.port or
             dobot_controller.dashboard_port != request.dashboard_port or
             dobot_controller.feed_port != request.feed_port)):
            # 断开现有连接
            if dobot_controller.connected:
                dobot_controller.disconnect()
            # 重新创建控制器
            dobot_controller = URController(
                ip=request.ip,
                port=request.port,
                dashboard_port=request.dashboard_port,
                feed_port=request.feed_port,
                acceleration=request.acceleration,
                velocity=request.velocity
            )
        elif dobot_controller is None:
            # 创建新的控制器实例
            dobot_controller = URController(
                ip=request.ip,
                port=request.port,
                dashboard_port=request.dashboard_port,
                feed_port=request.feed_port,
                acceleration=request.acceleration,
                velocity=request.velocity
            )

        # 如果未连接，则尝试连接
        if not dobot_controller.connected:
            dobot_controller.connect()

        return format_response_data({
            "connected": dobot_controller.connected,
            "ip": dobot_controller.ip,
            "port": dobot_controller.port,
            "dashboard_port": dobot_controller.dashboard_port,
            "feed_port": dobot_controller.feed_port,
            "message": "机械臂连接成功" if dobot_controller.connected else "机械臂连接失败"
        })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"连接机械臂失败: {str(e)}"
        )

@router.post("/disconnect", status_code=status.HTTP_200_OK)
@handle_route_exceptions("dobot")
@format_response()
async def disconnect_dobot(current_user: User = Depends(get_current_user_from_request)):
    """
    断开机械臂连接
    仅管理员可以操作
    """

    try:
        controller = get_dobot_controller()
        if controller.connected:
            controller.disconnect()

        return format_response_data({
            "connected": controller.connected,
            "message": "机械臂已断开连接"
        })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"断开机械臂连接失败: {str(e)}"
        )

@router.post("/enable", status_code=status.HTTP_200_OK)
@handle_route_exceptions("dobot")
@format_response()
async def enable_robot(current_user: User = Depends(get_current_user_from_request)):
    """
    使能机械臂
    仅管理员可以操作
    """
    try:
        controller = get_dobot_controller()
        if not controller.connected:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="机械臂未连接"
            )

        controller.enable_robot()
        return format_response_data({
            "message": "机械臂使能成功"
        })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"使能机械臂失败: {str(e)}"
        )

@router.post("/disable", status_code=status.HTTP_200_OK)
@handle_route_exceptions("dobot")
@format_response()
async def disable_robot(current_user: User = Depends(get_current_user_from_request)):
    """
    失能机械臂
    仅管理员可以操作
    """

    try:
        controller = get_dobot_controller()
        if not controller.connected:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="机械臂未连接"
            )

        controller.disable_robot()
        return format_response_data({
            "message": "机械臂失能成功"
        })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"失能机械臂失败: {str(e)}"
        )

@router.get("/status", response_model=StatusResponse)
@handle_route_exceptions("dobot")
@format_response()
async def get_robot_status(current_user: User = Depends(get_current_user_from_request)):
    """
    获取机械臂状态
    """

    try:
        controller = get_dobot_controller()
        return format_response_data({
            "connected": controller.connected,
            "enabled": None,  # 根据实际需要实现
            "error_status": controller.get_current_error(),
            "message": "状态获取成功"
        })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取机械臂状态失败: {str(e)}"
        )

@router.post("/clear-alarm", response_model=AlarmResponse)
@handle_route_exceptions("dobot")
@format_response()
async def clear_alarm(current_user: User = Depends(get_current_user_from_request)):
    """
    清除机械臂报警
    """

    try:
        controller = get_dobot_controller()
        if not controller.connected:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="机械臂未连接"
            )

        success = controller.clear_alarm()
        return format_response_data({
            "has_alarm": controller.is_alarm_active(),
            "error_code": controller.get_current_error(),
            "message": "报警清除成功" if success else "报警清除失败"
        })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"清除报警失败: {str(e)}"
        )

@router.get("/alarm", response_model=AlarmResponse)
@handle_route_exceptions("dobot")
@format_response()
async def get_alarm_status(current_user: User = Depends(get_current_user_from_request)):
    """
    获取机械臂报警状态
    """

    try:
        controller = get_dobot_controller()
        return format_response_data({
            "has_alarm": controller.is_alarm_active(),
            "error_code": controller.get_current_error(),
            "message": "当前有报警" if controller.is_alarm_active() else "无报警"
        })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取报警状态失败: {str(e)}"
        )

@router.post("/move-to", status_code=status.HTTP_200_OK)
@handle_route_exceptions("dobot")
@format_response()
async def move_to_point(request: MoveRequest, current_user: User = Depends(get_current_user_from_request)):
    """
    移动机械臂到指定位置（直线运动）
    """

    if len(request.point) != 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="位置参数必须包含6个值 [x, y, z, rx, ry, rz]"
        )

    try:
        controller = get_dobot_controller()
        if not controller.connected:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="机械臂未连接"
            )

        controller.run_point_l(request.point)
        return format_response_data({
            "message": f"机械臂已移动到位置 {request.point}"
        })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"移动机械臂失败: {str(e)}"
        )

@router.post("/move-joint", status_code=status.HTTP_200_OK)
@handle_route_exceptions("dobot")
@format_response()
async def move_joint(request: JointMoveRequest, current_user: User = Depends(get_current_user_from_request)):
    """
    关节运动到指定位置
    """

    if len(request.joints) != 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="关节参数必须包含6个值"
        )

    try:
        controller = get_dobot_controller()
        if not controller.connected:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="机械臂未连接"
            )

        controller.run_point_j(request.joints)
        return format_response_data({
            "message": f"机械臂已关节运动到位置 {request.joints}"
        })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"关节运动失败: {str(e)}"
        )

@router.get("/position", response_model=PositionResponse)
@handle_route_exceptions("dobot")
@format_response()
async def get_current_position(current_user: User = Depends(get_current_user_from_request)):
    """
    获取机械臂当前位置
    """

    try:
        controller = get_dobot_controller()
        if not controller.connected:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="机械臂未连接"
            )

        position = controller.get_current_position()
        if position is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="无法获取当前位置"
            )

        return format_response_data({
            "position": list(position),
            "message": "当前位置获取成功"
        })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取当前位置失败: {str(e)}"
        )

@router.post("/home", status_code=status.HTTP_200_OK)
@handle_route_exceptions("dobot")
@format_response()
async def move_to_home(current_user: User = Depends(get_current_user_from_request)):
    """
    移动到初始位置
    """

    try:
        controller = get_dobot_controller()
        if not controller.connected:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="机械臂未连接"
            )

        controller.move_home()
        return format_response_data({
            "message": "机械臂已回到初始位置"
        })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"回到初始位置失败: {str(e)}"
        )

@router.post("/pick-place", status_code=status.HTTP_200_OK)
@handle_route_exceptions("dobot")
@format_response()
async def pick_and_place(request: PointRequest, current_user: User = Depends(get_current_user_from_request)):
    """
    在指定位置执行吸取和放置操作
    """
    try:
        controller = get_dobot_controller()
        if not controller.connected:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="机械臂未连接"
            )

        success = controller.pick_place(request.x, request.y, request.z or 0.05)
        if success:
            return format_response_data({
                "message": "吸取和放置操作成功"
            })
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="吸取和放置操作失败"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"吸取和放置操作失败: {str(e)}"
        )

@router.post("/suction/on", status_code=status.HTTP_200_OK)
@handle_route_exceptions("dobot")
@format_response()
async def suction_on(io_index: int = 12, current_user: User = Depends(get_current_user_from_request)):
    """
    开启吸盘
    """

    try:
        controller = get_dobot_controller()
        if not controller.connected:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="机械臂未连接"
            )

        success = controller.set_do(io_index, 1)
        if success:
            return format_response_data({
                "message": f"DO[{io_index}] 已设置为 1 (开启吸盘)"
            })
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="开启吸盘失败"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"开启吸盘失败: {str(e)}"
        )

@router.post("/suction/off", status_code=status.HTTP_200_OK)
@handle_route_exceptions("dobot")
@format_response()
async def suction_off(io_index: int = 12, current_user: User = Depends(get_current_user_from_request)):
    """
    关闭吸盘
    """

    try:
        controller = get_dobot_controller()
        if not controller.connected:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="机械臂未连接"
            )

        success = controller.set_do(io_index, 0)
        if success:
            return format_response_data({
                "message": f"DO[{io_index}] 已设置为 0 (关闭吸盘)"
            })
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="关闭吸盘失败"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"关闭吸盘失败: {str(e)}"
        )


@router.post("/params/get", response_model=ParamResponse)
@handle_route_exceptions("dobot")
@format_response()
async def get_robot_params(
        request: GetParamsRequest,
        current_user: User = Depends(get_current_user_from_request)
):
    """
    获取机械臂参数
    支持获取多个参数，如速度、加速度等
    """
    try:
        controller = get_dobot_controller()
        if not controller.connected:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="机械臂未连接"
            )

        result = {}
        unsupported_params = []

        # 获取URController实例中的参数
        for param_name in request.params:
            if hasattr(controller, param_name):
                result[param_name] = getattr(controller, param_name)
            else:
                unsupported_params.append(param_name)

        response_data = {
            "retrieved_params": result,
            "unsupported_params": unsupported_params
        }

        return format_response_data(response_data, message="参数获取成功")

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取参数失败: {str(e)}"
        )


@router.post("/params/set", response_model=ParamResponse)
@handle_route_exceptions("dobot")
@format_response()
async def set_robot_params(
        request: SetParamsRequest,
        current_user: User = Depends(get_current_user_from_request)
):
    """
    设置机械臂参数
    支持设置多个参数，如速度、加速度等
    """
    try:
        controller = get_dobot_controller()
        if not controller.connected:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="机械臂未连接"
            )

        success_count = 0
        failed_params = []

        # 设置参数
        for param_name, param_value in request.params.items():
            try:
                if hasattr(controller, param_name):
                    # 特殊处理某些参数
                    if param_name == "velocity":
                        controller.set_velocity(float(param_value))
                        success_count += 1
                    elif param_name == "acceleration":
                        controller.set_acceleration(float(param_value))
                        success_count += 1
                    elif param_name == "speed":
                        controller.set_speed(int(param_value))
                        success_count += 1
                    elif param_name in ["ip", "port", "dashboard_port", "feed_port"]:
                        # 这些参数在连接后不能更改
                        failed_params.append({
                            "param": param_name,
                            "reason": "连接后无法修改"
                        })
                    else:
                        # 直接设置属性
                        setattr(controller, param_name, param_value)
                        success_count += 1
                else:
                    failed_params.append({
                        "param": param_name,
                        "reason": "不支持的参数"
                    })
            except Exception as e:
                failed_params.append({
                    "param": param_name,
                    "reason": str(e)
                })

        response_data = {
            "success_count": success_count,
            "total_count": len(request.params),
            "failed_params": failed_params
        }

        message = f"成功设置 {success_count}/{len(request.params)} 个参数"
        if failed_params:
            message += f"，{len(failed_params)} 个参数设置失败"

        return format_response_data(response_data, message=message)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"设置参数失败: {str(e)}"
        )


@router.get("/params/list")
@handle_route_exceptions("dobot")
@format_response()
async def list_available_params(current_user: User = Depends(get_current_user_from_request)):
    """
    列出可用的机械臂参数
    """
    try:
        controller = get_dobot_controller()

        # 获取URController类的所有属性和方法
        available_params = []
        ignored_attrs = ['dashboard', 'move', 'feed', 'alarm_thread', 'feed_thread']  # 忽略的属性

        for attr_name in dir(controller):
            # 忽略私有属性和方法
            if not attr_name.startswith('_') and attr_name not in ignored_attrs:
                attr_value = getattr(controller, attr_name)
                # 只包含基本数据类型和可序列化的对象
                if isinstance(attr_value, (int, float, str, bool, list, tuple, dict)) or attr_value is None:
                    available_params.append({
                        "name": attr_name,
                        "type": type(attr_value).__name__,
                        "value": attr_value
                    })

        return format_response_data(available_params, message="可用参数列表获取成功")

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取参数列表失败: {str(e)}"
        )


@router.post("/jog/stop", status_code=status.HTTP_200_OK)
@handle_route_exceptions("dobot")
@format_response()
async def stop_jog(current_user: User = Depends(get_current_user_from_request)):
    """
    停止点动运动
    仅管理员可以操作
    """
    try:
        controller = get_dobot_controller()
        if not controller.connected:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="机械臂未连接"
            )

        success = controller.stop_jog()

        if success:
            return format_response_data({
                "message": "点动运动已停止"
            })
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="停止点动运动失败"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"停止点动运动失败: {str(e)}"
        )
