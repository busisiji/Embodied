# api/routes/video_routes.py
import cv2
import pyrealsense2 as rs
import numpy as np
from fastapi import APIRouter, WebSocket, HTTPException, Query
from fastapi.responses import StreamingResponse
import asyncio
import json
import base64
from typing import Dict

from api.utils.decorators import handle_route_exceptions, format_response
from api.utils.response_utils import format_response_data

router = APIRouter(prefix="/video", tags=["Video Streaming"])

# 存储活跃的WebSocket连接
active_connections: Dict[str, WebSocket] = {}

# 全局相机实例
class CameraManager:
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.is_running = False

    def init_camera(self):
        """初始化RealSense相机"""
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
            profile = self.pipeline.start(self.config)

            # 设置相机参数
            sensors = profile.get_device().query_sensors()
            for sensor in sensors:
                if sensor.get_info(rs.camera_info.name) == "RGB Camera":
                    if sensor.supports(rs.option.enable_auto_exposure):
                        sensor.set_option(rs.option.enable_auto_exposure, True)
                    if sensor.supports(rs.option.sharpness):
                        sensor.set_option(rs.option.sharpness, 100)
            self.is_running = True
            return True
        except Exception as e:
            print(f"相机初始化失败: {e}")
            return False

    def get_frame(self):
        """获取一帧图像"""
        if not self.is_running or not self.pipeline:
            return None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None

            # 转换为numpy数组
            frame = np.asanyarray(color_frame.get_data())
            return frame
        except Exception as e:
            print(f"获取帧失败: {e}")
            return None

    def stop(self):
        """停止相机"""
        self.is_running = False
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None

# 全局相机管理器实例
camera_manager = CameraManager()

@router.websocket("/stream/ws")
async def video_websocket_endpoint(websocket: WebSocket):
    """WebSocket视频流端点 - 无认证版本"""
    await websocket.accept()

    client_id = None

    try:
        # 生成客户端ID
        client_id = f"client_{websocket.client.host}_{websocket.client.port}"
        active_connections[client_id] = websocket

        # 初始化相机
        if not camera_manager.is_running:
            if not camera_manager.init_camera():
                await websocket.send_text(json.dumps({"error": "相机初始化失败"}))
                await websocket.close()
                return

        await websocket.send_text(json.dumps({"status": "connected", "message": "视频流连接成功"}))

        # 发送视频流
        while True:
            try:
                frame = camera_manager.get_frame()
                if frame is not None:
                    # 缩放图像以减少传输数据量
                    if frame.shape[1] > 640:
                        scale_percent = 640 / frame.shape[1]
                        width = int(frame.shape[1] * scale_percent)
                        height = int(frame.shape[0] * scale_percent)
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                    # 编码为JPEG
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

                    # 发送到客户端
                    await websocket.send_text(json.dumps({
                        "type": "frame",
                        "data": jpg_as_text
                    }))

                # 控制帧率
                await asyncio.sleep(0.033)  # ~30 FPS

            except Exception as e:
                print(f"发送帧时出错: {e}")
                break

    except Exception as e:
        print(f"WebSocket连接错误: {e}")
        await websocket.send_text(json.dumps({"error": f"连接错误: {str(e)}"}))
    finally:
        # 清理连接
        if client_id and client_id in active_connections:
            del active_connections[client_id]
        if len(active_connections) == 0:
            camera_manager.stop()

@router.get("/stream/mjpeg")
@handle_route_exceptions("video_service")
@format_response()
async def video_stream():
    """HTTP视频流端点（MJPEG流）- 无认证版本"""
    if not camera_manager.is_running:
        if not camera_manager.init_camera():
            raise HTTPException(status_code=500, detail="相机初始化失败")

    def generate_frames():
        while True:
            frame = camera_manager.get_frame()
            if frame is not None:
                # 缩放图像
                if frame.shape[1] > 640:
                    scale_percent = 640 / frame.shape[1]
                    width = int(frame.shape[1] * scale_percent)
                    height = int(frame.shape[0] * scale_percent)
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                # 编码为JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            # 控制帧率
            import time
            time.sleep(0.033)

    return StreamingResponse(
        generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@router.post("/control/start")
@handle_route_exceptions("video_service")
@format_response()
async def start_camera():
    """启动相机 - 无认证版本"""
    if camera_manager.is_running:
        return format_response_data({"status": "already_running", "message": "相机已在运行"})

    if camera_manager.init_camera():
        return format_response_data({"status": "success", "message": "相机启动成功"})
    else:
        raise HTTPException(status_code=500, detail="相机启动失败")

@router.post("/control/stop")
@handle_route_exceptions("video_service")
@format_response()
async def stop_camera():
    """停止相机 - 无认证版本"""
    camera_manager.stop()
    return format_response_data({"status": "success", "message": "相机已停止"})

@router.get("/status")
@handle_route_exceptions("video_service")
@format_response()
async def camera_status():
    """获取相机状态 - 无认证版本"""
    status_data = {
        "running": camera_manager.is_running,
        "active_connections": len(active_connections),
        "supported_formats": "MJPEG stream and WebSocket"
    }
    return format_response_data(status_data)
