# main.py
# 这是项目的启动文件，主要职责包括：
# 初始化日志系统；
# 初始化数据库连接并创建表；
# 注册中间件（如请求日志）；3106
# 注册路由；
# 添加全局异常处理器；
# 启动 Uvicorn 服务器。
import argparse
import asyncio
import sys
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # 添加CORS支持
import uvicorn
from starlette.staticfiles import StaticFiles

from api.exceptions.handler import add_exception_handlers
from api.middleware.request_logger import log_requests
from manager.manager import system_manager
from runner.fruitSort_runner import FruitSortingApp

from src.tts_utils.edgeTTS import EdgeTTSWrapper

from api.routes.auth_routes import router as auth_router
from api.routes.user_routes import router as user_router
from api.routes.dobot_routes import router as dobot_router
from api.routes.config_routes import router as config_router
from api.routes.model_routes import router as model_router
from api.routes.process_routes import router as process_router
from api.routes.video_routes import router as video_router
from api.routes.websocket_routes import router as websocket_router
from api.routes.chess_game_routes import router as chess_game_router

from init_database import init_database

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 初始化系统管理器
    system_manager.initialize()

    # 创建应用实例
    fruit_app = FruitSortingApp()

    try:
        # 启动水果分拣
        await fruit_app.start_sorting()
        yield  # 应用程序在此处运行
    except Exception as e:
        print(f"应用程序启动错误: {e}")
    finally:
        # 清理资源
        system_manager.cleanup()
        print("程序退出")


app = FastAPI(lifespan=lifespan,debug=False)
# app = FastAPI(debug=False)
#
# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该指定具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# 挂载静态文件目录
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 注册中间件
app.middleware("http")(log_requests)

# 注册路由
app.include_router(auth_router)
app.include_router(user_router)
app.include_router(dobot_router)
app.include_router(config_router)
app.include_router(model_router)
app.include_router(process_router)
app.include_router(video_router)
app.include_router(chess_game_router)

app.include_router(websocket_router)
@app.get("/")
def read_root():
    return {"Hello": "World"}

# 添加全局异常处理器
add_exception_handlers(app)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=6018, help="服务端口 (默认: 6017)")
    args = parser.parse_args()

    # # 初始化数据库
    init_database()

    print("🚀 正在启动 API 服务...")
    print(f"🌐 监听地址: http://0.0.0.0:{args.port}")
    print(f"⚙️  Debug 模式: {'开启' if app.debug else '关闭'}")
    # 使用支持WebSocket的配置启动
    uvicorn.run("main:app", host="0.0.0.0", port=args.port, workers=1, log_level="info", ws="websockets")

