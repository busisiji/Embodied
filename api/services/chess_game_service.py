# api/services/chess_game_service.py
import os
import threading
import queue
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import asyncio
import argparse
import sys
from argparse import Namespace
import logging
from api.utils.decorators import handle_service_exceptions
from api.utils.websocket_utils import send_error_notification_sync
from src.cchessAI.core.frontend import get_active_window_ports

# from runner.chessPlayFlow_runner import ChessPlayFlow
# 配置日志
logger = logging.getLogger(__name__)

# 全局变量用于跟踪进程状态和日志
running_processes = {}
process_logs = {}
process_errors = {}  # 存储进程错误信息
game_states = {}
class ChessGameService:
    """
    中国象棋人机对弈服务类
    封装chessPlayFlow_runner功能，提供标准API接口
    """

    def __init__(self):
        self.game_instance = None
        self.game_status = "idle"  # idle,initializ, initializing, running
        self.process_id = None
        self.game_events = queue.Queue()  # 游戏事件队列
        self.board_state = None
        self.is_initialized = False
        self.current_player = None
        self.game_thread = None
        self.move_history = []
        self._event_thread = None
        self._stop_event = threading.Event()
        self.window_id = None  # 窗口ID属性
    def _game_event_handler(self):
        """
        处理游戏事件并发送到前端
        """
        while not self._stop_event.is_set() and self.game_status != "idle":
            try:
                if not self.game_events.empty():
                    event = self.game_events.get(timeout=1)
                    # 通过WebSocket发送事件到前端
                    try:
                        send_error_notification_sync("chess_game", self.process_id or "unknown", json.dumps(event))
                    except Exception as e:
                        print(f"WebSocket事件发送失败: {e}")
                time.sleep(0.1)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"事件处理错误: {e}")
                # 记录错误但继续运行
                continue

    def _create_args_namespace(self, kwargs: dict) -> Namespace:
        """
        创建参数命名空间对象
        """
        # 导入chessPlayFlow_runner中的parser
        from runner.chessPlayFlow_runner import create_parser

        # 使用chessPlayFlow_runner中定义的parser
        parser = create_parser()

        # 解析空参数以获取默认值
        args = parser.parse_args([])

        # 将kwargs中的参数更新到args中
        for key, value in kwargs.items():
            if hasattr(args, key):
                setattr(args, key, value)

        return args

    @handle_service_exceptions("chess_game")
    def initialize_game(self, **kwargs) -> Dict[str, Any]:
        """
        初始化对弈

        Args:
            **kwargs: 初始化参数
                robot_side: 机器人执子方 ('red' 或 'black')
                model_file: AI模型文件路径
                yolo_model_path: YOLO模型路径
                use_gpu: 是否使用GPU
                n_playout: MCTS模拟次数
                c_puct: MCTS参数
                ...
        """
        try:
            from runner.chessPlayFlow_runner import ChessPlayFlow
            if self.game_status == 'running':
                return {
                    "success": False,
                    "message": "正在对弈中，请先结束该局对弈"
                }

            if self.game_status == 'initializing':
                return {
                    "success": False,
                    "message": "正在对初始化中"
                }

            self.game_status = "initializing"
            self.process_id = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 为该游戏实例创建专用窗口ID
            self.window_id = f"game_window_{self.process_id}"

            # 启动事件处理线程
            self._stop_event.clear()
            self._event_thread = threading.Thread(target=self._game_event_handler, daemon=True)
            self._event_thread.start()

            # 创建参数命名空间
            args = self._create_args_namespace(kwargs)

            # 验证模型文件路径是否存在
            if hasattr(args, 'play_model_file') and args.play_model_file:
                if not os.path.exists(args.play_model_file):
                    raise FileNotFoundError(f"对弈模型文件不存在: {args.play_model_file}")
                if not os.path.isfile(args.play_model_file):
                    raise ValueError(f"对弈模型路径不是有效文件: {args.play_model_file}")

            if hasattr(args, 'yolo_model_path') and args.yolo_model_path:
                if not os.path.exists(args.yolo_model_path):
                    raise FileNotFoundError(f"YOLO模型文件不存在: {args.yolo_model_path}")
                if not os.path.isfile(args.yolo_model_path):
                    raise ValueError(f"YOLO模型路径不是有效文件: {args.yolo_model_path}")

            # 验证机器人执子方参数
            if args.robot_side not in ["red", "black"]:
                raise ValueError("robot_side 参数必须是 'red' 或 'black'")

            if args.nplayout <= 0:
                raise ValueError("nplayout 参数必须大于 0")

            if args.cpuct <= 0:
                raise ValueError("cpuct 参数必须大于 0")

            if not (0 <= args.conf <= 1):
                raise ValueError("conf 参数必须在 0-1 范围内")

            if not (0 <= args.iou <= 1):
                raise ValueError("iou 参数必须在 0-1 范围内")

            if not (-100 <= args.voice_volume <= 100):
                raise ValueError("voice_volume 参数必须在 -100-100 范围内")

            if not (-100 <= args.voice_rate <= 100):
                raise ValueError("voice_rate 参数必须在 -100 到 100 范围内")

            if not (-100 <= args.voice_pitch <= 100):
                raise ValueError("voice_pitch 参数必须在 -100 到 100 范围内")

            # 创建游戏实例
            self.game_instance = ChessPlayFlow(args)

            # 初始化游戏
            self.game_instance.initialize()
            self.is_initialized = True
            self.game_status = "initialized"

            return {
                "process_id": self.process_id,
                'url':self.game_instance.game.get_window_url(),
                "params": {
                    "音响": self.game_instance.voice_engine is not None,
                    "机械臂": True,
                    "相机": self.game_instance.pipeline is not None,
                    "识别模型": True,
                    "对弈模型": True
                }
            }

        except Exception as e:
            self.game_status = "initialized"
            return {
                "success": False,
                "message": f"游戏初始化失败: {str(e)}",
            }

    @handle_service_exceptions("chess_game")
    def start_game(self) -> Dict[str, Any]:
        """
        开始对弈
        """
        try:
            if not self.is_initialized:
                return {
                    "success": False,
                    "message": "游戏未初始化"
                }

            if self.game_status == 'running':
                return {
                    "success": False,
                    "message": "正在对弈中，请先结束该局对弈"
                }

            self.game_status = "running"

            # 在后台线程中运行游戏主循环
            self.game_thread = threading.Thread(target=self._run_game_loop, daemon=True)
            self.game_thread.start()

            return {
                "process_id": self.process_id,
            }

        except Exception as e:
            self.game_status = "error"
            return {
                "success": False,
                "message": f"游戏启动失败: {str(e)}",
            }

    @handle_service_exceptions("chess_game")
    def _run_game_loop(self):
        """
        运行游戏主循环
        """
        try:
            # 开始对弈
            self.game_instance.play_game()

            # 游戏结束
            self.game_status = "idle"

            # 发送游戏结束事件
            outcome = self.game_instance.board.outcome()
            if outcome:
                winner = "red" if outcome.winner == self.game_instance.board.RED else "black"
                # 判断机器人是否获胜
                robot_side = getattr(self.game_instance.args, 'robot_side', 'red')
                result = "win" if winner == robot_side else "lose"
            else:
                result = "draw"
                winner = "draw"

            self.game_events.put({
                "type": "info",
                "scene":"chess",
                'data':{"process_id": self.process_id,
                        "result": result,
                        "winner": winner,},
                "timestamp": datetime.now().isoformat(),
                "message": f"游戏结束，结果: {result}"
            })

        except Exception as e:
            self.game_status = "error"
            self.game_events.put({
                "type": "error",
                "scene":"chess",
                "data":{
                    "process_id": self.process_id,
                    "error": str(e),
                },
                "timestamp": datetime.now().isoformat(),
                "message": f"游戏运行错误: {str(e)}"
            })

    @handle_service_exceptions("chess_game")
    def get_game_status(self) -> Dict[str, Any]:
        """
        获取游戏状态
        """
        if self.game_status == "idle":
            return {
                "success": False,
                "message": f"游戏未初始化"
            }
        return {
            "game_status": self.game_status, # 游戏状态
            "process_id": self.process_id,
            "is_initialized": self.is_initialized, # 是否已初始化
            "current_player": self.current_player, # 当前玩家
            "board_state": self.game_instance.board.unicode(), # 棋盘状态
            "move_history": self.game_instance.move_history, # 移动历史
        }

    @handle_service_exceptions("chess_game")
    def get_board_recognition_result(self) -> Dict[str, Any]:
        """
        获取棋盘识别结果
        """
        try:
            if not self.is_initialized or not self.game_instance:
                return {
                    "success": False,
                    "message": "游戏未初始化"
                }
            return {
                "board_state": self.game_instance.chessboard_image,
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"获取棋盘识别结果失败: {str(e)}"
            }

    @handle_service_exceptions("chess_game")
    def surrender(self) -> Dict[str, Any]:
        """
        投降
        """
        try:
            if self.game_status != 'running':
                return {
                    "success": False,
                    "message": "你没有在对弈！"
                }

            self.game_instance.set_surrendered()
            self.game_status = "idle"

            return {}

        except Exception as e:
            return {
                "success": False,
                "message": f"投降失败: {str(e)}"
            }

    @handle_service_exceptions("chess_game")
    def setup_initial_board(self) -> Dict[str, Any]:
        """
        布局初始棋盘
        """
        try:
            if not self.is_initialized or not self.game_instance:
                return {
                    "success": False,
                    "message": "游戏未初始化"
                }

            if self.game_status == 'running':
                return {
                    "success": False,
                    "message": "正在对弈中，请先结束该局对弈"
                }

            self.game_instance.setup_initial_board()

            return {
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"布置初始棋盘失败: {str(e)}"
            }

    @handle_service_exceptions("chess_game")
    def collect_pieces(self) -> Dict[str, Any]:
        """
        收局
        """
        try:

            if not self.is_initialized or not self.game_instance:
                return {
                    "success": False,
                    "message": "游戏未初始化"
                }

            if self.game_status == 'running':
                return {
                    "success": False,
                    "message": "正在对弈中，请先结束该局对弈"
                }

            result = self.game_instance.collect_pieces_at_end()
            if not result:
                return {
                    "success": False,
                    "message": "收局失败"
                }

            return {
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"收局失败: {str(e)}"
            }

    @handle_service_exceptions("chess_game")
    def undo_move(self, steps: int = 2) -> Dict[str, Any]:
        """
        悔棋

        Args:
            steps: 悔棋步数，默认为1
        """
        try:
            if self.game_status != 'running':
                return {
                    "success": False,
                    "message": "你没有在对弈！"
                }

            if not self.is_initialized or not self.game_instance:
                return {
                    "success": False,
                    "message": "游戏未初始化"
                }

            # 调用游戏实例的悔棋方法
            success = self.game_instance.undo_move(steps)

            if success:
                return {
                }
            else:
                return {
                    "success": False,
                    "message": "悔棋失败"
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"悔棋失败: {str(e)}"
            }



    @handle_service_exceptions("chess_game")
    def stop_game(self) -> Dict[str, Any]:
        """
        停止对弈
        """
        try:
            previous_status = self.game_status
            self.game_status = "idle"
            self._stop_event.set()  # 停止事件处理线程
            self.is_initialized =  False

            # 清理资源
            if self.game_instance:
                self.game_instance.cleanup()

            return {
                "status": "success",
                "message": "游戏已停止"
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"停止游戏失败: {str(e)}"
            }
    @handle_service_exceptions("chess_game")
    def speak_text(self, text: str) -> Dict[str, Any]:
        """
        语音播报文本

        Args:
            text: 要播报的文本内容
        """
        try:
            if not self.is_initialized or not self.game_instance:
                return {
                    "success": False,
                    "message": "游戏未初始化，无法播报语音"
                }

            # 检查是否有语音引擎
            if not hasattr(self.game_instance, 'voice_engine') or self.game_instance.voice_engine is None:
                return {
                    "success": False,
                    "message": "语音引擎未初始化"
                }

            # 调用语音播报方法
            self.game_instance.speak(text)

            return {
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"语音播报失败: {str(e)}"
            }

    def get_active_window_ports(self):
        """
        获取当前活跃窗口的端口信息
        """
        try:
            active_windows = get_active_window_ports()
            return {
                "active_windows": active_windows
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"获取窗口端口信息失败: {str(e)}"
            }
    def __del__(self):
        """
        析构函数，确保资源被正确释放
        """
        try:
            self._stop_event.set()
            if self._event_thread and self._event_thread.is_alive():
                self._event_thread.join(timeout=1)
        except:
            pass


# 创建全局游戏服务实例
chess_game_service = ChessGameService()
