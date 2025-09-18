# api/services/process_service.py
import subprocess
import os
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import uuid
import threading
import queue

# 导入WebSocket管理器
from api.utils.decorators import handle_service_exceptions
from api.utils.websocket_utils import send_error_notification_sync
from src.cchessAI import cchess
from src.cchessAI.cchess import svg as chessSvg
from src.cchessAI.core.frontend import get_chess_window
from src.cchessAI.core.game import Game
from src.cchessAI.core.mcts import MCTS_AI
from src.cchessAI.core.net import PolicyValueNet
from utils.tools import move_id2move_action, move_action2move_id

# 配置日志
logger = logging.getLogger(__name__)

# 全局变量用于跟踪进程状态和日志
running_processes = {}
process_logs = {}
process_errors = {}  # 存储进程错误信息
game_states = {}


class ProcessService:
    @staticmethod
    @handle_service_exceptions("selfplay")
    def start_self_play(user_id: str = "admin", workers: int = 4, init_model: Optional[str] = None,
                        data_path: Optional[str] = None, use_gpu: bool = True, nplayout: int = 1200,
                        temp: float = 1.0, cpuct: float = 5.0, game_count: int = 100) -> Dict[str, Any]:
        """
        启动自我对弈数据采集
        """
        from utils.file_utils import init_user_file
        init_user_file(user_id)

        from src.cchessAI.parameters import MODEL_USER_PATH, DATA_USER_PATH

        if not init_model:
            init_model_path = None
        else:
            user_path = os.path.join(MODEL_USER_PATH, "user_" + user_id)
            # 根据文件后缀决定上一层目录
            file_extension = os.path.splitext(init_model)[1]
            if file_extension in ['.pkl', '.onnx', '.trt']:
                subdir = file_extension[1:]  # 去掉点号，如 'pkl', 'onnx', 'trt'
                init_model_path = os.path.join(user_path, subdir, init_model)
            else:
                init_model_path = None
        user_path = os.path.join(DATA_USER_PATH, "user_" + user_id)

        if data_path:
            data_path = os.path.join(user_path,'collect', data_path)
        else:
            data_path = os.path.join(user_path,'collect', 'data.pkl')

        # 根据 use_gpu 参数决定使用哪个收集脚本
        if use_gpu:
            # 使用 GPU 收集模式（这里应该调用支持更多参数的脚本）
            cmd = [
                "python", "-m", "src.cchessAI.collect.self_play.collect_rl",
                "--user-id", user_id,  # 添加 user_id 参数
                "--game-count", str(game_count),
                "--init-model", str(init_model_path),
                "--data-path", str(data_path),
                "--nplayout", str(nplayout),
                "--temp", str(temp),
                "--cpuct", str(cpuct)
            ]
        else:
            # 使用多线程 CPU 收集模式
            cmd = [
                "python", "-m", "src.cchessAI.collect.self_play.collect_multi_threads",
                "--user-id", user_id,  # 添加 user_id 参数
                "--workers", str(workers),
                "--init-model", str(init_model),
                "--data-path", str(data_path),
                "--game-count", str(game_count),
                "--nplayout", str(nplayout),
                "--temp", str(temp),
                "--cpuct", str(cpuct),
            ]

        # 在后台运行进程，指定UTF-8编码
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding='utf-8', errors='ignore')
        process_id = f"selfplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # 初始化日志存储
        process_logs[process_id] = queue.Queue()
        process_errors[process_id] = []

        # 启动日志读取线程
        log_thread = threading.Thread(target=ProcessService._read_process_output,
                                    args=(process, process_id), daemon=True)
        log_thread.start()

        running_processes[process_id] = {
            "process": process,
            "type": "selfplay",
            "start_time": datetime.now(),
            "log_thread": log_thread
        }

        mode_desc = "GPU" if use_gpu else "CPU多线程"
        return {
            "status": "started",
            "process_id": process_id,
            "tips": f"自我对弈数据采集已启动，使用 {mode_desc} 模式"
        }

    @staticmethod
    @handle_service_exceptions("training")
    def start_training(user_id: str = "admin", init_model: Optional[str] = None,
                       data_path: Optional[str] = None, epochs: int = 5,
                       batch_size: int = 512) -> Dict[str, Any]:
        """
        启动模型训练
        """
        # 构建命令行参数
        cmd = ["python", "-m", "src.cchessAI.train.train"]  # 添加 user_id 参数

        if init_model:
            cmd.extend(["--init-model", init_model])

        if data_path:
            cmd.extend(["--data-path", data_path])

        cmd.extend(["--epochs", str(epochs)])
        cmd.extend(["--batch-size", str(batch_size)])

        # 在后台运行进程，指定UTF-8编码
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding='utf-8', errors='ignore')
        process_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # 初始化日志存储
        process_logs[process_id] = queue.Queue()
        process_errors[process_id] = []

        # 启动日志读取线程
        log_thread = threading.Thread(target=ProcessService._read_process_output,
                                    args=(process, process_id), daemon=True)
        log_thread.start()

        running_processes[process_id] = {
            "process": process,
            "type": "training",
            "start_time": datetime.now(),
            "log_thread": log_thread
        }

        return {
            "status": "started",
            "process_id": process_id,
            "tips": "模型训练已启动"
        }

    @staticmethod
    @handle_service_exceptions("auto_training")
    def start_auto_training(user_id: str = "admin", init_model: Optional[str] = None,
                            interval_minutes: int = 60, workers: int = 4,
                            use_gpu: bool = True, collect_mode: str = "multi_thread",
                            temp: float = 1.0, cpuct: float = 5.0,
                            self_play_only: bool = False) -> Dict[str, Any]:
        """
        启动自动训练流程（自我对弈+模型训练）
        """
        # 构建命令行参数
        cmd = ["python", "src/cchessAI/autoTrainSelfplay_runner.py"]  # 添加 user_id 参数

        if init_model:
            cmd.extend(["--init-model", init_model])

        cmd.extend(["--interval", str(interval_minutes)])
        cmd.extend(["--workers", str(workers)])

        if use_gpu:
            cmd.append("--use-gpu")

        cmd.extend(["--collect-mode", collect_mode])
        cmd.extend(["--temp", str(temp)])
        cmd.extend(["--cpuct", str(cpuct)])

        if self_play_only:
            cmd.append("--self-play-only")

        # 在后台运行进程
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        process_id = f"auto_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # 初始化日志存储
        process_logs[process_id] = queue.Queue()
        process_errors[process_id] = []

        # 启动日志读取线程
        log_thread = threading.Thread(target=ProcessService._read_process_output,
                                    args=(process, process_id), daemon=True)
        log_thread.start()

        running_processes[process_id] = {
            "process": process,
            "type": "auto_training",
            "start_time": datetime.now(),
            "log_thread": log_thread
        }

        return {
            "status": "started",
            "process_id": process_id,
            "tips": "自动训练流程已启动"
        }

    @staticmethod
    @handle_service_exceptions("evaluation")
    def start_evaluation(user_id: str = "admin", model_path: str = None):
        """
        启动模型评估
        """
        # 构建命令行参数
        cmd = [
            "python", "-m", "src.cchessAI.evaluate.evaluate",
            "--model-path", model_path
        ]

        # 在后台运行进程
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        process_id = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # 初始化日志存储
        process_logs[process_id] = queue.Queue()
        process_errors[process_id] = []

        # 启动日志读取线程
        log_thread = threading.Thread(target=ProcessService._read_process_output,
                                    args=(process, process_id), daemon=True)
        log_thread.start()

        running_processes[process_id] = {
            "process": process,
            "type": "evaluation",
            "start_time": datetime.now(),
            "log_thread": log_thread
        }

        return {
            "status": "started",
            "process_id": process_id,
            "tips": "模型评估已启动"
        }

    @staticmethod
    def _read_process_output(process: subprocess.Popen, process_id: str):
        """
        读取进程的输出并存储到日志队列中
        """
        try:
            process_type = running_processes[process_id]["type"] if process_id in running_processes else "unknown"

            while True:
                # 读取stdout
                try:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        # 将日志添加到队列中
                        if process_id in process_logs:
                            process_logs[process_id].put({
                                "timestamp": datetime.now().isoformat(),
                                "tips": output.strip()
                            })
                except UnicodeDecodeError as e:
                    # 处理编码错误
                    logger.warning(f"读取stdout时发生编码错误: {str(e)}")
                    if process_id in process_logs:
                        process_logs[process_id].put({
                            "timestamp": datetime.now().isoformat(),
                            "tips": f"编码错误: {str(e)}",
                            "level": "warning"
                        })
                    continue

            # 读取stderr中的任何剩余输出
            try:
                stderr_output = process.stderr.read()
                if stderr_output:
                    error_lines = []
                    for line in stderr_output.splitlines():
                        if process_id in process_logs:
                            process_logs[process_id].put({
                                "timestamp": datetime.now().isoformat(),
                                "tips": line.strip(),
                                "level": "error"
                            })
                        error_lines.append(line.strip())

                    # 记录错误信息
                    if process_id in process_errors:
                        process_errors[process_id].extend(error_lines)

                    # 如果有错误，通过WebSocket发送通知
                    if error_lines:
                        error_msg = "进程发生错误:\n" + "\n".join(error_lines)
                        send_error_notification_sync(process_type, error_msg, process_id)
            except UnicodeDecodeError as e:
                logger.error(f"读取stderr时发生编码错误: {str(e)}")
                error_msg = f"读取进程输出时发生编码错误: {str(e)}"
                send_error_notification_sync(process_type, error_msg, process_id)
            # 检查进程退出状态
            return_code = process.returncode
            if return_code != 0:
                error_msg = f"进程异常退出，返回码: {return_code}"
                logger.error(error_msg)
                send_error_notification_sync(process_type, error_msg, process_id)
        except Exception as e:
            error_msg = f"读取进程输出时发生错误: {str(e)}"
            logger.error(error_msg)
            if process_id in running_processes:
                process_type = running_processes[process_id]["type"]
                send_error_notification_sync(process_type, error_msg, process_id)


    @staticmethod
    def get_process_logs(process_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取进程的日志输出
        """
        if process_id not in process_logs:
            raise Exception("进程日志不存在")

        logs = []
        log_queue = process_logs[process_id]

        # 获取所有可用的日志条目，最多limit条
        temp_logs = []
        while not log_queue.empty() and len(temp_logs) < limit:
            try:
                temp_logs.append(log_queue.get_nowait())
            except queue.Empty:
                break

        # 如果获取到日志，按时间排序并返回最新的
        if temp_logs:
            # 按时间戳排序
            temp_logs.sort(key=lambda x: x["timestamp"])
            # 只返回最新的limit条日志
            logs = temp_logs[-limit:]

            # 将日志重新放回队列以供下次获取
            for log in temp_logs:
                log_queue.put(log)

        return logs

    @staticmethod
    def get_process_errors(process_id: str) -> List[str]:
        """
        获取进程的错误信息
        """
        if process_id not in process_errors:
            return []
        return process_errors[process_id]


    @staticmethod
    def stop_all_processes() -> Dict[str, Any]:
        """
        停止所有运行中的进程
        """
        stopped_processes = []
        failed_processes = []

        # 创建当前运行进程的副本，避免在迭代时修改字典
        current_processes = list(running_processes.keys())

        for process_id in current_processes:
            try:
                # 停止进程
                ProcessService.stop_process(process_id)
                stopped_processes.append(process_id)
            except Exception as e:
                failed_processes.append({"process_id": process_id, "error": str(e)})

        return {
            "status": "completed",
            "tips": f"已停止 {len(stopped_processes)} 个进程，{len(failed_processes)} 个进程停止失败",
            "stopped_processes": stopped_processes,
            "failed_processes": failed_processes
        }


    @staticmethod
    @handle_service_exceptions("game")
    def start_game(init_model: Optional[str] = None, use_gpu: bool = True,
                   nplayout: int = 1200, cpuct: float = 5.0, human_first: bool = False) -> Dict[str, Any]:
        """
        开始一局人机对弈游戏
        """
        try:
            # 创建游戏ID
            process_id = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            # 初始化模型
            policy_value_net = PolicyValueNet(model_file=init_model, use_gpu=use_gpu)

            # 创建 MCTS 玩家
            mcts_player = MCTS_AI(policy_value_net.policy_value_fn, c_puct=cpuct, n_playout=nplayout)
            mcts_player.set_player_idx(1 if not human_first else 0)  # AI是红方(1)或黑方(0)

            # 创建棋盘
            board = cchess.Board()

            # 创建游戏实例
            game = Game(board)

            # 保存游戏状态
            game_states[process_id] = {
                "game": game,
                "board": board,
                "ai_player": mcts_player,
                "human_first": human_first,
                "start_time": datetime.now(),
                "moves": []
            }

            # 如果AI先手，让AI走第一步
            ai_move_str = None
            if not human_first:
                # AI走一步
                ai_move = mcts_player.get_action(board)
                ai_move_str = move_id2move_action[ai_move]
                board.push(cchess.Move.from_uci(ai_move_str))
                game_states[process_id]["moves"].append({
                    "player": "AI",
                    "move": ai_move_str,
                    "timestamp": datetime.now()
                })

            # 更新可视化界面
            try:
                svg = chessSvg.board(
                    board,
                    size=600,
                    coordinates=True,
                    axes_type=0,
                    checkers=board.checkers(),
                    lastmove=board.peek() if len(board.move_stack) > 0 else None,
                    orientation=cchess.RED,
                )

                current_player = "红方" if board.turn == cchess.RED else "黑方"
                status_text = f"当前走子: {current_player} - 步数: {len(board.move_stack)}"

                window = get_chess_window()
                if window:
                    window.update_board(svg, status_text)
            except Exception as e:
                logger.warning(f"无法更新可视化界面: {e}")

            # 将游戏添加到运行进程列表中，使其可以通过/list接口查看
            running_processes[process_id] = {
                "type": "game",
                "start_time": datetime.now(),
                "game_state": game_states[process_id]  # 关联游戏状态
            }

            return {
                "status": "success",
                "process_id": process_id,
                "tips": "游戏已开始",
                "ai_first_move": ai_move_str,
                "turn": "human" if human_first else "ai"
            }
        except Exception as e:
            logger.error(f"启动游戏失败: {str(e)}")
            raise Exception(f"启动游戏失败: {str(e)}")


    @staticmethod
    @handle_service_exceptions("game")
    def make_human_move(process_id: str, move: str) -> Dict[str, Any]:
        """
        用户走一步棋
        """
        try:
            # 检查游戏是否存在
            if process_id not in game_states:
                raise Exception("游戏不存在或已结束")

            game_state = game_states[process_id]
            game = game_state["game"]
            board = game_state["board"]
            ai_player = game_state["ai_player"]

            # 验证用户输入的走法
            if move not in move_action2move_id:
                raise Exception(f"无效的走法: {move}")

            move_id = move_action2move_id[move]
            uci_move = cchess.Move.from_uci(move)

            # 检查是否是合法走法
            if uci_move not in board.legal_moves:
                raise Exception(f"非法走法: {move}")

            # 执行用户走法
            board.push(uci_move)
            game_state["moves"].append({
                "player": "human",
                "move": move,
                "timestamp": datetime.now()
            })

            # 更新可视化界面（用户走棋后）
            try:
                svg = cchess.svg.board(
                    board,
                    size=600,
                    coordinates=True,
                    axes_type=0,
                    checkers=board.checkers(),
                    lastmove=board.peek() if len(board.move_stack) > 0 else None,
                    orientation=cchess.RED,
                )

                current_player = "红方" if board.turn == cchess.RED else "黑方"
                status_text = f"当前走子: {current_player} - 步数: {len(board.move_stack)}"

                window = get_chess_window()
                if window:
                    window.update_board(svg, status_text)
            except Exception as e:
                logger.warning(f"无法更新可视化界面: {e}")

            # 检查游戏是否结束
            if board.is_game_over():
                outcome = board.outcome()
                winner = outcome.winner if outcome.winner is not None else -1  # -1 表示平局
                winner_name = "human" if winner == (cchess.RED if game_state["human_first"] else cchess.BLACK) else \
                    "ai" if winner != -1 else "draw"

                # 显示游戏结束状态
                try:
                    svg = cchess.svg.board(
                        board,
                        size=600,
                        coordinates=True,
                        axes_type=0,
                        checkers=board.checkers(),
                        lastmove=board.peek() if len(board.move_stack) > 0 else None,
                        orientation=cchess.RED,
                    )

                    status_text = f"游戏结束，获胜方: {winner_name}"

                    window = get_chess_window()
                    if window:
                        window.update_board(svg, status_text)
                except Exception as e:
                    logger.warning(f"无法更新可视化界面: {e}")

                # 从运行进程和游戏状态中移除
                if process_id in running_processes:
                    del running_processes[process_id]
                del game_states[process_id]

                return {
                    "status": "game_over",
                    "process_id": process_id,
                    "winner": winner_name,
                    "final_board": str(board),
                    "tips": f"游戏结束，获胜方: {winner_name}"
                }

            # AI走一步
            ai_move_id = ai_player.get_action(board)
            ai_move_str = move_id2move_action[ai_move_id]
            ai_uci_move = cchess.Move.from_uci(ai_move_str)
            board.push(ai_uci_move)

            game_state["moves"].append({
                "player": "AI",
                "move": ai_move_str,
                "timestamp": datetime.now()
            })

            # 更新可视化界面（AI走棋后）
            try:
                svg = cchess.svg.board(
                    board,
                    size=600,
                    coordinates=True,
                    axes_type=0,
                    checkers=board.checkers(),
                    lastmove=board.peek() if len(board.move_stack) > 0 else None,
                    orientation=cchess.RED,
                )

                current_player = "红方" if board.turn == cchess.RED else "黑方"
                status_text = f"AI已走棋: {ai_move_str} | 当前走子: {current_player} - 步数: {len(board.move_stack)}"

                window = get_chess_window()
                if window:
                    window.update_board(svg, status_text)
            except Exception as e:
                logger.warning(f"无法更新可视化界面: {e}")

            # 再次检查游戏是否结束
            if board.is_game_over():
                outcome = board.outcome()
                winner = outcome.winner if outcome.winner is not None else -1  # -1 表示平局
                winner_name = "human" if winner == (cchess.RED if game_state["human_first"] else cchess.BLACK) else \
                    "ai" if winner != -1 else "draw"

                # 显示游戏结束状态
                try:
                    svg = cchess.svg.board(
                        board,
                        size=600,
                        coordinates=True,
                        axes_type=0,
                        checkers=board.checkers(),
                        lastmove=board.peek() if len(board.move_stack) > 0 else None,
                        orientation=cchess.RED,
                    )

                    status_text = f"游戏结束，获胜方: {winner_name}"

                    window = get_chess_window()
                    if window:
                        window.update_board(svg, status_text)
                except Exception as e:
                    logger.warning(f"无法更新可视化界面: {e}")

                # 从运行进程和游戏状态中移除
                if process_id in running_processes:
                    del running_processes[process_id]
                del game_states[process_id]

                return {
                    "status": "game_over",
                    "process_id": process_id,
                    "winner": winner_name,
                    "final_board": str(board),
                    "tips": f"游戏结束，获胜方: {winner_name}"
                }

            return {
                "status": "success",
                "process_id": process_id,
                "ai_move": ai_move_str,
                "turn": "human",
                "tips": f"AI已走棋: {ai_move_str}"
            }
        except Exception as e:
            logger.error(f"处理用户走法失败: {str(e)}")
            raise Exception(f"处理用户走法失败: {str(e)}")


    @staticmethod
    def list_processes() -> List[Dict[str, Any]]:
        """
        列出所有运行中的进程
        """
        result = []
        for process_id, process_info in running_processes.items():
            process = process_info.get("process")  # 普通进程有process字段

            # 如果是游戏进程
            if "type" in process_info and process_info["type"] == "game":
                result.append({
                    "process_id": process_id,
                    "type": "game",
                    "status": "running",
                    "start_time": process_info["start_time"]
                })
                continue

            # 检查普通进程是否仍在运行
            if process and process.poll() is None:
                status = "running"
            elif process:
                status = "completed" if process.returncode == 0 else "failed"
                # 清理已完成的进程日志
                if status in ["completed", "failed"]:
                    if process_id in process_logs:
                        del process_logs[process_id]
                    if process_id in process_errors:
                        del process_errors[process_id]
                    continue
            else:
                status = "unknown"

            result.append({
                "process_id": process_id,
                "type": process_info["type"],
                "status": status,
                "start_time": process_info["start_time"]
            })

        return result

    @staticmethod
    def get_process_status(process_id: str) -> Dict[str, Any]:
        """
        获取进程状态
        """
        if process_id not in running_processes:
            raise Exception("进程不存在")

        process_info = running_processes[process_id]

        # 如果是游戏进程
        if "type" in process_info and process_info["type"] == "game":
            return {
                "process_id": process_id,
                "type": "game",
                "status": "running",
                "start_time": process_info["start_time"]
            }

        process = process_info["process"]

        # 检查进程是否仍在运行
        if process.poll() is None:
            status = "running"
        else:
            status = "completed" if process.returncode == 0 else "failed"

        result = {
            "process_id": process_id,
            "type": process_info["type"],
            "status": status,
            "start_time": process_info["start_time"],
            "return_code": process.returncode if process.poll() is not None else None
        }

        # 如果进程失败，添加错误信息
        if status == "failed" and process_id in process_errors:
            result["errors"] = process_errors[process_id]

        return result

    # 修改 stop_process 方法，使其也能处理游戏进程
    @staticmethod
    def stop_process(process_id: str) -> Dict[str, Any]:
        """
        停止进程
        """
        if process_id not in running_processes:
            raise Exception("进程不存在")

        process_info = running_processes[process_id]

        # 如果是游戏进程
        if "type" in process_info and process_info["type"] == "game":
            # 从游戏状态中移除
            if process_id in game_states:
                del game_states[process_id]
            # 从运行进程中移除
            del running_processes[process_id]

            return {
                "status": "stopped",
                "process_id": process_id,
                "tips": "游戏已停止"
            }

        process = process_info["process"]

        # 终止普通进程
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

        # 清理日志和错误信息
        if process_id in process_logs:
            del process_logs[process_id]
        if process_id in process_errors:
            del process_errors[process_id]
        del running_processes[process_id]

        return {
            "status": "stopped",
            "process_id": process_id,
            "tips": "进程已停止"
        }

    @staticmethod
    @handle_service_exceptions("game")
    def get_game_status(process_id: str) -> Dict[str, Any]:
        """
        获取游戏状态
        """
        if process_id not in game_states:
            raise Exception("游戏不存在或已结束")

        game_state = game_states[process_id]
        board = game_state["board"]

        return {
            "status": "running",
            "process_id": process_id,
            "moves": game_state["moves"],
            "turn": "human",  # 默认人类走棋
            "start_time": game_state["start_time"]
        }



    @staticmethod
    def stop_process(process_id: str) -> Dict[str, Any]:
        """
        停止进程
        """
        if process_id not in running_processes:
            raise Exception("进程不存在")

        process_info = running_processes[process_id]

        # 如果是游戏进程
        if "type" in process_info and process_info["type"] == "game":
            # 从游戏状态中移除
            if process_id in game_states:
                del game_states[process_id]
            # 从运行进程中移除
            del running_processes[process_id]

            return {
                "status": "stopped",
                "process_id": process_id,
                "tips": "游戏已停止"
            }

        process = process_info["process"]

        # 终止普通进程
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

        # 清理日志和错误信息
        if process_id in process_logs:
            del process_logs[process_id]
        if process_id in process_errors:
            del process_errors[process_id]
        del running_processes[process_id]

        return {
            "status": "stopped",
            "process_id": process_id,
            "tips": "进程已停止"
        }

    @staticmethod
    @handle_service_exceptions("physical_game")
    def start_physical_game(init_model: Optional[str] = None, use_gpu: bool = True,
                           nplayout: int = 1200, cpuct: float = 5.0, robot_side: str = "red",
                           yolo_model_path: Optional[str] = None, pick_height: float = 100.0,
                           place_height: float = 110.0) -> Dict[str, Any]:
        """
        启动实机人机对弈游戏
        """
        try:
            # 创建游戏ID
            process_id = f"physical_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            # 构建实机对弈命令行参数
            cmd = ["python", "runner/chessPlayFlow_runner.py"]

            # 添加模型参数
            if init_model:
                cmd.extend(["--model_file", init_model])

            if yolo_model_path:
                cmd.extend(["--yolo_model_path", yolo_model_path])

            # 添加对弈参数
            cmd.extend([
                "--use_gpu" if use_gpu else "",
                "--n_playout", str(nplayout),
                "--c_puct", str(cpuct),
                "--robot_side", robot_side,
                "--pick_height", str(pick_height),
                "--place_height", str(place_height)
            ])

            # 移除空字符串参数
            cmd = [arg for arg in cmd if arg != ""]

            # 在后台运行进程
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # 初始化日志存储
            process_logs[process_id] = queue.Queue()
            process_errors[process_id] = []

            # 启动日志读取线程
            log_thread = threading.Thread(target=ProcessService._read_process_output,
                                        args=(process, process_id), daemon=True)
            log_thread.start()

            running_processes[process_id] = {
                "process": process,
                "type": "physical_game",
                "start_time": datetime.now(),
                "log_thread": log_thread,
                "robot_side": robot_side
            }

            return {
                "status": "started",
                "process_id": process_id,
                "tips": f"实机人机对弈已启动，机器人执{robot_side}方"
            }
        except Exception as e:
            logger.error(f"启动实机游戏失败: {str(e)}")
            raise Exception(f"启动实机游戏失败: {str(e)}")

    @staticmethod
    @handle_service_exceptions("physical_game")
    def make_physical_human_move(process_id: str, move: str) -> Dict[str, Any]:
        """
        用户在实机游戏中走一步棋
        """
        try:
            # 检查游戏是否存在
            if process_id not in running_processes:
                raise Exception("游戏不存在或已结束")

            process_info = running_processes[process_id]

            # 验证是否为实机游戏进程
            if process_info.get("type") != "physical_game":
                raise Exception("指定的游戏ID不是实机游戏")

            # 这里可以添加与实机游戏交互的逻辑
            # 例如通过共享文件、数据库或消息队列通知物理系统用户走棋

            return {
                "status": "success",
                "process_id": process_id,
                "tips": f"用户走棋 {move} 已记录，等待系统处理"
            }
        except Exception as e:
            logger.error(f"处理实机用户走法失败: {str(e)}")
            raise Exception(f"处理实机用户走法失败: {str(e)}")

    @staticmethod
    @handle_service_exceptions("crawl")
    def start_crawl(owner: str = "n", start_id: int = 1, end_id: int = 100,
                    mode: str = "append", id_mode: bool = False) -> Dict[str, Any]:
        """
        启动棋谱数据爬虫采集
        """
        try:
            # 构建命令行参数
            cmd = [
                "python", "-m", "src.cchessAI.collect.crawl.crawl_data_csv",
                "--owner", owner,
                "--start-id", str(start_id),
                "--end-id", str(end_id),
                "--mode", mode
            ]

            if id_mode:
                cmd.append("--id-mode")

            # 在后台运行进程
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            process_id = f"crawl_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            # 初始化日志存储
            process_logs[process_id] = queue.Queue()
            process_errors[process_id] = []

            # 启动日志读取线程
            log_thread = threading.Thread(target=ProcessService._read_process_output,
                                        args=(process, process_id), daemon=True)
            log_thread.start()

            running_processes[process_id] = {
                "process": process,
                "type": "crawl",
                "start_time": datetime.now(),
                "log_thread": log_thread,
                "crawl_params": {
                    "owner": owner,
                    "start_id": start_id,
                    "end_id": end_id,
                    "mode": mode,
                    "id_mode": id_mode
                }
            }

            return {
                "status": "started",
                "process_id": process_id,
                "tips": f"棋谱数据爬虫采集已启动 (类型: {owner}, 范围: {start_id}-{end_id})"
            }
        except Exception as e:
            logger.error(f"启动爬虫失败: {str(e)}")
            raise Exception(f"启动爬虫失败: {str(e)}")
