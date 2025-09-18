# 基础类定义
import argparse
import asyncio
import copy
import logging
import queue
import threading
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np
from parameters import RED_CAMERA, BLACK_CAMERA, POINT_DOWN, IO_QI, IO_SIDE, \
    WORLD_POINTS_RCV, POINT_RCV_DOWN, RCV_CAMERA, SAC_CAMERA, SRC_SAC_POINTS, SRC_RCV_POINTS, \
    DST_SAC_POINTS, DST_RCV_POINTS, RCV_H_LAY, SAC_H_LAY, POINT_SAC_DOWN, CHESS_POINTS_R, CHESS_POINTS_B, \
    WORLD_POINTS_R, WORLD_POINTS_B, CHESS_POINTS_RCV_H, CHESS_POINTS_RCV_L
from src.cchessAG.chinachess import MainGame
from src.cchessAI import cchess
# 添加项目路径到PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.cchessAI.core.game import Game, uci_to_coordinates, get_best_move_with_computer_play, \
    execute_computer_move


class ChessPlayFlowBase:
    def __init__(self, args):
        self.args = args
        self.urController = None
        self.detector = None
        self.board = cchess.Board()
        self.game = Game(self.board)
        self.board = cchess.Board()
        self.move_history =  []
        self.mcts_player = None
        self.human_player = None
        self.side = 'red'  # 开始棋子方为红方
        self.point_home = self.args.red_camera_position
        self.voice_engine_type = "edge"
        self.pipeline = None
        self.chessboard_image = None
        self.surrendered = False  # 添加投降标志

        # 棋盘状态
        self.is_chesk = False
        self.sac_nums = 0
        self.move_uci = ''                                  # 棋子移动 使用简谱坐标系
        # 棋子映射字典
        self.piece_map = {
            'r': '车', 'n': '马', 'b': '象', 'a': '士', 'k': '将', 'c': '炮', 'p': '卒',  # 黑方
            'R': '車', 'N': '馬', 'B': '相', 'A': '仕', 'K': '帥', 'C': '砲', 'P': '兵'   # 红方
        }
        self.his_chessboard = {} # 历史棋盘
        self.chess_positions = [                            # 使用数组坐标系
            ['r', 'n', 'b', 'a', 'k', 'a', 'b', 'n', 'r'],  # 0行 黑方
            ['.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 1行
            ['.', 'c', '.', '.', '.', '.', '.', 'c', '.'],  # 2行
            ['p', '.', 'p', '.', 'p', '.', 'p', '.', 'p'],  # 3行
            ['.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 4行
            ['.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 5行
            ['P', '.', 'P', '.', 'P', '.', 'P', '.', 'P'],  # 6行 红方
            ['.', 'C', '.', '.', '.', '.', '.', 'C', '.'],  # 7行
            ['.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 8行
            ['R', 'N', 'B', 'A', 'K', 'A', 'B', 'N', 'R']   # 9行
        ]
        # 初始化棋子像素坐标存储
        self.piece_pixel_positions = {}                 # '行列'：像素坐标   数组坐标系
        self.previous_positions = self.chess_positions  # 使用数组坐标系 # 识别棋盘

        # 初始化语音进程管理列表
        self.voice_executor = ThreadPoolExecutor(max_workers=2)
        self.voice_loop = None
        self.voice_thread = None
        self._init_voice_async_loop()

        # 语音队列和状态管理
        self.speech_queue = queue.Queue()  # 语音播报队列
        self.is_speaking = False  # 当前是否正在播报
        self.speech_lock = threading.Lock()  # 语音播报锁
        self.speech_thread = None  # 语音处理线程
        self.voice_cache = {}  # 添加语音缓存
        self.voice_cache_dir = "voice_cache"
        os.makedirs(self.voice_cache_dir, exist_ok=True)

        # 计算透视变换矩阵
        self.m_sac = cv2.getPerspectiveTransform(SRC_SAC_POINTS, DST_SAC_POINTS)
        self.m_rcv = cv2.getPerspectiveTransform(SRC_RCV_POINTS, DST_RCV_POINTS)
        self.rcv_h_lay = RCV_H_LAY # 收子分层高度
        self.sac_v_lay = SAC_H_LAY # 弃子分层高度

        # 初始化MainGame
        self.maingame = MainGame()
        self.maingame.piecesInit()

        # 初始化日志器
        self.logger = logging.getLogger(f"ChessPlayFlow-{os.getpid()}")
        self.logger.setLevel(logging.DEBUG)

        # 避免重复添加处理器
        if not self.logger.handlers:
            # 创建日志目录
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)

            # 创建文件处理器
            file_handler = logging.FileHandler(
                os.path.join(log_dir, f'chess_play_flow_{os.getpid()}.log'),
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)

            # 创建控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # 创建格式器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # 添加处理器到日志器
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def _init_voice_async_loop(self):
        """
        初始化异步语音播报的事件循环
        """

        def run_loop():
            self.voice_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.voice_loop)
            self.voice_loop.run_forever()

        self.voice_thread = threading.Thread(target=run_loop, daemon=True)
        self.voice_thread.start()

        # 等待循环初始化完成
        while self.voice_loop is None:
            time.sleep(0.01)
