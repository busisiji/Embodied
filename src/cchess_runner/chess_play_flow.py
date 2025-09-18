# 主流程控制模块
from parameters import IO_SIDE, BLACK_CAMERA, RED_CAMERA
from src.cchessAI.core.game import Game
import argparse
import asyncio
import copy
import logging
import queue
import threading
import time
import os
import sys
from datetime import datetime

import cv2

# 解决libgomp TLS内存分配问题
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from src.cchessAI import cchess
from src.cchess_runner.chess_play_flow_utils import ChessPlayFlowUtils



dir = os.path.dirname(os.path.abspath(__file__))
class ChessPlayFlow(ChessPlayFlowUtils):
    def set_side(self):
        if self.side == 'red':
            self.side = 'black'
        else:
            self.side = 'red'

    def _init_play_game(self):
        # 设置语音识别器的回调函数
        if self.speech_recognizer:
            self.speech_recognizer.callback = self.handle_voice_command


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
        self.previous_positions = self.chess_positions
        self.move_history = []
        self.board = cchess.Board()
        self.game = Game(self.board)
        self.surrendered = False
        self.captured_pieces_history = {}  # 记录被吃的棋子信息
        self.is_undo = False  # 添加悔棋标志
        self.move_count = 0
        self.move_uci = ''

        # 初始化MainGame
        self.maingame.restart_game()

        # 显示初始棋盘
        if self.args.show_board:
            self.game.graphic(self.board)

    def play_game(self):
        """
        执行完整对弈流程（修改版）
        """
        try:
            print("🎮 开始象棋对弈...")
            self.speak("开始对弈，请等待指示灯为绿色再落子")
            self.voice_engine_type = "edge"

            self._init_play_game()

            # 修改循环条件，添加投降检查
            while not self.board.is_game_over() and not self.surrendered:
                if self.surrendered:
                    return

                self.move_count += 1
                print(f"\n--- 第 {self.move_count} 回合 ---")
                if self.move_count == 1:
                    self.board = cchess.Board()
                # 判断当前回合
                is_robot_turn = (self.move_count + (0 if self.args.robot_side == 'red' else 1)) % 2 == 1

                if is_robot_turn:
                    self.urController.hll(f_5=1)  # 红灯
                    print(f"🤖 机器人回合")
                    self.speak("轮到机器人回合，请稍等")

                    # 3. 显示当前棋盘
                    if self.args.show_board:
                        self.game.graphic(self.board)

                    # 4. 计算下一步
                    move_uci = self.calculate_next_move()

                    # 6. 执行移动到棋盘对象
                    move = cchess.Move.from_uci(move_uci)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                    else:
                        self.speak("机器人无法执行该移动")
                        self.gama_over()
                        return

                    # 5. 执行移动
                    self.execute_move(move_uci)
                    self.move_history.append(move_uci)

                    print(f"当前{self.side}方")
                    self.set_side()
                    print(f"当前{self.side}方")


                    # 检查是否将军
                    if self.is_in_check(self.board,self.side):
                        self.speak("请注意，您已被将军！")

                    self.update_chess_positions_after_move(move_uci)
                    chinese_notation = self.uci_to_chinese_notation(move_uci, self.previous_positions)
                    self.speak(f"机器人已走子，{chinese_notation}")

                    # 7. 显示更新后的棋盘
                    if self.args.show_board:
                        self.game.graphic(self.board)

                    print(chinese_notation)

                else:
                    print("👤 人类回合")
                    self.urController.hll(f_4=1)  # 绿灯
                    self.speak("轮到您的回合，请落子")
                    print("⏳ 等待人类落子完成信号...")

                    # 修改等待逻辑，添加投降检查
                    while not self.urController.get_di(IO_SIDE, is_log=False) and not self.surrendered:
                        time.sleep(0.5)
                        if self.surrendered:
                            return
                        if self.is_undo:
                            break
                    if self.is_undo:
                        self.is_undo = False
                        continue
                        # 检查是否投降
                    if self.surrendered:
                        self.gama_over('surrender')
                        return

                    # 复位信号
                    self.urController.hll(f_5=1)  # 红灯
                    self.io_side = self.urController.get_di(IO_SIDE)
                    print("✅ 检测到人类落子完成信号")
                    self.speak("您已落子，请稍等")

                    # 识别当前棋盘状态以更新棋盘
                    print("🔍 识别棋盘以更新状态...")
                    self.his_chessboard[self.move_count-1] = copy.deepcopy(self.previous_positions)
                    # old_positions = self.previous_positions
                    # if self.move_count == 1:
                    #     old_positions = self.chess_positions
                    for i in range(10):
                        if i > 0:
                            positions = self.recognize_chessboard(True)
                        else:
                            positions = self.recognize_chessboard(True)
                        # 推断人类的移动
                        self.move_uci = self.infer_human_move(self.his_chessboard[self.move_count-1], positions)
                        if self.move_uci:
                            break
                    if self.move_uci:
                        print(f"✅ 人类推测走子: {self.move_uci}")
                        move = cchess.Move.from_uci(self.move_uci)
                        if move in self.board.legal_moves:
                            # 检查是否吃掉了机器人的将军
                            is_captured, king_side = self.is_king_captured_by_move(self.move_uci, self.previous_positions)
                            # 如果吃掉的是机器人的将/帅
                            if is_captured and king_side == self.args.robot_side:
                                self.gama_over('player')  # 人类玩家获胜
                                self.speak('吃掉了机器人的将军！')
                                return  # 结束游戏

                            self.board.push(move)

                        else:
                            # 检查是否被将军且无法解除将军状态
                            if self.is_in_check(self.board,self.args.robot_side):
                                # 移动无效，执行空移动
                                self.board.push(cchess.Move.null())

                                # 检查是否存在能吃掉将军的移动
                                move_uci = self.find_check_move()

                                # 检查这个移动是否真的是吃掉将军的移动
                                move = cchess.Move.from_uci(move_uci)
                                if move in self.board.legal_moves:
                                    # 检查目标位置是否是对方的将/帅
                                    target_piece = self.board.piece_at(move.to_square)
                                    if target_piece and target_piece.piece_type == cchess.KING:
                                        # 确实是吃掉将军的移动，执行它
                                        self.execute_move(move_uci)
                                        # self.speak("将军！吃掉你的将帅！")
                                        self.speak(f"很遗憾，您输了！")
                                        time.sleep(20)
                                        return  # 结束游戏

                            else:
                                self.speak("您违规了，请重新走子")
                                self.move_count = self.move_count - 1
                                self.urController.hll(f_4=1)  # 绿灯
                                continue
                    else:
                        print("错误！无法推断人类的移动")
                        self.speak("无法检测到走棋，请重新落子")
                        self.urController.hll(f_4=1)  # 绿灯
                        self.move_count = self.move_count - 1
                        continue

                    # 显示更新后的棋盘
                    if self.args.show_board:
                        self.game.graphic(self.board)

                    # 落子完成
                    self.update_chess_positions_after_move(self.move_uci)
                    print(f"✅ 人类走法已应用: {self.move_uci}")
                    chinese_notation = self.uci_to_chinese_notation(self.move_uci, self.previous_positions)
                    self.speak(f"您已走子，{chinese_notation}")
                    print(chinese_notation)

                    self.move_history.append(self.move_uci)
                    self.his_chessboard[self.move_count] = copy.deepcopy(self.previous_positions)

                    self.set_side()
                # 短暂等待以便观察
                #             time.sleep(1)
                # self.clear_cache()


            # 游戏结束
            if self.board.is_game_over() or self.surrendered:
                # 如果是投降结束的游戏
                if self.surrendered:
                    self.gama_over('surrender')
                else:
                    # 正常游戏结束
                    outcome = self.board.outcome()
                    if outcome is not None:
                        winner = "red" if outcome.winner == cchess.RED else "black"
                        print(f"获胜方是{winner}")
                        if winner == self.args.robot_side:
                            self.speak("您已被将死！")
                            self.gama_over('dobot')
                        else:
                            self.gama_over()
                    else:
                        self.gama_over('平局')
        except Exception as e:
            self.report_error(str(e))

    def gama_over(self,winner='player'):
        self.urController.hll()
        if winner == 'player':
            print(f'恭喜您获得胜利！')
            self.speak(f"恭喜您获得胜利！")
        elif winner == 'dobot':
            print(f'很遗憾，您输了！')
            self.speak(f"很遗憾，您输了！")
        elif winner == 'surrender':
            print(f'您已投降！')
            self.speak(f"您已投降！")
        else:
            print("🤝 游戏结束，平局")
            self.speak(f"游戏结束，平局")
        time.sleep(3)

    async def save_recognition_result_with_detections(self, chess_result, red_image, red_detections, black_image, black_detections):
        """
        异步保存带检测框的识别结果图像

        Args:
            chess_result: 棋盘识别结果
            red_image: 红方半区原始图像
            red_detections: 红方半区检测结果 (Results对象)
            black_image: 黑方半区原始图像
            black_detections: 黑方半区检测结果 (Results对象)
        """
        import cv2
        from copy import deepcopy
        import asyncio

        # 创建结果目录
        result_dir = self.args.result_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        async def save_red_detections():
            """异步保存红方检测结果"""
            if red_image is not None and red_detections is not None:
                red_image_with_detections = deepcopy(red_image)

                # 从Results对象中提取边界框信息
                boxes = red_detections[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())

                        # 绘制边界框
                        cv2.rectangle(red_image_with_detections, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # 添加标签
                        label = f"Red:{cls} {conf:.2f}"
                        cv2.putText(red_image_with_detections, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 保存带检测框的红方图像
                red_detected_path = os.path.join(result_dir,f"red_side_detected{self.move_count}.jpg")
                cv2.imwrite(red_detected_path, red_image_with_detections)
                print(f"💾 红方检测结果已保存至: {red_detected_path}")

        async def save_black_detections():
            """异步保存黑方检测结果"""
            if black_image is not None and black_detections is not None:
                black_image_with_detections = deepcopy(black_image)

                # 从Results对象中提取边界框信息
                boxes = black_detections[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())

                        # 绘制边界框
                        cv2.rectangle(black_image_with_detections, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        # 添加标签
                        label = f"Black:{cls} {conf:.2f}"
                        cv2.putText(black_image_with_detections, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # 保存带检测框的黑方图像
                black_detected_path = os.path.join(result_dir, f"black_side_detected{self.move_count}.jpg")
                cv2.imwrite(black_detected_path, black_image_with_detections)
                print(f"💾 黑方检测结果已保存至: {black_detected_path}")

        async def save_chessboard_layout():
            """异步保存棋盘布局图"""
            # 可视化完整的棋盘布局
            self.chessboard_image = self.visualize_chessboard(chess_result)
            chessboard_path = os.path.join(result_dir, f"chessboard_layout.jpg")
            cv2.imwrite(chessboard_path, self.chessboard_image)
            # 报告棋盘识别结果给web端
            if self.args.use_api:
                self.report_board_recognition_result(self.chessboard_image)

            print(f"💾 棋盘布局图已保存至: {chessboard_path}")

        # 并发执行保存操作
        await asyncio.gather(
            save_red_detections(),
            save_black_detections(),
            save_chessboard_layout()
        )

    def handle_voice_command(self, keywords, full_text):
        """
        处理语音命令

        Args:
            keywords: 识别到的关键字列表
            full_text: 完整的识别文本
        """
        print(f"收到的关键字: {keywords}")

        # 游戏控制命令
        if "开始" in keywords or "重新开始" in keywords:
            self.speak("重新开始游戏")

        elif "结束" in keywords or "退出" in keywords:
            self.speak("结束游戏")
            self.set_surrendered()  # 投降结束游戏

        elif "悔棋" in keywords:
            self.speak("执行悔棋")
            # 设置悔棋标志
            self.is_undo = True
            # 可以在这里添加具体悔棋逻辑

        elif "帮助" in keywords:
            self.speak("您可以使用语音控制游戏，说开始、结束、悔棋等命令")

        elif "认输" in keywords or "投降" in keywords:
            self.speak("您已认输，游戏结束")
            self.set_surrendered()

        # 添加收子关键字相关回调事件
        elif "收子" in keywords:
            self.speak("执行收子操作")
            try:
                # 调用收子方法
                self.collect_pieces_at_end()
            except Exception as e:
                self.speak("收子操作失败")
                print(f"收子操作失败: {e}")

        # 添加布局关键字相关回调事件
        elif "布局" in keywords or "摆子" in keywords:
            self.speak("执行初始布局操作")
            try:
                # 调用布局方法
                self.setup_initial_board()
            except Exception as e:
                self.speak("布局操作失败")
                print(f"布局操作失败: {e}")

        # 添加悔棋关键字相关回调事件undo_move
        elif "撤销" in keywords or "撤回" in keywords:
            self.speak("执行悔棋操作")
            try:
                self.undo_move()
            except Exception as e:
                self.speak("悔棋操作失败")
                print(f"悔棋操作失败: {e}")
    def speak(self, text):
        """
        使用统一的TTS管理器进行异步语音播报

        Args:
            text: 要播报的文本
        """
        # 检查是否启用语音
        if not self.args.enable_voice:
            return

        try:
            print(f"📢 语音播报: {text}")
            # 使用异步方式调用TTS管理器播报文本
            if hasattr(self, 'tts_manager') and self.tts_manager:
                # 提交到线程池异步执行
                async def async_speak():
                    await self.tts_manager.speak_async(text)
                asyncio.run(async_speak())
                time.sleep(1)
            else:
                print("⚠️ TTS管理器未初始化")
        except Exception as e:
            print(f"⚠️ 语音播报失败: {e}")

    def clear_cache(self):
        """
        清理缓存，释放内存
        """
        try:
            # 清理Python垃圾回收
            import gc
            gc.collect()

            # 如果使用了torch，清理GPU缓存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass

            print("✅ 缓存清理完成")

        except Exception as e:
            print(f"⚠️ 缓存清理时出错: {e}")

    def set_surrendered(self):
        self.surrendered = True
        time.sleep(3)
        self.urController.hll()

    def cleanup(self):
        """
        清理资源
        """
        try:
            self.surrendered = True

            # 断开机械臂
            if self.urController:
                self.urController.hll()
                print("🔌 断开机械臂连接...")
                self.urController.disconnect()

            # 清理相机
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                print("📷 关闭相机...")
                self.pipeline.stop()
                self.pipeline = None


            # 关闭OpenCV窗口
            if self.args.show_camera:
                cv2.destroyAllWindows()


            print("✅ 清理完成")
            self.speak("结束运行")
        except Exception as e:
            print(f"⚠️ 清理时出错: {e}")

    def report_piece_deviation(self, row, col, deviation_x, deviation_y, distance):
        """
        报告棋子偏移信息

        Args:
            row: 行号
            col: 列号
            deviation_x: X方向偏移(mm)
            deviation_y: Y方向偏移(mm)
            distance: 总偏移距离(mm)
        """
        # 发送偏移报警到游戏服务
        try:
            from api.services.chess_game_service import chess_game_service
            if hasattr(chess_game_service, 'game_events') and chess_game_service.game_events:
                chess_game_service.game_events.put({
                    "type": "error",
                    "scene": "chess/deviation",
                    "data" : {
                        "position": {"row": row, "col": col},
                        "deviation": {
                            "x": deviation_x,
                            "y": deviation_y,
                            "distance": distance
                        },
                    },
                    "timestamp": datetime.now().isoformat(),
                    "message": f"第{row + 1}行,第{col + 1}列棋子偏离标准位置{distance:.2f}mm"
                })
        except Exception as e:
            print(f"发送偏移报警失败: {e}")

    def report_move(self, player, move_uci, chinese_notation):
        """
        报告棋子移动信息

        Args:
            player: 玩家 ("human" 或 "robot")
            move_uci: UCI格式移动
            chinese_notation: 中文记谱法
        """
        # 发送移动信息到游戏服务
        try:
            from api.services.chess_game_service import chess_game_service
            if hasattr(chess_game_service, 'game_events') and chess_game_service.game_events:
                chess_game_service.game_events.put({
                    "type": "info",
                    "scene": "chess/move",
                    'data':{
                        "player": player,
                        "uci": move_uci,
                        "chinese": chinese_notation
                    },
                    "timestamp": datetime.now().isoformat(),
                    "message": f"{player}走棋: {chinese_notation} ({move_uci})"
                })
        except Exception as e:
            print(f"发送移动信息失败: {e}")

    def report_board_recognition_result(self, chessboard_image):
        """
        报告棋盘识别结果图像信息

        Args:
            chessboard_image: 识别后的棋盘图像(numpy数组)
        """
        # 发送棋盘识别结果到游戏服务
        try:
            from api.services.chess_game_service import chess_game_service
            if hasattr(chess_game_service, 'game_events') and chess_game_service.game_events:
                # 将图像转换为base64编码以便通过JSON传输
                import base64
                import cv2
                import numpy as np

                # 将图像编码为JPEG格式
                if chessboard_image is not None:
                    _, buffer = cv2.imencode('.jpg', chessboard_image)
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

                    chess_game_service.game_events.put({
                        "type": "info",
                        "scene": "chess/recognition",
                        "data": {
                            "image_data": jpg_as_text,
                        },
                        "timestamp": datetime.now().isoformat(),
                        "message": "棋盘识别结果已更新"
                    })
        except Exception as e:
            print(f"发送棋盘识别结果失败: {e}")

    def report_error(self, error_msg):
        """
        报告错误信息并记录日志

        Args:
            error_msg: 错误信息
        """
        # 记录错误日志
        self.logger.error(f"人机对弈错误: {error_msg}")

        # 发送错误信息到游戏服务
        try:
            pass
            # from api.services.chess_game_service import chess_game_service
            # if hasattr(chess_game_service, 'game_events') and chess_game_service.game_events:
            #     error_data = {
            #         "type": "error",
            #         "scene": "chess/error",
            #         "data": {},
            #         "timestamp": datetime.now().isoformat(),
            #         "message": error_msg
            #     }
            #     chess_game_service.game_events.put(error_data)
        except Exception as e:
            pass

def create_parser():
    """创建参数解析器"""
    parser = argparse.ArgumentParser(description='象棋自动对弈系统')

    # 显示和保存参数
    parser.add_argument('--use_api', default=False, help='是否使用api')
    parser.add_argument('--show_camera', default=False, action='store_true', help='是否显示相机实时画面')
    parser.add_argument('--show_board',  default=False, action='store_true', help='是否在窗口中显示棋局')
    parser.add_argument('--save_recognition_results', default=False, action='store_true', help='是否保存识别结果')
    parser.add_argument('--result_dir', type=str, default='chess_play_results',
                        help='结果保存目录')

    # 语音
    parser.add_argument('--enable_voice', default=True, action='store_true', help='是否启用语音提示')
    parser.add_argument('--voice_rate', type=int, default=0, help='语音语速，语速稍慢(-10)，音调较高(20)，音量适中(90)')
    parser.add_argument('--voice_volume', type=int, default=0, help='语音音量')
    parser.add_argument('--voice_pitch', type=int, default=0, help='语音音调')

    # 机械臂相关参数
    parser.add_argument('--robot_ip', type=str, default='192.168.5.1', help='机械臂IP地址')
    parser.add_argument('--robot_port', type=int, default=30003, help='机械臂移动控制端口')
    parser.add_argument('--robot_dashboard_port', type=int, default=29999, help='机械臂控制面板端口')
    parser.add_argument('--robot_feed_port', type=int, default=30005, help='机械臂反馈端口')

    # 模型路径参数
    parser.add_argument('--yolo_model_path', type=str,
                        default=os.path.join(dir,'../cchessYolo/runs/detect/chess_piece_detection_separate5/weights/best.pt'),
                        help='YOLO棋子检测模型路径')
    parser.add_argument('--play_model_file', type=str,
                        default=os.path.join(dir,'../cchessAI/models/admin/trt/current_policy_batch7483_202507170806.trt'),
                        help='对弈模型文件路径')
    # 相机位置参数
    parser.add_argument('--red_camera_position', type=float, nargs=6,
                        default=RED_CAMERA,
                        help='红方拍摄吸子位置 [x, y, z, rx, ry, rz]')
    parser.add_argument('--black_camera_position', type=float, nargs=6,
                        default=BLACK_CAMERA,
                        help='黑方拍摄位置 [x, y, z, rx, ry, rz]')
    parser.add_argument('--black_position', type=float, nargs=6,
                        default=[BLACK_CAMERA[0],BLACK_CAMERA[1],BLACK_CAMERA[2],RED_CAMERA[3],RED_CAMERA[4],RED_CAMERA[5]],
                        help='黑方吸子位置 [x, y, z, rx, ry, rz]')
    # 其他参数
    parser.add_argument('--robot_side', type=str, default='black', help='机器人执子方')
    parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--nplayout', type=int, default=400, help='MCTS模拟次数')
    parser.add_argument('--cpuct', type=float, default=5.0, help='MCTS参数')
    parser.add_argument('--conf', type=float, default=0.45, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.25, help='IOU阈值')

    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()

    # 创建对弈流程对象
    chess_flow = ChessPlayFlow(args)

    try:
        # 初始化
        chess_flow.initialize()

        # 收局
        # chess_flow.collect_pieces_at_end()

        # 布局
        # chess_flow.setup_initial_board()

        # 开始对弈
        chess_flow.play_game()

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断程序")
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        # import traceback
        # traceback.print_exc()
        chess_flow.report_error(str(e))
    finally:
        # 清理资源
        chess_flow.cleanup()

if __name__ == "__main__":
    main()