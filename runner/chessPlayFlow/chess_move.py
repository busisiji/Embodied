import asyncio
import copy
import os
import time

import cv2
import numpy as np

from parameters import WORLD_POINTS_R, WORLD_POINTS_RCV, WORLD_POINTS_B, CHESS_POINTS_R, CHESS_POINTS_RCV_H, \
    CHESS_POINTS_B, CHESS_POINTS_RCV_L, RCV_CAMERA, POINT_DOWN, POINT_RCV_DOWN, RED_CAMERA, BLACK_CAMERA, PIECE_SIZE, \
    IO_QI, RCV_H_LAY, SAC_CAMERA, POINT_SAC_DOWN
from src.cchessAI import cchess
from utils.calibrationManager import multi_camera_pixel_to_world, chess_to_world_position, get_area_center


class ChessPlayFlowMove():
    def __init__(self, parent):
        self.parent = parent
    # 移动
    def move_home(self,from_col=0,type='poi'):
        if from_col >=5:
            if type=='cam':
                self.parent.point_home = self.parent.args.black_camera_position
            elif type=='poi':
                self.parent.point_home = self.parent.args.black_position
        else:
            self.parent.point_home = self.parent.args.red_camera_position
        self.parent.urController.point_o = self.parent.point_home
        self.parent.urController.run_point_j(self.parent.point_home)

    def point_move(self,from_point,to_point,home_row=[0,0],is_run_box=False):
        """移动棋子"""
        from_x_world, from_y_world, pick_height = from_point
        to_x_world, to_y_world, place_height = to_point
        from_row , to_row = home_row

        # 移动到起始位置上方 (使用安全高度) 到不了角落点的上方
#         self.parent.urController.set_speed(0.8)
        self.move_home(from_row)
        # time.sleep(3)

        # 降低到吸取高度
        print("👇 降低到吸取高度")
#         self.parent.urController.set_speed(0.5)
        self.parent.urController.move_to(from_x_world, from_y_world, pick_height+15, use_safety=False)
        # time.sleep(1)
        self.parent.urController.move_to(from_x_world, from_y_world, pick_height, use_safety=False)
#         time.sleep(1)

        # 吸取棋子
        print("🫳 吸取棋子")
        self.parent.urController.set_do(IO_QI, 1)  # 吸合
        time.sleep(0.5)
        self.parent.urController.move_to(from_x_world, from_y_world, pick_height+15, use_safety=False)
#         time.sleep(1)

        # 抬起棋子到安全高度
        print("👆 抬起棋子到安全高度")
#         self.parent.urController.set_speed(0.8)
        self.move_home(from_row)
#         time.sleep(1)


        # 移动到目标位置上方（使用安全高度）
        if is_run_box:
            self.parent.urController.move_to(self.parent.box_center[0], self.parent.box_center[1], POINT_RCV_DOWN[2])
        else:
            print(f"📍 移动到目标位置上方: ({to_x_world}, {to_y_world})")
            self.move_home(to_row)
    #         time.sleep(1)

        # 降低到放置高度
        print("👇 降低到放置高度")
#         self.parent.urController.set_speed(0.5)
        self.parent.urController.move_to(to_x_world, to_y_world, POINT_RCV_DOWN[2])
#         time.sleep(1)

        if is_run_box:
            self.parent.urController.move_to(to_x_world, to_y_world, place_height+20)
            self.parent.urController.set_speed(0.3)

        self.parent.urController.move_to(to_x_world, to_y_world, place_height)
        if is_run_box:
            self.parent.urController.set_speed(0.8)
#         time.sleep(1)

        # 放置棋子
        print("🤲 放置棋子")
        self.parent.urController.set_do(IO_QI, 0)  # 释放
        time.sleep(0.5)
        self.parent.urController.move_to(to_x_world, to_y_world, POINT_RCV_DOWN[2])
#         time.sleep(1)

    def execute_move(self, move_uci):
        """
        执行移动操作前检查目标位置及周围位置的偏差

        Args:
            move_uci: 移动的UCI表示
        """
        print(f"🦾 执行移动: {move_uci}")
        pick_height = POINT_DOWN[0]

        # 解析移动 (UCI格式: 列行列表行) 简谱坐标系
        from_col = ord(move_uci[0]) - ord('a')  # 0-8 (a-i)
        from_row = int(move_uci[1])  # 0-9 (0-9)
        to_col = ord(move_uci[2]) - ord('a')  # 0-8 (a-i)
        to_row = int(move_uci[3])  # 0-9 (0-9)

        # 转换为数组行索引 数组坐标系
        from_row_idx = 9 - from_row
        to_row_idx = 9 - to_row

        # 检查目标位置及周围位置的偏差，如果有偏差超过容忍度则不断重新检查直到没有偏差为止
        print("🔍 检查目标位置及周围棋子位置偏差...")
        while not self.check_target_position_and_surroundings(from_row,from_col,to_row,to_col):
            if self.parent.surrendered:
                return
            if to_row <= 4:
                half_board = 'red'
            else:
                half_board = 'black'
            self.wait_for_player_adjustment(half_board=half_board)

                # 检查是否投降
            if self.parent.surrendered:
                self.parent.gama_over('surrender')
                return

        # 将棋盘坐标转换为世界坐标
        # 使用存储的像素坐标来提高精度
        piece_key = f"{from_row_idx}{from_col}"  # 使用数组索引
        if piece_key in self.parent.piece_pixel_positions:
            # 使用之前识别的精确像素坐标
            pixel_x, pixel_y = self.parent.piece_pixel_positions[piece_key]

            # 根据半区类型转换为世界坐标
            if from_row <= 4:  # 判断是红方还是黑方半区
                from_x_world, from_y_world = multi_camera_pixel_to_world(pixel_x, pixel_y, self.parent.inverse_matrix_r,
                                                                         "RED_CAMERA")
            else:
                from_x_world, from_y_world = multi_camera_pixel_to_world(pixel_x, pixel_y, self.parent.inverse_matrix_b,
                                                                         "BLACK_CAMERA")
            print('像素坐标：', pixel_x, pixel_y)
        else:
            # 如果没有存储的像素坐标，则使用原来的计算方法作为备选
            if from_row <= 4:
                half_board = 'red'
            else:
                half_board = 'black'
            from_x_world, from_y_world = chess_to_world_position(from_col, from_row, half_board)

        # 目标位置世界坐标转换
        if to_row <= 4:
            half_board = 'red'
        else:
            half_board = 'black'
        to_x_world, to_y_world = chess_to_world_position(to_col, to_row, half_board)
        print('世界坐标：', from_x_world, from_y_world, " to ", to_x_world, to_y_world)

        # 检查目标位置是否有棋子（即将被吃掉）
        target_piece_key = f"{to_row_idx}{to_col}"
        if self.parent.previous_positions[to_row_idx][to_col] != '.':
            captured_piece = self.parent.previous_positions[to_row_idx][to_col]
            print(f"⚔️ 吃掉棋子: {self.parent.piece_map[captured_piece]}")

            # 记录被吃的棋子信息，用于悔棋时恢复
            self.parent.captured_pieces_history[target_piece_key] = {
                'piece': captured_piece,
                'move': move_uci,
                'position': (to_row_idx, to_col)
            }

            # 移动被吃的棋子到弃子区
            self.move_piece_to_area(to_row_idx, to_col)

        # 移动棋子
        self.point_move([from_x_world, from_y_world, pick_height],
                        [to_x_world, to_y_world, pick_height],
                        [from_row, to_row])

        # 回到初始位置
        print("🏠 返回初始位置")
#         self.parent.urController.set_speed(0.5)
        self.move_home()
        print("✅ 移动执行完成")

        if self.parent.args.use_api:
            # 报告机器人移动
            chinese_notation = self.parent.uci_to_chinese_notation(move_uci, self.parent.previous_positions)
            self.parent.report_move("robot", move_uci, chinese_notation)

    def move_piece_to_area(self, row, col):
        """
        移动被吃的棋子到弃子区域的空位

        Args:
            row: 棋子所在行
            col: 棜子所在列
        """
        pick_height = POINT_DOWN[0]
        piece_key = f"{row}{col}"
        pixel_x, pixel_y = self.parent.piece_pixel_positions[piece_key]

        # 根据半区类型转换为世界坐标
        camera_type = "RED_CAMERA" if (9-row) <= 4 else "BLACK_CAMERA"
        inverse_matrix = self.parent.inverse_matrix_r if  (9-row) <= 4 else self.parent.inverse_matrix_b
        from_x_world, from_y_world = multi_camera_pixel_to_world(pixel_x, pixel_y,inverse_matrix, camera_type)
        print('像素坐标：', pixel_x, pixel_y)

        # 计算弃子区域偏移位置
        side = 30
        offset_map = {
            1: (-side, +side),
            2: (+side, +side),
            3: (+side, -side),
            4: (-side, -side)
        }
        mod = self.parent.sac_nums % 5
        if mod in offset_map:
            dx, dy = offset_map[mod]
            sac_camera = [SAC_CAMERA[0] + dx, SAC_CAMERA[1] + dy] + SAC_CAMERA[2:]
        else:
            sac_camera = SAC_CAMERA

        to_x_world, to_y_world = sac_camera[0], sac_camera[1]
        place_height = POINT_SAC_DOWN[1]

        # 使用 point_move 函数执行移动操作
        self.point_move(
            [from_x_world, from_y_world, pick_height],
            [to_x_world, to_y_world, place_height],
            [9-row, row]  # home_row 参数，用于控制 move_home 的行为
        )

        # 复位到标准弃子区域中心点上方
#         self.parent.urController.set_speed(0.5)
        self.parent.urController.run_point_j(SAC_CAMERA)
        self.parent.sac_nums += 1

    # 棋盘
    def visualize_chessboard(self, chess_result):
        """
        可视化棋盘布局

        Args:
            chess_result: 棋盘状态二维数组

        Returns:
            numpy数组: 可视化的棋盘图像
        """
        # 创建一个空白图像 (500x500 pixels)
        board_size = 500
        cell_size = board_size // 10  # 每个格子的大小
        img = np.ones((board_size, board_size, 3), dtype=np.uint8) * 255  # 白色背景

        # 绘制棋盘网格
        for i in range(11):  # 10行+1
            # 横线
            cv2.line(img, (0, i * cell_size), (9 * cell_size, i * cell_size), (0, 0, 0), 1)
            if i < 10:  # 竖线
                cv2.line(img, (i * cell_size, 0), (i * cell_size, 10 * cell_size), (0, 0, 0), 1)

        # 绘制九宫格斜线
        # 红方九宫格
        cv2.line(img, (3 * cell_size, 0), (5 * cell_size, 2 * cell_size), (0, 0, 0), 1)
        cv2.line(img, (5 * cell_size, 0), (3 * cell_size, 2 * cell_size), (0, 0, 0), 1)

        # 黑方九宫格
        cv2.line(img, (3 * cell_size, 7 * cell_size), (5 * cell_size, 9 * cell_size), (0, 0, 0), 1)
        cv2.line(img, (5 * cell_size, 7 * cell_size), (3 * cell_size, 9 * cell_size), (0, 0, 0), 1)


        # 在对应位置绘制棋子
        for row in range(10):
            for col in range(9):
                piece = chess_result[row][col]
                if piece != '.':
                    # 计算棋子中心位置
                    center_x = int(col * cell_size + cell_size // 2)
                    center_y = int(row * cell_size + cell_size // 2)

                    # 绘制棋子圆形
                    is_red = piece.isupper()  # 大写为红方
                    color = (0, 0, 255) if is_red else (0, 0, 0)  # 红方用红色，黑方用黑色
                    cv2.circle(img, (center_x, center_y), cell_size // 2 - 5, color, -1)
                    cv2.circle(img, (center_x, center_y), cell_size // 2 - 5, (0, 0, 0), 2)

                    # 绘制棋子文字
                    # text = piece_map.get(piece, piece)
                    text = piece
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    text_x = center_x - text_size[0] // 2
                    text_y = center_y + text_size[1] // 2
                    text_color = (255, 255, 255) if is_red else (255, 255, 255)  # 白色文字
                    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        return img
    def _calculate_piece_deviation(self, row, col, pixel_x, pixel_y,tolerance=10):
        """
        计算单个棋子位置偏差的通用函数

        Args:
            row: 棋子行号 (0-9) 简谱坐标系
            col: 棋子列号 (0-8) 简谱坐标系
            pixel_x: 棋子像素坐标x
            pixel_y: 棋子像素坐标y

        Returns:
            dict: 包含实际位置、标准位置和偏差信息的字典
        """
        # 转换为世界坐标
        if row <= 4:  # 红方区域(0-4行)
            x_world, y_world = multi_camera_pixel_to_world(pixel_x, pixel_y, self.parent.inverse_matrix_r, "RED_CAMERA")
            half_board = "red"
        else:  # 黑方区域(5-9行)
            x_world, y_world = multi_camera_pixel_to_world(pixel_x, pixel_y, self.parent.inverse_matrix_b, "BLACK_CAMERA")
            half_board = "black"

        # 计算标准位置的世界坐标
        standard_x, standard_y = chess_to_world_position(col, row, half_board)

        # 计算偏差距离
        cx = round(x_world - standard_x,2)
        cy = round(y_world - standard_y,2)
        distance = np.sqrt((x_world - standard_x)**2 + (y_world - standard_y)**2)
        is_deviation_exceeded = distance > tolerance
        # if row == 0:
        #     print('测试',col,distance,x_world - standard_x,y_world - standard_y)
        #     if y_world - standard_y >= 1:
        #         is_deviation_exceeded = True
        return {
            'world_position': (x_world, y_world),
            'standard_position': (standard_x, standard_y),
            'deviation_x':cx,
            'deviation_y':cy,
            'distance':distance,
            'is_deviation_exceeded': is_deviation_exceeded
        }

    def check_target_position_and_surroundings(self, from_row, from_col, target_row, target_col, tolerance=40):
        """
        检查目标位置及周围位置的棋子是否偏离标准位置，以及棋子之间距离是否过近

        Args:
            from_row: 起始行 (0-9) 简谱坐标系
            from_col: 起始列 (0-8) 简谱坐标系
            target_row: 目标行 (0-9) 简谱坐标系
            target_col: 目标列 (0-8) 简谱坐标系
            tolerance: 偏差容忍度(mm)

        Returns:
            bool: True表示没有问题，False表示存在问题
        """
        # 定义要检查的位置：目标位置及其周围8个位置
        surrounding_positions = [
            (target_row, target_col + 1),  # 上方
            (target_row - 1, target_col),  # 左侧
            (target_row + 1, target_col),  # 右侧
            (target_row, target_col - 1),  # 下方
        ]

        # 从检查位置中移除起始位置（如果存在）
        if (from_row, from_col) in surrounding_positions:
            surrounding_positions.remove((from_row, from_col))

        # 收集所有相关位置的棋子世界坐标
        piece_world_positions = {}

        # 先收集目标位置和周围位置的棋子世界坐标
        for row, col in surrounding_positions:
            row_idx = 9 - row
            piece_key = f"{row_idx}{col}"
            # 检查该位置是否有棋子
            if piece_key in self.parent.piece_pixel_positions:
                # 获取当前棋子的实际位置
                pixel_x, pixel_y = self.parent.piece_pixel_positions[piece_key]

                # 转换为世界坐标
                if row <= 4:  # 红方区域(0-4行)
                    x_world, y_world = multi_camera_pixel_to_world(pixel_x, pixel_y, self.parent.inverse_matrix_r,
                                                                   "RED_CAMERA")
                else:  # 黑方区域(5-9行)
                    x_world, y_world = multi_camera_pixel_to_world(pixel_x, pixel_y, self.parent.inverse_matrix_b,
                                                                   "BLACK_CAMERA")

                piece_world_positions[(row, col)] = (x_world, y_world)

                # # 使用通用函数计算偏差
                # deviation_data = self._calculate_piece_deviation(row, col, pixel_x, pixel_y, tolerance)
                # deviation_info[(row, col)] = deviation_data
                #
                # # 如果偏差超过容忍度，给出警告
                # if deviation_data['is_deviation_exceeded']:
                #     print(
                #         f"⚠️ 棋子({row_idx+1},{col+1})偏离标准位置X方向{abs(deviation_data['world_position'][0] - deviation_data['standard_position'][0]):.2f}mm，Y方向{abs(deviation_data['world_position'][1] - deviation_data['standard_position'][1]):.2f}mm，超过{tolerance}mm阈值")
                #     asyncio.run(self.parent.speak_cchess(
              #         f"第{row_idx+1}行,第{col+1}列的棋子偏离标准位置"))

        # 检查目标位置与周围棋子之间的距离，防止落子时碰撞
        # 目标位置世界坐标转换
        if target_row <= 4:
            half_board = 'red'
        else:
            half_board = 'black'
        x_world, y_world = chess_to_world_position(target_col, target_row, half_board)

        # 检查与周围棋子的距离
        for row, col in piece_world_positions.keys():
            neighbor_x, neighbor_y = piece_world_positions[(row, col)]
            # 计算与周围棋子的距离
            distance = np.sqrt((x_world - neighbor_x) ** 2 + (y_world - neighbor_y) ** 2)

            # 如果最近的棋子距离小于容忍度，发出警告并报告
            text = ''
            if distance < tolerance:
                row_idx = 9 - row
                point_type_str = self.parent.chess_positions[row_idx][col]
                point_type = self.parent.piece_map[point_type_str] if point_type_str in self.parent.piece_map else '未知'
                print(f"⚠️ 第({row_idx + 1},{col + 1})的{point_type}距离过近: {distance:.2f}mm，可能造成碰撞")

                if target_row > row :
                    text = f"请将第{row_idx + 1}行,第{col + 1}列的{point_type}向下移动"
                elif target_row < row :
                    text = f"请将第{row_idx + 1}行,第{col + 1}列的{point_type}向上移动"
                elif target_col > col :
                    text = f"请将第{row_idx + 1}行,第{col + 1}列的{point_type}向左移动"
                elif target_col < col :
                    text = f"请将第{row_idx + 1}行,第{col + 1}列的{point_type}向右移动"

                if self.parent.args.use_api:
                    # 报告偏移信息
                    deviation_x = abs(x_world - neighbor_x)
                    deviation_y = abs(y_world - neighbor_y)
                    self.parent.report_piece_deviation(row_idx, col, deviation_x, deviation_y, distance)

                asyncio.run(self.parent.speak_cchess(text))
                return False

        return True
    def _move_piece_back(self, from_row, from_col, to_row, to_col):
        """
        物理上将棋子从目标位置移回起始位置

        Args:
            from_row, from_col: 起始位置
            to_row, to_col: 目标位置
        """
        print(f"🔄 物理移动棋子从 ({to_row},{to_col}) 回到 ({from_row},{from_col})")

        pick_height = POINT_DOWN[0]

        # 计算世界坐标
        # 起始位置（现在是目标位置）
        if to_row <= 4:
            half_board = 'red'
            from_x_world, from_y_world = chess_to_world_position(to_col, to_row, half_board)
        else:
            half_board = 'black'
            from_x_world, from_y_world = chess_to_world_position(to_col, to_row, half_board)

        # 目标位置（现在是起始位置）
        if from_row <= 4:
            half_board = 'red'
            to_x_world, to_y_world = chess_to_world_position(from_col, from_row, half_board)
        else:
            half_board = 'black'
            to_x_world, to_y_world = chess_to_world_position(from_col, from_row, half_board)

        print(f'世界坐标：{from_x_world}, {from_y_world} -> {to_x_world}, {to_y_world}')

        # 执行移动
        self.point_move(
            [from_x_world, from_y_world, pick_height],
            [to_x_world, to_y_world, pick_height],
            [to_row, from_row]
        )

        # 回到初始位置
        print("🏠 返回初始位置")
        #         self.parent.urController.set_speed(0.5)
        self.move_home()

    def execute_updata_move(self, move_uci):
        """
        执行移动并更新状态
        """

        if not move_uci:
            return False

        try:
            # 验证移动是否合法
            move = cchess.Move.from_uci(move_uci)
            if move not in self.parent.board.legal_moves:
                print(f"无效移动: {move_uci}")
                return False

            # 执行移动
            self.execute_move(move_uci)
            # 执行移动到MainGame并保存历史信息

            # 更新棋盘状态
            self.parent.maingame.mgInit.move_to(self.parent.cBranch.uci_to_mg_coords(move_uci))
            self.parent.board.push(move)
            self.updat_previous_positions_after_move(move_uci)
            # 记录移动历史
            self.parent.move_history.append(move_uci)
            self.parent.his_chessboard[self.parent.move_count] = copy.deepcopy(self.parent.previous_positions)
            self.parent.move_count = len(self.parent.move_history)

            # 切换回合
            self.parent.set_side()

            # 显示更新后的棋盘
            if self.parent.args.show_board:
                self.parent.game.graphic(self.parent.board)

            return True

        except Exception as e:
            print(f"执行移动命令出错: {e}")
            self.parent.spaek_cchess(f"执行移动命令出错")
            return False
    def updat_previous_positions_after_move(self, move_uci):
        """
        根据移动UCI更新previous_positions状态
        """
        # 解析移动
        from_col= ord(move_uci[0]) - ord('a')
        from_row= int(move_uci[1])
        to_col=  ord(move_uci[2]) - ord('a')
        to_row= int(move_uci[3])

        # 将行列转换为数组索引 (棋盘坐标到数组索引)
        from_row_idx = 9 - from_row
        from_col_idx = from_col
        to_row_idx = 9 - to_row
        to_col_idx = to_col

        # 移动棋子
        piece = self.parent.previous_positions[from_row_idx][from_col_idx]
        self.parent.previous_positions[to_row_idx][to_col_idx] = piece
        self.parent.previous_positions[from_row_idx][from_col_idx] = '.'
    def wait_for_player_adjustment(self,half_board=None):
        """
        等待玩家调整棋子位置
        """
        print("⏳ 等待玩家调整棋子位置...")
        asyncio.run(self.parent.speak_cchess("请调整棋子位置,5秒后重新检测棋盘"))
        # 等待一段时间让玩家有时间调整
        time.sleep(5)
        print("🔍 重新检测棋盘...")
        # 重新识别棋盘状态
        asyncio.run(self.parent.speak_cchess("正在重新检测棋盘"))
        self.parent.cCamera.recognize_chessboard(True,half_board=half_board)
