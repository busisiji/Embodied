import asyncio
import os
import time

import numpy as np

from parameters import WORLD_POINTS_R, WORLD_POINTS_RCV, WORLD_POINTS_B, CHESS_POINTS_R, CHESS_POINTS_RCV_H, \
    CHESS_POINTS_B, CHESS_POINTS_RCV_L, RCV_CAMERA, POINT_DOWN, POINT_RCV_DOWN, RED_CAMERA, BLACK_CAMERA, PIECE_SIZE, \
    IO_QI, RCV_H_LAY
from src.cchessAI import cchess
from src.cchessYolo.detect_chess_box import calculate_4x4_collection_positions
from utils.calibrationManager import chess_to_world_position, get_area_center

class ChessPlayFlowBranch():
    # 分支函数
    def __init__(self, parent):
        self.parent = parent
    # 收棋
    def collect_pieces_at_end(self):
        """
        收局函数：识别棋盒位置，然后将所有棋子按颜色分类放入棋盒
        """
        try:
            print("🧹 开始收局...")
            asyncio.run(self.parent.speak_cchess("开始收局"))

            while 1:
                # 检查游戏状态
                surrendered, paused = self.parent.check_game_state()
                if surrendered:
                    return


                self.parent.urController.run_point_j(RCV_CAMERA)

                # 1. 识别棋盒位置（支持3或4个圆）
                chess_box_points = self.parent.cCamera.detect_chess_box(max_attempts=20)

                # 如果无法识别到棋盒位置，则报错
                if chess_box_points is None:
                    print("无法识别棋盒位置")
                    asyncio.run(self.parent.speak_cchess("无法识别棋盒位置"))
                    return

                print("✅ 成功识别棋盒位置")
                asyncio.run(self.parent.speak_cchess("成功识别棋盒位置"))

                chess_box_points = np.array([[point[0], point[1]] for point in chess_box_points])
                print("像素四角", chess_box_points)
                # 转换为世界坐标检查尺寸 注意镜像翻转
                world_corner_0 = self.parent.cUtils.pixel_to_world_cchess(chess_box_points[2][0], chess_box_points[2][1],
                                                             self.parent.inverse_matrix_r, "RCV_CAMERA",
                                                             use_tps=True)  # 棋盒左上角
                world_corner_1 = self.parent.cUtils.pixel_to_world_cchess(chess_box_points[3][0], chess_box_points[3][1],
                                                             self.parent.inverse_matrix_r, "RCV_CAMERA",
                                                             use_tps=True)  # 棋盒右上角
                world_corner_2 = self.parent.cUtils.pixel_to_world_cchess(chess_box_points[0][0], chess_box_points[0][1],
                                                             self.parent.inverse_matrix_r, "RCV_CAMERA",
                                                             use_tps=True)  # 棋盒右下角
                world_corner_3 = self.parent.cUtils.pixel_to_world_cchess(chess_box_points[1][0], chess_box_points[1][1],
                                                             self.parent.inverse_matrix_r, "RCV_CAMERA",
                                                             use_tps=True)  # 棋盒左下角

                topLeft = world_corner_0[0] - 0, world_corner_0[1] + 5
                topRight = world_corner_1[0] + 0, world_corner_1[1] + 5
                bottomRight = world_corner_2[0] + 0, world_corner_2[1] - 5
                bottomLeft = world_corner_3[0] - 0, world_corner_3[1] - 5
                chess_box_points = [topLeft, topRight, bottomRight, bottomLeft]

                if not self.parent.urController.is_point_reachable(bottomLeft[0], bottomLeft[1],
                                                                   POINT_RCV_DOWN[1] + 20):
                    print("机械臂无法到达棋盒，请重新放置到靠近机械臂的位置！")
                    asyncio.run(self.parent.speak_cchess("机械臂无法到达棋盒，请重新放置到靠近机械臂的位置！"))
                    raise ValueError("机械臂无法到达棋盒，请将棋盒放置到靠近机械臂的位置")

                # 计算4x4网格的世界坐标位置
                collection_positions = calculate_4x4_collection_positions(chess_box_points)
                print('棋盒坐标：', topLeft, topRight, bottomRight, bottomLeft)
                print("棋盒16位：", collection_positions)

                world_width = np.linalg.norm(np.array(topRight) - np.array(topLeft))
                world_height = np.linalg.norm(np.array(topLeft) - np.array(bottomLeft))
                center_x = (topLeft[0] + topRight[0] + bottomRight[0] + bottomLeft[0]) / 4
                center_y = (topLeft[1] + topRight[1] + bottomRight[1] + bottomLeft[1]) / 4
                self.parent.box_center = [center_x, center_y]

                print(f"✅ 棋盒尺寸检查通过，格子尺寸: {world_width:.2f}mm x {world_height:.2f}mm")

                # 3. 识别红方棋子并移动到棋盒下层
                print("🔴 开始收集红方棋子...")
                asyncio.run(self.parent.speak_cchess("开始收集红方棋子"))
                self.collect_half_board_pieces("red", collection_positions)

                # 4. 识别黑方棋子并移动到棋盒上层
                print("⚫ 开始收集黑方棋子...")
                asyncio.run(self.parent.speak_cchess("开始收集黑方棋子"))
                self.collect_half_board_pieces("black", collection_positions)

                print("✅ 收局完成")
                asyncio.run(self.parent.speak_cchess("收局完成"))
                time.sleep(5)
                return
        except Exception as e:
            print(e)
            asyncio.run(self.parent.speak_cchess("收局失败"))
            time.sleep(5)

    def collect_half_board_pieces(self, side, collection_positions):
        """
        收集指定颜色的棋子到棋盒

        Args:
            side: 收集棋子颜色("red"或"black")
            collection_positions: 收集位置列表
        """
        pick_height = POINT_DOWN[0]
        place_height = POINT_RCV_DOWN[0] if side == "red" else POINT_RCV_DOWN[1]  # red放底层，black放上层

        # 根据side决定要收集的棋子类型（大写为红方，小写为黑方）
        if side == "red":
            # 收集所有红方棋子（大写字母）
            target_class_names = ['R', 'N', 'B', 'A', 'K', 'C', 'P']
        else:
            # 收集所有黑方棋子（小写字母）
            target_class_names = ['r', 'n', 'b', 'a', 'k', 'c', 'p']

        # 1. 处理红方半区
        print(f"🔍 在红方半区寻找{side}方棋子...")
        red_piece_positions = self._collect_pieces_from_half_board(
            RED_CAMERA, "RED_CAMERA", target_class_names)

        print(f"✅ 找到{len(red_piece_positions)}个{side}方棋子")

        # 按从左到右、从上到下的顺序排序
        red_piece_positions.sort(key=lambda p: (p[1], p[0]))  # 按y坐标升序，x坐标升序

        # 立即移动红方半区识别到的棋子到棋盒
        position_index = 0
        print(f"🚚 开始移动红方半区识别到的{side}方棋子...")
        for x_world, y_world in red_piece_positions:
            if position_index >= len(collection_positions):
                print("⚠️ 棋盒位置不足")
                raise ValueError("棋盒位置不足")

            # 跳过底边
            if y_world >= -400:
                continue

            # 目标位置
            target_x, target_y = collection_positions[position_index]

            self.parent.cMove.point_move(
                [x_world, y_world, pick_height],
                [target_x, target_y, place_height],  # 根据side决定放置高度
                [0, 0],  # home_row 参数，控制 move_home 行为
                is_run_box = True
            )

            # 检查放置的棋子是否与左右棋子距离合适
            # self.parent._check_and_adjust_piece_position(position_index, collection_positions,
            #                                              target_x, target_y, place_height,
            #                                              pick_height)

            position_index += 1
            print(f"✅ 将{side}方棋子从({x_world:.1f},{y_world:.1f})放置到棋盒位置({position_index}/{len(red_piece_positions)})")

        print(f"✅ 完成移动红方半区{side}方棋子，共移动{position_index}个")

        # 2. 处理黑方半区
        print(f"🔍 在黑方半区寻找{side}方棋子...")
        black_piece_positions = self._collect_pieces_from_half_board(
            BLACK_CAMERA, "BLACK_CAMERA", target_class_names)
        print(f"✅ 找到{len(black_piece_positions)}个{side}方棋子")

        # 按从左到右、从上到下的顺序排序
        black_piece_positions.sort(key=lambda p: (p[1], p[0]))  # 按y坐标升序，x坐标升序

        # 移动黑方半区识别到的棋子到棋盒
        print(f"🚚 开始移动黑方半区识别到的{side}方棋子...")
        for x_world, y_world in black_piece_positions:
            if position_index >= len(collection_positions):
                print("⚠️ 棋盒位置不足")
                break

            # 目标位置
            target_x, target_y = collection_positions[position_index]

            self.parent.cMove.point_move(
                [x_world, y_world, pick_height],
                [target_x, target_y, place_height],  # 根据side决定放置高度
                [9, 9],  # home_row 参数，控制 move_home 行为
                is_run_box=True
            )

            # 检查放置的棋子是否与左右棋子距离合适
            # self.parent._check_and_adjust_piece_position(position_index, collection_positions,
            #                                              target_x, target_y, place_height,
            #                                              pick_height)

            position_index += 1
            print(f"✅ 将{side}方棋子从({x_world:.1f},{y_world:.1f})放置到棋盒位置({position_index}/{len(black_piece_positions)})")

        print(f"✅ 完成收集{side}方棋子，共收集{position_index}个")

    def _check_and_adjust_piece_position(self, position_index, collection_positions,
                                         target_x, target_y, place_height, pick_height):
        """
        检查放置的棋子与左右相邻棋子的距离，如果距离不合适则重新放置

        Args:
            position_index: 当前放置棋子的位置索引
            collection_positions: 所有收集位置
            target_x, target_y: 当前棋子的目标位置
            place_height: 放置高度
            pick_height: 吸取高度

        Returns:
            bool: 是否需要重新放置
        """
        # 移动到RCV_CAMERA点检查
        self.parent.urController.run_point_j(RCV_CAMERA)

        # 捕获图像检查棋子位置
        rcv_image, rcv_depth = self.parent.cCamera.capture_stable_image(is_chessboard=False)
        if rcv_image is None:
            print("⚠️ 无法捕获收子区图像进行检查")
            return False

        # 使用YOLO检测器识别棋盒中的棋子
        objects_info, _ = self.parent.detector.detect_objects_with_height(
            rcv_image, rcv_depth,
            conf_threshold=self.parent.args.conf,
            iou_threshold=self.parent.args.iou,
            mat=self.parent.m_rcv
        )

        # 查找刚刚放置的棋子
        current_piece = None
        min_distance = float('inf')

        for obj_info in objects_info:
            pixel_x, pixel_y = obj_info['center']
            # 转换像素坐标到世界坐标
            x_world, y_world = self.parent.cUtils.pixel_to_world_cchess(
                pixel_x, pixel_y, self.parent.inverse_matrix_r ,camera_type="RCV_CAMERA")

            # 计算与目标位置的距离
            distance = np.sqrt((x_world - target_x)**2 + (y_world - target_y)**2)
            if distance < min_distance:
                min_distance = distance
                current_piece = {
                    'x_world': x_world,
                    'y_world': y_world,
                    'pixel_x': pixel_x,
                    'pixel_y': pixel_y
                }

        if current_piece is None:
            print("⚠️ 未找到刚放置的棋子")
            return False

        # 检查与左右相邻位置棋子的距离
        left_neighbor_ok = True
        right_neighbor_ok = True

        # 检查左边相邻位置
        if position_index%4 > 0:
            left_pos_x, left_pos_y = collection_positions[position_index - 1]
            distance_to_left = np.sqrt((current_piece['x_world'] - left_pos_x)**2 +
                                       (current_piece['y_world'] - left_pos_y)**2)
            if distance_to_left >= PIECE_SIZE + 2:  # 距离阈值设为40mm
                left_neighbor_ok = False
                print(f"⚠️ 与左边棋子距离过远: {distance_to_left:.2f}mm")

        # 检查上边相邻位置
        if position_index > 3:
            right_pos_x, right_pos_y = collection_positions[position_index - 4]
            distance_to_right = np.sqrt((current_piece['x_world'] - right_pos_x)**2 +
                                        (current_piece['y_world'] - right_pos_y)**2)
            if distance_to_right >= PIECE_SIZE + 2:  # 距离阈值设为40mm
                right_neighbor_ok = False
                print(f"⚠️ 与上边棋子距离过远: {distance_to_right:.2f}mm")

        # 如果与左右棋子的距离都不合适，需要重新放置
        if not left_neighbor_ok or not right_neighbor_ok:
            print("🔄 重新放置棋子以调整位置")
            # 吸取棋子
            self.parent.urController.move_to(current_piece['x_world'], current_piece['y_world'], place_height + 20)
            self.parent.urController.move_to(current_piece['x_world'], current_piece['y_world'], place_height)
            self.parent.urController.set_do(IO_QI, 1)  # 吸合
            self.parent.urController.move_to(current_piece['x_world'], current_piece['y_world'], place_height + 20)

            # 放置棋子到原位置
            self.parent.urController.move_to(target_x, target_y, place_height + 20)
            self.parent.urController.move_to(target_x, target_y, place_height)
            self.parent.urController.set_do(IO_QI, 0)  # 释放
            self.parent.urController.move_to(target_x, target_y, place_height + 20)

            return True

        return False

    def _collect_pieces_from_half_board(self, camera_position, camera_type, target_class_names):
        """
        从指定半区收集目标棋子

        Args:
            camera_position: 相机位置
            camera_type: 相机类型 ("RED_CAMERA" 或 "BLACK_CAMERA")
            target_class_names: 目标棋子类型

        Returns:
            list: 棋子位置列表 [(x_world, y_world, row), ...]
        """
        piece_positions = []
        if camera_type == "RED_CAMERA":
            inverse_matrix = self.parent.inverse_matrix_r
        else:
            inverse_matrix = self.parent.inverse_matrix_b

        # 移动到拍照点
        self.parent.urController.run_point_j(camera_position)

        # 捕获图像
        image, depth = self.parent.cCamera.capture_stable_image(is_chessboard=False)
        if image is None:
            print(f"⚠️ 无法捕获{camera_type}图像")
            return piece_positions

        # 使用YOLO检测器识别棋子
        objects_info, results = self.parent.detector.detect_objects_with_height(
            image, depth,
            conf_threshold=self.parent.args.conf,
            iou_threshold=self.parent.args.iou
        )

        # 筛选出目标颜色的棋子
        for object_info in objects_info:
            if object_info['class_name'] in target_class_names:
                pixel_x, pixel_y = object_info['center']
                x_world, y_world = self.parent.cUtils.pixel_to_world_cchess(pixel_x, pixel_y,inverse_matrix, camera_type)
                piece_positions.append((x_world, y_world))

        # 保存识别结果（包括可视化检测结果）
        if self.parent.args.save_recognition_results:
            if camera_type == "RED_CAMERA":
                asyncio.run(self.parent.save_recognition_result_with_detections(
                    red_image=image, red_detections=results,move_count='half_board'
                ))

            else:
                asyncio.run(self.parent.save_recognition_result_with_detections(
                    black_image=image, black_detections=results,move_count='half_board'
                ))

        return piece_positions

    # 布局
    def setup_initial_board(self):
        """
        布局函数：从收子区取出棋子并按初始布局放到棋盘上
        先处理上层的黑方棋子，再处理下层的红方棋子
        """
        try:
            print("🎯 开始初始布局...")
            asyncio.run(self.parent.speak_cchess("开始初始布局"))

            # 定义中国象棋初始布局 (从上到下，从左到右)
            # 黑方在上半区(0-4行)，红方在下半区(5-9行)
            initial_layout = [
                ['r', 'n', 'b', 'a', 'k', 'a', 'b', 'n', 'r'],  # 0行 黑方
                ['.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 1行
                ['.', 'c', '.', '.', '.', '.', '.', 'c', '.'],  # 2行
                ['p', '.', 'p', '.', 'p', '.', 'p', '.', 'p'],  # 3行
                ['.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 4行
                ['.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 5行
                ['P', '.', 'P', '.', 'P', '.', 'P', '.', 'P'],  # 6行 红方
                ['.', 'C', '.', '.', '.', '.', '.', 'C', '.'],  # 7行
                ['.', '.', '.', '.', '.', '.', '.', '.', '.'],  # 8行
                ['R', 'N', 'B', 'A', 'K', 'A', 'B', 'N', 'R']  # 9行
            ]

            # 1. 处理上层黑方棋子
            print("⚫ 处理上层黑方棋子...")
            asyncio.run(self.parent.speak_cchess("正在布置黑方棋子"))
            for i in range(20):
                # 检查游戏状态
                surrendered, paused = self.parent.check_game_state()
                if surrendered:
                    return


                if self.setup_half_board_pieces("black", initial_layout):
                    break
                time.sleep(10)

            # 2. 处理下层红方棋子
            print("🔴 处理下层红方棋子...")
            asyncio.run(self.parent.speak_cchess("正在布置红方棋子"))
            for i in range(20):
                # 检查游戏状态
                surrendered, paused = self.parent.check_game_state()
                if surrendered:
                    return


                if self.setup_half_board_pieces("red", initial_layout):
                    break
                time.sleep(10)

            print("✅ 初始布局完成")
        except Exception as e:
            print(f"❌ 初始布局异常: {str(e)}")
            asyncio.run(self.parent.speak_cchess("初始布局异常"))
            raise e
    def setup_half_board_pieces(self, side, target_layout):
        """
        布置半区棋子，确保棋子类型与目标位置匹配

        Args:
            side: 棋子方("red"或"black")
            target_layout: 目标布局
        """
        # 移动到收子区拍照点
#         self.parent.urController.set_speed(0.8)
        self.parent.urController.run_point_j(RCV_CAMERA)
        # 捕获图像和深度信息
        rcv_image, rcv_depth = self.parent.cCamera.capture_stable_image(is_chessboard=False)
        if rcv_image is None:
            print("⚠️ 无法捕获收子区图像")
            return

        inverse_matrix = self.parent.inverse_matrix_rcv_h if side == "black" else self.parent.inverse_matrix_rcv_l

        # 使用YOLO检测器识别收子区的棋子（包含高度信息）
        objects_info, _ = self.parent.detector.detect_objects_with_height(
            rcv_image, rcv_depth,
            conf_threshold=self.parent.args.conf,
            iou_threshold=self.parent.args.iou,
            mat=self.parent.m_rcv
        )

        # 确定要处理的行范围和层
        if side == "black":
            rows = range(5, 10)
            # 上层棋子高度小于RCV_H_LAY
            is_target_layer = lambda h: h and h < RCV_H_LAY
            layer_name = "上层"
            target_class_names = ['r', 'n', 'b', 'a', 'k', 'c', 'p']  # 黑方棋子类型
        else:
            rows = range(0, 5)
            # 下层棋子高度大于等于RCV_H_LAY
            is_target_layer = lambda h: h and h >= RCV_H_LAY
            layer_name = "下层"
            target_class_names = ['R', 'N', 'B', 'A', 'K', 'C', 'P']  # 红方棋子类型

        pick_height = POINT_RCV_DOWN[1] if side == "black" else POINT_RCV_DOWN[0]
        print(f"📦 从收子区{layer_name}取{side}方棋子")

        # 创建棋子列表，按目标布局顺序排列
        target_pieces = []
        for row in rows:
            for col in range(9):
                piece = target_layout[9-row][col]
                if piece != '.' and piece in target_class_names:
                    target_pieces.append((row, col, piece))
        # 棋子棋盒位置列表
        available_pieces = {}

        # 从objects_info中提取棋子位置信息并按类型分类
        if objects_info:
            for i, obj_info in enumerate(objects_info):
                class_name = obj_info['class_name']
                # 检查是否为目标颜色的棋子
                if class_name not in target_class_names:
                    continue

                # 获取边界框中心点
                center_x, center_y = obj_info['center']

                # 根据高度判断是否为目标层
                height = obj_info.get('height', None)

                # 直接将像素坐标转换为世界坐标
                x_world, y_world = self.parent.cUtils.pixel_to_world_cchess(
                    center_x, center_y, inverse_matrix)

                # 按棋子类型分类存储
                if class_name not in available_pieces:
                    available_pieces[class_name] = []
                available_pieces[class_name].append((x_world, y_world, height))

        # 对每种类型的棋子按位置排序
        for piece_type in available_pieces:
            available_pieces[piece_type].sort(key=lambda p: (p[1], p[0]))  # 按y坐标升序，x坐标升序

        # 检查是否所有棋子都齐全
        required_pieces_count = {}
        for _, _, piece in target_pieces:
            required_pieces_count[piece] = required_pieces_count.get(piece, 0) + 1

        available_pieces_count = {}
        for piece_type, pieces in available_pieces.items():
            available_pieces_count[piece_type] = len(pieces)

        # 检查棋子是否完整
        is_complete = True
        missing_pieces = []

        for piece_type, required_count in required_pieces_count.items():
            available_count = available_pieces_count.get(piece_type, 0)
            if available_count < required_count:
                is_complete = False
                missing_pieces.append(f"{self.parent.piece_map[piece_type]}缺少{required_count - available_count}个")

        if not is_complete:
            print(f"⚠️ {side}方棋子不完整: {', '.join(missing_pieces)}")
            asyncio.run(self.parent.speak_cchess(f"{side}方棋子{', '.join(missing_pieces)}"))
            return  # 如果棋子不完整，直接返回不执行布置
        else:
            total_count = sum(available_pieces_count.values())
            print(f"✅ {side}方{total_count}个棋子齐全，开始布置")

        # 移动棋子到棋盘正确位置
        piece_counters = {piece: 0 for piece in target_class_names}  # 为每种棋子类型维护计数器

        for i, (target_row, target_col, target_piece) in enumerate(target_pieces):
            # 获取对应类型的下一个可用棋子
            if target_piece not in available_pieces or piece_counters[target_piece] >= len(available_pieces[target_piece]):
                print(f"⚠️ {layer_name}{side}方缺少棋子{target_piece}")
                continue

            # 获取该类型棋子的下一个可用实例
            piece_index = piece_counters[target_piece]
            x_world, y_world, piece_height = available_pieces[target_piece][piece_index]
            piece_counters[target_piece] += 1  # 增加该类型棋子的计数器

            # 计算目标位置世界坐标
            x_world_target, y_world_target = chess_to_world_position(target_col, target_row, side)
            place_height = POINT_DOWN[0]  # 放置高度

            rcv_center_x, rcv_center_y = get_area_center(CHESS_POINTS_RCV_H)
            rcv_world_x, rcv_world_y = self.parent.cUtils.pixel_to_world_cchess(
                    rcv_center_x, rcv_center_y, inverse_matrix)
            print(f"📥 将{side}方棋子{target_piece}从收子区放置到位置({target_row},{target_col})")

            # 移动到收子区拍照点
#             self.parent.urController.set_speed(0.8)
            # self.parent.urController.run_point_j(RCV_CAMERA)

            # 移动到中心点
            self.parent.urController.move_to(rcv_world_x, rcv_world_y, pick_height + 50)

            # 移动到棋子上方
            self.parent.urController.move_to(x_world, y_world, pick_height+20)

            # 降低到吸取高度
#             self.parent.urController.set_speed(0.5)
            self.parent.urController.move_to(x_world, y_world, pick_height)

            # 吸取棋子
            self.parent.urController.set_do(IO_QI, 1)  # 吸合


            # 抬起棋子到安全高度
#             self.parent.urController.set_speed(0.8)
            self.parent.urController.move_to(x_world, y_world, pick_height+20)

            # 移动到中心点
            self.parent.urController.move_to(rcv_world_x, rcv_world_y, pick_height+50)

            # 移动到棋盘上方
            col = 9 if side == "black" else 0
            self.parent.move_home(col)

            # 移动到目标位置上方
            self.parent.urController.move_to(x_world_target, y_world_target, place_height+20)

            # 降低到放置高度
#             self.parent.urController.set_speed(0.5)
            self.parent.urController.move_to(x_world_target, y_world_target, place_height+5)

            # 放置棋子
            self.parent.urController.set_do(IO_QI, 0)
            self.parent.urController.move_to(x_world_target, y_world_target, place_height+20)

            # 抬起机械臂到安全高度
#             self.parent.urController.set_speed(0.8)
            self.parent.cMove.move_home(col)

            print(f"✅ {side}方棋子{target_piece}已放置到位置({target_row},{target_col})")
        return True

    # 悔棋

    def undo_move(self, steps=2):
        """
        悔棋函数，将棋盘状态还原到前n步

        Args:
            steps: 要悔棋的步数，默认为1步
        """
        try:
            if self.parent.side == self.parent.args.robot_side:
                print(f"⚠️ 当前棋子方为 {self.parent.side}，无法悔棋")
                asyncio.run(self.parent.speak_cchess(f"机器人正在落子，无法悔棋"))
                raise Exception("机器人正在落子，无法悔棋")
            print(f"↩️ 执行悔棋，回到 {steps} 步前的状态")
            asyncio.run(self.parent.speak_cchess(f"正在执行悔棋"))
            self.parent.urController.hll(5)  # 红灯
            # 检查是否有足够的历史记录
            if len(self.parent.move_history) < steps:
                print(f"⚠️ 没有足够的移动历史，当前只有 {len(self.parent.move_history)} 步")
                asyncio.run(self.parent.speak_cchess("没有足够的移动历史"))
                return False

            # 从移动历史中获取要撤销的移动
            moves_to_undo = self.parent.move_history[-steps:]
            print(f".undo_move 将撤销的移动: {moves_to_undo}")

            # 逐步撤销移动
            for i in range(steps):
                move_uci = moves_to_undo[-(i+1)]  # 从最后一步开始撤销
                print(f"撤销移动: {move_uci}")

                # 解析UCI格式移动
                from_col = ord(move_uci[0]) - ord('a')  # 0-8 (a-i)
                from_row = int(move_uci[1])             # 0-9 (0-9 从下到上)
                to_col = ord(move_uci[2]) - ord('a')    # 0-8 (a-i)
                to_row = int(move_uci[3])               # 0-9 (0-9 从下到上)

                # 转换为数组索引
                from_row_idx = 9 - from_row
                to_row_idx = 9 - to_row

                # 检查目标位置是否有被吃的棋子需要恢复
                target_piece_key = f"{to_row_idx}{to_col}"
                if target_piece_key in self.parent.captured_pieces_history:
                    # 恢复被吃的棋子
                    captured_info = self.parent.captured_pieces_history[target_piece_key]
                    print(f"发现被吃的棋子需要恢复: {captured_info}")
                    asyncio.run(self.parent.speak_cchess(f"请将被吃的{self.parent.piece_map[captured_info['piece']]}放回棋盘"))

                    # 等待用户放回棋子
                    self.parent.cMove.wait_for_player_adjustment()


                # 物理上将棋子移回原位
                self.parent.cMove._move_piece_back(from_row, from_col, to_row, to_col)

            # 更新移动历史
            self.parent.move_history = self.parent.move_history[:-steps]

            # 更新全局变量 move_count 和 side
            self.parent.move_count = len(self.parent.move_history)
            self._update_side_after_undo()

            # 更新棋盘状态
            self._revert_board_state(steps)

            # 更新MainGame棋盘状态
            self._revert_maingame_state(steps)

            # 7. 显示更新后的棋盘
            if self.parent.args.show_board:
                self.parent.game.graphic(self.parent.board)

            print(f"✅ 悔棋完成，已回到 {steps} 步前的状态")
            asyncio.run(self.parent.speak_cchess("悔棋完成"))
            self.parent.is_undo = True
            return True
        except Exception as e:
            print(f"❌ 悔棋异常: {str(e)}")
            raise e

    def _revert_maingame_state(self, steps):
        """
        还原MainGame的棋盘状态

        Args:
            steps: 要还原的步数
        """
        print(f"🔄 还原MainGame棋盘状态，撤销 {steps} 步")

        # 重新初始化MainGame状态
        self.parent.maingame.restart_game()

        # 重新应用未被撤销的移动到MainGame
        moves_to_keep = self.parent.move_history
        for move_uci in moves_to_keep:
            try:
                move_mg = self.parent.cUtils.uci_to_mg_coords(move_uci)
                # 执行移动到MainGame并保存历史信息
                self.parent.maingame.mgInit.move_to(move_mg)

            except Exception as e:
                print(f"MainGame应用移动 {move_uci} 时出错: {e}")

    def _update_side_after_undo(self):
        """
        悔棋后更新当前回合方
        """
        # 根据已走步数和机器人执子方来确定当前回合方
        is_robot_turn = (self.parent.move_count + (0 if self.parent.args.robot_side == 'red' else 1)) % 2 == 1
        if not is_robot_turn:
            self.parent.side = self.parent.args.robot_side
        else:
            self.parent.side = 'black' if self.parent.args.robot_side == 'red' else 'red'
        print(f"🔄 悔棋后更新当前回合方为: {self.parent.side}")

    def _revert_board_state(self, steps):
        """
        还原棋盘逻辑状态

        Args:
            steps: 要还原的步数
        """
        print(f"🔄 还原棋盘逻辑状态，撤销 {steps} 步")

        # 重新初始化棋盘
        self.parent.board = cchess.Board()

        # 重新应用未被撤销的移动
        moves_to_keep = self.parent.move_history
        for move_uci in moves_to_keep:
            try:
                move = cchess.Move.from_uci(move_uci)
                if move in self.parent.board.legal_moves:
                    self.parent.board.push(move)
                    print(f"重新应用移动: {move_uci}")
            except Exception as e:
                print(f"应用移动 {move_uci} 时出错: {e}")

        # 更新棋盘位置状态
        self.parent.previous_positions = self.parent.his_chessboard[self.parent.move_count]
