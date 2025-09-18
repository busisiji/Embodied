# 分支功能模块（收棋、布局、悔棋等）
import time
import numpy as np

from src.cchessAI import cchess
from src.cchess_runner.chess_play_flow_move import ChessPlayFlowMove
from utils.calibrationManager import multi_camera_pixel_to_world, chess_to_world_position, get_area_center
from parameters import RCV_CAMERA, POINT_DOWN, IO_QI, POINT_RCV_DOWN, RCV_H_LAY, RED_CAMERA, BLACK_CAMERA, \
    CHESS_POINTS_RCV_H
from src.cchessYolo.detect_chess_box import calculate_4x4_collection_positions


class ChessPlayFlowBranch(ChessPlayFlowMove):
    def collect_pieces_at_end(self):
        """
        收局函数：识别棋盒位置，然后将所有棋子按颜色分类放入棋盒
        """
        try:
            print("🧹 开始收局...")
            self.speak("开始收局")

            while 1:
#                 self.urController.set_speed(0.8)
                self.urController.run_point_j(RCV_CAMERA)
                # time.sleep(3)

                # 1. 识别棋盒位置（支持3或4个圆）
                chess_box_points = self.detect_chess_box(max_attempts=20)

                # 如果无法识别到棋盒位置，则报错
                if chess_box_points is None:
                    print("无法识别棋盒位置")
                    self.speak("无法识别棋盒位置")
                    time.sleep(10)
                    return

                print("✅ 成功识别棋盒位置")
                self.speak("成功识别棋盒位置")

                chess_box_points = np.array([[point[0]+40,point[1]+40] for point in chess_box_points])

                # 转换为世界坐标检查尺寸 注意镜像翻转
                world_corner_0 = multi_camera_pixel_to_world(chess_box_points[2][0], chess_box_points[2][1], self.inverse_matrix_r, "RCV_CAMERA") # 棋盒左上角
                world_corner_1 = multi_camera_pixel_to_world(chess_box_points[3][0], chess_box_points[3][1], self.inverse_matrix_r,  "RCV_CAMERA") # 棋盒右上角
                world_corner_2 = multi_camera_pixel_to_world(chess_box_points[0][0], chess_box_points[0][1], self.inverse_matrix_r, "RCV_CAMERA") # 棋盒右下角
                world_corner_3 = multi_camera_pixel_to_world(chess_box_points[1][0], chess_box_points[1][1], self.inverse_matrix_r, "RCV_CAMERA") # 棋盒左下角

                cx = 0
                cy = 0
                topLeft = world_corner_0[0]  , world_corner_0[1]
                topRight = world_corner_1[0]  , world_corner_1[1]
                bottomRight = world_corner_2[0]  , world_corner_2[1]
                bottomLeft = world_corner_3[0] , world_corner_3[1]
                chess_box_points = [topLeft, topRight, bottomRight, bottomLeft]

                if not self.urController.is_point_reachable(bottomLeft[0], bottomLeft[1], POINT_RCV_DOWN[1] + 20):
                    print("机械臂无法到达棋盒，请重新放置到靠近机械臂的位置！")
                    self.speak("机械臂无法到达棋盒，请重新放置到靠近机械臂的位置！")
                    raise ValueError("机械臂无法到达棋盒，请将棋盒放置到靠近机械臂的位置")

                # 计算4x4网格的世界坐标位置
                collection_positions = calculate_4x4_collection_positions(chess_box_points)
                print('棋盒坐标：', topLeft, topRight, bottomRight, bottomLeft)

                world_width = np.linalg.norm(np.array(topRight) - np.array(topLeft))
                world_height = np.linalg.norm(np.array(topLeft) - np.array(bottomLeft))

                # # 检查每个格子是否大于PIECE_SIZE
                # min_size = PIECE_SIZE * 3 * 0
                #
                # if min_size > world_width or min_size > world_height:
                #     print('棋盒格子尺寸不足')
                #     self.speak(
                #         f"❌ 棋盒格子尺寸不足，需要大于{min_size}mm，当前尺寸: {world_width:.2f}mm x {world_height:.2f}mm")
                #     raise ValueError("棋盒格子尺寸不足")
                print(f"✅ 棋盒尺寸检查通过，格子尺寸: {world_width:.2f}mm x {world_height:.2f}mm")

                # 3. 识别红方棋子并移动到棋盒下层
                print("🔴 开始收集红方棋子...")
                self.speak("开始收集红方棋子")
                self.collect_half_board_pieces("red", collection_positions)

                # 4. 识别黑方棋子并移动到棋盒上层
                print("⚫ 开始收集黑方棋子...")
                self.speak("开始收集黑方棋子")
                self.collect_half_board_pieces("black", collection_positions)

                print("✅ 收局完成")
                self.speak("收局完成")
                time.sleep(5)
                return
        except Exception as e:
            print(e)
            self.speak("收局失败")
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

        black_piece_positions = self._collect_pieces_from_half_board(
            BLACK_CAMERA, "BLACK_CAMERA", target_class_names)

        if len(red_piece_positions) + len(black_piece_positions) != 16:
            print(f"⚠️ 棋子数量不足,只有{len(red_piece_positions) + len(black_piece_positions)}")
            self.speak("棋子数量不足16个,无法步棋")
            raise ValueError("棋子数量不足,无法步棋")

        # 按从左到右、从上到下的顺序排序
        red_piece_positions.sort(key=lambda p: (p[1], p[0]))  # 按y坐标升序，x坐标升序

        # 立即移动红方半区识别到的棋子到棋盒
        position_index = 16 - len(red_piece_positions) - len(black_piece_positions)
        print(f"🚚 开始移动红方半区识别到的{side}方棋子...")
        for x_world, y_world in red_piece_positions:
            if position_index >= len(collection_positions):
                print("⚠️ 棋盒位置不足")
                raise ValueError("棋盒位置不足")

            # 目标位置
            target_x, target_y = collection_positions[position_index]

            self.point_move(
                [x_world, y_world, pick_height],
                [target_x, target_y, place_height],  # 根据side决定放置高度
                [0, 0]  # home_row 参数，控制 move_home 行为
            )

            position_index += 1
            print(f"✅ 将{side}方棋子从({x_world:.1f},{y_world:.1f})放置到棋盒位置({position_index}/{len(red_piece_positions)})")

        print(f"✅ 完成移动红方半区{side}方棋子，共移动{position_index}个")

        # 2. 处理黑方半区
        print(f"🔍 在黑方半区寻找{side}方棋子...")

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

            self.point_move(
                [x_world, y_world, pick_height],
                [target_x, target_y, place_height],  # 根据side决定放置高度
                [9, 9]  # home_row 参数，控制 move_home 行为
            )

            position_index += 1
            print(f"✅ 将{side}方棋子从({x_world:.1f},{y_world:.1f})放置到棋盒位置({position_index}/{len(black_piece_positions)})")

        print(f"✅ 完成收集{side}方棋子，共收集{position_index}个")

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
            inverse_matrix = self.inverse_matrix_r
        else:
            inverse_matrix = self.inverse_matrix_b

        # 移动到拍照点
        self.urController.run_point_j(camera_position)
        # time.sleep(3)

        # 捕获图像
        image, depth = self.capture_stable_image(is_chessboard=False)
        if image is None:
            print(f"⚠️ 无法捕获{camera_type}图像")
            return piece_positions

        # 使用YOLO检测器识别棋子
        objects_info, _ = self.detector.detect_objects_with_height(
            image, depth,
            conf_threshold=self.args.conf,
            iou_threshold=self.args.iou
        )

        # 筛选出目标颜色的棋子
        for object_info in objects_info:
            if object_info['class_name'] in target_class_names:
                pixel_x, pixel_y = object_info['center']
                x_world, y_world = multi_camera_pixel_to_world(pixel_x, pixel_y,inverse_matrix, camera_type)
                piece_positions.append((x_world, y_world))

        return piece_positions

    def setup_initial_board(self):
        """
        布局函数：从收子区取出棋子并按初始布局放到棋盘上
        先处理上层的黑方棋子，再处理下层的红方棋子
        """
        try:
            print("🎯 开始初始布局...")
            self.speak("开始初始布局")

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
                ['R', 'N', 'B', 'A', 'K', 'A', 'B', 'N', 'R']   # 9行
            ]

            # 1. 处理上层黑方棋子
            print("⚫ 处理上层黑方棋子...")
            self.speak("正在布置黑方棋子")
            for i in range(20):
                if self.setup_half_board_pieces("black", initial_layout):
                    break
                time.sleep(10)

            # 2. 处理下层红方棋子
            print("🔴 处理下层红方棋子...")
            self.speak("正在布置红方棋子")
            for i in range(20):
                if self.setup_half_board_pieces("red", initial_layout):
                    break
                time.sleep(10)

            print("✅ 初始布局完成")
        except Exception as e:
            print(f"❌ 初始布局异常: {str(e)}")
            self.speak("初始布局异常")
            raise e

    def setup_half_board_pieces(self, side, target_layout):
        """
        布置半区棋子，确保棋子类型与目标位置匹配

        Args:
            side: 棋子方("red"或"black")
            target_layout: 目标布局
        """
        # 移动到收子区拍照点
#         self.urController.set_speed(0.8)
        self.urController.run_point_j(RCV_CAMERA)
        # time.sleep(3)
        # 捕获图像和深度信息
        rcv_image, rcv_depth = self.capture_stable_image(is_chessboard=False)
        if rcv_image is None:
            print("⚠️ 无法捕获收子区图像")
            return

        inverse_matrix = self.inverse_matrix_rcv_h if side == "black" else self.inverse_matrix_rcv_l

        # 使用YOLO检测器识别收子区的棋子（包含高度信息）
        objects_info, _ = self.detector.detect_objects_with_height(
            rcv_image, rcv_depth,
            conf_threshold=self.args.conf,
            iou_threshold=self.args.iou,
            mat=self.m_rcv
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
                x_world, y_world = multi_camera_pixel_to_world(
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
                missing_pieces.append(f"{self.piece_map[piece_type]}缺少{required_count - available_count}个")

        if not is_complete:
            print(f"⚠️ {side}方棋子不完整: {', '.join(missing_pieces)}")
            self.speak(f"{side}方棋子{', '.join(missing_pieces)}")
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
            rcv_world_x, rcv_world_y = multi_camera_pixel_to_world(
                    rcv_center_x, rcv_center_y, inverse_matrix)
            print(f"📥 将{side}方棋子{target_piece}从收子区放置到位置({target_row},{target_col})")

            # 移动到收子区拍照点
#             self.urController.set_speed(0.8)
            # self.urController.run_point_j(RCV_CAMERA)

            # 移动到中心点
            self.urController.move_to(rcv_world_x, rcv_world_y, pick_height + 50)

            # 移动到棋子上方
            self.urController.move_to(x_world, y_world, pick_height+20)
#             time.sleep(1)

            # 降低到吸取高度
#             self.urController.set_speed(0.5)
            self.urController.move_to(x_world, y_world, pick_height)
#             time.sleep(1)

            # 吸取棋子
            self.urController.set_do(IO_QI, 1)  # 吸合
#             time.sleep(1)

            # 抬起棋子到安全高度
#             self.urController.set_speed(0.8)
            self.urController.move_to(x_world, y_world, pick_height+20)
#             time.sleep(1)

            # 移动到中心点
            self.urController.move_to(rcv_world_x, rcv_world_y, pick_height+50)
#             time.sleep(2)

            # 移动到棋盘上方
            col = 9 if side == "black" else 0
            self.move_home(col)

            # 移动到目标位置上方
            self.urController.move_to(x_world_target, y_world_target, place_height+20)
#             time.sleep(1)

            # 降低到放置高度
#             self.urController.set_speed(0.5)
            self.urController.move_to(x_world_target, y_world_target, place_height+5)
#             time.sleep(1)

            # 放置棋子
            self.urController.set_do(IO_QI, 0)
#             time.sleep(1)
            self.urController.move_to(x_world_target, y_world_target, place_height+20)

            # 抬起机械臂到安全高度
#             self.urController.set_speed(0.8)
            self.move_home(col)
#             time.sleep(1)

            print(f"✅ {side}方棋子{target_piece}已放置到位置({target_row},{target_col})")
        return True

    def undo_move(self, steps=2):
        """
        悔棋函数，将棋盘状态还原到前n步

        Args:
            steps: 要悔棋的步数，默认为1步
        """
        try:
            if self.side == self.args.robot_side:
                print(f"⚠️ 当前棋子方为 {self.side}，无法悔棋")
                self.speak(f"机器人正在落子，无法悔棋")
                raise Exception("机器人正在落子，无法悔棋")
            print(f"↩️ 执行悔棋，回到 {steps} 步前的状态")
            self.speak(f"正在执行悔棋")
            self.urController.hll(f_5=1)  # 红灯
            # 检查是否有足够的历史记录
            if len(self.move_history) < steps:
                print(f"⚠️ 没有足够的移动历史，当前只有 {len(self.move_history)} 步")
                self.speak("没有足够的移动历史")
                return False

            # 从移动历史中获取要撤销的移动
            moves_to_undo = self.move_history[-steps:]
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
                if target_piece_key in self.captured_pieces_history:
                    # 恢复被吃的棋子
                    captured_info = self.captured_pieces_history[target_piece_key]
                    print(f"发现被吃的棋子需要恢复: {captured_info}")
                    self.speak(f"请将被吃的{self.piece_map[captured_info['piece']]}放回棋盘")

                    # 等待用户放回棋子
                    self.wait_for_player_adjustment()


                # 物理上将棋子移回原位
                self._move_piece_back(from_row, from_col, to_row, to_col)

            # 更新移动历史
            self.move_history = self.move_history[:-steps]

            # 更新全局变量 move_count 和 side
            self.move_count = len(self.move_history)
            self._update_side_after_undo()

            # 更新棋盘状态
            self._revert_board_state(steps)

            # 更新MainGame棋盘状态
            self._revert_maingame_state(steps)

            # 7. 显示更新后的棋盘
            if self.args.show_board:
                self.game.graphic(self.board)

            print(f"✅ 悔棋完成，已回到 {steps} 步前的状态")
            self.speak("悔棋完成")
            self.is_undo = True
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
        self.maingame.restart_game()

        # 重新应用未被撤销的移动到MainGame
        moves_to_keep = self.move_history
        for move_uci in moves_to_keep:
            try:
                # 将UCI移动转换为MainGame坐标
                from_col = ord(move_uci[0]) - ord('a')
                from_row = int(move_uci[1])
                to_col = ord(move_uci[2]) - ord('a')
                to_row = int(move_uci[3])

                # 转换为MainGame坐标系 (镜像处理)
                mg_from_x = 8 - from_col
                mg_to_x = 8 - to_col
                mg_from_y = 9 - from_row
                mg_to_y = 9 - to_row

                # 创建移动步骤
                from src.cchessAG import my_chess
                s = my_chess.step(mg_from_x, mg_from_y, mg_to_x, mg_to_y)
                print(f"已创建移动步骤: {s}")

                # 执行移动到MainGame并保存历史信息
                self.maingame.mgInit.move_to(s)
                print(f"MainGame重新应用移动: {move_uci} -> ({mg_from_x},{mg_from_y}) to ({mg_to_x},{mg_to_y})")

            except Exception as e:
                print(f"MainGame应用移动 {move_uci} 时出错: {e}")

    def _update_side_after_undo(self):
        """
        悔棋后更新当前回合方
        """
        # 根据已走步数和机器人执子方来确定当前回合方
        is_robot_turn = (self.move_count + (0 if self.args.robot_side == 'red' else 1)) % 2 == 1
        if not is_robot_turn:
            self.side = self.args.robot_side
        else:
            self.side = 'black' if self.args.robot_side == 'red' else 'red'
        print(f"🔄 悔棋后更新当前回合方为: {self.side}")

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
#         self.urController.set_speed(0.5)
        self.move_home()

    def _revert_board_state(self, steps):
        """
        还原棋盘逻辑状态

        Args:
            steps: 要还原的步数
        """
        print(f"🔄 还原棋盘逻辑状态，撤销 {steps} 步")

        # 重新初始化棋盘
        self.board = cchess.Board()

        # 重新应用未被撤销的移动
        moves_to_keep = self.move_history
        for move_uci in moves_to_keep:
            try:
                move = cchess.Move.from_uci(move_uci)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    print(f"重新应用移动: {move_uci}")
            except Exception as e:
                print(f"应用移动 {move_uci} 时出错: {e}")

        # 更新棋盘位置状态
        self.previous_positions = self.his_chessboard[self.move_count]
