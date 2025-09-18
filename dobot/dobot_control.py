# file: /media/jetson/KESU/code/Embodied/api/dobot/dobot_control.py

import threading
import time
import re
from concurrent.futures import ThreadPoolExecutor
from time import sleep

import numpy as np
from dobot.dobot_api import DobotApiDashboard, DobotApiMove, DobotApi, MyType
from api.utils.websocket_utils import send_error_notification_sync
from parameters import RED_CAMERA, POINT_HOME, POINT_TRANSIT, POINT_DOWN, POINT_UP, SAC_CAMERA, FOUR_WORLD_SAC, \
    POINT_SAC_DOWN, IO_QI, RCV_CAMERA, BLACK_CAMERA
from utils.calibrationManager import pixel_to_world


def _is_empty_error_list(error_info):
    """
    检查错误信息是否表示空错误列表
    @param error_info: 错误信息字符串
    @return: 如果是空错误列表返回True，否则返回False
    """
    # 检查是否为 [[]] 或类似格式
    if "[]" in error_info and not any(c.isdigit() for c in error_info.replace("[]", "")):
        return True
    # 检查是否只包含空数组
    cleaned = error_info.replace(" ", "").replace("[", "").replace("]", "").replace(",", "")
    return not cleaned or cleaned == ""


class URController:
    def __init__(self, ip="192.168.5.1", port=30003, dashboard_port=29999, feed_port=30006,
                 acceleration=0.5, velocity=0.3, tool_coordinates=(0, 0, 0.1)):
        """
        初始化UR机械臂控制器

        @param ip: 机械臂IP地址
        @param port: 移动控制端口
        @param dashboard_port: 控制面板端口
        @param feed_port: 反馈端口
        @param acceleration: 运动加速度 (0-1)
        @param velocity: 运动速度 (0-1)
        @param tool_coordinates: 工具坐标系 (x, y, z)
        """
        self.ip = ip
        self.port = port
        self.dashboard_port = dashboard_port
        self.feed_port = feed_port
        self.dashboard = None
        self.move = None
        self.feed = None
        self.current_actual = None  # 当前坐标
        self.is_wait = True

        self.tool_coordinates = tool_coordinates
        self.safety_zone = (5, 5, 0.05)  # 安全区域范围

        # Dobot机械臂相关参数
        self.point_o = POINT_HOME  # 初始点
        self.point_t = POINT_TRANSIT  # 中转点
        self.up_point = POINT_UP  # 上点位
        self.down_point = POINT_DOWN  # 下点位 吸 放
        self.io_status = [0, 0, 0, 0]  # IO状态

        # 添加报警相关属性
        self.alarm_thread = None
        self.alarm_monitoring = False
        self.current_error_status = None  # 当前错误状态

        # 添加限高功能相关属性
        self.height_limit_enabled = False  # 是否启用限高功能
        self.min_height = 0.0  # 最低移动高度

        # 启动连接
        self.connect()

        # 初始化线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
    def _execute_command(self, connection, func, *args, description="", **kwargs):
        """统一执行命令的方法"""
        if not connection:
            print(f"⚠️  {description}失败: 连接未建立")
            return None
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            print(f"❌ {description}失败: {str(e)}")
            return None

    def _execute_with_dashboard(self, func, *args, description="", **kwargs):
        """执行需要dashboard连接的操作"""
        return self._execute_command(self.dashboard, func, *args, description=description, **kwargs)

    def _execute_with_move(self, func, *args, description="", **kwargs):
        """执行需要move连接的操作"""
        result = self._execute_command(self.move, func, *args, description=description, **kwargs)
        return result is not None

    def _start_alarm_monitoring(self):
        """启动报警监控线程"""
        self.alarm_monitoring = True
        self.alarm_thread = threading.Thread(target=self._monitor_alarm)
        self.alarm_thread.daemon = True
        self.alarm_thread.start()
        print("🔔 报警监控已启动")

    def _monitor_alarm(self):
        """监控报警状态的线程函数"""
        previous_error_status = None
        num = 0

        while self.alarm_monitoring:
            try:
                if self.dashboard and self.is_connected():
                    # 获取错误ID
                    error_response = self.dashboard.GetErrorID()
                    if error_response and "GetErrorID" in error_response:
                        # 解析错误信息
                        error_info = error_response.split("{")[1].split("}")[0] if "{" in error_response else ""

                        # 检查是否为空错误列表
                        if error_info and error_info != "0" and not _is_empty_error_list(error_info):
                            self.current_error_status = error_info
                            # 检查错误状态是否发生变化
                            if self.current_error_status != previous_error_status:
                                self._handle_alarm_detected(error_info)
                                previous_error_status = self.current_error_status

                                if self.clear_alarm():
                                    print("✅ 报警清除成功")
                                else:
                                    print("❌ 报警清除失败")
                        else:
                            # 如果之前有错误而现在没有了，发送恢复通知
                            if previous_error_status is not None and previous_error_status != "0":
                                self._send_alarm_notification("机械臂报警已清除")
                            self.current_error_status = None
                            previous_error_status = None
                time.sleep(1)  # 每秒检查一次
            except Exception as e:
                num += 1
                print(f"⚠️ 机械臂报警监控异常: {str(e)}")
                time.sleep(1)
                # 不要在监控线程中直接调用disconnect，这会导致线程自等待
                # 只在错误次数过多时设置标志位，让监控线程自然退出
                if num > 10:
                    print("⚠️ 报警监控错误次数过多，将停止监控")
                    self.alarm_monitoring = False


    def _handle_alarm_detected(self, error_info):
        """处理检测到的报警"""
        print(f"🚨 报警: 错误代码 {error_info}")

        # 解析具体的错误代码
        error_codes = self._parse_error_codes(error_info)
        for code in error_codes:
            error_msg = self._get_error_message(code)
            print(f"📝 错误详情: {error_msg}")

        # 通过WebSocket发送报警信息
        self._send_alarm_notification(f"机械臂报警: 错误代码 {error_info}")

    def _send_alarm_notification(self, message):
        """发送报警通知"""
        try:
            send_error_notification_sync(
                process_type="dobot_alarm",
                error_message=message,
                process_id="dobot_controller",
            )
        except Exception as ws_error:
            print(f"⚠️ WebSocket通知发送失败: {str(ws_error)}")

    def _parse_error_codes(self, error_info):
        """
        解析错误代码
        @param error_info: 错误信息字符串
        @return: 错误代码列表
        """
        try:
            # 查找所有数字
            numbers = re.findall(r'\d+', error_info)
            return [int(n) for n in numbers]
        except:
            return []

    def _get_error_message(self, error_code):
        """
        获取错误代码的详细信息
        @param error_code: 错误代码
        @return: 错误描述
        """
        error_messages = {
            18: "关节超限 - 机械臂关节移动超出允许范围，请检查目标位置是否合理或执行回家操作",
            19: "奇异点 - 机械臂处于奇异位置，无法继续运动",
            20: "速度超限 - 运动速度超出限制",
            21: "加速度超限 - 运动加速度超出限制",
            22: "急停按钮被按下",
            23: "碰撞检测触发",
            24: "电机过热",
            25: "驱动器故障",
            26: "编码器故障",
            27: "通信超时",
            28: "位置超差",
            29: "力矩超限",
            30: "电源异常"
        }

        return error_messages.get(error_code, f"未知错误代码: {error_code}")

    def clear_alarm(self):
        """清除报警"""
        response = self._execute_with_dashboard(
            self.dashboard.ClearError,
            description="清除报警"
        )

        if response:
            if "OK" in response:
                print("✅ 报警已清除")
                self.current_error_status = None
                return True
            else:
                print(f"❌ 清除报警失败: {response}")
                return False
        return False

    def is_alarm_active(self):
        """检查是否有活动报警"""
        return self.current_error_status is not None and self.current_error_status != "0"

    def is_point_reachable(self, x, y, z, rx=None, ry=None, rz=None):
        """
        检查给定点是否可以到达

        @param x: X坐标
        @param y: Y坐标
        @param z: Z坐标
        @param rx: Rx轴角度（可选）
        @param ry: Ry轴角度（可选）
        @param rz: Rz轴角度（可选）
        @return: (bool, str) 是否可到达及原因说明
        """
        try:
            # # 1. 检查连接状态
            # if not self.connected:
            #     return False, "机械臂未连接"
            #
            # # 2. 检查报警状态
            # if self.is_alarm_active():
            #     return False, "机械臂处于报警状态"
            #
            # # 3. 检查高度限制
            # if self.height_limit_enabled and z < self.min_height:
            #     return False, f"目标高度 {z} 低于最小限制高度 {self.min_height}"
            #
            # # 4. 检查安全区域
            # if not self._in_safety_zone(x, y, z):
            #     return False, f"目标位置 ({x:.3f}, {y:.3f}, {z:.3f}) 超出安全区域"

            # 5. 检查工作范围（基于Dobot常见的工作范围）
            # 这些值可以根据具体的机械臂型号进行调整
            max_radius = 600  # 最大工作半径(mm)
            min_radius = 50  # 最小工作半径(mm)
            max_height = 400  # 最大工作高度(mm)
            min_height_limit = -100  # 最小工作高度(mm)

            # 计算到原点的水平距离
            horizontal_distance = np.sqrt(x ** 2 + y ** 2)

            if horizontal_distance > max_radius:
                return False, f"目标点超出最大工作半径 {max_radius}mm"

            if horizontal_distance < min_radius and z < 100:
                return False, f"目标点在最小工作半径 {min_radius}mm 内且高度过低"

            if z > max_height:
                return False, f"目标高度 {z} 超出最大工作高度 {max_height}mm"

            if z < min_height_limit:
                return False, f"目标高度 {z} 低于最小工作高度 {min_height_limit}mm"

            # 6. 检查奇异点区域（简化检查）
            # 接近Z轴时可能存在奇异点
            if horizontal_distance < 20 and z < 50:
                return False, "目标点接近奇异点区域"

            # 7. 检查角度限制（如果提供了角度参数）
            if rx is not None and ry is not None and rz is not None:
                # 检查角度是否在合理范围内（-180到180度）
                angles = [rx, ry, rz]
                for i, angle in enumerate(angles):
                    axis_name = ['Rx', 'Ry', 'Rz'][i]
                    if not -180 <= angle <= 180:
                        return False, f"{axis_name}轴角度 {angle} 超出范围 [-180, 180]"

            # 如果所有检查都通过
            return True

        except Exception as e:
            return False
    def is_connected(self, check_count=3, check_interval=0.1):
        """
        多次检查机械臂连接状态，提高准确性
        @param check_count: 检查次数
        @param check_interval: 检查间隔时间(秒)
        @return: bool 连接状态
        """
        # 进行多次检查以确保连接稳定性
        for i in range(check_count):
            try:
                # 检查所有必需的连接对象是否存在且未关闭
                dashboard_connected = (self.dashboard is not None and
                                     hasattr(self.dashboard, 'socket_dobot') and
                                     self.dashboard.socket_dobot is not None)

                move_connected = (self.move is not None and
                                 hasattr(self.move, 'socket_dobot') and
                                 self.move.socket_dobot is not None)

                feed_connected = (self.feed is not None and
                                 hasattr(self.feed, 'socket_dobot') and
                                 self.feed.socket_dobot is not None)

                # 检查套接字连接状态
                # if dashboard_connected:
                #     dashboard_connected = self._is_socket_alive(self.dashboard.socket_dobot)
                #
                # if move_connected:
                #     move_connected = self._is_socket_alive(self.move.socket_dobot)

                if feed_connected:
                    feed_connected = self._is_socket_alive(self.feed.socket_dobot)

                # 所有连接都必须正常
                all_connected = feed_connected

                # 如果任何一次检查失败，立即返回False
                if not all_connected:
                    if i < check_count - 1:  # 不是最后一次检查，等待后重试
                        time.sleep(check_interval)
                        continue
                    else:  # 最后一次检查仍失败
                        return False
                else:
                    # 连接正常，如果不是最后一次检查，继续确认
                    if i < check_count - 1:
                        time.sleep(check_interval)
                        continue
                    else:  # 所有检查都通过
                        return True

            except Exception as e:
                print(f"⚠️ 第{i+1}次连接状态检查异常: {str(e)}")
                if i < check_count - 1:  # 不是最后一次检查，等待后重试
                    time.sleep(check_interval)
                    continue
                else:  # 最后一次检查仍异常
                    return False

        return True  # 所有检查都通过


    def _is_socket_alive(self, sock):
        """
        检查套接字是否仍然存活
        @param sock: socket对象
        @return: bool 套接字是否存活
        """
        if sock is None:
            return False

        try:
            # 使用非阻塞方式检查套接字状态
            import socket
            sock.settimeout(0.1)
            data = sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
            return True
        except BlockingIOError:
            # 没有数据可读，但连接仍然存在
            return True
        except (ConnectionResetError, ConnectionAbortedError, socket.timeout):
            # 连接已断开或超时
            return False
        except Exception:
            # 其他异常，默认认为连接有效
            return True

    def get_current_error(self):
        """获取当前错误状态"""
        return self.current_error_status

    def connect(self):
        """连接到Dobot机械臂"""
        try:
            print("🔌 正在建立连接...")
            self.dashboard = DobotApiDashboard(self.ip, self.dashboard_port)
            self.move = DobotApiMove(self.ip, self.port)
            self.feed = DobotApi(self.ip, self.feed_port)

            # 启动反馈线程
            self._start_feed_thread()

            # 启动报警监控线程
            self._start_alarm_monitoring()

            # 上电和使能
            self.power_on()
            self.enable_robot()

            # 设置初始速度和加速度
            self.set_speed(0.5)

            if not self.is_connected():
                raise Exception("连接失败")

            print("✅ 连接成功")

        except Exception as e:
            print(f"❌ 连接失败: {str(e)}")
            raise  e

    def disconnect(self):
        """断开连接"""
        # 停止报警监控
        self.alarm_monitoring = False

        # 只有在不是当前线程且线程存在并活跃时才join
        if (self.alarm_thread and
            self.alarm_thread.is_alive() and
            self.alarm_thread != threading.current_thread()):
            self.alarm_thread.join(timeout=2)  # 等待最多2秒让线程结束

        self.disable_robot()
        if self.dashboard:
            self.dashboard.close()
        if self.move:
            self.move.close()
        if self.feed:
            self.feed.close()
        print("🔌 已断开连接")


    def _start_feed_thread(self):
        """启动反馈线程"""
        self.feed_thread = threading.Thread(target=self._get_feed, args=(self.feed,))
        self.feed_thread.setDaemon(True)
        self.feed_thread.start()

    def _get_feed(self, feed: DobotApi):
        """获取机械臂反馈数据"""
        hasRead = 0
        while True:
            try:
                data = bytes()
                while hasRead < 1440:
                    temp = feed.socket_dobot.recv(1440 - hasRead)
                    if len(temp) > 0:
                        hasRead += len(temp)
                        data += temp
                hasRead = 0

                a = np.frombuffer(data, dtype=MyType)
                if hex((a['test_value'][0])) == '0x123456789abcdef':
                    # 更新当前坐标
                    self.current_actual = a["tool_vector_actual"][0]

                time.sleep(0.001)
            except:
                if self.feed:
                    self.feed.close()
                return

    def wait_arrive(self, point_list, timeout=30):
        """等待机械臂到达目标位置"""
        start_time = time.time()
        last_valid_position_time = time.time()

        while True:
            # 检查是否超时
            if time.time() - start_time > timeout:
                print(f"⚠️ 等待机械臂到达超时 ({timeout}秒)")
                return False

            # 使用get_current_position方法获取当前位置
            current_pos = self.get_current_position()
            if current_pos is not None:
                # 检查位置数据是否有效 (基于返回的坐标值判断)
                x, y, z = current_pos[0], current_pos[1], current_pos[2]
                if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                    if time.time() - last_valid_position_time > 5:  # 5秒内未收到有效位置
                        print("⚠️ 位置数据持续异常")
                        return False
                    time.sleep(0.1)
                    continue
                else:
                    last_valid_position_time = time.time()

                # 检查是否到达目标位置
                is_arrive = True
                for index in range(len(current_pos)):
                    if abs(current_pos[index] - point_list[index]) > 20:
                        is_arrive = False
                        break

                if is_arrive:
                    print("✅ 机械臂已到达目标位置")
                    return True

            time.sleep(0.0001)

    def power_on(self):
        """机械臂上电"""
        result = self._execute_with_dashboard(
            self.dashboard.PowerOn,
            description="上电"
        )
        if result:
            print("🔋 正在上电...")
            time.sleep(5)  # 等待5秒让机械臂启动

    def enable_robot(self):
        """启用机械臂"""
        result = self._execute_with_dashboard(
            self.dashboard.EnableRobot,
            description="使能"
        )
        if result:
            print("⚡ 正在使能...")

    def disable_robot(self):
        """禁用机械臂"""
        result = self._execute_with_dashboard(
            self.dashboard.DisableRobot,
            description="失能"
        )
        if result:
            print("🛑 正在失能...")

    def set_speed(self, speed_factor=0.5):
        """设置运动速度因子"""
        if 0 < speed_factor < 1:
            speed_factor = int(speed_factor * 100)
        result = self._execute_with_dashboard(
            lambda: self.dashboard.SpeedFactor(speed_factor),
            description=f"设置速度因子为 {speed_factor}"
        )
        if result:
            print(f"⚙️ 设置速度因子为 {speed_factor}")

    def set_user_coordinate(self, user_index=0):
        """设置用户坐标系"""
        result = self._execute_with_dashboard(
            lambda: self.dashboard.User(user_index),
            description=f"设置用户坐标系为 {user_index}"
        )
        if result:
            print(f"🔧 设置用户坐标系为 {user_index}")

    def set_tool_coordinate(self, tool_index=0):
        """设置工具坐标系"""
        result = self._execute_with_dashboard(
            lambda: self.dashboard.Tool(tool_index),
            description=f"设置工具坐标系为 {tool_index}"
        )
        if result:
            print(f"🔧 设置工具坐标系为 {tool_index}")

    def get_param(self, param_name):
        """
        获取指定参数的值
        @param param_name: 参数名称
        @return: 参数值
        """
        if hasattr(self, param_name):
            return getattr(self, param_name)
        else:
            raise ValueError(f"参数 '{param_name}' 不存在")

    def set_param(self, param_name, value):
        """
        设置指定参数的值
        @param param_name: 参数名称
        @param value: 参数值
        @return: 是否设置成功
        """
        if not hasattr(self, param_name):
            raise ValueError(f"参数 '{param_name}' 不存在")

        # 特殊处理某些参数
        param_handlers = {
            "velocity": lambda v: self.set_velocity(float(v)),
            "acceleration": lambda v: self.set_acceleration(float(v)),
            "speed": lambda v: self.set_speed(int(v))
        }

        if param_name in param_handlers:
            param_handlers[param_name](value)
        elif param_name in ["ip", "port", "dashboard_port", "feed_port"]:
            raise ValueError(f"参数 '{param_name}' 在连接后无法修改")
        else:
            setattr(self, param_name, value)

        return True

    def get_all_params(self):
        """
        获取所有可序列化的参数
        @return: 参数字典
        """
        params = {}
        ignored_attrs = ['dashboard', 'move', 'feed', 'alarm_thread', 'feed_thread']

        for attr_name in dir(self):
            if (not attr_name.startswith('_') and
                attr_name not in ignored_attrs and
                hasattr(self, attr_name)):
                try:
                    attr_value = getattr(self, attr_name)
                    # 只包含基本数据类型
                    if isinstance(attr_value, (int, float, str, bool, list, tuple, dict)) or attr_value is None:
                        params[attr_name] = attr_value
                except:
                    # 忽略无法获取的属性
                    pass

        return params

    def set_height_limit(self, enabled=True, min_height=0.0):
        """
        设置机械臂限高功能
        @param enabled: 是否启用限高功能
        @param min_height: 最低移动高度
        """
        self.height_limit_enabled = enabled
        self.min_height = min_height
        status = "启用" if enabled else "禁用"
        print(f"📏 限高功能已{status}，最低高度: {min_height}mm")

    def _apply_height_limit(self, point_list):
        """
        应用高度限制到目标点
        @param point_list: 目标点坐标 [x, y, z, rx, ry, rz]
        @return: 应用高度限制后的坐标
        """
        if not self.height_limit_enabled:
            return point_list

        # 创建副本以避免修改原始数据
        limited_point = point_list.copy()

        # 如果目标点Z坐标低于最小高度，则将其设置为最小高度
        if limited_point[2] < self.min_height:
            limited_point[2] = self.min_height
            print(f"📏 高度限制已应用，Z坐标从 {point_list[2]} 调整为 {self.min_height}")

        return limited_point

    def _move_to_point(self, point_list: list, move_type: str):
        """通用移动函数"""
        # 应用高度限制
        limited_point = self._apply_height_limit(point_list)

        if move_type == "linear":
            func = self.move.MovL
            move_desc = "线性移动到"
        else:  # joint
            func = self.move.MovJ
            move_desc = "关节运动到"

        # 执行移动
        result = self._execute_with_move(
            lambda: func(limited_point[0], limited_point[1], limited_point[2],
                        limited_point[3], limited_point[4], limited_point[5]),
            description=f"{move_desc} X:{limited_point[0]:.3f}, Y:{limited_point[1]:.3f}, Z:{limited_point[2]:.3f}"
        )

        if result:
            print(f"🕹️ {move_desc} X:{limited_point[0]:.3f}, Y:{limited_point[1]:.3f}, Z:{limited_point[2]:.3f}")

            # 如果需要等待，则使用Sync等待运动完成
            if self.is_wait:
                sync_result = self._execute_with_move(
                    self.move.Sync,
                    description="等待运动完成"
                )
                if sync_result:
                    print("✅ 机械臂运动完成")
                else:
                    print("❌ 等待运动完成时发生错误")

            return result
        return False

    def set_arm_orientation(self, hand="right"):
        """
        设置机械臂手系方向
        @param hand: 手系类型 ("right" 或 "left")
        """
        # 定义左右手系参数
        hand_config = {
            "right": {
                "r": 1,    # 向前
                "d": 1,    # 上肘
                "n": -1,    # 手腕翻转
                "cfg": 1,  # 第六轴角度标识
                "name": "右手系"
            },
            "left": {
                "r": -1,   # 向后
                "d": -1,   # 下肘
                "n": -1,   # 手腕翻转
                "cfg": -1, # 第六轴角度标识
                "name": "左手系"
            }
        }

        # 检查输入参数
        if hand not in hand_config:
            print(f"⚠️ 无效的手系参数: {hand}，使用默认右手系")
            hand = "right"

        config = hand_config[hand]

        result = self._execute_with_dashboard(
            lambda: self.dashboard.SetArmOrientation(
                config["r"],
                config["d"],
                config["n"],
                config["cfg"]
            ),
            description=f"设置{config['name']}"
        )

        if result:
            print(f"🔄 设置机械臂为{config['name']}")
        return result

    def run_point_l(self, point_list: list):
        """运行到指定点(直线运动)"""
        self._move_to_point(point_list, "linear")

    def run_point_j(self, point_list: list):
        """关节运动到指定点"""
        self._move_to_point(point_list, "joint")

    def move_home(self):
        """移动到初始位置"""
        print("🏠 回到初始位置")
        # 先清除报警再回家
        if self.is_alarm_active():
            self.clear_alarm()
            time.sleep(1)
        return self.run_point_j(self.point_o)

    def move_to(self, arm_x, arm_y, arm_z=None, use_safety=False,is_wait=True):
        """
        移动到指定位置

        @param arm_x: X坐标（机器臂坐标系）
        @param arm_y: Y坐标（机器臂坐标系）
        @param arm_z: Z坐标（可选）
        @param use_safety: 是否使用安全区域检查
        @return: 是否成功移动
        """
        try:
            # 检查是否有报警
            if self.is_alarm_active():
                print("⚠️ 检测到报警状态，请先清除报警")
                print("🧹 尝试清除现有报警...")
                if self.clear_alarm():
                    print("✅ 报警清除成功")
                else:
                    print("❌ 报警清除失败")
                    return False
                # return False

            # 如果没有提供Z值，则使用工具坐标系的Z值
            arm_x = round(arm_x,2)
            arm_y = round(arm_y, 2)
            arm_z = round(arm_z, 2) if arm_z is not None else self.point_o[2]

            # 安全区域检查
            if use_safety:
                if not self._in_safety_zone(arm_x, arm_y, arm_z):
                    print(f"⚠️ 目标位置超出安全区域: ({arm_x:.3f}, {arm_y:.3f}, {arm_z:.3f})")
                    return False

            print(f"🕹️ 移动到 X:{arm_x:.3f}, Y:{arm_y:.3f}, Z:{arm_z:.3f}")

            # 创建目标点
            target_point = [arm_x, arm_y, arm_z] + self.point_o[3:]

            # 执行移动
            return self.run_point_j(target_point)

        except Exception as e:
            print(f"❌ 移动失败: {str(e)}")
            return False

    def pick_place(self, x, y, pick_z=0.05, place_z=0.1, use_safety=False):
        """
        抓取和放置操作

        @param x: X坐标（相机坐标系）
        @param y: Y坐标（相机坐标系）
        @param pick_z: 抓取高度
        @param place_z: 放置高度
        @param use_safety: 是否使用安全区域检查
        @return: 是否成功完成操作
        """
        try:
            # 移动到目标上方
            if not self.move_to(x, y, self.point_o[2], use_safety):
                return False

            # 下降抓取
            if not self.move_to(x, y, pick_z, False):
                return False

            print("🖐️ 执行抓取动作")
            self.set_do(12, 1)  # 吸合
            time.sleep(1)

            # 抬起
            if not self.move_to(x, y, self.point_o[2], False):
                return False

            # 移动到放置位置
            if not self.move_to(self.point_o[0], self.point_o[1], use_safety=use_safety):
                return False

            # 下降放置
            if not self.move_to(self.point_o[0], self.point_o[1], place_z, False):
                return False

            print("🖐️ 执行放置动作")
            self.set_do(12, 0)  # 释放

            # 返回初始位置
            return self.move_home()

        except Exception as e:
            print(f"❌ 操作失败: {str(e)}")
            return False

    def set_tool_coordinates(self, x, y, z):
        """设置工具坐标系"""
        self.tool_coordinates = (x, y, z)
        print(f"🔧 工具坐标系已设置为: ({x:.3f}, {y:.3f}, {z:.3f})")

    def set_safety_zone(self, x_range, y_range, z_range):
        """设置安全区域范围"""
        self.safety_zone = (x_range, y_range, z_range)
        print(f"🔒 安全区域已设置为: ±({x_range:.3f}, {y_range:.3f}, {z_range:.3f})")

    def _in_safety_zone(self, x, y, z):
        """检查位置是否在安全区域内"""
        return (abs(x) <= self.safety_zone[0] and
                abs(y) <= self.safety_zone[1] and
                abs(z - self.point_o[2]) <= self.safety_zone[2])

    def get_current_position(self):
        """
        获取当前机械臂位置

        @return: 当前位置坐标 (x, y, z) 或 None（如果获取失败）
        """
        try:
            if self.current_actual is not None:
                return (self.current_actual[0], self.current_actual[1], self.current_actual[2],
                        self.current_actual[3], self.current_actual[4], self.current_actual[5])
            return None
        except Exception as e:
            print(f"❌ 获取位置失败: {str(e)}")
            return None
    def set_do(self, io_index, value):
        """设置数字输出（同步版本，保持向后兼容）"""
        result = self._execute_with_dashboard(
            lambda: self.dashboard.DO(io_index, value),
            description=f"设置DO[{io_index}] = {value}"
        )
        if result:
            print(f"🔌 设置DO[{io_index}] = {value}")
            return True
        return False
    # def set_do(self, io_index, value, callback=None):
    #     """
    #     异步设置数字输出
    #
    #     @param io_index: IO索引
    #     @param value: 设置值 (0或1)
    #     @param callback: 回调函数 (io_index, value, success) -> None
    #     @return: Future对象
    #     """
    #
    #     def _set_do_task():
    #         try:
    #             result = self._execute_with_dashboard(
    #                 lambda: self.dashboard.DO(io_index, value),
    #                 description=f"设置DO[{io_index}] = {value}"
    #             )
    #             success = result is not None
    #             if success:
    #                 print(f"🔌 设置DO[{io_index}] = {value}")
    #
    #             # 执行回调
    #             if callback:
    #                 try:
    #                     callback(io_index, value, success)
    #                 except Exception as e:
    #                     print(f"⚠️ 回调函数执行失败: {e}")
    #
    #             return success
    #         except Exception as e:
    #             print(f"❌ 异步设置DO[{io_index}]失败: {str(e)}")
    #             if callback:
    #                 try:
    #                     callback(io_index, value, False)
    #                 except Exception as cb_e:
    #                     print(f"⚠️ 回调函数执行失败: {cb_e}")
    #             return False
    #
    #     return self.executor.submit(_set_do_task)


    def get_di(self, io_index,is_log=True):
        """获取数字输入"""
        result = self._execute_with_dashboard(
            lambda: self.dashboard.DI(io_index,is_log),
            description=f"获取DI[{io_index}]"
        )
        if result:
            try:
                # 检查返回结果是否为有效的数字格式
                if "{" in result and "}" in result:
                    value_str = result.split('{')[1].split('}')[0]
                    # 检查是否为空或无效值
                    if value_str.strip() and not ("[]" in value_str or "[" in value_str):
                        value = int(value_str)
                        # print(f"🔌 获取DI[{io_index}] = {value}")
                        return value
                # 如果是直接返回数字的格式
                elif result.strip().isdigit():
                    value = int(result.strip())
                    # print(f"🔌 获取DI[{io_index}] = {value}")
                    return value

                # print(f"⚠️ DI[{io_index}]返回无效值: {result}，默认返回0")
                return 0  # 返回默认值0而不是False，以兼容int()转换
            except (IndexError, AttributeError, ValueError) as e:
                # print(f"⚠️ 解析DI[{io_index}]结果失败: {result}, 错误: {str(e)}，默认返回0")
                return 0  # 返回默认值0而不是False，以兼容int()转换

        # print(f"⚠️ DI[{io_index}]无返回结果，默认返回0")
        return 0  # 返回默认值0而不是False，以兼容int()转换

    def start_jog(self, axis_id, coord_type=0, user=0, tool=0):
        """
        开始点动运动
        @param axis_id: 运动轴ID
        @param coord_type: 坐标系类型 (0: 关节坐标系, 1: 用户坐标系, 2: 工具坐标系)
        @param user: 用户坐标系索引
        @param tool: 工具坐标系索引
        @return: 是否成功发送指令
        """
        result = self._execute_with_move(
            lambda: self.move.MoveJog(axis_id, coord_type, user, tool),
            description="开始点动运动"
        )
        return result

    def stop_jog(self):
        """
        停止点动运动
        @return: 是否成功发送指令
        """
        result = self._execute_with_dashboard(
            self.dashboard.StopScript,
            description="停止点动运动"
        )
        return result is not None

    def close_all(self):
        """关闭所有连接和资源"""
        self.disable_robot()
        if self.dashboard:
            self.dashboard.close()
        if self.move:
            self.move.close()
        if self.feed:
            self.feed.close()
        print("🔌 所有连接已关闭")

    def reset_position_data(self):
        """重置位置数据"""
        print("🔄 重置位置数据...")
        self.current_actual = None
        # 可以考虑重新初始化机械臂或执行回家操作
        self.move_home()

    def is_position_valid(self):
        """检查当前位置数据是否有效"""
        if self.current_actual is None:
            return False

        x, y, z = self.current_actual[0], self.current_actual[1], self.current_actual[2]
        # 检查是否为合理数值（不是无穷大或NaN）
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            return False

        # 检查是否在工作范围内
        if abs(x) > 1000 or abs(y) > 1000 or z < -100 or z > 500:
            return False

        return True

    def handle_joint_limit_error(self):
        """
        处理关节超限错误（错误代码18）
        """
        print("🔧 处理关节超限错误...")

        # 1. 清除报警
        if not self.clear_alarm():
            print("❌ 无法清除报警")
            return False

        time.sleep(1)

        # 2. 执行回家操作
        print("🏠 执行回家操作以重置关节位置...")
        self.move_home()

        # 3. 等待一段时间确保回家完成
        time.sleep(3)

        # 4. 再次检查是否有报警
        if self.is_alarm_active():
            print("❌ 回家操作后仍有报警")
            return False

        print("✅ 关节超限错误已处理")
        return True

    # 所有移动函数放在类的最下面
    def MovJ(self, x, y, z, rx, ry, rz):
        """
        关节运动接口 (点到点运动模式)
        @param x: 笛卡尔坐标系中的x坐标
        @param y: 笛卡尔坐标系中的y坐标
        @param z: 笛卡尔坐标系中的z坐标
        @param rx: Rx轴位置
        @param ry: Ry轴位置
        @param rz: Rz轴位置
        """
        point = [x, y, z, rx, ry, rz]
        return self.run_point_j(point)

    def MovL(self, x, y, z, rx, ry, rz):
        """
        坐标系运动接口 (直线运动模式)
        @param x: 笛卡尔坐标系中的x坐标
        @param y: 笛卡尔坐标系中的y坐标
        @param z: 笛卡尔坐标系中的z坐标
        @param rx: Rx轴位置
        @param ry: Ry轴位置
        @param rz: Rz轴位置
        """
        point = [x, y, z, rx, ry, rz]
        return self.run_point_l(point)

    def JointMovJ(self, j1, j2, j3, j4, j5, j6):
        """
        关节运动接口 (线性运动模式)
        @param j1~j6: 各关节上的点位置值
        """
        # 注意：此函数需要将关节坐标转换为笛卡尔坐标
        # 这里直接调用底层API，绕过高度限制
        result = self._execute_with_move(
            lambda: self.move.JointMovJ(j1, j2, j3, j4, j5, j6),
            description=f"关节运动到 J1:{j1:.3f}, J2:{j2:.3f}, J3:{j3:.3f}, J4:{j4:.3f}, J5:{j5:.3f}, J6:{j6:.3f}"
        )
        return result

    def RelMovJ(self, offset1, offset2, offset3, offset4, offset5, offset6):
        """
        偏移运动接口 (点到点运动模式)
        @param offset1~offset6: 各关节上的偏移位置值
        """
        result = self._execute_with_move(
            lambda: self.move.RelMovJ(offset1, offset2, offset3, offset4, offset5, offset6),
            description=f"相对关节运动 Offset1:{offset1:.3f} ... Offset6:{offset6:.3f}"
        )
        return result

    def RelMovL(self, offsetX, offsetY, offsetZ):
        """
        偏移运动接口 (直线运动模式)
        @param offsetX: X轴偏移量
        @param offsetY: Y轴偏移量
        @param offsetZ: Z轴偏移量
        """
        result = self._execute_with_move(
            lambda: self.move.RelMovL(offsetX, offsetY, offsetZ),
            description=f"相对直线运动 OffsetX:{offsetX:.3f}, OffsetY:{offsetY:.3f}, OffsetZ:{offsetZ:.3f}"
        )
        return result

    def MovLIO(self, x, y, z, a, b, c, *dynParams):
        """
        在直线运动的同时并行设置数字输出端口状态
        @param x: 笛卡尔坐标系中的x坐标
        @param y: 笛卡尔坐标系中的y坐标
        @param z: 笛卡尔坐标系中的z坐标
        @param a: 笛卡尔坐标系中的a坐标
        @param b: 笛卡尔坐标系中的b坐标
        @param c: 笛卡尔坐标系中的c坐标
        @param dynParams: 参数设置（Mode、Distance、Index、Status）
        """
        # 应用高度限制
        limited_point = self._apply_height_limit([x, y, z, a, b, c])

        result = self._execute_with_move(
            lambda: self.move.MovLIO(limited_point[0], limited_point[1], limited_point[2],
                                   limited_point[3], limited_point[4], limited_point[5], *dynParams),
            description=f"直线运动并设置IO X:{limited_point[0]:.3f}, Y:{limited_point[1]:.3f}, Z:{limited_point[2]:.3f}"
        )
        return result

    def MovJIO(self, x, y, z, a, b, c, *dynParams):
        """
        在点到点运动的同时并行设置数字输出端口状态
        @param x: 笛卡尔坐标系中的x坐标
        @param y: 笛卡尔坐标系中的y坐标
        @param z: 笛卡尔坐标系中的z坐标
        @param a: 笛卡尔坐标系中的a坐标
        @param b: 笛卡尔坐标系中的b坐标
        @param c: 笛卡尔坐标系中的c坐标
        @param dynParams: 参数设置（Mode、Distance、Index、Status）
        """
        # 应用高度限制
        limited_point = self._apply_height_limit([x, y, z, a, b, c])

        result = self._execute_with_move(
            lambda: self.move.MovJIO(limited_point[0], limited_point[1], limited_point[2],
                                   limited_point[3], limited_point[4], limited_point[5], *dynParams),
            description=f"点到点运动并设置IO X:{limited_point[0]:.3f}, Y:{limited_point[1]:.3f}, Z:{limited_point[2]:.3f}"
        )
        return result

    def Arc(self, x1, y1, z1, a1, b1, c1, x2, y2, z2, a2, b2, c2):
        """
        圆弧运动指令
        @param x1, y1, z1, a1, b1, c1: 中间点坐标值
        @param x2, y2, z2, a2, b2, c2: 终点坐标值
        """
        # 应用高度限制到终点
        limited_point2 = self._apply_height_limit([x2, y2, z2, a2, b2, c2])

        result = self._execute_with_move(
            lambda: self.move.Arc(x1, y1, z1, a1, b1, c1,
                                limited_point2[0], limited_point2[1], limited_point2[2],
                                limited_point2[3], limited_point2[4], limited_point2[5]),
            description=f"圆弧运动到 X:{limited_point2[0]:.3f}, Y:{limited_point2[1]:.3f}, Z:{limited_point2[2]:.3f}"
        )
        return result

    def Circle(self, count, x1, y1, z1, a1, b1, c1, x2, y2, z2, a2, b2, c2):
        """
        整圆运动指令
        @param count: 运行圈数
        @param x1, y1, z1, a1, b1, c1: 中间点坐标值
        @param x2, y2, z2, a2, b2, c2: 终点坐标值
        """
        # 应用高度限制到终点
        limited_point2 = self._apply_height_limit([x2, y2, z2, a2, b2, c2])

        result = self._execute_with_move(
            lambda: self.move.Circle(count, x1, y1, z1, a1, b1, c1,
                                   limited_point2[0], limited_point2[1], limited_point2[2],
                                   limited_point2[3], limited_point2[4], limited_point2[5]),
            description=f"整圆运动到 X:{limited_point2[0]:.3f}, Y:{limited_point2[1]:.3f}, Z:{limited_point2[2]:.3f}"
        )
        return result

    def ServoJ(self, j1, j2, j3, j4, j5, j6):
        """
        基于关节空间的动态跟随指令
        @param j1~j6: 各关节上的点位置值
        """
        result = self._execute_with_move(
            lambda: self.move.ServoJ(j1, j2, j3, j4, j5, j6),
            description=f"关节动态跟随 J1:{j1:.3f} ... J6:{j6:.3f}"
        )
        return result

    def ServoP(self, x, y, z, a, b, c):
        """
        基于笛卡尔空间的动态跟随指令
        @param x, y, z, a, b, c: 笛卡尔坐标点值
        """
        # 应用高度限制
        limited_point = self._apply_height_limit([x, y, z, a, b, c])

        result = self._execute_with_move(
            lambda: self.move.ServoP(limited_point[0], limited_point[1], limited_point[2],
                                   limited_point[3], limited_point[4], limited_point[5]),
            description=f"笛卡尔动态跟随 X:{limited_point[0]:.3f}, Y:{limited_point[1]:.3f}, Z:{limited_point[2]:.3f}"
        )
        return result

    def MoveJog(self, axis_id, *dynParams):
        """
        关节运动
        @param axis_id: 关节运动轴
        @param dynParams: 参数设置（coord_type, user_index, tool_index）
        """
        result = self._execute_with_move(
            lambda: self.move.MoveJog(axis_id, *dynParams),
            description=f"点动运动 {axis_id}"
        )
        return result

    def StartTrace(self, trace_name):
        """
        轨迹拟合（轨迹文件笛卡尔点）
        @param trace_name: 轨迹文件名（包含后缀）
        """
        result = self._execute_with_move(
            lambda: self.move.StartTrace(trace_name),
            description=f"开始轨迹跟踪 {trace_name}"
        )
        return result

    def StartPath(self, trace_name, const, cart):
        """
        轨迹复现（轨迹文件关节点）
        @param trace_name: 轨迹文件名（包含后缀）
        @param const: 当const=1时，以恒定速度重复，将移除轨迹中的暂停和死区
        @param cart: 当cart=1时，按笛卡尔路径复现
        """
        result = self._execute_with_move(
            lambda: self.move.StartPath(trace_name, const, cart),
            description=f"开始路径跟踪 {trace_name}"
        )
        return result

    def StartFCTrace(self, trace_name):
        """
        带有力控的轨迹拟合（轨迹文件笛卡尔点）
        @param trace_name: 轨迹文件名（包含后缀）
        """
        result = self._execute_with_move(
            lambda: self.move.StartFCTrace(trace_name),
            description=f"开始力控轨迹跟踪 {trace_name}"
        )
        return result

    def Sync(self):
        """
        阻塞程序执行队列指令，所有队列指令执行完毕后返回
        """
        result = self._execute_with_move(
            lambda: self.move.Sync(),
            description="同步执行队列指令"
        )
        return result

    def RelMovJTool(self, offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, tool, *dynParams):
        """
        沿工具坐标系执行相对运动指令，末端运动模式为关节运动
        @param offset_x: X轴方向偏移
        @param offset_y: Y轴方向偏移
        @param offset_z: Z轴方向偏移
        @param offset_rx: Rx轴位置
        @param offset_ry: Ry轴位置
        @param offset_rz: Rz轴位置
        @param tool: 选择的工具坐标系
        @param dynParams: 参数设置（speed_j, acc_j, user）
        """
        result = self._execute_with_move(
            lambda: self.move.RelMovJTool(offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, tool, *dynParams),
            description=f"工具坐标系相对关节运动 OffsetX:{offset_x:.3f}, OffsetY:{offset_y:.3f}, OffsetZ:{offset_z:.3f}"
        )
        return result

    def RelMovLTool(self, offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, tool, *dynParams):
        """
        沿工具坐标系执行相对运动指令，末端运动模式为直线运动
        @param offset_x: X轴方向偏移
        @param offset_y: Y轴方向偏移
        @param offset_z: Z轴方向偏移
        @param offset_rx: Rx轴位置
        @param offset_ry: Ry轴位置
        @param offset_rz: Rz轴位置
        @param tool: 选择的工具坐标系
        @param dynParams: 参数设置（speed_l, acc_l, user）
        """
        result = self._execute_with_move(
            lambda: self.move.RelMovLTool(offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, tool, *dynParams),
            description=f"工具坐标系相对直线运动 OffsetX:{offset_x:.3f}, OffsetY:{offset_y:.3f}, OffsetZ:{offset_z:.3f}"
        )
        return result

    def RelMovJUser(self, offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, user, *dynParams):
        """
        沿用户坐标系执行相对运动指令，末端运动模式为关节运动
        @param offset_x: X轴方向偏移
        @param offset_y: Y轴方向偏移
        @param offset_z: Z轴方向偏移
        @param offset_rx: Rx轴位置
        @param offset_ry: Ry轴位置
        @param offset_rz: Rz轴位置
        @param user: 选择的用户坐标系
        @param dynParams: 参数设置（speed_j, acc_j, tool）
        """
        result = self._execute_with_move(
            lambda: self.move.RelMovJUser(offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, user, *dynParams),
            description=f"用户坐标系相对关节运动 OffsetX:{offset_x:.3f}, OffsetY:{offset_y:.3f}, OffsetZ:{offset_z:.3f}"
        )
        return result

    def RelMovLUser(self, offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, user, *dynParams):
        """
        沿用户坐标系执行相对运动指令，末端运动模式为直线运动
        @param offset_x: X轴方向偏移
        @param offset_y: Y轴方向偏移
        @param offset_z: Z轴方向偏移
        @param offset_rx: Rx轴位置
        @param offset_ry: Ry轴位置
        @param offset_rz: Rz轴位置
        @param user: 选择的用户坐标系
        @param dynParams: 参数设置（speed_l, acc_l, tool）
        """
        result = self._execute_with_move(
            lambda: self.move.RelMovLUser(offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, user, *dynParams),
            description=f"用户坐标系相对直线运动 OffsetX:{offset_x:.3f}, OffsetY:{offset_y:.3f}, OffsetZ:{offset_z:.3f}"
        )
        return result

    def RelJointMovJ(self, offset1, offset2, offset3, offset4, offset5, offset6, *dynParams):
        """
        沿各轴关节坐标系执行相对运动指令，末端运动模式为关节运动
        @param offset1~offset6: 各关节上的偏移位置值
        @param dynParams: 参数设置（speed_j, acc_j, user）
        """
        result = self._execute_with_move(
            lambda: self.move.RelJointMovJ(offset1, offset2, offset3, offset4, offset5, offset6, *dynParams),
            description=f"关节坐标系相对运动 Offset1:{offset1:.3f} ... Offset6:{offset6:.3f}"
        )
        return result

    def hll(self, f_4=0, f_5=0):
        """
        异步设置DO[4]和DO[5]的状态
        @param f_4: DO[4]的值 (0或1)
        @param f_5: DO[5]的值 (0或1)
        @return: Future对象
        """
        def _hll_task():
            try:
                # 设置DO[4]
                result_4 = self._execute_with_dashboard(
                    lambda: self.dashboard.DO(4, f_4),
                    description=f"设置DO[4] = {f_4}"
                )

                # 设置DO[5]
                result_5 = self._execute_with_dashboard(
                    lambda: self.dashboard.DO(5, f_5),
                    description=f"设置DO[5] = {f_5}"
                )

                success = (result_4 is not None and result_5 is not None)
                if success:
                    print(f"🔌 已设置 DO[4]={f_4}, DO[5]={f_5}")

                return success
            except Exception as e:
                print(f"❌ 异步设置DO[4]和DO[5]失败: {str(e)}")
                return False

        return self.executor.submit(_hll_task)


    def get_io_status_range(self, start_index, end_index):
        """
        获取指定范围内的数字输入状态

        @param start_index: 起始IO索引
        @param end_index: 结束IO索引
        @return: 包含IO状态的字典
        """
        io_status = {}
        for i in range(start_index, end_index + 1):
            status = self.get_di(i)
            io_status[i] = status
        return io_status

    def set_io_range_to_zero(self, start_index, end_index,value=0):
        """
        将指定范围内的数字输出IO设置为0

        @param start_index: 起始IO索引
        @param end_index: 结束IO索引
        @return: 设置成功的IO数量
        """
        success_count = 0
        for i in range(start_index, end_index + 1):
            if self.set_do(i, value):
                success_count += 1
        print(f"✅ 成功将IO {start_index}-{end_index}设置为0，共设置{success_count}个IO")
        return success_count
    def __del__(self):
        """析构函数，确保资源被正确释放"""
        self.disconnect()
def alarm_handling_test(controller):
    """
    报警处理测试函数
    @param controller: URController实例
    """
    print("🔧 开始报警处理测试")

    try:
        # 确保控制器已连接
        if not controller.is_connected():
            print("⚠️ 控制器未连接，尝试连接...")
            controller.connect()

        if not controller.is_connected():
            print("❌ 无法连接到控制器，测试终止")
            return False

        print("✅ 控制器连接正常")

        # 1. 检查当前报警状态
        print("\n1. 检查当前报警状态...")
        current_error = controller.get_current_error()
        if current_error and current_error != "0":
            print(f"🚨 检测到现有报警: {current_error}")

            # 解析错误代码
            error_codes = controller._parse_error_codes(current_error)
            for code in error_codes:
                error_msg = controller._get_error_message(code)
                print(f"📝 错误详情: {error_msg}")

                # 特殊处理关节超限错误
                if code == 18:
                    print("🔧 正在处理关节超限错误...")
                    controller.handle_joint_limit_error()

            print("🧹 尝试清除现有报警...")
            if controller.clear_alarm():
                print("✅ 报警清除成功")
            else:
                print("❌ 报警清除失败")
                return False
        else:
            print("✅ 当前无报警")

        # 2. 模拟触发报警情况（如果可能）
        print("\n2. 监控报警状态 (持续10秒)...")
        start_time = time.time()
        alarms_detected = []

        while time.time() - start_time < 10:
            current_error = controller.get_current_error()
            if current_error and current_error != "0" and current_error not in alarms_detected:
                print(f"🚨 检测到新报警: {current_error}")
                alarms_detected.append(current_error)

                # 解析错误代码
                error_codes = controller._parse_error_codes(current_error)
                for code in error_codes:
                    error_msg = controller._get_error_message(code)
                    print(f"📝 错误详情: {error_msg}")

                    # 特殊处理关节超限错误
                    if code == 18:
                        print("🔧 正在处理关节超限错误...")
                        controller.handle_joint_limit_error()

                # 尝试清除报警
                print("🧹 尝试清除报警...")
                if controller.clear_alarm():
                    print("✅ 报警清除成功")
                else:
                    print("❌ 报警清除失败")

            time.sleep(0.5)  # 每0.5秒检查一次

        # 3. 总结测试结果
        print("\n3. 测试总结:")
        if alarms_detected:
            print(f"✅ 测试期间检测到 {len(alarms_detected)} 个报警:")
            for alarm in alarms_detected:
                print(f"   - 报警代码: {alarm}")
        else:
            print("✅ 测试期间未检测到报警，报警监控功能正常")

        print("✅ 报警处理测试完成")
        return True

    except Exception as e:
        print(f"❌ 报警处理测试异常: {str(e)}")
        return False
    finally:
        # 确保最终清除任何报警
        try:
            if controller.is_alarm_active():
                print("🧹 清理残留报警...")
                controller.clear_alarm()
        except:
            pass


def connect_and_check_speed(ip="192.168.5.1", port=30003, dashboard_port=29999, feed_port=30006):
    """
    连接控制器并检查速度设置

    @param ip: 机械臂IP地址
    @param port: 移动控制端口
    @param dashboard_port: 控制面板端口
    @param feed_port: 反馈端口
    @param max_allowed_speed: 允许的最大速度值
    @return: URController实例或None
    """
    try:
        print("🔌 正在连接控制器...")

        # 创建控制器实例
        controller = URController(
            ip=ip,
            port=port,
            dashboard_port=dashboard_port,
            feed_port=feed_port
        )
        # 检查连接状态
        if not controller.is_connected():
            print("❌ 控制器连接失败")
            raise Exception("无法连接到机械臂")

        print("✅ 控制器连接成功")

        return controller

    except Exception as e:
        print(f"❌ 连接和检查过程中发生异常: {str(e)}")
        raise e

def io_test():
    """
    测试循环获取IO20状态并设置IO17,IO18,IO19,IO22状态
    """
    # 创建控制器实例
    controller = URController()

    if not controller.is_connected():
        print("❌ 无法连接到机械臂控制器")
        return

    print("✅ 控制器连接成功，开始IO测试...")

    # IO设置配置
    output_ios = [17, 18, 19, 22]  # 需要设置的输出IO
    input_io = 4  # 需要读取的输入IO

    try:
        # 循环测试10次
        for i in range(10):
            print(f"\n--- 测试循环 {i + 1}/10 ---")

            # 获取IO20的状态
            io20_status = controller.get_di(input_io)
            print(f"📥 IO[{input_io}] 状态: {io20_status}")

            controller.set_do(1, 1)
            controller.set_do(13, 1)
            # # 设置IO17, IO18, IO19, IO22的状态（交替设置0和1）
            # for io_index in output_ios:
            #     # 交替设置状态: 偶数循环设为1，奇数循环设为0
            #     status = 1 if i % 2 == 0 else 0
            #     success = controller.set_do(io_index, status)
            #     if success:
            #         print(f"📤 IO[{io_index}] 设置为: {status}")
            #     else:
            #         print(f"❌ IO[{io_index}] 设置失败")

            # 等待1秒后继续下一次循环
            time.sleep(10)

        print("\n✅ IO测试完成")

    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
    finally:
        # 清理资源
        controller.disconnect()
        print("🔌 控制器连接已断开")

def point_test_sac(urController):
    urController.set_do(IO_QI, 0)  # 吸合
    urController.run_point_j(SAC_CAMERA)
    urController.wait_arrive(SAC_CAMERA)
    urController.run_point_j([FOUR_WORLD_SAC[2][0], FOUR_WORLD_SAC[2][1], POINT_SAC_DOWN[1], -179, 0, -179])
    urController.wait_arrive([FOUR_WORLD_SAC[2][0], FOUR_WORLD_SAC[2][1], POINT_SAC_DOWN[1], -179, 0, -179])
    urController.set_do(IO_QI, 1)  # 吸合
    time.sleep(1)
    urController.run_point_j([FOUR_WORLD_SAC[3][0], FOUR_WORLD_SAC[3][1], POINT_SAC_DOWN[0] + 100, -179, 0, -179])
    urController.wait_arrive([FOUR_WORLD_SAC[3][0], FOUR_WORLD_SAC[3][1], POINT_SAC_DOWN[0] + 100, -179, 0, -179])
    urController.run_point_j([FOUR_WORLD_SAC[3][0], FOUR_WORLD_SAC[3][1], POINT_SAC_DOWN[0], -179, 0, -179])
    urController.wait_arrive([FOUR_WORLD_SAC[3][0], FOUR_WORLD_SAC[3][1], POINT_SAC_DOWN[0], -179, 0, -179])
    urController.set_do(IO_QI, 0)  # 吸合
    urController.run_point_j(SAC_CAMERA)
    urController.wait_arrive(SAC_CAMERA)


if __name__ == "__main__":
    # # 使用新的连接和检查函数
    urController = connect_and_check_speed()
    print(f"📍 当前位置: {urController.get_current_position()}")
    if urController:
        # x,y = pixel_to_world(114,140)
        # print(x,y)
        # urController.run_point_j([-173,-198,195,-179,0.2,-179])
        # time.sleep(5)
        # urController.run_point_j(RED_CAMERA)
        # urController.run_point_j([-28,-379,195,-179,0.2,-179])
        # time.sleep(5)
        # urController.run_point_j(RED_CAMERA)
        # urController.run_point_j([175,-538,195,-179,0.2,-179])
        # time.sleep(5)
        # urController.run_point_j(RED_CAMERA)
        # urController.run_point_j(BLACK_CAMERA)
        # time.sleep(5)
        # urController.run_point_j([125,-343,195,-179,0.2,-179])
        # time.sleep(5)
        urController.set_do(IO_QI, 0)  # 吸合123456

        # urController.run_point_j(RCV_CAMERA)
        # time.sleep(3)
        # urController.wait_arrive(BLACK_CAMERA)
        # urController.move_to(-410.96, -299.49,260)
        # time.sleep(10)
        # urController.move_to(216, -596,250)
        # time.sleep(10)
        # urController.move_to(-350,-440,250)
        # time.sleep(10)
        # urController.move_to(-230,-440,210)
        # time.sleep(5)
        # print(urController.is_point_reachable(-400, -440,319)

        # time.sleep(100)
        # alarm_handling_test(urController)
        # 断开连接
        urController.disconnect()
    else:
        print("❌ 无法建立控制器连接")
