# file: /media/jetson/KESU/code/Embodied/api/dobot/dobot_control.py
import socket
import threading
import time
import re
from concurrent.futures import ThreadPoolExecutor
from time import sleep

import numpy as np
# from dobot.dobot_api import DobotApiDashboard, DobotApiMove, DobotApi, MyType
from dobot.arm.dobot_api import DobotApiDashboard, DobotApi, MyType, DobotApiFeedBack
from api.utils.websocket_utils import send_error_notification_sync
from parameters import RED_CAMERA, POINT_HOME, POINT_TRANSIT, POINT_DOWN, POINT_UP, SAC_CAMERA, FOUR_WORLD_SAC, \
    POINT_SAC_DOWN, IO_QI, RCV_CAMERA, BLACK_CAMERA, FRUIT_CAMERA
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


class URController():
    def __init__(self, ip="192.168.5.1", port=30003, dashboard_port=29999, feed_port=30004):
        """
        初始化UR机械臂控制器

        @param ip: 机械臂IP地址
        @param port: 移动控制端口
        @param dashboard_port: 控制面板端口
        @param feed_port: 反馈端口
        @param acceleration: 运动加速度 (0-1)
        @param velocity: 运动速度 (0-1)
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
        self.is_wait_no = False # 是否不在等待

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

        # 添加暂停/恢复功能相关属性
        self.paused = False  # 是否处于暂停状态
        self.paused_operations = []  # 记录暂停时的操作
        self.pause_event = threading.Event()  # 用于暂停控制

        # 启动连接
        self.connect()

        # 初始化线程池
        self.executor = ThreadPoolExecutor(max_workers=4)

    # 连接
    def connect(self):
        """连接到Dobot机械臂"""
        try:
            print("🔌 正在建立连接...")
            # self.move = DobotApiMove(self.ip, self.port)
            # self.feed = DobotApi(self.ip, self.feed_port)
            # self.dashboard = DobotApiDashboard(self.ip, self.dashboard_port)
            self.dashboard = DobotApiDashboard(self.ip, self.dashboard_port)
            self.feed = DobotApiFeedBack(self.ip, self.feed_port)
            self.move = self.dashboard

            # 验证连接
            if not self.dashboard.socket_dobot:
                raise Exception("Dashboard连接失败")
            if not self.move.socket_dobot:
                raise Exception("Move连接失败")
            if not self.feed.socket_dobot:
                raise Exception("Feed连接失败")

            print(f"✅ Dashboard连接成功 (端口 {self.dashboard_port})")
            print(f"✅ Move连接成功 (端口 {self.port})")
            print(f"✅ Feed连接成功 (端口 {self.feed_port})")

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
            raise e

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

        # 安全地关闭所有连接
        connections = [
            ('Dashboard', self.dashboard),
            ('Move', self.move),
            ('Feed', self.feed)
        ]

        for name, connection in connections:
            if connection and hasattr(connection, 'socket_dobot') and connection.socket_dobot:
                try:
                    connection.close()
                    print(f"🔌 {name}连接已关闭")
                except Exception as e:
                    print(f"⚠️ 关闭{name}连接时出错: {e}")

        # 清空引用
        self.dashboard = None
        self.move = None
        self.feed = None

        print("🔌 所有连接已断开")

    def pause(self):
        """
        暂停机械臂操作
        记录当前操作以便恢复时重新执行
        """
        if self.paused:
            print("⚠️ 机械臂已处于暂停状态")
            return False

        try:
            # while self.paused:
            #     print("⚠️ 机械臂已处于暂停状态")
            #     time.sleep(1)
            # 发送暂停指令
            result = self._execute_with_dashboard(
                self.dashboard.Pause,
                description="暂停机械臂"
            )

            if result:
                self.paused = True
                self.pause_event.clear()  # 设置事件为未触发状态
                print("⏸️ 机械臂已暂停")
                return True
            else:
                print("❌ 暂停指令发送失败")
                return False

        except Exception as e:
            print(f"❌ 暂停过程中发生错误: {str(e)}")
            return False

    def resume(self):
        """
        恢复机械臂操作
        重新执行暂停前的操作
        """
        if not self.paused:
            print("⚠️ 机械臂未处于暂停状态")
            return False

        try:
            # 发送继续指令
            result = self._execute_with_dashboard(
                self.dashboard.Continue,
                description="恢复机械臂"
            )

            if result:
                self.paused = False
                self.pause_event.set()  # 设置事件为触发状态
                print("▶️ 机械臂已恢复")

                self.wait_mvoe()
                return True
            else:
                print("❌ 恢复指令发送失败")
                return False

        except Exception as e:
            print(f"❌ 恢复过程中发生错误: {str(e)}")
            return False
    # 基函数

    def _execute_command(self, func,  description=""):
        """统一执行命令的方法"""
        if not self.move:
            print(f"⚠️  {description}失败: 连接未建立")
            return None
        try:
            result = func()
            return result
        except Exception as e:
            print(f"❌ {description}失败: {str(e)}")
            return None

    def _execute_command_async(self, func, point_list=[], coordinate_mode=0, description=""):
        """异步执行命令的方法，避免阻塞调用线程"""
        if not self.move:
            print(f"⚠️  {description}失败: 连接未建立")
            return None

        # 如果提供了目标点，在执行移动前先检查可达性
        if point_list and len(point_list) >= 6:
            reachable, msg = self.is_point_reachable(
                point_list[0], point_list[1], point_list[2],
                point_list[3], point_list[4], point_list[5],
                coordinate_mode
            )
            if not reachable:
                print(f"⚠️ 目标点不可达: {msg}")
                return None

        import threading
        result_container = [None]
        exception_container = [None]
        event = threading.Event()

        def run_command():
            try:
                result_container[0] = func()
                # sync_result = self.move.Sync() # 会阻塞子线程
                if point_list:
                    if not self.wait_arrive(point_list):
                        print("❌ 机械臂无法到达指定位置")
                        return
                print("✅ 机械臂运动完成")
            except Exception as e:
                exception_container[0] = e
                print("运动时发生错误", e)
            finally:
                event.set()

        # 在新线程中执行命令
        command_thread = threading.Thread(target=run_command, daemon=True)
        command_thread.start()

        # 等待命令执行完成
        event.wait()

        # 检查是否有异常
        if exception_container[0] is not None:
            print(f"❌ {description}失败: {str(exception_container[0])}")
            return None

        return result_container

    def _execute_with_move(self, func, point_list=[], coordinate_mode=0, description=""):
        """执行移动操作"""
        # 记录操作（如果处于暂停状态）
        if self.paused:
            operation = ("move", func)
            self.paused_operations.append(operation)
            while self.paused:
                self.pause_event.wait()  # 阻塞直到事件被触发
        result = self._execute_command_async(func, point_list, coordinate_mode, description=description)
        return result is not None
    def _execute_with_dashboard(self, func, description=""):
        """执行需要dashboard连接的操作"""

        if not self.dashboard:
            print(f"⚠️  Dashboard连接未建立，{description}失败")
            return None

        # 检查socket是否有效
        if not hasattr(self.dashboard, 'socket_dobot') or not self.dashboard.socket_dobot:
            print(f"⚠️  Dashboard socket无效，{description}失败")
            return None

        try:
            result = func()
            return result
        except Exception as e:
            print(f"❌ {description}失败: {str(e)}")
            return None
    # 报警处理
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
        if '17' in error_info:
            self.is_wait_no = True

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
            if "0" == response.split(",")[0]:
                print("✅ 报警已清除")
                self.current_error_status = None
                return True
            else:
                print(f"❌ 清除报警失败: {response}")
                return False
        return False

    # 检查

    def is_alarm_active(self):
        """检查是否有活动报警"""
        return self.current_error_status is not None and self.current_error_status != "0"
    def is_point_reachable(self, x, y, z, rx, ry, rz, coordinate_mode=0):
        """
        使用API实际验证点位可达性
        @param x: X坐标或J1角度
        @param y: Y坐标或J2角度
        @param z: Z坐标或J3角度
        @param rx: Rx角度或J4角度
        @param ry: Ry角度或J5角度
        @param rz: Rz角度或J6角度
        @param coordinate_mode: 坐标模式，0为笛卡尔坐标(pose)，1为关节坐标(joint)
        @return: (bool, str) 是否可达和描述信息
        """
        try:
            if coordinate_mode == 0:  # 笛卡尔坐标模式
                # 使用逆解运算API验证点位是否可达
                result = self.dashboard.InverseSolution(x, y, z, rx, ry, rz)

                # 解析返回结果判断是否可达
                if result and "0" == result.split(",")[0]:  # ErrorID为0表示成功
                    return True, "点位可达"
                else:
                    return False, "点位不可达"

            elif coordinate_mode == 1:  # 关节坐标模式
                # 使用正解运算API验证关节坐标是否有效
                result = self.dashboard.PositiveSolution(x, y, z, rx, ry, rz)

                # 解析返回结果判断是否可达
                if result and "0" == result.split(",")[0]:  # ErrorID为0表示成功
                    return True, "关节坐标有效"
                else:
                    return False, "关节坐标无效"

            else:
                return False, f"不支持的坐标模式: {coordinate_mode}"

        except Exception as e:
            return False, f"验证过程出错: {str(e)}"

    def is_basic_connected(self):
        """
        基本连接检查 - 只检查socket连接状态
        """
        try:
            # 检查所有连接对象是否存在
            if not all([self.dashboard, self.move, self.feed]):
                return False

            # 检查socket对象是否存在
            if not all([self.dashboard.socket_dobot,
                       self.move.socket_dobot,
                       self.feed.socket_dobot]):
                return False

            # 尝试发送简单命令检查连接
            try:
                # 发送一个简单的命令来验证连接
                response = self.dashboard.RobotMode()
                return response is not None and len(response) > 0
            except:
                return False

        except Exception as e:
            print(f"基本连接检查出错: {e}")
            return False
    def is_mode_checked(self):
        """
        检查Dobot机械臂的详细状态
        """
        try:
            # 获取机器人模式
            robot_mode_response = self.dashboard.RobotMode()
            # print(f"原始机器人模式响应: {robot_mode_response}")

            # 解析响应
            if robot_mode_response:
                # 提取模式值 - 更安全的解析方法
                mode = None
                if "{" in robot_mode_response and "}" in robot_mode_response:
                    try:
                        mode_value = robot_mode_response.split("{")[1].split("}")[0]
                        if mode_value.strip():  # 确保不是空字符串
                            mode = int(mode_value)
                    except (ValueError, IndexError) as e:
                        print(f"解析模式值时出错: {e}")
                        return False
                else:
                    # 尝试直接从响应中提取数字
                    import re
                    numbers = re.findall(r'\d+', robot_mode_response)
                    if numbers:
                        mode = int(numbers[0])

                if mode is not None:
                    # 根据3.8 RobotMode文档更新的模式定义
                    mode_descriptions = {
                        1: "初始化",
                        2: "抱闸松开",
                        3: "保留位",
                        4: "未使能(抱闸未松开)",
                        5: "使能(空闲)",  # 可以接收指令的状态
                        6: "拖拽",
                        7: "运行状态",
                        8: "拖拽录制",
                        9: "报警",
                        10: "暂停状态",
                        11: "点动"
                    }

                    mode_description = mode_descriptions.get(mode, f"未知模式({mode})")
                    # print(f"机器人当前模式: {mode} - {mode_description}")

                    # 检查是否可以接收指令
                    if mode == 5:  # 使能(空闲)
                        print("✅ 机器人已使能，可以接收运动指令")
                        return True
                    elif mode == 7:  # 运行状态
                        print("⚠️ 机器人正在运行中")
                        return True
                    elif mode == 9:  # 报警
                        print("❌ 机器人处于报警状态")
                        # 检查具体错误
                        error_info = self.get_current_error()
                        if error_info:
                            print(f"错误代码: {error_info}")
                        return False
                    elif mode == 10:  # 暂停状态
                        print("⏸️ 机器人处于暂停状态")
                        return False
                    elif mode == 11:  # 点动
                        print("🕹️ 机器人处于点动状态")
                        return True
                    elif mode == 4:  # 未使能
                        print("⚠️ 机器人未使能，需要使能后才能接收指令")
                        return False
                    else:
                        print(f"⚠️ 机器人处于{mode_description}状态 (模式: {mode})")
                        # 模式1,2,3,6,8通常表示机器人不能立即接收运动指令
                        return mode in [1, 2, 3, 6, 8] == False  # 只有不在这些模式中才返回True
                else:
                    print("⚠️ 无法解析机器人模式值")
                    return False
            else:
                print("❌ 无法获取机器人模式信息")
                return False

        except Exception as e:
            print(f"❌ 检查机器人状态时出错: {e}")
            return False

    def is_connected(self):
        """
        检查Dobot机械臂的连接
        """
        return self.is_basic_connected()

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

    # 反馈

    def _start_feed_thread(self):
        """启动反馈线程"""
        self.feed_thread = threading.Thread(target=self._get_feed)
        self.feed_thread.setDaemon(True)
        self.feed_thread.start()

    def _get_feed(self):
        """获取机械臂反馈数据"""
        hasRead = 0
        print("数据获取线程已启动...")
        while True:
            try:
                # print("数据获取中...")
                result = self.feed.feedBackData()
                try:
                    self.ToolVectorActual = result["ToolVectorActual"][0]
                    self.QActual = result["QActual"][0]
                    self.DigitalInputs = result["DigitalInputs"][0]
                    self.DigitalOutputs = result["DigitalOutputs"][0]
                except Exception as e:
                    pass
                    # time.sleep(0.05)

                time.sleep(0.05)
            except:
                if self.feed:
                    self.feed.close()
                return

    def wait_arrive(self, point_list, timeout=10):
        """等待机械臂到达目标位置"""
        start_time = time.time()
        last_valid_position_time = time.time()

        while True:
            if self.is_wait_no:
                self.is_wait_no = False
                return False
                # 检查是否超时
            if time.time() - start_time > timeout:
                print(f"⚠️ 等待机械臂到达超时 ({timeout}秒)")
                return False

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
            else:
                if time.time() - last_valid_position_time > 5:  # 5秒内未收到有效位置
                    print("⚠️ 位置数据持续异常")
                    return False
                time.sleep(0.1)
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
    # 设置参数
    def set_speed(self, speed_factor=0.5):
        """设置运动速度因子"""
        if 0 <= speed_factor <= 1:
            speed_factor = int(speed_factor * 100)
        result = self._execute_with_dashboard(
            lambda: self.dashboard.SpeedFactor(speed_factor),
            description=f"设置速度因子为 {speed_factor}"
        )
        if result:
            print(f"⚙️ 设置速度因子为 {speed_factor}")

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

        coordinate_mode = 0  # 默认为笛卡尔坐标模式

        if move_type == "linear":
            func = self.move.MovL
            move_desc = "线性移动到"
        else:  # joint
            func = self.move.MovJ
            move_desc = "关节运动到"
            coordinate_mode = 1

        # 执行移动
        point_list = [limited_point[0], limited_point[1], limited_point[2],
                        limited_point[3], limited_point[4], limited_point[5]]
        result = self._execute_with_move(
            lambda: func(limited_point[0], limited_point[1], limited_point[2],
                 limited_point[3], limited_point[4], limited_point[5]),
            point_list,
            coordinate_mode,
            description=f"{move_desc} X:{limited_point[0]:.3f}, Y:{limited_point[1]:.3f}, Z:{limited_point[2]:.3f}"
        )

        if result:
            print(f"🕹️ {move_desc} X:{limited_point[0]:.3f}, Y:{limited_point[1]:.3f}, Z:{limited_point[2]:.3f}")
            return result
        return False

    def wait_mvoe(self):
        """等待运动完成"""
        sync_result = self._execute_with_move(
            self.move.Sync,
            description="等待运动完成"
        )
        if sync_result:
            print("✅ 机械臂运动完成")
        else:
            print("❌ 等待运动完成时发生错误")

    def move_jog(self, axisID):
        """
        点动/停止点动

        J1+ 表示关节1正方向运动， J1- 表示关节1负方向运动
        J2+ 表示关节2正方向运动， J2- 表示关节2负方向运动
        J3+ 表示关节3正方向运动，J3- 表示关节3负方向运动
        J4+ 表示关节4正方向运动，J4- 表示关节4负方向运动
        J5+ 表示关节5正方向运动，J5- 表示关节5负方向运动
        J6+ 表示关节6正方向运动，J6- 表示关节6负方向运动
        X+ 表示X轴正方向运动，X- 表示X轴负方向运动
        Y+ 表示Y轴正方向运动，Y- 表示Y轴负方向运动
        Z+ 表示Z轴正方向运动，Z- 表示Z轴负方向运动
        Rx+ 表示Rx轴正方向运动，Rx- 表示Rx轴负方向运动
        Ry+ 表示Ry轴正方向运动，Ry- 表示Ry轴负方向运动
        Rz+ 表示Rz轴正方向运动，Rz- 表示Rz轴负方向运动
        空字符串表示停止点动
        """
        try:
            if axisID == "":
                # 停止点动
                result = self._execute_with_move(
                    lambda: self.move.MoveJog(""),
                    description="停止点动"
                )
            else:
                # 开始点动
                result = self._execute_with_move(
                    lambda: self.move.MoveJog(axisID),
                    description=f"点动 {axisID}"
                )
            return result
        except Exception as e:
            print(f"❌ 点动操作失败: {str(e)}")
            return False

    def run_point_l(self, point_list: list):
        """运行到指定点(直线运动)"""
        self._move_to_point(point_list, "linear")

    def run_point_j(self, point_list: list):
        """关节运动到指定点"""
        return self._move_to_point(point_list, "joint")

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
    def move_incremental_spatial(self, j1=0, j2=0, j3=0, j4=0, j5=0, j6=0):
        """
        在当前位置基础上进行增量移动

        @param j1: 关节1增量角度
        @param j2: 关节2增量角度
        @param j3: 关节3增量角度
        @param j4: 关节4增量角度
        @param j5: 关节5增量角度
        @param j6: 关节6增量角度
        @return: 是否移动成功
        """
        try:
            # 获取当前关节位置
            current_joints = self.get_current_joint_position()
            if current_joints is None:
                print("❌ 无法获取当前关节位置")
                return False

            # 计算目标关节位置
            target_joints = [
                current_joints[0] + j1,
                current_joints[1] + j2,
                current_joints[2] + j3,
                current_joints[3] + j4,
                current_joints[4] + j5,
                current_joints[5] + j6
            ]

            # 执行关节运动到目标位置
            result = self.run_point_j(target_joints)

            if result:
                print(f"🕹️ 增量移动完成: J1:{j1}, J2:{j2}, J3:{j3}, J4:{j4}, J5:{j5}, J6:{j6}")
                return True
            else:
                print("❌ 增量移动失败")
                return False

        except Exception as e:
            print(f"❌ 增量移动过程中发生错误: {str(e)}")
            return False
    def move_incremental_cartesian(self, x=0, y=0, z=0, rx=0, ry=0, rz=0):
        """
        在当前位置基础上进行笛卡尔坐标增量移动

        @param x: X轴增量(mm)
        @param y: Y轴增量(mm)
        @param z: Z轴增量(mm)
        @param rx: Rx轴增量角度
        @param ry: Ry轴增量角度
        @param rz: Rz轴增量角度
        @return: 是否移动成功
        """
        try:
            # 获取当前笛卡尔位置
            current_position = self.get_current_position()
            if current_position is None:
                print("❌ 无法获取当前笛卡尔位置")
                return False

            # 计算目标位置
            target_position = [
                current_position[0] + x,
                current_position[1] + y,
                current_position[2] + z,
                current_position[3] + rx,
                current_position[4] + ry,
                current_position[5] + rz
            ]

            # 执行直线运动到目标位置
            result = self._execute_with_move(
                lambda: self.move.MovL(target_position[0], target_position[1], target_position[2],
                                      target_position[3], target_position[4], target_position[5]),
                target_position,
                description=f"笛卡尔增量移动 X:{x}, Y:{y}, Z:{z}, Rx:{rx}, Ry:{ry}, Rz:{rz}"
            )

            if result:
                print(f"🕹️ 笛卡尔增量移动完成: X:{x}, Y:{y}, Z:{z}, Rx:{rx}, Ry:{ry}, Rz:{rz}")
                return True
            else:
                print("❌ 笛卡尔增量移动失败")
                return False

        except Exception as e:
            print(f"❌ 笛卡尔增量移动过程中发生错误: {str(e)}")
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
            return self.ToolVectorActual
        except Exception as e:
            print(f"❌ 获取位置失败: {str(e)}")
            return None
    def get_current_joint_position(self):
        """
        获取当前关节位置

        @return: 当前关节位置 (q1, q2, q3, q4, q5, q6) 或 None（如果获取失败）
        """
        try:
            return self.QActual
        except Exception as e:
            print(f"❌ 获取关节位置失败: {str(e)}")
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

    def get_di(self, io_index,is_log=True):
        """获取数字输入"""
        result = self._execute_with_dashboard(
            lambda: self.dashboard.DI(io_index),
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

    def get_dis(self, *arg):
        """批量获取数字输入"""
        result = self._execute_with_dashboard(
            lambda: self.dashboard.DIGroup(*arg),
            description=f"获取批量DI{list(arg)}"
        )
        if result:
            try:
                # 解析返回结果 '0,{0,0,0},DIGroup(1,2,3);'
                # 提取 {0,0,0} 部分
                if "," in result:

                    values_str = result.split("{")[1].split("}")[0]
                    di_values = [int(x.strip()) for x in values_str.split(",")]

                    xt = time.time()
                    return di_values

                # 如果解析失败，返回默认值
                return [-1] * len(arg)
            except Exception as e:
                # print(f"⚠️ 解析DIGroup结果失败: {result}, 错误: {str(e)}")
                return [-1] * len(arg)
        else:
            # 如果没有返回结果，返回默认值
            return [-1] * len(arg)

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

    def Sync(self):
        """
        阻塞程序执行队列指令，所有队列指令执行完毕后返回
        """
        result = self._execute_with_move(
            lambda: self.move.Sync(),
            description="同步执行队列指令"
        )
        return result

    def hll(self, i=-1, dos=[4, 5]):
        """
        异步设置DO状态
        @param i: 需要点亮的DO编号，-1表示全部关闭
        @param dos: 需要控制的DO编号列表
        @return: Future对象列表
        """
        futures = []
        for do_num in dos:
            futures.append(do_num)
            futures.append(1 if do_num == i else 0)
        result = self._execute_with_dashboard(
            lambda: self.dashboard.DOGroup(*futures),
            description=f"设置DO"
        )
        return result is not None

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


def connect_and_check_speed(ip="192.168.5.1", port=30003, dashboard_port=29999, feed_port=30004):
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

def get_dis(urController,start_i,end_i):
    while 1:
        for i in range(start_i,end_i):
            di = urController.get_di(i, is_log=False)
            # if di == 1:
            print(i,di)
        time.sleep(1)


if __name__ == "__main__":
    # # 使用新的连接和检查函数
    urController = connect_and_check_speed()
    print(f"📍 当前位置: {urController.get_current_position()}")
    if urController:
        # urController.hll(5)
        # while 1:
        #     # print(get_dis(urController,1,4))
        #     print(urController.get_dis(1,2,3))
        #     time.sleep(0.1)
        # x,y = pixel_to_world(114,140)
        # print(x,y)
        # urController.run_point_j([-173,-198,195,-179,0.2,-179])
        # time.sleep(5)
        # urController.run_point_j(RED_CAMERA)
        # urController.pause()
        # print('暂停')
        # urController.run_point_j(BLACK_CAMERA)
        # urController.hll(5)
        # urController.resume()
        # print('继续')
        # urController.wait_mvoe()
        # urController.run_point_j(BLACK_CAMERA)

        # urController.run_point_j([154,-377,205,-179,0.2,-179])
        # time.sleep(10)
        # alarm_handling_test(urController)
        # urController.set_do(IO_QI, 1)  # 吸合123456
        # urController.run_point_j(RED_CAMERA)
        # urController.run_point_j([175,-538,195,-179,0.2,-179])
        # time.sleep(5)
        # urController.run_point_j(FRUIT_CAMERA)
        # urController.run_point_j(BLACK_CAMERA)
        # time.sleep(1000)
        # urController.run_point_j([125,-343,195,-179,0.2,-179])
        # time.sleep(5)
        # urController.set_do(IO_QI, 0)  # 吸合123456

        # urController.run_point_j(RCV_CAMERA)
        # alarm_handling_test(urController)
        time.sleep(3)
        # urController.wait_arrive(BLACK_CAMERA)
        # urController.move_to(-410.96, -299.49,260)-
        # time.sleep(10)
        # urController.move_to(216, -596,250)
        # time.sleep(10)
        # urController.move_to(-350,-440,250)
        # time.sleep(10)
        # urController.move_to(-230,-440,210)
        # time.sleep(5)
        # print(urController.is_point_reachable(-400, -440,319)

        # time.sleep(1000)
        alarm_handling_test(urController)
        # get_dis(urController,1,4)
        # get_dos(urController,1,4)
        # urController.set_do(5, 1)
        # time.sleep(3)
        # urController.set_do(5, 0)
        # time.sleep(3)
        # urController.set_do(1, 1)

        # 断开连接
        urController.disconnect()
    else:
        print("❌ 无法建立控制器连接")
