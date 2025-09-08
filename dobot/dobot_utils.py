import threading
import logging
from time import sleep

import numpy as np

from dobot.dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType

logger = logging.getLogger(__name__)


class DobotArm():
    def __init__(self, ip="192.168.5.1", dashboard_port=29999, move_port=30003, feed_port=30004):
        """
        初始化机器臂连接参数。

        Parameters:
            ip (str): 机器臂控制器的IP地址。
            dashboard_port (int): Dashboard服务端口。
            move_port (int): Move服务端口。
            feed_port (int): Feed服务端口。
        """
        self.ip = ip
        self.dashboard_port = dashboard_port
        self.move_port = move_port
        self.feed_port = feed_port
        self.dashboard = None
        self.move = None
        self.feed = None
        self.current_actual = None  # 当前实际位置信息

    def connect(self):
        """
        建立与机器臂控制器的连接。

        Raises:
            Exception: 如果连接失败，抛出异常。
        """
        try:
            print("正在建立连接...")
            self.dashboard = DobotApiDashboard(self.ip, self.dashboard_port)
            self.move = DobotApiMove(self.ip, self.move_port)
            self.feed = DobotApi(self.ip, self.feed_port)
            print(">.<连接成功>!<")
        except Exception as e:
            print(":(连接失败:(")
            raise e

    def run_point(self, point_list):
        """
        控制机器臂移动到指定的坐标点。

        Parameters:
            point_list (list): 包含六个元素的目标位置 [x, y, z, rx, ry, rz]。
        """
        self.move.MovL(point_list[0], point_list[1], point_list[2], point_list[3], point_list[4], point_list[5])

    def get_feed(self):
        """
        持续获取机器臂状态信息，并更新当前实际位置。

        注意：这是一个阻塞方法，建议在单独线程中运行。
        """
        hasRead = 0
        while True:
            data = bytes()
            while hasRead < 1440:
                temp = self.feed.socket_dobot.recv(1440 - hasRead)
                if len(temp) > 0:
                    hasRead += len(temp)
                    data += temp
            hasRead = 0

            a = np.frombuffer(data, dtype=MyType)
            if hex((a['test_value'][0])) == '0x123456789abcdef':
                self.current_actual = a["tool_vector_actual"][0]

    def wait_arrive(self, point_list):
        """
        等待机器臂到达指定目标位置。

        Parameters:
            point_list (list): 目标位置 [x, y, z, rx, ry, rz]。
        """
        while True:
            is_arrive = True

            if self.current_actual is not None:
                for index in range(len(self.current_actual)):
                    if (abs(self.current_actual[index] - point_list[index]) > 20):
                        is_arrive = False

                if is_arrive:
                    return

            sleep(0.00001)

    def close_all(self):
        """
        断开与机器臂的连接，包括失能机器人、停止脚本及关闭各接口。
        """
        # 断开连接
        self.dashboard.DisableRobot()  # 失能
        self.dashboard.StopScript()
        self.dashboard.close()
        self.move.close()
        self.feed.close()

    def hll(self, f_13=0, f_14=0, f_15=0, f_16=0):
        """
        控制机器臂的数字输出信号（DO）。

        Parameters:
            f_13 (int): DO13 输出值。
            f_14 (int): DO14 输出值。
            f_15 (int): DO15 输出值。
            f_16 (int): DO16 输出值。
        """
        self.dashboard.DO(13, f_13)
        self.dashboard.DO(14, f_14)
        self.dashboard.DO(15, f_15)
        self.dashboard.DO(16, f_16)

    def start_feed_thread(self):
        """
        启动一个后台线程来持续获取机器臂状态信息。
        """
        feed_thread = threading.Thread(target=self.get_feed, args=())
        feed_thread.setDaemon(True)
        feed_thread.start()
