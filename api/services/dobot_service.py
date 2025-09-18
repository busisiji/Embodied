# api/services/dobot_service.py
import time
from typing import Optional
from api.services.process_service import ProcessService
from dobot.dobot_control import URController


class DobotService:
    def __init__(self):
        self.controller: Optional[URController] = None
        self.is_connected = False

    def connect_dobot(self, ip="192.168.5.1", port=30003, dashboard_port=29999,
                     feed_port=30006, acceleration=0.5, velocity=0.3):
        """
        连接机械臂
        """
        try:
            if self.controller and self.is_connected:
                self.controller.disconnect()

            self.controller = URController(
                ip=ip,
                port=port,
                dashboard_port=dashboard_port,
                feed_port=feed_port,
                acceleration=acceleration,
                velocity=velocity
            )

            self.is_connected = self.controller.connected
            return self.is_connected
        except Exception as e:
            print(f"连接机械臂失败: {str(e)}")
            self.is_connected = False
            return False

    def disconnect_dobot(self):
        """
        断开机械臂连接
        """
        if self.controller and self.is_connected:
            self.controller.disconnect()
            self.is_connected = False

    def start_pick_and_place_task(self, x: float, y: float, z: float = 0.05):
        """
        启动抓取和放置任务，通过ProcessService管理
        """
        def pick_and_place_process():
            try:
                if not self.controller or not self.is_connected:
                    return {"status": "error", "message": "机械臂未连接"}

                success = self.controller.pick_place(x, y, z)
                return {
                    "status": "success" if success else "error",
                    "message": "抓取放置任务完成" if success else "抓取放置任务失败"
                }
            except Exception as e:
                return {"status": "error", "message": f"任务执行失败: {str(e)}"}

        # 使用ProcessService启动任务
        process_result = ProcessService.start_training(
            init_model=None,
            data_path=None,
            epochs=1,
            batch_size=1,
            custom_function=pick_and_place_process
        )

        return process_result

    def start_continuous_operation_task(self, operations):
        """
        启动连续操作任务
        """
        def continuous_operation_process():
            try:
                if not self.controller or not self.is_connected:
                    return {"status": "error", "message": "机械臂未连接"}

                results = []
                for op in operations:
                    op_type = op.get("type")
                    params = op.get("params", {})

                    if op_type == "move_to":
                        success = self.controller.move_to(**params)
                        results.append({"operation": "move_to", "success": success})
                    elif op_type == "pick_place":
                        success = self.controller.pick_place(**params)
                        results.append({"operation": "pick_place", "success": success})
                    elif op_type == "home":
                        self.controller.move_home()
                        results.append({"operation": "home", "success": True})
                    time.sleep(0.5)  # 操作间隔

                return {
                    "status": "success",
                    "message": "连续操作任务完成",
                    "results": results
                }
            except Exception as e:
                return {"status": "error", "message": f"任务执行失败: {str(e)}"}

        # 使用ProcessService启动任务
        process_result = ProcessService.start_training(
            init_model=None,
            data_path=None,
            epochs=1,
            batch_size=1,
            custom_function=continuous_operation_process
        )

        return process_result

# 全局实例
dobot_service = DobotService()
