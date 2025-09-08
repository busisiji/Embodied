# robotic_arm_control_ui.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import json
import os
from dobot.dobot_control import URController
import numpy as np

class RoboticArmControlUI:
    def __init__(self, root):
        self.root = root
        self.root.title("机械臂控制界面")
        self.root.geometry("1000x850")

        # 初始化机械臂控制器
        self.controller = URController()

        # 位置和状态变量
        self.current_position = [0.0, 0.0, 00, 0.0, 0.0, 0.0]
        self.current_velocity = 0.3
        self.current_acceleration = 0.5
        self.error_status = "无错误"

        # 存点数据
        self.saved_points = {}
        self.points_file = "saved_points.json"

        # 更新标志
        self.updating = True

        # 点动控制变量
        self.jog_step = 1.0  # 默认步值为1.0mm或1度
        self.jog_active = False  # 点动是否激活
        self.jog_axis = None    # 当前点动轴
        self.jog_direction = None  # 当前点动方向
        self.jog_thread = None  # 点动线程

        # 创建界面
        self.create_widgets()

        # 加载已保存的点
        self.load_saved_points()

        # 启动状态更新线程
        self.start_status_update()

    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # 状态显示区域
        status_frame = ttk.LabelFrame(main_frame, text="机械臂状态", padding="10")
        status_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # 位置信息
        position_frame = ttk.Frame(status_frame)
        position_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        ttk.Label(position_frame, text="当前位置:").grid(row=0, column=0, sticky=tk.W)
        self.position_label = ttk.Label(position_frame, text="X: 0.00, Y: 0.00, Z: 0.00, Rx: 0.00, Ry: 0.00, Rz: 0.00")
        self.position_label.grid(row=0, column=1, padx=(10, 0), sticky=tk.W)

        # 速度和加速度信息
        speed_frame = ttk.Frame(status_frame)
        speed_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

        ttk.Label(speed_frame, text="速度:").grid(row=0, column=0, sticky=tk.W)
        self.velocity_label = ttk.Label(speed_frame, text="0.30")
        self.velocity_label.grid(row=0, column=1, padx=(10, 20), sticky=tk.W)

        ttk.Label(speed_frame, text="加速度:").grid(row=0, column=2, sticky=tk.W)
        self.acceleration_label = ttk.Label(speed_frame, text="0.50")
        self.acceleration_label.grid(row=0, column=3, padx=(10, 0), sticky=tk.W)

        # 错误信息
        error_frame = ttk.Frame(status_frame)
        error_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

        ttk.Label(error_frame, text="错误状态:").grid(row=0, column=0, sticky=tk.W)
        self.error_label = ttk.Label(error_frame, text="无错误", foreground="green")
        self.error_label.grid(row=0, column=1, padx=(10, 0), sticky=tk.W)

        # 控制区域
        control_frame = ttk.LabelFrame(main_frame, text="机械臂控制", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        control_frame.columnconfigure(0, weight=1)

        # 移动控制
        move_frame = ttk.LabelFrame(control_frame, text="位置控制", padding="10")
        move_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        move_frame.columnconfigure(1, weight=1)
        move_frame.columnconfigure(3, weight=1)
        move_frame.columnconfigure(5, weight=1)

        # X坐标
        ttk.Label(move_frame, text="X:").grid(row=0, column=0, sticky=tk.W)
        self.x_entry = ttk.Entry(move_frame, width=10)
        self.x_entry.grid(row=0, column=1, padx=(5, 10), sticky=tk.W)
        self.x_entry.insert(0, "0.0")

        # Y坐标
        ttk.Label(move_frame, text="Y:").grid(row=0, column=2, sticky=tk.W)
        self.y_entry = ttk.Entry(move_frame, width=10)
        self.y_entry.grid(row=0, column=3, padx=(5, 10), sticky=tk.W)
        self.y_entry.insert(0, "0.0")

        # Z坐标
        ttk.Label(move_frame, text="Z:").grid(row=0, column=4, sticky=tk.W)
        self.z_entry = ttk.Entry(move_frame, width=10)
        self.z_entry.grid(row=0, column=5, padx=(5, 10), sticky=tk.W)
        self.z_entry.insert(0, "0.0")

        # 移动按钮
        move_button = ttk.Button(move_frame, text="移动到指定位置", command=self.move_to_position)
        move_button.grid(row=0, column=6, padx=(10, 0))

        # 六轴点动控制
        jog_frame = ttk.LabelFrame(control_frame, text="六轴点动控制", padding="10")
        jog_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # 步值设置
        step_frame = ttk.Frame(jog_frame)
        step_frame.grid(row=0, column=0, columnspan=7, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(step_frame, text="步值 (mm/度):").grid(row=0, column=0, sticky=tk.W)
        self.step_entry = ttk.Entry(step_frame, width=10)
        self.step_entry.grid(row=0, column=1, padx=(5, 10), sticky=tk.W)
        self.step_entry.insert(0, "1.0")

        set_step_button = ttk.Button(step_frame, text="设置步值", command=self.set_jog_step)
        set_step_button.grid(row=0, column=2, padx=(0, 10))

        # 显示当前步值
        self.step_value_label = ttk.Label(step_frame, text=f"当前步值: {self.jog_step}")
        self.step_value_label.grid(row=0, column=3, padx=(10, 0), sticky=tk.W)

        # 创建六轴点动控制按钮
        axes = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz']
        self.jog_buttons = {}  # 存储点动按钮引用

        for i, axis in enumerate(axes):
            axis_frame = ttk.Frame(jog_frame)
            axis_frame.grid(row=1, column=i, padx=5, sticky=(tk.W, tk.E))

            # 正向点动按钮
            plus_btn = tk.Button(
                axis_frame,
                text=f"{axis}+",
                width=5,
                repeatdelay=500,  # 按住0.5秒后开始重复
                repeatinterval=100  # 每0.1秒重复一次
            )
            plus_btn.grid(row=0, column=0, pady=(0, 5))
            # 绑定按下和释放事件
            plus_btn.bind('<ButtonPress-1>', lambda e, a=axis, d=1: self.start_continuous_jog(a, d))
            plus_btn.bind('<ButtonRelease-1>', lambda e: self.stop_jog())

            # 负向点动按钮
            minus_btn = tk.Button(
                axis_frame,
                text=f"{axis}-",
                width=5,
                repeatdelay=500,  # 按住0.5秒后开始重复
                repeatinterval=100  # 每0.1秒重复一次
            )
            minus_btn.grid(row=1, column=0, pady=(0, 5))
            # 绑定按下和释放事件
            minus_btn.bind('<ButtonPress-1>', lambda e, a=axis, d=-1: self.start_continuous_jog(a, d))
            minus_btn.bind('<ButtonRelease-1>', lambda e: self.stop_jog())

            # 停止按钮
            stop_btn = ttk.Button(
                axis_frame,
                text="停止",
                width=5,
                command=self.stop_jog
            )
            stop_btn.grid(row=2, column=0)

            self.jog_buttons[axis] = {
                'plus': plus_btn,
                'minus': minus_btn,
                'stop': stop_btn
            }

        # 存点控制
        save_point_frame = ttk.LabelFrame(control_frame, text="存点操作", padding="10")
        save_point_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        save_point_frame.columnconfigure(1, weight=1)

        ttk.Label(save_point_frame, text="点名称:").grid(row=0, column=0, sticky=tk.W)
        self.point_name_entry = ttk.Entry(save_point_frame, width=20)
        self.point_name_entry.grid(row=0, column=1, padx=(5, 10), sticky=(tk.W, tk.E))

        save_point_button = ttk.Button(save_point_frame, text="保存当前位置", command=self.save_current_point)
        save_point_button.grid(row=0, column=2, padx=(0, 10))

        delete_point_button = ttk.Button(save_point_frame, text="删除选中点", command=self.delete_selected_point)
        delete_point_button.grid(row=0, column=3, padx=(0, 10))

        # 已保存点列表
        points_list_frame = ttk.LabelFrame(control_frame, text="已保存点位", padding="10")
        points_list_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        points_list_frame.columnconfigure(0, weight=1)
        points_list_frame.rowconfigure(0, weight=1)

        # 创建列表框和滚动条
        listbox_frame = ttk.Frame(points_list_frame)
        listbox_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        listbox_frame.columnconfigure(0, weight=1)
        listbox_frame.rowconfigure(0, weight=1)

        self.points_listbox = tk.Listbox(listbox_frame)
        self.points_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        points_scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.points_listbox.yview)
        points_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.points_listbox.configure(yscrollcommand=points_scrollbar.set)

        # 绑定双击事件
        self.points_listbox.bind("<Double-Button-1>", self.move_to_saved_point)

        # 点操作按钮
        point_action_frame = ttk.Frame(points_list_frame)
        point_action_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

        move_to_point_button = ttk.Button(point_action_frame, text="移动到选中点", command=self.move_to_saved_point)
        move_to_point_button.grid(row=0, column=0, padx=(0, 10))

        refresh_points_button = ttk.Button(point_action_frame, text="刷新列表", command=self.refresh_points_list)
        refresh_points_button.grid(row=0, column=1, padx=(0, 10))

        save_points_file_button = ttk.Button(point_action_frame, text="导出点位", command=self.export_points)
        save_points_file_button.grid(row=0, column=2, padx=(0, 10))

        load_points_file_button = ttk.Button(point_action_frame, text="导入点位", command=self.import_points)
        load_points_file_button.grid(row=0, column=3, padx=(0, 10))

        # 预设位置控制
        preset_frame = ttk.LabelFrame(control_frame, text="预设位置", padding="10")
        preset_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        home_button = ttk.Button(preset_frame, text="回家", command=self.move_home)
        home_button.grid(row=0, column=0, padx=(0, 10))

        red_camera_button = ttk.Button(preset_frame, text="红方摄像头位置", command=self.move_to_red_camera)
        red_camera_button.grid(row=0, column=1, padx=(0, 10))

        black_camera_button = ttk.Button(preset_frame, text="黑方摄像头位置", command=self.move_to_black_camera)
        black_camera_button.grid(row=0, column=2, padx=(0, 10))

        # 速度控制
        speed_control_frame = ttk.LabelFrame(control_frame, text="速度控制", padding="10")
        speed_control_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(speed_control_frame, text="速度:").grid(row=0, column=0, sticky=tk.W)
        self.velocity_scale = ttk.Scale(speed_control_frame, from_=0.01, to=1.0, orient=tk.HORIZONTAL,
                                       command=self.update_velocity_display, length=200)
        self.velocity_scale.set(self.current_velocity)
        self.velocity_scale.grid(row=0, column=1, padx=(10, 10))

        self.velocity_value_label = ttk.Label(speed_control_frame, text=f"{self.current_velocity:.2f}")
        self.velocity_value_label.grid(row=0, column=2, padx=(0, 10))

        set_velocity_button = ttk.Button(speed_control_frame, text="设置速度", command=self.set_velocity)
        set_velocity_button.grid(row=0, column=3)

        # 报警处理
        alarm_frame = ttk.LabelFrame(control_frame, text="报警处理", padding="10")
        alarm_frame.grid(row=6, column=0, sticky=(tk.W, tk.E))

        clear_alarm_button = ttk.Button(alarm_frame, text="清除报警", command=self.clear_alarm)
        clear_alarm_button.grid(row=0, column=0, padx=(0, 10))

        # 连接/断开连接按钮
        connection_button = ttk.Button(main_frame, text="断开连接", command=self.disconnect)
        connection_button.grid(row=2, column=0, pady=(10, 0))

    def start_status_update(self):
        """启动状态更新线程"""
        self.status_thread = threading.Thread(target=self.update_status)
        self.status_thread.daemon = True
        self.status_thread.start()

    def update_status(self):
        """定期更新机械臂状态"""
        while self.updating:
            try:
                # 获取当前位置
                position = self.controller.get_current_position()
                if position:
                    self.current_position = position

                # 获取速度和加速度
                self.current_velocity = self.controller.velocity
                self.current_acceleration = self.controller.acceleration

                # 获取错误状态
                if self.controller.is_alarm_active():
                    self.error_status = self.controller.get_current_error() or "未知错误"
                else:
                    self.error_status = "无错误"

                # 更新界面
                self.root.after(0, self.update_ui)

                time.sleep(0.5)  # 每0.5秒更新一次
            except Exception as e:
                print(f"状态更新错误: {e}")
                time.sleep(1)

    def update_ui(self):
        """更新界面显示"""
        # 更新位置显示
        pos = self.current_position
        self.position_label.config(
            text=f"X: {pos[0]:.2f}, Y: {pos[1]:.2f}, Z: {pos[2]:.2f}, "
                 f"Rx: {pos[3]:.2f}, Ry: {pos[4]:.2f}, Rz: {pos[5]:.2f}"
        )

        # 更新速度和加速度显示
        self.velocity_label.config(text=f"{self.current_velocity:.2f}")
        self.acceleration_label.config(text=f"{self.current_acceleration:.2f}")

        # 更新错误状态显示
        if self.error_status == "无错误":
            self.error_label.config(text=self.error_status, foreground="green")
        else:
            self.error_label.config(text=self.error_status, foreground="red")

    def update_velocity_display(self, value):
        """更新速度显示"""
        # 确保velocity_value_label存在
        if hasattr(self, 'velocity_value_label'):
            self.velocity_value_label.config(text=f"{float(value):.2f}")

    def set_jog_step(self):
        """设置点动步值"""
        try:
            step_value = float(self.step_entry.get())
            if step_value <= 0:
                messagebox.showerror("错误", "步值必须大于0")
                return
            self.jog_step = step_value
            self.step_value_label.config(text=f"当前步值: {self.jog_step}")
            messagebox.showinfo("成功", f"点动步值已设置为 {self.jog_step}")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数值")

    def move_to_position(self):
        """移动到指定位置"""
        try:
            x = float(self.x_entry.get())
            y = float(self.y_entry.get())
            z = float(self.z_entry.get())

            # 执行移动
            self.controller.move_to(x, y, z)
            messagebox.showinfo("成功", "机械臂正在移动到指定位置")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的坐标值")
        except Exception as e:
            messagebox.showerror("错误", f"移动失败: {str(e)}")

    def move_home(self):
        """回家位置"""
        try:
            self.controller.move_home()
            messagebox.showinfo("成功", "机械臂正在回家")
        except Exception as e:
            messagebox.showerror("错误", f"回家失败: {str(e)}")

    def move_to_red_camera(self):
        """移动到红方摄像头位置"""
        try:
            from parameters import RED_CAMERA
            self.controller.run_point_j(RED_CAMERA)
            messagebox.showinfo("成功", "机械臂正在移动到红方摄像头位置")
        except Exception as e:
            messagebox.showerror("错误", f"移动失败: {str(e)}")

    def move_to_black_camera(self):
        """移动到黑方摄像头位置"""
        try:
            from parameters import BLACK_CAMERA
            self.controller.run_point_j(BLACK_CAMERA)
            messagebox.showinfo("成功", "机械臂正在移动到黑方摄像头位置")
        except Exception as e:
            messagebox.showerror("错误", f"移动失败: {str(e)}")

    def set_velocity(self):
        """设置速度"""
        try:
            velocity = float(self.velocity_scale.get())
            self.controller.set_velocity(velocity)
            messagebox.showinfo("成功", f"速度已设置为 {velocity:.2f}")
        except Exception as e:
            messagebox.showerror("错误", f"设置速度失败: {str(e)}")

    def clear_alarm(self):
        """清除报警"""
        try:
            if self.controller.clear_alarm():
                messagebox.showinfo("成功", "报警已清除")
            else:
                messagebox.showerror("错误", "清除报警失败")
        except Exception as e:
            messagebox.showerror("错误", f"清除报警时发生错误: {str(e)}")

    def save_current_point(self):
        """保存当前位置为点位"""
        point_name = self.point_name_entry.get().strip()
        if not point_name:
            messagebox.showerror("错误", "请输入点位名称")
            return

        try:
            # 获取当前位置
            position = self.controller.get_current_position()
            if not position:
                messagebox.showerror("错误", "无法获取当前位置")
                return

            # 保存点位
            self.saved_points[point_name] = {
                "x": position[0],
                "y": position[1],
                "z": position[2],
                "rx": position[3],
                "ry": position[4],
                "rz": position[5]
            }

            # 保存到文件
            self.save_points_to_file()

            # 更新列表显示
            self.refresh_points_list()

            messagebox.showinfo("成功", f"点位 '{point_name}' 已保存")
        except Exception as e:
            messagebox.showerror("错误", f"保存点位失败: {str(e)}")

    def delete_selected_point(self):
        """删除选中的点位"""
        selection = self.points_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请先选择一个点位")
            return

        point_name = self.points_listbox.get(selection[0]).split(" - ")[0]

        if messagebox.askyesno("确认", f"确定要删除点位 '{point_name}' 吗？"):
            try:
                if point_name in self.saved_points:
                    del self.saved_points[point_name]
                    self.save_points_to_file()
                    self.refresh_points_list()
                    messagebox.showinfo("成功", f"点位 '{point_name}' 已删除")
                else:
                    messagebox.showerror("错误", "点位不存在")
            except Exception as e:
                messagebox.showerror("错误", f"删除点位失败: {str(e)}")

    def move_to_saved_point(self, event=None):
        """移动到选中的保存点"""
        selection = self.points_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请先选择一个点位")
            return

        point_name = self.points_listbox.get(selection[0]).split(" - ")[0]

        if point_name not in self.saved_points:
            messagebox.showerror("错误", "点位不存在")
            return

        try:
            point = self.saved_points[point_name]
            self.controller.MovL(
                point["x"], point["y"], point["z"],
                point["rx"], point["ry"], point["rz"]
            )
            messagebox.showinfo("成功", f"机械臂正在移动到点位 '{point_name}'")
        except Exception as e:
            messagebox.showerror("错误", f"移动失败: {str(e)}")

    def refresh_points_list(self):
        """刷新点位列表显示"""
        self.points_listbox.delete(0, tk.END)
        for name, point in self.saved_points.items():
            self.points_listbox.insert(
                tk.END,
                f"{name} - X:{point['x']:.2f}, Y:{point['y']:.2f}, Z:{point['z']:.2f}"
            )

    def save_points_to_file(self):
        """保存点位到文件"""
        try:
            with open(self.points_file, 'w', encoding='utf-8') as f:
                json.dump(self.saved_points, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存点位文件失败: {e}")

    def load_saved_points(self):
        """从文件加载已保存的点位"""
        try:
            if os.path.exists(self.points_file):
                with open(self.points_file, 'r', encoding='utf-8') as f:
                    self.saved_points = json.load(f)
                self.refresh_points_list()
        except Exception as e:
            print(f"加载点位文件失败: {e}")
            self.saved_points = {}

    def export_points(self):
        """导出点位到文件"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.saved_points, f, ensure_ascii=False, indent=2)
                messagebox.showinfo("成功", f"点位已导出到 {file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"导出点位失败: {str(e)}")

    def import_points(self):
        """从文件导入点位"""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if file_path:
                with open(file_path, 'r', encoding='utf-8') as f:
                    imported_points = json.load(f)
                self.saved_points.update(imported_points)
                self.save_points_to_file()
                self.refresh_points_list()
                messagebox.showinfo("成功", f"点位已从 {file_path} 导入")
        except Exception as e:
            messagebox.showerror("错误", f"导入点位失败: {str(e)}")

    def start_continuous_jog(self, axis, direction):
        """
        开始连续点动运动
        :param axis: 运动轴 ('X', 'Y', 'Z', 'Rx', 'Ry', 'Rz')
        :param direction: 运动方向 (1 或 -1)
        """
        try:
            # 如果已经在点动，先停止
            # if self.jog_active:
            #     return
                # self.stop_jog()

            # 设置点动参数
            self.jog_active = True
            self.jog_axis = axis
            self.jog_direction = direction

            # 启动点动线程
            # self.jog_thread = threading.Thread(target=self._continuous_jog_worker)
            # self.jog_thread.daemon = True
            # self.jog_thread.start()
            self._continuous_jog_worker()
            print(f"开始连续{axis}轴{'正向' if direction > 0 else '负向'}点动，步值: {self.jog_step}")
        except Exception as e:
            messagebox.showerror("错误", f"点动控制失败: {str(e)}")

    def _continuous_jog_worker(self):
        """
        连续点动工作线程
        """
        try:
            # while self.jog_active:
            # 获取当前位置
            current_pos = self.controller.get_current_position()
            # if current_pos is None:
            #     print("无法获取当前位置")
            #     break

            # 计算目标位置
            target_pos = list(current_pos)
            axis_index = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz'].index(self.jog_axis)
            target_pos[axis_index] += self.jog_step * self.jog_direction

            # 移动到目标位置
            self.controller.MovL(
                target_pos[0], target_pos[1], target_pos[2],
                target_pos[3], target_pos[4], target_pos[5]
            )

                # # 等待移动完成或检查是否需要停止
                # start_time = time.time()
                # while self.jog_active and (time.time() - start_time) < 2.0:  # 最多等待2秒
                #     time.sleep(0.05)

        except Exception as e:
            print(f"连续点动运动错误: {e}")
        finally:
            self.jog_active = False

    def stop_jog(self):
        """停止点动运动"""
        try:
            self.jog_active = False
            self.jog_axis = None
            self.jog_direction = None

            # # 发送停止指令
            # self.controller.stop_jog()

            print("点动运动已停止")
        except Exception as e:
            messagebox.showerror("错误", f"停止点动运动失败: {str(e)}")

    def disconnect(self):
        """断开连接"""
        if messagebox.askyesno("确认", "确定要断开机械臂连接吗？"):
            # 停止所有点动运动
            self.stop_jog()

            self.updating = False
            self.controller.disconnect()
            self.root.quit()

    def on_closing(self):
        """关闭窗口时的处理"""
        self.disconnect()

def main():
    root = tk.Tk()
    app = RoboticArmControlUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
