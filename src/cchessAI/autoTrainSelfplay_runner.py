#!/usr/bin/env python
# -*- coding: utf-8 -*-
import multiprocessing
import os
import sys
import time
import argparse
import pickle
from collections import deque
from multiprocessing import Process, Manager

multiprocessing.set_start_method('spawn', force=True)
# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"项目根目录：{project_root}")
import torch
# 导入相关模块
from src.cchessAI.parameters import MODELS, DATA_SELFPLAY, DATA_PATH, BATCH_SIZE
from api.models.database_model import DatabaseManager
from src.cchessAI.core.collect.self_play.collect_multi_threads import get_available_gpus, worker
from src.cchessAI.core.train.train import TrainPipeline
from utils.data_utils import DataReader

class RLTrainingManager():
    def __init__(self, init_model=None, interval_minutes=30, use_gpu=True,
                 temp=1.0, cpuct=5.0, user_id="admin", data_path=None):
        """初始化强化学习管理器"""
        self.init_model = init_model
        self.temp = temp
        self.cpuct = cpuct
        self.interval = interval_minutes  # 存储为分钟数
        self.use_gpu = use_gpu
        self.user_id = user_id
        self.data_path = data_path
        self.last_train_time = None
        self.training_process = None
        self.selfplay_processes = []
        self.manager = Manager()
        self.running = True
        self.db_manager = DatabaseManager()

        self.model_path = init_model

    def start_selfplay(self):
        """启动自我对弈进程"""
        try:
            self.selfplay_processes = []
            if not self.use_gpu:
                print(f"[{time.strftime('%H:%M:%S')}] 启动 {self.workers} 个多进程自我对弈")
                # 多进程模式
                available_gpus = get_available_gpus()
                num_gpus = len(available_gpus)
                base_port = 8000

                for i in range(self.workers):
                    gpu_id = available_gpus[i % num_gpus] if (self.use_gpu and num_gpus > 0) else None
                    port = base_port + i
                    p = Process(target=worker, args=(1, gpu_id, port, False,self.init_model,self.data_path, self.temp, self.cpuct))
                    p.start()
                    self.selfplay_processes.append(p)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] 启动单GPU自我对弈")
                # 单GPU模式
                from src.cchessAI.core.collect.self_play.collect_rl import CollectPipeline
                collecting_pipeline = CollectPipeline(init_model=self.init_model)
                p = Process(target=collecting_pipeline.run, args=(False,))
                p.start()
                self.selfplay_processes.append(p)

            print(f"[{time.strftime('%H:%M:%S')}] 自我对弈进程已启动")

        except Exception as e:
            print(f"[ERROR] 启动自我对弈进程失败: {e}")
            self.stop_all()

    def stop_selfplay(self):
        """停止所有自我对弈进程"""
        for p in self.selfplay_processes:
            if p.is_alive():
                p.terminate()
                p.join()
    # 修改 merge_data 方法
    def merge_data(self):
        """合并子进程采集的数据"""
        try:
            print(f"[{time.strftime('%H:%M:%S')}] 开始合并子进程数据")

            merged_data = []
            total_iters = 0

            if not self.use_gpu:
                # 遍历所有子文件并读取数据
                for i in range(len(self.selfplay_processes)):
                    pid = 8000 + i  # 子进程端口号
                    sub_data_path = os.path.join(DATA_SELFPLAY, f"data_{pid}.pkl")

                    if os.path.exists(sub_data_path):
                        try:
                            with open(sub_data_path, "rb") as f:
                                while True:
                                    try:
                                        item = pickle.load(f)
                                        merged_data.extend(item.get("data_buffer", []))
                                        total_iters += item.get("iters", 0)
                                    except EOFError:
                                        break
                            print(
                                f"[{time.strftime('%H:%M:%S')}][PID={pid}] 已读取子文件数据，当前数据量: {len(merged_data)}, 局数{total_iters}")
                            os.remove(sub_data_path)  # 删除子文件
                        except Exception as e:
                            print(f"[{time.strftime('%H:%M:%S')}][PID={pid}] 读取子文件失败：{e}")

            # 获取最新的数据文件记录
            latest_data_record = self.db_manager.get_latest_data_file(self.user_id)

            # 如果存在历史数据文件，则加载并合并
            if latest_data_record and os.path.exists(latest_data_record['file_path']):
                try:
                    with open(latest_data_record['file_path'], "rb") as f:
                        data = pickle.load(f)
                        loaded_data = data.get("data_buffer", [])
                        merged_data.extend(loaded_data)
                        total_iters += data.get("iters", 0)
                        print(
                            f"[{time.strftime('%H:%M:%S')}] 已加载历史数据文件: {latest_data_record['file_path']}, 共 {len(loaded_data)} 条样本, 局数{total_iters}")
                except Exception as e:
                    print(f"[{time.strftime('%H:%M:%S')}] 加载历史数据文件失败：{e}")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] 未找到任何历史数据文件")

            # 直接保存到指定的数据文件路径，不重命名
            if not total_iters:
                return None

            with open(self.data_path, "wb") as f:
                pickle.dump({
                    "data_buffer": merged_data,
                    "iters": total_iters
                }, f)

            # 更新数据库中的数据文件记录
            if latest_data_record:
                # 更新现有记录
                self.db_manager.update_data_file(latest_data_record['id'], game_count=total_iters,
                                                 data_length=len(merged_data))
            else:
                # 创建新记录
                self.db_manager.add_data_file(
                    user_id=self.user_id,
                    game_count=total_iters,
                    data_length=len(merged_data),
                    file_path=self.data_path
                )

            print(
                f"[{time.strftime('%H:%M:%S')}] 数据已保存至 {self.data_path}, 共 {len(merged_data)} 条样本, 局数{total_iters}")
            return self.data_path

        except Exception as e:
            print(f"[ERROR] 数据合并失败: {e}")
            return None

    def train_model(self, data_path):
        """训练模型"""
        try:
            print(f"[{time.strftime('%H:%M:%S')}] 开始训练模型")

            # 初始化训练管道
            if not self.init_model:
                train_pipeline = TrainPipeline()
            else:
                train_pipeline = TrainPipeline(init_model=self.init_model)

            # 设置数据路径
            train_pipeline.reader = DataReader(data_path)

            try:
                print(f"[{time.strftime('%H:%M:%S')}] 正在加载训练数据...")
                all_data = []
                for chunk, iters in train_pipeline.reader.read_in_chunks(chunk_size=10000):
                    all_data.extend(chunk)
                    del chunk
                train_pipeline.data_buffer = deque(all_data)
                print(f"[{time.strftime('%H:%M:%S')}] 成功加载 {len(all_data)} 条训练样本")

            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] 数据加载失败: {e}")

            # 执行单轮训练
            loss, entropy = train_pipeline.policy_update()
            print(f"[{time.strftime('%H:%M:%S')}] 训练完成，loss={loss:.3f}, entropy={entropy:.3f}")

            # 保存模型到指定路径
            train_pipeline.handle_interruption()

            print(f"[{time.strftime('%H:%M:%S')}] 模型已保存至 {self.model_path}")

            # 获取最新的模型文件记录
            latest_model_record = self.db_manager.get_latest_model_file(self.user_id)

            # 更新数据库中的模型文件记录
            if latest_model_record:
                # 更新现有记录，增加训练轮次
                new_epochs = latest_model_record['training_epochs'] + 1
                self.db_manager.update_data_file(latest_model_record['id'], training_epochs=new_epochs)
            else:
                # 创建新记录
                self.db_manager.add_model_file(
                    user_id=self.user_id,
                    training_epochs=1,
                    file_path=self.model_path
                )

            # 删除旧模型文件
            old_model_paths = [
                train_pipeline.init_model,
                os.path.basename(train_pipeline.init_model).replace(".pkl",
                                                                    ".onnx") if train_pipeline.init_model else None,
                os.path.basename(train_pipeline.init_model).replace(".pkl",
                                                                    ".trt") if train_pipeline.init_model else None
            ]

            for old_model_path in old_model_paths:
                if old_model_path and os.path.exists(old_model_path) and old_model_path != self.init_model:
                    os.remove(old_model_path)
                    print(f"[{time.strftime('%H:%M:%S')}] 已删除旧模型文件: {old_model_path}")

            return train_pipeline

        except Exception as e:
            print(f"[ERROR] 模型训练失败: {e}")
            return None

    def run_training_cycle(self):
        """运行单次训练周期"""
        try:
            # 合并数据
            data_path = self.merge_data()
            if not data_path:
                print(f"[{time.strftime('%H:%M:%S')}] 数据不够1局，跳过本次训练")
                return False
            try:
                _,data_buffer = DataReader(data_path).read_all()
                if len(data_buffer) < BATCH_SIZE:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] 数据不足一个批次大小({len(data_buffer)} < {BATCH_SIZE})，跳过本次训练")
                    return False
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] 检查数据批次大小时出错: {e}")
                return False

            # 停止现有对弈
            self.stop_selfplay()
            # 训练模型
            train_pipeline = self.train_model(data_path)
            if not train_pipeline:
                print(f"[{time.strftime('%H:%M:%S')}] 模型训练失败，跳过更新")
                # 启动新模型的对弈
                self.start_selfplay()
                self.self_play_only = True # 仅进行数据采集
                return False
            # 启动新模型的对弈
            self.start_selfplay()

            print(f"[{time.strftime('%H:%M:%S')}] 训练周期完成")
            return True

        except Exception as e:
            print(f"[ERROR] 运行训练周期失败: {e}")
            return False


    def start_automated_training(self, workers=4, self_play_only=False):
        """启动自动化训练流程"""
        try:
            self.workers = workers
            self.self_play_only = self_play_only  # 添加此行
            mode = "仅数据采集" if self.self_play_only else "完整训练"
            print(f"[{time.strftime('%H:%M:%S')}] 开始{mode}流程，每 {self.interval} 分钟更新一次")

            # 初始启动自我对弈
            self.start_selfplay()
            self.last_train_time = time.time()

            while self.running:
                # 计算等待时间，确保按间隔执行训练
                elapsed = time.time() - self.last_train_time
                remaining = max(10, self.interval * 60 )

                print(f"[{time.strftime('%H:%M:%S')}] 等待 {remaining:.0f} 秒后进行下一次模型更新")
                time.sleep(remaining)
                if not self.running:
                    break

                # 如果仅进行数据采集，则只重启自我对弈进程
                if self.self_play_only:
                    print(f"[{time.strftime('%H:%M:%S')}] 仅数据采集模式，跳过模型训练")
                    self.stop_selfplay()
                    self.start_selfplay()
                    self.last_train_time = time.time()
                else:
                    # 执行训练周期
                    success = self.run_training_cycle()
                    if success:
                        self.last_train_time = time.time()
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] 训练周期执行失败，将继续尝试下一次更新")

        except KeyboardInterrupt:
            print("\n[INFO] 用户中断，正在保存当前模型...")
            self.stop_all()
        except Exception as e:
            print(f"[ERROR] 自动化训练流程异常终止: {e}")
            self.stop_all()


    def stop_all(self):
        """停止所有进程"""
        self.running = False
        self.stop_selfplay()

        # 清理残留进程
        for p in self.selfplay_processes:
            if p.is_alive():
                p.terminate()
                p.join()
        self.selfplay_processes = []


def main():
    parser = argparse.ArgumentParser(description="自动化强化学习训练系统")
    parser.add_argument("--data-path", type=str, default=os.path.join(DATA_PATH, "collect/data_20250724_140800_iters1606.pkl"),
                        help="初始数据文件路径")
    parser.add_argument("--init-model", type=str, default=os.path.join(MODELS,"trt/current_policy_batch7661_202507241306.trt"),
                        help="初始模型路径（可以是.pkl或.onnx或trt格式）")
    parser.add_argument("--user-id", type=str, default="admin",
                        help="用户ID")
    parser.add_argument("--interval", type=int, default=10,
                        help="模型更新间隔（分钟）")
    parser.add_argument("--workers", type=int, default=2,
                        help="使用的自我对弈进程数")
    parser.add_argument("--use-gpu", action="store_true", default=False,
                        help="是否使用GPU进行推理（默认为False）,GPU只能单进程")
    parser.add_argument("--temp", type=float, default=1.0, help="温度参数")
    parser.add_argument("--cpuct", type=float, default=5.0, help="CPUCT参数")
    # 添加新参数：是否自我迭代模型
    parser.add_argument("--self-play-only", action="store_true", default=False,
                        help="是否仅进行自我对弈数据采集而不训练模型（默认为False，即进行完整训练流程）")
    args = parser.parse_args()

    # 获取系统资源限制
    max_cpu_workers = multiprocessing.cpu_count()
    available_gpus = get_available_gpus()

    # 根据使用模式确定最大workers数
    max_workers = max_cpu_workers

    # 限制workers数量不超过系统资源
    workers = args.workers
    args.workers = min(workers, max_workers)

    if args.workers != workers:
        print(f"[{time.strftime('%H:%M:%S')}] 调整workers数量: {workers} -> {args.workers} (受限于系统资源)")
    if not torch.cuda.is_available() and args.use_gpu:
        args.use_gpu = False
        print(f"[{time.strftime('%H:%M:%S')}] 禁用GPU，因为当前设备不支持GPU")

    # 创建训练管理器
    manager = RLTrainingManager(
        init_model=args.init_model,
        interval_minutes=args.interval,
        use_gpu=args.use_gpu,
        temp=args.temp,
        cpuct=args.cpuct,
        user_id=args.user_id,
        data_path=args.data_path
    )

    try:
        # 启动自动化训练
        manager.start_automated_training(args.workers, self_play_only=args.self_play_only)

    except KeyboardInterrupt:
        print("\n[INFO] 程序终止，正在清理资源...")
    finally:
        manager.stop_all()


if __name__ == "__main__":
    main()
