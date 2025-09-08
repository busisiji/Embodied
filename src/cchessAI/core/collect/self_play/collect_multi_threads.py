# src/cchessAI/collect/self_play/collect_multi_threads.py
# 多进程采集数据
import os
import sys
from pathlib import Path

from api.models.database_model import DatabaseManager

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（cchess）
project_root = os.path.dirname((os.path.dirname(current_dir)))

# 如果不在 PYTHONPATH 中，则加入
if project_root not in sys.path:
    print(f"[INFO] 正在加入 {project_root}...")
    sys.path.insert(0, project_root)
import pickle
import time
import argparse
import torch
from multiprocessing import Process, Manager

from src.cchessAI.core.collect.self_play.collect_multi_thread import CollectPipeline


# 获取可用 GPU 列表
def get_available_gpus():
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    else:
        return []


def worker(save_interval, gpu_id, pid, is_shown=False,init_model=None,data_path=None, temp=1.0, cpuct=5.0, game_count=None):
    """
    子进程执行的任务函数：创建 CollectPipeline 实例并收集数据，通过共享列表传出。
    参数：
        save_interval: 每多少局保存一次数据
        gpu_id: 使用的 GPU 编号（None 表示使用 CPU ）
        pid: 当前子进程 PID
        is_shown: 是否显示棋盘对弈过程
        temp: 温度参数
        cpuct: UCT参数
        game_count: 总游戏局数限制（None表示无限制）
    """
    print(f"[{time.strftime('%H:%M:%S')}][PID={pid}] 子进程启动，使用 {'CPU' if gpu_id is None else f'GPU:{gpu_id}'}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) if gpu_id is not None else "-1"

    # 创建CollectPipeline实例并传入数据队列
    pipeline = CollectPipeline(pid=pid, port=pid,init_model=init_model,data_path=data_path,gpu_id=gpu_id, temp=temp, c_puct=cpuct)

    try:
        games_played = 0
        while game_count is None or games_played < game_count:
            # 计算本轮要玩的游戏数量
            batch_size = min(save_interval, game_count - games_played) if game_count is not None else save_interval
            iters = pipeline.collect_data(n_games=batch_size, is_shown=is_shown)
            games_played += batch_size
            print(f"[{time.strftime('%H:%M:%S')}][PID={pid}] 完成 {batch_size} 局，总完成局数: {games_played}/{game_count if game_count is not None else '∞'}")

            # 如果达到游戏数量限制，则退出
            if game_count is not None and games_played >= game_count:
                print(f"[{time.strftime('%H:%M:%S')}][PID={pid}] 已达到指定游戏数量 {game_count}，子进程退出")
                break
    except KeyboardInterrupt:
        print(f"[{time.strftime('%H:%M:%S')}][PID={pid}] 子进程中止，提交完成数据")



# 修改 main 函数中的参数解析部分
def main():
    parser = argparse.ArgumentParser(description="多进程采集中国象棋自对弈数据")
    parser.add_argument("--save-interval", type=int, default=1, help="每多少局保存一次数据")
    parser.add_argument("--init-model", type=str, default=None,
                        help="初始模型路径（可以是.pkl或.onnx或trt格式）")
    parser.add_argument("--data-path", type=str, default=None,
                        help="初始数据文件路径")
    parser.add_argument("--user-id", type=str, default="admin",
                        help="用户ID")
    parser.add_argument("--workers", type=int, default=int(os.cpu_count() * 0.75), help="使用的进程数")
    parser.add_argument("--show", action="store_true", default=False, help="是否显示棋盘对弈过程")
    parser.add_argument("--use-gpu", action="store_true", default=True, help="是否使用 GPU 进行推理（默认为 True）")
    parser.add_argument("--temp", type=float, default=1.0, help="温度参数")
    parser.add_argument("--cpuct", type=float, default=5.0, help="UCT探索参数")
    parser.add_argument("--game-count", type=int, default=100, help="总游戏局数")
    args = parser.parse_args()

    # 处理 init_model 参数，将字符串 "None" 转换为真正的 None
    if args.init_model == "None":
        args.init_model = None

    # 处理 data_path 参数，将字符串 "None" 转换为真正的 None
    if args.data_path == "None":
        args.data_path = None

    print(f"使用 {args.workers} 个进程进行数据采集，总游戏局数: {args.game_count}")

    manager = Manager()
    db_manager = DatabaseManager()

    # 设置环境变量控制线程数（适用于 TensorFlow/PyTorch）
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # 获取可用 GPU
    available_gpus = get_available_gpus()
    num_gpus = len(available_gpus)
    print(f"检测到可用 GPU 数量: {num_gpus}")

    # 起始端口
    base_port = 8000

    # 计算每个进程的游戏局数
    games_per_worker = args.game_count // args.workers
    remaining_games = args.game_count % args.workers

    # 启动多个 worker 进程
    workers = []
    for i in range(args.workers):
        # 分配游戏局数（第一个进程多分配剩余的游戏）
        worker_game_count = games_per_worker + (1 if i == 0 and remaining_games > 0 else 0)

        if args.use_gpu and num_gpus > 0:
            gpu_id = available_gpus[i % num_gpus]
        else:
            gpu_id = None

        port = base_port + i  # 每个进程分配不同端口

        p = Process(
            target=worker,
            args=(args.save_interval, gpu_id, port, args.show, args.init_model, args.data_path, args.temp, args.cpuct, worker_game_count)
        )
        p.start()
        workers.append(p)

    try:
        for p in workers:
            p.join()
    except KeyboardInterrupt:
        print("主进程收到中断信号，等待所有子进程结束...")

    print(f"[{time.strftime('%H:%M:%S')}] 所有子进程已完成，正在合并并保存数据...")

    # 收集所有子进程的数据
    merged_data = []
    iters = 0
    data_path_dir = Path(args.data_path).parent

    # 遍历所有子文件并读取数据
    for i in range(args.workers):
        pid = base_port + i
        sub_data_path = os.path.join(data_path_dir, f"data_{pid}.pkl")
        if os.path.exists(sub_data_path):
            try:
                with open(sub_data_path, "rb") as f:
                    while True:
                        try:
                            item = pickle.load(f)
                            merged_data.extend(item.get("data_buffer", []))
                            iters += item.get("iters", 0)
                        except EOFError:
                            break
                print(f"[{time.strftime('%H:%M:%S')}][PID={pid}] 已读取子文件数据，当前数据量: {len(merged_data)},局数{iters}")
                os.remove(sub_data_path)  # 删除子文件
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}][PID={pid}] 读取子文件失败：{e}")

    # 获取最新的数据文件记录
    latest_data_record = db_manager.get_latest_data_file(args.user_id)

    # 如果存在历史数据文件，则加载并合并
    if latest_data_record and os.path.exists(latest_data_record['file_path']):
        try:
            with open(latest_data_record['file_path'], "rb") as f:
                data = pickle.load(f)
                loaded_data = data.get("data_buffer", [])
                merged_data.extend(loaded_data)
                iters += data.get("iters", 0)
                print(f"[{time.strftime('%H:%M:%S')}] 已加载历史数据，共 {len(loaded_data)} 条样本,局数{iters}")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 加载旧数据失败：{e}")

    # 直接保存到指定的数据文件路径，不重命名
    data_path = args.data_path
    with open(data_path, "wb") as f:
        pickle.dump({
            "data_buffer": merged_data,
            "iters": iters
        }, f)

    # 更新数据库中的数据文件记录
    if latest_data_record:
        # 更新现有记录
        db_manager.update_data_file(latest_data_record['id'], game_count=iters, data_length=len(merged_data))
    else:
        # 创建新记录
        db_manager.add_data_file(
            user_id=args.user_id,
            game_count=iters,
            data_length=len(merged_data),
            file_path=data_path
        )

    print(f"[{time.strftime('%H:%M:%S')}] 最终数据已保存至 {data_path},共 {len(merged_data)} 条样本,局数{iters}")



if __name__ == "__main__":
    import torch
    main()
