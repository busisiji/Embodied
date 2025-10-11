# auto_rl_loop.py
import os
import subprocess
import time
import argparse
from src.cchessAI.parameters import DATA_DIR, MODEL_DIR

def run_collect(model_path=None, show_board=False):
    """运行数据收集阶段"""
    cmd = ["python", "collect.py"]
    if show_board:
        cmd.append("--show")
    if model_path:
        cmd.extend(["--model", model_path])

    print("开始数据收集...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"数据收集失败: {result.stderr}")
        return False
    print("数据收集完成")
    return True

def run_train(init_model=None, data_format="h5"):
    """运行模型训练阶段"""
    cmd = ["python", "train.py", "--data-format", data_format]
    if init_model:
        cmd.extend(["--init-model", init_model])

    print("开始模型训练...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"模型训练失败: {result.stderr}")
        return False
    print("模型训练完成")
    return True

def get_latest_model():
    """获取最新的模型文件"""
    model_path = os.path.join(MODEL_DIR, "current_policy.pkl")
    if os.path.exists(model_path):
        return model_path
    return None

def auto_rl_loop(iterations=10, games_per_iteration=10, show_board=False):
    """
    自动化自我强化学习循环

    Args:
        iterations: 循环迭代次数
        games_per_iteration: 每次迭代收集的游戏数量
        show_board: 是否显示棋盘
    """
    print(f"开始自动化自我强化学习循环，共 {iterations} 次迭代")

    for i in range(iterations):
        print(f"\n=== 第 {i+1} 次迭代 ===")

        # 获取当前最新模型
        current_model = get_latest_model()
        print(f"使用模型: {current_model if current_model else '随机初始化'}")

        # 数据收集阶段
        collect_success = run_collect(
            model_path=current_model,
            show_board=show_board
        )

        if not collect_success:
            print("数据收集失败，跳过本次迭代")
            continue

        # 模型训练阶段
        train_success = run_train(init_model=current_model)

        if not train_success:
            print("模型训练失败，终止循环")
            break

        print(f"第 {i+1} 次迭代完成\n")
        time.sleep(1)  # 短暂休息

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动化自我强化学习循环")
    parser.add_argument("--iterations", type=int, default=10, help="循环迭代次数")
    parser.add_argument("--games-per-iteration", type=int, default=10, help="每次迭代收集的游戏数量")
    parser.add_argument("--show-board", action="store_true", help="是否显示棋盘")

    args = parser.parse_args()
    auto_rl_loop(
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        show_board=args.show_board
    )
