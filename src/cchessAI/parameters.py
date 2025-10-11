# PUCT探索常数 score = Q(s,a) + C_PUCT * P(s,a) * √(N(s)) / (1 + N(s,a))
# Q(s,a)：状态s下采取行动a的预期价值
# P(s,a)：策略网络给出的先验概率
# N(s)：父节点的访问次数
# N(s,a)：当前节点的访问次数
# 值较小时，算法会更倾向于探索访问次数少的节点
# 值较大时，算法会更倾向于利用已知的高价值路径
import os
import sys

C_PUCT = 5
# Dirichlet噪声的ε参数，表示添加噪声的比例或强度
EPS = 0.25
# Dirichlet噪声的α参数，表示添加噪声的分布的形状
ALPHA = 0.2
# 每次移动的模拟次数
PLAYOUT = 1600
# 训练数据批次大小
BATCH_SIZE = 1024
# 每次训练轮数
EPOCHS = 10
# kl散度控制
KL_TARG = 0.02
# 保存模型的频率
CHECK_FREQ = 1
# 最大经验池大小
BUFFER_SIZE = 100000
# 输出到控制台的日志等级 ({"DEBUG": 1, "INFO": 2, "WARNING": 3, "ERROR": 4, "CRITICAL": 5})
LOG_LEVEL = 1

# 是否动态设置模拟次数
IS_DYNAMIC_PLAYOUT = True
# 是否显示调试输出
IS_DEBUG = False
# 是否自动使用最新的模型
IS_UPNEW = False
IS_WINDOW = True
# 数据集是否执行镜像翻转
SHOULD_FLIP = False

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将项目根目录添加到系统路径中
sys.path.insert(0, current_dir)
# 数据目录（训练过程中产生的所有数据文件）
DATA_DIR = os.path.join(current_dir, "datas")
# 模型目录（训练产生与加载的模型文件）
MODEL_DIR = os.path.join(current_dir, "models")
# 模型目录
MODELS = os.path.join(MODEL_DIR, "admin")
# 模型地址
MODEL_PATH = os.path.join(MODELS, "onnx/current_policy_7100.onnx")
# 训练数据容器目录
DATAS = os.path.join(DATA_DIR, "admin")
# 训练数据容器地址
DATA_SELFPLAY = os.path.join(DATAS, "collect")
DATA_BUFFER_PATH = os.path.join(DATAS, "collect/data_20250723_130251_iters1514.pkl")