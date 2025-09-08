# PUCT探索常数 score = Q(s,a) + C_PUCT * P(s,a) * √(N(s)) / (1 + N(s,a))
# Q(s,a)：状态s下采取行动a的预期价值
# P(s,a)：策略网络给出的先验概率
# N(s)：父节点的访问次数
# N(s,a)：当前节点的访问次数

# 蒙特卡洛树搜索（MCTS）过程
# C_PUCT
# 较小（如 1~3）更加偏向“利用”，优先选择已有高价值路径。模型已经训练成熟、需要稳定决策的阶段。
# 较大（如 5~10）更加偏向“探索”，优先选择新的未知路径。模型尚未训练或正在训练过程中，需要探索新的可能性。
# C_PUCT 值越大，AI 的探索能力越强，但可能也会导致 AI 探索的过深，从而导致模型过拟合。

# 策略网络输出的采样阶段
# TEMP
# TEMP ≈ 0 (如 1e-3 或更小)：
# AI 几乎完全按照策略网络输出的概率分布进行选择，只选取最高概率的动作。
# 适用于最终决策阶段或模型已经训练成熟时，强调利用。
# TEMP = 1：
# 按照原始策略网络输出的概率分布进行采样，保持一定平衡的探索与利用。
# 常用于训练过程中或需要合理探索的场景。
# TEMP > 1 (如 1.5, 2 等)：
# 提高动作选择的随机性，促使 AI 探索更多可能性。
# 适用于早期探索阶段或需要打破局部最优策略的情况。
# 极端高值 (如 TEMP > 10)：
# 动作选择变得非常随机，AI 几乎完全依赖探索。
# 可能用于特殊调试或环境复杂度极高的情况
import os.path
import sys

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将项目根目录添加到系统路径中
sys.path.insert(0, current_dir)

IS_WINDOW = True
# 数据集是否执行镜像翻转
SHOULD_FLIP = False
# 影响算法在“探索”和“利用”之间的权衡。
C_PUCT = 5
# Dirichlet噪声的ε参数，表示添加噪声的比例或强度
EPS = 0.25
# Dirichlet噪声的α参数，表示添加噪声的分布的形状
ALPHA = 0.2
# 每次移动的模拟次数
PLAYOUT = 1200 # 400-1200
# 是否动态设置模拟次数
IS_DYNAMIC_PLAYOUT = True
# 是否显示调试输出
IS_DEBUG = False
# 是否自动使用最新的模型
IS_UPNEW = False
# 搜索温度
TEMP = 1e-3 # 温度参数，它控制着 AI 在决策时的探索程度。当温度值较高(1)时，AI 的选择更倾向于随机；而当温度较低(1e-3)时，AI 更倾向于选择高概率的动作。
# 最大经验池大小
BUFFER_SIZE = 100000
# 保存数据的频率
DATA_CHECK_FREQ = 1
# 训练数据批次大小
BATCH_SIZE = 512
# 训练轮数
EPOCHS = 10
# 训练更新间隔时间
UPDATE_INTERVAL = 0.01
# kl散度控制
KL_TARG = 0.02
# 训练次数
GAME_BATCH_NUM = 300
# 保存模型的频率
CHECK_FREQ = 10

MODEL_USER_PATH = os.path.join(current_dir, "models")
DATA_USER_PATH = os.path.join(current_dir, "data")

# 模型目录
MODELS = os.path.join(current_dir, "models/admin")
# 模型地址
MODEL_PATH = os.path.join(MODELS, "onnx/current_policy_7100.onnx")
# 训练数据容器目录
DATA_PATH = os.path.join(current_dir, "data/admin")
# 训练数据容器地址
DATA_SELFPLAY = os.path.join(DATA_PATH, "collect")
DATA_BUFFER_PATH = os.path.join(DATA_PATH, "collect/data_20250723_130251_iters1514.pkl")