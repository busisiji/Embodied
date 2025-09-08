import os.path

from src.cchessAI.core.net import PolicyValueNet
from src.cchessAI.parameters import MODELS

if __name__ == '__main__':
    policy_value_net = PolicyValueNet(model_file=os.path.join(MODELS, 'current_policy_batch7100_2025-07-14_14-41-02.pkl'), use_gpu=True)
    policy_value_net.export_to_onnx(os.path.join(MODELS, 'onnx/current_policy_7100.onnx'))