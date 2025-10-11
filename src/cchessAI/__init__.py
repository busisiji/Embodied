import sys
import os
from pathlib import Path

# 获取项目根目录
project_root = Path(__file__).resolve().parent

# 将项目根目录添加到 Python 路径中
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
