# model_update.py
import os
import json

dir = os.path.dirname(os.path.abspath(__file__))


def analyze_model_structure(model_path):
    """
    分析模型结构并输出详细信息

    Args:
        model_path: 模型路径
    """
    print("模型详细结构分析:")
    print("=" * 50)

    # 检查主要目录
    directories = ["graph", "am", "ivector", "conf"]
    for directory in directories:
        dir_path = os.path.join(model_path, directory)
        if os.path.exists(dir_path):
            print(f"\n{directory}/ 目录:")
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"  {file} ({size} bytes)")

    # 特别检查graph目录
    graph_path = os.path.join(model_path, "graph")
    if os.path.exists(graph_path):
        print(f"\ngraph/ 目录详细内容:")
        for root, dirs, files in os.walk(graph_path):
            level = root.replace(graph_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                print(f"{subindent}{file} ({size} bytes)")

def create_vocabulary_file(model_path, words):
    """
    创建词汇表文件

    Args:
        model_path: 模型路径
        words: 词汇列表
    """
    vocab_file = os.path.join(model_path, "words.txt")
    try:
        with open(vocab_file, 'w', encoding='utf-8') as f:
            # 写入词汇，每行一个词
            for word in words:
                f.write(f"{word}\n")
        print(f"✓ 成功创建词汇表文件: {vocab_file}")
        print(f"  包含 {len(words)} 个词汇")
        return True
    except Exception as e:
        print(f"✗ 创建词汇表文件失败: {e}")
        return False

def suggest_vocabulary_for_chess():
    """
    为象棋游戏场景建议词汇表
    """
    print("\n为象棋游戏场景推荐的词汇表:")
    print("=" * 50)

    # 象棋游戏相关词汇
    chess_words = [
        # 唤醒词
        "小助手", "你好",

        # 游戏控制
        "开始", "重新开始", "结束", "暂停", "继续",
        "布局","重启","停止","复位","启动",
        "悔棋", "认输", "投降", "退出", "帮助",

        # 棋子名称
        "车", "马", "炮", "象", "相", "士", "仕",
        "将", "帅", "兵", "卒", "车", "马", "炮",

        # 棋盘坐标 (列)
        "一", "二", "三", "四", "五", "六", "七", "八", "九",

        # 棋盘坐标 (行)
        "前",  "中", "上", "下", "左", "右",

        # 移动指令
        "进", "退", "平", "走", "移动", "跳", "飞",
        "打", "吃", "捉", "献", "兑", "拦", "堵",

        # 难度设置
        "简单", "中等", "困难", "初级", "中级", "高级",
        "难度", "级别", "等级",

        # 游戏状态
        "思考中", "等待", "轮到", "回合", "时间",
        "超时", "违规", "重新走", "无效",

        # 语音反馈
        "我在", "好的", "明白", "确认", "取消",
        "是的", "不是", "不可以",

        # 常见干扰词（容易被误识别的词语）
        # 日常用语
        "什么", "怎么", "为什么", "可以", "不能",
        "现在", "这里", "那里", "这个", "那个",
        "一下", "一下下", "一点点", "很多", "一些",

        # 数字
        "零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
        "百", "千", "万", "亿",

        # 量词
        "个", "只", "台", "套", "件", "块", "片", "张", "条", "根",

        # 方位词
        "上", "下", "左", "右", "前", "后", "里", "外", "内", "中",

        # 动词
        "是", "有", "在", "做", "看", "听", "说", "想", "要", "会",

        # 形容词
         "坏", "大", "小", "多", "少", "快", "慢", "高", "低",

        # 语气词
        "啊", "啦", "吧",  "呢", "了", "的",

        # 常见名词
        "东西", "地方", "时候", "时间", "问题", "情况", "方法", "工作",
    ]

    #
    # for i, word in enumerate(chess_words, 1):
    #     print(f"{i:2d}. {word}")

    print(f"\n总计: {len(chess_words)} 个词汇")
    return chess_words

def check_model_compatibility(model_path):
    """
    检查模型兼容性

    Args:
        model_path: 模型路径
    """
    print("\n模型兼容性检查:")
    print("=" * 30)

    # 检查可能的文件名
    fst_files = []
    graph_path = os.path.join(model_path, "graph")
    if os.path.exists(graph_path):
        for file in os.listdir(graph_path):
            if file.endswith(".fst"):
                fst_files.append(file)

    required_files = {
        "am/final.mdl": "声学模型",
        "conf/mfcc.conf": "MFCC配置"
    }

    # 添加找到的FST文件
    for fst_file in fst_files:
        required_files[f"graph/{fst_file}"] = f"FST文件 ({fst_file})"

    missing_files = []
    found_files = []
    for file, description in required_files.items():
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            found_files.append((description, os.path.getsize(file_path)))
        else:
            missing_files.append((file, description))

    for description, size in found_files:
        print(f"✓ {description} 存在 ({size} bytes)")

    for file, description in missing_files:
        print(f"✗ {description} 缺失")

    if not missing_files:
        print("\n✓ 模型结构完整，可以正常使用")
    else:
        print(f"\n✗ 缺失 {len(missing_files)} 个关键文件")

    return len(missing_files) == 0

def create_model_config(model_path):
    """
    创建模型配置文件

    Args:
        model_path: 模型路径
    """
    config_file = os.path.join(model_path, "model_config.json")
    try:
        config = {
            "sample_rate": 16000,
            "feature_type": "mfcc",
            "latency": 200,
            "vocabulary_file": "words.txt"
        }

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        print(f"✓ 成功创建模型配置文件: {config_file}")
        return True
    except Exception as e:
        print(f"✗ 创建模型配置文件失败: {e}")
        return False

def main():
    model_path = os.path.join(dir, "speech_model")

    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在 {model_path}")
        return

    print("Vosk模型分析工具")
    print("=" * 50)

    # 分析模型结构
    analyze_model_structure(model_path)

    # 检查兼容性
    is_compatible = check_model_compatibility(model_path)

    # 为象棋游戏场景建议词汇
    vocab_words = suggest_vocabulary_for_chess()

    print("\n" + "=" * 50)
    print("创建词汇表和配置文件:")

    # 创建词汇表文件
    if create_vocabulary_file(model_path, vocab_words):
        # 创建模型配置文件
        create_model_config(model_path)

        print("\n创建完成！")
        print("\n使用说明:")
        print("1. 词汇表文件 (words.txt) 可用于限制识别范围，提高准确率")
        print("2. 在 speech_service.py 中可以通过以下方式使用:")
        print("   recognizer = KaldiRecognizer(model, sample_rate, json.dumps(vocab_words, ensure_ascii=False))")
        print("3. 词汇表将帮助模型专注于象棋游戏相关词汇，提高识别准确率")

if __name__ == "__main__":
    main()
