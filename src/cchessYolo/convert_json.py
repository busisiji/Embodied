import os
import json
# 定义标签映射关系
label_mappings = [{
    "12 0.620286 0.503311 0.097182 0.169762": "R",
    "9 0.405262 0.337073 0.097474 0.171797": "B",
    "5 0.303951 0.509705 0.097534 0.170775": "A",
    "3 0.522093 0.282602 0.095116 0.169368": "a",
    "8 0.181226 0.307285 0.093350 0.169181": "k",
    "8 0.721726 0.510625 0.097091 0.169764": "N",
    "12 0.403985 0.511040 0.097159 0.169386": "P",
    "13 0.618844 0.319058 0.095753 0.169466": "c",
    "11 0.719624 0.335990 0.096046 0.167380": "K",
    "12 0.514234 0.489558 0.096271 0.168697": "b",
    "7 0.411458 0.125303 0.095479 0.169119": "n",
    "2 0.280485 0.338217 0.098447 0.170168": "C",
    "12 0.820694 0.319392 0.094998 0.169518": "r",
    "6 0.191606 0.489253 0.091838 0.172703": "p"
},{
    "5 0.528906 0.386111 0.095312 0.166667": "R",
    "2 0.105859 0.662500 0.092969 0.158333": "B",
    "8 0.092969 0.338889 0.098437 0.172222": "A",
    "0 0.512891 0.647222 0.092969 0.169444": "a",
    "4 0.853906 0.352083 0.098437 0.179167": "k",
    "6 0.216016 0.822222 0.096094 0.175000": "N",
    "11 0.520703 0.211111 0.092969 0.166667": "P",
    "3 0.653125 0.398611 0.093750 0.166667": "c",
    "7 0.394922 0.511806 0.099219 0.168056": "K",
    "10 0.689063 0.220139 0.100000 0.173611": "b",
    "9 0.748047 0.662500 0.096094 0.183333": "n",
    "13 0.389844 0.303472 0.098437 0.173611": "C",
    "12 0.762956 0.863889 0.091276 0.140278": "r",
    "1 0.639453 0.794444 0.094531 0.177778": "p"
}]

def process_json_files(directory):
    """
    遍历目录中的所有JSON文件，并替换标签
    """
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)

            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 处理shapes中的标签
            if 'shapes' in data:
                for shape in data['shapes']:
                    original_label = shape.get('label', '')
                    for label_mapping in label_mappings:
                        if original_label in label_mapping:
                            shape['label'] = label_mapping[original_label]

            # 将修改后的数据写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
def convert_json_format(input_dir, output_dir):
    """
    遍历目录中的所有json文件，将格式转换为指定格式
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有json文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # 读取原始json文件
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 转换格式
            converted_data = convert_single_file(data)

            # 写入新格式的json文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, ensure_ascii=False, indent=2)

def convert_single_file(data):
    """
    转换单个文件的格式
    """
    labels = []

    # 遍历所有shapes
    for shape in data.get('shapes', []):
        # 获取label和points
        name = shape.get('label', '')
        points = shape.get('points', [])

        if len(points) >= 2:
            # 提取坐标值
            # points格式: [[x1, y1], [x2, y2], ...]
            # 我们需要的是左上角和右下角坐标
            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]

            x1 = min(x_coords)
            y1 = min(y_coords)
            x2 = max(x_coords)
            y2 = max(y_coords)

            # 添加到labels列表
            labels.append({
                'name': name,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })

    return {'labels': labels}

# 使用示例
if __name__ == '__main__':
    # 指定输入和输出目录
    input_directory = r'E:\code\Embodied\src\cchessYolo\Images'
    output_directory = r'E:\code\Embodied\src\cchessYolo\Converted'

    # 执行转换
    convert_json_format(input_directory, output_directory)
    print("转换完成！")

    # # 使用示例
    # process_json_files('E:\code\Embodied\src\cchessYolo\Images')