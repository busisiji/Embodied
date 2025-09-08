# api/utils/response_utils.py
from typing import Any, List, Dict, Union

def format_response_data(data: Any, remove_sensitive_func=None, token: str = None) -> Dict:
    """
    统一格式化响应数据的函数

    Args:
        data: 需要格式化的数据
        remove_sensitive_func: 可选的移除敏感字段函数
        token: 可选的认证令牌

    Returns:
        格式化后的数据字典
    """
    formatted_data = []

    # 处理列表数据
    if isinstance(data, list):
        items = data
    else:
        # 如果不是列表，将其包装成列表
        items = [data] if data is not None else []

    for item in items:
        # 提取数据
        if hasattr(item, '__data__'):
            item_data = item.__data__
        elif isinstance(item, dict):
            item_data = item
        elif hasattr(item, '__dict__'):
            item_data = item.__dict__
        else:
            item_data = item

        # 移除敏感字段（如果提供了相应的函数）
        if remove_sensitive_func:
            safe_item_data = remove_sensitive_func(item_data)
        else:
            safe_item_data = item_data

        # 如果提供了token，则添加到返回数据中
        if token is not None:
            if isinstance(safe_item_data, dict):
                safe_item_data["token"] = token
            else:
                # 如果safe_item_data不是字典，创建一个新的字典包含原始数据和token
                safe_item_data = {"data": safe_item_data, "token": token}

        formatted_data.append(safe_item_data)

    # 构建返回结果
    result = {
        "list": formatted_data
    }

    # 如果只有一个项目，直接返回该项目
    if len(items) <= 1 :
        # 合并单个项目的数据到结果中
        single_item = formatted_data[0]
        if isinstance(single_item, dict):
            result = single_item
        else:
            result["data"] = single_item

    return result
