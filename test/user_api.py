import requests
import json

# API基础URL
BASE_URL = "http://localhost:8080"

def create_user():
    """测试创建用户"""
    url = f"{BASE_URL}/users/"
    # 使用 JSON 格式发送数据
    data = {
        "user_id": "test011",
        "name": "张三",
        "permission": "student"
    }
    response = requests.post(url, json=data)  # 使用 json 参数而不是 params
    print("创建用户响应:", response.json())
    return response.json()


def get_all_users(limit=None, offset=None):
    """测试获取所有用户，支持分页"""
    url = f"{BASE_URL}/users/"
    params = {}
    if limit is not None:
        params['limit'] = limit
    if offset is not None:
        params['offset'] = offset

    response = requests.get(url, params=params)  # 添加 params 参数
    print("获取所有用户响应:", response.json())
    return response.json()
def get_user_by_id(user_id):
    """测试根据ID获取用户"""
    url = f"{BASE_URL}/users/{user_id}"
    response = requests.get(url)
    print(f"获取用户 {user_id} 响应:", response.json())
    return response.json()

def update_user(user_id):
    """测试更新用户"""
    url = f"{BASE_URL}/users/{user_id}"
    # 使用 JSON 格式发送数据
    data = {
        "name": "李四",
        "permission": "teacher"
    }
    response = requests.put(url, json=data)  # 使用 json 参数而不是 params
    print(f"更新用户 {user_id} 响应:", response.json())
    return response.json()

def delete_user(user_id):
    """测试删除用户"""
    url = f"{BASE_URL}/users/{user_id}"
    response = requests.delete(url)
    print(f"删除用户 {user_id} 响应:", response.json())
    return response.json()

if __name__ == "__main__":
    print("开始测试用户API...")
    # # 创建用户
    created_user = create_user()

    # 获取所有用户
    get_all_users(limit=100)

    # 获取特定用户
    if isinstance(created_user, dict) and '__data__' in created_user:
        user_id = created_user['__data__'].get('user_id', 'test001')
    elif isinstance(created_user, dict):
        user_id = created_user.get('user_id', 'test001')
    else:
        user_id = 'test001'
    print("获取用户", user_id)
    get_user_by_id(user_id)
    #
    # 更新用户
    update_user(user_id)

    # 验证更新
    get_user_by_id(user_id)

    # 删除用户
    delete_user(user_id)

    # 验证删除
    get_all_users()

    print("API测试完成")
