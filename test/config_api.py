# config_api.py
import requests
import json

# API基础URL
BASE_URL = "http://localhost:8080"


def test_config_table_crud():
    """测试配置表的增删改查功能"""
    print("=== 配置表增删改查测试 ===")

    # 1. 创建配置表
    print("\n1. 创建配置表")
    url = f"{BASE_URL}/config-tables/"
    data = {
        "table_name": "test_config_table",
        "description": "测试配置表"
    }
    response = requests.post(url, json=data)
    print("创建配置表响应:", response.json())

    # 2. 获取所有配置表
    print("\n2. 获取所有配置表")
    url = f"{BASE_URL}/config-tables/"
    response = requests.get(url)
    print("获取所有配置表响应:", response.json())

    # 3. 获取特定配置表
    print("\n3. 获取特定配置表")
    url = f"{BASE_URL}/config-tables/test_config_table"
    response = requests.get(url)
    print("获取特定配置表响应:", response.json())

    # 4. 更新配置表
    print("\n4. 更新配置表")
    url = f"{BASE_URL}/config-tables/test_config_table"
    data = {
        "description": "更新后的测试配置表"
    }
    response = requests.put(url, json=data)
    print("更新配置表响应:", response.json())

    # 5. 验证更新
    print("\n5. 验证配置表更新")
    url = f"{BASE_URL}/config-tables/test_config_table"
    response = requests.get(url)
    print("验证更新响应:", response.json())


def test_config_field_crud():
    """测试配置字段的增删改查功能"""
    print("\n\n=== 配置字段增删改查测试 ===")

    # 1. 添加字段定义 - 整数类型
    print("\n1. 添加整数类型字段")
    url = f"{BASE_URL}/config-tables/test_config_table/fields"
    data = {
        "field_name": "max_users",
        "field_type": "int",
        "is_required": True,
        "default_value": "100",
        "description": "最大用户数"
    }
    response = requests.post(url, json=data)
    print("添加整数字段响应:", response.json())

    # 2. 添加字段定义 - 布尔类型
    print("\n2. 添加布尔类型字段")
    url = f"{BASE_URL}/config-tables/test_config_table/fields"
    data = {
        "field_name": "enable_feature",
        "field_type": "boolean",
        "is_required": False,
        "default_value": "false",
        "description": "是否启用功能"
    }
    response = requests.post(url, json=data)
    print("添加布尔字段响应:", response.json())

    # 3. 添加字段定义 - 字符串类型
    print("\n3. 添加字符串类型字段")
    url = f"{BASE_URL}/config-tables/test_config_table/fields"
    data = {
        "field_name": "data_path",
        "field_type": "string",
        "is_required": True,
        "default_value": "/default/path",
        "description": "数据存储路径"
    }
    response = requests.post(url, json=data)
    print("添加字符串字段响应:", response.json())

    # 4. 获取配置表的所有字段
    print("\n4. 获取所有字段定义")
    url = f"{BASE_URL}/config-tables/test_config_table/fields"
    response = requests.get(url)
    print("获取字段定义响应:", response.json())

    # 5. 更新字段定义
    print("\n5. 同时更新字段名和其他属性")
    url = f"{BASE_URL}/config-tables/test_config_table/fields/test_field"
    data = {
        "new_field_name": "updated_field",
        "field_type": "integer",
        "is_required": True,
        "default_value": "42",
        "description": "更新后的字段"
    }
    response = requests.patch(url, json=data)
    print("综合更新响应:", response.json())

    # 6. 验证字段更新
    print("\n6. 验证字段更新")
    url = f"{BASE_URL}/config-tables/test_config_table/fields"
    response = requests.get(url)
    print("验证字段更新响应:", response.json())


def test_config_data_crud():
    """测试配置数据的增删改查功能"""
    print("\n\n=== 配置数据增删改查测试 ===")

    # 1. 设置配置数据 - 整数类型
    print("\n1. 设置整数类型配置数据")
    url = f"{BASE_URL}/config-tables/test_config_table/data"
    data = {
        "config_key": "max_users",
        "config_value": 150,
        "description": "设置最大用户数为150"
    }
    response = requests.post(url, json=data)
    print("设置整数配置数据响应:", response.json())

    # 2. 设置配置数据 - 布尔类型
    print("\n2. 设置布尔类型配置数据")
    url = f"{BASE_URL}/config-tables/test_config_table/data"
    data = {
        "config_key": "enable_feature",
        "config_value": True,
        "description": "启用功能"
    }
    response = requests.post(url, json=data)
    print("设置布尔配置数据响应:", response.json())

    # 3. 设置配置数据 - 字符串类型
    print("\n3. 设置字符串类型配置数据")
    url = f"{BASE_URL}/config-tables/test_config_table/data"
    data = {
        "config_key": "data_path",
        "config_value": "/custom/path/data",
        "description": "自定义数据路径"
    }
    response = requests.post(url, json=data)
    print("设置字符串配置数据响应:", response.json())

    # 4. 设置配置数据 - 复杂对象类型
    print("\n4. 设置复杂对象配置数据")
    url = f"{BASE_URL}/config-tables/test_config_table/data"
    data = {
        "config_key": "database_config",
        "config_value": {
            "host": "localhost",
            "port": 5432,
            "username": "admin",
            "ssl_enabled": True
        },
        "description": "数据库配置"
    }
    response = requests.post(url, json=data)
    print("设置复杂对象配置数据响应:", response.json())

    # 5. 获取配置表的所有数据
    print("\n5. 获取所有配置数据")
    url = f"{BASE_URL}/config-tables/test_config_table/data"
    response = requests.get(url)
    print("获取配置数据响应:", response.json())

    # 6. 更新配置数据
    print("\n6. 更新配置数据")
    url = f"{BASE_URL}/config-tables/test_config_table/data"
    data = {
        "config_key": "max_users",
        "config_value": 300,
        "description": "更新最大用户数为300"
    }
    response = requests.post(url, json=data)
    print("更新配置数据响应:", response.json())

    # 7. 验证数据更新
    print("\n7. 验证配置数据更新")
    url = f"{BASE_URL}/config-tables/test_config_table/data"
    response = requests.get(url)
    print("验证数据更新响应:", response.json())


def test_config_data_delete():
    """测试配置数据删除功能"""
    print("\n\n=== 配置数据删除测试 ===")

    # 1. 删除配置数据
    print("\n1. 删除配置数据")
    url = f"{BASE_URL}/config-tables/test_config_table/data/max_users"
    response = requests.delete(url)
    print("删除配置数据响应:", response.json())

    # 2. 验证数据删除
    print("\n2. 验证数据删除")
    url = f"{BASE_URL}/config-tables/test_config_table/data"
    response = requests.get(url)
    print("验证数据删除响应:", response.json())


def test_config_field_delete():
    """测试配置字段删除功能"""
    print("\n\n=== 配置字段删除测试 ===")

    # 1. 删除字段定义
    print("\n1. 删除字段定义")
    url = f"{BASE_URL}/config-tables/test_config_table/fields/max_users"
    response = requests.delete(url)
    print("删除字段响应:", response.json())

    # 2. 验证字段删除
    print("\n2. 验证字段删除")
    url = f"{BASE_URL}/config-tables/test_config_table/fields"
    response = requests.get(url)
    print("验证字段删除响应:", response.json())


def test_config_table_delete():
    """测试配置表删除功能"""
    print("\n\n=== 配置表删除测试 ===")

    # 1. 删除配置表
    print("\n1. 删除配置表")
    url = f"{BASE_URL}/config-tables/test_config_table"
    response = requests.delete(url)
    print("删除配置表响应:", response.json())

    # 2. 验证表删除
    print("\n2. 验证配置表删除")
    url = f"{BASE_URL}/config-tables/"
    response = requests.get(url)
    print("验证表删除响应:", response.json())


def comprehensive_test():
    """综合测试所有功能"""
    print("开始综合测试配置表API...")

    # 执行所有测试
    test_config_table_crud()
    test_config_field_crud()
    test_config_data_crud()
    test_config_data_delete()
    test_config_field_delete()
    test_config_table_delete()

    print("\n\n所有测试完成!")


if __name__ == "__main__":
    comprehensive_test()
