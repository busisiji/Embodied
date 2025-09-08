# tests/test_model_files_api.py
import requests
import json
import os
from datetime import datetime

# 基础URL配置
BASE_URL = "http://localhost:8000"  # 根据实际情况调整
API_PREFIX = "/model"

class ModelFilesAPITest:
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
        self.user_id = "test_user_001"
        self.headers = {
            "Content-Type": "application/json"
        }

    def test_list_user_data_files(self):
        """测试列出用户数据文件"""
        print("测试: 列出用户数据文件")
        try:
            response = requests.get(
                f"{self.base_url}{API_PREFIX}/data/list/{self.user_id}",
                headers=self.headers
            )
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"数据文件数量: {len(data)}")
                for file in data:
                    print(f"  - {file['name']}: {file['size']} bytes, 修改时间: {file['modified']}")
                return True
            else:
                print(f"错误: {response.text}")
                return False
        except Exception as e:
            print(f"异常: {str(e)}")
            return False

    def test_list_user_model_files(self):
        """测试列出用户模型文件"""
        print("测试: 列出用户模型文件")
        try:
            response = requests.get(
                f"{self.base_url}{API_PREFIX}/models/list/{self.user_id}",
                headers=self.headers
            )
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"模型文件数量: {len(data)}")
                for file in data:
                    print(f"  - {file['name']}: {file['size']} bytes, 修改时间: {file['modified']}")
                return True
            else:
                print(f"错误: {response.text}")
                return False
        except Exception as e:
            print(f"异常: {str(e)}")
            return False

    def test_get_user_data_file_detail(self, file_name):
        """测试获取用户数据文件详细信息"""
        print(f"测试: 获取数据文件 {file_name} 详细信息")
        try:
            response = requests.get(
                f"{self.base_url}{API_PREFIX}/data/{self.user_id}/{file_name}",
                headers=self.headers
            )
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"文件名: {data['name']}")
                print(f"路径: {data['path']}")
                print(f"大小: {data['size']} bytes")
                print(f"创建时间: {data['created_at']}")
                print(f"修改时间: {data['modified_at']}")
                print(f"对弈局数: {data['game_count']}")
                print(f"数据长度: {data['data_length']}")
                return True
            else:
                print(f"错误: {response.text}")
                return False
        except Exception as e:
            print(f"异常: {str(e)}")
            return False

    def test_get_user_model_file_detail(self, file_name):
        """测试获取用户模型文件详细信息"""
        print(f"测试: 获取模型文件 {file_name} 详细信息")
        try:
            response = requests.get(
                f"{self.base_url}{API_PREFIX}/models/{self.user_id}/{file_name}",
                headers=self.headers
            )
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"文件名: {data['name']}")
                print(f"路径: {data['path']}")
                print(f"大小: {data['size']} bytes")
                print(f"创建时间: {data['created_at']}")
                print(f"修改时间: {data['modified_at']}")
                print(f"扩展名: {data['extension']}")
                return True
            else:
                print(f"错误: {response.text}")
                return False
        except Exception as e:
            print(f"异常: {str(e)}")
            return False

    def test_sync_user_data_files_to_db(self):
        """测试同步用户数据文件到数据库"""
        print("测试: 同步用户数据文件到数据库")
        try:
            response = requests.post(
                f"{self.base_url}{API_PREFIX}/sync/data/{self.user_id}",
                headers=self.headers
            )
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"同步结果: {data['message']}")
                print(f"同步文件数: {data['synced_count']}")
                return True
            else:
                print(f"错误: {response.text}")
                return False
        except Exception as e:
            print(f"异常: {str(e)}")
            return False

    def test_sync_user_model_files_to_db(self):
        """测试同步用户模型文件到数据库"""
        print("测试: 同步用户模型文件到数据库")
        try:
            response = requests.post(
                f"{self.base_url}{API_PREFIX}/sync/models/{self.user_id}",
                headers=self.headers
            )
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"同步结果: {data['message']}")
                print(f"同步文件数: {data['synced_count']}")
                return True
            else:
                print(f"错误: {response.text}")
                return False
        except Exception as e:
            print(f"异常: {str(e)}")
            return False

    def test_sync_all_user_files_to_db(self):
        """测试同步用户所有文件到数据库"""
        print("测试: 同步用户所有文件到数据库")
        try:
            response = requests.post(
                f"{self.base_url}{API_PREFIX}/sync/all/{self.user_id}",
                headers=self.headers
            )
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"同步结果: {data['message']}")
                print(f"同步数据文件数: {data['data_files_synced']}")
                print(f"同步模型文件数: {data['model_files_synced']}")
                return True
            else:
                print(f"错误: {response.text}")
                return False
        except Exception as e:
            print(f"异常: {str(e)}")
            return False

    def test_get_user_data_files_from_db(self):
        """测试从数据库获取用户数据文件记录"""
        print("测试: 从数据库获取用户数据文件记录")
        try:
            response = requests.get(
                f"{self.base_url}{API_PREFIX}/db/data/{self.user_id}",
                headers=self.headers
            )
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"数据库中数据文件记录数: {len(data)}")
                for file in data:
                    print(f"  - ID: {file['id']}, 文件: {file['file_path']}")
                    print(f"    对弈局数: {file['game_count']}, 数据长度: {file['data_length']}")
                    print(f"    创建时间: {file['created_at']}, 更新时间: {file['updated_at']}")
                return True
            else:
                print(f"错误: {response.text}")
                return False
        except Exception as e:
            print(f"异常: {str(e)}")
            return False

    def test_get_user_model_files_from_db(self):
        """测试从数据库获取用户模型文件记录"""
        print("测试: 从数据库获取用户模型文件记录")
        try:
            response = requests.get(
                f"{self.base_url}{API_PREFIX}/db/models/{self.user_id}",
                headers=self.headers
            )
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"数据库中模型文件记录数: {len(data)}")
                for file in data:
                    print(f"  - ID: {file['id']}, 文件: {file['file_path']}")
                    print(f"    训练轮次: {file['training_epochs']}")
                    print(f"    创建时间: {file['created_at']}, 更新时间: {file['updated_at']}")
                return True
            else:
                print(f"错误: {response.text}")
                return False
        except Exception as e:
            print(f"异常: {str(e)}")
            return False

    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("开始测试模型文件API")
        print("=" * 60)

        tests = [
            self.test_list_user_data_files,
            self.test_list_user_model_files,
            self.test_sync_user_data_files_to_db,
            self.test_sync_user_model_files_to_db,
            self.test_sync_all_user_files_to_db,
            self.test_get_user_data_files_from_db,
            self.test_get_user_model_files_from_db,
        ]

        passed = 0
        failed = 0

        for test in tests:
            try:
                if test():
                    passed += 1
                    print("✓ 测试通过\n")
                else:
                    failed += 1
                    print("✗ 测试失败\n")
            except Exception as e:
                failed += 1
                print(f"✗ 测试异常: {str(e)}\n")

        print("=" * 60)
        print(f"测试完成: {passed} 通过, {failed} 失败")
        print("=" * 60)

# 简单的使用示例
if __name__ == "__main__":
    # 创建测试实例
    tester = ModelFilesAPITest()

    # 运行所有测试
    tester.run_all_tests()

    # 或者单独运行某个测试
    # tester.test_list_user_data_files()
    # tester.test_sync_all_user_files_to_db()
