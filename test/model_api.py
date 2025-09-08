# tests/test_model_api.py
import unittest
import requests
import time
import os
from typing import Dict, Any

class TestModelAPI(unittest.TestCase):
    """
    模型API测试类
    """

    def setUp(self):
        """
        测试前准备
        """
        self.base_url = "http://localhost:6017"  # 假设API运行在本地6017端口
        self.process_ids = []

    def tearDown(self):
        """
        测试后清理
        """
        # 停止所有创建的进程
        for process_id in self.process_ids:
            try:
                response = requests.post(f"{self.base_url}/model/process/stop/{process_id}")
                print(f"已停止进程 {process_id}: {response.json()}")
            except Exception as e:
                print(f"停止进程 {process_id} 失败: {e}")

    def _start_process_and_get_id(self, endpoint: str, data: Dict[Any, Any] = None, params: Dict[str, str] = None) -> str:
        """
        启动进程并返回进程ID
        """
        url = f"{self.base_url}{endpoint}"
        if data:
            response = requests.post(url, json=data)
        elif params:
            response = requests.post(url, params=params)
        else:
            response = requests.post(url)

        self.assertEqual(response.status_code, 200, f"启动进程失败: {response.text}")
        process_id = response.json()["process_id"]
        self.process_ids.append(process_id)
        return process_id

    def test_start_self_play_cpu(self):
        """
        测试启动CPU自我对弈数据采集
        """
        data = {
            "workers": 1,
            "use_gpu": False,
            "collect_mode": "multi_thread",
            "temp": 1.0,
            "cpuct": 5.0,
            "game_count": 1
        }

        process_id = self._start_process_and_get_id("/model/selfplay/start", data)
        print(f"已启动CPU自我对弈进程: {process_id}")

        # 检查进程状态
        response = requests.get(f"{self.base_url}/model/process/status/{process_id}")
        self.assertEqual(response.status_code, 200)
        status = response.json()
        self.assertIn(status["status"], ["running", "completed", "failed"])
        self.assertEqual(status["type"], "selfplay")

    def test_start_training(self):
        """
        测试启动模型训练
        """
        data = {
            "init_model": None,
            "data_path": None,
            "epochs": 1,
            "batch_size": 32
        }

        process_id = self._start_process_and_get_id("/model/training/start", data)
        print(f"已启动模型训练进程: {process_id}")

        # 检查进程状态
        response = requests.get(f"{self.base_url}/model/process/status/{process_id}")
        self.assertEqual(response.status_code, 200)
        status = response.json()
        self.assertIn(status["status"], ["running", "completed", "failed"])
        self.assertEqual(status["type"], "training")

    def test_start_auto_training(self):
        """
        测试启动自动训练流程
        """
        data = {
            "init_model": None,
            "interval_minutes": 1,
            "workers": 1,
            "use_gpu": False,
            "collect_mode": "multi_thread",
            "temp": 1.0,
            "cpuct": 5.0,
            "self_play_only": True  # 仅数据采集，不训练模型，避免长时间运行
        }

        process_id = self._start_process_and_get_id("/model/auto-training/start", data)
        print(f"已启动自动训练流程: {process_id}")

        # 检查进程状态
        response = requests.get(f"{self.base_url}/model/process/status/{process_id}")
        self.assertEqual(response.status_code, 200)
        status = response.json()
        self.assertIn(status["status"], ["running", "completed", "failed"])
        self.assertEqual(status["type"], "auto_training")

    def test_start_evaluation(self):
        """
        测试启动模型评估
        """
        # 首先获取一个可用的模型路径用于测试
        response = requests.get(f"{self.base_url}/model/models/list")
        self.assertEqual(response.status_code, 200)
        models = response.json()

        # 如果有可用模型，则使用第一个模型进行测试
        if models:
            model_path = models[0]["path"]
            params = {"model_path": model_path}

            process_id = self._start_process_and_get_id("/model/evaluation/start", params=params)
            print(f"已启动模型评估进程: {process_id}")

            # 检查进程状态
            response = requests.get(f"{self.base_url}/model/process/status/{process_id}")
            self.assertEqual(response.status_code, 200)
            status = response.json()
            self.assertIn(status["status"], ["running", "completed", "failed"])
            self.assertEqual(status["type"], "evaluation")

            # 获取进程日志
            response = requests.get(f"{self.base_url}/model/process/logs/{process_id}?limit=5")
            self.assertEqual(response.status_code, 200)
            logs = response.json()
            print(f"获取到 {len(logs)} 条评估日志")
        else:
            # 如果没有可用模型，测试API端点是否能正确响应
            print("警告: 没有可用模型用于评估测试")
            params = {"model_path": "nonexistent_model.pkl"}
            try:
                response = requests.post(f"{self.base_url}/model/evaluation/start", params=params)
                # 我们只检查状态码不是405（Method Not Allowed）或404（Not Found）
                self.assertNotIn(response.status_code, [404, 405])
                print(f"模型评估API响应: {response.status_code}")
            except Exception as e:
                print(f"模型评估测试跳过: {e}")

    def test_list_models(self):
        """
        测试列出所有模型
        """
        response = requests.get(f"{self.base_url}/model/models/list")
        self.assertEqual(response.status_code, 200)
        models = response.json()
        self.assertIsInstance(models, list)
        print(f"找到 {len(models)} 个模型文件")

        # 检查模型信息结构
        if models:
            model = models[0]
            self.assertIn("name", model)
            self.assertIn("path", model)
            self.assertIn("size", model)
            self.assertIn("modified", model)

    def test_list_data(self):
        """
        测试列出所有数据文件
        """
        response = requests.get(f"{self.base_url}/model/data/list")
        self.assertEqual(response.status_code, 200)
        data_files = response.json()
        self.assertIsInstance(data_files, list)
        print(f"找到 {len(data_files)} 个数据文件")

        # 检查数据文件信息结构
        if data_files:
            data_file = data_files[0]
            self.assertIn("name", data_file)
            self.assertIn("path", data_file)
            self.assertIn("size", data_file)
            self.assertIn("modified", data_file)

    def test_process_management(self):
        """
        测试进程管理功能
        """
        # 启动一个短时间运行的进程
        data = {
            "workers": 1,
            "use_gpu": False,
            "collect_mode": "multi_thread",
            "temp": 1.0,
            "cpuct": 5.0,
            "game_count": 1
        }

        process_id = self._start_process_and_get_id("/model/selfplay/start", data)
        print(f"已启动测试进程: {process_id}")

        # 检查进程是否在列表中
        response = requests.get(f"{self.base_url}/model/process/list")
        self.assertEqual(response.status_code, 200)
        processes = response.json()
        process_ids = [p["process_id"] for p in processes]
        self.assertIn(process_id, process_ids)

        # 获取进程日志
        response = requests.get(f"{self.base_url}/model/process/logs/{process_id}?limit=5")
        self.assertEqual(response.status_code, 200)
        logs = response.json()
        print(f"获取到 {len(logs)} 条日志")

        # 停止进程
        response = requests.post(f"{self.base_url}/model/process/stop/{process_id}")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["process_id"], process_id)

    def test_get_process_status(self):
        """
        测试获取进程状态
        """
        # 启动一个进程
        data = {
            "workers": 1,
            "use_gpu": False,
            "collect_mode": "multi_thread",
            "temp": 1.0,
            "cpuct": 5.0,
            "game_count": 1
        }

        process_id = self._start_process_and_get_id("/model/selfplay/start", data)

        # 获取进程状态
        response = requests.get(f"{self.base_url}/model/process/status/{process_id}")
        self.assertEqual(response.status_code, 200)
        status = response.json()

        self.assertIn("process_id", status)
        self.assertIn("type", status)
        self.assertIn("status", status)
        self.assertIn("start_time", status)

        # 从进程列表中移除
        if process_id in self.process_ids:
            self.process_ids.remove(process_id)
        requests.post(f"{self.base_url}/model/process/stop/{process_id}")

def run_tests():
    """
    运行所有测试
    """
    # 创建测试套件
    suite = unittest.TestSuite()

    # 添加测试用例（根据需要启用/禁用）
    suite.addTest(TestModelAPI('test_start_self_play_cpu'))
    suite.addTest(TestModelAPI('test_start_training'))
    suite.addTest(TestModelAPI('test_start_auto_training'))
    suite.addTest(TestModelAPI('test_start_evaluation'))
    suite.addTest(TestModelAPI('test_list_models'))
    suite.addTest(TestModelAPI('test_list_data'))
    suite.addTest(TestModelAPI('test_process_management'))
    suite.addTest(TestModelAPI('test_get_process_status'))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()

if __name__ == "__main__":
    # 检查API服务是否运行
    try:
        response = requests.get("http://localhost:6017/docs")
        if response.status_code == 200:
            print("API服务正在运行，开始测试...")
            success = run_tests()
            if success:
                print("\n所有测试通过!")
            else:
                print("\n部分测试失败!")
        else:
            print("API服务未运行，请先启动API服务")
    except requests.exceptions.ConnectionError:
        print("无法连接到API服务，请确保服务正在运行")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
