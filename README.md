## MyAPIServer

FastAPI 项目文档
这是一个基于 FastAPI 的 Web 应用项目，支持以下核心功能：

### 🧩 功能概览

|       功能       |                    描述                    |
| :--------------: | :----------------------------------------: |
|  ✅ FastAPI 框架  |           提供 RESTful API 接口            |
|   ✅ Peewee ORM   |       轻量级数据库操作，支持 SQLite        |
|  ✅ 用户管理接口  |        GET /users/ 和 POST /users/         |
|    ✅ 日志系统    |    输出访问日志和错误信息到控制台和文件    |
|    ✅ 异常处理    |      全局异常捕获并返回 JSON 错误信息      |
|  ✅ 请求日志记录  | 记录每个请求的基本信息（IP、方法、路径等） |
| ✅ 结构化日志输出 |   使用 JSON 格式记录日志，便于分析与集成   |

### 📁 项目目录结构

apiserver/
├── main.py
├── config.py
├── requirements.txt
└── app/
    ├── __init__.py
    ├── api/
    │   └── user_routes.py      ← 新增：用户 RESTful 路由
    ├── services/
    │   └── user_service.py     ← 新增：业务逻辑封装
    ├── models/
    │   └── user.py             ← 数据模型
    ├── db/
    │   └── database.py         ← 数据库连接配置
    ├── utils/
    │   ├── logger.py           ← 日志系统
    │   └── exceptions.py       ← 自定义异常类
    ├── middleware/
    │   └── request_logger.py   ← 请求日志中间件
    └── exceptions/
        └── handler.py          ← 全局异常处理器

### 🔌 技术栈

Python 3.8+
FastAPI：用于构建高性能的异步 API
Uvicorn：ASGI 服务器，支持异步请求
Peewee：轻量级 ORM，适用于小型项目
SQLite：轻量数据库，适合本地开发
python-json-logger：生成 JSON 格式的日志
RotatingFileHandler：日志文件轮转机制

### 📦 数据库配置

默认使用 SQLite，数据库文件位于 ./test.db。你可以在 config.py 中修改数据库连接 URL。

### 📝 日志配置

日志分为两部分输出：
控制台输出：实时查看运行日志。
文件输出：日志保存在 logs/app.log 文件中，并支持轮转（最多保留 3 个备份文件）。
日志格式
所有日志以 JSON 格式输出，包含以下字段：
timestamp: 时间戳
level: 日志级别（INFO, WARNING, ERROR 等）
name: 日志来源模块名
message: 日志内容
client_ip: 客户端 IP 地址
method: HTTP 方法（GET, POST 等）
path: 请求路径
status_code: HTTP 状态码
process_time_ms: 请求耗时（毫秒）

### 🚨 异常处理

全局异常处理器会拦截所有未处理的异常，并返回统一格式的 JSON 错误响应。
自定义异常类
AppException: 基类，支持自定义状态码、错误码和消息
ResourceNotFoundException: 资源未找到异常（404）
InvalidInputException: 输入无效异常（400）
示例错误响应

`{`
  `"error": "RESOURCE_NOT_FOUND",`
  `"message": "User not found"`
`}`

### 📋 请求日志记录

中间件会自动记录每次请求的详细信息，包括：
客户端 IP 地址
HTTP 方法
请求路径
响应状态码
请求耗时（毫秒）

### 🧪 接口说明

|  方法  |    路径     | 功能         |
| :----: | :---------: | ------------ |
|  GET   |   /users/   | 获取所有用户 |
|  POST  |   /users/   | 创建新用户   |
|  GET   | /users/{id} | 获取指定用户 |
|  PUT   | /users/{id} | 更新用户信息 |
| DELETE | /users/{id} | 删除用户     |



### 💡 启动服务

`uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1`

注意：由于 SQLite 不支持多进程并发写入，建议设置 workers=1。

### 📦 打包发布

你可以将该项目打包为 ZIP 文件，或部署到 Docker 容器中。

示例 Dockerfile

`FROM python:3.9-slim`

`WORKDIR /app`

`COPY requirements.txt .`
`RUN pip install -r requirements.txt`

`COPY . .`

`CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]`

### 📦 可选扩展建议

添加 Trace ID
每个请求分配唯一 ID，用于追踪整个调用链
集成 ELK
将 JSON 日志发送至 Elasticsearch
Prometheus + Loki
实现日志 + 指标一体化监控
多环境配置
开发/测试/生产使用不同日志级别


### 总流程

显示棋盘
移动步骤高亮
偏移棋子高亮
显示移动步骤：马八进一
投降按钮，布局按钮，收棋按钮，悔棋按钮
当前回合数回合方
