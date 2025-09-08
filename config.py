# 配置管理
import os

from pydantic.v1 import BaseSettings


class Settings(BaseSettings):
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    API_ROOT: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api")
    DATABASE_URL: str = f"sqlite:///{os.path.join(API_ROOT, 'data', 'test.db')}"


settings = Settings()
