# 根据 config_model.py 中的 DATABASE_URL 配置数据库连接。
import os

import peewee
from config import settings
from urllib.parse import urlparse

db_path = urlparse(settings.DATABASE_URL).path.replace('/', '')
db_dir = os.path.dirname(db_path)
if db_dir and not os.path.exists(db_dir):
    os.makedirs(db_dir)

# 根据 DATABASE_URL 自动识别数据库类型
if settings.DATABASE_URL.startswith("sqlite"):
    db = peewee.SqliteDatabase(db_path)
elif settings.DATABASE_URL.startswith("mysql"):
    from urllib.parse import urlparse
    url = urlparse(db_path)
    db = peewee.MySQLDatabase(
        database=url.path[1:],
        user=url.username,
        password=url.password,
        host=url.hostname,
        port=url.port or 3306
    )
elif settings.DATABASE_URL.startswith("postgres"):
    from urllib.parse import urlparse
    url = urlparse(db_path)
    db = peewee.PostgresqlDatabase(
        database=url.path[1:],
        user=url.username,
        password=url.password,
        host=url.hostname,
        port=url.port or 5432
    )
else:
    raise ValueError("Unsupported database type")
