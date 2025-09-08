import os
from urllib.parse import urlparse

# 改进的路径处理逻辑
def get_db_path(database_url):
    if database_url.startswith("sqlite"):
        # 解析URL
        parsed = urlparse(database_url)
        db_path = parsed.path

        # 处理Windows路径问题
        if os.name == 'nt':  # Windows系统
            # 移除开头的斜杠（如果有的话）
            if db_path.startswith('/'):
                db_path = db_path[1:]
            # 处理盘符情况（如 /C:/path/to/file -> C:/path/to/file）
            if len(db_path) > 2 and db_path[1] == ':':
                db_path = db_path[1:]  # 移除开头的斜杠

        return db_path
    return None