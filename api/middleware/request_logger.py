# 请求日志中间件
import time
import logging
from fastapi import Request

logger = logging.getLogger(__name__)

async def log_requests(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path

    start_time = time.time()

    try:
        response = await call_next(request)
    except Exception as e:
        raise e
    finally:
        process_time = (time.time() - start_time) * 1000  # 毫秒
        extra = {
            "client_ip": client_ip,
            "method": method,
            "path": path,
            "status_code": response.status_code,
            "process_time_ms": f"{process_time:.2f}"
        }
        logger.info(f"Request completed", extra=extra)

    return response
