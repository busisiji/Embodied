# 异常统一处理
from datetime import datetime

from fastapi import Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging
from api.utils.exceptions import AppException

logger = logging.getLogger(__name__)

def add_exception_handlers(app):
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        error_msg = "请求参数验证失败"
        return JSONResponse(
            status_code=422,
            content={
                "success" : False,
                "message": error_msg,
                "data": None,
                "timestamp": datetime.now().isoformat()
            }
        )

    # 对于其他HTTP异常也可以添加处理
    from starlette.exceptions import HTTPException

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "message": str(exc.detail),
                "data": None,
                "timestamp": datetime.now().isoformat()
            }
        )
