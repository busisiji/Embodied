# 统一定义业务异常，提升错误可读性和一致性。
from fastapi import HTTPException

class AppException(HTTPException):
    def __init__(self, status_code: int, error_code: str, message: str):
        super().__init__(status_code=status_code, detail=message)
        self.error_code = error_code
        self.message = message

class ResourceNotFoundException(AppException):
    def __init__(self, resource: str):
        super().__init__(
            status_code=404,
            error_code="RESOURCE_NOT_FOUND",
            message=f"{resource} not found"
        )

class InvalidInputException(AppException):
    def __init__(self, message: str):
        super().__init__(
            status_code=400,
            error_code="INVALID_INPUT",
            message=message
        )
