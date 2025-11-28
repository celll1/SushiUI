"""
Centralized error handling for the API
統一されたエラーハンドリング
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from datetime import datetime
import traceback


# ====================
# Custom Exception Classes
# ====================

class APIError(Exception):
    """Base API error"""
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        detail: str = None
    ):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(self.message)


class ValidationError(APIError):
    """Validation error (400)"""
    def __init__(self, message: str, detail: str = None):
        super().__init__(message, status.HTTP_400_BAD_REQUEST, detail)


class NotFoundError(APIError):
    """Resource not found (404)"""
    def __init__(self, message: str, detail: str = None):
        super().__init__(message, status.HTTP_404_NOT_FOUND, detail)


class GenerationError(APIError):
    """Generation failed (500)"""
    def __init__(self, message: str, detail: str = None):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR, detail)


class ModelError(APIError):
    """Model loading/operation error (500)"""
    def __init__(self, message: str, detail: str = None):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR, detail)


class AuthenticationError(APIError):
    """Authentication error (401)"""
    def __init__(self, message: str, detail: str = None):
        super().__init__(message, status.HTTP_401_UNAUTHORIZED, detail)


class PermissionError(APIError):
    """Permission denied (403)"""
    def __init__(self, message: str, detail: str = None):
        super().__init__(message, status.HTTP_403_FORBIDDEN, detail)


# ====================
# Error Response Builder
# ====================

def create_error_response(
    request: Request,
    error: str,
    status_code: int,
    detail: str = None
) -> JSONResponse:
    """
    Create standardized error response

    Returns:
        JSONResponse with format:
        {
            "error": str,           # Error message
            "detail": str,          # Detailed error information (optional)
            "status_code": int,     # HTTP status code
            "timestamp": str,       # ISO 8601 timestamp
            "path": str            # Request path
        }
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error,
            "detail": detail,
            "status_code": status_code,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "path": str(request.url.path)
        }
    )


# ====================
# Error Handlers
# ====================

async def api_error_handler(request: Request, exc: APIError):
    """Handle custom API errors"""
    print(f"[API Error] {exc.status_code} at {request.url.path}: {exc.message}")
    if exc.detail:
        print(f"[API Error] Detail: {exc.detail}")

    return create_error_response(
        request,
        exc.message,
        exc.status_code,
        exc.detail
    )


async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Handle FastAPI validation errors"""
    errors = exc.errors()

    # Format validation errors into readable message
    error_messages = []
    for err in errors:
        loc = " -> ".join(str(x) for x in err['loc'])
        error_messages.append(f"{loc}: {err['msg']}")

    detail = "; ".join(error_messages)

    print(f"[Validation Error] at {request.url.path}: {detail}")

    return create_error_response(
        request,
        "Validation error",
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail
    )


async def generic_error_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    error_detail = f"{str(exc)}\n\nTraceback:\n{traceback.format_exc()}"
    print(f"[ERROR] Unexpected error at {request.url.path}: {error_detail}")

    # In production, we might want to hide detailed error messages
    # For now, include them for debugging
    return create_error_response(
        request,
        "Internal server error",
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        str(exc)
    )


# ====================
# Registration Function
# ====================

def register_error_handlers(app):
    """
    Register all error handlers with the FastAPI app

    Usage:
        from api.error_handlers import register_error_handlers

        app = FastAPI()
        register_error_handlers(app)
    """
    # Custom API errors
    app.add_exception_handler(APIError, api_error_handler)

    # FastAPI validation errors
    app.add_exception_handler(RequestValidationError, validation_error_handler)

    # Generic errors (catch-all)
    app.add_exception_handler(Exception, generic_error_handler)

    print("[ErrorHandlers] Registered error handlers")
