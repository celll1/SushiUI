"""
Centralized error handling for the API
統一されたエラーハンドリング
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from datetime import datetime
from typing import Any, Dict, List, Optional
import traceback


_LORA_REFUSAL_CODES = frozenset({
    "lora_not_found",
    "lora_load_failed",
    "lora_incompatible",
    "lora_partial",
    "lora_uncond_unavailable",
    "ltx2_lora_arch_mismatch",
    "minimax_h3_lora_variant_mismatch",
    "lora_blockswap_unsupported",
})


def is_lora_refusal_code(code: Optional[str]) -> bool:
    """Whether ``code`` identifies a user-caused LoRA refusal."""
    return code in _LORA_REFUSAL_CODES


# ====================
# Custom Exception Classes
# ====================

class APIError(Exception):
    """Base API error.

    ``code`` is the machine-readable identifier of WHY the request was
    refused, drawn from the same taxonomy as `GenerationWarning.code`
    (`lora_not_found`, `lora_incompatible`, ...). It is optional: an
    unrelated failure carries no code and still produces a well-formed
    `ErrorResponse`. ``warnings`` carries the failing generation's
    `warnings[]` so a refusal reports what it recorded, not just its text.
    """
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        detail: str = None,
        code: Optional[str] = None,
        warnings: Optional[List[Dict[str, Any]]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        self.code = code
        self.warnings = warnings
        super().__init__(self.message)


class ValidationError(APIError):
    """Validation error (400)"""
    def __init__(self, message: str, detail: str = None,
                 code: Optional[str] = None,
                 warnings: Optional[List[Dict[str, Any]]] = None):
        super().__init__(message, status.HTTP_400_BAD_REQUEST, detail, code, warnings)


class NotFoundError(APIError):
    """Resource not found (404)"""
    def __init__(self, message: str, detail: str = None,
                 code: Optional[str] = None,
                 warnings: Optional[List[Dict[str, Any]]] = None):
        super().__init__(message, status.HTTP_404_NOT_FOUND, detail, code, warnings)


class GenerationError(APIError):
    """Generation failed (500), or a tagged LoRA refusal (400)."""
    def __init__(self, message: str, detail: str = None,
                 code: Optional[str] = None,
                 warnings: Optional[List[Dict[str, Any]]] = None):
        status_code = (status.HTTP_400_BAD_REQUEST if is_lora_refusal_code(code)
                       else status.HTTP_500_INTERNAL_SERVER_ERROR)
        super().__init__(message, status_code, detail, code, warnings)


class ModelError(APIError):
    """Model loading/operation error (500)"""
    def __init__(self, message: str, detail: str = None,
                 code: Optional[str] = None,
                 warnings: Optional[List[Dict[str, Any]]] = None):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR, detail, code, warnings)


class AuthenticationError(APIError):
    """Authentication error (401)"""
    def __init__(self, message: str, detail: str = None,
                 code: Optional[str] = None,
                 warnings: Optional[List[Dict[str, Any]]] = None):
        super().__init__(message, status.HTTP_401_UNAUTHORIZED, detail, code, warnings)


class PermissionError(APIError):
    """Permission denied (403)"""
    def __init__(self, message: str, detail: str = None,
                 code: Optional[str] = None,
                 warnings: Optional[List[Dict[str, Any]]] = None):
        super().__init__(message, status.HTTP_403_FORBIDDEN, detail, code, warnings)


# ====================
# Refusal codes on plain exceptions
# ====================

def with_error_code(exc: BaseException, code: str) -> BaseException:
    """Tag ``exc`` with the machine-readable ``code`` its refusal warned about.

    Used as ``raise with_error_code(RuntimeError(msg), "lora_incompatible")``.
    Tagging rather than converting to an `APIError` preserves the backend's
    exception contract. Generation routes use the tag, not the exception type,
    to answer user-caused LoRA refusals with 400 (see
    `api.generation_status.error_context`).
    """
    exc.code = code
    return exc


# ====================
# Error Response Builder
# ====================

def create_error_response(
    request: Request,
    error: str,
    status_code: int,
    detail: str = None,
    code: Optional[str] = None,
    warnings: Optional[List[Dict[str, Any]]] = None
) -> JSONResponse:
    """
    Create standardized error response

    Returns:
        JSONResponse with format:
        {
            "error": str,           # Error message
            "detail": str,          # Detailed error information (optional)
            "code": str,            # Machine-readable refusal code (null when none)
            "status_code": int,     # HTTP status code
            "timestamp": str,       # ISO 8601 timestamp
            "path": str,            # Request path
            "warnings": list        # Failing generation's warnings (omitted when none)
        }
    """
    content = {
        "error": error,
        "detail": detail,
        "code": code,
        "status_code": status_code,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "path": str(request.url.path)
    }
    if warnings:
        content["warnings"] = warnings
    return JSONResponse(status_code=status_code, content=content)


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
        exc.detail,
        code=getattr(exc, "code", None),
        warnings=getattr(exc, "warnings", None)
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
