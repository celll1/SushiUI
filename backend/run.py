"""
Alternative entry point that adds backend to Python path
Run this instead of main.py if you get import errors
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    import uvicorn
    from backend.config.settings import settings

    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        timeout_keep_alive=600,  # Keep connections alive for 10 minutes
        timeout_graceful_shutdown=30,  # 30 seconds for graceful shutdown
        log_level="warning",  # Reduce INFO logs (error, warning, info, debug, trace)
        access_log=False,  # Disable access logs (e.g., "GET /api/system/gpu-stats")
    )
