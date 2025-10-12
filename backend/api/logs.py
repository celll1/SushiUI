from fastapi import APIRouter
from typing import Optional
from utils.logger import log_capture

router = APIRouter()

@router.get("/logs")
async def get_logs(last: Optional[int] = None):
    """Get console logs"""
    logs = log_capture.get_logs(last_n=last)
    return {"logs": logs, "total": len(log_capture.logs)}

@router.delete("/logs")
async def clear_logs():
    """Clear all logs"""
    log_capture.clear()
    return {"success": True, "message": "Logs cleared"}
