from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import os
import logging
from PIL import Image

# Remove PIL image size limit for large images
# Reference: https://kakashibata.hatenablog.jp/entry/2022/03/27/232553
Image.MAX_IMAGE_PIXELS = None
print("[PIL] MAX_IMAGE_PIXELS limit removed (can handle large images)")

from api import router, websocket_endpoint
from api.logs import router as logs_router
from api.error_handlers import register_error_handlers
from database import init_db
from config.settings import settings
from utils.logger import setup_logging

# Setup logging capture
setup_logging()

# Custom log filter to exclude specific endpoints
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Exclude gpu-stats endpoint from logs
        return record.getMessage().find("/api/v1/system/gpu-stats") == -1

# Disable uvicorn access logs
logging.getLogger("uvicorn.access").disabled = True

# Add filter to uvicorn logger
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())
logging.getLogger("uvicorn").addFilter(EndpointFilter())

# Initialize database
init_db()

# Create FastAPI app
app = FastAPI(title="Stable Diffusion WebUI API", version="0.1.0")

# Register error handlers
register_error_handlers(app)

# Start WebSocket message sender on startup and load user directory settings
@app.on_event("startup")
async def startup_event():
    import asyncio
    from api.websocket import manager
    from database import GallerySessionLocal
    from database.models import UserSettings
    from core.lora_manager import lora_manager
    from core.controlnet_manager import controlnet_manager
    from core.pipeline import pipeline_manager

    asyncio.create_task(manager.start_sender())

    # Load user-configured directories for LoRA and ControlNet managers
    try:
        db = GallerySessionLocal()
        settings_record = db.query(UserSettings).first()
        if settings_record:
            print("[Startup] Loading user-configured directories...")
            if settings_record.lora_dirs:
                lora_manager.set_additional_dirs(settings_record.lora_dirs)
            if settings_record.controlnet_dirs:
                controlnet_manager.set_additional_dirs(settings_record.controlnet_dirs)
        db.close()
    except Exception as e:
        print(f"[Startup] Error loading user directory settings: {e}")

    # Load last used model in background (non-blocking)
    async def load_model_background():
        try:
            print("[Startup] Starting background model loading...")
            # Run in thread pool to avoid blocking event loop
            await asyncio.to_thread(pipeline_manager._auto_load_last_model)
        except Exception as e:
            print(f"[Startup] Error loading model in background: {e}")

    asyncio.create_task(load_model_background())

# WebSocket endpoint BEFORE middleware (to bypass CORS)
@app.websocket("/api/v1/ws/progress")
async def websocket_route(websocket: WebSocket):
    await websocket_endpoint(websocket)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with versioning
app.include_router(router, prefix="/api/v1")
app.include_router(logs_router, prefix="/api/v1")

# Backward compatibility: redirect old /api/* endpoints to /api/v1/*
@app.middleware("http")
async def redirect_legacy_endpoints(request: Request, call_next):
    """Redirect old /api/* endpoints to /api/v1/* for backward compatibility"""
    path = request.url.path

    # If path starts with /api/ but NOT /api/v1/, redirect to /api/v1/
    if path.startswith("/api/") and not path.startswith("/api/v1/"):
        # Extract the part after /api/
        suffix = path[5:]  # Remove "/api/"
        new_path = f"/api/v1/{suffix}"

        # Preserve query parameters
        query_string = request.url.query
        new_url = new_path if not query_string else f"{new_path}?{query_string}"

        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=new_url, status_code=308)  # 308 = Permanent Redirect, preserves method

    # Otherwise, proceed normally
    response = await call_next(request)
    return response

# Serve static files
if os.path.exists(settings.outputs_dir):
    print(f"[Static] Mounting /outputs -> {settings.outputs_dir}")
    app.mount("/outputs", StaticFiles(directory=settings.outputs_dir), name="outputs")
else:
    print(f"[Static] WARNING: outputs_dir does not exist: {settings.outputs_dir}")

if os.path.exists(settings.thumbnails_dir):
    print(f"[Static] Mounting /thumbnails -> {settings.thumbnails_dir}")
    app.mount("/thumbnails", StaticFiles(directory=settings.thumbnails_dir), name="thumbnails")
else:
    print(f"[Static] WARNING: thumbnails_dir does not exist: {settings.thumbnails_dir}")

# Mount training directory
training_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "training")
if os.path.exists(training_dir):
    print(f"[Static] Mounting /training -> {training_dir}")
    app.mount("/training", StaticFiles(directory=training_dir), name="training")
else:
    print(f"[Static] WARNING: training directory does not exist: {training_dir}")

@app.get("/")
async def root():
    return {"message": "Stable Diffusion WebUI API", "version": "0.1.0"}

def find_available_port(start_port: int, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port"""
    import socket

    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((settings.host, port))
                return port
        except OSError:
            print(f"[Server] Port {port} is already in use, trying next port...")
            continue

    raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts - 1}")

def save_port_info(port: int):
    """Save port info to file for frontend to read"""
    import json
    port_info_file = os.path.join(os.path.dirname(__file__), ".port_info")
    with open(port_info_file, "w") as f:
        json.dump({"port": port, "host": settings.host}, f)
    print(f"[Server] Port info saved to {port_info_file}")

if __name__ == "__main__":
    import uvicorn

    # Find available port
    actual_port = find_available_port(settings.port)

    if actual_port != settings.port:
        print(f"[Server] Port {settings.port} is in use, using port {actual_port} instead")

    # Save port info for frontend
    save_port_info(actual_port)

    uvicorn.run(
        app,
        host=settings.host,
        port=actual_port,
        timeout_keep_alive=600,  # Keep connections alive for 10 minutes
        timeout_graceful_shutdown=30,  # 30 seconds for graceful shutdown
        access_log=False,  # Disable access logs
    )
