from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from api import router, websocket_endpoint
from api.logs import router as logs_router
from database import init_db
from config.settings import settings
from utils.logger import setup_logging

# Setup logging capture
setup_logging()

# Initialize database
init_db()

# Create FastAPI app
app = FastAPI(title="Stable Diffusion WebUI API", version="0.1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api")
app.include_router(logs_router, prefix="/api")

# WebSocket endpoint
@app.websocket("/ws/progress")
async def websocket_route(websocket):
    await websocket_endpoint(websocket)

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

@app.get("/")
async def root():
    return {"message": "Stable Diffusion WebUI API", "version": "0.1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
