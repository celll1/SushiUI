"""
Aesthetic Scorer - FastAPI Application

Lightweight tool for scoring predicted latent quality in diffusion model training.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn

from api.routes import router
from database import init_db

# Initialize FastAPI app
app = FastAPI(
    title="Aesthetic Scorer",
    description="Tool for scoring predicted latent quality to prevent overbaked images",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api", tags=["aesthetic"])

# Serve static files (decoded images)
images_dir = Path("data/images")
images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    init_db()
    print("[AestheticScorer] Database initialized")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Aesthetic Scorer",
        "version": "1.0.0",
        "description": "Tool for scoring predicted latent quality",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aesthetic Scorer Backend")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8001, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    print("=" * 60)
    print("Aesthetic Scorer Backend")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"URL: http://{args.host}:{args.port}")
    print("=" * 60)

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
