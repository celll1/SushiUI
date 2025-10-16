from pydantic_settings import BaseSettings
from typing import Optional
import os

# Calculate paths at module level (use realpath for Windows compatibility)
_base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_root_dir = os.path.realpath(os.path.join(_base_dir, ".."))

class Settings(BaseSettings):
    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = True

    # Paths
    base_dir: str = _base_dir
    root_dir: str = _root_dir
    models_dir: str = os.path.realpath(os.path.join(_root_dir, "models"))
    outputs_dir: str = os.path.realpath(os.path.join(_root_dir, "outputs"))
    thumbnails_dir: str = os.path.realpath(os.path.join(_root_dir, "thumbnails"))
    lora_dir: str = os.path.realpath(os.path.join(_root_dir, "lora"))
    controlnet_dir: str = os.path.realpath(os.path.join(_root_dir, "controlnet"))

    # Database
    database_url: str = f"sqlite:///{os.path.realpath(os.path.join(_root_dir, 'sd_webui.db'))}"

    # Generation defaults
    default_steps: int = 20
    default_cfg_scale: float = 7.0
    default_sampler: str = "euler_a"
    default_width: int = 512
    default_height: int = 512

    # System
    max_batch_size: int = 4
    device: str = "cuda"

    class Config:
        env_file = ".env"

settings = Settings()

# Debug: Print paths on startup
print(f"[Settings] Root dir: {settings.root_dir}")
print(f"[Settings] Outputs dir: {settings.outputs_dir}")
print(f"[Settings] Models dir: {settings.models_dir}")
print(f"[Settings] LoRA dir: {settings.lora_dir}")
print(f"[Settings] ControlNet dir: {settings.controlnet_dir}")
print(f"[Settings] Thumbnails dir: {settings.thumbnails_dir}")
