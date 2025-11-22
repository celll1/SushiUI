from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class UserSettings(Base):
    """User settings for application configuration"""
    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, index=True)
    # Store directory paths as JSON arrays
    model_dirs = Column(JSON, default=list)  # Additional directories for base models
    lora_dirs = Column(JSON, default=list)   # Additional directories for LoRAs
    controlnet_dirs = Column(JSON, default=list)  # Additional directories for ControlNets
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "model_dirs": self.model_dirs or [],
            "lora_dirs": self.lora_dirs or [],
            "controlnet_dirs": self.controlnet_dirs or [],
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

class GeneratedImage(Base):
    __tablename__ = "generated_images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    prompt = Column(String)
    negative_prompt = Column(String, nullable=True)
    model_name = Column(String)
    sampler = Column(String)
    steps = Column(Integer)
    cfg_scale = Column(Float)
    seed = Column(Integer)
    ancestral_seed = Column(Integer, nullable=True)  # Seed for stochastic samplers (Euler a, etc.)
    width = Column(Integer)
    height = Column(Integer)
    generation_type = Column(String)  # txt2img, img2img, inpaint
    parameters = Column(JSON)  # Full generation parameters
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    is_favorite = Column(Boolean, default=False)

    # New metadata fields
    image_hash = Column(String, nullable=True, index=True)  # SHA256 hash of generated image
    source_image_hash = Column(String, nullable=True)  # Hash of source image for img2img/inpaint
    mask_data = Column(String, nullable=True)  # Base64 encoded mask for inpaint
    lora_names = Column(String, nullable=True)  # Comma-separated LoRA filenames
    model_hash = Column(String, nullable=True)  # SHA256 hash of model file

    def to_dict(self):
        result = {
            "id": self.id,
            "filename": self.filename,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "model_name": self.model_name,
            "sampler": self.sampler,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "seed": self.seed,
            "ancestral_seed": self.ancestral_seed,
            "width": self.width,
            "height": self.height,
            "generation_type": self.generation_type,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_favorite": self.is_favorite,
            "image_hash": self.image_hash,
            "source_image_hash": self.source_image_hash,
            "mask_data": self.mask_data,
            "lora_names": self.lora_names,
            "model_hash": self.model_hash,
        }

        # Extract Advanced CFG parameters from parameters JSON if available
        if self.parameters:
            if "cfg_schedule_type" in self.parameters:
                result["cfg_schedule_type"] = self.parameters["cfg_schedule_type"]
            if "cfg_schedule_min" in self.parameters:
                result["cfg_schedule_min"] = str(self.parameters["cfg_schedule_min"])
            if "cfg_schedule_max" in self.parameters:
                result["cfg_schedule_max"] = str(self.parameters["cfg_schedule_max"])
            if "cfg_schedule_power" in self.parameters:
                result["cfg_schedule_power"] = str(self.parameters["cfg_schedule_power"])
            if "cfg_rescale_snr_alpha" in self.parameters:
                result["cfg_rescale_snr_alpha"] = str(self.parameters["cfg_rescale_snr_alpha"])
            if "dynamic_threshold_percentile" in self.parameters:
                result["dynamic_threshold_percentile"] = str(self.parameters["dynamic_threshold_percentile"])
            if "dynamic_threshold_mimic_scale" in self.parameters:
                result["dynamic_threshold_mimic_scale"] = str(self.parameters["dynamic_threshold_mimic_scale"])

        return result
