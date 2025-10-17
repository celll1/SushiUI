from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

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
        return {
            "id": self.id,
            "filename": self.filename,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "model_name": self.model_name,
            "sampler": self.sampler,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "seed": self.seed,
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
