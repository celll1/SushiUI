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
    created_at = Column(DateTime, default=datetime.utcnow)
    is_favorite = Column(Boolean, default=False)

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
            "created_at": self.created_at.isoformat(),
            "is_favorite": self.is_favorite,
        }
