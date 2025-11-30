from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

# Helper function to get local time
def get_local_now():
    """Get current local time (not UTC)"""
    return datetime.now()

# Create separate declarative bases for each database
GalleryBase = declarative_base()
DatasetBase = declarative_base()
TrainingBase = declarative_base()

# ============================================================
# Gallery Models (gallery.db)
# ============================================================

class UserSettings(GalleryBase):
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

class GeneratedImage(GalleryBase):
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
    created_at = Column(DateTime, default=get_local_now, index=True)
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

        # Extract Advanced CFG and NAG parameters from parameters JSON if available
        if self.parameters:
            # NAG parameters
            nag_enable = self.parameters.get("nag_enable", False)
            if nag_enable:
                result["nag_enable"] = str(nag_enable)
                if "nag_scale" in self.parameters:
                    result["nag_scale"] = str(self.parameters["nag_scale"])
                if "nag_tau" in self.parameters:
                    result["nag_tau"] = str(self.parameters["nag_tau"])
                if "nag_alpha" in self.parameters:
                    result["nag_alpha"] = str(self.parameters["nag_alpha"])
                if "nag_sigma_end" in self.parameters:
                    result["nag_sigma_end"] = str(self.parameters["nag_sigma_end"])

            # Advanced CFG parameters (can coexist with NAG)
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

            # U-Net Quantization
            if "unet_quantization" in self.parameters:
                result["unet_quantization"] = self.parameters["unet_quantization"]

        return result


# ============================================================
# Dataset Management Models
# ============================================================



# ============================================================
# Dataset Models (datasets.db)
# ============================================================

class Dataset(DatasetBase):
    """Dataset for training/fine-tuning models"""
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    path = Column(String, nullable=False)
    description = Column(Text, nullable=True)

    # Caption settings
    caption_suffixes = Column(JSON, default=list)
    default_caption_type = Column(String, default="tags")

    # Image pair settings
    image_suffixes = Column(JSON, default=list)

    # Scanning settings
    recursive = Column(Boolean, default=True)
    max_depth = Column(Integer, nullable=True)
    file_extensions = Column(JSON, default=list)

    # Metadata settings
    read_exif = Column(Boolean, default=False)
    exif_caption_fields = Column(JSON, nullable=True)

    # Statistics
    total_items = Column(Integer, default=0)
    total_captions = Column(Integer, default=0)
    total_tags = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, default=get_local_now, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_scanned_at = Column(DateTime, nullable=True)

    # Relationships (within datasets.db only)
    items = relationship("DatasetItem", back_populates="dataset", cascade="all, delete-orphan")
    # Note: TrainingRun is in a separate database (training.db)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "caption_suffixes": self.caption_suffixes or [],
            "default_caption_type": self.default_caption_type,
            "image_suffixes": self.image_suffixes or [],
            "recursive": self.recursive,
            "max_depth": self.max_depth,
            "file_extensions": self.file_extensions or [],
            "read_exif": self.read_exif,
            "exif_caption_fields": self.exif_caption_fields,
            "total_items": self.total_items,
            "total_captions": self.total_captions,
            "total_tags": self.total_tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_scanned_at": self.last_scanned_at.isoformat() if self.last_scanned_at else None,
        }


class DatasetItem(DatasetBase):
    """Individual item (image or image group) in a dataset"""
    __tablename__ = "dataset_items"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)

    # Item identification
    item_type = Column(String, default="single", index=True)
    base_name = Column(String, index=True, nullable=False)
    group_id = Column(String, nullable=True, index=True)

    # Image paths
    image_path = Column(String, nullable=False, index=True)
    image_suffix = Column(String, nullable=True)
    related_images = Column(JSON, nullable=True)

    # Image properties
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    file_size = Column(Integer, nullable=True)
    image_hash = Column(String, nullable=True, index=True)

    # EXIF metadata
    exif_data = Column(JSON, nullable=True)

    # Statistics
    total_captions = Column(Integer, default=0)
    total_tags = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, default=get_local_now)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    dataset = relationship("Dataset", back_populates="items")
    captions = relationship("DatasetCaption", back_populates="item", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "item_type": self.item_type,
            "base_name": self.base_name,
            "group_id": self.group_id,
            "image_path": self.image_path,
            "image_suffix": self.image_suffix,
            "related_images": self.related_images,
            "width": self.width,
            "height": self.height,
            "file_size": self.file_size,
            "image_hash": self.image_hash,
            "exif_data": self.exif_data,
            "total_captions": self.total_captions,
            "total_tags": self.total_tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DatasetCaption(DatasetBase):
    """Caption associated with a dataset item"""
    __tablename__ = "dataset_captions"

    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, ForeignKey("dataset_items.id", ondelete="CASCADE"), nullable=False, index=True)

    # Caption type and content
    caption_type = Column(String, index=True, nullable=False)
    caption_subtype = Column(String, nullable=True)
    content = Column(Text, nullable=False)

    # Metadata
    language = Column(String, nullable=True)
    source = Column(String, default="manual", index=True)
    source_field = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=get_local_now)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    item = relationship("DatasetItem", back_populates="captions")

    def to_dict(self):
        return {
            "id": self.id,
            "item_id": self.item_id,
            "caption_type": self.caption_type,
            "caption_subtype": self.caption_subtype,
            "content": self.content,
            "language": self.language,
            "source": self.source,
            "source_field": self.source_field,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class TagDictionary(DatasetBase):
    """Global tag dictionary (Danbooru tags + custom tags)"""
    __tablename__ = "tag_dictionary"

    id = Column(Integer, primary_key=True, index=True)
    tag = Column(String, unique=True, index=True, nullable=False)

    # Tag metadata
    category = Column(String, index=True, nullable=False)
    count = Column(Integer, default=0, index=True)

    # Display and aliases
    display_name = Column(String, nullable=True)
    aliases = Column(JSON, nullable=True)
    description = Column(Text, nullable=True)

    # Source tracking
    source = Column(String, default="danbooru", index=True)
    is_official = Column(Boolean, default=True, index=True)
    is_deprecated = Column(Boolean, default=False, index=True)
    replacement_tag = Column(String, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=get_local_now)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "tag": self.tag,
            "category": self.category,
            "count": self.count,
            "display_name": self.display_name,
            "aliases": self.aliases,
            "description": self.description,
            "source": self.source,
            "is_official": self.is_official,
            "is_deprecated": self.is_deprecated,
            "replacement_tag": self.replacement_tag,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }




# ============================================================
# Training Models (training.db)
# ============================================================

class TrainingRun(TrainingBase):
    """Training run for model fine-tuning or LoRA training"""
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, nullable=False, index=True)  # No ForeignKey - different database
    run_id = Column(String, unique=True, nullable=False, index=True, default=lambda: str(uuid.uuid4()))  # Unique ID (UUID)

    # Run identification
    run_name = Column(String, unique=True, index=True, nullable=False)
    training_method = Column(String, nullable=False, index=True)  # 'lora', 'full_finetune'
    base_model_path = Column(String, nullable=False)
    
    # Configuration
    config_yaml = Column(Text)  # Full ai-toolkit YAML config
    
    # Status
    status = Column(String, default="pending", index=True)  # 'pending', 'running', 'paused', 'completed', 'failed'
    progress = Column(Float, default=0.0)  # 0.0 - 1.0
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, nullable=False)
    
    # Performance metrics
    loss = Column(Float, nullable=True)
    learning_rate = Column(Float, nullable=True)
    
    # Output
    output_dir = Column(String, nullable=False)
    checkpoint_paths = Column(JSON, default=list)  # List of checkpoint file paths
    
    # Logs
    log_file = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=get_local_now, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships (within training.db only)
    # Note: dataset_id references datasets.db, but no ForeignKey constraint
    checkpoints = relationship("TrainingCheckpoint", back_populates="run", cascade="all, delete-orphan")
    samples = relationship("TrainingSample", back_populates="run", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "training_method": self.training_method,
            "base_model_path": self.base_model_path,
            "config_yaml": self.config_yaml,
            "status": self.status,
            "progress": self.progress,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "output_dir": self.output_dir,
            "checkpoint_paths": self.checkpoint_paths,
            "log_file": self.log_file,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class TrainingCheckpoint(TrainingBase):
    """Training checkpoint saved during training"""
    __tablename__ = "training_checkpoints"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    
    checkpoint_name = Column(String, nullable=False)
    step = Column(Integer, nullable=False)
    epoch = Column(Integer, nullable=True)
    
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=True)  # bytes
    
    loss = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=get_local_now, index=True)
    
    # Relationships
    run = relationship("TrainingRun", back_populates="checkpoints")
    
    def to_dict(self):
        return {
            "id": self.id,
            "run_id": self.run_id,
            "checkpoint_name": self.checkpoint_name,
            "step": self.step,
            "epoch": self.epoch,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "loss": self.loss,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class TrainingSample(TrainingBase):
    """Sample image generated during training"""
    __tablename__ = "training_samples"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    
    step = Column(Integer, nullable=False)
    prompt = Column(Text, nullable=False)
    image_path = Column(String, nullable=False)
    
    created_at = Column(DateTime, default=get_local_now, index=True)
    
    # Relationships
    run = relationship("TrainingRun", back_populates="samples")
    
    def to_dict(self):
        return {
            "id": self.id,
            "run_id": self.run_id,
            "step": self.step,
            "prompt": self.prompt,
            "image_path": self.image_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
