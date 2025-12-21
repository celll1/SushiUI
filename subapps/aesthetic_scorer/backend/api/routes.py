"""
FastAPI routes for Aesthetic Scorer.
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import sys

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "backend"))

from subapps.aesthetic_scorer.backend.database import get_db
from subapps.aesthetic_scorer.backend.database.models import LatentRecord, AestheticModel
from subapps.aesthetic_scorer.backend.core.latent_generator import LatentGenerator
from subapps.aesthetic_scorer.backend.core.aesthetic_model import create_aesthetic_model
from subapps.aesthetic_scorer.backend.core.aesthetic_trainer import AestheticTrainer, create_dataloaders
from subapps.aesthetic_scorer.backend.utils.image_utils import decode_and_save_latent_pair, batch_decode_latents

# Import SushiUI database
from database.models import DatasetBase, Dataset as SushiUIDataset
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# SushiUI datasets.db connection
SUSHIUI_DB_PATH = Path(__file__).parent.parent.parent.parent.parent / "backend" / "datasets.db"
sushiui_engine = create_engine(f"sqlite:///{SUSHIUI_DB_PATH}", connect_args={"check_same_thread": False})
SushiUISessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sushiui_engine)


def get_sushiui_db():
    """Get SushiUI datasets.db session (read-only)."""
    db = SushiUISessionLocal()
    try:
        yield db
    finally:
        db.close()


router = APIRouter()


# ============================================================
# Request/Response Models
# ============================================================

class GenerateLatentsRequest(BaseModel):
    dataset_id: int
    model_path: str
    num_samples: Optional[int] = None
    timestep_range: tuple[float, float] = (0.0, 1.0)
    shuffle: bool = True


class ScoreLatentRequest(BaseModel):
    score: float  # 0.0-1.0


class TrainModelRequest(BaseModel):
    architecture: str = "LatentCNN"
    learning_rate: float = 1e-4
    num_epochs: int = 50
    batch_size: int = 16
    val_split: float = 0.1
    model_name: str = "aesthetic"


class DecodeLatentsRequest(BaseModel):
    record_ids: List[int]
    vae_path: str


# ============================================================
# Latent Generation
# ============================================================

@router.post("/generate_latents")
async def generate_latents(
    request: GenerateLatentsRequest,
    db: Session = Depends(get_db),
    sushiui_db: Session = Depends(get_sushiui_db),
):
    """
    Generate predicted latents from SushiUI dataset.

    Args:
        request: Generation parameters
        db: Aesthetic scorer database session
        sushiui_db: SushiUI datasets.db session (read-only)

    Returns:
        {
            "num_generated": int,
            "total_size_gb": float,
            "dataset_name": str,
        }
    """
    # Check if dataset exists in SushiUI
    dataset = sushiui_db.query(SushiUIDataset).filter(
        SushiUIDataset.id == request.dataset_id
    ).first()

    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {request.dataset_id} not found")

    # Initialize generator
    generator = LatentGenerator(
        model_path=request.model_path,
        device="cuda",
        weight_dtype="fp16",
        vae_dtype="fp16",
    )

    # Generate latents
    output_dir = Path("subapps/aesthetic_scorer/data/latents")
    generated_records = generator.generate_latents_from_dataset(
        sushiui_db_session=sushiui_db,
        dataset_id=request.dataset_id,
        num_samples=request.num_samples,
        timestep_range=request.timestep_range,
        output_dir=output_dir,
        save_mode="minimal",
        shuffle=request.shuffle,
    )

    # Insert records into aesthetic scorer database
    for record_data in generated_records:
        latent_record = LatentRecord(**record_data)
        db.add(latent_record)

    db.commit()

    # Calculate total size
    total_size_gb = sum(
        Path(r["filename"]).stat().st_size for r in generated_records
    ) / 1024**3

    return {
        "num_generated": len(generated_records),
        "total_size_gb": round(total_size_gb, 2),
        "dataset_name": dataset.name,
    }


# ============================================================
# Latent Records
# ============================================================

@router.get("/latents")
async def get_latents(
    skip: int = 0,
    limit: int = 100,
    scored_only: bool = False,
    unscored_only: bool = False,
    db: Session = Depends(get_db),
):
    """
    Get latent records.

    Args:
        skip: Offset
        limit: Number of records to return
        scored_only: Return only scored records
        unscored_only: Return only unscored records
        db: Database session

    Returns:
        {
            "records": List[LatentRecord],
            "total": int,
        }
    """
    query = db.query(LatentRecord)

    if scored_only:
        query = query.filter(LatentRecord.is_scored == True)
    elif unscored_only:
        query = query.filter(LatentRecord.is_scored == False)

    total = query.count()
    records = query.offset(skip).limit(limit).all()

    return {
        "records": [r.to_dict() for r in records],
        "total": total,
    }


@router.get("/latents/{record_id}")
async def get_latent(
    record_id: int,
    db: Session = Depends(get_db),
):
    """Get single latent record by ID."""
    record = db.query(LatentRecord).filter(LatentRecord.id == record_id).first()

    if not record:
        raise HTTPException(status_code=404, detail=f"Latent record {record_id} not found")

    return record.to_dict()


@router.post("/latents/{record_id}/score")
async def score_latent(
    record_id: int,
    request: ScoreLatentRequest,
    db: Session = Depends(get_db),
):
    """
    Update user score for latent record.

    Args:
        record_id: Latent record ID
        request: Score (0.0-1.0, where 0=best, 1=worst)
        db: Database session

    Returns:
        Updated LatentRecord
    """
    if not (0.0 <= request.score <= 1.0):
        raise HTTPException(status_code=400, detail="Score must be between 0.0 and 1.0")

    record = db.query(LatentRecord).filter(LatentRecord.id == record_id).first()

    if not record:
        raise HTTPException(status_code=404, detail=f"Latent record {record_id} not found")

    record.user_score = request.score
    record.is_scored = True

    db.commit()
    db.refresh(record)

    return record.to_dict()


@router.get("/latents/stats")
async def get_latent_stats(db: Session = Depends(get_db)):
    """Get statistics about latent records."""
    total = db.query(LatentRecord).count()
    scored = db.query(LatentRecord).filter(LatentRecord.is_scored == True).count()
    unscored = total - scored

    return {
        "total": total,
        "scored": scored,
        "unscored": unscored,
        "scored_percentage": round(scored / total * 100, 2) if total > 0 else 0,
    }


# ============================================================
# Image Decoding
# ============================================================

@router.post("/decode_latents")
async def decode_latents(
    request: DecodeLatentsRequest,
    db: Session = Depends(get_db),
):
    """
    Decode latents to images using VAE.

    Args:
        request: Record IDs and VAE path
        db: Database session

    Returns:
        {
            "num_decoded": int,
            "errors": List[int],
        }
    """
    from diffusers import AutoencoderKL
    import torch

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        request.vae_path,
        subfolder="vae",
        torch_dtype=torch.float16,
    )

    # Get records
    records = db.query(LatentRecord).filter(
        LatentRecord.id.in_(request.record_ids)
    ).all()

    if not records:
        raise HTTPException(status_code=404, detail="No records found")

    # Decode
    output_dir = Path("subapps/aesthetic_scorer/data/images")
    results = batch_decode_latents(
        vae=vae,
        record_ids=[r.id for r in records],
        latent_files=[Path(r.filename) for r in records],
        output_dir=output_dir,
        device="cuda",
    )

    # Update database
    errors = []
    for record in records:
        if record.id in results:
            true_path, pred_path = results[record.id]
            record.true_latent_image_path = true_path
            record.predicted_latent_image_path = pred_path
        else:
            errors.append(record.id)

    db.commit()

    return {
        "num_decoded": len(results),
        "errors": errors,
    }


# ============================================================
# Aesthetic Model Training
# ============================================================

@router.post("/train_model")
async def train_model(
    request: TrainModelRequest,
    db: Session = Depends(get_db),
):
    """
    Train aesthetic scoring model.

    Args:
        request: Training configuration
        db: Database session

    Returns:
        {
            "model_id": int,
            "model_path": str,
            "train_loss": float,
            "val_loss": float,
        }
    """
    import torch

    # Get scored records
    scored_records = db.query(LatentRecord).filter(
        LatentRecord.is_scored == True
    ).all()

    if len(scored_records) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough scored samples (found {len(scored_records)}, need at least 10)"
        )

    print(f"[Train] Using {len(scored_records)} scored samples")

    # Create model
    model = create_aesthetic_model(architecture=request.architecture)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        scored_records=[r.to_dict() for r in scored_records],
        batch_size=request.batch_size,
        val_split=request.val_split,
        num_workers=0,
    )

    # Train
    trainer = AestheticTrainer(
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=request.learning_rate,
    )

    save_dir = Path("subapps/aesthetic_scorer/models")
    summary = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=request.num_epochs,
        save_dir=save_dir,
        model_name=request.model_name,
        save_every=10,
    )

    # Save model record to database
    model_path = str(save_dir / f"{request.model_name}_best.safetensors")

    aesthetic_model = AestheticModel(
        name=request.model_name,
        version="1.0.0",
        architecture=request.architecture,
        parameters=model.get_config(),
        training_config={
            "learning_rate": request.learning_rate,
            "num_epochs": request.num_epochs,
            "batch_size": request.batch_size,
            "val_split": request.val_split,
        },
        num_scored_samples=len(scored_records),
        num_epochs=request.num_epochs,
        train_loss=summary["final_train_loss"],
        val_loss=summary["final_val_loss"],
        model_path=model_path,
        is_active=True,
    )

    # Deactivate other models
    db.query(AestheticModel).update({"is_active": False})

    db.add(aesthetic_model)
    db.commit()
    db.refresh(aesthetic_model)

    return {
        "model_id": aesthetic_model.id,
        "model_path": model_path,
        "train_loss": summary["final_train_loss"],
        "val_loss": summary["final_val_loss"],
    }


@router.get("/models")
async def get_models(db: Session = Depends(get_db)):
    """Get all aesthetic models."""
    models = db.query(AestheticModel).order_by(AestheticModel.created_at.desc()).all()
    return {"models": [m.to_dict() for m in models]}


@router.get("/models/{model_id}")
async def get_model(model_id: int, db: Session = Depends(get_db)):
    """Get aesthetic model by ID."""
    model = db.query(AestheticModel).filter(AestheticModel.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    return model.to_dict()


@router.post("/models/{model_id}/activate")
async def activate_model(model_id: int, db: Session = Depends(get_db)):
    """Set model as active."""
    model = db.query(AestheticModel).filter(AestheticModel.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    # Deactivate all models
    db.query(AestheticModel).update({"is_active": False})

    # Activate this model
    model.is_active = True
    db.commit()

    return {"message": f"Model {model_id} activated"}
