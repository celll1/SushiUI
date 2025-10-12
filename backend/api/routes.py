from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
import os
from PIL import Image
import io

from database import get_db
from database.models import GeneratedImage
from core.pipeline import pipeline_manager
from utils import save_image_with_metadata, create_thumbnail
from config.settings import settings

router = APIRouter()

# Pydantic models for requests
class GenerationParams(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    steps: int = 20
    cfg_scale: float = 7.0
    sampler: str = "euler_a"
    seed: int = -1
    width: int = 512
    height: int = 512
    model: str = ""

class Txt2ImgRequest(GenerationParams):
    pass

class Img2ImgRequest(GenerationParams):
    denoising_strength: float = 0.75

# Routes
@router.post("/generate/txt2img")
async def generate_txt2img(request: Txt2ImgRequest, db: Session = Depends(get_db)):
    """Generate image from text"""
    try:
        # Generate image
        params = request.dict()
        print(f"Generation params: {params}")

        image = pipeline_manager.generate_txt2img(params)

        # Save image with metadata
        filename = save_image_with_metadata(image, params, "txt2img")

        # Create thumbnail
        image_path = os.path.join(settings.outputs_dir, filename)
        create_thumbnail(image_path)

        # Save to database
        db_image = GeneratedImage(
            filename=filename,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            model_name=request.model,
            sampler=request.sampler,
            steps=request.steps,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
            width=request.width,
            height=request.height,
            generation_type="txt2img",
            parameters=params,
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        return {"success": True, "image": db_image.to_dict()}

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"Error generating image: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate/img2img")
async def generate_img2img(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    steps: int = Form(20),
    cfg_scale: float = Form(7.0),
    denoising_strength: float = Form(0.75),
    seed: int = Form(-1),
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Generate image from image"""
    try:
        # Load input image
        image_data = await image.read()
        init_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Generate image
        params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "denoising_strength": denoising_strength,
            "seed": seed,
        }
        result_image = pipeline_manager.generate_img2img(params, init_image)

        # Save image
        filename = save_image_with_metadata(result_image, params, "img2img")
        image_path = os.path.join(settings.outputs_dir, filename)
        create_thumbnail(image_path)

        # Save to database
        db_image = GeneratedImage(
            filename=filename,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_name="",
            sampler="",
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            width=result_image.width,
            height=result_image.height,
            generation_type="img2img",
            parameters=params,
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        return {"success": True, "image": db_image.to_dict()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/images")
async def get_images(
    skip: int = 0,
    limit: int = 50,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get list of generated images"""
    query = db.query(GeneratedImage)

    if search:
        query = query.filter(GeneratedImage.prompt.contains(search))

    images = query.order_by(GeneratedImage.created_at.desc()).offset(skip).limit(limit).all()
    return {"images": [img.to_dict() for img in images]}

@router.get("/images/{image_id}")
async def get_image(image_id: int, db: Session = Depends(get_db)):
    """Get single image details"""
    image = db.query(GeneratedImage).filter(GeneratedImage.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return image.to_dict()

@router.delete("/images/{image_id}")
async def delete_image(image_id: int, db: Session = Depends(get_db)):
    """Delete an image"""
    image = db.query(GeneratedImage).filter(GeneratedImage.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Delete files
    image_path = os.path.join(settings.outputs_dir, image.filename)
    thumb_path = os.path.join(settings.thumbnails_dir, image.filename)

    if os.path.exists(image_path):
        os.remove(image_path)
    if os.path.exists(thumb_path):
        os.remove(thumb_path)

    db.delete(image)
    db.commit()

    return {"success": True}

@router.get("/models")
async def get_models():
    """Get list of available models"""
    models = []
    if os.path.exists(settings.models_dir):
        for item in os.listdir(settings.models_dir):
            item_path = os.path.join(settings.models_dir, item)
            if os.path.isdir(item_path):
                # Diffusers format directory
                models.append({
                    "name": item,
                    "path": item_path,
                    "type": "diffusers",
                    "source_type": "diffusers"
                })
            elif item.endswith('.safetensors'):
                # Safetensors file
                file_size = os.path.getsize(item_path) / (1024**3)  # GB
                models.append({
                    "name": item.replace('.safetensors', ''),
                    "path": item_path,
                    "type": "safetensors",
                    "source_type": "safetensors",
                    "size_gb": round(file_size, 2)
                })
    return {"models": models}

@router.post("/models/load")
async def load_model(
    source_type: str = Form(...),
    source: str = Form(...),
    revision: Optional[str] = Form(None)
):
    """Load a model from various sources"""
    try:
        kwargs = {}
        if revision:
            kwargs["revision"] = revision

        pipeline_manager.load_model(
            source_type=source_type,
            source=source,
            pipeline_type="txt2img",
            **kwargs
        )

        return {
            "success": True,
            "message": "Model loaded successfully",
            "model_info": pipeline_manager.current_model_info
        }
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"Error loading model: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/upload")
async def upload_model(file: UploadFile = File(...)):
    """Upload a safetensors model file"""
    if not file.filename.endswith('.safetensors'):
        raise HTTPException(status_code=400, detail="Only .safetensors files are supported")

    try:
        os.makedirs(settings.models_dir, exist_ok=True)
        file_path = os.path.join(settings.models_dir, file.filename)

        # Save uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        return {
            "success": True,
            "message": "Model uploaded successfully",
            "filename": file.filename,
            "path": file_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/current")
async def get_current_model():
    """Get currently loaded model info"""
    if pipeline_manager.current_model_info:
        return {
            "loaded": True,
            "model_info": pipeline_manager.current_model_info
        }
    else:
        return {"loaded": False}
