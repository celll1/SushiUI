"""Style embedding and similarity endpoints (design §12)."""

from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from api.error_handlers import NotFoundError, ValidationError
from api.param_defaults import STYLE_SDXL_GENERATION_DEFAULTS as _G
from core.style_encoder.style_inference import decode_style_image, style_inference_service


router = APIRouter()


class StyleEmbedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    encoder_path: str
    image_base64: str


class StyleSimilarityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    encoder_path: str
    image_a_base64: str
    image_b_base64: str


class StyleSDXLGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sdxl_model_path: str
    encoder_path: str
    adapter_path: str
    reference_image_base64: str
    prompt: str
    negative_prompt: str = _G["negative_prompt"]
    width: int = Field(_G["width"], ge=256, le=2048)
    height: int = Field(_G["height"], ge=256, le=2048)
    steps: int = Field(_G["steps"], ge=1, le=150)
    guidance_scale: float = Field(_G["guidance_scale"], ge=0, le=30)
    style_strength: float = Field(_G["style_strength"], ge=0, le=2)
    seed: int = Field(_G["seed"], ge=0, le=2**31 - 1)


@router.post("/style/embed")
def style_embed(request: StyleEmbedRequest):
    try:
        image = decode_style_image(request.image_base64)
        embedding = style_inference_service.embed(request.encoder_path, image)
    except FileNotFoundError as exc:
        raise NotFoundError("Style encoder artifact not found", detail=str(exc)) from exc
    except (ValueError, KeyError) as exc:
        raise ValidationError("Invalid style embedding request", detail=str(exc)) from exc
    return {"embedding": embedding, "dim": len(embedding)}


@router.post("/style/similarity")
def style_similarity(request: StyleSimilarityRequest):
    try:
        first = decode_style_image(request.image_a_base64)
        second = decode_style_image(request.image_b_base64)
        similarity = style_inference_service.similarity(request.encoder_path, first, second)
    except FileNotFoundError as exc:
        raise NotFoundError("Style encoder artifact not found", detail=str(exc)) from exc
    except (ValueError, KeyError) as exc:
        raise ValidationError("Invalid style similarity request", detail=str(exc)) from exc
    return {"cosine_similarity": similarity}


@router.post("/style/sdxl-generate")
async def style_sdxl_generate(request: StyleSDXLGenerateRequest):
    if request.width % 64 or request.height % 64:
        raise ValidationError("Invalid SDXL resolution", detail="width and height must be multiples of 64")
    if not math.isfinite(request.guidance_scale) or not math.isfinite(request.style_strength):
        raise ValidationError("Invalid style generation strength", detail="strength values must be finite")
    from core.model_state_coordinator import ModelStateBusyError, model_state_coordinator
    from core.gpu_coordinator import gpu_coordinator
    from core.style_encoder.sdxl_ip_inference import sdxl_style_generation_service
    try:
        reference = decode_style_image(request.reference_image_base64)
        with model_state_coordinator.mutation("SDXL style generation"):
            async with gpu_coordinator.generation_slot(estimated_peak_gb=24.0, timeout=60.0):
                encoded = await run_in_threadpool(
                    sdxl_style_generation_service.generate,
                    model_path=request.sdxl_model_path, encoder_path=request.encoder_path,
                    adapter_path=request.adapter_path, reference=reference,
                    prompt=request.prompt, negative_prompt=request.negative_prompt,
                    width=request.width, height=request.height, steps=request.steps,
                    guidance_scale=request.guidance_scale, style_strength=request.style_strength,
                    seed=request.seed,
                )
    except ModelStateBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise NotFoundError("Style generation artifact not found", detail=str(exc)) from exc
    except (ValueError, KeyError) as exc:
        raise ValidationError("Invalid style generation request", detail=str(exc)) from exc
    return {"image_base64": encoded, "seed": request.seed}
