"""Style embedding and similarity endpoints (design §12)."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from api.error_handlers import NotFoundError, ValidationError
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
