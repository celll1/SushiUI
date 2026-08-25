from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, Text, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from typing import Any, Dict
import uuid

# Read-time redaction of filesystem paths in identity labels (see
# ``GeneratedImage.to_dict``). stdlib-only helper, no API-layer import.
from utils.path_redaction import display_name_for_path, redact_paths

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
    cache_dir = Column(String, nullable=True)  # Custom cache directory (default: backend/cache)
    training_dir = Column(String, nullable=True)  # Custom training output directory (default: training)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Generation settings
    # Inpaint method: False = mask blending (default, same as Z-Image/FLUX.2)
    #                 True = construct dedicated 9ch inpaint model (legacy SD/SDXL method)
    inpaint_use_dedicated_model = Column(Boolean, default=False)
    # Upper bound for the video frame-count SLIDER TRACK (frontend
    # VideoFrameCountSlider), not a value cap: the paired number box is never
    # bounded by this and always accepts a longer length. NULL (the default)
    # means unset -- the slider keeps its built-in track reach
    # (TRAINED_RANGE_SLIDER_HEADROOM / UNCAPPED_FRAME_SLIDER_CEILING in that
    # component), so a user who never opens Settings sees no change.
    video_frame_slider_max = Column(Integer, nullable=True)
    # General user-override mechanism for slider/number-input UPPER BOUNDS
    # (see backend/api/param_defaults.py's PARAM_BOUNDS registry for the
    # eligibility rule + the full list of overridable bounds). ONE JSON
    # column holding {bound_name: value} for every bound the user has
    # overridden -- deliberately NOT one column per bound, so adding a new
    # overridable bound is a PARAM_BOUNDS registry entry, never a migration.
    # A key absent from this dict means "use PARAM_BOUNDS[key]['builtin']".
    # `video_frame_slider_max` above predates this mechanism and is
    # deliberately NOT folded into it -- a data migration to move that one
    # column's value into this dict is not worth it for a single field, so
    # the two coexist: video_frame_slider_max stays its own legacy column,
    # and every bound added from here on lives in slider_bounds instead.
    slider_bounds = Column(JSON, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "model_dirs": self.model_dirs or [],
            "lora_dirs": self.lora_dirs or [],
            "controlnet_dirs": self.controlnet_dirs or [],
            "cache_dir": self.cache_dir,
            "training_dir": self.training_dir,
            "inpaint_use_dedicated_model": self.inpaint_use_dedicated_model if self.inpaint_use_dedicated_model is not None else False,
            "video_frame_slider_max": self.video_frame_slider_max,
            "slider_bounds": self.slider_bounds or {},
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

    def to_summary_dict(self):
        """Light per-row payload for the gallery LIST endpoint (``GET /images``).

        Omits the full ``parameters`` JSON and ``mask_data`` — both can run
        into multi-MB territory per row (embedded ControlNet/style-reference
        base64, inpaint masks) and are never rendered by the grid cell, only
        by the detail view (which uses ``to_dict()`` via ``GET /images/{id}``
        instead). Fields below are exactly what the frontend grid reads off a
        list item: cell rendering (id/filename/is_video/is_audio/
        generation_type/prompt in ``ImageList.tsx``), page-local tag search/
        suggestions (prompt/negative_prompt), and the "click a ControlNet/
        source image hash to jump to it within the current page" lookup
        (image_hash) in ``ImageGrid.tsx``.
        """
        is_video = bool(self.parameters.get("is_video")) if self.parameters else False
        is_audio = bool(self.parameters.get("is_audio")) if self.parameters else False
        return {
            "id": self.id,
            "filename": self.filename,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "generation_type": self.generation_type,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_favorite": self.is_favorite,
            "image_hash": self.image_hash,
            "is_video": is_video,
            "is_audio": is_audio,
        }

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

            # Post-decode options
            if self.parameters.get("color_flatten_strength"):
                result["color_flatten_strength"] = str(self.parameters["color_flatten_strength"])
            if self.parameters.get("vae_drift_correction"):
                result["vae_drift_correction"] = str(self.parameters["vae_drift_correction"])
            if self.parameters.get("flatten_in_loop"):
                result["flatten_in_loop"] = str(self.parameters["flatten_in_loop"])
                if "flatten_in_loop_last_steps" in self.parameters:
                    result["flatten_in_loop_last_steps"] = str(self.parameters["flatten_in_loop_last_steps"])
                if "flatten_in_loop_min_region" in self.parameters:
                    result["flatten_in_loop_min_region"] = str(self.parameters["flatten_in_loop_min_region"])

            # Upscale parameters (generation_type == "upscale")
            if "upscaler_backend" in self.parameters:
                result["upscaler_backend"] = self.parameters["upscaler_backend"]
                if self.parameters.get("upscaler_model"):
                    result["upscaler_model"] = self.parameters["upscaler_model"]
                if self.parameters.get("upscaler_model_hash"):
                    result["upscaler_model_hash"] = self.parameters["upscaler_model_hash"]
                if "scale_factor" in self.parameters:
                    result["scale_factor"] = str(self.parameters["scale_factor"])
                if "pil_resample" in self.parameters:
                    result["pil_resample"] = self.parameters["pil_resample"]
                if "tile_size" in self.parameters:
                    result["tile_size"] = str(self.parameters["tile_size"])
                if "tile_overlap" in self.parameters:
                    result["tile_overlap"] = str(self.parameters["tile_overlap"])
                if "rtx_vsr_quality" in self.parameters:
                    result["rtx_vsr_quality"] = self.parameters["rtx_vsr_quality"]
                if "diffusion_denoising_strength" in self.parameters:
                    result["diffusion_denoising_strength"] = str(self.parameters["diffusion_denoising_strength"])
                if "diffusion_pre_upscale_mode" in self.parameters:
                    result["diffusion_pre_upscale_mode"] = self.parameters["diffusion_pre_upscale_mode"]

            # Video parameters (generation_type == "txt2vid" / "img2vid")
            if self.parameters.get("is_video"):
                result["is_video"] = True
                # Only set when the master was written with FFV1 (video_lossless):
                # an H.264 proxy the gallery <video> element plays instead of the
                # (browser-undecodable) master. Download/"Send to" stay on `filename`.
                if self.parameters.get("preview_filename"):
                    result["preview_filename"] = self.parameters["preview_filename"]
                if "num_frames" in self.parameters:
                    result["num_frames"] = str(self.parameters["num_frames"])
                if "fps" in self.parameters:
                    result["fps"] = str(self.parameters["fps"])
                if "duration" in self.parameters:
                    result["duration"] = str(self.parameters["duration"])
                if "audio_enable" in self.parameters:
                    result["audio_enable"] = str(self.parameters["audio_enable"])
                if "guidance_scale" in self.parameters:
                    result["guidance_scale"] = str(self.parameters["guidance_scale"])
                if "num_inference_steps" in self.parameters:
                    result["num_inference_steps"] = str(self.parameters["num_inference_steps"])
                # Video chain provenance (design sec.13): which chain/plan/
                # segment produced this row. Present only on a chained
                # generation; the root prompt and canonical timeline are NOT
                # here by design -- the two hashes reference the manifest that
                # holds them once.
                if self.parameters.get("chain_id"):
                    for key in ("chain_id",
                                "chain_manifest_version",
                                "chain_plan_hash",
                                "chain_segment_index",
                                "chain_segment_count",
                                "chain_global_frame_start",
                                "chain_global_frame_end",
                                "chain_context_mode",
                                "chain_root_prompt_hash"):
                        if self.parameters.get(key) is not None:
                            result[key] = str(self.parameters[key])

            # Audio parameters (generation_type == "txt2aud")
            if self.parameters.get("is_audio"):
                result["is_audio"] = True
                if "duration" in self.parameters:
                    result["duration"] = str(self.parameters["duration"])
                if "sample_rate" in self.parameters:
                    result["sample_rate"] = str(self.parameters["sample_rate"])
                if "audio_duration" in self.parameters:
                    result["audio_duration"] = str(self.parameters["audio_duration"])
                if "inference_steps" in self.parameters:
                    result["inference_steps"] = str(self.parameters["inference_steps"])
                if "guidance_scale" in self.parameters:
                    result["guidance_scale"] = str(self.parameters["guidance_scale"])
                # MiniMax Music 3's differently-named equivalents (per-chunk
                # step count, flow-stage CFG) -- ACE-Step's `inference_steps`/
                # `guidance_scale` above are the ACE-Step turbo sampler's OWN
                # fields and are never populated for a Music3 row, so both
                # pairs are surfaced independently rather than one overwriting
                # the other.
                if "num_inference_steps" in self.parameters:
                    result["num_inference_steps"] = str(self.parameters["num_inference_steps"])
                if "flow_guidance_scale" in self.parameters:
                    result["flow_guidance_scale"] = str(self.parameters["flow_guidance_scale"])

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

            # Feature-degradation notices recorded during generation (list of
            # {"code", "message"} dicts) — kept as a list, not stringified.
            if "effective_warnings" in self.parameters:
                result["effective_warnings"] = self.parameters["effective_warnings"]

            # FLUX.2 Image Edit: Reference images
            if "ref_images" in self.parameters:
                result["ref_images"] = self.parameters["ref_images"]

            # Vision Encoder metadata.
            # Defensive gate: only surface VE info when the row actually used
            # reference images. Legacy/stray rows may carry vision_encoder_* from
            # the sticky VE session without any ref_images; suppress those.
            if self.parameters.get("ref_images"):
                if "vision_encoder_name" in self.parameters:
                    result["vision_encoder_name"] = self.parameters["vision_encoder_name"]
                if "vision_encoder_hash" in self.parameters:
                    result["vision_encoder_hash"] = self.parameters["vision_encoder_hash"]

            # VAE identity. The VAE always affects the decoded output, so surface it
            # unconditionally when recorded (no ref-image gate, unlike the VE above).
            #
            # Redacted at READ time: rows written between the override-label fix
            # and the privacy fix stored the override's ABSOLUTE PATH in
            # ``vae_name``. Rows are never rewritten (the stored value stays as
            # the audit trail, and ``vae_override_path`` in the returned raw
            # ``parameters`` still gives the frontend the full local path for
            # restore); only what is presented as the identity label is reduced
            # to the same display name a new row would carry. Nothing is lost:
            # ``vae_dec_IL02_v1_vae`` and ``vae_dec_IL02_v1_vae_noema`` remain
            # distinct, and the parenthesized provenance is preserved verbatim.
            if "vae_name" in self.parameters:
                result["vae_name"] = redact_paths(self.parameters["vae_name"])

            # Legacy rows (written before extract_vae_info consulted the override)
            # recorded the CHECKPOINT's VAE in vae_name even when a per-generation
            # VAE override produced the image. The override path was recorded all
            # along, so derive the label from it here rather than reporting the
            # wrong VAE. Rows written after the fix already carry the override in
            # vae_name and are left untouched. Read-only: no row is rewritten.
            #
            # Gated on evidence the override was APPLIED, never on the request:
            #   * ``vae_override_path`` is written by ``apply_overrides`` only.
            #     ``vae_path`` is the REQUESTED value, echoed into every row even
            #     when the override was dropped by the arch gate (pixel-space
            #     minit2i), never consulted (video endpoints), or failed to load.
            #   * an apply failure is recorded as a ``vae_override_error`` warning,
            #     in which case the model's own VAE decoded the image.
            # ``vae_override_source`` is NOT used: it stores the loaded module's
            # ``config._name_or_path``, which a fine-tune export inherits from its
            # base VAE and which would therefore name the wrong VAE. The path is
            # the only field that identifies what was actually loaded.
            failed = any(
                (w or {}).get("code") == "vae_override_error"
                for w in (self.parameters.get("effective_warnings") or [])
                if isinstance(w, dict)
            )
            override_path = self.parameters.get("vae_override_path")
            legacy_override = (
                bool(override_path)
                and not failed
                and not str(result.get("vae_name") or "").startswith("override: ")
            )
            if legacy_override:
                # Display name only, never the path — same rule as the producer
                # (``describe_vae_override``), so a legacy row and a new row read
                # identically. The sidecar provenance is not re-derived here: it
                # would mean touching the filesystem during a gallery read, and
                # the export it described may since have been overwritten by a
                # later step of the same run.
                result["vae_name"] = f"override: {display_name_for_path(override_path, strip_safetensors=True)}"

            # vae_hash on such a row is the hash of the model's own VAE, not of the
            # override that actually decoded the image -- suppress it rather than
            # pair a corrected name with a stale hash.
            if "vae_hash" in self.parameters and not legacy_override:
                result["vae_hash"] = self.parameters["vae_hash"]

            # Which MiniMax-H3 checkpoint (fl2va/ref2va) actually ran -- the
            # filename is the only thing that distinguishes them, so this is
            # what makes the row readable after either file is renamed.
            if "model_variant" in self.parameters:
                result["model_variant"] = self.parameters["model_variant"]

            # A merged ("hybrid") MiniMax-H3 DiT: which pair and which recipe
            # produced the row. `model_variant` alone says "hybrid" and nothing
            # more, and `model_hash` is the BASE file's, so every hybrid on one
            # base carries the same one -- these keys are the only thing that
            # tells two of them apart. Basenames and a digest, never paths
            # (record_model_variant sanitises them at the producer).
            for key in ("model_hybrid_base", "model_hybrid_overlay",
                        "model_hybrid_preset", "model_hybrid_block_range",
                        "model_hybrid_final_adaln_from_overlay",
                        "model_hybrid_digest", "model_hybrid_quantization"):
                if key in self.parameters:
                    result[key] = self.parameters[key]

            # Quantized-Linear GEMM path. Present only on rows produced by a
            # weight-only FP8 checkpoint (Ideogram 4 / Krea 2) or a weight-only
            # INT8 checkpoint (Krea 2 only, today); surfaced
            # verbatim, as the value is a mechanism label
            # ("w8a8_scaled_mm(tensorwise)" / "w8a8_int_mm(int_mm+fused)" /
            # "dequant" / "int8_dequant", possibly two joined with "+") with no
            # path in it. The key name is historical, not FP8-specific.
            if "fp8_gemm" in self.parameters:
                result["fp8_gemm"] = self.parameters["fp8_gemm"]

            # What the request asked for on the quantized-GEMM axis ("w8a8" /
            # "dequant"), when it asked. Omitted when null, i.e. when the
            # process-level value stood and the generation forced nothing.
            if self.parameters.get("quantized_gemm_mode"):
                result["quantized_gemm_mode"] = str(self.parameters["quantized_gemm_mode"])

        return result


class StudioRenderJob(GalleryBase):
    """Persistent state for a Studio timeline render.

    Render inputs are copied into a server-owned staging directory before the
    job is queued; ``input_dir`` is therefore an internal path, never a client
    supplied filename or URL.
    """
    __tablename__ = "studio_render_jobs"

    id = Column(String, primary_key=True)
    state = Column(String, nullable=False, default="queued", index=True)
    manifest = Column(JSON, nullable=False)
    input_dir = Column(String, nullable=False)
    progress = Column(Float, nullable=False, default=0.0)
    message = Column(String, nullable=True)
    gallery_image_id = Column(Integer, nullable=True)
    filename = Column(String, nullable=True)
    preview_filename = Column(String, nullable=True)
    error = Column(Text, nullable=True)
    # Feature-degradation notices (poster/thumbnail generation failures,
    # silent-audio output, cropped-edge fit_mode, ...) that do not fail the
    # job but the client should be told about. Mirrors the `warnings[]`
    # convention used by the live generation endpoints, but stored per-row
    # here since a render job has no live generation-id context.
    warnings = Column(JSON, nullable=True, default=list)
    created_at = Column(DateTime, default=get_local_now, index=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": True,
            "job_id": self.id,
            "state": self.state,
            "progress": float(self.progress or 0.0),
            "message": self.message,
            "gallery_image_id": self.gallery_image_id,
            "filename": self.filename,
            "preview_filename": self.preview_filename,
            "error": self.error,
            "warnings": list(self.warnings or []),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


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
    unique_id = Column(String, unique=True, index=True, nullable=False, default=lambda: str(uuid.uuid4()))  # UUID for cache directory naming
    name = Column(String, unique=True, index=True, nullable=False)
    path = Column(String, nullable=False)
    description = Column(Text, nullable=True)

    # Caption settings
    caption_suffixes = Column(JSON, default=list)
    default_caption_type = Column(String, default="tags")

    # Caption processing settings (for training)
    # These are stored in Dataset for reusability but applied per TrainingRun via dataset_configs
    caption_processing = Column(JSON, default=dict)  # {
    #   "caption_dropout_rate": 0.0,
    #   "token_dropout_rate": 0.0,
    #   "keep_tokens": 0,
    #   "shuffle_tokens": false,
    #   "shuffle_per_epoch": false,
    #   "shuffle_keep_first_n": 0,
    #   "tag_dropout_rate": 0.0,
    #   "tag_dropout_per_epoch": false,
    #   "tag_dropout_keep_first_n": 0,
    #   "tag_dropout_category_rates": {},
    #   "tag_dropout_exclude_person_count": false
    # }

    # Image pair settings
    image_suffixes = Column(JSON, default=list)

    # Reference image settings (for training with reference images)
    # When scanning, files with these suffixes are treated as reference images
    # Example: ["_source", "_ref"] -> image_source.png is reference for image_target.png
    reference_suffixes = Column(JSON, default=list)  # ["_source", "_ref"]
    target_suffixes = Column(JSON, default=list)  # ["_target"] - main training images
    caption_suffixes_for_reference = Column(JSON, default=list)  # ["_instruction"] - captions for target images

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

    # Tag statistics: {"tag": {"category": "...", "count": N}}
    tag_statistics = Column(JSON, default=dict)

    # Timestamps
    created_at = Column(DateTime, default=get_local_now, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_scanned_at = Column(DateTime, nullable=True)

    # Relationships (within datasets.db only)
    items = relationship("DatasetItem", back_populates="dataset", cascade="all, delete-orphan")
    # Note: TrainingRun is in a separate database (training.db)

    def to_dict(self, include_tag_statistics: bool = True):
        """Serialise to plain dict.

        ``include_tag_statistics=False`` omits the ``tag_statistics`` field —
        a potentially-megabyte JSON blob (per-tag counts across the full
        vocabulary).  Used by the dataset *list* endpoint which doesn't
        render per-tag stats; the detail endpoint keeps it on by default.
        Callers passing False should typically also ``defer(Dataset.
        tag_statistics)`` on the underlying query so the column isn't even
        read from disk.
        """
        out = {
            "id": self.id,
            "unique_id": self.unique_id,
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "caption_suffixes": self.caption_suffixes or [],
            "default_caption_type": self.default_caption_type,
            "caption_processing": self.caption_processing or {},
            "image_suffixes": self.image_suffixes or [],
            "reference_suffixes": self.reference_suffixes or [],
            "target_suffixes": self.target_suffixes or [],
            "caption_suffixes_for_reference": self.caption_suffixes_for_reference or [],
            "recursive": self.recursive,
            "max_depth": self.max_depth,
            "file_extensions": self.file_extensions or [],
            "read_exif": self.read_exif,
            "exif_caption_fields": self.exif_caption_fields,
            "total_items": self.total_items,
            "total_captions": self.total_captions,
            "total_tags": self.total_tags,
            "has_tags_captions": (self.total_tags or 0) > 0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_scanned_at": self.last_scanned_at.isoformat() if self.last_scanned_at else None,
        }
        if include_tag_statistics:
            out["tag_statistics"] = self.tag_statistics or {}
        return out


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
            # For item_type="video", per-clip metadata (video_path, fps,
            # num_frames, duration, width, height, codec) is stored in the
            # reused exif_data JSON column and surfaced here as video_meta.
            "video_meta": self.exif_data if self.item_type == "video" else None,
            # For item_type="audio", per-clip metadata (sample_rate, duration,
            # channels) is stored in the reused exif_data JSON column and
            # surfaced here as audio_meta (mirrors video_meta above).
            "audio_meta": self.exif_data if self.item_type == "audio" else None,
            # For item_type="video"/"audio", the scanner extracts a poster
            # frame / waveform PNG and writes it via create_thumbnail() keyed
            # by base_name (see routes.py dataset scan +
            # utils/image_utils.py create_thumbnail), published at the
            # /thumbnails static mount (backend/main.py).
            "thumbnail_url": f"/thumbnails/{self.base_name}.png" if self.item_type in ("video", "audio") else None,
            "total_captions": self.total_captions,
            "total_tags": self.total_tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class CaptionProcessingPreset(DatasetBase):
    """Preset for caption processing settings (reusable across datasets)"""
    __tablename__ = "caption_processing_presets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)

    # Caption processing configuration (same format as Dataset.caption_processing)
    config = Column(JSON, nullable=False)  # {
    #   "caption_dropout_rate": 0.0,
    #   "token_dropout_rate": 0.0,
    #   "keep_tokens": 0,
    #   "shuffle_tokens": false,
    #   "shuffle_per_epoch": false,
    #   "shuffle_keep_first_n": 0,
    #   "shuffle_tag_groups": [],
    #   "shuffle_groups_together": false,
    #   "tag_group_dir": "taglist",
    #   "exclude_person_count_from_shuffle": false,
    #   "tag_dropout_rate": 0.0,
    #   "tag_dropout_per_epoch": false,
    #   "tag_dropout_keep_first_n": 0,
    #   "tag_dropout_category_rates": {},
    #   "tag_dropout_exclude_person_count": false
    # }

    # Timestamps
    created_at = Column(DateTime, default=get_local_now)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "config": self.config or {},
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

    # Tag data with categories (for per-epoch shuffle/dropout optimization)
    # JSON format: [{"tag": "1girl", "category": "General"}, {"tag": "long_hair", "category": "General"}, ...]
    tag_data = Column(Text, nullable=True)  # Stored as JSON string

    # Caption format detection (for auto-handling tags vs natural language)
    field_category = Column(String, default="training")  # "training" | "metadata"
    is_tags_format = Column(Boolean, default=False)  # True if Danbooru tags format
    tag_match_rate = Column(Float, default=0.0)  # 0.0-1.0 (percentage of tokens matching taglist)

    # Metadata
    language = Column(String, nullable=True)
    # NOTE: no index on `source` — it is effectively single-valued ("file" for
    # every scanned caption), so an index on it is non-selective. Worse, without
    # table stats SQLite would pick that useless index for queries like
    # (item_id=? AND source=? AND caption_type IN (...)), turning a ~11-row
    # item_id lookup into a full 4M-row scan (~6s/query). Filter on item_id (which
    # IS indexed) instead.
    source = Column(String, default="manual")
    source_field = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=get_local_now)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    item = relationship("DatasetItem", back_populates="captions")

    def to_dict(self):
        result = {
            "id": self.id,
            "item_id": self.item_id,
            "caption_type": self.caption_type,
            "caption_subtype": self.caption_subtype,
            "content": self.content,
            "field_category": self.field_category,
            "is_tags_format": self.is_tags_format,
            "tag_match_rate": self.tag_match_rate,
            "language": self.language,
            "source": self.source,
            "source_field": self.source_field,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

        # Parse tag_data if present
        if self.tag_data:
            import json
            try:
                result["tag_data"] = json.loads(self.tag_data)
            except:
                result["tag_data"] = None
        else:
            result["tag_data"] = None

        return result


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
    dataset_id = Column(Integer, nullable=True, index=True)  # Deprecated - use dataset_configs instead
    dataset_configs = Column(JSON, nullable=True)  # List of {dataset_id, caption_types, filters}
    run_id = Column(String, unique=True, nullable=False, index=True, default=lambda: str(uuid.uuid4()))  # Unique ID (UUID)

    # Run identification
    run_name = Column(String, unique=True, index=True, nullable=False)
    training_method = Column(String, nullable=False, index=True)  # 'lora', 'full_finetune'
    base_model_path = Column(String, nullable=False)
    
    # Configuration
    config_yaml = Column(Text)  # Full ai-toolkit YAML config
    
    # Status
    status = Column(String, default="pending", index=True)  # 'pending', 'running', 'paused', 'completed', 'failed'
    progress = Column(Float, default=0.0)  # 0.0 - 100.0 (training phase progress)
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, nullable=False)

    # Phase tracking (for detailed progress during startup)
    phase = Column(String, default="initializing")  # 'initializing', 'latent_cache', 'text_encoder_cache', 'training'
    phase_progress = Column(Float, default=0.0)  # 0.0 - 100.0 (current phase progress)
    phase_detail = Column(String, nullable=True)  # Detailed status message (e.g., "Processing 500/1000 images")

    # Performance metrics
    loss = Column(Float, nullable=True)
    learning_rate = Column(Float, nullable=True)
    
    # Output
    output_dir = Column(String, nullable=False)
    checkpoint_paths = Column(JSON, default=list)  # List of checkpoint file paths
    
    # Logs
    log_file = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    # Structured notices the trainer emitted (settings overridden or ignored).
    # [{level, code, message}], capped by
    # core.training.training_events.MAX_PERSISTED_WARNINGS_PER_RUN so a run
    # cannot grow its own row without bound. Persisted because a notice a user
    # was not connected to see is no better than a console print.
    warnings = Column(JSON, default=list)

    # Timestamps
    created_at = Column(DateTime, default=get_local_now, index=True)
    started_at = Column(DateTime, nullable=True)
    last_resumed_at = Column(DateTime, nullable=True)  # Last resume time (for accurate ETA calculation)
    resumed_from_step = Column(Integer, nullable=True)  # Step at resume (for accurate ETA calculation)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships (within training.db only)
    # Note: dataset_id references datasets.db, but no ForeignKey constraint
    checkpoints = relationship("TrainingCheckpoint", back_populates="run", cascade="all, delete-orphan")
    samples = relationship("TrainingSample", back_populates="run", cascade="all, delete-orphan")
    
    def to_dict(self, summary: bool = False):
        """Serialise to dict.

        ``summary=True`` skips the fields that the run-list UI never reads:

        * ``config_yaml``      — full YAML text, several KB per run.
        * ``unet_lr`` / ``text_encoder_*_lr`` — derived by parsing the YAML;
          skipping config_yaml also lets us skip the per-row yaml.safe_load.
        * ``checkpoint_paths`` — requires a separate query through the
          ``checkpoints`` relationship (N+1 once that table has rows).
        * ``dataset_configs`` — JSON of which datasets / captions the run
          uses; only needed when editing or showing the detail view.

        ``GET /training/runs/{id}`` keeps the full payload by default so
        the detail view and edit flow still get everything in one call.
        """
        out: Dict[str, Any] = {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "training_method": self.training_method,
            "base_model_path": self.base_model_path,
            "status": self.status,
            "progress": self.progress,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "phase": self.phase,
            "phase_progress": self.phase_progress,
            "phase_detail": self.phase_detail,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "output_dir": self.output_dir,
            "log_file": self.log_file,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() + 'Z' if self.created_at else None,
            "started_at": self.started_at.isoformat() + 'Z' if self.started_at else None,
            "last_resumed_at": self.last_resumed_at.isoformat() + 'Z' if self.last_resumed_at else None,
            "resumed_from_step": self.resumed_from_step,
            "completed_at": self.completed_at.isoformat() + 'Z' if self.completed_at else None,
            "updated_at": self.updated_at.isoformat() + 'Z' if self.updated_at else None,
        }

        if summary:
            return out

        # Detail-only fields below.
        out["dataset_configs"] = self.dataset_configs
        out["config_yaml"]     = self.config_yaml
        out["warnings"]        = list(self.warnings or [])

        # Extract component-specific LRs from YAML config
        unet_lr = None
        text_encoder_lr = None
        text_encoder_1_lr = None
        text_encoder_2_lr = None
        if self.config_yaml:
            try:
                import yaml
                config = yaml.safe_load(self.config_yaml)
                train_config = config.get('config', {}).get('process', [{}])[0].get('train', {})
                unet_lr = train_config.get('unet_lr')
                text_encoder_lr = train_config.get('text_encoder_lr')
                text_encoder_1_lr = train_config.get('text_encoder_1_lr')
                text_encoder_2_lr = train_config.get('text_encoder_2_lr')
            except Exception:
                pass  # Silently fail if YAML parsing fails
        out["unet_lr"]            = unet_lr
        out["text_encoder_lr"]    = text_encoder_lr
        out["text_encoder_1_lr"]  = text_encoder_1_lr
        out["text_encoder_2_lr"]  = text_encoder_2_lr

        # Get checkpoints from DB (sorted by step descending = newest first)
        out["checkpoint_paths"] = [
            ckpt.file_path
            for ckpt in sorted(self.checkpoints, key=lambda x: x.step, reverse=True)
        ]
        return out


class TrainingPreset(TrainingBase):
    """Training configuration preset for quick reuse"""
    __tablename__ = "training_presets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)

    # Training method
    training_method = Column(String, nullable=False)  # 'lora' or 'full_finetune'

    # Configuration (JSON)
    config = Column(JSON, nullable=False)  # All training parameters except dataset and model path

    # Timestamps
    created_at = Column(DateTime, default=get_local_now, index=True)
    updated_at = Column(DateTime, default=get_local_now, onupdate=get_local_now)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "training_method": self.training_method,
            "config": self.config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
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


class TrainingMetrics(TrainingBase):
    """Training metrics (loss, learning_rate, grad_norm) logged during training.

    Features:
    - Dual logging: TensorBoard (for external tools) + DB (for fast queries)
    - UPSERT behavior: Same (run_id, step) will overwrite existing values
    - Indexed for fast filtering: WHERE run_id=? AND step>?
    """
    __tablename__ = "training_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    step = Column(Integer, nullable=False)
    # Epoch this step belongs to (for epoch-boundary markers in the metrics UI).
    epoch = Column(Integer, nullable=True)
    # Distinguishes successive resumes of the same run (0 = initial). Lets the UI
    # mark resume boundaries (and optionally split curves per resume later).
    resume_seq = Column(Integer, nullable=False, default=0)

    # Metrics
    loss = Column(Float, nullable=True)
    recon_loss = Column(Float, nullable=True)
    learning_rate = Column(Float, nullable=True)

    # Bespoke, arch/method-specific per-step scalar metrics (e.g. REPA alignment
    # for MiniT2I, generate-region MSE for outpaint ControlNet). Stored as a
    # {name: float} JSON dict so a new metric needs NO schema/threading change —
    # the trainer calls log_extra_metric(name, value) and the loss chart renders
    # it via core.training.metric_registry.EXTRA_METRIC_DEFS. Replaces the former
    # dedicated repa_loss column (backfilled by auto_migrate).
    extra_metrics = Column(JSON, nullable=True)

    # Gradient norms
    grad_norm = Column(Float, nullable=True)  # Total gradient norm (all parameters)
    grad_norm_text_encoder = Column(Float, nullable=True)  # Text encoder gradient norm (combined)
    grad_norm_text_encoder_1 = Column(Float, nullable=True)  # TE1 (CLIP ViT-L) gradient norm, SDXL only
    grad_norm_text_encoder_2 = Column(Float, nullable=True)  # TE2 (OpenCLIP ViT-bigG) gradient norm, SDXL only
    grad_norm_unet = Column(Float, nullable=True)  # U-Net/Transformer gradient norm
    grad_norm_vision_encoder = Column(Float, nullable=True)  # Vision Encoder gradient norm (SD/SDXL, optional)

    # Parameter change tracking (computed every N steps, CPU-side)
    # B: Step-wise update norm ||θ_t - θ_{t-K}||_F per component
    param_update_norm_unet = Column(Float, nullable=True)
    param_update_norm_te1 = Column(Float, nullable=True)
    param_update_norm_te2 = Column(Float, nullable=True)
    param_update_norm_ve = Column(Float, nullable=True)
    # C: Cumulative drift ||θ_t - θ_0||_F / ||θ_0||_F per component
    param_cumulative_drift_unet = Column(Float, nullable=True)
    param_cumulative_drift_te1 = Column(Float, nullable=True)
    param_cumulative_drift_te2 = Column(Float, nullable=True)
    param_cumulative_drift_ve = Column(Float, nullable=True)

    # Timestamp
    timestamp = Column(DateTime, default=get_local_now)

    # Composite unique constraint: (run_id, step) must be unique (UPSERT target)
    __table_args__ = (
        UniqueConstraint('run_id', 'step', name='uq_run_step'),
        Index('idx_run_step', 'run_id', 'step'),  # Composite index for fast queries
    )

    def to_dict(self):
        return {
            "step": self.step,
            "loss": self.loss,
            "recon_loss": self.recon_loss,
            "extra_metrics": self.extra_metrics or {},
            "learning_rate": self.learning_rate,
            "grad_norm": self.grad_norm,
            "grad_norm_text_encoder": self.grad_norm_text_encoder,
            "grad_norm_text_encoder_1": self.grad_norm_text_encoder_1,
            "grad_norm_text_encoder_2": self.grad_norm_text_encoder_2,
            "grad_norm_unet": self.grad_norm_unet,
            "grad_norm_vision_encoder": self.grad_norm_vision_encoder,
            "param_update_norm_unet": self.param_update_norm_unet,
            "param_update_norm_te1": self.param_update_norm_te1,
            "param_update_norm_te2": self.param_update_norm_te2,
            "param_update_norm_ve": self.param_update_norm_ve,
            "param_cumulative_drift_unet": self.param_cumulative_drift_unet,
            "param_cumulative_drift_te1": self.param_cumulative_drift_te1,
            "param_cumulative_drift_te2": self.param_cumulative_drift_te2,
            "param_cumulative_drift_ve": self.param_cumulative_drift_ve,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


# ============================================================
# Tagger Training Models (training.db)
# ============================================================

class TaggerTrainingRun(TrainingBase):
    """Training run for SigLIP2-based image tagger."""
    __tablename__ = "tagger_training_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, unique=True, index=True, nullable=False, default=lambda: str(uuid.uuid4()))
    run_name = Column(String, nullable=False, default="")

    # Status
    status = Column(String, default="pending", index=True)  # pending|running|paused|completed|failed|stopped
    progress = Column(Float, default=0.0)   # 0.0 - 1.0
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, default=0)
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)

    # Configuration
    training_method = Column(String, default="lora")        # "lora" | "full"
    vision_encoder_path = Column(String, nullable=False)    # path to .safetensors
    dataset_configs = Column(JSON, nullable=True)           # [{dataset_id: int, caption_types: [str]}]
    output_dir = Column(String, nullable=True)

    # Hyperparameters (stored as JSON for flexibility)
    config = Column(JSON, nullable=True)                    # full config dict

    # Vocabulary snapshot
    num_tags = Column(Integer, nullable=True)
    tag_vocabulary = Column(JSON, nullable=True)            # {tag_to_idx, idx_to_tag, tag_to_category, num_tags}

    # Best metrics
    best_f1 = Column(Float, nullable=True)
    best_threshold = Column(Float, nullable=True)
    threshold_f1_curve = Column(JSON, nullable=True)   # {"0.05": f1, ..., "0.95": f1}
    latest_loss = Column(Float, nullable=True)
    latest_lr = Column(Float, nullable=True)

    # Checkpoints
    best_checkpoint_path = Column(String, nullable=True)
    latest_checkpoint_path = Column(String, nullable=True)
    checkpoint_paths = Column(JSON, default=list)          # list of step checkpoint paths

    # Resume tracking
    resumed_from_step = Column(Integer, nullable=True)     # global_step at last resume
    last_resumed_at = Column(DateTime, nullable=True)

    # Error info
    error_message = Column(Text, nullable=True)
    status_message = Column(String, nullable=True)  # Human-readable phase message during preparation

    # Timestamps
    created_at = Column(DateTime, default=get_local_now)
    updated_at = Column(DateTime, default=get_local_now, onupdate=get_local_now)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    def to_list_dict(self):
        """Lightweight serialization for list views — excludes tag_vocabulary (~2MB per run)."""
        return {
            "id": self.id,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "status": self.status,
            "progress": self.progress,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "training_method": self.training_method,
            "vision_encoder_path": self.vision_encoder_path,
            "dataset_configs": self.dataset_configs,
            "output_dir": self.output_dir,
            "config": self.config,
            "num_tags": self.num_tags,
            "best_f1": self.best_f1,
            "best_threshold": self.best_threshold,
            "threshold_f1_curve": self.threshold_f1_curve,
            "latest_loss": self.latest_loss,
            "latest_lr": self.latest_lr,
            "best_checkpoint_path": self.best_checkpoint_path,
            "latest_checkpoint_path": self.latest_checkpoint_path,
            "checkpoint_paths": self.checkpoint_paths or [],
            "resumed_from_step": self.resumed_from_step,
            "last_resumed_at": self.last_resumed_at.isoformat() if self.last_resumed_at else None,
            "error_message": self.error_message,
            "status_message": self.status_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    def to_dict(self):
        return {
            **self.to_list_dict(),
            "tag_vocabulary": self.tag_vocabulary,
        }


class TaggerTrainingMetrics(TrainingBase):
    """Per-step metrics for tagger training runs.

    ``resume_seq`` distinguishes successive resumes of the same run: 0 for
    the initial run, 1 for the first resume, etc.  Together with ``step``
    it forms the uniqueness key, so overlapping step ranges across resumes
    are preserved as separate rows (and rendered as separate curves).
    """
    __tablename__ = "tagger_training_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False, index=True)
    resume_seq = Column(Integer, nullable=False, default=0, server_default="0")
    step = Column(Integer, nullable=False)
    epoch = Column(Integer, nullable=True)
    loss = Column(Float, nullable=True)
    f1 = Column(Float, nullable=True)
    train_f1 = Column(Float, nullable=True)
    threshold = Column(Float, nullable=True)
    learning_rate = Column(Float, nullable=True)
    # Macro precision/recall at the current threshold (nullable for backward compat)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=get_local_now)

    __table_args__ = (
        UniqueConstraint("run_id", "resume_seq", "step", name="uq_tagger_run_resume_step"),
        Index("idx_tagger_run_resume_step", "run_id", "resume_seq", "step"),
    )

    def to_dict(self):
        return {
            "step": self.step,
            "resume_seq": self.resume_seq,
            "epoch": self.epoch,
            "loss": self.loss,
            "f1": self.f1,
            "train_f1": self.train_f1,
            "threshold": self.threshold,
            "learning_rate": self.learning_rate,
            "precision": self.precision,
            "recall": self.recall,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
