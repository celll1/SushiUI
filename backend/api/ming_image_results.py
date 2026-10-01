"""Persist a request-owned Ming RGBA group without dropping native layers."""

import os
from pathlib import Path
from uuid import uuid4

from config.settings import settings
from core.models.ming_image_01.results import MingGenerationOutput
from database.models import GeneratedImage
from utils.image_utils import save_image_with_metadata, create_thumbnail, calculate_image_hash, extract_lora_names, encode_mask_to_base64
from .generation_utils import create_db_image_record, prepare_params_for_db
from .generation_status import get_warnings, complete_generation


class MingResultSaveCancelled(RuntimeError):
    pass


def persist_ming_results(output: MingGenerationOutput, *, db, params, generation_type,
                         model_info, model_name, model_hash, generation_id=None,
                         lora_names=None, source_image_hash=None, mask_data_base64=None,
                         skip_gallery=False, cancelled=None):
    """Commit all rows together; remove this request's files on any pre-commit failure."""
    output.validate()
    if model_info.get("type") != "ming_image_01":
        raise ValueError("Ming output requires Ming model metadata")
    if any(group.profile != model_info.get("profile") for group in output.groups):
        raise ValueError("Ming output profile disagrees with the loaded checkpoint")
    warnings = get_warnings(generation_id) if generation_id is not None else []
    files = []
    records = []
    groups = []

    def check_cancel():
        if cancelled is not None and cancelled():
            raise MingResultSaveCancelled("Ming output save cancelled")

    check_cancel()
    try:
        for batch_index, group in enumerate(output.groups):
            group_id = uuid4().hex
            filenames = []
            rows = []
            for frame_index, (image, role) in enumerate(zip(group.values, group.roles)):
                check_cancel()
                filename = f"ming_{group_id}_{frame_index}.png"
                item_params = {**params, "seed": group.seed, "width": group.width, "height": group.height,
                    "ming_image_01_actual": {"group_id": group_id, "batch_index": batch_index,
                        "profile": group.profile, "requested_layers": group.requested_layers,
                        "target_frames": group.target_frames, "frame_index": frame_index, "role": role}}
                if warnings:
                    item_params["effective_warnings"] = warnings
                saved = save_image_with_metadata(image, item_params, generation_type,
                    model_info=model_info, generation_id=generation_id, output_filename=filename)
                path = Path(settings.outputs_dir) / saved
                files.append(path)
                filenames.append(saved)
                if not skip_gallery:
                    thumb_root = Path(settings.thumbnails_dir)
                    thumbnail_paths = [thumb_root / saved, thumb_root / (Path(saved).stem + ".webp")]
                    if any(path.exists() for path in thumbnail_paths):
                        raise FileExistsError("Ming thumbnail filename collision")
                    files.extend(thumbnail_paths)
                    create_thumbnail(str(path))
                    record = create_db_image_record(GeneratedImage, filename=saved,
                        params=prepare_params_for_db(item_params, calculate_image_hash),
                        actual_seed=group.seed, generation_type=generation_type,
                        image_hash=calculate_image_hash(image), lora_names=lora_names,
                        model_name=model_name, model_hash=model_hash, result_image=image,
                        source_image_hash=source_image_hash, mask_data_base64=mask_data_base64)
                    db.add(record)
                    records.append(record)
                    rows.append(record)
            groups.append({"group_id": group_id, "batch_index": batch_index, "profile": group.profile,
                "requested_layers": group.requested_layers, "roles": list(group.roles),
                "filenames": filenames, "rows": rows, "seed": group.seed})
        check_cancel()
        if not skip_gallery:
            db.flush()
        response_images = [record.to_dict() for record in records]
        response_groups = [{**{key: value for key, value in group.items() if key != "rows"},
                            "image_ids": [row.id for row in group["rows"]]} for group in groups]
        response = {"success": True, "actual_seed": output.groups[0].seed, "groups": response_groups, "warnings": warnings}
        if skip_gallery:
            response.update(filename=groups[0]["filenames"][0],
                            image_path=f"/outputs/{groups[0]['filenames'][0]}")
        else:
            response.update(image=response_images[0], images=response_images)
            check_cancel()
            db.commit()
        return response
    except BaseException:
        try:
            if not skip_gallery:
                db.rollback()
        finally:
            for path in reversed(files):
                path.unlink(missing_ok=True)
        raise


def finish_ming_generation(output, *, db, params, generation_type, generation_id,
                           lora_configs, skip_gallery, source_image=None, mask_image=None):
    """Finish a rich output before the generic PIL/latent route can discard frames."""
    output.validate()
    info = output.model_info
    if not isinstance(info, dict) or info.get("type") != "ming_image_01":
        raise ValueError("Ming API output requires a request-owned model snapshot")
    if params["seed"] != output.groups[0].seed:
        raise ValueError("Ming route seed disagrees with its native output")
    from .quantized_gemm import report_quantized_gemm_outcome
    report_quantized_gemm_outcome(params.get("quantized_gemm_mode"), params.get("fp8_gemm"), "ming_image_01")
    response = persist_ming_results(output, db=db, params=params, generation_type=generation_type,
        model_info=info, model_name=os.path.basename(info.get("source") or ""),
        model_hash=info.get("model_hash") or "", generation_id=generation_id,
        lora_names=extract_lora_names(lora_configs), skip_gallery=skip_gallery,
        source_image_hash=calculate_image_hash(source_image) if source_image is not None else None,
        mask_data_base64=encode_mask_to_base64(mask_image) if mask_image is not None else None)
    first = response.get("image") or {}
    last_result = {"filename": first.get("filename") or response.get("filename"),
                   "seed": response["actual_seed"], "groups": response["groups"]}
    if first.get("id") is not None:
        last_result["image_id"] = first["id"]
    complete_generation(last_result, generation_id=generation_id)
    return response
