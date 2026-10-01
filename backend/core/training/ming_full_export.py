"""CPU inference export from a complete native floating full checkpoint."""
import os
import shutil
import tempfile
from pathlib import Path


def validate_checkpoint(checkpoint, output_dir):
    from core.models.ming_image_01.artifact import inspect_artifact, _json, _inside
    checkpoint, root = Path(checkpoint).resolve(), Path(output_dir).resolve()
    if checkpoint.parent != root or not checkpoint.is_dir():
        raise ValueError("Ming export source must be a native checkpoint directory inside this run")
    artifact = inspect_artifact(checkpoint)
    metadata = _json(_inside(artifact.root, "training.json"))
    if (type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1
            or metadata.get("kind") != "full_parameter_weights" or metadata.get("profile") != artifact.profile):
        raise ValueError("Ming export requires a native full-parameter checkpoint")
    if any(header.dtype not in {"F16", "BF16", "F32"} or key.endswith((".weight_scale", ".comfy_quant"))
            for component,headers in artifact.tensor_headers.items() if component not in {"mllm","connector"}
            for key,header in headers.items()):
        raise ValueError("Ming training export requires a floating full checkpoint")
    from core.models.ming_image_01.conditioner_quantization import prepare_conditioner_components
    import torch
    prepare_conditioner_components(artifact,torch.float32,"eager")
    return artifact


def export_ming_full_checkpoint(checkpoint, output_dir, *, overwrite=False):
    from core.models.ming_image_01.conversion import convert_package
    artifact = validate_checkpoint(checkpoint, output_dir)
    root = Path(output_dir).resolve()
    parent = root / "exports"
    parent.mkdir(exist_ok=True)
    if parent.resolve().parent != root:
        raise ValueError("Ming training export directory must stay inside this run")
    published, backup = parent / "int8_convrot", parent / ".int8_convrot_previous"
    for target in (published, backup):
        if target.resolve() != parent.resolve() / target.name:
            raise ValueError("Ming training export refuses linked destinations")
    if os.path.lexists(backup):
        raise FileExistsError(f"Previous export replacement is incomplete: {backup}")
    if os.path.lexists(published) and (not overwrite or not published.is_dir()):
        raise FileExistsError("INT8 ConvRot export already exists; choose overwrite for a native package")
    with tempfile.TemporaryDirectory(prefix=".ming_int8_convrot_", dir=parent) as temporary:
        staging = Path(temporary) / "package"
        convert_package(artifact.root, staging, format_="int8_convrot", device="cpu")
        if published.exists():
            os.replace(published, backup)
        try:
            os.replace(staging, published)
        except BaseException:
            if backup.exists():
                os.replace(backup, published)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    return published
