"""Native trainer previews with exclusive encoder ownership and state restoration."""
from contextlib import nullcontext
from .block_offload import execution_device, move_component
from dataclasses import replace
from PIL import Image
import torch

from .generation import MingGenerator
from .residency import MingPhaseResidency


def aligned_sample_dimensions(model, vae_scale, width, height):
    from .residual_unet import spatial_alignment
    ah, aw = spatial_alignment(model)
    mh, mw = vae_scale * 2 * ah, vae_scale * 2 * aw
    return ((width + mw - 1) // mw * mw,
            (height + mh - 1) // mh * mh)


@torch.no_grad()
def generate_sample(trainer, ctx, *, step_progress_callback=None):
    from api.param_defaults import TRAINING_DEFAULTS
    components = trainer.ming_image_01_components
    if ctx.negative_prompt:
        raise ValueError("Ming native guidance uses zero conditioning; negative prompts are unsupported")
    paths = [p for p in (ctx.reference_image_path, ctx.condition_image_path) if p]
    if len(set(paths)) > 1:
        raise ValueError("Ming trainer preview accepts one reference image path")
    images = []
    if paths:
        with Image.open(paths[0]) as image:
            images.append(image.convert("RGBA"))
    layers = (trainer.config.get("ming_image_01_sample_layers", TRAINING_DEFAULTS["ming_image_01_sample_layers"])
              if components.artifact.profile == "layer_decompose" else 1)
    task = ("layer-decompose" if components.artifact.profile == "layer_decompose"
            else "image-edit" if images else "text-to-image")
    modules = (components.conditioner, components.transformer, components.vae)
    devices, modes = [], []
    for module in modules:
        devices.append(execution_device(module))
        modes.extend((child, child.training) for child in module.modules())
    buffer = getattr(trainer, "ming_batch_buffer", None)
    exclusive = buffer.suspend_prefetch() if buffer is not None else nullcontext()
    with exclusive:
        try:
            for module in modules:
                module.eval()
            preview = replace(components, scheduler=type(components.scheduler).from_config(components.scheduler.config))
            generator = MingGenerator(preview)
            # FP32 adapter masters need the training autocast policy only in DiT prediction.
            generator.sampler.pipe.transformer.forward_context = lambda: torch.autocast(
                device_type=torch.device(trainer.device).type, dtype=trainer.training_dtype,
                enabled=trainer.mixed_precision, cache_enabled=False)
            width, height = aligned_sample_dimensions(components.transformer,
                generator.sampler.pipe.vae_scale_factor, ctx.width, ctx.height)
            if (width, height) != (ctx.width, ctx.height):
                print(f"[Ming sample] Aligning canvas {ctx.width}x{ctx.height} -> {width}x{height} "
                      "for VAE and U-Net topology", flush=True)
            residency = MingPhaseResidency(preview, device=trainer.device,
                cpu_text_encoding=getattr(trainer, "_text_encoding_mode", "") == "cpu_prefetch")
            def progress(index, total, timestep, latents):
                trainer._check_stop_requested()
                if step_progress_callback is not None:
                    step_progress_callback(index + 1, total)
            def cancelled():
                trainer._check_stop_requested()
                return False
            groups = generator.run(prompts=[ctx.prompt], references=[images], task=task,
                width=width, height=height, resolution=max(width, height),
                steps=ctx.num_inference_steps, cfg=ctx.guidance_scale, seed=ctx.seed,
                requested_layers=layers, cancelled=cancelled, step_callback=progress,
                phase_context=residency)
            primary = groups[0].values[0]
            primary._ming_result_group = groups[0]
            return primary
        finally:
            failures = []
            for module, device in zip(modules, devices):
                try:
                    move_component(module, device)
                except Exception as exc:
                    failures.append(exc)
            for module, mode in modes:
                module.training = mode
            if failures:
                raise RuntimeError("Ming preview residency restoration failed") from failures[0]


def publish_training_sample(sample, path, metadata):
    """Publish every native frame; rollback preserves an earlier sample set."""
    from copy import deepcopy
    from pathlib import Path
    import os
    from tempfile import TemporaryDirectory
    from .results import MingGenerationOutput
    group = getattr(sample, "_ming_result_group", None)
    if group is None:
        from api.training_media import publish_sample_png
        publish_sample_png(sample, path, metadata)
        return [path]
    MingGenerationOutput((group,)).validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    paths = [path] + [path.with_name(f"{path.stem}_{role}{path.suffix}") for role in group.roles[1:]]
    backups, published = {}, []
    with TemporaryDirectory(prefix=".ming-sample-", dir=path.parent) as temporary:
        staging = Path(temporary)
        for index, (image, role) in enumerate(zip(group.values, group.roles)):
            info = deepcopy(metadata)
            for key, value in {"ming_profile": group.profile, "ming_frame_role": role,
                               "width": group.width, "height": group.height,
                               "ming_frame_index": index, "ming_target_frames": group.target_frames,
                               "ming_requested_layers": group.requested_layers}.items():
                info.add_text(key, str(value))
            image.save(staging / str(index), format="PNG", pnginfo=info)
        try:
            for index, destination in enumerate(paths):
                if destination.exists():
                    backup = staging / f"backup-{index}"
                    os.replace(destination, backup)
                    backups[destination] = backup
                os.replace(staging / str(index), destination)
                published.append(destination)
        except BaseException:
            for destination in reversed(published):
                destination.unlink(missing_ok=True)
            for destination, backup in backups.items():
                os.replace(backup, destination)
            raise
    sample._ming_saved_paths = paths
    return paths
