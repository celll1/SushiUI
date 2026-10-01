"""Ming native lifecycle, task/profile routing and RGBA image operations."""

import math
import random
from copy import deepcopy
from contextlib import contextmanager

from PIL import Image, ImageChops, ImageFilter, ImageOps


class MingImage01Mixin:
    def _install_ming_image_01(self, result, *, model_id, source_type, source):
        from core.models.ming_image_01.inspection import loaded_model_info
        from core.models.components.wiring import MING_IMAGE_01_WIRING
        info = loaded_model_info(result["ming_native"])
        self.ming_image_01_components = result
        self.is_ming_image_01_model = True
        self.current_model = model_id
        self.current_attention_type = "normal"
        self._sushi_wiring = MING_IMAGE_01_WIRING.replace(te_out_dim=result["transformer"].config.cap_feat_dim)
        self.current_model_info = {**info, "source_type": source_type, "source": source,
                                   "is_video": False, "is_audio": False,
                                   "latent_channels": 16, "generation_available": True}

    def _ming_image_run(self, params, *, images, progress_callback=None, step_callback=None):
        from api.param_defaults import image_defaults_for_arch
        from api.error_handlers import ValidationError
        from core.models.ming_image_01.generation import MingGenerator
        from core.models.ming_image_01.residency import MingPhaseResidency
        from core.models.ming_image_01.results import MingGenerationOutput
        from core.models.ming_image_01.sampling import MingGenerationCancelled
        from core.inference.generation_timing import generation_timer

        components = self.ming_image_01_components["ming_native"]
        model_info = deepcopy(self.current_model_info)
        profile = components.artifact.profile
        defaults = image_defaults_for_arch("ming_image_01", profile=profile)
        resolved = {**defaults, **params}
        for key in ("steps", "cfg_scale"):
            if resolved[key] is None:
                resolved[key] = defaults[key]
        width, height, steps, cfg = (resolved[key] for key in ("width", "height", "steps", "cfg_scale"))
        if any(type(value) is not int or value <= 0 for value in (width, height, steps)):
            raise ValidationError("Ming dimensions/steps must be positive integers")
        if width % 16 or height % 16:
            raise ValidationError("Ming output dimensions require the native 16-pixel token grid")
        if not isinstance(cfg, (float, int)) or isinstance(cfg, bool) or not math.isfinite(cfg) or cfg < 0:
            raise ValidationError("Ming guidance must be finite and nonnegative")
        batch = resolved["batch_size"]
        if type(batch) is not int or batch < 1:
            raise ValidationError("Ming batch size must be a positive integer")
        seed = resolved["seed"]
        if type(seed) is not int or seed < -1 or seed >= 2**63:
            raise ValidationError("Ming seed must be -1 or an integer in [0, 2**63)")
        if seed == -1:
            seed = random.SystemRandom().randrange(2**31)
        if any(not isinstance(image, Image.Image) for image in images):
            raise ValidationError("Ming references must be decoded images")
        task = "layer-decompose" if profile == "layer_decompose" else "image-edit" if images else "text-to-image"
        layers = resolved["ming_image_01_layers"] if profile == "layer_decompose" else 1
        if type(layers) is not int or layers < 1:
            raise ValidationError("Ming layer count must be a positive integer")
        generator = MingGenerator(components)
        try:
            generator.sampler.profile.validate_task(task, has_reference_image=bool(images), num_layers=layers)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        if resolved["negative_prompt"]:
            raise ValidationError("Ming native CFG uses zero conditioning; negative prompts are unsupported")
        if resolved.get("loras"):
            raise ValidationError("Ming adapter generation wiring is not yet enabled")
        if resolved.get("controlnets") or resolved.get("controlnet_images") or resolved.get("style_transfers"):
            raise ValidationError("Ming reference-guide/style/ControlNet extensions are not yet enabled")
        pending = [key for key in ("unet_quantization", "text_encoder_quantization", "enable_block_swap",
                                  "keep_models_hot", "loop_decode", "vae_tile_threshold", "vae_tile_mode",
                                  "vae_tile_global_norm", "color_flatten_strength")
                   if resolved[key] != defaults[key]]
        if pending:
            raise ValidationError("Ming options awaiting runtime wiring: " + ", ".join(pending))
        topology = getattr(components.transformer, "_residual_unet_spec", None)
        if topology is not None:
            from core.rewiring.spec import digest, preview_plan
            try:
                preview_plan(topology, height // 16, width // 16)
            except ValueError as exc:
                raise ValidationError(str(exc)) from exc
            params["dit_rewiring_actual"] = {"spec": topology, "spec_sha256": digest(topology),
                                            "native_layout": "target_reference_padding_caption_v1", "kv_cache": False}
        else:
            params.pop("dit_rewiring_actual", None)
        if resolved["vae_tiling"]:
            components.vae.enable_tiling()
        else:
            components.vae.disable_tiling()
        residency = MingPhaseResidency(components, device=self.device, cpu_text_encoding=resolved["cpu_text_encoding"])

        @contextmanager
        def phase(name):
            with generation_timer.phase("text_encode" if name == "conditioner" else "denoise"), residency(name):
                yield

        def callback(index, total, timestep, latents):
            if progress_callback is not None:
                progress_callback(index, total, None, phase_label="Ming denoising")
            if step_callback is not None:
                step_callback(index, timestep, latents)

        try:
            groups = generator.run(prompts=[resolved["prompt"]] * batch,
                references=[list(images) for _ in range(batch)], task=task, width=width, height=height,
                resolution=round(math.sqrt(width * height)), steps=steps, cfg=float(cfg), seed=seed,
                requested_layers=layers, negative_prompt=resolved["negative_prompt"], output_type="pil",
                cancelled=lambda: self.cancel_requested, step_callback=callback, phase_context=phase)
        except MingGenerationCancelled as exc:
            raise RuntimeError("Generation cancelled by user") from exc
        model_info["dit_rewiring_spec"] = topology
        result = MingGenerationOutput(tuple(groups), model_info).validate()
        return result, seed, seed

    def _generate_txt2img_ming_image_01(self, params, progress_callback=None, step_callback=None):
        return self._ming_image_run(params, images=params.get("ref_images") or [],
            progress_callback=progress_callback, step_callback=step_callback)

    def _generate_img2img_ming_image_01(self, params, init_image, progress_callback=None, step_callback=None):
        from api.error_handlers import ValidationError
        if not isinstance(init_image, Image.Image):
            raise ValidationError("Ming image editing requires a decoded input image")
        return self._ming_image_run(params, images=[init_image, *(params.get("ref_images") or [])],
            progress_callback=progress_callback, step_callback=step_callback)

    def _generate_inpaint_ming_image_01(self, params, init_image, mask_image, progress_callback=None, step_callback=None):
        from api.error_handlers import ValidationError
        from api.param_defaults import image_defaults_for_arch
        if self.ming_image_01_components["ming_native"].artifact.profile != "generation_edit":
            raise ValidationError("Ming Layer decomposition does not implement inpaint/outpaint")
        if not isinstance(init_image, Image.Image) or not isinstance(mask_image, Image.Image):
            raise ValidationError("Ming inpaint requires decoded image and mask")
        rgba = init_image.convert("RGBA")
        mask = mask_image.convert("L").resize(rgba.size, Image.Resampling.NEAREST)
        reference = rgba.copy()
        reference.putalpha(ImageChops.multiply(rgba.getchannel("A"), ImageOps.invert(mask)))
        output, seed, ancestral = self._ming_image_run(params,
            images=[reference, *(params.get("ref_images") or [])],
            progress_callback=progress_callback, step_callback=step_callback)
        defaults = image_defaults_for_arch("ming_image_01", profile="generation_edit")
        blur = params.get("mask_blur", defaults["mask_blur"])
        for group in output.groups:
            original = rgba.resize((group.width, group.height), Image.Resampling.BILINEAR)
            blend = mask.resize(original.size, Image.Resampling.NEAREST)
            if blur > 0:
                blend = blend.filter(ImageFilter.GaussianBlur(blur))
            group.values[0] = Image.composite(group.values[0], original, blend)
        return output.validate(), seed, ancestral

    def _generate_outpaint_ming_image_01(self, params, init_image, progress_callback=None, step_callback=None):
        from api.error_handlers import ValidationError
        from api.param_defaults import OUTPAINT_DEFAULTS
        from core.inference.outpaint_utils import build_outpaint_canvas, build_outpaint_mask, validate_and_snap_placement
        if self.ming_image_01_components["ming_native"].artifact.profile != "generation_edit":
            raise ValidationError("Ming Layer decomposition does not implement inpaint/outpaint")
        placement = validate_and_snap_placement(params, init_image.size, align=16)
        canvas, _placed_rgb, rect = build_outpaint_canvas(init_image, params, align=16)
        crop = (placement["input_crop_x"], placement["input_crop_y"],
                placement["input_crop_x"] + placement["input_crop_w"],
                placement["input_crop_y"] + placement["input_crop_h"])
        placed = init_image.convert("RGBA").crop(crop).resize(
            (placement["place_width"], placement["place_height"]), Image.Resampling.LANCZOS)
        canvas = canvas.convert("RGBA")
        canvas.paste(placed, rect[:2])
        mask = build_outpaint_mask(canvas.size, rect, params.get("mask_blur", OUTPAINT_DEFAULTS["mask_blur"]))
        working = {**params, "width": canvas.width, "height": canvas.height, "mask_blur": 0}
        output, seed, ancestral = self._generate_inpaint_ming_image_01(working, canvas, mask,
            progress_callback=progress_callback, step_callback=step_callback)
        for group in output.groups:
            group.values[0].paste(placed, rect[:2])
        params.update({key: value for key, value in working.items() if key == "dit_rewiring_actual"})
        return output.validate(), seed, ancestral
