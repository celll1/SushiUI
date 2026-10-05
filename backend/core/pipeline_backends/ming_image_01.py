"""Ming native lifecycle, task/profile routing and RGBA image operations."""

import math
import random
from copy import deepcopy
from contextlib import contextmanager

from PIL import Image, ImageChops, ImageFilter, ImageOps


class MingImage01Mixin:
    def _validate_ming_partition_arch(self,params):
        from core.models.ming_image_01.partition_generation import normalize_partition_config
        if normalize_partition_config(params.get("ming_image_01_partition"))["mode"] != "off" and not getattr(self,"is_ming_image_01_model",False):
            from api.error_handlers import ValidationError
            raise ValidationError("Partitioned generation requires Ming-Image 0.1")

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
        from core.models.ming_image_01.partition_generation import resolve_partition_plan,plan_metadata
        frames=layers+1 if profile=="layer_decompose" else 1
        try:
            plan=resolve_partition_plan(resolved.get("ming_image_01_partition"),height//16,width//16,frames)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        generator = MingGenerator(components)
        try:
            generator.sampler.profile.validate_task(task, has_reference_image=bool(images), num_layers=layers)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        def flagged(entry, name):
            return bool(entry.get(name)) if isinstance(entry, dict) else bool(getattr(entry,name,False))
        def is_guide(entry):
            return flagged(entry,"is_reference_guide")
        entries = resolved.get("controlnets") or []
        if any(is_guide(entry) == flagged(entry,"is_style_transfer") for entry in entries):
            raise ValidationError("Ming conditioning entries must select exactly one of Reference Guide or Style Transfer")
        styles = resolved.get("style_transfers") or ([resolved["style_transfer"]] if resolved.get("style_transfer") else [])
        requested_styles = [entry for entry in entries if flagged(entry,"is_style_transfer")]
        if requested_styles and len(requested_styles) != len(styles):
            raise ValidationError("Every Ming Style Transfer requires a valid decoded image")
        from core.models.ming_image_01.style_generation import normalize_styles
        try:
            normalized_styles = normalize_styles(styles)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        combine_mode = resolved.get("style_combine_mode")
        if styles:
            from api.param_defaults import STYLE_TRANSFER_UI_DEFAULTS
            if combine_mode is None: combine_mode = STYLE_TRANSFER_UI_DEFAULTS["style_combine_mode"]
            if combine_mode not in {"stack","common_concept"}:
                raise ValidationError("Ming style combine mode must be stack or common_concept")
            if resolved["attention_type"] != "normal":
                raise ValidationError("Ming Style Transfer currently requires normal native attention")
            if any((cfg.style_guidance_scale is not None and cfg.style_guidance_scale > 0)
                   or cfg.rope_offset or cfg.late_release for _,cfg in normalized_styles):
                raise ValidationError("Ming style does not implement independent Style Guidance, RoPE offset or late release")
        from core.models.ming_image_01.reference_guide import normalize_guides
        guides = resolved.get("controlnet_images") or []
        requested_guides = [entry for entry in resolved.get("controlnets") or [] if is_guide(entry)]
        if requested_guides and len(requested_guides) != len(guides):
            raise ValidationError("Every Ming Reference Guide requires a valid decoded image")
        if requested_guides:
            from api.param_defaults import REFERENCE_GUIDE_DEFAULTS
            guides = [{**guide, **{key:REFERENCE_GUIDE_DEFAULTS[key]
                for key in ("strength","start_step","end_step")
                if isinstance(requested,dict) and key not in requested}}
                for requested,guide in zip(requested_guides,guides)]
        try:
            normalize_guides(guides)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        from core.models.ming_image_01.attention import install_attention
        try:
            install_attention(components.transformer, resolved["attention_type"])
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        self.current_attention_type = resolved["attention_type"]
        if resolved["negative_prompt"]:
            raise ValidationError("Ming native CFG uses zero conditioning; negative prompts are unsupported")
        if plan is not None and (guides or styles):
            raise ValidationError("Ming partitioned generation currently excludes Reference Guide and Style Transfer")
        pending = [key for key in ("text_encoder_quantization", "enable_block_swap",
                                  "keep_models_hot", "loop_decode", "vae_tile_threshold", "vae_tile_mode",
                                  "vae_tile_global_norm")
                   if resolved[key] != defaults[key]]
        if pending:
            raise ValidationError("Ming options awaiting runtime wiring: " + ", ".join(pending))
        if type(resolved["color_flatten_strength"]) is not int or not 0 <= resolved["color_flatten_strength"] <= 100:
            raise ValidationError("Ming Color Flatten strength must be an integer in [0, 100]")
        if resolved["unet_quantization"] not in (defaults["unet_quantization"], "none", "int8"):
            raise ValidationError("Ming runtime DiT quantization currently supports INT8 only")
        from core.vram_optimization import apply_runtime_int8_quantization
        from core.models.ming_image_01.inspection import loaded_model_info
        _, converted = apply_runtime_int8_quantization(self, components.transformer, "ming_image_01",
            resolved["unet_quantization"], label="Ming DiT", progress_callback=progress_callback,
            compute_dtype=components.transformer.dtype)
        if converted:
            self.current_model_info.update(loaded_model_info(components))
            model_info.update(self.current_model_info)
        from core.models.ming_image_01.generation_adapters import generation_adapters
        from core.extensions.lora_manager import lora_manager
        from api.generation_status import add_warning
        with generation_adapters(components, resolved.get("loras"),
                resolve_path=lora_manager._resolve_lora_path,
                warn=lambda message, code: add_warning(message, code)) as adapters:
            topology = getattr(components.transformer, "_residual_unet_spec", None)
            if topology is not None:
                from core.rewiring.spec import digest, preview_plan
                try:
                    preview_plan(topology, height // 16, width // 16)
                    if plan is not None:
                        for box in plan.inputs:preview_plan(topology,box.height,box.width)
                except ValueError as exc:
                    raise ValidationError(str(exc)) from exc
                params["dit_rewiring_actual"] = {"spec": topology, "spec_sha256": digest(topology),
                                                "native_layout": "target_reference_padding_caption_v1", "kv_cache": False}
            else:
                params.pop("dit_rewiring_actual", None)
            has_global=bool(getattr(components.transformer,"_ming_partition_global_payload",None) or
                any(entry["strength"] for entry in getattr(components.transformer,"_ming_partition_global_generation",[])))
            if plan is not None:
                params["ming_image_01_partition_actual"]=plan_metadata(plan,frames,global_adapter=has_global)
                if not has_global:add_warning("Ming partitioned generation has no global adapter; cross-region target attention is absent.","ming_partition_no_global_adapter")
            else:
                params.pop("ming_image_01_partition_actual",None)
                if has_global:add_warning("Ming full-frame generation ignores partition-global summaries.","ming_partition_global_adapter_ignored")
            if resolved["vae_tiling"]:
                components.vae.enable_tiling()
            else:
                components.vae.disable_tiling()
            residency = MingPhaseResidency(components, device=self.device, cpu_text_encoding=resolved["cpu_text_encoding"])

            @contextmanager
            def phase(name):
                with generation_timer.phase("text_encode" if name == "conditioner" else "denoise"), residency(name):
                    yield

            adapters.set_step(0, steps)

            def callback(index, total, timestep, latents):
                adapters.set_step(index + 1, total)
                if step_callback is not None:
                    step_callback(index, timestep, latents)

            def preview(index,total,timestep,latents,prediction):
                if progress_callback is not None:
                    from core.models.ming_image_01.generation_preview import primary_latent
                    progress_callback(index,total,primary_latent(latents,profile),None,
                        None if prediction is None else primary_latent(prediction,profile))
            preview.wants_predicted_x0=getattr(progress_callback,"wants_predicted_x0",None)

            try:
                groups = generator.run(prompts=[resolved["prompt"]] * batch,
                    references=[list(images) for _ in range(batch)], task=task, width=width, height=height,
                    resolution=round(math.sqrt(width * height)), steps=steps, cfg=float(cfg), seed=seed,
                    requested_layers=layers, negative_prompt=resolved["negative_prompt"], output_type="pil",
                    cancelled=lambda: self.cancel_requested, step_callback=callback, phase_context=phase,
                    reference_guides=guides,partition_plan=plan,
                    style_transfers=styles,style_combine_mode=combine_mode,
                    preview_callback=preview if progress_callback is not None else None,
                    color_flatten_strength=resolved["color_flatten_strength"])
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
        for key in ("dit_rewiring_actual","ming_image_01_partition_actual"):
            if key in working:params[key]=working[key]
            else:params.pop(key,None)
        return output.validate(), seed, ancestral
