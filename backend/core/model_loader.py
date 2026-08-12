from typing import Dict, Any, Optional, Literal, Union
import os
import sys
import json
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoencoderKL
from safetensors.torch import load_file
from pathlib import Path

ModelSource = Literal["safetensors", "diffusers", "huggingface"]
# DEUS support removed - architecture no longer maintained
# ModelType = Literal["sd15", "sdxl", "zimage", "deus", "flux2"]
ModelType = Literal["sd15", "sdxl", "zimage", "flux2", "anima", "lens", "ideogram4", "minit2i", "krea2", "ltx2", "acestep", "minimax_h3"]

class ModelLoader:
    """Handles loading models from various sources"""

    @staticmethod
    def _configure_v_prediction_scheduler(pipeline):
        """Configure scheduler for v-prediction models

        V-prediction models require:
        1. prediction_type = "v_prediction"
        2. timestep_spacing = "trailing" (recommended for v-prediction)

        Note: rescale_betas_zero_snr is intentionally set to False by default.
        While some v-prediction models were trained with zero terminal SNR,
        many SDXL v-prediction models (especially newer ones) work better
        WITHOUT rescale_betas_zero_snr=True, as it can cause extreme sigma
        values (e.g., 4096) leading to black or blurry outputs.

        References:
        - https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/16567
        - https://huggingface.co/docs/diffusers/using-diffusers/scheduler
        - https://github.com/comfyanonymous/ComfyUI/discussions/2794
        """
        try:
            # Register to scheduler config (this modifies the scheduler's configuration)
            # Note: rescale_betas_zero_snr is omitted (defaults to False in most schedulers)
            pipeline.scheduler.register_to_config(
                prediction_type="v_prediction",
                timestep_spacing="trailing"
            )

            print(f"[ModelLoader] V-prediction scheduler configured:")
            print(f"  - prediction_type: v_prediction")
            print(f"  - rescale_betas_zero_snr: False (default, avoids extreme sigma values)")
            print(f"  - timestep_spacing: trailing")

        except Exception as e:
            print(f"[ModelLoader] Warning: Could not configure v-prediction scheduler: {e}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def _dir_contains_minit2i(model_path: str, max_depth: int = 2) -> bool:
        """True if a MiniT2I variant dir exists at model_path or within max_depth.

        A variant dir has transformer/config.json with the MiniT2I marker. This lets
        detection accept a repo root (.../MiniT2I) or container (.../minit2i) that holds
        minit2i-b-16 / minit2i-l-16 so the loader can resolve it. JSON-only (no torch).
        """
        def _is_variant(d: str) -> bool:
            cfg = os.path.join(d, "transformer", "config.json")
            if not os.path.isfile(cfg):
                return False
            try:
                with open(cfg, "r", encoding="utf-8") as f:
                    tcfg = json.load(f)
            except Exception:
                return False
            return (tcfg.get("_class_name") == "MiniT2IMMJiTModel"
                    or ("depth_double" in tcfg and "pca_channels" in tcfg))

        def _walk(d: str, depth: int) -> bool:
            if _is_variant(d):
                return True
            if depth >= max_depth:
                return False
            try:
                for name in os.listdir(d):
                    sub = os.path.join(d, name)
                    if os.path.isdir(sub) and _walk(sub, depth + 1):
                        return True
            except OSError:
                return False
            return False

        return _walk(model_path, 0)

    @staticmethod
    def detect_prediction_config(model_path: str, model_type: str) -> Dict[str, str]:
        """Detect prediction configuration from model metadata and state dict

        Supports unified training framework with noise_process and prediction_target.

        Args:
            model_path: Path to model file or directory
            model_type: Model type ("sd15", "sdxl", "zimage")

        Returns:
            Dict with keys:
                - noise_process: "ddpm" or "flow"
                - prediction_target: "epsilon", "velocity", or "sample"
                - source: "modelspec", "state_dict", "legacy", "inferred"
        """
        try:
            metadata = {}
            state_dict_keys = []

            # Load metadata and state dict keys
            if model_path.endswith('.safetensors.index.json'):
                # Sharded sushiUI save: metadata and key names live in the index.
                with open(model_path, 'r', encoding='utf-8') as f:
                    index = json.load(f)
                metadata = {k: v for k, v in (index.get("metadata") or {}).items()
                            if isinstance(v, str)}
                state_dict_keys = list((index.get("weight_map") or {}).keys())

            elif model_path.endswith('.safetensors'):
                from safetensors import safe_open
                with safe_open(model_path, framework="pt", device="cpu") as f:
                    metadata = f.metadata() or {}
                    state_dict_keys = list(f.keys())

            elif os.path.isdir(model_path):
                # Check diffusers format config
                config_path = os.path.join(model_path, "scheduler", "scheduler_config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        # Map scheduler config to metadata format
                        if config.get('prediction_type'):
                            metadata['prediction_type'] = config['prediction_type']

            # Priority 1: ModelSpec metadata (modelspec.noise_process + modelspec.prediction_type)
            if "modelspec.noise_process" in metadata:
                print(f"[ModelLoader] Detected prediction config from ModelSpec metadata")
                return {
                    "noise_process": metadata["modelspec.noise_process"],
                    "prediction_target": metadata.get("modelspec.prediction_type", "epsilon"),
                    "source": "modelspec"
                }

            # Priority 1b: ModelSpec prediction_type WITHOUT an explicit noise_process
            # (a SushiUI SD1.5/SDXL save omits noise_process when it resolved to
            # "auto"). Honor the prediction_type so a v-pred roundtrip still closes;
            # default noise_process by architecture family (ddpm for SD/SDXL).
            if "modelspec.prediction_type" in metadata:
                pred_target = str(metadata["modelspec.prediction_type"]).strip().lower()
                default_np = "flow" if model_type in ("zimage", "flux2", "minit2i", "krea2", "anima", "lens", "ltx2", "minimax_h3") else "ddpm"
                print(f"[ModelLoader] Detected prediction_type from ModelSpec metadata: {pred_target}")
                return {
                    "noise_process": metadata.get("modelspec.noise_process", default_np),
                    "prediction_target": pred_target,
                    "source": "modelspec"
                }

            # Priority 2: State dict marker (v_pred tensor)
            if "v_pred" in state_dict_keys:
                print(f"[ModelLoader] Detected v-prediction from state_dict marker")
                return {
                    "noise_process": "ddpm",
                    "prediction_target": "velocity",
                    "source": "state_dict"
                }

            # Priority 3: Legacy metadata (prediction_type or v_pred in metadata)
            if metadata.get("v_pred") or metadata.get("prediction_type") == "v_prediction":
                print(f"[ModelLoader] Detected v-prediction from legacy metadata")
                return {
                    "noise_process": "ddpm",
                    "prediction_target": "velocity",
                    "source": "legacy"
                }

            if metadata.get("prediction_type") in ["epsilon", "sample"]:
                print(f"[ModelLoader] Detected prediction_type from legacy metadata: {metadata['prediction_type']}")
                return {
                    "noise_process": "ddpm",
                    "prediction_target": metadata["prediction_type"],
                    "source": "legacy"
                }

            # Priority 4: Infer from model architecture
            if model_type == "zimage":
                print(f"[ModelLoader] Inferred prediction config from Z-Image architecture")
                return {
                    "noise_process": "flow",
                    "prediction_target": "velocity",
                    "source": "inferred"
                }
            elif model_type == "flux2":
                # FLUX.2 uses Flow Matching with velocity prediction
                print(f"[ModelLoader] Inferred prediction config from FLUX.2 architecture")
                return {
                    "noise_process": "flow",
                    "prediction_target": "velocity",
                    "source": "inferred"
                }
            elif model_type == "minit2i":
                # MiniT2I uses flow matching with x0 (sample) prediction.
                print(f"[ModelLoader] Inferred prediction config from MiniT2I architecture")
                return {
                    "noise_process": "flow",
                    "prediction_target": "sample",
                    "source": "inferred"
                }
            elif model_type == "krea2":
                # Krea 2 uses flow matching (rectified flow) with velocity prediction.
                print(f"[ModelLoader] Inferred prediction config from Krea 2 architecture")
                return {
                    "noise_process": "flow",
                    "prediction_target": "velocity",
                    "source": "inferred"
                }
            elif model_type == "ltx2":
                # LTX-2.3 video model: flow matching (FlowMatchEuler) with velocity prediction.
                print(f"[ModelLoader] Inferred prediction config from LTX-2.3 architecture")
                return {
                    "noise_process": "flow",
                    "prediction_target": "velocity",
                    "source": "inferred"
                }
            elif model_type == "minimax_h3":
                # MiniMax-H3 joint video+audio model: flow matching with velocity
                # prediction. The sampler recovers x0 as ``x_t + sigma * v``
                # (READ from ComfyUI's ODE form and verified in K0.4), i.e. the
                # OPPOSITE sign convention to diffusers' own flow schedulers --
                # that belongs to the Phase-2 loop, not to this label.
                print(f"[ModelLoader] Inferred prediction config from MiniMax-H3 architecture")
                return {
                    "noise_process": "flow",
                    "prediction_target": "velocity",
                    "source": "inferred"
                }
            # DEUS support removed - architecture no longer maintained
            # elif model_type == "deus":
            #     # DEUS uses DDPM with epsilon prediction (same as SDXL base)
            #     print(f"[ModelLoader] Inferred prediction config from DEUS architecture")
            #     return {
            #         "noise_process": "ddpm",
            #         "prediction_target": "epsilon",
            #         "source": "inferred"
            #     }
            else:  # sd15, sdxl
                print(f"[ModelLoader] Inferred prediction config from {model_type.upper()} architecture")
                return {
                    "noise_process": "ddpm",
                    "prediction_target": "epsilon",
                    "source": "inferred"
                }

        except Exception as e:
            print(f"[ModelLoader] Error detecting prediction config: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to safe defaults
            return {
                "noise_process": "ddpm" if model_type != "zimage" else "flow",
                "prediction_target": "epsilon" if model_type != "zimage" else "velocity",
                "source": "error_fallback"
            }

    @staticmethod
    def detect_v_prediction(model_path: str) -> bool:
        """Legacy method for backward compatibility

        Deprecated: Use detect_prediction_config() instead

        Returns:
            True if v-prediction model, False otherwise
        """
        # Detect model type for prediction config
        model_type = ModelLoader.detect_model_type(model_path)
        config = ModelLoader.detect_prediction_config(model_path, model_type)
        return config["prediction_target"] == "velocity"

    @staticmethod
    def has_embedded_vae(model_path: str) -> bool:
        """Detect if a safetensors model has an embedded VAE

        Returns:
            True if VAE is embedded in the model, False otherwise
        """
        try:
            if model_path.endswith('.safetensors'):
                from safetensors import safe_open
                with safe_open(model_path, framework="pt", device="cpu") as f:
                    keys = list(f.keys())

                    # Check for VAE decoder keys (common patterns)
                    vae_patterns = [
                        'first_stage_model.decoder',  # Standard SD/SDXL
                        'vae.decoder',                # Alternative format
                        'first_stage_model.encoder',  # VAE encoder
                        'vae.encoder',
                    ]

                    vae_keys = [k for k in keys if any(pattern in k for pattern in vae_patterns)]
                    has_vae = len(vae_keys) > 0

                    if has_vae:
                        print(f"[ModelLoader] Detected embedded VAE in model (found {len(vae_keys)} VAE keys)")
                        # Show first few keys for debugging
                        sample_keys = vae_keys[:3]
                        print(f"[ModelLoader] Sample VAE keys: {sample_keys}")
                    else:
                        print(f"[ModelLoader] No embedded VAE detected in model")
                        # Show sample of all keys for debugging
                        print(f"[ModelLoader] Total keys in model: {len(keys)}")
                        print(f"[ModelLoader] Sample keys: {keys[:5] if len(keys) > 0 else 'none'}")

                    return has_vae

            return True  # Assume diffusers format has VAE

        except Exception as e:
            print(f"[ModelLoader] ERROR: Could not detect VAE status: {e}")
            import traceback
            traceback.print_exc()
            # Return False to trigger external VAE loading as a safety measure
            # Better to load external VAE unnecessarily than to have None VAE
            return False

    @staticmethod
    def is_valid_diffusers_directory(path: str) -> bool:
        """Check if a directory is a valid diffusers-format model directory.

        A valid diffusers directory must have either:
        - model_index.json (standard diffusers pipeline), OR
        - transformer/config.json with axes_dims + rope_theta (Z-Image format)

        Non-model directories (tensorboard logs, training output, etc.) are excluded.
        """
        if not os.path.isdir(path):
            return False
        # Z-Image: transformer/config.json with Z-Image-specific keys
        transformer_config = os.path.join(path, "transformer", "config.json")
        if os.path.exists(transformer_config):
            try:
                with open(transformer_config, 'r') as f:
                    config = json.load(f)
                if "axes_dims" in config and "rope_theta" in config:
                    return True
            except Exception:
                pass
        # Standard diffusers pipeline requires model_index.json
        return os.path.exists(os.path.join(path, "model_index.json"))

    @staticmethod
    def _reattach_embedded_weights(module, state_dict, label: str):
        """Load trained (embedded) weights into a freshly built base component.

        Raises when NOTHING matched — silently keeping the untrained base
        weights would reintroduce a lossy roundtrip.
        """
        print(f"[ModelLoader] Reattaching embedded {label} weights "
              f"({len(state_dict)} tensors) from checkpoint")
        info = module.load_state_dict(state_dict, strict=False)
        missing = list(getattr(info, "missing_keys", []) or [])
        unexpected = list(getattr(info, "unexpected_keys", []) or [])
        matched = len(state_dict) - len(unexpected)
        if missing:
            print(f"[ModelLoader]   embedded {label} missing: {len(missing)}")
        if unexpected:
            print(f"[ModelLoader]   embedded {label} unexpected: {len(unexpected)}")
        if matched <= 0:
            raise RuntimeError(
                f"Embedded {label} weights in the checkpoint did not match the "
                f"base {label} at all ({len(state_dict)} tensors, 0 matched). "
                f"The checkpoint's {label} section uses an incompatible key "
                f"layout; refusing to silently fall back to the untrained base "
                f"{label}."
            )

    @staticmethod
    def _keys_look_krea2(keys, metadata) -> bool:
        """Krea 2 single-file signature check.

        Matches: metadata flag, sushiUI combined save (transformer.text_fusion.*),
        diffusers keys (text_fusion.* + time_mod_proj.*), official raw keys
        (txtfusion.* + tmlp.* + first.*), or comfy-prefixed raw keys
        (model.diffusion_model.<raw keys>).
        """
        metadata = metadata or {}

        def _has(*prefixes):
            return all(
                any(k.startswith(p) or (".diffusion_model." + p) in k for k in keys)
                for p in prefixes
            )

        return (
            str(metadata.get("model_type", "")).lower() == "krea2"
            or any(k.startswith("transformer.text_fusion.") for k in keys)
            or _has("text_fusion.", "time_mod_proj.")
            or _has("txtfusion.", "tmlp.", "first.")
        )

    @staticmethod
    def _keys_look_lens(keys) -> bool:
        """Lens single-file signature: net.*-stripped keys carry both
        ``.attn.img_qkv.weight`` and ``.attn.txt_qkv.weight`` (dual-stream DiT).
        Operates on key NAMES only (usable against a shard weight_map)."""
        stripped = [k[len("net."):] if k.startswith("net.") else k for k in keys]
        has_img_qkv = any(k.endswith(".attn.img_qkv.weight") for k in stripped)
        has_txt_qkv = any(k.endswith(".attn.txt_qkv.weight") for k in stripped)
        return has_img_qkv and has_txt_qkv

    @staticmethod
    def _keys_look_ideogram4(keys) -> bool:
        """Ideogram 4 combined single-file signature: the asymmetric-CFG branch is
        stored under the ``unconditional_transformer.`` prefix alongside the
        conditional ``transformer.`` branch. The unconditional prefix is unique to
        Ideogram 4 (no other arch bundles a second transformer), so it disambiguates
        from minit2i (``transformer.model.net.``) / krea2 (``transformer.text_fusion.``).
        Key NAMES only (usable against a shard weight_map)."""
        has_uncond = any(k.startswith("unconditional_transformer.") for k in keys)
        has_cond = any(k.startswith("transformer.") for k in keys)
        return has_uncond and has_cond

    @staticmethod
    def _keys_look_anima(keys) -> bool:
        """Anima single-file signature (net.*-stripped ``blocks.*`` DiT with
        AdaLN-LoRA). Delegates to the anima loader's key-name check so the
        signature stays defined in one place. Key NAMES only."""
        try:
            from core.models.anima.anima_loader import is_anima_state_dict_keys
            return bool(is_anima_state_dict_keys(list(keys)))
        except Exception:
            return False

    @staticmethod
    def _looks_like_acestep_dir(model_path: str) -> bool:
        """ACE-Step 1.5 flat ComfyUI-style tree: a `diffusion_models/` subfolder
        containing one of the EXACT known `acestep_v1.5_{turbo,sft,base}.safetensors`
        filenames. Exact-name (not glob) so this cannot collide with Anima's
        similarly-shaped bare `diffusion_models/+text_encoders/+vae/` layout.

        A QUANTIZED EXPORT is accepted too, and needs the second clause: the
        weight-only int8/FP8 export writes `<root>/diffusion_models/<stem>.safetensors`
        under a name of the user's choosing (`suggested_output_path` derives it
        from the source directory), so it matches none of the exact names and
        this probe used to answer False for a tree the loader can read perfectly
        well. Rather than loosening to a glob -- which is exactly what would
        collide with Anima -- any OTHER `.safetensors` in that subfolder is
        identified by its own safetensors metadata (`modelspec.architecture` /
        `model_type` == "acestep"), which `acestep_export_metadata` writes and
        no Anima file carries. Header read only; no tensor bytes."""
        try:
            if not os.path.isdir(model_path):
                return False
            from core.models.acestep.loader import ACESTEP_DIT_PATTERNS
            dit_dir = os.path.join(model_path, "diffusion_models")
            if not os.path.isdir(dit_dir):
                return False
            if any(os.path.isfile(os.path.join(dit_dir, name)) for name in ACESTEP_DIT_PATTERNS):
                return True
            from safetensors import safe_open
            for name in sorted(os.listdir(dit_dir)):
                if not name.endswith(".safetensors"):
                    continue
                try:
                    with safe_open(os.path.join(dit_dir, name), framework="pt") as f:
                        md = f.metadata() or {}
                except Exception:
                    continue
                if "acestep" in (md.get("modelspec.architecture", ""), md.get("model_type", "")):
                    return True
            return False
        except Exception:
            return False

    @staticmethod
    def _looks_like_minimax_h3(model_path: str) -> bool:
        """MiniMax-H3, from a directory OR a single DiT ``.safetensors``.

        Three accepted spellings, all cheap (JSON reads and safetensors HEADERS
        only -- no tensor bytes):

        * a directory whose ``model_index.json`` declares
          ``MiniMaxH3ModularPipeline`` (MiniMax's own config-only tree). The
          class name is unique, so it cannot collide with any other arch's
          diffusers-dir signature;
        * the ComfyUI-style flat tree: a ``diffusion_models/`` subfolder holding
          a DiT whose KEY NAMES carry the MiniMax-H3 signature. Keyed on the
          keys rather than on the filename because the filename is the user's to
          choose; ACE-Step's similarly-shaped ``diffusion_models/`` tree is
          already matched (on exact filenames + metadata) by the branch above
          this one in ``detect_model_type``, and Anima's is matched below it, so
          a key-name probe here cannot steal either;
        * a single DiT ``.safetensors`` with that same key signature, wherever
          it lives.

        Delegates the signature itself to the loader package so it stays defined
        in exactly one place.
        """
        try:
            from core.models.minimax_h3.loader import (
                MINIMAX_H3_PIPELINE_CLASS, is_minimax_h3_safetensors,
            )

            if os.path.isdir(model_path):
                index = os.path.join(model_path, "model_index.json")
                if os.path.isfile(index):
                    try:
                        with open(index, "r", encoding="utf-8") as f:
                            if json.load(f).get("_class_name") == MINIMAX_H3_PIPELINE_CLASS:
                                return True
                    except Exception:
                        pass
                dit_dir = os.path.join(model_path, "diffusion_models")
                if not os.path.isdir(dit_dir):
                    return False
                for name in sorted(os.listdir(dit_dir)):
                    if name.endswith(".safetensors") and is_minimax_h3_safetensors(
                            os.path.join(dit_dir, name)):
                        return True
                return False
            if isinstance(model_path, str) and model_path.endswith(".safetensors") \
                    and os.path.isfile(model_path):
                return is_minimax_h3_safetensors(model_path)
            return False
        except Exception:
            return False

    @staticmethod
    def _is_krea2_safetensors(model_path: str) -> bool:
        """Open a .safetensors file and check for the Krea 2 key signature."""
        try:
            from safetensors import safe_open
            with safe_open(model_path, framework="pt", device="cpu") as f:
                return ModelLoader._keys_look_krea2(list(f.keys()), f.metadata())
        except Exception:
            return False

    @staticmethod
    def _map_model_type_string(mt: str) -> Optional[str]:
        """Map a metadata model_type string (incl. aliases) to a ModelType, or None."""
        mt = (mt or "").strip().lower()
        if not mt:
            return None
        if mt in ("flux2", "flux.2", "flux2-klein", "flux.2-klein"):
            return "flux2"
        if mt in ("sdxl", "sd-xl", "stable-diffusion-xl", "stable_diffusion_xl"):
            return "sdxl"
        if mt in ("sd15", "sd-1.5", "sd_1.5", "stable-diffusion", "stable_diffusion", "sd"):
            return "sd15"
        if mt in ("zimage", "z-image"):
            return "zimage"
        if mt in ("minit2i", "krea2", "anima", "lens", "ideogram4"):
            return mt
        if mt == "siglip2_vision_encoder":
            return "vision_encoder"
        return None

    @staticmethod
    def detect_model_type(model_path: str) -> ModelType:
        """Detect if model is SD1.5, SDXL, Z-Image, DEUS, or FLUX.2 based on config or structure

        Supports:
        - Z-Image diffusers format (directory with transformer/, vae/, etc.)
        - Z-Image Comfy format (single safetensors with transformer weights only)
        - FLUX.2 Klein (single safetensors with Flux2Transformer2DModel weights)
        - SD1.5/SDXL diffusers and safetensors
        Note: DEUS support has been removed (architecture no longer maintained)
        """
        # From-scratch MiniT2I sentinel ("scratch:minit2i:<variant>:<vae_type>"):
        # not a filesystem path — handled by the in-memory build path in the loader.
        if isinstance(model_path, str) and model_path.startswith("scratch:minit2i:"):
            return "minit2i"

        # sushiUI shard index (<stem>.safetensors.index.json): read metadata
        # model_type first; else probe weight_map KEY NAMES (no tensor open).
        if isinstance(model_path, str) and model_path.endswith(".safetensors.index.json"):
            try:
                with open(model_path, encoding="utf-8") as f:
                    index = json.load(f)
                md = index.get("metadata", {}) or {}
                keys = list((index.get("weight_map", {}) or {}).keys())
                mapped = ModelLoader._map_model_type_string(str(md.get("model_type", "")))
                if mapped is not None:
                    return mapped
                # Key-name signature fallback (minit2i / krea2 / ideogram4 shard).
                if (any(k.startswith("transformer.model.net.") for k in keys)
                        or any(k.startswith("model.net.double_blocks.") for k in keys)):
                    return "minit2i"
                # Ideogram 4 combined shard: unique unconditional_transformer. prefix.
                if ModelLoader._keys_look_ideogram4(keys):
                    return "ideogram4"
                if ModelLoader._keys_look_krea2(keys, md):
                    return "krea2"
                # Lens net.* dual-stream DiT signature (runs before Anima; the
                # two key sets are disjoint but metadata already resolves ties).
                if ModelLoader._keys_look_lens(keys):
                    return "lens"
                # Anima net.* cosmos-style DiT signature.
                if ModelLoader._keys_look_anima(keys):
                    return "anima"
                # FLUX.2 diffusers-layout key signature.
                if (any(k.startswith("time_guidance_embed.") for k in keys)
                        and any(k.startswith("double_stream_modulation_") for k in keys)):
                    return "flux2"
                # Z-Image key signature.
                zi = ["cap_embedder", "t_embedder", "context_refiner"]
                if all(any(k.startswith(p) for k in keys) for p in zi):
                    return "zimage"
            except Exception as e:
                print(f"[ModelLoader] Could not read shard index {model_path}: {e}")
            return "sd15"

        # Lens detection (microsoft/Lens diffusers directory or HF repo)
        if os.path.isdir(model_path):
            model_index_path = os.path.join(model_path, "model_index.json")
            if os.path.exists(model_index_path):
                try:
                    with open(model_index_path, "r") as f:
                        idx = json.load(f)
                    if idx.get("_class_name") == "LensPipeline":
                        return "lens"
                except Exception:
                    pass
            # Fallback: transformer/config.json with LensTransformer2DModel architecture
            transformer_config_path = os.path.join(model_path, "transformer", "config.json")
            if os.path.exists(transformer_config_path):
                try:
                    with open(transformer_config_path, "r") as f:
                        tcfg = json.load(f)
                    if "LensTransformer2DModel" in tcfg.get("architectures", []):
                        return "lens"
                except Exception:
                    pass

            # ACE-Step 1.5 detection (flat ComfyUI-style tree: diffusion_models/
            # + vae/ + text_encoders/, no model_index.json / config.json anywhere).
            # Matched on an EXACT acestep_v1.5_{turbo,sft,base}.safetensors filename
            # inside diffusion_models/ so this cannot collide with Anima's bare
            # diffusion_models/+text_encoders/+vae/ layout (different filenames).
            if ModelLoader._looks_like_acestep_dir(model_path):
                return "acestep"

            # MiniMax-H3 detection (flat ComfyUI-style tree keyed on the DiT's
            # KEY NAMES, or MiniMax's config-only directory keyed on
            # model_index.json's MiniMaxH3ModularPipeline). After ACE-Step, whose
            # tree has the same shape and is matched on exact filenames, and
            # before Anima's split-files probe further down for the same reason.
            if ModelLoader._looks_like_minimax_h3(model_path):
                return "minimax_h3"

            # LTX-2.3 detection (diffusers directory only: model_index.json with
            # _class_name == "LTX2Pipeline"). Unique class name, so it cannot
            # collide with the other archs' diffusers-dir signatures. Base repo
            # is diffusers-dir only — no single-file variant.
            if os.path.exists(model_index_path):
                try:
                    with open(model_index_path, "r") as f:
                        idx = json.load(f)
                    if idx.get("_class_name") == "LTX2Pipeline":
                        return "ltx2"
                except Exception:
                    pass

            # Krea 2 detection (diffusers directory: Krea2Pipeline / Krea2Transformer2DModel)
            if os.path.exists(model_index_path):
                try:
                    with open(model_index_path, "r") as f:
                        idx = json.load(f)
                    if idx.get("_class_name") == "Krea2Pipeline":
                        return "krea2"
                except Exception:
                    pass
            if os.path.exists(transformer_config_path):
                try:
                    with open(transformer_config_path, "r") as f:
                        tcfg = json.load(f)
                    # Krea2 single-stream MMDiT: unique config keys (text fusion + 3-axis rope).
                    if tcfg.get("_class_name") == "Krea2Transformer2DModel" or (
                        "Krea2Transformer2DModel" in tcfg.get("architectures", [])
                    ) or ("num_layerwise_text_blocks" in tcfg and "axes_dims_rope" in tcfg):
                        return "krea2"
                except Exception:
                    pass

            # Ideogram 4 detection (diffusers directory: Ideogram4Pipeline / Ideogram4Transformer2DModel)
            if os.path.exists(model_index_path):
                try:
                    with open(model_index_path, "r") as f:
                        idx = json.load(f)
                    if idx.get("_class_name") == "Ideogram4Pipeline":
                        return "ideogram4"
                except Exception:
                    pass
            if os.path.exists(transformer_config_path):
                try:
                    with open(transformer_config_path, "r") as f:
                        tcfg = json.load(f)
                    # Ideogram4 single-stream DiT: unique mrope_section + llm_features_dim config keys.
                    if tcfg.get("_class_name") == "Ideogram4Transformer2DModel" or (
                        "mrope_section" in tcfg and "llm_features_dim" in tcfg
                    ):
                        return "ideogram4"
                    # MiniT2I MM-JiT (pixel-space): unique config class / keys.
                    if tcfg.get("_class_name") == "MiniT2IMMJiTModel" or (
                        "depth_double" in tcfg and "pca_channels" in tcfg
                    ):
                        return "minit2i"
                except Exception:
                    pass

            # MiniT2I repo root (.../MiniT2I with MiniT2IPipeline model_index) or a
            # container (.../minit2i with MiniT2I/ inside): detect by scanning for a
            # variant dir within 2 levels so the loader can resolve it to a variant.
            if os.path.isdir(model_path) and ModelLoader._dir_contains_minit2i(model_path):
                return "minit2i"

        # MiniMax-H3 single-file detection (the Comfy DiT: token_refiner. plus a
        # MiniMax-only key -- see `keys_look_minimax_h3`, whose second clause
        # deliberately excludes the diffusers spellings LTX-2.3 shares). Runs before the
        # Lens/Anima net.* probes; the key sets are disjoint, but this file lives
        # in a `diffusion_models/` folder that Anima's split-layout probe also
        # inspects, so ordering it first keeps the answer independent of the
        # directory it happens to sit in.
        if isinstance(model_path, str) and model_path.endswith(".safetensors") \
                and os.path.isfile(model_path) and ModelLoader._looks_like_minimax_h3(model_path):
            return "minimax_h3"

        # Lens single-file detection (full-FT save: net.* DiT). Metadata-first,
        # with a net.* key-signature fallback. Runs BEFORE the Anima net.* probe;
        # Lens keys (transformer_blocks.*.attn.img_qkv/txt_qkv) are disjoint from
        # Anima's (blocks.*.self_attn/cross_attn), so there is no real ambiguity,
        # but metadata resolves any edge case deterministically.
        if isinstance(model_path, str) and model_path.endswith(".safetensors") and os.path.isfile(model_path):
            try:
                from safetensors import safe_open
                with safe_open(model_path, framework="pt", device="cpu") as _lf:
                    _lmd = _lf.metadata() or {}
                    _lkeys = list(_lf.keys())
                _mt = str(_lmd.get("model_type", "")).strip().lower()
                _arch = str(_lmd.get("modelspec.architecture", "")).strip().lower()
                if _mt == "lens" or _arch == "lens":
                    return "lens"
                # Key-signature fallback (net.*-prefixed Lens DiT).
                if ModelLoader._keys_look_lens(_lkeys):
                    return "lens"
            except Exception as e:
                print(f"[ModelLoader] Lens detection skipped: {e}")

        # Anima detection (split-files layout or single DiT safetensors)
        try:
            from core.models.anima.anima_loader import (
                detect_anima_split_layout, is_anima_safetensors,
            )
            if os.path.isdir(model_path):
                if detect_anima_split_layout(model_path):
                    return "anima"
            elif model_path.endswith(".safetensors"):
                # If the file is inside a split_files/diffusion_models/ tree, treat as Anima —
                # but only after ruling out Krea 2 by key signature, since Comfy-Org/Krea-2
                # also ships under split_files/diffusion_models/.
                if detect_anima_split_layout(model_path):
                    if not ModelLoader._is_krea2_safetensors(model_path):
                        return "anima"
                # Otherwise inspect keys.
                if is_anima_safetensors(model_path):
                    return "anima"
        except Exception as e:
            print(f"[ModelLoader] Anima detection skipped: {e}")

        # Z-Image detection (diffusers format)
        if os.path.isdir(model_path):
            # Z-Image has transformer/ directory with unique config
            transformer_config = os.path.join(model_path, "transformer", "config.json")
            if os.path.exists(transformer_config):
                try:
                    with open(transformer_config, 'r') as f:
                        config = json.load(f)
                        # Z-Image has unique structure with axes_dims, rope_theta
                        if "axes_dims" in config and "rope_theta" in config:
                            return "zimage"
                except Exception as e:
                    pass

            # Check for SDXL indicators
            config_path = os.path.join(model_path, "model_index.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # SDXL uses different UNet config
                    if "_class_name" in config and "XL" in config["_class_name"]:
                        return "sdxl"

        # Check safetensors files
        if model_path.endswith('.safetensors'):
            try:
                from safetensors import safe_open
                with safe_open(model_path, framework="pt", device="cpu") as f:
                    keys = list(f.keys())
                    metadata = f.metadata() or {}

                    # MiniT2I single-file (bundled FLAN-T5 + MM-JiT transformer).
                    if (metadata.get("model_type", "").lower() == "minit2i"
                            or any(k.startswith("transformer.model.net.") for k in keys)
                            or any(k.startswith("model.net.double_blocks.") for k in keys)):
                        return "minit2i"

                    # Ideogram 4 combined single-file (both transformers bundled;
                    # unique unconditional_transformer. prefix, or explicit metadata).
                    if (str(metadata.get("model_type", "")).lower() == "ideogram4"
                            or ModelLoader._keys_look_ideogram4(keys)):
                        return "ideogram4"

                    # Krea 2 single-file (see _keys_look_krea2 for the signatures).
                    if ModelLoader._keys_look_krea2(keys, metadata):
                        return "krea2"

                    # Priority 1: Check metadata for explicit model_type
                    if "model_type" in metadata:
                        model_type = metadata["model_type"].lower() if isinstance(metadata["model_type"], str) else str(metadata["model_type"]).lower()

                        # SigLIP2 Vision Encoder detection (training checkpoint format)
                        if model_type == "siglip2_vision_encoder":
                            return "vision_encoder"

                        if model_type in ["flux2", "flux.2", "flux2-klein", "flux.2-klein"]:
                            return "flux2"
                        elif model_type in ["sdxl", "sd-xl", "stable-diffusion-xl", "stable_diffusion_xl"]:
                            return "sdxl"
                        elif model_type in ["sd15", "sd-1.5", "sd_1.5", "stable-diffusion", "stable_diffusion", "sd"]:
                            return "sd15"
                        elif model_type == "zimage":
                            return "zimage"

                    # Priority 2: FLUX.2 detection by state_dict keys
                    # FLUX.2 has two key formats:
                    # - Diffusers format: time_guidance_embed, double_stream_modulation_*, single_stream_modulation
                    # - BFL/Comfy format: double_blocks.*.img_attn, single_blocks.*, guidance_in

                    # Diffusers format detection
                    has_time_guidance_embed = any(k.startswith('time_guidance_embed.') for k in keys)
                    has_double_stream_modulation = any(k.startswith('double_stream_modulation_') for k in keys)
                    has_single_stream_modulation = any(k.startswith('single_stream_modulation.') for k in keys)
                    has_single_transformer_blocks = any(k.startswith('single_transformer_blocks.') for k in keys)

                    if has_time_guidance_embed and has_double_stream_modulation and has_single_stream_modulation:
                        return "flux2"

                    # BFL/Comfy format detection (double_blocks.*.img_attn, single_blocks.*)
                    has_double_blocks = any(k.startswith('double_blocks.') for k in keys)
                    has_single_blocks = any(k.startswith('single_blocks.') for k in keys)
                    has_img_attn = any('.img_attn.' in k for k in keys)
                    has_guidance_in = any(k.startswith('guidance_in.') for k in keys)

                    if has_double_blocks and has_single_blocks and has_img_attn:
                        return "flux2"

                    # DEUS support removed - architecture no longer maintained
                    # Priority 3: DEUS detection by state_dict keys
                    # DEUS uses SigLIP-2 text encoder with key prefix "conditioner.embedders.0.model."
                    # AND it has U-Net keys (unlike Z-Image which uses transformer)
                    has_unet_keys = any(k.startswith('model.diffusion_model.') for k in keys)

                    # DEUS-specific detection removed:
                    # has_siglip2_keys = any(
                    #     k.startswith('conditioner.embedders.0.model.text_model.embeddings') for k in keys
                    # )
                    # has_siglip2_layers = any(
                    #     k.startswith('conditioner.embedders.0.model.text_model.encoder.layers') for k in keys
                    # )
                    # has_dual_clip = (
                    #     any(k.startswith('conditioner.embedders.0.transformer') for k in keys) and
                    #     any(k.startswith('conditioner.embedders.1.') for k in keys)
                    # )
                    # if has_unet_keys and has_siglip2_keys and has_siglip2_layers and not has_dual_clip:
                    #     print(f"[ModelLoader] Detected DEUS model (SigLIP-2 text encoder): {model_path}")
                    #     return "deus"

                    # Priority 3: SD/SDXL detection
                    if has_unet_keys:
                        # This is SD or SDXL, not Z-Image
                        # SDXL detection by file size (>6GB) or specific keys
                        file_size = os.path.getsize(model_path) / (1024**3)  # GB
                        if file_size > 6:
                            return "sdxl"
                        else:
                            return "sd15"

                    # Z-Image Comfy format detection
                    # Z-Image transformer has unique keys WITHOUT U-Net structure
                    # Check for required indicators (cap_embedder, t_embedder, context_refiner)
                    # Note: x_embedder may be "x_embedder" or "all_x_embedder" depending on architecture
                    required_indicators = ['cap_embedder', 't_embedder', 'context_refiner']
                    has_required = all(any(k.startswith(indicator) for k in keys) for indicator in required_indicators)
                    has_x_embedder = any(k.startswith('x_embedder') or k.startswith('all_x_embedder') for k in keys)

                    if has_required and has_x_embedder:
                        return "zimage"

                    # SigLIP2 Vision Encoder detection by key structure
                    # Our saved format: embeddings.patch_embedding.weight (no prefix)
                    # HuggingFace format: vision_model.embeddings.patch_embedding.weight
                    has_ve_direct = any(k == 'embeddings.patch_embedding.weight' or k.startswith('embeddings.patch_embedding.') for k in keys)
                    has_ve_prefixed = any(k.startswith('vision_model.embeddings.') for k in keys)
                    has_header_token = 'header_token' in keys
                    has_encoder_layers = any(('encoder.layers.' in k) for k in keys)
                    # VE files have no U-Net, flux, or zimage keys
                    if (has_ve_direct or has_ve_prefixed or has_header_token) and has_encoder_layers and not has_unet_keys:
                        return "vision_encoder"

                    # Fallback: SDXL detection by file size
                    file_size = os.path.getsize(model_path) / (1024**3)  # GB
                    if file_size > 6:
                        return "sdxl"
            except Exception as e:
                print(f"[ModelLoader] Warning: Could not read safetensors: {e}")
                # Fallback to file size check
                file_size = os.path.getsize(model_path) / (1024**3)  # GB
                if file_size > 6:
                    return "sdxl"

        return "sd15"

    @staticmethod
    def _convert_comfy_to_official_state_dict(
        comfy_state_dict: dict,
        n_heads: int,
        n_kv_heads: int,
        dim: int
    ) -> dict:
        """Convert ComfyUI's state dict to official Z-Image format

        ComfyUI format:
            - attention.qkv.weight: [n_heads*head_dim + 2*n_kv_heads*head_dim, dim] (fused QKV)
            - attention.out.weight: [dim, n_heads*head_dim]
            - attention.q_norm.weight / k_norm.weight
            - x_embedder.weight/bias: Single embedder
            - final_layer.linear.weight/bias: Single final layer

        Official format:
            - attention.to_q/to_k/to_v.weight: Split Q/K/V
            - attention.to_out.0.weight: Output projection
            - attention.norm_q/norm_k.weight: Norm layers
            - all_x_embedder.{patch_size}-{aspect}.weight/bias: Multi-resolution embedders
            - all_final_layer.{patch_size}-{aspect}.linear.weight/bias: Multi-resolution final layers

        Args:
            comfy_state_dict: State dict from Comfy-format safetensors
            n_heads: Number of attention heads
            n_kv_heads: Number of key/value heads
            dim: Model dimension

        Returns:
            Converted state dict in official format
        """
        head_dim = dim // n_heads
        official_state_dict = {}

        # Default resolution key (patch_size=2, aspect_ratio=1:1)
        default_resolution_key = "2-1"

        print(f"[ModelLoader] Converting ComfyUI format to official Z-Image format")
        print(f"  - Attention layers: n_heads={n_heads}, n_kv_heads={n_kv_heads}, dim={dim}, head_dim={head_dim}")
        print(f"  - Using default resolution key: {default_resolution_key}")

        for key, value in comfy_state_dict.items():
            # Split fused QKV weights
            if ".qkv.weight" in key:
                q_dim = n_heads * head_dim
                kv_dim = n_kv_heads * head_dim

                q_weight = value[:q_dim, :]
                k_weight = value[q_dim:q_dim + kv_dim, :]
                v_weight = value[q_dim + kv_dim:q_dim + 2*kv_dim, :]

                base_key = key.replace(".qkv.weight", "")
                official_state_dict[f"{base_key}.to_q.weight"] = q_weight
                official_state_dict[f"{base_key}.to_k.weight"] = k_weight
                official_state_dict[f"{base_key}.to_v.weight"] = v_weight

            # Rename output projection
            elif ".out.weight" in key:
                new_key = key.replace(".out.weight", ".to_out.0.weight")
                official_state_dict[new_key] = value

            # Rename norm layers
            elif ".q_norm.weight" in key:
                new_key = key.replace(".q_norm.weight", ".norm_q.weight")
                official_state_dict[new_key] = value
            elif ".k_norm.weight" in key:
                new_key = key.replace(".k_norm.weight", ".norm_k.weight")
                official_state_dict[new_key] = value

            # Map x_embedder to all_x_embedder with resolution key
            elif key.startswith("x_embedder."):
                param_name = key.replace("x_embedder.", "")
                new_key = f"all_x_embedder.{default_resolution_key}.{param_name}"
                official_state_dict[new_key] = value
                print(f"  Mapped {key} -> {new_key}")

            # Map final_layer to all_final_layer with resolution key
            elif key.startswith("final_layer."):
                param_name = key.replace("final_layer.", "")
                new_key = f"all_final_layer.{default_resolution_key}.{param_name}"
                official_state_dict[new_key] = value
                print(f"  Mapped {key} -> {new_key}")

            # Copy all other keys as-is
            else:
                official_state_dict[key] = value

        print(f"[ModelLoader] Conversion complete: {len(comfy_state_dict)} keys -> {len(official_state_dict)} keys")
        return official_state_dict

    @staticmethod
    def _build_zimage_transformer_from_state(
        transformer_config: dict,
        n_layers: int,
        in_channels: int,
        state_dict: dict,
        layout: str,
        torch_dtype: torch.dtype,
        *,
        path: str = "<state dict>",
    ):
        """Build the Z-Image DiT for ``state_dict`` and install the weights into it.

        Everything between "the geometry is decided" and "the weights are in":
        the meta-device construction, the ComfyUI-to-official key rewrite, the
        weight-only-quantized swap, and the strict load. Split out of
        ``load_zimage_from_comfy_safetensors`` -- which otherwise also downloads a
        base repo and builds a VAE, a text encoder, a tokenizer and a scheduler --
        so the quantized decision can be exercised END TO END on a tiny synthetic
        geometry (``quantized_checkpoint_guard_test.ScalelessLoadMatrixTest``)
        rather than through a proxy. Ideogram 4's
        ``_build_ideogram4_transformer_from_state`` exists for exactly the same
        reason and is the precedent that test is already written against.

        ``layout`` is ``_normalize_zimage_state_dict``'s verdict: ``"official"``
        (split q/k/v, multi-resolution embedders -- what a live module's
        ``state_dict()`` and every sushiUI save produce) or ``"comfy"`` (fused
        ``attention.qkv``, single-resolution embedders).

        ``state_dict`` is consumed: the rewrite branch builds a second dict and
        the original is dropped, so a caller must not reuse it afterwards.
        """
        from core.models.zimage_transformer import ZImageTransformer2DModel

        print("[ModelLoader] Creating Z-Image transformer...")
        with torch.device("meta"):
            transformer = ZImageTransformer2DModel(
                all_patch_size=tuple(transformer_config["all_patch_size"]),
                all_f_patch_size=tuple(transformer_config["all_f_patch_size"]),
                in_channels=in_channels,
                dim=transformer_config["dim"],
                n_layers=n_layers,
                n_refiner_layers=transformer_config["n_refiner_layers"],
                n_heads=transformer_config["n_heads"],
                n_kv_heads=transformer_config["n_kv_heads"],
                norm_eps=transformer_config["norm_eps"],
                qk_norm=transformer_config["qk_norm"],
                cap_feat_dim=transformer_config["cap_feat_dim"],
                rope_theta=transformer_config["rope_theta"],
                t_scale=transformer_config["t_scale"],
                axes_dims=transformer_config["axes_dims"],
                axes_lens=transformer_config["axes_lens"],
            ).to(torch_dtype)

        # Convert Comfy format (fused QKV) to official format (separate Q/K/V).
        # sushiUI full-FT saves are ALREADY in official layout - skip conversion.
        if layout == "official":
            print("[ModelLoader] State dict already in official Z-Image layout; skipping conversion")
        else:
            print("[ModelLoader] Converting Comfy format to official format...")
            state_dict = ModelLoader._convert_comfy_to_official_state_dict(
                state_dict,
                transformer_config["n_heads"],
                transformer_config["n_kv_heads"],
                transformer_config["dim"]
            )

        # Weight-only quantized checkpoints (offline
        # subapps/fp8_quantize/quantize_transformer_fp8.py --format int8, or
        # POST /models/export-quantized on a runtime-converted transformer).
        # Their Linears must become Int8Linear/Fp8Linear BEFORE the load: a
        # quantized weight reaching load_state_dict as a plain nn.Linear has its
        # per-row '.weight_scale' rejected (strict=True raises here; a
        # strict=False loader would only print) and its int8 CODES written into
        # the parameter -- a model that loads and generates noise.
        #
        # Narrowed by scaled_quantization_report: a checkpoint whose weights are
        # float8 with NO '.weight_scale' anywhere is a plain dtype cast (the
        # ComfyUI "fp8" distribution shape), not a scaled quantization. It needs
        # no swap and must keep loading exactly as it always did -- which on THIS
        # loader means the float8 tensors must be cast BEFORE the load, because
        # the load is ``assign=True`` (the module is built on meta, so assignment
        # is the only option) and assignment would otherwise install float8
        # parameters that no nn.Linear forward can multiply. Anima is the same
        # case and uses the same helper.
        from core.models.common.quantized_checkpoint_guard import (
            cast_float8_tensors, quantized_state_dict_report,
            scaled_quantization_report, verify_quantized_swap,
        )
        census = quantized_state_dict_report(state_dict)
        quant_report = scaled_quantization_report(
            census, arch="Z-Image", path=path, label="transformer")
        if quant_report is not None:
            if layout != "official":
                # Nothing legitimate produces this pair. Both writers of a
                # quantized Z-Image artifact emit official-layout keys (the
                # offline tool refuses a comfy-layout source outright; the
                # runtime export reads the live module, which the loader has
                # already converted), and the comfy->official rewrite would chunk
                # a fused '.weight_scale' alongside its weight and produce
                # something that looks right.
                raise RuntimeError(
                    f"the Z-Image transformer checkpoint ({path}) is weight-only "
                    f"QUANTIZED but its keys are in the ComfyUI (fused-qkv) layout. Only "
                    f"the official split layout is supported for a quantized file: the "
                    f"fused-qkv rewrite would split a per-row weight_scale alongside its "
                    f"weight and silently mis-scale every attention projection.")
            # Swapped INSIDE a meta context so the replacement modules' buffers
            # land on meta like the ones they replace -- assign=True below
            # installs the file's tensors over them, so a real allocation here
            # would be ~6 GB of int8 written once and thrown away.
            with torch.device("meta"):
                swapped = ModelLoader._swap_zimage_quantized_linears(
                    transformer, state_dict, torch_dtype)
            # The swap helpers require BOTH the '.weight_scale' sibling and the
            # weight dtype, while the census fires on either, so "we took this
            # branch" does not mean "every quantized layer is now a quantized
            # module".
            verify_quantized_swap(quant_report, swapped, arch="Z-Image",
                                  path=path, label="transformer")
        elif census is not None:
            # The pure float8 cast. See the paragraph above: assign=True makes
            # this cast mandatory rather than cosmetic.
            state_dict = cast_float8_tensors(state_dict, torch_dtype)

        transformer.load_state_dict(state_dict, strict=True, assign=True)
        return transformer

    @staticmethod
    def _swap_zimage_quantized_linears(model, sd: dict, dtype: torch.dtype) -> int:
        """Replace Z-Image ``nn.Linear``s that have a quantized saved weight. Returns the count.

        A no-op (and silent) on an ordinary bf16 checkpoint, so it is safe to
        call unconditionally; the caller gates on
        ``scaled_quantization_report`` only to know whether it must then run
        ``verify_quantized_swap``.

        INT8 and e4m3 are detected INDEPENDENTLY and both swaps run, because
        ``quantize_transformer_fp8.py --format int8`` emits a MIXED checkpoint on
        purpose: a layer whose per-row crest factor makes int8 worse than e4m3
        falls back to e4m3 in the same file. Each detector and each swap helper
        gates on the weight DTYPE as well as the shared ``.weight_scale`` suffix,
        so neither can claim the other's layers and the call order does not
        matter. Same helpers and same reasoning as
        ``_swap_flux2_quantized_linears`` and ``anima_loader._swap_quantized_linears``;
        Z-Image needs no prefix argument because a quantized artifact carries the
        module tree with no wrapper (see ``EXPORT_LAYOUTS["zimage"]``, where the
        empty prefix is a requirement of ``detect_model_type``'s key probes).

        The returned count is NOT decorative: the caller compares it against
        ``quantized_state_dict_report`` (``verify_quantized_swap``) and refuses
        the load when they disagree.
        """
        try:
            from core.models.ideogram4.vendor.int8_linear import (
                is_int8_state_dict, swap_linears_to_int8,
            )
            from core.models.ideogram4.vendor.fp8_linear import (
                is_fp8_state_dict, swap_linears_to_fp8,
            )
        except Exception as e:
            print(f"[ModelLoader] Z-Image weight-only quant support unavailable ({e}); "
                  f"the checkpoint would load as a silently wrong model")
            raise
        has_int8 = bool(is_int8_state_dict(sd))
        has_fp8 = bool(is_fp8_state_dict(sd))
        if not (has_int8 or has_fp8):
            return 0
        n_int8 = swap_linears_to_int8(model, sd, compute_dtype=dtype) if has_int8 else 0
        n_fp8 = swap_linears_to_fp8(model, sd, compute_dtype=dtype) if has_fp8 else 0
        parts = []
        if n_int8:
            parts.append(f"{n_int8} Int8Linear")
        if n_fp8:
            parts.append(f"{n_fp8} Fp8Linear")
        print(f"[ModelLoader] weight-only quantized Z-Image transformer: swapped "
              f"{' + '.join(parts) or 'no'} Linear(s); the remaining Linears load as {dtype}")
        return n_int8 + n_fp8

    @staticmethod
    def _normalize_zimage_state_dict(raw: dict):
        """Split a Z-Image single-file save into transformer / VAE / TE sections
        and detect the transformer key layout.

        Handles BOTH:
          * genuine ComfyUI checkpoints — fused-qkv layout, unprefixed ``layers.N``
            keys, single-resolution ``x_embedder`` / ``final_layer``.
          * sushiUI full-FT saves (ZImageFullParameterAdapter) — OFFICIAL split
            Q/K/V keys with multi-resolution ``all_x_embedder`` / ``all_final_layer``
            under a ``model.diffusion_model.`` prefix, plus embedded
            ``first_stage_model.*`` (VAE) and ``text_encoders.qwen3.*`` (TE)
            sections.

        Returns ``(transformer_sd, vae_sd, te_sd, layout)`` where ``layout`` is
        ``"official"`` or ``"comfy"``; ``vae_sd`` / ``te_sd`` are ``None`` when the
        corresponding section is absent (genuine Comfy files have neither).
        """
        vae_sd: dict = {}
        te_sd: dict = {}
        transformer_raw: dict = {}
        for k, v in raw.items():
            if k.startswith("first_stage_model."):
                vae_sd[k[len("first_stage_model."):]] = v
            elif k.startswith("text_encoders."):
                # Strip ``text_encoders.<name>.`` (e.g. text_encoders.qwen3.).
                rest = k[len("text_encoders."):]
                te_sd[rest.split(".", 1)[1] if "." in rest else rest] = v
            else:
                transformer_raw[k] = v

        # Strip the ComfyUI-style ``model.diffusion_model.`` prefix if present.
        prefix = "model.diffusion_model."
        if any(k.startswith(prefix) for k in transformer_raw):
            transformer_raw = {
                (k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in transformer_raw.items()
            }

        # Layout detection by key signature. Official layout has split Q/K/V and/or
        # multi-resolution embedders; Comfy layout has a fused ``qkv`` projection.
        has_split_qkv = any(k.endswith(".to_q.weight") for k in transformer_raw)
        has_multi_res = any(
            k.startswith("all_x_embedder.") or k.startswith("all_final_layer.")
            for k in transformer_raw
        )
        has_fused_qkv = any(k.endswith(".qkv.weight") for k in transformer_raw)
        if has_split_qkv or has_multi_res:
            layout = "official"
        elif has_fused_qkv:
            layout = "comfy"
        else:
            # Ambiguous: keep the historical Comfy path (no-op conversion is safe).
            layout = "comfy"

        return transformer_raw, (vae_sd or None), (te_sd or None), layout

    @staticmethod
    def load_zimage_from_comfy_safetensors(
        file_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        base_model_repo: str = "Tongyi-MAI/Z-Image-Turbo"
    ) -> Dict[str, Any]:
        """Load Z-Image from ComfyUI Lumina format with weight conversion

        This loads ComfyUI-format safetensors and converts the weights to match
        the official Z-Image transformer structure by:
        1. Splitting fused QKV weights into separate Q/K/V layers
        2. Mapping single-resolution embedders to multi-resolution format
        3. Converting key names to match official structure

        Args:
            file_path: Path to Comfy-format Z-Image safetensors
            device: Device to load models on
            torch_dtype: Data type for model weights (bfloat16 recommended)
            base_model_repo: HuggingFace repo ID for base components (VAE, text encoder, etc.)

        Returns:
            Dict containing transformer, vae, text_encoder, tokenizer, scheduler
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Z-Image Comfy safetensors not found: {file_path}")

        print(f"[ModelLoader] Loading Z-Image from Comfy safetensors: {file_path}")
        print(f"[ModelLoader] Base components will be downloaded from: {base_model_repo}")

        # Use SushiUI's internal Z-Image implementation (standalone)
        try:
            from transformers import AutoModel, AutoTokenizer
            from safetensors.torch import load_file
            import importlib.util

            # Load SushiUI's internal Z-Image modules (standalone implementation)
            sushiui_models_path = Path(__file__).parent / "models"

            # Load transformer module (Block Swap integrated)
            transformer_spec = importlib.util.spec_from_file_location(
                "zimage_transformer",
                sushiui_models_path / "zimage_transformer.py"
            )
            transformer_module = importlib.util.module_from_spec(transformer_spec)
            # Register in sys.modules BEFORE exec_module (required for Flash Attention setup)
            import sys as _sys
            _sys.modules['zimage_transformer'] = transformer_module
            transformer_spec.loader.exec_module(transformer_module)
            ZImageTransformer2DModel = transformer_module.ZImageTransformer2DModel
            print(f"[ModelLoader] Loaded SushiUI Z-Image Transformer (standalone, Block Swap integrated)")

            # Load autoencoder module
            autoencoder_spec = importlib.util.spec_from_file_location(
                "zimage_autoencoder",
                sushiui_models_path / "zimage_autoencoder.py"
            )
            autoencoder_module = importlib.util.module_from_spec(autoencoder_spec)
            autoencoder_spec.loader.exec_module(autoencoder_module)
            AutoencoderKL = autoencoder_module.AutoencoderKL
            print(f"[ModelLoader] Loaded SushiUI Z-Image AutoencoderKL (standalone)")

            # Load scheduler module
            scheduler_spec = importlib.util.spec_from_file_location(
                "zimage_scheduler",
                sushiui_models_path / "zimage_scheduler.py"
            )
            scheduler_module = importlib.util.module_from_spec(scheduler_spec)
            scheduler_spec.loader.exec_module(scheduler_module)
            FlowMatchEulerDiscreteScheduler = scheduler_module.FlowMatchEulerDiscreteScheduler
            print(f"[ModelLoader] Loaded SushiUI Z-Image Scheduler (standalone)")

            # Step 1: Download base components from HuggingFace
            print(f"[ModelLoader] Downloading base components from {base_model_repo}...")
            from huggingface_hub import snapshot_download
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            base_model_path = snapshot_download(
                base_model_repo,
                cache_dir=cache_dir,
                allow_patterns=["vae/*", "text_encoder/*", "tokenizer/*", "scheduler/*", "transformer/config.json"]
            )
            print(f"[ModelLoader] Base components downloaded to: {base_model_path}")

            # Step 2: Load transformer config from base model
            transformer_config_path = os.path.join(base_model_path, "transformer", "config.json")
            with open(transformer_config_path, 'r') as f:
                transformer_config = json.load(f)

            # Step 3: Detect actual layer count from safetensors file. Read once
            # via the shared reader so a sharded <stem>.safetensors.index.json
            # path (full-FT save >10 GB) loads transparently alongside a plain
            # single-file save.
            print(f"[ModelLoader] Loading Comfy transformer weights from: {file_path}")
            from core.models.common.single_file_format import read_state_dict
            raw_state_dict, _raw_metadata = read_state_dict(file_path)

            # Normalize: both genuine Comfy checkpoints AND sushiUI full-FT saves
            # (official-layout keys under model.diffusion_model. + embedded VAE/TE).
            # Splits out first_stage_model.* / text_encoders.* so they never pollute
            # the strict transformer load, strips the prefix, and detects the layout.
            comfy_state_dict, embedded_vae_sd, embedded_te_sd, zimage_layout = \
                ModelLoader._normalize_zimage_state_dict(raw_state_dict)
            del raw_state_dict
            print(f"[ModelLoader] Z-Image transformer layout detected: {zimage_layout} "
                  f"({len(comfy_state_dict)} transformer tensors; "
                  f"embedded VAE={'yes' if embedded_vae_sd else 'no'}, "
                  f"embedded TE={'yes' if embedded_te_sd else 'no'})")

            # Auto-detect layer count from state_dict (supports pruned models)
            layer_indices = set()
            for key in comfy_state_dict.keys():
                if "layers." in key:
                    parts = key.split(".")
                    if len(parts) > 1 and parts[0] == "layers":
                        try:
                            layer_idx = int(parts[1])
                            layer_indices.add(layer_idx)
                        except ValueError:
                            pass

            actual_n_layers = max(layer_indices) + 1 if layer_indices else transformer_config["n_layers"]

            if actual_n_layers != transformer_config["n_layers"]:
                print(f"[ModelLoader] WARNING: Detected {actual_n_layers} layers in model file, "
                      f"but config specifies {transformer_config['n_layers']} layers.")
                print(f"[ModelLoader] Using detected layer count: {actual_n_layers}")

            # Step 4: Detect in_channels from actual safetensors file
            # If the model has x_embedder with 4 input channels, use SDXL VAE
            # If it has 16 input channels, use FLUX VAE (standard Z-Image)
            actual_in_channels = transformer_config["in_channels"]  # Default from config (16)

            # Get patch size from config for x_embedder shape calculation
            # all_patch_size can be [patch_h, patch_w] = [4, 4] or single value [4]
            all_patch_size = transformer_config["all_patch_size"]
            if isinstance(all_patch_size, (list, tuple)):
                if len(all_patch_size) == 2:
                    patch_h, patch_w = all_patch_size
                elif len(all_patch_size) == 1:
                    patch_h = patch_w = all_patch_size[0]
                else:
                    patch_h = patch_w = 4  # Default fallback
            else:
                patch_h = patch_w = all_patch_size  # Single integer
            patch_product = patch_h * patch_w  # 16 for standard Z-Image (4x4)
            # NOTE: no f_patch factor here, unlike subapps/fp8_quantize/quantize_transformer_fp8.py's
            # _zimage_config, which multiplies by all_f_patch_size[0]. The two agree numerically
            # only while all_f_patch_size == (1,), true of every published Z-Image config today.
            # Keep both formulas in sync if that ever changes.

            # Try to detect from state_dict (x_embedder weight shape)
            # ComfyUI format: x_embedder.weight shape is [dim, in_channels * patch_h * patch_w]
            # - Standard Z-Image (FLUX VAE): [3840, 16 * 4 * 4] = [3840, 256]
            # - SDXL VAE version: [3840, 4 * 4 * 4] = [3840, 64]
            for key in comfy_state_dict.keys():
                if "x_embedder" in key and "weight" in key:
                    weight_shape = comfy_state_dict[key].shape
                    if len(weight_shape) == 2:
                        # 2D weight: [dim, in_channels * patch_h * patch_w]
                        flattened_in = weight_shape[1]
                        detected_in_channels = flattened_in // patch_product
                        if detected_in_channels != actual_in_channels:
                            print(f"[ModelLoader] Detected in_channels={detected_in_channels} from x_embedder "
                                  f"(weight shape {weight_shape}, patch={patch_h}x{patch_w}, "
                                  f"config has {actual_in_channels})")
                            actual_in_channels = detected_in_channels
                    elif len(weight_shape) == 4:
                        # 4D weight: [dim, in_channels, patch_h, patch_w]
                        detected_in_channels = weight_shape[1]
                        if detected_in_channels != actual_in_channels:
                            print(f"[ModelLoader] Detected in_channels={detected_in_channels} from x_embedder "
                                  f"(weight shape {weight_shape}, config has {actual_in_channels})")
                            actual_in_channels = detected_in_channels
                    break

            # Determine VAE type based on in_channels
            use_sdxl_vae = (actual_in_channels == 4)
            if use_sdxl_vae:
                print(f"[ModelLoader] Z-Image model uses 4-channel latents (SDXL VAE)")
            else:
                print(f"[ModelLoader] Z-Image model uses {actual_in_channels}-channel latents (FLUX VAE)")

            # Step 5: build the transformer and install the weights. Extracted
            # into ``_build_zimage_transformer_from_state`` so the quantized
            # branch can be driven end to end by a test without HuggingFace,
            # a VAE or a text encoder -- the same reason Ideogram 4 has
            # ``_build_ideogram4_transformer_from_state``.
            transformer = ModelLoader._build_zimage_transformer_from_state(
                transformer_config, actual_n_layers, actual_in_channels,
                comfy_state_dict, zimage_layout, torch_dtype, path=file_path)
            del comfy_state_dict

            print(f"[ModelLoader] Moving transformer to {device}...")
            transformer = transformer.to(device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            transformer.eval()

            # Step 6: Load VAE based on in_channels
            if use_sdxl_vae:
                # Load SDXL VAE (4-channel latents)
                print("[ModelLoader] Loading SDXL VAE (4-channel latents)...")
                sdxl_vae_repo = "madebyollin/sdxl-vae-fp16-fix"
                from diffusers import AutoencoderKL as DiffusersAutoencoderKL
                vae = DiffusersAutoencoderKL.from_pretrained(
                    sdxl_vae_repo,
                    torch_dtype=torch.float32  # VAE in fp32 for quality
                )
                vae.to(device=device)
                vae.eval()
                zimage_vae_source = sdxl_vae_repo
                zimage_vae_path = None
                print(f"[ModelLoader] SDXL VAE loaded: latent_channels={vae.config.latent_channels}, "
                      f"scaling_factor={vae.config.scaling_factor}")
            else:
                # Load FLUX VAE (16-channel latents). Primary: the model's own vae/
                # from the base repo (Tongyi Z-Image); fallback: the shared flux1 store
                # (diffusers/FLUX.1-vae, Apache-2.0) when the own vae/ is absent.
                print("[ModelLoader] Loading FLUX VAE (16-channel latents)...")
                vae_path = os.path.join(base_model_path, "vae")
                if not os.path.isfile(os.path.join(vae_path, "config.json")):
                    try:
                        from core.models.common.vae_store import resolve_vae_dir
                        store_dir = resolve_vae_dir("flux1", model_own_vae=vae_path)
                        if store_dir and os.path.isdir(store_dir):
                            print(f"[ModelLoader] FLUX VAE from shared flux1 store: {store_dir}")
                            vae_path = store_dir
                    except Exception as _e:
                        print(f"[ModelLoader] flux1 VAE store resolution failed: {_e}")
                vae_config_path = os.path.join(vae_path, "config.json")
                with open(vae_config_path, 'r') as f:
                    vae_config = json.load(f)

                vae = AutoencoderKL(
                    in_channels=vae_config["in_channels"],
                    out_channels=vae_config["out_channels"],
                    down_block_types=tuple(vae_config["down_block_types"]),
                    up_block_types=tuple(vae_config["up_block_types"]),
                    block_out_channels=tuple(vae_config["block_out_channels"]),
                    layers_per_block=vae_config["layers_per_block"],
                    latent_channels=vae_config["latent_channels"],
                    norm_num_groups=vae_config["norm_num_groups"],
                    scaling_factor=vae_config["scaling_factor"],
                    shift_factor=vae_config.get("shift_factor"),
                    use_quant_conv=vae_config.get("use_quant_conv", True),
                    use_post_quant_conv=vae_config.get("use_post_quant_conv", True),
                    mid_block_add_attention=vae_config.get("mid_block_add_attention", True),
                )

                vae_weights_path = os.path.join(vae_path, "diffusion_pytorch_model.safetensors")
                vae_state_dict = load_file(vae_weights_path, device="cpu")
                vae.load_state_dict(vae_state_dict, strict=False)
                del vae_state_dict
                vae.to(device=device, dtype=torch.float32)  # VAE uses fp32
                vae.eval()
                zimage_vae_source = str(vae_path)
                zimage_vae_path = str(vae_path) if os.path.isdir(str(vae_path)) else None
                print(f"[ModelLoader] FLUX VAE loaded: latent_channels={vae.config.latent_channels}, "
                      f"scaling_factor={vae.config.scaling_factor}")

            # Reattach the embedded (trained) VAE weights when present, overriding
            # the base VAE downloaded above. Absent => keep the base VAE.
            if embedded_vae_sd is not None:
                ModelLoader._reattach_embedded_weights(vae, embedded_vae_sd, "VAE")
                vae.to(device=device, dtype=torch.float32)
                vae.eval()
                zimage_vae_source = "embedded (checkpoint)"
                zimage_vae_path = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[ModelLoader] Loading text encoder...")
            text_encoder_path = os.path.join(base_model_path, "text_encoder")
            text_encoder = AutoModel.from_pretrained(
                text_encoder_path,
                dtype=torch_dtype,
                trust_remote_code=True,
            )
            print(f"[ModelLoader] Moving text encoder to {device}...")
            text_encoder.to(device)
            text_encoder.eval()

            # Reattach the embedded (trained) text encoder weights when present.
            if embedded_te_sd is not None:
                ModelLoader._reattach_embedded_weights(text_encoder, embedded_te_sd, "text encoder")
                text_encoder.to(device)
                text_encoder.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[ModelLoader] Loading tokenizer...")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            tokenizer_path = os.path.join(base_model_path, "tokenizer")
            if not os.path.exists(tokenizer_path):
                tokenizer_path = text_encoder_path
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
            )

            print("[ModelLoader] Loading scheduler...")
            scheduler_path = os.path.join(base_model_path, "scheduler")
            scheduler_config_path = os.path.join(scheduler_path, "scheduler_config.json")
            with open(scheduler_config_path, 'r') as f:
                scheduler_config = json.load(f)

            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=scheduler_config.get("num_train_timesteps", 1000),
                shift=scheduler_config.get("shift", 1.0),
                use_dynamic_shifting=scheduler_config.get("use_dynamic_shifting", False),
            )

            vae_type = "sdxl" if use_sdxl_vae else "flux"
            print("[ModelLoader] Z-Image Comfy format loaded successfully")
            print(f"  - Transformer: {actual_n_layers} layers, in_channels={actual_in_channels}")
            print(f"  - VAE: {vae_type.upper()} (latent_channels={vae.config.latent_channels})")
            print(f"  - Text Encoder, Tokenizer, Scheduler: Loaded from {base_model_repo}")

            return {
                "transformer": transformer,
                "vae": vae,
                "vae_source": zimage_vae_source,
                "vae_path": zimage_vae_path,
                "text_encoder": text_encoder,
                "tokenizer": tokenizer,
                "scheduler": scheduler,
                "vae_type": vae_type,  # "sdxl" or "flux" - for TAESD preview selection
            }

        except Exception as e:
            print(f"[ModelLoader] Error loading Z-Image Comfy format: {e}")
            import traceback
            traceback.print_exc()
            raise

    # DEUS support removed - architecture no longer maintained
    # @staticmethod
    # def load_deus_from_safetensors(
    #     file_path: str,
    #     device: str = "cuda",
    #     torch_dtype: torch.dtype = torch.float16
    # ) -> Dict[str, Any]:
    #     """Load DEUS model from safetensors file using diffusers DeusPipeline
    #
    #     DEUS architecture uses:
    #     - SigLIP-2 text encoder (1152d, variable sequence length)
    #     - U-Net with RoPE 2D positional encoding
    #     - SDXL VAE (same as SDXL)
    #     - 2-Pass CFG for inference
    #
    #     Args:
    #         file_path: Path to DEUS safetensors file
    #         device: Device to load models on
    #         torch_dtype: Data type for model weights
    #
    #     Returns:
    #         Dict containing unet, vae, text_encoder, tokenizer, scheduler, processor
    #     """
    #     if not os.path.exists(file_path):
    #         raise FileNotFoundError(f"DEUS model file not found: {file_path}")
    #
    #     print(f"[ModelLoader] Loading DEUS model from: {file_path}")
    #     print(f"[ModelLoader] Using diffusers DeusPipeline.from_single_file()")
    #
    #     try:
    #         # Import diffusers DEUS pipeline from custom location
    #         # Note: This uses the local diffusers installation at D:\celll1\diffusers
    #         sys.path.insert(0, "D:\\celll1\\diffusers\\src")
    #         from diffusers.pipelines.deus import DeusPipeline
    #         from diffusers.schedulers import EulerDiscreteScheduler
    #
    #         # Load pipeline using from_single_file
    #         pipeline = DeusPipeline.from_single_file(
    #             file_path,
    #             torch_dtype=torch_dtype,
    #         )
    #
    #         # Move components to device
    #         print(f"[ModelLoader] Moving DEUS components to {device}...")
    #         pipeline.text_encoder.to(device)
    #         pipeline.unet.to(device)
    #         # VAE uses fp32 for better quality (same as SDXL)
    #         pipeline.vae.to(device, dtype=torch.float32)
    #
    #         print(f"[ModelLoader] DEUS model loaded successfully")
    #         print(f"  - U-Net: {type(pipeline.unet).__name__}")
    #         print(f"  - Text Encoder: {type(pipeline.text_encoder).__name__}")
    #         print(f"  - VAE: {type(pipeline.vae).__name__}")
    #         print(f"  - Scheduler: {type(pipeline.scheduler).__name__}")
    #
    #         # Return components in dict format (consistent with Z-Image)
    #         return {
    #             "unet": pipeline.unet,
    #             "vae": pipeline.vae,
    #             "text_encoder": pipeline.text_encoder,
    #             "tokenizer": pipeline.processor.tokenizer if hasattr(pipeline.processor, 'tokenizer') else None,
    #             "processor": pipeline.processor,
    #             "scheduler": pipeline.scheduler,
    #             "pipeline": pipeline,  # Keep reference to pipeline for encode_prompt etc.
    #         }
    #
    #     except Exception as e:
    #         print(f"[ModelLoader] Error loading DEUS model: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         raise

    @staticmethod
    def _split_flux2_sushiui_state_dict(raw: dict):
        """Split a sushiUI FLUX.2 full-FT save into transformer / VAE / TE sub-dicts.

        The adapter writes transformer keys under ``model.diffusion_model.``, VAE
        under ``first_stage_model.`` and TE under ``text_encoders.qwen3.``. Returns
        ``(transformer_sd, vae_sd, te_sd)`` with the section prefixes stripped;
        ``vae_sd`` / ``te_sd`` are empty dicts when absent.
        """
        transformer_sd: dict = {}
        vae_sd: dict = {}
        te_sd: dict = {}
        for key, value in raw.items():
            if key.startswith('model.diffusion_model.'):
                transformer_sd[key[len('model.diffusion_model.'):]] = value
            elif key.startswith('first_stage_model.'):
                vae_sd[key[len('first_stage_model.'):]] = value
            elif key.startswith('text_encoders.qwen3.'):
                te_sd[key[len('text_encoders.qwen3.'):]] = value
            elif key.startswith('text_encoders.'):
                rest = key[len('text_encoders.'):]
                te_sd[rest.split('.', 1)[1] if '.' in rest else rest] = value
            else:
                # Unprefixed keys (already diffusers-layout transformer) pass through.
                transformer_sd[key] = value
        return transformer_sd, vae_sd, te_sd

    @staticmethod
    def _swap_flux2_quantized_linears(model, sd: dict, dtype: torch.dtype) -> int:
        """Replace FLUX.2 ``nn.Linear``s that have a quantized saved weight. Returns the count.

        A no-op (and silent) on an ordinary bf16 checkpoint, so it is safe to call
        unconditionally; the caller gates on ``quantized_state_dict_report`` only
        to know whether to skip the blanket dtype cast.

        INT8 and e4m3 are detected INDEPENDENTLY and both swaps run, because
        ``quantize_transformer_fp8.py --format int8`` emits a MIXED checkpoint on
        purpose: a layer whose per-row crest factor makes int8 worse than e4m3
        falls back to e4m3 in the same file. Each detector and each swap helper
        gates on the weight DTYPE as well as the shared ``.weight_scale`` suffix,
        so neither can claim the other's layers and the call order does not
        matter. Same helpers, same reasoning as
        ``anima_loader._swap_quantized_linears`` and
        ``krea2/vendor/single_file.build_krea2_transformer`` -- FLUX.2 needs no
        prefix argument because its single files carry the diffusers module tree
        with no wrapper.

        The returned count is NOT decorative: the caller compares it against
        ``quantized_state_dict_report`` (``verify_quantized_swap``) and refuses
        the load when they disagree, because a quantized layer this helper did
        not take is a layer whose codes ``load_state_dict`` will cast into a bf16
        parameter without a word.

        The caller also casts the module to ``dtype`` BEFORE calling this, and
        skips the usual post-load cast. Not because a later cast would corrupt an
        e4m3 buffer -- bf16 represents every e4m3 value exactly, and the dequant
        path still applies the scale -- but because it would double the buffer and
        drop ``Fp8Linear``'s ``_scaled_mm`` fast path, which gates on the weight
        dtype.
        """
        try:
            from core.models.ideogram4.vendor.int8_linear import (
                is_int8_state_dict, swap_linears_to_int8,
            )
            from core.models.ideogram4.vendor.fp8_linear import (
                is_fp8_state_dict, swap_linears_to_fp8,
            )
        except Exception as e:
            print(f"[ModelLoader] FLUX.2 weight-only quant support unavailable ({e}); "
                  f"the checkpoint would load as a silently wrong model")
            raise
        has_int8 = bool(is_int8_state_dict(sd))
        has_fp8 = bool(is_fp8_state_dict(sd))
        if not (has_int8 or has_fp8):
            return 0
        n_int8 = swap_linears_to_int8(model, sd, compute_dtype=dtype) if has_int8 else 0
        n_fp8 = swap_linears_to_fp8(model, sd, compute_dtype=dtype) if has_fp8 else 0
        parts = []
        if n_int8:
            parts.append(f"{n_int8} Int8Linear")
        if n_fp8:
            parts.append(f"{n_fp8} Fp8Linear")
        print(f"[ModelLoader] weight-only quantized FLUX.2 transformer: swapped "
              f"{' + '.join(parts) or 'no'} Linear(s); the remaining Linears load as {dtype}")
        return n_int8 + n_fp8

    @staticmethod
    def load_flux2_from_safetensors(
        file_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        base_model_repo: str = None
    ) -> Dict[str, Any]:
        """Load FLUX.2 Klein from safetensors file (transformer weights only)

        The safetensors file should contain only transformer weights.
        VAE, text encoder, tokenizer, and scheduler are downloaded from HuggingFace.

        Args:
            file_path: Path to FLUX.2 transformer safetensors
            device: Device to load models on
            torch_dtype: Data type for model weights (bfloat16 recommended)
            base_model_repo: HuggingFace repo ID for base components (auto-detected if None)

        Returns:
            Dict with transformer, vae, text_encoder, tokenizer, scheduler components
        """
        try:
            from diffusers import Flux2Transformer2DModel, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
            from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast
            from huggingface_hub import snapshot_download
            from safetensors import safe_open
            import os

            print(f"[ModelLoader] Loading FLUX.2 Klein from safetensors: {file_path}")

            # Single consolidated read: handles both a plain <stem>.safetensors
            # and a sharded <stem>.safetensors.index.json path, returning the
            # full state dict (CPU) plus the metadata block. All subsequent
            # key/metadata probes reuse these instead of re-opening the file.
            from core.models.common.single_file_format import read_state_dict
            transformer_state_dict, metadata = read_state_dict(file_path)
            all_keys = list(transformer_state_dict.keys())
            print(f"[ModelLoader] Loaded {len(transformer_state_dict)} tensors from safetensors")

            # Auto-detect base_model_repo from safetensors metadata if not specified
            if base_model_repo is None:
                print(f"[ModelLoader] Auto-detecting HuggingFace repo from metadata...")
                if True:
                    # Priority 1: Check metadata for base_model_repo (from finetuned models)
                    if "base_model_repo" in metadata:
                        base_model_repo = metadata["base_model_repo"]
                        print(f"[ModelLoader] Found base_model_repo in metadata: {base_model_repo}")
                    else:
                        # Priority 2: Detect from layer count (for original models)
                        # Klein Base 4B: 48 single layers
                        # Klein 4B (distilled): 24 single layers
                        # Klein 9B (distilled): 36 single layers
                        num_single_layers = None
                        for key in all_keys:
                            if "single_blocks.47." in key or "single_transformer_blocks.47." in key:
                                num_single_layers = 48
                                break
                            elif "single_blocks.35." in key or "single_transformer_blocks.35." in key:
                                num_single_layers = 36
                                break
                            elif "single_blocks.23." in key or "single_transformer_blocks.23." in key:
                                num_single_layers = 24
                                break

                        # Determine repo based on layer count
                        if num_single_layers == 48:
                            base_model_repo = "black-forest-labs/FLUX.2-klein-base-4B"
                            print(f"[ModelLoader] Detected Klein Base 4B (48 single layers)")
                        elif num_single_layers == 36:
                            base_model_repo = "black-forest-labs/FLUX.2-klein-9B"
                            print(f"[ModelLoader] Detected Klein 9B (36 single layers)")
                        elif num_single_layers == 24:
                            base_model_repo = "black-forest-labs/FLUX.2-klein-4B"
                            print(f"[ModelLoader] Detected Klein 4B (24 single layers)")
                        else:
                            # Fallback: use Base 4B as default
                            base_model_repo = "black-forest-labs/FLUX.2-klein-base-4B"
                            print(f"[ModelLoader] Could not detect model variant, defaulting to Klein Base 4B")

                print(f"[ModelLoader] Using HuggingFace repo: {base_model_repo}")

            # Step 1: Download base components from HuggingFace.
            # LICENSE: the VAE is deliberately EXCLUDED here and resolved separately
            # from the Apache-2.0 FLUX.2 store (see Step 4) so it never comes from the
            # FLUX.2-klein-9B repo (FLUX Non-Commercial), regardless of the detected
            # transformer variant. TE/tokenizer/scheduler still come from
            # base_model_repo — their cross-variant config compatibility (9B vs 4B)
            # could NOT be verified: FLUX.2-klein-9B is a gated HF repo (anonymous
            # and current-token fetches of text_encoder/config.json,
            # tokenizer/tokenizer_config.json and scheduler/scheduler_config.json
            # all return 403, checked 2026-07-03), and the 4B TE is Qwen3-4B
            # (hidden_size 2560) while the 9B variant plausibly pairs a larger
            # Qwen3 — so they are left per-variant rather than silently swapped.
            print(f"[ModelLoader] Downloading base components from {base_model_repo}...")
            cache_dir = snapshot_download(
                base_model_repo,
                allow_patterns=["text_encoder/*", "tokenizer/*", "scheduler/*", "transformer/config.json", "model_index.json"],
            )
            print(f"[ModelLoader] Base components downloaded to: {cache_dir}")

            # Step 2: Load transformer config
            transformer_config_path = os.path.join(cache_dir, "transformer", "config.json")
            with open(transformer_config_path, 'r') as f:
                transformer_config = json.load(f)
            print(f"[ModelLoader] Transformer config loaded:")
            print(f"  - in_channels: {transformer_config.get('in_channels', 128)}")
            print(f"  - num_layers: {transformer_config.get('num_layers', 8)} (dual stream)")
            print(f"  - num_single_layers: {transformer_config.get('num_single_layers', 48)} (single stream)")
            print(f"  - num_attention_heads: {transformer_config.get('num_attention_heads', 48)}")
            print(f"  - attention_head_dim: {transformer_config.get('attention_head_dim', 128)}")

            # Load is_distilled flag
            # Priority 1: Check safetensors metadata (from finetuned models)
            # Priority 2: Check model_index.json from HuggingFace repo
            is_distilled = False
            if True:
                # metadata already read once above (read_state_dict).
                if "is_distilled" in metadata:
                    is_distilled = metadata["is_distilled"].lower() == "true"
                    print(f"  - is_distilled (from metadata): {is_distilled}")
                else:
                    # Fallback to model_index.json
                    model_index_path = os.path.join(cache_dir, "model_index.json")
                    if os.path.exists(model_index_path):
                        with open(model_index_path, 'r') as f_index:
                            model_index = json.load(f_index)
                            is_distilled = model_index.get("is_distilled", False)
                    print(f"  - is_distilled (from model_index.json): {is_distilled}")

            # Step 3: Assemble transformer weights (already read once above via
            # read_state_dict, so no second file open — this also lets a sharded
            # <stem>.safetensors.index.json path load transparently).
            print(f"[ModelLoader] Using FLUX.2 transformer weights from: {file_path}")

            # Detect state_dict format and convert if needed
            # FLUX.2 state_dict can be in 3 formats:
            # 1. BFL/Comfy format: double_blocks.*, single_blocks.* (original BFL weights)
            # 2. Diffusers format: time_guidance_embed.*, double_stream_modulation_*, single_transformer_blocks.*
            # 3. SushiUI/musubi training format: model.diffusion_model.* prefix (ComfyUI-style but with diffusers keys inside)
            # WEIGHT-ONLY QUANTIZED CHECKPOINTS (int8 / e4m3 weights + per-row
            # ``.weight_scale`` siblings, written by
            # subapps/fp8_quantize/quantize_transformer_fp8.py or by
            # POST /models/export-quantized). FLUX.2 reads them: the matching
            # nn.Linear modules are replaced by Int8Linear / Fp8Linear BEFORE the
            # load, below. What must NOT happen is reaching the plain strict=False
            # load with them still as nn.Linear -- every scale would land in
            # unexpected_keys and every quantized weight would be cast into a bf16
            # parameter, i.e. the int8 CODES written as if they were the weights.
            # Detected here, before the format branch, because format 1 and 3 are
            # refused for it (see below).
            #
            # Narrowed to the SCALED case by ``scaled_quantization_report``: a
            # checkpoint whose weights are float8 with no scales anywhere is a
            # plain dtype cast (the ComfyUI "fp8" distribution shape), not a
            # weight-only quantization. It needs no swap, it is not refused for
            # its key layout below, and it loads the way it always did -- the
            # cast back to bf16 is exact.
            from core.models.common.quantized_checkpoint_guard import (
                quantized_state_dict_report, scaled_quantization_report,
            )
            quant_report = scaled_quantization_report(
                quantized_state_dict_report(transformer_state_dict),
                arch="FLUX.2", path=file_path, label="transformer")

            sample_keys = list(transformer_state_dict.keys())[:5]
            is_bfl_format = any(k.startswith('double_blocks.') for k in transformer_state_dict.keys())
            is_sushiui_format = any(k.startswith('model.diffusion_model.') for k in transformer_state_dict.keys())

            # Embedded (trained) VAE / TE sections from sushiUI full-FT saves; stay
            # empty for standard single-file transformer checkpoints.
            embedded_vae_state_dict: dict = {}
            embedded_te_state_dict: dict = {}

            if quant_report is not None and (is_bfl_format or is_sushiui_format):
                # Only the diffusers layout is supported for a quantized file, and
                # nothing legitimate produces the other two: the offline tool
                # applies the BFL->diffusers transform itself and always emits
                # diffusers keys, and the sushiUI full-FT save is written by a
                # trainer, which refuses a quantized base outright
                # (adapters/base_adapter.reject_quantized_base). Refusing is
                # cheap insurance against the alternative -- diffusers' converter
                # would happily chunk a fused ``.weight_scale`` alongside its
                # weight and produce something that looks right.
                raise RuntimeError(
                    f"the FLUX.2 transformer checkpoint ({file_path}) is weight-only "
                    f"QUANTIZED ({quant_report['scale_keys']} '.weight_scale' key(s)) AND in "
                    f"the {'BFL/Comfy' if is_bfl_format else 'sushiUI/musubi'} key layout. "
                    f"Quantized FLUX.2 checkpoints are supported only in the diffusers key "
                    f"layout, which is what both writers of this format emit. Quantize the "
                    f"unquantized checkpoint again with "
                    f"subapps/fp8_quantize/quantize_transformer_fp8.py --arch flux2.")

            if is_bfl_format:
                print(f"[ModelLoader] Detected BFL/Comfy format state_dict, converting to diffusers format...")
                from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
                transformer_state_dict = convert_flux2_transformer_checkpoint_to_diffusers(transformer_state_dict)
                print(f"[ModelLoader] Converted to diffusers format ({len(transformer_state_dict)} tensors)")
            elif is_sushiui_format:
                # SushiUI/musubi training saves with "model.diffusion_model." prefix.
                # Split transformer keys from embedded VAE ("first_stage_model.*")
                # and TE ("text_encoders.qwen3.*") sections so trained VAE/TE weights
                # can be reattached below instead of always re-downloading them.
                print(f"[ModelLoader] Detected SushiUI/musubi training format state_dict, stripping prefix...")
                original_count = len(transformer_state_dict)
                transformer_state_dict, embedded_vae_state_dict, embedded_te_state_dict = \
                    ModelLoader._split_flux2_sushiui_state_dict(transformer_state_dict)
                print(f"[ModelLoader] Extracted {len(transformer_state_dict)} transformer tensors "
                      f"from {original_count} total tensors "
                      f"(embedded VAE={'yes' if embedded_vae_state_dict else 'no'}, "
                      f"embedded TE={'yes' if embedded_te_state_dict else 'no'})")
            else:
                print(f"[ModelLoader] State dict is already in diffusers format")

            # Create transformer model
            print(f"[ModelLoader] Creating Flux2Transformer2DModel...")
            transformer = Flux2Transformer2DModel(**transformer_config)

            if quant_report is not None:
                # ORDER MATTERS. The dtype cast happens HERE, before the swap, and
                # the usual one after the load is skipped, because
                # nn.Module.to(dtype) casts every FLOATING-POINT buffer: it leaves
                # an int8 weight alone but WOULD convert an e4m3 weight buffer (a
                # mixed artifact always has some) to bf16. That conversion is
                # value-preserving -- e4m3 has 3 mantissa bits and an exponent
                # range wholly inside bf16's, so all 256 codes survive it exactly,
                # and the dequant path (weight.to(x.dtype) * weight_scale) keeps
                # producing the same numbers. What it costs is real but narrower
                # than "garbage": the weight buffer doubles in size, and
                # Fp8Linear._scaled_mm_forward gates on
                # ``w.dtype is FP8_WEIGHT_DTYPE`` (fp8_linear.py), so the W8A8 fast
                # path is silently lost for the rest of the process. Casting first
                # and swapping second gives the quantized modules their exact
                # buffer dtypes and never revisits them.
                transformer = transformer.to(dtype=torch_dtype)
                swapped = ModelLoader._swap_flux2_quantized_linears(
                    transformer, transformer_state_dict, torch_dtype)
                # The swap helpers require BOTH the scale sibling and the weight
                # dtype, while the report fires on either -- so "we took the new
                # branch" does not mean "every quantized layer is now a quantized
                # module". Anything left over would fall through to the plain
                # strict=False load below and be cast into bf16 parameters.
                from core.models.common.quantized_checkpoint_guard import (
                    verify_quantized_swap,
                )
                verify_quantized_swap(quant_report, swapped, arch="FLUX.2",
                                      path=file_path, label="transformer")

            # Load weights
            missing_keys, unexpected_keys = transformer.load_state_dict(transformer_state_dict, strict=False)
            if missing_keys:
                print(f"[ModelLoader] WARNING: Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"[ModelLoader] WARNING: Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"[ModelLoader] WARNING: Unexpected keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"[ModelLoader] WARNING: Unexpected keys: {unexpected_keys}")

            if quant_report is None:
                transformer = transformer.to(dtype=torch_dtype)
            print(f"[ModelLoader] Transformer loaded with {sum(p.numel() for p in transformer.parameters()):,} parameters")

            # Step 4: Load VAE — ALWAYS from the Apache-2.0 FLUX.2 store
            # (black-forest-labs/FLUX.2-klein-4B subfolder vae), NEVER from the
            # (possibly 9B) transformer variant repo. Falls back to the 4B repo
            # subfolder directly if the store cannot be resolved.
            print(f"[ModelLoader] Loading FLUX.2 VAE (Apache-2.0 store)...")
            flux2_vae_dir = None
            try:
                from core.models.common.vae_store import resolve_vae_dir
                flux2_vae_dir = resolve_vae_dir("flux2")
            except Exception as _e:
                print(f"[ModelLoader] FLUX.2 VAE store resolution failed: {_e}")
            if flux2_vae_dir and os.path.isdir(flux2_vae_dir):
                vae = AutoencoderKLFlux2.from_pretrained(
                    flux2_vae_dir, torch_dtype=torch.float32
                )
            else:
                vae = AutoencoderKLFlux2.from_pretrained(
                    "black-forest-labs/FLUX.2-klein-4B", subfolder="vae",
                    torch_dtype=torch.float32,  # VAE in fp32 for quality
                )
            print(f"[ModelLoader] VAE loaded: latent_channels={vae.config.latent_channels}")

            # Reattach the embedded (trained) VAE weights when present, overriding
            # the base VAE. Absent => keep the downloaded base VAE.
            if embedded_vae_state_dict:
                ModelLoader._reattach_embedded_weights(vae, embedded_vae_state_dict, "VAE")
                vae = vae.to(dtype=torch.float32)
                flux2_vae_source = "embedded (checkpoint)"
                flux2_vae_path = None
            elif flux2_vae_dir and os.path.isdir(flux2_vae_dir):
                flux2_vae_source = str(flux2_vae_dir)
                flux2_vae_path = str(flux2_vae_dir)
            else:
                flux2_vae_source = "black-forest-labs/FLUX.2-klein-4B (vae)"
                flux2_vae_path = None

            # Step 5: Load Text Encoder (Qwen3)
            print(f"[ModelLoader] Loading Qwen3 text encoder...")
            text_encoder = Qwen3ForCausalLM.from_pretrained(
                cache_dir,
                subfolder="text_encoder",
                torch_dtype=torch_dtype
            )
            print(f"[ModelLoader] Text encoder loaded: Qwen3ForCausalLM")

            # Reattach the embedded (trained) text encoder weights when present.
            if embedded_te_state_dict:
                ModelLoader._reattach_embedded_weights(text_encoder, embedded_te_state_dict, "text encoder")
                text_encoder = text_encoder.to(dtype=torch_dtype)

            # Step 6: Load Tokenizer
            print(f"[ModelLoader] Loading tokenizer...")
            tokenizer = Qwen2TokenizerFast.from_pretrained(
                cache_dir,
                subfolder="tokenizer"
            )
            print(f"[ModelLoader] Tokenizer loaded: Qwen2TokenizerFast")

            # Step 7: Load Scheduler
            print(f"[ModelLoader] Loading scheduler...")
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                cache_dir,
                subfolder="scheduler"
            )
            print(f"[ModelLoader] Scheduler loaded: FlowMatchEulerDiscreteScheduler")

            print(f"[ModelLoader] FLUX.2 Klein loaded successfully")
            print(f"  - Transformer: {transformer_config.get('num_layers', 8)} dual + {transformer_config.get('num_single_layers', 48)} single layers")
            print(f"  - VAE: AutoencoderKLFlux2 (latent_channels={vae.config.latent_channels})")
            print(f"  - Text Encoder: Qwen3ForCausalLM")
            print(f"  - Tokenizer: Qwen2TokenizerFast (max_length=512)")
            print(f"  - Scheduler: FlowMatchEulerDiscreteScheduler")

            # Add is_distilled and base_model_repo to config dict (for inference and training)
            config_dict = transformer_config.copy()
            config_dict["is_distilled"] = is_distilled
            config_dict["base_model_repo"] = base_model_repo

            return {
                "transformer": transformer,
                "vae": vae,
                "vae_source": flux2_vae_source,
                "vae_path": flux2_vae_path,
                "text_encoder": text_encoder,
                "tokenizer": tokenizer,
                "scheduler": scheduler,
                "config": config_dict,
                "model_type": "flux2",  # Distinguish from Z-Image
            }

        except Exception as e:
            print(f"[ModelLoader] Error loading FLUX.2 model: {e}")
            import traceback
            traceback.print_exc()
            raise

    @staticmethod
    def load_from_safetensors(
        file_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16
    ) -> Union[StableDiffusionPipeline, Dict[str, Any]]:
        """Load model from .safetensors file

        Returns:
            - StableDiffusionPipeline for SD1.5/SDXL
            - Dict of components for Z-Image/DEUS
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")

        model_type = ModelLoader.detect_model_type(file_path)
        print(f"[ModelLoader] Detected model type: {model_type}")

        # FLUX.2 format
        if model_type == "flux2":
            print(f"[ModelLoader] Loading as FLUX.2 Klein (Flux2Transformer2DModel)")
            return ModelLoader.load_flux2_from_safetensors(file_path, device, torch.bfloat16)

        # Anima (DiT + Qwen3 + Qwen-Image VAE)
        if model_type == "anima":
            print(f"[ModelLoader] Loading as Anima (Cosmos-Predict2 DiT)")
            return ModelLoader.load_anima_from_files(file_path, device, torch.bfloat16)

        # Lens single-file (full-FT net.* DiT; TE/VAE/tokenizer resolved from dirs)
        if model_type == "lens":
            print(f"[ModelLoader] Loading as Lens (single-file DiT)")
            return ModelLoader.load_lens_from_path(file_path, torch.bfloat16)

        # Ideogram 4 combined single-file (both transformers; TE/VAE/tokenizer/
        # scheduler resolved from a sibling base diffusers directory)
        if model_type == "ideogram4":
            print(f"[ModelLoader] Loading as Ideogram 4 (combined single-file)")
            return ModelLoader.load_ideogram4_from_path(file_path, torch.bfloat16)

        # DEUS support removed - architecture no longer maintained
        # if model_type == "deus":
        #     print(f"[ModelLoader] Loading as DEUS (SigLIP-2 text encoder)")
        #     return ModelLoader.load_deus_from_safetensors(file_path, device, torch_dtype)

        # Z-Image Comfy format
        if model_type == "zimage":
            print(f"[ModelLoader] Loading as Z-Image (Comfy safetensors format)")
            return ModelLoader.load_zimage_from_comfy_safetensors(file_path, device, torch.bfloat16)

        # MiniT2I single-file (bundled FLAN-T5 + MM-JiT transformer)
        if model_type == "minit2i":
            print(f"[ModelLoader] Loading as MiniT2I (single-file)")
            return ModelLoader.load_minit2i_from_path(file_path, torch.bfloat16)

        # Krea 2 single-file (diffusers / raw / comfy / sushiUI TE+DiT combined)
        if model_type == "krea2":
            print(f"[ModelLoader] Loading as Krea 2 (single-file)")
            return ModelLoader.load_krea2_from_path(file_path, torch.bfloat16)

        # MiniMax-H3 DiT single file. Selecting the FILE rather than the tree is
        # how the transformer VARIANT is chosen: MiniMax ships two partitions
        # (`fl2va`, which serves txt2vid/img2vid/outpaint, and `ref2va`, which
        # serves /generate/ref2vid) that share every other component and are
        # otherwise indistinguishable -- same config, same byte size, no
        # distinguishing key -- so the filename is the only thing that says
        # which one this is. The loader walks up from the file to the tree and
        # takes the remaining components from it.
        if model_type == "minimax_h3":
            print(f"[ModelLoader] Loading as MiniMax-H3 (DiT single file; variant selected by file)")
            return ModelLoader.load_minimax_h3_from_path(file_path, torch.bfloat16)

        is_v_prediction = ModelLoader.detect_v_prediction(file_path)

        # Reconstruct the SD1.5 / SDXL pipeline (custom-arch aware). Shared with the
        # training-resume path so both honor SushiUI sushi.* metadata identically.
        pipeline = ModelLoader.reconstruct_sd_sdxl_pipeline(
            file_path, model_type, torch_dtype, device
        )

        # Configure scheduler for v-prediction if detected
        if is_v_prediction:
            print(f"[ModelLoader] Configuring scheduler for v-prediction model")
            ModelLoader._configure_v_prediction_scheduler(pipeline)

        # Move components to device individually (avoid pipeline.to() which can cause issues)
        print(f"[ModelLoader] Moving pipeline components to {device}...")
        print(f"[ModelLoader] DEBUG: Before any moves - pipeline.vae is not None: {pipeline.vae is not None}")
        print(f"[ModelLoader] DEBUG: Before any moves - 'vae' in components: {'vae' in pipeline.components}")

        # Move each component individually
        if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
            print(f"[ModelLoader] DEBUG: Moving text_encoder...")
            pipeline.text_encoder.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After text_encoder move - pipeline.vae is not None: {pipeline.vae is not None}")

        if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
            print(f"[ModelLoader] DEBUG: Moving text_encoder_2...")
            pipeline.text_encoder_2.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After text_encoder_2 move - pipeline.vae is not None: {pipeline.vae is not None}")

        if hasattr(pipeline, 'unet') and pipeline.unet is not None:
            print(f"[ModelLoader] DEBUG: Moving unet...")
            pipeline.unet.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After unet move - pipeline.vae is not None: {pipeline.vae is not None}")

        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
            print(f"[ModelLoader] DEBUG: Moving vae...")
            pipeline.vae.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After vae move - pipeline.vae is not None: {pipeline.vae is not None}")

        print(f"[ModelLoader] All components moved to {device}")
        print(f"[ModelLoader] DEBUG: After all moves - pipeline.vae is not None: {pipeline.vae is not None}")

        # Verify VAE exists after moving to device
        if not hasattr(pipeline, 'vae') or pipeline.vae is None:
            print(f"[ModelLoader] ERROR: VAE is missing after component move")
            print(f"[ModelLoader] Pipeline components: {list(pipeline.components.keys())}")
            raise RuntimeError("VAE is missing after loading model")
        else:
            print(f"[ModelLoader] VAE verified: {type(pipeline.vae).__name__}")

        return pipeline

    @staticmethod
    def reconstruct_sd_sdxl_pipeline(
        file_path: str,
        model_type: str,
        torch_dtype: torch.dtype,
        device: str = "cuda",
    ):
        """Reconstruct an SD1.5 / SDXL pipeline from a single-file checkpoint.

        Honors SushiUI custom-arch metadata:
        - sushi.vae_type / sushi.in_channels -> swap to a non-standard latent VAE
          (e.g. FLUX.1 16ch) and resize the U-Net conv_in/conv_out to match.
        - sushi.te_type (+ sushi.te_*) -> rebuild a swapped text encoder and its
          bridge adapters and attach them to the pipeline.
        Absent => standard SD1.5/SDXL via diffusers from_single_file (byte-identical
        behavior to the legacy inline path).

        Returns the pipeline WITHOUT device placement or v-prediction scheduler
        configuration so callers (inference load / training resume) finish setup as
        needed. When a custom text encoder is present it is attached as
        pipeline._sushi_te / _sushi_te_tokenizer / _sushi_te_adapters /
        _sushi_te_max_len / _sushi_te_hidden_layer (used by the inference encode
        path). A summary of the reconstructed architecture is always attached as
        pipeline._sushi_arch for callers that must rebuild trainer state.
        """
        # Custom SDXL architecture (SushiUI): non-standard latent VAE (e.g. FLUX.1 16ch).
        # Read sushi.vae_type / sushi.in_channels so the U-Net conv_in/out and the VAE
        # are reconstructed after load. Absent => standard SDXL (unchanged path).
        custom_vae_type = None
        custom_in_channels = None
        custom_te = None  # dict(te_type, hidden_layer, max_len, dim, embedded) when custom TE
        if model_type == "sdxl":
            try:
                from safetensors import safe_open
                with safe_open(file_path, framework="pt") as _f:
                    _md = _f.metadata() or {}
                _vt = (_md.get("sushi.vae_type") or "").strip().lower()
                if _vt and _vt not in ("none", "sdxl"):
                    custom_vae_type = _vt
                    custom_in_channels = int(_md.get("sushi.in_channels", "0") or 0) or None
                    print(f"[ModelLoader] Custom SDXL arch: vae_type={custom_vae_type}, "
                          f"in_channels={custom_in_channels}")
                _tt = (_md.get("sushi.te_type") or "").strip().lower()
                if _tt and _tt not in ("none", "clip"):
                    custom_te = {
                        "te_type": _tt,
                        "hidden_layer": int(_md.get("sushi.te_hidden_layer", "-2") or -2),
                        "max_len": int(_md.get("sushi.te_max_len", "256") or 256),
                        "dim": int(_md.get("sushi.te_dim", "0") or 0),
                        "embedded": _md.get("sushi.te_embedded") == "1",
                    }
                    print(f"[ModelLoader] Custom SDXL text encoder: {custom_te}")
            except Exception as _e:
                print(f"[ModelLoader] custom-arch metadata read failed (standard load): {_e}")

        # Check if VAE is embedded
        print(f"[ModelLoader] Checking if model has embedded VAE...")
        has_vae = ModelLoader.has_embedded_vae(file_path)
        print(f"[ModelLoader] VAE detection result: {'embedded' if has_vae else 'not embedded'}")

        # Load external VAE only if not embedded
        external_vae = None
        sushi_vae_source = None  # VAE identity string for generation metadata
        if custom_vae_type:
            # Custom high-spec VAE is registry-referenced (not embedded); load it here.
            from core.models.sdxl_custom_arch import load_alt_vae
            print(f"[ModelLoader] Loading custom registry VAE: {custom_vae_type}")
            external_vae = load_alt_vae(custom_vae_type, torch_dtype=torch_dtype)
            has_vae = False
            sushi_vae_source = f"custom registry VAE ({custom_vae_type})"
        elif not has_vae:
            if model_type == "sdxl":
                vae_repo = "madebyollin/sdxl-vae-fp16-fix"
            else:  # SD1.5
                vae_repo = "stabilityai/sd-vae-ft-mse-original"
            sushi_vae_source = vae_repo

            print(f"[ModelLoader] Model without embedded VAE detected")
            print(f"[ModelLoader] Loading external VAE: {vae_repo}")
            try:
                external_vae = AutoencoderKL.from_pretrained(
                    vae_repo,
                    torch_dtype=torch_dtype,
                    use_safetensors=True
                )
                print(f"[ModelLoader] External VAE loaded successfully")
            except Exception as e:
                print(f"[ModelLoader] ERROR: Failed to load external VAE: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed to load external VAE: {e}")

        # Override U-Net in/out channels ONLY for a custom-arch SDXL (e.g. 16ch latent).
        # Both are needed: num_in_channels overrides conv_in, but conv_out is otherwise
        # built at the SDXL default (4) from the LDM config and would mismatch the trained
        # 16ch conv_out during from_single_file. out_channels is a UNet2DConditionModel
        # __init__ kwarg picked up by single_file_model's config update. For standard SDXL
        # custom_in_channels is None and BOTH must be omitted (None would break conv_in).
        # The custom path always sets external_vae, so out_channels does not reach the VAE.
        _sf_kw = ({"num_in_channels": custom_in_channels, "out_channels": custom_in_channels}
                  if custom_in_channels else {})

        # Use single_file loading which is the standard way to load safetensors
        print(f"[ModelLoader] Loading as {'SDXL' if model_type == 'sdxl' else 'SD1.5'} (standard pipeline)")
        try:
            if model_type == "sdxl":
                # Only pass vae parameter if external VAE was loaded
                if external_vae is not None:
                    pipeline = StableDiffusionXLPipeline.from_single_file(
                        file_path,
                        **_sf_kw,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        vae=external_vae,
                    )
                else:
                    # Use embedded VAE (don't pass vae parameter)
                    pipeline = StableDiffusionXLPipeline.from_single_file(
                        file_path,
                        **_sf_kw,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                    )
            else:
                pipeline = StableDiffusionPipeline.from_single_file(
                    file_path,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                )
        except Exception as e:
            # Fallback: try with float32
            print(f"Failed to load with fp16, trying with fp32: {e}")
            if model_type == "sdxl":
                # Only pass vae parameter if external VAE was loaded
                if external_vae is not None:
                    pipeline = StableDiffusionXLPipeline.from_single_file(
                        file_path,
                        **_sf_kw,
                        torch_dtype=torch.float32,
                        use_safetensors=True,
                        vae=external_vae,
                    )
                else:
                    # Use embedded VAE (don't pass vae parameter)
                    pipeline = StableDiffusionXLPipeline.from_single_file(
                        file_path,
                        **_sf_kw,
                        torch_dtype=torch.float32,
                        use_safetensors=True,
                    )
            else:
                pipeline = StableDiffusionPipeline.from_single_file(
                    file_path,
                    torch_dtype=torch.float32,
                    use_safetensors=True,
                )

        # Custom SDXL arch: ensure conv_in/conv_out match the custom latent channels and
        # carry the trained weights (from_single_file overrides in_channels but not
        # out_channels). num_in_channels above loaded conv_in; resize fixes conv_out, then
        # both convs are assigned directly from the file for correctness.
        if custom_vae_type and custom_in_channels and hasattr(pipeline, "unet"):
            try:
                from core.models.sdxl_custom_arch import (
                    resize_unet_in_out, load_custom_convs_from_single_file,
                )
                resize_unet_in_out(pipeline.unet, custom_in_channels)
                load_custom_convs_from_single_file(pipeline.unet, file_path)
                print(f"[ModelLoader] Custom SDXL reconstructed: {custom_in_channels}ch latents "
                      f"({custom_vae_type} VAE)")
            except Exception as _re:
                print(f"[ModelLoader] ERROR reconstructing custom SDXL: {_re}")
                import traceback
                traceback.print_exc()

        # Custom SDXL text encoder: rebuild the swapped encoder + bridge adapters and
        # attach to the pipeline (the inference encode path uses them in place of CLIP).
        if custom_te is not None:
            try:
                from safetensors import safe_open
                from core.models.sdxl_te_registry import load_sdxl_te
                from core.models.sdxl_te_adapter import SDXLTEAdapters
                encoder, tokenizer, dim = load_sdxl_te(
                    custom_te["te_type"], dtype=torch_dtype, device=device,
                    max_len=custom_te["max_len"],
                )
                adapters = SDXLTEAdapters(dim).to(device=device, dtype=torch_dtype)
                with safe_open(file_path, framework="pt") as _f:
                    keys = list(_f.keys())
                    ad_sd = {k[len("sushi.te_adapter."):]: _f.get_tensor(k)
                             for k in keys if k.startswith("sushi.te_adapter.")}
                    if ad_sd:
                        adapters.load_state_dict(ad_sd)
                    if custom_te["embedded"]:
                        body_sd = {k[len("sushi.te_encoder."):]: _f.get_tensor(k)
                                   for k in keys if k.startswith("sushi.te_encoder.")}
                        if body_sd:
                            encoder.load_state_dict(body_sd, strict=False)
                            print(f"[ModelLoader] Loaded fine-tuned custom-TE encoder body from file")
                encoder.eval(); adapters.eval()
                pipeline._sushi_te = encoder
                pipeline._sushi_te_tokenizer = tokenizer
                pipeline._sushi_te_adapters = adapters
                pipeline._sushi_te_max_len = custom_te["max_len"]
                pipeline._sushi_te_hidden_layer = custom_te["hidden_layer"]
                pipeline._sushi_te_dim = dim
                pipeline._sushi_te_embedded = bool(custom_te["embedded"])
                print(f"[ModelLoader] Custom SDXL text encoder attached: {custom_te['te_type']} "
                      f"(dim={dim}, max_len={custom_te['max_len']})")
            except Exception as _te:
                print(f"[ModelLoader] ERROR reconstructing custom SDXL text encoder: {_te}")
                import traceback
                traceback.print_exc()

        # VAE identity for generation metadata. custom/external set above; a bare
        # embedded-VAE checkpoint leaves sushi_vae_source None -> record "embedded".
        pipeline._sushi_vae_source = sushi_vae_source or "embedded (checkpoint)"

        # Architecture summary for callers that must rebuild trainer state (resume).
        # None for a standard SD1.5/SDXL checkpoint.
        pipeline._sushi_arch = {
            "vae_type": custom_vae_type,
            "in_channels": custom_in_channels,
            "te_type": (custom_te or {}).get("te_type"),
            "te_dim": getattr(pipeline, "_sushi_te_dim", None),
            "te_max_len": (custom_te or {}).get("max_len"),
            "te_hidden_layer": (custom_te or {}).get("hidden_layer"),
            "te_embedded": (custom_te or {}).get("embedded"),
        }
        return pipeline

    @staticmethod
    def load_zimage_from_diffusers(
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ) -> Dict[str, Any]:
        """Load Z-Image from diffusers format directory

        Returns:
            Dict containing transformer, vae, text_encoder, tokenizer, scheduler
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Z-Image model directory not found: {model_path}")

        print(f"[ModelLoader] Loading Z-Image from: {model_path}")

        # Check if model_path is a single safetensors file (Comfy format) or directory (diffusers format)
        is_single_file = os.path.isfile(model_path) and model_path.endswith('.safetensors')

        if is_single_file:
            print(f"[ModelLoader] Detected Comfy format safetensors, delegating to Comfy loader")
            # Use existing Comfy loader (already handles weight conversion)
            components = ModelLoader.load_zimage_from_comfy_safetensors(
                file_path=model_path,
                device=device,
                torch_dtype=torch_dtype
            )
            return components
        else:
            # Diffusers format directory
            raise NotImplementedError(
                f"Z-Image diffusers format directory loading is not yet implemented.\n"
                f"Please use Comfy format (.safetensors) instead.\n"
                f"Convert your model using ComfyUI's 'Save Model' feature."
            )

    @staticmethod
    def load_from_diffusers(
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16
    ) -> Union[StableDiffusionPipeline, Dict[str, Any]]:
        """Load model from diffusers format directory

        Returns:
            - StableDiffusionPipeline for SD1.5/SDXL
            - Dict of components for Z-Image/DEUS
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        model_type = ModelLoader.detect_model_type(model_path)

        # DEUS support removed - architecture no longer maintained
        # if model_type == "deus":
        #     # DEUS diffusers format directory loading
        #     # For now, raise NotImplementedError (safetensors is the primary format)
        #     raise NotImplementedError(
        #         f"DEUS diffusers format directory loading is not yet implemented.\n"
        #         f"Please use safetensors format instead."
        #     )

        # Z-Image uses component-based loading
        if model_type == "zimage":
            return ModelLoader.load_zimage_from_diffusers(model_path, device, torch.bfloat16)

        # Anima split-files directory layout
        if model_type == "anima":
            print(f"[ModelLoader] Loading as Anima (split-files layout)")
            return ModelLoader.load_anima_from_files(model_path, device, torch.bfloat16)

        # Lens (microsoft/Lens) diffusers directory
        if model_type == "lens":
            print(f"[ModelLoader] Loading as Lens (diffusers directory)")
            return ModelLoader.load_lens_from_path(model_path, torch.bfloat16)

        # Ideogram 4 diffusers directory (dual-branch DiT + Qwen3-VL + AutoencoderKLFlux2)
        if model_type == "ideogram4":
            print(f"[ModelLoader] Loading as Ideogram 4 (diffusers directory)")
            return ModelLoader.load_ideogram4_from_path(model_path, torch.bfloat16)

        # MiniT2I diffusers directory (pixel-space MM-JiT + FLAN-T5, no VAE)
        if model_type == "minit2i":
            print(f"[ModelLoader] Loading as MiniT2I (diffusers directory)")
            return ModelLoader.load_minit2i_from_path(model_path, torch.bfloat16)

        # Krea 2 diffusers directory (single-stream MMDiT + Qwen3-VL + Qwen-Image VAE)
        if model_type == "krea2":
            print(f"[ModelLoader] Loading as Krea 2 (diffusers directory)")
            return ModelLoader.load_krea2_from_path(model_path, torch.bfloat16)

        # LTX-2.3 diffusers directory (joint audio+video MM-DiT + Gemma-3 + LTX2 VAEs)
        if model_type == "ltx2":
            print(f"[ModelLoader] Loading as LTX-2.3 (diffusers directory)")
            return ModelLoader.load_ltx2_from_path(model_path, torch.bfloat16)

        # ACE-Step 1.5 flat ComfyUI-style tree (2B DiT + Oobleck VAE + Qwen3-Embedding-0.6B)
        if model_type == "acestep":
            print(f"[ModelLoader] Loading as ACE-Step 1.5 (flat model tree)")
            return ModelLoader.load_acestep_from_path(model_path, torch.bfloat16)

        # MiniMax-H3 flat ComfyUI-style tree (pruned joint video+audio DiT +
        # Qwen3-VL text encoder + video and audio VAEs)
        if model_type == "minimax_h3":
            print(f"[ModelLoader] Loading as MiniMax-H3 (flat model tree)")
            return ModelLoader.load_minimax_h3_from_path(model_path, torch.bfloat16)

        is_v_prediction = ModelLoader.detect_v_prediction(model_path)

        if model_type == "sdxl":
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )

        # Configure scheduler for v-prediction if detected
        if is_v_prediction:
            print(f"[ModelLoader] Configuring scheduler for v-prediction model")
            ModelLoader._configure_v_prediction_scheduler(pipeline)

        # Move components to device individually (avoid pipeline.to() which can cause issues)
        print(f"[ModelLoader] Moving pipeline components to {device}...")
        print(f"[ModelLoader] DEBUG: Before any moves - pipeline.vae is not None: {pipeline.vae is not None}")
        print(f"[ModelLoader] DEBUG: Before any moves - 'vae' in components: {'vae' in pipeline.components}")

        # Move each component individually
        if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
            print(f"[ModelLoader] DEBUG: Moving text_encoder...")
            pipeline.text_encoder.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After text_encoder move - pipeline.vae is not None: {pipeline.vae is not None}")

        if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
            print(f"[ModelLoader] DEBUG: Moving text_encoder_2...")
            pipeline.text_encoder_2.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After text_encoder_2 move - pipeline.vae is not None: {pipeline.vae is not None}")

        if hasattr(pipeline, 'unet') and pipeline.unet is not None:
            print(f"[ModelLoader] DEBUG: Moving unet...")
            pipeline.unet.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After unet move - pipeline.vae is not None: {pipeline.vae is not None}")

        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
            print(f"[ModelLoader] DEBUG: Moving vae...")
            pipeline.vae.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After vae move - pipeline.vae is not None: {pipeline.vae is not None}")

        print(f"[ModelLoader] All components moved to {device}")
        print(f"[ModelLoader] DEBUG: After all moves - pipeline.vae is not None: {pipeline.vae is not None}")

        # Verify VAE exists after moving to device
        if not hasattr(pipeline, 'vae') or pipeline.vae is None:
            print(f"[ModelLoader] ERROR: VAE is missing after component move")
            print(f"[ModelLoader] Pipeline components: {list(pipeline.components.keys())}")
            raise RuntimeError("VAE is missing after loading model")
        else:
            print(f"[ModelLoader] VAE verified: {type(pipeline.vae).__name__}")

        return pipeline

    @staticmethod
    def load_from_huggingface(
        repo_id: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        revision: Optional[str] = None
    ) -> StableDiffusionPipeline:
        """Load model from HuggingFace repository"""
        # Lens (microsoft/Lens): detect by repo_id prefix
        if "microsoft/lens" in repo_id.lower() or repo_id.lower().endswith("/lens"):
            print(f"[ModelLoader] Loading as Lens (HuggingFace Hub): {repo_id}")
            return ModelLoader.load_lens_from_path(repo_id, torch.bfloat16)

        # Detect model type from repo_id or try loading
        if "xl" in repo_id.lower() or "sdxl" in repo_id.lower():
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                repo_id,
                torch_dtype=torch_dtype,
                revision=revision,
                use_safetensors=True,
            )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                repo_id,
                torch_dtype=torch_dtype,
                revision=revision,
                use_safetensors=True,
            )

        # Move components to device individually (avoid pipeline.to() which can cause issues)
        print(f"[ModelLoader] Moving pipeline components to {device}...")
        print(f"[ModelLoader] DEBUG: Before any moves - pipeline.vae is not None: {pipeline.vae is not None}")
        print(f"[ModelLoader] DEBUG: Before any moves - 'vae' in components: {'vae' in pipeline.components}")

        # Move each component individually
        if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
            print(f"[ModelLoader] DEBUG: Moving text_encoder...")
            pipeline.text_encoder.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After text_encoder move - pipeline.vae is not None: {pipeline.vae is not None}")

        if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
            print(f"[ModelLoader] DEBUG: Moving text_encoder_2...")
            pipeline.text_encoder_2.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After text_encoder_2 move - pipeline.vae is not None: {pipeline.vae is not None}")

        if hasattr(pipeline, 'unet') and pipeline.unet is not None:
            print(f"[ModelLoader] DEBUG: Moving unet...")
            pipeline.unet.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After unet move - pipeline.vae is not None: {pipeline.vae is not None}")

        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
            print(f"[ModelLoader] DEBUG: Moving vae...")
            pipeline.vae.to(device, dtype=torch_dtype)
            print(f"[ModelLoader] DEBUG: After vae move - pipeline.vae is not None: {pipeline.vae is not None}")

        print(f"[ModelLoader] All components moved to {device}")
        print(f"[ModelLoader] DEBUG: After all moves - pipeline.vae is not None: {pipeline.vae is not None}")

        # Verify VAE exists after moving to device
        if not hasattr(pipeline, 'vae') or pipeline.vae is None:
            print(f"[ModelLoader] ERROR: VAE is missing after component move")
            print(f"[ModelLoader] Pipeline components: {list(pipeline.components.keys())}")
            raise RuntimeError("VAE is missing after loading model")
        else:
            print(f"[ModelLoader] VAE verified: {type(pipeline.vae).__name__}")

        return pipeline

    @staticmethod
    def load_model(
        source_type: ModelSource,
        source: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        **kwargs
    ) -> Union[StableDiffusionPipeline, Dict[str, Any]]:
        """Universal model loading method

        Returns:
            - StableDiffusionPipeline for SD1.5/SDXL
            - Dict of components for Z-Image
        """
        # Component switching uses Anima's existing explicit companion inputs.
        # No other architecture receives these paths through this generic API.
        if (kwargs.get("text_encoder_path") is not None or kwargs.get("vae_path") is not None):
            if ModelLoader.detect_model_type(source) != "anima":
                raise ValueError("Explicit text_encoder_path/vae_path model reload is supported only for Anima")
            return ModelLoader.load_anima_from_files(
                source,
                device,
                torch.bfloat16,
                text_encoder_path=kwargs.get("text_encoder_path"),
                vae_path=kwargs.get("vae_path"),
            )
        if source_type == "safetensors":
            return ModelLoader.load_from_safetensors(source, device, torch_dtype)
        elif source_type == "diffusers":
            return ModelLoader.load_from_diffusers(source, device, torch_dtype)
        elif source_type == "huggingface":
            return ModelLoader.load_from_huggingface(
                source,
                device,
                torch_dtype,
                revision=kwargs.get("revision")
            )
        else:
            raise ValueError(f"Unknown source type: {source_type}")

    @staticmethod
    def load_anima_from_files(
        path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        text_encoder_path: Optional[str] = None,
        vae_path: Optional[str] = None,
        models_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load Anima (DiT + Qwen3 + Qwen-Image VAE) from either a single DiT
        safetensors file or a split-files directory layout.

        Returns a component dict consumed by PipelineManager.load_model().
        """
        from core.models.anima.anima_loader import (
            load_anima_components, detect_anima_split_layout, discover_anima_components,
        )

        explicit_text_encoder = text_encoder_path is not None
        explicit_vae = vae_path is not None

        # If the user pointed at a directory (split layout), pick the DiT file
        dit_path = path
        if os.path.isdir(path):
            split = detect_anima_split_layout(path)
            if not split or not split.get("dit"):
                raise FileNotFoundError(
                    f"Anima split layout expected at {path} but no DiT safetensors found."
                )
            dit_path = split["dit"]
            if text_encoder_path is None:
                text_encoder_path = split.get("text_encoder")
            if vae_path is None:
                vae_path = split.get("vae")

        # Resolve models_root: the parent of the dit_path's "models" ancestor, or settings.
        if models_root is None:
            try:
                from config.settings import settings
                models_root = getattr(settings, "models_dir", None)
            except Exception:
                models_root = None
            if models_root is None:
                # Walk up to find a `models` directory ancestor
                p = os.path.abspath(dit_path)
                for _ in range(6):
                    p = os.path.dirname(p)
                    if not p:
                        break
                    if os.path.basename(p).lower() == "models":
                        models_root = p
                        break

        components = load_anima_components(
            dit_path=dit_path,
            text_encoder_path=text_encoder_path,
            vae_path=vae_path,
            models_root=models_root,
            device="cpu",  # Loaded to CPU; pipeline.py moves to GPU per stage
            dit_dtype=torch_dtype,
            te_dtype=torch_dtype,
            vae_dtype=torch_dtype,
        )
        components["text_encoder_origin"] = (
            "selected_external" if explicit_text_encoder else "architecture_default"
        )
        if explicit_vae:
            components["vae_origin"] = "selected_external"
        elif components.get("vae_source") == "embedded (checkpoint)":
            components["vae_origin"] = "embedded_checkpoint"
        else:
            components["vae_origin"] = "architecture_default"
        return components

    @staticmethod
    def load_lens_from_path(
        path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> dict:
        """Load Microsoft/Lens from a diffusers directory, HF Hub ID, or a
        single-file full-FT DiT save (net.* weights).

        Returns a component dict consumed by PipelineManager.load_model().
        """
        if isinstance(path, str) and (
            path.endswith(".safetensors") or path.endswith(".safetensors.index.json")
        ) and os.path.isfile(path):
            from core.models.lens.lens_loader import load_lens_single_file
            return load_lens_single_file(dit_path=path, torch_dtype=torch_dtype)
        from core.models.lens.lens_loader import load_lens_components
        return load_lens_components(model_path=path, torch_dtype=torch_dtype)

    @staticmethod
    def load_ideogram4_from_path(
        path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> dict:
        """Load Ideogram 4 from a local diffusers directory or a combined
        single-file save (both transformers bundled; a ``.safetensors`` file or a
        ``.safetensors.index.json`` shard index).

        Returns a component dict consumed by PipelineManager.load_model().
        """
        if isinstance(path, str) and os.path.isfile(path) and (
            path.endswith(".safetensors") or path.endswith(".safetensors.index.json")
        ):
            from core.models.ideogram4.ideogram4_loader import load_ideogram4_single_file
            return load_ideogram4_single_file(file_path=path, torch_dtype=torch_dtype)
        from core.models.ideogram4.ideogram4_loader import load_ideogram4_components
        return load_ideogram4_components(model_path=path, torch_dtype=torch_dtype)

    @staticmethod
    def load_minit2i_from_path(
        path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> dict:
        """Load MiniT2I from a diffusers directory or a single-file safetensors.

        Returns a component dict consumed by PipelineManager.load_model().
        """
        from core.models.minit2i.minit2i_loader import load_minit2i_components
        return load_minit2i_components(model_path=path, torch_dtype=torch_dtype)

    @staticmethod
    def load_krea2_from_path(
        path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> dict:
        """Load Krea 2 from a diffusers directory or a single-file safetensors.

        Returns a component dict consumed by PipelineManager.load_model().
        """
        from core.models.krea2.krea2_loader import load_krea2_components
        return load_krea2_components(model_path=path, torch_dtype=torch_dtype)

    @staticmethod
    def load_ltx2_from_path(
        path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> dict:
        """Load LTX-2.3 from a diffusers directory (model_index.json + subfolders).

        Returns a component dict consumed by PipelineManager.load_model()
        (type == "ltx2"). bf16 by default to halve the fp32 Gemma-3 text encoder.
        """
        from core.models.ltx2.loader import load_ltx2_from_diffusers
        return load_ltx2_from_diffusers(model_path=path, torch_dtype=torch_dtype)

    @staticmethod
    def load_acestep_from_path(
        path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> dict:
        """Load ACE-Step 1.5 from its flat ComfyUI-style model tree
        (diffusion_models/ + vae/ + text_encoders/, no diffusers subfolders).

        Returns a component dict consumed by PipelineManager.load_model()
        (type == "acestep"). Phase 0+1: components load; no sampler yet.
        """
        from core.models.acestep.loader import load_acestep_from_path as _load_acestep
        return _load_acestep(model_path=path, torch_dtype=torch_dtype)

    @staticmethod
    def load_minimax_h3_from_path(
        path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> dict:
        """Load MiniMax-H3 from its flat ComfyUI-style model tree
        (diffusion_models/ + vae/ + text_encoders/ + MiniMax's config-only
        official/ tree), or from that official/ directory itself.

        Returns a component dict consumed by PipelineManager.load_model()
        (type == "minimax_h3"). ``torch_dtype`` is the block stack's compute
        dtype; the loader overrides it per component where the checkpoint's own
        mixed precision requires it (float32 patch projections / output heads /
        AdaLN curve, fp8 codes left quantized, fp16 video VAE, float32 audio
        VAE, and the text encoder left at the file's bf16 so its CPU weights
        stay memory-mapped).
        """
        from core.models.minimax_h3.loader import load_minimax_h3_from_path as _load_h3
        return _load_h3(model_path=path, torch_dtype=torch_dtype)
