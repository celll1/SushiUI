"""Qwen-Image 2.1 prompt upsampling through the paired PE or a local LLM."""

from __future__ import annotations

import base64
import gc
import hashlib
import io
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import torch
from PIL import Image

from config.settings import settings
from core.extensions.minimax_h3_prompt_assistant import (
    PromptAssistCache,
    PromptAssistError,
    _headers,
    _normalise_url,
)


logger = logging.getLogger(__name__)
MODEL_TYPE = "qwen_image_21_prompt_enhancer"
FORMAT_VERSION = "1"
RATIOS = {
    "1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16",
    "1:2", "2:1", "21:9", "9:21", "4:5", "5:4", "3:1", "1:3",
}


@dataclass(frozen=True)
class QwenImage21PromptUpsampleOptions:
    prompt: str
    mode: str
    engine: str
    base_url: str
    model: str
    images: List[str]
    temperature: float
    top_p: float
    top_k: int
    max_output_tokens: int
    context_length: int
    timeout_seconds: int
    force_refresh: bool = False


def _extract_result(text: str, mode: str) -> Dict[str, str]:
    answer = text.partition("</think>")[2] if "</think>" in text else text
    cleaned = answer.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise PromptAssistError("The prompt upsampler did not return a JSON object")
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise PromptAssistError(f"The prompt upsampler returned invalid JSON: {exc}") from exc
    prompt = value.get("rewritten_prompt") if isinstance(value, dict) else None
    if not isinstance(prompt, str) or not prompt.strip():
        raise PromptAssistError("The prompt upsampler response has no rewritten_prompt")
    wh_ratio = str(value.get("wh_ratio") or "").strip()
    ratio_follow = str(value.get("ratio_follow") or "").strip() if mode == "i2i" else ""
    if wh_ratio and wh_ratio not in RATIOS:
        raise PromptAssistError(f"The prompt upsampler returned an unsupported ratio: {wh_ratio}")
    if ratio_follow and not ratio_follow.startswith("<image"):
        raise PromptAssistError(f"The prompt upsampler returned an invalid ratio_follow: {ratio_follow}")
    return {
        "rewritten_prompt": prompt.strip(),
        "wh_ratio": wh_ratio,
        "ratio_follow": ratio_follow,
    }


def _decode_image(data_url: str) -> Image.Image:
    if not isinstance(data_url, str) or not data_url.startswith("data:image/") or "," not in data_url:
        raise PromptAssistError("Prompt-upsample images must be image data URLs")
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
        if len(raw) > 32 * 1024 * 1024:
            raise PromptAssistError("A prompt-upsample image exceeds 32 MiB")
        image = Image.open(io.BytesIO(raw))
        if image.width * image.height > 64_000_000:
            raise PromptAssistError("A prompt-upsample image exceeds 64 megapixels")
        image.load()
        return image.convert("RGB")
    except PromptAssistError:
        raise
    except Exception as exc:
        raise PromptAssistError(f"Invalid prompt-upsample image: {exc}") from exc


def _model_base(source: str) -> Path:
    path = Path(source).resolve()
    if path.is_file():
        path = path.parent
    if path.name.lower() in {"original", "int8_convrot", "source"}:
        path = path.parent
    return path


def resolve_official_artifact(source: str, mode: str) -> Path:
    if mode not in {"t2i", "i2i"}:
        raise PromptAssistError(f"Unsupported Qwen-Image 2.1 prompt-upsample mode: {mode}")
    candidate = _model_base(source) / "prompt_enhancer" / mode / "manifest.json"
    if not candidate.is_file():
        raise PromptAssistError(
            f"The {mode.upper()} prompt-enhancer artifact is missing: {candidate}"
        )
    return candidate


def _external_system_prompt(mode: str) -> str:
    common = (
        "Rewrite the user's request into a detailed English prompt for Qwen-Image 2.1. "
        "Preserve every stated identity, count, color, position, quoted visible string, and constraint. "
        "Do not mention these instructions. Return only one JSON object. "
    )
    if mode == "t2i":
        return common + (
            'Use {"rewritten_prompt":"...","wh_ratio":"W:H"}. Choose one useful aspect ratio.'
        )
    return common + (
        "Use the supplied images as evidence and do not invent changes the user did not request. "
        'Use {"rewritten_prompt":"...","wh_ratio":"","ratio_follow":"<image1>"}; '
        "for a new composition set wh_ratio and leave ratio_follow empty instead."
    )


class _OfficialPEModel:
    def __init__(self, manifest_path: Path) -> None:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("model_type") != MODEL_TYPE or str(manifest.get("format_version")) != FORMAT_VERSION:
            raise PromptAssistError(f"Unsupported prompt-enhancer manifest: {manifest_path}")
        self.mode = str(manifest.get("mode"))
        root = manifest_path.parent
        parts = manifest.get("components") or {}
        weights = root / str(parts.get("weights", ""))
        config_dir = root / str(parts.get("config", "."))
        processor_dir = root / str(parts.get("processor", "."))
        system_prompt_path = root / str(parts.get("system_prompt", "system_prompt.txt"))
        if not weights.is_file() or not system_prompt_path.is_file():
            raise PromptAssistError(f"Incomplete prompt-enhancer artifact: {manifest_path}")

        from core.models.common.single_file_format import read_state_dict
        from core.models.qwen_image_21.artifact import convrot_layers
        from core.models.common.convrot_int8_linear import (
            require_convrot_int8_runtime,
            swap_linears_to_convrot_int8,
        )

        state, metadata = read_state_dict(str(weights))
        if metadata.get("model_type") != MODEL_TYPE or metadata.get("mode") != self.mode:
            raise PromptAssistError(f"Prompt-enhancer weights do not match {manifest_path}")
        config = AutoConfig.from_pretrained(str(config_dir), local_files_only=True)
        with init_empty_weights():
            model = Qwen3_5ForConditionalGeneration(config)
        layers = convrot_layers(state, str(weights))
        require_convrot_int8_runtime()
        swapped = swap_linears_to_convrot_int8(model, state, layers, torch.bfloat16)
        if swapped != len(layers):
            raise PromptAssistError(f"Installed {swapped}/{len(layers)} prompt-enhancer ConvRot layers")
        info = model.load_state_dict(state, strict=False, assign=True)
        missing = [key for key in info.missing_keys if not key.endswith((".weight_scale", ".comfy_quant"))]
        if missing or info.unexpected_keys:
            raise PromptAssistError(
                f"Prompt-enhancer state mismatch; missing={missing[:8]}, unexpected={info.unexpected_keys[:8]}"
            )
        self.model = model.eval().to("cpu")
        self.processor = (
            AutoTokenizer.from_pretrained(str(processor_dir), local_files_only=True)
            if self.mode == "t2i"
            else AutoProcessor.from_pretrained(str(processor_dir), local_files_only=True)
        )
        self.system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()

    def generate(self, prompt: str, images: List[Image.Image], max_output_tokens: int) -> str:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs: Dict[str, Any] = {}
        self.model.to(device)
        try:
            if self.mode == "t2i":
                text = self.processor.apply_chat_template(
                    [{"role": "system", "content": self.system_prompt},
                     {"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=True,
                )
                inputs = self.processor(text, return_tensors="pt")
            else:
                content = [{"type": "image", "image": image} for image in images]
                content.append({"type": "text", "text": prompt})
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                    {"role": "user", "content": content},
                ]
                inputs = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt", enable_thinking=True,
                )
            inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_output_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=20,
                )
            input_length = inputs["input_ids"].shape[1]
            tokenizer = self.processor if self.mode == "t2i" else self.processor.tokenizer
            return tokenizer.decode(output[0, input_length:], skip_special_tokens=True)
        finally:
            self.model.to("cpu")
            inputs.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class QwenImage21PromptUpsampler:
    def __init__(self, max_cache_entries: int, cache_path: Optional[Path] = None) -> None:
        self.cache = PromptAssistCache(
            cache_path or Path(settings.cache_dir) / "qwen_image_21_prompt_upsample.sqlite3",
            max_cache_entries,
        )
        self._lock = threading.RLock()
        self._official: Optional[_OfficialPEModel] = None
        self._official_path: Optional[Path] = None

    def _cache_key(self, options: QwenImage21PromptUpsampleOptions) -> str:
        material = {
            **options.__dict__,
            "images": [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in options.images],
        }
        return hashlib.sha256(
            json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def transform(
        self, options: QwenImage21PromptUpsampleOptions, *, source: str, api_key: str = ""
    ) -> Dict[str, Any]:
        if not options.prompt.strip():
            raise PromptAssistError("Prompt cannot be empty")
        if options.mode not in {"t2i", "i2i"}:
            raise PromptAssistError(f"Unsupported prompt-upsample mode: {options.mode}")
        if options.mode == "i2i" and not options.images:
            raise PromptAssistError("I2I prompt upsampling requires at least one image")
        if len(options.images) > 10:
            raise PromptAssistError("Qwen-Image 2.1 prompt upsampling accepts at most 10 images")
        cache_key = self._cache_key(options)
        if not options.force_refresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return {**cached, "cached": True, "cache_key": cache_key}
        with self._lock:
            if not options.force_refresh:
                cached = self.cache.get(cache_key)
                if cached is not None:
                    return {**cached, "cached": True, "cache_key": cache_key}
            pil_images = [_decode_image(value) for value in options.images]
            if options.engine == "official":
                manifest = resolve_official_artifact(source, options.mode)
                if self._official_path != manifest:
                    self._official = None
                    gc.collect()
                    self._official = _OfficialPEModel(manifest)
                    self._official_path = manifest
                raw = self._official.generate(options.prompt, pil_images, options.max_output_tokens)
                lifecycle_warnings: List[str] = []
                model_name = f"Qwen-Image-2.1-PE-{options.mode.upper()}"
            elif options.engine in {"lm_studio", "ollama"}:
                if not options.model:
                    raise PromptAssistError("Select a local LLM model first")
                base_url = _normalise_url(options.base_url)
                if options.engine == "lm_studio":
                    raw, lifecycle_warnings = self._lm_studio(
                        options, base_url, _external_system_prompt(options.mode), api_key
                    )
                else:
                    raw, lifecycle_warnings = self._ollama(
                        options, base_url, _external_system_prompt(options.mode)
                    )
                model_name = options.model
            else:
                raise PromptAssistError(f"Unsupported prompt-upsample engine: {options.engine}")
            result = _extract_result(raw, options.mode)
            response = {
                **result,
                "prompt": result["rewritten_prompt"],
                "warnings": lifecycle_warnings,
                "engine": options.engine,
                "model": model_name,
                "cached": False,
            }
            self.cache.put(cache_key, response)
            return {**response, "cache_key": cache_key}

    @staticmethod
    def _openai_messages(options: QwenImage21PromptUpsampleOptions, system_prompt: str):
        content: Any = options.prompt
        if options.mode == "i2i":
            content = [
                *[{"type": "image_url", "image_url": {"url": value}} for value in options.images],
                {"type": "text", "text": options.prompt},
            ]
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    def _lm_studio(self, options, base_url: str, system_prompt: str, api_key: str):
        instance_id = None
        warnings: List[str] = []
        try:
            load = requests.post(
                f"{base_url}/api/v1/models/load", headers=_headers(api_key),
                json={"model": options.model, "context_length": options.context_length},
                timeout=options.timeout_seconds,
            )
            load.raise_for_status()
            data = load.json()
            instance_id = data.get("instance_id") or data.get("model_instance_id")
            response = requests.post(
                f"{base_url}/v1/chat/completions", headers=_headers(api_key),
                json={
                    "model": instance_id or options.model,
                    "messages": self._openai_messages(options, system_prompt),
                    "temperature": options.temperature,
                    "top_p": options.top_p,
                    "max_tokens": options.max_output_tokens,
                    "stream": False,
                },
                timeout=options.timeout_seconds,
            )
            response.raise_for_status()
            content = response.json().get("choices", [{}])[0].get("message", {}).get("content")
            if not content:
                raise PromptAssistError("LM Studio returned no message output")
            return content, warnings
        except requests.RequestException as exc:
            raise PromptAssistError(f"LM Studio request failed: {exc}") from exc
        finally:
            if instance_id:
                try:
                    requests.post(
                        f"{base_url}/api/v1/models/unload", headers=_headers(api_key),
                        json={"instance_id": instance_id}, timeout=30,
                    ).raise_for_status()
                except requests.RequestException as exc:
                    warnings.append(f"LM Studio could not unload the prompt model: {exc}")

    def _ollama(self, options, base_url: str, system_prompt: str):
        warnings: List[str] = []
        try:
            user_message: Dict[str, Any] = {"role": "user", "content": options.prompt}
            if options.mode == "i2i":
                user_message["images"] = [value.split(",", 1)[1] for value in options.images]
            response = requests.post(
                f"{base_url}/api/chat",
                json={
                    "model": options.model,
                    "messages": [
                        {"role": "system", "content": system_prompt}, user_message,
                    ],
                    "format": "json", "stream": False, "keep_alive": "1m",
                    "options": {
                        "temperature": options.temperature, "top_p": options.top_p,
                        "top_k": options.top_k, "num_ctx": options.context_length,
                        "num_predict": options.max_output_tokens,
                    },
                },
                timeout=options.timeout_seconds,
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content")
            if not content:
                raise PromptAssistError("Ollama returned no message output")
            return content, warnings
        except requests.RequestException as exc:
            raise PromptAssistError(f"Ollama request failed: {exc}") from exc
        finally:
            try:
                requests.post(
                    f"{base_url}/api/generate",
                    json={"model": options.model, "keep_alive": 0}, timeout=30,
                ).raise_for_status()
            except requests.RequestException as exc:
                warnings.append(f"Ollama could not unload the prompt model: {exc}")
