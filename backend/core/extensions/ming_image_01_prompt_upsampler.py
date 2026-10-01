"""Native Ming prompt protocols over SushiUI's loopback provider transports."""
from dataclasses import dataclass, asdict, replace
import hashlib
import json
import math
from pathlib import Path
import re
import threading

from config.settings import settings
from core.extensions.minimax_h3_prompt_assistant import PromptAssistCache, PromptAssistError, _normalise_url
from core.extensions.qwen_image_21_prompt_upsampler import QwenImage21PromptUpsampler, _decode_image

TEMPLATES = Path(__file__).with_name("ming_image_01_prompt_templates")


@dataclass
class MingPromptOptions:
    prompt: str
    mode: str
    engine: str
    base_url: str
    model: str
    images: list[str]
    temperature: float
    top_p: float
    top_k: int
    max_output_tokens: int
    context_length: int
    timeout_seconds: int
    force_refresh: bool
    width: int
    height: int
    layers: int


def _clean_output(raw):
    if not isinstance(raw, str):
        raise PromptAssistError("Ming prompt provider returned no text")
    text = raw.partition("</think>")[2] if "</think>" in raw else raw
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|text)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def validate_output(raw, options):
    text = _clean_output(raw)
    if options.mode == "layer":
        header = re.match(r"Decompose this image into (\d+) layers with the following specifications:\s*Number of layers: (\d+)\s*", text)
        if not header or any(int(value) != options.layers for value in header.groups()):
            raise PromptAssistError("Ming rewritten layer count does not match the requested count")
        parts = re.split(r"(?m)^Layer (\d+):\s*", text[header.end():])
        if parts[0].strip() or len(parts) != 1 + options.layers * 2:
            raise PromptAssistError("Ming layer rewrite must list every layer once in front-to-back order")
        for index in range(options.layers):
            if int(parts[1+2*index]) != index+1 or not parts[2+2*index].strip():
                raise PromptAssistError("Ming layer rewrite has missing, empty or reordered layers")
        return text
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise PromptAssistError("Ming Design rewrite must return canvas/layers JSON") from exc
    if not isinstance(value, dict) or set(value) != {"canvas_settings", "layers"}:
        raise PromptAssistError("Ming Design rewrite requires exactly canvas_settings and layers")
    canvas = value["canvas_settings"]
    if not isinstance(canvas, dict) or set(canvas) != {"aspect_ratio", "ambient_lighting", "image_style"}:
        raise PromptAssistError("Ming Design canvas settings do not match the native protocol")
    if any(not isinstance(item, str) or not item.strip() for item in canvas.values()):
        raise PromptAssistError("Ming Design canvas settings must be nonempty strings")
    ratio = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?):([0-9]+(?:\.[0-9]+)?)", canvas["aspect_ratio"])
    if not ratio or float(ratio[2]) <= 0 or not math.isclose(float(ratio[1])/float(ratio[2]), options.width/options.height, rel_tol=1e-6):
        raise PromptAssistError("Ming Design rewrite changed the requested canvas aspect ratio")
    layers = value["layers"]
    if not isinstance(layers, list) or not layers:
        raise PromptAssistError("Ming Design rewrite has no visible layers")
    for layer in layers:
        if not isinstance(layer, dict) or set(layer) != {"description", "coordinates", "hierarchy_and_relation", "color_specs"}:
            raise PromptAssistError("Ming Design layer fields do not match the native protocol")
        if any(not isinstance(layer[key], str) or not layer[key].strip() for key in ("description", "coordinates", "hierarchy_and_relation")):
            raise PromptAssistError("Ming Design layer descriptions/coordinates must be nonempty strings")
        coords = re.fullmatch(r"cx: ([0-9]+(?:\.[0-9]+)?), cy: ([0-9]+(?:\.[0-9]+)?), w: ([0-9]+(?:\.[0-9]+)?), h: ([0-9]+(?:\.[0-9]+)?)", layer["coordinates"])
        if not coords:
            raise PromptAssistError("Ming Design coordinates must use the native normalized bbox string")
        x,y,w,h = map(float, coords.groups())
        if not (w>0 and h>0 and x-w/2>=-1e-6 and y-h/2>=-1e-6 and x+w/2<=1+1e-6 and y+h/2<=1+1e-6):
            raise PromptAssistError("Ming Design layer bbox lies outside the canvas")
        colors = layer["color_specs"]
        if not isinstance(colors, list) or not colors or any(not isinstance(c,str) or not re.fullmatch(r"#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?",c) for c in colors):
            raise PromptAssistError("Ming Design color_specs must contain hex colors")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class MingImage01PromptUpsampler:
    _lm_studio = QwenImage21PromptUpsampler._lm_studio
    _ollama = QwenImage21PromptUpsampler._ollama
    _openai_messages = staticmethod(QwenImage21PromptUpsampler._openai_messages)

    def __init__(self, max_cache_entries, cache_path=None):
        self.cache = PromptAssistCache(cache_path or Path(settings.cache_dir)/"ming_image_01_prompt_upsample.sqlite3", max_cache_entries)
        self._lock = threading.RLock()

    def transform(self, options, *, profile, api_key=""):
        if options.engine not in {"lm_studio", "ollama"} or not options.model.strip():
            raise PromptAssistError("Select a local Ming prompt provider and model")
        if options.mode not in {"t2i", "i2i", "layer"}:
            raise PromptAssistError("Unsupported Ming prompt rewrite mode")
        expected = "layer_decompose" if options.mode == "layer" else "generation_edit"
        if profile != expected:
            raise PromptAssistError("Ming prompt rewrite task does not match the loaded checkpoint profile")
        if not options.prompt.strip():
            raise PromptAssistError("Ming rough prompt/layer plan cannot be empty")
        if any(type(v) is not int or v < 16 or v % 16 for v in (options.width,options.height)) or type(options.layers) is not int or options.layers < 1:
            raise PromptAssistError("Ming rewrite requires aligned dimensions and a positive layer count")
        if (options.mode == "t2i" and options.images) or (options.mode == "i2i" and not options.images) or (options.mode == "layer" and len(options.images) != 1):
            raise PromptAssistError("Ming rewrite references do not match the task (Layer requires exactly one flattened image)")
        base_url = _normalise_url(options.base_url)
        for image in options.images:
            _decode_image(image)
        template = (TEMPLATES/('layer.txt' if options.mode == 'layer' else 'design.txt')).read_text(encoding='utf-8')
        if options.mode == "layer":
            system = template.replace("{spec}", "(See the user's rough plan in the next message.)")
            system += f"\nThe requested layer count is exactly {options.layers}; preserve it."
        else:
            system = template + f"\nUse canvas_settings.aspect_ratio = '{options.width}:{options.height}'."
            if options.mode == "i2i":
                system += " Ground the requested edit in the supplied ordered images and preserve every unrequested identity and visible string."
        material = {**asdict(options), "profile":profile,"template_sha256":hashlib.sha256(system.encode()).hexdigest(),"base_url":base_url}
        material.pop("force_refresh")
        material["images"] = [hashlib.sha256(image.encode()).hexdigest() for image in options.images]
        key = hashlib.sha256(json.dumps(material,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        with self._lock:
            cached = None if options.force_refresh else self.cache.get(key)
            if cached is not None:
                return {**cached,"cached":True,"cache_key":key}
            transport = replace(options, mode="i2i" if options.images else "t2i")
            transport.json_output = options.mode != "layer"
            if options.engine == "lm_studio":
                raw,warnings = self._lm_studio(transport,base_url,system,api_key)
            else:
                raw,warnings = self._ollama(transport,base_url,system)
            prompt = validate_output(raw,options)
            response = {"prompt":prompt,"rewritten_prompt":prompt,"wh_ratio":f"{options.width}:{options.height}","ratio_follow":"","warnings":warnings,"engine":options.engine,"model":options.model,"cached":False}
            self.cache.put(key,response)
            return {**response,"cache_key":key}
