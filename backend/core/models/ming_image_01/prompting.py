"""Shared native prompt/vision preparation and ordered RGBA references."""

from dataclasses import dataclass
from contextlib import nullcontext

import numpy as np
from PIL import Image
import torch


@dataclass
class MingPreparedInputs:
    model_inputs: dict
    references: list[torch.Tensor | None]
    texts: list[str]
    width: int
    height: int

    def generation_references(self):
        counts = {0 if item is None else item.shape[0] for item in self.references}
        if len(counts) != 1:
            raise ValueError("Ming generation batches require equal reference counts")
        return None if counts == {0} else torch.stack(self.references)


def prepare_inputs(processor, prompts, *, references, width, height, resolution):
    """The processor owns MLLM vision geometry; requested grids own VAE references."""
    if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) for p in prompts):
        raise ValueError("Ming prompts must be a nonempty string batch")
    if not isinstance(references, list) or len(references) != len(prompts):
        raise ValueError("Ming references require one ordered list per prompt")
    if any(not isinstance(items, list) or any(not isinstance(item, Image.Image) for item in items)
           for items in references):
        raise ValueError("Ming references must be decoded PIL images")
    if any(type(value) is not int or value <= 0 for value in (width, height, resolution)):
        raise ValueError("Ming requested dimensions and native resolution must be positive integers")
    if width % 16 or height % 16:
        raise ValueError("Ming requested pixels must align to the native 16-pixel token stride")
    texts, vision_images, rgba_references = [], [], []
    for prompt, images in zip(prompts, references):
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "HUMAN", "content": content}]
        texts.append(processor.apply_chat_template(messages, add_generation_prompt=True))
        vision, video, audio = processor.process_vision_info(messages)
        if video is not None or audio is not None:
            raise ValueError("Ming image prompting does not accept video or audio")
        if vision is not None:
            vision_images.extend(vision)
        rgba = []
        for image in images:
            resized = image.convert("RGBA").resize((width, height), Image.Resampling.BILINEAR)
            pixels = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float()
            rgba.append(pixels / 127.5 - 1.0)
        rgba_references.append(torch.cat(rgba) if rgba else None)
    processed = processor(text=texts, images=vision_images or None, return_tensors="pt",
        image_gen_highres=resolution, image_gen_aspect_ratio=width / height,
        padding=True, truncation=False)
    selected = {key: processed[key] for key in
                ("input_ids", "attention_mask", "pixel_values", "image_grid_thw") if key in processed}
    if not {"input_ids", "attention_mask"}.issubset(selected):
        raise ValueError("Ming processor did not produce native token inputs")
    if bool(vision_images) != ("pixel_values" in selected and "image_grid_thw" in selected):
        raise ValueError("Ming processor vision inputs disagree with ordered references")
    return MingPreparedInputs(selected, rgba_references, texts, width, height)


def encode_inputs(conditioner, prepared: MingPreparedInputs, *, requires_grad: bool):
    """Select a live training graph or an explicit inference-only forward."""
    device = conditioner.native.device
    dtype = next(conditioner.parameters()).dtype
    inputs = {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
              for key, value in prepared.model_inputs.items()}
    vision = getattr(conditioner, "training_vision", None)
    with vision("pixel_values" in inputs) if vision else nullcontext(), torch.set_grad_enabled(requires_grad):
        return conditioner(**inputs)
