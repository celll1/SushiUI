"""Checkpoint bundles for style training (design §11).

A bundle `{name}` is: `{name}_style_encoder.safetensors` + `{name}_metadata.json` (distributable, no
artist names), `{name}_style_lora.safetensors` in LoRA mode (adapter + pooling/projector, the resume
weights), and for resumable bundles `{name}_optimizer.pt`, `{name}_split_snapshot.npz` and
`{name}_state.json`, which is written last and marks the bundle complete.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import Siglip2VisionConfig, Siglip2VisionModel

from core.tagger.siglip2_tagger_model import _save_file_safe

from .style_model import StyleEncoder

ARTIFACT_FORMAT = "sushiui-style-encoder"
ARTIFACT_FORMAT_VERSION = 1
STATE_FORMAT_VERSION = 1
ENCODER_SUFFIX = "_style_encoder.safetensors"
METADATA_SUFFIX = "_metadata.json"
LORA_SUFFIX = "_style_lora.safetensors"
OPTIMIZER_SUFFIX = "_optimizer.pt"
SNAPSHOT_SUFFIX = "_split_snapshot.npz"
STATE_SUFFIX = "_state.json"
_STEP_BUNDLE = re.compile(r"^step_(\d+)_")
_ENCODER_LAYER = re.compile(r"^vision_encoder\.encoder\.layers\.(\d+)\.")
# Never run by StyleEncoder.forward, so absent from the distributable file.
_UNUSED_PREFIXES = ("vision_encoder.post_layernorm.", "vision_encoder.head.")


def artifact_path(output_dir: str, name: str, suffix: str) -> str:
    return os.path.join(output_dir, f"{name}{suffix}")


def write_json_atomic(path: str, payload: Any) -> None:
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _torch_save_atomic(obj: Any, path: str) -> None:
    tmp = f"{path}.tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _is_unused(key: str, depth: int) -> bool:
    if key.startswith(_UNUSED_PREFIXES):
        return True
    match = _ENCODER_LAYER.match(key)
    return match is not None and int(match.group(1)) >= depth


def _lora_prefixes(model: StyleEncoder) -> dict[str, Any]:
    return {f"vision_encoder.{name}": module for name, module in model.lora_modules.items()}


def distributable_state_dict(model: StyleEncoder) -> dict[str, torch.Tensor]:
    """Plain full-mode layout: LoRA merged into its base Linear, unused layers dropped."""
    lora = _lora_prefixes(model)
    out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for prefix, module in lora.items():
            base = module.base
            delta = (module.lora_A.float() @ module.lora_B.float()).T * module.scale
            out[f"{prefix}.weight"] = (base.weight.float() + delta).to(base.weight.dtype)
            if base.bias is not None:
                out[f"{prefix}.bias"] = base.bias.detach().clone()
    for key, value in model.state_dict().items():
        owner = key.rsplit(".", 2)[0] if key.endswith((".base.weight", ".base.bias")) else key.rsplit(".", 1)[0]
        if owner in lora or _is_unused(key, model.depth):
            continue
        out[key] = value
    return {k: v.detach().to("cpu").contiguous() for k, v in out.items()}


def lora_state_dict(model: StyleEncoder) -> dict[str, torch.Tensor]:
    """Trainable parameters of a LoRA-mode model: adapters, layer norms, token projection, pooling, projector."""
    return {name: p.detach().to("cpu").contiguous() for name, p in model.trainable_named_parameters()}


def architecture_metadata(model: StyleEncoder) -> dict:
    """What `load_style_encoder` needs; layers are stored resolved because the saved encoder is truncated
    to `depth` layers, under which a negative index would name a different layer."""
    vision_config = model.vision_encoder.config.to_dict()
    vision_config["num_hidden_layers"] = model.depth
    return {"vision_config": vision_config, "feature_layers_resolved": list(model.feature_layers),
            "token_dim": model.token_proj.out_features, "pool_heads": model.pooling.attn.num_heads,
            "projector_hidden_dim": model.projector[0].out_features, "embed_dim": model.projector[2].out_features}


def load_style_encoder(path: str, attn_implementation: str) -> StyleEncoder:
    """Rebuild a distributable encoder from its file and metadata alone (no base model download)."""
    meta_path = path[:-len(ENCODER_SUFFIX)] + METADATA_SUFFIX if path.endswith(ENCODER_SUFFIX) else None
    if meta_path is None or not os.path.isfile(meta_path):
        raise FileNotFoundError(f"no {METADATA_SUFFIX} next to {path}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("format") != ARTIFACT_FORMAT or meta.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(f"{meta_path} is not a {ARTIFACT_FORMAT} v{ARTIFACT_FORMAT_VERSION} file")
    arch = meta["architecture"]
    vision = Siglip2VisionModel(Siglip2VisionConfig(**{**arch["vision_config"],
                                                       "attn_implementation": attn_implementation}))
    model = StyleEncoder(vision, arch["feature_layers_resolved"], arch["token_dim"], arch["pool_heads"],
                         arch["projector_hidden_dim"], arch["embed_dim"], "full", None)
    state = load_file(path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not _is_unused(k, model.depth)]
    if missing or unexpected:
        raise ValueError(f"{path} does not match its metadata: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return model.eval()


def load_training_weights(model: StyleEncoder, output_dir: str, name: str) -> None:
    """Resume weights: every trainable parameter must be present, and nothing else may be."""
    expected = {k for k, _ in model.trainable_named_parameters()}
    if model.mode == "lora":
        state = load_file(artifact_path(output_dir, name, LORA_SUFFIX))
    else:
        path = artifact_path(output_dir, name, ENCODER_SUFFIX)
        with safe_open(path, framework="pt") as handle:
            state = {k: handle.get_tensor(k) for k in handle.keys() if k in expected}
    missing = expected - set(state)
    extra = set(state) - expected
    if missing or extra:
        raise ValueError(f"checkpoint {name} does not match the model: missing {sorted(missing)[:5]}, "
                         f"unexpected {sorted(extra)[:5]}")
    params = dict(model.named_parameters())
    with torch.no_grad():
        for key, value in state.items():
            params[key].copy_(value)


def save_bundle(model: StyleEncoder, output_dir: str, name: str, metadata: dict,
                resume: Optional[tuple[dict, dict, Any]] = None) -> str:
    """Write bundle `name`; `resume` = (state, optimizer_state, split snapshot) makes it resumable.
    Every file is staged first; only then is the old state removed, the staged files moved into place and
    the state written last. A failure while staging leaves the previous bundle intact and resumable."""
    os.makedirs(output_dir, exist_ok=True)
    staged: list[tuple[str, str]] = []

    def stage(final: str) -> str:
        path = final[:-len(".npz")] + ".staged.npz" if final.endswith(".npz") else final + ".staged"
        staged.append((path, final))
        return path

    path = artifact_path(output_dir, name, ENCODER_SUFFIX)
    try:
        _save_file_safe(distributable_state_dict(model), stage(path))
        if model.mode == "lora":
            _save_file_safe(lora_state_dict(model), stage(artifact_path(output_dir, name, LORA_SUFFIX)))
        write_json_atomic(stage(artifact_path(output_dir, name, METADATA_SUFFIX)), metadata)
        if resume is not None:
            state, optimizer_state, snapshot = resume
            _torch_save_atomic(optimizer_state, stage(artifact_path(output_dir, name, OPTIMIZER_SUFFIX)))
            snapshot.save(stage(artifact_path(output_dir, name, SNAPSHOT_SUFFIX)))
    except BaseException:
        for staged_path, _ in staged:
            if os.path.exists(staged_path):
                os.remove(staged_path)
        raise
    state_path = artifact_path(output_dir, name, STATE_SUFFIX)
    if os.path.exists(state_path):
        os.remove(state_path)
    for staged_path, final in staged:
        os.replace(staged_path, final)
    if resume is not None:
        write_json_atomic(state_path, {"format_version": STATE_FORMAT_VERSION, **state})
    return path


def resumable_bundle_files(output_dir: str) -> list[str]:
    """Files named like a resumable bundle (latest_* / step_N_*) with a bundle suffix."""
    if not os.path.isdir(output_dir):
        return []
    suffixes = (ENCODER_SUFFIX, METADATA_SUFFIX, LORA_SUFFIX, OPTIMIZER_SUFFIX, SNAPSHOT_SUFFIX, STATE_SUFFIX)
    return sorted(f for f in os.listdir(output_dir)
                  if (f.startswith("latest_") or _STEP_BUNDLE.match(f)) and f.endswith(suffixes))


def bundle_files(output_dir: str, name: str, mode: str) -> list[str]:
    suffixes = [ENCODER_SUFFIX, METADATA_SUFFIX, OPTIMIZER_SUFFIX, SNAPSHOT_SUFFIX, STATE_SUFFIX]
    if mode == "lora":
        suffixes.append(LORA_SUFFIX)
    return [artifact_path(output_dir, name, s) for s in suffixes]


def find_resume_checkpoint(output_dir: str) -> Optional[tuple[str, dict]]:
    """The complete resumable bundle with the highest global_step (latest or step_*)."""
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for filename in os.listdir(output_dir):
        if not filename.endswith(STATE_SUFFIX):
            continue
        name = filename[:-len(STATE_SUFFIX)]
        if name != "latest" and not re.fullmatch(r"step_\d+", name):
            continue
        with open(os.path.join(output_dir, filename), "r", encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("format_version") != STATE_FORMAT_VERSION:
            raise ValueError(f"{filename} has unsupported state format {state.get('format_version')!r}")
        if all(os.path.isfile(p) for p in bundle_files(output_dir, name, state["model"]["training_method"])):
            candidates.append((int(state["global_step"]), name == "latest", name, state))
    if not candidates:
        return None
    _, _, name, state = max(candidates, key=lambda c: (c[0], c[1]))
    return name, state


def prune_step_bundles(output_dir: str, keep_last_n: int) -> list[int]:
    """Delete whole step bundles beyond the newest `keep_last_n`; latest/best are never step bundles."""
    if keep_last_n <= 0:
        return []
    by_step: dict[int, list[str]] = {}
    for filename in os.listdir(output_dir):
        match = _STEP_BUNDLE.match(filename)
        if match:
            by_step.setdefault(int(match.group(1)), []).append(filename)
    expired = sorted(by_step)[:-keep_last_n]
    for step in expired:
        # State first: a bundle without its state is never picked for resume while being deleted.
        for filename in sorted(by_step[step], key=lambda f: not f.endswith(STATE_SUFFIX)):
            os.remove(os.path.join(output_dir, filename))
    return expired
