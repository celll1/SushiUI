"""Style encoder training loop (design §6, §8.1, §9, §10, §11)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass, fields
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from core.tagger.siglip2_tagger_model import _is_hf_repo_or_url, _load_vision_encoder
from core.tagger.tagger_offload import TaggerTrainerHandle
# core.tagger.tagger_trainer is imported where used: it imports bitsandbytes, which creates a CUDA context.

from .style_artifacts import (
    ARTIFACT_FORMAT, ARTIFACT_FORMAT_VERSION, OPTIMIZER_SUFFIX, SNAPSHOT_SUFFIX, architecture_metadata,
    artifact_path, find_resume_checkpoint, load_training_weights, prune_step_bundles, resumable_bundle_files,
    save_bundle,
)
from .style_dataset import VAL_ARTIST, VAL_IMAGE, SplitParams, StyleIndex, build_style_index, confound_scores, gather_csr
from .style_eval import Csr, RetrievalSummary, retrieve
from .style_ingest import IngestStats, StyleDatasetConfig, StyleIngest, build_label_resolver, parse_dataset_configs
from .style_labels import LabelStats, StyleLabelResolver
from .style_loader import EvalBatchKeys, StyleImageDataset, TrainBatchKeys, make_loader
from .style_loss import positive_weights, weighted_supcon_loss
from .style_model import LoRASpec, StyleEncoder, build_style_encoder, lora_dropout_entropy
from .style_resume import ResumeKey, SplitSnapshot, plan_resume, start_snapshot
from .style_sampler import StyleBatchSampler
from .style_views import ViewParams, artist_mask_ratios

EVAL_RECALL_KS = (1, 5)  # design §10


class StyleConfigError(ValueError):
    pass


def _int(lo: int = 0) -> Callable[[Any], bool]:
    return lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= lo


def _num(lo: float, hi: float = math.inf, lo_open: bool = False, hi_open: bool = False) -> Callable[[Any], bool]:
    def check(v: Any) -> bool:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return False
        return (v > lo if lo_open else v >= lo) and (v < hi if hi_open else v <= hi)
    return check


def _choice(*options: str) -> Callable[[Any], bool]:
    return lambda v: v in options


def _is_bool(v: Any) -> bool:
    return isinstance(v, bool)


def _is_str(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _str_list(v: Any) -> bool:
    return isinstance(v, (list, tuple)) and all(isinstance(t, str) and t.strip() for t in v)


def _int_list(v: Any) -> bool:
    return isinstance(v, (list, tuple)) and bool(v) and all(isinstance(t, int) and not isinstance(t, bool) for t in v)


def _optional_int(v: Any) -> bool:
    return v is None or _int(0)(v)


# Every key is required: defaults live in param_defaults (STYLE_TRAINING_DEFAULTS), not here.
_CHECKS: dict[str, Callable[[Any], bool]] = {
    "dataset_configs": lambda v: isinstance(v, (list, tuple)) and bool(v),
    "vision_encoder_path": _is_str,
    "vision_encoder_repo": _is_str,
    "training_method": _choice("full", "lora"),
    "lora_rank": _int(1),
    "lora_alpha": _num(0.0, lo_open=True),
    "lora_dropout": _num(0.0, 1.0, hi_open=True),
    "attn_implementation": _is_str,
    "mixed_precision": _choice("bf16", "fp32"),
    "gradient_checkpointing": _is_bool,
    "gpu_index": _optional_int,
    "optimizer": _choice("adamw", "adamw8bit", "lion8bit"),
    "learning_rate": _num(0.0, lo_open=True),
    "head_lr_multiplier": _num(0.0, lo_open=True),
    "weight_decay": _num(0.0),
    "warmup_steps": _int(0),
    "min_lr_ratio": _num(0.0, 1.0),
    "max_grad_norm": _num(0.0, lo_open=True),
    "epochs": _int(1),
    "num_workers": _int(0),
    "save_every_n_steps": _int(0),
    "keep_last_n_checkpoints": _int(0),
    "validate_every": _int(1),
    "log_every_n_steps": _int(1),
    "style_seed": _int(0),
    "style_split_seed": _int(0),
    "style_use_tag_aliases": _is_bool,
    "style_use_gelbooru_categories": _is_bool,
    "style_signature_tags": _str_list,
    "style_non_copyright_tags": _str_list,
    "style_val_artist_ratio": _num(0.0, 1.0, hi_open=True),
    "style_val_image_ratio": _num(0.0, 1.0, hi_open=True),
    "style_min_images_per_artist": _int(1),
    "style_artists_per_batch": _int(1),
    "style_images_per_artist": _int(1),
    "style_same_copyright_batch_ratio": _num(0.0, 1.0),
    "style_artist_sampling_power": _num(-math.inf),
    "style_mask_ratio_min": _num(0.0, 1.0, hi_open=True),
    "style_mask_ratio_max": _num(0.0, 1.0, hi_open=True),
    "style_max_num_patches": _int(1),
    "style_temperature": _num(0.0, lo_open=True),
    "style_view_positive_weight": _num(0.0, lo_open=True),
    "style_positive_weight_min": _num(0.0, 1.0),
    "style_feature_layers": _int_list,
    "style_token_dim": _int(1),
    "style_pool_heads": _int(1),
    "style_projector_hidden_dim": _int(1),
    "style_embed_dim": _int(1),
    "style_val_max_items": _int(0),
    "style_val_batch_size": _int(1),
    "style_eval_chunk_size": _int(1),
}
# A resumed run must rebuild the same parameters and optimizer state layout.
_IDENTITY_KEYS = ("training_method", "vision_encoder_path", "vision_encoder_repo", "style_feature_layers",
                  "style_token_dim", "style_pool_heads", "style_projector_hidden_dim", "style_embed_dim", "optimizer")
_LORA_IDENTITY_KEYS = ("lora_rank", "lora_alpha")
# Fixed at run start (optimizer param groups, scheduler lambda, total_steps); a resume keeps the saved values.
_SCHEDULE_KEYS = ("learning_rate", "head_lr_multiplier", "weight_decay", "warmup_steps", "min_lr_ratio", "epochs")


@dataclass(frozen=True)
class StyleTrainingConfig:
    dataset_configs: tuple[StyleDatasetConfig, ...]
    vision_encoder_path: str
    vision_encoder_repo: str
    training_method: str
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    attn_implementation: str
    mixed_precision: str
    gradient_checkpointing: bool
    gpu_index: Optional[int]
    optimizer: str
    learning_rate: float
    head_lr_multiplier: float
    weight_decay: float
    warmup_steps: int
    min_lr_ratio: float
    max_grad_norm: float
    epochs: int
    num_workers: int
    save_every_n_steps: int
    keep_last_n_checkpoints: int
    validate_every: int
    log_every_n_steps: int
    style_seed: int
    style_split_seed: int
    style_use_tag_aliases: bool
    style_use_gelbooru_categories: bool
    style_signature_tags: tuple[str, ...]
    style_non_copyright_tags: tuple[str, ...]
    style_val_artist_ratio: float
    style_val_image_ratio: float
    style_min_images_per_artist: int
    style_artists_per_batch: int
    style_images_per_artist: int
    style_same_copyright_batch_ratio: float
    style_artist_sampling_power: float
    style_mask_ratio_min: float
    style_mask_ratio_max: float
    style_max_num_patches: int
    style_temperature: float
    style_view_positive_weight: float
    style_positive_weight_min: float
    style_feature_layers: tuple[int, ...]
    style_token_dim: int
    style_pool_heads: int
    style_projector_hidden_dim: int
    style_embed_dim: int
    style_val_max_items: int
    style_val_batch_size: int
    style_eval_chunk_size: int

    @classmethod
    def from_dict(cls, config: dict) -> "StyleTrainingConfig":
        names = [f.name for f in fields(cls)]
        assert set(names) == set(_CHECKS), "config fields and checks disagree"
        missing = [name for name in names if name not in config]
        if missing:
            raise StyleConfigError(f"style training config is missing required keys: {missing}")
        values = {name: config[name] for name in names}
        bad = [f"{name}={values[name]!r}" for name in names if not _CHECKS[name](values[name])]
        if bad:
            raise StyleConfigError(f"invalid style training config values: {', '.join(bad)}")
        values["dataset_configs"] = parse_dataset_configs(values["dataset_configs"])
        for name in ("style_signature_tags", "style_non_copyright_tags", "style_feature_layers"):
            values[name] = tuple(values[name])
        if values["style_mask_ratio_min"] > values["style_mask_ratio_max"]:
            raise StyleConfigError("style_mask_ratio_min must not exceed style_mask_ratio_max")
        return cls(**values)

    def schedule(self) -> dict:
        return {k: getattr(self, k) for k in _SCHEDULE_KEYS}

    def identity(self) -> dict:
        keys = _IDENTITY_KEYS + (_LORA_IDENTITY_KEYS if self.training_method == "lora" else ())
        return {k: list(v) if isinstance(v, tuple) else v for k, v in ((k, getattr(self, k)) for k in keys)}


@dataclass(frozen=True)
class StyleEnvironment:
    """External dependencies, replaceable in tests (no DB, taglist or model download)."""
    open_datasets_session: Callable[[], Any]
    build_label_resolver: Callable[[StyleTrainingConfig], StyleLabelResolver]
    load_image_processor: Callable[[str], Any]
    load_vision_encoder: Callable[..., nn.Module]
    resolve_device: Callable[[Optional[int]], torch.device]


def _production_label_resolver(cfg: StyleTrainingConfig) -> StyleLabelResolver:
    from config import settings
    return build_label_resolver(settings.root_dir, cfg.style_use_tag_aliases, cfg.style_use_gelbooru_categories,
                                cfg.style_signature_tags, cfg.style_non_copyright_tags)


def _production_image_processor(repo_id: str):
    from transformers import AutoImageProcessor
    try:
        return AutoImageProcessor.from_pretrained(repo_id, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached yet
        return AutoImageProcessor.from_pretrained(repo_id)


def default_environment() -> StyleEnvironment:
    from core.tagger.tagger_trainer import resolve_training_device
    from database import DatasetsSessionLocal
    return StyleEnvironment(DatasetsSessionLocal, _production_label_resolver, _production_image_processor,
                            _load_vision_encoder, resolve_training_device)


class WarmupCosine:
    """LR factor: linear warmup, then cosine to `min_ratio` at `total_steps` (fixed at run start, §6)."""

    def __init__(self, warmup_steps: int, total_steps: int, min_ratio: float) -> None:
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_ratio = float(min_ratio)

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        span = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, (step - self.warmup_steps) / span)
        return self.min_ratio + (1.0 - self.min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


# Optimizer state bytes per trainable parameter before the first step allocates it.
_OPTIMIZER_STATE_BYTES = {"adamw": 8, "adamw8bit": 2, "lion8bit": 1}


class StyleTrainerHandle(TaggerTrainerHandle):
    """The tagger's pause/offload protocol; no training-model inference (`can_predict` stays False)."""

    def __init__(self, *args, optimizer_name: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.optimizer_name = optimizer_name

    def trainer_label(self) -> str:
        return f"style:{self.run_id[:8]}"

    def estimate_state_bytes(self) -> int:
        """Parameters and buffers plus the optimizer state actually allocated, or its per-type size."""
        if self._model is None:
            return 0
        model_bytes = sum(t.numel() * t.element_size() for t in (*self._model.parameters(), *self._model.buffers()))
        state = self._optimizer.state if self._optimizer is not None else {}
        state_bytes = sum(v.numel() * v.element_size() for per in state.values() for v in per.values()
                          if torch.is_tensor(v))
        if not state_bytes:
            state_bytes = _OPTIMIZER_STATE_BYTES[self.optimizer_name] * sum(
                p.numel() for p in self._model.parameters() if p.requires_grad)
        return model_bytes + state_bytes


@dataclass(frozen=True, eq=False)
class StyleData:
    index: StyleIndex
    sampler: StyleBatchSampler
    views: ViewParams
    dataset: StyleImageDataset
    ingest_stats: IngestStats
    label_stats: LabelStats
    skipped_examples: tuple[str, ...]


def prepare_style_data(cfg: StyleTrainingConfig, env: StyleEnvironment, processor, pinned_keys: np.ndarray,
                       progress: Optional[Callable[[int, int, str], None]] = None) -> StyleData:
    session = env.open_datasets_session()
    try:
        ingest = StyleIngest(session, cfg.dataset_configs, env.build_label_resolver(cfg), progress)
        split = SplitParams.create(cfg.style_val_artist_ratio, cfg.style_val_image_ratio,
                                   cfg.style_min_images_per_artist, cfg.style_split_seed)
        index = build_style_index(ingest.records(), split, pinned_keys)
    finally:
        session.close()
    sampler = StyleBatchSampler(index, cfg.style_artists_per_batch, cfg.style_images_per_artist,
                                cfg.style_same_copyright_batch_ratio, cfg.style_artist_sampling_power, cfg.style_seed)
    views = ViewParams.create(cfg.style_mask_ratio_min, cfg.style_mask_ratio_max, cfg.style_max_num_patches,
                              cfg.style_seed)
    ratios = artist_mask_ratios(confound_scores(index), cfg.style_mask_ratio_min, cfg.style_mask_ratio_max)
    dataset = StyleImageDataset(ingest.paths.take(index.item_ids), ratios[index.item_artist], processor,
                                cfg.style_max_num_patches)
    return StyleData(index, sampler, views, dataset, ingest.stats, ingest.resolver.stats,
                     tuple(ingest.skipped_examples))


def eval_rows(index: StyleIndex, split_code: int, max_items: int, seed: int) -> np.ndarray:
    """Rows of one held-out split; VAL_IMAGE rows only of eligible train artists. Past `max_items`, the
    set is the longest prefix of artists, in a seeded name-hash order, whose images all fit, but never less
    than the first artist (which alone may exceed the cap). Each kept artist keeps all its images, and an
    artist enters or leaves only when its rank prefix changes."""
    rows = np.flatnonzero(index.item_split == split_code)
    if split_code == VAL_IMAGE:
        rows = rows[index.artist_eligible[index.item_artist[rows]]]
    if max_items == 0 or rows.size <= max_items:
        return rows
    counts = np.bincount(index.item_artist[rows], minlength=len(index.artist_names))
    artists = np.flatnonzero(counts)

    def rank(a: int) -> bytes:
        return hashlib.blake2b(f"{seed}\0{index.artist_names[a]}".encode("utf-8"), digest_size=8,
                               person=b"style-evalset").digest()

    keep = np.zeros(len(index.artist_names), dtype=bool)
    total = 0
    for a in sorted(artists.tolist(), key=rank):
        if total and total + counts[a] > max_items:
            break
        keep[a] = True
        total += int(counts[a])
    return rows[keep[index.item_artist[rows]]]


def _row_csr(indptr: np.ndarray, values: np.ndarray, rows: np.ndarray) -> Csr:
    _, entries = gather_csr(indptr, values, rows)
    out = np.zeros(rows.size + 1, dtype=np.int64)
    np.cumsum(indptr[rows + 1] - indptr[rows], out=out[1:])
    return Csr(out, entries.astype(np.int64))


def _summary(summary: RetrievalSummary) -> dict:
    return {"queries": summary.queries, "used": summary.used, "excluded_no_positive": summary.excluded_no_positive,
            "excluded_no_negative": summary.excluded_no_negative, "map_at_r": summary.map_at_r,
            **{f"recall_at_{k}": v for k, v in summary.recall.items()}}


def _public_source(source: str) -> str:
    """HF repo ids as given; a local path only by file name, so no local path lands in a distributable file."""
    is_hf, repo = _is_hf_repo_or_url(source)
    return repo if is_hf else os.path.basename(source.strip().strip('"').strip("'"))


class StyleTrainer:
    def __init__(self, run_id: str, config: dict, output_dir: str, progress_callback: Optional[Callable] = None,
                 environment: Optional[StyleEnvironment] = None) -> None:
        self.run_id = run_id
        self.cfg = StyleTrainingConfig.from_dict(config)
        self.output_dir = output_dir
        self.callback = progress_callback
        self.env = environment or default_environment()
        self._stop_requested = False
        self._stop_event = threading.Event()
        self._optimizer_steps_completed = 0
        self._batches_entered = 0
        self._coordinator_handle = StyleTrainerHandle(
            run_id=run_id, output_dir=output_dir, pause_event=threading.Event(), resumed_event=threading.Event(),
            restored_event=threading.Event(), device_index=self.cfg.gpu_index, optimizer_name=self.cfg.optimizer)
        os.makedirs(output_dir, exist_ok=True)

    def stop(self) -> None:
        self._stop_requested = True
        self._stop_event.set()

    def _emit(self, event_type: str, data: dict) -> None:
        if self.callback:
            try:
                self.callback(self.run_id, event_type, data)
            except Exception as e:  # noqa: BLE001
                print(f"[StyleTrainer] Callback error: {e}")

    # ------------------------------------------------------------------ setup

    def _load_processor(self):
        processor = self.env.load_image_processor(self.cfg.vision_encoder_repo)
        from PIL import Image
        probe = processor(images=[Image.new("RGB", (64, 64))], max_num_patches=self.cfg.style_max_num_patches,
                          return_tensors="pt")
        if "pixel_attention_mask" not in probe or "spatial_shapes" not in probe:
            raise ValueError(f"{self.cfg.vision_encoder_repo} has no NaFlex image processor; the style encoder "
                             f"needs pixel_attention_mask and spatial_shapes")
        return processor

    def _build_model(self, processor, device: torch.device) -> StyleEncoder:
        cfg = self.cfg
        lora = LoRASpec(cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout) if cfg.training_method == "lora" else None
        model = build_style_encoder(cfg.vision_encoder_path, cfg.vision_encoder_repo, cfg.attn_implementation,
                                    cfg.style_feature_layers, cfg.style_token_dim, cfg.style_pool_heads,
                                    cfg.style_projector_hidden_dim, cfg.style_embed_dim, cfg.training_method, lora,
                                    load_vision_encoder=self.env.load_vision_encoder)
        patch = getattr(processor, "patch_size", None)
        if patch is not None and patch != model.vision_encoder.config.patch_size:
            raise ValueError(f"image processor patch size {patch} differs from the encoder's "
                             f"{model.vision_encoder.config.patch_size}")
        return model.to(device)

    def _log_data(self, data: StyleData) -> None:
        stats = {"ingest": data.ingest_stats.as_dict(), "labels": data.label_stats.as_dict(),
                 "index": data.index.stats.as_dict(), "index_hash": data.index.index_hash,
                 "epoch_length": data.sampler.epoch_length,
                 "same_copyright_candidates": int(data.sampler.qualifying_copyrights.size)}
        for section in ("ingest", "labels", "index"):
            print(f"[StyleTraining] {section}: " + ", ".join(f"{k}={v:,}" for k, v in stats[section].items()))
        index = data.index.stats
        print(f"[StyleTraining] Pinned to train: {index.pinned_items:,} sampled items; "
              f"{index.forced_train_artists:,} artists ({index.forced_train_items:,} items) whose name now hashes "
              f"to held-out stay on the train side")
        for example in data.skipped_examples:
            print(f"[StyleTraining]   skipped {example}")
        print(f"[StyleTraining] Index {data.index.index_hash[:12]}: epoch = {data.sampler.epoch_length:,} steps")
        self._emit("dataset_stats", stats)

    # --------------------------------------------------------------- training

    def _loss(self, embedding: torch.Tensor, rows: np.ndarray, index: StyleIndex) -> tuple[torch.Tensor, dict]:
        """Total loss and its logged parts; embedding rows i and i + len(rows) are the two views of rows[i]."""
        cfg = self.cfg
        weights = positive_weights(index.item_artist[rows].tolist(), [index.content_ids(int(r)) for r in rows],
                                   cfg.style_view_positive_weight, cfg.style_positive_weight_min)
        supcon = weighted_supcon_loss(embedding, weights, cfg.style_temperature)
        return supcon.loss, {"supcon_loss": float(supcon.loss.detach()),
                             "anchors_used": int(supcon.anchors_used), "anchors_excluded": int(supcon.anchors_excluded)}

    def _autocast(self, device: torch.device):
        if self.cfg.mixed_precision == "bf16" and device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _train_step(self, model, batch, data: StyleData, optimizer, scheduler, trainable, device, epoch: int,
                    step: int) -> Optional[dict]:
        rows = batch.rows.numpy()
        model.set_dropout_entropy(lora_dropout_entropy(self.cfg.style_seed, epoch, step))
        pv = batch.pixel_values.to(device)
        ss = batch.spatial_shapes.to(device)
        mask = torch.cat([batch.attention_mask, batch.masked_attention_mask]).to(device)
        optimizer.zero_grad(set_to_none=True)
        with self._autocast(device):
            out = model(torch.cat([pv, pv]), mask, torch.cat([ss, ss]))
        loss, parts = self._loss(out.embedding, rows, data.index)
        if not torch.isfinite(loss):
            print(f"[StyleTrainer] WARNING: non-finite loss, skipping batch")
            optimizer.zero_grad(set_to_none=True)
            return None
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(trainable, max_norm=self.cfg.max_grad_norm)
        if not torch.isfinite(grad_norm):
            print(f"[StyleTrainer] WARNING: non-finite grad norm ({float(grad_norm):.3g}), skipping optimizer step")
            optimizer.zero_grad(set_to_none=True)
            return None
        lr = scheduler.get_last_lr()[0]  # the rate this optimizer step applies
        optimizer.step()
        self._optimizer_steps_completed += 1
        scheduler.step()
        return {"loss": float(loss.detach()), "grad_norm": float(grad_norm), "lr": lr, "images": int(rows.size),
                **parts}

    def _build_optimizer(self, model: StyleEncoder, total_steps: int):
        from core.tagger.tagger_trainer import _build_optimizer
        cfg = self.cfg
        named = model.trainable_named_parameters()
        encoder = [p for n, p in named if n.startswith("vision_encoder.")]
        heads = [p for n, p in named if not n.startswith("vision_encoder.")]
        groups = [g for g in ({"params": encoder, "lr": cfg.learning_rate},
                              {"params": heads, "lr": cfg.learning_rate * cfg.head_lr_multiplier}) if g["params"]]
        optimizer = _build_optimizer(groups, cfg.optimizer, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = LambdaLR(optimizer, WarmupCosine(cfg.warmup_steps, total_steps, cfg.min_lr_ratio))
        return optimizer, scheduler, [p for _, p in named]

    # --------------------------------------------------------------- artifacts

    def _metadata(self, model, data: StyleData, processor, epoch: int, global_step: int, total_steps: int,
                  evaluation: Optional[dict]) -> dict:
        cfg = self.cfg
        preprocessing = json.loads(processor.to_json_string())
        preprocessing["max_num_patches"] = cfg.style_max_num_patches
        return {
            "format": ARTIFACT_FORMAT, "format_version": ARTIFACT_FORMAT_VERSION,
            "architecture": architecture_metadata(model),
            "base_model": {"repo_id": _public_source(cfg.vision_encoder_repo),
                           "source": _public_source(cfg.vision_encoder_path)},
            "attn_implementation": cfg.attn_implementation,
            "style_feature_layers": list(cfg.style_feature_layers), "style_embed_dim": cfg.style_embed_dim,
            "style_max_num_patches": cfg.style_max_num_patches, "preprocessing": preprocessing,
            "training_method": cfg.training_method,
            "lora": {"rank": cfg.lora_rank, "alpha": cfg.lora_alpha, "merged": True}
            if cfg.training_method == "lora" else None,
            "hsic_tagger": None,
            "dataset_snapshot": {"index_hash": data.index.index_hash,
                                 "dataset_ids": [c.dataset_id for c in cfg.dataset_configs],
                                 "index_stats": data.index.stats.as_dict()},
            "epoch": epoch, "global_step": global_step, "total_steps": total_steps, "evaluation": evaluation,
        }

    # -------------------------------------------------------------- evaluation

    def _embed(self, model, data: StyleData, rows: np.ndarray, masked: bool, device) -> tuple:
        loader = make_loader(data.dataset, EvalBatchKeys(rows, data.index.item_keys, self.cfg.style_seed,
                                                         self.cfg.style_val_batch_size), self.cfg.num_workers)
        kept, global_views, masked_views = [], [], []
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                pv = batch.pixel_values.to(device)
                ss = batch.spatial_shapes.to(device)
                if masked:
                    pv, ss = torch.cat([pv, pv]), torch.cat([ss, ss])
                    mask = torch.cat([batch.attention_mask, batch.masked_attention_mask])
                else:
                    mask = batch.attention_mask
                with self._autocast(device):
                    emb = model(pv, mask.to(device), ss).embedding.float().cpu()
                b = batch.rows.shape[0]
                kept.append(batch.rows.numpy())
                global_views.append(emb[:b])
                if masked:
                    masked_views.append(emb[b:])
        empty = torch.zeros(0, self.cfg.style_embed_dim)
        rows_out = np.concatenate(kept) if kept else np.zeros(0, dtype=np.int64)
        return (rows_out, torch.cat(global_views) if global_views else empty,
                (torch.cat(masked_views) if masked_views else empty) if masked else None)

    def _retrieve(self, index: StyleIndex, rows: np.ndarray, emb: torch.Tensor, device, same_copyright: bool = False):
        artists, keys = index.item_artist[rows], index.item_keys[rows]
        cps = _row_csr(index.item_copyright_indptr, index.item_copyrights, rows) if same_copyright else None
        g = emb.to(device)
        return retrieve(g, artists, keys, g, artists, keys, EVAL_RECALL_KS, self.cfg.style_eval_chunk_size,
                        self.cfg.style_seed, cps, cps)

    def _evaluate(self, model, data: StyleData, device, epoch: int, global_step: int) -> dict:
        cfg, index = self.cfg, data.index
        self._emit("phase", {"phase": "validation", "message": "Evaluating retrieval..."})
        was_training = model.training
        model.eval()
        try:
            artist_rows, artist_global, artist_masked = self._embed(
                model, data, eval_rows(index, VAL_ARTIST, cfg.style_val_max_items, cfg.style_seed), True, device)
            image_rows, image_global, _ = self._embed(
                model, data, eval_rows(index, VAL_IMAGE, cfg.style_val_max_items, cfg.style_seed), False, device)
        finally:
            model.train(was_training)
        held_out = self._retrieve(index, artist_rows, artist_global, device)
        signed = index.has_signature[artist_rows]
        metrics = {
            "epoch": epoch, "global_step": global_step, "index_hash": index.index_hash,
            "eval_items": {"held_out_artists": int(artist_rows.size), "train_artist_images": int(image_rows.size)},
            "held_out_artists": _summary(held_out.summarize()),
            "held_out_artists_same_copyright": _summary(
                self._retrieve(index, artist_rows, artist_global, device, same_copyright=True).summarize()),
            "held_out_artists_with_signature": _summary(held_out.summarize(signed)),
            "held_out_artists_without_signature": _summary(held_out.summarize(~signed)),
            "held_out_artists_masked": _summary(self._retrieve(index, artist_rows, artist_masked, device).summarize()),
            "train_artist_held_out_images": _summary(self._retrieve(index, image_rows, image_global, device).summarize()),
        }
        a = metrics["held_out_artists"]
        print(f"[StyleTrainer] Eval step {global_step}: held-out artists mAP@R={a['map_at_r']} "
              f"R@1={a['recall_at_1']} ({a['used']} used / {a['queries']} queries)")
        self._emit("validation", metrics)
        return metrics

    # -------------------------------------------------------------------- run

    def run(self) -> dict:
        cfg = self.cfg
        device = self.env.resolve_device(cfg.gpu_index)
        self._emit("phase", {"phase": "initializing", "message": "Preparing style training..."})
        resume = find_resume_checkpoint(self.output_dir)
        saved_snapshot = None
        if resume is None:
            leftovers = resumable_bundle_files(self.output_dir)
            if leftovers:
                raise ValueError(
                    f"{self.output_dir} holds checkpoint files {leftovers[:6]} but no complete resumable bundle. "
                    f"Starting fresh would discard this run's split snapshot (the items pinned to train); restore "
                    f"the bundle, remove these files, or use another output directory.")
        else:
            resume_name, saved = resume
            if saved["model"] != cfg.identity():
                raise ValueError(f"checkpoint {resume_name} was trained with {saved['model']}, the config now asks "
                                 f"for {cfg.identity()}; start a new run in another output directory instead")
            saved_snapshot = SplitSnapshot.load(artifact_path(self.output_dir, resume_name, SNAPSHOT_SUFFIX))
            print(f"[StyleTraining] Resuming from {resume_name} (step {saved['global_step']})")

        processor = self._load_processor()
        self._emit("phase", {"phase": "dataset", "message": "Loading datasets..."})
        data = prepare_style_data(
            cfg, self.env, processor, saved_snapshot.keys if saved_snapshot is not None else np.zeros(0, np.int64),
            lambda done, total, msg: self._emit("dataset_progress", {"step": done, "total": total, "message": msg}))
        self._log_data(data)

        best, last_eval_epoch, pending_eval = None, None, None
        if resume is not None:
            plan = plan_resume(ResumeKey.from_state(saved["resume_key"]), saved_snapshot, data.sampler, data.views,
                               int(saved["epoch"]), int(saved["step_in_epoch"]))
            epoch, step, snapshot = plan.epoch, plan.step_in_epoch, plan.next_snapshot
            global_step, total_steps, best = int(saved["global_step"]), int(saved["total_steps"]), saved["best"]
            schedule, last_eval_epoch = saved["schedule"], saved["last_eval_epoch"]
            ignored = {k: {"saved": schedule[k], "requested": v} for k, v in cfg.schedule().items() if schedule[k] != v}
            if ignored:
                message = (f"Resume keeps the run's schedule; these changed settings are ignored: {ignored}. "
                           f"Start a new run to apply them.")
                print(f"[StyleTraining] WARNING: {message}")
                self._emit("warning", {"message": message, "ignored": ignored})
            if step == 0 and epoch >= 1 and epoch % cfg.validate_every == 0 and last_eval_epoch != epoch - 1:
                pending_eval = epoch - 1
            print(f"[StyleTraining] Resume mode {plan.mode}: epoch {epoch}, step {step}, "
                  f"global step {global_step}/{total_steps}")
            self._emit("resume", {"mode": plan.mode, "resumed_from_step": global_step, "resume_ckpt_name": resume_name,
                                  "epoch": epoch, "step_in_epoch": step})
        else:
            epoch, step, snapshot = 0, 0, start_snapshot(data.index)
            global_step, total_steps = 0, cfg.epochs * data.sampler.epoch_length
            schedule = cfg.schedule()
        resume_key = ResumeKey.of(data.sampler, data.views)

        self._emit("phase", {"phase": "initializing", "message": "Building model..."})
        with torch.random.fork_rng(devices=[]):  # seeded pooling/projector/LoRA init
            torch.manual_seed(cfg.style_seed)
            model = self._build_model(processor, device)
        self.model = model
        if resume is not None:
            load_training_weights(model, self.output_dir, resume_name)
        if cfg.gradient_checkpointing:
            model.vision_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer, scheduler, trainable = self._build_optimizer(model, total_steps)
        if resume is not None:
            saved_opt = torch.load(artifact_path(self.output_dir, resume_name, OPTIMIZER_SUFFIX), map_location="cpu",
                                   weights_only=True)
            optimizer.load_state_dict(saved_opt["optimizer"])
            scheduler.load_state_dict(saved_opt["scheduler"])
            if scheduler.last_epoch != global_step:
                raise ValueError(f"scheduler position {scheduler.last_epoch} != global step {global_step}")
        print(f"[StyleTrainer] {sum(p.numel() for p in trainable):,} trainable parameters, {total_steps:,} total steps, "
              f"device={device}, amp={cfg.mixed_precision if device.type == 'cuda' else 'off'}")

        from core.gpu_coordinator import gpu_coordinator
        self._coordinator_handle.attach(model, optimizer, None)
        gpu_coordinator.register_trainer(self._coordinator_handle)
        try:
            return self._loop(model, data, processor, optimizer, scheduler, trainable, device, resume_key, snapshot,
                              epoch, step, global_step, total_steps, best, schedule, last_eval_epoch, pending_eval,
                              resume is not None)
        finally:
            gpu_coordinator.unregister_trainer(self._coordinator_handle)
            self._coordinator_handle.detach()

    def _loop(self, model, data: StyleData, processor, optimizer, scheduler, trainable, device, resume_key: ResumeKey,
              snapshot: SplitSnapshot, epoch: int, step: int, global_step: int, total_steps: int,
              best: Optional[dict], schedule: dict, last_eval_epoch: Optional[int], pending_eval: Optional[int],
              resumed: bool) -> dict:
        from core.tagger.tagger_trainer import _prefetch_loader
        cfg, sampler = self.cfg, data.sampler
        epoch_length = sampler.epoch_length
        last_eval: Optional[dict] = None
        saved_at: set = set()

        def save(name: str, resumable: bool, pos: tuple[int, int]) -> str:
            metadata = self._metadata(model, data, processor, pos[0], global_step, total_steps, last_eval)
            resume = None
            if resumable:
                state = {"global_step": global_step, "total_steps": total_steps, "epoch": pos[0],
                         "step_in_epoch": pos[1], "resume_key": resume_key.to_state(), "model": cfg.identity(),
                         "schedule": schedule, "best": best, "last_eval_epoch": last_eval_epoch,
                         "index_hash": data.index.index_hash}
                resume = (state, {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, snapshot)
            path = save_bundle(model, self.output_dir, name, metadata, resume)
            self._emit("checkpoint", {"name": name, "step": global_step, "epoch": pos[0], "path": path})
            return path

        def evaluate(eval_epoch: int) -> None:
            nonlocal last_eval, best, last_eval_epoch
            last_eval = self._evaluate(model, data, device, eval_epoch, global_step)
            last_eval_epoch = eval_epoch
            score = last_eval["held_out_artists"]["map_at_r"]
            # Resumed weights are trained; a fresh run's only once a step has completed.
            if score is None or not (resumed or self._optimizer_steps_completed > 0):
                return
            if best is not None and best.get("index_hash") != last_eval["index_hash"]:
                message = (f"Evaluation set changed (index {str(best.get('index_hash'))[:12]} -> "
                           f"{last_eval['index_hash'][:12]}); best mAP@R {best['map_at_r']:.4f} is not comparable, "
                           f"so this evaluation becomes the new best")
                print(f"[StyleTrainer] {message}")
                self._emit("warning", {"message": message})
                best = None
            if best is None or score > best["map_at_r"]:
                best = {"map_at_r": score, "global_step": global_step, "epoch": eval_epoch,
                        "index_hash": last_eval["index_hash"]}
                save("best", False, (eval_epoch, step))

        if pending_eval is not None:
            print(f"[StyleTrainer] Running epoch {pending_eval}'s evaluation, which the interrupted run skipped")
            evaluate(pending_eval)

        consecutive_skips = 0
        epoch_loss, epoch_steps = 0.0, 0
        batch_keys = TrainBatchKeys(sampler, cfg.style_seed, epoch, step, step)
        loader = make_loader(data.dataset, batch_keys, cfg.num_workers, persistent=True)
        print(f"[StyleTrainer] === Phase: training === ({global_step}/{total_steps} steps done)")
        self._emit("phase", {"phase": "training", "message": "Training started"})
        while global_step < total_steps and not self._stop_requested:
            stop_step = min(epoch_length, step + (total_steps - global_step))
            batch_keys.set_range(epoch, step, stop_step)
            # Per segment, so an early exit also ends the prefetch thread (it only watches this event).
            segment_done = threading.Event()
            batches = _prefetch_loader(loader, segment_done) if cfg.num_workers == 0 else iter(loader)
            model.train()
            try:
                for batch in batches:
                    if self._stop_requested:
                        break
                    self._batches_entered += 1
                    result = None if batch is None else self._train_step(
                        model, batch, data, optimizer, scheduler, trainable, device, epoch, step)
                    step += 1
                    if result is None:
                        consecutive_skips += 1
                        if consecutive_skips >= epoch_length:
                            self._refuse_skipped(consecutive_skips)
                    else:
                        consecutive_skips = 0
                        global_step += 1
                        epoch_loss += result["loss"]
                        epoch_steps += 1
                        if global_step % cfg.log_every_n_steps == 0:
                            self._emit("step", {"step": global_step, "epoch": epoch, "step_in_epoch": step,
                                                "progress": global_step / total_steps,
                                                "index_hash": data.index.index_hash, **result})
                        if cfg.save_every_n_steps > 0 and global_step % cfg.save_every_n_steps == 0:
                            save(f"step_{global_step:06d}", True, (epoch, step))
                            saved_at.add(global_step)
                            prune_step_bundles(self.output_dir, cfg.keep_last_n_checkpoints)
                    self._coordinator_handle.service_pause(self._stop_event, lambda: self._stop_requested)
            finally:
                segment_done.set()
                close = getattr(batches, "close", None)
                if close is not None:
                    close()
            if self._stop_requested:
                break
            if step < stop_step:
                # The prefetch thread ends the stream quietly when the loader raises; retrying would loop forever.
                raise RuntimeError(f"the batch loader stopped early in epoch {epoch}: steps {step}..{stop_step - 1} "
                                   f"were never delivered (the loader error is printed above)")
            if step == epoch_length:
                self._emit("epoch", {"epoch": epoch, "step": global_step,
                                     "loss": epoch_loss / epoch_steps if epoch_steps else None})
                if (epoch + 1) % cfg.validate_every == 0:
                    evaluate(epoch)
                if self._optimizer_steps_completed > 0:  # else the weights are the ones loaded
                    save("latest", True, (epoch, step))
                    saved_at.add(global_step)
                epoch, step, epoch_loss, epoch_steps = epoch + 1, 0, 0.0, 0

        if self._stop_requested:
            if self._optimizer_steps_completed == 0:
                print("[StyleTrainer] Stop requested before any optimizer step; no checkpoint written")
            else:
                save(f"step_{global_step:06d}", True, (epoch, step))
                save("latest", True, (epoch, step))
                prune_step_bundles(self.output_dir, cfg.keep_last_n_checkpoints)
            return {"stopped": True, "global_step": global_step, "total_steps": total_steps, "best": best}

        if self._batches_entered > 0 and self._optimizer_steps_completed == 0:
            self._refuse_skipped(self._batches_entered)
        if self._batches_entered > 0:
            if last_eval is None or last_eval["global_step"] != global_step:
                evaluate(epoch if step > 0 else epoch - 1)
            if global_step not in saved_at:
                save("latest", True, (epoch, step))
        summary = {"global_step": global_step, "total_steps": total_steps, "best": best, "evaluation": last_eval}
        self._emit("completed", summary)
        return summary

    def _refuse_skipped(self, batches: int) -> None:
        from core.training.base_trainer import NothingTrainedError
        if self._optimizer_steps_completed == 0:
            raise NothingTrainedError(
                f"Style training skipped {batches} batch(es) and completed no optimizer step: every batch was "
                f"unreadable images, a non-finite loss or a non-finite gradient norm. No checkpoint was written.")
        raise RuntimeError(f"Style training skipped {batches} consecutive batches (an epoch's worth) without an "
                           f"optimizer step; stopping instead of looping over unusable batches.")


def run_style_training(run_id: str, config: dict, output_dir: str, progress_callback: Optional[Callable] = None,
                       trainer_holder: Optional[list] = None,
                       environment: Optional[StyleEnvironment] = None) -> dict:
    """Thread target for a style training run; resumes from the furthest complete bundle in `output_dir`."""
    trainer = StyleTrainer(run_id, config, output_dir, progress_callback, environment)
    if trainer_holder is not None:
        trainer_holder.append(trainer)
    try:
        return trainer.run()
    finally:
        import gc
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
