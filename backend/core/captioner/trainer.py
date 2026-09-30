"""Item-first caption training with immutable snapshots and exact resume."""

from dataclasses import replace
from collections import Counter
import copy
import math
from pathlib import Path
import random
import threading
import uuid

import torch
from torch import nn

from api.param_defaults import CAPTIONER_TRAINING_DEFAULTS
from .artifact import load_bundle, save_bundle, tagger_identity
from .concepts import CONCEPT_CATEGORIES, ConceptSelector
from .dataset import build_snapshot, verify_snapshot
from .features import CaptionFeatures, SigLIP2FeatureProvider
from .model import CaptionConfig, CompactCaptioner
from .storage import FeatureCache, canonical_hash, file_hash, move_features, read_json, write_json
from .tokenizer import CaptionTokenizer


def validate_config(values):
    unknown = set(values) - set(CAPTIONER_TRAINING_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown caption training settings: {sorted(unknown)}")
    cfg = copy.deepcopy({**CAPTIONER_TRAINING_DEFAULTS, **values})
    for key in ("epochs", "batch_size", "save_every_steps", "max_concepts", "max_num_patches",
                "hidden_dim", "heads", "layers", "ffn_dim", "visual_queries", "max_length",
                "vocab_size", "tokenizer_min_frequency"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    for key in ("dropout", "label_smoothing", "concept_dropout", "all_concept_dropout", "warmup_ratio"):
        if not math.isfinite(cfg[key]) or not 0 <= cfg[key] < 1:
            raise ValueError(f"{key} must be finite in [0,1)")
    for key in ("learning_rate", "max_grad_norm", "cache_max_gb"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not math.isfinite(cfg["weight_decay"]) or cfg["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if cfg["hidden_dim"] % cfg["heads"] or cfg["vocab_size"] < 260:
        raise ValueError("Invalid decoder heads or byte-level vocabulary size")
    if cfg["device"] != "cpu" and cfg["device"] != "cuda:0":
        raise ValueError("Captioner currently supports cpu or cuda:0")
    weights = cfg["source_weights"]
    if set(weights) != {"predicted", "provided", "merged"} or any(
            not math.isfinite(value) or value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Invalid source_weights")
    if not cfg["language"].strip():
        raise ValueError("Select one corpus language")
    if not 0 <= cfg["val_ratio"] < 1 or not 0 <= cfg["test_ratio"] < 1 or cfg["val_ratio"] + cfg["test_ratio"] >= 1:
        raise ValueError("Invalid split ratios")
    if type(cfg["near_duplicate_distance"]) is not int or not 0 <= cfg["near_duplicate_distance"] <= 16:
        raise ValueError("Invalid near_duplicate_distance")
    if not isinstance(cfg["use_tag_aliases"], bool):
        raise ValueError("use_tag_aliases must be boolean")
    if not cfg["allowed_categories"] or any(c not in CONCEPT_CATEGORIES for c in cfg["allowed_categories"]):
        raise ValueError("Invalid allowed_categories")
    return cfg


def collate_features(samples, device):
    count = max(sample.spatial_tokens.shape[1] for sample in samples)
    batch = len(samples)
    dimension = samples[0].spatial_tokens.shape[-1]
    tokens = samples[0].spatial_tokens.new_zeros(batch, count, dimension)
    xy = torch.zeros(batch, count, 2)
    mask = torch.zeros(batch, count, dtype=torch.bool)
    for row, sample in enumerate(samples):
        length = sample.spatial_tokens.shape[1]
        tokens[row, :length] = sample.spatial_tokens[0]
        xy[row, :length] = sample.spatial_xy[0]
        mask[row, :length] = sample.spatial_mask[0]
    return move_features(CaptionFeatures(tokens, mask, xy, torch.cat([s.global_features for s in samples]),
                                         torch.cat([s.tag_logits for s in samples])), device)


class CaptionTrainer:
    def __init__(self, *, run_id, output_dir, config, dataset_configs, tagger_manager,
                 session_factory, progress, stop_event, alias_table, resume_path=None):
        self.run_id, self.root = run_id, Path(output_dir)
        self.cfg = validate_config(config)
        self.dataset_configs = dataset_configs
        self.manager = tagger_manager
        self.session_factory = session_factory
        self.progress, self.stop = progress, stop_event
        self.aliases = dict(alias_table) if self.cfg["use_tag_aliases"] else {}
        self.resume_path = resume_path

    def run(self):
        cfg = self.cfg
        device = torch.device(cfg["device"])
        if device.type == "cuda":
            torch.cuda.set_device(device)
        identity = tagger_identity(self.manager, max_num_patches=cfg["max_num_patches"])
        self.manager.model.requires_grad_(False)
        provider = SigLIP2FeatureProvider(self.manager.model)
        if self.resume_path:
            model, tokenizer, selector, manifest, state = load_bundle(
                self.resume_path, expected_identity=identity, require_resume=True)
            snapshot = state["snapshot"]
            verify_snapshot(snapshot)
            if (state["config"] != cfg or state["dataset_configs"] != self.dataset_configs
                    or manifest["snapshot_hash"] != snapshot["snapshot_hash"]):
                raise ValueError("Resume settings or dataset snapshot changed")
            if (self.root / "snapshot.json").exists():
                if read_json(self.root / "snapshot.json") != snapshot:
                    raise ValueError("Run snapshot differs from resume checkpoint")
            else:
                write_json(self.root / "snapshot.json", snapshot)
            self.aliases = manifest["aliases"]
        else:
            torch.manual_seed(cfg["seed"])
            self.progress({"phase": "dataset"})
            with self.session_factory() as db:
                snapshot = build_snapshot(db, self.dataset_configs, seed=cfg["seed"],
                    val_ratio=cfg["val_ratio"], test_ratio=cfg["test_ratio"],
                    near_duplicate_distance=cfg["near_duplicate_distance"], cancelled=self.stop.is_set)
            tokenizer = CaptionTokenizer.train((caption for r in snapshot["records"] if r["split"] == "train"
                                                for caption in r["targets"]), vocab_size=cfg["vocab_size"],
                                               min_frequency=cfg["tokenizer_min_frequency"])
            dropped = 0
            for record in snapshot["records"]:
                accepted = []
                for target in record["targets"]:
                    try:
                        tokenizer.encode_target(target, max_length=cfg["max_length"])
                        accepted.append(target)
                    except ValueError:
                        dropped += 1
                record["targets"] = accepted
            snapshot["records"] = [r for r in snapshot["records"] if r["targets"]]
            snapshot["split_counts"] = dict(Counter(r["split"] for r in snapshot["records"]))
            snapshot["coverage"]["overlong_or_special_targets"] = dropped
            snapshot.pop("snapshot_hash")
            snapshot["snapshot_hash"] = canonical_hash(snapshot)
            write_json(self.root / "snapshot.json", snapshot)
            from core.tagger.tag_alias_resolver import TagAliasResolver
            selector = ConceptSelector(identity["names"], identity["categories"], max_concepts=cfg["max_concepts"],
                                       allowed_categories=cfg["allowed_categories"],
                                       alias_resolver=TagAliasResolver(self.aliases).resolve)
            state = None
            first = self._extract(snapshot["records"][0], provider, None) if snapshot["records"] else None
            if first is None:
                raise ValueError("No caption targets fit the output limit")
            sample_bytes = sum(t.numel() * t.element_size() for t in vars(first).values())
            self.progress({"phase": "cache_estimate", "sample_feature_bytes": sample_bytes,
                           "estimated_feature_bytes": sample_bytes * len(snapshot["records"]),
                           "cache_limit_bytes": int(cfg["cache_max_gb"] * 1024**3),
                           "estimate_basis": "first_image", "coverage": snapshot["coverage"],
                           "split_counts": snapshot["split_counts"]})
            model = CompactCaptioner(CaptionConfig(
                spatial_dim=first.spatial_tokens.shape[-1], global_dim=first.global_features.shape[-1],
                num_tags=len(identity["names"]), vocab_size=tokenizer.vocab_size,
                **{key: cfg[key] for key in ("hidden_dim", "heads", "layers", "ffn_dim", "visual_queries", "max_length", "dropout")},
                num_categories=len(CONCEPT_CATEGORIES), pad_id=tokenizer.pad_id, bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id))
        train = [r for r in snapshot["records"] if r["split"] == "train"]
        validation = [r for r in snapshot["records"] if r["split"] == "val"]
        if (not train or cfg["val_ratio"] and not validation
                or cfg["test_ratio"] and not any(r["split"] == "test" for r in snapshot["records"])):
            raise ValueError("Train or requested holdout split has no usable captions")
        cache = FeatureCache(self.root / "features", identity=identity, max_bytes=int(cfg["cache_max_gb"] * 1024**3))
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        total_steps = math.ceil(len(train) / cfg["batch_size"]) * cfg["epochs"]
        warmup = int(total_steps * cfg["warmup_ratio"])
        def lr_scale(step):
            if warmup and step < warmup:
                return (step + 1) / warmup
            fraction = min(1, (step - warmup) / max(1, total_steps - warmup))
            return 0.5 * (1 + math.cos(math.pi * fraction))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        rng = random.Random(cfg["seed"])
        epoch, offset, step, best = 0, 0, 0, None
        if state:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            epoch, offset, step, best = state["epoch"], state["offset"], state["step"], state["best"]
            rng.setstate(state["sampling_rng"])
            torch.set_rng_state(state["torch_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(state["cuda_rng"], device)
        handle = None
        if device.type == "cuda":
            from core.gpu_coordinator import gpu_coordinator
            from core.tagger.tagger_offload import TaggerTrainerHandle
            handle = TaggerTrainerHandle(self.run_id, str(self.root), threading.Event(), threading.Event(),
                                         threading.Event(), device_index=0)
        def allocate():
            model.to(device)
            self.manager.model.to(device)
            for values in optimizer.state.values():
                for key, value in values.items():
                    if torch.is_tensor(value) and key != "step":
                        values[key] = value.to(device)
            if handle:
                handle.attach(nn.ModuleDict({"captioner": model, "tagger": self.manager.model}), optimizer, None)
                gpu_coordinator.register_trainer(handle)
        if handle:
            try:
                with gpu_coordinator.trainer_allocation():
                    allocate()
            except Exception:
                self.manager.model.to("cpu")
                model.to("cpu")
                raise
        else:
            allocate()

        def pause():
            if handle:
                handle.service_pause(self.stop, self.stop.is_set)

        def save():
            name = f"checkpoint_{step:08d}_{uuid.uuid4().hex[:8]}"
            resume = {"config": cfg, "dataset_configs": self.dataset_configs, "optimizer": optimizer.state_dict(),
                      "snapshot": snapshot,
                      "scheduler": scheduler.state_dict(), "epoch": epoch, "offset": offset, "step": step,
                      "best": best, "sampling_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
            save_bundle(self.root / name, model=model, tokenizer=tokenizer, selector=selector, identity=identity,
                        training_state=resume, snapshot_hash=snapshot["snapshot_hash"], aliases=self.aliases,
                        language=cfg["language"])
            write_json(self.root / "latest.json", {"checkpoint": name, "step": step})
            self.progress({"checkpoint": str(self.root / name), "step": step})

        def batch(records, training):
            pause()
            features = collate_features([self._extract(r, provider, cache) for r in records], device)
            mode = rng.choices(list(cfg["source_weights"]), weights=list(cfg["source_weights"].values()))[0] if training else "predicted"
            provided = [r["tags"] for r in records] if mode != "predicted" else None
            concepts = selector.select(features.tag_logits, mode=mode, provided=provided)
            if training:
                mask = concepts.mask.clone()
                for row in range(len(records)):
                    if rng.random() < cfg["all_concept_dropout"]:
                        mask[row] = False
                    else:
                        for column in range(mask.shape[1]):
                            if rng.random() < cfg["concept_dropout"]:
                                mask[row, column] = False
                permutation = list(range(mask.shape[1]))
                rng.shuffle(permutation)
                concepts = replace(concepts, mask=mask)
                concepts = replace(concepts, **{key: getattr(concepts, key)[:, permutation] for key in
                                                 ("ids", "scores", "categories", "sources", "mask")})
            targets = [tokenizer.encode_target(rng.choice(r["targets"]) if training else r["targets"][0],
                                               max_length=cfg["max_length"]) for r in records]
            sequences = torch.full((len(targets), max(map(len, targets))), tokenizer.pad_id, dtype=torch.long, device=device)
            for row, target in enumerate(targets):
                sequences[row, :len(target)] = torch.tensor(target, device=device)
            return model.teacher_loss(features, concepts, sequences, label_smoothing=cfg["label_smoothing"] if training else 0.0), sum(len(t) - 1 for t in targets)

        try:
            while epoch < cfg["epochs"] and not self.stop.is_set():
                order = list(range(len(train)))
                random.Random(cfg["seed"] + epoch).shuffle(order)
                model.train()
                while offset < len(order) and not self.stop.is_set():
                    selected = order[offset:offset + cfg["batch_size"]]
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = batch([train[i] for i in selected], True)
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite caption loss")
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"], error_if_nonfinite=True)
                    optimizer.step()
                    scheduler.step()
                    offset += len(selected)
                    step += 1
                    self.progress({"phase": "training", "step": step, "total_steps": total_steps,
                                   "epoch": epoch, "loss": loss.item(), "gradient_norm": float(norm),
                                   "learning_rate": optimizer.param_groups[0]["lr"]})
                    if step % cfg["save_every_steps"] == 0:
                        save()
                if self.stop.is_set():
                    break
                model.eval()
                numerator, denominator = 0.0, 0
                with torch.no_grad():
                    for start in range(0, len(validation), cfg["batch_size"]):
                        if self.stop.is_set():
                            break
                        loss, tokens = batch(validation[start:start + cfg["batch_size"]], False)
                        numerator += loss.item() * tokens
                        denominator += tokens
                if denominator:
                    validation_loss = numerator / denominator
                    best = validation_loss if best is None else min(best, validation_loss)
                    self.progress({"validation_loss": validation_loss, "step": step, "best_validation_loss": best})
                epoch += 1
                offset = 0
                save()
            save()
            return {"step": step, "epoch": epoch, "stopped": self.stop.is_set(), "best_validation_loss": best}
        except InterruptedError:
            save()
            return {"step": step, "epoch": epoch, "stopped": True, "best_validation_loss": best}
        finally:
            if handle:
                gpu_coordinator.unregister_trainer(handle)
                handle.detach()
            self.manager.model.to("cpu")
            model.to("cpu")

    def _extract(self, record, provider, cache):
        if self.stop.is_set():
            raise InterruptedError("Caption training cancelled")
        if file_hash(record["path"]) != record["image_hash"]:
            raise ValueError("Dataset image changed after snapshot")
        cached = cache.get(record["image_hash"]) if cache else None
        if cached is not None:
            return cached
        from core.tagger.tagger_dataset import _load_image
        image = _load_image(record["path"])
        kwargs = {"images": [image], "return_tensors": "pt"}
        if self.manager.is_naflex:
            kwargs["max_num_patches"] = self.cfg["max_num_patches"]
        inputs = self.manager.processor(**kwargs)
        device = next(self.manager.model.parameters()).device
        features = provider.extract(inputs["pixel_values"].to(device),
                inputs.get("pixel_attention_mask").to(device) if "pixel_attention_mask" in inputs else None,
                inputs.get("spatial_shapes").to(device) if "spatial_shapes" in inputs else None)
        features = move_features(features, "cpu")
        if cache:
            cache.put(record["image_hash"], features)
        return features
