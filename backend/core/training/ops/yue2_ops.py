"""YuE2 Phase-A score-planner training contracts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import contextlib
import hashlib
import math
import random
import re

import torch

from core.models.yue2.vendor.protocol import ABC_END, ABC_START, CONTEXT, EOD, SongRequest


YUE2_TRAINING_PROTOCOL_VERSION = "yue2-abc-ar-v1"


@dataclass(frozen=True)
class YuE2ARExample:
    input_ids: torch.LongTensor
    labels: torch.LongTensor
    attention_mask: torch.LongTensor
    target_tokens: int


def build_abc_ar_example(
    tokenizer,
    *,
    style: str,
    lyrics: str,
    abc: str,
    cot: str = "full",
    context: int = CONTEXT,
    allow_truncated_targets: bool = False,
) -> YuE2ARExample:
    """Build native protocol tokens and supervise only ABC content plus ABC_END."""
    if cot not in {"full", "melody"}:
        raise ValueError("YuE2 abc_ar training requires cot='full' or 'melody'")
    if not isinstance(style, str) or not style.strip():
        raise ValueError("YuE2 abc_ar training requires a non-empty style")
    if not isinstance(lyrics, str) or not lyrics.strip():
        raise ValueError("YuE2 abc_ar training requires non-empty structured lyrics")
    if not isinstance(abc, str) or not abc.strip():
        raise ValueError("YuE2 abc_ar training requires a non-empty ABC score")
    request = SongRequest(style=style.strip(), lyrics=lyrics.strip(), cot=cot)
    prefix = [EOD] + list(tokenizer.encode(request.text())) + [ABC_START]
    target = list(tokenizer.encode(abc.strip())) + [ABC_END]
    if any(type(token) is not int or not 0 <= token < EOD for token in target[:-1]):
        raise ValueError("ABC token IDs must remain inside the ordinary text vocabulary")
    available = context - len(prefix)
    if available < 1:
        raise ValueError("YuE2 conditioning prefix leaves no room for an ABC target")
    if len(target) > available:
        if not allow_truncated_targets:
            raise ValueError(
                f"YuE2 ABC target exceeds context ({len(prefix)} prefix + {len(target)} target > {context})"
            )
        target = target[:available]
        target[-1] = ABC_END
    ids = torch.tensor(prefix + target, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    labels[len(prefix):] = ids[len(prefix):]
    return YuE2ARExample(ids, labels, torch.ones_like(ids), len(target))


def collate_abc_ar(examples: Sequence[YuE2ARExample], pad_token_id: int = EOD) -> dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("Cannot collate an empty YuE2 batch")
    longest = max(example.input_ids.numel() for example in examples)
    batch = len(examples)
    input_ids = torch.full((batch, longest), pad_token_id, dtype=torch.long)
    labels = torch.full((batch, longest), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch, longest), dtype=torch.long)
    for row, example in enumerate(examples):
        length = example.input_ids.numel()
        input_ids[row, :length] = example.input_ids
        labels[row, :length] = example.labels
        attention_mask[row, :length] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def train_abc_ar_step(transformer, batch: dict[str, torch.Tensor]):
    """Run the ordinary causal-LM loss without cache or NAR/VAE work."""
    output = transformer(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        use_cache=False,
        logits_to_keep=0,
        return_dict=True,
    )
    if output.loss is None or not torch.isfinite(output.loss):
        raise FloatingPointError("YuE2 abc_ar produced a non-finite loss")
    return output.loss


def load_components(trainer) -> None:
    """Load the complete single file and stage only the AR planner on device."""
    objective = str((trainer.config or {}).get("yue2_training_objective", "abc_ar"))
    if objective != "abc_ar":
        raise ValueError(
            f"YuE2 objective {objective!r} is not released yet; Phase A supports only 'abc_ar'"
        )
    if getattr(trainer, "train_text_encoder", False):
        raise ValueError("YuE2 embeds its tokenizer/AR model; train_text_encoder must be false")
    from core.models.yue2.loader import load_yue2_from_path

    components = load_yue2_from_path(trainer.model_path, trainer.weight_dtype)
    trainer.transformer = components["transformer"]
    trainer.transformer_original = trainer.transformer
    trainer.tokenizer = components["tokenizer"]
    full_finetune = bool(getattr(trainer, "trains_base_weights", False))
    trainer.yue2_frozen_vae = components["vae"] if full_finetune else None
    trainer.vae = None
    trainer.text_encoder = None
    trainer.text_encoder_2 = None
    trainer.tokenizer_2 = None
    trainer.unet = None
    trainer.scheduler = None
    trainer.noise_scheduler = None
    trainer.vae_latent_channels = 64
    trainer.yue2_model_identity = components["model_identity"]
    trainer.transformer.requires_grad_(False)
    if getattr(trainer, "gradient_checkpointing", False):
        trainer.transformer.gradient_checkpointing_enable()
    # The complete file is resumable, but abc_ar never executes NAR or VAE.
    from core.models.yue2.pipeline import ar_modules, move
    move(ar_modules(trainer.transformer), trainer.device)


def _abc_path(item: dict) -> Path:
    source = Path(item.get("audio_path") or item.get("image_path") or "")
    if source.suffix.lower() == ".abc":
        return source
    return source.with_suffix(".abc")


def prepare_abc_items(trainer, datasets) -> list[YuE2ARExample]:
    """Resolve dataset rows into immutable native-token examples before training."""
    examples: list[YuE2ARExample] = []
    allow_truncated = bool((trainer.config or {}).get("yue2_allow_truncated_targets", False))
    cot = str((trainer.config or {}).get("yue2_abc_mode", "full"))
    for dataset in datasets:
        for item in dataset.items:
            trainer._check_stop_requested()
            score = _abc_path(item)
            if not score.is_file():
                raise ValueError(f"YuE2 abc_ar requires sibling ABC score: {score}")
            sidecar = score.with_suffix(".yue2.json")
            if sidecar.is_file() and not allow_truncated:
                import json
                try:
                    truncated = json.loads(sidecar.read_text(encoding="utf-8")).get("truncated", {})
                except (OSError, ValueError) as exc:
                    raise ValueError(f"Invalid YuE2 sidecar beside {score}") from exc
                if truncated.get("abc"):
                    raise ValueError(
                        f"YuE2 ABC target was truncated during generation: {score}; "
                        "set yue2_allow_truncated_targets only after reviewing it"
                    )
            examples.append(build_abc_ar_example(
                trainer.tokenizer,
                style=str(item.get("caption") or ""),
                lyrics=str(item.get("lyrics") or ""),
                abc=score.read_text(encoding="utf-8-sig"),
                cot=cot,
                allow_truncated_targets=allow_truncated,
            ))
    if not examples:
        raise ValueError("YuE2 abc_ar found no training examples")
    return examples


def _dataset_fingerprint(datasets) -> dict:
    rows = []
    for dataset in datasets:
        for item in dataset.items:
            score = _abc_path(item).resolve()
            content = hashlib.sha256(score.read_bytes()).hexdigest()
            rows.append("\0".join((str(score), content, str(item.get("caption") or ""),
                                     str(item.get("lyrics") or ""))))
    rows.sort()
    digest = hashlib.sha256("\0".join(rows).encode()).hexdigest()
    return {"total_item_count": len(rows), "image_paths_hash": digest}


def _save_recovery_bundle(trainer, *, step: int, epoch: int, batch_idx: int,
                          max_step_saves_to_keep: int,
                          max_optimizer_saves_to_keep: int) -> None:
    """Best-effort paired recovery save for the token-native loop."""
    if step < 1:
        print(f"{trainer.log_prefix} No completed YuE2 update to recover")
        return
    try:
        trainer.save_checkpoint(step, epoch)
    except Exception as exc:
        print(f"{trainer.log_prefix} YuE2 recovery checkpoint failed: {exc}")
        return
    try:
        trainer.save_optimizer_state(step)
    except Exception as exc:
        print(f"{trainer.log_prefix} YuE2 recovery optimizer state failed: {exc}")
    try:
        trainer.save_training_state(step, epoch, batch_idx)
    except Exception as exc:
        print(f"{trainer.log_prefix} YuE2 recovery training state failed: {exc}")
    if hasattr(trainer, "_cleanup_old_checkpoints"):
        try:
            trainer._cleanup_old_checkpoints(int(max_step_saves_to_keep or 0))
        except Exception as exc:
            print(f"{trainer.log_prefix} YuE2 recovery checkpoint cleanup failed: {exc}")
    if hasattr(trainer, "_cleanup_old_optimizer_states"):
        try:
            trainer._cleanup_old_optimizer_states(
                int(max_optimizer_saves_to_keep or 0), current_step=step)
        except Exception as exc:
            print(f"{trainer.log_prefix} YuE2 recovery optimizer cleanup failed: {exc}")


def train_abc_ar_loop(trainer, *, datasets, num_epochs=1, total_steps=None,
                      batch_size=1, save_every_n_steps=500,
                      gradient_accumulation_steps=1, max_grad_norm=1.0,
                      optimizer_type="adamw", lr_scheduler_type="constant",
                      progress_callback=None, update_total_steps_callback=None,
                      resume_from_checkpoint=None, max_step_saves_to_keep=10,
                      max_optimizer_saves_to_keep=1, **_unused):
    """Token-native ABC planner loop; it never enters image/latent cache code."""
    if batch_size < 1 or gradient_accumulation_steps < 1:
        raise ValueError("YuE2 batch size and gradient accumulation must be positive")
    if getattr(trainer, "trains_base_weights", False):
        if batch_size != 1:
            raise ValueError("YuE2 full_finetune requires physical batch_size=1")
        allowed = {"adamw8bit", "adamw8bit_ringbuffer", "lion8bit_ringbuffer", "adafactor"}
        if str(optimizer_type).lower() not in allowed:
            raise ValueError(
                f"YuE2 full_finetune requires a memory-bounded optimizer; got {optimizer_type!r}"
            )
    examples = prepare_abc_items(trainer, datasets)
    batches_per_epoch = math.ceil(len(examples) / batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / gradient_accumulation_steps)
    planned_steps = int(total_steps) if total_steps else int(num_epochs) * updates_per_epoch
    if planned_steps < 1:
        raise ValueError("YuE2 training requires at least one step")
    trainer._training_datasets = datasets
    trainer._dataset_fingerprint = _dataset_fingerprint(datasets)
    trainer._batches_per_epoch = batches_per_epoch
    trainer._grad_accum_steps = gradient_accumulation_steps
    trainer.setup_optimizer(optimizer_type, lr_scheduler_type, planned_steps)

    global_step = 0
    start_epoch = 0
    resume_batch = 0
    if resume_from_checkpoint:
        checkpoint = None
        if getattr(trainer, "trains_base_weights", False):
            loaded = Path(getattr(trainer, "_loaded_checkpoint_path", "") or "")
            if not loaded.is_file():
                raise ValueError("YuE2 full-finetune resume was not loaded as the base model")
            checkpoint = loaded
        elif str(resume_from_checkpoint).lower() == "latest":
            found = trainer.find_latest_checkpoint()
            if found is not None:
                checkpoint = Path(found[0])
        else:
            candidate = trainer.output_dir / str(resume_from_checkpoint)
            checkpoint = candidate if candidate.is_file() else Path(str(resume_from_checkpoint))
            if not checkpoint.is_file():
                raise ValueError(f"YuE2 resume checkpoint does not exist: {resume_from_checkpoint}")
        if checkpoint is not None:
            if getattr(trainer, "trains_base_weights", False):
                loaded = Path(getattr(trainer, "_loaded_checkpoint_path", "") or "")
                if not loaded.is_file() or loaded.resolve() != checkpoint.resolve():
                    raise ValueError(
                        "YuE2 full-finetune resume weights were not loaded as the base model"
                    )
                match = re.search(r"_step_(\d+)", loaded.name)
                if match is None:
                    raise ValueError(f"Cannot determine YuE2 resume step from {loaded.name}")
                global_step = int(match.group(1))
                from safetensors import safe_open
                with safe_open(loaded, framework="pt", device="cpu") as handle:
                    metadata = handle.metadata() or {}
                if metadata.get("yue2_training_scope") != "abc_ar_full" \
                        or metadata.get("yue2_training_protocol") != YUE2_TRAINING_PROTOCOL_VERSION:
                    raise ValueError("YuE2 full-finetune resume checkpoint has an incompatible contract")
            else:
                global_step = int(trainer.load_checkpoint(str(checkpoint)))
            state = trainer.load_training_state(global_step)
            trainer._fast_forward_lr_schedulers(global_step)
            optimizer_restored = trainer.load_optimizer_state(global_step)
            if not optimizer_restored:
                trainer._rearm_warmup_after_optimizer_reset(global_step)
            trainer._reassert_config_lr_on_resume()
            if state:
                start_epoch = int(state.get("epoch", 0))
                saved_fingerprint = state.get("dataset_fingerprint")
                if saved_fingerprint == trainer._dataset_fingerprint:
                    resume_batch = int(state.get("batch_idx", 0))
                    random.setstate(state["random_state"])
                else:
                    print(f"{trainer.log_prefix} YuE2 dataset changed; restarting "
                          f"epoch {start_epoch + 1} at batch 0 with checkpoint weights")
    if update_total_steps_callback:
        update_total_steps_callback(planned_steps)
    trainer.transformer.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    save_interval = max(0, int(save_every_n_steps or 0))
    last_saved = global_step
    accumulated = 0
    epoch = start_epoch
    completed_batch = resume_batch
    try:
        while global_step < planned_steps:
            trainer._current_epoch = epoch
            order = list(range(len(examples)))
            trainer._epoch_batch_rng_state = random.getstate()
            random.shuffle(order)
            first_item = min(len(order), resume_batch * batch_size)
            for batch_index in range(first_item, len(order), batch_size):
                if global_step >= planned_steps:
                    break
                trainer._check_stop_requested()
                rows = [examples[index] for index in order[batch_index:batch_index + batch_size]]
                batch = {key: value.to(trainer.device, non_blocking=True)
                         for key, value in collate_abc_ar(rows).items()}
                autocast_supported = (
                    trainer.device.type == "cuda"
                    or (trainer.device.type == "cpu"
                        and trainer.training_dtype == torch.bfloat16)
                )
                autocast = (torch.autocast(device_type=trainer.device.type,
                                            dtype=trainer.training_dtype)
                            if trainer.mixed_precision and autocast_supported
                            else contextlib.nullcontext())
                with autocast:
                    loss = train_abc_ar_step(trainer.transformer, batch)
                    scaled_loss = loss / gradient_accumulation_steps
                scaler = getattr(trainer, "scaler", None)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                accumulated += 1
                last_batch = batch_index + batch_size >= len(order)
                if accumulated < gradient_accumulation_steps and not last_batch:
                    continue
                parameters = [parameter for group in trainer.optimizer.param_groups
                              for parameter in group["params"] if parameter.grad is not None]
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(trainer.optimizer)
                if accumulated < gradient_accumulation_steps:
                    correction = gradient_accumulation_steps / accumulated
                    for parameter in parameters:
                        parameter.grad.mul_(correction)
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
                if scaler is not None and scaler.is_enabled():
                    scaler.step(trainer.optimizer)
                    scaler.update()
                else:
                    trainer.optimizer.step()
                trainer.optimizer.zero_grad(set_to_none=True)
                trainer.lr_scheduler.step()
                accumulated = 0
                global_step += 1
                completed_batch = batch_index // batch_size + 1
                loss_value = float(loss.detach())
                learning_rate = float(trainer.optimizer.param_groups[0]["lr"])
                if trainer.run_id is not None:
                    trainer._log_metrics_to_db(global_step, loss=loss_value,
                                               learning_rate=learning_rate,
                                               grad_norm=float(grad_norm))
                if progress_callback:
                    progress_callback(phase="training", step=global_step,
                                      total=planned_steps, epoch=epoch,
                                      loss=loss_value, lr=learning_rate)
                if save_interval and global_step % save_interval == 0:
                    trainer.save_checkpoint(global_step, epoch)
                    trainer.save_optimizer_state(global_step)
                    trainer.save_training_state(global_step, epoch,
                                                completed_batch)
                    if hasattr(trainer, "_cleanup_old_checkpoints"):
                        trainer._cleanup_old_checkpoints(int(max_step_saves_to_keep or 0))
                    if hasattr(trainer, "_cleanup_old_optimizer_states"):
                        trainer._cleanup_old_optimizer_states(
                            int(max_optimizer_saves_to_keep or 0), current_step=global_step)
                    last_saved = global_step
            resume_batch = 0
            epoch += 1
            completed_batch = 0
        if global_step and last_saved != global_step:
            trainer.save_checkpoint(global_step, max(0, epoch - 1))
            trainer.save_optimizer_state(global_step)
            trainer.save_training_state(global_step, max(0, epoch - 1), batches_per_epoch)
            if hasattr(trainer, "_cleanup_old_checkpoints"):
                trainer._cleanup_old_checkpoints(int(max_step_saves_to_keep or 0))
            if hasattr(trainer, "_cleanup_old_optimizer_states"):
                trainer._cleanup_old_optimizer_states(
                    int(max_optimizer_saves_to_keep or 0), current_step=global_step)
        return False
    except (KeyboardInterrupt, Exception):
        _save_recovery_bundle(
            trainer, step=global_step, epoch=epoch, batch_idx=completed_batch,
            max_step_saves_to_keep=max_step_saves_to_keep,
            max_optimizer_saves_to_keep=max_optimizer_saves_to_keep,
        )
        raise
    finally:
        trainer.transformer.eval()
        if trainer.run_id is not None and trainer._metrics_buffer:
            trainer._log_metrics_to_db(global_step, force_flush=True)
