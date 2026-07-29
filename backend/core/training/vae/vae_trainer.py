"""Decoder-only VAE fine-tuner (Phase 1).

Standalone by design — see ``core/training/vae/__init__.py`` for why this does
not subclass ``BaseTrainer``. What it *does* reuse, unchanged, is everything
that hangs off the ``TrainingRun`` row: the subprocess launch, the
``.stop_training`` sentinel, the checkpoint list/resume routes, the
``TrainingMetrics.extra_metrics`` chart channel and the Training Monitor UI.

Recipe (design.md §5.1 as revised by §9.2, i.e. stabilityai/sd-vae-ft-mse's
published shape): MSE 1.0 + LPIPS-VGG 0.1 + YCbCr Charbonnier 0.1, encoder
frozen, EMA on, bf16 autocast over an fp32 master copy of the weights.

Precision: the VAE weights are held in **fp32** (they ARE the optimizer's master
copy) and the forward runs under ``torch.autocast`` in the configured compute
dtype. fp16 is HARD-REFUSED for the SDXL VAE family — the documented activation
overflow is the entire reason ``madebyollin/sdxl-vae-fp16-fix`` exists, and it
is a training-time NaN as much as an inference-time one.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.training.vae.vae_config import VaeConfigError
from core.training.vae.vae_dataset import VaeRawImageDataset, make_validation_batch
from core.training.vae import vae_losses

_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# Metric names emitted into TrainingMetrics.extra_metrics (registered for
# charting in core/training/metric_registry.py).
M_RECON = "vae_recon_loss"
M_LPIPS = "vae_lpips_loss"
M_DC = "vae_dc_loss"
M_PATTERN = "vae_pattern_loss"
M_VAL_PSNR = "vae_val_psnr"
M_VAL_BLOCKINESS = "vae_val_blockiness"


class VaeTrainer:
    """Decoder-only fine-tune of an AutoencoderKL-family VAE."""

    log_prefix = "[VaeTrainer]"

    def __init__(
        self,
        cfg: Dict[str, Any],
        *,
        output_dir: str,
        run_name: str,
        run_id: Optional[int] = None,
        progress_callback=None,
    ):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.run_id = run_id
        self.progress_callback = progress_callback

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_dtype = _DTYPE_MAP[cfg["dtype"]]

        self.vae = None
        self.trainable_params: List[torch.nn.Parameter] = []
        self.trainable_names: List[str] = []
        self.optimizer = None
        self.lr_scheduler = None
        self.loss_bank = None
        self.ema: Optional[Dict[str, torch.Tensor]] = None
        self._ema_updates = 0
        self._ema_retained_init = 1.0

        self.resume_seq = 0
        self.global_step = 0
        self.stopped = False
        self._last_val_step = -1
        self._last_ckpt_step = -1
        self._base_vae_identity: Dict[str, Any] = {}

        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.samples_dir = self.output_dir / "samples"

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def load_base_vae(self):
        """Load the VAE to be fine-tuned, in fp32 (the optimizer master copy)."""
        from diffusers import AutoencoderKL

        source = self.cfg["vae_source"]
        path = str(self.cfg["vae_path"])

        if source == "store":
            from core.models.common.vae_store import resolve_vae_dir
            resolved = resolve_vae_dir(str(self.cfg["vae_arch"]))
            if not resolved:
                raise VaeConfigError(
                    f"vae_source='store' but vae_store could not resolve a "
                    f"directory for vae_arch={self.cfg['vae_arch']!r}"
                )
            path = resolved

        cfg_dir = _find_vae_config_dir(path)
        if cfg_dir is not None:
            cls_name = _read_json(os.path.join(cfg_dir, "config.json")).get(
                "_class_name") or "AutoencoderKL"
            import diffusers
            vae_cls = getattr(diffusers, cls_name, AutoencoderKL)
            print(f"{self.log_prefix} Loading {cls_name} from {cfg_dir} (fp32 master)")
            vae = vae_cls.from_pretrained(cfg_dir, torch_dtype=torch.float32)
            identity = {"format": "diffusers_dir", "path": cfg_dir, "class": cls_name}
        elif os.path.isfile(path):
            print(f"{self.log_prefix} Loading AutoencoderKL from single file {path} "
                  f"(fp32 master)")
            vae = AutoencoderKL.from_single_file(path, torch_dtype=torch.float32)
            identity = {"format": "single_file", "path": path, "class": "AutoencoderKL"}
        else:
            raise VaeConfigError(
                f"No loadable VAE at {path!r} (expected a diffusers directory "
                f"containing config.json, or a .safetensors file)."
            )

        vae.to(self.device, dtype=torch.float32)
        vae.requires_grad_(False)
        vae.eval()  # keeps any dropout/BN deterministic; grads still flow
        self.vae = vae

        identity["latent_channels"] = int(getattr(vae.config, "latent_channels", -1))
        identity["scaling_factor"] = float(getattr(vae.config, "scaling_factor", 1.0) or 1.0)
        identity["shift_factor"] = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
        self._base_vae_identity = identity

        if "fp16-fix" in str(path).replace("\\", "/").lower() or "fp16_fix" in str(path).lower():
            print(f"{self.log_prefix} WARNING: the base VAE looks like an "
                  f"fp16-fix checkpoint. That model's fp16 safety comes from a "
                  f"weight/bias rescaling that fine-tuning does not preserve; "
                  f"the result may no longer be safe to run in fp16.")

        print(f"{self.log_prefix} Base VAE: {identity}")

    def select_trainable(self):
        """Unfreeze exactly the requested decoder subset (design.md §4)."""
        blocks = self.cfg["decoder_blocks"]
        decoder = getattr(self.vae, "decoder", None)
        if decoder is None:
            raise VaeConfigError(
                f"The loaded VAE ({type(self.vae).__name__}) has no `.decoder` "
                f"submodule, so decoder-only training is not defined for it."
            )

        targets = []
        if blocks == "all":
            targets.append(("decoder", decoder))
            # post_quant_conv is part of the decode path (latent -> decoder input);
            # "decoder-only" in the ft-MSE sense includes it. Absent on some
            # AutoencoderKL variants, hence the guard.
            pqc = getattr(self.vae, "post_quant_conv", None)
            if pqc is not None:
                targets.append(("post_quant_conv", pqc))
        elif blocks == "up_blocks":
            targets.append(("decoder.up_blocks", decoder.up_blocks))
        elif blocks == "mid_block":
            targets.append(("decoder.mid_block", decoder.mid_block))
        elif blocks == "conv_out":
            targets.append(("decoder.conv_out", decoder.conv_out))
            if getattr(decoder, "conv_norm_out", None) is not None:
                targets.append(("decoder.conv_norm_out", decoder.conv_norm_out))
        else:  # unreachable: validated in vae_config
            raise VaeConfigError(f"Unknown decoder_blocks={blocks!r}")

        self.trainable_params = []
        self.trainable_names = []
        for prefix, module in targets:
            for name, param in module.named_parameters():
                param.requires_grad_(True)
                self.trainable_params.append(param)
                self.trainable_names.append(f"{prefix}.{name}")

        if not self.trainable_params:
            raise VaeConfigError(
                f"decoder_blocks={blocks!r} selected 0 parameters on "
                f"{type(self.vae).__name__}."
            )
        total = sum(p.numel() for p in self.trainable_params)
        print(f"{self.log_prefix} Trainable: {len(self.trainable_params)} tensors "
              f"/ {total/1e6:.2f}M params (decoder_blocks={blocks})")

        # Sanity: the encoder must be completely frozen in v1.
        encoder = getattr(self.vae, "encoder", None)
        if encoder is not None:
            leaked = [n for n, p in encoder.named_parameters() if p.requires_grad]
            if leaked:
                raise VaeConfigError(
                    f"Internal error: encoder parameters are trainable "
                    f"({leaked[:3]}...). Refusing to run."
                )

    def build_optimizer(self):
        from core.training.optimizer_factory import OptimizerFactory

        self.optimizer = OptimizerFactory.create_optimizer(
            optimizer_type=str(self.cfg["optimizer"]),
            params=self.trainable_params,
            learning_rate=self.cfg["learning_rate"],
            weight_decay=self.cfg["optimizer_weight_decay"],
        )

        try:
            from diffusers.optimization import get_scheduler
            self.lr_scheduler = get_scheduler(
                str(self.cfg["lr_scheduler"]),
                optimizer=self.optimizer,
                num_warmup_steps=int(self.cfg["lr_warmup_steps"]),
                num_training_steps=int(self.cfg["total_steps"]),
            )
        except Exception as e:
            print(f"{self.log_prefix} LR scheduler "
                  f"{self.cfg['lr_scheduler']!r} unavailable ({e}); using constant LR")
            self.lr_scheduler = None

    def build_losses(self):
        self.loss_bank = vae_losses.VaeLossBank(self.cfg, self.device)
        print(f"{self.log_prefix} Loss bank: {self.loss_bank.describe()}")

    def init_ema(self):
        if not self.cfg["ema_enabled"]:
            print(f"{self.log_prefix} EMA disabled. Note: both ft-EMA and PiD use "
                  f"EMA; a short fine-tune without it tends to carve noise into "
                  f"the decoder rather than converge.")
            self.ema = None
            return
        self.ema = {
            name: param.detach().clone().float()
            for name, param in zip(self.trainable_names, self.trainable_params)
        }
        self._ema_updates = 0
        self._ema_retained_init = 1.0
        print(f"{self.log_prefix} EMA enabled (target decay={self.cfg['ema_decay']}, "
              f"warmup-ramped)")

    def _ema_decay_at(self, update_index: int) -> float:
        """Warmup-ramped decay: ``min(target, (1 + n) / (10 + n))``.

        A bare ``mul_(d).add_(p, 1-d)`` retains ``d**N`` of the ORIGINAL weights
        after N updates. At the default 0.999 x 2000 steps that is 13.5% of the
        base VAE still in the exported file; at 0.9999 it is 82%, i.e. the
        fine-tune is effectively invisible; a run stopped at step 300 exports
        ~97% base weights. The standard warmup ramp (Adam-style bias correction
        expressed as a decay schedule) makes the early updates nearly
        unaveraged, so the EMA tracks the live weights until enough history
        exists for the target decay to be meaningful.
        """
        return min(float(self.cfg["ema_decay"]), (1.0 + update_index) / (10.0 + update_index))

    @torch.no_grad()
    def _update_ema(self):
        if self.ema is None:
            return
        d = self._ema_decay_at(self._ema_updates)
        for name, param in zip(self.trainable_names, self.trainable_params):
            self.ema[name].mul_(d).add_(param.detach().float(), alpha=1.0 - d)
        self._ema_updates += 1
        # Fraction of the ORIGINAL (base-VAE) weights still present in the EMA.
        # Tracked as a running product so it survives resume and is reported at
        # save time -- an over-damped EMA can otherwise hide a no-op export
        # behind healthy-looking loss/PSNR curves (those are measured on the
        # LIVE weights, not on the EMA that actually gets saved).
        self._ema_retained_init *= d

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    def train(self, dataset_items: List[Dict]) -> bool:
        """Run the fine-tune. Returns True if it was stopped by the user."""
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        seed = int(self.cfg["seed"])
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)

        self._detect_resume_seq()
        self.load_base_vae()
        self.select_trainable()
        self.build_optimizer()
        self.build_losses()
        self.init_ema()

        train_items, val_items = self._split_items(dataset_items)
        train_dataset = VaeRawImageDataset(
            train_items, self.cfg["resolution"], random_crop=True, seed=seed)
        loader = DataLoader(
            train_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=self.cfg["num_workers"],
            drop_last=len(train_dataset) >= self.cfg["batch_size"],
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=self.cfg["num_workers"] > 0,
        )

        self.val_batch = None
        try:
            self.val_batch = make_validation_batch(
                val_items, self.cfg["validation_resolution"],
                self.cfg["validation_num_images"])
            print(f"{self.log_prefix} Validation set: "
                  f"{self.val_batch.shape[0]} held-out image(s) @ "
                  f"{self.cfg['validation_resolution']}px")
        except Exception as e:
            print(f"{self.log_prefix} WARNING: no validation set ({e}); "
                  f"val PSNR/blockiness will not be charted. This is the only "
                  f"signal that a fine-tune is going wrong - fix the dataset.")

        if self.cfg["resume_from"]:
            self.load_checkpoint(str(self.cfg["resume_from"]))

        total_steps = int(self.cfg["total_steps"])
        accum = int(self.cfg["gradient_accumulation_steps"])
        save_every = int(self.cfg["save_every"])
        val_every = int(self.cfg["validation_every"])

        print(f"{self.log_prefix} Training: {total_steps} steps, "
              f"batch={self.cfg['batch_size']}x{accum}, "
              f"res={self.cfg['resolution']}, dtype={self.cfg['dtype']}")

        # Baseline validation so the chart has a "before" point.
        if self.val_batch is not None and self.global_step == 0:
            self._run_validation(step=0)

        stop_flag = self.output_dir / ".stop_training"
        micro = 0
        t0 = time.time()
        loader_iter = iter(loader)
        self.optimizer.zero_grad(set_to_none=True)

        while self.global_step < total_steps:
            # ---- stop sentinel, checked EVERY step --------------------
            if stop_flag.is_file():
                print(f"{self.log_prefix} Stop flag detected at step "
                      f"{self.global_step}; saving and exiting.")
                try:
                    stop_flag.unlink()
                except OSError:
                    pass
                self.stopped = True
                break

            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            loss, parts = self._train_micro_step(batch, accum)
            micro += 1
            if micro % accum != 0:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.trainable_params, self.cfg["max_grad_norm"])
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._update_ema()
            self.global_step += 1

            lr = self.optimizer.param_groups[0]["lr"]
            self._log_step(self.global_step, loss, parts, lr, float(grad_norm))

            if self.global_step % 10 == 0 or self.global_step <= 3:
                rate = self.global_step / max(time.time() - t0, 1e-6)
                print(f"{self.log_prefix} step {self.global_step}/{total_steps} "
                      f"loss={loss:.6f} " +
                      " ".join(f"{k}={v:.6f}" for k, v in parts.items()) +
                      f" lr={lr:.2e} ({rate:.2f} it/s)")

            if val_every > 0 and self.global_step % val_every == 0:
                self._run_validation(self.global_step)

            if save_every > 0 and self.global_step % save_every == 0:
                self.save_checkpoint(self.global_step)

        # Final validation + artifacts. Skipped when the periodic hooks already
        # ran at this exact step (so the last step is not measured/saved twice).
        if self.val_batch is not None and self._last_val_step != self.global_step:
            self._run_validation(self.global_step)
        if self._last_ckpt_step != self.global_step:
            self.save_checkpoint(self.global_step, final=True)
        self.save_diffusers_vae(self.global_step)
        self._flush_metrics()
        return self.stopped

    def _train_micro_step(self, batch: torch.Tensor, accum: int):
        pixels = batch.to(self.device, dtype=torch.float32, non_blocking=True)

        use_autocast = (self.device.type == "cuda"
                        and self.compute_dtype is not torch.float32)
        ctx = (torch.autocast(device_type="cuda", dtype=self.compute_dtype)
               if use_autocast else _NullCtx())

        with ctx:
            # Encoder is frozen: no_grad here is a genuine memory/compute saving,
            # NOT the base_trainer.encode_image no_grad that would break training
            # (the DECODE below is the one that must carry gradients).
            with torch.no_grad():
                latent = self.vae.encode(pixels).latent_dist.mode()
            recon = self.vae.decode(latent).sample

        loss, parts = self.loss_bank(recon, pixels)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"{self.log_prefix} Non-finite loss at step {self.global_step} "
                f"({loss.item()}); components={parts}. Aborting rather than "
                f"writing a corrupt checkpoint."
            )

        (loss / accum).backward()
        return float(loss.detach()), parts

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run_validation(self, step: int):
        if self.val_batch is None:
            return
        psnrs, blocks = [], []
        chunk = max(1, min(2, self.val_batch.shape[0]))
        use_autocast = (self.device.type == "cuda"
                        and self.compute_dtype is not torch.float32)
        for i in range(0, self.val_batch.shape[0], chunk):
            x = self.val_batch[i:i + chunk].to(self.device, dtype=torch.float32)
            ctx = (torch.autocast(device_type="cuda", dtype=self.compute_dtype)
                   if use_autocast else _NullCtx())
            with ctx:
                z = self.vae.encode(x).latent_dist.mode()
                y = self.vae.decode(z).sample
            psnrs.append(vae_losses.psnr(y, x))
            b = vae_losses.blockiness(y, x)
            if b == b:  # not NaN
                blocks.append(b)
        val_psnr = sum(psnrs) / len(psnrs) if psnrs else float("nan")
        val_block = sum(blocks) / len(blocks) if blocks else float("nan")
        self._last_val_step = step
        print(f"{self.log_prefix} [validation] step={step} "
              f"psnr={val_psnr:.3f}dB blockiness={val_block:.4f}")
        self._queue_metrics(step, extra={M_VAL_PSNR: val_psnr,
                                         M_VAL_BLOCKINESS: val_block})

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _trainable_state_dict(self, use_ema: bool = False) -> Dict[str, torch.Tensor]:
        if use_ema and self.ema is not None:
            return {k: v.detach().cpu().clone() for k, v in self.ema.items()}
        return {name: param.detach().cpu().clone()
                for name, param in zip(self.trainable_names, self.trainable_params)}

    def save_checkpoint(self, step: int, final: bool = False) -> Path:
        from safetensors.torch import save_file

        self._last_ckpt_step = step
        ckpt_dir = self.checkpoints_dir / f"step_{step:08d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        save_file(self._trainable_state_dict(), str(ckpt_dir / "vae_decoder.safetensors"))
        if self.ema is not None:
            save_file(self._trainable_state_dict(use_ema=True),
                      str(ckpt_dir / "ema.safetensors"))
        torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), ckpt_dir / "lr_scheduler.pt")
        torch.save(
            {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
            },
            ckpt_dir / "rng_state.pt",
        )
        with open(ckpt_dir / "train_state.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "step": step,
                    "run_name": self.run_name,
                    "run_id": self.run_id,
                    "final": bool(final),
                    "network_type": "vae_decoder",
                    "trainable_names": self.trainable_names,
                    "ema_enabled": self.ema is not None,
                    "ema_updates": self._ema_updates,
                    "ema_retained_init_fraction": self._ema_retained_init,
                    "resume_seq": self.resume_seq,
                    "base_vae": self._base_vae_identity,
                    "config": _jsonable(self.cfg),
                    "saved_at": datetime.utcnow().isoformat(),
                },
                f, indent=2,
            )
        print(f"{self.log_prefix} Checkpoint saved: {ckpt_dir}")
        self._record_checkpoint_row(ckpt_dir, step)
        self._prune_checkpoints()
        return ckpt_dir

    def load_checkpoint(self, checkpoint: str):
        """Resume from a checkpoint directory (or a run-relative step name)."""
        from safetensors.torch import load_file

        ckpt_dir = Path(checkpoint)
        if not ckpt_dir.is_dir():
            candidate = self.checkpoints_dir / checkpoint
            if candidate.is_dir():
                ckpt_dir = candidate
            else:
                raise VaeConfigError(
                    f"resume checkpoint not found: {checkpoint!r} (looked at "
                    f"{ckpt_dir} and {candidate})"
                )

        state = load_file(str(ckpt_dir / "vae_decoder.safetensors"))
        missing = [n for n in self.trainable_names if n not in state]
        if missing:
            raise VaeConfigError(
                f"Checkpoint {ckpt_dir} is missing {len(missing)} trainable "
                f"tensor(s) (e.g. {missing[:3]}). It was probably produced with a "
                f"different decoder_blocks setting."
            )
        with torch.no_grad():
            for name, param in zip(self.trainable_names, self.trainable_params):
                param.copy_(state[name].to(param.device, dtype=param.dtype))

        ema_path = ckpt_dir / "ema.safetensors"
        if self.ema is not None and ema_path.is_file():
            ema_state = load_file(str(ema_path))
            ema_missing = [n for n in self.trainable_names if n not in ema_state]
            if ema_missing:
                # A PARTIAL restore is worse than no restore: _update_ema()
                # indexes every trainable name, so a short dict would KeyError on
                # the first step after resume. Re-seed instead.
                print(f"{self.log_prefix} WARNING: ema.safetensors in {ckpt_dir} "
                      f"is missing {len(ema_missing)} of {len(self.trainable_names)} "
                      f"tensor(s) (e.g. {ema_missing[:3]}); re-seeding EMA from the "
                      f"restored weights.")
                self.init_ema()
            else:
                self.ema = {k: ema_state[k].float().to(self.device)
                            for k in self.trainable_names}
        elif self.ema is not None:
            print(f"{self.log_prefix} WARNING: no ema.safetensors in {ckpt_dir}; "
                  f"re-seeding EMA from the restored weights.")
            self.init_ema()

        opt_path = ckpt_dir / "optimizer.pt"
        if opt_path.is_file():
            self.optimizer.load_state_dict(
                torch.load(opt_path, map_location=self.device, weights_only=False))
        sched_path = ckpt_dir / "lr_scheduler.pt"
        if self.lr_scheduler is not None and sched_path.is_file():
            self.lr_scheduler.load_state_dict(
                torch.load(sched_path, map_location="cpu", weights_only=False))

        rng_path = ckpt_dir / "rng_state.pt"
        if rng_path.is_file():
            try:
                rng = torch.load(rng_path, map_location="cpu", weights_only=False)
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
                if rng.get("cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(
                        [s.cpu().to(torch.uint8) for s in rng["cuda"]])
            except Exception as e:
                print(f"{self.log_prefix} RNG restore failed (non-fatal): {e}")

        with open(ckpt_dir / "train_state.json", "r", encoding="utf-8") as f:
            train_state = json.load(f)
        self.global_step = int(train_state.get("step", 0))
        # Continue the EMA warmup ramp / retained-init product across resume
        # (only meaningful when the EMA itself was restored, not re-seeded).
        if self.ema is not None and train_state.get("ema_enabled"):
            self._ema_updates = int(train_state.get("ema_updates", self.global_step))
            self._ema_retained_init = float(
                train_state.get("ema_retained_init_fraction", 1.0))
        print(f"{self.log_prefix} Resumed from {ckpt_dir} at step {self.global_step}")

    def _prune_checkpoints(self):
        keep = int(self.cfg["max_step_saves_to_keep"])
        if keep <= 0:
            return
        dirs = sorted((d for d in self.checkpoints_dir.glob("step_*") if d.is_dir()),
                      key=lambda d: d.name)
        for stale in dirs[:-keep]:
            try:
                shutil.rmtree(stale)
                print(f"{self.log_prefix} Pruned old checkpoint {stale.name}")
                self._delete_checkpoint_row(stale)
            except Exception as e:
                print(f"{self.log_prefix} Could not prune {stale}: {e}")

    # ------------------------------------------------------------------
    # Final artifact
    # ------------------------------------------------------------------
    def save_diffusers_vae(self, step: int) -> Path:
        """Write a diffusers VAE directory the EXISTING inference VAE-override
        path loads unchanged (pipeline.py:1294-1304).

        The compat gate (api/generation_overrides.py:334-403) passes because
        latent_channels / latent_ndim / class family / spatial scale are all
        unchanged by a decoder-only fine-tune, and ``save_pretrained`` preserves
        ``scaling_factor`` / ``shift_factor`` in config.json.
        """
        out_dir = self.output_dir / f"{self.run_name}_vae"
        losses = {
            "mse_weight": self.cfg["mse_weight"],
            "l1_weight": self.cfg["l1_weight"],
            "lpips_weight": self.cfg["lpips_weight"],
            "lpips_net": self.cfg["lpips_net"],
            "ycbcr_dc_weight": self.cfg["ycbcr_dc_weight"],
            "ycbcr_dc_y_weight": self.cfg["ycbcr_dc_y_weight"],
            "ycbcr_dc_chroma_weight": self.cfg["ycbcr_dc_chroma_weight"],
            "ycbcr_dc_eps": self.cfg["ycbcr_dc_eps"],
            "pattern_weight": self.cfg["pattern_weight"],
            "pattern_size": self.cfg["pattern_size"],
        }

        def _write(target: Path, applied_ema: bool):
            target.mkdir(parents=True, exist_ok=True)
            backup = None
            if applied_ema:
                backup = self._trainable_state_dict()
                with torch.no_grad():
                    for name, param in zip(self.trainable_names, self.trainable_params):
                        param.copy_(self.ema[name].to(param.device, dtype=param.dtype))
            try:
                self.vae.save_pretrained(str(target))
            finally:
                if backup is not None:
                    with torch.no_grad():
                        for nm, prm in zip(self.trainable_names, self.trainable_params):
                            prm.copy_(backup[nm].to(prm.device, dtype=prm.dtype))
            sidecar = {
                "format_version": 1,
                "produced_by": "SushiUI VAE decoder fine-tune (network.type=vae_decoder)",
                "run_id": self.run_id,
                "run_name": self.run_name,
                "step": step,
                "base_vae": self._base_vae_identity,
                "decoder_blocks": self.cfg["decoder_blocks"],
                "ema_applied": applied_ema,
                "ema_target_decay": self.cfg["ema_decay"] if applied_ema else None,
                "ema_updates": self._ema_updates if applied_ema else None,
                # Fraction of the ORIGINAL base-VAE weights still present in
                # what was written. Close to 1.0 means the export is
                # substantially the base VAE, not the fine-tune.
                "ema_retained_init_fraction": (self._ema_retained_init
                                               if applied_ema else 0.0),
                "dtype": self.cfg["dtype"],
                "resolution": self.cfg["resolution"],
                "losses": losses,
                "saved_at": datetime.utcnow().isoformat(),
            }
            with open(target / "sushi_vae_training.json", "w", encoding="utf-8") as f:
                json.dump(sidecar, f, indent=2)

        if self.ema is not None:
            retained = self._ema_retained_init
            _write(out_dir, applied_ema=True)
            print(f"{self.log_prefix} Fine-tuned VAE written to {out_dir} "
                  f"[EMA, {self._ema_updates} updates, retained base-VAE weight "
                  f"fraction = {retained:.4g} ({retained*100:.2f}%)]")
            if retained > 0.5:
                # Loud, because the loss/PSNR/blockiness charts are all measured
                # on the LIVE weights and cannot show this.
                print(f"{self.log_prefix} WARNING: the EMA export still retains "
                      f"{retained*100:.1f}% of the ORIGINAL base VAE (too few "
                      f"updates for ema_decay={self.cfg['ema_decay']}). Prefer the "
                      f"non-EMA sibling directory below, lower ema_decay, or train "
                      f"for more steps.")
            # Always write the non-EMA weights too: on a short or user-stopped
            # run the EMA copy can be dominated by the base VAE, and this sibling
            # is the only usable artifact in that case.
            noema_dir = self.output_dir / f"{self.run_name}_vae_noema"
            _write(noema_dir, applied_ema=False)
            print(f"{self.log_prefix} Non-EMA (live weights) VAE written to {noema_dir}")
        else:
            _write(out_dir, applied_ema=False)
            print(f"{self.log_prefix} Fine-tuned VAE written to {out_dir} "
                  f"[live weights, EMA disabled]")

        print(f"{self.log_prefix} Load either directory via the VAE override in "
              f"the generation UI.")
        return out_dir

    # ------------------------------------------------------------------
    # DB plumbing (TrainingRun-based; no new tables)
    # ------------------------------------------------------------------
    def _split_items(self, items: List[Dict]):
        n_val = int(self.cfg["validation_num_images"])
        if len(items) <= n_val + 1:
            # Too small to hold anything back — validate on the training images
            # and say so, rather than silently producing an empty train set.
            print(f"{self.log_prefix} WARNING: only {len(items)} item(s); the "
                  f"validation images overlap the training set.")
            return items, items
        return items[:-n_val], items[-n_val:]

    def _log_step(self, step: int, loss: float, parts: Dict[str, float],
                  lr: float, grad_norm: float):
        extra = {"lr": lr}
        recon = parts.get("mse", 0.0) + parts.get("l1", 0.0)
        if "mse" in parts or "l1" in parts:
            extra[M_RECON] = recon
        if "lpips" in parts:
            extra[M_LPIPS] = parts["lpips"]
        if "ycbcr_dc" in parts:
            extra[M_DC] = parts["ycbcr_dc"]
        if "pattern" in parts:
            extra[M_PATTERN] = parts["pattern"]
        self._queue_metrics(step, loss=loss, recon_loss=recon,
                            learning_rate=lr, grad_norm=grad_norm, extra=extra)
        if self.progress_callback is not None:
            try:
                self.progress_callback("training", step,
                                       int(self.cfg["total_steps"]), 0, loss, lr)
            except Exception as e:
                print(f"{self.log_prefix} progress_callback failed: {e}")

    def _detect_resume_seq(self):
        """0 for a fresh run, one past the highest recorded seq when resuming.

        Same convention as base_trainer.py:8501-8519: the global step counter
        continues, so (run_id, step) stays unique; resume_seq only labels which
        session a row came from, which is what draws the resume boundary on the
        loss chart.
        """
        if self.run_id is None:
            return
        try:
            from database import get_training_db
            from database.models import TrainingMetrics
            from sqlalchemy import func as _sqlfunc
            db = next(get_training_db())
            try:
                max_seq = (db.query(_sqlfunc.max(TrainingMetrics.resume_seq))
                           .filter(TrainingMetrics.run_id == self.run_id).scalar())
            finally:
                db.close()
            self.resume_seq = (int(max_seq) + 1) if max_seq is not None else 0
        except Exception as e:
            print(f"{self.log_prefix} resume_seq detection failed ({e}); using 0")
            self.resume_seq = 0
        print(f"{self.log_prefix} Metrics resume_seq = {self.resume_seq}")

    _METRICS_FLUSH_EVERY = 20

    def _queue_metrics(self, step: int, *, loss: float = None,
                       recon_loss: float = None, learning_rate: float = None,
                       grad_norm: float = None, extra: Dict[str, float] = None):
        if not hasattr(self, "_metrics_buffer"):
            self._metrics_buffer: Dict[int, Dict[str, Any]] = {}
        entry = self._metrics_buffer.setdefault(step, {"step": step, "extra": {}})
        if loss is not None:
            entry["loss"] = loss
        if recon_loss is not None:
            entry["recon_loss"] = recon_loss
        if learning_rate is not None:
            entry["learning_rate"] = learning_rate
        if grad_norm is not None:
            entry["grad_norm"] = grad_norm
        for k, v in (extra or {}).items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):  # SQLite JSON1 rejects the NaN token
                entry["extra"][k] = fv
        if len(self._metrics_buffer) >= self._METRICS_FLUSH_EVERY:
            self._flush_metrics()

    def _flush_metrics(self):
        buffer = getattr(self, "_metrics_buffer", None)
        if not buffer or self.run_id is None:
            if buffer:
                buffer.clear()
            return
        try:
            from database import get_training_db
            from database.models import TrainingMetrics
            db = next(get_training_db())
            try:
                for step, entry in sorted(buffer.items()):
                    row = (db.query(TrainingMetrics)
                           .filter(TrainingMetrics.run_id == self.run_id,
                                   TrainingMetrics.step == step)
                           .first())
                    if row is None:
                        row = TrainingMetrics(run_id=self.run_id, step=step, epoch=0,
                                              resume_seq=self.resume_seq)
                        db.add(row)
                    else:
                        row.resume_seq = self.resume_seq
                    for column in ("loss", "recon_loss", "learning_rate", "grad_norm"):
                        if entry.get(column) is not None:
                            setattr(row, column, entry[column])
                    if entry["extra"]:
                        merged = dict(row.extra_metrics or {})
                        merged.update(entry["extra"])
                        row.extra_metrics = merged
                db.commit()
            finally:
                db.close()
        except Exception as e:
            print(f"{self.log_prefix} metrics flush failed (non-fatal): "
                  f"{type(e).__name__}: {e}")
        buffer.clear()

    def _record_checkpoint_row(self, ckpt_dir: Path, step: int):
        if self.run_id is None:
            return
        try:
            from database import get_training_db
            from database.models import TrainingCheckpoint
            size = sum(f.stat().st_size for f in ckpt_dir.glob("*") if f.is_file())
            db = next(get_training_db())
            try:
                existing = (db.query(TrainingCheckpoint)
                            .filter(TrainingCheckpoint.run_id == self.run_id,
                                    TrainingCheckpoint.step == step)
                            .first())
                if existing is None:
                    db.add(TrainingCheckpoint(
                        run_id=self.run_id,
                        checkpoint_name=ckpt_dir.name,
                        step=step,
                        epoch=0,
                        file_path=str(ckpt_dir),
                        file_size=size,
                    ))
                else:
                    existing.file_path = str(ckpt_dir)
                    existing.file_size = size
                db.commit()
            finally:
                db.close()
        except Exception as e:
            print(f"{self.log_prefix} checkpoint DB row failed (non-fatal): {e}")

    def _delete_checkpoint_row(self, ckpt_dir: Path):
        if self.run_id is None:
            return
        try:
            from database import get_training_db
            from database.models import TrainingCheckpoint
            db = next(get_training_db())
            try:
                (db.query(TrainingCheckpoint)
                 .filter(TrainingCheckpoint.run_id == self.run_id,
                         TrainingCheckpoint.checkpoint_name == ckpt_dir.name)
                 .delete())
                db.commit()
            finally:
                db.close()
        except Exception as e:
            print(f"{self.log_prefix} checkpoint row delete failed (non-fatal): {e}")

    def cleanup(self):
        self.vae = None
        self.loss_bank = None
        self.optimizer = None
        self.ema = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _find_vae_config_dir(path: str) -> Optional[str]:
    """Return the directory holding a diffusers VAE ``config.json``, or None.

    Re-implemented locally (a handful of lines) rather than importing
    ``api.generation_overrides._vae_config_dir``: nothing under
    ``core/training/`` should import the API package at module scope.
    """
    if not path or not os.path.isdir(path):
        return None
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    nested = os.path.join(path, "vae")
    if os.path.isfile(os.path.join(nested, "config.json")):
        return nested
    return None


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
