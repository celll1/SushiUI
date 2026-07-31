"""VAE fine-tuner: decoder by default, encoder behind a double gate.

Standalone by design — see ``core/training/vae/__init__.py`` for why this does
not subclass ``BaseTrainer``. What it *does* reuse, unchanged, is everything
that hangs off the ``TrainingRun`` row: the subprocess launch, the
``.stop_training`` sentinel, the checkpoint list/resume routes, the
``TrainingMetrics.extra_metrics`` chart channel and the Training Monitor UI.

Recipe (design.md §5.1 as revised by §9.2, i.e. stabilityai/sd-vae-ft-mse's
published shape): MSE 1.0 + LPIPS-VGG 0.1 + YCbCr Charbonnier 0.1, encoder
frozen, EMA on, bf16 autocast over an fp32 master copy of the weights.

Encoder training (design.md §4) requires BOTH ``train_encoder`` and
``acknowledge_latent_space_break``. When it is on, the latent is sampled from
the posterior instead of taken at its mode, the KL term is constructed, the
export directory is named ``<run>_vae_encoder_trained`` instead of
``<run>_vae``, the sidecar records ``encoder_trained: true`` and the bare-LDM
export is refused.

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

from core.training.lr_utils import reassert_config_lr
from core.training.vae.vae_config import VaeConfigError, strict_bool
from core.training.vae.vae_dataset import VaeRawImageDataset, make_validation_batch
from core.training.vae import vae_losses

_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# Metric names emitted into TrainingMetrics.extra_metrics (registered for
# charting in core/training/metric_registry.py).
M_RECON = "vae_recon_loss"
M_LPIPS = "vae_lpips_loss"
M_DC = "vae_dc_loss"
M_PATTERN = "vae_pattern_loss"
M_INVENTED = "vae_invented_loss"
M_INVENTED_COV = "vae_invented_cov"
M_KL = "vae_kl_loss"
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

        # Encoder training is gated twice in vae_config (train_encoder AND
        # acknowledge_latent_space_break); the trainer re-reads BOTH rather than
        # trusting train_encoder alone, so a hand-built cfg that skipped the
        # resolver cannot reach the encoder path with one key set.
        #
        # strict_bool, NOT bool(): a quoted "false" out of a hand-written YAML is
        # truthy in Python, so a bare cast here would open the gate rather than
        # guard it. The same parser runs in vae_config._validate; this call is
        # what protects callers that bypassed the resolver.
        asked_for_encoder = strict_bool(cfg.get("train_encoder"), "train_encoder")
        acknowledged = strict_bool(cfg.get("acknowledge_latent_space_break"),
                                   "acknowledge_latent_space_break")
        cfg["train_encoder"] = asked_for_encoder
        cfg["acknowledge_latent_space_break"] = acknowledged
        cfg["train_decoder"] = strict_bool(cfg.get("train_decoder"), "train_decoder")
        cfg["export_bare_ldm"] = strict_bool(cfg.get("export_bare_ldm"),
                                             "export_bare_ldm")
        cfg["ema_enabled"] = strict_bool(cfg.get("ema_enabled"), "ema_enabled")
        self.train_encoder = asked_for_encoder and acknowledged
        if asked_for_encoder and not self.train_encoder:
            raise VaeConfigError(
                "train_encoder=true without acknowledge_latent_space_break=true. "
                "Both are required (see vae_config.resolve_vae_training_config)."
            )

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
            # A single file carries no config.json, so the scaling_factor on the
            # loaded config may be diffusers' fallback rather than this VAE's own.
            # Training never reads it, but save_pretrained BAKES it into every
            # export -- so it is repaired here, before anything is written.
            identity["scaling_factor_source"] = repair_single_file_scaling_factor(
                vae, path, self.cfg.get("vae_arch"), log_prefix=self.log_prefix)
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
        """Unfreeze exactly the requested subset of the VAE (design.md §4).

        The decoder and the encoder are selected by the same block-granularity
        mechanism (``decoder_blocks`` / ``encoder_blocks``); the ONLY asymmetry
        is which side-conv counts as part of the path: ``post_quant_conv`` for
        the decode side, ``quant_conv`` for the encode side.
        """
        self.trainable_params = []
        self.trainable_names = []
        targets: List = []

        if self.train_encoder:
            targets += self._encoder_targets(self.cfg["encoder_blocks"])
        if self.cfg["train_decoder"]:
            targets += self._decoder_targets(self.cfg["decoder_blocks"])

        for prefix, module in targets:
            for name, param in module.named_parameters():
                param.requires_grad_(True)
                self.trainable_params.append(param)
                self.trainable_names.append(f"{prefix}.{name}")

        if not self.trainable_params:
            raise VaeConfigError(
                f"decoder_blocks={self.cfg['decoder_blocks']!r} / "
                f"encoder_blocks={self.cfg['encoder_blocks']!r} selected 0 "
                f"parameters on {type(self.vae).__name__}."
            )
        total = sum(p.numel() for p in self.trainable_params)
        scope = (f"decoder_blocks={self.cfg['decoder_blocks']}"
                 if self.cfg["train_decoder"] else "decoder frozen")
        if self.train_encoder:
            scope += f", encoder_blocks={self.cfg['encoder_blocks']}"
        print(f"{self.log_prefix} Trainable: {len(self.trainable_params)} tensors "
              f"/ {total/1e6:.2f}M params ({scope})")

        # Sanity, in BOTH directions: a decoder-only run must leave the encoder
        # completely frozen (that is the entire latent-space contract), and an
        # encoder run must actually have unfrozen something in it.
        encoder = getattr(self.vae, "encoder", None)
        if encoder is not None:
            live = [n for n, p in encoder.named_parameters() if p.requires_grad]
            if live and not self.train_encoder:
                raise VaeConfigError(
                    f"Internal error: encoder parameters are trainable "
                    f"({live[:3]}...) with train_encoder=false. Refusing to run."
                )
            if self.train_encoder and not live:
                raise VaeConfigError(
                    f"train_encoder=true but encoder_blocks="
                    f"{self.cfg['encoder_blocks']!r} unfroze no encoder parameter "
                    f"on {type(self.vae).__name__}."
                )

        if self.train_encoder:
            print(f"{self.log_prefix} ENCODER TRAINING IS ACTIVE. The latent "
                  f"distribution this VAE produces will change, so cached "
                  f"latents, LoRAs and diffusion checkpoints built against the "
                  f"original VAE will no longer match the exported result. The "
                  f"export goes to '{self.run_name}{self._export_suffix()}' and "
                  f"its sidecar records encoder_trained=true.")

    def _decoder_targets(self, blocks: str) -> List:
        decoder = getattr(self.vae, "decoder", None)
        if decoder is None:
            raise VaeConfigError(
                f"The loaded VAE ({type(self.vae).__name__}) has no `.decoder` "
                f"submodule, so decoder training is not defined for it."
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
        return targets

    def _encoder_targets(self, blocks: str) -> List:
        encoder = getattr(self.vae, "encoder", None)
        if encoder is None:
            raise VaeConfigError(
                f"The loaded VAE ({type(self.vae).__name__}) has no `.encoder` "
                f"submodule, so encoder training is not defined for it."
            )
        targets = []
        if blocks == "all":
            targets.append(("encoder", encoder))
            # The encode-side mirror of post_quant_conv.
            qc = getattr(self.vae, "quant_conv", None)
            if qc is not None:
                targets.append(("quant_conv", qc))
        elif blocks == "down_blocks":
            targets.append(("encoder.down_blocks", encoder.down_blocks))
        elif blocks == "mid_block":
            targets.append(("encoder.mid_block", encoder.mid_block))
        elif blocks == "conv_out":
            # The encoder's conv_out produces the posterior parameters, so this
            # is the encode-side analogue of touching only the final projection.
            targets.append(("encoder.conv_out", encoder.conv_out))
            if getattr(encoder, "conv_norm_out", None) is not None:
                targets.append(("encoder.conv_norm_out", encoder.conv_norm_out))
            qc = getattr(self.vae, "quant_conv", None)
            if qc is not None:
                targets.append(("quant_conv", qc))
        else:  # unreachable: validated in vae_config
            raise VaeConfigError(f"Unknown encoder_blocks={blocks!r}")
        return targets

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
        self.loss_bank = vae_losses.VaeLossBank(
            self.cfg, self.device, kl_enabled=self.train_encoder)
        print(f"{self.log_prefix} Loss bank: {self.loss_bank.describe()}")
        if not self.train_encoder and float(self.cfg["kl_weight"]) > 0:
            print(f"{self.log_prefix} kl_weight={self.cfg['kl_weight']} is IGNORED: "
                  f"with the encoder frozen the posterior KL does not depend on "
                  f"any trainable parameter, so the term contributes no gradient "
                  f"and is not constructed.")

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
            train_items, self.cfg["resolution"], random_crop=True, seed=seed,
            scale_policy=self.cfg["crop_scale_policy"],
            max_downscale=self.cfg["crop_scale_max_downscale"])
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
            # The validation crop policy is PINNED to 'downscale' in
            # make_validation_batch and takes no parameter, so vae_val_psnr keeps
            # the same meaning whatever crop_scale_policy the run trains under.
            print(f"{self.log_prefix} Validation set: "
                  f"{self.val_batch.shape[0]} held-out image(s) @ "
                  f"{self.cfg['validation_resolution']}px "
                  f"(centre crop, scale policy pinned to 'downscale')")
        except Exception as e:
            print(f"{self.log_prefix} WARNING: no validation set ({e}); "
                  f"val PSNR/blockiness will not be charted. This is the only "
                  f"signal that a fine-tune is going wrong - fix the dataset.")

        # Resolved BEFORE the loop so an unresolvable explicit checkpoint fails
        # immediately, and so "latest" with no checkpoints yet starts fresh.
        resume_target = self.resolve_resume_target(self.cfg["resume_from"])
        if resume_target is not None:
            self.load_checkpoint(resume_target)

        total_steps = int(self.cfg["total_steps"])
        accum = int(self.cfg["gradient_accumulation_steps"])
        save_every = int(self.cfg["save_every"])
        val_every = int(self.cfg["validation_every"])

        crop_policy_note = self.cfg["crop_scale_policy"]
        if crop_policy_note == "mixed" and self.cfg["crop_scale_max_downscale"] > 0:
            crop_policy_note += f"(<={self.cfg['crop_scale_max_downscale']:g}x)"
        print(f"{self.log_prefix} Training: {total_steps} steps, "
              f"batch={self.cfg['batch_size']}x{accum}, "
              f"res={self.cfg['resolution']}, crop_scale={crop_policy_note}, "
              f"dtype={self.cfg['dtype']}")

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
                      # .6g, not .6f: the weighted KL contribution is ~1e-7 by
                      # construction (it sits at LDM's balance) and would print
                      # as a flat 0.000000 under a fixed-point format.
                      " ".join(f"{k}={v:.6g}" for k, v in parts.items()) +
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

        posterior = None
        with ctx:
            if self.train_encoder:
                # The encode forward now carries gradients, and the latent is
                # SAMPLED from the posterior rather than taken at its mode: the
                # KL term only constrains a distribution that is actually being
                # sampled, and a mode-only path would let the encoder shrink the
                # variance for free.
                posterior = self.vae.encode(pixels).latent_dist
                latent = posterior.sample()
            else:
                # Encoder is frozen: no_grad here is a genuine memory/compute
                # saving, NOT the base_trainer.encode_image no_grad that would
                # break training (the DECODE below is the one that must carry
                # gradients). The mode is used, deterministically, which is the
                # decoder-only ft-MSE shape.
                with torch.no_grad():
                    latent = self.vae.encode(pixels).latent_dist.mode()
            recon = self.vae.decode(latent).sample

        loss, parts = self.loss_bank(recon, pixels, posterior)

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

    def _sorted_checkpoint_dirs(self) -> List[Path]:
        """Existing ``checkpoints/step_*`` directories, oldest step first."""
        if not self.checkpoints_dir.is_dir():
            return []
        found = []
        for d in self.checkpoints_dir.glob("step_*"):
            if not d.is_dir():
                continue
            try:
                found.append((int(d.name.split("step_")[-1]), d))
            except ValueError:
                continue
        return [d for _, d in sorted(found, key=lambda t: t[0])]

    def resolve_resume_target(self, checkpoint: Optional[str]) -> Optional[Path]:
        """Resolve a ``resume_from`` value to a checkpoint directory, or None.

        ``"latest"`` (case-insensitive) selects the highest-numbered
        ``checkpoints/step_*`` directory, and resolves to None when the run has
        no checkpoints yet -- i.e. it starts fresh rather than erroring. That is
        the diffusion path's semantics (base_trainer.py:1150-1165: an unmatched
        "latest" simply leaves checkpoint_to_load as None).

        An EXPLICIT path or step name that does not exist still raises. The
        diffusion path silently starts fresh there too (base_trainer.py:1167-1177),
        but a named checkpoint is an unambiguous user intent, and silently
        restarting a 5,000-step run from zero is not a failure mode worth
        inheriting.
        """
        if not checkpoint:
            return None

        if str(checkpoint).strip().lower() == "latest":
            dirs = self._sorted_checkpoint_dirs()
            if not dirs:
                print(f"{self.log_prefix} resume_from='latest' but no checkpoints "
                      f"exist under {self.checkpoints_dir}; starting fresh.")
                return None
            print(f"{self.log_prefix} resume_from='latest' -> {dirs[-1].name}")
            return dirs[-1]

        ckpt_dir = Path(checkpoint)
        if ckpt_dir.is_dir():
            return ckpt_dir
        candidate = self.checkpoints_dir / str(checkpoint)
        if candidate.is_dir():
            return candidate
        available = [d.name for d in self._sorted_checkpoint_dirs()]
        raise VaeConfigError(
            f"resume checkpoint not found: {checkpoint!r} (looked at {ckpt_dir} "
            f"and {candidate}). Available: {available or 'none'}. Use 'latest' to "
            f"resume from the newest checkpoint."
        )

    def _assert_component_set_matches(self, ckpt_dir: Path) -> None:
        """Refuse a resume whose checkpoint trained a different component set.

        Named, actionable and BEFORE any weight load. Optimizer state, EMA state
        and the trainable-name list are all indexed by the component set, so a
        mismatch is never recoverable — it is only ever a config mistake.

        Two further keys are compared but only WARNED about, at the end: they do
        not invalidate the checkpoint, they invalidate the *comparability of the
        charts across the resume*, which nothing else detects. See
        ``_warn_measurement_changes``.
        """
        state_path = ckpt_dir / "train_state.json"
        if not state_path.is_file():
            print(f"{self.log_prefix} WARNING: {ckpt_dir} has no train_state.json; "
                  f"cannot verify that it trained the same components as this run.")
            return
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                saved = (json.load(f).get("config") or {})
        except Exception as e:
            print(f"{self.log_prefix} WARNING: could not read {state_path} ({e}); "
                  f"skipping the component-set check.")
            return
        if not saved:
            return

        def _saved_bool(key: str, default: bool) -> bool:
            value = saved.get(key, default)
            # train_state.json is written through _jsonable, which stringifies
            # anything exotic; be as strict here as the config gate is.
            try:
                return strict_bool(value, key)
            except VaeConfigError:
                return bool(default)

        mismatches = []
        was_encoder = _saved_bool("train_encoder", False)
        if was_encoder != self.train_encoder:
            mismatches.append(
                f"train_encoder: checkpoint={was_encoder}, this run={self.train_encoder}")
        if _saved_bool("train_decoder", True) != bool(self.cfg["train_decoder"]):
            mismatches.append(
                f"train_decoder: checkpoint={saved.get('train_decoder')}, "
                f"this run={self.cfg['train_decoder']}")
        if saved.get("decoder_blocks", self.cfg["decoder_blocks"]) != self.cfg["decoder_blocks"]:
            mismatches.append(
                f"decoder_blocks: checkpoint={saved.get('decoder_blocks')!r}, "
                f"this run={self.cfg['decoder_blocks']!r}")
        if was_encoder and self.train_encoder and \
                saved.get("encoder_blocks", self.cfg["encoder_blocks"]) != self.cfg["encoder_blocks"]:
            mismatches.append(
                f"encoder_blocks: checkpoint={saved.get('encoder_blocks')!r}, "
                f"this run={self.cfg['encoder_blocks']!r}")

        if mismatches:
            raise VaeConfigError(
                f"Checkpoint {ckpt_dir.name} trained a different component set "
                f"than this run: " + "; ".join(mismatches) + ". A checkpoint "
                f"stores exactly the parameters that were trainable when it was "
                f"written, together with optimizer and EMA state indexed by that "
                f"same set, so it can only be resumed by a run with the same "
                f"settings. Match them, or start a new run."
            )

        self._warn_measurement_changes(ckpt_dir, saved)

    def _warn_measurement_changes(self, ckpt_dir: Path, saved: Dict[str, Any]) -> None:
        """Warn (never refuse) when a resume changes WHAT IS MEASURED or TRAINED ON.

        ``validation_resolution`` and ``crop_scale_policy`` are both legitimate
        deliberate changes, so neither can be a refusal. But neither is detectable
        after the fact either: the resumed run appends to the SAME
        ``vae_val_psnr`` / ``vae_val_blockiness`` series, and because
        ``global_step != 0`` on a resume the trainer emits no fresh baseline point
        to separate the two regimes. The chart then shows a step that no model
        change caused — validation PSNR is strongly content-dependent here (the
        same fine-tune measures +1.15 dB on downscaled content and +0.81 dB on
        native), and that chart is the only quality signal this modality has.

        This is not hypothetical: ``validation_resolution``'s default moved from
        512 to 1024, so a hand-written ``process.vae`` that OMITS the key now
        resolves to a different value than it did when the checkpoint was written.
        (A run created through the UI or ``generate_vae_config`` pins every key in
        its own YAML and is unaffected.)
        """
        notes = []
        for key, what in (
            ("validation_resolution",
             "the held-out PSNR / blockiness series is computed on "
             "differently-sized centre crops"),
            ("crop_scale_policy",
             "the decoder is now trained on a different resampling "
             "distribution, which shifts what the same metric reports"),
        ):
            if key not in saved:
                continue
            before, now = saved.get(key), self.cfg.get(key)
            # str() both sides: train_state.json goes through _jsonable, which
            # may have stringified the value.
            if str(before) != str(now):
                notes.append(f"{key}: checkpoint={before!r} -> this run={now!r} "
                             f"({what})")

        if not notes:
            return
        print(f"{self.log_prefix} WARNING: resuming {ckpt_dir.name} with a "
              f"changed measurement/training basis:")
        for note in notes:
            print(f"{self.log_prefix}   - {note}")
        print(f"{self.log_prefix}   Points recorded BEFORE and AFTER this resume "
              f"are measured on different content and are NOT comparable; the "
              f"chart will show a step that no model change caused, and no fresh "
              f"baseline point is emitted on a resume. This is a warning, not a "
              f"refusal - changing either key is a legitimate deliberate act. To "
              f"keep one comparable series, restore the checkpoint's values; to "
              f"read a clean series under the new ones, start a new run.")

    def load_checkpoint(self, checkpoint):
        """Resume from a checkpoint directory, a run-relative step name, or the
        already-resolved Path from :meth:`resolve_resume_target`."""
        from safetensors.torch import load_file

        ckpt_dir = checkpoint if isinstance(checkpoint, Path) else \
            self.resolve_resume_target(str(checkpoint))
        if ckpt_dir is None:
            return

        # The component set must match BEFORE any weight is touched. Both
        # mismatch directions are silent failures otherwise:
        #   - a decoder-only checkpoint resumed with the encoder on is a SUBSET,
        #     so the tensor-name check below fires but blames decoder_blocks;
        #   - an encoder-trained checkpoint resumed decoder-only is a SUPERSET,
        #     so nothing fires at all here and the run continues with the
        #     encoder half of the checkpoint silently discarded, failing later
        #     (if at all) inside the optimizer state load.
        # train_state.json has recorded the resolved config since Phase 1.
        self._assert_component_set_matches(ckpt_dir)

        state = load_file(str(ckpt_dir / "vae_decoder.safetensors"))
        missing = [n for n in self.trainable_names if n not in state]
        if missing:
            raise VaeConfigError(
                f"Checkpoint {ckpt_dir} is missing {len(missing)} trainable "
                f"tensor(s) (e.g. {missing[:3]}). It was probably produced with "
                f"different decoder_blocks / encoder_blocks settings "
                f"(this run: decoder_blocks={self.cfg['decoder_blocks']!r}, "
                f"encoder_blocks={self.cfg['encoder_blocks']!r})."
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

        # Both restores above re-import the checkpoint's LR (see
        # core/training/lr_utils.py). Re-assert the configured one,
        # unconditionally and loudly, so a mid-run `train.lr` edit is honoured
        # AND the log always states which LR is actually running. A scalar
        # cfg_lr is broadcast over the (single) param group.
        reassert_config_lr(
            self.optimizer, self.lr_scheduler,
            float(self.cfg["learning_rate"]),
            log_prefix=self.log_prefix,
            component_names=("VAE",),
        )

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
    def _export_suffix(self) -> str:
        """Directory suffix for the exported VAE.

        An encoder fine-tune produces a DIFFERENT VAE, not an improved drop-in
        for the base model: latents encoded by it do not match the ones every
        existing cache / LoRA / diffusion checkpoint was built against. The
        artifact is therefore given a different name, so that a directory listing
        alone distinguishes the two cases and a `_vae` directory always means
        "same latent space as its base model". The sidecar's `encoder_trained`
        flag is the machine-readable form of the same fact; the name is the form
        a human sees first.
        """
        return "_vae_encoder_trained" if self.train_encoder else "_vae"

    def save_diffusers_vae(self, step: int) -> Path:
        """Write a diffusers VAE directory the EXISTING inference VAE-override
        path loads unchanged (pipeline.py:1294-1304).

        For a decoder-only run the compat gate
        (api/generation_overrides.py:334-403) passes because latent_channels /
        latent_ndim / class family / spatial scale are all unchanged, and
        ``save_pretrained`` preserves ``scaling_factor`` / ``shift_factor`` in
        config.json.

        Because that preservation is verbatim, the value written here is exactly
        whatever ``load_base_vae`` ended up with -- which is why the single-file
        branch repairs it at load (``repair_single_file_scaling_factor``) rather
        than here: nothing between the two can change it, and an export is the
        only place a wrong one becomes visible (and then only silently). The
        sidecar's ``base_vae.scaling_factor_source`` records how it was decided.

        An encoder-trained run passes the same structural gate — the latent
        SHAPE is unchanged — but its latent DISTRIBUTION is not the base model's
        any more. Nothing downstream can detect that, so it is marked in the
        directory name and in the sidecar instead.
        """
        out_dir = self.output_dir / f"{self.run_name}{self._export_suffix()}"
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
            # The sidecar's loss block is hand-listed rather than a cfg dump, so
            # a new term has to be added here to be recorded. Sub-parameters are
            # written as None when the term was off, because they were then not
            # read at all (same convention as kl_weight under a frozen encoder).
            "l_invented_weight": self.cfg["l_invented_weight"],
            **({
                "l_invented_y_weight": self.cfg["l_invented_y_weight"],
                "l_invented_chroma_weight": self.cfg["l_invented_chroma_weight"],
                "l_invented_flat_t_y": self.cfg["l_invented_flat_t_y"],
                "l_invented_flat_t_c": self.cfg["l_invented_flat_t_c"],
            } if float(self.cfg["l_invented_weight"]) > 0 else {
                "l_invented_y_weight": None,
                "l_invented_chroma_weight": None,
                "l_invented_flat_t_y": None,
                "l_invented_flat_t_c": None,
            }),
            # Recorded as None when the encoder was frozen, because the term was
            # then not constructed at all and the configured value had no effect.
            "kl_weight": (self.cfg["kl_weight"] if self.train_encoder else None),
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
                # NOT bumped for ADDITIVE fields, deliberately. The only reader
                # of this sidecar (api/routes.py::_training_vae_export_dirs ->
                # generation_overrides) looks up named keys and is already
                # required to treat a missing one as tri-state "unknown" rather
                # than as benign, so a new key cannot break it — whereas a bump
                # would tell a reader that something it knows how to parse has
                # changed shape, which is false, and would make every older
                # export look stale. Bump it when a field is REMOVED, RENAMED, or
                # changes meaning/units, i.e. when an existing reader would be
                # wrong rather than merely incomplete. Fields added since v1:
                # crop_scale_policy, crop_scale_max_downscale (2026-07-30).
                "format_version": 1,
                "produced_by": (
                    "SushiUI VAE fine-tune (network.type=vae_decoder"
                    + (", encoder trained" if self.train_encoder else
                       ", decoder only, encoder frozen") + ")"
                ),
                "run_id": self.run_id,
                "run_name": self.run_name,
                "step": step,
                "base_vae": self._base_vae_identity,
                "train_decoder": bool(self.cfg["train_decoder"]),
                "decoder_blocks": self.cfg["decoder_blocks"],
                # The machine-readable form of "this is not a drop-in
                # replacement for the base model's VAE".
                "encoder_trained": bool(self.train_encoder),
                "encoder_blocks": (self.cfg["encoder_blocks"]
                                   if self.train_encoder else None),
                "kl_weight": (float(self.cfg["kl_weight"])
                              if self.train_encoder else None),
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
                # WHICH pixels this decoder was calibrated on. Recorded because
                # nothing in the weights reveals it and it is the dominant
                # control on how the fine-tune behaves on native-resolution
                # content (results_crop_geometry.md §3).
                "crop_scale_policy": self.cfg["crop_scale_policy"],
                "crop_scale_max_downscale": (
                    float(self.cfg["crop_scale_max_downscale"])
                    if self.cfg["crop_scale_policy"] == "mixed" else None),
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
            noema_dir = self.output_dir / f"{self.run_name}{self._export_suffix()}_noema"
            _write(noema_dir, applied_ema=False)
            print(f"{self.log_prefix} Non-EMA (live weights) VAE written to {noema_dir}")
        else:
            _write(out_dir, applied_ema=False)
            print(f"{self.log_prefix} Fine-tuned VAE written to {out_dir} "
                  f"[live weights, EMA disabled]")

        print(f"{self.log_prefix} Load either directory via the VAE override in "
              f"the generation UI.")
        if self.train_encoder:
            print(f"{self.log_prefix} This VAE was trained WITH THE ENCODER. It "
                  f"encodes to a different latent distribution than the base "
                  f"VAE, so latent caches, LoRAs and diffusion checkpoints built "
                  f"against the base VAE do not match it. Cached latents made "
                  f"with the base VAE must be re-encoded before they are used "
                  f"with this one.")

        if self.cfg.get("export_bare_ldm"):
            try:
                self.save_bare_ldm_safetensors(out_dir)
            except VaeConfigError as e:
                # Never lose the diffusers export over an optional extra one.
                print(f"{self.log_prefix} bare-LDM export skipped: {e}")
        return out_dir

    def save_bare_ldm_safetensors(self, source_dir: Path) -> Path:
        """Write the exported VAE as a bare LDM-format ``.safetensors``.

        REFUSED whenever the encoder was trained. A bare ``.safetensors`` has no
        ``config.json``, so whatever loads it inherits ``scaling_factor`` /
        ``shift_factor`` from the model it is plugged into
        (``pipeline.py:1283-1290``). For a decoder-only fine-tune those values
        are still correct, because the encoder that defined them is byte-identical
        to the base model's. After an encoder fine-tune they are silently wrong
        and nothing downstream can tell. ``vae_config._validate`` refuses the
        combination before the run even starts; this is the second gate, on the
        write itself, so the refusal holds for any caller.
        """
        if self.train_encoder:
            raise VaeConfigError(
                "Refusing to write a bare LDM .safetensors for an "
                "encoder-trained VAE: the file carries no config.json, so the "
                "consumer would inherit scaling_factor / shift_factor from the "
                "model it is loaded into, and those are exactly what an encoder "
                "fine-tune invalidates. Use the diffusers directory export "
                f"({source_dir.name}), which carries its own config.json and the "
                "sushi_vae_training.json provenance sidecar."
            )

        cls_name = type(self.vae).__name__
        if cls_name != "AutoencoderKL":
            raise VaeConfigError(
                f"bare-LDM export is only defined for AutoencoderKL (the LDM key "
                f"mapping in adapters/state_dict_converter.py is that "
                f"architecture's); this run trained a {cls_name}. The diffusers "
                f"directory export is unaffected."
            )

        from core.training.adapters.state_dict_converter import (
            convert_vae_state_dict_to_original,
        )
        from safetensors.torch import load_file as _load_file, save_file

        # Read back what was actually written (i.e. the EMA weights when EMA is
        # applied), rather than re-deriving which copy is live.
        shard = source_dir / "diffusion_pytorch_model.safetensors"
        state = (_load_file(str(shard)) if shard.is_file()
                 else {k: v.detach().cpu() for k, v in self.vae.state_dict().items()})
        converted = convert_vae_state_dict_to_original(
            {k: v.to(torch.float32).contiguous() for k, v in state.items()})

        out_path = source_dir.parent / f"{source_dir.name}.safetensors"
        save_file(converted, str(out_path))
        print(f"{self.log_prefix} Bare LDM VAE written to {out_path} "
              f"({len(converted)} tensors). It carries no config.json: whatever "
              f"loads it supplies scaling_factor / shift_factor, which are "
              f"unchanged from the base VAE because the encoder was frozen.")
        return out_path

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
        if "l_invented" in parts:
            extra[M_INVENTED] = parts["l_invented"]
            # Window coverage rides along with the term: a value that falls
            # because the term stopped firing looks identical to one that falls
            # because the decoder stopped inventing, unless both are charted.
            extra[M_INVENTED_COV] = parts["l_invented_cov"]
        if "kl_term" in parts:
            # The WEIGHTED contribution, not the raw KL: see metric_registry.
            extra[M_KL] = parts["kl_term"]
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


# Key prefixes that make up a VAE-ONLY safetensors file. A file whose every
# tensor starts with one of these carries no backbone, hence no evidence of which
# model family it belongs to -- which is exactly when from_single_file's
# scaling_factor is a fallback rather than a reading.
_VAE_ONLY_KEY_PREFIXES = ("encoder.", "decoder.", "quant_conv.",
                          "post_quant_conv.", "loss.")


def _is_bare_vae_single_file(path: str) -> Optional[bool]:
    """True when ``path`` is a VAE-ONLY safetensors file, False when it also
    holds a backbone (a full checkpoint), None when it cannot be determined.

    Reads the safetensors header only -- no tensor data.
    """
    if not isinstance(path, str) or not path.lower().endswith(".safetensors"):
        return None  # .ckpt/.pt/.bin: not inspectable without unpickling
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
    except Exception as e:
        print(f"[VaeTrainer] Could not read safetensors header of {path}: {e}")
        return None
    if not keys:
        return None
    return all(k.startswith(_VAE_ONLY_KEY_PREFIXES) for k in keys)


def repair_single_file_scaling_factor(vae, path: str, vae_arch: Optional[str],
                                      log_prefix: str = "[VaeTrainer]") -> str:
    """Give a VAE loaded from a bare single file its architecture's own
    ``scaling_factor`` / ``shift_factor``, and say so out loud.

    ``AutoencoderKL.from_single_file`` cannot tell an SDXL VAE from an SD1.5 one
    -- the architectures are byte-for-byte the same shape -- so for a VAE-ONLY
    file it falls back to LDM's 0.18215. Training never reads the value, but
    ``save_pretrained`` writes it verbatim into the exported ``config.json``,
    and the inference-side VAE override trusts a directory's config.json. An
    SDXL export carrying 0.18215 is a silent 1.40x latent-scale error.

    The correct value is decided by ARCHITECTURE (``process.vae.vae_arch``,
    resolved through the shared VAE registry), not by guesswork, and is applied
    only when the file gave diffusers nothing to go on:

    * full checkpoint (backbone present) -> diffusers READ the family from the
      checkpoint; left untouched.
    * unknown / non-scalar ``vae_arch`` (flux2, qwen_image, a typo) -> nothing
      can be substituted honestly; left untouched, loudly.
    * ``vae_arch``'s latent_channels disagree with the loaded VAE -> the config
      is wrong about which VAE this is, so REFUSE rather than stamp a
      foreign family's number onto the export.

    Returns a short string recording HOW the effective value was arrived at; it
    is stored in the provenance sidecar so an export is self-describing.
    """
    from core.models.common.vae_store import (
        LDM_SINGLE_FILE_DEFAULT_SCALING_FACTOR,
        canonical_latent_scaling,
    )

    loaded = getattr(vae.config, "scaling_factor", None)
    loaded_shift = getattr(vae.config, "shift_factor", None)

    bare = _is_bare_vae_single_file(path)
    if bare is False:
        print(f"{log_prefix} scaling_factor={loaded} comes from the checkpoint "
              f"itself (the file carries a backbone, so from_single_file "
              f"identified the family); left as loaded.")
        return "from_single_file (full checkpoint, family identified)"

    arch = (vae_arch or "").strip()
    canonical = canonical_latent_scaling(arch) if arch else None
    if canonical is None or canonical[0] is None:
        why = (f"vae_arch={arch!r} is not a known VAE-store key"
               if canonical is None else
               f"vae_arch={arch!r} has no single scalar scaling_factor "
               f"(it normalises with latents_mean/latents_std)")
        print(f"{log_prefix} WARNING: this is a VAE-only file, so its "
              f"scaling_factor={loaded} is whatever from_single_file assumed "
              f"(its fallback is {LDM_SINGLE_FILE_DEFAULT_SCALING_FACTOR}), and "
              f"{why}, so it cannot be corrected. Every export of this run will "
              f"carry {loaded}. Set process.vae.vae_arch to the matching key "
              f"(sdxl / sd15 / flux1) if that is wrong.")
        return f"from_single_file fallback, UNVERIFIED (vae_arch={arch!r})"

    expected, expected_shift, expected_channels = canonical
    channels = getattr(vae.config, "latent_channels", None)
    if expected_channels is not None and channels is not None and \
            int(channels) != int(expected_channels):
        raise VaeConfigError(
            f"vae_arch={arch!r} expects a {expected_channels}-channel latent, "
            f"but the VAE loaded from {path} has {channels}. Refusing to run: "
            f"vae_arch is what decides the scaling_factor written into every "
            f"export of this run, and it does not describe this file. Set "
            f"process.vae.vae_arch to the architecture of the VAE in vae_path."
        )

    if loaded is not None and abs(float(loaded) - float(expected)) < 1e-9:
        print(f"{log_prefix} scaling_factor={loaded} already matches "
              f"vae_arch={arch!r}; unchanged.")
        return f"from_single_file, confirmed against vae_arch={arch!r}"

    kwargs = {"scaling_factor": float(expected)}
    if expected_shift is not None:
        kwargs["shift_factor"] = float(expected_shift)
    vae.register_to_config(**kwargs)
    print(f"{log_prefix} CORRECTED scaling_factor {loaded} -> {expected} "
          f"(and shift_factor {loaded_shift} -> {kwargs.get('shift_factor', loaded_shift)}) "
          f"for vae_arch={arch!r}. A VAE-only file carries no config.json, so "
          f"from_single_file had to guess; SDXL and SD1.5 VAEs are "
          f"indistinguishable by their weights, and only vae_arch can tell them "
          f"apart. This value is baked into every export by save_pretrained. If "
          f"vae_arch is wrong for this file, stop the run and fix it.")
    return f"corrected from {loaded} to {expected} via vae_arch={arch!r}"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
