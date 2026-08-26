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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.training.lr_utils import reassert_config_lr
from core.training.vae.vae_config import VaeConfigError, strict_bool
from core.training.vae.vae_dataset import (
    VaeEpochCropSampler,
    VaeRawImageDataset,
    make_validation_batch,
)
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

# Algorithm tag of the frozen-weight fingerprint written into
# ``train_state.json`` / the export sidecar (see
# ``VaeTrainer._compute_frozen_fingerprint``). It is stored ALONGSIDE the digest
# so that changing how the digest is computed makes old and new values
# *incomparable* (-> warn) rather than *different* (-> refuse).
_FROZEN_FP_ALGO = "blake2b16-fp32-v1"

# Every file ``save_checkpoint`` writes beside ``train_state.json``. The list is
# shared by the writer (which records the size of each one it produced) and the
# resume guard (which verifies them); adding a new checkpoint artifact means
# adding it here, so that a resume can never silently ignore it.
# ``VaeTrainer._assert_checkpoint_complete`` documents the tier each one is
# verified at.
_CKPT_ARTIFACTS = (
    "vae_decoder.safetensors",
    "ema.safetensors",
    "optimizer.pt",
    "lr_scheduler.pt",
    "rng_state.pt",
)

# The subset of _CKPT_ARTIFACTS that save_checkpoint writes CONDITIONALLY: only
# when the run has an EMA / an LR scheduler. On a checkpoint that carries the
# artifact manifest, absence is proven to be "never written". Without a manifest
# it is genuinely ambiguous ("never written" vs "lost"), and that ambiguity must
# not be resolved as damage — the LR-scheduler fallback in build_optimizer
# ("... unavailable; using constant LR") writes exactly such a checkpoint, and
# refusing it would leave no way to resume once the cause is fixed.
_CKPT_CONDITIONAL = ("ema.safetensors", "lr_scheduler.pt")

# Cross-optimizer resume is deliberately an allow-list. These names are the
# public VAE config values, not Python class names. Every allowed pair must also
# have a lossless-enough state conversion in optimizer_state_convert.py.
_COMPATIBLE_OPTIMIZER_RESUME_PAIRS = {
    ("adamw", "adamw8bit"),
}

_OPTIMIZER_CLASS_TO_CONFIG_NAME = {
    "AdamW": "adamw",
    "AdamW8bit": "adamw8bit",
    "PagedAdamW": "paged_adamw",
    "PagedAdamW8bit": "paged_adamw8bit",
    "Adafactor": "adafactor",
    "Lion8bit": "lion8bit",
    "PagedLion8bit": "paged_lion8bit",
    "AdamW8bit_RingBuffer": "adamw8bit_ringbuffer",
    "Lion8bit_RingBuffer": "lion8bit_ringbuffer",
}


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
        # Set in train(); carries the data-pass counter that keys the crop RNG,
        # and is checkpointed so a resume continues into a FRESH pass.
        self.train_sampler = None
        self.global_step = 0
        # Backward-pass census. A run that completed none has the base VAE's
        # weights, so it neither reports success nor writes them back out.
        self._backwards_completed = 0
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

        self._record_frozen_fingerprint()

        if self.train_encoder:
            print(f"{self.log_prefix} ENCODER TRAINING IS ACTIVE. The latent "
                  f"distribution this VAE produces will change, so cached "
                  f"latents, LoRAs and diffusion checkpoints built against the "
                  f"original VAE will no longer match the exported result. The "
                  f"export goes to '{self.run_name}{self._export_suffix()}' and "
                  f"its sidecar records encoder_trained=true.")

    # ------------------------------------------------------------------
    # Base-VAE identity
    # ------------------------------------------------------------------
    def _compute_frozen_fingerprint(self) -> Optional[Dict[str, Any]]:
        """Digest the tensors a resume does NOT restore, i.e. the frozen half.

        This is the exact invariant a resume needs. ``load_checkpoint`` overwrites
        every *trainable* tensor with the checkpoint's copy, so those weights are
        fully determined by the checkpoint and their initial value is irrelevant.
        Everything else -- the frozen encoder of a decoder-only run, the decoder
        blocks outside ``decoder_blocks``, the quant convs -- comes from whatever
        base VAE THIS run loaded. If that half differs from the one the checkpoint
        was written against, the resumed model is a hybrid that no file records.

        Hashing precisely that half is also why the check does not misfire on the
        legitimate uses of a *different path*: the same weights loaded from a moved
        drive, from a differently-spelled path, or from a single file instead of a
        diffusers directory all produce the same digest. Conversely, restarting
        from an EXPORT of the same run (frozen half identical, trained half
        different) is correctly accepted -- the checkpoint overwrites the trained
        half anyway, so the resulting model is identical either way. A digest over
        the whole model would refuse that legitimate case.

        Tensors are cast to fp32 before hashing so that the digest does not depend
        on the *container* dtype: the same values held as fp16 and as fp32 hash
        alike. It does NOT make the digest indifferent to a file that was actually
        ROUNDED to fp16 — those values differ, so the digest differs and the resume
        is refused. That is the correct verdict (the frozen half really would be
        different weights), but it means "the fp16 copy of the same VAE" is a
        different base VAE as far as this check is concerned.

        Returns None when there is no model to hash (e.g. a trainer built for a
        unit test), which callers treat as "unknown", never as "mismatch".
        """
        vae = getattr(self, "vae", None)
        if vae is None:
            return None
        try:
            import hashlib

            digest = hashlib.blake2b(digest_size=16)
            count = 0
            tensors = list(vae.named_parameters()) + list(vae.named_buffers())
            for name, tensor in sorted(tensors, key=lambda kv: kv[0]):
                if getattr(tensor, "requires_grad", False):
                    continue  # restored from the checkpoint; not part of the base
                digest.update(name.encode("utf-8"))
                digest.update(str(tuple(tensor.shape)).encode("utf-8"))
                digest.update(
                    tensor.detach().to("cpu", torch.float32).contiguous()
                    .numpy().tobytes())
                count += 1
            if not count:
                return None
            return {"algo": _FROZEN_FP_ALGO,
                    "digest": digest.hexdigest(),
                    "tensor_count": count}
        except Exception as e:
            # A fingerprint is a safety net, never a reason a run cannot start.
            print(f"{self.log_prefix} WARNING: could not fingerprint the frozen "
                  f"base weights ({e}); a resume will fall back to comparing the "
                  f"recorded base-VAE path and structure only.")
            return None

    def _record_frozen_fingerprint(self) -> None:
        """Attach the frozen-weight fingerprint to the base-VAE identity.

        Computed once, here, because the frozen tensors never change during a run
        and because the trainable set (which defines the complement) is only known
        after ``select_trainable``. The value then rides along into every
        ``train_state.json`` and into the export sidecar for free.
        """
        fp = self._compute_frozen_fingerprint()
        if fp is None:
            return
        self._base_vae_identity["frozen_fingerprint"] = fp
        print(f"{self.log_prefix} Frozen base weights: {fp['tensor_count']} "
              f"tensor(s), fingerprint {fp['algo']}:{fp['digest']}")

    @staticmethod
    def _normalized_path(value: Any) -> str:
        """A path in a form that ignores separator/case/relative-form spelling.

        Deliberately does NOT resolve symlinks or drive mappings: this value only
        ever feeds a *warning*, and the authoritative identity check is the
        frozen-weight fingerprint.
        """
        if not value:
            return ""
        text = str(value)
        try:
            text = os.path.abspath(text)
        except Exception:
            pass
        text = text.replace("\\", "/").rstrip("/")
        if os.name == "nt":
            text = text.lower()
        return text

    def _assert_base_vae_matches(self, ckpt_dir: Path,
                                 saved: Optional[Dict[str, Any]]) -> None:
        """Refuse a resume that would splice this run's base VAE with another's.

        Runs AFTER the component-set check, because the fingerprint compared here
        covers the complement of the trainable set and is only meaningful once
        both sides are known to have trained the same components.

        Tiers, by how conclusive the recorded evidence is:

        * **frozen-weight fingerprint** -- conclusive in both directions. Equal
          digests prove the untouched half is bit-identical whatever anything
          else says; different digests prove a hybrid. **Refusal.**
        * **structure** (``class`` / ``latent_channels``) -- read off the loaded
          model rather than off a user string, so a mismatch is real. **Refusal**
          — *unless* the digests are equal, in which case the weights are proven
          bit-identical and the difference can only be in how the model is
          described (a diffusers release renaming ``_class_name``, or starting to
          report a ``latent_channels`` attribute that used to be absent and was
          recorded as -1). Refusing there would stop a long run over a library
          upgrade, so it is demoted to a warning. Structure is what protects
          checkpoints written before the fingerprint existed.
        * **``scaling_factor`` / ``shift_factor``** -- never a hybrid (training
          reads neither), but not spelling either: ``save_pretrained`` bakes them
          into every export and the sidecar/inference override path reads them.
          A silent change across a resume changes what the run finally writes, so
          it is **always warned about**, digests equal or not.
        * **path / format** -- suggestive only. The same VAE legitimately has
          several spellings (moved drive, relative vs absolute, diffusers
          directory vs single file). **Warning**, and only when there is no
          comparable fingerprint to settle the question.
        """
        # getattr, not attribute access: like _position_data_sampler, this runs on
        # __new__-built trainers in backend/tests/test_lr_resume_override.py, and
        # a resume must not die on an attribute __init__ never got to set.
        current = getattr(self, "_base_vae_identity", None) or {}
        if not current:
            # No identity recorded for this run (no model loaded yet — unit-test
            # harnesses drive this method directly). Nothing to compare against.
            return
        if not saved:
            print(f"{self.log_prefix} WARNING: checkpoint {ckpt_dir.name} records "
                  f"no base VAE, so it cannot be verified to have been trained on "
                  f"this run's base VAE ({current.get('path')!r}). Resuming it on "
                  f"a DIFFERENT base VAE would silently produce a hybrid model. "
                  f"Proceeding - the checkpoint predates base-VAE recording.")
            return

        old_fp = saved.get("frozen_fingerprint") or {}
        new_fp = current.get("frozen_fingerprint") or {}
        comparable_fp = bool(old_fp) and bool(new_fp) and \
            old_fp.get("algo") == new_fp.get("algo")
        digests_equal = comparable_fp and old_fp.get("digest") == new_fp.get("digest")

        structure: List[str] = []
        for key, label in (("class", "VAE class"),
                           ("latent_channels", "latent_channels")):
            before, now = saved.get(key), current.get(key)
            if before is None or now is None:
                continue
            if str(before) != str(now):
                structure.append(f"{label}: checkpoint={before!r}, this run={now!r}")

        fatal: List[str] = []
        if comparable_fp and not digests_equal:
            fatal.append(
                f"frozen base weights: checkpoint="
                f"{old_fp.get('digest')} ({old_fp.get('tensor_count')} tensors), "
                f"this run={new_fp.get('digest')} "
                f"({new_fp.get('tensor_count')} tensors)")
        if structure and not digests_equal:
            # Proven-identical weights outrank a structural label; see the
            # docstring. Without a digest to appeal to, structure IS the evidence.
            fatal.extend(structure)

        if fatal:
            raise VaeConfigError(
                f"Checkpoint {ckpt_dir.name} was trained on a DIFFERENT base VAE "
                f"than this run loaded: " + "; ".join(fatal) + ". "
                f"Checkpoint base: {saved.get('path')!r} "
                f"({saved.get('format')}); this run: {current.get('path')!r} "
                f"({current.get('format')}). A checkpoint stores only the "
                f"parameters that were trainable; the rest of the model comes "
                f"from the base VAE loaded now, so resuming across a base change "
                f"would produce a hybrid of the two that no file describes and "
                f"nothing later detects. Point vae_path / vae_source / vae_arch "
                f"back at the base this run's checkpoints were written against, "
                f"or start a new run against the new base."
            )

        notes: List[str] = []
        if structure:  # only reachable when digests_equal
            notes.extend(
                f"{item} (the frozen weights are bit-identical, so this is a "
                f"description/library difference, not a different model)"
                for item in structure)
        # ALWAYS compared, digests equal or not: neither factor is read during
        # training, but save_pretrained bakes both into every export and the
        # sidecar / inference VAE-override path reads them back, so a change here
        # silently changes what this run writes at the end.
        for key in ("scaling_factor", "shift_factor"):
            before, now = saved.get(key), current.get(key)
            if before is None or now is None:
                continue
            try:
                changed = float(before) != float(now)
            except (TypeError, ValueError):
                changed = str(before) != str(now)
            if changed:
                notes.append(f"{key}: checkpoint={before!r} -> this run={now!r} "
                             f"(baked into the exported config.json and the "
                             f"provenance sidecar)")
        if not comparable_fp:
            if self._normalized_path(saved.get("path")) != \
                    self._normalized_path(current.get("path")):
                notes.append(f"path: checkpoint={saved.get('path')!r} -> this run="
                             f"{current.get('path')!r}")
            if saved.get("format") and saved.get("format") != current.get("format"):
                notes.append(f"format: checkpoint={saved.get('format')!r} -> this "
                             f"run={current.get('format')!r}")
        if not notes:
            return
        print(f"{self.log_prefix} WARNING: resuming {ckpt_dir.name} with a "
              f"base VAE that is described differently:")
        for note in notes:
            print(f"{self.log_prefix}   - {note}")
        if comparable_fp:
            print(f"{self.log_prefix}   The frozen-weight fingerprints are equal, "
                  f"so the model this resume trains is the one the checkpoint was "
                  f"written against; no hybrid is possible and the run continues. "
                  f"A changed scaling_factor / shift_factor still ends up in the "
                  f"exported config.json, so check that the value above is the one "
                  f"this VAE should ship with.")
            return
        why = ("this checkpoint predates the frozen-weight fingerprint"
               if not old_fp else
               "the fingerprints were computed by different algorithms")
        print(f"{self.log_prefix}   {why}, so whether the weights are the same "
              f"file cannot be decided here. The same VAE moved, renamed, or "
              f"loaded as a single file instead of a diffusers directory is "
              f"expected to look like this and is harmless. A genuinely different "
              f"base VAE is NOT: only the trainable tensors come from the "
              f"checkpoint, so the rest of the model would come from the new base. "
              f"Verify before letting this run continue.")

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

    @staticmethod
    def optimizer_placement_note(optimizer_type: str) -> Optional[str]:
        """What the run actually gets, when the optimizer's NAME says otherwise.

        ``adamw8bit_ringbuffer`` / ``lion8bit_ringbuffer`` are accepted because
        they really run here (verified with a live ``step()``): ``OptimizerFactory``
        passes ``get_state_buffer=None`` and both implementations take their
        "Ring Buffer disabled: GPU allocation" branch. But the ring-buffer
        residency their name promises is exactly what this trainer does not wire
        up, so the run says what it is doing rather than letting the name speak
        for it. Returns None for every other optimizer.
        """
        from core.training.vae.vae_config import RINGBUFFER_OPTIMIZERS
        name = str(optimizer_type).strip().lower()
        if name not in RINGBUFFER_OPTIMIZERS:
            return None
        plain = "adamw8bit" if name.startswith("adamw") else "lion8bit"
        return (f"optimizer={optimizer_type}: this trainer passes no "
                f"ring-buffer allocator, so the 8-bit optimizer state is "
                f"allocated on the GPU - the same state placement as "
                f"{plain}. The ring-buffer part of the name does not apply "
                f"here; {plain} is the unambiguous spelling of what this run "
                f"gets. Its cautious / Schedule-Free / stochastic-rounding "
                f"options are not part of the VAE config: build_optimizer "
                f"passes optimizer, params, lr and weight_decay only, and "
                f"vae_config refuses the other optimizer_* keys instead of "
                f"accepting them.")

    def build_optimizer(self):
        from core.training.optimizer_factory import OptimizerFactory

        optimizer_type = str(self.cfg["optimizer"])
        note = self.optimizer_placement_note(optimizer_type)
        if note:
            print(f"{self.log_prefix} {note}")

        self.optimizer = OptimizerFactory.create_optimizer(
            optimizer_type=optimizer_type,
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

        self._backwards_completed = 0

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
        # Custom sampler instead of shuffle=True: it yields (index, visit) pairs
        # so that re-visiting an image in a later pass moves its crop window and
        # re-draws its 'mixed' scale. See VaeEpochCropSampler for why the counter
        # cannot live on the dataset (worker processes hold their own copies).
        self.train_sampler = VaeEpochCropSampler(
            len(train_dataset), seed=seed, shuffle=True)
        loader = DataLoader(
            train_dataset,
            batch_size=self.cfg["batch_size"],
            sampler=self.train_sampler,
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

            grad_norm = self._clip_gradients()
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

        return self._finalize(total_steps)

    def _finalize(self, total_steps: int) -> bool:
        """Write the run's final artifacts, or refuse when nothing was trained.

        ``global_step == 0`` means no optimizer step ran this run or an
        earlier one, so the in-memory weights equal the base VAE's already on
        disk. The exception arm below is an invariant assertion rather than a
        reachable path: ``vae_config`` pins ``total_steps >= 1``, so the only
        way to reach ``global_step == 0`` here is via the stop break, which
        takes the ``self.stopped`` arm first.
        """
        if self.global_step == 0:
            self._flush_metrics()
            if self.stopped:
                print(f"{self.log_prefix} Stopped before the first optimizer "
                      f"step; no checkpoint or export written (the weights are "
                      f"the base VAE's, unchanged).")
                return self.stopped
            from core.training.base_trainer import NothingTrainedError
            raise NothingTrainedError(
                f"VAE fine-tune completed no optimizer step over the "
                f"{total_steps} step(s) it was asked for, so its weights are the "
                f"base VAE's. No checkpoint or export was written. Check the "
                f"dataset: every image it produced was unreadable, or it "
                f"produced none."
            )

        # Final validation + artifacts. Skipped when the periodic hooks already
        # ran at this exact step (so the last step is not measured/saved twice).
        if self.val_batch is not None and self._last_val_step != self.global_step:
            self._run_validation(self.global_step)
        if self._last_ckpt_step != self.global_step:
            self.save_checkpoint(self.global_step, final=True)
        self.save_diffusers_vae(self.global_step)
        self._flush_metrics()
        return self.stopped

    def _clip_gradients(self) -> torch.Tensor:
        """Clip to ``max_grad_norm``, where **0 means "do not clip"**.

        That is the convention the rest of this repository already uses for the
        same key (``base_trainer`` and ``optimizers/fused_optimizer_groups``
        both guard their clip with ``if max_grad_norm > 0``, the latter
        documenting it as "0 to disable"), and it is the only reading under
        which the UI's ``min=0`` input is safe.

        Passing 0 straight to ``clip_grad_norm_`` does NOT disable clipping: the
        scale factor is ``max_norm / (total_norm + 1e-6)``, i.e. 0, so every
        gradient becomes exactly 0. The optimizer step is then a no-op except
        for AdamW's decoupled weight decay, which keeps shrinking the weights —
        so the run reports success, charts a flat loss and exports a VAE that
        was decayed rather than trained. ``vae_config`` refuses a NEGATIVE bound
        (which would negate the gradients); 0 is given the meaning it has
        everywhere else instead of being refused.

        The unclipped total norm is still computed and returned in both
        branches, because it is charted as ``grad_norm`` and is the only signal
        that a run is about to diverge.
        """
        max_norm = float(self.cfg["max_grad_norm"])
        if max_norm > 0:
            return torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm)
        grads = [p.grad for p in self.trainable_params if p.grad is not None]
        if not grads:
            return torch.zeros(())
        return torch.norm(torch.stack([g.detach().norm(2) for g in grads]), 2)

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
        self._backwards_completed += 1
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
        optimizer_state = self.optimizer.state_dict()
        # State key layouts are implementation-specific and some foreign dicts
        # pass load_state_dict only to fail at the first step. Record the writer
        # class so resume conversion can identify the source before loading.
        optimizer_state["_sushi_opt_class"] = type(self.optimizer).__name__
        torch.save(optimizer_state, ckpt_dir / "optimizer.pt")
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
        # train_state.json is written LAST and carries the manifest of everything
        # written before it, so it doubles as the "this checkpoint is complete"
        # marker: a save interrupted part-way leaves no manifest to check against,
        # and a directory copied part-way is caught by the recorded sizes.
        with open(ckpt_dir / "train_state.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "step": step,
                    "artifacts": self._checkpoint_manifest(ckpt_dir),
                    "run_name": self.run_name,
                    "run_id": self.run_id,
                    "final": bool(final),
                    "network_type": "vae_decoder",
                    "trainable_names": self.trainable_names,
                    "ema_enabled": self.ema is not None,
                    "ema_updates": self._ema_updates,
                    "ema_retained_init_fraction": self._ema_retained_init,
                    # Data-pass counter of VaeEpochCropSampler: the crop RNG is
                    # keyed by it, so without it a resume would replay the pass-0
                    # crops and undo the per-visit variation.
                    "data_epoch": self._data_epoch_for_checkpoint(),
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

    def _checkpoint_manifest(self, ckpt_dir: Path) -> Dict[str, int]:
        """Name -> byte size of every artifact this checkpoint actually wrote.

        Sizes, not hashes: the point is to catch a file that was cut short (an
        interrupted save, a copy that was still running, a full disk), and a
        truncated file is a shorter file. Hashing 400 MB of optimizer state on
        every save and every resume would cost far more than the failure mode is
        worth, and would still not detect the case sizes miss (a same-length
        corruption), which `torch.load` / `safetensors` catch anyway when the
        file is actually read.

        The recorded key set is also the record of what the writing run HAD:
        a checkpoint written without an LR scheduler simply has no
        ``lr_scheduler.pt`` entry, which is how a resume tells "never written"
        apart from "lost".
        """
        sizes: Dict[str, int] = {}
        for name in _CKPT_ARTIFACTS:
            path = ckpt_dir / name
            if path.is_file():
                sizes[name] = int(path.stat().st_size)
        return sizes

    def _artifact_status(self, ckpt_dir: Path, name: str,
                         manifest: Optional[Dict[str, Any]]) -> Tuple[str, str]:
        """Classify one checkpoint artifact.

        Returns ``(status, detail)`` where status is one of:

        * ``"ok"``           — present, and the size the manifest recorded.
        * ``"missing"``      — it is established that the file was written and
                               this directory has not got it (interrupted save,
                               partial copy, deletion).
        * ``"size_mismatch"``— present, but not the size it was saved at (or
                               empty), i.e. the file that is there is not the
                               file that was saved.
        * ``"not_written"``  — the manifest proves the writing run never wrote it
                               (e.g. no LR scheduler, no EMA). Not damage.
        * ``"absent_unverifiable"`` — absent, and CONDITIONALLY written, on a
                               checkpoint with no manifest: "never written" and
                               "lost" are indistinguishable here, so this is not
                               treated as damage. See ``_CKPT_CONDITIONAL``.
        * ``"unverified"``   — present, no manifest (checkpoint predates it);
                               presence is all that can be established.
        """
        path = ckpt_dir / name
        recorded = manifest.get(name) if isinstance(manifest, dict) else None
        if isinstance(manifest, dict) and recorded is None:
            # The manifest is authoritative about what was written. A file that
            # is not in it was not part of this checkpoint.
            return ("not_written", f"{name} was never written by this checkpoint")
        if not path.is_file():
            if manifest is None and name in _CKPT_CONDITIONAL:
                return ("absent_unverifiable",
                        f"{name} is absent from {ckpt_dir.name}, and the "
                        f"checkpoint predates the artifact manifest, so whether "
                        f"it was ever written cannot be established")
            return ("missing", f"{name} is absent from {ckpt_dir.name}")
        actual = int(path.stat().st_size)
        # An empty artifact is never valid state, with or without a manifest:
        # the same rule on both paths, so the verdict does not depend on the
        # checkpoint's generation.
        if actual == 0:
            return ("size_mismatch", f"{name} is empty (0 bytes)")
        if recorded is not None:
            try:
                expected = int(recorded)
            except (TypeError, ValueError):
                expected = None
            if expected is not None and actual != expected:
                return ("size_mismatch",
                        f"{name} is {actual} bytes but was {expected} bytes when "
                        f"the checkpoint recorded it (written or copied "
                        f"incompletely, or replaced since)")
            return ("ok", "")
        return ("unverified", "")

    def _assert_checkpoint_complete(self, ckpt_dir: Path) -> Tuple[Dict[str, Any],
                                                                   Dict[str, str]]:
        """Refuse a resume from a checkpoint that cannot be fully restored.

        Runs FIRST, before any state is touched: every check below is about the
        files on disk, and the config checks that follow read the same
        ``train_state.json`` this one validates.

        The tiers follow the existing rule of this trainer — *state that cannot
        be reconstructed and whose loss is invisible afterwards is a refusal;
        state that is re-derivable, or whose loss is announced and repaired, is a
        warning*:

        ============================  ========  ==============================
        artifact                      missing   why
        ============================  ========  ==============================
        ``train_state.json``          REFUSE    holds the step, the data pass,
                                                the component set and the base
                                                VAE identity; without it a
                                                resume can neither position
                                                itself nor run any other guard.
        ``vae_decoder.safetensors``   REFUSE    the trained weights themselves.
        ``optimizer.pt``              REFUSE    Adam's moments ARE this run's
                                                accumulated history; a fresh
                                                optimizer changes the effective
                                                step size for thousands of
                                                steps and looks exactly like a
                                                normal resume in every log and
                                                chart.
        ``lr_scheduler.pt``           REFUSE    (when this run has a scheduler
                                                and the checkpoint wrote one)
                                                the schedule would restart at
                                                position 0 while the step
                                                counter jumps to the
                                                checkpoint's — warmup and decay
                                                would be replayed silently.
        ``ema.safetensors``           warn      re-seeded from the restored
                                                weights, exactly as the
                                                pre-existing partial-EMA path
                                                does; the EMA is a derived
                                                average, and the re-seed is
                                                announced.
        ``rng_state.pt``              warn      only the noise/augmentation
                                                draw stream; the data order is
                                                restored separately and
                                                exactly by ``data_epoch``.
        ============================  ========  ==============================

        "Missing" and "the wrong size" are treated identically at every tier: a
        file that is not the one that was saved is not the state that was saved.

        Two of the artifacts are written CONDITIONALLY (``_CKPT_CONDITIONAL``:
        ``ema.safetensors``, ``lr_scheduler.pt``). On a checkpoint that carries
        the manifest their absence is PROVEN to be "never written" and warns; on
        one written before the manifest existed, "never written" and "lost"
        cannot be told apart, so the same absence warns there too rather than
        refusing. Refusing would strand every checkpoint produced by
        ``build_optimizer``'s "LR scheduler unavailable; using constant LR"
        fallback once the cause is fixed, with no way to resume it. The
        unconditionally-written artifacts have no such ambiguity: the writer
        always produces them, so absent means lost, manifest or not.

        A refusal lists the intact sibling checkpoints
        (``_intact_sibling_note``), because the damaged directory is typically
        the newest one and ``resume_from: latest`` selects exactly that.

        There is deliberately no "resume anyway" config key. A resume means
        *continue this run*; an optimizer reset is a different intent that is
        already expressible without one — stop, and start a NEW run whose base
        VAE is this run's exported ``*_vae`` directory. That gets a clean step
        counter, a clean LR schedule and a chart that does not silently splice
        two optimisation regimes into one series, which is what a "partial
        resume" would produce.
        """
        train_state, statuses, fatal, warnings, manifest = \
            self._inspect_checkpoint(ckpt_dir)

        if fatal:
            raise VaeConfigError(
                f"Checkpoint {ckpt_dir.name} is incomplete and cannot be resumed: "
                + "; ".join(fatal) + ". Resuming it would restore whatever is "
                f"present, and the step counter, while silently re-initialising "
                f"the rest - a discontinuity in the optimisation that no log, "
                f"metric or chart distinguishes from a normal resume."
                + self._intact_sibling_note(ckpt_dir) +
                f" To deliberately restart optimisation from these weights, "
                f"export the run and start a NEW run with that export as its base "
                f"VAE - that keeps the reset visible in the run history instead of "
                f"hiding it inside a resume."
            )

        if warnings:
            print(f"{self.log_prefix} WARNING: resuming {ckpt_dir.name} with "
                  f"state that is not fully restored:")
            for note in warnings:
                print(f"{self.log_prefix}   - {note}")
        if manifest is None:
            print(f"{self.log_prefix} Note: {ckpt_dir.name} predates the artifact "
                  f"manifest, so its files were checked for presence only; a file "
                  f"that was copied only in part cannot be detected here.")
        return train_state, statuses

    def _intact_sibling_note(self, ckpt_dir: Path) -> str:
        """" Resume from X instead" — the intact step_* directories next to this one.

        ``resume_from: latest`` is the shipped default and picks the HIGHEST step
        without looking at its contents, and the failure this guard exists for
        (a save interrupted, a copy still running) damages exactly the newest
        directory. So the refusal has to say which directories are still
        resumable, the way ``resolve_resume_target`` already lists what it found
        when an explicit name does not resolve. Deliberately a list rather than
        an automatic fallback: silently resuming an OLDER checkpoint would roll
        the step counter back by up to ``save_every`` steps and re-train that
        span, which is the same class of unannounced surprise this guard removes.
        """
        try:
            siblings = sorted(
                (d for d in ckpt_dir.parent.glob("step_*")
                 if d.is_dir() and d != ckpt_dir),
                key=lambda d: d.name)
        except Exception:
            return ""
        intact = []
        for sibling in siblings:
            try:
                if not self._inspect_checkpoint(sibling)[2]:
                    intact.append(sibling.name)
            except Exception:
                continue
        if not intact:
            return (" No other checkpoint in this run is resumable either, so "
                    "re-copy this directory if it is a partial copy, or start a "
                    "new run.")
        return (f" Resume from one of the intact checkpoints in the same folder "
                f"instead: {', '.join(intact)} (set resume_from_checkpoint to the "
                f"directory name; 'latest' will keep selecting this damaged one). "
                f"Re-copy this directory if it is a partial copy.")

    def _inspect_checkpoint(self, ckpt_dir: Path):
        """Classify a checkpoint's artifacts without raising.

        Returns ``(train_state, statuses, fatal, warnings, manifest)``;
        ``fatal`` empty means the directory is resumable by this run.
        Non-raising on purpose: ``_intact_sibling_note`` uses it to tell which
        neighbouring checkpoints are still usable.
        """
        state_path = ckpt_dir / "train_state.json"
        no_state = ({}, {n: "missing" for n in _CKPT_ARTIFACTS}, [], [], None)
        if not state_path.is_file():
            return no_state[:2] + ([
                f"train_state.json is absent from {ckpt_dir.name}, so the step "
                f"counter, the data pass, the trained component set and the base "
                f"VAE identity are all unavailable and no resume guard can run"],
                [], None)
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                train_state = json.load(f)
            if not isinstance(train_state, dict):
                raise ValueError(f"expected an object, got {type(train_state).__name__}")
        except Exception as e:
            return no_state[:2] + ([
                f"train_state.json in {ckpt_dir.name} is unreadable ({e}), so a "
                f"resume cannot position itself"], [], None)

        manifest = train_state.get("artifacts")
        if not isinstance(manifest, dict):
            manifest = None

        statuses: Dict[str, str] = {}
        details: Dict[str, str] = {}
        for name in _CKPT_ARTIFACTS:
            statuses[name], details[name] = self._artifact_status(
                ckpt_dir, name, manifest)

        fatal: List[str] = []
        warnings: List[str] = []
        # getattr, like _position_data_sampler / _assert_base_vae_matches: this
        # also runs on __new__-built trainers in the unit tests.
        scheduler = getattr(self, "lr_scheduler", None)
        ema = getattr(self, "ema", None)

        def _damaged(name: str) -> bool:
            return statuses[name] in ("missing", "size_mismatch")

        if statuses["vae_decoder.safetensors"] != "not_written":
            if _damaged("vae_decoder.safetensors"):
                fatal.append(f"{details['vae_decoder.safetensors']} - these are "
                             f"the trained weights themselves")
        else:
            fatal.append("vae_decoder.safetensors was never written, so this "
                         "directory holds no trained weights")

        if statuses["optimizer.pt"] == "not_written":
            fatal.append("optimizer.pt was never written, so the optimizer state "
                         "of the step this checkpoint claims does not exist")
        elif _damaged("optimizer.pt"):
            fatal.append(f"{details['optimizer.pt']} - the optimizer moments are "
                         f"this run's accumulated history and cannot be "
                         f"reconstructed from the weights")

        if scheduler is not None:
            if _damaged("lr_scheduler.pt"):
                fatal.append(f"{details['lr_scheduler.pt']} - the LR schedule "
                             f"would restart at position 0 while the step counter "
                             f"jumps to the checkpoint's")
            elif statuses["lr_scheduler.pt"] == "not_written":
                warnings.append(
                    "the checkpoint was written by a run with no LR scheduler, "
                    "but this run has one (lr_scheduler="
                    f"{self.cfg.get('lr_scheduler')!r}); it starts at schedule "
                    "position 0, so any warmup/decay is replayed from the "
                    "beginning at the resumed step count")
            elif statuses["lr_scheduler.pt"] == "absent_unverifiable":
                # No manifest, and the file is one save_checkpoint writes only
                # conditionally: this is the build_optimizer fallback's
                # checkpoint as much as it is a lost file, and the two cannot be
                # told apart. Announce the consequence, do not refuse.
                warnings.append(
                    f"{details['lr_scheduler.pt']}. This run has one "
                    f"(lr_scheduler={self.cfg.get('lr_scheduler')!r}), and it "
                    f"starts at schedule position 0, so any warmup/decay is "
                    f"replayed from the beginning at the resumed step count. If "
                    f"the file was lost rather than never written, resume from a "
                    f"checkpoint that still has it")
        elif statuses["lr_scheduler.pt"] in ("ok", "unverified"):
            warnings.append(
                "the checkpoint carries an LR schedule position but this run has "
                "no LR scheduler, so that position is discarded and the "
                "configured learning rate is used flat")

        # EMA and RNG: announced and repaired, never fatal. load_checkpoint reads
        # `statuses` back to decide whether to trust each file.
        if ema is not None and (_damaged("ema.safetensors")
                                or statuses["ema.safetensors"] == "absent_unverifiable"):
            warnings.append(f"{details['ema.safetensors']}; the EMA is re-seeded "
                            f"from the restored weights (its averaging history is "
                            f"lost, and the exported EMA weights will be closer to "
                            f"the raw ones for the next few thousand updates)")
        elif ema is None and statuses["ema.safetensors"] in ("ok", "unverified"):
            warnings.append("the checkpoint carries EMA weights but this run has "
                            "ema_enabled=false, so they are dropped and the export "
                            "will use the raw trained weights")

        if _damaged("rng_state.pt") or statuses["rng_state.pt"] == "not_written":
            warnings.append(
                f"{details['rng_state.pt'] or 'rng_state.pt is unusable'}; the "
                f"noise/augmentation draw stream restarts from the process seed "
                f"instead of continuing. The data order and per-visit crops are "
                f"restored separately and exactly (data_epoch), so this changes "
                f"which random draws are made, not which images are trained on")

        return train_state, statuses, fatal, warnings, manifest

    def _data_epoch_for_checkpoint(self) -> int:
        """The data pass in progress, for ``train_state.json``.

        0 when no sampler exists yet (a checkpoint written before ``train()``
        built the loader) -- the resume then simply starts at pass 1.
        """
        sampler = getattr(self, "train_sampler", None)
        if sampler is None:
            return 0
        return int(sampler.current_epoch)

    def _position_data_sampler(self, train_state: Dict[str, Any]) -> None:
        """Continue the data traversal into the pass AFTER the checkpointed one.

        The crop window and (under ``crop_scale_policy: mixed``) the scale factor
        of every item are keyed by this counter, so restarting at the
        checkpointed pass would re-serve the images the interrupted pass had
        already consumed with the very crops they just saw. ``+1`` makes the
        resumed stream a continuation rather than a partial repeat, and because
        it is a plain integer the resume stays exactly reproducible -- unlike the
        global-RNG restore next to it, which cannot express "where in the data
        order we were".

        Checkpoints written before ``data_epoch`` existed resume at pass 1, which
        is still distinct from the pass-0 crops they trained on.
        """
        # getattr, not attribute access: load_checkpoint is also driven against
        # __new__-built trainers (backend/tests/test_lr_resume_override.py), and
        # a resume must not die on an attribute the loop had not set yet.
        sampler = getattr(self, "train_sampler", None)
        if sampler is None:
            return
        sampler.set_next_epoch(int(train_state.get("data_epoch", 0)) + 1)

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
        """Refuse a resume whose checkpoint trained a different component set,
        or was trained on a different base VAE.

        Named, actionable and BEFORE any weight load. Optimizer state, EMA state
        and the trainable-name list are all indexed by the component set, so a
        mismatch is never recoverable — it is only ever a config mistake.

        The base-VAE check (``_assert_base_vae_matches``) runs in the same place
        for the same reason, and AFTER the component comparison: a checkpoint
        supplies only the tensors that were trainable, so everything else comes
        from the base VAE loaded by THIS run, and a base change across a resume
        is as unrecoverable — and far quieter — than a component change.

        Two further keys are compared but only WARNED about, at the end: they do
        not invalidate the checkpoint, they invalidate the *comparability of the
        charts across the resume*, which nothing else detects. See
        ``_warn_measurement_changes``.
        """
        state_path = ckpt_dir / "train_state.json"
        if not state_path.is_file():
            print(f"{self.log_prefix} WARNING: {ckpt_dir} has no train_state.json; "
                  f"cannot verify that it trained the same components as this run, "
                  f"nor that it was trained on the same base VAE.")
            return
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            saved = (state.get("config") or {})
        except Exception as e:
            print(f"{self.log_prefix} WARNING: could not read {state_path} ({e}); "
                  f"skipping the component-set and base-VAE checks.")
            return
        if not saved:
            self._assert_base_vae_matches(ckpt_dir, state.get("base_vae"))
            return

        saved_optimizer = saved.get("optimizer")
        current_optimizer = self.cfg.get("optimizer")
        if saved_optimizer is not None and current_optimizer is not None and \
                str(saved_optimizer).strip().lower() != \
                str(current_optimizer).strip().lower() and \
                (str(saved_optimizer).strip().lower(),
                 str(current_optimizer).strip().lower()) not in \
                _COMPATIBLE_OPTIMIZER_RESUME_PAIRS:
            raise VaeConfigError(
                f"Checkpoint {ckpt_dir.name} used optimizer "
                f"{saved_optimizer!r}, but this run uses {current_optimizer!r}. "
                f"Optimizer state is implementation-specific. The only "
                f"supported optimizer change during VAE resume is "
                f"AdamW -> AdamW8bit; otherwise match the checkpoint optimizer "
                f"or start a new run."
            )

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

        # Only now: the frozen-weight fingerprint covers the complement of the
        # trainable set, so it is only comparable once that set is known to match.
        self._assert_base_vae_matches(ckpt_dir, state.get("base_vae"))
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

        # Completeness FIRST: every later step assumes the files are the ones
        # save_checkpoint wrote, and the config guards below read the very
        # train_state.json this validates (and returns, so it is read once).
        train_state, artifacts = self._assert_checkpoint_complete(ckpt_dir)

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

        # EMA is the warn-and-repair tier: absent, short and unreadable all end
        # in an announced re-seed. _assert_checkpoint_complete has already warned
        # about an absent/truncated file; re-seed silently in that case rather
        # than printing the same fact twice.
        ema_restored = False
        if self.ema is not None:
            ema_state = None
            if artifacts["ema.safetensors"] in ("ok", "unverified"):
                try:
                    ema_state = load_file(str(ckpt_dir / "ema.safetensors"))
                except Exception as e:
                    print(f"{self.log_prefix} WARNING: ema.safetensors in "
                          f"{ckpt_dir} could not be read ({e}); re-seeding EMA "
                          f"from the restored weights.")
            elif artifacts["ema.safetensors"] == "not_written":
                print(f"{self.log_prefix} WARNING: {ckpt_dir.name} was written by "
                      f"a run with EMA disabled but this run has ema_enabled=true; "
                      f"seeding EMA from the restored weights.")
            if ema_state is None:
                self.init_ema()
            else:
                # A PARTIAL restore is worse than no restore: _update_ema()
                # indexes every trainable name, so a short dict would KeyError on
                # the first step after resume. Re-seed instead.
                ema_missing = [n for n in self.trainable_names if n not in ema_state]
                if ema_missing:
                    print(f"{self.log_prefix} WARNING: ema.safetensors in "
                          f"{ckpt_dir} is missing {len(ema_missing)} of "
                          f"{len(self.trainable_names)} tensor(s) "
                          f"(e.g. {ema_missing[:3]}); re-seeding EMA from the "
                          f"restored weights.")
                    self.init_ema()
                else:
                    self.ema = {k: ema_state[k].float().to(self.device)
                                for k in self.trainable_names}
                    ema_restored = True

        # Optimizer / scheduler are the refusal tier: _assert_checkpoint_complete
        # has already established that the files are there and the size they were
        # saved at, so anything that still fails here is a genuinely corrupt file
        # and must stop the run rather than leave the state re-initialised.
        try:
            # Always deserialize on CPU. Cross-optimizer conversion quantizes
            # fp32 AdamW moments there before load_state_dict casts the finished
            # state to each target parameter's device.
            optimizer_state = torch.load(
                ckpt_dir / "optimizer.pt", map_location="cpu", weights_only=False
            )
            saved_optimizer = ((train_state.get("config") or {}).get("optimizer"))
            current_optimizer = self.cfg.get("optimizer")
            saved_class = optimizer_state.get("_sushi_opt_class")
            saved_config_name = (str(saved_optimizer).strip().lower()
                                 if saved_optimizer is not None else None)
            saved_class_name = _OPTIMIZER_CLASS_TO_CONFIG_NAME.get(str(saved_class))
            current_class_name = _OPTIMIZER_CLASS_TO_CONFIG_NAME.get(
                type(self.optimizer).__name__
            )
            current_name = (str(current_optimizer).strip().lower()
                             if current_optimizer is not None else None)
            if saved_class_name is not None and saved_config_name is not None and \
                    saved_class_name != saved_config_name:
                raise ValueError(
                    f"optimizer.pt class {saved_class!r} contradicts "
                    f"train_state optimizer {saved_optimizer!r}"
                )
            if current_class_name is not None and current_name is not None and \
                    current_class_name != current_name:
                raise ValueError(
                    f"live optimizer class {type(self.optimizer).__name__!r} "
                    f"contradicts config optimizer {current_optimizer!r}"
                )
            saved_name = saved_class_name or saved_config_name
            # A pre-tag, config-less checkpoint cannot prove its source class.
            # Preserve the historical raw-load path in that case; same-optimizer
            # legacy resumes remain valid, while an incompatible raw dict still
            # receives the existing actionable load refusal.
            if saved_name is not None and current_name is not None and \
                    saved_name != current_name:
                pair = (saved_name, current_name)
                if pair not in _COMPATIBLE_OPTIMIZER_RESUME_PAIRS:
                    # Normally caught before weights are touched. Keep this
                    # second gate local to the state load for legacy config-less
                    # checkpoints and hand-built test callers.
                    raise ValueError(
                        f"unsupported optimizer resume pair {saved_optimizer!r} "
                        f"-> {current_optimizer!r}"
                    )
                from core.training.optimizers.optimizer_state_convert import (
                    maybe_convert_optimizer_state,
                )
                converted, _carry_step = maybe_convert_optimizer_state(
                    optimizer_state,
                    self.optimizer,
                    log_prefix=self.log_prefix,
                    source_optimizer_name=saved_name,
                )
                if converted is None:
                    # Never try the raw torch AdamW dict. bnb's loader can accept
                    # exp_avg keys and defer the actual KeyError until step().
                    raise ValueError(
                        f"required optimizer conversion {saved_optimizer!r} -> "
                        f"{current_optimizer!r} did not succeed"
                    )
                optimizer_state = converted
            self.optimizer.load_state_dict(optimizer_state)
        except Exception as e:
            raise VaeConfigError(
                f"Checkpoint {ckpt_dir.name} has an optimizer.pt this run cannot "
                f"load ({e}). Either the file is damaged, or it was written by a "
                f"run using a different optimizer than this one "
                f"(optimizer={self.cfg.get('optimizer')!r}) - an optimizer state "
                f"dict only loads back into the same optimizer type and param "
                f"group layout, except for the supported AdamW -> AdamW8bit "
                f"conversion. Resuming without it would restart the moment "
                f"estimates at the resumed step count, which nothing downstream "
                f"reports. Resume from an intact step_* directory, or match the "
                f"optimizer the checkpoint was written with."
                + self._intact_sibling_note(ckpt_dir)
            )
        if self.lr_scheduler is not None and \
                artifacts["lr_scheduler.pt"] in ("ok", "unverified"):
            try:
                self.lr_scheduler.load_state_dict(
                    torch.load(ckpt_dir / "lr_scheduler.pt",
                               map_location="cpu", weights_only=False))
            except Exception as e:
                raise VaeConfigError(
                    f"Checkpoint {ckpt_dir.name} has an lr_scheduler.pt this run "
                    f"cannot load ({e}). Either the file is damaged, or it was "
                    f"written under a different schedule than this run's "
                    f"(lr_scheduler={self.cfg.get('lr_scheduler')!r}). Resuming "
                    f"without it would replay warmup/decay from schedule position "
                    f"0 at the resumed step count."
                    + self._intact_sibling_note(ckpt_dir)
                )

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
        if "optimizer_weight_decay" in self.cfg:
            configured_weight_decay = float(self.cfg["optimizer_weight_decay"])
            restored_weight_decays = [group.get("weight_decay")
                                      for group in self.optimizer.param_groups]
            for group in self.optimizer.param_groups:
                group["weight_decay"] = configured_weight_decay
            if any(value is None or
                   not math.isclose(float(value), configured_weight_decay)
                   for value in restored_weight_decays):
                print(f"{self.log_prefix} Resume weight decay: checkpoint "
                      f"{restored_weight_decays} -> config "
                      f"{configured_weight_decay}")

        # RNG is the warn tier; _assert_checkpoint_complete has already explained
        # an absent/truncated file, so only a genuinely corrupt one prints here.
        if artifacts["rng_state.pt"] in ("ok", "unverified"):
            try:
                rng = torch.load(ckpt_dir / "rng_state.pt",
                                 map_location="cpu", weights_only=False)
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
                if rng.get("cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(
                        [s.cpu().to(torch.uint8) for s in rng["cuda"]])
            except Exception as e:
                print(f"{self.log_prefix} RNG restore failed (non-fatal): {e}")

        # train_state was read (and validated) by _assert_checkpoint_complete.
        self.global_step = int(train_state.get("step", 0))
        self._position_data_sampler(train_state)
        # Continue the EMA warmup ramp / retained-init product across resume --
        # ONLY when the EMA itself was restored. A re-seeded EMA equals the raw
        # weights, so adopting the checkpoint's update count would skip the
        # warmup ramp and average that seed in at the full decay, which is both
        # wrong and invisible; init_ema()'s counters are the honest ones.
        if self.ema is not None and ema_restored and train_state.get("ema_enabled"):
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
                # crop_scale_policy, crop_scale_max_downscale (2026-07-30),
                # base_vae.frozen_fingerprint (2026-08-01).
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


# What makes a single file a FULL CHECKPOINT is the presence of a BACKBONE, so
# that is what is looked for -- positively. The earlier version of this check
# asked the opposite question ("does every key look like a VAE key?"), which is
# an allow-list over a space nobody controls: a stock LDM
# `sdxl_vae.safetensors` also ships `model_ema.*`, and that one unlisted prefix
# made a plain VAE file classify as a full checkpoint, skipping the repair
# entirely and recording "family identified" in the provenance sidecar for a
# value diffusers had actually guessed (measured on a real 250-key stock file).
# Denoising diffusion backbones, by contrast, have a small and slow-moving set
# of top-level names, and a file that carries NONE of them carries no evidence
# of family whatever else is in it -- which is exactly the condition the repair
# is about. New VAE side-car tensors therefore no longer break the classifier;
# only a genuinely new backbone naming scheme would, and that fails SAFE (see
# the third state below and the vae_arch cross-check in the caller).
_BACKBONE_KEY_MARKERS = (
    "model.diffusion_model.",   # LDM / SD1.x / SD2 / SDXL UNet
    "model_ema.diffusion_model.",
    "diffusion_model.",         # some repacks drop the "model." level
    "unet.",                    # diffusers-style bundle
    "conditioner.",             # SDXL text encoders
    "cond_stage_model.",        # SD1.x text encoder
    "text_encoders.",           # ComfyUI-style bundles
    "text_model.",
    "double_blocks.",           # FLUX / DiT families
    "single_blocks.",
    "transformer.",
    "transformer_blocks.",
    "joint_blocks.",            # MMDiT (SD3)
)

# Prefixes a VAE-ONLY file is made of. Used only as a POSITIVE recognition step
# after the backbone check has already said "no backbone": a file made entirely
# of these is a VAE and nothing else. `model_ema.` / `first_stage_model.` /
# `vae.` are here because real VAE files ship them (EMA copies, LDM-namespaced
# and diffusers-namespaced dumps); in a full checkpoint they coexist with a
# backbone marker, which is tested first.
_VAE_ONLY_KEY_PREFIXES = ("encoder.", "decoder.", "quant_conv.",
                          "post_quant_conv.", "loss.", "model_ema.",
                          "first_stage_model.", "vae.")


def _classify_single_file(path: str):
    """Classify ``path`` as ``(verdict, reason)``.

    ``verdict`` is True for a VAE-ONLY file, False for a file carrying a
    backbone (a full checkpoint), and None when it cannot be determined.
    ``reason`` names WHICH of those situations it is, so the caller can say
    something true about it instead of guessing (a ``.ckpt`` and a corrupt
    safetensors header are both "None", but they are not the same problem).

    Reads the safetensors header only -- no tensor data.
    """
    if not isinstance(path, str) or not path.lower().endswith(".safetensors"):
        # .ckpt/.pt/.bin: telling the two apart would mean unpickling the file.
        return None, "not_safetensors"
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
    except Exception as e:
        print(f"[VaeTrainer] Could not read safetensors header of {path}: {e}")
        return None, "header_unreadable"
    if not keys:
        return None, "no_keys"
    if any(k.startswith(_BACKBONE_KEY_MARKERS) for k in keys):
        return False, "backbone"
    if all(k.startswith(_VAE_ONLY_KEY_PREFIXES) for k in keys):
        return True, "bare"
    # No backbone marker, but not recognisable as a pure VAE dump either. Do NOT
    # collapse this into either answer: an unrecognised layout is exactly where
    # a guess would be wrong in a way nothing downstream can see.
    return None, "unrecognised_keys"


def _is_bare_vae_single_file(path: str) -> Optional[bool]:
    """True when ``path`` is a VAE-ONLY safetensors file, False when it also
    holds a backbone (a full checkpoint), None when it cannot be determined."""
    return _classify_single_file(path)[0]


# How each "cannot be determined" reason is described to the user. The wording
# is the diagnosis, so it has to match the actual situation.
_UNKNOWN_REASON_PHRASE = {
    "not_safetensors": ("whose contents cannot be inspected without unpickling "
                        "it"),
    "header_unreadable": ("whose safetensors header could not be read (the "
                          "error is logged above)"),
    "no_keys": "whose safetensors header lists no tensors at all",
    "unrecognised_keys": ("whose key layout matches neither a VAE-only file nor "
                          "any known backbone"),
}

# What the user can actually DO about each of those states. Kept next to the
# phrase above and keyed the same way, because a remedy that does not apply to
# the state being reported ("save it as safetensors" for a file that already is
# one) sends the reader looking in the wrong place.
_UNKNOWN_REASON_REMEDY = {
    "not_safetensors": ("convert this VAE to .safetensors, or supply it as a "
                        "diffusers directory carrying its own config.json"),
    "header_unreadable": ("repair or re-fetch the file so its safetensors "
                          "header can be read, or supply this VAE as a "
                          "diffusers directory carrying its own config.json"),
    "no_keys": ("supply a file that actually contains the VAE tensors, or a "
                "diffusers directory carrying its own config.json"),
    "unrecognised_keys": ("supply this VAE as a diffusers directory carrying "
                          "its own config.json (which states scaling_factor "
                          "outright), or re-save the file with the standard "
                          "encoder./decoder./quant_conv./post_quant_conv. key "
                          "names so it is recognisable as a VAE-only file"),
}


def repair_single_file_scaling_factor(vae, path: str, vae_arch: Optional[str],
                                      log_prefix: str = "[VaeTrainer]") -> str:
    """Give a VAE loaded from a bare single file its architecture's own
    ``scaling_factor`` / ``shift_factor``, and say so out loud.

    ``AutoencoderKL.from_single_file`` cannot tell an SDXL VAE from an SD1.5 one
    -- the architectures are byte-for-byte the same shape -- so for a VAE-ONLY
    file it falls back to LDM's 0.18215. Training never reads the value, but
    ``save_pretrained`` writes it verbatim into the exported ``config.json``,
    and the inference-side VAE override trusts a directory's config.json. An
    SDXL export carrying 0.18215 is a silent 1.40x latent-scale error -- and an
    SD1.5 export carrying 0.13025 is the same error in the other direction, so
    the substituted value has to be STATED by the user, never assumed.

    ``process.vae.vae_arch`` (resolved through the shared VAE registry) is that
    statement. Its default is the empty string, i.e. "not stated": there is no
    family this function may fall back to, because both candidate falsehoods are
    equally undiagnosable downstream. The decision matrix:

    * full checkpoint (a backbone key is present) -> diffusers READ the family
      from the checkpoint, so the value is evidence and is never overwritten.
      It is still CROSS-CHECKED against a stated scalar ``vae_arch``, and a
      contradiction REFUSES: that disagreement is either a wrong ``vae_arch`` or
      a misclassified file, and both are things the run must not carry into an
      export. (This cross-check is also the backstop for the classifier itself:
      a VAE file misread as a checkpoint would otherwise skip the repair AND
      record "family identified" for a guessed number.)
    * ``vae_arch`` not stated -> REFUSE. Nothing here knows which family an
      unlabelled VAE belongs to, and guessing writes the wrong number to disk.
    * ``vae_arch`` stated but not a registry key (a typo) -> REFUSE, for the
      same reason: the run asked for a correction that cannot be made.
    * ``vae_arch`` names a family with no scalar scaling_factor (flux2,
      qwen_image: they normalise with latents_mean/latents_std) -> there is no
      number to substitute; left untouched, loudly, and recorded as UNVERIFIED.
    * ``vae_arch``'s latent_channels disagree with the loaded VAE -> the config
      is wrong about which VAE this is, so REFUSE rather than stamp a
      foreign family's number onto the export.
    * file not classifiable (a ``.ckpt`` / ``.pt`` / ``.bin``, an unreadable
      safetensors header, or a key layout that is neither a VAE dump nor any
      known backbone) -> the loaded value may be a genuine reading, so it is
      NEVER overwritten. It is only CHECKED
      against ``vae_arch``: agreement passes, disagreement REFUSES, because
      which of the two is right is exactly what cannot be determined here.
    * bare VAE-only ``.safetensors`` + a stated, matching-channel, scalar
      family -> corrected (this is the only branch that writes).

    Returns a short string recording HOW the effective value was arrived at; it
    is stored in the provenance sidecar so an export is self-describing.

    Raises:
        VaeConfigError: on any of the refusals above. This runs inside
        ``load_base_vae``, before a single training step or any export.
    """
    from core.models.common.vae_store import (
        LDM_SINGLE_FILE_DEFAULT_SCALING_FACTOR,
        canonical_latent_scaling,
    )

    loaded = getattr(vae.config, "scaling_factor", None)
    loaded_shift = getattr(vae.config, "shift_factor", None)

    bare, reason = _classify_single_file(path)
    arch = (vae_arch or "").strip()
    stated = canonical_latent_scaling(arch)[0] if (
        arch and canonical_latent_scaling(arch) is not None) else None

    if bare is False:
        # The file carries a backbone, so from_single_file read the family off
        # it. Evidence beats a config field -- but a stated scalar vae_arch that
        # CONTRADICTS that evidence is not a difference of opinion to be dropped
        # silently: one of the two is wrong, and either way the number that gets
        # baked into the export is in question.
        if stated is not None and loaded is not None and \
                abs(float(loaded) - float(stated)) > 1e-9:
            raise VaeConfigError(
                f"The base VAE at {path} carries a backbone, so "
                f"from_single_file identified its family and gave it "
                f"scaling_factor={loaded} -- but vae_arch={arch!r} says "
                f"{stated}. Refusing to run on that contradiction: either "
                f"vae_arch does not describe this file (leave it unset when the "
                f"base VAE comes from a full checkpoint, which states its own "
                f"value), or this is a VAE-only file that was misread as a "
                f"checkpoint, in which case {loaded} is diffusers' fallback "
                f"{LDM_SINGLE_FILE_DEFAULT_SCALING_FACTOR} and would be baked "
                f"into every export of this run by save_pretrained."
            )
        print(f"{log_prefix} scaling_factor={loaded} comes from the checkpoint "
              f"itself (the file carries a backbone, so from_single_file "
              f"identified the family); left as loaded"
              + (f", and it matches vae_arch={arch!r}." if stated is not None
                 else "."))
        return "from_single_file (full checkpoint, family identified)"

    # `bare is True`  -> a VAE-ONLY safetensors: `loaded` is provably a fallback.
    # `bare is None`  -> not classifiable: `loaded` MIGHT be a genuine reading.
    # Neither may proceed on an unstated family.
    if not arch:
        raise VaeConfigError(
            f"vae_arch is not set, and the base VAE at {path} is a single file "
            f"with no config.json"
            + (" (a VAE-only file, so its scaling_factor="
               f"{loaded} is diffusers' fallback "
               f"{LDM_SINGLE_FILE_DEFAULT_SCALING_FACTOR}, not a reading)"
               if bare else
               f" {_UNKNOWN_REASON_PHRASE.get(reason, 'that could not be classified')}"
               f", so its scaling_factor={loaded} may or may not be a reading")
            + ". SD1.5 and SDXL VAEs are byte-for-byte the same shape, so "
            "nothing here can tell them apart, and save_pretrained bakes "
            "whatever is on the config into every export of this run (0.18215 "
            "vs 0.13025 is a 1.40x latent-scale error that produces wrong "
            "images with no error and no warning). Refusing to guess: set "
            "process.vae.vae_arch to the architecture of this VAE "
            "(sdxl / sd15 / flux1 / flux2 / qwen_image) -- the 'VAE "
            "architecture' field in the VAE training form."
        )

    canonical = canonical_latent_scaling(arch)
    if canonical is None:
        raise VaeConfigError(
            f"vae_arch={arch!r} is not a known VAE-store key, and the base VAE "
            f"at {path} is a single file with no config.json, so vae_arch is "
            f"what has to decide the scaling_factor baked into every export of "
            f"this run. Refusing to run on a value that names no architecture: "
            f"set process.vae.vae_arch to one of "
            f"sdxl / sd15 / flux1 / flux2 / qwen_image."
        )

    if canonical[0] is None:
        print(f"{log_prefix} WARNING: the base VAE at {path} is a single file "
              f"with no config.json, so its scaling_factor={loaded} is whatever "
              f"from_single_file assumed (its fallback is "
              f"{LDM_SINGLE_FILE_DEFAULT_SCALING_FACTOR}), and vae_arch={arch!r} "
              f"has no single scalar scaling_factor (that family normalises with "
              f"latents_mean/latents_std), so there is no value to substitute. "
              f"Every export of this run will carry {loaded}. Consumers of this "
              f"family read latents_mean/latents_std instead.")
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

    if bare is None:
        # Not inspectable, and the loaded value contradicts the stated family.
        # Overwriting would corrupt a full checkpoint's genuine reading; keeping
        # it silently would leave a fallback in place. Both are undiagnosable
        # downstream, so neither is done.
        raise VaeConfigError(
            f"The base VAE at {path} carries scaling_factor={loaded}, but "
            f"vae_arch={arch!r} says {expected}. This is a single file "
            f"{_UNKNOWN_REASON_PHRASE.get(reason, 'that could not be classified')}"
            f", so it cannot be determined whether {loaded} was READ from a checkpoint "
            f"(in which case vae_arch is wrong) or is diffusers' fallback "
            f"{LDM_SINGLE_FILE_DEFAULT_SCALING_FACTOR} for a VAE-only file (in "
            f"which case it needs correcting). Refusing to pick one: fix "
            f"process.vae.vae_arch if it is wrong, or "
            + _UNKNOWN_REASON_REMEDY.get(
                reason,
                "supply this VAE as a diffusers directory carrying its own "
                "config.json")
            + ", which makes the answer determinable."
        )

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
