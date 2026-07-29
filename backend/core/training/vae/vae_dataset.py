"""Raw-pixel torch Dataset for VAE training.

Deliberately NOT the latent-cache path: VAE training is defined by a live
encode->decode forward on raw pixels (a pre-encoded cache is HARD-REFUSED in
``vae_config``).

The geometry/normalisation convention is the one from
``base_trainer.encode_image``'s preamble (base_trainer.py:4407-4508): shortest
side scaled up if needed, crop to the target, then ``(rgb/255 - 0.5) * 2`` into
a ``[3, H, W]`` float tensor in [-1, 1]. Those few lines are reproduced here
rather than imported, because ``encode_image`` continues into a
``torch.no_grad()`` VAE forward (base_trainer.py:4545) — precisely the thing a
VAE trainer must not inherit.

Captions are not read: a plain (non-text-conditioned) VAE has no use for them.

No aspect-ratio bucketing, and that is a measured decision rather than a
collation convenience (which is what this docstring used to claim, circularly).
The decoder's only non-local terms are ONE flattened mid-block self-attention
and 30 GroupNorms; ``AttnProcessor2_0`` reshapes ``[B,C,H,W]`` to ``[B,H*W,C]``
before attending and GroupNorm reduces over ``(C_group, H, W)``, so both observe
latent **area**, never aspect. Everything else is convolution. Constant-area
bucketing would therefore change nothing the architecture can perceive — and the
train/inference area gap it would not fix (the median generation gives the
decoder 5.06x the training token count) was measured harmless: the fine-tune's
PSNR advantage held at +0.93..+1.20 dB from 4,096 to 36,864 latent tokens with
no significant trend in any sharpness metric
(scratchpad/vae_training/results_crop_geometry.md §4).

**Crop scale policy** (``CROP_SCALE_POLICIES``). What *does* dominate is how much
an image is resampled before the crop is taken, and that is now a config choice
rather than a hardcoded one. See ``load_image_tensor`` for the three policies and
``results_crop_geometry.md`` for the measurement: over the 22 datasets in use
(3,842,897 items) the historical downscale-then-crop policy resamples 95.79% of
images by a median 2.30x (§1.2), a LANCZOS downscale *concentrates* high frequency
rather than blunting it (the production crop carries 4.06x the top-octave power of
a native crop, n=300, t=+21.6, §2), and the measured cost is calibration: the
fine-tune's accuracy gain is ~30% smaller on native-resolution content (edge
residual -7.7% vs -12.5%, positive on 19/19, t=+7.49, §3.2). §8 verifies this
implementation: ``downscale`` is pixel-identical to the pre-policy loader on
400/400 real dataset images, and the realised factor distribution of each policy
is tabulated there.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# Training datasets routinely contain slightly-truncated JPEGs; the rest of the
# repo's loaders tolerate them rather than crashing a multi-hour run.
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Crop scale policies. The enum itself lives with the other VALID_* enums in
# vae_config (the pure-config gate, which must not import torch/PIL), so there is
# exactly one definition of the allowed values; see load_image_tensor for the
# semantics and scratchpad/vae_training/results_crop_geometry.md for the
# measurement behind them.
from core.training.vae.vae_config import VALID_CROP_SCALE_POLICIES  # noqa: E402

CROP_SCALE_POLICIES = VALID_CROP_SCALE_POLICIES
DEFAULT_CROP_SCALE_POLICY = "downscale"


def resolve_crop_scale(
    short_side: int,
    resolution: int,
    *,
    scale_policy: str = DEFAULT_CROP_SCALE_POLICY,
    max_downscale: float = 0.0,
    rng: Optional[random.Random] = None,
) -> float:
    """The resize factor a policy asks for, as ``new_short / short_side``.

    Split out from ``load_image_tensor`` so the realised downscale-factor
    distribution of a policy can be measured without decoding pixels.

    ``< 1.0`` downscales, ``1.0`` means no resample at all, ``> 1.0`` upscales.
    The UPSCALE branch is deliberately common to every policy: an image whose
    short side is below ``resolution`` has no ``resolution``-sized window to crop,
    so it must be enlarged whatever the policy says. (4.21% of the corpus.)
    """
    rng = rng or random
    if short_side < resolution:
        # Identical expression to the pre-policy implementation, so this branch
        # is bit-identical under every policy.
        return resolution / short_side

    if scale_policy == "downscale":
        return resolution / short_side
    if scale_policy == "native":
        return 1.0
    if scale_policy != "mixed":
        raise ValueError(
            f"scale_policy must be one of {list(CROP_SCALE_POLICIES)}, "
            f"got {scale_policy!r}"
        )

    # mixed: draw the downscale factor per sample, LOG-uniformly over
    # [1, f_max]. Log- rather than linear-uniform because the factor is a
    # multiplicative quantity whose available range depends on the source size:
    # under linear-uniform sampling a 5120 px source (f_max = 10) would put 90%
    # of its draws above 2x while a 600 px source puts all of its draws near 1x,
    # so the corpus-level distribution would be dragged towards precisely the
    # heavily-resampled regime whose mis-calibration motivated this knob
    # (results_crop_geometry.md §3.2). Log-uniform gives equal weight per octave
    # of resampling instead: the median draw sits at sqrt(f_max), i.e. ~1.5x for
    # the corpus median f_max of 2.30, and 1x is a limit of the support rather
    # than a special case. The measured dose-response is monotone in the factor
    # and inference presents ~1x, so mass belongs near 1x — but not ALL of it,
    # since the downscaled regime is what run 113 has 52k steps of history on.
    f_max = short_side / resolution
    if max_downscale > 0:
        f_max = min(f_max, max_downscale)
    if f_max <= 1.0:
        return 1.0
    factor = math.exp(rng.uniform(0.0, math.log(f_max)))
    return 1.0 / factor


def load_image_tensor(
    path: str,
    resolution: int,
    *,
    random_crop: bool = True,
    rng: Optional[random.Random] = None,
    scale_policy: str = DEFAULT_CROP_SCALE_POLICY,
    max_downscale: float = 0.0,
) -> torch.Tensor:
    """Load one image as a ``[3, resolution, resolution]`` float32 tensor in [-1, 1].

    Aspect ratio is always preserved (nothing is squashed) and the crop is always
    square at ``resolution`` — randomly placed when ``random_crop`` is set
    (training), centred otherwise (validation, so the held-out metric is
    deterministic across steps).

    ``scale_policy`` decides how much the image is resampled BEFORE that crop,
    which is the variable the crop-geometry study found dominant:

    - ``"downscale"`` (default, and the historical behaviour): the SHORT side is
      scaled to exactly ``resolution``, up or down. 95.79% of the corpus is
      therefore downscaled, by a median 2.30x, and the crop covers ~73% of the
      source's area.
    - ``"native"``: never downscale. When the short side already reaches
      ``resolution`` the window is cut straight out of the full-size pixels, so
      the decoder sees genuine unresampled detail; only genuinely-smaller images
      are upscaled. The crop then covers ~20% of the median source's area.
    - ``"mixed"``: draw the downscale factor per sample over ``[1, f_max]``, so
      the decoder sees the whole range including 1x. ``max_downscale`` (0 = the
      image's own ``short/resolution``) bounds ``f_max``; see
      ``resolve_crop_scale`` for why the draw is log-uniform.

    ``max_downscale`` is read only under ``"mixed"`` — ``vae_config`` refuses the
    combination rather than letting it be silently ignored.
    """
    rng = rng or random
    with Image.open(path) as im:
        image = im.convert("RGB")

        w, h = image.size
        scale = resolve_crop_scale(
            min(w, h), resolution,
            scale_policy=scale_policy, max_downscale=max_downscale, rng=rng,
        )
        if scale != 1.0:
            new_w = max(resolution, int(round(w * scale)))
            new_h = max(resolution, int(round(h * scale)))
            image = image.resize((new_w, new_h), Image.LANCZOS)
            w, h = new_w, new_h

        max_left, max_top = w - resolution, h - resolution
        if random_crop:
            left = rng.randint(0, max_left) if max_left > 0 else 0
            top = rng.randint(0, max_top) if max_top > 0 else 0
        else:
            left, top = max_left // 2, max_top // 2
        image = image.crop((left, top, left + resolution, top + resolution))

        arr = np.array(image).astype(np.float32) / 255.0
        arr = (arr - 0.5) * 2.0

    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class VaeRawImageDataset(Dataset):
    """Random square crops at ``resolution`` from the shared dataset items.

    ``items`` are the dicts produced by ``train_runner.get_dataset_items_fast``
    (shape at train_runner.py:549-556); only ``image_path`` is used.

    ``scale_policy`` / ``max_downscale`` are passed through to
    ``load_image_tensor`` per sample, which is what makes ``"mixed"`` a per-sample
    draw rather than a per-run choice.

    An unreadable/corrupt image is skipped by walking forward to the next index
    rather than aborting the run — a single bad file in a 100k-item dataset must
    not kill a multi-hour fine-tune. Repeated failures are reported once each.
    """

    def __init__(
        self,
        items: List[Dict],
        resolution: int,
        *,
        random_crop: bool = True,
        seed: int = 0,
        scale_policy: str = DEFAULT_CROP_SCALE_POLICY,
        max_downscale: float = 0.0,
    ):
        self.paths = [it["image_path"] for it in items if it.get("image_path")]
        if not self.paths:
            raise ValueError("VaeRawImageDataset: no items with an image_path")
        self.resolution = int(resolution)
        self.random_crop = bool(random_crop)
        self.seed = int(seed)
        if scale_policy not in CROP_SCALE_POLICIES:
            raise ValueError(
                f"VaeRawImageDataset: scale_policy must be one of "
                f"{list(CROP_SCALE_POLICIES)}, got {scale_policy!r}"
            )
        self.scale_policy = scale_policy
        self.max_downscale = float(max_downscale)
        self._reported_failures = set()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        n = len(self.paths)
        # Per-item RNG so a worker-parallel loader is reproducible for a seed.
        rng = random.Random((self.seed * 1000003) ^ index)
        for attempt in range(min(n, 32)):
            i = (index + attempt) % n
            path = self.paths[i]
            try:
                return load_image_tensor(
                    path, self.resolution,
                    random_crop=self.random_crop, rng=rng,
                    scale_policy=self.scale_policy,
                    max_downscale=self.max_downscale,
                )
            except Exception as e:
                if path not in self._reported_failures:
                    self._reported_failures.add(path)
                    print(f"[VaeDataset] Skipping unreadable image {path}: "
                          f"{type(e).__name__}: {e}")
        raise RuntimeError(
            f"VaeRawImageDataset: 32 consecutive images failed to load starting "
            f"at index {index}; the dataset paths are probably invalid."
        )


def make_validation_batch(
    items: List[Dict],
    resolution: int,
    count: int,
) -> torch.Tensor:
    """Deterministic held-out batch: the LAST ``count`` items, centre-cropped.

    Taking them from the tail (and excluding them from the training set, see
    ``VaeTrainer._split_items``) keeps the validation signal honest without
    needing a separate dataset registration.

    **The validation crop scale policy is pinned to ``"downscale"`` and takes no
    parameter**, deliberately, so that it cannot follow the training policy:

    - it is deterministic by construction (no RNG at all — ``"mixed"`` would draw
      a fresh factor per call and make a held-out series noisy for a reason
      unrelated to the model);
    - ``vae_val_psnr`` stays comparable to every VAE run recorded so far,
      including run 113's 52k steps. A held-out metric whose *content
      distribution* moves when a training knob moves cannot be read across the
      change, and PSNR is strongly scale-dependent here — the same fine-tune
      measured +1.15 dB on downscaled content and +0.81 dB on native
      (results_crop_geometry.md §3.2), so a policy-following validation set would
      show a step no model change caused.

    The representativeness problem that motivates the policy work is addressed on
    the other axis instead: ``validation_resolution`` now defaults to 1024, for
    whose median 1131 px source the downscale factor is ~1.1x, i.e. nearly native
    anyway (results_crop_geometry.md §6.6).
    """
    picked = items[-count:] if count < len(items) else items
    tensors = []
    for it in picked:
        path = it.get("image_path")
        if not path:
            continue
        try:
            tensors.append(load_image_tensor(
                path, resolution, random_crop=False,
                scale_policy="downscale", max_downscale=0.0))
        except Exception as e:
            print(f"[VaeDataset] Validation image skipped {path}: "
                  f"{type(e).__name__}: {e}")
    if not tensors:
        raise ValueError("make_validation_batch: no loadable validation images")
    return torch.stack(tensors, dim=0)
