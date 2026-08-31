"""Does the gradient from noisy timesteps CONFLICT with the gradient from clean ones?

Why this exists
---------------
Min-SNR-gamma (Hang et al., ICCV 2023), "Addressing Negative Transfer in
Diffusion Models" (Go et al., NeurIPS 2023) and Switch-DiT all rest on one
premise: gradients from distant timesteps actively conflict -- negative cosine
-- so diffusion training is a multi-task problem needing gradient surgery,
timestep clustering or per-interval experts.

For SenseNova we have measured MAGNITUDES only: the understanding branch's
gradient norm falls monotonically with t (und/gen ratio 1.74 at the noisiest
decile, 0.52 at the cleanest) while the generation branch's is U-shaped. That
says the two ends differ. It does NOT say they disagree.

The distinction decides what to build. If the cosine is ~0, distant timesteps
are merely uncorrelated and this is a VARIANCE problem -- stratified sampling
and per-branch weighting are the answer, and they are nearly free. If it is
negative, there is real negative transfer and the expensive machinery
(PCGrad/Nash-MTL over parameter subsets, per-interval capacity) has a basis.
Nothing published answers this for a MoT whose conditioning tower is a separate
weight set, so it has to be measured here.

How
---
Accumulate each MNT window's gradients into two buckets split at the sampler's
median timestep, then report the cosine between the two accumulated vectors,
per component (generation branch vs understanding branch).

The gradients are far too large to hold twice (2 x 15 GiB per MoT half), so
each parameter's gradient is compressed by a BILINEAR SKETCH before
accumulation: for G of shape [out, in],

    S = L^T G R,    L ~ N(0, 1/k) of shape [out, k],  R ~ N(0, 1/k) of [in, k]

which gives E<S_A, S_B> = <A, B> exactly (take the expectation over L first,
then R; each contributes a Kronecker delta). Concatenated over the ~588
trainable Linears the sketch has 588*k^2 dimensions -- 37,632 at the default
k=8 -- which is ample for a stable cosine estimate, while costing 64 floats per
parameter instead of its full gradient. Compute is two small matmuls per
parameter, ~k/(2*tokens) of that parameter's own backward, i.e. ~0.1% at 4096
tokens.

L and R are shared by every parameter of the same size and are regenerated from
a fixed seed, so the two buckets are sketched with the SAME projection -- which
is what makes the inner product meaningful.

Opt-in, off by default: it costs a little compute and a little memory, and it
is a diagnostic, not part of training. It never modifies a gradient.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


class GradTimestepCosineProbe:
    """Two-bucket sketched gradient accumulator, split by timestep.

    Lifecycle per MNT window::

        probe.begin_window()
        for each MNT iteration:
            probe.begin_pass(t)          # picks the bucket for this pass
            ... backward ...             # probe.observe(param) fires per parameter
        metrics = probe.finish_window()  # cosines, or {} if the window was unusable
    """

    #: bucket index for passes at or below the split point, and above it
    LOW, HIGH = 0, 1

    def __init__(
        self,
        t_split: float,
        sketch_dim: int = 8,
        seed: int = 0,
        components: Optional[Dict[int, str]] = None,
    ):
        self.t_split = float(t_split)
        self.k = max(1, int(sketch_dim))
        self.seed = int(seed)
        self.components = components or {}
        # (in_or_out_size, device, dtype) -> projection matrix
        self._proj: Dict[Any, torch.Tensor] = {}
        # bucket -> id(param) -> sketch
        self._acc: Dict[int, Dict[int, torch.Tensor]] = {self.LOW: {}, self.HIGH: {}}
        self._bucket: Optional[int] = None
        self._passes = [0, 0]

    # -- projections ----------------------------------------------------
    def _projection(self, size: int, device, dtype) -> torch.Tensor:
        """A fixed [size, k] Gaussian with variance 1/k, shared by same-sized params.

        Built on a private generator seeded from (self.seed, size) so it never
        touches the training RNG streams -- the repo has already been bitten by
        a diagnostic consuming the training RNG (commit 28377024).
        """
        key = (size, str(device), dtype)
        cached = self._proj.get(key)
        if cached is not None:
            return cached
        gen = torch.Generator(device="cpu")
        gen.manual_seed((self.seed * 1_000_003 + size) & 0x7FFFFFFF)
        mat = torch.randn(size, self.k, generator=gen, dtype=torch.float32)
        mat = (mat / (self.k ** 0.5)).to(device=device, dtype=dtype)
        self._proj[key] = mat
        return mat

    # -- lifecycle ------------------------------------------------------
    def begin_window(self) -> None:
        self._acc = {self.LOW: {}, self.HIGH: {}}
        self._bucket = None
        self._passes = [0, 0]

    def begin_pass(self, t_value: float) -> None:
        self._bucket = self.LOW if float(t_value) <= self.t_split else self.HIGH
        self._passes[self._bucket] += 1

    def observe(self, param: torch.Tensor) -> None:
        """Sketch ``param.grad`` into the active bucket. Never raises.

        Called from the fused backward hook, BEFORE it clears the gradient. Any
        failure here must cost a diagnostic, not a training run, so everything
        is wrapped -- including the shape check, since a non-2D trainable
        parameter (norms, biases) simply has no bilinear sketch and is skipped
        rather than special-cased.
        """
        if self._bucket is None:
            return
        try:
            grad = param.grad
            if grad is None or grad.dim() != 2:
                return
            with torch.no_grad():
                g = grad.detach().float()
                left = self._projection(g.shape[0], g.device, g.dtype)
                right = self._projection(g.shape[1], g.device, g.dtype)
                sketch = left.t() @ g @ right  # [k, k]
                store = self._acc[self._bucket]
                key = id(param)
                prev = store.get(key)
                store[key] = sketch if prev is None else prev + sketch
        except Exception:
            return

    # -- readout --------------------------------------------------------
    def finish_window(self) -> Dict[str, float]:
        """Cosine between the two buckets' accumulated gradients, per component.

        Returns {} when either bucket saw no pass -- which happens whenever the
        window's draws all landed on one side of the split, and is normal at
        small MNT without stratification. A cosine from one bucket is not a
        number worth logging.
        """
        if min(self._passes) == 0:
            return {}
        low, high = self._acc[self.LOW], self._acc[self.HIGH]
        shared = [k for k in low if k in high]
        if not shared:
            return {}

        groups: Dict[str, list] = {"all": shared}
        for key in shared:
            groups.setdefault(self.components.get(key, "other"), []).append(key)

        out: Dict[str, float] = {}
        for name, keys in groups.items():
            dot = 0.0
            nl = 0.0
            nh = 0.0
            for key in keys:
                a, b = low[key], high[key]
                dot += float((a * b).sum())
                nl += float((a * a).sum())
                nh += float((b * b).sum())
            if nl <= 0.0 or nh <= 0.0:
                continue
            out[f"grad_cos_t_{name}"] = dot / ((nl ** 0.5) * (nh ** 0.5))
        out["grad_cos_t_npass_low"] = float(self._passes[self.LOW])
        out["grad_cos_t_npass_high"] = float(self._passes[self.HIGH])
        return out
