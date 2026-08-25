"""Per-component parameter-change tracking for training runs.

Split out of ``base_trainer.py`` (plan P8, optional item). ``base_trainer`` and
any historical importer keep working via the re-export in ``base_trainer.py``
(``from core.training.parameter_change_tracker import ParameterChangeTracker``).
"""

from typing import Dict, List, Optional

import torch


class ParameterChangeTracker:
    """
    Tracks per-component parameter changes during training.

    Computes two metrics every `interval` optimizer steps:
      B - Update norm:       ||θ_t - θ_{t-K}||_F  (how much changed in last K steps)
      C - Cumulative drift:  ||θ_t - θ_0||_F / ||θ_0||_F  (relative change from start)

    All computation and storage happens on CPU (fp16) → zero VRAM overhead.
    CPU RAM usage: ~2 × sum(component_param_bytes / 2) for full FT SDXL ≈ 14 GB total.
    """

    def __init__(self, components: Dict[str, torch.nn.Module], interval: int = 100):
        """
        Args:
            components: {name: module} for each trainable component
                        Keys: 'unet', 'te1', 'te2', 've'
            interval:   Compute metrics every N optimizer steps
        """
        self.components = {k: v for k, v in components.items() if v is not None}
        # compute() takes `step % interval`; 0 would raise instead of tracking.
        self.interval = max(1, int(interval or 1))

        # Reference snapshot for C (set once at init, never updated)
        self._reference: Dict[str, List[torch.Tensor]] = {}
        self._reference_norms: Dict[str, float] = {}

        # Previous snapshot for B (updated every `interval` steps)
        self._prev: Dict[str, List[torch.Tensor]] = {}

        self._initialize()

    def _snapshot(self, module: torch.nn.Module) -> List[torch.Tensor]:
        """Copy all trainable parameters to CPU as fp16 tensors."""
        return [p.detach().cpu().to(torch.float16)
                for p in module.parameters() if p.requires_grad]

    @staticmethod
    def _norm_sq(tensors: List[torch.Tensor]) -> float:
        """Compute sum of squared L2 norms (returns ||tensors||_F^2)."""
        total = 0.0
        for t in tensors:
            total += t.float().norm(2).item() ** 2
        return total

    @staticmethod
    def _delta_norm_sq(curr: List[torch.Tensor], ref: List[torch.Tensor]) -> float:
        """Compute ||curr - ref||_F^2 parameter-by-parameter to avoid large allocations."""
        total = 0.0
        for c, r in zip(curr, ref):
            delta = c.float() - r.float()
            total += delta.norm(2).item() ** 2
        return total

    def _initialize(self):
        total_params = 0
        total_bytes = 0
        for name, module in self.components.items():
            snap = self._snapshot(module)
            self._reference[name] = snap
            self._reference_norms[name] = self._norm_sq(snap) ** 0.5
            # Deep copy for prev (independent list of cloned tensors)
            self._prev[name] = [t.clone() for t in snap]
            n = sum(t.numel() for t in snap)
            total_params += n
            total_bytes += n * 2  # fp16 = 2 bytes per element
            print(f"[ParamTracker]   {name}: {n / 1e6:.1f}M params snapshot stored")
        print(f"[ParamTracker] Initialized. "
              f"Total tracked: {total_params / 1e6:.1f}M params, "
              f"~{total_bytes * 2 / 1e9:.1f} GB CPU RAM (ref + prev snapshots)")

    def compute(self, step: int) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Compute B and C metrics if `step` is a multiple of `interval`.

        Returns:
            {'update_norm': {name: float}, 'cumulative_drift': {name: float}}
            or None if not at interval boundary.
        """
        if step % self.interval != 0 or step == 0:
            return None

        update_norms: Dict[str, float] = {}
        cumulative_drifts: Dict[str, float] = {}

        for name, module in self.components.items():
            curr = self._snapshot(module)

            # B: update norm since last checkpoint
            update_norms[name] = self._delta_norm_sq(curr, self._prev[name]) ** 0.5

            # C: normalized cumulative drift from reference
            drift = self._delta_norm_sq(curr, self._reference[name]) ** 0.5
            ref_norm = self._reference_norms[name]
            cumulative_drifts[name] = drift / ref_norm if ref_norm > 0 else 0.0

            # Update prev for next B computation
            self._prev[name] = curr

        return {'update_norm': update_norms, 'cumulative_drift': cumulative_drifts}
