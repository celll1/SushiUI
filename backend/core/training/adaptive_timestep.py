"""Bounded, metric-driven adaptation of a flow-training timestep law.

The controller deliberately reacts to *learning progress* (fast/slow loss EMA),
not absolute loss.  This prevents a hard/noisy bin from attracting ever more
probability merely because its target has a larger native scale.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch

from .timestep_sampler import (
    MorphingTimestepSampler,
    QuantileTableSampler,
    TimestepSampler,
    build_sampler_from_expr,
    sampler_expr,
)


ADAPTIVE_MODES = ("off", "observe", "auto", "bounded")


def adaptive_defaults() -> Dict[str, Any]:
    from api.param_defaults import TRAINING_DEFAULTS

    return dict(TRAINING_DEFAULTS["timestep_sampling"]["adaptive"])


def validate_adaptive_timestep_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = adaptive_defaults()
    merged.update(dict(config or {}))
    mode = str(merged["mode"]).lower()
    if mode not in ADAPTIVE_MODES:
        raise ValueError(f"adaptive.mode must be one of {ADAPTIVE_MODES}, got {mode!r}")
    merged["mode"] = mode
    for key in ("warmup_updates", "control_interval", "bins", "morph_updates",
                "cooldown_updates", "min_observations", "auto_observe_controls",
                "auto_min_bin_observations"):
        raw = merged[key]
        if isinstance(raw, bool):
            raise ValueError(f"adaptive.{key} must be an integer")
        try:
            value = int(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"adaptive.{key} must be an integer") from exc
        if isinstance(raw, float) and raw != value:
            raise ValueError(f"adaptive.{key} must be an integer")
        merged[key] = value
    if merged["warmup_updates"] < 0 or merged["cooldown_updates"] < 0:
        raise ValueError("adaptive warmup/cooldown updates must be >= 0")
    if merged["control_interval"] < 1 or merged["morph_updates"] < 1:
        raise ValueError("adaptive control_interval/morph_updates must be >= 1")
    if not 2 <= merged["bins"] <= 32:
        raise ValueError("adaptive.bins must be in [2, 32]")
    if merged["min_observations"] < merged["bins"]:
        raise ValueError("adaptive.min_observations must be >= adaptive.bins")
    if merged["auto_observe_controls"] < 1:
        raise ValueError("adaptive.auto_observe_controls must be >= 1")
    if merged["auto_min_bin_observations"] < 1:
        raise ValueError("adaptive.auto_min_bin_observations must be >= 1")
    for key in ("log_snr_min", "log_snr_max", "coverage_floor",
                "max_density_ratio", "controller_gain", "auto_min_bin_probability"):
        merged[key] = float(merged[key])
        if not math.isfinite(merged[key]):
            raise ValueError(f"adaptive.{key} must be finite")
    if merged["log_snr_min"] >= merged["log_snr_max"]:
        raise ValueError("adaptive.log_snr_min must be < adaptive.log_snr_max")
    if not 0.0 < merged["coverage_floor"] <= 1.0:
        raise ValueError("adaptive.coverage_floor must be in (0, 1]")
    if merged["max_density_ratio"] < 1.0:
        raise ValueError("adaptive.max_density_ratio must be >= 1")
    if not 0.0 <= merged["controller_gain"] <= 1.0:
        raise ValueError("adaptive.controller_gain must be in [0, 1]")
    if not 0.0 <= merged["auto_min_bin_probability"] <= 1.0:
        raise ValueError("adaptive.auto_min_bin_probability must be in [0, 1]")
    return merged


class AdaptiveTimestepSampler(TimestepSampler):
    """Mutable sampler wrapper whose identity stays stable in the train loop."""

    _TABLE_POINTS = 1025

    def __init__(self, base: TimestepSampler, config: Dict[str, Any], *,
                 convention: str, prediction_type: str = "flow_velocity",
                 latent_centered_second_moment: Optional[float] = None,
                 resume_state: Optional[Dict[str, Any]] = None):
        cfg = validate_adaptive_timestep_config(config)
        super().__init__(base.min_timestep, base.max_timestep)
        if convention not in ("t0", "t1"):
            raise ValueError(f"adaptive timestep needs t0/t1 convention, got {convention!r}")
        supported_predictions = {
            "flow_velocity",
            "endpoint_observable_residual",
            "endpoint_observable_velocity",
        }
        if prediction_type not in supported_predictions:
            raise ValueError(
                "adaptive timestep does not support prediction type "
                f"{prediction_type!r}"
            )
        is_endpoint_observable = prediction_type.startswith("endpoint_observable_")
        q = None if latent_centered_second_moment is None else float(
            latent_centered_second_moment
        )
        if is_endpoint_observable and (
            q is None or not math.isfinite(q) or q <= 0.0
        ):
            raise ValueError(
                "endpoint-observable adaptive timestep requires a finite positive "
                "latent_centered_second_moment"
            )
        self.base = base
        self.current: TimestepSampler = base
        self.config = cfg
        self.convention = convention
        self.prediction_type = prediction_type
        self.latent_centered_second_moment = q
        self.update_step = 0
        self.last_control_update = -10**18
        self.observations_since_control = 0
        n = cfg["bins"]
        self.counts = [0] * n
        self.fast_ema = [math.nan] * n
        self.slow_ema = [math.nan] * n
        self.last_density_ratio = [1.0] * n
        self.base_bin_probability = self._base_bin_probabilities()
        if self.mode == "auto" and not self._auto_eligible_bins():
            raise ValueError(
                "adaptive.auto_min_bin_probability excludes every base-law bin")
        self.control_count = 0
        self.auto_promoted = False
        self.auto_promotion_update: Optional[int] = None
        self._last_action = "warming_up"
        if resume_state:
            self.load_state(resume_state)

    @property
    def mode(self) -> str:
        return self.config["mode"]

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.current.sample(batch_size, device)

    def icdf(self, u: torch.Tensor) -> torch.Tensor:
        return self.current.icdf(u)

    def sample_stratified(self, n_strata: int, batch_size: int,
                          device: torch.device) -> torch.Tensor:
        return self.current.sample_stratified(n_strata, batch_size, device)

    def _sigma(self, timestep: float) -> float:
        return 1.0 - timestep if self.convention == "t1" else timestep

    def _bin_and_x0_loss(self, timestep: float, prediction_loss: float):
        sigma = min(1.0 - 1e-6, max(1e-6, self._sigma(timestep)))
        if self.prediction_type.startswith("endpoint_observable_"):
            clean_time = 1.0 - sigma
            alpha = 2.0 * clean_time**2 - clean_time**3
            path_sigma = 1.0 - clean_time - clean_time**2 + clean_time**3
            alpha = max(1e-12, alpha)
            path_sigma = max(1e-12, path_sigma)
            log_snr = (
                math.log(self.latent_centered_second_moment)
                + 2.0 * (math.log(alpha) - math.log(path_sigma))
            )
            controller_loss = float(prediction_loss)
        else:
            log_snr = 2.0 * (math.log1p(-sigma) - math.log(sigma))
            # x0 = x_t - sigma*v for straight flow interpolation.
            controller_loss = float(prediction_loss) * sigma * sigma
        lo, hi = self.config["log_snr_min"], self.config["log_snr_max"]
        position = (min(hi, max(lo, log_snr)) - lo) / (hi - lo)
        index = min(self.config["bins"] - 1, int(position * self.config["bins"]))
        return index, controller_loss

    def observe(self, timesteps: torch.Tensor,
                prediction_loss: float | torch.Tensor) -> None:
        if self.mode == "off":
            return
        flat = timesteps.detach().reshape(-1)
        losses = torch.as_tensor(prediction_loss).detach().reshape(-1)
        if losses.numel() == 1 and flat.numel() != 1:
            raise ValueError(
                "adaptive timestep needs one prediction loss per timestep")
        if losses.numel() != flat.numel():
            raise ValueError(
                f"adaptive timestep received {losses.numel()} losses for "
                f"{flat.numel()} timesteps")
        for timestep, prediction in zip(flat.tolist(), losses.tolist()):
            loss = float(prediction)
            if not math.isfinite(loss) or loss < 0.0:
                continue
            index, x0_loss = self._bin_and_x0_loss(float(timestep), loss)
            self.counts[index] += 1
            self.observations_since_control += 1
            old_fast, old_slow = self.fast_ema[index], self.slow_ema[index]
            self.fast_ema[index] = x0_loss if not math.isfinite(old_fast) else 0.1 * x0_loss + 0.9 * old_fast
            self.slow_ema[index] = x0_loss if not math.isfinite(old_slow) else 0.01 * x0_loss + 0.99 * old_slow

    def set_optimizer_update_step(self, update_step: int) -> None:
        self.update_step = int(update_step)
        setter = getattr(self.current, "set_optimizer_update_step", None)
        if setter is not None:
            setter(self.update_step)
        if isinstance(self.current, MorphingTimestepSampler) and self.current.is_finished():
            self.current = self.current.target
        self._maybe_control()

    def _maybe_control(self) -> None:
        cfg = self.config
        if self.mode == "off" or self.update_step < cfg["warmup_updates"]:
            self._last_action = "warming_up"
            return
        wait_updates = max(cfg["control_interval"], cfg["cooldown_updates"])
        if self.mode == "bounded" or self.auto_promoted:
            wait_updates = max(wait_updates, cfg["morph_updates"])
        next_allowed = self.last_control_update + wait_updates
        if self.update_step < next_allowed:
            self._last_action = "cooldown"
            return
        if self.observations_since_control < cfg["min_observations"]:
            self._last_action = "collecting"
            return
        ratios = []
        for fast, slow in zip(self.fast_ema, self.slow_ema):
            if not (math.isfinite(fast) and math.isfinite(slow)) or slow <= 0.0:
                ratios.append(1.0)
            else:
                progress = max(-1.0, min(1.0, math.log(max(1e-12, fast / slow))))
                ratios.append(math.exp(cfg["controller_gain"] * progress))
        floor, ceiling = cfg["coverage_floor"], cfg["max_density_ratio"]
        # Project onto E_base[density_ratio] == 1 while enforcing both hard
        # bounds. Using equal bin weights would not bound the actual density
        # when the configured base law is non-uniform.
        low_scale, high_scale = 0.0, ceiling / max(1e-12, min(ratios)) * 2.0
        for _ in range(80):
            scale = 0.5 * (low_scale + high_scale)
            projected = [min(ceiling, max(floor, scale * value)) for value in ratios]
            mass = sum(p * value for p, value in zip(
                self.base_bin_probability, projected))
            if mass < 1.0:
                low_scale = scale
            else:
                high_scale = scale
        ratios = [min(ceiling, max(floor, high_scale * value)) for value in ratios]
        self.last_density_ratio = ratios
        self.last_control_update = self.update_step
        self.observations_since_control = 0
        self.control_count += 1
        if self.mode == "auto" and not self.auto_promoted:
            eligible = self._auto_eligible_bins()
            bins_ready = all(
                self.counts[index] >= cfg["auto_min_bin_observations"]
                for index in eligible
            )
            controls_ready = self.control_count >= cfg["auto_observe_controls"]
            if not (bins_ready and controls_ready):
                self._last_action = "auto_observing"
                return
            self.auto_promoted = True
            self.auto_promotion_update = self.update_step
            self._last_action = "auto_promoted"
        else:
            self._last_action = "observed" if self.mode == "observe" else "morphing"
        if self.mode == "observe":
            return
        target = self._weighted_target(ratios)
        source = self.current.freeze() if isinstance(
            self.current, MorphingTimestepSampler) else self.current
        self.current = MorphingTimestepSampler(
            source, target, steps=cfg["morph_updates"], curve="cosine",
            interpolation="quantile", start_update=self.update_step)
        self.current.set_optimizer_update_step(self.update_step)

    def _weighted_target(self, ratios) -> QuantileTableSampler:
        u = torch.linspace(0.0, 1.0, self._TABLE_POINTS, dtype=torch.float64)
        try:
            values = self.base.icdf(u).to(torch.float64)
        except NotImplementedError:
            # Deterministic fallback is only used for samplers without icdf.
            state = torch.random.get_rng_state()
            try:
                torch.manual_seed(0xA17E)
                values = torch.sort(self.base.sample(
                    self._TABLE_POINTS, torch.device("cpu")).to(torch.float64)).values
            finally:
                torch.random.set_rng_state(state)
        weights = []
        for value in values.tolist():
            index, _ = self._bin_and_x0_loss(float(value), 1.0)
            weights.append(ratios[index])
        weights_t = torch.tensor(weights, dtype=torch.float64)
        cdf = torch.cumsum(weights_t, dim=0)
        cdf = (cdf - cdf[0]) / max(1e-12, float(cdf[-1] - cdf[0]))
        q = torch.linspace(0.0, 1.0, self._TABLE_POINTS, dtype=torch.float64)
        hi = torch.searchsorted(cdf, q).clamp(1, self._TABLE_POINTS - 1)
        low = hi - 1
        span = (cdf[hi] - cdf[low]).clamp_min(1e-12)
        table = values[low] + (q - cdf[low]) / span * (values[hi] - values[low])
        return QuantileTableSampler(table.tolist(), self.min_timestep, self.max_timestep)

    def _base_bin_probabilities(self):
        points = 4096
        u = (torch.arange(points, dtype=torch.float64) + 0.5) / points
        try:
            values = self.base.icdf(u)
        except NotImplementedError:
            state = torch.random.get_rng_state()
            try:
                torch.manual_seed(0xA17E)
                values = self.base.sample(points, torch.device("cpu"))
            finally:
                torch.random.set_rng_state(state)
        counts = [0] * self.config["bins"]
        for value in values.tolist():
            index, _ = self._bin_and_x0_loss(float(value), 1.0)
            counts[index] += 1
        probabilities = [count / points for count in counts]
        # Empty extreme bins have zero base mass and cannot be manufactured by
        # a density ratio relative to that base law.
        total = sum(probabilities)
        return [value / total for value in probabilities]

    def _auto_eligible_bins(self):
        threshold = self.config["auto_min_bin_probability"]
        return [
            index for index, probability in enumerate(self.base_bin_probability)
            if probability > 0.0 and probability >= threshold
        ]

    def effective_sampler(self) -> TimestepSampler:
        if isinstance(self.current, MorphingTimestepSampler):
            return self.current.target if self.current.is_finished() else self.current.freeze()
        return self.current

    def state(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "config": dict(self.config),
            "convention": self.convention,
            "prediction_type": self.prediction_type,
            "latent_centered_second_moment": self.latent_centered_second_moment,
            "base": sampler_expr(self.base),
            "current": sampler_expr(self.current),
            "update_step": self.update_step,
            "last_control_update": self.last_control_update,
            "observations_since_control": self.observations_since_control,
            "counts": list(self.counts),
            "fast_ema": [None if not math.isfinite(v) else v for v in self.fast_ema],
            "slow_ema": [None if not math.isfinite(v) else v for v in self.slow_ema],
            "last_density_ratio": list(self.last_density_ratio),
            "base_bin_probability": list(self.base_bin_probability),
            "control_count": self.control_count,
            "auto_promoted": self.auto_promoted,
            "auto_promotion_update": self.auto_promotion_update,
            "last_action": self._last_action,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        if int(state.get("version", 0)) != 1:
            raise ValueError("unsupported adaptive timestep state version")
        if state.get("convention") != self.convention:
            raise ValueError("adaptive timestep convention changed across resume")
        if state.get("prediction_type") != self.prediction_type:
            raise ValueError("adaptive timestep prediction type changed across resume")
        saved_q = state.get("latent_centered_second_moment")
        if saved_q != self.latent_centered_second_moment:
            raise ValueError(
                "adaptive timestep latent centered second moment changed across resume"
            )
        saved_cfg = validate_adaptive_timestep_config(state.get("config"))
        for key in ("bins", "log_snr_min", "log_snr_max"):
            if saved_cfg[key] != self.config[key]:
                raise ValueError(f"adaptive timestep {key} changed across resume")
        if state.get("base") != sampler_expr(self.base):
            raise ValueError(
                "adaptive timestep base distribution changed across resume; disable "
                "adaptation for the transition")
        self.current = build_sampler_from_expr(state["current"])
        self.update_step = int(state.get("update_step", 0))
        self.last_control_update = int(state.get("last_control_update", -10**18))
        self.observations_since_control = int(state.get("observations_since_control", 0))
        for name in ("counts", "fast_ema", "slow_ema", "last_density_ratio"):
            values = list(state.get(name, getattr(self, name)))
            if len(values) != self.config["bins"]:
                raise ValueError(f"adaptive timestep {name} has wrong length")
            if name in ("fast_ema", "slow_ema"):
                values = [math.nan if value is None else float(value) for value in values]
            setattr(self, name, values)
        saved_probability = state.get("base_bin_probability")
        if saved_probability is not None:
            if len(saved_probability) != self.config["bins"]:
                raise ValueError("adaptive timestep base_bin_probability has wrong length")
            self.base_bin_probability = [float(value) for value in saved_probability]
        self.control_count = int(state.get("control_count", 0))
        saved_promoted = bool(state.get("auto_promoted", False))
        self.auto_promoted = (
            self.mode == "auto"
            and (saved_promoted or saved_cfg["mode"] == "bounded")
        )
        promotion = state.get("auto_promotion_update")
        self.auto_promotion_update = (
            None if not self.auto_promoted or promotion is None else int(promotion)
        )
        if self.mode == "observe" and isinstance(
                self.current, MorphingTimestepSampler):
            self.current = self.current.freeze()
        self._last_action = str(state.get("last_action", "resumed"))
        setter = getattr(self.current, "set_optimizer_update_step", None)
        if setter is not None:
            setter(self.update_step)

    def status(self) -> Dict[str, Any]:
        finite = [v for v in self.fast_ema if math.isfinite(v)]
        eligible = self._auto_eligible_bins()
        ready = sum(
            self.counts[index] >= self.config["auto_min_bin_observations"]
            for index in eligible
        )
        return {
            "mode": self.mode,
            "effective_mode": (
                "bounded" if self.mode == "bounded" or self.auto_promoted
                else "observe" if self.mode in ("observe", "auto") else "off"
            ),
            "action": self._last_action,
            "control_count": self.control_count,
            "observations_since_control": self.observations_since_control,
            "counts": list(self.counts),
            "fast_ema": [None if not math.isfinite(v) else v for v in self.fast_ema],
            "slow_ema": [None if not math.isfinite(v) else v for v in self.slow_ema],
            "density_ratio": list(self.last_density_ratio),
            "base_bin_probability": list(self.base_bin_probability),
            "auto_promoted": self.auto_promoted,
            "auto_promotion_update": self.auto_promotion_update,
            "auto_ready_bins": ready,
            "auto_required_bins": len(eligible),
            "mean_controller_loss": (sum(finite) / len(finite)) if finite else None,
            "mean_x0_loss": (
                (sum(finite) / len(finite))
                if finite and self.prediction_type == "flow_velocity" else None
            ),
        }
