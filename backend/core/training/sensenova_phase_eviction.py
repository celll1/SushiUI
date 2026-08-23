"""Training-only SenseNova MoT phase eviction state machine."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
from torch import nn

from core.models.sensenova.mot_cpu_staging import stage_modules_to_pinned_cpu
from core.models.sensenova.mot_weight_selector import select_mot_weight_modules

_PIN_FAILURE_MESSAGE = (
    "[SenseNova] Training MoT eviction could not pin CPU staging "
    "memory ({exc}); continuing with blocking pageable copies."
)


def _move_modules_to_cpu(
    modules: Iterable[nn.Module], *, warn_once: Dict[str, bool]
) -> None:
    stage_modules_to_pinned_cpu(
        modules, warn_once=warn_once, warn_message=_PIN_FAILURE_MESSAGE
    )


def _move_modules_to_device(modules: Iterable[nn.Module], device: Any) -> None:
    for module in modules:
        for parameter in module._parameters.values():
            if parameter is not None:
                parameter.data = parameter.data.to(device)
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and name not in module._non_persistent_buffers_set:
                module._buffers[name] = buffer.to(device)


class SenseNovaTrainingPhaseEvictor:
    """Keep only the phase-active MoT half resident while training LoRA."""

    def __init__(self, transformer: nn.Module, device: Any):
        selection = select_mot_weight_modules(
            transformer, require_exact_symmetry=True
        )
        self._gen_modules = selection.gen_modules
        self._und_modules = selection.und_modules
        self.transformer = transformer
        self.device = device
        self.state = "full"
        self._warn_once: Dict[str, bool] = {}

    def _best_effort_cpu(self) -> Exception | None:
        first_error = None
        for module in (*self._gen_modules, *self._und_modules):
            try:
                _move_modules_to_cpu((module,), warn_once=self._warn_once)
            except Exception as exc:
                first_error = first_error or exc
        try:
            self.transformer.to("cpu")
        except Exception as exc:
            first_error = first_error or exc
        return first_error

    def _transition(self, operations, next_state: str) -> None:
        if self.state == "failed":
            raise RuntimeError("SenseNova eviction cannot reuse a failed transfer state")
        try:
            for operation, modules in operations:
                if operation == "d2h":
                    _move_modules_to_cpu(modules, warn_once=self._warn_once)
                else:
                    _move_modules_to_device(modules, self.device)
        except Exception:
            self.state = "failed"
            self._best_effort_cpu()
            raise
        self.state = next_state

    def enter_prefix(self) -> None:
        if self.state == "failed":
            raise RuntimeError("SenseNova eviction cannot reuse a failed transfer state")
        if self.state == "prefix":
            return
        if self.state == "full":
            operations = (("d2h", self._gen_modules),)
        elif self.state == "denoise":
            operations = (
                ("d2h", self._gen_modules),
                ("h2d", self._und_modules),
            )
        else:
            raise RuntimeError(f"Invalid SenseNova eviction state: {self.state}")
        self._transition(operations, "prefix")

    def enter_denoise(self) -> None:
        if self.state == "failed":
            raise RuntimeError("SenseNova eviction cannot reuse a failed transfer state")
        if self.state == "denoise":
            return
        if self.state != "prefix":
            raise RuntimeError("SenseNova denoise phase requires a completed prefix phase")
        self._transition(
            (
                ("d2h", self._und_modules),
                ("h2d", self._gen_modules),
            ),
            "denoise",
        )

    def assert_generation_resident(self) -> None:
        if self.state != "denoise":
            raise RuntimeError(
                f"SenseNova generation work requires denoise state, got {self.state}"
            )
        expected = torch.device(self.device)

        def on_expected_device(tensor) -> bool:
            return tensor.device.type == expected.type and (
                expected.index is None or tensor.device.index == expected.index
            )

        for module in self._gen_modules:
            for parameter in module._parameters.values():
                if parameter is None:
                    continue
                if not on_expected_device(parameter):
                    raise RuntimeError("SenseNova generation parameter is not GPU-resident")
                if parameter.grad is not None and parameter.grad.device != parameter.device:
                    raise RuntimeError("SenseNova generation gradient is on the wrong device")
            for name, buffer in module._buffers.items():
                if (
                    buffer is not None
                    and name not in module._non_persistent_buffers_set
                    and not on_expected_device(buffer)
                ):
                    raise RuntimeError("SenseNova generation buffer is not GPU-resident")

    def teardown(self) -> None:
        if self.state == "closed":
            return
        if self.state not in ("full", "prefix", "denoise", "failed"):
            raise RuntimeError(f"Invalid SenseNova eviction state: {self.state}")
        error = self._best_effort_cpu()
        self.state = "closed"
        if error is not None:
            raise RuntimeError(
                "SenseNova eviction teardown could not normalize all weights to CPU"
            ) from error


def install_training_phase_eviction(trainer: Any) -> SenseNovaTrainingPhaseEvictor:
    evictor = SenseNovaTrainingPhaseEvictor(trainer.transformer, trainer.device)
    trainer.sensenova_phase_evictor = evictor
    return evictor
