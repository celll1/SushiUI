"""Adapter leaf layers: the wrappers that carry the trainable branch.

Two algebras live here already, and the eventual ``AdapterLayer`` protocol has
to accommodate both: the stock layer relies on an ambient ``torch.autocast`` to
reconcile its fp32 masters with a bf16 activation, the MiniMax-H3 subclass
casts per call because that architecture's forward runs without autocast.

Moved verbatim from ``core.training.adapters.{sd15,minimax_h3}_adapter`` in
Phase 1 step 1; both old paths re-export these classes.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinearLayer(nn.Module):
    """
    LoRA layer for Linear modules.

    Formula: output = original_output + (lora_up(lora_down(x))) * scale
    """

    def __init__(
        self,
        original_module: nn.Linear,
        rank: int,
        alpha: float,
        lora_name: str,
        lora_dtype: torch.dtype = torch.float32,
    ):
        """Initialize LoRA layer."""
        super().__init__()
        self.original_module = original_module
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.lora_name = lora_name
        self.lora_dtype = lora_dtype

        in_features = original_module.in_features
        out_features = original_module.out_features

        # Freeze original weights
        self.original_module.requires_grad_(False)

        # LoRA matrices (no bias)
        # Use lora_dtype for LoRA weights (can be different from main model dtype)
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        # Initialize: Kaiming uniform for down, zeros for up
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        # Move to same device as original, but use lora_dtype
        device = original_module.weight.device
        self.lora_down.to(device=device, dtype=lora_dtype)
        self.lora_up.to(device=device, dtype=lora_dtype)

    @property
    def weight(self):
        """Expose the wrapped Linear's weight so callers that introspect
        `.weight` (e.g. T5's DenseGatedActDense dtype check) keep working when a
        Linear is wrapped. Read-only delegate; not a trained parameter here."""
        return self.original_module.weight

    @property
    def bias(self):
        return getattr(self.original_module, "bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.

        Uses autocast to automatically handle mixed precision:
        - LoRA weights (fp32) are automatically converted to training dtype during forward
        - Gradients flow back to fp32 master weights correctly
        - GradScaler handles gradient scaling for fp16/bf16 training
        """
        org_out = self.original_module(x)

        # LoRA computation (autocast will handle dtype conversion automatically)
        # If we're in an autocast context (training_dtype), this will run in that dtype
        # Gradients will still flow back to fp32 master weights correctly
        lora_out = self.lora_up(self.lora_down(x))

        return org_out + lora_out * self.scale


class MiniMaxH3LoRALinearLayer(LoRALinearLayer):
    """``LoRALinearLayer`` with the LoRA branch cast to the ACTIVATION dtype.

    MiniMax-H3's training forward runs WITHOUT ``torch.autocast``: the vendored
    transformer owns its own mixed-precision policy (fp32 I/O heads and AdaLN
    projections, bf16 block stack, each activation aligned to its projection's
    parameter dtype), and an autocast context would override those casts and make
    training a different function from generation.

    The stock layer relies on autocast to reconcile its fp32 master weights with
    a bf16 activation. Without autocast that is not a style difference, it is a
    ``RuntimeError`` on the first ``F.linear`` -- and, if the branch happened to
    be built in the activation dtype instead, a silent loss of the fp32 master.
    So the masters stay fp32 and are cast per call; the gradient flows back
    through the cast to the fp32 parameters unchanged. This is exactly the LoRA
    shape Phase 0T measured (bitwise save->reload, 600/600 tensors receiving
    finite gradients).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        org_out = self.original_module(x)
        down = F.linear(x, self.lora_down.weight.to(x.dtype))
        up = F.linear(down, self.lora_up.weight.to(x.dtype))
        return org_out + up * self.scale
