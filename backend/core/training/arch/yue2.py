"""YuE2 Phase-A ABC score-planner training handler."""
from __future__ import annotations

from core.training.arch.base_arch import ArchHandler, PHASE2_PENDING, declare_adapter_capability, resolve_scope_csv
from core.training.components.wiring import YUE2_WIRING


class YuE2ArchHandler(ArchHandler):
    name = "yue2"
    wiring = YUE2_WIRING
    adapter_capability = declare_adapter_capability(
        "yue2", additive_family=False, initial_dora="refused",
        additive_reason=PHASE2_PENDING,
        quantized_base_reason="YuE2 Phase A supports ordinary LoRA over its frozen ConvRot INT8 base",
    )
    pixel_align = 1
    consumes_reconstruction_loss_weight = False

    def lora_adapter_class(self):
        from core.training.adapters import YuE2LoRAAdapter
        return YuE2LoRAAdapter

    def lora_adapter_kwargs(self, trainer):
        objective = str((trainer.config or {}).get("yue2_training_objective", "abc_ar"))
        scope_csv = resolve_scope_csv(trainer, "yue2_lora_scope", "attention")
        wanted = {part.strip() for part in scope_csv.split(",") if part.strip()}
        unknown = wanted - {"attention", "mlp"}
        if unknown:
            raise ValueError(f"Unknown YuE2 LoRA scope(s): {sorted(unknown)}")
        return {"objective": objective, "scope": {
            "attention": "attention" in wanted,
            "mlp": "mlp" in wanted,
        }}

    def load_components(self, trainer) -> None:
        from core.training.ops.yue2_ops import load_components
        load_components(trainer)

    def setup_block_swap(self, trainer) -> None:
        if int(getattr(trainer, "blocks_to_swap", 0) or 0):
            raise ValueError("YuE2 Phase-A training does not support block swap")

    def depth_blocks(self, trainer):
        model = getattr(getattr(trainer, "transformer", None), "model", None)
        return getattr(model, "layers", None)

    def setup_attention_backend(self, trainer) -> None:
        if getattr(trainer, "attention_backend", "native") != "native":
            raise ValueError("YuE2 training currently supports native SDPA attention only")

    def encode_prompt(self, trainer, prompt, *, requires_grad: bool = False):
        raise NotImplementedError("YuE2 builds native token sequences inside its training loop")

    def vae_encode(self, trainer, image_tensor, **kwargs):
        raise NotImplementedError("YuE2 abc_ar does not consume pixels or VAE latents")

    def vae_decode(self, trainer, latents, *, latent_h=None, latent_w=None):
        raise NotImplementedError("YuE2 Phase-A training samples are not implemented")

    def train_step(self, trainer, ctx):
        raise NotImplementedError("YuE2 uses its token-native training loop")

    def sample(self, trainer, sample_ctx):
        return None
