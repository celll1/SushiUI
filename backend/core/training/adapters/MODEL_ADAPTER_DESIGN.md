# Model Adapter Architecture Design

## Overview

This document describes the modular architecture for supporting multiple model types (SD1.5, SDXL, Z-Image) in LoRA and Full Parameter training.

## Problem

The monolithic `LoRATrainer` and `FullParameterTrainer` classes were becoming too large and difficult to maintain, with model-specific code scattered throughout.

**Issues with previous implementation**:
- Z-Image support was lost during full rewrite
- SDXL-specific code (text encoder layer selection, pooled embeddings, time_ids) was mixed with SD1.5 code
- Difficult to add new model architectures
- Hard to track which fixes apply to which models

## Solution: Model Adapter Pattern

Create separate adapter classes for each model architecture:

```
BaseAdapter (ABC)
├── SD15Adapter
├── SDXLAdapter
└── ZImageAdapter
```

Each adapter implements model-specific logic:
- LoRA layer injection
- Text encoder handling
- Checkpoint saving/loading
- Parameter grouping

## Learning-rate resolution (all adapters)

A param group's `lr` comes from `base_adapter.resolve_component_lr(trainer,
*attr_names)`. It returns the first attribute that is **not None** — an
explicit `0.0` is a configured rate, not "unset" — and falls back to
`trainer.learning_rate`; it raises when neither exists rather than inventing a
rate. `BaseTrainer.__init__` already derives `unet_lr` / `text_encoder*_lr` from
`learning_rate` when the config omits them, so the resolver's own fallback is
the last line rather than the usual path.

Do not write `getattr(trainer, "unet_lr", None) or 1e-4`: that idiom replaced a
configured `0.0` with a literal, and the literal differed per adapter (1e-4,
1e-5, 1e-6) so the same config trained at a different rate depending on the
architecture. `BaseTrainer._build_component_lr_list` resolves all seven of its
entries the same way.

### What a resume writes back

`setup_optimizer` ends by calling `_record_configured_group_lrs`, which
snapshots each live param group's BASE lr (`initial_lr`, since the scheduler
may already have scaled `lr`) into `_configured_group_lrs`. That snapshot is
the adapter's own description — group order and every per-component factor
(`anima_*_lr_factor`, `lens_*_lr_factor`, `minit2i_lr_factor`, the REPA
projector, the SDXL custom-TE bridge) included — so it cannot drift from what
trains. `_reassert_config_lr_on_resume` writes it back index-for-index.

`_build_component_lr_list` re-derives a description from trainer attributes
instead. It is EMPTY on every DiT architecture (`self.unet is None`; only the
SenseNova / ControlNet / VE branches fill it) and can be non-empty yet
misaligned (a MiniT2I/Flux2/Z-Image run with `train_text_encoder` describes
only `TE1` while group 0 is the transformer). It is now a fallback used only
when it matches the group count, and both consumers refuse to write a
description that does not: an unusable one emits
`component_lr_resume_unavailable` and leaves the checkpoint's rates in place
rather than broadcasting a scalar over every group. If a group carries a
`"name"` key (Lens does), that name labels it in the log.

`BaseTrainer._report_effective_component_lrs`, called last in
`setup_optimizer`, warns on the training-log channel when a group's base rate
is 0 (`component_lr_zero`) or when `num_optimizer_groups > 0` rebuilt the
optimizers from a flat parameter list at the base LR and discarded the
per-component rates (`component_lr_flattened`). It compares base rates, not
`group['lr']`, because a warmup lambda has already scaled the latter to 0.0 at
step 0. Its index-wise `component_lr_mismatch` check is a consistency check
between two producers of the same description, not an independent measurement,
and it cannot run where `_build_component_lr_list` is empty (every DiT
architecture); it prints a NOTE saying so instead of staying silent.

## Architecture

### Base Adapter Interface

```python
# backend/core/training/adapters/base_adapter.py

class BaseAdapter(ABC):
    """Abstract base class for model-specific adapters."""

    @abstractmethod
    def apply_lora_to_unet(self) -> int:
        """Apply LoRA to U-Net/Transformer. Returns number of layers injected."""
        pass

    @abstractmethod
    def apply_lora_to_text_encoders(self) -> int:
        """Apply LoRA to text encoder(s). Returns number of layers injected."""
        pass

    @abstractmethod
    def setup_trainable_parameters(self, lora_layers: Dict) -> List[Dict]:
        """Collect trainable parameters with per-component learning rates."""
        pass

    @abstractmethod
    def save_checkpoint(self, lora_layers: Dict, step: int, epoch: int, output_path: Path):
        """Save checkpoint in model-specific format."""
        pass
```

### SD1.5 Adapter

**Model characteristics**:
- Single text encoder (CLIP ViT-L/14)
- U-Net with Transformer2DModel blocks
- Simple text embeddings (no pooled output)

**LoRA targets**:
- U-Net: All `Transformer2DModel` attention layers
- Text Encoder: Optional (usually not trained)

**Key differences from SDXL**:
- Single text encoder (no TE2)
- No pooled embeddings
- No time_ids
- Simpler key naming convention

### SDXL Adapter

**Model characteristics**:
- Dual text encoders (TE1: CLIP ViT-L, TE2: OpenCLIP ViT-bigG)
- U-Net with 11 Transformer2DModel blocks
- Pooled embeddings from TE2
- Time IDs for micro-conditioning

**LoRA targets**:
- U-Net: All `Transformer2DModel` attention layers (11 blocks)
- Text Encoder 1: All MLP layers (12 layers × 2 = 24 LoRA layers)
- Text Encoder 2: All MLP layers (32 layers × 2 = 64 LoRA layers)

**Critical fixes from rewrite**:
1. **Text Encoder layer selection**: All layers, not specific layers
2. **TE2 penultimate layer**: `hidden_states[-2]` (NOT final layer)
3. **EOS token pooling workaround**: Manually pool last token for TI compatibility
4. **Component-specific learning rates**: te1_lr, te2_lr, unet_lr

**Old implementation issues** (fixed in new implementation):
- Incorrect TE1/TE2 layer selection
- Wrong pooled embedding extraction
- Missing EOS token workaround

### Z-Image Adapter

**Model characteristics**:
- Qwen3 text encoder (AutoModelForCausalLM)
- ZImageTransformer2DModel (flow matching, not DDPM)
- Chat template for text encoding
- BatchedZImageWrapper for batching
- Frame dimension: `[B, C, H, W]` → `[B, C, 1, H, W]`

**LoRA targets**:
- Transformer: `ZImageAttention` modules (to_q, to_k, to_v, to_out[0])
- Text Encoder: Frozen (no LoRA)

**Key implementation details**:
```python
# Find ZImageAttention modules
target_transformer = (
    self.transformer.transformer
    if hasattr(self.transformer, 'transformer')
    else self.transformer
)

for name, module in target_transformer.named_modules():
    if module.__class__.__name__ == "ZImageAttention":
        # Apply LoRA to: to_q, to_k, to_v
        # to_out is ModuleList, inject into to_out[0]
```

**Text encoding**:
```python
# Format with Qwen chat template
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)

# Encode with penultimate layer (like SDXL TE2)
encoder_output = text_encoder(input_ids, output_hidden_states=True)
prompt_embeds = encoder_output.hidden_states[-2]  # Penultimate layer

return prompt_embeds, attention_mask  # Different from SD/SDXL
```

**Training step differences**:
- Flow matching (not DDPM): `noise_process="flow"`, `prediction_target="velocity"`
- Timesteps from `[0, 1]` (not `[0, 1000]`)
- Frame dimension handling: Add/remove dimension before/after forward pass
- No Min-SNR weighting (uniform timestep distribution)

**Old implementation reference**:
- `_apply_lora_zimage()` (Line 303-371 in old lora_trainer.py)
- `encode_prompt_zimage()` (Line 1517-1577 in old base_trainer.py)
- `train_step_zimage()` (Line 2057-2200 in old base_trainer.py)

## Integration with Trainers

### LoRATrainer Usage

```python
class LoRATrainer(BaseTrainer):
    def __init__(self, lora_rank, lora_alpha, train_unet, train_text_encoder, **kwargs):
        super().__init__(**kwargs)

        # Detect model type and create adapter
        if self.is_zimage:
            self.adapter = ZImageAdapter(self, lora_rank, lora_alpha)
        elif self.is_sdxl:
            self.adapter = SDXLAdapter(self, lora_rank, lora_alpha)
        else:
            self.adapter = SD15Adapter(self, lora_rank, lora_alpha)

        # Apply LoRA using adapter
        self.lora_layers = {}
        if train_unet:
            unet_count = self.adapter.apply_lora_to_unet(self.lora_layers)
        if train_text_encoder:
            te_count = self.adapter.apply_lora_to_text_encoders(self.lora_layers)

    def setup_trainable_parameters(self):
        return self.adapter.setup_trainable_parameters(self.lora_layers)

    def save_checkpoint(self, step, epoch):
        checkpoint_path = self.output_dir / f"{self.run_name}_step_{step:06d}.safetensors"
        self.adapter.save_checkpoint(self.lora_layers, step, epoch, checkpoint_path)
```

### FullParameterTrainer Usage

Similar pattern, but adapters handle full parameter training instead of LoRA.

## Benefits

1. **Separation of Concerns**: Model-specific logic isolated in adapters
2. **Maintainability**: Easy to find and fix model-specific bugs
3. **Extensibility**: Add new models by creating new adapters
4. **Testability**: Test each adapter independently
5. **Reduced Code Duplication**: Shared logic in base classes
6. **Clear Documentation**: Each adapter documents its model's specifics

## Migration Plan

1. ✅ Design adapter architecture (this document)
2. Create `base_adapter.py` with abstract interface
3. Create `sd15_adapter.py` (simplest, start here)
4. Create `sdxl_adapter.py` (copy from new implementation)
5. Create `zimage_adapter.py` (restore from old implementation)
6. Refactor `LoRATrainer` to use adapters
7. Refactor `FullParameterTrainer` to use adapters
8. Test each model type independently
9. Remove old monolithic code

## Reference Commits

- **SDXL rewrite**: Current HEAD (2026-01-04)
  - Fixed text encoder layer selection
  - Fixed pooled embedding extraction
  - Added EOS token workaround

- **Old Z-Image implementation**: `729ee38`
  - `_apply_lora_zimage()` in lora_trainer.py
  - `encode_prompt_zimage()` in base_trainer.py
  - `train_step_zimage()` in base_trainer.py

## Notes

- BaseTrainer already has model detection (`self.is_sdxl`, `self.is_zimage`)
- BaseTrainer already has `encode_prompt_zimage()` and `train_step_zimage()`
- Only LoRA application logic needs to be restored for Z-Image
- Full Parameter Trainer will also benefit from this architecture
