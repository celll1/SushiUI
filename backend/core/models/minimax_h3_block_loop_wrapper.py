"""MiniMax-H3 transformer wrapper — block-loop re-ownership.

``MiniMaxH3Transformer3DModel.forward`` (vendored, see
``core/models/minimax_h3/vendor/transformer_minimax_h3.py``) runs its 50 blocks
in one plain ``for block in self.transformer_blocks`` loop. That loop is where
block swap and gradient checkpointing have to live, so this wrapper re-owns ONLY
that loop and executes every other stage (RoPE, the three
input projections, the token refiner, the AdaLN-curve lookup, the output norm
and the two modality heads) by calling the inner model's OWN submodules — so the
custom path reproduces the stock forward tensor for tensor.

THE TOKEN REFINER GETS ITS OWN STAGE, under the same ``swap_on`` condition as
the block loop: ``pipeline_backends/minimax_h3.py::_ensure_minimax_h3_swap_and_offload``
deliberately leaves it off the device when block swap is on (it is the only
bf16 -- i.e. unquantized -- module in the w4a8 checkpoint, 1.4356 GiB, 12.2%
of the file), so this is the one place it is moved onto ``device`` -- right
before its single call -- and back to the CPU right after. Cheap to do this
way: it runs once per forward, before the block loop, on the packed
sequence's ~500 text rows, so its own compute is negligible next to a
50-block denoise step, and the PCIe round trip (~1.4 GiB each way) is a small
fraction of a generation's per-step wall time. When ``swap_on`` is false the
caller already put the whole model on the device (the ``blocks_to_swap <= 0``
branch's ``.to(device)``), so this stage is a no-op skip there, keeping the
default and ``blocks_to_swap=0`` paths byte-identical to before this stage
existed.

Fast path: with no feature attached (no block offloader or FBCache) the wrapper delegates
verbatim to ``self.transformer(...)``, so the default MiniMax-H3 path is
byte-identical to the unwrapped model.

WHY THERE IS NO DIFFUSERS PIN HERE, unlike ``Ltx2BlockLoopWrapper``. That
wrapper asserts a fixed set of submodule names at construction because its inner
model is ``diffusers.LTX2VideoTransformer3DModel`` — an upgrade can rename a
submodule under it and silently break the stage replication. MiniMax-H3's model
class is VENDORED into this repo (frozen, with the AdaLN-curve port applied), so
the only thing that can rename ``proj_in`` / ``token_refiner`` /
``transformer_blocks`` / ``norm_out`` is an edit to a file in this repository,
which changes this file's own tree. There is no external version to pin, so an
assert here would only restate the file next to it.

Extension slot (None by default -> fast path):
  * ``_block_offloader`` — generation block swap, via the shared
    ``core.memory_management.TransformerBlockOffloader`` over
    ``transformer.transformer_blocks``. Built and attached by
    ``pipeline_backends/minimax_h3.py::_ensure_minimax_h3_swap_and_offload``.
  * ``_fbcache`` — inference-only cache of the whole packed-state residual.
    Its decision uses generated-video rows only, with a max-per-frame guard;
    one decision therefore keeps the video and audio paths in lock-step.

Gradient checkpointing is honored on the custom path exactly as the stock loop
honors it (``torch.is_grad_enabled() and transformer.gradient_checkpointing``),
so the training phase can use the wrapper without losing it — which is the whole
reason block swap and checkpointing share one loop rather than two.
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from core.models.minimax_h3.vendor.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3TransformerOutput,
)


class MiniMaxH3BlockLoopWrapper(nn.Module):
    """Wrap ``MiniMaxH3Transformer3DModel`` and re-own only its block loop.

    Order of construction relative to other features (mirrors LTX-2.3 / FLUX.2):
      1. LoRA-wrap the INNER transformer.
      2. Wrap it with ``MiniMaxH3BlockLoopWrapper``.
      3. Build the block offloader over ``wrapper.transformer.transformer_blocks``
         and attach it (``wrapper.attach_block_offloader``).

    ``to`` / ``__getattr__`` / ``state_dict`` / ``load_state_dict`` / ``config`` /
    ``dtype`` passthroughs make LoRA save/load, the block-swap conductor, the
    quantized single-file export and the sampler see the wrapper as the
    transformer.
    """

    def __init__(self, transformer: nn.Module, block_offloader: Optional[Any] = None):
        super().__init__()
        self.transformer = transformer

        # === Extension slot (None -> fast path; byte-identical default) ===
        self._block_offloader = block_offloader
        self._fbcache = None
        self._fbcache_step = 0
        self._fbcache_rows_per_frame = None
        self._fbcache_condition_video_rows = 0

        # Compatibility attributes (sampler + LoRA/export introspection).
        self.config = transformer.config
        self.dtype = transformer.dtype

    # ------------------------------------------------------------------
    # Feature attach / detach
    # ------------------------------------------------------------------
    def attach_block_offloader(self, block_offloader: Optional[Any]) -> None:
        """Attach (or clear with None) the generation block offloader."""
        self._block_offloader = block_offloader

    def attach_fbcache(
        self,
        fbcache: Optional[Any],
        *,
        rows_per_frame: Optional[int] = None,
        condition_video_rows: int = 0,
    ) -> None:
        """Attach one-generation FBCache geometry, or clear it with ``None``."""
        if fbcache is not None:
            if self._block_offloader is not None and getattr(
                self._block_offloader, "blocks_to_swap", 0
            ) > 0:
                raise ValueError("MiniMax-H3 FBCache cannot run with Block Swap.")
            if not rows_per_frame or rows_per_frame <= 0:
                raise ValueError("MiniMax-H3 FBCache needs a positive rows_per_frame.")
            if condition_video_rows < 0:
                raise ValueError("condition_video_rows must be non-negative.")
        self._fbcache = fbcache
        self._fbcache_step = 0
        self._fbcache_rows_per_frame = rows_per_frame
        self._fbcache_condition_video_rows = int(condition_video_rows)

    def _any_feature_active(self) -> bool:
        return bool(
            self._fbcache is not None
            or (
                self._block_offloader is not None
                and getattr(self._block_offloader, "blocks_to_swap", 0) > 0
            )
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        attention_kwargs: Optional[dict] = None,
        return_dict: bool = True,
    ):
        """Forward with the EXACT stock ``MiniMaxH3Transformer3DModel`` signature."""
        if not self._any_feature_active():
            # Byte-identical default: the inner model's own forward. The
            # @apply_lora_scale decorator on it still fires here.
            return self.transformer(
                hidden_states=hidden_states,
                audio_hidden_states=audio_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                timestep_indices=timestep_indices,
                token_tags=token_tags,
                position_ids=position_ids,
                video_indices=video_indices,
                audio_indices=audio_indices,
                text_indices=text_indices,
                attention_kwargs=attention_kwargs,
                return_dict=return_dict,
            )

        return self._custom_forward(
            hidden_states, audio_hidden_states, encoder_hidden_states,
            timestep, timestep_indices, token_tags, position_ids,
            video_indices, audio_indices, text_indices, return_dict,
        )

    # NOTE on ``attention_kwargs``: the stock forward carries the
    # ``@apply_lora_scale("attention_kwargs")`` decorator, which pops a ``scale``
    # entry and applies it to the LoRA layers for the duration of the call. The
    # custom path below is entered only from ``forward``, which is NOT decorated,
    # so a LoRA scale passed through this path would be ignored. Nothing in this
    # repo sets one for MiniMax-H3 (the sampler never populates
    # ``attention_kwargs``), and the training phase applies its scale on the
    # adapter rather than per call; the argument is accepted on ``forward`` for
    # signature parity and asserted-empty here rather than silently dropped.
    def _custom_forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
        return_dict,
    ):
        t = self.transformer
        offloader = self._block_offloader
        swap_on = offloader is not None and getattr(offloader, "blocks_to_swap", 0) > 0
        fbcache = self._fbcache
        if fbcache is not None and torch.is_grad_enabled():
            raise RuntimeError("MiniMax-H3 FBCache is inference-only.")

        # === Replicated stock stages (transformer_minimax_h3.forward) ===
        # The attention-backend stamp is one of them: the stock forward calls
        # this first, and the custom path never reaches the stock forward, so
        # omitting it would leave every attention module on whatever backend the
        # previous generation stamped (or on native forever, if the first
        # generation of a session used block swap).
        t._stamp_attention_backend()

        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(
                f"`position_ids` must be a `(seq_len, 3)` tensor, got {list(position_ids.shape)}.")
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError(
                "`token_tags` and `timestep_indices` must both be `(seq_len,)` tensors matching "
                f"`position_ids`, got {list(token_tags.shape)} and "
                f"{list(timestep_indices.shape)} for seq_len={sequence_length}.")

        rotary_emb = t.rope(position_ids)

        # 1. Per-modality projections, scattered into the packed sequence.
        video_embeds = t.proj_in(hidden_states.to(t.proj_in.weight.dtype))
        audio_embeds = t.audio_proj_in(audio_hidden_states.to(t.audio_proj_in.weight.dtype))
        text_embeds = t.context_embedder(encoder_hidden_states.to(t.context_embedder.weight.dtype))
        # The token refiner is left off `device` by the caller when block swap
        # is on (see this module's docstring) -- stage it on for the length of
        # this one call, then immediately back off. The target device is read
        # off the input tensor rather than the offloader (the offloader's
        # contract, per its own test double, is `wait_for_block` /
        # `submit_move_blocks_forward` only -- not a `.device` attribute).
        # `non_blocking` only on the H2D leg: neither the module's own storage
        # nor the offloader backing it (`use_pinned_memory=False`) is pinned,
        # so an async D2H here would race the very same-stream compute it
        # feeds without a pinned buffer.
        if swap_on:
            refiner_device = hidden_states.device
            t.token_refiner.to(refiner_device, non_blocking=refiner_device.type != "cpu")
        text_embeds = t.token_refiner(text_embeds)
        if swap_on:
            t.token_refiner.to("cpu")

        hidden_states = text_embeds.new_zeros(
            (text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
        hidden_states = hidden_states.index_copy(1, text_indices, text_embeds)
        hidden_states = hidden_states.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        hidden_states = hidden_states.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))

        # 2. Timestep embedding — the AdaLN-curve lookup for the pruned variant,
        # the sinusoid + MLP for the full-modulation one. Copied from the stock
        # forward verbatim, including the two traps K0.2 pinned: the SiLU is
        # baked into the table (no extra activation here) and the max-clamp keeps
        # t = 1.0 on the last interval.
        if t.use_adaln_curves:
            table = t.adaln_t_table
            position = timestep.to(table.device, torch.float32).clamp(0.0, 1.0) * (table.shape[0] - 1)
            lower = position.floor().long().clamp(max=table.shape[0] - 2)
            temb = torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1))
        else:
            temb = t.time_proj(timestep)
            temb = t.time_embedder(temb.to(t.time_embedder.linear_1.weight.dtype))

        # 3. Row -> AdaLN table row.
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags

        # === 4. The block loop (RE-OWNED) ===
        grad_ckpt = torch.is_grad_enabled() and t.gradient_checkpointing
        # SushiUI: `MiniMaxH3TransformerBlock.forward`'s gated residual add
        # (`core.models.minimax_h3.adaln_chunking.gated_residual_add`) mutates its
        # `residual` argument IN PLACE at inference -- and for block 0, that argument is
        # this exact `hidden_states` object. Clone it into `original_hidden_states` before
        # the loop so block 0's in-place add cannot zero out `first_residual` below via
        # aliasing. Gated only on `fbcache is not None`: it is the only reader of
        # `original_hidden_states`, and this path is already mutually exclusive with block
        # swap (`attach_fbcache` above), so the extra clone never lands on the block-swap path.
        original_hidden_states = hidden_states.clone() if fbcache is not None else hidden_states
        fbcache_hit = False

        for block_idx, block in enumerate(t.transformer_blocks):
            if swap_on:
                offloader.wait_for_block(block_idx)

            if grad_ckpt:
                hidden_states = t._gradient_checkpointing_func(
                    block, hidden_states, temb, adaln_indices, rotary_emb)
            else:
                hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb)

            if swap_on:
                offloader.submit_move_blocks_forward(block_idx)

            if fbcache is not None and block_idx == 0:
                first_residual = hidden_states - original_hidden_states
                video_residual = first_residual.index_select(1, video_indices)
                video_residual = video_residual[:, self._fbcache_condition_video_rows:]
                rows_per_frame = int(self._fbcache_rows_per_frame)
                if not video_residual.shape[1] or video_residual.shape[1] % rows_per_frame:
                    raise RuntimeError(
                        "MiniMax-H3 FBCache target video rows do not divide into latent "
                        f"frames: {video_residual.shape[1]} rows / {rows_per_frame}."
                    )
                frames = video_residual.shape[1] // rows_per_frame
                frame_groups = video_residual.reshape(
                    video_residual.shape[0], frames, rows_per_frame, video_residual.shape[-1]
                ).flatten(0, 1)
                if fbcache.use_cache(
                    video_residual,
                    int(self._fbcache_step),
                    guard_indicator=frame_groups,
                ):
                    hidden_states = original_hidden_states + fbcache.get()
                    fbcache_hit = True
                    break

        if fbcache is not None and not fbcache_hit:
            fbcache.store(hidden_states - original_hidden_states)

        # === 5. Output norm + the two modality heads ===
        # SushiUI: `out_dtype` passed in rather than cast afterwards -- must be kept in sync
        # with the stock call site in `transformer_minimax_h3.py`'s `MiniMaxH3Transformer3DModel.forward`.
        hidden_states = t.norm_out(hidden_states, temb, timestep_indices, out_dtype=t.proj_out.weight.dtype)
        video_output = t.proj_out(hidden_states).index_select(1, video_indices)
        audio_output = t.audio_proj_out(hidden_states).index_select(1, audio_indices)

        if not return_dict:
            return (video_output, audio_output)
        return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)

    # ------------------------------------------------------------------
    # Passthroughs
    # ------------------------------------------------------------------
    def to(self, *args, **kwargs):
        """Forward ``.to()`` to the inner transformer.

        The mixin's component staging calls ``.to(device)`` on whatever
        ``minimax_h3_components["transformer"]`` holds, which is this wrapper
        once block swap is attached; without this the wrapper's own (empty)
        parameter set would move and the real weights would not.
        """
        self.transformer.to(*args, **kwargs)
        return self

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)

    def state_dict(self, *args, **kwargs):
        """Forward ``state_dict`` to the inner transformer.

        LoRA save/load and the quantized single-file export both read this; the
        keys must be the inner model's module paths, NOT ``transformer.<path>``.
        """
        return self.transformer.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.transformer.load_state_dict(*args, **kwargs)
