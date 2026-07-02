"""NAG (Normalized Attention Guidance) for the MiniT2I MM-JiT transformer.

MiniT2I is a pixel-space dual-stream MM-DiT (``DoubleStreamDiTBlock``): every block
runs joint attention over the concatenated ``[text; image]`` sequence, so the image
query tokens attend to the text context. NAG extrapolates the IMAGE attention output
against the POSITIVE text context away from a separate NAG-NEGATIVE text context, in
attention-output space (see core.inference.nag_dit.nag_guidance).

Mechanism (mirrors the FLUX.2 reference, adapted to MiniT2I's batched CFG):
  MiniT2I CFG batches image=[cond, uncond] (batch 2, k=1) and text=[pos, u_text]
  (see minit2i_pipeline_ops._predict_x0_cfg). Because each batch element attends only
  within its own [text; image] sequence, we can obtain the pos- and nag_neg- attention
  outputs for the cond image simply by adding a third batch group whose text is the
  nag_neg context and whose image is the SAME cond image:

      image batch : [x_cond, x_uncond, x_cond]   (3k)
      text  batch : [t_pos,  t_u,      t_nag]     (3k)

  group 0: x_cond attends to [t_pos ; x_cond]  -> z_pos  (image tokens)
  group 2: x_cond attends to [t_nag ; x_cond]  -> z_neg  (image tokens)
  NAG guides group 0's image attention output; group 2 is a scratch group whose image
  output is discarded. After all blocks, groups [0, 1] = [cond, uncond] are returned so
  the caller's CFG combine ``uncond + (cond-uncond)*cfg_scale`` is unchanged.

  Distilled / no-CFG path (cfg not active this step): image batch [x] (1k), text
  [pos]. NAG doubles to image=[x, x], text=[pos, nag_neg]; group 0 guided, group 1
  discarded; group [0] returned.

Everything is OFF unless the wrapper installs NAG groups. When NAG is inactive the
patched block forward takes the identical code path as the original block.

Reference: https://github.com/ChenDarYen/Normalized-Attention-Guidance (MIT).
"""

from __future__ import annotations

import torch

from core.attention import AttentionMode
from core.inference.nag_dit import nag_guidance


def _nag_double_block_forward(block, x, txt, vec, grid_h: int, grid_w: int):
    """DoubleStreamDiTBlock.forward with an optional NAG branch on the image attention
    output. Behaviour is identical to the vendored forward unless ``block._nag_active``
    is True (set by MiniT2INAGWrapper for the duration of a NAG forward).

    NAG grouping (set on the block as attributes by the wrapper):
      _nag_pos_idx  : batch index of the positive cond image/text group
      _nag_neg_idx  : batch index of the nag-negative scratch group (same image, nag_neg text)
    The guided image attention output overwrites the positive group before img_attn_proj.
    """
    b, li, _ = x.shape
    lt = txt.shape[1]
    x_norm = block.img_norm1(x)
    txt_norm = block.txt_norm1(txt)
    qkv_i = block.img_qkv(x_norm).reshape(b, li, 3, block.num_heads, block.head_dim)
    qkv_t = block.txt_qkv(txt_norm).reshape(b, lt, 3, block.num_heads, block.head_dim)
    q_i, k_i, v_i = qkv_i[:, :, 0], qkv_i[:, :, 1], qkv_i[:, :, 2]
    q_t, k_t, v_t = qkv_t[:, :, 0], qkv_t[:, :, 1], qkv_t[:, :, 2]
    q_i, k_i = block.q_norm(q_i), block.k_norm(k_i)
    q_t, k_t = block.q_norm(q_t), block.k_norm(k_t)
    q = block.rope(torch.cat([q_t, q_i], dim=1), txt_len=lt, grid_h=grid_h, grid_w=grid_w)
    k = block.rope(torch.cat([k_t, k_i], dim=1), txt_len=lt, grid_h=grid_h, grid_w=grid_w)
    v = torch.cat([v_t, v_i], dim=1)

    from core.models.minit2i.vendor.mmjit import mem_efficient_sdpa
    out = mem_efficient_sdpa(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        backend=getattr(block, "_attn_backend", "native"),
        mode=getattr(block, "_attn_mode", AttentionMode.INFERENCE),
    ).transpose(1, 2).contiguous()  # [b, seq, heads, hd]

    img_out = out[:, lt:].reshape(b, li, -1)   # image attention output, flattened heads
    txt_out = out[:, :lt].reshape(b, lt, -1)

    if getattr(block, "_nag_active", False):
        pos = block._nag_pos_idx
        neg = block._nag_neg_idx
        z_pos = img_out[pos:pos + 1]
        z_neg = img_out[neg:neg + 1]
        guided = nag_guidance(
            z_pos, z_neg,
            scale=block._nag_scale, tau=block._nag_tau, alpha=block._nag_alpha, norm_p=2,
        )
        img_out = img_out.clone()
        img_out[pos:pos + 1] = guided

    x = x + block.img_attn_proj(img_out)
    txt = txt + block.txt_attn_proj(txt_out)
    x = x + block.img_mlp(block.img_norm2(x))
    txt = txt + block.txt_mlp(block.txt_norm2(txt))
    return x, txt


class MiniT2INAGWrapper:
    """Threads a NAG-negative text context through the MiniT2I transformer.

    Wraps a ``MiniT2IMMJiTModel`` (which exposes ``.model.net`` == MMJiT). Presents the
    same call signature the euler loop uses: ``wrapper(x, t, text, mask)`` -> predicted
    x0. Internally it builds the pos/nag_neg (and uncond, for CFG) batch, runs the MMJiT
    forward with NAG-guided image attention, and slices the result back to the caller's
    batch so the surrounding CFG math is unchanged.

    The wrapper is only installed when NAG is active; ``restore()`` un-patches the block
    forwards. When not installed the model runs its original code path (byte-identical).
    """

    def __init__(self, transformer, nag_neg_text, nag_neg_mask,
                 nag_scale=5.0, nag_tau=2.5, nag_alpha=0.25):
        self.transformer = transformer
        self.net = transformer.model.net  # MMJiT
        self.nag_neg_text = nag_neg_text   # [1, L, txt_input_size]
        self.nag_neg_mask = nag_neg_mask   # [1, L]
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
        self._orig_forwards = {}
        self._install()

    # ---- block patching -------------------------------------------------
    def _install(self):
        import types
        for block in self.net.double_blocks:
            self._orig_forwards[id(block)] = block.forward
            block._nag_scale = self.nag_scale
            block._nag_tau = self.nag_tau
            block._nag_alpha = self.nag_alpha
            block._nag_active = False
            block.forward = types.MethodType(_nag_double_block_forward, block)

    def restore(self):
        for block in self.net.double_blocks:
            fwd = self._orig_forwards.get(id(block))
            if fwd is not None:
                block.forward = fwd
            for attr in ("_nag_active", "_nag_pos_idx", "_nag_neg_idx",
                         "_nag_scale", "_nag_tau", "_nag_alpha"):
                if hasattr(block, attr):
                    delattr(block, attr)

    def _set_groups(self, active, pos_idx=0, neg_idx=0):
        for block in self.net.double_blocks:
            block._nag_active = active
            block._nag_pos_idx = pos_idx
            block._nag_neg_idx = neg_idx

    # ---- forward --------------------------------------------------------
    def __call__(self, x, t, text, mask):
        """x0 prediction with NAG. Detects the caller's batch layout:
          batch 2 -> CFG: [cond, uncond]; batch 1 -> distilled/no-cfg: [cond].
        """
        b = x.shape[0]
        nag_text = self.nag_neg_text.to(dtype=text.dtype, device=text.device)
        nag_mask = self.nag_neg_mask.to(device=mask.device)

        if b == 2:
            # CFG: incoming [cond, uncond]. Build [cond, uncond, cond] / [pos, u, nag].
            x_c = x[0:1]
            xx = torch.cat([x, x_c], dim=0)                       # [c, u, c]
            tt = torch.cat([t, t[0:1]], dim=0)
            yy = torch.cat([text, nag_text], dim=0)              # [pos, u, nag]
            mm = torch.cat([mask, nag_mask], dim=0)
            self._set_groups(active=True, pos_idx=0, neg_idx=2)
            try:
                out = self.net(xx, tt, yy, mm)
            finally:
                self._set_groups(active=False)
            return out[0:2]                                       # [cond, uncond]

        if b == 1:
            # distilled / no-cfg this step: [cond]. Double to [pos, nag].
            xx = torch.cat([x, x], dim=0)
            tt = torch.cat([t, t], dim=0)
            yy = torch.cat([text, nag_text], dim=0)
            mm = torch.cat([mask, nag_mask], dim=0)
            self._set_groups(active=True, pos_idx=0, neg_idx=1)
            try:
                out = self.net(xx, tt, yy, mm)
            finally:
                self._set_groups(active=False)
            return out[0:1]

        # Unexpected batch size: run without NAG (safety fallback).
        return self.net(x, t, text, mask)
