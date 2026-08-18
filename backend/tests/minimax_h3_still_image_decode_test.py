"""MiniMax-H3 video VAE: `_decode`'s `num_frames == 1` special case (Phase 0,
still-image generation support).

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_still_image_decode_test.py -v

WHY THIS FILE EXISTS
--------------------
`_decode`'s multi-chunk walk computes
``num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)``,
which is 0 for a single latent frame at this checkpoint's geometry
(``num_tokens = 1 + token_drop(3) = 4``, ``pad_tokens = (-4) % 5 = 1``,
``num_chunks = 5 // 5 - 1 = 0``): the for-loop over ``range(num_chunks)`` never
runs, ``decoded_chunks`` stays empty, and ``torch.cat([], dim=2)`` raises. The
fix mirrors ``_encode``'s own ``num_frames == 1`` special case: a lone latent
token is decoded directly through ``_decode_clip`` and cropped by the same
``frame_pre_padding`` leading pad every OTHER chunk already crops at ``j ==
0`` -- a lone token carries the same causal zero-pre-padding as token 0 of any
chunk, so the crop is the identical bookkeeping, not a new rule.

This does not instantiate the real (multi-GB) VAE. `_decode` is called
UNBOUND, directly off the class, against a minimal duck-typed stand-in that
carries only the attributes it reads (the geometry constants derived in
``AutoencoderKLMiniMaxH3.__init__``, lines ~611-617, at this checkpoint's
released defaults: ``clip_length=17``, ``token_drop=3``,
``temporal_compression_ratio=4`` -> ``frame_pre_padding=3``,
``tokens_chunk_size=5``, ``token_overlap=2``, ``frame_overlap=5``) plus a
stubbed ``_decode_clip`` -- so this exercises the REAL, PRODUCTION `_decode`
method's control flow and arithmetic, not a reimplementation of it.
"""

import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.minimax_h3.vendor.autoencoder_kl_minimax_h3 import (  # noqa: E402
    AutoencoderKLMiniMaxH3,
)
from core.models.minimax_h3.loader import minimax_h3_latent_frames  # noqa: E402


# Released-checkpoint geometry (AutoencoderKLMiniMaxH3.__init__ defaults):
# clip_length=17, token_drop=3, temporal_compression_ratio=4.
CLIP_LENGTH = 17
TOKEN_DROP = 3
TEMPORAL_RATIO = 4
FRAME_PRE_PADDING = (-CLIP_LENGTH) % TEMPORAL_RATIO  # 3
TOKENS_CHUNK_SIZE = -(-CLIP_LENGTH // TEMPORAL_RATIO)  # ceil(17/4) = 5
TOKEN_OVERLAP = (-TOKEN_DROP) % TOKENS_CHUNK_SIZE  # 2
FRAME_OVERLAP = max(TOKEN_OVERLAP * TEMPORAL_RATIO - FRAME_PRE_PADDING, 0)  # 5


class _FakeVAE:
    """Duck-typed stand-in for the attributes/methods `_decode` reads."""

    def __init__(self):
        self.tokens_chunk_size = TOKENS_CHUNK_SIZE
        self.temporal_compression_ratio = TEMPORAL_RATIO
        self.frame_pre_padding = FRAME_PRE_PADDING
        self.token_overlap = TOKEN_OVERLAP
        self.frame_overlap = FRAME_OVERLAP
        self.config = SimpleNamespace(token_drop=TOKEN_DROP, clip_length=CLIP_LENGTH)
        self.decode_clip_calls: list[tuple[int, ...]] = []
        # `_blend` is pure tensor arithmetic with no VAE-specific state (see
        # its body), so the REAL implementation is bound here rather than
        # stubbed -- only `_decode_clip` (the actual decoder forward) is fake.
        self._blend = types.MethodType(AutoencoderKLMiniMaxH3._blend, self)

    def _decode_clip(self, z: torch.Tensor) -> torch.Tensor:
        """Each input latent TOKEN expands into `temporal_compression_ratio`
        pixel frames, exactly as the real ViT decoder's `patch_size_t` does.
        Frames are stamped with their pixel-frame index (broadcast over H/W)
        so a crop can be verified by value, not just by shape."""
        self.decode_clip_calls.append(tuple(z.shape))
        b, _c, t, h, w = z.shape
        pixel_frames = t * self.temporal_compression_ratio
        out = torch.arange(pixel_frames, dtype=torch.float32).view(1, 1, pixel_frames, 1, 1)
        return out.repeat(b, 3, 1, h, w)


def test_t1_decodes_directly_without_chunk_walk():
    """The load-bearing case: a lone latent frame must not reach the
    chunk-walk (which would divide into `num_chunks == 0` and crash on an
    empty `torch.cat`)."""
    vae = _FakeVAE()
    z = torch.zeros(1, 24, 1, 4, 4)  # [B, C, T=1, H_lat, W_lat]

    dec = AutoencoderKLMiniMaxH3._decode(vae, z)

    assert vae.decode_clip_calls == [(1, 24, 1, 4, 4)]
    # 4 pixel frames come back from `_decode_clip` (patch_size_t=4); the
    # leading `frame_pre_padding` (3) causal-zero frames are cropped, leaving
    # exactly 1 true frame -- symmetric with `_encode`'s own T==1 branch.
    assert dec.shape == (1, 3, 1, 4, 4)
    assert torch.equal(dec[0, 0, 0], torch.full((4, 4), float(FRAME_PRE_PADDING)))


def test_t1_never_calls_blend():
    """The T==1 path returns from `_decode_clip` alone; `_blend` (cross-fade
    between consecutive chunks) has nothing to blend with a single chunk."""
    vae = _FakeVAE()
    orig_blend = vae._blend
    calls = []
    vae._blend = lambda *a, **kw: calls.append(1) or orig_blend(*a, **kw)

    AutoencoderKLMiniMaxH3._decode(vae, torch.zeros(1, 24, 1, 4, 4))

    assert calls == []


def test_t1_matches_loader_latent_frame_count():
    """`minimax_h3_latent_frames` (the caller's geometry function) says a
    still-image request encodes to exactly 1 latent frame -- the shape this
    decode path is built for."""
    assert minimax_h3_latent_frames(1) == 1
    assert minimax_h3_latent_frames(0) == 1


def test_multichunk_path_single_outer_chunk_is_unaffected_by_the_t1_special_case():
    """Regression guard: adding the T==1 branch must not change the ordinary
    multi-chunk decode. latent_frames=7 (num_frames=22, the shortest
    previously-decodable clip) is ONE outer chunk (`num_chunks == 1`), so
    `_blend` never fires here -- see the two-outer-chunk test below for that."""
    vae = _FakeVAE()
    latent_frames = 7  # 22 pixel frames -> 7 latent frames (loader's formula)
    assert minimax_h3_latent_frames(22) == latent_frames
    z = torch.zeros(1, 24, latent_frames, 4, 4)

    dec = AutoencoderKLMiniMaxH3._decode(vae, z)

    assert dec.shape == (1, 3, 22, 4, 4)
    assert all(shape[2] > 1 for shape in vae.decode_clip_calls), (
        "the multi-chunk path must still request more than one token per "
        "clip; a single-token call here would mean the T==1 branch leaked "
        "into a request it should not have touched"
    )


def test_multichunk_path_two_outer_chunks_still_blends_correctly():
    """latent_frames=12 (num_frames=39, MEASURED in `_clip_pixel_frames`'s
    sibling in h3_pipeline_ops.py) is TWO outer chunks (`num_chunks == 2`),
    which does exercise `_blend` (the cross-fade between consecutive chunks'
    `token_drop`-sized overlap) -- confirming the T==1 branch placed ABOVE
    this arithmetic in `_decode` does not disturb it."""
    vae = _FakeVAE()
    latent_frames = 12
    z = torch.zeros(1, 24, latent_frames, 4, 4)

    orig_blend = vae._blend
    blend_calls = []
    vae._blend = lambda *a, **kw: (blend_calls.append(1), orig_blend(*a, **kw))[1]

    dec = AutoencoderKLMiniMaxH3._decode(vae, z)

    assert dec.shape == (1, 3, 39, 4, 4)
    assert len(blend_calls) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
