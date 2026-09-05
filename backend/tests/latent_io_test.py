"""latent I/O resize property tests -- CPU, model-free, ~10s (design §6.6).

The channel-partial resize is the one piece of the VAE-swap machinery whose
failure is silent: a wrong pack order still produces the right SHAPES and a
model that trains, just on scrambled channels. So each arch is checked by
EQUIVALENCE against its own real pack/unpack function -- ``pack_latents`` /
``_patchify`` / ``_flux2_patchify_latents_for_training`` / anima's ``PatchEmbed``
Rearrange / zimage's ``batched_patchify`` / minit2i's ``BottleneckPatchEmbed`` /
ltx2's ``_pack_latents`` -- imported, never reimplemented here. A reimplemented
packer in a test proves only that the test and the code share a mistake.

Input and output are SEPARATE cases: anima folds C outermost on the way in and
innermost on the way out, so a test that only exercises one side passes on a
spec that has the other side backwards.

No model files, no downloads: every module is an ``nn.Linear``/``nn.Conv2d`` of
the shape the arch declares, or the arch's own small I/O class.

Run with (cwd backend/):
    ../venv/Scripts/python.exe -m pytest tests/latent_io_test.py -v
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from core.models.components.latent_io import resize_latent_io  # noqa: E402
from core.models.components.wiring import (  # noqa: E402
    ANIMA_LATENT_IO, FLUX2_LATENT_IO, KREA2_LATENT_IO, LENS_LATENT_IO,
    LTX2_LATENT_IO, MINIT2I_LATENT_IO, SD_UNET_LATENT_IO, ZIMAGE_LATENT_IO,
)

B = 1
HIDDEN = 8
C_OLD = 4
C_BIG = 6      # expansion target
C_SMALL = 2    # shrink target


def _module(root, path):
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _seeded(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


# ---------------------------------------------------------------------------
# Per-arch cases. `encode` = real pack + the declared input module;
# `decode` = the declared output module + real unpack, returning [B, C, ...].
# ---------------------------------------------------------------------------

class Case:
    name: str
    spec = None

    def build(self, c) -> nn.Module:
        raise NotImplementedError

    def latent(self, c):
        return _seeded(B, c, 4, 4, seed=1)

    def expand(self, x, c_old, c_new):
        pad = torch.zeros(x.shape[0], c_new - c_old, *x.shape[2:], dtype=x.dtype)
        return torch.cat([x, pad], dim=1)

    def truncate(self, x, c_new):
        return x[:, :c_new]

    def zero_tail(self, x, c_new):
        y = x.clone()
        y[:, c_new:] = 0
        return y

    def hidden(self):
        return _seeded(B, 4, HIDDEN, seed=2)

    def encode(self, root, x):
        raise NotImplementedError

    def decode(self, root, h):
        raise NotImplementedError


class SDCase(Case):
    """sd15/sdxl: conv on both sides, no packing at all."""

    name = "sd15/sdxl"
    spec = SD_UNET_LATENT_IO

    class _Unet(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.conv_in = nn.Conv2d(c, HIDDEN, 3, padding=1)
            self.conv_out = nn.Conv2d(HIDDEN, c, 3, padding=1)
            self.config = SimpleNamespace(in_channels=c, out_channels=c)

    class _Root(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.unet = SDCase._Unet(c)

    def build(self, c):
        return self._Root(c)

    def hidden(self):
        return _seeded(B, HIDDEN, 4, 4, seed=2)

    def encode(self, root, x):
        return root.unet.conv_in(x)

    def decode(self, root, h):
        return root.unet.conv_out(h)


class ZImageCase(Case):
    """zimage: C innermost on BOTH sides (the only arch that is)."""

    name = "zimage"
    spec = ZIMAGE_LATENT_IO

    class _Root(nn.Module):
        def __init__(self, c):
            super().__init__()
            from core.models.zimage_transformer import FinalLayer
            self.all_x_embedder = nn.ModuleDict({"2-1": nn.Linear(4 * c, HIDDEN)})
            self.all_final_layer = nn.ModuleDict({"2-1": FinalLayer(HIDDEN, 4 * c)})
            self.in_channels = c
            self.out_channels = c

    def build(self, c):
        return self._Root(c)

    def latent(self, c):
        return _seeded(B, c, 1, 4, 4, seed=1)

    def encode(self, root, x):
        from core.models.batched_zimage_wrapper import BatchedZImageWrapperOptimized
        cap = torch.zeros(B, 2, 2560)
        mask = torch.ones(B, 2, dtype=torch.bool)
        patches = BatchedZImageWrapperOptimized.batched_patchify(
            SimpleNamespace(SEQ_MULTI_OF=1), x, cap, mask, 2, 1,
        )[0]
        return root.all_x_embedder["2-1"](patches)

    def decode(self, root, h):
        from core.models.zimage_transformer import ZImageTransformer2DModel
        packed = root.all_final_layer["2-1"].linear(h)
        # out_channels comes from the ROOT, so a resize that failed to sync it
        # fails here rather than silently unpacking at the old width.
        out = ZImageTransformer2DModel.unpatchify(
            SimpleNamespace(out_channels=root.out_channels),
            [packed[0]], [(1, 4, 4)], 2, 1,
        )
        return out[0].unsqueeze(0)


class Krea2Case(Case):
    name = "krea2"
    spec = KREA2_LATENT_IO

    class _Root(nn.Module):
        def __init__(self, c):
            super().__init__()
            from core.models.krea2.vendor.transformer import Krea2FinalLayer
            self.img_in = nn.Linear(4 * c, HIDDEN)
            self.final_layer = Krea2FinalLayer(HIDDEN, out_channels=4 * c, eps=1e-6)

    def build(self, c):
        return self._Root(c)

    def encode(self, root, x):
        from core.models.krea2.krea2_pipeline_ops import pack_latents
        return root.img_in(pack_latents(x, 2))

    def decode(self, root, h):
        from core.models.krea2.krea2_pipeline_ops import unpack_latents
        packed = root.final_layer.linear(h)
        c = packed.shape[-1] // 4
        return unpack_latents(packed, 2, 2, 2).reshape(B, c, 4, 4)


class LensCase(Case):
    name = "lens"
    spec = LENS_LATENT_IO

    class _Root(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.img_in = nn.Linear(4 * c, HIDDEN)
            self.proj_out = nn.Linear(HIDDEN, 4 * c)

    def build(self, c):
        return self._Root(c)

    def hidden(self):
        return _seeded(B, 2, 2, HIDDEN, seed=2)

    def encode(self, root, x):
        from core.models.lens.lens_pipeline_ops import _patchify
        return root.img_in(_patchify(x).permute(0, 2, 3, 1))

    def decode(self, root, h):
        from core.models.lens.lens_pipeline_ops import _unpatchify
        return _unpatchify(root.proj_out(h).permute(0, 3, 1, 2))


class Flux2Case(Case):
    name = "flux2"
    spec = FLUX2_LATENT_IO

    class _Root(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.x_embedder = nn.Linear(4 * c, HIDDEN, bias=False)
            self.proj_out = nn.Linear(HIDDEN, 4 * c, bias=False)

    def build(self, c):
        return self._Root(c)

    def hidden(self):
        return _seeded(B, 2, 2, HIDDEN, seed=2)

    def encode(self, root, x):
        from core.training.base_trainer import BaseTrainer
        packed = BaseTrainer._flux2_patchify_latents_for_training(None, x)
        return root.x_embedder(packed.permute(0, 2, 3, 1))

    def decode(self, root, h):
        from core.training.base_trainer import BaseTrainer
        packed = root.proj_out(h).permute(0, 3, 1, 2)
        return BaseTrainer._flux2_unpatchify_latents(None, packed)


class AnimaCase(Case):
    """anima: C OUTER on the input side, INNER on the output side, plus the
    padding-mask channel riding along on the input only."""

    name = "anima"
    spec = ANIMA_LATENT_IO

    class _Root(nn.Module):
        def __init__(self, c):
            super().__init__()
            from core.models.anima.anima_models import FinalLayer, PatchEmbed
            self.x_embedder = PatchEmbed(2, 1, in_channels=c + 1, out_channels=HIDDEN)
            self.final_layer = FinalLayer(HIDDEN, 2, 1, out_channels=c)
            self.in_channels = c

    def build(self, c):
        return self._Root(c)

    def latent(self, c):
        return _seeded(B, c + 1, 1, 4, 4, seed=1)

    def expand(self, x, c_old, c_new):
        pad = torch.zeros(x.shape[0], c_new - c_old, *x.shape[2:], dtype=x.dtype)
        return torch.cat([x[:, :c_old], pad, x[:, c_old:]], dim=1)

    def truncate(self, x, c_new):
        return torch.cat([x[:, :c_new], x[:, -1:]], dim=1)

    def zero_tail(self, x, c_new):
        y = x.clone()
        y[:, c_new:-1] = 0
        return y

    def hidden(self):
        return _seeded(B, 1, 2, 2, HIDDEN, seed=2)

    def encode(self, root, x):
        return root.x_embedder(x)

    def decode(self, root, h):
        from core.models.anima.anima_models import Anima
        packed = root.final_layer.linear(h)
        return Anima.unpatchify(
            SimpleNamespace(patch_spatial=2, patch_temporal=1), packed,
        )


class MiniT2ICase(Case):
    """minit2i: conv on the way in, C innermost on the way out."""

    name = "minit2i"
    spec = MINIT2I_LATENT_IO

    class _Net(nn.Module):
        def __init__(self, c):
            super().__init__()
            from core.models.minit2i.vendor.mmjit import BottleneckPatchEmbed, FinalLayer
            self.img_embedder = BottleneckPatchEmbed(4, 2, c, 6, HIDDEN)
            self.final_layer = FinalLayer(HIDDEN, 2, c)
            self.cfg = SimpleNamespace(patch_size=2, in_channels=c)

    class _Root(nn.Module):
        """The spec's paths are rooted at the ConfigMixin wrapper, so the two
        latent layers sit under ``model.net``."""

        def __init__(self, c, net_cls):
            super().__init__()
            self.model = nn.Module()
            self.model.net = net_cls(c)

    def build(self, c):
        return self._Root(c, self._Net)

    def encode(self, root, x):
        return root.model.net.img_embedder(x)[0]

    def decode(self, root, h):
        from core.models.minit2i.vendor.mmjit import MMJiT
        packed = root.model.net.final_layer.linear(h)
        return MMJiT.unpatchify(SimpleNamespace(cfg=root.model.net.cfg), packed, 2, 2)


class Ltx2Case(Case):
    """ltx2: patch_size=1, so the packed axis is C alone."""

    name = "ltx2"
    spec = LTX2_LATENT_IO

    class _Root(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.proj_in = nn.Linear(c, HIDDEN)
            self.proj_out = nn.Linear(HIDDEN, c)

    def build(self, c):
        return self._Root(c)

    def latent(self, c):
        return _seeded(B, c, 1, 2, 2, seed=1)

    def encode(self, root, x):
        from core.training.ops.ltx2_ops import _pack_latents
        return root.proj_in(_pack_latents(x))

    def decode(self, root, h):
        from core.training.ops.ltx2_ops import _unpack_leading_frames
        return _unpack_leading_frames(root.proj_out(h), 1, 2, 2)


CASES = [SDCase(), ZImageCase(), Krea2Case(), LensCase(), Flux2Case(),
         AnimaCase(), MiniT2ICase(), Ltx2Case()]
CASE_IDS = [c.name for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_input_expansion_is_equivalent(case):
    """1. Appending zero channels to the latent must not change what the input
    module produces -- for that to hold, the copied weights have to land on the
    same channels, which is exactly what `in_channel_order` decides."""
    root = case.build(C_OLD)
    x = case.latent(C_OLD)
    y_old = case.encode(root, x)

    resize_latent_io(root, case.spec, C_BIG)
    y_new = case.encode(root, case.expand(x, C_OLD, C_BIG))

    assert y_new.shape == y_old.shape
    assert torch.allclose(y_new, y_old, atol=1e-6)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_input_shrink_is_equivalent(case):
    """2. Shrinking drops the tail channels: the narrowed module on a truncated
    latent must equal the original module on the same latent with those channels
    zeroed."""
    root = case.build(C_OLD)
    x = case.latent(C_OLD)
    y_old = case.encode(root, case.zero_tail(x, C_SMALL))

    resize_latent_io(root, case.spec, C_SMALL)
    y_new = case.encode(root, case.truncate(x, C_SMALL))

    assert torch.allclose(y_new, y_old, atol=1e-6)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_output_expansion_and_shrink(case):
    """3. On the output side the SAME hidden state must decode to the same
    latent channels, with the new ones exactly zero -- `out_channel_order`, read
    through the arch's own unpack function."""
    h = case.hidden()

    root = case.build(C_OLD)
    y_old = case.decode(root, h)

    grown = case.build(C_OLD)
    grown.load_state_dict(root.state_dict())
    resize_latent_io(grown, case.spec, C_BIG)
    y_big = case.decode(grown, h)
    assert y_big.shape[1] == C_BIG
    assert torch.allclose(y_big[:, :C_OLD], y_old, atol=1e-6)
    assert torch.count_nonzero(y_big[:, C_OLD:]) == 0

    shrunk = case.build(C_OLD)
    shrunk.load_state_dict(root.state_dict())
    resize_latent_io(shrunk, case.spec, C_SMALL)
    y_small = case.decode(shrunk, h)
    assert y_small.shape[1] == C_SMALL
    assert torch.allclose(y_small, y_old[:, :C_SMALL], atol=1e-6)


def test_anima_extra_input_channel_moves_to_the_new_position():
    """4. anima's padding-mask channel sits after the latent channels, so it has
    to MOVE from C_old to C_new -- copying the input weight as one block would
    leave it buried in the middle."""
    case = AnimaCase()
    root = case.build(C_OLD)
    old_w = _module(root, case.spec.in_module).weight.detach().clone()
    P = case.spec.pack_elems

    resize_latent_io(root, case.spec, C_BIG)
    new_w = _module(root, case.spec.in_module).weight.detach()

    old_view = old_w.reshape(HIDDEN, C_OLD + 1, P)      # outer: [hidden, c, s]
    new_view = new_w.reshape(HIDDEN, C_BIG + 1, P)
    assert torch.equal(new_view[:, C_BIG], old_view[:, C_OLD])
    assert torch.equal(new_view[:, :C_OLD], old_view[:, :C_OLD])
    assert torch.count_nonzero(new_view[:, C_OLD:C_BIG]) == 0


def test_sdxl_resize_unet_in_out_copies_bit_identically():
    """5. The public SDXL entry point still channel-partial copies, bit for bit,
    and now zeroes the new channels (behaviour change, design §6.2)."""
    from core.models.sdxl_custom_arch import resize_unet_in_out

    unet = SDCase._Unet(C_OLD)
    old_in_w = unet.conv_in.weight.detach().clone()
    old_in_b = unet.conv_in.bias.detach().clone()
    old_out_w = unet.conv_out.weight.detach().clone()
    old_out_b = unet.conv_out.bias.detach().clone()

    resize_unet_in_out(unet, C_BIG)

    assert torch.equal(unet.conv_in.weight[:, :C_OLD], old_in_w)
    assert torch.count_nonzero(unet.conv_in.weight[:, C_OLD:]) == 0
    assert torch.equal(unet.conv_in.bias, old_in_b)
    assert torch.equal(unet.conv_out.weight[:C_OLD], old_out_w)
    assert torch.count_nonzero(unet.conv_out.weight[C_OLD:]) == 0
    assert torch.equal(unet.conv_out.bias[:C_OLD], old_out_b)
    assert torch.count_nonzero(unet.conv_out.bias[C_OLD:]) == 0
    assert unet.config.in_channels == C_BIG and unet.config.out_channels == C_BIG

    # The asymmetric signature (out_channels != in_channels) is part of the
    # public contract even though no caller uses it today.
    unet2 = SDCase._Unet(C_OLD)
    resize_unet_in_out(unet2, C_BIG, C_SMALL)
    assert unet2.conv_in.in_channels == C_BIG
    assert unet2.conv_out.out_channels == C_SMALL
    assert unet2.config.in_channels == C_BIG and unet2.config.out_channels == C_SMALL


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_saved_weights_survive_a_second_resize(case):
    """6. Re-loading a swapped checkpoint: the zero-init must not survive the
    load. Resize -> train -> save -> (fresh model) resize -> load is bit-identical."""
    trained = case.build(C_OLD)
    resize_latent_io(trained, case.spec, C_BIG)
    with torch.no_grad():
        for path in (case.spec.in_module, case.spec.out_module):
            module = _module(trained, path)
            module.weight.normal_(std=0.02)
            if module.bias is not None:
                module.bias.normal_(std=0.02)
    saved = {k: v.clone() for k, v in trained.state_dict().items()}

    reloaded = case.build(C_OLD)
    resize_latent_io(reloaded, case.spec, C_BIG)
    reloaded.load_state_dict(saved)

    for path in (case.spec.in_module, case.spec.out_module):
        before = _module(trained, path)
        after = _module(reloaded, path)
        assert torch.equal(after.weight, before.weight)
        if before.bias is not None:
            assert torch.equal(after.bias, before.bias)
