"""SD1.5 / SDXL's REPA tap: a conv U-Net, so a feature map rather than tokens.

These two differ from the six wired DiTs in three ways, and each is pinned here:

  * the tapped state is ``[B, C, h, w]``. The teacher is interpolated to that
    map's OWN (h, w) and the two correspond cell for cell -- there is no packing
    order to derive, only the row-major flatten that makes the projector's
    channel-axis Linear a 1x1 convolution.
  * "depth" is not a block index. A U-Net's blocks have no total order by depth
    (the same reason ``depth_blocks`` is None for these archs), so
    ``repa_align_depth`` indexes a three-site menu -- deepest down block, mid
    block, first up block -- and out-of-range is refused, not clamped.
  * the block loop is diffusers', not ours, so the tap is a forward hook
    registered around the training forward and removed in a ``finally``.

Also pinned: the disabled path installs nothing, the alignment gradient reaches
the U-Net with gradient checkpointing on as well as off, and both full-parameter
adapters return the path they wrote so the projector sidecar can pair with it.

REPA was published for DiTs (arXiv:2410.06940); applying it to a conv U-Net's
mid block is an extrapolation from that paper, not a result it reports.

Everything runs on randomly initialised toy-geometry U-Nets; no checkpoint is
read and no GPU is used.

Run:
    venv/Scripts/python.exe -m pytest backend/tests/repa_sd_unet_tap_test.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from api.arch_capabilities import TRAINING_FEATURE_UNSUPPORTED  # noqa: E402
from diffusers import DDPMScheduler, UNet2DConditionModel  # noqa: E402
from core.training import repa as repa_module  # noqa: E402
from core.training.arch import ARCH_REGISTRY  # noqa: E402
from core.training.arch.base_arch import TrainStepContext  # noqa: E402
from core.training.base_trainer import BaseTrainer  # noqa: E402
from core.training.ops import sd_sdxl_ops  # noqa: E402

ARCHS = ("sd15", "sdxl")
CROSS_DIM = {"sd15": 8, "sdxl": 16}
POOLED_DIM = 12
TIME_EMBED = 4
LATENT_H, LATENT_W = 12, 8      # deliberately non-square: a transposed flatten shows
ENC_DIM = 6
ENC_GRID = 4



class _StubEncoder(nn.Module):
    """A frozen teacher shaped like SigLIP2: a square token grid, no CLS."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, pixel_values):
        b = pixel_values.shape[0]
        feat = torch.linspace(0, 1, ENC_GRID * ENC_GRID * ENC_DIM)
        feat = feat.reshape(1, ENC_GRID * ENC_GRID, ENC_DIM).expand(b, -1, -1)
        return SimpleNamespace(last_hidden_state=feat.contiguous())


def _tiny_unet(arch: str) -> UNet2DConditionModel:
    torch.manual_seed(0)
    common = dict(sample_size=16, in_channels=4, out_channels=4, layers_per_block=1,
                  attention_head_dim=2, norm_num_groups=4,
                  cross_attention_dim=CROSS_DIM[arch])
    if arch == "sdxl":
        return UNet2DConditionModel(
            block_out_channels=(8, 16, 32),
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D",
                              "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
            addition_embed_type="text_time", addition_time_embed_dim=TIME_EMBED,
            projection_class_embeddings_input_dim=TIME_EMBED * 6 + POOLED_DIM,
            transformer_layers_per_block=1, **common)
    return UNet2DConditionModel(
        block_out_channels=(8, 16, 32, 32),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                          "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
                        "CrossAttnUpBlock2D"), **common)


def _unet_inputs(arch: str, batch: int = 2, seed: int = 1):
    torch.manual_seed(seed)
    kwargs = dict(
        sample=torch.randn(batch, 4, LATENT_H, LATENT_W),
        timestep=torch.tensor([7] * batch),
        encoder_hidden_states=torch.randn(batch, 5, CROSS_DIM[arch]),
    )
    if arch == "sdxl":
        kwargs["added_cond_kwargs"] = {"text_embeds": torch.randn(batch, POOLED_DIM),
                                       "time_ids": torch.randn(batch, 6)}
    return kwargs


def _measured_site_maps(unet):
    """{label: output map} for the three sites, from one real forward."""
    caught = {}

    def _mk(label):
        def _hook(_m, _a, out):
            caught[label] = out[0] if isinstance(out, tuple) else out
        return _hook

    sites = repa_module.spatial_tap_sites(unet)
    handles = [block.register_forward_hook(_mk(label)) for label, block in sites]
    try:
        arch = "sdxl" if unet.config.addition_embed_type == "text_time" else "sd15"
        with torch.no_grad():
            unet(**_unet_inputs(arch, batch=1))
    finally:
        for handle in handles:
            handle.remove()
    return caught


def _trainer(arch: str, unet=None, **config):
    cfg = {"repa_enable": True, "repa_tagger_model_dir": "unused-because-stubbed"}
    cfg.update(config)
    return SimpleNamespace(
        config=cfg,
        arch=ARCH_REGISTRY[arch](),
        unet=unet if unet is not None else _tiny_unet(arch),
        is_sdxl=(arch == "sdxl"),
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        model_path="",
        log_prefix="[test]",
    )


def _stub_encoder_loader(monkeypatch):
    monkeypatch.setattr(repa_module, "load_repa_encoder",
                        lambda source, **kwargs: (_StubEncoder(), ENC_DIM, 32))


def _train_step_trainer(arch: str, unet, logged, align_depth: int = 1):
    trainer = _trainer(arch, unet)
    trainer.repa_enable = True  # _setup_repa's own flag, not the config key
    trainer.mixed_precision = False
    trainer.timestep_sampler = None
    trainer.noise_process = "ddpm"
    trainer.prediction_target = "epsilon"
    trainer.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    trainer.min_snr_gamma = 0.0
    trainer.reconstruction_loss_weight = 0.0
    trainer.snr_regularization_loss = None
    trainer.energy_regularization_loss = None
    trainer.crop_decode_loss_enable = False
    trainer.repa_encoder = _StubEncoder()
    trainer.repa_projector = repa_module.RepaProjector(
        repa_module.spatial_site_width(*repa_module.spatial_tap_sites(unet)[align_depth]),
        ENC_DIM, hidden=16)
    trainer.repa_size = 32
    trainer.repa_weight = 1.0
    trainer.repa_align_depth = align_depth
    trainer._repa_tap_module = unet
    trainer._ensure_repa_on_device = lambda: None
    trainer.log_extra_metric = lambda name, value: logged.append((name, value))
    unet._repa_tap_depth = align_depth
    return trainer


def _train_step(trainer, arch, *, repa_pixels, seed=7, batch=2):
    torch.manual_seed(seed)
    return sd_sdxl_ops.train_step(
        trainer,
        latents=torch.randn(batch, 4, LATENT_H, LATENT_W),
        text_embeddings=torch.randn(batch, 5, CROSS_DIM[arch]),
        pooled_embeddings=torch.randn(batch, POOLED_DIM) if arch == "sdxl" else None,
        timesteps=None,
        repa_pixels=repa_pixels,
    )


def _pixels(batch=2):
    torch.manual_seed(3)
    return torch.rand(batch, 3, 32, 32) * 2 - 1


def _hook_count(unet) -> int:
    return sum(len(m._forward_hooks) for m in unet.modules())



@pytest.mark.parametrize("arch", ARCHS)
def test_a_fresh_unet_carries_no_tap_state_and_no_hook(arch):
    unet = _tiny_unet(arch)
    assert not hasattr(unet, "_repa_tap_depth")
    assert not hasattr(unet, "_repa_tap_out")
    assert _hook_count(unet) == 0


@pytest.mark.parametrize("arch", ARCHS)
def test_a_disabled_step_installs_nothing_and_reports_the_plain_loss(arch):
    unet = _tiny_unet(arch).train()
    logged = []
    trainer = _train_step_trainer(arch, unet, logged)
    trainer.repa_enable = False

    loss, pred, _ = _train_step(trainer, arch, repa_pixels=_pixels())

    assert float(loss) == pytest.approx(pred, rel=0, abs=0)
    assert logged == []
    assert _hook_count(unet) == 0


@pytest.mark.parametrize("arch", ARCHS)
def test_an_armed_step_without_pixels_is_the_plain_loss_too(arch):
    """repa_pixels is None whenever a clean image failed to load for the batch."""
    unet = _tiny_unet(arch).train()
    logged = []
    trainer = _train_step_trainer(arch, unet, logged)

    loss, pred, _ = _train_step(trainer, arch, repa_pixels=None)

    assert float(loss) == pytest.approx(pred, rel=0, abs=0)
    assert logged == []
    assert _hook_count(unet) == 0


@pytest.mark.parametrize("arch", ARCHS)
def test_the_hook_does_not_change_the_forward_bitwise(arch):
    unet = _tiny_unet(arch).eval()
    inputs = _unet_inputs(arch)
    block = repa_module.spatial_tap_sites(unet)[1][1]

    with torch.no_grad():
        plain = unet(**inputs).sample
        handle = repa_module.arm_spatial_tap(unet, block)
        try:
            armed = unet(**inputs).sample
        finally:
            handle.remove()

    assert torch.equal(plain, armed)
    assert unet._repa_tap_out is not None


@pytest.mark.parametrize("arch", ARCHS)
def test_the_hook_is_removed_even_when_the_forward_raises(arch):
    unet = _tiny_unet(arch).train()
    trainer = _train_step_trainer(arch, unet, [])
    original = type(unet).forward

    def _boom(*args, **kwargs):
        raise RuntimeError("forward exploded")

    type(unet).forward = _boom
    try:
        with pytest.raises(RuntimeError, match="forward exploded"):
            _train_step(trainer, arch, repa_pixels=_pixels())
    finally:
        type(unet).forward = original

    assert _hook_count(unet) == 0


@pytest.mark.parametrize("arch", ARCHS)
def test_a_completed_step_leaves_no_hook_and_an_empty_tap(arch):
    unet = _tiny_unet(arch).train()
    trainer = _train_step_trainer(arch, unet, [])

    _train_step(trainer, arch, repa_pixels=_pixels())

    assert _hook_count(unet) == 0
    assert unet._repa_tap_out is None


# ---------------------------------------------------------------------------
# (b) the handler's answer: three sites, not a block index
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ARCHS)
def test_the_handler_answers_with_the_unet_and_a_three_site_menu(arch):
    trainer = _trainer(arch)
    tap = trainer.arch.repa_tap(trainer)

    assert tap.module is trainer.unet
    assert tap.depth == 3
    n_down = len(trainer.unet.down_blocks)
    assert tap.site_labels == (f"down_blocks[{n_down - 1}]", "mid_block", "up_blocks[0]")


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("index", (0, 1, 2))
def test_the_width_is_the_measured_channel_count_of_that_site(arch, index):
    """Not read from the config: a config that no longer describes the loaded
    tree would size the projector wrong, and all three sites carry the same
    channel count, so a mis-resolved site would not raise either."""
    unet = _tiny_unet(arch)
    maps = _measured_site_maps(unet)
    label, block = repa_module.spatial_tap_sites(unet)[index]

    assert repa_module.spatial_site_width(label, block) == maps[label].shape[1]

    trainer = _trainer(arch, unet, repa_align_depth=index)
    assert trainer.arch.repa_tap(trainer).hidden_size == maps[label].shape[1]


@pytest.mark.parametrize("arch", ARCHS)
def test_the_default_site_is_the_mid_block(arch, monkeypatch):
    _stub_encoder_loader(monkeypatch)
    trainer = _trainer(arch)
    maps = _measured_site_maps(trainer.unet)

    BaseTrainer._setup_repa(trainer)

    assert trainer.repa_align_depth == 1
    assert trainer.unet._repa_tap_depth == 1
    assert trainer._repa_tap_module is trainer.unet
    assert trainer.repa_projector.net[0].in_features == maps["mid_block"].shape[1]
    assert trainer.repa_projector.net[-1].out_features == ENC_DIM


@pytest.mark.parametrize("arch", ARCHS)
def test_an_out_of_range_depth_is_refused_rather_than_clamped(arch):
    """The same key means a transformer block index on every other arch, where it
    clamps. Clamping 12 into a three-item menu would silently pick up_blocks[0]."""
    trainer = _trainer(arch, repa_align_depth=12)

    with pytest.raises(ValueError, match="out of range"):
        trainer.arch.repa_tap(trainer)


@pytest.mark.parametrize("arch", ARCHS)
def test_a_controlnet_run_is_refused(arch):
    """Its forward is train_step_controlnet, which never reads the tap, and the
    U-Net is frozen there."""
    trainer = _trainer(arch)
    trainer.use_condition_images = True

    with pytest.raises(ValueError, match="ControlNet run"):
        trainer.arch.repa_tap(trainer)


@pytest.mark.parametrize("arch", ARCHS)
def test_the_unets_do_not_consume_the_block_loop_features(arch):
    """TREAD / BlockSkip are not installed in this forward, so the depth-conflict
    check must not run against config keys the forward ignores."""
    assert ARCH_REGISTRY[arch].consumes_block_loop_features is False



def _capture_tokens(monkeypatch):
    """Record what the spatial helper hands the shared token-sequence path."""
    seen = {}
    real = repa_module.apply_repa_loss

    def _spy(trainer, loss, image_tokens, repa_pixels, gh, gw):
        seen.update(tokens=image_tokens, gh=gh, gw=gw)
        return real(trainer, loss, image_tokens, repa_pixels, gh, gw)

    monkeypatch.setattr(repa_module, "apply_repa_loss", _spy)
    return seen


def _spatial_trainer(logged):
    return SimpleNamespace(
        repa_encoder=_StubEncoder(), repa_size=32, repa_weight=1.0,
        repa_projector=repa_module.RepaProjector(5, ENC_DIM, hidden=8),
        device=torch.device("cpu"), training_dtype=torch.float32,
        log_extra_metric=lambda name, value: logged.append((name, value)))


def test_the_teacher_grid_is_the_taps_own_height_and_width(monkeypatch):
    seen = _capture_tokens(monkeypatch)
    trainer = _spatial_trainer([])
    feature_map = torch.randn(2, 5, 3, 7, requires_grad=True)

    repa_module.apply_repa_loss_spatial(trainer, torch.zeros(()), feature_map,
                                        torch.rand(2, 3, 32, 32))

    assert (seen["gh"], seen["gw"]) == (3, 7)
    assert seen["tokens"].shape == (2, 21, 5)


@pytest.mark.parametrize("cell", [(0, 0), (0, 6), (2, 3), (2, 6)])
def test_one_cell_of_the_map_is_one_row_of_the_sequence(cell, monkeypatch):
    """Row-major (h*gw + w), the order encode_repa_targets builds its grid in.
    A transposed flatten would move a different row -- gh != gw here so it shows."""
    seen = _capture_tokens(monkeypatch)
    trainer = _spatial_trainer([])
    gh, gw = 3, 7
    base = torch.zeros(1, 5, gh, gw, requires_grad=True)

    repa_module.apply_repa_loss_spatial(trainer, torch.zeros(()), base,
                                        torch.rand(1, 3, 32, 32))
    before = seen["tokens"].detach().clone()

    h, w = cell
    bumped = base.detach().clone()
    bumped[0, :, h, w] += 1.0
    repa_module.apply_repa_loss_spatial(trainer, torch.zeros(()),
                                        bumped.requires_grad_(True),
                                        torch.rand(1, 3, 32, 32))
    moved = (seen["tokens"].detach() - before).abs().sum(dim=-1)[0]

    assert int(moved.argmax()) == h * gw + w
    assert int((moved > 0).sum()) == 1


def test_the_teacher_targets_are_row_major_over_the_same_grid():
    """The other half of the correspondence: the encoder's square grid is
    interpolated to (gh, gw) and flattened in the same order."""
    gh, gw = 3, 7

    class _Positional(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(1))

        def forward(self, pixel_values):
            grid = torch.arange(ENC_GRID * ENC_GRID, dtype=torch.float32)
            feat = grid.reshape(1, -1, 1).expand(1, -1, ENC_DIM)
            return SimpleNamespace(last_hidden_state=feat.contiguous())

    targets = repa_module.encode_repa_targets(
        _Positional(), torch.zeros(1, 3, 32, 32), gh, gw, 32)

    values = targets[0, :, 0].reshape(gh, gw)
    # The source ramps left to right then top to bottom, so the resampled grid
    # must increase along both axes in that order.
    assert torch.all(values[:, 1:] > values[:, :-1])
    assert torch.all(values[1:, :] > values[:-1, :])


def test_a_token_sequence_handed_to_the_spatial_helper_is_refused():
    trainer = _spatial_trainer([])
    with pytest.raises(RuntimeError, match=r"\[B, C, h, w\]"):
        repa_module.apply_repa_loss_spatial(trainer, torch.zeros(()),
                                            torch.randn(1, 6, 5, requires_grad=True),
                                            torch.rand(1, 3, 32, 32))


def test_a_detached_map_is_refused_rather_than_contributing_nothing():
    """What a reentrant checkpoint around the tapped block would produce."""
    trainer = _spatial_trainer([])
    with pytest.raises(RuntimeError, match="detached tensor"):
        repa_module.apply_repa_loss_spatial(trainer, torch.zeros(()),
                                            torch.randn(1, 5, 3, 7),
                                            torch.rand(1, 3, 32, 32))


def test_a_teacher_batch_that_does_not_pair_with_the_tap_is_refused():
    trainer = _spatial_trainer([])
    with pytest.raises(RuntimeError, match="clean image"):
        repa_module.apply_repa_loss_spatial(trainer, torch.zeros(()),
                                            torch.randn(2, 5, 3, 7, requires_grad=True),
                                            torch.rand(1, 3, 32, 32))



@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("checkpointing", (False, True))
def test_the_alignment_term_alone_reaches_the_blocks_up_to_the_tap(arch, checkpointing):
    """Only the REPA term is back-propagated here, so a non-zero grad on a
    parameter is the alignment loss reaching it and nothing else. Everything
    after the tap must stay untouched -- the term is taken at the tap."""
    unet = _tiny_unet(arch).train()
    if checkpointing:
        unet.enable_gradient_checkpointing()
    logged = []
    trainer = _spatial_trainer(logged)
    trainer.repa_projector = repa_module.RepaProjector(
        repa_module.spatial_site_width(*repa_module.spatial_tap_sites(unet)[1]),
        ENC_DIM, hidden=16)

    handle = repa_module.arm_spatial_tap(unet, repa_module.spatial_tap_sites(unet)[1][1])
    try:
        unet(**_unet_inputs(arch, batch=1)).sample
    finally:
        handle.remove()

    tap = unet._repa_tap_out
    assert tap.grad_fn is not None, "the tap must stay grad-connected"
    loss = repa_module.apply_repa_loss_spatial(trainer, torch.zeros(()), tap,
                                               _pixels(batch=1))
    loss.backward()

    assert [name for name, _ in logged] == ["repa_loss"]
    assert float(unet.conv_in.weight.grad.abs().sum()) > 0
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in unet.down_blocks[0].parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in trainer.repa_projector.parameters())
    assert unet.conv_out.weight.grad is None



@pytest.mark.parametrize("arch", ARCHS)
def test_train_step_adds_a_finite_alignment_term_that_reaches_the_unet(arch):
    unet = _tiny_unet(arch).train()
    logged = []
    trainer = _train_step_trainer(arch, unet, logged)

    loss, pred, _ = _train_step(trainer, arch, repa_pixels=_pixels())
    loss.backward()

    assert torch.isfinite(loss)
    assert [name for name, _ in logged] == ["repa_loss"]
    rloss = logged[0][1]
    assert 0.0 <= rloss <= 2.0
    # The alignment term is ON TOP of the diffusion loss the trainer reports.
    assert float(loss) == pytest.approx(pred + trainer.repa_weight * rloss, rel=1e-5)
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in trainer.repa_projector.parameters())
    assert float(unet.conv_in.weight.grad.abs().sum()) > 0


@pytest.mark.parametrize("arch", ARCHS)
def test_a_step_whose_tap_never_fired_refuses_instead_of_dropping_the_term(
        arch, monkeypatch):
    unet = _tiny_unet(arch).train()
    trainer = _train_step_trainer(arch, unet, [])
    monkeypatch.setattr(repa_module, "arm_spatial_tap",
                        lambda container, block: SimpleNamespace(remove=lambda: None))

    with pytest.raises(RuntimeError, match="stashed nothing"):
        _train_step(trainer, arch, repa_pixels=_pixels())


@pytest.mark.parametrize("arch", ARCHS)
def test_three_steps_run_and_stay_finite(arch):
    unet = _tiny_unet(arch).train()
    logged = []
    trainer = _train_step_trainer(arch, unet, logged)
    opt = torch.optim.SGD(list(unet.parameters())
                          + list(trainer.repa_projector.parameters()), lr=1e-4)

    for step in range(3):
        opt.zero_grad(set_to_none=True)
        loss, pred, _ = _train_step(trainer, arch, repa_pixels=_pixels(), seed=step)
        loss.backward()
        opt.step()
        assert torch.isfinite(loss), step
        print(f"[{arch}] step {step}: loss={float(loss):.6f} pred={pred:.6f} "
              f"repa={logged[-1][1]:.6f}")

    assert [name for name, _ in logged] == ["repa_loss"] * 3
    assert all(0.0 <= value <= 2.0 for _, value in logged)


@pytest.mark.parametrize("arch", ARCHS)
def test_the_arch_handler_passes_the_batch_pixels_through(arch):
    trainer = _train_step_trainer(arch, _tiny_unet(arch), [])
    seen = {}

    def _spy(_trainer, **kwargs):
        seen.update(kwargs)
        return torch.zeros(()), 0.0, 0.0

    pixels = _pixels()
    original = sd_sdxl_ops.train_step
    sd_sdxl_ops.train_step = _spy
    try:
        trainer.arch.train_step(trainer, TrainStepContext(
            latents=torch.zeros(2, 4, LATENT_H, LATENT_W),
            text_embeddings=torch.zeros(2, 5, CROSS_DIM[arch]),
            repa_pixels=pixels))
    finally:
        sd_sdxl_ops.train_step = original

    assert seen["repa_pixels"] is pixels



@pytest.mark.parametrize("arch", ARCHS)
def test_a_full_finetune_save_pairs_the_projector_with_the_file_it_wrote(arch, tmp_path):
    """Both write_checkpoint bodies returned None: handed a directory, the base
    adapter could not tell which file inside it the sidecar belongs to."""
    from core.training.adapters.sd15_adapter import SD15FullParameterAdapter
    from core.training.adapters.sdxl_adapter import SDXLFullParameterAdapter

    cls = SDXLFullParameterAdapter if arch == "sdxl" else SD15FullParameterAdapter
    trainer = _trainer(arch)
    trainer.repa_enable = True
    trainer.repa_projector = repa_module.RepaProjector(8, ENC_DIM, hidden=16)
    trainer.train_unet = False
    trainer.train_text_encoder = False
    trainer.text_encoder = None
    trainer.text_encoder_2 = None
    trainer.vae = None
    trainer.bundle_vae = False
    trainer.sdxl_te_type = None

    written = cls(trainer).save_checkpoint(5, 0, tmp_path)

    assert written == tmp_path / "model_step_5.safetensors"
    assert (tmp_path / "model_step_5.repa.safetensors").is_file()



@pytest.mark.parametrize("arch", ARCHS)
def test_both_unets_are_offered_the_repa_control(arch):
    assert "repa" not in TRAINING_FEATURE_UNSUPPORTED.get(arch, {})


@pytest.mark.parametrize("arch", ("zimage", "flux2"))
def test_the_deferred_archs_say_they_are_held_not_merely_unwired(arch):
    reason = TRAINING_FEATURE_UNSUPPORTED[arch]["repa"]["reason"]
    assert "DEFERRED" in reason and "demand" in reason
    assert "not wired yet" not in reason

    with pytest.raises(ValueError) as excinfo:
        ARCH_REGISTRY[arch]().repa_tap(SimpleNamespace())
    assert "DEFERRED" in str(excinfo.value)
