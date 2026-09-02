"""Training preview: stacking extra LoRAs on the in-training model, CPU, ~5s.

Drives the REAL ``TrainingPreviewGenerator._apply_additional_loras`` /
``_detach_additional_loras`` (and through them the real ``LoRAManager`` and the
real diffusers loader mixin) over a stub trainer holding a toy-width
``UNet2DConditionModel`` and ``CLIPTextModel``. No hub assets, no GPU.

The case that matters is the one that never worked: a preview requested DURING
a LoRA run, where the trainer has already replaced the target Linears with its
own ``LoRALinearLayer``. PEFT refuses to wrap one, so the extra LoRA is loaded
with those wrappers detoured out and spliced back around the PEFT layer.

The detach gates deliberately exercise the awkward orderings -- detach after the
trainer swapped a module, detach after a load failed half way, detach when a
restore raises -- because the clean sequence hides exactly the bookkeeping bugs
this path can have.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/training_preview_lora_stack_cheap_test.py -v
"""

import types

import pytest
import torch

from lora_roundtrip_common import LoRALinearLayer, randomise_lora_layers

from diffusers import UNet2DConditionModel  # noqa: E402
from diffusers.loaders.lora_pipeline import (  # noqa: E402
    StableDiffusionLoraLoaderMixin as SDLoader,
)
from transformers import CLIPTextConfig, CLIPTextModel  # noqa: E402

import core.training.temp_pipeline as temp_pipeline  # noqa: E402
from core.extensions.lora_manager import lora_manager  # noqa: E402
from core.training.adapters.sd15_adapter import SD15LoRAAdapter  # noqa: E402
from core.training.temp_pipeline import LoraStackingUnsupported  # noqa: E402
from core.training.training_inference import TrainingPreviewGenerator  # noqa: E402

RANK = 4
ALPHA = 8
STRENGTH = 0.7

_UNET_KWARGS = dict(
    sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
    block_out_channels=(8, 16), norm_num_groups=4,
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
    cross_attention_dim=8, attention_head_dim=2,
)
_TE_CONFIG = CLIPTextConfig(vocab_size=64, hidden_size=8, intermediate_size=16,
                            num_hidden_layers=2, num_attention_heads=2,
                            max_position_embeddings=16)


def build_unet():
    torch.manual_seed(0)
    return UNet2DConditionModel(**_UNET_KWARGS)


def build_text_encoder():
    torch.manual_seed(0)
    return CLIPTextModel(_TE_CONFIG)


def unet_inputs():
    generator = torch.Generator().manual_seed(11)
    return (torch.randn(1, 4, 8, 8, generator=generator),
            torch.tensor([7]),
            torch.randn(1, 4, 8, generator=generator))


def unet_forward(unet, inputs):
    sample, timestep, encoder_hidden_states = inputs
    with torch.no_grad():
        return unet(sample, timestep,
                    encoder_hidden_states=encoder_hidden_states).sample


def install_training_lora(unet, text_encoder, seed=99):
    """What a LoRA run does to the trainer's modules before any preview."""
    trainer = types.SimpleNamespace(unet=unet, text_encoder=text_encoder,
                                    unet_lr=1e-4, text_encoder_1_lr=1e-5)
    adapter = SD15LoRAAdapter(trainer, lora_rank=RANK, lora_alpha=ALPHA,
                              lora_dtype=torch.float32)
    layers = {}
    adapter.apply_lora_to_unet(layers)
    adapter.apply_lora_to_text_encoders(layers)
    # lora_up initialises to zeros: without this the training branch is inert
    # and "detach restored the forward" would hold for the wrong reason.
    randomise_lora_layers(layers, seed=seed, std=0.2)
    return layers


def save_extra_lora(tmp_path, name="extra.safetensors", seed=4321):
    """A real kohya-format SD1.5 LoRA file, written by the real adapter."""
    unet, text_encoder = build_unet(), build_text_encoder()
    trainer = types.SimpleNamespace(unet=unet, text_encoder=text_encoder,
                                    unet_lr=1e-4, text_encoder_1_lr=1e-5)
    adapter = SD15LoRAAdapter(trainer, lora_rank=RANK, lora_alpha=ALPHA,
                              lora_dtype=torch.float32)
    layers = {}
    adapter.apply_lora_to_unet(layers)
    adapter.apply_lora_to_text_encoders(layers)
    randomise_lora_layers(layers, seed=seed, std=0.3)
    adapter.save_checkpoint(layers, step=1, epoch=0, output_path=tmp_path / name)
    return name


def stub_trainer(unet, text_encoder, arch="sd15"):
    return types.SimpleNamespace(
        unet=unet, vae=None, text_encoder=text_encoder, text_encoder_2=None,
        tokenizer=None, tokenizer_2=None, is_sdxl=False,
        original_scheduler=None, arch=types.SimpleNamespace(name=arch),
    )


def peft_paths(model):
    return {name for name, module in model.named_modules()
            if getattr(module, "lora_A", None) is not None and len(module.lora_A)}


def wrapper_paths(model):
    return {name for name, module in model.named_modules()
            if isinstance(module, LoRALinearLayer)}


def module_at(model, path):
    return dict(model.named_modules())[path]


@pytest.fixture
def loras_in(tmp_path, monkeypatch):
    """Point the production LoRAManager singleton at the fixture directory."""
    monkeypatch.setattr(lora_manager, "lora_dir", tmp_path)
    monkeypatch.setattr(lora_manager, "seeded_dirs", [])
    monkeypatch.setattr(lora_manager, "additional_dirs", [])
    return tmp_path


def request_for(name, strength=STRENGTH):
    return [{"path": name, "strength": strength}]


# ---------------------------------------------------------------------------
# The case the feature exists for: stacking during a LoRA run.
# ---------------------------------------------------------------------------

def test_stack_over_an_in_training_lora_changes_the_forward(loras_in):
    name = save_extra_lora(loras_in)
    unet, text_encoder = build_unet(), build_text_encoder()
    install_training_lora(unet, text_encoder)
    inputs = unet_inputs()
    before = unet_forward(unet, inputs)

    generator = TrainingPreviewGenerator(stub_trainer(unet, text_encoder))
    generator._apply_additional_loras(request_for(name))

    # Structural claim: PEFT sits INSIDE the trainer's wrapper, so the preview
    # forward is base + extra + in-training, not one instead of the other.
    wrapped = wrapper_paths(unet)
    assert wrapped, "setup: the trainer must have wrapped some Linears"
    assert all(hasattr(module_at(unet, path).original_module, "base_layer")
               for path in wrapped)
    assert peft_paths(unet) == {f"{path}.original_module" for path in wrapped}
    assert not torch.allclose(unet_forward(unet, inputs), before, atol=1e-6)


def test_detach_restores_the_pre_preview_forward_and_module_identity(loras_in):
    name = save_extra_lora(loras_in)
    unet, text_encoder = build_unet(), build_text_encoder()
    install_training_lora(unet, text_encoder)
    inputs = unet_inputs()
    before = unet_forward(unet, inputs)
    wrapped = wrapper_paths(unet) | {f"te::{p}" for p in wrapper_paths(text_encoder)}
    wrappers = {p: module_at(unet, p) for p in wrapper_paths(unet)}
    inners = {p: w.original_module for p, w in wrappers.items()}
    te_wrappers = {p: module_at(text_encoder, p) for p in wrapper_paths(text_encoder)}
    te_inners = {p: w.original_module for p, w in te_wrappers.items()}

    generator = TrainingPreviewGenerator(stub_trainer(unet, text_encoder))
    generator._apply_additional_loras(request_for(name))
    generator._detach_additional_loras()

    # id(), not weight equality: the optimizer holds these parameters, so a
    # faithful-looking copy in their place would silently orphan the run.
    for path, wrapper in wrappers.items():
        assert module_at(unet, path) is wrapper, path
        assert wrapper.original_module is inners[path], path
    for path, wrapper in te_wrappers.items():
        assert module_at(text_encoder, path) is wrapper, path
        assert wrapper.original_module is te_inners[path], path
    assert not peft_paths(unet) and not peft_paths(text_encoder)
    assert wrapped == (wrapper_paths(unet)
                       | {f"te::{p}" for p in wrapper_paths(text_encoder)})
    assert torch.allclose(unet_forward(unet, inputs), before, atol=0.0)
    assert generator._lora_stack is None


def test_full_finetune_preview_stacks_and_detaches(loras_in):
    """No trainer wrappers at all: the detour must be a no-op, not a hazard."""
    name = save_extra_lora(loras_in)
    unet, text_encoder = build_unet(), build_text_encoder()
    inputs = unet_inputs()
    before = unet_forward(unet, inputs)
    originals = {name_: module for name_, module in unet.named_modules()}

    generator = TrainingPreviewGenerator(stub_trainer(unet, text_encoder))
    generator._apply_additional_loras(request_for(name))
    assert peft_paths(unet), "the extra LoRA reached the U-Net"
    assert not torch.allclose(unet_forward(unet, inputs), before, atol=1e-6)

    generator._detach_additional_loras()
    assert not peft_paths(unet) and not peft_paths(text_encoder)
    for path, module in originals.items():
        assert module_at(unet, path) is module, path
    assert torch.allclose(unet_forward(unet, inputs), before, atol=0.0)


def test_strength_reaches_the_stacked_adapter(loras_in):
    """LoRAManager forwards the per-LoRA strength through set_adapters; a
    preview that ignored it would look plausible and be wrong."""
    name = save_extra_lora(loras_in)
    unet, text_encoder = build_unet(), build_text_encoder()
    install_training_lora(unet, text_encoder)

    generator = TrainingPreviewGenerator(stub_trainer(unet, text_encoder))
    generator._apply_additional_loras(request_for(name, strength=0.5))

    scalings = {round(module_at(unet, path).scaling["lora_0"], 9)
                for path in peft_paths(unet)}
    assert scalings == {round((ALPHA / RANK) * 0.5, 9)}


# ---------------------------------------------------------------------------
# Refusals.
# ---------------------------------------------------------------------------

def test_non_sd_architecture_refuses_by_name(loras_in):
    name = save_extra_lora(loras_in)
    fake_dit = torch.nn.Linear(4, 4)
    generator = TrainingPreviewGenerator(stub_trainer(fake_dit, None, arch="zimage"))

    with pytest.raises(LoraStackingUnsupported) as excinfo:
        generator._apply_additional_loras(request_for(name))

    message = str(excinfo.value)
    assert "zimage" in message and "SD1.5 and SDXL only" in message
    assert generator._lora_stack is None


def test_preexisting_peft_adapters_refuse(loras_in):
    """Nothing in the trainer installs PEFT, so this is state the detach could
    not promise to restore. Refuse rather than delete someone else's adapter."""
    name = save_extra_lora(loras_in)
    unet, text_encoder = build_unet(), build_text_encoder()
    state_dict, alphas, metadata = SDLoader.lora_state_dict(
        str(loras_in), weight_name=name, return_lora_metadata=True)
    SDLoader.load_lora_into_unet(state_dict, network_alphas=alphas, unet=unet,
                                 adapter_name="someone_else", metadata=metadata)
    installed = peft_paths(unet)

    generator = TrainingPreviewGenerator(stub_trainer(unet, text_encoder))
    with pytest.raises(RuntimeError, match="already"):
        generator._apply_additional_loras(request_for(name))

    assert peft_paths(unet) == installed
    assert list(unet.peft_config) == ["someone_else"]
    assert generator._lora_stack is None


# ---------------------------------------------------------------------------
# Awkward detach orderings.
# ---------------------------------------------------------------------------

def test_a_stack_that_fails_half_way_still_detaches_what_loaded(loras_in):
    name = save_extra_lora(loras_in)
    unet, text_encoder = build_unet(), build_text_encoder()
    install_training_lora(unet, text_encoder)
    inputs = unet_inputs()
    before = unet_forward(unet, inputs)

    generator = TrainingPreviewGenerator(stub_trainer(unet, text_encoder))
    with pytest.raises(FileNotFoundError):
        generator._apply_additional_loras(
            request_for(name) + [{"path": "absent.safetensors", "strength": 1.0}])

    assert peft_paths(unet), "setup: the first LoRA of the stack did load"
    generator._detach_additional_loras()
    assert not peft_paths(unet) and not peft_paths(text_encoder)
    assert torch.allclose(unet_forward(unet, inputs), before, atol=0.0)


def test_detach_follows_the_modules_it_loaded_into_after_a_swap(loras_in):
    """Rebuilding the pipeline at detach time would clean the CURRENT modules
    and leave the loaded ones carrying the preview adapters."""
    name = save_extra_lora(loras_in)
    unet, text_encoder = build_unet(), build_text_encoder()
    install_training_lora(unet, text_encoder)
    inputs = unet_inputs()
    before = unet_forward(unet, inputs)
    trainer = stub_trainer(unet, text_encoder)

    generator = TrainingPreviewGenerator(trainer)
    generator._apply_additional_loras(request_for(name))

    replacement = build_unet()
    trainer.unet = replacement
    generator._detach_additional_loras()

    assert not peft_paths(unet), "the loaded U-Net kept preview adapters"
    assert not peft_paths(replacement)
    assert torch.allclose(unet_forward(unet, inputs), before, atol=0.0)


def test_a_failing_restore_is_loud_and_still_restores_the_rest(loras_in,
                                                               monkeypatch):
    name = save_extra_lora(loras_in)
    unet, text_encoder = build_unet(), build_text_encoder()
    install_training_lora(unet, text_encoder)

    generator = TrainingPreviewGenerator(stub_trainer(unet, text_encoder))
    generator._apply_additional_loras(request_for(name))
    sites = generator._lora_stack["sites"]
    # The LAST site, so the reversed restore loop hits it FIRST: a loop that
    # aborted on the first failure would strand every other wrapper too.
    victim = sites[-1][2]

    real_set_child = temp_pipeline._set_child

    def flaky_set_child(parent, attr, module):
        # Only the splice-BACK passes the wrapper itself as `module`.
        if module is victim:
            raise RuntimeError("injected splice-in failure")
        real_set_child(parent, attr, module)

    monkeypatch.setattr(temp_pipeline, "_set_child", flaky_set_child)
    with pytest.raises(RuntimeError, match="could not be detached"):
        generator._detach_additional_loras()

    # Every other wrapper is back where it belongs; only the victim is stranded.
    survivors = [m for model in (unet, text_encoder)
                 for _n, m in model.named_modules()
                 if isinstance(m, LoRALinearLayer)]
    assert victim not in survivors
    assert len(survivors) == len(sites) - 1
    assert not peft_paths(unet) and not peft_paths(text_encoder)
    # Cleared, so the next preview starts from a known state instead of
    # retrying a detach against a half-restored model.
    assert generator._lora_stack is None


def test_detach_without_an_apply_is_a_no_op():
    generator = TrainingPreviewGenerator(stub_trainer(build_unet(),
                                                      build_text_encoder()))
    generator._detach_additional_loras()
    generator._detach_additional_loras()
    assert generator._lora_stack is None
