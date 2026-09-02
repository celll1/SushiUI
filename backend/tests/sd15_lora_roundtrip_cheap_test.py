"""SD1.5: trainer save -> fresh-generation load round trip, CPU, ~2s.

Drives the REAL ``SD15LoRAAdapter`` (injection + ``save_checkpoint``) over a
toy-width ``UNet2DConditionModel`` and ``CLIPTextModel`` (63k + 2k parameters,
no hub assets), then the REAL generation-side conversion and injection.

WHAT "the generation loader" MEANS HERE. SD1.5 and SDXL do not have a
SushiUI-owned loader: ``LoRAManager.load_loras`` hands the file to
``pipeline.load_lora_weights``, whose whole body is
``lora_state_dict`` -> ``load_lora_into_unet`` -> ``load_lora_into_text_encoder``.
This file calls those three directly, because constructing a
``StableDiffusionPipeline`` needs a tokenizer and scheduler that are not
available offline. So the kohya-key conversion and the PEFT injection under
test are the production ones; only the pipeline wrapper around them is not.

NOT COVERED HERE: the refusal contract. ``LoRAManager.load_loras`` skips a
missing file with a console print and wraps its whole body in
``except Exception: print``, so no SD1.5/SDXL LoRA failure raises or reaches
``warnings[]``. That is a defect in that file, not something to assert around.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/sd15_lora_roundtrip_cheap_test.py -v
"""

import types

import torch
from safetensors.torch import load_file, save_file

from lora_roundtrip_common import lora_delta, module_ids, randomise_lora_layers

from diffusers import UNet2DConditionModel  # noqa: E402
from diffusers.loaders.lora_pipeline import (  # noqa: E402
    StableDiffusionLoraLoaderMixin as SDLoader,
)
from transformers import CLIPTextConfig, CLIPTextModel  # noqa: E402

from core.training.adapters.sd15_adapter import SD15LoRAAdapter  # noqa: E402

RANK = 4
ALPHA = 8  # != rank on purpose: PEFT's scaling must be 2.0, not 1.0
SCALE = ALPHA / RANK
STRENGTH = 0.5

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


def trainer_wrapped_paths(model):
    from lora_roundtrip_common import LoRALinearLayer
    return {name for name, module in model.named_modules()
            if isinstance(module, LoRALinearLayer)}


def peft_wrapped_paths(model):
    """Where PEFT actually installed an adapter, by dotted module path."""
    return {name for name, module in model.named_modules()
            if getattr(module, "lora_A", None) is not None and len(module.lora_A)}


def train_and_save(tmp_path, name="sd15.safetensors", seed=1234):
    unet, text_encoder = build_unet(), build_text_encoder()
    trainer = types.SimpleNamespace(unet=unet, text_encoder=text_encoder,
                                    unet_lr=1e-4, text_encoder_1_lr=1e-5)
    adapter = SD15LoRAAdapter(trainer, lora_rank=RANK, lora_alpha=ALPHA,
                              lora_dtype=torch.float32)
    layers = {}
    n_unet = adapter.apply_lora_to_unet(layers)
    n_te = adapter.apply_lora_to_text_encoders(layers)
    assert n_unet > 0 and n_te > 0
    randomise_lora_layers(layers, seed=seed, std=0.2)
    adapter.save_checkpoint(layers, step=1, epoch=0, output_path=tmp_path / name)
    return (str(tmp_path), name,
            trainer_wrapped_paths(unet), trainer_wrapped_paths(text_encoder))


def load_into(directory, name, unet, text_encoder, adapter_name="t"):
    """The three calls ``pipeline.load_lora_weights`` makes, verbatim."""
    state_dict, network_alphas, metadata = SDLoader.lora_state_dict(
        directory, weight_name=name, unet_config=unet.config, return_lora_metadata=True)
    SDLoader.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=unet,
                                 adapter_name=adapter_name, metadata=metadata)
    SDLoader.load_lora_into_text_encoder(state_dict, network_alphas=network_alphas,
                                         text_encoder=text_encoder,
                                         adapter_name=adapter_name, metadata=metadata)


def test_sd15_generation_wraps_exactly_the_targets_the_trainer_wrapped(tmp_path):
    directory, name, unet_paths, te_paths = train_and_save(tmp_path)

    unet, text_encoder = build_unet(), build_text_encoder()
    load_into(directory, name, unet, text_encoder)

    # Set EQUALITY on both components: a partial conversion (kohya stem -> module
    # path) is as wrong as none and much quieter.
    assert peft_wrapped_paths(unet) == unet_paths
    assert peft_wrapped_paths(text_encoder) == te_paths


def test_sd15_wrapped_forward_is_base_plus_scaled_branch(tmp_path):
    directory, name, unet_paths, te_paths = train_and_save(tmp_path)
    saved = load_file(f"{directory}/{name}")

    unet, text_encoder = build_unet(), build_text_encoder()
    load_into(directory, name, unet, text_encoder)
    from diffusers.utils.peft_utils import set_weights_and_activate_adapters

    unet.set_adapters(["t"], [STRENGTH])
    # What pipeline.set_adapters() does for a text encoder, which has no
    # set_adapters() of its own.
    set_weights_and_activate_adapters(text_encoder, ["t"], [STRENGTH])

    checked = 0
    for model, prefix in ((unet, "lora_unet_"), (text_encoder, "lora_te1_")):
        modules = dict(model.named_modules())
        for target in sorted(peft_wrapped_paths(model)):
            wrapper = modules[target]
            stem = prefix + target.replace(".", "_")
            x = torch.randn(3, wrapper.base_layer.in_features)
            base = wrapper.base_layer(x)
            expected = base + lora_delta(saved[f"{stem}.lora_down.weight"],
                                         saved[f"{stem}.lora_up.weight"],
                                         x, ALPHA, RANK, STRENGTH)
            assert torch.allclose(wrapper(x), expected, atol=1e-5), target
            assert not torch.allclose(wrapper(x), base, atol=1e-5), f"{target}: inert"
            checked += 1
    assert checked == len(unet_paths) + len(te_paths)


def test_sd15_alpha_beats_the_rank_fallback(tmp_path):
    """The adapter writes a per-key ``.alpha``; diffusers turns it into PEFT's
    ``scaling``. alpha=8 over rank=4 must be 2.0, not 1.0."""
    directory, name, unet_paths, te_paths = train_and_save(tmp_path)
    unet, text_encoder = build_unet(), build_text_encoder()
    load_into(directory, name, unet, text_encoder)

    modules = dict(unet.named_modules())
    assert {round(modules[t].scaling["t"], 9) for t in unet_paths} == {round(SCALE, 9)}
    te_modules = dict(text_encoder.named_modules())
    assert {round(te_modules[t].scaling["t"], 9) for t in te_paths} == {round(SCALE, 9)}

    # Strength multiplies it; this is the call LoRAManager makes on the pipeline.
    unet.set_adapters(["t"], [STRENGTH])
    assert {round(modules[t].scaling["t"], 9) for t in unet_paths} == \
        {round(SCALE * STRENGTH, 9)}

    # Same tensors with no alpha anywhere fall back to rank: scaling 1.0.
    stripped = {k: v for k, v in load_file(f"{directory}/{name}").items()
                if not k.endswith(".alpha")}
    save_file(stripped, f"{directory}/no_alpha.safetensors")
    unet2, te2 = build_unet(), build_text_encoder()
    load_into(directory, "no_alpha.safetensors", unet2, te2)
    modules2 = dict(unet2.named_modules())
    assert {round(modules2[t].scaling["t"], 9) for t in unet_paths} == {1.0}


def test_sd15_unload_restores_the_identical_objects_and_is_idempotent(tmp_path):
    directory, name, unet_paths, te_paths = train_and_save(tmp_path)

    unet, text_encoder = build_unet(), build_text_encoder()
    before_unet = {p: dict(unet.named_modules())[p] for p in unet_paths}
    before_te = {p: dict(text_encoder.named_modules())[p] for p in te_paths}
    load_into(directory, name, unet, text_encoder)
    assert peft_wrapped_paths(unet) == unet_paths

    from diffusers.utils import recurse_remove_peft_layers

    unet.unload_lora()
    recurse_remove_peft_layers(text_encoder)

    after_unet = dict(unet.named_modules())
    for target, original in before_unet.items():
        # id(), not tensor equality: PEFT wraps the ORIGINAL nn.Linear as
        # base_layer, and the unwrap must put that same object back rather than
        # a copy carrying the same weights.
        assert after_unet[target] is original, target
    after_te = dict(text_encoder.named_modules())
    for target, original in before_te.items():
        assert after_te[target] is original, target
    assert not peft_wrapped_paths(unet) and not peft_wrapped_paths(text_encoder)

    unet.unload_lora()  # second unload: a no-op, not a re-splice
    recurse_remove_peft_layers(text_encoder)
    assert dict(unet.named_modules()) == after_unet


def test_sd15_a_loaded_lora_lives_inside_its_own_model(tmp_path):
    """The reload gate, in the only form SD1.5 admits: PEFT installs the
    adapters INSIDE the model, so there is no manager-side module map that
    could outlive a model swap. Assert exactly that -- model B, built after A
    was wrapped, shares nothing with A and unwraps to itself."""
    directory, name, unet_paths, _te_paths = train_and_save(tmp_path)

    unet_a, te_a = build_unet(), build_text_encoder()
    load_into(directory, name, unet_a, te_a)
    a_ids = module_ids(unet_a) | module_ids(te_a)

    unet_b, te_b = build_unet(), build_text_encoder()
    b_ids_before = module_ids(unet_b) | module_ids(te_b)
    assert not (a_ids & b_ids_before), "setup: A and B must not already share modules"

    from diffusers.utils import recurse_remove_peft_layers

    unet_b.unload_lora()
    recurse_remove_peft_layers(te_b)
    assert module_ids(unet_b) | module_ids(te_b) == b_ids_before
    assert not (module_ids(unet_b) & a_ids)
    assert peft_wrapped_paths(unet_a) == unet_paths, "model A lost its adapters"

    # B loads independently and unwraps to its OWN modules.
    b_before = dict(unet_b.named_modules())
    load_into(directory, name, unet_b, te_b, adapter_name="second")
    assert peft_wrapped_paths(unet_b) == unet_paths
    unet_b.unload_lora()
    for target in unet_paths:
        assert dict(unet_b.named_modules())[target] is b_before[target], target
    assert not (module_ids(unet_b) & a_ids)
