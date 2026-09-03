"""SD1.5: trainer save -> fresh-generation load round trip, CPU, ~2s.

Drives the REAL ``SD15LoRAAdapter`` (injection + ``save_checkpoint``) over a
toy-width ``UNet2DConditionModel`` and ``CLIPTextModel`` (63k + 2k parameters,
no hub assets), then the REAL generation-side conversion and injection.

WHAT "the generation loader" MEANS HERE. SD1.5 and SDXL do not have a
SushiUI-owned loader: ``LoRAManager.load_loras`` hands the file to
``pipeline.load_lora_weights``, whose whole body is
``lora_state_dict`` -> ``load_lora_into_unet`` -> ``load_lora_into_text_encoder``.
The round-trip gates call those three directly, because constructing a
``StableDiffusionPipeline`` needs a tokenizer and scheduler that are not
available offline. So the kohya-key conversion and the PEFT injection under
test are the production ones; only the pipeline wrapper around them is not.

The refusal gates at the bottom drive the real ``load_loras`` over a facade
that subclasses the real diffusers mixin, so its ``load_lora_weights`` is the
production one too.

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


# ---------------------------------------------------------------------------
# The refusal contract, through the production entry point.
#
# The caveat at the top of this file used to end here: LoRAManager.load_loras
# skipped a missing file and swallowed every exception, so nothing could be
# asserted. It now refuses, and these gates drive the REAL load_loras over a
# pipeline facade that subclasses the REAL diffusers mixin -- load_lora_weights,
# set_adapters and unload_lora_weights are the ones a StableDiffusionPipeline
# would run; only the tokenizer/scheduler surface is absent.
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from lora_roundtrip_common import warning_codes, warning_probe  # noqa: E402

from core.extensions.lora_manager import LoRAManager  # noqa: E402


class Pipeline(SDLoader):
    _lora_loadable_modules = ["unet", "text_encoder"]
    hf_device_map = None

    def __init__(self, unet, text_encoder):
        self.unet = unet
        self.text_encoder = text_encoder

    @property
    def components(self):
        return {"unet": self.unet, "text_encoder": self.text_encoder}


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)


def manager_for(directory):
    manager = LoRAManager(lora_dir=str(directory))
    # The repo's training/ dir is seeded at construction; drop it so resolution
    # depends only on the fixture directory.
    manager.seeded_dirs = []
    return manager


def load_through_manager(tmp_path, name, unet, text_encoder, strength=1.0, **extra):
    pipeline = Pipeline(unet, text_encoder)
    manager_for(tmp_path).load_loras(
        pipeline, [{"path": name, "strength": strength, **extra}])
    return pipeline


def test_sd15_manager_applies_a_working_lora_and_warns_about_nothing(tmp_path,
                                                                    warnings_seen):
    """The false-refusal gate: the path most users exercise must stay silent."""
    _directory, name, unet_paths, te_paths = train_and_save(tmp_path)

    unet, text_encoder = build_unet(), build_text_encoder()
    load_through_manager(tmp_path, name, unet, text_encoder, strength=STRENGTH)

    assert peft_wrapped_paths(unet) == unet_paths
    assert peft_wrapped_paths(text_encoder) == te_paths
    assert warning_codes(warnings_seen) == []
    modules = dict(unet.named_modules())
    assert {round(modules[t].scaling["lora_0"], 9) for t in unet_paths} == \
        {round(SCALE * STRENGTH, 9)}


def test_sd15_manager_still_honours_unet_layer_weights(tmp_path, warnings_seen):
    """``unet_layer_weights`` is honoured only on this path. The refusal work
    must not have cost it."""
    _directory, name, unet_paths, _te = train_and_save(tmp_path)

    unet, text_encoder = build_unet(), build_text_encoder()
    load_through_manager(tmp_path, name, unet, text_encoder, strength=STRENGTH,
                         unet_layer_weights={"IN01": 0.25})

    modules = dict(unet.named_modules())
    scaled = {t for t in unet_paths if t.startswith("down_blocks.1.")}
    assert scaled, "setup: the toy UNet must have a down_blocks.1 target"
    assert {round(modules[t].scaling["lora_0"], 9) for t in scaled} == \
        {round(SCALE * STRENGTH * 0.25, 9)}
    assert {round(modules[t].scaling["lora_0"], 9) for t in unet_paths - scaled} == \
        {round(SCALE * STRENGTH, 9)}
    assert warning_codes(warnings_seen) == []


def test_sd15_missing_file_refuses_and_warns(tmp_path, warnings_seen):
    with pytest.raises(FileNotFoundError):
        load_through_manager(tmp_path, "no_such_sd15_lora.safetensors",
                             build_unet(), build_text_encoder())
    assert "lora_not_found" in warning_codes(warnings_seen)


def test_sd15_unreadable_file_refuses_and_names_no_path(tmp_path, warnings_seen):
    (tmp_path / "corrupt.safetensors").write_bytes(b"not a safetensors header")

    with pytest.raises(RuntimeError) as excinfo:
        load_through_manager(tmp_path, "corrupt.safetensors",
                             build_unet(), build_text_encoder())
    assert "lora_load_failed" in warning_codes(warnings_seen)
    message = warnings_seen[-1][1]
    assert "corrupt.safetensors" in message
    # The warning reaches a PNG text chunk, the API response and the DB.
    assert str(tmp_path) not in message and str(tmp_path) not in str(excinfo.value)


def test_sd15_lora_naming_absent_unet_modules_refuses(tmp_path, warnings_seen):
    """Kohya stems this UNet does not have. PEFT raises inside the apply, so
    this lands on lora_load_failed rather than lora_incompatible -- both are
    refusals, and the console log carries the PEFT message."""
    directory, name, _unet_paths, _te = train_and_save(tmp_path)
    ghost = {}
    for key, value in load_file(f"{directory}/{name}").items():
        if key.startswith("lora_unet_"):
            ghost["lora_unet_ghost_" + key[len("lora_unet_"):]] = value
    save_file(ghost, f"{directory}/ghost.safetensors")

    unet, text_encoder = build_unet(), build_text_encoder()
    with pytest.raises(RuntimeError):
        load_through_manager(tmp_path, "ghost.safetensors", unet, text_encoder)
    assert "lora_load_failed" in warning_codes(warnings_seen)
    assert not peft_wrapped_paths(unet)


def test_sd15_lora_for_another_architecture_refuses_and_warns(tmp_path,
                                                              warnings_seen):
    """The silent case: a Z-Image-style checkpoint matches neither the ``unet.``
    nor the ``text_encoder.`` prefix, so diffusers filters it to nothing and
    returns without raising. Only the read-back target count sees it."""
    foreign = {}
    for i in range(3):
        stem = f"lora_transformer_layers_{i}_attention_to_q"
        foreign[f"{stem}.lora_down.weight"] = torch.zeros(RANK, 8)
        foreign[f"{stem}.lora_up.weight"] = torch.zeros(8, RANK)
        foreign[f"{stem}.alpha"] = torch.tensor(float(ALPHA))
    save_file(foreign, str(tmp_path / "zimage_style.safetensors"))

    unet, text_encoder = build_unet(), build_text_encoder()
    with pytest.raises(RuntimeError, match="0 of 3 down/up"):
        load_through_manager(tmp_path, "zimage_style.safetensors", unet, text_encoder)
    assert "lora_incompatible" in warning_codes(warnings_seen)
    assert not peft_wrapped_paths(unet) and not peft_wrapped_paths(text_encoder)


def test_sd15_partly_matching_lora_applies_and_warns(tmp_path, warnings_seen):
    """Survivable: the real halves apply, the two invented stems do not."""
    directory, name, unet_paths, te_paths = train_and_save(tmp_path)
    partial = dict(load_file(f"{directory}/{name}"))
    for i in range(2):
        stem = f"lora_unet_ghost_block_{i}"
        partial[f"{stem}.lora_down.weight"] = torch.zeros(RANK, 8)
        partial[f"{stem}.lora_up.weight"] = torch.zeros(8, RANK)
        partial[f"{stem}.alpha"] = torch.tensor(float(ALPHA))
    save_file(partial, f"{directory}/partial.safetensors")

    unet, text_encoder = build_unet(), build_text_encoder()
    load_through_manager(tmp_path, "partial.safetensors", unet, text_encoder)

    assert peft_wrapped_paths(unet) == unet_paths
    assert peft_wrapped_paths(text_encoder) == te_paths
    assert warning_codes(warnings_seen) == ["lora_partial"]
    applied = len(unet_paths) + len(te_paths)
    assert f"applied {applied} of {applied + 2}" in warnings_seen[-1][1]


# ---------------------------------------------------------------------------
# Stacking. SD1.5 and SDXL are the two architectures that do NOT go through
# CompositeAdapterLayer: diffusers injects a PEFT adapter per file, and PEFT
# sums every ACTIVE one natively. The defect these gates pin was ours -- the
# loader activated one adapter name per file, and set_adapters REPLACES the
# active set, so every LoRA but the last was installed, counted and inert.
# ---------------------------------------------------------------------------

STRENGTH_B = 0.4  # the second LoRA's, so a shared weight shows up as a wrong sum


def stack_through_manager(tmp_path, configs, unet, text_encoder):
    pipeline = Pipeline(unet, text_encoder)
    manager = manager_for(tmp_path)
    manager.load_loras(pipeline, configs)
    return pipeline, manager


def stackable_targets(directory, name, unet_paths):
    """UNet targets whose kohya stem is present in the file, with the tensors."""
    saved = load_file(f"{directory}/{name}")
    found = {}
    for target in sorted(unet_paths):
        stem = "lora_unet_" + target.replace(".", "_")
        if f"{stem}.lora_down.weight" in saved:
            found[target] = (saved[f"{stem}.lora_down.weight"],
                             saved[f"{stem}.lora_up.weight"])
    assert found, "setup: no kohya stem round-tripped"
    return found


def test_sd15_two_loras_stay_active_together_and_their_deltas_sum(tmp_path,
                                                                  warnings_seen):
    directory, name_a, unet_paths, _te = train_and_save(tmp_path, "a.safetensors", seed=1234)
    _d, name_b, _u, _t = train_and_save(tmp_path, "b.safetensors", seed=4321)
    a_tensors = stackable_targets(directory, name_a, unet_paths)
    b_tensors = stackable_targets(directory, name_b, unet_paths)

    unet, text_encoder = build_unet(), build_text_encoder()
    pipeline, _m = stack_through_manager(
        tmp_path, [{"path": name_a, "strength": STRENGTH},
                   {"path": name_b, "strength": STRENGTH_B}], unet, text_encoder)

    assert set(pipeline.get_active_adapters()) == {"lora_0", "lora_1"}
    assert warning_codes(warnings_seen) == []

    base_modules = dict(build_unet().named_modules())
    modules = dict(unet.named_modules())
    for target, (down_a, up_a) in a_tensors.items():
        down_b, up_b = b_tensors[target]
        module = modules[target]
        # Both adapters present, each at ITS OWN strength: one shared weight
        # would put the same number on both.
        assert set(module.lora_A.keys()) == {"lora_0", "lora_1"}, target
        assert round(module.scaling["lora_0"], 9) == round(SCALE * STRENGTH, 9)
        assert round(module.scaling["lora_1"], 9) == round(SCALE * STRENGTH_B, 9)

        base = base_modules[target]
        x = torch.randn(2, base.in_features)
        expected = (base(x)
                    + lora_delta(down_a, up_a, x, ALPHA, RANK, STRENGTH)
                    + lora_delta(down_b, up_b, x, ALPHA, RANK, STRENGTH_B))
        assert torch.allclose(module(x), expected, atol=1e-5), target
        # Either alone is a different answer -- the last-wins defect passed the
        # "an adapter is installed" check and failed exactly here.
        assert not torch.allclose(
            module(x), base(x) + lora_delta(down_b, up_b, x, ALPHA, RANK, STRENGTH_B),
            atol=1e-5), f"{target}: only the last LoRA is in effect"
        assert not torch.allclose(
            module(x), base(x) + lora_delta(down_a, up_a, x, ALPHA, RANK, STRENGTH),
            atol=1e-5), f"{target}: only the first LoRA is in effect"


def test_sd15_stacked_result_is_independent_of_selection_order(tmp_path):
    """To one fp32 ULP, NOT bit-exactly, and that is a real difference from the
    composite architectures.

    ``peft.tuners.lora.Linear.forward`` accumulates INTO the base
    (``result = result + delta`` per active adapter), so swapping two adapters
    reassociates a three-operand sum. ``CompositeAdapterLayer`` adds the deltas
    to each other first and the base once, which commutes exactly for two.
    Measured on the toy UNet: 0 of 40 targets bit-identical, worst relative
    deviation 1.27e-07 against an fp32 eps of 1.19e-07.
    """
    directory, name_a, unet_paths, _te = train_and_save(tmp_path, "a.safetensors", seed=1234)
    _d, name_b, _u, _t = train_and_save(tmp_path, "b.safetensors", seed=4321)
    a = {"path": name_a, "strength": STRENGTH}
    b = {"path": name_b, "strength": STRENGTH_B}

    forward_unet, forward_te = build_unet(), build_text_encoder()
    stack_through_manager(tmp_path, [a, b], forward_unet, forward_te)
    reverse_unet, reverse_te = build_unet(), build_text_encoder()
    stack_through_manager(tmp_path, [b, a], reverse_unet, reverse_te)

    forward_modules = dict(forward_unet.named_modules())
    reverse_modules = dict(reverse_unet.named_modules())
    for target in stackable_targets(directory, name_a, unet_paths):
        one, two = forward_modules[target], reverse_modules[target]
        x = torch.randn(2, one.in_features)
        assert torch.allclose(one(x), two(x), rtol=2e-7, atol=1e-6), target


def test_sd15_unload_removes_both_adapters_and_restores_the_base(tmp_path):
    directory, name_a, unet_paths, te_paths = train_and_save(tmp_path, "a.safetensors")
    _d, name_b, _u, _t = train_and_save(tmp_path, "b.safetensors", seed=4321)

    unet, text_encoder = build_unet(), build_text_encoder()
    pipeline, manager = stack_through_manager(
        tmp_path, [{"path": name_a, "strength": STRENGTH},
                   {"path": name_b, "strength": STRENGTH_B}], unet, text_encoder)
    assert peft_wrapped_paths(unet) == unet_paths

    manager.unload_loras(pipeline)
    assert not peft_wrapped_paths(unet) and not peft_wrapped_paths(text_encoder)
    base_modules = dict(build_unet().named_modules())
    modules = dict(unet.named_modules())
    for target in stackable_targets(directory, name_a, unet_paths):
        base = base_modules[target]
        x = torch.randn(2, base.in_features)
        assert torch.allclose(modules[target](x), base(x), atol=1e-6), target


def test_sd15_step_range_still_drops_one_adapter_mid_generation(tmp_path):
    """``step_range`` is honoured only on this path, and only through the step
    callback -- which the route wires when ANY selected LoRA has a non-default
    range. Activating the whole set at load must not defeat it."""
    _directory, name_a, _unet_paths, _te = train_and_save(tmp_path, "a.safetensors", seed=1234)
    _d, name_b, _u, _t = train_and_save(tmp_path, "b.safetensors", seed=4321)

    unet, text_encoder = build_unet(), build_text_encoder()
    pipeline, manager = stack_through_manager(
        tmp_path, [{"path": name_a, "strength": STRENGTH, "step_range": [0, 500]},
                   {"path": name_b, "strength": STRENGTH_B}], unet, text_encoder)
    callback = manager.create_step_callback(pipeline, total_steps=10)

    callback(pipeline, 0, 0.0, {})
    assert set(pipeline.get_active_adapters()) == {"lora_0", "lora_1"}
    callback(pipeline, 7, 0.0, {})
    assert pipeline.get_active_adapters() == ["lora_1"]


def test_sd15_step_callback_no_longer_discards_unet_layer_weights(tmp_path):
    """``set_adapters`` recomputes ``scaling`` from the adapter weight, so the
    per-block factor has to be folded in after EVERY activation, not once at
    load. Before, a LoRA carrying both lost its block weights at step 0."""
    _directory, name_a, unet_paths, _te = train_and_save(tmp_path, "a.safetensors")
    _d, name_b, _u, _t = train_and_save(tmp_path, "b.safetensors", seed=4321)

    unet, text_encoder = build_unet(), build_text_encoder()
    pipeline, manager = stack_through_manager(
        tmp_path, [{"path": name_a, "strength": STRENGTH,
                    "unet_layer_weights": {"IN01": 0.25}},
                   {"path": name_b, "strength": STRENGTH_B}], unet, text_encoder)

    scaled = {t for t in unet_paths if t.startswith("down_blocks.1.")}
    assert scaled, "setup: the toy UNet must have a down_blocks.1 target"
    modules = dict(unet.named_modules())
    assert {round(modules[t].scaling["lora_0"], 9) for t in scaled} == \
        {round(SCALE * STRENGTH * 0.25, 9)}

    manager.create_step_callback(pipeline, total_steps=10)(pipeline, 0, 0.0, {})
    modules = dict(unet.named_modules())
    assert {round(modules[t].scaling["lora_0"], 9) for t in scaled} == \
        {round(SCALE * STRENGTH * 0.25, 9)}
    assert {round(modules[t].scaling["lora_1"], 9) for t in scaled} == \
        {round(SCALE * STRENGTH_B, 9)}
