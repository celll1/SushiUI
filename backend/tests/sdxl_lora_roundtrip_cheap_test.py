"""SDXL: trainer save -> fresh-generation load round trip, CPU, ~2s.

Drives the REAL ``SDXLLoRAAdapter`` (injection + ``save_checkpoint``) over a
toy-width ``UNet2DConditionModel`` (67k parameters, ``use_linear_projection``
and ``text_time`` conditioning as SDXL has them) and BOTH CLIP text encoders,
then the REAL generation-side conversion and injection.

Same construction as the SD1.5 gate: SD1.5/SDXL have no SushiUI-owned loader.
``LoRAManager.load_loras`` calls ``pipeline.load_lora_weights``, whose body is
``lora_state_dict`` -> ``load_lora_into_unet`` -> ``load_lora_into_text_encoder``
twice (once per encoder, with the ``text_encoder`` / ``text_encoder_2``
prefixes). The round-trip gates call those four directly; a real pipeline needs
tokenizers and a scheduler that are not available offline.

What SDXL adds over SD1.5 and what makes the set equality worth asserting:
``use_linear_projection=True`` turns ``proj_in``/``proj_out`` into Linears, so
the adapter targets 12 modules per Transformer2DModel rather than 10, and the
checkpoint carries two text-encoder namespaces (``lora_te1_*``/``lora_te2_*``)
that must land on different models.

The refusal gates at the bottom drive the real ``load_loras`` over a facade
that subclasses the real diffusers mixin, so its ``load_lora_weights`` is the
production one too.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/sdxl_lora_roundtrip_cheap_test.py -v
"""

import types

import torch
from safetensors.torch import load_file, save_file

from lora_roundtrip_common import lora_delta, module_ids, randomise_lora_layers

from diffusers import UNet2DConditionModel  # noqa: E402
from diffusers.loaders.lora_pipeline import (  # noqa: E402
    StableDiffusionXLLoraLoaderMixin as SDXLLoader,
)
from transformers import (  # noqa: E402
    CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection,
)

from core.training.adapters.sdxl_adapter import SDXLLoRAAdapter  # noqa: E402

RANK = 4
ALPHA = 8  # != rank on purpose: PEFT's scaling must be 2.0, not 1.0
SCALE = ALPHA / RANK
STRENGTH = 0.5

_UNET_KWARGS = dict(
    sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
    block_out_channels=(8, 16), norm_num_groups=4,
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
    cross_attention_dim=16, attention_head_dim=2, transformer_layers_per_block=1,
    use_linear_projection=True, addition_embed_type="text_time",
    addition_time_embed_dim=8, projection_class_embeddings_input_dim=40,
)
_TE1_CONFIG = CLIPTextConfig(vocab_size=64, hidden_size=8, intermediate_size=16,
                             num_hidden_layers=2, num_attention_heads=2,
                             max_position_embeddings=16)
_TE2_CONFIG = CLIPTextConfig(vocab_size=64, hidden_size=8, intermediate_size=16,
                             num_hidden_layers=2, num_attention_heads=2,
                             max_position_embeddings=16, projection_dim=8)


def build_unet():
    torch.manual_seed(0)
    return UNet2DConditionModel(**_UNET_KWARGS)


def build_text_encoders():
    torch.manual_seed(0)
    return CLIPTextModel(_TE1_CONFIG), CLIPTextModelWithProjection(_TE2_CONFIG)


def trainer_wrapped_paths(model):
    from lora_roundtrip_common import LoRALinearLayer
    return {name for name, module in model.named_modules()
            if isinstance(module, LoRALinearLayer)}


def peft_wrapped_paths(model):
    return {name for name, module in model.named_modules()
            if getattr(module, "lora_A", None) is not None and len(module.lora_A)}


def train_and_save(tmp_path, name="sdxl.safetensors", seed=1234):
    unet = build_unet()
    te1, te2 = build_text_encoders()
    trainer = types.SimpleNamespace(unet=unet, text_encoder=te1, text_encoder_2=te2,
                                    unet_lr=1e-4, text_encoder_1_lr=1e-5,
                                    text_encoder_2_lr=1e-5)
    adapter = SDXLLoRAAdapter(trainer, lora_rank=RANK, lora_alpha=ALPHA,
                              lora_dtype=torch.float32)
    layers = {}
    n_unet = adapter.apply_lora_to_unet(layers)
    n_te = adapter.apply_lora_to_text_encoders(layers)
    assert n_unet > 0 and n_te > 0
    randomise_lora_layers(layers, seed=seed, std=0.2)
    adapter.save_checkpoint(layers, step=1, epoch=0, output_path=tmp_path / name)
    return (str(tmp_path), name, trainer_wrapped_paths(unet),
            trainer_wrapped_paths(te1), trainer_wrapped_paths(te2))


def load_into(directory, name, unet, te1, te2, adapter_name="t"):
    """The four calls ``pipeline.load_lora_weights`` makes, verbatim."""
    state_dict, network_alphas, metadata = SDXLLoader.lora_state_dict(
        directory, weight_name=name, unet_config=unet.config, return_lora_metadata=True)
    SDXLLoader.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=unet,
                                   adapter_name=adapter_name, metadata=metadata)
    SDXLLoader.load_lora_into_text_encoder(
        state_dict, network_alphas=network_alphas, text_encoder=te1,
        prefix="text_encoder", adapter_name=adapter_name, metadata=metadata)
    SDXLLoader.load_lora_into_text_encoder(
        state_dict, network_alphas=network_alphas, text_encoder=te2,
        prefix="text_encoder_2", adapter_name=adapter_name, metadata=metadata)


def test_sdxl_generation_wraps_exactly_the_targets_the_trainer_wrapped(tmp_path):
    directory, name, unet_paths, te1_paths, te2_paths = train_and_save(tmp_path)

    unet = build_unet()
    te1, te2 = build_text_encoders()
    load_into(directory, name, unet, te1, te2)

    assert peft_wrapped_paths(unet) == unet_paths
    assert peft_wrapped_paths(te1) == te1_paths
    assert peft_wrapped_paths(te2) == te2_paths
    # The two Linears SD1.5 does not have (its proj_in/proj_out are Conv2d).
    assert any(p.endswith(".proj_in") for p in unet_paths)
    assert any(p.endswith(".proj_out") for p in unet_paths)


def test_sdxl_wrapped_forward_is_base_plus_scaled_branch(tmp_path):
    directory, name, unet_paths, te1_paths, te2_paths = train_and_save(tmp_path)
    saved = load_file(f"{directory}/{name}")

    unet = build_unet()
    te1, te2 = build_text_encoders()
    load_into(directory, name, unet, te1, te2)

    from diffusers.utils.peft_utils import set_weights_and_activate_adapters

    unet.set_adapters(["t"], [STRENGTH])
    # What pipeline.set_adapters() does for the encoders, which have none.
    set_weights_and_activate_adapters(te1, ["t"], [STRENGTH])
    set_weights_and_activate_adapters(te2, ["t"], [STRENGTH])

    checked = 0
    for model, prefix in ((unet, "lora_unet_"), (te1, "lora_te1_"), (te2, "lora_te2_")):
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
    assert checked == len(unet_paths) + len(te1_paths) + len(te2_paths)


def test_sdxl_alpha_beats_the_rank_fallback(tmp_path):
    directory, name, unet_paths, te1_paths, te2_paths = train_and_save(tmp_path)
    unet = build_unet()
    te1, te2 = build_text_encoders()
    load_into(directory, name, unet, te1, te2)

    for model, paths in ((unet, unet_paths), (te1, te1_paths), (te2, te2_paths)):
        modules = dict(model.named_modules())
        assert {round(modules[t].scaling["t"], 9) for t in paths} == {round(SCALE, 9)}

    unet.set_adapters(["t"], [STRENGTH])
    modules = dict(unet.named_modules())
    assert {round(modules[t].scaling["t"], 9) for t in unet_paths} == \
        {round(SCALE * STRENGTH, 9)}

    stripped = {k: v for k, v in load_file(f"{directory}/{name}").items()
                if not k.endswith(".alpha")}
    save_file(stripped, f"{directory}/no_alpha.safetensors")
    unet2 = build_unet()
    te1b, te2b = build_text_encoders()
    load_into(directory, "no_alpha.safetensors", unet2, te1b, te2b)
    modules2 = dict(unet2.named_modules())
    assert {round(modules2[t].scaling["t"], 9) for t in unet_paths} == {1.0}


def test_sdxl_unload_restores_the_identical_objects_and_is_idempotent(tmp_path):
    directory, name, unet_paths, te1_paths, te2_paths = train_and_save(tmp_path)

    unet = build_unet()
    te1, te2 = build_text_encoders()
    before = [(unet, {p: dict(unet.named_modules())[p] for p in unet_paths}),
              (te1, {p: dict(te1.named_modules())[p] for p in te1_paths}),
              (te2, {p: dict(te2.named_modules())[p] for p in te2_paths})]
    load_into(directory, name, unet, te1, te2)
    assert peft_wrapped_paths(unet) == unet_paths

    from diffusers.utils import recurse_remove_peft_layers

    unet.unload_lora()
    recurse_remove_peft_layers(te1)
    recurse_remove_peft_layers(te2)

    for model, originals in before:
        live = dict(model.named_modules())
        for target, original in originals.items():
            # id(), not tensor equality: the unwrap must put the ORIGINAL
            # nn.Linear back, not a copy with the same weights.
            assert live[target] is original, target
        assert not peft_wrapped_paths(model)

    unet.unload_lora()  # second unload: a no-op, not a re-splice
    recurse_remove_peft_layers(te1)
    for target, original in before[0][1].items():
        assert dict(unet.named_modules())[target] is original, target


def test_sdxl_a_loaded_lora_lives_inside_its_own_model(tmp_path):
    """The reload gate, in the only form SDXL admits: PEFT installs the
    adapters INSIDE each component, so there is no manager-side module map that
    could outlive a model swap."""
    directory, name, unet_paths, te1_paths, te2_paths = train_and_save(tmp_path)

    unet_a = build_unet()
    te1_a, te2_a = build_text_encoders()
    load_into(directory, name, unet_a, te1_a, te2_a)
    a_ids = module_ids(unet_a) | module_ids(te1_a) | module_ids(te2_a)

    unet_b = build_unet()
    te1_b, te2_b = build_text_encoders()
    b_ids_before = module_ids(unet_b) | module_ids(te1_b) | module_ids(te2_b)
    assert not (a_ids & b_ids_before), "setup: A and B must not already share modules"

    from diffusers.utils import recurse_remove_peft_layers

    unet_b.unload_lora()
    recurse_remove_peft_layers(te1_b)
    recurse_remove_peft_layers(te2_b)
    assert module_ids(unet_b) | module_ids(te1_b) | module_ids(te2_b) == b_ids_before
    assert not (module_ids(unet_b) & a_ids)
    assert peft_wrapped_paths(unet_a) == unet_paths, "model A lost its adapters"

    b_before = dict(unet_b.named_modules())
    load_into(directory, name, unet_b, te1_b, te2_b, adapter_name="second")
    assert peft_wrapped_paths(unet_b) == unet_paths
    unet_b.unload_lora()
    for target in unet_paths:
        assert dict(unet_b.named_modules())[target] is b_before[target], target
    assert not (module_ids(unet_b) & a_ids)


# ---------------------------------------------------------------------------
# The refusal contract, through the production entry point. Same construction
# as the SD1.5 gate: the facade subclasses the REAL SDXL mixin, so
# load_lora_weights / set_adapters are the ones a StableDiffusionXLPipeline
# would run, and LoRAManager.load_loras is the production one.
#
# SDXL's third component is what makes these worth repeating rather than
# sharing: text_encoder_2 is a fourth injection call, and the zero-target count
# has to be summed across all three.
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from lora_roundtrip_common import warning_codes, warning_probe  # noqa: E402

from core.extensions.lora_manager import LoRAManager  # noqa: E402


class Pipeline(SDXLLoader):
    _lora_loadable_modules = ["unet", "text_encoder", "text_encoder_2"]
    hf_device_map = None

    def __init__(self, unet, te1, te2):
        self.unet = unet
        self.text_encoder = te1
        self.text_encoder_2 = te2

    @property
    def components(self):
        return {"unet": self.unet, "text_encoder": self.text_encoder,
                "text_encoder_2": self.text_encoder_2}


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)


def manager_for(directory):
    manager = LoRAManager(lora_dir=str(directory))
    # The repo's training/ dir is seeded at construction; drop it so resolution
    # depends only on the fixture directory.
    manager.seeded_dirs = []
    return manager


def load_through_manager(tmp_path, name, unet, te1, te2, strength=1.0, **extra):
    pipeline = Pipeline(unet, te1, te2)
    manager_for(tmp_path).load_loras(
        pipeline, [{"path": name, "strength": strength, **extra}])
    return pipeline


def test_sdxl_manager_applies_a_working_lora_and_warns_about_nothing(tmp_path,
                                                                    warnings_seen):
    """The false-refusal gate: the path most users exercise must stay silent."""
    _directory, name, unet_paths, te1_paths, te2_paths = train_and_save(tmp_path)

    unet = build_unet()
    te1, te2 = build_text_encoders()
    load_through_manager(tmp_path, name, unet, te1, te2, strength=STRENGTH)

    assert peft_wrapped_paths(unet) == unet_paths
    assert peft_wrapped_paths(te1) == te1_paths
    assert peft_wrapped_paths(te2) == te2_paths
    assert warning_codes(warnings_seen) == []
    modules = dict(unet.named_modules())
    assert {round(modules[t].scaling["lora_0"], 9) for t in unet_paths} == \
        {round(SCALE * STRENGTH, 9)}


def test_sdxl_missing_file_refuses_and_warns(tmp_path, warnings_seen):
    unet = build_unet()
    te1, te2 = build_text_encoders()
    with pytest.raises(FileNotFoundError):
        load_through_manager(tmp_path, "no_such_sdxl_lora.safetensors", unet, te1, te2)
    assert "lora_not_found" in warning_codes(warnings_seen)


def test_sdxl_unreadable_file_refuses_and_names_no_path(tmp_path, warnings_seen):
    (tmp_path / "corrupt.safetensors").write_bytes(b"not a safetensors header")

    unet = build_unet()
    te1, te2 = build_text_encoders()
    with pytest.raises(RuntimeError) as excinfo:
        load_through_manager(tmp_path, "corrupt.safetensors", unet, te1, te2)
    assert "lora_load_failed" in warning_codes(warnings_seen)
    message = warnings_seen[-1][1]
    assert "corrupt.safetensors" in message
    # The warning reaches a PNG text chunk, the API response and the DB.
    assert str(tmp_path) not in message and str(tmp_path) not in str(excinfo.value)


def test_sdxl_lora_naming_absent_unet_modules_refuses(tmp_path, warnings_seen):
    """Kohya stems this UNet does not have. PEFT raises inside the apply, so
    this lands on lora_load_failed rather than lora_incompatible."""
    directory, name, _unet, _te1, _te2 = train_and_save(tmp_path)
    ghost = {}
    for key, value in load_file(f"{directory}/{name}").items():
        if key.startswith("lora_unet_"):
            ghost["lora_unet_ghost_" + key[len("lora_unet_"):]] = value
    save_file(ghost, f"{directory}/ghost.safetensors")

    unet = build_unet()
    te1, te2 = build_text_encoders()
    with pytest.raises(RuntimeError):
        load_through_manager(tmp_path, "ghost.safetensors", unet, te1, te2)
    assert "lora_load_failed" in warning_codes(warnings_seen)
    assert not peft_wrapped_paths(unet)


def test_sdxl_lora_for_another_architecture_refuses_and_warns(tmp_path,
                                                              warnings_seen):
    """The silent case: a Z-Image-style checkpoint matches none of the three
    component prefixes, so diffusers filters it to nothing and returns without
    raising. Only the read-back target count sees it."""
    foreign = {}
    for i in range(3):
        stem = f"lora_transformer_layers_{i}_attention_to_q"
        foreign[f"{stem}.lora_down.weight"] = torch.zeros(RANK, 8)
        foreign[f"{stem}.lora_up.weight"] = torch.zeros(8, RANK)
        foreign[f"{stem}.alpha"] = torch.tensor(float(ALPHA))
    save_file(foreign, str(tmp_path / "zimage_style.safetensors"))

    unet = build_unet()
    te1, te2 = build_text_encoders()
    with pytest.raises(RuntimeError, match="0 of 3 down/up"):
        load_through_manager(tmp_path, "zimage_style.safetensors", unet, te1, te2)
    assert "lora_incompatible" in warning_codes(warnings_seen)
    assert not peft_wrapped_paths(unet)
    assert not peft_wrapped_paths(te1) and not peft_wrapped_paths(te2)


def test_sdxl_partly_matching_lora_applies_and_warns(tmp_path, warnings_seen):
    """Survivable: all three real halves apply, the two invented stems do not."""
    directory, name, unet_paths, te1_paths, te2_paths = train_and_save(tmp_path)
    partial = dict(load_file(f"{directory}/{name}"))
    for i in range(2):
        stem = f"lora_unet_ghost_block_{i}"
        partial[f"{stem}.lora_down.weight"] = torch.zeros(RANK, 8)
        partial[f"{stem}.lora_up.weight"] = torch.zeros(8, RANK)
        partial[f"{stem}.alpha"] = torch.tensor(float(ALPHA))
    save_file(partial, f"{directory}/partial.safetensors")

    unet = build_unet()
    te1, te2 = build_text_encoders()
    load_through_manager(tmp_path, "partial.safetensors", unet, te1, te2)

    assert peft_wrapped_paths(unet) == unet_paths
    assert peft_wrapped_paths(te1) == te1_paths
    assert peft_wrapped_paths(te2) == te2_paths
    assert warning_codes(warnings_seen) == ["lora_partial"]
    applied = len(unet_paths) + len(te1_paths) + len(te2_paths)
    assert f"applied {applied} of {applied + 2}" in warnings_seen[-1][1]


# ---------------------------------------------------------------------------
# Stacking. SDXL shares LoRAManager.load_loras with SD1.5, so the defect and
# the fix are the same; what is SDXL's own is that the active set has to cover
# THREE components. A per-file activation left every LoRA but the last inert.
# ---------------------------------------------------------------------------

STRENGTH_B = 0.4  # the second LoRA's, so a shared weight shows up as a wrong sum


def stack_through_manager(tmp_path, configs, unet, te1, te2):
    pipeline = Pipeline(unet, te1, te2)
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


def test_sdxl_two_loras_stay_active_together_and_their_deltas_sum(tmp_path,
                                                                  warnings_seen):
    directory, name_a, unet_paths, te1_paths, te2_paths = train_and_save(
        tmp_path, "a.safetensors", seed=1234)
    _d, name_b, _u, _t1, _t2 = train_and_save(tmp_path, "b.safetensors", seed=4321)
    a_tensors = stackable_targets(directory, name_a, unet_paths)
    b_tensors = stackable_targets(directory, name_b, unet_paths)

    unet = build_unet()
    te1, te2 = build_text_encoders()
    pipeline, _m = stack_through_manager(
        tmp_path, [{"path": name_a, "strength": STRENGTH},
                   {"path": name_b, "strength": STRENGTH_B}], unet, te1, te2)

    assert set(pipeline.get_active_adapters()) == {"lora_0", "lora_1"}
    assert warning_codes(warnings_seen) == []
    # All three components carry both adapters, not just the UNet.
    for model, paths in ((unet, unet_paths), (te1, te1_paths), (te2, te2_paths)):
        modules = dict(model.named_modules())
        assert paths, "setup: every component must have targets"
        for target in paths:
            assert set(modules[target].lora_A.keys()) == {"lora_0", "lora_1"}, target

    base_modules = dict(build_unet().named_modules())
    modules = dict(unet.named_modules())
    for target, (down_a, up_a) in a_tensors.items():
        down_b, up_b = b_tensors[target]
        module = modules[target]
        assert round(module.scaling["lora_0"], 9) == round(SCALE * STRENGTH, 9)
        assert round(module.scaling["lora_1"], 9) == round(SCALE * STRENGTH_B, 9)

        base = base_modules[target]
        x = torch.randn(2, base.in_features)
        expected = (base(x)
                    + lora_delta(down_a, up_a, x, ALPHA, RANK, STRENGTH)
                    + lora_delta(down_b, up_b, x, ALPHA, RANK, STRENGTH_B))
        assert torch.allclose(module(x), expected, atol=1e-5), target
        assert not torch.allclose(
            module(x), base(x) + lora_delta(down_b, up_b, x, ALPHA, RANK, STRENGTH_B),
            atol=1e-5), f"{target}: only the last LoRA is in effect"


def test_sdxl_unload_removes_both_adapters_and_restores_the_base(tmp_path):
    directory, name_a, unet_paths, te1_paths, te2_paths = train_and_save(
        tmp_path, "a.safetensors")
    _d, name_b, _u, _t1, _t2 = train_and_save(tmp_path, "b.safetensors", seed=4321)

    unet = build_unet()
    te1, te2 = build_text_encoders()
    pipeline, manager = stack_through_manager(
        tmp_path, [{"path": name_a, "strength": STRENGTH},
                   {"path": name_b, "strength": STRENGTH_B}], unet, te1, te2)
    assert peft_wrapped_paths(unet) == unet_paths

    manager.unload_loras(pipeline)
    for model in (unet, te1, te2):
        assert not peft_wrapped_paths(model)
    base_modules = dict(build_unet().named_modules())
    modules = dict(unet.named_modules())
    for target in stackable_targets(directory, name_a, unet_paths):
        base = base_modules[target]
        x = torch.randn(2, base.in_features)
        assert torch.allclose(modules[target](x), base(x), atol=1e-6), target


def test_sdxl_unet_layer_weights_survive_a_stack_and_the_step_callback(tmp_path):
    """``unet_layer_weights`` is honoured on this path only, and ``set_adapters``
    recomputes ``scaling``, so the per-block factor has to be re-folded after
    every activation -- including the one that makes the second LoRA active."""
    _directory, name_a, unet_paths, _t1, _t2 = train_and_save(tmp_path, "a.safetensors")
    _d, name_b, _u, _tt1, _tt2 = train_and_save(tmp_path, "b.safetensors", seed=4321)

    unet = build_unet()
    te1, te2 = build_text_encoders()
    pipeline, manager = stack_through_manager(
        tmp_path, [{"path": name_a, "strength": STRENGTH,
                    "unet_layer_weights": {"IN01": 0.25}},
                   {"path": name_b, "strength": STRENGTH_B}], unet, te1, te2)

    scaled = {t for t in unet_paths if t.startswith("down_blocks.1.")}
    assert scaled, "setup: the toy UNet must have a down_blocks.1 target"
    for _ in range(2):
        modules = dict(unet.named_modules())
        assert {round(modules[t].scaling["lora_0"], 9) for t in scaled} == \
            {round(SCALE * STRENGTH * 0.25, 9)}
        assert {round(modules[t].scaling["lora_1"], 9) for t in scaled} == \
            {round(SCALE * STRENGTH_B, 9)}
        manager.create_step_callback(pipeline, total_steps=10)(pipeline, 0, 0.0, {})
