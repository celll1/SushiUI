"""REPA must not feed a teacher outside the preprocessing it was trained with.

REPA squishes the latent's region to ONE square and calls the encoder with
``pixel_values`` alone. A NaFlex-style SigLIP2 (``max_num_patches``, no fixed
``size``, no ``image_size`` in its vision config) was trained on
aspect-preserving patch grids, and ``Siglip2VisionModel.forward`` takes flattened
patches plus ``spatial_shapes``/``pixel_attention_mask`` -- so that call cannot
succeed, and ``repa_size`` used to fall back to 384 without a word.

What is pinned here:
  (a) a NaFlex teacher is refused, at setup, before the encoder is read;
  (b) the verdict comes from the processor / vision configs, not the repo NAME;
  (c) a fixed-resolution teacher is unaffected;
  (d) unreadable configs refuse nothing (offline is not evidence), and the size
      fallback then says so;
  (e) repa_enable=false reads none of it.

Run:
    venv/Scripts/python.exe -m pytest backend/tests/repa_teacher_domain_test.py -v

Static: no model weights, no GPU, no network.
"""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace  # noqa: E402

from core.training import repa as repa_module  # noqa: E402
from core.training.base_trainer import BaseTrainer  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NAFLEX_DIR = os.path.join(REPO_ROOT, "tagger_models", "048122a9-e381-4c16-b221-1ddb8c96f92a")
FIXED_DIR = os.path.join(REPO_ROOT, "tagger_models", "cca72ce1-7420-4164-9f24-c30ae77cdf2f")

NAFLEX_PROC = {"image_processor_type": "Siglip2ImageProcessorFast",
               "max_num_patches": 256, "patch_size": 16, "resample": 2,
               "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}
NAFLEX_CFG = {"vision_config": {"model_type": "siglip2_vision_model",
                                "hidden_size": 1152, "num_hidden_layers": 27}}
FIXED_PROC = {"image_processor_type": "SiglipImageProcessor",
              "size": {"height": 384, "width": 384}, "resample": 2}
FIXED_CFG = {"vision_config": {"model_type": "siglip_vision_model",
                               "hidden_size": 1152, "image_size": 384,
                               "patch_size": 14}}


def _repo_dir(tmp_path, name, cfg, proc):
    """A teacher repo as a local directory: the configs, and no weights."""
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (d / "preprocessor_config.json").write_text(json.dumps(proc), encoding="utf-8")
    return str(d)


def _tagger_dir(tmp_path, name, base_repo, *, ckpt_repo=None):
    """A tagger checkpoint dir naming ``base_repo``; the .safetensors stays empty."""
    d = tmp_path / name
    d.mkdir()
    if base_repo is not None:
        (d / "base_model_metadata.json").write_text(
            json.dumps({"vision_encoder_repo": base_repo}), encoding="utf-8")
    (d / "best_f1.safetensors").write_bytes(b"")
    if ckpt_repo is not None:
        (d / "best_f1_metadata.json").write_text(
            json.dumps({"vision_encoder_repo": ckpt_repo}), encoding="utf-8")
    return str(d)


def test_exact_safetensors_path_bypasses_directory_preference(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "base_model_metadata.json").write_text(
        json.dumps({"vision_encoder_repo": "base/repo"}), encoding="utf-8")
    (d / "best_f1.safetensors").write_bytes(b"best")
    latest = d / "latest.safetensors"
    latest.write_bytes(b"latest")

    checkpoint, repo = repa_module._resolve_tagger_checkpoint(str(latest))

    assert checkpoint == str(latest.resolve())
    assert repo == "base/repo"


def test_exact_onnx_path_uses_its_sidecar_repo(tmp_path):
    d = tmp_path / "v2"
    d.mkdir()
    model = d / "model.onnx"
    model.write_bytes(b"graph")
    (d / "model_metadata.json").write_text(
        json.dumps({"vision_encoder_repo": "export/repo"}), encoding="utf-8")

    checkpoint, repo = repa_module._resolve_tagger_checkpoint(str(model))

    assert checkpoint == str(model.resolve())
    assert repo == "export/repo"


def test_deployment_directory_falls_back_to_model_onnx(tmp_path):
    d = tmp_path / "v2"
    d.mkdir()
    model = d / "model.onnx"
    model.write_bytes(b"graph")

    checkpoint, _repo = repa_module._resolve_tagger_checkpoint(str(d))

    assert checkpoint == str(model)


def test_onnx_repa_subgraphs_expose_pixels_and_embedding_inputs(tmp_path):
    onnx = pytest.importorskip("onnx")
    pixel_name = "pixel_values"
    embed_name = repa_module._ONNX_EMBED_INPUT_SUFFIX
    output_name = repa_module._ONNX_PATCH_OUTPUT_SUFFIX
    pixel = onnx.helper.make_tensor_value_info(
        pixel_name, onnx.TensorProto.FLOAT, ["batch", 3, 384, 384])
    embed = onnx.helper.make_tensor_value_info(
        embed_name, onnx.TensorProto.FLOAT, ["batch", 729, 1152])
    output = onnx.helper.make_tensor_value_info(
        output_name, onnx.TensorProto.FLOAT, ["batch", 729, 1152])
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Flatten", [pixel_name], [embed_name]),
            onnx.helper.make_node("Identity", [embed_name], [output_name]),
        ],
        "repa-test", [pixel], [output], value_info=[embed])
    source = tmp_path / "model.onnx"
    onnx.save(onnx.helper.make_model(graph), source)

    pixels = repa_module._write_onnx_subgraph(str(source), "pixels")
    trunk = repa_module._write_onnx_subgraph(str(source), "trunk")

    assert pixels[1:] == (pixel_name, output_name, 384, 1152)
    assert trunk[1:] == (embed_name, output_name, 384, 1152)
    pixel_model = onnx.load(pixels[0], load_external_data=False)
    trunk_model = onnx.load(trunk[0], load_external_data=False)
    assert [node.op_type for node in pixel_model.graph.node] == ["Flatten", "Identity"]
    assert [node.op_type for node in trunk_model.graph.node] == ["Identity"]


def _require_real(model_dir):
    if not os.path.isdir(model_dir):
        pytest.skip(f"{model_dir} is not present in this clone")
    repos = repa_module._teacher_candidate_repos("tagger", model_dir, "")
    if all(repa_module.teacher_fixed_input_size(r) == (None, None) for r in repos):
        pytest.skip(f"{repos} configs are not in the local HuggingFace cache")



def test_real_naflex_tagger_checkpoint_is_refused():
    _require_real(NAFLEX_DIR)
    with pytest.raises(ValueError, match="no fixed square input"):
        repa_module.assert_repa_teacher_fixed_resolution("tagger", tagger_model_dir=NAFLEX_DIR)


def test_real_fixed_resolution_tagger_checkpoint_is_accepted():
    _require_real(FIXED_DIR)
    assert repa_module.assert_repa_teacher_fixed_resolution(
        "tagger", tagger_model_dir=FIXED_DIR) == 384


# ---------------------------------------------------------------------------
# (b) the verdict is on the configs, not on the name
# ---------------------------------------------------------------------------

def test_a_variable_resolution_repo_under_any_name_is_refused(tmp_path):
    repo = _repo_dir(tmp_path, "some-future-encoder-v9", NAFLEX_CFG, NAFLEX_PROC)
    with pytest.raises(ValueError, match="max_num_patches=256"):
        repa_module.assert_repa_teacher_fixed_resolution("siglip2", siglip2_repo=repo)


def test_a_fixed_resolution_repo_under_a_naflex_looking_name_is_accepted(tmp_path):
    repo = _repo_dir(tmp_path, "siglip2-so400m-patch16-naflex", FIXED_CFG, FIXED_PROC)
    assert repa_module.assert_repa_teacher_fixed_resolution("siglip2", siglip2_repo=repo) == 384


def test_a_processor_size_alone_declares_the_domain(tmp_path):
    repo = _repo_dir(tmp_path, "no-image-size-in-config", NAFLEX_CFG, FIXED_PROC)
    assert repa_module.assert_repa_teacher_fixed_resolution("siglip2", siglip2_repo=repo) == 384


def test_a_checkpoint_sidecar_naming_a_variable_repo_is_refused(tmp_path):
    """A dir with no base_model_metadata.json resolves to the fixed-res default;
    its own checkpoint metadata is then the only record of what it was trained as."""
    fixed = _repo_dir(tmp_path, "fixed-base", FIXED_CFG, FIXED_PROC)
    naflex = _repo_dir(tmp_path, "naflex-base", NAFLEX_CFG, NAFLEX_PROC)
    d = _tagger_dir(tmp_path, "run", fixed, ckpt_repo=naflex)
    with pytest.raises(ValueError, match="no fixed square input"):
        repa_module.assert_repa_teacher_fixed_resolution("tagger", tagger_model_dir=d)



def test_unreadable_configs_refuse_nothing(tmp_path):
    d = _tagger_dir(tmp_path, "run", str(tmp_path / "not-a-repo"))
    assert repa_module.assert_repa_teacher_fixed_resolution("tagger", tagger_model_dir=d) is None


def test_an_unresolvable_tagger_dir_is_left_to_the_loader(tmp_path):
    assert repa_module.assert_repa_teacher_fixed_resolution(
        "tagger", tagger_model_dir=str(tmp_path / "missing")) is None



def _trainer(tagger_dir, *, enable=True):
    from core.training.arch import ARCH_REGISTRY
    return SimpleNamespace(
        config={"repa_enable": enable, "repa_tagger_model_dir": tagger_dir},
        arch=ARCH_REGISTRY["minit2i"](),
        transformer=SimpleNamespace(
            mmjit_config=SimpleNamespace(hidden_size=16, depth_double=28, patch_size=16),
            model=SimpleNamespace(net=SimpleNamespace(_repa_tap_depth=None,
                                                      _repa_tap_out=None))),
        device=torch.device("cpu"), training_dtype=torch.float32,
        model_path="", log_prefix="[test]",
        tread_config=None, block_skip_config=None, blockskip_config=None,
    )


def _stub_loader(monkeypatch, native, calls):
    monkeypatch.setattr(repa_module, "load_repa_encoder",
                        lambda *a, **k: calls.append(a) or (None, 8, native))


def test_setup_refuses_before_the_encoder_is_read(tmp_path, monkeypatch):
    calls = []
    _stub_loader(monkeypatch, 384, calls)
    repo = _repo_dir(tmp_path, "naflex-base", NAFLEX_CFG, NAFLEX_PROC)
    t = _trainer(_tagger_dir(tmp_path, "run", repo))

    with pytest.raises(ValueError, match="no fixed square input"):
        BaseTrainer._setup_repa(t)

    assert calls == [], "the encoder must not be read before the refusal"


def test_setup_is_unchanged_for_a_fixed_resolution_teacher(tmp_path, monkeypatch):
    calls = []
    _stub_loader(monkeypatch, 384, calls)
    repo = _repo_dir(tmp_path, "fixed-base", FIXED_CFG, FIXED_PROC)
    t = _trainer(_tagger_dir(tmp_path, "run", repo))

    BaseTrainer._setup_repa(t)

    assert calls and t.repa_size == 384 and t.repa_enable is True


def test_an_undeclared_size_falls_back_out_loud(tmp_path, monkeypatch, capsys):
    calls = []
    _stub_loader(monkeypatch, None, calls)  # a config with no image_size
    t = _trainer(_tagger_dir(tmp_path, "run", str(tmp_path / "not-a-repo")))

    BaseTrainer._setup_repa(t)

    assert t.repa_size == 384
    assert "declares an input size" in capsys.readouterr().out


def test_disabled_repa_reads_no_teacher_config(tmp_path, monkeypatch):
    reads = []
    monkeypatch.setattr(repa_module, "_read_teacher_config",
                        lambda *a, **k: reads.append(a) or None)
    t = SimpleNamespace(config={"repa_enable": False,
                                "repa_tagger_model_dir": NAFLEX_DIR})

    BaseTrainer._setup_repa(t)

    assert t.repa_enable is False and reads == []
