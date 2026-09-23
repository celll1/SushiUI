"""Style training loop on a tiny SigLIP2, CPU only (design §9, §10, §11; T8, T10 trainer part)."""

from pathlib import Path
import math
import os
import re
import sys

import subprocess

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import style_training_fixture  # noqa: F401  (CUDA stubs before the trainer stack is imported)

from safetensors.torch import load_file

import numpy as np

import core.style_encoder.style_artifacts as artifacts
import core.style_encoder.style_loader as loader_module
from core.style_encoder.style_artifacts import (
    ENCODER_SUFFIX, architecture_metadata, distributable_state_dict, find_resume_checkpoint, load_style_encoder,
    load_training_weights, save_bundle,
)
from core.style_encoder.style_dataset import (
    VAL_ARTIST, VAL_IMAGE, SplitParams, artist_split_unit, build_style_index, stable_item_key,
)
from core.style_encoder.style_model import LoRASpec, StyleEncoder
from core.style_encoder.style_trainer import (
    StyleConfigError, StyleTrainer, StyleTrainerHandle, StyleTrainingConfig, WarmupCosine, eval_rows,
    run_style_training,
)
from core.training.base_trainer import NothingTrainedError
from style_training_fixture import (
    PATCH, add_item, build_corpus, config, environment, label_resolver, tiny_vision,
)


class _Recorder:
    def __init__(self, stop_at=None):
        self.events = []
        self.stop_at = stop_at
        self.holder = []

    def __call__(self, run_id, event, data):
        self.events.append((event, data))
        if event == "step" and data["step"] == self.stop_at:
            self.holder[0].stop()

    def of(self, event):
        return [d for e, d in self.events if e == event]

    def losses(self):
        return {d["step"]: d["loss"] for d in self.of("step")}


def _run(factory, out, cfg, stop_at=None):
    recorder = _Recorder(stop_at)
    result = run_style_training("run", cfg, str(out), recorder, recorder.holder, environment(factory))
    return result, recorder


def _probe_batch():
    torch.manual_seed(3)
    pixels = torch.randn(2, 16, 3 * PATCH * PATCH)
    mask = torch.zeros(2, 16, dtype=torch.int32)
    mask[0, :12], mask[1, :16] = 1, 1
    return pixels, mask, torch.tensor([[3, 4], [4, 4]])


def test_missing_config_keys_are_refused():
    cfg = config([1])
    del cfg["style_temperature"], cfg["epochs"]
    with pytest.raises(StyleConfigError, match="epochs.*style_temperature|style_temperature.*epochs"):
        StyleTrainingConfig.from_dict(cfg)
    with pytest.raises(StyleConfigError, match="optimizer"):
        StyleTrainingConfig.from_dict(config([1], optimizer="sgd"))


def test_short_run_writes_reloadable_bundles_without_artist_names(tmp_path):
    factory, ids, names = build_corpus(tmp_path)
    out = tmp_path / "out"
    result, rec = _run(factory, out, config(ids, save_every_n_steps=1))

    losses = rec.losses()
    assert result["global_step"] == result["total_steps"] == len(losses) >= 3
    assert rec.of("step")[0]["lr"] == pytest.approx(1e-3)  # warmup_steps=1: the first step applies the full rate
    assert all(math.isfinite(v) for v in losses.values())
    steps = sorted(int(m.group(1)) for f in os.listdir(out) if (m := re.match(r"step_(\d+)_state\.json$", f)))
    assert steps == [result["total_steps"] - 1, result["total_steps"]]
    for step in steps:
        for suffix in (ENCODER_SUFFIX, "_metadata.json", "_optimizer.pt", "_split_snapshot.npz"):
            assert (out / f"step_{step:06d}{suffix}").is_file()
    assert not any(f.startswith(f"step_{result['total_steps'] - 2:06d}_") for f in os.listdir(out))
    assert (out / "best_style_encoder.safetensors").is_file() and not (out / "best_state.json").exists()

    for filename in os.listdir(out):
        blob = (out / filename).read_bytes()
        for name in names:
            assert name.encode() not in blob and name.replace("_", " ").encode() not in blob, filename

    saved = load_file(str(out / "latest_style_encoder.safetensors"))
    assert not any(k.startswith(("vision_encoder.encoder.layers.2.", "vision_encoder.post_layernorm.",
                                 "vision_encoder.head.")) for k in saved)
    reloaded = load_style_encoder(str(out / "latest_style_encoder.safetensors"), "sdpa")
    live = rec.holder[0].model.eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded.embed(*_probe_batch()), live.embed(*_probe_batch()))

    (validation,) = rec.of("validation")
    assert validation["index_hash"] == rec.of("dataset_stats")[0]["index_hash"]
    for key in ("held_out_artists", "held_out_artists_same_copyright", "held_out_artists_masked",
                "train_artist_held_out_images"):
        m = validation[key]
        assert m["used"] + m["excluded_no_positive"] + m["excluded_no_negative"] == m["queries"]
    signed = validation["held_out_artists_with_signature"]["queries"]
    assert signed + validation["held_out_artists_without_signature"]["queries"] == \
        validation["held_out_artists"]["queries"] > 0


def _weights(path):
    return load_file(str(path / "latest_style_encoder.safetensors"))


@pytest.mark.parametrize("method,dropout,stop_at", [("full", 0.0, 3), ("lora", 0.0, 3), ("lora", 0.3, 3),
                                                     ("full", 0.0, 4)])
def test_exact_resume_reproduces_the_uninterrupted_run(tmp_path, method, dropout, stop_at):
    factory, ids, _ = build_corpus(tmp_path)
    cfg = config(ids, training_method=method, lora_dropout=dropout, epochs=2)
    reference, ref_rec = _run(factory, tmp_path / "ref", cfg)
    total = reference["total_steps"]
    assert ref_rec.of("dataset_stats")[0]["epoch_length"] == 4

    stopped, first = _run(factory, tmp_path / "resumed", cfg, stop_at=stop_at)
    assert stopped["stopped"] and stopped["global_step"] == stop_at
    torch.manual_seed(12345)  # dropout must not depend on the process-wide RNG
    resumed, second = _run(factory, tmp_path / "resumed", cfg)

    (resume,) = second.of("resume")
    assert resume["mode"] == "exact" and resume["resumed_from_step"] == stop_at
    assert resumed["global_step"] == resumed["total_steps"] == total
    expected = ref_rec.losses()
    assert {s: expected[s] for s in range(1, stop_at + 1)} == first.losses()
    assert {s: expected[s] for s in range(stop_at + 1, total + 1)} == second.losses()
    a, b = _weights(tmp_path / "ref"), _weights(tmp_path / "resumed")
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    if stop_at == 4:  # stopped after the epoch's last batch: epoch 0's evaluation runs before training resumes
        assert (resume["epoch"], resume["step_in_epoch"]) == (1, 0)
        assert [e for e, _ in second.events if e in ("validation", "step")][0] == "validation"
        assert second.of("validation")[0]["epoch"] == 0
    if dropout:
        _, plain = _run(factory, tmp_path / "plain", config(ids, training_method=method, epochs=2))
        assert plain.losses()[2] != expected[2]


def test_data_change_resumes_at_the_epoch_boundary_keeping_the_step_budget(tmp_path):
    factory, ids, _ = build_corpus(tmp_path)
    cfg = config(ids, epochs=2)
    out = tmp_path / "out"
    stopped, first = _run(factory, out, cfg, stop_at=5)
    epoch_length = first.of("dataset_stats")[0]["epoch_length"]
    total = 2 * epoch_length
    old_hash = first.of("dataset_stats")[0]["index_hash"]

    newcomer = next(f"artist_new{k}" for k in range(100) if artist_split_unit(f"artist new{k}", 0) >= 0.3)
    for k in range(4):
        add_item(factory, ids[0], tmp_path / "ds_a" / "new" / f"{k}.png", [("tags", None, newcomer, True)],
                 color=(1, 2, 3 + k))
    resumed, second = _run(factory, out, config(ids, epochs=2, learning_rate=5e-4))

    assert second.of("dataset_stats")[0]["epoch_length"] != epoch_length
    (resume,) = second.of("resume")
    assert (resume["mode"], resume["epoch"], resume["step_in_epoch"]) == ("epoch_boundary", 2, 0)
    first_step = second.of("step")[0]
    assert (first_step["step"], first_step["epoch"], first_step["step_in_epoch"]) == (6, 2, 1)
    assert resumed["total_steps"] == total and resumed["global_step"] == total
    warnings = [w["message"] for w in second.of("warning")]
    assert any("learning_rate" in m and "5e-4" in m.replace("0.0005", "5e-4") for m in warnings)
    assert any("Evaluation set changed" in m for m in warnings)
    new_hash = second.of("dataset_stats")[0]["index_hash"]
    assert new_hash != old_hash and resumed["best"]["index_hash"] == new_hash

    with pytest.raises(ValueError, match="trained with"):
        _run(factory, out, config(ids, epochs=2, style_embed_dim=4))


def test_run_without_an_optimizer_step_is_refused_and_writes_nothing(tmp_path):
    factory, ids, _ = build_corpus(tmp_path, corrupt=True)
    out = tmp_path / "out"
    with pytest.raises(NothingTrainedError):
        _run(factory, out, config(ids, save_every_n_steps=1))
    assert os.listdir(out) == []


def _lora_encoder(dropout=0.0):
    model = StyleEncoder(tiny_vision(), (1, -2), 8, 2, 12, 6, "lora", LoRASpec(rank=2, alpha=2.0, dropout=dropout))
    for module in model.lora_modules.values():
        torch.nn.init.normal_(module.lora_B)
    return model


def test_lora_step_bundle_has_only_resume_weights_and_recovers_interrupted_publish(tmp_path, monkeypatch):
    from core.style_encoder.style_artifacts import LORA_SUFFIX, recover_staged_bundles
    from core.style_encoder.style_resume import SplitSnapshot

    model = _lora_encoder()
    state = {"global_step": 3, "model": {"training_method": "lora"}}
    resume = (state, {"marker": 3}, SplitSnapshot(np.arange(3, dtype=np.int64)))
    real_replace = os.replace

    def interrupted_replace(src, dst):
        if str(dst).endswith("step_000003_state.json"):
            raise OSError("interrupted before marker")
        return real_replace(src, dst)

    with monkeypatch.context() as m:
        m.setattr(os, "replace", interrupted_replace)
        with pytest.raises(OSError, match="interrupted before marker"):
            save_bundle(model, str(tmp_path), "step_000003", {"training_method": "lora"}, resume)
    assert recover_staged_bundles(str(tmp_path))[0] == ["step_000003"]
    assert not (tmp_path / f"step_000003{ENCODER_SUFFIX}").exists()
    assert (tmp_path / f"step_000003{LORA_SUFFIX}").is_file()
    name, found = find_resume_checkpoint(str(tmp_path))
    assert name == "step_000003" and found["has_distributable"] is False
    assert not any(".staged" in f for f in os.listdir(tmp_path))


def test_old_lora_step_bundle_with_full_encoder_still_resumes(tmp_path):
    from core.style_encoder.style_artifacts import LORA_SUFFIX
    from core.style_encoder.style_resume import SplitSnapshot
    import json

    model = _lora_encoder()
    state = {"global_step": 2, "model": {"training_method": "lora"}}
    save_bundle(model, str(tmp_path), "latest", {"training_method": "lora"},
                (state, {"marker": 2}, SplitSnapshot(np.arange(2, dtype=np.int64))))
    for suffix in (ENCODER_SUFFIX, LORA_SUFFIX, "_metadata.json", "_optimizer.pt", "_split_snapshot.npz",
                   "_state.json"):
        os.replace(tmp_path / f"latest{suffix}", tmp_path / f"step_000002{suffix}")
    state_path = tmp_path / "step_000002_state.json"
    old_state = json.loads(state_path.read_text(encoding="utf-8"))
    old_state.pop("has_distributable")
    state_path.write_text(json.dumps(old_state), encoding="utf-8")
    assert find_resume_checkpoint(str(tmp_path))[0] == "step_000002"


def test_hsic_training_uses_frozen_tagger_and_records_path_free_metadata(tmp_path, monkeypatch):
    from core.style_encoder.style_hsic import HSICTagger
    import core.style_encoder.style_trainer as trainer_module
    from style_training_fixture import image_processor
    import json

    class TinyTagger(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, pixel_values, pixel_attention_mask=None, spatial_shapes=None):
            values = pixel_values.float().mean(dim=(1, 2)) * self.scale
            return torch.stack([values, -values, values * 0.5], dim=1)

    tagger = TinyTagger()
    fake = HSICTagger(tagger, image_processor(), True,
                      {"Character": torch.tensor([0]), "Copyright": torch.tensor([1]),
                       "Rating": torch.tensor([2])}, "tagger.safetensors", "f" * 64,
                      max_num_patches=16)
    monkeypatch.setattr(trainer_module, "load_hsic_tagger", lambda path, weights, device: fake)
    factory, ids, _ = build_corpus(tmp_path)
    out = tmp_path / "out"
    result, events = _run(factory, out, config(ids, style_hsic_tagger_path=str(tmp_path / "private" / "tagger.safetensors")))
    assert result["global_step"] > 0
    assert events.of("step") and all("hsic_loss" in event for event in events.of("step"))
    assert all(p.grad is None for p in tagger.parameters())
    metadata = json.loads((out / "latest_metadata.json").read_text(encoding="utf-8"))
    assert metadata["hsic_tagger"]["checkpoint"] == "tagger.safetensors"
    assert metadata["hsic_tagger"]["sha256"] == "f" * 64
    assert str(tmp_path) not in json.dumps(metadata)


def test_lora_distributable_is_merged_and_adapter_weights_reload(tmp_path):
    model = _lora_encoder().eval()
    save_bundle(model, str(tmp_path), "x", {"format": "sushiui-style-encoder", "format_version": 1,
                                            "architecture": architecture_metadata(model)})
    assert not any("lora_" in k or ".base." in k for k in distributable_state_dict(model))
    with torch.no_grad():
        expected = model.embed(*_probe_batch())
        torch.testing.assert_close(load_style_encoder(str(tmp_path / f"x{ENCODER_SUFFIX}"), "sdpa").embed(
            *_probe_batch()), expected)
        fresh = StyleEncoder(tiny_vision(), (1, -2), 8, 2, 12, 6, "lora", LoRASpec(2, 2.0, 0.0)).eval()
        load_training_weights(fresh, str(tmp_path), "x")
        torch.testing.assert_close(fresh.embed(*_probe_batch()), expected)


@pytest.mark.parametrize("method,dropout", [("full", 0.0), ("lora", 0.0), ("lora", 0.5)])
def test_gradient_checkpointing_runs_through_the_direct_layer_calls(method, dropout):
    def grads(checkpointing):
        model = _lora_encoder(dropout) if method == "lora" else StyleEncoder(tiny_vision(), (1, -2), 8, 2, 12, 6,
                                                                             "full", None)
        model.set_dropout_entropy((1, 2, 3, 4))
        calls = []
        if checkpointing:
            model.vision_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            for layer in model.vision_encoder.encoder.layers:
                inner = layer._gradient_checkpointing_func
                layer._gradient_checkpointing_func = lambda *a, _f=inner, **k: calls.append(1) or _f(*a, **k)
        model.train()
        model.embed(*_probe_batch()).square().sum().backward()
        return calls, {n: p.grad.clone() for n, p in model.trainable_named_parameters() if p.grad is not None}

    calls, with_ckpt = grads(True)
    _, plain = grads(False)
    assert len(calls) == 2  # layers [0, depth) run; the third layer is past the deepest feature layer
    assert with_ckpt.keys() == plain.keys()
    assert any("lora_B" in n for n in plain) == (method == "lora")
    for name in plain:
        torch.testing.assert_close(with_ckpt[name], plain[name])


def test_importing_the_trainer_creates_no_cuda_context():
    code = ("import torch, core.style_encoder.style_trainer; "
            "raise SystemExit(1 if torch.cuda.is_initialized() else 0)")
    backend = Path(__file__).resolve().parents[1]
    assert subprocess.run([sys.executable, "-c", code], cwd=backend).returncode == 0


def test_loader_failure_raises_instead_of_retrying(tmp_path, monkeypatch):
    factory, ids, _ = build_corpus(tmp_path)
    real, failed = loader_module.train_view_entropy, []

    def flaky(seed, epoch, step, position):
        if step == 2 and not failed:
            failed.append(step)
            raise OSError("injected")
        return real(seed, epoch, step, position)

    monkeypatch.setattr(loader_module, "train_view_entropy", flaky)
    with pytest.raises(RuntimeError, match=r"epoch 0: steps 2\.\.3 were never delivered"):
        _run(factory, tmp_path / "out", config(ids))


def test_crash_while_overwriting_latest_keeps_the_previous_bundle_resumable(tmp_path, monkeypatch):
    factory, ids, _ = build_corpus(tmp_path)
    cfg, out = config(ids, epochs=2), tmp_path / "out"
    real, calls = artifacts._torch_save_atomic, []

    def crash_on_second_latest(obj, path):
        if os.path.basename(path).startswith("latest_"):
            calls.append(path)
            if len(calls) == 2:
                raise OSError("disk full")
        real(obj, path)

    monkeypatch.setattr(artifacts, "_torch_save_atomic", crash_on_second_latest)
    with pytest.raises(OSError, match="disk full"):
        _run(factory, out, cfg)
    monkeypatch.setattr(artifacts, "_torch_save_atomic", real)
    assert not any(".staged" in f for f in os.listdir(out))
    name, state = find_resume_checkpoint(str(out))
    assert (name, state["global_step"]) == ("latest", 4)

    resumed, rec = _run(factory, out, cfg)
    assert rec.of("resume")[0]["resumed_from_step"] == 4 and resumed["global_step"] == 8


def test_fresh_start_is_refused_over_an_incomplete_bundle(tmp_path):
    factory, ids, _ = build_corpus(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "latest_style_encoder.safetensors").write_bytes(b"left over")
    (out / "best_style_encoder.safetensors").write_bytes(b"not a resume bundle")
    with pytest.raises(ValueError, match="no complete resumable bundle"):
        _run(factory, out, config(ids))


def test_non_finite_loss_and_gradient_skip_the_step_without_ending_the_run(tmp_path, monkeypatch):
    factory, ids, _ = build_corpus(tmp_path)
    loss_calls, clip_calls = [], []
    real_loss, real_clip = StyleTrainer._loss, torch.nn.utils.clip_grad_norm_

    def nan_second(self, *args):
        loss, parts = real_loss(self, *args)
        loss_calls.append(1)
        return (loss * float("nan") if len(loss_calls) == 2 else loss), parts

    def inf_fourth(params, max_norm):
        clip_calls.append(1)
        norm = real_clip(params, max_norm=max_norm)
        return torch.tensor(float("inf")) if len(clip_calls) == 4 else norm

    monkeypatch.setattr(StyleTrainer, "_loss", nan_second)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", inf_fourth)
    result, rec = _run(factory, tmp_path / "out", config(ids, epochs=2))
    trainer = rec.holder[0]
    assert result["global_step"] == result["total_steps"] == trainer._optimizer_steps_completed == 8
    assert trainer._batches_entered == 10
    assert sorted(rec.losses()) == list(range(1, 9))


def _cap_index():
    resolver = label_resolver()
    sizes = {f"artist_c{a}": 1 + a % 4 for a in range(12)}
    records = [(i, stable_item_key(1, f"{i}.png"), resolver.resolve(None, name))
               for i, name in enumerate(n for n, k in sizes.items() for _ in range(k))]
    return build_style_index(records, SplitParams.create(0.99, 0.0, 1, 0), np.zeros(0, dtype=np.int64))


def test_eval_cap_keeps_a_rank_prefix_of_whole_artists():
    index = _cap_index()
    everything = eval_rows(index, VAL_ARTIST, 0, 3)
    assert everything.size == int((index.item_split == VAL_ARTIST).sum())
    counts = np.bincount(index.item_artist, minlength=len(index.artist_names))
    previous = set()
    oversized_first = False
    for cap in range(1, everything.size):
        rows = eval_rows(index, VAL_ARTIST, cap, 3)
        artists = set(index.item_artist[rows].tolist())
        assert rows.size == sum(counts[a] for a in artists) and artists
        assert rows.size <= cap or len(artists) == 1
        oversized_first |= rows.size > cap
        assert previous <= artists
        previous = artists
    assert oversized_first  # the first-ranked artist alone exceeds cap 1 here and is still kept
    assert eval_rows(index, VAL_IMAGE, 0, 3).size == 0


def test_warmup_cosine_factors():
    f = WarmupCosine(warmup_steps=2, total_steps=6, min_ratio=0.1)
    assert [f(0), f(1), f(2)] == [0.5, 1.0, 1.0]
    assert f(4) == pytest.approx(0.55) and f(6) == pytest.approx(0.1) and f(10) == pytest.approx(0.1)
    assert WarmupCosine(0, 4, 0.0)(0) == 1.0


def test_pause_estimate_uses_the_optimizer_state():
    import threading

    model = StyleEncoder(tiny_vision(), (1, -2), 8, 2, 12, 6, "full", None)
    trainable = [p for _, p in model.trainable_named_parameters()]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    handle = StyleTrainerHandle(run_id="r", output_dir=".", pause_event=threading.Event(),
                                resumed_event=threading.Event(), restored_event=threading.Event(),
                                optimizer_name="adamw")
    handle.attach(model, optimizer, None)
    model_bytes = sum(t.numel() * t.element_size() for t in (*model.parameters(), *model.buffers()))
    assert handle.estimate_state_bytes() == model_bytes + 8 * sum(p.numel() for p in trainable)
    model.embed(*_probe_batch()).sum().backward()
    optimizer.step()
    state = sum(v.numel() * v.element_size() for per in optimizer.state.values() for v in per.values())
    assert handle.estimate_state_bytes() == model_bytes + state


def test_encoder_dropout_is_refused():
    vision = tiny_vision()
    vision.config.attention_dropout = 0.1
    with pytest.raises(ValueError, match="attention_dropout=0.1"):
        StyleEncoder(vision, (1, -2), 8, 2, 12, 6, "full", None)
    vision = tiny_vision()
    vision.encoder.layers[0].mlp.fc1 = torch.nn.Sequential(vision.encoder.layers[0].mlp.fc1, torch.nn.Dropout(0.2))
    with pytest.raises(ValueError, match=r"encoder\.layers\.0\.mlp\.fc1\.1"):
        StyleEncoder(vision, (1, -2), 8, 2, 12, 6, "lora", LoRASpec(2, 2.0, 0.0))


class _Kill(BaseException):
    """A process kill: nothing after it runs, not even cleanup."""


def _killable_file_ops(monkeypatch, kill_at):
    """Every file mutation of a bundle save or recovery counts as one operation; from operation `kill_at`
    on, every one raises _Kill. Returns the running count."""
    import json
    import core.tagger.siglip2_tagger_model as siglip
    ops = {"n": 0}

    def wrap(fn):
        def op(*args, **kwargs):
            if kill_at is not None and ops["n"] >= kill_at:
                raise _Kill()
            ops["n"] += 1
            return fn(*args, **kwargs)
        return op

    for owner, name in ((os, "replace"), (os, "remove"), (torch, "save"), (np, "savez_compressed"),
                        (json, "dump"), (siglip, "_save_file_raw")):
        monkeypatch.setattr(owner, name, wrap(getattr(owner, name)))
    return ops


def _versioned_bundle(model, out, step, mode):
    from core.style_encoder.style_resume import SplitSnapshot
    state = {"global_step": step, "model": {"training_method": mode}}
    snapshot = SplitSnapshot(np.arange(step, dtype=np.int64))
    save_bundle(model, str(out), "latest", {"global_step": step}, (state, {"marker": step}, snapshot))


def _assert_consistent_bundle(out, models):
    """The resumable bundle found is complete and every file comes from the same save as its state."""
    import json
    from core.style_encoder.style_resume import SplitSnapshot
    found = find_resume_checkpoint(str(out))
    assert found is not None
    name, state = found
    step = state["global_step"]
    assert json.loads((out / f"{name}_metadata.json").read_text(encoding="utf-8"))["global_step"] == step
    assert torch.load(out / f"{name}_optimizer.pt", weights_only=True)["marker"] == step
    assert SplitSnapshot.load(out / f"{name}_split_snapshot.npz").keys.tolist() == list(range(step))
    saved = load_file(str(out / f"{name}{ENCODER_SUFFIX}"))
    expected = distributable_state_dict(models[step])
    assert saved.keys() == expected.keys() and all(torch.equal(saved[k], expected[k]) for k in saved)
    return step


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_a_kill_at_any_file_operation_of_a_save_leaves_one_consistent_resumable_bundle(tmp_path, monkeypatch,
                                                                                      mode):
    from core.style_encoder.style_artifacts import recover_staged_bundles
    old = _lora_encoder() if mode == "lora" else StyleEncoder(tiny_vision(), (1, -2), 8, 2, 12, 6, "full", None)
    new = _lora_encoder() if mode == "lora" else StyleEncoder(tiny_vision(), (1, -2), 8, 2, 12, 6, "full", None)
    with torch.no_grad():
        for p in new.parameters():
            p.add_(1.0)
    models = {1: old, 2: new}

    with monkeypatch.context() as m:
        counter = _killable_file_ops(m, None)
        _versioned_bundle(new, tmp_path / "count", 2, mode)
    save_ops = counter["n"]
    assert save_ops >= 12

    outcomes = []
    for kill_at in range(save_ops + 1):
        for recovery_kill in (None, 0, 1, 2, 3, 4, 5, 6):
            out = tmp_path / f"k{kill_at}_{recovery_kill}"
            _versioned_bundle(old, out, 1, mode)
            with monkeypatch.context() as m:
                _killable_file_ops(m, kill_at)
                try:
                    _versioned_bundle(new, out, 2, mode)
                except _Kill:
                    pass
            if recovery_kill is not None:
                with monkeypatch.context() as m:
                    _killable_file_ops(m, recovery_kill)
                    try:
                        recover_staged_bundles(str(out))
                    except _Kill:
                        pass
            _, refused, _ = recover_staged_bundles(str(out))
            assert refused == []
            step = _assert_consistent_bundle(out, models)
            assert not any(".staged" in f for f in os.listdir(out))
            outcomes.append((kill_at, step))
    by_kill = {}
    for kill_at, step in outcomes:
        by_kill.setdefault(kill_at, set()).add(step)
    assert all(len(steps) == 1 for steps in by_kill.values())
    steps = [by_kill[k].pop() for k in range(save_ops + 1)]
    # The old bundle survives until the state is staged; from then on the new one is finished.
    assert steps[0] == 1 and steps[-1] == 2 and steps == sorted(steps)


def test_an_interrupted_save_missing_a_staged_file_is_refused(tmp_path, monkeypatch):
    factory, ids, _ = build_corpus(tmp_path)
    cfg, out = config(ids), tmp_path / "out"
    model = StyleEncoder(tiny_vision(), (1, -2), 8, 2, 12, 6, "full", None)
    _versioned_bundle(model, out, 1, "full")
    with monkeypatch.context() as m:
        _killable_file_ops(m, 10)  # after the state is staged, before anything is moved into place
        with pytest.raises(_Kill):
            _versioned_bundle(model, out, 2, "full")
    assert (out / "latest_state.json.staged").is_file()
    (out / "latest_optimizer.pt.staged").unlink()
    (out / "latest_optimizer.pt").unlink()
    with pytest.raises(ValueError, match=r"interrupted checkpoint saves \['latest'\]"):
        _run(factory, out, cfg)
    assert (out / "latest_state.json.staged").is_file()


@pytest.mark.parametrize("workers", [0, 1])
def test_one_persistent_loader_serves_training_and_every_evaluation(tmp_path, monkeypatch, workers):
    import core.style_encoder.style_trainer as trainer_module
    from torch.utils.data import dataloader
    factory, ids, _ = build_corpus(tmp_path)
    loaders, worker_sets = [], []
    real_make, real_iter = trainer_module.make_loader, dataloader._MultiProcessingDataLoaderIter.__init__
    monkeypatch.setattr(trainer_module, "make_loader", lambda *a, **k: loaders.append(1) or real_make(*a, **k))
    monkeypatch.setattr(dataloader._MultiProcessingDataLoaderIter, "__init__",
                        lambda self, *a, **k: worker_sets.append(1) or real_iter(self, *a, **k))
    result, rec = _run(factory, tmp_path / "out", config(ids, epochs=2, num_workers=workers))
    assert len(rec.of("validation")) == 2 and len(loaders) == 1 and len(worker_sets) == min(workers, 1)
    if workers:
        _, serial = _run(factory, tmp_path / "serial", config(ids, epochs=2, num_workers=0))
        assert rec.losses() == serial.losses()
        assert rec.of("validation") == serial.of("validation")


def test_an_empty_encoder_path_loads_the_repo_release_and_records_it(tmp_path):
    import dataclasses
    import json
    factory, ids, _ = build_corpus(tmp_path)
    loads = []

    def loader(source, repo_id=None, attn_implementation="sdpa"):
        loads.append((source, repo_id))
        return tiny_vision(source, repo_id, attn_implementation)

    env = dataclasses.replace(environment(factory), load_vision_encoder=loader)
    repo = "google/siglip2-so400m-patch16-naflex"
    recorder = _Recorder()
    run_style_training("run", config(ids, vision_encoder_path="", vision_encoder_repo=repo), str(tmp_path / "out"),
                       recorder, recorder.holder, env)
    assert loads == [(repo, repo)]
    metadata = json.loads((tmp_path / "out" / "latest_metadata.json").read_text(encoding="utf-8"))
    assert metadata["base_model"] == {"repo_id": repo, "source": repo}


def _add_broad_dataset_over(factory, root, dataset_id=0):
    """A dataset with a lower id whose root contains every existing dataset: one copy of each item."""
    from database.models import Dataset, DatasetCaption, DatasetItem
    with factory() as session:
        session.add(Dataset(id=dataset_id, name="broad", path=str(root)))
        items = session.query(DatasetItem).all()
        for item in items:
            copy = DatasetItem(dataset_id=dataset_id, base_name=item.base_name, image_path=item.image_path)
            session.add(copy)
            session.flush()
            for caption in session.query(DatasetCaption).filter(DatasetCaption.item_id == item.id).all():
                session.add(DatasetCaption(item_id=copy.id, caption_type=caption.caption_type,
                                           content=caption.content, tag_data=caption.tag_data,
                                           is_tags_format=caption.is_tags_format))
        session.commit()
        return len(items)


def test_a_lower_id_dataset_over_trained_files_keeps_the_trained_copies_on_the_train_side(tmp_path):
    from core.style_encoder.style_dataset import TRAIN
    from core.style_encoder.style_resume import SplitSnapshot
    from core.style_encoder.style_trainer import prepare_style_data
    from style_training_fixture import image_processor
    factory, ids, _ = build_corpus(tmp_path)
    out = tmp_path / "out"
    _run(factory, out, config(ids, epochs=2), stop_at=3)
    trained = SplitSnapshot.load(out / "latest_split_snapshot.npz").keys
    assert trained.size

    copies = _add_broad_dataset_over(factory, tmp_path)
    widened = StyleTrainingConfig.from_dict(config([0, *ids], epochs=2))
    pinned = prepare_style_data(widened, environment(factory), image_processor(), trained)
    assert pinned.ingest_stats.items_duplicate_path == copies
    keys = pinned.index.item_keys
    order = np.argsort(keys)
    rows = order[np.minimum(np.searchsorted(keys[order], trained), keys.size - 1)]
    assert np.array_equal(keys[rows], trained)  # every trained copy survives under its key
    assert (pinned.index.item_split[rows] == TRAIN).all()
    fresh = prepare_style_data(widened, environment(factory), image_processor(), np.zeros(0, dtype=np.int64))
    assert not np.isin(trained, fresh.index.item_keys).any()  # without pins the lower-id copies win

    resumed, rec = _run(factory, out, config([0, *ids], epochs=2))
    assert rec.of("resume")[0]["resumed_from_step"] == 3 and resumed["global_step"] == resumed["total_steps"]


@pytest.mark.parametrize("first,second", [("", "google/siglip2-base-patch16-naflex"),
                                          ("google/siglip2-base-patch16-naflex", "")])
def test_an_empty_encoder_path_and_the_repo_id_are_the_same_resume_identity(tmp_path, first, second):
    factory, ids, _ = build_corpus(tmp_path)
    repo = "google/siglip2-base-patch16-naflex"
    out = tmp_path / "out"
    _run(factory, out, config(ids, epochs=2, vision_encoder_path=first, vision_encoder_repo=repo), stop_at=2)
    resumed, rec = _run(factory, out, config(ids, epochs=2, vision_encoder_path=second, vision_encoder_repo=repo))
    assert rec.of("resume")[0]["resumed_from_step"] == 2 and resumed["global_step"] == resumed["total_steps"]


def _best_bundle(model, out, step, mode):
    save_bundle(model, str(out), "best", {"global_step": step, "training_method": mode})


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_a_kill_at_any_file_operation_of_a_best_save_leaves_a_consistent_best(tmp_path, monkeypatch, mode):
    import json
    from core.style_encoder.style_artifacts import LORA_SUFFIX, lora_state_dict, recover_staged_bundles
    make = (lambda: _lora_encoder()) if mode == "lora" else \
        (lambda: StyleEncoder(tiny_vision(), (1, -2), 8, 2, 12, 6, "full", None))
    old, new = make(), make()
    with torch.no_grad():
        for p in new.parameters():
            p.add_(1.0)
    models = {1: old, 2: new}
    with monkeypatch.context() as m:
        counter = _killable_file_ops(m, None)
        _best_bundle(new, tmp_path / "count", 2, mode)
    save_ops = counter["n"]
    steps = []
    for kill_at in range(save_ops + 1):
        for recovery_kill in (None, 0, 1, 2, 3):
            out = tmp_path / f"k{kill_at}_{recovery_kill}"
            _best_bundle(old, out, 1, mode)
            with monkeypatch.context() as m:
                _killable_file_ops(m, kill_at)
                try:
                    _best_bundle(new, out, 2, mode)
                except _Kill:
                    pass
            if recovery_kill is not None:
                with monkeypatch.context() as m:
                    _killable_file_ops(m, recovery_kill)
                    try:
                        recover_staged_bundles(str(out))
                    except _Kill:
                        pass
            _, refused, _ = recover_staged_bundles(str(out))
            assert refused == [] and not any(".staged" in f for f in os.listdir(out))
            step = json.loads((out / "best_metadata.json").read_text(encoding="utf-8"))["global_step"]
            saved = load_file(str(out / f"best{ENCODER_SUFFIX}"))
            expected = distributable_state_dict(models[step])
            assert saved.keys() == expected.keys() and all(torch.equal(saved[k], expected[k]) for k in saved)
            if mode == "lora":
                adapter, expected_adapter = load_file(str(out / f"best{LORA_SUFFIX}")), lora_state_dict(models[step])
                assert all(torch.equal(adapter[k], expected_adapter[k]) for k in expected_adapter)
            if recovery_kill is None:
                steps.append(step)
    # The old best survives until the metadata is staged; from then on the new one is finished.
    assert steps[0] == 1 and steps[-1] == 2 and steps == sorted(steps)


class _SetupStopper(_Recorder):
    def __init__(self, when):
        super().__init__()
        self.when = when

    def __call__(self, run_id, event, data):
        super().__call__(run_id, event, data)
        if self.when(event, data):
            self.holder[0].stop()


@pytest.mark.parametrize("when", [lambda e, d: e == "dataset_stats",
                                  lambda e, d: e == "phase" and d.get("message") == "Building model..."])
def test_a_stop_during_setup_ends_the_run_before_the_gpu_is_touched(tmp_path, when):
    import dataclasses
    factory, ids, _ = build_corpus(tmp_path)
    devices = []
    env = dataclasses.replace(environment(factory), resolve_device=lambda g: devices.append(g) or torch.device("cpu"))
    recorder = _SetupStopper(when)
    result = run_style_training("run", config(ids), str(tmp_path / "out"), recorder, recorder.holder, env)
    assert result["stopped"] and result["global_step"] == 0 and devices == []
    assert os.listdir(tmp_path / "out") == [] and not recorder.of("step")


def test_a_stop_after_the_model_is_built_ends_the_run_before_the_loader_starts(tmp_path, monkeypatch):
    import core.style_encoder.style_trainer as trainer_module
    factory, ids, _ = build_corpus(tmp_path)
    real_build, loaders = StyleTrainer._build_optimizer, []

    def build_then_stop(self, *args):
        built = real_build(self, *args)
        self.stop()
        return built

    monkeypatch.setattr(StyleTrainer, "_build_optimizer", build_then_stop)
    monkeypatch.setattr(trainer_module, "make_loader", lambda *a, **k: loaders.append(1))
    result, rec = _run(factory, tmp_path / "out", config(ids))
    assert result["stopped"] and loaders == [] and not rec.of("step")


def test_a_feature_layer_out_of_range_names_the_layer_count():
    with pytest.raises(ValueError, match=r"this encoder has 3 layers, so entries must lie in \[-4, 3\]"):
        StyleEncoder(tiny_vision(), (9,), 8, 2, 12, 6, "full", None)
