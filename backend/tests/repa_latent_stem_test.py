import json
import sqlite3
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from core.training.repa import RepaProjector, apply_repa_loss
from core.training.repa_latent_stem import (
    LatentRepaStem, economic_gate, encode_latent_targets, load_latent_stem,
    save_latent_stem, vae_encoder_identity,
)
from core.training.probes import distill_repa_latent_stem as distill_probe
from core.training.probes.distill_repa_latent_stem import (
    _images, _load_resume_checkpoint, _onnx_companion,
    _save_resume_checkpoint, _start_monitor,
)


class _VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 1)
        self.quant_conv = nn.Conv2d(4, 8, 1)
        self.decoder = nn.Conv2d(4, 3, 1)
        self.config = {
            "scaling_factor": 0.13025,
            "shift_factor": None,
            "latents_mean": None,
            "latents_std": None,
            "batch_norm_eps": None,
        }


class _Trunk(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Linear(dim, dim, bias=False)

    def forward(self, inputs_embeds, return_dict=True):
        return SimpleNamespace(last_hidden_state=self.layer(inputs_embeds))


class _Teacher(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.position_embedding = nn.Embedding(27 * 27, dim)
        self.embeddings.register_buffer(
            "position_ids", torch.arange(27 * 27).unsqueeze(0), persistent=False)
        self.encoder = _Trunk(dim)
        self.post_layernorm = nn.LayerNorm(dim)


def test_vae_encoder_identity_ignores_decoder_but_not_encoder_or_normalization():
    vae = _VAE()
    initial, norm = vae_encoder_identity(vae)
    assert norm["scaling_factor"] == 0.13025

    with torch.no_grad():
        vae.decoder.weight.add_(1)
    assert vae_encoder_identity(vae)[0] == initial

    with torch.no_grad():
        vae.encoder.weight.add_(1)
    assert vae_encoder_identity(vae)[0] != initial

    changed = _VAE()
    changed.load_state_dict(_VAE().state_dict())
    before = vae_encoder_identity(changed)[0]
    changed.config["scaling_factor"] = 0.5
    assert vae_encoder_identity(changed)[0] != before


def test_artifact_roundtrip_and_identity_refusal(tmp_path):
    vae = _VAE()
    vae_id, norm = vae_encoder_identity(vae)
    stem = LatentRepaStem(4, 8, 32)
    path = tmp_path / "stem.safetensors"
    save_latent_stem(path, stem, {
        "vae_encoder_identity": vae_id,
        "vae_normalization": norm,
        "teacher_identity": "teacher-a",
    })
    loaded, metadata = load_latent_stem(
        path, vae=vae, teacher_identity="teacher-a", encoder_dim=8,
        device="cpu", dtype=torch.float32)
    assert metadata["grid"] == 27
    assert not any(parameter.requires_grad for parameter in loaded.parameters())

    with pytest.raises(ValueError, match="teacher_identity"):
        load_latent_stem(
            path, vae=vae, teacher_identity="teacher-b", encoder_dim=8,
            device="cpu", dtype=torch.float32)


def test_latent_targets_keep_teacher_position_and_trunk():
    teacher = _Teacher().requires_grad_(False)
    stem = LatentRepaStem(4, 8, 32)
    latents = torch.randn(2, 4, 16, 20)
    targets = encode_latent_targets(teacher, stem, latents, 9, 11)
    assert targets.shape == (2, 99, 8)

    targets.sum().backward()
    assert stem.in_proj.weight.grad is not None
    assert teacher.encoder.layer.weight.grad is None


def test_economic_gate_reports_break_even():
    result = economic_gate(
        redistill_items=100, distill_ms_per_item=20,
        online_steps=100, batch_size=4,
        replaced_ms_per_item=6.02, stem_ms_per_item=0.08)
    assert result["passes"] is True
    assert result["break_even_steps"] == pytest.approx(84.175084, rel=1e-6)


def test_dataset_ids_form_one_deterministic_image_pool(tmp_path):
    db_path = tmp_path / "datasets.db"
    paths = []
    for index in range(12):
        path = tmp_path / f"image-{index}.png"
        path.write_bytes(b"image")
        paths.append(path)
    with sqlite3.connect(db_path) as database:
        database.execute(
            "CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT, total_items INTEGER)")
        database.execute(
            "CREATE TABLE dataset_items "
            "(id INTEGER PRIMARY KEY, dataset_id INTEGER, image_path TEXT)")
        database.executemany(
            "INSERT INTO datasets VALUES (?, ?, ?)",
            [(1, "large", 10), (2, "small", 2)])
        database.executemany(
            "INSERT INTO dataset_items VALUES (?, ?, ?)",
            [(index + 1, 1 if index < 10 else 2, str(path))
             for index, path in enumerate(paths)])

    first = _images([], [1, 2], db_path, seed=7, max_items=8)
    second = _images([], [2, 1, 2], db_path, seed=7, max_items=8)
    assert first == second
    assert len(first) == len(set(first)) == 8
    assert set(first) <= set(paths)


def test_onnx_distillation_finds_parent_safetensors(tmp_path):
    export = tmp_path / "v2_01a"
    export.mkdir()
    onnx = export / "model.onnx"
    onnx.write_bytes(b"onnx")
    companion = tmp_path / "latest.safetensors"
    companion.write_bytes(b"weights")
    assert _onnx_companion(str(onnx)) == str(companion.resolve())


def test_ephemeral_monitor_serves_only_html_and_progress(tmp_path):
    progress = tmp_path / "run.progress.jsonl"
    progress.write_text('{"step": 3, "train_loss": 0.25}\n', encoding="utf-8")
    server = _start_monitor(progress, 0)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/", timeout=2) as response:
            assert b"REPA stem distillation" in response.read()
        with urllib.request.urlopen(base + "/progress", timeout=2) as response:
            assert json.loads(response.read()) == [{"step": 3, "train_loss": 0.25}]
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(base + "/other", timeout=2)
        assert refused.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_ephemeral_monitor_falls_back_from_a_blocked_port(tmp_path, monkeypatch):
    real_server = distill_probe.http.server.ThreadingHTTPServer

    def bind(address, handler):
        if address[1] == 8765:
            raise PermissionError(10013, "blocked")
        return real_server(address, handler)

    monkeypatch.setattr(distill_probe.http.server, "ThreadingHTTPServer", bind)
    server = _start_monitor(tmp_path / "progress.jsonl", 8765)
    try:
        assert server is not None
        assert server.server_port != 8765
    finally:
        server.shutdown()
        server.server_close()


def test_resume_checkpoint_restores_model_optimizer_and_progress(tmp_path):
    stem = LatentRepaStem(4, 8, 32)
    optimizer = torch.optim.AdamW(stem.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    stem(torch.randn(1, 4, 8, 8)).sum().backward()
    optimizer.step()
    expected = {key: value.detach().clone() for key, value in stem.state_dict().items()}
    path = tmp_path / "stem.resume.pt"
    _save_resume_checkpoint(
        path, stem=stem, optimizer=optimizer, scaler=scaler, step=17,
        processed_items=68, recent_losses=[0.4, 0.3], item_time_sum=12.5,
        item_time_count=2, contract={"teacher": "a"})

    restored = LatentRepaStem(4, 8, 32)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    state = _load_resume_checkpoint(
        path, stem=restored, optimizer=restored_optimizer, scaler=scaler,
        contract={"teacher": "a"}, device=torch.device("cpu"), planned_steps=20)
    assert state == {
        "step": 17, "processed_items": 68, "recent_losses": [0.4, 0.3],
        "item_time_sum": 12.5, "item_time_count": 2,
    }
    assert all(torch.equal(restored.state_dict()[key], value)
               for key, value in expected.items())
    assert restored_optimizer.state_dict()["state"]

    with pytest.raises(ValueError, match="resume contract mismatch"):
        _load_resume_checkpoint(
            path, stem=restored, optimizer=restored_optimizer, scaler=scaler,
            contract={"teacher": "b"}, device=torch.device("cpu"), planned_steps=20)


def test_production_loss_switch_consumes_clean_latents():
    teacher = _Teacher().requires_grad_(False)
    stem = LatentRepaStem(4, 8, 32).requires_grad_(False)
    logged = []
    trainer = SimpleNamespace(
        device=torch.device("cpu"), training_dtype=torch.float32,
        repa_target_source="latent_stem", repa_encoder=teacher,
        repa_latent_stem=stem, repa_projector=RepaProjector(6, 8, hidden=16),
        repa_weight=0.5, log_extra_metric=lambda name, value: logged.append((name, value)),
    )
    image_tokens = torch.randn(2, 9 * 11, 6, requires_grad=True)
    clean_latents = torch.randn(2, 4, 16, 20)
    result = apply_repa_loss(
        trainer, image_tokens.square().mean(), image_tokens, clean_latents, 9, 11)
    result.backward()
    assert image_tokens.grad is not None
    assert logged and logged[0][0] == "repa_loss"
