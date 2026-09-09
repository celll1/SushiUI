"""Distill an SDXL VAE latent stem for the frozen REPA SigLIP trunk.

This deliberately lives outside training runs.  It writes one identity-bound
safetensors artifact plus a JSON report containing validation and gate-3 costs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import http.server
import json
import os
import random
import signal
import sqlite3
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.models.components.vae_registry import normalize
from core.training.image_preprocessing import flatten_to_rgb
from core.training.repa import encode_repa_targets, load_repa_encoder
from core.training.repa_latent_stem import (
    LatentRepaStem, economic_gate, encode_latent_targets, save_latent_stem,
    teacher_content_identity, vae_encoder_identity,
)

_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_RESUME_FORMAT = "sushi.repa_latent_stem.resume.v1"

_MONITOR_HTML = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>REPA stem distillation</title><style>
body{margin:0;background:#101318;color:#e8edf2;font:14px system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:24px}
h1{font-size:20px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:18px 0}
.card,section{background:#181d24;border:1px solid #29313c;border-radius:9px;padding:14px}.value{font-size:22px;margin-top:5px}
canvas{width:100%;height:420px}small{color:#9ca8b6}.train{color:#65b9ff}.val{color:#ffbd66}
</style></head><body><main><h1>REPA stem distillation</h1><small id="status">接続中…</small>
<div class="cards"><div class="card">進捗<div class="value" id="step">—</div></div>
<div class="card">train mean100<div class="value train" id="train">—</div></div>
<div class="card">validation loss<div class="value val" id="val">—</div></div>
<div class="card">速度<div class="value" id="speed">—</div></div>
<div class="card">最大VRAM<div class="value" id="vram">—</div></div>
<div class="card">経過時間<div class="value" id="elapsed">—</div></div></div>
<section><canvas id="chart"></canvas><small><span class="train">● train mean100</span>　<span class="val">● validation loss</span></small></section>
<script>
const $=id=>document.getElementById(id), cv=$('chart'), ctx=cv.getContext('2d');
const num=(v,n=6)=>Number.isFinite(v)?v.toFixed(n):'—';
function duration(s){s=Math.max(0,Math.round(s||0));return `${Math.floor(s/3600)}:${String(Math.floor(s/60)%60).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`}
function draw(rows){const dpr=devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;cv.width=w*dpr;cv.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
 const vals=rows.flatMap(r=>[r.train_loss_mean_100,r.validation_loss]).filter(Number.isFinite);if(!vals.length)return;
 let lo=Math.min(...vals),hi=Math.max(...vals);if(hi===lo){lo-=.01;hi+=.01}const pad=35,x=i=>pad+(w-2*pad)*(rows.length===1?0:i/(rows.length-1)),y=v=>h-pad-(h-2*pad)*(v-lo)/(hi-lo);
 ctx.strokeStyle='#3a4553';ctx.fillStyle='#9ca8b6';ctx.font='11px system-ui';for(let i=0;i<=4;i++){const yy=pad+(h-2*pad)*i/4;ctx.beginPath();ctx.moveTo(pad,yy);ctx.lineTo(w-pad,yy);ctx.stroke();ctx.fillText((hi-(hi-lo)*i/4).toFixed(3),2,yy+4)}
 function line(key,color){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();let started=false;rows.forEach((r,i)=>{const v=r[key];if(!Number.isFinite(v))return;started?ctx.lineTo(x(i),y(v)):ctx.moveTo(x(i),y(v));started=true});ctx.stroke()}line('train_loss_mean_100','#65b9ff');line('validation_loss','#ffbd66')}
async function update(){try{const res=await fetch('/progress',{cache:'no-store'}),rows=await res.json();if(!rows.length){$('status').textContent='ログ待機中';return}const r=rows.at(-1),v=[...rows].reverse().find(x=>Number.isFinite(x.validation_loss));$('status').textContent='2秒ごとに自動更新';$('step').textContent=`${r.step.toLocaleString()} / ${r.steps.toLocaleString()}`;$('train').textContent=num(r.train_loss_mean_100);$('val').textContent=v?num(v.validation_loss):'—';$('speed').textContent=`${num(r.mean_ms_per_item,1)} ms/item`;$('vram').textContent=`${num(r.max_vram_gib,2)} GiB`;$('elapsed').textContent=duration(r.elapsed_seconds);draw(rows)}catch(e){$('status').textContent='CLIへの接続が終了しました（最後の表示を保持）'}}
update();setInterval(update,2000);addEventListener('resize',update);
</script></main></body></html>"""


def _start_monitor(progress_path: Path, port: int):
    """Serve an ephemeral localhost-only view of one progress JSONL file."""
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                payload, content_type = _MONITOR_HTML.encode(), "text/html; charset=utf-8"
            elif self.path == "/progress":
                rows = []
                if progress_path.is_file():
                    for line in progress_path.read_text(encoding="utf-8").splitlines():
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                payload = json.dumps(rows).encode()
                content_type = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format, *args):
            pass

    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as error:
        try:
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        except OSError as fallback_error:
            print(
                f"[REPA stem] monitor disabled: {fallback_error} "
                f"(requested port {port} also failed: {error})",
                flush=True)
            return None
        print(
            f"[REPA stem] monitor port {port} unavailable; using dynamic port "
            f"{server.server_port}", flush=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[REPA stem] monitor=http://127.0.0.1:{server.server_port}", flush=True)
    return server


def _selection_identity(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _resume_contract(args, *, paths, vae_identity, teacher_identity):
    return {
        "selection_identity": _selection_identity(paths),
        "vae_encoder_identity": vae_identity,
        "teacher_identity": teacher_identity,
        "width": args.width,
        "height": args.height,
        "bucket_strategy": args.bucket_strategy,
        "width_channels": args.width_channels,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dtype": args.dtype,
        "vae_dtype": args.vae_dtype,
    }


def _save_resume_checkpoint(path: Path, *, stem, optimizer, scaler, step,
                            processed_items, recent_losses, item_time_sum,
                            item_time_count, contract) -> None:
    stem_device = stem.in_proj.weight.device
    cuda_rng_state = (
        torch.cuda.get_rng_state(stem_device)
        if stem_device.type == "cuda" else torch.empty(0, dtype=torch.uint8))
    payload = {
        "format": _RESUME_FORMAT,
        "contract": contract,
        "stem": stem.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "processed_items": processed_items,
        "recent_losses": list(recent_losses),
        "item_time_sum": item_time_sum,
        "item_time_count": item_time_count,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": cuda_rng_state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_resume_checkpoint(path: Path, *, stem, optimizer, scaler, contract,
                            device, planned_steps: int):
    if not path.is_file():
        raise FileNotFoundError(f"REPA stem resume checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != _RESUME_FORMAT:
        raise ValueError(f"{path}: unsupported resume format {payload.get('format')!r}")
    saved_contract = payload.get("contract", {})
    mismatches = [
        f"{key}: checkpoint={saved_contract.get(key)!r}, run={value!r}"
        for key, value in contract.items() if saved_contract.get(key) != value
    ]
    if mismatches:
        raise ValueError(f"{path}: resume contract mismatch (" + "; ".join(mismatches) + ")")
    step = int(payload["step"])
    if planned_steps < step:
        raise ValueError(
            f"planned steps {planned_steps} is below resumed step {step}")
    stem.load_state_dict(payload["stem"])
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(payload["cuda_rng_state"].cpu(), device)
    return {
        "step": step,
        "processed_items": int(payload["processed_items"]),
        "recent_losses": list(payload.get("recent_losses", [])),
        "item_time_sum": float(payload.get("item_time_sum", 0.0)),
        "item_time_count": int(payload.get("item_time_count", 0)),
    }


def _load_vae(args, dtype):
    if args.vae_source:
        from core.models.common.vae_source import resolve_vae_source
        return resolve_vae_source(args.vae_source, arch="sdxl").load_module(dtype)
    path = Path(args.base_model)
    if path.is_dir():
        from diffusers import AutoencoderKL
        return AutoencoderKL.from_pretrained(str(path), subfolder="vae", torch_dtype=dtype)
    from diffusers import StableDiffusionXLPipeline
    pipeline = StableDiffusionXLPipeline.from_single_file(
        str(path), torch_dtype=dtype, use_safetensors=True)
    vae = pipeline.vae
    pipeline.vae = None
    del pipeline
    return vae


def _fit(image: Image.Image, width: int, height: int, strategy: str) -> Image.Image:
    image = flatten_to_rgb(image)
    if strategy == "resize":
        return image.resize((width, height), Image.LANCZOS)
    scale = max(width / image.width, height / image.height)
    resized = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy((array - 0.5) * 2.0).permute(2, 0, 1)


def _images(roots: list[Path], dataset_ids: list[int], datasets_db: Path,
            seed: int, max_items: int) -> list[Path]:
    dataset_ids = list(dict.fromkeys(dataset_ids))
    if not roots and not dataset_ids:
        raise ValueError("provide at least one --image-dir or --dataset-id")
    rng = random.Random(seed)
    selected: list[Path] = []
    seen = 0
    reservoir_size = max_items * 2 if max_items > 0 else 0

    def consider(raw_path) -> None:
        nonlocal seen
        path = Path(raw_path)
        if path.suffix.lower() not in _EXTENSIONS:
            return
        seen += 1
        if reservoir_size <= 0 or len(selected) < reservoir_size:
            selected.append(path)
            return
        replace = rng.randrange(seen)
        if replace < reservoir_size:
            selected[replace] = path

    for root in roots:
        for path in root.rglob("*"):
            if path.is_file():
                consider(path)

    if dataset_ids:
        uri = f"file:{datasets_db.resolve().as_posix()}?mode=ro"
        placeholders = ",".join("?" for _ in dataset_ids)
        with sqlite3.connect(uri, uri=True) as database:
            known = {
                row[0]: (row[1], row[2]) for row in database.execute(
                    f"SELECT id, name, total_items FROM datasets WHERE id IN ({placeholders})",
                    dataset_ids)
            }
            missing = sorted(set(dataset_ids) - set(known))
            if missing:
                raise ValueError(f"unknown dataset id(s): {missing}")
            print("[REPA stem] dataset pool: " + ", ".join(
                f"{dataset_id}:{known[dataset_id][0]} ({known[dataset_id][1]} items)"
                for dataset_id in dataset_ids))
            rows = database.execute(
                f"SELECT image_path FROM dataset_items "
                f"WHERE dataset_id IN ({placeholders}) ORDER BY dataset_id, id",
                dataset_ids)
            for row in rows:
                consider(row[0])

    paths = list(dict.fromkeys(path for path in selected if path.is_file()))
    if max_items > 0 and len(paths) < max_items:
        raise ValueError(
            f"selected only {len(paths)} existing unique images from {seen} candidates; "
            f"requested {max_items}")
    rng.shuffle(paths)
    return paths[:max_items] if max_items > 0 else paths


def _onnx_companion(checkpoint: str) -> str:
    """Find the differentiable safetensors source for a deployment ONNX."""
    from core.training.repa import _checkpoint_metadata

    checkpoint_path = Path(checkpoint).resolve()
    metadata = _checkpoint_metadata(str(checkpoint_path))
    for key in ("source_checkpoint", "checkpoint_path"):
        named = metadata.get(key)
        if not isinstance(named, str) or not named.strip():
            continue
        candidate = Path(named)
        if not candidate.is_absolute():
            candidate = checkpoint_path.parent / candidate
        if candidate.is_file() and candidate.suffix.lower() == ".safetensors":
            return str(candidate.resolve())

    for directory in (checkpoint_path.parent, checkpoint_path.parent.parent):
        for name in ("latest.safetensors", "best_f1.safetensors"):
            candidate = directory / name
            if candidate.is_file():
                return str(candidate.resolve())
    raise FileNotFoundError(
        f"{checkpoint}: ONNX distillation needs its source .safetensors for "
        "differentiable trunk gradients; add source_checkpoint to model_metadata.json "
        "or keep latest.safetensors in the ONNX directory or its parent")


def _paired_batch(paths: list[Path], args, teacher_size: int | None = None):
    images, teachers = [], []
    for path in paths:
        with Image.open(path) as image:
            fitted = _fit(image, args.width, args.height, args.bucket_strategy)
            images.append(_tensor(fitted))
            if teacher_size is not None:
                teachers.append(_tensor(fitted.resize(
                    (teacher_size, teacher_size), Image.BICUBIC)))
    image_batch = torch.stack(images)
    return (image_batch, torch.stack(teachers)) if teacher_size is not None else image_batch


def _batch(paths: list[Path], args) -> torch.Tensor:
    return _paired_batch(paths, args)


def _latent(vae, pixels: torch.Tensor) -> torch.Tensor:
    pixels = pixels.to(dtype=next(vae.parameters()).dtype)
    posterior = vae.encode(pixels).latent_dist
    return normalize(posterior.sample(), vae, None)


def _cosine(student: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(student.float(), target.float(), dim=-1).mean()


def _validation_probe(vae, teacher, stem, paths, args, teacher_size, dtype,
                      teacher_dtype, device) -> float:
    """Measure a fixed held-out subset without retaining teacher activations."""
    weighted_cosine = 0.0
    items = 0
    stem.eval()
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    with torch.random.fork_rng(devices=[device_index]):
        torch.manual_seed(args.seed + 10_000_003)
        with torch.no_grad():
            for offset in range(0, len(paths), args.batch_size):
                selected = paths[offset:offset + args.batch_size]
                cpu, teacher_cpu = _paired_batch(selected, args, teacher_size)
                pixels = cpu.to(device=device, dtype=dtype)
                teacher_pixels = teacher_cpu.to(device=device, dtype=teacher_dtype)
                latents = _latent(vae, pixels)
                targets = encode_repa_targets(
                    teacher, teacher_pixels, 27, 27, teacher_size)
                with torch.autocast(
                        "cuda", dtype=dtype,
                        enabled=teacher_dtype != torch.float32):
                    predicted = encode_latent_targets(
                        teacher, stem, latents, 27, 27)
                weighted_cosine += (
                    float(_cosine(predicted, targets).item()) * len(selected))
                items += len(selected)
    stem.train()
    return weighted_cosine / items


def run(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("distill_repa_latent_stem requires CUDA")
    if not args.vae_source and not args.base_model:
        raise ValueError("provide --base-model or --vae-source")
    if args.steps <= 0 or args.batch_size <= 0 or args.online_batch_size <= 0:
        raise ValueError("steps and batch sizes must be positive")
    if (args.validation_every <= 0 or args.validation_probe_items <= 0
            or args.checkpoint_every <= 0):
        raise ValueError("validation/checkpoint intervals and probe items must be positive")
    if not 0 <= args.monitor_port <= 65535:
        raise ValueError("monitor-port must be between 0 and 65535")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val-fraction must be between zero and one")
    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    vae_dtype = {
        "fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16,
    }[args.vae_dtype]
    torch.manual_seed(args.seed)

    paths = _images(
        args.image_dir, args.dataset_id, args.datasets_db, args.seed, args.max_items)
    if len(paths) < 3:
        raise ValueError("distillation needs at least three readable images")
    split = min(len(paths) - 2, max(1, round(len(paths) * (1.0 - args.val_fraction))))
    train_paths, val_paths = paths[:split], paths[split:]
    validation_probe_paths = val_paths[:args.validation_probe_items]

    vae = _load_vae(args, vae_dtype).to(
        device=device, dtype=vae_dtype).eval().requires_grad_(False)
    vae_identity, vae_norm = vae_encoder_identity(vae)
    from core.training.repa import _resolve_tagger_checkpoint
    teacher_checkpoint, teacher_repo = _resolve_tagger_checkpoint(args.tagger_model)
    teacher, enc_dim, native_size = load_repa_encoder(
        "tagger", tagger_model_dir=args.tagger_model, dtype=dtype,
        device=device, attn_implementation=args.attention)
    teacher_identity = teacher_content_identity(teacher)
    teacher_size = int(native_size or 384)
    verification_teacher = None
    distill_checkpoint = teacher_checkpoint
    distill_teacher = teacher
    distill_dtype = dtype
    if teacher_checkpoint.lower().endswith(".onnx"):
        distill_checkpoint = _onnx_companion(teacher_checkpoint)
        distill_dtype = torch.float32
        distill_teacher, distill_dim, distill_size = load_repa_encoder(
            "tagger", tagger_model_dir=distill_checkpoint, dtype=distill_dtype,
            device=device, attn_implementation=args.attention)
        if distill_dim != enc_dim or int(distill_size or 384) != teacher_size:
            raise ValueError(
                "ONNX and safetensors companion disagree on REPA dimensions: "
                f"onnx=({teacher_size}, {enc_dim}), "
                f"safetensors=({distill_size}, {distill_dim})")
        verification_teacher = teacher
    print(f"[REPA stem] distillation target=post_layernorm backend={distill_checkpoint}")
    if (hasattr(distill_teacher, "gradient_checkpointing_enable")
            and args.gradient_checkpointing):
        distill_teacher.gradient_checkpointing_enable()

    in_channels = int(getattr(vae.config, "latent_channels", 4))
    stem = LatentRepaStem(in_channels, enc_dim, args.width_channels).to(device=device)
    optimizer = torch.optim.AdamW(stem.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "fp16")
    progress_path = args.progress_jsonl or args.output.with_suffix(".progress.jsonl")
    resume_path = args.resume_checkpoint or args.output.with_suffix(".resume.pt")
    resume_contract = _resume_contract(
        args, paths=paths, vae_identity=vae_identity, teacher_identity=teacher_identity)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        resumed = _load_resume_checkpoint(
            resume_path, stem=stem, optimizer=optimizer, scaler=scaler,
            contract=resume_contract, device=device, planned_steps=args.steps)
    else:
        progress_path.write_text("", encoding="utf-8")
        resumed = {
            "step": 0, "processed_items": 0, "recent_losses": [],
            "item_time_sum": 0.0, "item_time_count": 0,
        }
    monitor = _start_monitor(progress_path, args.monitor_port) if args.monitor_port else None
    step = resumed["step"]
    processed_items = resumed["processed_items"]
    item_time_sum = resumed["item_time_sum"]
    item_time_count = resumed["item_time_count"]
    recent_losses = deque(resumed["recent_losses"], maxlen=100)
    if args.resume:
        print(f"[REPA stem] resumed={resume_path} step={step}/{args.steps}", flush=True)
    run_started = time.perf_counter()
    batches_per_epoch = (len(train_paths) + args.batch_size - 1) // args.batch_size
    stop_requested = False
    previous_sigint = signal.getsignal(signal.SIGINT)

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        if stop_requested:
            signal.signal(signal.SIGINT, previous_sigint)
            raise KeyboardInterrupt
        stop_requested = True
        print(
            "\n[REPA stem] Ctrl+C received; finishing this optimizer step, then saving...",
            flush=True)

    signal.signal(signal.SIGINT, request_stop)
    stem.train()
    order = []
    current_epoch = -1
    while step < args.steps:
        if stop_requested:
            break
        epoch = step // batches_per_epoch
        batch_index = step % batches_per_epoch
        if epoch != current_epoch:
            order = list(train_paths)
            random.Random(args.seed + epoch).shuffle(order)
            current_epoch = epoch
        offset = batch_index * args.batch_size
        selected = order[offset:offset + args.batch_size]
        cpu, teacher_cpu = _paired_batch(selected, args, teacher_size)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        pixels = cpu.to(device=device, dtype=dtype)
        teacher_pixels = teacher_cpu.to(device=device, dtype=distill_dtype)
        with torch.no_grad():
            latents = _latent(vae, pixels)
            targets = encode_repa_targets(
                distill_teacher, teacher_pixels, 27, 27, teacher_size)
            if verification_teacher is not None:
                exported = encode_repa_targets(
                    verification_teacher, teacher_pixels, 27, 27, teacher_size)
                agreement = float(_cosine(exported, targets).item())
                if agreement < 0.999:
                    raise ValueError(
                        f"ONNX/source safetensors feature cosine is {agreement:.6f}; "
                        "refusing a mismatched distillation companion")
                print(f"[REPA stem] ONNX/source feature cosine={agreement:.8f}")
                verification_teacher._sessions.clear()
                verification_teacher = None
                teacher = None
                gc.collect()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
                "cuda", dtype=dtype, enabled=distill_dtype != torch.float32):
            predicted = encode_latent_targets(
                distill_teacher, stem, latents, 27, 27)
            loss = 1.0 - _cosine(predicted, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        item_time_sum += (time.perf_counter() - started) * 1000.0 / len(selected)
        item_time_count += 1
        processed_items += len(selected)
        step += 1
        loss_value = float(loss.item())
        recent_losses.append(loss_value)
        should_validate = (
            step == 1 or step % args.validation_every == 0 or step >= args.steps)
        validation_cosine = None
        if should_validate and not stop_requested:
            validation_cosine = _validation_probe(
                vae, distill_teacher, stem, validation_probe_paths, args,
                teacher_size, dtype, distill_dtype, device)
        if (step == 1 or step % args.log_every == 0 or step >= args.steps
                or should_validate or stop_requested):
            entry = {
                "step": step,
                "steps": args.steps,
                "items": processed_items,
                "train_loss": loss_value,
                "train_loss_mean_100": sum(recent_losses) / len(recent_losses),
                "elapsed_seconds": time.perf_counter() - run_started,
                "mean_ms_per_item": item_time_sum / item_time_count,
                "max_vram_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                "interrupted": stop_requested,
            }
            if validation_cosine is not None:
                entry["validation_items"] = len(validation_probe_paths)
                entry["validation_patch_cosine"] = validation_cosine
                entry["validation_loss"] = 1.0 - validation_cosine
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
            validation_text = (
                "" if validation_cosine is None
                else f" val_loss={1.0 - validation_cosine:.6f}")
            print(
                f"[REPA stem] step={step}/{args.steps} loss={loss_value:.6f} "
                f"mean100={entry['train_loss_mean_100']:.6f}{validation_text}",
                flush=True)
        if (step % args.checkpoint_every == 0 or step >= args.steps
                or stop_requested):
            _save_resume_checkpoint(
                resume_path, stem=stem, optimizer=optimizer, scaler=scaler, step=step,
                processed_items=processed_items, recent_losses=recent_losses,
                item_time_sum=item_time_sum, item_time_count=item_time_count,
                contract=resume_contract)
        if stop_requested:
            break
    signal.signal(signal.SIGINT, previous_sigint)
    interrupted = stop_requested

    distill_ms = item_time_sum / item_time_count
    metadata = {
        "vae_encoder_identity": vae_identity,
        "vae_normalization": vae_norm,
        "teacher_identity": teacher_identity,
        "teacher_checkpoint": str(Path(teacher_checkpoint).resolve()),
        "teacher_repo": teacher_repo,
        "teacher_size": teacher_size,
        "teacher_dtype": args.dtype,
        "distillation_target": "post_layernorm",
        "distillation_backend_checkpoint": distill_checkpoint,
        "distillation_backend_dtype": str(distill_dtype).removeprefix("torch."),
        "vae_dtype": args.vae_dtype,
        "training_resolution": [args.height, args.width],
        "bucket_strategy": args.bucket_strategy,
        "dataset_ids": list(dict.fromkeys(args.dataset_id)),
        "image_dirs": [str(path.resolve()) for path in args.image_dir],
        "selected_unique_items": len(paths),
        "training_items": len(train_paths),
        "validation_items": len(val_paths),
        "progress_jsonl": str(progress_path.resolve()),
        "resume_checkpoint": str(resume_path.resolve()),
        "interrupted": interrupted,
        "distill_items": processed_items,
        "distill_steps": step,
        "planned_distill_steps": args.steps,
    }
    save_latent_stem(args.output, stem, metadata)
    if interrupted:
        print(f"[REPA stem] interrupted artifact saved: {args.output}", flush=True)

    evaluation_paths = validation_probe_paths if interrupted else val_paths
    stem.eval()
    cosines, shuffled = [], []
    with torch.no_grad():
        for offset in range(0, len(evaluation_paths), args.batch_size):
            cpu, teacher_cpu = _paired_batch(
                evaluation_paths[offset:offset + args.batch_size], args, teacher_size)
            pixels = cpu.to(device=device, dtype=dtype)
            teacher_pixels = teacher_cpu.to(device=device, dtype=distill_dtype)
            latents = _latent(vae, pixels)
            targets = encode_repa_targets(
                distill_teacher, teacher_pixels, 27, 27, teacher_size)
            with torch.autocast(
                    "cuda", dtype=dtype, enabled=distill_dtype != torch.float32):
                predicted = encode_latent_targets(
                    distill_teacher, stem, latents, 27, 27)
            cosines.append(float(_cosine(predicted, targets).item()))
            shuffled.append(float(_cosine(predicted, targets.flip(1)).item()))

        pair_cpu, pair_teacher_cpu = _paired_batch(
            evaluation_paths[:2], args, teacher_size)
        pair_pixels = pair_cpu.to(device=device, dtype=dtype)
        pair_teacher_pixels = pair_teacher_cpu.to(device=device, dtype=distill_dtype)
        pair_latents = _latent(vae, pair_pixels)
        pair_targets = encode_repa_targets(
            distill_teacher, pair_teacher_pixels, 27, 27, teacher_size)
        with torch.autocast(
                "cuda", dtype=dtype, enabled=distill_dtype != torch.float32):
            pair_predicted = encode_latent_targets(
                distill_teacher, stem, pair_latents, 27, 27)
        different_image_cosine = float(
            _cosine(pair_predicted, pair_targets.roll(1, dims=0)).item())

    probe_latent = _latent(vae, _batch([val_paths[0]], args).to(device=device, dtype=dtype))
    def _stem_forward():
        with torch.autocast("cuda", dtype=dtype):
            return stem(probe_latent)

    for _ in range(5):
        _stem_forward()
    samples = []
    for _ in range(30):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _stem_forward()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    stem_ms = sorted(samples)[len(samples) // 2]
    gate = economic_gate(
        redistill_items=processed_items, distill_ms_per_item=distill_ms,
        online_steps=args.expected_online_steps, batch_size=args.online_batch_size,
        replaced_ms_per_item=args.replaced_ms_per_item, stem_ms_per_item=stem_ms)
    report = {
        **metadata,
        "artifact": str(args.output.resolve()),
        "evaluated_validation_items": len(evaluation_paths),
        "validation_patch_cosine": sum(cosines) / len(cosines),
        "validation_different_image_cosine": different_image_cosine,
        "validation_spatially_reversed_cosine": sum(shuffled) / len(shuffled),
        "distill_ms_per_item": distill_ms,
        "stem_median_ms_per_item": stem_ms,
        "economic_gate": gate,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if monitor is not None:
        monitor.shutdown()
        monitor.server_close()
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, action="append", default=[])
    parser.add_argument("--dataset-id", type=int, action="append", default=[])
    parser.add_argument(
        "--datasets-db", type=Path,
        default=Path(__file__).resolve().parents[4] / "datasets.db")
    parser.add_argument("--base-model", default="")
    parser.add_argument("--vae-source", default="")
    parser.add_argument(
        "--tagger-model", "--tagger-dir", dest="tagger_model", required=True,
        help="Exact .onnx/.safetensors file, or a legacy tagger model directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--bucket-strategy", choices=("resize", "crop"), default="resize")
    parser.add_argument("--width-channels", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--max-items", type=int, default=4096)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--validation-every", type=int, default=500)
    parser.add_argument("--validation-probe-items", type=int, default=64)
    parser.add_argument("--progress-jsonl", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument(
        "--monitor-port", type=int, default=0,
        help="Serve an ephemeral localhost loss monitor on this port (0 disables it)")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--vae-dtype", choices=("fp32", "bf16", "fp16"), default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--expected-online-steps", type=int, required=True)
    parser.add_argument("--online-batch-size", type=int, default=4)
    parser.add_argument("--replaced-ms-per-item", type=float, default=6.02)
    return parser


if __name__ == "__main__":
    result = run(_parser().parse_args())
    print(json.dumps(result, indent=2))
