"""Standalone end-to-end smoke/gate driver for SenseNova-U1.5-8B-MoT.

Loads the model directly (bypassing ``PipelineManager``/routes, which is a
later unit's job), runs SushiUI's own denoise loop
(``core.models.sensenova.sensenova_pipeline_ops``), and reports the numbers
the Unit-3 measurement gate needs: peak VRAM, prefill time, denoise time,
total wall clock.

Two load modes:
  * ``converted`` (default) -- the sushiUI int8 shard index this repo's own
    loader (``core.models.sensenova.loader.load_sensenova_from_path``) reads.
  * ``bf16-staged`` -- the untouched upstream HF checkpoint tree (13 shards,
    ``model.safetensors.index.json``), for the int8-vs-bf16 A/B (gate arm c).
    Streamed ONE SHARD AT A TIME straight to the target device under
    ``init_empty_weights()`` (``accelerate.set_module_tensor_to_device``) so
    host-RAM peak is one shard (~6.6 GiB max on the real checkpoint), never
    the whole ~46.8 GiB file -- see the repo's host-RAM gate.

VRAM-ceiling probing (``--probe-adaptive``) runs ONE RESOLUTION PER
SUBPROCESS: each arm is a fresh invocation of this same script (via
``sys.executable``, i.e. always the venv interpreter -- see the guard
below), so VRAM is released by process exit rather than relying on
``torch.cuda.empty_cache()`` inside a long-lived process. OOM never raises
on this machine (WDDM spills into shared system memory instead of raising
``torch.cuda.OutOfMemoryError``), so the ladder's PRIMARY stop condition is
each arm's own peak-VRAM-vs-device-total readout (``spilled``, see
``_vram_spill_readout`` -- a spilled allocation still succeeds under WDDM
but is still counted, so this is unambiguous and driver-independent). A
token-count-normalized s/step comparison against the baseline arm is a
backstop only, NOT the primary detector -- a raw s/step comparison trips on
ordinary quadratic attention-cost scaling from one resolution rung to the
next and reports a bogus ceiling. Two independent watchdogs (load-stage,
run-stage) are a third stop condition; on expiry the arm is killed and
reported as the ceiling.

Not collected by pytest (no ``test_`` name); run directly with the repo's
own venv interpreter (never the system Python -- this script refuses to
run under anything else, see ``_assert_venv_interpreter()``). Lives beside
the rest of this architecture's code (``core/models/sensenova/``), matching
the repo's other direct-load smoke drivers (e.g.
``core/models/minimax_music3/smoke.py``) rather than the top-level
``backend/scripts/`` this used to live under -- that tree isn't described in
``ARCHITECTURE_MAP.md``/``DOC_MAP.md`` and had no other member (see the L7
audit note):
    "<repo>/venv/Scripts/python.exe" backend/core/models/sensenova/smoke.py --model-path ...
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_BACKEND, ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)


def _venv_python_path() -> str:
    """Absolute path to the repo's own venv interpreter for this OS."""
    if sys.platform == "win32":
        return os.path.join(_REPO_ROOT, "venv", "Scripts", "python.exe")
    return os.path.join(_REPO_ROOT, "venv", "bin", "python")


def _assert_venv_interpreter() -> None:
    """Refuse to run under anything but the repo venv interpreter.

    This must not import torch or anything heavy -- it has to work (and
    fail loudly) even when invoked with an interpreter that has no CUDA
    build of torch installed at all.
    """
    expected = os.path.normcase(os.path.abspath(_venv_python_path()))
    actual = os.path.normcase(os.path.abspath(sys.executable))
    if actual != expected:
        print(
            "[SenseNova.smoke] REFUSING TO RUN: this script must be invoked with the repo's "
            "venv interpreter, not the current one (see CLAUDE.md/AGENTS.md environment rules).\n"
            f"  current interpreter  : {sys.executable}\n"
            f"  required interpreter : {_venv_python_path()}\n"
            "Re-run as:\n"
            f'  "{_venv_python_path()}" "{os.path.abspath(__file__)}" ...',
            file=sys.stderr,
        )
        sys.exit(1)


def _announce_host_ram(extra_note: str = "") -> None:
    try:
        import psutil

        vm = psutil.virtual_memory()
        print(f"[SenseNova.smoke] host RAM: {vm.available / 2**30:.1f} GiB available / "
              f"{vm.total / 2**30:.1f} GiB total.{(' ' + extra_note) if extra_note else ''}")
    except Exception as exc:
        print(f"[SenseNova.smoke] could not query host RAM via psutil ({exc!r}); proceeding anyway.")


def _load_converted(model_path: str, device, dtype):
    from core.models.sensenova.loader import load_sensenova_from_path

    components = load_sensenova_from_path(model_path, torch_dtype=dtype)
    model = components["transformer"].to(device)
    return model, components["config"], components["tokenizer"]


def _load_bf16_streaming(model_root: str, device, dtype):
    """Stream the untouched upstream HF tree straight to ``device``, one shard
    at a time (see module docstring). Never materializes the full ~46.8 GiB
    file on the host."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors.torch import load_file

    from core.models.sensenova.vendor import NEOChatConfig, NEOChatModel

    with open(os.path.join(model_root, "config.json"), encoding="utf-8") as f:
        cfg_dict = json.load(f)
    config = NEOChatConfig(**cfg_dict)

    with init_empty_weights():
        model = NEOChatModel(config)

    with open(os.path.join(model_root, "model.safetensors.index.json"), encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    shards: dict = {}
    for key, shard in weight_map.items():
        shards.setdefault(shard, []).append(key)

    model_keys = set(dict(model.named_parameters()).keys()) | set(dict(model.named_buffers()).keys())
    seen = set()
    for shard_name in sorted(shards):
        shard_path = os.path.join(model_root, shard_name)
        shard_gib = os.path.getsize(shard_path) / 2**30
        print(f"[SenseNova.smoke] streaming shard {shard_name} ({shard_gib:.2f} GiB) ...")
        shard_sd = load_file(shard_path, device="cpu")
        for key in shards[shard_name]:
            if key not in model_keys:
                continue  # tied weights etc. resolve through the live module without a direct assignment.
            set_module_tensor_to_device(model, key, device, value=shard_sd[key], dtype=dtype)
            seen.add(key)
        del shard_sd
        gc.collect()

    missing = model_keys - seen
    if missing:
        print(f"[SenseNova.smoke] WARNING: {len(missing)} model key(s) never assigned "
              f"(tied weights are expected here); first 5: {sorted(missing)[:5]}")

    model.eval()
    model.requires_grad_(False)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_root)
    return model, config, tokenizer


def _peak_vram_gb() -> float:
    import torch
    return torch.cuda.max_memory_allocated() / 2**30


def _vram_spill_readout() -> dict:
    """Peak allocated bytes vs. this device's total VRAM. On WDDM a spilled
    allocation still succeeds (into shared system memory) and is still
    counted by ``max_memory_allocated`` -- so ``peak_bytes > total_bytes`` is
    an unambiguous, driver-independent spill signal, unlike wall-clock
    scaling (see H2 audit note on ``_run_probe_adaptive``)."""
    import torch
    peak_bytes = torch.cuda.max_memory_allocated()
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    return {
        "peak_vram_bytes": peak_bytes,
        "total_vram_bytes": total_bytes,
        "spilled": peak_bytes > total_bytes,
    }


def run_generation(model, tokenizer, args, width: int, height: int, num_steps: int) -> dict:
    """One full txt2img generation. Returns a dict of measured numbers."""
    import torch

    from core.inference.generation_timing import generation_timer
    from core.models.sensenova import sensenova_pipeline_ops as ops

    device = next(model.parameters()).device
    width, height = ops.normalize_resolution(width, height)

    ops.set_attention_backend(model, args.attn_backend)

    torch.cuda.reset_peak_memory_stats()
    generation_timer.reset()
    wall_start = time.perf_counter()

    def _prefill_note():
        print("[SenseNova.smoke] prefill: building prefix KV cache(s) -- this can take several seconds ...")

    prefix = ops.encode_prompt(
        model, tokenizer, args.prompt, height, width, args.cfg_scale,
        prefill_callback=_prefill_note,
    )

    def _progress(step, total):
        print(f"[SenseNova.smoke] step {step}/{total}")

    x = ops.denoise_loop(
        model, prefix, seed=args.seed, cfg_scale=args.cfg_scale, timestep_shift=args.timestep_shift,
        num_inference_steps=num_steps, progress_callback=_progress,
    )
    wall_total = time.perf_counter() - wall_start

    phases = generation_timer.phases_dict()
    result = {
        "width": width, "height": height, "steps": num_steps,
        # sensenova_pipeline_ops.encode_prompt is now @time_phase("text_encode")
        # (was "prefill", a key _PHASE_KEYS never mapped -- see M5 audit note).
        "prefill_s": phases.get("time_text_encode", 0.0),
        "denoise_s": phases.get("time_denoise", 0.0),
        "wall_s": wall_total,
        "peak_vram_gb": _peak_vram_gb(),
        "image": ops.tensor_to_image(x) if args.output else None,
    }
    result.update(_vram_spill_readout())
    return result


def _mp_to_square_side(mp: float) -> int:
    """Round-trip a target megapixel count to a square side length; the
    child process still snaps it to the model's required multiple via
    ``ops.normalize_resolution``, this only picks the starting guess."""
    side = int(round((mp * 1e6) ** 0.5))
    return max(64, side)


def _query_nvidia_smi_used_mib() -> Optional[int]:
    """Best-effort ``nvidia-smi`` VRAM-used query; ``None`` if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        return int(out.stdout.strip().splitlines()[0].strip())
    except Exception as exc:  # noqa: BLE001 -- purely diagnostic, never fatal
        print(f"[SenseNova.smoke] nvidia-smi query failed ({exc!r}); skipping VRAM readout.")
        return None


def _run_single_arm_subprocess(args, width: int, height: int, load_timeout_s: float,
                                run_timeout_s: float) -> dict:
    """Run exactly one resolution in its own fresh subprocess (venv
    interpreter, proven by ``_assert_venv_interpreter`` in the child), with
    TWO INDEPENDENT wall-clock watchdogs: a load-timeout that covers only
    "process start -> checkpoint resident" (the child touches a marker file
    the instant its ~19 GiB load completes) and a separate run-timeout that
    covers only "checkpoint resident -> generation done". Timing the whole
    child under one watchdog (the earlier version of this function) let a
    slow disk on a big arm masquerade as a VRAM ceiling -- see the M8 audit
    note. VRAM is released by process exit, never by relying on in-process
    ``empty_cache()``."""
    fd, json_path = tempfile.mkstemp(prefix="sensenova_probe_", suffix=".json")
    os.close(fd)
    os.remove(json_path)  # child creates it fresh; absence lets us detect "never wrote"
    loaded_marker_path = json_path + ".loaded"
    if os.path.exists(loaded_marker_path):
        os.remove(loaded_marker_path)

    cmd = [
        _venv_python_path(), os.path.abspath(__file__),
        "--model-path", args.model_path,
        "--load-mode", args.load_mode,
        "--dtype", args.dtype,
        "--attn-backend", args.attn_backend,
        "--width", str(width),
        "--height", str(height),
        "--steps", str(args.probe_steps),
        "--cfg-scale", str(args.cfg_scale),
        "--timestep-shift", str(args.timestep_shift),
        "--seed", str(args.seed),
        "--prompt", args.prompt,
        "--probe-json-out", json_path,
        "--probe-loaded-marker", loaded_marker_path,
    ]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[SenseNova.smoke] PROBE arm: launching subprocess for {width}x{height} "
          f"(load_timeout={load_timeout_s:.0f}s, run_timeout={run_timeout_s:.0f}s) ...")

    status = "pass"
    poll_interval_s = 1.0
    proc = subprocess.Popen(cmd, env=env)
    t_start = time.perf_counter()
    marker_seen_at: Optional[float] = None
    try:
        while True:
            ret = proc.poll()
            if ret is not None:
                status = "pass" if ret == 0 else "error"
                break
            now = time.perf_counter()
            if marker_seen_at is None:
                if os.path.exists(loaded_marker_path):
                    marker_seen_at = now
                    print(f"[SenseNova.smoke] PROBE arm {width}x{height}: checkpoint resident after "
                          f"{now - t_start:.1f}s -- switching to run-timeout watchdog.")
                elif now - t_start > load_timeout_s:
                    status = "timeout_load"
                    print(f"[SenseNova.smoke] PROBE arm {width}x{height}: LOAD TIMEOUT after "
                          f"{load_timeout_s:.0f}s (checkpoint never became resident) -- killing.")
                    proc.kill()
                    proc.wait()
                    break
            else:
                if now - marker_seen_at > run_timeout_s:
                    status = "timeout_run"
                    print(f"[SenseNova.smoke] PROBE arm {width}x{height}: RUN TIMEOUT after "
                          f"{run_timeout_s:.0f}s post-load -- killing, treating as ceiling.")
                    proc.kill()
                    proc.wait()
                    break
            time.sleep(poll_interval_s)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    vram_used_mib = _query_nvidia_smi_used_mib()
    if vram_used_mib is not None:
        print(f"[SenseNova.smoke] PROBE arm {width}x{height}: nvidia-smi memory.used="
              f"{vram_used_mib} MiB after subprocess exit.")

    result = None
    if status == "pass" and os.path.exists(json_path):
        try:
            with open(json_path, encoding="utf-8") as f:
                result = json.load(f)
        except Exception as exc:  # noqa: BLE001
            print(f"[SenseNova.smoke] PROBE arm {width}x{height}: could not read child result ({exc!r}).")
            status = "error"
    elif status == "pass":
        status = "error"
        print(f"[SenseNova.smoke] PROBE arm {width}x{height}: child exited 0 but wrote no result file.")

    for p in (json_path, loaded_marker_path):
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    # "timeout" (bare) is kept as the status the ceiling-detection caller
    # checks for either watchdog firing -- the distinction between
    # timeout_load/timeout_run is preserved for the printed diagnostic only.
    reported_status = "timeout" if status in ("timeout_load", "timeout_run") else status
    return {"status": reported_status, "raw_status": status, "result": result, "vram_used_mib": vram_used_mib}


_TOKEN_GRID_ALIGN = 32  # duplicated from sensenova_pipeline_ops.TOKEN_GRID_ALIGN -- this script
                        # must not import torch-touching modules at parse time (_assert_venv_interpreter
                        # runs before any heavy import), so the constant is a bare literal here too.


def _normalized_s_per_step(s_per_step: float, width: int, height: int) -> float:
    """Normalize per-step time by token-count SQUARED (attention is quadratic
    in token count; the ViT's linear term is subdominant at the resolutions
    this ladder explores), so this metric is roughly resolution-INVARIANT
    under ordinary compute scaling. A raw ``s_per_step`` comparison (the
    earlier version of this function) trips on every rung purely from 4MP ->
    8MP being ~4x the attention cost -- that is not a VRAM ceiling, it is
    the model doing more work. See the H2 audit note."""
    tokens = max(1.0, (width / _TOKEN_GRID_ALIGN) * (height / _TOKEN_GRID_ALIGN))
    return s_per_step / (tokens ** 2)


def _run_probe_adaptive(args) -> None:
    """Adaptive VRAM-ceiling ladder: start at a known-good baseline (~4 MP
    square) and step upward by ``--probe-step-mp`` per arm, one subprocess
    per arm.

    PRIMARY stop condition: the child's own peak-VRAM-vs-device-total
    readout (``spilled`` in its result, see ``_vram_spill_readout``). On
    WDDM a spilled allocation still succeeds (into shared system memory)
    rather than raising ``torch.cuda.OutOfMemoryError``, but it still shows
    up in ``torch.cuda.max_memory_allocated()`` -- so this is an unambiguous,
    driver-independent ceiling signal.

    BACKSTOP: a token-count-NORMALIZED s/step comparison against the
    baseline arm (``_normalized_s_per_step``) -- catches an anomaly (e.g.
    thermal/driver throttling) that isn't a straightforward "peak exceeded
    total" spill. A RAW s/step comparison (the earlier version of this
    function) is not used for detection any more: it triggers on ordinary
    quadratic attention-cost scaling from one rung to the next and reports a
    bogus ceiling that is actually just compute, not memory -- see the H2
    audit note.

    A per-arm timeout (load- and run-timeouts are independent, see
    ``_run_single_arm_subprocess``) is a third, independent stop condition.
    """
    baseline_normalized: Optional[float] = None
    last_good = None
    mp = args.probe_start_mp
    table = []

    while mp <= args.probe_max_mp:
        side = _mp_to_square_side(mp)
        arm = _run_single_arm_subprocess(args, side, side, args.probe_load_timeout_s, args.probe_timeout_s)

        if arm["status"] == "timeout":
            print(f"[SenseNova.smoke] PROBE CEILING at ~{mp:.2f} MP ({side}x{side}): "
                  f"arm timed out ({arm.get('raw_status', 'timeout')}, never completed) -- stopping sweep.")
            table.append({"mp": mp, "width": side, "height": side, "status": arm.get("raw_status", "timeout")})
            break

        if arm["status"] != "pass" or arm["result"] is None:
            print(f"[SenseNova.smoke] PROBE CEILING at ~{mp:.2f} MP ({side}x{side}): "
                  f"arm failed (non-zero exit / no result) -- stopping sweep.")
            table.append({"mp": mp, "width": side, "height": side, "status": "error"})
            break

        r = arm["result"]
        s_per_step = r["denoise_s"] / max(1, r["steps"])
        actual_mp = (r["width"] * r["height"]) / 1e6
        normalized = _normalized_s_per_step(s_per_step, r["width"], r["height"])
        spilled = bool(r.get("spilled", False))
        row = {
            "mp": actual_mp, "width": r["width"], "height": r["height"],
            "status": "pass", "s_per_step": s_per_step, "normalized_s_per_step": normalized,
            "peak_vram_gb": r["peak_vram_gb"], "total_vram_gb": r.get("total_vram_bytes", 0) / 2**30,
            "spilled": spilled, "vram_used_mib_after_exit": arm["vram_used_mib"],
        }
        table.append(row)
        print(f"[SenseNova.smoke] PROBE {r['width']}x{r['height']} ({actual_mp:.2f} MP): PASS "
              f"peak_vram={r['peak_vram_gb']:.2f}GiB/{row['total_vram_gb']:.2f}GiB "
              f"spilled={spilled} s/step={s_per_step:.3f}s prefill={r['prefill_s']:.1f}s "
              f"denoise={r['denoise_s']:.1f}s")

        if spilled:
            print(f"[SenseNova.smoke] PROBE CEILING: {actual_mp:.2f} MP spilled "
                  f"(peak {r['peak_vram_gb']:.2f}GiB > device total {row['total_vram_gb']:.2f}GiB) "
                  f"-- stopping sweep. Last known-good: "
                  f"{last_good['width']}x{last_good['height']} ({last_good['mp']:.2f} MP)."
                  if last_good is not None else
                  f"[SenseNova.smoke] PROBE CEILING: {actual_mp:.2f} MP spilled on the FIRST arm "
                  f"-- baseline itself does not fit.")
            break

        if baseline_normalized is None:
            baseline_normalized = normalized
            print(f"[SenseNova.smoke] PROBE baseline established: {s_per_step:.3f}s/step "
                  f"at {actual_mp:.2f} MP (normalized={normalized:.3e}, backstop cliff = "
                  f"{args.probe_cliff_multiplier:.1f}x normalized).")
            last_good = table[-1]
        elif normalized > baseline_normalized * args.probe_cliff_multiplier:
            print(f"[SenseNova.smoke] PROBE CEILING (backstop): {actual_mp:.2f} MP is "
                  f"{normalized / baseline_normalized:.1f}x baseline's token-normalized s/step "
                  f"(threshold {args.probe_cliff_multiplier:.1f}x) -- not explained by ordinary "
                  f"quadratic scaling, stopping sweep. Last known-good: "
                  f"{last_good['width']}x{last_good['height']} ({last_good['mp']:.2f} MP).")
            break
        else:
            last_good = table[-1]

        mp += args.probe_step_mp

    print("[SenseNova.smoke] PROBE table:")
    for row in table:
        print(f"  {row}")
    if last_good is not None:
        print(f"[SenseNova.smoke] PROBE ceiling summary: last known-good arm = "
              f"{last_good['width']}x{last_good['height']} ({last_good['mp']:.2f} MP, "
              f"{last_good['s_per_step']:.3f}s/step, peak_vram={last_good['peak_vram_gb']:.2f}GiB).")


def main(argv=None) -> int:
    _assert_venv_interpreter()

    # steps/cfg-scale/timestep-shift defaults are sourced from param_defaults.py
    # (AGENTS.md: never hardcode a default anywhere else) -- this is a plain-dict
    # import, no torch/CUDA touched, safe to do after _assert_venv_interpreter().
    from api.param_defaults import SENSENOVA_GENERATION_DEFAULTS

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-path", required=True,
                        help="'converted': path to the sushiUI shard index (*.safetensors.index.json). "
                             "'bf16-staged': path to the upstream HF checkpoint directory.")
    parser.add_argument("--load-mode", choices=["converted", "bf16-staged"], default="converted")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                        help="Compute/activation dtype -- bf16 is the production path (int8 dequants into it).")
    parser.add_argument("--attn-backend", default="native",
                        help="Conduit backend name: native (SDPA) / flash / sage / tq.")
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=SENSENOVA_GENERATION_DEFAULTS["steps"])
    parser.add_argument("--cfg-scale", type=float, default=SENSENOVA_GENERATION_DEFAULTS["cfg_scale"])
    parser.add_argument("--timestep-shift", type=float, default=SENSENOVA_GENERATION_DEFAULTS["timestep_shift"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="A photo of a red panda eating a bamboo leaf, studio lighting.")
    parser.add_argument("--output", default=None, help="PNG path to save the result. Omit to skip saving.")
    parser.add_argument("--probe-adaptive", action="store_true",
                        help="VRAM-ceiling probe: adaptive ladder starting at --probe-start-mp, stepping by "
                             "--probe-step-mp, one subprocess per arm, stopping at the first confirmed VRAM "
                             "spill (primary) or timeout. Replaces the old pre-enumerated --probe-resolutions "
                             "list.")
    parser.add_argument("--probe-start-mp", type=float, default=4.0,
                        help="Starting (known-good) megapixel count for --probe-adaptive.")
    parser.add_argument("--probe-step-mp", type=float, default=1.0,
                        help="Megapixel increment per arm for --probe-adaptive.")
    parser.add_argument("--probe-max-mp", type=float, default=20.0,
                        help="Hard safety cap on megapixels for --probe-adaptive, in case neither stop "
                             "condition ever trips.")
    parser.add_argument("--probe-timeout-s", type=float, default=600.0,
                        help="Per-arm RUN-stage watchdog timeout (seconds, post-checkpoint-load) for "
                             "--probe-adaptive. On expiry the arm is killed and reported as the ceiling.")
    parser.add_argument("--probe-load-timeout-s", type=float, default=300.0,
                        help="Per-arm LOAD-stage watchdog timeout (seconds, checkpoint read from disk), "
                             "independent of --probe-timeout-s -- a slow disk must not read as a VRAM "
                             "ceiling (see M8 audit note).")
    parser.add_argument("--probe-cliff-multiplier", type=float, default=3.0,
                        help="Backstop only (primary ceiling detection is the child's own VRAM-spill "
                             "readout): an arm is additionally declared the ceiling when its token-count-"
                             "normalized s/step exceeds this multiple of the baseline arm's normalized "
                             "s/step.")
    parser.add_argument("--probe-steps", type=int, default=4,
                        help="Step count for each --probe-adaptive arm (peak VRAM does not grow with step "
                             "count in this arch -- the KV cache is built once in the prefix -- so a short "
                             "run suffices).")
    parser.add_argument("--probe-json-out", default=None,
                        help=argparse.SUPPRESS)  # internal: child-process result handoff for --probe-adaptive
    parser.add_argument("--probe-loaded-marker", default=None,
                        help=argparse.SUPPRESS)  # internal: "checkpoint resident" marker for --probe-adaptive
    args = parser.parse_args(argv)

    # Force unbuffered/line-buffered stdout so orchestrator + child logs interleave live,
    # regardless of whether the caller remembered `-u` / PYTHONUNBUFFERED.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    if args.probe_adaptive:
        # Orchestrator only: never loads a model itself, so a stuck arm can only ever
        # hold VRAM for as long as its own subprocess (bounded by --probe-timeout-s).
        _announce_host_ram("Adaptive VRAM-ceiling probe: orchestrator process, no model load here.")
        _run_probe_adaptive(args)
        print("[SenseNova.smoke] DONE")
        return 0

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    if device.type != "cuda":
        print("[SenseNova.smoke] WARNING: no CUDA device found; this will be extremely slow / may not fit in RAM.")

    _announce_host_ram(
        "Loading a ~19 GiB int8 SenseNova checkpoint." if args.load_mode == "converted"
        else "Streaming the ~46.8 GiB bf16 staged checkpoint ONE SHARD AT A TIME (peak ~6.6 GiB/shard).")

    load_start = time.perf_counter()
    if args.load_mode == "converted":
        model, config, tokenizer = _load_converted(args.model_path, device, dtype)
    else:
        model, config, tokenizer = _load_bf16_streaming(args.model_path, device, dtype)
    print(f"[SenseNova.smoke] load: {time.perf_counter() - load_start:.1f}s")

    if args.probe_loaded_marker:
        # Signals the orchestrator's load-stage watchdog to stop counting and
        # switch to the run-stage watchdog -- see M8 audit note.
        with open(args.probe_loaded_marker, "w", encoding="utf-8") as f:
            f.write("1")

    try:
        result = run_generation(model, tokenizer, args, args.width, args.height, args.steps)
        print(f"[SenseNova.smoke] RESULT {result['width']}x{result['height']} steps={result['steps']}: "
              f"prefill={result['prefill_s']:.2f}s denoise={result['denoise_s']:.2f}s "
              f"wall={result['wall_s']:.2f}s peak_vram={result['peak_vram_gb']:.2f}GiB/"
              f"{result.get('total_vram_bytes', 0) / 2**30:.2f}GiB spilled={result.get('spilled')} "
              f"s/step={result['denoise_s'] / max(1, result['steps']):.3f}")
        if args.output and result["image"] is not None:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
            result["image"].save(args.output)
            print(f"[SenseNova.smoke] saved: {args.output}")
        if args.probe_json_out:
            # Single-arm (child-process) mode for --probe-adaptive: hand the measured
            # numbers back to the orchestrator via a file, not stdout parsing.
            json_result = {k: v for k, v in result.items() if k != "image"}
            with open(args.probe_json_out, "w", encoding="utf-8") as f:
                json.dump(json_result, f)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[SenseNova.smoke] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
