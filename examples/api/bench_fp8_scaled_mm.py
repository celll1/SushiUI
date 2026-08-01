"""Measurement gate G1 for the FP8 W8A8 scaled-GEMM fast path (Fp8Linear).

The fast path (``backend/core/models/ideogram4/vendor/fp8_linear.py``,
``_scaled_mm_forward``) is OPT-IN behind ``SUSHI_FP8_SCALED_MM=1`` and stays off
until this gate passes.

===========================================================================
PRE-REGISTERED DECISION RULE  --  written down BEFORE any number existed
===========================================================================
This block is an anti-rationalization device. It was committed to the repo
before the first timing was taken; ``--report`` evaluates the measurements
against it and prints which branch applies. Numbers are compared against this
rule, never the rule against the numbers. Do not edit the thresholds after a
measurement exists.

FLIP THE DEFAULT (and proceed to Phase 2) requires ALL of:
  1. Krea 2 ``fp8_fast`` >= 1.10x the steps/s of Krea 2 ``bf16``
     (median of >= 3 timed runs, 1 warmup, fixed prompt/seed/shape, >= 20 steps).
  2. Ideogram 4 ``fp8_fast`` >= 1.00x the steps/s of Ideogram 4 ``fp8_dequant``
     (a no-regression check on the arch that ships FP8 today).
  3. BOTH quality A/Bs clean (human judgement, images saved by ``--quality``).

IF KREA 2 LANDS IN 1.00x - 1.10x:
  Keep the code path, but reframe it as "removes the dequantization step for
  models already stored in FP8". Make NO speed claim anywhere -- not in the UI,
  not in a docstring, not in a commit message. Do NOT proceed to Phase 2 and do
  NOT generalize it to the runtime ``unet_quantization`` enum: Phase 2's whole
  value proposition was speed, and this branch means there is none to sell.

IF KREA 2 IS BELOW 1.00x:
  Revert the fast path.

RECORDED EXPLICITLY: "beats the dequant path" is a valid reason to KEEP code for
checkpoints that are already FP8 on disk. It is NEVER on its own a reason to
flip a default or to widen the surface (the enum, new archs, new call sites).
Only criterion 1 above can do that.

===========================================================================
TWO VEHICLES
===========================================================================
``krea2``  -- carries the SPEED gate. Krea 2 ships bf16 locally, is a single
    transformer that fits VRAM, and bf16 is its shipping production
    configuration today, so ``fp8_fast`` vs ``bf16`` measures the GEMM and not
    the memory system. The matched FP8 checkpoint is produced by
    ``subapps/fp8_quantize/quantize_transformer_fp8.py`` using the repo's own
    ``quantize_weight_to_fp8``, so the two arms differ only in weight format and
    GEMM path.

``ideogram4`` -- carries the REGRESSION + QUALITY arm: ``fp8_fast`` vs
    ``fp8_dequant``, both on the SAME shipped FP8 checkpoint. It does NOT carry
    the >= 1.10x-vs-bf16 claim. A dequantized-bf16 Ideogram 4 arm is invalid and
    must not be built: Ideogram 4 keeps BOTH a conditional and an unconditional
    transformer resident (asymmetric CFG), so bf16 would be 2 x ~18.6 GB plus
    the ~17 GB text encoder, would not co-reside in the available VRAM, and
    would pay offload traffic that neither FP8 arm pays -- the ratio would
    measure memory traffic, not the GEMM.

===========================================================================
WHAT IS TIMED
===========================================================================
SAMPLER STEPS, not wall clock including model load. The script subscribes to
``/api/v1/ws/progress`` (see ``backend/api/WS_PROTOCOL.md``) and timestamps each
``progress`` message client-side, then reports
``(last_step - first_step) / (t_last - t_first)``. That window excludes model
staging before the first step and VAE decode / PNG save / DB write after the
last one. End-to-end HTTP time is recorded alongside for reference only.

(``metrics_db`` timestamps are known-unusable for timing, which is why this
instruments the progress channel directly.) If the WebSocket yields no usable
progress messages the run still completes, but it is recorded with
``timing_source = "http_end_to_end"``; a ratio computed from those numbers is
CONSERVATIVE -- the fixed per-generation overhead is identical across arms, so
it understates a real speedup rather than inventing one.

The progress channel is a global unfiltered broadcast: nothing else may be
generating or training on this backend while a timed arm runs, and the GPU must
otherwise be idle.

===========================================================================
HELD FIXED ACROSS ARMS
===========================================================================
Resolution, steps, sampler, schedule, seed, prompt, ``keep_models_hot=false``,
attention type/impl, and the text-encoder configuration (each vehicle's arms use
the same TE directory -- the FP8 Krea 2 checkpoint resolves the SAME
``text_encoder``/``vae`` as the bf16 one through sibling links). Only the
transformer's weight format and the GEMM path differ.

``SUSHI_FP8_FAST_ACCUM`` must be at its SHIPPING default (1) for every arm,
above all the quality A/B: judging quality in a non-shipping accumulation mode
measures something nobody runs. The script aborts if it sees the variable set to
"0" in its own environment.

===========================================================================
PROTOCOL (the backend env is fixed at process start)
===========================================================================
``SUSHI_FP8_SCALED_MM`` is read at import time in ``fp8_linear``, so an arm is
fixed for the lifetime of a backend process, and ``POST /system/restart-backend``
cannot inject it (it passes no ``env=``). The repo owner must start the backend
with the value each arm needs; agents must not start/stop servers (AGENTS.md).

    # 0. record the GPU's scaled-GEMM capability mode FIRST -- a gate result is
    #    meaningless without knowing which mode was measured (rowwise vs
    #    tensorwise). No backend needed.
    venv/Scripts/python.exe examples/api/bench_fp8_scaled_mm.py --probe

    # backend WITHOUT SUSHI_FP8_SCALED_MM (or =0)
    # arm 1 -- krea2 bf16 baseline
    venv/Scripts/python.exe examples/api/bench_fp8_scaled_mm.py \
        --vehicle krea2 --arm bf16 \
        --source-type safetensors --source <bf16 krea2 index/file> --no-dry-run
    # arm 2 -- ideogram4 fp8, dequant path (today's default)
    venv/Scripts/python.exe examples/api/bench_fp8_scaled_mm.py \
        --vehicle ideogram4 --arm fp8_dequant \
        --source-type diffusers --source <ideogram4 fp8 dir> --no-dry-run

    # backend restarted WITH SUSHI_FP8_SCALED_MM=1
    # arm 3 -- krea2 fp8, scaled-GEMM fast path
    venv/Scripts/python.exe examples/api/bench_fp8_scaled_mm.py \
        --vehicle krea2 --arm fp8_fast \
        --source-type safetensors --source <fp8 krea2 index> --no-dry-run
    # arm 4 -- ideogram4 fp8, scaled-GEMM fast path
    venv/Scripts/python.exe examples/api/bench_fp8_scaled_mm.py \
        --vehicle ideogram4 --arm fp8_fast \
        --source-type diffusers --source <ideogram4 fp8 dir> --no-dry-run

    # quality A/B: 4 prompts x 2 seeds per arch, per arm (8 images each), run in
    # the backend session that matches the arm
    venv/Scripts/python.exe examples/api/bench_fp8_scaled_mm.py \
        --vehicle krea2 --arm fp8_fast --quality \
        --source-type safetensors --source <fp8 krea2 index> --no-dry-run

    # evaluate against the pre-registered rule
    venv/Scripts/python.exe examples/api/bench_fp8_scaled_mm.py --report

Results accumulate in ``tmp/fp8_bench_results.json`` keyed ``<vehicle>:<arm>``;
images land in ``tmp/fp8_bench_images/<vehicle>/<arm>/``.
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PORT_INFO = os.path.join(REPO_ROOT, "backend", ".port_info")
OUT_DIR = os.path.join(REPO_ROOT, "tmp")
RESULTS = os.path.join(OUT_DIR, "fp8_bench_results.json")
IMAGE_DIR = os.path.join(OUT_DIR, "fp8_bench_images")

# ---------------------------------------------------------------------------
# Fixed measurement configuration (identical work in every arm)
# ---------------------------------------------------------------------------

PROMPT = "a photograph of a stone bridge over a river, morning fog"
NEGATIVE = ""
SEED = 12345
WIDTH = 1024
HEIGHT = 1024
STEPS = 24          # >= 20 required by the rule
RUNS = 3            # >= 3 timed runs, median reported
WARMUP = 1

# Quality A/B: 4 prompts x 2 seeds = 8 pairs per architecture, per arm.
# The set is chosen for what FP8 breaks first, not for prettiness:
#   [0] text rendering -- Ideogram 4's specialty and the most quantization-
#       fragile thing either model does (glyph shape is high-frequency and
#       unforgiving; a wrong letter is unambiguous, unlike "slightly different").
#   [1] flat colour + smooth gradient -- banding / mottle sensitivity; W8A8
#       error shows up in low-texture regions where nothing masks it.
#   [2] skin and soft shading -- mid-frequency, where subtle desaturation or
#       plasticity would appear.
#   [3] dense high-frequency clutter -- worst case for accumulated per-token
#       activation quantization error.
QUALITY_PROMPTS = [
    'a storefront window with a hand-painted sign reading "SUSHI UI - OPEN DAILY 11:30", '
    "gold leaf lettering on glass, afternoon light",
    "a minimal flat poster: a smooth vertical gradient from deep indigo to pale peach, "
    "one large off-white circle centred, no texture, no grain",
    "close-up portrait of an elderly fisherman, weathered skin, overcast daylight, 85mm lens",
    "overhead photo of a workbench covered in electronic components, resistors, ribbon cable, "
    "fine printed markings on the chips",
]
QUALITY_SEEDS = [12345, 987654321]

PASS_RATIO_KREA2 = 1.10        # criterion 1: krea2 fp8_fast vs krea2 bf16
PASS_RATIO_IDEOGRAM4 = 1.00    # criterion 2: ideogram4 fp8_fast vs fp8_dequant

VEHICLES = {
    # cfg_scale is per-vehicle because the CFG convention differs per arch
    # (Krea 2 maps UI cfg_scale to guidance = cfg_scale - 1; Ideogram 4 runs an
    # asymmetric two-transformer CFG). Ratios are only ever taken WITHIN a
    # vehicle, so a per-vehicle value costs nothing and keeps each arch in a
    # sane operating point.
    "krea2": {"cfg_scale": 4.0},
    "ideogram4": {"cfg_scale": 4.0},
}
ARMS = ("bf16", "fp8_dequant", "fp8_fast")

# The four records the decision rule needs.
REQUIRED = ("krea2:bf16", "krea2:fp8_fast", "ideogram4:fp8_dequant", "ideogram4:fp8_fast")

# Must match ModelSource in backend/core/model_loader.py; ModelLoader.load_model
# raises ValueError (HTTP 500) on anything else.
SOURCE_TYPES = ("safetensors", "diffusers", "huggingface")


def base_url():
    """Resolve the backend's host/port from the file it writes on startup."""
    with open(PORT_INFO, encoding="utf-8") as fh:
        info = json.load(fh)
    return f"http://{info['host']}:{info['port']}"


def ws_url():
    with open(PORT_INFO, encoding="utf-8") as fh:
        info = json.load(fh)
    return f"ws://{info['host']}:{info['port']}/api/v1/ws/progress"


def load_results():
    if not os.path.exists(RESULTS):
        return {}
    with open(RESULTS, encoding="utf-8") as fh:
        return json.load(fh)


def store_result(key, record):
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_results()
    data[key] = record
    with open(RESULTS, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


# ---------------------------------------------------------------------------
# B. Capability probe -- run FIRST, recorded with the results
# ---------------------------------------------------------------------------

def probe():
    """Record which ``torch._scaled_mm`` scaling mode this GPU/torch accepts.

    A gate result without this is uninterpretable: the rowwise and tensorwise
    epilogues are different amounts of work and different numerics. Runs in this
    process (16x16 operands -- no meaningful VRAM or time) with the fast path
    force-enabled, independent of how the backend was launched.
    """
    os.environ["SUSHI_FP8_SCALED_MM"] = "1"
    sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))
    import torch
    from core.models.ideogram4.vendor.fp8_linear import _USE_FAST_ACCUM, _probe_scaled_mm

    if not torch.cuda.is_available():
        print("no CUDA device; cannot probe")
        return 1
    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    record = {
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "fast_accum_default": bool(_USE_FAST_ACCUM),
        "modes": {
            str(dt).rsplit(".", 1)[-1]: _probe_scaled_mm(device, dt)
            for dt in (torch.bfloat16, torch.float16, torch.float32)
        },
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    store_result("_probe", record)
    print(json.dumps(record, indent=2))
    print(f"\nrecorded in {RESULTS} under '_probe'")
    return 0


# ---------------------------------------------------------------------------
# Per-step timing via the progress WebSocket
# ---------------------------------------------------------------------------

class StepTimer:
    """Collect (step, receive time) from ``/ws/progress`` in a background thread.

    Client-side receive timestamps over loopback. The quantity reported is the
    span between the FIRST and LAST progress message of a run divided by the
    number of steps between them, so per-run one-off costs on either side of the
    denoise loop are excluded from the rate.
    """

    def __init__(self):
        self.samples = []           # (step, total_steps, perf_counter)
        self.error = None
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        time.sleep(0.5)             # let the connection register before the POST
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self):
        try:
            from websockets.sync.client import connect
        except Exception as exc:                       # pragma: no cover
            self.error = f"websockets client unavailable: {exc}"
            return
        try:
            # max_size=None: a progress message can carry a base64 preview JPEG
            # larger than the 1 MiB default frame limit.
            with connect(ws_url(), max_size=None, open_timeout=10) as ws:
                while not self._stop.is_set():
                    try:
                        raw = ws.recv(timeout=1.0)
                    except TimeoutError:
                        continue
                    now = time.perf_counter()
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    if msg.get("type") != "progress":
                        continue
                    step, total = msg.get("step"), msg.get("total_steps")
                    if isinstance(step, int) and isinstance(total, int):
                        self.samples.append((step, total, now))
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def take(self, expected_steps):
        """Consume the samples for one run -> (steps_counted, span_s) or None."""
        samples, self.samples = self.samples, []
        samples = [s for s in samples if s[1] == expected_steps]
        if len(samples) < 2:
            return None
        first, last = samples[0], samples[-1]
        steps = last[0] - first[0]
        span = last[2] - first[2]
        if steps <= 0 or span <= 0:
            return None
        return steps, span


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def load_model(root, source_type, source):
    """POST /models/load. multipart form, like every other route here."""
    resp = requests.post(
        f"{root}/api/v1/models/load",
        data={"source_type": source_type, "source": source},
        timeout=3600,
    )
    resp.raise_for_status()
    return resp.json()


def form_data(vehicle, prompt=PROMPT, seed=SEED, steps=STEPS):
    """/generate/txt2img is a Form(...) route, not JSON (see txt2img_minimal.py).

    Everything that could differ between two backend sessions is pinned here
    rather than left to the endpoint's defaults.
    """
    return {
        "prompt": prompt,
        "negative_prompt": NEGATIVE,
        "steps": steps,
        "cfg_scale": VEHICLES[vehicle]["cfg_scale"],
        "sampler": "euler",
        "schedule_type": "uniform",
        "seed": seed,
        "width": WIDTH,
        "height": HEIGHT,
        "keep_models_hot": "false",
        "attention_type": "normal",
        "attention_impl": "conduit",
    }


def txt2img(root, data):
    t0 = time.perf_counter()
    resp = requests.post(f"{root}/api/v1/generate/txt2img", data=data, timeout=7200)
    resp.raise_for_status()
    return time.perf_counter() - t0, resp.json()


def save_image(root, body, vehicle, arm, tag):
    """Download the generated PNG so a human A/B is possible after the fact."""
    image = body.get("image") or {}
    filename = image.get("filename") or body.get("filename")
    if not filename:
        print(f"    (no filename in response; nothing to save for {tag})")
        return None
    dest_dir = os.path.join(IMAGE_DIR, vehicle, arm)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{tag}_{os.path.basename(filename)}")
    resp = requests.get(f"{root}/outputs/{filename}", timeout=600)
    resp.raise_for_status()
    with open(dest, "wb") as fh:
        fh.write(resp.content)
    return dest


def check_fast_accum():
    """Refuse to measure in a non-shipping accumulation mode.

    Only this process's environment is visible, but an operator who exported
    ``SUSHI_FP8_FAST_ACCUM=0`` for the backend almost certainly has it exported
    here too, and a false negative is cheap next to a quality verdict taken in a
    mode nobody ships.
    """
    if os.environ.get("SUSHI_FP8_FAST_ACCUM") == "0":
        print("ABORT: SUSHI_FP8_FAST_ACCUM=0 is set in this shell. Every arm of "
              "this gate -- above all the quality A/B -- must run with the "
              "SHIPPING default (1). Unset it here and in the backend's launch "
              "environment, restart the backend, and re-run.")
        raise SystemExit(2)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def run_timed(vehicle, arm, source_type, source):
    check_fast_accum()
    root = base_url()
    print(f"vehicle={vehicle} arm={arm} source={source}")
    print(f"  {STEPS} steps  {WIDTH}x{HEIGHT}  seed={SEED}  {WARMUP} warmup + {RUNS} timed")
    print(f"  env in THIS shell: SUSHI_FP8_SCALED_MM="
          f"{os.environ.get('SUSHI_FP8_SCALED_MM', '(unset)')} "
          f"-- the BACKEND's value is what selects the arm, not this one")

    info = load_model(root, source_type, source)
    print(f"  model loaded: {info.get('model_info')}")

    data = form_data(vehicle)
    times, step_rates, images = [], [], []
    timing_source = "ws_progress_steps"

    with StepTimer() as timer:
        for i in range(WARMUP):
            elapsed, body = txt2img(root, data)
            timer.take(STEPS)
            print(f"  warmup {i}: {elapsed:.2f}s  warnings={body.get('warnings')}")
        if timer.error:
            print(f"  WARNING: progress WebSocket unusable ({timer.error})")

        for i in range(RUNS):
            elapsed, body = txt2img(root, data)
            measured = timer.take(STEPS)
            times.append(elapsed)
            if measured is not None:
                steps, span = measured
                step_rates.append(steps / span)
                print(f"  run {i}: sampler {steps} steps in {span:.2f}s -> "
                      f"{steps / span:.3f} steps/s   (end-to-end {elapsed:.2f}s)")
            else:
                print(f"  run {i}: no usable progress samples; end-to-end {elapsed:.2f}s")
            path = save_image(root, body, vehicle, arm, f"run{i}")
            if path:
                images.append(path)

    if len(step_rates) == RUNS:
        steps_per_s = statistics.median(step_rates)
    else:
        timing_source = "http_end_to_end"
        steps_per_s = STEPS / statistics.median(times)
        print("  NOTE: falling back to end-to-end HTTP time. That window includes "
              "VAE decode, PNG save and the DB write, which are identical across "
              "arms, so the resulting ratio is CONSERVATIVE (it understates a "
              "real speedup rather than inventing one).")

    record = {
        "vehicle": vehicle,
        "arm": arm,
        "source_type": source_type,
        "source": source,
        "model_info": info.get("model_info"),
        "steps": STEPS,
        "width": WIDTH,
        "height": HEIGHT,
        "seed": SEED,
        "prompt": PROMPT,
        "timing_source": timing_source,
        "step_rates": step_rates,
        "http_times_s": times,
        "median_http_s": statistics.median(times),
        "steps_per_s": steps_per_s,
        "fast_accum_env": os.environ.get("SUSHI_FP8_FAST_ACCUM", "(unset -> shipping default 1)"),
        "images": images,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    store_result(f"{vehicle}:{arm}", record)
    print(f"  MEDIAN {steps_per_s:.3f} steps/s  [{timing_source}]  (saved to {RESULTS})")


def run_quality(vehicle, arm, source_type, source, steps):
    """4 prompts x 2 seeds -> 8 images for this arm, for a human A/B."""
    check_fast_accum()
    root = base_url()
    print(f"quality A/B: vehicle={vehicle} arm={arm} "
          f"{len(QUALITY_PROMPTS)} prompts x {len(QUALITY_SEEDS)} seeds, {steps} steps")
    info = load_model(root, source_type, source)
    print(f"  model loaded: {info.get('model_info')}")

    saved = []
    for pi, prompt in enumerate(QUALITY_PROMPTS):
        for si, seed in enumerate(QUALITY_SEEDS):
            data = form_data(vehicle, prompt=prompt, seed=seed, steps=steps)
            elapsed, body = txt2img(root, data)
            path = save_image(root, body, vehicle, arm, f"q{pi}_s{si}")
            saved.append(path)
            print(f"  p{pi} seed={seed}: {elapsed:.2f}s -> {path}")

    key = f"{vehicle}:{arm}:quality"
    store_result(key, {
        "vehicle": vehicle,
        "arm": arm,
        "source_type": source_type,
        "source": source,
        "steps": steps,
        "prompts": QUALITY_PROMPTS,
        "seeds": QUALITY_SEEDS,
        "images": saved,
        "fast_accum_env": os.environ.get("SUSHI_FP8_FAST_ACCUM", "(unset -> shipping default 1)"),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    print(f"  {len([p for p in saved if p])} image(s) saved; recorded under '{key}'")


# ---------------------------------------------------------------------------
# Report -- evaluated against the pre-registered rule
# ---------------------------------------------------------------------------

def report():
    data = load_results()
    if not data:
        print(f"no results yet: {RESULTS} does not exist")
        return 1

    probe_record = data.get("_probe")
    print("=== capability probe ===")
    if not probe_record:
        print("MISSING. Run --probe. Without the selected scaling mode "
              "(rowwise vs tensorwise) the numbers below are uninterpretable.")
    else:
        print(f"  {probe_record['gpu']} (sm_{probe_record['compute_capability'].replace('.', '')}), "
              f"torch {probe_record['torch']} / cuda {probe_record['cuda']}")
        print(f"  scaled_mm mode: {probe_record['modes']}")
        print(f"  fast accum default: {probe_record['fast_accum_default']}")

    missing = [k for k in REQUIRED if k not in data]
    if missing:
        print(f"\nmissing arms: {missing}")
        return 1

    print("\n=== arms ===")
    for key in REQUIRED:
        r = data[key]
        print(f"{key:24s} {r['steps_per_s']:.3f} steps/s  [{r['timing_source']}]  "
              f"source={r['source']}")

    krea_ratio = data["krea2:fp8_fast"]["steps_per_s"] / data["krea2:bf16"]["steps_per_s"]
    ideo_ratio = (data["ideogram4:fp8_fast"]["steps_per_s"]
                  / data["ideogram4:fp8_dequant"]["steps_per_s"])

    print("\n=== against the PRE-REGISTERED rule ===")
    print(f"  krea2     fp8_fast / bf16        = {krea_ratio:.3f}x  "
          f"(criterion 1: >= {PASS_RATIO_KREA2:.2f}x)")
    print(f"  ideogram4 fp8_fast / fp8_dequant = {ideo_ratio:.3f}x  "
          f"(criterion 2: >= {PASS_RATIO_IDEOGRAM4:.2f}x)")

    quality_dirs = [os.path.join(IMAGE_DIR, v, a) for v, a in
                    (("krea2", "fp8_fast"), ("krea2", "bf16"),
                     ("ideogram4", "fp8_fast"), ("ideogram4", "fp8_dequant"))]

    if krea_ratio < 1.00:
        branch = "REVERT"
        print("\nBRANCH: below 1.00x -> REVERT the fast path.")
        rc = 1
    elif krea_ratio < PASS_RATIO_KREA2:
        branch = "KEEP, NO CLAIM"
        print(f"\nBRANCH: {1.00:.2f}x <= krea2 < {PASS_RATIO_KREA2:.2f}x -> KEEP the code path, "
              "reframed as 'removes the dequantization step for models already stored in FP8'.")
        print("  - make NO speed claim anywhere (UI, docstrings, commit messages)")
        print("  - do NOT flip the default")
        print("  - do NOT proceed to Phase 2 and do NOT generalize to the runtime "
              "unet_quantization enum (Phase 2's value proposition was speed)")
        rc = 1
    elif ideo_ratio < PASS_RATIO_IDEOGRAM4:
        branch = "BLOCKED (ideogram4 regression)"
        print("\nBRANCH: krea2 clears 1.10x but ideogram4 REGRESSES against its own "
              "dequant path -> criterion 2 fails, the default does NOT flip.")
        rc = 1
    else:
        branch = "ELIGIBLE TO FLIP (pending quality)"
        print("\nBRANCH: criteria 1 and 2 both PASS. The default may flip ONLY IF "
              "criterion 3 (both quality A/Bs clean) also holds.")
        rc = 0

    print("\ncriterion 3 is MANUAL. A/B at matching seeds, in these directories:")
    for d in quality_dirs:
        print(f"  {d}")
    print("  (krea2: fp8_fast vs bf16;  ideogram4: fp8_fast vs fp8_dequant -- the "
          "dequant path is what users run today, so it is the quality reference)")
    print("\nreminder from the pre-registered rule: 'beats the dequant path' can "
          "justify KEEPING this code for already-FP8 checkpoints, but never on its "
          "own justifies flipping a default or widening the surface.")
    print(f"\nbranch: {branch}")
    return rc


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vehicle", choices=sorted(VEHICLES))
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--source", help="checkpoint path/id as /models/load expects it")
    ap.add_argument("--source-type", default="diffusers", choices=SOURCE_TYPES,
                    help="source_type for POST /models/load. An Ideogram 4 model "
                         "directory is 'diffusers'; a Krea 2 single file or shard "
                         "index is 'safetensors'. (default: diffusers)")
    ap.add_argument("--quality", action="store_true",
                    help="run the quality A/B set (4 prompts x 2 seeds) instead of the timed runs")
    ap.add_argument("--quality-steps", type=int, default=STEPS,
                    help=f"steps for the quality set (default {STEPS}, same as the timed arm)")
    ap.add_argument("--probe", action="store_true",
                    help="record this GPU's torch._scaled_mm scaling mode and exit")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                    help="actually load the model and run generations (heavy, GPU-mutating)")
    ap.set_defaults(dry_run=True)
    args = ap.parse_args()

    if args.probe:
        return probe()
    if args.report:
        return report()
    if not args.vehicle or not args.arm or not args.source:
        ap.error("--vehicle, --arm and --source are required unless --probe/--report is given")
    if args.vehicle == "krea2" and args.arm == "fp8_dequant":
        print("note: krea2:fp8_dequant is not part of the decision rule (krea2 carries "
              "fp8_fast vs bf16). It is recorded, and it is useful context, but it "
              "cannot substitute for any required arm.")
    if args.vehicle == "ideogram4" and args.arm == "bf16":
        ap.error("an ideogram4 bf16 arm is INVALID -- see the module docstring. "
                 "Ideogram 4 keeps two transformers resident; a bf16 arm would measure "
                 "offload traffic, not the GEMM.")

    if args.dry_run:
        root = base_url()
        print("=== DRY RUN (no request sent) ===")
        print(f"Would POST {root}/api/v1/models/load "
              f"data={{'source_type': {args.source_type!r}, 'source': {args.source!r}}}")
        if args.quality:
            print(f"Would POST {root}/api/v1/generate/txt2img "
                  f"{len(QUALITY_PROMPTS)}x{len(QUALITY_SEEDS)} = "
                  f"{len(QUALITY_PROMPTS) * len(QUALITY_SEEDS)} times at "
                  f"{args.quality_steps} steps, prompts:")
            for p in QUALITY_PROMPTS:
                print(f"  - {p}")
        else:
            print(f"Would POST {root}/api/v1/generate/txt2img "
                  f"({WARMUP} warmup + {RUNS} timed) data=")
            print(json.dumps(form_data(args.vehicle), indent=2))
            print(f"Would time sampler steps via {ws_url()}")
        print(f"Would save images to {os.path.join(IMAGE_DIR, args.vehicle, args.arm)}")
        print(f"Would record results in {RESULTS}")
        print("\nThis loads a model and runs real generations on the GPU. Re-run with "
              "--no-dry-run when the GPU is idle, nothing else is generating or "
              "training on this backend, AND the backend was started with the "
              "SUSHI_FP8_SCALED_MM value this arm requires.")
        return 0

    if args.quality:
        run_quality(args.vehicle, args.arm, args.source_type, args.source, args.quality_steps)
    else:
        run_timed(args.vehicle, args.arm, args.source_type, args.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
