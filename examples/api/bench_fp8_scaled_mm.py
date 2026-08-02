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
PROTOCOL (arms are switched in one backend process)
===========================================================================
The GEMM path is a per-process setting exposed at
``GET/POST /api/v1/system/fp8-scaled-mm``. This script flips it between arms, so
BOTH arms of a vehicle are measured inside ONE backend process. That removes the
cross-session variance a restart injects (allocator state, clock/power state,
page cache) from a ratio the rule reads to two decimal places.

``SUSHI_FP8_SCALED_MM`` still sets the value the backend STARTS with; it is no
longer how an arm is selected, and no restart is needed between arms. The
endpoint refuses (409) while a generation or a training run is active, so nothing
else may be running on this backend.

    # 0. record the GPU's scaled-GEMM capability mode FIRST -- a gate result is
    #    meaningless without knowing which mode was measured (rowwise vs
    #    tensorwise). No backend needed.
    venv/Scripts/python.exe examples/api/bench_fp8_scaled_mm.py --probe

    # 1. ideogram4 -- SAME checkpoint in both arms, so the toggle is the only
    #    difference. Runs are interleaved dequant/fast/dequant/fast..., which
    #    also spreads any monotonic drift across both arms instead of loading it
    #    onto whichever ran second.
    venv/Scripts/python.exe examples/api/bench_fp8_scaled_mm.py \
        --vehicle ideogram4 --pair \
        --source-type diffusers --source <ideogram4 fp8 dir> --no-dry-run

    # 2. krea2 -- the arms are DIFFERENT checkpoints (bf16 vs fp8), so they
    #    cannot be interleaved per run; a model load sits between them. Order:
    #    bf16 reps -> toggle on + load fp8 -> fp8_fast reps -> reload bf16 for
    #    ONE closing replicate (drift sentinel: if it does not match the opening
    #    bf16 median, the session drifted and the ratio is not trustworthy).
    venv/Scripts/python.exe examples/api/bench_fp8_scaled_mm.py \
        --vehicle krea2 --pair \
        --source-type safetensors --source <bf16 krea2 index/file> \
        --fp8-source <fp8 krea2 index> --no-dry-run

    # single arm (still supported; sets the toggle itself before running)
    venv/Scripts/python.exe examples/api/bench_fp8_scaled_mm.py \
        --vehicle ideogram4 --arm fp8_fast \
        --source-type diffusers --source <ideogram4 fp8 dir> --no-dry-run

    # quality A/B: 4 prompts x 2 seeds per arch, per arm (8 images each)
    venv/Scripts/python.exe examples/api/bench_fp8_scaled_mm.py \
        --vehicle krea2 --arm fp8_fast --quality \
        --source-type safetensors --source <fp8 krea2 index> --no-dry-run

    # evaluate against the pre-registered rule
    venv/Scripts/python.exe examples/api/bench_fp8_scaled_mm.py --report

Results accumulate in ``tmp/fp8_bench_results.json`` keyed ``<vehicle>:<arm>``;
images land in ``tmp/fp8_bench_images/<vehicle>/<arm>/``. Every record carries the
toggle state (``enabled``/``origin``) reported by the backend before the arm ran
and the ``resolved_modes`` it reported after, so a record states which path it
actually measured rather than which one was requested.

===========================================================================
MEASUREMENT GATE G2 -- int8 W8A8 (separate from, and additional to, G1 above)
===========================================================================
This gate governs a DIFFERENT fast path: int8 (``torch._int_mm``) W8A8, not
FP8. It is written down BEFORE any int8 arm, CLI flag, or scaled-GEMM code
exists in this script -- same discipline as G1, for the same reason. Nothing
in this section may be edited once a measurement exists, and nothing above
this section (the FP8 rule, RUNS/WARMUP, prompts, seeds, shapes, ``report()``)
is touched by it.

IMPLEMENTATION STATUS: NOT IMPLEMENTED. There is no int8 arm, no ``--vehicle
... --arm int8_*`` flag, and ``report()`` does not evaluate this gate. This
section is the rule the future code will be judged against, not a promise
that the code exists. Do not half-wire it -- when the int8 path is built, its
own commit adds the arm plumbing AND a ``report()`` branch for this gate in
one place, so the two never drift apart.

VEHICLE: Krea 2 only, quantized to int8 FROM ITS BF16 SOURCE. It must NOT be
produced by dequantizing the shipped e4m3 checkpoint: e4m3 has already
discarded weight information that int8 quantization cannot recover, so an
e4m3-derived int8 arm would be judged against a floor already lowered by a
different lossy step, not against the same bf16 anchor the speed axis uses.

FIVE ARMS, same process, same discipline as G1's interleaving:
  - bf16                     (the anchor, both axes)
  - int8_weight_only         (weight-only, activations stay bf16)
  - int8_w8a8_eager          (W8A8, eager ``torch._int_mm`` chain)
  - int8_w8a8_fused          (W8A8, fused GEMM path -- the one this gate is
                              actually deciding whether to default on)
  - int8_w8a8_hadamard       (W8A8 + Hadamard rotation -- built ONLY if the
                              pre-authorized retry below is triggered; absent
                              otherwise)

QUALITY -- ALL FOUR required, measured on the int8_w8a8_fused arm against the
bf16 arm, same protocol as G1's quality A/B (4 prompts x 2 seeds):

  1. Flat-region residual (flattest 256x256 tile, high-pass sigma=6) <=
     **1.15x bf16**, at BOTH seed 987654321 AND seed 12345.
     Calibration from the FP8 gate's actual numbers, so this bar is not
     invented in a vacuum: at seed 12345 the FP8 run measured bf16=0.199,
     dequant=0.319, fast=0.398; at seed 987654321 it measured bf16=0.351,
     dequant=0.358, fast=0.532. A 1.15x bar against THIS gate's own bf16
     anchor would admit a dequant-shaped result (0.358/0.351 = 1.02x) and
     reject a fast-shaped one (0.532/0.351 = 1.52x) -- i.e. it separates the
     two classes the FP8 gate already showed exist.
  2. Residual power-spectrum ratio at the 32-128px mottle wavelength <=
     **1.3x bf16**. The FP8 fast arm measured 3.0-8.4x here -- this bar
     rejects anything in that shape.
  3. Brightness drift vs bf16 must NOT be one-signed across all 8 quality
     pairs, AND mean |dV| <= **1.0**. The FP8 fast arm was +2.93 mean,
     positive in 8/8 pairs -- this bar rejects that shape on either symptom
     alone (a one-signed drift under 1.0, or a two-signed drift over 1.0,
     both fail).
  4. Blind human A/B clean at the mottle seed (987654321, quality prompt
     index 1, the flat-gradient prompt). This seed/prompt pair is the target
     specifically because its bf16 reference is genuinely clean, which is
     what makes "clean" or "not clean" a real judgement rather than a
     coin-flip on an already-imperfect reference.

SPEED -- int8_w8a8_fused >= **1.10x** the steps/s of bf16 (same protocol as
G1 criterion 1: median of >= 3 timed runs, 1 warmup, fixed prompt/seed/shape,
>= 20 steps). This is the same bar the FP8 fast path cleared (1.155x on
Krea 2). RECORDED EXPLICITLY: there is NO requirement that int8 beat the FP8
fast path's speed number. That path failed its own quality gate and is not
the shipped default -- outrunning a rejected arm proves nothing about
whether int8 is safe to ship.

BRANCHES:

  BOTH quality and speed pass:
    The int8 W8A8 path may default ON for int8 checkpoints.
    RECORD WHY THIS IS LICENSED WHERE THE FP8 FLIP WAS NOT: this gate is
    anchored to bf16 on BOTH axes (quality residual vs bf16, speed ratio vs
    bf16) -- it is not "beats fp8_dequant" or "beats fp8_fast", which was the
    weaker, checkpoint-relative comparison G1 used for its Ideogram 4 arm.
    A bf16-anchored pass on both axes is the strongest claim this repo's
    gates make about a quantization path, which is why it is allowed to move
    a default and G1's dequant-relative pass was not.

  Quality passes, speed fails:
    Ships as a factual VRAM-reduction format only. No speed claim anywhere --
    not in the UI, not in a docstring, not in a commit message. Same rule G1
    applies to its 1.00-1.10x band, for the same reason: a claim not backed
    by this gate's own numbers does not get made because a *different* gate
    passed.

  Quality fails in an outlier-shaped way (i.e. the failure is plausibly one
  or a few pathological rows/channels rather than a systemic W8A8 error, the
  same shape Phase 0 already found once -- see below):
    ONE pre-authorized retry, and only one: rebuild the int8_w8a8_hadamard
    arm (W8A8 + Hadamard rotation to spread outlier magnitude across the
    channel before quantizing) and re-run this exact gate against it. If that
    retry also fails, there is no second retry and no third arm invented to
    try again -- proceed to the next branch.

  Anything else (quality fails in a non-outlier-shaped way, the retry above
  also fails, or the result is ambiguous rather than a clean pass/fail):
    The code is removed, not parked. No ``SUSHI_INT8_*`` flag is left behind
    "in case it's useful later" -- an unshippable path sitting behind a flag
    is exactly the kind of surface this repo's gates exist to prevent.

===========================================================================
PHASE 0 MEASUREMENTS THIS GATE WAS DESIGNED AGAINST
===========================================================================
Recorded so a later reader can see what was already known when the G2 rule
above was fixed -- these numbers PRE-DATE the rule and did not shape its
thresholds after the fact; the thresholds above were chosen independently
and these are cited for context, not as the source of the bars.

  - Raw ``torch._int_mm`` at Krea 2's shapes: 2.857-3.075x bf16,
    layer-count-weighted mean 3.009x. (For calibration: the smaller-scope G1
    threshold this Phase 0 work was originally sized against was 1.30x, not
    the 1.10x this gate uses -- the two numbers are not the same claim.)
  - Eager int8 W8A8 chain: 1.515x bf16, vs the shipped fused FP8 path's
    1.550x on the same hardware. Fused int8: 2.561x bf16.
  - Per-row accuracy on 112 real Krea 2 layers: e4m3 error was flat across
    layers at 0.02628-0.02649; int8 per-row error ranged 0.01016-0.03117.
    Geometric-mean advantage of int8 over e4m3: 2.06x for weight-only,
    2.19x for W8A8. A separate simulation had predicted 3.3x; the real
    per-layer measurement came in about 60% more modest than that
    simulation, which is why this gate does not lean on simulated numbers
    for its thresholds.
  - One inverted layer: ``transformer_blocks.27.ff.down`` had int8 error
    (0.03117) WORSE than its e4m3 error (0.02628) -- the one layer in 112
    where int8 lost to FP8. It was predicted in advance by within-row crest
    factor: 32.6 for this layer vs a typical 4.5-6 elsewhere. This is the
    concrete precedent behind the "outlier-shaped failure" retry clause
    above -- a single high-crest-factor layer is exactly the failure mode
    Hadamard rotation targets, and this layer is why that retry exists
    rather than being invented generically.
  - The GPU was running at a 240W power cap, 735 MHz SM clock under load
    (vs a 3105 MHz maximum) for all of the above. All arms in a given
    session throttled equally, so the RATIOS between arms hold, but none of
    the absolute steps/s or ms/layer figures above generalize to hardware
    running at its normal clocks.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
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


# Set once per invocation by ``check_gpu_exclusivity()`` (called from ``main()``
# before anything that records a number) and attached to EVERY record written
# from then on, so a contaminated run can never later be mistaken for a clean
# one just by looking at the number itself.
_RUN_GPU_STATE = None


def store_result(key, record):
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_results()
    record = dict(record)
    if _RUN_GPU_STATE is not None:
        record["gpu_exclusivity"] = _RUN_GPU_STATE
    data[key] = record
    with open(RESULTS, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


# ---------------------------------------------------------------------------
# GPU exclusivity precondition -- run BEFORE any number is recorded.
#
# The module docstring already states the requirement ("nothing else may be
# generating or training on this backend while a timed arm runs, and the GPU
# must otherwise be idle"); this is what enforces it. A whole G1 session was
# invalidated once because that requirement was documented but not checked:
# two unrelated CUDA training processes ran at 100% utilization throughout,
# and seven contaminated generations were recorded before anyone noticed.
# ---------------------------------------------------------------------------

def _query_compute_apps():
    """``nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv``.

    Returns a list of ``{"pid", "used_memory", "name"}`` dicts, or ``None`` if
    nvidia-smi could not be run at all (absent, PATH issue, driver error). The
    caller must treat ``None`` as "unknown", never as "clean".
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
    except FileNotFoundError:
        return None
    except Exception as exc:                          # pragma: no cover
        print(f"WARNING: could not run nvidia-smi ({type(exc).__name__}: {exc})")
        return None
    if proc.returncode != 0:
        print(f"WARNING: nvidia-smi --query-compute-apps exited {proc.returncode}: "
              f"{proc.stderr.strip()}")
        return None
    procs = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        pid_s, mem_s = parts[0], parts[1]
        name = ",".join(parts[2:])  # a process path could itself contain a comma
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        procs.append({"pid": pid, "used_memory": mem_s, "name": name})
    return procs


_PROCESS_TABLE_ROW = re.compile(r"^\|\s*\d+\s+\S+\s+\S+\s+(\d+)\s+(C\+G|C|G)\s")


def _query_process_types():
    """Best-effort PID -> Type ('C' / 'G' / 'C+G') from the plain ``nvidia-smi``
    process table.

    ``--query-compute-apps`` reports every GPU-accelerated desktop process as a
    "compute app" on Windows/WDDM (verified empirically on this machine:
    Explorer, Notepad, browser helpers all show up, each with
    ``used_memory=[N/A]``), which would make an exclusivity check fire on an
    otherwise-idle desktop. The plain table's ``Type`` column is the only place
    this driver actually distinguishes pure CUDA compute (``C``) from a
    graphics/UI context (``G`` / ``C+G``), so it narrows the compute-apps list
    down to real compute work. Returns ``{}`` (not ``None``) on any parse
    failure: the caller then falls back to the unfiltered compute-apps list,
    which fails toward flagging too much rather than missing a real foreign
    process.
    """
    try:
        proc = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
    except Exception:                                  # pragma: no cover
        return {}
    types = {}
    for line in proc.stdout.splitlines():
        m = _PROCESS_TABLE_ROW.match(line)
        if m:
            types[int(m.group(1))] = m.group(2)
    return types


def _find_backend_pid():
    """PID of the process listening on the backend's own host:port, via psutil.

    Returns ``(pid, None)`` on success or ``(None, reason)`` on failure. Reuses
    ``backend/.port_info`` (already the source of truth for the backend's
    address elsewhere in this script) rather than adding a new discovery
    mechanism, and psutil rather than a new process-listing dependency --
    psutil is already imported elsewhere in this repo (see
    ``backend/core/gpu_coordinator.py``).
    """
    try:
        import psutil
    except ImportError:
        return None, "psutil is not installed"
    try:
        with open(PORT_INFO, encoding="utf-8") as fh:
            info = json.load(fh)
        port = info["port"]
    except Exception as exc:
        return None, f"could not read {PORT_INFO} ({type(exc).__name__}: {exc})"
    try:
        conns = psutil.net_connections(kind="inet")
    except Exception as exc:
        # On Windows this can require elevated privileges to see other users'
        # sockets; the backend's own socket (same user, same session) is
        # normally still visible, but fail open rather than assume that.
        return None, f"psutil.net_connections() failed ({type(exc).__name__}: {exc})"
    for c in conns:
        if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port:
            return c.pid, None
    return None, f"no LISTEN socket found on port {port} -- is the backend running?"


def check_gpu_exclusivity(allow_foreign):
    """Startup precondition: the GPU must be idle apart from the backend.

    Returns a dict describing what was found. That dict is attached to EVERY
    result record written for the rest of this process (via ``store_result``
    reading ``_RUN_GPU_STATE``), so ``--allow-foreign-gpu`` can never silently
    produce a number that looks clean.

    Raises ``SystemExit`` (without ``--allow-foreign-gpu``) if a non-backend
    CUDA compute process is found. nvidia-smi or psutil being unavailable is a
    WARNING, not a hard failure: the check degrades to "not verified" rather
    than blocking every environment that lacks the CLI tool or admin rights to
    enumerate sockets.
    """
    state = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "verified": False,
        "foreign_processes": [],
        "allow_foreign_gpu": bool(allow_foreign),
        "note": None,
    }

    compute_apps = _query_compute_apps()
    if compute_apps is None:
        state["note"] = "nvidia-smi unavailable; GPU exclusivity NOT verified"
        print(f"WARNING: {state['note']}")
        return state

    types = _query_process_types()
    if types:
        candidates = [p for p in compute_apps if types.get(p["pid"]) == "C"]
    else:
        candidates = compute_apps
        state["note"] = ("could not read the Type column from plain nvidia-smi; "
                          "falling back to the unfiltered --query-compute-apps list, "
                          "which may over-report on Windows/WDDM")
        print(f"WARNING: {state['note']}")

    backend_pid, backend_err = _find_backend_pid()
    if backend_pid is None:
        note = f"could not identify the backend's own PID ({backend_err}); GPU exclusivity NOT verified"
        state["note"] = f"{state['note']}; {note}" if state["note"] else note
        print(f"WARNING: {note}")
        return state

    foreign = [p for p in candidates if p["pid"] != backend_pid]
    state["verified"] = True
    state["backend_pid"] = backend_pid
    state["foreign_processes"] = foreign

    if not foreign:
        print(f"GPU exclusivity OK: only the backend (pid {backend_pid}) is doing "
              "CUDA compute work on this GPU.")
        return state

    print(f"\n{'=' * 70}")
    print(f"FOREIGN GPU PROCESS DETECTED -- the benchmark protocol requires the "
          f"GPU to be idle apart from the backend (pid {backend_pid}):")
    for p in foreign:
        print(f"  pid={p['pid']:<8} used_memory={p['used_memory']:<12} name={p['name']}")
    print("Any number recorded now would be contaminated by this process (see the "
          "module docstring: VAE decode -- a path with no FP8 code at all -- ran "
          "3.0-4.8x slower the last time this was missed).")
    if allow_foreign:
        print("--allow-foreign-gpu given: proceeding anyway. This will be recorded "
              "in EVERY result written during this run.")
        print("=" * 70 + "\n")
        return state
    print("Stop the foreign process(es), or pass --allow-foreign-gpu to proceed "
          "anyway (the contamination will be recorded in every result).")
    print("=" * 70)
    raise SystemExit(3)


def _query_gpu_power_and_clock():
    """Power limit and SM clock context for the ``_probe`` record.

    A ratio the decision rule reads to two decimal places is meaningless
    without knowing whether the card was power- or clock-throttled when it was
    measured. The investigation that motivated this found the card running at
    a 240 W software power cap against a 300 W default (SW Power Cap active),
    with SM clock pinned at 1575 MHz against a 2505 MHz applications clock --
    caused by an unrelated foreign workload, invisible unless recorded
    alongside the number it affected. Returns ``{}`` if nvidia-smi cannot
    answer (absence is a warning, not a hard failure -- same policy as the
    exclusivity check above).
    """
    fields = ["power.limit", "power.default_limit", "power.draw",
              "clocks.sm", "clocks.max.sm"]
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
    except Exception as exc:                          # pragma: no cover
        print(f"WARNING: could not query GPU power/clock state ({type(exc).__name__}: {exc})")
        return {}
    if proc.returncode != 0:
        print(f"WARNING: nvidia-smi power/clock query exited {proc.returncode}: "
              f"{proc.stderr.strip()}")
        return {}
    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    keys = ["power_limit", "power_default_limit", "power_draw", "sm_clock", "sm_clock_max"]
    if len(parts) != len(keys):
        return {"raw": line}
    return dict(zip(keys, parts))


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
        "power_and_clock": _query_gpu_power_and_clock(),
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


def get_toggle(root):
    """GET the backend's current FP8 GEMM state."""
    resp = requests.get(f"{root}/api/v1/system/fp8-scaled-mm", timeout=60)
    resp.raise_for_status()
    return resp.json()


def set_toggle(root, enabled):
    """POST the FP8 GEMM state. 409 means something else is running on this backend."""
    resp = requests.post(
        f"{root}/api/v1/system/fp8-scaled-mm", json={"enabled": bool(enabled)}, timeout=60
    )
    if resp.status_code == 409:
        print(f"ABORT: the backend refused the toggle -- {resp.text}\n"
              "Nothing else may be generating or training on this backend while a "
              "timed arm runs (the progress channel is a global broadcast anyway).")
        raise SystemExit(2)
    resp.raise_for_status()
    return resp.json()


def toggle_for_arm(root, arm):
    """Put the backend in the GEMM state this arm requires and report it."""
    enabled = arm == "fp8_fast"
    state = set_toggle(root, enabled)
    print(f"  toggle -> enabled={state['enabled']} origin={state['origin']} "
          f"(arm {arm}); probe cache cleared")
    if state["enabled"] != enabled:
        print("ABORT: the backend did not take the requested GEMM state.")
        raise SystemExit(2)
    return state


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

class ArmSamples:
    """Accumulator for one arm's timed replicates (identical work per replicate)."""

    def __init__(self, vehicle, arm, source_type, source, info, toggle_before):
        self.vehicle = vehicle
        self.arm = arm
        self.source_type = source_type
        self.source = source
        self.info = info
        self.toggle_before = toggle_before
        self.times = []
        self.step_rates = []
        self.images = []

    def add(self, root, data, timer, tag):
        """Run one timed replicate and record it."""
        elapsed, body = txt2img(root, data)
        measured = timer.take(STEPS)
        self.times.append(elapsed)
        if measured is not None:
            steps, span = measured
            self.step_rates.append(steps / span)
            print(f"  [{self.arm}] {tag}: sampler {steps} steps in {span:.2f}s -> "
                  f"{steps / span:.3f} steps/s   (end-to-end {elapsed:.2f}s)")
        else:
            print(f"  [{self.arm}] {tag}: no usable progress samples; "
                  f"end-to-end {elapsed:.2f}s")
        path = save_image(root, body, self.vehicle, self.arm, tag)
        if path:
            self.images.append(path)
        return body

    def store(self, root, key_suffix=""):
        """Reduce to the pre-registered statistic (median of the timed runs) and save."""
        if len(self.step_rates) == len(self.times) and self.step_rates:
            timing_source = "ws_progress_steps"
            steps_per_s = statistics.median(self.step_rates)
        else:
            timing_source = "http_end_to_end"
            steps_per_s = STEPS / statistics.median(self.times)
            print("  NOTE: falling back to end-to-end HTTP time. That window includes "
                  "VAE decode, PNG save and the DB write, which are identical across "
                  "arms, so the resulting ratio is CONSERVATIVE (it understates a "
                  "real speedup rather than inventing one).")
        try:
            toggle_after = get_toggle(root)
        except Exception as exc:                       # pragma: no cover
            toggle_after = {"error": f"{type(exc).__name__}: {exc}"}
        record = {
            "vehicle": self.vehicle,
            "arm": self.arm,
            "source_type": self.source_type,
            "source": self.source,
            "model_info": self.info.get("model_info"),
            "steps": STEPS,
            "width": WIDTH,
            "height": HEIGHT,
            "seed": SEED,
            "prompt": PROMPT,
            "timing_source": timing_source,
            "step_rates": self.step_rates,
            "http_times_s": self.times,
            "median_http_s": statistics.median(self.times),
            "steps_per_s": steps_per_s,
            "fast_accum_env": os.environ.get("SUSHI_FP8_FAST_ACCUM",
                                             "(unset -> shipping default 1)"),
            # Which GEMM path this record actually measured, as the backend
            # reported it -- not which one was requested. ``resolved_modes`` in
            # ``toggle_after`` distinguishes "fast path ran" from "fast path was
            # enabled but the probe rejected every scaling mode".
            "toggle_before": self.toggle_before,
            "toggle_after": toggle_after,
            "images": self.images,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        key = f"{self.vehicle}:{self.arm}{key_suffix}"
        store_result(key, record)
        print(f"  MEDIAN [{self.arm}] {steps_per_s:.3f} steps/s  [{timing_source}]  "
              f"resolved_modes={toggle_after.get('resolved_modes')}  "
              f"(saved to {RESULTS} as '{key}')")
        return record


def _banner(vehicle, extra=""):
    print(f"  {STEPS} steps  {WIDTH}x{HEIGHT}  seed={SEED}  {WARMUP} warmup + {RUNS} timed"
          f"{extra}")
    print(f"  vehicle={vehicle}  arms are switched through "
          f"POST /api/v1/system/fp8-scaled-mm (no backend restart)")


def run_timed(vehicle, arm, source_type, source):
    """One arm, in the current backend process. The toggle is set from the arm."""
    check_fast_accum()
    root = base_url()
    print(f"vehicle={vehicle} arm={arm} source={source}")
    _banner(vehicle)

    toggle_before = toggle_for_arm(root, arm)
    info = load_model(root, source_type, source)
    print(f"  model loaded: {info.get('model_info')}")

    data = form_data(vehicle)
    samples = ArmSamples(vehicle, arm, source_type, source, info, toggle_before)

    with StepTimer() as timer:
        for i in range(WARMUP):
            elapsed, body = txt2img(root, data)
            timer.take(STEPS)
            print(f"  warmup {i}: {elapsed:.2f}s  warnings={body.get('warnings')}")
        if timer.error:
            print(f"  WARNING: progress WebSocket unusable ({timer.error})")
        for i in range(RUNS):
            samples.add(root, data, timer, f"run{i}")

    samples.store(root)


def run_pair_ideogram4(source_type, source):
    """Both ideogram4 arms on the SAME checkpoint, interleaved dequant/fast.

    Nothing but the toggle differs, so the arms can alternate replicate by
    replicate; any monotonic session drift is then shared by both arms instead of
    being charged to whichever ran second. Warmup and the median-of-RUNS
    statistic are exactly as the pre-registered rule specifies, per arm.
    """
    check_fast_accum()
    root = base_url()
    vehicle = "ideogram4"
    arms = ("fp8_dequant", "fp8_fast")
    print(f"vehicle={vehicle} PAIRED arms={arms} source={source}")
    _banner(vehicle, "  per arm, interleaved")

    info = load_model(root, source_type, source)
    print(f"  model loaded: {info.get('model_info')}")
    data = form_data(vehicle)

    samples = {}
    with StepTimer() as timer:
        for arm in arms:
            state = toggle_for_arm(root, arm)
            samples[arm] = ArmSamples(vehicle, arm, source_type, source, info, state)
            for i in range(WARMUP):
                elapsed, body = txt2img(root, data)
                timer.take(STEPS)
                print(f"  [{arm}] warmup {i}: {elapsed:.2f}s  "
                      f"warnings={body.get('warnings')}")
        if timer.error:
            print(f"  WARNING: progress WebSocket unusable ({timer.error})")
        for i in range(RUNS):
            for arm in arms:
                toggle_for_arm(root, arm)
                samples[arm].add(root, data, timer, f"run{i}")

    for arm in arms:
        samples[arm].store(root)


def run_pair_krea2(source_type, source, fp8_source, fp8_source_type):
    """Both krea2 arms in one process: bf16 -> fp8_fast -> one closing bf16.

    The arms are different checkpoints, so a model load sits between them and
    they cannot be interleaved. The closing bf16 replicate is a DRIFT SENTINEL,
    not part of the statistic: if it lands far from the opening bf16 median, the
    session moved during the run and the ratio is not trustworthy.
    """
    check_fast_accum()
    root = base_url()
    vehicle = "krea2"
    print(f"vehicle={vehicle} PAIRED arms=('bf16', 'fp8_fast')")
    print(f"  bf16 source={source}")
    print(f"  fp8  source={fp8_source}")
    _banner(vehicle, "  per arm")

    data = form_data(vehicle)
    results = {}

    with StepTimer() as timer:
        # --- arm 1: bf16 baseline (toggle off; a bf16 checkpoint has no
        # Fp8Linear at all, so this only pins the process state) --------------
        state = toggle_for_arm(root, "bf16")
        info = load_model(root, source_type, source)
        print(f"  model loaded: {info.get('model_info')}")
        bf16 = ArmSamples(vehicle, "bf16", source_type, source, info, state)
        for i in range(WARMUP):
            elapsed, body = txt2img(root, data)
            timer.take(STEPS)
            print(f"  [bf16] warmup {i}: {elapsed:.2f}s  warnings={body.get('warnings')}")
        if timer.error:
            print(f"  WARNING: progress WebSocket unusable ({timer.error})")
        for i in range(RUNS):
            bf16.add(root, data, timer, f"run{i}")
        results["bf16"] = bf16.store(root)

        # --- arm 2: fp8 checkpoint, scaled-GEMM path ------------------------
        state = toggle_for_arm(root, "fp8_fast")
        info8 = load_model(root, fp8_source_type, fp8_source)
        print(f"  model loaded: {info8.get('model_info')}")
        fast = ArmSamples(vehicle, "fp8_fast", fp8_source_type, fp8_source, info8, state)
        for i in range(WARMUP):
            elapsed, body = txt2img(root, data)
            timer.take(STEPS)
            print(f"  [fp8_fast] warmup {i}: {elapsed:.2f}s  warnings={body.get('warnings')}")
        for i in range(RUNS):
            fast.add(root, data, timer, f"run{i}")
        results["fp8_fast"] = fast.store(root)

        # --- closing drift sentinel: ONE bf16 replicate ----------------------
        state = toggle_for_arm(root, "bf16")
        info_s = load_model(root, source_type, source)
        sentinel = ArmSamples(vehicle, "bf16", source_type, source, info_s, state)
        sentinel.add(root, data, timer, "sentinel0")
        results["sentinel"] = sentinel.store(root, key_suffix="_sentinel")

    opening = results["bf16"]["steps_per_s"]
    closing = results["sentinel"]["steps_per_s"]
    drift = closing / opening if opening else float("nan")
    print(f"\ndrift sentinel: closing bf16 {closing:.3f} steps/s vs opening "
          f"{opening:.3f} steps/s -> {drift:.3f}x")
    print("  This is NOT part of the decision rule. It is the session-stability "
          "check the interleaved ideogram4 pair gets for free and this vehicle "
          "cannot have: read the krea2 ratio in light of it.")


def run_quality(vehicle, arm, source_type, source, steps):
    """4 prompts x 2 seeds -> 8 images for this arm, for a human A/B."""
    check_fast_accum()
    root = base_url()
    print(f"quality A/B: vehicle={vehicle} arm={arm} "
          f"{len(QUALITY_PROMPTS)} prompts x {len(QUALITY_SEEDS)} seeds, {steps} steps")
    toggle_before = toggle_for_arm(root, arm)
    info = load_model(root, source_type, source)
    print(f"  model loaded: {info.get('model_info')}")

    saved = []
    warnings_seen = []
    for pi, prompt in enumerate(QUALITY_PROMPTS):
        for si, seed in enumerate(QUALITY_SEEDS):
            data = form_data(vehicle, prompt=prompt, seed=seed, steps=steps)
            elapsed, body = txt2img(root, data)
            path = save_image(root, body, vehicle, arm, f"q{pi}_s{si}")
            saved.append(path)
            warnings_seen.append({"prompt_index": pi, "seed": seed,
                                  "warnings": body.get("warnings")})
            print(f"  p{pi} seed={seed}: {elapsed:.2f}s -> {path}")

    try:
        toggle_after = get_toggle(root)
    except Exception as exc:                           # pragma: no cover
        toggle_after = {"error": f"{type(exc).__name__}: {exc}"}

    key = f"{vehicle}:{arm}:quality"
    store_result(key, {
        "vehicle": vehicle,
        "arm": arm,
        "source_type": source_type,
        "source": source,
        # Recorded so a quality record is self-describing months later: which
        # checkpoint produced these images, whether any generation reported a
        # degradation, and which GEMM path was actually in force.
        "model_info": info.get("model_info"),
        "warnings": warnings_seen,
        "toggle_before": toggle_before,
        "toggle_after": toggle_after,
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
    ap.add_argument("--pair", action="store_true",
                    help="run BOTH of the vehicle's arms in this one backend process, "
                         "switching the GEMM path through POST /system/fp8-scaled-mm "
                         "(ideogram4: interleaved on one checkpoint; krea2: bf16 -> "
                         "fp8_fast -> one closing bf16 drift sentinel, needs --fp8-source)")
    ap.add_argument("--fp8-source", help="krea2 --pair only: the FP8 checkpoint "
                                         "(--source is then the bf16 one)")
    ap.add_argument("--fp8-source-type", default=None, choices=SOURCE_TYPES,
                    help="source_type for --fp8-source (defaults to --source-type)")
    ap.add_argument("--quality", action="store_true",
                    help="run the quality A/B set (4 prompts x 2 seeds) instead of the timed runs")
    ap.add_argument("--quality-steps", type=int, default=STEPS,
                    help=f"steps for the quality set (default {STEPS}, same as the timed arm)")
    ap.add_argument("--probe", action="store_true",
                    help="record this GPU's torch._scaled_mm scaling mode and exit")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                    help="actually load the model and run generations (heavy, GPU-mutating)")
    ap.add_argument("--allow-foreign-gpu", action="store_true",
                    help="proceed even if a non-backend CUDA compute process is detected "
                         "on the GPU. The fact is recorded in EVERY result written during "
                         "this run so a contaminated number can never be mistaken for a "
                         "clean one. Off by default: the pre-registered rule reads ratios "
                         "to two decimal places and a foreign workload has been shown to "
                         "silently invalidate an entire session.")
    ap.set_defaults(dry_run=True)
    args = ap.parse_args()

    global _RUN_GPU_STATE

    if args.probe:
        _RUN_GPU_STATE = check_gpu_exclusivity(args.allow_foreign_gpu)
        return probe()
    if args.report:
        return report()
    if args.pair:
        if not args.vehicle or not args.source:
            ap.error("--pair needs --vehicle and --source")
        if args.arm:
            ap.error("--pair runs both of the vehicle's arms; do not also pass --arm")
        if args.quality:
            ap.error("--pair is for the timed arms; run --quality per arm "
                     "(the toggle is set from --arm there too)")
        if args.vehicle == "krea2" and not args.fp8_source:
            ap.error("krea2 --pair needs --fp8-source: its two arms are DIFFERENT "
                     "checkpoints (bf16 vs fp8), unlike ideogram4's")
        if args.vehicle == "ideogram4" and args.fp8_source:
            ap.error("ideogram4 --pair uses ONE checkpoint for both arms (only the "
                     "toggle differs); --fp8-source is meaningless here")
        fp8_source_type = args.fp8_source_type or args.source_type
        _RUN_GPU_STATE = check_gpu_exclusivity(args.allow_foreign_gpu)
        if args.dry_run:
            root = base_url()
            print("=== DRY RUN (no request sent) ===")
            if args.vehicle == "ideogram4":
                print(f"Would load {args.source!r} once, then per arm "
                      f"({WARMUP} warmup each) run {RUNS} timed replicates "
                      f"INTERLEAVED fp8_dequant/fp8_fast, flipping "
                      f"{root}/api/v1/system/fp8-scaled-mm between replicates.")
            else:
                print(f"Would run krea2 bf16 ({args.source!r}) {WARMUP}+{RUNS}, then "
                      f"toggle on and load {args.fp8_source!r} ({fp8_source_type}) "
                      f"{WARMUP}+{RUNS}, then reload bf16 for ONE closing drift "
                      f"sentinel replicate.")
            print(f"Would record results in {RESULTS}")
            print("\nThis loads models and runs real generations on the GPU. Re-run "
                  "with --no-dry-run when the GPU is idle and nothing else is "
                  "generating or training on this backend.")
            return 0
        if args.vehicle == "ideogram4":
            run_pair_ideogram4(args.source_type, args.source)
        else:
            run_pair_krea2(args.source_type, args.source, args.fp8_source, fp8_source_type)
        return 0

    if not args.vehicle or not args.arm or not args.source:
        ap.error("--vehicle, --arm and --source are required unless "
                 "--pair/--probe/--report is given")
    if args.vehicle == "krea2" and args.arm == "fp8_dequant":
        print("note: krea2:fp8_dequant is not part of the decision rule (krea2 carries "
              "fp8_fast vs bf16). It is recorded, and it is useful context, but it "
              "cannot substitute for any required arm.")
    if args.vehicle == "ideogram4" and args.arm == "bf16":
        ap.error("an ideogram4 bf16 arm is INVALID -- see the module docstring. "
                 "Ideogram 4 keeps two transformers resident; a bf16 arm would measure "
                 "offload traffic, not the GEMM.")

    _RUN_GPU_STATE = check_gpu_exclusivity(args.allow_foreign_gpu)

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
        print(f"Would POST {root}/api/v1/system/fp8-scaled-mm "
              f"{{'enabled': {args.arm == 'fp8_fast'}}} before running")
        print("\nThis loads a model and runs real generations on the GPU. Re-run with "
              "--no-dry-run when the GPU is idle and nothing else is generating or "
              "training on this backend (the toggle endpoint refuses with 409 while "
              "either is active).")
        return 0

    if args.quality:
        run_quality(args.vehicle, args.arm, args.source_type, args.source, args.quality_steps)
    else:
        run_timed(args.vehicle, args.arm, args.source_type, args.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
