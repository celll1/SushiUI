"""Measurement gate for the FP8 W8A8 scaled-GEMM fast path (Fp8Linear).

The fast path (``backend/core/models/ideogram4/vendor/fp8_linear.py``,
``_scaled_mm_forward``) is OPT-IN behind ``SUSHI_FP8_SCALED_MM=1`` and stays off
until this gate passes. Ideogram 4 and Krea 2 are the architectures that load
weight-only-FP8 checkpoints, so those are what it measures.

PASS CRITERION (both must hold)
-------------------------------
1. SPEED: arm (c) ``fp8_fast`` must reach >= 1.10x the steps/s of arm (a)
   ``bf16`` -- the bf16-checkpoint baseline.
2. QUALITY: a human A/B of the saved arm (c) images against the arm (b)
   ``fp8_dequant`` images at the same seed must show no visible degradation.
   The dequant path is what users run today, so it -- not bf16 -- is the
   quality reference.

Anything short of both leaves the default at the dequant path. The measured
cost of the fast path is real: its error against an fp32 reference is
~3.7e-02 rel RMS versus ~2.6e-02 for the dequant path (~44% more), on the
Ideogram 4 transformer, the Ideogram 4 text encoder, and Krea 2 alike
(see ``tmp/fp8_scaled_mm_numerics.py``). Speed alone does not buy that.

Arm (b) is also the "before" number: it shows what the fast path replaced.

WHY THIS NEEDS THREE BACKEND SESSIONS
-------------------------------------
``SUSHI_FP8_SCALED_MM`` is read at import time in ``fp8_linear``, so the arm is
fixed for the lifetime of a backend process. The backend must be RESTARTED with
the right env value before each fp8 arm (arm (a) uses a bf16 checkpoint, so no
``Fp8Linear`` is constructed and the env value is irrelevant). Agents must not
start/stop servers themselves -- ask the repo owner (see AGENTS.md).

``--source-type`` must be one of ``safetensors`` / ``diffusers`` /
``huggingface`` (``ModelSource`` in ``backend/core/model_loader.py``). It
defaults to ``diffusers``, which is what an Ideogram 4 model directory is; a
Krea 2 single file needs ``--source-type safetensors``.

    # arm (a) -- bf16 checkpoint baseline
    venv/Scripts/python.exe examples/api/bench_fp8_scaled_mm.py \
        --arm bf16 --source-type diffusers --source <bf16 ckpt path> --no-dry-run

    # restart backend with SUSHI_FP8_SCALED_MM unset (or =0), then:
    # arm (b) -- fp8 checkpoint, dequant path (today's default)
    venv/Scripts/python.exe examples/api/bench_fp8_scaled_mm.py \
        --arm fp8_dequant --source-type diffusers --source <fp8 ckpt path> --no-dry-run

    # restart backend with SUSHI_FP8_SCALED_MM=1, then:
    # arm (c) -- fp8 checkpoint, scaled-GEMM fast path
    venv/Scripts/python.exe examples/api/bench_fp8_scaled_mm.py \
        --arm fp8_fast --source-type diffusers --source <fp8 ckpt path> --no-dry-run

    # once all three are recorded
    venv/Scripts/python.exe examples/api/bench_fp8_scaled_mm.py --report

Every arm uses the same prompt, seed, resolution and step count, runs 1 warmup
then 3 timed generations, and reports the MEDIAN. Images from every generation
are downloaded to ``tmp/fp8_bench_images/<arm>/`` so the quality half of the
gate can actually be judged; results accumulate in ``tmp/fp8_bench_results.json``.

The GPU must be otherwise idle for the timings to mean anything.
"""

import argparse
import json
import os
import statistics
import time

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PORT_INFO = os.path.join(REPO_ROOT, "backend", ".port_info")
OUT_DIR = os.path.join(REPO_ROOT, "tmp")
RESULTS = os.path.join(OUT_DIR, "fp8_bench_results.json")
IMAGE_DIR = os.path.join(OUT_DIR, "fp8_bench_images")

# Fixed across every arm: identical work, so only the GEMM path differs.
PROMPT = "a photograph of a stone bridge over a river, morning fog"
NEGATIVE = ""
SEED = 12345
WIDTH = 1024
HEIGHT = 1024
STEPS = 24          # >= 20 required by the gate
RUNS = 3            # >= 3 timed runs, median reported
WARMUP = 1

PASS_RATIO = 1.10
ARMS = ("bf16", "fp8_dequant", "fp8_fast")

# Must match ModelSource in backend/core/model_loader.py; ModelLoader.load_model
# raises ValueError (HTTP 500) on anything else.
SOURCE_TYPES = ("safetensors", "diffusers", "huggingface")


def base_url():
    """Resolve the backend's host/port from the file it writes on startup."""
    with open(PORT_INFO, encoding="utf-8") as fh:
        info = json.load(fh)
    return f"http://{info['host']}:{info['port']}"


def load_model(root, source_type, source):
    """POST /models/load. multipart form, like every other route here."""
    resp = requests.post(
        f"{root}/api/v1/models/load",
        data={"source_type": source_type, "source": source},
        timeout=1800,
    )
    resp.raise_for_status()
    return resp.json()


def form_data():
    """/generate/txt2img is a Form(...) route, not JSON (see txt2img_minimal.py)."""
    return {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE,
        "steps": STEPS,
        "cfg_scale": 4.0,
        "sampler": "euler",
        "schedule_type": "uniform",
        "seed": SEED,
        "width": WIDTH,
        "height": HEIGHT,
    }


def txt2img(root):
    t0 = time.perf_counter()
    resp = requests.post(f"{root}/api/v1/generate/txt2img", data=form_data(), timeout=3600)
    resp.raise_for_status()
    body = resp.json()
    return time.perf_counter() - t0, body


def save_image(root, body, arm, tag):
    """Download the generated PNG so a human A/B is possible after the fact."""
    image = body.get("image") or {}
    filename = image.get("filename") or body.get("filename")
    if not filename:
        print(f"    (no filename in response; nothing to save for {tag})")
        return None
    dest_dir = os.path.join(IMAGE_DIR, arm)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{tag}_{os.path.basename(filename)}")
    resp = requests.get(f"{root}/outputs/{filename}", timeout=300)
    resp.raise_for_status()
    with open(dest, "wb") as fh:
        fh.write(resp.content)
    return dest


def run_arm(arm, source_type, source):
    root = base_url()
    print(f"arm={arm} source={source} steps={STEPS} {WIDTH}x{HEIGHT} seed={SEED}")
    print(f"  env in this shell: SUSHI_FP8_SCALED_MM="
          f"{os.environ.get('SUSHI_FP8_SCALED_MM', '(unset)')} "
          f"-- note the BACKEND's value is what counts, not this one")

    info = load_model(root, source_type, source)
    print(f"  model loaded: {info.get('model_info')}")

    for i in range(WARMUP):
        elapsed, body = txt2img(root)
        print(f"  warmup {i}: {elapsed:.2f}s  warnings={body.get('warnings')}")

    # NOTE on "steps/s": ``elapsed`` is end-to-end HTTP time, so it includes VAE
    # decode, PNG save, thumbnail and the DB write as well as the denoise loop.
    # That fixed overhead is identical across arms, so the ARM RATIO is
    # conservative -- it understates any real speedup rather than inventing one.
    times, images = [], []
    for i in range(RUNS):
        elapsed, body = txt2img(root)
        times.append(elapsed)
        path = save_image(root, body, arm, f"run{i}")
        if path:
            images.append(path)
        print(f"  run {i}: {elapsed:.2f}s -> {STEPS / elapsed:.3f} steps/s  saved={path}")

    median = statistics.median(times)
    record = {
        "arm": arm,
        "source_type": source_type,
        "source": source,
        "steps": STEPS,
        "width": WIDTH,
        "height": HEIGHT,
        "seed": SEED,
        "prompt": PROMPT,
        "times_s": times,
        "median_s": median,
        "steps_per_s": STEPS / median,
        "images": images,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    data = {}
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as fh:
            data = json.load(fh)
    data[arm] = record
    with open(RESULTS, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"  median {median:.2f}s -> {record['steps_per_s']:.3f} steps/s  (saved to {RESULTS})")


def report():
    if not os.path.exists(RESULTS):
        print(f"no results yet: {RESULTS} does not exist")
        return 1
    with open(RESULTS, encoding="utf-8") as fh:
        data = json.load(fh)
    missing = [a for a in ARMS if a not in data]
    if missing:
        print(f"missing arms: {missing}")
        return 1

    for arm in ARMS:
        r = data[arm]
        print(f"{arm:12s} {r['steps_per_s']:.3f} steps/s  "
              f"(median {r['median_s']:.2f}s, source {r['source']})")

    base = data["bf16"]["steps_per_s"]
    fast = data["fp8_fast"]["steps_per_s"]
    slow = data["fp8_dequant"]["steps_per_s"]
    ratio = fast / base
    print(f"\nfp8_fast    / bf16        = {ratio:.3f}x  (speed criterion >= {PASS_RATIO:.2f}x)")
    print(f"fp8_dequant / bf16        = {slow / base:.3f}x")
    print(f"fp8_fast    / fp8_dequant = {fast / slow:.3f}x")
    speed_ok = ratio >= PASS_RATIO
    print(f"\nspeed criterion: {'PASS' if speed_ok else 'FAIL'}")
    print("quality criterion: MANUAL -- A/B these at the same seed and confirm "
          "no visible degradation:")
    print(f"  reference (what users run today): {os.path.join(IMAGE_DIR, 'fp8_dequant')}")
    print(f"  candidate  (fast path):           {os.path.join(IMAGE_DIR, 'fp8_fast')}")
    print("\nBoth criteria must pass before SUSHI_FP8_SCALED_MM is made the default.")
    return 0 if speed_ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--source", help="checkpoint path/id as /models/load expects it")
    ap.add_argument("--source-type", default="diffusers",
                    choices=SOURCE_TYPES,
                    help="source_type for POST /models/load. Ideogram 4 is a "
                         "diffusers directory; a Krea 2 single file is "
                         "'safetensors'. (default: diffusers)")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                    help="actually load the model and run generations (heavy, GPU-mutating)")
    ap.set_defaults(dry_run=True)
    args = ap.parse_args()

    if args.report:
        return report()
    if not args.arm or not args.source:
        ap.error("--arm and --source are required unless --report is given")

    if args.dry_run:
        root = base_url()
        print("=== DRY RUN (no request sent) ===")
        print(f"Would POST {root}/api/v1/models/load "
              f"data={{'source_type': {args.source_type!r}, 'source': {args.source!r}}}")
        print(f"Would POST {root}/api/v1/generate/txt2img "
              f"({WARMUP} warmup + {RUNS} timed) data=")
        print(json.dumps(form_data(), indent=2))
        print(f"Would save images to {os.path.join(IMAGE_DIR, args.arm)}")
        print(f"Would record results in {RESULTS}")
        print("\nThis loads a model and runs real generations on the GPU. "
              "Re-run with --no-dry-run when the GPU is idle AND the backend was "
              "restarted with the SUSHI_FP8_SCALED_MM value this arm requires.")
        return 0

    run_arm(args.arm, args.source_type, args.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
