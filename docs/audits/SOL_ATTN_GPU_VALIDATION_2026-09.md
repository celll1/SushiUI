# Sol-Attn GPU validation (2026-09-12)

## Scope and result

The pinned official `sol-attn` 0.5.0 package and SushiUI's MiniMax-H3 adapter pass
their executable GPU gates on an NVIDIA RTX 6000 Ada Generation (SM89, 48 GB),
driver 581.94, PyTorch 2.10.0+cu130, and FlashAttention 2.8.3.

The package selected its portable Triton backend. The optional SM89 CuTe runtime
could not be resolved by pip on this Windows environment because the matching
`nvidia-cutlass-dsl-libs-base` distribution was unavailable. This is not a Sol-Attn
failure: the official dispatcher intentionally falls back to Triton, and all tests
below ran through that fallback. Do not add unresolved CuTe packages to the base or
experimental requirements.

## Correctness coverage

`backend/tests/sol_attention_gpu_test.py` passed 9 tests:

- full-sink output versus dense SDPA at token lengths 63, 64, 65, 127, 128, and
  129, covering both sides of the 64-token routing boundary;
- sparse output shape, contiguity, dtype, and finite-value checks;
- exact dense restoration of MiniMax-H3 conditioning-prefix query rows;
- a three-block instance of the real vendored H3 transformer, including stable
  layer stamping, dense-layer policy, sparse layers, and the custom block-loop path.

These tests deliberately do not assert that approximate sparse output equals dense.
Full-sink and prefix rows are exact within BF16 tolerance; target-video rows are an
explicit approximation.

Run the suite with:

```powershell
venv\Scripts\python.exe -m pytest backend/tests/sol_attention_gpu_test.py -q
```

## Synthetic steady-state measurements

The standalone probe excludes first-call compilation through warm-up and reports
median kernel wall time and PyTorch peak allocated memory. Inputs are random BF16
Q/K/V, so approximation error is a plumbing diagnostic, not a model-quality metric.

| Shape `[B,T,H,D]` | Dense | Sol | Speedup | Dense peak | Sol peak |
|---|---:|---:|---:|---:|---:|
| `[1,8192,8,128]` | 2.176 ms | 0.882 ms | 2.47x | 80 MiB | 96.5 MiB |
| `[1,16384,24,128]` | 41.269 ms | 9.369 ms | 4.40x | 480 MiB | 579 MiB |

On these shapes Sol-Attn reduces attention compute time but uses more temporary
allocated memory than PyTorch SDPA. End-to-end H3 peak VRAM can therefore only be
claimed after measuring the loaded checkpoint; this kernel result is not a VRAM
reduction claim.

Re-run or extend the measurement with:

```powershell
venv\Scripts\python.exe -m backend.core.attention.sol_gpu_probe --tokens 16384 --heads 24 --prefix-tokens 512 --tau 1.0 --threshold-type diag --warmup 2 --repeats 5
```

## Remaining release gate

Run fixed-seed dense/Sol pairs through all five MiniMax-H3 video endpoints after the
backend has been restarted by the owner. Cover short/default/long clips, representative
aspect ratios, block swap on/off, and output-head fusion on/off. Record complete
generation latency, peak allocated/reserved VRAM, compilation time, and observed
attention provenance. Review motion, prompt adherence, reference identity, temporal
consistency, audio intelligibility, and synchronization. Until that matrix passes,
`h3_sol_attn` remains opt-in and experimental.
