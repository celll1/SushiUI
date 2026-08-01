# FP8 Quantization Tool

Converts a bf16 transformer checkpoint into the repo's **weight-only FP8**
layout — the one `backend/core/models/ideogram4/vendor/fp8_linear.py` defines
and every FP8-aware loader (Ideogram 4, Krea 2) already reads.

The output loads through the **normal production path** (`POST /models/load`);
nothing in the backend needs changing.

## Why

To measure the opt-in FP8 W8A8 `torch._scaled_mm` fast path, an FP8 arm has to
be compared against a bf16 arm of the *same* architecture on the *same*
hardware. Krea 2 ships bf16 locally and is a single transformer that fits VRAM,
so it is the speed vehicle for gate G1; this tool produces its matched FP8 arm.
See `examples/api/bench_fp8_scaled_mm.py` for the protocol and the
pre-registered decision rule.

## Format produced

Per quantized `nn.Linear` named `<name>`:

```
<name>.weight        float8_e4m3fn  (out, in)
<name>.weight_scale  float32        (out,)     <- presence gates the FP8 swap
<name>.bias          original dtype (out,)     [untouched]
```

Dequantization is `weight.to(dtype) * weight_scale[:, None]`. The quantization
itself calls the repo's own `fp8_linear.quantize_weight_to_fp8`, not a
reimplementation, so the arms differ only in weight format and GEMM path.

Everything that is not a quantized Linear weight — norms, embeddings, biases,
modulation tables, non-Linear parameters — is copied through unchanged in its
original dtype.

Output is written as sushiUI single-file shards plus a
`<stem>.safetensors.index.json`, with `model_type` metadata, so the loader
detects the architecture from the index without a filename convention.

## Which Linears get quantized

Every `nn.Linear` in the model **except** those whose `in_features` or
`out_features` is not a multiple of `--min-align` (16). `Fp8Linear`'s scaled-GEMM
path rejects unaligned shapes outright, so quantizing such a layer would add
error for zero possible speed. For Krea 2 that excludes exactly one layer
(`text_fusion.projector`, 12×1) and quantizes 263 of 264.

"All Linears" is the convention of the reference FP8 checkpoint this format
comes from (`ideogram-4-fp8` quantizes the input/output projections and the
timestep MLP too), not an aggressive choice. Narrow it further with `--exclude`
(repeatable regex against the module path).

Module paths come from instantiating the architecture on the **meta** device, so
enumeration costs no memory even for a 13 B-parameter model, and the set of
quantized keys is exactly the set `swap_linears_to_fp8` will swap.

## Usage

```
venv/Scripts/python.exe subapps/fp8_quantize/quantize_transformer_fp8.py \
    --arch krea2 \
    --source "<bf16 shard index / safetensors / diffusers dir>" \
    --output "<scratch dir>/krea2_fp8/krea2_fp8.safetensors" \
    --link-siblings "<bf16 model dir>"
```

Add `--dry-run` first: it prints how many Linears would be quantized and why
each skipped one was skipped, and writes nothing.

**Write the output to a scratch location, not under a `M:/model/<arch>/` root.**
Those roots hold the vanilla checkpoints, and their sibling directories are
completion sources for the loaders.

`--link-siblings SRC` creates directory junctions (`mklink /J`, no admin rights;
symlinks on POSIX) for `text_encoder` / `vae` / `tokenizer` / `scheduler` next
to the output. The loaders resolve those components by probing siblings of the
checkpoint, so without the links a scratch-located checkpoint would fall back to
a hub download and would no longer be a matched arm.

Source and destination are streamed shard-by-shard; peak RAM is roughly one
input shard plus one output shard, not the whole model.

## Adding an architecture

`ARCHS` in the script maps an arch name to four things: the key prefix its
single-file loader expects, a config resolver, a meta-device model builder, and
the metadata block to write. Add one entry; nothing else changes.
