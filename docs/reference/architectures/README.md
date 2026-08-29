# Architecture structure reference

One file per architecture, describing what the modules are, how tensors move
through them, and where the repository's cross-cutting machinery attaches. This
is the reference to read before designing against an architecture or modifying
one.

Every file follows the same eight sections, so the same question is always in
the same place:

| Section | Answers |
|---|---|
| Components | Which classes the loader builds, and which are vendored |
| Load path | Which checkpoint layouts are accepted, how they are told apart, what is refused |
| Denoiser structure | A diagram of the block stack with one block expanded, and where each conditioning signal enters |
| Tensor contract | Latent space, embedding dimensions, positional encoding, timestep convention, prediction target |
| Generation path | The pipeline backend, the sampling loop, the CFG shape |
| Training path | The adapter and arch handler, trainable parameters, LoRA targets, refused combinations |
| Hook points | Attention conduit, block swap, FBCache, quantized Linear swap, keep-hot, activation dispatch |
| Constraints | Structural limits the code enforces, each with the symbol that enforces it |

## Architectures

| Key | Modality | File |
|---|---|---|
| `sd15` | image | [Stable Diffusion 1.5](sd15.md) |
| `sdxl` | image | [Stable Diffusion XL](sdxl.md) |
| `zimage` | image | [Z-Image](zimage.md) |
| `flux2` | image | [FLUX.2 Klein](flux2.md) |
| `anima` | image | [Anima](anima.md) |
| `lens` | image | [Lens](lens.md) |
| `krea2` | image | [Krea 2](krea2.md) |
| `ideogram4` | image | [Ideogram 4](ideogram4.md) |
| `minit2i` | image | [MiniT2I](minit2i.md) |
| `sensenova` | image | [SenseNova U1.5](sensenova.md) |
| `ltx2` | video + audio | [LTX-2.3](ltx2.md) |
| `minimax_h3` | video + audio | [MiniMax-H3](minimax_h3.md) |
| `acestep` | audio | [ACE-Step 1.5](acestep.md) |
| `minimax_music3` | audio | [MiniMax Music 3](minimax_music3.md) |

The keys are the ones `ModelType` in `backend/core/model_loader.py` uses.
`ARCH_REGISTRY` in `backend/core/training/arch/__init__.py` carries the
training-capable subset, which excludes `minimax_music3`.

## Scope

These files describe STRUCTURE. Behavioral facts — what a parameter does, what
a feature costs, which combinations were measured — stay in
`docs/guides/MODEL_FACTS.md`. Repository layout and the shared subsystems stay
in `docs/guides/ARCHITECTURE_MAP.md`. The procedure for adding an architecture
is `docs/guides/ADD_A_MODEL_ARCHITECTURE.md`.

Claims here are sourced to a symbol rather than a line number. Statements that
were reasoned rather than read are marked as inferred, and numbers that only a
checkpoint could settle say so instead of guessing.
