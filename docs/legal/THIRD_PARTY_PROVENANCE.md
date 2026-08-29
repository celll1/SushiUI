# Third-party code provenance

This ledger covers source code copied or substantially adapted into SushiUI.
It is a provenance aid, not a declaration of SushiUI's own license. Model
weights may have different terms from their accompanying code; weight terms
are recorded separately and are not redistributed by this repository.

Common license texts currently verified for vendored code are retained under
`docs/legal/licenses/`. `Apache-2.0.txt` is the standard Apache License 2.0
text; `MIT-MiniT2I.txt` preserves MiniT2I's upstream copyright notice. Missing
package-specific MIT notices remain explicit open items in the table below.

Status meanings:

- **verified**: the upstream source and license are identified locally.
- **needs notice check**: the license family is known, but the exact upstream
  copyright or notice text still needs to be preserved locally.
- **needs file audit**: the package-level source is known, but every vendored
  file has not yet been mapped to a source revision and modification status.

| Local package | Upstream source and revision | Code license | Weight terms | Status / required follow-up |
|---|---|---|---|---|
| `backend/core/models/acestep/vendor/` | ACE-Step 1.5 custom model code distributed with `acestep-v15-turbo`; exact source revision is not yet recorded | Apache-2.0 | Verify at model acquisition | **needs file audit**; pin the exact upstream repository/revision and retain notices |
| `backend/core/models/ideogram4/vendor/` | Hugging Face Diffusers `transformer_ideogram4` plus local loader and quantization adaptations; exact Diffusers revision is not yet recorded | Apache-2.0 for identified Diffusers-derived files | Governed by the model provider; verify separately | **needs file audit**; pin the Diffusers revision and mark each modified file |
| `backend/core/models/krea2/vendor/` | Krea 2 reference `mmdit.py` and Hugging Face Diffusers `transformer_krea2`; exact revisions are not yet recorded | Apache-2.0 | Krea Community License | **needs file audit**; pin both revisions and review model-service obligations separately |
| `backend/core/models/lens/vendor/` | `https://github.com/dxqb/Lens`; exact revision is not yet recorded | MIT | Verify at model acquisition | **needs notice check**; preserve the upstream MIT copyright notice and map local modifications |
| `backend/core/models/minimax_h3/vendor/` | Hugging Face Diffusers MiniMax-H3 work from PRs 14355 and 14371, retrieved 2026-08-05 | Apache-2.0 | MiniMax-H3 Community License | **needs file audit**; replace PR-only provenance with an immutable commit and retain notices |
| `backend/core/models/minimax_h3/hybrid_reader.py` and related hybrid selection code | `https://github.com/scottmudge/ComfyUI_MinimaxH3HybridLoader`; exact revision and copied-function mapping are not yet recorded | MIT | Not applicable | **needs file audit**; notice is retained in `licenses/MIT-MiniMaxH3-HybridLoader.txt`, but adapted versus independent portions still need mapping |
| `backend/core/models/minimax_music3/vendor/` | Hugging Face Diffusers PR 14456, commit `dafe3733e35df3e2ba829b1e29244d6a1476c6d2` | Apache-2.0 | MiniMax Music 3 terms; verify separately | **needs file audit**; map files and modifications to the recorded commit |
| `backend/core/models/minit2i/vendor/` | `https://github.com/Hope7Happiness/minit2i-torch`; exact revision is not yet recorded | MIT | Verify at model acquisition | **needs notice check**; preserve `Copyright (c) 2026 MiniT2I contributors` and pin the source revision |
| `backend/core/models/sensenova/vendor/` | `https://github.com/OpenSenseNova/SenseNova-U1`, branch `feat/u1.5`, commit `a1ce053d25835e0785a0869ca1c97e717212ef64` | Apache-2.0 | Apache-2.0 for the targeted SenseNova U1.5 release | **verified at package level**; keep per-file modified/unmodified markers and audit notice completeness |
| `backend/core/inference/fbcache.py` and H3 FBCache wiring | `chengzeyi/Comfy-WaveSpeed`, `chengzeyi/ParaAttention`, and `duckyshell/ComfyUI-MiniMaxH3-FirstBlockCache`; exact source revisions are not yet pinned | MIT | Not applicable | **needs file audit**; notices are retained in `licenses/MIT-FBCache.txt`, but adapted functions still need revision mapping |
| `backend/core/inference/reference_style.py` and architecture hooks | `jieg9341-lab/ComfyUI-Krea2-StyleTransfer`; exact source revision is not yet pinned | MIT | Not applicable | **needs file audit**; notice is retained in `licenses/MIT-Krea2-StyleTransfer.txt`, but adapted functions still need revision mapping |

## Redistribution gate

Before distributing a source bundle or binary that contains a package marked
`needs notice check` or `needs file audit`:

1. resolve the package to an immutable upstream revision;
2. compare every local file with that revision and record whether it is copied,
   adapted, or independently implemented;
3. retain all upstream copyright and attribution notices required by the
   license;
4. include the full applicable license text and any upstream `NOTICE` file;
5. review model-weight and hosted-service terms independently from code terms.

Clean-room claims require contemporaneous evidence of separation between the
person who studies the reference and the person who writes the implementation.
A design note saying "clean room" is not sufficient. If that evidence does not
exist, describe the code as adapted and comply with the upstream license until
the provenance is resolved.
