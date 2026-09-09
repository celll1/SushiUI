# Training equivalence refactor audit and plan

Date: 2026-09-10

## Scope

This pass covers `backend/core/training/` and its direct tests. It is limited to
refactors whose observable training configuration and runtime behavior can stay
unchanged. Numerical kernels, loss formulas, optimizer updates, architecture
contracts, API parameters, and defaults are out of scope.

## Findings and evidence

### 1. Disconnected aesthetic-loss prototype

`backend/core/training/aesthetic_loss.py` has no importer or call site in the
repository. Its only external references are in the convergence-acceleration
documents, which identify it as a non-differentiable example because its scorer
forward runs under `torch.no_grad()`. The file also contains a 60-line inert
triple-quoted integration example. Removing this module changes no reachable
runtime path and removes both dead code and redundant commentary.

### 2. Repeated configuration-generator plumbing

`TrainingConfigGenerator` repeats the same parameter merge/normalization,
mutually-exclusive `total_steps`/`epochs` validation, and dataset-entry assembly
across LoRA, full-parameter, ControlNet, and VAE generation. The copies already
have only two intentional variations: LoRA dataset entries include a resolution,
and VAE entries use `VAE_TRAINING_DEFAULTS["resolution"]`. Extracting small pure
helpers can preserve insertion order, truthiness rules, error text, and output
YAML while reducing drift risk.

### 3. Repeated architecture detection

LoRA and full-parameter configuration generation call `_detect_arch()` three or
four times for the same `base_model_path`; ControlNet calls it three times. Each
call can invoke checkpoint detection and exception fallback. Resolve the value
once per top-level generator and pass it to sample-default, train-section, and
sample-section builders. This preserves the result for deterministic detection
while avoiding repeated checkpoint inspection.

## Implementation units

Each unit is committed independently after this plan commit.

1. Remove the disconnected aesthetic-loss prototype and update the two current
   convergence documents so they record the removal rather than link to a
   nonexistent implementation.
2. Extract and test pure configuration helpers for parameter preparation,
   steps/epochs validation, and dataset-entry assembly. Replace the existing
   copies without changing serialized YAML.
3. Detect architecture once per configuration generation and add a regression
   test that pins one detector call and the generated architecture-dependent
   values.

## Verification

- Search the full repository before deleting the dead module and after updating
  its documentation references.
- Add focused regression tests for helper edge cases and serialized output.
- Run the relevant training-configuration tests, `py_compile` on every changed
  backend file, and a real import with CUDA initialization stubbed as required by
  `AGENTS.md`.
- Inspect each staged diff and keep commits scoped to one implementation unit.

At audit time, `venv/Scripts/python.exe` cannot start because its configured base
interpreter (`Python311/python.exe`) is absent. Code changes may proceed, but the
Python verification commands above remain mandatory before this pass is declared
complete; if the environment is not repaired during the work, the exact blocked
commands will be reported.

## Equivalence boundary

Deletion is limited to code with no repository call path. Helper extraction must
preserve mapping key order, defaults, omission rules, legacy-key precedence, and
exception messages. Architecture detection is treated as a pure function of the
model path, matching every existing caller's assumption.
