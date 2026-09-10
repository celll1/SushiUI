# SenseNova zero block-swap fix

## Failure

`FullParameterTrainer` calls every architecture's `setup_block_swap()` after
installing its adapter. SenseNova's implementation raised unconditionally, so
even the supported `blocks_to_swap=0` configuration failed after the model and
checkpoint had loaded. Activation dispatch is independent of this hook and did
not cause the failure.

## Change

1. Make SenseNova block-swap setup a no-op when `blocks_to_swap` is zero.
2. Preserve an explicit refusal for positive values because branch-aware block
   streaming remains unimplemented.
3. Add a focused regression test for both outcomes and trim the stale test
   comment that documented the unconditional exception.
4. Run the focused CPU tests, compile the changed backend file, and import the
   handler with CUDA initialization stubbed.
