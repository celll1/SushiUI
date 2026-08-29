# MiniMax Music 3 integration contract

MiniMax Music 3 is an inference-only audio architecture in SushiUI. This guide
records the shipped capability and state contracts; it is not a rollout plan.

## Capability boundary

| Capability | Status |
|---|---|
| Text and lyrics to music | Supported |
| Negative prompt | Refused; the released pipeline has no such input contract |
| Reference-audio conditioning for voice, timbre, or instrument | Refused; the released RVQ encoder needed to produce semantic codes is unavailable |
| Extend a SushiUI-generated song forward | Supported through stored frame-code replay |
| Regenerate from a time onward | Supported |
| Re-render a range while retaining semantic codes | Supported |
| Mid-song infill with a preserved tail | Refused by the causal language-model contract |
| Training | Not registered; generation-only architecture |

`prompt` describes the music and `lyrics` carries the structured lyric text.
`audio_duration` is an upper bound because the model may emit its end token
early. API defaults are owned only by `backend/api/param_defaults.py`.

## Continuation state

Every generated song that may later be extended or repainted needs its frame-code
sidecar. The sidecar stores compact RVQ frame codes plus the rates and generation
inputs required to interpret them. Hidden states are reconstructed with a
teacher-forced replay; they are not persisted.

The frame-code sidecar, not a seed, is the cross-layout reproduction contract.
Full-vocabulary and pruned-vocabulary language-model layouts can consume random
numbers differently during categorical sampling even when their restricted
logits agree. Existing full-vocabulary seed behavior is therefore not rewritten
to make seeds portable between layouts.

## Weight formats

The loader supports the released component tree and the flat formats explicitly
recognized by the production loader. GGUF container support validates tensor
types before materialization; unsupported quantized tensor types are refused.
Do not infer support for a file merely from its extension.

The vendored model code is tied to the upstream Diffusers revision recorded in
`docs/legal/THIRD_PARTY_PROVENANCE.md`. Model-weight terms are separate from the
Apache-2.0 code terms and must be reviewed independently.
