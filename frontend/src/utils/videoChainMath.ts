// Video frame-count and chain-length arithmetic over an architecture's declared
// constraints. Pure functions; the interfaces they read stay in api.ts, which
// re-exports these, so the type dependency runs one way only.

import type {
  ArchCapabilities,
  ChainContextCapability,
  VideoChainPlan,
  VideoConstraints,
} from "./api";

// The chain-context capability for the LOADED arch/variant pair, or undefined
// when the architecture cannot be chained (or the matrix is not loaded). The
// variant is the one the backend reports for the loaded checkpoint
// (`currentModelInfo.model_info.variant`), never a file name.
export const chainContextCapability = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  variant?: string | null
): ChainContextVariantCapability | undefined => {
  const entry = arch ? caps?.chain_context?.[arch] : undefined;
  if (!entry) return undefined;
  const key = (variant || "").trim().toLowerCase();
  return entry.variants?.[key] ?? entry;
};

// The `continuation_overlap_frames` values a `pinned_tail` continuation can be
// given on this arch/variant, ascending. A latent frame is conditioned or
// generated whole, so the candidate lengths are the cumulative sums of the
// arch's `latent_chunk_pattern` — derived from the served pattern through the
// SAME `latentGroupSpans` the inpaint range uses, never a second hardcoded list
// (the pattern CYCLES: MiniMax-H3's [1,4,4,4,4] gives 1, 5, 9, 13, 17, 18, ...
// — 17 is followed by 18, not 33) — kept inside the served
// [min, max] window. Both bounds come from the backend, including the measured
// floor that keeps a one-frame pin off the list; a client that offered a
// shorter one would only earn a 400.
export const chainContinuationOverlapLengths = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  variant?: string | null
): number[] => {
  const capability = chainContextCapability(caps, arch, variant);
  const max = capability?.chain_context_max_frames ?? 0;
  const min = capability?.chain_context_min_frames ?? 1;
  const pattern = arch ? caps?.video_constraints?.[arch]?.latent_chunk_pattern : undefined;
  if (!max || !pattern?.length) return [];
  return latentGroupSpans(pattern, max)
    .map(([, end]) => end)
    .filter((end) => end >= min && end <= max);
};

// Capability readers, moved to ./trainingCapabilities so this file keeps the
// client. Re-exported so every existing import site is unchanged.

// `[pixelStart, pixelEndExclusive]` per latent frame, for a clip of `frames`
// pixel frames. Mirrors the backend's `latent_frame_spans`
// (backend/api/generation_utils.py): the pattern is cycled, and on a valid clip
// length the spans tile the clip exactly. An empty pattern yields no spans,
// which is how "this arch declares no chunking" reaches the UI.
export const latentGroupSpans = (
  pattern: number[] | undefined,
  frames: number
): [number, number][] => {
  const spans: [number, number][] = [];
  if (!pattern || pattern.length === 0 || frames <= 0) return spans;
  let cursor = 0;
  for (let index = 0; cursor < frames; index += 1) {
    const width = pattern[index % pattern.length];
    // `Math.min` here trims a final span that would run past `frames`; the
    // backend's `latent_frame_spans` does not do this trim, so on an
    // off-grid `frames` this span could come out narrower than the
    // backend's. Currently unreachable in practice because
    // `videoTrimmedLengthValid` (InpaintPanel.tsx) blocks submit unless
    // `frames` is already a multiple of the pattern.
    spans.push([cursor, Math.min(frames, cursor + width)]);
    cursor += width;
  }
  return spans;
};

// The range the server would actually regenerate for a requested `[start, end)`:
// expanded OUTWARD to latent-group boundaries, never shrunk — the same rule
// `plan_video_inpaint_span` applies. Returns the request unchanged when the arch
// declares no chunking (the backend refuses that case with its own message).
export const snapRangeToLatentGroups = (
  spans: [number, number][],
  start: number,
  end: number
): { start: number; end: number } => {
  if (!spans.length || !(start < end)) return { start, end };
  const touched = spans.filter(([lo, hi]) => lo < end && hi > start);
  if (!touched.length) return { start, end };
  return { start: touched[0][0], end: touched[touched.length - 1][1] };
};

// The longest clip length this architecture accepts that is <= `frames`, or
// null when even its shortest clip is longer. This is what a temporal-inpaint
// UI trims DOWN to: the clip length itself must be on the grid there, and the
// backend refuses an off-grid length rather than snapping it (snapping would
// delete frames the caller asked to keep).
export const largestValidVideoFrameCount = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  frames: number
): number | null => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c || !Number.isFinite(frames)) return null;
  const ceiling = c.max_frames != null ? Math.min(frames, c.max_frames) : frames;
  const k = Math.floor((ceiling - c.frame_offset) / c.frame_multiple);
  const length = k * c.frame_multiple + c.frame_offset;
  return length >= c.min_frames ? length : null;
};

// The valid clip length ON THE GRID (`multiple*n + offset`) closest to
// `frames`, clamped into `[min_frames, max_frames]` first. Unlike
// `largestValidVideoFrameCount` (which only ever snaps DOWN, because a
// temporal-inpaint clip length must not silently grow past what the caller
// trimmed to), this is for a control that lets the user ask for ANY length —
// a slider/number box, not a trim target — so the nearest grid point in
// EITHER direction is the right answer, the same as how a drag handle lands
// on the nearest tick. Returns null on the same "arch unknown / matrix not
// loaded" condition its neighbours do.
export const nearestValidVideoFrameCount = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  frames: number
): number | null => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c || !Number.isFinite(frames)) return null;
  const kMin = Math.ceil((c.min_frames - c.frame_offset) / c.frame_multiple);
  const kMax = c.max_frames != null
    ? Math.floor((c.max_frames - c.frame_offset) / c.frame_multiple)
    : Infinity;
  if (kMax < kMin) return null;
  const kRaw = (frames - c.frame_offset) / c.frame_multiple;
  const k = Math.min(kMax, Math.max(kMin, Math.round(kRaw)));
  return k * c.frame_multiple + c.frame_offset;
};

// The valid clip length the BACKEND would produce for a requested one: the
// grid point at or above `frames`, clamped into the arch's producible range.
// This mirrors `TemporalSpec.snap_length` (backend/core/models/components/
// wiring.py) exactly, including its floor of `max(min_frames,
// min_decodable_frames)` and its silent clamp at `max_frames` -- so a panel
// can show the length a request will ACTUALLY come back as, before spending
// the generation to find out. It rounds UP where `nearestValidVideoFrameCount`
// rounds to whichever side is closer, because that is what the backend does;
// do not swap one for the other to save a helper.
export const snapUpValidVideoFrameCount = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  frames: number
): number | null => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c || !Number.isFinite(frames)) return null;
  const lo = Math.max(c.min_frames, c.min_decodable_frames);
  const kLo = Math.ceil((lo - c.frame_offset) / c.frame_multiple);
  let k = Math.max(Math.ceil((frames - c.frame_offset) / c.frame_multiple), kLo);
  if (c.max_frames != null) {
    k = Math.min(k, Math.floor((c.max_frames - c.frame_offset) / c.frame_multiple));
  }
  return k * c.frame_multiple + c.frame_offset;
};

// --- Opt-in video length chaining (frontend-only orchestration) ---
//
// A video architecture's `max_frames` is a SINGLE-INFERENCE limit
// (backend/core/models/components/wiring.py's TemporalSpec), not a hard wall:
// a clip longer than it can only be reached by chaining several requests
// together via POST /generate/outpaint/video's `extend_forward` placement,
// each one continuing from the previous segment's own output. This is never
// automatic (see CLAUDE.md / the panels that call it) because a continuation
// segment is conditioned on the BOUNDARY FRAME of the clip it continues from
// (plus, for ref2va/ia2v, original image references and an automatic video
// reference derived from the previous segment's end), not the rest of its
// content, while the SAME full-length prompt is resent unchanged on every
// segment — prompt adherence degrades across segment boundaries in a way a
// single inference does not have.
//
// The arithmetic below mirrors OutpaintPanel's own extend_forward handling
// (`preservedFrames + effectiveGeneratedFrames - sharedAnchorFrames`): the
// GENERATED span (not the request's `total_frames`) is what has to land on
// `max_frames`, because the preserved (already-produced) prefix is placed,
// not regenerated. `sharedAnchorFrames` is 1 for extend_forward with no
// bridge clip (the placement this feature always uses).
const VIDEO_CHAIN_ANCHOR_FRAMES = 1;

// The per-segment length chaining arithmetic should use: the caller-supplied
// `segmentFrames` (`chain_segment_frames`, user-settable, null/undefined =
// unset) when it is a positive finite number, otherwise the architecture's
// own single-inference cap (`max_frames`) when it still has one, otherwise
// no cap at all (`Infinity`, meaning "nothing to chain unless the user opts
// in").
//
// `chain_segment_frames` is independent client-side orchestration state, NOT
// a backend parameter -- the backend only ever sees the resulting
// `total_frames` on each independent request (see videoChain.ts's header).
// Its default (unset) intentionally falls back to `max_frames` rather than
// straight to `Infinity`: that keeps every architecture that still has a
// real single-inference wall (LTX-2.3) chaining automatically exactly as it
// did before this control existed, while fixing the regression this control
// was added to fix -- an architecture whose `max_frames` went null
// (MiniMax-H3; see `trained_max_frames`) no longer has ANY server-enforced
// wall, so with no explicit segment length from the user there is nothing to
// split on, and "raise the total, nothing splits" is correct there by
// default. Setting `chain_segment_frames` is what turns chaining into a
// voluntary choice on an uncapped architecture too (e.g. to keep every
// segment within the documented trained range for quality, even though the
// backend would accept one huge request).
const chainSegmentCap = (
  c: VideoConstraints | undefined,
  segmentFrames: number | null | undefined
): number => {
  if (segmentFrames != null && Number.isFinite(segmentFrames) && segmentFrames > 0) {
    return segmentFrames;
  }
  return c?.max_frames ?? Number.POSITIVE_INFINITY;
};

// The `total_frames` value to send to POST /generate/outpaint/video to
// continue a chain that has produced `accumulatedFrames` so far, aiming for
// `targetFrames` overall. Also, on success, the new accumulated frame count
// (the extend_forward output is exactly this many frames — preserved prefix
// plus the newly generated span, minus the one shared anchor frame). Returns
// null when there is no effective per-segment cap to chain against (see
// `chainSegmentCap`) or the segment would make no forward progress (guards
// against a pathological arch table looping forever).
export const nextVideoChainTotalFrames = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  accumulatedFrames: number,
  targetFrames: number,
  segmentFrames?: number | null
): number | null => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c) return null;
  const cap = chainSegmentCap(c, segmentFrames);
  if (!Number.isFinite(cap)) return null;
  const remaining = targetFrames - accumulatedFrames;
  if (remaining <= 0) return null;
  const requestedGenerated = Math.min(remaining, cap);
  const generatedSpan = snapUpValidVideoFrameCount(caps, arch, requestedGenerated);
  if (generatedSpan == null || generatedSpan <= VIDEO_CHAIN_ANCHOR_FRAMES) return null;
  return accumulatedFrames + generatedSpan - VIDEO_CHAIN_ANCHOR_FRAMES;
};

// The client-side plan for reaching `targetFrames` when the effective
// per-segment cap (`chainSegmentCap`: `segmentFrames` if the user set one,
// else the architecture's `max_frames`, else uncapped) is below it, using the
// same segment-by-segment arithmetic `nextVideoChainTotalFrames` applies at
// execution time.
//
// Returns null for three DIFFERENT reasons that all mean "nothing to plan",
// kept as separate early-returns (not folded into one condition) so each
// stays independently readable/greppable even though the caller only ever
// sees "no plan":
//   1. the arch/matrix is unknown or `targetFrames` is not a real number;
//   2. there is no effective segment cap at all (uncapped arch, no
//      `segmentFrames` set) -- nothing CAN be chained, by design;
//   3. `targetFrames` already fits inside one segment -- chaining is not
//      NEEDED.
// A caller that must tell these apart (e.g. to phrase "nothing to chain
// automatically -- set a segment length" vs "already fits") calls
// `chainSegmentCap`-derived logic itself; the Generate-time gate
// (`if (chainPlan != null)`) only ever needed the null/non-null distinction,
// which this preserves.

export const planVideoChain = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  targetFrames: number,
  segmentFrames?: number | null
): VideoChainPlan | null => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c || !Number.isFinite(targetFrames)) return null;
  const cap = chainSegmentCap(c, segmentFrames);
  if (!Number.isFinite(cap)) return null;
  if (targetFrames <= cap) return null;

  let accumulated = cap; // segment 1: a normal request, at the segment cap
  let segments = 1;
  // Bounded so a pathological arch table can never loop forever; 500
  // segments is far beyond anything this feature would reasonably run.
  for (let guard = 0; guard < 500 && accumulated < targetFrames; guard++) {
    const next = nextVideoChainTotalFrames(caps, arch, accumulated, targetFrames, segmentFrames);
    if (next == null) break;
    accumulated = next;
    segments += 1;
  }
  return { capFrames: cap, segments, finalFrames: accumulated };
};

// Per-CONTINUATION-segment `total_frames` values for reaching `targetFrames`,
// i.e. everything `planVideoChain` above computes except segment 1 itself
// (which is always a plain request at the effective segment cap). Used to
// give the queue items for segments 2..N a real initial `total_frames` at
// enqueue time, so the whole plan is visible in the queue immediately -- each
// item's value is still re-derived from the ACTUAL previous segment's
// reported frame count right before that item runs (see videoChain.ts),
// because a real generation can snap slightly differently than this
// pre-flight estimate. Returns null under the same "nothing to chain"
// conditions as `planVideoChain`.
export const planVideoChainSegments = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  targetFrames: number,
  segmentFrames?: number | null
): number[] | null => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c || !Number.isFinite(targetFrames)) return null;
  const cap = chainSegmentCap(c, segmentFrames);
  if (!Number.isFinite(cap)) return null;
  if (targetFrames <= cap) return null;

  const segments: number[] = [];
  let accumulated = cap;
  for (let guard = 0; guard < 500 && accumulated < targetFrames; guard++) {
    const next = nextVideoChainTotalFrames(caps, arch, accumulated, targetFrames, segmentFrames);
    if (next == null) break;
    segments.push(next);
    accumulated = next;
  }
  return segments;
};

// The length of clip ANY SINGLE generation request in a chain (or a plain,
// unchained request) can actually produce: `requestedFrames` itself when it
// already fits in one segment, otherwise the effective per-segment cap
// (`chainSegmentCap`: the user's `chain_segment_frames` if set, else the
// architecture's single-inference cap, else uncapped). Nothing this feature
// ever sends to a generation endpoint -- segment 1 of a chain, a `Generate at
// cap` single inference, or an unchained request -- is longer than this, so
// it is what a per-segment duration (H3 Prompt Assist) must be computed
// from, never `requestedFrames` itself. Falls back to `requestedFrames` when
// the arch/matrix is unknown, the same "assume supported" convention
// `archSupportsFeature` uses.
export const effectiveSegmentFrames = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  requestedFrames: number,
  segmentFrames?: number | null
): number => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c) return requestedFrames;
  const cap = chainSegmentCap(c, segmentFrames);
  if (!Number.isFinite(cap)) return requestedFrames;
  return Math.min(requestedFrames, cap);
};

// --- Backend video-chain planner (POST /video-chain/plan, /video-chain/validate) ---
//
// Types transcribed from `openapi.yaml`'s `VideoChain*` schemas, which are the
// contract. The chain-length helpers ABOVE stay: they are what the queue is
// still built from, and they are the parity reference the backend planner is
// ported against, so both exist until the migration is finished.
//
// Frame ranges are half-open `[owned_start_frame, owned_end_frame)` in integer
// GLOBAL frames; seconds anywhere in this feature are display-only.

