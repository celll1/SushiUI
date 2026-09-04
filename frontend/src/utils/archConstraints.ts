// Reading an architecture's declared constraints: which frame counts and canvas
// sizes it accepts, how it names itself, and which quantization modes it takes.
// Pure functions over the served capability payload -- no request, no client --
// so they sit beside api.ts, which re-exports them.
//
// Types arrive through `import type`, erased by TypeScript, so nothing here
// depends on api.ts at runtime.

import type { ArchCapabilities } from "./api";

// True when `frames` is a length the architecture really accepts: on the grid
// (`multiple * n + offset`) and inside the production range. `suggested_frames`
// is only a SUBSET of these (it is capped, and LTX-2.3 omits lengths it accepts
// but does not advertise), so this is what decides whether a value the user
// already holds may stay.
export const isValidVideoFrameCount = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  frames: number | null | undefined
): boolean => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c || frames == null || !Number.isFinite(frames)) return false;
  if (frames < c.min_frames) return false;
  if (c.max_frames != null && frames > c.max_frames) return false;
  const k = (frames - c.frame_offset) / c.frame_multiple;
  return Number.isInteger(k) && k >= 0;
};

// Same grid test as `isValidVideoFrameCount`, but WITHOUT the `max_frames`
// ceiling -- for callers that must not treat "above the single-inference cap"
// as "invalid", because the video-length chaining feature makes that a
// legitimate value to hold (see the opt-in chaining section below).
// `isValidVideoFrameCount` stays strict (single-inference requests, e.g.
// the temporal-inpaint trim target in InpaintPanel, really do need <= max_frames).
export const isOnGridVideoFrameCount = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  frames: number | null | undefined
): boolean => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c || frames == null || !Number.isFinite(frames)) return false;
  if (frames < c.min_frames) return false;
  const k = (frames - c.frame_offset) / c.frame_multiple;
  return Number.isInteger(k) && k >= 0;
};

// The clip-length <Select> options for the loaded video arch, from the backend's
// own valid-length rule. Falls back to LTX-2.3's historical hardcoded list only
// when the matrix has not loaded (or the arch is unknown), so the offered
// lengths are never a second copy of a rule the backend owns.
//
// `current` — the value the control is bound to. A <select> renders ONLY the
// options it is handed, so a current value missing from the list makes the
// control render BLANK while the panel keeps sending that value. If it is a
// length this architecture accepts it is merged in (in order); if it is not,
// normalizeVideoFrames() below is what replaces it.
export const videoFrameOptions = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  current?: number | null
): { value: string; label: string }[] => {
  const suggested = arch ? caps?.video_constraints?.[arch]?.suggested_frames : undefined;
  const lengths = suggested?.length ? [...suggested] : [9, 17, 25, 33, 49, 65, 81, 97, 121];
  if (current != null && !lengths.includes(current)) {
    // Unknown arch / matrix not loaded: keep the value rather than blanking the
    // control, the same "assume supported" convention as archSupportsFeature.
    const known = !!(arch && caps?.video_constraints?.[arch]);
    if (!known || isValidVideoFrameCount(caps, arch, current)) {
      lengths.push(current);
      lengths.sort((a, b) => a - b);
    }
  }
  return lengths.map((n) => ({ value: String(n), label: String(n) }));
};

// The clip length to hold after the loaded architecture changed: the current
// value when that architecture accepts it, otherwise the NEAREST offered length
// (ties go up). Mirrors normalizeUnetQuantization: a value carried over from
// another architecture -- LTX-2.3's 121 onto MiniMax-H3, whose grid starts at
// 124 -- would otherwise sit in the control unselectable and be sent anyway,
// only to be snapped server-side with a warning.
// Uses `isOnGridVideoFrameCount`, NOT `isValidVideoFrameCount`: a value ABOVE
// `max_frames` that is still on the frame grid is the opt-in entry point for
// video-length chaining (see below) and must survive a mount / model-change
// pass, not get silently snapped back down to a suggested in-cap length --
// that used to happen on every remount and every model reload, discarding the
// user's chosen target with no notice (see VideoChainConfirmDialog / the
// panels' Generate-time chain prompt for where the choice is actually made).
export const normalizeVideoFrames = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  frames: number | null | undefined
): number | null => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c || frames == null) return frames ?? null;
  if (isOnGridVideoFrameCount(caps, arch, frames)) return frames;
  const offered = c.suggested_frames?.length ? c.suggested_frames : null;
  if (!offered) return frames;
  return offered.reduce((best, n) =>
    Math.abs(n - frames) < Math.abs(best - frames) ? n : best, offered[0]);
};

// Label for that control, stating the arch's own rule ("17n+5, 124-362")
// rather than a hardcoded "8k+1". Always states the floor, even when there is
// no hard ceiling: `max_frames == null` used to drop the whole range clause,
// losing the floor hint too. When the arch declares no hard ceiling but does
// have a `trained_max_frames` (advisory-only, e.g. MiniMax-H3's 362), that is
// shown as a "+" open range rather than a bound, since going past it is
// documented-untested, not invalid.
export const videoFrameLabel = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined
): string => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  if (!c) return "Frames";
  const rule = c.frame_offset === 0
    ? `${c.frame_multiple}n`
    : `${c.frame_multiple}n+${c.frame_offset}`;
  let range: string;
  if (c.max_frames != null) {
    range = `, ${c.min_frames}-${c.max_frames}`;
  } else if (c.trained_max_frames != null) {
    range = `, ${c.min_frames}+ (trained to ${c.trained_max_frames})`;
  } else {
    range = `, ${c.min_frames}+`;
  }
  return `Frames (${rule}${range})`;
};

// The alignment both spatial axes must land on for `arch`. An arch the matrix
// does not describe (or a matrix that has not loaded) falls back to 32 — the
// same "assume supported" convention as archSupportsFeature, with the backend
// re-validating regardless. Single definition so the canvas fitter, the rule
// sentence and the slider bounds can never disagree about the grid.
const videoPixelAlign = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined
): number => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  return c?.pixel_align && c.pixel_align > 0 ? c.pixel_align : 32;
};

/**
 * The NEAREST CANVAS THIS ARCHITECTURE ACCEPTS to `srcWidth x srcHeight`
 * scaled by `scale`, plus why it differs from that when it does.
 *
 * Video generation does not take an arbitrary size: both axes round to
 * `pixel_align`, and an architecture may cap the canvas envelope
 * (`max_pixel_hw` = [short edge, long edge], orientation-agnostic). So
 * "generate at the input clip's own resolution" is often not literally
 * reachable — a 1920x1080 clip cannot be, on MiniMax-H3: 1080 is not a
 * multiple of 32 and 1920 is past the 1344 long-edge policy cap.
 *
 * The aspect ratio is preserved as closely as the grid allows: the cap is
 * applied as a single uniform factor to BOTH axes before rounding, so a capped
 * canvas is a scaled-down clip rather than a squashed one. Whatever aspect
 * mismatch the rounding leaves is resolved by the backend's
 * `center_crop_resize_frames`, which CENTRE-CROPS to the target aspect — it
 * does not letterbox — so callers should surface `cropped` to the user.
 *
 * An unknown architecture (or a capability matrix that has not loaded) gets
 * `pixel_align` 32 and no cap: the same "assume supported" convention as
 * archSupportsFeature, with the backend re-validating regardless.
 */
export const fitVideoCanvas = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  srcWidth: number,
  srcHeight: number,
  scale: number = 1
): { width: number; height: number; matchesSource: boolean; cropped: boolean } => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  const align = videoPixelAlign(caps, arch);
  const cap = c?.max_pixel_hw ?? null;

  let width = Math.max(1, srcWidth) * scale;
  let height = Math.max(1, srcHeight) * scale;

  // Uniform down-scale to fit the envelope (never an up-scale: the cap is a
  // ceiling, not a target).
  if (cap) {
    const [capShort, capLong] = cap;
    const shortEdge = Math.min(width, height);
    const longEdge = Math.max(width, height);
    const factor = Math.min(1, capShort / shortEdge, capLong / longEdge);
    width *= factor;
    height *= factor;
  }

  const round = (v: number) => Math.max(align, Math.round(v / align) * align);
  width = round(width);
  height = round(height);

  // Rounding can push an edge back over the cap (e.g. 756 -> 768 against a 768
  // short-edge cap is fine, but 1350 -> 1344 is not automatic). Step down to
  // the largest multiple of `align` that fits, per axis, using the ORIENTATION
  // OF THE SOURCE so the two caps are not swapped by a rounding tie. A square
  // source is bound by the short-edge cap on both axes, which is what
  // "short <= capShort AND long <= capLong" means for width == height.
  if (cap) {
    const [capShort, capLong] = cap;
    const floorTo = (v: number, limit: number) =>
      v <= limit ? v : Math.max(align, Math.floor(limit / align) * align);
    const widthIsLong = srcWidth > srcHeight;
    const heightIsLong = srcHeight > srcWidth;
    width = floorTo(width, widthIsLong ? capLong : capShort);
    height = floorTo(height, heightIsLong ? capLong : capShort);
  }

  const matchesSource = scale === 1 && width === srcWidth && height === srcHeight;
  // Aspect mismatch = the preprocessing discards content from the edges.
  const cropped =
    srcWidth > 0 && srcHeight > 0 &&
    Math.abs(srcWidth / srcHeight - width / height) > 1e-3;

  return { width, height, matchesSource, cropped };
};

/**
 * The canvas rule of the loaded video architecture, in words, for a UI that has
 * to explain why a requested size was not reachable. Reads the same
 * capability entry `fitVideoCanvas` does, so the two never disagree.
 */
export const videoCanvasRule = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined
): string => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  const align = videoPixelAlign(caps, arch);
  const cap = c?.max_pixel_hw ?? null;
  const alignRule = `both sides must be a multiple of ${align}`;
  if (!cap) return alignRule;
  return `${alignRule}, the short side is capped at ${cap[0]} and the long side at ${cap[1]}`;
};

// The ceiling of the Absolute width/height sliders where the loaded
// architecture declares NO envelope (LTX-2.3: `max_pixel_hw` null). It is a UI
// range, not an architecture fact -- the backend imposes no upper spatial bound
// there beyond `pixel_align` -- which is why it is a constant here instead of
// something read out of the capability matrix. It is also the historical range
// of those sliders, so an uncapped arch keeps exactly the reach it had.
const UNCAPPED_VIDEO_EDGE = 2048;

// Upper bound offered by the video routes' Block Swap number field
// (Txt2Img/Img2Img/Inpaint/Outpaint panels' video modes). The backend clamps
// `blocks_to_swap` to `num_blocks - 1` for whatever architecture is actually
// loaded (`core.memory_management.transformer_registry`), and there is no
// schema/capability endpoint that reports a loaded architecture's block
// count, so the frontend cannot derive this bound -- it is a defensible
// constant rather than a computed one. MiniMax-H3, the deepest video
// architecture wired today, has 50 transformer blocks (49 swappable, since
// at least one block must stay resident), so 49 is used here: LTX-2.3 has
// fewer blocks and the backend clamp still applies if a value above its own
// count is sent, so this constant only needs to not undershoot the largest
// loaded architecture.
export const VIDEO_BLOCK_SWAP_MAX = 49;

/**
 * The bounds one Absolute canvas slider may offer, given where the OTHER axis
 * currently sits.
 *
 * `max_pixel_hw` is `[short edge, long edge]` and the backend
 * (`validate_video_geometry`) compares it ORIENTATION-AGNOSTICALLY: a canvas is
 * legal when `min(w,h) <= short_cap` AND `max(w,h) <= long_cap`. So there is no
 * such thing as a fixed per-axis maximum. A single cap of `long_cap` on both
 * axes would offer the illegal 1344x1344; a single cap of `short_cap` on both
 * would forbid the perfectly legal 1344x768. The reachable maximum for THIS
 * axis is therefore a function of the other one:
 *
 *   other <= short_cap  -> this axis may be the long edge   -> long_cap
 *   other >  short_cap  -> the other axis is already the long edge, so this one
 *                          must be the short edge            -> short_cap
 *
 * which makes both 1344x768 and 768x1344 reachable and 1344x1345 not. When the
 * other axis is itself past `long_cap` (a value carried over from an uncapped
 * architecture) no value of this axis can make the pair legal; the tightest
 * bound is returned and `videoCanvasExceedsEnvelope` is what tells the user the
 * canvas is out of range.
 *
 * `min`/`step` are the arch's `pixel_align`, so the slider cannot land off-grid
 * either. An unknown arch (or a matrix that has not loaded) gets align 32 and
 * no cap: the same "assume supported" convention as archSupportsFeature.
 */
/**
 * The loaded video architecture's own floor on `num_inference_steps`, for the
 * step slider's `min`.
 *
 * This is a CORRECTNESS bound, not a UI one: `validate_video_steps`
 * (backend/api/generation_utils.py) answers 400 below it. MiniMax-H3 declares
 * 2 -- its step count is a sigma GRID POINT count, so N drives N-1 model
 * evaluations and 1 evaluates nothing -- while LTX-2.3 declares 1. Three of the
 * four video panels hardcoded `min={1}`, which let the user pick a value that
 * could only ever come back as a 400; this exists so the fallback rule lives in
 * one place instead of being re-derived per panel.
 *
 * An unknown arch (or a matrix that has not loaded) gets 1, matching the
 * "assume supported, let the backend re-validate" convention used elsewhere
 * here -- the request is still checked server-side either way.
 */
export const videoMinInferenceSteps = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined
): number => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  return c?.min_inference_steps ?? 1;
};

export const videoCanvasAxisBounds = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  otherEdge: number | null | undefined
): { min: number; max: number; step: number; capped: boolean } => {
  const c = arch ? caps?.video_constraints?.[arch] : undefined;
  const align = videoPixelAlign(caps, arch);
  const cap = c?.max_pixel_hw ?? null;
  // Floor the ceiling onto the alignment grid so the slider's top value is one
  // the backend actually accepts, the same way fitVideoCanvas steps down.
  const onGrid = (v: number) => Math.max(align, Math.floor(v / align) * align);
  if (!cap) {
    return { min: align, max: onGrid(UNCAPPED_VIDEO_EDGE), step: align, capped: false };
  }
  const capShort = Math.min(cap[0], cap[1]);
  const capLong = Math.max(cap[0], cap[1]);
  const other = otherEdge != null && Number.isFinite(otherEdge) ? otherEdge : 0;
  return {
    min: align,
    max: onGrid(other <= capShort ? capLong : capShort),
    step: align,
    capped: true,
  };
};

// True when `width x height` is outside the loaded arch's envelope — the exact
// comparison validate_video_geometry makes, so the panel's warning and the
// server's 400 agree. No envelope (LTX-2.3) or an unknown arch = never outside.
// The alignment rule is deliberately NOT folded in: the sliders' `step` already
// keeps both axes on the grid, whereas the envelope can be violated by a value
// that was legal on the architecture the user just switched away from.
export const videoCanvasExceedsEnvelope = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  width: number | null | undefined,
  height: number | null | undefined
): boolean => {
  const cap = (arch ? caps?.video_constraints?.[arch] : undefined)?.max_pixel_hw ?? null;
  if (!cap || width == null || height == null) return false;
  if (!Number.isFinite(width) || !Number.isFinite(height)) return false;
  return (
    Math.min(width, height) > Math.min(cap[0], cap[1]) ||
    Math.max(width, height) > Math.max(cap[0], cap[1])
  );
};

// Human-readable architecture names. Used where a model's architecture is shown
// to the user; MiniMax H3's entry also carries its required attribution.
const ARCH_DISPLAY_NAMES: Record<string, string> = {
  sd15: "Stable Diffusion 1.5",
  sdxl: "SDXL",
  zimage: "Z-Image",
  flux2: "FLUX.2",
  krea2: "Krea 2",
  lens: "Lens",
  anima: "Anima",
  minit2i: "MiniT2I",
  ideogram4: "Ideogram 4",
  ltx2: "LTX-2.3",
  acestep: "ACE-Step 1.5",
  minimax_h3: "MiniMax H3",
  minimax_music3: "MiniMax Music 3",
};

export function archDisplayName(arch: string | null | undefined): string;
export function archDisplayName(
  caps: ArchCapabilities | null | undefined,
  arch: string
): string;
export function archDisplayName(
  capsOrArch: ArchCapabilities | string | null | undefined,
  arch?: string
): string {
  if (typeof capsOrArch !== "string") {
    return (arch && capsOrArch?.arch_display_names?.[arch]) || arch || "";
  }
  return ARCH_DISPLAY_NAMES[capsOrArch] || capsOrArch;
}

// True when `arch` honors `feature`. An unknown arch, or capabilities that have
// not loaded yet, are treated as SUPPORTING the feature — the same convention as
// the backend's arch_supports_feature(), so a control is never hidden merely
// because the matrix was unavailable.
// `value` (optional): a value listed in `supported_values` counts as supported
// even when the feature as a whole is unsupported on that arch — the same rule
// as the backend's arch_supports_feature(arch, feature, value).
export const archSupportsFeature = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  feature: string,
  value?: string
): boolean => {
  if (!caps || !arch) return true;
  if (!(caps.unsupported?.[arch] && feature in caps.unsupported[arch])) return true;
  if (value === undefined) return false;
  return (caps.supported_values?.[arch]?.[feature] ?? []).includes(value);
};

// True when `arch`'s transformer can be converted to the weight-only INT8
// layout AT RUNTIME, in place, from an ordinary bf16 checkpoint.
//
// Read from the capability payload's `runtime_int8_archs`, which the backend
// serves straight from RUNTIME_INT8_ARCHS — the tuple the converter itself
// gates on. There is deliberately no fallback list here: a hardcoded copy is
// exactly what went stale as architectures were added. While the matrix has not
// loaded (or on an older backend that does not send the field) the value is not
// offered, the conservative direction for an opt-in control whose backend would
// otherwise refuse the request and warn.
export const archSupportsRuntimeInt8 = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined
): boolean => !!arch && (caps?.runtime_int8_archs ?? []).includes(arch);

// Label for the weight-quantization selector on the CURRENTLY loaded model.
//
// `unet_quantization` is one request parameter across every architecture, but
// only SD1.5/SDXL have a U-Net: every other architecture in this app is a DiT
// (Z-Image, FLUX.2, Anima, Lens, MiniT2I, Krea 2, Ideogram 4, LTX-2.3) or an
// audio DiT (ACE-Step), and calling the control "U-Net Quantization" there names
// a module the model does not contain. Only the two U-Net architectures are
// listed, because that set cannot grow; anything else, including an arch this
// build has never heard of, gets the neutral both-ways label rather than a
// guess.
const UNET_ARCHS = new Set(["sd15", "sdxl"]);

export const transformerQuantizationLabel = (
  arch: string | null | undefined
): string => {
  if (!arch) return "Transformer / U-Net Quantization";
  return UNET_ARCHS.has(arch) ? "U-Net Quantization" : "Transformer Quantization";
};

// Options for the "Transformer / U-Net Quantization" selector, filtered by what
// the loaded architecture actually applies. When the capability matrix has not
// loaded, every FP8 value is offered (the same "assume supported" convention as
// archSupportsFeature), so a control is never narrowed merely because the matrix
// was unavailable.
export const unetQuantizationOptions = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined
): { value: string; label: string }[] => {
  const allow = (v: string) =>
    archSupportsFeature(caps, arch, "unet_quantization", v);
  const options = [{ value: "none", label: "None" }];
  if (allow("fp8_e4m3fn")) options.push({ value: "fp8_e4m3fn", label: "FP8 E4M3" });
  if (allow("fp8_e5m2")) options.push({ value: "fp8_e5m2", label: "FP8 E5M2" });
  if (archSupportsRuntimeInt8(caps, arch) && allow("int8")) {
    options.push({
      value: "int8",
      label: "INT8 (in-place, applied once per model load)",
    });
  }
  return options;
};

// A persisted (localStorage) unet_quantization can name a value the CURRENTLY
// loaded architecture does not offer — e.g. `fp8_e4m3fn` carried over onto a
// krea2 model, where only `int8` is applied. Left alone, the <select> holds a
// value that is not among its options (it renders blank) while the panel keeps
// SENDING the value. Returns the value to keep, or null when it is not offered.
export const normalizeUnetQuantization = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  value: string | null | undefined
): string | null => {
  if (!value || value === "none") return null;
  const offered = unetQuantizationOptions(caps, arch).some((o) => o.value === value);
  return offered ? value : null;
};

// ---------------------------------------------------------------------------
// Loop-generation decode-mode response helpers
// ---------------------------------------------------------------------------
// The 3 generation endpoints (txt2img/img2img/inpaint) accept `loop_decode`
// ("full"|"cheap"|"none") + `skip_gallery` (img2img additionally accepts
// `input_latent_id`). Depending on those flags the response shape varies:
//   - normal decode:      { success, image: { filename, seed, ... }, actual_seed, warnings }
//   - loop_decode="none": { success, latent_id, actual_seed, warnings }            (NO image)
//   - skip_gallery=true:  { success, filename, image_path, actual_seed, warnings } (saved file, no DB record)
// These helpers read whichever shape is present so loop-generation chaining
// code doesn't need to special-case every call site.
