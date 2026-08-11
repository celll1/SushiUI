"use client";

import { cn } from "@/lib/utils";
import NumberInput from "./NumberInput";

/**
 * The subset of a video panel's params the acceleration block reads/writes.
 * Every one of the four video-capable panels (Txt2Img/Img2Img/Inpaint/Outpaint)
 * declares these exact fields, so a panel's own params object satisfies this
 * shape structurally -- no cast needed at the call site.
 */
export interface VideoAccelerationValues {
  video_blocks_to_swap?: number;
  fuse_output_proj?: boolean;
  fbcache_enable?: boolean;
  fbcache_threshold?: number;
  fbcache_warmup_steps?: number;
  spectrum_enable?: boolean;
  spectrum_w?: number;
  spectrum_warmup_steps?: number;
}

interface VideoAccelerationControlsProps {
  /** Prefix for DOM ids, so four instances on one page never collide. */
  idPrefix: string;
  values: VideoAccelerationValues;
  onChange: (patch: Partial<VideoAccelerationValues>) => void;
  supportsSpectrum: boolean;
  supportsFbcache: boolean;
  /** archSupportsFeature(caps, arch, "fuse_output_proj") -- MiniMax-H3 only. */
  supportsFuseOutputProj: boolean;
  /** param_defaults.VIDEO_GEN_DEFAULTS["blocks_to_swap_enabled_default"] (per-arch resolved). */
  blocksToSwapEnabledDefault: number;
  blockSwapMax: number;
  /**
   * Optional caller-supplied reason to force FBCache disabled beyond the
   * Block Swap / Spectrum mutual exclusion below (e.g. a spatial mask
   * timeline with keyframes, which the backend refuses to combine with an
   * active FBCache -- see `fbcache_active()` in
   * `core/inference/fbcache.py`). Omit this prop to leave FBCache gated
   * purely by Block Swap/Spectrum, unchanged from before this prop existed.
   */
  fbcacheLockedReason?: string;
}

/**
 * Video-mode acceleration controls (Block Swap / First Block Cache /
 * Spectrum forecasting), shared by every video-capable generation panel.
 *
 * ONE component rather than four copies is deliberate: the four panels
 * historically diverged on exactly which of these three controls they
 * exposed (see the "video acceleration parity" fix), and a hand-maintained
 * fourth/fifth copy is exactly the mechanism that produced that drift. A
 * shared component makes the three controls -- and the mutual-exclusion
 * rules between them -- structurally impossible to diverge again.
 *
 * Mutual exclusion mirrors the backend exactly (see
 * `core/pipeline_backends/ltx2.py::_ltx2_build_fbcache`/`_ltx2_build_spectrum`
 * and `core/models/minimax_h3_block_loop_wrapper.py::attach_fbcache`, both
 * read-only references for this component -- not edited here):
 *   - FBCache and Spectrum are each mutually exclusive with Block Swap (a
 *     cache/forecast skip step bypasses the block loop, desyncing the
 *     per-block swap prefetch rotation). Turning Block Swap on clears both.
 *   - FBCache and Spectrum are mutually exclusive with EACH OTHER (same
 *     trajectory-redundancy target); Spectrum takes precedence on the
 *     backend, so turning Spectrum on here clears FBCache to match.
 * Both facts are surfaced as disabled controls with a factual note, rather
 * than left for the user to discover after a long generation silently ran
 * without the acceleration they thought they enabled.
 */
export default function VideoAccelerationControls({
  idPrefix,
  values,
  onChange,
  supportsSpectrum,
  supportsFbcache,
  supportsFuseOutputProj,
  blocksToSwapEnabledDefault,
  blockSwapMax,
  fbcacheLockedReason,
}: VideoAccelerationControlsProps) {
  const blockSwapOn = (values.video_blocks_to_swap ?? 0) > 0;
  const spectrumOn = !!values.spectrum_enable;
  const spectrumDisabled = blockSwapOn;
  const fbcacheLocked = !!fbcacheLockedReason;
  const fbcacheDisabled = blockSwapOn || spectrumOn || fbcacheLocked;

  return (
    <>
      <div className="text-sm font-semibold text-gray-400 mt-4 mb-1">Acceleration</div>

      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          id={`${idPrefix}_block_swap_enable`}
          checked={blockSwapOn}
          onChange={(e) => {
            if (e.target.checked) {
              // Mirrors the backend: Block Swap forces FBCache/Spectrum off
              // rather than leaving them checked-but-silently-ignored.
              onChange({
                video_blocks_to_swap: blocksToSwapEnabledDefault,
                fbcache_enable: false,
                spectrum_enable: false,
              });
            } else {
              onChange({ video_blocks_to_swap: 0 });
            }
          }}
          className="rounded"
        />
        <label htmlFor={`${idPrefix}_block_swap_enable`} className="text-sm text-gray-300">
          Block Swap (Transformer offloading)
        </label>
      </div>
      <p className="text-xs text-gray-500 ml-6">
        Trades device memory for host/device weight transfers: the given number of
        transformer blocks stay off the GPU and stream over during the denoise loop
        instead of staying resident. Off by default. Measured on MiniMax-H3, the added
        step time did not grow with the number of blocks swapped, so a small count pays
        close to the same fixed cost as a large one for less memory saved.
      </p>
      {blockSwapOn && (
        <div className="ml-6 mt-1">
          <label className="block text-xs text-gray-400 mb-1">Blocks to swap</label>
          <NumberInput
            label="Blocks to swap"
            value={values.video_blocks_to_swap ?? blocksToSwapEnabledDefault}
            onCommit={(v) => onChange({ video_blocks_to_swap: Math.max(1, v) })}
            min={1}
            max={blockSwapMax}
            step={1}
            parse="int"
            className="w-24"
          />
        </div>
      )}

      {supportsFuseOutputProj && (
        <div className="flex items-center gap-2 mt-2">
          <input
            type="checkbox"
            id={`${idPrefix}_fuse_output_proj`}
            checked={!!values.fuse_output_proj}
            onChange={(e) => onChange({ fuse_output_proj: e.target.checked })}
            className="rounded"
          />
          <label htmlFor={`${idPrefix}_fuse_output_proj`} className="text-sm text-gray-300">
            Fuse output projection
          </label>
        </div>
      )}
      {supportsFuseOutputProj && (
        <p className="text-xs text-gray-500 ml-6">
          Folds the output-tail projection heads into the chunked output-norm loop instead of
          materializing the full intermediate first. Measured on MiniMax-H3 at 768x1248, 345
          frames, 40 blocks swapped: peak reserved 16.881 to 14.934 GiB, with the per-forward
          time inside its run-to-run spread. Not bit-exact: max deviation measured against the
          default path at 4.6e-6 relative (video head), roughly 30-40x fp32 machine epsilon.
          Off by default for that reason, not for cost.
        </p>
      )}

      {supportsSpectrum && (
        <div className={cn("flex items-center gap-2 mt-2", spectrumDisabled && "opacity-50")}>
          <input
            type="checkbox"
            id={`${idPrefix}_spectrum_enable`}
            checked={spectrumOn}
            disabled={spectrumDisabled}
            onChange={(e) => {
              if (e.target.checked) {
                // Spectrum takes precedence over FBCache on the backend;
                // clear FBCache here so the UI never shows both checked.
                onChange({ spectrum_enable: true, fbcache_enable: false });
              } else {
                onChange({ spectrum_enable: false });
              }
            }}
            className="rounded"
          />
          <label htmlFor={`${idPrefix}_spectrum_enable`} className="text-sm text-gray-300">
            Spectrum (Spectral Feature Forecasting)
          </label>
          <span className="text-xs text-gray-500">
            (mutually exclusive with FBCache; disabled if Block Swap is on)
          </span>
        </div>
      )}
      {supportsSpectrum && spectrumOn && !spectrumDisabled && (
        <div className="ml-6 mt-1 grid grid-cols-2 gap-2">
          <label className="text-xs text-gray-400 flex items-center gap-1">
            Mix w
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={values.spectrum_w ?? 0.5}
              onChange={(e) => onChange({ spectrum_w: parseFloat(e.target.value) })}
              className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
            />
          </label>
          <label className="text-xs text-gray-400 flex items-center gap-1">
            Warmup
            <input
              type="number"
              min={1}
              step={1}
              value={values.spectrum_warmup_steps ?? 3}
              onChange={(e) => onChange({ spectrum_warmup_steps: parseInt(e.target.value) || 3 })}
              className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
            />
          </label>
        </div>
      )}

      {supportsFbcache && (
        <div className={cn("flex items-center gap-2 mt-2", fbcacheDisabled && "opacity-50")}>
          <input
            type="checkbox"
            id={`${idPrefix}_fbcache_enable`}
            checked={!!values.fbcache_enable}
            disabled={fbcacheDisabled}
            onChange={(e) => onChange({ fbcache_enable: e.target.checked })}
            className="rounded"
          />
          <label htmlFor={`${idPrefix}_fbcache_enable`} className="text-sm text-gray-300">
            First Block Cache (dynamic caching)
          </label>
          <span className="text-xs text-gray-500">
            {fbcacheLocked
              ? `(${fbcacheLockedReason})`
              : "(mutually exclusive with Spectrum; disabled if Block Swap is on)"}
          </span>
        </div>
      )}
      {supportsFbcache && values.fbcache_enable && !fbcacheDisabled && (
        <div className="ml-6 mt-1 grid grid-cols-2 gap-2">
          <label className="text-xs text-gray-400 flex items-center gap-1">
            Residual threshold
            <NumberInput
              min={0}
              step={0.01}
              parse="float"
              value={values.fbcache_threshold ?? 0.12}
              defaultValue={0.12}
              onCommit={(v) => onChange({ fbcache_threshold: v })}
              className="w-20"
            />
          </label>
          <label className="text-xs text-gray-400 flex items-center gap-1">
            Warmup steps
            <NumberInput
              min={0}
              step={1}
              value={values.fbcache_warmup_steps ?? 1}
              defaultValue={1}
              onCommit={(v) => onChange({ fbcache_warmup_steps: v })}
              className="w-20"
            />
          </label>
        </div>
      )}
    </>
  );
}
