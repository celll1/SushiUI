"use client";

import { VideoChainPlan } from "@/utils/api";

export interface VideoChainConfirmDialogProps {
  isOpen: boolean;
  /** The length actually held in the frame control, in frames. */
  requestedFrames: number;
  /**
   * The length of clip ONE request in the plan produces, in frames -- the
   * user's `chain_segment_frames` when they set one, otherwise the loaded
   * architecture's own single-inference cap (see api.ts's `chainSegmentCap`).
   * Not necessarily a hard technical wall: on an architecture with no
   * `max_frames` at all, this is purely the user's own chosen segment size.
   */
  capFrames: number;
  /** Pre-formatted (e.g. `.toFixed(2)`) seconds readout, or null if unknown. */
  capSeconds: string | null;
  /** Pre-formatted seconds readout for `plan.finalFrames`, or null if unknown. */
  finalSeconds: string | null;
  plan: VideoChainPlan | null;
  /**
   * Extra, caller-supplied disclosure lines specific to what THIS request
   * would drop or approximate by chaining -- e.g. which reference tracks stop
   * conditioning after segment 1, or what a keyframe anchor's `-1` ("pin to
   * end") placement resolves against. Rendered verbatim, one per paragraph;
   * omitted entirely when there is nothing to disclose.
   */
  notes?: string[];
  onCancel: () => void;
  /** Default action: generate once, at the cap (snapped). */
  onGenerateAtCap: () => void;
  /** Explicit, non-default action: enqueue the chain. */
  onStartChain: () => void;
}

/**
 * The choice CLAUDE.md's opt-in-chaining requirement mandates: a value held
 * in a video length control above the loaded architecture's single-inference
 * cap must never silently become either "clamp to the cap" or "chain
 * automatically" -- Generate has to force a deliberate pick between the two,
 * with the single-inference request as the DEFAULT.
 *
 * Built as its own small dialog (same visual language as
 * `common/ConfirmDialog`) rather than reusing that component directly: this
 * needs THREE actions (cancel / generate-at-cap / start-chain), and
 * ConfirmDialog only ever renders two.
 */
export default function VideoChainConfirmDialog({
  isOpen,
  requestedFrames,
  capFrames,
  capSeconds,
  finalSeconds,
  plan,
  notes,
  onCancel,
  onGenerateAtCap,
  onStartChain,
}: VideoChainConfirmDialogProps) {
  if (!isOpen) return null;

  const overshoot = plan != null ? plan.finalFrames - requestedFrames : 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-gray-800 rounded-lg shadow-xl max-w-md w-full mx-4 border border-gray-700">
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h3 className="text-lg font-semibold text-white">Length exceeds the current segment length</h3>
        </div>

        <div className="p-4 space-y-2">
          <p className="text-sm text-gray-300">
            {requestedFrames} frames exceeds the current segment length of {capFrames} frames
            {capSeconds != null ? ` (${capSeconds}s)` : ""}.
          </p>
          {plan != null && (
            <p className="text-sm text-gray-300">
              Reaching {requestedFrames} frames takes {plan.segments} generation requests, chained via temporal
              outpaint. The chain actually reaches {plan.finalFrames} frames
              {finalSeconds != null ? ` (${finalSeconds}s)` : ""}
              {overshoot > 0 ? `, ${overshoot} more than requested (the arithmetic that lands each segment on the model's frame grid does not land exactly on the requested total)` : ""}.
              Segments after the first are conditioned only on the boundary frame of the previous segment, not the
              rest of its content or the original prompt context.
            </p>
          )}
          {notes != null && notes.map((note, index) => (
            <p key={index} className="text-sm text-gray-400">{note}</p>
          ))}
        </div>

        <div className="flex flex-col gap-2 p-4 border-t border-gray-700">
          <button
            onClick={onGenerateAtCap}
            className="px-4 py-2 rounded text-sm font-medium transition-colors bg-blue-600 hover:bg-blue-500 text-white"
          >
            Generate at {capFrames} frames (single request)
          </button>
          <button
            onClick={onStartChain}
            className="px-4 py-2 rounded text-sm font-medium transition-colors bg-amber-700 hover:bg-amber-600 text-white"
          >
            {plan != null
              ? `Start chain: ${plan.segments} segments, reaches ${plan.finalFrames} frames`
              : "Start chain"}
          </button>
          <button
            onClick={onCancel}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm font-medium transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
