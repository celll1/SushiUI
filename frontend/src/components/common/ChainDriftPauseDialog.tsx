"use client";

// Design §4.1 (scratchpad/video_chain_context_design.md): when a chain
// continuation's actual accumulated frame count drifts from the manifest's
// planned value by more than `chain_drift_tolerance_frames`, the chain must
// PAUSE and let the user choose to continue or stop -- never continue
// silently. This dialog is that choice. It reports only measured facts
// (planned frames, actual frames, drift, tolerance, which segment) with no
// subjective framing of whether the drift is a problem.
import { ChainDriftPause } from "@/utils/videoChain";

export interface ChainDriftPauseDialogProps {
  pause: ChainDriftPause | null;
  onContinue: () => void;
  onStop: () => void;
}

export default function ChainDriftPauseDialog({
  pause,
  onContinue,
  onStop,
}: ChainDriftPauseDialogProps) {
  if (!pause) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-gray-800 rounded-lg shadow-xl max-w-lg w-full mx-4 border border-gray-700">
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h3 className="text-lg font-semibold text-white">Video chain paused: frame drift</h3>
        </div>

        <div className="p-4 space-y-3">
          <p className="text-sm text-gray-300">
            After segment {pause.segmentsCompleted}, the clip's actual accumulated frame count is{" "}
            {pause.driftFrames} frames away from the chain plan, above the plan's tolerance of{" "}
            {pause.toleranceFrames} frames.
          </p>
          <div className="rounded border border-gray-700 p-3 space-y-1 text-xs text-gray-300 font-mono">
            <div>planned accumulated frames: {pause.plannedAccumulatedFrames}</div>
            <div>actual accumulated frames: {pause.actualAccumulatedFrames}</div>
            <div>drift: {pause.driftFrames} frames (tolerance {pause.toleranceFrames})</div>
          </div>
          <p className="text-xs text-gray-400">
            Continuing sends the next segment at the plan's frame count regardless of this drift; the
            segment prompt's own local timing is unaffected. Stopping keeps the {pause.segmentsCompleted}{" "}
            segment(s) already completed, saved to the gallery.
          </p>
        </div>

        <div className="flex flex-col gap-2 p-4 border-t border-gray-700">
          <button
            onClick={onContinue}
            className="px-4 py-2 rounded text-sm font-medium transition-colors bg-amber-700 hover:bg-amber-600 text-white"
          >
            Continue the chain anyway
          </button>
          <button
            onClick={onStop}
            className="px-4 py-2 rounded text-sm font-medium transition-colors bg-gray-700 hover:bg-gray-600 text-white"
          >
            Stop the chain here
          </button>
        </div>
      </div>
    </div>
  );
}
