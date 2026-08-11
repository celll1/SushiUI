"use client";

import { useMemo, useState, type ChangeEvent } from "react";
import Button from "../common/Button";
import Select from "../common/Select";
import {
  clampFrame,
  DEFAULT_MASK_INTERPOLATION,
  MAX_MASK_KEYFRAMES,
  MAX_MASK_SCALE,
  MIN_MASK_SCALE,
  pruneKeyframesToFrameRange,
  removeKeyframe,
  upsertKeyframe,
  type MaskInterpolation,
  type MaskTransform,
  type VideoMaskKeyframe,
} from "@/utils/videoMaskTimeline";

export interface VideoInpaintMaskTimelineProps {
  keyframes: VideoMaskKeyframe[];
  /** Current playhead in the same trimmed-clip frame space as the range. */
  currentFrame: number;
  /** The inpaint interval is [rangeStart, rangeEnd), matching the API. */
  rangeStart: number;
  rangeEnd: number;
  /** Total decoded frames; the track domain is always 0..totalFrames - 1. */
  totalFrames?: number;
  onChange: (keyframes: VideoMaskKeyframe[]) => void;
  onEditKeyframe: (keyframe: VideoMaskKeyframe) => void;
  /** The parent owns mask creation; this callback receives the requested frame. */
  onAddKeyframe: (frame: number) => void;
  disabled?: boolean;
}

const interpolationOptions: Array<{ value: MaskInterpolation; label: string }> = [
  { value: "hold", label: "Hold" },
  { value: "affine", label: "Affine" },
  { value: "sdf", label: "SDF morph" },
];

const transformFields: Array<{
  key: keyof MaskTransform;
  label: string;
  step: string;
  min?: string;
  max?: string;
}> = [
  { key: "x", label: "X", step: "1" },
  { key: "y", label: "Y", step: "1" },
  { key: "scaleX", label: "Scale X", step: "0.05", min: String(MIN_MASK_SCALE), max: String(MAX_MASK_SCALE) },
  { key: "scaleY", label: "Scale Y", step: "0.05", min: String(MIN_MASK_SCALE), max: String(MAX_MASK_SCALE) },
  { key: "rotation", label: "Rotation (degrees)", step: "1" },
];

function frameDescription(frame: number): string {
  return `Frame ${frame}`;
}

function safeFrameCount(totalFrames: number | undefined, fallbackFrameCounts: number[]): number {
  if (typeof totalFrames === "number" && Number.isFinite(totalFrames) && totalFrames > 0) {
    return Math.max(1, Math.floor(totalFrames));
  }
  return fallbackFrameCounts.reduce((maximum, value) => {
    if (typeof value !== "number" || !Number.isFinite(value)) return maximum;
    return Math.max(maximum, Math.ceil(value));
  }, 1);
}

function safeInteger(value: number | undefined, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? Math.round(value) : fallback;
}

function uniqueCopyId(source: VideoMaskKeyframe, keyframes: VideoMaskKeyframe[]): string {
  const ids = new Set(keyframes.map((keyframe) => keyframe.id));
  const base = `${source.id}-copy`;
  let candidate = base;
  let suffix = 2;
  while (ids.has(candidate)) candidate = `${base}-${suffix++}`;
  return candidate;
}

function findFreeFrame(
  source: VideoMaskKeyframe,
  keyframes: VideoMaskKeyframe[],
  minFrame: number,
  maxFrame: number,
): number | null {
  const occupied = new Set(keyframes.map((keyframe) => keyframe.frame));
  occupied.delete(source.frame);
  const sourceFrame = clampFrame(source.frame, minFrame, maxFrame);
  for (let distance = 1; distance <= maxFrame - minFrame; distance += 1) {
    const candidates = [sourceFrame + distance, sourceFrame - distance];
    for (const candidate of candidates) {
      if (candidate >= minFrame && candidate <= maxFrame && !occupied.has(candidate)) return candidate;
    }
  }
  return null;
}

export default function VideoInpaintMaskTimeline({
  keyframes,
  currentFrame,
  rangeStart,
  rangeEnd,
  totalFrames,
  onChange,
  onEditKeyframe,
  onAddKeyframe,
  disabled = false,
}: VideoInpaintMaskTimelineProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const safeTotalFrames = safeFrameCount(totalFrames, [
    rangeEnd,
    typeof currentFrame === "number" ? currentFrame + 1 : Number.NaN,
    ...keyframes.map((keyframe) => keyframe.frame + 1),
  ]);
  const lastFrame = safeTotalFrames - 1;
  const rawRangeStart = safeInteger(rangeStart, 0);
  const rawRangeEnd = safeInteger(rangeEnd, safeTotalFrames);
  const safeRangeStart = clampFrame(Math.min(rawRangeStart, rawRangeEnd), 0, lastFrame);
  const safeRangeEnd = Math.min(
    safeTotalFrames,
    Math.max(safeRangeStart + 1, clampFrame(Math.max(rawRangeStart, rawRangeEnd), 0, safeTotalFrames)),
  );
  const safeCurrentFrame = clampFrame(safeInteger(currentFrame, 0), 0, lastFrame);
  const lastFrameInRange = Math.max(safeRangeStart, safeRangeEnd - 1);
  const hasRange =
    Number.isFinite(rangeStart) &&
    Number.isFinite(rangeEnd) &&
    rangeEnd > rangeStart &&
    rangeEnd > 0 &&
    rangeStart < safeTotalFrames;
  const orderedKeyframes = useMemo(
    () =>
      pruneKeyframesToFrameRange(
        keyframes,
        hasRange ? safeRangeStart : 0,
        hasRange ? lastFrameInRange : lastFrame,
      ),
    [hasRange, keyframes, lastFrame, lastFrameInRange, safeRangeStart],
  );

  const addFrame = clampFrame(safeCurrentFrame, safeRangeStart, lastFrameInRange);
  const existingAtAddFrame = orderedKeyframes.find((keyframe) => keyframe.frame === addFrame);
  // Checked against the raw `keyframes` prop (the whole manifest), not
  // `orderedKeyframes` (only the ones currently inside the regenerate
  // range): the cap validateVideoMaskManifest enforces at generate time is
  // on the whole manifest, so a proactive warning here has to use the same
  // count or it would let the user draw well past the limit before ever
  // being told, which is the failure mode this is meant to avoid.
  const atKeyframeCap = !existingAtAddFrame && keyframes.length >= MAX_MASK_KEYFRAMES;

  const framePercent = (frame: number): number => {
    const safeFrame = clampFrame(safeInteger(frame, 0), 0, lastFrame);
    return lastFrame > 0 ? (safeFrame / lastFrame) * 100 : 0;
  };

  const addAtPlayhead = () => {
    if (disabled || !hasRange) return;
    if (existingAtAddFrame) {
      setNotice(null);
      setSelectedId(existingAtAddFrame.id);
      onEditKeyframe(existingAtAddFrame);
      return;
    }
    if (atKeyframeCap) {
      setNotice(`This clip already has the maximum of ${MAX_MASK_KEYFRAMES} mask keyframes. Delete one before adding another.`);
      return;
    }
    setNotice(null);
    onAddKeyframe(addFrame);
  };

  // These four handlers upsert/remove against the raw `keyframes` prop, NOT
  // `orderedKeyframes` (the range-pruned view used only for what this
  // component renders). Building the next value from `orderedKeyframes`
  // would silently drop every keyframe currently outside the regenerate
  // range as a side effect of editing an unrelated in-range one -- that
  // range is transient (it moves as the user drags the range timeline) and
  // out-of-range keyframes are kept on purpose (InpaintPanel surfaces a
  // read-only count of them instead), so this component must not be the
  // place that discards them.
  const changeInterpolation = (
    keyframe: VideoMaskKeyframe,
    event: ChangeEvent<HTMLSelectElement>,
    isFinalKeyframe: boolean,
  ) => {
    if (disabled || isFinalKeyframe) return;
    const interpolation = event.target.value as MaskInterpolation;
    onChange(
      upsertKeyframe(keyframes, {
        ...keyframe,
        interpolationToNext: interpolation || DEFAULT_MASK_INTERPOLATION,
      }),
    );
  };

  const changeTransform = (
    keyframe: VideoMaskKeyframe,
    field: keyof MaskTransform,
    value: string,
  ) => {
    if (disabled) return;
    const numericValue = Number(value);
    if (!Number.isFinite(numericValue)) return;
    onChange(
      upsertKeyframe(keyframes, {
        ...keyframe,
        transform: { ...keyframe.transform, [field]: numericValue },
      }),
    );
  };

  const duplicateKeyframe = (source: VideoMaskKeyframe) => {
    if (disabled) return;
    if (keyframes.length >= MAX_MASK_KEYFRAMES) {
      setNotice(`This clip already has the maximum of ${MAX_MASK_KEYFRAMES} mask keyframes. Delete one before duplicating another.`);
      return;
    }
    // Placement still only considers in-range, in-view keyframes: the
    // duplicate is meant to land somewhere the user can see and edit it.
    const frame = findFreeFrame(source, orderedKeyframes, safeRangeStart, lastFrameInRange);
    if (frame === null) {
      setNotice("There is no free frame in the inpaint range for a duplicate.");
      return;
    }
    const duplicate: VideoMaskKeyframe = {
      ...source,
      id: uniqueCopyId(source, keyframes),
      frame,
      transform: { ...source.transform },
    };
    setNotice(null);
    setSelectedId(duplicate.id);
    onChange(upsertKeyframe(keyframes, duplicate));
  };

  const deleteKeyframe = (keyframe: VideoMaskKeyframe) => {
    if (disabled) return;
    setSelectedId((selected) => (selected === keyframe.id ? null : selected));
    onChange(removeKeyframe(keyframes, keyframe.id));
  };

  return (
    <section className="space-y-3" aria-label="Video inpaint mask timeline">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-medium text-gray-300">Mask keyframes</h3>
          <p className="text-xs text-gray-500">
            Draw a mask at keyframes and choose how it applies until the next one.
          </p>
        </div>
        <Button
          type="button"
          variant="secondary"
          size="sm"
          disabled={disabled || !hasRange || atKeyframeCap}
          onClick={addAtPlayhead}
          aria-label={`Add or edit mask keyframe at frame ${addFrame}`}
          title={atKeyframeCap ? `This clip already has the maximum of ${MAX_MASK_KEYFRAMES} mask keyframes.` : undefined}
        >
          {existingAtAddFrame ? "Edit at playhead" : "Add at playhead"} ({addFrame})
        </Button>
      </div>

      <div className="relative h-20 rounded border border-gray-700 bg-gray-800 select-none">
        <div className="absolute inset-x-0 top-0 h-12 overflow-hidden rounded-t">
          <div
            className="absolute inset-y-0 border-x border-amber-500/70 bg-amber-500/20"
            style={{
              left: `${framePercent(safeRangeStart)}%`,
              width: `${Math.max(0.5, framePercent(safeRangeEnd) - framePercent(safeRangeStart))}%`,
            }}
            title="Inpaint range"
          >
            <span className="absolute left-1 top-1 text-[10px] text-amber-200 pointer-events-none">
              Inpaint range
            </span>
          </div>
          <div className="absolute inset-x-0 top-1/2 h-px bg-gray-600" />
          <div
            className="absolute bottom-0 top-0 w-0.5 bg-emerald-400"
            style={{ left: `${framePercent(safeCurrentFrame)}%` }}
            title={`Playhead: ${frameDescription(safeCurrentFrame)}`}
          />
        </div>

        <div className="absolute inset-x-0 top-0 h-12" role="list" aria-label="Mask keyframe markers">
          {orderedKeyframes.map((keyframe) => {
            const selected = selectedId === keyframe.id;
            return (
              <span key={keyframe.id} role="listitem">
                <button
                  type="button"
                  disabled={disabled}
                  className={`absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rotate-45 border focus:outline-none focus:ring-2 focus:ring-violet-300 ${
                    selected ? "border-white bg-violet-400" : "border-violet-300 bg-violet-600"
                  }`}
                  style={{ left: `${framePercent(keyframe.frame)}%` }}
                  onClick={() => {
                    if (disabled) return;
                    setSelectedId(keyframe.id);
                  }}
                  aria-label={`${frameDescription(keyframe.frame)} mask ${keyframe.maskId}`}
                  aria-pressed={selected}
                  title={`${frameDescription(keyframe.frame)} - ${keyframe.maskId}`}
                />
              </span>
            );
          })}
        </div>

        <div className="absolute inset-x-1 bottom-1 flex justify-between text-[10px] text-gray-500 pointer-events-none">
          <span>Frame 0</span>
          <span>Frame {lastFrame}</span>
        </div>
      </div>

      <div className="flex justify-between text-xs text-gray-500">
        <span>Inpaint range: [{safeRangeStart}, {safeRangeEnd})</span>
        <span>Playhead {safeCurrentFrame}</span>
      </div>

      {notice && <p className="text-xs text-amber-300" role="status">{notice}</p>}

      {orderedKeyframes.length === 0 ? (
        <p className="rounded border border-dashed border-gray-700 px-3 py-4 text-xs text-gray-500">
          No mask keyframes yet. Add one at the playhead to begin.
        </p>
      ) : (
        <div className="space-y-2" role="list" aria-label="Mask keyframes">
          {orderedKeyframes.map((keyframe, index) => (
            <div
              key={keyframe.id}
              role="listitem"
              className={`rounded border p-2 ${
                selectedId === keyframe.id ? "border-violet-400/70 bg-gray-800" : "border-gray-700 bg-gray-900/40"
              }`}
            >
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  disabled={disabled}
                  className="text-left text-xs text-gray-200 hover:text-white focus:outline-none focus:ring-1 focus:ring-violet-400 rounded"
                  onClick={() => {
                    if (disabled) return;
                    setSelectedId(keyframe.id);
                    onEditKeyframe(keyframe);
                  }}
                  aria-label={`Edit mask keyframe at frame ${keyframe.frame}`}
                >
                  <span className="font-medium">Frame {keyframe.frame}</span>
                  <span className="ml-2 text-gray-500">{keyframe.maskId}</span>
                </button>
                <span className="text-[10px] text-gray-600">
                  {index === orderedKeyframes.length - 1 ? "last (no interpolation)" : "to next"}
                </span>
                <Select
                  title={index === orderedKeyframes.length - 1 ? "Final keyframe has no next segment" : undefined}
                  className="min-w-[7rem]"
                  options={interpolationOptions}
                  value={keyframe.interpolationToNext || DEFAULT_MASK_INTERPOLATION}
                  onChange={(event) =>
                    changeInterpolation(keyframe, event, index === orderedKeyframes.length - 1)
                  }
                  disabled={disabled || index === orderedKeyframes.length - 1}
                  aria-label={`Interpolation after frame ${keyframe.frame}`}
                />
                <details className="w-full rounded border border-gray-800 bg-gray-950/40 px-2 py-1">
                  <summary className="cursor-pointer text-[10px] text-gray-500">
                    Transform (canvas center pivot)
                  </summary>
                  <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-5">
                    {transformFields.map((field) => (
                      <label key={field.key} className="text-[10px] text-gray-500">
                        {field.label}
                        <input
                          type="number"
                          step={field.step}
                          min={field.min}
                          max={field.max}
                          value={keyframe.transform[field.key]}
                          disabled={disabled}
                          onChange={(event) => changeTransform(keyframe, field.key, event.target.value)}
                          className="mt-1 h-7 w-full rounded border border-gray-700 bg-gray-800 px-1.5 text-xs text-white"
                          aria-label={`${field.label} for frame ${keyframe.frame}`}
                        />
                      </label>
                    ))}
                  </div>
                </details>
                <div className="ml-auto flex items-center gap-1">
                  <Button
                    type="button"
                    variant="secondary"
                    size="xs"
                    disabled={disabled}
                    onClick={() => {
                      if (disabled) return;
                      setSelectedId(keyframe.id);
                      onEditKeyframe(keyframe);
                    }}
                    aria-label={`Edit mask at frame ${keyframe.frame}`}
                  >
                    Edit mask
                  </Button>
                  <Button
                    type="button"
                    variant="secondary"
                    size="xs"
                    disabled={disabled || keyframes.length >= MAX_MASK_KEYFRAMES}
                    onClick={() => duplicateKeyframe(keyframe)}
                    aria-label={`Duplicate mask keyframe at frame ${keyframe.frame}`}
                    title={keyframes.length >= MAX_MASK_KEYFRAMES ? `This clip already has the maximum of ${MAX_MASK_KEYFRAMES} mask keyframes.` : undefined}
                  >
                    Duplicate
                  </Button>
                  <Button
                    type="button"
                    variant="danger"
                    size="xs"
                    disabled={disabled}
                    onClick={() => deleteKeyframe(keyframe)}
                    aria-label={`Delete mask keyframe at frame ${keyframe.frame}`}
                  >
                    Delete
                  </Button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
