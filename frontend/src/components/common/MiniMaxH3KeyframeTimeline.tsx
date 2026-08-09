"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Button from "./Button";
import InlineHelp from "./InlineHelp";
import { MiniMaxH3Keyframe, toBase64 } from "@/utils/api";

/**
 * Keyframe placement for MiniMax-H3's `fl2va` workflow (POST /generate/img2vid).
 *
 * Each anchor is an image pinned to ONE EXACT PIXEL FRAME of the generated
 * clip: the packed sequence's time axis is pixel-frame time, so a frame index
 * has an exact rotary coordinate and there is nothing to snap an anchor to.
 * The addressable unit is therefore the integer frame.
 *
 * DRAGGING IS DISCRETE. A marker can be dragged along the track and the track
 * looks slider-like, but the pointer position is rounded to a whole frame on
 * every move, the marker snaps to that frame's position rather than following
 * the cursor, and the frame number it lands on is drawn on the marker. Arrow
 * keys step it by exactly one frame. So the interaction is a picker over
 * `lastIndex + 1` positions, not a continuous scrubber: the UI must not imply
 * sub-frame placement the model does not have. The numeric entry below the
 * track stays authoritative and editable for the same reason.
 *
 * WHAT THE THREE STORAGE SLOTS ARE (they are not redundant):
 *
 * * the uploaded input image is an anchor like any other, and its placement is
 *   `input_image_frame_index`;
 * * `last_frame_image` is the endpoint's live alias for an anchor at the last
 *   frame, kept because it is a shipped field with its own persistence and
 *   send-to wiring. The timeline renders it as the "last" chip;
 * * everything else is a `keyframes` entry with its own index.
 *
 * `-1` means the LAST frame and is rendered as "last". It is a sentinel rather
 * than a convenience: the server snaps the clip length to the model's own
 * `17n + 5` grid after the request is sent, so a client cannot know the last
 * index at the time it builds the request.
 *
 * THE AUDIO LANE beneath the track is a different kind of conditioning and is
 * drawn differently on purpose: an uploaded track's rows are the clip's OWN
 * audio rows, pinned clean for every frame, so the bar spans the whole track
 * and carries no offset handles. Whole-clip is not a default here — it is the
 * only supported placement, because the condition count is a prefix and the
 * audio rows are channel-major, so "half" would pin one stereo channel's entire
 * timeline. Drawing handles would promise a feature that does not exist.
 */

interface MiniMaxH3KeyframeTimelineProps {
  /** Clip length in frames, from the panel's (grid-constrained) length control. */
  numFrames: number;
  /** Frames per second, used only to show each anchor's time. */
  frameRate: number;
  /** The uploaded input image, as a data URL, for the first chip's thumbnail. */
  inputImage: string | null;
  inputImageFrameIndex: number;
  onInputImageFrameIndexChange: (frameIndex: number) => void;
  keyframes: MiniMaxH3Keyframe[];
  onKeyframesChange: (keyframes: MiniMaxH3Keyframe[]) => void;
  /** The alias slot: a data URL, or null when there is no end anchor. */
  lastFrameImage: string | null;
  onLastFrameImageChange: (dataUrl: string | null) => void;
  /**
   * The ia2v lane. Omit both to hide it: the lane is rendered only where the
   * loaded architecture declares `audio_conditioning`, so the panel decides,
   * not this component.
   */
  inputAudio?: File | null;
  onInputAudioChange?: (file: File | null) => void;
  /**
   * The panel's `audio_enable`. With it off nothing is muxed at all -- the
   * track still conditions the video and the backend says so in a warning --
   * so the lane must not describe an output file that will have no audio.
   */
  audioEnabled?: boolean;
  disabled?: boolean;
}

type ChipSource =
  | { kind: "input" }
  | { kind: "keyframe"; index: number }
  | { kind: "last" };

interface Chip {
  key: string;
  label: string;
  source: ChipSource;
  image: string | null;
  /** As stored: -1 means "the last frame". */
  requested: number;
  /** Where it actually lands on this clip. */
  frame: number;
  removable: boolean;
  /** The frame field is editable (the alias chip is pinned by definition). */
  editable: boolean;
}

/** Half the marker thumbnail's width (w-14 = 56px), in px. */
const MARKER_HALF_PX = 28;

function resolveFrame(requested: number, numFrames: number): number {
  return requested === -1 ? Math.max(0, numFrames - 1) : requested;
}

export default function MiniMaxH3KeyframeTimeline({
  numFrames,
  frameRate,
  inputImage,
  inputImageFrameIndex,
  onInputImageFrameIndexChange,
  keyframes,
  onKeyframesChange,
  lastFrameImage,
  onLastFrameImageChange,
  inputAudio = null,
  onInputAudioChange,
  audioEnabled = true,
  disabled = false,
}: MiniMaxH3KeyframeTimelineProps) {
  const fileInput = useRef<HTMLInputElement>(null);
  const audioInput = useRef<HTMLInputElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  // The marker currently being dragged, by chip key. Only one at a time; the
  // pointer is captured by the marker itself so the drag survives leaving the
  // track's box.
  const [dragKey, setDragKey] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  // Duration of the picked file, read from an <audio> element rather than
  // decoded: it is only used to tell the user whether the track is long enough
  // BEFORE the request, and the server does the authoritative check.
  const [audioSeconds, setAudioSeconds] = useState<number | null>(null);
  const lastIndex = Math.max(0, numFrames - 1);
  const fps = frameRate > 0 ? frameRate : 24;
  const clipSeconds = numFrames / fps;
  const audioTooShort = audioSeconds !== null && audioSeconds + 0.02 < clipSeconds;

  // Frame 0 and the last frame are drawn HALF A MARKER inside the track's box,
  // so an anchor at either end is fully visible instead of hanging outside the
  // card. The drag mapping below uses the same inset, so the frame the pointer
  // resolves to is the frame the marker is drawn on.
  const markerLeft = (percent: number) =>
    `calc(${MARKER_HALF_PX}px + (100% - ${MARKER_HALF_PX * 2}px) * ${percent / 100})`;

  /** Pointer x -> the WHOLE frame it is nearest. Never returns a fraction. */
  const frameFromClientX = (clientX: number): number => {
    const el = trackRef.current;
    if (!el || lastIndex <= 0) return 0;
    const rect = el.getBoundingClientRect();
    const usable = rect.width - MARKER_HALF_PX * 2;
    if (usable <= 0) return 0;
    const ratio = (clientX - rect.left - MARKER_HALF_PX) / usable;
    return Math.max(0, Math.min(lastIndex, Math.round(ratio * lastIndex)));
  };

  useEffect(() => {
    if (!inputAudio) {
      setAudioSeconds(null);
      return;
    }
    const url = URL.createObjectURL(inputAudio);
    const probe = new Audio();
    probe.preload = "metadata";
    probe.onloadedmetadata = () => {
      setAudioSeconds(Number.isFinite(probe.duration) ? probe.duration : null);
      URL.revokeObjectURL(url);
    };
    probe.onerror = () => {
      setAudioSeconds(null);
      URL.revokeObjectURL(url);
    };
    probe.src = url;
    return () => URL.revokeObjectURL(url);
  }, [inputAudio]);

  const chips: Chip[] = useMemo(() => {
    const built: Chip[] = [
      {
        key: "input",
        label: "Input image",
        source: { kind: "input" },
        image: inputImage,
        requested: inputImageFrameIndex,
        frame: resolveFrame(inputImageFrameIndex, numFrames),
        removable: false,
        editable: true,
      },
      ...keyframes.map((keyframe, index) => ({
        key: `keyframe-${index}`,
        label: `Keyframe ${index + 1}`,
        source: { kind: "keyframe" as const, index },
        image: typeof keyframe.image === "string" ? keyframe.image : null,
        requested: keyframe.frame_index,
        frame: resolveFrame(keyframe.frame_index, numFrames),
        removable: true,
        editable: true,
      })),
    ];
    if (lastFrameImage) {
      built.push({
        key: "last",
        label: "Last frame",
        source: { kind: "last" },
        image: lastFrameImage,
        requested: -1,
        frame: lastIndex,
        removable: true,
        editable: false,
      });
    }
    return built.sort((a, b) => a.frame - b.frame);
  }, [inputImage, inputImageFrameIndex, keyframes, lastFrameImage, numFrames, lastIndex]);

  // Clip length changed under an explicit index: clamp it and SAY SO. The
  // alternative (leaving it) is a 400 the user never asked for, and silently
  // dropping the anchor would be worse.
  useEffect(() => {
    if (numFrames <= 0) return;
    let clamped = 0;
    if (inputImageFrameIndex > lastIndex) {
      clamped += 1;
      onInputImageFrameIndexChange(lastIndex);
    }
    if (keyframes.some((keyframe) => keyframe.frame_index > lastIndex)) {
      clamped += keyframes.filter((keyframe) => keyframe.frame_index > lastIndex).length;
      onKeyframesChange(
        keyframes.map((keyframe) =>
          keyframe.frame_index > lastIndex ? { ...keyframe, frame_index: lastIndex } : keyframe,
        ),
      );
    }
    if (clamped > 0) {
      setNotice(
        `The clip is now ${numFrames} frames long, so ${clamped} anchor(s) past its end were ` +
          `moved to frame ${lastIndex}.`,
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [numFrames]);

  const occupied = useMemo(() => {
    const counts = new Map<number, number>();
    chips.forEach((chip) => counts.set(chip.frame, (counts.get(chip.frame) || 0) + 1));
    return counts;
  }, [chips]);
  const collisions = Array.from(occupied.entries())
    .filter(([, count]) => count > 1)
    .map(([frame]) => frame);

  const setFrame = (source: ChipSource, requested: number) => {
    setNotice(null);
    if (source.kind === "input") {
      onInputImageFrameIndexChange(requested);
      return;
    }
    if (source.kind === "keyframe") {
      onKeyframesChange(
        keyframes.map((keyframe, index) =>
          index === source.index ? { ...keyframe, frame_index: requested } : keyframe,
        ),
      );
    }
  };

  const remove = (source: ChipSource) => {
    setNotice(null);
    if (source.kind === "last") {
      onLastFrameImageChange(null);
      return;
    }
    if (source.kind === "keyframe") {
      onKeyframesChange(keyframes.filter((_keyframe, index) => index !== source.index));
    }
  };

  const addKeyframe = async (file: File | undefined) => {
    if (!file) return;
    const dataUrl = await toBase64(file);
    // Land the new anchor on a free frame near the middle, so a second "Add"
    // does not immediately collide with the first.
    const taken = new Set(chips.map((chip) => chip.frame));
    let frame = Math.floor(lastIndex / 2);
    while (frame < lastIndex && taken.has(frame)) frame += 1;
    while (frame > 0 && taken.has(frame)) frame -= 1;
    setNotice(null);
    onKeyframesChange([...keyframes, { image: dataUrl, frame_index: frame }]);
  };

  const pinToEnd = (chip: Chip) => {
    if (chip.requested === -1) {
      setFrame(chip.source, Math.max(0, Math.min(chip.frame, lastIndex)));
      return;
    }
    setFrame(chip.source, -1);
  };

  return (
    <div className="mt-3 space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1">
          <label className="block text-sm font-medium text-gray-300">Keyframes</label>
          <InlineHelp label="Keyframe placement details">
            <p>
              Anchors snap to exact frames ({fps} fps). Drag a marker, use its arrow keys, or enter a frame number. Pin to end follows the final snapped clip length.
            </p>
            <p>
              MiniMax-H3 documents first/last-frame conditioning with up to two images. Intermediate anchors and audio conditioning use the same mechanism but are not covered by its model card.
            </p>
          </InlineHelp>
        </div>
        <Button
          variant="secondary"
          size="sm"
          disabled={disabled}
          onClick={() => fileInput.current?.click()}
        >
          Add keyframe
        </Button>
      </div>
      <input
        ref={fileInput}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={async (e) => {
          await addKeyframe(e.target.files?.[0]);
          e.target.value = "";
        }}
      />

      {/* The track. Positions are exact frames, so the markers sit at
          frame/lastIndex and carry the frame number itself. Each marker is a
          drag handle whose position is rounded to a whole frame on every
          pointer move (see the discrete-dragging note at the top of the file);
          the thumbnail rides with it so an anchor is identifiable where it
          sits. */}
      <div
        ref={trackRef}
        className="relative h-20 rounded bg-gray-800 border border-gray-700 select-none"
      >
        {/* Baseline the markers stand on. */}
        <div className="absolute left-0 right-0 bottom-6 h-[2px] bg-gray-600" />
        <span className="absolute right-1 bottom-1 text-[10px] text-gray-400">
          {numFrames} frames · {(numFrames / fps).toFixed(2)}s
        </span>
        <span className="absolute left-1 bottom-1 text-[10px] text-gray-500">
          frame 0
        </span>
        {chips.map((chip) => {
          const percent = lastIndex > 0 ? (chip.frame / lastIndex) * 100 : 0;
          const draggable = !disabled && chip.editable && chip.requested !== -1;
          return (
            <div
              key={chip.key}
              className="absolute bottom-6 -translate-x-1/2 flex flex-col items-center"
              style={{ left: markerLeft(percent) }}
            >
              <div
                role={draggable ? "slider" : undefined}
                tabIndex={draggable ? 0 : -1}
                aria-label={draggable ? `${chip.label} frame` : undefined}
                aria-valuemin={draggable ? 0 : undefined}
                aria-valuemax={draggable ? lastIndex : undefined}
                aria-valuenow={draggable ? chip.frame : undefined}
                title={
                  draggable
                    ? `${chip.label} @ frame ${chip.frame} — drag along the track, or use the arrow keys, to move it one whole frame at a time`
                    : `${chip.label} @ frame ${chip.frame} (pinned to the end of the clip)`
                }
                className={`relative rounded border ${
                  dragKey === chip.key ? "border-blue-400" : "border-gray-600"
                } ${draggable ? "cursor-grab active:cursor-grabbing" : "cursor-default"} ${
                  draggable ? "focus:outline-none focus:ring-2 focus:ring-blue-500" : ""
                } bg-gray-900 overflow-hidden`}
                onPointerDown={(e) => {
                  if (!draggable) return;
                  e.preventDefault();
                  e.currentTarget.setPointerCapture(e.pointerId);
                  setDragKey(chip.key);
                  setNotice(null);
                }}
                onPointerMove={(e) => {
                  if (dragKey !== chip.key) return;
                  const frame = frameFromClientX(e.clientX);
                  if (frame !== chip.requested) setFrame(chip.source, frame);
                }}
                onPointerUp={(e) => {
                  if (dragKey !== chip.key) return;
                  e.currentTarget.releasePointerCapture(e.pointerId);
                  setDragKey(null);
                }}
                onPointerCancel={() => setDragKey(null)}
                onKeyDown={(e) => {
                  if (!draggable) return;
                  const delta =
                    e.key === "ArrowLeft" ? -1 : e.key === "ArrowRight" ? 1 : 0;
                  if (delta === 0) return;
                  e.preventDefault();
                  setFrame(
                    chip.source,
                    Math.max(0, Math.min(lastIndex, chip.frame + delta)),
                  );
                }}
              >
                {chip.image ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={chip.image}
                    alt={chip.label}
                    draggable={false}
                    className="h-9 w-14 object-cover pointer-events-none"
                  />
                ) : (
                  <span className="block h-9 w-14 bg-gray-800" />
                )}
              </div>
              {/* The exact frame the anchor is on, drawn under the marker so a
                  drag never has to be interpreted from a pixel position. */}
              <span className="mt-0.5 text-[10px] leading-3 text-gray-300 bg-gray-900/80 px-1 rounded">
                {chip.requested === -1 ? `end (${chip.frame})` : chip.frame}
              </span>
              <span className="h-2 w-[2px] bg-blue-400" />
            </div>
          );
        })}
      </div>

      <div className="space-y-1">
        {chips.map((chip) => (
          <div key={chip.key} className="flex items-center gap-2 text-xs text-gray-300">
            {chip.image ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={chip.image}
                alt={chip.label}
                className="h-8 w-12 object-cover rounded border border-gray-700"
              />
            ) : (
              <span className="h-8 w-12 rounded border border-gray-700 bg-gray-800" />
            )}
            <span className="w-24 shrink-0 text-gray-400">{chip.label}</span>
            <input
              type="number"
              className="w-20 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-xs"
              value={chip.requested === -1 ? lastIndex : chip.requested}
              min={0}
              max={lastIndex}
              step={1}
              disabled={disabled || !chip.editable || chip.requested === -1}
              onChange={(e) => {
                const parsed = parseInt(e.target.value, 10);
                if (Number.isNaN(parsed)) return;
                setFrame(chip.source, Math.max(0, Math.min(lastIndex, parsed)));
              }}
            />
            <span className="w-16 shrink-0 text-gray-500">
              {(chip.frame / fps).toFixed(2)}s
            </span>
            {chip.editable && (
              <label className="flex items-center gap-1 cursor-pointer">
                <input
                  type="checkbox"
                  checked={chip.requested === -1}
                  disabled={disabled}
                  onChange={() => pinToEnd(chip)}
                />
                <span>pin to end</span>
              </label>
            )}
            {!chip.editable && <span className="text-gray-500">pinned to end</span>}
            {chip.removable && (
              <button
                type="button"
                className="text-red-400 hover:text-red-300 px-1"
                disabled={disabled}
                onClick={() => remove(chip.source)}
                title="Remove this anchor"
              >
                ✕
              </button>
            )}
          </div>
        ))}
      </div>

      {/* The ia2v lane. FULL WIDTH BECAUSE WHOLE-CLIP IS THE ONLY SUPPORTED
          PLACEMENT: the track's rows are the clip's own audio rows, pinned for
          every frame, so there is no offset to drag and no handles are drawn
          for a control that does not exist. */}
      {onInputAudioChange && (
        <div className="mt-3 space-y-1 border-t border-gray-800 pt-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1">
              <label className="block text-sm font-medium text-gray-300">
                Input audio (optional)
              </label>
              <InlineHelp label="Input audio conditioning details">
                <p>The track conditions the whole clip; partial placement is not supported.</p>
                <p>A longer track is trimmed. A shorter track is refused rather than padded.</p>
                <p>
                  With audio output enabled, the file supplies the muxed soundtrack. With output disabled, it still conditions the video but is not muxed.
                </p>
                <p>Validation covered impulsive material; speech, pitch and timbre were not measured.</p>
              </InlineHelp>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="secondary"
                size="sm"
                disabled={disabled}
                onClick={() => audioInput.current?.click()}
              >
                {inputAudio ? "Replace" : "Choose audio"}
              </Button>
              {inputAudio && (
                <button
                  type="button"
                  className="text-red-400 hover:text-red-300 px-1"
                  disabled={disabled}
                  onClick={() => onInputAudioChange(null)}
                  title="Remove the input audio track"
                >
                  ✕
                </button>
              )}
            </div>
          </div>
          <input
            ref={audioInput}
            type="file"
            accept="audio/*"
            className="hidden"
            onChange={(e) => {
              onInputAudioChange(e.target.files?.[0] ?? null);
              e.target.value = "";
            }}
          />
          <div
            className={`relative h-6 rounded border ${
              inputAudio
                ? "bg-emerald-900/40 border-emerald-700"
                : "bg-gray-800 border-gray-700 border-dashed"
            }`}
            title={
              "The track conditions the entire clip. A longer track is trimmed to the clip; " +
              "a shorter one is refused, not padded."
            }
          >
            <span className="absolute left-2 top-0 text-[10px] leading-6 text-gray-300 truncate max-w-[70%]">
              {inputAudio ? inputAudio.name : "No track — the soundtrack is generated with the video"}
            </span>
            {inputAudio && (
              <span className="absolute right-2 top-0 text-[10px] leading-6 text-gray-400">
                whole clip
              </span>
            )}
          </div>
          {inputAudio && (
            <p className={`text-xs ${audioTooShort ? "text-amber-400" : "text-gray-400"}`}>
              {audioSeconds !== null
                ? `Track ${audioSeconds.toFixed(2)}s / clip ${clipSeconds.toFixed(2)}s. `
                : `Clip ${clipSeconds.toFixed(2)}s. `}
              {audioTooShort
                ? "The track is too short; this request will be refused."
                : audioEnabled
                  ? "Whole-clip conditioning and soundtrack."
                  : "Audio output is off; conditioning remains active."}
            </p>
          )}
        </div>
      )}

      {collisions.length > 0 && (
        <p className="text-xs text-amber-400">
          Two anchors are on frame {collisions.join(", ")}. One frame holds one
          anchor; the server refuses a duplicate placement.
        </p>
      )}
      {notice && <p className="text-xs text-amber-400">{notice}</p>}

    </div>
  );
}
