"use client";

import { ReactNode, useEffect, useRef, useState } from "react";
import Card from "./Card";
import Button from "./Button";
import Select from "./Select";
import { MiniMaxH3References } from "@/utils/api";
import { persistH3References, restoreH3References } from "@/utils/h3ReferenceStorage";

/**
 * Reference inputs for MiniMax-H3's `ref2va` (omni-reference) workflow.
 *
 * Lives with the other reference-style inputs (ControlNetSelector,
 * LoRASelector) and is shown only when the loaded MiniMax-H3 checkpoint is the
 * `ref2va` transformer variant — the `fl2va` one, which serves txt2vid /
 * img2vid / video outpaint, was never trained to read reference rows, and the
 * backend refuses the request by name rather than running it.
 *
 * THE ORDER IS SEMANTIC. It fixes the `<Picture i>` / `<Audio j>` /
 * `<Video k>` labels the prompt refers to, and it lays the references out on
 * the packed sequence's shared rotary clock, so reordering them is a different
 * request. Nothing here sorts or regroups: each list is sent in the order it is
 * shown, and the packed order is images, then videos (each preceded by its own
 * soundtrack), then standalone audio.
 *
 * The limits (9 images, 3 videos, 3 audio, 12 in total, and never audio alone)
 * are the released checkpoint's. They are enforced server-side with the reason;
 * this component only keeps the UI from building a request that is already
 * known to be refused.
 */

export const MAX_IMAGES = 9;
export const MAX_VIDEOS = 3;
export const MAX_AUDIOS = 3;
export const MAX_TOTAL = 12;

interface MiniMaxH3ReferenceSelectorProps {
  value: MiniMaxH3References;
  onChange: (references: MiniMaxH3References) => void;
  referenceImageSize: "max" | "match";
  onReferenceImageSizeChange: (size: "max" | "match") => void;
  disabled?: boolean;
  // Video outpaint's ref2va surface (extend_forward only): the preserved
  // clip is ALWAYS the sole video reference there, so that endpoint has no
  // reference_videos/reference_audios field. Hides those two sections and
  // the title/copy that would otherwise describe them.
  imagesOnly?: boolean;
  // Temporal inpaint's ref2va surface only: the preserved frames outside the
  // regenerate range already condition the vision stream, so an audio-only
  // reference set is not refused there (unlike /generate/ref2vid and video
  // outpaint, where it is). Suppresses the "cannot be the only kind sent"
  // notice below, which would otherwise describe a restriction this endpoint
  // does not have.
  allowAudioAlone?: boolean;
  // Stable per-panel key for persisting File-backed references across panel
  // unmounts and browser reloads.
  storageKey?: string;
}

export const EMPTY_MINIMAX_H3_REFERENCES: MiniMaxH3References = {
  images: [],
  videos: [],
  videoAudios: [],
  audios: [],
};

export function countMiniMaxH3References(references: MiniMaxH3References): number {
  return (
    (references.images?.length || 0) +
    (references.videos?.length || 0) +
    (references.audios?.length || 0)
  );
}

export default function MiniMaxH3ReferenceSelector({
  value,
  onChange,
  referenceImageSize,
  onReferenceImageSizeChange,
  disabled = false,
  imagesOnly = false,
  allowAudioAlone = false,
  storageKey,
}: MiniMaxH3ReferenceSelectorProps) {
  const imageInput = useRef<HTMLInputElement>(null);
  const videoInput = useRef<HTMLInputElement>(null);
  const audioInput = useRef<HTMLInputElement>(null);
  const soundtrackInput = useRef<HTMLInputElement>(null);
  const soundtrackTarget = useRef<number>(-1);
  const onChangeRef = useRef(onChange);
  const onReferenceImageSizeChangeRef = useRef(onReferenceImageSizeChange);
  const [isDragging, setIsDragging] = useState(false);
  const [isRestoring, setIsRestoring] = useState(Boolean(storageKey));
  // Reports what a drop skipped and why. Cleared on the next successful
  // add (button or drop) rather than on a timer, so it stays visible until
  // the user has done something that could have addressed it.
  const [dropNotice, setDropNotice] = useState<string | null>(null);

  useEffect(() => {
    onChangeRef.current = onChange;
    onReferenceImageSizeChangeRef.current = onReferenceImageSizeChange;
  }, [onChange, onReferenceImageSizeChange]);

  useEffect(() => {
    if (!storageKey) {
      setIsRestoring(false);
      return;
    }
    let cancelled = false;
    setIsRestoring(true);
    void restoreH3References(storageKey)
      .then((stored) => {
        if (cancelled || !stored) return;
        onChangeRef.current(stored.references);
        onReferenceImageSizeChangeRef.current(stored.referenceImageSize);
      })
      .catch((error) => {
        console.error("Failed to restore MiniMax-H3 references:", error);
      })
      .finally(() => {
        if (!cancelled) setIsRestoring(false);
      });
    return () => {
      cancelled = true;
    };
  }, [storageKey]);

  useEffect(() => {
    if (!storageKey || isRestoring) return;
    void persistH3References(storageKey, value, referenceImageSize).catch((error) => {
      console.error("Failed to persist MiniMax-H3 references:", error);
    });
  }, [storageKey, value, referenceImageSize, isRestoring]);

  const total = countMiniMaxH3References(value);
  const remaining = (imagesOnly ? MAX_IMAGES : MAX_TOTAL) - total;
  const kindLabel = (kind: "images" | "videos" | "audios") =>
    kind === "images" ? "image" : kind === "videos" ? "video" : "audio";
  const kindMax = (kind: "images" | "videos" | "audios") =>
    kind === "images" ? MAX_IMAGES : kind === "videos" ? MAX_VIDEOS : MAX_AUDIOS;
  const inputDisabled = disabled || isRestoring;

  // Routes a drop by MIME type into the matching bucket (images/videos/
  // audios), each capped at its own max and at the shared total. A file
  // that matches none of the three -- or a video/audio when imagesOnly --
  // is refused rather than misfiled into the wrong list; both that refusal
  // and any cap truncation are reported in dropNotice, since a drop that
  // silently does less than it looks like reads as broken.
  const addDropped = (files: FileList | null) => {
    if (inputDisabled || !files || files.length === 0) return;
    const buckets: { images: File[]; videos: File[]; audios: File[] } = {
      images: [],
      videos: [],
      audios: [],
    };
    let unsupported = 0;
    for (const file of Array.from(files)) {
      if (file.type.startsWith("image/")) buckets.images.push(file);
      else if (!imagesOnly && file.type.startsWith("video/")) buckets.videos.push(file);
      else if (!imagesOnly && file.type.startsWith("audio/")) buckets.audios.push(file);
      else unsupported++;
    }

    let room = remaining;
    const next: MiniMaxH3References = { ...value };
    const overLimit: Record<"images" | "videos" | "audios", { count: number; atKindCap: boolean }> = {
      images: { count: 0, atKindCap: false },
      videos: { count: 0, atKindCap: false },
      audios: { count: 0, atKindCap: false },
    };
    (["images", "videos", "audios"] as const).forEach((kind) => {
      const kindFiles = buckets[kind];
      if (kindFiles.length === 0) return;
      const perKindMax = kindMax(kind);
      const kindRoom = perKindMax - next[kind].length;
      const perKindRoom = Math.min(kindRoom, room);
      const added = kindFiles.slice(0, Math.max(0, perKindRoom));
      const skipped = kindFiles.length - added.length;
      if (skipped > 0) overLimit[kind] = { count: skipped, atKindCap: kindRoom <= room };
      if (added.length === 0) return;
      next[kind] = [...next[kind], ...added];
      if (kind === "videos") next.videoAudios = [...next.videoAudios, ...added.map(() => null)];
      room -= added.length;
    });
    onChange(next);

    const notices: string[] = [];
    if (unsupported > 0) {
      notices.push(
        `${unsupported} file${unsupported === 1 ? "" : "s"} skipped: ${
          imagesOnly ? "this surface takes images only" : "unsupported file type"
        }.`,
      );
    }
    const overLimitNotices = (["images", "videos", "audios"] as const)
      .filter((kind) => overLimit[kind].count > 0)
      .map((kind) => {
        const { count, atKindCap } = overLimit[kind];
        const limit = atKindCap
          ? `the ${kindLabel(kind)} limit of ${kindMax(kind)}`
          : `the total limit of ${imagesOnly ? MAX_IMAGES : MAX_TOTAL}`;
        return `${count} ${kindLabel(kind)}${count === 1 ? "" : "s"} skipped: over ${limit}.`;
      });
    notices.push(...overLimitNotices);
    setDropNotice(notices.length > 0 ? notices.join(" ") : null);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!inputDisabled) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!e.currentTarget.contains(e.relatedTarget as Node)) setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    addDropped(e.dataTransfer.files);
  };

  const add = (kind: "images" | "videos" | "audios", files: FileList | null) => {
    if (inputDisabled || !files || files.length === 0) return;
    const perKindMax = kind === "images" ? MAX_IMAGES : kind === "videos" ? MAX_VIDEOS : MAX_AUDIOS;
    const room = Math.min(perKindMax - value[kind].length, remaining);
    const added = Array.from(files).slice(0, Math.max(0, room));
    if (added.length === 0) return;
    setDropNotice(null);
    const next: MiniMaxH3References = { ...value, [kind]: [...value[kind], ...added] };
    if (kind === "videos") {
      // A video's soundtrack slot is positional, so it grows with the list.
      next.videoAudios = [...value.videoAudios, ...added.map(() => null)];
    }
    onChange(next);
  };

  const remove = (kind: "images" | "videos" | "audios", index: number) => {
    const next: MiniMaxH3References = {
      ...value,
      [kind]: value[kind].filter((_file, i) => i !== index),
    };
    if (kind === "videos") {
      next.videoAudios = value.videoAudios.filter((_file, i) => i !== index);
    }
    onChange(next);
  };

  const move = (kind: "images" | "videos" | "audios", index: number, delta: number) => {
    const target = index + delta;
    if (target < 0 || target >= value[kind].length) return;
    const list = [...value[kind]];
    [list[index], list[target]] = [list[target], list[index]];
    const next: MiniMaxH3References = { ...value, [kind]: list };
    if (kind === "videos") {
      const soundtracks = [...value.videoAudios];
      [soundtracks[index], soundtracks[target]] = [soundtracks[target], soundtracks[index]];
      next.videoAudios = soundtracks;
    }
    onChange(next);
  };

  const setSoundtrack = (index: number, file: File | null) => {
    const soundtracks = [...value.videoAudios];
    while (soundtracks.length < value.videos.length) soundtracks.push(null);
    soundtracks[index] = file;
    onChange({ ...value, videoAudios: soundtracks });
  };

  const row = (
    label: string,
    kind: "images" | "videos" | "audios",
    index: number,
    file: File,
    extra?: ReactNode,
  ) => (
    <div key={`${kind}-${index}`} className="flex items-center gap-2 py-0.5 text-xs text-gray-300">
      <span className="text-gray-500 w-20 shrink-0">{label}</span>
      <span className="truncate flex-1" title={file.name}>{file.name}</span>
      {extra}
      <button
        type="button"
        className="text-gray-500 hover:text-gray-200 px-1"
        disabled={inputDisabled || index === 0}
        onClick={() => move(kind, index, -1)}
        title="Move earlier (the order is part of the request)"
      >
        ↑
      </button>
      <button
        type="button"
        className="text-gray-500 hover:text-gray-200 px-1"
        disabled={inputDisabled || index === value[kind].length - 1}
        onClick={() => move(kind, index, 1)}
        title="Move later (the order is part of the request)"
      >
        ↓
      </button>
      <button
        type="button"
        className="text-red-400 hover:text-red-300 px-1"
        disabled={inputDisabled}
        onClick={() => remove(kind, index)}
        title="Remove"
      >
        ✕
      </button>
    </div>
  );

  return (
    <Card title={`References (MiniMax-H3 ref2va) — ${total}/${imagesOnly ? MAX_IMAGES : MAX_TOTAL}`}>
      <div
        className={`space-y-2 rounded-lg transition-colors ${
          isDragging ? "ring-2 ring-blue-500 bg-gray-800/50" : ""
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {isDragging && (
          <p className="text-xs text-blue-400 text-center border border-dashed border-blue-500 rounded py-2">
            Drop {imagesOnly ? "images" : "images, videos or audio"} here —
            sorted by file type, in drop order
          </p>
        )}
        {!isDragging && dropNotice && (
          <p className="text-xs text-amber-400">{dropNotice}</p>
        )}
        <details className="rounded border border-gray-800 bg-gray-950/40 px-2 py-1 text-xs text-gray-500">
          <summary className="cursor-pointer select-none text-gray-400">Reference labels and limits</summary>
        {imagesOnly ? (
          <p className="text-xs text-gray-400">
            Up to 9 image references, read in upload order and shown to the
            model as <code>&lt;Picture i&gt;</code>. The preserved clip is
            always the video reference here; there is no separate video/audio
            reference slot on this endpoint.
          </p>
        ) : (
          <p className="text-xs text-gray-400">
            Up to 9 images, 3 videos and 3 audio clips, 12 files in total. Refer to
            them in the prompt by the labels they are given here —{" "}
            <code>&lt;Picture i&gt;</code>, <code>&lt;Video k&gt;</code>,{" "}
            <code>&lt;Audio j&gt;</code>. The order is part of the request.
          </p>
        )}
        </details>

        {/* Images */}
        <div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-300">
              Images ({value.images.length}/{MAX_IMAGES}) — &lt;Picture i&gt;
            </span>
            <Button
              variant="secondary"
              size="xs"
              onClick={() => imageInput.current?.click()}
              disabled={inputDisabled || value.images.length >= MAX_IMAGES || remaining <= 0}
            >
              Add
            </Button>
          </div>
          <input
            ref={imageInput}
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={(e) => {
              add("images", e.target.files);
              e.target.value = "";
            }}
          />
          {value.images.map((file, index) => row(`Picture ${index + 1}`, "images", index, file))}
        </div>

        {/* Videos + their positional soundtracks */}
        {!imagesOnly && (
        <div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-300">
              Videos ({value.videos.length}/{MAX_VIDEOS}) — &lt;Video k&gt;
            </span>
            <Button
              variant="secondary"
              size="xs"
              onClick={() => videoInput.current?.click()}
              disabled={inputDisabled || value.videos.length >= MAX_VIDEOS || remaining <= 0}
            >
              Add
            </Button>
          </div>
          <input
            ref={videoInput}
            type="file"
            accept="video/*"
            multiple
            className="hidden"
            onChange={(e) => {
              add("videos", e.target.files);
              e.target.value = "";
            }}
          />
          <input
            ref={soundtrackInput}
            type="file"
            accept="audio/*"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0] ?? null;
              if (soundtrackTarget.current >= 0) setSoundtrack(soundtrackTarget.current, file);
              soundtrackTarget.current = -1;
              e.target.value = "";
            }}
          />
          {value.videos.map((file, index) =>
            row(
              `Video ${index + 1}`,
              "videos",
              index,
              file,
              <button
                type="button"
                className="text-gray-400 hover:text-gray-200 px-1 whitespace-nowrap"
                disabled={inputDisabled}
                onClick={() => {
                  soundtrackTarget.current = index;
                  soundtrackInput.current?.click();
                }}
                title="Soundtrack of THIS reference video: it is conditioned on as the video's own, packed immediately before it. A reference clip's embedded audio is not read automatically."
              >
                {value.videoAudios[index] ? "♪ " + value.videoAudios[index]!.name : "+ soundtrack"}
              </button>,
            ),
          )}
        </div>
        )}

        {/* Standalone audio */}
        {!imagesOnly && (
        <div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-300">
              Audio ({value.audios.length}/{MAX_AUDIOS}) — &lt;Audio j&gt;
            </span>
            <Button
              variant="secondary"
              size="xs"
              onClick={() => audioInput.current?.click()}
              disabled={inputDisabled || value.audios.length >= MAX_AUDIOS || remaining <= 0}
            >
              Add
            </Button>
          </div>
          <input
            ref={audioInput}
            type="file"
            accept="audio/*"
            multiple
            className="hidden"
            onChange={(e) => {
              add("audios", e.target.files);
              e.target.value = "";
            }}
          />
          {value.audios.map((file, index) => row(`Audio ${index + 1}`, "audios", index, file))}
          {!allowAudioAlone && value.audios.length > 0 && value.images.length === 0 && value.videos.length === 0 && (
            <p className="text-xs text-amber-400 mt-1">
              An audio reference cannot be the only kind sent: it never reaches
              the conditioner, so the vision stream would be conditioned on
              nothing. Add an image or a video reference.
            </p>
          )}
        </div>
        )}

        <Select
          label="Image reference size"
          value={referenceImageSize}
          onChange={(e) => onReferenceImageSizeChange(e.target.value as "max" | "match")}
          options={[
            { value: "max", label: "max — 2048px short edge (the released recipe)" },
            { value: "match", label: "match — scale down to the generation's pixel area" },
          ]}
          disabled={inputDisabled}
        />
        <details className="text-xs text-gray-500">
          <summary className="cursor-pointer select-none text-gray-400">Reference performance note</summary>
        <p className="mt-1">
          A reference&apos;s rows ride through every sampling step, so a larger
          image reference lengthens the packed sequence for the whole
          generation.{" "}
          {imagesOnly
            ? "The automatic source-clip video reference is unaffected: it always follows the canvas rule the generated video follows."
            : "Video references are unaffected: they always follow the canvas rule the generated video follows."}
        </p>
        </details>
      </div>
    </Card>
  );
}
