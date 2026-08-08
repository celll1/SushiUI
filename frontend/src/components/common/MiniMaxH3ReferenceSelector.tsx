"use client";

import { ReactNode, useRef } from "react";
import Card from "./Card";
import Button from "./Button";
import Select from "./Select";
import { MiniMaxH3References } from "@/utils/api";

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
}: MiniMaxH3ReferenceSelectorProps) {
  const imageInput = useRef<HTMLInputElement>(null);
  const videoInput = useRef<HTMLInputElement>(null);
  const audioInput = useRef<HTMLInputElement>(null);
  const soundtrackInput = useRef<HTMLInputElement>(null);
  const soundtrackTarget = useRef<number>(-1);

  const total = countMiniMaxH3References(value);
  const remaining = (imagesOnly ? MAX_IMAGES : MAX_TOTAL) - total;

  const add = (kind: "images" | "videos" | "audios", files: FileList | null) => {
    if (!files || files.length === 0) return;
    const perKindMax = kind === "images" ? MAX_IMAGES : kind === "videos" ? MAX_VIDEOS : MAX_AUDIOS;
    const room = Math.min(perKindMax - value[kind].length, remaining);
    const added = Array.from(files).slice(0, Math.max(0, room));
    if (added.length === 0) return;
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
    <div key={`${kind}-${index}`} className="flex items-center gap-2 text-xs text-gray-300 py-1">
      <span className="text-gray-500 w-20 shrink-0">{label}</span>
      <span className="truncate flex-1" title={file.name}>{file.name}</span>
      {extra}
      <button
        type="button"
        className="text-gray-500 hover:text-gray-200 px-1"
        disabled={disabled || index === 0}
        onClick={() => move(kind, index, -1)}
        title="Move earlier (the order is part of the request)"
      >
        ↑
      </button>
      <button
        type="button"
        className="text-gray-500 hover:text-gray-200 px-1"
        disabled={disabled || index === value[kind].length - 1}
        onClick={() => move(kind, index, 1)}
        title="Move later (the order is part of the request)"
      >
        ↓
      </button>
      <button
        type="button"
        className="text-red-400 hover:text-red-300 px-1"
        disabled={disabled}
        onClick={() => remove(kind, index)}
        title="Remove"
      >
        ✕
      </button>
    </div>
  );

  return (
    <Card title={`References (MiniMax-H3 ref2va) — ${total}/${imagesOnly ? MAX_IMAGES : MAX_TOTAL}`}>
      <div className="space-y-3">
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

        {/* Images */}
        <div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-300">
              Images ({value.images.length}/{MAX_IMAGES}) — &lt;Picture i&gt;
            </span>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => imageInput.current?.click()}
              disabled={disabled || value.images.length >= MAX_IMAGES || remaining <= 0}
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
            <span className="text-sm text-gray-300">
              Videos ({value.videos.length}/{MAX_VIDEOS}) — &lt;Video k&gt;
            </span>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => videoInput.current?.click()}
              disabled={disabled || value.videos.length >= MAX_VIDEOS || remaining <= 0}
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
                disabled={disabled}
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
            <span className="text-sm text-gray-300">
              Audio ({value.audios.length}/{MAX_AUDIOS}) — &lt;Audio j&gt;
            </span>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => audioInput.current?.click()}
              disabled={disabled || value.audios.length >= MAX_AUDIOS || remaining <= 0}
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
          {value.audios.length > 0 && value.images.length === 0 && value.videos.length === 0 && (
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
          disabled={disabled}
        />
        <p className="text-xs text-gray-500">
          A reference&apos;s rows ride through every sampling step, so a larger
          image reference lengthens the packed sequence for the whole
          generation.{" "}
          {imagesOnly
            ? "The automatic source-clip video reference is unaffected: it always follows the canvas rule the generated video follows."
            : "Video references are unaffected: they always follow the canvas rule the generated video follows."}
        </p>
      </div>
    </Card>
  );
}
