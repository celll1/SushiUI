"use client";

import { useRef } from "react";
import Card from "./Card";
import Button from "./Button";
import { MiniMaxH3Keyframe, toBase64 } from "@/utils/api";

/**
 * Optional keyframe anchors for a `/generate/ref2vid` request (C5: anchors x
 * references). A SEPARATE track from the reference list above it: a reference
 * is content conditioning (read by the prompt, no placement guarantee); an
 * anchor here is placement conditioning (pinned to one exact pixel frame),
 * laid out AFTER every reference block.
 *
 * Deliberately NOT `MiniMaxH3KeyframeTimeline`: that component has a primary
 * "input image" chip (`/generate/img2vid`'s required `image` field), which
 * ref2vid has no equivalent of — every anchor here is an "additional" one, so
 * this is a plain list, not a timeline with a distinguished first chip.
 */

interface MiniMaxH3Ref2VidKeyframeSelectorProps {
  value: MiniMaxH3Keyframe[];
  onChange: (keyframes: MiniMaxH3Keyframe[]) => void;
  disabled?: boolean;
}

export default function MiniMaxH3Ref2VidKeyframeSelector({
  value,
  onChange,
  disabled = false,
}: MiniMaxH3Ref2VidKeyframeSelectorProps) {
  const fileInput = useRef<HTMLInputElement>(null);

  const add = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const dataUrl = await toBase64(files[0]);
    onChange([...value, { image: dataUrl, frame_index: 0 }]);
  };

  const setFrameIndex = (index: number, frameIndex: number) => {
    onChange(value.map((keyframe, i) => (i === index ? { ...keyframe, frame_index: frameIndex } : keyframe)));
  };

  const remove = (index: number) => {
    onChange(value.filter((_keyframe, i) => i !== index));
  };

  return (
    <Card title={`Keyframe anchors (optional) — ${value.length}`}>
      <div className="space-y-2">
        <p className="text-xs text-gray-400">
          Placed on the generated clip AFTER every reference block, each
          pinned to one exact pixel frame (0 = first frame, -1 = the resolved
          last frame). Combining anchors with references is beyond MiniMax's
          model card and always warns.
        </p>
        <input
          ref={fileInput}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            add(e.target.files);
            e.target.value = "";
          }}
        />
        <Button
          variant="secondary"
          size="sm"
          onClick={() => fileInput.current?.click()}
          disabled={disabled}
        >
          Add anchor
        </Button>
        {value.map((keyframe, index) => (
          <div key={index} className="flex items-center gap-2 text-xs text-gray-300 py-1">
            {typeof keyframe.image === "string" && (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={keyframe.image} alt="" className="w-10 h-10 object-cover rounded" />
            )}
            <span className="text-gray-500">Frame</span>
            <input
              type="number"
              className="w-20 bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-gray-200"
              value={keyframe.frame_index}
              disabled={disabled}
              onChange={(e) => setFrameIndex(index, parseInt(e.target.value, 10) || 0)}
            />
            <button
              type="button"
              className="text-red-400 hover:text-red-300 px-1 ml-auto"
              disabled={disabled}
              onClick={() => remove(index)}
              title="Remove"
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </Card>
  );
}
