"use client";

import { RotateCcw } from "lucide-react";
import { PostEditState, NEUTRAL_POST_EDIT, isNeutral } from "@/utils/postEdit";
import NumberInput from "./NumberInput";

interface PostEditControlsProps {
  value: PostEditState;
  onChange: (value: PostEditState) => void;
  /** Optional extra classes for the container. */
  className?: string;
}

// Shared thumb styling for the bare range inputs (kept in sync with Slider.tsx's
// look, but this component intentionally does NOT use the shared Slider
// component -- Slider's own number box has the unclearable-zero anti-pattern
// (parseInt(...) || 0 on every keystroke), and we don't want to touch Slider
// globally since other consumers depend on its current behavior).
const RANGE_CLASSNAME =
  "flex-1 min-w-0 h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer " +
  "[&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5 " +
  "[&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-600 [&::-webkit-slider-thumb]:cursor-pointer " +
  "[&::-webkit-slider-thumb]:hover:bg-blue-700 " +
  "[&::-moz-range-thumb]:w-3.5 [&::-moz-range-thumb]:h-3.5 [&::-moz-range-thumb]:rounded-full " +
  "[&::-moz-range-thumb]:bg-blue-600 [&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:hover:bg-blue-700 " +
  "[&::-moz-range-thumb]:border-0";

/**
 * Single-line compact post-edit row: `B [slider][num]  S [slider][num]  [reset]`.
 *
 * Deliberately NOT built on the shared Slider component (see RANGE_CLASSNAME
 * comment) -- the number box next to each slider is a plain NumberInput
 * (common/NumberInput.tsx), which lets the field be freely cleared/retyped
 * instead of snapping back to a coerced value on every keystroke.
 */
export default function PostEditControls({ value, onChange, className = "" }: PostEditControlsProps) {
  const nonNeutral = !isNeutral(value);

  return (
    <div className={`flex items-center gap-2 flex-wrap ${className}`}>
      <label
        htmlFor="post-edit-brightness-range"
        className="text-xs text-gray-400 font-mono flex-shrink-0"
        title="Brightness (%)"
      >
        B
      </label>
      <input
        id="post-edit-brightness-range"
        type="range"
        min={0}
        max={200}
        step={1}
        value={value.brightness}
        onChange={(e) => onChange({ ...value, brightness: parseInt(e.target.value, 10) })}
        className={RANGE_CLASSNAME}
        title="Brightness (%)"
      />
      <NumberInput
        label="Brightness (%)"
        value={value.brightness}
        onCommit={(brightness) => onChange({ ...value, brightness })}
        defaultValue={100}
        min={0}
        max={200}
        step={1}
        parse="int"
        className="w-14 flex-shrink-0"
      />

      <label
        htmlFor="post-edit-saturation-range"
        className="text-xs text-gray-400 font-mono flex-shrink-0"
        title="Saturation (%)"
      >
        S
      </label>
      <input
        id="post-edit-saturation-range"
        type="range"
        min={0}
        max={200}
        step={1}
        value={value.saturation}
        onChange={(e) => onChange({ ...value, saturation: parseInt(e.target.value, 10) })}
        className={RANGE_CLASSNAME}
        title="Saturation (%)"
      />
      {/* Slider stays 0-200 for everyday use, but the number box allows far
          larger values for diagnostics (e.g. sat 10000 makes residual chroma
          mottling obvious). A typed value above 200 simply pins the slider
          thumb at its max while the real value applies. */}
      <NumberInput
        label="Saturation (%)"
        value={value.saturation}
        onCommit={(saturation) => onChange({ ...value, saturation })}
        defaultValue={100}
        min={0}
        max={100000}
        step={1}
        parse="int"
        className="w-14 flex-shrink-0"
      />

      <label
        htmlFor="post-edit-flatten-range"
        className="text-xs text-gray-400 font-mono flex-shrink-0"
        title="Color flatten（色ムラ除去）"
      >
        F
      </label>
      <input
        id="post-edit-flatten-range"
        type="range"
        min={0}
        max={100}
        step={1}
        value={value.flatten}
        onChange={(e) => onChange({ ...value, flatten: parseInt(e.target.value, 10) })}
        className={RANGE_CLASSNAME}
        title="Color flatten（色ムラ除去）"
      />
      <NumberInput
        label="Color flatten（色ムラ除去）"
        value={value.flatten}
        onCommit={(flatten) => onChange({ ...value, flatten })}
        defaultValue={0}
        min={0}
        max={100}
        step={1}
        parse="int"
        className="w-14 flex-shrink-0"
      />

      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onChange({ ...NEUTRAL_POST_EDIT });
        }}
        disabled={!nonNeutral}
        className={`flex-shrink-0 p-1 rounded ${
          nonNeutral
            ? "text-blue-400 hover:text-blue-300 hover:bg-gray-700"
            : "text-gray-600 cursor-default"
        }`}
        title="Reset brightness, saturation and color flatten"
      >
        <RotateCcw className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
