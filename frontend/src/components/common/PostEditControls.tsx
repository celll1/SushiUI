"use client";

import { ChevronDown, ChevronRight, RotateCcw } from "lucide-react";
import { PostEditState, NEUTRAL_POST_EDIT, isNeutral } from "@/utils/postEdit";
import NumberInput from "./NumberInput";

interface PostEditControlsProps {
  value: PostEditState;
  onChange: (value: PostEditState) => void;
  /** Optional extra classes for the container. */
  className?: string;
  /**
   * Layout variant:
   * - "compact" (default): single wrapping row `B [slider][num] S ... [reset]`.
   *   Used in space-constrained overlays (full-size image popup).
   * - "stacked": each control is a full-width labeled row with a larger touch
   *   target. Used in the gallery detail sidebar.
   */
  variant?: "compact" | "stacked";
  /**
   * Collapsible header (stacked variant only). When `collapsed` is provided,
   * the "Post-edit" header becomes a toggle button and the slider rows are
   * hidden while collapsed. The Reset button stays visible in the header so a
   * non-neutral edit state remains discoverable when collapsed.
   */
  collapsed?: boolean;
  onToggleCollapsed?: () => void;
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
export default function PostEditControls({ value, onChange, className = "", variant = "compact", collapsed, onToggleCollapsed }: PostEditControlsProps) {
  const nonNeutral = !isNeutral(value);

  if (variant === "stacked") {
    const resetButton = (
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onChange({ ...NEUTRAL_POST_EDIT });
        }}
        disabled={!nonNeutral}
        className={`flex-shrink-0 flex items-center gap-1 px-2 py-1 rounded text-xs ${
          nonNeutral
            ? "text-blue-400 hover:text-blue-300 hover:bg-gray-700"
            : "text-gray-600 cursor-default"
        }`}
        title="Reset brightness, saturation and color flatten"
      >
        <RotateCcw className="h-3.5 w-3.5" />
        Reset
      </button>
    );

    const rows: {
      id: string;
      label: string;
      title: string;
      sliderMax: number;
      numberMax: number;
      numberDefault: number;
      valueKey: "brightness" | "saturation" | "flatten";
    }[] = [
      { id: "post-edit-brightness-range", label: "Brightness (%)", title: "Brightness (%)", sliderMax: 200, numberMax: 100000, numberDefault: 100, valueKey: "brightness" },
      { id: "post-edit-saturation-range", label: "Saturation (%)", title: "Saturation (%)", sliderMax: 200, numberMax: 100000, numberDefault: 100, valueKey: "saturation" },
      { id: "post-edit-flatten-range", label: "Color flatten（色ムラ除去）", title: "Color flatten（色ムラ除去）", sliderMax: 100, numberMax: 1000, numberDefault: 0, valueKey: "flatten" },
    ];

    const isCollapsible = collapsed !== undefined;

    return (
      <div className={`space-y-3 ${className}`}>
        <div className="flex items-center justify-between">
          {isCollapsible ? (
            <button
              type="button"
              onClick={onToggleCollapsed}
              className="flex items-center gap-1 text-xs font-medium text-gray-300 hover:text-white"
              title={collapsed ? "Show post-edit controls" : "Hide post-edit controls"}
            >
              {collapsed ? <ChevronRight className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
              Post-edit
            </button>
          ) : (
            <span className="text-xs font-medium text-gray-300">Post-edit</span>
          )}
          {resetButton}
        </div>
        {!(isCollapsible && collapsed) && rows.map((row) => (
          <div key={row.id} className="space-y-1">
            <label htmlFor={row.id} className="block text-xs text-gray-400" title={row.title}>
              {row.label}
            </label>
            <div className="flex items-center gap-2">
              <input
                id={row.id}
                type="range"
                min={0}
                max={row.sliderMax}
                step={1}
                value={value[row.valueKey]}
                onChange={(e) => onChange({ ...value, [row.valueKey]: parseInt(e.target.value, 10) })}
                className="flex-1 min-w-0 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-600 [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:hover:bg-blue-700 [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-blue-600 [&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:hover:bg-blue-700 [&::-moz-range-thumb]:border-0"
                title={row.title}
              />
              <NumberInput
                label={row.title}
                value={value[row.valueKey]}
                onCommit={(v) => onChange({ ...value, [row.valueKey]: v })}
                defaultValue={row.numberDefault}
                min={0}
                max={row.numberMax}
                step={1}
                parse="int"
                className="w-16 flex-shrink-0"
              />
            </div>
          </div>
        ))}
      </div>
    );
  }

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
      {/* Sliders stay at their everyday ranges; the number boxes allow far
          larger values for diagnostics (see saturation note below). */}
      <NumberInput
        label="Brightness (%)"
        value={value.brightness}
        onCommit={(brightness) => onChange({ ...value, brightness })}
        defaultValue={100}
        min={0}
        max={100000}
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
      {/* Typed values above 100 extrapolate the smoothing radius/eps for
          diagnostics (blend is capped at 1.0 inside flattenChroma so chroma
          never inverts); the slider stays 0-100. */}
      <NumberInput
        label="Color flatten（色ムラ除去）"
        value={value.flatten}
        onCommit={(flatten) => onChange({ ...value, flatten })}
        defaultValue={0}
        min={0}
        max={1000}
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
