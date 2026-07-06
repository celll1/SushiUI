"use client";

import { PostEditState, NEUTRAL_POST_EDIT, isNeutral } from "@/utils/postEdit";
import Slider from "./Slider";

interface PostEditControlsProps {
  value: PostEditState;
  onChange: (value: PostEditState) => void;
  /** Optional extra classes for the container. */
  className?: string;
}

/**
 * Compact two-slider panel (Brightness + Saturation) for client-side image
 * post-editing. Defaults are neutral (100%). A Reset button appears only when
 * the current state is non-neutral. Small footprint, suitable under an image.
 */
export default function PostEditControls({ value, onChange, className = "" }: PostEditControlsProps) {
  const nonNeutral = !isNeutral(value);

  return (
    <div className={`space-y-2 ${className}`}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-gray-400 uppercase tracking-wide">
          Post-Edit
        </span>
        {nonNeutral && (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onChange({ ...NEUTRAL_POST_EDIT });
            }}
            className="text-xs text-blue-400 hover:text-blue-300"
            title="Reset brightness and saturation"
          >
            Reset
          </button>
        )}
      </div>
      <Slider
        label="Brightness"
        min={0}
        max={200}
        step={1}
        value={value.brightness}
        onChange={(e) => onChange({ ...value, brightness: parseInt(e.target.value, 10) || 0 })}
      />
      <Slider
        label="Saturation"
        min={0}
        max={200}
        step={1}
        value={value.saturation}
        onChange={(e) => onChange({ ...value, saturation: parseInt(e.target.value, 10) || 0 })}
      />
    </div>
  );
}
