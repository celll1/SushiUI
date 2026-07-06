"use client";

import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

interface NumberInputProps {
  id?: string;
  label?: string;
  /** Current committed value (from params/state). */
  value: number;
  /** Called with the parsed number whenever the draft text parses to a valid number. */
  onCommit: (value: number) => void;
  /**
   * Value to commit when the field is blurred while empty or unparsable.
   * Defaults to `value` (i.e. revert to the last committed value).
   */
  defaultValue?: number;
  min?: number;
  max?: number;
  step?: number;
  parse?: "int" | "float";
  placeholder?: string;
  className?: string;
  disabled?: boolean;
}

/**
 * Numeric input that lets the user freely clear/retype the field.
 *
 * The classic bug this avoids: binding `value={someNumber}` directly to the
 * <input> and coercing empty/partial text to 0 in onChange (e.g.
 * `parseInt(e.target.value) || 0`) makes it impossible to clear the field —
 * typing "1024" over a "0" default yields "01024" because the input is
 * immediately snapped back to a numeric value after every keystroke.
 *
 * Fix: keep an independent string "draft" while the field has focus. Only
 * push the committed number back into the draft when the field is not
 * focused (i.e. the prop changed from outside). While typing, call
 * `onCommit` as soon as the draft parses to a valid number (preserving live
 * update wiring), but allow empty/partial/invalid text to sit in the field
 * without being clobbered. On blur, an empty/unparsable draft resolves to
 * `defaultValue` (or the last committed `value` if not provided).
 */
export default function NumberInput({
  id,
  label,
  value,
  onCommit,
  defaultValue,
  min,
  max,
  step,
  parse = "int",
  placeholder,
  className,
  disabled,
}: NumberInputProps) {
  const [draft, setDraft] = useState<string>(String(value));
  const focusedRef = useRef(false);

  // Sync draft from the external value only while the field is not focused,
  // so we don't clobber what the user is typing.
  useEffect(() => {
    if (!focusedRef.current) {
      setDraft(String(value));
    }
  }, [value]);

  const parseValue = (text: string): number | null => {
    if (text.trim() === "") return null;
    const parsed = parse === "float" ? parseFloat(text) : parseInt(text, 10);
    if (isNaN(parsed)) return null;
    return parsed;
  };

  const clamp = (num: number): number => {
    let clamped = num;
    if (min !== undefined && clamped < min) clamped = min;
    if (max !== undefined && clamped > max) clamped = max;
    return clamped;
  };

  return (
    <input
      id={id}
      type="number"
      min={min}
      max={max}
      step={step}
      placeholder={placeholder}
      disabled={disabled}
      aria-label={label}
      className={cn(
        "px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs",
        className
      )}
      value={draft}
      onFocus={() => {
        focusedRef.current = true;
      }}
      onChange={(e) => {
        const text = e.target.value;
        setDraft(text);
        const parsed = parseValue(text);
        if (parsed !== null) {
          onCommit(clamp(parsed));
        }
      }}
      onBlur={() => {
        focusedRef.current = false;
        const parsed = parseValue(draft);
        if (parsed === null) {
          const fallback = defaultValue !== undefined ? defaultValue : value;
          onCommit(clamp(fallback));
          setDraft(String(clamp(fallback)));
        } else {
          const clamped = clamp(parsed);
          onCommit(clamped);
          setDraft(String(clamped));
        }
      }}
    />
  );
}
