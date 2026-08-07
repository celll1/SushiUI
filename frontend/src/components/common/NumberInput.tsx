"use client";

import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

interface NumberInputProps {
  id?: string;
  /**
   * ACCESSIBLE NAME ONLY — this is passed to `aria-label` and NOTHING IS
   * DRAWN. A caller that wants a caption on screen must render its own
   * `<label>` next to the field (that is what the generation panels do, and
   * what `common/Slider` does for its own numeric box).
   *
   * This component deliberately does not render one: it is a bare `<input>` by
   * design, and several callers already draw their own label around it
   * (`PostEditControls` puts the box inside a slider row with a fixed `w-14`
   * width; `OutpaintPanel`'s FBCache fields wrap it in a `<label>` with the
   * text inline). Rendering a label here would duplicate that text and change
   * the layout of every one of those rows, so the caption stays the caller's
   * job.
   */
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
  /**
   * Optional commit-time quantization to the nearest multiple of `snap`
   * (e.g. `snap={64}` forces committed values onto 0, 64, 128, ...).
   *
   * This is independent of `step`: `step` is only the spinner's
   * increment/decrement amount (and a UA hint for native validation styling)
   * — it never rewrites typed/pasted input. `snap` is the actual backend
   * constraint and is applied only at commit time (live-commit-while-typing
   * and blur-resolve), never to the draft string while the user is mid-type.
   *
   * Ordering: snap is applied BEFORE min/max clamping, then the result is
   * re-clamped. Snapping first and clamping second (rather than the
   * reverse) is intentional: clamping first and then snapping can push the
   * value back out of [min, max] (e.g. min=10, snap=64 clamped-then-snapped
   * rounds 10 down to 0, violating min); snapping first and clamping after
   * guarantees the final value both respects [min, max] and is as close to
   * a multiple of `snap` as the range allows.
   */
  snap?: number;
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
 *
 * `parse="int"` rounds rather than truncates: the raw text is always parsed
 * as a float first (so "1.7" is read as 1.7, not truncated to 1 by
 * `parseInt`), then `Math.round`ed to the nearest integer at commit time.
 * This matches user expectation ("1.7" -> 2) instead of JS's default
 * truncate-toward-zero behavior.
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
  snap,
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

  // Always parse as float first, regardless of `parse` mode — this lets
  // "int" mode round ("1.7" -> 2) rather than truncate ("1.7" -> 1, which is
  // what parseInt would silently do).
  const parseValue = (text: string): number | null => {
    if (text.trim() === "") return null;
    const parsed = parseFloat(text);
    if (isNaN(parsed)) return null;
    return parse === "int" ? Math.round(parsed) : parsed;
  };

  const clamp = (num: number): number => {
    let clamped = num;
    if (min !== undefined && clamped < min) clamped = min;
    if (max !== undefined && clamped > max) clamped = max;
    return clamped;
  };

  // Round to the nearest multiple of `snap`, cleaning up float error via
  // decimal-count-aware toFixed (e.g. snap=0.1 must not leave 0.30000000004).
  const snapValue = (num: number): number => {
    if (snap === undefined || snap <= 0) return num;
    const snapped = Math.round(num / snap) * snap;
    const snapDecimals = (snap.toString().split(".")[1] || "").length;
    return snapDecimals > 0 ? parseFloat(snapped.toFixed(snapDecimals)) : snapped;
  };

  // Commit-time normalization: snap first, then clamp, so the final value is
  // guaranteed inside [min, max] even if snapping alone would have pushed it
  // outside that range (see the `snap` prop doc for why this order matters).
  const normalize = (num: number): number => clamp(snapValue(num));

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
          onCommit(normalize(parsed));
        }
      }}
      onBlur={() => {
        focusedRef.current = false;
        const parsed = parseValue(draft);
        if (parsed === null) {
          const fallback = defaultValue !== undefined ? defaultValue : value;
          const resolved = normalize(fallback);
          onCommit(resolved);
          setDraft(String(resolved));
        } else {
          const normalized = normalize(parsed);
          onCommit(normalized);
          setDraft(String(normalized));
        }
      }}
    />
  );
}
