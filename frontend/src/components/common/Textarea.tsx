import { TextareaHTMLAttributes, useEffect, useRef, KeyboardEvent } from "react";
import { cn } from "@/lib/utils";

interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  enableWeightControl?: boolean;
  resizeStorageKey?: string;
}

export const GENERATION_PROMPT_HEIGHT_KEY = "generation_prompt_height";
export const GENERATION_NEGATIVE_PROMPT_HEIGHT_KEY = "generation_negative_prompt_height";
export const GENERATION_LYRICS_HEIGHT_KEY = "generation_lyrics_height";

/**
 * Find the extent of an existing emphasis syntax at cursor position
 * Returns null if cursor is not inside emphasis syntax
 */
function findEmphasisAtCursor(
  text: string,
  cursorPos: number
): { start: number; end: number; innerText: string; weight: number } | null {
  // Search backwards for opening parenthesis
  let openParen = -1;
  let depth = 0;

  for (let i = cursorPos - 1; i >= 0; i--) {
    if (text[i] === ')') {
      depth++;
    } else if (text[i] === '(') {
      if (depth === 0) {
        openParen = i;
        break;
      }
      depth--;
    }
  }

  if (openParen === -1) return null;

  // Search forwards for closing parenthesis and weight
  let closeParen = -1;
  depth = 0;

  for (let i = cursorPos; i < text.length; i++) {
    if (text[i] === '(') {
      depth++;
    } else if (text[i] === ')') {
      if (depth === 0) {
        closeParen = i;
        break;
      }
      depth--;
    }
  }

  if (closeParen === -1) return null;

  // Check if this matches emphasis syntax (allow negative weights for NegPip)
  const emphasisText = text.substring(openParen, closeParen + 1);
  const weightMatch = emphasisText.match(/^\((.*?)(?::(-?[0-9.]+))?\)$/);

  if (weightMatch) {
    return {
      start: openParen,
      end: closeParen + 1,
      innerText: weightMatch[1],
      weight: weightMatch[2] ? parseFloat(weightMatch[2]) : 1.1,
    };
  }

  return null;
}

/**
 * Adjust prompt weight using Ctrl+Up/Down
 * Supports A1111-style emphasis: (text:1.2)
 */
function adjustPromptWeight(
  text: string,
  selectionStart: number,
  selectionEnd: number,
  increment: number
): { newText: string; newStart: number; newEnd: number } {
  let start = selectionStart;
  let end = selectionEnd;
  let selectedText: string;

  // If nothing is selected, check if cursor is inside existing emphasis
  if (selectionStart === selectionEnd) {
    const existingEmphasis = findEmphasisAtCursor(text, selectionStart);

    if (existingEmphasis) {
      // Cursor is inside emphasis syntax - adjust the existing emphasis
      start = existingEmphasis.start;
      end = existingEmphasis.end;
      selectedText = text.substring(start, end);
    } else {
      // Select the current tag (between commas)
      start = selectionStart;
      end = selectionEnd;

      // Find start of tag (search backwards for comma or start of string)
      while (start > 0 && text[start - 1] !== ',') {
        start--;
      }
      // Skip leading whitespace
      while (start < text.length && (text[start] === ' ' || text[start] === '\n')) {
        start++;
      }

      // Find end of tag (search forwards for comma or end of string)
      while (end < text.length && text[end] !== ',') {
        end++;
      }
      // Skip trailing whitespace
      while (end > start && (text[end - 1] === ' ' || text[end - 1] === '\n')) {
        end--;
      }

      selectedText = text.substring(start, end);
    }
  } else {
    // User has made a selection - use it faithfully
    selectedText = text.substring(start, end);
  }

  // Check if selected text already has weight syntax (allow negative weights for NegPip)
  const weightMatch = selectedText.match(/^\((.*?)(?::(-?[0-9.]+))?\)$/);

  // Weight range allows negatives: a negative weight triggers NegPip (the token's
  // attention value is negated, subtracting the concept) instead of plain emphasis.
  const WEIGHT_MIN = -2.0;
  const WEIGHT_MAX = 2.0;

  let newText: string;
  let innerText: string;
  let currentWeight: number;

  if (weightMatch) {
    // Already has weight syntax - adjust it
    innerText = weightMatch[1];
    currentWeight = weightMatch[2] ? parseFloat(weightMatch[2]) : 1.1;
    const newWeight = Math.max(WEIGHT_MIN, Math.min(WEIGHT_MAX, currentWeight + increment));

    if (Math.abs(newWeight - 1.0) < 0.01) {
      // Close to 1.0, remove emphasis
      newText = innerText;
    } else {
      newText = `(${innerText}:${newWeight.toFixed(2)})`;
    }
  } else {
    // No weight syntax yet - add it
    innerText = selectedText;
    const newWeight = Math.max(WEIGHT_MIN, Math.min(WEIGHT_MAX, 1.0 + increment));

    if (Math.abs(newWeight - 1.0) < 0.01) {
      // No change needed
      newText = innerText;
    } else {
      newText = `(${innerText}:${newWeight.toFixed(2)})`;
    }
  }

  const before = text.substring(0, start);
  const after = text.substring(end);
  const fullText = before + newText + after;

  return {
    newText: fullText,
    newStart: start,
    newEnd: start + newText.length,
  };
}

export default function Textarea({
  label,
  className,
  enableWeightControl = false,
  resizeStorageKey,
  onKeyDown,
  onChange,
  value,
  ...props
}: TextareaProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea || !resizeStorageKey) return;

    const savedHeight = Number(localStorage.getItem(resizeStorageKey));
    if (Number.isFinite(savedHeight) && savedHeight >= 48 && savedHeight <= 640) {
      textarea.style.height = `${savedHeight}px`;
    }

    let observer: ResizeObserver | null = null;
    const frame = requestAnimationFrame(() => {
      observer = new ResizeObserver(() => {
        const height = Math.round(textarea.getBoundingClientRect().height);
        if (height >= 48 && height <= 640) {
          localStorage.setItem(resizeStorageKey, String(height));
        }
      });
      observer.observe(textarea);
    });

    return () => {
      cancelAnimationFrame(frame);
      observer?.disconnect();
    };
  }, [resizeStorageKey]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Call original onKeyDown if provided
    if (onKeyDown) {
      onKeyDown(e);
    }

    // Ctrl+Home: Move cursor to start
    if (e.ctrlKey && e.key === "Home") {
      e.preventDefault();
      const textarea = textareaRef.current;
      if (textarea) {
        textarea.selectionStart = 0;
        textarea.selectionEnd = 0;
      }
      return;
    }

    // Ctrl+End: Move cursor to end
    if (e.ctrlKey && e.key === "End") {
      e.preventDefault();
      const textarea = textareaRef.current;
      if (textarea) {
        const endPos = textarea.value.length;
        textarea.selectionStart = endPos;
        textarea.selectionEnd = endPos;
      }
      return;
    }

    // Ctrl+Up/Down for weight adjustment
    if (enableWeightControl && e.ctrlKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
      e.preventDefault();

      const textarea = textareaRef.current;
      if (!textarea) return;

      const increment = e.key === 'ArrowUp' ? 0.05 : -0.05;
      const result = adjustPromptWeight(
        textarea.value,
        textarea.selectionStart,
        textarea.selectionEnd,
        increment
      );

      // Call onChange directly with a synthetic event
      if (onChange) {
        const syntheticEvent = {
          target: {
            value: result.newText,
          },
          currentTarget: {
            value: result.newText,
          },
        } as React.ChangeEvent<HTMLTextAreaElement>;

        onChange(syntheticEvent);
      }

      // Use setTimeout to ensure React has updated before restoring selection
      setTimeout(() => {
        if (textarea) {
          textarea.setSelectionRange(result.newStart, result.newEnd);
        }
      }, 0);
    }
  };

  return (
    <div className="space-y-1">
      {label && (
        <label className="block text-xs font-medium text-gray-400">
          {label}
          {enableWeightControl && (
            <span className="ml-2 text-xs text-gray-500 hidden lg:inline">
              (Ctrl+↑/↓ to adjust weight)
            </span>
          )}
        </label>
      )}
      <textarea
        ref={textareaRef}
        className={cn(
          "w-full rounded-md border border-gray-700 bg-gray-800 px-2.5 py-2 text-sm text-gray-100 placeholder-gray-500 focus:border-violet-500 focus:outline-none focus:ring-1 focus:ring-violet-500 resize-vertical disabled:cursor-not-allowed disabled:bg-gray-900 disabled:text-gray-500 disabled:opacity-70",
          className
        )}
        onKeyDown={handleKeyDown}
        onChange={onChange}
        value={value}
        {...props}
      />
    </div>
  );
}
