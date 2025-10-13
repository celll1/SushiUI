import { TextareaHTMLAttributes, useRef, KeyboardEvent } from "react";
import { cn } from "@/lib/utils";

interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  enableWeightControl?: boolean;
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
  // If nothing is selected, select the current word/phrase
  if (selectionStart === selectionEnd) {
    // Find word boundaries
    let start = selectionStart;
    let end = selectionEnd;

    while (start > 0 && text[start - 1] !== ' ' && text[start - 1] !== ',' && text[start - 1] !== '\n') {
      start--;
    }
    while (end < text.length && text[end] !== ' ' && text[end] !== ',' && text[end] !== '\n') {
      end++;
    }

    selectionStart = start;
    selectionEnd = end;
  }

  const selectedText = text.substring(selectionStart, selectionEnd);

  // Check if already has weight syntax
  const weightMatch = selectedText.match(/^\((.*?)(?::([0-9.]+))?\)$/);

  let newText: string;
  let innerText: string;
  let currentWeight: number;

  if (weightMatch) {
    // Already has weight syntax
    innerText = weightMatch[1];
    currentWeight = weightMatch[2] ? parseFloat(weightMatch[2]) : 1.1;
    const newWeight = Math.max(0.1, Math.min(2.0, currentWeight + increment));

    if (Math.abs(newWeight - 1.0) < 0.01) {
      // Close to 1.0, remove emphasis
      newText = innerText;
    } else {
      newText = `(${innerText}:${newWeight.toFixed(2)})`;
    }
  } else {
    // No weight syntax yet
    innerText = selectedText;
    const newWeight = Math.max(0.1, Math.min(2.0, 1.0 + increment));

    if (Math.abs(newWeight - 1.0) < 0.01) {
      // No change needed
      newText = innerText;
    } else {
      newText = `(${innerText}:${newWeight.toFixed(2)})`;
    }
  }

  const before = text.substring(0, selectionStart);
  const after = text.substring(selectionEnd);
  const fullText = before + newText + after;

  return {
    newText: fullText,
    newStart: selectionStart,
    newEnd: selectionStart + newText.length,
  };
}

export default function Textarea({
  label,
  className,
  enableWeightControl = false,
  onKeyDown,
  onChange,
  value,
  ...props
}: TextareaProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Call original onKeyDown if provided
    if (onKeyDown) {
      onKeyDown(e);
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
        <label className="block text-sm font-medium text-gray-300">
          {label}
          {enableWeightControl && (
            <span className="ml-2 text-xs text-gray-500">
              (Ctrl+↑/↓ to adjust weight)
            </span>
          )}
        </label>
      )}
      <textarea
        ref={textareaRef}
        className={cn(
          "w-full rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-gray-100 placeholder-gray-500 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 resize-vertical",
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
