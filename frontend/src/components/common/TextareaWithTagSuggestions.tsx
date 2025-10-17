"use client";

import { useState, useRef, useEffect, KeyboardEvent, ChangeEvent, TextareaHTMLAttributes } from "react";
import Textarea from "./Textarea";
import TagSuggestions from "./TagSuggestions";
import {
  searchTags,
  getCurrentTag,
  replaceCurrentTag,
  deleteTagAtCursor,
} from "@/utils/tagSuggestions";

interface TagSuggestion {
  tag: string;
  count: number;
  category: string;
}

interface TextareaWithTagSuggestionsProps extends Omit<TextareaHTMLAttributes<HTMLTextAreaElement>, 'onChange'> {
  label?: string;
  value: string;
  onChange: (e: ChangeEvent<HTMLTextAreaElement>) => void;
  enableWeightControl?: boolean;
  rows?: number;
}

/**
 * TextareaWithTagSuggestions - Textarea with tag autocompletion
 * Features:
 * - Tag suggestions while typing
 * - Up/Down arrow keys to navigate suggestions
 * - Enter/Tab to accept suggestion
 * - Ctrl+Backspace to delete tag at cursor
 */
export default function TextareaWithTagSuggestions({
  label,
  value,
  onChange,
  enableWeightControl = false,
  rows = 4,
  ...props
}: TextareaWithTagSuggestionsProps) {
  const [suggestions, setSuggestions] = useState<TagSuggestion[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [suggestionsPosition, setSuggestionsPosition] = useState({ top: 0, left: 0 });
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const searchTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Get textarea ref from Textarea component
  useEffect(() => {
    if (textareaRef.current) {
      const textarea = textareaRef.current.querySelector("textarea");
      if (textarea) {
        textareaRef.current = textarea as any;
      }
    }
  }, []);

  // Search for tag suggestions
  const updateSuggestions = async (text: string, cursorPos: number) => {
    const currentTag = getCurrentTag(text, cursorPos);

    if (currentTag.length >= 2) {
      // Clear previous timeout
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }

      // Debounce search
      searchTimeoutRef.current = setTimeout(async () => {
        const results = await searchTags(currentTag, 20);
        setSuggestions(results);
        setSelectedIndex(results.length > 0 ? 0 : -1);

        // Calculate position for suggestions dropdown
        if (textareaRef.current) {
          const textarea = textareaRef.current as any;
          if (textarea.tagName === "TEXTAREA") {
            const rect = textarea.getBoundingClientRect();
            setSuggestionsPosition({
              top: rect.bottom + window.scrollY,
              left: rect.left + window.scrollX,
            });
          }
        }
      }, 150);
    } else {
      setSuggestions([]);
      setSelectedIndex(-1);
    }
  };

  // Handle text change
  const handleChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e);
    const cursorPos = e.target.selectionStart;
    updateSuggestions(e.target.value, cursorPos);
  };

  // Handle key down for navigation and shortcuts
  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Ctrl+Backspace: Delete tag at cursor
    if (e.ctrlKey && e.key === "Backspace") {
      e.preventDefault();
      const textarea = e.currentTarget;
      const cursorPos = textarea.selectionStart;
      const result = deleteTagAtCursor(value, cursorPos);

      // Create synthetic event for onChange
      const syntheticEvent = {
        ...e,
        target: { ...textarea, value: result.text },
        currentTarget: { ...textarea, value: result.text },
      } as ChangeEvent<HTMLTextAreaElement>;

      onChange(syntheticEvent);

      // Set cursor position after state update
      setTimeout(() => {
        textarea.selectionStart = result.cursorPos;
        textarea.selectionEnd = result.cursorPos;
      }, 0);

      setSuggestions([]);
      return;
    }

    // Handle suggestions navigation
    if (suggestions.length > 0) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIndex((prev) => (prev < suggestions.length - 1 ? prev + 1 : 0));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIndex((prev) => (prev > 0 ? prev - 1 : suggestions.length - 1));
      } else if (e.key === "Enter" || e.key === "Tab") {
        if (selectedIndex >= 0) {
          e.preventDefault();
          acceptSuggestion(suggestions[selectedIndex].tag);
        }
      } else if (e.key === "Escape") {
        setSuggestions([]);
        setSelectedIndex(-1);
      }
    }
  };

  // Accept a suggestion
  const acceptSuggestion = (tag: string) => {
    if (!textareaRef.current) return;

    const textarea = textareaRef.current as any;
    if (textarea.tagName !== "TEXTAREA") return;

    const cursorPos = textarea.selectionStart;
    const result = replaceCurrentTag(value, cursorPos, tag);

    // Create synthetic event for onChange
    const syntheticEvent = {
      target: { ...textarea, value: result.text },
      currentTarget: { ...textarea, value: result.text },
    } as ChangeEvent<HTMLTextAreaElement>;

    onChange(syntheticEvent);

    // Set cursor position and clear suggestions after state update
    setTimeout(() => {
      textarea.selectionStart = result.cursorPos;
      textarea.selectionEnd = result.cursorPos;
      textarea.focus();
    }, 0);

    setSuggestions([]);
    setSelectedIndex(-1);
  };

  return (
    <div className="relative" ref={textareaRef as any}>
      <Textarea
        label={label}
        value={value}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        enableWeightControl={enableWeightControl}
        rows={rows}
        {...props}
      />

      {suggestions.length > 0 && (
        <TagSuggestions
          suggestions={suggestions}
          selectedIndex={selectedIndex}
          onSelect={acceptSuggestion}
          position={suggestionsPosition}
        />
      )}
    </div>
  );
}
