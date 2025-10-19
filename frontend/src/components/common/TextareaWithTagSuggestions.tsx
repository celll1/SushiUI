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

  // Calculate cursor position in pixels using a hidden div mirror
  const getCursorCoordinates = (textarea: HTMLTextAreaElement, position: number) => {
    const style = window.getComputedStyle(textarea);
    const rect = textarea.getBoundingClientRect();

    // Create a mirror div to measure text position with wrapping
    const mirror = document.createElement('div');
    const mirrorStyle = mirror.style;

    // Copy all relevant styles from textarea to mirror
    const properties = [
      'boxSizing', 'width', 'fontFamily', 'fontSize', 'fontWeight',
      'fontStyle', 'letterSpacing', 'textTransform', 'wordSpacing',
      'textIndent', 'whiteSpace', 'lineHeight', 'padding', 'border',
    ];

    properties.forEach(prop => {
      mirrorStyle.setProperty(prop, style.getPropertyValue(prop));
    });

    // Set mirror-specific styles
    mirrorStyle.position = 'absolute';
    mirrorStyle.visibility = 'hidden';
    mirrorStyle.whiteSpace = 'pre-wrap'; // Important: match textarea wrapping
    mirrorStyle.wordWrap = 'break-word';
    mirrorStyle.overflowWrap = 'break-word';
    mirrorStyle.top = '0';
    mirrorStyle.left = '0';

    document.body.appendChild(mirror);

    // Set text content up to cursor position
    const textBeforeCursor = textarea.value.substring(0, position);
    mirror.textContent = textBeforeCursor;

    // Create a span at the cursor position to measure
    const cursorSpan = document.createElement('span');
    cursorSpan.textContent = '|'; // Placeholder character
    mirror.appendChild(cursorSpan);

    // Get the position of the cursor span
    const cursorRect = cursorSpan.getBoundingClientRect();
    const mirrorRect = mirror.getBoundingClientRect();

    // Calculate relative position within mirror
    const relativeTop = cursorRect.top - mirrorRect.top;
    const relativeLeft = cursorRect.left - mirrorRect.left;

    // Clean up
    document.body.removeChild(mirror);

    // Calculate absolute position (account for scroll and padding)
    const paddingTop = parseFloat(style.paddingTop) || 0;
    const paddingLeft = parseFloat(style.paddingLeft) || 0;

    return {
      top: rect.top + paddingTop + relativeTop - textarea.scrollTop,
      left: rect.left + paddingLeft + relativeLeft,
    };
  };

  // Search for tag suggestions
  const updateSuggestions = async (text: string, cursorPos: number) => {
    // Clear previous timeout immediately
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
      searchTimeoutRef.current = null;
    }

    const currentTag = getCurrentTag(text, cursorPos);
    console.log('[TagSuggestions] Current tag:', currentTag, 'at position:', cursorPos);

    // If no tag or tag too short, clear suggestions immediately
    if (currentTag.length < 2) {
      setSuggestions([]);
      setSelectedIndex(-1);
      return;
    }

    // Debounce search with 300ms delay
    searchTimeoutRef.current = setTimeout(async () => {
      console.log('[TagSuggestions] Searching for:', currentTag);
      const results = await searchTags(currentTag, 20);
      console.log('[TagSuggestions] Found results:', results.length);

      // Check if the tag is still valid (user might have continued typing)
      const textarea = textareaRef.current as any;
      if (textarea && textarea.tagName === "TEXTAREA") {
        const latestTag = getCurrentTag(textarea.value, textarea.selectionStart);
        // Only show results if the tag hasn't changed
        if (latestTag === currentTag) {
          setSuggestions(results);
          setSelectedIndex(results.length > 0 ? 0 : -1);

          // Calculate position for suggestions dropdown near cursor
          const coords = getCursorCoordinates(textarea, cursorPos);
          setSuggestionsPosition({
            top: coords.top,
            left: coords.left,
          });
        }
      }
    }, 300); // Increased delay to 300ms
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

  // Handle blur - clear suggestions when focus is lost
  const handleBlur = () => {
    // Use setTimeout to allow click on suggestion to register before clearing
    setTimeout(() => {
      setSuggestions([]);
      setSelectedIndex(-1);
    }, 200);
  };

  return (
    <div className="relative" ref={textareaRef as any}>
      <Textarea
        label={label}
        value={value}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        onBlur={handleBlur}
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
