"use client";

import { useState, useRef, useEffect, KeyboardEvent, ChangeEvent, TextareaHTMLAttributes, forwardRef, useImperativeHandle } from "react";
import Textarea from "./Textarea";
import TagSuggestions from "./TagSuggestions";
import {
  searchTags,
  getCurrentTag,
  replaceCurrentTag,
  deleteTagAtCursor,
  swapTagWithAdjacent,
  jumpToNextDelimiter,
  jumpToPreviousDelimiter,
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
const TextareaWithTagSuggestions = forwardRef<HTMLTextAreaElement, TextareaWithTagSuggestionsProps>(({
  label,
  value,
  onChange,
  enableWeightControl = false,
  rows = 4,
  onKeyDown: externalOnKeyDown,
  ...props
}, forwardedRef) => {
  const [suggestions, setSuggestions] = useState<TagSuggestion[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [suggestionsPosition, setSuggestionsPosition] = useState({ top: 0, left: 0 });
  const internalTextareaRef = useRef<HTMLTextAreaElement>(null);
  const searchTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Undo/Redo history management (max 50 entries)
  const [history, setHistory] = useState<Array<{ text: string; cursorPos: number }>>([]);
  const [historyIndex, setHistoryIndex] = useState(0);
  const isUndoRedoRef = useRef(false); // Flag to prevent adding to history during undo/redo
  const lastValueRef = useRef<string>(""); // Track last value to detect external changes
  const isTagOperationRef = useRef(false); // Flag to prevent double history during tag operations

  // Expose the internal textarea ref to parent components
  useImperativeHandle(forwardedRef, () => internalTextareaRef.current as HTMLTextAreaElement);

  // Get textarea ref from Textarea component
  useEffect(() => {
    if (containerRef.current) {
      const textarea = containerRef.current.querySelector("textarea");
      if (textarea) {
        internalTextareaRef.current = textarea as HTMLTextAreaElement;
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
      const textarea = internalTextareaRef.current;
      if (textarea && textarea.tagName === "TEXTAREA") {
        const latestTag = getCurrentTag(textarea.value, textarea.selectionStart);
        // Only show results if the tag hasn't changed
        if (latestTag === currentTag) {
          // Don't show suggestions if the only result is an exact match
          const currentTagLower = currentTag.toLowerCase().replace(/_/g, ' ');
          const hasOnlyExactMatch = results.length === 1 &&
            results[0].tag.toLowerCase().replace(/_/g, ' ') === currentTagLower;

          if (hasOnlyExactMatch) {
            console.log('[TagSuggestions] Only exact match found, hiding suggestions');
            setSuggestions([]);
            setSelectedIndex(-1);
          } else {
            console.log('[TagSuggestions] Showing suggestions:', results.length);
            setSuggestions(results);
            setSelectedIndex(results.length > 0 ? 0 : -1);

            // Calculate position for suggestions dropdown near cursor
            if (results.length > 0) {
              const coords = getCursorCoordinates(textarea, cursorPos);
              setSuggestionsPosition({
                top: coords.top,
                left: coords.left,
              });
            }
          }
        }
      }
    }, 300); // Increased delay to 300ms
  };

  // Track external changes to value (e.g., from "send to", initial load, etc.)
  useEffect(() => {
    // Check if value changed externally (not from undo/redo or tag operations)
    if (value !== lastValueRef.current && !isUndoRedoRef.current && !isTagOperationRef.current) {
      const textarea = internalTextareaRef.current;
      const cursorPos = textarea?.selectionStart || 0;

      // If history is empty, initialize it
      if (history.length === 0) {
        setHistory([{ text: value, cursorPos }]);
        setHistoryIndex(0);
        lastValueRef.current = value;
      } else {
        // Value changed externally, add as new history entry
        // This handles "send to" and other external updates
        const newHistory = history.slice(0, historyIndex + 1);
        newHistory.push({ text: value, cursorPos });

        if (newHistory.length > 50) {
          newHistory.shift();
          setHistoryIndex(49);
        } else {
          setHistoryIndex(newHistory.length - 1);
        }

        setHistory(newHistory);
        lastValueRef.current = value;
      }
    }
  }, [value, history, historyIndex]);

  // Add new state to history after an operation
  const addToHistory = (text: string, cursorPos: number) => {
    if (isUndoRedoRef.current) {
      // Don't add to history if this is from undo/redo
      return;
    }

    lastValueRef.current = text; // Update last value tracker

    setHistory((prevHistory) => {
      // Remove any entries after current index (when user makes new change after undo)
      const newHistory = prevHistory.slice(0, historyIndex + 1);
      // Add new entry
      newHistory.push({ text, cursorPos });
      // Limit to 50 entries
      if (newHistory.length > 50) {
        newHistory.shift();
        // Don't increment index if we removed from start
        setHistoryIndex(49);
        return newHistory;
      }
      setHistoryIndex(newHistory.length - 1);
      return newHistory;
    });
  };

  // Undo operation (Ctrl+Z)
  const handleUndo = () => {
    if (historyIndex <= 0) return; // Nothing to undo

    const prevState = history[historyIndex - 1];
    isUndoRedoRef.current = true;
    lastValueRef.current = prevState.text; // Update last value tracker

    const textarea = internalTextareaRef.current;
    if (textarea && textarea.tagName === "TEXTAREA") {
      // Create synthetic event for onChange
      const syntheticEvent = {
        target: { ...textarea, value: prevState.text },
        currentTarget: { ...textarea, value: prevState.text },
      } as ChangeEvent<HTMLTextAreaElement>;

      onChange(syntheticEvent);

      // Set cursor position
      setTimeout(() => {
        textarea.selectionStart = prevState.cursorPos;
        textarea.selectionEnd = prevState.cursorPos;
        isUndoRedoRef.current = false;
      }, 0);
    }

    setHistoryIndex((prev) => prev - 1);
  };

  // Redo operation (Ctrl+Shift+Z)
  const handleRedo = () => {
    if (historyIndex >= history.length - 1) return; // Nothing to redo

    const nextState = history[historyIndex + 1];
    isUndoRedoRef.current = true;
    lastValueRef.current = nextState.text; // Update last value tracker

    const textarea = internalTextareaRef.current;
    if (textarea && textarea.tagName === "TEXTAREA") {
      // Create synthetic event for onChange
      const syntheticEvent = {
        target: { ...textarea, value: nextState.text },
        currentTarget: { ...textarea, value: nextState.text },
      } as ChangeEvent<HTMLTextAreaElement>;

      onChange(syntheticEvent);

      // Set cursor position
      setTimeout(() => {
        textarea.selectionStart = nextState.cursorPos;
        textarea.selectionEnd = nextState.cursorPos;
        isUndoRedoRef.current = false;
      }, 0);
    }

    setHistoryIndex((prev) => prev + 1);
  };

  // Handle text change
  const handleChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e);
    const cursorPos = e.target.selectionStart;
    updateSuggestions(e.target.value, cursorPos);
  };

  // Handle key down for navigation and shortcuts
  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Call external onKeyDown if provided
    if (externalOnKeyDown) {
      externalOnKeyDown(e);
      // If external handler prevented default or stopped propagation, respect that
      if (e.defaultPrevented) {
        return;
      }
    }
    // Ctrl+Z: Undo
    if (e.ctrlKey && !e.shiftKey && e.key === "z") {
      e.preventDefault();
      handleUndo();
      return;
    }

    // Ctrl+Shift+Z: Redo
    if (e.ctrlKey && e.shiftKey && e.key === "Z") {
      e.preventDefault();
      handleRedo();
      return;
    }

    // Ctrl+Shift+Left: Swap tag with previous tag
    if (e.ctrlKey && e.shiftKey && e.key === "ArrowLeft") {
      e.preventDefault();
      const textarea = e.currentTarget;
      const cursorPos = textarea.selectionStart;

      const result = swapTagWithAdjacent(value, cursorPos, 'left');

      if (result) {
        isTagOperationRef.current = true;
        lastValueRef.current = result.text;

        // Create synthetic event for onChange
        const syntheticEvent = {
          target: { ...textarea, value: result.text },
          currentTarget: { ...textarea, value: result.text },
        } as ChangeEvent<HTMLTextAreaElement>;

        onChange(syntheticEvent);

        // Add new state to history AFTER swap
        setTimeout(() => {
          textarea.selectionStart = result.cursorPos;
          textarea.selectionEnd = result.cursorPos;
          addToHistory(result.text, result.cursorPos);
          isTagOperationRef.current = false;
        }, 0);
      }

      return;
    }

    // Ctrl+Shift+Right: Swap tag with next tag
    if (e.ctrlKey && e.shiftKey && e.key === "ArrowRight") {
      e.preventDefault();
      const textarea = e.currentTarget;
      const cursorPos = textarea.selectionStart;

      const result = swapTagWithAdjacent(value, cursorPos, 'right');

      if (result) {
        isTagOperationRef.current = true;
        lastValueRef.current = result.text;

        // Create synthetic event for onChange
        const syntheticEvent = {
          target: { ...textarea, value: result.text },
          currentTarget: { ...textarea, value: result.text },
        } as ChangeEvent<HTMLTextAreaElement>;

        onChange(syntheticEvent);

        // Add new state to history AFTER swap
        setTimeout(() => {
          textarea.selectionStart = result.cursorPos;
          textarea.selectionEnd = result.cursorPos;
          addToHistory(result.text, result.cursorPos);
          isTagOperationRef.current = false;
        }, 0);
      }

      return;
    }

    // Ctrl+Left: Jump to previous delimiter
    if (e.ctrlKey && !e.shiftKey && e.key === "ArrowLeft") {
      e.preventDefault();
      const textarea = e.currentTarget;
      const cursorPos = textarea.selectionStart;

      const newPos = jumpToPreviousDelimiter(value, cursorPos);
      textarea.selectionStart = newPos;
      textarea.selectionEnd = newPos;

      return;
    }

    // Ctrl+Right: Jump to next delimiter
    if (e.ctrlKey && !e.shiftKey && e.key === "ArrowRight") {
      e.preventDefault();
      const textarea = e.currentTarget;
      const cursorPos = textarea.selectionStart;

      const newPos = jumpToNextDelimiter(value, cursorPos);
      textarea.selectionStart = newPos;
      textarea.selectionEnd = newPos;

      return;
    }

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

      // Add new state to history AFTER deletion
      setTimeout(() => {
        textarea.selectionStart = result.cursorPos;
        textarea.selectionEnd = result.cursorPos;
        addToHistory(result.text, result.cursorPos);
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
    if (!internalTextareaRef.current) return;

    const textarea = internalTextareaRef.current;
    if (textarea.tagName !== "TEXTAREA") return;

    const cursorPos = textarea.selectionStart;
    const result = replaceCurrentTag(value, cursorPos, tag);

    // Create synthetic event for onChange
    const syntheticEvent = {
      target: { ...textarea, value: result.text },
      currentTarget: { ...textarea, value: result.text },
    } as ChangeEvent<HTMLTextAreaElement>;

    onChange(syntheticEvent);

    // Add new state to history AFTER accepting suggestion
    setTimeout(() => {
      textarea.selectionStart = result.cursorPos;
      textarea.selectionEnd = result.cursorPos;
      textarea.focus();
      addToHistory(result.text, result.cursorPos);
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
    <div className="relative" ref={containerRef}>
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
});

TextareaWithTagSuggestions.displayName = 'TextareaWithTagSuggestions';

export default TextareaWithTagSuggestions;
