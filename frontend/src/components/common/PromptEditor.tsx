"use client";

import { useState, useRef, ChangeEvent, useEffect } from "react";
import { Menu, X } from "lucide-react";
import Button from "./Button";
import TextareaWithTagSuggestions from "./TextareaWithTagSuggestions";
import TemplatePanel from "./TemplatePanel";
import CategoryOrderPanel from "./CategoryOrderPanel";
import WildcardPanel from "./WildcardPanel";
import TIPOPanel, { TIPOPanelRef } from "./TIPOPanel";
import ImageTaggerPanel from "./ImageTaggerPanel";
import { TIPOSettings } from "./TIPODialog";

interface PromptEditorProps {
  initialPrompt: string;
  initialNegativePrompt: string;
  onSave: (prompt: string, negativePrompt: string) => void;
  onClose: () => void;
}

type PanelType = "main" | "template" | "category" | "wildcard" | "tipo" | "tagger";

export default function PromptEditor({
  initialPrompt,
  initialNegativePrompt,
  onSave,
  onClose,
}: PromptEditorProps) {
  const [prompt, setPrompt] = useState(initialPrompt);
  const [negativePrompt, setNegativePrompt] = useState(initialNegativePrompt);
  const [activePanel, setActivePanel] = useState<PanelType>("main");
  const [activePromptType, setActivePromptType] = useState<"positive" | "negative">("positive");
  const [cursorPosition, setCursorPosition] = useState<number>(0);
  const [isPanelSidebarOpen, setIsPanelSidebarOpen] = useState(false);

  // TIPO settings - load from localStorage or use defaults
  const [tipoSettings] = useState<TIPOSettings>(() => {
    const saved = localStorage.getItem("tipo_settings");
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
        console.error("Failed to parse TIPO settings from localStorage", e);
      }
    }
    return {
      model_name: "KBlueLeaf/TIPO-500M",
      tag_length: "short",
      nl_length: "short",
      temperature: 0.5,
      top_p: 0.9,
      top_k: 40,
      max_new_tokens: 256,
      categories: [],
    };
  });

  const promptTextareaRef = useRef<HTMLTextAreaElement>(null);
  const negativePromptTextareaRef = useRef<HTMLTextAreaElement>(null);
  const editorContainerRef = useRef<HTMLDivElement>(null);
  const tipoPanelRef = useRef<TIPOPanelRef>(null);

  const handleSave = () => {
    onSave(prompt, negativePrompt);
    onClose();
  };

  const handlePromptChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setPrompt(e.target.value);
  };

  const handleNegativePromptChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setNegativePrompt(e.target.value);
  };

  // Track cursor position
  const updateCursorPosition = () => {
    const textarea = activePromptType === "positive"
      ? promptTextareaRef.current
      : negativePromptTextareaRef.current;

    if (textarea) {
      setCursorPosition(textarea.selectionStart);
    }
  };

  // Global keyboard shortcuts
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      // Only handle when no input/textarea (except our prompts) is focused
      const activeElement = document.activeElement;
      const isInputFocused = activeElement instanceof HTMLInputElement ||
        (activeElement instanceof HTMLTextAreaElement &&
         activeElement !== promptTextareaRef.current &&
         activeElement !== negativePromptTextareaRef.current);

      if (isInputFocused) return;

      // TIPO Panel shortcuts (work globally when TIPO panel is active)
      if (activePanel === "tipo" && tipoPanelRef.current) {
        console.log('[PromptEditor] In TIPO panel, ref exists:', !!tipoPanelRef.current);
        // Ctrl+Enter: Insert TIPO result
        if (e.ctrlKey && e.key === "Enter" && !e.shiftKey) {
          const hasResult = tipoPanelRef.current?.hasResult();
          console.log('[PromptEditor] Ctrl+Enter pressed, hasResult:', hasResult);
          if (hasResult) {
            e.preventDefault();
            console.log('[PromptEditor] Calling insertResult');
            tipoPanelRef.current.insertResult();
            return;
          }
        }
        // Ctrl+Shift+Enter: Overwrite with TIPO result
        if (e.ctrlKey && e.shiftKey && e.key === "Enter") {
          const hasResult = tipoPanelRef.current?.hasResult();
          console.log('[PromptEditor] Ctrl+Shift+Enter pressed, hasResult:', hasResult);
          if (hasResult) {
            e.preventDefault();
            console.log('[PromptEditor] Calling overwriteResult');
            tipoPanelRef.current.overwriteResult();
            return;
          }
        }
        // Ctrl+M: Copy from main to TIPO input
        if (e.ctrlKey && e.key === "m") {
          e.preventDefault();
          tipoPanelRef.current.copyFromMain();
          return;
        }
      }

      // Ctrl+key shortcuts - focus the active prompt textarea
      if (e.ctrlKey && !e.shiftKey && !e.altKey) {
        const textarea = activePromptType === "positive"
          ? promptTextareaRef.current
          : negativePromptTextareaRef.current;

        if (textarea) {
          textarea.focus();
          // Let the textarea's own handler deal with the key
        }
      }
    };

    const container = editorContainerRef.current;
    if (container) {
      container.addEventListener('keydown', handleGlobalKeyDown);
      return () => container.removeEventListener('keydown', handleGlobalKeyDown);
    }
  }, [activePromptType, activePanel]);

  const handleInsertTemplate = (content: string) => {
    const textarea = activePromptType === "positive"
      ? promptTextareaRef.current
      : negativePromptTextareaRef.current;

    if (!textarea) return;

    const currentValue = activePromptType === "positive" ? prompt : negativePrompt;
    const cursorPos = cursorPosition || textarea.selectionStart || 0;

    // Insert template at cursor position with proper formatting
    const before = currentValue.substring(0, cursorPos);
    const after = currentValue.substring(cursorPos);

    const trimmedBefore = before.trimEnd();
    const trimmedAfter = after.trimStart();

    let prefix = "";
    let suffix = "";

    // Add delimiter before template if needed
    if (trimmedBefore.length > 0 && !trimmedBefore.endsWith(",") && !trimmedBefore.endsWith("\n")) {
      prefix = trimmedBefore + ", ";
    } else if (trimmedBefore.length > 0 && trimmedBefore.endsWith(",")) {
      prefix = trimmedBefore + " ";
    } else {
      prefix = trimmedBefore;
    }

    // Add delimiter after template if needed
    if (trimmedAfter.length > 0 && !trimmedAfter.startsWith(",") && !trimmedAfter.startsWith("\n")) {
      suffix = ", " + trimmedAfter;
    } else {
      suffix = trimmedAfter;
    }

    const newValue = prefix + content + suffix;
    const newCursorPos = prefix.length + content.length;

    if (activePromptType === "positive") {
      setPrompt(newValue);
    } else {
      setNegativePrompt(newValue);
    }

    // Set cursor position after insertion
    setTimeout(() => {
      if (textarea) {
        textarea.selectionStart = newCursorPos;
        textarea.selectionEnd = newCursorPos;
        setCursorPosition(newCursorPos);
        textarea.focus();
      }
    }, 0);
  };

  const handleOverwriteTemplate = (content: string) => {
    if (activePromptType === "positive") {
      setPrompt(content);
    } else {
      setNegativePrompt(content);
    }

    // Focus textarea
    const textarea = activePromptType === "positive"
      ? promptTextareaRef.current
      : negativePromptTextareaRef.current;

    setTimeout(() => {
      if (textarea) {
        textarea.focus();
        textarea.selectionStart = content.length;
        textarea.selectionEnd = content.length;
      }
    }, 0);
  };

  return (
    <>
      <style jsx global>{`
        /* Keep caret visible even when textarea is not focused */
        .prompt-editor-textarea textarea {
          caret-color: #60a5fa;
        }

        /* Highlight the active prompt editor with a subtle glow */
        .prompt-editor-textarea textarea:not(:focus) {
          box-shadow: inset 0 0 0 1px rgba(96, 165, 250, 0.3);
        }

        .prompt-editor-textarea textarea:focus {
          outline: 2px solid rgba(96, 165, 250, 0.6);
          outline-offset: -2px;
        }
      `}</style>
      <div
        ref={editorContainerRef}
        className="fixed inset-0 z-50 bg-gray-900 flex flex-col"
        tabIndex={-1}
      >
        {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-gray-700 bg-gray-800">
        <h2 className="text-xl font-semibold text-gray-100">Prompt Editor</h2>
        <div className="flex gap-2">
          <Button onClick={handleSave} variant="primary">
            Save
          </Button>
          <Button onClick={onClose} variant="secondary">
            Cancel
          </Button>
        </div>
      </div>

      {/* Mobile: Panel Toggle Button */}
      <button
        onClick={() => setIsPanelSidebarOpen(!isPanelSidebarOpen)}
        className="fixed top-20 left-4 z-50 p-3 rounded-lg bg-gray-800 text-white shadow-lg lg:hidden"
        aria-label="Toggle panel menu"
      >
        {isPanelSidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
      </button>

      {/* Mobile: Panel Sidebar Overlay */}
      {isPanelSidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={() => setIsPanelSidebarOpen(false)}
        />
      )}

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar - Panel Selection */}
        <div className={`
          fixed lg:relative top-0 left-0 h-full w-48 z-50 lg:z-auto
          transform transition-transform duration-200 ease-in-out
          ${isPanelSidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
          bg-gray-800 border-r border-gray-700 p-4 pt-20 lg:pt-4 space-y-2
        `}>
          <button
            onClick={() => {
              setActivePanel("main");
              setIsPanelSidebarOpen(false);
            }}
            className={`w-full text-left px-3 py-2 rounded ${
              activePanel === "main"
                ? "bg-blue-600 text-white"
                : "text-gray-300 hover:bg-gray-700"
            }`}
          >
            Main Editor
          </button>
          <button
            onClick={() => {
              setActivePanel("template");
              setIsPanelSidebarOpen(false);
            }}
            className={`w-full text-left px-3 py-2 rounded ${
              activePanel === "template"
                ? "bg-blue-600 text-white"
                : "text-gray-300 hover:bg-gray-700"
            }`}
          >
            Templates
          </button>
          <button
            onClick={() => {
              setActivePanel("category");
              setIsPanelSidebarOpen(false);
            }}
            className={`w-full text-left px-3 py-2 rounded ${
              activePanel === "category"
                ? "bg-blue-600 text-white"
                : "text-gray-300 hover:bg-gray-700"
            }`}
          >
            Category Order
          </button>
          <button
            onClick={() => {
              setActivePanel("wildcard");
              setIsPanelSidebarOpen(false);
            }}
            className={`w-full text-left px-3 py-2 rounded ${
              activePanel === "wildcard"
                ? "bg-blue-600 text-white"
                : "text-gray-300 hover:bg-gray-700"
            }`}
          >
            Wildcards
          </button>
          <button
            onClick={() => {
              setActivePanel("tipo");
              setIsPanelSidebarOpen(false);
            }}
            className={`w-full text-left px-3 py-2 rounded ${
              activePanel === "tipo"
                ? "bg-blue-600 text-white"
                : "text-gray-300 hover:bg-gray-700"
            }`}
          >
            TIPO
          </button>
          <button
            onClick={() => {
              setActivePanel("tagger");
              setIsPanelSidebarOpen(false);
            }}
            className={`w-full text-left px-3 py-2 rounded ${
              activePanel === "tagger"
                ? "bg-blue-600 text-white"
                : "text-gray-300 hover:bg-gray-700"
            }`}
          >
            Image Tagger
          </button>
        </div>

        {/* Center - Main Editor */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Top: Prompt Editor (Always visible) */}
          <div className="border-b border-gray-700 p-6 bg-gray-850">
            {/* Prompt Type Selector */}
            <div className="flex gap-2 mb-4">
              <button
                onClick={() => setActivePromptType("positive")}
                className={`px-4 py-2 rounded ${
                  activePromptType === "positive"
                    ? "bg-green-600 text-white"
                    : "bg-gray-700 text-gray-300"
                }`}
              >
                Positive Prompt
              </button>
              <button
                onClick={() => setActivePromptType("negative")}
                className={`px-4 py-2 rounded ${
                  activePromptType === "negative"
                    ? "bg-red-600 text-white"
                    : "bg-gray-700 text-gray-300"
                }`}
              >
                Negative Prompt
              </button>
            </div>

            {/* Prompt Editor */}
            {activePromptType === "positive" ? (
              <TextareaWithTagSuggestions
                ref={promptTextareaRef}
                label="Positive Prompt"
                value={prompt}
                onChange={handlePromptChange}
                onSelect={updateCursorPosition}
                onClick={updateCursorPosition}
                onKeyUp={updateCursorPosition}
                enableWeightControl={true}
                rows={8}
                className="font-mono prompt-editor-textarea"
              />
            ) : (
              <TextareaWithTagSuggestions
                ref={negativePromptTextareaRef}
                label="Negative Prompt"
                value={negativePrompt}
                onChange={handleNegativePromptChange}
                onSelect={updateCursorPosition}
                onClick={updateCursorPosition}
                onKeyUp={updateCursorPosition}
                enableWeightControl={true}
                rows={8}
                className="font-mono prompt-editor-textarea"
              />
            )}
          </div>

          {/* Bottom: Panel Content (Scrollable) */}
          <div className="flex-1 overflow-auto p-6">
            {activePanel === "main" && (
              <div className="text-gray-300">
                <h3 className="text-lg font-semibold mb-4">Main Editor</h3>
                <p className="text-gray-400">
                  Edit your prompts above. Use the panels on the left to access additional tools:
                </p>
                <ul className="list-disc list-inside mt-2 space-y-1 text-gray-400">
                  <li>Templates: Save and reuse prompt templates</li>
                  <li>Category Order: Configure tag category order for TIPO and suggestions</li>
                  <li>Wildcards: Use random tag replacement</li>
                  <li>TIPO: AI-powered tag interpolation</li>
                  <li>Image Tagger: Extract tags from images</li>
                </ul>
              </div>
            )}

            {activePanel === "template" && (
              <TemplatePanel
                currentPrompt={activePromptType === "positive" ? prompt : negativePrompt}
                onInsert={handleInsertTemplate}
              />
            )}

            {activePanel === "category" && (
              <CategoryOrderPanel
                currentPrompt={activePromptType === "positive" ? prompt : negativePrompt}
                onApplyOrder={(reordered) => {
                  if (activePromptType === "positive") {
                    setPrompt(reordered);
                  } else {
                    setNegativePrompt(reordered);
                  }
                }}
              />
            )}

            {activePanel === "wildcard" && (
              <WildcardPanel onInsert={handleInsertTemplate} />
            )}

            {activePanel === "tipo" && (
              <TIPOPanel
                ref={tipoPanelRef}
                onInsert={handleInsertTemplate}
                onOverwrite={(content) => {
                  // Overwrite the active prompt (positive or negative)
                  if (activePromptType === "positive") {
                    setPrompt(content);
                  } else {
                    setNegativePrompt(content);
                  }
                }}
                currentPrompt={activePromptType === "positive" ? prompt : negativePrompt}
                tipoSettings={tipoSettings}
              />
            )}

            {activePanel === "tagger" && (
              <ImageTaggerPanel
                onInsert={handleInsertTemplate}
                onOverwrite={handleOverwriteTemplate}
                currentPrompt={activePromptType === "positive" ? prompt : negativePrompt}
              />
            )}
          </div>
        </div>

        {/* Right Sidebar - Quick Info/Help (Desktop only) */}
        <div className="hidden lg:block w-64 bg-gray-800 border-l border-gray-700 p-4 overflow-auto">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Keyboard Shortcuts</h3>
          <div className="space-y-2 text-xs text-gray-400">
            <div><kbd className="px-1 bg-gray-700 rounded">Ctrl+Z</kbd> Undo</div>
            <div><kbd className="px-1 bg-gray-700 rounded">Ctrl+Shift+Z</kbd> Redo</div>
            <div><kbd className="px-1 bg-gray-700 rounded">Ctrl+←/→</kbd> Jump tags</div>
            <div><kbd className="px-1 bg-gray-700 rounded">Ctrl+Shift+←/→</kbd> Swap tags</div>
            <div><kbd className="px-1 bg-gray-700 rounded">Ctrl+Backspace</kbd> Delete tag</div>
            <div><kbd className="px-1 bg-gray-700 rounded">Ctrl+↑/↓</kbd> Adjust weight</div>
            <div><kbd className="px-1 bg-gray-700 rounded">Tab/Enter</kbd> Accept suggestion</div>
          </div>
        </div>
      </div>
    </div>
    </>
  );
}
