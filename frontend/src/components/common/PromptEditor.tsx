"use client";

import { useState, useRef, ChangeEvent } from "react";
import Button from "./Button";
import TextareaWithTagSuggestions from "./TextareaWithTagSuggestions";
import TemplatePanel from "./TemplatePanel";

interface PromptEditorProps {
  initialPrompt: string;
  initialNegativePrompt: string;
  onSave: (prompt: string, negativePrompt: string) => void;
  onClose: () => void;
}

type PanelType = "main" | "template" | "wildcard" | "tipo" | "tagger";

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

  const promptTextareaRef = useRef<HTMLTextAreaElement>(null);
  const negativePromptTextareaRef = useRef<HTMLTextAreaElement>(null);

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

  const handleInsertTemplate = (content: string) => {
    const textarea = activePromptType === "positive"
      ? promptTextareaRef.current
      : negativePromptTextareaRef.current;

    if (!textarea) return;

    const currentValue = activePromptType === "positive" ? prompt : negativePrompt;
    const cursorPos = textarea.selectionStart || 0;

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
        textarea.focus();
      }
    }, 0);
  };

  return (
    <div className="fixed inset-0 z-50 bg-gray-900 flex flex-col">
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

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar - Panel Selection */}
        <div className="w-48 bg-gray-800 border-r border-gray-700 p-4 space-y-2">
          <button
            onClick={() => setActivePanel("main")}
            className={`w-full text-left px-3 py-2 rounded ${
              activePanel === "main"
                ? "bg-blue-600 text-white"
                : "text-gray-300 hover:bg-gray-700"
            }`}
          >
            Main Editor
          </button>
          <button
            onClick={() => setActivePanel("template")}
            className={`w-full text-left px-3 py-2 rounded ${
              activePanel === "template"
                ? "bg-blue-600 text-white"
                : "text-gray-300 hover:bg-gray-700"
            }`}
          >
            Templates
          </button>
          <button
            onClick={() => setActivePanel("wildcard")}
            className={`w-full text-left px-3 py-2 rounded ${
              activePanel === "wildcard"
                ? "bg-blue-600 text-white"
                : "text-gray-300 hover:bg-gray-700"
            }`}
          >
            Wildcards
          </button>
          <button
            onClick={() => setActivePanel("tipo")}
            className={`w-full text-left px-3 py-2 rounded ${
              activePanel === "tipo"
                ? "bg-blue-600 text-white"
                : "text-gray-300 hover:bg-gray-700"
            }`}
          >
            TIPO
          </button>
          <button
            onClick={() => setActivePanel("tagger")}
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
        <div className="flex-1 flex flex-col p-6 overflow-auto">
          {activePanel === "main" && (
            <div className="space-y-4">
              {/* Prompt Type Selector */}
              <div className="flex gap-2">
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
                  enableWeightControl={true}
                  rows={20}
                  className="font-mono"
                />
              ) : (
                <TextareaWithTagSuggestions
                  ref={negativePromptTextareaRef}
                  label="Negative Prompt"
                  value={negativePrompt}
                  onChange={handleNegativePromptChange}
                  enableWeightControl={true}
                  rows={20}
                  className="font-mono"
                />
              )}
            </div>
          )}

          {activePanel === "template" && (
            <TemplatePanel
              currentPrompt={activePromptType === "positive" ? prompt : negativePrompt}
              onInsert={handleInsertTemplate}
            />
          )}

          {activePanel === "wildcard" && (
            <div className="text-gray-300">
              <h3 className="text-lg font-semibold mb-4">Wildcard Editor</h3>
              <p className="text-gray-400">Wildcard functionality coming soon...</p>
            </div>
          )}

          {activePanel === "tipo" && (
            <div className="text-gray-300">
              <h3 className="text-lg font-semibold mb-4">TIPO (Tag Interpolation)</h3>
              <p className="text-gray-400">TIPO functionality coming soon...</p>
            </div>
          )}

          {activePanel === "tagger" && (
            <div className="text-gray-300">
              <h3 className="text-lg font-semibold mb-4">Image Tagger</h3>
              <p className="text-gray-400">Image tagger functionality coming soon...</p>
            </div>
          )}
        </div>

        {/* Right Sidebar - Quick Info/Help */}
        <div className="w-64 bg-gray-800 border-l border-gray-700 p-4 overflow-auto">
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
  );
}
