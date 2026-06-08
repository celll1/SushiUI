"use client";

import { useState, KeyboardEvent, useRef } from "react";

interface ChipInputProps {
  chips: string[];
  onChange: (chips: string[]) => void;
  placeholder?: string;
  chipColor?: string;
}

export default function ChipInput({
  chips,
  onChange,
  placeholder = "追加...",
  chipColor = "#374151",
}: ChipInputProps) {
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const addChip = (value: string) => {
    const v = value.trim();
    if (!v || chips.includes(v)) { setInput(""); return; }
    onChange([...chips, v]);
    setInput("");
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      addChip(input);
    } else if (e.key === "Backspace" && !input && chips.length > 0) {
      onChange(chips.slice(0, -1));
    }
  };

  return (
    <div
      className="flex flex-wrap gap-1 px-2 py-1 bg-gray-800 border border-gray-600 rounded cursor-text min-h-[30px]"
      onClick={() => inputRef.current?.focus()}
    >
      {chips.map((chip) => (
        <span
          key={chip}
          className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-xs text-white"
          style={{ backgroundColor: chipColor }}
        >
          {chip}
          <button
            onClick={(e) => { e.stopPropagation(); onChange(chips.filter((c) => c !== chip)); }}
            className="text-gray-400 hover:text-white leading-none ml-0.5"
          >
            ×
          </button>
        </span>
      ))}
      <input
        ref={inputRef}
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        onBlur={() => { if (input.trim()) addChip(input); }}
        placeholder={chips.length === 0 ? placeholder : ""}
        className="flex-1 min-w-[60px] bg-transparent text-xs text-white placeholder-gray-500 outline-none"
      />
    </div>
  );
}
