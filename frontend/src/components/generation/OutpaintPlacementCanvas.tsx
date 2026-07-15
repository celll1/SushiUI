"use client";

import { useRef, useState } from "react";
import NumberInput from "../common/NumberInput";
import Slider from "../common/Slider";
import Select from "../common/Select";
import Button from "../common/Button";

// ---------------------------------------------------------------------------
// Small local helpers (intentionally NOT shared -- this is the only place
// that needs bbox rounding/clamping for the outpaint placement UI).
// ---------------------------------------------------------------------------
const roundToN = (value: number, n: number): number => (n > 0 ? Math.round(value / n) * n : Math.round(value));
const clamp = (value: number, min: number, max: number): number => Math.min(Math.max(value, min), max);

const MIN_PLACE_SIZE = 8;
const PREVIEW_MAX_DIM = 480;

export interface OutpaintPlacementParams {
  canvas_width: number;
  canvas_height: number;
  place_x: number;
  place_y: number;
  place_width: number;
  place_height: number;
  input_crop_x: number;
  input_crop_y: number;
  input_crop_w: number;
  input_crop_h: number;
  outpaint_fill_mode: string;
  mask_blur: number;
}

interface OutpaintPlacementCanvasProps {
  inputImagePreview: string | null;
  inputImageSize: { width: number; height: number } | null;
  params: OutpaintPlacementParams;
  onChange: (patch: Partial<OutpaintPlacementParams>) => void;
}

type DragMode = "move" | "resize-nw" | "resize-ne" | "resize-sw" | "resize-se" | null;

interface DragStart {
  mouseX: number;
  mouseY: number;
  placeX: number;
  placeY: number;
  placeWidth: number;
  placeHeight: number;
}

const fillModeOptions = [
  { value: "replicate", label: "Replicate (edge-extend)" },
  { value: "reflect", label: "Reflect (mirror)" },
  { value: "mean", label: "Mean color (solid)" },
  { value: "noise", label: "Noise (uniform RGB)" },
];

export default function OutpaintPlacementCanvas({
  inputImagePreview,
  inputImageSize,
  params,
  onChange,
}: OutpaintPlacementCanvasProps) {
  const {
    canvas_width: canvasWidth,
    canvas_height: canvasHeight,
    place_x: placeX,
    place_y: placeY,
    place_width: placeWidth,
    place_height: placeHeight,
    input_crop_x: inputCropX,
    input_crop_y: inputCropY,
    input_crop_w: inputCropW,
    input_crop_h: inputCropH,
    outpaint_fill_mode: outpaintFillMode,
    mask_blur: maskBlur,
  } = params;

  const [snapEnabled, setSnapEnabled] = useState(true);
  const [dragMode, setDragMode] = useState<DragMode>(null);
  const dragStartRef = useRef<DragStart | null>(null);

  // Fit the canvas into a PREVIEW_MAX_DIM x PREVIEW_MAX_DIM box, preserving
  // its aspect ratio -- this is purely a display transform; all committed
  // values stay in canvas pixel space.
  const scale = canvasWidth > 0 && canvasHeight > 0
    ? PREVIEW_MAX_DIM / Math.max(canvasWidth, canvasHeight)
    : 1;
  const previewW = Math.max(1, Math.round(canvasWidth * scale));
  const previewH = Math.max(1, Math.round(canvasHeight * scale));

  const commitDrag = (patch: Partial<OutpaintPlacementParams>) => {
    onChange(patch);
  };

  const startDrag = (mode: DragMode) => (e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    setDragMode(mode);
    dragStartRef.current = {
      mouseX: e.clientX,
      mouseY: e.clientY,
      placeX,
      placeY,
      placeWidth,
      placeHeight,
    };
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragMode || !dragStartRef.current) return;
    const start = dragStartRef.current;
    const dx = (e.clientX - start.mouseX) / scale;
    const dy = (e.clientY - start.mouseY) / scale;

    if (dragMode === "move") {
      const newX = clamp(Math.round(start.placeX + dx), 0, Math.max(0, canvasWidth - start.placeWidth));
      const newY = clamp(Math.round(start.placeY + dy), 0, Math.max(0, canvasHeight - start.placeHeight));
      onChange({ place_x: newX, place_y: newY });
      return;
    }

    let newX = start.placeX;
    let newY = start.placeY;
    let newW = start.placeWidth;
    let newH = start.placeHeight;

    if (dragMode.includes("e")) newW = start.placeWidth + dx;
    if (dragMode.includes("w")) { newW = start.placeWidth - dx; newX = start.placeX + dx; }
    if (dragMode.includes("s")) newH = start.placeHeight + dy;
    if (dragMode.includes("n")) { newH = start.placeHeight - dy; newY = start.placeY + dy; }

    // Shift = preserve the input's original aspect ratio (derived from the
    // drag-start size, not the live one, so it stays stable mid-drag).
    if (e.shiftKey && start.placeHeight > 0) {
      const aspect = start.placeWidth / start.placeHeight;
      newH = newW / aspect;
      if (dragMode.includes("n")) newY = start.placeY + start.placeHeight - newH;
    }

    newW = clamp(newW, MIN_PLACE_SIZE, canvasWidth);
    newH = clamp(newH, MIN_PLACE_SIZE, canvasHeight);
    newX = clamp(newX, 0, canvasWidth - newW);
    newY = clamp(newY, 0, canvasHeight - newH);

    onChange({
      place_x: Math.round(newX),
      place_y: Math.round(newY),
      place_width: Math.round(newW),
      place_height: Math.round(newH),
    });
  };

  const onPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragMode) return;
    setDragMode(null);
    dragStartRef.current = null;
    if (snapEnabled) {
      commitDrag({
        place_x: roundToN(placeX, 8),
        place_y: roundToN(placeY, 8),
        place_width: Math.max(MIN_PLACE_SIZE, roundToN(placeWidth, 8)),
        place_height: Math.max(MIN_PLACE_SIZE, roundToN(placeHeight, 8)),
      });
    }
  };

  const handleCenter = () => {
    onChange({
      place_x: Math.max(0, Math.round((canvasWidth - placeWidth) / 2)),
      place_y: Math.max(0, Math.round((canvasHeight - placeHeight) / 2)),
    });
  };

  const handleFit15x = () => {
    const baseW = inputImageSize?.width || placeWidth || 1024;
    const baseH = inputImageSize?.height || placeHeight || 1024;
    const newCanvasW = Math.max(64, roundToN(baseW * 1.5, 16));
    const newCanvasH = Math.max(64, roundToN(baseH * 1.5, 16));
    onChange({
      canvas_width: newCanvasW,
      canvas_height: newCanvasH,
      place_width: baseW,
      place_height: baseH,
      place_x: Math.max(0, Math.round((newCanvasW - baseW) / 2)),
      place_y: Math.max(0, Math.round((newCanvasH - baseH) / 2)),
    });
  };

  // "Expand <direction>" pins the input against the opposite edge and grows
  // the canvas by a 50%-of-input margin on the requested side only (a bbox
  // preset, per the plan's directional-expand convenience buttons).
  const handleExpand = (direction: "left" | "right" | "top" | "bottom") => {
    const w = placeWidth || inputImageSize?.width || 1024;
    const h = placeHeight || inputImageSize?.height || 1024;
    if (direction === "left" || direction === "right") {
      const margin = Math.max(64, roundToN(w * 0.5, 16));
      const newCanvasW = roundToN(w + margin, 16);
      onChange({
        canvas_width: newCanvasW,
        canvas_height: roundToN(h, 16),
        place_width: w,
        place_height: h,
        place_x: direction === "right" ? 0 : newCanvasW - w,
        place_y: 0,
      });
    } else {
      const margin = Math.max(64, roundToN(h * 0.5, 16));
      const newCanvasH = roundToN(h + margin, 16);
      onChange({
        canvas_width: roundToN(w, 16),
        canvas_height: newCanvasH,
        place_width: w,
        place_height: h,
        place_x: 0,
        place_y: direction === "bottom" ? 0 : newCanvasH - h,
      });
    }
  };

  const handles: Array<{ key: "nw" | "ne" | "sw" | "se"; className: string; cursor: string }> = [
    { key: "nw", className: "-top-1.5 -left-1.5", cursor: "cursor-nwse-resize" },
    { key: "ne", className: "-top-1.5 -right-1.5", cursor: "cursor-nesw-resize" },
    { key: "sw", className: "-bottom-1.5 -left-1.5", cursor: "cursor-nesw-resize" },
    { key: "se", className: "-bottom-1.5 -right-1.5", cursor: "cursor-nwse-resize" },
  ];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Canvas Width</label>
          <NumberInput
            value={canvasWidth}
            onCommit={(v) => onChange({ canvas_width: Math.max(64, v) })}
            min={64}
            max={8192}
            step={16}
            snap={16}
            parse="int"
            className="w-full"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Canvas Height</label>
          <NumberInput
            value={canvasHeight}
            onCommit={(v) => onChange({ canvas_height: Math.max(64, v) })}
            min={64}
            max={8192}
            step={16}
            snap={16}
            parse="int"
            className="w-full"
          />
        </div>
      </div>
      <p className="text-xs text-gray-500">
        Canvas dimensions are snapped to multiples of 16 (the loaded architecture's latent grid); the backend re-validates regardless.
      </p>

      <div className="flex flex-wrap gap-2">
        <Button onClick={handleCenter} variant="secondary" size="sm">Center</Button>
        <Button onClick={handleFit15x} variant="secondary" size="sm">Fit 1.5x</Button>
        <Button onClick={() => handleExpand("left")} variant="secondary" size="sm">Expand Left</Button>
        <Button onClick={() => handleExpand("right")} variant="secondary" size="sm">Expand Right</Button>
        <Button onClick={() => handleExpand("top")} variant="secondary" size="sm">Expand Top</Button>
        <Button onClick={() => handleExpand("bottom")} variant="secondary" size="sm">Expand Bottom</Button>
      </div>

      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          id="outpaint_snap8"
          checked={snapEnabled}
          onChange={(e) => setSnapEnabled(e.target.checked)}
          className="rounded"
        />
        <label htmlFor="outpaint_snap8" className="text-sm text-gray-300">
          Snap placement to 8px on release
        </label>
        <span className="text-xs text-gray-500">(cosmetic only -- the backend always guarantees exact placement)</span>
      </div>

      {/* Scaled visual preview: canvas box + draggable/resizable placed rect */}
      <div className="flex justify-center py-2">
        <div
          className="relative bg-gray-800 border border-gray-600 rounded overflow-hidden touch-none select-none"
          style={{ width: previewW, height: previewH }}
        >
          {/* Fill-mode backdrop hint */}
          <div className="absolute inset-0 bg-[repeating-linear-gradient(45deg,rgba(255,255,255,0.03)_0,rgba(255,255,255,0.03)_6px,transparent_6px,transparent_12px)]" />

          <div
            onPointerDown={startDrag("move")}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
            className={`absolute border-2 border-blue-500 bg-blue-500/10 ${dragMode === "move" ? "cursor-grabbing" : "cursor-grab"}`}
            style={{
              left: placeX * scale,
              top: placeY * scale,
              width: Math.max(1, placeWidth * scale),
              height: Math.max(1, placeHeight * scale),
            }}
            title="Drag to move; drag corner handles to resize (hold Shift to preserve aspect)"
          >
            {inputImagePreview && (
              <img
                src={inputImagePreview}
                alt="Placed input"
                className="w-full h-full object-cover pointer-events-none"
                draggable={false}
              />
            )}
            {handles.map((h) => (
              <div
                key={h.key}
                onPointerDown={startDrag(`resize-${h.key}` as DragMode)}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                onPointerCancel={onPointerUp}
                className={`absolute w-3 h-3 bg-blue-500 border border-white rounded-sm ${h.className} ${h.cursor}`}
              />
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-xs text-gray-400 mb-1">Place X</label>
          <NumberInput
            value={placeX}
            onCommit={(v) => onChange({ place_x: clamp(v, 0, Math.max(0, canvasWidth - placeWidth)) })}
            min={0}
            max={canvasWidth}
            step={1}
            parse="int"
            className="w-full"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Place Y</label>
          <NumberInput
            value={placeY}
            onCommit={(v) => onChange({ place_y: clamp(v, 0, Math.max(0, canvasHeight - placeHeight)) })}
            min={0}
            max={canvasHeight}
            step={1}
            parse="int"
            className="w-full"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Place Width</label>
          <NumberInput
            value={placeWidth}
            onCommit={(v) => onChange({ place_width: Math.max(MIN_PLACE_SIZE, v) })}
            min={MIN_PLACE_SIZE}
            max={canvasWidth}
            step={1}
            parse="int"
            className="w-full"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Place Height</label>
          <NumberInput
            value={placeHeight}
            onCommit={(v) => onChange({ place_height: Math.max(MIN_PLACE_SIZE, v) })}
            min={MIN_PLACE_SIZE}
            max={canvasHeight}
            step={1}
            parse="int"
            className="w-full"
          />
        </div>
      </div>

      <details className="bg-gray-800/40 border border-gray-700 rounded-lg p-3">
        <summary className="text-sm font-semibold text-gray-300 cursor-pointer select-none">
          Input Crop (trim before placement)
        </summary>
        <p className="text-xs text-gray-500 mt-2 mb-2">0 width/height = no trim (use the full input).</p>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Crop X</label>
            <NumberInput
              value={inputCropX}
              onCommit={(v) => onChange({ input_crop_x: Math.max(0, v) })}
              min={0}
              step={1}
              parse="int"
              className="w-full"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Crop Y</label>
            <NumberInput
              value={inputCropY}
              onCommit={(v) => onChange({ input_crop_y: Math.max(0, v) })}
              min={0}
              step={1}
              parse="int"
              className="w-full"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Crop Width (0 = full)</label>
            <NumberInput
              value={inputCropW}
              onCommit={(v) => onChange({ input_crop_w: Math.max(0, v) })}
              min={0}
              step={1}
              parse="int"
              className="w-full"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Crop Height (0 = full)</label>
            <NumberInput
              value={inputCropH}
              onCommit={(v) => onChange({ input_crop_h: Math.max(0, v) })}
              min={0}
              step={1}
              parse="int"
              className="w-full"
            />
          </div>
        </div>
      </details>

      <Select
        label="Canvas Fill Mode"
        value={outpaintFillMode || "replicate"}
        onChange={(e) => onChange({ outpaint_fill_mode: e.target.value })}
        options={fillModeOptions}
      />

      <Slider
        label="Mask Blur (outward-only)"
        min={0}
        max={64}
        step={1}
        value={maskBlur}
        onChange={(e) => onChange({ mask_blur: parseInt(e.target.value) })}
      />
    </div>
  );
}
