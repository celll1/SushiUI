"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import Button from "./Button";

interface ImageEditorProps {
  imageUrl: string;
  onSave: (editedImageUrl: string) => void;
  onClose: () => void;
}

type Tool = "pen" | "eraser" | "blur";

interface HistoryState {
  imageData: ImageData;
}

export default function ImageEditor({ imageUrl, onSave, onClose }: ImageEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tool, setTool] = useState<Tool>("pen");
  const [brushSize, setBrushSize] = useState(5);
  const [rgb, setRgb] = useState({ r: 0, g: 0, b: 0 });
  const [hue, setHue] = useState(0);
  const [saturation, setSaturation] = useState(100);
  const [lightness, setLightness] = useState(0);
  const [isDrawing, setIsDrawing] = useState(false);
  const [history, setHistory] = useState<HistoryState[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null);

  // Calculate color from RGB
  const getColorFromRGB = () => {
    return `rgb(${rgb.r}, ${rgb.g}, ${rgb.b})`;
  };

  // Calculate color from HSL and lightness
  const getColorFromHSL = () => {
    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
  };

  // Update RGB when HSL changes
  useEffect(() => {
    const color = getColorFromHSL();
    // Convert HSL to RGB for display
    const temp = document.createElement('div');
    temp.style.color = color;
    document.body.appendChild(temp);
    const computed = window.getComputedStyle(temp).color;
    document.body.removeChild(temp);

    const match = computed.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
    if (match) {
      setRgb({ r: parseInt(match[1]), g: parseInt(match[2]), b: parseInt(match[3]) });
    }
  }, [hue, saturation, lightness]);

  // Load image and initialize canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      // Save initial state
      saveToHistory(ctx);
    };
    img.src = imageUrl;
  }, [imageUrl]);

  const saveToHistory = useCallback((ctx: CanvasRenderingContext2D) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Remove any history after current index
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push({ imageData });

    // Limit history to 50 states
    if (newHistory.length > 50) {
      newHistory.shift();
    } else {
      setHistoryIndex(historyIndex + 1);
    }

    setHistory(newHistory);
  }, [history, historyIndex]);

  const undo = useCallback(() => {
    if (historyIndex <= 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const newIndex = historyIndex - 1;
    setHistoryIndex(newIndex);
    ctx.putImageData(history[newIndex].imageData, 0, 0);
  }, [history, historyIndex]);

  const redo = useCallback(() => {
    if (historyIndex >= history.length - 1) return;

    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const newIndex = historyIndex + 1;
    setHistoryIndex(newIndex);
    ctx.putImageData(history[newIndex].imageData, 0, 0);
  }, [history, historyIndex]);

  const getCanvasPoint = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const applyBlur = (ctx: CanvasRenderingContext2D, x: number, y: number, radius: number) => {
    const imageData = ctx.getImageData(
      Math.max(0, x - radius),
      Math.max(0, y - radius),
      radius * 2,
      radius * 2
    );

    // Simple box blur
    const pixels = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    const blurred = new Uint8ClampedArray(pixels);

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = (y * width + x) * 4;

        // Average surrounding pixels
        let r = 0, g = 0, b = 0, count = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nidx = ((y + dy) * width + (x + dx)) * 4;
            r += pixels[nidx];
            g += pixels[nidx + 1];
            b += pixels[nidx + 2];
            count++;
          }
        }

        blurred[idx] = r / count;
        blurred[idx + 1] = g / count;
        blurred[idx + 2] = b / count;
      }
    }

    const blurredImageData = new ImageData(blurred, width, height);
    ctx.putImageData(blurredImageData, Math.max(0, x - radius), Math.max(0, y - radius));
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const point = getCanvasPoint(e);
    setIsDrawing(true);

    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = brushSize;

    if (tool === "pen") {
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = getColorFromRGB();
      ctx.beginPath();
      ctx.moveTo(point.x, point.y);
    } else if (tool === "eraser") {
      ctx.globalCompositeOperation = "destination-out";
      ctx.strokeStyle = "rgba(0,0,0,1)"; // Eraser needs a stroke style
      ctx.beginPath();
      ctx.moveTo(point.x, point.y);
    } else if (tool === "blur") {
      applyBlur(ctx, point.x, point.y, brushSize);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const point = getCanvasPoint(e);

    // Store screen coordinates for cursor preview
    const rect = canvas.getBoundingClientRect();
    setCursorPos({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    });

    if (isDrawing) {
      if (tool === "pen" || tool === "eraser") {
        ctx.lineTo(point.x, point.y);
        ctx.stroke();
      } else if (tool === "blur") {
        applyBlur(ctx, point.x, point.y, brushSize);
      }
    }
  };

  const handleMouseUp = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    if (isDrawing) {
      setIsDrawing(false);
      saveToHistory(ctx);
    }
  };

  const handleMouseLeave = () => {
    setCursorPos(null);
    handleMouseUp();
  };

  const handleSave = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      onSave(url);
    }, "image/png");
  };

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.ctrlKey || e.metaKey) {
      if (e.key === "z" && !e.shiftKey) {
        e.preventDefault();
        undo();
      } else if (e.key === "z" && e.shiftKey || e.key === "y") {
        e.preventDefault();
        redo();
      }
    }
  }, [undo, redo]);

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
      <div className="bg-gray-900 rounded-lg p-6 max-w-7xl max-h-[90vh] overflow-auto">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold text-white">Image Editor</h2>
          <Button onClick={onClose} variant="secondary" size="sm">
            ×
          </Button>
        </div>

        {/* Toolbar */}
        <div className="mb-4 space-y-4">
          {/* Tools */}
          <div className="flex gap-2">
            <Button
              onClick={() => setTool("pen")}
              variant={tool === "pen" ? "primary" : "secondary"}
              size="sm"
            >
              ✏️ Pen
            </Button>
            <Button
              onClick={() => setTool("eraser")}
              variant={tool === "eraser" ? "primary" : "secondary"}
              size="sm"
            >
              🧹 Eraser
            </Button>
            <Button
              onClick={() => setTool("blur")}
              variant={tool === "blur" ? "primary" : "secondary"}
              size="sm"
            >
              🌫️ Blur
            </Button>
          </div>

          {/* Brush Size */}
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-300">Size:</label>
            <input
              type="range"
              min="1"
              max="50"
              value={brushSize}
              onChange={(e) => setBrushSize(parseInt(e.target.value))}
              className="flex-1"
            />
            <span className="text-sm text-gray-300 w-8">{brushSize}</span>
          </div>

          {/* Color Picker */}
          <div className="space-y-2">
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 rounded border-2 border-gray-600" style={{ backgroundColor: getColorFromRGB() }} />
              <div className="flex-1 space-y-1">
                <div className="flex items-center gap-2">
                  <label className="text-xs text-gray-400 w-6">R:</label>
                  <input
                    type="range"
                    min="0"
                    max="255"
                    value={rgb.r}
                    onChange={(e) => setRgb({ ...rgb, r: parseInt(e.target.value) })}
                    className="flex-1"
                  />
                  <span className="text-xs text-gray-300 w-8">{rgb.r}</span>
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-gray-400 w-6">G:</label>
                  <input
                    type="range"
                    min="0"
                    max="255"
                    value={rgb.g}
                    onChange={(e) => setRgb({ ...rgb, g: parseInt(e.target.value) })}
                    className="flex-1"
                  />
                  <span className="text-xs text-gray-300 w-8">{rgb.g}</span>
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-gray-400 w-6">B:</label>
                  <input
                    type="range"
                    min="0"
                    max="255"
                    value={rgb.b}
                    onChange={(e) => setRgb({ ...rgb, b: parseInt(e.target.value) })}
                    className="flex-1"
                  />
                  <span className="text-xs text-gray-300 w-8">{rgb.b}</span>
                </div>
              </div>
            </div>

            {/* Gradient Palette */}
            <div>
              <label className="text-xs text-gray-400 mb-1 block">Hue:</label>
              <input
                type="range"
                min="0"
                max="360"
                value={hue}
                onChange={(e) => setHue(parseInt(e.target.value))}
                className="w-full"
                style={{
                  background: `linear-gradient(to right,
                    hsl(0, 100%, 50%),
                    hsl(60, 100%, 50%),
                    hsl(120, 100%, 50%),
                    hsl(180, 100%, 50%),
                    hsl(240, 100%, 50%),
                    hsl(300, 100%, 50%),
                    hsl(360, 100%, 50%))`
                }}
              />
              <label className="text-xs text-gray-400 mb-1 block mt-2">Saturation:</label>
              <input
                type="range"
                min="0"
                max="100"
                value={saturation}
                onChange={(e) => setSaturation(parseInt(e.target.value))}
                className="w-full"
              />
              <label className="text-xs text-gray-400 mb-1 block mt-2">Lightness:</label>
              <input
                type="range"
                min="0"
                max="100"
                value={lightness}
                onChange={(e) => setLightness(parseInt(e.target.value))}
                className="w-full"
              />
            </div>
          </div>

          {/* Undo/Redo */}
          <div className="flex gap-2">
            <Button
              onClick={undo}
              disabled={historyIndex <= 0}
              variant="secondary"
              size="sm"
            >
              ↶ Undo (Ctrl+Z)
            </Button>
            <Button
              onClick={redo}
              disabled={historyIndex >= history.length - 1}
              variant="secondary"
              size="sm"
            >
              ↷ Redo (Ctrl+Shift+Z)
            </Button>
          </div>
        </div>

        {/* Canvas */}
        <div className="bg-gray-800 p-4 rounded-lg mb-4 overflow-auto max-h-[60vh] relative">
          <div className="relative inline-block">
            <canvas
              ref={canvasRef}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseLeave}
              className="max-w-full cursor-none"
              style={{ imageRendering: "pixelated" }}
            />
            {/* Brush Preview Cursor */}
            {cursorPos && canvasRef.current && (() => {
              const canvas = canvasRef.current;
              const rect = canvas.getBoundingClientRect();
              const scale = rect.width / canvas.width;
              const scaledSize = brushSize * scale;

              return (
                <div
                  className="absolute pointer-events-none rounded-full border-2"
                  style={{
                    left: `${cursorPos.x}px`,
                    top: `${cursorPos.y}px`,
                    width: `${scaledSize}px`,
                    height: `${scaledSize}px`,
                    transform: 'translate(-50%, -50%)',
                    borderColor: tool === "eraser" ? "#ffffff" : getColorFromRGB(),
                    backgroundColor: tool === "eraser" ? "rgba(255,255,255,0.2)" : `${getColorFromRGB()}33`,
                  }}
                />
              );
            })()}
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-2 justify-end">
          <Button onClick={onClose} variant="secondary">
            Cancel
          </Button>
          <Button onClick={handleSave} variant="primary">
            Save & Use
          </Button>
        </div>
      </div>
    </div>
  );
}
